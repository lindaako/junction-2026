"""Make Blueprints Talk: live app. Upload a floor plan, ask counting questions, see the evidence.

  python scripts/stoa_app.py                     # then open http://127.0.0.1:8100
  python scripts/stoa_app.py --host 0.0.0.0 --port 8100          # on a server
  python scripts/stoa_app.py --import-run stoa_graph_fable --root <dataset> --limit 8   # preload examples

Needs ANTHROPIC_API_KEY (.env) and: pip install anthropic pillow   (pymupdf only for PDF uploads).
No web framework: Python's built-in HTTP server. Data is kept in ./stoa_app_data/.

Upload = 3 vision calls (1 layout + 2 zoomed crops), done once per drawing.
Question = 0 vision calls; a new question wording costs one small text call, then it is cached.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stoa_graph as sg  # noqa: E402
import stoa_vision_test as vt  # noqa: E402

sg.USE_SOURCES = True
DATA = Path("stoa_app_data")
LOCK = threading.Lock()
CFG: dict = {}
SCOPE_NOTE = ("This prototype answers counting questions about rooms (bedrooms, kitchens, bathrooms, WC rooms, "
              "saunas, balconies), doors, windows, fixtures (toilets, sinks, showers, bathtubs) and floors.")


def load_env() -> None:
    for folder in [HERE, *HERE.parents][:4] + [Path.cwd()]:
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def conv_for(coll: str) -> dict:
    return CFG["conventions"].get(coll) or sg.default_conv(coll)


def accepted_ids(graph: dict) -> dict:
    """ids the counting conventions accept per collection (the rest are rejected candidates)"""
    return {c: sg.answer(graph, {"collection": c, "types": []}, conv_for(c))[1]
            for c in ("doors", "windows", "fixtures")}


def save_plan(plan: dict) -> None:
    (DATA / "plans").mkdir(parents=True, exist_ok=True)
    (DATA / "plans" / f"{plan['id']}.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")


def load_plan(pid: str) -> dict | None:
    f = DATA / "plans" / f"{re.sub(r'[^A-Za-z0-9_.-]', '', pid)}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def list_plans() -> list[dict]:
    out = []
    for f in sorted((DATA / "plans").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        p = json.loads(f.read_text(encoding="utf-8"))
        g = p["graph"]
        out.append({"id": p["id"], "name": p["name"], "source": p.get("source", "upload"), "img": p["img"],
                    "questions": len(p["questions"]), "rooms": len(g.get("rooms", [])), "floors": g.get("floors")})
    return out


def to_image(data: bytes, name: str) -> Image.Image:
    if name.lower().endswith(".pdf") or data[:4] == b"%PDF":
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ValueError("PDF upload needs PyMuPDF on the server: pip install pymupdf") from e
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=150)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return Image.open(io.BytesIO(data)).convert("RGB")


def ingest(data: bytes, name: str) -> dict:
    """read one drawing into a graph: 1 layout call + 2 zoomed symbol calls"""
    orig = to_image(data, name)
    pid = f"{re.sub(r'[^A-Za-z0-9_-]', '_', Path(name).stem)[:40]}_{hashlib.md5(data).hexdigest()[:6]}"
    existing = load_plan(pid)
    if existing:
        return existing
    if CFG.get("max_uploads") is not None:   # protects the API budget on a public demo server
        n = sum(1 for p in list_plans() if p["source"] == "upload")
        if n >= CFG["max_uploads"]:
            raise ValueError("This demo server has reached its upload limit. The example drawings still work, "
                             "and the app can be run locally with your own API key.")
    usage = sg.Usage()
    lay_call = sg.make_text_model("anthropic", CFG["layout_model"], CFG["timeout"], usage, CFG["mock"])
    sym_call = sg.make_text_model("anthropic", CFG["symbol_model"], CFG["timeout"], usage, CFG["mock"])
    t0 = time.time()
    W, H = orig.size
    tiles = sg.make_tiles(W, H, 2)

    def layout():
        img = sg.fit_image(orig, 1536, 2048)
        raw = vt.parse_json(lay_call(sg.EXTRACT_PROMPT.format(room_types=sg.ROOM_TYPES, rules=""),
                                     vt.to_png_bytes(img)))
        return sg.clean_graph(raw)[0]

    def symbols(k):
        crop = sg.fit_image(orig.crop(tiles[k]), 1536, 2048)
        part = f"crop {k + 1} of {len(tiles)} (overlapping)"
        data_ = vt.parse_json(sym_call(sg.SYMBOL_PROMPT.format(part=part), vt.to_png_bytes(crop)))
        return sg.tile_items_to_plan(data_, tiles[k], W, H)

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        f_lay = ex.submit(layout)
        f_sym = [ex.submit(symbols, k) for k in range(len(tiles))]
        lay = f_lay.result()
        merged = {"windows": [], "doors": [], "fixtures": []}
        for f in f_sym:
            for c, items in f.result().items():
                merged[c] += items
    graph, _ = sg.merge_symbols(lay, merged)
    view = orig.copy()
    view.thumbnail((1800, 1800), Image.LANCZOS)
    graph["size"] = list(view.size)
    (DATA / "img").mkdir(parents=True, exist_ok=True)
    view.save(DATA / "img" / f"{pid}.jpg", quality=85)
    plan = {"id": pid, "name": name, "img": f"img/{pid}.jpg", "source": "upload", "graph": graph,
            "accepted": accepted_ids(graph),
            "models": {"layout": CFG["layout_model"], "symbols": CFG["symbol_model"]},
            "cost": {"vision_calls": 1 + len(tiles), "tokens_in": usage.input_tokens,
                     "tokens_out": usage.output_tokens, "seconds": round(time.time() - t0, 1)},
            "questions": []}
    save_plan(plan)
    return plan


def map_question(question: str) -> dict:
    qfile = DATA / "query_map.json"
    with LOCK:
        qmap = json.loads(qfile.read_text(encoding="utf-8")) if qfile.is_file() else {}
    key = question.strip()
    if key in qmap:
        return {**qmap[key], "cached": True}
    call = sg.make_text_model("anthropic", CFG["layout_model"], CFG["timeout"], sg.Usage(), CFG["mock"])
    m = vt.parse_json(call(sg.MAP_PROMPT.format(room_types=sg.ROOM_TYPES, fixture_types=sg.FIXTURE_TYPES,
                                                collections=sg.COLLECTIONS, question=key), None, 300))
    entry = {"collection": m.get("collection"), "types": m.get("types") or [],
             "answerable": bool(m.get("answerable", True)) and m.get("collection") in sg.COLLECTIONS}
    with LOCK:
        qmap = json.loads(qfile.read_text(encoding="utf-8")) if qfile.is_file() else {}
        qmap[key] = entry
        qfile.write_text(json.dumps(qmap, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**entry, "cached": False}


def ask(pid: str, question: str) -> dict:
    plan = load_plan(pid)
    if plan is None:
        raise KeyError(pid)
    m = map_question(question)
    if not m["answerable"]:
        res = {"q": question, "answerable": False, "pred": None, "used": [], "collection": None,
               "note": "Not a question this prototype can answer from the graph. " + SCOPE_NOTE}
    else:
        coll = m["collection"]
        pred, used = sg.answer(plan["graph"], m, conv_for(coll))
        res = {"q": question, "answerable": True, "pred": pred, "used": used, "collection": coll,
               "types": m["types"], "text_calls": 0 if m["cached"] else 1, "vision_calls": 0}
    plan["questions"] = [q for q in plan["questions"] if q["q"] != question] + [res]
    save_plan(plan)
    return res


LINE = re.compile(r"^(?P<q>.*?)(?:\s*(?:=|\t|\|)\s*(?P<a>[-+]?\d+(?:[.,]\d+)?)\s*)?$")


def parse_checks(text: str) -> list[dict]:
    """one question per line, optionally followed by '= expected answer' (also tab or |)"""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE.match(line)
        q, a = (m.group("q").strip(), m.group("a")) if m else (line, None)
        if not q:
            continue
        exp = None
        if a is not None:
            v = float(a.replace(",", "."))
            exp = int(v) if v.is_integer() else v
        items.append({"q": q, "expected": exp})
    return items[:60]


def check(pid: str, text: str) -> dict:
    """answer a list of questions with known answers; each becomes a question on the plan with its
    expected value as the truth, so its evidence can be opened like any other answer"""
    plan = load_plan(pid)
    if plan is None:
        raise KeyError(pid)
    rows = []
    for it in parse_checks(text):
        m = map_question(it["q"])
        if m["answerable"]:
            pred, used = sg.answer(plan["graph"], m, conv_for(m["collection"]))
            res = {"q": it["q"], "answerable": True, "pred": pred, "used": used, "collection": m["collection"],
                   "types": m["types"], "truth": it["expected"], "checked": True}
        else:
            res = {"q": it["q"], "answerable": False, "pred": None, "used": [], "collection": None,
                   "truth": it["expected"], "checked": True}
        plan["questions"] = [q for q in plan["questions"] if q["q"] != it["q"]] + [res]
        rows.append(res)
    for r in rows:
        r["index"] = next(i for i, q in enumerate(plan["questions"]) if q["q"] == r["q"])
    scored = [r for r in rows if r["truth"] is not None and r["answerable"]]   # unsupported ones are reported apart
    summary = {"rows": rows, "scored": len(scored),
               "correct": sum(1 for r in scored if r["pred"] is not None and float(r["pred"]) == float(r["truth"])),
               "unsupported": sum(1 for r in rows if not r["answerable"])}
    plan["check"] = summary
    save_plan(plan)
    return summary


def import_run(run: Path, root: Path, sfx: str, limit: int) -> None:
    """preload plans from a finished stoa_graph.py run (with their dataset questions and answers)"""
    rows = [json.loads(l) for l in (run / f"answers{sfx}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_plan: dict[str, list] = {}
    for r in rows:
        by_plan.setdefault(r["path"], []).append(r)
    tiles = re.search(r"_v2t(\d)", sfx)
    n = 0
    for p, qs in by_plan.items():
        if n >= limit:
            break
        key = sg.plan_key(p)
        gfile, sfile = run / "graphs" / f"{key}.json", run / "symbols" / f"{key}_t{tiles.group(1) if tiles else 2}.json"
        src = vt.resolve_image(root, p)
        if not (gfile.is_file() and sfile.is_file() and src):
            continue
        saved, sym = json.loads(gfile.read_text(encoding="utf-8")), json.loads(sfile.read_text(encoding="utf-8"))
        graph, _ = sg.merge_symbols(saved["graph"], sym["items"])
        img = Image.open(src).convert("RGB")
        img.thumbnail((1800, 1800), Image.LANCZOS)
        graph["size"] = list(img.size)
        (DATA / "img").mkdir(parents=True, exist_ok=True)
        img.save(DATA / "img" / f"{key}.jpg", quality=85)
        save_plan({"id": key, "name": Path(p).name, "img": f"img/{key}.jpg", "source": "dataset", "graph": graph,
                   "accepted": accepted_ids(graph),
                   "models": {"layout": saved.get("model"), "symbols": sym.get("model")},
                   "cost": {"vision_calls": 1 + sym.get("calls", 2)},
                   "questions": [{"q": r["question"], "answerable": True, "pred": r["pred"], "used": r["used"],
                                  "collection": r["collection"], "truth": r["truth"]} for r in qs]})
        n += 1
    print(f"imported {n} plans from {run}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200) -> None:
        self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self.send(200, (HERE / "stoa_app.html").read_bytes(), "text/html; charset=utf-8")
        if path.startswith("/img/"):
            f = DATA / "img" / re.sub(r"[^A-Za-z0-9_.-]", "", path[5:])
            return self.send(200, f.read_bytes(), "image/jpeg") if f.is_file() else self.json({"error": "not found"}, 404)
        if path == "/api/plans":
            return self.json(list_plans())
        if path == "/api/info":
            return self.json({"layout_model": CFG["layout_model"], "symbol_model": CFG["symbol_model"], "scope": SCOPE_NOTE})
        m = re.fullmatch(r"/api/plans/([^/]+)", path)
        if m:
            plan = load_plan(m.group(1))
            return self.json(plan) if plan else self.json({"error": "not found"}, 404)
        self.json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if path == "/api/plans":
                if len(body) > 40 * 1024 * 1024:
                    return self.json({"error": "file too large (40 MB max)"}, 413)
                name = parse_qs(url.query).get("name", ["drawing.png"])[0]
                return self.json(ingest(body, name))
            m = re.fullmatch(r"/api/plans/([^/]+)/check", path)
            if m:
                text = str(json.loads(body or b"{}").get("text", ""))
                if not text.strip():
                    return self.json({"error": "no questions"}, 400)
                return self.json(check(m.group(1), text))
            m = re.fullmatch(r"/api/plans/([^/]+)/ask", path)
            if m:
                q = str(json.loads(body or b"{}").get("question", "")).strip()
                if not q:
                    return self.json({"error": "empty question"}, 400)
                return self.json(ask(m.group(1), q))
            self.json({"error": "not found"}, 404)
        except KeyError:
            self.json({"error": "plan not found"}, 404)
        except ValueError as e:  # a reason meant for the user (limit reached, unreadable file)
            self.json({"error": str(e)[:300]}, 400)
        except Exception as e:  # show the reason in the page instead of a silent failure
            self.json({"error": f"{type(e).__name__}: {str(e)[:300]}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--layout-model", default="claude-sonnet-5-5")
    ap.add_argument("--symbol-model", default="claude-fable-5-2")
    ap.add_argument("--conventions", help="conventions json from stoa_graph.py (default: newest one found, else defaults)")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--import-run", help="preload plans from this stoa_graph.py output folder")
    ap.add_argument("--root", help="dataset folder (needed with --import-run)")
    ap.add_argument("--sfx", default="_v2t2")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--mock", action="store_true", help="no API calls (pipeline test)")
    ap.add_argument("--max-uploads", type=int, help="stop accepting new drawings after this many (public demo)")
    args = ap.parse_args()
    load_env()
    conv = {}
    cpath = Path(args.conventions) if args.conventions else next(
        (p for p in [Path("stoa_graph_out/conventions_v2t2.json"), HERE / "stoa_conventions.json"] if p.is_file()), None)
    if cpath and cpath.is_file():
        conv = json.loads(cpath.read_text(encoding="utf-8"))
        print(f"counting conventions: {cpath}")
    else:
        print("counting conventions: defaults")
    CFG.update(layout_model=args.layout_model, symbol_model=args.symbol_model, timeout=args.timeout,
               mock=args.mock, conventions=conv, max_uploads=args.max_uploads)
    (DATA / "plans").mkdir(parents=True, exist_ok=True)
    if args.import_run:
        if not args.root:
            sys.exit("--import-run needs --root <dataset folder>")
        import_run(Path(args.import_run), Path(args.root), args.sfx, args.limit)
    if not args.mock and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY (repo .env).")
    print(f"Make Blueprints Talk: http://{args.host}:{args.port}   (Ctrl+C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()