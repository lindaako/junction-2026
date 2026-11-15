"""Stoa go/no-go test: can a vision model count objects on the raster floor plans?

Pipeline per question ("detect first, count second"):
  1. load the plan image, optionally upscale / greyscale it
  2. ask Gemini for ONE BOX PER INSTANCE of what the question asks about
     (plus its own direct answer, as a second signal)
  3. merge duplicate boxes (IoU), count = number of boxes  -> deterministic
  4. draw the boxes on the plan (visual citation + debugging)
  5. score exact match against the ground truth

Report: accuracy per question template (= per type x language, because the
questions are templated), predicted-vs-true count matrix (catch-all check),
"most common answer" baseline, agreement between box count and direct answer.

Plans are split into tune / test halves by a hash of the file path, so you
can tune the prompt on one half and report on plans you never looked at.

Usage (PowerShell, from the folder that contains this script):
  pip install google-genai pillow
  $env:GEMINI_API_KEY = "..."            # or put GEMINI_API_KEY in a .env next to the script
  python scripts/stoa_vision_test.py --root "C:\\Users\\danie\\Downloads\\junction-dataset\\junction-dataset" --dry-run
  python scripts/stoa_vision_test.py --root "..." --n 100 --split test
  Claude instead of Gemini:  pip install anthropic ; ANTHROPIC_API_KEY in the repo .env ;
  python scripts/stoa_vision_test.py --root "..." --n 10 --provider anthropic --tag claude
Results go to ./stoa_out/ (results.jsonl is resumable: re-running skips done questions).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "anthropic": "claude-sonnet-5-5"}

PROMPT = """You are reading an architectural floor plan image and must answer a counting question exactly.

Question: {question}

Work in three steps:
1. Decide exactly what must be counted. Follow the wording of the question literally,
   and use standard drawing conventions (for example: a door is drawn as a leaf line
   with a swing arc, an opening in a wall without a leaf is not a door; a window is a
   thin multi-line segment inside an exterior wall; fixtures such as WC, sinks or
   sauna stoves are drawn as small symbols; room labels may be Finnish abbreviations
   such as MH, KH, WC, PH, SH, OH, K, ET).
2. Find EVERY instance on the plan. Return one item per instance, with a tight box.
   Never return two items for the same instance.
3. Give your final answer as an integer.
{rules}
Return ONLY JSON in this form:
{{"target": "<what you counted>",
  "rule": "<how you recognised one instance>",
  "items": [{{"box_2d": [ymin, xmin, ymax, xmax], "note": "<short>"}}],
  "answer": <integer>}}
box_2d values are normalised to 0-1000 relative to the image.
If the thing asked about is not a single drawn object (for example the number of
floors or of apartments), let each item mark the region that justifies one unit."""

# ----------------------------------------------------------------------------- data

QUESTION_KEYS = ["question", "q", "query", "prompt", "text"]
ANSWER_KEYS = ["answer", "a", "ground_truth", "gt", "label", "target"]
PATH_KEYS = ["path", "file", "file_path", "image", "image_path", "filename"]
TYPE_KEYS = ["type", "question_type", "qtype", "kind", "category"]
LANG_KEYS = ["lang", "language", "locale"]
ID_KEYS = ["id", "qid", "question_id", "uid"]


def pick_key(rec: dict, candidates: list[str], override: str | None) -> str | None:
    if override:
        return override
    for k in candidates:
        if k in rec:
            return k
    return None


def load_questions(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["_line"] = i
                out.append(rec)
    return out


def as_int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else None
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.0+)?)\s*", v)
        return int(float(m.group(1))) if m else None
    return None


def plan_half(path: str) -> str:
    return "tune" if int(hashlib.md5(path.encode()).hexdigest(), 16) % 2 == 0 else "test"


def resolve_image(root: Path, rel: str) -> Path | None:
    p = Path(rel)
    for cand in (p, root / p, root / p.name):
        if cand.is_file():
            return cand
    # last resort: search by file name (slow, only once per missing file)
    hits = list(root.rglob(p.name))
    return hits[0] if hits else None


# ----------------------------------------------------------------------------- image

def prepare_image(path: Path, min_side: int, grey: bool, max_side: int = 2048) -> Image.Image:
    img = Image.open(path)
    img = img.convert("L" if grey else "RGB")
    w, h = img.size
    long_side = max(w, h)
    if min_side and long_side < min_side:
        s = min_side / long_side
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    elif max_side and long_side > max_side:
        # APIs reject huge images (Claude: 10 MB) and downscale them internally anyway
        s = max_side / long_side
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return img.convert("RGB")


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def iou(a, b) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter = ix * iy
    union = (ay1 - ay0) * (ax1 - ax0) + (by1 - by0) * (bx1 - bx0) - inter
    return inter / union if union > 0 else 0.0


def valid_box(b) -> list[float] | None:
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return None
    try:
        y0, x0, y1, x1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    y0, y1 = sorted((y0, y1))
    x0, x1 = sorted((x0, x1))
    if y1 - y0 <= 0 or x1 - x0 <= 0:
        return None
    return [y0, x0, y1, x1]


def dedupe(items: list[dict], thr: float) -> tuple[list[dict], int]:
    kept: list[dict] = []
    dropped = 0
    for it in items:
        b = valid_box(it.get("box_2d"))
        if b is None:
            dropped += 1
            continue
        if any(iou(b, k["box_2d"]) > thr for k in kept):
            dropped += 1
            continue
        kept.append({**it, "box_2d": b})
    return kept, dropped


def draw_overlay(img: Image.Image, items: list[dict], title: str, out: Path) -> None:
    im = img.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(2, round(max(w, h) / 400))
    for i, it in enumerate(items, 1):
        y0, x0, y1, x1 = it["box_2d"]
        box = (x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h)
        d.rectangle(box, outline=(220, 30, 30), width=lw)
        d.text((box[0] + lw, box[1] + lw), str(i), fill=(220, 30, 30))
    d.rectangle((0, 0, w, 14 + lw * 4), fill=(255, 255, 255))
    d.text((4, 2), title, fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


# ----------------------------------------------------------------------------- model

class RateLimited(Exception):
    pass


def make_anthropic_caller(model: str, timeout_s: int):
    try:
        import anthropic
    except ImportError:
        sys.exit("Missing package: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY (environment variable or the repo .env).")
    # the SDK itself retries overloaded / 5xx errors with backoff
    client = anthropic.Anthropic(timeout=timeout_s, max_retries=4)

    def call(img_bytes: bytes, prompt: str) -> str:
        b64 = base64.standard_b64encode(img_bytes).decode()
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
        except anthropic.RateLimitError as e:
            raise RateLimited(str(e)) from e
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return call


def make_caller(model: str, mock: bool, timeout_s: int = 90, provider: str = "gemini"):
    if provider == "anthropic" and not mock:
        return make_anthropic_caller(model, timeout_s)
    if mock:
        def call(img_bytes: bytes, prompt: str) -> str:
            rnd = random.Random(len(img_bytes) + len(prompt))
            n = rnd.choice([0, 1, 1, 2, 3, 4, 5])
            items = []
            for _ in range(n):
                y, x = rnd.randint(0, 900), rnd.randint(0, 900)
                items.append({"box_2d": [y, x, y + 60, x + 60], "note": "mock"})
            return json.dumps({"target": "mock", "rule": "mock", "items": items,
                               "answer": n + rnd.choice([0, 0, 0, 1])})
        return call

    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        sys.exit("Missing package: pip install google-genai")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Set GEMINI_API_KEY (environment variable or .env next to this script).")
    # timeout in milliseconds: a stuck request becomes an error instead of hanging forever
    client = genai.Client(http_options=types.HttpOptions(timeout=timeout_s * 1000))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)

    def call(img_bytes: bytes, prompt: str) -> str:
        # 503/500 = Google overloaded (temporary): retry with growing pauses.
        # 429 = our own quota: do NOT retry, stop and let the user resume later.
        waits = [5, 15, 30, 60]
        for attempt in range(len(waits) + 1):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/png"), prompt],
                    config=cfg,
                )
                return resp.text or ""
            except errors.APIError as e:
                code = getattr(e, "code", None)
                if code == 429:
                    raise RateLimited(str(e)) from e
                if code in (500, 502, 503, 504) and attempt < len(waits):
                    print(f"\n        {code} from Gemini (overloaded), retrying in {waits[attempt]} s ...",
                          flush=True)
                    time.sleep(waits[attempt])
                    continue
                raise
    return call


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


# ----------------------------------------------------------------------------- report

def write_report(results: list[dict], baseline: dict[str, int], out: Path) -> str:
    L = []
    n = len(results)
    if not n:
        return "no results"
    ok_box = sum(r["correct_box"] for r in results)
    ok_dir = sum(r["correct_direct"] for r in results)
    ok_base = sum(baseline.get(r["template"]) == r["truth"] for r in results)
    answered = [r for r in results if r["pred_box"] is not None and r["pred_direct"] is not None]
    agree = [r for r in answered if r["pred_box"] == r["pred_direct"]]
    dis = [r for r in answered if r["pred_box"] != r["pred_direct"]]
    errs = sum(1 for r in results if r.get("error"))

    L.append(f"# Stoa vision test: {n} questions, model {results[0]['model']}\n")
    L.append("| method | exact-match accuracy |\n|---|---|")
    L.append(f"| count of boxes (detect, then count) | **{ok_box/n:.1%}** ({ok_box}/{n}) |")
    L.append(f"| model's direct answer | {ok_dir/n:.1%} ({ok_dir}/{n}) |")
    L.append(f"| baseline: most common answer for that question | {ok_base/n:.1%} ({ok_base}/{n}) |")
    if agree:
        a_ok = sum(r["correct_box"] for r in agree)
        L.append(f"| box count == direct answer ({len(agree)} q) | {a_ok/len(agree):.1%} |")
    if dis:
        d_ok = sum(r["correct_box"] for r in dis)
        L.append(f"| box count != direct answer ({len(dis)} q) | {d_ok/len(dis):.1%} (box count) |")
    L.append(f"\nErrors (API / parse): {errs}. Duplicate or invalid boxes dropped: "
             f"{sum(r['dropped'] for r in results)}.\n")

    L.append("## Per question template (type x language)\n")
    L.append("| question | n | boxes | direct | baseline | mean error (pred-true) |\n|---|---|---|---|---|---|")
    by_t = defaultdict(list)
    for r in results:
        by_t[r["template"]].append(r)
    for t, rs in sorted(by_t.items(), key=lambda kv: -len(kv[1])):
        k = len(rs)
        bias = sum((r["pred_box"] if r["pred_box"] is not None else 0) - r["truth"] for r in rs) / k
        L.append(f"| {t[:80]} | {k} | {sum(r['correct_box'] for r in rs)/k:.0%} | "
                 f"{sum(r['correct_direct'] for r in rs)/k:.0%} | "
                 f"{sum(baseline.get(t) == r['truth'] for r in rs)/k:.0%} | {bias:+.2f} |")

    L.append("\n## Catch-all check: predicted vs true count (box count)\n")
    L.append("Rows = true count, columns = predicted. A column that collects many rows is a catch-all.\n")
    vals = sorted({r["truth"] for r in results} | {r["pred_box"] for r in results if r["pred_box"] is not None})
    vals = [v for v in vals if v <= 15] + (["16+"] if any(v > 15 for v in vals) else [])
    def bucket(v):
        return "16+" if isinstance(v, int) and v > 15 else v
    mat = Counter((bucket(r["truth"]), bucket(r["pred_box"])) for r in results if r["pred_box"] is not None)
    L.append("| true \\ pred | " + " | ".join(map(str, vals)) + " |")
    L.append("|---|" + "---|" * len(vals))
    for tv in vals:
        L.append(f"| **{tv}** | " + " | ".join(str(mat.get((tv, pv), "")) for pv in vals) + " |")
    pred_c = Counter(bucket(r["pred_box"]) for r in results)
    true_c = Counter(bucket(r["truth"]) for r in results)
    L.append("\n| count | times predicted | times true |\n|---|---|---|")
    for v in vals:
        L.append(f"| {v} | {pred_c.get(v, 0)} | {true_c.get(v, 0)} |")

    L.append("\n## Wrong answers (first 25)\n")
    L.append("| question | file | true | boxes | direct | overlay |\n|---|---|---|---|---|---|")
    for r in [r for r in results if not r["correct_box"]][:25]:
        L.append(f"| {r['question'][:60]} | {Path(r['path']).name} | {r['truth']} | {r['pred_box']} | "
                 f"{r['pred_direct']} | {r.get('overlay', '')} |")
    text = "\n".join(L) + "\n"
    out.write_text(text, encoding="utf-8")
    return text


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset folder (the one containing questions.jsonl)")
    ap.add_argument("--questions", help="path to questions.jsonl (default: <root>/questions.jsonl)")
    ap.add_argument("--source", default="cubicasa5k", help="only questions whose path contains this")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", choices=["tune", "test", "all"], default="tune",
                    help="tune the prompt on 'tune', report the final number on 'test'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--model", help="model name (default depends on --provider)")
    ap.add_argument("--min-side", type=int, default=1536, help="upscale images whose long side is smaller (0 = off)")
    ap.add_argument("--max-side", type=int, default=2048, help="downscale images whose long side is larger (0 = off)")
    ap.add_argument("--grey", action="store_true", help="convert plans to greyscale before sending")
    ap.add_argument("--iou", type=float, default=0.5, help="boxes overlapping more than this are one instance")
    ap.add_argument("--rules-file", help="text file with counting definitions to add to the prompt")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between API calls")
    ap.add_argument("--timeout", type=int, default=90, help="seconds before one API call is given up")
    ap.add_argument("--overlays", type=int, default=40, help="max overlay PNGs to save")
    ap.add_argument("--out", default="stoa_out")
    ap.add_argument("--tag", default="", help="name for this run (separate results file)")
    ap.add_argument("--dry-run", action="store_true", help="show fields and the sample, call nothing")
    ap.add_argument("--mock", action="store_true", help="fake model (tests the pipeline offline)")
    for k in ("question", "answer", "path", "type", "lang", "id"):
        ap.add_argument(f"--{k}-key", help=f"field name for {k} (auto-detected if omitted)")
    args = ap.parse_args()
    args.model = args.model or DEFAULT_MODELS[args.provider]

    # read GEMINI_API_KEY etc. from the nearest .env (script folder, then parents, then cwd)
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents][:4] + [Path.cwd()]:
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break

    root = Path(args.root)
    qpath = Path(args.questions) if args.questions else root / "questions.jsonl"
    allq = load_questions(qpath)
    first = allq[0]
    kq = pick_key(first, QUESTION_KEYS, args.question_key)
    ka = pick_key(first, ANSWER_KEYS, args.answer_key)
    kp = pick_key(first, PATH_KEYS, args.path_key)
    kt = pick_key(first, TYPE_KEYS, args.type_key)
    kl = pick_key(first, LANG_KEYS, args.lang_key)
    kid = pick_key(first, ID_KEYS, args.id_key)
    print(f"{len(allq):,} questions. Fields in first record: {sorted(k for k in first if k != '_line')}")
    print(f"Using question={kq!r} answer={ka!r} path={kp!r} type={kt!r} lang={kl!r} id={kid!r}")
    if not (kq and ka and kp):
        sys.exit("Could not detect question/answer/path fields: pass --question-key/--answer-key/--path-key.")

    pool = []
    for r in allq:
        p = str(r.get(kp, ""))
        if args.source.lower() not in p.lower().replace("\\", "/"):
            continue
        truth = as_int(r.get(ka))
        if truth is None:
            continue
        pool.append({
            "id": str(r.get(kid)) if kid else f"line{r['_line']}",
            "question": str(r[kq]), "truth": truth, "path": p,
            "template": str(r[kq]).strip(),   # questions are templated: the text IS the type x language
            "qtype": r.get(kt) if kt else None, "lang": r.get(kl) if kl else None,
            "half": plan_half(p),
        })
    print(f"{len(pool):,} integer-answer questions from '{args.source}'.")
    if not pool:
        sys.exit("Nothing to test: check --source and the field names.")

    sample_pool = pool if args.split == "all" else [q for q in pool if q["half"] == args.split]
    sample_ids = {q["id"] for q in sample_pool}
    # baseline = most common answer per template, learned from questions NOT in this split
    base_c: dict[str, Counter] = defaultdict(Counter)
    for q in pool:
        if args.split == "all" or q["id"] not in sample_ids:
            base_c[q["template"]][q["truth"]] += 1
    baseline = {t: c.most_common(1)[0][0] for t, c in base_c.items()}

    # stratified round-robin over templates so every type/language is represented
    rnd = random.Random(args.seed)
    by_t = defaultdict(list)
    for q in sample_pool:
        by_t[q["template"]].append(q)
    for lst in by_t.values():
        rnd.shuffle(lst)
    templates = sorted(by_t)
    rnd.shuffle(templates)
    sample = []
    while len(sample) < args.n and any(by_t.values()):
        for t in templates:
            if by_t[t] and len(sample) < args.n:
                sample.append(by_t[t].pop())
    print(f"{len(by_t)} templates; sampled {len(sample)} questions from the '{args.split}' half.")

    if args.dry_run:
        print("\nTemplates (count in pool):")
        for t, c in Counter(q["template"] for q in sample_pool).most_common():
            print(f"  {c:6d}  {t[:100]}")
        print("\nFirst 5 sampled:")
        for q in sample[:5]:
            img = resolve_image(root, q["path"])
            print(f"  {q['question'][:70]!r} -> {q['truth']}  [{q['path']}] {'OK' if img else 'IMAGE NOT FOUND'}")
        return

    rules = ""
    if args.rules_file:
        rules = "\nCounting definitions from the dataset:\n" + Path(args.rules_file).read_text(encoding="utf-8") + "\n"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    res_path = out / f"results{tag}.jsonl"
    done = {}
    if res_path.is_file():
        for line in res_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):          # errored questions are retried on the next run
                    done[r["id"]] = r
    call = make_caller(args.model, args.mock, args.timeout, args.provider)
    n_overlays = sum(1 for r in done.values() if r.get("overlay"))

    with res_path.open("a", encoding="utf-8") as f:
      try:
        for i, q in enumerate(sample, 1):
            if q["id"] in done:
                continue
            img_path = resolve_image(root, q["path"])
            rec = {**q, "model": args.model if not args.mock else "mock", "pred_box": None,
                   "pred_direct": None, "dropped": 0, "items": [], "error": None}
            if img_path is None:
                rec["error"] = "image not found"
            else:
                try:
                    img = prepare_image(img_path, args.min_side, args.grey, args.max_side)
                    print(f"[{i:3d}/{len(sample)}] calling {args.model} ({img.size[0]}x{img.size[1]}) ...",
                          end="\r", flush=True)
                    t0 = time.time()
                    raw = call(to_png_bytes(img), PROMPT.format(question=q["question"], rules=rules))
                    rec["seconds"] = round(time.time() - t0, 2)
                    data = parse_json(raw)
                    items, dropped = dedupe(data.get("items") or [], args.iou)
                    rec.update(pred_box=len(items), pred_direct=as_int(data.get("answer")),
                               dropped=dropped, items=items,
                               target=data.get("target"), rule=data.get("rule"))
                except RateLimited as e:
                    print(f"\nRate limited (429) after {i-1} questions: stopping. Re-run later to resume.\n{e}")
                    break
                except Exception as e:  # keep going; the report counts errors
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["correct_box"] = rec["pred_box"] == q["truth"]
            rec["correct_direct"] = rec["pred_direct"] == q["truth"]
            if img_path is not None and rec["items"] is not None and n_overlays < args.overlays \
                    and (not rec["correct_box"] or n_overlays < 10):
                name = f"{i:03d}_{'ok' if rec['correct_box'] else 'WRONG'}_{Path(q['path']).stem}.png"
                title = f"{q['question'][:70]} | true {q['truth']} | boxes {rec['pred_box']} | direct {rec['pred_direct']}"
                try:
                    draw_overlay(img, rec["items"], title, out / f"overlays{tag}" / name)
                    rec["overlay"] = f"overlays{tag}/{name}"
                    n_overlays += 1
                except Exception:
                    pass
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[q["id"]] = rec
            mark = "ok " if rec["correct_box"] else "ERR" if rec["error"] else "x  "
            secs = f"{rec.get('seconds', 0):5.1f}s"
            print(f"[{i:3d}/{len(sample)}] {mark} {secs} true {q['truth']:>3} boxes {str(rec['pred_box']):>4} "
                  f"direct {str(rec['pred_direct']):>4}  {q['question'][:60]}")
            if rec["error"]:
                print(f"        error: {rec['error']}")
            if not args.mock:
                time.sleep(args.sleep)
      except KeyboardInterrupt:
        print("\nStopped with Ctrl+C: writing the report for what is done. Re-run to resume.")

    results = [done[q["id"]] for q in sample if q["id"] in done]
    if not results:
        print("No successful answers yet: fix the error above and re-run.")
        return
    report = write_report(results, baseline, out / f"report{tag}.md")
    print("\n" + report.split("## Per question")[0])
    print(f"Full report: {out / f'report{tag}.md'}   overlays: {out / f'overlays{tag}'}")


if __name__ == "__main__":
    main()