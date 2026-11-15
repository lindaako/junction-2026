"""Pack a finished stoa_graph run into a standalone demo folder (no backend, no API calls).

  python scripts/stoa_export_demo.py --root "C:\\...\\junction-dataset" --run stoa_graph_fable --out demo
Then open demo/index.html in a browser. Needs stoa_graph.py, stoa_vision_test.py and
stoa_viewer.html next to this script.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stoa_graph as sg  # noqa: E402
import stoa_vision_test as vt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset folder")
    ap.add_argument("--run", required=True, help="output folder of stoa_graph.py, e.g. stoa_graph_fable")
    ap.add_argument("--sfx", default="_v2t2", help="which version of that run (suffix of answers/report files)")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--max-side", type=int, default=1600, help="longest image side in the demo")
    ap.add_argument("--title", default="Make Blueprints Talk")
    args = ap.parse_args()

    run, out, root = Path(args.run), Path(args.out), Path(args.root)
    rows = [json.loads(l) for l in (run / f"answers{args.sfx}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    m = re.search(r"_v2t(\d)", args.sfx)
    tiles = int(m.group(1)) if m else None
    by_plan = defaultdict(list)
    for r in rows:
        by_plan[r["path"]].append(r)

    (out / "img").mkdir(parents=True, exist_ok=True)
    plans = []
    for p, qs in by_plan.items():
        key = sg.plan_key(p)
        gfile = run / "graphs" / f"{key}.json"
        src = vt.resolve_image(root, p)
        if not gfile.is_file() or src is None:
            print(f"skipped (graph or image missing): {p}")
            continue
        saved = json.loads(gfile.read_text(encoding="utf-8"))
        graph, model_sym = saved["graph"], None
        if tiles:
            sfile = run / "symbols" / f"{key}_t{tiles}.json"
            if not sfile.is_file():
                print(f"skipped (symbol pass missing): {p}")
                continue
            sym = json.loads(sfile.read_text(encoding="utf-8"))
            graph, _ = sg.merge_symbols(graph, sym["items"])   # same merge as the scored run -> same node ids
            model_sym = sym.get("model")
        img = Image.open(src).convert("RGB")
        img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        img.save(out / "img" / f"{key}.jpg", quality=85)
        plans.append({
            "key": key, "name": Path(p).stem, "path": p, "img": f"img/{key}.jpg",
            "w": img.size[0], "h": img.size[1], "graph": graph,
            "models": {"layout": saved.get("model"), "symbols": model_sym},
            "questions": [{"q": r["question"], "truth": r["truth"], "pred": r["pred"], "used": r["used"],
                           "collection": r["collection"], "correct": r["correct"]} for r in qs],
        })

    n = sum(len(p["questions"]) for p in plans)
    ok = sum(q["correct"] for p in plans for q in p["questions"])
    stats = {"plans": len(plans), "questions": n, "correct": ok, "baseline": None, "calls_per_question": None}
    rep = run / f"report{args.sfx}.md"
    if rep.is_file():
        text = rep.read_text(encoding="utf-8")
        b = re.search(r"baseline[^|]*\|\s*([\d.]+)%", text)
        c = re.search(r"=\s*([\d.]+) calls per question", text)
        stats["baseline"] = float(b.group(1)) if b else None
        stats["calls_per_question"] = float(c.group(1)) if c else None
    data = {"title": args.title, "stats": stats, "plans": sorted(plans, key=lambda p: p["name"])}
    (out / "data.js").write_text("window.DEMO = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    shutil.copyfile(HERE / "stoa_viewer.html", out / "index.html")
    print(f"{len(plans)} plans, {n} questions ({ok / max(n, 1):.1%} correct) -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
