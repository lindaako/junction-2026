"""Stoa step 2: extract each raster plan ONCE into a graph of nodes, answer questions from the graph.

  plan image --(1 vision call per plan)--> graph JSON (rooms, doors, windows, fixtures, floors)
  question   --(1 text call per distinct question wording, cached)--> query {collection, types}
  query + graph + counting conventions --> answer + the nodes that were counted (visual proof)

Counting conventions are NOT hard-coded. Every node records attributes (status: printed /
hand_drawn / crossed_out, cut_off, door leaves, where a door leads, window panes), and the
convention that reproduces the answer key best on the TUNE plans is chosen per collection
(e.g. "count leaves or doors?", "skip crossed-out items?", "only passage doors?").
Then check the chosen conventions on TEST plans you did not tune on.

Usage (PowerShell, repo root):
  # same plans as the 100-question vision test, so the result compares with 77%:
  python scripts/stoa_graph.py --root "C:\\...\\junction-dataset" --from-results stoa_out/results_claude.jsonl
  # or N new plans from one half:
  python scripts/stoa_graph.py --root "..." --plans 30 --split test --conventions stoa_graph_out/conventions.json
  # the reported configuration: Sonnet reads the layout, Fable reads the zoomed symbol crops
  python scripts/stoa_graph.py --root "..." --plans 40 --split test --symbols \\
      --model claude-sonnet-5-5 --symbol-model claude-fable-5-2 --conventions scripts/stoa_conventions.json
Outputs in ./stoa_graph_out/: graphs/<plan>.json, overlays/<plan>.png, query_map.json,
conventions.json, answers.jsonl, report.md. Re-running reuses cached graphs and queries.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import threading
import hashlib
import base64
import itertools
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoa_vision_test as vt  # noqa: E402  (shared helpers: image prep, json parsing, paths)

ROOM_TYPES = ["bedroom", "living", "kitchen", "bathroom", "wc", "sauna", "balcony",
              "hall", "storage", "utility", "other"]
FIXTURE_TYPES = ["wc_seat", "sink", "shower", "bathtub", "sauna_stove"]
STATUSES = ["printed", "hand_drawn", "crossed_out"]
OUTSIDE_SIDES = ["outside", "stairwell", "closet", "unknown"]   # non-room ends of a door/opening
COLLECTIONS = ["rooms", "doors", "windows", "fixtures", "floors"]

EXTRACT_PROMPT = """You are digitising an architectural floor plan image into a graph.
Find EVERY instance of each class below. Be exhaustive and never list the same instance twice.

Drawing conventions: walls are thick or double lines. A door is a leaf line, usually with a
swing arc (a quarter circle, "pizza slice"), sitting in a gap in a wall. A double door is ONE
door with two leaves meeting in the middle of ONE gap. Two doors side by side in different gaps
are TWO doors: give each its own item and box. A door-like symbol attached to a single thin line
(not a wall) is a cupboard/wardrobe front. A gap in a wall without any leaf is an opening, not a door.

Classes:
- rooms: every room or space, with id "R1", "R2", ... label_text = the label exactly as written
  (e.g. "MH", "KH", "OH", "K", "S", "PARV", "ET", "VH", "WC"), "" if unlabelled.
  room_type is one of {room_types}. Finnish codes: MH=bedroom, OH=living,
  K/KK/KT/keittokomero=kitchen, KH/KPH/PH/PESUH/SH=bathroom, WC=wc, S/SAUNA=sauna,
  PARV/parveke (also glazed)=balcony, ET/AULA/TK=hall, VH/VAR/KOMERO=storage, KHH/TEKN=utility.
  An open kitchen inside another room (stove and counter, no own label) is a separate kitchen room.
- doors: id "D1", ... leaves = 1 or 2. in_wall = true if it sits in a gap of a wall.
  connects = the two sides: room ids, or "outside" (exterior / stairwell / corridor outside this
  apartment), "closet" (cupboard/wardrobe/shaft you cannot walk into), "unknown".
- openings: id "O1", ... gaps in walls WITHOUT a leaf between two spaces. connects as for doors.
- windows: every window in an exterior wall. panes = number of separate glass segments drawn.
- fixtures: every wc_seat, sink (kitchen or bathroom; bowls = 1 or 2 for a double-bowl sink),
  shower, bathtub, sauna_stove. in_room = id of the room it is in.
- floors: number of separate floor levels (storeys) drawn on this sheet.

For EVERY item also give:
- box_2d: [ymin, xmin, ymax, xmax] normalised 0-1000 to this image
- status: "printed" (part of the original drawing), "hand_drawn" (pen/marker sketch or
  annotation added on top), or "crossed_out" (printed but struck through / scribbled over)
- cut_off: true if the item is partly outside the image edge
{rules}
Return ONLY JSON:
{{"floors": <int>,
  "rooms":    [{{"id": "R1", "label_text": "", "room_type": "", "box_2d": [], "status": "", "cut_off": false}}],
  "doors":    [{{"id": "D1", "leaves": 1, "in_wall": true, "connects": ["R1", "R2"], "box_2d": [], "status": "", "cut_off": false}}],
  "openings": [{{"id": "O1", "connects": ["R1", "R2"], "box_2d": [], "status": "", "cut_off": false}}],
  "windows":  [{{"panes": 1, "box_2d": [], "status": "", "cut_off": false}}],
  "fixtures": [{{"type": "", "bowls": 1, "in_room": "R1", "box_2d": [], "status": "", "cut_off": false}}]}}"""

MAP_PROMPT = """Translate a question about a floor plan into a query over this schema.
Collections and their type values:
- rooms: room_type in {room_types}
- doors (no types), windows (no types)
- fixtures: type in {fixture_types}
- floors (the number of storeys)
The question may be in any language (often English or Finnish).
Question: {question}
Return ONLY JSON: {{"collection": "<one of {collections}>", "types": [<type values, [] = all>],
"answerable": true/false}}
Set answerable=false if the question is not a count over one of these collections."""


# ----------------------------------------------------------------------------- model calls

class Usage:
    def __init__(self):
        self.local = threading.local()   # .last = (input_tokens, output_tokens) of this thread's last call
        self.lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


def make_text_model(provider: str, model: str, timeout: int, usage: Usage, mock: bool):
    """returns call(prompt, img_bytes|None) -> text"""
    if mock:
        return mock_model
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            sys.exit("Missing package: pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY (repo .env).")
        client = anthropic.Anthropic(timeout=timeout, max_retries=4)

        def call(prompt: str, img: bytes | None = None, max_tokens: int = 8000) -> str:
            content = []
            if img is not None:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.standard_b64encode(img).decode()}})
            content.append({"type": "text", "text": prompt})
            try:
                msg = client.messages.create(model=model, max_tokens=max_tokens,
                                             messages=[{"role": "user", "content": content}])
            except anthropic.RateLimitError as e:
                raise vt.RateLimited(str(e)) from e
            tin, tout = getattr(msg.usage, "input_tokens", 0), getattr(msg.usage, "output_tokens", 0)
            usage.local.last = (tin, tout)
            with usage.lock:
                usage.calls += 1
                usage.input_tokens += tin
                usage.output_tokens += tout
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return call

    gem = vt.make_caller(model, False, timeout, "gemini")

    def call(prompt: str, img: bytes | None = None, max_tokens: int = 8000) -> str:
        if img is None:  # gemini helper always sends an image: send a 1x1 blank
            img = vt.to_png_bytes(Image.new("RGB", (1, 1), "white"))
        usage.calls += 1
        return gem(img, prompt)
    return call


def mock_model(prompt: str, img: bytes | None = None, max_tokens: int = 0) -> str:
    """offline stand-in: random graphs, keyword query mapping (tests only)"""
    if img is None:
        q = prompt.split("Question:", 1)[1].split("\n", 1)[0].lower()
        for words, coll, types in [
            (("door", "ove"), "doors", []), (("window", "ikkun"), "windows", []),
            (("sauna",), "rooms", ["sauna"]), (("bedroom", "makuu"), "rooms", ["bedroom"]),
            (("toilet", "wc-ist"), "fixtures", ["wc_seat"]), (("floor", "kerros"), "floors", [])]:
            if any(w in q for w in words):
                return json.dumps({"collection": coll, "types": types, "answerable": True})
        return json.dumps({"collection": "rooms", "types": [], "answerable": False})
    rnd = random.Random(len(img))
    if "CANDIDATE" in prompt:
        return json.dumps({"decisions": {str(i): rnd.random() < 0.8 for i in range(1, 15)},
                           "missed": [{"box_2d": [500, 500, 540, 560]}] if rnd.random() < 0.5 else []})

    def item(**kw):
        y, x = rnd.randint(0, 900), rnd.randint(0, 900)
        return {"box_2d": [y, x, y + rnd.randint(30, 90), x + rnd.randint(30, 90)],
                "status": rnd.choice(["printed"] * 6 + ["hand_drawn", "crossed_out"]),
                "cut_off": rnd.random() < 0.1, **kw}
    rooms = [item(id=f"R{i}", label_text="MH", room_type=rnd.choice(ROOM_TYPES)) for i in range(1, rnd.randint(3, 7))]
    ends = [r["id"] for r in rooms] + OUTSIDE_SIDES

    def two():
        return rnd.sample(ends, 2)
    return json.dumps({
        "floors": 1, "rooms": rooms,
        "doors": [item(id=f"D{i}", leaves=rnd.choice([1, 1, 2]), in_wall=rnd.random() < 0.9, connects=two())
                  for i in range(1, rnd.randint(3, 10))],
        "openings": [item(id=f"O{i}", connects=two()) for i in range(1, rnd.randint(1, 3))],
        "windows": [item(panes=rnd.choice([1, 2])) for _ in range(rnd.randint(1, 6))],
        "fixtures": [item(type=rnd.choice(FIXTURE_TYPES), bowls=1, in_room=rnd.choice(ends)) for _ in range(rnd.randint(1, 5))],
    })


# ----------------------------------------------------------------------------- graph

def overlap_dup(a, b, iou_thr=0.5, contain_thr=0.8) -> bool:
    """same instance if IoU is high OR one box lies mostly inside the other"""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter = ix * iy
    aa, ba = (ay1 - ay0) * (ax1 - ax0), (by1 - by0) * (bx1 - bx0)
    if inter <= 0:
        return False
    return inter / (aa + ba - inter) > iou_thr or inter / min(aa, ba) > contain_thr


def side_kind(sides: list[str], room_ids: set[str]) -> str:
    """what a door/opening connects: passage (two rooms), entrance, closet or unknown"""
    rooms = [x for x in sides if x in room_ids]
    if len(set(rooms)) == 2:
        return "passage"
    if "closet" in sides:
        return "closet"
    if "outside" in sides and rooms:
        return "entrance"
    return "unknown"


# Finnish labels of spaces a person cannot walk into (cupboards, technical cabinets, shafts).
# Standard drawing vocabulary, like MH = bedroom; whether their doors count is decided by the
# fitted door_scope convention, not here.
CABINET_LABELS = {"KOM", "KOMERO", "SK", "SIIVOUSKOMERO", "SPK", "SÄHKÖPÄÄKESKUS", "HORMI", "KAAPPI"}


def mark_cabinets(g: dict) -> None:
    """rooms labelled as cabinets are not enterable; a door into one is a cabinet door"""
    cab = set()
    for r in g.get("rooms", []):
        label = re.sub(r"[^A-ZÅÄÖ]", "", str(r.get("label_text", "")).upper())
        r["enterable"] = label not in CABINET_LABELS
        if not r["enterable"]:
            cab.add(r["id"])
    for d in g.get("doors", []):
        if cab & set(d.get("connects", [])):
            d["kind"] = "closet"


def clean_graph(raw: dict) -> tuple[dict, int]:
    """validate items, normalise fields, keep the model's room ids (doors refer to them),
    drop duplicates, derive what each door/opening connects; returns graph, dropped"""
    g = {"floors": vt.as_int(raw.get("floors")) or 1}
    dropped = 0

    def base(it):
        box = vt.valid_box(it.get("box_2d")) if isinstance(it, dict) else None
        if box is None:
            return None
        node = dict(it)
        node["box_2d"] = box
        node["status"] = it.get("status") if it.get("status") in STATUSES else "printed"
        node["cut_off"] = bool(it.get("cut_off", False))
        return node

    # rooms first: their ids are referenced by doors, openings and fixtures
    rooms, seen = [], set()
    for it in raw.get("rooms") or []:
        node = base(it)
        rid = str(it.get("id", "")) if isinstance(it, dict) else ""
        if node is None or not rid or rid in seen:
            dropped += 1
            continue
        node["id"] = rid
        node["room_type"] = it.get("room_type") if it.get("room_type") in ROOM_TYPES else "other"
        seen.add(rid)
        rooms.append(node)
    g["rooms"] = rooms
    room_ids = {r["id"] for r in rooms}

    def sides(it):
        c = it.get("connects") if isinstance(it.get("connects"), list) else []
        c = [str(x) for x in c][:2]
        c = [x if (x in room_ids or x in OUTSIDE_SIDES) else "unknown" for x in c]
        return c + ["unknown"] * (2 - len(c))

    prefix = {"doors": "D", "openings": "O", "windows": "W", "fixtures": "F"}
    for coll in ("doors", "openings", "windows", "fixtures"):
        kept = []
        for it in raw.get(coll) or []:
            node = base(it)
            if node is None:
                dropped += 1
                continue
            if coll in ("doors", "openings"):
                node["connects"] = sides(it)
                node["kind"] = side_kind(node["connects"], room_ids)
            if coll == "doors":
                node["leaves"] = 2 if vt.as_int(it.get("leaves")) == 2 else 1
                node["in_wall"] = it.get("in_wall") is not False
                if not node["in_wall"]:
                    node["kind"] = "closet"     # door symbol on a single line = cupboard front
            if coll == "windows":
                node["panes"] = max(1, vt.as_int(it.get("panes")) or 1)
            if coll == "fixtures":
                if it.get("type") not in FIXTURE_TYPES:
                    dropped += 1
                    continue
                node["bowls"] = 2 if vt.as_int(it.get("bowls")) == 2 else 1
                node["in_room"] = it.get("in_room") if it.get("in_room") in room_ids else None
            same_kind = [k for k in kept if coll != "fixtures" or k["type"] == node["type"]]
            if any(overlap_dup(node["box_2d"], k["box_2d"]) for k in same_kind):
                dropped += 1
                continue
            kept.append(node)
        for i, node in enumerate(kept, 1):
            node["id"] = f"{prefix[coll]}{i}"
        g[coll] = kept
    mark_cabinets(g)
    g["unreachable"] = unreachable_rooms(g)
    return g, dropped


def unreachable_rooms(g: dict) -> list[str]:
    """rooms not connected (via doors or openings, crossed-out ones ignored) to the largest
    connected group of rooms: usually a missed door or opening -> flag for a second look"""
    ids = [r["id"] for r in g["rooms"] if r["status"] != "crossed_out"]
    if len(ids) < 2:
        return []
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for coll in ("doors", "openings"):
        for n in g.get(coll, []):
            a, b = n["connects"]
            if n["status"] != "crossed_out" and a in parent and b in parent:
                parent[find(a)] = find(b)
    groups = Counter(find(i) for i in ids)
    main = groups.most_common(1)[0][0]
    return [i for i in ids if find(i) != main]


SYMBOL_PROMPT = """This image is {part} of an architectural floor plan. Find EVERY instance of the
small symbols below. Be exhaustive: check every wall segment and every wet room. Never list the
same instance twice. Ignore anything outside the drawn building.

- windows: a window is drawn as 2-3 thin parallel lines (sometimes with a centre line or small
  frame marks) set INTO an exterior wall, interrupting the thick wall. One window unit = one item,
  even if it has several glass segments; panes = number of segments. Bay or corner windows: one
  item per straight wall section.
- doors: a leaf line with a swing arc (quarter circle) or a sliding door, in a gap in a wall.
  A double door (two leaves in one gap) is ONE item with leaves = 2. Doors side by side in
  different gaps are separate items. in_wall = false if the symbol is attached to a single thin
  line instead of a wall (cupboard/wardrobe front).
- fixtures:
  * wc_seat: toilet bowl (oval with a tank)
  * sink: wash basin or kitchen sink, in bathrooms, WCs, kitchens AND utility rooms (KHH);
    a double-bowl sink is ONE item with bowls = 2
  * shower: square or corner area (often with a drain dot, diagonal lines, a curved screen or
    a shower head symbol), usually inside a bathroom (KH, PH, PESUH, SH)
  * bathtub: long rounded rectangle
  * sauna_stove: stove/heater symbol inside a sauna (S, LH, SAUNA)

For every item: box_2d [ymin, xmin, ymax, xmax] normalised 0-1000 to THIS image; status
"printed" / "hand_drawn" (pen sketch added on top) / "crossed_out" (struck through);
cut_off true if the item continues beyond the edge of this image.
Return ONLY JSON:
{{"windows": [{{"panes": 1, "box_2d": [], "status": "", "cut_off": false}}],
  "doors": [{{"leaves": 1, "in_wall": true, "box_2d": [], "status": "", "cut_off": false}}],
  "fixtures": [{{"type": "", "bowls": 1, "box_2d": [], "status": "", "cut_off": false}}]}}"""


def make_tiles(w: int, h: int, n: int, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """overlapping crops (x0, y0, x1, y1): n=1 whole, 2 = halves along the long side, 4 = 2x2"""
    def split(size, parts):
        if parts == 1:
            return [(0, size)]
        length = round(size * (1 + overlap) / 2)
        return [(0, length), (size - length, size)]
    if n == 1:
        return [(0, 0, w, h)]
    if n == 2:
        if w >= h:
            return [(a, 0, b, h) for a, b in split(w, 2)]
        return [(0, a, w, b) for a, b in split(h, 2)]
    return [(xa, ya, xb, yb) for ya, yb in split(h, 2) for xa, xb in split(w, 2)]


def fit_image(img: Image.Image, min_side: int, max_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if min_side and long_side < min_side:
        s = min_side / long_side
    elif max_side and long_side > max_side:
        s = max_side / long_side
    else:
        return img
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def tile_items_to_plan(data: dict, tile, W: int, H: int, edge: float = 0.015) -> dict:
    """convert boxes from tile coords to whole-plan 0-1000 coords; drop items that touch an
    INTERNAL tile edge (the neighbouring tile overlaps there and sees them whole)"""
    x0, y0, x1, y1 = tile
    tw, th = x1 - x0, y1 - y0
    out = {"windows": [], "doors": [], "fixtures": []}
    for coll in out:
        for it in data.get(coll) or []:
            b = vt.valid_box(it.get("box_2d")) if isinstance(it, dict) else None
            if b is None:
                continue
            by0, bx0, by1, bx1 = b
            touches = ((bx0 < edge * 1000 and x0 > 0) or (bx1 > (1 - edge) * 1000 and x1 < W) or
                       (by0 < edge * 1000 and y0 > 0) or (by1 > (1 - edge) * 1000 and y1 < H))
            if touches:
                continue
            gb = [(y0 + by0 / 1000 * th) / H * 1000, (x0 + bx0 / 1000 * tw) / W * 1000,
                  (y0 + by1 / 1000 * th) / H * 1000, (x0 + bx1 / 1000 * tw) / W * 1000]
            out[coll].append({**it, "box_2d": gb})
    return out


def merge_symbols(layout: dict, sym: dict) -> tuple[dict, int]:
    """layout graph (rooms, openings, floors, door connections) + zoomed symbol pass (windows,
    doors, fixtures). Items found by both passes are merged (sources = [symbol, layout]); items
    found by only one pass are kept with that source, so the counting conventions can decide
    from the answer key whether to trust the symbol pass, the layout pass, or their union.
    Doors inherit what they connect from the overlapping layout door; fixtures get the room
    whose box contains their centre."""
    raw = {"floors": layout.get("floors"), "rooms": layout.get("rooms", []),
           "openings": layout.get("openings", []),
           "doors": sym.get("doors", []), "windows": sym.get("windows", []), "fixtures": sym.get("fixtures", [])}
    g, dropped = clean_graph(raw)
    prefix = {"doors": "D", "windows": "W", "fixtures": "F"}
    for coll in ("doors", "windows", "fixtures"):
        for n in g[coll]:
            n["sources"] = ["symbol"]
        extra = []
        for ln in layout.get(coll, []):
            same = [n for n in g[coll] if coll != "fixtures" or n["type"] == ln["type"]]
            match = next((n for n in same if overlap_dup(n["box_2d"], ln["box_2d"], 0.3, 0.6)), None)
            if match is not None:
                if "layout" not in match["sources"]:
                    match["sources"].append("layout")
                if coll == "doors":
                    match["connects"] = list(ln["connects"])
                    if match["in_wall"]:
                        match["kind"] = ln["kind"]
                    match["layout_id"] = ln["id"]
            else:
                extra.append({**ln, "sources": ["layout"]})
        g[coll] += extra
        for i, n in enumerate(g[coll], 1):
            n["id"] = f"{prefix[coll]}{i}"
    for f in g["fixtures"]:
        cy, cx = (f["box_2d"][0] + f["box_2d"][2]) / 2, (f["box_2d"][1] + f["box_2d"][3]) / 2
        inside = [r for r in g["rooms"] if r["box_2d"][0] <= cy <= r["box_2d"][2] and r["box_2d"][1] <= cx <= r["box_2d"][3]]
        if inside:
            f["in_room"] = min(inside, key=lambda r: (r["box_2d"][2] - r["box_2d"][0]) * (r["box_2d"][3] - r["box_2d"][1]))["id"]
    mark_cabinets(g)
    g["unreachable"] = unreachable_rooms(g)
    return g, dropped


VERIFY_DEFS = {
    "windows": ("window",
                "a window is 2-3 thin parallel lines (sometimes with a centre line or frame marks) set "
                "INTO an exterior wall, interrupting the thick wall. One window unit = one item even with "
                "several glass segments; a bay/corner window = one item per straight wall section. NOT a "
                "window: wall openings, doors, balcony railings, radiators, furniture, dimension lines."),
    "doors": ("door",
              "a door is a leaf line, usually with a swing arc (quarter circle), or a sliding door, sitting "
              "in a gap of a wall. A double door (two leaves in one gap) is ONE door. NOT a door: a gap "
              "without a leaf (opening), a cupboard/wardrobe front attached to a single thin line, "
              "furniture, stairs, window symbols."),
}

VERIFY_PROMPT = """Numbered red boxes on this floor plan mark CANDIDATE {plural}. Check each one.
Definition: {definition}
For each number decide: true if the box contains exactly one real {name}; false if it is not a
{name}, or is a duplicate of another numbered box (keep the better one). Then look over the whole
plan for any real {name} that has NO box and add it.
Return ONLY JSON:
{{"decisions": {{"1": true, "2": false}},
  "missed": [{{"box_2d": [ymin, xmin, ymax, xmax]}}]}}
box_2d normalised 0-1000 to this image."""


def draw_candidates(img: Image.Image, nodes: list[dict]) -> Image.Image:
    from PIL import ImageFont
    im = img.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(2, round(max(w, h) / 500))
    try:
        font = ImageFont.load_default(size=max(14, round(max(w, h) / 70)))
    except TypeError:   # older Pillow
        font = ImageFont.load_default()
    for i, n in enumerate(nodes, 1):
        y0, x0, y1, x1 = n["box_2d"]
        box = (x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h)
        d.rectangle(box, outline=(230, 0, 0), width=lw)
        tx, ty = box[0], max(0, box[1] - font.size - 2) if hasattr(font, "size") else box[1]
        d.rectangle((tx, ty, tx + (len(str(i)) + 1) * (getattr(font, "size", 10) * 0.6), ty + getattr(font, "size", 10) + 2),
                    fill=(255, 255, 255))
        d.text((tx + 2, ty), str(i), fill=(230, 0, 0), font=font)
    return im


def pass_counts(g: dict, coll: str) -> tuple[int, int]:
    """how many printed items each pass found (layout pass vs zoomed symbol pass)"""
    items = [n for n in g.get(coll, []) if n["status"] != "crossed_out"]
    return (sum("layout" in n.get("sources", []) for n in items),
            sum("symbol" in n.get("sources", []) for n in items))


def apply_verification(g: dict, coll: str, v: dict) -> None:
    """mark candidates kept/rejected and add missed items; the graph remembers it was verified"""
    dec = v.get("decisions") or {}
    cands = v["candidates"]
    by_id = {n["id"]: n for n in g[coll]}
    for i, nid in enumerate(cands, 1):
        if nid in by_id:
            by_id[nid]["verified"] = dec.get(str(i)) is True
    prefix = {"windows": "W", "doors": "D"}[coll]
    k = len(g[coll])
    for it in v.get("missed") or []:
        b = vt.valid_box(it.get("box_2d")) if isinstance(it, dict) else None
        if b is None or any(overlap_dup(b, n["box_2d"]) for n in g[coll] if n.get("verified")):
            continue
        k += 1
        node = {"id": f"{prefix}{k}", "box_2d": b, "status": "printed", "cut_off": False,
                "sources": ["verify"], "verified": True}
        if coll == "doors":
            node.update(leaves=1, in_wall=True, connects=["unknown", "unknown"], kind="unknown")
        if coll == "windows":
            node["panes"] = 1
        g[coll].append(node)
    g.setdefault("verified", []).append(coll)


# ----------------------------------------------------------------------------- conventions

CONVENTION_OPTIONS = {
    # first option of each list = default / preferred on ties (simplest reading)
    "statuses": [("printed",), ("printed", "crossed_out"), ("printed", "hand_drawn"),
                 ("printed", "hand_drawn", "crossed_out")],
    "cut_off": [True, False],                       # count items cut by the image edge?
    "door_unit": ["door", "leaf"],
    # which doors: all / not cupboard fronts / also not the front door / only between two known rooms
    "door_scope": ["all", "no_closet", "interior", "passage_only"],
    "window_unit": ["window", "pane"],
    # v2 graphs only: which pass to trust for symbols (zoomed pass, both passes, layout pass)
    "sources": [("symbol",), ("symbol", "layout"), ("layout",)],
    # doors: skip long thin boxes (a bar in a wall has no swing arc; also drops sliding doors)
    "door_shape": ["any", "no_sliver"],
    # doors: skip a box that lies mostly inside a bigger door box (same door boxed twice)
    "door_nested": [True, False],
    # crossed-out ROOMS: also skip the doors, windows and fixtures that belong to them?
    "void": [False, True],
}
APPLIES = {
    "rooms": ["statuses", "cut_off"],
    "doors": ["statuses", "cut_off", "door_unit", "door_scope", "void", "door_shape", "door_nested"],
    "windows": ["statuses", "cut_off", "window_unit", "void"],
    "fixtures": ["statuses", "cut_off", "void"],
    "floors": [],
}
DOOR_SCOPE_EXCLUDES = {"all": set(), "no_closet": {"closet"}, "interior": {"closet", "entrance"},
                       "passage_only": {"closet", "entrance", "unknown"}}


USE_SOURCES = False   # set in main() when the symbol pass is used


def conv_keys(coll: str) -> list[str]:
    keys = list(APPLIES[coll])
    if USE_SOURCES and coll in ("doors", "windows", "fixtures"):
        keys.append("sources")
    return keys


def default_conv(coll: str) -> dict:
    return {k: CONVENTION_OPTIONS[k][0] for k in conv_keys(coll)}


def in_void(graph: dict, coll: str, n: dict, margin: float = 15) -> bool:
    """does this item belong to a room that is crossed out on the drawing?
    By link (a door connects to it, a fixture is in it), else by position: the item's centre
    lies in a crossed-out room's box (slightly enlarged, items sit on walls) and in no other room."""
    void = [r for r in graph.get("rooms", []) if r["status"] == "crossed_out"]
    if not void:
        return False
    void_ids = {r["id"] for r in void}
    if coll == "doors" and void_ids & set(n.get("connects", [])):
        return True
    if coll == "fixtures" and n.get("in_room") in void_ids:
        return True
    cy, cx = (n["box_2d"][0] + n["box_2d"][2]) / 2, (n["box_2d"][1] + n["box_2d"][3]) / 2

    def inside(r, m):
        y0, x0, y1, x1 = r["box_2d"]
        return y0 - m <= cy <= y1 + m and x0 - m <= cx <= x1 + m
    if not any(inside(r, margin) for r in void):
        return False
    return not any(inside(r, 0) for r in graph["rooms"] if r["status"] != "crossed_out")


def is_sliver(graph: dict, n: dict, ratio: float = 3.0) -> bool:
    """box much longer than wide, measured in pixels (boxes are normalised per axis)"""
    w, h = graph.get("size") or (1, 1)
    bh = (n["box_2d"][2] - n["box_2d"][0]) * h
    bw = (n["box_2d"][3] - n["box_2d"][1]) * w
    return max(bh, bw) > ratio * max(min(bh, bw), 1e-6)


def nested_in_bigger(n: dict, others: list[dict], share: float = 0.6) -> bool:
    y0, x0, y1, x1 = n["box_2d"]
    area = (y1 - y0) * (x1 - x0)
    for o in others:
        if o is n:
            continue
        oy0, ox0, oy1, ox1 = o["box_2d"]
        if (oy1 - oy0) * (ox1 - ox0) <= area:
            continue
        inter = max(0, min(y1, oy1) - max(y0, oy0)) * max(0, min(x1, ox1) - max(x0, ox0))
        if area > 0 and inter / area >= share:
            return True
    return False


def answer(graph: dict, query: dict, conv: dict) -> tuple[int | None, list[str]]:
    coll = query.get("collection")
    if coll == "floors":
        return graph.get("floors"), []
    nodes = graph.get(coll)
    if nodes is None:
        return None, []
    types = set(query.get("types") or [])
    total, used = 0, []
    for n in nodes:
        if n["status"] not in conv["statuses"]:
            continue
        if n["cut_off"] and not conv["cut_off"]:
            continue
        if coll == "rooms" and types and n["room_type"] not in types:
            continue
        if coll == "fixtures" and types and n["type"] not in types:
            continue
        if coll == "doors" and n["kind"] in DOOR_SCOPE_EXCLUDES[conv["door_scope"]]:
            continue
        if conv.get("void") and in_void(graph, coll, n):
            continue
        if coll == "doors" and conv.get("door_shape") == "no_sliver" and is_sliver(graph, n):
            continue
        if coll == "doors" and conv.get("door_nested") and nested_in_bigger(
                n, [o for o in nodes if o["status"] in conv["statuses"]]):
            continue
        if coll in graph.get("verified", []):
            if not n.get("verified"):
                continue
        elif "sources" in conv and not any(src in conv["sources"] for src in n.get("sources", ["layout"])):
            continue
        k = 1
        if coll == "doors" and conv["door_unit"] == "leaf":
            k = n["leaves"]
        if coll == "windows" and conv["window_unit"] == "pane":
            k = n["panes"]
        total += k
        used.append(n["id"])
    return total, used


def fit_conventions(qs: list[dict], graphs: dict, qmap: dict) -> tuple[dict, dict]:
    """per collection: try every combination of options, keep the most accurate on these questions"""
    by_coll = defaultdict(list)
    for q in qs:
        m = qmap.get(q["question"])
        if m and m.get("answerable") and q["path"] in graphs:
            by_coll[m["collection"]].append(q)
    best, table = {}, {}
    for coll in COLLECTIONS:
        keys = conv_keys(coll)
        rows = []
        for combo in itertools.product(*(CONVENTION_OPTIONS[k] for k in keys)):
            conv = dict(zip(keys, combo))
            ok = sum(answer(graphs[q["path"]], qmap[q["question"]], conv)[0] == q["truth"]
                     for q in by_coll[coll])
            rows.append((ok, conv))
        if not rows:
            continue
        top = max(r[0] for r in rows)                     # ties: keep the earliest (= simplest)
        best[coll] = next(c for ok, c in rows if ok == top)
        table[coll] = {"n": len(by_coll[coll]), "best": top,
                       "default": next(ok for ok, c in rows if c == default_conv(coll)),
                       "ranking": sorted(rows, key=lambda r: -r[0])[:5]}
    return best, table


# ----------------------------------------------------------------------------- overlays

COLORS = {"rooms": (230, 140, 0), "openings": (150, 60, 200), "doors": (220, 30, 30),
          "windows": (30, 90, 220), "fixtures": (20, 150, 60)}
KIND_TAG = {"closet": "c", "entrance": "e", "unknown": "?", "passage": ""}


def draw_graph(img: Image.Image, graph: dict, out: Path, highlight: set[str] | None = None) -> None:
    im = img.copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(2, round(max(w, h) / 450))
    unreachable = set(graph.get("unreachable", []))
    for coll, color in COLORS.items():
        for n in graph.get(coll, []):
            y0, x0, y1, x1 = n["box_2d"]
            box = (x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h)
            c = color
            if n["status"] == "crossed_out":
                c = (150, 150, 150)
            rejected = coll in graph.get("verified", []) and not n.get("verified")
            if rejected:
                c = (170, 170, 170)
            if highlight is not None and n["id"] not in highlight:
                c = (205, 205, 205)
            width = lw if n["status"] != "hand_drawn" else max(1, lw // 2)
            if coll == "rooms":
                width = max(1, lw // 2)
            d.rectangle(box, outline=c, width=width)
            tag = n["id"] + {"crossed_out": "x", "hand_drawn": "~"}.get(n["status"], "")
            if rejected:
                tag += " rejected"
            elif "verify" in n.get("sources", []):
                tag += " +added"
            if coll == "doors":
                tag += KIND_TAG.get(n.get("kind"), "")
            if coll == "rooms":
                tag += " " + (n.get("label_text") or n["room_type"]) + (" !" if n["id"] in unreachable else "")
            d.text((box[0] + lw, box[1] + lw), tag, fill=c)
    legend = ("rooms orange (! = unreachable) | doors red (e=front door, c=cupboard, ?=unclear) | "
              "openings purple | windows blue | fixtures green | grey=crossed out | ~=hand drawn")
    d.rectangle((0, h - 16, w, h), fill=(255, 255, 255))
    d.text((4, h - 14), legend, fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


# ----------------------------------------------------------------------------- main

def plan_key(p: str) -> str:
    """file-safe unique name for a plan (stems can repeat across folders)"""
    return f"{Path(p).stem}_{hashlib.md5(p.encode()).hexdigest()[:6]}"


def main() -> None:
    global USE_SOURCES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--questions")
    ap.add_argument("--source", default="cubicasa5k")
    ap.add_argument("--from-results", help="results*.jsonl of the vision test: use exactly those plans/questions")
    ap.add_argument("--symbols", action="store_true",
                    help="v2: add a zoomed symbol pass (windows, doors, fixtures) on top of the cached layout graphs")
    ap.add_argument("--verify", default="", help="comma list of classes to verify where the two passes "
                    "disagree, e.g. windows,doors (needs --symbols)")
    ap.add_argument("--verify-all", action="store_true", help="verify even where both passes agree")
    ap.add_argument("--tiles", type=int, choices=[1, 2, 4], default=2, help="crops per plan for the symbol pass")
    ap.add_argument("--limit", type=int, help="only the first N plans (e.g. a quick 10-plan check)")
    ap.add_argument("--workers", type=int, default=3, help="plans extracted in parallel")
    ap.add_argument("--plans", type=int, default=30, help="number of plans when not using --from-results")
    ap.add_argument("--split", choices=["tune", "test", "all"], default="tune")
    ap.add_argument("--conventions", help="use this conventions.json instead of fitting (for the test half)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic")
    ap.add_argument("--model", help="layout pass, question mapping and verification (default: the provider's default)")
    ap.add_argument("--symbol-model", help="zoomed symbol pass (--symbols); default: same as --model. "
                                           "The reported run: --model claude-sonnet-5-5 --symbol-model claude-fable-5-2")
    ap.add_argument("--min-side", type=int, default=1536)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--rules-file")
    ap.add_argument("--price-in", type=float, help="USD per million input tokens (for the cost line)")
    ap.add_argument("--price-out", type=float, help="USD per million output tokens")
    ap.add_argument("--out", default="stoa_graph_out")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    args.model = args.model or vt.DEFAULT_MODELS[args.provider]
    args.symbol_model = args.symbol_model or args.model

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
    allq = vt.load_questions(Path(args.questions) if args.questions else root / "questions.jsonl")
    pool = []
    for r in allq:
        p = str(r.get("path", ""))
        t = vt.as_int(r.get("answer"))
        if args.source in p.replace("\\", "/") and t is not None:
            pool.append({"id": f"line{r['_line']}", "question": str(r["question"]).strip(),
                         "truth": t, "path": p, "half": vt.plan_half(p)})
    by_plan = defaultdict(list)
    for q in pool:
        by_plan[q["path"]].append(q)

    focus_ids: set[str] = set()
    if args.from_results:   # match on (file, question): robust to how the vision test numbered them
        keys = set()
        for line in Path(args.from_results).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["path"], str(r["question"]).strip()))
        focus_ids = {q["id"] for q in pool if (q["path"], q["question"]) in keys}
        plans = sorted({q["path"] for q in pool if q["id"] in focus_ids})
    else:
        cand = sorted(p for p in by_plan if args.split == "all" or vt.plan_half(p) == args.split)
        random.Random(args.seed).shuffle(cand)
        plans = cand[:args.plans]
    if args.limit:
        plans = plans[:args.limit]
    print(f"{len(plans)} plans, {sum(len(by_plan[p]) for p in plans)} questions on them.")

    out = Path(args.out)
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    usage = Usage()
    call = make_text_model(args.provider, args.model, args.timeout, usage, args.mock)
    sym_call = (call if args.symbol_model == args.model
                else make_text_model(args.provider, args.symbol_model, args.timeout, usage, args.mock))
    rules = ""
    if args.rules_file:
        rules = "\nExtra definitions:\n" + Path(args.rules_file).read_text(encoding="utf-8") + "\n"
    prompt = EXTRACT_PROMPT.format(room_types=ROOM_TYPES, rules=rules)

    # 1. extract each plan once (cached, in parallel)
    graphs, imgs = {}, {}
    other_model: Counter = Counter()
    t_start = time.time()
    todo = []
    for p in plans:
        img_path = vt.resolve_image(root, p)
        if img_path is None:
            print(f"image not found: {p}")
            continue
        imgs[p] = vt.prepare_image(img_path, args.min_side, False, args.max_side)
        gpath = out / "graphs" / (plan_key(p) + ".json")
        if gpath.is_file():
            saved = json.loads(gpath.read_text(encoding="utf-8"))
            graphs[p] = saved["graph"]
            mark_cabinets(graphs[p])
            if saved.get("model") and saved["model"] != args.model:
                other_model[saved["model"]] += 1
        else:
            todo.append(p)
    print(f"{len(graphs)} graphs cached, {len(todo)} to extract with {args.workers} workers.")
    if other_model:   # the cache is keyed by plan, not by model: say so instead of silently mixing
        print(f"  note: cached layout graphs from another model than {args.model}: {dict(other_model)}")
    stop = threading.Event()

    def extract(p):
        if stop.is_set():
            return p, None, "skipped", 0.0
        t0 = time.time()
        try:
            usage.local.last = (0, 0)
            raw = vt.parse_json(call(prompt, vt.to_png_bytes(imgs[p])))
            tokens = list(usage.local.last)
            g, dropped = clean_graph(raw)
        except vt.RateLimited as e:
            stop.set()
            return p, None, f"RATE LIMITED: {str(e)[:150]}", time.time() - t0
        except Exception as e:
            return p, None, f"ERROR {type(e).__name__}: {str(e)[:200]}", time.time() - t0
        (out / "graphs" / (plan_key(p) + ".json")).write_text(
            json.dumps({"path": p, "model": args.model, "seconds": round(time.time() - t0, 1),
                        "tokens": tokens, "dropped": dropped, "graph": g}, ensure_ascii=False, indent=1), encoding="utf-8")
        if not args.mock:
            time.sleep(args.sleep)
        return p, g, dropped, time.time() - t0

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool_ex:
        futs = [pool_ex.submit(extract, p) for p in todo]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            p, g, info, secs = fut.result()
            if g is None:
                if info != "skipped":
                    print(f"[{i}/{len(todo)}] {Path(p).name}: {info}")
                continue
            graphs[p] = g
            counts = " ".join(f"{c}={len(g[c])}" for c in ("rooms", "doors", "openings", "windows", "fixtures"))
            if g["unreachable"]:
                counts += f" UNREACHABLE {','.join(g['unreachable'])}"
            print(f"[{i}/{len(todo)}] {secs:5.1f}s {Path(p).name}: {counts} (dup/invalid dropped {info})")
    if stop.is_set():
        print("Stopped early because of rate limits: re-run later, finished graphs are cached.")
    layout_graphs = dict(graphs)
    sym_tokens = [0, 0]
    sym_calls = 0
    if args.symbols:
        (out / "symbols").mkdir(exist_ok=True)

        def symbols(p):
            spath = out / "symbols" / f"{plan_key(p)}_t{args.tiles}.json"
            if spath.is_file():
                d = json.loads(spath.read_text(encoding="utf-8"))
                if d.get("model") and d["model"] != args.symbol_model:
                    print(f"  note: cached symbols for {Path(p).name} are from {d['model']}, not {args.symbol_model}")
                return p, d, None, True
            img_path = vt.resolve_image(root, p)
            orig = Image.open(img_path).convert("RGB")
            W, H = orig.size
            tiles = make_tiles(W, H, args.tiles)
            merged = {"windows": [], "doors": [], "fixtures": []}
            tok = [0, 0]
            t0 = time.time()
            try:
                for k, t in enumerate(tiles):
                    crop = fit_image(orig.crop(t), args.min_side, args.max_side)
                    part = "the whole" if len(tiles) == 1 else f"crop {k + 1} of {len(tiles)} (overlapping)"
                    usage.local.last = (0, 0)
                    data = vt.parse_json(sym_call(SYMBOL_PROMPT.format(part=part), vt.to_png_bytes(crop)))
                    tok = [tok[0] + usage.local.last[0], tok[1] + usage.local.last[1]]
                    part_items = tile_items_to_plan(data, t, W, H)
                    for c in merged:
                        merged[c] += part_items[c]
            except vt.RateLimited as e:
                stop.set()
                return p, None, f"RATE LIMITED: {str(e)[:150]}", False
            except Exception as e:
                return p, None, f"ERROR {type(e).__name__}: {str(e)[:200]}", False
            d = {"path": p, "model": args.symbol_model, "tiles": args.tiles, "calls": len(tiles), "tokens": tok,
                 "seconds": round(time.time() - t0, 1), "items": merged}
            spath.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
            if not args.mock:
                time.sleep(args.sleep)
            return p, d, None, False

        todo_s = [p for p in graphs]
        print(f"Symbol pass ({args.tiles} crops per plan) on {len(todo_s)} plans ...")
        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool_ex:
            futs = [pool_ex.submit(symbols, p) for p in todo_s]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                p, d, err, cached = fut.result()
                if d is None:
                    print(f"[{i}/{len(todo_s)}] {Path(p).name}: {err}")
                    continue
                g2, dropped = merge_symbols(layout_graphs[p], d["items"])
                graphs[p] = g2
                sym_calls += d.get("calls", 0)
                sym_tokens = [sym_tokens[0] + d["tokens"][0], sym_tokens[1] + d["tokens"][1]]
                if not cached:
                    lg = layout_graphs[p]
                    print(f"[{i}/{len(todo_s)}] {d['seconds']:5.1f}s {Path(p).name}: windows {len(lg['windows'])}->{len(g2['windows'])} "
                          f"doors {len(lg['doors'])}->{len(g2['doors'])} fixtures {len(lg['fixtures'])}->{len(g2['fixtures'])}")
    pre_verify = {p: json.loads(json.dumps(g)) for p, g in graphs.items()}
    ver_calls, ver_tokens, ver_triggered, ver_checked = 0, [0, 0], 0, 0
    vclasses = [c.strip() for c in args.verify.split(",") if c.strip() in VERIFY_DEFS]
    if args.symbols and vclasses:
        (out / "verify").mkdir(exist_ok=True)
        jobs = []
        for p, g in graphs.items():
            for coll in vclasses:
                ver_checked += 1
                lay_n, sym_n = pass_counts(g, coll)
                if lay_n != sym_n or args.verify_all:
                    jobs.append((p, coll))
        ver_triggered = len(jobs)
        print(f"Verification: {len(jobs)} of {ver_checked} plan/class pairs disagree -> one call each")

        def verify(job):
            p, coll = job
            vpath = out / "verify" / f"{plan_key(p)}_{coll}.json"
            g = graphs[p]
            cands = [n for n in g[coll] if n["status"] != "crossed_out"]
            if vpath.is_file():
                return job, json.loads(vpath.read_text(encoding="utf-8")), True
            name, definition = VERIFY_DEFS[coll]
            img = draw_candidates(imgs[p], cands)
            t0 = time.time()
            try:
                usage.local.last = (0, 0)
                data = vt.parse_json(call(VERIFY_PROMPT.format(plural=coll, name=name, definition=definition),
                                          vt.to_png_bytes(img), 3000))
            except vt.RateLimited as e:
                stop.set()
                return job, {"error": str(e)[:150]}, False
            except Exception as e:
                return job, {"error": f"{type(e).__name__}: {str(e)[:150]}"}, False
            v = {"candidates": [n["id"] for n in cands], "decisions": data.get("decisions") or {},
                 "missed": data.get("missed") or [], "tokens": list(usage.local.last),
                 "seconds": round(time.time() - t0, 1)}
            vpath.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
            if not args.mock:
                time.sleep(args.sleep)
            return job, v, False

        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool_ex:
            for i, fut in enumerate(cf.as_completed([pool_ex.submit(verify, j) for j in jobs]), 1):
                (p, coll), v, cached = fut.result()
                if "error" in v:
                    print(f"[verify {i}/{len(jobs)}] {Path(p).name} {coll}: {v['error']}")
                    continue
                before = len([n for n in graphs[p][coll] if n["status"] != "crossed_out"])
                apply_verification(graphs[p], coll, v)
                ver_calls += 1
                ver_tokens = [ver_tokens[0] + v["tokens"][0], ver_tokens[1] + v["tokens"][1]]
                if not cached:
                    after = sum(1 for n in graphs[p][coll] if n.get("verified"))
                    print(f"[verify {i}/{len(jobs)}] {v['seconds']:5.1f}s {Path(p).name} {coll}: "
                          f"{before} candidates -> {after} verified")
    extract_calls = usage.calls

    # 2. map each distinct question wording to a query (cached across runs)
    qmap_path = out / "query_map.json"
    qmap = json.loads(qmap_path.read_text(encoding="utf-8")) if qmap_path.is_file() else {}
    texts = sorted({q["question"] for p in graphs for q in by_plan[p]} - set(qmap))
    for t in texts:
        try:
            m = vt.parse_json(call(MAP_PROMPT.format(room_types=ROOM_TYPES, fixture_types=FIXTURE_TYPES,
                                                     collections=COLLECTIONS, question=t), None, 300))
            if m.get("collection") not in COLLECTIONS:
                m["answerable"] = False
            qmap[t] = {"collection": m.get("collection"), "types": m.get("types") or [],
                       "answerable": bool(m.get("answerable", True))}
        except Exception as e:
            print(f"query mapping failed for {t[:60]!r}: {e}")
    qmap_path.write_text(json.dumps(qmap, ensure_ascii=False, indent=1), encoding="utf-8")

    USE_SOURCES = bool(args.symbols)
    sfx = (f"_v2t{args.tiles}" + ("_ver" if vclasses else "")) if args.symbols else ""
    for gs in (graphs, layout_graphs):
        for p_, g_ in gs.items():
            if p_ in imgs:
                g_["size"] = list(imgs[p_].size)
    # 3. conventions: fit on these plans' questions, or load
    qs = [q for p in graphs for q in by_plan[p]]
    if args.conventions:
        conv = json.loads(Path(args.conventions).read_text(encoding="utf-8"))
        fit_table = {}
    else:
        conv, fit_table = fit_conventions(qs, graphs, qmap)
        (out / f"conventions{sfx}.json").write_text(json.dumps(conv, indent=1), encoding="utf-8")

    # 4. answer every question from the graph
    rows = []
    for q in qs:
        m = qmap.get(q["question"])
        if not (m and m.get("answerable")):
            pred, used, coll = None, [], None
        else:
            coll = m["collection"]
            pred, used = answer(graphs[q["path"]], m, conv.get(coll, default_conv(coll)))
            dpred = answer(graphs[q["path"]], m, default_conv(coll))[0]
        rows.append({**q, "collection": coll, "pred": pred, "pred_default": dpred if coll else None,
                     "used": used, "correct": pred == q["truth"]})
    with (out / f"answers{sfx}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. overlays: one per plan with all nodes
    failed = []
    for p, g in graphs.items():
        if p in imgs:
            try:
                draw_graph(imgs[p], g, out / f"overlays{sfx}" / (plan_key(p) + ".png"))
            except OSError as e:   # e.g. the PNG is open in an image viewer on Windows
                failed.append(f"{plan_key(p)}.png ({e.strerror})")
    if failed:
        print(f"Could not write {len(failed)} overlay(s), close them in your image viewer and re-run: "
              + ", ".join(failed[:5]))

    # 6. report
    models = (f"model {args.model}" if not args.symbols or args.symbol_model == args.model
              else f"layout model {args.model}, symbol model {args.symbol_model}")
    L = [f"# Stoa graph test: {len(graphs)} plans, {len(rows)} questions, {models}\n"]

    def acc(rs, key="correct"):
        return f"{sum(r[key] for r in rs) / len(rs):.1%} ({sum(r[key] for r in rs)}/{len(rs)})" if rs else "-"
    for r in rows:
        r["correct_default"] = r["pred_default"] == r["truth"]
    L.append("| set | fitted conventions | default conventions |\n|---|---|---|")
    L.append(f"| all questions on these plans | **{acc(rows)}** | {acc(rows, 'correct_default')} |")
    if focus_ids:
        fr = [r for r in rows if r["id"] in focus_ids]
        L.append(f"| the {len(fr)} vision-test questions (was 77% question by question) | **{acc(fr)}** | "
                 f"{acc(fr, 'correct_default')} |")
    if args.symbols:
        USE_SOURCES = False
        conv1, _ = fit_conventions(qs, layout_graphs, qmap)
        USE_SOURCES = True
        ok1 = 0
        for q in qs:
            m = qmap.get(q["question"])
            if m and m.get("answerable"):
                ok1 += answer(layout_graphs[q["path"]], m, conv1.get(m["collection"], default_conv(m["collection"])))[0] == q["truth"]
        L.append(f"| same plans, layout graph only (v1) | {ok1 / max(len(qs), 1):.1%} ({ok1}/{len(qs)}) | |")
    if ver_calls:
        okn = 0
        for q in qs:
            m = qmap.get(q["question"])
            if m and m.get("answerable"):
                okn += answer(pre_verify[q["path"]], m, conv.get(m["collection"], default_conv(m["collection"])))[0] == q["truth"]
        L.append(f"| same plans, v2 without verification | {okn / max(len(qs), 1):.1%} ({okn}/{len(qs)}) | |")
    # baseline: always answer the most common answer for that question wording (learned from other plans)
    these = set(graphs)
    base_c: dict[str, Counter] = defaultdict(Counter)
    for q in pool:
        if q["path"] not in these:
            base_c[q["question"]][q["truth"]] += 1
    base = {t: c.most_common(1)[0][0] for t, c in base_c.items()}
    okb = sum(base.get(r["question"]) == r["truth"] for r in rows)
    L.append(f"| baseline: most common answer per question (no model) | {okb / max(len(rows), 1):.1%} ({okb}/{len(rows)}) | |")
    L.append("\nDefault = count printed items only, including cut-off ones; every door and window as one unit.")
    if args.conventions:
        L.append(f"\nConventions loaded from `{args.conventions}` (not fitted on these plans).")

    n_q = len(rows)
    L.append("\n## Cost (all plans in this report, from cached call records)\n")
    lay_tok = [0, 0]
    lay_known = 0
    for p in graphs:
        f = out / "graphs" / (plan_key(p) + ".json")
        if f.is_file():
            t = json.loads(f.read_text(encoding="utf-8")).get("tokens")
            if t and sum(t) > 0:
                lay_tok = [lay_tok[0] + t[0], lay_tok[1] + t[1]]
                lay_known += 1
    n_plans = len(graphs)
    calls = n_plans + sym_calls + ver_calls
    L.append(f"- vision calls: {n_plans} layout + {sym_calls} symbol + {ver_calls} verification = {calls} for {n_q} questions "
             f"= {calls / max(n_q, 1):.3f} calls per question (question by question: 1.0)")
    if lay_known:
        L.append(f"- layout tokens (recorded for {lay_known} plans): {lay_tok[0]:,} in, {lay_tok[1]:,} out")
    if ver_calls:
        L.append(f"- verification: {ver_triggered} of {ver_checked} plan/class pairs disagreed and were verified; "
                 f"tokens {ver_tokens[0]:,} in, {ver_tokens[1]:,} out")
    if sym_calls:
        L.append(f"- symbol pass tokens: {sym_tokens[0]:,} in, {sym_tokens[1]:,} out")
    if args.price_in and args.price_out:
        if lay_known:
            per_plan = [lay_tok[0] / lay_known, lay_tok[1] / lay_known]
        else:   # layout tokens of older runs were not recorded: assume the measured 10-plan average
            per_plan = [4353, 1675]
        tin = per_plan[0] * n_plans + sym_tokens[0] + ver_tokens[0]
        tout = per_plan[1] * n_plans + sym_tokens[1] + ver_tokens[1]
        cost = tin / 1e6 * args.price_in + tout / 1e6 * args.price_out
        L.append(f"- estimated cost: ${cost:.2f} = ${cost / max(n_q, 1):.4f} per question")
    L.append(f"- query mapping: {len(qmap)} question wordings mapped once each (cached, text only)")
    L.append(f"- wall time: {time.time() - t_start:.0f} s")

    if fit_table:
        L.append("\n## Fitted counting conventions (what the answer key rewards)\n")
        L.append("| collection | questions | best | default | chosen convention |\n|---|---|---|---|---|")
        for coll, t in fit_table.items():
            if t["n"]:
                L.append(f"| {coll} | {t['n']} | {t['best']} | {t['default']} | `{json.dumps(conv[coll])}` |")
        L.append("\nTop alternatives per collection (correct answers, convention):\n")
        for coll, t in fit_table.items():
            if t["n"] and len(t["ranking"]) > 1:
                L.append(f"- **{coll}**: " + "; ".join(f"{ok} `{json.dumps(c)}`" for ok, c in t["ranking"]))

    n_unr = sum(1 for g in graphs.values() if g.get("unreachable"))
    kinds = Counter(n.get("kind") for g in graphs.values() for n in g.get("doors", []))
    L.append("\n## Graph checks\n")
    L.append(f"- plans with an unreachable room (likely a missed door/opening): {n_unr} of {len(graphs)}")
    n_void = sum(1 for g in graphs.values() if any(r["status"] == "crossed_out" for r in g.get("rooms", [])))
    L.append(f"- plans with a crossed-out room: {n_void} of {len(graphs)}")
    L.append(f"- doors by kind: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    if n_unr:
        rs = [r for r in rows if r["collection"] == "doors"]
        bad = [r for r in rs if graphs[r["path"]].get("unreachable")]
        good = [r for r in rs if not graphs[r["path"]].get("unreachable")]
        L.append(f"- door accuracy on plans that pass the check: {acc(good)}; on flagged plans: {acc(bad)}")

    L.append("\n## Per question template\n")
    L.append("| question | collection | n | accuracy | mean error (pred-true) |\n|---|---|---|---|---|")
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["question"]].append(r)
    for t, rs in sorted(by_t.items(), key=lambda kv: -len(kv[1])):
        bias = sum((r["pred"] or 0) - r["truth"] for r in rs) / len(rs)
        L.append(f"| {t[:70]} | {rs[0]['collection']} | {len(rs)} | {acc(rs)} | {bias:+.2f} |")

    L.append("\n## Wrong answers (first 30)\n")
    L.append("| plan | question | true | graph | counted nodes |\n|---|---|---|---|---|")
    for r in [r for r in rows if not r["correct"]][:30]:
        L.append(f"| {plan_key(r['path'])} | {r['question'][:50]} | {r['truth']} | {r['pred']} | "
                 f"{' '.join(r['used'][:20])} |")
    L.append("\nOverlays: `overlays/<plan>.png` (node ids match the table).")
    (out / f"report{sfx}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L[:8]))
    print(f"\nFull report: {out / f'report{sfx}.md'}")


if __name__ == "__main__":
    main()
