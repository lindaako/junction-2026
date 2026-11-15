# Make Blueprints Talk: read each drawing once, answer from a graph

Junction 2026, Stoa challenge. Team OneCodeAtATime.

Upload a scanned floor plan and ask how many doors, windows, bedrooms or toilets it has, in
English, Finnish or another language. You get the count, and every counted item is circled and
numbered on the drawing, so a person can check the answer in seconds. Each drawing is read once
into a graph; every question after that is a query over the graph, not another look at the image.

**Live demo: http://70.34.222.185** (upload your own blueprint, or open one of the examples)

## Try it

```
git clone -b stoa/make-blueprints-talk https://github.com/lindaako/junction-2026.git
cd junction-2026
python -m venv .venv            # then activate it
pip install -r requirements.txt
```

An Anthropic API key is required to start the app, even to browse the examples. Copy
`.env.example` to `.env` in the repo root using a text editor, and put your key after
`ANTHROPIC_API_KEY=`. Create the file in an editor rather than with `echo ... > .env`: Windows
PowerShell saves redirected output as UTF-16, which the app cannot read.

```
python scripts/stoa_app.py      # open http://127.0.0.1:8100
```

The app needs no other services. It starts with five example plans (see `examples/`), whose
answers need no API calls. Reading a new drawing makes three vision calls (about 20 cents, once);
questions after that cost nothing, except one small text call the first time a new wording is used.

On a drawing's page, **Check against known answers** takes a list of questions with the answers you
expect (`How many doors are there? = 5`, one per line) and returns a score with a ✓/✗ table. Each
row opens the evidence on the drawing. Questions it cannot answer, such as areas, are reported as
not supported instead of guessed.

To run it on a server: `python scripts/stoa_app.py --host 0.0.0.0 --port 80 --max-uploads 40`.
`--max-uploads` caps how many new drawings the server accepts, which caps the API spend.

## Result in one table

Measured on held-out floor plans that were never used for tuning.

| Method | Plans / questions | Exact-match accuracy | Vision calls per question |
|---|---|---|---|
| No model: most common answer per question | 40 / 750 | 58.4% | 0 |
| Graph, one model for everything (Claude Sonnet 5.5) | 39 / 726 | 70.8% | 0.16 |
| **Graph, Sonnet 5.5 for layout + Fable 5.2 for zoomed symbols** | **40 / 750** | **78.4%** | **0.16** |

The Sonnet-only run covers 39 of the 40 plans; the 40th plan's layout was read later for the final run.
For reference, asking Sonnet 5.5 each question separately scored 77% on 100 questions from the
tuning half, at 1.0 calls per question (baseline there: 62%).

Per question type on the 40 held-out plans (final configuration):

| Reliable | | Weak | |
|---|---|---|---|
| Floors | 100% | Sinks | 72.5% |
| Separate WC rooms | 100% | Balconies | 75% |
| Kitchens | 93% | **Doors** | **53%** |
| Saunas | 90% | **Windows** | **37.5%** |
| Bathrooms | 86% | | |
| WC seats | 85% | | |
| Bedrooms | 83% | | |
| Showers and bathtubs | 82% | | |

Most room types and labels are close to solved. Small repeated symbols, above all windows and
doors, are not, and exact-match scoring gives nothing for being off by one.

## The problem

The dataset has 127,044 questions. 67% of them are exact counts on low-resolution raster floor
plans (the cubicasa5k folder): doors, windows, toilets, sinks, saunas, bedrooms, in English and
Finnish. The plans carry deliberate noise: pen sketches, crossed-out rooms, removal notes,
mirrored sheets, plans that fill a small corner of a blank page. This project covers that
raster part. The vector and BIM parts of the dataset (Swiss DXF, IFC) are not implemented here.

## How it works

1. **Layout pass, once per plan.** A vision model reads the whole plan into a graph: rooms
   with labels and types, doors and openings with the two spaces each one connects, floors.
   Every item has a box on the drawing and a status (printed, hand-drawn, crossed out).
2. **Symbol pass, once per plan.** The plan is cut into two overlapping crops and a stronger
   model looks only for windows, doors and fixtures. Boxes are mapped back to plan
   coordinates and merged with the layout graph.
3. **Question to query, once per wording.** Each distinct question text is translated once
   into a query such as `{collection: fixtures, types: [wc_seat]}` and cached. The dataset has
   24 wordings for these plans.
4. **Answer = graph query.** No model call. The answer comes with the ids of the counted
   nodes, so the viewer can highlight them.
5. **Counting conventions are fitted, not hard-coded.** Whether crossed-out items count,
   whether a double door is one or two, which pass to trust for each class: each option is
   scored against the answer key on tuning plans, and the winning set is applied unchanged
   to held-out plans.

Plans are split into a tuning half and a test half by a hash of the file path.

## Showing the reasoning

- Every answer lists the counted items in words ("Door between MH and ET", "Toilet in KH"), with
  numbers that match red circles on the drawing. Hovering a row highlights its circle.
- Items it saw but did not count (crossed out, sketched in pen, cupboard fronts) are listed with
  the reason.
- Without a question selected, the drawing shows everything that was read, with a switch per
  layer (rooms, doors, windows, fixtures, openings) and one for rejected candidates.
- `scripts/stoa_export_demo.py` builds the same view as a static page from a scoring run, with
  the dataset answer next to each of ours.

## Cost

- 0.16 vision calls per question (3 calls per plan, about 19 questions per plan).
- Measured from the API balance: the zoomed symbol pass costs about $0.04 per plan with
  Sonnet 5.5 and about $0.17 per plan with Fable 5.2. The layout pass was not measured
  separately.
- Every further question on a plan that has been read is free, including new wordings after
  one small text call.

## What we tried that did not help

- **Verification pass.** Showing the model numbered candidate boxes and asking it to accept
  or reject each: 70.5% against 70.8% without, on the same held-out plans, for 43% more calls.
  Left out.
- **Expert rules in the extraction prompt.** We went through wrong door counts by hand with
  people who read floor plans, and wrote the explanations into the prompts (see the rule log
  below). On the 30 tuning plans the new reading scored 91.1%. On the 40 held-out plans it
  scored 75.5%, against 78.4% for the simpler prompt, with doors at 50% against 53%. The rules
  explain individual plans but did not improve unseen ones, so the reported configuration
  keeps the simpler prompt. The gap between 91% and 75% is also a measure of how much
  easier the tuning sample was and of how far tuning numbers can mislead.
- **A general rule that was wrong elsewhere.** "A thin bar without a swing arc is not a door"
  fixed one plan and would have removed real doors on an electrical plan that draws doors as
  wedges. It is a fitted option for that reason.

The lesson we take from this: the remaining errors are mostly perception (symbols missed,
boxed twice, or two doors in one box), not counting rules.

## Rule log from manual error analysis

| # | Rule | Evidence |
|---|---|---|
| 1 | An opening without a door leaf is not a door | stated in the question text |
| 2 | A cupboard or wardrobe front is not a door | several plans |
| 3 | Count one door per hinge point: leaves hung on opposite sides of an opening are two doors, leaves hung at the same point are one | three plans, one exact |
| 4 | Front doors count | fitted on all tuning plans |
| 5 | Items in, attached to, or only serving a crossed-out area do not count | two plans |
| 6 | A door into a cabinet space (SPK, KOM, SK) is not a door | one plan |
| 7 | A thin bar without a swing arc is not a door | one plan for, one against |
| 8 | A door box nested in a bigger door box is a duplicate | mechanical |
| 9 | A cupboard is a space too small to step into whose door spans most of its width; a space with its own label and area is a room | several plans |
| 10 | A wall or item marked for removal ("pois", zigzag, filled in) is gone, with its door | two plans |
| 11 | Door notation depends on the drawing type (arc, or thin wedge on technical plans) | one plan |

## Limitations

- Only the raster floor plans are covered. The Swiss DXF plans and IFC models, about a third
  of the questions, have exact geometry and should be answered from it without any model;
  that is not built.
- Windows and doors are unreliable.
- Some plans produce duplicated items across the two crops (one held-out plan counts 22 doors
  where there are 14).
- The reachability check (every room should be reachable through a door or opening) flags
  most plans, so it is not yet a useful confidence signal.
- Questions about areas, lengths, volumes, schedules and piping are not supported; the app
  says so rather than guessing.
- The demo server runs plain http on one small machine.
- Samples are small: 40 held-out plans, and each question appears in two languages, so 750
  questions are about 375 independent ones.

## Next steps

1. Answer Swiss DXF and IFC questions from geometry, into the same graph.
2. Crop blank margins before reading, so small plans fill the image.
3. More crops for windows.
4. Train a small symbol detector on boxes that the answer key confirms, to replace the
   symbol pass and cut cost further.

## Reproduce the numbers

Needs the challenge dataset and `ANTHROPIC_API_KEY` in `.env`. Run from the repo root.

```
pip install -r requirements.txt

# 1. Tuning plans: the 100-question baseline, question by question with Sonnet 5.5 (writes stoa_out/results_claude.jsonl)
python scripts/stoa_vision_test.py --root <dataset> --n 100 --split tune --provider anthropic --tag claude

# 2. Read the 96 plans behind those questions into graphs (Sonnet 5.5 for both passes) and fit the counting conventions
python scripts/stoa_graph.py --root <dataset> --from-results stoa_out/results_claude.jsonl --symbols --out stoa_graph_out

# 3. Held-out plans: Sonnet 5.5 reads the layout, Fable 5.2 reads the zoomed symbol crops, conventions fixed from step 2
python scripts/stoa_graph.py --root <dataset> --plans 40 --split test --symbols --model claude-sonnet-5-5 --symbol-model claude-fable-5-2 --conventions stoa_graph_out/conventions_v2t2.json --out stoa_graph_fable

# 4. Static viewer of the held-out run
python scripts/stoa_export_demo.py --root <dataset> --run stoa_graph_fable --out demo
```

Steps 1 and 2 can be skipped by passing `--conventions scripts/stoa_conventions.json` in step 3: that
file holds the conventions fitted in step 2. The plan samples are fixed by the scripts' default seeds.

Graphs are cached per plan, so re-scoring with different conventions makes no API calls. In the
reported run, one of the 40 layouts (`high_quality_architectural-10303`) was read by Fable 5.2
instead of Sonnet 5.5, so a fresh run of step 3 may differ slightly from 78.4%.
`scripts/stoa_vision_test.py` is also the question-by-question baseline.

## Files

| File | Purpose |
|---|---|
| `scripts/stoa_app.py`, `scripts/stoa_app.html` | Web app: upload, ask, evidence view, check against known answers |
| `scripts/stoa_graph.py` | Pipeline: read plans into graphs, map questions to queries, fit conventions, score |
| `scripts/stoa_conventions.json` | Counting conventions fitted on the tuning plans, used by the app |
| `scripts/stoa_vision_test.py` | Question-by-question baseline and shared helpers |
| `scripts/stoa_export_demo.py`, `scripts/stoa_viewer.html` | Static viewer of a scoring run |
| `examples/` | Five example plans with their graphs; `examples/LICENSE.md` gives their source and license |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for `.env` (API keys) |
| `LICENSE` | MIT license for the code |

## Credits and license

The code is released under the MIT license (`LICENSE`).

Example plans are from the CubiCasa5K dataset (Kalervo, Ylioinas, Häikiö, Karhu, Kannala, 2019,
https://zenodo.org/records/2613548), licensed CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/), as distributed in the Stoa Junction 2026
challenge dataset, some with markings added there. Vision models: Claude Sonnet 5.5 and
Claude Fable 5.2 (Anthropic).
