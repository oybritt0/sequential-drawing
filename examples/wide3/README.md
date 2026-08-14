# wide3 — an archived run, with its inputs and outputs

This is the run that produced the sheet the method was judged on. Everything the
Gemini stage produced is here: the input composition, every prompt, the base
sheet, the assembled result, the unified result, and the region key.

**The local LoRA finishing pass that followed is deliberately not included** —
it needs a checkpoint and adapter that cannot be redistributed, and it is not
part of this repo's method.

## What ran

| | |
| --- | --- |
| model | `gemini-3-pro-image` (`nano-banana-pro`) |
| composition | `composition.svg`, 24 tagged regions, coordinate x 11.34 y 6.54 |
| size | 1024 |
| calls | 25 — 24 regions plus one unifying pass |
| style references | three: `lai_1.jpg`, `lai_2.png`, `RB_1.png` |
| wall clock | ~9 minutes of the 35.9 the four-sheet batch took |
| failures | none |

## Files

| | |
| --- | --- |
| `composition.svg` | the input — 24 regions, each tagged with period and four facets |
| `composition.png` | the sparse_fit raster the curves were traced from |
| `fragments.csv` | every region's tags, in full |
| `00_base.png` | the starting sheet |
| `sequence/` | the canvas after regions 1, 6, 12, 18 and 24 — the build-up |
| `final.png` | all 24 regions drawn |
| `final_restyled.png` | after the whole-frame unifying pass |
| `prompt_map.png` `.csv` `.md` | which region got which prompt |

## Reproducing it

```bash
python -m sequential_drawing.run examples/wide3/composition.svg \
    --model nano-banana-pro --refs examples/wide3/reference --legacy-v3
```

`--legacy-v3` restores the two things that have since changed: the crop is sent
plain, with the instruction naming "the shape at the centre" rather than marking
the region, and the prompt carries neither the line-convention block nor the
no-text block.

Without that flag you get the current method, which is **better behaved on the
same input** — fewer centred rotundas, no leaked lettering, double-line walls
instead of solid poché.

## Three honest caveats

**Only one of the three style references is here.** `RB_1.png` is included;
`lai_1.jpg` and `lai_2.png` are third-party drawings and are not ours to
redistribute. They shaped the look, so a re-run will differ. `lai_1` in
particular is a dense plan-collage — it is what leaked a folded ramp motif and,
later, room labels including a named artist, which is why `--ref-mode patches`
became the default and why the no-text block was added.

**The run predates part of the current code.** wide3 finished at 14:22 on
2026-08-05; the crop-marking fix and the text suppression landed afterwards.
`--legacy-v3` covers the two that change the output most, not every line.

**`ref_mode` was not recorded.** The run log captured model, references, size and
timing but not the reference mode, so it is inferred as `patches`, the default at
the time. Runs since record every setting — that gap is closed going forward.

**There is no seed.** Gemini's image models expose no seed control, so the same
command will not return the same image twice. What replicates is the method, the
geometry and the prompts, not the pixels.
