r"""
run.py — draw one composition, region by region, then unify it.

Two stages:

    1. every curated region is generated in turn, each with its own prompt, each
       composited back through its own polygon so the authored geometry is exact
    2. one whole-frame pass over the assembled sheet to settle it into a single
       drawing hand, because regions generated independently drift apart

Resumable: a region whose file already exists is skipped, so a run stopped by a
rate limit or a lost connection can simply be started again.

    $env:GEMINI_API_KEY = "..."
    python -m sequential_drawing.run examples/composition.svg
    python -m sequential_drawing.run comp.svg --mode section --limit 6
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from . import backend as RB
from . import geometry as RD


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svg", type=Path, help="a composition SVG, or a folder holding one")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default="nano-banana-2",
                    help="nano-banana, nano-banana-2 (default), nano-banana-pro, "
                         "nano-banana-lite, or a raw Gemini model id")
    ap.add_argument("--mode", default="plan", choices=["plan", "section", "mixed"],
                    help="plan = horizontal cut, section = vertical cut, mixed = "
                         "each region takes what its own organization term implies")
    ap.add_argument("--refs", type=Path, default=None,
                    help="folder of style references; the first 3 are used")
    ap.add_argument("--ref-mode", default="patches", dest="ref_mode",
                    choices=["patches", "whole", "none"],
                    help="patches crops references to texture tiles so their "
                         "motifs cannot be copied; whole sends the full drawing")
    ap.add_argument("--base", default="full",
                    choices=["full", "lines", "flat", "detail", "source"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--feather", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N regions")
    ap.add_argument("--no-mono", action="store_false", dest="mono", default=True,
                    help="keep the model's paper tint instead of normalising it")
    ap.add_argument("--ink-clean", action="store_true", dest="ink_clean",
                    help="lift mid-grey hatching toward white after each region")
    ap.add_argument("--no-finish", action="store_false", dest="finish", default=True,
                    help="skip the whole-frame unifying pass")
    ap.add_argument("--legacy-v3", action="store_true", dest="legacy",
                    help="reproduce the archived examples/wide3 run: plain crop "
                         "with no region marking, and the older instruction "
                         "without the line-convention or no-text blocks")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the order, nesting and prompts; call nothing")
    a = ap.parse_args(argv)

    svg = a.svg if a.svg.suffix.lower() == ".svg" else a.svg / "composition.svg"
    if not svg.exists():
        raise SystemExit(f"no SVG at {svg}")
    frags = RD.nest(RD.parse_curated(svg))
    if not frags:
        raise SystemExit(f"no curated regions found in {svg.name}. Groups must be "
                         f"marked with data-reading=\"curated\" or named "
                         f"curated__frag_NN__...")
    if a.limit:
        frags = frags[:a.limit]

    plan = []
    for i, f in enumerate(frags):
        prompt = RD.prompt_for(f, frags, a.mode)
        par = "-" if f["parent"] is None else f"region {f['parent']:02d}"
        plan.append({"i": i, "name": f["name"], "parent": f["parent"],
                     "depth": f.get("depth", 0),
                     "projection": RD.mode_for(f, a.mode), "prompt": prompt})
        print(f"[{i:02d}] depth {f.get('depth', 0)}  parent {par:10s} "
              f"{RD.mode_for(f, a.mode):7s}  {prompt}")
    nested = sum(1 for f in frags if f["parent"] is not None)
    print(f"\n{len(frags)} regions, {nested} nested")
    if a.dry_run:
        return 0

    out = a.out or svg.parent / f"drawing_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    refs = []
    if a.refs and a.refs.exists():
        refs = sorted(p for p in a.refs.glob("*")
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[:3]
        print(f"style references: {', '.join(p.name for p in refs)}")

    backend = RB.build(negative=RD.negative_for(a.mode),
                       style_refs=[str(p) for p in refs], model=a.model,
                       ref_mode=a.ref_mode if refs else "none",
                       mono=a.mono, ink_clean=a.ink_clean, mode=a.mode,
                       legacy=a.legacy)
    print(f"backend: {backend.model}, {len(backend.style_refs)} reference images\n")

    canvas_path = out / "canvas.png"
    if canvas_path.exists():
        canvas = Image.open(canvas_path).convert("RGB")
        print("resuming from the existing canvas")
    else:
        canvas = RD.base_sheet(frags, a.size, a.base, svg.with_suffix(".png"))
        canvas.save(out / "00_base.png")

    t0, done, failed = time.time(), 0, []
    for i, f in enumerate(frags):
        dst = out / f"{i + 1:02d}_{f['name'][:44]}.png"
        if dst.exists():
            canvas = Image.open(dst).convert("RGB")
            continue
        for attempt in range(a.retries):
            try:
                canvas = backend.fill(canvas, f["ring"], plan[i]["prompt"],
                                      size=a.size, feather=a.feather,
                                      mode=plan[i]["projection"])
                break
            except Exception as ex:
                wait = 5 * (attempt + 1)
                print(f"   region {i:02d} attempt {attempt + 1} failed "
                      f"({type(ex).__name__}: {str(ex)[:90]}), retrying in {wait}s",
                      flush=True)
                time.sleep(wait)
        else:
            failed.append(i)
            continue
        canvas.save(dst)
        canvas.save(canvas_path)
        done += 1
        print(f"   [{i + 1}/{len(frags)}] {f['name'][:46]}  "
              f"({(time.time() - t0) / 60:.1f}m)", flush=True)

    canvas.save(out / "final.png")
    if a.finish:
        try:
            backend.restyle(canvas).save(out / "final_restyled.png")
            done += 1
            print("   unifying pass -> final_restyled.png", flush=True)
        except Exception as ex:
            print(f"   unifying pass failed: {type(ex).__name__}: {str(ex)[:110]}")

    (out / "run.json").write_text(json.dumps({
        "svg": str(svg), "model": backend.model, "mode": a.mode,
        "ref_mode": a.ref_mode if refs else "none", "mono": a.mono,
        "ink_clean": a.ink_clean, "legacy_v3": a.legacy, "base": a.base,
        "size": a.size, "regions": len(frags), "calls": done,
        "failed": failed, "style_refs": [p.name for p in refs],
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2), encoding="utf-8")
    print(f"\n{done} calls in {(time.time() - t0) / 60:.1f}m -> {out}"
          + (f"\n{len(failed)} regions failed: {failed}" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
