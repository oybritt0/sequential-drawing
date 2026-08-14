r"""
prompt_map.py — which region got which prompt.

Draws the curated regions numbered over a result (or over the base sheet), with a
keyed legend giving each region's period, its four authored facets, and the
parent it inherited context from. Writes the same key as CSV and Markdown so it
can travel with the image.

    python -m sequential_drawing.prompt_map composition.svg --image out/final.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon

from . import geometry as RD

CMAP = plt.get_cmap("tab20").colors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("composition", type=Path)
    ap.add_argument("--image", type=Path, default=None,
                    help="result to underlay; defaults to the base sheet")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N regions, to match a run that was cut short")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    svg = (a.composition if a.composition.suffix.lower() == ".svg"
           else a.composition / "composition.svg")
    frags = RD.nest(RD.parse_curated(svg))
    shown = frags[:a.limit] if a.limit else frags
    out = a.out or (a.image.parent if a.image else a.composition)
    out.mkdir(parents=True, exist_ok=True)

    if a.image and a.image.exists():
        under = Image.open(a.image).convert("RGB")
    else:
        under = RD.base_sheet(frags, 1024, "full", svg.with_suffix(".png"))
    S = under.size[0]

    fig = plt.figure(figsize=(19, 9.5), facecolor="white")
    ax = fig.add_axes([0.01, 0.03, 0.44, 0.92])
    ax.imshow(under, extent=[0, 1, 1, 0])
    for i, f in enumerate(frags):
        live = f in shown
        c = np.array(CMAP[i % len(CMAP)])
        ring = np.vstack([f["ring"], f["ring"][:1]])
        ax.plot(ring[:, 0], ring[:, 1], color=c, lw=2.2 if live else 0.8,
                alpha=1.0 if live else 0.35)
        if live:
            ax.fill(ring[:, 0], ring[:, 1], color=c, alpha=0.13)
            cx, cy = Polygon(f["ring"]).representative_point().coords[0]
            ax.text(cx, cy, str(i), fontsize=13, weight="bold", color="white",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.28", fc=c, ec="white", lw=1.4))
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_axis_off()
    ax.set_title(f"{svg.stem}\n{len(shown)} of {len(frags)} regions drawn",
                 fontsize=10)

    ax = fig.add_axes([0.46, 0.03, 0.53, 0.92])
    ax.set_axis_off()
    y, dy = 0.985, 0.985 / max(len(shown), 1)
    dy = min(dy, 0.115)
    for i, f in enumerate(frags):
        if f not in shown:
            continue
        c = np.array(CMAP[i % len(CMAP)])
        par = "—" if f["parent"] is None else f"inherits from region {f['parent']}"
        ax.add_patch(plt.Rectangle((0.0, y - 0.030), 0.022, 0.028,
                                   color=c, transform=ax.transAxes, clip_on=False))
        ax.text(0.030, y - 0.016, f"{i}", fontsize=12, weight="bold",
                va="center", transform=ax.transAxes)
        ax.text(0.062, y - 0.004, f"{f['period']}   ·   depth {f.get('depth',0)}   "
                f"·   {par}", fontsize=8.5, color="#444", va="center",
                transform=ax.transAxes)
        ax.text(0.062, y - 0.030,
                f"{f['material']}  |  {f['organization']}", fontsize=9.5,
                va="center", transform=ax.transAxes)
        ax.text(0.062, y - 0.052,
                f"{f['gestalt']}  |  {f['surface']}", fontsize=9.5, color="#333",
                va="center", transform=ax.transAxes)
        y -= dy
    ax.set_title("region  →  the terms its prompt was built from", fontsize=10,
                 loc="left")

    dst = out / "prompt_map.png"
    fig.savefig(dst, dpi=115, facecolor="white")
    print(dst)

    with open(out / "prompt_map.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "drawn", "depth", "parent", "work_id", "period",
                    "material", "organization", "gestalt", "surface", "prompt"])
        for i, f in enumerate(frags):
            w.writerow([i, f in shown, f.get("depth", 0),
                        "" if f["parent"] is None else f["parent"], f["work_id"],
                        f["period"], f["material"], f["organization"],
                        f["gestalt"], f["surface"], RD.prompt_for(f, frags)])
    lines = [f"# {a.composition.name} — region key", ""]
    for i, f in enumerate(frags):
        if f not in shown:
            continue
        par = "" if f["parent"] is None else f" (inherits from {f['parent']})"
        lines += [f"**{i}** — {f['period']}, depth {f.get('depth',0)}{par}  ",
                  f"`{RD.prompt_for(f, frags)}`", ""]
    (out / "prompt_map.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "prompt_map.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
