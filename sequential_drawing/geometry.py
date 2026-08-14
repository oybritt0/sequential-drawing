r"""
geometry.py — a tagged SVG composition read as ordered, nested drawing regions.

Parses the curated closed curves out of a composition SVG, works out which region
sits inside which, and builds one prompt per region from that region's own
semantic tags plus its parent's as inherited context.

No model code here. See backend.py for generation and run.py for the driver.

Each fragment already carries its period and its strongest term in each of the
four facets of verbal_vocab.json. This treats the curated closed curve as an
inpainting mask and gives that region its own prompt built from its own tags, so
the drawing is not one image described once but a set of regions each described
by the work it came from.

Order is sparse_fit's own draw order -- big behind, small in front -- so a
fragment sitting inside an earlier one is painted after it and reads as nested.
Where that containment is real (measured, not assumed) the child's prompt carries
its parent's material and organization as context, so an enclosure drawn in
drystone can hold a vault drawn in concrete and the two stay legible as a
relation rather than a collision.

    python -m sequential_drawing.run composition.svg --dry-run
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import Polygon

TRIGGER = ""          # set to a LoRA trigger word if you use one
NS = "{http://www.w3.org/2000/svg}"

_NEG_BASE = ("photograph, photorealistic, render, perspective view, axonometric, "
             "isometric, oblique or sheared view, people, sky, colour, cream or "
             "sepia paper, solid black fill, thick heavy outline, dense "
             "crosshatching, blurry, watermark, signature, text, lettering, "
             "room names, labels, titles, annotations")
# "section" and "elevation" belong in the negative only when drawing plans --
# left in for section or mixed they fight the very thing being asked for
NEGATIVE = _NEG_BASE + ", section, elevation"


def negative_for(mode="plan"):
    return _NEG_BASE if mode in ("section", "mixed") else NEGATIVE


def _ring(path_el, scale, samples_per_seg=1):
    """Points from any valid SVG path, not just the one this project writes.

    Opening a sheet in Illustrator or Inkscape and saving rewrites the geometry
    into compressed relative form -- l/h/v shorthand, no separators -- which a
    naive "pairs of numbers" reader silently mangles or crashes on. svgpathtools
    handles the whole grammar, so an edited sheet still round-trips.
    """
    d = path_el.get("d") or ""
    try:
        from svgpathtools import parse_path
        path = parse_path(d)
        pts = []
        for seg in path:
            n = samples_per_seg if seg.__class__.__name__ == "Line" else 8
            for k in range(n):
                z = seg.point(k / n)
                pts.append((z.real, z.imag))
        if pts:
            return np.array(pts) / scale
    except Exception:
        pass
    nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", d)]
    return np.array(nums[:len(nums) // 2 * 2]).reshape(-1, 2) / scale


def _reading_of(g):
    """Which layer a group belongs to, however the file has been re-saved.

    Illustrator keeps ids and labels but drops data-* attributes, so a sheet that
    has been edited by hand arrives with none of the metadata this pipeline
    writes. The id still encodes it -- curated__frag_07__period__material__gestalt
    -- so fall back to that rather than refusing the file.
    """
    r = g.get("data-reading")
    if r in ("curated", "true"):
        return r
    gid = g.get("id") or g.get("{http://www.inkscape.org/namespaces/inkscape}label") or ""
    for r in ("curated", "true"):
        if gid.startswith(r + "__"):
            return r
    return None


def _tags_from_id(gid):
    """period / material / gestalt out of the group id, when the attributes are gone."""
    parts = (gid or "").split("__")
    if len(parts) < 3:
        return {}
    return {"period": parts[2] if len(parts) > 2 else "",
            "material": (parts[3] if len(parts) > 3 else "").replace("-", " "),
            "gestalt": (parts[4] if len(parts) > 4 else "").replace("-", " ")}


def enrich_from(frags, source_dir: Path):
    """Fill organization / surface / work_id from the composition it came from."""
    import csv as _csv
    f = Path(source_dir) / "fragments.csv"
    if not f.exists():
        return frags
    rows = list(_csv.DictReader(open(f, encoding="utf-8")))
    by_name = {r["name"]: r for r in rows}
    # slug lengths changed between runs, so an older sheet's ids will not match
    # by name; frag_NN is stable because it is the draw order
    by_index = {r["name"].split("__", 1)[0]: r for r in rows}
    for fr in frags:
        key = fr["name"].split("__", 1)[-1]
        r = by_name.get(key) or by_index.get(key.split("__", 1)[0])
        if not r:
            continue
        for k in ("organization", "surface", "work_id"):
            if not fr.get(k):
                fr[k] = r[k]
        for k in ("period", "material", "gestalt"):
            fr[k] = r[k] or fr.get(k, "")
    return frags


def parse_curated(svg: Path):
    """Curated ring, the matching true ring, and every facet.

    The SVG carries both readings; the curated one drives the masks, but the
    true one is what actually describes the fragment's detail, so it is kept and
    drawn into the base rather than thrown away.
    """
    root = ET.parse(svg).getroot()
    vb = [float(v) for v in root.get("viewBox").split()]
    curated, true = [], {}
    for g in root.iter(NS + "g"):
        reading = _reading_of(g)
        if reading not in ("curated", "true"):
            continue
        p = g.find(NS + "path")
        if p is None:
            continue
        gid = (g.get("id") or "").split("__", 1)[-1]
        if reading == "true":
            true[gid] = _ring(p, vb[2])
            continue
        fallback = _tags_from_id(g.get("id"))
        curated.append({
            "name": g.get("id"),
            "key": gid,
            "ring": _ring(p, vb[2]),
            "work_id": g.get("data-work-id", ""),
            "period": ((g.get("data-period") or fallback.get("period", ""))
                       .split("_", 1)[-1]),
            "material": g.get("data-material") or fallback.get("material", ""),
            "organization": g.get("data-organization", ""),
            "gestalt": g.get("data-gestalt") or fallback.get("gestalt", ""),
            "surface": g.get("data-surface", ""),
        })
    for f in curated:
        f["true_ring"] = true.get(f["key"])
    return curated


def nest(frags, thresh=0.6):
    """Parent = the last earlier fragment that mostly contains this one.

    Draw order is already big-behind-small-in-front, so a real parent is always
    earlier in the list; `thresh` is the share of this fragment's area that has
    to fall inside the candidate before the relation counts.
    """
    polys = [Polygon(f["ring"]).buffer(0) for f in frags]
    for i, f in enumerate(frags):
        f["parent"] = None
        a = polys[i]
        if a.is_empty or a.area <= 0:
            continue
        best, best_cover = None, thresh
        for j in range(i):
            b = polys[j]
            if b.is_empty:
                continue
            cover = a.intersection(b).area / a.area
            # prefer the tightest container, not merely the first
            if cover >= best_cover and (best is None or b.area <= polys[best].area):
                best, best_cover = j, cover
        f["parent"] = best
        f["depth"] = 0 if best is None else frags[best].get("depth", 0) + 1
    return frags


# Terms that describe how a thing is stacked, spanned or cut read naturally in
# section; terms that describe how it is laid out read in plan. In "mixed" the
# fragment's own organization term picks its projection, so the choice comes from
# the authored vocabulary rather than from an arbitrary alternation.
SECTION_TERMS = {
    "layered stratification", "a dome over a drum", "vaulted enclosure",
    "post and lintel construction", "a load-bearing arch",
    "a cantilevered overhang",
}


def mode_for(f, mode):
    """Resolve "mixed" to plan or section for one fragment."""
    if mode != "mixed":
        return mode
    return "section" if f.get("organization") in SECTION_TERMS else "plan"


def prompt_for(f, frags, mode="plan"):
    """The region's own tags, with its parent's as inherited context."""
    kind = ("orthographic building section, cut through floors and walls"
            if mode_for(f, mode) == "section" else "orthographic floor plan")
    bits = [TRIGGER, "architectural drawing", kind,
            "ink line on paper", "hatching and poche"]
    if f["parent"] is not None:
        p = frags[f["parent"]]
        bits.append(f"set within {p['organization']} of {p['material']}")
    bits += [f["organization"], f["material"], f["gestalt"], f["surface"]]
    if f["period"]:
        bits.append(f["period"].replace("_", " "))
    return ", ".join(b for b in bits if b)


def mask_for(ring, size, feather):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon([(float(x * size), float(y * size)) for x, y in ring],
                              fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather else m


def base_sheet(frags, size, mode="full", source=None):
    """The starting sheet — the smoothed curated curves, not the traced detail.

    Hairline outlines alone mark about 2% of the frame while the regions they
    bound cover 45%, which leaves the model almost nothing to work from and reads
    as one tangle rather than distinct figures. "full" therefore backs the curated
    curve with a light tonal fill per region, graded by nesting depth so
    containment is legible in the base itself. The true high-resolution silhouette
    is deliberately NOT drawn: the gestural reading is the one being built on.
    """
    img = Image.new("RGB", (size, size), (252, 252, 252))
    if mode == "source" and source is not None and source.exists():
        return Image.open(source).convert("RGB").resize((size, size), Image.LANCZOS)

    def px(ring):
        return [(float(x * size), float(y * size)) for x, y in ring]

    if mode in ("full", "flat", "detail"):
        wash = Image.new("RGB", (size, size), (252, 252, 252))
        wd = ImageDraw.Draw(wash)
        for f in sorted(frags, key=lambda f: -Polygon(f["ring"]).area):
            v = max(214, 246 - 8 * int(f.get("depth", 0)))
            wd.polygon(px(f["ring"]), fill=(v, v, v))
        img = Image.blend(img, wash, 0.75)

    d = ImageDraw.Draw(img)
    if mode == "detail":                      # opt-in only
        for f in frags:
            if f.get("true_ring") is not None:
                d.line(px(f["true_ring"]) + [px(f["true_ring"])[0]],
                       fill=(120, 120, 126), width=1)
    if mode != "flat":
        for f in frags:
            d.line(px(f["ring"]) + [px(f["ring"])[0]], fill=(40, 40, 46), width=1)
    return img
