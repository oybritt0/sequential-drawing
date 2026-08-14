r"""
backend.py — the region filler, against Gemini's image models.

Both backends answer the same question: given the canvas, one region's mask and
that region's prompt, return a new canvas. Everything above them --  the curated
curves, the nesting, the tag-built prompts -- is unchanged.

They get there differently, because Gemini's image models do not take a binary
mask; region selection there is by text description, which cannot express an
arbitrary closed polygon. So the Gemini backend does the masking itself:

    crop to the region's bounding box  ->  generate  ->  composite back through
    the polygon, feathered

That keeps the geometry exactly as authored rather than trusting the model to
respect it, and has a side benefit -- a small fragment gets the model's full
resolution on its own crop instead of a few hundred pixels of a shared canvas.

There is no LoRA here. The nearest equivalent is passing up to three drawings as
style references, and `--ref-mode patches` crops those to texture tiles so their
motifs cannot be copied wholesale -- a whole reference sheet leaked a distinctive
plan element verbatim before that was added.

    GEMINI_API_KEY=...  python -m sequential_drawing.run composition.svg
"""
from __future__ import annotations

import io
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

GEMINI_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")

# "Nano Banana" is Google's own nickname for the Gemini image models, so accept
# it as a name rather than making anyone look up which release maps to which id
NANO_BANANA = {
    "nano-banana":      "gemini-2.5-flash-image",   # Aug 2025, legacy
    "nano-banana-2":    "gemini-3.1-flash-image",   # Feb 2026, the workhorse
    "nano-banana-pro":  "gemini-3-pro-image",       # Nov 2025, best at complex work
    "nano-banana-lite": "gemini-3.1-flash-lite-image",
}


def resolve_model(name: str | None) -> str:
    if not name:
        return GEMINI_MODEL
    return NANO_BANANA.get(name.strip().lower().replace(" ", "-"), name)


PROJECTION = {
    "plan": (
        "STRICTLY a horizontal cut seen from directly overhead, like a floor "
        "plan. No axonometric, no isometric, no oblique or sheared view, no "
        "section, no elevation, no depth or shading that implies a 3D view. "
        "Every region of this sheet must read at the same scale and in the "
        "same projection."),
    "mixed": (
        "This sheet deliberately mixes projections: some regions are horizontal "
        "cuts seen from overhead and others are vertical cuts through floors and "
        "walls. Keep each region in the projection it was drawn in -- do NOT "
        "convert them to a single one. Unify only the line hand, the weight and "
        "the paper. No axonometric, isometric or perspective anywhere."),
    "section": (
        "STRICTLY a vertical cut through the building seen straight on, like an "
        "architectural section. Floor levels stack one above another against a "
        "consistent ground line, and the cut passes through floors, walls and "
        "roofs. What lies beyond the cut is drawn in lighter line. No plan, no "
        "axonometric, no isometric, no perspective. Every region of this sheet "
        "must read at the same scale, on the same levels, and in the same "
        "projection."),
}


# ------------------------------------------------------------------ helpers ---

def ring_bbox(ring, size, pad=0.10):
    """Pixel bbox of a region, padded, clamped, and squared off.

    Padding gives the model context beyond the region so the edit meets what
    surrounds it; squaring avoids the aspect distortion a thin sliver would
    otherwise get when the result is resized back.
    """
    xs, ys = ring[:, 0] * size, ring[:, 1] * size
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    m = max(w, h) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    m = min(m, size)
    x0 = int(round(min(max(cx - m / 2, 0), size - m)))
    y0 = int(round(min(max(cy - m / 2, 0), size - m)))
    m = int(round(m))
    return x0, y0, x0 + m, y0 + m


def texture_patches(path, n=2, frac=0.16, grid=6):
    """Small dense crops of a reference, instead of the whole drawing.

    A whole sheet carries its motifs as well as its hand, and the model copies
    both -- a distinctive folded ramp in one of the references came back almost
    verbatim. Cropping to patches at texture scale keeps the mark-making and
    line weight while leaving nothing large enough to recognise. Patches are
    picked by ink density so they are not blank paper, and spread across the
    sheet so they do not all come from one motif.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = int(min(w, h) * frac)
    if side < 32:
        return [img]
    scored = []
    for gy in range(grid):
        for gx in range(grid):
            x = int(gx * (w - side) / max(grid - 1, 1))
            y = int(gy * (h - side) / max(grid - 1, 1))
            tile = img.crop((x, y, x + side, y + side))
            a = np.asarray(tile.convert("L"), dtype=np.float32)
            ink = float((a < 200).mean())
            # mid-density reads as drawing; near-1 is a solid fill, near-0 is paper
            scored.append((abs(ink - 0.35), gx, gy, tile))
    scored.sort(key=lambda t: t[0])
    picked, used = [], set()
    for _, gx, gy, tile in scored:
        if any(abs(gx - ux) + abs(gy - uy) < 2 for ux, uy in used):
            continue
        picked.append(tile)
        used.add((gx, gy))
        if len(picked) >= n:
            break
    return picked or [scored[0][3]]


def to_mono(img):
    """Neutral grey, paper pushed to white.

    The models return a warm cream paper. Left alone it compounds: each region is
    generated from a canvas already tinted by the last, so the cast deepens across
    a run. Normalising every returned region keeps the sheet neutral end to end,
    which no prompt wording reliably achieves.
    """
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=(1, 1))
    return g.convert("RGB")


def clean_ink(img, lift=0.55, floor=0.22):
    """Lift mid-grey tone toward white, leave the black line work alone.

    The models answer "poche" with dense crosshatch and solid fill. This keeps
    anything darker than `floor` as line, sends anything lighter than `lift` to
    paper, and ramps between -- so hatching thins out while the drawn lines stay
    black. Prompt wording alone did not hold this.
    """
    g = np.asarray(ImageOps.grayscale(img), dtype=np.float32) / 255.0
    lo, hi = floor, lift
    out = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    out = out ** 0.7                      # gentle, so light hatch fades first
    return Image.fromarray((out * 255).astype(np.uint8)).convert("RGB")


def polygon_mask(ring, size, box, feather):
    """The region's mask in crop coordinates."""
    x0, y0, x1, y1 = box
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(m).polygon(
        [(float(x * size - x0), float(y * size - y0)) for x, y in ring], fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather)) if feather else m


# ----------------------------------------------------------------- Gemini ----

class GeminiBackend:
    """Crop, generate, composite. The mask never leaves this process."""

    name = "gemini"

    def __init__(self, negative, model=GEMINI_MODEL, style_refs=None, pad=0.10,
                 api_key=None, ref_mode="patches", patches_per_ref=2, mono=True, ink_clean=False, mode="plan"):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY") or \
            os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit(
                "no API key. Set it in your shell, do not paste it into a file:\n"
                '    $env:GEMINI_API_KEY = "..."      # PowerShell\n'
                '    export GEMINI_API_KEY=...        # bash')
        self.client = genai.Client(api_key=key)
        self.model = resolve_model(model)
        self.negative, self.pad = negative, pad
        refs = list(style_refs or [])
        if ref_mode == "patches":
            tiles = []
            for r in refs:
                tiles += texture_patches(r, n=patches_per_ref)
            self.style_refs = tiles[:3]
        elif ref_mode == "none":
            self.style_refs = []
        else:
            self.style_refs = [Image.open(r).convert("RGB") for r in refs][:3]
        self.ref_mode = ref_mode
        self.mono = mono
        self.ink_clean = ink_clean
        self.mode = mode if mode in PROJECTION else "plan"
        self.calls = 0

    def _instruction(self, prompt, mode=None):
        # Gemini edits by description, so the region has to be named in words as
        # well as cropped; "only within" holds the edit off the surrounding pad.
        return (
            "The black outline marks ONE region of a much larger drawing. Fill "
            "ONLY inside that outline. The faded area around it belongs to other "
            "regions -- leave it alone.\n\n"
            "Do NOT compose for this square crop and do NOT centre a symmetrical "
            "or circular figure in it. This region is a fragment of a bigger "
            "drawing and should run off its own edges as if continuing.\n\n"
            f"Draw: {prompt}\n\n"
            f"Avoid: {self.negative}\n\n"
            "Pure black ink line work on WHITE paper. Greyscale only -- no colour, "
            "no cream, sepia, yellow or warm paper tint.\n\n"
            "LINE CONVENTIONS, follow exactly:\n"
            "- Walls are drawn as TWO thin parallel black lines with the space "
            "between them left WHITE. Never fill a wall solid black.\n"
            "- Uniform thin line weight throughout. No thick, heavy or variable "
            "strokes, no fat outlines around the shape.\n"
            "- No crosshatching, no dense tonal shading, no large areas of solid "
            "black. Most of the interior stays white paper.\n"
            "- Delicate, precise draughting: the drawing is made of lines, not "
            "of fills.\n"
            "- NO text anywhere: no room names, labels, titles, dimensions or "
            "annotations. The drawing carries no lettering at all."
        )

    _REF_RULE = ("Use these ONLY for line weight, hatch density and mark-making. "
                 "Do NOT copy any shape, motif, symbol or plan element from them. "
                 "Invent the forms. Reproduce NO text of any kind -- the "
                 "references contain room names and titles and none of them may "
                 "appear in the output.")

    _FINISH = (
        "Redraw this entire architectural plan in ONE consistent drawing hand.\n\n"
        "Keep the composition exactly as it is: every shape, boundary, room, "
        "wall position and plan element stays where it is and the same size. "
        "Do not add, remove, move or invent anything. Change only HOW it is "
        "drawn.\n\n"
        "Draw it as: pure black ink line work on white paper. Walls are TWO thin "
        "parallel black lines with WHITE between them, never solid black. Uniform "
        "thin line weight throughout, no thick or heavy strokes, no fat outlines. "
        "No crosshatching, no tonal shading, no large areas of solid black or "
        "grey. Most of the sheet stays white paper.\n\n"
        "The regions were drawn separately and read inconsistently. Unify them "
        "into a single coherent drawing, at one scale, in one projection.\n\n"
        "Remove any lettering, room names, labels or titles that appear; the "
        "finished drawing carries no text at all."
    )

    def restyle(self, canvas, extra=""):
        """One whole-frame pass over the finished sheet, to unify the hand.

        The regions are generated independently, so line weight and convention
        drift between them. Sending the assembled sheet back through unmasked
        settles it in one go, which no per-region wording achieved.
        """
        parts = [self._FINISH + "\n\n" + PROJECTION[self.mode]
                 + (f"\n\n{extra}" if extra else "")]
        if self.style_refs:
            parts.append(self._REF_RULE)
            parts += self.style_refs
        parts.append(canvas)
        resp = self.client.models.generate_content(model=self.model, contents=parts)
        img = self._first_image(resp)
        self.calls += 1
        if img is None:
            raise RuntimeError("restyle returned no image")
        img = img.resize(canvas.size, Image.LANCZOS)
        return to_mono(img) if self.mono else img

    @staticmethod
    def _first_image(resp):
        for cand in getattr(resp, "candidates", []) or []:
            for part in getattr(cand.content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return Image.open(io.BytesIO(inline.data)).convert("RGB")
        return None

    def _mark_crop(self, canvas, ring, size, box):
        """Send the crop with the target region made unmistakable.

        Saying "the shape at the centre" leaves the model to guess the extent, so
        it composes for the whole square crop -- which is why so many regions came
        back centred on a rotunda. Dimming everything outside the polygon and
        drawing its boundary states the area exactly; the region is often less
        than half the crop, so the guess was rarely close.
        """
        crop = canvas.crop(box)
        m = polygon_mask(ring, size, box, 0)
        faded = Image.blend(crop, Image.new("RGB", crop.size, (255, 255, 255)), 0.78)
        out = Image.composite(crop, faded, m)
        d = ImageDraw.Draw(out)
        pts = [(float(x * size - box[0]), float(y * size - box[1])) for x, y in ring]
        d.line(pts + [pts[0]], fill=(0, 0, 0), width=2)
        return out

    def fill(self, canvas, ring, prompt, *, size, feather=3.0, mode=None, **_):
        box = ring_bbox(ring, size, self.pad)
        crop = self._mark_crop(canvas, ring, size, box)
        parts = [self._instruction(prompt, mode)]
        if self.style_refs:
            parts.append(self._REF_RULE)
            parts += self.style_refs
        parts.append(crop)

        resp = self.client.models.generate_content(model=self.model, contents=parts)
        img = self._first_image(resp)
        self.calls += 1
        if img is None:
            fb = getattr(resp, "prompt_feedback", None)
            raise RuntimeError(f"no image returned (feedback: {fb})")

        img = img.resize(crop.size, Image.LANCZOS)
        if self.mono:
            img = to_mono(img)
        if self.ink_clean:
            img = clean_ink(img)
        out = canvas.copy()
        out.paste(img, box[:2], polygon_mask(ring, size, box, feather))
        return out


def build(*, negative, style_refs=None, model=None, ref_mode="patches",
          mono=True, ink_clean=False, mode="plan", api_key=None):
    return GeminiBackend(negative, resolve_model(model), style_refs,
                         ref_mode=ref_mode, mono=mono, ink_clean=ink_clean,
                         mode=mode, api_key=api_key)
