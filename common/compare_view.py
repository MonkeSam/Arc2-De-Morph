"""
compare_view.py - generate one comparison image (grid + labels) for each morph entry.

Dependencies: pillow   (pip install pillow)

Usage:
    from compare_view import export_sheets

    paths = [{
        "morph_stem": stem, "pose": pose, "morphed": ..., "criminal": ...,
        "accomplice": ..., "demorphed_criminal": ..., "demorphed_accomplice": ...,
    }, ...]
    export_sheets(paths, interp_mode="encoded-slerp", out_dir="comparisons")
    # -> comparisons/encoded-slerp/001_<morph_stem>_<pose>.png, ...
"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Entry keys shown in each sheet, in display order (the key name is used as label)
IMAGE_KEYS = ("criminal", "demorphed_criminal", "morphed", "demorphed_accomplice", "accomplice")


def _load(img):
    return img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")


def _font(size):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def make_contact_sheet(images, labels=None, cols=4, cell=384, pad=10,
                       bg=(24, 24, 24), out=None, title=None):
    """Build a grid of images with labels. If `out` is given, save it as a PNG."""
    imgs = [_load(i) for i in images]
    labels = labels or [str(i) for i in range(len(imgs))]
    rows = -(-len(imgs) // cols)  # ceil division
    label_h = 28
    top = 40 if title else 0

    sheet = Image.new("RGB", (cols * cell + (cols + 1) * pad,
                              top + rows * (cell + label_h) + (rows + 1) * pad), bg)
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad, 10), str(title), fill=(255, 255, 255), font=_font(22))

    font = _font(16)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell + pad)
        y0 = top + pad + r * (cell + label_h + pad)
        im = im.copy()
        im.thumbnail((cell, cell), Image.LANCZOS)  # keep aspect ratio
        sheet.paste(im, (x0 + (cell - im.width) // 2, y0))
        draw.text((x0 + 4, y0 + cell + 4), str(lab), fill=(230, 230, 230), font=font)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
    return sheet


def export_sheets(entries, interp_mode, out_dir="comparisons", cols=5, cell=320, keys=IMAGE_KEYS):
    """
    Generate one numbered PNG (001_..., 002_...) for each entry.
    Sheets are saved in out_dir/<interp_mode>/ (e.g. arcface-slerp or encoded-slerp),
    so each interpolation mode gets its own folder.

    entries : list of dicts with "morph_stem", "pose" and one image (path or PIL image)
              for each name in `keys`
    """
    out_dir = Path(out_dir) / interp_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for n, entry in enumerate(entries, start=1):
        name = f"{entry['morph_stem']}_{entry['pose']}"
        slug = re.sub(r"[^\w\-]+", "_", name).strip("_")[:80] or "entry"
        f = out_dir / f"{n:03d}_{slug}.png"
        make_contact_sheet(
            images=[entry[k] for k in keys],
            labels=[k.replace("_", " ") for k in keys],
            cols=cols, cell=cell, out=f,
            title=f"{entry['morph_stem']} | pose: {entry['pose']}",
        )
        created.append(f)
    return created
