"""
Guard against multi-object conditioning images.

TRELLIS assumes the input shows ONE object. When SDXL returns a collage — which
it did on a car prompt, producing four overlapping vehicles — TRELLIS faithfully
fuses them into a single mangled mesh. That failure is silent: the pipeline
reports success and every geometry metric looks fine.

So: inspect the rembg alpha mask, count separate blobs, and either reject the
image (so the caller can reroll the seed) or keep only the dominant subject.
"""

import numpy as np
from PIL import Image


def log(msg):
    print(f"[subject] {msg}", flush=True)


def _label(mask):
    """Connected components. Uses scipy when present, else a BFS fallback."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, n
    except Exception:
        pass

    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    cur = 0
    # Iterative flood fill; a recursive one would blow the stack at 1024².
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or lab[sy, sx]:
                continue
            cur += 1
            stack = [(sy, sx)]
            lab[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
    return lab, cur


def analyze(img_rgba, min_frac=0.005, rival_frac=0.15, downscale=256):
    """
    Returns (is_single, info). `rival_frac` is how large the second blob may be
    relative to the largest before we call the image multi-object.
    """
    a = np.asarray(img_rgba.convert("RGBA"))
    alpha = a[:, :, 3]

    # Work small — labeling at full res is slow and adds nothing.
    small = np.asarray(
        Image.fromarray(alpha).resize((downscale, downscale), Image.NEAREST)
    )
    mask = small > 128
    total = mask.sum()
    if total == 0:
        return False, {"reason": "empty mask", "blobs": 0, "coverage": 0.0}

    lab, n = _label(mask)
    areas = np.array([(lab == i).sum() for i in range(1, n + 1)])
    keep = areas >= max(total * min_frac, 4)   # ignore speckle
    areas = np.sort(areas[keep])[::-1]

    coverage = float(total) / mask.size
    if len(areas) == 0:
        return False, {"reason": "no significant blob", "blobs": 0, "coverage": coverage}

    ratio = float(areas[1] / areas[0]) if len(areas) > 1 else 0.0
    single = len(areas) == 1 or ratio < rival_frac

    # Connected components alone are NOT sufficient: overlapping duplicates form
    # ONE blob. A real collage of four cars scored single=True with 89% coverage.
    # A well-framed product shot leaves margin; near-total coverage means the
    # subject was tiled or duplicated across the frame.
    # Connected components alone are NOT sufficient: overlapping duplicates form
    # ONE blob (a real 4-car collage scored single=True). High coverage is a hint
    # that the subject was tiled across the frame.
    #
    # NOT used to reject, deliberately. Tested against a known-good mug image it
    # also fired (0.87), so the threshold is not yet trustworthy on real rembg
    # masks and would cause false rerolls. Logged for calibration only — promote
    # to a rejection once validated against a corpus of real masks.
    high_coverage = coverage > 0.78
    if high_coverage:
        log(f"NOTE coverage {coverage:.2f} is high; possible duplicated subject "
            f"(advisory only, not rejecting)")
    info = {
        "high_coverage": bool(high_coverage),
        "blobs": int(len(areas)),
        "largest_frac": float(areas[0] / total),
        "second_ratio": round(ratio, 3),
        "coverage": round(coverage, 3),
    }
    return single, info


def isolate_main(img_rgba, downscale=256):
    """Zero out everything except the largest connected blob."""
    a = np.asarray(img_rgba.convert("RGBA")).copy()
    alpha = a[:, :, 3]
    h, w = alpha.shape

    small = np.asarray(Image.fromarray(alpha).resize((downscale, downscale), Image.NEAREST))
    mask = small > 128
    lab, n = _label(mask)
    if n <= 1:
        return img_rgba

    areas = [((lab == i).sum(), i) for i in range(1, n + 1)]
    areas.sort(reverse=True)
    keep_id = areas[0][1]

    keep_small = (lab == keep_id).astype(np.uint8) * 255
    keep_full = np.asarray(
        Image.fromarray(keep_small).resize((w, h), Image.NEAREST)
    )
    a[:, :, 3] = np.where(keep_full > 128, alpha, 0)
    log(f"isolated largest of {n} blobs")
    return Image.fromarray(a, mode="RGBA")
