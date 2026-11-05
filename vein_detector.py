"""Vein detection & marking for arm/hand photos (classical computer vision)."""
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops

PALETTE = [  # BGR colours, one per detected vein
    (0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (0, 0, 255), (255, 0, 128),
]


def _skin_mask(bgr):
    """Foreground (skin) mask so background / edges aren't detected as veins."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, (40, 133, 70), (255, 185, 140))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m &= (gray > 35).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    # keep largest blob
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    return m



def auto_hand_cut(skin):
    """Find the wrist (narrowest point near an end of the arm) and return a mask that is
    True for the hand/finger side. Returns None if no clear hand is found."""
    ys, xs = np.where(skin > 0)
    if len(xs) < 500:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    ax, perp = vt[0], vt[1]
    t = (pts - mean) @ ax
    p = (pts - mean) @ perp
    lo, hi = t.min(), t.max()
    nb = 60
    bins = np.minimum(((t - lo) / (hi - lo) * nb).astype(int), nb - 1)
    width = np.zeros(nb)
    for b in range(nb):
        pb = p[bins == b]
        width[b] = (pb.max() - pb.min()) if len(pb) > 3 else 0
    width = np.convolve(width, np.ones(5) / 5, mode="same")
    best = None
    for end in (0, 1):
        w = width if end == 0 else width[::-1]
        lim = int(0.45 * nb)
        seg = w[3:lim]
        if len(seg) == 0:
            continue
        k = int(np.argmin(seg)) + 3
        tip = w[:k].max()
        score = tip / max(w[k], 1e-3)
        if score > 1.15 and (best is None or score > best[0]):
            best = (score, end, k)
    if best is None:
        return None
    _, end, k = best
    cut_t = lo + (k + 1.5) / nb * (hi - lo) if end == 0 else hi - (k + 1.5) / nb * (hi - lo)
    full = np.zeros(skin.shape, np.float32)
    yy, xx = np.mgrid[0:skin.shape[0], 0:skin.shape[1]]
    tt = (np.stack([xx, yy], -1).astype(np.float32) - mean) @ ax
    return (tt <= cut_t) if end == 0 else (tt >= cut_t)

def detect_veins(bgr, sensitivity=0.7, min_length=35, scale=1.0, ignore=None, hand=None, hand_frac=0.28):
    """Return (skeleton_mask, skin_mask). sensitivity in [0,1]; higher = more veins."""
    h, w = bgr.shape[:2]
    skin = _skin_mask(bgr)

    # Veins appear darker/bluer-green than skin -> use green channel + CLAHE
    g = bgr[:, :, 1]
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    # remove broad shading (muscle form) -> keep thin structures
    bg = cv2.GaussianBlur(g, (0, 0), 6 * scale)
    detail = cv2.subtract(bg, g).astype(np.float32)  # dark ridges positive
    detail = np.clip(detail, 0, None)

    sigmas = [s * scale for s in (1.2, 1.6, 2.0, 2.5)]
    v = frangi(detail, sigmas=sigmas, black_ridges=False, beta=0.5, gamma=None)

    # keep only inside eroded skin (avoid silhouette edge)
    inner = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    if hand == "auto":
        cut = auto_hand_cut(skin)
        if cut is not None:
            inner[cut] = 0
    elif hand in ("top", "bottom", "left", "right"):  # skip fingers/fist at that end of the arm
        ys, xs = np.where(skin > 0)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if hand == "top":    inner[:y0 + int(hand_frac * (y1 - y0)), :] = 0
        if hand == "bottom": inner[y1 - int(hand_frac * (y1 - y0)):, :] = 0
        if hand == "left":   inner[:, :x0 + int(hand_frac * (x1 - x0))] = 0
        if hand == "right":  inner[:, x1 - int(hand_frac * (x1 - x0)):] = 0
    if ignore:  # (x0,y0,x1,y1) as fractions of image: region to skip (e.g. fingers)
        x0, y0, x1, y1 = ignore
        inner[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 0
    v = v * (inner > 0)
    if v.max() == 0:
        return np.zeros((h, w), bool), skin
    v /= v.max()

    # hysteresis threshold controlled by sensitivity
    hi = np.percentile(v[inner > 0], 100 - 4 * sensitivity - 1)
    lo = hi * 0.45
    strong, weak = v >= hi, v >= lo
    lab = label(weak, connectivity=2)
    keep = np.unique(lab[strong]); keep = keep[keep > 0]
    mask = np.isin(lab, keep)

    mask = remove_small_objects(mask, max_size=40)
    # bridge gaps: close along many orientations so collinear fragments join into long veins
    m8 = mask.astype(np.uint8)
    joined = m8.copy()
    L = max(9, int(11 * scale))
    for ang in range(0, 180, 30):
        k = np.zeros((L, L), np.uint8)
        c = L // 2
        dx, dy = np.cos(np.radians(ang)) * c, np.sin(np.radians(ang)) * c
        cv2.line(k, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1, 1)
        joined |= cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
    mask = joined.astype(bool) & (inner > 0)
    skel = skeletonize(mask)

    # split the skeleton at junctions into branches; keep long, smooth ones
    sk = skel.astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junc = (sk > 0) & (nb >= 3)
    junc_d = cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    branches = skel & ~junc_d
    lab = label(branches, connectivity=2)
    out = np.zeros_like(skel)
    for r in regionprops(lab):
        h0, w0, h1, w1 = r.bbox
        extent = np.hypot(h1 - h0, w1 - w0)
        if r.area >= min_length and extent / r.area > 0.45:   # long and fairly straight
            out[lab == r.label] = True
    # re-attach junction pixels touching kept branches so lines stay continuous
    touch = cv2.dilate(out.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out |= (skel & junc_d & touch)
    return out, skin


def draw_veins(bgr, skel, thickness=2, label_veins=True):
    """Overlay each vein as a distinct coloured line + numbered tag."""
    out = bgr.copy()
    lab = label(skel, connectivity=2)
    regions = sorted(regionprops(lab), key=lambda r: -r.area)
    for i, r in enumerate(regions):
        col = PALETTE[i % len(PALETTE)]
        pts = r.coords[:, ::-1]  # (x, y)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), max(1, thickness // 2), col, -1, cv2.LINE_AA)
        if label_veins:
            x, y = pts[len(pts) // 2]
            cv2.putText(out, f"V{i+1}", (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, f"V{i+1}", (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out, len(regions)


def process(bgr, sensitivity=0.7, min_length=35, thickness=2, label_veins=True, ignore=None, hand=None, hand_frac=0.28):
    # work at a consistent resolution for stable results
    h, w = bgr.shape[:2]
    s = 800 / max(h, w)
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    skel, _ = detect_veins(small, sensitivity, min_length, ignore=ignore, hand=hand, hand_frac=hand_frac)
    marked, n = draw_veins(small, skel, thickness, label_veins)
    return small, marked, n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("-o", "--out", default="veins_marked.png")
    ap.add_argument("-s", "--sensitivity", type=float, default=0.7)
    ap.add_argument("--ignore", type=float, nargs=4, metavar=("X0","Y0","X1","Y1"), help="fractions of image to skip, e.g. 0.5 0.1 1 0.4")
    ap.add_argument("--hand", choices=["auto","top","bottom","left","right"], help="side where fingers/fist are; skipped")
    ap.add_argument("--hand-frac", type=float, default=0.28)
    a = ap.parse_args()
    from PIL import Image
    img = cv2.cvtColor(np.array(Image.open(a.image).convert("RGB")), cv2.COLOR_RGB2BGR)
    _, marked, n = process(img, a.sensitivity, ignore=a.ignore, hand=a.hand, hand_frac=a.hand_frac)
    cv2.imwrite(a.out, marked); print(f"{n} veins marked -> {a.out}")
