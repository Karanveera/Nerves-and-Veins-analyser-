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


def _largest_blob(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    return m


def _clean(m, ks=9):
    m = cv2.medianBlur(m, 9)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return _largest_blob(m)


def _skin_mask(bgr):
    """Foreground (skin) mask so background / edges aren't detected as veins.
    Works for light AND dark skin and for dim, colour-cast camera images."""
    h, w = bgr.shape[:2]
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    Cr = ycc[:, :, 1]
    # 1) hue/saturation + Cr based (handles dark skin and warm or green camera casts)
    m = ((Cr >= 129) & ((H <= 32) | (H >= 170)) & (S >= 15) & (V >= 40)).astype(np.uint8) * 255
    m = _clean(m)
    if (m > 0).mean() >= 0.04:
        return m
    # 2) classic YCrCb range
    m = cv2.inRange(ycc, (40, 133, 70), (255, 185, 140))
    m &= (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 35).astype(np.uint8) * 255
    m = _clean(m, 7)
    if (m > 0).mean() >= 0.04:
        return m
    # 3) give up on colour: use the central part of the picture
    m = np.zeros((h, w), np.uint8)
    m[int(0.08 * h):int(0.92 * h), int(0.08 * w):int(0.92 * w)] = 255
    return m



def enhance_image(bgr, skin=None):
    """Make a dim / blurry / noisy camera frame clearer: denoise, brighten dark skin,
    boost local contrast and sharpen. Returns a natural-looking colour image."""
    den = cv2.bilateralFilter(bgr, 7, 18, 5)           # smooth noise, keep edges
    den = cv2.bilateralFilter(den, 7, 18, 5)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    # lift dark skin towards a comfortable brightness (gamma), measured on skin only
    ref = L[skin > 0] if skin is not None and (skin > 0).sum() > 200 else L.ravel()
    m = float(np.median(ref)) / 255.0
    if 0.02 < m < 0.48:
        gamma = np.log(0.48) / np.log(m)
        gamma = float(np.clip(gamma, 0.45, 1.0))
        L = (255.0 * (L / 255.0) ** gamma).astype(np.uint8)
    L = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(L)
    lab[:, :, 0] = L
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 3.0)          # unsharp mask
    out = cv2.addWeighted(out, 1.8, blur, -0.8, 0)
    return out


def finger_mask(skin):
    """Mask of the fingers, so they are never searched for veins.
    1) The palm/arm is thick and fingers are thin: opening with a big disc removes the thin parts.
    2) The knuckle line is found from where the fingers begin; everything beyond it is removed too,
       so finger bases and the ring area are skipped as well."""
    sk = (skin > 0).astype(np.uint8)
    dist = cv2.distanceTransform(sk, cv2.DIST_L2, 5)
    maxd = float(dist.max())
    if maxd < 12:
        return None
    r = int(0.5 * maxd)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    body = cv2.morphologyEx(sk, cv2.MORPH_OPEN, k)
    thin = ((sk > 0) & (body == 0)).astype(np.uint8)
    thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if thin.sum() < 0.01 * sk.sum():
        return None
    cores = (thin > 0) & (dist >= 0.1 * maxd)          # middle of the fingers (not thin rims)
    if cores.sum() < 50 or body.sum() == 0:
        cores = thin > 0
    by, bx = [float(v.mean()) for v in np.where(body > 0)]
    fy, fx = [float(v.mean()) for v in np.where(cores)]
    d = np.array([fx - bx, fy - by])
    nrm = np.linalg.norm(d)
    if nrm < 1:
        return None
    d /= nrm
    yy, xx = np.mgrid[0:sk.shape[0], 0:sk.shape[1]]
    t = (xx - bx) * d[0] + (yy - by) * d[1]
    t_cut = np.percentile(t[cores], 5) - 0.08 * maxd      # knuckle line
    beyond = (t > t_cut).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    total = float(sk.sum())
    for cand, limit in (((thin | beyond), 0.75), (thin, 0.6)):   # safety: never throw away most of the skin
        cand = cv2.dilate(cand.astype(np.uint8), ker) > 0
        if (cand & (sk > 0)).sum() <= limit * total:
            return cand
    return None


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


def _trace(coords_set, start):
    """Order the pixels of a skeleton branch starting from an endpoint."""
    path, seen, cur = [start], {start}, start
    while True:
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (cur[0] + dy, cur[1] + dx)
                if q != cur and q in coords_set and q not in seen:
                    nxt = q; break
            if nxt: break
        if nxt is None:
            return path
        path.append(nxt); seen.add(nxt); cur = nxt


def _link_branches(skel, min_piece=12, min_final=45, gap=34, max_turn=40):
    """Split skeleton at junctions, then chain branches that point at each other
    (end-to-end within `gap` px and roughly collinear) into long continuous veins."""
    sk = skel.astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junc = (sk > 0) & (nb >= 3)
    junc_d = cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lab = label(skel & ~junc_d, connectivity=2)

    pieces = []  # (ordered path as array of (y,x))
    for r in regionprops(lab):
        if r.area < min_piece:
            continue
        cs = set(map(tuple, r.coords))
        ends = [p for p in cs if sum(((p[0]+a, p[1]+b) in cs) for a in (-1,0,1) for b in (-1,0,1)) - 1 <= 1]
        if not ends:
            continue
        path = np.array(_trace(cs, ends[0]), float)
        if len(path) >= min_piece:
            pieces.append(path)

    n = len(pieces)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def end_dir(path, head):
        k = min(10, len(path) - 1)
        v = (path[0] - path[k]) if head else (path[-1] - path[-1 - k])  # pointing outward
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    ends = []  # (piece, point, outward direction)
    for idx, p in enumerate(pieces):
        ends.append((idx, p[0], end_dir(p, True)))
        ends.append((idx, p[-1], end_dir(p, False)))

    links = []
    cosmax = np.cos(np.radians(max_turn))
    cand = []
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            ia, pa, da = ends[a]; ib, pb, db = ends[b]
            if ia == ib:
                continue
            vec = pb - pa
            d = np.linalg.norm(vec)
            if d > gap or d == 0:
                continue
            u = vec / d
            if da @ u > cosmax and db @ (-u) > cosmax:   # both ends point at each other
                cand.append((d, a, b))
    used = set()
    for d, a, b in sorted(cand):
        if a in used or b in used:
            continue
        ia, ib = ends[a][0], ends[b][0]
        if find(ia) == find(ib):
            continue
        parent[find(ia)] = find(ib)
        used.update((a, b))
        links.append((ends[a][1], ends[b][1]))

    groups = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    out = np.zeros(skel.shape, np.uint8)
    for g in groups.values():
        pts = np.vstack([pieces[i] for i in g])
        length = sum(len(pieces[i]) for i in g)
        h0, w0 = pts.min(0); h1, w1 = pts.max(0)
        span = np.hypot(h1 - h0, w1 - w0)
        if length + 0 < min_final or span / length < 0.5:
            continue
        mask_g = np.zeros(skel.shape, np.uint8)
        for i in g:
            mask_g[pieces[i][:, 0].astype(int), pieces[i][:, 1].astype(int)] = 1
        for (pa, pb) in links:
            if any(np.any(np.all(pieces[i] == pa, 1)) for i in g):
                cv2.line(mask_g, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), 1, 1)
        out |= mask_g
    return out.astype(bool)

def detect_veins(bgr, sensitivity=0.8, min_length=35, scale=1.0, ignore=None, hand=None, hand_frac=0.28,
                 skin=None, with_response=False):
    """Return (skeleton_mask, skin_mask). sensitivity in [0,1]; higher = more veins."""
    h, w = bgr.shape[:2]
    if skin is None:
        skin = _skin_mask(bgr)

    # Veins appear darker/bluer-green than skin -> use green channel + CLAHE
    g = bgr[:, :, 1]
    # stretch the contrast of the skin area only (dim / low-contrast cameras)
    sv = g[skin > 0]
    if sv.size > 100:
        lo_, hi_ = np.percentile(sv, (2, 98))
        g = np.clip((g.astype(np.float32) - lo_) * 255.0 / max(hi_ - lo_, 1), 0, 255).astype(np.uint8)
    g = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (0, 0), 1.2 * scale)
    # remove broad shading (muscle form) -> keep thin structures
    bg = cv2.GaussianBlur(g, (0, 0), 7 * scale)
    detail = cv2.subtract(bg, g).astype(np.float32)  # dark ridges positive
    detail = np.clip(detail, 0, None)

    sigmas = [s * scale for s in (1.5, 2.0, 2.5, 3.0, 4.0)]
    v = frangi(detail, sigmas=sigmas, black_ridges=False, beta=0.5, gamma=None)

    # keep only inside eroded skin (avoid silhouette edge)
    er = int(np.clip(0.07 * np.sqrt((skin > 0).sum()) * max(1.0, 0.8 * scale), 9, 61)) | 1   # smaller skin area -> smaller margin
    inner = cv2.erode(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
    mgn = 10
    inner[:mgn, :] = 0; inner[-mgn:, :] = 0; inner[:, :mgn] = 0; inner[:, -mgn:] = 0  # ignore picture border
    if hand == "auto":
        fm = finger_mask(skin)                      # remove the fingers
        if fm is not None:
            inner[fm] = 0
        else:
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
    hsv_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    shiny = ((hsv_[:, :, 2] > 215) & (hsv_[:, :, 1] < 70)).astype(np.uint8)   # ring, glare, nails
    inner[cv2.dilate(shiny, np.ones((15, 15), np.uint8)) > 0] = 0
    v = v * (inner > 0)
    if v.max() == 0:
        return (np.zeros((h, w), bool), skin, v) if with_response else (np.zeros((h, w), bool), skin)
    v /= v.max()

    # hysteresis threshold controlled by sensitivity
    hi = np.percentile(v[inner > 0], 100 - 4 * sensitivity - 1)
    lo = hi * 0.35
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

    out = _link_branches(skel, min_piece=12, min_final=min_length, gap=int(40 * scale)) & (inner > 0)
    return (out, skin, v) if with_response else (out, skin)


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


def _keep_best(skel, resp, max_veins):
    """Keep the most vein-like lines: long, continuous and with a strong response.
    Weak lines (noise, skin texture) are dropped."""
    lab = label(skel, connectivity=2)
    regs = regionprops(lab)
    if not regs:
        return skel
    info = []
    for r in regs:
        cc = r.coords
        strength = float(resp[cc[:, 0], cc[:, 1]].mean())
        info.append((r.area * strength, strength, r))
    best = max(i[1] for i in info)
    info = [i for i in info if i[1] >= 0.35 * best]          # drop weak responses
    info.sort(key=lambda i: -i[0])
    out = np.zeros_like(skel, dtype=bool)
    for _, _, r in info[:max_veins or len(info)]:
        out[lab == r.label] = True
    return out


def process(bgr, sensitivity=0.8, min_length=35, thickness=2, label_veins=True, ignore=None,
            hand=None, hand_frac=0.28, max_veins=12, return_enhanced=False):
    # consistent working resolution (also enlarge tiny camera frames)
    h, w = bgr.shape[:2]
    s = 800 / max(h, w)
    if s < 1:
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    elif s > 1.15:
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    else:
        small = bgr.copy()

    # small / low-resolution pictures are blurrier -> look for wider lines (and ignore JPEG blocks)
    scale = float(np.clip(800.0 / max(h, w), 1.0, 2.5))
    skin = _skin_mask(small)                 # find the skin on the original colours
    clear = enhance_image(small, skin)       # clearer picture -> veins are searched here

    skel = resp = None
    # try the requested settings first; if nothing is found, relax them step by step
    for sens, ml in ((sensitivity, min_length), (1.0, min_length * 0.7), (1.0, min_length * 0.45)):
        skel, _, resp = detect_veins(clear, sens, max(12, int(ml)), scale=scale, ignore=ignore, hand=hand,
                                     hand_frac=hand_frac, skin=skin, with_response=True)
        if skel.any():
            break
    skel = _keep_best(skel, resp, max_veins)
    marked, n = draw_veins(clear, skel, thickness, label_veins)
    if return_enhanced:
        return clear, marked, n
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
