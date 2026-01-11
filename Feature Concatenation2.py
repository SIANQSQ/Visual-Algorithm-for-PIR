import os, glob, math, argparse
import numpy as np
import cv2 as cv

def imread_all(img_dir):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.JPG","*.PNG")
    paths = sorted(sum([glob.glob(os.path.join(img_dir, e)) for e in exts], []))
    imgs = []
    for p in paths:
        im = cv.imread(p, cv.IMREAD_COLOR)
        if im is None: continue
        imgs.append((os.path.basename(p), im))
    if len(imgs) < 2:
        raise SystemExit("Need >=2 images in dir: %s" % img_dir)
    return imgs

def detect_and_match(img1, img2, method="sift"):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    if method == "sift":
        feat = cv.SIFT_create()
    elif method == "orb":
        feat = cv.ORB_create(nfeatures=5000)
    else:
        raise ValueError("Unknown feature method")
    k1, d1 = feat.detectAndCompute(gray1, None)
    k2, d2 = feat.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1)<4 or len(k2)<4: return None
    if method == "sift":
        idx_params = dict(algorithm=1, trees=8)  # KDTree
        search_params = dict(checks=64)
        matcher = cv.FlannBasedMatcher(idx_params, search_params)
        d1f = d1; d2f = d2
    else:
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        d1f = d1; d2f = d2
    knn = matcher.knnMatch(d1f, d2f, k=2)
    good = []
    for m,n in knn:
        if m.distance < 0.75*n.distance:
            good.append(m)
    if len(good) < 4: return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv.findHomography(src, dst, cv.RANSAC, 3.0, maxIters=5000, confidence=0.999)
    if H is None: return None
    inliers = int(mask.ravel().sum())
    return H, good, inliers, len(good)

def compose_chain_homographies(images, method="sift"):
    # Simple left-to-right chain. For unordered set, you can pick central ref by max degree,
    # but here we keep deterministic behavior.
    n = len(images)
    H_to_ref = [np.eye(3, dtype=np.float64) for _ in range(n)]
    for i in range(1, n):
        H, matches, inl, tot = detect_and_match(images[i][1], images[i-1][1], method)
        if H is None:
            raise SystemExit("Failed to match images %d and %d" % (i-1, i))
        H_to_ref[i] = H_to_ref[i-1] @ H  # map i -> i-1 -> ... -> 0
    return H_to_ref  # each maps image i to ref 0

def compute_canvas_bounds(images, H_to_ref):
    corners = []
    for (name, im), H in zip(images, H_to_ref):
        h,w = im.shape[:2]
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        wrap = cv.perspectiveTransform(pts, H)
        corners.append(wrap.reshape(-1,2))
    allpts = np.vstack(corners)
    min_x, min_y = np.floor(allpts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(allpts.max(axis=0)).astype(int)
    tx, ty = -min_x, -min_y
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    W, Hh = int(max_x-min_x), int(max_y-min_y)
    return T, (W, Hh)

def mask_from_image(img):
    # 1 where valid pixels exist, else 0
    m = np.ones(img.shape[:2], dtype=np.uint8)*255
    return m

def distance_transform_weight(mask):
    # Prefer seams far from borders; higher weight where far from boundary
    inv = cv.bitwise_not(mask)
    dist = cv.distanceTransform(mask, cv.DIST_L2, 3).astype(np.float32)
    dist = dist / (dist.max() + 1e-6)
    return dist

def graphcut_seam_select(imgA, imgB, maskA, maskB):
    # Use GrabCut-like GC on overlapping region to select source. We build an 8-bit 3-channel image as data term.
    overlap = cv.bitwise_and(maskA>0, maskB>0)
    if overlap.sum() == 0:
        # no overlap, trivial
        selectA = maskA>0
        selectB = maskB>0
        return selectA.astype(np.uint8)*255, selectB.astype(np.uint8)*255

    # Build trimap for cv.grabCut: 0 bg, 1 fg, 2 prob bg, 3 prob fg
    # We'll run twice: once assuming A is FG, once B is FG, then pick better energy by simple boundary length heuristic.
    def run_gc(fg_img, bg_img, fg_mask, bg_mask):
        # Region of interest: bounding box of overlap
        ys, xs = np.where(overlap)
        y0,y1 = ys.min(), ys.max()+1
        x0,x1 = xs.min(), xs.max()+1
        roi = (x0,y0,x1-x0,y1-y0)
        trimap = np.zeros(fg_mask.shape, dtype=np.uint8)
        # sure FG/BG
        trimap[fg_mask>0] = 3
        trimap[bg_mask>0] = 2
        # Outside both gets bg
        trimap[np.logical_not(np.logical_or(fg_mask>0, bg_mask>0))] = 0

        # Prepare data for grabCut
        comp = np.clip(fg_img.astype(np.int16)-bg_img.astype(np.int16), -255, 255).astype(np.int16)
        comp = np.abs(comp).astype(np.uint8)
        # Use composite image as data term
        img_gc = cv.cvtColor(comp, cv.COLOR_BGR2RGB)

        mask_gc = np.where(trimap==3, cv.GC_FGD,
                  np.where(trimap==2, cv.GC_PR_BGD,
                  np.where(trimap==0, cv.GC_BGD, cv.GC_PR_FGD))).astype(np.uint8)

        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv.grabCut(img_gc, mask_gc, roi, bgdModel, fgdModel, 2, cv.GC_INIT_WITH_MASK)
        seg = np.where((mask_gc==cv.GC_FGD)|(mask_gc==cv.GC_PR_FGD), 1, 0).astype(np.uint8)
        # Keep only where fg or bg existed
        seg = seg * ((fg_mask>0)|(bg_mask>0)).astype(np.uint8)
        return seg

    segA = run_gc(imgA, imgB, maskA, maskB)
    segB = run_gc(imgB, imgA, maskB, maskA)
    # Choose the one that yields shorter seam length (proxy for simpler seam)
    def seam_len(seg):
        edge = cv.Canny((seg*255).astype(np.uint8), 50, 150)
        return int(edge.sum())
    if seam_len(segA) <= seam_len(segB):
        selA = segA
        selB = ((maskA>0)|(maskB>0)).astype(np.uint8) - selA
    else:
        selB = segB
        selA = ((maskA>0)|(maskB>0)).astype(np.uint8) - selB
    return (selA*255).astype(np.uint8), (selB*255).astype(np.uint8)

def multiband_blend(dst, src, mask):
    # Use OpenCV detail.MultiBandBlender for high quality blending
    h,w = dst.shape[:2]
    blender = cv.detail_MultiBandBlender()
    # levels auto: based on image size
    blender.setNumBands(5)
    blender.prepare((0,0,w,h))
    dst32 = dst.astype(np.float32)
    src32 = src.astype(np.float32)
    blender.feed(src32, mask, (0,0))
    blender.feed(dst32, np.where(cv.cvtColor(dst, cv.COLOR_BGR2GRAY)>0, 255, 0).astype(np.uint8), (0,0))
    result, result_mask = blender.blend(None, None)
    out = np.clip(result, 0, 255).astype(np.uint8)
    return out

def stitch_nonredundant(img_dir, feature="sift", blend=True, out="panorama.png", debug=False):
    images = imread_all(img_dir)  # [(name,img)]
    # 1) global mapping
    H_to_ref = compose_chain_homographies(images, method=feature)
    # 2) canvas
    T, (W, Hh) = compute_canvas_bounds(images, H_to_ref)
    H_to_canvas = [T @ H for H in H_to_ref]
    # 3) warp and build masks
    warped = []
    masks = []
    for (name, im), H in zip(images, H_to_canvas):
        pano = cv.warpPerspective(im, H, (W, Hh))
        mask = cv.warpPerspective(mask_from_image(im), H, (W, Hh))
        warped.append(pano)
        masks.append(mask)
    # 4) sequentially merge with graph-cut seam to avoid duplicates
    accum = np.zeros((Hh, W, 3), dtype=np.uint8)
    accum_mask = np.zeros((Hh, W), dtype=np.uint8)

    for idx, (imw, msk) in enumerate(zip(warped, masks)):
        if accum_mask.sum() == 0:
            accum[:] = imw
            accum_mask[:] = msk
            continue
        # overlap selection
        selA, selB = graphcut_seam_select(accum, imw, accum_mask, msk)
        # compose masks
        keepA = selA>0
        keepB = selB>0
        merged = accum.copy()
        merged[keepB] = imw[keepB]
        new_mask = ((accum_mask>0)|(msk>0)).astype(np.uint8)*255
        if blend:
            # Soft blend only near seam to reduce ghosting
            seam = cv.Canny((keepB.astype(np.uint8)*255), 0, 1)
            dil = cv.dilate(seam, np.ones((5,5), np.uint8), iterations=1)
            blend_zone = cv.dilate(dil, np.ones((7,7), np.uint8), iterations=1)>0
            alpha = np.zeros((Hh, W, 1), dtype=np.float32)
            alpha[blend_zone] = 0.5
            merged[blend_zone] = (accum[blend_zone].astype(np.float32)*(1-alpha[blend_zone]) + imw[blend_zone].astype(np.float32)*alpha[blend_zone]).astype(np.uint8)

        accum = merged
        accum_mask = new_mask

        if debug:
            cv.imwrite(f"debug_step_{idx:02d}.png", accum)
    # 5) optional multiband
    if blend:
        # Build final mask of accum
        final_mask = (accum_mask>0).astype(np.uint8)*255
        # Note: Using multiband on the whole can be heavy; we skip here because we already seam-selected.
        pass

    # 6) crop empty borders
    gray = cv.cvtColor(accum, cv.COLOR_BGR2GRAY)
    nz = cv.findNonZero((gray>0).astype(np.uint8))
    if nz is not None:
        x,y,w,h = cv.boundingRect(nz)
        accum = accum[y:y+h, x:x+w]

    cv.imwrite(out, accum)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="input image directory")
    ap.add_argument("--feature", default="sift", choices=["sift","orb"])
    ap.add_argument("--blend", action="store_true", help="enable soft blend around seams")
    ap.add_argument("--out", default="panorama_noredundant.png")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    path = stitch_nonredundant(args.img_dir, feature=args.feature, blend=args.blend, out=args.out, debug=args.debug)
    print("Saved:", path)

if __name__ == "__main__":
    main()