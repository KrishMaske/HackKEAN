"""
Dedicated KFC Bucket Masking — Per-frame bbox + GrabCut shape extraction.

Shot boundaries and bucket positions verified from grid overlays:
  Scene cuts at frames: 49, 122, 172, 242
  
  Shot A (0-48):   Wide dinner table — bucket center, x=520-730 y=290-520
  Shot B (49-121): Close-up boy      — bucket lower,  x=460-830 y=520-710
  Shot C (122-171): Parents eating   — bucket center,  x=390-720 y=490-710
  Shot D (172-241): Boy again        — bucket lower,  x=460-830 y=520-710
  Shot E (242-300): Parents/wide     — bucket left,   x=340-700 y=470-680
"""

import cv2
import numpy as np
import os


# Hardcoded per-shot search regions (generous bboxes)
SHOT_MAP = [
    (0,   48,  [520, 290, 730, 520]),   # Wide dinner table
    (49,  121, [460, 520, 830, 710]),   # Close-up boy
    (122, 171, [390, 490, 720, 710]),   # Parents eating
    (172, 241, [460, 520, 830, 710]),   # Boy again
    (242, 300, [340, 470, 700, 680]),   # Parents/wide
]


def get_bbox_for_frame(idx):
    for start, end, bbox in SHOT_MAP:
        if start <= idx <= end:
            return bbox
    return SHOT_MAP[-1][2]  # fallback


def mask_bucket_in_frame(frame, bbox):
    """GrabCut within bbox, then keep white+red pixels (bucket signature)."""
    x0, y0, x1, y1 = bbox
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 10 or bh <= 10:
        return np.zeros((h, w), dtype=np.uint8)

    # GrabCut
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, mask, (x0, y0, bw, bh), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except:
        return np.zeros((h, w), dtype=np.uint8)
    gc_fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # Color mask: white + red + orange + cream (KFC bucket under warm light)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, np.array([0, 0, 130]),   np.array([180, 65, 255]))
    red1   = cv2.inRange(hsv, np.array([0, 40, 40]),    np.array([15, 255, 255]))
    red2   = cv2.inRange(hsv, np.array([165, 40, 40]),  np.array([180, 255, 255]))
    orange = cv2.inRange(hsv, np.array([10, 60, 60]),   np.array([25, 255, 255]))
    cream  = cv2.inRange(hsv, np.array([15, 15, 110]),  np.array([40, 110, 255]))
    color  = white | red1 | red2 | orange | cream

    # Intersect
    combined = cv2.bitwise_and(gc_fg, color)

    # Clip to bbox
    clip = np.zeros((h, w), dtype=np.uint8)
    clip[y0:y1, x0:x1] = 255
    combined = cv2.bitwise_and(combined, clip)

    # Close gaps between red stripes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Largest blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 1:
        return cv2.bitwise_and(gc_fg, clip)
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[best, cv2.CC_STAT_AREA] < 400:
        return cv2.bitwise_and(gc_fg, clip)
    blob = np.where(labels == best, 255, 0).astype(np.uint8)

    # Convex hull for clean bucket shape
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(out, hull, 255)
        return cv2.bitwise_and(out, clip)
    return blob


def run():
    video = "assets/input/STRANGER_THINGS_CLIP.mp4"
    alpha_dir = "assets/masks/stranger_things_83/alpha"
    os.makedirs(alpha_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    total = len(frames)
    print(f"Loaded {total} frames @ {fps:.1f} fps")

    for idx in range(total):
        bbox = get_bbox_for_frame(idx)
        m = mask_bucket_in_frame(frames[idx], bbox)
        cv2.imwrite(os.path.join(alpha_dir, f"frame_{idx:04d}.png"), m)
        if idx % 30 == 0:
            print(f"  [{idx}/{total}] bbox={bbox} pixels={int(m.sum()/255)}")

    print(f"✅ {total} masks saved")

    # Preview
    prev = "assets/masks/stranger_things_83_preview.mp4"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(prev, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for idx in range(total):
        mp = os.path.join(alpha_dir, f"frame_{idx:04d}.png")
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        vis = frames[idx].copy()
        if mask is not None and mask.sum() > 0:
            green = np.zeros_like(vis); green[:,:,1] = 255
            vis[mask > 0] = cv2.addWeighted(vis, 0.5, green, 0.5, 0)[mask > 0]
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        out.write(vis)
    out.release()
    print(f"✅ Preview: {prev}")


if __name__ == "__main__":
    run()
