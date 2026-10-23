"""Find FallVision duplicate videos / recording session groups -> data/raw/duplicates.csv

dup_group (for removal, only one keep=1 per group):
  - exact      : identical file md5
  - anonymized : X_anonymized(.._resized) in the same folder as the original (face-blurred copy)
session_group (for split, same group goes to same split):
  - mean dHash distance <= --session-dist. Fixed camera so background dominates the hash -> different takes
    from the same room/session get grouped. Groups regardless of label/scenario (prevents Fall/No Fall leakage within a session).
dHash: 64 bits for each of 5 probe frames per video. Signatures cached in data/raw/video_signatures.npz.
"""
import argparse
import csv
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

N_PROBES = 5


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def signature(path):
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bits = []
    for q in np.linspace(0.1, 0.9, N_PROBES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(q * max(n - 1, 0)))
        ok, frame = cap.read()
        bits.append(dhash(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if ok else np.zeros(64, bool))
    cap.release()
    return md5, n, np.concatenate(bits)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def labels(self):
        roots = [self.find(i) for i in range(len(self.p))]
        ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
        return np.array([ids[r] for r in roots])


def load_signatures(videos, root, cache, workers):
    rel = [str(p.relative_to(root)) for p in videos]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["rel"]) == rel:
            return list(z["md5"]), z["nframes"], z["hash"]
    with ProcessPoolExecutor(workers) as ex:
        sigs = list(ex.map(signature, videos, chunksize=16))
    md5 = [s[0] for s in sigs]
    nfr = np.array([s[1] for s in sigs])
    H = np.stack([s[2] for s in sigs]).astype(np.uint8)
    np.savez(cache, rel=np.array(rel), md5=np.array(md5), nframes=nfr, hash=H)
    return md5, nfr, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/FallVision")
    ap.add_argument("--out", default="data/raw/duplicates.csv")
    ap.add_argument("--cache", default="data/raw/video_signatures.npz")
    ap.add_argument("--session-dist", type=float, default=6.0, help="same-session threshold, mean hamming per frame")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    root = Path(args.root)
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".mp4")
    md5, nfr, H = load_signatures(videos, root, Path(args.cache), args.workers)

    dup, ses = UnionFind(len(videos)), UnionFind(len(videos))
    reason = [""] * len(videos)

    first = {}
    for i, h in enumerate(md5):
        if h in first:
            dup.union(i, first[h])
            reason[i] = "exact"
        first.setdefault(h, i)

    # X_anonymized / X_resized_anonymized / X_anonymized_resized -> same base name as original
    base = {}
    for i, p in enumerate(videos):
        key = (p.parent, re.sub(r"_(anonymized|resized)", "", p.stem))
        base.setdefault(key, []).append(i)
    for members in base.values():
        origs = [i for i in members if "_anonymized" not in videos[i].stem]
        for i in members:
            if origs and i not in origs:
                dup.union(i, origs[0])
                reason[i] = reason[i] or "anonymized"

    # Close frame hashes -> same session. Not used for copy detection: with a fixed camera and a small person,
    # different takes can also have hash distance <= 1 (visually checked, e.g. C_N_48 vs C_N_52)
    for a in range(len(videos)):
        d = (H[a + 1:] != H[a]).sum(1) / N_PROBES
        for off in np.where(d <= args.session_dist)[0]:
            ses.union(a, a + 1 + off)

    # dups must always be in the same session
    for i in range(len(videos)):
        ses.union(i, dup.find(i))
    dup_id, ses_id = dup.labels(), ses.labels()

    seen = set()
    rows = []
    for i, p in enumerate(videos):
        keep = int(dup_id[i] not in seen)
        seen.add(dup_id[i])
        rows.append([str(p.relative_to(root)), md5[i], int(nfr[i]), dup_id[i], ses_id[i], keep,
                     "" if keep else reason[i] or "group"])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "md5", "n_frames", "dup_group", "session_group", "keep", "drop_reason"])
        w.writerows(rows)

    labels = np.array([p.relative_to(root).parts[0] for p in videos])
    keep = np.array([r[5] for r in rows], bool)
    ses_sizes = np.bincount(ses_id[keep])
    mixed = sum(len(set(labels[(ses_id == s) & keep])) > 1 for s in np.unique(ses_id[keep]))
    drops = {k: sum(r[6] == k for r in rows) for k in ("exact", "anonymized", "group")}
    print(f"videos {len(videos)} -> keep {keep.sum()} (drop {drops})")
    print(f"sessions {len(ses_sizes[ses_sizes > 0])}, largest {np.sort(ses_sizes)[-5:][::-1].tolist()}, "
          f"singletons {(ses_sizes == 1).sum()}, sessions with both labels {mixed}")


if __name__ == "__main__":
    main()
