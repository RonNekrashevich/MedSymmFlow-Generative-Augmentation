"""Preprocess the APTOS 2019 Kaggle dataset into a small npz for MSF training.

APTOS 2019 (aptos2019-blindness-detection) carries the same 5-grade DR labels
as RetinaMNIST but comes from a different source (Aravind Eye Hospital), so it
can serve as an external generator corpus with zero benchmark contamination.

Run LOCALLY after downloading from Kaggle:
    python project/preprocess_aptos.py --csv train.csv --images train_images \
        --out aptos_28.npz --size 28

Output npz: images uint8 (N, size, size, 3), labels int64 (N,) in 0..4.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def center_square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv with id_code,diagnosis")
    ap.add_argument("--images", required=True, help="directory of <id_code>.png files")
    ap.add_argument("--out", default="aptos_28.npz")
    ap.add_argument("--size", type=int, default=28)
    args = ap.parse_args()

    import csv
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    imgs, labels, missing = [], [], 0
    for i, r in enumerate(rows):
        p = Path(args.images) / f"{r['id_code']}.png"
        if not p.exists():
            missing += 1
            continue
        img = Image.open(p).convert("RGB")
        img = center_square(img).resize((args.size, args.size), Image.LANCZOS)
        imgs.append(np.asarray(img, dtype=np.uint8))
        labels.append(int(r["diagnosis"]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")
    images = np.stack(imgs)
    labels = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(args.out, images=images, labels=labels)
    counts = np.bincount(labels, minlength=5).tolist()
    print(f"wrote {args.out}: {len(labels)} images {images.shape[1:]},"
          f" per grade {counts}, missing files {missing}")


if __name__ == "__main__":
    main()
