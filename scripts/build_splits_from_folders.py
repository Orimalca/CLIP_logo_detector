from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def list_images(folder: Path) -> List[Path]:
    out = []
    if not folder.exists():
        return out
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out.append(p.resolve())
    return sorted(out)

def norm_label(name: str) -> str:
    n = name.strip().lower()
    if n in {"mcdonalds", "mcdonald's", "mcdonald"}:
        return "mcdonalds"
    return n

def write_csv(rows: List[Tuple[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

def cap_per_class(rows: List[Tuple[str, str]], max_per_class: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    by = {}
    for p, lab in rows:
        by.setdefault(lab, []).append((p, lab))
    out = []
    for lab, items in by.items():
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="folder containing 0/ and 1/")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--tiny", action="store_true", help="cap each class for fast debugging")
    ap.add_argument("--tiny_max_per_class", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.dataset_root)

    rows = []

    pos_root = root / "1"
    # tolerate both "McDonalds" and "mcdonalds" folder names
    for cls in ["cocacola", "disney", "McDonalds", "mcdonalds", "starbucks"]:
        imgs = list_images(pos_root / cls)
        if len(imgs) == 0:
            continue
        lab = norm_label(cls)
        rows += [(str(p), lab) for p in imgs]

    neg_imgs = list_images(root / "0")
    rows += [(str(p), "background") for p in neg_imgs]

    if len(rows) == 0:
        raise RuntimeError("No images found. Check dataset_root and folder structure.")

    if args.tiny:
        rows = cap_per_class(rows, args.tiny_max_per_class, args.seed)

    X = [p for p, _ in rows]
    y = [lab for _, lab in rows]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=args.seed, stratify=y_trainval
    )

    out_dir = Path(args.out_dir)
    write_csv(list(zip(X_train, y_train)), out_dir / "train.csv")
    write_csv(list(zip(X_val, y_val)), out_dir / "val.csv")
    write_csv(list(zip(X_test, y_test)), out_dir / "test.csv")

    def counts(labels):
        c = {}
        for lab in labels:
            c[lab] = c.get(lab, 0) + 1
        return c

    print("Wrote:")
    print(" train:", counts(y_train))
    print(" val  :", counts(y_val))
    print(" test :", counts(y_test))

if __name__ == "__main__":
    main()
