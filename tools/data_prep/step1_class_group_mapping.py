"""
Step 1: Create class_group_mapping.csv.

Prefer the prepared class_distribution.csv because it was computed on the
3972 matched train samples. If that file is missing, fall back to scanning
train masks, but only count masks that have a matching image.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

BACKGROUND_ID = 103
NUM_CLASSES = 104
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

BASE = Path(__file__).resolve().parents[2]
DATASET = BASE / "datasets" / "foodseg103-full"
OUT = BASE / "processed"


def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_class_names():
    with open(DATASET / "class_mapping.json", "r", encoding="utf-8") as f:
        cmap = json.load(f)
    id_to_name = {int(k): v for k, v in cmap["id_to_class"].items()}
    id_to_name[BACKGROUND_ID] = "background"
    return id_to_name


def scan_train_presence():
    id_to_name = load_class_names()
    presence = np.zeros(NUM_CLASSES, dtype=np.int64)

    mask_dir = DATASET / "train" / "mask"
    img_dir = DATASET / "train" / "img"
    mask_files = sorted(mask_dir.glob("*.png"))
    matched_masks = [mf for mf in mask_files if find_image(img_dir, mf.stem) is not None]
    skipped = len(mask_files) - len(matched_masks)

    print(
        f"[Step 1] class_distribution.csv not found. "
        f"Scanning {len(matched_masks)} matched train masks..."
    )
    if skipped:
        print(f"  Skipping {skipped} masks without matching image.")

    for mf in tqdm(matched_masks, desc="Counting classes"):
        m = np.array(Image.open(mf), dtype=np.uint8)
        for cid in np.unique(m):
            if 0 <= cid < NUM_CLASSES:
                presence[cid] += 1

    total = max(1, len(matched_masks))
    rows = []
    for cid in range(NUM_CLASSES):
        rows.append({
            "class_id": cid,
            "class_name": id_to_name.get(cid, f"class_{cid}"),
            "presence_count": int(presence[cid]),
            "presence_ratio": float(presence[cid]) / total,
        })
    return pd.DataFrame(rows)


def build_class_group_mapping():
    (OUT / "mappings").mkdir(parents=True, exist_ok=True)

    dist_path = DATASET / "class_distribution.csv"
    if dist_path.exists():
        df = pd.read_csv(dist_path)
        print(f"[Step 1] Loaded class distribution -> {dist_path}")
        required = {"class_id", "class_name", "presence_count"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{dist_path} missing columns: {sorted(missing)}")
        if len(df) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} classes in {dist_path}, got {len(df)}")
    else:
        df = scan_train_presence()

    def assign_group(row):
        cid = int(row["class_id"])
        pc = int(row["presence_count"])
        if cid == BACKGROUND_ID:
            return "background"
        if pc >= 500:
            return "head"
        if pc >= 50:
            return "medium"
        if pc >= 10:
            return "tail"
        return "extreme_tail"

    df["class_group"] = df.apply(assign_group, axis=1)

    group_mult = {
        "background": 0.0,
        "head": 1.0,
        "medium": 1.5,
        "tail": 3.0,
        "extreme_tail": 6.0,
    }
    df["class_sampling_mult"] = df["class_group"].map(group_mult)

    out_path = OUT / "mappings" / "class_group_mapping.csv"
    df.to_csv(out_path, index=False)
    print(f"[Step 1] Saved -> {out_path}")
    print(f"  Groups: {df['class_group'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    build_class_group_mapping()
