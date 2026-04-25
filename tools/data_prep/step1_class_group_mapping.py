"""
Step 1: Tạo class_group_mapping.csv từ raw masks.
Scan tất cả train masks để đếm presence_count, gán group và sampling multiplier.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BACKGROUND_ID = 103
NUM_CLASSES = 104

BASE = Path(__file__).resolve().parents[2]
DATASET = BASE / "datasets" / "foodseg103-full"
OUT = BASE / "processed"


def build_class_group_mapping():
    (OUT / "mappings").mkdir(parents=True, exist_ok=True)

    # Load class names
    with open(DATASET / "class_mapping.json", "r", encoding="utf-8") as f:
        cmap = json.load(f)
    id_to_name = {int(k): v for k, v in cmap["id_to_class"].items()}
    id_to_name[BACKGROUND_ID] = "background"

    # Count presence per class across train masks
    presence = np.zeros(NUM_CLASSES, dtype=np.int64)
    mask_dir = DATASET / "train" / "mask"
    mask_files = sorted(mask_dir.glob("*.png"))
    print(f"[Step 1] Scanning {len(mask_files)} train masks...")

    for mf in tqdm(mask_files, desc="Counting classes"):
        m = np.array(Image.open(mf), dtype=np.uint8)
        for cid in np.unique(m):
            if 0 <= cid < NUM_CLASSES:
                presence[cid] += 1

    # Build DataFrame
    rows = []
    for cid in range(NUM_CLASSES):
        rows.append({
            "class_id": cid,
            "class_name": id_to_name.get(cid, f"class_{cid}"),
            "presence_count": int(presence[cid]),
        })
    df = pd.DataFrame(rows)

    # Assign groups
    def assign_group(row):
        if int(row["class_id"]) == BACKGROUND_ID:
            return "background"
        pc = int(row["presence_count"])
        if pc >= 500:
            return "head"
        if pc >= 50:
            return "medium"
        if pc >= 10:
            return "tail"
        return "extreme_tail"

    df["class_group"] = df.apply(assign_group, axis=1)

    group_mult = {
        "background": 0.0, "head": 1.0, "medium": 1.5,
        "tail": 3.0, "extreme_tail": 6.0,
    }
    df["class_sampling_mult"] = df["class_group"].map(group_mult)

    out_path = OUT / "mappings" / "class_group_mapping.csv"
    df.to_csv(out_path, index=False)
    print(f"[Step 1] Saved → {out_path}")
    print(f"  Groups: {df['class_group'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    build_class_group_mapping()
