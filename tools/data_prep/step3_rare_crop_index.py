"""
Step 3: Tạo rare_crop_index.csv.
Scan train masks để tìm connected components của tail/extreme_tail classes,
lưu bbox chính xác cho crop-based augmentation.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BACKGROUND_ID = 103
BASE = Path(__file__).resolve().parents[2]
DATASET = BASE / "datasets" / "foodseg103-full"
OUT = BASE / "processed"


def build_rare_crop_index():
    (OUT / "aug_bank").mkdir(parents=True, exist_ok=True)

    sample_map = pd.read_csv(OUT / "metadata" / "sample_training_mapping.csv")
    class_map = pd.read_csv(OUT / "mappings" / "class_group_mapping.csv")

    id_to_name = dict(zip(class_map["class_id"], class_map["class_name"]))
    id_to_group = dict(zip(class_map["class_id"], class_map["class_group"]))
    rare_groups = {"tail", "extreme_tail"}

    train_df = sample_map[sample_map["split"] == "train"]
    # Only process images that have tail/extreme_tail
    candidates = train_df[train_df["has_tail"] | train_df["has_extreme_tail"]]
    print(f"[Step 3] Processing {len(candidates)} images with rare classes...")

    rows = []
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Rare crops"):
        mask = np.array(Image.open(row["mask_path"]), dtype=np.uint8)
        h, w = mask.shape

        for cid in np.unique(mask):
            cid = int(cid)
            if cid == BACKGROUND_ID:
                continue
            group = id_to_group.get(cid)
            if group not in rare_groups:
                continue

            binary = (mask == cid).astype(np.uint8)
            num_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            for comp_id in range(1, num_cc):
                x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                y = int(stats[comp_id, cv2.CC_STAT_TOP])
                bw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                bh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                area = int(stats[comp_id, cv2.CC_STAT_AREA])

                if area < 50:
                    continue

                area_ratio = area / max(1, h * w)
                crop_size = 512 if max(bw, bh) > 300 else 384

                rows.append({
                    "stem": row["stem"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "class_id": cid,
                    "class_name": id_to_name.get(cid, f"class_{cid}"),
                    "class_group": group,
                    "component_id": comp_id,
                    "x1": x, "y1": y,
                    "x2": x + bw, "y2": y + bh,
                    "area_pixels": area,
                    "area_ratio": round(area_ratio, 6),
                    "crop_size": crop_size,
                })

    df = pd.DataFrame(rows)
    out_path = OUT / "aug_bank" / "rare_crop_index.csv"
    df.to_csv(out_path, index=False)
    print(f"[Step 3] Saved -> {out_path}")
    print(f"  Total rare components: {len(df)}")
    if len(df) > 0:
        print(f"  By group: {df['class_group'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    build_rare_crop_index()
