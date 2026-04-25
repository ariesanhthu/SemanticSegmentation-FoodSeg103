"""
Step 2: Tạo sample_training_mapping.csv.
Scan từng mask để tính fg_ratio, present_classes, rare_score, difficulty, weights, aug_policy.
"""
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BACKGROUND_ID = 103
NUM_CLASSES = 104
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

BASE = Path(__file__).resolve().parents[2]
DATASET = BASE / "datasets" / "foodseg103-full"
OUT = BASE / "processed"


def find_image(img_dir, stem):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_sample_training_mapping():
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    class_map = pd.read_csv(OUT / "mappings" / "class_group_mapping.csv")
    id_to_name = dict(zip(class_map["class_id"], class_map["class_name"]))
    id_to_group = dict(zip(class_map["class_id"], class_map["class_group"]))
    id_to_presence = dict(zip(class_map["class_id"], class_map["presence_count"]))

    rows = []
    for split in ["train", "test"]:
        mask_dir = DATASET / split / "mask"
        img_dir = DATASET / split / "img"
        mask_files = sorted(mask_dir.glob("*.png"))
        print(f"[Step 2] Processing {split}: {len(mask_files)} masks...")

        for mf in tqdm(mask_files, desc=f"  {split}"):
            stem = mf.stem
            img_path = find_image(img_dir, stem)
            if img_path is None:
                continue

            mask = np.array(Image.open(mf), dtype=np.uint8)
            h, w = mask.shape
            unique_ids = [int(c) for c in np.unique(mask) if 0 <= c < NUM_CLASSES]
            fg_ids = [c for c in unique_ids if c != BACKGROUND_ID]

            fg_pixels = int(np.sum(mask != BACKGROUND_ID))
            fg_ratio = fg_pixels / max(1, h * w)

            # Connected components (all foreground)
            fg_binary = (mask != BACKGROUND_ID).astype(np.uint8)
            num_cc, _, _, _ = cv2.connectedComponentsWithStats(fg_binary, connectivity=8)
            num_cc = max(0, num_cc - 1)  # subtract background component

            # Groups
            groups = [id_to_group.get(c, "unknown") for c in unique_ids]
            names = [id_to_name.get(c, f"class_{c}") for c in unique_ids]
            has_tail = any(id_to_group.get(c) == "tail" for c in unique_ids)
            has_extreme = any(id_to_group.get(c) == "extreme_tail" for c in unique_ids)

            # Rare score
            rare_score = 0.0
            for c in fg_ids:
                pc = max(1, id_to_presence.get(c, 1))
                rare_score += 1.0 / np.sqrt(pc)

            rows.append({
                "stem": stem,
                "split": split,
                "image_path": str(img_path),
                "mask_path": str(mf),
                "height": h,
                "width": w,
                "fg_ratio": round(fg_ratio, 4),
                "num_present_classes": len(unique_ids),
                "num_connected_components": num_cc,
                "present_classes": unique_ids,
                "present_class_names": names,
                "present_class_groups": groups,
                "has_tail": has_tail,
                "has_extreme_tail": has_extreme,
                "rare_score": round(rare_score, 4),
            })

    df = pd.DataFrame(rows)

    # Hard score GT
    df["hard_score_gt"] = (
        2.0 * (1.0 - df["fg_ratio"])
        + 0.15 * df["num_connected_components"].clip(upper=50)
        + 0.25 * df["num_present_classes"]
        + 0.8 * df["has_tail"].astype(float)
        + 1.2 * df["has_extreme_tail"].astype(float)
    )

    # Difficulty levels (based on train quantiles)
    train_scores = df.loc[df["split"] == "train", "hard_score_gt"]
    q40 = train_scores.quantile(0.40)
    q75 = train_scores.quantile(0.75)
    df["difficulty_level"] = df["hard_score_gt"].apply(
        lambda s: "easy" if s <= q40 else ("medium" if s <= q75 else "hard")
    )

    # Stage weights
    s1_mult = {"easy": 1.5, "medium": 1.0, "hard": 0.4}
    s2_mult = {"easy": 0.4, "medium": 1.0, "hard": 2.0}
    df["stage1_weight"] = 1.0 + df["rare_score"] + df["difficulty_level"].map(s1_mult)
    df["stage2_weight"] = (
        1.0 + 2.0 * df["rare_score"]
        + df["difficulty_level"].map(s2_mult)
        + 1.5 * df["has_extreme_tail"].astype(float)
    )

    # Aug policy
    def assign_aug_policy(row):
        if row["split"] != "train":
            return "none"
        if row["has_extreme_tail"] and row["difficulty_level"] == "hard":
            return "mix"
        if row["has_extreme_tail"]:
            return "rare_copy_paste"
        if row["has_tail"]:
            return "rare_crop"
        if row["difficulty_level"] == "hard":
            return "hard_crop"
        if row["difficulty_level"] == "medium":
            return "light"
        return "none"

    df["aug_policy"] = df.apply(assign_aug_policy, axis=1)

    out_path = OUT / "metadata" / "sample_training_mapping.csv"
    df.to_csv(out_path, index=False)
    print(f"[Step 2] Saved → {out_path}")
    print(f"  Total samples: {len(df)} (train={len(df[df['split']=='train'])}, test={len(df[df['split']=='test'])})")
    print(f"  Difficulty: {df[df['split']=='train']['difficulty_level'].value_counts().to_dict()}")
    print(f"  Aug policy: {df[df['split']=='train']['aug_policy'].value_counts().to_dict()}")
    return df


# Need cv2 import at top
import cv2

if __name__ == "__main__":
    build_sample_training_mapping()
