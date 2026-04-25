"""
Step 6: Tạo augmentation_manifest.csv.
Kế hoạch augment cho từng ảnh dựa trên aug_policy, chưa tạo ảnh thật.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "processed"


def build_augmentation_manifest():
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    (OUT / "augmented" / "img").mkdir(parents=True, exist_ok=True)
    (OUT / "augmented" / "mask").mkdir(parents=True, exist_ok=True)

    sample_map = pd.read_csv(OUT / "metadata" / "sample_training_mapping.csv")
    train_df = sample_map[sample_map["split"] == "train"].copy()

    rng = np.random.default_rng(42)
    manifest = []

    # Policy → (n_aug, aug_types)
    policy_config = {
        "none": (0, []),
        "light": (1, ["light_color"]),
        "hard_crop": (2, ["hard_crop"]),
        "rare_crop": (3, ["rare_crop"]),
        "rare_copy_paste": (5, ["rare_crop", "copy_paste"]),
        "mix": (6, ["rare_crop", "hard_crop", "copy_paste"]),
    }

    print(f"[Step 6] Building augmentation manifest from {len(train_df)} train samples...")
    for _, row in train_df.iterrows():
        policy = row["aug_policy"]
        n_aug, aug_types = policy_config.get(policy, (0, []))

        if n_aug == 0 or not aug_types:
            continue

        for k in range(n_aug):
            aug_type = rng.choice(aug_types)
            aug_id = f"{row['stem']}_{aug_type}_{k:02d}"

            manifest.append({
                "aug_id": aug_id,
                "base_stem": row["stem"],
                "base_image_path": row["image_path"],
                "base_mask_path": row["mask_path"],
                "aug_type": aug_type,
                "source_patch_id": "",
                "source_class_id": "",
                "target_x": "",
                "target_y": "",
                "output_image_path": str(OUT / "augmented" / "img" / f"{aug_id}.png"),
                "output_mask_path": str(OUT / "augmented" / "mask" / f"{aug_id}.png"),
            })

    df = pd.DataFrame(manifest)
    out_path = OUT / "metadata" / "augmentation_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"[Step 6] Saved → {out_path}")
    print(f"  Total planned augmentations: {len(df)}")
    if len(df) > 0:
        print(f"  By type: {df['aug_type'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    build_augmentation_manifest()
