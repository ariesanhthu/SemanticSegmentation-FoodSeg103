"""
Step 8: Tạo train_stage2_manifest.csv.
Gộp original train + augmented thành manifest cuối cho Stage 2.
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "processed"


def build_train_stage2_manifest():
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)

    sample_map = pd.read_csv(OUT / "metadata" / "sample_training_mapping.csv")
    aug_manifest = pd.read_csv(OUT / "metadata" / "augmentation_manifest.csv")

    orig_train = sample_map[sample_map["split"] == "train"]

    # Original samples
    orig_rows = []
    for _, r in orig_train.iterrows():
        orig_rows.append({
            "sample_id": r["stem"],
            "image_path": r["image_path"],
            "mask_path": r["mask_path"],
            "source_type": "original",
            "base_stem": r["stem"],
            "sampling_weight": r["stage2_weight"],
        })

    # Augmented samples
    aug_rows = []
    for _, r in aug_manifest.iterrows():
        aug_rows.append({
            "sample_id": r["aug_id"],
            "image_path": r["output_image_path"],
            "mask_path": r["output_mask_path"],
            "source_type": r["aug_type"],
            "base_stem": r["base_stem"],
            "sampling_weight": 2.0,
        })

    final = pd.DataFrame(orig_rows + aug_rows)
    out_path = OUT / "metadata" / "train_stage2_manifest.csv"
    final.to_csv(out_path, index=False)
    print(f"[Step 8] Saved → {out_path}")
    print(f"  Original: {len(orig_rows)}, Augmented: {len(aug_rows)}, Total: {len(final)}")
    print(f"  Source types: {final['source_type'].value_counts().to_dict()}")
    return final


if __name__ == "__main__":
    build_train_stage2_manifest()
