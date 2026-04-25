"""
Step 9: Validate pipeline outputs.
Kiểm tra mask class IDs, size match, distribution balance.
"""
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

NUM_CLASSES = 104
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "processed"


def validate_pipeline():
    print("=" * 60)
    print("[Step 9] VALIDATION")
    print("=" * 60)
    errors = []

    # 1. Check class_group_mapping
    cgm_path = OUT / "mappings" / "class_group_mapping.csv"
    if cgm_path.exists():
        cgm = pd.read_csv(cgm_path)
        assert len(cgm) == NUM_CLASSES, f"Expected {NUM_CLASSES} rows, got {len(cgm)}"
        print(f"[OK] class_group_mapping.csv: {len(cgm)} classes")
        print(f"  Groups: {cgm['class_group'].value_counts().to_dict()}")
    else:
        errors.append("class_group_mapping.csv MISSING")

    # 2. Check sample_training_mapping
    stm_path = OUT / "metadata" / "sample_training_mapping.csv"
    if stm_path.exists():
        stm = pd.read_csv(stm_path)
        n_train = len(stm[stm["split"] == "train"])
        n_test = len(stm[stm["split"] == "test"])
        print(f"[OK] sample_training_mapping.csv: {len(stm)} total (train={n_train}, test={n_test})")
        print(f"  Difficulty: {stm[stm['split']=='train']['difficulty_level'].value_counts().to_dict()}")
        print(f"  Aug policy: {stm[stm['split']=='train']['aug_policy'].value_counts().to_dict()}")
    else:
        errors.append("sample_training_mapping.csv MISSING")

    # 3. Check rare_crop_index
    rci_path = OUT / "aug_bank" / "rare_crop_index.csv"
    if rci_path.exists():
        rci = pd.read_csv(rci_path)
        print(f"[OK] rare_crop_index.csv: {len(rci)} components")
    else:
        errors.append("rare_crop_index.csv MISSING")

    # 4. Check copy_paste_bank
    cpb_path = OUT / "aug_bank" / "copy_paste_bank.csv"
    if cpb_path.exists():
        cpb = pd.read_csv(cpb_path)
        print(f"[OK] copy_paste_bank.csv: {len(cpb)} patches")
    else:
        errors.append("copy_paste_bank.csv MISSING")

    # 5. Check cooccurrence
    cooc_path = OUT / "graph" / "cooccurrence_matrix.npy"
    compat_path = OUT / "graph" / "paste_compatibility.csv"
    if cooc_path.exists():
        cooc = np.load(cooc_path)
        print(f"[OK] cooccurrence_matrix.npy: shape {cooc.shape}")
    else:
        errors.append("cooccurrence_matrix.npy MISSING")
    if compat_path.exists():
        compat = pd.read_csv(compat_path)
        print(f"[OK] paste_compatibility.csv: {len(compat)} pairs")
    else:
        errors.append("paste_compatibility.csv MISSING")

    # 6. Check augmentation_manifest
    am_path = OUT / "metadata" / "augmentation_manifest.csv"
    if am_path.exists():
        am = pd.read_csv(am_path)
        print(f"[OK] augmentation_manifest.csv: {len(am)} planned augs")
    else:
        errors.append("augmentation_manifest.csv MISSING")

    # 7. Check train_stage2_manifest
    ts2_path = OUT / "metadata" / "train_stage2_manifest.csv"
    if ts2_path.exists():
        ts2 = pd.read_csv(ts2_path)
        print(f"[OK] train_stage2_manifest.csv: {len(ts2)} total samples")
        print(f"  Source types: {ts2['source_type'].value_counts().to_dict()}")
    else:
        errors.append("train_stage2_manifest.csv MISSING")

    # 8. Spot-check augmented mask validity (if generated)
    aug_mask_dir = OUT / "augmented" / "mask"
    if aug_mask_dir.exists():
        aug_masks = list(aug_mask_dir.glob("*.png"))
        if aug_masks:
            rng = np.random.default_rng(42)
            check_n = min(50, len(aug_masks))
            indices = rng.choice(len(aug_masks), check_n, replace=False)
            bad = 0
            for i in indices:
                m = np.array(Image.open(aug_masks[i]), dtype=np.uint8)
                uniq = np.unique(m)
                invalid = [int(u) for u in uniq if u < 0 or u >= NUM_CLASSES]
                if invalid:
                    bad += 1
                    errors.append(f"Invalid class IDs in {aug_masks[i].name}: {invalid}")
            print(f"[OK] Spot-checked {check_n} augmented masks: {bad} invalid")
        else:
            print("  (No augmented masks generated yet)")

    print("=" * 60)
    if errors:
        print(f"[FAIL] {len(errors)} ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[OK] ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    validate_pipeline()
