"""
Step 4: Tạo copy_paste_bank.csv + patches/.
Extract patches của rare class components để dùng cho copy-paste augmentation.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "processed"


def build_copy_paste_bank():
    patch_dir = OUT / "aug_bank" / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    rare_df = pd.read_csv(OUT / "aug_bank" / "rare_crop_index.csv")
    print(f"[Step 4] Extracting patches from {len(rare_df)} rare components...")

    bank_rows = []
    for idx, row in tqdm(rare_df.iterrows(), total=len(rare_df), desc="Patches"):
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["mask_path"]), dtype=np.uint8)

        cid = int(row["class_id"])
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

        margin = 8
        h, w = mask.shape
        x1m, y1m = max(0, x1 - margin), max(0, y1 - margin)
        x2m, y2m = min(w, x2 + margin), min(h, y2 + margin)

        patch_img = image[y1m:y2m, x1m:x2m]
        patch_mask = (mask[y1m:y2m, x1m:x2m] == cid).astype(np.uint8) * 255

        if patch_mask.sum() == 0:
            continue

        patch_id = f"{row['stem']}_c{cid}_comp{int(row['component_id'])}"
        img_out = patch_dir / f"{patch_id}_img.png"
        mask_out = patch_dir / f"{patch_id}_mask.png"

        Image.fromarray(patch_img).save(img_out)
        Image.fromarray(patch_mask).save(mask_out)

        bank_rows.append({
            "patch_id": patch_id,
            "source_stem": row["stem"],
            "class_id": cid,
            "class_name": row["class_name"],
            "class_group": row["class_group"],
            "patch_image_path": str(img_out),
            "patch_mask_path": str(mask_out),
            "area_pixels": int(row["area_pixels"]),
            "area_ratio": float(row["area_ratio"]),
            "quality_score": float(row["area_ratio"]),
        })

    df = pd.DataFrame(bank_rows)
    out_path = OUT / "aug_bank" / "copy_paste_bank.csv"
    df.to_csv(out_path, index=False)
    print(f"[Step 4] Saved → {out_path}")
    print(f"  Total patches: {len(df)}")
    return df


if __name__ == "__main__":
    build_copy_paste_bank()
