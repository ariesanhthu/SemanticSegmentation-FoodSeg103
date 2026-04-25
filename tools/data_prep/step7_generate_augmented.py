"""
Step 7: Generate augmented images từ manifest.
Đọc manifest CSV, apply augmentation ops, lưu output.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from augmentation_ops import light_color_aug, crop_around_bbox, random_foreground_crop, paste_patch

BACKGROUND_ID = 103
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "processed"

ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_augmented_images():
    manifest = pd.read_csv(OUT / "metadata" / "augmentation_manifest.csv")
    rare_crop_path = OUT / "aug_bank" / "rare_crop_index.csv"
    bank_path = OUT / "aug_bank" / "copy_paste_bank.csv"

    rare_crop_df = pd.read_csv(rare_crop_path) if rare_crop_path.exists() else pd.DataFrame()
    bank_df = pd.read_csv(bank_path) if bank_path.exists() else pd.DataFrame()

    rng = np.random.default_rng(42)
    skipped = 0

    print(f"[Step 7] Generating {len(manifest)} augmented images...")
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Augmenting"):
        out_img_path = row["output_image_path"]
        out_mask_path = row["output_mask_path"]

        # Skip if already generated
        if Path(out_img_path).exists() and Path(out_mask_path).exists():
            skipped += 1
            continue

        image = np.array(Image.open(row["base_image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["base_mask_path"]), dtype=np.uint8)
        aug_type = row["aug_type"]

        if aug_type == "light_color":
            aug_img = light_color_aug(image, rng)
            aug_mask = mask

        elif aug_type == "hard_crop":
            aug_img, aug_mask = random_foreground_crop(
                image, mask, crop_size=384, background_id=BACKGROUND_ID, rng=rng,
            )

        elif aug_type == "rare_crop":
            candidates = rare_crop_df[rare_crop_df["stem"] == row["base_stem"]] if len(rare_crop_df) > 0 else pd.DataFrame()
            if len(candidates) > 0:
                c = candidates.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
                bbox = (int(c["x1"]), int(c["y1"]), int(c["x2"]), int(c["y2"]))
                aug_img, aug_mask = crop_around_bbox(
                    image, mask, bbox, crop_size=int(c["crop_size"]),
                    background_id=BACKGROUND_ID,
                )
            else:
                aug_img, aug_mask = random_foreground_crop(
                    image, mask, crop_size=384, background_id=BACKGROUND_ID, rng=rng,
                )

        elif aug_type == "copy_paste":
            if len(bank_df) > 0:
                patch_row = bank_df.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
                patch_img = np.array(Image.open(patch_row["patch_image_path"]).convert("RGB"))
                patch_mask = np.array(Image.open(patch_row["patch_mask_path"]), dtype=np.uint8)

                h, w = mask.shape
                ph, pw = patch_mask.shape

                # Paste on foreground area for context
                ys, xs = np.where(mask != BACKGROUND_ID)
                if len(xs) > 0:
                    idx = rng.integers(0, len(xs))
                    x = int(xs[idx] - pw // 2)
                    y = int(ys[idx] - ph // 2)
                else:
                    x = int(rng.integers(0, max(1, w - pw)))
                    y = int(rng.integers(0, max(1, h - ph)))

                aug_img, aug_mask = paste_patch(
                    image, mask, patch_img, patch_mask,
                    int(patch_row["class_id"]), x, y,
                )
                aug_img = light_color_aug(aug_img, rng)
            else:
                aug_img = light_color_aug(image, rng)
                aug_mask = mask
        else:
            aug_img, aug_mask = image, mask

        Image.fromarray(aug_img).save(out_img_path)
        Image.fromarray(aug_mask.astype(np.uint8)).save(out_mask_path)

    print(f"[Step 7] Done. Skipped {skipped} already-existing files.")


if __name__ == "__main__":
    generate_augmented_images()
