"""
run_pipeline.py — Master script chạy toàn bộ pipeline hoặc từng bước.

Usage:
    # Chạy tất cả metadata steps (không generate ảnh):
    python tools/data_prep/run_pipeline.py --steps 1,2,3,4,5,6,8,9

    # Chạy full pipeline (bao gồm generate ảnh):
    python tools/data_prep/run_pipeline.py --steps all

    # Chạy từng bước:
    python tools/data_prep/run_pipeline.py --steps 1
    python tools/data_prep/run_pipeline.py --steps 7

Thứ tự dependencies:
    1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
"""
import argparse
import time
import sys
from pathlib import Path

# Add data_prep to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_step(step_num: int):
    """Chạy một bước pipeline."""
    t0 = time.time()

    if step_num == 1:
        from step1_class_group_mapping import build_class_group_mapping
        build_class_group_mapping()

    elif step_num == 2:
        from step2_sample_mapping import build_sample_training_mapping
        build_sample_training_mapping()

    elif step_num == 3:
        from step3_rare_crop_index import build_rare_crop_index
        build_rare_crop_index()

    elif step_num == 4:
        from step4_copy_paste_bank import build_copy_paste_bank
        build_copy_paste_bank()

    elif step_num == 5:
        from step5_cooccurrence import build_cooccurrence_graph
        build_cooccurrence_graph()

    elif step_num == 6:
        from step6_aug_manifest import build_augmentation_manifest
        build_augmentation_manifest()

    elif step_num == 7:
        from step7_generate_augmented import generate_augmented_images
        generate_augmented_images()

    elif step_num == 8:
        from step8_final_manifest import build_train_stage2_manifest
        build_train_stage2_manifest()

    elif step_num == 9:
        from step9_validate import validate_pipeline
        validate_pipeline()

    else:
        print(f"Unknown step: {step_num}")
        return

    elapsed = time.time() - t0
    print(f"  ⏱ Step {step_num} done in {elapsed:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="FoodSeg103 Data Prep Pipeline")
    parser.add_argument(
        "--steps", type=str, default="1,2,3,4,5,6,8,9",
        help="Comma-separated step numbers or 'all'. Default: 1,2,3,4,5,6,8,9 (skip image generation)",
    )
    args = parser.parse_args()

    if args.steps.lower() == "all":
        steps = list(range(1, 10))
    else:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    print(f"{'='*60}")
    print(f"FoodSeg103 Data Mapping & Augmentation Pipeline")
    print(f"Steps to run: {steps}")
    print(f"{'='*60}\n")

    for s in steps:
        run_step(s)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
