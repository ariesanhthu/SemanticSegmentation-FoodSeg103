from pathlib import Path
import os

import torch


CFG = {
    # ------------------------------------------------------------------
    # Reproducibility / runtime
    # ------------------------------------------------------------------
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "cudnn_benchmark": True,

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    "data_root": os.getenv("DATA_ROOT", "datasets/foodseg103-full"),
    "train_img_dir": "train/img",
    "train_mask_dir": "train/mask",
    "test_img_dir": "test/img",
    "test_mask_dir": "test/mask",
    "class_mapping_name": "class_mapping.json",
    "num_ingredient_classes": 103,
    "background_id": 103,
    "num_classes": 104,
    "ignore_index": 255,
    "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    "validate_samples": True,
    "max_decode_retries": 16,

    # ------------------------------------------------------------------
    # Model: CCNet baseline with ResNet-50 output_stride=8
    # ------------------------------------------------------------------
    "model_name": "ccnet",
    "backbone_name": "resnet50",
    "backbone_pretrained": True,
    "output_stride": 8,
    "cc_channels": 512,
    "cc_recurrence": 2,
    "use_aux_head": True,
    # Inference from standard CCNet / FCN auxiliary supervision setups.
    "aux_weight": 0.4,
    "dropout": 0.1,
    "align_corners": False,

    # ------------------------------------------------------------------
    # FoodSeg103 benchmark-style training recipe
    # ------------------------------------------------------------------
    # Baseline-aligned: switch train crop from 512 to 768.
    "train_size": (768, 768),
    # Keep fixed-size eval by default; set to None to evaluate at original size.
    "eval_size": (768, 768),
    "scale_range": (0.5, 2.0),
    "hflip_prob": 0.5,
    "use_color_jitter": True,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.05,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    "batch_size": 8,
    "eval_batch_size": 1,
    "num_workers": 2,
    "pin_memory": True,
    "drop_last": True,

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "poly_power": 0.9,
    # Baseline-aligned: iterate to 80k instead of short epoch runs.
    "max_iters": 80000,
    "eval_interval": 4000,
    "checkpoint_interval": 4000,
    "eval_milestones": [50, 100],
    "checkpoint_milestones": [50, 100],
    "print_freq": 20,
    "grad_clip_norm": None,

    # ------------------------------------------------------------------
    # Checkpoint / logging
    # ------------------------------------------------------------------
    "work_dir": os.getenv("WORK_DIR", "work_dirs/ccnet_foodseg103_resnet50"),
    "resume": True,
    "save_last_name": "last.pth",
    # Kept for older eval/report scripts; training writes only save_last_name.
    "save_best_name": "last.pth",
    "metrics_csv_name": "metrics.csv",
    "eval_dirname": "eval_results",
}


def get_paths(cfg: dict) -> dict:
    root = Path(cfg["data_root"])
    return {
        "data_root": root,
        "train_img_dir": root / cfg["train_img_dir"],
        "train_mask_dir": root / cfg["train_mask_dir"],
        "test_img_dir": root / cfg["test_img_dir"],
        "test_mask_dir": root / cfg["test_mask_dir"],
        "class_mapping_path": root / cfg["class_mapping_name"],
        "work_dir": Path(cfg["work_dir"]),
        "eval_dir": Path(cfg["work_dir"]) / cfg["eval_dirname"],
    }
