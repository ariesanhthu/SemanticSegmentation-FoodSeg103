"""Configuration for the standalone BiSeNet-RTB FoodSeg103 pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import torch


CFG = {
    # Model
    "model_name": "bisenet_rtb",
    "backbone_name": "xception39",
    "backbone_pretrained": False,
    "backbone_pretrained_path": None,
    "backbone_strict_load": False,
    "freeze_backbone": False,
    # Dataset
    "project_root": os.getenv("PROJECT_ROOT", "."),
    "data_root": os.getenv("DATA_ROOT", "datasets/foodseg103-full"),
    "manifest": os.getenv(
        "TRAIN_MANIFEST",
        "datasets/processed/metadata/sample_training_mapping.csv",
    ),
    "train_stage": os.getenv("TRAIN_STAGE", "full"),
    "init_checkpoint": os.getenv("INIT_CHECKPOINT", None),
    "min_init_checkpoint_coverage": float(os.getenv("MIN_INIT_CKPT_COVERAGE", 0.2)),
    "train_img_dir": "train/img",
    "train_mask_dir": "train/mask",
    "test_img_dir": "test/img",
    "test_mask_dir": "test/mask",
    "num_classes": 104,
    "ignore_index": 255,
    "background_id": 103,
    "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    # RTB model
    "graph_prior_path": os.getenv(
        "GRAPH_PRIOR_PATH",
        "work_dirs/graph_prior_104.pt",
    ),
    "require_graph_prior": os.getenv("REQUIRE_GRAPH_PRIOR", "true").lower() == "true",
    "rtb_tex_ch": int(os.getenv("RTB_TEX_CH", 64)),
    "rtb_graph_dim": int(os.getenv("RTB_GRAPH_DIM", 128)),
    "rtb_graph_layers": int(os.getenv("RTB_GRAPH_LAYERS", 2)),
    "rtb_graph_eta": float(os.getenv("RTB_GRAPH_ETA", 0.5)),
    "rtb_graph_xi": float(os.getenv("RTB_GRAPH_XI", 0.15)),
    # RTB loss
    "rtb_aux_weight": float(os.getenv("RTB_AUX_WEIGHT", 0.4)),
    "rtb_pre_weight": float(os.getenv("RTB_PRE_WEIGHT", 0.4)),
    "rtb_edge_weight": float(os.getenv("RTB_EDGE_WEIGHT", 1.0)),
    "rtb_presence_weight": float(os.getenv("RTB_PRESENCE_WEIGHT", 0.15)),
    "rtb_boundary_width": int(os.getenv("RTB_BOUNDARY_WIDTH", 3)),
    # Checkpoint / logging
    "work_dir": os.getenv("WORK_DIR", "./work_dirs/bisenet_rtb"),
    "resume": os.getenv("RESUME", "false").lower() == "true",
    "save_last_name": "last_rtb.pth",
    "save_best_name": "best_miou_rtb.pth",
    "print_freq": 20,
    "eval_every": 1,
    # Reproducibility / runtime
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "cudnn_benchmark": True,
    "num_gpus": int(os.getenv("NUM_GPUS", 1)),
    # Input / augmentation
    "train_size": (768, 768),
    "test_size": (1024, 2048),
    "scale_range": (0.5, 2.0),
    "hflip_prob": 0.5,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    # Dataloader
    "batch_size": int(os.getenv("BATCH_SIZE", 8)),
    "eval_batch_size": int(os.getenv("EVAL_BATCH_SIZE", 4)),
    "num_workers": int(os.getenv("NUM_WORKERS", 2)),
    "pin_memory": True,
    "drop_last": True,
    "overfit_samples": int(os.getenv("OVERFIT_SAMPLES", 0)),
    # Optimizer / scheduler
    "epochs": int(os.getenv("EPOCHS", 80)),
    "lr": float(os.getenv("LR", 1e-3)),
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "poly_power": 0.9,
    "grad_clip_norm": None,
    # Visualization
    "num_vis": 4,
}


def get_paths(cfg: dict) -> dict:
    """Resolve filesystem paths used by the RTB training script."""
    root = Path(cfg["data_root"])
    return {
        "train_img_dir": root / cfg["train_img_dir"],
        "train_mask_dir": root / cfg["train_mask_dir"],
        "test_img_dir": root / cfg["test_img_dir"],
        "test_mask_dir": root / cfg["test_mask_dir"],
        "work_dir": Path(cfg["work_dir"]),
        "backbone_pretrained_path": (
            None
            if cfg["backbone_pretrained_path"] is None
            else Path(cfg["backbone_pretrained_path"])
        ),
    }
