from pathlib import Path
import torch
import os

CFG = {

# ---------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------
    # =====================================================================
    # Model / backbone
    # =====================================================================
    "model_name": "bisenetv1",
    "backbone_name": "xception39",
    "backbone_pretrained": False,
    "backbone_pretrained_path": None,
    "backbone_strict_load": False,
    "freeze_backbone": False,

    # ---------------------------------------------------------------------
    # Dataset
    # Benchmark-aligned mode:
    #   - train split: official train
    #   - val split  : official test
    #   - test split : official test
    # This mirrors the FoodSeg103 benchmark repository behavior.
    # ---------------------------------------------------------------------
    # "data_root": "/content/data/FoodSeg103",
    # "data_root": "../../dataset/foodseg103-full",
    "data_root": os.getenv("DATA_ROOT", "../../dataset/foodseg103-full"),
    "train_img_dir": "train/img",
    "train_mask_dir": "train/mask",
    "test_img_dir": "test/img",
    "test_mask_dir": "test/mask",
    "num_classes": 104,
    "ignore_index": 255,
    "background_id": 103, #!!!!!!!!! FoodSeg103 has 103 foreground classes + 1 background class (id=103)
    "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],

    # ---------------------------------------------------------------------
    # Checkpoint / logging
    # ---------------------------------------------------------------------
    # "work_dir": "/content/drive/MyDrive/[PROJECT][COMPUTER-VISION]/bisenet_foodseg103_benchmark",
    "work_dir": os.getenv("WORK_DIR", "./work_dirs/bisenet_foodseg103"),
    "resume": True,
    "save_last_name": "last.pth",
    "save_best_name": "best_miou.pth",
    "print_freq": 20,
    "eval_every": 1,

    # ---------------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------------
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "cudnn_benchmark": True,

    # ---------------------------------------------------------------------
    # Input / augmentation
    # BiSeNet paper uses random scale + random crop + random hflip.
    # FoodSeg103 benchmark baselines resize with ratio range and crop.
    # We keep a clean, explicit hybrid recipe for this project:
    #   - ratio range: 0.5 to 2.0 (FoodSeg103 benchmark spirit)
    #   - crop size : 512 x 1024 (FoodSeg103 baseline table)
    # ---------------------------------------------------------------------
    # Dựa theo thống kê mean_height = 584.47, mean_width = 703.62, median_height = 384, median_width = 512 
    "train_size": (384, 512),  
    "test_size": None,          # None = evaluate at original image size
    "scale_range": (0.5, 2.0),
    "hflip_prob": 0.5,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],

    # ---------------------------------------------------------------------
    # Dataloader
    # ---------------------------------------------------------------------
    "batch_size": 8,
    "eval_batch_size": 4,
    "num_workers": 2,
    "pin_memory": True,
    "drop_last": True,

    # ---------------------------------------------------------------------
    # Optimizer / scheduler
    # BiSeNet paper defaults:
    #   SGD, momentum=0.9, weight_decay=1e-4, poly lr, power=0.9
    #   initial lr = 2.5e-2
    # ---------------------------------------------------------------------
    "epochs": 80,
    "lr": 2.5e-2,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "poly_power": 0.9,
    "aux_weight": 1.0,
    "grad_clip_norm": None,


    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    "num_vis": 4,

}


def get_paths(cfg: dict) -> dict:
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
