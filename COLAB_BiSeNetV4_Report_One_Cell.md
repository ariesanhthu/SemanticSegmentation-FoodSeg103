# Colab one-cell report for BiSeNet V4

Copy the whole cell below into one Google Colab code cell.

Before running, edit these variables near the top if needed:

- `DATA_ROOT`: FoodSeg103 root folder containing `class_mapping.json`, `train/`, and `test/`.
- `CKPT_PATH`: BiSeNet checkpoint path, usually on Google Drive.
- `MAX_ITEMS`: set `50` for a smoke run, or `None` for the full test split.

```python
# ============================================================
# BiSeNet V4 one-cell Colab report
# - clone repo
# - load BiSeNet checkpoint
# - run model on FoodSeg103
# - export per-class / per-image / confusion / error analysis
# ============================================================

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from contextlib import nullcontext


# ----------------------------
# 1) User config
# ----------------------------
REPO_URL = "https://github.com/ariesanhthu/SemanticSegmentation-BiSeNet-FoodSeg103.git"
REPO_DIR = Path("/content/SemanticSegmentation-BiSeNet-FoodSeg103")
BRANCH = "main"

# Folder must contain: class_mapping.json, train/img, train/mask, test/img, test/mask.
DATA_ROOT = Path("/content/data/foodseg103-full")

# This file is not tracked in GitHub because *.pth/work_dirs are ignored.
# Put your checkpoint on Drive, then point this variable to it.
CKPT_PATH = Path("/content/drive/MyDrive/checkpoints/bisenet_v4/bisenet_v4.pth")

# Optional direct checkpoint URL. Leave empty if using CKPT_PATH from Drive.
CKPT_URL = ""

SPLIT = "test"
EVAL_SIZE = (768, 768)   # use None to evaluate at original image size
BATCH_SIZE = 1
MAX_ITEMS = None         # example: 50 for quick smoke test, None for full split
NUM_WORKERS = 2
REPORT_DIR = Path("/content/bisenet_v4_report")


# ----------------------------
# 2) Setup helpers
# ----------------------------
def run_cmd(cmd, cwd=None):
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_bisenet_summary(output_path, summary, per_class_rows, confusion_rows, per_image_rows):
    worst_classes = per_class_rows[:10]
    worst_images = sorted(per_image_rows, key=lambda row: row["mIoU_present"])[:10]
    top_pairs = confusion_rows[:10]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("BiSeNet FoodSeg103 Evaluation Summary\n")
        f.write("=" * 80 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nWorst 10 classes by IoU\n")
        f.write("-" * 80 + "\n")
        for row in worst_classes:
            f.write(
                f"{row['class_id']:>3} | {row['class_name']:<28} | "
                f"IoU={row['IoU']:.4f} | Acc={row['Acc']:.4f} | gt={row['gt_pixels']}\n"
            )

        f.write("\nTop 10 confusion pairs\n")
        f.write("-" * 80 + "\n")
        for row in top_pairs:
            f.write(f"{row['gt_name']} -> {row['pred_name']} | count={row['count']}\n")

        f.write("\nWorst 10 images\n")
        f.write("-" * 80 + "\n")
        for row in worst_images:
            f.write(
                f"{row['stem']} | mIoU_present={row['mIoU_present']:.4f} | "
                f"pixel_acc={row['pixel_acc']:.4f} | classes={row['num_present_classes']}\n"
            )


def load_checkpoint_flexible(model, checkpoint_path, device):
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model")
            or checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported state_dict format: {checkpoint_path}")

    cleaned_state = {key.replace("module.", ""): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)

    return {
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "global_iter": checkpoint.get("global_iter") if isinstance(checkpoint, dict) else None,
        "best_miou": checkpoint.get("best_miou") if isinstance(checkpoint, dict) else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


# ----------------------------
# 3) Mount Drive, clone repo, install deps
# ----------------------------
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print(f"[WARN] Could not mount Drive automatically: {exc}")

if not REPO_DIR.exists():
    run_cmd(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR])
else:
    run_cmd(["git", "fetch", "--depth", "1", "origin", BRANCH], cwd=REPO_DIR)
    run_cmd(["git", "checkout", BRANCH], cwd=REPO_DIR)
    run_cmd(["git", "pull", "--ff-only"], cwd=REPO_DIR)

run_cmd([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-r",
    REPO_DIR / "requirements.txt",
    "albumentations",
    "pandas",
    "seaborn",
])

sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)


# ----------------------------
# 4) Resolve checkpoint / data
# ----------------------------
if not CKPT_PATH.exists() and CKPT_URL:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in CKPT_URL:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        run_cmd(["gdown", "--fuzzy", CKPT_URL, "-O", CKPT_PATH])
    else:
        import urllib.request
        print(f"Downloading checkpoint from {CKPT_URL}")
        urllib.request.urlretrieve(CKPT_URL, CKPT_PATH)

if not CKPT_PATH.exists():
    try:
        from google.colab import files
        print("Checkpoint not found. Upload bisenet_v4.pth now.")
        uploaded = files.upload()
        if uploaded:
            uploaded_name = next(iter(uploaded.keys()))
            CKPT_PATH = Path("/content") / uploaded_name
    except Exception as exc:
        print(f"[WARN] Upload fallback failed: {exc}")

if not CKPT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CKPT_PATH}\n"
        "Set CKPT_PATH to your Drive checkpoint or set CKPT_URL."
    )

required_dirs = [
    DATA_ROOT / "test" / "img",
    DATA_ROOT / "test" / "mask",
]
missing_dirs = [str(path) for path in required_dirs if not path.exists()]
if missing_dirs:
    raise FileNotFoundError(
        "DATA_ROOT is not ready. Missing:\n"
        + "\n".join(missing_dirs)
        + "\nSet DATA_ROOT to the FoodSeg103 folder that contains class_mapping.json, train/, test/."
    )


# ----------------------------
# 5) Imports from repo
# ----------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.bisenet_foodseg103 import CFG, get_paths
from datasets.foodseg103 import FoodSegDataset, EvalTransform, build_samples
from datasets.foodseg103_ccnet import load_class_mapping
from models.builder import build_model
from analysis.ccnet_report.runner import (
    analyze_error_patterns,
    build_confusion_rows,
    build_per_class_rows,
    compute_hist_np,
    compute_per_image_row,
    compute_scores_from_hist,
    save_csv,
)
from analysis.ccnet_report.plots import (
    save_case_grid,
    save_confusion_heatmap,
    save_frequency_vs_iou_plot,
    save_metric_distribution_plot,
    save_top_confusions_plot,
    save_worst_classes_plot,
)


# ----------------------------
# 6) Runtime config
# ----------------------------
cfg = CFG.copy()
cfg["data_root"] = str(DATA_ROOT)
cfg["work_dir"] = str(REPORT_DIR)
cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
cfg["eval_batch_size"] = BATCH_SIZE
cfg["num_workers"] = NUM_WORKERS
cfg["pin_memory"] = bool(torch.cuda.is_available())
cfg["test_size"] = EVAL_SIZE
cfg["amp"] = bool(torch.cuda.is_available())

mapping = load_class_mapping(
    data_root=DATA_ROOT,
    mapping_name="class_mapping.json",
    fallback_num_classes=cfg["num_classes"],
    fallback_background_id=cfg["background_id"],
    fallback_num_ingredient_classes=cfg["num_classes"] - 1,
)
cfg.update(mapping)
paths = get_paths(cfg)

if SPLIT == "train":
    img_dir = paths["train_img_dir"]
    mask_dir = paths["train_mask_dir"]
else:
    img_dir = paths["test_img_dir"]
    mask_dir = paths["test_mask_dir"]

samples = build_samples(img_dir, mask_dir)
if MAX_ITEMS is not None:
    samples = samples[: max(0, int(MAX_ITEMS))]
if len(samples) == 0:
    raise RuntimeError(f"No samples found in {img_dir} and {mask_dir}")

transform = EvalTransform(
    mean=cfg["imagenet_mean"],
    std=cfg["imagenet_std"],
    ignore_index=cfg["ignore_index"],
    num_classes=cfg["num_classes"],
    out_size=EVAL_SIZE,
)
loader = DataLoader(
    FoodSegDataset(samples, transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=cfg["pin_memory"],
)

print(f"Device: {cfg['device']}")
print(f"Samples: {len(samples)}")
print(f"Eval size: {EVAL_SIZE}")
print(f"Checkpoint: {CKPT_PATH}")


# ----------------------------
# 7) Build and load model
# ----------------------------
model = build_model(cfg, paths).to(cfg["device"])
load_info = load_checkpoint_flexible(model, CKPT_PATH, cfg["device"])
model.eval()

print("Checkpoint load info:")
print(json.dumps({
    "epoch": load_info["epoch"],
    "global_iter": load_info["global_iter"],
    "best_miou": load_info["best_miou"],
    "missing_key_count": len(load_info["missing_keys"]),
    "unexpected_key_count": len(load_info["unexpected_keys"]),
}, indent=2))


# ----------------------------
# 8) Run inference and metrics
# ----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
global_hist = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
per_image_rows = []
running_loss = 0.0

amp_context = (
    torch.autocast(device_type="cuda", dtype=torch.float16)
    if cfg["device"].startswith("cuda") and cfg["amp"]
    else nullcontext()
)

with torch.no_grad():
    for images, masks, stems, img_paths, mask_paths in tqdm(loader, desc="BiSeNet eval"):
        images = images.to(cfg["device"], non_blocking=True)
        masks = masks.to(cfg["device"], non_blocking=True)

        with amp_context:
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = criterion(logits, masks)

        running_loss += float(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        gt_batch = masks.detach().cpu().numpy()

        for idx in range(len(stems)):
            pred = preds[idx]
            gt = gt_batch[idx]
            per_image_rows.append(
                compute_per_image_row(
                    pred=pred,
                    gt=gt,
                    num_classes=cfg["num_classes"],
                    ignore_index=cfg["ignore_index"],
                    stem=stems[idx],
                    img_path=img_paths[idx],
                    mask_path=mask_paths[idx],
                )
            )
            global_hist += compute_hist_np(
                pred,
                gt,
                num_classes=cfg["num_classes"],
                ignore_index=cfg["ignore_index"],
            )

dataset_scores = compute_scores_from_hist(global_hist)
dataset_scores["loss"] = running_loss / max(1, len(loader))

per_class_rows = build_per_class_rows(global_hist, cfg)
confusion_rows = build_confusion_rows(global_hist, cfg)
error_rows = analyze_error_patterns(
    per_image_rows=per_image_rows,
    per_class_rows=per_class_rows,
    confusion_rows=confusion_rows,
    background_id=int(cfg["background_id"]),
)

summary = {
    "checkpoint": str(CKPT_PATH),
    "split": SPLIT,
    "num_images": len(per_image_rows),
    "num_classes": cfg["num_classes"],
    "background_id": cfg["background_id"],
    "eval_loss": dataset_scores["loss"],
    "dataset_aAcc": dataset_scores["aAcc"],
    "dataset_mAcc": dataset_scores["mAcc"],
    "dataset_mIoU": dataset_scores["mIoU"],
    "mean_image_pixel_acc": float(np.mean([row["pixel_acc"] for row in per_image_rows])) if per_image_rows else 0.0,
    "mean_image_mIoU_present": float(np.mean([row["mIoU_present"] for row in per_image_rows])) if per_image_rows else 0.0,
    "mean_image_mAcc_present": float(np.mean([row["mAcc_present"] for row in per_image_rows])) if per_image_rows else 0.0,
    "mean_num_present_classes": float(np.mean([row["num_present_classes"] for row in per_image_rows])) if per_image_rows else 0.0,
}


# ----------------------------
# 9) Save report files and plots
# ----------------------------
REPORT_DIR.mkdir(parents=True, exist_ok=True)
plots_dir = REPORT_DIR / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

save_json(summary, REPORT_DIR / "summary.json")
save_json(dataset_scores, REPORT_DIR / "dataset_scores.json")
save_json(error_rows, REPORT_DIR / "error_analysis.json")
write_bisenet_summary(REPORT_DIR / "summary.txt", summary, per_class_rows, confusion_rows, per_image_rows)

save_csv(
    REPORT_DIR / "per_image_metrics.csv",
    per_image_rows,
    ["stem", "img_path", "mask_path", "pixel_acc", "mIoU_present", "mAcc_present", "num_present_classes", "valid_pixels"],
)
save_csv(
    REPORT_DIR / "per_class_metrics.csv",
    per_class_rows,
    ["class_id", "class_name", "gt_pixels", "pred_pixels", "tp_pixels", "IoU", "Acc"],
)
save_csv(
    REPORT_DIR / "top_confusion_pairs.csv",
    confusion_rows,
    ["gt_id", "gt_name", "pred_id", "pred_name", "count"],
)
save_csv(
    REPORT_DIR / "error_analysis.csv",
    error_rows,
    ["pattern", "evidence", "direction"],
)

save_worst_classes_plot(
    per_class_rows,
    plots_dir / "01_worst_classes_iou.png",
    background_id=int(cfg["background_id"]),
    top_k=20,
)
save_top_confusions_plot(
    confusion_rows,
    plots_dir / "02_top_confusion_pairs.png",
    top_k=20,
)
save_metric_distribution_plot(
    per_image_rows,
    plots_dir / "03_image_metric_distributions.png",
)
save_frequency_vs_iou_plot(
    per_class_rows,
    plots_dir / "04_class_frequency_vs_iou.png",
    background_id=int(cfg["background_id"]),
)
save_confusion_heatmap(
    global_hist,
    per_class_rows,
    plots_dir / "05_confusion_heatmap_worst_classes.png",
    background_id=int(cfg["background_id"]),
    top_k=12,
)

try:
    worst_rows = sorted(per_image_rows, key=lambda row: row["mIoU_present"])[:4]
    best_rows = sorted(per_image_rows, key=lambda row: row["mIoU_present"], reverse=True)[:4]
    save_case_grid(
        model,
        worst_rows,
        transform,
        cfg,
        plots_dir / "06_worst_cases.png",
        title=f"Worst {len(worst_rows)} cases by image-wise mIoU_present",
    )
    save_case_grid(
        model,
        best_rows,
        transform,
        cfg,
        plots_dir / "07_best_cases.png",
        title=f"Best {len(best_rows)} cases by image-wise mIoU_present",
    )
except Exception as exc:
    print(f"[WARN] Case-grid plots skipped: {exc}")

zip_base = shutil.make_archive(str(REPORT_DIR), "zip", REPORT_DIR)


# ----------------------------
# 10) Show result tables
# ----------------------------
print("\n=== Dataset summary ===")
for key, value in summary.items():
    print(f"{key}: {value}")

per_class_df = pd.DataFrame(per_class_rows)
per_image_df = pd.DataFrame(per_image_rows)
confusion_df = pd.DataFrame(confusion_rows)
error_df = pd.DataFrame(error_rows)

print("\n=== Worst 20 classes by IoU ===")
display(per_class_df[per_class_df["gt_pixels"] > 0].sort_values("IoU").head(20))

print("\n=== Top 20 classes by GT pixels ===")
display(per_class_df.sort_values("gt_pixels", ascending=False).head(20))

print("\n=== Top 20 confusion pairs ===")
display(confusion_df.head(20) if len(confusion_df) else pd.DataFrame(columns=["gt_name", "pred_name", "count"]))

print("\n=== Error analysis ===")
display(error_df if len(error_df) else pd.DataFrame(columns=["pattern", "evidence", "direction"]))

print(f"\nReport folder: {REPORT_DIR}")
print(f"Report zip: {zip_base}")
print("Main files:")
for name in [
    "summary.txt",
    "summary.json",
    "dataset_scores.json",
    "per_class_metrics.csv",
    "per_image_metrics.csv",
    "top_confusion_pairs.csv",
    "error_analysis.csv",
]:
    print(" -", REPORT_DIR / name)
```
