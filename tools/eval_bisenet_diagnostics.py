from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs.bisenet_foodseg103 import CFG, get_paths
from datasets.foodseg103 import EvalTransform, FoodSegDataset, build_samples
from models.builder import build_model
from utils.misc import ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose BiSeNet prediction collapse on FoodSeg103. "
            "This is an eval-side debugging report, not a benchmark-only report."
        )
    )
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-name", type=str, default="collapse_diagnostics")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--eval-size",
        type=str,
        default=None,
        help="Eval resize as HxW, H,W, one integer, or none/original. Defaults to checkpoint cfg test_size if present.",
    )
    parser.add_argument("--background-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--examples-per-group", type=int, default=8)
    parser.add_argument("--sink-image-ratio", type=float, default=0.50)
    parser.add_argument("--fg-low", type=float, default=0.15)
    parser.add_argument("--fg-good", type=float, default=0.35)
    parser.add_argument("--class-low", type=float, default=0.10)
    parser.add_argument("--skip-train-presence", action="store_true")
    parser.add_argument("--presence-max-items", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_size(value: str | None, fallback: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return fallback

    normalized = value.strip().lower()
    if normalized in {"none", "null", "original", "orig"}:
        return None

    normalized = normalized.replace("x", ",").replace(" ", "")
    parts = [part for part in normalized.split(",") if part]
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Invalid --eval-size value: {value}")


def safe_div(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_mapping(data_root: Path, cfg: dict) -> dict:
    num_classes = int(cfg["num_classes"])
    background_id = int(cfg["background_id"])
    class_names = [f"class_{idx}" for idx in range(num_classes)]

    mapping_path = data_root / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        background_id = int(mapping.get("background_id", background_id))
        num_ingredient_classes = int(mapping.get("num_ingredient_classes", num_classes - 1))
        ids: list[int] = [background_id, num_ingredient_classes]

        for raw_id, name in mapping.get("id_to_class", {}).items():
            class_id = int(raw_id)
            ids.append(class_id)
            while class_id >= len(class_names):
                class_names.append(f"class_{len(class_names)}")
            class_names[class_id] = str(name)

        for _, raw_id in mapping.get("class_to_id", {}).items():
            ids.append(int(raw_id))

        num_classes = max(num_classes, max(ids) + 1 if ids else num_classes)
        while len(class_names) < num_classes:
            class_names.append(f"class_{len(class_names)}")

    if 0 <= background_id < len(class_names):
        class_names[background_id] = "background"

    return {
        "num_classes": num_classes,
        "background_id": background_id,
        "class_names": class_names[:num_classes],
        "mapping_path": str(mapping_path) if mapping_path.exists() else None,
    }


def resolve_checkpoint_path(cfg: dict, paths: dict, ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg)

    candidates = [
        paths["work_dir"] / cfg["save_best_name"],
        paths["work_dir"] / "bisenet_v4.pth",
        paths["work_dir"] / cfg["save_last_name"],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def peek_checkpoint_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("cfg"), dict):
        return checkpoint["cfg"]
    return {}


def load_checkpoint_flexible(model: torch.nn.Module, ckpt_path: Path, device: str) -> dict:
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            maybe_state = checkpoint.get(key)
            if isinstance(maybe_state, dict):
                state_dict = maybe_state
                break

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model_keys = set(model.state_dict().keys())
    matched = len(model_keys.intersection(cleaned.keys()))
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    return {
        "checkpoint": str(ckpt_path),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "global_iter": checkpoint.get("global_iter") if isinstance(checkpoint, dict) else None,
        "best_miou": checkpoint.get("best_miou") if isinstance(checkpoint, dict) else None,
        "matched_keys": matched,
        "model_key_count": len(model_keys),
        "coverage": safe_div(matched, len(model_keys)),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def compute_hist_np(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    valid = target != ignore_index
    pred_v = pred[valid].astype(np.int64, copy=False)
    target_v = target[valid].astype(np.int64, copy=False)
    valid_cls = (target_v >= 0) & (target_v < num_classes) & (pred_v >= 0) & (pred_v < num_classes)
    indices = num_classes * target_v[valid_cls] + pred_v[valid_cls]
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_scores_from_hist(hist: np.ndarray) -> dict:
    hist_f = hist.astype(np.float64, copy=False)
    diag = np.diag(hist_f)
    gt_sum = hist_f.sum(axis=1)
    pred_sum = hist_f.sum(axis=0)
    valid = gt_sum > 0
    union = gt_sum + pred_sum - diag

    return {
        "aAcc": safe_div(diag.sum(), hist_f.sum()),
        "mAcc": float((diag / np.maximum(gt_sum, 1.0))[valid].mean()) if valid.any() else 0.0,
        "mIoU": float((diag / np.maximum(union, 1.0))[valid].mean()) if valid.any() else 0.0,
        "IoU_per_class": (diag / np.maximum(union, 1.0)).tolist(),
        "Acc_per_class": (diag / np.maximum(gt_sum, 1.0)).tolist(),
        "valid_class_mask": valid.tolist(),
    }


def compute_fg_bg_metrics_from_hist(hist: np.ndarray, background_id: int) -> dict:
    fg_ids = [idx for idx in range(hist.shape[0]) if idx != background_id]
    bg_bg = int(hist[background_id, background_id])
    bg_to_fg = int(hist[background_id, fg_ids].sum())
    fg_to_bg = int(hist[np.ix_(fg_ids, [background_id])].sum())
    fg_to_fg = int(hist[np.ix_(fg_ids, fg_ids)].sum())
    fg_correct_class = int(np.diag(hist)[fg_ids].sum())
    fg_wrong_class = max(0, fg_to_fg - fg_correct_class)

    return {
        "fg_iou": safe_div(fg_to_fg, fg_to_fg + fg_to_bg + bg_to_fg),
        "fg_precision": safe_div(fg_to_fg, fg_to_fg + bg_to_fg),
        "fg_recall": safe_div(fg_to_fg, fg_to_fg + fg_to_bg),
        "bg_iou": safe_div(bg_bg, bg_bg + bg_to_fg + fg_to_bg),
        "bg_precision": safe_div(bg_bg, bg_bg + fg_to_bg),
        "bg_recall": safe_div(bg_bg, bg_bg + bg_to_fg),
        "fg_correct_class_pixels": fg_correct_class,
        "fg_wrong_class_pixels": fg_wrong_class,
        "missed_foreground_pixels": fg_to_bg,
        "false_foreground_pixels": bg_to_fg,
        "semantic_accuracy_on_gt_foreground": safe_div(fg_correct_class, fg_to_fg + fg_to_bg),
        "oracle_foreground_iou": safe_div(fg_to_fg, fg_to_fg + fg_to_bg + bg_to_fg),
    }


def classify_image_failure(row: dict, args: argparse.Namespace, background_id: int) -> str:
    if row["gt_fg_pixels"] > 0 and row["pred_fg_ratio"] < 0.02:
        return "background_overprediction"
    if row["dominant_pred_class_id"] != background_id and row["dominant_pred_ratio"] >= args.sink_image_ratio:
        return "sink_class_overprediction"
    if row["fg_iou"] < args.fg_low and row["miou_present_fg"] < args.class_low:
        return "localization_failure"
    if row["fg_iou"] >= args.fg_good and row["miou_present_fg"] < args.class_low:
        return "classification_failure"
    if row["fg_iou"] >= args.fg_good and row["miou_present_fg"] >= args.class_low:
        return "partially_working"
    return "mixed_failure"


def compute_per_image_row(
    pred: np.ndarray,
    gt: np.ndarray,
    hist: np.ndarray,
    cfg: dict,
    stem: str,
    img_path: str,
    mask_path: str,
    args: argparse.Namespace,
) -> dict:
    num_classes = int(cfg["num_classes"])
    background_id = int(cfg["background_id"])
    ignore_index = int(cfg["ignore_index"])
    valid = gt != ignore_index
    valid_pixels = int(valid.sum())

    diag = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pred_sum = hist.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - diag
    iou = diag / np.maximum(union, 1.0)

    present_fg = gt_sum > 0
    if 0 <= background_id < num_classes:
        present_fg[background_id] = False
    miou_present_fg = float(iou[present_fg].mean()) if present_fg.any() else 0.0
    macc_present_fg = float((diag / np.maximum(gt_sum, 1.0))[present_fg].mean()) if present_fg.any() else 0.0

    gt_v = gt[valid]
    pred_v = pred[valid]
    pixel_acc = safe_div(int((gt_v == pred_v).sum()), valid_pixels)

    gt_fg = gt_v != background_id
    pred_fg = pred_v != background_id
    fg_inter = int((gt_fg & pred_fg).sum())
    fg_union = int((gt_fg | pred_fg).sum())
    fg_fp = int((~gt_fg & pred_fg).sum())
    fg_fn = int((gt_fg & ~pred_fg).sum())
    bg_inter = int((~gt_fg & ~pred_fg).sum())
    bg_union = int((~gt_fg | ~pred_fg).sum())

    pred_counts = np.bincount(pred_v.astype(np.int64, copy=False), minlength=num_classes)
    dominant_pred_id = int(pred_counts.argmax()) if valid_pixels else background_id
    dominant_pred_pixels = int(pred_counts[dominant_pred_id]) if valid_pixels else 0

    row = {
        "stem": stem,
        "img_path": img_path,
        "mask_path": mask_path,
        "pixel_acc": pixel_acc,
        "miou_present_fg": miou_present_fg,
        "macc_present_fg": macc_present_fg,
        "fg_iou": safe_div(fg_inter, fg_union),
        "fg_precision": safe_div(fg_inter, fg_inter + fg_fp),
        "fg_recall": safe_div(fg_inter, fg_inter + fg_fn),
        "bg_iou": safe_div(bg_inter, bg_union),
        "valid_pixels": valid_pixels,
        "gt_fg_pixels": int(gt_fg.sum()),
        "pred_fg_pixels": int(pred_fg.sum()),
        "gt_fg_ratio": safe_div(int(gt_fg.sum()), valid_pixels),
        "pred_fg_ratio": safe_div(int(pred_fg.sum()), valid_pixels),
        "correct_fg_class_pixels": int((gt_fg & pred_fg & (gt_v == pred_v)).sum()),
        "fg_wrong_class_pixels": int((gt_fg & pred_fg & (gt_v != pred_v)).sum()),
        "missed_fg_pixels": int((gt_fg & ~pred_fg).sum()),
        "false_fg_pixels": int((~gt_fg & pred_fg).sum()),
        "num_gt_classes": int((gt_sum > 0).sum()),
        "num_gt_fg_classes": int(present_fg.sum()),
        "num_pred_classes": int((pred_sum > 0).sum()),
        "num_pred_fg_classes": int(((pred_sum > 0) & (np.arange(num_classes) != background_id)).sum()),
        "dominant_pred_class_id": dominant_pred_id,
        "dominant_pred_class_name": cfg["class_names"][dominant_pred_id],
        "dominant_pred_pixels": dominant_pred_pixels,
        "dominant_pred_ratio": safe_div(dominant_pred_pixels, valid_pixels),
    }
    row["failure_group"] = classify_image_failure(row, args, background_id)
    return row


@torch.no_grad()
def collect_eval_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    cfg: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray, float]:
    model.eval()
    num_classes = int(cfg["num_classes"])
    ignore_index = int(cfg["ignore_index"])
    device = cfg["device"]
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    global_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    eval_gt_presence = np.zeros(num_classes, dtype=np.int64)
    eval_pred_presence = np.zeros(num_classes, dtype=np.int64)
    per_image_rows: list[dict] = []
    running_loss = 0.0

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if str(device).startswith("cuda") and bool(cfg.get("amp", False))
        else nullcontext()
    )

    for images, masks, stems, img_paths, mask_paths in tqdm(loader, desc="Collect BiSeNet diagnostics"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

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
            hist = compute_hist_np(pred, gt, num_classes, ignore_index)
            global_hist += hist
            eval_gt_presence += (hist.sum(axis=1) > 0).astype(np.int64)
            eval_pred_presence += (hist.sum(axis=0) > 0).astype(np.int64)
            per_image_rows.append(
                compute_per_image_row(
                    pred=pred,
                    gt=gt,
                    hist=hist,
                    cfg=cfg,
                    stem=stems[idx],
                    img_path=img_paths[idx],
                    mask_path=mask_paths[idx],
                    args=args,
                )
            )

    return global_hist, per_image_rows, eval_gt_presence, eval_pred_presence, running_loss / max(1, len(loader))


def scan_mask_presence(
    samples: list[tuple[str, str, str]],
    num_classes: int,
    ignore_index: int,
    max_items: int | None,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    presence = np.zeros(num_classes, dtype=np.int64)
    pixels = np.zeros(num_classes, dtype=np.int64)
    selected = samples if max_items is None else samples[: max(0, int(max_items))]

    for _, mask_path, _ in tqdm(selected, desc=desc):
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)
        valid = mask != ignore_index
        values = mask[valid]
        values = values[(values >= 0) & (values < num_classes)]
        if values.size == 0:
            continue
        classes, counts = np.unique(values, return_counts=True)
        presence[classes] += 1
        pixels[classes] += counts.astype(np.int64)
    return presence, pixels


def build_class_diagnostics(
    hist: np.ndarray,
    cfg: dict,
    train_presence: np.ndarray,
    eval_gt_presence: np.ndarray,
    eval_pred_presence: np.ndarray,
) -> list[dict]:
    num_classes = int(cfg["num_classes"])
    background_id = int(cfg["background_id"])
    class_names = cfg["class_names"]

    gt_pixels = hist.sum(axis=1).astype(np.int64)
    pred_pixels = hist.sum(axis=0).astype(np.int64)
    tp = np.diag(hist).astype(np.int64)
    fn = gt_pixels - tp
    fp = pred_pixels - tp
    union = gt_pixels + pred_pixels - tp
    recall = tp / np.maximum(gt_pixels, 1)
    precision = tp / np.maximum(pred_pixels, 1)
    iou = tp / np.maximum(union, 1)

    pred_order = np.argsort(-pred_pixels)
    pred_rank = np.empty(num_classes, dtype=np.int64)
    pred_rank[pred_order] = np.arange(1, num_classes + 1)

    rows: list[dict] = []
    for class_id in range(num_classes):
        row_conf = hist[class_id].copy()
        row_conf[class_id] = 0
        top_wrong_pred_id = int(row_conf.argmax()) if row_conf.sum() > 0 else -1
        top_wrong_pred_pixels = int(row_conf[top_wrong_pred_id]) if top_wrong_pred_id >= 0 else 0

        pred_gt_ratio = (
            float("inf")
            if int(gt_pixels[class_id]) == 0 and int(pred_pixels[class_id]) > 0
            else safe_div(int(pred_pixels[class_id]), int(gt_pixels[class_id]))
        )
        if class_id == background_id:
            error_type = "background"
        elif gt_pixels[class_id] == 0 and pred_pixels[class_id] > 0:
            error_type = "predicted_absent_in_gt"
        elif gt_pixels[class_id] > 0 and pred_pixels[class_id] == 0:
            error_type = "never_predicted"
        elif pred_gt_ratio > 5.0 and fp[class_id] > tp[class_id]:
            error_type = "over_predicted_sink"
        elif gt_pixels[class_id] > 0 and pred_gt_ratio < 0.1:
            error_type = "under_predicted"
        elif gt_pixels[class_id] > 0 and recall[class_id] < 0.05:
            error_type = "low_recall"
        elif pred_pixels[class_id] > 0 and precision[class_id] < 0.05:
            error_type = "low_precision"
        else:
            error_type = "partially_learned"

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "train_presence": int(train_presence[class_id]),
                "eval_gt_presence": int(eval_gt_presence[class_id]),
                "eval_pred_presence": int(eval_pred_presence[class_id]),
                "gt_pixels": int(gt_pixels[class_id]),
                "pred_pixels": int(pred_pixels[class_id]),
                "tp": int(tp[class_id]),
                "fp": int(fp[class_id]),
                "fn": int(fn[class_id]),
                "recall": float(recall[class_id]),
                "precision": float(precision[class_id]),
                "iou": float(iou[class_id]),
                "pred_gt_ratio": float(pred_gt_ratio),
                "pred_rank": int(pred_rank[class_id]),
                "top_wrong_pred_id": top_wrong_pred_id,
                "top_wrong_pred_name": class_names[top_wrong_pred_id] if top_wrong_pred_id >= 0 else "",
                "top_wrong_pred_pixels": top_wrong_pred_pixels,
                "error_type": error_type,
            }
        )

    order = {
        "never_predicted": 0,
        "predicted_absent_in_gt": 1,
        "over_predicted_sink": 2,
        "under_predicted": 3,
        "low_recall": 4,
        "low_precision": 5,
        "partially_learned": 6,
        "background": 7,
    }
    return sorted(rows, key=lambda row: (order.get(row["error_type"], 99), row["iou"], -row["gt_pixels"]))


def build_sink_class_analysis(hist: np.ndarray, cfg: dict, top_k: int) -> tuple[list[dict], list[dict]]:
    background_id = int(cfg["background_id"])
    class_names = cfg["class_names"]
    rows: list[dict] = []
    source_rows: list[dict] = []

    for pred_id in range(hist.shape[1]):
        if pred_id == background_id:
            continue

        col = hist[:, pred_id].copy()
        tp = int(col[pred_id])
        col[pred_id] = 0
        fp_pixels = int(col.sum())
        pred_pixels = int(hist[:, pred_id].sum())
        fg_source_mask = (col > 0) & (np.arange(hist.shape[0]) != background_id)
        num_gt_sources = int(fg_source_mask.sum())

        top_source_id = int(col.argmax()) if fp_pixels > 0 else -1
        top_source_pixels = int(col[top_source_id]) if top_source_id >= 0 else 0
        rows.append(
            {
                "pred_class_id": pred_id,
                "pred_class_name": class_names[pred_id],
                "pred_pixels": pred_pixels,
                "tp_pixels": tp,
                "fp_pixels": fp_pixels,
                "fp_ratio_in_prediction": safe_div(fp_pixels, pred_pixels),
                "num_gt_sources": num_gt_sources,
                "top_source_gt_id": top_source_id,
                "top_source_gt_name": class_names[top_source_id] if top_source_id >= 0 else "",
                "top_source_pixels": top_source_pixels,
                "sink_score": float(fp_pixels * math.log1p(max(0, num_gt_sources))),
            }
        )

    rows = sorted(rows, key=lambda row: (row["fp_pixels"], row["num_gt_sources"]), reverse=True)
    top_pred_ids = [int(row["pred_class_id"]) for row in rows[:top_k]]
    for pred_id in top_pred_ids:
        col = hist[:, pred_id].copy()
        col[pred_id] = 0
        fp_total = int(col.sum())
        if fp_total <= 0:
            continue
        source_ids = np.argsort(-col)[:top_k]
        for gt_id in source_ids:
            count = int(col[gt_id])
            if count <= 0:
                continue
            source_rows.append(
                {
                    "pred_class_id": pred_id,
                    "pred_class_name": class_names[pred_id],
                    "gt_source_id": int(gt_id),
                    "gt_source_name": class_names[int(gt_id)],
                    "pixels": count,
                    "share_of_fp": safe_div(count, fp_total),
                    "share_of_pred": safe_div(count, int(hist[:, pred_id].sum())),
                }
            )

    return rows, source_rows


def build_distribution_rows(hist: np.ndarray, cfg: dict, eval_gt_presence: np.ndarray, eval_pred_presence: np.ndarray) -> list[dict]:
    gt_pixels = hist.sum(axis=1).astype(np.int64)
    pred_pixels = hist.sum(axis=0).astype(np.int64)
    total_gt = int(gt_pixels.sum())
    total_pred = int(pred_pixels.sum())
    gt_rank_order = np.argsort(-gt_pixels)
    pred_rank_order = np.argsort(-pred_pixels)
    gt_rank = np.empty(len(gt_pixels), dtype=np.int64)
    pred_rank = np.empty(len(pred_pixels), dtype=np.int64)
    gt_rank[gt_rank_order] = np.arange(1, len(gt_pixels) + 1)
    pred_rank[pred_rank_order] = np.arange(1, len(pred_pixels) + 1)

    rows = []
    for class_id in range(len(gt_pixels)):
        rows.append(
            {
                "class_id": class_id,
                "class_name": cfg["class_names"][class_id],
                "gt_pixels": int(gt_pixels[class_id]),
                "pred_pixels": int(pred_pixels[class_id]),
                "gt_share": safe_div(int(gt_pixels[class_id]), total_gt),
                "pred_share": safe_div(int(pred_pixels[class_id]), total_pred),
                "pred_gt_ratio": safe_div(int(pred_pixels[class_id]), int(gt_pixels[class_id])),
                "gt_rank": int(gt_rank[class_id]),
                "pred_rank": int(pred_rank[class_id]),
                "eval_gt_presence": int(eval_gt_presence[class_id]),
                "eval_pred_presence": int(eval_pred_presence[class_id]),
            }
        )
    return rows


def build_collapse_summary(hist: np.ndarray, cfg: dict, distribution_rows: list[dict]) -> dict:
    background_id = int(cfg["background_id"])
    gt_pixels = hist.sum(axis=1).astype(np.int64)
    pred_pixels = hist.sum(axis=0).astype(np.int64)
    fg_mask = np.arange(len(gt_pixels)) != background_id
    gt_present = (gt_pixels > 0) & fg_mask
    pred_present = (pred_pixels > 0) & fg_mask
    gt_present_and_predicted = gt_present & pred_present
    never_predicted = gt_present & ~pred_present
    pred_probs = pred_pixels[fg_mask].astype(np.float64)
    pred_probs = pred_probs[pred_probs > 0]
    pred_probs = pred_probs / max(pred_probs.sum(), 1.0)
    entropy = float(-(pred_probs * np.log(pred_probs + 1e-12)).sum()) if pred_probs.size else 0.0
    entropy_norm = safe_div(entropy, math.log(max(2, int(gt_present.sum()))))

    top_pred = sorted(distribution_rows, key=lambda row: row["pred_pixels"], reverse=True)
    top_pred_fg = [row for row in top_pred if int(row["class_id"]) != background_id]
    top5_pred_share = sum(float(row["pred_share"]) for row in top_pred_fg[:5])

    return {
        "num_fg_classes_with_gt_pixels": int(gt_present.sum()),
        "num_fg_classes_with_pred_pixels": int(pred_present.sum()),
        "pred_coverage": safe_div(int(pred_present.sum()), int(gt_present.sum())),
        "num_gt_present_fg_classes_predicted": int(gt_present_and_predicted.sum()),
        "gt_class_prediction_coverage": safe_div(int(gt_present_and_predicted.sum()), int(gt_present.sum())),
        "num_gt_present_classes_never_predicted": int(never_predicted.sum()),
        "num_predicted_fg_classes_absent_in_gt": int((pred_present & ~gt_present).sum()),
        "top1_pred_class": top_pred_fg[0]["class_name"] if top_pred_fg else "",
        "top1_pred_class_id": int(top_pred_fg[0]["class_id"]) if top_pred_fg else -1,
        "top1_pred_share": float(top_pred_fg[0]["pred_share"]) if top_pred_fg else 0.0,
        "top5_pred_share": float(top5_pred_share),
        "background_gt_share": safe_div(int(gt_pixels[background_id]), int(gt_pixels.sum())),
        "background_pred_share": safe_div(int(pred_pixels[background_id]), int(pred_pixels.sum())),
        "pred_entropy": entropy,
        "pred_entropy_normalized": entropy_norm,
        "effective_num_predicted_fg_classes": float(math.exp(entropy)) if pred_probs.size else 0.0,
    }


def summarize_image_groups(per_image_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in per_image_rows:
        groups[row["failure_group"]].append(row)

    summary = []
    for group, rows in sorted(groups.items()):
        summary.append(
            {
                "failure_group": group,
                "num_images": len(rows),
                "mean_fg_iou": float(np.mean([row["fg_iou"] for row in rows])) if rows else 0.0,
                "mean_miou_present_fg": float(np.mean([row["miou_present_fg"] for row in rows])) if rows else 0.0,
                "mean_pixel_acc": float(np.mean([row["pixel_acc"] for row in rows])) if rows else 0.0,
                "mean_pred_fg_ratio": float(np.mean([row["pred_fg_ratio"] for row in rows])) if rows else 0.0,
                "mean_dominant_pred_ratio": float(np.mean([row["dominant_pred_ratio"] for row in rows])) if rows else 0.0,
            }
        )
    return summary


def build_image_threshold_summary(per_image_rows: list[dict]) -> dict:
    return {
        "num_images": len(per_image_rows),
        "num_images_miou_eq_0": sum(row["miou_present_fg"] == 0 for row in per_image_rows),
        "num_images_miou_lt_0_05": sum(row["miou_present_fg"] < 0.05 for row in per_image_rows),
        "num_images_miou_lt_0_10": sum(row["miou_present_fg"] < 0.10 for row in per_image_rows),
        "num_images_pixel_acc_lt_0_10": sum(row["pixel_acc"] < 0.10 for row in per_image_rows),
        "num_images_fg_iou_lt_0_10": sum(row["fg_iou"] < 0.10 for row in per_image_rows),
        "num_images_fg_iou_gt_0_35_miou_lt_0_10": sum(
            row["fg_iou"] >= 0.35 and row["miou_present_fg"] < 0.10 for row in per_image_rows
        ),
    }


def build_palette(num_classes: int, background_id: int | None = None) -> np.ndarray:
    cmap = plt.get_cmap("tab20", num_classes)
    palette = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    if background_id is not None and 0 <= background_id < num_classes:
        palette[background_id] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def denorm_image(img: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    image = img.detach().cpu().permute(1, 2, 0).numpy()
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(image * std_arr + mean_arr, 0.0, 1.0)


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    output[valid] = palette[mask[valid]]
    return output


def overlay_mask(image: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_mask(mask, palette).astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * image + alpha * color, 0.0, 1.0)


def make_error_type_map(gt: np.ndarray, pred: np.ndarray, background_id: int, ignore_index: int) -> np.ndarray:
    valid = gt != ignore_index
    gt_bg = gt == background_id
    pred_bg = pred == background_id
    gt_fg = gt != background_id
    pred_fg = pred != background_id

    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[valid & gt_bg & pred_bg] = np.array([0, 0, 0], dtype=np.uint8)
    out[valid & gt_fg & pred_fg & (gt == pred)] = np.array([0, 200, 0], dtype=np.uint8)
    out[valid & gt_fg & pred_fg & (gt != pred)] = np.array([255, 220, 0], dtype=np.uint8)
    out[valid & gt_fg & pred_bg] = np.array([255, 0, 0], dtype=np.uint8)
    out[valid & gt_bg & pred_fg] = np.array([160, 0, 255], dtype=np.uint8)
    out[~valid] = np.array([128, 128, 128], dtype=np.uint8)
    return out


def shorten(name: str, max_len: int = 24) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "."


def plot_gt_pred_distribution(rows: list[dict], output_path: Path, top_k: int) -> None:
    selected = sorted(rows, key=lambda row: row["gt_pixels"] + row["pred_pixels"], reverse=True)[:top_k]
    if not selected:
        return
    ensure_dir(output_path.parent)
    x = np.arange(len(selected))
    width = 0.42
    names = [shorten(row["class_name"], 16) for row in selected]
    gt = np.asarray([max(1, int(row["gt_pixels"])) for row in selected], dtype=np.float64)
    pred = np.asarray([max(1, int(row["pred_pixels"])) for row in selected], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(12, len(selected) * 0.45), 6))
    ax.bar(x - width / 2, gt, width=width, label="GT pixels", color="#2563eb")
    ax.bar(x + width / 2, pred, width=width, label="Pred pixels", color="#f97316")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel("Pixels (log scale)")
    ax.set_title("GT vs predicted pixel distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pred_gt_ratio(class_rows: list[dict], output_path: Path, top_k: int) -> None:
    fg_rows = [row for row in class_rows if row["error_type"] != "background" and row["gt_pixels"] > 0]
    if not fg_rows:
        return
    over = sorted(fg_rows, key=lambda row: row["pred_gt_ratio"], reverse=True)[:top_k]
    under = sorted(fg_rows, key=lambda row: (row["pred_gt_ratio"], -row["gt_pixels"]))[:top_k]
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_k * 0.35)))

    for ax, rows, title, color in [
        (axes[0], over, "Over-predicted / sink candidates", "#dc2626"),
        (axes[1], under, "Under-predicted / never-called classes", "#2563eb"),
    ]:
        names = [shorten(row["class_name"], 22) for row in rows]
        values = [max(float(row["pred_gt_ratio"]), 1e-4) for row in rows]
        ax.barh(names, values, color=color, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("pred_pixels / gt_pixels (log)")
        ax.set_title(title)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sink_classes(rows: list[dict], output_path: Path, top_k: int) -> None:
    selected = rows[:top_k]
    if not selected:
        return
    ensure_dir(output_path.parent)
    names = [shorten(row["pred_class_name"], 22) for row in selected]
    fp = np.asarray([row["fp_pixels"] for row in selected], dtype=np.float64)
    sources = np.asarray([row["num_gt_sources"] for row in selected], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(12, max(6, len(selected) * 0.42)))
    y = np.arange(len(selected))
    ax1.barh(y, fp, color="#dc2626", alpha=0.82, label="FP pixels")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("False-positive pixels")
    ax1.invert_yaxis()
    ax1.set_title("Top sink classes: FP pixels and number of GT sources")

    ax2 = ax1.twiny()
    ax2.plot(sources, y, color="#111827", marker="o", linewidth=1.5, label="GT sources")
    ax2.set_xlabel("Number of foreground GT source classes")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fg_metric_histograms(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    ensure_dir(output_path.parent)
    metrics = [
        ("fg_iou", "Foreground IoU", "#2563eb"),
        ("fg_precision", "Foreground precision", "#16a34a"),
        ("fg_recall", "Foreground recall", "#f97316"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (key, title, color) in zip(axes, metrics):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        ax.hist(values, bins=24, range=(0.0, 1.0), color=color, alpha=0.82, edgecolor="white")
        ax.axvline(values.mean(), color="black", linestyle="--", label=f"mean={values.mean():.3f}")
        ax.axvline(np.median(values), color="gray", linestyle=":", label=f"median={np.median(values):.3f}")
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Image count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fg_iou_vs_class_miou(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    ensure_dir(output_path.parent)
    x = np.asarray([row["fg_iou"] for row in rows], dtype=np.float64)
    y = np.asarray([row["miou_present_fg"] for row in rows], dtype=np.float64)
    c = np.asarray([row["dominant_pred_ratio"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(x, y, c=c, cmap="magma", s=36, alpha=0.82, edgecolors="black", linewidths=0.2)
    ax.axvline(0.35, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Foreground IoU")
    ax.set_ylabel("Class mIoU on present foreground classes")
    ax.set_title("Localization vs class prediction failure")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Dominant predicted class ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recall_vs_presence(class_rows: list[dict], output_path: Path) -> None:
    fg_rows = [row for row in class_rows if row["error_type"] != "background" and row["gt_pixels"] > 0]
    if not fg_rows:
        return
    ensure_dir(output_path.parent)
    x = np.asarray([max(1, row["train_presence"] or row["eval_gt_presence"]) for row in fg_rows], dtype=np.float64)
    y = np.asarray([row["recall"] for row in fg_rows], dtype=np.float64)
    sizes = np.asarray([max(10, math.log10(max(10, row["gt_pixels"])) * 24) for row in fg_rows], dtype=np.float64)
    colors = np.asarray([math.log10(max(1, row["pred_pixels"])) for row in fg_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x, y, c=colors, s=sizes, cmap="viridis", alpha=0.82, edgecolors="black", linewidths=0.25)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Train presence count (fallback to eval GT presence)")
    ax.set_ylabel("Class recall")
    ax.set_title("Per-class recall vs presence")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(pred pixels)")
    worst = sorted(fg_rows, key=lambda row: (row["recall"], -row["gt_pixels"]))[:10]
    for row in worst:
        ax.annotate(shorten(row["class_name"], 14), (max(1, row["train_presence"] or row["eval_gt_presence"]), row["recall"]), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gt_vs_pred_pixels(class_rows: list[dict], output_path: Path) -> None:
    rows = [row for row in class_rows if row["error_type"] != "background"]
    if not rows:
        return
    ensure_dir(output_path.parent)
    gt = np.asarray([max(1, row["gt_pixels"]) for row in rows], dtype=np.float64)
    pred = np.asarray([max(1, row["pred_pixels"]) for row in rows], dtype=np.float64)
    recall = np.asarray([row["recall"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(gt, pred, c=recall, cmap="RdYlGn", s=42, alpha=0.82, edgecolors="black", linewidths=0.25)
    max_value = max(gt.max(), pred.max())
    ax.plot([1, max_value], [1, max_value], color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("GT pixels")
    ax.set_ylabel("Predicted pixels")
    ax.set_title("GT pixels vs predicted pixels")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Recall")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_presence_scatter(class_rows: list[dict], output_path: Path) -> None:
    rows = [row for row in class_rows if row["error_type"] != "background"]
    if not rows:
        return
    ensure_dir(output_path.parent)
    gt = np.asarray([row["eval_gt_presence"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["eval_pred_presence"] for row in rows], dtype=np.float64)
    gt_pixels = np.asarray([math.log10(max(1, row["gt_pixels"])) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(gt, pred, c=gt_pixels, cmap="plasma", s=48, alpha=0.85, edgecolors="black", linewidths=0.25)
    max_value = max(1, gt.max(), pred.max())
    ax.plot([0, max_value], [0, max_value], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("GT image presence count")
    ax.set_ylabel("Prediction image presence count")
    ax.set_title("GT presence vs predicted presence")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(GT pixels)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sink_source_heatmap(source_rows: list[dict], output_path: Path, top_k: int) -> None:
    if not source_rows:
        return
    ensure_dir(output_path.parent)
    pred_ids = []
    gt_ids = []
    pred_names = {}
    gt_names = {}
    for row in source_rows:
        pred_id = int(row["pred_class_id"])
        gt_id = int(row["gt_source_id"])
        if pred_id not in pred_ids:
            pred_ids.append(pred_id)
        if gt_id not in gt_ids:
            gt_ids.append(gt_id)
        pred_names[pred_id] = row["pred_class_name"]
        gt_names[gt_id] = row["gt_source_name"]
    pred_ids = pred_ids[: min(top_k, len(pred_ids))]
    gt_ids = gt_ids[: min(top_k, len(gt_ids))]
    if not pred_ids or not gt_ids:
        return

    value = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    pred_index = {class_id: idx for idx, class_id in enumerate(pred_ids)}
    gt_index = {class_id: idx for idx, class_id in enumerate(gt_ids)}
    for row in source_rows:
        pred_id = int(row["pred_class_id"])
        gt_id = int(row["gt_source_id"])
        if pred_id in pred_index and gt_id in gt_index:
            value[pred_index[pred_id], gt_index[gt_id]] = float(row["share_of_fp"])

    fig, ax = plt.subplots(figsize=(max(8, len(gt_ids) * 0.55), max(5, len(pred_ids) * 0.45)))
    image = ax.imshow(value, cmap="YlOrRd", vmin=0.0, vmax=max(0.1, value.max()))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of sink FP")
    ax.set_xticks(np.arange(len(gt_ids)))
    ax.set_yticks(np.arange(len(pred_ids)))
    ax.set_xticklabels([shorten(gt_names[class_id], 16) for class_id in gt_ids], rotation=45, ha="right")
    ax.set_yticklabels([shorten(pred_names[class_id], 16) for class_id in pred_ids])
    ax.set_xlabel("GT source class")
    ax.set_ylabel("Predicted sink class")
    ax.set_title("GT source distribution for top predicted sink classes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_error_legend(output_path: Path) -> None:
    ensure_dir(output_path.parent)
    labels = [
        ("TN background", [0, 0, 0]),
        ("TP foreground correct class", [0, 200, 0]),
        ("Foreground overlap wrong class", [255, 220, 0]),
        ("Missed foreground", [255, 0, 0]),
        ("False foreground on background", [160, 0, 255]),
        ("Ignore", [128, 128, 128]),
    ]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for idx, (label, color) in enumerate(labels):
        ax.add_patch(plt.Rectangle((0, idx), 0.8, 0.8, color=np.asarray(color) / 255.0))
        ax.text(1.0, idx + 0.4, label, va="center", fontsize=10)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, len(labels))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_group_examples(rows: list[dict], examples_per_group: int) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["failure_group"]].append(row)

    sorters = {
        "background_overprediction": lambda row: (row["pred_fg_ratio"], row["fg_iou"]),
        "sink_class_overprediction": lambda row: (-row["dominant_pred_ratio"], row["miou_present_fg"]),
        "localization_failure": lambda row: (row["fg_iou"], row["miou_present_fg"]),
        "classification_failure": lambda row: (row["miou_present_fg"], -row["fg_iou"]),
        "partially_working": lambda row: (-row["miou_present_fg"], -row["fg_iou"]),
        "mixed_failure": lambda row: (row["miou_present_fg"], row["fg_iou"]),
    }
    selected: dict[str, list[dict]] = {}
    for group, group_rows in groups.items():
        selected[group] = sorted(group_rows, key=sorters.get(group, lambda row: row["miou_present_fg"]))[:examples_per_group]
    return selected


@torch.no_grad()
def save_group_case_grid(
    model: nn.Module,
    rows: list[dict],
    transform: EvalTransform,
    cfg: dict,
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        return
    ensure_dir(output_path.parent)
    palette = build_palette(cfg["num_classes"], cfg["background_id"])
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(18, max(4.2, n * 3.7)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    model.eval()
    for row_idx, row in enumerate(rows):
        image = Image.open(row["img_path"]).convert("RGB")
        mask = Image.open(row["mask_path"])
        image_t, mask_t = transform(image, mask)
        logits = model(image_t.unsqueeze(0).to(cfg["device"]))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy()
        gt = mask_t.detach().cpu().numpy()
        image_np = denorm_image(image_t, cfg["imagenet_mean"], cfg["imagenet_std"])
        error = make_error_type_map(gt, pred, cfg["background_id"], cfg["ignore_index"])

        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title(f"{row['stem']}\nimage")
        axes[row_idx, 1].imshow(overlay_mask(image_np, gt, palette))
        axes[row_idx, 1].set_title("GT overlay")
        axes[row_idx, 2].imshow(overlay_mask(image_np, pred, palette))
        axes[row_idx, 2].set_title(f"Prediction\n{row['dominant_pred_class_name']} {row['dominant_pred_ratio']:.2f}")
        axes[row_idx, 3].imshow(error)
        axes[row_idx, 3].set_title(f"5-type error\nfgIoU={row['fg_iou']:.3f}, mIoU={row['miou_present_fg']:.3f}")

        for col in range(4):
            axes[row_idx, col].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    output_path: Path,
    summary: dict,
    collapse_summary: dict,
    binary_metrics: dict,
    image_threshold_summary: dict,
    group_summary: list[dict],
    class_rows: list[dict],
    sink_rows: list[dict],
) -> None:
    ensure_dir(output_path.parent)
    never_predicted = [row for row in class_rows if row["error_type"] == "never_predicted"]
    over_predicted = [row for row in class_rows if row["error_type"] == "over_predicted_sink"]
    low_recall = [row for row in class_rows if row["error_type"] in {"never_predicted", "under_predicted", "low_recall"}]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("BiSeNet FoodSeg103 Collapse Diagnostics\n")
        f.write("=" * 80 + "\n")
        f.write("\nRun summary\n")
        f.write("-" * 80 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nPrediction collapse check\n")
        f.write("-" * 80 + "\n")
        for key, value in collapse_summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nFG/BG binary evaluation\n")
        f.write("-" * 80 + "\n")
        for key, value in binary_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nImage threshold counts\n")
        f.write("-" * 80 + "\n")
        for key, value in image_threshold_summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nFailure groups\n")
        f.write("-" * 80 + "\n")
        for row in group_summary:
            f.write(
                f"{row['failure_group']}: n={row['num_images']}, "
                f"fg_iou={row['mean_fg_iou']:.4f}, "
                f"class_miou={row['mean_miou_present_fg']:.4f}, "
                f"dominant_ratio={row['mean_dominant_pred_ratio']:.4f}\n"
            )

        f.write("\nTop sink classes\n")
        f.write("-" * 80 + "\n")
        for row in sink_rows[:10]:
            f.write(
                f"{row['pred_class_id']:>3} | {row['pred_class_name']:<26} | "
                f"pred={row['pred_pixels']} | fp={row['fp_pixels']} | "
                f"sources={row['num_gt_sources']} | top_source={row['top_source_gt_name']}\n"
            )

        f.write("\nNever predicted foreground classes with largest GT pixels\n")
        f.write("-" * 80 + "\n")
        for row in sorted(never_predicted, key=lambda item: item["gt_pixels"], reverse=True)[:15]:
            f.write(f"{row['class_id']:>3} | {row['class_name']:<26} | gt={row['gt_pixels']} | train_presence={row['train_presence']}\n")

        f.write("\nOver-predicted sink candidates\n")
        f.write("-" * 80 + "\n")
        for row in sorted(over_predicted, key=lambda item: item["pred_pixels"], reverse=True)[:15]:
            f.write(
                f"{row['class_id']:>3} | {row['class_name']:<26} | "
                f"gt={row['gt_pixels']} | pred={row['pred_pixels']} | ratio={row['pred_gt_ratio']:.2f}\n"
            )

        f.write("\nLarge GT classes with low recall\n")
        f.write("-" * 80 + "\n")
        for row in sorted(low_recall, key=lambda item: item["gt_pixels"], reverse=True)[:15]:
            f.write(
                f"{row['class_id']:>3} | {row['class_name']:<26} | "
                f"gt={row['gt_pixels']} | pred={row['pred_pixels']} | recall={row['recall']:.4f}\n"
            )


def build_loader(cfg: dict, split: str, max_items: int | None) -> tuple[DataLoader, EvalTransform, list[tuple[str, str, str]]]:
    paths = get_paths(cfg)
    if split == "train":
        img_dir = paths["train_img_dir"]
        mask_dir = paths["train_mask_dir"]
    else:
        img_dir = paths["test_img_dir"]
        mask_dir = paths["test_mask_dir"]

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Dataset split folders not found: {img_dir} | {mask_dir}")

    samples = build_samples(img_dir, mask_dir)
    if max_items is not None:
        samples = samples[: max(0, int(max_items))]
    if not samples:
        raise RuntimeError(f"No samples found in {img_dir} and {mask_dir}")

    transform = EvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=cfg.get("test_size"),
    )
    loader = DataLoader(
        FoodSegDataset(samples, transform),
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    return loader, transform, samples


def main() -> None:
    args = parse_args()
    cfg = CFG.copy()

    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    if args.batch_size is not None:
        cfg["eval_batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.background_id is not None:
        cfg["background_id"] = args.background_id

    seed_everything(cfg["seed"])
    paths = get_paths(cfg)
    ckpt_path = resolve_checkpoint_path(cfg, paths, args.ckpt)
    ckpt_cfg = peek_checkpoint_cfg(ckpt_path)
    if "test_size" in ckpt_cfg and args.eval_size is None:
        cfg["test_size"] = tuple(ckpt_cfg["test_size"]) if ckpt_cfg["test_size"] is not None else None
    cfg["test_size"] = parse_size(args.eval_size, cfg.get("test_size"))
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["amp"] = bool(cfg.get("amp", False) and torch.cuda.is_available())
    cfg["pin_memory"] = bool(cfg.get("pin_memory", False) and torch.cuda.is_available())

    mapping = load_mapping(Path(cfg["data_root"]), cfg)
    cfg.update(mapping)
    paths = get_paths(cfg)
    output_dir = Path(args.output_dir) if args.output_dir else paths["work_dir"] / "diagnostics" / args.report_name
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Device       : {cfg['device']}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Data root    : {cfg['data_root']}")
    print(f"Eval split   : {args.split}")
    print(f"Eval size    : {cfg['test_size']}")
    print(f"Output dir   : {output_dir}")

    loader, transform, eval_samples = build_loader(cfg, args.split, args.max_items)
    print(f"Eval samples : {len(eval_samples)}")

    train_presence = np.zeros(cfg["num_classes"], dtype=np.int64)
    train_pixels = np.zeros(cfg["num_classes"], dtype=np.int64)
    if not args.skip_train_presence:
        train_samples = build_samples(paths["train_img_dir"], paths["train_mask_dir"])
        train_presence, train_pixels = scan_mask_presence(
            train_samples,
            cfg["num_classes"],
            cfg["ignore_index"],
            args.presence_max_items,
            desc="Scan train presence",
        )
        save_csv(
            output_dir / "train_presence.csv",
            [
                {
                    "class_id": class_id,
                    "class_name": cfg["class_names"][class_id],
                    "train_presence": int(train_presence[class_id]),
                    "train_pixels": int(train_pixels[class_id]),
                }
                for class_id in range(cfg["num_classes"])
            ],
            ["class_id", "class_name", "train_presence", "train_pixels"],
        )

    model = build_model(cfg, paths).to(cfg["device"])
    load_info = load_checkpoint_flexible(model, ckpt_path, cfg["device"])
    model.eval()
    save_json(load_info, output_dir / "checkpoint_load_info.json")

    hist, per_image_rows, eval_gt_presence, eval_pred_presence, eval_loss = collect_eval_diagnostics(
        model,
        loader,
        cfg,
        args,
    )
    dataset_scores = compute_scores_from_hist(hist)
    dataset_scores["loss"] = eval_loss
    binary_metrics = compute_fg_bg_metrics_from_hist(hist, cfg["background_id"])
    class_rows = build_class_diagnostics(hist, cfg, train_presence, eval_gt_presence, eval_pred_presence)
    distribution_rows = build_distribution_rows(hist, cfg, eval_gt_presence, eval_pred_presence)
    sink_rows, sink_source_rows = build_sink_class_analysis(hist, cfg, args.top_k)
    collapse_summary = build_collapse_summary(hist, cfg, distribution_rows)
    group_summary = summarize_image_groups(per_image_rows)
    image_threshold_summary = build_image_threshold_summary(per_image_rows)

    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_images": len(per_image_rows),
        "num_classes": cfg["num_classes"],
        "background_id": cfg["background_id"],
        "eval_size": cfg["test_size"],
        "eval_loss": dataset_scores["loss"],
        "dataset_aAcc": dataset_scores["aAcc"],
        "dataset_mAcc": dataset_scores["mAcc"],
        "dataset_mIoU": dataset_scores["mIoU"],
        "fg_iou": binary_metrics["fg_iou"],
        "fg_precision": binary_metrics["fg_precision"],
        "fg_recall": binary_metrics["fg_recall"],
        "pred_coverage": collapse_summary["pred_coverage"],
        "gt_class_prediction_coverage": collapse_summary["gt_class_prediction_coverage"],
        "num_gt_present_classes_never_predicted": collapse_summary["num_gt_present_classes_never_predicted"],
        "top1_pred_class": collapse_summary["top1_pred_class"],
        "top1_pred_share": collapse_summary["top1_pred_share"],
        "mean_image_pixel_acc": float(np.mean([row["pixel_acc"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_image_fg_iou": float(np.mean([row["fg_iou"] for row in per_image_rows])) if per_image_rows else 0.0,
        "mean_image_miou_present_fg": float(np.mean([row["miou_present_fg"] for row in per_image_rows])) if per_image_rows else 0.0,
    }

    np.save(output_dir / "confusion_matrix.npy", hist)
    save_json(summary, output_dir / "summary.json")
    save_json(dataset_scores, output_dir / "dataset_scores.json")
    save_json(binary_metrics, output_dir / "binary_fg_bg_metrics.json")
    save_json(collapse_summary, output_dir / "collapse_summary.json")
    save_json(image_threshold_summary, output_dir / "image_threshold_summary.json")
    save_json(group_summary, output_dir / "image_failure_group_summary.json")
    write_summary_text(
        output_dir / "summary.txt",
        summary,
        collapse_summary,
        binary_metrics,
        image_threshold_summary,
        group_summary,
        class_rows,
        sink_rows,
    )

    save_csv(
        output_dir / "per_image_diagnostics.csv",
        per_image_rows,
        [
            "stem",
            "img_path",
            "mask_path",
            "failure_group",
            "pixel_acc",
            "miou_present_fg",
            "macc_present_fg",
            "fg_iou",
            "fg_precision",
            "fg_recall",
            "bg_iou",
            "valid_pixels",
            "gt_fg_pixels",
            "pred_fg_pixels",
            "gt_fg_ratio",
            "pred_fg_ratio",
            "correct_fg_class_pixels",
            "fg_wrong_class_pixels",
            "missed_fg_pixels",
            "false_fg_pixels",
            "num_gt_fg_classes",
            "num_pred_fg_classes",
            "dominant_pred_class_id",
            "dominant_pred_class_name",
            "dominant_pred_pixels",
            "dominant_pred_ratio",
        ],
    )
    save_csv(
        output_dir / "class_diagnostics.csv",
        class_rows,
        [
            "class_id",
            "class_name",
            "error_type",
            "train_presence",
            "eval_gt_presence",
            "eval_pred_presence",
            "gt_pixels",
            "pred_pixels",
            "tp",
            "fp",
            "fn",
            "recall",
            "precision",
            "iou",
            "pred_gt_ratio",
            "pred_rank",
            "top_wrong_pred_id",
            "top_wrong_pred_name",
            "top_wrong_pred_pixels",
        ],
    )
    save_csv(
        output_dir / "prediction_distribution.csv",
        distribution_rows,
        [
            "class_id",
            "class_name",
            "gt_pixels",
            "pred_pixels",
            "gt_share",
            "pred_share",
            "pred_gt_ratio",
            "gt_rank",
            "pred_rank",
            "eval_gt_presence",
            "eval_pred_presence",
        ],
    )
    save_csv(
        output_dir / "sink_class_analysis.csv",
        sink_rows,
        [
            "pred_class_id",
            "pred_class_name",
            "pred_pixels",
            "tp_pixels",
            "fp_pixels",
            "fp_ratio_in_prediction",
            "num_gt_sources",
            "top_source_gt_id",
            "top_source_gt_name",
            "top_source_pixels",
            "sink_score",
        ],
    )
    save_csv(
        output_dir / "sink_source_distribution.csv",
        sink_source_rows,
        [
            "pred_class_id",
            "pred_class_name",
            "gt_source_id",
            "gt_source_name",
            "pixels",
            "share_of_fp",
            "share_of_pred",
        ],
    )
    save_csv(
        output_dir / "image_failure_group_summary.csv",
        group_summary,
        [
            "failure_group",
            "num_images",
            "mean_fg_iou",
            "mean_miou_present_fg",
            "mean_pixel_acc",
            "mean_pred_fg_ratio",
            "mean_dominant_pred_ratio",
        ],
    )

    if not args.no_plots:
        plot_gt_pred_distribution(distribution_rows, plots_dir / "01_gt_vs_pred_pixel_distribution.png", args.top_k * 2)
        plot_pred_gt_ratio(class_rows, plots_dir / "02_pred_gt_ratio.png", args.top_k)
        plot_sink_classes(sink_rows, plots_dir / "03_top_sink_classes.png", args.top_k)
        plot_fg_metric_histograms(per_image_rows, plots_dir / "04_fg_binary_metric_histograms.png")
        plot_fg_iou_vs_class_miou(per_image_rows, plots_dir / "05_fg_iou_vs_class_miou.png")
        plot_recall_vs_presence(class_rows, plots_dir / "06_recall_vs_train_presence.png")
        plot_gt_vs_pred_pixels(class_rows, plots_dir / "07_gt_vs_pred_pixels.png")
        plot_presence_scatter(class_rows, plots_dir / "08_gt_presence_vs_pred_presence.png")
        plot_sink_source_heatmap(sink_source_rows, plots_dir / "09_sink_source_heatmap.png", min(args.top_k, 20))
        save_error_legend(plots_dir / "error_map_legend.png")

        selected = select_group_examples(per_image_rows, args.examples_per_group)
        for group, rows in selected.items():
            save_group_case_grid(
                model,
                rows,
                transform,
                cfg,
                plots_dir / f"cases_{group}.png",
                title=f"{group} examples",
            )

    print("\nDone.")
    print(f"Report directory: {output_dir}")
    print(f"Summary: {output_dir / 'summary.txt'}")
    print(f"Class diagnostics: {output_dir / 'class_diagnostics.csv'}")
    print(f"Sink analysis: {output_dir / 'sink_class_analysis.csv'}")
    print(f"Per-image diagnostics: {output_dir / 'per_image_diagnostics.csv'}")


if __name__ == "__main__":
    main()
