import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.ccnet_foodseg103 import get_paths
from datasets.foodseg103_ccnet import EvalTransform, FoodSegDataset, build_samples, resolve_dataset_meta
from models.ccnet import CCNetSeg
from utils.misc import ensure_dir, load_checkpoint, save_json

from .plots import (
    save_case_grid,
    save_confusion_heatmap,
    save_frequency_vs_iou_plot,
    save_metric_distribution_plot,
    save_top_confusions_plot,
    save_worst_classes_plot,
)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_hist_np(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    valid = target != ignore_index
    pred = pred[valid].astype(np.int64, copy=False)
    target = target[valid].astype(np.int64, copy=False)
    valid_cls = (target >= 0) & (target < num_classes)
    indices = num_classes * target[valid_cls] + pred[valid_cls]
    return np.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_scores_from_hist(hist: np.ndarray) -> dict:
    hist = hist.astype(np.float64, copy=False)
    diag = np.diag(hist)
    gt_sum = hist.sum(axis=1)
    pred_sum = hist.sum(axis=0)
    total = hist.sum()

    acc_cls = diag / np.maximum(gt_sum, 1.0)
    iou = diag / np.maximum(gt_sum + pred_sum - diag, 1.0)
    valid = gt_sum > 0

    return {
        "aAcc": safe_div(diag.sum(), total),
        "mAcc": float(acc_cls[valid].mean()) if valid.any() else 0.0,
        "mIoU": float(iou[valid].mean()) if valid.any() else 0.0,
        "Acc_per_class": acc_cls.tolist(),
        "IoU_per_class": iou.tolist(),
        "valid_class_mask": valid.tolist(),
    }


def compute_per_image_row(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int,
    stem: str,
    img_path: str,
    mask_path: str,
) -> dict:
    hist = compute_hist_np(pred, gt, num_classes, ignore_index)
    diag = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pred_sum = hist.sum(axis=0).astype(np.float64)
    valid_classes = gt_sum > 0
    union = gt_sum + pred_sum - diag
    iou = diag / np.maximum(union, 1.0)

    valid_pixels = gt != ignore_index
    pixel_acc = safe_div((pred[valid_pixels] == gt[valid_pixels]).sum(), valid_pixels.sum())
    miou_present = float(iou[valid_classes].mean()) if valid_classes.any() else 0.0
    macc_present = float((diag / np.maximum(gt_sum, 1.0))[valid_classes].mean()) if valid_classes.any() else 0.0

    return {
        "stem": stem,
        "img_path": img_path,
        "mask_path": mask_path,
        "pixel_acc": pixel_acc,
        "mIoU_present": miou_present,
        "mAcc_present": macc_present,
        "num_present_classes": int(valid_classes.sum()),
        "valid_pixels": int(valid_pixels.sum()),
    }


def build_per_class_rows(hist: np.ndarray, cfg: dict) -> list[dict]:
    hist = hist.astype(np.float64, copy=False)
    rows: list[dict] = []
    for class_id in range(cfg["num_classes"]):
        gt_pixels = int(hist[class_id, :].sum())
        pred_pixels = int(hist[:, class_id].sum())
        tp = int(hist[class_id, class_id])
        union = gt_pixels + pred_pixels - tp
        rows.append(
            {
                "class_id": class_id,
                "class_name": cfg["class_names"][class_id],
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "tp_pixels": tp,
                "IoU": safe_div(tp, union),
                "Acc": safe_div(tp, gt_pixels),
            }
        )
    return sorted(rows, key=lambda row: (row["IoU"], row["class_id"]))


def build_confusion_rows(hist: np.ndarray, cfg: dict) -> list[dict]:
    rows: list[dict] = []
    background_id = int(cfg["background_id"])
    for gt_id in range(cfg["num_classes"]):
        for pred_id in range(cfg["num_classes"]):
            if gt_id == pred_id:
                continue
            if gt_id == background_id or pred_id == background_id:
                continue
            count = int(hist[gt_id, pred_id])
            if count <= 0:
                continue
            rows.append(
                {
                    "gt_id": gt_id,
                    "gt_name": cfg["class_names"][gt_id],
                    "pred_id": pred_id,
                    "pred_name": cfg["class_names"][pred_id],
                    "count": count,
                }
            )
    return sorted(rows, key=lambda row: row["count"], reverse=True)


def analyze_error_patterns(
    per_image_rows: list[dict],
    per_class_rows: list[dict],
    confusion_rows: list[dict],
    background_id: int,
) -> list[dict]:
    rows: list[dict] = []
    foreground_rows = [
        row for row in per_class_rows if int(row["class_id"]) != background_id and int(row["gt_pixels"]) > 0
    ]

    rare_bad = [row for row in sorted(foreground_rows, key=lambda row: row["gt_pixels"])[:15] if row["IoU"] < 0.05]
    if rare_bad:
        rows.append(
            {
                "pattern": "rare-class fail",
                "evidence": f"{len(rare_bad)} rare classes have IoU < 0.05",
                "direction": "class-balanced sampling or reweighting and repeat-factor style training",
            }
        )

    if confusion_rows:
        rows.append(
            {
                "pattern": "texture or semantic confusion",
                "evidence": "top confusion pairs repeat across many pixels",
                "direction": "hard-example mining by class pair or stronger local-detail modeling",
            }
        )

    if per_image_rows:
        mean_pixel_acc = float(np.mean([row["pixel_acc"] for row in per_image_rows]))
        mean_miou_present = float(np.mean([row["mIoU_present"] for row in per_image_rows]))
        if mean_pixel_acc - mean_miou_present > 0.20:
            rows.append(
                {
                    "pattern": "boundary or local-detail fail",
                    "evidence": "pixel accuracy is much higher than image-wise mIoU_present",
                    "direction": "boundary-aware supervision or better high-resolution refinement",
                }
            )

        hard_many = sorted(per_image_rows, key=lambda row: row["mIoU_present"])[:20]
        if hard_many and np.mean([row["num_present_classes"] for row in hard_many]) >= 4:
            rows.append(
                {
                    "pattern": "co-occurrence / multi-ingredient fail",
                    "evidence": "hard images contain many simultaneous ingredient classes",
                    "direction": "stronger relation-aware decoding or crop policy that preserves multiple ingredients",
                }
            )

    very_low_iou = [row for row in foreground_rows if row["IoU"] < 0.03]
    if very_low_iou:
        rows.append(
            {
                "pattern": "small-object or thin-structure fail",
                "evidence": f"{len(very_low_iou)} foreground classes have IoU < 0.03",
                "direction": "higher-resolution inference or better feature fusion for small objects",
            }
        )

    return rows


def build_transform(cfg: dict) -> EvalTransform:
    return EvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=cfg["eval_size"],
    )


def build_loader(cfg: dict, split: str, batch_size: int | None, max_items: int | None) -> tuple[DataLoader, EvalTransform]:
    paths = get_paths(cfg)
    if split == "train":
        img_dir = paths["train_img_dir"]
        mask_dir = paths["train_mask_dir"]
    else:
        img_dir = paths["test_img_dir"]
        mask_dir = paths["test_mask_dir"]

    samples = build_samples(
        img_dir,
        mask_dir,
        validate_files=bool(cfg.get("validate_samples", True)),
    )
    if max_items is not None:
        samples = samples[: max(0, int(max_items))]

    transform = build_transform(cfg)
    loader = DataLoader(
        FoodSegDataset(
            samples,
            transform,
            max_decode_retries=int(cfg.get("max_decode_retries", 16)),
        ),
        batch_size=batch_size or (1 if cfg["eval_size"] is None else cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    return loader, transform


def build_model(cfg: dict, checkpoint_path: Path) -> CCNetSeg:
    model = CCNetSeg(
        num_classes=cfg["num_classes"],
        backbone_pretrained=False,
        output_stride=cfg["output_stride"],
        channels=cfg["cc_channels"],
        recurrence=cfg["cc_recurrence"],
        use_aux=cfg["use_aux_head"],
        dropout=cfg["dropout"],
        align_corners=cfg["align_corners"],
    ).to(cfg["device"])
    load_checkpoint(checkpoint_path, model=model, map_location=cfg["device"])
    model.eval()
    return model


@torch.no_grad()
def run_dataset_eval(model: nn.Module, loader: DataLoader, cfg: dict) -> tuple[dict, np.ndarray]:
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    global_hist = np.zeros((cfg["num_classes"], cfg["num_classes"]), dtype=np.int64)
    per_image_rows: list[dict] = []
    running_loss = 0.0

    for images, masks, stems, img_paths, mask_paths in loader:
        images = images.to(cfg["device"], non_blocking=True)
        masks = masks.to(cfg["device"], non_blocking=True)
        logits = model(images)
        running_loss += float(criterion(logits, masks).item())

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        gt_batch = masks.detach().cpu().numpy()

        for idx in range(len(stems)):
            row = compute_per_image_row(
                pred=preds[idx],
                gt=gt_batch[idx],
                num_classes=cfg["num_classes"],
                ignore_index=cfg["ignore_index"],
                stem=stems[idx],
                img_path=img_paths[idx],
                mask_path=mask_paths[idx],
            )
            global_hist += compute_hist_np(
                preds[idx],
                gt_batch[idx],
                num_classes=cfg["num_classes"],
                ignore_index=cfg["ignore_index"],
            )
            per_image_rows.append(row)

    dataset_scores = compute_scores_from_hist(global_hist)
    dataset_scores["loss"] = running_loss / max(1, len(loader))
    return {"per_image_rows": per_image_rows, "dataset_scores": dataset_scores}, global_hist


def prepare_output_dir(work_dir: Path, checkpoint_path: Path, report_name: str | None) -> Path:
    name = report_name or f"{checkpoint_path.stem}_report"
    output_dir = work_dir / "reports" / name
    ensure_dir(output_dir)
    ensure_dir(output_dir / "plots")
    return output_dir


def write_summary_text(output_path: Path, summary: dict, per_class_rows: list[dict], confusion_rows: list[dict], per_image_rows: list[dict]) -> None:
    worst_classes = per_class_rows[:10]
    worst_images = sorted(per_image_rows, key=lambda row: row["mIoU_present"])[:10]
    top_pairs = confusion_rows[:10]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("CCNet FoodSeg103 Evaluation Summary\n")
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


def generate_ccnet_report(
    cfg: dict,
    checkpoint_path: Path,
    split: str = "test",
    report_name: str | None = None,
    max_items: int | None = None,
    batch_size: int | None = None,
    num_worst_cases: int = 8,
    num_best_cases: int = 8,
    top_k_classes: int = 20,
    top_k_confusions: int = 20,
) -> dict:
    cfg = resolve_dataset_meta(cfg.copy())
    paths = get_paths(cfg)
    output_dir = prepare_output_dir(paths["work_dir"], checkpoint_path, report_name)

    loader, transform = build_loader(cfg, split=split, batch_size=batch_size, max_items=max_items)
    model = build_model(cfg, checkpoint_path)
    eval_result, hist = run_dataset_eval(model, loader, cfg)

    per_image_rows = eval_result["per_image_rows"]
    dataset_scores = eval_result["dataset_scores"]
    per_class_rows = build_per_class_rows(hist, cfg)
    confusion_rows = build_confusion_rows(hist, cfg)
    error_rows = analyze_error_patterns(
        per_image_rows=per_image_rows,
        per_class_rows=per_class_rows,
        confusion_rows=confusion_rows,
        background_id=int(cfg["background_id"]),
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
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

    save_json(summary, output_dir / "summary.json")
    write_summary_text(output_dir / "summary.txt", summary, per_class_rows, confusion_rows, per_image_rows)
    save_json(dataset_scores, output_dir / "dataset_scores.json")
    save_json(error_rows, output_dir / "error_analysis.json")

    save_csv(
        output_dir / "per_image_metrics.csv",
        per_image_rows,
        ["stem", "img_path", "mask_path", "pixel_acc", "mIoU_present", "mAcc_present", "num_present_classes", "valid_pixels"],
    )
    save_csv(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        ["class_id", "class_name", "gt_pixels", "pred_pixels", "tp_pixels", "IoU", "Acc"],
    )
    save_csv(
        output_dir / "top_confusion_pairs.csv",
        confusion_rows,
        ["gt_id", "gt_name", "pred_id", "pred_name", "count"],
    )
    save_csv(
        output_dir / "error_analysis.csv",
        error_rows,
        ["pattern", "evidence", "direction"],
    )

    plots_dir = output_dir / "plots"
    save_worst_classes_plot(
        per_class_rows,
        plots_dir / "01_worst_classes_iou.png",
        background_id=int(cfg["background_id"]),
        top_k=top_k_classes,
    )
    save_top_confusions_plot(
        confusion_rows,
        plots_dir / "02_top_confusion_pairs.png",
        top_k=top_k_confusions,
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
        hist,
        per_class_rows,
        plots_dir / "05_confusion_heatmap_worst_classes.png",
        background_id=int(cfg["background_id"]),
        top_k=min(12, top_k_classes),
    )

    worst_rows = sorted(per_image_rows, key=lambda row: row["mIoU_present"])[:num_worst_cases]
    best_rows = sorted(per_image_rows, key=lambda row: row["mIoU_present"], reverse=True)[:num_best_cases]
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

    report = {
        "output_dir": str(output_dir),
        "summary": summary,
        "dataset_scores": dataset_scores,
        "num_confusion_pairs": len(confusion_rows),
        "num_error_patterns": len(error_rows),
    }
    save_json(report, output_dir / "report.json")
    return report
