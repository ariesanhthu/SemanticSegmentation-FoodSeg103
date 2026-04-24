import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.foodseg103_ccnet import load_mask_image, load_rgb_image


plt.style.use("seaborn-v0_8-whitegrid")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float = 0.45,
    ignore_index: int | None = None,
) -> np.ndarray:
    color = colorize_mask(mask, palette).astype(np.float32) / 255.0
    mixed = (1.0 - alpha) * image + alpha * color
    if ignore_index is None:
        return np.clip(mixed, 0.0, 1.0)

    output = image.copy()
    valid = mask != ignore_index
    output[valid] = mixed[valid]
    return np.clip(output, 0.0, 1.0)


def error_map(pred: np.ndarray, gt: np.ndarray, ignore_index: int) -> np.ndarray:
    output = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != ignore_index
    output[valid & (pred == gt)] = np.array([22, 163, 74], dtype=np.uint8)
    output[valid & (pred != gt)] = np.array([214, 40, 40], dtype=np.uint8)
    output[~valid] = np.array([0, 0, 0], dtype=np.uint8)
    return output


def shorten(name: str, max_len: int = 28) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def save_worst_classes_plot(
    per_class_rows: list[dict],
    output_path: Path,
    background_id: int,
    top_k: int = 20,
) -> None:
    rows = [
        row
        for row in per_class_rows
        if int(row["class_id"]) != background_id and int(row["gt_pixels"]) > 0
    ]
    rows = sorted(rows, key=lambda row: row["IoU"])[:top_k]
    if not rows:
        return

    ensure_parent(output_path)
    names = [shorten(row["class_name"]) for row in rows]
    ious = [float(row["IoU"]) for row in rows]
    gt_pixels = [int(row["gt_pixels"]) for row in rows]
    colors = plt.cm.RdYlGn(np.clip(np.asarray(ious), 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.45)))
    bars = ax.barh(names, ious, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlim(0.0, max(0.1, max(ious) * 1.15))
    ax.set_xlabel("IoU")
    ax.set_title(f"Worst {len(rows)} classes by IoU")
    ax.invert_yaxis()

    for bar, iou, pixels in zip(bars, ious, gt_pixels):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{iou:.3f} | gt={pixels:,}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top_confusions_plot(
    confusion_rows: list[dict],
    output_path: Path,
    top_k: int = 20,
) -> None:
    rows = confusion_rows[:top_k]
    if not rows:
        return

    ensure_parent(output_path)
    labels = [
        f"{shorten(row['gt_name'], 18)} -> {shorten(row['pred_name'], 18)}"
        for row in rows
    ]
    counts = np.asarray([int(row["count"]) for row in rows], dtype=np.float64)
    colors = plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(rows)))

    fig, ax = plt.subplots(figsize=(13, max(6, len(rows) * 0.42)))
    bars = ax.barh(labels, counts, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Misclassified pixels")
    ax.set_title(f"Top {len(rows)} confusion pairs")
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(count):,}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metric_distribution_plot(per_image_rows: list[dict], output_path: Path) -> None:
    if not per_image_rows:
        return

    ensure_parent(output_path)
    miou = np.asarray([float(row["mIoU_present"]) for row in per_image_rows], dtype=np.float64)
    acc = np.asarray([float(row["pixel_acc"]) for row in per_image_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    specs = [
        (axes[0], miou, "Image-wise mIoU_present", "#1f77b4"),
        (axes[1], acc, "Image-wise pixel accuracy", "#ff7f0e"),
    ]

    for ax, values, title, color in specs:
        ax.hist(values, bins=24, color=color, alpha=0.82, edgecolor="white")
        ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.25, label=f"mean={values.mean():.3f}")
        ax.axvline(np.median(values), color="gray", linestyle=":", linewidth=1.25, label=f"median={np.median(values):.3f}")
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Image count")
        ax.legend(loc="upper left")

    fig.suptitle("Per-image metric distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_frequency_vs_iou_plot(
    per_class_rows: list[dict],
    output_path: Path,
    background_id: int,
    annotate_k: int = 10,
) -> None:
    rows = [
        row
        for row in per_class_rows
        if int(row["class_id"]) != background_id and int(row["gt_pixels"]) > 0
    ]
    if not rows:
        return

    ensure_parent(output_path)
    gt_pixels = np.asarray([max(1, int(row["gt_pixels"])) for row in rows], dtype=np.float64)
    ious = np.asarray([float(row["IoU"]) for row in rows], dtype=np.float64)
    accs = np.asarray([float(row["Acc"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(gt_pixels, ious, c=accs, cmap="viridis", s=42, alpha=0.85, edgecolors="black", linewidths=0.25)
    ax.set_xscale("log")
    ax.set_xlabel("GT pixels (log scale)")
    ax.set_ylabel("IoU")
    ax.set_title("Class frequency vs IoU")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Class accuracy")

    worst_rows = sorted(rows, key=lambda row: row["IoU"])[:annotate_k]
    for row in worst_rows:
        ax.annotate(
            shorten(row["class_name"], 18),
            (max(1, int(row["gt_pixels"])), float(row["IoU"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_heatmap(
    hist: np.ndarray,
    per_class_rows: list[dict],
    output_path: Path,
    background_id: int,
    top_k: int = 12,
) -> None:
    rows = [
        row
        for row in per_class_rows
        if int(row["class_id"]) != background_id and int(row["gt_pixels"]) > 0
    ]
    rows = sorted(rows, key=lambda row: row["IoU"])[:top_k]
    if len(rows) < 2:
        return

    ensure_parent(output_path)
    class_ids = [int(row["class_id"]) for row in rows]
    labels = [shorten(row["class_name"], 18) for row in rows]
    sub_hist = hist[np.ix_(class_ids, class_ids)].astype(np.float64)
    row_sums = sub_hist.sum(axis=1, keepdims=True)
    normalized = np.divide(sub_hist, np.maximum(row_sums, 1.0))

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized, cmap="magma", vmin=0.0, vmax=max(0.25, normalized.max()))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized confusion")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Confusion heatmap on worst-IoU foreground classes")

    if len(labels) <= 12:
        for row_idx in range(len(labels)):
            for col_idx in range(len(labels)):
                value = normalized[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.5 * normalized.max() else "black",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_case_grid(
    model: torch.nn.Module,
    rows: Iterable[dict],
    transform,
    cfg: dict,
    output_path: Path,
    title: str,
) -> None:
    rows = list(rows)
    if not rows:
        return

    ensure_parent(output_path)
    palette = build_palette(cfg["num_classes"], cfg.get("background_id"))
    num_rows = len(rows)
    fig, axes = plt.subplots(num_rows, 4, figsize=(18, max(4.5, num_rows * 3.8)))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    model.eval()

    for row_idx, row in enumerate(rows):
        image = load_rgb_image(row["img_path"])
        mask = load_mask_image(row["mask_path"])
        image_t, mask_t = transform(image, mask)
        logits = model(image_t.unsqueeze(0).to(cfg["device"]))
        pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy()
        gt = mask_t.detach().cpu().numpy()
        image_np = denorm_image(image_t, cfg["imagenet_mean"], cfg["imagenet_std"])

        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title(f"{row['stem']}\nImage")
        axes[row_idx, 1].imshow(overlay_mask(image_np, gt, palette, ignore_index=cfg["ignore_index"]))
        axes[row_idx, 1].set_title("Ground Truth")
        axes[row_idx, 2].imshow(overlay_mask(image_np, pred, palette, ignore_index=None))
        axes[row_idx, 2].set_title(f"Prediction\nmIoU={row['mIoU_present']:.3f}")
        axes[row_idx, 3].imshow(error_map(pred, gt, cfg["ignore_index"]))
        axes[row_idx, 3].set_title("Error Map")

        for col_idx in range(4):
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
