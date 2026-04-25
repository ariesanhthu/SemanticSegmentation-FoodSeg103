"""Evaluate the CCNet-ResNet50 model on the FoodSeg103 test set.

This script loads a trained CCNet checkpoint, runs inference on the
FoodSeg103 test split, computes standard segmentation metrics
(aAcc, mAcc, mIoU) plus the average cross-entropy loss, and saves
the results as JSON to the evaluation directory.

Usage::

    python tools/eval_ccnet.py [--data-root DIR] [--work-dir DIR] [--checkpoint PATH]
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.ccnet_foodseg103 import CFG, get_paths
from datasets.foodseg103_ccnet import (
    EvalTransform,
    FoodSegDataset,
    build_samples,
    resolve_dataset_meta,
)
from models.ccnet import CCNetSeg
from utils.metrics import compute_segmentation_scores, fast_hist
from utils.misc import ensure_dir, load_checkpoint, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CCNet evaluation.

    Returns:
        argparse.Namespace: Parsed arguments with optional overrides
            for ``data_root``, ``work_dir``, and ``checkpoint``.
    """
    parser = argparse.ArgumentParser(description="Evaluate FoodSeg103 CCNet-ResNet50 baseline.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", type=str, default=None, help="Override work directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to best_miou.pth in work_dir.",
    )
    return parser.parse_args()


def get_runtime_cfg(args: argparse.Namespace) -> dict:
    """Build the runtime configuration dict from defaults and CLI overrides.

    Applies any user-specified ``--data-root`` or ``--work-dir`` overrides
    on top of the base :data:`CFG`, then resolves dataset metadata
    (class names, number of classes, etc.) via :func:`resolve_dataset_meta`.

    Args:
        args: Parsed CLI arguments.

    Returns:
        dict: Fully resolved configuration dictionary.
    """
    cfg = CFG.copy()
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    return resolve_dataset_meta(cfg)


def to_serializable(obj):
    """Convert a value to a JSON-serializable type.

    If *obj* is a :class:`torch.Tensor`, it is moved to the CPU and
    converted to a nested Python list.  All other types pass through
    unchanged.

    Args:
        obj: Any Python object, potentially a tensor.

    Returns:
        A JSON-safe representation of *obj*.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def format_scores(scores: dict) -> dict:
    """Make every value in *scores* JSON-serializable.

    Args:
        scores: Mapping of metric names to values (which may include tensors).

    Returns:
        dict: A new dict with all tensors converted to Python lists.
    """
    return {key: to_serializable(value) for key, value in scores.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: dict, criterion: nn.Module) -> dict:
    """Run the model over the full data loader and compute segmentation metrics.

    Accumulates a confusion histogram across all batches, then derives
    overall segmentation scores (aAcc, mAcc, mIoU) together with the
    mean batch loss.

    Args:
        model: The segmentation model (already on the correct device).
        loader: DataLoader yielding ``(images, masks, ...)`` tuples.
        cfg: Runtime configuration dict (must include ``num_classes``,
            ``device``, and ``ignore_index``).
        criterion: Loss function (e.g. ``CrossEntropyLoss``).

    Returns:
        dict: Segmentation scores including ``aAcc``, ``mAcc``, ``mIoU``,
            per-class IoU/Acc arrays, and the average ``loss``.
    """
    model.eval()
    hist = torch.zeros((cfg["num_classes"], cfg["num_classes"]), device=cfg["device"])
    running_loss = 0.0

    for images, masks, *_ in loader:
        images = images.to(cfg["device"], non_blocking=True)
        masks = masks.to(cfg["device"], non_blocking=True)

        logits = model(images)
        running_loss += float(criterion(logits, masks).item())
        preds = logits.argmax(dim=1)
        hist += fast_hist(preds, masks, cfg["num_classes"], cfg["ignore_index"])

    scores = compute_segmentation_scores(hist)
    scores["loss"] = running_loss / max(1, len(loader))
    return scores


def main() -> None:
    """Entry point for CCNet evaluation.

    Orchestrates the full pipeline:

    1. Parse CLI arguments and build runtime config.
    2. Construct the test dataset and data loader.
    3. Instantiate :class:`CCNetSeg` and load the checkpoint.
    4. Call :func:`evaluate` to compute metrics.
    5. Save results to ``eval_dir / eval_only_latest.json`` and print them.
    """
    args = parse_args()
    cfg = get_runtime_cfg(args)
    seed_everything(cfg["seed"])

    paths = get_paths(cfg)
    samples = build_samples(
        paths["test_img_dir"],
        paths["test_mask_dir"],
        validate_files=bool(cfg.get("validate_samples", True)),
    )

    transform = EvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=cfg["eval_size"],
    )
    loader = DataLoader(
        FoodSegDataset(
            samples,
            transform,
            max_decode_retries=int(cfg.get("max_decode_retries", 16)),
        ),
        batch_size=1 if cfg["eval_size"] is None else cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

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
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])

    ckpt_path = Path(args.checkpoint) if args.checkpoint else paths["work_dir"] / cfg["save_best_name"]
    load_checkpoint(ckpt_path, model=model, map_location=cfg["device"])

    scores = evaluate(model, loader, cfg, criterion)
    ensure_dir(paths["eval_dir"])
    save_json(
        {
            "type": "eval_only",
            "checkpoint": str(ckpt_path),
            **scores,
        },
        paths["eval_dir"] / "eval_only_latest.json",
    )
    print(json.dumps(format_scores(scores), indent=2))


if __name__ == "__main__":
    main()
