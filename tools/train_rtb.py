"""Standalone training script for BiSeNet-RTB.

The legacy ``tools/train.py`` and ``models/builder.py`` are intentionally not
called here because they expect the original BiSeNetV1 tuple outputs.  This
script mirrors the current runtime parameters while training the RTB model and
loss in an isolated pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.bisenet_rtb_foodseg103 import CFG, get_paths
from datasets.foodseg103 import (
    AlbuTrainTransform,
    EvalTransform,
    FoodSegDataset,
    build_samples,
    set_seed_for_worker,
)
from datasets.foodseg_manifest import FoodSegManifestDataset
from models.backbones.xception39 import build_xception39
from models.bisenet_rtb import BiSeNetRTB
from tools.loss_rtb import RTBLoss
from utils.metrics import compute_segmentation_scores, fast_hist
from utils.misc import ensure_dir, load_checkpoint, save_checkpoint, seed_everything


def _build_grad_scaler(amp_enabled: bool, device: str) -> Any:
    """Create a GradScaler with compatibility across PyTorch AMP APIs."""
    use_cuda = str(device).startswith("cuda")
    enabled = bool(amp_enabled) and use_cuda
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: str, amp_enabled: bool) -> Any:
    """Return an autocast context manager with new and old AMP API support."""
    use_cuda = str(device).startswith("cuda")
    enabled = bool(amp_enabled) and use_cuda
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


class NumpyCompatibleEvalTransform:
    """Adapter that lets ``EvalTransform`` consume PIL images or numpy arrays."""

    def __init__(
        self,
        mean: list[float],
        std: list[float],
        ignore_index: int,
        num_classes: int,
        out_size: tuple[int, int] | None = None,
    ) -> None:
        self.inner = EvalTransform(
            mean=mean,
            std=std,
            ignore_index=ignore_index,
            num_classes=num_classes,
            out_size=out_size,
        )

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
        """Convert numpy inputs to PIL before applying the existing eval transform."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        return self.inner(image, mask)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RTB training pipeline."""
    parser = argparse.ArgumentParser("Train BiSeNet-RTB without modifying old pipeline.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--graph-prior", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gpu",
        "--gpus",
        "--num-gpus",
        dest="num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use. Defaults to 1.",
    )
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="Number of samples for overfitting. 0 indicates full dataset training.",
    )
    return parser.parse_args()


def get_runtime_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    """Create a runtime config copy with CLI overrides."""
    cfg = CFG.copy()

    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_gpus < 1:
        raise ValueError("--gpu/--gpus/--num-gpus must be greater than or equal to 1.")
    cfg["num_gpus"] = args.num_gpus

    cfg["overfit_samples"] = args.overfit

    if args.project_root is not None:
        cfg["project_root"] = args.project_root
    if args.manifest is not None:
        cfg["manifest"] = args.manifest
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = args.init_checkpoint
    if args.graph_prior is not None:
        cfg["graph_prior_path"] = args.graph_prior
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.resume:
        cfg["resume"] = True

    return cfg


def get_runtime_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve paths for the current RTB run."""
    return get_paths(cfg)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the inner module when a DataParallel wrapper is present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_wrap_data_parallel(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """Wrap the model with DataParallel when the requested GPU setup is valid."""
    requested_gpus = int(cfg.get("num_gpus", 1))

    if requested_gpus == 1:
        cfg["num_gpus"] = 1
        return model

    if not torch.cuda.is_available() or not str(cfg["device"]).startswith("cuda"):
        print(f"Requested {requested_gpus} GPUs, but CUDA is not available. Using 1 device.")
        cfg["num_gpus"] = 1
        return model

    available_gpus = torch.cuda.device_count()
    num_gpus = min(requested_gpus, available_gpus)

    if num_gpus < requested_gpus:
        print(
            f"Requested {requested_gpus} GPUs, but only {available_gpus} CUDA device(s) "
            f"are available. Using {num_gpus} GPU(s)."
        )

    if num_gpus <= 1:
        cfg["num_gpus"] = 1
        return model

    batch_size = int(cfg.get("batch_size", 1))
    min_safe_batch_size = num_gpus * 2
    if batch_size < min_safe_batch_size:
        print(
            f"Requested {num_gpus} GPUs with batch_size={batch_size}. "
            f"DataParallel can create per-GPU batch size < 2 and break BatchNorm. "
            "Using 1 GPU instead."
        )
        cfg["num_gpus"] = 1
        return model

    device_ids = list(range(num_gpus))
    cfg["num_gpus"] = num_gpus
    print(f"Using DataParallel on GPUs: {device_ids}")
    return nn.DataParallel(model, device_ids=device_ids)


def build_transforms(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """Build train and evaluation transforms matching the current pipeline."""
    out_size = (
        cfg["train_size"][0]
        if isinstance(cfg["train_size"], (list, tuple))
        else cfg["train_size"]
    )

    train_tf = AlbuTrainTransform(
        size=out_size,
        ignore_index=cfg["ignore_index"],
        background_id=cfg["background_id"],
    )

    eval_tf = NumpyCompatibleEvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=None,
    )
    return train_tf, eval_tf


def build_datasets(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> Tuple[Dataset, Dataset]:
    """Build train/eval datasets from either manifest or benchmark directories."""
    train_tf, eval_tf = build_transforms(cfg)
    overfit_n = int(cfg.get("overfit_samples", 0))

    if cfg.get("manifest"):
        if overfit_n > 0:
            print("=" * 80)
            print(f"RTB OVERFIT MODE ENABLED on manifest: {overfit_n} samples")
            print("=" * 80)

            train_tf = NumpyCompatibleEvalTransform(
                mean=cfg["imagenet_mean"],
                std=cfg["imagenet_std"],
                ignore_index=cfg["ignore_index"],
                num_classes=cfg["num_classes"],
                out_size=cfg["train_size"],
            )

        train_ds = FoodSegManifestDataset(
            manifest_csv=cfg["manifest"],
            data_root=cfg["data_root"],
            train_stage=cfg.get("train_stage", "full"),
            transform=train_tf,
        )

        if overfit_n > 0:
            train_ds.df = train_ds.df.head(overfit_n).copy()
            eval_ds = FoodSegManifestDataset(
                manifest_csv=cfg["manifest"],
                data_root=cfg["data_root"],
                train_stage=cfg.get("train_stage", "full"),
                transform=eval_tf,
            )
            eval_ds.df = train_ds.df.copy()
        else:
            test_samples = build_samples(paths["test_img_dir"], paths["test_mask_dir"])
            eval_ds = FoodSegDataset(test_samples, eval_tf)
    else:
        train_samples = build_samples(paths["train_img_dir"], paths["train_mask_dir"])
        test_samples = build_samples(paths["test_img_dir"], paths["test_mask_dir"])
        eval_ds = FoodSegDataset(test_samples, eval_tf)

        if overfit_n > 0:
            print("=" * 80)
            print(f"RTB OVERFIT MODE ENABLED: Allocating {overfit_n} samples for debugging.")
            print("=" * 80)
            train_samples = train_samples[:overfit_n]
            eval_ds = FoodSegDataset(train_samples, eval_tf)

        train_ds = FoodSegDataset(train_samples, train_tf)

    return train_ds, eval_ds


def build_loaders(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> Tuple[DataLoader, DataLoader]:
    """Build train and evaluation dataloaders."""
    train_ds, eval_ds = build_datasets(cfg, paths)
    overfit_n = int(cfg.get("overfit_samples", 0))

    if cfg.get("manifest") and overfit_n <= 0:
        weights = train_ds.df["sampling_weight"].astype("float32").values
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            drop_last=cfg["drop_last"],
            worker_init_fn=set_seed_for_worker,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
            drop_last=False if overfit_n > 0 else cfg["drop_last"],
            worker_init_fn=set_seed_for_worker,
        )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    return train_loader, eval_loader


def load_graph_prior(cfg: Dict[str, Any]) -> torch.Tensor:
    """Load graph prior from disk or return identity prior if unavailable."""
    path = cfg.get("graph_prior_path")
    num_classes = cfg["num_classes"]

    if path is None or str(path).strip() == "":
        print("[WARN] graph_prior_path is empty. Using identity prior.")
        return torch.eye(num_classes, dtype=torch.float32)

    path = Path(path)
    if not path.exists():
        print(f"[WARN] graph prior not found: {path}. Using identity prior.")
        return torch.eye(num_classes, dtype=torch.float32)

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if "prior" in obj:
            return obj["prior"].float()
        if "graph_prior" in obj:
            return obj["graph_prior"].float()

    if torch.is_tensor(obj):
        return obj.float()

    raise ValueError(f"Unsupported graph prior format: {path}")


def build_model_and_optim(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    """Build RTB model, criterion, optimizer, and AMP scaler."""
    backbone = build_xception39(
        pretrained=cfg["backbone_pretrained"],
        pretrained_path=(
            None
            if paths["backbone_pretrained_path"] is None
            else str(paths["backbone_pretrained_path"])
        ),
        strict=cfg["backbone_strict_load"],
    )

    graph_prior = load_graph_prior(cfg)

    model = BiSeNetRTB(
        num_classes=cfg["num_classes"],
        backbone=backbone,
        background_id=cfg["background_id"],
        graph_prior=graph_prior,
        tex_ch=cfg["rtb_tex_ch"],
        graph_dim=cfg["rtb_graph_dim"],
        graph_layers=cfg["rtb_graph_layers"],
        graph_eta=cfg["rtb_graph_eta"],
        graph_xi=cfg["rtb_graph_xi"],
    ).to(cfg["device"])
    model = maybe_wrap_data_parallel(model, cfg)

    criterion = RTBLoss(
        num_classes=cfg["num_classes"],
        ignore_index=cfg["ignore_index"],
        background_id=cfg["background_id"],
        aux_weight=cfg["rtb_aux_weight"],
        pre_weight=cfg["rtb_pre_weight"],
        edge_weight=cfg["rtb_edge_weight"],
        presence_weight=cfg["rtb_presence_weight"],
        boundary_width=cfg["rtb_boundary_width"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )

    scaler = _build_grad_scaler(amp_enabled=bool(cfg["amp"]), device=str(cfg["device"]))
    return model, criterion, optimizer, scaler


def poly_lr(base_lr: float, cur_iter: int, max_iter: int, power: float = 0.9) -> float:
    """Polynomial learning-rate schedule."""
    return base_lr * (1.0 - float(cur_iter) / float(max_iter)) ** power


def maybe_resume(
    cfg: Dict[str, Any],
    last_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> Tuple[int, int, float]:
    """Resume model, optimizer, and scaler state from the last checkpoint."""
    start_epoch = 0
    global_iter = 0
    best_miou = -1.0

    if cfg["resume"] and last_path.exists():
        ckpt = load_checkpoint(
            last_path,
            unwrap_model(model),
            optimizer,
            scaler,
            map_location=cfg["device"],
        )
        start_epoch = ckpt["epoch"] + 1
        global_iter = ckpt.get("global_iter", 0)
        best_miou = ckpt.get("best_miou", -1.0)
        print(f"Resumed RTB from {last_path} at epoch={start_epoch}")

    return start_epoch, global_iter, best_miou


def load_model_weights_only(model: nn.Module, ckpt_path: str, device: str) -> None:
    """Load matching model weights from a checkpoint without optimizer state."""
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model_state = model.state_dict()
    loaded = {}
    skipped = []

    for key, value in state.items():
        clean_key = key.replace("module.", "")
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            loaded[clean_key] = value
        else:
            skipped.append(key)

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)

    print(f"Loaded weights only from: {ckpt_path}")
    print(f"Loaded keys: {len(loaded)}")
    print(f"Skipped keys: {len(skipped)}")
    if skipped:
        print("Skipped examples:", skipped[:10])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Dict[str, Any],
    criterion: nn.Module,
) -> Dict[str, float]:
    """Evaluate RTB model with benchmark-style segmentation metrics."""
    model.eval()

    hist = torch.zeros(
        (cfg["num_classes"], cfg["num_classes"]),
        device=cfg["device"],
    )
    running_loss = 0.0

    for images, masks, *_ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(cfg["device"], non_blocking=True)
        masks = masks.to(cfg["device"], non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += float(loss.item())

        preds = logits.argmax(1)
        hist += fast_hist(preds, masks, cfg["num_classes"], cfg["ignore_index"])

    scores = compute_segmentation_scores(hist)
    scores["loss"] = running_loss / max(1, len(loader))
    return scores


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: Dict[str, Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_iter: int,
    max_iter: int,
) -> Tuple[float, int]:
    """Train RTB model for one epoch."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"RTB Train {epoch:03d}")

    for step, (images, masks, stems, *_rest) in enumerate(pbar):
        images = images.to(cfg["device"], non_blocking=True)
        masks = masks.to(cfg["device"], non_blocking=True)

        if int(cfg.get("overfit_samples", 0)) > 0:
            cur_lr = cfg["lr"]
        else:
            cur_lr = poly_lr(cfg["lr"], global_iter, max_iter, cfg["poly_power"])

        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device=str(cfg["device"]), amp_enabled=bool(cfg["amp"])):
            outputs = model(images)
            logits = outputs["logits"]
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        if cfg.get("grad_clip_norm") is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        global_iter += 1

        if step == 0:
            print("-" * 80)
            print(f"RTB Epoch {epoch:03d} step 0")
            print("images:", tuple(images.shape))
            print("masks :", tuple(masks.shape))
            print("logits:", tuple(logits.shape))
            print("edge  :", tuple(outputs["edge_logits"].shape))
            print("pres  :", tuple(outputs["presence_logits"].shape))
            print("lr    :", cur_lr)
            print("loss  :", float(loss.item()))
            print("stems :", list(stems[:4]))
            print("mask unique:", torch.unique(masks).detach().cpu().tolist()[:50])
            print("pred unique:", torch.unique(logits.argmax(1)).detach().cpu().tolist()[:50])
            print("-" * 80)

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.6f}")

    return running_loss / max(1, len(loader)), global_iter


def run_training(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> None:
    """Run full RTB training, evaluation, CSV logging, and checkpointing."""
    ensure_dir(paths["work_dir"])

    last_path = paths["work_dir"] / cfg["save_last_name"]
    best_path = paths["work_dir"] / cfg["save_best_name"]
    csv_path = paths["work_dir"] / "metrics_rtb.csv"

    start_epoch, global_iter, best_miou = maybe_resume(
        cfg=cfg,
        last_path=last_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )

    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "mIoU", "mAcc", "aAcc"])

    max_iter = max(1, cfg["epochs"] * len(train_loader))

    for epoch in range(start_epoch, cfg["epochs"]):
        train_loss, global_iter = train_one_epoch(
            model=model,
            loader=train_loader,
            cfg=cfg,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_iter=global_iter,
            max_iter=max_iter,
        )

        print(f"RTB Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        val_loss, miou, macc, aacc = -1.0, -1.0, -1.0, -1.0

        if (epoch + 1) % cfg["eval_every"] == 0:
            scores = evaluate(model, eval_loader, cfg, criterion)
            val_loss = scores["loss"]
            miou = scores["mIoU"]
            macc = scores["mAcc"]
            aacc = scores["aAcc"]

            print(
                f"RTB Eval {epoch:03d} | "
                f"loss={val_loss:.4f} | "
                f"mIoU={miou:.4f} | "
                f"mAcc={macc:.4f} | "
                f"aAcc={aacc:.4f}"
            )

            if miou > best_miou:
                best_miou = miou
                save_checkpoint(
                    best_path,
                    epoch,
                    global_iter,
                    best_miou,
                    unwrap_model(model),
                    optimizer,
                    scaler,
                    cfg,
                )
                print(f"Saved new RTB best checkpoint to {best_path}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, miou, macc, aacc])

        save_checkpoint(
            last_path,
            epoch,
            global_iter,
            best_miou,
            unwrap_model(model),
            optimizer,
            scaler,
            cfg,
        )


def main() -> None:
    """Program entry point for RTB training/evaluation."""
    args = parse_args()

    cfg = get_runtime_cfg(args)
    paths = get_runtime_paths(cfg)

    seed_everything(cfg["seed"])
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True

    ensure_dir(paths["work_dir"])

    train_loader, eval_loader = build_loaders(cfg, paths)
    model, criterion, optimizer, scaler = build_model_and_optim(cfg, paths)

    if cfg.get("init_checkpoint"):
        load_model_weights_only(
            unwrap_model(model),
            cfg["init_checkpoint"],
            cfg["device"],
        )

    if args.eval_only:
        scores = evaluate(model, eval_loader, cfg, criterion)
        print(scores)
        return

    run_training(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        eval_loader=eval_loader,
        cfg=cfg,
        paths=paths,
    )


if __name__ == "__main__":
    main()
