from pathlib import Path
import argparse
import sys
from typing import Any, Dict, Tuple
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
# Project imports are added after ROOT is appended to sys.path below.

# =============================================================================
# Project import setup
# =============================================================================
# Cho phép chạy file trực tiếp từ thư mục tools/ mà vẫn import được module gốc.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from models.builder import build_model
from tools.loss import CombinedLoss
from configs.bisenet_foodseg103 import CFG, get_paths
from datasets.foodseg103 import (
    FoodSegDataset,
    RandomResizeCrop,
    AlbuTrainTransform,
    EvalTransform,
    build_samples,
    set_seed_for_worker,
)
from datasets.foodseg_manifest import FoodSegManifestDataset
from utils.metrics import fast_hist, compute_segmentation_scores
from utils.misc import seed_everything, ensure_dir, load_checkpoint, save_checkpoint


def _build_grad_scaler(amp_enabled: bool, device: str):
    """Create GradScaler using new AMP API with backward compatibility.

    Args:
        amp_enabled: Whether AMP is enabled in runtime config.
        device: Runtime device string, e.g. ``"cuda"`` or ``"cpu"``.
    Returns:
        A GradScaler instance compatible with the current PyTorch version.
    Raises:
        None.
    """
    use_cuda = str(device).startswith("cuda")
    enabled = bool(amp_enabled) and use_cuda
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: str, amp_enabled: bool):
    """Return autocast context manager using modern AMP API when available.

    Args:
        device: Runtime device string.
        amp_enabled: Whether AMP should be enabled.
    Returns:
        Autocast context manager object.
    Raises:
        None.
    """
    use_cuda = str(device).startswith("cuda")
    enabled = bool(amp_enabled) and use_cuda
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training/evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
        Hiện tại hỗ trợ:
        - --eval-only: chỉ chạy evaluation trên tập eval/test.
    """
    parser = argparse.ArgumentParser(
        description="Train or evaluate BiSeNetV1 on FoodSeg103 benchmark."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only without training.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override static dataset path.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Override static checkpoint directory.",
    )
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override global batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total training epochs.",
    )
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


# =============================================================================
# Runtime config / path helpers
# =============================================================================
def get_runtime_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create a runtime copy of global config.

    Dùng `copy()` để tránh sửa trực tiếp CFG global khi đang chạy.

    Returns
    -------
    Dict[str, Any]
        Runtime configuration dictionary.
    """
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

    if args.lr is not None:
        cfg["lr"] = args.lr

    # CLI --resume chỉ bật resume khi thật sự truyền
    if args.resume:
        cfg["resume"] = True
    
    return cfg


def get_runtime_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Resolve all filesystem paths used by the current run.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Runtime configuration.

    Returns
    -------
    Dict[str, Path]
        Dictionary chứa các path quan trọng:
        - train_img_dir
        - train_mask_dir
        - test_img_dir
        - test_mask_dir
        - work_dir
        - ...
    """
    return get_paths(cfg)


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the underlying model when using wrappers such as DataParallel.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_wrap_data_parallel(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """
    Wrap model with DataParallel when more than one GPU is requested.
    """
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


# =============================================================================
# Data preparation
# =============================================================================
# def build_transforms(cfg: Dict[str, Any]) -> Tuple[RandomResizeCrop, EvalTransform]:
def build_transforms(cfg: Dict[str, Any]) -> Tuple[Any, EvalTransform]:
    """
    Build train and evaluation transforms.

    Train transform:
    - random resize
    - random horizontal flip
    - random crop
    - normalize

    Eval transform:
    - optional resize
    - normalize
    - no augmentation

    Parameters
    ----------
    cfg : Dict[str, Any]
        Runtime configuration.

    Returns
    -------
    Tuple[RandomResizeCrop, EvalTransform]
        (train_transform, eval_transform)
    """
    out_size = cfg["train_size"][0] if isinstance(cfg["train_size"], (list, tuple)) else cfg["train_size"]
    
    train_tf = AlbuTrainTransform(
        size=out_size,
        ignore_index=cfg["ignore_index"],
        background_id=cfg["background_id"]
    )
    # train_tf = RandomResizeCrop(
    #     out_size=cfg["train_size"],
    #     scale_range=cfg["scale_range"],
    #     hflip_prob=cfg["hflip_prob"],
    #     mean=cfg["imagenet_mean"],
    #     std=cfg["imagenet_std"],
    #     ignore_index=cfg["ignore_index"],
    #     num_classes=cfg["num_classes"],
    # )

    eval_tf = EvalTransform(
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
    """
    Build train and evaluation datasets from benchmark directories.

    Benchmark-style:
    - train dataset lấy từ official train split
    - eval dataset lấy từ official test split

    Parameters
    ----------
    cfg : Dict[str, Any]
        Runtime configuration.

    paths : Dict[str, Path]
        Resolved project/data paths.

    Returns
    -------
    Tuple[Dataset, Dataset]
        (train_dataset, eval_dataset)
    """
    train_tf, eval_tf = build_transforms(cfg)

    test_samples = build_samples(paths["test_img_dir"], paths["test_mask_dir"])
    eval_ds = FoodSegDataset(test_samples, eval_tf)

    if cfg.get("manifest"):
        overfit_n = cfg.get("overfit_samples", 0)

        if overfit_n > 0:
            print("=" * 80)
            print(f"OVERFIT MODE ENABLED on manifest: {overfit_n} samples")
            print("=" * 80)

            # Debug overfit: bỏ augmentation random để test pipeline có học thuộc không
            train_tf = EvalTransform(
                mean=cfg["imagenet_mean"],
                std=cfg["imagenet_std"],
                ignore_index=cfg["ignore_index"],
                num_classes=cfg["num_classes"],
                out_size=cfg["train_size"],
            )

        train_ds = FoodSegManifestDataset(
            manifest_csv=cfg["manifest"],
            data_root=cfg["data_root"],
            train_stage=cfg.get("train_stage", "easy"),
            transform=train_tf,
        )

        if overfit_n > 0:
            # Train đúng 8 sample từ manifest sau khi filter stage
            train_ds.df = train_ds.df.head(overfit_n).copy()

            # Eval cũng đúng 8 sample đó, nhưng dùng eval transform riêng
            eval_ds = FoodSegManifestDataset(
                manifest_csv=cfg["manifest"],
                data_root=cfg["data_root"],
                train_stage=cfg.get("train_stage", "easy"),
                transform=eval_tf,
            )
            eval_ds.df = train_ds.df.copy()
    else:
        train_samples = build_samples(paths["train_img_dir"], paths["train_mask_dir"])

        overfit_n = cfg.get("overfit_samples", 0)
        if overfit_n > 0:
            print("=" * 80)
            print(f"OVERFIT MODE ENABLED: Allocating {overfit_n} samples for debugging.")
            print("=" * 80)
            train_samples = train_samples[:overfit_n]
            eval_ds = FoodSegDataset(train_samples, eval_tf)

        train_ds = FoodSegDataset(train_samples, train_tf)

    return train_ds, eval_ds


def build_loaders(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> Tuple[DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for training and evaluation.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Runtime configuration.

    paths : Dict[str, Path]
        Resolved project/data paths.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, eval_loader)
    """
    train_ds, eval_ds = build_datasets(cfg, paths)

    if cfg.get("manifest"):
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
            drop_last=cfg["drop_last"],
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


# =============================================================================
# Model / optimizer / scaler
# =============================================================================
def build_model_and_optim(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    """
    Build model, criterion, optimizer, and AMP scaler.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Runtime configuration.
        
    paths : Dict[str, Path]
        Resolved filesystem paths mapping.

    Returns
    -------
    Tuple[nn.Module, nn.Module, torch.optim.Optimizer, GradScaler]
        model, criterion, optimizer, scaler
    """
    model = build_model(cfg, paths).to(cfg["device"])
    model = maybe_wrap_data_parallel(model, cfg)

    # criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    criterion = CombinedLoss(
            num_classes=cfg.get("num_classes", 104),
            ignore_index=cfg["ignore_index"],
            ce_weight=1.0,
            dice_weight=1.0,
            bg_weight=0.1
        )
    criterion.ce.weight = criterion.ce.weight.to(cfg["device"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )

    scaler = _build_grad_scaler(amp_enabled=bool(cfg["amp"]), device=str(cfg["device"]))
    return model, criterion, optimizer, scaler


# =============================================================================
# LR schedule
# =============================================================================
def poly_lr(base_lr: float, cur_iter: int, max_iter: int, power: float = 0.9) -> float:
    """
    Polynomial learning rate schedule used by many segmentation papers.

    Formula
    -------
    lr = base_lr * (1 - cur_iter / max_iter) ** power

    Parameters
    ----------
    base_lr : float
        Initial learning rate.

    cur_iter : int
        Current global iteration.

    max_iter : int
        Total number of training iterations.

    power : float, default=0.9
        Polynomial decay power.

    Returns
    -------
    float
        Learning rate at current iteration.
    """
    return base_lr * (1.0 - float(cur_iter) / float(max_iter)) ** power


# =============================================================================
# Checkpoint helpers
# =============================================================================
def maybe_resume(
    cfg: Dict[str, Any],
    last_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> Tuple[int, int, float]:
    """
    Resume training state from last checkpoint if enabled and available.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Runtime configuration.

    last_path : Path
        Path to the last checkpoint.

    model : nn.Module
        Model instance.

    optimizer : torch.optim.Optimizer
        Optimizer instance.

    scaler : torch.cuda.amp.GradScaler
        AMP scaler instance.

    Returns
    -------
    Tuple[int, int, float]
        start_epoch, global_iter, best_miou
    """
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
        print(f"Resumed from {last_path} at epoch={start_epoch}")

    return start_epoch, global_iter, best_miou


def load_model_weights_only(model: nn.Module, ckpt_path: str, device: str) -> None:
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

    for k, v in state.items():
        kk = k.replace("module.", "")
        if kk in model_state and model_state[kk].shape == v.shape:
            loaded[kk] = v
        else:
            skipped.append(k)

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)

    print(f"Loaded weights only from: {ckpt_path}")
    print(f"Loaded keys: {len(loaded)}")
    print(f"Skipped keys: {len(skipped)}")
    if skipped:
        print("Skipped examples:", skipped[:10])


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Dict[str, Any],
    criterion: nn.Module,
) -> Dict[str, float]:
    """
    Evaluate model on the evaluation loader using benchmark-style metrics.

    Metrics:
    - mIoU
    - mAcc
    - aAcc

    Các metric được tính từ confusion matrix toàn tập eval,
    đúng tinh thần segmentation benchmark chuẩn.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.

    loader : DataLoader
        Evaluation dataloader.

    cfg : Dict[str, Any]
        Runtime configuration.

    criterion : nn.Module
        Loss function used during evaluation.

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics and loss.
    """
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


# =============================================================================
# Training
# =============================================================================
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
    """
    Train model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.

    loader : DataLoader
        Train dataloader.

    cfg : Dict[str, Any]
        Runtime configuration.

    criterion : nn.Module
        Main segmentation loss.

    optimizer : torch.optim.Optimizer
        Optimizer.

    scaler : torch.cuda.amp.GradScaler
        AMP scaler.

    epoch : int
        Current epoch index.

    global_iter : int
        Current global iteration count before this epoch starts.

    max_iter : int
        Total number of training iterations for the whole run.

    Returns
    -------
    Tuple[float, int]
        train_loss, updated_global_iter
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Train {epoch:03d}")

    for step, (images, masks, stems, *_rest) in enumerate(pbar):
        images = images.to(cfg["device"], non_blocking=True)
        masks = masks.to(cfg["device"], non_blocking=True)

        # -------------------------------------------------------------
        # Update learning rate per iteration using poly schedule
        # -------------------------------------------------------------
        cur_lr = poly_lr(cfg["lr"], global_iter, max_iter, cfg["poly_power"])
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)

        # -------------------------------------------------------------
        # Forward + loss
        # BiSeNet training mode returns:
        # - main logits
        # - aux16 logits
        # - aux32 logits
        # -------------------------------------------------------------
        with _autocast(device=str(cfg["device"]), amp_enabled=bool(cfg["amp"])):
            logits, aux16, aux32 = model(images)

            loss_main = criterion(logits, masks)
            loss_aux16 = criterion(aux16, masks)
            loss_aux32 = criterion(aux32, masks)

            loss = loss_main + cfg["aux_weight"] * (loss_aux16 + loss_aux32)

        # -------------------------------------------------------------
        # Backward + optimizer step
        # -------------------------------------------------------------
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        global_iter += 1

        # -------------------------------------------------------------
        # Debug print for first step of each epoch
        # -------------------------------------------------------------
        if step == 0:
            print("-" * 80)
            print(f"Epoch {epoch:03d} step 0")
            print("images:", tuple(images.shape))
            print("masks :", tuple(masks.shape))
            print("logits:", tuple(logits.shape))
            print("lr    :", cur_lr)
            print("loss  :", float(loss.item()))
            print("stems :", list(stems[:4]))
            print("-" * 80)

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.6f}")

    train_loss = running_loss / max(1, len(loader))
    return train_loss, global_iter


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
    """
    Full training loop.

    Workflow
    --------
    1. Resume nếu có checkpoint.
    2. Train từng epoch.
    3. Evaluate theo chu kỳ `eval_every`.
    4. Save last checkpoint mỗi epoch.
    5. Save best checkpoint khi mIoU cải thiện.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.

    criterion : nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer.

    scaler : torch.cuda.amp.GradScaler
        AMP scaler.

    train_loader : DataLoader
        Training dataloader.

    eval_loader : DataLoader
        Evaluation dataloader.

    cfg : Dict[str, Any]
        Runtime configuration.

    paths : Dict[str, Path]
        Resolved runtime paths.
    """
    last_path = paths["work_dir"] / cfg["save_last_name"]
    best_path = paths["work_dir"] / cfg["save_best_name"]

    csv_path = paths["work_dir"] / "metrics.csv" # Đường dẫn file CSV

    start_epoch, global_iter, best_miou = maybe_resume(
        cfg=cfg,
        last_path=last_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )

    if start_epoch == 0:
        ensure_dir(paths["work_dir"])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'mIoU', 'mAcc', 'aAcc'])

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

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        # -------------------------------------------------------------
        # Periodic evaluation on benchmark eval/test split
        # -------------------------------------------------------------
        # Giá trị mặc định cho metrics nếu không chạy evaluation trong epoch hiện tại
        val_loss, miou, macc, aacc = -1.0, -1.0, -1.0, -1.0

        if (epoch + 1) % cfg["eval_every"] == 0:
            scores = evaluate(model, eval_loader, cfg, criterion)
            val_loss = scores['loss']
            miou = scores['mIoU']
            macc = scores['mAcc']
            aacc = scores['aAcc']

            print(
                f"Eval {epoch:03d} | "
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
                print(f"Saved new best checkpoint to {best_path}")

        # Ghi lại kết quả vào tệp CSV (Lưu trên Drive thông qua work_dir)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, miou, macc, aacc])

        # Save last checkpoint every epoch
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


# =============================================================================
# Main entry
# =============================================================================
def main() -> None:
    """
    Program entry point.

    Steps
    -----
    1. Parse args
    2. Load runtime config
    3. Seed everything
    4. Build paths / dataloaders
    5. Build model / optimizer / scaler
    6. Run eval-only or full training
    """
    args = parse_args()

    # -------------------------------------------------------------
    # Runtime config and paths
    # -------------------------------------------------------------
    cfg = get_runtime_cfg(args)
    paths = get_runtime_paths(cfg)

    # -------------------------------------------------------------
    # Reproducibility and speed settings
    # -------------------------------------------------------------
    seed_everything(cfg["seed"])
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True

    # -------------------------------------------------------------
    # Prepare working directory
    # -------------------------------------------------------------
    ensure_dir(paths["work_dir"])

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------
    train_loader, eval_loader = build_loaders(cfg, paths)

    # -------------------------------------------------------------
    # Model / loss / optimizer / scaler
    # -------------------------------------------------------------
    model, criterion, optimizer, scaler = build_model_and_optim(cfg, paths)

    if cfg.get("init_checkpoint"):
        load_model_weights_only(
            unwrap_model(model),
            cfg["init_checkpoint"],
            cfg["device"],
        )

    # -------------------------------------------------------------
    # Eval-only mode
    # -------------------------------------------------------------
    if args.eval_only:
        scores = evaluate(model, eval_loader, cfg, criterion)
        print(scores)
        return

    # -------------------------------------------------------------
    # Full training
    # -------------------------------------------------------------
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
