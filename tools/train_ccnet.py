import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.ccnet_foodseg103 import CFG, get_paths
from datasets.foodseg103_ccnet import (
    EvalTransform,
    FoodSegDataset,
    RandomResizeCrop,
    build_samples,
    resolve_dataset_meta,
    set_seed_for_worker,
)
from models.ccnet import CCNetSeg
from utils.metrics import compute_segmentation_scores, fast_hist
from utils.misc import ensure_dir, load_checkpoint, save_checkpoint, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FoodSeg103 CCNet-ResNet50 baseline.")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", type=str, default=None, help="Override work directory.")
    parser.add_argument("--max-iters", type=int, default=None, help="Override max training iterations.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Convenience override. If set, max_iters = epochs * len(train_loader).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size.")
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="Use the first N train samples for debugging and evaluate on the same subset.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation.")
    return parser.parse_args()


def get_runtime_cfg(args: argparse.Namespace) -> Dict[str, object]:
    cfg = CFG.copy()
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters
    if args.epochs is not None and args.max_iters is None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    cfg["overfit_samples"] = max(0, int(args.overfit))
    return resolve_dataset_meta(cfg)


def build_loaders(cfg: Dict[str, object]) -> Tuple[DataLoader, DataLoader]:
    paths = get_paths(cfg)
    train_samples = build_samples(paths["train_img_dir"], paths["train_mask_dir"])
    test_samples = build_samples(paths["test_img_dir"], paths["test_mask_dir"])

    overfit_samples = int(cfg.get("overfit_samples", 0))
    if overfit_samples > 0:
        print("=" * 80)
        print(f"OVERFIT MODE ENABLED: using first {overfit_samples} train samples.")
        print("=" * 80)
        train_samples = train_samples[:overfit_samples]
        test_samples = train_samples.copy()

    train_tf = RandomResizeCrop(
        out_size=cfg["train_size"],
        scale_range=cfg["scale_range"],
        hflip_prob=cfg["hflip_prob"],
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        use_color_jitter=cfg["use_color_jitter"],
        brightness=cfg["color_jitter_brightness"],
        contrast=cfg["color_jitter_contrast"],
        saturation=cfg["color_jitter_saturation"],
        hue=cfg["color_jitter_hue"],
    )
    eval_tf = EvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=cfg["eval_size"],
    )

    train_loader = DataLoader(
        FoodSegDataset(train_samples, train_tf),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        drop_last=bool(cfg["drop_last"]) and len(train_samples) >= int(cfg["batch_size"]),
        worker_init_fn=set_seed_for_worker,
    )

    eval_batch_size = cfg["eval_batch_size"]
    if cfg["eval_size"] is None:
        eval_batch_size = 1

    eval_loader = DataLoader(
        FoodSegDataset(test_samples, eval_tf),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    return train_loader, eval_loader


def build_model(
    cfg: Dict[str, object],
    backbone_pretrained: bool | None = None,
) -> nn.Module:
    return CCNetSeg(
        num_classes=cfg["num_classes"],
        backbone_pretrained=(
            cfg["backbone_pretrained"]
            if backbone_pretrained is None
            else backbone_pretrained
        ),
        output_stride=cfg["output_stride"],
        channels=cfg["cc_channels"],
        recurrence=cfg["cc_recurrence"],
        use_aux=cfg["use_aux_head"],
        dropout=cfg["dropout"],
        align_corners=cfg["align_corners"],
    ).to(cfg["device"])


def to_serializable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def format_scores(scores: Dict[str, object]) -> Dict[str, object]:
    return {key: to_serializable(value) for key, value in scores.items()}


def poly_lr(base_lr: float, cur_iter: int, max_iter: int, power: float = 0.9) -> float:
    progress = min(float(cur_iter) / float(max_iter), 1.0)
    return base_lr * (1.0 - progress) ** power


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Dict[str, object],
    criterion: nn.Module,
) -> Dict[str, float]:
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


def maybe_resume(
    cfg: Dict[str, object],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    last_path: Path,
) -> Tuple[int, float]:
    global_iter = 0
    best_miou = -1.0

    if cfg["resume"] and last_path.exists():
        checkpoint = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=cfg["device"],
        )
        global_iter = int(checkpoint.get("global_iter", 0))
        best_miou = float(checkpoint.get("best_miou", -1.0))
        print(f"Resumed from {last_path} at iter={global_iter}")

    return global_iter, best_miou


def run_training(cfg: Dict[str, object]) -> None:
    seed_everything(cfg["seed"])
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True

    paths = get_paths(cfg)
    ensure_dir(paths["work_dir"])
    save_json(cfg, paths["work_dir"] / cfg["config_json_name"])

    train_loader, eval_loader = build_loaders(cfg)
    if len(train_loader) == 0:
        raise RuntimeError("Train loader is empty.")

    if cfg.get("epochs") is not None:
        cfg["max_iters"] = max(1, int(cfg["epochs"]) * len(train_loader))
        print(
            f"Resolved max_iters from epochs: epochs={cfg['epochs']} "
            f"x iters_per_epoch={len(train_loader)} -> max_iters={cfg['max_iters']}"
        )

    last_path = paths["work_dir"] / cfg["save_last_name"]
    best_path = paths["work_dir"] / cfg["save_best_name"]
    use_pretrained_backbone = bool(cfg["backbone_pretrained"]) and not (
        cfg["resume"] and last_path.exists()
    )

    model = build_model(cfg, backbone_pretrained=use_pretrained_backbone)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg["amp"]) and str(cfg["device"]).startswith("cuda")
    )

    global_iter, best_miou = maybe_resume(cfg, model, optimizer, scaler, last_path)

    if global_iter >= cfg["max_iters"]:
        print(f"Checkpoint already reached max_iters={cfg['max_iters']}. Running eval only.")
        scores = evaluate(model, eval_loader, cfg, criterion)
        print(json.dumps(format_scores(scores), indent=2))
        return

    data_iter = iter(train_loader)
    running_loss = 0.0
    log_steps = 0

    while global_iter < cfg["max_iters"]:
        try:
            images, masks, stems, *_ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, masks, stems, *_ = next(data_iter)

        images = images.to(cfg["device"], non_blocking=True)
        masks = masks.to(cfg["device"], non_blocking=True)

        cur_lr = poly_lr(
            base_lr=cfg["lr"],
            cur_iter=global_iter,
            max_iter=cfg["max_iters"],
            power=cfg["poly_power"],
        )
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=bool(cfg["amp"]) and str(cfg["device"]).startswith("cuda")
        ):
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits, aux_logits = outputs
                loss_main = criterion(logits, masks)
                loss_aux = criterion(aux_logits, masks)
                loss = loss_main + cfg["aux_weight"] * loss_aux
            else:
                logits = outputs
                loss_main = criterion(logits, masks)
                loss_aux = torch.zeros((), device=images.device)
                loss = loss_main

        scaler.scale(loss).backward()

        if cfg["grad_clip_norm"] is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()

        global_iter += 1
        running_loss += float(loss.item())
        log_steps += 1

        if global_iter == 1 or global_iter % cfg["print_freq"] == 0:
            avg_loss = running_loss / max(1, log_steps)
            running_loss = 0.0
            log_steps = 0
            print(
                f"Iter {global_iter:06d}/{cfg['max_iters']:06d} | "
                f"loss={avg_loss:.4f} | main={loss_main.item():.4f} | "
                f"aux={loss_aux.item():.4f} | lr={cur_lr:.6e} | "
                f"batch={tuple(images.shape)} | stems={list(stems[:2])}"
            )

        should_eval = global_iter % cfg["eval_interval"] == 0 or global_iter == cfg["max_iters"]
        should_ckpt = (
            global_iter % cfg["checkpoint_interval"] == 0 or global_iter == cfg["max_iters"]
        )

        if should_eval:
            scores = evaluate(model, eval_loader, cfg, criterion)
            print(
                f"Eval iter {global_iter:06d} | loss={scores['loss']:.4f} | "
                f"mIoU={scores['mIoU']:.4f} | mAcc={scores['mAcc']:.4f} | "
                f"aAcc={scores['aAcc']:.4f}"
            )

            if scores["mIoU"] > best_miou:
                best_miou = scores["mIoU"]
                save_checkpoint(
                    best_path,
                    epoch=global_iter // len(train_loader),
                    global_iter=global_iter,
                    best_miou=best_miou,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    cfg=cfg,
                )
                print(f"Saved best checkpoint to {best_path}")

        if should_ckpt:
            save_checkpoint(
                last_path,
                epoch=global_iter // len(train_loader),
                global_iter=global_iter,
                best_miou=best_miou,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
            )
            print(f"Saved last checkpoint to {last_path}")


def main() -> None:
    args = parse_args()
    cfg = get_runtime_cfg(args)

    if args.eval_only:
        seed_everything(cfg["seed"])
        paths = get_paths(cfg)
        ensure_dir(paths["work_dir"])

        _, eval_loader = build_loaders(cfg)
        model = build_model(cfg, backbone_pretrained=False)
        criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
        ckpt_path = paths["work_dir"] / cfg["save_best_name"]
        load_checkpoint(ckpt_path, model=model, map_location=cfg["device"])
        scores = evaluate(model, eval_loader, cfg, criterion)
        print(json.dumps(format_scores(scores), indent=2))
        return

    run_training(cfg)


if __name__ == "__main__":
    main()
