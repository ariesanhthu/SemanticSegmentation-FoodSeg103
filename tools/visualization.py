import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.bisenet_foodseg103 import CFG, get_paths
from datasets.foodseg103 import FoodSegDataset, EvalTransform, build_samples
from models.bisenetv1 import BiSeNetV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize segmentation overlays (original | ground truth | predict)."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to pretrained checkpoint. Defaults to work_dir/save_best_name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split for visualization.",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=None,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used for inference during visualization.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay blending factor in [0, 1].",
    )
    return parser.parse_args()


def build_palette(num_classes: int, background_id: int | None = None) -> np.ndarray:
    cmap = plt.get_cmap("tab20", num_classes)
    palette = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    if background_id is not None and 0 <= background_id < num_classes:
        palette[background_id] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def denorm_image(img: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    np_img = img.detach().cpu().permute(1, 2, 0).numpy()
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    np_img = np.clip(np_img * std_arr + mean_arr, 0.0, 1.0)
    return np_img


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    out[valid] = palette[mask[valid]]
    return out


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float,
    ignore_index: int | None,
) -> np.ndarray:
    color = colorize_mask(mask, palette).astype(np.float32) / 255.0
    blended = (1.0 - alpha) * image + alpha * color
    if ignore_index is not None:
        valid = mask != ignore_index
        out = image.copy()
        out[valid] = blended[valid]
        return np.clip(out, 0.0, 1.0)
    return np.clip(blended, 0.0, 1.0)


def load_pretrained_weights(model: torch.nn.Module, ckpt_path: Path, device: str) -> tuple[list[str], list[str]]:
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "")
        cleaned_state_dict[cleaned_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    return list(missing), list(unexpected)


def build_loader(cfg: dict, split: str) -> DataLoader:
    paths = get_paths(cfg)
    if split == "train":
        img_dir = paths["train_img_dir"]
        mask_dir = paths["train_mask_dir"]
    else:
        img_dir = paths["test_img_dir"]
        mask_dir = paths["test_mask_dir"]

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            "Dataset folder not found. "
            f"Expected image dir: {img_dir}, mask dir: {mask_dir}. "
            "Please set DATA_ROOT to the FoodSeg103 root folder."
        )

    samples = build_samples(img_dir, mask_dir)
    if len(samples) == 0:
        raise ValueError(f"No samples found in {img_dir} and {mask_dir}.")

    transform = EvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=cfg.get("test_size"),
    )

    return DataLoader(
        FoodSegDataset(samples, transform),
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict,
    num_vis: int,
) -> list[tuple[torch.Tensor, np.ndarray, np.ndarray, str]]:
    model.eval()
    outputs: list[tuple[torch.Tensor, np.ndarray, np.ndarray, str]] = []

    for images, masks, stems, *_ in loader:
        images = images.to(cfg["device"], non_blocking=True)
        logits = model(images)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        preds = logits.argmax(dim=1).cpu().numpy()
        images_cpu = images.cpu()
        masks_np = masks.numpy()

        for i in range(images_cpu.shape[0]):
            outputs.append((images_cpu[i], masks_np[i], preds[i], stems[i]))
            if len(outputs) >= num_vis:
                return outputs

    return outputs


def visualize_triptych(samples: list[tuple[torch.Tensor, np.ndarray, np.ndarray, str]], cfg: dict, alpha: float) -> None:
    if len(samples) == 0:
        raise ValueError("No predictions collected for visualization.")

    palette = build_palette(cfg["num_classes"], cfg.get("background_id"))
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (img_t, gt, pred, stem) in enumerate(samples):
        img = denorm_image(img_t, cfg["imagenet_mean"], cfg["imagenet_std"])

        gt_overlay = overlay_mask(
            image=img,
            mask=gt,
            palette=palette,
            alpha=alpha,
            ignore_index=cfg["ignore_index"],
        )
        pred_overlay = overlay_mask(
            image=img,
            mask=pred,
            palette=palette,
            alpha=alpha,
            ignore_index=None,
        )

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original\n{stem}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title(f"Ground Truth Overlay\nclasses={len(np.unique(gt[gt != cfg['ignore_index']]))}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(f"Predict Overlay\nclasses={len(np.unique(pred))}")
        axes[row, 2].axis("off")

    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    cfg = CFG.copy()

    if args.num_vis is not None:
        cfg["num_vis"] = args.num_vis
    if args.batch_size is not None:
        cfg["eval_batch_size"] = args.batch_size

    paths = get_paths(cfg)
    ckpt_path = Path(args.ckpt) if args.ckpt else (paths["work_dir"] / cfg["save_best_name"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    loader = build_loader(cfg, split=args.split)
    model = BiSeNetV1(num_classes=cfg["num_classes"]).to(cfg["device"])

    missing, unexpected = load_pretrained_weights(model, ckpt_path, cfg["device"])
    if missing:
        print(f"[WARN] Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading checkpoint: {len(unexpected)}")
    print(f"Loaded pretrained checkpoint: {ckpt_path}")

    samples = collect_predictions(model, loader, cfg, num_vis=cfg["num_vis"])
    visualize_triptych(samples, cfg, alpha=float(np.clip(args.alpha, 0.0, 1.0)))


if __name__ == "__main__":
    main()