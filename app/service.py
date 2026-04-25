"""Backend service layer for the FoodSeg103 Gradio demo.

Provides :class:`FoodSegDemoService` which manages model presets,
checkpoint loading/caching, single-image and video inference pipelines,
and result formatting (overlays, metrics, timing).
"""

from __future__ import annotations

import colorsys
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs.bisenet_foodseg103 import CFG as BISENET_CFG
from configs.bisenet_foodseg103 import get_paths as get_bisenet_paths
from configs.ccnet_foodseg103 import CFG as CCNET_CFG
from configs.ccnet_foodseg103 import get_paths as get_ccnet_paths
from datasets.foodseg103_ccnet import IMG_EXTS, load_class_mapping, resolve_dataset_meta
from models.builder import build_model as build_bisenet_model
from models.ccnet import CCNetSeg
from utils.metrics import compute_segmentation_scores, fast_hist


def _to_path(value: Any) -> Path | None:
    """Coerce *value* to a :class:`Path`, or return ``None`` if not possible."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return Path(stripped) if stripped else None
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate:
                return Path(candidate)
    return None


def _safe_float(value: float, digits: int = 4) -> float:
    """Round *value* to *digits* decimal places, ensuring a plain float."""
    return round(float(value), digits)


def _build_palette(num_classes: int, background_id: int | None = None) -> np.ndarray:
    """Generate a deterministic HSV-based colour palette for *num_classes*."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for idx in range(num_classes):
        hue = ((idx * 37) % 360) / 360.0
        sat = 0.55 + ((idx % 4) * 0.1)
        val = 0.85 + ((idx % 3) * 0.05)
        rgb = colorsys.hsv_to_rgb(hue, min(sat, 0.95), min(val, 1.0))
        palette[idx] = np.asarray(rgb, dtype=np.float32) * 255
    if background_id is not None and 0 <= background_id < num_classes:
        palette[background_id] = np.array([30, 30, 30], dtype=np.uint8)
    return palette


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a label mask to *(W, H)* using nearest-neighbour interpolation."""
    pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L")
    resized = pil_mask.resize(size, Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map integer label *mask* to an RGB image using *palette*."""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    colored[valid] = palette[mask[valid]]
    return colored


def _overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Alpha-blend a colourised *mask* onto *image_rgb*."""
    color = _colorize_mask(mask, palette).astype(np.float32)
    image = image_rgb.astype(np.float32)
    overlay = ((1.0 - alpha) * image) + (alpha * color)
    if ignore_index is not None:
        valid = mask != ignore_index
        output = image.copy()
        output[valid] = overlay[valid]
        return np.clip(output, 0.0, 255.0).astype(np.uint8)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _load_rgb_image(path: Path) -> Image.Image:
    """Open an image file and return it as an RGB :class:`PIL.Image.Image`."""
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def _load_label_mask(
    path: Path,
    target_size: tuple[int, int] | None,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """Load a ground-truth mask as an ``int64`` tensor, clamping invalid labels."""
    with Image.open(path) as image:
        image.load()
        if target_size is not None:
            image = image.resize(target_size, Image.NEAREST)
        mask = np.asarray(image, dtype=np.int64)
        if mask.ndim == 3:
            mask = mask[..., 0]
        invalid = (mask < 0) | (mask >= num_classes)
        mask[invalid] = ignore_index
        return torch.from_numpy(mask)


def _load_display_mask(
    path: Path,
    target_size: tuple[int, int] | None,
    ignore_index: int,
) -> np.ndarray:
    """Load a mask for display purposes (uint8 NumPy array)."""
    with Image.open(path) as image:
        image.load()
        if target_size is not None:
            image = image.resize(target_size, Image.NEAREST)
        mask = np.asarray(image, dtype=np.int64)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask.astype(np.uint8)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a raw state-dict from a checkpoint object."""
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def _load_checkpoint_flexible(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    """Load weights into *model*, tolerating common checkpoint variations."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = _extract_state_dict(checkpoint)
    cleaned_state = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model_keys = set(model.state_dict().keys())
    matched_keys = len(model_keys.intersection(cleaned_state.keys()))
    coverage = matched_keys / max(1, len(model_keys))
    if coverage < 0.5:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not look compatible with the selected model "
            f"(matched {matched_keys}/{len(model_keys)} keys)."
        )

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    return {
        "matched_keys": matched_keys,
        "model_key_count": len(model_keys),
        "coverage": _safe_float(coverage),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _autocast_context(device: str, amp_enabled: bool):
    """Return a mixed-precision autocast context if applicable, else a no-op."""
    if device.startswith("cuda") and amp_enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _summarize_classes(
    mask: np.ndarray,
    class_names: list[str],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Return the *top_k* most frequent classes in *mask* with pixel counts."""
    class_ids, counts = np.unique(mask, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = max(1, int(mask.size))
    rows: list[dict[str, Any]] = []
    for idx in order[:top_k]:
        class_id = int(class_ids[idx])
        pixels = int(counts[idx])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                "pixels": pixels,
                "ratio": _safe_float(pixels / total),
            }
        )
    return rows


def _summarize_quality_scores(
    scores: dict[str, Any],
    class_names: list[str],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Rank valid classes by IoU and return the *top_k* best performers."""
    iou = np.asarray(scores["IoU_per_class"])
    acc = np.asarray(scores["Acc_per_class"])
    valid = np.asarray(scores["valid_class_mask"]).astype(bool)
    class_ids = np.where(valid)[0]
    ranked = sorted(class_ids.tolist(), key=lambda idx: float(iou[idx]), reverse=True)
    summary: list[dict[str, Any]] = []
    for class_id in ranked[:top_k]:
        summary.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                "iou": _safe_float(iou[class_id]),
                "accuracy": _safe_float(acc[class_id]),
            }
        )
    return summary


@dataclass(frozen=True)
class ModelPreset:
    """Immutable descriptor for a registered segmentation model variant."""

    key: str
    label: str
    cfg: dict[str, Any]
    checkpoint_path: Path | None
    test_img_dir: Path
    test_mask_dir: Path
    class_names: list[str]
    input_size: tuple[int, int] | None
    build_model: Callable[[dict[str, Any]], torch.nn.Module]


@dataclass
class LoadedModel:
    """A model that has been instantiated, loaded from a checkpoint, and is ready for inference."""

    preset: ModelPreset
    checkpoint_path: Path
    model: torch.nn.Module
    palette: np.ndarray
    load_info: dict[str, Any]


class FoodSegDemoService:
    """Stateful service that owns model presets, caches loaded models, and runs inference."""

    def __init__(self) -> None:
        """Initialise presets, label mappings, and test-sample index."""
        self.root = ROOT
        self.presets = self._build_presets()
        self.label_to_key = {preset.label: key for key, preset in self.presets.items()}
        self._test_samples = self._index_test_samples()
        self._loaded_models: dict[tuple[str, str], LoadedModel] = {}

    def _build_presets(self) -> dict[str, ModelPreset]:
        """Create :class:`ModelPreset` entries for BiSeNet and CCNet."""
        bisenet_cfg = BISENET_CFG.copy()
        bisenet_paths = get_bisenet_paths(bisenet_cfg)
        mapping = load_class_mapping(
            data_root=Path(bisenet_cfg["data_root"]),
            mapping_name="class_mapping.json",
            fallback_num_classes=bisenet_cfg["num_classes"],
            fallback_background_id=bisenet_cfg["background_id"],
            fallback_num_ingredient_classes=bisenet_cfg["num_classes"] - 1,
        )
        bisenet_cfg.update(mapping)

        ccnet_cfg = resolve_dataset_meta(CCNET_CFG.copy())
        ccnet_paths = get_ccnet_paths(ccnet_cfg)
        ccnet_candidates = [
            ccnet_paths["work_dir"] / ccnet_cfg["save_best_name"],
            self.root / "work_dirs" / "ccnet_report_smoke" / ccnet_cfg["save_best_name"],
        ]
        ccnet_checkpoint = next((path for path in ccnet_candidates if path.exists()), None)

        def make_bisenet(cfg: dict[str, Any]) -> torch.nn.Module:
            model_cfg = cfg.copy()
            return build_bisenet_model(model_cfg, get_bisenet_paths(model_cfg))

        def make_ccnet(cfg: dict[str, Any]) -> torch.nn.Module:
            return CCNetSeg(
                num_classes=cfg["num_classes"],
                backbone_pretrained=False,
                output_stride=cfg["output_stride"],
                channels=cfg["cc_channels"],
                recurrence=cfg["cc_recurrence"],
                use_aux=cfg["use_aux_head"],
                dropout=cfg["dropout"],
                align_corners=cfg["align_corners"],
            )

        presets = {
            "bisenet": ModelPreset(
                key="bisenet",
                label="BiSeNetV1 (FoodSeg103)",
                cfg=bisenet_cfg,
                checkpoint_path=(bisenet_paths["work_dir"] / bisenet_cfg["save_best_name"]),
                test_img_dir=bisenet_paths["test_img_dir"],
                test_mask_dir=bisenet_paths["test_mask_dir"],
                class_names=bisenet_cfg["class_names"],
                input_size=bisenet_cfg.get("test_size"),
                build_model=make_bisenet,
            ),
            "ccnet": ModelPreset(
                key="ccnet",
                label="CCNet-ResNet50 (FoodSeg103)",
                cfg=ccnet_cfg,
                checkpoint_path=ccnet_checkpoint,
                test_img_dir=ccnet_paths["test_img_dir"],
                test_mask_dir=ccnet_paths["test_mask_dir"],
                class_names=ccnet_cfg["class_names"],
                input_size=ccnet_cfg.get("eval_size"),
                build_model=make_ccnet,
            ),
        }
        return presets

    def _index_test_samples(self) -> dict[str, dict[str, Path]]:
        """Scan the test-set directories and build an image-name → path mapping."""
        samples: dict[str, dict[str, Path]] = {}
        reference_preset = self.presets["ccnet"] if "ccnet" in self.presets else next(iter(self.presets.values()))
        if not reference_preset.test_img_dir.exists() or not reference_preset.test_mask_dir.exists():
            return samples

        for mask_path in sorted(reference_preset.test_mask_dir.glob("*.png")):
            stem = mask_path.stem
            image_path = None
            for ext in IMG_EXTS:
                candidate = reference_preset.test_img_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            samples[image_path.name] = {
                "image_path": image_path,
                "mask_path": mask_path,
                "stem": stem,
            }
        return samples

    def get_model_labels(self) -> list[str]:
        """Return human-readable labels for all registered models."""
        return [preset.label for preset in self.presets.values()]

    def get_default_model_label(self) -> str:
        """Return the label of the first registered model (the default)."""
        return next(iter(self.presets.values())).label

    def get_test_image_choices(self, limit: int = 24) -> list[str]:
        """Return up to *limit* test-image filenames for the dropdown."""
        return list(self._test_samples.keys())[:limit]

    def get_default_test_image(self) -> str | None:
        """Return the first test image name, or ``None`` if none exist."""
        choices = self.get_test_image_choices(limit=1)
        return choices[0] if choices else None

    def describe_model(self, model_label: str) -> str:
        """Return a Markdown snippet describing the selected model's config."""
        preset = self._get_preset(model_label)
        checkpoint = preset.checkpoint_path
        checkpoint_text = str(checkpoint) if checkpoint is not None else "Default checkpoint not found"
        size_text = (
            f"{preset.input_size[0]} x {preset.input_size[1]}"
            if preset.input_size is not None
            else "keep original resolution"
        )
        return (
            f"**Model:** `{preset.label}`\n\n"
            f"- Default checkpoint: `{checkpoint_text}`\n"
            f"- Device: `{preset.cfg['device']}`\n"
            f"- Inference size: `{size_text}`\n"
            f"- Classes: `{preset.cfg['num_classes']}`"
        )

    def _get_preset(self, model_label: str) -> ModelPreset:
        """Look up a :class:`ModelPreset` by its human-readable label."""
        key = self.label_to_key.get(model_label)
        if key is None:
            raise ValueError(f"Unknown model label: {model_label}")
        return self.presets[key]

    def _resolve_checkpoint(self, preset: ModelPreset, checkpoint_override: str | None) -> Path:
        """Determine the checkpoint path, preferring *checkpoint_override* if given."""
        override_path = _to_path(checkpoint_override)
        checkpoint = override_path if override_path is not None else preset.checkpoint_path
        if checkpoint is None:
            raise FileNotFoundError(
                f"No default checkpoint found for model '{preset.label}'."
            )
        if not checkpoint.is_absolute():
            checkpoint = (self.root / checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    def _get_loaded_model(
        self,
        model_label: str,
        checkpoint_override: str | None = None,
    ) -> LoadedModel:
        """Return a cached :class:`LoadedModel`, building and loading it on first access."""
        preset = self._get_preset(model_label)
        checkpoint = self._resolve_checkpoint(preset, checkpoint_override)
        cache_key = (preset.key, str(checkpoint))
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        model = preset.build_model(preset.cfg).to(preset.cfg["device"])
        load_info = _load_checkpoint_flexible(model, checkpoint, preset.cfg["device"])
        model.eval()

        loaded = LoadedModel(
            preset=preset,
            checkpoint_path=checkpoint,
            model=model,
            palette=_build_palette(preset.cfg["num_classes"], preset.cfg.get("background_id")),
            load_info=load_info,
        )
        self._loaded_models[cache_key] = loaded
        return loaded

    def _prepare_input_tensor(
        self,
        image: Image.Image,
        preset: ModelPreset,
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """Resize, normalise, and batch-wrap an image for model input."""
        original_w, original_h = image.size
        resized = image
        if preset.input_size is not None:
            resized_h, resized_w = preset.input_size
            resized = image.resize((resized_w, resized_h), Image.BILINEAR)

        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(preset.cfg["imagenet_mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(preset.cfg["imagenet_std"], dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(preset.cfg["device"])
        return tensor, (original_w, original_h), resized.size

    def _run_single_image(
        self,
        image_path: Path,
        loaded: LoadedModel,
        alpha: float,
    ) -> dict[str, Any]:
        """Run inference on a single image and return all outputs with timing."""
        overall_start = perf_counter()
        load_start = perf_counter()
        image = _load_rgb_image(image_path)
        image_rgb = np.asarray(image, dtype=np.uint8)
        load_ms = (perf_counter() - load_start) * 1000.0

        preprocess_start = perf_counter()
        tensor, original_size, model_size = self._prepare_input_tensor(image, loaded.preset)
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        inference_start = perf_counter()
        with torch.inference_mode():
            with _autocast_context(loaded.preset.cfg["device"], bool(loaded.preset.cfg.get("amp", False))):
                logits = loaded.model(tensor)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values[0].detach().cpu().numpy()
        prediction_small = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        inference_ms = (perf_counter() - inference_start) * 1000.0

        postprocess_start = perf_counter()
        prediction_display = _resize_mask(prediction_small, original_size)
        overlay = _overlay_mask(
            image_rgb=image_rgb,
            mask=prediction_display,
            palette=loaded.palette,
            alpha=float(np.clip(alpha, 0.0, 1.0)),
            ignore_index=None,
        )
        color_mask = _colorize_mask(prediction_display, loaded.palette)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0

        total_ms = (perf_counter() - overall_start) * 1000.0
        return {
            "input_image": image_rgb,
            "overlay": overlay,
            "color_mask": color_mask,
            "prediction_small": prediction_small,
            "confidence": confidence,
            "timing": {
                "load_ms": _safe_float(load_ms, 2),
                "preprocess_ms": _safe_float(preprocess_ms, 2),
                "inference_ms": _safe_float(inference_ms, 2),
                "postprocess_ms": _safe_float(postprocess_ms, 2),
                "total_ms": _safe_float(total_ms, 2),
            },
            "model_input_size": {"width": int(model_size[0]), "height": int(model_size[1])},
            "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
        }

    def _resolve_image_source(
        self,
        image_path: Any,
        selected_test_image: str | None,
        ground_truth_mask_path: Any,
    ) -> tuple[Path, Path | None, str]:
        """Resolve the image source from upload or test-set selection."""
        uploaded_image = _to_path(image_path)
        uploaded_mask = _to_path(ground_truth_mask_path)
        if uploaded_image is not None:
            return uploaded_image, uploaded_mask, "upload"

        if selected_test_image:
            sample = self._test_samples.get(selected_test_image)
            if sample is None:
                raise FileNotFoundError(f"Test image not found: {selected_test_image}")
            return sample["image_path"], sample["mask_path"], "test"

        raise ValueError("Select a test image or upload an image to continue.")

    def predict_image(
        self,
        model_label: str,
        selected_test_image: str | None,
        uploaded_image_path: Any,
        uploaded_mask_path: Any,
        checkpoint_override: str | None,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any], dict[str, Any]]:
        """Run single-image inference and return input, GT overlay, prediction overlay, metrics, and timing."""
        loaded = self._get_loaded_model(model_label, checkpoint_override)
        image_path, gt_path, source_type = self._resolve_image_source(
            image_path=uploaded_image_path,
            selected_test_image=selected_test_image,
            ground_truth_mask_path=uploaded_mask_path,
        )
        prediction = self._run_single_image(image_path, loaded, alpha)
        gt_preview: np.ndarray | None = None

        metrics: dict[str, Any] = {
            "source": {
                "type": source_type,
                "image_path": str(image_path),
                "ground_truth_mask": str(gt_path) if gt_path is not None else None,
            },
            "model": {
                "name": loaded.preset.label,
                "checkpoint": str(loaded.checkpoint_path),
                "device": loaded.preset.cfg["device"],
                "checkpoint_load": loaded.load_info,
            },
            "prediction": {
                "num_predicted_classes": int(np.unique(prediction["prediction_small"]).size),
                "top_classes": _summarize_classes(
                    prediction["prediction_small"],
                    class_names=loaded.preset.class_names,
                    top_k=8,
                ),
                "mean_confidence": _safe_float(float(prediction["confidence"].mean())),
                "max_confidence": _safe_float(float(prediction["confidence"].max())),
                "min_confidence": _safe_float(float(prediction["confidence"].min())),
            },
        }

        if gt_path is not None:
            gt_preview_mask = _load_display_mask(
                gt_path,
                target_size=(
                    prediction["original_size"]["width"],
                    prediction["original_size"]["height"],
                ),
                ignore_index=loaded.preset.cfg["ignore_index"],
            )
            gt_preview = _overlay_mask(
                image_rgb=prediction["input_image"],
                mask=gt_preview_mask,
                palette=loaded.palette,
                alpha=float(np.clip(alpha, 0.0, 1.0)),
                ignore_index=loaded.preset.cfg["ignore_index"],
            )
            gt_mask = _load_label_mask(
                gt_path,
                target_size=(
                    prediction["model_input_size"]["width"],
                    prediction["model_input_size"]["height"],
                ),
                num_classes=loaded.preset.cfg["num_classes"],
                ignore_index=loaded.preset.cfg["ignore_index"],
            )
            pred_tensor = torch.from_numpy(prediction["prediction_small"].astype(np.int64))
            hist = fast_hist(
                pred_tensor,
                gt_mask,
                loaded.preset.cfg["num_classes"],
                loaded.preset.cfg["ignore_index"],
            )
            scores = compute_segmentation_scores(hist)
            metrics["segmentation_metrics"] = {
                "aAcc": _safe_float(scores["aAcc"]),
                "mAcc": _safe_float(scores["mAcc"]),
                "mIoU": _safe_float(scores["mIoU"]),
                "top_present_classes": _summarize_quality_scores(
                    scores,
                    class_names=loaded.preset.class_names,
                    top_k=8,
                ),
            }
        else:
            metrics["segmentation_metrics"] = {
                "available": False,
                "message": "No ground-truth mask was provided, so only prediction statistics are available.",
            }

        timing = prediction["timing"] | {
            "image_width": prediction["original_size"]["width"],
            "image_height": prediction["original_size"]["height"],
        }
        return prediction["input_image"], gt_preview, prediction["overlay"], metrics, timing

    def process_video(
        self,
        model_label: str,
        uploaded_video_path: Any,
        checkpoint_override: str | None,
        alpha: float,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Run frame-by-frame inference on a video and return the overlay video path, metrics, and timing."""
        video_path = _to_path(uploaded_video_path)
        if video_path is None:
            raise ValueError("Upload a video to continue.")

        loaded = self._get_loaded_model(model_label, checkpoint_override)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fd, output_path = tempfile.mkstemp(prefix="foodseg_overlay_", suffix=".mp4")
        os.close(fd)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Cannot create the output video file.")

        total_start = perf_counter()
        processed_frames = 0
        inference_ms_total = 0.0
        total_ms_total = 0.0
        mean_conf_sum = 0.0
        class_hist = np.zeros((loaded.preset.cfg["num_classes"],), dtype=np.int64)

        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb, mode="RGB")

                frame_start = perf_counter()
                pil_image = frame_image
                image_rgb = np.asarray(pil_image, dtype=np.uint8)

                tensor, _, _ = self._prepare_input_tensor(pil_image, loaded.preset)

                inference_start = perf_counter()
                with torch.inference_mode():
                    with _autocast_context(loaded.preset.cfg["device"], bool(loaded.preset.cfg.get("amp", False))):
                        logits = loaded.model(tensor)
                        if isinstance(logits, (list, tuple)):
                            logits = logits[0]
                probs = torch.softmax(logits, dim=1)
                confidence = probs.max(dim=1).values[0].detach().cpu().numpy()
                pred_small = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
                inference_ms = (perf_counter() - inference_start) * 1000.0

                pred_display = _resize_mask(pred_small, (width, height))
                overlay = _overlay_mask(
                    image_rgb=image_rgb,
                    mask=pred_display,
                    palette=loaded.palette,
                    alpha=float(np.clip(alpha, 0.0, 1.0)),
                    ignore_index=None,
                )
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                class_hist += np.bincount(
                    pred_small.reshape(-1),
                    minlength=loaded.preset.cfg["num_classes"],
                )
                processed_frames += 1
                mean_conf_sum += float(confidence.mean())
                inference_ms_total += inference_ms
                total_ms_total += (perf_counter() - frame_start) * 1000.0
        finally:
            capture.release()
            writer.release()

        total_elapsed_ms = (perf_counter() - total_start) * 1000.0
        if processed_frames == 0:
            raise RuntimeError("The video did not contain any readable frames.")

        ranked_ids = np.argsort(class_hist)[::-1]
        top_classes: list[dict[str, Any]] = []
        total_pixels = int(class_hist.sum()) or 1
        for class_id in ranked_ids[:8]:
            pixels = int(class_hist[class_id])
            if pixels <= 0:
                continue
            top_classes.append(
                {
                    "class_id": int(class_id),
                    "class_name": loaded.preset.class_names[class_id],
                    "pixels": pixels,
                    "ratio": _safe_float(pixels / total_pixels),
                }
            )

        metrics = {
            "source": {
                "video_path": str(video_path),
                "frame_count": frame_total,
                "resolution": {"width": width, "height": height},
                "fps": _safe_float(fps, 2),
            },
            "model": {
                "name": loaded.preset.label,
                "checkpoint": str(loaded.checkpoint_path),
                "device": loaded.preset.cfg["device"],
            },
            "prediction": {
                "top_classes": top_classes,
                "mean_frame_confidence": _safe_float(mean_conf_sum / processed_frames),
            },
        }
        timing = {
            "processed_frames": processed_frames,
            "avg_inference_ms_per_frame": _safe_float(inference_ms_total / processed_frames, 2),
            "avg_total_ms_per_frame": _safe_float(total_ms_total / processed_frames, 2),
            "effective_fps": _safe_float((processed_frames * 1000.0) / max(total_elapsed_ms, 1.0), 2),
            "total_video_runtime_ms": _safe_float(total_elapsed_ms, 2),
        }
        return output_path, metrics, timing
