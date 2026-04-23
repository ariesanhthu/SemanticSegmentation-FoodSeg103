import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


Sample = Tuple[str, str, str]
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def load_class_mapping(
    data_root: Path,
    mapping_name: str,
    fallback_num_classes: int,
    fallback_background_id: int,
    fallback_num_ingredient_classes: int,
) -> Dict[str, object]:
    mapping_path = data_root / mapping_name

    num_classes = int(fallback_num_classes)
    background_id = int(fallback_background_id)
    num_ingredient_classes = int(fallback_num_ingredient_classes)
    class_names: List[str] = [f"class_{idx}" for idx in range(num_classes)]

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        background_id = int(mapping.get("background_id", background_id))
        num_ingredient_classes = int(
            mapping.get("num_ingredient_classes", num_ingredient_classes)
        )

        ids: List[int] = []
        id_to_class = mapping.get("id_to_class", {})
        for raw_idx, class_name in id_to_class.items():
            idx = int(raw_idx)
            ids.append(idx)
            if idx >= len(class_names):
                class_names.extend(
                    f"class_{extra_idx}"
                    for extra_idx in range(len(class_names), idx + 1)
                )
            class_names[idx] = str(class_name)

        class_to_id = mapping.get("class_to_id", {})
        ids.extend(int(idx) for idx in class_to_id.values())
        ids.append(background_id)

        num_classes = max(
            max(ids) + 1 if ids else num_classes,
            num_ingredient_classes + 1,
            background_id + 1,
        )

        if len(class_names) < num_classes:
            class_names.extend(
                f"class_{idx}" for idx in range(len(class_names), num_classes)
            )

    if 0 <= background_id < len(class_names):
        class_names[background_id] = "background"

    return {
        "mapping_path": str(mapping_path) if mapping_path.exists() else None,
        "num_classes": num_classes,
        "background_id": background_id,
        "num_ingredient_classes": num_ingredient_classes,
        "class_names": class_names[:num_classes],
    }


def resolve_dataset_meta(cfg: dict) -> dict:
    meta = load_class_mapping(
        data_root=Path(cfg["data_root"]),
        mapping_name=cfg["class_mapping_name"],
        fallback_num_classes=cfg["num_classes"],
        fallback_background_id=cfg["background_id"],
        fallback_num_ingredient_classes=cfg["num_ingredient_classes"],
    )
    resolved = cfg.copy()
    resolved.update(meta)
    return resolved


def set_seed_for_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class RandomResizeCrop:
    def __init__(
        self,
        out_size: Tuple[int, int],
        scale_range: Tuple[float, float],
        hflip_prob: float,
        mean: List[float],
        std: List[float],
        ignore_index: int,
        num_classes: int,
        use_color_jitter: bool,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
    ):
        self.out_h, self.out_w = out_size
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.color_jitter = (
            ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
            if use_color_jitter
            else None
        )

    def __call__(self, image: Image.Image, mask: Image.Image):
        width, height = image.size
        ratio = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = max(32, int(round(width * ratio)))
        new_h = max(32, int(round(height * ratio)))

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        pad_w = max(0, self.out_w - new_w)
        pad_h = max(0, self.out_h - new_h)
        if pad_w > 0 or pad_h > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        top, left, crop_h, crop_w = T.RandomCrop.get_params(
            image,
            output_size=(self.out_h, self.out_w),
        )
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        invalid = (mask < 0) | (mask >= self.num_classes)
        mask[invalid] = self.ignore_index
        return image, mask


class EvalTransform:
    def __init__(
        self,
        mean: List[float],
        std: List[float],
        ignore_index: int,
        num_classes: int,
        out_size: Optional[Tuple[int, int]] = None,
    ):
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.out_size = out_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.out_size is not None:
            out_h, out_w = self.out_size
            image = image.resize((out_w, out_h), Image.BILINEAR)
            mask = mask.resize((out_w, out_h), Image.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        invalid = (mask < 0) | (mask >= self.num_classes)
        mask[invalid] = self.ignore_index
        return image, mask


def find_image_path(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_samples(img_dir: Path, mask_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    for mask_path in sorted(mask_dir.iterdir()):
        if not mask_path.is_file():
            continue
        if mask_path.suffix.lower() != ".png":
            continue

        stem = mask_path.stem
        img_path = find_image_path(img_dir, stem)
        if img_path is None:
            continue
        samples.append((str(img_path), str(mask_path), stem))
    return samples


class FoodSegDataset(Dataset):
    def __init__(self, samples: List[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, stem = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = self.transform(image, mask)
        return image, mask, stem, img_path, mask_path
