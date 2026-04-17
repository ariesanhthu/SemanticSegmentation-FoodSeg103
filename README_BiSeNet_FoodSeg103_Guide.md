# BiSeNet V1 + FoodSeg103 Benchmark Guide

Mục tiêu của guide này là dựng một pipeline **chuẩn, sạch, dễ đọc, tách file rõ ràng** cho bài toán train **BiSeNet V1** trên **FoodSeg103** theo tinh thần:

- **Chuẩn BiSeNet V1 paper** về kiến trúc và tối ưu hóa.
- **Chuẩn FoodSeg103 benchmark** về split và metric.
- **Code clean**, comment rõ, config tách riêng, dễ sửa từng phần.

---

## 1) Nguyên tắc chuẩn phải giữ

### 1.1. Chuẩn BiSeNet V1

Giữ đúng các điểm cốt lõi của paper:

- kiến trúc gồm:
  - `Spatial Path`
  - `Context Path`
  - `Attention Refinement Module (ARM)`
  - `Feature Fusion Module (FFM)`
  - `main head + 2 auxiliary heads`
- backbone context path kiểu `Xception39-like`
- optimizer: `SGD`
- scheduler: `poly lr`
- loss: `main CE + aux16 CE + aux32 CE`
- random scale + random crop + horizontal flip

### 1.2. Chuẩn FoodSeg103 benchmark

Giữ đúng logic benchmark:

- dataset gốc chỉ có **2 split chính**:
  - `train`
  - `test`
- trong repo benchmark, `val` và `test` cùng trỏ vào `official test`
- metric là **benchmark-style semantic segmentation metrics**:
  - `mIoU`
  - `mAcc`
  - `aAcc`
- metric phải tính trên **toàn bộ split eval**, không phải kiểu image-wise present-class.

---

## 2) Quy ước thực hiện trong project này

Để bám paper + benchmark nhưng vẫn clean và dễ debug, ta dùng quy ước sau:

### 2.1. Split

- train trên `official train`
- eval trong lúc train trên `official test`
- test cuối cũng là `official test`

> Đây là đúng tinh thần repo benchmark.
> Nếu sau này muốn làm research nghiêm ngặt hơn, có thể tách thêm internal val ở một nhánh khác, nhưng **branch chuẩn benchmark** này không làm vậy.

### 2.2. Metric

Dùng benchmark metric chuẩn:

Với class `c`:

\[
IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}
\]

\[
mIoU = \frac{1}{C'}\sum_{c \in \mathcal{V}} IoU_c
\]

Trong đó:

- `hist[row, col]` = số pixel có `GT=row`, `Pred=col`
- `\mathcal{V}` là tập class **thực sự có GT pixel trong split eval**
- `C' = |\mathcal{V}|`

Per-class accuracy:

\[
Acc_c = \frac{TP_c}{TP_c + FN_c}
\]

\[
mAcc = \frac{1}{C'}\sum_{c \in \mathcal{V}} Acc_c
\]

All-pixel accuracy:

\[
aAcc = \frac{\sum_c TP_c}{\text{all valid pixels}}
\]

### 2.3. Augmentation

Dùng phiên bản clean, bám sát paper và benchmark:

- random resize theo ratio range `0.5 -> 2.0`
- random horizontal flip
- random crop `512 x 1024`
- normalize theo ImageNet mean/std

> BiSeNet paper dùng scale set rời rạc.
> FoodSeg103 benchmark baselines dùng resize ratio range và crop.
> Ở đây ta chọn bản clean, gần benchmark hơn cho FoodSeg103, nhưng vẫn giữ đúng tinh thần BiSeNet.

---

## 3) Cấu trúc thư mục chuẩn

```text
bisenet_foodseg103/
├─ configs/
│  └─ bisenet_foodseg103.py
├─ datasets/
│  └─ foodseg103.py
├─ models/
│  └─ bisenetv1.py
├─ utils/
│  ├─ metrics.py
│  └─ misc.py
├─ tools/
│  ├─ train.py
│  └─ eval.py
└─ README_BiSeNet_FoodSeg103_Guide.md
```

### Ý nghĩa từng file

- `configs/bisenet_foodseg103.py`
  - toàn bộ config ở một nơi
  - sửa batch size, lr, crop size, work_dir, data_root ở đây

- `datasets/foodseg103.py`
  - đọc dữ liệu
  - transform train/eval
  - build sample list

- `models/bisenetv1.py`
  - định nghĩa toàn bộ BiSeNet V1 sạch, tách module rõ ràng

- `utils/metrics.py`
  - confusion matrix + benchmark metrics chuẩn

- `utils/misc.py`
  - seed, checkpoint, save/load tiện ích chung

- `tools/train.py`
  - train loop chuẩn
  - eval mỗi epoch
  - lưu `last` và `best_miou`

- `tools/eval.py`
  - load checkpoint tốt nhất và đánh giá lại trên official test

---

## 4) Dataset layout kỳ vọng

Project này giả định cấu trúc dataset như sau:

```text
/content/data/FoodSeg103/
├─ Images/
│  ├─ img_dir/
│  │  ├─ train/
│  │  └─ test/
│  └─ ann_dir/
│     ├─ train/
│     └─ test/
```

- `img_dir/train`: ảnh train
- `ann_dir/train`: mask train
- `img_dir/test`: ảnh test
- `ann_dir/test`: mask test

Nếu dataset của bạn đặt ở chỗ khác, chỉ sửa `data_root` và các path con trong file config.

---

## 5) File config

### `configs/bisenet_foodseg103.py`

```python
from pathlib import Path
import torch

CFG = {
    # ---------------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------------
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "cudnn_benchmark": True,

    # ---------------------------------------------------------------------
    # Dataset
    # Benchmark-aligned mode:
    #   - train split: official train
    #   - val split  : official test
    #   - test split : official test
    # This mirrors the FoodSeg103 benchmark repository behavior.
    # ---------------------------------------------------------------------
    "data_root": "/content/data/FoodSeg103",
    "train_img_dir": "Images/img_dir/train",
    "train_mask_dir": "Images/ann_dir/train",
    "test_img_dir": "Images/img_dir/test",
    "test_mask_dir": "Images/ann_dir/test",
    "num_classes": 104,
    "ignore_index": 255,
    "background_id": 0,
    "image_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],

    # ---------------------------------------------------------------------
    # Input / augmentation
    # ---------------------------------------------------------------------
    "train_size": (512, 1024),
    "test_size": None,
    "scale_range": (0.5, 2.0),
    "hflip_prob": 0.5,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],

    # ---------------------------------------------------------------------
    # Dataloader
    # ---------------------------------------------------------------------
    "batch_size": 8,
    "eval_batch_size": 4,
    "num_workers": 2,
    "pin_memory": True,
    "drop_last": True,

    # ---------------------------------------------------------------------
    # Optimizer / scheduler
    # ---------------------------------------------------------------------
    "epochs": 80,
    "lr": 2.5e-2,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "poly_power": 0.9,
    "aux_weight": 1.0,
    "grad_clip_norm": None,

    # ---------------------------------------------------------------------
    # Checkpoint / logging
    # ---------------------------------------------------------------------
    "work_dir": "/content/drive/MyDrive/[PROJECT][COMPUTER-VISION]/bisenet_foodseg103_benchmark",
    "resume": True,
    "save_last_name": "last.pth",
    "save_best_name": "best_miou.pth",
    "print_freq": 20,
    "eval_every": 1,

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    "num_vis": 4,
}


def get_paths(cfg: dict) -> dict:
    root = Path(cfg["data_root"])
    return {
        "train_img_dir": root / cfg["train_img_dir"],
        "train_mask_dir": root / cfg["train_mask_dir"],
        "test_img_dir": root / cfg["test_img_dir"],
        "test_mask_dir": root / cfg["test_mask_dir"],
        "work_dir": Path(cfg["work_dir"]),
    }
```

### Ghi chú quan trọng

- `num_classes = 104` để bám **FoodSeg103 benchmark repo**.
- Nếu bạn dùng bản cleaned/rebalanced riêng của bạn thì class count có thể khác. Khi đó **đừng gọi đó là benchmark mode nữa**.

---

## 6) Dataset code

### `datasets/foodseg103.py`

```python
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


Sample = Tuple[str, str, str]


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def set_seed_for_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class RandomResizeCrop:
    def __init__(self, out_size, scale_range, hflip_prob, mean, std, ignore_index, num_classes):
        self.out_h, self.out_w = out_size
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def __call__(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        ratio = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = max(32, int(round(w * ratio)))
        new_h = max(32, int(round(h * ratio)))

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        pad_w = max(0, self.out_w - new_w)
        pad_h = max(0, self.out_h - new_h)
        if pad_w > 0 or pad_h > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        i, j, h_crop, w_crop = torch.randint(0, max(1, image.height - self.out_h + 1), (1,)).item(), \
                               torch.randint(0, max(1, image.width - self.out_w + 1), (1,)).item(), \
                               self.out_h, self.out_w
        image = TF.crop(image, i, j, h_crop, w_crop)
        mask = TF.crop(mask, i, j, h_crop, w_crop)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        invalid = (mask < 0) | (mask >= self.num_classes)
        mask[invalid] = self.ignore_index
        return image, mask


class EvalTransform:
    def __init__(self, mean, std, ignore_index, num_classes, out_size=None):
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


def find_image_path(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, stem = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = self.transform(image, mask)
        return image, mask, stem, img_path, mask_path
```

---

## 7) Model code

### `models/bisenetv1.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class XBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

        layers = []
        filters = in_ch
        if grow_first:
            layers.append(self.relu if start_with_relu else nn.Identity())
            layers.append(SeparableConv2d(in_ch, out_ch, 3, 1, 1, False))
            filters = out_ch

        for _ in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(filters, filters, 3, 1, 1, False))

        if not grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_ch, out_ch, 3, 1, 1, False))

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))

        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        return self.rep(x) + self.skip(x)


class Xception39(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = XBlock(16, 32, reps=3, stride=2, start_with_relu=False, grow_first=True)
        self.block2 = XBlock(32, 64, reps=3, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = XBlock(64, 128, reps=3, stride=2, start_with_relu=True, grow_first=True)
        self.out_channels = (32, 64, 128)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        feat8 = self.block1(x)
        feat16 = self.block2(feat8)
        feat32 = self.block3(feat16)
        return feat8, feat16, feat32


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        self.attn_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.attn_bn = nn.BatchNorm2d(out_ch)
        self.attn_sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        attn = F.adaptive_avg_pool2d(feat, output_size=1)
        attn = self.attn_conv(attn)
        attn = self.attn_bn(attn)
        attn = self.attn_sigmoid(attn)
        return feat * attn


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, 7, 2, 3)
        self.conv2 = ConvBNReLU(64, 64, 3, 2, 1)
        self.conv3 = ConvBNReLU(64, 64, 3, 2, 1)
        self.conv_out = ConvBNReLU(64, 128, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class ContextPath(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = Xception39() if backbone is None else backbone
        _, c16, c32 = self.backbone.out_channels

        self.arm16 = AttentionRefinementModule(c16, 128)
        self.arm32 = AttentionRefinementModule(c32, 128)
        self.global_context = ConvBNReLU(c32, 128, 1, 1, 0)
        self.refine16 = ConvBNReLU(128, 128, 3, 1, 1)
        self.refine32 = ConvBNReLU(128, 128, 3, 1, 1)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)

        tail = F.adaptive_avg_pool2d(feat32, output_size=1)
        tail = self.global_context(tail)
        tail = F.interpolate(tail, size=feat32.shape[-2:], mode="bilinear", align_corners=False)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + tail
        feat32_up = F.interpolate(self.refine32(feat32_sum), size=feat16.shape[-2:], mode="bilinear", align_corners=False)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(self.refine16(feat16_sum), size=feat8.shape[-2:], mode="bilinear", align_corners=False)

        return feat16_up, feat16_sum, feat32_sum


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblk = ConvBNReLU(in_ch, out_ch, 1, 1, 0)
        self.attn1 = nn.Conv2d(out_ch, out_ch // 4, kernel_size=1, bias=False)
        self.attn_relu = nn.ReLU(inplace=True)
        self.attn2 = nn.Conv2d(out_ch // 4, out_ch, kernel_size=1, bias=False)
        self.attn_sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(feat)
        attn = F.adaptive_avg_pool2d(feat, output_size=1)
        attn = self.attn1(attn)
        attn = self.attn_relu(attn)
        attn = self.attn2(attn)
        attn = self.attn_sigmoid(attn)
        return feat + feat * attn


class SegHead(nn.Module):
    def __init__(self, in_ch, mid_ch, num_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, mid_ch, 3, 1, 1)
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(mid_ch, num_classes, kernel_size=1)

    def forward(self, x, out_size=None):
        x = self.conv(x)
        x = self.drop(x)
        x = self.cls(x)
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class BiSeNetV1(nn.Module):
    def __init__(self, num_classes=19, backbone=None):
        super().__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath(backbone)
        self.ffm = FeatureFusionModule(256, 256)
        self.head = SegHead(256, 256, num_classes)
        self.aux_head16 = SegHead(128, 64, num_classes)
        self.aux_head32 = SegHead(128, 64, num_classes)

    def forward(self, x):
        out_size = x.shape[-2:]
        feat_sp = self.spatial_path(x)
        feat_cp8, feat_cp16, feat_cp32 = self.context_path(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        logits = self.head(feat_fuse, out_size)
        aux16 = self.aux_head16(feat_cp16, out_size)
        aux32 = self.aux_head32(feat_cp32, out_size)

        if self.training:
            return logits, aux16, aux32
        return logits
```

### Ghi chú quan trọng

Đây là bản **BiSeNet V1 sạch, tự code**, bám paper:

- `Spatial Path`: 3 lần stride-2 để giữ chi tiết không gian ở mức /8
- `Context Path`: backbone downsample nhanh để lấy receptive field lớn
- `ARM`: reweight context feature bằng global descriptor
- `FFM`: trộn feature low-level và high-level theo gating
- `2 aux heads`: deep supervision đúng tinh thần paper

---

## 8) Metrics code

### `utils/metrics.py`

```python
import torch


@torch.no_grad()
def fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    valid = (target >= 0) & (target < num_classes)
    indices = num_classes * target[valid].to(torch.int64) + pred[valid].to(torch.int64)
    hist = torch.bincount(indices, minlength=num_classes ** 2)
    return hist.reshape(num_classes, num_classes)


@torch.no_grad()
def compute_segmentation_scores(hist: torch.Tensor):
    """
    Benchmark-style metrics on one whole evaluation split.

    hist[row, col] = number of pixels with GT=row and PRED=col.
    """
    hist = hist.float()

    # All-pixel accuracy.
    aacc = torch.diag(hist).sum() / hist.sum().clamp_min(1.0)

    # Per-class accuracy.
    acc_cls = torch.diag(hist) / hist.sum(dim=1).clamp_min(1.0)
    valid_acc = hist.sum(dim=1) > 0
    macc = acc_cls[valid_acc].mean() if valid_acc.any() else torch.tensor(0.0, device=hist.device)

    # Per-class IoU.
    denom = hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)
    iou = torch.diag(hist) / denom.clamp_min(1.0)
    valid_iou = hist.sum(dim=1) > 0
    miou = iou[valid_iou].mean() if valid_iou.any() else torch.tensor(0.0, device=hist.device)

    return {
        "aAcc": float(aacc.item()),
        "mAcc": float(macc.item()),
        "mIoU": float(miou.item()),
        "IoU_per_class": iou.detach().cpu(),
        "Acc_per_class": acc_cls.detach().cpu(),
        "valid_class_mask": valid_iou.detach().cpu(),
    }
```

### Vì sao metric này là đúng benchmark?

- không average theo ảnh
- không chỉ lấy class hiện có trong từng ảnh rồi average
- mà **gom toàn bộ split eval** thành một confusion matrix chung
- sau đó tính `IoU_c`, `Acc_c`, rồi mới lấy mean trên class hợp lệ

Đây là cách đúng để bám benchmark segmentation.

---

## 9) Misc utils

### `utils/misc.py`

```python
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def save_checkpoint(path, epoch, global_iter, best_miou, model, optimizer, scaler, cfg):
    torch.save(
        {
            "epoch": epoch,
            "global_iter": global_iter,
            "best_miou": best_miou,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg,
        },
        path,
    )
```

---

## 10) Train loop

### `tools/train.py`

```python
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.bisenet_foodseg103 import CFG, get_paths
from datasets.foodseg103 import (
    FoodSegDataset,
    RandomResizeCrop,
    EvalTransform,
    build_samples,
    set_seed_for_worker,
)
from models.bisenetv1 import BiSeNetV1
from utils.metrics import fast_hist, compute_segmentation_scores
from utils.misc import seed_everything, ensure_dir, load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def build_loaders(cfg):
    paths = get_paths(cfg)
    train_samples = build_samples(paths["train_img_dir"], paths["train_mask_dir"])
    test_samples = build_samples(paths["test_img_dir"], paths["test_mask_dir"])

    train_tf = RandomResizeCrop(
        out_size=cfg["train_size"],
        scale_range=cfg["scale_range"],
        hflip_prob=cfg["hflip_prob"],
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
    )
    eval_tf = EvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=None,
    )

    train_ds = FoodSegDataset(train_samples, train_tf)
    eval_ds = FoodSegDataset(test_samples, eval_tf)

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


def poly_lr(base_lr, cur_iter, max_iter, power=0.9):
    return base_lr * (1.0 - float(cur_iter) / float(max_iter)) ** power


@torch.no_grad()
def evaluate(model, loader, cfg, criterion):
    model.eval()
    hist = torch.zeros((cfg["num_classes"], cfg["num_classes"]), device=cfg["device"])
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


def main():
    args = parse_args()
    cfg = CFG.copy()
    seed_everything(cfg["seed"])
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True

    paths = get_paths(cfg)
    ensure_dir(paths["work_dir"])

    train_loader, eval_loader = build_loaders(cfg)

    model = BiSeNetV1(num_classes=cfg["num_classes"]).to(cfg["device"])
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])

    last_path = paths["work_dir"] / cfg["save_last_name"]
    best_path = paths["work_dir"] / cfg["save_best_name"]

    start_epoch = 0
    global_iter = 0
    best_miou = -1.0

    if cfg["resume"] and last_path.exists():
        ckpt = load_checkpoint(last_path, model, optimizer, scaler, map_location=cfg["device"])
        start_epoch = ckpt["epoch"] + 1
        global_iter = ckpt.get("global_iter", 0)
        best_miou = ckpt.get("best_miou", -1.0)
        print(f"Resumed from {last_path} at epoch={start_epoch}")

    if args.eval_only:
        scores = evaluate(model, eval_loader, cfg, criterion)
        print(scores)
        return

    max_iter = max(1, cfg["epochs"] * len(train_loader))

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch:03d}")

        for step, (images, masks, stems, *_rest) in enumerate(pbar):
            images = images.to(cfg["device"], non_blocking=True)
            masks = masks.to(cfg["device"], non_blocking=True)

            cur_lr = poly_lr(cfg["lr"], global_iter, max_iter, cfg["poly_power"])
            for group in optimizer.param_groups:
                group["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["amp"]):
                logits, aux16, aux32 = model(images)
                loss_main = criterion(logits, masks)
                loss_aux16 = criterion(aux16, masks)
                loss_aux32 = criterion(aux32, masks)
                loss = loss_main + cfg["aux_weight"] * (loss_aux16 + loss_aux32)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            global_iter += 1

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

        train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        if (epoch + 1) % cfg["eval_every"] == 0:
            scores = evaluate(model, eval_loader, cfg, criterion)
            print(
                f"Eval {epoch:03d} | loss={scores['loss']:.4f} | "
                f"mIoU={scores['mIoU']:.4f} | mAcc={scores['mAcc']:.4f} | aAcc={scores['aAcc']:.4f}"
            )

            if scores["mIoU"] > best_miou:
                best_miou = scores["mIoU"]
                save_checkpoint(best_path, epoch, global_iter, best_miou, model, optimizer, scaler, cfg)
                print(f"Saved new best checkpoint to {best_path}")

        save_checkpoint(last_path, epoch, global_iter, best_miou, model, optimizer, scaler, cfg)


if __name__ == "__main__":
    main()
```

### Điểm chuẩn trong train loop

- `CrossEntropyLoss(ignore_index=255)`
- `SGD + poly lr`
- `main + aux16 + aux32`
- eval sau mỗi epoch trên `official test`
- lưu `last.pth`
- lưu `best_miou.pth`

---

## 11) Eval script

### `tools/eval.py`

```python
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from configs.bisenet_foodseg103 import CFG, get_paths
from datasets.foodseg103 import FoodSegDataset, EvalTransform, build_samples
from models.bisenetv1 import BiSeNetV1
from utils.metrics import fast_hist, compute_segmentation_scores
from utils.misc import load_checkpoint


def main():
    cfg = CFG.copy()
    paths = get_paths(cfg)

    samples = build_samples(paths["test_img_dir"], paths["test_mask_dir"])
    transform = EvalTransform(
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        ignore_index=cfg["ignore_index"],
        num_classes=cfg["num_classes"],
        out_size=None,
    )
    loader = DataLoader(FoodSegDataset(samples, transform), batch_size=cfg["eval_batch_size"], shuffle=False)

    model = BiSeNetV1(num_classes=cfg["num_classes"]).to(cfg["device"])
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    ckpt_path = paths["work_dir"] / cfg["save_best_name"]
    load_checkpoint(ckpt_path, model, map_location=cfg["device"])

    hist = torch.zeros((cfg["num_classes"], cfg["num_classes"]), device=cfg["device"])
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, masks, *_ in loader:
            images = images.to(cfg["device"])
            masks = masks.to(cfg["device"])
            logits = model(images)
            running_loss += float(criterion(logits, masks).item())
            preds = logits.argmax(1)
            hist += fast_hist(preds, masks, cfg["num_classes"], cfg["ignore_index"])

    scores = compute_segmentation_scores(hist)
    scores["loss"] = running_loss / max(1, len(loader))
    print(scores)


if __name__ == "__main__":
    main()
```

---

## 12) Cách chạy

### 12.1. Train

```bash
python tools/train.py
```

### 12.2. Chỉ eval checkpoint hiện tại

```bash
python tools/train.py --eval-only
```

hoặc:

```bash
python tools/eval.py
```

---

## 13) Workflow chuẩn từ đầu đến cuối

### Bước 1 — Chuẩn bị dữ liệu

- tải FoodSeg103 đúng cấu trúc benchmark
- giải nén vào `data_root`
- kiểm tra thư mục:
  - `img_dir/train`
  - `ann_dir/train`
  - `img_dir/test`
  - `ann_dir/test`

### Bước 2 — Sửa config duy nhất ở một nơi

Chỉ sửa trong:

```python
configs/bisenet_foodseg103.py
```

Các biến thường sửa:

- `data_root`
- `work_dir`
- `batch_size`
- `epochs`
- `num_workers`

### Bước 3 — Train chuẩn benchmark

```bash
python tools/train.py
```

### Bước 4 — Lấy model tốt nhất

Script sẽ tự lưu:

- `last.pth`
- `best_miou.pth`

### Bước 5 — Test cuối

```bash
python tools/eval.py
```

---

## 14) Những thứ KHÔNG làm trong branch benchmark này

Để branch này giữ đúng chuẩn benchmark, **không làm** các thứ sau trong code chính:

- không split thêm internal validation
- không đổi metric sang present-class image-wise
- không thêm weighted CE / Dice vào branch chuẩn benchmark
- không thêm texture branch vào branch chuẩn benchmark
- không thay BiSeNet V1 bằng BiSeNet V2
- không thay Xception39-like bằng backbone khác

Những thứ đó nên để ở branch thử nghiệm khác.

---

## 15) Gợi ý branch sạch để làm việc lâu dài

Nên tách thành 3 branch logic:

### A. `benchmark_bisenet_v1`

Mục tiêu:
- bám paper BiSeNet
- bám FoodSeg103 benchmark
- kết quả baseline chuẩn

### B. `practical_cleaned_dataset`

Mục tiêu:
- dùng cleaned/rebalanced dataset riêng của bạn
- chấp nhận class count khác
- chấp nhận split khác
- chấp nhận metric debug khác

### C. `texture_aware_variant`

Mục tiêu:
- thêm nhánh texture-aware
- so với baseline chuẩn từ branch A

Như vậy baseline chuẩn sẽ không bị pha tạp.

---

## 16) Checklist trước khi chạy thật

### Dataset
- [ ] đúng official FoodSeg103 benchmark split
- [ ] mask values không vượt `num_classes - 1`
- [ ] mask invalid được đưa về `ignore_index`

### Model
- [ ] `BiSeNetV1(num_classes=104)`
- [ ] `main + aux16 + aux32`

### Train
- [ ] `SGD`
- [ ] `lr = 2.5e-2`
- [ ] `momentum = 0.9`
- [ ] `weight_decay = 1e-4`
- [ ] `poly lr`

### Metric
- [ ] confusion matrix toàn split
- [ ] `mIoU`, `mAcc`, `aAcc`
- [ ] không dùng metric image-wise custom

---

## 17) Kết luận

Nếu mục tiêu của bạn là:

> **“train BiSeNet nhưng chuẩn, bám paper benchmark, phương pháp chuẩn từ đầu đến cuối”**

thì branch/code guide này chính là cách làm đúng nhất:

- kiến trúc chuẩn BiSeNet V1
- optimizer/schedule chuẩn paper
- split chuẩn benchmark FoodSeg103
- metric chuẩn benchmark segmentation
- code clean, tách config rõ, dễ đọc, dễ sửa

Sau khi baseline này chạy ổn, mọi cải tiến sau đó mới có ý nghĩa so sánh.
