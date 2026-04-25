"""
augmentation_ops.py — Augmentation functions cho FoodSeg103 Stage 2.

Cung cấp các hàm augment ở mức ảnh/mask:
  - light_color_aug: color jitter nhẹ, giữ texture food
  - crop_around_bbox: crop quanh vùng rare component
  - random_foreground_crop: crop random quanh foreground
  - paste_patch: copy-paste object vào ảnh khác với feathered blending

Thiết kế:
  - Không phá texture/màu quá mạnh (food-specific)
  - Mask luôn được xử lý song song, đảm bảo class id nhất quán
  - Copy-paste dùng Gaussian blur alpha để bớt viền cứng
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. Light Color Augmentation
# ──────────────────────────────────────────────────────────────────────

def light_color_aug(image: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Color jitter nhẹ cho food images.

    Không nên quá mạnh vì food texture/màu là tín hiệu phân biệt
    giữa các ingredient class (ví dụ: potato vàng vs pumpkin cam).

    Parameters
    ----------
    image : np.ndarray
        Ảnh RGB uint8, shape (H, W, 3).
    rng : np.random.Generator, optional
        Random generator để reproducible.

    Returns
    -------
    np.ndarray
        Ảnh đã augment, cùng shape và dtype.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.float32)

    # Brightness shift nhẹ
    beta = rng.uniform(-12, 12)
    img = img + beta

    # Contrast nhẹ
    alpha = rng.uniform(0.9, 1.1)
    img = (img - 127.5) * alpha + 127.5

    # Saturation nhẹ trong HSV
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.9, 1.1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img


# ──────────────────────────────────────────────────────────────────────
# 2. Crop Around Bbox (Rare-class crop)
# ──────────────────────────────────────────────────────────────────────

def crop_around_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple,
    crop_size: int = 384,
    background_id: int = 103,
) -> tuple:
    """
    Crop ảnh quanh bbox của rare component để giữ object trong frame.

    Crop center tại trung tâm bbox, pad nếu ảnh nhỏ hơn crop_size.
    Background id được dùng để fill vùng pad trong mask.

    Parameters
    ----------
    image : np.ndarray
        Ảnh RGB uint8, shape (H, W, 3).
    mask : np.ndarray
        Mask uint8, shape (H, W).
    bbox : tuple
        (x1, y1, x2, y2) bbox của component.
    crop_size : int
        Kích thước crop vuông.
    background_id : int
        ID background dùng fill pad mask.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (crop_img, crop_mask) đều shape (crop_size, crop_size, ...).
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    half = crop_size // 2

    # Pad nếu ảnh nhỏ hơn crop
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        mask = cv2.copyMakeBorder(
            mask, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=background_id,
        )
        h, w = mask.shape[:2]

    # Tính top-left sao cho center gần bbox center nhất
    left = int(np.clip(cx - half, 0, max(0, w - crop_size)))
    top = int(np.clip(cy - half, 0, max(0, h - crop_size)))

    crop_img = image[top:top + crop_size, left:left + crop_size]
    crop_mask = mask[top:top + crop_size, left:left + crop_size]

    return crop_img, crop_mask


# ──────────────────────────────────────────────────────────────────────
# 3. Random Foreground Crop (fallback)
# ──────────────────────────────────────────────────────────────────────

def random_foreground_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int = 384,
    background_id: int = 103,
    min_fg_ratio: float = 0.03,
    max_attempts: int = 20,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Random crop sao cho vùng crop chứa đủ foreground.

    Thử tối đa max_attempts lần, mỗi lần chọn top-left random
    và check foreground ratio >= min_fg_ratio.
    Nếu không tìm được, trả về center crop.

    Parameters
    ----------
    image : np.ndarray
        Ảnh RGB uint8.
    mask : np.ndarray
        Mask uint8.
    crop_size : int
        Kích thước crop.
    background_id : int
        ID background.
    min_fg_ratio : float
        Tỉ lệ foreground tối thiểu.
    max_attempts : int
        Số lần thử tối đa.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (crop_img, crop_mask).
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = mask.shape[:2]

    # Pad nếu cần
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        mask = cv2.copyMakeBorder(
            mask, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=background_id,
        )
        h, w = mask.shape[:2]

    total_crop_pixels = crop_size * crop_size

    for _ in range(max_attempts):
        top = int(rng.integers(0, max(1, h - crop_size + 1)))
        left = int(rng.integers(0, max(1, w - crop_size + 1)))

        crop_mask = mask[top:top + crop_size, left:left + crop_size]
        fg_ratio = float(np.sum(crop_mask != background_id)) / total_crop_pixels

        if fg_ratio >= min_fg_ratio:
            crop_img = image[top:top + crop_size, left:left + crop_size]
            return crop_img, crop_mask

    # Fallback: center crop
    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)
    return (
        image[top:top + crop_size, left:left + crop_size],
        mask[top:top + crop_size, left:left + crop_size],
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Copy-Paste Augmentation
# ──────────────────────────────────────────────────────────────────────

def paste_patch(
    base_img: np.ndarray,
    base_mask: np.ndarray,
    patch_img: np.ndarray,
    patch_mask_bin: np.ndarray,
    class_id: int,
    x: int,
    y: int,
) -> tuple:
    """
    Paste một object patch vào ảnh base.

    Dùng Gaussian blur trên alpha mask để feather viền,
    tránh artifact cứng khi paste.

    Parameters
    ----------
    base_img : np.ndarray
        Ảnh nền RGB uint8, shape (H, W, 3).
    base_mask : np.ndarray
        Mask nền uint8, shape (H, W).
    patch_img : np.ndarray
        Patch RGB uint8, shape (ph, pw, 3).
    patch_mask_bin : np.ndarray
        Binary mask uint8 (0 hoặc 255), shape (ph, pw).
    class_id : int
        Class id để ghi vào mask.
    x : int
        Tọa độ x paste (top-left).
    y : int
        Tọa độ y paste (top-left).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (out_img, out_mask) đã paste.
    """
    out_img = base_img.copy()
    out_mask = base_mask.copy()

    h, w = base_mask.shape[:2]
    ph, pw = patch_mask_bin.shape[:2]

    # Clamp vị trí paste
    x = int(np.clip(x, 0, max(0, w - pw)))
    y = int(np.clip(y, 0, max(0, h - ph)))

    # Tính alpha feathered
    alpha = (patch_mask_bin > 0).astype(np.float32)
    alpha_blur = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha_blur = alpha_blur[..., np.newaxis]  # (ph, pw, 1)

    # Blend
    region = out_img[y:y + ph, x:x + pw].astype(np.float32)
    patch = patch_img[:ph, :pw].astype(np.float32)

    # Đảm bảo shapes match (xử lý edge case)
    rh, rw = region.shape[:2]
    if rh < ph or rw < pw:
        ph, pw = rh, rw
        alpha_blur = alpha_blur[:ph, :pw]
        patch = patch[:ph, :pw]
        alpha = alpha[:ph, :pw]
        region = out_img[y:y + ph, x:x + pw].astype(np.float32)

    blended = region * (1 - alpha_blur) + patch * alpha_blur
    out_img[y:y + ph, x:x + pw] = np.clip(blended, 0, 255).astype(np.uint8)

    # Ghi mask
    paste_pixels = alpha[:ph, :pw] > 0
    out_mask[y:y + ph, x:x + pw][paste_pixels] = class_id

    return out_img, out_mask
