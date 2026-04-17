from pathlib import Path
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------
# Mỗi sample gồm:
# - đường dẫn ảnh RGB
# - đường dẫn mask PNG
# - stem chung của file
Sample = Tuple[str, str, str]


# ---------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------
# Mask benchmark thường lưu dạng .png, còn ảnh có thể ở nhiều định dạng khác nhau.
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def set_seed_for_worker(worker_id: int) -> None:
    """
    Khởi tạo random seed riêng cho từng DataLoader worker.

    Hàm này thường được truyền vào `worker_init_fn` của PyTorch DataLoader
    để mỗi worker có trạng thái random độc lập nhưng vẫn reproducible.

    Vì mỗi worker là một process riêng, nếu không seed lại thì:
    - random augmentation có thể bị trùng giữa các worker
    - kết quả train có thể khó tái lập

    Parameters
    ----------
    worker_id : int
        ID của worker do PyTorch cấp.
        Biến này không được dùng trực tiếp trong tính seed, nhưng vẫn giữ
        trong signature để tương thích với `worker_init_fn`.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class RandomResizeCrop:
    """
    Transform dùng cho train:
    - random resize theo scale range
    - random horizontal flip
    - pad nếu ảnh nhỏ hơn crop size
    - random crop về kích thước cố định
    - convert sang tensor + normalize
    - chuẩn hóa mask và gán ignore_index cho label không hợp lệ

    Đây là transform train kiểu segmentation cơ bản, bám sát tinh thần:
    - random scale
    - random horizontal flip
    - random crop

    Parameters
    ----------
    out_size : Tuple[int, int]
        Kích thước đầu ra cuối cùng sau crop, theo dạng (height, width).

    scale_range : Tuple[float, float]
        Khoảng scale ngẫu nhiên để resize ảnh trước khi crop.
        Ví dụ: (0.5, 2.0).

    hflip_prob : float
        Xác suất lật ngang ảnh và mask.

    mean : List[float] | Tuple[float, float, float]
        Mean dùng để normalize ảnh RGB.

    std : List[float] | Tuple[float, float, float]
        Std dùng để normalize ảnh RGB.

    ignore_index : int
        Giá trị dùng để đánh dấu pixel mask không hợp lệ / bị ignore.

    num_classes : int
        Tổng số class hợp lệ trong bài toán segmentation.
        Pixel nào ngoài [0, num_classes - 1] sẽ bị gán ignore_index.
    """

    def __init__(
        self,
        out_size,
        scale_range,
        hflip_prob,
        mean,
        std,
        ignore_index,
        num_classes,
    ):
        self.out_h, self.out_w = out_size
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def __call__(self, image: Image.Image, mask: Image.Image):
        """
        Áp dụng toàn bộ pipeline augmentation/train preprocessing.

        Parameters
        ----------
        image : PIL.Image.Image
            Ảnh RGB đầu vào.

        mask : PIL.Image.Image
            Mask phân đoạn đầu vào. Giá trị pixel là class id.

        Returns
        -------
        image : torch.Tensor
            Tensor ảnh đã normalize, shape [3, H, W].

        mask : torch.Tensor
            Tensor mask kiểu int64, shape [H, W].
            Các pixel không hợp lệ được gán `ignore_index`.
        """
        # -------------------------------------------------------------
        # 1) Random resize
        # -------------------------------------------------------------
        # Chọn ngẫu nhiên một scale ratio trong khoảng cấu hình.
        w, h = image.size
        ratio = random.uniform(self.scale_range[0], self.scale_range[1])

        # Bảo đảm chiều mới không quá nhỏ để tránh lỗi crop / tensor rỗng.
        new_w = max(32, int(round(w * ratio)))
        new_h = max(32, int(round(h * ratio)))

        # Ảnh dùng bilinear để nội suy mượt.
        # Mask dùng nearest để không sinh class id lạ.
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # -------------------------------------------------------------
        # 2) Random horizontal flip
        # -------------------------------------------------------------
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # -------------------------------------------------------------
        # 3) Pad nếu ảnh nhỏ hơn crop size
        # -------------------------------------------------------------
        # Sau random scale, ảnh có thể nhỏ hơn kích thước crop mong muốn.
        # Khi đó cần pad để crop luôn hợp lệ.
        pad_w = max(0, self.out_w - new_w)
        pad_h = max(0, self.out_h - new_h)

        if pad_w > 0 or pad_h > 0:
            # Ảnh pad bằng 0 (đen).
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)

            # Mask pad bằng ignore_index để vùng pad không ảnh hưởng loss/metric.
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=self.ignore_index)

        # -------------------------------------------------------------
        # 4) Random crop về kích thước cố định
        # -------------------------------------------------------------
        # Vì ảnh lúc này chắc chắn đã >= crop size, ta chọn ngẫu nhiên góc trái trên.
        max_top = max(1, image.height - self.out_h + 1)
        max_left = max(1, image.width - self.out_w + 1)

        i = torch.randint(0, max_top, (1,)).item()
        j = torch.randint(0, max_left, (1,)).item()

        h_crop = self.out_h
        w_crop = self.out_w

        image = TF.crop(image, i, j, h_crop, w_crop)
        mask = TF.crop(mask, i, j, h_crop, w_crop)

        # -------------------------------------------------------------
        # 5) Convert image sang tensor + normalize
        # -------------------------------------------------------------
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        # -------------------------------------------------------------
        # 6) Convert mask sang tensor int64
        # -------------------------------------------------------------
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # Mọi label ngoài range hợp lệ đều bị đưa về ignore_index.
        invalid = (mask < 0) | (mask >= self.num_classes)
        mask[invalid] = self.ignore_index

        return image, mask


class EvalTransform:
    """
    Transform dùng cho evaluation / validation / test:
    - có thể resize về một kích thước cố định nếu cần
    - convert sang tensor + normalize
    - chuẩn hóa mask và gán ignore_index cho label không hợp lệ

    Khác với train transform:
    - không random scale
    - không flip
    - không random crop

    Parameters
    ----------
    mean : List[float] | Tuple[float, float, float]
        Mean dùng để normalize ảnh RGB.

    std : List[float] | Tuple[float, float, float]
        Std dùng để normalize ảnh RGB.

    ignore_index : int
        Giá trị dùng để đánh dấu pixel mask không hợp lệ / bị ignore.

    num_classes : int
        Tổng số class hợp lệ trong bài toán segmentation.

    out_size : Optional[Tuple[int, int]], default=None
        Nếu không None, resize ảnh và mask về (height, width) trước khi convert.
        Nếu None, giữ nguyên kích thước gốc.
    """

    def __init__(self, mean, std, ignore_index, num_classes, out_size=None):
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.out_size = out_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        """
        Áp dụng pipeline preprocessing cho evaluation.

        Parameters
        ----------
        image : PIL.Image.Image
            Ảnh RGB đầu vào.

        mask : PIL.Image.Image
            Mask phân đoạn đầu vào.

        Returns
        -------
        image : torch.Tensor
            Tensor ảnh đã normalize, shape [3, H, W].

        mask : torch.Tensor
            Tensor mask int64, shape [H, W].
        """
        # Resize cố định nếu config yêu cầu.
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
    """
    Tìm file ảnh tương ứng với một stem trong thư mục ảnh.

    Vì ảnh có thể được lưu dưới nhiều extension khác nhau, hàm này sẽ
    thử lần lượt các extension trong `IMG_EXTS`.

    Parameters
    ----------
    img_dir : Path
        Thư mục chứa ảnh RGB.

    stem : str
        Tên file không kèm extension.

    Returns
    -------
    Optional[Path]
        Path đến file ảnh nếu tìm thấy, ngược lại trả về None.
    """
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_samples(img_dir: Path, mask_dir: Path) -> List[Sample]:
    """
    Xây danh sách sample từ thư mục ảnh và thư mục mask.

    Logic:
    - duyệt toàn bộ file mask `.png`
    - lấy `stem` của mask
    - tìm ảnh RGB tương ứng trong `img_dir`
    - nếu tìm thấy cặp ảnh-mask hợp lệ thì thêm vào danh sách samples

    Chọn duyệt theo mask trước là hợp lý vì:
    - segmentation benchmark thường coi mask là nguồn ground-truth chính
    - chỉ giữ sample nào chắc chắn có annotation

    Parameters
    ----------
    img_dir : Path
        Thư mục chứa ảnh.

    mask_dir : Path
        Thư mục chứa mask `.png`.

    Returns
    -------
    List[Sample]
        Danh sách sample dạng:
        `(img_path_str, mask_path_str, stem)`
    """
    samples: List[Sample] = []

    for mask_path in sorted(mask_dir.iterdir()):
        # Bỏ qua nếu không phải file thật.
        if not mask_path.is_file():
            continue

        # Chỉ nhận mask PNG.
        if mask_path.suffix.lower() != ".png":
            continue

        stem = mask_path.stem
        img_path = find_image_path(img_dir, stem)

        # Nếu không có ảnh tương ứng thì bỏ qua sample này.
        if img_path is None:
            continue

        samples.append((str(img_path), str(mask_path), stem))

    return samples


class FoodSegDataset(Dataset):
    """
    Dataset cơ bản cho bài toán semantic segmentation của FoodSeg-style benchmark.

    Dataset này:
    - nhận vào danh sách samples đã được build sẵn
    - đọc ảnh RGB và mask từ disk
    - áp dụng transform
    - trả ra tensor ảnh, tensor mask, stem, và đường dẫn gốc

    Output đầy đủ giúp:
    - train/eval được bình thường
    - dễ debug sample lỗi
    - dễ visualize theo stem/path sau này

    Parameters
    ----------
    samples : List[Sample]
        Danh sách sample dạng `(img_path, mask_path, stem)`.

    transform : callable
        Một transform nhận `(image, mask)` và trả về `(image_tensor, mask_tensor)`.
        Có thể là `RandomResizeCrop`, `EvalTransform`, hoặc transform custom khác.
    """

    def __init__(self, samples: List[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """
        Trả về số lượng sample trong dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Đọc một sample tại vị trí `idx`.

        Parameters
        ----------
        idx : int
            Chỉ số sample cần lấy.

        Returns
        -------
        image : torch.Tensor
            Tensor ảnh sau transform, shape [3, H, W].

        mask : torch.Tensor
            Tensor mask sau transform, shape [H, W].

        stem : str
            Tên file không kèm extension.

        img_path : str
            Đường dẫn ảnh gốc.

        mask_path : str
            Đường dẫn mask gốc.
        """
        img_path, mask_path, stem = self.samples[idx]

        # Luôn convert ảnh sang RGB để bảo đảm đúng số kênh.
        image = Image.open(img_path).convert("RGB")

        # Mask giữ nguyên giá trị label, không convert RGB.
        mask = Image.open(mask_path)

        image, mask = self.transform(image, mask)

        return image, mask, stem, img_path, mask_path