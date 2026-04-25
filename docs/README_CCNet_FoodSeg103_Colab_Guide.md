# CCNet + FoodSeg103 Colab Guide

Mục tiêu của guide này là chỉ ra cách gọi đúng để train **CCNet ResNet-50** trên **FoodSeg103** trong Colab với code hiện tại của repo.

Guide này áp dụng cho các file:

- [configs/ccnet_foodseg103.py](<F:\ANHTHU\1-HCMUS\1 - STUDY\HKVIII\CV\PROJECT\WORKING\source\0-src\configs\ccnet_foodseg103.py>)
- [datasets/foodseg103_ccnet.py](<F:\ANHTHU\1-HCMUS\1 - STUDY\HKVIII\CV\PROJECT\WORKING\source\0-src\datasets\foodseg103_ccnet.py>)
- [models/ccnet.py](<F:\ANHTHU\1-HCMUS\1 - STUDY\HKVIII\CV\PROJECT\WORKING\source\0-src\models\ccnet.py>)
- [tools/train_ccnet.py](<F:\ANHTHU\1-HCMUS\1 - STUDY\HKVIII\CV\PROJECT\WORKING\source\0-src\tools\train_ccnet.py>)
- [tools/eval_ccnet.py](<F:\ANHTHU\1-HCMUS\1 - STUDY\HKVIII\CV\PROJECT\WORKING\source\0-src\tools\eval_ccnet.py>)

## 1. Cấu trúc dữ liệu kỳ vọng

Code CCNet hiện tại mặc định đọc dataset theo layout:

```text
/content/data/foodseg103-full/
├─ class_mapping.json
├─ train/
│  ├─ img/
│  └─ mask/
└─ test/
   ├─ img/
   └─ mask/
```

Các điểm quan trọng:

- `class_mapping.json` nên nằm ngay dưới `DATA_ROOT`
- train dùng `train/img` và `train/mask`
- eval dùng `test/img` và `test/mask`
- code tự đọc:
  - `num_ingredient_classes = 103`
  - `background_id = 103`
  - `num_classes = 104`

## 2. Mount Google Drive

Trong Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. Đi tới repo

Ví dụ nếu repo đang ở Drive:

```bash
%cd /content/drive/MyDrive/your-repo-folder
```

Kiểm tra nhanh:

```bash
!ls
```

Bạn cần nhìn thấy các folder như:

- `configs`
- `datasets`
- `models`
- `tools`

## 4. Thiết lập biến môi trường

Ví dụ đúng như cách gọi bạn muốn:

```bash
# 2. Thiết lập biến môi trường cho phiên làm việc này
%env DATA_ROOT=/content/data/foodseg103-full
%env WORK_DIR=/content/drive/MyDrive/checkpoints/ccnet[all-change]
```

Giải thích:

- `DATA_ROOT` trỏ tới thư mục dataset FoodSeg103
- `WORK_DIR` là nơi lưu:
  - `last.pth`
  - `best_miou.pth`
  - `config.json`

## 5. Cài dependency cơ bản

Nếu Colab chưa có sẵn PyTorch hoặc torchvision phù hợp:

```bash
!pip install -q torch torchvision
```

Nếu repo dùng thêm package khác thì cài tiếp từ `requirements.txt`:

```bash
!pip install -q -r requirements.txt
```

## 6. Train CCNet trên Colab

### 6.1. Train nhanh để debug

Lệnh này đúng với script hiện tại:

```bash
!PYTHONPATH=. python tools/train_ccnet.py --overfit 8 --epochs 100 --batch-size 2
```

Ý nghĩa:

- `PYTHONPATH=.` để Python import đúng module trong repo
- `--overfit 8` chỉ lấy 8 sample đầu để debug
- `--epochs 100` sẽ được quy đổi nội bộ thành:
  - `max_iters = epochs * len(train_loader)`
- `--batch-size 2` dùng batch nhỏ để đỡ OOM trên Colab

### 6.2. Train bình thường

```bash
!PYTHONPATH=. python tools/train_ccnet.py
```

Script sẽ lấy config từ [configs/ccnet_foodseg103.py](<F:\ANHTHU\1-HCMUS\1 - STUDY\HKVIII\CV\PROJECT\WORKING\source\0-src\configs\ccnet_foodseg103.py>).

### 6.3. Override trực tiếp khi train

Ví dụ:

```bash
!PYTHONPATH=. python tools/train_ccnet.py \
  --data-root /content/data/foodseg103-full \
  --work-dir /content/drive/MyDrive/checkpoints/ccnet_run_01 \
  --batch-size 4 \
  --epochs 40
```

Nếu bạn muốn train theo số iteration cố định thay vì `epochs`:

```bash
!PYTHONPATH=. python tools/train_ccnet.py --max-iters 80000
```

Lưu ý:

- nếu truyền cả `--max-iters` và `--epochs`, script ưu tiên `--max-iters`

## 7. Eval checkpoint

### 7.1. Eval từ script train

```bash
!PYTHONPATH=. python tools/train_ccnet.py --eval-only
```

### 7.2. Eval bằng script riêng

```bash
!PYTHONPATH=. python tools/eval_ccnet.py
```

Hoặc chỉ định checkpoint cụ thể:

```bash
!PYTHONPATH=. python tools/eval_ccnet.py \
  --checkpoint /content/drive/MyDrive/checkpoints/ccnet[all-change]/best_miou.pth
```

## 8. Các file output sẽ được lưu ở đâu

Trong `WORK_DIR`, script sẽ lưu:

- `last.pth`
- `best_miou.pth`
- `config.json`

Ví dụ:

```bash
!ls /content/drive/MyDrive/checkpoints/ccnet[all-change]
```

## 9. Kiểm tra nhanh trước khi train thật

Bạn nên chạy theo thứ tự:

1. Mount Drive.
2. `cd` vào repo.
3. Set `DATA_ROOT` và `WORK_DIR`.
4. Chạy overfit nhỏ:

```bash
!PYTHONPATH=. python tools/train_ccnet.py --overfit 8 --epochs 100 --batch-size 2
```

Nếu lệnh này chạy qua được, mới tăng lên train thật.

## 10. Một cell Colab tối thiểu

Bạn có thể copy nguyên cell này:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/drive/MyDrive/your-repo-folder

%env DATA_ROOT=/content/data/foodseg103-full
%env WORK_DIR=/content/drive/MyDrive/checkpoints/ccnet[all-change]

!PYTHONPATH=. python tools/train_ccnet.py --overfit 8 --epochs 100 --batch-size 2
```

## 11. Một số lỗi thường gặp

### `ModuleNotFoundError`

Nguyên nhân thường là chưa set `PYTHONPATH=.`

Cách gọi đúng:

```bash
!PYTHONPATH=. python tools/train_ccnet.py ...
```

### `Train loader is empty`

Nguyên nhân:

- sai `DATA_ROOT`
- thiếu `train/img`
- thiếu `train/mask`
- tên file ảnh và mask không match stem

### Không thấy checkpoint lưu ra Drive

Kiểm tra:

- `WORK_DIR` đã set đúng chưa
- Drive đã mount chưa
- Colab có quyền ghi vào folder đó chưa

### OOM CUDA

Giảm:

- `--batch-size`

Ví dụ:

```bash
!PYTHONPATH=. python tools/train_ccnet.py --batch-size 1
```

## 12. Gợi ý lệnh dùng thực tế

Debug nhanh:

```bash
!PYTHONPATH=. python tools/train_ccnet.py --overfit 8 --epochs 100 --batch-size 2
```

Train đầy đủ:

```bash
!PYTHONPATH=. python tools/train_ccnet.py --max-iters 80000 --batch-size 8
```

Eval:

```bash
!PYTHONPATH=. python tools/eval_ccnet.py
```
