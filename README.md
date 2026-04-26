# Texture-Aware Lightweight Semantic Segmentation for Fine-Grained Food Images

Dự án nghiên cứu và triển khai phân đoạn hình ảnh thực phẩm (Food Segmentation)

## 📦 Dataset & Checkpoints

Tải checkpoints và datasets tại: [Google Drive](https://drive.google.com/drive/folders/1CHdDO-6Xh1uiY8Cu1E7-L_6Arwmb8Zpm?usp=sharing)

## 🚀 Quick Start (Chạy Demo)

Để chạy ứng dụng demo Gradio trên máy cục bộ:

```bash
# Cài đặt môi trường (nếu chưa có)
pip install -r requirements.txt

# Chạy ứng dụng
python app/main.py --host 127.0.0.1 --port 7860
```

Truy cập `http://127.0.0.1:7860` để bắt đầu trải nghiệm.

## 📂 Cấu trúc Dự án

Dưới đây là tóm tắt các thành phần chính:

- **`app/`**: Ứng dụng demo Gradio (UI & Backend).
- **`configs/`**: Cấu hình mô hình và tham số huấn luyện.
- **`models/`**: Triển khai kiến trúc BiSeNetV1, CCNet.
- **`tools/`**: Tập lệnh Train, Eval và Phân tích lỗi.
- **`utils/`**: Các hàm tiện ích và metrics.

Để biết thêm chi tiết về cấu trúc và cách đóng góp, vui lòng đọc [**Hướng dẫn Chi tiết (Detailed Guide)**](docs/guide-detail.md).

## 🧪 Cấu hình Train BiSeNet-RTB-GNN (x2 H100)

Profile được khuyến nghị:

- GPU: 2 x H100
- Epochs: 80
- Batch size: 16
- Crop size: 768 x 768
- Thời gian mục tiêu: khoảng 5 giờ (phụ thuộc tốc độ I/O, dataloader và tần suất eval)

Lưu ý:

- Pipeline RTB hiện dùng crop train mặc định 768 x 768 trong `configs/bisenet_rtb_foodseg103.py`.
- Nếu chạy đa GPU, hãy truyền rõ `--num-gpus 2` khi train.

Ví dụ chạy trên Windows PowerShell:

```powershell
$env:EPOCHS="80"
$env:BATCH_SIZE="32"
$env:NUM_WORKERS="8"
python tools/train_rtb.py --num-gpus 2 --epochs 80 --batch-size 32
```

Ví dụ chạy trên Linux/macOS:

```bash
EPOCHS=80 BATCH_SIZE=32 NUM_WORKERS=8 \
python tools/train_rtb.py --num-gpus 2 --epochs 80 --batch-size 32
```

## 🎓 Thông tin Sinh viên

- **Nguyễn Anh Thư** - 23127266
- **Nguyễn Thiên Nhã Trân** - 23127272