# Texture-Aware Lightweight Semantic Segmentation for Fine-Grained Food Images

Dự án nghiên cứu và triển khai phân đoạn hình ảnh thực phẩm (Food Segmentation) sử dụng các kiến trúc mạng nơ-ron hiệu quả như **BiSeNetV1** và **CCNet**.

## 📦 Dataset & Checkpoints

Tải tại: [Google Drive](https://drive.google.com/drive/folders/1CHdDO-6Xh1uiY8Cu1E7-L_6Arwmb8Zpm?usp=sharing)


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

## 🎓 Thông tin Sinh viên

- **Nguyễn Anh Thư** - 23127266
- **Nguyễn Thiên Nhã Trân** - 23127272

---
*Dự án cho môn học Computer Vision - HCMUS.*
