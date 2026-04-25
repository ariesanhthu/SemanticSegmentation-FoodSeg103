# Hướng dẫn Chi tiết về Dự án

Tài liệu này cung cấp cái nhìn sâu sắc về cấu trúc thư mục, các tệp cốt lõi và hướng dẫn cách mở rộng hoặc cải tiến dự án.

## 1. Cấu trúc Thư mục

Dự án được tổ chức theo mô-đun để tách biệt giữa kiến trúc mô hình, quy trình huấn luyện và ứng dụng demo.

| Thư mục | Mô tả |
| :--- | :--- |
| `app/` | Chứa mã nguồn cho ứng dụng Gradio demo (Giao diện người dùng và logic backend). |
| `configs/` | Lưu trữ các tệp cấu hình (hyperparameters, đường dẫn tệp) cho từng mô hình. |
| `models/` | Triển khai kiến trúc các mạng nơ-ron (BiSeNetV1, CCNet, backbones). |
| `tools/` | Các tập lệnh thực thi để huấn luyện (`train.py`), đánh giá (`eval.py`) và chẩn đoán. |
| `utils/` | Các tiện ích chung như tính toán metrics, xử lý tệp và các hàm hỗ trợ khác. |
| `datasets/` | (Mặc định) Thư mục chứa dữ liệu FoodSeg103 (không được upload lên repo). |
| `work_dirs/` | Nơi lưu trữ checkpoints, logs và kết quả huấn luyện. |
| `docs/` | Tài liệu hướng dẫn và báo cáo chi tiết. |

---

## 2. Các Tệp Cốt lõi

### Ứng dụng Demo (`app/`)
- **`app/main.py`**: Khởi tạo giao diện Gradio, thiết lập các block UI (Hình ảnh, Video) và định nghĩa các hàm xử lý sự kiện.
- **`app/service.py`**: Tầng logic backend. Quản lý việc load model, cache checkpoints, thực hiện inference và xử lý hậu kỳ (overlay mask, tính metrics).

### Kiến trúc Mô hình (`models/`)
- **`models/bisenetv1.py`**: Triển khai BiSeNetV1 với Spatial Path và Context Path.
- **`models/ccnet.py`**: Triển khai Criss-Cross Network với module Recurrent Criss-Cross Attention.
- **`models/builder.py`**: Hàm factory để khởi tạo mô hình dựa trên cấu hình.

### Công cụ Huấn luyện & Đánh giá (`tools/`)
- **`tools/train.py`**: Quy trình huấn luyện chính (loop, optimizer, scheduler).
- **`tools/eval.py`**: Đánh giá hiệu năng mô hình trên tập test.
- **`tools/eval_bisenet_diagnostics.py`**: Công cụ chuyên sâu để phân tích các lỗi phổ biến (như collapse prediction).

---

## 3. Cách Cải tiến và Code thêm

### Thêm một Mô hình mới
Để tích hợp một kiến trúc mới vào dự án, bạn nên làm theo các bước sau:
1. **Kiến trúc**: Tạo tệp mới trong `models/` (ví dụ: `models/mynet.py`).
2. **Cấu hình**: Tạo tệp config trong `configs/` để định nghĩa số lớp, input size, v.v.
3. **Builder**: Cập nhật `models/builder.py` để nhận diện mô hình mới.
4. **Service**: Nếu muốn sử dụng trong demo, cập nhật `FoodSegDemoService._build_presets` trong `app/service.py`.

### Thêm Công cụ mới
Nếu bạn muốn tạo một script phân tích dữ liệu hoặc xử lý ảnh mới:
- Đặt tệp vào thư mục `tools/`.
- Sử dụng `argparse` để quản lý các tham số dòng lệnh.
- Tận dụng các hàm trong `utils/` và `configs/` để giữ code ngắn gọn và nhất quán.

### Cải tiến Giao diện Demo
- Chỉnh sửa `app/main.py` để thêm các component Gradio mới.
- Thêm các hàm xử lý tương ứng vào `FoodSegDemoService` trong `app/service.py`.
- Sử dụng `custom_css` trong `app/main.py` để tinh chỉnh thẩm mỹ.

---

## 4. Quy trình làm việc (Workflow)

1. **Chuẩn bị Dữ liệu**: Đảm bảo FoodSeg103 được đặt đúng cấu trúc trong `datasets/`.
2. **Huấn luyện**:
   ```bash
   python tools/train.py --config configs/bisenet_foodseg103.py
   ```
3. **Đánh giá**:
   ```bash
   python tools/eval.py --config configs/bisenet_foodseg103.py --ckpt work_dirs/bisenet/model_final.pth
   ```
4. **Chạy Demo**:
   ```bash
   python app/main.py
   ```

### Hướng dẫn Nâng cao: Chẩn đoán lỗi **Collapse Prediction** (Dự đoán ồ ạt)

Một trong những vấn đề nghiêm trọng nhất trong các mô hình phân đoạn thực phẩm là hiện tượng **Collapse Prediction**, khi mô hình chỉ dự đoán một hoặc hai class duy nhất cho toàn bộ ảnh (thường là "Background"), dẫn đến điểm số mIoU rất cao nhưng thực tế hoàn toàn vô dụng.

Để phân tích chuyên sâu hiện tượng này, chúng tôi đã phát triển công cụ `eval_bisenet_diagnostics.py`. Công cụ này không chỉ tính metrics thông thường mà còn phân tích từng pixel để tìm ra nguyên nhân gốc rễ của lỗi.

#### 1. Các loại lỗi được phát hiện

Script sẽ phân loại các pixel không chính xác thành 5 nhóm rõ ràng:

| Màu sắc | Ý nghĩa | Mức độ nghiêm trọng |
| :--- | :--- | :--- |
| **Đen** | **TN (True Negative)**: Đúng là Background và mô hình dự đoán đúng Background. | Tốt |
| **Đỏ** | **FP (False Positive)**: Dự đoán sai là Background (thực tế không phải Background). | Nghiêm trọng |
| **Lục lam** | **FN (False Negative)**: Không dự đoán được Background (thực tế là Background). | Nghiêm trọng |
| **Xanh lá cây** | **FP/FN (Class-level)**: Pixel thuộc về một Food Class, nhưng bị nhầm sang một Food Class khác. | Trung bình |
| **Vàng** | **Over-predicted Sink**: Pixel thuộc về một Food Class, nhưng bị ép xuống thành Background. | Rất nghiêm trọng |

#### 2. Cách chạy công cụ chẩn đoán

Bạn có thể chạy script để phân tích tập Train, Val hoặc Test. Dưới đây là ví dụ phân tích tập **Train**:

```bash
python tools/eval_bisenet_diagnostics.py \
    --config configs/bisenet_foodseg103.py \
    --split train \
    --max-items 2000 \
    --output work_dirs/bisenet/diagnostics_train
```

- `--split train`: Phân tích tập huấn luyện (để xem mô hình "học" sai ở đâu).
- `--max-items`: Giới hạn số ảnh xử lý (để chạy nhanh).
- `--output`: Thư mục lưu kết quả.

#### 3. Phân tích kết quả đầu ra

Script sẽ tạo 2 loại file chính trong thư mục output:

1.  **`error_map_legend.png`**: Giải thích màu sắc của các loại lỗi (như bảng trên).
2.  **`diagnostics.txt`**: Báo cáo thống kê chi tiết.
3.  **`FAILURE_GROUPS/`**: Folder chứa ảnh minh họa cho từng loại lỗi.

**Cách đọc `diagnostics.txt`**: 
- **Class-Specific False Positives**:
    - **Over-predicted sink**: Cho biết class nào bị mô hình "ghét" nhất, luôn bị ép xuống Background. (Ví dụ: "Shrimp: 787 pixels... Over-predicted sink: 787 pixels". Nếu con số này lớn, mô hình đang gặp vấn đề về khả năng nhận diện object cụ thể đó).
- **Never Predicted Classes**: Liệt kê các class chưa từng xuất hiện trong dự đoán của mô hình.

**Cách đọc `FAILURE_GROUPS/`**:
- Mở các file `GROUP_FP_FP_*.png`. Nếu bạn thấy ảnh chỉ có màu đen (Background) và màu xanh lá (Food), điều đó chứng tỏ mô hình đang gặp lỗi **Confusion** (nhầm lẫn class), không phải lỗi **Collapse**.
- Nếu bạn thấy ảnh chỉ có màu đen và màu vàng, đó là lỗi **Collapse** (Foreground bị ép thành Background).
