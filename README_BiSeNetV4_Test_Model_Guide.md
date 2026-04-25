# Huong dan test model BiSeNet V4

Guide nay huong dan cach test checkpoint BiSeNet trong thu muc:

```text
work_dirs/bisenet_v4/
|-- bisenet_v4.pth
|-- last.pth
`-- metrics.csv
```

Luu y: trong code hien tai model van duoc khai bao la `BiSeNetV1` trong `models/bisenetv1.py`. Ten `bisenet_v4` o day la ten lan train/checkpoint, khong phai mot file model architecture rieng.

## 1. Chuan bi moi truong

Chay lenh tu thu muc root cua repo, noi co cac folder `configs`, `datasets`, `models`, `tools`.

Windows PowerShell:

```powershell
$env:DATA_ROOT="datasets/foodseg103-full"
$env:WORK_DIR="work_dirs/bisenet_v4"
python -m pip install -r requirements.txt
python -m pip install albumentations
```

Colab:

```bash
%cd /content/drive/MyDrive/your-repo-folder
%env DATA_ROOT=/content/data/foodseg103-full
%env WORK_DIR=/content/drive/MyDrive/checkpoints/bisenet_v4
!pip install -q -r requirements.txt
!pip install -q albumentations
```

Dataset can dung layout:

```text
DATA_ROOT/
|-- class_mapping.json
|-- train/
|   |-- img/
|   `-- mask/
`-- test/
    |-- img/
    `-- mask/
```

## 2. Kiem tra checkpoint

Checkpoint chinh nen test:

```text
work_dirs/bisenet_v4/bisenet_v4.pth
```

Thong tin checkpoint da kiem tra:

```text
epoch      = 99
global_iter = 400
best_miou  = 0.21574662625789642
num_classes = 104
model_name  = bisenetv1
```

Ket qua cuoi trong `metrics.csv`:

```text
epoch=99
val_loss=2.439805179834366
mIoU=0.21574662625789642
mAcc=0.35700753331184387
aAcc=0.5068405866622925
```

## 3. Test nhanh bang visualize

Cach nay de xem model predict ra mask co hop ly khong. Script se hien thi 3 cot:

- anh goc
- ground truth overlay
- prediction overlay

Windows PowerShell:

```powershell
python tools\visualization.py `
  --ckpt work_dirs\bisenet_v4\bisenet_v4.pth `
  --split test `
  --num-vis 4 `
  --batch-size 1 `
  --alpha 0.45
```

Colab:

```bash
!PYTHONPATH=. python tools/visualization.py \
  --ckpt "$WORK_DIR/bisenet_v4.pth" \
  --split test \
  --num-vis 4 \
  --batch-size 1 \
  --alpha 0.45
```

Neu bi CUDA OOM, giam `--batch-size 1` va giam `--num-vis`.

## 4. Eval metric tren test set

Script `tools/eval.py` hien tai mac dinh load:

```text
WORK_DIR/best_miou.pth
```

Trong thu muc `bisenet_v4` checkpoint lai co ten `bisenet_v4.pth`, nen can tao alias `best_miou.pth` truoc khi chay eval.

Windows PowerShell:

```powershell
$env:DATA_ROOT="datasets/foodseg103-full"
$env:WORK_DIR="work_dirs/bisenet_v4"

if (!(Test-Path "work_dirs\bisenet_v4\best_miou.pth")) {
  Copy-Item "work_dirs\bisenet_v4\bisenet_v4.pth" "work_dirs\bisenet_v4\best_miou.pth"
}

python tools\eval.py
```

Colab:

```bash
%env DATA_ROOT=/content/data/foodseg103-full
%env WORK_DIR=/content/drive/MyDrive/checkpoints/bisenet_v4

!test -f "$WORK_DIR/best_miou.pth" || cp "$WORK_DIR/bisenet_v4.pth" "$WORK_DIR/best_miou.pth"
!PYTHONPATH=. python tools/eval.py
```

Output mong doi la mot dict co cac key chinh:

```text
mIoU
mAcc
aAcc
loss
```

So lieu co the khac nhe so voi `metrics.csv` neu config preprocess/eval size da bi thay doi sau khi train.

## 5. Test qua Gradio app

Chay app:

```powershell
python app\main.py --host 127.0.0.1 --port 7860
```

Mo:

```text
http://127.0.0.1:7860
```

Trong app:

1. Chon model `BiSeNetV1 (FoodSeg103)`.
2. Neu default checkpoint khong tro dung `bisenet_v4`, dien checkpoint override:

```text
work_dirs/bisenet_v4/bisenet_v4.pth
```

3. Chon test image co san hoac upload anh rieng.
4. Neu upload them mask ground truth, app se tinh them metric.

## 6. Cach doc ket qua

Metric trong segmentation:

- `aAcc`: pixel accuracy tren toan bo pixel hop le.
- `mAcc`: trung binh accuracy theo class co xuat hien trong split eval.
- `mIoU`: mean IoU theo class co ground truth trong split eval.
- `loss`: loss trung binh tren eval loader.

Khi xem anh visualize:

- Neu prediction gan nhu toan background, model co the underfit hoac checkpoint/load sai.
- Neu prediction co vung segment dung vi tri nhung class sai, can xem lai class mapping va metric per-class.
- Neu anh/mask bi lech kich thuoc, kiem tra lai `DATA_ROOT`, file mask, va resize/eval transform.

## 7. Loi thuong gap

### `Checkpoint not found`

Kiem tra duong dan:

```powershell
Test-Path work_dirs\bisenet_v4\bisenet_v4.pth
```

Neu chay `tools/eval.py`, nho tao alias:

```powershell
Copy-Item work_dirs\bisenet_v4\bisenet_v4.pth work_dirs\bisenet_v4\best_miou.pth
```

### `ModuleNotFoundError`

Tren Colab, chay voi:

```bash
!PYTHONPATH=. python tools/visualization.py --ckpt "$WORK_DIR/bisenet_v4.pth"
```

Tren Windows, chay lenh tu root repo.

Neu loi thieu `albumentations`, cai them:

```powershell
python -m pip install albumentations
```

### `Dataset folder not found`

Set lai `DATA_ROOT` dung folder chua `train/` va `test/`:

```powershell
$env:DATA_ROOT="datasets/foodseg103-full"
```

### CUDA OOM

Dung batch nho:

```powershell
python tools\visualization.py --ckpt work_dirs\bisenet_v4\bisenet_v4.pth --batch-size 1 --num-vis 2
```

## 8. Checklist test model

- [ ] `DATA_ROOT` tro dung `datasets/foodseg103-full`.
- [ ] Checkpoint ton tai: `work_dirs/bisenet_v4/bisenet_v4.pth`.
- [ ] Visualize duoc it nhat 4 anh test.
- [ ] Eval metric chay xong va in ra `mIoU`, `mAcc`, `aAcc`, `loss`.
- [ ] App Gradio load duoc checkpoint qua override path.
