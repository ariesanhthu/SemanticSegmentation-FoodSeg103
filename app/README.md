# FoodSeg103 Gradio App

Run the app:

```bash
python app/main.py --host 127.0.0.1 --port 7860
```

Features:

- choose `BiSeNet-RTB + GNN` or `BiSeNetV1 v4`
- select built-in test images from `datasets/foodseg103-full/test/img`
- upload a custom image and an optional ground-truth mask
- upload a video and generate an overlay video
- show `Input image`, `Ground truth`, and `Result overlay`
- return metrics and timing

Notes:

- Full segmentation metrics are only available when a ground-truth mask is provided.
- For videos, the app returns an overlay video, class statistics, and average processing speed.
