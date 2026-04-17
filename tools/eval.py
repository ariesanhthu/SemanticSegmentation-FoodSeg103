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
