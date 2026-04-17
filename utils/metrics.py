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
