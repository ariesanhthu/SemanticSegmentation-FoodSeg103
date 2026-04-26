"""Loss functions for the BiSeNet-RTB training pipeline."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_presence_target(
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Build image-level multi-hot targets from segmentation masks."""
    b = masks.shape[0]
    target = torch.zeros((b, num_classes), device=masks.device, dtype=torch.float32)

    for i in range(b):
        valid = masks[i] != ignore_index
        labels = torch.unique(masks[i][valid])
        labels = labels[(labels >= 0) & (labels < num_classes)]
        target[i, labels.long()] = 1.0

    return target


def make_boundary_target(
    masks: torch.Tensor,
    ignore_index: int = 255,
    width: int = 3,
) -> torch.Tensor:
    """Convert semantic masks to binary boundary targets."""
    valid = masks != ignore_index

    m = masks.clone()
    m[~valid] = 0

    edge = torch.zeros_like(m, dtype=torch.float32)

    diff_h = (m[:, 1:, :] != m[:, :-1, :]).float()
    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], diff_h)
    edge[:, :-1, :] = torch.maximum(edge[:, :-1, :], diff_h)

    diff_w = (m[:, :, 1:] != m[:, :, :-1]).float()
    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], diff_w)
    edge[:, :, :-1] = torch.maximum(edge[:, :, :-1], diff_w)

    edge = edge * valid.float()

    if width > 1:
        edge = F.max_pool2d(
            edge.unsqueeze(1),
            kernel_size=width,
            stride=1,
            padding=width // 2,
        ).squeeze(1)

    return edge.unsqueeze(1)


class RTBLoss(nn.Module):
    """Composite RTB loss for segmentation, auxiliary, boundary, and presence heads."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        background_id: int = 103,
        aux_weight: float = 0.4,
        pre_weight: float = 0.4,
        edge_weight: float = 1.0,
        presence_weight: float = 0.15,
        boundary_width: int = 3,
        ignore_background_presence: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.background_id = background_id

        self.aux_weight = aux_weight
        self.pre_weight = pre_weight
        self.edge_weight = edge_weight
        self.presence_weight = presence_weight
        self.boundary_width = boundary_width
        self.ignore_background_presence = ignore_background_presence

        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _edge_loss(self, edge_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute BCE plus soft Dice loss for predicted boundaries."""
        target = make_boundary_target(
            masks,
            ignore_index=self.ignore_index,
            width=self.boundary_width,
        )

        pos = target.sum()
        neg = target.numel() - pos
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(max=20.0)

        bce = F.binary_cross_entropy_with_logits(
            edge_logits,
            target,
            pos_weight=pos_weight.detach(),
        )

        prob = torch.sigmoid(edge_logits)
        inter = (prob * target).sum(dim=(0, 2, 3))
        denom = (prob + target).sum(dim=(0, 2, 3)).clamp_min(1e-6)
        dice = 1.0 - (2.0 * inter / denom).mean()

        return bce + 0.5 * dice

    def _presence_loss(self, presence_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute image-level multi-label presence loss."""
        target = make_presence_target(
            masks,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        if self.ignore_background_presence and 0 <= self.background_id < self.num_classes:
            keep = torch.ones(self.num_classes, device=masks.device, dtype=torch.bool)
            keep[self.background_id] = False
            return F.binary_cross_entropy_with_logits(
                presence_logits[:, keep],
                target[:, keep],
            )

        return F.binary_cross_entropy_with_logits(presence_logits, target)

    def forward(self, outputs: torch.Tensor | Mapping[str, torch.Tensor], masks: torch.Tensor) -> torch.Tensor:
        """Return total loss for either plain logits or RTB output dictionaries."""
        if torch.is_tensor(outputs):
            return self.ce(outputs, masks)

        logits = outputs["logits"]
        loss = self.ce(logits, masks)

        if self.pre_weight > 0:
            loss = loss + self.pre_weight * self.ce(outputs["pre_logits"], masks)

        if self.aux_weight > 0:
            loss = loss + self.aux_weight * (
                self.ce(outputs["aux16"], masks)
                + self.ce(outputs["aux32"], masks)
            )

        if self.edge_weight > 0:
            loss = loss + self.edge_weight * self._edge_loss(outputs["edge_logits"], masks)

        if self.presence_weight > 0:
            loss = loss + self.presence_weight * self._presence_loss(
                outputs["presence_logits"],
                masks,
            )

        return loss
