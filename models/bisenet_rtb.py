"""BiSeNet-RTB model with texture, boundary, and class-relation reasoning.

This module intentionally lives next to the existing BiSeNetV1 implementation
instead of modifying it.  It reuses the stable BiSeNetV1 building blocks and
adds RTB-specific branches that return a dictionary during training.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bisenetv1 import (
    ContextPath,
    ConvBNReLU,
    FeatureFusionModule,
    SegHead,
    SpatialPath,
)


class TextureBranch(nn.Module):
    """Extract multi-scale local texture cues from the spatial path feature."""

    def __init__(self, in_ch: int = 128, tex_ch: int = 64) -> None:
        super().__init__()

        self.dw3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.dw_dilated = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(in_ch * 4, tex_ch, 1, bias=False),
            nn.BatchNorm2d(tex_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(tex_ch, tex_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(tex_ch),
            nn.ReLU(inplace=True),
        )

    def _local_contrast(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Return normalized local contrast for each channel."""
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)
        mean_sq = F.avg_pool2d(x * x, kernel_size, stride=1, padding=kernel_size // 2)
        var = (mean_sq - mean * mean).clamp_min(1e-6)
        return torch.abs(x - mean) / (torch.sqrt(var) + 1e-6)

    def forward(self, feat_sp: torch.Tensor) -> torch.Tensor:
        """Build texture features from spatial-path activations."""
        t3 = self.dw3(feat_sp)
        t5 = self.dw5(feat_sp)
        td = self.dw_dilated(feat_sp)
        contrast = self._local_contrast(feat_sp)
        return self.project(torch.cat([t3, t5, td, contrast], dim=1))


class PresenceHead(nn.Module):
    """Predict image-level class presence from fused segmentation features."""

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        hidden = max(128, in_ch // 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Return class-presence logits with shape ``[B, K]``."""
        return self.head(feat)


class BoundaryHead(nn.Module):
    """Predict a binary semantic boundary map from spatial features."""

    def __init__(self, in_ch: int = 128, mid_ch: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, 3, 1, 1),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
        )

    def forward(self, feat: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        """Return boundary logits resized to the input image resolution."""
        edge = self.head(feat)
        return F.interpolate(edge, size=out_size, mode="bilinear", align_corners=False)


class PriorGATLayer(nn.Module):
    """Single graph-attention layer gated by prior adjacency and class presence."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        prior: torch.Tensor,
        presence_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Update class nodes using prior-biased graph attention."""
        _, k, d = h.shape

        q = self.q(h)
        key = self.k(h)
        val = self.v(h)

        score = torch.matmul(q, key.transpose(1, 2)) / math.sqrt(d)

        pair_gate = presence_prob.unsqueeze(2) * presence_prob.unsqueeze(1)
        adj = prior.unsqueeze(0).to(h.device, h.dtype)
        adj = adj * (1.0 + 0.5 * pair_gate)

        eye = torch.eye(k, device=h.device, dtype=h.dtype).unsqueeze(0)
        adj = adj + eye
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        score = score + torch.log(adj.clamp_min(1e-6))
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)

        msg = torch.matmul(attn, val)
        out = self.out(msg)

        return self.norm(h + out)


class ClassGraphReasoner(nn.Module):
    """Refine dense logits with class prototypes and a graph prior."""

    def __init__(
        self,
        num_classes: int,
        feat_ch: int = 256,
        tex_ch: int = 64,
        graph_dim: int = 128,
        num_layers: int = 2,
        eta: float = 0.5,
        xi: float = 0.15,
        background_id: int = 103,
        graph_prior: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.graph_dim = graph_dim
        self.eta = eta
        self.xi = xi
        self.background_id = background_id

        if graph_prior is None:
            graph_prior = torch.eye(num_classes, dtype=torch.float32)
        else:
            graph_prior = graph_prior.float()

        if graph_prior.shape != (num_classes, num_classes):
            raise ValueError(
                f"graph_prior must have shape {(num_classes, num_classes)}, "
                f"got {tuple(graph_prior.shape)}"
            )

        graph_prior = graph_prior.clone()

        if 0 <= background_id < num_classes:
            graph_prior[background_id, :] = 0.0
            graph_prior[:, background_id] = 0.0
            graph_prior[background_id, background_id] = 1.0

        graph_prior = graph_prior / graph_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("graph_prior", graph_prior)

        self.class_embed = nn.Parameter(torch.randn(num_classes, graph_dim) * 0.02)

        self.feat_proj = nn.Linear(feat_ch, graph_dim)
        self.tex_proj = nn.Linear(tex_ch, graph_dim)
        self.scalar_proj = nn.Linear(2, graph_dim)
        self.node_norm = nn.LayerNorm(graph_dim)

        self.layers = nn.ModuleList(
            [PriorGATLayer(graph_dim, dropout=0.1) for _ in range(num_layers)]
        )

        self.bias_head = nn.Linear(graph_dim, 1)
        self.rel_embed = nn.Linear(graph_dim, graph_dim)
        self.pixel_proj = nn.Conv2d(feat_ch, graph_dim, 1, bias=False)

    def _soft_prototypes(self, feat: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes by soft pooling feature maps with probabilities."""
        _, _, hf, wf = feat.shape
        prob_small = F.interpolate(prob, size=(hf, wf), mode="bilinear", align_corners=False)
        weight = prob_small / prob_small.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        feat_flat = feat.flatten(2).transpose(1, 2)
        weight_flat = weight.flatten(2)

        return torch.matmul(weight_flat, feat_flat)

    def forward(
        self,
        feat_fuse: torch.Tensor,
        tex_feat: torch.Tensor,
        pre_logits: torch.Tensor,
        presence_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return graph-refined segmentation logits."""
        prob = F.softmax(pre_logits, dim=1)
        presence_prob = torch.sigmoid(presence_logits)

        feat_proto = self._soft_prototypes(feat_fuse, prob)
        tex_proto = self._soft_prototypes(tex_feat, prob)

        area = prob.mean(dim=(2, 3))
        scalar = torch.stack([area, presence_prob], dim=-1)

        node = (
            self.feat_proj(feat_proto)
            + self.tex_proj(tex_proto)
            + self.scalar_proj(scalar)
            + self.class_embed.unsqueeze(0)
        )
        node = self.node_norm(node)

        for layer in self.layers:
            node = layer(node, self.graph_prior, presence_prob)

        class_bias = self.bias_head(node).squeeze(-1)
        class_rel = self.rel_embed(node)

        pixel = self.pixel_proj(feat_fuse)
        affinity = torch.einsum("bdhw,bkd->bkhw", pixel, class_rel)
        affinity = F.interpolate(
            affinity,
            size=pre_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return pre_logits + self.eta * class_bias[:, :, None, None] + self.xi * affinity


class BiSeNetRTB(nn.Module):
    """BiSeNetV1-compatible segmentation model with RTB refinement heads."""

    def __init__(
        self,
        num_classes: int = 104,
        backbone: Optional[nn.Module] = None,
        background_id: int = 103,
        graph_prior: Optional[torch.Tensor] = None,
        tex_ch: int = 64,
        graph_dim: int = 128,
        graph_layers: int = 2,
        graph_eta: float = 0.5,
        graph_xi: float = 0.15,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.background_id = background_id

        self.spatial_path = SpatialPath()
        self.context_path = ContextPath(backbone)

        self.texture_branch = TextureBranch(128, tex_ch)

        self.texture_to_spatial = nn.Sequential(
            nn.Conv2d(128 + tex_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ffm = FeatureFusionModule(256, 256)

        self.pre_head = SegHead(256, 256, num_classes)
        self.aux_head16 = SegHead(128, 64, num_classes)
        self.aux_head32 = SegHead(128, 64, num_classes)

        self.presence_head = PresenceHead(256, num_classes)
        self.boundary_head = BoundaryHead(128, 64)

        self.graph_reasoner = ClassGraphReasoner(
            num_classes=num_classes,
            feat_ch=256,
            tex_ch=tex_ch,
            graph_dim=graph_dim,
            num_layers=graph_layers,
            eta=graph_eta,
            xi=graph_xi,
            background_id=background_id,
            graph_prior=graph_prior,
        )

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """Run RTB segmentation.

        During training, return a dictionary consumed by ``RTBLoss``.  During
        evaluation, return only dense logits unless ``return_dict=True``.
        """
        out_size = x.shape[-2:]

        feat_sp = self.spatial_path(x)
        feat_cp8, feat_cp16, feat_cp32 = self.context_path(x)

        tex = self.texture_branch(feat_sp)
        feat_sp_tex = feat_sp + self.texture_to_spatial(torch.cat([feat_sp, tex], dim=1))

        feat_fuse = self.ffm(feat_sp_tex, feat_cp8)

        pre_logits = self.pre_head(feat_fuse, out_size)
        presence_logits = self.presence_head(feat_fuse)

        logits = self.graph_reasoner(
            feat_fuse=feat_fuse,
            tex_feat=tex,
            pre_logits=pre_logits,
            presence_logits=presence_logits,
        )

        aux16 = self.aux_head16(feat_cp16, out_size)
        aux32 = self.aux_head32(feat_cp32, out_size)
        edge_logits = self.boundary_head(feat_sp_tex, out_size)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "pre_logits": pre_logits,
            "aux16": aux16,
            "aux32": aux32,
            "edge_logits": edge_logits,
            "presence_logits": presence_logits,
        }

        if self.training or return_dict:
            return outputs

        return logits
