import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def build_attention_mask(batch_size, height, width, device, dtype):
    diagonal = torch.full((height,), float("inf"), device=device, dtype=dtype)
    return -torch.diag(diagonal).unsqueeze(0).repeat(batch_size * width, 1, 1)


class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.shape

        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query_h = (
            query.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * width, -1, height)
            .permute(0, 2, 1)
        )
        key_h = (
            key.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * width, -1, height)
        )
        value_h = (
            value.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * width, -1, height)
        )

        energy_h = torch.bmm(query_h, key_h)
        energy_h = energy_h + build_attention_mask(
            batch_size=batch_size,
            height=height,
            width=width,
            device=x.device,
            dtype=energy_h.dtype,
        )
        energy_h = energy_h.view(batch_size, width, height, height).permute(0, 2, 1, 3)

        query_w = (
            query.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * height, -1, width)
            .permute(0, 2, 1)
        )
        key_w = (
            key.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * height, -1, width)
        )
        value_w = (
            value.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * height, -1, width)
        )

        energy_w = torch.bmm(query_w, key_w)
        energy_w = energy_w.view(batch_size, height, width, width)

        attention = self.softmax(torch.cat([energy_h, energy_w], dim=3))

        attention_h = (
            attention[:, :, :, :height]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * width, height, height)
        )
        attention_w = (
            attention[:, :, :, height : height + width]
            .contiguous()
            .view(batch_size * height, width, width)
        )

        out_h = torch.bmm(value_h, attention_h.permute(0, 2, 1))
        out_h = out_h.view(batch_size, width, -1, height).permute(0, 2, 3, 1)

        out_w = torch.bmm(value_w, attention_w.permute(0, 2, 1))
        out_w = out_w.view(batch_size, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_h + out_w) + x


class CCHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, recurrence=2, dropout=0.1):
        super().__init__()
        self.recurrence = recurrence
        self.conva = ConvBNReLU(in_channels, channels, kernel_size=3, padding=1)
        self.cca = CrissCrossAttention(channels)
        self.convb = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        # Baseline-aligned CCNet head: concat local feature x with RCCA feature.
        self.bottleneck = ConvBNReLU(
            in_channels + channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.conva(x)
        for _ in range(self.recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = torch.cat([x, output], dim=1)
        output = self.bottleneck(output)
        output = self.dropout(output)
        return self.classifier(output)


class FCNAuxHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, dropout=0.1):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return self.classifier(x)


class ResNetDilatedBackbone(nn.Module):
    def __init__(self, pretrained=True, output_stride=8):
        super().__init__()
        if output_stride != 8:
            raise ValueError(f"Unsupported output_stride={output_stride}; expected 8.")

        weights = None
        if pretrained:
            weights = getattr(ResNet50_Weights, "IMAGENET1K_V2", ResNet50_Weights.DEFAULT)

        base = models.resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, True, True],
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.out_channels = 2048
        self.aux_channels = 1024

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        aux = self.layer3(x)
        out = self.layer4(aux)
        return out, aux


class CCNetSeg(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_pretrained=True,
        output_stride=8,
        channels=512,
        recurrence=2,
        use_aux=True,
        dropout=0.1,
        align_corners=False,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.use_aux = use_aux

        self.backbone = ResNetDilatedBackbone(
            pretrained=backbone_pretrained,
            output_stride=output_stride,
        )
        self.decode_head = CCHead(
            in_channels=self.backbone.out_channels,
            channels=channels,
            num_classes=num_classes,
            recurrence=recurrence,
            dropout=dropout,
        )

        if self.use_aux:
            self.aux_head = FCNAuxHead(
                in_channels=self.backbone.aux_channels,
                channels=256,
                num_classes=num_classes,
                dropout=dropout,
            )

    def forward(self, x):
        out_size = x.shape[-2:]
        feature, aux_feature = self.backbone(x)

        logits = self.decode_head(feature)
        logits = F.interpolate(
            logits,
            size=out_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        if self.training and self.use_aux:
            aux_logits = self.aux_head(aux_feature)
            aux_logits = F.interpolate(
                aux_logits,
                size=out_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            return logits, aux_logits

        return logits
