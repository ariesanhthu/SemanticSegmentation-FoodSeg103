import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class XBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

        layers = []
        filters = in_ch
        if grow_first:
            layers.append(self.relu if start_with_relu else nn.Identity())
            layers.append(SeparableConv2d(in_ch, out_ch, 3, 1, 1, False))
            filters = out_ch

        for _ in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(filters, filters, 3, 1, 1, False))

        if not grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_ch, out_ch, 3, 1, 1, False))

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))

        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        return self.rep(x) + self.skip(x)


class Xception39(nn.Module):
    """
    Minimal Xception39-like backbone for BiSeNetV1.
    Outputs three feature maps at /8, /16, /32 scale.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = XBlock(16, 32, reps=3, stride=2, start_with_relu=False, grow_first=True)
        self.block2 = XBlock(32, 64, reps=3, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = XBlock(64, 128, reps=3, stride=2, start_with_relu=True, grow_first=True)
        self.out_channels = (32, 64, 128)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        feat8 = self.block1(x)
        feat16 = self.block2(feat8)
        feat32 = self.block3(feat16)
        return feat8, feat16, feat32


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        self.attn_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.attn_bn = nn.BatchNorm2d(out_ch)
        self.attn_sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        attn = F.adaptive_avg_pool2d(feat, output_size=1)
        attn = self.attn_conv(attn)
        attn = self.attn_bn(attn)
        attn = self.attn_sigmoid(attn)
        return feat * attn


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, 7, 2, 3)
        self.conv2 = ConvBNReLU(64, 64, 3, 2, 1)
        self.conv3 = ConvBNReLU(64, 64, 3, 2, 1)
        self.conv_out = ConvBNReLU(64, 128, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class ContextPath(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = Xception39() if backbone is None else backbone
        _, c16, c32 = self.backbone.out_channels

        self.arm16 = AttentionRefinementModule(c16, 128)
        self.arm32 = AttentionRefinementModule(c32, 128)
        self.global_context = ConvBNReLU(c32, 128, 1, 1, 0)
        self.refine16 = ConvBNReLU(128, 128, 3, 1, 1)
        self.refine32 = ConvBNReLU(128, 128, 3, 1, 1)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)

        tail = F.adaptive_avg_pool2d(feat32, output_size=1)
        tail = self.global_context(tail)
        tail = F.interpolate(tail, size=feat32.shape[-2:], mode="bilinear", align_corners=False)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + tail
        feat32_up = F.interpolate(self.refine32(feat32_sum), size=feat16.shape[-2:], mode="bilinear", align_corners=False)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(self.refine16(feat16_sum), size=feat8.shape[-2:], mode="bilinear", align_corners=False)

        return feat16_up, feat16_sum, feat32_sum


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblk = ConvBNReLU(in_ch, out_ch, 1, 1, 0)
        self.attn1 = nn.Conv2d(out_ch, out_ch // 4, kernel_size=1, bias=False)
        self.attn_relu = nn.ReLU(inplace=True)
        self.attn2 = nn.Conv2d(out_ch // 4, out_ch, kernel_size=1, bias=False)
        self.attn_sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(feat)
        attn = F.adaptive_avg_pool2d(feat, output_size=1)
        attn = self.attn1(attn)
        attn = self.attn_relu(attn)
        attn = self.attn2(attn)
        attn = self.attn_sigmoid(attn)
        return feat + feat * attn


class SegHead(nn.Module):
    def __init__(self, in_ch, mid_ch, num_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, mid_ch, 3, 1, 1)
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(mid_ch, num_classes, kernel_size=1)

    def forward(self, x, out_size=None):
        x = self.conv(x)
        x = self.drop(x)
        x = self.cls(x)
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class BiSeNetV1(nn.Module):
    def __init__(self, num_classes=19, backbone=None):
        super().__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath(backbone)
        self.ffm = FeatureFusionModule(256, 256)
        self.head = SegHead(256, 256, num_classes)
        self.aux_head16 = SegHead(128, 64, num_classes)
        self.aux_head32 = SegHead(128, 64, num_classes)

    def forward(self, x):
        out_size = x.shape[-2:]
        feat_sp = self.spatial_path(x)
        feat_cp8, feat_cp16, feat_cp32 = self.context_path(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        logits = self.head(feat_fuse, out_size)
        aux16 = self.aux_head16(feat_cp16, out_size)
        aux32 = self.aux_head32(feat_cp32, out_size)

        if self.training:
            return logits, aux16, aux32
        return logits
