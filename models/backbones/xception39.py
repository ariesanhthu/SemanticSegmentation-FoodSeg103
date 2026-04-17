from typing import Optional, Tuple
import math
import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    Depthwise separable convolution:
    - depthwise conv
    - pointwise conv

    Đây là thành phần cốt lõi của Xception.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class Block(nn.Module):
    """
    Xception-style residual block.

    Parameters
    ----------
    in_filters : int
        Input channels.
    out_filters : int
        Output channels.
    reps : int
        Number of separable conv repetitions.
    strides : int
        Stride of the block.
    start_with_relu : bool
        Whether to start block with ReLU.
    grow_first : bool
        Whether to increase channels in the first conv.
    """

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        reps: int,
        strides: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ) -> None:
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(
                in_filters, out_filters, kernel_size=1, stride=strides, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters

        if grow_first:
            rep.append(self.relu if start_with_relu else nn.Identity())
            rep.append(
                SeparableConv2d(
                    in_channels=in_filters,
                    out_channels=out_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(
                SeparableConv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(
                SeparableConv2d(
                    in_channels=in_filters,
                    out_channels=out_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )

        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=strides, padding=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rep(x)

        if self.skip is not None:
            identity = self.skipbn(self.skip(identity))

        out = out + identity
        return out


class Xception39(nn.Module):
    """
    Xception39-like backbone used for BiSeNet Context Path.

    Output
    ------
    feat8  : feature map at stride 8
    feat16 : feature map at stride 16
    feat32 : feature map at stride 32
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.block1 = Block(16, 32, reps=3, strides=2, start_with_relu=False, grow_first=True)   # /8
        self.block2 = Block(32, 64, reps=3, strides=2, start_with_relu=True, grow_first=True)    # /16
        self.block3 = Block(64, 128, reps=3, strides=2, start_with_relu=True, grow_first=True)   # /32

        self.out_channels = (32, 64, 128)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize conv and batchnorm weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))   # /2
        x = self.relu(self.bn2(self.conv2(x)))   # /4

        feat8 = self.block1(x)                   # /8
        feat16 = self.block2(feat8)              # /16
        feat32 = self.block3(feat16)             # /32

        return feat8, feat16, feat32


def build_xception39(
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    strict: bool = False,
) -> Xception39:
    """
    Build Xception39 backbone.

    Behavior
    --------
    - Always create backbone from source code.
    - If `pretrained=True`, try loading checkpoint.
    - If `pretrained=False`, return randomly initialized backbone.

    Parameters
    ----------
    pretrained : bool
        Whether to load pretrained weights.
    pretrained_path : Optional[str]
        Path to local checkpoint file.
    strict : bool
        Whether to enforce exact state_dict key matching.

    Returns
    -------
    Xception39
        Constructed backbone model.
    """
    model = Xception39()

    if pretrained:
        if pretrained_path is None:
            raise ValueError("pretrained=True nhưng pretrained_path=None")
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=strict)

    return model