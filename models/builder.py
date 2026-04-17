from models.backbones.xception39 import build_xception39
from models.bisenetv1 import BiSeNetV1

def build_backbone(cfg, paths):
    if cfg["backbone_name"] == "xception39":
        return build_xception39(
            pretrained=cfg["backbone_pretrained"],
            pretrained_path=None if paths["backbone_pretrained_path"] is None else str(paths["backbone_pretrained_path"]),
            strict=cfg["backbone_strict_load"],
        )
    raise ValueError(f"Unsupported backbone: {cfg['backbone_name']}")


def build_model(cfg, paths):
    backbone = build_backbone(cfg, paths)
    model = BiSeNetV1(
        num_classes=cfg["num_classes"],
        backbone=backbone,
    )
    return model