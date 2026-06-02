"""Image encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.models.detection as tvd
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from project.models.model_store import (
    ensure_local_hf_snapshot,
    ensure_torchvision_state_dict,
    load_hf_resource,
    load_state_dict_from_local_dir,
    slugify_model_id,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _default_image_transform(
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size)), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class BaseImageEncoder(nn.Module):
    def __init__(self, freeze_backbone: bool):
        super().__init__()
        self.freeze_backbone = bool(freeze_backbone)
        self.output_dim = 0

    def train(self, mode: bool = True):
        return super().train(False if self.freeze_backbone else mode)

    def get_image_transform(self, image_size: int):
        raise NotImplementedError


class ResNetEncoder(BaseImageEncoder):
    """ResNet feature extractor with optional local pretrained weights."""

    def __init__(
        self,
        variant: str,
        local_model_root: Path,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__(freeze_backbone=freeze_backbone)
        if variant == "resnet18":
            model = tvm.resnet18(weights=None)
            weights_enum = getattr(tvm, "ResNet18_Weights").DEFAULT
        elif variant == "resnet50":
            model = tvm.resnet50(weights=None)
            weights_enum = getattr(tvm, "ResNet50_Weights").DEFAULT
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        if pretrained:
            local_dir = Path(local_model_root) / "image" / slugify_model_id(variant)
            state_dict = ensure_torchvision_state_dict(
                url=weights_enum.url,
                local_dir=local_dir,
                filename="weights.pt",
            )
            model.load_state_dict(state_dict)
        self.output_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        self.backbone = model
        if self.freeze_backbone:
            _freeze_module(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_image_transform(self, image_size: int):
        return _default_image_transform(
            image_size=image_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )


class VGG19Encoder(BaseImageEncoder):
    def __init__(
        self,
        local_model_root: Path,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__(freeze_backbone=freeze_backbone)
        model = tvm.vgg19(weights=None)
        if pretrained:
            weights_enum = getattr(tvm, "VGG19_Weights").DEFAULT
            local_dir = Path(local_model_root) / "image" / slugify_model_id("vgg19")
            state_dict = ensure_torchvision_state_dict(
                url=weights_enum.url,
                local_dir=local_dir,
                filename="weights.pt",
            )
            model.load_state_dict(state_dict)
        self.output_dim = 4096
        self.backbone = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
            *list(model.classifier.children())[:-1],
        )
        if self.freeze_backbone:
            _freeze_module(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_image_transform(self, image_size: int):
        return _default_image_transform(
            image_size=image_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )


class DINOv2Encoder(BaseImageEncoder):
    def __init__(
        self,
        local_model_root: Path,
        model_name: str,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(freeze_backbone=freeze_backbone)
        if not pretrained:
            raise ValueError("DINOv2 requires pretrained weights in this run.")
        local_dir = Path(local_model_root) / "image" / slugify_model_id(model_name)
        self.processor = load_hf_resource(loader=AutoImageProcessor, model_id=model_name, local_dir=local_dir)
        self.backbone = load_hf_resource(loader=AutoModel, model_id=model_name, local_dir=local_dir)
        self.output_dim = int(getattr(self.backbone.config, "hidden_size"))
        if self.freeze_backbone:
            _freeze_module(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        return outputs.last_hidden_state[:, 0, :]

    def get_image_transform(self, image_size: int):
        image_mean = getattr(self.processor, "image_mean", IMAGENET_MEAN)
        image_std = getattr(self.processor, "image_std", IMAGENET_STD)
        return _default_image_transform(
            image_size=image_size,
            mean=image_mean,
            std=image_std,
            interpolation=InterpolationMode.BICUBIC,
        )


class FasterRCNNEncoder(BaseImageEncoder):
    def __init__(
        self,
        local_model_root: Path,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(freeze_backbone=freeze_backbone)
        model = tvd.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        if pretrained:
            weights_enum = getattr(tvd, "FasterRCNN_ResNet50_FPN_V2_Weights").DEFAULT
            local_dir = Path(local_model_root) / "image" / slugify_model_id(
                "fasterrcnn_resnet50_fpn_v2"
            )
            state_dict = ensure_torchvision_state_dict(
                url=weights_enum.url,
                local_dir=local_dir,
                filename="weights.pt",
            )
            model.load_state_dict(state_dict)
        self.backbone = model.backbone
        self.output_dim = 256
        if self.freeze_backbone:
            _freeze_module(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = []
        for feat in feats.values():
            pooled.append(F.adaptive_avg_pool2d(feat, output_size=1).flatten(1))
        return torch.stack(pooled, dim=0).mean(dim=0)

    def get_image_transform(self, image_size: int):
        # Current image provider requires fixed-size batched tensors, so detection inputs are
        # normalized with ImageNet statistics and square-resized at the configured size.
        return _default_image_transform(
            image_size=image_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )


class InceptionResNetV2Encoder(BaseImageEncoder):
    def __init__(
        self,
        local_model_root: Path,
        model_name: str,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(freeze_backbone=freeze_backbone)
        try:
            import timm
            from timm.data import create_transform, resolve_data_config
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("timm is required for inception_resnet_v2 experiments.") from exc

        self._timm_create_transform = create_transform
        self._timm_resolve_data_config = resolve_data_config
        self.model_name = str(model_name)
        model = timm.create_model(self.model_name, pretrained=False)
        if pretrained:
            local_dir = Path(local_model_root) / "image" / slugify_model_id(self.model_name)
            ensure_local_hf_snapshot(f"timm/{self.model_name}", local_dir)
            state_dict = load_state_dict_from_local_dir(local_dir)
            model.load_state_dict(state_dict, strict=False)
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(0)
        self.backbone = model
        self.output_dim = int(getattr(model, "num_features"))
        if self.freeze_backbone:
            _freeze_module(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_image_transform(self, image_size: int):
        data_config = self._timm_resolve_data_config({}, model=self.backbone)
        data_config["input_size"] = (3, int(image_size), int(image_size))
        return self._timm_create_transform(**data_config, is_training=False)


def build_image_encoder(
    name: str,
    local_model_root: Path,
    model_name: str = "",
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    if name == "resnet18":
        return ResNetEncoder(
            "resnet18",
            local_model_root=local_model_root,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if name == "resnet50":
        return ResNetEncoder(
            "resnet50",
            local_model_root=local_model_root,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if name == "vgg19":
        return VGG19Encoder(
            local_model_root=local_model_root,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if name == "dinov2_base":
        resolved_name = str(model_name or "facebook/dinov2-base")
        return DINOv2Encoder(
            local_model_root=local_model_root,
            model_name=resolved_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if name == "fasterrcnn_resnet50_fpn_v2":
        return FasterRCNNEncoder(
            local_model_root=local_model_root,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if name == "inception_resnet_v2":
        resolved_name = str(model_name or "inception_resnet_v2.tf_ens_adv_in1k")
        return InceptionResNetV2Encoder(
            local_model_root=local_model_root,
            model_name=resolved_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if name in {"vit", "dinov2"}:
        raise NotImplementedError(f"image encoder not implemented: {name}")
    raise ValueError(f"Unknown image encoder: {name}")
