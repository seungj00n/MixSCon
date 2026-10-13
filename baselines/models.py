"""Model and preprocessing adapters for the study's local implementations."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Union, List, Tuple, Dict, Sequence, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

CLASS_NAMES = ["Acinar", "Cribriform", "Lepidic", "Micropapillary", "Papillary", "Solid"]


N_CLASSES = len(CLASS_NAMES)


STANDARD_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


ALSUBAIE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


CELLOMAP_TRANSFORM = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


class ManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, mode: str):
        self.manifest = manifest
        self.mode = mode

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.manifest.iloc[index]
        if self.mode == "cellomaps":
            with Image.open(row["cellmap_path"]) as opened:
                image = opened.convert("RGB")
            tensor = CELLOMAP_TRANSFORM(image)
        else:
            with Image.open(row["raw_path"]) as opened:
                image = opened.convert("RGB")
            if self.mode == "alsubaie":
                width, height = image.size
                small = image.resize(
                    (max(1, int(width * 0.5)), max(1, int(height * 0.5))), Image.BICUBIC
                )
                pseudo_low = small.resize((width, height), Image.BICUBIC)
                tensor = torch.cat(
                    [ALSUBAIE_TRANSFORM(image), ALSUBAIE_TRANSFORM(pseudo_low)], dim=0
                )
            else:
                tensor = STANDARD_TRANSFORM(image)
        return tensor, index


class ResNet50SixChannel(nn.Module):
    def __init__(self, num_classes: int = N_CLASSES):
        super().__init__()
        base = models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


class DilationLayer(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.register_buffer("kernel", torch.ones(1, 1, kernel_size, kernel_size))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        x = F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=channels)
        return self.pool(x)


class CellOMapsResNet50(nn.Module):
    def __init__(self, num_classes: int = N_CLASSES):
        super().__init__()
        self.dilate = DilationLayer(kernel_size=5)
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.dilate(x))


class MixSConBackbone(nn.Module):
    """Architecture matching the stored MixSCon CE/FL checkpoints."""

    def __init__(
        self,
        num_classes: int,
        out_dim: int,
        backbone: str,
        use_timm: bool,
        use_cosine_classifier: bool,
        tau_cls: float,
    ):
        super().__init__()
        self.use_timm = use_timm
        self.use_cosine_classifier = use_cosine_classifier
        self.tau_cls = tau_cls

        if use_timm:
            try:
                import timm
            except ImportError as exc:
                raise RuntimeError("timm is required by this MixSCon checkpoint") from exc
            self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
            feat_dim = self.backbone.num_features
        else:
            if backbone != "resnet50":
                raise ValueError(f"Unsupported torchvision MixSCon backbone: {backbone}")
            rn = models.resnet50(weights=None)
            self.conv1, self.bn1, self.relu, self.maxpool = (
                rn.conv1,
                rn.bn1,
                rn.relu,
                rn.maxpool,
            )
            self.layer1, self.layer2, self.layer3, self.layer4 = (
                rn.layer1,
                rn.layer2,
                rn.layer3,
                rn.layer4,
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.backbone = None
            feat_dim = 2048

        self.proj_head = nn.Linear(feat_dim, out_dim)
        if use_cosine_classifier:
            self.cls_weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        else:
            self.cls_head = nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Linear(feat_dim, 1024),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, num_classes),
            )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_timm:
            return self.backbone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._features(x)
        if self.use_cosine_classifier:
            features_norm = F.normalize(features, dim=1)
            weights_norm = F.normalize(self.cls_weight, dim=1)
            return (features_norm @ weights_norm.T) / max(self.tau_cls, 1e-12)
        return self.cls_head(features)


def load_torch(path: Path) -> object:
    # These are local, user-provided research checkpoints. CPU loading keeps GPU memory bounded.
    return torch.load(str(path), map_location="cpu", weights_only=True)


def extract_state_dict(checkpoint: object, key: str = "model_state_dict") -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint).__name__}")
    if key in checkpoint:
        state = checkpoint[key]
    else:
        state = checkpoint
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint field {key!r} is not a state dictionary")
    return state


def build_model(
    name: str, checkpoint_spec: Union[Path, List[Path]], device: torch.device
) -> Union[nn.Module, List[nn.Module]]:
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        model.load_state_dict(extract_state_dict(load_torch(Path(checkpoint_spec))))
        return model.to(device).eval()

    if name == "alsubaie":
        model = ResNet50SixChannel(N_CLASSES)
        model.load_state_dict(extract_state_dict(load_torch(Path(checkpoint_spec))))
        return model.to(device).eval()

    if name == "ding":
        ensemble: List[nn.Module] = []
        for path in checkpoint_spec:  # type: ignore[union-attr]
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
            model.load_state_dict(extract_state_dict(load_torch(Path(path))))
            ensemble.append(model.to(device).eval())
        return ensemble

    if name == "cellomaps":
        checkpoint = load_torch(Path(checkpoint_spec))
        model = CellOMapsResNet50(N_CLASSES)
        model.load_state_dict(extract_state_dict(checkpoint, key="model"))
        return model.to(device).eval()

    if name in {"mixscon_ce", "mixscon_fl", "class_mixup"}:
        checkpoint = load_torch(Path(checkpoint_spec))
        state = extract_state_dict(checkpoint)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
        use_cosine = bool(config.get("use_cosine_classifier", "cls_weight" in state))
        out_dim = int(state["proj_head.weight"].shape[0])
        model = MixSConBackbone(
            num_classes=int(checkpoint.get("num_classes", N_CLASSES)),
            out_dim=out_dim,
            backbone=str(checkpoint.get("backbone", config.get("backbone", "resnet50"))),
            use_timm=bool(config.get("use_timm", False)),
            use_cosine_classifier=use_cosine,
            tau_cls=float(config.get("temperature_cls", 0.07)),
        )
        model.load_state_dict(state)
        return model.to(device).eval()

    if name == "vit16":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "ViT-16 requires timm. Install it in this environment with: pip install timm"
            ) from exc
        checkpoint = load_torch(Path(checkpoint_spec))
        state = extract_state_dict(checkpoint)
        model_name = (
            str(checkpoint.get("model_name", "vit_base_patch16_224"))
            if isinstance(checkpoint, Mapping)
            else "vit_base_patch16_224"
        )
        model = timm.create_model(model_name, pretrained=False, num_classes=N_CLASSES)
        model.load_state_dict(state)
        return model.to(device).eval()

    raise ValueError(f"Unknown model: {name}")


def dataset_mode(model_name: str) -> str:
    if model_name == "alsubaie":
        return "alsubaie"
    if model_name == "cellomaps":
        return "cellomaps"
    return "standard"


@torch.inference_mode()
def infer_probabilities(
    model: Union[nn.Module, List[nn.Module]],
    manifest: pd.DataFrame,
    mode: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = ManifestDataset(manifest, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    probabilities = np.empty((len(dataset), N_CLASSES), dtype=np.float32)

    for batch_number, (images, indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if isinstance(model, list):
            batch_probs = torch.stack([m(images).softmax(dim=1) for m in model], dim=0).mean(dim=0)
        else:
            output = model(images)
            if isinstance(output, tuple):
                output = output[1]
            batch_probs = output.softmax(dim=1)
        probabilities[indices.numpy()] = batch_probs.cpu().numpy()
        if batch_number % 10 == 0 or batch_number == len(loader):
            print(f"    inference batches: {batch_number}/{len(loader)}", flush=True)
    return probabilities
