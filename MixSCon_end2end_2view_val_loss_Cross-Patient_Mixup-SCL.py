# -*- coding: utf-8 -*-
"""
Optimal MixSCon for LUAD (Patient-aware + Adaptive MixSCL + (Optional) Cosine Classifier)

Dataset layout:
    CV_ROOT/Round{1..5}/(train|val)/{1_acinar,...,6_solid}
    TEST_ROOT with identical class subdirectories

Patient proxy:
    The filename prefix before the first underscore (e.g., CASE123_*.png) is used as patient/WSI ID.

Training loss:
    L_total = L_SCL(patient-aware) + beta * L_MixSCL(adaptive, cross-patient preferred) + gamma * L_CE

Validation / Early Stop:
    CE-only on logits for stable validation feedback (same as baseline)

Saved artifacts per round:
    best_model_round_X.pth (with optimizer state),
    final_weights_round_X.pth (state_dict only),
    checkpoint_round_X.pth (full metadata),
    confusion_matrix_round_X.png,
    training_history_round_X.png,
    results_round_X.json

Cross-validation summary:
    best_overall_model.pth (selected by highest test accuracy across rounds),
    cross_validation_results.json
"""

import os, json, time, shutil, math
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False


# =========================
# Config
# =========================
CONFIG = {
    # Training / loss weights
    'epochs': 200,
    'patience': 30,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,

    # Temperatures
    'temperature_scl': 0.07,   # SupCon / MixSCL
    'temperature_cls': 0.07,   # (optional) cosine classifier

    # Loss coefficients
    'beta': 1.0,               # Mixup-SCL
    'gamma': 1.0,              # CE

    # Patient-aware weighting
    'w_same_patient': 1.0,
    'w_cross_patient': 1.8,

    # Adaptive Mixup parameters
    'mix_beta_alpha': 0.5,     # initial Beta(alpha,alpha) (fallback only)
    'mix_kappa': 0.5,          # lambda = clip(0.5 + kappa*(0.5 - distance), eps, 1-eps)
    'mix_eps': 0.05,           # clipping epsilon
    'prefer_cross_patient_for_mix': True,
    'prefer_same_class_for_mix': True,

    # Model
    'backbone': 'resnet50',
    'use_timm': False,
    'proj_dim': 128,
    'use_cosine_classifier': True,

    # Augmentations
    'two_views': True,
    'color_jitter': (0.2, 0.2, 0.2, 0.05),
    'gaussian_blur_p': 0.2,
    'hflip_p': 0.5,
    'vflip_p': 0.5,
    'rot90': True,

    # Misc
    'num_workers': 4,
    'pin_memory': True,
    'min_delta': 1e-4,
    'seed': 2025,
}


# =========================
# Seed utilities
# =========================
def set_seed(seed: int = 2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Transforms
# =========================
def _normalize_tfm():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

class RandomRotate90:
    def __init__(self, enable: bool = True):
        self.enable = enable
    def __call__(self, x):
        if not self.enable:
            return x
        k = torch.randint(0, 4, (1,)).item()
        if k == 0:
            return x
        return transforms.functional.rotate(x, 90 * k)

def build_train_transforms(cfg: Dict):
    cj = transforms.ColorJitter(*cfg['color_jitter'])
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    common_geo = [
        transforms.RandomHorizontalFlip(p=cfg['hflip_p']),
        transforms.RandomVerticalFlip(p=cfg['vflip_p']),
        RandomRotate90(enable=cfg['rot90']),
        transforms.Resize((224, 224)),
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
        _normalize_tfm(),
    ]
    t = transforms.Compose([
        transforms.RandomApply([cj], p=0.8),
        transforms.RandomApply([blur], p=cfg['gaussian_blur_p']),
        *common_geo,
    ])
    return t, t

def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
        _normalize_tfm(),
    ])


# =========================
# Dataset wrappers (include patient ID)
# =========================
def _extract_patient_id_from_path(path: str) -> str:
    name = Path(path).name
    stem = Path(path).stem
    if '_' in name:
        return stem.split('_')[0]
    return stem

class ImageFolderWithPatient(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, y = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        patient = _extract_patient_id_from_path(path)
        return img, y, patient

class TwoViewDatasetWithPatient(Dataset):
    """Return two independent augmented views + patient ID."""
    def __init__(self, folder: datasets.ImageFolder, t1: transforms.Compose, t2: transforms.Compose):
        self.base = folder
        self.t1 = t1
        self.t2 = t2
        self.samples = self.base.samples
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.loader = self.base.loader
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = self.loader(path)
        return self.t1(img), self.t2(img), y, _extract_patient_id_from_path(path)


# =========================
# Dataloaders
# =========================
def make_loaders_for_round(round_dir: Path, test_dir: Path, cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    train_t1, train_t2 = build_train_transforms(cfg)
    eval_tf = build_eval_transform()

    train_base = datasets.ImageFolder(round_dir / 'train')
    val_set = ImageFolderWithPatient(round_dir / 'val', transform=eval_tf)
    test_set = ImageFolderWithPatient(test_dir, transform=eval_tf)

    if cfg['two_views']:
        train_set = TwoViewDatasetWithPatient(train_base, t1=train_t1, t2=train_t2)
    else:
        train_set = ImageFolderWithPatient(round_dir / 'train', transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

    num_classes = len(train_base.classes)
    return train_loader, val_loader, test_loader, num_classes, train_base.classes


# =========================
# Model
# =========================
class End2EndBackbone(nn.Module):
    """Backbone + projection head + (optional) cosine classifier."""
    def __init__(self, num_classes: int, out_dim: int = 128, backbone: str = 'resnet50',
                 use_timm: bool = False, use_cosine_classifier: bool = True, tau_cls: float = 0.07):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.use_timm = use_timm and HAS_TIMM
        self.backbone_name = backbone
        self.use_cosine_classifier = use_cosine_classifier
        self.tau_cls = tau_cls

        if self.use_timm:
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            feat_dim = self.backbone.num_features
        else:
            assert backbone == 'resnet50'
            rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.conv1, self.bn1, self.relu, self.maxpool = rn.conv1, rn.bn1, rn.relu, rn.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = rn.layer1, rn.layer2, rn.layer3, rn.layer4
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            feat_dim = 2048
            self.backbone = None

        self.proj_head = nn.Linear(feat_dim, out_dim)

        if use_cosine_classifier:
            self.cls_weight = nn.Parameter(torch.empty(num_classes, feat_dim))
            nn.init.kaiming_normal_(self.cls_weight, nonlinearity='linear')
        else:
            self.cls_head = nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Linear(feat_dim, 1024), nn.SiLU(), nn.Dropout(0.2),
                nn.Linear(1024, 512), nn.SiLU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.SiLU(),
                nn.Linear(256, 128), nn.SiLU(),
                nn.Linear(128, num_classes)
            )
            for m in self.cls_head.modules():
                if isinstance(m, nn.Linear):
                    if m.out_features == num_classes:
                        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                        if m.bias is not None: nn.init.zeros_(m.bias)
                    else:
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        if m.bias is not None: nn.init.zeros_(m.bias)

        nn.init.kaiming_normal_(self.proj_head.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj_head.bias)

    def _forward_feat(self, x):
        if self.use_timm:
            f = self.backbone(x)
        else:
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            x = self.avgpool(x); f = torch.flatten(x, 1)
        return f

    def _cls_logits(self, feat):
        if self.use_cosine_classifier:
            f_n = F.normalize(feat, dim=1)
            w_n = F.normalize(self.cls_weight, dim=1)
            logits = (f_n @ w_n.T) / max(self.tau_cls, 1e-12)
            return logits
        else:
            return self.cls_head(feat)

    def forward(self, x):
        feat = self._forward_feat(x)
        proj = F.normalize(self.proj_head(feat), dim=1)
        logits = self._cls_logits(feat)
        return proj, logits, feat


# =========================
# Losses
# =========================
def _mask_diagonal_for_softmax(sim: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    return sim.masked_fill(eye, -1e9)

def patient_aware_supcon(z: torch.Tensor, y: torch.Tensor, patients: List[str],
                         tau: float, w_same: float, w_cross: float) -> torch.Tensor:
    """
    Patient-aware weighted supervised contrastive loss.
    Assigns different weights to same-class pairs depending on
    whether they originate from the same patient or different patients.
    """
    B = z.size(0)
    sim = (z @ z.T) / max(tau, 1e-12)
    sim = _mask_diagonal_for_softmax(sim)
    logp = F.log_softmax(sim, dim=1)

    y = y.view(-1,1)
    pos = (y == y.T).float().to(z.device)

    same_patient = torch.zeros(B, B, device=z.device, dtype=torch.bool)
    pids = list(patients)
    for i in range(B):
        for j in range(B):
            if i != j and pids[j] == pids[i]:
                same_patient[i, j] = True

    omega = torch.where(same_patient, torch.tensor(w_same, device=z.device), torch.tensor(w_cross, device=z.device)).float()

    eye = torch.eye(B, device=z.device)
    pos = pos * (1 - eye)
    weighted_pos = omega * pos

    denom = weighted_pos.sum(dim=1).clamp(min=1e-12)
    loss_i = -(logp * weighted_pos).sum(dim=1) / denom
    zero_row = (pos.sum(dim=1) == 0)
    loss_i = torch.where(zero_row, torch.zeros_like(loss_i), loss_i)
    return loss_i.mean()

def _choose_mix_partner(y: torch.Tensor, patients: List[str],
                        prefer_cross_patient: bool = True, prefer_same_class: bool = True) -> torch.Tensor:
    """
    Partner selection priority:
        1) same class AND different patient
        2) same class
        3) any sample
    """
    B = y.size(0)
    pids = list(patients)
    partner = torch.arange(B)

    cls_to_indices = {}
    for i in range(B):
        cls_to_indices.setdefault(int(y[i].item()), []).append(i)

    for i in range(B):
        yi = int(y[i].item())
        cand = []

        if prefer_same_class:
            same_class = cls_to_indices.get(yi, [])
            if prefer_cross_patient:
                cand = [j for j in same_class if j != i and pids[j] != pids[i]]
            if not cand:
                cand = [j for j in same_class if j != i]

        if not cand:
            cand = [j for j in range(B) if j != i]

        if cand:
            partner[i] = np.random.choice(cand)
        else:
            partner[i] = i

    return partner

def adaptive_mixscl(z: torch.Tensor, y: torch.Tensor, patients: List[str],
                    tau: float, num_classes: int,
                    kappa: float = 0.5, eps: float = 0.05,
                    prefer_cross_patient: bool = True, prefer_same_class: bool = True) -> torch.Tensor:
    """
    Adaptive MixSCL:
        z_mix_i = λ_i*z_i + (1-λ_i)*z_j
        λ_i = clip(0.5 + kappa*(0.5 - d_ij), eps, 1-eps)
        d_ij = (1 - cos(z_i,z_j)) / 2
        Uses soft labels based on mixing proportions.
    """
    device = z.device
    B, D = z.size()
    partner = _choose_mix_partner(y, patients, prefer_cross_patient, prefer_same_class).to(device)
    z_j = z[partner]

    cos_ij = (z * z_j).sum(dim=1).clamp(-1+1e-6, 1-1e-6)
    d_ij = (1 - cos_ij) * 0.5
    lam = torch.clamp(0.5 + kappa * (0.5 - d_ij), eps, 1 - eps).view(B, 1)

    z_mix = lam * z + (1 - lam) * z_j

    y_one = F.one_hot(y, num_classes=num_classes).float()
    y_partner = F.one_hot(y[partner], num_classes=num_classes).float()
    y_soft = lam * y_one + (1 - lam) * y_partner

    labels_onehot_batch = F.one_hot(y, num_classes=num_classes).float()
    pos_mask = y_soft @ labels_onehot_batch.T

    sim = (z_mix @ z.T) / max(tau, 1e-12)
    sim = _mask_diagonal_for_softmax(sim)
    logp = F.log_softmax(sim, dim=1)

    eye = torch.eye(B, device=device)
    pos_mask = pos_mask * (1 - eye)

    denom = pos_mask.sum(dim=1).clamp(min=1e-12)
    loss_i = -(logp * pos_mask).sum(dim=1) / denom

    zero_row = (pos_mask.sum(dim=1) == 0)
    loss_i = torch.where(zero_row, torch.zeros_like(loss_i), loss_i)
    return loss_i.mean()


# =========================
# Training / Validation / Test
# =========================
def train_one_epoch(model, loader, optimizer, device, num_classes, cfg: Dict):
    model.train()
    total = {'loss':0.0, 'scl':0.0, 'mix':0.0, 'ce':0.0}
    n = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        if cfg['two_views']:
            v1, v2, labels, patients = batch
            v1, v2 = v1.to(device), v2.to(device)
            labels = labels.to(device)
            patients = list(patients)
            images = torch.cat([v1, v2], dim=0)
            labels_dup = torch.cat([labels, labels], dim=0)
            patients_dup = patients + patients
        else:
            images, labels, patients = batch
            images, labels = images.to(device), labels.to(device)
            patients = list(patients)
            labels_dup = labels
            patients_dup = patients

        proj, logits, feat = model(images)

        L_scl = patient_aware_supcon(
            proj, labels_dup, patients_dup,
            tau=cfg['temperature_scl'],
            w_same=cfg['w_same_patient'],
            w_cross=cfg['w_cross_patient']
        )

        if cfg['two_views']:
            B = labels.size(0)
            z_base = proj[:B]
            y_base = labels
            patients_base = patients
        else:
            z_base = proj
            y_base = labels
            patients_base = patients

        L_mix = adaptive_mixscl(
            z=z_base, y=y_base, patients=patients_base,
            tau=cfg['temperature_scl'], num_classes=num_classes,
            kappa=cfg['mix_kappa'], eps=cfg['mix_eps'],
            prefer_cross_patient=cfg['prefer_cross_patient_for_mix'],
            prefer_same_class=cfg['prefer_same_class_for_mix']
        )

        if cfg['two_views']:
            l1, l2 = torch.chunk(logits, 2, dim=0)
            L_ce = 0.5 * (F.cross_entropy(l1, labels) + F.cross_entropy(l2, labels))
            bs = labels.size(0)
        else:
            L_ce = F.cross_entropy(logits, labels)
            bs = labels.size(0)

        loss = L_scl + cfg['beta']*L_mix + cfg['gamma']*L_ce
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total['loss']  += float(loss.item())  * bs
        total['scl']   += float(L_scl.item()) * bs
        total['mix']   += float(L_mix.item()) * bs
        total['ce']    += float(L_ce.item())  * bs
        n += bs

        pbar.set_postfix(loss=f"{total['loss']/max(n,1):.4f}")

    return (total['loss']/n, total['scl']/n, total['mix']/n, total['ce']/n)

@torch.no_grad()
def validate(model, loader, device, cfg: Dict):
    """
    Validation loss uses CE-only (contrastive losses excluded).
    """
    model.eval()
    total_loss = 0.0
    correct = n = 0
    pbar = tqdm(loader, desc="Val", leave=False)
    for images, labels, _patients in pbar:
        images, labels = images.to(device), labels.to(device)
        _proj, logits, _feat = model(images)
        loss_ce  = F.cross_entropy(logits, labels)

        bs = labels.size(0)
        total_loss += loss_ce.item() * bs
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        n += bs
        pbar.set_postfix(acc=f"{100*correct/max(n,1):.2f}%")
    acc = 100.0 * correct / max(1, n)
    return total_loss / n, acc

@torch.no_grad()
def test_eval(model, loader, device, class_names: List[str]) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Test", leave=False)
    for x, y, _patients in pbar:
        x = x.to(device)
        _, logits, _ = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(y.numpy()))
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return dict(
        accuracy=acc, precision=precision, recall=recall, f1=f1,
        precision_per_class=p_c, recall_per_class=r_c, f1_per_class=f1_c,
        confusion_matrix=cm, predictions=all_preds, labels=all_labels,
        class_names=class_names
    )


# =========================
# Plot & Save
# =========================
def plot_confusion_matrix(cm, class_names, round_num, save_dir: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Round {round_num}')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_round_{round_num}.png', dpi=300)
    plt.close()

def save_training_history(history: Dict, round_num: int, save_dir: Path):
    epochs = range(1, len(history['train_total']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, history['train_total'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'r-', label='Val Loss')
    ax1.set_title(f'Loss - Round {round_num}'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history['val_acc'], 'g-', label='Val Acc (%)')
    ax2.set_title(f'Accuracy - Round {round_num}'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'training_history_round_{round_num}.png', dpi=300)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.counter = 0
            self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


# =========================
# Train one round
# =========================
def train_single_round(round_num: int, cv_root: Path, test_root: Path, save_root: Path, cfg: Dict) -> Dict:
    print(f"\n{'='*70}\nTRAINING ROUND {round_num}\n{'='*70}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    round_dir = cv_root / f'Round{round_num}'
    train_loader, val_loader, test_loader, num_classes, class_names = make_loaders_for_round(round_dir, test_root, cfg)

    model = End2EndBackbone(num_classes=num_classes, out_dim=cfg['proj_dim'],
                            backbone=cfg['backbone'], use_timm=cfg['use_timm'],
                            use_cosine_classifier=cfg['use_cosine_classifier'],
                            tau_cls=cfg['temperature_cls']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    early_stopping = EarlyStopping(patience=cfg['patience'], min_delta=cfg['min_delta'])
    history = dict(train_total=[], train_scl=[], train_mix=[], train_ce=[],
                   val_loss=[], val_acc=[])

    best_val = float('inf'); best_val_acc = 0.0
    best_path = save_root / f'best_model_round_{round_num}.pth'

    start = time.time()
    for ep in range(1, cfg['epochs']+1):
        tr_tot, tr_scl, tr_mix, tr_ce = train_one_epoch(
            model, train_loader, optimizer, device, num_classes, cfg
        )
        va_loss, va_acc = validate(model, val_loader, device, cfg)

        history['train_total'].append(tr_tot)
        history['train_scl'].append(tr_scl)
        history['train_mix'].append(tr_mix)
        history['train_ce'].append(tr_ce)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        print(f"Epoch {ep:03d} | Train: total={tr_tot:.4f} | SCL={tr_scl:.4f} Mix={tr_mix:.4f} "
              f"CE={tr_ce:.4f} | Val: loss={va_loss:.4f}, acc={va_acc:.2f}%")

        if best_val - va_loss > cfg['min_delta']:
            best_val = va_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': va_acc,
                'val_loss': va_loss,
                'config': cfg,
                'num_classes': num_classes,
                'backbone': cfg['backbone'],
            }, best_path)
            print(f"  Saved best checkpoint to {best_path}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc

        if early_stopping(va_loss, model):
            print(f"Early stopping at epoch {ep}")
            break

    train_time = time.time() - start
    print(f"Training time: {train_time/60:.2f} min | Best Val Acc: {best_val_acc:.2f}%")

    final_weights_path = save_root / f'final_weights_round_{round_num}.pth'
    torch.save(model.state_dict(), final_weights_path)

    ckpt_path = save_root / f'checkpoint_round_{round_num}.pth'
    torch.save({
        'round': round_num,
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'training_time': train_time,
        'epochs_trained': len(history['train_total']),
        'config': cfg,
        'class_names': class_names,
        'backbone': cfg['backbone'],
        'num_classes': num_classes
    }, ckpt_path)

    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights, strict=True)
        model.to(device)

    print("Evaluating on Test set...")
    test_res = test_eval(model, test_loader, device, class_names)

    results = {
        'round': round_num,
        'best_val_acc': best_val_acc,
        'training_time_sec': train_time,
        'epochs_trained': len(history['train_total']),
        'test_accuracy': float(test_res['accuracy']),
        'test_precision': float(test_res['precision']),
        'test_recall': float(test_res['recall']),
        'test_f1': float(test_res['f1']),
        'test_precision_per_class': test_res['precision_per_class'].tolist(),
        'test_recall_per_class': test_res['recall_per_class'].tolist(),
        'test_f1_per_class': test_res['f1_per_class'].tolist(),
        'class_names': class_names
    }

    save_training_history(history, round_num, save_root)
    plot_confusion_matrix(test_res['confusion_matrix'], class_names, round_num, save_root)
    with open(save_root / f'results_round_{round_num}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"ROUND {round_num} - Test Acc: {results['test_accuracy']:.4f}, F1: {results['test_f1']:.4f}")
    return results


# =========================
# Cross Validation Runner
# =========================
def run_cross_validation(cv_root: str, test_root: str, save_root: str, cfg: Dict):
    set_seed(cfg.get('seed', 2025))
    cv_root = Path(cv_root); test_root = Path(test_root); save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("5-FOLD CROSS VALIDATION - Optimal MM-SCL (Patient-aware + Adaptive MixSCL)")
    print("="*80)
    print(f"CV Root: {cv_root}")
    print(f"Test Root: {test_root}")
    print(f"Save Dir: {save_root}")
    print(f"Config: {cfg}")

    all_results = []
    for k in range(1, 6):
        try:
            res = train_single_round(k, cv_root, test_root, save_root, cfg)
            all_results.append(res)
        except Exception as e:
            print(f"Error in Round {k}: {e}")

    if not all_results:
        print("No successful rounds. Exiting.")
        return

    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    overall_stats = {}
    print("\nOVERALL RESULTS")
    print("-"*80)
    header = f"{'Metric':<12} {'Mean':<8} {'Std':<8} " + " ".join([f"R{r['round']}" for r in all_results])
    print(header); print("-"*80)
    for m in metrics:
        vals = [r[m] for r in all_results]
        overall_stats[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': list(map(float, vals))}
        line = f"{m.replace('test_', '').title():<12} {overall_stats[m]['mean']:<8.4f} {overall_stats[m]['std']:<8.4f} " + \
               " ".join([f"{v:<8.4f}" for v in overall_stats[m]['values']])
        print(line)

    best_round = max(all_results, key=lambda r: r['test_accuracy'])
    best_round_num = best_round['round']; best_acc = best_round['test_accuracy']
    print(f"\nBest round: Round {best_round_num} (Acc: {best_acc:.4f})")
    src = save_root / f'best_model_round_{best_round_num}.pth'
    if src.exists():
        shutil.copy2(src, save_root / 'best_overall_model.pth')
        print("Saved: best_overall_model.pth")

    summary = {
        'config': cfg,
        'individual_results': all_results,
        'overall_statistics': overall_stats,
        'summary': {
            'mean_accuracy': overall_stats['test_accuracy']['mean'],
            'std_accuracy': overall_stats['test_accuracy']['std'],
            'mean_f1': overall_stats['test_f1']['mean'],
            'std_f1': overall_stats['test_f1']['std']
        }
    }
    with open(save_root / 'cross_validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nAll results saved to:", save_root)


# =========================
# Main
# =========================
if __name__ == "__main__":
    CV_ROOT  = "/home/sj-baek/Manifold_Mixco/LUAD_Dataset_CV"
    TEST_ROOT= "/home/sj-baek/Manifold_Mixco/LUAD_Dataset_CV/test"
    BASE_SAVE = "/home/sj-baek/Manifold_Mixco/LUAD_ASAN/Optimal_MM_SCL_cv_results_multi_Pure"

    save_dir = Path(BASE_SAVE)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_cross_validation(CV_ROOT, TEST_ROOT, save_dir, CONFIG)

    print("\n" + "="*80)
    print("Experiment Completed: SCL + β MixSCL + γ CE")
    print("="*80)
    print("Results saved under:")
    print(f"  {BASE_SAVE}/")
