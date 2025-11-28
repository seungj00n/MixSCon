# -*- coding: utf-8 -*-
"""
5-Fold Cross Validation with Proposed End-to-End MM-SCL
- Backbone: ResNet50 (timm 백본도 옵션)
- Loss: SCL + beta*Mixup-SCL + gamma*CE
- 2-view augmentation (train only)
- 분자·분모 모두에서 자기항(대각) 제거

저장물:
- best_model_round_X.pth : 최적 검증 성능(optimizer 상태 포함)
- final_weights_round_X.pth : 최종 state_dict
- checkpoint_round_X.pth : 라운드별 전체 체크포인트
- confusion_matrix_round_X.png, training_history_round_X.png
- results_round_X.json
- best_overall_model.pth : 5 folds 중 test acc 최고 모델(각 라운드의 best 기준)
- cross_validation_results.json : 교차검증 요약
"""

import os, json, time, shutil
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
    # 학습/손실
    'epochs': 200,
    'patience': 30,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'temperature': 0.07,
    'beta': 1.0,   # Mixup-SCL 가중치
    'gamma': 1.0,  # CE 가중치
    # 모델
    'backbone': 'resnet50',
    'use_timm': False,  # True면 timm 백본 사용 (e.g., convnext_base 등)
    'proj_dim': 128,
    # 증강
    'two_views': True,   # train에서만 2-view
    'color_jitter': (0.2, 0.2, 0.2, 0.05),
    'gaussian_blur_p': 0.2,
    'hflip_p': 0.5,
    'vflip_p': 0.5,
    'rot90': True,
    # 기타
    'num_workers': 4,
    'pin_memory': True,
    'min_delta': 1e-4,
}

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

class TwoViewDataset(Dataset):
    """ImageFolder를 감싸 같은 원본으로부터 서로 다른 2개의 view 반환"""
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
        return self.t1(img), self.t2(img), y

def build_train_transforms(cfg: Dict):
    cj = transforms.ColorJitter(*cfg['color_jitter'])
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    common_geo = [
        transforms.RandomHorizontalFlip(p=cfg['hflip_p']),
        transforms.RandomVerticalFlip(p=cfg['vflip_p']),
        RandomRotate90(enable=cfg['rot90']),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _normalize_tfm(),
    ]
    t = transforms.Compose([
        transforms.RandomApply([cj], p=0.8),
        transforms.RandomApply([blur], p=cfg['gaussian_blur_p']),
        *common_geo,
    ])
    # 두 뷰 동일 정책
    return t, t

def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _normalize_tfm(),
    ])

# =========================
# Dataloaders
# =========================
def make_loaders_for_round(round_dir: Path, test_dir: Path, cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """
    round_dir: RoundK 폴더 (train/val 포함)
    test_dir: 공통 test 폴더
    """
    train_t1, train_t2 = build_train_transforms(cfg)
    eval_tf = build_eval_transform()

    train_base = datasets.ImageFolder(round_dir / 'train')
    val_set = datasets.ImageFolder(round_dir / 'val', transform=eval_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    if cfg['two_views']:
        train_set = TwoViewDataset(train_base, t1=train_t1, t2=train_t2)
    else:
        train_set = datasets.ImageFolder(round_dir / 'train', transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'])

    num_classes = len(train_base.classes)
    return train_loader, val_loader, test_loader, num_classes, train_base.classes

# =========================
# Losses (SCL + Mixup-SCL) — feature-level mixup
# =========================
def manifold_mixup(features: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float = 0.4):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    b = features.size(0)
    index = torch.randperm(b, device=features.device)
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    mixed_features = lam * features + (1 - lam) * features[index]
    mixed_labels   = lam * labels_one_hot + (1 - lam) * labels_one_hot[index]
    return mixed_features, mixed_labels, lam, index

def _mask_diagonal_for_softmax(sim: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    return sim.masked_fill(eye, -1e9)

def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07, eps: float = 1e-12):
    feats = F.normalize(features, dim=1)
    sim = torch.matmul(feats, feats.T) / max(temperature, 1e-12)
    sim = _mask_diagonal_for_softmax(sim)  # 분모에서 자기항 제거
    log_prob = F.log_softmax(sim, dim=1)

    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    eye = torch.eye(mask.size(0), device=mask.device)
    mask = mask * (1 - eye)  # 분자에서 자기항 제거

    denom = mask.sum(dim=1).clamp(min=1)
    loss = -torch.sum(mask * log_prob, dim=1) / (denom + eps)
    zero_pos = (denom == 0)
    if zero_pos.any():
        loss = torch.where(zero_pos, torch.zeros_like(loss), loss)
    return loss.mean()

def manifold_mixup_scl(features: torch.Tensor, labels: torch.Tensor, num_classes: int, temperature: float = 0.07, eps: float = 1e-12):
    with torch.no_grad():
        labels_oh = F.one_hot(labels, num_classes=num_classes).float()

    mixed_feats, mixed_labels, _, _ = manifold_mixup(features, labels, num_classes)
    mixed_feats = F.normalize(mixed_feats, dim=1)
    base_feats  = F.normalize(features,     dim=1)

    sim = torch.matmul(mixed_feats, base_feats.T) / max(temperature, 1e-12)
    sim = _mask_diagonal_for_softmax(sim)  # 분모에서 자기항 제거
    log_prob = F.log_softmax(sim, dim=1)

    pos_mask = torch.matmul(mixed_labels, labels_oh.T)
    eye = torch.eye(pos_mask.size(0), device=pos_mask.device)
    pos_mask = pos_mask * (1 - eye)  # 분자에서 자기항 제거

    denom = pos_mask.sum(dim=1).clamp(min=1e-12)
    loss = -torch.sum(pos_mask * log_prob, dim=1) / (denom + eps)
    return loss.mean()

def contrastive_total(features: torch.Tensor, labels: torch.Tensor, num_classes: int, temperature: float, beta: float):
    c1 = supervised_contrastive_loss(features, labels, temperature)
    c2 = manifold_mixup_scl(features, labels, num_classes, temperature)
    return c1 + beta * c2

# =========================
# Model
# =========================
class End2EndBackbone(nn.Module):
    def __init__(self, num_classes: int, out_dim: int = 128, backbone: str = 'resnet50', use_timm: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.use_timm = use_timm and HAS_TIMM
        self.backbone_name = backbone

        if self.use_timm:
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            feat_dim = self.backbone.num_features
        else:
            assert backbone == 'resnet50', "Only torchvision resnet50 unless use_timm=True"
            rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.conv1, self.bn1, self.relu, self.maxpool = rn.conv1, rn.bn1, rn.relu, rn.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = rn.layer1, rn.layer2, rn.layer3, rn.layer4
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            feat_dim = 2048
            self.backbone = None

        self.proj_head = nn.Linear(feat_dim, out_dim)
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 1024), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, num_classes)
        )
        self._init_heads()

    def _init_heads(self):
        nn.init.kaiming_normal_(self.proj_head.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj_head.bias)
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:
                    nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                    if m.bias is not None: nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)

    def _forward_feat(self, x):
        if self.use_timm:
            f = self.backbone(x)
        else:
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            x = self.avgpool(x); f = torch.flatten(x, 1)
        return f

    def forward(self, x):
        feat = self._forward_feat(x)
        proj = F.normalize(self.proj_head(feat), dim=1)  # contrastive head
        logits = self.cls_head(feat)                     # classification head
        return proj, logits, feat

# =========================
# Train / Validate / Test (한 epoch or 전체)
# =========================
def train_one_epoch(model, loader, optimizer, device, num_classes, beta, gamma, temperature, two_views: bool):
    model.train()
    total_loss = total_con = total_ce = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if two_views:
            v1, v2, labels = batch
            v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)
            images = torch.cat([v1, v2], dim=0)
            labels_dup = torch.cat([labels, labels], dim=0)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            labels_dup = labels

        optimizer.zero_grad(set_to_none=True)
        proj, logits, _ = model(images)

        # contrastive_total = SCL + beta*Mixup-SCL
        loss_con = contrastive_total(proj, labels_dup, num_classes, temperature, beta)

        # CE 손실
        if two_views:
            l1, l2 = torch.chunk(logits, 2, dim=0)
            loss_ce = 0.5 * (F.cross_entropy(l1, labels) + F.cross_entropy(l2, labels))
            bs = labels.size(0)
        else:
            loss_ce = F.cross_entropy(logits, labels)
            bs = labels.size(0)

        loss = loss_con + gamma * loss_ce
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * bs
        total_con  += loss_con.item() * bs
        total_ce   += loss_ce.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{total_loss/max(n,1):.4f}")
    return total_loss / n, total_con / n, total_ce / n

@torch.no_grad()
def validate(model, loader, device, num_classes, beta, gamma, temperature):
    """
    ✅ 검증 손실은 CE만 사용 (contrastive 항 제거)
    """
    model.eval()
    total_loss = total_con = total_ce = 0.0
    correct = n = 0
    pbar = tqdm(loader, desc="Val", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        proj, logits, _ = model(images)

        loss_con = torch.tensor(0.0, device=device)  # contrastive 미사용
        loss_ce  = F.cross_entropy(logits, labels)
        loss = loss_con + gamma * loss_ce

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_con  += loss_con.item() * bs
        total_ce   += loss_ce.item() * bs

        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        n += bs
        pbar.set_postfix(acc=f"{100*correct/max(n,1):.2f}%")
    acc = 100.0 * correct / max(1, n)
    return total_loss / n, total_con / n, total_ce / n, acc

@torch.no_grad()
def test_eval(model, loader, device, class_names: List[str]) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Test", leave=False)
    for x, y in pbar:
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
# Utils: Plot & Save
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
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title(f'Loss - Round {round_num}'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history['val_acc'], 'g-', label='Val Acc (%)')
    ax2.set_title(f'Accuracy - Round {round_num}'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'training_history_round_{round_num}.png', dpi=300)
    plt.close()

class EarlyStopping:
    """
    ✅ 수정: best 가중치를 안전하게 스냅샷(깊은 복사)하여 보존
    """
    def __init__(self, patience=30, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.counter = 0
            # ✅ 각 파라미터를 CPU로 복사 후 clone()하여 안전 저장
            self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

# =========================
# Train one Round
# =========================
def train_single_round(round_num: int, cv_root: Path, test_root: Path, save_root: Path, cfg: Dict) -> Dict:
    print(f"\n{'='*70}\nTRAINING ROUND {round_num}\n{'='*70}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    round_dir = cv_root / f'Round{round_num}'
    train_loader, val_loader, test_loader, num_classes, class_names = make_loaders_for_round(round_dir, test_root, cfg)

    # 모델/옵티마이저
    model = End2EndBackbone(num_classes=num_classes, out_dim=cfg['proj_dim'],
                            backbone=cfg['backbone'], use_timm=cfg['use_timm']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    early_stopping = EarlyStopping(patience=cfg['patience'], min_delta=cfg['min_delta'])
    history = dict(train_loss=[], train_con=[], train_ce=[], val_loss=[], val_con=[], val_ce=[], val_acc=[])

    best_val = float('inf'); best_val_acc = 0.0
    best_path = save_root / f'best_model_round_{round_num}.pth'

    start = time.time()
    for ep in range(1, cfg['epochs']+1):
        tr_loss, tr_con, tr_ce = train_one_epoch(
            model, train_loader, optimizer, device, num_classes,
            cfg['beta'], cfg['gamma'], cfg['temperature'], cfg['two_views']
        )
        # ✅ 검증은 CE만
        va_loss, va_con, va_ce, va_acc = validate(
            model, val_loader, device, num_classes,
            cfg['beta'], cfg['gamma'], cfg['temperature']
        )

        history['train_loss'].append(tr_loss); history['train_con'].append(tr_con); history['train_ce'].append(tr_ce)
        history['val_loss'].append(va_loss); history['val_con'].append(va_con); history['val_ce'].append(va_ce); history['val_acc'].append(va_acc)

        print(f"Epoch {ep:03d} | Train: loss={tr_loss:.4f} (con={tr_con:.4f}, ce={tr_ce:.4f}) | "
              f"Val: loss={va_loss:.4f} (CE-only), acc={va_acc:.2f}%")

        # best 저장 (val loss 기준)
        if best_val - va_loss > cfg['min_delta']:
            best_val = va_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': va_acc,
                'val_loss': va_loss,
                'train_con': tr_con, 'train_ce': tr_ce,
                'config': cfg,
                'num_classes': num_classes,
                'backbone': cfg['backbone'],
            }, best_path)
            print(f"  ✅ Saved best checkpoint to {best_path}")

        # best acc 추적(리포팅용)
        if va_acc > best_val_acc:
            best_val_acc = va_acc

        # Early stopping on val loss (CE-only)
        if early_stopping(va_loss, model):
            print(f"⛳ Early stopping at epoch {ep}")
            break

    train_time = time.time() - start
    print(f"Training time: {train_time/60:.2f} min | Best Val Acc: {best_val_acc:.2f}%")

    # 최종/체크포인트 저장
    final_weights_path = save_root / f'final_weights_round_{round_num}.pth'
    torch.save(model.state_dict(), final_weights_path)
    ckpt_path = save_root / f'checkpoint_round_{round_num}.pth'
    torch.save({
        'round': round_num,
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'training_time': train_time,
        'epochs_trained': len(history['train_loss']),
        'config': cfg,
        'class_names': class_names,
        'backbone': cfg['backbone'],
        'num_classes': num_classes
    }, ckpt_path)

    # ✅ EarlyStopping이 보관한 best weights로 로드하여 테스트
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights, strict=True)
        model.to(device)
    
    # 테스트
    print("Evaluating on Test set...")
    test_res = test_eval(model, test_loader, device, class_names)

    # 결과 저장
    results = {
        'round': round_num,
        'best_val_acc': best_val_acc,
        'training_time_sec': train_time,
        'epochs_trained': len(history['train_loss']),
        'test_accuracy': float(test_res['accuracy']),
        'test_precision': float(test_res['precision']),
        'test_recall': float(test_res['recall']),
        'test_f1': float(test_res['f1']),
        'test_precision_per_class': test_res['precision_per_class'].tolist(),
        'test_recall_per_class': test_res['recall_per_class'].tolist(),
        'test_f1_per_class': test_res['f1_per_class'].tolist(),
        'class_names': class_names
    }

    # 플롯/JSON 저장
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
    cv_root = Path(cv_root); test_root = Path(test_root); save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("5-FOLD CROSS VALIDATION with Proposed End-to-End MM-SCL")
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

    # 요약 통계
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

    # ✅ best overall: 각 라운드의 best 체크포인트를 기준
    best_round = max(all_results, key=lambda r: r['test_accuracy'])
    best_round_num = best_round['round']; best_acc = best_round['test_accuracy']
    print(f"\nBest round: Round {best_round_num} (Acc: {best_acc:.4f})")
    src = save_root / f'best_model_round_{best_round_num}.pth'  # ✅ 수정된 소스
    if src.exists():
        shutil.copy2(src, save_root / 'best_overall_model.pth')
        print("Saved: best_overall_model.pth")

    # 전체 결과 저장
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

    # Paths (modify these according to your setup)
    CV_ROOT = "/home/sj-baek/Manifold_Mixco/LUAD_Dataset_CV"  # Directory containing Round1, Round2, ..., Round5
    TEST_ROOT = "/home/sj-baek/Manifold_Mixco/LUAD_Dataset_CV/test"  # Test data directory
    SAVE_ROOT = "/home/sj-baek/Manifold_Mixco/LUAD_ASAN/MixSCon_cv_results"  # Directory to save results
    run_cross_validation(CV_ROOT, TEST_ROOT, SAVE_ROOT, CONFIG)

    print("\n" + "="*80)
    print("SAVED FILES")
    print("="*80)
    print("1) best_model_round_X.pth       : best checkpoint (with optimizer)")
    print("2) final_weights_round_X.pth    : final model weights only (state_dict)")
    print("3) checkpoint_round_X.pth       : round checkpoint (metadata 포함)")
    print("4) best_overall_model.pth       : best among 5 rounds (using best_model_round_X.pth)")
    print("5) results_round_X.json         : per-round metrics")
    print("6) cross_validation_results.json: CV summary")
    print("7) confusion_matrix_round_X.png / training_history_round_X.png")
