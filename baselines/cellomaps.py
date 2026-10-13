import os
import json
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
import seaborn as sns
NUM_CLASSES = 6
CLASS_NAMES = ['1_acinar', '2_cribriform', '3_lepidic', '4_micropapillary', '5_papillary', '6_solid']
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-05
PATIENCE = 30
NUM_WORKERS = min(8, max(0, (os.cpu_count() or 1) - 1))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
IMG_SIZE = 448
SAVE_EVERY = 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

class SimpleImageDataset(Dataset):

    def __init__(self, split_dir: str, transform=None):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_path = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_path):
                continue
            patterns = ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.bmp', '**/*.tif', '**/*.tiff']
            for pat in patterns:
                for p in glob.glob(os.path.join(cls_path, pat), recursive=True):
                    if os.path.isfile(p):
                        self.samples.append((p, cls_idx))
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            raise RuntimeError(f'Failed to open image {p}: {e}')
        if self.transform:
            img = self.transform(img)
        return (img, y)

class DilationLayer(nn.Module):

    def __init__(self, kernel_size=5):
        super().__init__()
        k = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)
        self.register_buffer('kernel', k)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel.repeat(C, 1, 1, 1)
        y = F.conv2d(x, k, padding=k.shape[-1] // 2, groups=C)
        y = self.pool(y)
        return y

class CellOMapsResNet50(nn.Module):

    def __init__(self, num_classes=6, use_pretrained=True):
        super().__init__()
        self.dilate = DilationLayer(kernel_size=5)
        try:
            if use_pretrained:
                try:
                    weights = models.ResNet50_Weights.IMAGENET1K_V2
                    backbone = models.resnet50(weights=weights)
                except Exception:
                    backbone = models.resnet50(pretrained=True)
            else:
                backbone = models.resnet50(weights=None)
        except Exception as e:
            print('Warning: failed to load pretrained weights (network/download?). Falling back to random init.', file=sys.stderr)
            backbone = models.resnet50(weights=None)
        self.backbone = backbone
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        x = self.dilate(x)
        return self.backbone(x)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0.7, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', weight=self.alpha)
        pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-06, 1 - 1e-06)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class EarlyStopping:

    def __init__(self, patience=30, mode='min', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.best is None:
            self.best = value
            return True
        if self.mode == 'min' and value < self.best - self.delta or (self.mode == 'max' and value > self.best + self.delta):
            self.best = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

def get_transforms(img_size=448, is_train=True):
    if is_train:
        return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()])
    else:
        return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

def ovr_macro_auc(y_true, y_prob):
    try:
        y_bin = np.eye(NUM_CLASSES)[y_true]
        return float(roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr'))
    except Exception:
        return float('nan')

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses, ys, ps = ([], [], [])
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        ys.append(y.detach().cpu().numpy())
        ps.append(logits.detach().softmax(1).cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    y_pred = y_prob.argmax(1) if y_prob.size else np.array([])
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average='macro')) if y_true.size else 0.0
    macro_prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0)) if y_true.size else 0.0
    macro_rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0)) if y_true.size else 0.0
    macro_auc = ovr_macro_auc(y_true, y_prob) if y_true.size else 0.0
    return (float(np.mean(losses)) if losses else 0.0, acc, macro_f1, macro_prec, macro_rec, macro_auc, y_true, y_pred, y_prob)

def plot_history(history, out_png):
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_confmat(y_true, y_pred, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    try:
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
    except Exception:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

def sanity_count(split_dir: str) -> int:
    n = 0
    if not os.path.isdir(split_dir):
        return 0
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        patterns = ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.bmp', '**/*.tif', '**/*.tiff']
        for pat in patterns:
            n += len(glob.glob(os.path.join(cls_dir, pat), recursive=True))
    return n

def train_one_fold(DATA_ROOT, run_dir, ckpt_dir, fold_idx, use_pretrained_weights=False, test_root=None):
    fold_dir = DATA_ROOT
    train_dir = os.path.join(fold_dir, 'train')
    val_dir = os.path.join(fold_dir, 'val')
    test_dir = str(test_root) if test_root is not None else os.path.join(os.path.dirname(DATA_ROOT), 'test')
    os.makedirs(ckpt_dir, exist_ok=True)
    n_tr, n_va, n_te = (sanity_count(train_dir), sanity_count(val_dir), sanity_count(test_dir))
    print(f'[Fold {fold_idx}] counts: train={n_tr}, val={n_va}, test={n_te}')
    if n_tr == 0 or n_va == 0 or n_te == 0:
        raise RuntimeError(f'[Fold {fold_idx}] 데이터가 비어 있습니다.')
    train_ds = SimpleImageDataset(train_dir, transform=get_transforms(IMG_SIZE, True))
    val_ds = SimpleImageDataset(val_dir, transform=get_transforms(IMG_SIZE, False))
    test_ds = SimpleImageDataset(test_dir, transform=get_transforms(IMG_SIZE, False))
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    model = CellOMapsResNet50(num_classes=NUM_CLASSES, use_pretrained=use_pretrained_weights).to(DEVICE)
    criterion = FocalLoss(gamma=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early = EarlyStopping(patience=PATIENCE, mode='min')
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1', 'train_prec', 'val_prec', 'train_rec', 'val_rec', 'train_auc', 'val_auc']}
    best_path = os.path.join(ckpt_dir, 'best_val_loss.pth')
    print(f'[Fold {fold_idx}] Training started...')
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, tr_prec, tr_rec, tr_auc, *_ = run_epoch(model, train_ld, criterion, optimizer)
        va_loss, va_acc, va_f1, va_prec, va_rec, va_auc, *_ = run_epoch(model, val_ld, criterion, optimizer=None)
        scheduler.step()
        for k, v in zip(['train_loss', 'train_acc', 'train_f1', 'train_prec', 'train_rec', 'train_auc', 'val_loss', 'val_acc', 'val_f1', 'val_prec', 'val_rec', 'val_auc'], [tr_loss, tr_acc, tr_f1, tr_prec, tr_rec, tr_auc, va_loss, va_acc, va_f1, va_prec, va_rec, va_auc]):
            history[k].append(v)
        if early.step(va_loss):
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()}, best_path)
        if early.should_stop:
            print(f'[Fold {fold_idx}] Early stopped at epoch {epoch}')
            break
    plot_history(history, os.path.join(run_dir, 'history.png'))
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump({'epochs_run': len(history['train_loss']), 'best_epoch': int(torch.load(best_path)['epoch']), 'history': history}, f, indent=2)
    return best_path
import pandas as pd
from sklearn.preprocessing import label_binarize

def evaluate_model(model, test_ld, criterion, run_dir, ckpt_path, prefix='test'):
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state['model'])
    model.eval()
    te_loss, te_acc, te_f1, te_prec, te_rec, te_auc, y_true, y_pred, y_prob = run_epoch(model, test_ld, criterion, optimizer=None)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    try:
        y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        aucs = []
        for i in range(NUM_CLASSES):
            if np.sum(y_bin[:, i]) > 0:
                aucs.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
            else:
                aucs.append(np.nan)
    except Exception:
        aucs = [np.nan] * NUM_CLASSES
    df_metrics = pd.DataFrame({'Class': CLASS_NAMES, 'Precision': precision, 'Recall': recall, 'F1-score': f1, 'AUC': aucs})
    df_metrics.to_csv(os.path.join(run_dir, f'{prefix}_per_class_metrics.csv'), index=False)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(run_dir, f'{prefix}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f'Test Results: acc={te_acc:.4f}, macro_f1={te_f1:.4f}, prec={te_prec:.4f}, rec={te_rec:.4f}, macro_auc_ovr={te_auc:.4f}')
    return {'test_loss': te_loss, 'test_acc': te_acc, 'test_macro_f1': te_f1, 'test_macro_prec': te_prec, 'test_macro_rec': te_rec, 'test_macro_auc_ovr': te_auc}

def data_loader(data_dir):
    data_ds = SimpleImageDataset(data_dir, transform=get_transforms(IMG_SIZE, False))
    data_ld = DataLoader(data_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return data_ld
