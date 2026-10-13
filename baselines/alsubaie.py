"""
Reimplementation of:
Alsubaie et al. (2023)
Growth Pattern Fingerprinting for Automatic Analysis of Lung Adenocarcinoma Overall Survival

Dataset:
- LUAD_Dataset (Round1~5 + test)
- Each Round contains: train/, val/
- test/ is shared across folds

Classes: 6 subtypes (acinar, papillary, micropapillary, solid, lepidic, others)

Training:
- Epochs = 200
- Early stopping patience = 30 (val_loss 기준)
- Optimizer = Adam, LR=1e-4
- Loss = CrossEntropyLoss
- Metric: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Dice
- Save per-class accuracy, model weights, and metrics for both validation and test
"""
import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
EPOCHS = 200
EARLY_STOP = 30
BATCH_SIZE = 32
LR = 0.0001
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MULTIRES_MODE = 'synthetic'
LOWRES_SCALE = 0.5
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transform(train=True):
    if train:
        return transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    else:
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

class LabeledTileDataset(Dataset):

    def __init__(self, root, transform=None, mode='single', lowres_scale=0.5):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.lowres_scale = lowres_scale
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            for f in os.listdir(cls_path):
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    self.samples.append((os.path.join(cls_path, f), self.class_to_idx[cls]))
        if len(self.samples) == 0:
            raise RuntimeError(f'No images found under {root}. Check path or extensions.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.mode == 'synthetic':
            w, h = img.size
            small = img.resize((max(1, int(w * self.lowres_scale)), max(1, int(h * self.lowres_scale))), Image.BICUBIC)
            pseudo_low = small.resize((w, h), Image.BICUBIC)
            img_hi = self.transform(img) if self.transform else transforms.ToTensor()(img)
            img_lo = self.transform(pseudo_low) if self.transform else transforms.ToTensor()(pseudo_low)
            x = torch.cat([img_hi, img_lo], dim=0)
            return (x, label)
        else:
            x = self.transform(img) if self.transform else transforms.ToTensor()(img)
            return (x, label)

class ResNet50_6ch(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()
        base = models.resnet50(weights='IMAGENET1K_V1')
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        w = base.conv1.weight.data
        self.conv1.weight.data[:, :3, :, :] = w
        self.conv1.weight.data[:, 3:, :, :] = w
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train_model(dataloaders, model, criterion, optimizer, num_epochs=EPOCHS, patience=EARLY_STOP, save_dir='./'):
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, corrects, total = (0.0, 0, 0)
            for x, y in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch + 1}'):
                x, y = (x.to(DEVICE), y.to(DEVICE))
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(x)
                    loss = criterion(out, y)
                    _, pred = torch.max(out, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * x.size(0)
                corrects += torch.sum(pred == y.data).item()
                total += y.size(0)
            epoch_loss = running_loss / total
            epoch_acc = corrects / total
            print(f'{phase} Epoch {epoch + 1}/{num_epochs} Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}')
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        model.load_state_dict(best_state)
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
                        return (model, history)
    model.load_state_dict(best_state)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    return (model, history)

def evaluate(model, dataloader, classes, save_path):
    model.eval()
    preds, labels, probs = ([], [], [])
    with torch.no_grad():
        for x, y in dataloader:
            x, y = (x.to(DEVICE), y.to(DEVICE))
            out = model(x)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(prob.cpu().numpy())
    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()
    try:
        roc = roc_auc_score(labels, probs, multi_class='ovr')
    except:
        roc = None
    acc_per_class = {}
    labels_arr, preds_arr = (np.array(labels), np.array(preds))
    for i, cls in enumerate(classes):
        mask = labels_arr == i
        acc_per_class[cls] = float(np.mean(preds_arr[mask] == labels_arr[mask])) if mask.sum() > 0 else 0.0
    dice_scores = {}
    for i, c in enumerate(classes):
        tp = np.sum((labels_arr == i) & (preds_arr == i))
        fp = np.sum((labels_arr != i) & (preds_arr == i))
        fn = np.sum((labels_arr == i) & (preds_arr != i))
        dice = 2 * tp / (2 * tp + fp + fn + 1e-06)
        dice_scores[c] = dice
    results = {'report': report, 'confusion_matrix': cm, 'roc_auc': roc, 'per_class_accuracy': acc_per_class, 'dice': dice_scores}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    return results
