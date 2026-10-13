NUM_WORKERS = 4
import os
import random
import json
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
EPOCHS_SSL = 50
EPOCHS_DOWNSTREAM = 200
BATCH_SIZE = 32
LR = 0.0001
EARLY_STOP = 30
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transform(train=True):
    if train:
        return transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class LowHighPairDataset(Dataset):
    """SSL1: 저배율-고배율 (resize 대체)"""

    def __init__(self, img_folder, transform=None, scale_factor=2):
        self.paths = []
        for cls in os.listdir(img_folder):
            cls_path = os.path.join(img_folder, cls)
            for f in os.listdir(cls_path):
                self.paths.append(os.path.join(cls_path, f))
        self.transform = transform
        self.sf = scale_factor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        w, h = img.size
        high = img
        low = img.resize((w // self.sf, h // self.sf), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if self.transform:
            return (self.transform(low), self.transform(high))
        return (low, high)

class SpatialRelationDataset(Dataset):
    """SSL2: 공간적 관계 예측 (quadrant shuffle 대체)"""

    def __init__(self, img_folder, transform=None):
        self.paths = []
        for cls in os.listdir(img_folder):
            cls_path = os.path.join(img_folder, cls)
            for f in os.listdir(cls_path):
                self.paths.append(os.path.join(cls_path, f))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB').resize((224, 224))
        arr = np.array(img)
        h, w, _ = arr.shape
        quadrants = [arr[:h // 2, :w // 2], arr[:h // 2, w // 2:], arr[h // 2:, :w // 2], arr[h // 2:, w // 2:]]
        label = random.randint(0, 3)
        target = quadrants[label]
        if self.transform:
            return (self.transform(Image.fromarray(target)), label)
        return (target, label)

class StainSeparationDataset(Dataset):
    """SSL3: Stain separation (color deconvolution 기반 단순화)"""

    def __init__(self, img_folder, transform=None):
        self.paths = []
        for cls in os.listdir(img_folder):
            cls_path = os.path.join(img_folder, cls)
            for f in os.listdir(cls_path):
                self.paths.append(os.path.join(cls_path, f))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB').resize((224, 224))
        arr = np.array(img)
        h = arr[:, :, 0]
        e = arr[:, :, 1]
        h_img = Image.fromarray(h).convert('RGB')
        e_img = Image.fromarray(e).convert('RGB')
        if self.transform:
            return (self.transform(h_img), self.transform(e_img))
        return (h_img, e_img)

class EncoderDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.decoder = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 224 * 224 * 3), nn.Sigmoid())

    def forward(self, x):
        f = self.encoder(x).view(x.size(0), -1)
        out = self.decoder(f).view(-1, 3, 224, 224)
        return out

def train_ssl(dataset, save_path, epochs=EPOCHS_SSL, bs=BATCH_SIZE, lr=LR, task='ssl1'):
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)
    if task == 'ssl2':
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 4)
        criterion = nn.CrossEntropyLoss()
    else:
        model = EncoderDecoder()
        criterion = nn.MSELoss()
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)
    best = float('inf')
    counter = 0
    for e in range(epochs):
        model.train()
        total = 0
        for data in loader:
            if task == 'ssl1':
                low, high = data
                low, high = (low.to(DEVICE), high.to(DEVICE))
                pred = model(low)
                loss = criterion(pred, high)
                bs_curr = low.size(0)
            elif task == 'ssl3':
                h, e_img = data
                h, e_img = (h.to(DEVICE), e_img.to(DEVICE))
                pred = model(h)
                loss = criterion(pred, e_img)
                bs_curr = h.size(0)
            else:
                imgs, lbl = data
                imgs, lbl = (imgs.to(DEVICE), lbl.to(DEVICE))
                pred = model(imgs)
                loss = criterion(pred, lbl)
                bs_curr = imgs.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * bs_curr
        avg = total / len(dataset)
        print(f'[{task}] Epoch {e + 1}/{epochs}, Loss={avg:.4f}')
        if avg < best:
            best = avg
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
        if counter >= EARLY_STOP:
            break
    return save_path

def train_downstream(fold_dir, test_dir, ssl_weights, save_dir, epochs=EPOCHS_DOWNSTREAM, patience=EARLY_STOP, bs=BATCH_SIZE, lr=LR):
    dataloaders = {'train': DataLoader(datasets.ImageFolder(os.path.join(fold_dir, 'train'), transform=get_transform(True)), batch_size=bs, shuffle=True, num_workers=NUM_WORKERS), 'val': DataLoader(datasets.ImageFolder(os.path.join(fold_dir, 'val'), transform=get_transform(False)), batch_size=bs, shuffle=False, num_workers=NUM_WORKERS), 'test': DataLoader(datasets.ImageFolder(test_dir, transform=get_transform(False)), batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)}
    classes = dataloaders['train'].dataset.classes
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    if ssl_weights:
        state_dict = torch.load(ssl_weights, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)
    best = float('inf')
    patience_c = 0
    best_state = None
    stop_flag = False
    for e in range(epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running = 0.0
            for x, y in dataloaders[phase]:
                x, y = (x.to(DEVICE), y.to(DEVICE))
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(x)
                    loss = criterion(out, y)
                    _, pred = torch.max(out, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running += loss.item() * x.size(0)
            epoch_loss = running / len(dataloaders[phase].dataset)
            print(f'[{phase}] Epoch {e + 1}, Loss={epoch_loss:.4f}')
            if phase == 'val':
                if epoch_loss < best:
                    best = epoch_loss
                    best_state = model.state_dict()
                    patience_c = 0
                else:
                    patience_c += 1
                    if patience_c >= patience:
                        model.load_state_dict(best_state)
                        print(f'Early stopping at epoch {e + 1}')
                        stop_flag = True
                        break
        if stop_flag:
            break

    def eval_split(split):
        model.eval()
        preds = []
        labels = []
        probs = []
        with torch.no_grad():
            for x, y in dataloaders[split]:
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
        acc_pc = {cls: float(np.mean(np.array(preds)[np.array(labels) == i] == i)) for i, cls in enumerate(classes)}
        return {'report': report, 'confusion_matrix': cm, 'roc_auc': roc, 'accuracy_per_class': acc_pc}
    results = {split: eval_split(split) for split in ['val', 'test']}
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    return results

def ensemble_predict(models_paths, test_dir, class_names, save_path, weights=None):
    loader = DataLoader(datasets.ImageFolder(test_dir, transform=get_transform(False)), batch_size=32, shuffle=False)
    models_list = []
    for p in models_paths:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(class_names))
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m.to(DEVICE)
        m.eval()
        models_list.append(m)
    preds = []
    labels = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = (x.to(DEVICE), y.to(DEVICE))
            out_sum = 0
            for i, m in enumerate(models_list):
                w = 1.0 if weights is None else weights[i]
                out_sum += w * torch.softmax(m(x), dim=1)
            avg_prob = out_sum / (sum(weights) if weights else len(models_list))
            _, pred = torch.max(avg_prob, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(avg_prob.cpu().numpy())
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()
    try:
        roc = roc_auc_score(labels, probs, multi_class='ovr')
    except:
        roc = None
    acc_pc = {cls: float(np.mean(np.array(preds)[np.array(labels) == i] == i)) for i, cls in enumerate(class_names)}
    res = {'report': report, 'confusion_matrix': cm, 'roc_auc': roc, 'accuracy_per_class': acc_pc}
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'ensemble_metrics.json'), 'w') as f:
        json.dump(res, f, indent=4)
    return res

def ensemble_predict_both(models_paths, dataloaders, class_names, save_path, weights=None):
    models_list = []
    for p in models_paths:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(class_names))
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m.to(DEVICE)
        m.eval()
        models_list.append(m)

    def eval_loader(loader):
        preds, labels, probs = ([], [], [])
        with torch.no_grad():
            for x, y in loader:
                x, y = (x.to(DEVICE), y.to(DEVICE))
                out_sum = 0
                for i, m in enumerate(models_list):
                    w = 1.0 if weights is None else weights[i]
                    out_sum += w * torch.softmax(m(x), dim=1)
                avg_prob = out_sum / (sum(weights) if weights else len(models_list))
                _, pred = torch.max(avg_prob, 1)
                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())
                probs.extend(avg_prob.cpu().numpy())
        report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds).tolist()
        try:
            roc = roc_auc_score(labels, probs, multi_class='ovr')
        except:
            roc = None
        acc_pc = {cls: float(np.mean(np.array(preds)[np.array(labels) == i] == i)) for i, cls in enumerate(class_names)}
        return {'report': report, 'confusion_matrix': cm, 'roc_auc': roc, 'accuracy_per_class': acc_pc}
    results = {'val': eval_loader(dataloaders['val']), 'test': eval_loader(dataloaders['test'])}
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'ensemble_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    return results
