# run_fair_ablation_study_cv_v2.py
# =====================================================================================
# Ablation Study: (Input) Mixup / Manifold(Embedding) Mixup + Supervised Contrastive Learning
#
# 6 experiments:
#   1) Baseline
#   2) Baseline + Input Mixup            (이미지 레벨 mixup)
#   3) Baseline + Manifold Mixup         (임베딩 레벨 mixup)
#   4) Baseline + SCL
#   5) Baseline + Input Mixup + SCL      (이미지 레벨 mixup + SCL)
#   6) Baseline + Manifold Mixup + SCL   (임베딩 레벨 mixup + SCL)
# =====================================================================================

import os
import argparse
from pathlib import Path
from mixscon.data import CLASS_FOLDERS, map_labels, check_folders
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, auc, roc_curve
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# =====================================================================================
# 전역 하이퍼파라미터 (6개 실험, encoder/classifier 단계 모두 동일하게 적용)
# =====================================================================================
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
DO_TSNE = True
EPOCHS = 200
PATIENCE = 30
MIN_DELTA = 1e-4
EMBED_DIM = 128
LR = 1e-3
TEMPERATURE = 0.07
MIXUP_ALPHA_INPUT = 1.0       # 표준 Mixup 논문 권장값 (Zhang et al., 2018)
MIXUP_ALPHA_MANIFOLD = 0.4    # Manifold Mixup 논문 권장값 (Verma et al., 2019)
BETA = 1.0                    # loss = SCL + beta * mixup-SCL
MIXUP_LAYER = 3


# =====================================================================================
# 공통 유틸리티
# =====================================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imagenet_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_data(data_dir, test_dir, batch_size=BATCH_SIZE):

    transform = imagenet_transform()
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_dataset = map_labels(train_dataset)
    val_dataset = map_labels(val_dataset)
    test_dataset = map_labels(test_dataset)

    # drop_last=True: 마지막 미니배치가 너무 작아 BatchNorm/SCL positive-mask 계산이
    # 불안정해지는 것을 방지 (6개 실험 모두 동일 적용). test_loader는 평가용이라 미적용.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


    return train_loader, val_loader, test_loader, len(train_dataset.classes), train_dataset.classes


# =====================================================================================
# Mixup 유틸리티
# =====================================================================================

def mixup_data(x, y, alpha=MIXUP_ALPHA_INPUT):
    """입력(이미지) 레벨 표준 Mixup"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def manifold_mixup(features, labels, num_classes, alpha=MIXUP_ALPHA_MANIFOLD):
    """임베딩(feature) 레벨 Manifold Mixup. mixed_labels는 soft one-hot."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    mixed_features = lam * features + (1 - lam) * features[index, :]
    mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index, :]
    return mixed_features, mixed_labels, lam, index


def soft_cross_entropy(logits, soft_targets):
    """soft label(one-hot 보간)에 대한 cross entropy.
    manifold_mixup이 실제로 만든 mixed_labels를 그대로 사용하므로
    forward에서 쓰인 lam/index와 loss 계산이 항상 정확히 일치한다."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# =====================================================================================
# Contrastive Loss (정규화 + 자기 자신 제외 + eps 안정화)
# =====================================================================================

def supervised_contrastive_loss(features, labels, temperature=TEMPERATURE, eps=1e-12):
    feats = F.normalize(features, dim=1)
    sim = torch.matmul(feats, feats.T) / temperature
    log_prob = F.log_softmax(sim, dim=1)

    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    eye = torch.eye(mask.size(0), device=mask.device)
    mask = mask * (1 - eye)  # 자기 자신과의 positive 쌍 제외

    denom = mask.sum(dim=1).clamp(min=1)
    loss = -torch.sum(mask * log_prob, dim=1) / (denom + eps)
    return loss.mean()


def embedding_mixup_scl_loss(mixed_features, mixed_labels_oh, base_features, labels_oh,
                              temperature=TEMPERATURE, eps=1e-12):

    mixed_n = F.normalize(mixed_features, dim=1)
    base_n = F.normalize(base_features, dim=1)

    sim = torch.matmul(mixed_n, base_n.T) / temperature
    log_prob = F.log_softmax(sim, dim=1)

    pos_mask = torch.matmul(mixed_labels_oh, labels_oh.T)  # soft positive mask
    eye = torch.eye(pos_mask.size(0), device=pos_mask.device)
    pos_mask = pos_mask * (1 - eye)

    denom = pos_mask.sum(dim=1).clamp(min=1)
    loss = -torch.sum(pos_mask * log_prob, dim=1) / (denom + eps)
    return loss.mean()


# =====================================================================================
# 모델 정의 (모든 실험이 공유하는 단일 Encoder)
# =====================================================================================

class Encoder(nn.Module):


    def __init__(self, out_dim=EMBED_DIM, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, out_dim)

    def forward(self, x, apply_mixup_at_layer=None, labels=None, num_classes=None,
                mixup_alpha=MIXUP_ALPHA_MANIFOLD):

        mix_info = None

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if apply_mixup_at_layer == 1:
            x, mixed_labels, lam, index = manifold_mixup(x, labels, num_classes, alpha=mixup_alpha)
            mix_info = (mixed_labels, lam, index)

        x = self.layer2(x)
        if apply_mixup_at_layer == 2:
            x, mixed_labels, lam, index = manifold_mixup(x, labels, num_classes, alpha=mixup_alpha)
            mix_info = (mixed_labels, lam, index)

        x = self.layer3(x)
        if apply_mixup_at_layer == 3:
            x, mixed_labels, lam, index = manifold_mixup(x, labels, num_classes, alpha=mixup_alpha)
            mix_info = (mixed_labels, lam, index)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x, mix_info  # (B, out_dim) 임베딩, mix_info


class Classifier(nn.Module):
    """공통 분류기 (Stage 2에서 frozen encoder 위에 새로 학습)"""

    def __init__(self, feature_dim=EMBED_DIM, num_classes=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def initialize_weights(model):
    """Xavier 초기화 (encoder Stage-1 임시 head, Stage-2 Classifier 공통 적용)"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# =====================================================================================
# Encoder 학습 함수 (Stage 1) - 6개 모두 동일한 시그니처
# =====================================================================================

def _encoder_loop_finalize(model, epoch, epochs, avg_loss, best_loss, patience_counter,
                            min_delta, patience, save_path, save_fn):
    """early-stopping/체크포인트 공통 로직"""
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    stop = False
    if best_loss - avg_loss > min_delta:
        best_loss = avg_loss
        patience_counter = 0
        save_fn()
        print(f"[INFO] Best encoder saved: {save_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            stop = True
    return best_loss, patience_counter, stop


def train_baseline_encoder(model, train_loader, epochs, device, num_classes, lr,
                            patience, min_delta, save_path, **kwargs):
    """1. Baseline: CE. Stage-1 전용 임시 head로 128-dim 임베딩 차원 버그 해결."""
    model.train()
    head = nn.Linear(EMBED_DIM, num_classes).to(device)
    initialize_weights(head)
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            feat, _ = model(images)
            logits = head(feat)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best_loss, patience_counter, stop = _encoder_loop_finalize(
            model, epoch, epochs, avg_loss, best_loss, patience_counter, min_delta,
            patience, save_path, lambda: torch.save(model.state_dict(), save_path))
        if stop:
            break


def train_mixup_encoder(model, train_loader, epochs, device, num_classes, lr,
                         patience, min_delta, save_path, mixup_alpha_input=MIXUP_ALPHA_INPUT,
                         **kwargs):
    """2. Baseline + Input Mixup (이미지 레벨)"""
    model.train()
    head = nn.Linear(EMBED_DIM, num_classes).to(device)
    initialize_weights(head)
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha_input)
            feat, _ = model(mixed_images)
            logits = head(feat)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best_loss, patience_counter, stop = _encoder_loop_finalize(
            model, epoch, epochs, avg_loss, best_loss, patience_counter, min_delta,
            patience, save_path, lambda: torch.save(model.state_dict(), save_path))
        if stop:
            break


def train_manifold_mixup_encoder(model, train_loader, epochs, device, num_classes, lr,
                                  patience, min_delta, save_path,
                                  mixup_alpha_manifold=MIXUP_ALPHA_MANIFOLD,
                                  mixup_layer=MIXUP_LAYER, **kwargs):
    """3. Baseline + Manifold Mixup (중간 레이어에서 섞고 나머지 레이어를 통과).
    forward가 실제로 사용한 lam/mixed_labels를 그대로 loss에 사용하므로
    v1의 라벨-람다 불일치 버그가 재발하지 않는다."""
    model.train()
    head = nn.Linear(EMBED_DIM, num_classes).to(device)
    initialize_weights(head)
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            feat, mix_info = model(images, apply_mixup_at_layer=mixup_layer, labels=labels,
                                    num_classes=num_classes, mixup_alpha=mixup_alpha_manifold)
            mixed_labels, lam, index = mix_info
            logits = head(feat)
            loss = soft_cross_entropy(logits, mixed_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best_loss, patience_counter, stop = _encoder_loop_finalize(
            model, epoch, epochs, avg_loss, best_loss, patience_counter, min_delta,
            patience, save_path, lambda: torch.save(model.state_dict(), save_path))
        if stop:
            break


def train_scl_encoder(model, train_loader, epochs, device, num_classes, lr,
                       patience, min_delta, save_path, temperature=TEMPERATURE, **kwargs):
    """4. Baseline + SCL"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            feat, _ = model(images)
            loss = supervised_contrastive_loss(feat, labels, temperature)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best_loss, patience_counter, stop = _encoder_loop_finalize(
            model, epoch, epochs, avg_loss, best_loss, patience_counter, min_delta,
            patience, save_path, lambda: torch.save(model.state_dict(), save_path))
        if stop:
            break


def train_mixup_scl_encoder(model, train_loader, epochs, device, num_classes, lr,
                             patience, min_delta, save_path, beta=BETA,
                             temperature=TEMPERATURE, mixup_alpha_input=MIXUP_ALPHA_INPUT,
                             **kwargs):
    """5. Baseline + Input Mixup + SCL (이미지 레벨 mixup + SCL).
    clean forward 1회 + mixed-image forward 1회 (총 2회 forward)."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            base_feat, _ = model(images)  # clean forward
            scl_loss = supervised_contrastive_loss(base_feat, labels, temperature)

            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha_input)
            mixed_feat, _ = model(mixed_images)  # mixed-image forward

            labels_oh = F.one_hot(labels, num_classes=num_classes).float()
            mixed_labels_oh = (lam * F.one_hot(y_a, num_classes=num_classes).float()
                                + (1 - lam) * F.one_hot(y_b, num_classes=num_classes).float())

            mixup_scl_loss = embedding_mixup_scl_loss(
                mixed_feat, mixed_labels_oh, base_feat, labels_oh, temperature)

            loss = scl_loss + beta * mixup_scl_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best_loss, patience_counter, stop = _encoder_loop_finalize(
            model, epoch, epochs, avg_loss, best_loss, patience_counter, min_delta,
            patience, save_path, lambda: torch.save(model.state_dict(), save_path))
        if stop:
            break


def train_manifold_mixup_scl_encoder(model, train_loader, epochs, device, num_classes, lr,
                                      patience, min_delta, save_path, beta=BETA,
                                      temperature=TEMPERATURE,
                                      mixup_alpha_manifold=MIXUP_ALPHA_MANIFOLD,
                                      mixup_layer=MIXUP_LAYER, **kwargs):
    """6. Baseline + Manifold Mixup + SCL (중간 레이어에서 섞고 나머지 레이어를 통과).
    exp3와 동일한 mixup_layer를 공유해야 "같은 Manifold Mixup" 정의로 비교된다.
    중간 레이어 mixing은 비선형을 통과하므로 embedding-level처럼 대수적으로 축약할 수
    없어, clean forward 1회 + mixed forward 1회 (총 2회, exp5와 동일 비용)가 필요하다."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            base_feat, _ = model(images)  # clean forward
            scl_loss = supervised_contrastive_loss(base_feat, labels, temperature)

            mixed_feat, mix_info = model(images, apply_mixup_at_layer=mixup_layer, labels=labels,
                                          num_classes=num_classes, mixup_alpha=mixup_alpha_manifold)
            mixed_labels_oh, lam, index = mix_info
            labels_oh = F.one_hot(labels, num_classes=num_classes).float()

            mixup_scl_loss = embedding_mixup_scl_loss(
                mixed_feat, mixed_labels_oh, base_feat, labels_oh, temperature)

            loss = scl_loss + beta * mixup_scl_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best_loss, patience_counter, stop = _encoder_loop_finalize(
            model, epoch, epochs, avg_loss, best_loss, patience_counter, min_delta,
            patience, save_path, lambda: torch.save(model.state_dict(), save_path))
        if stop:
            break


# =====================================================================================
# Classifier 학습 (Stage 2) - encoder는 항상 frozen, 6개 실험 공통
# =====================================================================================

def train_classifier(encoder, classifier, train_loader, val_loader, device, epochs, lr,
                      patience, save_path):
    encoder.eval()
    classifier.train()

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = float('-inf')
    patience_counter = 0

    for epoch in range(epochs):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                feat, _ = encoder(images)
            outputs = classifier(feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / max(1, total)

        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                feat, _ = encoder(images)
                outputs = classifier(feat)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(1, val_total)

        print(f"Epoch [{epoch + 1}/{epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(classifier.state_dict(), save_path)
            print(f"[INFO] Best classifier saved: {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_val_acc


# =====================================================================================
# 평가 / Feature 추출 / t-SNE
# =====================================================================================

def evaluate_model(encoder, classifier, data_loader, device, class_names, save_dir, prefix="test"):
    os.makedirs(save_dir, exist_ok=True)

    encoder.eval()
    classifier.to(device)
    classifier.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            feats, _ = encoder(images)
            outputs = classifier(feats)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"[INFO] {prefix.capitalize()} Accuracy: {overall_acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    for cname, cacc in zip(class_names, per_class_acc):
        print(f"{prefix.capitalize()} {cname} Accuracy: {cacc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=class_names,
                                    digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(save_dir, f"{prefix}_classification_report.csv")
    report_df.to_csv(report_path)
    print(f"[INFO] Classification report saved to {report_path}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{prefix.capitalize()} Confusion Matrix")
    cm_path = os.path.join(save_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    y_true_bin = label_binarize(all_labels, classes=range(len(class_names)))
    per_class_auc = {}
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', len(class_names))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        auc_score = auc(fpr, tpr)
        per_class_auc[cname] = auc_score
        print(f"{prefix.capitalize()} {cname} AUC: {auc_score:.4f}")
        plt.plot(fpr, tpr, color=colors(i), lw=2, label=f"{cname} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix.capitalize()} ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_all_path = os.path.join(save_dir, f"{prefix}_roc_all_classes.png")
    plt.savefig(roc_all_path, dpi=300)
    plt.close()
    print(f"[INFO] Combined ROC curves saved to {roc_all_path}")

    metrics_data = {
        "Metric": ["Overall Accuracy"] + [f"{c} Accuracy" for c in class_names]
                  + [f"{c} AUC" for c in class_names],
        "Value": [overall_acc] + list(per_class_acc) + [per_class_auc[c] for c in class_names],
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(save_dir, f"{prefix}_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Metrics summary saved to {metrics_path}")

    return overall_acc, report


def extract_features(encoder, dataloader, device):
    encoder.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            feats, _ = encoder(images)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(features), np.hstack(labels)


def plot_tsne(features, labels, class_names, save_path, title="t-SNE Visualization"):
    n = features.shape[0]
    if n < 2:
        raise ValueError('t-SNE requires at least two tiles; use --skip-tsne.')
    perplexity = int(min(n - 1, max(5, min(30, (n - 1) // 3))))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED, init="pca")
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.rcParams["axes.facecolor"] = "white"
    cmap = plt.cm.get_cmap("tab10", len(class_names))
    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=[cmap(i)],
                    label=class_name, alpha=0.85, edgecolor="k", linewidth=0.3, s=50)
    plt.title(title, fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"[INFO] t-SNE plot saved: {save_path}")
    plt.close()


# =====================================================================================
# 단일 실험 실행 (5 round x 6 실험 = 30회 모두 이 함수를 통과)
# =====================================================================================

def run_experiment(exp_name, train_func, data_dir, test_dir, save_dir, device, num_classes,
                    class_names, beta=BETA):

    set_seed(SEED)

    print(f"\n{'=' * 50}")
    print(f"Running Experiment: {exp_name} (beta={beta})")
    print(f"{'=' * 50}")

    models_dir = os.path.join(save_dir, "models")
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    safe_name = exp_name.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "").replace(".", "")
    encoder_path = os.path.join(models_dir, f"{safe_name}_encoder.pth")
    classifier_path = os.path.join(models_dir, f"{safe_name}_classifier.pth")
    tsne_path = os.path.join(results_dir, f"{safe_name}_tsne.png")
    if os.path.exists(encoder_path) or os.path.exists(classifier_path):
        raise FileExistsError('Choose a fresh output directory to preserve checkpoints.')

    train_loader, val_loader, test_loader, _, _ = load_data(data_dir, test_dir, batch_size=BATCH_SIZE)

    # ---- Stage 1: Encoder 학습 ----
    print(f"\n--- Stage 1: Training {exp_name} Encoder ---")
    encoder = Encoder(out_dim=EMBED_DIM).to(device)
    train_func(model=encoder, train_loader=train_loader, epochs=EPOCHS, device=device,
               num_classes=num_classes, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA,
               save_path=encoder_path, beta=beta, temperature=TEMPERATURE,
               mixup_alpha_input=MIXUP_ALPHA_INPUT, mixup_alpha_manifold=MIXUP_ALPHA_MANIFOLD,
               mixup_layer=MIXUP_LAYER)

    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        print(f"[INFO] Loaded best encoder: {encoder_path}")
    else:
        print(f"[WARNING] Encoder checkpoint not found at {encoder_path} (continuing with current weights)")

    # ---- Stage 2: Classifier 학습 (encoder frozen, 6개 실험 공통) ----
    print(f"\n--- Stage 2: Training Classifier for {exp_name} ---")
    classifier = Classifier(feature_dim=EMBED_DIM, num_classes=num_classes).to(device)
    initialize_weights(classifier)

    for p in encoder.parameters():
        p.requires_grad = False

    best_val_acc = train_classifier(encoder, classifier, train_loader, val_loader, device,
                                     epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                     save_path=classifier_path)

    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
        print(f"[INFO] Loaded best classifier: {classifier_path}")
    else:
        print(f"[WARNING] Classifier checkpoint not found at {classifier_path} (continuing with current weights)")

    # ---- Stage 3: 최종 평가 ----
    print(f"\n--- Stage 3: Final Evaluation for {exp_name} ---")
    overall_acc, report = evaluate_model(encoder, classifier, test_loader, device, class_names,
                                          results_dir, prefix=f"test-{exp_name}")

    # ---- Stage 4: t-SNE ----
    print(f"\n--- Stage 4: t-SNE Visualization for {exp_name} ---")
    if DO_TSNE:
        features, labels = extract_features(encoder, test_loader, device)
        plot_tsne(features, labels, class_names, tsne_path, title=f"t-SNE: {exp_name}")
    else:
        tsne_path = None

    result = {
        'experiment': exp_name,
        'beta': beta,
        'encoder_path': encoder_path,
        'classifier_path': classifier_path,
        'tsne_path': tsne_path,
        'best_val_acc': best_val_acc,
        'test_acc': overall_acc,
        'test_report': report,
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n[INFO] {exp_name} Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {overall_acc:.4f}")

    return result


# =====================================================================================
# 메인
# =====================================================================================

EXPERIMENTS = [
    ("1. Baseline (ResNet50)", train_baseline_encoder),
    ("2. Baseline + Input Mixup", train_mixup_encoder),
    ("3. Baseline + Manifold Mixup", train_manifold_mixup_encoder),
    ("4. Baseline + SCL", train_scl_encoder),
    ("5. Baseline + Input Mixup + SCL", train_mixup_scl_encoder),
    ("6. Baseline + Manifold Mixup + SCL", train_manifold_mixup_scl_encoder),
]

def main():
    global EPOCHS, BATCH_SIZE, NUM_WORKERS, SEED, DO_TSNE
    parser = argparse.ArgumentParser(description='Train the six two-stage ablation configurations.')
    parser.add_argument('--data-root', type=Path, required=True, help='Contains Round1 through Round5.')
    parser.add_argument('--test-root', type=Path, help='Default: DATA_ROOT/test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--folds', type=int, nargs='+', choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument('--experiments', type=int, nargs='+', choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-tsne', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0:
        parser.error('Invalid epoch, batch size or worker count.')
    if len(set(args.folds)) != len(args.folds) or len(set(args.experiments)) != len(args.experiments):
        parser.error('Duplicate folds or experiments.')
    test_root = args.test_root or args.data_root / 'test'
    counts = check_folders(args.data_root, test_root, args.folds, args.batch_size)
    if args.dry_run:
        print(json.dumps(counts, indent=2))
        return
    EPOCHS, BATCH_SIZE, NUM_WORKERS, SEED = args.epochs, args.batch_size, args.num_workers, args.seed
    DO_TSNE = not args.skip_tsne
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'run_config.json').write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2),
        encoding='utf-8')
    for fold in args.folds:
        save_dir = args.output_dir / f'Round{fold}'
        results = []
        for index in args.experiments:
            name, train_func = EXPERIMENTS[index - 1]
            results.append(run_experiment(name, train_func, args.data_root / f'Round{fold}',
                                          test_root, save_dir, device, len(CLASS_FOLDERS), CLASS_FOLDERS))
        (save_dir / 'results/ablation_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
