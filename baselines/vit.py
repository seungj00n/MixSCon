import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
import shutil
from pathlib import Path

class LUADDataset(Dataset):
    """LUAD Dataset for loading histopathology images"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = ['1_acinar', '2_cribriform', '3_lepidic', '4_micropapillary', '5_papillary', '6_solid']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return (image, label)

class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience=30, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

def create_transforms(use_augmentation=True):
    """Create data transforms for training and validation/test"""
    if use_augmentation:
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(degrees=90), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return (train_transform, val_test_transform)

def create_model(num_classes=6):
    """Create ViT-Base-Patch16-224 model using timm"""
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from saved checkpoint

    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the model on

    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary with training info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 6)
    model_name = checkpoint.get('model_name', 'vit_base_patch16_224')
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f'Model loaded from {checkpoint_path}')
    print(f"Round: {checkpoint.get('round', 'Unknown')}")
    print(f"Best Val Accuracy: {checkpoint.get('best_val_acc', 'Unknown')}")
    print(f"Training Time: {checkpoint.get('training_time', 'Unknown')} seconds")
    return (model, checkpoint)

def load_model_weights_only(weights_path, num_classes=6, device='cpu'):
    """
    Load model from weights file (state_dict only)

    Args:
        weights_path (str): Path to the weights file
        num_classes (int): Number of classes
        device (str): Device to load the model on

    Returns:
        model: Loaded model
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f'Model weights loaded from {weights_path}')
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = (images.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'Loss': f'{running_loss / len(dataloader):.4f}', 'Acc': f'{100.0 * correct / total:.2f}%'})
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return (epoch_loss, epoch_acc)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = (images.to(device), labels.to(device))
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'Loss': f'{running_loss / len(dataloader):.4f}', 'Acc': f'{100.0 * correct / total:.2f}%'})
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return (epoch_loss, epoch_acc)

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'precision_per_class': precision_per_class, 'recall_per_class': recall_per_class, 'f1_per_class': f1_per_class, 'confusion_matrix': cm, 'predictions': all_predictions, 'labels': all_labels}

def plot_confusion_matrix(cm, class_names, round_num, save_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Round {round_num}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_round_{round_num}.png', dpi=300)
    plt.close()

def save_training_history(history, round_num, save_dir):
    """Save training history plot"""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title(f'Loss - Round {round_num}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title(f'Accuracy - Round {round_num}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'training_history_round_{round_num}.png', dpi=300)
    plt.close()

def train_single_round(round_num, cv_data_dir, test_data_dir, save_dir, config):
    """Train and evaluate a single round"""
    print(f"\n{'=' * 60}")
    print(f'TRAINING ROUND {round_num}')
    print(f"{'=' * 60}")
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')
    train_transform, val_test_transform = create_transforms(use_augmentation=config.get('use_augmentation', True))
    round_dir = cv_data_dir / f'Round{round_num}'
    train_dataset = LUADDataset(round_dir / 'train', transform=train_transform)
    val_dataset = LUADDataset(round_dir / 'val', transform=val_test_transform)
    test_dataset = LUADDataset(test_data_dir, transform=val_test_transform)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4), pin_memory=True)
    model = create_model(num_classes=len(train_dataset.class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    early_stopping = EarlyStopping(patience=config['patience'])
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print('-' * 40)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_acc': val_acc, 'val_loss': val_loss, 'train_acc': train_acc, 'train_loss': train_loss, 'config': config}, save_dir / f'best_model_round_{round_num}.pth')
        if early_stopping(val_loss, model):
            print(f'Early stopping at epoch {epoch + 1}')
            break
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)
    torch.save(model.state_dict(), save_dir / f'final_weights_round_{round_num}.pth')
    final_checkpoint = {'round': round_num, 'model_state_dict': model.state_dict(), 'best_val_acc': best_val_acc, 'training_time': training_time, 'epochs_trained': len(history['train_loss']), 'config': config, 'class_names': train_dataset.class_names, 'model_name': 'vit_base_patch16_224', 'num_classes': len(train_dataset.class_names)}
    torch.save(final_checkpoint, save_dir / f'checkpoint_round_{round_num}.pth')
    print(f'\nEvaluating on test set...')
    test_results = evaluate_model(model, test_loader, device, train_dataset.class_names)
    results = {'round': round_num, 'best_val_acc': best_val_acc, 'training_time': training_time, 'epochs_trained': len(history['train_loss']), 'test_accuracy': test_results['accuracy'], 'test_precision': test_results['precision'], 'test_recall': test_results['recall'], 'test_f1': test_results['f1'], 'test_precision_per_class': test_results['precision_per_class'].tolist(), 'test_recall_per_class': test_results['recall_per_class'].tolist(), 'test_f1_per_class': test_results['f1_per_class'].tolist(), 'class_names': train_dataset.class_names}
    save_training_history(history, round_num, save_dir)
    plot_confusion_matrix(test_results['confusion_matrix'], train_dataset.class_names, round_num, save_dir)
    with open(save_dir / f'results_round_{round_num}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f'\nROUND {round_num} TEST RESULTS:')
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1-Score: {test_results['f1']:.4f}")
    return results
