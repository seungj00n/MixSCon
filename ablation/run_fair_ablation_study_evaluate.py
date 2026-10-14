import os
import argparse
from pathlib import Path
from mixscon.data import CLASS_FOLDERS, map_labels
from . import run_fair_ablation_study as training
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, auc, roc_curve,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from .run_fair_ablation_study import (
    Encoder, Classifier, extract_features, plot_tsne,
    imagenet_transform, set_seed, SEED, BATCH_SIZE, EMBED_DIM,
)

NUM_WORKERS = 4
DO_TSNE = True
CLASS_NAMES = CLASS_FOLDERS
RESULT_SUBDIR = 'results'
EXPERIMENTS = [name for name, _ in training.EXPERIMENTS]


def safe_name_of(exp_name):
    # 원본 run_experiment()의 파일명 규칙과 반드시 동일해야 함
    return (exp_name.replace(" ", "_").replace("+", "").replace("(", "")
            .replace(")", "").replace(".", ""))


def build_test_loader(test_dir):
    dataset = datasets.ImageFolder(test_dir, transform=imagenet_transform())
    dataset = map_labels(dataset, CLASS_NAMES)

    counts = np.bincount(dataset.targets, minlength=len(CLASS_NAMES))
    for name, n in zip(CLASS_NAMES, counts):
        print(f"  {name:<20} {n:>6}")
    absent = [c for c, n in zip(CLASS_NAMES, counts) if n == 0]
    if absent:
        print(f"[WARNING] 테스트 데이터에 없는 클래스: {absent} "
              f"(해당 클래스의 Accuracy/AUC는 NaN, macro는 존재 클래스 기준으로도 별도 계산)")
    print(f"Test images: {len(dataset)}")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def evaluate_model_partial(encoder, classifier, data_loader, device, class_names, save_dir,
                           prefix="test"):

    os.makedirs(save_dir, exist_ok=True)
    n_cls = len(class_names)
    all_idx = list(range(n_cls))

    encoder.eval()
    classifier.to(device)
    classifier.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            feats, _ = encoder(images)
            outputs = classifier(feats)
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    rows = pd.DataFrame({
        'filename': [Path(path).name for path, _ in data_loader.dataset.samples],
        'label': all_labels, 'predicted_label': all_preds,
        'true': [class_names[i] for i in all_labels],
        'pred': [class_names[i] for i in all_preds],
        'confidence score': all_probs.max(axis=1),
    })
    for i, name in enumerate(training.CLASS_FOLDERS):
        rows['prob_' + name.split('_', 1)[1]] = all_probs[:, i]
    rows.to_csv(os.path.join(save_dir, f'{prefix}_predictions.csv'), index=False)

    support = np.bincount(all_labels, minlength=n_cls)
    present = [i for i in all_idx if support[i] > 0]
    absent = [i for i in all_idx if support[i] == 0]

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"[INFO] {prefix} Accuracy: {overall_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=all_idx)
    per_class_acc = np.full(n_cls, np.nan)
    per_class_acc[present] = cm.diagonal()[present] / support[present]
    for cname, cacc in zip(class_names, per_class_acc):
        print(f"{prefix} {cname} Accuracy: {cacc:.4f}")

    # Classification report: labels 고정
    report = classification_report(all_labels, all_preds, labels=all_idx,
                                   target_names=class_names, digits=4,
                                   output_dict=True, zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=present,
                                                  average="macro", zero_division=0)
    report["macro avg (present classes)"] = {
        "precision": p, "recall": r, "f1-score": f1, "support": float(support[present].sum())}
    report_path = os.path.join(save_dir, f"{prefix}_classification_report.csv")
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"[INFO] Classification report saved to {report_path}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{prefix} Confusion Matrix")
    cm_path = os.path.join(save_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    # ROC: 존재 클래스만 (one-vs-rest)
    per_class_auc = {c: np.nan for c in class_names}
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap("tab10", n_cls)
    for i in present:
        y_bin = (all_labels == i).astype(int)
        if y_bin.min() == y_bin.max():
            continue
        fpr, tpr, _ = roc_curve(y_bin, all_probs[:, i])
        per_class_auc[class_names[i]] = auc(fpr, tpr)
        print(f"{prefix} {class_names[i]} AUC: {per_class_auc[class_names[i]]:.4f}")
        plt.plot(fpr, tpr, color=colors(i), lw=2,
                 label=f"{class_names[i]} (AUC = {per_class_auc[class_names[i]]:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(save_dir, f"{prefix}_roc_all_classes.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[INFO] ROC curves saved to {roc_path}")

    present_aucs = [per_class_auc[class_names[i]] for i in present
                    if not np.isnan(per_class_auc[class_names[i]])]
    metrics = (
        [("Overall Accuracy", overall_acc)]
        + [(f"{c} Accuracy", a) for c, a in zip(class_names, per_class_acc)]
        + [(f"{c} AUC", per_class_auc[c]) for c in class_names]
        + [("Macro AUC (present classes)", np.mean(present_aucs) if present_aucs else np.nan),
           ("Macro F1 (present classes)", f1)]
        + [(f"Predicted as absent class {class_names[i]} (count)",
            int((all_preds == i).sum())) for i in absent]
    )
    metrics_path = os.path.join(save_dir, f"{prefix}_metrics_summary.csv")
    pd.DataFrame(metrics, columns=["Metric", "Value"]).to_csv(metrics_path, index=False)
    print(f"[INFO] Metrics summary saved to {metrics_path}")
    for i in absent:
        print(f"[WARNING] 정답에 없는 {class_names[i]}로 예측된 샘플: {int((all_preds == i).sum())}개 (오답 처리됨)")

    return overall_acc, report


def evaluate_one(exp_name, model_dir, save_dir, test_loader, device):
    set_seed(SEED)
    models_dir = os.path.join(model_dir, "models")
    results_dir = os.path.join(save_dir, RESULT_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)

    name = safe_name_of(exp_name)
    encoder_path = os.path.join(models_dir, f"{name}_encoder.pth")
    classifier_path = os.path.join(models_dir, f"{name}_classifier.pth")
    for p in (encoder_path, classifier_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    encoder = Encoder(out_dim=EMBED_DIM, pretrained=False).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    classifier = Classifier(feature_dim=EMBED_DIM, num_classes=len(CLASS_NAMES)).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    encoder.eval()
    classifier.eval()
    print(f"[INFO] Loaded: {encoder_path}\n[INFO] Loaded: {classifier_path}")

    overall_acc, report = evaluate_model_partial(encoder, classifier, test_loader, device, CLASS_NAMES,
                                                 results_dir, prefix=f"newtest-{exp_name}")

    tsne_path = None
    if DO_TSNE:
        tsne_path = os.path.join(results_dir, f"{name}_newtest_tsne.png")
        feats, labels = extract_features(encoder, test_loader, device)
        plot_tsne(feats, labels, CLASS_NAMES, tsne_path, title=f"t-SNE (new test): {exp_name}")

    return {
        'experiment': exp_name,
        'encoder_path': encoder_path,
        'classifier_path': classifier_path,
        'test_dir': str(test_loader.dataset.root),
        'tsne_path': tsne_path,
        'test_acc': overall_acc,
        'test_report': report,
        'timestamp': datetime.now().isoformat(),
    }


def main():
    global BATCH_SIZE, NUM_WORKERS, SEED, DO_TSNE
    parser = argparse.ArgumentParser(description='Evaluate explicit ablation fold checkpoints on AMC, TCGA or DHMC.')
    parser.add_argument('--data-root', type=Path, required=True, help='ImageFolder with named growth-pattern folders.')
    parser.add_argument('--cohort', choices=['AMC', 'TCGA', 'DHMC'], required=True)
    parser.add_argument('--model-root', type=Path, required=True, help='Training output root containing Round*/models.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--folds', type=int, nargs='+', choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument('--experiments', type=int, nargs='+', choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-tsne', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Invalid batch size or worker count.')
    if len(set(args.folds)) != len(args.folds) or len(set(args.experiments)) != len(args.experiments):
        parser.error('Duplicate folds or experiments.')
    BATCH_SIZE, NUM_WORKERS, SEED = args.batch_size, args.num_workers, args.seed
    training.SEED = SEED
    DO_TSNE = not args.skip_tsne
    loader = build_test_loader(args.data_root)
    if args.cohort == 'TCGA' and 1 in loader.dataset.targets:
        parser.error('The study TCGA cohort has no Cribriform reference labels.')
    for fold in args.folds:
        for index in args.experiments:
            name = safe_name_of(EXPERIMENTS[index - 1])
            for part in ['encoder', 'classifier']:
                path = args.model_root / f'Round{fold}/models/{name}_{part}.pth'
                if not path.is_file():
                    raise FileNotFoundError(path)
    if args.dry_run:
        print('Class mapping, input data and checkpoint paths validated.')
        return
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'run_config.json').write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2),
        encoding='utf-8')
    for fold in args.folds:
        output = args.output_dir / f'Round{fold}'
        results = [evaluate_one(EXPERIMENTS[i - 1], args.model_root / f'Round{fold}',
                                output, loader, device) for i in args.experiments]
        (output / 'results/external_evaluation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
