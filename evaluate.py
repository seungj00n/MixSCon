"""Evaluate explicit checkpoints on AMC, TCGA or DHMC tiles and save ROC curves."""
import argparse
import json
from pathlib import Path
from mixscon.data import CLASSES, read_manifest, label_number, image_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, choices=['AMC', 'TCGA', 'DHMC'])
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--checkpoint-config', required=True, type=Path)
    parser.add_argument('--weights-root', required=True, type=Path)
    parser.add_argument('--data-root', type=Path, help='Relative image path base; default: manifest directory')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Invalid batch size or worker count.')
    cfg = json.loads(args.checkpoint_config.read_text(encoding='utf-8'))
    if cfg['class_order'] != CLASSES:
        parser.error('Checkpoint class_order must match the six documented classes.')
    names = args.models or list(cfg['checkpoints'])
    supported = {'resnet50', 'vit16', 'ding', 'alsubaie', 'cellomaps', 'mixscon_ce', 'mixscon_fl', 'class_mixup'}
    if not names or len(names) != len(set(names)) or not set(names).issubset(supported & set(cfg['checkpoints'])):
        parser.error('Unknown or duplicate model name.')
    specs = {}
    for name in names:
        spec = cfg['checkpoints'][name]
        if name == 'ding' and (not isinstance(spec, list) or len(spec) != 3):
            parser.error('Ding requires exactly three checkpoint paths.')
        if name != 'ding' and not isinstance(spec, str):
            parser.error('Each other model requires one checkpoint path.')
        files = [image_path(s, args.weights_root) for s in (spec if isinstance(spec, list) else [spec])]
        specs[name] = files if isinstance(spec, list) else files[0]
    rows = read_manifest(args.manifest, ['sample_id', 'label', 'raw_path', 'cellmap_path'])
    seen = set()
    for row in rows:
        if not row['sample_id'] or row['sample_id'] in seen:
            parser.error('Empty or duplicate sample_id.')
        seen.add(row['sample_id'])
        row['label'] = label_number(row['label'])
        if args.cohort == 'TCGA' and row['label'] == 1:
            parser.error('The study TCGA protocol has no Cribriform reference labels.')
        for field, needed in [('raw_path', any(n != 'cellomaps' for n in names)), ('cellmap_path', 'cellomaps' in names)]:
            if needed:
                row[field] = str(image_path(row[field], args.data_root or args.manifest.resolve().parent))
    if args.dry_run:
        print(json.dumps({'tiles': len(rows), 'models': names, 'status': 'inputs validated; inference not run'}))
        return
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix
    from baselines.models import build_model, dataset_mode, infer_probabilities
    from analysis.statistics import metric_values
    device = torch.device(args.device)
    frame = pd.DataFrame(rows)
    labels = frame['label'].to_numpy(dtype=np.int64)
    present = sorted(set(labels.tolist()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    per_class = []
    for name in names:
        model = build_model(name, specs[name], device)
        probs = infer_probabilities(model, frame, dataset_mode(name), device, args.batch_size, args.num_workers)
        if probs.shape != (len(frame), 6) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError('Invalid probability matrix for ' + name)
        predicted = probs.argmax(1)
        metrics.append({'cohort': args.cohort, 'model': name, 'tiles': len(labels), **metric_values(labels, probs, present)})
        columns = [c for c in ['sample_id', 'patient_id', 'wsi_id', 'label'] if c in frame]
        predictions = frame[columns].copy()
        for i, cls in enumerate(CLASSES):
            predictions['prob_' + cls] = probs[:, i]
        predictions['predicted_label'] = predicted
        predictions.to_csv(args.output_dir / f'{name}_tile_predictions.csv', index=False)
        precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=range(6), zero_division=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, cls in enumerate(CLASSES):
            binary = (labels == i).astype(int)
            auc = float('nan')
            if len(set(binary)) == 2:
                auc = float(roc_auc_score(binary, probs[:, i]))
                fpr, tpr, _ = roc_curve(binary, probs[:, i])
                ax.plot(fpr, tpr, label=f'{cls}: {auc:.3f}')
            per_class.append({'model': name, 'class': cls, 'auc': auc, 'precision': precision[i],
                              'recall': recall[i], 'f1': f1[i], 'support': int(support[i])})
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{args.cohort}: {name}')
        if any(len(set(labels == i)) == 2 for i in present):
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / f'{name}_roc.png', dpi=200)
        plt.close(fig)
        pd.DataFrame(confusion_matrix(labels, predicted, labels=range(6)), index=CLASSES, columns=CLASSES).to_csv(args.output_dir / f'{name}_confusion_matrix.csv')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(args.output_dir / 'point_estimates.csv', index=False)
    pd.DataFrame(per_class).to_csv(args.output_dir / 'per_class_metrics.csv', index=False)
    metadata = {'cohort': args.cohort, 'class_order': CLASSES, 'metric_labels_present': present,
                'averaging': 'macro OvR AUC over evaluable reference classes; macro precision/F1 over reference classes present',
                'prediction': 'six-way argmax; Cribriform retained; no probability renormalization',
                'checkpoints_used': {k: [str(p) for p in v] if isinstance(v, list) else str(v) for k, v in specs.items()},
                'device': str(device), 'torch': torch.__version__}
    (args.output_dir / 'run_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved point_estimates.csv, per-class metrics, predictions, confusion matrices and ROC curves.')

if __name__ == '__main__':
    main()
