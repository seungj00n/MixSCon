"""Portable entry points around the recovered per-fold training routines."""
import argparse
import importlib
import json
from pathlib import Path
import random
from .data import check_folders

def run(kind):
    parser = argparse.ArgumentParser(description=f'Train {kind} MixSCon on supplied patient-disjoint folders.')
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--test-root', type=Path, help='Default: DATA_ROOT/test')
    parser.add_argument('--output-dir', required=True, type=Path)
    if kind == 'patient':
        parser.add_argument('--loss', choices=['ce', 'fl'], default='ce')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0 or len(set(args.folds)) != len(args.folds):
        parser.error('Invalid epoch, batch, worker or duplicate fold argument.')
    test_root = args.test_root or args.data_root / 'test'
    counts = check_folders(args.data_root, test_root, args.folds, args.batch_size)
    if args.dry_run:
        print(json.dumps(counts, indent=2))
        return
    import numpy as np
    import torch
    module = importlib.import_module('mixscon.training_' + ('patient' if kind == 'patient' else 'class'))
    if kind == 'patient':
        config_path = Path(__file__).resolve().parents[1] / 'configs' / f'mixscon_{args.loss}.json'
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
    else:
        cfg = module.CONFIG.copy()
    cfg.update(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    if args.device:
        cfg['device'] = args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / f'best_model_round_{fold}.pth').exists() for fold in args.folds):
        raise FileExistsError('Choose a fresh output directory to preserve existing checkpoints.')
    results = [module.train_single_round(fold, args.data_root, test_root, args.output_dir, cfg) for fold in args.folds]
    stats = {key: {'mean': float(np.mean([r[key] for r in results])),
                   'std': float(np.std([r[key] for r in results]))}
             for key in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']}
    # Keep fold checkpoints separate; test metrics are reporting-only.
    summary = {'config': cfg, 'individual_results': results, 'overall_statistics': stats,
               'checkpoint_selection': {'scope': 'per_fold', 'monitor': 'validation_loss',
                                        'min_delta': cfg['min_delta']},
               'training_report_average': 'weighted precision, recall and F1'}
    (args.output_dir / 'cross_validation_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved checkpoints, per-fold metrics, training curves and cross_validation_results.json.')
