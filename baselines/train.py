"""Run the available study baseline training routines with explicit paths."""
import argparse
import importlib
import json
from pathlib import Path
from mixscon.data import check_folders

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['vit16', 'alsubaie', 'cellomaps', 'ding'])
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--test-root', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--ssl-checkpoints', nargs=3, type=Path, help='Ding only: supplied fold-specific SSL1/SSL2/SSL3 encoder states')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0 or len(args.folds) != len(set(args.folds)):
        parser.error('Invalid training arguments.')
    test_root = args.test_root or args.data_root / 'test'
    counts = check_folders(args.data_root, test_root, args.folds, args.batch_size)
    if args.model == 'ding':
        if len(args.folds) != 1 or not args.ssl_checkpoints:
            parser.error('Ding training requires one --folds value and three fold-specific --ssl-checkpoints.')
        if any(not p.is_file() for p in args.ssl_checkpoints):
            parser.error('Missing SSL checkpoint.')
    if args.dry_run:
        print(json.dumps(counts, indent=2))
        return
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    module = importlib.import_module('baselines.' + ('vit' if args.model == 'vit16' else args.model))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    module.DEVICE = device
    module.NUM_WORKERS = args.num_workers
    module.EPOCHS = args.epochs
    module.BATCH_SIZE = args.batch_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'run_config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'ssl_checkpoints'}, indent=2), encoding='utf-8')
    if args.model == 'vit16':
        config = {'epochs': args.epochs, 'patience': 30, 'batch_size': args.batch_size,
                  'learning_rate': 1e-3, 'use_augmentation': False,
                  'num_workers': args.num_workers, 'device': str(device)}
        for fold in args.folds:
            module.train_single_round(fold, args.data_root, test_root, args.output_dir, config)
    elif args.model == 'alsubaie':
        for fold in args.folds:
            root = args.data_root / f'Round{fold}'
            output = args.output_dir / f'Round{fold}'
            output.mkdir(parents=True, exist_ok=True)
            datasets = {split: module.LabeledTileDataset(str(test_root if split == 'test' else root / split),
                       transform=module.get_transform(split == 'train'), mode=module.MULTIRES_MODE, lowres_scale=module.LOWRES_SCALE)
                        for split in ['train', 'val', 'test']}
            loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == 'train', num_workers=args.num_workers) for split, ds in datasets.items()}
            model = module.ResNet50_6ch(num_classes=6).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=module.LR)
            model, history = module.train_model(loaders, model, torch.nn.CrossEntropyLoss(), optimizer,
                                                 num_epochs=args.epochs, patience=30, save_dir=str(output))
            (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            for split in ['val', 'test']:
                module.evaluate(model, loaders[split], datasets['train'].classes, str(output / f'{split}_metrics.json'))
    elif args.model == 'cellomaps':
        for fold in args.folds:
            root = args.data_root / f'Round{fold}'
            output = args.output_dir / f'Round{fold}'
            checkpoints = output / 'checkpoints'
            checkpoints.mkdir(parents=True, exist_ok=True)
            checkpoint = module.train_one_fold(str(root), str(output), str(checkpoints), fold, use_pretrained_weights=False, test_root=str(test_root))
            model = module.CellOMapsResNet50(num_classes=6, use_pretrained=False).to(device)
            for split in ['val', 'test']:
                loader = module.data_loader(str(test_root if split == 'test' else root / split))
                module.evaluate_model(model, loader, module.FocalLoss(gamma=0.7), str(output), checkpoint, prefix=split)
    else:
        fold = args.folds[0]
        for index, checkpoint in enumerate(args.ssl_checkpoints, 1):
            # The recovered routine cannot map EncoderDecoder encoder.* keys.
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if not isinstance(state, dict) or 'conv1.weight' not in state or any(k.startswith('encoder.') for k in state):
                raise ValueError('Ding requires SSL states in ResNet key format; encoder.* checkpoints need an independently verified conversion.')
            module.train_downstream(str(args.data_root / f'Round{fold}'), str(test_root), str(checkpoint),
                                    str(args.output_dir / f'Round{fold}' / f'ssl{index}'),
                                    epochs=args.epochs, patience=30, bs=args.batch_size, lr=module.LR)

if __name__ == '__main__':
    main()
