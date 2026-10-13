"""Materialize an authorized split manifest; never infer the original study splits."""
import argparse
from pathlib import Path
import shutil
from mixscon.data import CLASSES, read_manifest, label_number, image_path

def prepare(manifest, data_root, output_dir, dry_run=False):
    rows = read_manifest(manifest, ['sample_id', 'patient_id', 'wsi_id', 'fold', 'split', 'label', 'path'])
    seen = set()
    samples, wsi_patients, source_samples = {}, {}, {}
    partitions = {}
    test_patients, test_wsis = set(), set()
    for row in rows:
        for key in ['sample_id', 'patient_id', 'wsi_id']:
            if not row[key].strip():
                raise ValueError(f'Empty {key}.')
        row['label'] = label_number(row['label'])
        row['source'] = image_path(row['path'], data_root)
        if row['source'].suffix.lower() != '.png':
            raise ValueError('Use PNG tiles, as expected by the study baseline loaders.')
        split = row['split']
        if split == 'test':
            if row['fold'] not in {'', '0'}:
                raise ValueError('Use fold 0 for the shared test set, once per tile.')
            fold = 0
            test_patients.add(row['patient_id'])
            test_wsis.add(row['wsi_id'])
        else:
            fold = int(row['fold'])
            if fold not in range(1, 6) or split not in {'train', 'val'}:
                raise ValueError('Use folds 1..5 and split train/val, or fold 0 and split test.')
        row['fold'] = fold
        key = (fold, row['sample_id'])
        if key in seen:
            raise ValueError('A sample occurs more than once in a fold.')
        seen.add(key)
        identity = (row['patient_id'], row['wsi_id'], row['label'], row['source'])
        if row['sample_id'] in samples and samples[row['sample_id']] != identity:
            raise ValueError('Inconsistent sample identity across folds.')
        samples[row['sample_id']] = identity
        previous_sample = source_samples.setdefault(row['source'], row['sample_id'])
        if previous_sample != row['sample_id']:
            raise ValueError('The same image path is assigned different sample IDs.')
        if row['wsi_id'] in wsi_patients and wsi_patients[row['wsi_id']] != row['patient_id']:
            raise ValueError('A WSI belongs to multiple patients.')
        wsi_patients[row['wsi_id']] = row['patient_id']
        for kind in ['patient_id', 'wsi_id']:
            key = (fold, kind, row[kind])
            if key in partitions and partitions[key] != split:
                raise ValueError('Patient or WSI leakage between train and validation.')
            partitions[key] = split
    for row in rows:
        if row['split'] != 'test' and (row['patient_id'] in test_patients or row['wsi_id'] in test_wsis):
            raise ValueError('A test patient or WSI occurs in training/validation.')
    patients = {key: f'p{i:06d}' for i, key in enumerate(sorted({r['patient_id'] for r in rows}), 1)}
    tile_keys = {key: f't{i:08d}' for i, key in enumerate(sorted(samples), 1)}
    jobs = []
    output_dir = Path(output_dir)
    for row in rows:
        split_dir = Path('test') if row['split'] == 'test' else Path(f"Round{row['fold']}") / row['split']
        cls = row['label']
        name = patients[row['patient_id']] + '_' + tile_keys[row['sample_id']] + row['source'].suffix.lower()
        target = output_dir / split_dir / f'{cls + 1}_{CLASSES[cls]}' / name
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        jobs.append((row['source'], target))
    if not dry_run:
        for source, target in jobs:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return {'tiles_to_copy': len(jobs), 'patients': len(patients), 'dry_run': dry_run}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(prepare(args.manifest, args.data_root, args.output_dir, args.dry_run))

if __name__ == '__main__':
    main()
