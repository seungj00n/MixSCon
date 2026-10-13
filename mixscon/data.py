"""Manifest validation and folder checks; no study identifiers are embedded."""
from pathlib import Path
import csv

CLASSES = ['acinar', 'cribriform', 'lepidic', 'micropapillary', 'papillary', 'solid']
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

def read_manifest(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError('Required manifest columns: ' + ', '.join(required))
        rows = list(reader)
    if not rows:
        raise ValueError('The manifest is empty. Example files contain headers only.')
    return rows

def label_number(value):
    number = int(value)
    if str(number) != str(value).strip() or number not in range(6):
        raise ValueError('Labels must be integers 0..5 in the documented class order.')
    return number

def image_path(value, base):
    path = Path(value)
    if not value:
        raise ValueError('Missing image path.')
    if not path.is_absolute():
        path = Path(base) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()

def check_folders(data_root, test_root, folds, batch_size=1):
    """Check six-class order and patient-disjoint filename prefixes per fold."""
    roots = [('test', Path(test_root))]
    roots += [(f'Round{k}/{split}', Path(data_root) / f'Round{k}' / split)
              for k in folds for split in ['train', 'val']]
    members = {}
    counts = {}
    for name, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        names = [p.name for p in dirs]
        if names != [f'{i + 1}_{cls}' for i, cls in enumerate(CLASSES)]:
            raise ValueError(f'{root}: expected six ordered class folders; use preprocessing.prepare_splits.')
        files = [p for directory in dirs for p in directory.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS]
        if any(not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in directory.rglob('*')) for directory in dirs):
            raise ValueError(f'{root}: each class must contain an image.')
        if name.endswith('/train') and len(files) < batch_size:
            raise ValueError('Training set is smaller than a batch (drop_last=True).')
        if any('_' not in p.stem for p in files):
            raise ValueError('Training image names must start with a patient key followed by an underscore.')
        members[name] = {p.stem.split('_', 1)[0] for p in files}
        counts[name] = len(files)
    for fold in folds:
        groups = [members[f'Round{fold}/train'], members[f'Round{fold}/val'], members['test']]
        if any(groups[i] & groups[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError(f'Patient leakage in fold {fold}.')
    return counts
