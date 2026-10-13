"""Extract 768-pixel TIFF tiles using same-basename ASAP XML annotations."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--patch-size', type=int, default=768)
    args = parser.parse_args()
    if not args.data_root.is_dir() or args.patch_size < 1:
        parser.error('Supply a valid input directory and positive patch size.')
    from ._tif import run
    run(str(args.data_root), str(args.output_dir), args.patch_size)

if __name__ == '__main__':
    main()
