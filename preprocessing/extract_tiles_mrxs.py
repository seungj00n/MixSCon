"""Extract annotated MRXS tiles using tumor and growth-pattern ASAP XML files."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wsi', required=True, type=Path)
    parser.add_argument('--tumor-xml', required=True, type=Path)
    parser.add_argument('--pattern-xml', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--patch-size', type=int, default=768)
    parser.add_argument('--tissue-threshold', type=float, default=0.3)
    args = parser.parse_args()
    if any(not p.is_file() for p in [args.wsi, args.tumor_xml, args.pattern_xml]) or args.patch_size < 1 or not 0 <= args.tissue_threshold <= 1:
        parser.error('Check input files, patch size and tissue threshold (0..1).')
    from ._mrxs import create_patches
    create_patches(str(args.wsi), str(args.tumor_xml), str(args.pattern_xml), args.patch_size, str(args.output_dir), args.tissue_threshold)

if __name__ == '__main__':
    main()
