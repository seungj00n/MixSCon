"""Overlay tile predictions and ASAP annotation contours on a WSI (Fig. 4)."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from mixscon.data import CLASSES, class_index, evaluation_label


class WSIOverlayVisualizer:
    # Class colors and rendering settings used in Fig. 4.
    FIXED_CLASS_COLOR_MAP = {
        'acinar': (32, 100, 255),
        'cribriform': (255, 187, 32),
        'lepidic': (157, 246, 232),
        'micropapillary': (212, 255, 32),
        'papillary': (100, 255, 32),
        'solid': (187, 32, 255),
    }

    def __init__(self, wsi_path, xml_path, csv_path, save_path,
                 image_level=3, conf_threshold=0.7, patch_scale=5.0,
                 patch_size=768, coordinate_origin='center'):
        if (not 0 <= conf_threshold <= 1 or not np.isfinite(patch_scale)
                or patch_scale <= 0 or patch_size < 1 or image_level < 0):
            raise ValueError('Invalid confidence, patch size/scale or image level.')
        if coordinate_origin not in {'center', 'top-left'}:
            raise ValueError('Choose center or top-left coordinates.')
        self.wsi_path, self.xml_path = Path(wsi_path), Path(xml_path)
        self.csv_path, self.save_path = Path(csv_path), Path(save_path)
        self.case_name = self.wsi_path.stem
        self.image_level, self.conf_threshold = image_level, conf_threshold
        self.patch_scale, self.patch_size = patch_scale, patch_size
        self.coordinate_origin = coordinate_origin
        self.annotations = self._load_annotations()
        self.predictions = self._load_predictions()
        import openslide
        self.slide = openslide.OpenSlide(str(self.wsi_path))
        if image_level >= self.slide.level_count:
            self.slide.close()
            raise ValueError('image-level exceeds the available WSI pyramid levels.')

    def _load_annotations(self):
        annotations = []
        for ann in ET.parse(self.xml_path).getroot().findall('.//Annotation'):
            name = (ann.get('Name') or '').strip().lower()
            for candidate in [name, ann.get('PartOfGroup', '')]:
                try:
                    name = CLASSES[class_index(candidate)]
                    break
                except ValueError:
                    pass
            coords = [(float(c.get('X')), float(c.get('Y'))) for c in ann.findall('.//Coordinate')]
            if len(coords) < 3 or not np.isfinite(coords).all():
                raise ValueError('Each annotation polygon needs at least three finite coordinates.')
            annotations.append((name, coords))
        return annotations

    def _load_predictions(self):
        frame = pd.read_csv(self.csv_path, dtype={'filename': str, 'wsi_id': str})
        if 'pred' not in frame and 'predicted_label' in frame:
            frame['pred'] = frame['predicted_label']
        if 'pred' not in frame:
            raise ValueError('Predictions need a pred or predicted_label column.')
        frame['pred'] = frame['pred'].map(lambda v: CLASSES[evaluation_label(v)])
        prob_columns = ['prob_' + c for c in CLASSES]
        if 'confidence score' not in frame and set(prob_columns).issubset(frame.columns):
            frame['confidence score'] = frame[prob_columns].max(axis=1)
        if 'confidence score' not in frame:
            raise ValueError('Predictions need confidence score or six probability columns.')
        confidence = pd.to_numeric(frame['confidence score'], errors='raise')
        if not np.isfinite(confidence).all() or not confidence.between(0, 1).all():
            raise ValueError('Confidence scores must be finite and between 0 and 1.')
        has_coordinates = {'wsi_id', 'x', 'y'}.issubset(frame.columns)
        if not has_coordinates and 'filename' not in frame:
            raise ValueError('Supply coordinate-bearing filenames or wsi_id,x,y columns.')
        return frame.loc[confidence >= self.conf_threshold].copy()

    def _parse_patch_filename(self, filename):
        filename = str(filename).replace('\\', '/').rsplit('/', 1)[-1]
        classes = '|'.join(CLASSES + ['unknown'])
        pattern = rf'^(?P<wsi>.+?)_(?:\d+_)?(?:{classes})_x_(?P<x>\d+)_y_(?P<y>\d+)\.[^.]+$'
        match = re.match(pattern, filename, re.IGNORECASE)
        if not match:
            raise ValueError(f'Expected WSI_class_x_X_y_Y filename, or explicit wsi_id,x,y: {filename}')
        return match['wsi'], int(match['x']), int(match['y'])

    def _row_coordinates(self, row):
        if all(key in row and pd.notna(row[key]) for key in ['wsi_id', 'x', 'y']):
            x, y = float(row['x']), float(row['y'])
            if not np.isfinite([x, y]).all() or min(x, y) < 0:
                raise ValueError('Tile coordinates must be finite, nonnegative level-0 pixels.')
            return str(row['wsi_id']), x, y
        return self._parse_patch_filename(row['filename'])

    def visualize(self):
        fig = None
        try:
            downsample = self.slide.level_downsamples[self.image_level]
            size = self.slide.level_dimensions[self.image_level]
            img = np.array(self.slide.read_region((0, 0), self.image_level, size).convert('RGB'))
            img = np.where(img.sum(axis=-1, keepdims=True) == 0, np.ones_like(img) * 255, img)
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.imshow(img)
            for name, coords in self.annotations:
                color = np.array(self.FIXED_CLASS_COLOR_MAP.get(name, (0, 0, 0))) / 255.0
                scaled = [(x / downsample, y / downsample) for x, y in coords]
                ax.add_patch(Polygon(scaled, closed=True, fill=False, edgecolor=color,
                                     linewidth=2, alpha=1.0))
            count = 0
            for _, row in self.predictions.iterrows():
                wsi_name, x, y = self._row_coordinates(row)
                if wsi_name != self.case_name:
                    continue
                x_scaled, y_scaled = int(x / downsample), int(y / downsample)
                side = max(1, int(self.patch_size * self.patch_scale / downsample))
                if self.coordinate_origin == 'center':
                    x_scaled -= side // 2
                    y_scaled -= side // 2
                color = np.array(self.FIXED_CLASS_COLOR_MAP[row['pred']]) / 255.0
                ax.add_patch(patches.Rectangle((x_scaled, y_scaled), side, side, linewidth=1,
                                               edgecolor=color, facecolor=color, alpha=0.5))
                count += 1
            handles = [patches.Patch(color=np.array(rgb) / 255.0, label=name.capitalize())
                       for name, rgb in self.FIXED_CLASS_COLOR_MAP.items()]
            ax.legend(handles=handles, fontsize=12, loc='upper right', framealpha=1, facecolor='white')
            ax.axis('off')
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_path, bbox_inches='tight', dpi=300)
            settings = dict(image_level=self.image_level, confidence_threshold=self.conf_threshold,
                            patch_scale=self.patch_scale, patch_size=self.patch_size,
                            coordinate_origin=self.coordinate_origin, rendered_tiles=count)
            self.save_path.with_suffix('.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
            print(f'{self.case_name}: saved overlay with {count} tiles to {self.save_path}')
            return settings
        finally:
            if fig is not None:
                plt.close(fig)
            self.slide.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wsi', type=Path, required=True)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-level', type=int, default=3)
    parser.add_argument('--confidence-threshold', type=float, default=0.7)
    parser.add_argument('--patch-scale', type=float, default=5.0)
    parser.add_argument('--patch-size', type=int, default=768)
    parser.add_argument('--coordinate-origin', choices=['center', 'top-left'], default='center')
    args = parser.parse_args()
    WSIOverlayVisualizer(args.wsi, args.xml, args.predictions, args.output,
                         args.image_level, args.confidence_threshold, args.patch_scale,
                         args.patch_size, args.coordinate_origin).visualize()


if __name__ == '__main__':
    main()
