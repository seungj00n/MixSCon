import os
import openslide
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
from PIL import Image
from functools import partial
import numpy as np

def parse_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for annotation in root.find('Annotations'):
        part_of_group = annotation.attrib['PartOfGroup']
        coordinates = annotation.find('Coordinates')
        polygon = []
        for coord in coordinates:
            x = float(coord.attrib['X'])
            y = float(coord.attrib['Y'])
            polygon.append((x, y))
        annotations.append({'class': part_of_group, 'polygon': Polygon(polygon)})
    return annotations

def is_background_patch(patch_img, threshold=0.09):
    """
    Returns True if the patch is mostly background (i.e., not enough tissue).
    `threshold` is the minimum percentage of non-background pixels.
    threshold > 0.09 의미있는 조직 제외될 수 있음, 0.07 픽셀 적은 패치 제외
    """
    img = np.array(patch_img)
    hsv = Image.fromarray(img).convert('HSV')
    hsv = np.array(hsv)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    tissue_mask = (saturation > 30) & (value < 230)
    tissue_ratio = np.sum(tissue_mask) / (patch_img.size[0] * patch_img.size[1])
    return tissue_ratio < threshold

def process_single_file(tif_path, xml_path, output_dir, patch_size):
    try:
        slide = openslide.OpenSlide(tif_path)
        annotations = parse_annotations(xml_path)
        width, height = slide.dimensions
        basename = os.path.splitext(os.path.basename(tif_path))[0]
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                center = Point(x + patch_size / 2, y + patch_size / 2)
                for ann in annotations:
                    if ann['polygon'].contains(center):
                        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
                        if is_background_patch(patch, threshold=0.05):
                            continue
                        filename = f"{basename}_{ann['class']}_x_{x}_y_{y}.png"
                        patch.save(os.path.join(output_dir, filename))
                        break
    except Exception as e:
        print(f'[ERROR] Processing {tif_path} failed: {e}')
        raise

def run(root_dir, output_dir, patch_size):
    all_files = os.listdir(root_dir)
    tif_files = [f for f in all_files if f.endswith('.tif')]
    for casename in tif_files:
        case_name = casename.split('.')[0]
        tif_path = os.path.join(root_dir, f'{case_name}.tif')
        xml_path = os.path.join(root_dir, f'{case_name}.xml')
        output_path = os.path.join(output_dir, case_name)
        os.makedirs(output_path, exist_ok=True)
        process_single_file(tif_path, xml_path, output_path, patch_size)
        print(f'{case_name}: Path Extraction Complete!')
    print('______________END________________')
