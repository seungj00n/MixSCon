import openslide
import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    regions = []
    for annotation in root.findall('.//Annotation'):
        region = []
        for coordinate in annotation.findall('.//Coordinate'):
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            region.append((x, y))
        regions.append(np.array(region, dtype=np.int32))
    return regions

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def is_patch_valid(patch, threshold, intensity_thresh=30):
    patch_array_gray = np.array(patch.convert('L'))
    _, binary_patch = cv2.threshold(patch_array_gray, intensity_thresh, 255, cv2.THRESH_BINARY)
    non_empty_pixels = np.sum(binary_patch > 0)
    total_pixels = binary_patch.size
    non_empty_ratio = non_empty_pixels / total_pixels
    if non_empty_ratio < threshold:
        return False
    patch_array_rgb = np.array(patch)[:, :, :3]
    non_backgroud_pixels = np.sum(np.any(patch_array_rgb < [244, 244, 244], axis=-1))
    non_backgroud_ratio = non_backgroud_pixels / total_pixels
    if non_backgroud_ratio < threshold:
        return False
    patch_hsv = cv2.cvtColor(patch_array_rgb, cv2.COLOR_RGB2HSV)
    saturation_channel = patch_hsv[:, :, 1]
    otsu_thresh, _ = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_pixels = np.sum(saturation_channel > otsu_thresh)
    tissue_ratio = tissue_pixels / total_pixels
    if tissue_ratio < threshold:
        return False
    return True

def create_patches(image_file, tumor_annotation, pattern_annotation, patch_size, output_dir, thresh_param):
    case_name = os.path.splitext(os.path.basename(image_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    slide = openslide.OpenSlide(image_file)
    slide_dims = slide.dimensions
    tumor_regions = parse_annotation(tumor_annotation)
    pattern_data = ET.parse(pattern_annotation)
    pattern_root = pattern_data.getroot()
    pattern_regions = []
    pattern_names = []
    for annotation in pattern_root.findall('.//Annotation'):
        pattern_name = annotation.get('PartOfGroup')
        pattern = []
        for coordinate in annotation.findall('.//Coordinate'):
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            pattern.append((x, y))
        pattern_regions.append(np.array(pattern, dtype=np.int32))
        pattern_names.append(pattern_name)
    for y in range(0, slide_dims[1], patch_size):
        for x in range(0, slide_dims[0], patch_size):
            patch_coords = (x + patch_size // 2, y + patch_size // 2)
            in_tumor = any((point_in_polygon(patch_coords, region) for region in tumor_regions))
            if not in_tumor:
                continue
            pattern_name = 'unknown'
            for i, pattern_region in enumerate(pattern_regions):
                if point_in_polygon(patch_coords, pattern_region):
                    pattern_name = pattern_names[i]
                    break
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
            if not is_patch_valid(patch, thresh_param):
                continue
            patch_filename = f'{case_name}_{pattern_name}_x_{x}_y_{y}.png'
            patch.save(os.path.join(output_dir, patch_filename))
    print(f'{case_name} : Creating patches completed!')
