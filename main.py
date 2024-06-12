from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label
from skimage.measure import regionprops


def get_polygons(mask_img):
    mask_array = np.array(mask_img)
    binary_mask = mask_array > 128
    labeled_mask, _ = label(binary_mask, return_num=True)
    regions = regionprops(labeled_mask)
    return labeled_mask, regions, mask_array


def get_most_freq_color(polygon_coords, color_img):
    pixel_colors = color_img[polygon_coords[:, 0], polygon_coords[:, 1]]
    color_counts = Counter(map(tuple, pixel_colors))
    return color_counts.most_common(1)[0][0]


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_img', help='Input image', required=True)
    parser.add_argument('-p', '--polygon_img', help='Polygon image', required=True)
    parser.add_argument('-o', '--output_img', help='Output image', required=True)
    args = parser.parse_args()

    orig_img = np.array(Image.open(Path(args.input_img)))
    mask_img = Image.open(Path(args.polygon_img)).convert('L')
    _, regions, mask_array = get_polygons(mask_img)
    res_img = Image.new('RGBA', mask_img.size)

    for region in regions:
        polygon_coords = region.coords
        most_freq_color = get_most_freq_color(polygon_coords, orig_img)
        polygon_mask = np.zeros_like(mask_array)
        polygon_mask[polygon_coords[:, 0], polygon_coords[:, 1]] = 1
        color_fill_img = Image.new('RGBA', mask_img.size, most_freq_color)
        polygon_mask_img = Image.fromarray((polygon_mask * 255).astype(np.uint8))
        res_img = Image.composite(color_fill_img, res_img, polygon_mask_img)

    res_img.save(args.output_img)
    

if __name__ == '__main__':
    main()
