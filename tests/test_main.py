from pathlib import Path

import numpy as np
from pytest import fixture
from PIL import Image

from main import get_polygons
from main import get_most_freq_color


@fixture
def mask_img():
    return np.array(Image.open(Path('tests/data/mask.png')))


@fixture
def polygons(mask_img):
    return get_polygons(mask_img)


def test_get_polygons(mask_img):
    polygons = get_polygons(mask_img)
    assert len(polygons) == 3

    for p in polygons:
        assert len(p)


def test_most_freq_color(mask_img):
    orig_img = np.array(Image.open(Path('tests/data/color_mask.png')))
    _, regions, _ = get_polygons(mask_img)
    polygon_coords = regions[0].coords
    most_freq_color = get_most_freq_color(polygon_coords, orig_img)
    assert most_freq_color == (230, 91, 32, 255) 
