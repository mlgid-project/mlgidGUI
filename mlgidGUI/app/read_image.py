# -*- coding: utf-8 -*-
from typing import Union
from pathlib import Path

import numpy as np
from PIL import Image
from .edf_reader import read_edf
import tifffile


def read_image(filepath: Union[Path, str]) -> np.array:
    if isinstance(filepath, Path):
        filepath = str(filepath.resolve())

    if filepath.endswith('.edf') or filepath.endswith('.edf.gz'):
        image = read_edf(filepath)
    else:
        image = tifffile.imread(filepath)
        if len(image.shape) > 2:
            image = np.array(Image.open(filepath).convert('L'))
    return image