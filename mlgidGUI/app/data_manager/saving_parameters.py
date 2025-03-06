from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from pathlib import Path

from ..file_manager import FolderKey, ImageKey


class SaveFormats(Enum):
    entire_h5 = 'Save entire project as h5 (recommended, import possible)'
    partial_h5 = 'Save parts of project as h5 (import not possible)'
    object_detection = 'Save as pascal-VOC dataset'
    text = 'Text formats'


class TextFormats(Enum):
    csv = 'csv'
    txt = 'txt'


class MetaTextFormats(Enum):
    json = 'json'
    yaml = 'yaml'


class RoiSavingType(Enum):
    group_by_image = 'Group segments by image'
    group_by_time = 'Group segments by time'


class SaveMode(Enum):
    create = 'New save'
    add = 'Update save'


@dataclass
class SavingParameters:
    selected_images: Dict[FolderKey, List[ImageKey]]
    path: Path
    save_mode: SaveMode = SaveMode.create

    save_image: bool = True
    save_polar_image: bool = True
    save_geometries: bool = True
    save_baselines: bool = True
    save_positions: bool = True
    save_roi_types: bool = True
    save_roi_keys: bool = True
    save_roi_metadata: bool = True

    format: SaveFormats = SaveFormats.entire_h5
    text_format: TextFormats = TextFormats.csv
    meta_text_format: MetaTextFormats = MetaTextFormats.yaml
    roi_saving_type: RoiSavingType = RoiSavingType.group_by_image

    BOOL_FLAGS = {'save_image': 'Save images',
                  'save_polar_image': 'Save polar images',
                  'save_geometries': 'Save geometries',
                  'save_baselines': 'Save baselines',
                  'save_positions': 'Save roi positions',
                  'save_roi_types': 'Save roi types',
                  'save_roi_keys': 'Save roi keys',
                  'save_roi_metadata': 'Save roi names',
                  }

    def set_entire_h5_params(self):
        self.save_image: bool = True
        self.save_polar_image: bool = True
        self.save_geometries: bool = True
        self.save_baselines: bool = True
        self.save_positions: bool = True
        self.save_roi_types: bool = True
        self.save_roi_keys: bool = True
        self.save_roi_metadata: bool = True

    def set_partial_h5_params(self):
        self.save_image: bool = False
        self.save_polar_image: bool = False
        self.save_geometries: bool = True
        self.save_baselines: bool = False
        self.save_positions: bool = True
        self.save_roi_types: bool = True
        self.save_roi_keys: bool = False
        self.save_roi_metadata: bool = False

    @property
    def num_images(self) -> int:
        return sum(map(len, self.selected_images.values()))
