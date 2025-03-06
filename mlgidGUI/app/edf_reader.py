"""
Interface functions for reading edf files and saving images as edf.
"""

from typing import Union, Tuple, List, Dict
from pathlib import Path
from io import BytesIO
import gzip

import numpy as np

__all__ = ['read_edf', 'CorruptedFileError']


class CorruptedFileError(ValueError):
    """
    Raises when file cannot be read.
    """


def read_edf(filepath: Union[Path, str, BytesIO], header_dict: dict = None,
             return_dict: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Parses .edf file and returns numpy array if successful. If return_dict is True,
    returns tuple of image and header dict. If header_dict is provided, uses it instead of
    parsing the file header, it may accelerate reading.

    Raises CorruptedFileError if any parsing error occurs.

    :param filepath: Union[Path, str, BytesIO]. If str provided, it will be converted to Path first.
    :param header_dict: dict = None. Optional, use to accelerate parsing. Should contain
    the following keys: 'headerSize', 'Size', 'DataType', 'Dim_1', 'Dim_2'.
    :param return_dict: bool = False. If True, returns header_dict along with the image.
    :return: np.ndarray if not return_dict else Tuple[np.ndarray, dict]]
    """
    data = _get_data_from_filepath(filepath)  # type: bytes
    image, header_dict = _read_edf_from_data(data, header_dict)  # type: np.ndarray, dict
    if return_dict:
        return image, header_dict
    else:
        return image


def _get_data_from_filepath(filepath: Union[Path, str, BytesIO]) -> bytes:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if isinstance(filepath, Path):
        if not filepath.is_file():
            raise FileNotFoundError(f'File {filepath} does not exist.')

        if filepath.suffix == '.edf':
            with open(str(filepath), 'rb') as f:
                return f.read()  # type: bytes
        elif filepath.suffix == '.gz':
            with gzip.open(str(filepath), 'rb') as f:
                return f.read()  # type: bytes
        else:
            raise TypeError(f'Unknown file extension {filepath.suffix}.')
    elif isinstance(filepath, BytesIO):
        return filepath.read()  # type: bytes
    else:
        raise TypeError(f'Unknown filepath type {type(filepath).__name__}')


def _read_edf_from_data(data: bytes, header_dict: dict = None):
    header_dict = header_dict or _read_header_from_data(data)
    try:
        header_end_index = header_dict['headerSize']
        image_size = int(header_dict['Size'])
        raw_image_data = data[header_end_index:header_end_index + image_size]  # type: bytes
        data_type = _get_numpy_type(header_dict['DataType'])
        image_shape = (int(header_dict['Dim_2']), int(header_dict['Dim_1']))
    except KeyError as err:
        raise CorruptedFileError(f'Header dict misses key: {err}')
    except IndexError as err:
        raise CorruptedFileError(f'Could not read edf file, header parameters may be wrong: {err}.')

    data = np.frombuffer(raw_image_data, data_type)  # type: np.ndarray
    data = np.rot90(np.reshape(data, image_shape))
    return data, header_dict


def _read_header_from_data(data: bytes) -> dict:
    header_end_index = data.find(b'}\n') + 2  # type: int
    if header_end_index == 1:  # data.find did not find b'}\n' symbols and returned -1
        raise CorruptedFileError(f'Could not find edf header.')
    header = data[1:header_end_index].decode('utf-8')  # type: str
    try:
        header_dict = _get_header_dict(header)  # type: dict
    except Exception as err:
        raise CorruptedFileError(f'Could not read edf header: {err}')
    header_dict['headerSize'] = header_end_index
    return header_dict


def _get_header_dict(header: str) -> dict:
    header_dict = {}
    raw_list = header.replace('\n', '').strip(). \
        replace(' ', ''). \
        replace('{', ''). \
        replace('}', ''). \
        split(';')  # type: List[str]
    for item in raw_list:
        items = item.split('=')  # type: List[str]
        if len(items) == 2:
            header_dict.update([items])
    return header_dict


_DICT_TYPES = {
    'SIGNEDBYTE': np.int8,  # "b"
    'UNSIGNEDBYTE': np.uint8,  # "B"
    'SIGNEDSHORT': np.int16,  # "h"
    'UNSIGNEDSHORT': np.uint16,  # "H"
    'SIGNEDINTEGER': np.int32,  # "i"
    'UNSIGNEDINTEGER': np.uint32,  # "I"
    'SIGNEDLONG': np.int32,  # "i"
    'UNSIGNEDLONG': np.uint32,  # "I"
    'SIGNED64': np.int64,  # "l"
    'UNSIGNED64': np.uint64,  # "L"
    'FLOATVALUE': np.float32,  # "f"
    'FLOAT': np.float32,  # "f"
    'DOUBLEVALUE': np.float64
}  # type: Dict[str, type]


def _get_numpy_type(edf_type: str) -> type:
    """
    Returns NumPy type based on edf type.
    """
    try:
        return _DICT_TYPES[edf_type.upper()]
    except KeyError:
        raise CorruptedFileError(f'Unknown edf type {edf_type}.')
