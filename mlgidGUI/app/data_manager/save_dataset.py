from .saving_parameters import SavingParameters
from .pascal.protocol_implementations import BndBox, Object, Size
from .pascal.pascal_annotation import Annotation
from .save_h5 import _get_boxes_n_labels
from ..file_manager import  ImageKey
from .load_data import ImageData, ImageDataFlags, load_image_data
import xml.etree.ElementTree as xml
from tifffile import imwrite


def pretty_xml(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        pretty_xml(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('\t' * depth)
        else:
            parent[index - 1].tail = '\n' + ('\t' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('\t' * (depth - 1))



def save_for_object_detection(fm, image_holder, params: SavingParameters):
    output_folder = params.path / 'pascal-voc'
    output_folder.mkdir(parents=True, exist_ok=True)

    count: int = 0

    for _, image_keys in params.selected_images.items():
        for image_key in image_keys:
            if _save_image_for_object_detection(fm, image_holder, image_key, str(count), output_folder):
                count += 1

    return True


def _save_image_for_object_detection(fm, image_holder, image_key: ImageKey, name: str, output_folder) -> bool:
    image_data: ImageData = load_image_data(fm, image_holder,
        image_key, flags=ImageDataFlags.POLAR_IMAGE | ImageDataFlags.GEOMETRY | ImageDataFlags.ROI_DATA)

    if image_data.polar_image is None or image_data.roi_data is None or image_data.geometry is None:
        return False

    boxes, labels = _get_boxes_n_labels(image_data.roi_data.to_array(), image_data.geometry)

    obj = [Object(str(key),
        BndBox((round(box[0]), 4),(round(box[1]), 4), (round(box[2]), 4), (round(box[3]), 4),))
        for key, box in enumerate(boxes)]

    filename = str(image_holder.current_key)
    imwrite((output_folder /  (filename + '.tiff')) , image_data.polar_image)

    ann = Annotation(filename + '.tiff', obj, Size(image_data.polar_image.shape[1],image_data.polar_image.shape[0])).to_xml()
    tree = xml.ElementTree(ann)
    pretty_xml(tree.getroot())
    tree.write((output_folder / (filename + '.xml')))