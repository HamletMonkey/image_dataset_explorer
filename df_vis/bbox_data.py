import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def bounding_box_data(ANN_PATH):

    """
    Takes in path to XML annotations folder (containing all annotation files) and extracts all bounding boxes information of an image.

    # Arguments
        ANN_PATH: path, annotation folder path

    # Returns
        df_bbox: pd.DataFrame, with columns: 'image_id','class','bbox_area','bbox_ar'
            'image_id': str, filename from annotation
            'class': str, bounding box class name
            'bbox_area': int, bounding box area
            'bbox_aspect_ratio': float, bounding box aspect ratio, round up to 2 decimal points
    """

    # get the file name from annotation folder
    allfiles = [f.parts[-1].split(".")[0] for f in Path(ANN_PATH).iterdir()]

    data = []
    for img in allfiles:
        tree = ET.parse(os.path.join(ANN_PATH, f"{img}.xml"))
        root = tree.getroot()
        for object in root.findall("object"):
            name = object.find("name").text
            for value in object.findall("bndbox"):
                xmin = int(value.find("xmin").text)
                ymin = int(value.find("ymin").text)
                xmax = int(value.find("xmax").text)
                ymax = int(value.find("ymax").text)
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin
                bbox_area = int(bbox_w * bbox_h)
                try:
                    bbox_aspect_ratio = round(bbox_w / bbox_h, 2)
                    values = [img, name, bbox_area, bbox_aspect_ratio]
                    data.append(values)
                except ZeroDivisionError:
                    # for extremely small bounding box, return the image id
                    print(
                        f'WARNING! \nimage_id "{img}" contains bounding box of height equals to zero: class--{name}, bounding box coordinates--{[xmin, ymin, xmax, ymax]}'
                    )
                    pass

    df_bbox = pd.DataFrame(
        data, columns=["image_id", "class", "bbox_area", "bbox_aspect_ratio"]
    )

    return df_bbox
