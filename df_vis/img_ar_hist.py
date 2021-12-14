import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def aspect_ratio_histogram(ANN_PATH):

    """
    Takes in path to XML annotations folder (containing all annotation files) and extracts the width, height of images to compute the image aspect ratio.
    The output dataframe can be used to plot histogram of image's aspect ratio.

    # Arguments
        ANN_PATH: path, annotation folder path

    # Returns
        df_imageAR: pd.DataFrame, with columns: 'image_id','img_width','img_height','img_aspect_ratio'
            'image_id': str, filename from annotation
            'img_width': int, image width
            'img_height': int, image height
            'img_aspect_ratio': float, image aspect ratio, round up to 2 decimal points
    """

    # get the file name from annotation folder
    allfiles = [f.parts[-1].split(".")[0] for f in Path(ANN_PATH).iterdir()]
    df_imageAR = pd.DataFrame(
        columns=["image_id", "img_width", "img_height", "img_aspect_ratio"]
    )
    df_imageAR["image_id"] = allfiles
    # initialize with 0 value
    (
        df_imageAR["img_width"],
        df_imageAR["img_height"],
        df_imageAR["img_aspect_ratio"],
    ) = [
        0,
        0,
        0,
    ]
    df_imageAR.set_index("image_id", inplace=True)

    for img in allfiles:
        tree = ET.parse(os.path.join(ANN_PATH, f"{img}.xml"))
        root = tree.getroot()
        for object in root.findall("size"):
            w = int(object.find("width").text)
            h = int(object.find("height").text)
            ar = round(w / h, 2)
            df_imageAR.loc[img, "img_width"] = w
            df_imageAR.loc[img, "img_height"] = h
            df_imageAR.loc[img, "img_aspect_ratio"] = ar

    return df_imageAR
