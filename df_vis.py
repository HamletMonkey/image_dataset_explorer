import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def cooccurence_mtrx_jaccard(ANN_PATH):

    """
    Takes in path to XML annotations folder (containing all annotation files) and creates a co-occurence matrix of different classes using Jaccard Similarity.
    Also useful to filter images based on object class and obtain the list of class existing in the image dataset.

    # Arguments
        ANN_PATH: path, annotation folder path

    # Returns
        df: pd.DataFrame, with items in 'class_list' as columns, in the form of one-hot encoding
        coocc_norm: pd.DataFrame, Jaccard Similarity co-occurence matrix
        class_list: list, list of class in dataset
    """

    # get the file names from annotation folder
    allfiles = [f.parts[-1].split(".")[0] for f in Path(ANN_PATH).iterdir()]
    columns = ["image_id"]

    # dataframe for class co-occurence per image
    df = pd.DataFrame(columns=columns)
    df["image_id"] = allfiles
    df.set_index("image_id", inplace=True)

    # update the df with co-occurence information
    for ann in allfiles:
        tree = ET.parse(os.path.join(ANN_PATH, f"{ann}.xml"))
        root = tree.getroot()
        result = set()
        for object in root.findall("object"):
            name = object.find("name").text
            result.add(name)
        # print(result)
        for name in result:
            df.loc[ann, name] = 1
    df = df.fillna(0)
    df = df.astype(np.int32)

    # list of class
    class_list = list(df.columns)

    # create the co-occurence matrix
    coocc_df = df.T.dot(df)
    # total images in a class
    class_count = dict(zip(coocc_df.index, np.diag(coocc_df)))
    # normalizing the co-occurence matrix using Jaccard similarity
    coocc_norm = coocc_df.copy()
    for col in coocc_norm:
        for index in coocc_norm.index:
            # print(index, col)
            if index == col:
                coocc_norm.loc[index, col] = round(
                    coocc_norm.loc[index, col] / coocc_norm.loc[index, col], 2
                )
            else:
                coocc_norm.loc[index, col] = round(
                    coocc_norm.loc[index, col]
                    / (class_count[index] + class_count[col]),
                    2,
                )

    return df, coocc_norm, class_list


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
