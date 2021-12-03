# libraries
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from datetime import datetime
import argparse


def cooccurence_mtrx_jaccard(ANN_PATH):

    """
    Takes in path to XML annotations folder (containing all annotation files) and creates a co-occurence matrix of different classes using Jaccard Similarity:

    # Arguments
        ANN_PATH: path, annotation folder path

    # Returns
        df: pd.DataFrame, with items in 'class_list' as columns, in the form of one-hot encoding
        coocc_norm: pd.DataFrame, Jaccard Similarity co-occurence matrix
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

    return df, coocc_norm


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
                values = [img, name, int(bbox_w * bbox_h), round(bbox_w / bbox_h, 2)]
                data.append(values)

    df_bbox = pd.DataFrame(
        data, columns=["image_id", "class", "bbox_area", "bbox_aspect_ratio"]
    )

    return df_bbox


def create_save_vis(ANN_PATH):
    """
    Creates plots for dataset visualization based on files in XML annotations folder (containing all annotation files) and saves the visualization.

    # Arguments
        ANN_PATH: path, annotation folder path

    # Returns
        image dataset visualization with 6 subplots (2 rows 3 columns):
        1. Class distribution of image dataset
        2. Normalized co-occurence matrix (Jaccard similarity)
        3. Histogram of image aspect ratio
        4. Mean area of bounding box per class
        5. Aspect ratio of bounding box in image dataset
        6. Relative area (size) of bounding box to image (per class)

        df_bbox_comb: pd.DataFrame, with columns:
            'image_id': str, filename from annotation
            'class': str, bounding box class name
            'bbox_area': int, bounding box area
            'bbox_aspect_ratio': float, bounding box aspect ratio, round up to 2 decimal points
            'img_width': int, image width
            'img_height': int, image height
            'img_aspect_ratio': float, image aspect ratio, round up to 2 decimal points
            'img_area': int, area of image
            'rel_area': float, relative area of bounding box to area of image
            'rel_area_sqrt': float, square root of relative area

    """

    df, coocc_norm = cooccurence_mtrx_jaccard(ANN_PATH)
    df_hist_AR = aspect_ratio_histogram(ANN_PATH)
    df_bndbox = bounding_box_data(ANN_PATH)

    df_hist_AR_ = df_hist_AR.reset_index()
    cls_list = list(df.columns)

    # getting the mean area of bounding boxes per class
    df_bndbox_mean_area = df_bndbox.groupby("class")[["bbox_area"]].mean().reset_index()
    df_bndbox_mean_area.sort_values(by="bbox_area", ascending=False, inplace=True)

    # getting relative area
    df_bbox_comb = df_bndbox.merge(df_hist_AR_, on="image_id", how="left")
    df_bbox_comb["img_area"] = df_bbox_comb["img_width"] * df_bbox_comb["img_height"]
    df_bbox_comb["rel_area"] = df_bbox_comb["bbox_area"] / df_bbox_comb["img_area"]
    df_bbox_comb["rel_area_sqrt"] = np.sqrt(df_bbox_comb["rel_area"])

    # create plots!
    sns.set(font_scale=1.8)
    fig = plt.figure(figsize=(30, 18))
    # setting values to rows and column variables
    rows = 2
    columns = 3

    fig.add_subplot(rows, columns, 1)
    plt.bar(df.columns, df.sum())
    plt.title(f"Class Distribution- total {len(df)} images")
    plt.xlabel("class_name")
    plt.xticks(rotation=45)
    plt.ylabel("class_count")

    fig.add_subplot(rows, columns, 2)
    sns.heatmap(
        coocc_norm,
        annot=True,
        cmap="rocket_r",
        cbar=True,
        linewidth=1,
        linecolor="white",
    )
    plt.title("Normalized Co-occurence Matrix \n--using Jaccard Similarity")

    fig.add_subplot(rows, columns, 3)
    plt.hist(df_hist_AR["img_aspect_ratio"], bins=100, range=[0, 5], edgecolor="none")
    plt.title("Histogram of Image Aspect Ratios")
    plt.xlabel("aspect_ratio")
    plt.ylabel("image_count")

    fig.add_subplot(rows, columns, 4)
    plt.bar(df_bndbox_mean_area["class"], df_bndbox_mean_area["bbox_area"])
    plt.title("Mean Area of Bounding Box per Class")
    plt.xlabel("class")
    plt.xticks(rotation=45)
    plt.ylabel("mean_area")

    fig.add_subplot(rows, columns, 5)
    plt.hist(df_bndbox["bbox_aspect_ratio"], bins=650, edgecolor="none")
    plt.title("Aspect Ratio of Bounding Box in Dataset")
    plt.xlabel("ann_ar")
    plt.ylabel("count")

    # 10 different colour for max 10 different classes
    col_list = [
        "powderblue",
        "gold",
        "tomato",
        "limegreen",
        "turquoise",
        "plum",
        "sandybrown",
        "dodgerblue",
        "silver",
    ]

    fig.add_subplot(rows, columns, 6)
    for index, item in enumerate(cls_list):
        plt.hist(
            df_bbox_comb[df_bbox_comb["class"] == item]["rel_area_sqrt"],
            bins=80,
            edgecolor="none",
            alpha=0.4,
            color=col_list[index],
            label=item,
        )
    plt.title("Relative Area of Bounding Box to Image")
    plt.xlabel("rel_area_sqrt")
    plt.ylabel("bbox_count")
    plt.legend()

    fig.tight_layout()

    fig.savefig(f'dataset_visualisation-{datetime.today().strftime("%Y-%m-%d")}.png')

    return df_bbox_comb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_path",
        type=str,
        required=True,
        help="path to XML annotations folder",
    )
    args = parser.parse_args()

    create_save_vis(ANN_PATH=args.ann_path)
