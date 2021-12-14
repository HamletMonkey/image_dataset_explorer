import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def cooccurence_mtrx_jaccard(ANN_PATH):

    """
    Takes in path to XML annotations folder (containing all annotation files) and creates a co-occurence matrix of different classes using Jaccard Similarity:

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
