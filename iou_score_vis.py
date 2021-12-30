import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

# original function to calculate iou score between detection and groundtruth
def score_iou(det_bbox, gt_bbox):
    """
    Compute IoU between a single bboxes

    Arguments:
        det_bbox (ndarray): detection bbox in xyxy format
        gt_bbox (ndarray): ground truth bbox in xyxy format format
    Returns:
        float: IoU score
    """
    ixmin = np.maximum(gt_bbox[0], det_bbox[0])
    iymin = np.maximum(gt_bbox[1], det_bbox[1])
    ixmax = np.minimum(gt_bbox[2], det_bbox[2])
    iymax = np.minimum(gt_bbox[3], det_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (det_bbox[2] - det_bbox[0] + 1.0) * (det_bbox[3] - det_bbox[1] + 1.0)
        + (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0)
        - inters
    )

    return inters / uni


def iou_score_plot(XML_PATH):
    """
    Visualizing the IoU score for bounding boxes pair in images, to find out possible overlapping of annotations on same object.

    # Arguments
        XML_PATH: path, annotation folder path

    # Returns
        df_sort: pd.DataFrame, with columns: 'image_id','bbox_coord_pair','bbox_class_pair','iou_score'
            'image_id': str, filename from annotation
            'bbox_coord_pair': list, pair of bounding box coordinations (xmin, ymmin, xmax, ymax)
            'bbox_class_pair': list, pair of object class corresponding to the coordinations
            'iou_score': float, IoU score for corresponding pair of bounding box coordinations
    """

    raw_xml = [f.parts[-1].split(".")[0] for f in Path(XML_PATH).iterdir()]
    print(f"Number of XML files: {len(raw_xml)}")

    # dictionary to store iamge_id with its diff bounding box combinations:
    raw_d = {}
    for item in raw_xml:  # read the xml file
        result = []
        tree = ET.parse(os.path.join(XML_PATH, f"{item}.xml"))
        root = tree.getroot()
        for object in root.findall("object"):
            # including the class name as well
            name = object.find("name").text
            ymin = int(object.find("bndbox/ymin").text)
            xmin = int(object.find("bndbox/xmin").text)
            ymax = int(object.find("bndbox/ymax").text)
            xmax = int(object.find("bndbox/xmax").text)
            result.append([name, xmin, ymin, xmax, ymax])
        # only take images with more than one bounding box
        if len(result) > 1:
            raw_d[item] = [x for x in combinations(result, 2)]

    print(f"Number of images with more than 1 bounding box: {len(raw_d)}")

    # dictionary to store image_id with IoU score of its diff bounding box combinations
    iou_score_d = {}
    for k, v in raw_d.items():
        score_list = []
        for item in v:
            x, y = item
            score_list.append(score_iou(x[1:], y[1:]))
            iou_score_d[k] = score_list

    # creating dataframe from raw_d and iou_score_d
    df_coord = pd.DataFrame(
        list(raw_d.items()), columns=["image_id", "bbox_coord_pair"]
    ).explode("bbox_coord_pair")
    df_iou = pd.DataFrame(
        list(iou_score_d.items()), columns=["image_id", "iou_score"]
    ).explode("iou_score")

    # concatenating both dataframes
    assert df_coord["image_id"].equals(
        df_iou["image_id"]
    ), "image_id of both dataframes does not match!"
    df = pd.concat([df_coord, df_iou["iou_score"]], axis=1)
    df["iou_score"] = df["iou_score"].astype("float32")

    # extracting the class pair
    df["bbox_class_pair"] = df["bbox_coord_pair"].apply(lambda x: [x[0][0], x[1][0]])
    df["bbox_coord_pair"] = df["bbox_coord_pair"].apply(lambda x: [x[0][1:], x[1][1:]])
    df = df[["image_id", "bbox_coord_pair", "bbox_class_pair", "iou_score"]]

    # for bounding box pair with IoU score > 0
    df_sort = df.sort_values(
        by=["iou_score", "image_id"], ascending=[False, True], ignore_index=True
    )
    df_sort_largethan_0 = df_sort[df_sort["iou_score"] > 0]

    # for maximum score of IoU per image
    df_max_score = (
        df.groupby(["image_id"])[["iou_score"]]
        .max()
        .sort_values(by=["iou_score"], ascending=False)
        .reset_index()
    )

    # plotting bar plot
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    ax[0].bar(
        df_sort_largethan_0.index, df_sort_largethan_0["iou_score"], edgecolor=None
    )
    ax[0].set_yticks(np.arange(0, 1, 0.05))
    ax[0].set_title(f"IoU Score per bounding boxes pair -- greater than 0", fontsize=14)
    ax[0].set_xlabel("bbox_pair", fontsize=12)
    ax[0].set_ylabel("iou_score", fontsize=12)
    ax[0].grid(True)

    ax[1].bar(df_max_score.index, df_max_score["iou_score"], edgecolor=None)
    ax[1].set_yticks(np.arange(0, 1, 0.05))
    ax[1].set_title(
        f"Max IoU score per image - total {len(df_max_score)} images", fontsize=14
    )
    ax[1].set_xlabel("image_index", fontsize=12)
    ax[1].set_ylabel("max_iou_score", fontsize=12)
    ax[1].grid(True)

    fig.tight_layout()
    plt.show()

    return df_sort


def iou_inter_coord(first_bbox, second_bbox):
    ixmin = np.maximum(second_bbox[0], first_bbox[0])
    iymin = np.maximum(second_bbox[1], first_bbox[1])
    ixmax = np.minimum(second_bbox[2], first_bbox[2])
    iymax = np.minimum(second_bbox[3], first_bbox[3])
    result = [ixmin, iymin, ixmax, iymax]
    return result
