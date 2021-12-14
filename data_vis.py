# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from df_vis.coocc_mtrx_jaccard import cooccurence_mtrx_jaccard
from df_vis.img_ar_hist import aspect_ratio_histogram
from df_vis.bbox_data import bounding_box_data


def create_save_vis(ANN_PATH, title=None):
    """
    Creates plots for dataset visualization based on files in XML annotations folder (containing all annotation files) and saves the visualization.

    # Arguments
        ANN_PATH: path, annotation folder path
        title: str, visualization plot main title, default=None

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

    df, coocc_norm, _ = cooccurence_mtrx_jaccard(ANN_PATH)
    df_hist_AR = aspect_ratio_histogram(ANN_PATH)
    df_bndbox = bounding_box_data(ANN_PATH)

    df_hist_AR_ = df_hist_AR.reset_index()

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
    # figure main title
    if title != None:
        fig.suptitle(f'{title} -- as of {datetime.today().strftime("%Y-%m-%d")}')
    # setting values to rows and column variables
    rows = 2
    columns = 3

    fig.add_subplot(rows, columns, 1)
    sorted_df = df.sum().sort_values(ascending=False)
    plt.bar(sorted_df.index, sorted_df.values)
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
    plt.xlabel("bbox_ar")
    plt.ylabel("count")

    # 10 different colour for max 10 different classes
    col_list = [
        "dodgerblue",
        "gold",
        "tomato",
        "limegreen",
        "turquoise",
        "plum",
        "sandybrown",
        "powderblue",
        "silver",
    ]

    fig.add_subplot(rows, columns, 6)
    for index, item in enumerate(sorted_df.index):
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
    print("saving visualization plot..")
    fig.savefig(
        f'dataset_visualisation-{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}.png'
    )
    print("visualization plot saved!")

    return df_bbox_comb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_path",
        type=str,
        required=True,
        help="path to XML annotations folder",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="main title of visualization plot",
    )
    args = parser.parse_args()

    create_save_vis(ANN_PATH=args.ann_path, title=args.title)
