import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt


def img_xml_plot(IMG_PATH, ANN_PATH, class_list, rows=1, columns=1):
    """
    Provides a quick visualisation of image with bounding box(es).

    # Arguments
        IMG_PATH: path, image folder path
        ANN_PATH: path, annotation folder path
        class_list: list, list of object class, maximum of 10 classes
        rows: int, number of rows of image subplots
        columns: int, number of columns of image subplots

    # Returns
        A matplotlib plot of the image with bounding box(es)

    """
    # get file IDs
    filename = [f.parts[-1].split(".")[0] for f in Path(ANN_PATH).iterdir()]
    rand_filename = random.sample(filename, int(rows * columns))
    print(rand_filename)
    # 10 different colours for max 10 list
    colour_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 127, 0),
        (7, 185, 252),
        (204, 204, 255),
        (153, 153, 255),
    ]
    colour_combi = dict(zip(class_list, colour_list[: len(class_list)]))
    # sample of rand_filename WindowsPath('person_bags_ann/COCO_10014.xml')
    fig = plt.figure(figsize=(20, 20))
    for index, file in enumerate(rand_filename):
        # tree = ET.parse(file)
        tree = ET.parse(os.path.join(ANN_PATH, f"{file}.xml"))
        root = tree.getroot()
        result = (
            []
        )  # store different bbox and object class within the same file (image)
        for object in root.findall("object"):
            name = object.find("name").text
            for value in object.findall("bndbox"):
                xmin = int(float(value.find("xmin").text))
                ymin = int(float(value.find("ymin").text))
                xmax = int(float(value.find("xmax").text))
                ymax = int(float(value.find("ymax").text))
            result.append([name, xmin, ymin, xmax, ymax])
        # read the respective image
        img = cv2.imread(os.path.join(IMG_PATH, f"{file}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(result)):
            start = tuple(result[i][1:3])
            end = tuple(result[i][3:])
            img = cv2.rectangle(img, start, end, colour_combi[result[i][0]], 2)
            img = cv2.putText(
                img,
                result[i][0],
                start,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colour_combi[result[i][0]],
                2,
            )
        fig.add_subplot(rows, columns, index + 1)
        plt.imshow(img)
        plt.title(f"{file}")
    fig.tight_layout()
    plt.show()
