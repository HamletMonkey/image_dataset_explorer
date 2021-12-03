# Image Dataset Explorer
To visualize the distribution of image datasets compiled.

The result returned is an image with 6 subplots (2 rows 3 columns) containing:
1. Class distribution in image dataset
2. Normalized co-occurence matrix (Jaccard similarity)
3. Histogram of image aspect ratio
4. Mean area of bounding box per class
5. Aspect ratio of bounding box in image dataset
6. Square root of relative area (size) of bounding box to image (per class)

## Running of Script
Run main python file in terminal:
`python data_vis.py --ann_path ./path/to/annotations`

The result image will be saved in working directory.