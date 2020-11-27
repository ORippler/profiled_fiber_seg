import cv2
import numpy as np


def eliminateWatershedline(labels):
    """ Replacing borderlabel '0' with greatest adjacent label

    :param labels: segmentation of image filled with integers
    :return: original segmentation with replaced border labels
    """
    dim = labels.shape
    for i in range(dim[0])[1:-2]:  # avoid border of image
        for j in range(dim[1])[1:-2]:  # avoid border of image
            if labels[i, j] == 0:  # looking for borderlabel
                options = labels[i - 1 : i + 1, j - 1 : j + 1]
                options = options.flatten("K")
                options.sort()
                labels[i, j] = options[
                    -1
                ]  # replace borderlabel with greatest labelvalue of neighbours
    for i in range(dim[0]):
        labels[i, 0] = 1
        labels[i, -1] = 1

    for j in range(1, dim[1] - 1):
        labels[0, j] = 1
        labels[-1, j] = 1
    return labels


def watershed(processedImg, markerSeed, unknown, centroids):
    """Performs seeded watershed using markers provided by "seed_generation"

    :param processedImg: preprocessed image, used to get initial segmentation
    :param path: if provided, image with marked seeds will be saved
    :return: initial segmentation
    """

    markerSeed = markerSeed + 1  # makes space for new value 0
    markerSeed[
        unknown == 255
    ] = 0  # 0 is used as key to watershed function. Spaces filled with 0 will be evaluated

    imgResultRGB = cv2.cvtColor(
        processedImg, cv2.COLOR_GRAY2RGB
    )  # Watershed uses RGB

    markers = cv2.watershed(imgResultRGB, markerSeed)

    markers_no_border = np.copy(markers)

    markers_no_border[
        markers == -1
    ] = 0  # border labels are -1, but scikit uses labels from 0 to N
    print("Getting rid of label boundaries...")
    markers_no_border = eliminateWatershedline(markers_no_border)

    return markers_no_border, markers
