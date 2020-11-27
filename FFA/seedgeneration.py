import cv2
from skimage.morphology import remove_small_holes
import skimage.morphology
from skimage.feature import peak_local_max
import numpy as np
import networkx as nx
import math


def select_maxValue(g, threshold):
    """
    Uses graph g to find maximum node within a fiber. Works in-place and deletes nodes of lower value within a fiber
    :param g: grap:deadh of seedPoints. edge between a, b consists of lowest distTrans value on direct path between a and b
    :param threshold: nodes connected by edge with value above threshold are considered to be within the same fiber
    :return: nothing
    """

    toDelete = set()  # within sets no duplicates occur
    for (u, v, w) in g.edges.data("weight"):  # nodes u,v and edge weight w
        if w > threshold:  # nodes u,v within same fiber
            if g.nodes[u]["weight"] > g.nodes[v]["weight"]:
                toDelete.add(v)
            else:
                toDelete.add(u)
    for item in toDelete:
        g.remove_node(item)


def create_edges(coords, maxDist, img, graph, distThreshold):
    """
    connects nodes of graph and assigns a weight according to the lowest value of img allong the path between two nodes.
    Only evaluates edges of nodes if within maxDist of one another.
    :param coords: list of coordinates for nodes of graph
    :param maxDist: max distance between nodes for them to still be considered candidates for being within a fiber
    :param img: distance transform of image
    :param graph: graph of seedPoints
    :param distThreshold: nodes connected by an edge of value lower than distThreshold are considered to be within separate fibers
    :return:
    """

    for neighborA in coords:
        for neighborB in coords:
            u = (str(neighborA[0]), str(neighborA[1]))
            v = (str(neighborB[0]), str(neighborB[1]))
            graph.add_edge(
                u,
                v,
                weight=min_pathValue(
                    neighborA, neighborB, img, maxDist, distThreshold
                ),
            )


def min_pathValue(
    pointA, pointB, valueImg, maxDist, distThreshold, stepLength=3
):
    """
    Selects the lowest value of valueImg among a direct path between point A and B when considering every stepLength value
    :param pointA: coordinate of point A
    :param pointB: coordinate of point B
    :param valueImg: image used to select lowest value among path between A and B
    :param maxDist: maximum distance between points for them to be evaluated. Assigns 0 if not evaluated
    :param distThreshold: nodes connected by an edge of value lower than distThreshold are considered to be within separate fibers
    :param stepLength: every stepLength value among path between A and B will be considered for lowest value
    :return: lowest value among path between A and B
    """
    if pointA[0] == pointB[0] and pointA[1] == pointB[1]:
        return 0
    if pointA[1] > pointB[1]:
        tmp = pointA
        pointA = pointB
        pointB = tmp

    path = pointB - pointA
    length = math.sqrt(path[0] ** 2 + path[1] ** 2)
    if length > maxDist:
        return 0
    y_valsFloat = np.linspace(
        pointA[1], pointB[1], int(abs(path[0]) / stepLength)
    )  # float values will need to be converted to int indices
    y_vals = [
        int(y + 0.5) for y in y_valsFloat
    ]  # use nearest int instead of float
    if pointA[0] > pointB[0]:
        x_vals = [reversed(range(pointB[0], pointA[0], stepLength))]
    else:
        x_vals = [range(pointA[0], pointB[0], stepLength)]

    steps = list(zip(*x_vals, y_vals))  # * used to evaluate range list

    min = 255
    for step in steps:
        val = valueImg[step[0]][step[1]]
        if val < min:
            min = val
            if min < distThreshold:
                break
    return min


def seed_generation(
    processedImg,
    mode,
    dilation=None,
    threshold=None,
    path="",
    maxfilterSize=12,
    distThreshold=5,
    neighborDist=220,
):
    """Uses otsu thresholding as basis. From that binary image, fore- and background will be determined via
    morphological opening and dilation. Space that is neither fore- nor background will be marked as unknown.
    This unknown space will be explored by watershed to get an initial segmentation
    :param maxfilterSize: min distance between local maxima of distance transform
    :param distThreshold: used to select nodes via nodeWeight and edgeWeight for being considered within the same fiber
    :param neighborDist: maximum distance between nodes to be considered for being within the same fiber
    """
    _, bwOtsu = cv2.threshold(
        processedImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )  # use 0 for auto threshold

    opening = cv2.morphologyEx(
        bwOtsu, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
    )  # will later be used as seed

    closedHoles = remove_small_holes(
        opening, area_threshold=350, connectivity=1
    )

    sure_bg = cv2.dilate(
        opening,  # closedHoles.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=3,
    )  # space that will not be part of segments

    dist_transform = cv2.distanceTransform(
        closedHoles.astype("uint8"), cv2.DIST_L2, 5
    )

    if mode == 1:
        sure_fg = cv2.normalize(
            dist_transform,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )  # to enable relative thresholding

        _, sure_fg = cv2.threshold(
            sure_fg, threshold, 1, cv2.THRESH_BINARY
        )  # points 35% furthest from border of foreground are used as seed

        sure_fg = np.copy(
            sure_fg.astype("uint8") * 255
        )  # converting binary image to grayscale

    elif mode == 2:
        coordinates = peak_local_max(
            dist_transform, min_distance=maxfilterSize
        ).tolist()

        seedPoints = nx.Graph()
        toDelete = []
        for coord in coordinates:
            u = (
                str(coord[0]),
                str(coord[1]),
            )  # string tuples are hashable. Needed to be viable as key
            weight = dist_transform[coord[0]][coord[1]]
            if weight > distThreshold:
                seedPoints.add_node(u, weight=weight)
            else:
                toDelete.append(coord)
        for item in toDelete:
            coordinates.remove(item)
        coordinates = np.asarray(coordinates)

        create_edges(
            coordinates,
            neighborDist,
            dist_transform,
            seedPoints,
            distThreshold,
        )

        select_maxValue(seedPoints, distThreshold)

        sure_fg = np.zeros(processedImg.shape)

        for coord in seedPoints:
            sure_fg[int(coord[0])][int(coord[1])] = 1
        sure_fg = np.copy(
            skimage.morphology.binary_dilation(
                sure_fg, skimage.morphology.disk(5)
            ).astype("uint8")
            * 255
        )
    else:
        raise ValueError

    if dilation is not None:
        sure_fg = cv2.dilate(
            sure_fg, np.ones((dilation, dilation), np.uint8), iterations=1
        )

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markerSeed, stats, centroids = cv2.connectedComponentsWithStats(
        sure_fg, connectivity=4
    )  # assigns distinct value to every unconnected component

    return ret, markerSeed, stats, centroids, unknown
