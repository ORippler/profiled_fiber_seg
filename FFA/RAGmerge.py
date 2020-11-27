from skimage.future import graph
from skimage import measure
from skimage.filters import sobel
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def weight_boundary(graph, src, dst, n):
    """function used by merge_hierarchical to reevaluate weight of edges after merging two nodes.

    The distance between two segments and their relative size are taken into account when evaluating the weight.
    The closer, the better. Smaller segments rather get merged with bigger ones, than similar ones. Merging Segments
    that would result in a Segment greater than maxSegmentSize, will get disabled from merging.

    for additional info on parameters, see: https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/graph_merge.py#L59

    :param graph: internal parameter of merge_hierarchical
    :param src: internal parameter of merge_hierarchical
    :param dst: internal parameter of merge_hierarchical
    :param n: internal parameter of merge_hierarchical
    :return: new weight
    """
    default = {"weight": 0.0, "count": 0}

    count_src = graph[src].get(n, default)["count"]
    count_dst = graph[dst].get(n, default)["count"]

    weight_src = graph[src].get(n, default)["weight"]
    weight_dst = graph[dst].get(n, default)["weight"]

    count = count_src + count_dst
    return {
        "count": count,
        "weight": (count_src * weight_src + count_dst * weight_dst) / count,
    }


def before_mergeBoundary(graph, src, dst):
    """Called before merging two nodes. Updates new pixel count of merged nodes.

    for additional info on parameters, see: https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/graph_merge.py#L59

    :param graph: internal parameter of merge_hierarchical
    :param src: internal parameter of merge_hierarchical
    :param dst: internal parameter of merge_hierarchical
    :return:
    """
    pass


def weight_fiberSegmentsNewWeight(graph, src, dst, n):
    """function used by merge_hierarchical to reevaluate weight of edges after merging two nodes.

    The distance between two segments and their relative size are taken into account when evaluating the weight.
    The closer, the better. Smaller segments rather get merged with bigger ones, than similar ones. Merging Segments
    that would result in a Segment greater than maxSegmentSize, will get disabled from merging.

    for additional info on parameters, see: https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/graph_merge.py#L59

    :param graph: internal parameter of merge_hierarchical
    :param src: internal parameter of merge_hierarchical
    :param dst: internal parameter of merge_hierarchical
    :param n: internal parameter of merge_hierarchical
    :return: new weight
    """
    x = (
        graph.nodes[dst]["centroid"][0] - graph.nodes[n]["centroid"][0]
    )  # maybe y
    y = (
        graph.nodes[dst]["centroid"][1] - graph.nodes[n]["centroid"][1]
    )  # maybe x

    if graph.nodes[dst]["pixel count"] > graph.nodes[n]["pixel count"]:
        # wgt = graph.nodes[n]['pixel count']/graph.nodes[dst]['pixel count']
        wgt = maxSegmentSize / graph.nodes[dst]["pixel count"]
    else:
        # wgt = graph.nodes[dst]['pixel count']/graph.nodes[n]['pixel count']
        wgt = maxSegmentSize / graph.nodes[n]["pixel count"]

    estimatedMaxDist = math.sqrt(maxSegmentSize * 2)

    dist = math.sqrt(x * x + y * y)
    if dist > estimatedMaxDist:
        distWgt = (dist / estimatedMaxDist) ** 5
    else:
        distWgt = dist / estimatedMaxDist
    diff = distWgt * wgt

    if (
        graph.nodes[dst]["pixel count"] + graph.nodes[n]["pixel count"]
        > maxSegmentSize
    ):
        diff = 255
    # elif (dist>maxDistMerge):
    #    diff = 255

    return {"weight": diff}


def before_merge(graph, src, dst):
    """Called before merging two nodes. Updates new pixel count of merged nodes.

    for additional info on parameters, see: https://github.com/scikit-image/scikit-image/blob/master/skimage/future/graph/graph_merge.py#L59

    :param graph: internal parameter of merge_hierarchical
    :param src: internal parameter of merge_hierarchical
    :param dst: internal parameter of merge_hierarchical
    :return:
    """
    graph.nodes[dst]["pixel count"] += graph.nodes[src]["pixel count"]


def RAGmerge(markers, image, mode):
    """Uses original oversegmentized labelmap and original image to merge segments into more fitting ones.

    :param markers: original segmentation
    :param image: original image
    :param mode: 0:no merging, 1:custom weight, 2:ragBoundary with sobel, 3:ragBoundary with laplacian
    :return: merged segmentation
    """

    # explicitly creaty copy instead of reference to prevent accidental data corruption
    scikitMarkers = np.copy(markers)
    #    scikitMarkers[markers == -1] = 0 #border labels are -1, but scikit uses labels from 0 to N
    #
    #    print('Getting rid of label boundaries...')
    #    scikitMarkers = eliminateWatershedline(scikitMarkers) #remove border of segments

    print("Creating RAG...")
    if mode == 0:
        return scikitMarkers
    if mode == 1:
        g = graph.rag_mean_color(
            image, scikitMarkers, connectivity=1
        )  # use rag_mean_color to initialize rag/variables pixel count etc
        # g.remove_edges_from((list(g.edges(0))))  # remove edges instead of nodes to prevent compatibility issues
        g.remove_edges_from((list(g.edges(1))))
        # look for greatest number of pixels in segments to choose maxSegmentSize
        # save all pixel counts
        segments = []
        for n in g:
            segments.append([n, g.nodes[n]["pixel count"]])

        # sort them
        segments.sort(
            key=lambda x: x[1]
        )  # use number of pixels as key for sort algorithm
        global maxSegmentSize
        maxSegmentSize = (
            segments[-2][1] * 1.1
        )  # use nr of pixels of biggest segment (not background) +10% as maxSegmentsize
        # maxSegmentSize = np.percentile(np.array(segments), 90)*1.1
        offset = 1
        map_array = np.arange(scikitMarkers.max() + 1)
        for n, d in g.nodes(data=True):
            for label in d["labels"]:
                map_array[label] = offset
            offset += 1

        rag_labels = map_array[scikitMarkers]
        regions = measure.regionprops(rag_labels)

        for (n, data), region in zip(g.nodes(data=True), regions):
            data["centroid"] = tuple(map(int, region["centroid"]))

        # reevaluate weight of edges before merging (right now mean color is still used for weight)
        for nodeA, nodeB, d in g.edges(data=True):
            x = (
                g.nodes[nodeA]["centroid"][0] - g.nodes[nodeB]["centroid"][0]
            )  # maybe y
            y = (
                g.nodes[nodeA]["centroid"][1] - g.nodes[nodeB]["centroid"][1]
            )  # maybe x

            if g.nodes[nodeA]["pixel count"] > g.nodes[nodeB]["pixel count"]:
                # wgt = g.nodes[nodeB]['pixel count'] / g.nodes[nodeA]['pixel count']
                wgt = maxSegmentSize / g.nodes[nodeA]["pixel count"]
            else:
                # wgt = g.nodes[nodeA]['pixel count'] / g.nodes[nodeB]['pixel count']
                wgt = maxSegmentSize / g.nodes[nodeB]["pixel count"]

            estimatedMaxDist = math.sqrt(maxSegmentSize * 2)

            dist = math.sqrt(x * x + y * y)
            if dist > estimatedMaxDist:
                distWgt = (dist / estimatedMaxDist) ** 3
            else:
                distWgt = dist / estimatedMaxDist

            d["weight"] = distWgt * wgt
            if (
                g.nodes[nodeA]["pixel count"] + g.nodes[nodeB]["pixel count"]
                > maxSegmentSize
            ):
                d["weight"] = 255

        g_orig = g.copy()
        print("Merging RAG...")
        labelsCustom = graph.merge_hierarchical(
            scikitMarkers,
            g,
            thresh=2,
            rag_copy=False,
            in_place_merge=True,
            merge_func=before_merge,
            weight_func=weight_fiberSegmentsNewWeight,
        )

        plt.close("all")
        return labelsCustom, g, g_orig

    if mode == 2 or mode == 3:
        if mode == 2:
            edges = sobel(image)
        if mode == 3:
            imgLaplacian = cv2.Laplacian(image, 0, 255, cv2.CV_32F, 3)
            edges = cv2.medianBlur(imgLaplacian, 7) / 255

        g = graph.rag_boundary(scikitMarkers, edges, connectivity=1)
        g_orig = g.copy()

        g.remove_edges_from(list(g.edges(1)))

        print("Merging RAG...")
        g_orig = g.copy()
        labelsCustom = graph.merge_hierarchical(
            scikitMarkers,
            g,
            thresh=1,
            rag_copy=False,
            in_place_merge=True,
            merge_func=before_mergeBoundary,
            weight_func=weight_boundary,
        )

        plt.close("all")
        return labelsCustom, g, g_orig
