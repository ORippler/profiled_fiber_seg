import cv2
import numpy as np
from tqdm import tqdm
from .utils import create_bbox


def grabcut(processedImg, sure_fg, unknown):

    if len(processedImg.shape) == 2:
        shape = processedImg.shape
        processedImg = cv2.cvtColor(processedImg, cv2.COLOR_GRAY2RGB)

    mask = np.full(shape, 2, dtype=np.uint8)

    mask[sure_fg] = 1
    sure_bg = np.invert(sure_fg.astype(bool) | unknown.astype(bool))
    mask[sure_bg] = 0

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, 450, 290)

    new_mask, bgdModel, fgdModel = cv2.grabCut(
        processedImg, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
    )
    new_mask = np.where((new_mask == 2) | (new_mask == 0), 0, 1).astype(
        "uint8"
    )

    return new_mask


def grabcut_all(processed_img, markerSeed, stats, unknown, centroids):
    gc_all = []
    for element in range(np.max(markerSeed)):
        if element not in np.unique(markerSeed):
            print(element)
    for idx in tqdm(range(np.max(markerSeed))):
        if idx > 0:
            gc = np.zeros_like(markerSeed)
            s = stats[idx]
            c = centroids[idx]
            msk = markerSeed == idx
            box = create_bbox(s, c)
            cut = grabcut(
                processed_img[box[2] : box[3], box[0] : box[1]],
                msk[box[2] : box[3], box[0] : box[1]],
                unknown[box[2] : box[3], box[0] : box[1]],
            )

            # maybe perform erosion before, and dilation after ?
            num_cpts, cpts = cv2.connectedComponents(cut, connectivity=4)
            for i in range(1, num_cpts):
                cpt = cpts == i
                if np.sum(
                    cpt & msk[box[2] : box[3], box[0] : box[1]].astype(bool)
                ):
                    gc[box[2] : box[3], box[0] : box[1]] += cpt
            gc_all.append(gc)
    return gc_all, None
