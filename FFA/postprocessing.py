import numpy as np
import cv2
from tqdm import tqdm
import copy


def size_exclusion(mask, threshold=4000):

    if isinstance(mask, np.ndarray):
        labels = np.unique(mask)[1:]
        mask_new = np.zeros_like(mask)

    elif isinstance(mask, list):
        labels = mask
        mask_new = []

    for idx, label in enumerate(tqdm(labels)):
        if isinstance(mask, np.ndarray):
            binary = mask == label
            if binary.sum() > threshold:
                mask_new[binary] = label
        elif isinstance(mask, list):
            binary = label.astype(np.uint8)
            if binary.sum() >= threshold:
                mask_new.append(label)

    return mask_new
