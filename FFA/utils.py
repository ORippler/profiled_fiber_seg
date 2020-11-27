from matplotlib import pyplot as plt
import numpy as np
from functools import partial
from skimage.color import gray2rgb


def plot_fiber_results(img, img_preprocessed, markerSeed, segments, path):
    all_seg = segments
    if isinstance(all_seg, list):
        _all_seg = np.zeros_like(all_seg[0])
        for idx, seg in enumerate(all_seg):
            _all_seg[seg.astype(bool)] = idx
        all_seg = _all_seg

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(100, 40))

    from skimage.color import label2rgb

    label2rgb = partial(label2rgb, image=np.copy(img))

    row_1 = [
        gray2rgb(img),
        markerSeed,
        all_seg,
    ]
    titles_1 = ["image", "seeds", "final_seg"]

    row_2 = [
        gray2rgb(img_preprocessed),
        label2rgb(markerSeed),
        label2rgb(all_seg),
    ]
    titles_2 = [
        "image_preprocessed",
        "",
        "",
    ]

    for ax, img, title in zip(axs[0], row_1, titles_1):
        ax.imshow(img)
        ax.set_title(title, fontsize=40)
        ax.set_axis_off()

    for ax, img, title in zip(axs[1], row_2, titles_2):
        ax.imshow(img)
        ax.set_title(title, fontsize=40)
        ax.set_axis_off()

    fig.savefig(path)
    plt.close(fig)


def create_bbox(stats, centroids):
    x_min, x_max = 0, 2048
    y_min, y_max = 0, 1536

    x_orig, y_orig, x_len, y_len = list(stats)[:-1]

    centroid_x, centroid_y = centroids

    if x_len < 70:
        x_len = 70
    if y_len < 70:
        y_len = 70

    bbox = [
        centroid_x - 1.5 * x_len,
        centroid_x + 1.5 * x_len,
        centroid_y - 1.5 * y_len,
        centroid_y + 1.5 * y_len,
    ]

    bbox_corrected = []

    for i, element in enumerate(zip(bbox, (x_min, x_max, y_min, y_max))):
        if not i % 2:
            bbox_corrected.append(
                int(element[0]) if element[0] > element[1] else int(element[1])
            )
        else:
            bbox_corrected.append(
                int(element[0]) if element[0] < element[1] else int(element[1])
            )
    return bbox_corrected
