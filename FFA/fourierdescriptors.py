import cv2
import numpy
from skimage import measure
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from spatial_efd import spatial_efd
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import collections


def extract_cnts_coeffs_from_mask(mask, harmonic=10, size_invariant=True):
    coeffs = collections.OrderedDict()
    cnts = collections.OrderedDict()
    rots = collections.OrderedDict()

    if isinstance(mask, np.ndarray):
        labels = np.unique(mask)[1:]
    elif isinstance(mask, list):
        labels = mask

    for idx, label in enumerate(tqdm(labels)):
        if isinstance(mask, np.ndarray):
            binary = (mask == label).astype(np.uint8)
        elif isinstance(mask, list):
            binary = label.astype(np.uint8)

        (
            num_labels,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(binary, connectivity=4)

        if len(np.unique(labels)) == 2:
            # make area size flexible
            contours, hierarchies = cv2.findContours(
                labels.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            if not np.all(hierarchies, -1).all():
                print(
                    "hierarchical contours found for {}. Only selecting the outermost contours".format(
                        idx
                    )
                )
                # continue
            for n, (contour, hierarchy) in enumerate(
                zip(contours, hierarchies)
            ):
                if hierarchy[-1][-1] == -1:
                    contour = contour.squeeze()
                    if len(contour.shape) == 2:
                        raw = spatial_efd.CalculateEFD(
                            contour[:, 0], contour[:, 1], harmonics=harmonic
                        )
                        normalized, rotation = spatial_efd.normalize_efd(
                            raw, size_invariant=size_invariant
                        )
                        coeffs[idx] = normalized
                        cnts[idx] = contour
                        rots[idx] = rotation
                    else:
                        print(
                            "shape only 1 pixel long, skipping {}".format(idx)
                        )
                else:
                    pass
                    # print(hierarchy)
        else:
            print("more than one compoment found, skipping {}".format(idx))
    return coeffs, cnts, rots


def reduce_cnts_based_on_coeffs(
    coeffs: list, cnts: list, percentile=75, plot=True
) -> list:
    avgcoeffs = spatial_efd.AverageCoefficients(coeffs)
    SDcoeffs = spatial_efd.AverageSD(coeffs, avgcoeffs)

    if plot:
        median = np.median(np.array(coeffs), axis=0)

        x_med, y_med = spatial_efd.inverse_transform(median, harmonic=10)
        x_avg, y_avg = spatial_efd.inverse_transform(avgcoeffs, harmonic=10)
        x_sd, y_sd = spatial_efd.inverse_transform(SDcoeffs, harmonic=10)

        ax = spatial_efd.InitPlot()
        spatial_efd.PlotEllipse(ax, x_avg, y_avg, color="w", width=2.0)
        spatial_efd.PlotEllipse(ax, x_med, y_med, color="b", width=2.0)

        # Plot avg +/- 1 SD error ellipses
        spatial_efd.PlotEllipse(
            ax, x_avg + x_sd, y_avg + y_sd, color="r", width=1.0
        )
        spatial_efd.PlotEllipse(
            ax, x_avg - x_sd, y_avg - y_sd, color="r", width=1.0
        )

        plt.close("all")

    arr = np.array(coeffs)
    reshaped = np.reshape(arr, [arr.shape[0], -1])
    MCD = MinCovDet()
    MCD.fit(reshaped)

    a = MCD.mahalanobis(reshaped)

    if plot:
        plt.boxplot(a)
        plt.show()
        plt.close("all")

    percentile = np.percentile(a, percentile)

    reduced = list(np.array(coeffs)[a < percentile])

    avgcoeffs = spatial_efd.AverageCoefficients(reduced)
    SDcoeffs = spatial_efd.AverageSD(reduced, avgcoeffs)

    median = np.median(np.array(reduced), axis=0)

    x_med, y_med = spatial_efd.inverse_transform(median, harmonic=10)
    x_avg, y_avg = spatial_efd.inverse_transform(avgcoeffs, harmonic=10)
    x_sd, y_sd = spatial_efd.inverse_transform(0.1 * SDcoeffs, harmonic=10)

    if plot:
        ax = spatial_efd.InitPlot()
        spatial_efd.PlotEllipse(ax, x_avg, y_avg, color="w", width=2.0)
        spatial_efd.PlotEllipse(ax, x_med, y_med, color="b", width=2.0)

        # Plot avg +/- 1 SD error ellipses
        spatial_efd.PlotEllipse(
            ax, x_avg + x_sd, y_avg + y_sd, color="r", width=1.0
        )
        spatial_efd.PlotEllipse(
            ax, x_avg - x_sd, y_avg - y_sd, color="r", width=1.0
        )

        i = 10
        plt.figure()
        ax = plt.gca()
        spatial_efd.plotComparison(
            ax,
            coeffs[i],
            10,
            cnts[i][:, 0],
            cnts[i][:, 1],
            color1="w",
            rotation=rots[i],
        )
        plt.show()
        plt.close("all")

    reduced_cnts = np.array(cnts)[a < percentile]

    return reduced_cnts, a < percentile
