import cv2
import numpy as np


def preprocessing(img, mode=None, l0_additionally=None, lp_blur_size=7):
    """Applies denoising filter, clahe and uses laplacian to increase contrast at edges of fibers

    :param img: image to enhance
    :param mode: (1): fast non-local means, (2): meadian blur, (3): gaussian blur
    :param l0_additionally: Whether to perform l0 smoothing after preprocessing
    :param lp_blur_size: size of deblurring applied to the laplacian
    :return: enhanced image
    """
    if mode is None:
        return img
    srcGray = img
    if mode == 1:
        denoiseGray = cv2.fastNlMeansDenoising(
            srcGray, None, 4, 9, 21
        )  # good at conserving contrast at edges of fibers
    if mode == 2:
        denoiseGray = cv2.medianBlur(srcGray, 7)
    if mode == 3:
        denoiseGray = cv2.GaussianBlur(srcGray, (7, 7), 0)
    if mode == 4:
        denoiseGray = cv2.ximgproc.l0Smooth(srcGray)

    if l0_additionally is not None:
        denoiseGray = cv2.ximgproc.l0Smooth(denoiseGray)

    imgLaplacian = cv2.Laplacian(denoiseGray, 0, 255, cv2.CV_32F, 3)

    clahe = cv2.createCLAHE(
        clipLimit=3, tileGridSize=(22, 22)
    )  # used to equalize uneven colored surfaces of fibers
    cl1 = clahe.apply(denoiseGray)
    sharp = np.float32(
        cl1
    )  # convert to higher precision float, to prevent information loss when subtracting laplacian
    denoiseLaplace1 = cv2.medianBlur(
        imgLaplacian, lp_blur_size
    )  # remove noise from laplacian
    # LoG = cv2.Laplacian(cv2.GaussianBlur(denoiseGray, (7,7), 0), 0, 255, cv2.CV_32F, 7)

    # beware, uint8 means clipping is applied by the unsharp masking
    imgResult = cv2.addWeighted(
        sharp.astype(np.uint8), 1.0, denoiseLaplace1, -1.0, 0
    )

    return (
        imgResult,
        {"denoised": denoiseGray, "CLAHE": cl1, "Laplace": imgResult},
    )
