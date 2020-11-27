import cv2
import os
import sys
import numpy as np
import pandas as pd
from FFA.postprocessing import size_exclusion
from FFA.preprocessing import preprocessing
from FFA.watershed import watershed
from FFA.seedgeneration import seed_generation
from FFA.grabcut import grabcut_all
from FFA.RAGmerge import RAGmerge
from FFA.fourierdescriptors import (
    extract_cnts_coeffs_from_mask,
    reduce_cnts_based_on_coeffs,
)
from FFA.utils import plot_fiber_results
from tqdm import tqdm
import timeit
import yaml


class segment_pipeline:
    def __init__(self, srcPath=""):
        self.fpath = srcPath

    @staticmethod
    def discover_images(srcPath=""):

        image_list = []
        for file_entry in os.scandir(srcPath):
            if file_entry.name.endswith(".tif") or file_entry.name.endswith(
                ".png"
            ):
                image_list.append(file_entry.path)

        return image_list

    @staticmethod
    def preprocess_img(img, settings):
        return preprocessing(img, **settings)

    @staticmethod
    def generate_seeds(processedimg, settings):
        return seed_generation(processedimg, **settings)

    @staticmethod
    def generate_segments(
        processedimg, seeds, stats, unknown, centroids, setting
    ):
        if "watershed" in setting:
            return watershed(processedimg, seeds, unknown, centroids,)

        elif "grabcut" in setting:
            return grabcut_all(processedimg, seeds, stats, unknown, centroids,)

    @staticmethod
    def merge_segments(segments, image, mode):
        return RAGmerge(segments, image, mode)

    @staticmethod
    def size_exclusion(segments, threshold=4000):
        return size_exclusion(segments, threshold=threshold)

    @staticmethod
    def fourier_postprocessing(segments, settings):
        coeffs, cnts, rots = extract_cnts_coeffs_from_mask(segments)
        coeffs_list = [coeff for coeff in coeffs.values()]
        cnts_list = [cnt for cnt in cnts.values()]
        _, arrays = reduce_cnts_based_on_coeffs(
            coeffs_list, cnts_list, **settings
        )

        if isinstance(segments, np.ndarray):
            seg_reduced = np.zeros_like(segments)
        elif isinstance(segments, list):
            seg_reduced = []
        else:
            raise TypeError
        for key, array in zip(coeffs.keys(), arrays):
            if array:
                if isinstance(segments, np.ndarray):
                    seg_reduced[segments == key] = key
                elif isinstance(segments, list):
                    seg_reduced.append(segments[key])
                else:
                    raise TypeError
        return seg_reduced

    @staticmethod
    def read_image(img):
        return cv2.imread(img, 0)

    def segment_all(self, settings: dict, image_list=None):
        if not image_list:
            image_list = self.discover_images(self.fpath)

        path = os.path.join(os.path.join("Output", settings["name"]))

        for counter, image in enumerate(tqdm(image_list)):

            loaded_img = self.read_image(image)

            preprocessed_img, imges = self.preprocess_img(
                loaded_img, settings["preprocess"]
            )

            _, markerSeed, stats, centroids, unknown = self.generate_seeds(
                preprocessed_img, settings["seed_gen"]
            )

            segments = markerSeed

            if "segmentation" in settings.keys():
                segments, _ = self.generate_segments(
                    preprocessed_img,
                    markerSeed,
                    stats,
                    unknown,
                    centroids,
                    settings["segmentation"],
                )

            if "merge" in settings.keys():
                segments, _, _ = self.merge_segments(
                    segments, preprocessed_img, settings["merge"]
                )

            segments = self.size_exclusion(segments)

            if "fourier" in settings.keys():
                segments = self.fourier_postprocessing(
                    segments, settings["fourier"]
                )

            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(
                path, os.path.split(image)[-1].strip(".tif") + ".png"
            )
            plot_fiber_results(
                loaded_img, preprocessed_img, markerSeed, segments, save_path
            )

            # following can be used to visualize results
            # of preprocessing step
            # for step, img in imges.items():
            #    cv2.imwrite(
            #       save_path.strip('.png') + '_' + step + '.png',
            #       img.astype(np.uint8)
            #       )


if __name__ == "__main__":
    from multiprocessing import Pool
    import logging

    logging.basicConfig(filename="main.log", level=logging.INFO)

    pipe = segment_pipeline(srcPath="Input")
    with open("./settings_all.yml", "r") as stream:
        settings_all = yaml.load(stream)
    Pool().map(pipe.segment_all, settings_all)
