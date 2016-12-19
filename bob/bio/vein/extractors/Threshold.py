#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np
from bob.bio.base.extractor import Extractor
import bob.ip.base
import bob.io.base
import cv2
# import bob.learn.libsvm
# import bob.io.base
from skimage.filters.rank import mean_percentile
from skimage.morphology import disk
from skimage import exposure
from skimage.filters.rank import median


class Threshold(Extractor):
    """
    Class to compute vein filter output on an given input image.

    **Parameters:**

    name : :py:class:`str`
        name of predefinied extractor;
    median : :py:class:`bool`
        Flag to indicate, if Median filter is applied to output. Default -
        False
    size : :py:class:`int`
        Size of median filter. Default - 5
    """

    def __init__(self,
                 name,
                 median=False,
                 size=5
                 ):
        Extractor.__init__(self,
                           name=name,
                           median=median,
                           size=size
                           )
        self.name = name
        self.median = median
        self.size = int(size)

    def __apply_baseline__(self, image):
        if self.name == "Otsu":
            _, image = cv2.threshold(image, 0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            image = ~np.array(image, dtype=np.bool)
            image = np.array(image, dtype=np.uint8)
            return image
        elif self.name == "Otsu_blur":
            image = cv2.GaussianBlur(image, (5, 5), 0)
            _, image = cv2.threshold(image, 0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            image = ~np.array(image, dtype=np.bool)
            image = np.array(image, dtype=np.uint8)
            return image
        elif self.name.startswith("Adaptive") and not \
            self.name.startswith("Adaptive_h") and not \
            self.name.startswith("Adaptive_c") and not \
            self.name.startswith("Adaptive_ski"):

            params = self.name.split("_")
            p1 = int(params[1])
            p2 = int(params[2])
            image = cv2.medianBlur(image, 5)
            image = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          p1,
                                          p2)
            image = ~np.array(image, dtype=np.bool)
            image = np.array(image, dtype=np.uint8)
            return image
        elif self.name.startswith("Adaptive_h"):
            params = self.name.split("_")
            p1 = int(params[2])
            p2 = float(params[3])
            bob.ip.base.histogram_equalization(image)

            image = cv2.medianBlur(image, 5)
            image = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          p1,
                                          p2)
            image = ~np.array(image, dtype=np.bool)
            image = np.array(image, dtype=np.uint8)
            return image
        elif self.name.startswith("Adaptive_c"):
            params = self.name.split("_")
            p1 = int(params[2])
            p2 = int(params[3])
            p_clahe = int(params[4])
            clahe = cv2.createCLAHE(clipLimit=p_clahe, tileGridSize=(8, 8))
            image = clahe.apply(image)
            image = cv2.medianBlur(image, 5)
            image = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          p1,
                                          p2)
            image = ~np.array(image, dtype=np.bool)
            image = np.array(image, dtype=np.uint8)
            return image
        elif self.name.startswith("Adaptive_ski"):
            params = self.name.split("_")
            p1 = int(params[2])
            p2 = int(params[3])
            p_disk = int(params[4])

            local_mean = mean_percentile(image,
                                         selem=disk(p_disk),
                                         p0=0.01,
                                         p1=0.99)
            image = np.float32(image)
            image -= local_mean
            image = exposure.rescale_intensity(image, out_range=(0, 1))
            import warnings
            with warnings.catch_warnings(record=True) as w:
                image = exposure.equalize_adapthist(image, clip_limit=0.05)

            image = image * 255
            image = np.array(image, dtype=np.uint8)
            image = cv2.medianBlur(image, 5)
            image = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY,
                                          p1,
                                          p2)
            image = ~np.array(image, dtype=np.bool)
            image = np.array(image, dtype=np.uint8)
            return image
        else:
            raise IOError("Unknown algorithm - {}".
                          format(self.name))

    def __call__(self, input_data):
        """
        Apply thresholding vein extraction to an imput image.
        """
        image = input_data[0]
        mask = input_data[1]

        # mask the image:
        image = np.where(mask < 1,
                         255,
                         image)

        min_ROI_value = np.min(image)

        image = np.where(mask < 1,
                         min_ROI_value,
                         image)
        #import ipdb; ipdb.sset_trace()
        image = self.__apply_baseline__(image)

        if self.median:
            image = median(image, disk(self.size))

        image = np.where(mask < 1,
                         0,
                         image)
        return image
