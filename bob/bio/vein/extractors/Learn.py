#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np
from bob.bio.base.extractor import Extractor
import bob.ip.base
# import bob.io.base
# import bob.learn.libsvm
import bob.io.base
from skimage.morphology import disk
from skimage.filters.rank import median
import os
import bob.bio.base
import pickle


class Learn(Extractor):
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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        machine_path = os.path.join(dir_path, "machines")
        machine_path = os.path.join(machine_path, self.name)
        f = bob.io.base.HDF5File(machine_path + ".hdf5")
        self.machine = bob.learn.linear.Machine(f)

        with open(machine_path+".p", "rb") as f:
            _ = pickle.load(f)
            self.threshold = pickle.load(f)

        self.block_x = int(self.name.split("_")[1])
        self.block_y = int(self.name.split("_")[1])

    def __filter_image__(self, image):

        output_shape = (image.shape[0], image.shape[1])
        block_size = (self.block_y, self.block_x)
        block_overlap = tuple(i - 1 for i in block_size)
        if len(image.shape) == 2:
            image = np.pad(image,
                           ((self.block_y/2, self.block_y/2),
                            (self.block_x/2, self.block_x/2)),
                           'edge')
            image_patches = \
                bob.ip.base.block(image,
                                  block_size=block_size,
                                  block_overlap=block_overlap,
                                  flat=False)
            # import ipdb; ipdb.sset_trace()
            image_patches = np.array(image_patches, dtype=np.float64)
            image_patches = np.reshape(image_patches, (image_patches.shape[0],
                                                       image_patches.shape[1],
                                                       image_patches.shape[2] *
                                                       image_patches.shape[3]))
            image_patches = np.reshape(image_patches, (image_patches.shape[0] *
                                                       image_patches.shape[1],
                                                       image_patches.shape[2]))
            output = self.machine(image_patches)
            output = np.where(output >= self.threshold,
                              1,
                              0)
            output = np.reshape(output, (output_shape))
            output = np.array(output, dtype=np.uint8)
            return output
        else:
            raise IOError("so far supporting only 2D input images")

    def __call__(self, input_data):
        """
        Apply thresholding vein extraction to an imput image.
        """
        image = input_data[0]
        mask = input_data[1]

        image = self.__filter_image__(image)

        if self.median:
            image = median(image, disk(self.size))

        image = np.where(mask < 1,
                         0,
                         image)
        return image
