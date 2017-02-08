#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:14:53 2017

@author: onikisins
"""

#==============================================================================
# Import what is needed here:
import numpy as np

from bob.bio.base.preprocessor import Preprocessor

from scipy import ndimage

from skimage.morphology import convex_hull_image

import bob.learn.em # for the k-means

from skimage import exposure

from skimage import transform as tf

from skimage.morphology import disk, binary_opening

#==============================================================================
# Class implementation:

class KMeansRoiFast(Preprocessor):
    """
    K-means based ROI extraction algorithm.
    First 2 centroids are obtained for the gray-scale values of the pixels in the input image.
    Next image threshold is calculated as an average of the centroids.
    The largest binary region is then selected and smoothed using morphological operations.
    """

    #==========================================================================
    def __init__(self, filter_name = "median_filter", mask_size = 7,
                 correct_mask_flag = False, correction_erosion_factor = 10,
                 erode_mask_flag = False, erosion_factor = 20,
                 convexity_flag = False,
                 rotation_centering_flag = False,
                 normalize_scale_flag = False,
                 mask_to_image_area_ratio = 0.2,
                 equalize_adapthist_flag = False,
                 speedup_flag = False):
        """
        **Parameters:**

        ``filter_name`` : :py:class:`str`
            Name of the filter to be applied before further processing.
            Possible optionsa are defined in ``available_filter_names`` parameter of the class.
            Default: "median_filter".

        ``mask_size`` : :py:class:`int`
            Size of the filter mask. Default: 7.

        ``correct_mask_flag`` : :py:class:`bool`
            Correct the possible ROI outliers if set to True. Default: False.

        ``correction_erosion_factor`` : :py:class:`int`
            In the roi correction step the binary mask of the ROI will be
            eroded/dilated with ellipse kernel of the size:
            (image_height / correction_erosion_factor). Default value: 10.

        ``erode_mask_flag`` : :py:class:`bool`
            Erode the binary mask of ROI if flag set to True. Default value: False.

        ``erosion_factor`` : :py:class:`int`
            In the roi erosion step the binary mask of the ROI will be
            eroded with square kernel of the size:
            (image_height / erosion_factor)*2+1. Default value: 20.

        ``convexity_flag`` : :py:class:`bool`
            Make mask convex if True. Default value: False.

        ``rotation_centering_flag`` : :py:class:`bool`
            Rotate and center the binary mask of the ROI and the input image if True. Default value: False.

        ``normalize_scale_flag`` : :py:class:`bool`
            Normalize the scale of the ROI and of the image if True. Default value: False.

        ``mask_to_image_area_ratio`` : :py:class:`float`
            Defines the area of the normalized ROI, as a fraction of the area
            of the input image. Default value: 0.2.

        ``equalize_adapthist_flag`` : :py:class:`bool`
            Enhance the contrast of the image if set to True. Default value: False.

        ``speedup_flag`` : :py:class:`bool`
            Use fast but approximate implementation if set to ``True``. This option
            will make the boundary of the ROI grainy. Default value: ``False``.
        """

        Preprocessor.__init__(self,
                                filter_name = filter_name,
                                mask_size = mask_size,
                                correct_mask_flag = correct_mask_flag,
                                correction_erosion_factor = correction_erosion_factor,
                                erode_mask_flag = erode_mask_flag,
                                erosion_factor = erosion_factor,
                                convexity_flag = convexity_flag,
                                rotation_centering_flag = rotation_centering_flag,
                                normalize_scale_flag = normalize_scale_flag,
                                mask_to_image_area_ratio = mask_to_image_area_ratio,
                                equalize_adapthist_flag = equalize_adapthist_flag,
                                speedup_flag = speedup_flag)

        self.filter_name = filter_name
        self.mask_size = mask_size
        self.correct_mask_flag = correct_mask_flag
        self.correction_erosion_factor = correction_erosion_factor
        self.erode_mask_flag = erode_mask_flag
        self.erosion_factor = erosion_factor
        self.convexity_flag = convexity_flag
        self.available_filter_names = ["gaussian_filter", "median_filter"]
        self.rotation_centering_flag = rotation_centering_flag
        self.normalize_scale_flag = normalize_scale_flag
        self.mask_to_image_area_ratio = mask_to_image_area_ratio
        self.equalize_adapthist_flag = equalize_adapthist_flag
        self.speedup_flag = speedup_flag


    #==========================================================================
    def filter_image(self, image, filter_name):
        """
        Filter input image. Available filter options are defiend on available_filter_names
        parameter of the class.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            input image.

        ``filter_name`` : :py:class:`str`
            filter image before processing.
            Possible optionsa are defined in ``available_filter_names`` parameter of the class.

        **Returns:**

        ``filtered_image`` : 2D :py:class:`numpy.ndarray`
            filtered image.
        """

        if not( isinstance( image, np.ndarray ) ): # check if input image is of correct data type
            raise Exception("Input image must be of the numpy.ndarray type")

        if image.shape[0] <= 0: # check if input image is not an empty array
            raise Exception("Empty array detected")

        if len(image.shape)!=2: # check if input image is not of grayscale format
            raise Exception("The image must be a 2D array / grayscale format")

        if not(filter_name in self.available_filter_names):
            raise Exception("The specified filter is not in the list of available_filter_names")

        if filter_name == "gaussian_filter":

            filtered_image = ndimage.gaussian_filter( image, sigma = 1 )

        if filter_name == "median_filter":

            filtered_image = ndimage.median_filter( image, self.mask_size )

        else:
            filtered_image = image

        return filtered_image


    #==========================================================================
    def connectedComponentsWithStats(self, img_binary):
        """
        This function is similar to the OpenCV function ``cv2.connectedComponentsWithStats()``.
        The implementation is based on scipy package and is not dependent on OpenCV.
        It finds the blobs and their statistics.

        **Parameters:**

        ``img_binary`` : 2D :py:class:`numpy.ndarray`
            binary image.

        **Returns:**

        ``nr_objects`` :
            see OpenCV function ``cv2.connectedComponentsWithStats()`` for more details.

        ``labeled`` :
            see OpenCV function ``cv2.connectedComponentsWithStats()`` for more details.

        ``stats_sc`` :
            see OpenCV function ``cv2.connectedComponentsWithStats()`` for more details.

        ``centroids_sc`` :
            see OpenCV function ``cv2.connectedComponentsWithStats()`` for more details.
        """

        # find connected components
        labeled, nr_objects = ndimage.label( img_binary )

        # find centroids and make the output similar to cv2.connectedComponentsWithStats() function:
        centroids_sc = ndimage.measurements.center_of_mass( img_binary, labeled, range(nr_objects + 1)[1:] )
        centroids_sc = np.array( centroids_sc )
        centroids_sc = np.fliplr( centroids_sc )

        stats_sc = []

        for label in range( nr_objects + 1 )[ 1: ]:

            x, y = np.where( labeled == label )

            stats_sc.append( [ min( y ), min( x ), max( y ) - min( y ) + 1, max( x ) - min( x ) + 1, len( x ) ] )

        stats_sc = np.array( stats_sc )

        return nr_objects, labeled, stats_sc, centroids_sc


    #==========================================================================
    def get_blob(self, image, speedup_flag):
        """
        K-means based algorithm for the extraction of the binary mask of the ROI.
        First, 2 centroids are obtained for the gray-scale values of the pixels in the input image.
        Next, image threshold is calculated as an average of the centroids.
        The values above/below threshold are set to 1/0 obtaining the binary image.
        The largest blob is then selected in that binary image.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            input image.

        ``speedup_flag`` : :py:class:`bool`
            Use fast but approximate implementation if set to ``True``. In this case
            not all data (1 out of 50 data points) will be used in the k-means algorithm.

        **Returns:**

        ``mask_binary`` : 2D :py:class:`numpy.ndarray`
            binary mask of the ROI.
        """

        k = 2 # number of the clusters is set to be 2

        data = image.flatten()

        data = np.delete( data, np.where( data == 255 ) ) # remove oversaturated pixels

        data = data.reshape( data.shape[ 0 ], 1 )

        step = len(data)/10000

        if speedup_flag:

            data = data[::step] # select ~10000 data points

        data = data.astype( np.float64 ) # must be float64 for the bob k-means

        kmeans = bob.learn.em.KMeansMachine( k , data.shape[1] ) # k clusters with a feature dimensionality of N

        kmeansTrainer = bob.learn.em.KMeansTrainer( initialization_method = 'KMEANS_PLUS_PLUS' )

        bob.learn.em.train( kmeansTrainer, kmeans, data, max_iterations = 100, convergence_threshold = 1e-5 ) # Train the KMeansMachine

        decision_boundary = np.mean( kmeans.means )

        labelled_image = np.zeros( image.shape )

        labelled_image[ image > decision_boundary ] = 1

        retval, labels, stats, centroids = self.connectedComponentsWithStats( labelled_image )

        selected_blob_idx = np.argmax( stats[:,4] ) + 1

        mask_binary = np.zeros( image.shape, dtype = np.uint8 ) # this

        mask_binary[ labels == selected_blob_idx ] = 1

        return mask_binary.astype( np.uint8 )


    #==========================================================================
    def generate_binary_ellipse_kernel(self, kernel_diameter):
        """
        Generate the ellipse kernel for morphological operations.
        The value of ``kernel_diameter`` will be converted to odd number if it is an even.

        **Parameters:**

        ``kernel_diameter`` : :py:class:`int`
            diameter of the ellipse kernel.

        **Returns:**

        ``ellipse_kernel`` : 2D :py:class:`numpy.ndarray`
            binary image of the kernel.
        """

        if kernel_diameter % 2 == 0:
            kernel_diameter = np.uint(kernel_diameter + 1) # make the mask odd

        ellipse_kernel = np.zeros( ( kernel_diameter, kernel_diameter ) )

        radius = ( kernel_diameter - 1 )/2
        y, x = np.ogrid[ -radius : radius + 1, -radius : radius + 1 ]
        mask = x**2 + y**2 <= radius**2

        ellipse_kernel[ mask ] = 1

        return ellipse_kernel


    #==========================================================================
    def correct_mask(self, binary_image, kernel_diameter, speedup_flag):
        """
        This function is composed of the following steps:

            1. Apply morphological erosion to the input ``binary_image`` using ellipse kernel
               of the diameter ``kernel_diameter``.
            2. Select the largest blob in the resulting image.
            3. Apply morphological dilation to the selected blob and return the result.

        **Parameters:**

        ``binary_image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.

        ``kernel_diameter`` : :py:class:`int`
            Diameter of the ellipse kernel.

        ``speedup_flag`` : :py:class:`bool`
            Use fast but approximate implementation if set to ``True``. This option
            will make the boundary of the ROI grainy.

        **Returns:**

        ``mask_binary`` : 2D :py:class:`numpy.ndarray`
            binary mask of the ROI.
        """

        if speedup_flag:

            N = 4

            disk_kernel = disk(kernel_diameter / (2*N))

            binary_downsample = tf.rescale(binary_image, scale = 1./N, order=1,
                                           mode='constant', cval=0, clip=True, preserve_range=True)

            binary_open = binary_opening( binary_downsample, selem = disk_kernel ).astype(np.uint8)

            mask_binary = tf.rescale(binary_open, scale = N, order=1, mode='constant', cval=0, clip=True, preserve_range=True)

            mask_binary = 1 * (mask_binary > 0.5)

        else:

            ellipse_kernel = self.generate_binary_ellipse_kernel(kernel_diameter)

            eroded_image = ndimage.morphology.binary_erosion(binary_image, structure = ellipse_kernel).astype(np.uint8)

            retval, labels, stats, centroids = self.connectedComponentsWithStats( eroded_image )

            selected_blob_idx = np.argmax( stats[:,4] ) + 1

            selected_blob_image = np.zeros( binary_image.shape, dtype = np.uint8 ) # this

            selected_blob_image[ labels == selected_blob_idx ] = 1

            mask_binary = ndimage.morphology.binary_dilation(selected_blob_image, structure = ellipse_kernel).astype(np.uint8)


        return mask_binary.astype( np.uint8 )


    #==========================================================================
    def make_mask_convex(self, binary_image):
        """
        Make the binary mask of the ROI of the convex hull shape.
        Binary closing is applied to the ``binary_image`` before
        the mask is converted to the convex hull shape.

        **Parameters:**

        ``binary_image`` : 2D :py:class:`numpy.ndarray`
            input binary image.

        **Returns:**

        ``mask_binary`` : 2D :py:class:`numpy.ndarray`
            convex binary mask of the ROI.
        """

        kernel_diameter = binary_image.shape[0] / self.erosion_factor

        ellipse_kernel = self.generate_binary_ellipse_kernel(kernel_diameter)

        mask_binary = ndimage.binary_closing( binary_image, structure = ellipse_kernel )

        if np.sum(mask_binary):

            mask_binary = convex_hull_image(mask_binary)

        return mask_binary.astype( np.uint8 )


    #==========================================================================
    def erode_mask(self, binary_image, kernel_radius):
        """
        Erode a binary mask of the ROI.

        **Parameters:**

        ``binary_image`` : 2D :py:class:`numpy.ndarray`
            input binary image.

        ``kernel_radius`` : :py:class:`int`
            the resulting size of the sqaure kernel used for erosion is: (kernel_radius*2+1)

        **Returns:**

        ``mask_binary`` : 2D :py:class:`numpy.ndarray`
            eroded binary mask of the ROI.
        """

        kernel_diameter = np.int(kernel_radius*2+1)

        padded_mask = np.lib.pad(binary_image, (kernel_diameter,), 'constant', constant_values=(0))
        # kernel and erode operation:
        kernel = np.ones((kernel_diameter,kernel_diameter),np.uint8)

        mask_binary = ndimage.binary_erosion( padded_mask,
                                             structure = kernel )[ kernel_diameter:-kernel_diameter,
                                             kernel_diameter:-kernel_diameter ]

        return mask_binary.astype( np.uint8 )


    #==========================================================================
    def rotate_and_center_roi_and_image(self, mask_binary, image):
        """
        Rotate and center the binary mask of the ROI and the input image.
        The angle between eigenvector of the blob and the vertical axis defines the rotation.
        The center of mass of the binary mask of the ROI is aligned with the
        center of the image. The same centering is applied to the original image.

        **Parameters:**

        ``mask_binary`` : 2D :py:class:`numpy.ndarray`
            binary mask of the ROI

        ``image`` : 2D :py:class:`numpy.ndarray`
            input image.

        **Returns:**

        ``mask_binary_transformed`` : 2D :py:class:`numpy.ndarray`
            binary mask of the ROI after centering and rotation.

        ``image_transformed`` : 2D :py:class:`numpy.ndarray`
            input image after centering and rotation.
        """

        y, x = np.nonzero(mask_binary) # coordinates of ROI pixels.
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x = x - x_mean # mean normalized coordinates.
        y = y - y_mean
        coords = np.vstack([x, y]) # 2D array of coordinates
        # Covariance matrix and its eigenvectors and eigenvalues:
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1] # Sort eigenvalues in decreasing order
        evec1, evec2 = evecs[:, sort_indices]
        x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evec2

        theta = np.tanh((x_v1)/(y_v1)) # Orientation of the largest eigenvector
        rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]]) # Transformation matrix

        # Offset to allign the center of mass and image center:
        offset = np.array(ndimage.center_of_mass(mask_binary)).dot(rotation_mat) - np.array(mask_binary.shape)/2

        mask_binary_transformed = ndimage.affine_transform(mask_binary,
                                                            rotation_mat, offset=(offset[0,0], offset[0,1]),
                                                            output_shape=None, output=None,
                                                            order=0, mode='constant', cval=0.0, prefilter=True) # was order=3

        image_transformed = ndimage.affine_transform(image,
                                                    rotation_mat, offset=(offset[0,0], offset[0,1]),
                                                    output_shape=None, output=None,
                                                    order=2, mode='constant', cval=0.0, prefilter=True) # was order=3

        return (mask_binary_transformed, image_transformed)


    #==========================================================================
    def normalize_scale(self, image, mask, mask_to_image_area_ratio):
        """
        This function normalizes the scale of the ROI and of the input image.
        The parameter ``mask_to_image_area_ratio`` defines the area of the
        normalized ROI, as a fraction of the area of the input image.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        **Returns:**

        ``image_scaled`` : 2D :py:class:`numpy.ndarray`
            Image after scaling.

        ``mask_scaled`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI with normalized scale.
        """

        scale = ( np.sqrt( np.sum(mask) ) ) / (np.sqrt( mask_to_image_area_ratio * image.shape[0] * image.shape[1]))

        tform = tf.SimilarityTransform(scale = scale, rotation = 0, translation = (0, 0))

        image_scaled = tf.warp(image, tform, preserve_range = True)

        mask_scaled = tf.warp(mask, tform, preserve_range = True)

        image_scaled = image_scaled.astype(np.uint8)

#        mask_scaled[mask_scaled>0.1] = 1

        mask_scaled = 1 * (mask_scaled > 0.1)

        mask_scaled = mask_scaled.astype(np.uint8)

        return image_scaled, mask_scaled


    #==========================================================================
    def equalize_adapthist(self, image):
        """
        Enhance local contrast of the input image using
        Contrast Limited Adaptive Histogram Equalization (CLAHE) method.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        **Returns:**

        ``image_norm`` : 2D :py:class:`numpy.ndarray`
            Image after contrast enhancement.
        """

        image_norm = exposure.equalize_adapthist(image)

        image_norm = exposure.rescale_intensity(image_norm, out_range = np.uint8)

        image_norm = image_norm.astype(np.uint8)

        return image_norm


    #==========================================================================
    def get_roi(self, image):
        """
        Get the binary image of the ROI and modify it according to specified options.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            input image.

        **Returns:**

        ``mask_binary`` : 2D :py:class:`numpy.ndarray`
            binary mask of the ROI.
        """

        filtered_image = self.filter_image(image, self.filter_name) # Filter the image first

        mask_binary = self.get_blob(filtered_image, speedup_flag = self.speedup_flag) # obtain the binary mask of the ROI

        if self.correct_mask_flag:

            # The kernel diameter for the ROI correction stage:
            kernel_diameter = np.uint( mask_binary.shape[0] / self.correction_erosion_factor )

            mask_binary = self.correct_mask(mask_binary, kernel_diameter, speedup_flag = self.speedup_flag)

        if self.erode_mask_flag:

            # The kernel diameter for the ROI erosion stage:
            kernel_radius = np.uint( mask_binary.shape[0] / self.erosion_factor )

            mask_binary = self.erode_mask(mask_binary, kernel_radius)

        if self.convexity_flag:

            mask_binary = self.make_mask_convex(mask_binary)

        return mask_binary.astype( np.uint8 )


    #==========================================================================
    def __call__(self, image, annotations):
        """
        Get the binary image of the ROI. The ROI should fit into the limits specified in the blob_xywh_offsets list.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            input image.

        **Returns:**

        ``image_transformed`` : 2D :py:class:`numpy.ndarray`
            transformed image.

        ``mask_binary_transformed`` : 2D :py:class:`numpy.ndarray`
            binary mask of the ROI.
        """

        mask_binary = self.get_roi(image)

        if self.normalize_scale_flag:

            image_transformed, mask_binary_transformed = self.normalize_scale(image,
                                                                              mask_binary,
                                                                              self.mask_to_image_area_ratio)
        else:

            image_transformed = image
            mask_binary_transformed = mask_binary

        if self.rotation_centering_flag:

            mask_binary_transformed, image_transformed = self.rotate_and_center_roi_and_image(mask_binary_transformed,
                                                                                              image_transformed)

        if self.equalize_adapthist_flag:

            image_transformed = self.equalize_adapthist(image_transformed)


        return (image_transformed, mask_binary_transformed)

    #==========================================================================
    def write_data( self, data, file_name ):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_data() method of the Preprocessor class.

        **Parameters:**

        ``data`` :
            data returned by the __call__ method of the class.

        ``file_name`` : :py:class:`str`
            name of the file.
        """

        f = bob.io.base.HDF5File( file_name, 'w' )
        f.set( 'image', data[ 0 ] )
        f.set( 'mask', data[ 1 ] )
        del f

    #==========================================================================
    def read_data( self, file_name ):
        """
        Reads the preprocessed data from file.
        his method overwrites the read_data() method of the Preprocessor class.

        **Parameters:**

        ``file_name`` : :py:class:`str`
            name of the file.

        **Returns:**

        ``(image, mask)`` : :py:class:`list`
            a tuple containing the image and the binary mask of the ROI.
        """
        f = bob.io.base.HDF5File( file_name, 'r' )
        image = f.read( 'image' )
        mask = f.read( 'mask' )
        del f
        return ( image, mask )





