#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.sp
import bob.ip.base
import numpy as np
from bob.bio.base.algorithm import Algorithm
from scipy import ndimage
import scipy.signal
import skimage
from skimage import transform
import scipy

class MMManual(Algorithm):
    """
    Vein matching: match ratio

    Based on N. Miura, A. Nagasaka, and T. Miyatake. Feature extraction of finger
    vein patterns based on repeated line tracking and its application to personal
    identification. Machine Vision and Applications, Vol. 15, Num. 4, pp.
    194--203, 2004

    The pre-alignment step is added to this class. The following alignment methods are implemented:

    1. image centering based on the center of mass. Both enroll and probe images are centered independently before Miura matching.

    **Parameters:**

    ch : uint
        Maximum search displacement in y-direction. Default value: 10.

    cw : uint
        Maximum search displacement in x-direction. Default value: 10.

    alignment_flag : bool
        If set to "True" pre-alignment of the images is done before the matching. Default value: True.

    alignment_method : str
        Name of the prealignment method. Possible values: "center_of_mass".
        "center_of_mass" - image centering based on the center of mass.

    dilation_flag : bool
        If set to "True" binary dilation of the images is done before the matching. Default value: False.

    ellipse_mask_size : uint
        Diameter of the elliptical kernel in pixels. Default value: 5.
    """


    def __init__( self,
                 ch = 10, cw = 10,
                 alignment_flag = True,
                 alignment_method = "manual_alignment",
                 dilation_flag = False,
                 ellipse_mask_size = 5):

        Algorithm.__init__( self,
                           ch = ch,
                           cw = cw,
                           alignment_flag = alignment_flag,
                           alignment_method = alignment_method,
                           dilation_flag = dilation_flag,
                           ellipse_mask_size = ellipse_mask_size )

        self.ch = ch
        self.cw = cw
        self.alignment_flag = alignment_flag
        self.alignment_method = alignment_method
        self.available_alignment_methods = ["manual_alignment_c", "manual_alignment"]
        self.dilation_flag = dilation_flag
        self.ellipse_mask_size = ellipse_mask_size


    def enroll(self, enroll_features):
        """
        enroll(enroll_features) -> model


        **Parameters:**

        enroll_features : py:class:`list`
            A list of features used for the enrollment of one model.
            list consists of tuples - first tuple object - image; second -
            alignment annotations.

        **Returns:**

        model : object
            The model enrolled from the ``enroll_features``.
            In this case, ``model`` is a tuple object, consisting of images and
            alignment annotations. each of then can also be a list.


            Must be writable with the :py:meth:`write_model` function and
            readable with the :py:meth:`read_model` function.
        """

        if len(enroll_features) == 1:
            enroll_features = np.squeeze(enroll_features)
            return enroll_features
        else:
            images = []
            alignment_annotations = []
            for t in enroll_features:
                images.append(t[0])
                alignment_annotations.append(t[1])

            return (images, alignment_annotations)


    def __convfft__( self, t, a ):
        # Determine padding size in x and y dimension
        size_t  = np.array(t.shape)
        size_a  = np.array(a.shape)
        outsize = size_t + size_a - 1

        # Determine 2D cross correlation in Fourier domain
        taux = np.zeros(outsize)
        taux[0:size_t[0],0:size_t[1]] = t
        Ft = bob.sp.fft(taux.astype(np.complex128))
        aaux = np.zeros(outsize)
        aaux[0:size_a[0],0:size_a[1]] = a
        Fa = bob.sp.fft(aaux.astype(np.complex128))

        convta = np.real(bob.sp.ifft(Ft*Fa))

        [w, h] = size_t-size_a+1
        output = convta[size_a[0]-1:size_a[0]-1+w, size_a[1]-1:size_a[1]-1+h]

        return output

    def center_image( self, input_array ):
        """
        This function shifts the image so as center of mass of the image and image center are alligned.

        **Parameters:**

        input_array : 2D :py:class:`numpy.ndarray`
            Input image to be shifted.

        **Returns:**

        shifted_array : 2D :py:class:`numpy.ndarray`
            Shifted image.
        """

        # center of mass of the input image:s
        coords = np.round( ndimage.measurements.center_of_mass( input_array ) )

        # center of the image
        center_location = np.round( np.array( input_array.shape )/2 )

        # resulting displacement:
        displacement = center_location - coords

        shifted_array = ndimage.interpolation.shift( input_array, displacement, cval = 0 )

        shifted_array[shifted_array<0.5] = 0
        shifted_array[shifted_array>=0.5] = 1

        return shifted_array


    def binary_dilation_with_ellipse( self, image ):
        """
        binary_dilation_with_ellipse( image, ellipse_mask_size ) -> image_dilated

        Dilates an input binary image with the ellipse kernel of the size ``ellipse_mask_size``.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray`
            Input binary image.

        **Returns:**

        image_dilated : 2D :py:class:`numpy.ndarray`
            Dilated image.
        """
        if self.ellipse_mask_size == 0:
            return np.array(image, dtype = np.float64)
        else:
            if self.ellipse_mask_size % 2 == 0:
                self.ellipse_mask_size = np.uint( self.ellipse_mask_size + 1 ) # make the mask odd

            # Make the elliptical kernel
            kernel = np.zeros( ( self.ellipse_mask_size, self.ellipse_mask_size ) )
            radius = ( self.ellipse_mask_size - 1 )/2
            y, x = np.ogrid[ -radius : radius + 1, -radius : radius + 1 ]
            mask = x**2 + y**2 <= radius**2
            kernel[ mask ] = 1

            # dilate the image
            image_dilated = ndimage.binary_dilation( image, structure = kernel )

            return image_dilated.astype(np.float64)

    def __min_point_count__(self, enroll_alignment, probe_alignment):
        """
        returns `enroll_alignment` and `probe_alignment` with equal length
        """
        min_point_count = np.min([len(enroll_alignment),
                                  len(probe_alignment)])
        enroll_alignment = enroll_alignment[:min_point_count]
        probe_alignment = probe_alignment[:min_point_count]
        return enroll_alignment, probe_alignment




    def compute_transformation(self, keypoints1, keypoints2):
        """
        Compute the affine transformation occurring between the keypoints 1 and keypoints 2. First, determine the
        inliers set by a RANSAC strategy. Then, estimate the affine transformation with a regression approach.

        **Parameters:**

        ``keypoints1`` : 2D :py:class:`numpy.ndarray`
            Keypoints set in correspondence in template 1

        ``keypoints2`` : 2D :py:class:`numpy.ndarray`
            Keypoints set in correspondence in template 2

        **Returns:**

        ``affine_transformation`` : :py:class:`skimage.transform.AffineTransform`
            Estimated affine transformation that fits the best the both sets of keypoints

        """

        # Construction of the matrix X and the vector y
        n = keypoints1.shape[0]
        X = np.hstack((keypoints1, np.ones((n, 1))))  # shape: (n x 3) with [x1, x2, 1]
        y = keypoints2  # shape: (n x 2) with [y1, y2]

        # Selection of the inliers set with a RANSAC strategy
        # X, y = select_RANSAC_model(X, y)

        if X.shape[0] < 3:
            return None

        # Resolution of the system XW = y where the unknown variable is the transformation W with shape: (3 x 2)
        W = self.solve_least_squares(X, y)
        # W = self.solve_least_euclidean_distance(X, y)
        # W = self.solve_linprog_LAD(X, y)
        # W = self.solve_iterative_LAD(X, y)

        if W is None:
            return None

        # Construction of the square affine transformation matrix A
        # where A = [[w1, w2, w3], [w4, w5, w6], [0.0, 0.0, 1.0]]
        A = np.vstack((W.T, np.array([0.0, 0.0, 1.0])))
        affine_transformation = transform.AffineTransform(matrix=A)

        return affine_transformation

    def solve_least_squares(self, X, y):
        """
        Solve a (Total) Least Squares regression on the equation XW = y using the closed form:
        W = (X^T * X)^(-1) * X^T * y

        **Parameters:**

        ``X`` : 2D :py:class:`numpy.ndarray` (shape: n x 3)
            First putative set of correspondence keypoints with the additional constant column

        ``y`` : 2D :py:class:`numpy.ndarray` (shape: n x 2)
            Second putative set of correspondence keypoints

        **Returns:**

        ``W`` : 2D :py:class:`numpy.ndarray` (shape: 3 x 2)
            Affine transformation matrix that solves the (Total) Least Squares problem

        """

        W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))

        return W

    def manual_align_image_c(self, image, enroll_alignment, probe_alignment):
        enroll_alignment, probe_alignment = \
            self.__min_point_count__(enroll_alignment, probe_alignment)

        probe_alignment = np.array(probe_alignment)
        enroll_alignment = np.array(enroll_alignment)

        keypoints_1 = []
        keypoints_2 = []
        for point in probe_alignment:
            keypoints_1.append([point[1], point[0]])

        for point in enroll_alignment:
            keypoints_2.append([point[1], point[0]])

        keypoints_1 = np.array(keypoints_1)
        keypoints_2 = np.array(keypoints_2)

        transformation = self.compute_transformation(keypoints_1,
                                                     keypoints_2)

        image = transform.warp(np.float64(image),
                               transformation,
                               order=3,
                               preserve_range=True)

        return image

    def manual_align_image(self, image, enroll_alignment, probe_alignment):
        enroll_alignment, probe_alignment = \
            self.__min_point_count__(enroll_alignment, probe_alignment)

        probe_alignment = np.array(probe_alignment)
        enroll_alignment = np.array(enroll_alignment)

        keypoints_1 = []
        keypoints_2 = []
        for point in probe_alignment:
            keypoints_1.append([point[1], point[0]])

        for point in enroll_alignment:
            keypoints_2.append([point[1], point[0]])

        keypoints_1 = np.array(keypoints_1)
        keypoints_2 = np.array(keypoints_2)

        transformation = skimage.transform.SimilarityTransform()
        transformation.estimate(keypoints_1,
                                keypoints_2)

        image = transform.warp(np.float64(image),
                               transformation,
                               order=3,
                               preserve_range=True)

        return image

    def score(self, model, probe):
        """
        score(model, probe) -> score

        Computes the score of the probe and the model using Miura matching algorithm.
        Prealignment with selected method is performed before matching if "alignment_flag = True".
        Score has a value between 0 and 0.5, larger value is better match.

        **Parameters:**

        model : 2D/3D :py:class:`numpy.ndarray`
            The model enrolled by the :py:meth:`enroll` function.

        probe : 2D :py:class:`numpy.ndarray`
            The probe read by the :py:meth:`read_probe` function.

        **Returns:**

        score_mean : float
            The resulting similarity score.
        """

        scores = []




#        if isinstance(model, np.ndarray):

#            if len( model.shape ) == 2:
#
#                model = [ model ] # this is necessary for unit tests only
#
#            else:
#
#                num_models = model.shape[0] # number of enroll samples
#
#                model = np.split( model, num_models, 0 ) # split 3D array into a list of 2D arrays of dimensions: (1,H,W)
#
#        model = [ np.squeeze( item ) for item in model ] # remove single-dimensional entries from the shape of an array

        model_alignment = model[1]
        model = model[0]
        probe_alignment = probe[1]
        probe = probe[0]

        if len( model.shape ) == 2:
            model = [model]
            model_alignment = [model_alignment]

        #import ipdb; ipdb.sset_trace()

#        if self.alignment_flag: # if prealignment is allowed
#            if self.alignment_method == "center_of_mass": # centering based on the center of mass of the image
#                probe = self.center_image( probe )

        if self.dilation_flag:
            probe = self.binary_dilation_with_ellipse( probe )

        for nr, enroll in enumerate(model):

            if not( self.alignment_method in self.available_alignment_methods ):
                raise Exception("Specified alignment method is not in the list of available_alignment_methods")

            if len( enroll.shape ) != 2 or len( probe.shape ) != 2: # check if input image is not of grayscale format
                raise Exception("The image must be a 2D array / grayscale format")

            enroll_ = enroll
            if self.alignment_flag: # if prealignment is allowed
                if self.alignment_method == "manual_alignment_c":
                    enroll = \
                    self.manual_align_image_c(image=enroll,
                                            enroll_alignment=model_alignment[nr],
                                            probe_alignment=probe_alignment)
                elif self.alignment_method == "manual_alignment":
                    enroll = \
                    self.manual_align_image(image=enroll,
                                            enroll_alignment=model_alignment[nr],
                                            probe_alignment=probe_alignment)

            if np.sum(enroll) == 0:
                enroll = enroll_

            if self.dilation_flag:
                enroll = self.binary_dilation_with_ellipse(enroll)

#            import matplotlib.pyplot as plt
#            fig = plt.figure()
#            ax = plt.subplot(121)
#            ax.imshow(enroll, cmap='Greys_r', interpolation='none')
#            ax = plt.subplot(122)
#            ax.imshow(enroll+probe, cmap='Greys_r', interpolation='none')
#            plt.show(fig)

            I = probe.astype( np.float64 )

            R = enroll.astype( np.float64 )



            h, w = R.shape

            crop_R = R[ self.ch: h-self.ch, self.cw: w-self.cw ]

            rotate_R = np.zeros( ( crop_R.shape[0], crop_R.shape[1] ) )

            bob.ip.base.rotate( crop_R, rotate_R, 180 )

            #Nm = self.__convfft__( I, rotate_R )
            Nm = scipy.signal.convolve2d(I, rotate_R, 'valid')

            t0, s0 = np.unravel_index( Nm.argmax(), Nm.shape )

            Nmm = Nm[t0,s0]

            scores.append( Nmm / ( sum( sum( crop_R ) ) + sum( sum( I[ t0: t0 + h - 2 * self.ch, s0: s0 + w - 2 * self.cw ] ) ) ) )

        score_mean = np.mean( scores )

        return score_mean

    def read_probe(self, file_name):
        """
        Reads the preprocessed data from file.
        his method overwrites the read_data() method of the Preprocessor class.

        **Parameters:**

        file_name : :py:class:`str`
            name of the file.

        **Returns:**

        output : ``tuple``
            a tuple containing the image and it's alignment annotations
        """
        f = bob.io.base.HDF5File(file_name, 'r')
        image = f.read('image')
        alignment_annotations = f.read('alignment annotations')
        del f
        output = (image, alignment_annotations)
        return output

    def read_model(self, file_name):
        """
        Reads the preprocessed data from file.
        his method overwrites the read_data() method of the Preprocessor class.

        **Parameters:**

        file_name : :py:class:`str`
            name of the file.

        **Returns:**

        output : ``tuple``
            a tuple containing the image and it's alignment annotations
        """
        f = bob.io.base.HDF5File(file_name, 'r')
        image = f.read('image')
        alignment_annotations = f.read('alignment annotations')
        del f
        output = (image, alignment_annotations)
        return output

    def write_model(self, data, file_name):
        """
        Writes the given data (that has been generated using the __call__
        function of this class) to file.
        This method overwrites the write_data() method of the Preprocessor
        class.

        **Parameters:**

        data :
            data returned by the __call__ method of the class.

        file_name : :py:class:`str`
            name of the file.
        """

        f = bob.io.base.HDF5File(file_name, 'w')
        f.set('image', data[0])
        f.set('alignment annotations', data[1])
        del f
