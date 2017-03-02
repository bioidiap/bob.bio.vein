#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

#==============================================================================
# Import what is needed here:

import math
import numpy as np

import bob.core
import bob.io.base

from bob.bio.base.extractor import Extractor

#from .. import utils

from skimage import transform as tf
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import center_of_mass, correlate1d

from skimage.morphology import binary_dilation

from skimage.morphology import binary_closing

from skimage.morphology import disk

#==============================================================================
# Class implementation:

class MaximumCurvatureScaleRotation (Extractor):
    """
    MiuraMax feature extractor.

    Based on N. Miura, A. Nagasaka, and T. Miyatake, Extraction of Finger-Vein
    Pattern Using Maximum Curvature Points in Image Profiles. Proceedings on IAPR
    conference on machine vision applications, 9 (2005), pp. 347--350

    The extractor is updated with two options:

    1.  Normalization of the mean distance between the point pairs in the
        input binary image is done if ``norm_p2p_dist_flag`` is set to ``True``.

    2.  The output image, which is the sum of input binary images rotated
        in the specified range is generated if ``sum_of_rotated_images_flag``
        is set to ``True``.

    **Parameters:**

    ``sigma`` : :py:class:`int`
        Sigma used for determining derivatives.
        Default: 5.

    ``norm_p2p_dist_flag`` : :py:class:`bool`
        If ``True`` normalize the mean distance between the point pairs in the
        input binary image to the ``selected_mean_dist`` value.
        Default: ``False``.

    ``selected_mean_dist`` : :py:class:`float`
        Normalize the mean distance between the point pairs in the
        input binary image to this value.
        Default: 100.

    ``sum_of_rotated_images_flag`` : :py:class:`bool`
        If ``True`` generate the output image, which is the sum of input images rotated
        in the specified range with the defined step.
        Default: ``False``.

    ``angle_limit`` : :py:class:`float`
        Rotate the image in the range [-angle_limit, +angle_limit] degrees.
        Default: 10.

    ``angle_step`` : :py:class:`float`
        Rotate the image with this step in degrees.
        Default: 1.

    ``speed_up_flag`` : :py:class:`bool`
        If ``False`` the output vein pattern is identical to the one introduced
        in the original paper. If ``True``, the output is slightly different,
        bu the result is obtained faster.
        Default: ``False``.
    """

    #==========================================================================
    def __init__(self, sigma = 5,
                 norm_p2p_dist_flag = False, selected_mean_dist = 100,
                 sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1,
                 speed_up_flag = False):

        Extractor.__init__(self,
                           sigma = sigma,
                           norm_p2p_dist_flag = norm_p2p_dist_flag,
                           selected_mean_dist = selected_mean_dist,
                           sum_of_rotated_images_flag = sum_of_rotated_images_flag,
                           angle_limit = angle_limit,
                           angle_step = angle_step,
                           speed_up_flag = speed_up_flag)

        self.sigma = sigma
        self.norm_p2p_dist_flag = norm_p2p_dist_flag
        self.selected_mean_dist = selected_mean_dist
        self.sum_of_rotated_images_flag = sum_of_rotated_images_flag
        self.angle_limit = angle_limit
        self.angle_step = angle_step
        self.speed_up_flag = speed_up_flag


    #==========================================================================
    def filter_image_two_1d(self, image, mask1, mask2, mode='constant', cval=0):
        """
        Filter input image with two 1D filters in x and y directions.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask1`` : 1D :py:class:`numpy.ndarray`
            Filter mask for the axis 0.

        ``mask2`` : 1D :py:class:`numpy.ndarray`
            Filter mask for the axis 1.

        ``mode`` : :py:class:`str`
            {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
            How to handle values outside the image borders.

        ``cval`` : :py:class:`float`
            Optional.
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        **Returns:**

        ``image_filt`` : 2D :py:class:`numpy.ndarray`
            Filtered image.
        """

        if image.dtype != np.float:

            image = image.astype(np.float)

        image_1 = correlate1d(image, mask1, axis = 0, mode=mode, cval=cval)
        image_filt = correlate1d(image_1, mask2, axis = 1, mode=mode, cval=cval)

        return image_filt


    #==========================================================================
    def filter_image_with_separable_filters(self, image, sigma):
        """
        Filter image with hx, hxx, hy, hyy and hxy filters in the fast way
        using their separability property.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel.

        **Returns:**

        ``fx_fast`` : 2D :py:class:`numpy.ndarray`
            Image filtered with hx mask.

        ``fxx_fast`` : 2D :py:class:`numpy.ndarray`
            Image filtered with hxx mask.

        ``fy_fast`` : 2D :py:class:`numpy.ndarray`
            Image filtered with hy mask.

        ``fyy_fast`` : 2D :py:class:`numpy.ndarray`
            Image filtered with hyy mask.

        ``fxy_fast`` : 2D :py:class:`numpy.ndarray`
            Image filtered with hxy mask.
        """

        image = image.astype(np.float)

        winsize = np.ceil(4*sigma)

        x = np.arange(-winsize, winsize + 1, 1.0)
        h_1 = (1/(np.sqrt(2*math.pi)*sigma)) * np.exp(-(x**2)/(2*sigma**2))
        h_2 = h_1

        # First filter:
        hx_1 = h_1
        hx_2 = (x/(sigma**2))*h_2
        fx_fast = self.filter_image_two_1d(image, hx_1, hx_2, mode='constant', cval=0)

        # Second filter:
        hxx_1 = h_1
        hxx_2 = ((x**2 - sigma**2)/(sigma**4))*h_2
        fxx_fast = self.filter_image_two_1d(image, hxx_1, hxx_2, mode='constant', cval=0)


        # Third filter:
        fy_fast = self.filter_image_two_1d(image, hx_2, hx_1, mode='constant', cval=0)

        # Fourth filter:
        fyy_fast = self.filter_image_two_1d(image, hxx_2, hxx_1, mode='constant', cval=0)

        # Fith filter:
        hxy_1 = x/(sigma**2)*h_1
        hxy_2 = x/(sigma**2)*h_2
        fxy_fast = self.filter_image_two_1d(image, hxy_1, hxy_2, mode='constant', cval=0)

        return fx_fast/255., fxx_fast/255., fy_fast/255., fyy_fast/255., fxy_fast/255.

#        return fx_fast, fxx_fast, fy_fast, fyy_fast, fxy_fast


    #==========================================================================
    def maximum_curvature(self, image, mask):
        """Computes and returns the Maximum Curvature features for the given input
        fingervein image

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        **Returns:**

        ``img_veins_bin`` : 2D :py:class:`numpy.ndarray`
            Binary image of the veins.
        """

        if image.dtype != np.uint8:
           image = bob.core.convert(image,np.uint8,(0,255),(0,1))
        #No es necesario pasarlo a uint8, en matlab lo dejan en float64. Comprobar si varian los resultados en vera database y ajustar.

        finger_mask = np.zeros(mask.shape)
        finger_mask[mask == True] = 1

#        winsize = np.ceil(4*self.sigma)
#
#        x = np.arange(-winsize, winsize+1)
#        y = np.arange(-winsize, winsize+1)
#        X, Y = np.meshgrid(x, y)
#
#        h = (1/(2*math.pi*self.sigma**2))*np.exp(-(X**2 + Y**2)/(2*self.sigma**2))
#        hx = (-X/(self.sigma**2))*h
#        hxx = ((X**2 - self.sigma**2)/(self.sigma**4))*h
#        hy = hx.T
#        hyy = hxx.T
#        hxy = ((X*Y)/(self.sigma**4))*h
#
#        # Do the actual filtering
#
#        fx = utils.imfilter(image, hx)
#        fxx = utils.imfilter(image, hxx)
#        fy = utils.imfilter(image, hy)
#        fyy = utils.imfilter(image, hyy)
#        fxy = utils.imfilter(image, hxy)

        fx, fxx, fy, fyy, fxy = self.filter_image_with_separable_filters(image, self.sigma)

        f1  = 0.5*np.sqrt(2)*(fx + fy)   # \  #
        f2  = 0.5*np.sqrt(2)*(fx - fy)   # /  #
        f11 = 0.5*fxx + fxy + 0.5*fyy       # \\ #
        f22 = 0.5*fxx - fxy + 0.5*fyy       # // #

        img_h, img_w = image.shape  #Image height and width

        # Calculate curvatures
        k = np.zeros((img_h, img_w, 4))
        k[:,:,0] = (fxx/((1 + fx**2)**(3/2)))*finger_mask  # hor #
        k[:,:,1] = (fyy/((1 + fy**2)**(3/2)))*finger_mask  # ver #
        k[:,:,2] = (f11/((1 + f1**2)**(3/2)))*finger_mask  # \   #
        k[:,:,3] = (f22/((1 + f2**2)**(3/2)))*finger_mask  # /   #

        # Scores
        Vt = np.zeros(image.shape)
        Wr = 0

        # Horizontal direction
        bla = k[:,:,0] > 0
        for y in range(0,img_h):
            for x in range(0,img_w):
                if (bla[y,x]):
                    Wr = Wr + 1
                if ( Wr > 0 and (x == (img_w-1) or not bla[y,x]) ):
                    if (x == (img_w-1)):
                        # Reached edge of image
                        pos_end = x
                    else:
                        pos_end = x - 1

                    pos_start = pos_end - Wr + 1 # Start pos of concave
                    if (pos_start == pos_end):
                        I=np.argmax(k[y,pos_start,0])
                    else:
                        I=np.argmax(k[y,pos_start:pos_end+1,0])

                    pos_max = pos_start + I
                    Scr = k[y,pos_max,0]*Wr
                    Vt[y,pos_max] = Vt[y,pos_max] + Scr
                    Wr = 0


        # Vertical direction
        bla = k[:,:,1] > 0
        for x in range(0,img_w):
            for y in range(0,img_h):
                if (bla[y,x]):
                    Wr = Wr + 1
                if ( Wr > 0 and (y == (img_h-1) or not bla[y,x]) ):
                    if (y == (img_h-1)):
                        # Reached edge of image
                        pos_end = y
                    else:
                        pos_end = y - 1

                    pos_start = pos_end - Wr + 1 # Start pos of concave
                    if (pos_start == pos_end):
                        I=np.argmax(k[pos_start,x,1])
                    else:
                        I=np.argmax(k[pos_start:pos_end+1,x,1])

                    pos_max = pos_start + I
                    Scr = k[pos_max,x,1]*Wr

                    Vt[pos_max,x] = Vt[pos_max,x] + Scr
                    Wr = 0

        # Direction: \ #
        bla = k[:,:,2] > 0
        for start in range(0,img_w+img_h-1):
            # Initial values
            if (start <= img_w-1):
                x = start
                y = 0
            else:
                x = 0
                y = start - img_w + 1
            done = False

            while (not done):
                if(bla[y,x]):
                    Wr = Wr + 1

                if ( Wr > 0 and (y == img_h-1 or x == img_w-1 or not bla[y,x]) ):
                    if (y == img_h-1 or x == img_w-1):
                        # Reached edge of image
                        pos_x_end = x
                        pos_y_end = y
                    else:
                        pos_x_end = x - 1
                        pos_y_end = y - 1

                    pos_x_start = pos_x_end - Wr + 1
                    pos_y_start = pos_y_end - Wr + 1

                    if (pos_y_start == pos_y_end and pos_x_start == pos_x_end):
                        d = k[pos_y_start, pos_x_start, 2]
                    elif (pos_y_start == pos_y_end):
                        d = np.diag(k[pos_y_start, pos_x_start:pos_x_end+1, 2])
                    elif (pos_x_start == pos_x_end):
                        d = np.diag(k[pos_y_start:pos_y_end+1, pos_x_start, 2])
                    else:
                        d = np.diag(k[pos_y_start:pos_y_end+1, pos_x_start:pos_x_end+1, 2])

                    I = np.argmax(d)

                    pos_x_max = pos_x_start + I
                    pos_y_max = pos_y_start + I

                    Scr = k[pos_y_max,pos_x_max,2]*Wr

                    Vt[pos_y_max,pos_x_max] = Vt[pos_y_max,pos_x_max] + Scr
                    Wr = 0

                if((x == img_w-1) or (y == img_h-1)):
                    done = True
                else:
                    x = x + 1
                    y = y + 1

        # Direction: /
        bla = k[:,:,3] > 0
        for start in range(0,img_w+img_h-1):
            # Initial values
            if (start <= (img_w-1)):
                x = start
                y = img_h-1
            else:
                x = 0
                y = img_w+img_h-start-1
            done = False

            while (not done):
                if(bla[y,x]):
                    Wr = Wr + 1
                if ( Wr > 0 and (y == 0 or x == img_w-1 or not bla[y,x]) ):
                    if (y == 0 or x == img_w-1):
                        # Reached edge of image
                        pos_x_end = x
                        pos_y_end = y
                    else:
                        pos_x_end = x - 1
                        pos_y_end = y + 1

                    pos_x_start = pos_x_end - Wr + 1
                    pos_y_start = pos_y_end + Wr - 1

                    if (pos_y_start == pos_y_end and pos_x_start == pos_x_end):
                        d = k[pos_y_end, pos_x_start, 3]
                    elif (pos_y_start == pos_y_end):
                        d = np.diag(np.flipud(k[pos_y_end, pos_x_start:pos_x_end+1, 3]))
                    elif (pos_x_start == pos_x_end):
                        d = np.diag(np.flipud(k[pos_y_end:pos_y_start+1, pos_x_start, 3]))
                    else:
                        d = np.diag(np.flipud(k[pos_y_end:pos_y_start+1, pos_x_start:pos_x_end+1, 3]))

                    I = np.argmax(d)
                    pos_x_max = pos_x_start + I
                    pos_y_max = pos_y_start - I
                    Scr = k[pos_y_max,pos_x_max,3]*Wr
                    Vt[pos_y_max,pos_x_max] = Vt[pos_y_max,pos_x_max] + Scr
                    Wr = 0

                if((x == img_w-1) or (y == 0)):
                    done = True
                else:
                    x = x + 1
                    y = y - 1

        ## Connection of vein centres
#        Cd = np.zeros((img_h, img_w, 4))
#        for x in range(2,img_w-3):
#            for y in range(2,img_h-3):
#                Cd[y,x,0] = min(np.amax(Vt[y,x+1:x+3]), np.amax(Vt[y,x-2:x]))   # Hor  #
#                Cd[y,x,1] = min(np.amax(Vt[y+1:y+3,x]), np.amax(Vt[y-2:y,x]))   # Vert #
#                Cd[y,x,2] = min(np.amax(Vt[y-2:y,x-2:x]), np.amax(Vt[y+1:y+3,x+1:x+3])) # \  #
#                Cd[y,x,3] = min(np.amax(Vt[y+1:y+3,x-2:x]), np.amax(Vt[y-2:y,x+1:x+3])) # /  #
#
#        #Veins
#        img_veins = np.amax(Cd,axis=2)
#
#        # Binarise the vein image
#        md = np.median(img_veins[img_veins>0])
#        img_veins_bin = img_veins > md


        if not(self.speed_up_flag):

            ## The code below is functionally identical to the one commented above,
            ## but is optimised for speed:

            Vt_binary = np.zeros(Vt.shape, dtype = np.uint8)
            Vt_binary[Vt>0] = 1

            Vt_binary_dil = binary_dilation(Vt_binary, selem = np.ones((5,5)))

            coords = np.argwhere(Vt_binary_dil[:img_h-3, :img_w-3]>0)

            y = coords[:,0]
            x = coords[:,1]

            x1 = x+1 # precompute some common coordinates
            x3 = x+3
            x_2 = x-2

            y1 = y+1
            y3 = y+3
            y_2 = y-2

            vec1 = np.max([ Vt[ val, x1[idx]: x3[idx] ] for idx, val in enumerate(y) ], axis = 1)
            vec2 = np.max([ Vt[ val, x_2[idx]: x[idx] ] for idx, val in enumerate(y) ], axis = 1)
            cd_0 = np.min( np.vstack([vec1, vec2]), axis = 0 )

            vec1 = np.max([ Vt[ y1[idx]: y3[idx], val ] for idx, val in enumerate(x) ], axis = 1)
            vec2 = np.max([ Vt[ y_2[idx]: y[idx], val ] for idx, val in enumerate(x) ], axis = 1)
            cd_1 = np.min( np.vstack([vec1, vec2]), axis = 0 )

            vec1 = np.max( np.max([ Vt[ y_2[idx]:val, x_2[idx]: x[idx] ] for idx, val in enumerate(y) ], axis = 1), axis = 1 )
            vec2 = np.max( np.max([ Vt[ y1[idx]:y3[idx], x1[idx]: x3[idx] ] for idx, _ in enumerate(y) ], axis = 1), axis = 1 )
            cd_2 = np.min( np.vstack([vec1, vec2]), axis = 0 )

            vec1 = np.max( np.max([ Vt[ y1[idx]:y3[idx], x_2[idx]: val ] for idx, val in enumerate(x) ], axis = 1), axis = 1 )
            vec2 = np.max( np.max([ Vt[ y_2[idx]:val, x1[idx]: x3[idx] ] for idx, val in enumerate(y) ], axis = 1), axis = 1 )
            cd_3 = np.min( np.vstack([vec1, vec2]), axis = 0 )

            img_veins_val = np.amax( [ cd_0, cd_1, cd_2, cd_3 ], axis = 0 )

            img_veins = np.zeros((img_h, img_w))

            for idx, (y,x) in enumerate(coords):

                img_veins[y,x] = img_veins_val[idx]

            # Binarise the vein image:
            md = np.median(img_veins[img_veins>0])

            img_veins_bin = img_veins > md

        else:

            Vt_binary = np.zeros(Vt.shape)

            md_1 = np.median(Vt[Vt>0])

            Vt_binary[Vt>md_1] = 1

            selem = disk(2)

            img_veins_bin = binary_closing(Vt_binary, selem = selem)


        return img_veins_bin.astype(np.float64)


    #==========================================================================
    def find_scale(self, binary_image, selected_mean_dist):
        """
        Find the scale normalizing the mean distance between the point pairs in the
        input binary image to the ``selected_mean_dist`` value.

        **Parameters:**

        ``binary_image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.

        ``selected_mean_dist`` : :py:class:`float`
            Normalize the mean distance to this value.

        **Returns:**

        ``scale`` : :py:class:`float`
            The scale to be applied to the input binary image to
            normalize the mean distance between the point pairs.
        """

        X = np.argwhere(binary_image == 1)[::10,:]

        dist_mat = squareform(pdist(X, metric='euclidean'))

        dist_mat_mean = np.mean(dist_mat)

#        scale = dist_mat_mean / selected_mean_dist

        scale = selected_mean_dist / dist_mat_mean

        return scale


    #==========================================================================
    def scale_binary_image(self, image, scale):
        """
        Scale the input binary image. The center of mass of the scaled/output binary
        image is aligned with the center of the input image.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.

        ``scale`` : :py:class:`float`
            The scale to be applied to the input binary image.

        **Returns:**

        ``image_scaled_translated`` : 2D :py:class:`numpy.ndarray`
            The scaled image.
        """

        h, w = image.shape

        image_coords = np.argwhere(image) # centered coordinates of the vein (non-zero) pixels

        offset = np.mean(image_coords, axis=0)

        image_coords = image_coords - offset

        scale_matrix = np.array([[scale, 0],
                                 [0, scale]]) # scaling matrix

        center_offset = np.array(image.shape)/2.

        coords_scaled = np.round( np.dot( image_coords, scale_matrix ) ) + center_offset

        coords_scaled = coords_scaled.astype(np.int)

        coords_scaled[coords_scaled < 0] = 0
        coords_scaled[:, 0][coords_scaled[:, 0] >= h] = h-1
        coords_scaled[:, 1][coords_scaled[:, 1] >= w] = w-1

        image_scaled_centerd = np.zeros((h, w))

        image_scaled_centerd[coords_scaled[:,0], coords_scaled[:,1]] = 1

        image_scaled_centerd = binary_closing(image_scaled_centerd, selem = np.ones((2,2)))

        image_scaled_centerd[0, : ] = 0
        image_scaled_centerd[-1, :] = 0
        image_scaled_centerd[:, 0 ] = 0
        image_scaled_centerd[:, -1] = 0

        return image_scaled_centerd.astype(np.float64)


    #==========================================================================
    def sum_of_rotated_images(self, image, angle_limit, angle_step):
        """
        Generate the output image, which is the sum of input images rotated
        in the specified range with the defined step.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``angle_limit`` : :py:class:`float`
            Rotate the image in the range [-angle_limit, +angle_limit] degrees.

        ``angle_step`` : :py:class:`float`
            Rotate the image with this step in degrees.

        **Returns:**

        ``output_image`` : 2D :py:class:`numpy.ndarray`
            Sum of rotated images.

        ``output_image`` : 3D :py:class:`numpy.ndarray`
            A stack of rotated images. Array size:
            (N_images, Height, Width)
        """

        output_image = np.zeros(image.shape)

        angles = np.arange(-angle_limit, angle_limit + 1, angle_step)

        rotated_images = []

        for angle in angles:

            image_rotated = tf.rotate(image, angle, preserve_range=True)

            output_image = output_image + image_rotated

            rotated_images.append(image_rotated)

        return output_image, np.array(rotated_images)


    #==========================================================================
    def __call__(self, image):
        """Reads the input image, extract the features based on Maximum Curvature
        of the fingervein image, and writes the resulting template"""

        input_image = image[0] #Normalized image with or without histogram equalization
        mask = image[1]

        binary_image = self.maximum_curvature(input_image, mask)

        return_data = binary_image

        if self.norm_p2p_dist_flag:

            scale = self.find_scale(binary_image, self.selected_mean_dist)

            binary_image = self.scale_binary_image(binary_image, scale)

            return_data = binary_image

        if self.sum_of_rotated_images_flag:

            sum_of_rotated_img, rotated_images_array = self.sum_of_rotated_images(binary_image, self.angle_limit, self.angle_step)

            return_data = ( binary_image, sum_of_rotated_img, rotated_images_array )

        return return_data


    #==========================================================================
    def write_feature( self, data, file_name ):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_feature() method of the Extractor class.

        **Parameters:**

        ``data`` : obj
            Data returned by the __call__ method of the class.

        ``file_name`` : :py:class:`str`
            Name of the file.
        """

        f = bob.io.base.HDF5File( file_name, 'w' )

        if not(self.sum_of_rotated_images_flag):

            f.set( 'array', data )

        else:

            f.set( 'binary_image', data[0] )
            f.set( 'sum_of_rotated_img', data[1] )
            f.set( 'rotated_images_array', data[2] )

        del f


    #==========================================================================
    def read_feature( self, file_name ):
        """
        Reads the preprocessed data from file.
        This method overwrites the read_feature() method of the Extractor class.

        **Parameters:**

        ``file_name`` : :py:class:`str`
            Name of the file.

        **Returns:**

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Maximum eigenvalues of Hessian matrices.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        """

        f = bob.io.base.HDF5File( file_name, 'r' )

        if not(self.sum_of_rotated_images_flag):

            return_data = f.read( 'array' )

        else:

            binary_image = f.read( 'binary_image' )

            sum_of_rotated_img = f.read( 'sum_of_rotated_img' )

            rotated_images_array = f.read( 'rotated_images_array' )

            return_data = ( binary_image, sum_of_rotated_img, rotated_images_array )

        del f

        return return_data


