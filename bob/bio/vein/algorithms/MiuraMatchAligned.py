#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.sp
import bob.ip.base

import numpy as np

from bob.bio.base.algorithm import Algorithm

from scipy import ndimage

class MiuraMatchAligned( Algorithm ):
    """ Vein matching: match ratio
    Based on N. Miura, A. Nagasaka, and T. Miyatake. Feature extraction of finger
    vein patterns based on repeated line tracking and its application to personal
    identification. Machine Vision and Applications, Vol. 15, Num. 4, pp.
    194--203, 2004
    
    The pre-alignment step is added to this class. The following alignment methods are implemented:
    1) image centering based on the center of mass. Both enroll and probe images are centered independently before Miura matching.
    
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


    def __init__( self, ch = 10, cw = 10, alignment_flag = True, alignment_method = "center_of_mass",
                 dilation_flag = False, ellipse_mask_size = 5 ):

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
        self.available_alignment_methods = ["center_of_mass"]
        self.dilation_flag = dilation_flag
        self.ellipse_mask_size = ellipse_mask_size


    def enroll(self, enroll_features):
        """enroll(enroll_features) -> model
        
        This function will enroll and return the model from the given list of features.
        It must be overwritten by derived classes.
        
        **Parameters:**
        
        enroll_features : [object]
            A list of features used for the enrollment of one model.
        
        **Returns:**
        
        model : object
            The model enrolled from the ``enroll_features``.
            Must be writable with the :py:meth:`write_model` function and readable with the :py:meth:`read_model` function.
        """
        
#        return np.array(enroll_features) # Do nothing in our case
        return enroll_features # Do nothing in our case



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
        """binary_dilation_with_ellipse( image, ellipse_mask_size ) -> image_dilated
        
        Dilates an input binary image with the ellipse kernel of the size ``ellipse_mask_size``.
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input binary image.
            
        **Returns:**
        
        image_dilated : 2D :py:class:`numpy.ndarray`
            Dilated image.
        """
        
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


    def score(self, model, probe):
        """score(model, probe) -> score
        
        Computes the score of the probe and the model using Miura matching algorithm.
        Prealignment with selected method is performed before matching if "alignment_flag = True".
        Score has a value between 0 and 0.5, larger value is better match.
        
        **Parameters:**
        
        model : 2D :py:class:`numpy.ndarray`
            The model enrolled by the :py:meth:`enroll` function.
        probe : 2D :py:class:`numpy.ndarray`
            The probe read by the :py:meth:`read_probe` function.
        **Returns:**
        score : float
            The resulting similarity score.         
        """
        
        if len( model.shape ) == 3 and model.shape[0] == 1:
            model = np.squeeze(model) # remove single-dimensional entries from the shape of an array if needed
        
        if len( probe.shape ) == 3 and probe.shape[0] == 1:
            probe = np.squeeze(probe) # remove single-dimensional entries from the shape of an array if needed
        
        if not( self.alignment_method in self.available_alignment_methods ):
            raise Exception("Specified alignment method is not in the list of available_alignment_methods")
        
        if len( model.shape ) != 2 or len( probe.shape ) != 2: # check if input image is not of grayscale format
            raise Exception("The image must be a 2D array / grayscale format")
            
        if self.alignment_flag: # if prealignment is allowed
            
            if self.alignment_method == "center_of_mass": # centering based on the center of mass of the image
                
                probe = self.center_image( probe )
                
                model = self.center_image( model )
        
        if self.dilation_flag:
            
            model = self.binary_dilation_with_ellipse( model )
            
            probe = self.binary_dilation_with_ellipse( probe )
        
        I = probe.astype( np.float64 )
        
        R = model.astype( np.float64 )
        
        h, w = R.shape
        
        crop_R = R[ self.ch: h-self.ch, self.cw: w-self.cw ]
        
        rotate_R = np.zeros( ( crop_R.shape[0], crop_R.shape[1] ) )
        
        bob.ip.base.rotate( crop_R, rotate_R, 180 )
        
        Nm = self.__convfft__( I, rotate_R )
        
        t0, s0 = np.unravel_index( Nm.argmax(), Nm.shape )
        
        Nmm = Nm[t0,s0]
        
        score = Nmm / ( sum( sum( crop_R ) ) + sum( sum( I[ t0: t0 + h - 2 * self.ch, s0: s0 + w - 2 * self.cw ] ) ) )
        
        return score



