#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np

from bob.bio.base.algorithm import Algorithm

from scipy import ndimage

from skimage.feature import match_template

class MatchTemplate(Algorithm):
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


    def __init__(self, dilation_flag = False, ellipse_mask_size = 5):

        Algorithm.__init__(self, 
                           dilation_flag = dilation_flag,
                           ellipse_mask_size = ellipse_mask_size)
        
        self.dilation_flag = dilation_flag
        self.ellipse_mask_size = ellipse_mask_size


    def enroll(self, enroll_features):
        """
        enroll(enroll_features) -> model
        
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
        
        if isinstance(model, np.ndarray):
            
            if len( model.shape ) == 2:
                
                model = [ model ] # this is necessary for unit tests only
                
            else:
                
                num_models = model.shape[0] # number of enroll samples
                
                model = np.split( model, num_models, 0 ) # split 3D array into a list of 2D arrays of dimensions: (1,H,W)
                
        model = [ np.squeeze( item ) for item in model ] # remove single-dimensional entries from the shape of an array
        
        if self.dilation_flag:      
            
            probe = self.binary_dilation_with_ellipse( probe )
            
        for enroll in model:
            
            if len( enroll.shape ) != 2 or len( probe.shape ) != 2: # check if input image is not of grayscale format
                raise Exception("The image must be a 2D array / grayscale format")
            
            if self.dilation_flag:
                
                enroll = self.binary_dilation_with_ellipse( enroll )
            
            match_result = match_template(enroll.astype(np.uint8), probe.astype(np.uint8), pad_input=True, mode='constant', constant_values=0)
            
            score = np.max(match_result)
            
            scores.append(score)
            
            
        score_mean = np.mean( scores )
        
        return score_mean



