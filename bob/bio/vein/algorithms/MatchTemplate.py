#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np

from bob.bio.base.algorithm import Algorithm

from scipy import ndimage

from skimage.feature import match_template

class MatchTemplate(Algorithm):
    """
    This class is designed for matching and computing of the similarity score of two binary images. 
    The matching is composed of the following steps:
        
        1. First the ``enroll`` and ``probe`` images are aligned using the ``skimage.feature.match_template``.
        
        2. Once ``probe`` is aligned to the ``enroll`` the data in both images is masked 
           using the rectangular binary mask of the joint ROI.
           
    Once matching/alignment is done the similarity score is computed as follows:
        
        score = 2 * (intersection of ``enroll`` and ``probe``) / (area of ``enroll`` + area of ``probe``)
    
    **Parameters:**
    
    ``dilation_flag`` : :py:class:`bool`
        If set to "True" binary dilation of the images is done before the matching. Default value: False.
    
    ``ellipse_mask_size`` : :py:class:`int`
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
        
        ``enroll_features`` : [object]
            A list of features used for the enrollment of one model.
        
        **Returns:**
        
        ``model`` : object
            The model enrolled from the ``enroll_features``.
            Must be writable with the :py:meth:`write_model` function and readable with the :py:meth:`read_model` function.
        """
        
        return enroll_features # Do nothing in our case
        
        
    def binary_dilation_with_ellipse( self, image ):
        """
        binary_dilation_with_ellipse( image, ellipse_mask_size ) -> image_dilated
        
        Dilates an input binary image with the ellipse kernel of the size ``ellipse_mask_size``.
        
        **Parameters:**
        
        ``image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.
            
        **Returns:**
        
        ``image_dilated`` : 2D :py:class:`numpy.ndarray`
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
            
            
    def generate_mask(self, image):
        """
        Generate the binary mask for the input binary image based on the coordinates
        of the non-zero elements in the input.
        
        **Parameters:**
        
        ``image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.
            
        **Returns:**
        
        ``mask`` : 2D :py:class:`numpy.ndarray`
            A rectangular binary mask covering non-zero elements in the input image.
        """
        y, x = np.nonzero(image.astype(np.uint8)) 
        
        y_min = np.min(y)
        y_max = np.max(y)
        
        x_min = np.min(x)
        x_max = np.max(x)
        
        mask = np.zeros(image.shape)
        
        mask[y_min:y_max, x_min:x_max] = 1
        
        return mask
        
        
    def align_and_mask_enroll_probe(self, enroll, probe):
        """
        This function aligns two binary images using cross-corelation approach 
        implemented in ``skimage.feature.match_template``. ``probe`` is aligned/shifted to 
        the ``enroll``. The data/veins in the binary images are also masked after the alignment
        using the rectangular binary mask of the joint ROI.
        
        **Parameters:**
        
        ``image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.
            
        **Returns:**
        
        ``enroll_masked`` : 2D :py:class:`numpy.ndarray`
            Masked enroll image.
        
        ``probe_shifted_masked`` : 2D :py:class:`numpy.ndarray`
            Masked and shifted (aligned to the enroll) probe image.
        """
        
        if enroll.dtype != np.uint8:
            enroll = enroll.astype(np.uint8)
            probe = probe.astype(np.uint8)
            
        match_result = match_template(enroll, probe, pad_input=True, mode='constant', constant_values=0)
        
        shift = np.array(np.unravel_index(match_result.argmax(), match_result.shape)) - np.array(match_result.shape)/2
        
        shift = shift.astype(np.int)
        
        probe_shifted = ndimage.interpolation.shift( probe, shift, cval = 0 )
        
        mask_enroll = self.generate_mask(enroll) # rectangular mask
        
        mask_probe = self.generate_mask(probe) # rectangular mask
        
        mask_probe_shifted = ndimage.interpolation.shift( mask_probe, shift, cval = 0 )
        
        mask_probe_shifted[mask_probe_shifted>0.5] = 1
        mask_probe_shifted[mask_probe_shifted!=1] = 0
        
        mask_overlap = mask_enroll * mask_probe_shifted
        
        enroll_masked = enroll*mask_overlap
        
        probe_shifted_masked = probe_shifted*mask_overlap
        
        enroll_masked[enroll_masked>0.5] = 1
        enroll_masked[enroll_masked!=1] = 0

        probe_shifted_masked[probe_shifted_masked>0.5] = 1
        probe_shifted_masked[probe_shifted_masked!=1] = 0
        
        return (enroll_masked.astype(np.uint8), probe_shifted_masked.astype(np.uint8))
        
        
    def score(self, model, probe):
        """
        score(model, probe) -> score
        
        Computes the score of the probe and the model.
        Score has a value between 0 and 1, larger value is better match.
        
        **Parameters:**
        
        ``model`` : 2D/3D :py:class:`numpy.ndarray`
            The model enrolled by the :py:meth:`enroll` function.
        
        ``probe`` : 2D :py:class:`numpy.ndarray`
            The probe read by the :py:meth:`read_probe` function.
        
        **Returns:**
        
        ``score_mean`` : :py:class:`float`
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
            
            (enroll_masked, probe_shifted_masked) = self.align_and_mask_enroll_probe(enroll, probe)
            
            score = 2.0*np.sum( enroll_masked * probe_shifted_masked ) / ( np.sum(enroll_masked) + np.sum(probe_shifted_masked) )
            
#            match_result = match_template(enroll_masked, probe_shifted_masked, pad_input=True, mode='constant', constant_values=0)
            
            scores.append( score )
            
        score_mean = np.mean( scores )
        
        return score_mean



