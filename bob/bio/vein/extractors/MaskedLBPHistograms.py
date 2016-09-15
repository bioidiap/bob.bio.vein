#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


#==============================================================================
# Import what is needed here:

import numpy as np

from bob.bio.base.extractor import Extractor

# for LBP routines:
import bob.ip.base

#==============================================================================
# Class implementation:

class MaskedLBPHistograms( Extractor ):
    """
    Class to compute an array of MCT (or MCT or LBP cancatenated with MCT) histograms for each pair of parameters: (radius, neighbors).
    The histograms are computed taking the binary mask / ROI into account.
    
    **Parameters:**
    neighbors : single uint value, or a list of uint values.
        Neighbours parameter / parameters in the LBP operator. Possible values: 4, 8.
    radius : single uint value, or a list of uint values. 
        Radius parameter / parameters in the LBP operator.
    to_average : bool
        [default: False] Compare the neighbors to the average of the pixels instead of the central pixel?
    add_average_bit : bool
        [default: False] (only useful if to_average is True) Add another bit to compare the central pixel to the average of the pixels?
    concatenate_lbp_mct: bool
        [default: False] (only useful if to_average=True and add_average_bit=True) 
        Compute both LBP and MCT histograms and concatenate them if set to True.
    """

    def __init__( self, neighbors, radius, to_average = False, add_average_bit = False, concatenate_lbp_mct = False ):

        Extractor.__init__( self, 
                           neighbors = neighbors,
                           radius = radius,
                           to_average = to_average,
                           add_average_bit = add_average_bit,
                           concatenate_lbp_mct = concatenate_lbp_mct )

        self.neighbors = neighbors
        self.radius = radius
        self.to_average = to_average
        self.add_average_bit = add_average_bit
        self.concatenate_lbp_mct = concatenate_lbp_mct

    #==========================================================================
    def compute_lbp_image( self, image, neighbors, radius, to_average, add_average_bit ):
        """
        Compute LBP or MCT image of the input image.
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input image.
        neighbors : single uint value, or a list of uint values.
            Neighbours parameter / parameters in the LBP operator. Possible values: 4, 8.
        radius : single uint value, or a list of uint values. 
            Radius parameter / parameters in the LBP operator.
        to_average : bool
            [default: False] Compare the neighbors to the average of the pixels instead of the central pixel?
        add_average_bit : bool
            [default: False] (only useful if to_average is True) Add another bit to compare the central pixel to the average of the pixels?
        
        **Returns:**
        
        lbp_image : 2D :py:class:`numpy.ndarray`
            LBP or MCT image.
        """
        
        # the size of the LBP image is the same as the size of the input image ('wrap' option):
        lbp_extractor = bob.ip.base.LBP ( neighbors = neighbors, radius = radius, border_handling = 'wrap',
                                         to_average = to_average,
                                         add_average_bit = add_average_bit )
        
        lbp_image = lbp_extractor(image)
        
        return lbp_image


    #==========================================================================
    def normalized_hist_given_mask( self, image, mask ):
        """
        This method computes the histogram of the input image taking the binary mask / ROI into account.
        The output histogram is also normalized making the sum of histogram entries equal to "1".
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input image.
        mask : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        
        **Returns:**
        
        hist_mask_norm : 1D :py:class:`numpy.ndarray`
            Normalized histogram of the input image given ROI.
        """
        
        image = image.astype(np.int64) # conversion is needed for np.histogram function
        mask = mask.astype(np.int64) # conversion is needed for np.histogram function
        
        image_max = np.max( image ) # max value in the image
        
        if image_max <= 15:
            bin_num = 2**4 + 1 # number of bins in the histogram for 4-bit encoded images
        elif image_max <= 255:
            bin_num = 2**8 + 1 # number of bins in the histogram for 8-bit encoded images
        elif image_max <= 511:
            bin_num = 2**9 + 1 # number of bins in the histogram for 8-bit encoded images
        elif image_max <= 65535:
            bin_num = 2**16 + 1 # number of bins in the histogram for 16-bit encoded images
        else:
            bin_num = image_max + 2 # number of bins in the histogram in other cases
        
        hist_mask = np.histogram( image, bins = np.arange( bin_num ), weights = mask )[0]
        
        hist_mask = hist_mask.astype( np.float )
        
        hist_mask_norm = hist_mask / sum( hist_mask )
        
        return hist_mask_norm.astype( np.float )

    #==========================================================================
    def __convert_arrays_list_to_array__( self, input_list ):
        """
        This function stacks a list of 1D arrays with variable lengths into 2D array.
        Zeros are substituted to the end of short vectors to make vectors equal in length.
        
        **Parameters:**
        
        input_list : a list of 1D vectors.
            Input list of 1D vectors with possibly unequal lengths.
        
        **Returns:**
        
        output_array : 2D :py:class:`numpy.ndarray`
            Output 2D array of the size ( N_arrays_in_input_list x Length_of_longest_array_in_input_list )
        """
        
        lengths = []
        
        for item in input_list:
            
            lengths.append( len( item ) )
        
        max_length = np.max( lengths )
        
        output_list = []
        
        for item in input_list:
            
            if len( item ) < max_length:
                
                item = np.hstack( [ item, np.zeros( max_length - len( item ) ) ] )
                
            output_list.append( item )
            
        output_array = np.vstack( output_list )
        
        output_array = np.squeeze( output_array ) # remove single dimensions
        
        return output_array

    #==========================================================================
    def __get_array_of_norm_hist__( self, image, mask, neighbors, radius, to_average, add_average_bit ):
        """
        Computes an array of LBP or MCT histograms given neighbors and radius lists.
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input image.
        mask : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        neighbors : a list of uint values.
            Neighbours parameter / parameters in the LBP operator. Possible values: 4, 8.
        radius : a list of uint values. 
            Radius parameter / parameters in the LBP operator.
        to_average : bool
            [default: False] Compare the neighbors to the average of the pixels instead of the central pixel?
        add_average_bit : bool
            [default: False] (only useful if to_average is True) Add another bit to compare the central pixel to the average of the pixels?
        
        **Returns:**
        
        hist_mask_norm_array : 1D or 2D :py:class:`numpy.ndarray`
            Normalized LBP or MCT histograms of the input image given ROI for each pair of parameters: (radius, neighbors).
        """
        
        hist_mask_norm_list = []
        
        for neighbors_val, radius_val in zip( neighbors, radius ):
            
            lbp_image = self.compute_lbp_image( image, neighbors_val, radius_val, to_average, add_average_bit ) # compute LBP image
            
            hist_mask_norm = self.normalized_hist_given_mask( lbp_image, mask ) # compute LBP histogram given mask
            
            hist_mask_norm_list.append( hist_mask_norm )
        
        hist_mask_norm_array = self.__convert_arrays_list_to_array__( hist_mask_norm_list )
        
        return hist_mask_norm_array


    #==========================================================================
    def masked_lbp_histograms( self, image, mask ):
        """
        Compute normalized LBP (or MCT or LBP cancatenated with MCT) histograms of the input image given ROI 
        for each pair of parameters: (radius, neighbors).
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input image.
        mask : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        
        **Returns:**
        
        hist_mask_norm_array : 1D or 2D :py:class:`numpy.ndarray`
            Normalized LBP (or MCT or LBP cancatenated with MCT) histograms of the input image given ROI 
            for each pair of parameters: (radius, neighbors).
        """
        
        if not( isinstance( self.radius, list ) ):
            self.radius = [ self.radius ]
        
        if not( isinstance( self.neighbors, list ) ):
            self.neighbors = [ self.neighbors ]
            
        if len(self.neighbors) != len(self.radius): # check if input image is of correct data type
            raise Exception("The number of elements in the neighbors and radius lists must be the same.")
        
        if self.to_average == True and self.add_average_bit == True and self.concatenate_lbp_mct == True:
            
            hist_mask_norm_array_lbp = self.__get_array_of_norm_hist__( image, mask, self.neighbors, self.radius,
                                                                      False, False )
            
            hist_mask_norm_array_mct = self.__get_array_of_norm_hist__( image, mask, self.neighbors, self.radius,
                                                                      self.to_average, self.add_average_bit )
            
            hist_mask_norm_array = np.hstack([ hist_mask_norm_array_lbp, hist_mask_norm_array_mct ])
            
        else:
            
            hist_mask_norm_array = self.__get_array_of_norm_hist__( image, mask, self.neighbors, self.radius,
                                                                   self.to_average, self.add_average_bit )
        return hist_mask_norm_array


    #==========================================================================
    def __call__( self, input_data ):
        """
        Compute an array of normalized LBP (or MCT or LBP cancatenated with MCT) histograms for each pair of parameters: (radius, neighbors).
        The histograms are computed taking the binary mask / ROI into account.
        """
        
        image = input_data[0] # Input image
        
        mask = input_data[1] # binary mask of the ROI
        
        return self.masked_lbp_histograms( image, mask )



