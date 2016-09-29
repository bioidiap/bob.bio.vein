#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


#==============================================================================
# Import what is needed here:
import numpy as np

from bob.bio.base.preprocessor import Preprocessor

from .TopographyCutRoi import  TopographyCutRoi

import bob.learn.em # for the k-means

#==============================================================================
# Class implementation:

class KMeansRoi( TopographyCutRoi, Preprocessor ):
    """
    K-means based ROI extraction algorithm. First 2 centroids are obtained for the gray-scale values of the pixels in the input image.
    Next image threshold is calculated as an average of the centroids.
    The largest binary region is then selected and smoothed using morphological operations.
    """
    
    #==========================================================================
    def __init__( self, filter_name = "medianBlur", mask_size = 7, erode_mask_flag = False, erosion_factor = 20, 
                 convexity_flag = True, **kwargs ):
        """
        
        Arguments:
        filter_name - filter image before processing. Possible options: "GaussianBlur", "medianBlur",
        mask_size - size of the filetr mask. Defaults value: 7,
        erode_mask_flag - reduce the area of the mask / erode it if flag set to True. Default value: False,
        erosion_factor - the mask will be eroded with elipse kernel of the size: (image_heights/erosion_factor). Default value: 20,
        convexity_flag - make mask convex if True. Default value: True.
        **kwargs - .
        """
        
        # initialize the TopographyCutRoi class:
        TopographyCutRoi.__init__( self,
                                  filter_name = filter_name,
                                  mask_size = mask_size,
                                  erode_mask_flag = erode_mask_flag,
				  erosion_factor = erosion_factor,
                                  convexity_flag = convexity_flag,
                                  **kwargs )
        
        Preprocessor.__init__( self, 
			      filter_name = filter_name, 
                              mask_size = mask_size, 
                              erode_mask_flag = erode_mask_flag, 
			      erosion_factor = erosion_factor, 
                              convexity_flag = convexity_flag, 
                              **kwargs )
        
        self.filter_name = filter_name # the preprocessing filter name
        self.mask_size = mask_size # the size of the filter mask
        self.erode_mask_flag = erode_mask_flag # erode binary mask if True
        self.erosion_factor = erosion_factor
        self.convexity_flag = convexity_flag # make mask convex if True
        
        self.filtered_image = np.array([], dtype = np.uint8) # store filtered image
        self.mask_binary = []
#        self.stats_selected = [] # the statistics for the selected blob
#        self.thresh_selected = [] # the selected threshold for the image
#        self.retval_selected = [] # number of blobs
#        self.centroid_selected = [] # centoid of the selected region
#        self.labels_selected = [] # image of labels
#        self.blob_idx_slected = [] # the blob index/value we select from the image of labels


    #==========================================================================
    def __get_blob__( self, image ):
        """
        Find the blob with largest area.
        
        Arguments:
        image - input image.
        """
        
        k = 2 # number of the clusters is set to be 2
        
        data = self.filtered_image.flatten()
        
        data = np.delete( data, np.where( data == 255 ) ) # remove oversaturated pixels
        
        data = data.reshape( data.shape[ 0 ], 1 )
        
        data = data.astype( np.float64 ) # must be float64 for the bob k-means
        
        kmeans = bob.learn.em.KMeansMachine( k , data.shape[1] ) # k clusters with a feature dimensionality of N
        
        kmeansTrainer = bob.learn.em.KMeansTrainer( initialization_method = 'KMEANS_PLUS_PLUS' )
        
        bob.learn.em.train( kmeansTrainer, kmeans, data, max_iterations = 100, convergence_threshold = 1e-5 ) # Train the KMeansMachine
        
        decision_boundary = np.mean( kmeans.means )
        
        labelled_image = np.zeros( image.shape )
        
        labelled_image[ self.filtered_image > decision_boundary ] = 1
        
        retval, labels, stats, centroids = self.connectedComponentsWithStats( labelled_image )
        
        selected_blob_idx = np.argmax( stats[:,4] ) + 1
        
        self.mask_binary = np.zeros( image.shape, dtype = np.uint8 ) # this
        
        self.mask_binary[ labels == selected_blob_idx ] = 1


    #==========================================================================        
    def get_ROI( self, image ):
        """
        Get the binary image of the ROI.
        
        Arguments:
        image - input image.
        
        Return:
        Binary mask of the ROI.
        """
        
        self.__reset__()
        
        if not( isinstance( image, np.ndarray ) ): # check if input image is of correct data type
            raise Exception("Input image must be of the numpy.ndarray type")
        if image.shape[0] <= 0: # check if input image is not an empty array
            raise Exception("Empty array detected")
        if len(image.shape)!=2: # check if input image is not of grayscale format
            raise Exception("The image must be a 2D array / grayscale format")
        
        
        self.__filter_image__( image )
        self.__get_blob__( image )
        
        if self.convexity_flag:
            self.__make_mask_convex__( image, self.erosion_factor )
        
        return self.mask_binary.astype( np.uint8 )


    #==========================================================================
    def __call__(self, image, annotations):
        """
        Get the binary image of the ROI. The ROI should fit into the limits specified in the blob_xywh_offsets list.
        
        Arguments:
        image - input image.
        
        Return:
        image - original image,
        binary roi / mask for the image.
        """
        
        return ( image, self.get_ROI( image ) )
    
    #==========================================================================
    def write_data( self, data, file_name ):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_data() method of the Preprocessor class.
        
        Arguments:
        data - data returned by the __call__ method of the class,
        file_name - name of the file
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
        
        Arguments:
        file_name - name of the file.
        
        Return:
        
        """
        f = bob.io.base.HDF5File( file_name, 'r' )
        image = f.read( 'image' )
        mask = f.read( 'mask' )
        del f
        return ( image, mask )







