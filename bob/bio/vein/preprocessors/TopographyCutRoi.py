#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


#==============================================================================
# Import what is needed here:
import numpy as np

from bob.bio.base.preprocessor import Preprocessor

#import cv2

from skimage.morphology import convex_hull_image

from scipy import ndimage

# to enable .png .jpg ... extensions in bob.io.base.load
import bob.io.image

#==============================================================================
# Class implementation:

class TopographyCutRoi( Preprocessor ):
    """
    Get the ROI using the topography cuts (iterative thresholding) followed by the largest blob selection within the
    predetermoned region of the input image.
    """
    
    #==========================================================================
    def __init__( self, blob_xywh_offsets = [1,1,1,1], 
                 filter_name = "medianBlur", mask_size = 7, 
                 topography_step = 20, erode_mask_flag = False, 
                 convexity_flag = True, **kwargs ):
        """
        
        Arguments:
        blob_xywh_offsets - a list of 4 values: [x_start, y_start, w_offset, h_offset], where
            x_start - is the x starting position of the blob bounding box,
            y_start - is the y starting position of the blob bounding box,
            w_offset - defines the x ending position of the blob bounding box as follows: x_end = image_width - w_offset,
            h_offset - defines the y ending position of the blob bounding box as follows: y_end = image_hight - h_offset.
            Default value: [1,1,1,1],
        filter_name - filter image before processing. Possible options: "GaussianBlur", "medianBlur",
        mask_size - size of the filetr mask. Defaults value: 7,
        topography_step - thresholding step. Default value: 20,
        erode_mask_flag - reduce the area of the mask / erode it if flag set to True. Default value: False,
        convexity_flag - make mask convex if True. Default value: True.
        **kwargs - .
        """
        
        Preprocessor.__init__( self, blob_xywh_offsets = blob_xywh_offsets, 
                              filter_name = filter_name, 
                              mask_size = mask_size, 
                              topography_step = topography_step, 
                              erode_mask_flag = erode_mask_flag, 
                              convexity_flag = convexity_flag, 
                              **kwargs )
        
        self.blob_xywh_offsets = blob_xywh_offsets
        self.filter_name = filter_name # the preprocessing filter name
        self.mask_size = mask_size # the size of the filter mask
        self.topography_step = topography_step # define the step between level cuts
        self.erode_mask_flag = erode_mask_flag # erode binary mask if True
        self.convexity_flag = convexity_flag # make mask convex if True
        
        self.filtered_image = np.array([], dtype = np.uint8) # store filtered image
        self.mask_binary = []
        self.stats_selected = [] # the statistics for the selected blob
        self.thresh_selected = [] # the selected threshold for the image
        self.retval_selected = [] # number of blobs
        self.centroid_selected = [] # centoid of the selected region
        self.labels_selected = [] # image of labels
        self.blob_idx_slected = [] # the blob index/value we select from the image of labels

    
    #==========================================================================
    def connectedComponentsWithStats( self, img_binary ):
        """
        This function is similar to the OpenCV function cv2.connectedComponentsWithStats().
        The implementation is based on scipy package and is not dependent on OpenCV.
        It finds the blobs and their statistics.
        
        Arguments:
        img_binary - binary image.
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
    def __filter_image__( self, image ):
        """
        Filter input image. Available filter options are GaussianBlur or medianBlur.
        medianBlur is selected by default.
        """
        if self.filter_name == "GaussianBlur":
            
#            self.filtered_image = cv2.GaussianBlur( image, (self.mask_size, self.mask_size), 1 ) # blur the image
            self.filtered_image = ndimage.gaussian_filter( image, sigma = 1 )
                        
        if self.filter_name == "medianBlur":
            
#            self.filtered_image = cv2.medianBlur( image, self.mask_size ) # filter the image
            self.filtered_image = ndimage.median_filter( image, self.mask_size )
            
        else:
            self.filtered_image = image

    #==========================================================================
    def get_topography_image( self, image ):
        """
        This method generates an image of topography cuts of the input. 
        """
        
        if not( isinstance( image, np.ndarray ) ): # check if input image is of correct data type
            raise Exception("Input image must be of the numpy.ndarray type")
        if image.shape[0] <= 0: # check if input image is not an empty array
            raise Exception("Empty array detected")
        
        self.__filter_image__(image) # filter image first

        thresh_start = np.min(self.filtered_image) + self.topography_step
        thresh_end = np.max(self.filtered_image) - self.topography_step
        
        img_geodesic = np.zeros(self.filtered_image.shape, dtype = np.uint8) # this image is just for showing the geodesical contours in the image        
        
        if thresh_start < 0 or thresh_end <= 0: # check if the thresholding limits are OK
            return img_geodesic
        
        for thresh in xrange( thresh_start, thresh_end, self.topography_step):
            img_geodesic[ self.filtered_image >= thresh ] = np.uint8( thresh )
            
        return img_geodesic
    
    #==========================================================================
    def __get_blob__( self, image ):
        """
        Find the blob with largest area satisfying our pre-set location and dimensions.
        
        Arguments:
        image - input image.
        """
        
        blob_xywh_limits = ( self.blob_xywh_offsets[ 0:2 ] +
                            list( np.array( [ image.shape[ 1 ], image.shape[ 0 ] ] ) - np.array( self.blob_xywh_offsets[ 2:4 ] ) ) )
        
        stats_result = [] # initialize the list to store the statistics about the blobs
        thresh_result = [] # list to store the threshoild values
        retval_result = [] # list to store the number of blobs found in each topography level
        labels_result = [] # list to store images of blob labels
        centroids_result = [] # centroids for each blob
        experiment_num = [] # idx values in the bottom loop
        blob_idx = [] # indexes of blobs in correct order
        
        thresh_start = np.min(self.filtered_image) + self.topography_step
        thresh_end = np.max(self.filtered_image) - self.topography_step
        
        self.mask_binary = np.zeros( self.filtered_image.shape, dtype=np.uint8 ) # this image is just for showing the geodesical contours in the image
        
        if thresh_start < 0 or thresh_end <= 0: # check if the thresholding limits are OK
            return
        
        for idx, thresh in enumerate( xrange(thresh_start, thresh_end, self.topography_step) ): # this is the main processing loop
        
            img_binary = np.zeros(self.filtered_image.shape, dtype=np.uint8) # initialize the binary image
            
            img_binary[ self.filtered_image >= thresh ] = 1 # set to 1, where the filtered_image is above threshold
            
            retval, labels, stats, centroids = self.connectedComponentsWithStats( img_binary ) # find the blobs and their statistics
            
            """
            The data in stats vector is returned as follows:
                CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
                CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
                CC_STAT_WIDTH The horizontal size of the bounding box
                CC_STAT_HEIGHT The vertical size of the bounding box
                CC_STAT_AREA The total area (in pixels) of the connected component
            """
            
            # create and fill the corresponding lists (just saves the results):
            thresh_vec = np.empty(retval); thresh_vec.fill(thresh)
            retval_vec = np.empty(retval); retval_vec.fill(retval)
            experiment_num_vec = np.empty(retval); experiment_num_vec.fill(idx)
            
            # append arrays to the corresponding lists (just saves the results):
            stats_result.append(stats)
            thresh_result.append(thresh_vec)
            retval_result.append(retval_vec)
            experiment_num.append(experiment_num_vec)
            labels_result.append(labels)    
            centroids_result.append(centroids)
            blob_idx.append( np.array( range( retval + 1 )[ 1: ] ) ) # save blob indexes for each experiment
        
        # make an array out of the list of arrays by stacking them vertically or horizontally:
        stats_result = np.vstack(stats_result)
        centroids_result = np.vstack(centroids_result)
        thresh_result = np.hstack(thresh_result)
        retval_result = np.hstack(retval_result)
        experiment_num = np.hstack(experiment_num)
        blob_idx = np.hstack(blob_idx)
        
        min_x = blob_xywh_limits[0]
        min_y = blob_xywh_limits[1]
        width = blob_xywh_limits[2] - min_x + 1
        height = blob_xywh_limits[3] - min_y + 1
        
        # Check if our blobs satisfy the "location and dimensions" conditions:
        # np.all(...) - logical AND along specified axis
        suitable_blobs_cond1 = np.all( np.hstack( [stats_result[:, 0:2] >= [min_x, min_y], stats_result[:, 2:4] <= [width, height]] ), axis = 1 )    
        
        stats_result[ suitable_blobs_cond1 == False, 4] = 0 # set the area of unwanted blobs to 0
        
        
        selected_blob_num = np.argmax( stats_result[:,4] ) # the blob we select as ROI
        self.stats_selected = stats_result[selected_blob_num, :] # the statistics for the selected blob
        self.thresh_selected = thresh_result[selected_blob_num] # the selected threshold for the image
        self.retval_selected = retval_result[selected_blob_num] # number of blobs
        experiment_selected = experiment_num[selected_blob_num] # the experiment number "idx" number in the above loop
        self.centroid_selected = centroids_result[selected_blob_num, :] # centoid of the selected region
        self.labels_selected = labels_result[ np.int( experiment_selected ) ] # image of labels
        self.blob_idx_slected = blob_idx[selected_blob_num] # the blob index/value we select from the image of labels
        
        self.mask_binary[ self.labels_selected == self.blob_idx_slected ] = 1
        
    #==========================================================================
    def __make_mask_convex__( self, image, k = 20 ):
        """
        Make the detected ROI of the convex hull shape
        """
        
        
        ellipse_mask = np.uint( image.shape[0] / k )
        if ellipse_mask % 2 == 0:
            ellipse_mask = np.uint(ellipse_mask + 1) # make the mask odd
        
        kernel = np.zeros( ( ellipse_mask, ellipse_mask ) )
        radius = ( ellipse_mask - 1 )/2
        y, x = np.ogrid[ -radius : radius + 1, -radius : radius + 1 ]
        mask = x**2 + y**2 <= radius**2
        kernel[ mask ] = 1
        
#        print("ellipse mask = {}".format(ellipse_mask))
#        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ellipse_mask, ellipse_mask))
        
#        self.mask_binary = cv2.morphologyEx(self.mask_binary, cv2.MORPH_CLOSE, kernel) # perform morphological closing on the image
        
        self.mask_binary = ndimage.binary_closing( self.mask_binary, structure = kernel )
        
        if np.sum(self.mask_binary):
            # added lines:
            # image padding:
            
            if self.erode_mask_flag:
                
                padded_mask = np.lib.pad(self.mask_binary, (k,), 'constant', constant_values=(0))
                # kernel and erode operation:
                kernel = np.ones((ellipse_mask*2+1,ellipse_mask*2+1),np.uint8)
                
#                self.mask_binary = cv2.erode(padded_mask,kernel,iterations = 1)[k:-k, k:-k]
                
                self.mask_binary = ndimage.binary_erosion( padded_mask, structure = kernel )[ k:-k, k:-k ]
                
                self.mask_binary = convex_hull_image(self.mask_binary)
                
            else:
                
                self.mask_binary = convex_hull_image(self.mask_binary)
                
    
    #==========================================================================
    def get_image_with_highlighted_ROI( self, image ):
        """
        Generate the the image with highlighted ROI. This is used for the visualization purposes only.
        
        Arguments:
        image - input image.
        """
        
        self.get_ROI( image )
        
        output_image = image.copy() # do not modify the original image, make a copy instead (!!!)
        
        output_image[ self.mask_binary == 1 ] = output_image[ self.mask_binary == 1 ] / 2
        
        return output_image
    
    #==========================================================================
    def __reset__( self ):
        """
        Reset the values defined in the __init__ method.
        """
        
        self.filtered_image = np.array([], dtype = np.uint8) # store filtered image
        self.mask_binary = []        
        self.stats_selected = [] # the statistics for the selected blob
        self.thresh_selected = [] # the selected threshold for the image
        self.retval_selected = [] # number of blobs
        self.centroid_selected = [] # centoid of the selected region
        self.labels_selected = [] # image of labels
        self.blob_idx_slected = [] # the blob index/value we select from the image of labels
   
    
    #==========================================================================        
    def get_ROI( self, image ):
        """
        Get the binary image of the ROI. The ROI should fit into the limits specified in the blob_xywh_offsets list.
        
        Arguments:
        image - input image.
        
        Return:
        Binary mask of the ROI.
        """
        
        self.__reset__()
        
        blob_xywh_limits = ( self.blob_xywh_offsets[ 0:2 ] +
                            list( np.array( [ image.shape[ 1 ], image.shape[ 0 ] ] ) - np.array( self.blob_xywh_offsets[ 2:4 ] ) ) )
        
        if not( isinstance( image, np.ndarray ) ): # check if input image is of correct data type
            raise Exception("Input image must be of the numpy.ndarray type")
        if image.shape[0] <= 0: # check if input image is not an empty array
            raise Exception("Empty array detected")
        if len(image.shape)!=2: # check if input image is not of grayscale format
            raise Exception("The image must be a 2D array / grayscale format")
            
        if not( isinstance(blob_xywh_limits, list) ):
            raise Exception("blob_xywh_limits must be a list")
        if  len(blob_xywh_limits) != 4:
            raise Exception("The length of the blob_xywh_limits must be equal to 4")
        
        self.__filter_image__( image )
        self.__get_blob__( image )
        
        if self.convexity_flag:
            self.__make_mask_convex__( image )
        
        return self.mask_binary.astype( np.uint8 )
    
    #==========================================================================
    def mask_the_image( self, image ):
        """
        Mask the input image with ROI.
        
        Arguments:
        image - input image.
        
        Return:
        Masked image.
        """
        
        return image * self.get_ROI( image )
        
    
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









'''

import TopographyCutRoi
reload ( TopographyCutRoi )

extractor = TopographyCutRoi.TopographyCutRoi()

preprocessor_biowave = TopographyCutRoi.TopographyCutRoi(  blob_xywh_offsets = [ 10, 0, 10, 0 ], 
                                                                     filter_name = "medianBlur", 
                                                                     mask_size = 7, 
                                                                     topography_step = 20, 
                                                                     erode_mask_flag = False, 
                                                                     convexity_flag = True )

roi = preprocessor_biowave(image)

plt.figure()
plt.imshow( roi, cmap = 'gray' ), plt.xticks([]), plt.yticks([])
plt.show()

'''







