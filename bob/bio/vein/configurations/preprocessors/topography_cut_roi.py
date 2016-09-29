#!/usr/bin/env python

"""
The settings defined here are for ROI detection in the images of the BIOWAVE database.
"""
#==============================================================================
# Import what is needed here:

#import bob.bio.vein

from ...preprocessors import TopographyCutRoi

#==============================================================================
# Initialize the instance of the preprocessor:

# blob_xywh_offsets - defines the bounding box of the blob to be selected
# filter_name - filter the image before processing
# mask_size - size of the filter mask
# topography_step - thresholding step
# erode_mask_flag - erode the binary mask if True
# convexity_flag - make the mask binary if True

#preprocessor = bob.bio.vein.preprocessors.TopographyCutRoi.TopographyCutRoi( blob_xywh_offsets = [ 1, 1, 1, 1 ], 
#                                                                                     filter_name = "medianBlur", 
#                                                                                     mask_size = 7, 
#                                                                                     topography_step = 20, 
#                                                                                     erode_mask_flag = False, 
#                                                                                     convexity_flag = True )


preprocessor = TopographyCutRoi( blob_xywh_offsets = [ 1, 1, 1, 1 ], 
                                 filter_name = "medianBlur", 
                                 mask_size = 7, 
                                 topography_step = 20, 
                                 erode_mask_flag = False, 
                                 convexity_flag = True )



