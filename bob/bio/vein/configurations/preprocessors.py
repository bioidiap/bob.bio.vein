#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..preprocessors import FingerCrop
from ..preprocessors.TopographyCutRoi import TopographyCutRoi
from ..preprocessors.KMeansRoi import KMeansRoi
none = FingerCrop()
he = FingerCrop(postprocessing='HE')
hfe = FingerCrop(postprocessing='HFE')
circgabor = FingerCrop(postprocessing='CircGabor')

topography_cut_roi = TopographyCutRoi( blob_xywh_offsets = [ 1, 1, 1, 1 ], 
                                 filter_name = "medianBlur", 
                                 mask_size = 7, 
                                 topography_step = 20, 
                                 erode_mask_flag = False, 
                                 convexity_flag = True )

kmeans_roi = TopographyCutRoi( filter_name = "medianBlur", 
                                 mask_size = 7, 
                                 erode_mask_flag = False, 
                                 convexity_flag = True )
