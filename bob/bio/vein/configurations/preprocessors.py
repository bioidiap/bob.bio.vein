#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..preprocessor import FingerCrop

from ..preprocessors import TopographyCutRoi
from ..preprocessors import KMeansRoi
from ..preprocessors import PreNone
from ..preprocessors import ConstructAnnotations
from ..preprocessors import ManualRoi

none = FingerCrop()
he = FingerCrop(postprocessing='HE')
hfe = FingerCrop(postprocessing='HFE')
circgabor = FingerCrop(postprocessing='CircGabor')

topography_cut_roi_conv = TopographyCutRoi( blob_xywh_offsets = [ 1, 1, 1, 1 ], 
                                 filter_name = "median_filter", 
                                 mask_size = 7, 
                                 topography_step = 20, 
                                 erode_mask_flag = False, 
                                 convexity_flag = True )

topography_cut_roi_conv_erode = TopographyCutRoi( blob_xywh_offsets = [ 1, 1, 1, 1 ], 
                                 filter_name = "median_filter", 
                                 mask_size = 7, 
                                 topography_step = 20, 
                                 erode_mask_flag = True, 
                                 convexity_flag = True )

topography_cut_roi = TopographyCutRoi( blob_xywh_offsets = [ 1, 1, 1, 1 ], 
                                 filter_name = "median_filter", 
                                 mask_size = 7, 
                                 topography_step = 20, 
                                 erode_mask_flag = False, 
                                 convexity_flag = False )

kmeans_roi_conv = KMeansRoi( filter_name = "median_filter", 
                                 mask_size = 7, 
                                 erode_mask_flag = False, 
                                 convexity_flag = True )

kmeans_roi = KMeansRoi( filter_name = "median_filter", 
                                 mask_size = 7, 
                                 erode_mask_flag = False, 
                                 convexity_flag = False )

kmeans_roi_conv_erode_40 = KMeansRoi( filter_name = "median_filter", 
                                 mask_size = 7, 
                                 erode_mask_flag = True, 
                                 erosion_factor = 40, 
                                 convexity_flag = True )

kmeans_roi_corrected_eroded_40 = KMeansRoi(filter_name = "median_filter", mask_size = 7, 
                                            correct_mask_flag = True, correction_erosion_factor = 7,
                                            erode_mask_flag = True, erosion_factor = 40, 
                                            convexity_flag = False)

prenone = PreNone()

constructannotations_center_rotate = ConstructAnnotations(center = True, rotate = True)
constructannotations_center = ConstructAnnotations(center = True, rotate = False)

manualroi = ManualRoi(erode_size = 0)
manualroi33 = ManualRoi(erode_size = 33)

