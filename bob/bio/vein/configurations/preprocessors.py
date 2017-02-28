#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..preprocessor import FingerCrop

from ..preprocessors import TopographyCutRoi
from ..preprocessors import KMeansRoi
from ..preprocessors import PreNone
from ..preprocessors import ConstructAnnotations
from ..preprocessors import ManualRoi
from ..preprocessors import KMeansRoiFast

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

kmeans_roi_corr_rot_eroded_40 = KMeansRoi(filter_name = "median_filter", mask_size = 7,
                                            correct_mask_flag = True, correction_erosion_factor = 7,
                                            erode_mask_flag = True, erosion_factor = 40,
                                            convexity_flag = False,
                                            rotation_centering_flag = True)

kmeans_roi_corrected_eroded_rotated_40_scaled = KMeansRoi(filter_name = "median_filter", mask_size = 7,
                                                        correct_mask_flag = True, correction_erosion_factor = 7,
                                                        erode_mask_flag = True, erosion_factor = 40,
                                                        convexity_flag = False,
                                                        rotation_centering_flag = True,
                                                        normalize_scale_flag = True,
                                                        mask_to_image_area_ratio = 0.2)

kmeans_roi_corrected_eroded_rotated_40_eqlzd = KMeansRoi(filter_name = "median_filter", mask_size = 7,
                                                        correct_mask_flag = True, correction_erosion_factor = 7,
                                                        erode_mask_flag = True, erosion_factor = 40,
                                                        convexity_flag = False,
                                                        rotation_centering_flag = True,
                                                        normalize_scale_flag = False,
                                                        mask_to_image_area_ratio = 0.2,
                                                        equalize_adapthist_flag = True)

kmeans_roi_corrected_eroded_rotated_40_scaled_eqlzd = KMeansRoi(filter_name = "median_filter", mask_size = 7,
                                                                correct_mask_flag = True, correction_erosion_factor = 7,
                                                                erode_mask_flag = True, erosion_factor = 40,
                                                                convexity_flag = False,
                                                                rotation_centering_flag = True,
                                                                normalize_scale_flag = True,
                                                                mask_to_image_area_ratio = 0.2,
                                                                equalize_adapthist_flag = True)

kmeans_roi_corrected_eroded_rotated_40_scaled_fast = KMeansRoiFast(filter_name = "gaussian_filter", mask_size = 7,
                                                                    correct_mask_flag = True, correction_erosion_factor = 7,
                                                                    erode_mask_flag = True, erosion_factor = 40,
                                                                    convexity_flag = False,
                                                                    rotation_centering_flag = True,
                                                                    normalize_scale_flag = True,
                                                                    mask_to_image_area_ratio = 0.2,
                                                                    equalize_adapthist_flag = False,
                                                                    speedup_flag = True)


kmeans_roi_corrected_eroded_rotated_30_scaled_fast = KMeansRoiFast(filter_name = "gaussian_filter", mask_size = 7,
						                   correct_mask_flag = True, correction_erosion_factor = 7,
						                   erode_mask_flag = True, erosion_factor = 30,
						                   convexity_flag = False,
						                   rotation_centering_flag = True,
						                   normalize_scale_flag = True,
								   mask_to_image_area_ratio = 0.2,
								   equalize_adapthist_flag = False,
						                   speedup_flag = True)


putvein_kmeans_roi_centered_scaled_fast = KMeansRoiFast(erosion_factor=100, normalize_scale_flag=True,
						        correction_erosion_factor=7, erode_mask_flag=False,
						        mask_size=7, equalize_adapthist_flag=False, mask_to_image_area_ratio=0.2,
						        speedup_flag=True, rotation_centering_flag=False,
						        filter_name='gaussian_filter', correct_mask_flag=False, convexity_flag=False,
						        centering_flag = True)


putvein_kmeans_roi_centered_scaled_corrected_fast = KMeansRoiFast(erosion_factor=100, normalize_scale_flag=True,
                                correction_erosion_factor=7, erode_mask_flag=False,
                                mask_size=7, equalize_adapthist_flag=False, mask_to_image_area_ratio=0.2,
                                speedup_flag=True, rotation_centering_flag=False,
                                filter_name='gaussian_filter', correct_mask_flag=True, convexity_flag=False,
                                centering_flag = True)


prenone = PreNone()

constructannotations_center_rotate = ConstructAnnotations(center = True, rotate = True)
constructannotations_center = ConstructAnnotations(center = True, rotate = False)

manualroi = ManualRoi(erode_size = 0)
manualroi33 = ManualRoi(erode_size = 33)

