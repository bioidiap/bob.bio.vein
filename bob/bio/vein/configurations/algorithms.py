#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..algorithm import MiuraMatch
from ..algorithms import MiuraMatchAligned
from ..algorithms import HistogramsMatch

# HessianAlignment - supportive class (returns transformation matrix for the alignment of enroll and probe) for the HessianHistMatchAligned class
from ..algorithms import HessianAlignment
from ..algorithms import HessianHistMatchAligned

from ..algorithms import AnnotationMatch

from ..algorithms import MatchTemplate

from ..algorithms import KeypointsMatcher

from ..algorithms import MiuraMatchFusion

huangwl = MiuraMatch(ch=18, cw=28)
miuramax = MiuraMatch(ch=80, cw=90)
miurarlt = MiuraMatch(ch=65, cw=55)

miura_wrist_20 = MiuraMatch( ch = 20, cw = 20 )
miura_wrist_40 = MiuraMatch( ch = 40, cw = 40 )
miura_wrist_60 = MiuraMatch( ch = 60, cw = 60 )
miura_wrist_80 = MiuraMatch( ch = 80, cw = 80 )
miura_wrist_100 = MiuraMatch( ch = 100, cw = 100 )
miura_wrist_120 = MiuraMatch( ch = 120, cw = 120 )
miura_wrist_140 = MiuraMatch( ch = 140, cw = 140 )
miura_wrist_160 = MiuraMatch( ch = 160, cw = 160 )

miura_wrist_aligned_20 = MiuraMatchAligned( ch = 20, cw = 20, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_40 = MiuraMatchAligned( ch = 40, cw = 40, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_60 = MiuraMatchAligned( ch = 60, cw = 60, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_80 = MiuraMatchAligned( ch = 80, cw = 80, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_100 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_120 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_140 = MiuraMatchAligned( ch = 140, cw = 140, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_160 = MiuraMatchAligned( ch = 160, cw = 160, alignment_flag = True, alignment_method = "center_of_mass" )

miura_wrist_dilation_5 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 5 )
miura_wrist_dilation_7 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 7 )
miura_wrist_dilation_9 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 9 )
miura_wrist_dilation_11 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 11 )
miura_wrist_dilation_13 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 13 )
miura_wrist_dilation_15 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 15 )
miura_wrist_dilation_17 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 17 )

miura_wrist_chw_120_dilation_5 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 5 )
miura_wrist_chw_120_dilation_7 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 7 )
miura_wrist_chw_120_dilation_9 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 9 )
miura_wrist_chw_120_dilation_11 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 11 )
miura_wrist_chw_120_dilation_13 = MiuraMatchAligned( ch = 120, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 13 )


mm_t1 = MiuraMatchAligned( ch = 140, cw = 140, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 9 )
mm_t2 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 9 )
mm_t3 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 13 )
mm_t4 = MiuraMatchAligned( ch = 120, cw = 130, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 5 )
chi_square = HistogramsMatch( similarity_metrics_name = "chi_square" )

# Successfull parameters for the biowave_test DB:
window_size = 20
enroll_center_method = "largest_vector_magnitude"
N_points = 15
gap = 10
step = 7
align_power = 4

aligner = HessianAlignment( window_size = window_size, N_points = N_points, gap = gap, step = step, align_power = align_power, 
                           enroll_center_method = enroll_center_method )


hessian_hist_match_aligned_nb50p1 = HessianHistMatchAligned( aligner = aligner, n_bins = 50, eigenval_power = 1, 
								binarize_weights = False, 
								similarity_metrics_name = "chi_square", 
								alignment_method_name = "max_dot_product" )

hessian_hist_match_aligned_nb50p2 = HessianHistMatchAligned( aligner = aligner, n_bins = 50, eigenval_power = 2, 
								binarize_weights = False, 
								similarity_metrics_name = "chi_square", 
								alignment_method_name = "max_dot_product" )

hessian_hist_match_aligned_nb20p1 = HessianHistMatchAligned( aligner = aligner, n_bins = 20, eigenval_power = 1, 
								binarize_weights = False, 
								similarity_metrics_name = "chi_square", 
								alignment_method_name = "max_dot_product" )

hessian_hist_match_aligned_nb20p2 = HessianHistMatchAligned( aligner = aligner, n_bins = 20, eigenval_power = 2, 
								binarize_weights = False, 
								similarity_metrics_name = "chi_square", 
								alignment_method_name = "max_dot_product" )

hessian_hist_match_aligned_nb50p1bin = HessianHistMatchAligned( aligner = aligner, n_bins = 50, eigenval_power = 1, 
								binarize_weights = True, 
								similarity_metrics_name = "chi_square", 
								alignment_method_name = "max_dot_product" )

hessian_hist_match_aligned_nb20p1bin = HessianHistMatchAligned( aligner = aligner, n_bins = 20, eigenval_power = 1, 
								binarize_weights = True, 
								similarity_metrics_name = "chi_square", 
								alignment_method_name = "max_dot_product" )




annotationmatch_0_min = AnnotationMatch(sigma=0, score_method='min')
annotationmatch_1_min = AnnotationMatch(sigma=1, score_method='min')
annotationmatch_2_min = AnnotationMatch(sigma=2, score_method='min')
annotationmatch_3_min = AnnotationMatch(sigma=3, score_method='min')
annotationmatch_4_min = AnnotationMatch(sigma=4, score_method='min')
annotationmatch_5_min = AnnotationMatch(sigma=5, score_method='min')
annotationmatch_6_min = AnnotationMatch(sigma=6, score_method='min')
annotationmatch_7_min = AnnotationMatch(sigma=7, score_method='min')

annotationmatch_0_max = AnnotationMatch(sigma=0, score_method='max')
annotationmatch_1_max = AnnotationMatch(sigma=1, score_method='max')
annotationmatch_2_max = AnnotationMatch(sigma=2, score_method='max')
annotationmatch_3_max = AnnotationMatch(sigma=3, score_method='max')
annotationmatch_4_max = AnnotationMatch(sigma=4, score_method='max')
annotationmatch_5_max = AnnotationMatch(sigma=5, score_method='max')
annotationmatch_6_max = AnnotationMatch(sigma=6, score_method='max')
annotationmatch_7_max = AnnotationMatch(sigma=7, score_method='max')

annotationmatch_0_mean = AnnotationMatch(sigma=0, score_method='mean')
annotationmatch_1_mean = AnnotationMatch(sigma=1, score_method='mean')
annotationmatch_2_mean = AnnotationMatch(sigma=2, score_method='mean')
annotationmatch_3_mean = AnnotationMatch(sigma=3, score_method='mean')
annotationmatch_4_mean = AnnotationMatch(sigma=4, score_method='mean')
annotationmatch_5_mean = AnnotationMatch(sigma=5, score_method='mean')
annotationmatch_6_mean = AnnotationMatch(sigma=6, score_method='mean')
annotationmatch_7_mean = AnnotationMatch(sigma=7, score_method='mean')

match_template_dilation_0 = MatchTemplate(dilation_flag = False, ellipse_mask_size = 5)
match_template_dilation_5 = MatchTemplate(dilation_flag = True, ellipse_mask_size = 5)
match_template_dilation_7 = MatchTemplate(dilation_flag = True, ellipse_mask_size = 7)
match_template_dilation_9 = MatchTemplate(dilation_flag = True, ellipse_mask_size = 9)
match_template_dilation_11 = MatchTemplate(dilation_flag = True, ellipse_mask_size = 11)

akaze_keypoints_matcher_075 = KeypointsMatcher(ratio_to_match = 0.75)

miura_match_fusion_120_max = MiuraMatchFusion(ch = 120, cw = 120, score_fusion_method = 'max')
miura_match_fusion_120_median = MiuraMatchFusion(ch = 120, cw = 120, score_fusion_method = 'median')
miura_match_fusion_120_adaptive_mean = MiuraMatchFusion(ch = 120, cw = 120, score_fusion_method = 'adaptive_mean')






































