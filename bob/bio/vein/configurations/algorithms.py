#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..algorithm import MiuraMatch
from ..algorithms import MiuraMatchAligned
from ..algorithms import HistogramsMatch

from ..algorithms import AnnotationMatch

from ..algorithms import MatchTemplate

from ..algorithms import MiuraMatchFusion

from ..algorithms import CrossCorrelationMatch

from ..algorithms import MiuraMatchMaxEigenvalues

from ..algorithms import MiuraMatchRotation

from ..algorithms import MiuraMatchRotationFast

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

miura_match_fusion_120_max = MiuraMatchFusion(ch = 120, cw = 120, score_fusion_method = 'max')
miura_match_fusion_300_225_max = MiuraMatchFusion(ch = 300, cw = 225, score_fusion_method = 'max')

cross_correlation_match_mean1_fusion = CrossCorrelationMatch(is_mean_normalize_flag = False, score_fusion_method = 'mean')
cross_correlation_match_mean1_fusion_mean_norm = CrossCorrelationMatch(is_mean_normalize_flag = True, score_fusion_method = 'mean')

miura_match_max_eigenvalues_120_mean1 = MiuraMatchMaxEigenvalues(ch = 120, cw = 120, score_fusion_method = 'mean')

miura_match_120_rotation_10_step_1_max1 = MiuraMatchRotation(ch = 120, cw = 120, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')

miura_match_fast_120_rotation_10_step_1_pert_1_max1 = MiuraMatchRotationFast(ch = 120, cw = 120, angle_limit = 10, angle_step = 1, perturbation_matching_flag = True, kernel_radius = 1, score_fusion_method = 'max')

miura_match_fast_120_rotation_10_step_1_pert_2_max1 = MiuraMatchRotationFast(ch = 120, cw = 120, angle_limit = 10, angle_step = 1, perturbation_matching_flag = True, kernel_radius = 2, score_fusion_method = 'max')

miura_match_fast_120_rotation_10_step_1_pert_3_max1 = MiuraMatchRotationFast(ch = 120, cw = 120, angle_limit = 10, angle_step = 1, perturbation_matching_flag = True, kernel_radius = 3, score_fusion_method = 'max')


miura_match_fast_70_rotation_10_step_1_max1 = MiuraMatchRotationFast(ch = 70, cw = 70, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')
miura_match_fast_80_rotation_10_step_1_max1 = MiuraMatchRotationFast(ch = 80, cw = 80, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')
miura_match_fast_90_rotation_10_step_1_max1 = MiuraMatchRotationFast(ch = 90, cw = 90, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')
miura_match_fast_100_rotation_10_step_1_max1 = MiuraMatchRotationFast(ch = 100, cw = 100, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')
miura_match_fast_110_rotation_10_step_1_max1 = MiuraMatchRotationFast(ch = 110, cw = 110, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')
miura_match_fast_120_rotation_10_step_1_max1 = MiuraMatchRotationFast(ch = 120, cw = 120, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')
miura_match_fast_130_rotation_10_step_1_max1 = MiuraMatchRotationFast(ch = 130, cw = 130, angle_limit = 10, angle_step = 1, score_fusion_method = 'max')

miura_match_fast_120_rotation_15_step_1_max1 = MiuraMatchRotationFast(ch = 120, cw = 120, angle_limit = 15, angle_step = 1, score_fusion_method = 'max')

miura_match_fast_120_rotation_10_step_1_5_max1 = MiuraMatchRotationFast(ch = 120, cw = 120, angle_limit = 10, angle_step = 1.5, score_fusion_method = 'max')
miura_match_fast_120_rotation_10_step_2_max1 = MiuraMatchRotationFast(ch = 120, cw = 120, angle_limit = 10, angle_step = 2, score_fusion_method = 'max')


putvein_miura_match_fast_300_200_rotation_5_step_1_pert_1_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=5, kernel_radius=1, cw=200)

putvein_miura_match_fast_300_200_rotation_5_step_1_pert_2_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=5, kernel_radius=2, cw=200)

putvein_miura_match_fast_300_200_rotation_5_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=5, kernel_radius=3, cw=200)

putvein_miura_match_fast_300_200_rotation_7_step_1_pert_1_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=7, kernel_radius=1, cw=200)

putvein_miura_match_fast_300_200_rotation_7_step_1_pert_2_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=7, kernel_radius=2, cw=200)

putvein_miura_match_fast_300_200_rotation_7_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=7, kernel_radius=3, cw=200)

putvein_miura_match_fast_300_200_rotation_10_step_1_pert_1_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=1, cw=200)

putvein_miura_match_fast_300_200_rotation_10_step_1_pert_2_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=2, cw=200)

putvein_miura_match_fast_300_200_rotation_10_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=3, cw=200)



putvein_miura_match_fast_200_150_rotation_10_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=200, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=3, cw=150)

putvein_miura_match_fast_233_175_rotation_10_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=233, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=3, cw=175)

putvein_miura_match_fast_267_200_rotation_10_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=267, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=3, cw=200)




putvein_miura_match_fast_300_225_rotation_10_step_1_pert_1_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=1, cw=225)

putvein_miura_match_fast_300_225_rotation_10_step_1_pert_2_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=2, cw=225)

putvein_miura_match_fast_300_225_rotation_10_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=3, cw=225)

putvein_miura_match_fast_300_225_rotation_10_step_1_pert_4_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=4, cw=225)

putvein_miura_match_fast_300_225_rotation_10_step_1_pert_5_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=5, cw=225)




putvein_miura_match_fast_333_250_rotation_10_step_1_pert_3_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=333, angle_step=1,
                                   							perturbation_matching_flag=True, angle_limit=10, kernel_radius=3, cw=250)

putvein_miura_match_fast_300_225_rotation_10_step_1_gray_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=300, angle_step=1,
                                   													   perturbation_matching_flag=False, angle_limit=10, kernel_radius=3, cw=225, gray_scale_input_flag = True)

miura_match_fast_120_120_rotation_10_step_1_gray_max1 = MiuraMatchRotationFast(score_fusion_method='max', ch=120, angle_step=1,
                                   													   perturbation_matching_flag=False, angle_limit=10, kernel_radius=3, cw=120, gray_scale_input_flag = True)

miura_match_fast_120_120_rotation_10_step_2_gray_mean1 = MiuraMatchRotationFast(score_fusion_method='mean', ch=120, angle_step=2,
                                   													   perturbation_matching_flag=False, angle_limit=10, kernel_radius=3, cw=120, gray_scale_input_flag = True)






















