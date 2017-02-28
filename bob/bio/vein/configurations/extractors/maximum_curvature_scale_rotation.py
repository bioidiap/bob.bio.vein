#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import MaximumCurvatureScaleRotation


maximum_curvature_scale_rotation_p2pnorm_90 = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 90, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1)
maximum_curvature_scale_rotation_p2pnorm_100 = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 100, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1)
maximum_curvature_scale_rotation_p2pnorm_110 = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 110, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1)
maximum_curvature_scale_rotation_p2pnorm_120 = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 120, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1)
maximum_curvature_scale_rotation_p2pnorm_130 = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 130, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1)

maximum_curvature_scale_rotation_p2pnorm_90_fast = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 90, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1, speed_up_flag=True)
maximum_curvature_scale_rotation_p2pnorm_100_fast = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 100, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1, speed_up_flag=True)
maximum_curvature_scale_rotation_p2pnorm_110_fast = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 110, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1, speed_up_flag=True)
maximum_curvature_scale_rotation_p2pnorm_120_fast = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 120, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1, speed_up_flag=True)
maximum_curvature_scale_rotation_p2pnorm_130_fast = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 130, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1, speed_up_flag=True)

maximum_curvature_scale_rotation_p2pnorm_rot_im = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = True, selected_mean_dist = 100, sum_of_rotated_images_flag = True, angle_limit = 10, angle_step = 1)

maximum_curvature_scale_rotation_rot_im = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = False, selected_mean_dist = 100, sum_of_rotated_images_flag = True, angle_limit = 10, angle_step = 1)

putvein_maximum_curvature_sigma_7 = MaximumCurvatureScaleRotation(selected_mean_dist=200, angle_step=1, norm_p2p_dist_flag=False, sum_of_rotated_images_flag=False, speed_up_flag=False, angle_limit=10, sigma=7)
putvein_maximum_curvature_sigma_9 = MaximumCurvatureScaleRotation(selected_mean_dist=200, angle_step=1, norm_p2p_dist_flag=False, sum_of_rotated_images_flag=False, speed_up_flag=False, angle_limit=10, sigma=9)
putvein_maximum_curvature_sigma_11 = MaximumCurvatureScaleRotation(selected_mean_dist=200, angle_step=1, norm_p2p_dist_flag=False, sum_of_rotated_images_flag=False, speed_up_flag=False, angle_limit=10, sigma=11)
putvein_maximum_curvature_sigma_13 = MaximumCurvatureScaleRotation(selected_mean_dist=200, angle_step=1, norm_p2p_dist_flag=False, sum_of_rotated_images_flag=False, speed_up_flag=False, angle_limit=10, sigma=13)
putvein_maximum_curvature_sigma_15 = MaximumCurvatureScaleRotation(selected_mean_dist=200, angle_step=1, norm_p2p_dist_flag=False, sum_of_rotated_images_flag=False, speed_up_flag=False, angle_limit=10, sigma=15)
putvein_maximum_curvature_sigma_17 = MaximumCurvatureScaleRotation(selected_mean_dist=200, angle_step=1, norm_p2p_dist_flag=False, sum_of_rotated_images_flag=False, speed_up_flag=False, angle_limit=10, sigma=17)

putvein_maximum_curvature_sigma_9_p2p_norm = MaximumCurvatureScaleRotation(selected_mean_dist=200, angle_step=1, norm_p2p_dist_flag=True, sum_of_rotated_images_flag=False, speed_up_flag=False, angle_limit=10, sigma=9)

maximum_curvature_scale_rotation_s5 = MaximumCurvatureScaleRotation(sigma = 5, norm_p2p_dist_flag = False, selected_mean_dist = 100, sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1)
