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
