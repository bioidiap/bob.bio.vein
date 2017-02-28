#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import MaxEigenvalues

max_eigenvalues_s9_segment_veins = MaxEigenvalues(sigma = 9, segment_veins_flag=True)

max_eigenvalues_s9_segment_veins_amplified = MaxEigenvalues(sigma = 9, segment_veins_flag=True, amplify_segmented_veins_flag = True)

max_eigenvalues_s7_segment_veins_amplified = MaxEigenvalues(sigma = 7, segment_veins_flag=True, amplify_segmented_veins_flag = True)

max_eigenvalues_s11_segment_veins_amplified = MaxEigenvalues(sigma = 11, segment_veins_flag=True, amplify_segmented_veins_flag = True)



max_eigenvalues_s9_two_layer_segment_veins_amplified = MaxEigenvalues(sigma = 9, segment_veins_flag=True, amplify_segmented_veins_flag = True, two_layer_segmentation_flag = True)

max_eigenvalues_s5_two_layer_segment_veins_amplified = MaxEigenvalues(sigma = 5, segment_veins_flag=True, amplify_segmented_veins_flag = True, two_layer_segmentation_flag = True)

max_eigenvalues_s9_two_layer_segment_veins_amplified_binary = MaxEigenvalues(sigma = 9, segment_veins_flag=True, amplify_segmented_veins_flag = True, two_layer_segmentation_flag = True,
																			 binarize_flag = True, kernel_size = 3)

max_eigenvalues_s5_two_layer_segment_veins_amplified_binary = MaxEigenvalues(sigma = 5, segment_veins_flag=True, amplify_segmented_veins_flag = True, two_layer_segmentation_flag = True,
																			 binarize_flag = True, kernel_size = 3)

max_eigenvalues_s5_two_layer_segment_veins_amplified_binary_p2p_norm = MaxEigenvalues(sigma = 5, segment_veins_flag=True, amplify_segmented_veins_flag = True, two_layer_segmentation_flag = True,
																			 binarize_flag = True, kernel_size = 3, norm_p2p_dist_flag = True, selected_mean_dist = 100)

max_eigenvalues_s7_two_layer_segment_veins_amplified_binary = MaxEigenvalues(sigma = 7, segment_veins_flag=True, amplify_segmented_veins_flag = True, two_layer_segmentation_flag = True,
																			 binarize_flag = True, kernel_size = 3)

max_eigenvalues_s7_two_layer_segment_veins_amplified_binary_p2p_norm = MaxEigenvalues(sigma = 7, segment_veins_flag=True, amplify_segmented_veins_flag = True, two_layer_segmentation_flag = True,
																			 binarize_flag = True, kernel_size = 3, norm_p2p_dist_flag = True, selected_mean_dist = 100)