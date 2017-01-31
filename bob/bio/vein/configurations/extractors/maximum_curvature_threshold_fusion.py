#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import MaximumCurvatureThresholdFusion

maximum_curvature_threshold_fusion = MaximumCurvatureThresholdFusion(sigma = 5,
					                             norm_p2p_dist_flag = False, selected_mean_dist = 100,
					                             name = 'Adaptive_ski_25_3_50',
					                             median = True,
					                             size = 5)

maximum_curvature_threshold_fusion_p2pnorm = MaximumCurvatureThresholdFusion(sigma = 5,
							                     norm_p2p_dist_flag = True, selected_mean_dist = 100,
							                     name = 'Adaptive_ski_25_3_50',
							                     median = True,
							                     size = 5)

