#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import HessianAkazeFeatures

smoothing_sigma = 2
image_size = (320, 320)
hessian_sigma = 4
hessian_threshold = 0.4
hessian_saturation = 0.2

hessian_akaze_feature_extractor = HessianAkazeFeatures(smoothing_sigma = smoothing_sigma,
                                 image_size = image_size,
                                 hessian_sigma = hessian_sigma,
                                 hessian_threshold = hessian_threshold,
                                 hessian_saturation = hessian_saturation)
