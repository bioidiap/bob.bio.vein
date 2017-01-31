#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import ThresholdExtractor


threshold_extractor_thin_veins = ThresholdExtractor(name='Adaptive_ski_25_3_50', median=True, size=5, thin_veins_flag = True)
