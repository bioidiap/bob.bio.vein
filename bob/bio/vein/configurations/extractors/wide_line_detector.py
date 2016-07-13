#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import WideLineDetector

# Radius of the circular neighbourhood region
RADIUS_NEIGHBOURHOOD_REGION = 5
NEIGHBOURHOOD_THRESHOLD = 1

#Sum of neigbourhood threshold
SUM_NEIGHBOURHOOD = 41
RESCALE = True


feature_extractor = WideLineDetector(
    radius=RADIUS_NEIGHBOURHOOD_REGION,
    threshold=NEIGHBOURHOOD_THRESHOLD,
    g=SUM_NEIGHBOURHOOD,
    rescale=RESCALE
    )
