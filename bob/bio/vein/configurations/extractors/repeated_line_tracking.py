#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import RepeatedLineTracking

# Maximum number of iterations
NUMBER_ITERATIONS = 3000

# Distance between tracking point and cross section of profile
DISTANCE_R = 1

# Width of profile
PROFILE_WIDTH = 21


feature_extractor = RepeatedLineTracking(
    iterations=NUMBER_ITERATIONS,
    r=DISTANCE_R,
    profile_w=PROFILE_WIDTH
    )
