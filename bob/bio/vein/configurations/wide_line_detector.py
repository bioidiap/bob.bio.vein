#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 27 Sep 2016 16:48:16 CEST

'''Huang's Wide-Line Detector and Miura Matching baseline

References:

1. [HDLTL10]_
2. [TV13]_
3. [TVM14]_

'''

sub_directory = 'wld'
"""Sub-directory where results will be placed.

You may change this setting using the ``--sub-directory`` command-line option
or the attribute ``sub_directory`` in a configuration file loaded **after**
this resource.
"""

from ..preprocessor import NoCrop, Padder, TomesLeeMask, \
    HuangNormalization, NoFilter, Preprocessor

# Filter sizes for the vertical "high-pass" filter
FILTER_HEIGHT = 4
FILTER_WIDTH = 40

# Padding (to create a buffer during normalization)
PAD_WIDTH = 5
PAD_CONST = 51

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=TomesLeeMask(filter_height=FILTER_HEIGHT, filter_width=FILTER_WIDTH,
      padder=Padder(padding_width=PAD_WIDTH, padding_constant=PAD_CONST)),
    normalize=HuangNormalization(padding_width=PAD_WIDTH,
      padding_constant=PAD_CONST),
    filter=NoFilter(),
    )
"""Preprocessing using gray-level based finger cropping and no post-processing
"""

from ..extractor import WideLineDetector

# Radius of the circular neighbourhood region
RADIUS_NEIGHBOURHOOD_REGION = 5
NEIGHBOURHOOD_THRESHOLD = 1

#Sum of neigbourhood threshold
SUM_NEIGHBOURHOOD = 41
RESCALE = True

extractor = WideLineDetector(
    radius=RADIUS_NEIGHBOURHOOD_REGION,
    threshold=NEIGHBOURHOOD_THRESHOLD,
    g=SUM_NEIGHBOURHOOD,
    rescale=RESCALE
    )
"""Features are the output of the maximum curvature algorithm, as described on
[HDLTL10]_.

Defaults taken from [TV13]_.
"""

# Notice the values of ch and cw are different than those from the
# repeated-line tracking **and** maximum curvature baselines.
from ..algorithm import MiuraMatch
algorithm = MiuraMatch(ch=18, cw=28)
"""Miura-matching algorithm with specific settings for search displacement

Defaults taken from [TV13]_.
"""
