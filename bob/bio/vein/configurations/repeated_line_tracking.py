#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 27 Sep 2016 16:48:08 CEST

'''Repeated-Line Tracking and Miura Matching baseline

References:

1. [MNM04]_
2. [TV13]_
3. [TVM14]_

'''

sub_directory = 'rlt'
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

from ..extractor import RepeatedLineTracking

# Maximum number of iterations
NUMBER_ITERATIONS = 3000

# Distance between tracking point and cross section of profile
DISTANCE_R = 1

# Width of profile
PROFILE_WIDTH = 21

extractor = RepeatedLineTracking(
    iterations=NUMBER_ITERATIONS,
    r=DISTANCE_R,
    profile_w=PROFILE_WIDTH,
    seed=0, #Sets numpy.random.seed() to this value
    )
"""Features are the output of repeated-line tracking, as described on [MNM04]_.

Defaults taken from [TV13]_.
"""

from ..algorithm import MiuraMatch
algorithm = MiuraMatch(ch=65, cw=55)
"""Miura-matching algorithm with specific settings for search displacement

Defaults taken from [TV13]_.
"""
