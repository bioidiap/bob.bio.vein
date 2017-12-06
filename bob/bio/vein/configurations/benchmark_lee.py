#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 05 Dec 2017 10:19:21 CET

'''RoI Preprocessing based on Lee's masking, from [LLP09]_.'''

from ..preprocessor import NoCrop, LeeMask, NoNormalization, NoFilter, \
    Preprocessor

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=LeeMask(),
    normalize=NoNormalization(),
    filter=NoFilter(),
    )
"""Preprocessing using gray-level based finger cropping and no post-processing
"""

from .maximum_curvature import *
