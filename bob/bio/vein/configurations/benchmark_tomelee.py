#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 05 Dec 2017 10:19:21 CET

'''RoI Preprocessing based on Tome's variant of Lee's masking in [LLP09]_.'''

from ..preprocessor import NoCrop, TomesLeeMask, NoNormalization, \
    NoFilter, Preprocessor

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=TomesLeeMask(),
    normalize=NoNormalization(),
    filter=NoFilter(),
    )
"""Preprocessing using gray-level based finger cropping and no post-processing
"""

from .maximum_curvature import *
