#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 05 Dec 2017 10:19:21 CET

'''RoI Preprocessing based on Kono's masking, from [KUU02]_.
'''

from ..preprocessor import NoCrop, KonoMask, NoNormalization, NoFilter, \
    Preprocessor

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=KonoMask(),
    normalize=NoNormalization(),
    filter=NoFilter(),
    )
"""Preprocessing using gray-level based finger cropping and no post-processing
"""
