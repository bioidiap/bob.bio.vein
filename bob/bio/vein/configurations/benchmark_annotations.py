#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 05 Dec 2017 13:36:47 CET

'''RoI Preprocessing based on hand-made annotations'''


from ..preprocessor import NoCrop, AnnotatedRoIMask, NoNormalization, \
    NoFilter, Preprocessor

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=AnnotatedRoIMask(),
    normalize=NoNormalization(),
    filter=NoFilter(),
    )
"""Preprocessing using RoI annotations
"""
