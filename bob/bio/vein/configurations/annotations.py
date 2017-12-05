#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 05 Dec 2017 13:36:47 CET

'''Baseline configuration for masking based on hand-made annotations'''


from ..preprocessor import NoCrop, AnnotatedRoIMask, HuangNormalization, \
    NoFilter, Preprocessor

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=AnnotatedRoIMask(),
    normalize=HuangNormalization(),
    filter=NoFilter(),
    )
"""Preprocessing using RoI annotations
"""
