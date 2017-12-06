#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 05 Dec 2017 10:19:21 CET

'''RoI Preprocessing based on Watershed-based masking'''

from ..preprocessor import NoCrop, WatershedMask, NoNormalization, \
    NoFilter, Preprocessor

from os.path import join as _join
from pkg_resources import resource_filename as _filename

try:
  _model = _filename(__name__, _join('data', database + '.hdf5'))
except NameError:
  # makes documentation compile fine
  _model = _filename(__name__, _join('data', 'verafinger.hdf5'))


preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=WatershedMask(
      model=_model,
      foreground_threshold=0.6, #the higher, the stricter - max < 1.0
      background_threshold=0.2, #the lower, the stricter - min > 0.0
      ),
    normalize=NoNormalization(),
    filter=NoFilter(),
    )
"""Preprocessing using morphology and watershedding

Settings are optimised for the image quality of the specific dataset.
"""
