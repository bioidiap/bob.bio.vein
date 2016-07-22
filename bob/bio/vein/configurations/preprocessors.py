#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..preprocessors import FingerCrop
nonorm = FingerCrop()
he = FingerCrop(postprocessing='HE')
hfe = FingerCrop(postprocessing='HFE')
circgabor = FingerCrop(postprocessing='CircGabor')
