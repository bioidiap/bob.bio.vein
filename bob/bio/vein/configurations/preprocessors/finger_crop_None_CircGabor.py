#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...preprocessors import FingerCrop

# Contour localization mask
CONTOUR_MASK_HEIGHT = 4 # Height of the mask
CONTOUR_MASK_WIDTH = 40 # Width of the mask

PADDING_OFFSET = 5
PADDING_THRESHOLD = 0.2 #Threshold for padding black zones

FINGERCONTOUR = 'leemaskMod' # Options: 'leemaskMatlab', 'konomask'
POSTPROCESSING = 'CircGabor' # Options: None, 'HE', 'HFE', 'CircGabor'

# define the preprocessor
preprocessor = FingerCrop(
    mask_h=CONTOUR_MASK_HEIGHT,
    mask_w=CONTOUR_MASK_WIDTH,
    padding_offset=PADDING_OFFSET,
    padding_threshold=PADDING_THRESHOLD,
    fingercontour=FINGERCONTOUR,
    postprocessing=POSTPROCESSING,
    )
