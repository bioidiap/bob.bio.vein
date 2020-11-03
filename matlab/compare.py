#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

# Run this script to output a debugging comparison of the Python implementation
# against matlab references you just extracted

import numpy
import bob.io.base
from bob.bio.vein.extractor import MaximumCurvature

# Load inputs
image  = bob.io.base.load('../bob/bio/vein/tests/extractors/image.hdf5')
image  = image.T.astype('float64')/255.
region = bob.io.base.load('../bob/bio/vein/tests/extractors/mask.hdf5')
region = region.T.astype('bool')

# Loads matlab references
kappa_matlab = bob.io.base.load('mc_kappa_matlab.hdf5')
kappa_matlab = kappa_matlab.transpose(2,1,0)
V_matlab = bob.io.base.load('mc_v_matlab.hdf5')
V_matlab = V_matlab.transpose(2,1,0)
Vt_matlab = bob.io.base.load('mc_vt_matlab.hdf5')
Vt_matlab = Vt_matlab.T
Cd_matlab = bob.io.base.load('mc_cd_matlab.hdf5')
Cd_matlab = Cd_matlab.transpose(2,1,0)
G_matlab = bob.io.base.load('mc_g_matlab.hdf5')
G_matlab = G_matlab.T

# Apply Python implementation
from bob.bio.vein.extractor.MaximumCurvature import MaximumCurvature
MC = MaximumCurvature(3)

kappa = MC.detect_valleys(image, region) #OK
Vt = MC.eval_vein_probabilities(kappa) #OK
Cd = MC.connect_centres(Vt) #OK
G = numpy.amax(Cd, axis=2) #OK

# Compare outputs
for k in range(4):
  print('Comparing kappa[%d]: %s' % (k,
    numpy.abs(kappa[...,k]-kappa_matlab[...,k]).sum()))

print('Comparing Vt: %s' % numpy.abs(Vt-Vt_matlab).sum())

for k in range(4):
  print('Comparing Cd[%d]: %s' % (k,
    numpy.abs(Cd[2:-3,2:-3,k]-Cd_matlab[2:-3,2:-3,k]).sum()))

print('Comparing G: %s' % numpy.abs(G[2:-3,2:-3]-G_matlab[2:-3,2:-3]).sum())
