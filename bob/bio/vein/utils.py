#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy
import scipy.signal
import bob.ip.base
import bob.sp
import bob.core


def imfilter(a, b, gpu=False, conv=True):
  """imfilter function based on MATLAB implementation."""

  if (a.dtype == numpy.uint8):
      a= bob.core.convert(a,numpy.float64,(0,1))
  if len(a.shape) != 2:
    a = a.reshape( ( len( a ), 1 ) )
  M, N = a.shape
  if conv == True:
      b = bob.ip.base.rotate(b, 180)
  shape = numpy.array((0,0))
  shape[0] = a.shape[0] + b.shape[0] - 1
  shape[1] = a.shape[1] + b.shape[1] - 1
  a_ext = numpy.ndarray(shape=shape, dtype=numpy.float64)
  bob.sp.extrapolate_nearest(a, a_ext)

  if gpu == True:
    import xbob.cusp
    return xbob.cusp.conv(a_ext, b)
  else:
    return scipy.signal.convolve2d(a_ext, b, 'valid')
    #return = self.convfft(a_ext, b)
