#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy
import scipy.signal
import bob.ip.base
import bob.sp
import bob.core


def imfilter(a, b):
  """Applies a 2D filtering between images

  This implementation was created to work similarly like the Matlab one.


  Parameters:

    a (numpy.ndarray): A 2-dimensional :py:class:`numpy.ndarray` which
      represents the image to be filtered. The dtype of the array is supposed
      to be 64-floats. You can also pass an 8-bit unsigned integer array,
      loaded from a file (for example). In this case it will be scaled as
      with :py:func:`bob.core.convert` and the range reset to ``[0.0, 1.0]``.

    b (numpy.ndarray): A 64-bit float 2-dimensional :py:class:`numpy.ndarray`
      which represents the filter to be applied to the image. The input filter
      has to be rotated by 180 degrees as we use
      :py:func:`scipy.signal.convolve2d` to apply it. You can rotate your
      filter ``b`` with the help of :py:func:`bob.ip.base.rotate`.

  """

  if a.dtype == numpy.uint8:
      a = bob.core.convert(a, numpy.float64, (0,1))

  shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
  a_ext = numpy.ndarray(shape=shape, dtype=numpy.float64)
  bob.sp.extrapolate_nearest(a, a_ext)

  return scipy.signal.convolve2d(a_ext, b, 'valid')
