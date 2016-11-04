#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Unit tests against references extracted from

Matlab code from Bram Ton available on the matlab central website:

https://www.mathworks.com/matlabcentral/fileexchange/35754-wide-line-detector

This code implements the detector described in [HDLTL10] (see the references in
the generated sphinx documentation)
"""

import os
import numpy
import numpy as np
import nose.tools

import pkg_resources

import bob.io.base
import bob.io.matlab
import bob.io.image


def F(parts):
  """Returns the test file path"""

  return pkg_resources.resource_filename(__name__, os.path.join(*parts))


def _show_image(image):
  """Shows a single image

  Parameters:

    image (numpy.ndarray): A 2D numpy.ndarray compose of 8-bit unsigned
      integers containing the original image

  """

  from PIL import Image
  img = Image.fromarray(image)
  img.show()


def _show_mask_over_image(image, mask, color='red'):
  """Plots the mask over the image of a finger, for debugging purposes

  Parameters:

    image (numpy.ndarray): A 2D numpy.ndarray compose of 8-bit unsigned
      integers containing the original image

    mask (numpy.ndarray): A 2D numpy.ndarray compose of boolean values
      containing the calculated mask

  """

  from PIL import Image
  img = Image.fromarray(image).convert(mode='RGBA')
  msk = Image.fromarray((~mask).astype('uint8')*80)
  red = Image.new('RGBA', img.size, color=color)
  img.paste(red, mask=msk)
  img.show()


def test_finger_crop():

  input_filename = F(('preprocessors', '0019_3_1_120509-160517.png'))
  output_img_filename  = F(('preprocessors',
    '0019_3_1_120509-160517_img_lee_huang.mat'))
  output_fvr_filename  = F(('preprocessors',
    '0019_3_1_120509-160517_fvr_lee_huang.mat'))

  img = bob.io.base.load(input_filename)

  from bob.bio.vein.preprocessor.FingerCrop import FingerCrop
  preprocess = FingerCrop(fingercontour='leemaskMatlab', padding_width=0)

  preproc, mask = preprocess(img)
  #_show_mask_over_image(preproc, mask)

  mask_ref = bob.io.base.load(output_fvr_filename).astype('bool')
  preproc_ref = bob.core.convert(bob.io.base.load(output_img_filename),
      numpy.uint8, (0,255), (0.0,1.0))

  assert numpy.mean(numpy.abs(mask - mask_ref)) < 1e-2

  # Very loose comparison!
  #_show_image(numpy.abs(preproc.astype('int16') - preproc_ref.astype('int16')).astype('uint8'))
  assert numpy.mean(numpy.abs(preproc - preproc_ref)) < 1.3e2


def test_max_curvature():

  #Maximum Curvature method against Matlab reference

  input_img_filename  = F(('extractors', 'miuramax_input_img.mat'))
  input_fvr_filename  = F(('extractors', 'miuramax_input_fvr.mat'))
  output_filename     = F(('extractors', 'miuramax_output.mat'))

  # Load inputs
  input_img = bob.io.base.load(input_img_filename)
  input_fvr = bob.io.base.load(input_fvr_filename)

  # Apply Python implementation
  from bob.bio.vein.extractor.MaximumCurvature import MaximumCurvature
  MC = MaximumCurvature(5)
  output_img = MC((input_img, input_fvr))

  # Load Matlab reference
  output_img_ref = bob.io.base.load(output_filename)

  # Compare output of python's implementation to matlab reference
  # (loose comparison!)
  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 8e-3


def test_repeated_line_tracking():

  #Repeated Line Tracking method against Matlab reference

  input_img_filename  = F(('extractors', 'miurarlt_input_img.mat'))
  input_fvr_filename  = F(('extractors', 'miurarlt_input_fvr.mat'))
  output_filename     = F(('extractors', 'miurarlt_output.mat'))

  # Load inputs
  input_img = bob.io.base.load(input_img_filename)
  input_fvr = bob.io.base.load(input_fvr_filename)

  # Apply Python implementation
  from bob.bio.vein.extractor.RepeatedLineTracking import RepeatedLineTracking
  RLT = RepeatedLineTracking(3000, 1, 21, False)
  output_img = RLT((input_img, input_fvr))

  # Load Matlab reference
  output_img_ref = bob.io.base.load(output_filename)

  # Compare output of python's implementation to matlab reference
  # (loose comparison!)
  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 0.5


def test_wide_line_detector():

  #Wide Line Detector method against Matlab reference

  input_img_filename  = F(('extractors', 'huangwl_input_img.mat'))
  input_fvr_filename  = F(('extractors', 'huangwl_input_fvr.mat'))
  output_filename     = F(('extractors', 'huangwl_output.mat'))

  # Load inputs
  input_img = bob.io.base.load(input_img_filename)
  input_fvr = bob.io.base.load(input_fvr_filename)

  # Apply Python implementation
  from bob.bio.vein.extractor.WideLineDetector import WideLineDetector
  WL = WideLineDetector(5, 1, 41, False)
  output_img = WL((input_img, input_fvr))

  # Load Matlab reference
  output_img_ref = bob.io.base.load(output_filename)

  # Compare output of python's implementation to matlab reference
  assert numpy.allclose(output_img, output_img_ref)


def test_miura_match():

  #Match Ratio method against Matlab reference

  template_filename = F(('algorithms', '0001_2_1_120509-135338.mat'))
  probe_gen_filename = F(('algorithms', '0001_2_2_120509-135558.mat'))
  probe_imp_filename = F(('algorithms', '0003_2_1_120509-141255.mat'))

  template_vein = bob.io.base.load(template_filename)
  probe_gen_vein = bob.io.base.load(probe_gen_filename)
  probe_imp_vein = bob.io.base.load(probe_imp_filename)

  from bob.bio.vein.algorithm.MiuraMatch import MiuraMatch
  MM = MiuraMatch(ch=18, cw=28)
  score_gen = MM.score(template_vein, probe_gen_vein)

  assert numpy.isclose(score_gen, 0.382689335394127)

  score_imp = MM.score(template_vein, probe_imp_vein)
  assert numpy.isclose(score_imp, 0.172906739278421)


def test_assert_points():

  # Tests that point assertion works as expected
  from ..preprocessor import utils

  area = (10, 5)
  inside = [(0,0), (3,2), (9, 4)]
  utils.assert_points(area, inside) #should not raise

  def _check_outside(point):
    # should raise, otherwise it is an error
    try:
      utils.assert_points(area, [point])
    except AssertionError as e:
      assert str(point) in str(e)
    else:
      raise AssertionError("Did not assert %s is outside of %s" % (point, area))

  outside = [(-1, 0), (10, 0), (0, 5), (10, 5), (15,12)]
  for k in outside: _check_outside(k)


def test_fix_points():

  # Tests that point clipping works as expected
  from ..preprocessor import utils

  area = (10, 5)
  inside = [(0,0), (3,2), (9, 4)]
  fixed = utils.fix_points(area, inside)
  assert numpy.array_equal(inside, fixed), '%r != %r' % (inside, fixed)

  fixed = utils.fix_points(area, [(-1, 0)])
  assert numpy.array_equal(fixed, [(0, 0)])

  fixed = utils.fix_points(area, [(10, 0)])
  assert numpy.array_equal(fixed, [(9, 0)])

  fixed = utils.fix_points(area, [(0, 5)])
  assert numpy.array_equal(fixed, [(0, 4)])

  fixed = utils.fix_points(area, [(10, 5)])
  assert numpy.array_equal(fixed, [(9, 4)])

  fixed = utils.fix_points(area, [(15, 12)])
  assert numpy.array_equal(fixed, [(9, 4)])


def test_poly_to_mask():

  # Tests we can generate a mask out of a polygon correctly
  from ..preprocessor import utils

  area = (10, 9) #10 rows, 9 columns
  polygon = [(2, 2), (2, 7), (7, 7), (7, 2)] #square shape, (y, x) format
  mask = utils.poly_to_mask(area, polygon)
  nose.tools.eq_(mask.dtype, numpy.bool)

  # This should be the output:
  expected = numpy.array([
      [False, False, False, False, False, False, False, False, False],
      [False, False, False, False, False, False, False, False, False],
      [False, False, True,  True,  True,  True,  True,  True,  False],
      [False, False, True,  True,  True,  True,  True,  True,  False],
      [False, False, True,  True,  True,  True,  True,  True,  False],
      [False, False, True,  True,  True,  True,  True,  True,  False],
      [False, False, True,  True,  True,  True,  True,  True,  False],
      [False, False, True,  True,  True,  True,  True,  True,  False],
      [False, False, False, False, False, False, False, False, False],
      [False, False, False, False, False, False, False, False, False],
      ])
  assert numpy.array_equal(mask, expected)

  polygon = [(3, 2), (5, 7), (8, 7), (7, 3)] #trapezoid, (y, x) format
  mask = utils.poly_to_mask(area, polygon)
  nose.tools.eq_(mask.dtype, numpy.bool)

  # This should be the output:
  expected = numpy.array([
      [False, False, False, False, False, False, False, False, False],
      [False, False, False, False, False, False, False, False, False],
      [False, False, False, False, False, False, False, False, False],
      [False, False, True,  False, False, False, False, False, False],
      [False, False, True,  True,  True,  False, False, False, False],
      [False, False, False, True,  True,  True,  True,  True,  False],
      [False, False, False, True,  True,  True,  True,  True,  False],
      [False, False, False, True,  True,  True,  True,  True,  False],
      [False, False, False, False, False, False, False, True,  False],
      [False, False, False, False, False, False, False, False, False],
      ])
  assert numpy.array_equal(mask, expected)


def test_mask_to_image():

  # Tests we can correctly convert a boolean array into an image
  # that makes sense according to the data types
  from ..preprocessor import utils

  sample = numpy.array([False, True])
  nose.tools.eq_(sample.dtype, numpy.bool)

  def _check_uint(n):
    conv = utils.mask_to_image(sample, 'uint%d' % n)
    nose.tools.eq_(conv.dtype, getattr(numpy, 'uint%d' % n))
    target = [0, (2**n)-1]
    assert numpy.array_equal(conv, target), '%r != %r' % (conv, target)

  _check_uint(8)
  _check_uint(16)
  _check_uint(32)
  _check_uint(64)

  def _check_float(n):
    conv = utils.mask_to_image(sample, 'float%d' % n)
    nose.tools.eq_(conv.dtype, getattr(numpy, 'float%d' % n))
    assert numpy.array_equal(conv, [0, 1.0]), '%r != %r' % (conv, target)

  _check_float(32)
  _check_float(64)
  _check_float(128)


  # This should be unsupported
  try:
    conv = utils.mask_to_image(sample, 'int16')
  except TypeError as e:
    assert 'int16' in str(e)
  else:
    raise AssertionError('Conversion to int16 did not trigger a TypeError')
