#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Test Units
"""

import os
import numpy
import nose.tools

import pkg_resources

import bob.io.base
import bob.io.matlab


def F(parts):
  """Returns the test file path"""

  return pkg_resources.resource_filename(__name__, os.path.join(*parts))


def test_finger_crop():

  #Test finger vein image preprocessing

  input_filename = F(('preprocessing', '0019_3_1_120509-160517.png'))
  output_img_filename  = F(('preprocessing',
    '0019_3_1_120509-160517_img_lee_huang.mat'))
  output_fvr_filename  = F(('preprocessing',
    '0019_3_1_120509-160517_fvr_lee_huang.mat'))

  img = bob.io.base.load(input_filename)

  from bob.fingervein.preprocessing.FingerCrop import FingerCrop
  FC = FingerCrop(4, 40, False, False)
  #FC = FingerCrop(4, 40, False, 5, 0.2, False)

  output_img, finger_mask_norm, finger_mask2, spoofingValue = FC(img)

  # Load Matlab reference
  output_img_ref = bob.io.base.load(output_img_filename)
  output_fvr_ref = bob.io.base.load(output_fvr_filename)

  # Compare output of python's implementation to matlab reference
  # (loose comparison!)
  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 1e2


def test_miuramax():

  #Maximum Curvature method against Matlab reference

  input_img_filename  = F(('features', 'miuramax_input_img.mat'))
  input_fvr_filename  = F(('features', 'miuramax_input_fvr.mat'))
  output_filename     = F(('features', 'miuramax_output.mat'))

  # Load inputs
  input_img = bob.io.base.load(input_img_filename)
  input_fvr = bob.io.base.load(input_fvr_filename)

  # Apply Python implementation
  from bob.fingervein.features.MaximumCurvature import MaximumCurvature
  MC = MaximumCurvature(5, False)
  output_img = MC((input_img, input_fvr))

  # Load Matlab reference
  output_img_ref = bob.io.base.load(output_filename)

  # Compare output of python's implementation to matlab reference
  # (loose comparison!)
  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 8e-3


def test_miurarlt():

  #Repeated Line Tracking method against Matlab reference

  input_img_filename  = F(('features', 'miurarlt_input_img.mat'))
  input_fvr_filename  = F(('features', 'miurarlt_input_fvr.mat'))
  output_filename     = F(('features', 'miurarlt_output.mat'))

  # Load inputs
  input_img = bob.io.base.load(input_img_filename)
  input_fvr = bob.io.base.load(input_fvr_filename)

  # Apply Python implementation
  from bob.fingervein.features.RepeatedLineTracking import RepeatedLineTracking
  RLT = RepeatedLineTracking(3000, 1, 21, False)
  output_img = RLT((input_img, input_fvr))

  # Load Matlab reference
  output_img_ref = bob.io.base.load(output_filename)

  # Compare output of python's implementation to matlab reference
  # (loose comparison!)
  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 0.5


def test_huangwl():

  #Wide Line Detector method against Matlab reference

  input_img_filename  = F(('features', 'huangwl_input_img.mat'))
  input_fvr_filename  = F(('features', 'huangwl_input_fvr.mat'))
  output_filename     = F(('features', 'huangwl_output.mat'))

  # Load inputs
  input_img = bob.io.base.load(input_img_filename)
  input_fvr = bob.io.base.load(input_fvr_filename)

  # Apply Python implementation
  from bob.fingervein.features.WideLineDetector import WideLineDetector
  WL = WideLineDetector(5, 1, 41, False)
  output_img = WL((input_img, input_fvr))

  # Load Matlab reference
  output_img_ref = bob.io.base.load(output_filename)

  # Compare output of python's implementation to matlab reference
  assert numpy.allclose(output_img, output_img_ref)


def test_miura_match():
  """Test matching: Match Ratio method against Matlab reference"""

  template_filename = F(('matching', '0001_2_1_120509-135338.mat'))
  probe_gen_filename = F(('matching', '0001_2_2_120509-135558.mat'))
  probe_imp_filename = F(('matching', '0003_2_1_120509-141255.mat'))

  template_vein = bob.io.base.load(template_filename)
  probe_gen_vein = bob.io.base.load(probe_gen_filename)
  probe_imp_vein = bob.io.base.load(probe_imp_filename)

  from bob.fingervein.tools.MiuraMatch import MiuraMatch
  MM = MiuraMatch(ch=18, cw=28)
  score_gen = MM.score(template_vein, probe_gen_vein)

  assert numpy.isclose(score_gen, 0.382689335394127)

  score_imp = MM.score(template_vein, probe_imp_vein)
  assert numpy.isclose(score_imp, 0.172906739278421)

  if False: #testing gpu enabled calculations
    MM = MiuraMatch(ch=18, cw=28, gpu=True)
    score_gen = MM.score(template_vein, probe_gen_vein)
    assert numpy.isclose(score_gen, 0.382689335394127)

    score_imp = MM.score(template_vein, probe_imp_vein)
    assert numpy.isclose(score_imp, 0.172906739278421)
