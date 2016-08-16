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
import bob.io.image


def F(parts):
  """Returns the test file path"""

  return pkg_resources.resource_filename(__name__, os.path.join(*parts))

#
#def test_finger_crop():
#
#  #Test finger vein image preprocessors
#
#  input_filename = F(('preprocessors', '0019_3_1_120509-160517.png'))
#  output_img_filename  = F(('preprocessors',
#    '0019_3_1_120509-160517_img_lee_huang.mat'))
#  output_fvr_filename  = F(('preprocessors',
#    '0019_3_1_120509-160517_fvr_lee_huang.mat'))
#
#  img = bob.io.base.load(input_filename)
#
#  from bob.bio.vein.preprocessors.FingerCrop import FingerCrop
#  FC = FingerCrop(4, 40, False, False)
#  #FC = FingerCrop(4, 40, False, 5, 0.2, False)
#
#  output_img, finger_mask_norm, finger_mask2, spoofingValue = FC(img)
#
#  # Load Matlab reference
#  output_img_ref = bob.io.base.load(output_img_filename)
#  output_fvr_ref = bob.io.base.load(output_fvr_filename)
#
#  # Compare output of python's implementation to matlab reference
#  # (loose comparison!)
#  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 1e2
#
#
#def test_miuramax():
#
#  #Maximum Curvature method against Matlab reference
#
#  input_img_filename  = F(('extractors', 'miuramax_input_img.mat'))
#  input_fvr_filename  = F(('extractors', 'miuramax_input_fvr.mat'))
#  output_filename     = F(('extractors', 'miuramax_output.mat'))
#
#  # Load inputs
#  input_img = bob.io.base.load(input_img_filename)
#  input_fvr = bob.io.base.load(input_fvr_filename)
#
#  # Apply Python implementation
#  from bob.bio.vein.extractors.MaximumCurvature import MaximumCurvature
#  MC = MaximumCurvature(5, False)
#  output_img = MC((input_img, input_fvr))
#
#  # Load Matlab reference
#  output_img_ref = bob.io.base.load(output_filename)
#
#  # Compare output of python's implementation to matlab reference
#  # (loose comparison!)
#  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 8e-3
#
#
#def test_miurarlt():
#
#  #Repeated Line Tracking method against Matlab reference
#
#  input_img_filename  = F(('extractors', 'miurarlt_input_img.mat'))
#  input_fvr_filename  = F(('extractors', 'miurarlt_input_fvr.mat'))
#  output_filename     = F(('extractors', 'miurarlt_output.mat'))
#
#  # Load inputs
#  input_img = bob.io.base.load(input_img_filename)
#  input_fvr = bob.io.base.load(input_fvr_filename)
#
#  # Apply Python implementation
#  from bob.bio.vein.extractors.RepeatedLineTracking import RepeatedLineTracking
#  RLT = RepeatedLineTracking(3000, 1, 21, False)
#  output_img = RLT((input_img, input_fvr))
#
#  # Load Matlab reference
#  output_img_ref = bob.io.base.load(output_filename)
#
#  # Compare output of python's implementation to matlab reference
#  # (loose comparison!)
#  assert numpy.mean(numpy.abs(output_img - output_img_ref)) < 0.5
#
#
#def test_huangwl():
#
#  #Wide Line Detector method against Matlab reference
#
#  input_img_filename  = F(('extractors', 'huangwl_input_img.mat'))
#  input_fvr_filename  = F(('extractors', 'huangwl_input_fvr.mat'))
#  output_filename     = F(('extractors', 'huangwl_output.mat'))
#
#  # Load inputs
#  input_img = bob.io.base.load(input_img_filename)
#  input_fvr = bob.io.base.load(input_fvr_filename)
#
#  # Apply Python implementation
#  from bob.bio.vein.extractors.WideLineDetector import WideLineDetector
#  WL = WideLineDetector(5, 1, 41, False)
#  output_img = WL((input_img, input_fvr))
#
#  # Load Matlab reference
#  output_img_ref = bob.io.base.load(output_filename)
#
#  # Compare output of python's implementation to matlab reference
#  assert numpy.allclose(output_img, output_img_ref)
#
#
#def test_miura_match():
#
#  #Match Ratio method against Matlab reference
#
#  template_filename = F(('algorithms', '0001_2_1_120509-135338.mat'))
#  probe_gen_filename = F(('algorithms', '0001_2_2_120509-135558.mat'))
#  probe_imp_filename = F(('algorithms', '0003_2_1_120509-141255.mat'))
#
#  template_vein = bob.io.base.load(template_filename)
#  probe_gen_vein = bob.io.base.load(probe_gen_filename)
#  probe_imp_vein = bob.io.base.load(probe_imp_filename)
#
#  from bob.bio.vein.algorithms.MiuraMatch import MiuraMatch
#  MM = MiuraMatch(ch=18, cw=28)
#  score_gen = MM.score(template_vein, probe_gen_vein)
#
#  assert numpy.isclose(score_gen, 0.382689335394127)
#
#  score_imp = MM.score(template_vein, probe_imp_vein)
#  assert numpy.isclose(score_imp, 0.172906739278421)
#
#  if False: #testing gpu enabled calculations
#    MM = MiuraMatch(ch=18, cw=28, gpu=True)
#    score_gen = MM.score(template_vein, probe_gen_vein)
#    assert numpy.isclose(score_gen, 0.382689335394127)
#
#    score_imp = MM.score(template_vein, probe_imp_vein)
#    assert numpy.isclose(score_imp, 0.172906739278421)


def test_manualRoiCut():
    from bob.bio.vein.preprocessors.utils.utils import ManualRoiCut
    image_path      = F(('preprocessors', '0019_3_1_120509-160517.png'))
    annotation_path  = F(('preprocessors', '0019_3_1_120509-160517.txt'))
    #-------------------
    #image_path = "/remote/idiap.svm/home.active/teglitis/Desktop/bob.bio.vein/bob/bio/vein/tests/preprocessors/0019_3_1_120509-160517.png"
    #annotation_path = "/remote/idiap.svm/home.active/teglitis/Desktop/bob.bio.vein/bob/bio/vein/tests/preprocessors/0019_3_1_120509-160517.txt"
    #-------------------
    c = ManualRoiCut(annotation_path, image_path)
    mask_1 = c.roi_mask()
    image_1 = c.roi_image()
    # create mask using size:
    c = ManualRoiCut(annotation_path, sizes=(672,380))
    mask_2 = c.roi_mask()
    
    # loading image:
    image = bob.io.base.load(image_path)
    c = ManualRoiCut(annotation_path, image)
    mask_3 = c.roi_mask()
    image_3 = c.roi_image()
    # load text file:
    with open(annotation_path,'r') as f:
        retval = numpy.loadtxt(f, ndmin=2)
        
    # carefully -- this is BOB format --- (x,y)
    annotation = list([tuple([k[0], k[1]]) for k in retval])
    c = ManualRoiCut(annotation, image)
    mask_4 = c.roi_mask()
    image_4 = c.roi_image()
    
    assert (mask_1 == mask_2).all()
    assert (mask_1 == mask_3).all()
    assert (mask_1 == mask_4).all()
    assert (image_1 == image_3).all()
    assert (image_1 == image_4).all()
