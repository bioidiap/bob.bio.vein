#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Pedro Tome <pedro.tome@idiap.ch>
#
# Copyright (C) 2014-2015 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""Test Units
"""

import unittest
import bob.io.base
import bob.io.matlab
import os
import pkg_resources

def F_pre(name, f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(name, os.path.join('preprocessing', f))

def F_feat(name, f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(name, os.path.join('features', f))

def F_mat(name, f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(name, os.path.join('matching', f))


class FingerveinTests(unittest.TestCase):

  def test_finger_crop(self):
    """Test finger vein image preprocessing"""
    import numpy
    input_filename = F_pre(__name__, '0019_3_1_120509-160517.png')
    output_img_filename  = F_pre(__name__, '0019_3_1_120509-160517_img_lee_huang.mat')
    output_fvr_filename  = F_pre(__name__, '0019_3_1_120509-160517_fvr_lee_huang.mat')
        
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
    #For debugging    
    #import ipdb; ipdb.set_trace()    
       
    self.assertTrue(numpy.mean(numpy.abs(output_img - output_img_ref)) < 1e2)


  def test_miuramax(self):
    """Test feature extraction: Maximum Curvature method against Matlab reference"""

    import numpy
    input_img_filename  = F_feat(__name__, 'miuramax_input_img.mat')
    input_fvr_filename  = F_feat(__name__, 'miuramax_input_fvr.mat')
    output_filename     = F_feat(__name__, 'miuramax_output.mat')

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
    self.assertTrue(numpy.mean(numpy.abs(output_img - output_img_ref)) < 8e-3)
    
  
  def test_miurarlt(self):
    """Test feature extraction: Repeated Line Tracking method against Matlab reference"""

    import numpy
    input_img_filename  = F_feat(__name__, 'miurarlt_input_img.mat')
    input_fvr_filename  = F_feat(__name__, 'miurarlt_input_fvr.mat')
    output_filename     = F_feat(__name__, 'miurarlt_output.mat')

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
    self.assertTrue(numpy.mean(numpy.abs(output_img - output_img_ref)) < 0.5)
    
    
  def test_huangwl(self):
    """Test feature extraction: Wide Line Detector method against Matlab reference"""

    import numpy
    input_img_filename  = F_feat(__name__, 'huangwl_input_img.mat')
    input_fvr_filename  = F_feat(__name__, 'huangwl_input_fvr.mat')
    output_filename     = F_feat(__name__, 'huangwl_output.mat')

    # Load inputs
    input_img = bob.io.base.load(input_img_filename)
    input_fvr = bob.io.base.load(input_fvr_filename)
    
    # Apply Python implementation
    from bob.fingervein.features.WideLineDetector import WideLineDetector
    WL = WideLineDetector(5, 1, 41, False)
    output_img = WL((input_img, input_fvr))

    # Load Matlab reference
    output_img_ref = bob.io.base.load(output_filename)

    #For debugging    
    #import ipdb; ipdb.set_trace()    
    #from PIL import Image
    #Image.fromarray(bob.core.convert(output_img,numpy.uint8,(0,255),(0,1))).show()
    #Image.fromarray(bob.core.convert(output_img_ref,numpy.uint8,(0,255),(0,1))).show()
    
    # Compare output of python's implementation to matlab reference
    self.assertTrue(numpy.allclose(output_img, output_img_ref))


  def test_miura_match(self):
    """Test matching: Match Ratio method against Matlab reference"""
    
    template_filename = F_mat(__name__, '0001_2_1_120509-135338.mat')
    probe_gen_filename = F_mat(__name__, '0001_2_2_120509-135558.mat')
    probe_imp_filename = F_mat(__name__, '0003_2_1_120509-141255.mat')

    template_vein = bob.io.base.load(template_filename)
    probe_gen_vein = bob.io.base.load(probe_gen_filename)
    probe_imp_vein = bob.io.base.load(probe_imp_filename)
    
    from bob.fingervein.tools.MiuraMatch import MiuraMatch
    MM = MiuraMatch(ch=18, cw=28) 
    score_gen = MM.score(template_vein, probe_gen_vein)
    self.assertAlmostEqual(score_gen, 0.382689335394127)
   
    #import ipdb; ipdb.set_trace()    

    score_imp = MM.score(template_vein, probe_imp_vein)
    self.assertAlmostEqual(score_imp, 0.172906739278421)

    if False:
      MM = MiuraMatch(ch=18, cw=28, gpu=True) 
      score_gen = MM.score(template_vein, probe_gen_vein)
      self.assertAlmostEqual(score_gen, 0.382689335394127)

      score_imp = MM.score(template_vein, probe_imp_vein)
      self.assertAlmostEqual(score_imp, 0.172906739278421)

