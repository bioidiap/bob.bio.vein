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


def test_miuramax():

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


def test_miurarlt():

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


def test_huangwl():

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
  
def test_manualRoiCut():
    """
    Test ManualRoitCut
    """
    from bob.bio.vein.preprocessor.utils import ManualRoiCut
    image_path      = F(('preprocessors', '0019_3_1_120509-160517.png'))
    annotation_path  = F(('preprocessors', '0019_3_1_120509-160517.txt'))

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
    
def test_ConstructAnnotations():
  """
  Test ConstructAnnotations preprocessor
  """
  image_filename = F( ( 'preprocessors', 'ConstructAnnotations.png' ) )
  roi_annotations_filename = F( ( 'preprocessors', 'ConstructAnnotations.txt' ) )
  vein_annotations_filename = F( ( 'preprocessors', 'ConstructAnnotations.npy' ) )
  
  image = bob.io.base.load( image_filename )
  roi_annotations = np.loadtxt(roi_annotations_filename, dtype='uint16')
  roi_annotations =  [tuple([point[0], point[1]]) for point in roi_annotations]
  fp = open(vein_annotations_filename, 'rt')
  vein_annotations = np.load(fp)
  vein_annotations = vein_annotations['arr_0'].tolist()
  fp.close()
  vein_annotations = [[tuple([point[0], point[1]]) for point in line] for line in vein_annotations]
  
  annotation_dictionary = {"image" : image, "roi_annotations" : roi_annotations, "vein_annotations" : vein_annotations}
  from bob.bio.vein.preprocessor.utils import ConstructVeinImage
  from bob.bio.vein.preprocessor.utils import RotateImage
  output = ConstructVeinImage(annotation_dictionary, center = True).return_annotations()
  output = RotateImage(output, dark_lines = False).rotate()
  assert np.array_equal(output, image)
  
  
  
  
