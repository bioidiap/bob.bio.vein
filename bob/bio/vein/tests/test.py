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

from ..preprocessor import utils as preprocessor_utils

import numpy as np

# import bob.ip.color

# for the TopographyCutRoi tests:
from bob.bio.vein.preprocessors import TopographyCutRoi

# for the KMeansRoi tests:
from bob.bio.vein.preprocessors import KMeansRoi

# for the MiuraMatchAligned tests:
from bob.bio.vein.algorithms import MiuraMatchAligned

# for the MaskedLBPHistograms class tests:
from bob.bio.vein.extractors import MaskedLBPHistograms





def F(parts):
  """Returns the test file path"""

  return pkg_resources.resource_filename(__name__, os.path.join(*parts))



#==============================================================================
def test_AlignedMatching():
    """
    Test the AlignedMatching class, which is a matching Algorithm with alignment.
    """
    
    #==========================================================================
    # 1) Test the pipeline based on Masked Spatially Enhanced Hessian Histograms:
    
    from bob.bio.vein.algorithms.aligners import HessianCrossCorrAlignment
    from bob.bio.vein.algorithms.transformers import ShiftEnrollProbeMasked
    from bob.bio.vein.algorithms.extractors import SpatEnhancHessianHistMasked
    from bob.bio.vein.algorithms.algorithms import HistogramsMatching
    from bob.bio.vein.algorithms import AlignedMatching
    
    enroll_filename = F( ( 'algorithms', 'AlignedMatching_test_data_1.hdf5' ) ) # filename with enroll data
    
    probe_filename = F( ( 'algorithms', 'AlignedMatching_test_data_2.hdf5' ) ) # filename with probe data
    
    # Initialize the aligner, which alignes the magnitudes of eigenvectors of Hessian matrices
    align_power = 1
    data_name_to_align = "eigenvectors_magnitude"
    aligner = HessianCrossCorrAlignment( align_power = align_power, data_name_to_align = data_name_to_align )
    
    # Initialize the transformer, which updates the enroll and probe
    transformer = ShiftEnrollProbeMasked()
    
    # Initialize the extractor, which extracts the Spatially Enhanced Hessian Histogram
    n_bins = 25
    eigenval_power = 1
    extractor = SpatEnhancHessianHistMasked( n_bins = n_bins, eigenval_power = eigenval_power )
    
    # Initialize the matching algorithm, which matches histograms using chi_square metrics
    algorithm = HistogramsMatching()    
    
    # Initialize the instance of the AlignedMatching class:
    matcher = AlignedMatching( aligner, transformer, extractor, algorithm )
    
    
    enroll = matcher.read_probe( enroll_filename )
    probe = matcher.read_probe( probe_filename )
        
    model = [enroll]
    
    score = matcher.score( model, probe )
    
    #==========================================================================
    # 2) Test the pipeline based on Masked Hessian Histograms:

    from bob.bio.vein.algorithms.extractors import HessianHistMasked
    
    # Initialize the extractor, which extracts the Hessian Histogram
    n_bins = 50
    eigenval_power = 1
    extractor = HessianHistMasked( n_bins = n_bins, eigenval_power = eigenval_power )
    
    matcher = AlignedMatching( aligner, transformer, extractor, algorithm )
    
    score_hessian_hist = matcher.score( model, probe )
    
    #==========================================================================
    # 3) Test the pipeline based on Spatially Enhanced LBP Histograms applied to Hessian images:
    
    from bob.bio.vein.algorithms.extractors import SpatEnhancLBPHistMasked
    
    neighbors = 8
    radius = 4
    to_average = False
    add_average_bit = False
    
    extractor = SpatEnhancLBPHistMasked( neighbors = neighbors, radius = radius, to_average = to_average, add_average_bit = add_average_bit )
    
    matcher = AlignedMatching( aligner, transformer, extractor, algorithm )
    
    score_spat_enh_lbp_hist = matcher.score( model, probe )
        
    #==========================================================================
    # 4) Test the pipeline based on Re-Alignment of Spatially Enhanced Hessian Images:
    
    from bob.bio.vein.algorithms.extractors import SpatEnhancEigenvalMasked
    from bob.bio.vein.algorithms.algorithms import SpatEnhancEigenvalMatching
    
    # Initialize the extractor, which splits the image of eigenvalues into 4 sub-regions
    extractor = SpatEnhancEigenvalMasked()
    
    # Initialize the matching algorithm, which 4 subregions computed by above extractor
    similarity_metrics_name = "error_mean"
    algorithm = SpatEnhancEigenvalMatching( similarity_metrics_name = similarity_metrics_name )
    
    # Initialize the instance of the AlignedMatching class:
    matcher = AlignedMatching( aligner, transformer, extractor, algorithm )
    
    score_spat_enh_eigenvalues = matcher.score( model, probe )
    
    #==========================================================================
    
    assert ( (score + 0.35447476151901641) < 1e-10 ) # Test the score value
    assert ( (score_hessian_hist + 0.019778549180568854) < 1e-10 ) # Test the score produced by the second pipeline
    assert ( (score_spat_enh_lbp_hist + 0.86723861364068888) < 1e-10 ) # Test the score produced by the third pipeline
    assert ( (score_spat_enh_eigenvalues + 0.47366791419007259) < 1e-10 ) # Test the score produced by the last pipeline
    
    
#==============================================================================
def test_KMeansRoi():
    """
    Test the ROI extraction algorithm namely KMeansRoi.
    """
    
    input_filename = F( ( 'preprocessors', 'TopographyCutRoi_test_image.png' ) ) # the same image is used for testing of TopographyCutRoi and KMeansRoi algorithms
    
    output_filename = F( ( 'preprocessors', 'KMeansRoi_result_image.hdf5' ) )
    
    image = bob.io.base.load( input_filename )
    
    extractor = KMeansRoi()
    
    roi = extractor.get_ROI( image )
    
    f = bob.io.base.HDF5File( output_filename )
    
    roi_loaded = f.read('data')
    
    del f
    
    assert ( ( np.sum( np.abs( roi - roi_loaded ) ) ) < 100 ) # the conditions are not strict, because the behaviour of the k-means module may vary slightly


#==============================================================================
def test_TopographyCutRoi():
    """
    Test the ROI extraction algorithm namely TopographyCutRoi.
    """
    
    input_filename = F( ( 'preprocessors', 'TopographyCutRoi_test_image.png' ) )
    
    output_filename = F( ( 'preprocessors', 'TopographyCutRoi_result_image.hdf5' ) )
    
    image = bob.io.base.load( input_filename )
    
    extractor = TopographyCutRoi()
    
    roi = extractor.get_ROI( image )
    
    f = bob.io.base.HDF5File( output_filename )
    
    roi_loaded = f.read('data')
    
    del f
    
    assert (roi == roi_loaded).all()
    
#==============================================================================
def test_MiuraMatchAligned():
    """
    Test the vein matching algorithm namely MiuraMatchAligned.
    """
    
    model_filename = F( ( 'algorithms', 'MiuraMatchAligned_test_data_1.hdf5' ) )
    
    probe_filename = F( ( 'algorithms', 'MiuraMatchAligned_test_data_2.hdf5' ) )
    
    f = bob.io.base.HDF5File( model_filename )
    
    model = f.read('data')
    
    del f
    
    f = bob.io.base.HDF5File( probe_filename )
    
    probe = f.read('data')
    
    del f
    
    miura_matcher = MiuraMatchAligned( ch = 10, cw = 10, alignment_flag = False, alignment_method = "center_of_mass" )

    score = miura_matcher.score( model, probe )
    
    assert np.abs( score - 0.040135929463629753 ) < 0.000001

#==============================================================================
def test_MaskedLBPHistograms():
    """
    Test the feature extraction algorithm namely MaskedLBPHistograms.
    """
    
    input_filename = F( ( 'extractors', 'MaskedLBPHistograms_test_image.png' ) )
    
    results_filename = F( ( 'extractors', 'MaskedLBPHistograms_test_data.hdf5' ) )
    
    image = bob.io.base.load( input_filename )
    
    mask = np.ones( image.shape )
    
    radius = [4, 5, 6]
    neighbors = [4, 8, 8]
    
    lbp_extractor_instance = MaskedLBPHistograms( neighbors, radius )
    
    computed = lbp_extractor_instance.masked_lbp_histograms( image, mask )
    
    f = bob.io.base.HDF5File( results_filename )
    
    downloaded = f.read('data')
    
    del f
    
    assert np.abs( np.sum( computed - downloaded ) ) < 0.000001

#==============================================================================

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


def test_manualRoiCut():
  from bob.bio.vein.preprocessors.utils.utils import ManualRoiCut
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
  #preprocessor_utils.show_mask_over_image(preproc, mask)

  mask_ref = bob.io.base.load(output_fvr_filename).astype('bool')
  preproc_ref = bob.core.convert(bob.io.base.load(output_img_filename),
      numpy.uint8, (0,255), (0.0,1.0))

  assert numpy.mean(numpy.abs(mask - mask_ref)) < 1e-2

 # Very loose comparison!
  #preprocessor_utils.show_image(numpy.abs(preproc.astype('int16') - preproc_ref.astype('int16')).astype('uint8'))
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


def test_max_curvature_HE():
  # Maximum Curvature method when Histogram Equalization post-processing is applied to the preprocessed vein image

  # Read in input image
  input_img_filename = F(('preprocessors', '0019_3_1_120509-160517.png'))
  input_img = bob.io.base.load(input_img_filename)
  
  # Preprocess the data and apply Histogram Equalization postprocessing (same parameters as in maximum_curvature.py configuration file + postprocessing)
  from bob.bio.vein.preprocessor.FingerCrop import FingerCrop
  FC = FingerCrop(postprocessing = 'HE')
  preproc_data = FC(input_img)

  # Extract features from preprocessed and histogram equalized data using MC extractor (same parameters as in maximum_curvature.py configuration file)
  from bob.bio.vein.extractor.MaximumCurvature import MaximumCurvature
  MC = MaximumCurvature(sigma = 5)
  extr_data = MC(preproc_data)


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


def test_repeated_line_tracking_HE():
  # Repeated Line Tracking method when Histogram Equalization post-processing is applied to the preprocessed vein image

  # Read in input image
  input_img_filename = F(('preprocessors', '0019_3_1_120509-160517.png'))
  input_img = bob.io.base.load(input_img_filename)
  
  # Preprocess the data and apply Histogram Equalization postprocessing (same parameters as in repeated_line_tracking.py configuration file + postprocessing)
  from bob.bio.vein.preprocessor.FingerCrop import FingerCrop
  FC = FingerCrop(postprocessing = 'HE')
  preproc_data = FC(input_img)

  # Extract features from preprocessed and histogram equalized data using RLT extractor (same parameters as in repeated_line_tracking.py configuration file)
  from bob.bio.vein.extractor.RepeatedLineTracking import RepeatedLineTracking
  # Maximum number of iterations
  NUMBER_ITERATIONS = 3000
  # Distance between tracking point and cross section of profile
  DISTANCE_R = 1
  # Width of profile
  PROFILE_WIDTH = 21
  RLT = RepeatedLineTracking(iterations = NUMBER_ITERATIONS, r = DISTANCE_R, profile_w = PROFILE_WIDTH, seed = 0)
  extr_data = RLT(preproc_data)


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


def test_wide_line_detector_HE():
  # Wide Line Detector method when Histogram Equalization post-processing is applied to the preprocessed vein image

  # Read in input image
  input_img_filename = F(('preprocessors', '0019_3_1_120509-160517.png'))
  input_img = bob.io.base.load(input_img_filename)
  
  # Preprocess the data and apply Histogram Equalization postprocessing (same parameters as in wide_line_detector.py configuration file + postprocessing)
  from bob.bio.vein.preprocessor.FingerCrop import FingerCrop
  FC = FingerCrop(postprocessing = 'HE')
  preproc_data = FC(input_img)

  # Extract features from preprocessed and histogram equalized data using WLD extractor (same parameters as in wide_line_detector.py configuration file)
  from bob.bio.vein.extractor.WideLineDetector import WideLineDetector
  # Radius of the circular neighbourhood region
  RADIUS_NEIGHBOURHOOD_REGION = 5
  NEIGHBOURHOOD_THRESHOLD = 1
  # Sum of neigbourhood threshold
  SUM_NEIGHBOURHOOD = 41
  RESCALE = True
  WLD = WideLineDetector(radius = RADIUS_NEIGHBOURHOOD_REGION, threshold = NEIGHBOURHOOD_THRESHOLD, g = SUM_NEIGHBOURHOOD, rescale = RESCALE)
  extr_data = WLD(preproc_data)


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
  area = (10, 5)
  inside = [(0,0), (3,2), (9, 4)]
  preprocessor_utils.assert_points(area, inside) #should not raise

  def _check_outside(point):
    # should raise, otherwise it is an error
    try:
      preprocessor_utils.assert_points(area, [point])
    except AssertionError as e:
      assert str(point) in str(e)
    else:
      raise AssertionError("Did not assert %s is outside of %s" % (point, area))

  outside = [(-1, 0), (10, 0), (0, 5), (10, 5), (15,12)]
  for k in outside: _check_outside(k)


def test_fix_points():

  # Tests that point clipping works as expected
  area = (10, 5)
  inside = [(0,0), (3,2), (9, 4)]
  fixed = preprocessor_utils.fix_points(area, inside)
  assert numpy.array_equal(inside, fixed), '%r != %r' % (inside, fixed)

  fixed = preprocessor_utils.fix_points(area, [(-1, 0)])
  assert numpy.array_equal(fixed, [(0, 0)])

  fixed = preprocessor_utils.fix_points(area, [(10, 0)])
  assert numpy.array_equal(fixed, [(9, 0)])

  fixed = preprocessor_utils.fix_points(area, [(0, 5)])
  assert numpy.array_equal(fixed, [(0, 4)])

  fixed = preprocessor_utils.fix_points(area, [(10, 5)])
  assert numpy.array_equal(fixed, [(9, 4)])

  fixed = preprocessor_utils.fix_points(area, [(15, 12)])
  assert numpy.array_equal(fixed, [(9, 4)])


def test_poly_to_mask():

  # Tests we can generate a mask out of a polygon correctly
  area = (10, 9) #10 rows, 9 columns
  polygon = [(2, 2), (2, 7), (7, 7), (7, 2)] #square shape, (y, x) format
  mask = preprocessor_utils.poly_to_mask(area, polygon)
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
  mask = preprocessor_utils.poly_to_mask(area, polygon)
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
  sample = numpy.array([False, True])
  nose.tools.eq_(sample.dtype, numpy.bool)

  def _check_uint(n):
    conv = preprocessor_utils.mask_to_image(sample, 'uint%d' % n)
    nose.tools.eq_(conv.dtype, getattr(numpy, 'uint%d' % n))
    target = [0, (2**n)-1]
    assert numpy.array_equal(conv, target), '%r != %r' % (conv, target)

  _check_uint(8)
  _check_uint(16)
  _check_uint(32)
  _check_uint(64)

  def _check_float(n):
    conv = preprocessor_utils.mask_to_image(sample, 'float%d' % n)
    nose.tools.eq_(conv.dtype, getattr(numpy, 'float%d' % n))
    assert numpy.array_equal(conv, [0, 1.0]), '%r != %r' % (conv, target)

  _check_float(32)
  _check_float(64)
  _check_float(128)


  # This should be unsupported
  try:
    conv = preprocessor_utils.mask_to_image(sample, 'int16')
  except TypeError as e:
    assert 'int16' in str(e)
  else:
    raise AssertionError('Conversion to int16 did not trigger a TypeError')


def test_jaccard_index():

  # Tests to verify the Jaccard index calculation is accurate
  a = numpy.array([
    [False, False],
    [True, True],
    ])

  b = numpy.array([
    [True, True],
    [True, False],
    ])

  nose.tools.eq_(preprocessor_utils.jaccard_index(a, b), 1.0/4.0)
  nose.tools.eq_(preprocessor_utils.jaccard_index(a, a), 1.0)
  nose.tools.eq_(preprocessor_utils.jaccard_index(b, b), 1.0)
  nose.tools.eq_(preprocessor_utils.jaccard_index(a, numpy.ones(a.shape, dtype=bool)), 2.0/4.0)
  nose.tools.eq_(preprocessor_utils.jaccard_index(a, numpy.zeros(a.shape, dtype=bool)), 0.0)
  nose.tools.eq_(preprocessor_utils.jaccard_index(b, numpy.ones(b.shape, dtype=bool)), 3.0/4.0)
  nose.tools.eq_(preprocessor_utils.jaccard_index(b, numpy.zeros(b.shape, dtype=bool)), 0.0)


def test_intersection_ratio():

  # Tests to verify the intersection ratio calculation is accurate
  a = numpy.array([
    [False, False],
    [True, True],
    ])

  b = numpy.array([
    [True, False],
    [True, False],
    ])

  nose.tools.eq_(preprocessor_utils.intersect_ratio(a, b), 1.0/2.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio(a, a), 1.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio(b, b), 1.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio(a, numpy.ones(a.shape, dtype=bool)), 1.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio(a, numpy.zeros(a.shape, dtype=bool)), 0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio(b, numpy.ones(b.shape, dtype=bool)), 1.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio(b, numpy.zeros(b.shape, dtype=bool)), 0)

  nose.tools.eq_(preprocessor_utils.intersect_ratio_of_complement(a, b), 1.0/2.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio_of_complement(a, a), 0.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio_of_complement(b, b), 0.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio_of_complement(a, numpy.ones(a.shape, dtype=bool)), 1.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio_of_complement(a, numpy.zeros(a.shape, dtype=bool)), 0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio_of_complement(b, numpy.ones(b.shape, dtype=bool)), 1.0)
  nose.tools.eq_(preprocessor_utils.intersect_ratio_of_complement(b, numpy.zeros(b.shape, dtype=bool)), 0)


def test_correlation():

  # A test for convolution performance. Correlations are used on the Miura
  # Match algorithm, therefore we want to make sure we can perform them as fast
  # as possible.
  import numpy
  import scipy.signal
  import bob.sp

  # Rough example from Vera fingervein database
  Y = 250
  X = 600
  CH = 80
  CW = 90

  def gen_ab():
    a = numpy.random.randint(256, size=(Y, X)).astype(float)
    b = numpy.random.randint(256, size=(Y-CH, X-CW)).astype(float)
    return a, b


  def bob_function(a, b):

    # rotate input image by 180 degrees
    b = numpy.rot90(b, k=2)

    # Determine padding size in x and y dimension
    size_a  = numpy.array(a.shape)
    size_b  = numpy.array(b.shape)
    outsize = size_a + size_b - 1

    # Determine 2D cross correlation in Fourier domain
    a2 = numpy.zeros(outsize)
    a2[0:size_a[0],0:size_a[1]] = a
    Fa = bob.sp.fft(a2.astype(numpy.complex128))

    b2 = numpy.zeros(outsize)
    b2[0:size_b[0],0:size_b[1]] = b
    Fb = bob.sp.fft(b2.astype(numpy.complex128))

    conv_ab = numpy.real(bob.sp.ifft(Fa*Fb))

    h, w = size_a - size_b + 1

    Nm = conv_ab[size_b[0]-1:size_b[0]-1+h, size_b[1]-1:size_b[1]-1+w]

    t0, s0 = numpy.unravel_index(Nm.argmax(), Nm.shape)

    # this is our output
    Nmm = Nm[t0,s0]

    # normalizes the output by the number of pixels lit on the input
    # matrices, taking into consideration the surface that produced the
    # result (i.e., the eroded model and part of the probe)
    h, w = b.shape
    return Nmm/(sum(sum(b)) + sum(sum(a[t0:t0+h-2*CH, s0:s0+w-2*CW])))


  def scipy_function(a, b):
    b = numpy.rot90(b, k=2)

    Nm = scipy.signal.convolve2d(a, b, 'valid')

    # figures out where the maximum is on the resulting matrix
    t0, s0 = numpy.unravel_index(Nm.argmax(), Nm.shape)

    # this is our output
    Nmm = Nm[t0,s0]

    # normalizes the output by the number of pixels lit on the input
    # matrices, taking into consideration the surface that produced the
    # result (i.e., the eroded model and part of the probe)
    h, w = b.shape
    return Nmm/(sum(sum(b)) + sum(sum(a[t0:t0+h-2*CH, s0:s0+w-2*CW])))


  def scipy2_function(a, b):
    b = numpy.rot90(b, k=2)
    Nm = scipy.signal.fftconvolve(a, b, 'valid')

    # figures out where the maximum is on the resulting matrix
    t0, s0 = numpy.unravel_index(Nm.argmax(), Nm.shape)

    # this is our output
    Nmm = Nm[t0,s0]

    # normalizes the output by the number of pixels lit on the input
    # matrices, taking into consideration the surface that produced the
    # result (i.e., the eroded model and part of the probe)
    h, w = b.shape
    return Nmm/(sum(sum(b)) + sum(sum(a[t0:t0+h-2*CH, s0:s0+w-2*CW])))


  def scipy3_function(a, b):
    Nm = scipy.signal.correlate2d(a, b, 'valid')

    # figures out where the maximum is on the resulting matrix
    t0, s0 = numpy.unravel_index(Nm.argmax(), Nm.shape)

    # this is our output
    Nmm = Nm[t0,s0]

    # normalizes the output by the number of pixels lit on the input
    # matrices, taking into consideration the surface that produced the
    # result (i.e., the eroded model and part of the probe)
    h, w = b.shape
    return Nmm/(sum(sum(b)) + sum(sum(a[t0:t0+h-2*CH, s0:s0+w-2*CW])))

  a, b = gen_ab()

  assert numpy.allclose(bob_function(a, b), scipy_function(a, b))
  assert numpy.allclose(scipy_function(a, b), scipy2_function(a, b))
  assert numpy.allclose(scipy2_function(a, b), scipy3_function(a, b))

  # if you want to test timings, uncomment the following section
  '''
  import time

  start = time.clock()
  N = 10
  for i in range(N):
    a, b = gen_ab()
    bob_function(a, b)
  total = time.clock() - start
  print('bob implementation, %d iterations - %.2e per iteration' % (N, total/N))

  start = time.clock()
  for i in range(N):
    a, b = gen_ab()
    scipy_function(a, b)
  total = time.clock() - start
  print('scipy+convolve, %d iterations - %.2e per iteration' % (N, total/N))

  start = time.clock()
  for i in range(N):
    a, b = gen_ab()
    scipy2_function(a, b)
  total = time.clock() - start
  print('scipy+fftconvolve, %d iterations - %.2e per iteration' % (N, total/N))

  start = time.clock()
  for i in range(N):
    a, b = gen_ab()
    scipy3_function(a, b)
  total = time.clock() - start
  print('scipy+correlate2d, %d iterations - %.2e per iteration' % (N, total/N))
  '''

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


def test_AnnotationMatch():
  test_pattern = np.zeros((480,480))
  test_pattern[np.random.randint(0,480,size=1000), np.random.randint(0,480,size=1000)] = 1
  test_pattern2 = np.zeros((480,480))
  test_pattern2[np.random.randint(0,480,size=1000), np.random.randint(0,480,size=1000)] = 1
  from bob.bio.vein.algorithms import AnnotationMatch
  AM = AnnotationMatch(sigma=0, score_method='mean')
  assert np.isclose(AM.score(test_pattern, test_pattern), 1)
  AM = AnnotationMatch(sigma=5, score_method='mean')
  assert np.isclose(AM.score(test_pattern, test_pattern), 1)
  AM = AnnotationMatch(sigma=0, score_method='min')
  assert np.isclose(AM.score(test_pattern, test_pattern), 1)
  AM = AnnotationMatch(sigma=5, score_method='min')
  assert np.isclose(AM.score(test_pattern, test_pattern), 1)
  AM = AnnotationMatch(sigma=0, score_method='max')
  assert np.isclose(AM.score(test_pattern, test_pattern), 1)
  AM = AnnotationMatch(sigma=5, score_method='max')
  assert np.isclose(AM.score(test_pattern, test_pattern), 1)
  
  AM = AnnotationMatch(sigma=0, score_method='mean')
  assert not np.isclose(AM.score(test_pattern, test_pattern2), 1)
  AM = AnnotationMatch(sigma=5, score_method='mean')
  assert not np.isclose(AM.score(test_pattern, test_pattern2), 1)
  AM = AnnotationMatch(sigma=0, score_method='min')
  assert not np.isclose(AM.score(test_pattern, test_pattern2), 1)
  AM = AnnotationMatch(sigma=5, score_method='min')
  assert not np.isclose(AM.score(test_pattern, test_pattern2), 1)
  AM = AnnotationMatch(sigma=0, score_method='max')
  assert not np.isclose(AM.score(test_pattern, test_pattern2), 1)
  AM = AnnotationMatch(sigma=5, score_method='max')
  assert not np.isclose(AM.score(test_pattern, test_pattern2), 1)
  
def test_PreNone():
  """
  Test empty prepocesor - PreNone
  """
  input_filename = F( ( 'preprocessors', 'TopographyCutRoi_test_image.png' ) )
  image = bob.io.base.load( input_filename )
  from bob.bio.vein.preprocessors import PreNone
  preprocesor = PreNone()
  output = preprocesor(image)
  assert (output == image).all()


def test_ExtNone():
  """
  Test empty extractor - ExtNone
  """
  input_filename = F( ( 'preprocessors', 'TopographyCutRoi_test_image.png' ) )
  image = bob.io.base.load( input_filename )
  from bob.bio.vein.extractors import ExtNone
  extractor = ExtNone()
  output = extractor(image)
  assert (output == image).all()


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
  fp = open(vein_annotations_filename, 'rb')
  vein_annotations = np.load(fp)
  vein_annotations = vein_annotations['arr_0'].tolist()
  fp.close()
  vein_annotations = [[tuple([point[0], point[1]]) for point in line] for line in vein_annotations]
  
  annotation_dictionary = {"image" : image, "roi_annotations" : roi_annotations, "vein_annotations" : vein_annotations}
  from bob.bio.vein.preprocessors import ConstructAnnotations
  preprocessor = ConstructAnnotations(center = True, rotate = True)
  output = preprocessor(annotation_dictionary)
  assert np.array_equal(output, image)

