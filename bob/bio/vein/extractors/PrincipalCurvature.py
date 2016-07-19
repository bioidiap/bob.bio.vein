#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy

import bob.io.base

from bob.bio.base.features.Extractor import Extractor


class MaximumCurvature (Extractor):
  """MiuraMax feature extractor

  Based on J.H. Choi, W. Song, T. Kim, S.R. Lee and H.C. Kim, Finger vein
  extraction using gradient normalization and principal curvature. Proceedings
  on Image Processing: Machine Vision Applications II, SPIE 7251, (2009)
  """

  def __init__(
      self,
      sigma = 2, # Gaussian standard deviation applied
      threshold = 1.3, # Percentage of maximum used for hard thresholding
      gpu = False,
      ):

    # call base class constructor
    Extractor.__init__(
        self,
        sigma = sigma,
        threshold = threshold,
        gpu = gpu,
        )

    # block parameters
    self.sigma = sigma
    self.threshold = threshold
    self.gpu = gpu


  def principal_curvature(self, image, mask):
    """Computes and returns the Maximum Curvature features for the given input
    fingervein image"""

    finger_mask = numpy.zeros(mask.shape)
    finger_mask[mask == True] = 1

    sigma = numpy.sqrt(self.sigma**2/2)

    gx = ut_gauss(img,sigma,1,0)
    gy = ut_gauss(img,sigma,0,1)

    Gmag = numpy.sqrt(gx**2 + gy**2) #  Gradient magnitude

    # Apply threshold
    gamma = (self.threshold/100)*max(max(Gmag))

    indices = find(Gmag < gamma)

    gx[indices] = 0
    gy[indices] = 0

    # Normalize
#    Gmag( find(Gmag == 0) ) = 1  # Avoid dividing by zero
    Gmag[ Gmag == 0 ] = 1  # Avoid dividing by zero    
    
    gx = gx/Gmag
    gy = gy/Gmag

    hxx = ut_gauss(gx,sigma,1,0)
    hxy = ut_gauss(gx,sigma,0,1)
    hyy = ut_gauss(gy,sigma,0,1)

    lambda1 = 0.5*(hxx + hyy + numpy.sqrt(hxx**2 + hyy**2 - 2*hxx**hyy + 4*hxy**2))
    veins = lambda1*finger_mask

    # Normalise
#    veins = veins - min( veins(:) )
#    veins = veins/max( veins(:) )
    
    veins = veins - min( veins[:] )
    veins = veins/max( veins[:] )

    veins = veins*finger_mask



    # Binarise the vein image by otsu
    md = numpy.median(img_veins[img_veins>0])
    img_veins_bin = img_veins > md

    return img_veins_bin.astype(numpy.float64)


  def __call__(self, image):
    """Reads the input image, extract the features based on Principal Curvature
    of the fingervein image, and writes the resulting template"""

    finger_image = image[0]    #Normalized image with or without histogram equalization
    finger_mask = image[1]

    return self.principal_curvature(finger_image, finger_mask)


  def save_feature(self, feature, feature_file):
    f = bob.io.base.HDF5File(feature_file, 'w')
    f.set('feature', feature)


  def read_feature(self, feature_file):
    f = bob.io.base.HDF5File(feature_file, 'r')
    image = f.read('feature')
    return (image)
