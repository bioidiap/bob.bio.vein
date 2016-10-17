#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np
import scipy.ndimage.filters as fi

from scipy.signal import convolve2d
from bob.bio.base.algorithm import Algorithm


class AnnotationMatch (Algorithm):
  """Annotation Matching
  
  Annotations are simply matched by blurring them using a Gausian blur filter,
  multiplying 2 blurred annotations. After that the square root is extracted from
  the resultant image and 2 scores are calculated - deviding the sum of this 
  summary image with each of the input images pixel sum.
  Finaly there are twice as much scores as the images in the ``model``, the final
  score is calculated as set by the ``score_method`` varaible -- the choices are 
  ``min, ``max`` and ``mean``.

  Parameters:

    sigma (:py:class:`int`, Optional): Gausian sigma value. By defult value is 0 - Gausian 
      filter isn't used.
    
    size (:py:class:`int`, Optional): Gausian filter kernal size. Defult value is ``27``.
    
    score_method (:py:class:`str`, Optional): method that is used when the final result is 
      calculated from all scores. Default is ``mean``, possible other choices 
      are ``min`` and ``max``.
    """

  def __init__(self,
      sigma         = 0,
      size          = 27,
      score_method  = 'mean'
      ):

    # call base class constructor
    Algorithm.__init__(
        self,
        sigma                   = sigma,
        size                    = size,
        score_method            = score_method,
        multiple_model_scoring  = None,
        multiple_probe_scoring  = None
    )

    self.sigma          = sigma
    self.size           = size
    self.score_method   = score_method


  def __guss_mask__(self,guss_size=27, sigma=6):
      """Returns a 2D Gaussian kernel array."""
      inp = np.zeros((guss_size, guss_size))
      inp[guss_size//2, guss_size//2] = 1
      return fi.gaussian_filter(inp, sigma)


  def __compare_2_images_gausian__(self,image_0, image_1, mask):
    """
    results = compare_2_images(image_1, image_2, sigma, mask)
  
    Function comperes 2 images, returns 2 values.
    """
    results = []
    blurred_image0 = convolve2d(image_0, mask, mode='same')
    blurred_image1 = convolve2d(image_1, mask, mode='same')
    result_image = blurred_image0 * blurred_image1
    result_image = np.sqrt(result_image)
    results.append(result_image.sum() / blurred_image0.sum())
    results.append(result_image.sum() / blurred_image1.sum())
    return results

  def __compare_2_images__(self,image_0, image_1):
    """
    results = compare_2_images(image_1, image_2, sigma, mask)
  
    Function comperes 2 images, returns 2 values.
    """
    results = []
    result_image = image_0 * image_1
    result_image = np.sqrt(result_image)
    results.append(result_image.sum() / image_0.sum())
    results.append(result_image.sum() / image_1.sum())
    return results

  def enroll(self, enroll_features):
    """Enrolls the model by computing an average graph for each model"""
    
    enroll_features = np.array(enroll_features, dtype = np.float)
    return enroll_features
  
  
  def score(self, model, probe):
    """Computes the score of the probe and the model
         Return score - Value between 0 and 1, larger value is better match
    """
    
    I=probe.astype(np.float)
    model = model.astype(np.float)
    if len(model.shape) == 2:
      model = np.array([model])
    scores = []
    if self.sigma == 0:
      for i in range(model.shape[0]):
        R = model[i,:]
        ses = self.__compare_2_images__(I, R)
        for s in ses:
          scores.append(s)
    else:
      mask = self.__guss_mask__(guss_size=self.size, sigma=self.sigma)
      for i in range(model.shape[0]):
        R = model[i,:]
        ses = self.__compare_2_images_gausian__(I, R, mask)
        for s in ses:
          scores.append(s)
    
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = plt.subplot(121)
#    ax.imshow(R, cmap='Greys_r', interpolation='none')
#    ax = plt.subplot(122)
#    ax.imshow(I, cmap='Greys_r', interpolation='none')
#    fig.tight_layout()
#    plt.show(fig)
    scores = np.array(scores)
    if self.score_method == 'min':
      result = scores.min()
    elif self.score_method == 'max':
      result = scores.max()
    else:
      result = scores.mean()
    return result
