#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


#==============================================================================
# Import what is needed here:

import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d
import scipy.ndimage.filters as fi



from bob.bio.base.extractor import Extractor


class ExtVeinsBiosig( Extractor ):
  """
  Class to compute vein filter output on an given input image.
  
  **Parameters:**
     
  a : :py:class:`int`
      Filter parameter - defines distance between sparsified filter kernel 
      pixels in the direction perpendicular to the extrated line direction 
      (withwise). Corelates with the the width of lines that one wants to 
      extract (theoretically optimal line width to be extracted is 3*a).
      Default value - 3.
  b : :py:class:`int`
      Filter parameter - defines distance between sparsified filter kernel 
      pixels in the direction paralel to the extrated line direction 
      (lengthwise):

      - When ``b`` is *bigger*, filter looses it's flexibility and can 
        extract lines only under certain angles (defined below);
      - When ``b`` is *smaller*, filter is more flexible and can extract
        lines under different angles (also bowed lines);
          
      Default value - 4.
  sigma : :py:class:`int`
      Filter parameter - Gaussian filter, that is aplied first, sigma.
      Gaisian filter makes it possible to sparsify filter kernel.
      Default value - 4.
  guss_size : :py:class:`int`
      Filter parameter - Gaussian filter size (size should be an odd integer)
      Default value - 15.
  only_lines : :py:class:`bool`
      If set to ``True`` (Default), filter extracts only lines.
      Whereas if set to ``False``, filter extracts line-line objects, as 
      published in paper **Complex 2D matched filtering without Halo 
      artifacts**
  dark_lines : :py:class:`bool`
      Default - ``True``. If set to ``True``, extracts dark lines, if set to
      ``False``, extracts light lines.
  No_of_angles : :py:class:`int`
      Default - 32. Parameter defines how many kernel rotations are used.
      A minimal rotation count is 4, maximal *logical* -- 32 / 64. This 
      The parameter linearly prolong the time necesary to calculate result.
  binarise : :py:class:`bool`
      Default - ``False``. if set to ``True``, output is binarized, using the 
      given threshold.
  threshold :py:class:`float`
      Default - 0. Defines binarization treshold (treshold is applied to the 
      vein filter output module)

  """
  def __init__(self, 
               a = 3,
               b = 4,
               sigma = 4,
               guss_size = 15,
               only_lines = True,
               dark_lines = True,
               No_of_angles = 32,
               binarise = False,
               treshold = 0
               ):
    Extractor.__init__( self, 
                       a = a,
                       b = b,
                       sigma = sigma,
                       guss_size = guss_size,
                       only_lines = only_lines,
                       dark_lines = dark_lines,
                       No_of_angles = No_of_angles,
                       binarise = binarise,
                       treshold = treshold
                       )
    self.a = a
    self.b = b
    self.sigma = sigma
    self.guss_size = guss_size
    self.only_lines = only_lines
    self.dark_lines = dark_lines
    self.No_of_angles = No_of_angles
    self.binarise = binarise
    self.treshold = treshold
  #==========================================================================
    
    
  def __rotate_point__(self, x,y, angle):
    """
    [xp, yp] = __rotate_point__(x,y, angle)
    """
    if type(x) is list:
      if len(x) != len(y):
        raise IOError("Lenght of x and y should be equal")
      xp = []
      yp = []
      for nr in range(len(x)):
        xp.append(x[nr] * np.cos(np.radians(angle)) - y[nr] * np.sin(np.radians(angle)))
        yp.append(y[nr] * np.cos(np.radians(angle)) + x[nr] * np.sin(np.radians(angle)))
    else:
      xp = x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle))
      yp = y * np.cos(np.radians(angle)) + x * np.sin(np.radians(angle))
    
    return int(np.round(xp)), int(np.round(yp))
  
  def __guss_mask__(self, guss_size=27, sigma=6):
      """Returns a 2D Gaussian kernel array."""
      inp = np.zeros((guss_size, guss_size))
      inp[guss_size//2, guss_size//2] = 1
      return fi.gaussian_filter(inp, sigma)
  
  def __ramp__(self, a):
    a = np.array(a)
    a[a < 0]=0
    return a
    
    
  def __vein_filter__(self, image, a = 3, b = 4, sigma = 4, guss_size = 15, only_lines = True, dark_lines = True, No_of_angles = 32):
      """
      Vein filter. As published in the BIOSIG 2015.
      
      The filter is fully parametrizable to be used to extract certain width 
      lines.
      
      
      **Parameters:**
      
      image :  2D :py:class:`numpy.ndarray`
          Input image, data type will be converted to the py:type:`numpy.float`
          
      a : :py:class:`int`
          Filter parameter - defines distance between sparsified filter kernel 
          pixels in the direction perpendicular to the extrated line direction 
          (withwise). Corelates with the the width of lines that one wants to 
          extract (theoretically optimal line width to be extracted is 3*a).
          Default value - 3.
      b : :py:class:`int`
          Filter parameter - defines distance between sparsified filter kernel 
          pixels in the direction paralel to the extrated line direction 
          (lengthwise):
          
              * When ``b`` is *bigger*, filter looses it's flexibility and can 
              extract lines only under certain angles (defined below);
              * When ``b`` is *smaller*, filter is more flexible and can extract
              lines under different angles (also bowed lines);
              
          Default value - 4.
      sigma : :py:class:`int`
          Filter parameter - Gaussian filter, that is aplied first, sigma.
          Gaisian filter makes it possible to sparsify filter kernel.
          Default value - 4.
      guss_size : :py:class:`int`
          Filter parameter - Gaussian filter size (size should be an odd integer)
          Default value - 15.
      only_lines : :py:class:`bool`
          If set to ``True`` (Default), filter extracts only lines.
          Whereas if set to ``False``, filter extracts line-line objects, as 
          published in paper **Complex 2D matched filtering without Halo 
          artifacts**
      dark_lines : :py:class:`bool`
          Default - ``True``. If set to ``True``, extracts dark lines, if set to
          ``False``, extracts light lines.
      No_of_angles : :py:class:`int`
          Default - 32. Parameter defines how many kernel rotations are used.
          A minimal rotation count is 4, maximal *logical* -- 32 / 64. This 
          The parameter linearly prolong the time necesary to calculate result.
          
      **Returns:**
      
      result : 2D :py:class:`numpy.ndarray`
          An output complex image - filter result representation in ``2 Phi`` 
          complex plane.
          The modolus of result represents input image corelation with the filter
          kernal. The angle of result represents the extracted objects (lines) 
          angle in the ``2 Phi`` plane (angle is represented from ``-90 deg`` 
          to ``90 deg``).
      """
      if dark_lines == True:
        Z = 1
      else:
        Z = -1
      
      angle_step = 180. / No_of_angles
      
        
      if type(image) != np.ndarray:
        image = np.array(image, dtype = np.float)
      
      padsize = 2*a+b
      gaussian_mask = self.__guss_mask__(guss_size, sigma)
      
      
      f2 = np.lib.pad(image, ((padsize, padsize), (padsize, padsize)), 'edge')
      f2 = convolve2d(f2, gaussian_mask, mode='same')
      
      result = np.zeros(image.shape)
      
      for angle in np.arange(0,179,angle_step):
        [ap, bp] = self.__rotate_point__(-b,-2*a, angle)
        mask_1 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        [ap, bp] = self.__rotate_point__(-b,-1*a, angle)
        mask_2 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        [ap, bp] = self.__rotate_point__(-b,   0, angle)
        mask_3 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        [ap, bp] = self.__rotate_point__(-b, 1*a, angle)
        mask_4 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        [ap, bp] = self.__rotate_point__(-b, 2*a, angle)
        mask_5 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
    
        
        [ap, bp] = self.__rotate_point__(+b,-2*a, angle)
        mask_6 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp] 
        
        [ap, bp] = self.__rotate_point__(+b,-1*a, angle)
        mask_7 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        [ap, bp] = self.__rotate_point__(+b,   0, angle)
        mask_8 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        [ap, bp] = self.__rotate_point__(+b, 1*a, angle)
        mask_9 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        [ap, bp] = self.__rotate_point__(+b, 2*a, angle)
        mask_10 = f2[padsize+ap:-padsize+ap,padsize+bp:-padsize+bp]
        
        
        amplitude_rez = self.__ramp__(Z*(mask_1+mask_5+mask_6+mask_10)*3 \
                         -Z*(mask_2+mask_3+mask_4+mask_7+mask_8+mask_9)*2)
                         
        if only_lines == True:
          col = np.zeros((6,image.shape[0], image.shape[1]))
          col[0] = np.minimum(self.__ramp__(-Z*mask_2+Z*mask_1),self.__ramp__(-Z*mask_2+Z*mask_5))
          col[1] = np.minimum(self.__ramp__(-Z*mask_3+Z*mask_1),self.__ramp__(-Z*mask_3+Z*mask_5))
          col[2] = np.minimum(self.__ramp__(-Z*mask_4+Z*mask_1),self.__ramp__(-Z*mask_4+Z*mask_5))
          col[3] = np.minimum(self.__ramp__(-Z*mask_7+Z*mask_6),self.__ramp__(-Z*mask_7+Z*mask_10))
          col[4] = np.minimum(self.__ramp__(-Z*mask_8+Z*mask_6),self.__ramp__(-Z*mask_8+Z*mask_10))
          col[5] = np.minimum(self.__ramp__(-Z*mask_9+Z*mask_6),self.__ramp__(-Z*mask_9+Z*mask_10))
          angle_rez = np.min(col, axis = 0)
          amplitude_rez[angle_rez==0] = 0
          
        result = result + amplitude_rez*np.exp(1j*2*(angle - 90)*np.pi/180)
        
      result = np.abs(result) * np.exp(1j*np.angle(result)/2)
      return result


    #==========================================================================
  def __call__( self, input_data ):
    """
    Compute an array of normalized LBP (or MCT or LBP cancatenated with MCT) histograms for each pair of parameters: (radius, neighbors).
    The histograms are computed taking the binary mask / ROI into account.
    """
    image    = input_data[0]
    mask     = input_data[1]

    output = self.__vein_filter__(image     = image,
                             a              = self.a,
                             b              = self.b,
                             sigma          = self.sigma,
                             guss_size      = self.guss_size,
                             only_lines     = self.only_lines,
                             dark_lines     = self.dark_lines,
                             No_of_angles   = self.No_of_angles)
    output = np.where(np.array(mask, dtype = np.uint8)>0, output, 0)
    if self.binarise == True:
      output = np.abs(output)
      output = np.where(output>self.treshold, 1, 0)
#TBD:
#   * Maybe also enable to delete roi- outside object deletion?
#   * maybe also line ``anti-erousion`` ?
    return output
