#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d
import scipy.ndimage.filters as fi
from bob.bio.base.preprocessor import Preprocessor

class PreRotate (Preprocessor):
  """PreNone preprocessor class - an empty preprocessor that only re-saves data
  """
  def __init__(self, postprocessing = None, **kwargs):
    Preprocessor.__init__(self,postprocessing = postprocessing,**kwargs)
  
  
  def __rotate_point__(x,y, angle):
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
  
  def __guss_mask__(guss_size=27, sigma=6):
      """Returns a 2D Gaussian kernel array."""
      inp = np.zeros((guss_size, guss_size))
      inp[guss_size//2, guss_size//2] = 1
      return fi.gaussian_filter(inp, sigma)
  
  def __ramp__(a):
    a = np.array(a)
    a[a < 0]=0 
    return a
  
  def __vein_filter__(self, image, a = 3, b = 4, sigma = 4, guss_size = 15, only_lines = True, dark_lines = True):
    """
    __vein_filter__(image, a = 3, b = 4, sigma = 4, guss_size = 15, only_lines = True)
    
    Vein filter
    """
    if dark_lines == True:
      Z = 1
    else:
      Z = -1
    
    if type(image) != np.ndarray:
      image = np.array(image, dtype = np.float)
    
    padsize = 2*a+b
    gaussian_mask = self.__guss_mask__(guss_size, sigma)
    
    
    f2 = np.lib.pad(image, ((padsize, padsize), (padsize, padsize)), 'edge')
    f2 = convolve2d(f2, gaussian_mask, mode='same')
    
    result = np.zeros(image.shape)
    
    for angle in np.arange(0,179,11.25 / 2):
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
    
    
  def __get_rotatation_angle__(self,image):
    """
    angle = get_rotatation_angle(image)
    
    Returns the rotation angle in deg.
    """
    result = self.__vein_filter__(image, a = 4, b = 1, sigma = 2, guss_size = 15, only_lines = True, dark_lines = False)
    result_nonzero = result[np.abs(result) > np.abs(result).max() / 2]
    result_angle = np.angle(result_nonzero, deg=True)
    angle = result_angle.mean()
    return angle
  
  def __rotate_image__(image, angle):
    """
    image = rotate_image(image, angle)
    """
    image = scipy.ndimage.rotate(image, angle, reshape = False, cval=0)
    image[image > 1] = 1
    image[image < 0] = 0
    return image
  
  def __align_image__(self, image, precision = 0.5, iterations = 25):
    """
    [image, rotation_angle, angle_error] = align_image(image, precision = 0.5, iterations = 25)
    """
    rotation_angle = 0
    angle_error = self.__get_rotatation_angle__(image)
    if abs(angle_error) <= precision:
      return image, rotation_angle, angle_error
    for k in range(iterations):
      rotation_angle = rotation_angle + (angle_error * 0.33)
      image = self.__rotate_image__(image, angle_error * 0.33)
      angle_error = self.__get_rotatation_angle__(image)
      print(rotation_angle)
      if abs(angle_error) <= precision or k == iterations - 1:
        return image, rotation_angle, angle_error

  def __call__(self, image, annotations=None):
    """An empty __call_method that returns the inputted image
    """
    image = np.array(image, dtype = np.float)
    [rotated_image, rotation_angle, angle_error] = self.__align_image__(image)
    return rotated_image
