# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:12:41 2016
"""

# import what is needed:
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import scipy.ndimage
from scipy.signal import convolve2d
import scipy.ndimage.filters as fi
import os
import six


class ManualRoiCut():
  """
  Class for manual roi extraction -- ``ManualRoiCut``.
  
  Parameters:
  
  annotation (File, list): The name of annotation file, with full path containing 
  ROI annotation data (``Bob`` format, ``(x, y)``) **or** the list of annotation 
  points (tuples) in ``Bob`` format -- ``(x, y)``
    
  image (File, :py:class:`numpy.ndarray`), optional: The name of the image to be annotation -  
  full path or image data as :py:class:`numpy.ndarray`. Image is an optional parameter,
  because it isn't needed to generate ROI binary mask.
  
  sizes (tuple): optional - a tuple of image size in ``Bob`` format ``(x,y)``.
  This parameter is used **if** no image is given to generate binary mask.
    
  Returns:
  
  A ``uint8`` :py:class:`numpy.ndarray` 2D array (image) containing ROI mask.
  Value ``1`` determines ROI area, value ``0`` -- outside ROI area. ``uint8``
  is chosen so that annotations could be used in the ``bob.bio.vein`` platform
  (there seems to be problems when saving / loading ``bool`` objects).
    
  Examples:
  
  -  generate ROI mask::
  
      from bob.bio.vein.preprocessors.utils import ManualRoiCut
      roi = ManualRoiCut(roi_annotation_points).roi_mask()

  - replace image's outside-ROI with value ``pixel_value``::
  
      from bob.bio.vein.preprocessors.utils import ManualRoiCut
      image_cutted = ManualRoiCut(roi_annotation_points, image).roi_image(pixel_value=0)
  
  """

  def __init__(self,annotation, image = None, sizes = (480, 480)):
    if isinstance(annotation, six.string_types):
        if os.path.exists(annotation):
            with open(annotation,'r') as f:
                retval = np.loadtxt(f, ndmin=2)
            self.annotation = list([tuple([k[1], k[0]]) for k in retval])
        else:
            raise IOError("Doesn' t exist file: {}".format(annotation))
            return 1
    else :
        # Convert from Bob format(x,y) to regular (y, x)
        self.annotation = list([tuple([k[1], k[0]]) for k in annotation])
    
    #load image:
    if image is not None:
          if isinstance(image, six.string_types):
              if os.path.exists(image):
                  image = Image.open(image)
                  self.image = np.array(image)
              else:
                  raise IOError("Doesn't exist file: {}".format(annotation))
                  return 1
          else:
              self.image = np.array(image)
          self.size_y = self.image.shape[0]
          self.size_x = self.image.shape[1]
    else:
        self.image = None
        self.size_y = sizes[1]
        self.size_x = sizes[0]
  def roi_mask(self):
      """Method roi_mask - generates ROI mask.
        
      Returns: A ``uint8`` :py:class:`numpy.ndarray` 2D array (image)
      containing ROI mask. Value ``1`` determines ROI area, ``0`` -- outside
      ROI area.
      """
      mask = Image.new('L', (self.size_x, self.size_y), 0)
      ImageDraw.Draw(mask).polygon(self.annotation, outline=1, fill=1)
      mask = np.array(mask, dtype = np.uint8)
      mask = 0 < mask
      return mask
  def roi_image(self, pixel_value = 0):
      """Method roi_image - replaces outside ROI pixel values with ``pixel_value``
      (default - 0).
      
      pixel_value (integer): if given, outside-ROI region is replaced with this 
      value. By default replaced with 0.
        
      Returns: A copy of image that class was initialized with, outside ROI pixel
      values are replaced with ``pixel_value``.
      """
      if self.image is not None:
          mask = self.roi_mask()
          self.image[mask == 0] = pixel_value
          return self.image
      else:
          raise IOError("No input image given, can't perform non-ROI region removal")
          return 1

            
class ConstructVeinImage():
  """
  Constructs a binary image from manual annotations. The class is made to be used with
  the ``bob.db.biowave_v1`` database.
  
  The returned 2D array (see ``return value``, below) corresponds to a person's
  vein pattern, marked by human-expert.
  
  Parameters:
  
  annotation_dictionary (:py:class:`dict`): Dictionary containing image and annotation data.
  Such :py:class:`dict` can be returned by the high level ``bob.db.biowave_v1`` 
  implementation of the ``bob.db.biowave_v1`` database. It is supposed to contain
  fields:
  - ``image``
  - ``roi_annotations``
  - ``vein_annotations``
  
  Although only the ``image.shape[0]``, ``image.shape[1]`` and variable 
  ``vein_annotations`` are used.
  
  center (:py:class:`bool`): Flag, if set to ``True``, annotations are centered.
  
  Returns:
  
  :py:class:`numpy.ndarray` : A 2D array with ``uint8`` values - value ``1``
  represents annotated vein object. The output image is constructed using
  annotation information - points.
  Each line's points are connected and 5 pixels wide line is drawn. After 
  all lines are drawn, lines are smoothed using Median filter with 
  size 5x5 pixels.
    
  Examples::
  
      from bob.bio.vein.preprocessors.utils import ConstructVeinImage
      vein_image = ConstructVeinImage(annotation_dictionary, center = self.center).return_annotations()
  """
  def __init__(self, annotation_dictionary, center = False):
    self.image            = annotation_dictionary["image"]
    #self.roi_annotations  = annotation_dictionary["roi_annotations"]
    self.vein_annotations = annotation_dictionary["vein_annotations"]
    self.center           = center
  def return_annotations(self):
    """method that returns annotations"""
    im = Image.new('L', (self.image.shape[0], self.image.shape[1]), (0)) 
    draw = ImageDraw.Draw(im)
    if self.center == True:
      xes_all = [point[1] for line in self.vein_annotations for point in line]
      yes_all = [point[0] for line in self.vein_annotations for point in line]
      for line in self.vein_annotations:
        xes = [point[1] - np.round(np.mean(xes_all)) + 239 for point in line]
        yes = [point[0] - np.round(np.mean(yes_all)) + 239 for point in line]
        for point in range(len(line) - 1):
          draw.line((xes[point],yes[point], xes[point+1], yes[point+1]), fill=1, width = 5)
    else:
      for line in self.vein_annotations:
        xes = [point[1] for point in line]
        yes = [point[0] for point in line]
        for point in range(len(line) - 1):
          draw.line((xes[point],yes[point], xes[point+1], yes[point+1]), fill=1, width = 5)
    im = im.filter(ImageFilter.MedianFilter(5))
    im = np.array(im, dtype = np.uint8)
    return im
  
    
    
    
    
    
class RotateImage():
  """
  RotateImage - automatically rotates image.
  
  So far tested only with annotations (binary images). Algorithm iteratively
  search for rotation angle such that when image is filtered with the 
  ``vein filter`` (As published in the BIOSIG 2015), the ``mean`` filtered 
  image's vector angle (for the pixels in filtered image with a magnitude at least 1/2 of the 
  maximal value of the filtered image) is ``+/- 0.5`` [deg].
  
  Parameters:
  
  image (:py:class:`numpy.ndarray`) : A 2D array containing input image. 
  Currently tested only with binary images.
  
  dark_lines (:py:class:`bool`) : A flag (default value - ``False``)
  that determines what kind of lines algorithm is going to search for.
  With default value ``False`` it will search for *whiter than
  background* lines (as is the case with annotations). If set 
  to ``True`` -- will search for *darker than background* lines 
  (as is the case with vein images).
      
  Returns:
  
    :py:class:`numpy.ndarray` : A 2D array with rotated input image
    
  Examples::
  
      from bob.bio.vein.preprocessors.utils import RotateImage
      image = RotateImage(image, dark_lines = False).rotate()
  """
  def __init__(self, image, dark_lines = False):
    self.image            = image
    self.dark_lines       = dark_lines
  def __rotate_point__(self, x,y, angle):
    """
    [xp, yp] = __rotate_point__(x,y, angle)
    """
    if type(x) is list:
      if len(x) != len(y):
        raise IOError("Length of x and y should be equal")
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
  
  def __vein_filter__(self, image, a = 3, b = 4, sigma = 4, guss_size = 15, only_lines = True, dark_lines = True):
    """
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
    
  def __get_rotatation_angle__(self,image, dark_lines = False):
    """
    angle = get_rotatation_angle(image)
    
    Returns the rotation angle in deg.
    """
    result = self.__vein_filter__(image, a = 4, b = 1, sigma = 2, guss_size = 15, only_lines = True, dark_lines = False)
    result_nonzero = result[np.abs(result) > np.abs(result).max() / 2]
    result_angle = np.angle(result_nonzero, deg=True)
    angle = result_angle.mean()
    return angle
  
  def __rotate_image__(self, image, angle):
    """
    image = rotate_image(image, angle)
    """
    image = scipy.ndimage.rotate(image, angle, reshape = False, cval=0)
    image[image > 255] = 255
    image[image < 0]   = 0
    return image
  
  def __align_image__(self, image, precision = 0.5, iterations = 25, dark_lines = False):
    """
    [image, rotation_angle, angle_error] = align_image(image, precision = 0.5, iterations = 25)
    """
    rotation_angle = 0
    angle_error = self.__get_rotatation_angle__(image, dark_lines)
    if abs(angle_error) <= precision:
      return image, rotation_angle, angle_error
    for k in range(iterations):
      rotation_angle = rotation_angle + (angle_error * 0.33)
      image = self.__rotate_image__(image, angle_error * 0.33)
      angle_error = self.__get_rotatation_angle__(image, dark_lines)
      #print(rotation_angle)
      if abs(angle_error) <= precision or k == iterations - 1:
        return image, rotation_angle, angle_error

  def rotate(self):
    """A call method that executes image rotation
    """
    [rotated_image, rotation_angle, angle_error] = self.__align_image__(image = self.image, dark_lines = self.dark_lines)
    rotated_image = np.array(rotated_image, dtype = self.image.dtype)
    return rotated_image
