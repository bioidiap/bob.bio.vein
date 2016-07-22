#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import math
import numpy
from PIL import Image

import bob.io.base
import bob.io.image
import bob.ip.base
import bob.sp
import bob.core

from bob.bio.base.preprocessor import Preprocessor

from .. import utils


class FingerCrop (Preprocessor):
  """Extracts the mask and pre-processes fingervein images

  Based on the implementation: E.C. Lee, H.C. Lee and K.R. Park. Finger vein
  recognition using minutia-based alignment and local binary pattern-based
  feature extraction. International Journal of Imaging Systems and
  Technology. Vol. 19, No. 3, pp. 175-178, September 2009.

  In this implementation, the finger image is (in this order):

  1. Padded
  2. The mask is extracted
  3. The finger is normalized (made horizontal)
  3. (optionally) Post processed


  Parameters:

    mask_h (int, Optional): Height of contour mask in pixels, must be an even
      number

    mask_w (int, Optional): Width of the contour mask in pixels

    padding_width (int, Optional): How much padding (in pixels) to add around
      the borders of the input image. We normally always keep this value on its
      default (5 pixels).

    padding_constant (int, Optional): What is the value of the pixels added
      to the padding. This number should be a value between 0 and 255. (From
      Pedro Tome: for UTFVP (high-quality samples), use 0. For the VERA
      Fingervein database (low-quality samples), use 51 (that corresponds to
      0.2 in a float image with values between 0 and 1).

    fingercontour (str, Optional): Select between three finger contour
      implementations: leemaskMod, leemaskMatlab or konomask. (From Pedro Tome:
      the option ``leemaskMatlab`` was just implemented for testing purposes so
      we could compare with MAT files generated from Matlab code of other
      authors. He only used it with the UTFVP database, using ``leemaskMod``
      with that database yields slight worse results.)

    postprocessing (str, Optional): Select between ``HE`` (histogram
      equalization, as with :py:func:`bob.ip.base.histogram_equalization`),
      ``HFE`` (high-frequency emphasis filter, with hard-coded parameters - see
      implementation) or ``CircGabor`` (circular Gabor filter with band-width
      1.12 octaves and standard deviation of 5 pixels (this is hard-coded)). By
      default, no postprocessing is applied on the image.

  """


  def __init__(self, mask_h = 4, mask_w = 40,
      padding_width = 5, padding_constant = 51,
      fingercontour = 'leemaskMod', postprocessing = None, **kwargs):

    Preprocessor.__init__(self,
        mask_h = mask_h,
        mask_w = mask_w,
        padding_width = padding_width,
        padding_constant = padding_constant,
        fingercontour = fingercontour,
        postprocessing = postprocessing,
        **kwargs
        )

    self.mask_h = mask_h
    self.mask_w = mask_w

    self.fingercontour = fingercontour
    self.postprocessing = postprocessing

    self.padding_width = padding_width
    self.padding_constant = padding_constant


  def __konomask__(self, image, sigma):
    """Finger vein mask extractor

    Based on the work of M. Kono, H. Ueki and S. Umemura. Near-infrared finger
    vein patterns for personal identification, Applied Optics, Vol. 41, Issue
    35, pp. 7429-7436 (2002).

    """

    sigma = 5
    img_h,img_w = image.shape

    # Determine lower half starting point
    if numpy.mod(img_h,2) == 0:
        half_img_h = img_h/2 + 1
    else:
        half_img_h = numpy.ceil(img_h/2)

    #Construct filter kernel
    winsize = numpy.ceil(4*sigma)

    x = numpy.arange(-winsize, winsize+1)
    y = numpy.arange(-winsize, winsize+1)
    X, Y = numpy.meshgrid(x, y)

    hy = (-Y/(2*math.pi*sigma**4))*numpy.exp(-(X**2 + Y**2)/(2*sigma**2))

    # Filter the image with the directional kernel
    fy = utils.imfilter(image, hy)

    # Upper part of filtred image
    img_filt_up = fy[0:half_img_h,:]
    y_up = img_filt_up.argmax(axis=0)

    # Lower part of filtred image
    img_filt_lo = fy[half_img_h-1:,:]
    y_lo = img_filt_lo.argmin(axis=0)

    # Fill region between upper and lower edges
    finger_mask = numpy.ndarray(image.shape, numpy.bool)
    finger_mask[:,:] = False

    for i in range(0,img_w):
      finger_mask[y_up[i]:y_lo[i]+image.shape[0]-half_img_h+2,i] = True

    # Extract y-position of finger edges
    edges = numpy.zeros((2,img_w))
    edges[0,:] = y_up
    edges[1,:] = y_lo + image.shape[0] - half_img_h + 1

    return (finger_mask, edges)


  def __leemaskMod__(self, image):
    """A method to calculate the finger mask

    Based on the work of Finger vein recognition using minutia-based alignment
    and local binary pattern-based feature extraction, E.C. Lee, H.C. Lee and
    K.R. Park, International Journal of Imaging Systems and Technology, Volume
    19, Issue 3, September 2009, Pages 175--178, doi: 10.1002/ima.20193

    This code is a variant of the Matlab implementation by Bram Ton, available
    at:

    https://nl.mathworks.com/matlabcentral/fileexchange/35752-finger-region-localisation/content/lee_region.m

    In this variant from Pedro Tome, the technique of filtering the image with
    a horizontal filter is also applied on the vertical axis.


    Parameters:

      image (numpy.ndarray): raw image to use for finding the mask, as 2D array
        of unsigned 8-bit integers


    Returns:

      numpy.ndarray: A 2D boolean array with the same shape of the input image
        representing the cropping mask. ``True`` values indicate where the
        finger is.

      numpy.ndarray: A 2D array with 64-bit floats indicating the indexes where
       the mask, for each column, starts and ends on the original image. The
       same of this array is (2, number of columns on input image).

    """


    img_h,img_w = image.shape

    # Determine lower half starting point
    half_img_h = img_h/2
    half_img_w = img_w/2

    # Construct mask for filtering (up-bottom direction)
    mask = numpy.ones((self.mask_h, self.mask_w), dtype='float64')
    mask[(self.mask_h/2):,:] = -1.0

    img_filt = utils.imfilter(image, mask)

    # Upper part of filtred image
    img_filt_up = img_filt[:half_img_h,:]
    y_up = img_filt_up.argmax(axis=0)

    # Lower part of filtred image
    img_filt_lo = img_filt[half_img_h:,:]
    y_lo = img_filt_lo.argmin(axis=0)

    img_filt = utils.imfilter(image, mask.T)

    # Left part of filtered image
    img_filt_lf = img_filt[:,:half_img_w]
    y_lf = img_filt_lf.argmax(axis=1)

    # Right part of filtred image
    img_filt_rg = img_filt[:,half_img_w:]
    y_rg = img_filt_rg.argmin(axis=1)

    finger_mask = numpy.zeros(image.shape, dtype='bool')

    for i in range(0,y_up.size):
        finger_mask[y_up[i]:y_lo[i]+img_filt_lo.shape[0]+1,i] = True

    # Left region
    for i in range(0,y_lf.size):
        finger_mask[i,0:y_lf[i]+1] = False

    # Right region has always the finger ending, crop the padding with the
    # meadian
    finger_mask[:,numpy.median(y_rg)+img_filt_rg.shape[1]:] = False

    # Extract y-position of finger edges
    edges = numpy.zeros((2,img_w))
    edges[0,:] = y_up
    edges[0,0:round(numpy.mean(y_lf))+1] = edges[0,round(numpy.mean(y_lf))+1]

    edges[1,:] = numpy.round(y_lo + img_filt_lo.shape[0])
    edges[1,0:round(numpy.mean(y_lf))+1] = edges[1,round(numpy.mean(y_lf))+1]

    return finger_mask, edges


  def __leemaskMatlab__(self, image):
    """A method to calculate the finger mask

    Based on the work of Finger vein recognition using minutia-based alignment
    and local binary pattern-based feature extraction, E.C. Lee, H.C. Lee and
    K.R. Park, International Journal of Imaging Systems and Technology, Volume
    19, Issue 3, September 2009, Pages 175--178, doi: 10.1002/ima.20193

    This code is based on the Matlab implementation by Bram Ton, available at:

    https://nl.mathworks.com/matlabcentral/fileexchange/35752-finger-region-localisation/content/lee_region.m

    In this method, we calculate the mask of the finger independently for each
    column of the input image. Firstly, the image is convolved with a [1,-1]
    filter of size ``(self.mask_h, self.mask_w)``. Then, the upper and lower
    parts of the resulting filtered image are separated. The location of the
    maxima in the upper part is located. The same goes for the location of the
    minima in the lower part. The mask is then calculated, per column, by
    considering it starts in the point where the maxima is in the upper part
    and goes up to the point where the minima is detected on the lower part.


    Parameters:

      image (numpy.ndarray): raw image to use for finding the mask, as 2D array
        of unsigned 8-bit integers


    Returns:

      numpy.ndarray: A 2D boolean array with the same shape of the input image
        representing the cropping mask. ``True`` values indicate where the
        finger is.

      numpy.ndarray: A 2D array with 64-bit floats indicating the indexes where
       the mask, for each column, starts and ends on the original image. The
       same of this array is (2, number of columns on input image).

    """

    img_h,img_w = image.shape

    # Determine lower half starting point
    half_img_h = img_h/2

    # Construct mask for filtering
    mask = numpy.ones((self.mask_h,self.mask_w), dtype='float64')
    mask[(self.mask_h/2):,:] = -1.0

    img_filt = utils.imfilter(image, mask)

    # Upper part of filtered image
    img_filt_up = img_filt[:half_img_h,:]
    y_up = img_filt_up.argmax(axis=0)

    # Lower part of filtered image
    img_filt_lo = img_filt[half_img_h:,:]
    y_lo = img_filt_lo.argmin(axis=0)

    # Translation: for all columns of the input image, set to True all pixels
    # of the mask from index where the maxima occurred in the upper part until
    # the index where the minima occurred in the lower part.
    finger_mask = numpy.zeros(image.shape, dtype='bool')
    for i in range(img_filt.shape[1]):
      finger_mask[y_up[i]:(y_lo[i]+img_filt_lo.shape[0]+1), i] = True

    # Extract y-position of finger edges
    edges = numpy.zeros((2,img_w), dtype='float64')
    edges[0,:] = y_up
    edges[1,:] = numpy.round(y_lo + img_filt_lo.shape[0])

    return finger_mask, edges


  def __huangnormalization__(self, image, mask, edges):
    """Simple finger normalization

    Based on B. Huang, Y. Dai, R. Li, D. Tang and W. Li, Finger-vein
    authentication based on wide line detector and pattern normalization,
    Proceedings on 20th International Conference on Pattern Recognition (ICPR),
    2010.

    This implementation aligns the finger to the centre of the image using an
    affine transformation. Elliptic projection which is described in the
    referenced paper is not included.

    In order to defined the affine transformation to be performed, the
    algorithm first calculates the center for each edge (column wise) and
    calculates the best linear fit parameters for a straight line passing
    through those points.


    Parameters:

      image (numpy.ndarray): raw image to normalize as 2D array of unsigned
        8-bit integers

      mask (numpy.ndarray): mask to normalize as 2D array of booleans

      edges (numpy.ndarray): edges of the mask, 2D array with 2 rows and as
        many columns as the input image, containing the start of the mask and
        the end of the mask.


    Returns:

      numpy.ndarray: A 2D boolean array with the same shape and data type of
        the input image representing the newly aligned image.

      numpy.ndarray: A 2D boolean array with the same shape and data type of
        the input mask representing the newly aligned mask.


    """

    img_h, img_w = image.shape

    bl = edges.mean(axis=0) #baseline
    x = numpy.arange(0,img_w)
    A = numpy.vstack([x, numpy.ones(len(x))]).T

    # Fit a straight line through the base line points
    w = numpy.linalg.lstsq(A,bl)[0] # obtaining the parameters

    angle = -1*math.atan(w[0])  # Rotation
    tr = img_h/2 - w[1]         # Translation
    scale = 1.0                 # Scale

    #Affine transformation parameters
    sx=sy=scale
    cosine = math.cos(angle)
    sine = math.sin(angle)

    a = cosine/sx
    b = -sine/sy
    #b = sine/sx
    c = 0 #Translation in x

    d = sine/sx
    e = cosine/sy
    f = tr #Translation in y
    #d = -sine/sy
    #e = cosine/sy
    #f = 0

    g = 0
    h = 0
    #h=tr
    i = 1

    T = numpy.matrix([[a,b,c],[d,e,f],[g,h,i]])
    Tinv = numpy.linalg.inv(T)
    Tinvtuple = (Tinv[0,0],Tinv[0,1], Tinv[0,2], Tinv[1,0],Tinv[1,1],Tinv[1,2])

    img=Image.fromarray(image)
    image_norm = img.transform(img.size, Image.AFFINE, Tinvtuple, resample=Image.BICUBIC)
    image_norm = numpy.array(image_norm)

    finger_mask = numpy.zeros(mask.shape)
    finger_mask[mask] = 1

    img_mask=Image.fromarray(finger_mask)
    mask_norm = img_mask.transform(img_mask.size, Image.AFFINE, Tinvtuple, resample=Image.BICUBIC)
    mask_norm = numpy.array(mask_norm).astype('bool')

    return (image_norm, mask_norm)


  def __HE__(self, image):
    """Applies histogram equalization on the input image


    Parameters:

        image (numpy.ndarray): raw image to be filtered, as 2D array of
          unsigned 8-bit integers


    Returns:

        numpy.ndarray: normalized image as a 2D array of unsigned 8-bit
        integers

    """

    #Umbralization based on the pixels non zero
    retval = numpy.zeros(image.shape, dtype=numpy.uint8)
    bob.ip.base.histogram_equalization(image, retval)
    return retval


  def __circularGabor__(self, image, bw, sigma):
    """Applies a circular gabor filter on the input image, with parameters


    Parameters:

        image (numpy.ndarray): raw image to be filtered, as 2D array of
          unsigned 8-bit integers

        bw (float): bandwidth (1.12 octave)

        sigma (int): standard deviation (5  pixels)


    Returns:

        numpy.ndarray: normalized image as a 2D array of unsigned 8-bit
        integers

    """

    # Converts image to doubles
    image_new = bob.core.convert(image,numpy.float64,(0,1),(0,255))
    img_h, img_w = image_new.shape

    fc = (1/math.pi * math.sqrt(math.log(2)/2) * (2**bw+1)/(2**bw-1))/sigma

    sz = numpy.fix(8*numpy.max([sigma,sigma]))

    if numpy.mod(sz,2) == 0: sz = sz+1

    #Constructs filter kernel
    winsize = numpy.fix(sz/2)

    x = numpy.arange(-winsize, winsize+1)
    y = numpy.arange(winsize, numpy.fix(-sz/2)-1, -1)
    X, Y = numpy.meshgrid(x, y)
    # X (right +)
    # Y (up +)

    gaborfilter = numpy.exp(-0.5*(X**2/sigma**2+Y**2/sigma**2))*numpy.cos(2*math.pi*fc*numpy.sqrt(X**2+Y**2))*(1/(2*math.pi*sigma))

    # Without normalisation
    #gaborfilter = numpy.exp(-0.5*(X**2/sigma**2+Y**2/sigma**2))*numpy.cos(2*math.pi*fc*numpy.sqrt(X**2+Y**2))

    imageEnhance = utils.imfilter(image, gaborfilter)
    imageEnhance = numpy.abs(imageEnhance)

    return bob.core.convert(imageEnhance,numpy.uint8, (0,255),
        (imageEnhance.min(),imageEnhance.max()))


  def __HFE__(self,image):
    """ High Frequency Emphasis Filtering (HFE)

    """

    ### Hard-coded parameters for the HFE filtering
    D0 = 0.025
    a = 0.6
    b = 1.2
    n = 2.0

    # converts image to doubles
    image_new = bob.core.convert(image,numpy.float64, (0,1), (0,255))
    img_h, img_w = image_new.shape

    # DFT
    Ffreq = bob.sp.fftshift(bob.sp.fft(image_new.astype(numpy.complex128))/math.sqrt(img_h*img_w))

    row = numpy.arange(1,img_w+1)
    x = (numpy.tile(row,(img_h,1)) - (numpy.fix(img_w/2)+1)) /img_w
    col = numpy.arange(1,img_h+1)
    y =  (numpy.tile(col,(img_w,1)).T - (numpy.fix(img_h/2)+1))/img_h

    # D  is  the  distance  from  point  (u,v)  to  the  centre  of the
    # frequency rectangle.
    radius = numpy.sqrt(x**2 + y**2)

    f = a + b / (1.0 + (D0 / radius)**(2*n))
    Ffreq = Ffreq * f

    # implements the inverse DFT
    imageEnhance = bob.sp.ifft(bob.sp.ifftshift(Ffreq))

    # skips complex part
    imageEnhance = numpy.abs(imageEnhance)

    # renormalizes and returns
    return bob.core.convert(imageEnhance, numpy.uint8, (0, 255),
        (imageEnhance.min(), imageEnhance.max()))


  def __call__(self, image, annotations=None):
    """Reads the input image, extract the mask of the fingervein, postprocesses
    """

    # 1. Pads the input image if any padding should be added
    image = numpy.pad(image, self.padding_width, 'constant',
        constant_values = self.padding_constant)

    ## Finger edges and contour extraction:
    if self.fingercontour == 'leemaskMatlab':
      finger_mask, finger_edges = self.__leemaskMatlab__(image) #for UTFVP
    elif self.fingercontour == 'leemaskMod':
      finger_mask, finger_edges = self.__leemaskMod__(image) #for VERA
    elif self.fingercontour == 'konomask':
      finger_mask, finger_edges = self.__konomask__(image, sigma=5)

    ## Finger region normalization:
    image_norm, finger_mask_norm = self.__huangnormalization__(image,
        finger_mask, finger_edges)

    ## veins enhancement:
    if self.postprocessing == 'HE':
      image_norm = self.__HE__(image_norm)
    elif self.postprocessing == 'HFE':
      image_norm = self.__HFE__(image_norm)
    elif self.postprocessing == 'CircGabor':
      image_norm = self.__circularGabor__(image_norm, 1.12, 5)

    ## returns the normalized image and the finger mask
    return image_norm, finger_mask_norm


  def write_data(self, data, filename):
    '''Overrides the default method implementation to handle our tuple'''

    f = bob.io.base.HDF5File(filename, 'w')
    f.set('image', data[0])
    f.set('finger_mask', data[1])


  def read_data(self, filename):
    '''Overrides the default method implementation to handle our tuple'''

    f = bob.io.base.HDF5File(filename, 'r')
    image = f.read('image')
    finger_mask = f.read('finger_mask')
    return (image, finger_mask)
