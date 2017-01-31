#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import math
import numpy

import bob.core
import bob.io.base

from bob.bio.base.extractor import Extractor

from .. import utils


class MaximumCurvatureAndMask (Extractor):
  """
  MiuraMax feature extractor.

  Based on N. Miura, A. Nagasaka, and T. Miyatake, Extraction of Finger-Vein
  Pattern Using Maximum Curvature Points in Image Profiles. Proceedings on IAPR
  conference on machine vision applications, 9 (2005), pp. 347--350

  **Parameters:**

  sigma : :py:class:`int`
      Optional: Sigma used for determining derivatives.
  """


  def __init__(self, sigma = 5):
    Extractor.__init__(self, sigma = sigma)
    self.sigma = sigma


  def maximum_curvature(self, image, mask):
    """Computes and returns the Maximum Curvature features for the given input
    fingervein image"""

    if image.dtype != numpy.uint8:
       image = bob.core.convert(image,numpy.uint8,(0,255),(0,1))
    #No es necesario pasarlo a uint8, en matlab lo dejan en float64. Comprobar si varian los resultados en vera database y ajustar.

    finger_mask = numpy.zeros(mask.shape)
    finger_mask[mask == True] = 1

    winsize = numpy.ceil(4*self.sigma)

    x = numpy.arange(-winsize, winsize+1)
    y = numpy.arange(-winsize, winsize+1)
    X, Y = numpy.meshgrid(x, y)

    h = (1/(2*math.pi*self.sigma**2))*numpy.exp(-(X**2 + Y**2)/(2*self.sigma**2))
    hx = (-X/(self.sigma**2))*h
    hxx = ((X**2 - self.sigma**2)/(self.sigma**4))*h
    hy = hx.T
    hyy = hxx.T
    hxy = ((X*Y)/(self.sigma**4))*h

    # Do the actual filtering

    fx = utils.imfilter(image, hx)
    fxx = utils.imfilter(image, hxx)
    fy = utils.imfilter(image, hy)
    fyy = utils.imfilter(image, hyy)
    fxy = utils.imfilter(image, hxy)

    f1  = 0.5*numpy.sqrt(2)*(fx + fy)   # \  #
    f2  = 0.5*numpy.sqrt(2)*(fx - fy)   # /  #
    f11 = 0.5*fxx + fxy + 0.5*fyy       # \\ #
    f22 = 0.5*fxx - fxy + 0.5*fyy       # // #

    img_h, img_w = image.shape  #Image height and width

    # Calculate curvatures
    k = numpy.zeros((img_h, img_w, 4))
    k[:,:,0] = (fxx/((1 + fx**2)**(3/2)))*finger_mask  # hor #
    k[:,:,1] = (fyy/((1 + fy**2)**(3/2)))*finger_mask  # ver #
    k[:,:,2] = (f11/((1 + f1**2)**(3/2)))*finger_mask  # \   #
    k[:,:,3] = (f22/((1 + f2**2)**(3/2)))*finger_mask  # /   #

    # Scores
    Vt = numpy.zeros(image.shape)
    Wr = 0

    # Horizontal direction
    bla = k[:,:,0] > 0
    for y in range(0,img_h):
        for x in range(0,img_w):
            if (bla[y,x]):
                Wr = Wr + 1
            if ( Wr > 0 and (x == (img_w-1) or not bla[y,x]) ):
                if (x == (img_w-1)):
                    # Reached edge of image
                    pos_end = x
                else:
                    pos_end = x - 1

                pos_start = pos_end - Wr + 1 # Start pos of concave
                if (pos_start == pos_end):
                    I=numpy.argmax(k[y,pos_start,0])
                else:
                    I=numpy.argmax(k[y,pos_start:pos_end+1,0])

                pos_max = pos_start + I
                Scr = k[y,pos_max,0]*Wr
                Vt[y,pos_max] = Vt[y,pos_max] + Scr
                Wr = 0


    # Vertical direction
    bla = k[:,:,1] > 0
    for x in range(0,img_w):
        for y in range(0,img_h):
            if (bla[y,x]):
                Wr = Wr + 1
            if ( Wr > 0 and (y == (img_h-1) or not bla[y,x]) ):
                if (y == (img_h-1)):
                    # Reached edge of image
                    pos_end = y
                else:
                    pos_end = y - 1

                pos_start = pos_end - Wr + 1 # Start pos of concave
                if (pos_start == pos_end):
                    I=numpy.argmax(k[pos_start,x,1])
                else:
                    I=numpy.argmax(k[pos_start:pos_end+1,x,1])

                pos_max = pos_start + I
                Scr = k[pos_max,x,1]*Wr

                Vt[pos_max,x] = Vt[pos_max,x] + Scr
                Wr = 0

    # Direction: \ #
    bla = k[:,:,2] > 0
    for start in range(0,img_w+img_h-1):
        # Initial values
        if (start <= img_w-1):
            x = start
            y = 0
        else:
            x = 0
            y = start - img_w + 1
        done = False

        while (not done):
            if(bla[y,x]):
                Wr = Wr + 1

            if ( Wr > 0 and (y == img_h-1 or x == img_w-1 or not bla[y,x]) ):
                if (y == img_h-1 or x == img_w-1):
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y - 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end - Wr + 1

                if (pos_y_start == pos_y_end and pos_x_start == pos_x_end):
                    d = k[pos_y_start, pos_x_start, 2]
                elif (pos_y_start == pos_y_end):
                    d = numpy.diag(k[pos_y_start, pos_x_start:pos_x_end+1, 2])
                elif (pos_x_start == pos_x_end):
                    d = numpy.diag(k[pos_y_start:pos_y_end+1, pos_x_start, 2])
                else:
                    d = numpy.diag(k[pos_y_start:pos_y_end+1, pos_x_start:pos_x_end+1, 2])

                I = numpy.argmax(d)

                pos_x_max = pos_x_start + I
                pos_y_max = pos_y_start + I

                Scr = k[pos_y_max,pos_x_max,2]*Wr

                Vt[pos_y_max,pos_x_max] = Vt[pos_y_max,pos_x_max] + Scr
                Wr = 0

            if((x == img_w-1) or (y == img_h-1)):
                done = True
            else:
                x = x + 1
                y = y + 1

    # Direction: /
    bla = k[:,:,3] > 0
    for start in range(0,img_w+img_h-1):
        # Initial values
        if (start <= (img_w-1)):
            x = start
            y = img_h-1
        else:
            x = 0
            y = img_w+img_h-start-1
        done = False

        while (not done):
            if(bla[y,x]):
                Wr = Wr + 1
            if ( Wr > 0 and (y == 0 or x == img_w-1 or not bla[y,x]) ):
                if (y == 0 or x == img_w-1):
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y + 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end + Wr - 1

                if (pos_y_start == pos_y_end and pos_x_start == pos_x_end):
                    d = k[pos_y_end, pos_x_start, 3]
                elif (pos_y_start == pos_y_end):
                    d = numpy.diag(numpy.flipud(k[pos_y_end, pos_x_start:pos_x_end+1, 3]))
                elif (pos_x_start == pos_x_end):
                    d = numpy.diag(numpy.flipud(k[pos_y_end:pos_y_start+1, pos_x_start, 3]))
                else:
                    d = numpy.diag(numpy.flipud(k[pos_y_end:pos_y_start+1, pos_x_start:pos_x_end+1, 3]))

                I = numpy.argmax(d)
                pos_x_max = pos_x_start + I
                pos_y_max = pos_y_start - I
                Scr = k[pos_y_max,pos_x_max,3]*Wr
                Vt[pos_y_max,pos_x_max] = Vt[pos_y_max,pos_x_max] + Scr
                Wr = 0

            if((x == img_w-1) or (y == 0)):
                done = True
            else:
                x = x + 1
                y = y - 1

    ## Connection of vein centres
    Cd = numpy.zeros((img_h, img_w, 4))
    for x in range(2,img_w-3):
        for y in range(2,img_h-3):
            Cd[y,x,0] = min(numpy.amax(Vt[y,x+1:x+3]), numpy.amax(Vt[y,x-2:x]))   # Hor  #
            Cd[y,x,1] = min(numpy.amax(Vt[y+1:y+3,x]), numpy.amax(Vt[y-2:y,x]))   # Vert #
            Cd[y,x,2] = min(numpy.amax(Vt[y-2:y,x-2:x]), numpy.amax(Vt[y+1:y+3,x+1:x+3])) # \  #
            Cd[y,x,3] = min(numpy.amax(Vt[y+1:y+3,x-2:x]), numpy.amax(Vt[y-2:y,x+1:x+3])) # /  #

    #Veins
    img_veins = numpy.amax(Cd,axis=2)

    # Binarise the vein image
    md = numpy.median(img_veins[img_veins>0])
    img_veins_bin = img_veins > md

    return img_veins_bin.astype(numpy.float64)


  def __call__(self, image):
    """Reads the input image, extract the features based on Maximum Curvature of the fingervein image, and writes the resulting template"""

    finger_image = image[0] #Normalized image with or without histogram equalization
    finger_mask = image[1]

    return self.maximum_curvature(finger_image, finger_mask), finger_mask



  #==========================================================================
  def write_feature( self, data, file_name ):
    """
    Writes the given data (that has been generated using the __call__ function of this class) to file.
    This method overwrites the write_feature() method of the Extractor class.

    **Parameters:**

    ``data`` : obj
        Data returned by the __call__ method of the class.

    ``file_name`` : :py:class:`str`
        Name of the file.
    """

    f = bob.io.base.HDF5File( file_name, 'w' )
    f.set( 'image', data[ 0 ] )
    f.set( 'mask', data[ 1 ] )
    del f


  #==========================================================================
  def read_feature( self, file_name ):
    """
    Reads the preprocessed data from file.
    This method overwrites the read_feature() method of the Extractor class.

    **Parameters:**

    ``file_name`` : :py:class:`str`
        Name of the file.

    **Returns:**

    ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
        Maximum eigenvalues of Hessian matrices.

    ``mask`` : 2D :py:class:`numpy.ndarray`
        Binary mask of the ROI.
    """

    f = bob.io.base.HDF5File( file_name, 'r' )
    max_eigenvalues = f.read( 'image' )
    mask = f.read( 'mask' )
    del f

    return max_eigenvalues, mask














































