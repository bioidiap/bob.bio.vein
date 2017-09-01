#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 07 Nov 2016 15:20:26 CET


"""Visualizes a particular sample throughout many processing stages

Usage: %(prog)s [-v...] [-s <path>] <database> <processed> <stem> [<stem>...]
       %(prog)s --help
       %(prog)s --version


Arguments:

  <database>  Path to the database with the image to be inspected
  <processed>  Path with the directory holding the preprocessed and extracted
               sub-directories containing the processing results of a
               bob.bio.vein toolchain
  <stem>       Name of the object on the database to display, with the root or
               the extension


Options:

  -h, --help                Shows this help message and exits
  -V, --version             Prints the version and exits
  -v, --verbose             Increases the output verbosity level
  -s <path>, --save=<path>  If set, saves image into a file instead of
                            displaying it


Examples:

  Visualize to processing toolchain over a single image

     $ %(prog)s /database /mc client/sample

  Visualize multiple masks (like in a proof-sheet):

     $ %(prog)s /database /mc client/sample1 client/sample2

"""


import os
import sys

import numpy
import bob.core
logger = bob.core.log.setup("bob.bio.vein")

import matplotlib.pyplot as mpl
from ..preprocessor import utils

import bob.io.base
import bob.io.image


def save_figures(title, image, mask, image_pp, binary):
  '''Saves individual images on a directory


  Parameters:

    title (str): A title for this plot

    image (numpy.ndarray): The original image representing the finger vein (2D
      array with dtype = ``uint8``)

    mask (numpy.ndarray): A 2D boolean array with the same size of the original
      image containing the pixels in which the image is valid (``True``) or
      invalid (``False``).

    image_pp (numpy.ndarray): A version of the original image, pre-processed by
      one of the available algorithms

    binary (numpy.ndarray): A binarized version of the original image in which
      all pixels (should) represent vein (``True``) or not-vein (``False``)

  '''

  os.makedirs(title)
  bob.io.base.save(image, os.path.join(title, 'original.png'))

  # add preprocessed image
  from ..preprocessor import utils
  img = utils.draw_mask_over_image(image_pp, mask)
  img = numpy.array(img).transpose(2,0,1)
  bob.io.base.save(img[:3], os.path.join(title, 'preprocessed.png'))

  # add binary image
  bob.io.base.save(binary.astype('uint8')*255, os.path.join(title,
    'binarized.png'))


def proof_figure(title, image, mask, image_pp, binary=None):
  '''Builds a proof canvas out of individual images


  Parameters:

    title (str): A title for this plot

    image (numpy.ndarray): The original image representing the finger vein (2D
      array with dtype = ``uint8``)

    mask (numpy.ndarray): A 2D boolean array with the same size of the original
      image containing the pixels in which the image is valid (``True``) or
      invalid (``False``).

    image_pp (numpy.ndarray): A version of the original image, pre-processed by
      one of the available algorithms

    binary (numpy.ndarray, Optional): A binarized version of the original image
      in which all pixels (should) represent vein (``True``) or not-vein
      (``False``)


  Returns:

    matplotlib.pyplot.Figure: A figure canvas containing the proof for the
    particular sample on the database

  '''

  fig = mpl.figure(figsize=(6,9), dpi=100)

  images = 3 if binary is not None else 2

  # add original image
  mpl.subplot(images, 1, 1)
  mpl.title('%s - original' % title)
  mpl.imshow(image, cmap="gray")

  # add preprocessed image
  from ..preprocessor import utils
  img = utils.draw_mask_over_image(image_pp, mask)
  mpl.subplot(images, 1, 2)
  mpl.title('Preprocessed')
  mpl.imshow(img)

  if binary is not None:
    # add binary image
    mpl.subplot(3, 1, 3)
    mpl.title('Binarized')
    mpl.imshow(binary.astype('uint8')*255, cmap="gray")

  return fig


def main(user_input=None):

  if user_input is not None:
    argv = user_input
  else:
    argv = sys.argv[1:]

  import docopt
  import pkg_resources

  completions = dict(
      prog=os.path.basename(sys.argv[0]),
      version=pkg_resources.require('bob.bio.vein')[0].version
      )

  args = docopt.docopt(
      __doc__ % completions,
      argv=argv,
      version=completions['version'],
      )

  # Sets-up logging
  verbosity = int(args['--verbose'])
  bob.core.log.set_verbosity_level(logger, verbosity)

  # Loads the image, the mask and save it to a PNG file
  for stem in args['<stem>']:
    image = bob.bio.base.load(os.path.join(args['<database>'], stem + '.png'))
    image = numpy.rot90(image, k=-1)
    pp = bob.io.base.HDF5File(os.path.join(args['<processed>'],
      'preprocessed', stem + '.hdf5'))
    mask  = pp.read('mask')
    image_pp = pp.read('image')
    binary_path = os.path.join(args['<processed>'], 'extracted', stem + '.hdf5')
    if os.path.exists(binary_path):
      binary = bob.io.base.load(binary_path)
    else:
      binary = None
    fig = proof_figure(stem, image, mask, image_pp, binary)
    if args['--save']:
      save_figures(args['--save'], image, mask, image_pp, binary)
    else:
      mpl.show()
      print('Close window to continue...')
