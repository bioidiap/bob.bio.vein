#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 07 Nov 2016 15:20:26 CET


"""Visualizes masks applied to vein imagery

Usage: %(prog)s [-v...] [options] <file> [<file>...]
       %(prog)s --help
       %(prog)s --version


Arguments:
  <file>  The HDF5 file to load image and mask from


Options:
  -h, --help             Shows this help message and exits
  -V, --version          Prints the version and exits
  -v, --verbose          Increases the output verbosity level
  -s path, --save=path   If set, saves image into a file instead of displaying
                         it


Examples:

  Visualize the mask on a single image:

     $ %(prog)s data.hdf5

  Visualize multiple masks (like in a proof-sheet):

     $ %(prog)s *.hdf5

"""


import os
import sys

import bob.core
logger = bob.core.log.setup("bob.bio.vein")

from ..preprocessor import utils


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
  from ..preprocessor import utils
  for filename in args['<file>']:
    f = bob.io.base.HDF5File(filename)
    image = f.read('image')
    mask  = f.read('mask')
    img = utils.draw_mask_over_image(image, mask)
    if args['--save']:
      img.save(args['--save'])
    else:
      img.show()
