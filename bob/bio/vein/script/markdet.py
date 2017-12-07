#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Trains a new MLP to perform pre-watershed marker detection

Usage: %(prog)s [-v...] [--samples=N] [--model=PATH] [--points=N] [--hidden=N]
                [--batch=N] [--iterations=N] [--hollow] [--plot]
                [--maximum-error=F] <database> <protocol> <group> <size>
       %(prog)s --help
       %(prog)s --version


Arguments:

  <database>  Name of the database to use for creating the model (options are:
              "fv3d", "verafinger", "hkpu" or "thufvdt")
  <protocol>  Name of the protocol to use for creating the model (options
              depend on the database chosen)
  <group>     Name of the group to use on the database/protocol with the
              samples to use for training the model (options are: "train",
              "dev" or "eval")
  <size>      The size (see scipy.ndimage.generic_filter) of the window to use
              for determining the input to the neural network. Valid values are
              odd numbers starting from 3 (3, 5, 7, 9, 11, ...).  Odd values
              are required so the network output is produced w.r.t. an existing
              (center) pixel. As you increase the window size, the amount of
              data passed to the neural network increases exponetially. For
              example, if this parameter is set to 3, the input size to the
              neural network is 11 (9 pixels + center pixel location). If this
              parameter is set to 5, then the input becomes 27 in size and so
              on. You can optionally specify the ``--hollow`` command-line
              option to make the footprint of the window passed to the neural
              network smaller.


Options:

  -h, --help             Shows this help message and exits
  -V, --version          Prints the version and exits
  -v, --verbose          Increases the output verbosity level. Using "-vv"
                         allows the program to output informational messages as
                         it goes along.
  -m PATH, --model=PATH  Path to the generated model file [default: model.hdf5]
  -s N, --samples=N      Maximum number of samples to use for training and
                         validation. If not set, use half of the samples for
                         training and the other half for validation. If all
                         samples are used for training, then no samples will be
                         used for validation.
  -p, --plot             Plot samples of the validation set exposed to the just
                         trained neural network. Useful to visualize where
                         errors happen more frequently.
  -P N, --points=N       Maximum number of samples to use for plotting
                         ground-truth and classification errors. The more
                         points, the less responsive the plot becomes
                         [default: 1000]
  -H N, --hidden=N       Number of neurons on the hidden layer of the
                         multi-layer perceptron [default: 5]
  -b N, --batch=N        Number of samples to use for every batch [default: 1]
  -i N, --iterations=N   Number of iterations to train the neural net for
                         [default: 2000]
  -x, --hollow           If set, then the image sub-window passed to the neural
                         network will be hollow - the pixels that are not in
                         the outside border of the window are not passed
                         (except for the center pixel value).
  -e, --maximum-error=F  Maximum relative error to allow for foreground and
                         background marker detection [default: 0.03]


Examples:

  Trains on the 3D Fingervein database, uses a filled window size of 3x3 pixels:

     $ %(prog)s -vv fv3d central dev 3

  Saves the model to a different file, use only 100 samples:

    $ %(prog)s -vv -s 100 --model=/path/to/saved-model.hdf5 fv3d central dev 3

"""


import os
import sys
import schema
import docopt
import numpy
import scipy.ndimage

logger = None


class Filter(object):
  '''Callable for filtering/converting the input image into features'''


  def __init__(self, image_shape):

    # builds indexes before hand, based on image dimensions
    idx = numpy.mgrid[:image_shape[0], :image_shape[1]]
    self.indexes = numpy.array([idx[0].flatten(), idx[1].flatten()],
        dtype='float64')
    self.indexes[0,:] /= image_shape[0]
    self.indexes[1,:] /= image_shape[1]
    self.current = 0


  def __call__(self, arr, output):

    output[self.current, :-2] = arr/255
    output[self.current, -2:] = self.indexes[:,self.current]
    self.current += 1
    return 1.0


def validate(args):
  '''Validates command-line arguments, returns parsed values

  This function uses :py:mod:`schema` for validating :py:mod:`docopt`
  arguments. Logging level is not checked by this procedure (actually, it is
  ignored) and must be previously setup as some of the elements here may use
  logging for outputing information.


  Parameters:

    args (dict): Dictionary of arguments as defined by the help message and
      returned by :py:mod:`docopt`


  Returns

    dict: Validate dictionary with the same keys as the input and with values
      possibly transformed by the validation procedure


  Raises:

    schema.SchemaError: in case one of the checked options does not validate.

  '''

  from .validate import check_model_does_not_exist, validate_protocol, \
      validate_group

  sch = schema.Schema({
    '--model': check_model_does_not_exist,
    '--samples': schema.Or(schema.Use(int), None),
    '--points': schema.Use(int),
    '--hidden': schema.Use(int),
    '--batch': schema.Use(int),
    '--iterations': schema.Use(int),
    '--maximum-error': schema.Use(float),
    '<database>': lambda n: n in ('fv3d', 'verafinger', 'hkpu', 'thufvdt'),
    '<protocol>': validate_protocol(args['<database>']),
    '<group>': validate_group(args['<database>']),
    '<size>': schema.And(schema.Use(int), lambda n: n >= 3,
      lambda n: n % 2 == 1, error='<size> must be an odd number greater ' \
          'or equal 3'),
    str: object, #ignores strings we don't care about
    }, ignore_extra_keys=True)

  return sch.validate(args)


def load_data(objects, original_directory, original_extension, footprint):
  '''Loads data, separates it in features and targets'''

  from ..preprocessor.utils import poly_to_mask

  features = None
  target = None
  pixels = 0
  loaded = 0
  logger.info('Loading at most %d samples - use CTRL-C to halt' % len(objects))

  for k, sample in enumerate(objects):

    try:
      path = sample.make_path(directory=original_directory,
          extension=original_extension)
      logger.info('Loading sample %d/%d (%s)...', loaded+1, len(objects), path)
      image = sample.load(directory=original_directory,
          extension=original_extension)
      if not (hasattr(image, 'metadata') and 'roi' in image.metadata):
        logger.info('Skipping sample (no ROI)')
        continue

      # initializes our converter filter for the current image
      filterfun = Filter(image.shape)
      pixels = numpy.prod(image.shape) #OK, this is not necessary

      if features is None and target is None:
        features = numpy.zeros(
            (len(objects)*pixels, footprint.sum()+2), dtype='float64')
        target = numpy.zeros(len(objects)*pixels, dtype='bool')

      scipy.ndimage.filters.generic_filter(
          image, filterfun, footprint=footprint, mode='nearest',
          extra_arguments=(features[k*pixels:(k+1)*pixels,:],))
      target[k*pixels:(k+1)*pixels] = poly_to_mask(image.shape,
          image.metadata['roi']).flatten()

      loaded += 1

    except KeyboardInterrupt:
      print() #avoids the ^C line
      logger.info('Gracefully stopping loading samples before limit ' \
          '(%d samples)', len(objects))
      break

  # if number of loaded samples is smaller than expected, clip features array
  features = features[:loaded*pixels]
  target = target[:loaded*pixels]

  target_float = target.astype('float64')
  target_float[~target] = -1.0
  target_float = target_float.reshape(len(target), 1)

  return features, target_float, loaded


def scan_thresholds(machine, negatives, positives, max_error=0.01):
  '''Calculates best thresholds for detecting negatives and positives'''

  neg_output = machine(negatives)
  pos_output = machine(positives)

  step = 0.01
  fg = 1.0 - step
  bg = None
  for k in numpy.arange(0.0+step, 1.0, 0.01):
    neg_errors = float((neg_output >= k).sum())/ len(negatives)
    pos_errors = float((pos_output < k).sum())/ len(positives)
    #print(k, neg_errors, pos_errors)
    if neg_errors < max_error and bg is None:
      logger.debug('Reset background threshold to %g (error = %g)', k,
          neg_errors)
      bg = k
    if pos_errors < max_error:
      logger.debug('Reset foreground threshold to %g (error = %g)', k,
          pos_errors)
      fg = k

  if bg is None:
    bg = 0.0 + step

  if fg < bg:
    bg = fg - step
    neg_errors = float((neg_output >= bg).sum())/ len(negatives)
    logger.debug('Reset background threshold to %g (error = %g) since it was '
        'bigger than the foreground threshold', bg, neg_errors)

  return fg, bg


def analyze(machine, negatives, positives, name, fg_threshold, bg_threshold):
  '''Prints performance analysis'''

  # describe errors
  neg_output = machine(negatives)
  pos_output = machine(positives)
  neg_errors = neg_output >= fg_threshold
  pos_errors = pos_output < bg_threshold
  hter = ((sum(neg_errors) / float(len(negatives))) + \
      (sum(pos_errors)) / float(len(positives))) / 2.0
  logger.info('%s set HTER: %.2f%%', name.capitalize(), 100*hter)
  logger.info('  Errors on negatives: %d / %d', sum(neg_errors), len(negatives))
  logger.info('  Errors on positives: %d / %d', sum(pos_errors), len(positives))


def plot(machine, negatives, positives, npoints, sample, directory, extension,
    fg_threshold, bg_threshold):
  '''Provides a graphical overview of errors'''

  # plot separation threshold
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  # only plot N random samples otherwise it makes it too slow
  N = numpy.random.randint(min(len(negatives), len(positives)),
      size=min(len(negatives), len(positives), npoints))

  fig = plt.figure()

  image = sample.load(directory=directory, extension=extension)

  ax = fig.add_subplot(211, projection='3d')
  ax.scatter(image.shape[1]*negatives[N,-1], image.shape[0]*negatives[N,-2],
      255*negatives[N,4], label='negatives', color='blue', marker='.')
  ax.scatter(image.shape[1]*positives[N,-1], image.shape[0]*positives[N,-2],
      255*positives[N,4], label='positives', color='red', marker='.')
  ax.set_xlabel('Width')
  ax.set_xlim(0, image.shape[1])
  ax.set_ylabel('Height')
  ax.set_ylim(0, image.shape[0])
  ax.set_zlabel('Intensity')
  ax.set_zlim(0, 255)
  ax.legend()
  ax.grid()
  ax.set_title('Ground Truth')
  plt.tight_layout()

  neg_output = machine(negatives)
  pos_output = machine(positives)

  pos_output = machine(positives)
  ax = fig.add_subplot(212, projection='3d')
  neg_plot = negatives[neg_output[:,0]>=fg_threshold]
  pos_plot = positives[pos_output[:,0]<bg_threshold]
  N = numpy.random.randint(min(len(neg_plot), len(pos_plot)),
      size=min(len(neg_plot), len(pos_plot), npoints))
  ax.scatter(image.shape[1]*neg_plot[N,-1], image.shape[0]*neg_plot[N,-2],
      255*neg_plot[N,4], label='negatives', color='red', marker='.')
  ax.scatter(image.shape[1]*pos_plot[N,-1], image.shape[0]*pos_plot[N,-2],
      255*pos_plot[N,4], label='positives', color='blue', marker='.')
  ax.set_xlabel('Width')
  ax.set_xlim(0, image.shape[1])
  ax.set_ylabel('Height')
  ax.set_ylim(0, image.shape[0])
  ax.set_zlabel('Intensity')
  ax.set_zlim(0, 255)
  ax.legend()
  ax.grid()
  ax.set_title('Classifier Errors')
  plt.tight_layout()

  print('Close plot window to continue...')
  plt.show()


def main(user_input=None):

  if user_input is not None:
    argv = user_input
  else:
    argv = sys.argv[1:]

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

  try:
    global logger #affects global logger variable
    from .validate import setup_logger
    logger = setup_logger('bob.bio.vein', args['--verbose'])
    args = validate(args)
  except schema.SchemaError as e:
    sys.exit(e)

  if args['<database>'] == 'fv3d':
    from ..configurations.fv3d import database as db
  elif args['<database>'] == 'verafinger':
    from ..configurations.verafinger import database as db
  elif args['<database>'] == 'hkpu':
    from ..configurations.hkpu import database as db
  elif args['<database>'] == 'thufvdt':
    from ..configurations.thufvdt import database as db
  else:
    raise schema.SchemaError('Database %s is not supported' % \
        args['<database>'])

  database_replacement = "%s/.bob_bio_databases.txt" % os.environ["HOME"]
  db.replace_directories(database_replacement)
  objects = db.objects(protocol=args['<protocol>'], groups=args['<group>'])
  logger.info('There are %d samples on input dataset', len(objects))
  if args['--samples'] is None:
    args['--samples'] = int(len(objects)/2)


  # calculates the footprint
  sz = args['<size>']
  if args['--hollow']:
    footprint = numpy.zeros((sz, sz), dtype=bool)
    footprint[:,0] = True # left column
    footprint[:,-1] = True # right column
    footprint[0,:] = True # top row
    footprint[-1,:] = True # bottom row
    footprint[int(sz/2), int(sz/2)] = True #center pixel
  else:
    footprint = numpy.ones((sz, sz), dtype=bool)

  logger.debug('Footprint is:\n%s', footprint.astype('int'))


  # loads data for training and validation
  train_features, train_targets, loaded = load_data(objects[:args['--samples']],
      db.original_directory, db.original_extension, footprint)

  valid_features, valid_targets, loaded = load_data(
      objects[loaded:(loaded+args['--samples'])],
      db.original_directory, db.original_extension, footprint)

  train_positives = train_features[train_targets[:,0]>0.]
  train_negatives = train_features[train_targets[:,0]<0.]
  logger.info('There are %d training samples on input dataset',
      len(train_targets))
  logger.info('  %d are negatives', len(train_negatives))
  logger.info('  %d are positives', len(train_positives))

  valid_positives = valid_features[valid_targets[:,0]>0.]
  valid_negatives = valid_features[valid_targets[:,0]<0.]
  logger.info('There are %d validation samples on input dataset',
      len(valid_targets))
  logger.info('  %d are negatives', len(valid_negatives))
  logger.info('  %d are positives', len(valid_positives))


  # machine training
  import bob.learn.mlp

  # by default, machine uses hyperbolic tangent output
  machine = bob.learn.mlp.Machine(
      (train_features.shape[1], args['--hidden'], 1))
  logger.debug('Machine architecture is %d-%d-1', train_features.shape[1],
      args['--hidden'])
  machine.randomize() #initialize weights randomly
  loss = bob.learn.mlp.SquareError(machine.output_activation)
  train_biases = True
  trainer = bob.learn.mlp.RProp(args['--batch'], loss, machine, train_biases)
  trainer.reset()
  train_shuffler = bob.learn.mlp.DataShuffler(
      [train_negatives, train_positives], [[-1.0], [+1.0]])
  valid_shuffler = bob.learn.mlp.DataShuffler(
      [valid_negatives, valid_positives], [[-1.0], [+1.0]])

  # start cost
  train_output = machine(train_features)
  train_cost = loss.f(train_output, train_targets)
  valid_output = machine(valid_features)
  valid_cost = loss.f(valid_output, valid_targets)
  logger.info('[initial] MSE = %g / %g', train_cost.mean(),
      valid_cost.mean())

  # trains the network until number of iterations or CTRL-C is pressed
  for i in range(args['--iterations']):
    try:
      train_feats, train_tgts = train_shuffler.draw(args['--batch'])
      valid_feats, valid_tgts = valid_shuffler.draw(args['--batch'])
      trainer.train(machine, train_feats, train_tgts)
      train_mse = trainer.cost(train_tgts)
      valid_mse = loss.f(machine(valid_feats), valid_tgts).mean()
      logger.info('[%d] MSE = %g / %g', i, train_mse, valid_mse)
    except KeyboardInterrupt:
      print() #avoids the ^C line
      logger.info('Gracefully stopping training before limit (%d iterations)',
          args['--batch'])
      break

  # replaces output function with a sigmoid (output between 0.0 and 1.0)
  import bob.learn.activation
  machine.output_activation = bob.learn.activation.Logistic()

  # check what is the best threshold for the just trained neural network
  fg, bg = scan_thresholds(machine, valid_negatives, valid_positives,
      args['--maximum-error'])
  logger.info('Background threshold = %g', bg)
  logger.info('Foreground threshold = %g', fg)

  # runs analysis
  analyze(machine, train_negatives, train_positives, 'training', fg, bg)
  analyze(machine, valid_negatives, valid_positives, 'validation', fg, bg)

  if args['--plot']:
    plot(machine, valid_negatives, valid_positives, args['--points'],
        objects[0], db.original_directory, db.original_extension, fg, bg)

  # save models
  import bob.io.base
  h5f = bob.io.base.HDF5File(args['--model'], 'w')
  machine.save(h5f)
  h5f['footprint'] = footprint
  h5f['fg_threshold'] = fg
  h5f['bg_threshold'] = bg
  del h5f
  logger.info('Saved MLP model and footprint to %s', args['--model'])
