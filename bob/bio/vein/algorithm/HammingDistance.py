#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.base.algorithm import Distance
import scipy.spatial.distance


class HammingDistance (Distance):
  """This class calculates the Hamming distance between two binary images.

  Each binary image is first flattened by concatenating its rows to form a one-dimensional vector.  The Hamming distance is then calculated between the two binary vectors.
  The Hamming distance is computed using :py:func:`scipy.spatial.distance.hamming`, which returns a scalar ``float`` to represent the proportion of mismatching corresponding bits between the two binary vectors.

  **Parameters:**

  ``distance_function`` : function
    Set this parameter to ``scipy.spatial.distance.hamming`` to ensure we are calculating the Hamming distance

  ``is_distance_function`` : bool
    Set this flag to ``False`` to ensure that Hamming distances are returned as positive values rather than negative 

  """


  def __init__(
      self,
      distance_function = scipy.spatial.distance.hamming,
      is_distance_function = False  # setting this to False ensures that Hamming distances are returned as positive values rather than negative
  ):

    # Call base class constructor
    Distance.__init__(
        self,
        distance_function = distance_function,
        is_distance_function = is_distance_function
    )