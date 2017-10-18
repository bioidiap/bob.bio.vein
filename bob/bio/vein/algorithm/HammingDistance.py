#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.base.algorithm import Distance
import scipy.spatial.distance


class HammingDistance (Distance):
  """Finger vein matching: Hamming Distance between binary fingervein feature vectors
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