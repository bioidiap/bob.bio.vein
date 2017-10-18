# Calculates the Hamming distance (proportion of mismatching corresponding bits) between two binary vectors

import bob.bio.base
import scipy.spatial.distance
algorithm = bob.bio.base.algorithm.Distance(
    distance_function = scipy.spatial.distance.hamming,
    is_distance_function = False  # setting this to False ensures that Hamming distances are returned as positive values rather than negative
)