.. vim: set fileencoding=utf-8 :
.. Mon 11 Jul 2016 16:35:18 CEST

.. _experiments:

=====================
 Running Experiments
=====================

For running experiments with a defined setup, you should use ``bin/verify.py``
directly. Follow the instructions on bob.bio.base_ for listing and using all
resources available in this package. In this section, we discuss specificities
for added plugins.


.. _databases:

Databases
---------


Required Parameters
~~~~~~~~~~~~~~~~~~~

* ``name``: The name of the database, in lowercase letters without special
  characters. This name will be used as a default sub-directory to separate
  resulting files of different experiments.
* ``protocol``: The name of the protocol that should be used. If omitted, the
  protocol ``Default`` will be used (which might not be available in all
  databases, so please specify).


.. _preprocessors:

Preprocessors
-------------


Vein Cropping Parameters
~~~~~~~~~~~~~~~~~~~~~~~~


* ``mask_h``: Height of the cropping finger mask.
* ``mask_w``: Width of the cropping finger mask.
* ``padding_offset``: An offset to the paddy array to be applied arround the
  fingervein image.
* ``padding_threshold``: The pixel value of this paddy array. Defined to 0.2 to
  uncontrolled (low quality) fingervein databases and to 0 for controlled (high
  quality) fingervein databases. (By default 0.2).
* ``preprocessing``: The pre-processing applied to the fingervein image before
  finger contour extraction. By default equal to ``None``.
* ``fingercontour``: The algorithm used to localize the finger contour.
  Options: 'leemaskMatlab' - Implementation based on [LLP09]_, 'leemaskMod' -
  Modification based on [LLP09]_ for uncontrolled images introduced by author,
  and 'konomask' - Implementation based on [KUU02]_.
* ``postprocessing``: The post-processing applied to the fingervein image after
  the finger contour extraction.  Options: 'None', 'HE' - Histogram
  Equalization, 'HFE' - High Frequency Enphasis Filtering [ZTXL09]_,
  'CircGabor' - Circular Gabor Filters [ZY09]_.


.. note::
   Currently, the pre-processing is fixed to ``None`` by default.


.. _algorithms:

Recognition Algorithms
----------------------

There are also a variety of recognition algorithms implemented in the
FingerveinRecLib.  All finger recognition algorithms are based on the
:py:class:`FingerveinRecLib.tools.Tool` base class.  This base class has
parameters that some of the algorithms listed below share.  These parameters
mainly deal with how to compute a single score when more than one feature is
provided for the model or for the probe:

Here is a list of the most important algorithms and their parameters:


* :py:class:`FingerveinRecLib.tools.MiuraMatch`: Computes the match ratio based
  on [MNM04]_ convolving the two template image.  Return score - Value between
  0 and 0.5, larger value is better match.

  * ``ch``: Maximum search displacement in y-direction. Different defult values
    based on the different features.
  * ``cw``: Maximum search displacement in x-direction. Different defult values
    based on the different features.

* :py:class:`FingerveinRecLib.tools.HammingDistance`: Computes the Hamming Distance between two fingervein templates.


.. include:: links.rst
