.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Fri 08 Jul 2016 15:38:56 CEST

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.fingervein/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.fingervein/master/index.html
.. image:: https://travis-ci.org/bioidiap/bob.fingervein.svg?branch=master
   :target: https://travis-ci.org/bioidiap/bob.fingervein
.. image:: https://coveralls.io/repos/bioidiap/bob.fingervein/badge.png
   :target: https://coveralls.io/r/bioidiap/bob.fingervein
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/bioidiap/bob.fingervein/tree/master
.. image:: http://img.shields.io/pypi/v/bob.fingervein.png
   :target: https://pypi.python.org/pypi/bob.fingervein
.. image:: http://img.shields.io/pypi/dm/bob.fingervein.png
   :target: https://pypi.python.org/pypi/bob.fingervein


=========================================
 The Biometrics Vein Recognition Library
=========================================

Welcome to the Finger vein Recognition Library based on Bob. This library is
designed to perform a fair comparison of finger vein recognition algorithms.
It contains scripts to execute various kinds of finger vein recognition
experiments on a variety of finger vein image databases, and running the help
is as easy as going to the command line and typing::

  $ bin/veinverify.py --help


About
-----

This library is developed at the `Biometrics group
<http://www.idiap.ch/scientific-research/research-groups/biometric-person-recognition>`_
at the `Idiap Research Institute <http://www.idiap.ch>`_.  The vein recognition
library is designed to run vein recognition experiments in a comparable and
reproducible manner.


Databases
---------

To achieve this goal, interfaces to some publicly available vein image
databases are contained, and default evaluation protocols are defined, e.g.:

- UTFVP - University of Twente Finger Vein Database [http://website]
- VERA Finger vein Database [http://www.idiap.ch/scientific-research/resources]


Algorithms
----------

Together with that, implementations of a variety of traditional and
state-of-the-art vein recognition algorithms are provided:

* Maximum Curvature [MNM+05]_
* Repeated Line Tracking [MNM+04]_
* Wide Line Detector [HDLTL+10]_

Tools to evaluate the results can easily be used to create scientific plots. We
also provide handles to run experiments using parallel processes or an SGE
grid.


Extensions
----------

On top of these already pre-coded algorithms, the vein recognition library
provides an easy Python interface for implementing new image preprocessors,
feature types, finger vein recognition algorithms or database interfaces, which
directly integrate into the fingervein recognition experiment. Hence, after a
short period of coding, researchers can compare their new invention directly
with already existing algorithms in a fair manner.


References
----------

.. [MNM+05]  *N. Miura, A. Nagasaka, and T. Miyatake*. **Extraction of Finger-Vein Pattern Using Maximum Curvature Points in Image Profiles**. Proceedings on IAPR conference on machine vision applications, 9, pp. 347--350, 2005.

.. [MNM+04]  *N. Miura, A. Nagasaka, and T. Miyatake*. **Feature extraction of finger vein patterns based on repeated line tracking and its application to personal identification**. Machine Vision and Applications, Vol. 15, Num. 4, pp. 194--203, 2004.

.. [HDLTL+10]  *B. Huang, Y. Dai, R. Li, D. Tang and W. Li*. **Finger-vein authentication based on wide line detector and pattern normalization**. Proceedings of the 20th International Conference on Pattern Recognition (ICPR), 2010.


Installation
------------

To download the finger vein library, go to
http://pypi.python.org/pypi/bob.bio.vein click on the **download** button and
extract the .zip file to a folder of your choice.

The FingerVeinRecLib is a satellite package of the free signal processing and
machine learning library Bob_. We advise you install Bob first and then use
that installation to bootstrap the installation of this package::

  $ python bootstrap.py
  $ bin/buildout

This will download any further dependencies and install them locally.


Running tests
-------------

To verify that your installation worked as expected, you might want to run our
unit tests with::

  $ bin/nosetests

Usually, all tests should pass, if you use the latest packages of Bob_.


Cite our paper
--------------

If you use this library in any of your experiments, please cite the following
paper::

  @inproceedings{Tome_IEEEBIOSIG2014,
      author = {Tome, Pedro and Vanoni, Matthias and Marcel, S{\'{e}}bastien},
      keywords = {Biometrics, Finger vein, Spoofing Attacks},
      projects = {Idiap, BEAT},
      month = sep,
      title = {On the Vulnerability of Finger Vein Recognition to Spoofing},
      booktitle = {IEEE International Conference of the Biometrics Special Interest Group (BIOSIG)},
      volume = {230},
      year = {2014},
      ocation = {Darmstadt, Germay},
      pdf = {http://publications.idiap.ch/downloads/papers/2014/Tome_IEEEBIOSIG2014.pdf}
}


.. _bob: http://www.idiap.ch/software/bob
.. _idiap: http://www.idiap.ch
