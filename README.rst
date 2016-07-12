.. vim: set fileencoding=utf-8 :
.. Fri 08 Jul 2016 15:38:56 CEST

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.bio.vein/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.bio.vein/master/index.html
.. image:: https://travis-ci.org/bioidiap/bob.bio.vein.svg?branch=master
   :target: https://travis-ci.org/bioidiap/bob.bio.vein
.. image:: https://coveralls.io/repos/bioidiap/bob.bio.vein/badge.png
   :target: https://coveralls.io/r/bioidiap/bob.bio.vein
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/bioidiap/bob.bio.vein/tree/master
.. image:: http://img.shields.io/pypi/v/bob.bio.vein.png
   :target: https://pypi.python.org/pypi/bob.bio.vein
.. image:: http://img.shields.io/pypi/dm/bob.bio.vein.png
   :target: https://pypi.python.org/pypi/bob.bio.vein


=========================================
 The Biometrics Vein Recognition Library
=========================================

Welcome to this Vein Recognition Library based on Bob. This library is designed
to perform a fair comparison of vein recognition algorithms. It contains
scripts to execute various of vein recognition experiments on a variety
of vein image databases, and running the help is as easy as going to the
command line and typing::

  $ bin/verify.py --help


About
-----

This library is currently developed at the `Biometrics group
<http://www.idiap.ch/scientific-research/research-groups/biometric-person-recognition>`_
at the `Idiap Research Institute <http://www.idiap.ch>`_.  The vein recognition
library is designed to run vein recognition experiments in a comparable and
reproducible manner.


Databases
---------

To achieve this goal, interfaces to some publicly available vein image
databases are contained, and default evaluation protocols are defined, e.g.:

- UTFVP - University of Twente Finger Vein Database [http://www.sas.ewi.utwente.nl/]
- VERA - Finger vein Database [http://www.idiap.ch/scientific-research/resources]
- PUT - The PUT biometric vein (palm and wrist) recognition dataset [http://biometrics.put.poznan.pl/vein-dataset/]


Algorithms
----------

Together with that, implementations of a variety of traditional and
state-of-the-art vein recognition algorithms are provided:

* Maximum Curvature [MNM05]_
* Repeated Line Tracking [MNM04]_
* Wide Line Detector [HDLTL10]_

Tools to evaluate the results can easily be used to create scientific plots. We
also provide handles to run experiments using parallel processes or an SGE
grid.


Extensions
----------

On top of these already pre-coded algorithms, the vein recognition library
provides an easy Python interface for implementing new image preprocessors,
feature types, vein recognition algorithms or database interfaces, which
directly integrate into the vein recognition experiment. Hence, after a short
period of coding, researchers can compare new ideas directly with already
existing algorithms in a fair manner.


References
----------

.. [MNM05] *N. Miura, A. Nagasaka, and T. Miyatake*. **Extraction of Finger-Vein Pattern Using Maximum Curvature Points in Image Profiles**. Proceedings on IAPR conference on machine vision applications, 9, pp. 347--350, 2005.

.. [MNM04] *N. Miura, A. Nagasaka, and T. Miyatake*. **Feature extraction of finger vein patterns based on repeated line tracking and its application to personal identification**. Machine Vision and Applications, Vol. 15, Num. 4, pp. 194--203, 2004.

.. [HDLTL10] *B. Huang, Y. Dai, R. Li, D. Tang and W. Li*. **Finger-vein authentication based on wide line detector and pattern normalization**. Proceedings of the 20th International Conference on Pattern Recognition (ICPR), 2010.


Installation
------------

The latest version of the vein recognition library can be installed with our
`Conda-based builds`_. Once you have installed Bob_, just go on and install
this package using the same installation mechanism.


Development
-----------

In order to develop the latest version of this package, install Bob_ as
indicated above. Once that is done, do this::

  $ git clone https://gitlab.idiap.ch/biometrc/bob.bio.vein.git
  $ cd bob.bio.vein
  $ python bootstrap-buildout.py
  $ ./bin/buildout

After those steps, you should have a functional **development** environment to
test the package. The python interpreter and base environment on line 3 above
should be pre-installed with all dependencies required for Bob_ to operate
correctly. For example, you may start from our `conda-based builds`_ and then
use the Python interpreter in there to bootstrap your local development
environment.


Running tests
-------------

To verify that your installation worked as expected, you might want to run our
unit tests with::

  $ ./bin/nosetests -sv



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
.. _conda-based builds: https://github.com/idiap/bob/wiki/Binary-Installation
