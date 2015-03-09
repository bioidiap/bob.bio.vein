.. vim: set fileencoding=utf-8 :
.. author: Pedro Tome <pedro.tome@idiap.ch>
.. date: Wed Jan 14 11:58:57 CEST 2015

.. _fingerveinreclib:

=============================================
 Welcome to FingerveinRecLib's documentation!
=============================================

The FingerveinRecLib is an open source tool (based on FaceRecLib_) that is designed to run comparable and reproducible fingervein recognition experiments.
   Most of this documentation is based on the documentation of FaceRecLib_.


To design a fingervein recognition experiment, one has to choose:

* an image databases and its according protocol,
* finger contours detection and image preprocessing algorithms,
* the type of features to extract from the fingervein image,
* the fingervein recognition algorithm to employ, and
* the way to evaluate the results

For any of these parts, several different types are implemented in the FingerveinRecLib, and basically any combination of the five parts can be executed.
For each type, several meta-parameters can be tested.
This results in a nearly infinite amount of possible fingervein recognition experiments that can be run using the current setup.
But it is also possible to use your own database, preprocessing, feature type, or fingervein recognition algorithm and test this against the baseline algorithms implemented in the FingerveinRecLib.

If you are interested, please continue reading:


===========
Users Guide
===========

.. toctree::
   :maxdepth: 2

   installation
   experiments
   evaluate
   contribute
   satellite
   references


This documentation is still under development.


.. include:: links.rst
.. _facereclib: http://pythonhosted.org/facereclib/index.html