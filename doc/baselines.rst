.. vim: set fileencoding=utf-8 :
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.vein.baselines:

===============================
 Executing Baseline Algorithms
===============================

The first thing you might want to do is to execute one of the vein
recognition algorithms that are implemented in ``bob.bio.vein``.


Setting up Databases
--------------------

In order to run vein recognition algorithms using this package, you'll need to
make sure to download the raw files corresponding to the databases you'd like
to process. The raw files are not distributed with Bob_ software as biometric
data is, to most countries, considered sensible data that cannot be obtained
without explicit licensing from a data controller. You must visit the websites
below, sign the license agreements and then download the data before trying out
to run the baselines.

.. note::

   If you're at the Idiap Research Institute in Switzlerand, the datasets in
   the baselines mentioned in this guide are already downloaded and
   pre-installed on our shared file system. You don't need to re-download
   databases or create a ``~/.bob_bio_databases.txt`` file.


The current system readily supports the following freely available datasets:

* ``vera``: `Vera Fingervein`_
* ``utfvp``: `UTFVP`_
* ``put``: `PUT`_ Vein Dataset


After downloading the databases, annotate the base directories in which they
are installed. Then, follow the instructions in
:ref:`bob.bio.base.installation` to let this framework know where databases are
located on your system.


Running Baseline Experiments
----------------------------

To run the baseline experiments, you can use the ``./bin/verify.py`` script by
just going to the console and typing:

.. code-block:: sh

   $ ./bin/verify.py


This script is explained in more detail in :ref:`bob.bio.base.experiments`.
The ``./bin/verify.py --help`` option shows you, which other options you can
set.

Usually it is a good idea to have at least verbose level 2 (i.e., calling
``./bin/verify.py --verbose --verbose``, or the short version ``./bin/verify.py
-vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can define an SGE grid or local host
   (multi-processing) configurations as explained in
   :ref:`running_in_parallel`.

   In short, to run in the Idiap SGE grid, you can simply add the ``--grid``
   command line option, without parameters. To run experiments in parallel on
   the local machine, simply add a ``--parallel <N>`` option, where ``<N>``
   specifies the number of parallel jobs you want to execute.


In the remainder of this section we introduce baseline experiments you can
readily run with this tool without further configuration.


Repeated Line-Tracking with Miura Matching
==========================================

You can find the description of this method on the paper from Miura *et al.*
[MNM04]_.

To run the baseline on the `VERA fingervein`_ database, using the ``1vsAll``
protocol (1-fold cross-validation), do the following:

.. code-block:: sh

   ./bin/verify.py --database=vera --protocol=1vsAll --preprocessor=none --extractor=repeatedlinetracking --algorithm=match-rlt --sub-directory="vera-1vsall-mnm04" --verbose --verbose

This command line selects the following implementations for the toolchain:

  * Database: Use the base Bob API for the VERA database implementation,
    protocol variant ``1vsAll`` which corresponds to the 1-fold
    cross-validation evaluation protocol described in [TVM14]_
  * Preprocessor: Simple finger cropping, with no extra pre-processing
  * Feature extractor: Repeated line tracking, as explained in [MNM04]_
  * Matching algorithm: "Miura" matching, as explained on the same paper

As the tool runs, you'll see printouts that show how it advances through
preprocessing, feature extraction and matching.


Available Resources
-------------------

This package provides various different ``bob.bio.base`` resource
configurations to handle a variety of techniques in vein recognition: database
adaptors, preprocessors (cropping and illumination
normalization), feature extractors and matching algorithms. In order to list
each contribution, use the script ``./bin/resources.py``.

For available resources:

  .. code-block:: sh

     $ ./bin/resources.py --packages=bob.bio.vein

For a detailed explanation and the configurability of each resource, consult
:ref:`bob.bio.vein.api`.


.. include:: links.rst
