.. vim: set fileencoding=utf-8 :
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.vein.baselines:

===============================
 Executing Baseline Algorithms
===============================

The first thing you might want to do is to execute one of the vein
recognition algorithms that are implemented in ``bob.bio.vein``.


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
readily run with this tool without further configuration. Baselines examplified
in this guide were published in [TVM14]_.

Database setups and baselines are encoded using
:ref:`bob.bio.base.configuration-files`, all stored inside the package root, in
the directory ``bob/bio/vein/configurations``. Documentation for each resource
is available on the section :ref:`bob.bio.vein.resources`.

.. warning::

   You **cannot** run experiments just by executing the command line
   instructions described in this guide. You **need first** to procure yourself
   the raw data files that correspond to *each* database used here in order to
   correctly run experiments with those data. Biometric data is considered
   private date and, under EU regulations, cannot be distributed without a
   consent or license. You may consult our
   :ref:`bob.bio.vein.resources.databases` resources section for checking
   currently supported databases and accessing download links for the raw data
   files.

   Once the raw data files have been downloaded, particular attention should be
   given to the directory locations of those. Unpack the databases carefully
   and annotate the root directory where they have been unpacked.

   Then, carefully read the *Databases* section of
   :ref:`bob.bio.base.installation` on how to correctly setup the
   ``~/.bob_bio_databases.txt`` file.

   Use the following keywords on the left side of the assignment (see
   :ref:`bob.bio.vein.resources.databases`):

   .. code-block:: text

      [YOUR_VERAFINGER_DIRECTORY] = /complete/path/to/verafinger
      [YOUR_UTFVP_DIRECTORY] = /complete/path/to/utfvp
      [YOUR_BIOWAVE_TEST_DIRECTORY] = /complete/path/to/biowave_test

   Notice it is rather important to use the strings as described above,
   otherwise ``bob.bio.base`` will not be able to correctly load your images.

   Once this step is done, you can proceed with the instructions below.


Repeated Line-Tracking with Miura Matching
==========================================

Detailed description at :ref:`bob.bio.vein.resources.recognition.rlt`.

To run the baseline on the `VERA fingervein`_ database, using the ``Nom``
protocol, do the following:


.. code-block:: sh

   $ ./bin/verify.py verafinger rlt -vv


.. tip::

   If you have more processing cores on your local machine and don't want to
   submit your job for SGE execution, you can run it in parallel (using 4
   parallel tasks) by adding the options ``--parallel=4 --nice=10``.


This command line selects and runs the following implementations for the
toolchain:

* :ref:`bob.bio.vein.resources.database.verafinger`
* :ref:`bob.bio.vein.resources.recognition.rlt`

As the tool runs, you'll see printouts that show how it advances through
preprocessing, feature extraction and matching. In a 4-core machine and using
4 parallel tasks, it takes around 4 hours to process this baseline with the
current code implementation.

To complete the evaluation, run the command bellow, that will output the equal
error rate (EER) and plot the detector error trade-off (DET) curve with the
performance:

.. code-block:: sh

   $ ./bin/bob_eval_threshold.py  --scores <path-to>/verafinger/rlt/Nom/nonorm/scores-dev --criterium=eer
   ('Threshold:', 0.32045327)
   FAR : 26.362% (12701/48180)
   FRR : 26.364% (58/220)
   HTER: 26.363%


Maximum Curvature with Miura Matching
=====================================

Detailed description at :ref:`bob.bio.vein.resources.recognition.mc`.

To run the baseline on the `VERA fingervein`_ database, using the ``Nom``
protocol like above, do the following:


.. code-block:: sh

   $ ./bin/verify.py verafinger mc -vv


This command line selects and runs the following implementations for the
toolchain:

* :ref:`bob.bio.vein.resources.database.verafinger`
* :ref:`bob.bio.vein.resources.recognition.mc`

In a 4-core machine and using 4 parallel tasks, it takes around 1 hour and 40
minutes to process this baseline with the current code implementation. Results
we obtained:

.. code-block:: sh

   $ ./bin/bob_eval_threshold.py  --scores <path-to>/verafinger/mc/Nom/nonorm/scores-dev --criterium=eer
   ('Threshold:', 0.078274325)
   FAR : 3.182% (1533/48180)
   FRR : 3.182% (7/220)
   HTER: 3.182%


Wide Line Detector with Miura Matching
======================================

You can find the description of this method on the paper from Huang *et al.*
[HDLTL10]_.

To run the baseline on the `VERA fingervein`_ database, using the ``NOM``
protocol like above, do the following:


.. code-block:: sh

   $ ./bin/verify.py verafinger wld -vv


This command line selects and runs the following implementations for the
toolchain:

* :ref:`bob.bio.vein.resources.database.verafinger`
* :ref:`bob.bio.vein.resources.recognition.wld`

In a 4-core machine and using 4 parallel tasks, it takes only around 5 minutes
minutes to process this baseline with the current code implementation.Results
we obtained:

.. code-block:: sh

   $ ./bin/bob_eval_threshold.py  --scores <path-to>/verafinger/wld/NOM/nonorm/scores-dev --criterium=eer
   ('Threshold:', 0.239141175)
   FAR : 10.455% (5037/48180)
   FRR : 10.455% (23/220)
   HTER: 10.455%


Results for other Baselines
===========================

This package may generate results for other combinations of protocols and
databases. Here is a summary table for some variants (results are expressed the
the equal-error rate on the development set, in percentage):

======================== ================= ====== ====== ====== ======
               Approach                        UTFVP          Vera
------------------------------------------ ------------- -------------
   Feature Extractor      Post Processing     B    Full    B     Full
======================== ================= ====== ====== ====== ======
Maximum Curvature         Histogram Eq.
Maximum Curvature            None                                 3.2
Repeated Line Tracking    Histogram Eq.
Repeated Line Tracking       None                                26.4
Wide Line Detector        Histogram Eq.                           8.2
Wide Line Detector           None                                10.4
======================== ================= ====== ====== ====== ======

WLD + HEQ (preproc) @ Vera/Full = 10.9%


.. include:: links.rst
