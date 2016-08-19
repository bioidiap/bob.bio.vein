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


Repeated Line-Tracking with Miura Matching
==========================================

You can find the description of this method on the paper from Miura *et al.*
[MNM04]_.

To run the baseline on the `VERA fingervein`_ database, using the ``NOM``
protocol (called ``Full`` in [TVM14]_), do the following:


.. code-block:: sh

   $ ./bin/verify.py --database=verafinger --protocol=NOM --preprocessor=nopp --extractor=repeatedlinetracking --algorithm=match-rlt --sub-directory="rlt" --verbose --verbose


.. tip::

   If you have more processing cores on your local machine and don't want to
   submit your job for SGE execution, you can run it in parallel (using 4
   parallel tasks) by adding the options ``--parallel=4 --nice=10``.


This command line selects and runs the following implementations for the
toolchain:

* Database: Use the base Bob API for the VERA database implementation,
  protocol variant ``NOM`` which corresponds to the ``Full`` evaluation
  protocol described in [TVM14]_
* Preprocessor: Simple finger cropping, with no extra post-processing, as
  defined in [LLP09]_
* Feature extractor: Repeated line tracking, as explained in [MNM04]_
* Matching algorithm: "Miura" matching, as explained on the same paper
* Subdirectory: This is the subdirectory in which the scores and intermediate
  results of this baseline will be stored.


As the tool runs, you'll see printouts that show how it advances through
preprocessing, feature extraction and matching. In a 4-core machine and using
4 parallel tasks, it takes around 2 hours to process this baseline with the
current code implementation.

To complete the evaluation, run the commands bellow, that will output the equal
error rate (EER) and plot the detector error trade-off (DET) curve with the
performance:

.. code-block:: sh

   $ ./bin/bob_eval_threshold.py  --scores <path-to>/verafinger/rlt/NOM/nonorm/scores-dev --criterium=eer
   ('Threshold:', 0.320748535)
   FAR : 26.478% (12757/48180)
   FRR : 26.364% (58/220)
   HTER: 26.421%


Maximum Curvature with Miura Matching
=====================================

You can find the description of this method on the paper from Miura *et al.*
[MNM05]_.

To run the baseline on the `VERA fingervein`_ database, using the ``NOM``
protocol like above, do the following:


.. code-block:: sh

   $ ./bin/verify.py --database=verafinger --protocol=NOM --preprocessor=nopp --extractor=maximumcurvature --algorithm=match-mc --sub-directory="mc" --verbose --verbose


This command line selects and runs the following implementations for the
toolchain, with comparison to the previous baseline:

* Feature extractor: Maximum Curvature, as explained in [MNM05]_

In a 4-core machine and using 4 parallel tasks, it takes around 1 hour and 40
minutes to process this baseline with the current code implementation.  Results
we obtained:

.. code-block:: sh

   $ ./bin/bob_eval_threshold.py  --scores <path-to>/verafinger/mc/NOM/nonorm/scores-dev --criterium=eer
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

   $ ./bin/verify.py --database=verafinger --protocol=NOM --preprocessor=nopp --extractor=widelinedetector --algorithm=match-wld --sub-directory="wld" --verbose --verbose


This command line selects and runs the following implementations for the
toolchain, with comparison to the previous baseline:

* Feature extractor: Wide Line Detector, as explained in [HDLTL10]_

In a 4-core machine and using 4 parallel tasks, it takes only around 5 minutes
minutes to process this baseline with the current code implementation.
Results we obtained:

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
