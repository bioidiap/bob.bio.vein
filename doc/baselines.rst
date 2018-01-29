.. vim: set fileencoding=utf-8 :
.. date: Wed 06 Dec 2017 11:07:29 CET

.. _bob.bio.vein.baselines:

===============================
 Executing Baseline Algorithms
===============================

The first thing you might want to do is to execute one of the vein
recognition algorithms that are implemented in ``bob.bio.vein``.


Running Baseline Experiments
----------------------------

To run the baseline experiments, you can use the ``verify.py`` script by
just going to the console and typing:

.. code-block:: sh

   $ verify.py


This script is explained in more detail in :ref:`bob.bio.base.experiments`.
The ``verify.py --help`` option shows you, which other options you can
set.

Usually it is a good idea to have at least verbose level 2 (i.e., calling
``verify.py --verbose --verbose``, or the short version ``verify.py
-vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can define an SGE grid or local host
   (multi-processing) configurations as explained in
   :ref:`running_in_parallel`.

   In short, to run in the Idiap SGE grid, you can simply add the ``--grid``
   command line option, without parameters. To run experiments in parallel on
   the local machine, simply add a ``--parallel <N>`` option, where ``<N>``
   specifies the number of parallel jobs you want to execute.


Database setups and baselines are encoded using
:ref:`bob.bio.base.configuration-files`, all stored inside the package root, in
the directory ``bob/bio/vein/configurations``. Documentation for each resource
is available on the section :ref:`bob.bio.vein.resources`.

.. warning::

   You **cannot** run experiments just by executing the command line
   instructions described in this guide. You **need first** to procure yourself
   the raw data files that correspond to *each* database used here in order to
   correctly run experiments with those data. Biometric data is considered
   private data and, under EU regulations, cannot be distributed without a
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
      [YOUR_FV3D_DIRECTORY] = /complete/path/to/fv3d
      [YOUR_HKPU_DIRECTORY] = /complete/path/to/hkpu
      [YOUR_THUFVDT_DIRECTORY] = /complete/path/to/thu-fvfdt
      [YOUR_MMCBNU6K_DIRECTORY] = /complete/path/to/mmcbnu-6000
      [YOUR_HMTVEIN_DIRECTORY] = /complete/path/to/sdumla-hmt-vein
      [YOUR_PUTVEIN_DIRECTORY] = /complete/path/to/put-vein


   Notice it is rather important to use the strings as described above,
   otherwise ``bob.bio.base`` will not be able to correctly load your images.

   Once this step is done, you can proceed with the instructions below.


In the remainder of this section we introduce baseline experiments you can
readily run with this tool without further configuration. Baselines examplified
in this guide were published in [TVM14]_.


Repeated Line-Tracking with Miura Matching
==========================================

Detailed description at
:py:mod:`bob.bio.vein.configurations.repeated_line_tracking`.

To run the baseline on the `VERA fingervein`_ database, using the ``Nom``
protocol, do the following:


.. code-block:: sh

   $ verify.py verafinger annotations rlt -vv


.. tip::

   If you have more processing cores on your local machine and don't want to
   submit your job for SGE execution, you can run it in parallel (using 4
   parallel tasks) by adding the options ``--parallel=4 --nice=10``.

   Optionally, you may use the ``parallel`` resource configuration which
   already sets the number of parallel jobs to the number of hardware cores you
   have installed on your machine (as with
   :py:func:`multiprocessing.cpu_count`) and sets ``nice=10``. For example:

   .. code-block:: sh

      $ verify.py verafinger annotations rlt parallel -vv

   To run on the Idiap SGE grid using our stock
   io-big-48-slots-4G-memory-enabled (see
   :py:mod:`bob.bio.vein.configurations.gridio4g48`) configuration, use:

   .. code-block:: sh

      $ verify.py verafinger annotations rlt grid -vv

   You may also, optionally, use the configuration resource ``gridio4g48``,
   which is just an alias of ``grid`` in this package.



This command line selects and runs the following implementations for the
toolchain:

* :py:mod:`bob.bio.vein.configurations.verafinger`
* :py:mod:`bob.bio.vein.configurations.annotations`
* :py:mod:`bob.bio.vein.configurations.repeated_line_tracking`

As the tool runs, you'll see printouts that show how it advances through
preprocessing, feature extraction and matching. In a 4-core machine and using
4 parallel tasks, it takes around 4 hours to process this baseline with the
current code implementation.

To complete the evaluation, run the command bellow, that will output the equal
error rate (EER) and plot the detector error trade-off (DET) curve with the
performance:

.. code-block:: sh

   $ evaluate.py -c EER -D det.pdf -d <path-to>/verafinger/rlt/Nom/nonorm/scores-dev
   The EER of the development set of '<path-to>/verafinger/rlt/Nom/nonorm/scores-dev' is 23.636%


Maximum Curvature with Miura Matching
=====================================

Detailed description at :ref:`bob.bio.vein.resources.recognition.mc`.

To run the baseline on the `VERA fingervein`_ database, using the ``Nom``
protocol like above, do the following:


.. code-block:: sh

   $ verify.py verafinger mc -vv


This command line selects and runs the following implementations for the
toolchain:

* :py:mod:`bob.bio.vein.configurations.verafinger`
* :py:mod:`bob.bio.vein.configurations.annotations`
* :py:mod:`bob.bio.vein.configurations.maximum_curvature`

In a 4-core machine and using 4 parallel tasks, it takes around 1 hour and 40
minutes to process this baseline with the current code implementation. Results
we obtained:


.. code-block:: sh

   $ evaluate.py -c EER -D det.pdf -d <path-to>/verafinger/mc/Nom/nonorm/scores-dev
   The EER of the development set of '<path-to>/verafinger/mc/Nom/nonorm/scores-dev' is 4.467%


Wide Line Detector with Miura Matching
======================================

You can find the description of this method on the paper from Huang *et al.*
[HDLTL10]_.

To run the baseline on the `VERA fingervein`_ database, using the ``Nom``
protocol like above, do the following:


.. code-block:: sh

   $ verify.py verafinger wld -vv


This command line selects and runs the following implementations for the
toolchain:

* :py:mod:`bob.bio.vein.configurations.verafinger`
* :py:mod:`bob.bio.vein.configurations.annotations`
* :py:mod:`bob.bio.vein.configurations.wide_line_detector`

In a 4-core machine and using 4 parallel tasks, it takes only around 5 minutes
minutes to process this baseline with the current code implementation.Results
we obtained:

.. code-block:: sh

   $ evaluate.py -c EER -D det.pdf -d <path-to>/verafinger/wld/Nom/nonorm/scores-dev
   The EER of the development set of '<path-to>/verafinger/wld/Nom/nonorm/scores-dev' is 9.658%


Results for other Baselines
===========================

This package may generate results for other combinations of protocols and
databases. Here is a summary table for some variants (results expressed
correspond to the the equal-error rate on the development set, in percentage):


.. _baselines_table_annotations:
.. table:: Baselines Available (Hand Annotations)
   :widths: auto

   ========================== ============= ======= ======= =======
     Database (resource)       --protocol     rlt     wld     mc
   ========================== ============= ======= ======= =======
    UTFVP (utfvp)               nom
    VERA-finger (verafinger)    Nom          19.1      7.2    1.8
    HKPU (hkpu)                 A             6.4      6.3    4.2
    THU-FVFDT (thufvdt)         p3            4.4      3.8    2.4
    MMCBNU_6000 (mmcbnu6k)      default
    SDUMLA-HMT (hmtvein)        default      31.4     18.7   11.6
   ========================== ============= ======= ======= =======

.. _baselines_table_watershedding:
.. table:: Baselines Available (Watershed Mask)
   :widths: auto

   ========================== ============= ======= ======= =======
     Database (resource)       --protocol     rlt     wld     mc
   ========================== ============= ======= ======= =======
    UTFVP (utfvp)               nom
    VERA-finger (verafinger)    Nom          19.5     6.4     2.8
    HKPU (hkpu)                 A             8.6     7.2     4.8
    THU-FVFDT (thufvdt)         p3           11.4    13.1    11.0
    MMCBNU_6000 (mmcbnu6k)      default
    SDUMLA-HMT (hmtvein)        default      32.2    19.2    12.0
   ========================== ============= ======= ======= =======

.. _baselines_table_tomemask:
.. table:: Baselines Available (Tome's Mask)
   :widths: auto

   ========================== ============= ======= ======= =======
     Database (resource)       --protocol     rlt     wld     mc
   ========================== ============= ======= ======= =======
    UTFVP (utfvp)               nom           1.4     1.9     0.4
    VERA-finger (verafinger)    Nom          23.6     9.7     4.5
    HKPU (hkpu)                 A            42.3    14.7    14.5
    THU-FVFDT (thufvdt)         p3           33.8    28.2    33.1
    MMCBNU_6000 (mmcbnu6k)      default       9.3     8.8     3.2
    SDUMLA-HMT (hmtvein)        default      31.3    19.2    11.8
   ========================== ============= ======= ======= =======

.. _database_dimensions:
.. table:: Database Dimensions
   :widths: auto

   ========================== ============= ========== ========== ========== =========== ===========
     Database (resource)       --protocol    subjects   fingers    genuines   impostors    scores
   ========================== ============= ========== ========== ========== =========== ===========
    UTFVP (utfvp)               nom             18         108         216        23112       23328
    VERA-finger (verafinger)    Nom            110         220         220        48180       48400
    HKPU (hkpu)                 A              156         218        7560      1500040     1587600
    THU-FVFDT (thufvdt)         p3             610         610         610       371490      327100
    MMCBNU_6000 (mmcbnu6k)      default        100         600       15000      8985000     9000000
    SDUMLA-HMT (hmtvein)        default        106         636        5724      3634740     3640464
   ========================== ============= ========== ========== ========== =========== ===========


Numbers in :numref:`baselines_table_annotations` correspond to the percentual
Equal-Error-Rate (EER) on the development set of these datasets/protocols.
Legend: ``rlt`` - Repeated Line Tracking; ``wld`` - Wide Line Detector; ``mc``
- Maximum Curvature. Each baseline is followed by a Miura-Matching algorithm
(correlation). The ``(resource)`` entry refers to the resource name describing
the database configuration while the ``--protocol`` entry defines the protocol
override so that experiment runs using the defined evaluation protocol. For
example, to execute the ``mc`` baseline against VERA-finger using the ``B``
protocol, using all cores on your machine, do this:


.. code-block:: sh

   $ verify.py verafinger --protocol=B annotations mc -vvv parallel


Numbers in :numref:`database_dimensions` show some information about the number
of subjects and unique fingers available in each database/protocol combination.
We also display the number of scores for the evaluation of each protocol. These
numbers given an estimate on the amount of processing power required to run the
protocol and on the reliability of the error rates reported in
:numref:`baselines_table_annotations`.


The DET curves of such combinations of database, protocols and baselines can be
generated using a command-line similar to this:

.. code-block:: sh

   $ evaluate.py -D det.pdf -d <path-to>/verafinger/{rlt,wld,mc}/Nom/nonorm/scores-dev -T 'VERA-finger (Nom)' -l rlt wld mc


Which generates a figure like this:


.. figure:: img/det.*
   :scale: 50%

   Example Detector-Error-Tradeoff (DET) curve of multiple experiments on the
   VERA-finger dataset using the Nom protocol.


To reproduce all baselines, just repeat it for every combination of database
and protocol.

.. note::

   The script ``evaluate.py`` may generate other types of plots including ROC
   and EPC plots. Read its help message for detailed instructions.



Modifying Baseline Experiments
------------------------------

It is fairly easy to modify baseline experiments available in this package. To
do so, you must copy the configuration files for the given baseline you want to
modify, edit them to make the desired changes and run the experiment again.

For example, suppose you'd like to change the protocol on the Vera Fingervein
database, *through a modification on a configuration file*, and use the
protocol ``B`` instead of the default protocol ``Nom``.  First, you need to
identify where the configuration file sits:

.. code-block:: sh

   $ resources.py -tc -p bob.bio.vein
   - bob.bio.vein X.Y.Z @ /path/to/bob.bio.vein:
     + grid       --> bob.bio.vein.configurations.gridio4g48
     + gridio4g48 --> bob.bio.vein.configurations.gridio4g48
     + parallel   --> bob.bio.vein.configurations.parallel
     + utfvp      --> bob.bio.vein.configurations.utfvp
     + verafinger --> bob.bio.vein.configurations.verafinger
     + putvein    --> bob.bio.vein.configurations.putvein
     + hkpu       --> bob.bio.vein.configurations.hkpu
     + thufvdt    --> bob.bio.vein.configurations.thufvdt
     + mmcbnu6k   --> bob.bio.vein.configurations.mmcbnu6k
     + hmtvein    --> bob.bio.vein.configurations.hmtvein
     + fv3d       --> bob.bio.vein.configurations.fv3d
     + mc         --> bob.bio.vein.configurations.maximum_curvature
     + rlt        --> bob.bio.vein.configurations.repeated_line_tracking
     + wld        --> bob.bio.vein.configurations.wide_line_detector


In order to modify it, create a local file called, e.g., ``verafinger_b.py``
that imports all assets in the original configuration file and change the
``protocol`` variable to set it to the required value (``B``). The final
version of the new configuration file could look like this:

.. code-block:: python

   from bob.bio.vein.configurations.verafinger import *

   #override the "protocol" setting to the value wanted
   protocol = 'B'


Now, re-run the experiment using your modified database descriptor:

.. code-block:: sh

   $ verify.py ./verafinger_b.py wld -vv


Notice we replace the use of the registered configuration file named
``verafinger`` by the local file ``verafinger_b.py``. This makes the program
``verify.py`` take that into consideration instead of the original file.


Other Tools
-----------


Pipeline Display
================

You can use the program ``view_sample.py`` to display the images after
full processing using:

.. code-block:: sh

   $ ./bin/view_sample.py --save=output-dir verafinger /path/to/processed/directory 030-M/030_L_1
   $ # open output-dir

And you should be able to view images like these (example taken from the Vera
fingervein database, using the automatic annotator and Maximum Curvature
feature extractor):

.. figure:: img/preprocessed.*
   :scale: 50%

   Example RoI overlayed on finger vein image of the Vera fingervein database,
   as produced by the script ``view_sample.py``.


.. figure:: img/binarized.*
   :scale: 50%

   Example of fingervein image from the Vera fingervein database, binarized by
   using Maximum Curvature, after pre-processing.


.. include:: links.rst
