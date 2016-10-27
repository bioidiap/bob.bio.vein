.. vim: set fileencoding=utf-8 :

.. _bob.bio.vein.results-biowave-v1:

============================================
Baseline experiments for BIOWAVE V1 database
============================================


Introduction
------------

In this page is summarized baseline experiment results on `BioWave V1`_ database. More information -- :py:mod:`bob.bio.vein.configurations.biowave_v1` and also -- `BioWave V1`_ .


Fully-automated experiments
---------------------------

The baseline verification results are summarized in this section. 
The evaluation of verification pipe-lines is done in two steps:

  1. First, the papameters of each algorithm are adjusted using the grid search on the ``Idiap_1_1_R_a`` protocol of the `BioWave V1`_ database. 
     Only ``'dev'`` set of the database is used in the grid search.
  2. Once best parameters are selected the performance is comuted for 
     ``Idiap_1_1_R``, ``Idiap_1_5_R``, ``Idiap_5_5_R``, ``Idiap_1_1_L``, ``Idiap_1_5_L``, ``Idiap_5_5_L`` protocols of the `BioWave V1`_ database.

Maximum Curvature Features + Miura Matching Algorithm
********************************************************

**Evaluated verification pipe-line:**

  \{ ``KMeansRoi`` or ``TopographyCutRoi`` \} Preprocessor + ``MaximumCurvature`` Extractor + ``MiuraMatch`` Algorithm

In the first sequence of experiments the best performing ``Preprocessor`` is selected. 
The following options are iteratively passed to the verification algorithm (arguments for the ``verify.py`` script):

  1. ``--preprocessor`` \{ ``kmeans-roi``, ``kmeans-roi-conv``, ``kmeans-roi-conv-erode-40``, 
     ``topography-cut-roi``, ``topography-cut-roi-conv``, ``topography-cut-roi-conv-erode`` \}
  2. ``--extractor maximumcurvature``
  3. ``--algorithm miura-match-wrist-100``

The results are summarized in the following Table:

EER for the ``'dev'`` set, ``Idiap_1_1_R_a`` protocol of the `BioWave V1`_ database.

+-----------------------------------+----------+
|          ``Preprocessor``         |  EER,\%  |
+===================================+==========+
|           ``kmeans-roi``          |  24.375  |
+-----------------------------------+----------+
|        ``kmeans-roi-conv``        |  25.625  |
+-----------------------------------+----------+
|    ``kmeans-roi-conv-erode-40``   |  26.250  |
+-----------------------------------+----------+
|       ``topography-cut-roi``      |  22.188  |
+-----------------------------------+----------+
|    ``topography-cut-roi-conv``    |  25.938  |
+-----------------------------------+----------+
| ``topography-cut-roi-conv-erode`` |  27.837  |
+-----------------------------------+----------+

The ROC curves for the particular experiment can be downlooaded from here:

:download:`ROC curve <img/ROC_verification_experiment_1.pdf>`

Based on the inspection of ROC curves k-means-based ROI extraction algorithms outperform the topography-cut-based methods. 
The lowest EER among k-means-based ROI approaches is obtained for ``kmeans-roi``, therefore upcoming experiments incorporate this method in the preprocessing stage. 


In the second sequence of experiments the best performing ``Algorithm`` is selected. 
The following options are iteratively passed to the verification algorithm (arguments for the ``verify.py`` script):

  1. ``--preprocessor kmeans-roi``
  2. ``--extractor maximumcurvature``
  3. ``--algorithm`` \{ ``miura-match-wrist-20``, ``miura-match-wrist-40``, ``miura-match-wrist-60``, ``miura-match-wrist-80``, 
     ``miura-match-wrist-100``, ``miura-match-wrist-120``, ``miura-match-wrist-140``, ``miura-match-wrist-160`` \}

Options in the ``--algorithm`` stage represent the search region in the Miura matching algorithm, which is iteratively increased from 20 to 160 pixels in both X and Y directions.

The results are summarized in the following Table:

EER for the ``'dev'`` set, ``Idiap_1_1_R_a`` protocol of the `BioWave V1`_ database.

+------------------------------+----------+
|         ``Algorithm``        |  EER,\%  |
+==============================+==========+
|   ``miura-match-wrist-20``   |  37.812  |
+------------------------------+----------+
|   ``miura-match-wrist-40``   |  34.062  |
+------------------------------+----------+
|   ``miura-match-wrist-60``   |  29.375  |
+------------------------------+----------+
|   ``miura-match-wrist-80``   |  25.312  |
+------------------------------+----------+
|   ``miura-match-wrist-100``  |**24.375**|
+------------------------------+----------+
|   ``miura-match-wrist-120``  |  24.688  |
+------------------------------+----------+
|   ``miura-match-wrist-140``  |  25.938  |
+------------------------------+----------+
|   ``miura-match-wrist-160``  |  28.438  |
+------------------------------+----------+

The ROC curves for the particular experiment can be downlooaded from here:

:download:`ROC curve <img/ROC_verification_experiment_2.pdf>`

According to the above table the optimal search region in the Miura Matching algorithm is 100 pixels (both horizontal and vertical directions).
This hyperparameter is now also fixed in the next sequence of grid-search experiments.

In the third sequence of experiments the morphological dilation of the vein patterns is evaluated. 
In this scenario the veins are first dilated with disk shaped kernel and then Miura matching is applied.
The following options are iteratively passed to the verification algorithm (arguments for the ``verify.py`` script):

  1. ``--preprocessor kmeans-roi``
  2. ``--extractor maximumcurvature``
  3. ``--algorithm`` \{ ``miura-match-wrist-dilation-5``, ``miura-match-wrist-dilation-7``, ``miura-match-wrist-dilation-9``, 
     ``miura-match-wrist-dilation-11``, ``miura-match-wrist-dilation-13``, ``miura-match-wrist-dilation-15``, ``miura-match-wrist-dilation-17`` \}

Number in the ``--algorithm`` names represents the value of the diameter (in pixels) of the disk shaped kernel used for the dilation of the veins.

The results are summarized in the following Table:

EER for the ``'dev'`` set, ``Idiap_1_1_R_a`` protocol of the `BioWave V1`_ database.

+--------------------------------------+----------+
|         ``Algorithm``                |  EER,\%  |
+======================================+==========+
|   ``miura-match-wrist-dilation-5``   |**24.083**|
+--------------------------------------+----------+
|   ``miura-match-wrist-dilation-7``   |  24.367  |
+--------------------------------------+----------+
|   ``miura-match-wrist-dilation-9``   |  24.724  |
+--------------------------------------+----------+
|   ``miura-match-wrist-dilation-11``  |  25.625  |
+--------------------------------------+----------+
|   ``miura-match-wrist-dilation-13``  |  26.562  |
+--------------------------------------+----------+
|   ``miura-match-wrist-dilation-15``  |  28.125  |
+--------------------------------------+----------+
|   ``miura-match-wrist-dilation-17``  |  30.000  |
+--------------------------------------+----------+

The ROC curves for the particular experiment can be downlooaded from here:

:download:`ROC curve <img/ROC_verification_experiment_3.pdf>`


Based on the above results the best perfoming verification pipe-line is composed of the following modules:

  ``kmeans-roi`` Preprocessor + ``maximumcurvature`` Extractor + ``miura-match-wrist-dilation-5`` Algorithm




Annotation comparison
---------------------

In this section annotation comparison results are summarized.

Protocol ``Idiap_1_1_R``
************************

Algorithm - ``AnnotationMatch``, ``not centred`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|  Algorithm - ``AnnotationMatch``    | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|    47.500%    |    43.417%   |    50.353%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|    48.125%    |    42.540%   |    51.562%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|    47.736%    |    43.125%   |    51.562%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|    47.500%    |    43.201%   |    51.562%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|    47.812%    |    43.069%   |    52.175%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|    47.873%    |    42.188%   |    53.438%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|    47.500%    |    41.603%   |    53.758%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|    47.812%    |  **41.310%** |    54.062%    |
+-------------------------------------+---------------+--------------+---------------+

Algorithm - ``AnnotationMatch``, ``centred`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|  Algorithm - ``AnnotationMatch``    | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|   43.750%     |   39.375%    |   47.812%     |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|   42.476%     |   38.510%    |   46.875%     |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|   40.625%     |   36.875%    |   46.186%     |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|   38.750%     |   35.337%    |   45.389%     |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|   38.706%     |   33.125%    |   44.744%     |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|   37.576%     |   32.188%    |   45.000%     |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|   37.552%     | **31.875%**  |   45.312%     |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|   37.500%     | **31.875%**  |   45.373%     |
+-------------------------------------+---------------+--------------+---------------+

Algorithm - ``AnnotationMatch``, ``not centred`` **but** ``rotated`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|  Algorithm - ``AnnotationMatch``    | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|    49.756%    |    44.375%   |    52.881%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|    49.928%    |    43.157%   |    53.990%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|    50.008%    |    43.121%   |    53.750%    |   
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|    50.000%    |    42.812%   |    53.750%    |   
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|    49.375%    |    42.500%   |    53.458%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|    49.451%    |    42.500%   |    53.125%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|    49.062%    |    42.496%   |    52.812%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|    49.062%    |  **42.428%** |    52.444%    |
+-------------------------------------+---------------+--------------+---------------+

Algorithm - ``AnnotationMatch``, ``centred`` **and** ``rotated`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|  Algorithm - ``AnnotationMatch``    | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|    46.562%    |    41.631%   |    49.062%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|    45.312%    |    39.631%   |    48.750%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|    43.438%    |    37.232%   |    48.438%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|    40.000%    |    34.375%   |    47.500%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|    38.438%    |    32.812%   |    45.938%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|    37.500%    |    32.188%   |    45.661%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|    36.875%    |  **31.562%** |    45.240%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|    36.310%    |  **31.262%** |    45.000%    |
+-------------------------------------+---------------+--------------+---------------+


Algorithm - ``AnnotationMatch``, ``centred`` **but not** ``rotated`` annotations.

+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|   Algorithm - ``MiuraMatch``        |               ``EER`` using different ``dilation`` parameters                |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|        Algorithm parameters         |        0      |      5       |      9        |      13       |     17        |
+=====================================+===============+==============+===============+===============+===============+
|          ``ch = cw = 40``           |    26.250%    |   25.962%    |    24.948%    |    23.750%    |    23.750%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 60``           |    25.000%    |   23.429%    |    23.125%    |    22.504%    |    22.865%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 80``           |    23.750%    |   22.776%    |    22.812%    |    22.812%    |    22.232%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 100``          |    24.062%    |   22.188%    |    22.500%    |    23.125%    |    22.812%    | 
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 120``          |    24.960%    |   23.125%    |    21.875%    |    22.188%    |    22.500%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 140``          |    26.562%    |   24.062%    |  **21.278%**  |    21.875%    |    23.057%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 160``          |    25.625%    |   23.718%    |    22.812%    |    22.188%    |    24.375%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+




Algorithm - ``MiuraMatch``, ``centred`` **and** ``rotated`` annotations.


+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|   Algorithm - ``MiuraMatch``        |               ``EER`` using different ``dilation`` parameters                |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|        Algorithm parameters         |        0      |      5       |      9        |      13       |     17        |
+=====================================+===============+==============+===============+===============+===============+
|          ``ch = cw = 40``           |     24.688%   |    23.177%   |    23.125%    |    23.438%    |    24.375%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 60``           |     22.812%   |    22.812%   |    22.188%    |    21.875%    |    23.125%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 80``           |     21.875%   |    22.188%   |    21.875%    |    22.188%    |    22.532%    |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 100``          |     22.508%   |    22.812%   |    21.903%    |    22.500%    |     23.438%   |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 120``          |     23.750%   |    21.562%   |  **20.625%**  |    21.831%    |     24.062%   |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 140``          |     24.062%   |    21.875%   |    20.938%    |    22.135%    |     23.510%   |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+
|          ``ch = cw = 160``          |     25.312%   |    22.812%   |    21.250%    |    22.188%    |     23.125%   |
+-------------------------------------+---------------+--------------+---------------+---------------+---------------+


Protocol ``Idiap_5_5_R``
************************

Algorithm - ``AnnotationMatch``, ``not centred`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|  Algorithm - ``AnnotationMatch``    | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|    48.085%    |    45.000%   |    49.936%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|    48.758%    |    44.399%   |    50.497%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|    48.758%    |    44.399%   |    50.497%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|    47.500%    |    43.750%   |    51.354%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|    46.971%    |    43.125%   |    52.356%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|    46.995%    |    43.269%   |    51.250%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|    47.540%    |    43.021%   |    51.875%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|    47.997%    |  **42.604%** |    52.027%    |
+-------------------------------------+---------------+--------------+---------------+

Algorithm - ``AnnotationMatch``, ``centred`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|               Algorithm             | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|     40.601%   |     40.601%  |     45.625%   |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|     38.125%   |     40.577%  |     46.178%   |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|     38.750%   |     38.125%  |     43.750%   |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|     36.362%   |     36.771%  |     44.327%   |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|     35.000%   |     35.649%  |     43.125%   |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|     35.000%   |     35.000%  |     43.125%   |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|     35.761%   |     34.375%  |     42.003%   | 
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|     36.250%   |   **34.279%**|     43.021%   |
+-------------------------------------+---------------+--------------+---------------+


Algorithm - ``AnnotationMatch``, ``not centred`` **but** ``rotated`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|  Algorithm - ``AnnotationMatch``    | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|    50.625%    |    45.617%   |    52.476%    |   
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|    48.598%    |    44.904%   |    50.625%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|    48.846%    |    44.952%   |    51.875%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|    48.750%    |    43.662%   |    51.875%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|    49.375%    |    43.101%   |    52.500%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|    48.229%    |    42.372%   |    53.053%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|    47.516%    |  **41.875%** |    53.750%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|    47.388%    |    41.971%   |    53.694%    |
+-------------------------------------+---------------+--------------+---------------+

Algorithm - ``AnnotationMatch``, ``centred`` **and** ``rotated`` annotations.

+-------------------------------------+---------------+--------------+---------------+
|  Algorithm - ``AnnotationMatch``    | ``EER`` using different score fusion methods |
+-------------------------------------+---------------+--------------+---------------+
|        Algorithm parameters         |   ``mean``    |   ``min``    |   ``max``     |
+=====================================+===============+==============+===============+
|                        ``sigma = 0``|     43.125%   |    42.500%   |    46.875%    | 
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 1``|     42.500%   |    41.963%   |    46.250%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 2``|     41.875%   |    40.112%   |    44.375%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 3``|     39.920%   |    39.375%   |    42.500%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 4``|     37.500%   |    36.250%   |    43.093%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 5``|     36.250%   |    34.968%   |    42.500%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 6``|     36.242%   |    34.375%   |    43.125%    |
+-------------------------------------+---------------+--------------+---------------+
|                        ``sigma = 7``|     36.875%   |  **33.862%** |    42.644%    |
+-------------------------------------+---------------+--------------+---------------+



ROI detection
-------------

In this section the automatic ROI detection algorithms are compared to the manually annotated ROI data.
For this purpose the annotated images of the `BioWave V1`_ database are used.

Let's introduce the following notations: 

  * :math:`ROI_a` - a set representing the automatically obtained ROI.
  * :math:`ROI_m` - a set representing the manually annotated ROI.


Three ROI evaluation metrices are used in our case:

  1. :math:`m_1 = (ROI_a \cap ROI_m) / ROI_m` - how large area of manual ROI is coverd by automatic ROI,
     relative to manual ROI
  2. :math:`m_2 = (ROI_a - ROI_m) / ROI_m` - how large area of automatic ROI is located outside of manual ROI,
     relative to manual ROI
  3. :math:`m_3` - Euclidean distance between centers of mass of :math:`ROI_a` and :math:`ROI_m`


K-means based ROI
*******************

Fisrt, the k-means based ROI detection approach is tested. 
This approach is implemented in ``bob.bio.vein.preprocessors.KMeansRoi``.

The arguments of the class are as follows:

.. code-block:: sh
  
  KMeansRoi(filter_name = "medianBlur", mask_size = 7, erode_mask_flag = False, convexity_flag = False)

The mean/average values of the above evaluation metrices over all annotated files are as follows:

  1. :math:`\bar{m_1} = 0.5767`
  2. :math:`\bar{m_2} = 0.0028`
  3. :math:`\bar{m_3} = 16.096`

The large values of :math:`m_2` are mostly caused by the ROI misdetections, which are displayed in the image below.
In this image the ROI's satisfying the condition :math:`m_2 > 0.01` are displayed:

.. image:: img/ROI_outliers_k_means.png


Topography-cut based ROI
**************************

Second, the topography-cut based ROI detection approach is tested. 
This approach is implemented in ``bob.bio.vein.preprocessors.TopographyCutRoi``.

The arguments of the class are as follows:

.. code-block:: sh
  
  TopographyCutRoi(blob_xywh_offsets = [ 1, 1, 1, 1 ], 
                   filter_name = "medianBlur", 
                   mask_size = 7, 
                   topography_step = 20, 
                   erode_mask_flag = False, 
                   convexity_flag = False)

The mean/average values of the above evaluation metrices over all annotated files are as follows:

  1. :math:`\bar{m_1} = 0.7622`
  2. :math:`\bar{m_2} = 0.0464`
  3. :math:`\bar{m_3} = 16.279`



Vein segmentation
-----------------

TBD



.. include:: links.rst

