.. vim: set fileencoding=utf-8 :

.. _bob.bio.vein.results-put-vein:

=========================================================
Baseline experiments for PUT Vein (wrist only) database
=========================================================


Introduction
------------

In this section the results of verification experiments for the `PUT`_ database are summarized. In particular **only the wrist** images are utilized in the experiments. 


State of the art verification results (using binary vein images)
------------------------------------------------------------------

This section introduces the most successful verification experiments, which are based on matching of **binary** patterns of the veins. The results are reported for the ``wrist-L-1``, ``wrist-LR-1``, ``wrist-R-1``, ``wrist-RL-1``, ``wrist-L-4``, ``wrist-LR-4``, ``wrist-R-4`` and ``wrist-RL-4`` protocols of the `PUT`_ database.

The verification pipeline utilizes the following classes:

  1. preprocessor ``KMeansRoi`` :py:class:`bob.bio.vein.preprocessors.KMeansRoi`,
  2. extractor ``MaximumCurvatureScaleRotation`` :py:class:`bob.bio.vein.extractors.MaximumCurvatureScaleRotation`,
  3. matching algorithm ``MiuraMatchRotationFast`` :py:class:`bob.bio.vein.algorithms.MiuraMatchRotationFast`.

The successful parameters for the instances of the above classes are listed below:

.. code-block:: python

   preprocessor = KMeansRoi(filter_name = "gaussian_filter", mask_size = 7,
                            correct_mask_flag = False, correction_erosion_factor = 7,
                            erode_mask_flag = False, erosion_factor = 100,
                            convexity_flag = False,
                            rotation_centering_flag = False,
                            centering_flag = True,
                            normalize_scale_flag = True,
                            mask_to_image_area_ratio = 0.2,
                            equalize_adapthist_flag = False,
                            speedup_flag = True)

   extractor = MaximumCurvatureScaleRotation(sigma = 9,
                                             norm_p2p_dist_flag = False, 
                                             selected_mean_dist = 100,
                                             sum_of_rotated_images_flag = False, 
                                             angle_limit = 10, angle_step = 1,
                                             speed_up_flag = False)

   algorithm = MiuraMatchRotationFast(ch = 300, cw = 225,
                                      angle_limit = 10, angle_step = 1,
                                      perturbation_matching_flag = True,
                                      kernel_radius = 3,
                                      score_fusion_method = 'max',
                                      gray_scale_input_flag = False)

To reproduce the experiments using ``verify.py`` script the following entry points can be used:

  1. ``--preprocessor putvein-kmeans-roi-centered-scaled-fast``
  2. ``--extractor putvein-maximum-curvature-sigma-9``
  3. ``--algorithm putvein-miura-match-fast-300-225-rotation-10-step-1-pert-3-max1``

An example of the command to run the verification algorithm for the ``wrist-L_1`` protocol of the database is:

.. code-block:: shell-session

  ./bin/verify.py putvein --preprocessor putvein-kmeans-roi-centered-scaled-fast --extractor putvein-maximum-curvature-sigma-9 --algorithm putvein-miura-match-fast-300-225-rotation-10-step-1-pert-3-max1 --protocol wrist-L_1 --groups 'dev' 'eval' --sub-directory <PATH_TO_SAVE_THE_RESULTS> 

The EER/HTER errors are summarized in the Table below.

EER (``'dev'`` set), HTER (``'eval'`` set), different protocols of the `PUT`_ database.

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``wrist-L-1``   |  15.151  |  10.315  |
+-------------------+----------+----------+
|   ``wrist-LR-1``  |  11.000  |  9.262   |
+-------------------+----------+----------+
|   ``wrist-R-1``   |  11.625  |**8.094** |
+-------------------+----------+----------+
|   ``wrist-RL-1``  |  7.812   |  9.721   |
+-------------------+----------+----------+
|   ``wrist-L-4``   |  8.000   |  4.792   |
+-------------------+----------+----------+
|   ``wrist-LR-4``  |  4.500   |  3.584   |
+-------------------+----------+----------+
|   ``wrist-R-4``   |  6.000   |**3.240** |
+-------------------+----------+----------+
|   ``wrist-RL-4``  |  3.000   |  4.245   |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_putvein_binary_state_of_art.pdf>`

------------

State of the art verification results (using gray-scale vein images)
----------------------------------------------------------------------

This section introduces the most successful verification experiments, which are based on matching of **gray-scale** patterns of the veins. The results are reported for the ``wrist-L-1``, ``wrist-LR-1``, ``wrist-R-1``, ``wrist-RL-1``, ``wrist-L-4``, ``wrist-LR-4``, ``wrist-R-4`` and ``wrist-RL-4`` protocols of the `PUT`_ database.

The verification pipeline utilizes the following classes:

  1. preprocessor ``KMeansRoi`` :py:class:`bob.bio.vein.preprocessors.KMeansRoi`,
  2. extractor ``MaxEigenvalues`` :py:class:`bob.bio.vein.extractors.MaxEigenvalues`,
  3. matching algorithm ``MiuraMatchRotationFast`` :py:class:`bob.bio.vein.algorithms.MiuraMatchRotationFast`.

The successful parameters for the instances of the above classes are listed below:

.. code-block:: python

   preprocessor = KMeansRoi(filter_name = "gaussian_filter", mask_size = 7,
                            correct_mask_flag = False, correction_erosion_factor = 7,
                            erode_mask_flag = False, erosion_factor = 100,
                            convexity_flag = False,
                            rotation_centering_flag = False,
                            centering_flag = True,
                            normalize_scale_flag = True,
                            mask_to_image_area_ratio = 0.2,
                            equalize_adapthist_flag = False,
                            speedup_flag = True)

   extractor = MaxEigenvalues(sigma = 9, 
                              segment_veins_flag = True,
                              amplify_segmented_veins_flag = True,
                              two_layer_segmentation_flag = True,
                              binarize_flag = False,
                              kernel_size = 3,
                              norm_p2p_dist_flag = False,
                              selected_mean_dist = 100)

   algorithm = MiuraMatchRotationFast(ch = 300, cw = 225,
                                      angle_limit = 10, angle_step = 1,
                                      perturbation_matching_flag = False,
                                      kernel_radius = 3,
                                      score_fusion_method = 'max',
                                      gray_scale_input_flag = True)

To reproduce the experiments using ``verify.py`` script the following entry points can be used:

  1. ``--preprocessor putvein-kmeans-roi-centered-scaled-fast``
  2. ``--extractor max-eigenvalues-s9-two-layer-segment-veins-amplified``
  3. ``--algorithm putvein-miura-match-fast-300-225-rotation-10-step-1-gray-max1``

An example of the command to run the verification algorithm for the ``wrist-L_1`` protocol of the database is:

.. code-block:: shell-session

  ./bin/verify.py putvein --preprocessor putvein-kmeans-roi-centered-scaled-fast --extractor max-eigenvalues-s9-two-layer-segment-veins-amplified --algorithm putvein-miura-match-fast-300-225-rotation-10-step-1-gray-max1 --protocol wrist-L_1 --groups 'dev' 'eval' --sub-directory <PATH_TO_SAVE_THE_RESULTS>

The EER/HTER errors are summarized in the Table below.

EER (``'dev'`` set), HTER (``'eval'`` set), different protocols of the `PUT`_ database.

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``wrist-L-1``   |  23.125  |  16.906  |
+-------------------+----------+----------+
|   ``wrist-LR-1``  |  14.625  |  12.930  |
+-------------------+----------+----------+
|   ``wrist-R-1``   |  16.625  |**11.599**|
+-------------------+----------+----------+
|   ``wrist-RL-1``  |  11.625  |  13.517  |
+-------------------+----------+----------+
|   ``wrist-L-4``   |  13.000  |  9.865   |
+-------------------+----------+----------+
|   ``wrist-LR-4``  |  8.000   |  6.202   |
+-------------------+----------+----------+
|   ``wrist-R-4``   |  6.500   |**3.750** |
+-------------------+----------+----------+
|   ``wrist-RL-4``  |  3.750   |  5.992   |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_putvein_grayscale_state_of_art.pdf>`

------------

Baseline verification results (using binary vein images)
------------------------------------------------------------------

Maximum Curvature Features + Miura Matching Algorithm
********************************************************

This section introduces baseline verification experiments, which are based on matching of **binary** patterns of the veins. The results are reported for the ``wrist-L-1``, ``wrist-LR-1``, ``wrist-R-1``, ``wrist-RL-1``, ``wrist-L-4``, ``wrist-LR-4``, ``wrist-R-4`` and ``wrist-RL-4`` protocols of the `PUT`_ database.

The verification pipeline utilizes the following classes:

  1. preprocessor ``KMeansRoi`` :py:class:`bob.bio.vein.preprocessors.KMeansRoi`,
  2. extractor ``MaximumCurvatureScaleRotation`` :py:class:`bob.bio.vein.extractors.MaximumCurvatureScaleRotation`,
  3. matching algorithm ``MiuraMatchFusion`` :py:class:`bob.bio.vein.algorithms.MiuraMatchFusion`.

The successful parameters for the instances of the above classes are listed below:

.. code-block:: python

   preprocessor = KMeansRoi(filter_name = "gaussian_filter", mask_size = 7,
                            correct_mask_flag = False, correction_erosion_factor = 7,
                            erode_mask_flag = False, erosion_factor = 100,
                            convexity_flag = False,
                            rotation_centering_flag = False,
                            centering_flag = True,
                            normalize_scale_flag = True,
                            mask_to_image_area_ratio = 0.2,
                            equalize_adapthist_flag = False,
                            speedup_flag = True)

   extractor = MaximumCurvatureScaleRotation(sigma = 9,
                                             norm_p2p_dist_flag = False, 
                                             selected_mean_dist = 100,
                                             sum_of_rotated_images_flag = False, 
                                             angle_limit = 10, angle_step = 1,
                                             speed_up_flag = False)

   algorithm = MiuraMatchFusion(ch = 300, cw = 225, 
                                score_fusion_method = 'max')


To reproduce the experiments using ``verify.py`` script the following entry points can be used:

  1. ``--preprocessor putvein-kmeans-roi-centered-scaled-fast``
  2. ``--extractor putvein-maximum-curvature-sigma-9``
  3. ``--algorithm miura-match-fusion-300-225-max``

An example of the command to run the verification algorithm for the ``wrist-L_1`` protocol of the database is:

.. code-block:: shell-session

  ./bin/verify.py putvein --preprocessor putvein-kmeans-roi-centered-scaled-fast --extractor putvein-maximum-curvature-sigma-9 --algorithm miura-match-fusion-300-225-max --protocol wrist-L_1 --groups 'dev' 'eval' --sub-directory <PATH_TO_SAVE_THE_RESULTS> 

The EER/HTER errors are summarized in the Table below.

EER (``'dev'`` set), HTER (``'eval'`` set), different protocols of the `PUT`_ database.

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``wrist-L-1``   |  16.000  |  9.685   |
+-------------------+----------+----------+
|   ``wrist-LR-1``  |  10.938  |  9.584   |
+-------------------+----------+----------+
|   ``wrist-R-1``   |  12.750  |**8.596** |
+-------------------+----------+----------+
|   ``wrist-RL-1``  |  9.062   |  10.635  |
+-------------------+----------+----------+
|   ``wrist-L-4``   |  11.000  |  6.104   |
+-------------------+----------+----------+
|   ``wrist-LR-4``  |  6.750   |**5.855** |
+-------------------+----------+----------+
|   ``wrist-R-4``   |  10.000  |  6.083   |
+-------------------+----------+----------+
|   ``wrist-RL-4``  |  5.283   |  6.355   |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_putvein_binary_baseline_1.pdf>`

------------

Baseline verification results (using gray-scale vein images)
------------------------------------------------------------------

Hessian-based Features + Miura Matching Algorithm
********************************************************

This section introduces baseline verification experiments, which are based on matching of **gray-scale** patterns of the veins. The results are reported for the ``wrist-L-1``, ``wrist-LR-1``, ``wrist-R-1``, ``wrist-RL-1``, ``wrist-L-4``, ``wrist-LR-4``, ``wrist-R-4`` and ``wrist-RL-4`` protocols of the `PUT`_ database.

The verification pipeline utilizes the following classes:

  1. preprocessor ``KMeansRoi`` :py:class:`bob.bio.vein.preprocessors.KMeansRoi`,
  2. extractor ``MaxEigenvalues`` :py:class:`bob.bio.vein.extractors.MaxEigenvalues`,
  3. matching algorithm ``MiuraMatchFusion`` :py:class:`bob.bio.vein.algorithms.MiuraMatchFusion`.

The successful parameters for the instances of the above classes are listed below:

.. code-block:: python

   preprocessor = KMeansRoi(filter_name = "gaussian_filter", mask_size = 7,
                            correct_mask_flag = False, correction_erosion_factor = 7,
                            erode_mask_flag = False, erosion_factor = 100,
                            convexity_flag = False,
                            rotation_centering_flag = False,
                            centering_flag = True,
                            normalize_scale_flag = True,
                            mask_to_image_area_ratio = 0.2,
                            equalize_adapthist_flag = False,
                            speedup_flag = True)

   extractor = MaxEigenvalues(sigma = 9, 
                              segment_veins_flag = True,
                              amplify_segmented_veins_flag = True,
                              two_layer_segmentation_flag = True,
                              binarize_flag = False,
                              kernel_size = 3,
                              norm_p2p_dist_flag = False,
                              selected_mean_dist = 100)

   algorithm = MiuraMatchFusion(ch = 300, cw = 225, 
                                score_fusion_method = 'max')

To reproduce the experiments using ``verify.py`` script the following entry points can be used:

  1. ``--preprocessor putvein-kmeans-roi-centered-scaled-fast``
  2. ``--extractor max-eigenvalues-s9-two-layer-segment-veins-amplified``
  3. ``--algorithm miura-match-fusion-300-225-max``

An example of the command to run the verification algorithm for the ``wrist-L_1`` protocol of the database is:

.. code-block:: shell-session

  ./bin/verify.py putvein --preprocessor putvein-kmeans-roi-centered-scaled-fast --extractor max-eigenvalues-s9-two-layer-segment-veins-amplified --algorithm miura-match-fusion-300-225-max --protocol wrist-L_1 --groups 'dev' 'eval' --sub-directory <PATH_TO_SAVE_THE_RESULTS>

The EER/HTER errors are summarized in the Table below.

EER (``'dev'`` set), HTER (``'eval'`` set), different protocols of the `PUT`_ database.

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``wrist-L-1``   |  23.875  |  17.620  |
+-------------------+----------+----------+
|   ``wrist-LR-1``  |  16.070  |  13.888  |
+-------------------+----------+----------+
|   ``wrist-R-1``   |  18.125  |**12.607**|
+-------------------+----------+----------+
|   ``wrist-RL-1``  |  12.750  |  14.816  |
+-------------------+----------+----------+
|   ``wrist-L-4``   |  12.500  |  10.042  |
+-------------------+----------+----------+
|   ``wrist-LR-4``  |  8.500   |  7.760   |
+-------------------+----------+----------+
|   ``wrist-R-4``   |  10.000  |**6.271** |
+-------------------+----------+----------+
|   ``wrist-RL-4``  |  6.500   |  7.579   |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_putvein_grayscale_baseline_1.pdf>`

------------

.. include:: links.rst
