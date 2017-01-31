#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import MaxEigenvalues

max_eigenvalues_non_zero_neg_mean_norm_s5 = MaxEigenvalues(sigma = 5, set_negatives_to_zero = False, mean_normalization_flag = True)

max_eigenvalues_zero_neg_mean_norm_s5 = MaxEigenvalues(sigma = 5, set_negatives_to_zero = True, mean_normalization_flag = True)

max_eigenvalues_s5 = MaxEigenvalues(sigma = 5, set_negatives_to_zero = False, mean_normalization_flag = False)

max_eigenvalues_zero_neg_mean_norm_s5_binary = MaxEigenvalues(sigma = 5, set_negatives_to_zero = True, mean_normalization_flag = True, binarization_flag = True)

max_eigenvalues_s5_hist_eqlz = MaxEigenvalues(sigma = 5, set_negatives_to_zero = False, mean_normalization_flag = False, equalize_adapthist_flag = True)

