#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors.MaskedHessianHistogram import MaskedHessianHistogram

hesshist_extractor_sigma7bins50 = MaskedHessianHistogram( sigma = 7, n_bins = 50 )
hesshist_extractor_sigma7bins100 = MaskedHessianHistogram( sigma = 7, n_bins = 100 )
hesshist_extractor_sigma7bins200 = MaskedHessianHistogram( sigma = 7, n_bins = 200 )

hesshist_extractor_sigma3bins50 = MaskedHessianHistogram( sigma = 3, n_bins = 50 )
hesshist_extractor_sigma5bins50 = MaskedHessianHistogram( sigma = 5, n_bins = 50 )
hesshist_extractor_sigma10bins50 = MaskedHessianHistogram( sigma = 10, n_bins = 50 )
hesshist_extractor_sigma15bins50 = MaskedHessianHistogram( sigma = 15, n_bins = 50 )

hesshist_extractor_sigma10bins50pow05 = MaskedHessianHistogram( sigma = 10, n_bins = 50, power = 0.5 )
hesshist_extractor_sigma10bins50pow2 = MaskedHessianHistogram( sigma = 10, n_bins = 50, power = 2 )
hesshist_extractor_sigma10bins50pow5 = MaskedHessianHistogram( sigma = 10, n_bins = 50, power = 5 )
hesshist_extractor_sigma10bins50pow10 = MaskedHessianHistogram( sigma = 10, n_bins = 50, power = 10 )

hesshist_extractor_sigma5bins50pow05 = MaskedHessianHistogram( sigma = 5, n_bins = 50, power = 0.5 )
hesshist_extractor_sigma5bins50pow2 = MaskedHessianHistogram( sigma = 5, n_bins = 50, power = 2 )
hesshist_extractor_sigma5bins50pow5 = MaskedHessianHistogram( sigma = 5, n_bins = 50, power = 5 )
hesshist_extractor_sigma5bins50pow10 = MaskedHessianHistogram( sigma = 5, n_bins = 50, power = 10 )
