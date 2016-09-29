#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import MaskedLBPHistograms

lbp_extractor_n8r2 = MaskedLBPHistograms( neighbors = 8, radius = 2 )
lbp_extractor_n8r3 = MaskedLBPHistograms( neighbors = 8, radius = 3 )
lbp_extractor_n8r4 = MaskedLBPHistograms( neighbors = 8, radius = 4 )
lbp_extractor_n8r5 = MaskedLBPHistograms( neighbors = 8, radius = 5 )
lbp_extractor_n8r6 = MaskedLBPHistograms( neighbors = 8, radius = 6 )
lbp_extractor_n8r7 = MaskedLBPHistograms( neighbors = 8, radius = 7 )

mct_extractor_n8r2 = MaskedLBPHistograms( neighbors = 8, radius = 2, to_average = True, add_average_bit = True )
mct_extractor_n8r3 = MaskedLBPHistograms( neighbors = 8, radius = 3, to_average = True, add_average_bit = True )
mct_extractor_n8r4 = MaskedLBPHistograms( neighbors = 8, radius = 4, to_average = True, add_average_bit = True )
mct_extractor_n8r5 = MaskedLBPHistograms( neighbors = 8, radius = 5, to_average = True, add_average_bit = True )
mct_extractor_n8r6 = MaskedLBPHistograms( neighbors = 8, radius = 6, to_average = True, add_average_bit = True )
mct_extractor_n8r7 = MaskedLBPHistograms( neighbors = 8, radius = 7, to_average = True, add_average_bit = True )

# concatenated LBP and MCT histograms:
lbp_mct_extractor_n8r5 = MaskedLBPHistograms( neighbors = 8, radius = 5, to_average = True, add_average_bit = True, concatenate_lbp_mct = True )
