#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ..algorithms import MiuraMatch
from ..algorithms.MiuraMatchAligned import MiuraMatchAligned
from ..algorithms.HistogramsMatch import HistogramsMatch

huangwl = MiuraMatch(ch=18, cw=28)
miuramax = MiuraMatch(ch=80, cw=90)
miurarlt = MiuraMatch(ch=65, cw=55)

miura_wrist_20 = MiuraMatch( ch = 20, cw = 20 )
miura_wrist_40 = MiuraMatch( ch = 40, cw = 40 )
miura_wrist_60 = MiuraMatch( ch = 60, cw = 60 )
miura_wrist_80 = MiuraMatch( ch = 80, cw = 80 )
miura_wrist_100 = MiuraMatch( ch = 100, cw = 100 )
miura_wrist_120 = MiuraMatch( ch = 120, cw = 120 )
miura_wrist_140 = MiuraMatch( ch = 140, cw = 140 )
miura_wrist_160 = MiuraMatch( ch = 160, cw = 160 )

miura_wrist_aligned_20 = MiuraMatchAligned( ch = 20, cw = 20, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_40 = MiuraMatchAligned( ch = 40, cw = 40, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_60 = MiuraMatchAligned( ch = 60, cw = 60, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_80 = MiuraMatchAligned( ch = 80, cw = 80, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_100 = MiuraMatchAligned( ch = 100, cw = 100, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_120 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_140 = MiuraMatchAligned( ch = 140, cw = 140, alignment_flag = True, alignment_method = "center_of_mass" )
miura_wrist_aligned_160 = MiuraMatchAligned( ch = 160, cw = 160, alignment_flag = True, alignment_method = "center_of_mass" )

miura_wrist_dilation_5 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 5 )
miura_wrist_dilation_7 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 7 )
miura_wrist_dilation_9 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 9 )
miura_wrist_dilation_11 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 11 )
miura_wrist_dilation_13 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 13 )
miura_wrist_dilation_15 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 15 )
miura_wrist_dilation_17 = MiuraMatchAligned( ch = 120, cw = 120, alignment_flag = False, alignment_method = "center_of_mass", dilation_flag = True, ellipse_mask_size = 17 )

chi_square = HistogramsMatch( similarity_metrics_name = "chi_square" )
