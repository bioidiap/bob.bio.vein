# Available aligners:
from ...algorithms.aligners import HessianCrossCorrAlignment
# Available transformers:
from ...algorithms.transformers import ShiftEnrollProbeMasked
# Available feature extractors:
from ...algorithms.extractors import HessianHistMasked
from ...algorithms.extractors import SpatEnhancHessianHistMasked
from ...algorithms.extractors import SpatEnhancLBPHistMasked
# Available matching algorithms:
from ...algorithms.algorithms import HistogramsMatching
# main matching algorithm class:
from ...algorithms import AlignedMatching

import itertools as it


#==============================================================================
def combinations( input_dict ):
    """
    Obtain all possible key-value combinations in the input dictionary.
    
    Arguments:
    input_dict - input dictionary
    
    Return:
    List of dictionaries containing the combinations.
    """
    
    varNames = sorted( input_dict )
    
    combinations = [ dict( zip( varNames, prod ) ) for prod in it.product( *( input_dict[ varName ] for varName in varNames ) ) ]    
    
    return combinations


"""
Configurations 1:
"""
    
#==============================================================================
def set_matcher1( data_name_to_align, n_bins, eigenval_power ):
    """
    Set the parameters of the matching algorithm.
    The following pipeline is set with this function:
    
    1. aligner = HessianCrossCorrAlignment
    2. extractor = HessianHistMasked
    3. algorithm = HistogramsMatching
    4. matcher = AlignedMatching
    """
    # Set up the aligner:
    align_power = 1 # we keep this parameter fixed:
    aligner = HessianCrossCorrAlignment( align_power = align_power, data_name_to_align = data_name_to_align )
    #==============================================================================
    # Set up the transformer:
    transformer = ShiftEnrollProbeMasked()
    #==============================================================================
    # Set up the extractor:
    extractor = HessianHistMasked( n_bins = n_bins, eigenval_power = eigenval_power )
    #==============================================================================
    # Set up the algorithm:
    algorithm = HistogramsMatching()
    #==============================================================================
    # Set up matching algorithm:
    matcher = AlignedMatching( aligner, transformer, extractor, algorithm )
    
    return matcher

#==============================================================================
# Define parameters we want to test here:
params_dict = {}
params_dict["data_name_to_align"] = ['eigenvectors_magnitude', 'eigenvectors_angles']
params_dict["n_bins"] = [25, 50, 100, 200]
params_dict["eigenval_power"] = [1,2,3]
possible_combinations = combinations( params_dict )

matcher_dict = {}

for idx, item in enumerate( possible_combinations ):
    
    matcher_name = 'matcher_align_{}_hes_hist_bin{}pow{}'.format( item['data_name_to_align'][13:17], 
                    item['n_bins'], item['eigenval_power'] )
    
    matcher_dict[ matcher_name ] = set_matcher1( **item )


## Use this to print the bottom text:
#for item in matcher_dict.keys():
#    print item + ' = matcher_dict[ "{}" ]'.format( item ) 

# The list of algorithms we want to test:
matcher_align_magn_hes_hist_bin50pow1 = matcher_dict[ "matcher_align_magn_hes_hist_bin50pow1" ]
matcher_align_magn_hes_hist_bin50pow2 = matcher_dict[ "matcher_align_magn_hes_hist_bin50pow2" ]
matcher_align_magn_hes_hist_bin50pow3 = matcher_dict[ "matcher_align_magn_hes_hist_bin50pow3" ]
matcher_align_angl_hes_hist_bin50pow3 = matcher_dict[ "matcher_align_angl_hes_hist_bin50pow3" ]
matcher_align_angl_hes_hist_bin50pow2 = matcher_dict[ "matcher_align_angl_hes_hist_bin50pow2" ]
matcher_align_angl_hes_hist_bin50pow1 = matcher_dict[ "matcher_align_angl_hes_hist_bin50pow1" ]
matcher_align_angl_hes_hist_bin100pow1 = matcher_dict[ "matcher_align_angl_hes_hist_bin100pow1" ]
matcher_align_angl_hes_hist_bin100pow2 = matcher_dict[ "matcher_align_angl_hes_hist_bin100pow2" ]
matcher_align_angl_hes_hist_bin100pow3 = matcher_dict[ "matcher_align_angl_hes_hist_bin100pow3" ]
matcher_align_magn_hes_hist_bin25pow1 = matcher_dict[ "matcher_align_magn_hes_hist_bin25pow1" ]
matcher_align_magn_hes_hist_bin100pow1 = matcher_dict[ "matcher_align_magn_hes_hist_bin100pow1" ]
matcher_align_angl_hes_hist_bin25pow2 = matcher_dict[ "matcher_align_angl_hes_hist_bin25pow2" ]
matcher_align_magn_hes_hist_bin200pow1 = matcher_dict[ "matcher_align_magn_hes_hist_bin200pow1" ]
matcher_align_angl_hes_hist_bin25pow3 = matcher_dict[ "matcher_align_angl_hes_hist_bin25pow3" ]
matcher_align_magn_hes_hist_bin100pow3 = matcher_dict[ "matcher_align_magn_hes_hist_bin100pow3" ]
matcher_align_angl_hes_hist_bin25pow1 = matcher_dict[ "matcher_align_angl_hes_hist_bin25pow1" ]
matcher_align_magn_hes_hist_bin100pow2 = matcher_dict[ "matcher_align_magn_hes_hist_bin100pow2" ]
matcher_align_angl_hes_hist_bin200pow3 = matcher_dict[ "matcher_align_angl_hes_hist_bin200pow3" ]
matcher_align_angl_hes_hist_bin200pow2 = matcher_dict[ "matcher_align_angl_hes_hist_bin200pow2" ]
matcher_align_angl_hes_hist_bin200pow1 = matcher_dict[ "matcher_align_angl_hes_hist_bin200pow1" ]
matcher_align_magn_hes_hist_bin200pow2 = matcher_dict[ "matcher_align_magn_hes_hist_bin200pow2" ]
matcher_align_magn_hes_hist_bin200pow3 = matcher_dict[ "matcher_align_magn_hes_hist_bin200pow3" ]
matcher_align_magn_hes_hist_bin25pow2 = matcher_dict[ "matcher_align_magn_hes_hist_bin25pow2" ]
matcher_align_magn_hes_hist_bin25pow3 = matcher_dict[ "matcher_align_magn_hes_hist_bin25pow3" ]


## Use this loop to generate the entry points for the setup.py
#for item in matcher_dict.keys():    
#    print "        '" + item[14:18] + '-align-hes-hist-' + item[-9:] + ' = bob.bio.vein.configurations.alignment_algorithms.algorithms:'+ item + "',"

## Use this loop for bash scripts purposes:
#alg_string = ""
#for item in matcher_dict.keys():
#    
#    alg_string = alg_string + ' "' + item[14:18] + '-align-hes-hist-' + item[-9:] + '" '

## Use this loop for bash scripts purposes:
#subdir_string = ""
#for item in matcher_dict.keys():
#    
#    subdir_string = subdir_string + ' "' + "kmce40_mea71_" + item[14:18] + "HesHist" + item[-9:] + '/" '



"""
Configurations 2:
"""
    
#==============================================================================
def set_matcher2( data_name_to_align, n_bins, eigenval_power ):
    """
    Set the parameters of the matching algorithm.
    The following pipeline is set with this function:
    
    1. aligner = HessianCrossCorrAlignment
    2. extractor = SpatEnhancHessianHistMasked
    3. algorithm = HistogramsMatching
    4. matcher = AlignedMatching
    """
    # Set up the aligner:
    align_power = 1 # we keep this parameter fixed:
    aligner = HessianCrossCorrAlignment( align_power = align_power, data_name_to_align = data_name_to_align )
    #==============================================================================
    # Set up the transformer:
    transformer = ShiftEnrollProbeMasked()
    #==============================================================================
    # Set up the extractor:
    extractor = SpatEnhancHessianHistMasked( n_bins = n_bins, eigenval_power = eigenval_power )
    #==============================================================================
    # Set up the algorithm:
    algorithm = HistogramsMatching()
    #==============================================================================
    # Set up matching algorithm:
    matcher = AlignedMatching( aligner, transformer, extractor, algorithm )
    
    return matcher

#==============================================================================
# Define parameters we want to test here:
params_dict = {}
params_dict["data_name_to_align"] = [ 'eigenvectors_magnitude', 'eigenvectors_angles' ]
params_dict["n_bins"] = [ 25, 50, 100 ]
params_dict["eigenval_power"] = [ 1, 2 ]
possible_combinations = combinations( params_dict )

matcher_dict = {}

for idx, item in enumerate( possible_combinations ):
    
    matcher_name = 'matcher_align_{}_spat_enh_hes_hist_bin{}pow{}'.format( item['data_name_to_align'][13:17], 
                    item['n_bins'], item['eigenval_power'] )
    
    matcher_dict[ matcher_name ] = set_matcher2( **item )


## Use this to print the bottom text:
#for item in matcher_dict.keys():
#    print item + ' = matcher_dict[ "{}" ]'.format( item ) 


matcher_align_magn_spat_enh_hes_hist_bin25pow2 = matcher_dict[ "matcher_align_magn_spat_enh_hes_hist_bin25pow2" ]
matcher_align_magn_spat_enh_hes_hist_bin25pow1 = matcher_dict[ "matcher_align_magn_spat_enh_hes_hist_bin25pow1" ]
matcher_align_magn_spat_enh_hes_hist_bin50pow2 = matcher_dict[ "matcher_align_magn_spat_enh_hes_hist_bin50pow2" ]
matcher_align_magn_spat_enh_hes_hist_bin100pow2 = matcher_dict[ "matcher_align_magn_spat_enh_hes_hist_bin100pow2" ]
matcher_align_magn_spat_enh_hes_hist_bin100pow1 = matcher_dict[ "matcher_align_magn_spat_enh_hes_hist_bin100pow1" ]
matcher_align_magn_spat_enh_hes_hist_bin50pow1 = matcher_dict[ "matcher_align_magn_spat_enh_hes_hist_bin50pow1" ]
matcher_align_angl_spat_enh_hes_hist_bin100pow1 = matcher_dict[ "matcher_align_angl_spat_enh_hes_hist_bin100pow1" ]
matcher_align_angl_spat_enh_hes_hist_bin100pow2 = matcher_dict[ "matcher_align_angl_spat_enh_hes_hist_bin100pow2" ]
matcher_align_angl_spat_enh_hes_hist_bin25pow2 = matcher_dict[ "matcher_align_angl_spat_enh_hes_hist_bin25pow2" ]
matcher_align_angl_spat_enh_hes_hist_bin25pow1 = matcher_dict[ "matcher_align_angl_spat_enh_hes_hist_bin25pow1" ]
matcher_align_angl_spat_enh_hes_hist_bin50pow2 = matcher_dict[ "matcher_align_angl_spat_enh_hes_hist_bin50pow2" ]
matcher_align_angl_spat_enh_hes_hist_bin50pow1 = matcher_dict[ "matcher_align_angl_spat_enh_hes_hist_bin50pow1" ]


## Use this loop to generate the entry points for the setup.py
#for item in matcher_dict.keys():    
#    print "        '" + item[14:18] + '-align-spat-enh-hes-hist-' + item[ item.find("bin"): ] + ' = bob.bio.vein.configurations.alignment_algorithms.algorithms:'+ item + "',"

## Use this loop for bash scripts purposes:
#alg_string = ""
#for item in matcher_dict.keys():
#    
#    alg_string = alg_string + ' "' + item[14:18] + '-align-spat-enh-hes-hist-' + item[ item.find("bin"): ] + '" '

## Use this loop for bash scripts purposes:
#subdir_string = ""
#for item in matcher_dict.keys():
#    
#    subdir_string = subdir_string + ' "' + "kmce40_mea71_" + item[14:18] + "SpatEnhHesHist" + item[ item.find("bin"): ] + '/" '



"""
Configurations 3:
"""
    
#==============================================================================
def set_matcher3( data_name_to_align, neighbors, radius ):
    """
    Set the parameters of the matching algorithm.
    The following pipeline is set with this function:
    
    1. aligner = HessianCrossCorrAlignment
    2. extractor = SpatEnhancLBPHistMasked
    3. algorithm = HistogramsMatching
    4. matcher = AlignedMatching
    """
    # Set up the aligner:
    align_power = 1 # we keep this parameter fixed:
    aligner = HessianCrossCorrAlignment( align_power = align_power, data_name_to_align = data_name_to_align )
    #==============================================================================
    # Set up the transformer:
    transformer = ShiftEnrollProbeMasked()
    #==============================================================================
    # Set up the extractor:
    to_average = True
    add_average_bit = False
    extractor = SpatEnhancLBPHistMasked( neighbors = neighbors, radius = radius, to_average = to_average, add_average_bit = add_average_bit )
    #==============================================================================
    # Set up the algorithm:
    algorithm = HistogramsMatching()
    #==============================================================================
    # Set up matching algorithm:
    matcher = AlignedMatching( aligner, transformer, extractor, algorithm )
    
    return matcher

#==============================================================================
# Define parameters we want to test here:
params_dict = {}
params_dict["data_name_to_align"] = [ 'eigenvectors_magnitude', 'eigenvectors_angles' ]
params_dict["neighbors"] = [ 4, 8 ]
params_dict["radius"] = [ 2, 3, 4, 5 ]
possible_combinations = combinations( params_dict )

matcher_dict = {}

for idx, item in enumerate( possible_combinations ):
    
    matcher_name = 'matcher_align_{}_spat_enh_lbp_hist_neigh{}rd{}'.format( item['data_name_to_align'][13:17], 
                    item['neighbors'], item['radius'] )
    
    matcher_dict[ matcher_name ] = set_matcher3( **item )


## Use this to print the bottom text:
#for item in matcher_dict.keys():
#    print item + ' = matcher_dict[ "{}" ]'.format( item ) 


matcher_align_magn_spat_enh_lbp_hist_neigh8rd5 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh8rd5" ]
matcher_align_magn_spat_enh_lbp_hist_neigh8rd4 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh8rd4" ]
matcher_align_magn_spat_enh_lbp_hist_neigh4rd3 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh4rd3" ]
matcher_align_magn_spat_enh_lbp_hist_neigh4rd2 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh4rd2" ]
matcher_align_magn_spat_enh_lbp_hist_neigh4rd5 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh4rd5" ]
matcher_align_magn_spat_enh_lbp_hist_neigh4rd4 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh4rd4" ]
matcher_align_magn_spat_enh_lbp_hist_neigh8rd3 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh8rd3" ]
matcher_align_magn_spat_enh_lbp_hist_neigh8rd2 = matcher_dict[ "matcher_align_magn_spat_enh_lbp_hist_neigh8rd2" ]
matcher_align_angl_spat_enh_lbp_hist_neigh4rd4 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh4rd4" ]
matcher_align_angl_spat_enh_lbp_hist_neigh4rd5 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh4rd5" ]
matcher_align_angl_spat_enh_lbp_hist_neigh8rd2 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh8rd2" ]
matcher_align_angl_spat_enh_lbp_hist_neigh8rd3 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh8rd3" ]
matcher_align_angl_spat_enh_lbp_hist_neigh8rd4 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh8rd4" ]
matcher_align_angl_spat_enh_lbp_hist_neigh8rd5 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh8rd5" ]
matcher_align_angl_spat_enh_lbp_hist_neigh4rd2 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh4rd2" ]
matcher_align_angl_spat_enh_lbp_hist_neigh4rd3 = matcher_dict[ "matcher_align_angl_spat_enh_lbp_hist_neigh4rd3" ]


## Use this loop to generate the entry points for the setup.py
#for item in matcher_dict.keys():    
#    print "        '" + item[14:18] + '-align-spat-enh-lbp-hist-' + item[ item.find("neigh"): ] + ' = bob.bio.vein.configurations.alignment_algorithms.algorithms:'+ item + "',"

# Use this loop for bash scripts purposes:
#alg_string = ""
#for item in matcher_dict.keys():
#    
#    alg_string = alg_string + ' "' + item[14:18] + '-align-spat-enh-lbp-hist-' + item[ item.find("neigh"): ] + '" '
#
## Use this loop for bash scripts purposes:
#subdir_string = ""
#for item in matcher_dict.keys():
#    
#    subdir_string = subdir_string + ' "' + "kmce40_mea51_" + item[14:18] + "SpatEnhLBPHist" + item[ item.find("neigh"): ] + '/" '





















