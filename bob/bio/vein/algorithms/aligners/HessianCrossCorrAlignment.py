#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

import numpy as np

from skimage.feature import register_translation

from . import AlignerBase

import bob.io.base

#==============================================================================
# Main body of the class:

class HessianCrossCorrAlignment( AlignerBase ):
    """
    This class is designed to find the translation between two "Hessian" images.
    
    The following data can be used in the alignment process (specified by ``data_name_to_align``):
        
        1. image of magnitudes of largest eigenvectors of hessian matrices. 
           The images to be aligned are masked and mean normalized before the alignment.
           It is also possible to rise the magnitudes to the power ``align_power`` before alignment.
        2. image of orientations of largest eigenvectors of hessian matrices.
    
    **Parameters:**
    
    ``align_power`` : :py:class:`float`
        Allign the elements of an image of magnitudes to the power ``align_power`` before alignment.
    
    ``data_name_to_align`` : :py:class:`list` of :py:class:`str`
        The name of the data to use in the alignment routines.
        Possible options are specified in the ``available_data_names`` list, which is the memeber of this class.
        Default: "eigenvectors_magnitude".
    """

    def __init__( self, align_power, data_name_to_align = "eigenvectors_magnitude" ):
        
        AlignerBase.__init__( self )
        
        self.align_power = align_power # raise eigenvalues to the power of "align_power" before alignment
        self.data_name_to_align = data_name_to_align # the name of the data to be used in the alignment routines
        self.available_data_names = ["eigenvectors_magnitude", "eigenvectors_angles"]
        
        
    #==========================================================================
    def unroll_data( self, enroll, probe ):
        """
        Unroll the data in the enroll and probe lists.
        The output of this method is used in the :py:meth:`get_transformation_matrix` method of this class.
        
        **Parameters:**
        
        ``enroll`` : :py:class:`list`
            A list of arrays representing the enroll data.
        
        ``probe`` : :py:class:`list`
            A list of arrays representing the probe data.
        
        **Returns:**
        
        ``eigenvalues_enroll`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Enroll image of magnitudes of largest eigenvectors of hessian matrices.
        
        ``angles_enroll`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Enroll image of orientations of largest eigenvectors of hessian matrices.
        
        ``mask_enroll`` : 2D :py:class:`numpy.ndarray` of type :py:class:`uint8`
            Binary mask of the ROI for the enroll image.
        
        ``eigenvalues_probe`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Probe image of magnitudes of largest eigenvectors of hessian matrices.
        
        ``angles_probe`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Probe image of orientations of largest eigenvectors of hessian matrices.
        
        ``mask_probe`` : 2D :py:class:`numpy.ndarray` of type :py:class:`uint8`
            Binary mask of the ROI for the probe image.
        """
	
        if ( len( enroll ) == 3 ) and ( len( probe ) == 3 ):
            eigenvalues_enroll = enroll[0]
            angles_enroll = enroll[1]
            mask_enroll = enroll[2]
            
            eigenvalues_probe = probe[0]
            angles_probe = probe[1]
            mask_probe = probe[2]
        else:
            raise Exception("Enroll and probe features must be a list of lenght 3")
        
        return ( eigenvalues_enroll, angles_enroll, mask_enroll, 
                eigenvalues_probe, angles_probe, mask_probe )
        
        
    #==========================================================================
    def get_transformation_matrix( self, enroll, probe ):
        """
        Get transformation matrix containing the shift of the probe to register it with enroll.
        
        **Parameters:**
        
        ``enroll`` : :py:class:`list`
            A list of arrays representing the enroll data.
            The following data is stored in the list: [``eigenvalues_enroll``, ``angles_enroll``, ``mask_enroll``].
            This data is unrolled using the :py:meth:`unroll_data` method of this class.
        
        ``probe`` : :py:class:`list`
            A list of arrays representing the probe data.
            The following data is stored in the list: [``eigenvalues_probe``, ``angles_probe``, ``mask_probe``].
            This data is unrolled using the :py:meth:`unroll_data` method of this class.
        
        **Returns:**
        
        ``M`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Transformation matrix of the size (3, 3).
            Example: if (dx>0 and dy>0) in M, then probe must be shifted (right and down) to allign it with enroll.
            Coordinates (x, y) of the pixels in probe will increase by (dx, dy) values.
        """
        
        ( eigenvalues_enroll, angles_enroll, mask_enroll, 
         eigenvalues_probe, angles_probe, mask_probe ) = self.unroll_data( enroll, probe )
        
        if not( self.data_name_to_align in self.available_data_names ):
            raise Exception("Specified data_name_to_align is not in the list of available_data_names")
        
        eigenvalues_enroll = eigenvalues_enroll ** self.align_power # raise eigenvalues to the power
        eigenvalues_probe = eigenvalues_probe ** self.align_power # raise eigenvalues to the power
        
        enroll_mean = np.sum( eigenvalues_enroll ) / np.sum( mask_enroll )
        probe_mean = np.sum( eigenvalues_probe ) / np.sum( mask_probe )
        
        image_enroll = ( eigenvalues_enroll - enroll_mean ) * mask_enroll
        image_probe = ( eigenvalues_probe - probe_mean ) * mask_probe
        
        shift = np.array( [ 0, 0 ] )
        
        if self.data_name_to_align == "eigenvectors_magnitude":
            shift, error, diffphase = register_translation( image_enroll, image_probe )
            
        elif self.data_name_to_align == "eigenvectors_angles":
            shift, error, diffphase = register_translation( angles_enroll, angles_probe )
        
        M = np.float32( [ [ 1, 0, shift[1] ], [ 0, 1, shift[0] ], [ 0, 0, 1 ] ] )
        
        return M


    #==========================================================================
    def read_probe( self, probe_file ):
        """
        Reads the probe feature from file.
        
        **Parameters:**
        
        ``probe_file`` : :py:class:`str`
            The file name to read from.
        
        **Returns:**
        
        ``probe`` : :py:class:`tuple`
            Probe tuple with the following structure: ( max_eigenvalues, angles, mask ).
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices raised to the specified power.
            angles - 2D `numpy.ndarray` containing the angles (in radians) of eigenvectors with maximum eigenvalues.
            mask - 2D `numpy.ndarray` containing the binary mask of the ROI.
        """
        
        f = bob.io.base.HDF5File( probe_file, 'r' )
        max_eigenvalues = f.read( 'max_eigenvalues' )
        angles = f.read( 'angles' )
        mask = f.read( 'mask' )
        del f
        
        probe = ( max_eigenvalues, angles, mask )
        
        return probe
        
        
    #==========================================================================
    def read_model( self, model_file ):
        """
        Loads the enrolled model from file.
        
        **Parameters:**
        
        ``model_file`` : :py:class:`str`
            The file name to read from.
        
        **Returns:**
        
        ``list_of_features`` : :py:class:`list` of :py:class:`tuple`
            List ``list_of_features`` contains the tuples ( max_eigenvalues, angles, mask ), where:
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices raised to the specified power.
            angles - 2D `numpy.ndarray` containing the angles (in radians) of eigenvectors with maximum eigenvalues.
            mask - 2D `numpy.ndarray` containing the binary mask of the ROI.
        """
        
        f = bob.io.base.HDF5File( model_file, 'r' )
        features = f.read( 'array' )
        del f
        
        list_of_features = []
        
        for feature in features:
            
            max_eigenvalues, angles, mask = np.vsplit( feature, 3 )
            list_of_features.append( ( max_eigenvalues, angles, mask ) )
        
        return list_of_features

