#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:06:06 2016

@author: onikisins
"""

from bob.bio.base.algorithm import Algorithm

class AlignerBase( Algorithm ):
    """
    Base class for the alignment algorithms used in the AlignedMatching class.
    """
    
    def __init__( self ):
        
        Algorithm.__init__( self )
    
    #==========================================================================
    def unroll_data( self, enroll, probe ):
        """
        unroll_data(model, probe) -> tuple of objects used for alignment in :py:meth:`get_transformation_matrix`
        
        This function will unroll the data from the enroll and probe containers (usually lists or tuples) 
        into a tuple with arguments.
        The output of this method is used in the :py:meth:`get_transformation_matrix` method of the derived class.
        
        **Parameters:**
        
        ``enroll`` : object
            A container with data (usually a tuple or a list) representing the enroll.
        
        ``probe`` : object
            A container with data (usually a tuple or a list) representing the probe.
            
        **Returns:**
        
        ``returned_tuple`` : :py:class:`tuple`
            A tuple of objects/arguments used in the :py:meth:`get_transformation_matrix` method of the derived class.
        """
        
        raise NotImplementedError("This function must be overwritten in the derived classes.")
        
    #==========================================================================
    def get_transformation_matrix( self, enroll, probe ):
        """
        Get transformation matrix for registration of the probe to the enroll.
        
        **Parameters:**
        
        ``enroll`` : object
            A container with data (usually a tuple or a list) representing the enroll.
            This data is unrolled using the :py:meth:`unroll_data` method of the derived class.
        
        ``probe`` : object
            A container with data (usually a tuple or a list) representing the probe.
            This data is unrolled using the :py:meth:`unroll_data` method of the derived class.
        
        **Returns:**
        
        ``M`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Transformation matrix of the size (2, 3).
        """
        
        raise NotImplementedError("This function must be overwritten in the derived classes.")
        
        
    #==========================================================================
    def read_probe( self, probe_file ):
        """
        read_probe(probe_file) -> probe
        
        Reads the probe feature from file.
        By default, the probe feature is identical to the projected feature.
        Hence, this base class implementation simply calls :py:meth:`read_feature`.
        
        If your algorithm requires different behavior, please overwrite this function
        in the class derived from base class.
        
        **Parameters:**
        
        ``probe_file`` : :py:class:`str` or :py:class:`bob.io.base.HDF5File`
            The file open for reading, or the file name to read from.
        
        **Returns:**
        
        ``probe`` : object
            The probe that was read from file.
        """
                
        probe = super( AlignerBase, self ).read_probe( probe_file ) # call the read_probe() of the Algorithm
        
        return probe


    #==========================================================================
    def read_model( self, model_file ):
        """
        read_model(model_file) -> model
        
        Loads the enrolled model from file.
        In this base class implementation, it uses :py:func:`bob.io.base.load` to do that.
        
        If your algorithm requires different behavior, please overwrite this function
        in the class derived from base class.
        
        **Parameters:**
        
        ``model_file`` : :py:class:`str` or :py:class:`bob.io.base.HDF5File`
            The file open for reading, or the file name to read from.
        
        **Returns:**
        
        ``model`` : object
            The model that was read from file.
        """
        
        model = super( AlignerBase, self ).read_model( model_file ) # call the read_model() of the Algorithm
        
        return model

