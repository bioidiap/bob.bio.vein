#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

import numpy as np

#from bob.bio.base.algorithm import Algorithm

from scipy import ndimage

from bob.ip.base import integral

import itertools

#from numpy import unravel_index == np.unravel_index

#==============================================================================
# Main body of the class:

class HessianAlignment: # Algorithm ):
    """
    
    """


    def __init__( self, window_size, N_points, gap, step, align_power, enroll_center_method = "center_of_mass" ):

#        Algorithm.__init__( self,
#                           window_size = window_size,
#                           N_points = N_points,
#                           gap = gap,
#                           step = step,
#                           align_power = align_power,
#                           enroll_center_method = enroll_center_method )

        self.window_size = window_size # size of the square sliding window to sum the elements in
        self.N_points = N_points # number of points in the grid
        self.gap = gap # gap in pixels between the nodes in the grid
        self.step = step # alignment step in pixels. Probe is shifted relative to enroll with this step.
        self.align_power = align_power # raise eigenvalues to the power of "align_power" before alignment
        self.enroll_center_method = enroll_center_method # the name of the method to find the point in the enroll to allign the probe to
        self.available_enroll_center_methods = [ "center_of_mass", "largest_vector_magnitude" ]
        
#        self.similarity_metrics_name = similarity_metrics_name
#        self.available_similarity_metrics = [ "chi_square", "histogram_intersection" ]


    #==========================================================================
    def window_sum( self, integral_image, window_size ):
        """
        
        """
        
        row_max, col_max = integral_image.shape
        sum_array = np.zeros( ( row_max - window_size, col_max - window_size ) )
        
        for row in range(row_max - window_size):    
            for col in range(col_max - window_size):
                
                A = integral_image[ row, col ]
                B = integral_image[ row, col + window_size ]
                C = integral_image[ row + window_size, col ]
                D = integral_image[ row + window_size, col + window_size ]
                
                sum_array[ row, col ] = D - B - C + A
        
        return sum_array


    #==========================================================================
    def __fit_coords_to_image_dimensions__( self, coord_vec, image ):
        """
        
        """
        
        coord_vec = coord_vec - np.min( [ 0, np.min( coord_vec ) ] ) # avoid negative coordinates
        
        if np.max( coord_vec ) >= np.min( image.shape ): # avoid coordinates outside of image boundaries
            
            coord_vec = coord_vec - ( np.max( coord_vec ) - np.min( image.shape ) + 1 )
            
        return np.uint( coord_vec )


    #==========================================================================
    def find_relative_probe_shift( self, eigenvalues_enroll, angles_enroll, mask_enroll, eigenvalues_probe, angles_probe, mask_probe, 
                                  window_size,
                                  N_points,
                                  gap,
                                  step,
                                  align_power,
                                  enroll_center_method ):
        """
        
        """
        
        eigenvalues_enroll = eigenvalues_enroll ** align_power # raise eigenvalues to the power
        eigenvalues_probe = eigenvalues_probe ** align_power # raise eigenvalues to the power
        
        dx_enroll = eigenvalues_enroll * np.cos( angles_enroll ) # dx component of the vectors
        dy_enroll = eigenvalues_enroll * np.sin( angles_enroll ) # dy components of the vectors
        
        dx_probe = eigenvalues_probe * np.cos( angles_probe ) # dx component of the vectors
        dy_probe = eigenvalues_probe * np.sin( angles_probe ) # dy components of the vectors
        
        # Compute 4 integral images:
        dx_enroll_integral = np.zeros(dx_enroll.shape)
        integral( dx_enroll, dx_enroll_integral)
        dy_enroll_integral = np.zeros(dy_enroll.shape)
        integral( dy_enroll, dy_enroll_integral)
        
        dx_probe_integral = np.zeros(dx_probe.shape)
        integral( dx_probe, dx_probe_integral)
        dy_probe_integral = np.zeros(dy_probe.shape)
        integral( dy_probe, dy_probe_integral)
        
        # Sum the elements in the sliding window of the size: (window_size) x (window_size) pixels
        dx_enroll_sum = self.window_sum( dx_enroll_integral, window_size )
        dy_enroll_sum = self.window_sum( dy_enroll_integral, window_size )
        
        dx_probe_sum = self.window_sum( dx_probe_integral, window_size )
        dy_probe_sum = self.window_sum( dy_probe_integral, window_size )
        
        # The point in the enroll to allign the probe to:
        if enroll_center_method == "center_of_mass":
            center_row, center_col = np.round( ndimage.measurements.center_of_mass( mask_enroll ) )
            
        elif enroll_center_method == "largest_vector_magnitude":
            vectors_sum_magn = ( dx_enroll_sum ** 2 + dy_enroll_sum ** 2 ) ** 0.5
            center_row, center_col = np.unravel_index(vectors_sum_magn.argmax(), vectors_sum_magn.shape)
            center_row = center_row + window_size/2
            center_col = center_col + window_size/2
            
        else:
            raise Exception("Specified enroll_center_method method is not implemented")
        
        if gap*N_points >= np.min( dx_enroll_sum.shape ):
            raise Exception("Grid is larger than the array produced by `window_sum` function. Reduce `N_points` or `gap`.")
        
        # Coordinates of the nodes of the grid in the enroll image:
        row_grid_enroll = ( center_row - N_points/2 * gap + range( 0, gap*N_points, gap ) )
        row_grid_enroll = self.__fit_coords_to_image_dimensions__( row_grid_enroll, dx_enroll_sum ) # update coordinates if needed
        
        col_grid_enroll = ( center_col - N_points/2 * gap + range( 0, gap*N_points, gap ) )
        col_grid_enroll = self.__fit_coords_to_image_dimensions__( col_grid_enroll, dx_enroll_sum ) # update coordinates if needed
        
        row_col_grid_enroll = np.array( list( itertools.product( row_grid_enroll, col_grid_enroll ) ) ) # all (row, col) coordinates of the nodes
        
        # Select vector projections located in the positions of the nodes of the grid:
        dx_enroll_sum_selected = np.array( [ dx_enroll_sum[ tuple(i) ] for i in row_col_grid_enroll ] )
        dy_enroll_sum_selected = np.array( [ dy_enroll_sum[ tuple(i) ] for i in row_col_grid_enroll ] )
        
        
        row_grid_probe = np.uint( range(0, gap*N_points, gap) )
        col_grid_probe = np.uint( range(0, gap*N_points, gap) )
        
        row_col_grid_probe = np.array( list( itertools.product( row_grid_probe, col_grid_probe ) ) )
        
        row_max, col_max = np.uint( dx_probe_sum.shape )
        
        row_max_grid = np.uint( np.max( row_grid_probe ) )
        col_max_grid = np.uint( np.max( col_grid_probe ) )
        
        result_dot_products = np.zeros( np.uint( ( row_max - row_max_grid + 1, col_max - col_max_grid + 1 ) ) )
        
        for row_offset in range(0, row_max - row_max_grid, step):
            for col_offset in range(0, col_max - col_max_grid, step):
                
                row_col_grid_probe_updated = row_col_grid_probe + np.uint( [ row_offset, col_offset ] )
                
                dx_probe_sum_selected = np.array( [ dx_probe_sum[ tuple(i) ] for i in row_col_grid_probe_updated ] )
                dy_probe_sum_selected = np.array( [ dy_probe_sum[ tuple(i) ] for i in row_col_grid_probe_updated ] )
                
                dot_product = np.sum( dx_enroll_sum_selected * dx_probe_sum_selected + dy_enroll_sum_selected * dy_probe_sum_selected )
                
                result_dot_products[ row_offset, col_offset ] = dot_product
        
        # Find coordinates of the maximum in the array of dot products:
        coords_max_dot_product = np.unravel_index(result_dot_products.argmax(), result_dot_products.shape)
        
        relative_probe_shift = ( coords_max_dot_product[0] - np.min( row_grid_enroll ), coords_max_dot_product[1] - np.min( col_grid_enroll ) )
        
        return ( np.int( relative_probe_shift[0] ), np.int( relative_probe_shift[1] ) )
        
        
        
        
    #==========================================================================
    def get_transformation_matrix( self, eigenvalues_enroll, angles_enroll, mask_enroll, eigenvalues_probe, angles_probe, mask_probe ):
        """
        
        """
        
        relative_probe_shift = self.find_relative_probe_shift( eigenvalues_enroll, angles_enroll, mask_enroll, 
                                                              eigenvalues_probe, angles_probe, mask_probe, 
                                                              self.window_size,
                                                              self.N_points,
                                                              self.gap,
                                                              self.step,
                                                              self.align_power,
                                                              self.enroll_center_method )
        
        M = np.float32( [ [ 1, 0, -relative_probe_shift[1] ], [ 0, 1, -relative_probe_shift[0] ] ] )
        
        return M




