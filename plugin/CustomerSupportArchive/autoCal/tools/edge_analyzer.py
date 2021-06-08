"""
Class to facilitate edge analysis as measured by # of microwells from the active array perimeter.
For example, studying edge effects related to poor loading or flowcell attach glue issues.
Leveraged in edgeEffects, spatialRL, and chipDiagnostics plugins.
"""

from scipy import ndimage
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Bring in chipcal in case we need it for the initial mask.
import chipcal

# for 510/520/530 and 540/541, we may want to add in 50, 150, and 250 wells.
RINGS = [0,50,100,150,200,250,300,400,500,1000,1500,2000,2500,5000,7500,10000]

class EdgeAnalyzer:
    """ 
    Class for edge analysis of chip data.  Works for both full flowcell chips and multilane chips.
    Often requires chip calibration data (gain) in order to determine mask of active pixels.
    """
    def __init__( self , start_mask , is_multilane=False ):
        self.start_mask = start_mask
        self.is_multilane = is_multilane
        self.get_smaller_dim( )

    def iter_rings( self ):
        """ Helper code to iterate through all the possible 'rings' of pixels for this particular chip size."""
        strings = [ str(x) for x in self.chip_rings ] + ['center']
        return enumerate( strings )
    
    def get_smaller_dim( self ):
        """ returns the size of the smallest dimension, usually rows . . . """
        if self.is_multilane:
            mask = self.start_mask[:,::4]
        else:
            mask = self.start_mask
        min_dim  = np.argmin( mask.shape )
        min_size = mask.shape[ min_dim ]
        self.min_size = min_size
        return None
    
    def parse_chip( self ):
        """ Creates a series of masks all at once for edge to center analysis. """
        #rings      = [0,100,200,300,400,500,1000,1500,2000,2500,5000,7500]
        chip_rings = []
        for i,r in enumerate(RINGS):
            minimum_width = 1000
            if (self.min_size - 2*r) < minimum_width:
                chip_rings = RINGS[:i]
                break
            else:
                chip_rings = RINGS
            
        # Create integer mask array
        ring_mask = np.zeros( self.start_mask.shape , np.int8 )
        for i,diff in enumerate( np.diff(chip_rings) ):
            if i == 0:
                new_mask = self.start_mask
            new_mask , edge = self.get_edge_mask( new_mask , diff )
            ring_mask[ edge ] = (i+1)
            
        # At the end of the loop, the new center needs to be added as well
        ring_mask[ new_mask ] = (i+2)
        self.chip_rings       = chip_rings
        self.ring_mask        = ring_mask
        self.ring_pixel_count = [ (ring_mask == i).sum() for i in range( ring_mask.max() + 1 ) ]
        
        # Create xlabels
        self.xlabels = [ str(y) for y in self.chip_rings ] + ['center']
        
        return None
    
    def make_ringplot( self, data, ylabel=None, ylim=None, imgpath=None, operation=np.mean, downsample=False ):
        """ 
        Creates a visualization of data as a function of 'rings' from the edge of the active array.
        Different operations can be applied using the operation kwarg, such as np.std.
        Downsampling is available for the largest of chiptypes, where this image creation can take a bit of time.
        """
        vals = []
        x    = self.chip_rings[:-1] + np.diff( self.chip_rings ) / 2.
        for idx in range( 1 , self.ring_mask.max()+1 ):
            if downsample:
                vals.append( operation( data[ self.ring_mask==idx ][::10] ) )
            else:
                vals.append( operation( data[ self.ring_mask==idx ] ) )
            
        plt.figure(figsize=(12,4))
        xax = np.append( x , np.mean([self.chip_rings[-1],self.min_size/2]) )
        plt.plot( xax , vals , 'o-' )
        plt.xticks( self.chip_rings + [self.min_size/2] , self.xlabels , rotation=90 , fontsize=8 )
        
        if ylabel:
            plt.ylabel( ylabel )
        if ylim:
            plt.ylim( ylim )
            
        plt.xlabel( 'Pixels from Glue Line' )
        plt.tight_layout( )
        plt.grid()
        if imgpath is None:
            plt.show( )
        else:
            plt.savefig( imgpath )
        plt.close  ( )
        return (xax , vals)
    
    def plot_rings( self , imgpath=None ):
        """ Makes a spatial map of the rings of pixels used.  """
        data = self.ring_mask
        
        # Get array limits and prep for x,yticks
        r,c  = data.shape
        xdivs , ydivs = np.arange( 0 , (c+1) , c/4 ) , np.arange( 0 , (r+1) , r/4 )
        # Max plottable image is 600x800.  Downsample everything else
        if data.shape[0] > 600:
            yscale = data.shape[0]/600
        else:
            yscale = 1
        if data.shape[1] > 800:
            xscale = data.shape[1]/800
        else:
            xscale = 1
        extent = [ 0, data.shape[1], 0, data.shape[0] ]  # This might technically clip the outter pixels
            
        plt.figure ( )
        plt.imshow ( self.ring_mask[::yscale,::xscale] , origin='lower' , interpolation='nearest' , extent=extent )
        plt.title  ( 'Spatial Map of Edge Pixel Masks' )
        plt.xticks ( xdivs , [ str(x) for x in xdivs ] )
        plt.yticks ( ydivs , [ str(y) for y in ydivs ] )
        if imgpath is None:
            plt.show()
        else:
            plt.savefig( imgpath )
        plt.close()
        return None
    
    @classmethod
    def from_chip_gain( cls , chipcal_path , chiptype ):
        """ Returns an EdgeAnalyzer class based on chip calibration. """
        cc = chipcal.ChipCal( chipcal_path , chiptype )
        cc.load_gain     ( )
        cc.find_refpix   ( )
        cc.determine_lane( )
        
        edge = cls( cc.active , cc.is_multilane )
        return edge
    
    @staticmethod
    def get_edge_mask( mask , pixels ):
        """ Creates a new edge mask from mask from <pixels> pixels from the edge of the array. """
        center = ndimage.morphology.binary_erosion( mask , iterations=pixels )
        edge   = np.logical_xor( mask, center ) # was [edge = mask - center] but bool '-' is now deprecated in np
        return center, edge

