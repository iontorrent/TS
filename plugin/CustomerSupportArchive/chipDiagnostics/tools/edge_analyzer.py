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
try:
    import chipcal
except ImportError:
    try:
        from tools import chipcal
    except ImportError:
        print( 'Did not import chipcal in tools.edge_analyzer' )

# for 510/520/530 and 540/541, we may want to add in 50, 150, and 250 wells.
RINGS               = [0,50,100,150,200,250,300,400,500,1000,1500,2000,2500,5000,7500,10000]
IMAGE_RINGS         = [0,2,4,6,8,10,13,16,32,50] # each superpixel is ~31px for GX5v2
SMALL_IMAGE_RINGS   = [0,1,2,3,4,5,6,7,8,10,13,16,32,50] 
RING        = 'ring'
RECTANGLE   = 'rectangle'

class EdgeAnalyzer:
    """ 
    Class for edge analysis of chip data.  Works for both full flowcell chips and multilane chips.
    Often requires chip calibration data (gain) in order to determine mask of active pixels.
    """
    def __init__( self , start_mask , is_multilane=False, rings=RINGS ):
        # pass in RINGS to make it a changeable attribute in the class instance
        self.RINGS = rings
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
    
    def parse_chip( self, mask_shape=RING, minimum_width=1000 ):
        """ Creates a series of masks all at once for edge to center analysis. """
        #rings      = [0,100,200,300,400,500,1000,1500,2000,2500,5000,7500]
        chip_rings = []
        for i,r in enumerate(self.RINGS):
            minimum_width = minimum_width
            if (self.min_size - 2*r) < minimum_width:
                chip_rings = self.RINGS[:i]
                break
            else:
                chip_rings = self.RINGS
        # Create integer mask array
        ring_mask = np.zeros( self.start_mask.shape , np.int8 )
        if mask_shape==RECTANGLE:
            for i,diff in enumerate( chip_rings[1:] ):
                if i == 0:
                    new_mask = self.start_mask
                    initial = True
                else: 
                    initial = False

                new_mask , edge = self.get_rectangle_edge_mask( new_mask , diff, initial=initial )

                ring_mask[ edge ] = (i+1)
        else:
            for i,diff in enumerate( np.diff(chip_rings) ):
                if i == 0:
                    # Start with a filled in active --> no holes from pinned
                    filled_new_mask = ndimage.morphology.binary_fill_holes( self.start_mask )

                # erode the filled in active --> symmetric masks with no holes
                filled_new_mask , filled_edge = self.get_edge_mask( filled_new_mask , diff )
                
                # AND with filled and start_mask --> put holes back in
                new_mask    = np.logical_and( filled_new_mask, self.start_mask )
                edge        = np.logical_and( filled_edge, self.start_mask )

                # proceed as usual
                ring_mask[ edge ] = (i+1)

        # At the end of the loop, the new center needs to be added as well
        ring_mask[ new_mask ] = (i+2)
        self.chip_rings       = chip_rings
        self.ring_mask        = ring_mask
        self.ring_pixel_count = [ (ring_mask == i).sum() for i in range( len(chip_rings) + 1 ) ]
        
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
        for idx in range( 1 , len(x)+2 ):
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
            yscale = int( data.shape[0]/600 )
        else:
            yscale = 1
        if data.shape[1] > 800:
            xscale = int( data.shape[1]/800 )
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
    
    def get_rectangle_edge_mask( self, mask, pixels, initial=False ):
        # Generate a rectangular array
        if initial:
            # if this attribute doesn't exist, it's the first time through
            # --> generate start_edge_left and start_edge_right
            test = np.array( mask ).astype( np.int )
            test_mean = np.mean( test, axis=0 )
            # only look for edges in the outer 1/8ths of the array
            cutoff_edge = np.int( np.float(test_mean.shape[0])/8. )
            for i, x in enumerate( test_mean ):
                if x>0 and i<cutoff_edge:
                    self.start_edge_left = i
                    break
                elif i>=cutoff_edge:
                    self.start_edge_left=None
                    break
            for i, x in enumerate( np.flipud(test_mean) ):
                if x>0 and i<cutoff_edge:
                    self.start_edge_right = test_mean.shape[0] - i
                    break
                elif i>=cutoff_edge:
                    self.start_edge_right=None
            # get rid of inlet/outlet
            trim = np.int( np.float( test.shape[0] )/8. )
            temp = np.zeros( mask.shape ).astype( np.bool )
            temp[trim:-trim] = mask[trim:-trim]
            mask = temp
        center = np.zeros( mask.shape ).astype( np.bool )
        
        if self.start_edge_left:    left_lim = self.start_edge_left + pixels
        else:                       left_lim = 0

        if self.start_edge_right:   right_lim = self.start_edge_right - pixels
        else:                       right_lim = mask.shape[1]-1

        center[:, left_lim:right_lim] = mask[:, left_lim:right_lim]
        edge   = np.logical_xor( mask, center )

        return center, edge

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

