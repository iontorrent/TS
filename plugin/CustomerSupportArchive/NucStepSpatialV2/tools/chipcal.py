"""
Module for interaction, analysis, and plotting of chip calibration files for all chiptypes.
"""

import argparse
import sys, os, time, json
import numpy as np
import numpy.ma as ma
from scipy import ndimage
import matplotlib
matplotlib.use('Agg',warn=False)
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Set default cmap to jet, which is viridis on newer TS software (matplotlib 2.1)
matplotlib.rcParams['image.cmap'] = 'jet'

try:
    from . import stats, misc
    from . import chiptype as ct
    from . import datprops as dp
    from . import chip
    from . import imtools
    from . import lanes as mltools
except ( ImportError, ValueError ):
    # for a long time, chipcal lived outside of tools but was heavily dependent
    # Many things still access chipcal via symlink
    # These import statements break in that case
    #from warnings import warn
    #warn( 'chipcal has been moved inside of tools.  Please update your code to reflect the new location' )
    from tools import stats, misc
    from tools import chiptype as ct
    from tools import datprops as dp
    from tools import chip
    from tools import imtools
    from tools import lanes as mltools

BlockMath = stats.BlockMath

moduleDir = os.path.abspath( os.path.dirname( __file__ ) )
np.seterr( all='ignore' )
allow_missing = True

class ChipCal:
    """ 
    This is a class to deal with calibration files.
    
    Useful tips:
    - For efficiency, call high level superpixel_analysis functions to calculate these BlockMath functions only once
    - For 560 chips and above, it may work best to use chunked BlockMath analysis by calling the enable_super_RC method
    - superpixel analysis arrays (localavg, localstd) will be stored as attributes, e.g.  self.gain_localstd
    - HD superpixel analysis arrays will have an appended '_hd', e.g. self.noise_localavg_hd
    """
    def __init__( self , path , chiptype , outdir='' , verbose=False, force=False, dr=None, no_fc_mask=False ):
        """
        path - directory that contains the calibration data files.
        chiptype - a string or an object of chiptype class
        outdir - path to where images or metrics should be saved.

        Default behavior is to set no_fc_mask=False for historical reasons.
        ECC and WetTest set this to True in their function calls
        """
        if path == '.':
            self.caldir = os.getcwd()
        else:
            self.caldir = path

        # If outdir = '', then save files in the path given to cal files.
        if outdir == '' or not os.path.exists( outdir ):
            self.output_dir = self.caldir
        else:
            self.output_dir = outdir

        self.dr = dr

        self.no_fc_mask = no_fc_mask

        self.chiptype  = ct.ChipType( chiptype, blockdir=path )
        self.metrics    = {}
        
        self.logfile = os.path.join( self.output_dir , 'cal.log' )
        
        self.force = force
        
        # This isn't actually necessary right now...also PGM isn't set up to work yet.
        self.rows = self.chiptype.chipR
        self.cols = self.chiptype.chipC
        #self.blockR = self.chiptype.blockR
        #self.blockC = self.chiptype.blockC
        self.blockR = self.chiptype.miniR
        self.blockC = self.chiptype.miniC
            
        if self.chiptype.series == 'proton':
            #self.noise_lims  = ( 0 , 600 )
            self.noise_lims  = ( 0 , 1000 )  # For P2
            self.offset_lims = ( 0 , 400 )
            self.gain_lims   = ( 500 , 1600 )
            
            # Difference limits used in diff_plots
            self.noise_diff_lims  = (  -50 ,  50 )
            self.offset_diff_lims = (  -50 ,  50 )
            self.gain_diff_lims   = ( -100 , 100 )
            
        if self.chiptype.series == 'pgm':
            self.noise_lims  = ( 0 , 150 )
            self.offset_lims = ( 0 , 236 )
            self.gain_lims   = ( 550 , 750 )
            
            # Difference limits used in diff_plots
            self.noise_diff_lims  = (  -50 ,  50 )
            self.offset_diff_lims = (  -50 ,  50 )
            self.gain_diff_lims   = (  -50 ,  50 )
            
            
        # might want to have auto chiptype detection here . . . 
        self.verbose     = verbose
        self.noise_units = 'uV'
        self.gain_units  = 'mV/V'
        self.offset_units= 'mV'
        
        # Define multilane constants
        self.is_multilane = False
        self.lane_1       = False
        self.lane_2       = False
        self.lane_3       = False
        self.lane_4       = False
        self.lane_metrics = { 'lane_1':{} , 'lane_2':{} , 'lane_3':{} , 'lane_4':{} }
        
        # Adding attributes determining if we do superR and superC chunking of block math
        self.superR = None
        self.superC = None
        
    def calc_metrics_by_lane( self , metric ):
        """
        Calculates metrics by lane and stores in the self.lane_metrics dictionary.
        """
        if not self.is_multilane:
            print( 'Skipping multilane analysis, as this is not a multilane chip' )
            return None
        
        m = metric.lower()
        try:
            data = getattr( self , m )
            lims = self.select_lims( m )
        except AttributeError:
            print( 'Error, metric ({}) not found.'.format( m ) )
            return None
        
        # If we make it here, let's iterate through lanes
        for i, lane, cs in self.iter_lane_slice( data ):
            lane_data = data[:,cs]
            if hasattr( self , 'active' ) and m in ['noise','offset','gain']:
                mask = self.active[:,cs]
            else:
                mask = None
            
            # Get statistics
            self.lane_metrics[lane][m] = self.calc_metrics_from_array( lane_data, mask )
                
    def calc_pinned_metrics_by_lane( self, hd=True, min_threshold = 0. ):
        """
        Calculates pinned pixel metrics by lane and stores in the self.lane_metrics dictionary.
        
        I am creating a new default to use hd=True here.  It is the way of the future.
        
        This does a number of different metric calculations at once, including:
        - Pinned Pixel Density metrics in the vein of localstd metrics for percentages pinned low, high, and either.
            - Duplicates of the above if filtering out superpixels with < (min_threshold)% pinned pixels. Default is 1%
        - Summary metrics of total % pinned low, high, or either per lane.
        """
        if not self.is_multilane:
            print( 'Skipping multilane analysis, as this is not a multilane chip' )
            return None
        
        suffix = ''
        if hd:
            suffix = '_hd'
            
        # Let's do the easy part first -- aggregate metrics
        # pinned pixels masks are stored in self.pinned_low and self.pinned_high
        try:
            aggregates = [ ('pinned_low' , self.pinned_low ),
                           ('pinned_high', self.pinned_high),
                           ('pinned'     , np.logical_or( self.pinned_low, self.pinned_high ) ) ]
        except AttributeError:
            print( "ERROR!  No pinned pixel data found.  Have you loaded gain, found refpix and loaded offsets?" )
            return None
            
        for name, data in aggregates:
            for i, lane, cs in self.iter_lane_slice( data ):
                # Count number of available (active) pixels per lane so we can get to the percentage
                lane_data = data[:,cs]
                if hasattr( self , 'active' ):
                    lane_mask = self.active[:,cs]
                    pinned    = lane_data[lane_mask].sum()
                    available = lane_mask.sum()
                else:
                    # We don't have active, so lets assume all pixels are fair game.  I hope we don't end up here.
                    print( "Warning!  Active pixel array does not exist -- calculating percent pinned pixels with full array . . . ")
                    pinned    = lane_data.sum()
                    available = lane_data.size
                    
                self.lane_metrics[lane][name] = pinned
                
                try:
                    self.lane_metrics[lane]['perc_{}'.format( name )] = 100. * float( pinned ) / float( available )
                except ZeroDivisionError:
                    print( 'Error!  Trying to calculate {} and encountered zero division errors -- is this lane really active?'.format( name ) )
                        
        # Pull in pinned percentages
        try:
            perc_low    = self.safe_load_perc_pinned( low=True, hd=hd )
            perc_high   = self.safe_load_perc_pinned( low=False, hd=hd )
            perc_pinned = perc_low + perc_high
        except (AttributeError, NameError):
            print( 'Error!  Unable to safely load perc_pinned pixel data' )
            return None
        
        # If we make it here, let's iterate through lanes
        analyses = [ ('local_pinned_low' , perc_low),
                     ('local_pinned_high', perc_high),
                     ('local_pinned',      perc_pinned) ]
        
        for name, data in analyses:
            for i, lane, cs in self.iter_lane_slice( data ):
                lane_data = data[:,cs]
                try:
                    # Note reversion of the not_reference logic to find perc with < 50% to avoid calling logical not on the array.
                    perc_refpix   = self.safe_load_refpix_heatmap( hd=hd )
                    not_reference = perc_refpix <= 50.
                    mask = not_reference[:,cs]
                    min_mask = np.logical_and( mask, lane_data > min_threshold )
                except AttributeError:
                    print( 'Error pulling in refpix mask . . . ignoring that for now.' )
                    mask = None
                    min_mask = lane_data > min_threshold
                    
                # Get statistics, first for all data
                self.lane_metrics[lane]['{}_all{}'.format(name,suffix)] = self.calc_metrics_from_array( lane_data , mask )
                self.lane_metrics[lane]['{}{}'.format(name,suffix)]     = self.calc_metrics_from_array( lane_data, min_mask )
                
    def enable_super_RC( self, enable=True ):
        """
        Special function that enables chunk analysis of BlockMath / local analysis calls.
        Calling the function will by default set chunks to chip blockR and blockC (96 blocks)
        This will not be enabled by default, but may be a good idea with 560 chips and higher.
        """
        if enable:
            self.superR = self.chiptype.blockR
            self.superC = self.chiptype.blockC
        else:
            self.superR = None
            self.superC = None

    def multilane_boxplot( self , metric , lims=None ):
        """ Makes a boxplot to compare distributions of metrics across lanes """
        if not self.is_multilane:
            print( 'Skipping multilane analysis, as this is not a multilane chip' )
            return None
        
        m = metric.lower()
        try:
            data = getattr( self , m )
            if not lims:
                lims = self.select_lims( m )
            units = self.get_units( m )
        except AttributeError:
            print( 'Error, metric ({}) not found.'.format( m ) )
            return None
        
        # If we make it here, let's iterate through lanes
        boxdata = [ [],[],[],[] ]
        for i, lane, lane_data in self.iter_lane_data( data ):
            mask = np.logical_and( lane_data > lims[0], lane_data < lims[1] ) # Ignore upper limit to make boxplot, like hist
            boxdata[i-1] = list( lane_data[mask][::100] ) # downsampling to save some compute time
            
        # Make boxplot
        labels = ['Lane 1','Lane 2','Lane 3','Lane 4']
        plt.figure ( )
        plt.boxplot( boxdata , sym='' , positions=np.arange(4) )
        plt.xlim   ( -0.5 , 3.5 )
        plt.xticks ( range(4), labels )
        plt.ylim   ( lims[0] , lims[1] )
        plt.ylabel ( '{} ({})'.format( m.title() , units ) )
        plt.grid   ( ls='--' , color='grey' , axis='y' )
        plt.savefig( os.path.join( self.output_dir , 'multilane_{}_boxplot.png'.format( m ) ) )
        plt.close  ( )
        return None

    def multilane_pinned_heatmaps( self , transpose=True, hd=False ):
        """ Makes a series of wafermap spatial plots for percentage of pixels pinned low, high, or both. """
        # Pull in pinned percentages
        perc_low  = self.safe_load_perc_pinned( low=True, hd=hd )
        perc_high = self.safe_load_perc_pinned( low=False, hd=hd )
        
        suffix = ''
        if hd:
            suffix = '_hd'
        
        pinned_plots = [('perc_pinned_low',perc_low), ('perc_pinned_high',perc_high), ('perc_pinned',perc_low+perc_high)]
        
        for name, zmax in [('_hs', 5), ('',20), ('_full',100)]:
            for label, data in pinned_plots:
                for i, lane, lane_data in self.iter_lane_data( data ):
                    m    = '{}{}{}'.format( label, name, suffix )
                    lims = [0, zmax ]
                    
                    # Determine proper image size for data
                    if transpose:
                        lane_data  = lane_data.T
                        img_name = 'lane_{}_{}_wafermap.png'.format(i,m)
                    else:
                        img_name = 'lane_{}_{}_wafermap_nonT.png'.format(i,m)
                        
                    w,h = matplotlib.figure.figaspect( lane_data )
                    aspect = float(w)/float(h)
                    dpi = float( matplotlib.rcParams['figure.dpi'] )
                    
                    if transpose:
                        w = (210. / dpi)
                        h = w / aspect
                    else:
                        h = (210. / dpi)
                        w = h * aspect
                        
                    fig = plt.figure( figsize=(w,h) )
                    ax  = fig.add_subplot(111)
                    plt.axis  ( 'off' )
                    plt.imshow( lane_data , origin='lower' , aspect='equal' , interpolation='nearest' , clim=lims )
                    
                    # Set extent of image to exactly that of the chip image
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig( os.path.join( self.output_dir , img_name ) , bbox_inches=extent )
                    plt.close  ( )
                    
    def custom_multilane_wafermap_spatial( self , data , metric_name , clims=None, transpose=True, cmap=None ):
        """
        Direct method to create arbitrary wafermap images per active lane.
        Modeled after self.wafermap_spatial.
        """
        if not self.is_multilane:
            print( 'Skipping multilane analysis, as this is not a multilane chip' )
            return None
        
        if cmap:
            cm = cmap
        else:
            cm = matplotlib.rcParams['image.cmap']
            
        for i, lane, lane_data in self.iter_lane_data( data ):
            # Determine proper image size for data
            if transpose:
                lane_data  = lane_data.T
                img_name = 'lane_{}_{}_wafermap.png'.format(i,metric_name)
            else:
                img_name = 'lane_{}_{}_wafermap_nonT.png'.format(i,metric_name)
                
            w,h = matplotlib.figure.figaspect( lane_data )
            aspect = float(w)/float(h)
            dpi = float( matplotlib.rcParams['figure.dpi'] )
            
            if transpose:
                w = (210. / dpi)
                h = w / aspect
                
            fig = plt.figure( figsize=(w,h) )
            ax  = fig.add_subplot(111)
            plt.axis  ( 'off' )
            if clims==None:
                plt.imshow( lane_data, origin='lower', aspect='equal', interpolation='nearest', cmap=cm )
            else:
                plt.imshow( lane_data, origin='lower', aspect='equal', interpolation='nearest', clim=clims, cmap=cm)

            # Set extent of image to exactly that of the chip image
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig( os.path.join( self.output_dir , img_name ) , bbox_inches=extent )
            plt.close  ( )
            
    def multilane_wafermap_spatial( self , metric , clims=None , transpose=True , cmap=None ):
        """ 
        Plots a series of wafermap spatial plots when given the full chip data array. 
        Does not create an image if the lane was inactive.
        Used for commonly held calibration metrics that are stored as attributes: getattr( self, metric )
        """
        if not self.is_multilane:
            print( 'Skipping multilane analysis, as this is not a multilane chip' )
            return None
        
        m = metric.lower()
        try:
            full_chip_array = getattr( self , m )
            if not clims:
                clims = self.select_lims( m )
        except AttributeError:
            print( 'Error, metric ({}) not found.'.format( m ) )
            return None
        
        self.custom_multilane_wafermap_spatial( full_chip_array , m , clims=clims, transpose=transpose, cmap=cmap )
        
    def analyze_lanes( self , metrics , make_plots=True ):
        """ 
        This master set of code is here to analyze metrics by lane, if it is a multilane chip 
        metrics is a list of metrics to plot
        One can avoid plotting by using the kwarg: make_plots=False.
        """
        if not self.is_multilane:
            print( 'Skipping multilane analysis, as this is not a multilane chip' )
            return None
        
        # If a list of specific metrics are applied, then let's use those.
        if not metrics:
            # Otherwise, we can default to all of them.
            metrics = ['noise','noise_localstd','gain','gain_localstd','offset','offset_localstd']
            
        for m in metrics:
            self.calc_metrics_by_lane( m )
            if make_plots:
                # Note that this will use default limits.
                self.multilane_wafermap_spatial( m )
                
    def set_lanes_from_flag( self, flag ):
        ''' Set lanes from an integer flag '''
        if flag:
            self.is_multilane = True
        lanes = '{0:04b}'.format(flag)[::-1]
        for i, v in enumerate(lanes):
            setattr( self, 'lane_{}'.format(i+1), bool(int(v)) )

    def determine_lane( self , gain_threshold=500. , leg_column_threshold=50 ):
        """ 
        Identifies multilane chips and, if so, determines which lanes are currently active 
        Based off of intial algorithm developed by Creighton for Lane_Diagnostics.
        Note: key properties are set in the init region for backwards compatability.
        """
        print( 'Determining flowcell type and/or active lanes' )
        
        # This algorithm uses gain.  Load it if it isn't there.
        if not hasattr( self , 'gain' ):
            print( 'Loading gain . . .' )
            self.load_gain( )
            print( ' . . . done.' )
            
        # Deal with the fact if someone loads gain in V/V mode.
        if self.gain_units == 'V/V':
            gain_threshold /= 1000.
            
        # Create the median gain per column by averaging across rows
        # Then, let's determine if the chip is multilane or not, using total number of columns with zero gain
        # --> Median will ignore extreme cases such as blooming which can lead to non-zero gain under the glue line
        med_col_gain = np.median( self.gain, axis=0 )
        # Trim the first and last 1/8th of the cols to exclude outermost perimeter
        tidx    = int(self.cols/8.) 
        print( 'tidx = {}'.format( tidx ) )
        num_zero_cols   = (med_col_gain[tidx:-tidx]==0).sum()
        print( 'num_zero_cols {}'.format( num_zero_cols )  )
        self.is_multilane   = bool( num_zero_cols > leg_column_threshold )
        #self.is_multilane   = bool( (avg_col_gain[int(self.cols/8):-int(self.cols/8)] == 0).sum() > leg_column_threshold )

        # Create the average gain per column by averaging across rows
        avg_col_gain = self.gain.mean(0)
        if self.is_multilane:
            print( 'Multilane Chip Detected.' )
            # Let's reshape it into four "lanes" in a 4 x (cols/4) array and iterate lane detection
            for i,lane in enumerate(avg_col_gain.reshape(4,-1)):
                setattr( self , 'lane_{}'.format(i+1) , bool(lane.mean() > gain_threshold) )
                print( 'Lane {} Active?: {}'.format( i+1, getattr(self , 'lane_{}'.format(i+1)) ) )
                
            # While we're in here, lets record the number of fluidically addressible wells.
            lane_width = self.gain.shape[1] / 4
            for i in range(4):
                cs = slice( lane_width*i , lane_width*(i+1) )
                self.lane_metrics[ 'lane_{}'.format(i+1) ]['addressable_wells'] = ( self.gain[:,cs] > gain_threshold ).sum()
        else:
             print( 'Standard Flowcell Detected.' )
             
        return None

    def init_edge_mask( self , dist=100 ):
        """
        Helper function for creating a mask of pixels around flowcell edge
        """
        mask = np.zeros( (self.rows,self.cols) , np.int16 )
        if dist > 0:
            mask[:dist ,:] = 1
            mask[-dist:,:] = 1
            mask[:, :dist] = 1
            mask[:,-dist:] = 1
        return mask
    
    def iter_lanes( self, *lanes ):
        """ 
        Wrapper function that returns a lane generator for simple iteration using lane_{} attributes unless lanes is supplied.
        returns lane id (integer in [1-4]) and string lane name.
        """
        if lanes:
            return mltools.iter_lanes( *lanes )
        else:
            return mltools.iter_lanes( self.lane_1, self.lane_2, self.lane_3, self.lane_4 )
        
    def iter_lane_data( self, data, *lanes ):
        """
        Wrapper function on lanes.iter_lane_data for simple iteration using lane_{} attributes unless lanes is supplied.
        returns lane id (integer in [1-4]), string lane name, and sliced data array for that lane.
        """
        if lanes:
            return mltools.iter_lane_data( data, *lanes )
        else:
            return mltools.iter_lane_data( data, self.lane_1, self.lane_2, self.lane_3, self.lane_4 )
        
    def iter_lane_slice( self, data, *lanes ):
        """
        Wrapper function on lanes.iter_lane_data for simple iteration using lane_{} attributes unless lanes is supplied.
        returns lane id (integer in [1-4]), string lane name, and slice for that lane.
        """
        if lanes:
            return mltools.iter_lane_slice( data, *lanes )
        else:
            return mltools.iter_lane_slice( data, self.lane_1, self.lane_2, self.lane_3, self.lane_4 )
        
    def iter_masked_lane_data( self, data, mask, *lanes ):
        """
        Wrapper function on lanes.iter_lane_masked_data for simple iteration using lane_{} attributes unless lanes is supplied.
        returns lane id (integer in [1-4]), string lane name, sliced data array for that lane, and a sliced mask array for that lane.
        """
        if lanes:
            return mltools.iter_masked_lane_data( data, mask, *lanes )
        else:
            return mltools.iter_masked_lane_data( data, mask, self.lane_1, self.lane_2, self.lane_3, self.lane_4 )
        
    def measure_flowcell_placement( self , outdir=None ):
        """ 
        Measures placement of flowcells in units of columns.
        CURRENTLY ONLY IMPLEMENTED FOR MULTILANE FLOWCELLS!
        if outdir is given, then images will be saved to that location.  Otherwise, image creation is skipped.
        """
        def make_colormap( color_list=['black','red','yellow','orange'] ):
            """
            Helper function to create a simple colormap for Ti Cup CMP Roll-off analysis
            """
            N     = len(color_list)
            clist = [colors.colorConverter.to_rgb(c) for c in color_list]
            cmap  = LinearSegmentedColormap.from_list( 'Overlap', clist , N=N )
            return cmap
        
        # Skip out of analysis if this isn't a multilane chip.
        if not self.is_multilane:
            print( "This is not a multilane chip!  Skipping this analysis." )
            return None
        
        # Set up metrics for later use
        fca = {}
        
        # Create colormap and legend items for the roll-off analysis plots
        cmap_colors  = ['black','blue','green','yellow']
        options      = ['Unused Wells','Shorter Wells (CMP)','Active Wells','Shorter Active Wells']
        legend_items = [patches.Patch( edgecolor='black', facecolor=c, label=o ) for (c,o) in zip(cmap_colors,options) ]
        cmap         = make_colormap(cmap_colors)
        
        # Identify edges of each lane
        lane_cols = self.cols/4
        for i, laneid, cs in self.iter_lane_slice( self.active ):
            print( 'Analyzing flowcell alignment for lane {}'.format( i ) )
            # This is meant to give an integer for slicing, even if real answer is a decimal.  Scales with chip (541,551,etc)
            offset = self.rows / 10  
            lane   = self.active[offset:-offset,cs]
            
            # Use row average to find mean start column for the lane.
            # Note that this gives edges in lane-relative columns.  Should also save as global columns.
            rowavg     = lane.mean(0)
            lane_cols  = rowavg.size
            left_edge  = np.diff( rowavg[:lane_cols/2 ] ).argmax( )
            print( 'Found left edge of flowcell lane at column {}'.format( left_edge ) )
            
            # Because the following code measures columns from right edge of lane, need to convert back to col number
            right_edge = np.diff( rowavg[-lane_cols/2:][::-1] ).argmax( )
            right_edge = (lane_cols - right_edge )
            print( 'Found right edge of flowcell lane at column {}'.format( right_edge ) )
            
            fca[laneid] = {'start': int(  left_edge ) , 'abs_start': int(  left_edge + (i-1)*lane_cols ) ,
                           'end'  : int( right_edge ) , 'abs_end'  : int( right_edge + (i-1)*lane_cols ) ,
                           'lane_width' : int( right_edge - left_edge ) }
            
            # Calculate % impacted active pixels as a function of *potential* CMP rolloff
            affected = np.zeros( (11,) , float )
            totalpix = self.active[:,cs].sum()
            for k in range( 11 ):
                dist = k * 25
                mask = self.init_edge_mask( dist )[:,cs]
                mask[ self.active[:,cs] ] += 2
                affected[k] = 100. * (mask == 3).sum() / totalpix
                
            fca[laneid]['perc_affected_by_25'] = affected.tolist()
            
            if outdir:
                if not os.path.exists( outdir ):
                    os.makedirs( outdir )
                    
                # Make the spatial plot showing overlap of the active lane and a potentially affected Ti Cup region.
                # Uses default of 100 microwells in.  That corresponds to affected[4] for % affected.
                mask = self.init_edge_mask( )
                mask[ self.active ] += 2
                
                # Select lane and downsample every 5 pixels
                arr = mask[:,cs][::5,::5]
                
                plt.figure( figsize=(12,6) )
                plt.imshow( arr.T , interpolation='nearest',origin='lower', cmap=cmap , clim=[0,3] )
                plt.xticks( [] )
                plt.yticks( [] )
                plt.ylabel( 'Columns' )
                plt.xlabel( 'Rows'    )
                plt.title ( 'Overlap of 100 edge microwells with active microwells | {:.1f}% Impacted | {}'.format( affected[4] ,
                                                                                                                    laneid) )
                plt.legend( handles=legend_items , loc='center' , bbox_to_anchor=(0.5,-0.12), ncol=4, fontsize=12 )
                plt.tight_layout( )
                plt.savefig( os.path.join( outdir , '{}_100_well_spatial_plot.png'.format( laneid ) ) )
                plt.close ( )
                
        return fca
    
    def check_chip_rc( self , r , c ):
        """ Takes supposedly known r,c of chip from relaible source and checks if it matches. """
        # Converting r to int(r) and c to int(c) handles a bug in the behavior of formatting type 
        #  '<u4' from an np.array on protons
        #   This likely stems from the python version 2.7.2+ -OR- numpy version 1.5.1
        #   as it is not an issue for python 2.7.15+ or numpy 1.16.6
        r = int(r)
        c = int(c)
        if r == self.chiptype.chipR and c == self.chiptype.chipC:
            print( 'Chip Type confirmed to be {}:{:d} rows and {:d} cols.'.format(self.chiptype.name , r , c ) )
            return None
        else:
            try:
                self.chiptype = ct.get_ct_from_rc( r , c )
            except ValueError:
                self.chiptype = ct.make_ct_from_rc( r, c, base_ct=self.chiptype )
            self.rows = self.chiptype.chipR
            self.cols = self.chiptype.chipC
            self.blockR = self.chiptype.miniR
            self.blockC = self.chiptype.miniC
            print( 'Chip Type mismatch found!  Now running with: ' )
            print( '{0.name}: {0.chipR} rows x {0.chipC} cols | {0.blockR} x {0.blockC} blocks | {0.miniR} x {0.miniC} miniblocks | {0.microR} x {0.microC} microblocks'.format( self.chiptype ) )
            return None
        
    def analyze_refpix( self ):
        if os.path.exists( os.path.join( self.output_dir, 'masked_refpix_quadrant.png' ) ) and not self.force:
            print( '... Noise has already been analyzed.  Use --force to override' )
            return
        if self.chiptype.ref_rows is None and self.chiptype.ref_cols is None:
            print( '... Reference pixels have not yet been defined for this chiptype' )
            return
        m = 'refpix_noise'
        start_time = time.time()
        self.annotate     ( 'Analyzing %s . . .' % m )
        # Load data
        self.annotate     ( "...Loading data" )
        self.load_offset  (   )
        self.load_noise   (   )
        # Get the flattened reference pixels for each quadrant
        self.annotate     ( "...Getting reference pixels" )
        self.get_reference_pixels( 'noise' )
        self.get_reference_pixels( 'offset' )
        # Get rid of data we don't need
        self.annotate     ( "...closing full chip data" )
        self.close_noise      ( )
        self.close_offset     ( )
        # Mask out noise pixels where offset is in the top or bottom 25% of the dac window.
        # These pixels likely pinned at some point and should be excluded
        self.annotate     ( "...Masking reference pixels" )
        self.mask_refpix( 'noise' )
        # Set noise limits roughly by chip type
        noisemean = self.refpix_noise.mean()
        self.set_refpix_props( 'noise', adjust=noisemean )
        self.set_refpix_props( 'noise', adjust=noisemean, masked=True )
        self.set_refpix_props( 'offset' )
        # Calculate metrics
        self.annotate     ( "...Calculating metrics" )
        self.calc_refpix_metrics( 'noise' )
        self.calc_refpix_metrics( 'noise', masked=True )
        self.calc_refpix_metrics( 'offset' )
        # Calculate the histograms for the full chip
        self.annotate     ( "...Plotting histograms" )
        self.overlay_histogram ( 'refpix_noise', 'refpix_noise_masked' )
        #self.histogram( 'refpix_noise' )
        #self.histogram( 'refpix_noise_masked' )
        self.histogram( 'refpix_offset' )
        # Make simple 2x2 plots of the quadrants
        self.annotate     ( "...Plotting quadrants" )
        self.plot_quadrants( 'refpix_noise' )
        self.plot_quadrants( 'refpix_noise_masked' )
        self.plot_quadrants( 'refpix_offset' )
        self.plot_quadrants( 'masked_refpix' )
        # Save the results
        self.annotate     ( "...Saving to disk" )
        self.save_json ( 'refpix' )
        self.annotate     ( "...%s calculations completed in %0.1f s" % ( m , (time.time() - start_time) ) )

    def plot_quadrants( self, metric ):
        ''' Creates a 2x2 plot showing the average reference pixel behavior in each quadrant '''
        m    = metric.lower()
        if metric in ['masked_refpix']:
            data = np.array( [ [ self.metrics['%s_lower_left' % m], self.metrics['%s_lower_right' % m] ],
                               [ self.metrics['%s_upper_left' % m], self.metrics['%s_upper_right' % m] ] ] )
        else:
            data = np.array( [ [ self.metrics['%s_lower_left_q2' % m], self.metrics['%s_lower_right_q2' % m] ],
                               [ self.metrics['%s_upper_left_q2' % m], self.metrics['%s_upper_right_q2' % m] ] ] )
        lims = getattr( self, '%s_lims' % metric )

        plt.figure   ( facecolor='w' )
        plt.imshow   ( data, origin='lower', interpolation='nearest', clim=lims )
        if metric in ['masked_refpix']:
            plt.title ( '%i Reference Pixels Masked' % self.metrics[ metric ] )
        else:
            plt.title ( self.create_title( metric ) )
        plt.colorbar ( )
        plt.savefig  ( '%s/%s_quadrant.png' % ( self.output_dir, m ) )
        plt.close    ( )

    def analyze_property( self, prop, buffering_gc=False, true_metrics=True ):
        ''' Default behavior is to analyze true_metrics '''
        m = prop.lower()
        
        if m == 'noise':
            if self.chiptype.series == 'pgm':
                arrmask = None

                print( 'Extracting background noise info . . .' )
                self.noise = imtools.GBI( self.noise[:,:,np.newaxis] , self.noise < 1 , 10 ).squeeze()
                self.noise[ np.isnan( self.noise ) ] = 0.
            else:
                # Set noise limits roughly by chip type
                noisemean = self.noise.mean()
                if noisemean < 100:
                    self.noise_lims = [ 0, 200 ]
                elif noisemean > 200:
                    self.noise_lims = [ 0, 1000 ]
                else:
                    self.noise_lims = [0, 600]
                
                if self.no_fc_mask:
                    arrmask = None
                else:
                    arrmask = self.load_arrmask()
        else:
            if self.no_fc_mask:
                arrmask = None
            else:
                arrmask = self.load_arrmask()
        
        self.calc_metrics ( m , mask=arrmask, ignore_upper_limit=True )
        self.histogram    ( m , mask=arrmask, rescale_ylim=True ) 
            
        self.spatial                ( m )
        self.wafermap_spatial_metric( m )
        # Cannot apply mask until here because we want to plot all noise values
        if arrmask is not None:
            getattr( self, m )[~arrmask] = 0
        self.colavg       ( m )
        self.plot_colavg  ( m )
        self.plot_colavg  ( m , True )
        
        ##################################################
        # Generate (spam) superpixel attributes
        ##################################################
        # This is the historical default
        self.superpixel_analysis( m ) 
        self.local_std    ( m )
        self.local_avg    ( m )

        self.diff_img     ( m )
        self.plot_diff_img( m )
        self.diff_img_hd  ( m )
        
        if m == 'noise':
            self.edge_noise   (   )
        elif m == 'offset':
            self.edge_offset_test(   )
        elif m == 'gain' and buffering_gc:
            self.buffering_gc( )

        self.plot_colavg  ( m+'_localstd' )
        
        if true_metrics:
            try:
                self.gain
                self.find_refpix() # need to explicitly call because it isn't done in loading
            except AttributeError:
                self.load_gain()
                self.find_refpix() # need to explicitly call after loading gain
                self.close_gain()
            # Deal with offsets AFTER gain in order to exclude reference pixels.
            try:
                # offset must have been loaded but, getting pinned is complicated
                # --> close offset, and load it again
                # TODO: Revisit how offset and pinned are utilized B.P. 06Dec2019
                self.offset
                self.close_offset()
                self.load_offset()
                self.close_offset()
            except AttributeError:
                self.load_offset() # load_offset calls find_pinned so we're fine
                self.close_offset()

            # This is the better way to do it (ignore_pinned=True)
            self.superpixel_analysis( m, ignore_pinned=True )
            self.local_std    ( m, ignore_pinned=True )
            
            # This is the best way to do it (ignore_pinned=True and hd=True)
            self.superpixel_analysis( m, ignore_pinned=True, hd=True ) 
            self.local_std    ( m, ignore_pinned=True, hd=True )
            self.local_avg    ( m, ignore_pinned=True )
            self.diff_img_hd  ( m, ignore_pinned=True )
        ##################################################
        
        self.save_json    ( m )
        return None

    def analyze_noise( self, close=True, true_metrics=True ):
        ''' Default behavior is to analyze true_metrics '''
        m          = 'noise'
        if os.path.exists( os.path.join( self.output_dir, 'edge_noise_colavg.dat' ) ) and not self.force:
            print( '... Noise has already been analyzed.  Use --force to override' )
            return
        start_time = time.time()
        self.annotate     ( 'Analyzing calibration %s . . .' % m)
        self.load_noise   ( )

        self.analyze_property( m, true_metrics=true_metrics )
        
        if close:
            self.close_noise  (   )

        if self.is_multilane:
            self.calc_metrics_by_lane( m )
            self.calc_metrics_by_lane( m+'_localstd' )
        else:
            self.annotate( 'Skipping multilane analysis, as this is not a multilane chip' )

        self.annotate     ( "...%s calculations completed in %0.1f s" % ( m , (time.time() - start_time) ))
        return None

    def analyze_offset( self, close=True, true_metrics=True ):
        ''' Defaut behavior is to analyze true_metrics '''
        m          = 'offset'
        if os.path.exists( os.path.join( self.output_dir, 'offset_colavg_dd.dat' ) ) and not self.force:
            print( '... Offsets have already been analyzed.  Use --force to override' )
            return
        start_time = time.time()
        self.annotate        ( 'Analyzing calibration %s . . .' % m)
        self.load_offset     (  )

        self.analyze_property( m, true_metrics=true_metrics )

        if close:
            self.close_offset    (   )

        # Analyze pinned metrics and move value into the offset output section
        self.pinned_heatmaps( )
        self.pinned_heatmaps( hd=True )
        self.save_json( 'pinned', flatten=True )

        # analyze multilane behavior
        if self.is_multilane:
            self.calc_metrics_by_lane( m )
            self.calc_metrics_by_lane( m+'_localstd' )
            self.calc_pinned_metrics_by_lane( )
            self.multilane_pinned_heatmaps( hd=False )
            self.multilane_pinned_heatmaps( hd=True  )
        else:
            self.annotate( 'Skipping multilane analysis, as this is not a multilane chip' )

        self.annotate        ( "...%s calculations completed in %0.1f s" % ( m , (time.time() - start_time) ))
        return None

    def analyze_gain( self , buffering_gc=False, close=True, true_metrics=True ):
        ''' Default behavior is to analyze true_metrics '''
        if os.path.exists( os.path.join( self.output_dir, 'offset_diff_img.png' ) ) and not self.force:
            if buffering_gc and not any( [ 'buffering_gc' in f for f in os.listdir( self.output_dir ) ] ):
                pass
            else:
                print( '... Gain has already been analyzed.  Use --force to override' )
                return
        m          = 'gain'
        start_time = time.time()
        self.annotate     ( 'Analyzing calibration %s . . .' % m)
        self.load_gain    ( )

        self.analyze_property( m, buffering_gc=buffering_gc, true_metrics=true_metrics )

        if close:
            self.close_gain   (   )

        if self.is_multilane:
            self.calc_metrics_by_lane( m )
            self.calc_metrics_by_lane( m+'_localstd' )
        else:
            self.annotate( 'Skipping multilane analysis, as this is not a multilane chip' )

        self.annotate     ( "...%s calculations completed in %0.1f s" % ( m , (time.time() - start_time) ))
        return None

    def annotate( self , text ):
        """ writes to output if verbose, but always writes to log file. """
        if self.verbose:
            print( text )
        log = open ( self.logfile , 'a' )
        log.write  ( '%s\n' % text )
        log.close  ( )
        return None
        
    def buffering_gc( self, all_buffering=True):
        """ 
        Does gain correction of buffering metric 
        First attempts to do so at the full chip level.  If files don't exist, then it does so at the block level
        set all_buffering = True to apply the gain correction to any file entitled *buffering.dat
        """

        if all_buffering:
            files = os.listdir( self.caldir )
            files = [ '%s/%s' % ( self.caldir, f ) for f in files if 'buffering.dat' in f and '_avg_' not in f and '_std_' not in f ]
        else:
            files = [ '%s/buffering.dat' % ( self.caldir ) ]

        if not len( files ):
            # If there aren't any full chip files, do it at the block level
            return self.buffering_gc_block( all_buffering=all_buffering )

        for f in files:
            # Skip symlinks wince these will otherwise be detected
            if os.path.islink( f ):
                continue
            print( 'ANALYZING BUFFERING_GC: {}' ).format(f)
            try:
                buf     = dp.read_dat( f, 'buffering' ).reshape( self.chiptype.chipR, self.chiptype.chipC )
                #buf     = np.fromfile( f , dtype=dt ).reshape( self.chiptype.chipR, self.chiptype.chipC ) / scale
                buffering_gc = self.gain * np.array( buf , float )
            except:
                if allow_missing:
                    buffering_gc = np.zeros( (self.chiptype.chipR, self.chiptype.chipC ) )
                else:
                    raise

            if self.gain_units == 'mV/V':
                buffering_gc /= 1000.

            # Save buffering to file
            filename = '%s_gc.dat' % ( f[:-4] )

            dp.write_dat( buffering_gc, filename, 'buffering_gc' )

        return None

    def buffering_gc_block( self, all_buffering=True):
        """ Does gain correction of buffering metric if desired. """
        if self.chiptype.series== 'proton':
            blkR = self.chiptype.blockR
            blkC = self.chiptype.blockC
            for r in range( self.chiptype.yBlocks ):
                for c in range( self.chiptype.xBlocks ):
                    rws = blkR * np.array(( r , r + 1 ))
                    cls = blkC * np.array(( c , c + 1 ))
                    if self.gain_units == 'mV/V':
                        roi = self.gain[ rws[0]:rws[1] , cls[0]:cls[1] ] / 1000.
                    else:
                        roi = self.gain[ rws[0]:rws[1] , cls[0]:cls[1] ]
                        
                    block = 'X%s_Y%s' % ( blkC * c , blkR * r )
                    buf_dir = os.path.join( self.caldir , block )
                    if all_buffering:
                        files = os.listdir( buf_dir )
                        files = [ '%s/%s' % ( buf_dir, f ) for f in files if 'buffering.dat' in f ]
                    else:
                        files = [ '%s/buffering.dat' % ( buf_dir) ]

                    #dt    = datprops['buffering']['dtype']
                    #scale = datprops['buffering']['scale']

                    for f in files:
                        print( 'ANALYZING BUFFERING_GC %s' ).format( f )
                        try:
                            buf     = dp.read_dat( f, 'buffering' ).reshape( blkR , blkC )
                            #buf     = np.fromfile( f , dtype=dt ).reshape( blkR , blkC ) / scale
                            buffering_gc = roi * np.array( buf , float )
                        except:
                            if allow_missing:
                                buffering_gc = np.zeros( (blkR , blkC) , np.int16 )
                            else:
                                raise
                            
                        # Save buffering to file
                        filename = '%s_gc.dat' % ( f[:-4] )
                        dp.write_dat( buffering_gc, filename, 'buffering_gc' )
        return None
    
    def calc_refpix_metrics( self, metric, masked=False ):
        ''' Handler to calc_metrics to operate on each quadrant '''
        if masked:
            tail = '_masked'
        else:
            tail = ''
        self.calc_metrics( 'refpix_%s%s' % ( metric, tail ) )
        self.calc_metrics( 'refpix_%s%s_lower_left'  % ( metric, tail ) )
        self.calc_metrics( 'refpix_%s%s_lower_right' % ( metric, tail ) )
        self.calc_metrics( 'refpix_%s%s_upper_left'  % ( metric, tail ) )
        self.calc_metrics( 'refpix_%s%s_upper_right' % ( metric, tail ) )

    def select_lims( self, metric ):
        """ Cleanses metric of true and hd additions to return the important limits """
        lims_met = metric.lower()
        if 'true_' in lims_met:
            lims_met = lims_met.replace( 'true_', '' )
        if '_hd' in lims_met:
            lims_met = lims_met.replace( '_hd', '' )
        lims = getattr( self , '%s_lims' % lims_met )
        return lims
        
    def calc_metrics( self , metric, mask=None, ignore_upper_limit=False ):
        """ 
        Calculates set of standard metrics for given metric of choice.
        """
        data = self.safe_load( metric )
        m    = metric.lower()
        
        # Need to remove true and hd in order to get correct lims
        lims = self.select_lims( m )
        
        # Apply mask
        if mask is None:
            mask = data > np.ones( data.shape, dtype=np.bool )
        mask = np.logical_and( mask, data > lims[0] )
        if not ignore_upper_limit:
            mask = np.logical_and( mask, data < lims[1] )
        #mask = np.logical_and( data > lims[0] , data < lims[1] )
        
        # Check if we've found refpix, and mask them out if so.  Only works for noise, offset, and gain.
        if hasattr( self , 'active' ) and metric in ['noise','offset','gain']:
            # print( 'Now calculating metrics for %s but now masking refpix . . .' % metric )
            mask = np.logical_and( mask, self.active )
            
        # Calculate difference between top and bottom halves of the chip.
        self.metrics['{}_tb_diff'.format(m)] = stats.top_bottom_diff( data, mask )
        
        vals  = data[ mask ]
        percs = stats.percentiles( vals )
        
        try:
            self.metrics[ '%s_sample' % m ] = 100 * float( vals.size ) / float( data.size )
        except:
            self.metrics[ '%s_sample' % m ] = 0.
        self.metrics[ '%s_mean' % m ]   = vals.mean()
        self.metrics[ '%s_std' % m  ]   = vals.std()
        try:
            self.metrics['%s_P90' % m] = float( percs['p90'])
        except:
            self.metrics['%s_P90' % m] = 0.
        try:
            self.metrics['%s_q1' % m] = float( percs['q1'])
        except:
            self.metrics['%s_q1' % m] = 0.
        try:
            self.metrics['%s_q2' % m] = float( percs['q2'])
        except:
            self.metrics['%s_q2' % m] = 0.
        try:
            self.metrics['%s_q3' % m] = float( percs['q3'])
        except:
            self.metrics['%s_q3' % m] = 0.
        try:
            self.metrics['%s_iqr' % m] = float( percs['iqr'])
        except:
            self.metrics['%s_iqr' % m] = 0.
            
        if m == 'offset':
            for x in [ y for y in self.metrics.keys() if 'offset' in y ]:
                self.metrics[ 'pix_%s' % x ] = self.metrics[ x ]
        return None
    
    def create_title( self , metric, opt_metric=None ):
        """ Makes standard plot title based on metric of choice """
        # Define patterns
        means = "fc = %0.0f%% | (Mean, SD, P90) = (%0.0f, %0.0f, %0.0f) %s"
        hists = "(Q2, IQR, Mode) = (%0.0f, %0.0f, %0.0f) %s"

        met   = metric.lower()
        units = self.get_units( metric )

        means_main = means % ( self.metrics.get( '%s_sample' % met , 0. ) , 
                               self.metrics.get( '%s_mean' % met , 0. ) , 
                               self.metrics.get( '%s_std' % met , 0. ) , 
                               self.metrics.get( '%s_P90' % met , 0. ) , 
                               units )
        hists_main = hists % ( self.metrics.get( '%s_q2' % met , 0. ) , 
                               self.metrics.get( '%s_iqr' % met , 0 ) , 
                               self.metrics.get( '%s_mode' % met , 0. ) ,
                               units )
        if opt_metric is None:
            return '%s\n%s' % ( means_main, hists_main )
        else:
            met2   = opt_metric.lower()
            units2 = self.get_units( opt_metric )

            hists_opt = hists % ( self.metrics.get( '%s_q2' % met2 , 0. ) , 
                                  self.metrics.get( '%s_iqr' % met2 , 0 ) , 
                                  self.metrics.get( '%s_mode' % met2 , 0. ) ,
                                  units2 )
            return '%s | %s\n%s | %s' % ( met,  hists_main,
                                          met2, hists_opt )
    
    def edge_noise( self ):
        """ Runs test for edge noise and generates a few metrics of use in assessing edge noise """
        if not hasattr( self , 'noise_diff' ):
            self.diff_img( 'noise' )
        
        blockC = self.noise_diff.shape[1] / 12
        roi    = self.noise_diff[ 2:-2 , 3*blockC:9*blockC ]
        avg    = roi.mean(1)
        sd     = roi.std(1)
        pts    = np.arange( len(avg) ) - len(avg) / 2.
        
        self.metrics['noise_diff_colavg_sum'] = float( avg.sum() )
        self.metrics['noise_diff_colavg_sd' ] = float( avg.std() )
        
        plt.figure      ( )
        plt.plot        ( pts , avg , '-' , lw = 2.5 )
        plt.fill_between( pts , 0 , avg , interpolate=True , color='red'   , alpha=0.7 , where=(avg > 0))
        plt.fill_between( pts , 0 , avg , interpolate=True , color='green' , alpha=0.7 , where=(avg < 0))
        plt.axhline     ( y=0 , color='black' , lw = 2.5 )
        plt.grid        ( )
        plt.xlim        ( pts[0] , pts[-1] )
        plt.ylim        ( -20    , 100     )
        plt.ylabel      ( 'Noise above central mean (uV)' )
        plt.xlabel      ( 'Pixel blocks from center row'  )
        plt.title       ( 'Noise difference colavg | AUC = %.1f , SD = %.1f' % ( self.metrics['noise_diff_colavg_sum'] , self.metrics['noise_diff_colavg_sd'] ) )
        plt.savefig     ( os.path.join( self.output_dir , 'edge_noise_colavg.png' ) )
        plt.close       ( )
        
        # Save the y values to file
        self.save_dat( avg, 'edge_noise_colavg' )
        
        return None

    def edge_offset_test( self ):
        """ runs a second derivative of offset colavg and looks for drastically different offsets """
        if not hasattr( self , 'offset_colavg' ):
            self.colavg( 'offset' )
            
        ddo = np.diff( np.diff( self.offset_colavg ) )
        ddo_metric = np.abs( ddo[ np.abs(ddo) > 5 ] ).sum()
        self.metrics[ 'offset_colavg_dd_sum' ] = ddo_metric
        
        # Plot 2nd derivative
        plt.figure    ( )
        plt.plot      ( ddo , 'o:' )
        plt.axhline   ( y=5  , color='green' , lw=1.5 )
        plt.axhline   ( y=-5 , color='green' , lw=1.5 )
        plt.ylim      ( -25 , 25 )
        plt.grid      ( )
        plt.title     ( 'Second derivative of offset colavg | metric = %.1f' % ddo_metric )
        plt.savefig   ( os.path.join( self.output_dir , 'offset_colavg_dd.png' ) )
        plt.close     ( )
        
        # Save the values to file
        self.save_dat( ddo, 'offset_colavg_dd' )
        
        return None
    
    def find_refpix( self , gain_cutoff=500 ):
        if not hasattr( self , 'gain' ):
            print( "Error!  Have not yet loaded gain.  Please load and try again." )
            return None
        
        # Create binary footprint for binary_opening operation
        footprint = np.zeros((5,5))
        footprint[1:4,:] = 1
        footprint[:,1:4] = 1
        mask = ndimage.morphology.binary_opening( self.gain < gain_cutoff , structure=footprint , iterations=2 )
        
        # Correct for binary_opening false Falses at extreme corners.
        mask[ 0:2 , 0:2 ] = True
        mask[ 0:2 ,-2:  ] = True
        mask[-2:  , 0:2 ] = True
        mask[-2:  ,-2:  ] = True
        
        self.active = ~mask
        self.refpix = mask
        
    def get_reference_pixels( self, metric ):
        ''' Extract the flattened reference pixel information by quadant '''
        # Load the reference pixel mask
        refmask = self.load_refmask()

        # get the data
        data = getattr( self, metric )

        # make slices
        bot   = slice( 0, self.chiptype.chipR/2 )
        top   = slice( self.chiptype.chipR/2, -1 )
        left  = slice( 0, self.chiptype.chipC/2 )
        right = slice( self.chiptype.chipC/2, -1 )

        # Extract the regions
        lower_left  = data[bot,left][refmask[bot,left]].flatten()
        lower_right = data[bot,right][refmask[bot,right]].flatten()
        upper_left  = data[top,left][refmask[top,left]].flatten()
        upper_right = data[top,right][refmask[top,right]].flatten()

        # Combine the regions
        refpix       = np.append( lower_left, lower_right )
        refpix       = np.append( refpix, upper_left )
        refpix       = np.append( refpix, upper_right ).flatten()

        # Save
        setattr( self, 'refpix_%s' % metric, refpix )
        setattr( self, 'refpix_%s_lower_left'  % metric, lower_left )
        setattr( self, 'refpix_%s_lower_right' % metric, lower_right )
        setattr( self, 'refpix_%s_upper_left'  % metric, upper_left )
        setattr( self, 'refpix_%s_upper_right' % metric, upper_right )

    def load_arrmask( self ):
        ''' Loads and returns the array mask (array=True) from the flowcell property
        If no mask is specified, returns None'''

        try:
            return self.arrmask
        except AttributeError:
            pass
        fc = self.chiptype.flowcell 
        if fc:
            filename = os.path.join( moduleDir, 'tools', 'dats', fc )
            try:
                self.arrmask = chip.load_mask( filename, self.chiptype )
            except IOError:
                filename = os.path.join( moduleDir, 'dats', fc )
                self.arrmask = chip.load_mask( filename, self.chiptype )
        else:
            return None
        return self.arrmask

    def load_refmask( self ):
        ''' Loads and returns the referencepixel mask (refpix=True) from the ref_array property

        If no mask is specified, then a mask is defined from ref_rows and ref_cols
        '''
        try:
            return self.refmask
        except AttributeError:
            pass

        ra = self.chiptype.ref_array
        if ra:
            filename = os.path.join( moduleDir, 'tools', 'dats', ra )
            try:
                # Need to invert because load_mask is designed to return array=True
                self.refmask = ~chip.load_mask( filename, self.chiptype )
            except IOError:
                filename = os.path.join( moduleDir, 'dats', ra )
                self.refmask = ~chip.load_mask( filename, self.chiptype )
        else:
            self.refmask = np.zeros( ( self.chiptype.chipR, self.chiptype.chipC ), dtype=np.bool )
            ref_rows = self.chiptype.ref_rows
            ref_cols = self.chiptype.ref_cols if self.chiptype.ref_cols is not None else 0
            if ref_rows:
                self.refmask[:ref_rows,:]  = True
                self.refmask[-ref_rows:,:] = True
            if ref_cols:
                self.refmask[:,:ref_cols]  = True
                self.refmask[:,-ref_cols:] = True
        return self.refmask

    def load_gain( self , volt_per_volt = False, mask=False, filename=None):
        """
        Opens up gain files for calibration data.
        Please use chip name [ "314R" , "P0" , "P2" ] for the chiptype input rather than '900' or 'P1.2.18'
        NOTE: This code writes out gain as a float, in units of mV/V, standard for each chip type.
        """
        img = np.zeros( 0 )
        if filename is None:
            if self.chiptype.series == 'pgm':
                gainfile = os.path.join( self.caldir , 'gainimage.dat' )
            else:
                gainfile = os.path.join( self.caldir , 'gainImage0.dat' )
            if not os.path.exists( gainfile ):
                gainfile = os.path.join( self.caldir , 'gainImage2.dat' )
            if not os.path.exists( gainfile ):
                gainfile = os.path.join( self.caldir , 'gainImage3.dat' )
        else:
            gainfile = filename
        if not os.path.exists( gainfile ):
            if allow_missing:
                print( 'File does not exist' )
            else:
                raise IOError( 'File (gainImage*.dat) does not exist' )

        if os.path.exists( gainfile ):
            self.gainfile = os.path.basename( gainfile )
            with open( gainfile , 'r' ) as f:
                if self.chiptype.series == 'pgm':
                    img       = np.fromfile( f, count=self.rows*self.cols, dtype=np.dtype('>H')).reshape( self.rows , self.cols , order='C')
                    self.gain = np.array( img, dtype=float ) / 69. / 50.
                elif self.chiptype.series == 'proton':
                    hdr  = np.fromfile (f, count=4, dtype=np.dtype('<u4'))
                    flag = hdr[1]
                    cols = hdr[2]
                    rows = hdr[3]
                    self.check_chip_rc( rows, cols )
                    img  = np.fromfile (f, count=rows*cols, dtype=np.dtype('<u2')).reshape (rows, cols, order='C')
                    
                    # New method (PW 01-2016)
                    # print('Flag = %s' % flag )
                    if int(flag) == 1:
                        # flag=1 signifies newly scaled gain in file.
                        # gainImage#.dat is uint16, with scaled output of [0,16383] mapped to [0,2.0] gain (V/V)
                        self.gain = np.array( img , dtype=float ) * 2.0 / 16383.
                    else:
                        # gainImage#.dat is the raw measured step in counts (uint16), 
                        # To scale gain with flag=0, need to use DR and calgainstep, or worse guess that step is
                        # just an 8th of the window, i.e. multiply by 8/16383
                        # ** Since there are several versions of DC floating around, going to keep the 8/16383
                        # ** Until the flagged version propogates through software.
                        self.gain = np.array( img , dtype=float ) * 8. / 16383.
                    
                else:
                    print( 'ERROR: Unknown chip type supplied. (%s)' % self.chiptype )
            if volt_per_volt:
                self.gain_units = 'V/V'
                self.gain_diff_lims = ( -0.1 , 0.1 )
            else:
                # Units remain as mV/V
                self.gain *= 1000.
        else:
            print( 'ERROR: Gain file does not exist.' )

        try:
            if mask:
                arrmask = self.load_arrmask()
                if arrmask is not None:
                    self.gain[~arrmask] = 0
        except AttributeError:
            pass

        return None

    def find_pinned( self , raw_offsets , exclude_refpix=True , low=80 , high=16300 ):
        ''' finds pinned pixels in the active array, but only runs if active/refpix have been found.'''
        # Initialize metrics
        total_pix    = float( raw_offsets.shape[0] ) * float( raw_offsets.shape[1] )
        inactive_pix = 0
        
        self.metrics['PixelLow']      = 0
        self.metrics['PixelHigh']     = 0
        self.metrics['PixelInactive'] = 0
        self.metrics['PixelInRange']  = 0
        self.metrics['PixelCount' ]   = total_pix
        self.metrics['PercPinnedLow']  = 0
        self.metrics['PercPinnedHigh'] = 0
        self.metrics['PercPinned_TB_Diff'] = 0
        
        # Add in cutoffs used to detect pinned pixels.
        self.metrics['PinnedLowThreshold']  = low
        self.metrics['PinnedHighThreshold'] = high
        
        if exclude_refpix:
            if not hasattr( self , 'active' ):
                print( 'Error!  Cannot calculate true pinned because we have not run find_refpix().' )
                return None
            inactive_pix     = self.refpix.sum()
            #self.pinned_low  = raw_offsets[self.active] < low
            #self.pinned_high = raw_offsets[self.active] > high
            self.pinned_low  = np.logical_and( raw_offsets < low , self.active )
            self.pinned_high = np.logical_and( raw_offsets > high, self.active )
            pixel_low        = self.pinned_low.sum()
            pixel_high       = self.pinned_high.sum()
            pixel_in_range   = total_pix - inactive_pix - pixel_low - pixel_high
            
            # Top - bottom difference in % pinned pixels
            tb_diff = 100. * stats.top_bottom_diff( np.logical_or( self.pinned_low, self.pinned_high ),
                                                    self.active )
        else:
            self.pinned_low  = raw_offsets < low
            self.pinned_high = raw_offsets > high
            pixel_low        = (raw_offsets < low).sum()
            pixel_high       = (raw_offsets > high).sum()
            pixel_in_range   = total_pix - pixel_low - pixel_high
            
            # Top - bottom difference in % pinned pixels
            tb_diff = 100. * stats.top_bottom_diff( np.logical_or(raw_offsets < low, raw_offsets < high) )
            
        self.metrics['PixelLow']       = int( pixel_low )
        self.metrics['PixelHigh']      = int( pixel_high )
        self.metrics['PixelInactive']  = int( inactive_pix )
        self.metrics['PixelInRange']   = int( pixel_in_range )
        self.metrics['PixelCount' ]    = int( total_pix - inactive_pix )
        self.metrics['PercPinnedLow']  = 100. * float(pixel_low)  / self.metrics['PixelCount' ]
        self.metrics['PercPinnedHigh'] = 100. * float(pixel_high) / self.metrics['PixelCount' ]
        self.metrics['PercPinned']     = self.metrics['PercPinnedLow'] + self.metrics['PercPinnedHigh']
        
        self.metrics['PercPinned_TB_Diff'] = float( tb_diff )
        
        return None
    
    def safe_load_perc_pinned( self, low=True, hd=False ):
        """ 
        Safe way to bring in pinned pixel averages if they don't exist 
        default is to do this non-HD...
        """
        try:
            if low:
                source = self.pinned_low
                attr = 'perc_low'
            else:
                source = self.pinned_high
                attr = 'perc_high'
        except AttributeError:
            print( 'Error!  You need to load gain, find refpix, then load offsets!' )
            raise
        
        if hd:
            attr += '_hd'
            
        try:
            return getattr( self, attr )
        except AttributeError:
            pass
        
        try:
            bm = BlockMath.from_chipcal( self, np.array( source, float ), hd=hd )
            setattr( self, attr, 100. * bm.get_mean() )
            return getattr( self, attr )
        except AttributeError:
            raise NameError('Error!  Unknown metric given!')

    def safe_load_refpix_heatmap( self, hd=False ):
        """
        Safe way to load and avoid recalculation of reference pixel heatmaps.
        default is non-HD....still...
        """
        try:
            source = self.refpix
            attr   = 'perc_refpix'
        except AttributeError:
            print( 'Error!  You need to load gain and find refpix in order to load refpix heatmaps. . .' )
            raise
        
        if hd:
            attr += '_hd'
            
        try:
            return getattr( self, attr )
        except AttributeError:
            ref_mask  = BlockMath.from_chipcal( self, self.refpix.astype(float) , hd=hd )
            setattr( self, attr, 100. * ref_mask.get_mean() )

        return getattr( self, attr )
    
    def pinned_heatmaps( self, hd=False, min_threshold = 0. ):
        """ 
        Makes pinned pixel heatmaps to see pinned spatial distribution 
        
        Now in v4.0.0, also creates local metrics.  Or "uniformity" of pinned pixel density.
        """
        # Start with pinned low
        perc_low = self.safe_load_perc_pinned( low=True, hd=hd )
        
        # Create reference pixel mask in superpixel space.  mask out superpixels that are >= 50% refpix....for now?
        # Note reversion of the not_reference logic to find perc with < 50% to avoid calling logical not on the array.
        perc_refpix = self.safe_load_refpix_heatmap( hd=hd )
        not_reference = perc_refpix <= 50.
        
        suffix = ''
        if hd:
            suffix = '_hd'
            
        plt.figure  ( )
        plt.imshow  ( perc_low , origin='lower' , interpolation='nearest' , clim=[0,5] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Percent Pixels Pinned Low' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_low_hs_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        plt.figure  ( )
        plt.imshow  ( perc_low , origin='lower' , interpolation='nearest' , clim=[0,20] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Percent Pixels Pinned Low' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_low_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        plt.figure  ( )
        plt.imshow  ( perc_low , origin='lower' , interpolation='nearest' , clim=[0,100] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Percent Pixels Pinned Low' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_low_full_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        # Histogram
        plt.figure  ( )
        plt.hist    ( perc_low[ not_reference ] , bins=np.linspace(0,100,51) )
        plt.xlabel  ( '% pixels pinned low ("superpixel")' )
        plt.xlim    ( 0 , 100 )
        plt.ylabel  ( '"Superpixel" count' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_low_histogram{}.png'.format(suffix) ) )
        plt.close   ( )
        
        # Test wafermap_spatial rotated image
        self.wafermap_spatial( perc_low , 'perc_pinned_low_hs{}'.format(suffix)  , [0,  5] )
        self.wafermap_spatial( perc_low , 'perc_pinned_low{}'.format(suffix)      , [0, 20] )
        self.wafermap_spatial( perc_low , 'perc_pinned_low_full{}'.format(suffix) , [0,100] )
        
        perc_high = self.safe_load_perc_pinned( low=False, hd=hd )
        
        plt.figure  ( )
        plt.imshow  ( perc_high , origin='lower' , interpolation='nearest' , clim=[0,5] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Percent Pixels Pinned High' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_high_hs_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        plt.figure  ( )
        plt.imshow  ( perc_high , origin='lower' , interpolation='nearest' , clim=[0,20] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Percent Pixels Pinned High' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_high_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        plt.figure  ( )
        plt.imshow  ( perc_high , origin='lower' , interpolation='nearest' , clim=[0,100] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Percent Pixels Pinned High' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_high_full_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        # Histogram
        plt.figure  ( )
        plt.hist    ( perc_high[ not_reference ] , bins=np.linspace(0,100,51) )
        plt.xlim    ( 0 , 100 )
        plt.xlabel  ( '% pixels pinned high ("superpixel")' )
        plt.ylabel  ( '"Superpixel" count' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_high_histogram{}.png'.format(suffix) ) )
        plt.close   ( )
        
        # Test wafermap_spatial rotated image
        self.wafermap_spatial( perc_high , 'perc_pinned_high_hs{}'.format(suffix)  , [0,  5] )
        self.wafermap_spatial( perc_high , 'perc_pinned_high{}'.format(suffix)      , [0, 20] )
        self.wafermap_spatial( perc_high , 'perc_pinned_high_full{}'.format(suffix) , [0,100] )
        
        perc_pinned = perc_low + perc_high
        
        plt.figure  ( )
        plt.imshow  ( perc_pinned , origin='lower' , interpolation='nearest' , clim=[0,5] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Total Percent Pinned Pixels' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_hs_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        plt.figure  ( )
        plt.imshow  ( perc_pinned , origin='lower' , interpolation='nearest' , clim=[0,20] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Total Percent Pinned Pixels' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        plt.figure  ( )
        plt.imshow  ( perc_pinned , origin='lower' , interpolation='nearest' , clim=[0,100] )
        plt.colorbar( shrink=0.7 )
        plt.title   ( 'Total Percent Pinned Pixels' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_full_spatial{}.png'.format(suffix) ) )
        plt.close   ( )
        
        # Histogram
        plt.figure  ( )
        plt.hist    ( perc_pinned[ not_reference ] , bins=np.linspace(0,100,51) )
        plt.xlim    ( 0 , 100 )
        plt.xlabel  ( '% Pixels Pinned ("SuperPixel")' )
        plt.ylabel  ( '"SuperPixel" count' )
        plt.savefig ( os.path.join( self.output_dir , 'perc_pinned_histogram{}.png'.format(suffix) ) )
        plt.close   ( )
        
        # Test wafermap_spatial rotated image
        self.wafermap_spatial( perc_pinned , 'perc_pinned_hs{}'.format(suffix)  , [0,  5] )
        self.wafermap_spatial( perc_pinned , 'perc_pinned{}'.format(suffix)      , [0, 20] )
        self.wafermap_spatial( perc_pinned , 'perc_pinned_full{}'.format(suffix) , [0,100] )
        
        # Let's define some uniformity metrics.
        analyses = [('pinned_low' ,perc_low),
                    ('pinned_high',perc_high),
                    ('pinned'     ,perc_pinned) ]
        
        for name, array in analyses:
            # First apply for all superpixels, then apply for all superpixels above given min_threshold.
            self.metrics['local_{}_all{}'.format(name,suffix)] = self.calc_metrics_from_array(array, not_reference)
            
            valid_mask = np.logical_and( not_reference, array > min_threshold )
            self.metrics['local_{}{}'.format(name,suffix)] = self.calc_metrics_from_array( array, valid_mask )
            
    def wafermap_spatial_metric( self, metric ):
        data    = self.safe_load( metric )
        m       = metric.lower()
        figname = metric
        #if 'localstd' in metric:
        #    clims   = self.local_std_lims( m )
        #else:
        clims = self.select_lims( m )
        dr = max(data.shape[0]/600,1)
        dc = max(data.shape[1]/800,1)
        self.wafermap_spatial( data[::dr,::dc], metric, clims )

    def wafermap_spatial( self , array , figname , clims=None, transpose=True, cmap=None ):
        # Determine proper image size for data
        if transpose:
            array = array.T
            
        if cmap:
            cm = cmap
        else:
            cm = matplotlib.rcParams['image.cmap']
            
        w,h = matplotlib.figure.figaspect( array )
            
        fig = plt.figure( figsize=(w,h) )
        ax  = fig.add_subplot(111)
        plt.axis  ( 'off' )
        if clims==None:
            plt.imshow( array , origin='lower' , aspect='equal' , interpolation='nearest', cmap=cm )
        else:
            plt.imshow( array , origin='lower' , aspect='equal' , interpolation='nearest', clim=clims, cmap=cm )
            
        # Set extent of image to exactly that of the chip image
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig( os.path.join( self.output_dir , '%s_wafermap.png' % figname ) , bbox_inches=extent )
        plt.close  ( )
        
    def select_offset( self, filename=None ):
        ''' Select the appropriate offset file '''
        if filename is None:
            if self.chiptype.series == 'pgm':
                pixfile = os.path.join( self.caldir , 'piximage.dat' )
            else:
                pixfile = os.path.join( self.caldir , 'PixImage0.dat' )
            if not os.path.exists( pixfile ):
                pixfile = os.path.join( self.caldir , 'PixImage2.dat' )
            if not os.path.exists( pixfile ):
                pixfile = os.path.join( self.caldir , 'PixImage3.dat' )
        else:
            pixfile = filename
        if not os.path.exists( pixfile ):
            if allow_missing:
                print( 'File does not exist' )
            else:
                raise IOError( 'File (PixImage*.dat) does not exist' )
        return pixfile

    def load_offset_rc( self, filename=None ):
        ''' Load the row and colums from the offset file header '''
        pixfile = self.select_offset( filename )
        if os.path.exists( pixfile ):
            with open( pixfile , 'r' ) as f:
                hdr  = np.fromfile (f, count=4, dtype=np.dtype('<u4'))
                cols = hdr[2]
                rows = hdr[3]
            return rows, cols
        else:
            print( 'ERROR: Pixel offset file does not exist.' )
            return None, None

    def load_offset( self, mask=False, filename=None, DR=None, pinned_low_cutoff=500, pinned_high_cutoff=15883, raw=False ):
        """
        Opens up pixel offset files for calibration data.
        Please use chip name [ "314R" , "P0" , "P2" ] for the chiptype input rather than '900' or 'P1.2.18'
        NOTE: This code writes out offset as a float, in units of mV, standard for each chip type.
        UPDATE 2016-05-02: Calibration actually occurs with 300 mV DR, not 400. May need to use DynamicRangeActual value.
        """
        img            = np.zeros( 0 )
        total_pix      = 0
        pixel_low      = 0
        pixel_high     = 0
        pixel_in_range = 0

        # If DR is not specified, pull from the chiptype table
        if DR is None:
            if self.dr is None:
                DR = self.chiptype.dynamic_range
            else:
                DR = self.dr
        self.offset_lims = [0,DR]

        pixfile = self.select_offset( filename )

        if os.path.exists( pixfile ):
            self.pixfile = os.path.basename( pixfile )
            with open( pixfile , 'r' ) as f:
                if self.chiptype.series == 'pgm':
                    img = np.fromfile( f, count=self.rows*self.cols, dtype=np.dtype('>H')).reshape( self.rows , self.cols , order='C')
                    if raw:
                        self.offset = img
                        return img
                    # Not sure if the refpix finder works with PGM . . . 25 Jan 2017 - PW
                    # TODO: In merging multiple code bases, it seems like w are repeating pin finding.
                    #       May want to clean up
                    self.find_pinned( img, low=pinned_low_cutoff, high=pinned_high_cutoff )  
                    self.offset = np.array( img , dtype=float ) / 69.

                    # Calculate pinned pixels
                    rows,cols      = img.shape
                    total_pix      = float(rows) * float(cols)
                    pixel_low      = (img < 10).sum()
                    pixel_high     = (img > 16373).sum()
                    pixel_in_range = total_pix - pixel_low - pixel_high

                elif self.chiptype.series == 'proton':
                    hdr  = np.fromfile (f, count=4, dtype=np.dtype('<u4'))
                    cols = hdr[2]
                    rows = hdr[3]
                    self.check_chip_rc( rows, cols )
                    img  = np.fromfile (f, count=rows*cols, dtype=np.dtype('<u2')).reshape (rows, cols, order='C')
                    if raw:
                        self.offset = img
                        return img
                    self.find_pinned( img, low=pinned_low_cutoff, high=pinned_high_cutoff )  
                    
                    # Calculate pinned pixels
                    total_pix      = float(rows) * float(cols)
                    pixel_low      = (img < 10).sum()
                    pixel_high     = (img > 16373).sum()
                    pixel_in_range = total_pix - pixel_low - pixel_high
                    
                    img = np.array( img, dtype=float )
                    self.offset = img * float( DR ) / ( 4. * 4092. )
                    
                else:
                    print( 'ERROR: Unknown chip type supplied.' )

            # Set pinned pixel metrics as 'offset' metrics
            self.metrics['offset_total_pix']      = total_pix
            self.metrics['offset_pixel_low']      = pixel_low
            self.metrics['offset_pixel_high']     = pixel_high
            self.metrics['offset_pinned']         = pixel_low + pixel_high
            self.metrics['offset_pixel_in_range'] = pixel_in_range

            try:
                if mask:
                    arrmask = self.load_arrmask()
                    if arrmask is not None:
                        self.offset[~arrmask] = 0
            except AttributeError:
                pass
        else:
            print( 'ERROR: Pixel offset file does not exist.' )
        return None
    
    def load_noise( self, mask=False, filename=None, pgm=False ):
        """
        Adapted from cal.py in order to simply load data from calibration noise .dat file for post processing.

        if mask=True, then false values in chiptype.flowcell will be set to 0
        """
        if filename is None:
            ### 2017-01-25 PW -- This isn't set up to work with PGM?!?!  Need to fix!!  What about gain correction?!
            pgm = False
            fpath = os.path.join( self.caldir , 'rmsimage.dat' )  # PGM
            if os.path.exists( fpath ):
                pgm = True
            else:
                fpath = os.path.join( self.caldir , 'NoisePic0.dat' )
            if not os.path.exists( fpath ):
                fpath = os.path.join( self.caldir , 'NoisePic3.dat' ) 
        else:
            fpath = filename

        if not os.path.exists( fpath ):
            if allow_missing:
                print('File does not exist!')
                return None
            else:
                raise IOError( 'File (NoisePic*.dat) does not exist' )

        self.noisefile = os.path.basename( fpath )

        if pgm:
            # Right now I am going to assume that pgm noise data is gain corrected . . . Phil 14 Aug 2017
            
            # data types
            dt_be = np.dtype ('>H')  # big-endian unsigned short
            dt_fl = np.dtype (float) # python-compatible floating point number

            # get noise data depending on chip type
            f      = open( fpath , 'r' )
            hdr    = np.fromfile (f, count=4, dtype=np.dtype('>u4'))
            flag   = int( 1 )
            rows   = hdr[2]
            cols   = hdr[3]
            img_fc = np.fromfile (f, count=rows*cols, dtype=np.dtype('>f4')).reshape (rows, cols, order='C')

            # process noise data
            img_fc  = np.asarray (img_fc, dtype=dt_fl)
            img_fc  = (img_fc * 1000) / 69.
            f.close ( )
        else:
            f       = open( fpath , 'r' )
            hdr     = np.fromfile (f, count=4, dtype=np.dtype('<u4'))
            flag    = hdr[1]
            cols    = hdr[2]
            rows    = hdr[3]
            self.check_chip_rc( rows, cols )
            img_fc  = np.fromfile (f, count=rows*cols, dtype=np.dtype('<u2')).reshape (rows, cols, order='C')
            img_fc  = np.asarray (img_fc, dtype=np.dtype('<u2'))
            f.close ( )
        
        # Gain correct noise if needed. A non-zero flag means that it was saved as gain corrected uV.
        #  print('Flag = %s' % flag )
        if int(flag) == 0:
            # Do gain correction, by pixel.  If gain isn't loaded already, do that first.
            print('Now gain correcting the noise (by pixel)...')
            if not hasattr( self , 'gain' ):
                try:
                    self.load_gain( mask=mask)
                    loaded_gain = True
                    # Gain scale is a scaling factor to take gain back to unitless gain.
                    # By default, we use mV/V gain for more granularity.
                    gain_scale = 1000.
                except:
                    self.noise = img_fc
                    print( 'Could not figure out gain or it does not exist.  Noise is not gain-corrected.' )
                    return None
            else:
                loaded_gain = False
                
            if self.gain_units == 'V/V':
                gain_scale = 1.
            # Do gain correction and clear gain from memory if it was loaded in this function
            if hasattr( self, 'gain' ):
                img_fc /= (self.gain / gain_scale)
                if loaded_gain:
                    self.close_gain()
            else:
                self.noise = img_fc
                print( 'Could not figure out gain or it does not exist.  Noise is not gain-corrected.' )
                return None
                
        elif int(flag) == 1:
            print('Proceeding as usual, this noise is already gain-corrected.')
            
        self.noise = img_fc

        # Apply the array mask
        if mask:
            arrmask = self.load_arrmask()
            if arrmask is not None:
                self.noise[~arrmask] = 0

        return None

    def mask_refpix( self, metric ):
        '''
        Mask out noise pixels where offset is in the top or bottom 25% of the dac window.
        These pixels likely pinned at some point and should be excluded
        ''' 
        offset_min = ( (2**14)/4 ) * self.chiptype.uvcts/1000.
        offset_max = offset_min * 3
        regions = [ '', '_lower_left', '_lower_right', '_upper_left', '_upper_right' ]
        fields = [ 'refpix_%s%s', 'refpix_%s%s_lower_left', 'refpix_%s%s_lower_right', 'refpix_%s%s_upper_left', 'refpix_%s%s_upper_right' ]
        metric_fields = [ 'refpix_%s%s%s'   % ( metric,   ''       , r ) for r in regions ]
        offset_fields = [ 'refpix_%s%s%s'   % ( 'offset', ''       , r ) for r in regions ]
        save_fields   = [ 'refpix_%s%s%s'   % ( metric,   '_masked', r ) for r in regions ]
        masked_fields = [ 'masked_refpix%s' % ( r ) for r in regions ]
        for ind in range( len( fields ) ):
            data   = getattr( self, metric_fields[ind] )
            offset = getattr( self, offset_fields[ind] )
            keep   = ( offset > offset_min ) & ( offset < offset_max )
            masked = data[ keep ]
            self.metrics[ masked_fields[ind] ] = len( keep ) - keep.sum()
            print( '{}: {}'.format( masked_fields[ind], self.metrics[ masked_fields[ind] ] ) )
            setattr( self, save_fields[ind], masked )

        lims = [ 0, getattr( self, metric_fields[0] ).size/4 ]   # Limit is number of the pixels in each quadrant
        self.masked_refpix_lims = lims

    def close_gain( self ):
        del self.gain
        return None
    
    def close_offset( self ):
        del self.offset
        return None
    
    def close_noise( self ):
        del self.noise
        return None

    def column_average( self , data , start=3 , end=9 , shave=20 ):
        """ Generic algorithm that will spit out points, colavg_mean, and colavg_sd """
        fcblockC = self.cols / 12
        roi      = data[ : , (start * self.cols/12):(end * self.cols/12) ]
        colavg   = ma.masked_array( roi , (roi == 0) ).mean( 1 )[shave:-shave].data
        
        # Trim it to a multiple of 100 rows
        l       = 100
        N       = len(colavg)
        mod     = N % l
        if mod == 0:
            block  = colavg.reshape(-1,l)
        else:
            block  = colavg[ mod/2:-mod/2 ].reshape(-1,l)
            
        # Save the data to the class, and then to file.
        points    = np.arange( l/2 , (N - mod) , l ) - (N - mod)/2
        mean_data = ma.masked_array( block,(block==0) ).mean( 1 ).data
        sd_data   = ma.masked_array( block,(block==0) ).std(  1 ).data
        
        return points , mean_data , sd_data , colavg
    
    def colavg( self , metric , start=3 , end=9 , shave=20 , save=True ): # Changed shave to 20 (from 10) to match edge.boil
        ''' 
        gets column average of the metric of choice.  
        Note that this is just barely different from typical ECC 'edge' class . . .
        '''
        data = self.safe_load( metric )
        m    = metric.lower()
        units= self.get_units( metric )
        
        # Calculate raw column average
        self.colavg_pts , avg , sd , raw = self.column_average( data , start , end , shave )
        setattr( self , 'raw_%s_colavg' % m , raw )
        setattr( self , '%s_colavg'    % m  , avg )
        setattr( self , '%s_colavg_sd' % m  , sd  )
        
        if save:
            if 'localstd' in metric:
                self.save_dat( self.colavg_pts, 'col_avg_coords_lstd' )
            else:
                self.save_dat( self.colavg_pts, 'col_avg_coords' )
            self.save_dat( avg, 'col_avg_%s' % m )
            self.save_dat( sd, 'col_std_%s' % m )
           
        return None
    
    def plot_colavg( self , metric , errorbar_plot=False ):
        """ plots column average data """
        m = metric.lower()
        if not hasattr( self , '%s_colavg' % m ):
            self.colavg( m )
        units = self.get_units( metric )
            
        # Pull in data
        mean_data = getattr( self , '%s_colavg'    % m )
        sd_data   = getattr( self , '%s_colavg_sd' % m )
        
        if not errorbar_plot:
            # Make figure
            plt.figure  ( )
            plt.plot    ( self.colavg_pts , mean_data , 'o--' , color='green' )
            plt.title   ( '%s column average' % m.title() )
            plt.ylabel  ( '%s (%s)' % (m.title() , units) )
            plt.xlabel  ( 'Rows from center' )
            plt.grid    ( )
            plt.savefig ( os.path.join( self.output_dir , '%s_colavg.png' % m ) )
            plt.close   ( )
        else:
            # Make figure with errorbars
            plt.figure  ( )
            plt.errorbar( self.colavg_pts , mean_data , yerr=sd_data , fmt = 'o:' )
            plt.title   ( '%s column average' % m.title() )
            plt.ylabel  ( '%s (%s)' % (m.title() , units) )
            plt.xlabel  ( 'Rows from center' )
            plt.grid    ( )
            plt.savefig ( os.path.join( self.output_dir , '%s_colavg_errorbar.png' % m ) )
            plt.close   ( )
            
        return None
    
    def diff_img( self , metric , negate=False , rcdist = ( 1000 , 1000 ) ):
        """
        Takes difference of the metric with the mean of the chip center.
        This is useful for detecting edge effects and subtle variations across the chip.
        """
        data = self.safe_load( metric )
        m    = metric.lower()
        c    = [ int(self.rows / 2) , int(self.cols / 2) ]
        cavg = data[ c[0]-rcdist[0] : c[0]+rcdist[0] , c[1]-rcdist[1] : c[1]+rcdist[1] ].mean()
        diff = data - cavg
        if negate:
            diff *= -1
            setattr( self , '%s_diff_lims' % m, ( 0 , 100 ) )
            
        # Set mask for each data type to ignore edges
        lims = getattr( self , '%s_lims' % m )
        msk  = np.logical_and( data > lims[0] , data < lims[1] )
        img  = stats.block_avg( diff , self.blockR , self.blockC , msk )
        setattr( self , '%s_diff' % m , img )
        
        return None
    
    def plot_diff_img( self , metric , lims=[] ):
        """
        test
        """
        m = metric.lower()
        if not hasattr( self , '%s_diff' % m ):
            self.diff_img( m )
        data = getattr( self , '%s_diff' % m )
        dlim = getattr( self , '%s_diff_lims' % m )
        
        if lims == []:
            lims = dlim
            
        #r , c = self.rows / self.blockR , self.cols / self.blockC
        #xdivs , ydivs = np.arange( 0 , (c+1) , c/4 ) , np.arange( 0 , (r+1) , r/4 )
            
        plt.figure   ( )
        plt.imshow   ( data , origin='lower' , interpolation='nearest' , clim = lims , cmap=plt.cm.RdYlBu )
        plt.colorbar ( )
        plt.title    ( '%s difference image' % m.title() )
        #plt.xticks   ( xdivs / 6 , [ str(x) for x in xdivs ] )
        #plt.yticks   ( ydivs / 6 , [ str(y) for y in ydivs ] )
        plt.savefig  ( os.path.join( self.output_dir , '%s_diff_img.png' % m ) )
        plt.close    ( )
        return None

    def superpixel_safe_load( self, metric, sp_type, ignore_pinned=False, finite=False, hd=False, ignore_refpix=False ):
        """ 
        Helper function to safely load superpixel analysis functions. 
        If you are clever and called superpixel_analysis in the right way, this will never be needed.
        
        metric is the high level metric: noise, offset, gain, etc.
        sp_type is the type of analysis, like localavg or localstd
        """
        prefix = ''
        if ignore_pinned:
            prefix = 'true_'
            
        full_metric = '{}_{}{}'.format( metric.lower(), prefix, sp_type.lower() )
        if hd:
            full_metric += '_hd'

        # This line is for the wacky local_avg and diff_img_hd version of things
        if (not ignore_pinned) and ignore_refpix:
            full_metric += '_no_refpix'
            
        try:
            data = getattr( self, full_metric )
        except AttributeError:
            # Calculate the metric we need.
            self.superpixel_analysis( metric, ignore_pinned=ignore_pinned, finite=finite, hd=hd, ignore_refpix=ignore_refpix )
            data = getattr( self, full_metric )
            
        return data
            
    def diff_img_hd( self, metric, lims=[], ignore_pinned=False ):
        """ 
        An improved, higher def, and zoomed-in difference image. Multiple analyses happen here. 
        For historical reasons, this ignores pinned pixels by default.  
        """
        m     = metric.lower()
        units = self.get_units( m )
        
        # Add the true prefix for when we ignore pinned pixels like we should have been doing all along
        prefix = ''
        if ignore_pinned:
            prefix = 'true_'
        
        avg   = self.superpixel_safe_load( m, 'localavg', ignore_pinned=ignore_pinned, hd=True, ignore_refpix=True )
        diff  = avg - avg[avg > 0].mean()
        f_avg = ndimage.median_filter( diff , size=(5,5) ) # Filter the data to make it clearer
        
        if not lims:
            initial = getattr( self , '{}_diff_lims'.format(m) )
            lims    = [ initial[0]/2, initial[1]/2 ]
            
        # Make spatial map
        fname = '{}{}_diff_img_hd'.format( prefix, m )
        plt.figure( )
        plt.imshow( f_avg , origin='lower', interpolation='nearest', clim=lims, cmap='seismic' )
        plt.colorbar( shrink=0.7 )
        plt.title   ( '{} Difference Image (HD) [{}]'.format( m.title() , units ) )
        plt.savefig ( os.path.join( self.output_dir , '{}.png'.format( fname ) ) )
        plt.close   ( )
        
        # Make wafermap spatial
        self.wafermap_spatial( f_avg , figname=fname , clims=lims , cmap='seismic' )
        
        # If multilane, let's make those multilane spatials
        if self.is_multilane:
            # Omit transpose -- default is true which is what we want.
            self.custom_multilane_wafermap_spatial( f_avg, fname, clims=lims, cmap='seismic' )
        
        # Make histogram
        valid = f_avg[f_avg > lims[0]]
        
        plt.figure ( )
        plt.hist   ( valid , bins=np.linspace( lims[0], lims[1], (lims[1]-lims[0])+1 ) )
        plt.xlabel ( '{} ({})'.format( m, units ) )
        plt.title  ( '{} Difference Image (HD)'.format( m.title() ) )
        plt.savefig( os.path.join( self.output_dir, '{}{}_diff_img_hd_hist.png'.format( prefix, m ) ) )
        plt.close  ( )
        
        # Save some spatial variability metrics
        percs  = stats.percentiles( valid )
        m80    = percs['m80']
        std    = valid.std()
        
        # Define a metric that is percent of superpixels outside a fixed range.  Defining as +/- 20% of lims
        outlim = (lims[1] - lims[0]) / 5.
        p_out  = 100. * np.logical_or( valid > outlim , valid < -outlim ).sum() / valid.size
        
        self.metrics['{}{}_spatial_var_m80'.format(prefix,m)]      = m80
        self.metrics['{}{}_spatial_var_std'.format(prefix,m)]      = std
        self.metrics['{}{}_spatial_var_perc_out'.format(prefix,m)] = p_out
        
    def get_units( self , metric ):
        """ Spits out units by metric """
        m = metric.lower()
        return getattr( self , '%s_units' % m , '' )

    def histogram( self , metric, mask=None, rescale_ylim=False ):
        """ 
        Plots typical histogram of data as normally seen in ECC data 
        
        NEW!  rescale_ylim will ignore the edge 5 bins on both sides and renormalize the histogram to show the primary distribution of pixels
              This eliminates an oft-seen stack of the next-to-pinned-high bins that can swamp multilane chip offset histograms.
        """
        data = self.safe_load( metric )
        m    = metric.lower()
        units= self.get_units( metric )
            
        lims = self.select_lims( m )
        # Apply mask
        if mask is None:
            mask = data > np.ones( data.shape, dtype=np.bool )
        mask = np.logical_and( mask, data > lims[0] )
        mask = np.logical_and( mask, data < lims[1] )
        #mask = np.logical_and( data > lims[0] , data < lims[1] )
        vals  = data[ mask ]
        
        # Make the plot
        hn,hb,_ = plt.hist( vals , bins = np.linspace( lims[0] , lims[1] , 101 ) )
        self.metrics['%s_mode' % m] = stats.hist_mode( hn , hb )
        plt.xlim   ( lims[0] , lims[1] )
        plt.title  ( self.create_title( metric ) )
        plt.xlabel ( '%s (%s)' % ( m.title() , units ) )
        plt.ylabel ( 'Number of pixels' )
        
        if rescale_ylim:
            # trim off bin heights to get bin max, then round up to next nearest 100k 10% higher
            max_counts   = hn[5:-5].max()
            nearest_100k = 100000. * np.ceil( 1.1 * max_counts / 100000. )
            plt.ylim( 0, nearest_100k )
        
        if m == 'offset':
            plt.savefig( '%s/pix_%s_histogram.png' % ( self.output_dir , m ) )
        else:
            plt.savefig( '%s/%s_histogram.png' % ( self.output_dir , m ) )
        plt.close  ( )
        return None
        
    def low_gain_histogram( self , lims=[0,500] ):
        """ Plots typical histogram of data as normally seen in ECC data """
        data = self.safe_load( 'gain' )
        m    = 'gain'
        units= self.get_units( m )
        
        # Apply mask
        mask = np.logical_and( data > lims[0] , data < lims[1] )
        vals = data[ mask ]
        
        # Make the plot
        hn,hb,_ =  plt.hist( vals , bins = np.linspace( lims[0] , lims[1] , 101 ) )
        plt.xlim   ( lims[0] , lims[1] )
        plt.title  ( 'Low Pixel Electrical Gain Histogram' )
        plt.xlabel ( '%s (%s)' % ( m.title() , units ) )
        plt.ylabel ( 'Number of pixels' )
        plt.savefig( '%s/low_gain_histogram.png' % ( self.output_dir ) )
        plt.close  ( )
        return None
        
    def histogram2d( self , metricx, metricy, filename=None ):
        """ Plots typical histogram of data as normally seen in ECC data """
        xdata  = self.safe_load( metricx )
        mx     = metricx.lower()
        unitsx = self.get_units( metricx )
        ydata  = self.safe_load( metricy )
        my     = metricy.lower()
        unitsy = self.get_units( metricy )
            
        limsx = getattr( self , '%s_lims' % mx )
        limsy = getattr( self , '%s_lims' % my)
        xbins = np.linspace( limsx[0], limsx[1], 101 )
        ybins = np.linspace( limsy[0], limsy[1], 101 )
        mask = np.logical_and( limsx[0] < xdata , xdata < limsx[1] )
        mask = np.logical_and( mask, limsy[0] < ydata )
        mask = np.logical_and( mask, ydata < limsy[1] )
        
        # Make the plot
        h, x, y = np.histogram2d( xdata[mask].flatten(), ydata[mask].flatten(), bins=( xbins, ybins ) )
        extent = [ xbins[0], xbins[-1], ybins[0], ybins[-1] ]
        plt.imshow( h, extent=extent, origin='lower', aspect='auto' )
        plt.xlabel ( '%s (%s)' % ( mx.title() , unitsx ) )
        plt.ylabel ( '%s (%s)' % ( my.title() , unitsy ) )
        if filename:
            plt.savefig( filename )
        else:
            plt.savefig( '%s/%s_vs_%s_2dhistogram.png' % ( self.output_dir , mx, my ) )
        plt.close()

        counts = h.sum( axis=0 )
        mask   = counts > counts.sum()/(len(x)*2)
        peaks  = h.argmax( axis=0 )
        x = (x[1:]+x[:-1])/2.
        y = (y[1:]+y[:-1])/2.
        X = x[mask]
        Y = y[peaks][mask]
        return X, Y
        
    def block_reshape( self, data, blocksize ):
        """ THIS SHOULD NOT BE USED!  Use reshaping tools in the average module instead. """
        rows, cols = data.shape
        numR = rows/blocksize[0]
        numC = cols/blocksize[1]
        return data.reshape(rows , numC , -1 ).transpose((1,0,2)).reshape(numC,numR,-1).transpose((1,0,2))
    
    def masked_block_reshape( self, data, blocksize, ignore_mask ):
        """ 
        THIS SHOULD NOT BE USED!  Use reshaping tools in the average module instead.
        
        A function similar to block reshape but that's useful in finding localstd without pinned pixels.
        A key example would be sneaky clusters.
        """
        rows, cols = data.shape
        masked = ma.masked_array( data , mask=ignore_mask )
        numR = rows/blocksize[0]
        numC = cols/blocksize[1]
        return masked.reshape(rows , numC , -1 ).transpose((1,0,2)).reshape(numC,numR,-1).transpose((1,0,2))
    
    def analyze_sneaky_clusters( self , threshold=1.5 ):
        ''' 
        This function works independently of the find_pinned method, which relies on the "find_refpix" function.
        Why?  Because that was designed with a properly functioning chip in mind and won't work well with sneaky clusters as 
          they will be interpreted as reference pixels.
        
        This was carefully developed for non-HD data, so it will not be upgraded and made to work with HD functionalities.
        '''
        def find_bad_superpix( pin_perc , localstd , perc_gt=7.5 , std_lt=70 ):
            return np.logical_and( pin_perc > perc_gt , localstd < std_lt ).sum()
        
        block       = [self.chiptype.miniR,self.chiptype.miniC]
        pinned_low  = self.offset < threshold
        pinned_high = self.offset > (self.offset_lims[1] - threshold)
        any_pinned  = np.logical_or( pinned_low , pinned_high )
        pinlowperc  = 100. * stats.block_avg( pinned_low.astype( float ), block[0], block[1] )
        localstd    = stats.block_std( self.offset, block[0], block[1], goodpix=np.logical_not( any_pinned ) )
        
        # Add a metric for percentage of bad superpixels and then include in the plot
        bad_superpixels = find_bad_superpix( pinlowperc , localstd )
        self.metrics[ 'sneaky_superpixel_count' ] = int( bad_superpixels )
        
        # Create alternate metric matrix for future-proofing of metric.
        perc_pinned_thresholds   = [0.5,1,2,4,6,7.5,10,12.5,15,17.5,20]
        offset_local_thresholds  = [40,50,55,60,65,70,75,80,85,90,100]
        sneaky_superpixel_matrix = []
        for perc in perc_pinned_thresholds:
            bad = [ find_bad_superpix( pinlowperc , localstd , perc , std ) for std in offset_local_thresholds ]
            sneaky_superpixel_matrix.append( bad )
        self.metrics[ 'sneaky_superpixel_matrix' ] = sneaky_superpixel_matrix
        self.metrics[ 'perc_pinned_thresholds'   ] = perc_pinned_thresholds
        self.metrics[ 'offset_local_thresholds'  ] = offset_local_thresholds
        
        # Make plot with limits
        plt.plot( localstd.flatten() , pinlowperc.flatten() , 'o' )
        plt.xlim( 0 , 120  )
        plt.ylim( 0 , 30   )
        plt.axvline( 70 , color='red' , linestyle='-' )
        plt.axhline( 7.5 , color='red' , linestyle='-' )
        plt.axvspan( 0 , 70 , 0.25 , 1 , color='red' , alpha=0.2 )
        plt.text( 10, 28 , 'Sneaky\ncluster\nsuperpixels' , va='center' , ha='center' , fontsize=10 ,
                  weight='semibold' , color='red' )
        plt.xlabel( 'Local Offset Standard Deviation, Excluding Pinned Pixels (mV)' )
        plt.ylabel( '% Pinned Pixels per SuperPixel' )
        plt.title ( 'Number of Bad SuperPixels: {}'.format( bad_superpixels ) )
        plt.savefig ( os.path.join( self.output_dir , 'sneaky_cluster_plot.png' ) )
        plt.close   ( )

        self.offset_localstd_nopins      = localstd
        self.offset_localstd_nopins_lims = self.offset_localstd_lims
        self.calc_metrics( 'offset_localstd_nopins' , ignore_upper_limit=True )
        self.spatial( 'offset_localstd_nopins' )

        plt.figure  ( )
        plt.imshow  ( np.array(sneaky_superpixel_matrix), origin='lower', interpolation='nearest', clim=[0,2500] )
        plt.colorbar( )
        plt.xticks  ( range(11) , offset_local_thresholds )
        plt.yticks  ( range(11) , perc_pinned_thresholds  )
        plt.xlabel  ( 'Non-Pinned Offset Localstd Threshold (<)' )
        plt.ylabel  ( '% Pinned Pixels per SuperPixel Threshold (>)' )
        plt.title   ( 'Bad SuperPixels vs. Sneaky Pixel Thresholds' )
        plt.savefig ( os.path.join( self.output_dir , 'sneaky_clusters_bad_superpixels.png' ) )
        plt.close   ( )
        
    def local_avg( self, metric, hd=True, ignore_pinned=False ):
        """ calculates and plots the local average of the metric on miniblocks (or micro) """
        m = metric.lower()
        units = self.get_units( metric )
        
        # Use new fixed limits.
        mlims= { 'noise' : [50,250] , 'gain' : [900,1100] , 'offset' : self.offset_lims }
        lims = mlims[metric]
        
        prefix = ''
        if ignore_pinned:
            prefix = 'true_'
            
        suffix = ''
        if hd:
            suffix = '_hd'
            
        avg   = self.superpixel_safe_load( m, 'localavg', ignore_pinned=ignore_pinned, hd=hd, ignore_refpix=True )
        f_avg = ndimage.median_filter( avg , size=(5,5) ) # Filter the data to make it clearer
        
        # Save to file
        f_avg.tofile( os.path.join( self.output_dir ,
                                    '{}{}_local_avg_{}_{}.dat'.format( prefix, m, f_avg.shape[0], f_avg.shape[1] ) ))
        
        figname = '{}{}_local_avg{}'.format( prefix, m, suffix )
        plt.figure  ( )
        plt.imshow  ( f_avg , origin='lower', interpolation='nearest', clim=lims )
        plt.colorbar( shrink=0.7 )
        plt.title   ( '{} Local Average ({})'.format( m.title(), units ) )
        plt.savefig ( os.path.join( self.output_dir , '{}.png'.format( figname ) ) )
        plt.close   ( )
        
        self.wafermap_spatial( f_avg , figname , clims=lims )
        
    def local_std( self, metric, ignore_pinned=False, hd=False ):
        """ 
        Calculates and plots the local standard deviation on the minblocks 
        Note that this by default still includes pinned pixels in the metrics and does non-HD analysis.
        """
        metric = metric.lower()
        prefix = ''
        if ignore_pinned:
            prefix = 'true_'
            
        full_metric = '{}_{}localstd'.format( metric, prefix )
        if hd:
            full_metric += '_hd'
        
        units = self.get_units( metric )
        clim  = self.local_std_lims( metric )
        
        # Safely load the superpixel data
        std = self.superpixel_safe_load( metric, 'localstd', ignore_pinned=ignore_pinned, hd=hd )
        
        setattr( self, '%s_localstd_lims' % metric, clim )
        self.calc_metrics( full_metric , ignore_upper_limit=True ) # From PixelUniformity 2/20/17
        self.histogram( full_metric )
        
        self.spatial( full_metric )
        self.wafermap_spatial_metric( full_metric )
        
        # Make custom column average plot
        rows,cols = std.shape
        if self.is_multilane:
            masked = ma.masked_array( std, mask=(std==0) )
            avg = masked.mean(1)
        else:
            mid = cols/2
            avg = std[:,mid-10:mid+10].mean(1)
        setattr( self , '%s_localstd_trace' % metric , avg )
        
        plt.figure  ( )
        # Throw out one row from top and bottom.
        plt.plot    ( np.arange( rows )[1:-1] , avg[1:-1] , 'o--' , color='green' )
        plt.title   ( '%s Local Standard Deviation Colavg' % metric.title() )
        plt.ylabel  ( '%s localstd (%s)' % ( metric.title() , units ) )
        plt.xlabel  ( 'Row' )
        plt.grid    ( )
        plt.savefig ( os.path.join( self.output_dir , '{}_colavg.png'.format( full_metric ) ) )
        plt.close   ( )
        
    def local_std_lims( self, metric ):
        ''' Get the clim for the localstd version of the metric '''
        m = metric.lower()
        lims = self.select_lims( m )
        clim = [ 0, (lims[1]-lims[0])/5 ]
        return clim
    
    def superpixel_analysis( self, metric, ignore_pinned=False, finite=False, hd=False, ignore_refpix=False ):
        """
        This high-level function will perform all BlockMath calculations necessary for a given metric, e.g. gain, 
        in an efficient manner by only doing the reshape operations once.
        
        Note that here the default is the historical default case that includes pinned pixels and is non-HD.
        It is highly recommended by Phil to move towards ignoring pinned pixels and making HD arrays.
        
        Also, when ignoring pinned, we are going to ignore pixels that are not active (self.refpix).  This is the way of the future
        
        For compatability with local_avg and diff_img_hd, ignore_refpix=True can be set with ignore_pinned=False
        """
        m    = metric.lower()
        data = self.safe_load( m )
        
        if ignore_pinned:
            if not hasattr( self, 'refpix' ):
                self.safe_load( 'gain' )
                self.find_refpix( )
                
            prefix  = 'true_'
            pinned  = np.logical_or( self.pinned_low, self.pinned_high )
            goodpix = np.logical_not( np.logical_or( pinned, self.refpix ))
            bm = BlockMath.from_chipcal( self, data, goodpix=goodpix, finite=finite, hd=hd )
        else:
            # This is the old way of doing it and we will keep doing it in parallel for historical comparison
            # It includes refpix for reasons I don't know/remember, but will not stop doing this anytime soon
            #
            # The exceptions are the local_avg and diff_img_hd functions which ignore_refpix
            prefix = ''
            if ignore_refpix:
                if not hasattr( self, 'refpix' ):
                    self.safe_load( 'gain' )
                    self.find_refpix( )
                    
                goodpix = np.logical_not( self.refpix )
                bm = BlockMath.from_chipcal( self, data, goodpix=goodpix, finite=finite, hd=hd )
            else:
                # This is the default for the old way of doing things for localstd metrics
                bm = BlockMath.from_chipcal( self, data, finite=finite, hd=hd )
            
        # Save the requisite arrays
        if (not ignore_pinned) and ignore_refpix:
            # This is the one-off case for local_avg
            if hd:
                setattr( self, '{}_localavg_hd_no_refpix'.format( m ), bm.get_mean())
                setattr( self, '{}_localstd_hd_no_refpix'.format( m ), bm.get_std() )
            else:
                setattr( self, '{}_localavg_no_refpix'.format( m ), bm.get_mean())
                setattr( self, '{}_localstd_no_refpix'.format( m ), bm.get_std() )
        else:
            if hd:
                setattr( self, '{}_{}localavg_hd'.format( m, prefix ), bm.get_mean())
                setattr( self, '{}_{}localstd_hd'.format( m, prefix ), bm.get_std() )
            else:
                setattr( self, '{}_{}localavg'.format( m, prefix ), bm.get_mean())
                setattr( self, '{}_{}localstd'.format( m, prefix ), bm.get_std() )
        
    def overlay_histogram( self , top_metric, bot_metric ):
        """ 
        Plots both histograms, overlaid 
        """
        all_lims = [ np.inf, -np.inf ]
        for metric in [ top_metric, bot_metric ]:
            data = self.safe_load( metric )
            m    = metric.lower()
            units= self.get_units( metric )
                
            lims = self.select_lims( m )
            all_lims[0] = min( all_lims[0], lims[0] )
            all_lims[1] = max( all_lims[1], lims[1] )
            vals = data[ np.logical_and( data > lims[0] , data < lims[1] ) ]
            
            # Make the plot
            hn,hb,_ = plt.hist( vals , bins = np.linspace( lims[0] , lims[1] , 101 ) )
            self.metrics['%s_mode' % m] = stats.hist_mode( hn , hb )

        plt.xlim   ( all_lims )
        plt.title  ( self.create_title( top_metric, bot_metric ) )
        plt.xlabel ( '%s, %s (%s)' % ( top_metric.title() , bot_metric.title() , units ) )
        plt.ylabel ( 'Number of pixels' )
        plt.savefig( '%s/%s_%s_histogram.png' % ( self.output_dir , top_metric.lower(), bot_metric.lower() ) )
        plt.close  ( )
        return None

    def spatial( self , metric ):
        """ Plots typical spatial plot of data as normally seen in ECC data """
        data = self.safe_load( metric )
        m    = metric.lower()
        lims = self.select_lims( m )
        
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

        
        # Make the plot
        #plt.imshow   ( data[::6,::6] , origin='lower' , interpolation='nearest', clim=lims )
        plt.imshow   ( data[::yscale,::xscale] , origin='lower' , interpolation='nearest', clim=lims, extent=extent )
        plt.title    ( self.create_title( metric ) )
        plt.xticks   ( xdivs , [ str(x) for x in xdivs ] )
        plt.yticks   ( ydivs , [ str(y) for y in ydivs ] )
        #plt.xticks   ( xdivs / 6 , [ str(x) for x in xdivs ] )
        #plt.yticks   ( ydivs / 6 , [ str(y) for y in ydivs ] )
        plt.colorbar ( )
        if m == 'offset':
            plt.savefig  ( '%s/pix_%s_spatial.png' % ( self.output_dir , m ) )
        else:
            plt.savefig  ( '%s/%s_spatial.png' % ( self.output_dir , m ) )
        plt.close    ( )
        return None

    def low_gain_spatial( self , lims=[0,500] ):
        """ Plots a gain spatial plot with focus on low (0-500 mV/V) gain pixels. """
        data = self.safe_load( 'gain' )
        m    = 'gain'
        
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
        
        # Make the plot
        plt.imshow   ( data[::yscale,::xscale] , origin='lower' , interpolation='nearest', clim=lims, extent=extent )
        plt.title    ( 'Low Pixel Electrical Gain Spatial Map' )
        plt.xticks   ( xdivs , [ str(x) for x in xdivs ] )
        plt.yticks   ( ydivs , [ str(y) for y in ydivs ] )
        plt.colorbar ( )
        plt.savefig  ( '%s/low_gain_spatial.png' % ( self.output_dir ) )
        plt.close    ( )
        return None

    def plot_gcoff( self ):
        """ Plots gain-corrected offset values....don't know if they are useful or not."""
        self.gcoff      = self.offset / self.gain * 1000.
        self.gcoff[ self.gain == 0 ] = 0

        # Arbitrarily set gcoff lims at 25% larger than offset lims.  Or not.
        # self.gcoff_lims = (0 , self.offset_lims[1] * 1.25)
        self.gcoff_lims = self.offset_lims
        
        self.calc_metrics( 'gcoff' , True )
        
        self.spatial  ( 'gcoff' )
        self.histogram( 'gcoff' )

    def safe_load( self , metric ):
        """
        oft-used block of code to load particular metrics if they haven't been already.
        """
        met = metric.lower()
        # Try to return the metric
        try:
            return getattr( self, met )
        except AttributeError:
            pass
        # Try to calculate the metric
        try:
            getattr( self, 'load_%s' % met )()
            return getattr( self, met )
        except AttributeError:
            raise NameError('ERROR! Unknown metric given.')
        
    def save_json( self , metric, flatten=False ):
        """ saves json metrics to file based on metric """
        met  = metric.lower()
        data = {}
        for key in [ x for x in self.metrics.keys() if met in x.lower() ]:
            data[key] = self.metrics[key]
        if flatten==True:
            data = misc.flatten_dict( data )
        misc.serialize( data )
        #data = { key : self.metrics[ key ] for key in [ x for x in self.metrics.keys() if met in x ] }
        if met == 'offset':
            fname = 'pix'
        else:
            fname = met
        f    = open( '%s/%s.json' % ( self.output_dir , fname ) , 'w' )
        json.dump  ( data , f )
        f.close    ( )
        return None

    def save_lane_json( self, update=True ):
        ''' write each lane to a json object, updating the json object if it alread exits '''
        for lane, vals in self.lane_metrics.items():
            fn = '{}/{}.json'.format( self.output_dir , lane.replace( '_', '' ) ) # string replace for ecc compatibility with fc_, tn_, lane1_, ...
            print( fn )
            if os.path.exists( fn ) and update:
                f      = open ( fn, 'r' )
                jsonin = json.load( f )
                f.close()
                # Use update function instead of looping through all metrics.
            else:
                jsonin = {}
            data = misc.flatten_dict( vals )
            # Clean up the json file to serialize it
            misc.serialize( data )
            jsonin.update( data )
            # Create json file and dump metrics into it.
            json.dump ( jsonin  , open( fn, 'w' ) )

    def save_dat( self, data, metric, path=None ):
        ''' Saves the dat file.  If path is note, uses self.output_dir '''
        # set the path
        if path is None:
            path = self.output_dir
        filename = os.path.join( path, '%s.dat' % metric )
        dp.write_dat( data, filename, metric )

    def set_refpix_props( self, metric, adjust=None, masked=False ):
        ''' Sets the limits and units for all reference pixel regions '''
        if metric == 'noise':
            lims = [ 0, 600 ]
        #if metric == 'noise':
        #    if adjust < 100:
        #        lims = [ 0, 200 ]
        #    elif adjust > 200:
        #        lims = [ 0, 1000 ]
        #    else:
        #        lims = [0, 600]
        else:
            lims = getattr( self, '%s_lims' % metric )

        if masked:
            tail = '_masked'
        else:
            tail = ''
        setattr( self, 'refpix_%s%s_lims' % ( metric, tail ), lims )
        setattr( self, 'refpix_%s%s_lower_left_lims'  % ( metric, tail ), lims )
        setattr( self, 'refpix_%s%s_lower_right_lims' % ( metric, tail ), lims )
        setattr( self, 'refpix_%s%s_upper_left_lims'  % ( metric, tail ), lims )
        setattr( self, 'refpix_%s%s_upper_right_lims' % ( metric, tail ), lims )

        # Set the units
        units = getattr( self, '%s_units' % metric )
        setattr( self, 'refpix_%s%s_units' % ( metric, tail ), units )
        setattr( self, 'refpix_%s%s_lower_left_units'  % ( metric, tail ), units )
        setattr( self, 'refpix_%s%s_lower_right_units' % ( metric, tail ), units )
        setattr( self, 'refpix_%s%s_upper_left_units'  % ( metric, tail ), units )
        setattr( self, 'refpix_%s%s_upper_right_units' % ( metric, tail ), units )

    @staticmethod
    def calc_metrics_from_array( data, valid_mask=None ):
        """ 
        Shorthand method for generating statistics from an array of data. 
        Meant for use outside of normal offset, gain, noise metrics.  For those, use self.calc_metrics().
        
        This eliminates the ridiculous previous convention of saving both p90 and P90 key duplicates in metrics.
        """
        if valid_mask is None:
            valid   = data.flatten()
            tb_diff = stats.top_bottom_diff( data )
        else:
            if valid_mask.shape == data.shape:
                valid = data[valid_mask]
                tb_diff = stats.top_bottom_diff( data, valid_mask )
            else:
                print( 'Input mask not of same shape as data!  Ignoring mask.' )
                print( 'data: {}'.format( data.shape ) )
                print( 'mask: {}'.format( mask.shape ) )
                valid = data.flatten()
                
        metric_dict = {}
        if valid.any():
            metric_dict = stats.named_stats( valid )
            metric_dict['tb_diff'] = tb_diff
            
        return metric_dict


class PI_noise: 
    """ class to support PI noise and the annotation of noisy column dacs. """
    def __init__( self , noise , badcols ):
        self.badcols = badcols
        self.valmask = np.logical_and( noise > 0 , noise < 400 )
        self.noise   = noise
        (self.rows , self.cols) = noise.shape
            
        # Define miniblocks
        if noise.shape[0] == 800 and noise.shape[1] == 1200:
            self.fc    = False
            self.miniR = 100
            self.miniC = 100
        elif noise.shape[0] == 666 and noise.shape[1] == 640:
            # P0 full chip block
            self.fc    = False
            self.miniR = 74
            self.miniC = 64
        elif noise.shape[0] == 664 and noise.shape[1] == 640:
            # P0 full chip block (faster version)
            self.fc    = False
            self.miniR = 83
            self.miniC = 80
        elif noise.shape[0] == 8*666 and noise.shape[1] == 12*640:
            # P0 full chip
            self.fc    = True
            self.miniR = 666
            self.miniC = 640
        elif noise.shape[0] == 8*664 and noise.shape[1] == 12*640:
            # P0 full chip (faster version)
            self.fc    = True
            self.miniR = 664
            self.miniC = 640
        elif noise.shape[0] == 1332 and noise.shape[1] == 1288:
            self.fc    = False
            self.miniR = 111
            self.miniC = 92
        elif noise.shape[0] == 8*1332 and noise.shape[1] == 12*1288:
            self.fc    = True
            self.miniR = 1332
            self.miniC = 1288
        elif noise.shape[0] == 8*1332*2 and noise.shape[1] == 12*1288*2:
            # P2 full chip full speed
            self.fc    = True
            self.miniR = 1332
            self.miniC = 1288
        elif noise.shape[0] == 8*1332*2 and noise.shape[1] == 12*1288*2:
            # P2 full chip full speed
            self.fc    = True
            self.miniR = 1332
            self.miniC = 1288
        elif noise.shape[0] == 8*1332*2/4 and noise.shape[1] == 12*1288*2:
            # P2 full chip 1/4 speed
            self.fc    = True
            self.miniR = 1332/2
            self.miniC = 1288*2
        else:
            self.fc    = False
            self.miniR = 0
            self.miniC = 0
            self.rows  = 0
            self.cols  = 0
            
    def byblock( self , bad_col_correct=True ):
        """ Returns mean noise by block.  Default is to correct for bad columns. """
        if bad_col_correct == True:
            return stats.block_avg( self.noise , self.miniR , self.miniC , np.logical_and( self.valmask , ~self.badcols ), finite=True )
        else:
            return stats.block_avg( self.noise , self.miniR , self.miniC , self.valmask, finite=True )
            
    def badcolnum( self ):
        return np.sum( self.badcols , axis = 1 )[0]
    
    def getblock( self , xblock , yblock ):
        """ Returns full chip block as a new instance of PI_noise """
        if self.fc:
            # This is a full chip
            rws = np.array(( yblock , yblock + self.miniR ))
            cls = np.array(( xblock , xblock + self.miniC ))
            roi = self.noise[   rws[0]:rws[1] , cls[0]:cls[1] ]
            bcs = self.badcols[ rws[0]:rws[1] , cls[0]:cls[1] ]
            return PI_noise( roi , bcs )
        else:
            return None

class PGM_noise: 
    """ Class to support pgm noise and the annotation of noisy column dacs. """
    def __init__( self , noise , badcols , img ):
        self.badcols = badcols
        self.valmask = np.logical_and( noise > 0 , noise < 150 )
        self.noise   = noise
        self.miniR   = img.miniR
        self.miniC   = img.miniC
        self.rows    = img.rows
        self.cols    = img.cols
        
    def badcolnum( self ):
        return np.sum( self.badcols , axis = 1 )[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Analyzes calibration .dat files" )
    
    # Required arguments
    parser.add_argument( '-i', '--input-dir', dest='inputdir', required=True, help="path to calibration files")
    
    # Optional arguments (for now . . . )
    parser.add_argument( '-c', '--chip-type', dest='chiptype', required=False, default='', help="chip specific string identifying chip type" )
    parser.add_argument( '-o', '--output-dir', dest='outputdir', required=False, default='', help="path to output most files (block-specific files will be autosaved in subdirs)")
    parser.add_argument( '--dr', type=float, help='Dynamic range (in mV)' )
    
    # Flags
    parser.add_argument( '-v', '--verbose', dest='verbose', help="print log file output to the terminal", action="store_true" )
    parser.add_argument( '--bc', help="do gain correction of buffering readings for the chip while the gain file is open for analysis" , action="store_true" )
    parser.add_argument( '--multilane', nargs='?', default=0, type=int, help='4-byte integer representing active lanes' )
    
    # Flags for piecewise analysis
    parser.add_argument( '--all', help="run all noise, pix offset, and gain calculations at once.", action="store_true" )
    parser.add_argument( '--noise', help="run noise calculations", action="store_true" )
    parser.add_argument( '--offset', help="run pixel offset calculations", action="store_true" )
    parser.add_argument( '--gain', help="run gain calculations", action="store_true" )
    parser.add_argument( '--refpix', help="run reference pixel calculations", action="store_true" )
    parser.add_argument( '--force', help="force recalculations if results already exist", action="store_true" )
    parser.add_argument( '--no_true', help="skip true metrics processing", action="store_true" )
    parser.add_argument( '--nofcmask', help="turn off use of pre-defined flowcell glueline mask if avaialable", action="store_true" )
    
    args = parser.parse_args()
    
    #--------------------#
    # Set variables
    #--------------------#
    
    input_dir  = args.inputdir
    output_dir = args.outputdir
    
    if args.chiptype == '':
        chiptype  = input_dir.split('/')[2]
    else:
        chiptype  = args.chiptype

        
    gaincorr   = args.bc
    
    calc_noise      = args.noise
    calc_offset     = args.offset
    calc_gain       = args.gain
    calc_refpix     = args.refpix
    true_metrics    = not args.no_true # default behavior is to evaluate true metrics
    
    if args.all:
        calc_noise  = True
        calc_offset = True
        calc_gain   = True
        calc_refpix = True
    
    if any( [calc_noise , calc_offset , calc_gain , calc_refpix] ):
        cc = ChipCal( input_dir , chiptype , output_dir , args.verbose, args.force, dr=args.dr, no_fc_mask=args.nofcmask )
        if args.multilane:
            cc.set_lanes_from_flag( args.multilane )
        cc.annotate( 'Chip type: %s' % chiptype )
        if calc_noise:
            cc.analyze_noise( close=False, true_metrics=true_metrics )
        if calc_offset:
            cc.analyze_offset( close=False, true_metrics=true_metrics )
        if calc_gain:
            cc.analyze_gain( buffering_gc=gaincorr, close=False, true_metrics=true_metrics )
        if calc_refpix:
            cc.analyze_refpix()
        if args.multilane:
            cc.save_lane_json()
            
#------------------------------------------------------------------------------
#                                 END OF FILE
#------------------------------------------------------------------------------
