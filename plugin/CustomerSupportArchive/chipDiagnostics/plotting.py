import sys, os
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt

##################################################
# Line & Scatter Plots!                          #
##################################################

class Plot:
    '''
    Generic class for making plots that will be parent class to a whole host of different types of plots.
    Used generic __init__ function, which should be overwritten for other plots that need user inputs.
    Default behavior is a scatter plot with no lines between points.
    '''
    x          = None
    y          = None
    xlims      = None
    ylims      = None
    xlabel     = ''
    ylabel     = ''
    xscale     = 'linear'
    yscale     = 'linear'
    title      = ''
    fmt        = 'o'
    markersize = 5
    grid       = False
    legend     = False
    legend_loc = None
    
    # TODO: We still need a legend function here!
    def __init__( self , norm=False ):
        # Normalize plot by it's initial value.
        self.norm       = norm
        
    def draw( self ):
        '''Makes items on the plot.  Expect for this to be overwritten many times.'''
        if self.norm:
            plt.plot( self.x , self.y - self.y[0] ,  self.fmt , ms=self.markersize )
        else:
            plt.plot( self.x , self.y ,  self.fmt , ms=self.markersize )
            
    def make_legend( self , location='best' ):
        self.legend     = True
        self.legend_loc = location
        
    def set_x( self , vals , label=None , log=False ):
        ''' Simultaneously set x values and x axis label '''
        self.x      = np.array( vals )
        if label:
            self.xlabel = label
        if log:
            self.xscale = 'log'
        
    def set_xticks( self , locs , labels ):
        ''' Sets x ticks at predetermined positions '''
        plt.xticks( locs , labels )

    def set_y( self , vals , label=None , log=False ):
        ''' Simultaneously set y values and y axis label '''
        self.y      = np.array( vals )
        if label:
            self.ylabel = label
        if log:
            self.yscale = 'log'
        
    def set_yticks( self , locs , labels ):
        ''' Sets y ticks at predetermined positions '''
        plt.yticks( locs , labels )

    def set_xy_labels( self , xlabel , ylabel ):
        self.xlabel , self.ylabel = xlabel , ylabel
        
    def plot( self , figname=None ):
        # Create figure, draw trace(s), etc.
        plt.figure()
        self.draw()
        
        # Adjust axes limits -- None has no affect on default limits!
        plt.xlim  ( self.xlims  )
        plt.ylim  ( self.ylims  )
        plt.xscale( self.xscale )
        plt.yscale( self.yscale )
        
        # Make normal annotations
        if self.grid:
            plt.grid()
            
        plt.xlabel( self.xlabel )
        plt.ylabel( self.ylabel )
        plt.title ( self.title  )
        if self.legend:
            plt.legend( loc=self.legend_loc )
        
        # Save or display
        if figname:
            plt.savefig( figname )
            plt.close  ( )
        else:
            plt.show()

class Lines( Plot ):
    ''' Class for plotting multiple traces '''
    ls      = '-'
    marker  = ''
    label   = ''
    
    def __init__( self , norm=False ):
        # Normalize y data by subtracting off the initial value.
        self.norm = norm
        
        # This cannot be defined above because then it carries to other instances of Lines. . . . .
        # Series is a list of couples: ( x , y , ls , marker , label )
        self.series = []
        self.formats= []
        self.labels = []
        
    def add_series( self , x , y , ls=None , marker=None , label=None ):
        if x == None:
            x = range( len(y) )
        self.series.append( ( x, y, ls if ls is not None else self.ls, marker if marker else self.marker, label if label else self.label ) )
        if label:
            self.make_legend( )
        
    def draw( self ):
        if self.series:
            for ( x , y , ls , m , l ) in self.series:
                if np.all( y == 0 ):
                    continue
                elif self.norm:
                    plt.plot( x , y-y[0] , ls+m , label=l )
                else:
                    plt.plot( x , y , ls+m , label=l )
        else:
            for i in range(self.N):
                if np.all( self.array[i,:]-self.array[i,0] == 0 ):
                    continue
                elif self.norm:
                    plt.plot( self.x , self.array[i,:]-self.array[i,0] , self.formats[i]  , label=self.labels[i] )
                else:
                    plt.plot( self.x , self.array[i,:] , self.formats[i]  , label=self.labels[i] )

    def from_array( self , array , xvals=None , formats=None  , labels=None ):
        ''' Sets up data such that different series are by row '''
        self.array     = array
        self.N,self.dp = array.shape
        if xvals:
            self.x = xvals
        else:
            self.x = np.arange( self.dp )
            
        if formats and len(formats) == self.N:
            self.formats = formats
        else:
            self.formats = self.N * [ (self.ls + self.marker) ]
            
        if labels and len(labels) ==  self.N:
            self.labels = labels
            self.make_legend( )
        else:
            self.labels = self.N * [ '' ]
            
class Scatter( Plot ):
    fmt = 'o'
    def __init__( self , x , y , xlabel='' ,  ylabel='' , fmt=None , norm=False ):
        self.norm = norm
        if fmt:
            self.fmt = fmt
        if x == None:
            x = range( len(y) )
        self.set_x( x , xlabel )
        self.set_y( y , ylabel )

class ColorScatter( Plot ):
    '''
    This plot will plot a scatter plot and colorcode the marker colors by a 3rd dimension, z.
    You can set the scale manually, or let it ride to the min, max values.  Recommended use is to supply limits.
    '''
    markersize = 10
    def __init__( self , annotate=False ):
        ''' Overwrite Plot __init__() function to allow us to decide if we want data labels or not.'''
        self.annotate = annotate
    
    def set_z( self , vals , scale=None ):
        ''' Simultaneously sets z (marker color) values and their color limits. '''
        self.z = np.array( vals )
        self.ccmap = matplotlib.cm.ScalarMappable( cmap = matplotlib.cm.jet )
        if scale:
            self.ccmap.set_clim( vmin=scale[0] , vmax=scale[1] )
        else:
            self.ccmap.set_clim( vmin=self.z.min() , vmax=self.z.max() )
            
    def draw( self ):
        for (x,y,z,c) in zip( self.x , self.y ,  self.z , self.ccmap.to_rgba( self.z ) ):
            plt.plot( x , y , 'o' , color=c , ms=self.markersize  )
            if self.annotate:
                plt.annotate( '%d' % z , xy = (x+2,y+2) )

##################################################
# Spatial Plots!                                 #
##################################################

class Spatial( Plot ):
    '''
    Basic class for a spatial plot.  Nothing fancy here yet.
    '''
    climits         = None
    colorbar_shrink = 0.7
    
    # Overwrite Plot.set_x and .set_y
    def set_x( self ):
        ''' Not used! '''
        print( 'Warning!  The set_x method is not used in the Spatial Class.  use set_xy_labels and set_data.' )
    
    def set_y( self ):
        ''' Not used! '''
        print( 'Warning!  The set_y method is not used in the Spatial Class.  use set_xy_labels and set_data.' )
    
    def set_data( self , data , clim=None ):
        ''' Sets spatial data and optionally changes color limits '''
        self.data = data
        if clim:
            self.climits = clim
            
    def draw( self ):
        if self.climits:
            plt.imshow( self.data , origin='lower', interpolation='nearest' , clim=self.climits )
        else:
            plt.imshow( self.data , origin='lower', interpolation='nearest' )
        plt.colorbar  ( shrink=self.colorbar_shrink )

class BlockAvgSpatial( Spatial ):
    ''' Spatial plot that will downsample original data through BlockAvg '''
    xlabel = 'Columns'
    ylabel = 'Rows'
    
    def __init__( self , data , blocksize , mask=None , clim=None , gap=None ):
        self.set_data( data , blocksize , mask, clim , gap )

    def BlockAvg( self ):
        ''' Performs a masked averaging operation in pixel blocks of self.blocksize. '''
        if hasattr( self , 'data' ):
            reshaped = self.BlockReshape( self.data , self.blocksize )
            remasked = np.array( self.BlockReshape( self.mask , self.blocksize ) , bool )
            masked   = ma.masked_array( reshaped , remasked )
            return masked.mean(2).data
        else:
            print( 'Error!  Can not do BlockAvg until set_data has been called!' )
            return None
    
    def BlockSum( self ):
        if hasattr( self , 'data' ):
            reshaped = self.BlockReshape( self.data , self.blocksize )
            remasked = np.array( self.BlockReshape( self.mask , self.blocksize ) , bool )
            masked   = ma.masked_array( reshaped , remasked )
            return masked.sum(2).data
        else:
            print( 'Error!  Can not do BlockSum until set_data has been called!' )
            return None
    
    def BlockReshape( self , data , blocksize ):
        ''' Does series of reshapes and transpositions to get into miniblocks. '''
        rows , cols = data.shape
        blockR = rows / blocksize[0]
        blockC = cols / blocksize[1]
        return data.reshape(rows , blockC , -1 ).transpose((1,0,2)).reshape(blockC,blockR,-1).transpose((1,0,2))
    
    def draw( self ):
        if self.climits:
            plt.imshow( self.avg_data , origin='lower', interpolation='nearest' , clim=self.climits )
        else:
            plt.imshow( self.avg_data , origin='lower', interpolation='nearest' )
        self.set_ticks( )
        plt.colorbar  ( shrink=self.colorbar_shrink )
        
    def set_data( self , data , blocksize , mask=None , clim=None , gap=None ):
        ''' Important note: mask is for pixels what we want to ignore!  Like pinned, e.g. '''
        self.data      = np.array( data , dtype=float )
        self.blocksize = blocksize
        
        # Again, masked pixels are ones we want to ignore.   If using all, we need np.zeros.
        if mask == None:
            self.mask = np.zeros( self.data.shape )
        else:
            self.mask = mask
            
        self.avg_data = self.BlockAvg( )
        if clim:
            self.climits  = clim
        
        # Redo gap if there's not going to be enough data points.  Default behavior assumes chip data.
        r,c = self.data.shape
        if gap:
            self.gap = gap
        else:
            if ( r/500 > 3 and c/500 > 3 ):
                self.gap = 500
            else:
                self.gap = 200
                
    def set_ticks( self ):
        r,c       = self.data.shape
        subr,subc = self.avg_data.shape
        plt.xticks( np.arange( 0 , (subc+1) , self.gap / self.blocksize[1] ) , np.arange( 0 , c+1 , self.gap  ) )
        plt.yticks( np.arange( 0 , (subr+1) , self.gap / self.blocksize[0] ) , np.arange( 0 , r+1 , self.gap  ) )

class RawPlot( Spatial ):
    '''  Class for raw, axes-less plot and associated colorbar saved separately. '''
    cbar_ticks = 8
    cbar_label = None
    scale      = 1
    
    def __init__( self , data , clim=None , colorbar_label=None ):
        self.set_data( data , clim ) 
        self.cbar_label = colorbar_label
        
    def draw( self ):
        ''' Overwritten for this custom case '''
        pass
    
    def plot( self , figname=None ):
        # Create figure, draw trace(s), etc.
        fig = plt.figure( figsize=self.figsize )
        ax  = plt.Axes( fig , [0,0,1,1] )
        ax.set_axis_off()
        fig.add_axes( ax )
        
        # Replacing self.draw() here.
        if self.climits:
            im = ax.imshow( self.data , origin='lower', interpolation='nearest' , clim=self.climits )
        else:
            im = ax.imshow( self.data , origin='lower', interpolation='nearest' )
            self.climits = list( im.get_clim() )
            
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        # Save or display
        if figname:
            self.figname = figname
            plt.savefig( figname , bbox_inches=extent )
            plt.close  ( )
            self.save_colorbar()
        else:
            plt.show()
            
    def auto( self , path , name ):
        ''' makes plot and saves raw dat automatically. '''
        self.plot( os.path.join( path , 'raw_%s.png'  % name ) )
        self.save_raw_dat( os.path.join( path , '%s.dat' % name ) )
            
    def set_data( self , data , clim=None ):
        ''' Sets spatial data and optionally changes color limits '''
        # Set data and define required figure size
        self.data = data
        rows,cols = data.shape
        if (rows,cols) == (1332,1288) or (rows,cols) == (664,640):
            self.figsize = (1.93 , 2)
        else:
            self.figsize = ( np.rint( (float(cols) / float(rows)* 10.) * 100. ) / 100. , 10. )
            
        if clim:
            self.climits = clim
            
    def save_colorbar( self ):
        # Make colorbar with 8 evenly spaced ticks.
        cmin,cmax = self.climits
        if cmin == 0:
            data = np.tile( np.linspace( cmin , cmax , (cmax + 1) ) , ( np.rint( 15./400. * cmax ) , 1 ) )
        else:
            divs = 201
            data = np.tile( np.linspace( cmin , cmax , divs ) , ( np.rint( 15./400. * (divs-1) ) , 1 ) )
            
        plt.figure ( figsize=( 7.5 , 1.2 ) )
        plt.imshow ( data , interpolation='nearest' , origin='lower' , clim=self.climits )
        
        if cmin == 0:
            plt.xticks ( np.arange( cmin , (cmax + 1) , cmax / self.cbar_ticks ) )
        else:
            labels = []
            for i in range( self.cbar_ticks+1 ):
                labels.append( str( cmin + i * ( cmax - cmin ) / self.cbar_ticks ) )
            plt.xticks ( np.arange( 0 , divs , (divs-1) / self.cbar_ticks ) , labels )
        plt.yticks ( [] , ('') )
        if self.cbar_label:
            plt.xlabel ( self.cbar_label )
        
        fs = os.path.basename( self.figname ).split('.')
        fn = os.path.join( os.path.dirname( self.figname ) , '%s_cb.%s' % ( fs[0] , fs[1] ) )
        plt.savefig( fn )
        plt.close  ( )
        
    def save_raw_dat( self , filepath , scale_factor=None ):
        ''' Saves raw plotted data to file.  Optionally can scale by 10, which is commonly used. '''
        if scale_factor:
            self.scale = scale_factor
        np.array( np.rint( self.scale * self.data ) , dtype=np.int16 ).tofile( filepath )

##################################################
# Histograms!                                    #
##################################################

class Histogram( Plot ):
    ylabel = 'Count'
    align  = 'mid'
    norm   = False
    
    def set_bins( self , start , end , count , xlabel=None , align=None ):
        ''' 
        Sets up the bins to use for the histogram. 
        NB: 'count' will likely be an even number.  Will need to add one to make count+1 bin edges...
        '''
        self.bin_limits  = np.linspace(start,end,count+1)
        self.xlims = (start,end)
        if xlabel:
            self.xlabel = xlabel
        if str(align).lower() in ['left' , 'mid' , 'right']:
            self.align = align.lower()

    def set_data( self , data , title=None ):
        ''' Sets spatial data and optionally changes color limits '''
        self.data = data
        if title:
            self.title = title
            
    # Overwrite Plot.set_x and .set_y
    def set_x( self ):
        ''' Not used! '''
        print( 'Warning!  The set_x method is not used in the Spatial Class.  use set_xy_labels and set_data.' )
        
    def set_y( self ):
        ''' Not used! '''
        print( 'Warning!  The set_y method is not used in the Spatial Class.  use set_xy_labels and set_data.' )
        
    def draw( self ):
        self.n , self.bins , _ = plt.hist( self.data , self.bin_limits , align=self.align )
        self.mode = self.calc_mode( )
        
    def calc_mode( self ):
        """
        Simple routine to get the mode of the distribution of a histogram.
        In the rare case that more than one modes are found, we take the average. 
        If it's a bimodal distribution, we wouldn't believe this value anyway.
        """
        mode = self.bins[ np.where( self.n == self.n.max() ) ]
        
        # Non-ideal solution
        if len( mode ) > 1:
            return mode.mean()
        else:
            return mode[0]

class ColorHist( Histogram ):
    ''' 
    Extra special histogram version that adds another data dimension through the color of each histogram bar.
    
    Proper usage follows:
    - set_data
    - set_bins
    - set_colordata
    '''

    def set_colordata( self , colordata , label , lims=[] , colormap=matplotlib.cm.jet , use_std=False ):
        ''' 
        Set up color data to be used with histogram based on inputs.  
        Left out ability to further mask, assumes colordata matches data shape/validity of self.data
        '''
        self.cbar_label = label
        self.colormap   = colormap
        
        # Initialize arrays
        # - meanvals = mean colordata value per histogram bin
        # - midpoints= calculated center point of each histogram bin.  e.g. midpoint of [0,10] is 5
        N         = len(self.bin_limits)
        self.meanvals  = np.zeros(( N-1 ) , dtype=float)
        self.midpoints = np.zeros(( N-1 ) , dtype=float)
        self.stdvals   = np.zeros(( N-1 ) , dtype=float)

        for i in np.arange(0,N-1):
            # get mask of pixels in present histogram bin
            binned       = np.logical_and( self.data >= self.bin_limits[i] , self.data < self.bin_limits[i+1] )
            
            self.midpoints[i] = ( self.bin_limits[i+1] + self.bin_limits[i] ) / 2.
            self.meanvals[i]  = colordata[ binned ].mean()
            self.stdvals[i]   = colordata[ binned ].std()
            
        # Clean out nans
        self.meanvals[ np.isnan( self.meanvals ) ] = 0
        self.stdvals[  np.isnan( self.stdvals )  ] = 0
        
        # Define limits for the colorscale if not given
        if lims == []:
            valid = self.meanvals[self.meanvals > 0 ]
            self.color_lims = [ valid.min() , valid.max() ]
        else:
            self.color_lims = lims
            
        # Generate list of colors corresponding to mean
        self.ccmap = matplotlib.cm.ScalarMappable( cmap = colormap )
        self.ccmap.set_clim( vmin = self.color_lims[0] , vmax = self.color_lims[1] )
        if use_std:
            self.colors = self.ccmap.to_rgba( self.stdvals )
        else:
            self.colors = self.ccmap.to_rgba( self.meanvals )

    def draw( self ):
        # This is a crude approach, but set up dummy imshow plot with colorbar then clear figure, replace with hist
        canvas = plt.imshow( np.zeros((5,5)) , clim=self.color_lims , cmap=self.colormap )
        plt.clf()
        
        self.n , self.bins , self.patches = plt.hist( self.data , self.bin_limits , align=self.align )
        self.mode = self.calc_mode( )
        
        # Repaint the histogram bars
        for j in range(len(self.patches)):
            self.patches[j].set_facecolor( self.colors[j,:] )
            
        plt.colorbar( canvas ).set_label( self.cbar_label , rotation=270 , style='italic' , labelpad=20 )
