import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

def subplot_to_grid( fig , shape, loc, rowspan=1, colspan=1 ):
    """ A mirror function of matplotlib.pyplot.subplot2grid that doesn't try to import tkinter """
    gridspec    = GridSpec( shape[0], shape[1])
    subplotspec = gridspec.new_subplotspec(loc, rowspan, colspan)
    axis        = fig.add_subplot( subplotspec )
    return axis

class MultilanePlot:
    def __init__( self , data , title , metric_label , units='' , clims=None , cmap=matplotlib.cm.nipy_spectral , bin_scale=1 ):
        self.data         = data
        self.title        = title
        self.metric_label = metric_label
        self.units        = units
        self.bin_scale    = bin_scale
        
        # self.cmap is the color plan, and used in imshow.  self.cm is the cmap mapped onto our given limits.
        self.cmap         = cmap
        
        self.update_clims( clims )
        
        # Set other secretly configurable properties
        self.figsize = (12,8)
        
    def update_clims( self , clims ):
        if not clims:
            _ = self.calc_clims()
        else:
            self.clims = clims
            
        self.calc_bins( )
        self.set_cm( self.cmap )
        return None
        
    def update_cmap( self , cmap ):
        self.cmap = cmap
        self.set_cm( self.cmap )
        
    def get_xlabel( self ):
        if self.units != '':
            xl = '{} ({})'.format( self.metric_label , self.units )
        else:
            xl = '{}'.format( self.metric_label )
        return xl
    
    def plot_one( self , lane_id , figpath=None , figsize=None):
        lane_id = int( lane_id )
        if lane_id not in [1,2,3,4]:
            print( "Hey!  there's only 4 lanes.  Please enter a lane_id of 1-4." )
            return None
        else:
            # Get lane data
            lane_data = self.lane_slice( lane_id )
            
        if figsize:
            fig     = plt.figure( figsize=figsize )
        else:
            fig     = plt.figure( figsize=(6,8) )
        spatial = subplot_to_grid( fig , (3,1) , (0,0) )
        lane    = subplot_to_grid( fig , (3,1) , (1,0) , rowspan=2 )
        
        # Spatial plot
        im = spatial.imshow ( lane_data.T[:,::-1] , interpolation='nearest', origin='lower', clim=self.clims , cmap=self.cmap )
        spatial.set_xticks  ( [] )
        spatial.set_yticks  ( [] )
        spatial.set_title   ( 'Lane {} | {}'.format( lane_id , self.title ) )
        
        # Histogram
        n, bins, patches = lane.hist( lane_data[lane_data > 0] , bins=self.bins, zorder=0 )
        colors = self.cm.to_rgba( bins[:-1] + 0.5 * np.diff(bins) )
        for i in range( len(patches) ):
            patches[i].set_facecolor( colors[i,:] )
            patches[i].set_edgecolor( colors[i,:] )
            
        # X-axis config
        xt = np.arange      ( self.clims[0] , self.clims[1]+0.1 , float(self.clims[1]-self.clims[0])/4. )
        lane.set_xlim       ( self.clims )
        lane.set_xticks     ( xt )
        lane.set_xticklabels( [ '{:.1f}'.format(x) for x in xt ] )
        lane.set_xlabel     ( self.get_xlabel() )
        
        avg = lane_data[ lane_data > 0 ].mean()
        q2  = np.median( lane_data[ lane_data > 0 ] )
        
        if self.units == '%':
            _   = lane.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f}{}'.format( avg, self.units ) )
            _   = lane.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f}{}'.format( q2 , self.units ) )
        else:
            _   = lane.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f} {}'.format( avg, self.units ) )
            _   = lane.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f} {}'.format( q2 , self.units ) )
        
        lane.legend        ( loc='best' , fontsize=10 )
        lane.grid          ( ':' , color='grey' , zorder=3 )
        plt.tight_layout   ( )
        plt.subplots_adjust( hspace=0, wspace=0 , top=0.94 )
        if not figpath:
            plt.show( )
        else:
            plt.savefig( figpath )
            plt.close  ( )

    def plot_histograms_only( self , figpath=None , figsize=None , color=False ):
        """ WARNING -- Lane 4 still at top to align with rotated chip.  Need to have labels """
        if figsize:
            fig     = plt.figure( figsize=figsize )
        else:
            fig     = plt.figure( figsize=(6,8) )
        fig     = plt.figure( figsize = self.figsize )
        lane4   = fig.add_subplot( 411 )
        lane3   = fig.add_subplot( 412 )
        lane2   = fig.add_subplot( 413 )
        lane1   = fig.add_subplot( 414 )
        
        # Create Histograms
        max_ys = []
        for i,ax in zip( [4,3,2,1], [lane4,lane3,lane2,lane1] ):
            lane_data          = self.lane_slice( i )
            n, bins, patches   = ax.hist( lane_data[lane_data > 0] , bins=self.bins, zorder=0 )
            ax.grid            ( ':' , color='grey' , zorder=3 )
            ax.set_xlim        ( self.clims )
            ax.yaxis.tick_right( )
            max_ys.append      ( ax.get_ylim()[1] )
            
            masked = lane_data[ lane_data > 0 ]
            if masked.any():
                avg = masked.mean()
                q2  = np.median( masked )
            else:
                avg = 0.
                q2  = 0.
            _   = ax.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f}{}'.format( avg, self.units ) )
            _   = ax.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f}{}'.format( q2 , self.units ) )
            ax.legend( loc='best' , fontsize=10 )
            
            if color:
                colors = self.cm.to_rgba( bins[:-1] + 0.5 * np.diff(bins) )
                for i in range( len(patches) ):
                    patches[i].set_facecolor( colors[i,:] )
                    patches[i].set_edgecolor( colors[i,:] )
                    
            xt = np.arange    ( self.clims[0] , self.clims[1]+0.1 , float(self.clims[1]-self.clims[0])/4. )
            _  = ax.set_xticks( xt )
            if i > 1:
                _ =ax.set_xticklabels( [] )
                
        lane1.set_xticklabels( [ '{:.1f}'.format(x) for x in xt ] )
        lane1.set_xlabel     ( self.get_xlabel() )
        ymax = max( max_ys )
        for i,ax in zip( [4,3,2,1], [lane4,lane3,lane2,lane1] ):
            ax.set_ylim       ( 0 , ymax )
            ax.set_yticks     ( np.arange( 0 , ymax , ymax/5 ) )
            ax.set_yticklabels( np.arange( 0 , ymax , ymax/5 , dtype=int) )
            ax.set_ylabel     ( 'Lane {}'.format( i ) )
            
        plt.suptitle       ( self.title , fontsize=16 )
        plt.tight_layout   ( )
        plt.subplots_adjust( hspace=0, wspace=0 , top=0.94 )
        if not figpath:
            plt.show( )
        else:
            plt.savefig( figpath )
            plt.close  ( )

    def plot_single_lane_histogram_only( self , lane_num=None, figpath=None , figsize=None , color=False ):
        """ Plots a histogram for a single lane only """
       
        # Create Histograms
        if lane_num is not None:
            if figsize:
                fig     = plt.figure( figsize=figsize )
            else:
                fig     = plt.figure( figsize=(6,8) )
            fig = plt.figure( figsize = self.figsize )
            ax  = fig.add_subplot( 111 )

            lane_data          = self.lane_slice( lane_num )
            n, bins, patches   = ax.hist( lane_data[lane_data > 0] , bins=self.bins, zorder=0 )
            ax.grid            ( ':' , color='grey' , zorder=3 )
            ax.set_xlim        ( self.clims )
            ax.yaxis.tick_right( )
            ymax              = ax.get_ylim()[1]
            
            masked = lane_data[ lane_data > 0 ]
            if masked.any():
                avg = masked.mean()
                q2  = np.median( masked )
            else:
                avg = 0.
                q2  = 0.
            _   = ax.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f}{}'.format( avg, self.units ) )
            _   = ax.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f}{}'.format( q2 , self.units ) )
            ax.legend( loc='best' , fontsize=10 )
            
            if color:
                colors = self.cm.to_rgba( bins[:-1] + 0.5 * np.diff(bins) )
                for i in range( len(patches) ):
                    patches[i].set_facecolor( colors[i,:] )
                    patches[i].set_edgecolor( colors[i,:] )
                    
            xt = np.arange    ( self.clims[0] , self.clims[1]+0.1 , float(self.clims[1]-self.clims[0])/4. )
            _  = ax.set_xticks( xt )
                
            ax.set_xticklabels( [ '{:.1f}'.format(x) for x in xt ] )
            ax.set_xlabel     ( self.get_xlabel() )
            ax.set_ylim       ( 0 , ymax )
            ax.set_yticks     ( np.arange( 0 , ymax , ymax/5 ) )
            ax.set_yticklabels( np.arange( 0 , ymax , ymax/5 , dtype=int) )
            ax.set_ylabel     ( 'Lane {}'.format( lane_num ) )
            
            plt.suptitle       ( self.title , fontsize=16 )
            plt.tight_layout   ( )
            plt.subplots_adjust( hspace=0, wspace=0 , top=0.94 )
 
            if not figpath:
                plt.show( )
            else:
                plt.savefig( figpath )
                plt.close  ( )

    def plot_all( self , figpath=None ):
        # Set up fig and axes
        fig     = plt.figure( figsize = self.figsize )
        spatial = subplot_to_grid( fig , (4,5) , (0,0) , colspan=3 , rowspan=4 )
        lane4   = subplot_to_grid( fig , (4,5) , (0,3) , colspan=2 )
        lane3   = subplot_to_grid( fig , (4,5) , (1,3) , colspan=2 )
        lane2   = subplot_to_grid( fig , (4,5) , (2,3) , colspan=2 )
        lane1   = subplot_to_grid( fig , (4,5) , (3,3) , colspan=2 )
        
        # Spatial plot
        im = spatial.imshow( self.data.T[:,::-1] , interpolation='nearest', origin='lower', clim=self.clims , cmap=self.cmap )
        for y in np.arange( 0 , self.data.shape[1] , self.data.shape[1]/4. )[1:]:
            _ = spatial.axhline( y , ls='-' , color='black' )
        
        ytl       = [''] * 8
        ytl[1::2] = ['Lane 1','Lane 2','Lane 3','Lane 4']
        spatial.set_xticks     ( [] )
        spatial.set_xlabel     ( 'Inlet ----> Outlet' )
        spatial.set_yticks     ( np.arange( 0 , self.data.shape[1]+1 , self.data.shape[1]/8. ) )
        spatial.set_yticklabels( ytl )
        
        # Create Histograms
        max_ys = []
        for i,ax in zip( [4,3,2,1], [lane4,lane3,lane2,lane1] ):
            lane_data          = self.lane_slice( i )
            n, bins, patches   = ax.hist( lane_data[lane_data > 0] , bins=self.bins, zorder=0 )
            ax.grid            ( ':' , color='grey' , zorder=3 )
            ax.set_xlim        ( self.clims )
            ax.yaxis.tick_right( )
            max_ys.append      ( ax.get_ylim()[1] )
            
            masked = lane_data[ lane_data > 0 ]
            if masked.any():
                avg = masked.mean()
                q2  = np.median( masked )
            else:
                avg = 0.
                q2  = 0.
                
            if self.units == '%':
                _   = ax.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f}{}'.format( avg, self.units ) )
                _   = ax.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f}{}'.format( q2 , self.units) )
            else:
                _   = ax.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f} {}'.format( avg, self.units ) )
                _   = ax.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f} {}'.format( q2 , self.units) )
            ax.legend( loc='best' , fontsize=10 )
            
            colors = self.cm.to_rgba( bins[:-1] + 0.5 * np.diff(bins) )
            for i in range( len(patches) ):
                patches[i].set_facecolor( colors[i,:] )
                patches[i].set_edgecolor( colors[i,:] )
                
            xt = np.arange    ( self.clims[0] , self.clims[1]+0.1 , float(self.clims[1]-self.clims[0])/4. )
            _  = ax.set_xticks( xt )
            if i > 1:
                _ =ax.set_xticklabels( [] )
                
        lane1.set_xticklabels( [ '{:.1f}'.format(x) for x in xt ] )
        lane1.set_xlabel     ( self.get_xlabel() )
        ymax = max( max_ys )
        for ax in [lane4,lane3,lane2,lane1]:
            ax.set_ylim       ( 0 , ymax )
            ax.set_yticks     ( np.arange( 0 , ymax , ymax/5 ) )
            ax.set_yticklabels( np.arange( 0 , ymax , ymax/5 , dtype=int) )
            
        plt.suptitle       ( self.title , fontsize=16 )
        plt.tight_layout   ( )
        plt.subplots_adjust( hspace=0, wspace=0 , top=0.94 )
        if not figpath:
            plt.show( )
        else:
            plt.savefig( figpath )
            plt.close  ( )
            
    def calc_clims( self ):
        print( 'data shape', self.data.shape )
        valid = self.data[ self.data > 0 ]
        print( 'valid shape (i.e. >0)', valid.shape )
        if valid.shape == (0,):
            self.clims = [-1,1]
            return self.clims
        m     = valid.mean()
        sigma = valid.std()
        skew  = (valid.max() - m) - (m - valid.min())
        if skew < (-3 * sigma):
            low  = m - 4.5 * sigma
            high = m + 1.5 * sigma
        elif skew > (3 * sigma):
            low  = m - 1.5 * sigma
            high = m + 4.5 * sigma
        else:
            low  = m - 3 * sigma
            high = m + 3 * sigma
        zmin = 10. * ( np.floor( low / 10. ) )
        zmax = 10. * ( np.ceil ( high / 10. ) )
        if zmin < 0:
            zmin = 0
        self.clims = [zmin , zmax]
        return [zmin,zmax]
    
    def calc_bins ( self , bin_scale=None ):
        a , b     = self.clims
        if bin_scale:
            self.bin_scale = bin_scale
        
        # Auto bin scale scaling
        if self.bin_scale == 1:
            if (b-a) <= 20:
                print( 'Adjusting bin_scale to 4 due to small clims.' )
                self.bin_scale = 4
            elif (b-a) > 20 and (b-a) <= 50:
                print( 'Adjusting bin_scale to 2 due to modest clims.' )
                self.bin_scale = 2
        #print( 'bin_scale', self.bin_scale ) 
        #print( 'a, b', a, b )
        self.bins = np.linspace( a , b , self.bin_scale*(b-a) + 1 )
        
    def lane_slice( self , lane_number ):
        """ Takes a data array and returns data from only the lane of interest.  lane_number is 1-indexed. """
        lane_width = self.data.shape[1] / 4
        cs         = slice( lane_width*(lane_number-1) , lane_width*(lane_number) )
        return self.data[:,cs]
    
    def set_cm( self , cmap ):
        self.cm = self.create_cmap( self.clims , cmap )
        
    @staticmethod
    def create_cmap( lims , cmap ):
        cm = matplotlib.cm.ScalarMappable( cmap=cmap )
        cm.set_clim( *lims )
        return cm
    
def awesome_plot( data , title , metric_label , clims=None , cmap=matplotlib.cm.nipy_spectral , bin_scale=1):
    def calc_clims( data ):
        valid = data[data>0]
        print( 'data shape', data.shape )
        print( 'valid shape (i.e. >0)', valid.shape )
        if valid.shape == (0,): return [-1,1]
        m     = valid.mean()
        sigma = valid.std()
        skew  = (valid.max() - m) - (m - valid.min())
        if skew < (-3 * sigma):
            low  = m - 4.5 * sigma
            high = m + 1.5 * sigma
        elif skew > (3 * sigma):
            low  = m - 1.5 * sigma
            high = m + 4.5 * sigma
        else:
            low  = m - 3 * sigma
            high = m + 3 * sigma
        zmin = 10. * ( np.floor( low / 10. ) )
        zmax = 10. * ( np.ceil ( high / 10. ) )
        if zmin < 0:
            zmin = 0
        return [zmin,zmax]
    
    def create_cmap( lims , cmap ):
        cm = matplotlib.cm.ScalarMappable( cmap=cmap )
        cm.set_clim( *lims )
        return cm
    
    def lane_slice( data , lane_number ):
        """ Takes a data array and returns data from only the lane of interest.  lane_number is 1-indexed. """
        lane_width = data.shape[1] / 4
        cs         = slice( lane_width*(lane_number-1) , lane_width*(lane_number) )
        return data[:,cs]
    
    # Set up fig and axes
    fig     = plt.figure( figsize = (12,8) )
    spatial = subplot_to_grid( fig , (4,5) , (0,0) , colspan=3 , rowspan=4 )
    lane4   = subplot_to_grid( fig , (4,5) , (0,3) , colspan=2 )
    lane3   = subplot_to_grid( fig , (4,5) , (1,3) , colspan=2 )
    lane2   = subplot_to_grid( fig , (4,5) , (2,3) , colspan=2 )
    lane1   = subplot_to_grid( fig , (4,5) , (3,3) , colspan=2 )
    
    # Spatial plot
    if not clims:
        clims = calc_clims( data )
    cm = create_cmap( clims , cmap )
    im = spatial.imshow( data.T , interpolation='nearest', origin='lower', clim=clims , cmap=cmap )
    for y in np.arange( 0 , data.shape[1] , data.shape[1]/4. )[1:]:
        _ = spatial.axhline( y , ls='-' , color='black' )
        
    ytl       = [''] * 8
    ytl[1::2] = ['Lane 1','Lane 2','Lane 3','Lane 4']
    spatial.set_xticks     ( [] )
    spatial.set_yticks     ( np.arange( 0 , data.shape[1]+1 , data.shape[1]/8. ) )
    spatial.set_yticklabels( ytl )
    
    # Create Histograms
    max_ys = []
    for i,ax in zip( [4,3,2,1], [lane4,lane3,lane2,lane1] ):
        lane_data          = lane_slice( data , i )
        n, bins, patches   = ax.hist( lane_data[lane_data > 0] ,
                                      bins=np.linspace( clims[0],clims[1], bin_scale*(clims[1]-clims[0])+1 ) ,
                                      zorder=0 )
        ax.grid            ( ':' , color='grey' , zorder=3 )
        ax.set_xlim        ( clims )
        ax.yaxis.tick_right( )
        max_ys.append      ( ax.get_ylim()[1] )
        
        avg = lane_data[ lane_data > 0 ].mean()
        q2  = np.median( lane_data[ lane_data > 0 ] )
        _   = ax.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f}{}'.format( avg, self.units ) )
        _   = ax.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f}{}'.format( q2 , self.units ) )
        ax.legend( loc='best' , fontsize=10 )
        
        colors = cm.to_rgba( bins[:-1] + 0.5 * np.diff(bins) )
        for i in range( len(patches) ):
            patches[i].set_facecolor( colors[i,:] )
            patches[i].set_edgecolor( colors[i,:] )
            
        xt = np.arange    ( clims[0] , clims[1]+1 , (clims[1]-clims[0])/5 )
        _  = ax.set_xticks( xt )
        if i > 1:
            _ =ax.set_xticklabels( [] )
            
    lane1.set_xticklabels( [ '{:.1f}'.format(x) for x in xt ] )
    lane1.set_xlabel     ( metric_label )
    ymax = max( max_ys )
    for ax in [lane4,lane3,lane2,lane1]:
        ax.set_ylim       ( 0 , ymax )
        ax.set_yticks     ( np.arange( 0 , ymax , ymax/5 ) )
        ax.set_yticklabels( np.arange( 0 , ymax , ymax/5 , dtype=int) )
        
    plt.suptitle       ( title , fontsize=16 )
    plt.tight_layout   ( )
    plt.subplots_adjust( hspace=0, wspace=0 , top=0.94 )
    plt.show           ( )
    
