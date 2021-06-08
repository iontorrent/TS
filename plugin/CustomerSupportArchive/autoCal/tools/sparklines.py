import datetime
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

class Annotation( object ):
    """ Mixin class for Label / Number objects """
    def set_gridspec( self, gs ):
        self.gs = gs
        
    def get_axes( self , fig , gs=None ):
        if gs:
            self.set_gridspec( gs )
            
        self.fig = fig
        self.ax  = plt.Subplot( self.fig, self.gs )
        
    def draw( self , *args, **kwargs ):
        """ Creates the axes, writes the text, and adds the subplot to the figure. """
        x         = kwargs.pop( 'x' , 0.5)
        y         = kwargs.pop( 'y' , 0.5)
        ha        = kwargs.pop( 'ha', 'center' )
        va        = kwargs.pop( 'va', 'center' )
        transform = kwargs.pop('transform', self.ax.transAxes)
        
        self.ax.axis('off')
        self.ax.text( x, y, self.text, transform=transform, ha=ha, va=va, **kwargs )
        self.fig.add_subplot( self.ax )
              
class Label( Annotation ):
    """ 
    Class for a text label on a sparkline, which will take up an axis.  
    kwargs will get fed into the plt.text argument
    kwargs can also be used to change the plt.text x, y, and transform arguments.
    """
    def __init__( self , text , width=1 , **kwargs ):
        self.text   = text
        self.width  = width
        self.kwargs = kwargs
        
        # Auto adjustment code if width is None
        if self.width == None:
            # Assumes 4 characters per width ratio unit
            self.width = len( text ) / 3
            rem        = len( text ) % 3
            if rem > 0:
                self.width += 1

            print( 'Adjusted relative width of label to {}!'.format( self.width ) )
                
        
    def draw( self ):
        """ Customizes call to parent class draw by using self.kwargs """
        super( Label , self ).draw( **self.kwargs )
             
class Number( Annotation ):
    """ 
    Class for a numberical annotation on a sparkline, which will also take up an axis 
    kwargs will get fed into the plt.text argument
    kwargs can also be used to change the plt.text x, y, and transform arguments.
    """
    def __init__( self , value , fmt='{:d}' , width=1 , **kwargs ):
        self.value  = value
        self.fmt    = fmt
        self.width  = width
        self.kwargs = kwargs
        
    @property
    def text( self ):
        return self.fmt.format( self.value )
    
    def draw( self ):
        """ Customizes call to parent class draw by using self.kwargs """
        super( Number , self ).draw( **self.kwargs )
        
class Sparkline( object ):
    """ 
    Base clase for sparkline plot - defined as an plot with or without numerical annotations and text labels 
    
    IMPORTANT NOTE!
    In most cases, your numerical annotations (numbers) will depend on the data to be put in the plot.
    As such, it is recommended that you process your data first before calling draw on the Annotations or the data
    """
    def __init__( self , fig, gs , data_axis_width=5 ):
        """ Initializes the sparkline to live on the particular gridspec supplied. """
        self.fig             = fig
        self.outer_grid      = gs
        self.data_axis_width = data_axis_width
        
        # Set some defaults
        self.left_labels     = []
        self.right_labels    = []
        self.left_numbers    = []
        self.right_numbers   = []
        
        # These booleans set the order of labels
        # Normal is [ left_label, left_num, <sparkline>, right_num, right_label ] which is covered by:
        self.left_label_out  = True
        self.right_label_out = True
        
    def add_label( self, label, side ):
        side = side.lower()
        if side in ['left','right']:
            getattr( self, '{}_labels'.format( side ) ).append( label )
        else:
            raise ValueError( 'side must be either "left" or "right"!' )
            
    def add_number( self, number, side ):
        side = side.lower()
        if side in ['left','right']:
            getattr( self, '{}_numbers'.format( side ) ).append( number )
        else:
            raise ValueError( 'side must be either "left" or "right"!' )
        
    def get_left_objects( self ):
        """ Orders and returns objects on RHS of sparkline plot. """
        objects = []
        if self.left_label_out:
            objects.extend( self.left_labels  )
            objects.extend( self.left_numbers )
        else:
            objects.extend( self.left_numbers )
            objects.extend( self.left_labels  )
        return objects
    
    def get_right_objects( self ):
        """ Orders and returns objects on RHS of sparkline plot. """
        objects = []
        if self.right_label_out:
            objects.extend( self.right_labels  )
            objects.extend( self.right_numbers )
        else:
            objects.extend( self.right_numbers )
            objects.extend( self.right_labels  )
        return objects
    
    def create_subgrid( self , wspace=0.0, hspace=0.0 , width_ratios=None, **kwargs ):
        """ This creates a subgrid gridspec and assigns those gridspecs to the objects of the plot """
        possible_sections = ['left_labels','right_labels','left_numbers','right_numbers']
        columns           = 1 + sum( [ len( getattr( self, x ) ) for x in possible_sections ] )
        
        # Deal with width ratios
        if width_ratios:
            # Note!  This will ignore the data_axis_width attribute
            if len(width_ratios) != columns:
                raise ValueError( 'Width ratios given do not have the same number of elements ({}) as the plot ({})!'.format( len(width_ratios), columns ) )
            else:
                wr = width_ratios
        else:
            # Use width ratio from objects if none given
            wr = []
            
            # insert data axis width in center
            wr.extend( [l.width for l in self.get_left_objects()]  )
            wr.append( self.data_axis_width ) 
            wr.extend( [r.width for r in self.get_right_objects()] )
            
        self.subgrid = gridspec.GridSpecFromSubplotSpec( 1, columns, subplot_spec=self.outer_grid ,
                                                         wspace=wspace, hspace=hspace, width_ratios=wr )
        # Assign axes objects to the labels and self
        plot_items = self.get_left_objects() + [self] + self.get_right_objects()
        for i, obj in enumerate( plot_items ):
            # Assign subgridspec
            if isinstance(obj,Sparkline):
                # Create axes for the sparkline data
                obj.get_data_axes( self.subgrid[i] )
            else:
                # Create axes for the Annotations
                obj.get_axes( self.fig, self.subgrid[i] )
            
    def set_gridspec( self, gs ):
        """ Echos set_gridspec of Annotation Class objects. This is where the data will be plotted. """
        self.data_gridspec = gs
        
    def get_data_axes( self, gs=None ):
        if gs:
            self.set_gridspec( gs )
            
        self.ax = plt.Subplot( self.fig, self.data_gridspec )
        
        # Just in case
        self.data_ax = self.ax
        
    def draw( self ):
        """ 
        Adds all subplots and objects to the given figure 
        This should probably be overwritten in children classes or not done at all for custom plots.
        For instance, if you want to change the color of the text you need to pass custom kwargs to the object.
        """
        self.fig.add_subplot( self.ax )
        for obj in self.get_left_objects() + self.get_right_objects():
            obj.draw()
            
        return None

    def plot_2d_data( self, x, y, regions=None, xlim=None, ylim=None, show_axes=False, plot_kwargs=None ):
        """ 
        This function plots 2d data on the main plot grid.
        Regions of shading can be supplied as a list of dictionary properties.
            e.g. {'low':0,'high':30,'color':'red','alpha':0.2}
            low/high are required
            color/alpha must be supplied but can be None or value
            Keys are 'low', 'high', 'color', and 'alpha'
                low/high refer to x-axis values
                color is for the shading
                alpha controls the level of transparency
        ylim can be supplied to control the ylim of the plot
        """
        # example plot_kwargs {'color':'k', 'linestyle':'', 'marker':'None','lw':0.75, 'alpha':1}
        if plot_kwargs is not None:
            self.ax.plot( x, y, **plot_kwargs )
        else:
            self.ax.plot( x, y )
        if regions:
            for r in regions:
                self.ax.axvspan( r['low'], r['high'], color=r['color'], alpha=r['alpha'] )
        if ylim:
            self.ax.set_ylim( ylim[0], ylim[1] )
        if xlim:
            self.ax.set_xlim( xlim[0], xlim[1] )
        if not show_axes:
            # Remove the frame
            self.ax.set_frame_on( False )
            self.ax.xaxis.set_visible( False )
            self.ax.yaxis.set_visible( False )

    def hbox( self, data, regions=None, xlim=None, ylim=None, show_axes=False, hbox_kwargs=None ):
        # example hbox_kwargs {'sym':'o'}
        if hbox_kwargs is not None:
            self.ax.boxplot( data, vert=False, positions=[0], **hbox_kwargs )
        else:
            self.ax.boxplot( data, vert=False, positions=[0] )
        if regions:
            for r in regions:
                self.ax.axvspan( r['low'], r['high'], color=r['color'], alpha=r['alpha'] )
        if ylim:
            self.ax.set_ylim( ylim[0], ylim[1] )
        if xlim:
            self.ax.set_xlim( xlim[0], xlim[1] )
        if not show_axes:
            # Remove the frame
            self.ax.set_frame_on( False )
            self.ax.xaxis.set_visible( False )
            self.ax.yaxis.set_visible( False )

    def show_xticks( self , label_tuples=None, fontsize=8 ):
        """ Turn on xticks, if this is a single plot or if it's the last one in a stack of sparklines """
        # Turn the x axis back on
        self.ax.xaxis.set_visible( True )
        self.ax.minorticks_on    ( )
        self.ax.xaxis.set_ticks_position( 'bottom' )
        
        if label_tuples:
            xt, xtl = zip(*label_tuples)
            self.ax.set_xticks     ( xt  )
            self.ax.set_xticklabels( xtl , fontsize=fontsize )

    @staticmethod
    def get_histogram_trace( data, bins ):
        """ Returns x,y data for a histogram that can be plotted as a line. """
        n, bins   = np.histogram( data, bins=bins )
        bin_width = bins[1]-bins[0]
        x         = bins[1:] - bin_width/2
        return x, n
    
class SparkTimeline( Sparkline ):
    """ Child class that deals with data on a date-based x-axis """
    
    def highlight_weekdays( self , *args, **kwargs ):
        """ Algorithm to highlight weekdays.  Kwargs go into the axes.axvspan argument. """
        # Let's get the data axis xlims and minor ticks, which should be in ordinal format.
        xlims  = self.ax.get_xlim()
        xticks = self.ax.get_xticks( 'minor' )

        # In some cases, like if we've killed the axes, there will be no xticks.  reconstruct here.
        if not any(xticks):
            start  = np.ceil(  xlims[0] ) 
            end    = np.floor( xlims[1] ) 
            xticks = np.arange( start , end+1 )
        
        # Define defaults
        color  = kwargs.pop( 'color' , 'green' )
        alpha  = float( kwargs.pop( 'alpha' , 0.4 ) )
        
        active = False
        for tick in xticks:
            # Get the weekday -- 0=Monday, 4=Friday
            day       = datetime.date.fromordinal( int(tick) ).weekday()
            is_monday = (day==0)
            is_friday = (day==4)
            if not active and is_monday:
                active = True
                first  = tick - 0.4 # We want the shading to encompass datapoints so they are +/- 25%
            elif active and is_friday:
                self.ax.axvspan( first, tick+0.4, color=color, alpha=alpha, **kwargs )
                del first
                active = False
            elif not active and is_friday:
                # This might happen in the beginning of the plot
                self.ax.axvspan( xlims[0], tick+0.4, color=color, alpha=alpha, **kwargs )
                
        # Now, if we are still active but the end of the xlims isn't to Friday yet . . .
        if active:
            self.ax.axvspan( first, xlims[1], color=color, alpha=alpha, **kwargs )
            
        self.ax.set_xlim( *xlims )
        
        
    @staticmethod
    def is_weekday( ordinal ):
        """ Returns boolean if the ordinal date is a weekday. """
        return datetime.datetime.fromordinal( int(ordinal) ).weekday() <= 4
    
    
    @staticmethod
    def is_weekend( ordinal ):
        """ Returns boolean if the ordinal date is a weekday. """
        return datetime.datetime.fromordinal( int(ordinal) ).weekday() > 4
    
class BooleanTimeline( SparkTimeline ):
    """ Child class of timeline sparklines meant for 'boolean' plots per day -- e.g. Instrument Usage. """
    def get_data_axes( self, gs=None ):
        """ Extends parent method with some standardized axis parameters. """
        super( SparkTimeline, self ).get_data_axes( gs=gs )
        
        # Remove the frame
        self.ax.set_frame_on( False )
        self.ax.xaxis.set_visible( False )
        self.ax.yaxis.set_visible( False )
        
        # Set xticklabel formatter to something like Apr 04
        self.ax.xaxis.set_major_formatter( matplotlib.dates.DateFormatter( '%b %d' ) )
        
        # This is a boolean timeline.  So we're going to zoom in on the "1" values.  I hope we can do this first.
        self.ax.set_ylim( 0.9, 1.1 )
        
        
    def show_xticks( self , label_day='monday' ):
        """ Turn on xticks, if this is a single plot or if it's the last one in a stack of sparklines """
        # Turn the x axis back on
        self.ax.xaxis.set_visible( True )
        self.ax.minorticks_on    ( )
        self.ax.xaxis.set_ticks_position( 'bottom' )
        self.ax.xaxis.set_minor_locator ( matplotlib.dates.DayLocator() )
        
        # Label the day of interest, by default Mondays
        days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
        
        try:               day_to_label = days.index( label_day.lower() )
        except ValueError: day_to_label = None
        
        if day_to_label > -1:
            xticks = self.ax.get_xticks( 'minor' )
            xinfo  = [ (d, datetime.date.fromordinal(int(d)).strftime('%b %d')) for d in xticks
                       if datetime.date.fromordinal(int(d)).weekday()==day_to_label ]
            
            xt, xtl = zip(*xinfo)
            self.ax.set_xticks     ( xt  )
            self.ax.set_xticklabels( xtl , fontsize=8 )
            
            
    def get_usage_percs( self, dates, usage ):
        """ Calculates the percentage of usage on weekdays and weekends. """
        output   = {'weekdays' : 0. , 'weekends' : 0. }
        weekdays = np.array( [used for day,used in zip( dates, usage ) if self.is_weekday( day )] )
        weekends = np.array( [used for day,used in zip( dates, usage ) if self.is_weekend( day )] )
        
        if weekdays.any(): output['weekdays'] = 100. * weekdays.mean()
        if weekends.any(): output['weekends'] = 100. * weekends.mean()
        
        return output
    
    
    @staticmethod
    def used_dates_to_usage( used, start, end=None ):
        """Converts a list of used dates to a trendline of booleans from start to end (default=today)"""
        used_ordinals = [ float(d.toordinal()) for d in used ]
        
        if end == None:
            end = datetime.date.today()
            
        # dates is the x axis and usage is the y axis
        dates = matplotlib.dates.drange( start, end, datetime.timedelta(days=1) )
        usage = [d in used_ordinals for d in dates]
        
        return dates, usage
    
    
    @classmethod
    def from_used_dates( cls, used, start, end=None, left_label='', label_usage_percs=False ):
        """ 
        Helper method to create a single sparkline based on a list of datetime objects where the item was used/true 
        Default behavior uses today as the end date if none is supplied. 
        """
        dates, usage = cls.used_dates_to_usage( used, start, end=end )
        
        fig = plt.figure( figsize=(8,0.5) )
        gs  = gridspec.GridSpec( 1, 1, left=0.00, right=1.0, bottom=0.45, top=0.95 )
        
        spark = cls( fig, gs[0], data_axis_width=8 )
        
        if left_label:
            spark.add_label( Label( left_label, width=1 ) , 'left' )
            
        if label_usage_percs:
            usage_percs = spark.get_usage_percs( dates, usage )
            
            spark.add_number( Number( usage_percs['weekdays'], '{:0.1f}%', color='green', alpha=0.8 ), 'right' )
            spark.add_number( Number( usage_percs['weekends'], '{:0.1f}%' ), 'right' )
            
        spark.create_subgrid( )
        
        spark.ax.plot    ( dates, usage, 'o', markersize=4 )
        spark.show_xticks( )
        spark.highlight_weekdays( )
        spark.draw()
        
        return fig
    
        
    @classmethod
    def multi_use_plot( cls, usage_dict, start, end=None, label_usage_percs=False , sort_keys=True ):
        """ Takes in list of multiple usage information and plots them together by key. """
        num = len(usage_dict)

        fig = plt.figure( figsize=(8, 0.25 * num ) )
        gs  = gridspec.GridSpec( num, 1, left=0.00, right=1.0, bottom=0.08, top=0.98, hspace=0.2, wspace=0.0 )
        
        if sort_keys:
            key_list = sorted( usage_dict.keys(), key=lambda x: x.lower() )
        else:
            key_list = usage_dict.keys()

        for i, key in enumerate( key_list ):
            spark        = cls( fig, gs[i] )
            dates, usage = spark.used_dates_to_usage( usage_dict[key], start, end=end )
            
            spark.add_label( Label( key, width=1 ) , 'left' )
            
            if label_usage_percs:
                usage_percs = spark.get_usage_percs( dates, usage )
                
                spark.add_number( Number( usage_percs['weekdays'], '{:0.1f}%', color='green', alpha=0.8 ), 'right' )
                spark.add_number( Number( usage_percs['weekends'], '{:0.1f}%' ), 'right' )
                
            spark.create_subgrid( )
            spark.ax.plot( dates, usage, '8', markersize=4 )
            spark.ax.set_xlim( dates[0]-2, dates[-1] )
            
            # Show ticks on last plot
            if i == (num - 1):
                spark.show_xticks( )
                
            spark.highlight_weekdays( )
            spark.draw( )
            
        return fig
                    
class SparkHistogram( Sparkline ):
    """ Child class that creates and modulates histogram data """
    
    @classmethod
    def strand_bias( cls, fig, gs, data, bc_label='', fontsize=10 ):
        """ 
        Class method to create a strand bias plot in one fell swoop. 
        You might want to expand the width of your figure if you send in a bc label like IonHDdual_0101. . .
        """
        spark = cls( fig, gs )
        x,y   = spark.get_histogram_trace( data, bins=np.linspace(0,100,101) )
        low   = y[x<=30].sum()
        high  = y[x>=70].sum()
        
        if bc_label:
            spark.add_label     ( Label ( left_label, width=None , fontsize=fontsize) , 'left' )
            
        spark.add_number    ( Number( low , color='red' , fontsize=fontsize) , 'left' )
        spark.add_number    ( Number( high, color='red' , fontsize=fontsize), 'right' )
        spark.create_subgrid( )
        
        spark.ax.axis    ('off')
        spark.ax.plot    ( x, y, color='green', lw=0.75 )
        spark.ax.axvspan (  0,  30, color='red', alpha=0.2 )
        spark.ax.axvspan ( 70, 100, color='red', alpha=0.2 )
        spark.ax.set_ylim( -1, spark.ax.get_ylim()[1]+5 )
        
        spark.draw()
        return spark
    

    @classmethod
    def multi_barcode_strand_bias( cls, amplicat_dict ):
        """ 
        Input is a dictionary of CSVBase objects with barcode names as keys.
        Results could come from Amplicat or CovAnalaysis objects.
        """
        bc_num = len( amplicat_dict )
        
        fig = plt.figure( figsize=(4,0.25*bc_num) )
        gs  = gridspec.GridSpec( bc_num, 1, left=0.00, right=1.0, bottom=0.05, top=0.95 )
        
        for i, k in enumerate( sorted( amplicat_dict.keys() )):
            spark = cls( fig, gs[i] )
            data  = amplicat_dict[ k ].get_column( 'fwd_pc', float )
            x,y   = spark.get_histogram_trace( data, bins=np.linspace(0,100,101) )
            low   = y[x<=30].sum()
            high  = y[x>=70].sum()
            
            spark.add_label     ( Label ( k , width=None , fontsize=10) , 'left' )
            
            spark.add_number    ( Number( low , color='red' , fontsize=10) , 'left' )
            spark.add_number    ( Number( high, color='red' , fontsize=10), 'right' )
            spark.create_subgrid( )
            
            spark.ax.axis('off')
            spark.ax.plot( x, y, color='green', lw=0.75 )
            spark.ax.axvspan(  0,  30, color='red', alpha=0.2 )
            spark.ax.axvspan( 70, 100, color='red', alpha=0.2 )
            spark.ax.set_ylim( -1, spark.ax.get_ylim()[1]+5 )
            
            spark.draw()
            
        return fig
        
    
    @staticmethod
    def get_histogram_trace( data, bins ):
        """ Returns x,y data for a histogram that can be plotted as a line. """
        n, bins   = np.histogram( data, bins=bins )
        bin_width = bins[1]-bins[0]
        x         = bins[1:] - bin_width/2
        return x, n

##################################################
# Test functions likely to be deleted
##################################################
def single_fwd_bias( data ):
    fig = plt.figure( figsize=(2,0.25) )
    gs  = gridspec.GridSpec( 1, 1, left=0.00, right=1.0, bottom=0.05, top=0.95 )
    
    spark = SparkHistogram( fig, gs[0] )
    x,y   = spark.get_histogram_trace( data, bins=np.linspace(0,100,101) )
    low   = y[x<=30].sum()
    high  = y[x>=70].sum()
    
    spark.add_number    ( Number( low , color='red' , fontsize=12) , 'left' )
    spark.add_number    ( Number( high, color='red' , fontsize=12), 'right' )
    spark.create_subgrid( )
    
    spark.ax.axis('off')
    spark.ax.plot( x, y, color='green', lw=0.75 )
    spark.ax.axvspan(  0,  30, color='red', alpha=0.2 )
    spark.ax.axvspan( 70, 100, color='red', alpha=0.2 )
    spark.ax.set_ylim( -1, spark.ax.get_ylim()[1]+5 )
    
    spark.draw()
    return fig

# When using long labels, you may want to expand the width of your figure....
def labeled_single_fwd_bias( data , left_label='' ):
    fig = plt.figure( figsize=(4,0.25) )
    gs  = gridspec.GridSpec( 1, 1, left=0.00, right=1.0, bottom=0.05, top=0.95 )
    
    spark = SparkHistogram( fig, gs[0] )
    x,y   = spark.get_histogram_trace( data, bins=np.linspace(0,100,101) )
    low   = y[x<=30].sum()
    high  = y[x>=70].sum()

    #spark.add_label     ( Label ( '113', fontsize=12) , 'left' )
    #spark.add_label     ( Label ( '113', width=None , fontsize=10) , 'left' )
    if left_label:
        spark.add_label     ( Label ( left_label, width=None , fontsize=10) , 'left' )
        
    spark.add_number    ( Number( low , color='red' , fontsize=10) , 'left' )
    spark.add_number    ( Number( high, color='red' , fontsize=10), 'right' )
    spark.create_subgrid( )
    
    spark.ax.axis('off')
    spark.ax.plot( x, y, color='green', lw=0.75 )
    spark.ax.axvspan(  0,  30, color='red', alpha=0.2 )
    spark.ax.axvspan( 70, 100, color='red', alpha=0.2 )
    spark.ax.set_ylim( -1, spark.ax.get_ylim()[1]+5 )
    
    spark.draw()
    return fig

def multi_labeled_fwd_bias( amplicat_dict ):
    bc_num = len( amplicat_dict )
    
    fig = plt.figure( figsize=(4,0.25*bc_num) )
    gs  = gridspec.GridSpec( bc_num, 1, left=0.00, right=1.0, bottom=0.05, top=0.95 )

    for i, k in enumerate( sorted( amplicat_dict.keys() )):
        spark = SparkHistogram( fig, gs[i] )
        data  = amplicat_dict[ k ].get_column( 'fwd_pc', float )
        x,y   = spark.get_histogram_trace( data, bins=np.linspace(0,100,101) )
        low   = y[x<=30].sum()
        high  = y[x>=70].sum()

        spark.add_label     ( Label ( k , width=None , fontsize=10) , 'left' )
        
        spark.add_number    ( Number( low , color='red' , fontsize=10) , 'left' )
        spark.add_number    ( Number( high, color='red' , fontsize=10), 'right' )
        spark.create_subgrid( )
        
        spark.ax.axis('off')
        spark.ax.plot( x, y, color='green', lw=0.75 )
        spark.ax.axvspan(  0,  30, color='red', alpha=0.2 )
        spark.ax.axvspan( 70, 100, color='red', alpha=0.2 )
        spark.ax.set_ylim( -1, spark.ax.get_ylim()[1]+5 )
        
        spark.draw()
        
    return fig
    
