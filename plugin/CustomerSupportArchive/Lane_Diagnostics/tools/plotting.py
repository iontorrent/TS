import matplotlib.pyplot as plt 
from matplotlib.figure import Figure

import numpy as np
from scipy import stats

import re, string

#################################################
#                   GLOBALS                     #
#################################################
BOXCOLORS   = ( 'pink','lightblue','lightgreen', 'sandybrown', 'mediumpurple', 'palegoldenrod' )
PVALCOLORS  = {'0.01':'green', '0.05':'chartreuse', '0.1':'orange', 'else':'gray' }
# Expanded letters to handle more than 26 elements
LETTERS     = list( string.ascii_uppercase ) + sorted( list( set( [ '{}{}'.format(l_i,l_j) for l_j in string.ascii_uppercase for l_i in string.ascii_uppercase ] ) ) )
FMT_N       = '\nN={}'
class _dummy:
    ''' Dummy class for duck typing inputs to prevent unnecessary imports '''
    type    = ''
    name    = ''
    series  = ''
    ff_tmin = None
    ff_tmax = None
_backend = 'pyplot'

def set_backend( backend='pyplot' ):
    global _backend

    backend = backend.lower()
    allowed = ( 'pyplot', 'figure' )
    if backend in allowed:
        _backend = backend
    else:
        allowed_str = ', '.join( [ '"{}"'.format(a) for a in allowed ] ) 
        raise ValueError( 'backend "{}" is unknown.  Choose from the following choices: \n    {}'.format( allowed_str ) )

#################################################
#                  FULL FIGURES                 #
#################################################
# NOTE: In order to support multiple backends, all functions in this section should support both the 
#       pyplot backend and the Figure backend.  In order to do that, new figure creation should use the 
#       "figure" function defined in this module instead of calling pyplot.figure or Figure directly
def figure( *args, **kwargs ):
    ''' Generic figure creation for pyplot or Figure backends '''
    if _backend == 'pyplot':
        return plt.figure( *args, **kwargs )
    elif _backend == 'figure': 
        return Figure( *args, **kwargs )

def plot_spatial( data, metricname=None, prefix='', savedir='.', metricstats={}, chiptype=_dummy(), clim=None ):
    ''' 
    Makes a 2d plot of data.  
    Only some metrics require a chiptype to determine plot limits
    '''
    #TODO: This could be updated to be more consistnt with recent chipdb improvements (2/14/2019)
    #      Especially the creation of a figure within the plot. 

    name = ''
    basename = ''
    if prefix:
        prefix += '_'
    if metricname:
        basename = '%s%s' % (prefix,metricname)
        name     = basename + '_'

    # Get the metric-specific properties
    adjust = metricstats.get( '%smean' % name, data.mean() )
    lims, units = plot_params( metricname, adjust, chiptype )
    if clim:
        lims = clim

    # If image is too big,  have to skip datapoints...
    scale = 1
    # Pick a comon scale factor to preserve the aspect ratio
    for dim in data.shape:
        if dim >= 10000:
            # Maximum data size will be 1800 pixels in any dimension
            scale = max( scale, dim/1800 )
    # Plot the data
    extent = [0, data.shape[1]-1, 0, data.shape[0]-1] 
    plt.imshow   ( data[::scale,::scale] , extent=extent , interpolation='nearest' , origin='lower' , clim=lims )
    try: 
        metric_text = ' | (Mean, SD, P90) = (%.1f, %.1f, %.1f) %s\n(Q2, IQR, Mode) = (%.1f, %.1f, %.1f) %s' % (  metricstats.get( '%smean' % name, np.nan ),
                                                                                                                 metricstats.get( '%sstd'  % name, np.nan ),
                                                                                                                 metricstats.get( '%sP90'  % name, np.nan ),
                                                                                                                 units ,
                                                                                                                 metricstats.get( '%sq2'   % name, np.nan ),
                                                                                                                 metricstats.get( '%siqr'  % name, np.nan ),
                                                                                                                 metricstats.get( '%smode' % name, np.nan ),
                                                                                                                 units )
    except (KeyError, AttributeError):
        metric_text = ' | %s' % ( units )
    plt.title    ( '%s%s' % ( basename, metric_text ) )
    plt.colorbar ( shrink = 0.75 )
    plt.savefig  ( '%s/%sspatial.png' % ( savedir, name ))
    plt.close    ( )

def make_legend_fig( names=['blank'], colors=True, type='boxes', use_letters=False ):
    # Make two figs --> one for plots and the other for the legend
    fig         = Figure( facecolor='w') # dummy figure
    figlegend   = figure( facecolor='w' )
    ax          = fig.add_subplot( 111 )
    # make plots that legend will be built from
    if type == 'boxes':
        vals = [[0] for x in range( len(names) ) ]
        if use_letters:
            labels = []
            for i, name in enumerate( names ):
                text = LETTERS[i] + str(': ') + name
                labels.append( text )
        else:
            labels = names
        bp = subplot_box( ax, vals, labels, colors=colors, pvals=False )
        handles = bp['boxes']
    if type == 'pvals':
        keys = list( PVALCOLORS.keys() )
        keys.sort()
        for key in keys:
            if key != 'else':
                ax.plot( [0,1],[0,1], color=PVALCOLORS[key], label='p < '+key )
        handles, labels = ax.get_legend_handles_labels()
    # Make legend
    figlegend.legend( handles, labels, 'center' )
    # Dynamically adjust legend figure size
    max_len = 1
    for label in labels:
        new_max = len(label)
        if max_len<new_max: max_len=new_max
    num_labels = len(labels)
    # figsize = (width, height)
    figlegend.set_size_inches( 0.1*max_len+0.6, 0.21*num_labels+0.2 )
    figlegend.set_tight_layout( True ) 

    return figlegend

def histplot( data, bins=50, xlims=None, xlabel=None, title=None ): # TODO: silicon_plots
    fig = figure( facecolor='w', figsize=(7,6),  )
    ax = fig.add_subplot( 111 )
    ax.hist( data, bins=bins, range=xlims )
    if xlims:
        ax.set_xlim( xlims )
    if xlabel:
        ax.set_xlabel( xlabel )
    if title:
        ax.set_title( title )
    set_font( ax, 18 )
    fig.tight_layout  ( )

    return fig

def simple_boxplot( data, names, figsize=(3,3), fontsize=12, ylims=None, ylabel=None, xlabel=None, title=None, pvals=None, points=1, whis=1.5, colors=True, rotation=90, use_letters=False, rescale=False, relmax=None, fmt_n=FMT_N, show_mean=False ):
    ''' Make the standard boxplot with usual options. 
        Data should be entered as a list of lists and should be cleaned of Nones.

        points: Input marker size (float)
                If 0, no points will be displayed
        colors: True              : uses the default BOXCOLORS palate to color each box individually
                False, None, []   : Do not re-color the boxes
                [c1, c2, c3, ...] : Use the user-defined list of colors for each box
    '''

    fig = figure( facecolor='w', figsize=figsize )
    ax = fig.add_subplot( 111 )

    # Determine labels to use
    if use_letters:
        labels = []
        for i, name in enumerate( names ):
            labels.append( LETTERS[i] )
    else:
        labels = names
    # Make subplot
    subplot_box( ax, data, labels, fontsize=fontsize, ylims=ylims, ylabel=ylabel, xlabel=xlabel, title=title, pvals=pvals, points=points, whis=whis, colors=colors, rotation=rotation, rescale=rescale, relmax=relmax, fmt_n=FMT_N, show_mean=show_mean )
    # Make the figure
    fig.set_tight_layout( True )

    return fig

def simple_scatterplot( data, names, fontsize=12, figsize=(3,4), title=None, ylims=None, ylabel=None, use_letters=None, time_data=False, legend=False ):
    '''Generates a single scatterplot from a list of lists of pairs'''
    fig = figure( facecolor='w', figsize=figsize )
    ax = fig.add_subplot( 111 )

    # Determine labels to use
    if use_letters:
        labels = []
        for i, name in enumerate( names ):
            labels.append( LETTERS[i] )
    else:
        labels = names
    # Make subplot
    subplot_scatter( ax, data, labels, fontsize=fontsize, title=title, ylims=ylims, ylabel=ylabel, legend=legend )
    
    if time_data:
        fig.autofmt_xdate()

    # Make the figure
    fig.set_tight_layout( True )
    return fig


#################################################
#                  SUBPLOTS                     #
#################################################
'''These functions act on an input figure axes and, if specified, return an artist'''

def subplot_box( ax, data, labels, fontsize=12, ylims=None, ylabel=None, xlabel=None, title=None, pvals=None, points=1, whis=1.5, colors=True, rotation=90, rescale=False, relmax=None, fmt_n='\n({})', show_mean=False ):
    ''' Makes a SUBPLOT standard boxplot with usual options. 
        Requires an axes instance to be passed in for figure defined outside the function.
        The returned value is the boxplot artist dictionary for use in making legends
        **Designed to be called repeatedly to make an array of subplots

        Data should be entered as a list of lists and should be cleaned of Nones.

        points:     Input marker size (positive float)
                    If 0, no points will be displayed
        colors:     True               : uses the default BOXCOLORS palate to color each box individually
                    False, None, []    : Do not re-color the boxes
                    [c1, c2, c3, ...]  : Use the user-defined list of colors for each box
        rotation:   0, False, None, [] : No rotation
                    any integer        : A rotation by that number of degrees
    '''

    positions = list( range( len( labels ) ) )
    if fmt_n:
        newlabels = []
        for l, d in zip( labels, data ):
            newlabels.append( l+fmt_n.format( len(d) ) )
        labels = newlabels
    if rescale:
        data = rescale_data( data, relmax=relmax )
    if show_mean:
        # placeholder for changes to the formatting of the mean line
        # currently a dashed green line
        pass
    bp = ax.boxplot( data , positions=positions , sym='', whis=whis, patch_artist=True, meanline=show_mean, showmeans=show_mean )
    # Recolor the boxes
    if colors:
        if colors == True:
            colors = BOXCOLORS
        for i, patch in enumerate(bp['boxes']):
            # modulo handles re-cycling through colors if fewer colors than data sets
            ci = i % len(colors)
            color = colors[ci]
            patch.set_facecolor( color )
            patch.set_alpha( 0.5 )
    # add scatter points
    if points:
        l_pos = len(positions)
        if l_pos > 1: width = np.minimum( (len(positions)-1)*0.15, 0.5 )
        else:         width = 0.5

        for i in positions:
            y = data[i]
            x = np.random.normal( i, 0.1*width, size = len(y)  )
            ax.plot( x, y, 'ok', alpha=0.5, ms=10*2**np.log(points) )
    # Set X axis
    ax.set_xticks( positions )
    # Determine X axis label rotation
    if rotation:
        ax.set_xticklabels( labels, rotation=rotation )
    else:
        ax.set_xticklabels( labels, rotation=0 )

    # Add p-values
    if (pvals == True) or (pvals is None):
        pairs = []
        for i, ld1 in enumerate( data ):
            for j, ld2 in enumerate( data[i+1:] ):
                if len(ld1)>1 and len(ld2)>1:
                    pval = ttest( ld1, ld2 )
                    if pval < 0.01:   c = PVALCOLORS['0.01']
                    elif pval < 0.05: c = PVALCOLORS['0.05']
                    elif pval < 0.1:  c = PVALCOLORS['0.1']
                    else:             c = PVALCOLORS['else']
                    if c != PVALCOLORS['else']:
                        pairs.append( ( i, i+j+1, c ) )
        # If there are too many to plot and user was ambigious, don't plot!
        if pvals and not pairs:
            display_nopval_message_with_adjusted_ylims( ax, ylims )
        if pvals is None and len( pairs ) > 6:
            pairs = []
        if pairs:
            adjust_ylims_for_pvals( ax, ylims, pairs )

    # Set Y-axis
    if ylims:
        ax.set_ylim( ylims )
    if ylabel:
        ax.set_ylabel( ylabel )
    if xlabel:
        ax.set_xlabel( xlabel )
    # Set Title
    if title:
        ax.set_title( title )
    # Set Font Size
    set_font( ax, fontsize )
    # return boxplot artist for legend use
    return bp

def subplot_grouped_boxes( ax, grouped_data, group_labels, data_labels, pad=0.3, max_width=0.4, fontsize=12, ylims=None, ylabel=None, xlabel=None, title=None, pvals=None, points=0.04, whis=1.5, colors=True, rotation=90, fmt_n='\n({})', show_mean=False, **kwargs ):
    ''' Creates groups of boxplots around xticks
            EX:  group --> lane --> data 
                    group is x-tick
                    lane is reused at each tick
                    data is the corresponding data set for the given group and lane
        Input:  ax = axes object
                grouped_data    = list of lists of lists of data ordered by group-->label
                    [ [ [g1_lane1_data], [g2_lane2_data],...], [ [...],[...],...], ...]
                group_labels    = list of group names
                data_labels     = list of data_labels
        Output: artists
    '''
    if fmt_n:
        newlabels = []
        flat_group_data = []
        for i, data_list in enumerate( grouped_data ):
            flat_group_data.append( [ val for d in data_list for val in d ] )
        for l, d in zip( group_labels, flat_group_data ):
            newlabels.append( l+fmt_n.format( len(d) ) )
        group_labels = newlabels
    if show_mean:
        # placeholder for changes to the formatting of the mean line
        # currently a dashed green line
        pass

    num_labels = len( data_labels )
    group_pos = np.arange( len( group_labels ) )

    total_padding = pad * ( num_labels - 1 )
    width = ( max_width - total_padding ) / num_labels

    kwargs['widths'] = width

    def data_positions( i ):
        span = width*num_labels + pad*(num_labels - 1)
        ends = ( span - width )/2
        x = np.linspace( -ends, ends, num_labels )
        return x + i 

    bp_artists = []
    pairs = None
    g_pairs = None
    for i, group in enumerate( grouped_data, start=0):
        data_pos = data_positions( i )
        bp = ax.boxplot( group, positions=data_pos, patch_artist=True, meanline=show_mean, showmeans=show_mean, **kwargs )
        bp_artists.append( bp )
        # Recolor the boxes
        if colors:
            if colors == True:
                colors = BOXCOLORS
            for j, patch in enumerate(bp['boxes']):
                # modulo handles re-cycling through colors if fewer colors than data sets
                cj = j % len(colors)
                color = colors[cj]
                patch.set_facecolor( color )
                patch.set_alpha( 0.5 )
        # add scatter points
        if points:
            for j, pos in enumerate( data_pos ):
                y = group[j]
                x = np.random.normal( pos, points, size = len(y)  )
                ax.plot( x, y, 'ok', alpha=0.5 )
        # Add p-values within groups
        if (pvals == True) or (pvals is None):
            pairs = []
            for j, ld1 in enumerate( group ):
                for k, ld2 in enumerate( group[j+1:] ):
                    if len(ld1)>1 and len(ld2)>1:
                        pval = ttest( ld1, ld2 )
                        if pval < 0.01:   c = PVALCOLORS['0.01']
                        elif pval < 0.05: c = PVALCOLORS['0.05']
                        elif pval < 0.1:  c = PVALCOLORS['0.1']
                        else:             c = PVALCOLORS['else']
                        if c != PVALCOLORS['else']:
                            pairs.append( ( data_pos[j], data_pos[j+k+1], c ) )
            # If there are too many to plot and user was ambigious, don't plot!
            if pvals is None and len( pairs ) > 6:
                pairs = []
            if pairs:
                adjust_ylims_for_pvals( ax, ylims, pairs )

    # Add p-values between groups
    if (pvals == True) or (pvals is None):
        g_pairs = []
        for i, ld1 in enumerate( grouped_data ):
            for j, ld2 in enumerate( grouped_data ):
                for k in range( len( data_labels ) ):
                    if len(ld1[k])>1 and len(ld2[k])>1:
                        pval = ttest( ld1[k], ld2[k] )
                        if pval < 0.01:   c = PVALCOLORS['0.01']
                        elif pval < 0.05: c = PVALCOLORS['0.05']
                        elif pval < 0.1:  c = PVALCOLORS['0.1']
                        else:             c = PVALCOLORS['else']
                        if c != PVALCOLORS['else']:
                            g_pairs.append( ( data_positions(i)[k], data_positions(j)[k], c ) )
        if pvals is None and len( pairs ) > 6:
            g_pairs = []
        if g_pairs:
            adjust_ylims_for_pvals( ax, None, g_pairs )
    if pvals and (not pairs and not g_pairs):
        display_nopval_message_with_adjusted_ylims( ax, ylims )

    proxy_artists = bp_artists[-1]['boxes']
    ax.legend( proxy_artists, data_labels, loc='best',fontsize=fontsize)

    # Set X axis
    ax.set_xticks( group_pos )
    # Determine X axis label rotation
    if rotation:
        ax.set_xticklabels( group_labels, rotation=rotation )
    else:
        ax.set_xticklabels( group_labels, rotation=0 )
    # Set Y-axis
    if ylims:
        ax.set_ylim( ylims )
    if ylabel:
        ax.set_ylabel( ylabel )
    if xlabel:
        ax.set_xlabel( xlabel )
    # Set Title
    if title:
        ax.set_title( title )
    # Set Font Size
    set_font( ax, fontsize )

    ax.autoscale( enable=True, axis='x', tight=False )

    return bp_artists

def subplot_scatter( ax, data, labels, fontsize=12, title=None, ylims=None, ylabel=None, legend=False ):
    ''' Makes a SUBPLOT standard plot with usual options. 
        Requires an axes instance to be passed in for figure defined outside the function.
        **Designed to be called repeatedly to make an array of subplots

        Data should be entered as a list of lists of pairs and should be cleaned of Nones.
    '''

    for i, d in enumerate( data ):
        if d:
            x, y = zip(*d)
            ax.plot( x, y, '.',  label=labels[i] )
    if ylims:
        ax.set_ylim( ylims )
    if ylabel:
        ax.set_ylabel( ylabel )
    # Set Title
    if title:
        ax.set_title( title )
    if legend:
        xlims = ax.get_xlim()
        delta = xlims[1]-xlims[0]
        ax.set_xlim( xlims[0], xlims[1]+0.3*delta )
        ax.legend( loc='center right' )
    # Set Font Size
    set_font( ax, fontsize )

#########################################################
#                   HELPER FUNCTIONS                    #
#########################################################
def better_boxplot_ylims( data_list, scalar=3 ):
    ''' Loops through list of lists for data.  Calculates lim_max and lim_min for a single boxplot.  Sets ylim_max to the largest lim_max, ylim_min to the smalles lim_min'''
    calc_ylims = None
    for d in data_list:
        if len(d) > 2:
            med = np.median( d )
            iqr = np.subtract( *np.nanpercentile( d, [75, 25] ) )
            lim_max = med + scalar*iqr
            lim_min = med - scalar*iqr
            if calc_ylims:
                if calc_ylims[0]>lim_min: calc_ylims[0]=lim_min
                if calc_ylims[1]<lim_max: calc_ylims[1]=lim_max
            else: calc_ylims = [lim_min, lim_max]
    return calc_ylims

def adjust_ylims_for_pvals( ax, ylims, pairs ):
    yl = ax.get_ylim()
    # check if specified lower bound exceeds both autobounds
    if ylims and (ylims[0] is not None and ylims[1] is None):
        if ylims[0]>yl[0] and ylims[0]>yl[1]:
            ylims[1] = ylims[0] + (yl[1]-yl[0])
    # check if specified upper bound exceeds both autobounds
    if ylims and (ylims[1] is not None and ylims[0] is None):
        if ylims[1]>yl[0] and ylims[1]>yl[1]:
            ylims[0] = ylims[1] - (yl[1]-yl[0])

    imgfrac = 0.04*len(pairs) - 0.02
    imgfrac = min( imgfrac, 0.5 )
    if ylims is None or ( ylims[0] is None and ylims[1] is None ):
        # Free to adjust limits
        yl = ax.get_ylim()
        newsize = (yl[1] - yl[0])/(1-imgfrac)
        bot = yl[1] - newsize
        spacing = (newsize * imgfrac)/len(pairs)
    elif ylims[1] is None:
        # lower bound is fixed.  Let's put p-values at the top
        ax.set_ylim( ylims )
        yl = ax.get_ylim()
        newsize = (yl[1] - yl[0])/(1-imgfrac)
        bot = yl[1]
        spacing = (newsize * imgfrac)/len(pairs)
        ylims[1] = bot + spacing * (len(pairs )+1)
    elif ylims[0] is None:
        # upper bound is fixed
        ax.set_ylim( ylims )
        yl = ax.get_ylim()
        newsize = (yl[1] - yl[0])/(1-imgfrac)
        bot = yl[1] - newsize
        spacing = (newsize * imgfrac)/len(pairs)
        ylims[0] = bot
    else:
        # Both axes are fixed.  Let's just try to squeeze it in.
        yl = ylims
        newsize = (yl[1] - yl[0])
        bot = yl[0]
        spacing = (newsize * imgfrac)/len(pairs)
    for pair in pairs:
        bot += spacing
        ax.plot( (pair[0],pair[1]), (bot, bot), '.-', color=pair[2] )

def display_nopval_message_with_adjusted_ylims( ax, ylims ):
    yl = ax.get_ylim()
    # check if specified lower bound exceeds both autobounds
    if ylims[0] is not None and ylims[1] is None:
        if ylims[0]>yl[0] and ylims[0]>yl[1]:
            ylims[1] = ylims[0] + (yl[1]-yl[0])
    # check if specified upper bound exceeds both autobounds
    if ylims[1] is not None and ylims[0] is None:
        if ylims[1]>yl[0] and ylims[1]>yl[1]:
            ylims[0] = ylims[1] - (yl[1]-yl[0])

    imgfrac = 0.05
    if (ylims is None) or (ylims[0] is None) or ( ylims[0] is None and ylims[1] is None ):
        # no upper bound OR upper bound is fixed
        ax.set_ylim( ylims )
        yl = ax.get_ylim()
        newsize = (yl[1] - yl[0])/(1-imgfrac)
        ylims[0] = yl[1] - newsize
        msg_loc = (0.95, 0.01,)
    elif ylims[1] is None:
        # lower bound is fixed.  Let's put p-val message at the top
        ax.set_ylim( ylims )
        yl = ax.get_ylim()
        newsize = (yl[1] - yl[0])/(1-imgfrac)
        bot = yl[1]
        spacing = (newsize * imgfrac)
        # update upper limit
        ylims[1] = bot + spacing
        msg_loc = (0.95, 0.95,)
    else:
        # Both axes are fixed. Just gonna leave it at the bottom
        msg_loc = (0.95, 0.01,)
    ax.text( msg_loc[0], msg_loc[1], 'All p-vals > 0.1', verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=6 )


def rescale_data( data, relmax=None ):
    # check if list of lists
    lol_any = any( isinstance( el, list ) for el in data )
    lol_all = all( isinstance( el, list ) for el in data )
    if lol_all:
        # Data is a list of lists
        data=rescale_multi_array(data, relmax=relmax )
    elif not lol_any:
        # Data is strictly a single list
        data=rescale_single_array(array, relmax=relmax )
    else: print( 'Data is not a list or list of lists' )
    return data

def rescale_multi_array( data, relmax=None):
    ''' 
    Input can be a list of lists or np.array of np.arrays of individual numerical elements
    The output will be of the same type
    A list will be converted to a numpy.array before relative scaling with dtype=np.float64
    Data will be scaled by the ABSOLUTE MAXIMUM, then converted to a list if necessry
    '''
    if relmax is None: 
        relmax=-1
        for array in data:
            new_max=abs( max(array,key=abs) )
            if relmax<new_max: relmax=new_max
    for i, array in enumerate(data):
        data[i] = rescale_single_array(array,relmax)
    return( data )    

def rescale_single_array( array, relmax=None ):
    ''' 
    Input can be a list or numpy.array of individual numerical elements
    The output will be of the same type
    A list will be converted to a numpy.array before relative scaling with dtype=np.float64
    Data will be scaled, then converted to a list if necessry
    '''   
    # Make sure max is set to a value
    if relmax is None:
        relmax=abs( max(array,key=abs) )
    if isinstance( array, list ):
        vals = np.array( array, dtype=np.float64 )
        return list( vals/relmax )
    elif isinstance( array, np.ndarray ):
        return array/relmax
    else:
        print( 'Input is not a list or a np.ndarray' )

def ttest( array1 , array2 , one_tailed=False , equal_var=False , verbose=False):
    """                                                                                               
    My own implementation of the t-test, which defaults to unequal_var (Welch's t-test)               
    Assumes a two-tailed test.                                                                        
    """
    x1 = np.array( array1 , float )
    x2 = np.array( array2 , float )
    if x1.size < 1 or x2.size < 1:
        return 1

    tails = 2
    if one_tailed:
        tails = 1
        
    v1 , n1 = x1.var() , x1.size
    v2 , n2 = x2.var() , x2.size
    
    if equal_var:
        v12 = np.sqrt( ((n1-1) * v1**2 + (n2-1) * v2**2) / ( n1 + n2 - 2 ) )
        t   = ( x1.mean() - x2.mean() ) / ( v12 * np.sqrt( 1./n1 + 1./n2 ) )
        df  = n1 + n2 -2
    else:
        t      = np.nan_to_num( ( x1.mean() - x2.mean() ) / np.sqrt( v1/n1 + v2/n2 ) )
        df_top = np.nan_to_num( np.power( v1/n1 + v2/n2 , 2 ) )
        df_bot = np.nan_to_num( np.power( v1/n1 , 2 ) / ( n1-1 ) + np.power( v2/n2 , 2 ) / ( n2-1 ) )
        df     = np.nan_to_num( df_top / df_bot )
       
    pval = stats.t.sf( np.abs( t ) , df ) * tails
    
    if verbose:
        print( 't-statistic = %6.3f | p-value = %6.4f' % ( t , pval ) )
        
    return pval

def set_font( ax, fs ):
    for item in ( [ ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() ):
        item.set_fontsize( fs )

def downsample(data, scale=None, blocksize=None, subsample=True, clipedges=False ):
    """
    downsamples the data.  This is usefull during plotting large images to prevent segmentation faults.

    downsampling can either be performed by a scale factor, in which case the scale can either be None 
    or >= 1, or by blocksize however both cannot be used.  blocksize can either be a scaler or
    a two-integer tupple

    If subsample is true, then downsampling is performed by picking points out of data.  otherwise, it is
    performed by averaging blocks of data. 

    If clipedges is true, then local averaging ignores any regions which don't comprise a full block.  This 
    will reduce noise from edges which have fewer pixels in the average
    """
    if scale is None and blocksize is None:
        return data
    elif scale is not None and blocksize is not None:
        print( 'Scale and blocksize were both specified when downsampling data.  Utilizing the blocksize only.' )
    elif scale is not None:
        if scale == 1:
            return data
        elif scale < 1:
            'Scale must be >= 1.  No downsampling has been applied'
            return data
        blocksize = (scale, scale)

    if subsample:
        # Return the sub-sampled data, centered in the block
        return data[blocksize[0]/2::blocksize[0],blocksize[1]/2::blocksize[1]]

    # Calculate the size of locally averaged data
    avgsize = [x/float(y) for (x,y) in zip(data.shape,blocksize)]
    if clipedges:
        avgsize = [int(x) for x in avgsize]
    else:
        avgsize = [int(np.ceil(x)) for x in avgsize]
    avgdata = np.zeros(avgsize)

    # Perform local averaging
    for i in range(avgsize[0]):
        imin = i*blocksize[0]
        imax = min([(i+1)*blocksize[0],data.shape[0]])

        for j in range(avgsize[1]):
            jmin = j*blocksize[1]
            jmax = min([(j+1)*blocksize[1],data.shape[1]])

            # Redoing the average to ignore nan/inf
            #avgdata[i,j] = np.mean(data[imin:imax,jmin:jmax])
            avgdata[i,j] = np.ma.masked_invalid(data[imin:imax,jmin:jmax]).mean()
    return avgdata

def plot_params( metric, adjust=None, chiptype=_dummy() ):
    ''' Lookup function for standard plot limits and units depending on chip '''
    # strip off the full chip 
    if re.match( r'^fc_', metric ):
        metric = metric[3:]

    if metric in [ 'buffering', 'buffering_gc', 'GC_buffering_gc' ]:
        if any([adjust > 200, chiptype.type in ['P2'], 'P1.2' in chiptype.name]):  # TODO: may need to remove the P2 correction if buffering ever improves
            lims  = [0, 500]
        else:
            lims  = [0, 300]
        units = 'AU'
    elif metric in [ 'ebfiqr' ]:
        lims  = [0, 2000]
        units = 'uV'
    elif metric in [ 'bf_std' ]:
        if chiptype.series == 'pgm':
            lims  = [  0, 5000]
        else:
            lims  = [500, 2500]
        units = 'uV'
    elif metric in [ 'noise' ]:
        if chiptype.series == 'pgm':
            lims  = [  0, 150 ]
        else:
            if adjust < 100:
                lims  = [ 0, 200]
            elif adjust > 200:
                lims  = [ 0, 1000]
            else:
                lims  = [ 0, 400]
        units = 'uV'
    elif metric in [ 'delta_noise' ]:
        lims  = [ -20, 20 ]  
        units = 'uV'
    elif metric in [ 'ebfvals' ]:
        lims  = [-5000,5000]
        units = 'uV'
    elif metric in [ 'qsnr' ]:
        lims  = [0,200]
        units = 'AU'
    elif metric in [ 'gaincorr' ]:
        lims  = [950,1050]
        units = 'mV/V'
    elif metric in [ 'gain_iqr', 'gain_iqr_hd' ]:
        lims  = [0,40]
        units = 'mV/V'
    elif metric in [ 'phslopes' ]:
        lims  = [20,100]
        units = 'mV/pH'
    elif metric in [ 'delta_buffering_gc' ]:
        lims  = [-60,60]
        units = '$10^{-4} s/mV$'
    elif metric in [ 'buffering_gsc' ]:
        lims  = [0,200]
        units = '$10^{-2} s/pH$'
    elif metric in [ 'scales' ]:
        if adjust > 500:
            lims = [ 0, 6000 ]
        elif adjust < -500:
            lims = [ -6000, 0 ]
        else:
            lims  = [-3000, 3000]
        units = 'Counts'
    elif metric in [ 'slopes' ]:
        lims  = [-300000, 0]
        units = ''
    elif metric in [ 'actpix', 'pinned' ]:
        lims  = [0, 1]
        units = ''
    elif metric in [ 't0' ]:
        lims  = [ chiptype.ff_tmin, chiptype.ff_tmax ]
        units = 'frames'
    elif metric in [ 'driftrate' ]:
        lims  = [ -2000, 4000 ]
        units = 'uV/s'
    elif metric in [ 'driftrate_iqr' ]:
        lims  = [ 0, 750 ]
        units = 'uV/s'
    elif metric in [ 'offset_noisetest', 'offset_noisetest_iqr' ]:
        # This needs to be specified depending on the dynamic range
        lims  = [ None, None ]
        units = 'mV'
    else:
        lims  = [None, None]
        units = ''
        print( 'WARNING! Unknown metric for plotting: %s' % metric )
        #raise ValueError( 'Unknown metric for plotting: %s' % metric )
    return lims, units

