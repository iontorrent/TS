from scipy.stats import scoreatpercentile
import re
import numpy as np
import matplotlib.pyplot as plt
try:
    from . import plotting
except ImportError:
    print( 'WARNING: unable to import "plotting" in stats. some functions may be disabled' )

def named_stats( data, name='', histDir='', metricType='', maxsize=None, histlims=None ):
    ''' 
    This is a higher level version of percentiles

    Calculates data distribution properties and returns a dictionary 
    of metrics, each labeled with "name".  If name is unspecified,
    no flags will be applied. 

    If you specify histDir, then a histogram is generated and saved.
    If you don't, then the mode will not be returned

    metricType is the type of metric (e.g. "buffering", not "W2_buffering") 
    and is only used for making histograms.  Defaults to name if unspecified

    For very large datasets, you may want to specify a maxsize to speed up the calculation
    '''
    # Make sure the data is 1D
    flat = data.flatten()
    # Downsample large data sets
    if maxsize:
        scale = int(max( 1, np.ceil( len(flat)/maxsize ) ))
        flat = flat[::scale]

    # Append an underscore to the end of the name to separate from metrics
    if name:
        if name[-1] != '_':
            name += '_'

    # Calculate the metrics
    metrics = {}
    metrics[ '%smean' % name ] = flat[ ~np.isnan( flat ) ].mean()
    metrics[ '%sstd'  % name ] = flat[ ~np.isnan( flat ) ].std()
    percs                      = percentiles( flat )
    metrics[ '%sP10'  % name ] = float( percs['p10'] )
    metrics[ '%sP90'  % name ] = float( percs['p90'] )
    metrics[ '%sq1'   % name ] = float( percs['q1']  )
    metrics[ '%sq2'   % name ] = float( percs['q2']  )
    metrics[ '%sq3'   % name ] = float( percs['q3']  )
    metrics[ '%siqr'  % name ] = float( percs['iqr'] )

    # Plot the histogram
    if histDir:
        # Set limits for histogram if free limits were used
        if not metricType:
            metricType = name
        lims, units = plotting.plot_params( metricType, adjust=metrics[ '%smean' % name ] )
        if histlims:
            lims = histlims
        if lims[0] is None:
            lims[0] = flat.min()
        if lims[1] is None:
            lims[1] = flat.max()
        if not len( flat ):
            flat = np.zeros(1)
        if re.search( r'phslopes$', name ):
            bins = 41
        else:
            bins = 101

        hn,hb,_ = plt.hist   ( flat , bins = np.linspace( lims[0] , lims[1] , bins ) , align='mid' , hold=False )

        metrics[ '%smode' % name ] = hist_mode( hn , hb )
        plt.xlabel ( '%s (%s)' % (metricType, units) )
        plt.xlim   ( lims[0] , lims[1] )
        try:
            if name[-1] == '_':
                plotname = name[:-1]
            else:
                plotname = name
        except IndexError:
            plotname = name
        plt.title  ( '%s | (Mean, SD, P90) = (%.1f, %.1f, %.1f) %s\n(Q2, IQR, Mode) = (%.1f, %.1f, %.1f) %s' % 
                    ( plotname,
                      metrics['%smean' % name],
                      metrics['%sstd'  % name],
                      metrics['%sP90'  % name],
                      units ,
                      metrics['%sq2'   % name],
                      metrics['%siqr'  % name],
                      metrics['%smode' % name],
                      units ))
        
        plt.savefig  ( '%s/%shistogram.png' % ( histDir, name ))
        plt.close    ( )

    return metrics
        
def hist_mode( n , bins ):
    """
    Simple routine to get the mode of the distribution of a histogram.
    In the rare case that more than one modes are found, we take the average. 
    If its a bimodal distribution, we wouldn't believe this value anyway.
    """
    mode = bins[ np.where( n == n.max() ) ]
    
    # Non-ideal solution
    if len( mode ) > 1:
        return mode.mean()
    else:
        return mode[0]

def percentiles( data , extras=[]):
    """
    This used to be "ecc_tools.pp"
    This gains much faster speeds over running percentile functions in parallel, being able to do the sorting all at once rather than 6 times in a row for 6 different numbers...
    """
    data.sort()
    percs = [ 10., 25., 50., 75., 90. ]
    for i in extras:
        percs.append( float( i ) )
    names  = ('q1','q2','q3','iqr','p10','p90','m80')
    output = dict.fromkeys(names,0)
    N      = float( len( data ) )
    #print ('N = %.0f' % N )
    for i in range( len(percs) ):
        #print percs[i]
        r = percs[i] / 100. * N - 0.5
        #print( 'r = %.2f' % r )
        if r%1 == 0:
            p = data[int(r)]
        else:
            try:
                p = (data[int(np.ceil(r))] - data[int(np.floor(r))] ) * (r - np.floor(r)) + data[int(np.floor(r))]
                #print p
            except:
                p = np.nan
        #print p
        if not np.isnan(p):
            if percs[i] == 25:
                output['q1']  = p
            elif percs[i] == 50:
                output['q2']  = p
            elif percs[i] == 75:
                output['q3']  = p
            else:
                output['p%i' % percs[i]] = p
    output['iqr'] = output['q3']  - output['q1']
    output['m80'] = output['p90'] - output['p10']
    return output

def calc_iqr( data ):
    """ Single function that will return the IQR of a given dataset. """
    return percentiles(data)['iqr']

def calc_blocksize( data, nominal=(100,100) ):
    '''Returns block dimensions that are as close to log10(nominal) as possible'''
    X, Y            = data.shape
    x_nom, y_nom    = nominal

    x_low           = None
    y_low           = None

    x_high          = None
    y_high          = None

    x_fin           = None
    y_fin           = None
    
    def set_low_and_high( nominal, array_dim ):
        low     = None
        high    = None

        for v in range( nominal, 0, -1 ):
            if array_dim%v == 0: 
                low = v
                break
        for v in range( nominal+1, array_dim, 1 ):
            if array_dim%v == 0:
                high = v
                break
        return (low,high,)

    def set_final( low, high, nominal ):
        final = None

        def compare_to_nominal( low, high, nominal ):
            dif_log_low     = np.absolute( np.log10( nominal ) - np.log10( low ) )
            dif_log_high    = np.absolute( np.log10( nominal ) - np.log10( high ) )
            if dif_log_low < dif_log_high:
                return low
            else:
                return high

        if ( low is not None) and ( high is not None):
            final = compare_to_nominal( low, high, nominal )
        elif low is not None:
            final = low
        elif high is not None:
            final = high
        return final

    x_low, x_high = set_low_and_high( x_nom, X )
    y_low, y_high = set_low_and_high( y_nom, Y )

    x_fin = set_final( x_low, x_high, x_nom )
    y_fin = set_final( y_low, y_high, y_nom )

    return (x_fin, y_fin,)


def uniformity( data, blocksize, exclude=None, only_values=False, iqr=True, std=False ):
    ''' 
    Calculates the uniformity of the input data.
    block types are specified in chips.csv 
    Usual values are ( 'mini', 'micro', chip' )
    Returns a dictionary
        'q2'            : Median of full chip array
        'iqr'           : IQR of full chip array
        'blocks_q2'     : 2D array of local medians
        'blocks_iqr'    : 2D array of local IQRs
        'blocks_var'    : IQR of blocks_q2
        'iqr_ratio'     : blocks_var / iqr
        'blocks_iqrvar' : IQR of blocks_iqr

        'mean'           : mean of full chip array
        'std'            : std of full chip array
        'blocks_mean'    : 2D array of local means 
        'blocks_std'     : 2D array of local stds
        'blocks_mvar'    : std of blocks_mean
        'std_ratio'      : blocks_mvar / std 
        'blocks_stdmvar' : std of blocks_mean

    if stdonly is selcted, then q2 and iqr properties are not calculated.  This may be faster
    '''
    def block_reshape( data, blocksize ):
        rows, cols = data.shape
        numR = rows/blocksize[0]
        numC = cols/blocksize[1]
        return data.reshape(rows , numC , -1 ).transpose((1,0,2)).reshape(numC,numR,-1).transpose((1,0,2))

    output = {}
    if exclude is not None:
        data                   = data.astype( np.float )
        data[ exclude.astype( np.bool ) ]        = np.nan

    data[ np.isinf(data) ] = np.nan

    # Divide into miniblocks
    blocks  = block_reshape( data, blocksize )
    newsize = ( data.shape[0] / blocksize[0] ,
                data.shape[1] / blocksize[1] )

    if iqr:
        percs    = [ 25, 50, 75 ]
        # Calculate the full data properties
        flat = data[ ~np.isnan( data ) ].flatten()
        flat.sort()
        numwells = len( flat )
        levels   = {}
        for perc in percs:
            r = perc / 100. * numwells - 0.5
            if r%1 == 0:
                p = flat[int(r)]
            else:
                try:
                    p = (flat[np.ceil(r)] - flat[np.floor(r)] ) * (r - np.floor(r)) + flat[np.floor(r)]
                except:
                    p = np.nan
            levels[perc] = p
        output[ 'q2' ]  = levels[50]
        output[ 'iqr' ] = levels[75] - levels[25]

        # Calculate the local properties
        blocks.sort( axis=2 )
        numwells = (~np.isnan( blocks )).sum( axis=2 )

        regions  = {}

        def get_inds( blocks, inds2d, inds ):
            selected_blocks = blocks[inds2d]
            selected_wells  = inds.astype( np.int )[inds2d]
            rows            = range( selected_wells.size )
            output          = selected_blocks[ rows, selected_wells ]
            return output

        for perc in percs:
            r = perc / 100. * numwells - 0.5
            r[ r < 0 ] = 0
            r[ r > numwells ] > numwells[ r > numwells ]

            inds = r%1 == 0
            p    = np.empty( r.shape )
            p[:] = np.nan
            p[  inds ] = get_inds( blocks,  inds, r )
            ceil_vals  = get_inds( blocks, ~inds, np.ceil(r) )
            floor_vals = get_inds( blocks, ~inds, np.floor(r) )
            deltas = r[~inds] - np.floor(r[~inds] )
            p[ ~inds ] = ( ceil_vals - floor_vals ) * deltas + floor_vals
            p = p.reshape( newsize )
            regions[perc] = p

        blocks_iqr = regions[75] - regions[25]
        if not only_values:
            output[ 'blocks_q2' ]  = regions[50]
            output[ 'blocks_iqr' ] = blocks_iqr

        # Calculate the propertis of the regional averages
        flat = regions[50][ ~np.isnan( regions[50] ) ].flatten()
        flat.sort()
        numwells = len( flat )
        levels   = {}
        for perc in percs:
            r = perc / 100. * numwells - 0.5
            if r%1 == 0:
                p = flat[int(r)]
            else:
                try:
                    p = ( flat[np.ceil(r)] - flat[np.floor(r)] ) * (r - np.floor(r)) + flat[np.floor(r)]
                except:
                    p = np.nan
            levels[perc] = p

        output[ 'blocks_var' ] = levels[75] - levels[25]
        output['iqr_ratio'] = output[ 'blocks_var' ] / output[ 'iqr' ]

        # Calculate the variability of the regional variability
        flat = blocks_iqr[ ~np.isnan( regions[50] ) ].flatten()
        flat.sort()
        numwells = len( flat )
        levels   = {}
        for perc in percs:
            r = perc / 100. * numwells - 0.5
            if r%1 == 0:
                p = flat[int(r)]
            else:
                try:
                    p = ( flat[np.ceil(r)] - flat[np.floor(r)] ) * (r - np.floor(r)) + flat[np.floor(r)]
                except:
                    p = np.nan
            levels[perc] = p

        output[ 'blocks_iqrvar' ] = levels[75] - levels[25]

    if std:
        output[ 'mean' ]       = np.nanmean( data )
        output[ 'std'  ]       = np.nanstd( data )
        block_mean  = np.nanmean( blocks, axis=2 ).reshape( newsize )
        block_std   = np.nanstd( blocks, axis=2 ).reshape( newsize )
        if not only_values:
            output[ 'blocks_mean' ] = block_mean
            output[ 'blocks_std' ]  = block_std
        output[ 'blocks_mvar' ] = np.nanstd( block_mean )
        output[ 'std_ratio' ] = output[ 'blocks_mvar' ] / output[ 'std' ]
        output[ 'blocks_stdmvar' ] = np.nanstd( block_std )

    return output

def chip_uniformity( data, chiptype, block='mini', exclude=None, only_values=False, iqr=True, std=False ):
    ''' 
    A wrapper on uniformity, designed to take a chiptype as an input instead of a block size
    '''
    blocksize = ( getattr( chiptype, '%sR' % block ), 
                  getattr( chiptype, '%sC' % block ) )
    output = uniformity( data, blocksize, exclude=exclude, only_values=only_values, iqr=iqr, std=std )

    return output

###########################################
# rename functions for historical reasons #
###########################################
HistMode = hist_mode
pp = percentiles
