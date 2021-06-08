import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from . import misc, stats
from . import chiptype as ct
from . import datprops as dp
''' 
Module for processing row and column correlated noise
'''

def compile( datadir, chiptype, force=False, delete=True, makeplot=True ):
    ''' 
    Compiles the block-level noise data.  This must be done differently because the dimensions 
    are ( rows x frames ) or ( cols x frames ) instead of ( rows x cols ) 
    '''
    if os.path.exists( os.path.join( datadir, 'rownoise.dat' ) ) and not force:
        print '...Skipping previously compiled rc noise'
        return None

    # Initialize arrays for block-averaged values
    row_tn   = np.zeros( ( chiptype.yBlocks, chiptype.xBlocks ) )
    col_tn   = np.zeros( ( chiptype.yBlocks, chiptype.xBlocks ) )

    # Get list of directories
    try:
        directories = chiptype.blocknames
    except:
        chiptype.blockdir = datadir
        directories = chiptype.read_blocks()
    # Loop over all blocks
    for d in directories:
        # Calculate indicies
        coords = ct.block_rc( d )
        tn_r   = coords[0]/chiptype.blockR
        tn_c   = coords[1]/chiptype.blockC

        # Check if this is an edge block
        edge   = False
        if tn_r == 0 or tn_r == chiptype.yBlocks-1:
            edge = True
        if tn_c == 0 or tn_c == chiptype.xBlocks-1:
            edge = True
        tophalf = tn_r >= chiptype.yBlocks/2


        # Read row noise
        fn = os.path.join( datadir, d, 'rownoise_raw.dat' )
        try:
            data = dp.read_dat( fn, 'rownoise_raw' )
            data = dp.reshape( data, chiptype, size=[chiptype.blockR,-1] )
        except IOError:
            print 'Unable to find %s' % fn
            continue

        # Calculate the average for the block
        row_tn[tn_r,tn_c] = data.std(axis=1).mean()
        # Add to arrays for full chip noise.  Edge blocks are excluded so I don't have to worry about weighted averages due to glue line
        if not edge:
            # Dynamically create so we can deal with an arbitrary number of frames. 
            try:
                rownoise[ coords[0]:coords[0]+chiptype.blockR, :data.shape[1] ] += data
                rowcount[ coords[0]:coords[0]+chiptype.blockR, :data.shape[1] ] += 1
            except NameError:
                rownoise = np.zeros( ( chiptype.chipR, data.shape[1] ) )
                rowcount = np.zeros( ( chiptype.chipR, data.shape[1] ) )
                rownoise[ coords[0]:coords[0]+chiptype.blockR, :data.shape[1] ] += data
                rowcount[ coords[0]:coords[0]+chiptype.blockR, :data.shape[1] ] += 1
            except ValueError:
                # We don't need all the data because smaller data sets blow up the standard deviation
                rownoise[ coords[0]:coords[0]+chiptype.blockR, : ] += data[:,:rownoise.shape[1]]
                rowcount[ coords[0]:coords[0]+chiptype.blockR, : ] += 1

        # Read col noise
        fn = os.path.join( datadir, d, 'colnoise_raw.dat' )
        try:
            data = dp.read_dat( fn, 'colnoise_raw' )
            data = dp.reshape( data, chiptype, size=[chiptype.blockC,-1] )
        except IOError:
            print 'Unable to find %s' % fn
            continue
        # Calculate the average for the block
        col_tn[tn_r,tn_c] = data.std(axis=1).mean()
        # Add to arrays for full chip noise.  Edge blocks are excluded so I don't have to worry about weighted averages due to glue line
        if not edge:
            # Dynamically create so we can deal with an arbitrary number of frames
            if tophalf:
                try:
                    colnoise_top[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += data
                    colcount_top[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += 1
                except NameError:
                    colnoise_top = np.zeros( ( chiptype.chipC, data.shape[1] ) )
                    colcount_top = np.zeros( ( chiptype.chipC, data.shape[1] ) )
                    colnoise_top[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += data
                    colcount_top[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += 1
                except ValueError:
                    # We don't need all the data because smaller data sets blow up the standard deviation
                    colnoise_top[ coords[1]:coords[1]+chiptype.blockC, : ] += data[:,:colnoise_top.shape[1]]
                    colcount_top[ coords[1]:coords[1]+chiptype.blockC, : ] += 1
            else:
                try:
                    colnoise_bot[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += data
                    colcount_bot[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += 1
                except NameError:
                    colnoise_bot = np.zeros( ( chiptype.chipC, data.shape[1] ) )
                    colcount_bot = np.zeros( ( chiptype.chipC, data.shape[1] ) )
                    colnoise_bot[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += data
                    colcount_bot[ coords[1]:coords[1]+chiptype.blockC, :data.shape[1] ] += 1
                except ValueError:
                    # We don't need all the data because smaller data sets blow up the standard deviation
                    colnoise_bot[ coords[1]:coords[1]+chiptype.blockC, : ] += data[:,:colnoise_bot.shape[1]]
                    colcount_bot[ coords[1]:coords[1]+chiptype.blockC, : ] += 1

        if delete:
            devnull = open( os.devnull, 'w' )
            dirname = os.path.join( datadir, d, 'rownoise_raw.dat' )
            subprocess.call( 'rm -r %s' % dirname, shell=True, stdout=devnull, stderr=devnull )
            dirname = os.path.join( dirname, d, 'colnoise_raw.dat' )
            subprocess.call( 'rm -r %s' % dirname, shell=True, stdout=devnull, stderr=devnull )

    # divide the full chip row and col noise arrays by the number of blocks which were actually incorperated
    try:
        rownoise     /= rowcount
        colnoise_top /= colcount_top
        colnoise_bot /= colcount_bot
    except:
        # If you got here, it's probably because no dat files were present
        return None
    # Clear out the infinities
    rownoise[ np.isinf(rownoise) ] = np.nan
    colnoise_top[ np.isinf(colnoise_top) ] = np.nan
    colnoise_bot[ np.isinf(colnoise_bot) ] = np.nan
    # Calculate a single value per row or column
    mask = np.isnan( rownoise ) | ( rowcount != rowcount.max() )
    noise_rows = ma.masked_array( rownoise, mask ).std( axis=1 )
    mask = np.isnan( colnoise_top ) | ( colcount_top != colcount_top.max() )
    noise_cols_top = ma.masked_array( colnoise_top, np.isnan( colnoise_top ) ).std( axis=1 )
    mask = np.isnan( colnoise_bot ) | ( colcount_bot != colcount_bot.max() )
    noise_cols_bot = ma.masked_array( colnoise_bot, np.isnan( colnoise_bot ) ).std( axis=1 )

    # Calculate stats
    metrics = {}
    metrics.update( stats.named_stats( misc.flatten( noise_rows    , metric='rcnoise' ), name='rownoise' ) )
    metrics.update( stats.named_stats( misc.flatten( noise_cols_top, metric='rcnoise' ), name='colnoise_top' ) )
    metrics.update( stats.named_stats( misc.flatten( noise_cols_bot, metric='rcnoise' ), name='colnoise_bot' ) )
    noise_cols = np.append( noise_cols_top, noise_cols_bot)
    metrics.update( stats.named_stats( misc.flatten( noise_cols    , metric='rcnoise' ), name='colnoise' ) )

    # Write dat files
    dp.write_dat( noise_rows,     os.path.join( datadir, 'rownoise.dat' ),     'rownoise' ) 
    dp.write_dat( noise_cols_top, os.path.join( datadir, 'top_colnoise.dat' ), 'colnoise' ) 
    dp.write_dat( noise_cols_bot, os.path.join( datadir, 'bot_colnoise.dat' ), 'colnoise' ) 
    dp.write_dat( row_tn,         os.path.join( datadir, 'tn_rownoise.dat' ),  'rownoise' ) 
    dp.write_dat( col_tn,         os.path.join( datadir, 'tn_colnoise.dat' ),  'colnoise' ) 

    if makeplot:
        plot( noise_rows, axis='row', center=True, savedir=datadir )
        plot( noise_cols_top, noise_cols_bot, axis='col', center=True, savedir=datadir )
        plot_tn( row_tn, axis='row', savedir=datadir )
        plot_tn( col_tn, axis='col', savedir=datadir )

    output = { 'metrics'      : metrics,
               'rownoise'     : noise_rows,    
               'top_colnoise' : noise_cols_top,
               'bot_colnoise' : noise_cols_bot,
               'tn_rownoise'  : row_tn,        
               'tn_colnoise'  : col_tn }

    return output 

def plot( data1, data2=[], axis='Row', center=False, ymax=15, savedir='.' ):
    ''' 
    data1 and (optional) data2 must be 1D arrays
    Plots the row or column noise at either the block or full chip level
    axis can be either 'Row' or 'Col'
    if center is True, data are centered around x=0
    '''
    if center:
        x1 = np.arange( -len( data1 )/2, len( data1 )/2 )
        x2 = np.arange( -len( data2 )/2, len( data2 )/2 )
    else:
        x1 = np.arange( len( data1 ) )
        x2 = np.arange( len( data2 ) )
    if len( data2 ):
        xmin = min( x1[0], x2[0] )
        xmax = min( x1[-1], x2[-1] )
    else:
        xmin = x1[0] 
        xmax = x1[-1]

    plt.figure()
    plt.plot( x1, data1, 'b+' )
    if len( data2 ):
        plt.plot( x2, data2, 'rx' )
    plt.ylim( 0, ymax )
    plt.xlim( xmin, xmax )
    plt.ylabel( '%s Noise (DN14 counts)' % axis.capitalize() )
    plt.xlabel( axis.capitalize() )
    plt.savefig( os.path.join( savedir, '%snoise.png' % axis.lower() ) )
    plt.close()

def plot_tn( data, axis='Row', ymax=15, savedir='.' ):
    ''' 
    Plots the thumbnail spatial plot for row or column noise
    axis can be either 'Row' or 'Col'
    '''
    plt.figure()
    plt.imshow( data, clim=[0, ymax], interpolation='nearest', origin='lower' )
    plt.colorbar()
    plt.title( '%s Noise (DN14 counts)' % axis.capitalize() )
    plt.savefig( os.path.join( savedir, '%snoise-tn.png' % axis.lower() ) )
    plt.close()

