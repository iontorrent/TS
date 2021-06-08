''' Functions for averaging arrays '''
import numpy as np
import numpy.ma as ma

def block_avg( data , blockR, blockC, goodpix=None, finite=False ):
    """
    Averages the specified data, averaging regions of blockR x blockC
    This returns a data of size 
        ( data.shape[0]/blockR, data.shape[1]/blockC )

    goodpix is a boolean array of same size as data where pixels to be averaged are True. By default all pixels are averaged
    raising finite sets all nans and infs to 0
    """
    rows = data.shape[0]
    cols = data.shape[1]
    
    if ( rows % blockR ) or ( cols % blockC ):
        raise ValueError( '[block_avg] Data size (%i, %i) must be an integer multiple of %i x %i' % ( rows, cols, blockR, blockC ) )
    else:
        numR = rows / blockR
        numC = cols / blockC

    if goodpix is None:
        goodpix = np.ones( data.shape ) 

    output = np.zeros((numR,numC))
    
    # Previous algorithm was slow and used annoying looping
    # Improved algorithm that doeesn't need any looping.  takes about 1.4 seconds instead of 60.
    masked = np.array( data , float ) * np.array( goodpix, float )
    step1  = masked.reshape(rows, numC , -1 ).sum(2)
    step2  = step1.transpose().reshape(numC , numR , -1).sum(2).transpose()
    mask1  = goodpix.reshape(rows, numC , -1 ).sum(2)
    count  = mask1.transpose().reshape(numC , numR , -1).sum(2).transpose()
    output = step2 / count
    
    if finite:
        output[ np.isnan(output) ] = 0
        output[ np.isinf(output) ] = 0

    return output

def masked_avg( image, pinned ):  # AVERAGE
    '''
    Calculates average trace while excluding pinned pixels. 
    If pinned = True, that pixel is excluded from the average.
    Note: This is opposite compared to matlab functionality.
    '''
    avgtrace = np.mean ( image[ ~pinned ] , axis=0 )
    return avgtrace

def stripe_avg( data , ax ):
    """
    Custom averaging algorithm to deal with pinned pixels
    """
    # EDIT 10-11-13
    # Utilize numpy.masked_array to do this much more quickly
    # Not so custom anymore!
    return ma.masked_array( data , (data == 0) ).mean( ax ).data

########################################
# Compatibility                        #
########################################

StripeAvg    = stripe_avg
GetMaskedAvg = masked_avg

def BlockAvg( data, blocksize, mask ):
    return block_avg( data, blocksize[0], blocksize[1], goodpix=mask, finite=True )
