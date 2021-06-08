''' 
THIS MODULE IS DEPRECATED!

Functions for averaging arrays 
'''
import numpy as np
import numpy.ma as ma
import warnings

# For compatibility with move to stats . . .
from . import stats

warnings.simplefilter('default')
warnings.warn( 'Average.py is deprecated and you should remove all references to it from your code.  Please see stats.py for similar functions.', DeprecationWarning )

########################################
# Compatibility                        #
########################################

stripe_avg   = stats.stripe_avg
StripeAvg    = stripe_avg

masked_avg   = stats.masked_avg
GetMaskedAvg = masked_avg

block_reshape   = stats.reshape.block_reshape
block_unreshape = stats.reshape.block_unreshape

block_avg = stats.block_avg
block_std = stats.block_std

def BlockAvg( data, blocksize, mask ):
    return block_avg( data, blocksize[0], blocksize[1], goodpix=mask, finite=True )
