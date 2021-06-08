"""
Contains helpful functions to unravel or ravel chip data between physical columns and per-adc quasi-columns.
"""

import numpy as np

def block_reshape( data, blocksize ):
    ''' Reshapes the 2D data into 3D data, with the 3rd dimension being adjacent pixels '''
    rows, cols = data.shape
    numR = rows/blocksize[0]
    numC = cols/blocksize[1]
    return data.reshape(rows , numC , -1 ).transpose((1,0,2)).reshape(numC,numR,-1).transpose((1,0,2))

def unreshape( data, blocksize ):
    ''' Reverses the output of block_reshape '''
    numR, numC, els = data.shape
    rows = numR*blocksize[0]
    cols = numC*blocksize[1]
    return data.transpose((1,0,2)).reshape( numC, rows, -1 ).transpose((1,0,2)).reshape( rows, cols )


def im2adc_550( frame ):
    ''' Converts a 550 "datacollect" image (well layout) to an "ADC" image (pixel layout)'''
    if frame.ndim == 3:
        return np.array( [ im2adc_550(f) for f in frame.transpose((2,0,1)) ] ).transpose((1,2,0))
    blocks  = block_reshape( frame, (3,4) )
    blocks1 = blocks[:,:,(0,4,1,6,2,3,8,9,5,10,7,11)]
    return unreshape( blocks1, (2,6) )

def adc2im_550( frame ):
    ''' Converts a 550 "ADC" image (pixel layout) to a "datacollect" image (well laout) '''
    if frame.ndim == 3:
        return np.array( [ adc2im_550(f) for f in frame.transpose((2,0,1)) ] ).transpose((1,2,0))
    blocks  = block_reshape( frame, (2,6) )
    blocks1 = blocks[:,:,(0,2,4,5,1,8,3,10,6,7,9,11)]
    return unreshape( blocks1, (3,4) )

def im2adc_550_mb( frame ):
    ''' This undos Mark B's reshape function in /software/p2/dev.py:convertImage (real code is burried in assembly) '''
    if frame.ndim == 3:
        return np.array( [ im2adc_550(f) for f in frame.transpose((2,0,1)) ] ).transpose((1,2,0))
    blocks  = block_reshape( frame, (3,4) )
    #blocks1 = blocks[:,:,(0,4,1,6,2,3,8,9,5,10,7,11)]
    blocks1 = blocks[:,:,(4,0,1,2,6,3,8,9,5,10,7,11)]
    return unreshape( blocks1, (2,6) )

def adc2im_550_mb( frame ):
    ''' THis mimics Mark B's reshape function '''
    if frame.ndim == 3:
        return np.array( [ adc2im_550(f) for f in frame.transpose((2,0,1)) ] ).transpose((1,2,0))
    blocks  = block_reshape( frame, (2,6) )
    #blocks1 = blocks[:,:,(0,2,4,5,1,8,3,10,6,7,9,11)]
    #blocks1 = blocks[:,:,(1,2,4,5,0,8,3,10,6,7,9,11)]
    blocks1 = blocks[:,:,(1,2,3,5,0,8,4,10,6,7,9,11)]
    return unreshape( blocks1, (3,4) )

