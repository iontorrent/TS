"""
Module for an assortment of image analysis/processing tools.
Of most interest would be GBI - GenerateBackgroundImage, used for 'background subtraction' from the array.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import ndimage
from . import stats

def array_insert(arr,new,start):
    """
    Inserts the new array into arr at the starting coordinates.  This operates in place
    """
    try:
        arr[start[0]:start[0]+new.shape[0],
            start[1]:start[1]+new.shape[1]] = new
    except:
        print( 'Error inserting into array.  Likely a shape-mismatch.' )
        print( '   arr.shape:   %s, %s' % ( arr.shape ) )
        print( '   new.shape:   %s, %s' % ( new.shape ) )
        print( '   start:       %s, %s' % ( start ) )
        print( '   end:         %s, %s' % ( start[0]+new.shape[0], start[1]+new.shape[1] ) )
        raise

def downsample(data,scale=None,blocksize=None,subsample=True,clipedges=False):
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
            print( 'Scale must be >= 1.  No downsampling has been applied' )
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
            avgdata[i,j] = np.ma.masked_invalid(data[imin:imax,jmin:jmax]).mean()
    return avgdata

def GBI( image , mask=None , dist=10 , nn_gain=1 , ignore_self=False, block_rc=None ):
    """
    Adapted from Todd Rearick's GenerateBackgroundImage Matlab function for nn-subtraction
    Note that mask is pinned pixels and will be ignored for the calculations.
    
    ignore_self takes the center pixel out of the equation.  Useful for some interesting calculations.

    specifying block_rc = [ rows, cols ] will cause this to process a larger image, dividing it
       up into the specified blocks before running the GBI algorithm
    """
    if block_rc is None:
        block_rc = image.shape[:2]

    if nn_gain is None:
        nn_gain = 1

    # Excpand the rc 
    block_R, block_C = block_rc

    output = np.zeros_like( image )
    rows, cols = image.shape[:2]

    if rows % block_R or cols % block_C:
        raise ValueError( 'rc must evenly divide into the image size' )

    # Calculate the number of blocks
    region_rows = int( rows / block_R )
    region_cols = int( cols / block_C )

    for r in range( region_rows ):
        for c in range( region_cols ):
            rws = slice( r * block_R, ( r+1 ) * block_R )
            cls = slice( c * block_C, ( c+1 ) * block_C )

            roi    = image[ rws, cls ]
            if mask is None:
                badpx   = np.zeros( roi.shape[:-1], dtype=np.bool )   # All pixels are good
            else:
                badpx   = mask[rws,cls]

            try:
                gain = nn_gain[rws,cls].reshape(block_R, block_C,1)
            except TypeError:
                # must be a scalar
                gain = nn_gain


            output[rws,cls] = _GBI_block( roi, badpx, dist, nn_gain=gain )
    return output

def _GBI_block( image , mask , dist , nn_gain=1 , ignore_self=False ):
    """
    Adapted from Todd Rearick's GenerateBackgroundImage Matlab function for nn-subtraction
    Note that mask is pinned pixels and will be ignored for the calculations.
    
    ignore_self takes the center pixel out of the equation.  Useful for some interesting calculations.

    11/26/2018 - STP: This was formerly GBI.  GBI has since been replaced with a wrapper on this function
                      which can automatically divide up the chip.  Usage of GBI should be unaffected
                      You normally should call GBI instead of calling this directly
    """
    # t = time.time()
    [r,c,f] = image.shape
    # Redefine mask as a boolean array
    pins   = np.array( mask , dtype = bool)
    pins.resize(r,c,1)
    active = ~(np.tile(pins, (1,1,f)))
    
    frame = image * active
    foot = np.ones((2*dist+1,2*dist+1))
    if ignore_self:
        foot[ dist , dist ] = 0
    norm = ndimage.filters.convolve( np.array( ~mask , dtype=np.int16 ) , foot, mode='constant' , cval=0.0 ) / nn_gain
    norm.resize(r,c,1)
    
    background = np.zeros((r,c,f))
    
    # Integrate down rows
    imat = np.zeros((r+2*dist+1,c+2*dist+1,f))
    
    for rw in (np.arange(r) + dist): 
        imat[rw+1,dist+1:dist+c+1,:] = frame[rw-dist,:,:] + imat[rw,dist+1:dist+c+1,:]
        
    for rw in (np.arange(dist) + r + dist):
        imat[rw+1,:,:] = imat[rw,:,:]
        
    # Integrate across columns
    for cl in (np.arange(c + dist - 1) + dist + 1):
        imat[:,cl+1,:] = imat[:,cl+1,:] + imat [:,cl,:]
        
    # Each output point can be calculated from the four corners of the box around the point, "trust me" - Todd
    fbg = (   imat[  (2*dist+1):,  (2*dist+1): , :] 
            - imat[:-(2*dist+1) ,  (2*dist+1): , :] 
            - imat[  (2*dist+1):,:-(2*dist+1)  , :] 
            + imat[:-(2*dist+1) ,:-(2*dist+1)  , :] )
    
    if ignore_self:
        fbg -= image
    background = fbg / np.tile(norm , (1,1,f))
    if ignore_self:
        background[ np.isnan( background ) ] = 0
    # print('took %.3f' % (time.time() - t))
    return background

def imresize( image, newsize ):
    ''' resizes the 2D image using bilinear interpolation '''
    imsize = image.shape
    if imsize == newsize:
        return image
    if not ( len(imsize) == len(newsize) == 2 ):
        raise ValueError( 'Image and new size must be 2D' )

    x = range( imsize[0] )
    newx = np.linspace( 0, imsize[0]-1, newsize[0] )
    newimage = np.zeros( (newsize[0], imsize[1]) )
    for index in range( imsize[1] ):
        f = interp1d( x, image[:,index], kind='linear' )    # interp2d is too memory intensive
        newimage[:,index] = f(newx)
    y = range( imsize[1] )
    newy = np.linspace( 0, imsize[1]-1, newsize[1] )
    image = np.zeros( newsize )
    for index in range( newsize[0] ):
        f = interp1d( y, newimage[index,:], kind='linear' )
        image[index,:] = f(newy)

    return image

def spatial_iqr( data , blocksize ):  # Formerly spots
    ''' Calculates the local iqr, rastering acrros the array by the specified block size '''
    out = np.zeros( ( data.shape[0] / blocksize[0] , data.shape[1] / blocksize[1] ) , float )
    for i in range( out.shape[0] ):
        for j in range( out.shape[1] ):
            r0  = blocksize[0]*i 
            r1  = blocksize[0]*(i+1)
            c0  = blocksize[1]*j
            c1  = blocksize[1]*(j+1)
            out[i,j] = stats.percentiles( data[ r0:r1 , c0:c1 ][ data[ r0:r1 , c0:c1 ] > 0 ] )['iqr']
    return out

#########################
# Compatibility         #
#########################
spots = spatial_iqr
