'''
convolve provides additional functionality for applying convolutions.
'''
from numpy import convolve, ones
from scipy.ndimage import convolve as convN
import numpy as np

def conv(arr,v,mode='full',norm=False,inplace=False):
    '''
    conv just calls the numpy.convolve function and returns the result. It adds the functionality of normalizing to 
    remove zero-padding around the edges (norm=True).  This is usualy best done in conjunction with the mode='same' 
    option, but is not required

    Convolve can still be called by convolve.convolve() or numpy.convolve()
    '''
    if not inplace:
        arr = arr.copy()

    if norm:
        return convolve(arr,v,mode=mode)/convolve(ones(arr.shape),v,mode)
    else:
        return convolve(arr,v,mode=mode)

def conv1(arr,v,axis=0,norm=True,inplace=False):
    '''
    NOTE: THIS FUNCTION OPERATES ON arr IN PLACE

    conv1 takes the 1D convolution of an array (arr) with the vector (v) along the specified axis.
    for example:
        v = [1 1 1]
        arr = ([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14]])       (shape: (3,5))
        
        axis=0 -> ([[ 0.5,  1.0,  2.0,  3.0,  3.5],
                    [ 5.5,  6.0,  7.0,  8.0,  8.5],
                    [10.5, 11.0, 12.0, 13.0, 13.5]])
        axis=1 -> ([[ 2.5,  3.5,  4.5,  5.5,  6.5],
                      5.0,  6.0,  7.0,  8.0,  9.0],
                      7.5,  8.5,  9.5, 10.5, 11.5]])
    conv1 uses the 'same' option to convolve(), but unless norm=False, edge effects (zero-padding) are normalized out

    By: Scott Parker,  Jan. 20, 2014
    '''
    if not inplace:
        arr = arr.copy()

    # Make sure array is 2D
    if arr.ndim != 2:
        raise TypeError('arr must be 2D')

    # Transpose the array if necessary
    if axis == 0:
        pass
    elif axis == 1:
        arr = arr.T
    else:
        raise TypeError('Axis value is invalid')

    vnorm = convolve(ones(arr.shape[1]),v,mode='same')
    for row in arr:
        if norm:
            row[:] = convolve(row, v, mode='same')/vnorm
        else:
            row[:] = convolve(row, v, mode='same')

    # Re-transpose to original orientation
    if axis == 1:
        arr = arr.T

    return arr

def conv2(arr,v,norm=True,inplace=False):
    '''
    NOTE: THIS FUNCTION OPERATES ON arr IN PLACE

    conv2 takes the 2D convolution of an array (arr) with the vector (v) by convolving first across the rows and then the columns.
    conv2 uses the mode='same' option in convolve(), but unless norm=False, edge effects (zero-padding) are normalized out.

    By: Scott Parker,  Jan 20, 2014
    '''
    if not inplace:
        arr = arr.copy()
    
    return conv1( conv1(arr, v, axis=0, norm=norm, inplace=inplace) , v, axis=1, norm=norm, inplace=inplace)

def masked( arr, v, mask ):
    """ Performs the 1D convolution on arr, omitting pixels under the mask """
    arr = arr.astype(float)
    smoothed = conv( arr*mask, v )
    norm     = conv( mask.astype(np.float), v )
    return smoothed/norm

def masked2( arr, v, mask ):
    """ Performs the 2D convolution on arr, omitting pixels under the mask """
    arr = arr.astype(float)
    arr[mask] = 0
    smoothed = conv2( arr, v )
    norm     = conv2( np.logical_not(mask).astype(np.float), v )
    return smoothed/norm

def gaussian(x,xc,sigma,h=1,offset=0,norm=False):
    '''
    returns a numpy array with the same shape as x, plotting the gaussian function
        y = h*exp(-(x-xc)^2/(2 \sigma^2) + o
    where:
        x is a 1D vector
        xc is the center of the distribution
        sigma is the width of the distribution
        h is the height of the distribution peak
        offset is the 'o' parameter above

        If norm=True, the result is normalized such that \int^\infty_-infty(y) = 1
        This overwrites the amplitude value and sets offset = 1
    '''

    if norm:
        offset = 0
        h = 1.0/(sigma*np.sqrt(2*np.pi))

    return h*np.exp(-(np.array(x)-xc)**2 / (2 * sigma**2)) + offset
