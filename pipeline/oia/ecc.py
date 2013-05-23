#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
#
# On-instrument analysis of extended chip check (ECC) data
# Phil Waggoner
# 
# This script loads a particular block from the ECC W1Step acquisition and writes several flat data files out for analysis elsewhere.
# Input arguments
# 
# [1] blockpath  - Path to block folder containing data file
# [2] resultsDir - Path to write output .dat files.
#
# Outputs
# -- actpix.dat   (1.7 MB)
# -- array.log    ( tiny )
# -- ebfvals.dat  (3.4 MB)
# -- gaincorr.dat (3.4 MB)
# -- pinned.dat   (1.7 MB)
# -- slopes.dat   (3.4 MB)
# -- t0.dat       (0.1 KB)

import argparse
import numpy as np
import sys, os, time
import scipy.stats as stats

# Add path for deinterlace.so on instrument
sys.path.append('/software/testing')

import deinterlace as di


#--------------------------------------------------#
# Function definitions
#--------------------------------------------------#

def BeadFind( BF ):
    '''
    Generic beadfind algorithm adapted from Todd Rearick's Thumbnail beadfind matlab script
    
    BF input is of the deinterlace class that is output from ReadDat
    '''
    # Start timing algorithm
    start_time = time.time()
    
    # Grab image data
    img = BF.data
    
    # Find active pixels
    simg   = np.std( img , axis = 2 , ddof = 1)
    actpix = simg > 500
    pins   = np.array( BF.pinned , dtype = bool )
    actpix = actpix & ~pins
    
    bfmat  = np.zeros(BF.pinned.shape)
    bfgain = np.zeros(BF.pinned.shape)
    
    if BF.miniR == 0 and BF.miniC == 0:
        print('Error! dat file is of unexpected size')
        beadfind = {}
    else:
        rows = BF.rows / BF.miniR
        cols = BF.cols / BF.miniC
        
        for r in range(rows):
            for c in range(cols):
                rws = BF.miniR * np.array(( r , r + 1 ))
                cls = BF.miniC * np.array(( c , c + 1 ))
                goodpx = actpix[ rws[0]:rws[1] , cls[0]:cls[1] ]
                
                # print('Now calculating region ( %s:%s , %s:%s ) . . .' % ( rws[0] , rws[1] , cls[0] , cls[1] ) )
                # For cleanliness, define the ROI for this tn block here
                roi = img[ rws[0]:rws[1] , cls[0]:cls[1] , :]
                
                bfg = BeadfindGainNorm( roi , ~goodpx )
                
                # Account for the far reaches of the chip that have wacky beadfind signals:
                # Modification by Phil 2-22-2013 -- using 500 as the number is only useful for 100 x 100 blocks or larger.  This breaks on 3-series.  Now use 5% of total px.
                if ( bfg['t1'] - bfg['t0'] ) <= 0 or goodpx.sum() < np.ceil( 0.05 * BF.miniR * BF.miniC ):
                    goodpx = np.zeros( goodpx.shape , dtype=bool )
                else:
                    # Trim ROI to minimize frames needed for calc assuming bf signal comes between (t0-5) and t1
                    roi = roi[ : , : , ( bfg['t0']-5 ) : bfg['t1'] ]
                    
                if goodpx.any():
                    nnimg = roi - GenerateBackgroundImage( roi , ~goodpx , 10 , bfg['bfgain'] )
                    nnimg[ np.isnan(nnimg) ] = 0
                    # blank pinned or inactive pixels
                    for frm in range(nnimg.shape[2]):
                        frame = nnimg[ : , : , frm ]
                        frame[ ~goodpx ] = 0
                        nnimg[ : , : , frm ] = frame
                        
                    # blank anything with an excessive standard deviation
                    s3 = np.std ( nnimg , axis = 2 , ddof = 1 )
                    oddpixelthresh = stats.scoreatpercentile(s3.flatten(),99)
                    if ( oddpixelthresh > 2 * np.mean( s3 ) ):
                        oddpixels = s3 >= oddpixelthresh;
                        goodpx = goodpx & ~oddpixels
                        for frm in range(nnimg.shape[2]):
                            fn = nnimg[ : , : , frm ] 
                            fn[ oddpixels ] = 0
                            nnimg[ : , : , frm ] = fn
                    # Note to self -- eventually need to add a new if goodpx.sum() < 500 here
                    try:
                        vspread = np.std ( nnimg.reshape( -1 , nnimg.shape[2] ) , axis = 0 , ddof = 1 )
                    except ( ValueError ):
                        print ('Error in this block due to bad pixels . . . now skipping' )
                    else:
                        maxdif  = np.argmax ( vspread )
                        bfval   = nnimg[ : , : , maxdif ]
                        bfval[ np.isnan(bfval) ] = 0
                        
                        # Output gain correction and beadfind value matrices for further analysis
                        gains = bfg['bfgain']
                        bfgain [ rws[0]:rws[1] , cls[0]:cls[1] ] = goodpx * gains
                        bfmat  [ rws[0]:rws[1] , cls[0]:cls[1] ] = goodpx * bfval
                        
        exectime = (time.time() - start_time)
        # print ( 'Execution time = %0.1f s' % exectime )
        beadfind = { 'actpix' : actpix , 'bfmat' : bfmat , 'gains' : bfgain , 'time' : exectime }
        
    return beadfind

def BeadfindGainNorm(image, pinned):
    '''Algorithm to normalize beadfind data to average trace, returning beadfind "gain" '''
    # Calculate average trace
    avgtrace = GetMaskAverage( image , pinned )
    # Define search parameters to trim out beadfind-relative (steep) part of the curves
    first = 0.05 * np.min(avgtrace)
    last  = 0.95 * np.min(avgtrace)
    
    f = image.shape[2]
    
    t0 = 0
    t1 = 0
    for i in range(f):
	# Check for steep edge of signal
        if ( avgtrace[i] < first and t0 == 0 ):
            t0 = i
        
	# Check for nearly leveled out signal
	if ( avgtrace[i] < last  and t1 == 0 ):
            t1 = i
        
	if (t0 != 0 and t1 != 0 ):
            trimmed = np.dstack(( image[ : , : , 0:t0 ] , image[ : , : , t1:(f - 1)] ))
            avgtrim = np.concatenate(( avgtrace[  0:t0 ] , avgtrace[ t1:(f - 1)] )).reshape(-1,1)
	    fits    = np.linalg.lstsq( avgtrim.reshape(-1,1) , np.transpose(trimmed.reshape(-1,trimmed.shape[2])))
	    bfgain  = fits[0].reshape( pinned.shape ) * ~pinned
	else:
            bfgain  = np.ones( pinned.shape ) * ~pinned
    gain = { 'bfgain' : bfgain , 't0' : t0 , 't1' : t1 }
    
    return gain

def BeadfindSlope( image , pinned , startframe=15 ):
    '''
    Algorithm to fit a linear slope to a given trace, using simple linear regression.
    
    startframe input added to support 3-series data compression that starts much earlier that frame 15.
    '''
    # Calculate average trace
    avgtrace = GetMaskAverage( image , pinned )
    
    # Define search parameter to pull out linear part of the beadfind curves
    first = 0.05 * np.min(avgtrace)
    
    f = image.shape[2]
    
    t0 = 0
    t1 = 0
    # Start looking at frame 15 (1s in) THIS ONLY WORKS FOR PROTON!!
    i = startframe
    while i < 46 and t0 == 0:
	# Check for steep edge of signal
        if avgtrace[i] < first:
            t0 = i
            # Check for super steep beadfinds.  Some go from zero to -6000 in a frame...
            if avgtrace[t0] < (5 * first):
                t0 = i - 1
            t1 = t0 + 2
        i = i + 1
        if i == f:
            break
        
    if (t0 != 0 and t1 != 0 ):
        trimmed = image[ : , : , t0:(t1 + 1) ]
        x       = np.hstack([ np.arange( t0 , t1 + 1 ).reshape(-1,1) , np.array(( 1 , 1 , 1 )).reshape(-1,1) ])
        fits    = np.linalg.lstsq( x , np.transpose(trimmed.reshape(-1,trimmed.shape[2])))[0]
        slopes  = fits[0].reshape( pinned.shape ) * ~pinned
    else:
        slopes  = np.zeros( pinned.shape )
    
    output = { 'slopes' : slopes , 't0' : t0 , 't1' : t1 }
    
    return output

def BufferTest( BT ):
    """
    This function is analogous to the BeadFind function but will calcualte buffering data.
    The only input needed is the acquisition metafile.
    """
    # Start timing algorithm
    start_time = time.time()
    
    # Grab image data
    img = BT.data
    
    # Determine chip type and assign startframe accordingly for BeadfindSlope function.
    if BT.type == '900':
        startframe = 15
    else:
        startframe = 1
    
    # Find active pixels
    simg   = np.std( img , axis = 2 , ddof = 1)
    actpix = simg > 500
    pins   = np.array( BT.pinned , dtype = bool )
    actpix = actpix & ~pins
    
    slopes = np.zeros(BT.pinned.shape)
    
    if BT.miniR == 0 and BT.miniC == 0:
        print('Error! dat file is of unexpected size')
        buffertest = {}
    else:
        rows = BT.rows / BT.miniR
        cols = BT.cols / BT.miniC
        
        # Initialize value matrices that only have one value per block
        tzeros = np.zeros (( rows, cols ))
        
        for r in range(rows):
            for c in range(cols):
                rws = BT.miniR * np.array(( r , r + 1 ))
                cls = BT.miniC * np.array(( c , c + 1 ))
                
                # These are assumed to be good pixels (i.e. NOT pinned)
                goodpx = actpix[ rws[0]:rws[1] , cls[0]:cls[1] ] 
                
                # print('Now calculating region ( %s:%s , %s:%s ) . . .' % ( rws[0] , rws[1] , cls[0] , cls[1] ) )
                # For cleanliness, define the ROI for this tn block here
                roi = img[ rws[0]:rws[1] , cls[0]:cls[1] , :]
                
                if goodpx.any():
                    # Recall, BeadfindSlope asks for pinned, not goodpx...
                    bufferdata = BeadfindSlope ( roi , ~goodpx , startframe )
                    slopes [ rws[0]:rws[1] , cls[0]:cls[1] ] = bufferdata['slopes']
                    tzeros [ r,c ] = bufferdata['t0']
                    
        exectime = (time.time() - start_time)
        # print ( 'Execution time = %0.1f s' % exectime )
        buffertest = { 'slopes' : slopes , 't0' : tzeros , 'actpix' : actpix , 'time' : exectime }
        
    return buffertest

def GenerateBackgroundImage(image,mask,dist,nn_gain):
    """
    Adapted from Todd Rearick's GenerateBackgroundImage Matlab function for nn-subtraction
    Note that mask is pinned pixels and will be ignored for the calculations.
    """
    [r,c,f] = image.shape
    # Redefine mask as a boolean array
    mask   = np.array( mask, dtype = bool )
    active = ~mask
    
    background = np.zeros((r,c,f))
    
    # take zeros out of gain correction
    nn_gain[ nn_gain == 0 ] = 1
    
    # Go through all the frames.  First frame (-1) is actually the mask.
    for nframe in np.arange(f+1)-1:
        
        fbg = np.zeros((r,c))
	
	# first time through we do the exact same calculation on the active pixel matrix in order to build the normalization matrix
	if nframe == -1:
            frame = active
	else:
	    frame = image[:,:,nframe] * active
	
	# Integrate down rows
	imat = np.zeros((r+2*dist+1,c+2*dist+1))
	
	for rw in (np.arange(r) + dist): 
            imat[rw+1,dist+1:dist+c+1] = frame[rw-dist,:] + imat[rw,dist+1:dist+c+1]
	    
	for rw in (np.arange(dist) + r + dist):
       	    imat[rw+1,:] = imat[rw,:]
	    
	# Integrate across columns
	for cl in (np.arange(c + dist - 1) + dist + 1):
	    imat[:,cl+1] = imat[:,cl+1] + imat [:,cl]
	    
	# Each output point can be calculated from the four corners of the box around the point, "trust me" - Todd
        fbg = (   imat[  (2*dist+1):,  (2*dist+1):] 
                - imat[:-(2*dist+1) ,  (2*dist+1):] 
                - imat[  (2*dist+1):,:-(2*dist+1)] 
                + imat[:-(2*dist+1) ,:-(2*dist+1)] )
        
	#for rw in np.arange(r):
	#    for cl in np.arange(c):
	#	fbg[rw,cl] = imat[rw+2*dist+1,cl+2*dist+1] - imat[rw,cl+2*dist+1] - imat[rw+2*dist+1,cl] + imat[rw,cl]
		
	# First time through, capture the normalization matrix.  After that, capture background matrix and scale it
	if nframe == -1:
            norm = fbg / nn_gain
            # modify norm to get rid of zeros
            norm[ norm == 0 ] = 1
	else:
            background[ : , : , nframe ] = fbg / norm
 	    
    return background

def GetMaskAverage ( image, pinned ):
    '''
    Calculates average trace while excluding pinned pixels. 
    If pinned = True, that pixel is excluded from the average.
    Note: This is opposite compared to matlab functionality.
    '''
    avgtrace = np.mean ( image[ ~pinned ] , axis=0 )
    return avgtrace

def ReadDat ( datFile , norm=True ):
    '''
    Generic function that will load an arbitrarily sized acquisition file.
    Returns array size details, pixel data, and pinned well information.
    
    ** If norm=True, then it will also sets all traces to begin at zero the same way TorrentExplorer 
    ** and TorrentR do.  This would be important if looking for pixel offset information.
    
    Properties:
    .rows
    .cols
    .frames
    .timestamps (in milliseconds)
    .data
    '''
    if os.path.exists( datFile ):
        # Read in .dat file
        acq = di.deinterlace_c ( datFile )
        
        # Reshape data array from (frame,R,C) to (R,C,frame)
        img = np.rollaxis ( acq.data , 0 , 3 )
        img = np.array    ( img , dtype='i2' )
        
        acq.miniR = 0
        acq.miniC = 0
        # Determine chip type
        if img.shape[0] == 800 and img.shape[1] == 1200:
            # Proton thumbnail ( 8 x 12 blocks )
            acq.type  = '900'
            acq.istn  = True
            acq.miniR = 100
            acq.miniC = 100
            
        if img.shape[0] == 1332 and img.shape[1] == 1288:
            # Proton full chip block ( 12 x 14 blocks )
            acq.type  = '900'
            acq.istn  = False 
            acq.miniR = 111
            acq.miniC = 92
           
        if img.shape[0] == 1152 and img.shape[1] == 1280:
            # 314 ( 10 x 9 blocks )
            acq.type  = '314'
            acq.istn  = False
            acq.miniR = 128
            acq.miniC = 128
            
        if img.shape[0] == 2640 and img.shape[1] == 2736:
            # 316 ( 22 x 24 blocks )
            acq.type  = '316'
            acq.istn  = False
            acq.miniR = 120
            acq.miniC = 114
            
        if img.shape[0] == 3792 and img.shape[1] == 3392:
            # 318 ( 48 x 53 blocks )
            acq.type  = '318'
            acq.istn  = False
            acq.miniR = 79
            acq.miniC = 64
            
        # Add new property to the class defining pixels that are pinned.
        acq.pinned = np.any( img > 16370 , axis=2 ) | np.any( img < 10 , axis=2 )
        
        if norm==True:
            # Normalize all pixels to start their signals at 0 before the flow begins.
            bkg = np.mean ( img[:,:,1:5] , axis=2 )
            bkg = np.array( bkg , dtype='i2' )
            # img = img - np.tile ( bkg.reshape( acq.rows , acq.cols , 1 ) , ( 1 , 1 , acq.frames ))
            for i in range( acq.frames ):
                img[ : , : , i ] = np.array( (img[ : , : , i ] - bkg) , dtype='i2' )
            acq.data = img
            acq.norm = True
        else:
            acq.data = img
            acq.norm = False
        
    else:
        acq = np.zeros(0)
        print('Error, file %s not found.' % datFile)
    return acq

#--------------------------------------------------#
# Main Function
#--------------------------------------------------#

def main():
    """Main Function"""
    # Read dat file and save pinned pixels
    if os.path.exists( datFile ):
        # Make output directory
        if not os.path.exists( resultsDir ):
            os.system( 'mkdir %s' % resultsDir )
            
        img = ReadDat ( datFile )
        np.array( img.pinned , dtype=bool ).tofile('%s/pinned.dat' % resultsDir )
        
        # Save details of image size and miniblock sizes
        f = open( '%s/array.log' % resultsDir , 'w' )
        f.write ( '%s\t%s\t%s\t%s' % ( img.rows , img.cols , img.miniR , img.miniC ) )
        f.close ( )
        
        # Empty chip beadfind analysis
        bf  = BeadFind ( img )
        np.array( bf['actpix'] , dtype=bool ).tofile('%s/actpix.dat' % resultsDir )
        
        # Note that these are only useful in the miniblock sizes 
        # 12 x 14 blocks (miniR = 111, miniC = 92)
        # For thumbnail, it's different, of course.  8 x 12 blocks of (miniR = 100 , miniC = 100)
        np.array( np.rint( bf['bfmat'] ) , dtype=np.dtype('i2') ).tofile('%s/ebfvals.dat' % resultsDir )
        np.array( np.rint( 10000 * bf['gains'] ) , dtype=np.dtype('i2') ).tofile('%s/gaincorr.dat' % resultsDir )
        
        # Buffering analysis
        bt  = BufferTest( img )
        
        np.array( np.rint( bt['slopes'] ) , dtype=np.dtype('i2') ).tofile('%s/slopes.dat' % resultsDir )
        np.array( bt['t0'] , dtype=np.dtype('i1') ).tofile('%s/t0.dat' % resultsDir )
        
    else:
        print('Error!  Acquisition file not found.  Please do not skip the calibration before loading.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-i', '--input-dir', dest='inputdir', required=True, help='Path to block folder containing data file  e.g. /results/F3--W1Step/X6440_Y0', metavar="DATFILE")
    parser.add_argument('-o', '--output-dir', dest='outputdir', required=True, help='Where to write out flat dat files.  e.g. <expt_dir>/block_X6440_Y0/sigproc_results', metavar="OUTPUTDIR")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.inputdir):
        print("ERROR: input directory %s doesn't exist" % args.inputdir)
        sys.exit(1)

    if not os.path.exists(args.outputdir):
        print("ERROR: output directory %s doesn't exist" % args.outputdir)
        sys.exit(1)


    # Path to dat file for analysis.  e.g. /results/<experiment_dir>/block_X6440_Y0
    datFile    = os.path.join(args.inputdir,'BT3_CalBeadfind_0000.dat')

    # Where to write out flat dat files.  e.g. <expt_dir>/block_X6440_Y0/sigproc_results
    # Will create an output folder to contain the group of above output files.
    resultsDir = os.path.join(args.outputdir,'ecc')

    sys.exit( main() )

#------------------------------------------------------------#
#                         END OF FILE                        #
#------------------------------------------------------------#
