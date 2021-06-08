"""
Module for chip-based data manipultions, including:
- Edge analysis
- PGM edge analysis (transforming into inlet-outlet and transverse axes)
- other chip data file tools such as compile_dat and outline masking 
"""

import os, time, re
import numpy as np
import numpy.ma as ma
import subprocess
from matplotlib.pyplot import hist
from scipy.ndimage import binary_closing, binary_opening

from . import average, stats
from . import chiptype as ct
from . import datprops as dp

class Edge:
    """ 
    This class is used to interact with and extract edge data from fc data 
    I have stopped caring about buffering and average it along the length of the chip in the middle 2 stripes of the chip.
    """
    def __init__( self , data , rng=[0,500] , chip=None ):
        s          = 200
        center     = np.array( data.shape ) / 2
        self.data  = data
        self.rows  = center[0] * 2
        self.cols  = center[1] * 2
        roi        = data[ (center[0]-s/2):(center[0]+s/2) , (center[1]-s/2):(center[1]+s/2) ]
        self.datarng = rng

        if ct is None:
            self.chip = ct.get_ct_from_rc( self.rows, self.cols )
        else:
            self.chip = chip
        self.chip.isblock = self.rows == self.chip.blockR
        
        # There may be a better way to deal with this rather than using matplotlib histogram to find mode . . .
        hn,hb,_    = hist( roi[ np.logical_and( roi > rng[0] , roi < rng[1] ) ] , bins = np.linspace( rng[0] , rng[1] , 101 ) )
        #clf ( )
        self.rmode = stats.hist_mode( hn , hb )
        
        # Switch to dealing with PGM chips in transverse and inlet-outlet mode:
        if self.chip.series == 'pgm':
            self.pgm_xform( )
        elif self.chip.tn == 'self':
            # Proton tn 
            start = 3*self.chip.blockR
            end   = self.chip.chipR - start
            self.rowavg = average.stripe_avg( data[ start:end, : ] , 0 )

            start = 3*self.chip.blockC
            end   = self.chip.chipC - start
            self.colavg = average.stripe_avg( data[ : , start:end] , 1 )

            self.boil( )
        elif self.chip.isblock:
            # P1 block ( Don't use with corner blocks. . . )

            start = int( self.chip.blockR * 0.4 )   # Used to be center-100
            end   = int( self.chip.blockR * 0.6 )   # used to be center+100
            self.rowavg = average.stripe_avg( data[ start:end , : ], 0 )

            start = int( self.chip.blockC * 0.4 )
            end   = int( self.chip.blockC * 0.6 )
            self.colavg = average.stripe_avg( data[ : , start:end ], 1 )

            self.boil( )
        else:
            # full chips 
            start = data.shape[0]*2/8
            end   = data.shape[0]*6/8
            #start = 2 * self.chip.blockR
            #end   = self.chip.chipR - start
            top    = data.shape[1]*1/24
            bottom = data.shape[1]*23/24
            self.rowavg = average.stripe_avg( data[ start:end , top:bottom ] , 0 )

            #start = 3 * self.chip.blockC
            #end   = self.chip.chipC - start
            start = data.shape[1]*3/12
            end   = data.shape[1]*9/12
            self.colavg = average.stripe_avg( data[ : , start:end ] , 1 )

            self.boil( )
            
    def pgm_xform( self , perc=0.2 ):
        """ 
        This function rotates the 3-series chips so that the flow direction points in the -x direction.
        The transverse-to-flow direction (y') will be called self.colavg to correspond with proton chips where the col avg is transverse to flow.
        Along the flow direction will be called self.rowavg.
        perc defines the approximate percentage of pixels away from midline for each direction of edge analysis.
        valmin is a minimum value acceptance for the values.  Will need to be changed based on data type coming in.  This being set to 10 works for noise and buffering.
        """
        xf = PGM_Transform( self.data.shape )
        
        # Define stepsize to simplify averaging for 316, 318 chips?
        if self.rows == 1152:
            stepsize = 16.
            fc = np.fromfile( '/results/python/eccserver/flowcell_314.dat' , dtype=bool ).reshape( self.rows , -1 )
        elif self.rows == 2640:
            stepsize = 32.
            fc = np.fromfile( '/results/python/eccserver/flowcell_316.dat' , dtype=bool ).reshape( self.rows , -1 )
        elif self.rows == 3792:
            stepsize = 64.
            fc = np.fromfile( '/results/python/eccserver/flowcell_318.dat' , dtype=bool ).reshape( self.rows , -1 )
            
        refpix  = np.zeros(( self.rows , self.cols ) , bool )
        refpix[4:-4,4:-4] = True
            
        # Col averaging ( x' )
        mid     = xf.xpmax / 2.
        dist    = perc * xf.xpmax / 2.
        colmask = np.logical_and( xf.xp >= mid-dist , xf.xp < mid+dist )
        valmask = np.logical_and ( self.data > self.datarng[0] , self.data < self.datarng[1] )
        
        # Here we are figuring out where the flowcell stops so that we don't include annoying zeros at refpix
        yp_mask = np.logical_and ( colmask , np.logical_and( refpix , ~fc ) )
        yp_dist = np.array( np.abs( xf.yp[yp_mask].max() ) , np.abs( xf.yp[yp_mask].min() ) ).min()
        rng     = np.arange( -yp_dist , yp_dist , stepsize )
        N       = len( rng )
        points  = np.zeros((N))
        values  = np.zeros((N))
        stdevs  = np.zeros((N))
        
        # Loop through range of distances from midline
        for i in range( N ):
            points[ i ] = rng[ i ] + stepsize / 2.
            
            # Deal with masking.  Assume that values == 0 are going to be ignored.
            posmask     = np.logical_and ( colmask , np.logical_and( xf.yp >= rng[i] , xf.yp < rng[i] + stepsize ))
            masked      = ma.masked_array( self.data , ~np.logical_and( posmask , valmask ) )
            values[ i ] = masked.mean()
            stdevs[ i ] = masked.std()
            
        self.col_pts         = points
        self.boiled_col_avg  = values
        self.boiled_col_std  = stdevs
        
        # Row averaging ( y' )
        mid     = 0
        dist    = perc * xf.ypmax
        rowmask = np.logical_and( xf.yp >= mid-dist , xf.yp < mid+dist )
        valmask = np.logical_and ( self.data > self.datarng[0] , self.data < self.datarng[1] )
        rng     = np.arange( 0 , xf.xpmax , stepsize )
        N       = len( rng )
        points  = np.zeros((N))
        values  = np.zeros((N))
        stdevs  = np.zeros((N))
        
        # Loop through range of distances from outlet
        for i in range( N ):
            points[ i ] = rng[ i ] + stepsize / 2.
            
            # Deal with masking.  Assume that values == 0 are going to be ignored.
            posmask     = np.logical_and( rowmask , np.logical_and( xf.xp >= rng[i] , xf.xp < rng[i] + stepsize ))
            masked      = ma.masked_array( self.data , ~np.logical_and( posmask , valmask ) )
            values[ i ] = masked.mean()
            stdevs[ i ] = masked.std()
            
        # Get rid of ridiculously low values and change points to be zero-centered.
        points              -= xf.xpmax / 2.
        self.row_pts         = points
        self.boiled_row_avg  = values
        self.boiled_row_std  = stdevs
            
    def boil( self ):
        """ This function boils down the row and column averages by averaging every 100 px together so that the results are plottable and interesting """
        # Remove extreme edge pixels.  We're looking for global trends here.
        for i in range(2):
            if i == 0:
                avg_data = self.rowavg
            else:
                avg_data = self.colavg
                
            trim = avg_data.size/500
            if trim == 0:
                trimmed = avg_data
            else:
                trimmed    = avg_data[trim:-trim]
            l          = max( avg_data.size/100, 1 )
            N          = len(trimmed)
            mod        = N % l
            
            if mod == 0:
                block      = trimmed.reshape(-1,l)
            else:
                block      = trimmed[ mod/2:-mod/2 ].reshape(-1,l)
                
            msk        = ( block == 0 )
            boiled     = ma.masked_array( block , msk ).mean( 1 ).data
            stdevs     = ma.masked_array( block , msk ).std( 1 ).data
            
            if mod != 0:    
                # Get number of non-zero values
                b0         = l - msk[ 0,:].sum()
                b1         = l - msk[-1,:].sum()
                
                # Deal with mod values that are feeling left out.  Count how many non-zero values there are
                left       = ( trimmed[ :mod/2] > 0 ).sum()
                right      = ( trimmed[-mod/2:] > 0 ).sum()
                
                boiled[0]  = ( boiled[ 0] * b0 + trimmed[ :mod/2].sum() ) / (b0 + left )
                boiled[-1] = ( boiled[-1] * b1 + trimmed[-mod/2:].sum() ) / (b1 + right)
                
            # Calculate zero-centered coordinates, taking into account the mod pixels
            pts        = np.arange( l/2 , (N - mod) , l ) - (N - mod)/2
            pts[ 0]   -= mod/2
            pts[-1]   += mod/2
            
            if i == 0:
                self.row_pts        = pts
                self.boiled_row_avg = boiled
                self.boiled_row_std = stdevs
            else:
                self.col_pts        = pts
                self.boiled_col_avg = boiled
                self.boiled_col_std = stdevs
                
        return None
    
    def get_offsets( self ):
        """ Returns offsets of middle rows and cols of data in the odd case that we'll normalize this data before reloading it. """
        if not hasattr( self , 'row_pts' ):
            self.boil()
            
        # Select middle value, depending on even or odd vector length
        mid = len( self.row_pts ) / 2
        if len( self.row_pts ) % 2 == 0:
            self.row_offset = np.mean( self.boiled_row_avg[ (mid-1):(mid+1) ] )
        else:
            self.row_offset = self.boiled_row_avg[ np.floor( mid/2 ) ]
            
        mid = len( self.col_pts ) / 2
        if len( self.col_pts ) % 2 == 0:
            self.col_offset = np.mean( self.boiled_col_avg[ (mid-1):(mid+1) ] )
        else:
            self.col_offset = self.boiled_col_avg[ np.floor( mid/2 ) ]
            
        return ( self.row_offset , self.col_offset )
    
    def save( self , output_dir , metric ):
        """ Saves coordinate files and data files named after metric. """
        np.array( self.row_pts , dtype = np.int16 ).tofile( os.path.join( output_dir , 'row_avg_coords_e.dat' ) )
        np.array( self.col_pts , dtype = np.int16 ).tofile( os.path.join( output_dir , 'col_avg_coords_e.dat' ) )
        
        # Save as 10x so that we can get a decimal point in.
        np.array( np.rint( 10 * self.boiled_row_avg ) , dtype = np.int16 ).tofile( os.path.join( output_dir , 'row_avg_%s.dat' % metric ) )
        np.array( np.rint( 10 * self.boiled_col_avg ) , dtype = np.int16 ).tofile( os.path.join( output_dir , 'col_avg_%s.dat' % metric ) )
        np.array( np.rint( 10 * self.boiled_row_std ) , dtype = np.int16 ).tofile( os.path.join( output_dir , 'row_std_%s.dat' % metric ) )
        np.array( np.rint( 10 * self.boiled_col_std ) , dtype = np.int16 ).tofile( os.path.join( output_dir , 'col_std_%s.dat' % metric ) )
        
        # Is there a way to kill the class from within? something to research when I find internets again.
        return None

class PGM_Transform:
    """
    Class to support axes transforms into flow direction.
    This will be useful for flow front analysis and potentially other areas.
    """
    def __init__( self , datashape ):
        ( self.rows , self.cols ) = datashape
        
        y = np.arange( self.rows , dtype=float )
        y = np.tile( y[:,np.newaxis] , (1 , self.cols) )
        x = np.arange( self.cols , dtype=float )
        x = np.tile( x[np.newaxis,:] , (self.rows , 1) )
        
        # compute denominator
        self.d = np.sqrt( np.power( self.rows , 2 ) + np.power( self.cols , 2 ) )
        
        # define some handy points of reference
        self.ypmin = - self.rows * self.cols / self.d
        self.ypmax =   self.rows * self.cols / self.d
        self.xpmin = 0
        self.xpmax = self.d
        self.xp , self.yp = self.transform( x , y )
        
    def transform( self , x , y ):
        """ Transforms lists of x and y coordinates into the new flow-space coordinates """
        # compute sign of y' . . . is it above or below midline?
        sign = np.sign( (y - (float(self.rows)/float(self.cols)) * x ) )
        
        xprime =       (self.cols * x + self.rows * y) / self.d
        yprime = np.abs(self.rows * x - self.cols * y) / self.d * sign
        
        return xprime , yprime
    
    def reverse( self , x_primes , y_primes ):
        """
        Takes coordinates in the (x',y') space and reverts to origin (x,y) or rows,cols
        """
        # Get x coords (b needs to be negated)
        b , a = self.transform( 0 , self.rows )
        b    *= -1
        denom = np.sqrt( pow( a , 2 ) + pow( b , 2 ) )
        x = np.abs( a * x_primes + b * y_primes ) / denom
        
        # Get y coords (b needs to be negated)
        b , a = self.transform( self.cols , 0 )
        b    *= -1
        denom = np.sqrt( pow( a , 2 ) + pow( b , 2 ) )
        y = np.abs( a * x_primes + b * y_primes ) / denom
        
        return np.array( np.rint(x) , int ) , np.array( np.rint(y) , int )

def dat_compile( path , filename , chip_type , dt , delete=False ):
    """ 
    Compiles flat dat files into a single large datfile.
    returns the array as a float for later use....

    path:      Parent directory of block directories
    filename:  file to compile into a full chip
    chiptype:  A ChipType instance 
    dt:     :  data type (storage format) of the dat files
    delete  :  if true, it will delete the little pieces if no errors are encountered.
    """
    paths = []
    # Get array details
    if chip_type.series == 'proton':
        rows = chip_type.chipR
        cols = chip_type.chipC
        microR = chip_type.microR
        microC = chip_type.microC
        miniR = chip_type.miniR
        miniC = chip_type.miniC
        blockR = chip_type.blockR
        blockC = chip_type.blockC
        firstblock = True
        for i in range(chip_type.yBlocks):
            for j in range(chip_type.xBlocks):
                blockpath = os.path.join( path , 'X%s_Y%s/%s' % ( blockC*j , blockR*i , filename ) )
                paths.append( blockpath )
                #print blockpath
                #if (i == 0) and (j==0):
                if firstblock:  # Cannot key off of i=0, j=0 in case X0_Y0 is missing
                    start_time = time.time()
                    try:
                        data = np.fromfile( blockpath , dtype=dt )
                        if data.shape[0] == rows * cols:
                            #print( "Blocksize determined to be (%s , %s)" % ( rows , cols ) )
                            data = data.reshape( rows, cols )
                            r,c  = rows,cols
                        elif data.shape[0] == blockR*blockC:
                            #print( "Blocksize determined to be (%s , %s)" % ( blockR , blockC ) )
                            data = data.reshape( blockR, blockC )
                            r,c  = blockR,blockC
                        elif data.shape[0] == miniR*miniC:
                            #print( "Blocksize determined to be (%s , %s)" % ( miniR , miniC ) )
                            data = data.reshape( miniR , miniC )
                            r,c  = miniR,miniC
                        elif data.shape[0] == blockR*blockC/(miniR*miniC):
                            #print( "Blocksize determined to be (%s , %s)" % ( miniR , miniC ) )
                            data = data.reshape( blockR/miniR , blockC/miniC )
                            r,c  = blockR/miniR , blockC/miniC
                        elif data.shape[0] == blockR*blockC/(microR*microC):
                            #print( "Blocksize determined to be (%s , %s)" % ( microR , microC ) )
                            data = data.reshape( blockR/microR , blockC/microC )
                            r,c  = blockR/microR , blockC/microC
                        else:
                            raise TypeError('ERROR: Unexpected number of values (%s) loaded from %s' % ( data.shape[0] , blockpath ) )
                        datsize = data.shape
                        #print datsize
                        img = np.zeros( (chip_type.yBlocks*datsize[0] , chip_type.xBlocks*datsize[1] ) , dt )
                        img[ r*i:r*(i+1) , c*j:c*(j+1) ] = data
                        firstblock = False
                    except IOError: 
                        pass
                else:
                    #print blockpath
                    try:
                        data = np.fromfile( blockpath , dtype=dt ).reshape( datsize[0] , datsize[1] )
                    except IOError:
                        data = np.zeros( (datsize[0] , datsize[1]) , dt )
                    img[ r*i:r*(i+1) , c*j:c*(j+1) ] = data
                
    elif chip_type.series == 'pgm':
        if 'ecc' in os.path.basename( path ):
            eccpath = path
        else:
            eccpath = os.path.join( path , 'ecc' )
        rows = chip_type.chipR
        cols = chip_type.chipC
        miniR = chip_type.miniR
        miniC = chip_type.miniC
        blockR = chip_type.blockR
        blockC = chip_type.blockC
        filepath = os.path.join( eccpath , filename )
        if os.path.exists( filepath ):
            img = np.fromfile( filepath , dtype = dt )
            if img.shape[0] == rows * cols:
                img = img.reshape( rows, cols )
            elif img.shape[0] == blockR*blockC:
                img = img.reshape( blockR, blockC )
            else:
                raise TypeError('ERROR: Unexpected number of values (%s) loaded from %s' % ( img.shape[0] , 
                                                                                             filepath ) )
        else:
            img = np.zeros((rows,cols), dt )
    else:
        try:
            raise TypeError('ERROR: Unexpected chip type (%s) given. . .' % chip_type.name )
        except:
            raise TypeError('ERROR: Unexpected chip type (%s) given. . .  Must be a ChipType instance' % chip_type )

    
    if delete and ( chip_type.type in ct.proton_chips ):
        devnull = open( os.devnull, 'w' )
        for x in paths:
            if os.path.exists( x ): # Needed to speed up if files aren't actually pressent
                subprocess.Popen( 'rm %s' % x, stdout=devnull, stderr=devnull, shell=True )
            
    try:
        return np.array( img , dtype = float )
    except: 
        raise IOError( 'missing files in datfile %s' % filename )

def dat_compile_dp( path , metric, chiptype , filename=None, delete=False, transpose=False ):
    """ 
    Updated dat_compile based on datprops instead of guessing sizes
    Compiles flat dat files into a single large datfile.
    returns the array as a float for later use....

    path:      Parent directory of block directories
    metric:    Metric to compile
    chiptype:  A ChipType instance 
    filename:  (Default: '%s.dat' % metric).  Filename to compile. 
    delete  :  if true, it will delete the little pieces if no errors are encountered.
    transpose: Transpose the data before assembling into blocks
    """
    paths = []
    # Get array details
    dt = dp.get_dtype( metric )
    if chiptype.series == 'proton':
        firstblock = True
        for i in range(chiptype.yBlocks):
            for j in range(chiptype.xBlocks):
                blockpath = os.path.join( path , 'X%s_Y%s/%s' % ( chiptype.blockC*j , chiptype.blockR*i , filename ) )
                paths.append( blockpath )
                #print blockpath
                #if (i == 0) and (j==0):
                if firstblock:  # Cannot key off of i=0, j=0 in case X0_Y0 is missing
                    start_time = time.time()
                    try:
                        data = dp.read_dat( blockpath, metric, chiptype=chiptype )
                        if transpose:
                            data = data.T
                        datsize = data.shape
                        if datsize == (1,):
                            datsize = ( 1, 1 )
                        img = np.zeros( (chiptype.yBlocks*datsize[0] , chiptype.xBlocks*datsize[1] ) , dt )
                        img[ datsize[0]*i:datsize[0]*(i+1) , datsize[1]*j:datsize[1]*(j+1) ] = data
                        firstblock = False
                        print 'Block Size: %s' % ( datsize, )
                    except IOError: 
                        pass
                else:
                    #print blockpath
                    try:
                        data = dp.read_dat( blockpath, metric, chiptype=chiptype )
                        if transpose:
                            data = data.T
                        #data = np.fromfile( blockpath , dtype=dt ).reshape( datsize[0] , datsize[1] )
                    except IOError:
                        data = np.zeros( (datsize[0] , datsize[1]) , dt )
                    img[ datsize[0]*i:datsize[0]*(i+1) , datsize[1]*j:datsize[1]*(j+1) ] = data
                
    elif chiptype.series == 'pgm':
        if 'ecc' in os.path.basename( path ):
            eccpath = path
        else:
            eccpath = os.path.join( path , 'ecc' )
        filepath = os.path.join( eccpath , filename )
        if os.path.exists( filepath ):
            img = dp.read_dat( filepath, metric, chiptype=chiptype )
        else:
            img = np.zeros( (chiptype.chipR, chiptype.chipC), dt )
    else:
        try:
            raise TypeError('ERROR: Unexpected chip type (%s) given. . .' % chiptype.name )
        except:
            raise TypeError('ERROR: Unexpected chip type (%s) given. . .  Must be a ChipType instance' % chiptype )

    
    if delete and ( chiptype.type in ct.proton_chips ):
        devnull = open( os.devnull, 'w' )
        for x in paths:
            if os.path.exists( x ): # Needed to speed up if files aren't actually pressent
                subprocess.Popen( 'rm %s' % x, stdout=devnull, stderr=devnull, shell=True )
            
    try:
        return np.array( img , dtype = float )
    except: 
        raise IOError( 'missing files in datfile %s' % filename )

def load_mask_old( filename, chiptype ):
    ''' Reads and generates the well mask from the outline file '''

    data = np.fromfile( filename, dtype=np.uint16 ).reshape( -1, 2 )
    mask_size_0 = data[0,0]
    mask_size_1 = data[0,1]
    edges = data[1:,:]

    #as p1mask.dat is correlated to the p1 type chip (10656,15456)
    wellmask = np.zeros( (mask_size_0, mask_size_1), dtype=np.bool )
    for i, lims in enumerate( edges ):
        wellmask[i,lims[0]:lims[1]] = True

    if   chiptype.tn == 'self':
         # regular thumbnail
         if chiptype.burger:
            burger = (chiptype.burger.chipR-chiptype.fullchip.chipR)/2
            burger = int(mask_size_0*burger/chiptype.burger.chipR)
            wellmask = wellmask[burger:-1*burger, :]

         R_b = wellmask.shape[0]/chiptype.yBlocks
         C_b = wellmask.shape[1]/chiptype.xBlocks
         R_s = chiptype.blockR
         C_s = chiptype.blockC
         r_h = int(R_s/2)
         c_h = int(C_s/2)
         mask = np.array([wellmask[i*R_b:(i+1)*R_b,j*C_b:(j+1)*C_b] for (i,j) in np.ndindex(chiptype.yBlocks, chiptype.xBlocks)])
         mask_tn = np.array([mask[i, int(R_b/2)-r_h:int(R_b/2)+r_h, int(C_b/2)-c_h:int(C_b/2)+c_h] for i in range(mask.shape[0])])      
         wellmask = mask_tn.reshape(chiptype.yBlocks, chiptype.xBlocks, R_s, C_s).swapaxes(1,2).reshape(chiptype.chipR, chiptype.chipC)  

    else:    
         #full chip or spa tn (P0, P1, P2 can share the same mask file)
         if chiptype.burger and chiptype.spatn == 'self':      #spatn burger chip
            burger = (chiptype.burger.chipR-chiptype.fullchip.chipR)/2
            burger = int(mask_size_0*burger/chiptype.burger.chipR)
            wellmask = wellmask[burger:-1*burger, :]
            '''
            print 'burger', burger
            print 'fullchip R', chiptype.fullchip.chipR
            print 'burger R', chiptype.burger.chipR
            print 'wellmask', wellmask.shape
            '''
         if chiptype.burger and chiptype.spatn != 'self':  #full size burger chip
            chipR = chiptype.burger.chipR
            chipC = chiptype.burger.chipC
         else:
              chipR = chiptype.chipR
              chipC = chiptype.chipC

         b = np.linspace(0.0, wellmask.shape[0], chipR, endpoint=False)
         a = np.linspace(0.0, wellmask.shape[1], chipC, endpoint=False)
         coord_C = np.array([int(round(A)) for A in a])
         coord_R = np.array([int(round(B)) for B in b])

         mask_new = np.zeros((chipR, chipC))
         for index_R, i in enumerate(coord_R):
             for index_C, j in enumerate(coord_C):
                 mask_new[index_R, index_C] = wellmask[i, j]
         #print 'mask_new shape', mask_new.shape

         if chiptype.burger and chiptype.spatn != 'self': 
            burger = (chiptype.burger.chipR-chiptype.chipR)/2
            mask_new = mask_new[burger:-1*burger, :]
            #print 'after removing', mask_new.shape, burger

         wellmask = mask_new                    

    return wellmask

def load_mask ( filename, chiptype ):
    ''' Reads and generates the well mask from the outline file (.omask)
    Returns a boolean array where active array = True

    outline files are structured ( ( #rows, #cols ), 
                                   ( left_0, right_0 ),  --
                                   ( left_1, right_1 ),   |
                                   ( left_2, right_2 ),   |- 1 entry per row
                                   ( left_3, right_3 ),   |
                                   ...                   --
                                   )
    left and right denote conventional python indexing meaning:
        row[left_i:right_i] = True
    Therefore left_0 is the first point inside the active array
              right_0 is the first point outisde the active array

    outline files are scaled to other chip sizes based on the dimensions of the first row and the dimensions of the chip
    This means that a coarse mask can be defined using a relativly small masksize
    '''
    chiptype = ct.ChipType( chiptype )

    if re.search( '\.T(\..*)*$', filename ):
        transpose = True
    elif chiptype.transpose:
        transpose = True
    else:
        transpose = False
    if transpose:
        chiptype = chiptype.transposed()

    if '.omask' not in filename:
        filename += '.omask'

    ## Check if a . is in the filename, but is not an extension (hidden folder?)
    #if '.' in filename:
    #    if '/' in filename.split('.')[-1]:
    #        filename += '.omask' 
    ## Check if a . is not in the filename
    #if '.' not in filename:
    #    filename += '.omask' 

    # Load and reshape the mask
    data = np.fromfile( filename, dtype=np.uint16 ).reshape( -1, 2 )
    # Get the mask size from the first row
    mask_size_0 = data[0,0]
    mask_size_1 = data[0,1]
    # The remaining rows represent the left and right sides of each row
    edges = data[1:,:]

    if   chiptype.tn =='self':
         wellmask = np.zeros( (mask_size_0, mask_size_1), dtype=np.bool )
         for i, lims in enumerate( edges ):
             wellmask[i,lims[0]:lims[1]] = True

         # regular thumbnail
         if chiptype.burger:
            burger = (chiptype.burger.chipR-chiptype.fullchip.chipR)/2
            burger = int(mask_size_0*burger/chiptype.burger.chipR)
            wellmask = wellmask[burger:-1*burger, :]

         R_b = wellmask.shape[0]/chiptype.yBlocks
         C_b = wellmask.shape[1]/chiptype.xBlocks
         R_s = chiptype.blockR
         C_s = chiptype.blockC
         r_h = int(R_s/2)
         c_h = int(C_s/2)
         mask = np.array([wellmask[i*R_b:(i+1)*R_b,j*C_b:(j+1)*C_b] for (i,j) in np.ndindex(chiptype.yBlocks, chiptype.xBlocks)])
         mask_tn = np.array([mask[i, int(R_b/2)-r_h:int(R_b/2)+r_h, int(C_b/2)-c_h:int(C_b/2)+c_h] for i in range(mask.shape[0])])      
         wellmask = mask_tn.reshape(chiptype.yBlocks, chiptype.xBlocks, R_s, C_s).swapaxes(1,2).reshape(chiptype.chipR, chiptype.chipC)  

    else:
         if chiptype.burger and chiptype.spatn == 'self':  #spatn burger chip
            burger = (chiptype.burger.chipR-chiptype.fullchip.chipR)/2
            burger = int(mask_size_0*burger/chiptype.burger.chipR)
            edges = edges[burger:-1*burger, :]

         if chiptype.burger and chiptype.spatn != 'self':  #full size burger chip
            chipR = chiptype.burger.chipR
            chipC = chiptype.burger.chipC
         else:
              chipR = chiptype.chipR
              chipC = chiptype.chipC

         a = np.linspace(0.0, edges.shape[0], chipR, endpoint=False)
         coord_R = np.array([int(round(A)) for A in a])
         edges_new = np.zeros((len(coord_R), 2))
         for i, r in enumerate (coord_R):
             edges_new[i,:] = edges[r,:]

         center = mask_size_1/2
         center_new = chipC/2
         ratio_C = float(mask_size_1)/chipC
         edges_new2 = np.around((edges_new-center)/ratio_C)+center_new 

         wellmask = np.zeros( (chipR, chipC), dtype=np.bool )
         for i, lims in enumerate( edges_new2 ):
             wellmask[i,int(lims[0]):int(lims[1])] = True

         if chiptype.burger and chiptype.spatn != 'self': 
            burger = (chiptype.burger.chipR-chiptype.chipR)/2
            wellmask = wellmask[burger:-1*burger, :]

    if transpose:
        wellmask = wellmask.T

    return wellmask    

def make_mask( img, filename ):
    ''' Creates an outline mask which can be read by load_mask
    img is a boolean array where img=True cooresponds to points 
    inside of the active array 

    Point-defects will be repaired, but this may not fix everything
    If defects remain, the longest continious band of True will be selected
    For example
      Inupt:  ||||||----|-------||||||
      Output: ----------|-------||||||
      Outline:           x      x
    Severe defects mean multiple sets of >=5 True were found in a given row.  It is recomended to fix them

    Function returns boolean indicating if the event above occured

    '''
    # Transpose the image if needed
    if re.search( '\.T(\..*)*$', filename ):
        img = img.T

    badrows = 0
    reallybad = 0
    img = img.copy()     # Copy the array because we will be modifying the mask in place
    ## Do a binary closing to remove very small holes
    img = binary_closing( img ) # Remove small holes inside of the array
    img = binary_opening( img ) # Remove small holes outside of the array
    ic = img.copy()
    #img = ~binary_closing( ~img )
    outline = [ img.shape ]
    for rn, row in enumerate( img ):
        outline.append( ( 0, 0 ) )
        br = False
        tr = row.sum()
        disp = []
        while row.any():
            first = np.argmax( row )  # This gets the first True
            row[:first] = True        # Make a temporary structure
            last  = np.argmin( row )  # This gets the next False
            row[:last]  = False       # Pin low and check for any more

            savedlength = outline[-1][1] - outline[-1][0]
            runlength = last-first
            if runlength != tr:
                if not br:
                    print 'WARNING: Multiple sections found in row %i' % rn
                    badrows += 1
                br = True
                if runlength > 5:
                    disp.append( '         %i/%i' % ( runlength, tr ) )
                    # Attempt to string together large segments that are close together
                    #if savedlength > 5: 
                    #    if first - outline[-1][1] > 5:
                    #        first = outline[-1][0]
                    #        runlength = last-first
                    #        disp[-1] += '  (appended)'
            
            # Yeah, I know this is a bit redundant
            #if runlength > savedlength and runlength/tr > 0.5:
            if runlength > savedlength:
                #outline.append( ( first, last ) )
                outline[-1] = ( first, last )
        if len(disp) > 1:
            reallybad += 1
            for d in disp:
                print d

        #outline.append( ( 0, 0 ) )
        #if row.any():
        #    first = np.argmax( row )  # This gets the first True
        #    row[:first] = True        # Make a temporary structure
        #    last  = np.argmin( row )  # This gets the next False
        #    row[:last]  = False       # Pin low and check for any more
        #    if row.any():
        #        print 'WARNING: Multiple sections found in row %i' % rn
        #        badrows += 1
        #    else:
        #        outline.append( ( first, last ) )
    np.array( outline ).astype( np.uint16 ).tofile( filename )

    if badrows: 
        print ''
        print '%i/%i rows were discontinious' % ( badrows, img.shape[0] )
        print '%i/%i rows were severe' % ( reallybad, badrows )

    return reallybad, badrows


############################################
# Compatibility                            #
############################################
DatCompile = dat_compile
edge = Edge
from .efuse import EfuseClass
