import matplotlib.pyplot as plt
import json
import numpy as np
import os
import re
import subprocess
from warnings import warn

from . import stats
from . import convolve as conv

# Modules which may not be on all servers but are not necessary for all functions
modules = [ 'h5py', 'pysam' ]
for m in modules:
    try:
        globals()[m] = __import__( m )
    except ImportError as e:
        warn( getattr( e, 'message', 'message not found' ), ImportWarning )

#---------------------------------------------------------------------------#
# Define classes
#---------------------------------------------------------------------------#

class BamFile:
    """
    Class for reading and parsing bam files
    """
    def __init__( self, filename, origin=(0,0), shape=(1332*8,1288*12), verbose=False ):
        """
        Initializes the bam reader
        if filename is a list of filenames, all listed files will be read into a single array
        If shape is unspecifed, we attempt to read it from the block directories in the same folder as filename
        """
        self.filename = filename
        self.verbose = verbose
        self.origin = origin
        if shape is None:
            self.shape = self.get_rowcol()
        else:
            self.shape = shape


    def count_indels( self ):
        self.readlength   = np.zeros( self.shape )
        self.mappedlength = np.zeros( self.shape )
        self.inserts      = np.zeros( self.shape )
        self.deletions    = np.zeros( self.shape )

        # Open the file
        filenames = self.filename
        if isinstance( filenames, basestring ):
            filenames = [filenames]
        for fn in filenames:
            try: 
                self._logging( 'Now reading %s' % fn )
                bam = pysam.Samfile( fn )
            except IOError:
                self._logging( '  Error reading %s' % fn, 0 )

            count = 0
            # Read each well
            for read in bam:
                count += 1
                if not (count % 100000):
                    self._logging( '  Well %s' % count )

                if not read.is_unmapped:
                    name = read.qname

                    # parsing ABCDE:00001:00001
                    row = int(name.split(':')[1]) - self.origin[0]
                    col = int(name.split(':')[2]) - self.origin[1]

                    self.readlength[row,col]   = read.rlen
                    self.mappedlength[row,col] = read.qlen

                    n_insertions = 0
                    if 'I' in read.cigarstring:
                        n_insertions = len( [m.start() for m in re.finditer('I',read.cigarstring)] )

                    n_deletions = 0
                    if 'D' in read.cigarstring:
                        n_deletions = len( [m.start() for m in re.finditer('D',read.cigarstring)] )

                    self.inserts[row,col]   = n_insertions
                    self.deletions[row,col] = n_deletions

    def count_indels_from_table( self ):
        """ 
        Reads a table from prevous analysis of the format:  
        row, col, readlength, mappedlength, inserts, deletions
        """
        self.readlength   = np.zeros( self.shape )
        self.mappedlength = np.zeros( self.shape )
        self.inserts      = np.zeros( self.shape )
        self.deletions    = np.zeros( self.shape )

        filenames = self.filename
        if isinstance( filenames, basestring ):
            filenames = [filenames]
        for fn in filenames:
            count = 0
            for line in open( fn, 'r' ):
                count += 1
                if not (count % 100000):
                    self._logging( '  Well %s' % count )

                parts = line.split()
                row = parts[0]
                col = parts[1]
                self.readlength[row,col]   = parts[2]
                self.mappedlength[row,col] = parts[3]
                self.inserts[row,col]      = parts[4]
                self.deletions[row,col]    = parts[5]

    def get_rowcol( self ):
        """
        Reads the block names from the block names in the current directory

        returns row, col
        """
        # Read the block directories
        block = re.compile( r'block_X[0-9]+_Y[0-9]+')
        if isinstance( self.filename, (tuple, list) ):
            dirname = os.path.dirname( self.filename[0] )
        else:
            dirname = os.path.dirname( self.filename )
        files = os.listdir( dirname )
        files = [ d for d in files if block.match( d ) ]
        
        # Get the x and y coordinates for each block
        xfiles = [ f for f in files if 'Y0' in f ]
        xstarts = [ int( f.split('_')[1][1:] ) for f in xfiles ]
        xstarts.sort()
        yfiles = [ f for f in files if 'X0' in f ]
        ystarts = [ int( f.split('_')[2][1:] ) for f in yfiles ]
        ystarts.sort()
        block_cols = xstarts[1] - xstarts[0]
        block_rows = ystarts[1] - ystarts[0]
        nxblocks = len(xfiles)
        nyblocks = len(yfiles)
        rows = block_rows * nyblocks
        cols = block_cols * nxblocks
        
        return rows, cols

        # make array of filenames
        files = []
        for y in self.ystarts:
            row = []
            for x in self.xstarts:
                row.append( 'block_X%i_Y%i' % ( x, y ) )
            files.append( row )

        self.blocks = files

    def reverse_complement( self, sequence ):
        """ returns reverse complement of the sequence """
        COMPLEMENT = {'A': 'T','T': 'A','C': 'G','G': 'C', 'N': 'N'}
        return ''.join(COMPLEMENT[b] for b in sequence[::-1])

    def _logging( self, string, level=1 ):
        """
        Function for logging output by verbosity
        """
        if level <= self.verbose:
            print( string )
    
    def save( self, directory='.' ):
        self.save_indels( directory )
        self.save_lengths( directory )

    def save_indels( self, directory='.' ):
        try: 
            path = os.path.join( directory, 'inserts.dat' )
            np.int8(self.inserts).tofile( path ) 
        except AttributeError:
            pass
        try: 
            path = os.path.join( directory, 'deletions.dat' )
            np.int8(self.deletions).tofile( path ) 
        except AttributeError:
            pass

    def save_lengths( self, directory='.' ):
        try: 
            path = os.path.join( directory, 'readlength.dat' )
            np.int16(self.readlength).tofile( path ) 
        except AttributeError:
            pass
        try: 
            path = os.path.join( directory, 'mappedlength.dat' )
            np.int16(self.mappedlength).tofile( path ) 
        except AttributeError:
            pass

    def load( self, directory='.' ):
        self.load_indels( directory )
        self.load_lengths( directory )

    def load_indels( self, directory='.' ):
        try: 
            path = os.path.join( directory, 'inserts.dat' )
            self.inserts = np.fromfile( path, dtype=np.int8 ).reshape( self.shape )
        except IOError:
            pass
        try: 
            path = os.path.join( directory, 'deletions.dat' )
            self.deletions = np.fromfile( path, dtype=np.int8 ).reshape( self.shape )
        except IOError:
            pass

    def load_lengths( self, directory='.' ):
        try: 
            path = os.path.join( directory, 'readlength.dat' )
            self.readlength = np.fromfile( path, dtype=np.int16 ).reshape( self.shape )
        except IOError:
            pass
        try: 
            path = os.path.join( directory, 'mappedlength.dat' )
            self.mappedlength = np.fromfile( path, dtype=np.int16 ).reshape( self.shape )
        except IOError:
            pass

class Bfmask:
    """ Class to support bfmask.bin files."""
    def __init__( self , bfFile, best=False ):
        """ 
        Reads info from beadfind file and sets up properties 

        if best=True, then bfFile should be the directory and the best one will be chosen depending on what is left
        the 'best' order is: 1) sigproc_results/analysis.bfmask.bin
                                sigproc_results/block_X#_Y#/analysis.bfmask.bin
                                sigproc_results/bfmask.bin
                                sigproc_results/block_X#_Y#/bfmask.bin
        """
        if best:
            self._get_best( bfFile )
        else:
            self.file = bfFile
            self.read_file ( bfFile )

    def read_file( self, bfFile ):
        raw = np.fromfile( bfFile , dtype=np.dtype('u2') )
        
        # Read header to extract rows, cols.
        hdr = raw[0:4]
        self.rows = hdr[0]
        self.cols = hdr[2]
        self.masks= [ 'empty' , 'bead' , 'live' , 'dud' , 'reference' , 'tf' , 'lib' , 'pinned' , 'ignore' ,
                      'washout' , 'exclude' , 'keypass' , 'badkey' , 'short' , 'badppf' , 'badresidual' ]
        
        # miniblock is the block over which heatmaps are generated
        # block is the row, col size of the 96-blocked chip
        # interval is a common divisor that will allow column averaging, etc.
        if ( self.rows , self.cols ) == ( 8*1332 , 12*1288 ):
            # PI
            self.miniblock = [ 16 , 16 ]
            self.block     = [ 1332 , 1288 ]
            self.interval  = [ 24 , 24 ]
        elif ( self.rows , self.cols ) == ( 800 , 1200 ):
            # Thumbnail
            self.miniblock = [ 10 , 10 ]
            self.block     = [ 100 , 100 ]
            self.interval  = [ 10 , 10 ]
        elif ( self.rows, self.cols ) == ( 1332, 1288 ):
            # PI analysis block
            self.miniblock = [ 16 , 16 ]
            self.block     = [ 1332 , 1288 ]
            self.interval  = [ 24 , 24 ]
        elif ( self.rows, self.cols ) == ( 8*664, 12*640 ):
            # P0 
            self.miniblock = [ 8 , 10 ]
            self.block     = [ 664 , 640 ]
            self.interval  = [ 24 , 24 ]
        elif ( self.rows, self.cols ) == ( 664, 640 ):
            # P0 Aalysis block
            self.miniblock = [ 8 , 10 ]
            self.block     = [ 664 , 640 ]
            self.interval  = [ 24 , 24 ]
        
        # Get mask reshaped and initialize masks
        self.data = raw[4:].reshape(( self.rows , self.cols ))
        bits = self._interpret()
        self.empty       = bits[:,:,0]
        self.bead        = bits[:,:,1]
        self.live        = bits[:,:,2]
        self.dud         = bits[:,:,3]
        self.reference   = bits[:,:,4]
        self.tf          = bits[:,:,5]
        self.lib         = bits[:,:,6]
        self.pinned      = bits[:,:,7]
        self.ignore      = bits[:,:,8]
        self.washout     = bits[:,:,9]
        self.exclude     = bits[:,:,10]
        self.keypass     = bits[:,:,11]
        self.badkey      = bits[:,:,12]
        self.short       = bits[:,:,13]
        self.badppf      = bits[:,:,14]
        self.badresidual = bits[:,:,15]
        
        # Add some post-processing
        self.masks.extend(['filtered','filtpass'])
        self.filtered = bits[:,:,12:].sum(2) > 0
        self.filtpass = np.logical_and( self.lib , ~self.filtered )
        
    def _get_best(self, datadir ):
        path = os.path.join( datadir, 'sigproc_results', 'analysis.bfmask.bin' )
        try: 
            self.read_file( path )
            self.file = path 
            return
        except IOError:
            pass

        dirs = self._get_blocks( os.path.join ( datadir, 'sigproc_results' ) )
        filenames = [ os.path.join( datadir, 'sigproc_results', d, 'analysis.bfmask.bin' ) for d in dirs ]
        if self._stich_blocks( filenames ):
            return

        path = os.path.join( datadir, 'sigproc_results', 'bfmask.bin' )
        try: 
            self.read_file( path )
            self.file = path 
            return
        except IOError:
            pass

        filenames = [ os.path.join( datadir, 'sigproc_results', d, 'bfmask.bin' ) ]
        if self._stich_blocks( filenames ):
            return

        raise IOError('Unable to find an appropriate bfmask file')

    def _get_blocks( self, directory ):
        block = re.compile( r'block_X[0-9]+_Y[0-9]+')

        files = os.listdir( directory )
        return [ d for d in files if block.match( d ) ]

    def _block_start( self, directory ):
        """
        Returns the X and Y coordinates from the directory name
        """
        block = re.compile( r'block_X[0-9]+_Y[0-9]+')
        blockname = block.search( directory ).group()
        r = int(blockname.split('_')[-1][1:])
        c = int(blockname.split('_')[-2][1:])
        return r, c

    def _stich_blocks( self, filenames ):
        """
        Loads the Bfmask files from each filename.
        """

        # Make sure all files are present
        if not all([ os.path.exists( f ) for f in filenames ]):
            return False
        # Read the first block to set parameters
        block = Bfmask( filenames.pop() )
        self.file = filenames
        if (block.rows, block.cols) == (1332, 1288):
            # PI chip
            self.rows = 1332*8
            self.cols = 1288*12
        elif (block.rows, block.cols) == (664, 640):
            # P0 chip
            self.rows = 664*8
            self.cols = 640*12
        self.masks = block.masks
        self.miniblock = block.miniblock
        self.block = block.block
        self.interval = block.interval
        # Initialize boolean arrays
        empty = np.zeros( ( self.rows, self.cols ), dtype=np.bool )
        self.empty = empty.copy()
        self.bead = empty.copy()
        self.live = empty.copy()
        self.dud = empty.copy()
        self.reference = empty.copy()
        self.tf = empty.copy()
        self.lib = empty.copy()
        self.pinned = empty.copy()
        self.ignore = empty.copy()
        self.washout = empty.copy()
        self.exclude = empty.copy()
        self.keypass = empty.copy()
        self.badkey = empty.copy()
        self.short = empty.copy()
        self.badppf = empty.copy()
        self.badresidual = empty.copy()
        self.filtered = empty.copy()
        self.filtpass = empty.copy()
        # Add the block
        self._block_insert( block )

        for f in filenames:
            block = Bfmask( f )
            self._block_insert( block )

        return True

    def _block_insert( self, block ):
        rc = self._block_start( block.file )
        self._array_insert( self.empty, block.empty, rc )
        self._array_insert( self.bead, block.bead, rc )
        self._array_insert( self.live, block.live, rc )
        self._array_insert( self.dud, block.dud, rc )
        self._array_insert( self.reference, block.reference, rc )
        self._array_insert( self.tf, block.tf, rc )
        self._array_insert( self.lib, block.lib, rc )
        self._array_insert( self.pinned, block.pinned, rc )
        self._array_insert( self.ignore, block.ignore, rc )
        self._array_insert( self.washout, block.washout, rc )
        self._array_insert( self.exclude, block.exclude, rc )
        self._array_insert( self.badkey, block.badkey, rc )
        self._array_insert( self.short, block.short, rc )
        self._array_insert( self.badppf, block.badppf, rc )
        self._array_insert( self.badresidual, block.badresidual, rc )
        self._array_insert( self.filtered, block.filtered, rc )
        self._array_insert( self.filtpass, block.filtpass, rc )

    def _array_insert( self, arr, new, start ):
        """
        Inserts the new array into arr at the starting coordinates.  This operates in place
        """
        try:
            arr[start[0]:start[0]+new.shape[0],
                start[1]:start[1]+new.shape[1]] = new
        except:
            print( 'Error inserting into array.  Likely a shape-mismatch.')
            print( '   arr.shape:   %s, %s' % ( arr.shape ))
            print( '   new.shape:   %s, %s' % ( new.shape ))
            print( '   start:       %s, %s' % ( start ))
            print( '   end:         %s, %s' % ( start[0]+new.shape[0], start[1]+new.shape[1] ) )
            raise

    def _interpret( self ):
        """ Converts a 2d array of unsigned integers to a 3d array of bits """
        # Check for acceptable bits and datatype
        d = self.data.dtype
        if d == np.uint8:
            bitlen = 8
        elif d == np.uint16:
            bitlen = 16
        elif d == np.uint32:
            bitlen = 32
        elif d == np.uint64:
            bitlen = 64
        else:
            raise TypeError('Data type is not unsigned integer of 8,16,32, or 64 bits')
        bits = np.zeros ( (self.data.shape[0] , self.data.shape[1] , bitlen ) , bool )
        test = np.ones  ( self.data.shape , self.data.dtype )
        # shift through the bits
        for i in range(bitlen):
            j = (bitlen-1) - i
            x = np.right_shift( self.data , j )
            bits[:,:,j] = np.bitwise_and( x , test )
        return bits
    
    def heatmap( self , bfmask , path='' ):
        """ Plots a heatmap of the desired beadfind mask """
        msk  = getattr( self , bfmask )
        x = stats.block_avg( np.array( msk , float ) , self.miniblock , np.ones( msk.shape ))
        plt.imshow   ( x , interpolation='nearest' , origin='lower' , clim=[0,1] )
        # Might want to turn off axis ticks since we're block averaging...?
        plt.xticks   ( np.arange( 0 , x.shape[1] + 1 , x.shape[1] / 6 ) , np.arange( 0 , self.cols + 1 , self.cols / 6 ) )
        plt.yticks   ( np.arange( 0 , x.shape[0] + 1 , x.shape[0] / 4 ) , np.arange( 0 , self.rows + 1 , self.rows / 4 ) )
        plt.title    ( 'bfmask.%s' % bfmask )
        colorbar ( shrink = 0.7 )
        if path == '':
            plt.show ( )
        else:
            plt.savefig( os.path.join( path , '%s_heatmap.png' % ( bfmask ) ))
            plt.close  ( ) # Not sure we need this statement
        return None
    
    def plot_mask( self , bfmask , path='' ):
        """ Plots a plain mask of the desired beadfind mask [binary] """
        plt.imshow ( getattr( self , bfmask ) , interpolation='nearest' , origin='lower' , clim=[0,1] )
        plt.title    ( 'bfmask.%s' % bfmask )
        plt.xticks   ( np.arange( 0 , self.cols + 1 , self.cols / 6 ) )
        plt.yticks   ( np.arange( 0 , self.rows + 1 , self.rows / 4 ) )
        if path == '':
            plt.show ( )
        else:
            plt.savefig( os.path.join( path , '%s_mask.png' % ( bfmask ) ))
            plt.close  ( ) # Not sure we need this statement
        return None
    
    def edge_plot( self, bfmask , top=False , rows_to_avg=12 , path='' ):
        """ Plots edge beadfind mask effects using edge blocks on toppp or bottom (default) of chip """
        mod  = self.block[0] % rows_to_avg
        msk  = getattr( self , bfmask )
        if top:
            ttl  = 'top'
            data = np.array( msk[ 7*self.block[0]: , 3*self.block[1]:9*self.block[1] ] , float ).mean(1)
            if mod == 0:
                y = data.reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , self.block[0] , rows_to_avg )
            else:
                y = data[ mod: ].reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , (self.block[0]-mod) , rows_to_avg )
            plt.plot( x[::-1] , 100. * y )
        else:
            ttl = 'bottom'
            data = np.array( msk[ :self.block[0] , 3*self.block[1]:9*self.block[1] ] , float ).mean(1)
            if mod == 0:
                y = data.reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , self.block[0] , rows_to_avg )
            else:
                y = data[ :-mod ].reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , (self.block[0]-mod) , rows_to_avg )
            plt.plot  ( x , 100. * y )
        plt.title ( 'bfmask.%s per row at %s of chip (%%)' % ( bfmask , ttl ) )
        plt.xlabel( 'rows from %s of chip' % ttl )
        plt.ylabel( 'Percent pixels in %s mask' % bfmask )
        if path == '':
            plt.show ( )
        else:
            plt.savefig( os.path.join( path , '%s_%s_edge.png' % ( bfmask , ttl ) ))
            plt.close  ( ) # Not sure we need this statement
        return x , y
    
    def linear( self , bfmask , ax=1 , error_shade=False , path='' ):
        """ Creates a row or col average plot of the desired bfmask (ax=1 is a transverse, 'column averaged' view)"""
        msk  = getattr( self , bfmask )
        lbls = ['columns' , 'rows']
        alt  = [ 1 , 0 ]
        if ax == 0:
            view = np.array( msk[ 3*self.block[0]:5*self.block[0] , : ] , float ).mean(0).reshape(-1,self.interval[0])
            data = view.mean(0)
            err  = view.std(0)
            ttl  = 'bfmask.%s: row-averaged trend, [ inlet --> outlet ]' % bfmask
            avgd = 'row'
        elif ax == 1:
            view = np.array( msk[ : , 3*self.block[1]:9*self.block[1] ] , float ).mean(1).reshape(-1,self.interval[1])
            data = view.mean(1)
            err  = view.std(1)
            ttl  = 'bfmask.%s: col-averaged trend, [ bottom --> top ]' % bfmask
            avgd = 'col'
        else:
            raise ValueError( "Axis must be either 0 or 1" )
        
        x = np.arange( data.shape[0] )
        plt.plot   ( x , 100. * data , 'b-' )
        xlim   ( 0 , data.shape[0] )
        plt.xticks ( np.arange( 0 , data.shape[0]+1 , data.shape[0] / 6 ) , np.arange( 0 , msk.shape[alt[ax]]+1 , msk.shape[alt[ax]] / 6 ) )
        ylim   ( 0 , 100 )
        plt.title  ( ttl )
        plt.xlabel ( lbls[ax] )
        plt.ylabel ( 'Percent pixels in %s mask' % bfmask )
        if error_shade:
            plt.plot         ( x , 100. * (data + err) , '-' , color='grey' )
            plt.plot         ( x , 100. * (data - err) , '-' , color='grey' )
            fill_between ( x , 100.*(data-err) , 100.*(data+err) , facecolor='blue' , alpha=0.5 )
        if path == '':
            plt.show ( )
        else:
            plt.savefig( os.path.join( path , '%s_%s_avg.png' % ( bfmask , avgd ) ))
            plt.close  ( ) # Not sure we need this statement
        return x , data
    
    def odds_ratio( self , top=False , rows_to_avg=12 , path='' ):
        """ Plots edge beadfind mask effects using at top or bottom (default) of chip using odds ratio """
        mod  = self.block[0] % rows_to_avg
        msk  = self.bead
        if top:
            ttl  = 'top'
            data = np.array( msk[ 7*self.block[0]: , 3*self.block[1]:9*self.block[1] ] , float ).mean(1)
            if mod == 0:
                y = data.reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , self.block[0] , rows_to_avg )
            else:
                y = data[ mod: ].reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , (self.block[0]-mod) , rows_to_avg )
            OR = y/(1-y)
            plt.plot( x[::-1] , OR )
        else:
            ttl = 'bottom'
            data = np.array( msk[ :self.block[0] , 3*self.block[1]:9*self.block[1] ] , float ).mean(1)
            if mod == 0:
                y = data.reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , self.block[0] , rows_to_avg )
            else:
                y = data[ :-mod ].reshape( -1 , rows_to_avg ).mean(1)
                x = np.arange( 0 , (self.block[0]-mod) , rows_to_avg )
            OR = y/(1-y)
            plt.plot  ( x , OR )
        plt.title ( 'Odds ratio for loading per row at %s of chip (%%)' % ( ttl ) )
        plt.xlabel( 'rows from %s of chip' % ttl )
        plt.ylabel( 'Odds ratio for loading' )
        if path == '':
            plt.show ( )
        else:
            plt.savefig( os.path.join( path , 'bead_%s_edge_OR.png' % ( ttl ) ))
            plt.close  ( ) # Not sure we need this statement
        return x , OR
    
    def spamalyze( self , outdir , masks=[]):
        """ Performs a canned set of analysis and plot creation for a given object of Bfmask class """
        output = {}
        if outdir == '':
            print( 'ERROR: Cannot use spamalyze without saving images. Will not spam image displaying.')
        else:
            if not os.path.exists( outdir ):
                os.mkdir( outdir )
            # Spamalyze!
            if masks == []:
                tbs = self.masks
            else:
                tbs = masks
            for mask in tbs:
                self.heatmap( mask , outdir )
                output['x'] , output['%s_bot' % mask ] = self.edge_plot( mask , top=False , path=outdir )
                output['x'] , output['%s_top' % mask ] = self.edge_plot( mask , top=True  , path=outdir )
                _           , output['odds_ratio_bot'] = self.odds_ratio( top=False , path=outdir )
                _           , output['odds_ratio_top'] = self.odds_ratio( top=True  , path=outdir )
                output['fc_x'] , output['%s_col_avg' % mask ] = self.linear( mask , path=outdir )
        return output

class BfmaskStats:
    """ Class for interaction with analysis.bfmask.stats files """
    def __init__( self, stats_file=None, body='' ):
        """ 
        stats_file is the path to the file (includes the name of the file)
        Reads in the analysis.bfmask.stats file for reading. Lives in sigproc_results.
        Not all values are very helpful but will still be pulled into memory.
        filtering data doesn't match front page report.
        """
        self.data = {}
        
        # First check to see if we directly supplied the file body text:
        if body:
            for line in body.splitlines():
                self.read_line( line )
        elif stats_file:
            with open( stats_file, 'r' ) as f:
                for line in f.readlines():
                    self.read_line( line )
                    
    def calc_loading( self ):
        """ Returns loading percentage (ignores ignored wells) """
        available_wells = self.data['Total Wells'] - self.data['Excluded Wells']
        return 100. * float( self.data['Bead Wells'] ) / available_wells
    
    def read_line( self, line ):
        line = line.strip()
        if '=' in line:
            key = line.split('=')[0].strip()
            val = line.split('=')[1].strip()
            if '.' in val:
                self.data[key] = float( val )
            else:
                self.data[key] = int( val )
                
class DatasetsBasecaller:
    """ Class for interaction with the basecaller dataset """
    def __init__( self, basecaller_dir, planned_barcodes=[] ):
        """ 
        basecaller_dir:   Directory where datasets_basecaller.json lives OR a sequencing run link from base to the actual basecaller_dir.
        planned_barcodes: list of barcode names from plan, if not supplied this will read all barcodes!
        """
        try:
            with open( os.path.join( basecaller_dir, 'datasets_basecaller.json' ), 'r' ) as f:
                self.data = json.load( f )
        except:
            print( 'Failed to find local data . . . trying the web.' )
            pass
        
        try:
            _apiuser = 'ionadmin'
            _apipwd  = 'ionadmin'
            _apiauth = requests.auth.HTTPBasicAuth( _apiuser, _apipwd )
            
            # Add some error proofing to allow use of TS run links and permutations thereof
            if '/metal' in basecaller_dir:
                run_url = basecaller_dir.split('/metal')[0]
            else:
                run_url = basecaller_dir
                
            full_url = '{}/metal/basecaller_results/datasets_basecaller.json'.format( basecaller_dir )
            print( full_url )
            ans = requests.get( full_url, auth=_apiauth ) 
            self.data = ans.json()
            print( 'Loaded from web url' )
        except:
            print( "Error!  There is no datasets_basecaller file!" )
            return None
        
        self.all_barcodes     = [ g.split('.')[1] for g in self.data['read_groups'].keys() ]
        self.planned_barcodes = planned_barcodes
        self.metrics = {}
        self.barcode_data = {}

        # High level metrics
        self.metrics['total_reads'] = np.sum( [self.data['read_groups'][g]['read_count'] for g in self.data['read_groups']] )
        
    def get_barcode_data( self, barcodes=[] ):
        """
        Gets Q20 data from all the barcodes unless planned barcodes are supplied.
        """
        rgs = self.data['read_groups']
        
        # If no barcodes are supplied, use planned barcodes.
        if not barcodes:
            if self.planned_barcodes:
                barcodes = self.planned_barcodes
            else:
                # Default to all.....
                barcodes = self.all_barcodes
                
        try:
            runid = self.data['read_groups'].keys()[0].split('.')[0]
        except TypeError:
            for k in self.data['read_groups'].keys():
                runid = k.split('.')[0]
                break
            
        for bc in barcodes:
            key = '{}.{}'.format( runid, bc )
            self.barcode_data[bc] = { 'total_bases' : rgs[key].get('total_bases', 0),
                                      'Q20_bases'   : rgs[key].get('Q20_bases', 0),
                                      'read_count'  : rgs[key].get('read_count', 0) }
            
            if self.barcode_data[bc]['total_bases'] > 0:
                self.barcode_data[bc]['percent_Q20'] = 100.*self.barcode_data[bc]['Q20_bases']/self.barcode_data[bc]['total_bases']
            else:
                self.barcode_data[bc]['percent_Q20'] = 0.

    def get_nomatch_data( self ):
        """ Reads nomatch dataset for non-barcoded reads. """
        try:
            key = [ k for k in self.data['read_groups'] if 'nomatch' in k ][0]
        except IndexError:
            print( 'Did not find any data for nomatch barcodes . . . hoping it really is zero.' )
            self.metrics['non_barcoded_reads'] = 0
            
        nonbc_data = self.data['read_groups'][key]
        
        self.non_barcoded_data = nonbc_data
        self.metrics['non_barcoded_reads'] = nonbc_data['read_count']
        
    def calc_unexpected_barcodes( self, planned_bc=[] ):
        """ Calculates sum of reads from unexpected barcodes we found """
        if not planned_bc:
            planned_bc = self.planned_barcodes
            
        expected = np.sum( [self.data['read_groups'][g]['read_count'] for g in self.data['read_groups'] if g.split('.')[1] in planned_bc ] )
        
        try:
            nomatch = self.metrics['non_barcoded_reads']
        except KeyError:
            self.get_nomatch_data( )
            nomatch = self.metrics['non_barcoded_reads']
            
        unexpected = self.metrics['total_reads'] - nomatch - expected
        self.metrics['unexpected_barcoded_reads'] = unexpected
        
    def analyze_q20_data ( self, outdir='' ):
        """ Does analysis of the q20 data from datasets_basecaller """
        # Calculate overall metrics.  Realized we don't know if they do plain average or weighted average.
        lazy_average = np.mean( [self.barcode_data[bc]['percent_Q20'] for bc in self.barcode_data ] )
        
        total_bases  = np.sum( [self.barcode_data[bc]['total_bases'] for bc in self.barcode_data ] )
        Q20_bases    = np.sum( [self.barcode_data[bc]['Q20_bases'] for bc in self.barcode_data ] )
        
        total_reads  = np.sum( [self.barcode_data[bc]['read_count'] for bc in self.barcode_data ] )
        weighted_sum = np.sum( [self.barcode_data[bc]['read_count'] * self.barcode_data[bc]['percent_Q20'] for bc in self.barcode_data ] )
        
        self.metrics['percent_Q20_mean']     = lazy_average
        self.metrics['percent_Q20_raw' ]     = float( Q20_bases ) / total_bases
        self.metrics['percent_Q20_weighted'] = weighted_sum / total_reads
        
        # metadata list
        metadata = [ (self.barcode_data[bc]['percent_Q20'], bc) for bc in sorted( self.barcode_data ) ]
        N = len( metadata )
        x = []
        y = []
        yticklabels = []
        
        for i, md in enumerate( metadata ):
            x.append( md[0] )
            y.append( N - i )
            yticklabels.append( md[1] )
            
        # Make a plot.  Woohoo!
        if N < 32:
            plt.figure( figsize=(8, int(N/2)+1 ) )
        else:
            plt.figure( figsize=(10, 20) )
            
        plt.plot( x, y, 'o' )
        plt.yticks( np.arange( 1, N+1 ), yticklabels[::-1] )
        plt.xlim( 0, 100 )
        plt.xlabel( '%Q20 Bases' )
        plt.ylim( 0, N+1 )
        plt.axvline( x = self.metrics['percent_Q20_mean'], ls='--', color='red' )
        plt.grid( axis='y', ls=':', color='grey' )
        plt.title( 'Barcode Average %Q20 (by bases) = {:.1f}%'.format( self.metrics['percent_Q20_mean'] ) )
        plt.tight_layout()
        if outdir:
            plt.savefig( os.path.join( outdir, 'percent_q20_plot.png' ) )
        else:
            plt.show()
        plt.close( )
        
class ProtonData:
    '''
    ProtonData class which provides handlers for loading sequencing data
    also provides links to related reports

    Key Pointers are:
        path:               the path to the directory
        report:             report number for the particular run
        run_id:             the run id for the particular proton run
        is_tn:              boolean True if report is a thumbnail
        is_auto:            boolean True if report is auto_user
        name:               the report name (experimental details)
        associated_reports: A dictionary list of reports on the same chip with the following fields:
            report:       as above
            is_tn:        as above
            is_auto:      as above
            name:         as above
    '''
    # specify regex searches
    _re_run    = re.compile(r'[-_][R][0-9]+[-_]')
    _re_tn     = re.compile(r'_tn_') 
    _re_auto   = re.compile(r'^Auto_user') 
    _re_report = re.compile(r'[0-9]+$')
    
    def __init__(self, search, method='path', results_path='/ion-results/analysis/output/Home', verbose=False, related=True):   #TODO: Need to make this be able to search from a text file
        '''
        Initializes ProtonData, which provides handlers for loading sequencing data
        also provides links to related reports

        Inputs: 
            search:             search parameter according to the specified method
            method:             way to find the desired method:
                'folder'        Folder name of the results directory, located in results_path
                'path'          Absolute path to the results directory
                'report'        Report number from head node web interface (e.g. blackbird.ite)
                'run'           Report number from run-id (e.g. R123456)
            results_path:       Path to results data folder.  Ignored if using 'path'
            related:            Brings in related reports with the same runid.  Turning this off will make ProtonData load faster

        Key Pointers are:
            path:               the path to the directory
            report:             report number for the particular run
            run_id:             the run id for the particular proton run
            is_tn:              boolean True if report is a thumbnail
            is_auto:            boolean True if report is auto_user
            name:               the report name (experimental details)
            associated_reports: A dictionary list of reports on the same chip with the following fields:
                report:       as above
                is_tn:        as above
                is_auto:      as above
                name:         as above
        '''
        self.verbose = verbose

        # Get the report information
        self.results_path = results_path
        method = method.lower()
        if method == 'folder':
            search = os.path.join( results_path, search )
            self._get_report_by_path( search )
        elif method == 'path':
            self._get_report_by_path( search )
        elif method == 'report':
            self._get_report_by_number( search )
        elif method == 'run':
            self._get_report_by_run( search )
        else:
            raise ValueError('Unknown search method: ' + method)
        
        # Get the related reports
        if related:
            self._get_related_reports()

        # Get barcode files
        self._get_barcodes()

        # Get plugins:
        self._get_plugins()

        # Get block sizes
        self._get_blocks( self.path )

    def __repr__(self):
        return '<ProtonData for %s>' % (self.name)

    def __str__(self):
        return self.name

    def check_loaded_bam( self ):
        """ Checks if the minimal key data was loaded """
        fields = ['inserts', 'deletions','readlength', 'mappedlength']
        loaded = [ 1 for f in fields if hasattr(self.bam, f) ] 
        return sum(loaded) == len(fields)

    def check_loaded_keys( self ):
        """ Checks if the minimal key data was loaded """
        fields = ['wells', 'key', 'rescale']
        loaded = [ 1 for f in fields if hasattr(self.keys, f) ] 
        return sum(loaded) == len(fields)

    def _get_bamfiles( self ):
        bam = re.compile(r'rawlib.bam$')
        files = os.listdir( self.path )
        self.bamfiles = [ f for f in files if bam.search(f) ]
        
    def _get_barcodes( self ):
        if not hasattr( self, 'bamfiles' ):
            self._get_bamfiles()
        
        self.barcodes = []
        for f in self.bamfiles:
            if f == 'rawlib.bam':
                # No barcode
                self.barcodes.append( '' )
            elif 'IonXpress' in f:
                self.barcodes.append( '_'.join( f.split('_')[:2] ) )
            else:
                self.barcodes.append( 'Unknown: %s' % f )
                print( 'Warning! unknown barcode: %s' % f )

    def _get_blocks( self, directory ):
        """
        Reads the block names from the directory and parses the XY positions

        sets:  xstarts:   x coordinate of the start of each block
               ystarts:   y coordinate of the start of each block
               block_cols:  number of columns in a block
               block_rows:  number of rows in a block
               blocks:    2D array of file names
               rows:      # of rows in the chip
               cols:      # of cols in the chip
               nxblocks:  # of X analysis blocks
               nyblocks:  # of Y analysis blocks

        """
        # Read the block directories
        block = re.compile( r'block_X[0-9]+_Y[0-9]+')
        files = os.listdir( self.path )
        files = [ d for d in files if block.match( d ) ]
        
        # Get the x and y coordinates for each block
        xfiles = [ f for f in files if 'Y0' in f ]
        self.xstarts = [ int( f.split('_')[1][1:] ) for f in xfiles ]
        self.xstarts.sort()
        yfiles = [ f for f in files if 'X0' in f ]
        self.ystarts = [ int( f.split('_')[2][1:] ) for f in yfiles ]
        self.ystarts.sort()
        self.block_cols = self.xstarts[1] - self.xstarts[0]
        self.block_rows = self.ystarts[1] - self.ystarts[0]
        self.nxblocks = len(xfiles)
        self.nyblocks = len(yfiles)
        self.rows = self.block_rows * self.nyblocks
        self.cols = self.block_cols * self.nxblocks

        # make array of filenames
        files = []
        for y in self.ystarts:
            row = []
            for x in self.xstarts:
                row.append( 'block_X%i_Y%i' % ( x, y ) )
            files.append( row )

        self.blocks = files

    def _get_plugins( self ):
        '''
        Gets the list of run or running plugins
        '''
        self.plugins = []
        # Get all plugins
        plugin_folders = os.listdir( os.path.join( self.path, 'plugin_out' ) )
        for p in plugin_folders:
            if '_out.' not in p:
                continue
            path = os.path.join( self.path, 'plugin_out', p )
            parts = p.split('_out.')
            plugin_name = parts[0]
            plugin_number = parts[-1]

            self.plugins.append( { 'name':plugin_name, 'number':plugin_number, 'path':path } )

    def _get_related_reports( self ):
        '''
        Searches the server for all reports matching the specified run
        '''
        # Build the search command for the report number, located at the end of the folder
        grep_command = ['ls','-1',self.results_path,'|','grep','-e',str(self.run_id)]

        # Query the run
        query = subprocess.Popen(' '.join(grep_command), shell=True, stdout=subprocess.PIPE).communicate()[0].rstrip() # Python 2.6
        #query = subprocess.check_output(grep_command)
        number_found = query.count('\n')
        if number_found == 0:
            return
            #raise IOError('Run ' + str(self.run_id) + ' not found.')

        # Parse each find
        self.associated_reports = []
        for name in query.rstrip().split():
            report = ProtonData._re_report.search(name).group()
            is_tn = ProtonData._re_tn.search(name) is not None
            is_auto = ProtonData._re_auto.search(name) is not None
            self.associated_reports.append(dict(report=report, is_tn=is_tn, is_auto=is_auto, name=name))

        # Remove this report from the list
        self._trim_related_reports()

    def _get_report_by_number( self, report ):
        '''
        Determines the report information from the report number
        '''
        # Build the search command for the report number, located at the end of the folder
        # Grep is much faser than iterating over all the elements
        grep_command = ['ls','-1',self.results_path,'|','grep',str(report)+'$']

        # Query the report number
        query = subprocess.Popen(' '.join(grep_command), shell=True, stdout=subprocess.PIPE).communicate()[0].rstrip()  # For Python 2.6
        #query = subprocess.check_output(grep_command).rstrip()		# // this is not avalible in Python 2.6

        # Query is a string of results, separated by \n.  If there is more than
        # one result found, that is bad.  We can find that if there are any \n's
        # remaining in query
        if '\n' in query:
            raise IOError('Report number ' + str(report) + ' found multiple results: ' + query)
        elif not query:
            raise IOError('Report number ' + str(report) + ' not found.')
        
        self._get_report_by_path( os.path.join( self.results_path, query ) )

    def _get_report_by_path( self, path ):
        '''
        Extracts report information from the full directory name
        '''
        # Extract the folder name
        self.results_path = os.path.dirname(path)
        folder = os.path.basename(path)

        # Get the path to the data
        self.path = path

        # Get the report number
        self.report = ProtonData._re_report.search(folder).group()

        # Get the run id
        try:
            self.run_id = ProtonData._re_run.search(folder).group()[1:-1]
        except:
            self.run_id = ''

        # Get the experiment name
        self.name = folder 

        # Get run properties
        self.is_tn = ProtonData._re_tn.search( path ) is not None
        self.is_auto = ProtonData._re_auto.search( path ) is not None

    def _get_report_by_run( self, run ):
        '''
        Determines the report information from the run number.  By default, this will 
        try to select the full chip auto_user report.  If not availailable, it will move
        to the thumbnail.  If still not available, it will pick the first one it finds
        '''
        # Make sure that run is a string, prepened with 'R'
        if str(run)[0] != 'R':
            self.run_id = 'R' + str(run)
        else:
            self.run_id = str(run)


        # Get all of the reports based on the run id 
        self._get_related_reports()

        # Find the correct report
        for report in self.associated_reports:
            if report['is_auto'] and not report['is_tn']:
                self._get_report_by_path( os.path.join( self.results_path, report['name'] ) )
                return
        # If you made it here, then the first search failed
        for report in self.associated_reports:
            if report['is_auto'] and report['is_tn']:
                self._get_report_by_path( os.path.join( self.results_path, report['name'] ) )
                return
        # ok, that failed too.  Just grab the first one
        self._get_report_by_path( os.path.join( self.results_path, self.associated_reports[0]['name'] ) )

    def _logging( self, text, level=1 ):
        '''
        Function for controlling output at different logging levels
        '''
        if self.verbose >= level:
            print( text )

    def _trim_related_reports(self):
        '''
        Analyzes the associated_reports to see if the current report is included.
        If it is, it removes it.
        '''
        try:
            self.associated_reports = [ar for ar in self.associated_reports if ar['report'] != self.report]
        except AttributeError:
            pass

    def read_errors_by_well( self ):
        """ Read the bamfiles to count indels and read lengths """
        if not hasattr( self, 'bamfiles' ):
            self._get_bamfiles()
        files = [ os.path.join( self.path, b ) for b in self.bamfiles ]
        self.bam = BamFile( files, shape=(self.rows, self.cols),  verbose=self.verbose )
        self.bam.count_indels()

    def read_loading( self ):
        """ Reads the loading (bfmask.bin') from whatever hasn't been cleaned up already """
        self.loading = Bfmask( self.path, best=True )

    def read_keys( self, tfs=False ):
        """ Reads the key signals from 'raw_peak_signals' and 1.wells files """
        self.keys = WellsFile( self.path, tfs=False, verbose=self.verbose )

    def load_analyzed_bam( self, directory=None ):
        """ Loads a previously analyzed bam result, first checking plugin results """
        # Check if bam file was parsed in the SNOWLI plugin:
        if not hasattr( self, 'plugins' ):
            self._get_plugins()
        plugins = [ p for p in self.plugins if p['name'] == 'SNOWLI_plots' ]
        if len( plugins ) > 0 and directory=='plugin':
            # Sort the plugins to get the most recent
            plugins = sorted( plugins, key=lambda p: p['number'] )
            files = [ os.path.join( plugins[-1]['path'], f ) for f in os.listdir( plugins[-1]['path'] ) if 'rawlib.indels.txt' in f ]
            self._logging('Reading results from SNOWLI_plots plugin')
            self.bam = BamFile( files, shape=(self.rows, self.cols), verbose=self.verbose )
            self.bam.count_indels_from_table()
        else:
            if directory is None:
                directory = '.'
            if not hasattr( self, 'bamfiles' ):
                self._get_bamfiles()
            files = [ os.path.join( self.path, b ) for b in self.bamfiles ]
            self.bam = BamFile( files, shape=(self.rows, self.cols), verbose=self.verbose )
            self.bam.load( directory )

    def load_analyzed_keys( self, directory=None ):
        """ Loads previously analyzed key signals, first checking pluin results """
        if not hasattr( self, 'plugins' ):
            self._get_plugins()
        plugins = [ p for p in self.plugins if p['name'] == 'SpatialKeySignal' ]
        if len( plugins ) > 0 and directory=='plugin':
            # Sort the plugins to get the most recent
            plugins = sorted( plugins, key=lambda p: p['number'] )
            self._logging('Reading results from SpatialKeySignal plugin')
            self.keys = WellsFile( path=plugins[-1]['path'], state='load', verbose=self.verbose )
        else:
            if directory is None:
                directory = '.'
            self.keys = WellsFile( path=directory, state='load', verbose=self.verbose )

    def load_bam( self, directory=None ):
        """ 
        Reads bam data, first from the specified directory, then
        from the plugin, and then from the bam data
        """
        self._logging('...Attempting to read local data')
        self.load_analyzed_bam( directory )
        if not self.check_loaded_bam():
            # OK, local data failed.  attempt to read from plugins
            self._logging('...Attempting to read plugin data')
            self.load_analyzed_bam( 'plugin' )
            if not self.check_loaded_bam():
                # OK, no plugin data either.  build from files
                self._logging('...Could not read from saved data.  building from bam files')
                self.read_errors_by_well()

    def load_keys( self, directory=None ):
        """ 
        Reads 1.wells data, first from the specified directory, then
        from the plugin, and then from the 1.welss data
        """
        # Attempt to read existing local data
        self._logging('...Attempting to read local data')
        self.load_analyzed_keys( directory )
        if not self.check_loaded_keys():
            # OK, local data failed.  attempt to read from plugins
            self._logging('...Attempting to read plugin data')
            self.load_analyzed_keys( 'plugin' )
            if not self.check_loaded_keys():
                # OK, no plugin data either.  build from files
                self._logging('...Could not read from saved data.  building from 1.wells')
                self.read_keys()

    def summary( self ):
        num_associated = len(self.associated_reports)
        output = '-'*20 + '\n'
        output += '  Name:\t\t%s\n  Report #:\t%s\n  Run #:\t%s\n\n' % (self.name,self.report,self.run_id)
        output += '  Path:\t\t%s\n'   % (self.path)
        output += '  Auto_user?\t%s\n  Thumbnail?\t%s\n\n' % (self.is_auto,self.is_tn)
        output += '  %i related reports\n'    % (num_associated)
        output += '-'*20
        return output

class SequencedRun(ProtonData):
    """
    A class for comparing ECC data to a sequenced run.  While the ProtonData class contains information simply about a run,
    this class contains functionss specifically to compare to ECC metrics
    """
    #import matplotlib
    #matplotlib.use('agg',warn=False)
    #import matplotlib.pyplot as plt
    #import convolve as conv
    #import numpy as np
    #import os
    #import string
    #import sys

    def set_ecc( self, eccdata ):
        """ Should be called after __init__.  Sets the ecc data.  should be a ECC_Analysis object """
        self.ecc = eccdata
        self.outputs = os.path.join( self.ecc.output_dir, 'sequencing' )

    def analyze( self ):
        """ Shorthand function to go through standard analysis.  call set_ecc first """
        print('Reading sequencing data')
        self.read()
        print('Saving sequencing data')
        self.save()
        self.ecc.norm_buffering()
        print('Plotting results')
        self.plot()
        self.plot(14)
        print('Making HTML table')
        self.make_report()

    def hist2d( self, x, y, title='', savename='' ):
        """ 
        Calculates and plots the 2d histogram from 2 data sets of the same size.
        x, y are tuples of the form: (data, bins, label)
        If savename is set, the figure is automatically saved and closed
        """

        # Get the elements of each data set
        xdata, xbin, xlabel = x
        ydata, ybin, ylabel = y
        # Calculate the histogram
        H, x, y = np.histogram2d( xdata, ydata, bins=(xbin,ybin) )
        H = H.transpose()
        # Plot the histogram
        extent = [ x[0], x[-1], y[0], y[-1] ]
        plt.figure( facecolor='w' )
        plt.imshow( H, origin='lower', extent=extent, aspect='auto', interpolation='nearest' )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.title( title )
        if savename:
            plt.savefig( savename )
            plt.close()

    def plot( self, window=0, blocks=False, xaxis='buffering'):
        """ 
        Makes comparison plots with the existing data 
        if window = 0, then plots are made using raw values
        otherwise, data is plotted using a window of the specified size for a convolution filter
        If blocks = True, then a subfolder is created and individual analysis blocks are plotted instead
        The x axis for each comparision is set by the xaxis variable
            current suppoted values: 'buffering', 'slopes', 'flowcorr'
        """

        if blocks:
            # Setup the limits for each block
            lims = []
            for x in self.xstarts:
                for y in self.ystarts:
                    lims.append( ( y, y+self.block_cols, x, x+self.block_rows ) )
            dirs = { 'loading':'loading', 'readlength':'readlength', 'indels':'indels', 'key':'key' }
            # Setup directories to save each file
            for d in dirs.iteritems():
                system.makedir( os.path.join( self.outputs, d[1] ) )
        else:
            lims = [ ( None, None, None, None ) ]
            dirs = { 'loading':'', 'readlength':'', 'indels':'', 'key':'' }

        for l in lims:
            mask = np.zeros( (self.rows, self.cols), dtype=np.bool )
            mask[ l[0]:l[1], l[2]:l[3] ] = 1
            mask = self.wellmask * mask
            if window: 
                raw = ''
                self.smooth_sequencing( window=window )

                if xaxis.lower() == 'buffering':
                    self.smooth_buffering( window=window )
                    xvals = self.smooth_buffering[ mask ]
                    xbins = np.arange(0,3,0.05)
                elif xaxis.lower() == 'slopes':
                    self.smooth_slopes( window=window )
                    xvals = self.smoothed_slopes[ mask ]
                    xbins = np.arange(0,3,0.05)
                elif xaxis.lower() == 'flowcorr':
                    self.smooth_flowcorr( window=window )
                    xvals = self.smoothed_flowcorr[ mask ]
                    xbins = np.arange(0,3,0.05)
                else:
                    raise ValueError( 'Unknown xaxis %s' % xaxis )

                loading_bins = np.arange(0,1,0.05)
                loading = self.loading_density[mask]

                length_bins = np.arange(int(np.nanmax(self.bam.readlength)))
                readlength = self.avg_readlength[mask]

                errors_bins = np.arange(0,3,0.02)
                errors = self.error_density[mask]

                key_bins    = np.arange(0,3,0.05)
                key = self.avg_key[mask]
            else:
                raw = '_raw'

                if xaxis.lower() == 'buffering':
                    self.smooth_buffering( window=window )
                    xvals = self.ecc.buffering[mask] / self.ecc.norm_buffering_scale
                    xbins = np.arange(0,3,0.05)
                elif xaxis.lower() == 'slopes':
                    self.smooth_slopes( window=window )
                    xvals = self.smoothed_slopes[ mask ]
                    xbins = np.arange(0,3,0.05)
                elif xaxis.lower() == 'flowcorr':
                    self.smooth_flowcorr( window=window )
                    xvals = self.smoothed_flowcorr[ mask ]
                    xbins = np.arange(0,3,0.05)
                else:
                    raise ValueError( 'Unknown xaxis %s' % xaxis )

                loading_bins = np.arange(0,1.1,.5)
                loading = self.loading.bead[mask]

                length_bins = np.arange(1,int(np.nanmax(self.bam.readlength)))
                readlength = self.bam.readlength[mask]

                errors_bins = np.arange(9)
                errors = self.bam.inserts[mask] + self.bam.deletions[mask]

                key_bins    = np.arange(0,5,.25)
                key = self.keys.wells[mask]

            if blocks:
                blockname = 'block_X%i_Y%i_' % ( l[2], l[0] )
            else:
                blockname = ''

            print('  Plotting loading')
            self.hist2d( ( xvals, xbins, xaxis ), 
                         ( loading, loading_bins, 'loading' ), 
                         savename='%s/%s/%s%s_%s%s.png' % ( self.outputs, dirs['loading'], blockname, xaxis, 'loading', raw )  )
            print('  Plotting read length')
            self.hist2d( ( xvals, xbins, xaxis ), 
                         ( readlength, length_bins, 'read length' ), 
                         savename='%s/%s/%s%s_%s%s.png' % ( self.outputs, dirs['readlength'], blockname, xaxis, 'readlength', raw )  )
            print('  Plotting InDels')
            self.hist2d( ( xvals, xbins, xaxis ), 
                         ( errors, errors_bins, 'InDels' ), 
                         savename='%s/%s/%s%s_%s%s.png' % ( self.outputs, dirs['indels'], blockname, xaxis, 'indels', raw )  )
            print('  Plotting keys')
            self.hist2d( ( xvals, xbins, xaxis ), 
                         ( key, key_bins, '1.wells key' ), 
                         savename='%s/%s/%s%s_%s%s.png' % ( self.outputs, dirs['key'], blockname, xaxis, 'key', raw )  )

        if blocks:
            self._plot_block_report( dirs, xaxis, raw )

    def _plot_block_report( self, directories, xaxis, raw ):
        """ 
        Makes a single html page with blocks arranged side-by-side.  
        directories is a dictionary of directories
        """

        print( 'Writing html output page' )
        
        for key, item in directories.iteritems():
            html = '%s/%s_%s%s.html'   % ( self.ecc.output_dir, xaxis, item, raw )
            with open( html , 'w' ) as f:
                # Page header
                f.write( '<html><head>%s</head><body>\n' % item.capitalize() )

                # Table header 
                f.write( '<table border="1" cellpadding="0" width="100%%">\n' )
                # Loading Table
                try:
                    width = 100/len(self.xstarts)
                    for y in reversed(self.ystarts):
                        row = '<tr>\n'
                        for x in self.xstarts:
                            filename = 'sequencing/%s/block_X%i_Y%i_%s_%s%s.png' % ( item, x, y, xaxis, item, raw )
                            row  += '  <td width="%i%%"><a href="%s"><img src="%s" width="100%%" /></a></td>\n' % ( width, filename, filename )
                        row += '</tr>\n'
                        f.write (row)
                except AttributeError:
                    pass
                f.write( '</table>' )

                # End of page
                f.write( '</body></html>' )

    def read( self ):
        """
        Reads the results from sequencing experiments.  This can often be slow and so it first attempts to read from saved files in ecc.output_dir.  
        It is usually a good idea to call save() after calling read()
        """
        print('Reading loading data')
        self.read_loading()
    
        print('Reading key signals by well')
        self.load_keys( self.outputs )

        print('Reading errors by well')
        self.load_bam( self.outputs )

        self.get_wellmask()

    def make_report( self ):
        def Deindent( text ):
            """
            Removes leading whitespace from block text, preserving new-lines.
            This is particularly useful so that block text can be defined with python-friendly indentation and then called just before the text is written.
            """
            lines = text.split('\n')
            newlines = [l.lstrip() for l in lines]
            return '\n'.join(newlines)

        def MakePageHeader():
            """ Makes subpage headers & common styles etc. """
            header = '''
                     <html>
                     <head>
                     <title>ECC Sequencing Comparision</title>
                     </head>
                     <body>
                     <style type="text/css">tr.d0 {font-size:1.5em; text-align:right; background-color: #eee;}</style>
                     <style type="text/css">tr.d1 {font-size:1.5em; text-align:right; background-color: #fff;}</style>
                     <style type="text/css">table.p {border-collapse:collapse; border: 2px solid black;}</style>
                     <style type="text/css">table.i {border-collapse:collapse; border: 1px solid grey; padding:0px;}</style>
                     <style type="text/css">td   {font-size:0.75em; padding:0px;}</style>
                     <style type="text/css">td.n  {font-size:1em; padding:0px;}</style>
                     <style type="text/css">td.i {border: 1px solid grey; padding:0px;}</style>
                     <style type="text/css">td.t {background-color: #333; color:white; text-align:center; vertical-align:middle; font-size:1em; font-weight:bold; border: 2px solid black;}</style>
                     <style type="text/css">td.p {border: 2px solid black;}</style>
                     <style type="text/css">th.p {border: 2px solid black;}</style>
                     <style type="text/css">th.sm {font-size:0.75em; padding:0px;}</style>
                     <style type="text/css">div.g {background-color: #ddd; border-width:1px; border-color:#333; border-top-style:solid; border-bottom-style:solid;}</style>
                     <style type="text/css">div.w {background-color: #fff; border-width:1px; border-color:#333; border-top-style:solid; border-bottom-style:solid;}</style>
                     <style type="text/css">em {padding-left:1em};</style>
                     <style type="text/css">p  {text-indent:30px};</style>
                     <h1><center>ECC Sequening comparison</center></h1><br>
                     '''

            return header

        def MakeTableHeader( title ):
            header = '''<h2>%s</h2><br>
                     <table border="0" width="100%%" cellpadding="0">
                     <tr><th>Raw Wells</th>
                         <th>Avg Wells</th>
                     </tr>
                     ''' % (title)
            return header

        def MakeLine( x, y ):
            fn = 'sequencing/%s_%s_raw.png' % (x,y)
            line = '<tr>'
            line += '<td width="50%%"><a href="%s"><img src="%s"/></a></td>' % (fn, fn)
            fn = 'sequencing/%s_%s.png' % (x,y)
            line += '<td width="50%%"><a href="%s"><img src="%s"/></a></td>' % (fn, fn)
            line += '</tr>\n'
            
            return line

        def MakeTableFooter(): 
            return '</table>\n'

        def MakePageFooter():
            return '</body>'

        tables = ['buffering']
        sequencing = ['loading','readlength','indels','key']
        text = ''
        text += MakePageHeader()
        for t in tables:
            text += MakeTableHeader( t )
            for s in sequencing:
                text += MakeLine( t, s )
            text += MakeTableFooter()
        text += MakePageFooter()

        fn = os.path.join( self.ecc.output_dir, 'sequencing.html' )
        open( fn, 'w' ).write( Deindent( text ) )

    def get_wellmask( self ):
        ''' Reads and generates the well mask from the outline file '''
        print('Reading edge mask')
        wellmaskdir = os.path.join( moduleDir, 'dats' )
        #wellmaskfile = os.path.join( wellmaskdir, 'p1mask.bin' )
        wellmaskfile = os.path.join( wellmaskdir, 'p1mask.dat' )
        #self.wellmask = np.fromfile( wellmaskfile, dtype=np.bool ).reshape( self.ecc.rows, self.ecc.cols )
        edges = np.fromfile( wellmaskfile, dtype=np.bool ).reshape( -1, 2 )
        self.wellmask = np.zeros( (self.ecc.rows, self.ecc.cols), dtype=np.bool )
        for i, lims in enumerate( edges ):
            self.wellmask[i,lims[0]:lims[1]] = True

    def save( self ):
        """ Saves the analyzed proton data to the ecc output directory """
        self.bam.save( self.outputs )
        self.keys.save( self.outputs )
        
    def smooth_buffering( self, window=14 ):
        ''' Smoothes self.buffering and normalizes by the median, saving to self.smooth_buffering '''
        try:
            # This is a lengthly calculation.  Let's not try to repeat it
            if self._buffering_smoothed == window:
                return
        except AttributeError:
            pass

        self.ecc.norm_buffering()
        norm_buffering = self.ecc.buffering / self.ecc.norm_buffering_scale
        mask = np.logical_or( np.isinf( norm_buffering ), np.isnan( norm_buffering ) )
        self.smooth_buffering = conv.masked2( norm_buffering, conv.ones(window), mask )

        self._buffering_smoothed = window

    def smooth_slopes( self, window=14 ):
        ''' Smoothes self.slopes and normalizes by the median, saving to self.smooth_slopes '''
        try:
            # This is a lengthly calculation.  Let's not try to repeat it
            if self._slopes_smoothed == window:
                return
        except AttributeError:
            pass

        norm_slopes = self.ecc.slopes / np.median( self.ecc.slopes )
        mask = np.logical_or( np.isinf( norm_slopes ), np.isnan( norm_slopes ) )
        self.smoothed_slopes = conv.masked2( norm_slopes, conv.ones(window), mask )

        self._slopes_smoothed = window

    def smooth_flowcorr( self, window=14 ):
        ''' Smoothes self.flowcorr and normalizes by the median, saving to self.smooth_flowcorr '''
        try:
            # This is a lengthly calculation.  Let's not try to repeat it
            if self._flowcorr_smoothed == window:
                return
        except AttributeError:
            pass

        norm_flowcorr = self.ecc.flowcorr / np.median( self.ecc.flowcorr )
        mask = np.logical_or( np.isinf( norm_flowcorr ), np.isnan( norm_flowcorr ) )
        self.smoothed_flowcorr = conv.masked2( norm_flowcorr, conv.ones(window), mask )

        self._flowcorr_smoothed = window

    def smooth_loading( self, window=14 ):
        ''' Smoothes self.loading saving to self.loading_density '''
        try:
            # This is a lengthly calculation.  Let's not try to repeat it
            if self._loading_smoothed == window:
                return
        except AttributeError:
            pass

        beads = self.loading.bead.astype(np.float)
        self.loading_density = conv.conv2( beads, conv.ones(window) )

        self._loading_smoothed = window

    def smooth_indels( self, window=14 ):
        ''' Smoothes indels saving to self.error_density '''
        try:
            # This is a lengthly calculation.  Let's not try to repeat it
            if self._indels_smoothed == window:
                return
        except AttributeError:
            pass

        mask = self.bam.readlength == 0
        errors = self.bam.inserts + self.bam.deletions
        self.error_density = conv.masked2( errors, conv.ones(window), mask )

        self._indels_smoothed = window

    def smooth_readlength( self, window=14 ):
        ''' Smoothes readlength saving to self.avg_readlength '''
        try:
            # This is a lengthly calculation.  Let's not try to repeat it
            if self._readlength_smoothed == window:
                return
        except AttributeError:
            pass

        mask = self.bam.readlength == 0
        self.avg_readlength = conv.masked2( self.bam.readlength, conv.ones(window), mask )

        self._readlength_smoothed = window

    def smooth_key( self, window=14 ):
        ''' Smoothes key signal saving to self.avg_keys '''
        try:
            # This is a lengthly calculation.  Let's not try to repeat it
            if self._key_smoothed == window:
                return
        except AttributeError:
            pass

        mask = self.bam.readlength == 0
        self.avg_key = conv.masked2( self.keys.wells, conv.ones(window), mask )

        self._key_smoothed = window

    def smooth_sequencing( self, window=14 ):
        self.smooth_loading( window )
        self.smooth_indels( window )
        self.smooth_readlength( window )
        self.smooth_key( window )
    
class TorrentHub:
    """ Class to parse torrenthub.csv files """
    # Object constants
    empty_fields  = [ '', 'BC', 'edit' ]
    string_fields = [ 'owner', 'expdetails', 'protonfailuremode', 'date', 'runname', 'sample', 'bead', 'library', 'project', 'ionsite', 
                      'chipefuse', 'chiptype', 'a.id', 'a.runid', 'a.status', 'a.server' ]
    int_fields    = [ 'id', 'emulsionid', 'flows', 'oversample', 'a.100q17', 'a.100q20', 'a.200q17', 'a.200q20', 'a.aq17bases', 'a.aq20bases', 'a.libclonalclonal', 
                      'a.librarybeads', 'a.livebeads',  'a.q17mean', 'a.q20mean', 'a.totalreads', 'a.q20alignments', 'a.SNOWLI_insertion_rate_first_nonzero_percentile', 
                      'a.100q47', 'a.200q47', 'a.dudbeads', 'a.40q7' ]
    float_fields  = [ 'a.loadingdensity', 'a.peaksignal', 'a.percentclonal', 'a.percenttemppos', 'a.rawAccuracy', 'a.snr', 'a.tfpeaksignal', 'a.200q17/100q17', 
                      'a.cf', 'a.droop', 'a.filtkeypassfailperc', 'a.filtpolyclonalperc', 'a.ie', 'a.q20basesperwell', 'a.raw_acc_noDeletions', 'a.raw_acc_noInsertions',
                      'a.raw_acc_noMismatches', 'a.raw_acc_totalBases', 'a.1TA_raw_insertion_error_at_100bp', 'a.SNOWLI_insertion_rate_75_percentile',
                      'a.SNOWLI_insertion_rate_90_percentile', 'a.200q20/librarybeads', 'a.200q20/100q20', 'a.200q47/100q47', 'a.100q20/totalreads', 'a.200q20/totalreads',
                      'a.200q47/totalreads', 'a.totalreads/librarybeads' ]

    def __init__( self, filename='torrenthub.csv', excel=False ):
        """" 
        loads and parses the table 
        set excel=True if it's been edited in excel
        """
        self.filename = filename
        self.excel = excel

        f = open( filename,'r' )
        self.parse_header( f.readline() )

        self.parse_rows( f )

    def parse_header( self, line ):
        parts = line.split(',')
        self.keys = []
        for item in parts:
            # Remove quotes at each end
            item = item.replace( '"', '' ).strip()
            self.keys.append( item )

    def split_line( self, line ):
        """ Intelegently splits the line, removing quotes and ignoring bad commas """
        realparts = []
        infield = False
        for part in line.split(','):
            if not len( part.strip() ):
                # Empty field
                realparts.append( '' )
            elif part.strip()[0] == '"' and part.strip()[-1] == '"':
                # Add this entry
                realparts.append( part[1:-1].strip() )
            elif part[0] == '"':
                # This is the start of the efuse or number
                field = part
                infield = True
            elif part.strip()[-1] == '"' and infield:
                # This is the end of the efuse or number
                field += ','
                field += part.rstrip()
                realparts.append( field[1:-1].strip() )
                infield = False
            elif infield:
                # Tis is the middle of the efuse or number
                field += ','
                field += part
            else:
                # This is a stand alone excel-added entry
                realparts.append( part.strip() )
        return realparts

    def parse_rows( self, fileobj ):
        self.rows = []
        for line in fileobj:
            if len( set( line.strip() ) ) <= 1:
                # empty row.  Skip
                continue
            if line.strip()[0] == '#':
                # Commented line.  Skip
                continue

            parts = self.split_line( line )
            row = {}

            for (part, key) in zip( parts, self.keys ):
                if key in TorrentHub.empty_fields:
                    continue
                elif key in TorrentHub.string_fields:
                    row[key] = part
                elif key in TorrentHub.int_fields:
                    try:
                        row[key] = int(part.replace(',',''))
                    except ValueError:
                        row[key] = None
                elif key in TorrentHub.float_fields:
                    try:
                        row[key] = float(part.replace(',',''))
                    except ValueError:
                        row[key] = None
                else:
                    try: 
                        row[key] = float(part)
                        print( 'WARNING: Unknown field %s was saved as float' % key )
                    except ValueError:
                        row[key] = part
                        print( 'WARNING: Unknown field %s was saved as string' % key )
            try: 
                row['report'] = int(row['a.id'].split('.')[-1])
            except (KeyError,ValueError):
                pass
            self.rows.append( row )

class WellsFile:
    """ 
    Class for reading and storing the 1.wells values 
    
    the following fields are always present:
    .filenames:  list of all files compiled herein
    .wells:     average of key flows for each well. 
    .key:       multiplier to the .wells to get the per-well key signal.  Set to 1 if unable to read
    .key_tf:    multiplier for the test fragments. Set to 1 if unable to read
    .rows:      number of rows in the file
    .cols:      number of columns in the file
    .rescale:   scaling factor to the convert .wells to 1-normalized
    
    the following could be generated for any type of WellsFile but has to be manually called
    .tfs:    average of TF key flows for each well. Must be called in init.

    the following are generated for whole directories (single=False)
    .key_blocks:    The key signal on a per-block basis.  Set to array of 1s of unable to read
    .key_tf_blocks: The TF key signal on a per-block basis
    .block_rows:    number of rows in an analysis block
    .block_cols:    number of cols in an analysis block
    .nx_blocks:     number of blocks in the x direction
    .ny_blocks:     number of blocks in the y direction
    """
    def __init__( self, path='.', state='dir', tfs=False, verbose=False, isfile=False ):
        """ 
        Sets up and loads the wells object
        
        path:   path to either a specific 1.wells file or the sequencing directory (top level). 
        state:  type of path provided.  this may be auto-detected from path 
                    'dir':  default.  Path is a directory containing blocks with 1.wells files
                    'file': path is a specific 1.wells file
                    'load': path is a directory containing previously analyzed 1.wells data
        tfs:    set to True if you would like to get the key signals from TFs as well
        """
        self.verbose = verbose

        # Raise the flag if path is a 1.wells file
        if os.path.basename(path) == '1.wells' or isfile:
            self._logging( "1.wells file detected.  setting state='file'" )
            self.state = 'file'
            self.filenames = [ path ]
        else:
            self.state = state

        if self.state == 'load':
            self.load( path )
            return
        elif self.state == 'dir':
            self._get_files( path )

        # get the block sizes
        self._get_rowcols()

        # read the key signals reported in raw_peak_signal files
        self._get_block_keys()

        # read the 1.wells files
        self._get_wells( tfs )

        # rescale the wells to center around 1
        self._rescale_wells()

    def _array_insert( self, arr, new, start ):
        """
        Adds the array 'new' into 'arr' with the origin at 'start'
        this is an in-place opperation in arr
        """
        try:
            arr[start[0]:start[0]+new.shape[0],
                start[1]:start[1]+new.shape[1]] = new
        except:
            self._logging( 'arr.shape:   %s, %s' % ( arr.shape ), 0 )
            self._logging( 'new.shape:   %s, %s' % ( new.shape ), 0 )
            self._logging( 'start:       %s, %s' % ( start ), 0 )
            self._logging( 'end:         %s, %s' % ( start[0]+new.shape[0], start[1]+new.shape[1] ), 0 )
            raise

    def _blockname2xy( self, blockname ):
        """
        Parses a block name and extracts the X and Y offset.
        returns x, y
        """

        blockParts = blockname.split('_')
        X = int(blockParts[1][1:])
        Y = int(blockParts[2][1:])

        return X, Y

    def _get_files( self, path ):
        """ saves a list of 1.wells files to self.filenames """
        dirlist = os.listdir( path )

        pattern = r'block_X\d+_Y\d+'
        regex = re.compile( pattern )
        blocks = [ x for x in dirlist if regex.match(x) ]
        self.filenames = [ os.path.join( path, d, 'sigproc_results', '1.wells' ) for d in blocks ]
        self._logging( '%i files found' % len(self.filenames) )

    def _get_block_keys( self ):
        """ 
        Reads the key signals for each block, reported in 'raw_peak_signal'
        If unable to find or read a file, the key metric is set to 1
        
        sets self.key
             self.key_tf
             self.key_blocks        if analzing a full directory
             self.key_tf_blocks
        """
        if len( self.filenames ) > 1:
            self._logging( 'Reading full-chip raw_peak_signal' )
            self.key_blocks = np.zeros( ( self.ny_blocks, self.nx_blocks ) )
            self.key_tf_blocks = np.zeros( ( self.ny_blocks, self.nx_blocks ) )

            sigproc_dir = os.path.dirname( self.filenames[0] )
            block_dir = os.path.dirname( sigproc_dir )
            expt_dir = os.path.dirname( block_dir )
            rps_file = os.path.join( expt_dir , 'raw_peak_signal' )
            self.key, self.key_tf = self._read_raw_peaks( rps_file )

        self._logging( 'Reading key raw_peak_signal files for block keys' )
        for f in self.filenames:
            sigproc_dir = os.path.dirname( f )
            block_dir = os.path.dirname( sigproc_dir )
            rps_file = os.path.join( block_dir, 'raw_peak_signal' )
            key, tfkey = self._read_raw_peaks( rps_file )
            try:
                x, y = self._blockname2xy( f.split('/')[-3] )
                self.key_blocks[ y/self.block_rows, x/self.block_cols ] = key
                self.key_tf_blocks[ y/self.block_rows, x/self.block_cols ] = tfkey
            except (AttributeError, KeyError, ValueError, IndexError):
                self._logging( '  Setting global key values' )
                self.key = key
                self.key_tf = tfkey

    def _get_wells( self, set_tfs=False ):
        """ 
        Reads the incorperation signals for each block, reported in '1.wells'
        
        sets self.wells
             self.tfs
        """
        self._logging( 'Reading 1.wells files:' )
        self.wells = np.zeros( ( self.rows, self.cols ), dtype=np.float32 )
        if set_tfs:
            self.tfs = self.wells.copy()

        for f in self.filenames:
            self._logging( '    %s' % f )
            key, tfkey = self._read_wells( f )
            try:
                x, y = self._blockname2xy( f.split('/')[-3] )
                self._array_insert( self.wells, key, (y,x) )
                if set_tfs:
                    self._array_insert( self.tfs, tfkey, (y,x) )
            except IndexError:
                self.wells = key
                self.tfs = tfkey

    def _get_rowcols( self ):
        """
        Sets the size of the data:
            .rows
            .cols
        For compiled blocks, also these:
            .block_rows
            .block_cols
            .nx_blocks
            .ny_blocks
        """
        if len( self.filenames ) == 1:
            # TODO need to attempt to read this information from other files in the block first
            self._logging( '  WARNING: Assuming P1 analysis block (1332,1288)', 1 )
            self.rows = 1332
            self.cols = 1288
            return

        self._logging( 'Getting block sizes from filenames' )
        # Get the name of the directories
        try:
            blocknames = [ p.split('/')[-3] for p in self.filenames ]
            y = [ self._blockname2xy(X)[1] for X in blocknames if 'X0' in X ]
            x = [ self._blockname2xy(Y)[0] for Y in blocknames if 'Y0' in Y ]
        except (KeyError, ValueError):
            self._logging( '  WARNING: Unable to determine array sizes from block names.  Assuming P1 (10656x15456)', 0 )
            self.block_rows = 1332
            self.block_cols = 1288
            self.nx_blocks = 12
            self.ny_blocks = 8
            self.rows = self.block_rows * self.ny_blocks
            self.cols = self.block_cols * self.nx_blocks
            return

        # Calculate rows and cols
        x.sort()
        y.sort()
        self.block_rows = y[1] - y[0]
        self.block_cols = x[1] - x[0]
        self.nx_blocks = len(x)
        self.ny_blocks = len(y)
        self.rows = self.block_rows * self.ny_blocks
        self.cols = self.block_cols * self.nx_blocks

    def _interpret_chiptype( self, arr ):
        """
        attempts to determine the chip type from the total number of wells read in.  Sets
        fields related to well sizes
        """
        if len( arr ) == 1332*1288:
            self._logging( '  WARNING: Assuming P1 block (1332x1288)', 1 )
            self.rows = 1332
            self.cols = 1288
            return arr.reshape( ( self.rows, self.cols ) )
        elif len( arr ) == 1332*1288*8*12:
            self._logging( '  WARNING: Assuming P1 full chip (10656x15456)', 1 )
            self.block_rows = 1332
            self.block_cols = 1288
            self.nx_blocks = 12
            self.ny_blocks = 8
            self.rows = self.block_rows * self.ny_blocks
            self.cols = self.block_cols * self.nx_blocks
            return arr.reshape( ( self.rows, self.cols ) )
        else:
            raise IOError( 'Unable to determine chip type from input dat file' )

    def _logging( self, text, level=1 ):
        '''
        Function for controlling output at different logging levels
        '''
        if self.verbose >= level:
            print (text)

    def _read_raw_peaks( self, filename ):
        """
        Reads the key signals from the specified raw_peak_signals file
        if unable to read a value, returns 1.
        returns libkey, tfkey
        """
        libkey = 1
        tfkey = 1
        try:
            for line in open( filename ):
                if 'Library' in line:
                    libkey = np.int16( line.split()[-1] )
                if 'Test Fragment' in line:
                    tfkey = np.int16( line.split()[-1] )
        except:
            self._logging( 'WARNING: error reading %s.  Setting key to 1' % filename, 0 )
        return libkey, tfkey

    def _read_wells( self, filename, tfs=False ):
        """
        Reads the 1.wells file specifed by filename.  
            returns libkey, tfkey
        if tfs=False, then 
            returns libkey, 0
        """
        # Read the 1.wells file:
        f = h5py.File( filename, 'r' )
        key_flows = np.array( [ np.array( f['wells'][:,:,0] ), 
                                np.array( f['wells'][:,:,2] ),
                                np.array( f['wells'][:,:,5] ) ] )
        keys = key_flows.mean( axis=0 )

        if tfs:
            tf_flows = np.array( [ np.array( f['wells'][:,:,1] ), 
                                   np.array( f['wells'][:,:,3] ),
                                   np.array( f['wells'][:,:,4] ),
                                   np.array( f['wells'][:,:,6] ) ] )
            tfs = tf_flows.mean( axis=0 )
        else:
            tfs = 0

        return keys, tfs

    def _rescale_wells( self ):
        """ 
        Emperical correction factor to the 1.wells values.  
        Generally, the average of full-chip 1.wells values are ~0.6.
        If I exclude the -1 entries, then the average is ~1.7
        The average should be ~1.  I found that I can get an aveage right around 1 by setting the -1(masked) values
        to 0 and then taking the average.
        In order to correct the data, I will rescale the 1.wells values by the average, excluding the masked values
        so that the mean is 1. 
        This works better on full-chips than it does on blocks
        """
        unmasked_wells = self.wells != -1 
        # Need to convert to float64 to prevent roll-over errors in np.mean()   9/22/2014 STP
        self.rescale = self.wells.astype(np.float64)[ unmasked_wells ].mean().astype(np.float32)
        self.wells[ unmasked_wells ] = self.wells[ unmasked_wells ] / self.rescale
        try:
            unmasked_wells = self.tfs != -1
            self.tfs[ unmasked_wells ] = self.tfs[ unmasked_wells ] / self.rescale
        except AttributeError:
            pass

        if len( self.filenames ) == 1:
            self._logging( '  WARNING: Wells data has been re-centered around 1, but this is less accurate for a single 1.wells file', 0 )

    def load( self, path ):
        """
        Saves the data in the specified path.  Path should be a directory, not a filename
        """
        blocks_path = os.path.join( path, 'key_blocks.json' )
        wells_path  = os.path.join( path, 'key_library.dat' )
        tf_path     = os.path.join( path, 'key_tfs.dat' )

        self.filenames = []

        self.load_blocks( blocks_path )
        self.load_wells( wells_path )
        self.load_tfs( tf_path )

    def load_blocks( self, filename ):
        """
        loads the supporting data (usually chip size information) from the json file
        """
        self._logging( 'Reading %s ' % filename )
        try:
            jsontxt = '\n'.join( open( filename, 'r' ).readlines() )
            data = json.loads( jsontxt )
            self.filenames.append( filename )
        except IOError:
            self._logging( 'ERROR loading %s' % filename, 1 )
            return

        for k in data.keys():
            setattr( self, k, data[k] )

    def load_wells( self, filename ):
        """ Loads the specified dat file to self.wells.  Data is read as np.uint8 and divided by 32 """
        self._logging( 'Reading %s ' % filename )
        scale = 32.
        try:
            data = np.fromfile( filename, dtype = np.uint8 )
            self.filenames.append( filename )
        except IOError:
            self._logging( '  WARNING: error reading %s to .wells' % filename, 0 )
            return
        masked = data == 255
        pins = data == 254
        data = data.astype(np.float)/scale
        data[masked] = -1
        data[pins] = 8
        try: 
            self.wells= data.reshape( self.rows, self.cols )
        except AttributeError:
            self.wells = self._interpret_chiptype( data )

    def load_tfs( self, filename ):
        """ Loads the specified dat file to self.tfs.  Data is read as np.uint8 and divided by 32 """
        self._logging( 'Reading %s ' % filename )
        scale = 32.
        try:
            data = np.fromfile( filename, dtype = np.uint8 )
            self.filenames.append( filename )
        except IOError:
            self._logging( '  WARNING: error reading %s to .tfs' % filename, 1 )
            return
        masked = data == 255
        pins = data == 254
        data = data.astype(np.float)/scale
        data[masked] = -1
        data[pins] = 8
        try: 
            self.tfs = data.reshape( self.rows, self.cols )
        except AttributeError:
            self.tfs = self._interpret_chiptype( arr )

    def plot( self, directory='.' ):
        """
        Makes all plots possible using default filenames in the specified directory
        """
        self.plot_blocks( os.path.join( directory, 'block_keys.png' ),   field='lib' )
        self.plot_blocks( os.path.join( directory, 'block_tfkeys.png' ), field='tf' )
        self.plot_wells(  os.path.join( directory, 'keys.png' ),         field='lib' )
        self.plot_wells(  os.path.join( directory, 'tfkeys.png' ),       field='tf' )

    def plot_blocks( self, filename, clim=[50,100], field='lib' ):
        """ 
        Plots the key signal by block.  Saves it to filename
        
        if you want to autoscale the plot, set clim to None
        field is either 'lib' or 'tf'.  
        """
        # get the data
        try:
            if field.lower() == 'lib':
                data = self.key_blocks
                title = 'Library Keys'
            elif field.lower() == 'tf':
                data = self.key_tf_blocks
                title = 'TF Keys'
            else:
                raise ValueError( 'Unknown plot field: %s' % field )
        except AttributeError:
            self._logging( 'WARNING: %s keys do not exist' % field, 2 )
            return
                
        # Make the figure
        plt.figure( facecolor='w' )
        plt.imshow( data, origin='lower', interpolation='nearest' )
        if clim is not None:
            plt.clim( clim )
        plt.colorbar()
        plt.savefig( filename )
        plt.close()

    def plot_wells( self, filename, clim=[0,3], field='lib', scale=False ):
        """
        Plots the wells file, saving to filename
        
        if you want to autoscale the plot, set clim to None
        field is either 'lib' or 'tf'
        if scale is set to True, then the keys are scaled by either self.keys or self.keys_tf.  
            usually, it is best to set clim=None
        """
        # get the data
        try: 
            if field.lower() == 'lib':
                data = self.wells
                datascale = self.key
                title = 'Wells key signal'
            elif field.lower() == 'tf':
                data = self.tfs
                datascale = self.key_tf
                title = 'TF key signals'
            else:
                raise ValueError( 'Unknown plot field: %s' % field )
        except AttributeError:
            self._logging( 'WARNING: %s keys do not exist' % field, 2 )
            return

        
        # scale the data
        if scale:
            data = data*datascale

        # Downsample data to prevent segmentation faults from really large data sets
        extent = [0, data.shape[1], 0, data.shape[0]]
        if data.size > 2000*2000:
            data = data[::10,::10]

        # Make the figure
        plt.figure( facecolor='w' )
        plt.imshow( data, origin='lower', interpolation='nearest', extent=extent )
        if clim is not None:
            plt.clim( clim )
        plt.colorbar()
        plt.savefig( filename )
        plt.close()

    def save( self, path ):
        """
        Saves the data in the specified path.  Path should be a directory, not a filename
        """
        blocks_path = os.path.join( path, 'key_blocks.json' )
        wells_path  = os.path.join( path, 'key_library.dat' )
        tf_path     = os.path.join( path, 'key_tfs.dat' )

        self.save_blocks( blocks_path )
        self.save_wells( wells_path )
        self.save_tfs( tf_path )

    def save_blocks( self, filename ):
        """
        Saves the folloiwng to a json file:
          .rows
          .cols
          .block_rows
          .block_cols
          .nx_blocks
          .ny_blocks
          .key
          .key_tf
          .key_blocks
          .key_tf_blocks
          .rescale
        """
        data = {}
        keys = ['rows','cols','block_rows','block_cols','nx_blocks','ny_blocks','key','key_tf','key_blocks','key_tf_blocks','rescale']
        for k in keys:
            try:
                data[k] = getattr(self, k).tolist()
            except AttributeError:
                try:
                    data[k] = getattr(self, k)
                except AttributeError:
                    pass

        open( filename, 'w' ).write( json.dumps( data ) ) 
        return

    def save_wells( self, filename ):
        """
        Saves the .wells field to a binary .dat file.  For space, data is multiplied by 32 and saved as np.uint8
        """
        scale = 32.
        maxval = 254/scale
        pinval = 255/scale
        try:
            data = self.wells.copy()
            data[ data > maxval ] = maxval
            data[ data < 0 ] = pinval
            data = np.uint8( data*scale )
            data.tofile( filename )
        except AttributeError:
            self._logging( 'WARNING: wells key signal was not saved', 0 )

    def save_tfs( self, filename ):
        """
        Saves the .wells field to a binary .dat file.  For space, data is multiplied by 32 and saved as np.uint8
        """
        scale = 32.
        maxval = 254/scale
        pinval = 255/scale
        try:
            data = self.tfs.copy()
            data[ data > maxval ] = maxval
            data[ data < 0 ] = pinval
            data = np.uint8( data*scale )
            data.tofile( filename )
        except AttributeError:
            self._logging( '  WARNING: tf key signals not present and not saved', 2 )

#---------------------------------------------------------------------------#
# Define functions
#---------------------------------------------------------------------------#
        
