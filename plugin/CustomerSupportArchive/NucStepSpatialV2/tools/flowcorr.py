import os
import numpy as np
from . import imtools, datprops
from .datfile import DatFile
from .chiptype import ChipType

moduleDir = os.path.abspath( os.path.dirname( __file__ ) )

class FlowCorr:
    def __init__( self, chiptype, xblock=None, yblock=None, rootdir='.', method='' ):
        ''' 
        Initialize a flowcorr object 
        chiptype:  a ChipType object
        xblock:    The full-chip column origin; setting to None returns a full chip
        yblock:    The full-chip row origin; setting to None returns a full chip
        rootdir:   root directory to look for flowcorr files. 
                       search will also look up a level, within the 
                       module directory, and in the dats directory
        method:    if specified, automaticaly loads the corresponding flowcorr
                       'buffer'
                       'file'
                   if advanced options need to be passed into the load functions,
                   they should be called separatly with method being left empty
        '''
        self.chiptype    = ChipType(chiptype)
        self.xblock      = xblock
        self.yblock      = yblock
        self.searchpath = [ rootdir, 
                            os.path.join( rootdir, '..' ), 
                            os.path.join( moduleDir, '../dats' ), 
                            moduleDir, 
                            os.path.join( moduleDir, 'dats' ) ]
        if method.lower()   == 'buffer':
            self.frombuffer()
        elif method.lower() == 'file':
            self.fromfile()
        elif not method:
            pass
        else:
            raise ValueError( 'Flowcorr method "%s" is undefined' % method )
        
    def frombuffer(self, flow_file='C2_step.dat', force=False, framerate=15):
        '''
        Returns the flow correction measured from a buffered flow
        flowfile:  measurement file used to calculate the flowcorr
        force:     calculate the data from raw, even if an existing analysis is present
        framerate: fps
        '''

        try:
            if force:
                raise IOError
            self.filename = os.path.join( self.searchpath[0], 'flowcorr_slopes.dat' )
            self.flowcorr = datprops.read_dat( self.filename, 'flowcorr', chiptype=self.chiptype )
        except IOError:
            # Read the dat file
            found = False
            for dirname in self.searchpath:
                self.filename = os.path.join( dirname, flow_file )
                if os.path.exists( self.filename ):
                    found = True
                    break
            if not found:
                raise IOError( '%s was not found' % self.filename )
            data = DatFile( self.filename, chiptype=self.chiptype )

            # Calculate properties
            self.flowcorr = data.measure_slope( method='maxslope' )
            self.time_offset = np.min(data.measure_t0( method='maxslope' ))  #TODO:  This is not very robust.  should just shift t0 here and record the offest instead of trying to do things later with it
            self.pinned = data.measure_pinned()
            # remove pins
            self.flowcorr[ self.pinned ] = 1

            # Save a few more variables
            self.t0      = data.measure_t0( meathod='maxslope' )
            self.actpix  = data.measure_actpix
            self.phpoint = data.measure_plateau()

        return self.flowcorr

    def fromfile( self, fc_type ):
        '''
        Loads the flow correction from file based on the chip type and scales up from miniblocks to full chips or analysis blocks.  
        This method only differentiates based on thumbnail or full chip/analysis block.  All other differences are rolled into ChipType.
        fc_type:  can be 'ecc' or 'wt'.  
                  flowcorr file is defined by self.chiptype.flowcorr_<fc_type>
        '''
        # Thumbnails are enough different to have their own function
        if self.chiptype.tn == 'self':
            return self.tn_fromfile( fc_type )
        # Spatial thumbnails are just subsampled data.  We don't need special loading


        # Calculate the size of the flowcorr files
        xMiniBlocks = self.chiptype.chipC / self.chiptype.miniC
        yMiniBlocks = self.chiptype.chipR / self.chiptype.miniR

        # Set the flowcorr path starting local before using the default
        for path in self.searchpath:
            filename = os.path.join( path, '%s.dat' % getattr( self.chiptype, 'flowcorr_%s' % fc_type ) )
            try:
                flowcorr = datprops.read_dat( filename , metric='flowcorr' )
                break
            except IOError:
                continue
            raise IOError( 'Could not find a flowcorr file' )

        # Scale the flowcorr data to the entire well
        sizes = [ ( 96, 168 ), # This is an unscaled P1-sized flowcorr file.  This is the most likely size when reading fc_flowcorr.dat
                  ( yMiniBlocks, xMiniBlocks ), # This is the historical per-chip file.  This is ( 96, 168 ) for a P1/540 chip
                  ( self.chiptype.chipR, self.chiptype.chipC ) ] # This is the pre-compiled value
        try:
            fc_xMiniBlocks = self.chiptype.fullchip.chipC / self.chiptype.fullchip.miniC
            fc_yMiniBlocks = self.chiptype.fullchip.chipR / self.chiptype.fullchip.miniR
            sizes.append( ( fc_yMiniBlocks, fc_xMiniBlocks ) ) 
            sizes.append( ( self.chiptype.fullchip.chipR, self.chiptype.fullchip.chipC ) ) 
        except AttributeError:
            pass

        for size in sizes:
            try:
                flowcorr = flowcorr.reshape( size )
                break
            except ValueError:
                # Keep going until you itterate through all possible sizes.  If you still get an error, then die
                if size == sizes[-1]:
                    print 'Possible Sizes'
                    print sizes
                    print 'Elements'
                    print flowcorr.shape

                    raise ValueError( 'Could not determine flowcorr size' )
                continue
        # Resize the image to the current size
        if self.chiptype.burger is None:
            # This is a standard resize operation
            flowcorr = imtools.imresize( flowcorr, ( self.chiptype.chipR, self.chiptype.chipC ) )
        elif self.chiptype.spatn != 'self':
            # This is burger mode on a full size chip
            flowcorr = imtools.imresize( flowcorr, ( self.chiptype.burger.chipR, self.chiptype.burger.chipC ) )
            # Clip off the top and bottom
            first = ( flowcorr.shape[0] - self.chiptype.chipR ) / 2
            last  = first + self.chiptype.chipR
            flowcorr = flowcorr[ first:last, : ]
        else:
            # This is burger mode on a spatial thumbnail
            # This has the effect of adding more rows beyond the 800 typically used for a spatial thumbnail
            rows = self.chiptype.chipR * self.chiptype.burger.chipR / self.chiptype.fullchip.chipR
            flowcorr = imtools.imresize( flowcorr, ( rows, self.chiptype.chipC ) )
            # Clip off the top and bottom
            first = ( flowcorr.shape[0] - self.chiptype.chipR ) / 2
            last  = first + self.chiptype.chipR
            flowcorr = flowcorr[ first:last, : ]

        # Reduce to a single analysis block
        if ( self.xblock is not None and self.yblock is not None and 
             self.xblock != -1       and self.yblock != -1 ):
            flowcorr = flowcorr[ self.yblock: self.chiptype.blockR + self.yblock, 
                                 self.xblock: self.chiptype.blockC + self.xblock ]

        self.flowcorr = flowcorr 
        return flowcorr

    def tn_fromfile( self, fc_type ):
        '''
        Gets the per-well flowcorrection for a STANDARD (not spatial) thumbnail
        '''
        # Calculate the size of the flowcorr files
        xMiniBlocks = self.chiptype.chipC / self.chiptype.miniC
        yMiniBlocks = self.chiptype.chipR / self.chiptype.miniR

        # Set the flowcorr path starting local before using the default
        for path in self.searchpath:
            filename = os.path.join( path, '%s.dat' % getattr( self.chiptype, 'flowcorr_%s' % fc_type ) )
            try:
                flowcorr = datprops.read_dat( filename , metric='flowcorr' )
                break
            except IOError:
                continue
            raise IOError( 'Could not find a flowcorr file' )

        # Scale the flowcorr data to the entire well
        sizes = ( ( 96, 168 ), # This is an unscaled P1-sized flowcorr file.  
                  ( 48, 96 ) , # This is an unscaled P0-sized flowcorr file.  
                  ( yMiniBlocks, xMiniBlocks ), # This is the historical thumbnail flowcorr (swapped x & y - STP 7/13/2015)
                  ( self.chiptype.fullchip.chipR, self.chiptype.fullchip.chipC ) ) # This is the pre-compiled value
        for size in sizes:
            try:
                flowcorr = flowcorr.reshape( size )
                break
            except ValueError:
                # Keep going until you itterate through all possible sizes.  If you still get an error, then die
                if size == sizes[-1]:
                    raise ValueError( 'Could not determine flowcorr size' )
                continue
        # Resize the image to the full chip size
        if self.chiptype.burger is None:
            # This is a standard resize operation based on the full chip
            flowcorr = imtools.imresize( flowcorr, ( self.chiptype.fullchip.chipR, self.chiptype.fullchip.chipC ) )
        else:
            # This is burger mode on a regular thumbnail.  Full chip is actually specified by burger and then we have to clip
            flowcorr = imtools.imresize( flowcorr, ( self.chiptype.burger.chipR, self.chiptype.burger.chipC ) )
            # Clip off the top and bottom
            first = ( flowcorr.shape[0] - self.chiptype.fullchip.chipR ) / 2
            last  = first + self.chiptype.fullchip.chipR
            flowcorr = flowcorr[ first:last, : ]

        # Reduce to thumbnail data
        tnflowcorr = np.zeros( ( self.chiptype.chipR, self.chiptype.chipC ) )
        for r in range( self.chiptype.yBlocks ):
            tn_rstart = r*self.chiptype.blockR
            tn_rend   = tn_rstart + self.chiptype.blockR
            #fc_rstart = int( (r+0.5)*self.chiptype.fullchip.blockR ) - self.chiptype.blockR/2
            #            middle of block       in case the thumbnail different yBlocks       center within the block
            fc_rstart = int( (r+0.5)*(self.chiptype.fullchip.chipR/self.chiptype.yBlocks) ) - self.chiptype.blockR/2
            fc_rend   = fc_rstart + self.chiptype.blockR
            for c in range( self.chiptype.xBlocks ):
                tn_cstart = c*self.chiptype.blockC
                tn_cend   = tn_cstart + self.chiptype.blockC
                fc_cstart = int( (c+0.5)*self.chiptype.fullchip.blockC ) - self.chiptype.blockC/2
                fc_cend   = fc_cstart + self.chiptype.blockC
                tnflowcorr[ tn_rstart:tn_rend, tn_cstart:tn_cend ] = flowcorr[ fc_rstart:fc_rend, fc_cstart:fc_cend ]

        self.flowcorr = tnflowcorr 
        return self.flowcorr
