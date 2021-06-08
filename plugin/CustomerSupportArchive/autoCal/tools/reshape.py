''' 
Functions specifically for reshaping arrays

NB: For reshaped array math and/or uniformity, please see stats.py
'''
import numpy as np
import numpy.ma as ma

class BlockReshaper( object ):
    """
    Class facilitating intelligent block reshaping funcationality, typically used as an input to local uniformity analyses.
    
    The reshaped array will live in the 'reshaped' attribute.
    
    Routines herein will return an array of size
        ( data.shape[0]/roi_R, data.shape[1]/roi_C ) = ( numR, numC )
    
    Note the blockR, blockC nomenclature change to roi_R, roi_C for the averaging block size, to avoid confusion with chip 
    block[R,C] in chipinfo.
    
    goodpix is a boolean array of the same size as data where pixels to be averaged are True.  By default all pixels are averaged
    raising finite sets all nans and infs to 0
    
    For large data sets on computers with limited memory, it may help to set superR and superC to break the data into chunks
        This is especially true for masked reshapes where we now have 2x the data size needing to be in memory.
    """
    def __init__( self, data, roi_R, roi_C, goodpix=None, finite=False, superR=None, superC=None ):
        """
        Reminder, goodpix is the mask of pixels you want to include in the calculations (avg. or otherwise).  
        Default is all pixels.
        """
        self.data  = data
        rows, cols = data.shape
        self.roi_R = roi_R
        self.roi_C = roi_C
        
        if ( rows % roi_R ) or ( cols % roi_C ):
            raise ValueError( '[BlockReshaper] Data size (%i, %i) must be an integer multiple of %i x %i' % (rows,cols,roi_R,roi_C ) )
        else:
            self.numR  = rows / roi_R
            self.numC  = cols / roi_C
        
        self.goodpix   = goodpix
        self.blocksize = [roi_R, roi_C] # for posterity to remember variable renaming convention.
        self.finite    = finite
        
        # Args for data chunk support - both are required to be defined for them to be used.
        self.superR = superR
        self.superC = superC
        
        # This is where the magic happens
        self.reshaped = self.block_reshape( )
        
    def mask_exists( self ):
        """ Checks if mask exists and is non-zero, which takes two steps """
        try:
            # This works if self.mask is a np.array
            return self.goodpix.any()
        except AttributeError:
            # It's either None or some other data type which doesn't make any sense.
            return False
        
    def block_reshape( self ):
        """ Higher level reshape function that will do the reshape in a smart fashion. """
        if self.superR and self.superC:
            # Smart chunk
            reshaped = self._block_reshape_by_chunk( self.data, self.roi_R, self.roi_C, self.superR, self.superC )
            if self.mask_exists():
                if self.goodpix.shape != self.data.shape:
                    raise ValueError(
                        '[BlockReshaper] Mask shape (%s) must match that of the input data (%s)!'.format( self.goodpix.shape,
                                                                                                      self.data.shape)
                        )
                else:
                    # Return the reshaped, masked array
                    remasked = np.array( self._block_reshape_by_chunk( self.goodpix, self.roi_R, self.roi_C, self.superR,
                                                                       self.superC), bool )
                    return ma.masked_array( reshaped, np.logical_not( remasked ) )
            else:
                # Return the reshaped, normal array
                return reshaped
        else:
            # Just do it
            reshaped = self._block_reshape( self.data, self.roi_R, self.roi_C )
            if self.mask_exists():
                if self.goodpix.shape != self.data.shape:
                    raise ValueError(
                        '[BlockReshaper] Mask shape (%s) must match that of the input data (%s)!'.format( self.goodpix.shape,
                                                                                                      self.data.shape) )
                else:
                    # Return the reshaped, masked array
                    remasked = np.array( self._block_reshape( self.goodpix, self.roi_R, self.roi_C ), bool )
                    return ma.masked_array( reshaped, np.logical_not( remasked ) )
            else:
                # Return the reshaped, normal array
                return reshaped
            
    def block_unreshape( self, blockmath_result ):
        ''' 
        Expand the results of BlockReshaper operations back to an array of the original data size 
        '''
        return self._block_unreshape( blockmath_result, self.roi_R, self.roi_C )
    
    @classmethod
    def from_chipcal( cls, cc, data, goodpix=None, finite=False, hd=False ):
        """ 
        Handy helper class that leverages all the useful attributes of chipcal objects for all their BlockReshaper needs.
        cc is the chipcal.ChipCal object to be passed in
        data is the array to be analyzed, of course also being of the shape chipR, chipC
        """
        if hd:
            bm = cls( data, cc.chiptype.microR, cc.chiptype.microC, goodpix=goodpix, finite=finite,
                      superR=cc.superR, superC=cc.superC )
        else:
            bm = cls( data, cc.chiptype.miniR,  cc.chiptype.miniC,  goodpix=goodpix, finite=finite,
                      superR=cc.superR, superC=cc.superC )
            
        return bm
    
    @classmethod
    def from_chiptype( cls, chiptype_obj, data, goodpix=None, finite=False, hd=False ):
        """ 
        Handy helper class that leverages all the useful attributes of chiptype objects for all their BlockReshaper needs.
        cc is the chipcal.ChipCal object to be passed in
        data is the array to be analyzed, of course also being of the shape chipR, chipC
        """
        
    @staticmethod
    def _block_reshape( data, roi_R, roi_C ):
        """ 
        Generic helper method for reshaping arrays.  Does not do data chunking.
        """
        rows , cols = data.shape
        if ( rows % roi_R ) or ( cols % roi_C ):
            raise ValueError( '[BlockReshaper] Data size (%i, %i) must be an integer multiple of %i x %i' % (rows,cols,roi_R,roi_C ) )
        else:
            numR = int( rows / roi_R )
            numC = int( cols / roi_C )
        return data.reshape(rows , numC , -1 ).transpose((1,0,2)).reshape(numC,numR,-1).transpose((1,0,2))
    
    @staticmethod
    def _block_reshape_by_chunk( data, roi_R, roi_C, superR, superC ):
        """ 
        Generic helper method to reshape arrays in chunks, useful for very large data files.
        """
        rows , cols = data.shape
        if ( rows % roi_R ) or ( cols % roi_C ):
            raise ValueError( '[BlockReshaper] Data size (%i, %i) must be an integer multiple of %i x %i' % (rows,cols,roi_R,roi_C ) )
        else:
            numR = rows / roi_R
            numC = cols / roi_C
            
        output = np.zeros( (numR,numC,roi_R*roi_C) )
        for sr in ( range( 0, rows, superR ) ):
            slicer   = slice( sr, sr+superR )
            slicerin = slice( sr/roi_R, (sr+superR)/roi_R )
            for sc in ( range( 0, cols, superC ) ):
                slicec = slice( sc, sc+superC )
                slicecin = slice( sc/roi_C, (sc+superC)/roi_C )
                output[slicerin,slicecin] = BlockReshaper._block_reshape( data[slicer,slicec], roi_R, roi_C )
                
        return output
    
    @staticmethod
    def _block_unreshape( data, roi_R, roi_C ):
        ''' Expand the results of block_avg back to its original size '''
        numR, numC = data.shape
        rows = numR*roi_R
        cols = numC*roi_C
        els  = roi_R * roi_C
        data = np.tile( data, (els, 1, 1) ).transpose((1,2,0))
        return data.transpose((1,0,2)).reshape(numC, rows, -1).transpose((1,0,2)).reshape( rows, cols )

    
# For compatability . . . but not recommended for general use.
block_reshape   = BlockReshaper._block_reshape
block_unreshape = BlockReshaper._block_unreshape
