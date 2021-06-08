import struct
import numpy as np

''' Module for interacting with gain.lsr files '''

class GainLSR:
    ''' Class for interacting with the gain lsr file '''
    def __init__( self, path ):
        self.path = path
        self.read()

    def read( self ):
        ''' Read the gain LSR file '''
        with open( self.path ):
            data = f.read()
            hdr  = struct.unpack( '<IIII', data[0:16] )
            dat  = np.array( scruct.unpack_from( '<%df' % ( hdr[2]*hdr[3] ), data[16:] ) ).reshape( [ hdr[2], hdr[3] ] ) 

