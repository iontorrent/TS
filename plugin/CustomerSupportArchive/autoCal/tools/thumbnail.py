''' Contains functions for extracting the thumbnail from fullchip data '''
from .datfile import DatFile
from . import chiptype as ctm
import numpy as np

tn_valid = ( '550_3525', '530v1', '540v1' )
spa_valid = ()

def get_thumbnail( data, chiptype=None, spa=False ):
    ''' Function for getting thumbnail based on block size.
        Chip type is only used if the generic matcher is utilized 
        '''
    rows = data.shape[0]
    cols = data.shape[1]

    # Set parameters for regular or spatial thumbnail
    if spa:
        validated = spa_valid
        root = 'get_spa_%s'
        generic = get_spa_generic
    else:
        validated = tn_valid
        root = 'get_thumbnail_%s'
        generic = get_thumbnail_generic

    for chip in validated:
        ct = ctm.ChipType( chip )
        if ( rows, cols ) == ( ct.chipR, ct.chipC ):
            try:
                return globals()[ root % chip.replace('.','_') ]( data )
            except KeyError:
                pass
    #raise NotImplementedError( 'Thumbnail creation not validated for this chiptype' )
    return generic( data, chiptype )
    
def get_thumbnail_550_3525( data ):
    ct = ctm.ChipType( '550_3525' )

    tn = np.zeros( ( 800, 1200 ), dtype=data.dtype )
    for R in range( 8 ):
        for C in range( 12 ):
            tn_r = R*100
            tn_c = C*100
            data_r = R*ct.blockR + ( 830 if R%2 else 828 )
            data_c = C*ct.blockC + 806
            tn[tn_r:tn_r+100,tn_c:tn_c+100] = data[data_r:data_r+100,data_c:data_c+100]
    return tn

def get_thumbnail_530v1( data ):
    ct = ctm.ChipType( '530v1' )

    tn = np.zeros( ( 800, 1200 ), dtype=data.dtype )
    for R in range( 8 ):
        for C in range( 12 ):
            tn_r = R*100
            tn_c = C*100
            data_r = R*ct.blockR + 284
            data_c = C*ct.blockC + 270
            tn[tn_r:tn_r+100,tn_c:tn_c+100] = data[data_r:data_r+100,data_c:data_c+100]
    return tn

def get_thumbnail_540v1( data ):
    ct = ctm.ChipType( '540v1' )

    tn = np.zeros( ( 800, 1200 ), dtype=data.dtype )
    for R in range( 8 ):
        for C in range( 12 ):
            tn_r = R*100
            tn_c = C*100
            data_r = R*ct.blockR + 616
            data_c = C*ct.blockC + 594
            tn[tn_r:tn_r+100,tn_c:tn_c+100] = data[data_r:data_r+100,data_c:data_c+100]
    return tn

def get_thumbnail_generic( data, chiptype=None ):
    ''' Generic thumbnail creator. Not guaranteed to be accurate! '''
    print( 'WARNING! Creating thumbnail from full chip, but thumbnail not guaranteed to be accurate ' )
    if chiptype is None:
        chiptype = ctm.guess_chiptype( data )
        print( 'WARNING! Chip type guessed to be {}'.format( chiptype.name ) )

    R_b = np.shape(data)[0]/chiptype.yBlocks
    C_b = np.shape(data)[1]/chiptype.xBlocks
    R_s = 100
    C_s = 100
    r_h = int(R_s/2)
    c_h = int(C_s/2)
    chipR = 800
    chipC = 1200

    mask = np.array([data[i*R_b:(i+1)*R_b,j*C_b:(j+1)*C_b] for (i,j) in np.ndindex(chiptype.yBlocks, chiptype.xBlocks)])
    data_tn = np.array([mask[i, int(R_b/2)-r_h:int(R_b/2)+r_h, int(C_b/2)-c_h:int(C_b/2)+c_h] for i in range(mask.shape[0])])      
    data_tn = data_tn.reshape(chiptype.yBlocks, chiptype.xBlocks, R_s, C_s).swapaxes(1,2).reshape(chipR, chipC)  

    return data_tn

def get_spa_generic( data, chiptype=None ):
    ''' Generic spatial thumbnail creator. Not guaranteed to be accurate! '''
    print( 'WARNING! Creating spatial thumbnail from full chip, but thumbnail not guaranteed to be accurate ' )
    #if chiptype is None:
    #    chiptype = ctm.guess_chiptype( data )
    #    print( 'WARNING! Chip type guessed to be {}'.format( chiptype.name ) )

    chipR = 800
    chipC = 1200

    b = np.linspace(0.0, np.shape(data)[0], chipR, endpoint=False)
    a = np.linspace(0.0, np.shape(data)[1], chipC, endpoint=False)
    coord_C = np.array([int(round(A)) for A in a])
    coord_R = np.array([int(round(B)) for B in b])

    data_spa = np.zeros((chipR, chipC))
    for index_R, i in enumerate(coord_R):
        for index_C, j in enumerate(coord_C):
            data_spa[index_R, index_C] = data[i, j]
    return data_spa

def match_helper( dirname, outcsv ):
    ''' Function to help the user determine how a thumbnail is construction, relative to full chip 
    The key output are the Block Thumbnail Origin columns '''

    def process_block( R, C ):
        # Load the data
        blockname = 'X%i_Y%i' % ( ct.blockC*C, ct.blockR*R )
        block = DatFile( filename % blockname, norm=False )

        # Find the uncompressed region
        first = np.where( np.diff( block.inds ) == 1 )[0][0]
        last  = np.where( np.diff( block.inds ) == 1 )[0][-1]
        
        br = block.rows/2
        bc = block.cols/2

        # PART 1: Match the center of the block to the thumbnail
        # get center:
        center = block.data[ br, bc, first:last ]
        tn_window = tn.data[:,:,block.inds[first]:block.inds[last]]
        # Match the data
        match = ( tn_window == center ).all( axis=2 )

        found_center = False
        if match.sum() == 0:
            msg = 'Could not find a match'
            out = [[0],[0]]
        elif match.sum() > 1:
            msg = 'Multiple locations found'
            out = [[0],[0]]
        else:
            out = np.where( match )
            msg = '(%4i, %4i) -> (%3i, %4i)' % ( br, bc, out[0][0], out[1][0] )
            found_center = True
        
        # PART 2: Match the origin
        found_origin = False
        if found_center:
            origin_r = br - out[0][0] % 100
            origin_c = bc - out[1][0] % 100
            origin = block.data[ origin_r, origin_c, first:last ]
            or_tn_r = R*100
            or_tn_c = C*100
            if ( tn_window[or_tn_r,or_tn_c,:] == origin ).all():
                found_origin = True
        if found_origin:
            msg += ' | (%4i, %4i) -> (%3i, %4i)' % ( origin_r, origin_c, or_tn_r, or_tn_c )
        else:
            msg += ' | ERROR Finding origin'
            origin_r = origin_c = or_tn_r = or_tn_c = 0

        # PART 3: Match opposite corner
        found_opposite = False
        if found_origin:
            opposite_r = origin_r + 99
            opposite_c = origin_c + 99
            opposite = block.data[ opposite_r, opposite_c, first:last ]
            op_tn_r = or_tn_r + 99
            op_tn_c = or_tn_c + 99
            if ( tn_window[op_tn_r,op_tn_c,:] == opposite ).all():
                found_opposite = True
        if found_opposite:
            msg += ' | (%4i, %4i) -> (%3i, %4i)' % ( opposite_r, opposite_c, op_tn_r, op_tn_c )
        else:
            msg += ' | ERROR Finding opposite'
            opposite_r = opposite_c = op_tn_r = op_tn_c = 0

        print '(%1i, %2i) %s' % ( R, C, msg )
        return ( br, bc, out[0][0], out[1][0], origin_r, origin_c, or_tn_r, or_tn_c, opposite_r, opposite_c, op_tn_r, op_tn_c )
        
    filename = dirname + '/%s/W1_step.dat'
    ct = chiptype.get_ct_from_dir( dirname )
    print 'Found Chiptype: %s' % chiptype.name
    tn = DatFile( filename % 'thumbnail', norm=False )

    with open( outcsv, 'w' ) as f:
        f.write( 'Block-R, Block-C, Block Center R, Block Center C, Thumbnail R, Thumbnail C, Block Thumbnail Origin R, Block Thumbnail Origin C, Thumbnail Origin R, Thumbnail Origin C, Block End R, Block End C, Thumbnail End R, Thumbnail End C\n' )
        for R in range( ct.chipR / ct.blockR ):
            for C in range( ct.chipC / ct.blockC ):
                row = process_block( R, C )
                f.write( '%1i, %2i, %4i, %4i, %3i, %4i, %4i, %4i, %3i, %4i, %4i, %4i, %3i, %4i\n' % ( R, C, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11] ) )


