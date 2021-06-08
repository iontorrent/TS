from numpy import concatenate, meshgrid, sum
import warnings

def read_profile( filename = 'dats/VFCProfile.txt' ):
    ''' Reads the VFCProfile, returning a dictionary of profiles '''
    profiles = {}
    for line in open( filename ):
        reg = int( line.split(':')[0].split()[-1] )
        profile = [ int(p) for p in line.split(':')[1].split('(')[0].strip().split(',')[:-1] ]
        profiles[reg] = profile
    return profiles

def uncomp_mask( profile, tn=False ):
    ''' 
    Returns a mask indicating which frames are uncompressed 
    if tn is True, the first frame is turned off, since the first two frames are identical for thumbnails
    '''
    keep = [ p == 1 for p in profile ]
    if tn:
        keep[0] = False
    return keep

def expand_uncomp_mask( profile, tn=False, frames=105 ):
    ''' 
    Generates a mask of frames which are uncompressed.
    Returns 105 frame list
    if tn is True, the first frame is turned off, since the first two frames are identical for thumbnails
    '''
    keep = concatenate( [ [ True if p == 1 else False ] * p for p in profile ] )
    # 8/12/16 - 550s seem to have 107 frames on the right side of the chip.
    # Assuming this is an artifact and that they really have 105
    # Trim array if necessary 
    # Convert to bool because contatenating an empty array results in integer arry
    if len(keep) > frames:
        #print 'WARNING:  VFC profile has more frames (%i) than expected (%i).  Trimming to range' % ( len(keep), frames )
        warnings.warn( 'VFC profile has more frames (%i) than expected (%i).  Trimming to range' % ( len(keep), frames ) )
    keep = concatenate( ( keep[:frames], [False] * ( frames-len(keep) ) ) ).astype( bool )   
    if tn:
        keep[0] = False
    return keep

def unify_uncomp_mask( keeps ):
    ''' 
    Removes extra frames from the end so that all registers appear to
    have the same number of uncompressed frames
    Useful before doing a noise calculation
    This runs in place.  Returns number of unified frames
    '''
    uncomp = {}
    for reg in keeps :
        uncomp[reg] = sum(keeps[reg])
    minuncomp = min( [ uncomp[u] for u in uncomp ] )
    if len( set( [ uncomp[u] for u in uncomp ] ) ) == 1:
        return minuncomp
    # Start at the back and remove frames to get to the minimum value
    # TODO: May want to even things out a little more
    for reg in keeps:
        for pos in reversed( range( len(keeps[reg]) ) ):
            if sum( keeps[reg] ) == minuncomp:
                break
            keeps[reg][pos] = False
    return minuncomp

def remove_tns( profiles ):
    ''' Removes any thumbnail registers '''
    keys = profiles.keys()
    for k in keys:
        if k >= 96:
            profiles.pop(k)
    return profiles

def reg_from_rc( r, c, chiptype=None ):
    ''' 
    Returns the register for a set of coordinates,
    given a chip type instance.  
    If no chiptype is given, assumes a thumbnail
    '''
    try:
        # Number of blocks
        rblocks = chiptype.RBlocks
        cblocks = chiptype.CBlocks
        # Block size
        blockR  = chiptype.blockR
        blockC  = chiptype.blockC
    except:
        rblocks = 8
        cblocks = 12
        blockR  = 100
        blockC  = 100

    reg = (r/blockR)*cblocks + (c/blockC)
    return reg

def read_common_frames( filename='dats/VFCProfile.txt' ):
    profile = read_profile( filename )
    remove_tns( profile )

    keeps = expand_uncomp_mask( profile[profile.keys()[0]] )
    for p in profile:
        keeps = [ k & e for k,e in zip( keeps, expand_uncomp_mask( profile[p] ) ) ]
    # Return the full length if all are OK
    if all( keeps ):
        return 0, len(keeps)
    # Funny stuff happens the the first frame.  If the second is compressed, assume the first is too
    if not keeps[1]:
        keeps[0] = False
    # Check if no single frame exists
    if not any( keeps ):
        return None, None

    first = None
    last  = None
    for i, k in enumerate(keeps):
        if k == 1 and first is None:
            first = i
        if k == 0 and last is None and first is not None:
            last = i
    if last is None:
        last = len(keeps)
    return first, last

if __name__ == '__main__':
    profile = read_profile()
    remove_tns( profile )

    keeps = {}
    for p in profile:
        keeps[p] = expand_uncomp_mask( profile[p], tn=True )

    unify_uncomp_mask( keeps )
    
    x,y = meshgrid( range(1200), range(800) )
    reg = reg_from_rc( y, x )

