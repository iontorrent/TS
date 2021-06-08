"""
Module for multilane-friendly operations and iterations
"""

def interpret_lanes( *lanes ):
    """
    Helper function to interpret inputs for lanes argument.
    Input can either be:
    - A list of integers or floats in the range [1,2,3,4] or arbitrary length (other numbers will be ignored)
    - A list of precisely four booleans corresponding to lanes 1, 2, 3, and 4.
    
    Other inputs will be ignored!
    
    This function could return an empty list without exception.
    """
    if lanes:
        types = set( [type(l) for l in lanes ] )
        if len(types) > 1:
            raise TypeError( '[iter_lanes] You cannot mix types for input to this function!' )
        
        t = types.pop()
        if t in [int, float]:
            # Then we are asking for given lanes.  Values must be from 1-4.
            valid_lanes = list(set( [ x for x in lanes if x in [1,2,3,4] ] ))
            invalid     = list(set( [ y for y in lanes if y not in [1,2,3,4] ] ))
            if invalid:
                print( 'WARNING!  You are asking for invalid lane numbers [{}]!  Please use 1-4.'.format( invalid ) )
                
        elif t == bool:
            # We were given a list of booleans for each lane.  This only works for a list of 4 (or more)
            # Will raise an index error
            if len(lanes) > 4:
                raise IndexError( 'Too many boolean inputs given!  We do not dare interpret the meaning of this . . .' )
            
            try:
                valid_lanes  = [ (i+1) for i in range(4) if lanes[i] ]
            except IndexError:
                raise IndexError( 'Insufficient boolean inputs given!  4 are required!' )
        else:
            raise TypeError( "Invalid lanes input [{}].  Skipping iteration.".format( lanes ) )
        
        return valid_lanes
            
def iter_lanes( *lanes ):
    """
    Base iterator for running through different lanes. 
    Lanes can be a list/array with length of 4 booleans or a list of integers in the range 1-4.
    """
    try:
        valid_lanes = interpret_lanes( *lanes )
    except (TypeError, IndexError) as E:
        print( str(E) )
        raise StopIteration
        
    if not valid_lanes:
        raise StopIteration
    
    for i in valid_lanes:
        yield i, 'lane_{}'.format(i)

def iter_lane_data( data, *lanes ):
    """
    Next level lane iterator to yield array data by lane for given lanes.
    """
    try:
        valid_lanes = interpret_lanes( *lanes )
    except (TypeError, IndexError) as E:
        print( str(E) )
        raise StopIteration
    
    if not valid_lanes:
        raise StopIteration
    
    # Interpret lane size from data array
    lane_width = data.shape[1] / 4 # Note this could truncate in some superpixel arrays.
    
    if data.ndim == 2:
        for i in valid_lanes:
            name = 'lane_{}'.format( i )
            cs = slice( lane_width*(i-1), lane_width*i )
            yield i, name, data[:,cs]
    elif data.ndim == 3:
        for i in valid_lanes:
            name = 'lane_{}'.format( i )
            cs = slice( lane_width*(i-1), lane_width*i )
            yield i, name, data[:,cs,:]
    else:
        print( 'WHAT IS GOING ON?  YOU INPUT AN ARRAY OF {} DIMENSIONS??'.format( data.ndim ) )
        raise StopIteration

def iter_lane_slice( data, *lanes ):
    """
    Next level lane iterator to yield column slice for given lanes.
    """
    try:
        valid_lanes = interpret_lanes( *lanes )
    except (TypeError, IndexError) as E:
        print( str(E) )
        raise StopIteration
    
    if not valid_lanes:
        raise StopIteration
    
    # Interpret lane size from data array
    lane_width = data.shape[1] / 4 # Note this could truncate in some superpixel arrays.
    for i in valid_lanes:
        name = 'lane_{}'.format( i )
        cs = slice( lane_width*(i-1), lane_width*i )
        yield i, name, cs

def iter_masked_lane_data( data, mask, *lanes ):
    """
    Next level lane iterator to yield array data by lane for given lanes.
    """
    try:
        valid_lanes = interpret_lanes( *lanes )
    except (TypeError, IndexError) as E:
        print( str(E) )
        raise StopIteration
    
    if not valid_lanes:
        raise StopIteration
    
    # Interpret lane size from data array
    lane_width = data.shape[1] / 4 # Note this could truncate in some superpixel arrays.
    
    if data.ndim == 2:
        for i in valid_lanes:
            name = 'lane_{}'.format( i )
            cs = slice( lane_width*(i-1), lane_width*i )
            yield i, name, data[:,cs], mask[:,cs]
    elif data.ndim == 3:
        for i in valid_lanes:
            name = 'lane_{}'.format( i )
            cs = slice( lane_width*(i-1), lane_width*i )
            yield i, name, data[:,cs,:], mask[:,cs]
    else:
        print( 'WHAT IS GOING ON?  YOU INPUT AN ARRAY OF {} DIMENSIONS??'.format( data.ndim ) )
        raise StopIteration
            
def wrapped_iter_lanes( ):
    """ A demonstration of a likely wrapper that could be put on iter_lane_data in plugins or other functions """
    # This will always iterate through lanes 1, 2, and 3.
    lanes = [1,2,3]
    return iter_lanes( *lanes )

def test( ):
    """
    interpret lanes unit test
    """
    inputs = [ [1,2,3],
               [1,2,3,4],
               [3,4,5,7],
               [1,2,3,4,4,4,3],
               [5],
               [False,True],
               [True,True,False,True],
               [True,True,True,True,True],
               [None],
               []
    ]
    
    for i in inputs:
        print( 'Input = {}'.format( i ) )
        try:
            v = interpret_lanes( *i )
            print( 'Valid Lanes = {}'.format(v) )
        except (TypeError, IndexError) as E:
            print( str( E ) )
        finally:
            print('--------------------\n')
            
            
