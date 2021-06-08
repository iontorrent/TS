''' Expanded JSON fuctions '''
# Python 2 and 3 compatible
from json import *
import json as _json
import numpy as _np
import datetime as _dt
import traceback as _tb


class JSONEncoder( _json.JSONEncoder ):
    ''' 
    Generic encoder to serialize complex objects.
    For speed efficiency, you might what to rebuild in your own 
    project, using only the functions you need and not using 
    any additional import statements
    '''
    def default( self, obj ):
        # Datetime objects
        if isinstance( obj, _dt.date ):
            return obj.strftime( '%m/%d/%Y' )
        if isinstance( obj, _dt.datetime ):
            return obj.strftime( '%m/%d/%Y %H:%M:%S' )

        # NaN
        try:
            if obj != obj:
                return None
        except ValueError:
            pass
        try:
            if obj == float( 'inf' ):
                return None
        except ValueError:
            pass
        try:
            if obj == -float( 'inf' ):
                return None
        except ValueError:
            pass

        # All numpy objects
        if 'numpy' in obj.__class__.__module__:
            # Arrays:
            try:
                # Numpy items
                return damn_nans( obj.item() )
            except (AttributeError,ValueError):
                if isinstance( obj, ( _np.ndarray, _np.ma.core.MaskedArray ) ):
                    return damn_nans( tuple( obj ) )
            print( 'to string' )
            return str( obj )

        # Decimal Objects
        if obj.__class__.__name__ == 'Decimal':
            return float( obj )

        # Default encoder
        try:
            return _json.JSONEncoder.default( self, obj )
        except TypeError:
            tb = _tb.format_exc()
            print(tb)    
            return ''
        except ValueError:
            return ''

def serialize( jsondict ):
    ''' 
    serializes the dictionary replacing numpy items with python natives
    this is particularly useful when getting ready 
    to save a json file to disk
    '''
    for key in jsondict:
        try:
            jsondict[key] = jsondict[key].item()
        except AttributeError:
            pass

def damn_nans( obj ):
    ''' Remove NaNs from nested data '''
    if isinstance( obj, dict ):
        output = {}
        for key, val in obj.items():
            key = damn_nans( key )
            val = damn_nans( val )
            output[key] = val
        return output
    if isinstance( obj, (list,tuple) ):
        newobj = [ damn_nans(o) for o in obj ]
        if isinstance( obj, tuple ):
            newobj = tuple( newobj )
        return newobj
    try:
        if obj != obj:
            return None
    except:
        pass
    try:
        if obj == float( 'inf' ):
            return None
    except ValueError:
        pass
    try:
        if obj == -float( 'inf' ):
            return None
    except ValueError:
        pass
    return obj

def dumps( obj ):
    ''' 
    Safely returns a json string using the custom encoder.
    If an error is encountered, it is reported and an empty string is returned
    '''
    obj = damn_nans( obj )
    try:
        return _json.dumps( obj, cls=JSONEncoder, encoding='latin1', allow_nan=False ) # python 2
    except TypeError:
        return _json.dumps( obj, cls=JSONEncoder, allow_nan=False ) # Python 3
    except:
        tb = _tb.format_exc()
        print(tb)    
        return ''

def dump( obj, fo ):
    ''' 
    Safely converts the object to a json string using the custom
    encoder and then writes to file.  This ensures that a complete
    json file will be written
    '''
    text = dumps( obj )
    fo.write( text )


