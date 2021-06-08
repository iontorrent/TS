''' 
Tools which do not readily fall into other categories
Please don't let this file get too big.
If enough functions show up here, you can probably group them together.
For backwards compatibility, when you move a function out, please
   include an import statement in this module so that it is still accessible
'''
import numpy as np
import xml.etree.ElementTree as ET

def flatten( data, metric=None ):
    ''' 
    Flatten data removing outliers
    Outliers are defined by the metric name
    '''
    keep = np.ones( data.shape, dtype=bool )
    if metric in [ 'buffering', 'bf_std', 'qsnr', 'gaincorr', 'gain_iqr', 'phslopes', 'buffering_gc', 'buffering_gsc', 'rcnoise' ]:
        keep *= data > 0
    if metric in [ 'ebfvals' ] :
        keep *= data != 0
    if metric in [ 'buffering', 'buffering_gc', 'buffering_gsc' ]:
        # Remove extreemly high buffering values which tend to throw off the mean
        keep *= data < 1000
    if metric in [ 'scales' ]:
        # Remove extreemly high buffering values which tend to throw off the mean
        keep *= data < -100 
    # Remove NaN and Inf
    keep *= ~np.isnan( data ) 
    keep *= ~np.isinf( data )
    flat = data[ keep ]
    return flat

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

def parse_xml( text ):
    ''' Converts the xml output to grouped fields.
        This adds an outer set of tags if necessary '''
    try:
        return ET.fromstring   ( text )
    except ET.ParseError:    # This fixes the "junk after ..." commands.  Maybe this should be the default method [STP 12/16/2014]
        text = '<data>\n' + text + '</data>'
        return ET.fromstring   ( text )

def flatten_dict( data, root='' ):
    ''' Flatten nested dictionary keys 
    This can convert results.json to a flat dictionary which can be used in the models above
    '''
    if not isinstance( data, dict ):
        root = root.replace( '>=', '_gte_' )
        root = root.replace( '<=', '_lte_' )
        root = root.replace( '>', '_gt_' )
        root = root.replace( '<', '_lt_' )
        root = root.replace( '-', '_' )
        root = root.replace( '/', '_' )
        root = root.replace( '%', '_percent' )
        root = root.replace( ' ', '_' )
        output = { root: data }
        return output
    output = {}
    for d in data:
        base = root
        if root:
            base += '_'
        base += d

        output.update( flatten_dict( data[d], base ) )
    if not root:
        for o in output:
            if o[0].isdigit():
                output['n'+o] = output.pop( o )
    return output

#########################################
# Compatibility                         #
#########################################
flatten_data = flatten
from .json2 import serialize, JSONEncoder
