'''
Module for parsing chip types 
'''

import os
import re
import subprocess
import copy 
from .chipinfo import params, aliases

#TODO:  Need to incorperate by-chip functions
#           TrimDat                (eccTools.py)
#           analyze_cal_re         (cal.py)
#           analyze_cal_noise_data (cal.py)
#           

moduleDir = os.path.abspath( os.path.dirname( __file__ ) )

# Thise are old deffinitions left for compatibility.  Do not add anything more here.  Additional fields will be parsed from chiptypes
proton_chips = ['900' , 'P0' , 'P1' , 'P2', 'R0' ] 
pgm_chips    = [ '314' , '314V2' , '314R' , '316D' , '316E' , '316V2' , '318B' , '318C' , '318G' ] 

# Add the chip types from the table
proton_chips = [ t['type'] for t in params if t['series'] == 'proton' ]
pgm_chips    = [ t['type'] for t in params if t['series'] == 'pgm' ]
# Add old defaults
proton_chips += ['900' , 'P0' , 'P1' , 'P2', 'R0' ] 
pgm_chips    += [ '314' , '314V2' , '314R' , '316D' , '316E' , '316V2' , '318B' , '318C' , '318G' ] 

# Regex paterns
seq_block = re.compile( r'block_X[0-9]+_Y[0-9]+' )
ecc_block = re.compile( r'X[0-9]+_Y[0-9]+' )

class ChipType(object):
    '''
    Class to store variables for each chip type.
    Chip type is reported by _________________
    '''
    def __new__( cls, *args, **kwargs ):
        ''' Detect if input is chip type and return it if so.
        NOTE: __init__ will still run, so make sure it knows to exit
        '''
        try:
            if isinstance( args[0], ChipType ):
                return copy.deepcopy( args[0] )
        except IndexError:
            try:
                if isinstance( kwargs['name'], ChipType ):
                    print( 'existing instance by name' )
                    return copy.deepcopy( kwargs['name'] )
            except KeyError:
                pass
        # OK, just to make sure, let's do it by name too
        try:
            if 'ChipType' in args[0].__class__.__name__:
                print( 'existing instance' )
                return copy.deepcopy( args[0] )
        except IndexError:
            try:
                if 'ChipType' in kwargs['name'].__class__.__name__:
                    print( 'existing instance by name' )
                    return copy.deepcopy( kwargs['name'] )
            except KeyError:
                pass
        return super( ChipType, cls ).__new__( cls ) # python3: cannot pass args, kwargs

    def __init__( self, name=None, record=None, blockdir='/dev/null', tn=False, _norecurse=False ):
        '''
        Sets the chip type variables.  You can set the chiptype by either name or record but not both.
        If neither method is specified dummy structure will be built from the existing block directories in blockdirs
        Name      : Name is the type of chip (e.g. P1.0.20)
        record    : A param record
        blockdirs : parrent directory for block directories.  This is used if multiple chip types have
                    the same name.  If only one record is present, this is not used.  Also this filtering
                    is turned of off if blockdirs is set to None
        tn        : Set if this is a thumbnail.  Values:
                        True/'tn':  Data is a traditional thumbnail
                        'spa'    :  Data is a spatial thumbnail
        '''
        # Check if __new__ caught this as an existing instance.  If so, we don't need to init
        if isinstance( name, ChipType ):
            return

        self.blockdir = blockdir

        if name is not None and record is None:
            # Chip type specified by name
            # Default chip identification.  Do not appy custom analysis here
            records = [ record for record in params if record['name'] == str(name) ]
            if not records:
                records = [ record for record in params if record['name'] == aliases[str(name)] ]
            #records = self._sort_by_block_size( records )
            records = self._check_for_minimum_block( records )

            # APPLY FURTHER CUSTOM FILTERING HERE.  
            # All filtering should still output a list of records, even if there is only 1 record

        elif record is not None and name is None:
            # Chip type specifed by record
            records = [ record ]

        elif record is None and name is None:
            # Create a dummy chiptype from the existing block data
            self._make_dummy_block()
            return

        elif record is not None and name is not None:
            # too many methods used
            raise ValueError( 'You can either specify chip type by record or name; not both.' )

        else:
            raise NotImplementedError( 'Unknown chip type entry method' )

        # Alert if couldn't find a unique result
        if len( records ) == 0:
            raise ValueError( 'Could not identify chip type "%s"' % name )

        if len( records ) >= 2:
            if not _norecurse:
                try:
                    returned = get_ct_from_dir( self.blockdir, records=records )    # Errors below here might mean you matched a block but that block was already filtered by name
                    static = dir( ChipType )
                    for a in dir(returned):
                        # Whatever filtering is applied here, it must not copy FUNCTIONS or else calls to those functions will reference "returned", not self
                        if a not in static:
                            setattr( self, a, getattr( returned, a ) )
                    return
                except IndexError:  # No blocks found
                    print( "WARNING! Unable to uniquly identify '%s' from the blockdir (%s). Using last entry in the table" % ( name, self.blockdir ) )
                    pass
            else:
                print( "WARNING! Could not uniquly identify '%s'.  Using last entry in table" % name )

        # Reduce to 1 record
        record = records[-1]

        # Apply thumbnail
        if tn:
            record = self._apply_tn( record, tn )
        # Apply burger mode if applicable
        record = self._apply_burger( record )

        # Apply chip properties
        for key in record:
            setattr( self, key, record[key] )

        # Set multilane properties
        self.reset_lanes()

    def _apply_tn( self, record, tntype='tn' ):
        '''
        Get's the appropriate thumbnail metric for the specified chip
        record:   The corresponding full chip record
        flag  :   'tn'  for standard thumbnail
                  'spa' for spatial thumbnail
        '''
        # Get the thumbnail record
        if tntype == 'spa':
            records = [ r for r in params if r['name'] == record['spatn'] ]
        elif tntype == 'tn' or tntype:
            records = [ r for r in params if r['name'] == record['tn'] ]
        else:
            raise ValueError( 'Unknown thumbnail flag %s' % tntype )

        # Make sure it was well found
        if len(records) == 0:
            raise NotImplementedError( 'No thumbnail for specified chip type' )
        elif len( records ) > 1:
            raise ValueError( "UhOh! You shouldn't be here!" )
        # Isolate the record
        tn_record = records[0].copy()

        # Replace None's with chip-specific params
        for key in tn_record:
            if tn_record[key] is None:
                tn_record[key] = record[key]
        tn_record['name'] = record['name']

        # Create a chiptype instance for the full chip and attach it into the thumbnail
        tn_record['fullchip'] = ChipType( record=record )

        return tn_record

    def _apply_burger( self, record ):
        ''' Reads the burger field and replaces it with a chiptype instance of the corresponding chip '''
        if record.get( 'burger', None ):
            # Isolate the record so we don't overwrite the main table
            record = record.copy()
            record['burger'] = ChipType( record['burger'] )
        return record
            
    def _sort_by_block_size( self, records ):
        '''
        Sorts the records by the total number of wells in a block
        '''
        return sorted( records, key=lambda r: r['blockR']*r['blockC'] )

    def _check_for_minimum_block( self, records ):
        '''
        Gets the block name for the minimum block and checks if it is present in the block parrent directory
        '''
        # Skip if only one record found
        if len( records ) == 1:
            return records
        
        # Skip if no directory is specified
        if self.blockdir is None:
            return records

        records = self._sort_by_block_size( records )

        # Get list of blocks files in self.blockdir
        dirlist = listdir( self.blockdir )
        for record in records:
            # Check for block name varients
            seq_blockname = 'block_X%s_Y%s' % ( record['blockR'], record['blockC'] )  # Proton run
            ecc_blockname = 'X%s_Y%s' % ( record['blockR'], record['blockC'] ) # ECC server
            if seq_blockname in dirlist or ecc_blockname in dirlist:
                return [ record ]
        # OK couldn't find it.  return for further filtering
        return records

    def _make_dummy_block( self ):
        ''' Makes a dummy chip from the block directories in self.blockdir '''
        # Read the existing blocks
        self.read_blocks()
        if not len( self.blocknames ):
            raise IOError( 'No blocks were found in %s' % self.blockdir )

        # Get the XY coordinates for each block, assuming blocks are regularly spaced
        y_vals = sorted( list ( set( [ int( bn.split('_')[-1][1:] ) for bn in self.blocknames ] ) ) )
        x_vals = sorted( list ( set( [ int( bn.split('_')[-2][1:] ) for bn in self.blocknames ] ) ) )
        
        # Get block sizes
        self.blockR = y_vals[1] - y_vals[0]
        self.blockC = x_vals[1] - x_vals[0]

        # Get block counts
        self.xBlocks = len( x_vals )
        self.yBlocks = len( y_vals )

        # Get chip counts
        self.chipR = self.yBlocks * self.blockR
        self.chipC = self.xBlocks * self.blockC

        # Get miniblocks
        self.miniR = self._multiple( self.blockR, (10, 200), 80 )
        self.miniC = self._multiple( self.blockC, (10, 200), 80 )

        # Get microblocs
        self.microR = self._multiple( self.blockR, (4, 30), 10 )
        self.microC = self._multiple( self.blockC, (4, 30), 10 )

        # Set name fields
        self.name = ''
        self.type = ''
        
    def _multiple( self, numerator, target_lims, target ):
        ''' Calculates an integer multiple close to target '''
        multiples = [ m for m in range( target_lims[0], target_lims[1] ) if not numerator%m ]
        if len( multiples ) == 0:
            print( 'Could not identify a multiple in range.  Using default target' )
            return target
        diffs = [ abs(target-m) for m in multiples ]
        return multiples[ diffs.index( min( diffs ) ) ]

    def read_blocks( self ):
        ''' 
        Reads the block names in the folder self.blockdir
        Saves to self.blocknames
        Applies sorting based on self.sort_blocknames
        '''
        # Get list of blocks files in self.blockdir
        dirlist = listdir( self.blockdir )

        # Look for block names.  hopefully only one of these patterns is present
        self.blocknames  = [ d for d in dirlist if seq_block.match( d ) ]
        self.blocknames += [ d for d in dirlist if ecc_block.match( d ) ]
        
        self.sort_blocknames()

        return self.blocknames

    def reset_lanes( self ):
        self.is_multilane = False
        self.lanes = [ False ] * 4

    def set_lanes( self, lanemask ):
        ''' Add multilane information into the chiptype.  
            Only call this function on a multilane chip 
            lanemask is a boolean array
            '''
        self.is_multilane = True
        self.lanes = lanemask

    def sort_blocknames( self ):
        ''' 
        sorts the values in self.blocknams in place
        Also, self.sorted_blocknames is generated as a 2D list as follows [[row],[row],[row],...]
        '''
        # Sort by Y value
        self.blocknames = sorted( self.blocknames, key=lambda bn: int( bn.split('_')[-1][1:] ) )   # this will handle ECC and Seq block names
        # Sort by X value
        self.blocknames = sorted( self.blocknames, key=lambda bn: int( bn.split('_')[-2][1:] ) )

        # Sort to a 2D array
        self.sorted_blocknames = []
        last_y = -1
        y_vals = [ int( bn.split('_')[-1][1:] ) for bn in self.blocknames ]
        for bn in self.blocknames:
            this_y = int( bn.split('_')[-1][1:] )
            if last_y == this_y:
                self.sorted_blocknames[-1].append( bn )
            else:
                last_y = this_y
                self.sorted_blocknames.append( [ bn ] )

    def make_blocknames( self, style='seq' ):
        '''
        Make blocknames based on the chip type
        style sets the output format:
          'ecc' : X#_Y#
          'seq' : block_X#_Y#
        '''
        if style == 'seq':
            prefix = 'block_'
        elif style == 'ecc':
            prefix = ''
        else:
            raise ValueError( 'Unknown style: %s' % style )

        self.blocknames = []
        for x in range( 0, self.chipC, self.blockC ):
            for y in range( 0, self.chipR, self.blockR ):
                self.blocknames.append( '%sX%s_Y%s' % ( prefix, x, y ) )
        self.sort_blocknames()

    def get_from_rc( self ):
        '''
        gets the chip type from block properties
        '''
        self = get_ct_from_dir( self.blockdir )

    def transposed( self ):
        ''' Returns the transposed chiptype '''
        newchip = ChipType( self )
        newchip.transpose = not( self.transpose )

        swap = ( ( 'chipR', 'chipC' ), 
                 ( 'blockR', 'blockC' ), 
                 ( 'miniR', 'miniC' ), 
                 ( 'microR', 'microC' ), 
                 ( 'ref_rows', 'ref_cols' ), 
                 ( 'xBlocks', 'yBlocks' ), 
                 ( 'RBlocks', 'CBlocks' ), 
                 ( 'miniRblocks', 'miniCblocks' ), 
                 ( 'blockminiRblocks', 'blockminiCblocks' ), 
                 ( 'microRblocks', 'microCblocks' ), 
                 ( 'blockmicroRblocks', 'blockmicroCblocks' ) 
                 )

        for rc in swap:
            try:
                setattr( newchip, rc[0], getattr( self, rc[1] ) )
                setattr( newchip, rc[1], getattr( self, rc[0] ) )
            except AttributeError:
                pass

        return newchip

def get_ct_from_dir( blockdir, records=None, tn=False ):
    '''
    Function to guess the chip type from the block parameters
    returns a chiptype
    '''
    # Get list of blocks files in self.blockdir
    dirlist = listdir( blockdir )

    # Look for block names.  hopefully only one of these patterns is present
    blocknames  = [ d for d in dirlist if seq_block.match( d ) ]
    blocknames += [ d for d in dirlist if ecc_block.match( d ) ]

    # Get the XY coordinates
    y_vals = sorted( list ( set( [ int( bn.split('_')[-1][1:] ) for bn in blocknames ] ) ) )
    x_vals = sorted( list ( set( [ int( bn.split('_')[-2][1:] ) for bn in blocknames ] ) ) )

    return get_ct_from_rc( y_vals[1], x_vals[1], blockdir=blockdir, records=records, tn=tn )

def get_ct_from_rc( rows, cols, blockdir='.', records=None, tn=False ):
    '''
    Function to guess the chip type from the block parameters
    returns a chiptype
    '''
    if records is None:
        records = params

    # Get the appropriate record by looking for the first non-zero block
    records = [ r for r in records if r['blockR'] == rows and r['blockC'] == cols ]

    # This is mostly to handle thumbnails
    if not len( records ):
        records = [ r for r in params if r['chipR'] == rows and r['chipC'] == cols ]

    if len( records ) == 0:
        raise ValueError( 'Unable to find a matching block for X:%s, Y:%s' % ( cols, rows ) )
    elif len( records ) > 1:
        #print 'Unable to find a UNIQUE matching block for X:%s, Y:%s' % ( cols, rows ) 
        prefered = [ r for r in records if r['prefered'] ]
        if prefered:
            return ChipType( record=prefered[-1], blockdir=blockdir, tn=tn )
        return ChipType( record=records[-1], blockdir=blockdir, tn=tn )
    else:
        prefered = [ r for r in records if r['prefered'] ]
        if prefered:
            return ChipType( record=prefered[-1], blockdir=blockdir, tn=tn )
        return ChipType( record=records[-1], blockdir=blockdir, tn=tn )

def get_ct_from_chip():
    '''
    Calls command control to get the chip type
    '''
    cmd = ['cmdControl','query','ExpInfo']
    p = subprocess.Popen( cmd, stdout = subprocess.PIPE )
    output = p.communicate()[0]
    output = output.split('\n')
    for line in output:
        if 'TsChipType' in line:
            match = re.search( r'<value>.*</value>', line ).group()
            chip = match[6:-8].strip()
            return ChipType( name=chip )
    raise ValueError( 'Could not parse chip type from cmdControl' )

def block_rc( blockdir ):
    '''
    Extracts the starting row and column from the specified block
    '''
    try:
        if seq_block.match( blockdir ):
            parts = blockdir.split('_')
            col   = int( parts[1][1:] )
            row   = int( parts[2][1:] )
            return ( row, col )
        elif ecc_block.match( blockdir ):
            parts = blockdir.split('_')
            col   = int( parts[0][1:] )
            row   = int( parts[1][1:] )
            return ( row, col )
    except: 
        pass

    print( 'Unable to parse block name (%s)' % blockdir )
    return ( None, None )

def listdir( dirname ):
    try:
        return os.listdir( dirname )
    except OSError:
        return []

def guess_chiptype( data ):
    ''' guesses the chip type from the input data size '''
    for param in params:
        sizes = [ [ param['chipR']*param['chipC'], 1 ], 
                  [ param['chipR'], param['chipC'] ],
                  [ param['blockR']*param['blockC'], 1 ],
                  [ param['blockR'], param['blockC'] ] ]
        for possiblesize in sizes:
            try:
                data.reshape( sizes )
            except ValueError:
                continue
            break
        return ChipType( record=param )
    raise ValueError( 'Could not guess chip type from input data size' )

def validate_block_sizes():
    ''' Loads all chip types and verifies that block sizes are evenly divisible '''

    def status( name, num, den ):
        msg = '  ' + name + ':' + ' '*(7-len(name) )
        try:
            if num % den:
                msg += 'FAIL'
            else:
                msg += 'PASS'
        except:
            msg += '(none)'
        print( msg )
        return 'FAIL' not in msg

    fails = []
    for record in params:
        print( record['name'] )
        rc = []
        rc.append( status( 'blockR' , record['chipR']  , record['blockR'] ) )
        rc.append( status( 'blockC' , record['chipC']  , record['blockC'] ) )
        rc.append( status( 'miniR'  , record['blockR'] , record['miniR']  ) )
        rc.append( status( 'miniC'  , record['blockC'] , record['miniC']  ) )
        rc.append( status( 'microR' , record['blockR'] , record['microR'] ) )
        rc.append( status( 'microC' , record['blockC'] , record['microC'] ) )
        if not all( rc ):
            fails.append( record['name'] )

    if fails:
        print( 'ERRORS in the following chips:' ) 
        for f in fails:
            print( '  ' + f )
    else:
        print( '\nALL CHIPS PASS' )

def validate_chiptype( chiptype, blockdir=None, rc=None ):
    ''' Provides a method to validate the provided chip type against
        either the underlying directory structure or a tuple (rows, cols) '''
    try:
        read_chiptype = get_ct_from_dir( blockdir )
    except: # there are no blocks, or blockdir is None
        try:
            read_chiptype = get_ct_from_rc( rc[0], rc[1] )
        except:
            raise ValueError( 'unable to determine chiptype from provided info' )
    if chiptype.chipR == read_chiptype.chipR and chiptype.chipC == read_chiptype.chipC:
        ''' Chip types are close enough to allow most code to continue '''
        return chiptype
    return read_chiptype


