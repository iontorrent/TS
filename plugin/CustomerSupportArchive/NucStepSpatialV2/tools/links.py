# Links are represented by a 24-bit word
# when autocal reads link status from explog, we see:
# 0x<******>00<******> -- the first group is cdrStat and second is rxtStat
# 1's mean the links are up.
import re
import numpy as np

class Links:
    """ Class to deal with link errors, both regionslips and linklosses.  Be careful not to mix."""
    def __init__( self , initial=0xffffff ):
        self.bits        = 24
        self.state       = self.hex_to_array( initial )
        self.bad_links   = ~self.state
        self.fail_count  = np.zeros( self.bits , int )
        self.first_fails = 24*['']
        
    def add_fail( self , hex_num , flow_name ):
        # Interpret hex
        new_state  = self.hex_to_array( hex_num )
        new_bad    = ~new_state
        
        # update state and fail count
        self.state[new_bad]       = False
        self.bad_links            = ~self.state
        self.fail_count[new_bad] += 1
        
        # record new first_fail.  If this is done during active parsing, only write if it's ''.
        for idx in np.where( new_bad )[0]:
            if self.first_fails[idx] == '':
                self.first_fails[idx] = flow_name
                
    def show_state( self ):
        print('{}:\t{}'.format( self.array_to_hex( self.state ) , self.array_to_binary( self.state )))
        print('Bad links:\t' + ', '.join( [ str(x) for x in np.where( self.bad_links )[0] ] ) )
        
    def calc_metrics( self ):
        """ 
        Calculates metrics of interest.  They should be flattened for database migration purposes.
        autoCal plugin will leave them as nested dictionaries.  On db, we can flatten from scraperdb.models.
        """
        metrics = { 'summary_state' : self.array_to_hex( self.state ) ,
                    'summary_fails' : [ x for x in np.where( self.bad_links)[0] ] ,
                    'link'          : {} }
        for i in range(self.bits):
            metrics['link'][str(i)] = { 'first' : self.first_fails[i] , 'instances' : self.fail_count[i] }
            
        self.metrics = metrics
        
    @classmethod
    def from_misc( cls, misc_dict, regionslips=True ):
        """ 
        This method aims to help with parsing of misc fields in the normalduck database to
        get data on synclink fails.  Assumes misc_dict is a dict, e.g. ( json.loads( x.misc ) ).
        """
        # Filesorter blatantly stolen from Scott.
        def filesorter( filename ):
            section = '_'.join(filename.split( '_' )[:-1])
            index   = int( filename.split( '_' )[-1] )
            sections = [ 'beadfind_pre', 'prerun', 'extraG', 'acq' ]
            try:
                sid = sections.index( section )
            except:
                sid = -1
            return sid*10000+index
        
        rs   = re.compile( 'regionslips_([a-f0-9]*)_instances' )
        ll   = re.compile( 'linklosses_([a-f0-9]{6})00([a-f0-9]*)_instances' )
        data = []
        
        # Pick out fails from dictionary
        if regionslips:
            print( 'Analyzing regionslips from misc field . . .' )
            for k in misc_dict:
                m = rs.match( k )
                if m:
                    flow            = misc_dict['regionslips_{}_first'.format( m.group(1) ) ]
                    hex_count_first = ( m.group(1), int(misc_dict[k]), flow )
                    data.append( hex_count_first )
        else:
            print( 'Analyzing linklosses from misc field . . .' )
            for k in misc_dict:
                m = ll.match( k )
                if m:
                    flow            = misc_dict['linklosses_{}00{}_first'.format( m.group(1) , m.group(2) ) ]
                    hex_count_first = ( m.group(2), int(misc_dict[k]), flow )
                    data.append( hex_count_first )
                    
        links = cls( )
        for (hex_num, count, flow_name) in data:
            print( hex_num, count, flow_name )
            # Interpret hex
            new_state  = links.hex_to_array( hex_num )
            new_bad    = ~new_state
            
            # update state and fail count
            links.state[new_bad]       = False
            links.bad_links            = ~links.state
            links.fail_count[new_bad] += count
            
            # record new first_fail.  This requires some sorting and ingenuity.
            for idx in np.where( new_bad )[0]:
                first_fail = links.first_fails[idx]
                if first_fail == '':
                    links.first_fails[idx] = flow_name
                else:
                    # We have to sort and pick earliest.
                    links.first_fails[idx] = sorted([ first_fail , flow_name ], key=filesorter )[0]
                    
        return links
    
    @classmethod
    def from_nested_dict( cls, nested_dict, regionslips=True ):
        """ 
        This method aims to help with parsing of misc fields in the normalduck database to
        get data on synclink fails.  Assumes nested_dict is a nested dict, from a pure dictionary
        just like what would come from autoCal plugin or our explog reading code.
        """
        # Filesorter blatantly stolen from Scott.
        def filesorter( filename ):
            section = '_'.join(filename.split( '_' )[:-1])
            index   = int( filename.split( '_' )[-1] )
            sections = [ 'beadfind_pre', 'prerun', 'extraG', 'acq' ]
            try:
                sid = sections.index( section )
            except:
                sid = -1
            return sid*10000+index
        
        data = []
        
        # Pick out fails from dictionary
        if regionslips:
            print( 'Analyzing regionslips . . .' )
            rs_data = nested_dict
            for k in [key for key in rs_data if key != 'total']:
                hex_num         = str(k)
                instances       = rs_data[k]['instances']
                flow            = rs_data[k]['first']
                hex_count_first = ( hex_num, instances, flow )
                data.append( hex_count_first )
        else:
            print( 'Analyzing linklosses . . .' )
            ll_data = nested_dict
            for k in [key for key in ll_data if key != 'total']:
                hex_num         = str(k[-6:])
                instances       = ll_data[k]['instances']
                flow            = ll_data[k]['first']
                hex_count_first = ( hex_num, instances, flow )
                data.append( hex_count_first )
                
        links = cls( )
        for (hex_num, count, flow_name) in data:
            print( hex_num, count, flow_name )
            # Interpret hex
            new_state  = links.hex_to_array( hex_num )
            new_bad    = ~new_state
            
            # update state and fail count
            links.state[new_bad]       = False
            links.bad_links            = ~links.state
            links.fail_count[new_bad] += count
            
            # record new first_fail.  This requires some sorting and ingenuity.
            for idx in np.where( new_bad )[0]:
                first_fail = links.first_fails[idx]
                if first_fail == '':
                    links.first_fails[idx] = flow_name
                else:
                    # We have to sort and pick earliest.
                    links.first_fails[idx] = sorted([ first_fail , flow_name ], key=filesorter )[0]
                    
        return links
            
    @staticmethod
    def int_to_array( num24 ):
        return np.array( [ bool(int(x)) for x in list( np.binary_repr( num24 , 24 ) ) ], dtype=bool)
    
    @staticmethod
    def hex_to_array( hex_num ):
        # We need the int statement to be agnostic to whether hex_num is 0xffffff or 'ffffff'
        try:
            integer = int( hex_num )
        except ValueError:
            integer = int( hex_num , 16 )
        # Note that the LSB is link 0.  This was backwards previously.
        arr = np.array( [ bool(int(x)) for x in list( np.binary_repr( integer , 24 ) ) ], dtype=bool)
        return arr[::-1]
    
    @staticmethod
    def array_to_binary( arr ):
        return ''.join( [ str(int(x)) for x in arr[::-1] ] )
    
    @staticmethod
    def array_to_int( arr ):
        return int( ''.join( [ str(int(x)) for x in arr[::-1] ] ) , 2 )
    
    @staticmethod
    def array_to_hex( arr ):
        return hex( int( ''.join( [ str(int(x)) for x in arr[::-1] ] ) , 2 ) )

    
