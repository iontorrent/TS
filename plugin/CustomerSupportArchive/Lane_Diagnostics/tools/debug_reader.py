import datetime, re, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

TODAY       = datetime.datetime.today()
DEBUG_REGEX = re.compile( r"""(?P<file>[\w/.]+):(?P<timestamp>[\w:\s]{15})\s(?P<inst>[\w]+)\s(?P<source>[\w.]+):\s(?P<message>[\w\W\s]+)""" )

"""
Run timing is as follows (debug file used unless otherwise noted:

Start of run:    explog_final.txt: Start Time
<dead time 1>
Review Run Plan: planStatus Review
<dead time 2>
Library start:  planStatus Library Preparation Started
Library end:    planStatus Library Preparation Completed
<dead time 3>
Templ. start:  planStatus Templating Started
Templ. end:    planStatus Templating Completed
<dead time 4>
Seq. start:    planStatus Sequencing Started
Seq. end:      planStatus Sequencing Completed  -- this is identical to explog_final.txt: End Time

After this level of interest, we can dive into how long submodules take.

Note that as the grepping gets more serious, we can save time by only grepping once and 
  using multiple -e arguments.
"""

class DebugLog( object ):
    """ Class for interaction with /var/log/debug. """
    def __init__( self, debug_path ):
        self.path = debug_path
        
    def search( self, grep_phrase, case_sensitive=False ):
        """ 
        Searches the debug file for the input phrase by using grep. 
        Returns lines as they are for further parsing.
        """
        cmd      = 'grep '
        if not case_sensitive:
            cmd += '-i '
            
        cmd     += '"{}" {}'.format( grep_phrase, self.path )
        p        = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True )
        ans, err = p.communicate()
        try:
            return ans.splitlines()
        except AttributeError:
            return []
        
    
    def search_many( self, *strings, **kwargs ):
        """ Searches the file for many grep strings at once. """
        case_sensitive = kwargs.get( 'case_sensitive', False )
        
        cmd      = 'grep '
        if not case_sensitive:
            cmd += '-i '
            
        for string in strings:
            cmd     += '-e "{}" '.format( string )
            
        cmd     += self.path
        p        = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True )
        ans, err = p.communicate()
        try:
            return ans.splitlines()
        except AttributeError:
            return []
        
        
    def read_line( self, line ):
        """ Matches the line pattern and extracts the relevant components.  Replaces date with datetime obj."""
        m = DEBUG_REGEX.match( line )
        if m:
            data = m.groupdict()
            data['timestamp'] = self.get_timestamp( data['timestamp'] )
            return data
        else:
            return {}
        
    
    @staticmethod
    def get_hms( hours , include_seconds=False ):
        """ Returns a string of Hours:Minutes:Seconds (seconds if desired) from hours decimal. """
        h = int( np.floor( hours ) )
        minutes = (hours - h) * 60
        m = int( np.floor( minutes ) )
        seconds = (minutes - m) * 60
        s = int( np.rint( seconds )  ) # take it to the nearest integer.
        
        if include_seconds:
            return '{}:{:02d}:{:02d}'.format( h, m, s )
        else:
            # Check if we need to round minutes up.
            if seconds >= 30.0:
                m += 1
                
            return '{}:{:02d}'.format( h, m )
        
        
    @staticmethod
    def get_timestamp( date_string ):
        """ This is potentially annoying because what about runs on new year's eve?  There's no year here. """
        stamp = datetime.datetime.strptime( date_string.strip() , "%b %d %H:%M:%S" )
        # This is where we would try to fix things if today is New Year's Day
        return stamp.replace( TODAY.year )
    

class ValkyrieDebug( DebugLog ):
    """ Specific class for parsing the Valkyire Debug log for workflow timing and status messages. """
    def parallel_grep( self , start_timestamp=None ):
        """ Merges all phrases for grepping into a single operation, stores to self.all_lines """
        greps = [ 'do_',        # for modules
                  'planStatus', # for high level timing
                  'start magnetic isp', # for mag to seq timing, possibly
                  ]
        
        self.all_lines = self.search_many( *greps )
        
        if start_timestamp:
            # filter all_lines by lines that have timestamp after the official experiment start.
            self.all_lines = [ line for line in self.all_lines if self.read_line( line )['timestamp'] > start_timestamp ]
        
        
    def detect_modules( self ):
        """ Reads log for workflow components, allowing detection of e2e runs. """
        # Primary search criteria is 'do_'
        modules = { 'libprep'   : False,
                    'harpoon'   : False,
                    'magloading': False,
                    'coca'      : False,
                    'sequencing': False  }
        
        if hasattr( self, 'all_lines' ):
            lines = [ line for line in self.all_lines if 'do_' in line ]
        else:
            lines = self.search( 'do_' )
            
        parsed  = [ self.read_line( line ) for line in lines ]
        
        conditions = [ ('libprep'   , ['do_libprep'] ),
                       ('harpoon'   , ['do_harpoon'] ),
                       ('magloading', ['do_magneticLoading'] ),
                       ('coca'      , ['do_coca'] ),
                       ('sequencing', ['do_sequencing'] ) ]
        
        for key, words in conditions:
            for line in parsed:
                message_words = line['message'].split()
                if set(words).issubset( set(message_words) ):
                    modules[ key ] = 'true' in [ w.lower() for w in message_words ]
                    
        self.modules = modules
    
    
    def get_overall_timing( self ):
        if hasattr( self, 'all_lines' ):
            lines = [ line for line in self.all_lines if 'planstatus' in line.lower() ]
        else:
            lines = self.search( 'planStatus' )
            
        parsed     = [ self.read_line( line ) for line in lines ]
        timing     = {}
        conditions = [ ('review',['Review,']),
                       ('library_start',['Library','Started,']),
                       ('library_end',['Library','Completed,']),
                       ('templating_start',['Templating','Started,']),
                       ('templating_end',['Templating','Completed,']),
                       ('sequencing_start',['Sequencing','Started,']),
                       ('sequencing_end',['Sequencing','Completed,']) ]
        
        for key,words in conditions:
            timing[ key ] = None
            for line in parsed:
                if set(words).issubset( set( line['message'].split() ) ):
                    timing[ key ] = line['timestamp']
                    
        self.timing = timing
        
    
    def plot_workflow( self, savepath='' ):
        if self.modules['libprep']:
            lib   = (self.timing['library_end'] - self.timing['library_start']).seconds / 3600.
            dead3 = (self.timing['templating_start'] - self.timing['library_end']).seconds / 3600.
        else:
            lib   = 0
            dead3 = 0
            
        if self.modules['harpoon'] or self.modules['magloading'] or self.modules['coca']:
            temp  = (self.timing['templating_end'] - self.timing['templating_start']).seconds / 3600.
            dead4 = (self.timing['sequencing_start'] - self.timing['templating_end']).seconds / 3600.
        else:
            temp  = 0
            dead4 = 0
            
        seq   = (self.timing['sequencing_end'] - self.timing['sequencing_start']).seconds / 3600.
        
        fig = plt.figure( figsize=(8,2) )
        ax  = fig.add_subplot( 111 )
        last = 0
        
        # Library preparation time
        if lib > 0:
            ax.barh( 0.5 , lib , 0.5, left=0, color='blue', alpha=0.4, align='center' )
            ax.text( lib/2., 0.5, 'Library Prep\n{}'.format( self.get_hms( lib ) ), color='blue',
                     ha='center', va='center', fontsize=10 )
            last += lib
            
            # Need dead time
            ax.barh( 0.5 , dead3 , 0.5, left=last, color='grey', alpha=0.4, align='center' )
            last += dead3
        
        # Templating time
        if temp > 0:
            ax.barh( 0.5 , temp, 0.5, left=last, color='green', alpha=0.4, align='center' )
            ax.text( last + temp/2., 0.5, 'Templating\n{}'.format( self.get_hms( temp ) ), color='green',
                     ha='center', va='center', fontsize=10  )
            last += temp
            
            # Dead time again
            ax.barh( 0.5 , dead4 , 0.5, left=last, color='grey', alpha=0.4, align='center' )
            last += dead4
        
        # Sequencing time
        ax.barh( 0.5 , seq , 0.5, left=last, color='red', alpha=0.4, align='center' )
        ax.text( last + seq/2. , 0.5, 'Sequencing\n{}'.format( self.get_hms( seq ) ), color='red',
                 ha='center', va='center', fontsize=10 )
        last += seq
        
        ax.text( last + 0.1 , 0.5, self.get_hms( last ),
                 va='center', fontsize=12 )
        
        ax.set_xlim( 0, ax.get_xlim( )[1] + 1 )
        ax.set_ylim( 0,1 )
        ax.yaxis.set_visible( False )
        ax.set_xlabel( 'Run Time (hours)' )
        fig.tight_layout( )
        
        if savepath:
            fig.savefig( savepath )
        else:
            fig.show( )

        timing_metrics = { 'total'     : last,
                           'libprep'   : lib ,
                           'templating': temp,
                           'sequencing': seq ,
                           'dead_time' : dead3 + dead4 ,
                           }
        return timing_metrics
        
    def analyze_templating( self ):
        """ 
        Pulls out components of templating and measures duration. 
        
        Harpoon     - library conversion, extend library, capture ISPs (enrich harpooned beads)
        Mag Loading - 
        COCA        - 1st, 2nd amp, meltoff, prepare for sequencing (primer + polymerase)
        """
        #temp_timing = {}
        pass

# grep -i Process: debug
##################################################
# Templating Stuff
##################################################

##################################################
# Templating Stuff
##################################################
# Harpoon
# Beginning of templating until start of mag loading

# Magnetic Loading
# grep -i "start magnetic isp" debug
# grep -i "post loading" debug

# COCA
# grep -i amp# debug
# grep -i meltoff debug
# grep -i "sequencing primer" debug 

