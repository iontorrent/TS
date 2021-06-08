import datetime, re, subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

TODAY       = datetime.datetime.today()
WF_REGEX    = re.compile( r""".*working directory: (?P<wd>[\w/.]+), workflowVersion: (?P<version>[\w/.=\s\-]+)""" )
GIT_RE      = re.compile( r"""git branch\s+=\s+(?P<branch>[\w/_/-]+)\s+git commit =\s+(?P<commit>[\w]+)""" )
DEBUG_REGEX = re.compile( r"""(?P<file>[\w/.]+):(?P<timestamp>[\w:\s]{15})\s(?P<inst>[\w\-_]+)\s(?P<source>[\w.]+):\s(?P<message>[\w\W\s]+)""" )

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
    def __init__( self, debug_path, start_timestamp=None, end_timestamp=None ):
        self.path = debug_path
        
        # Allows setting of the start point right up front to ignore previous runs' information.
        self.set_start_timestamp( start_timestamp )

        # This manages if runs are sequencing only and missing much of the normal architecture.
        self.set_end_timestamp( end_timestamp )
                
    def search( self, grep_phrase, case_sensitive=False , after=None, before=None, context=None):
        """ 
        Searches the debug file for the input phrase by using grep. 
        Returns lines as they are for further parsing.
        """
        cmd      = 'grep '
        if not case_sensitive:
            cmd += '-i '

        if context:
            if isinstance( context, int ):
                cmd += '--context {} '.format( context )
            else:
                print( 'Error, context input must be an integer.' )
        else:
            if before:
                if isinstance( before, int ):
                    cmd += '-B {} '.format( before )
                else:
                    print( 'Error, before input must be an integer.' )
            
            if after:
                if isinstance( after, int ):
                    cmd += '-A {} '.format( after )
                else:
                    print( 'Error, after input must be an integer.' )
                    
        cmd     += '"{}" {}'.format( grep_phrase, self.path )
        print( cmd )
        p        = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True )
        ans, err = p.communicate()
        print( ans )
        try:
            lines = ans.splitlines()
            print(lines)
            if self.start_timestamp:
                return self.filter_lines( lines )
            else:
                return lines
        except AttributeError:
            print( 'wtf' ) # Leaving for posterity :) 
            return []
          
    def search_many( self, *strings, **kwargs ):
        """ Searches the file for many grep strings at once. """
        case_sensitive = kwargs.get( 'case_sensitive', False )
        context        = kwargs.get( 'context', False )
        before         = kwargs.get( 'before', False )
        after          = kwargs.get( 'after', False )
        
        cmd      = 'grep '
        if not case_sensitive:
            cmd += '-i '
            
        if context:
            if isinstance( context, int ):
                cmd += '--context {} '.format( context )
            else:
                print( 'Error, context input must be an integer.' )
        else:
            if before:
                if isinstance( before, int ):
                    cmd += '-B {} '.format( before )
                else:
                    print( 'Error, before input must be an integer.' )
            
            if after:
                if isinstance( after, int ):
                    cmd += '-A {} '.format( after )
                else:
                    print( 'Error, after input must be an integer.' )
                    
        for string in strings:
            cmd     += '-e "{}" '.format( string )
            
        cmd     += self.path
        p        = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True )
        ans, err = p.communicate()
        try:
            lines = ans.splitlines()
            if self.start_timestamp:
                return self.filter_lines( lines )
            else:
                return lines
        except AttributeError:
            return []
        
    def search_blocks( self, block_start, block_stop, *strings, **kwargs ):
        ''' Finds a section that might be repeated and return selected lines
        
        blocks are lists of lines bracketed by endpoints block_start and block_stop
        
        Inputs -->  regex for block_start, block_stop, and *strings
                        NOTE: block_start and block_stop need to be rigorous regex strings with wildcards if necessary
                        strings does not have to be populated but will check for lines within the endpoints
                            --> strings can be simple phrases used by grep
                    endpoints is a bool to include or exclude the start/stop lines in output
                        default is False == exclude
                        NOTE: endpoints has to live in **kwargs due to Py2 behavior

                    kwargs are for the debug.search_many function
    
        Output --> list of blocks, where each block is a list of lines
        '''
        #NOTE: need to remove endpoints from kwargs if it exists
        #       Required by Py2
        try:                endpoints = kwargs.pop( 'endpoints' )
        except KeyError:    endpoints = False

        lines = self.search_many( block_start, block_stop, *strings, **kwargs )

        # Get blocks of lines for further parsing
        blocks = []
        add_line = False
        #print( 'initial -- add_line', add_line )
        regex_start = re.compile( block_start )
        regex_stop  = re.compile( block_stop )

        for l in lines:
            if not add_line:
                match = regex_start.match( l )
                if match:
                    add_line = True
                    temp = []
                    if endpoints: temp.append( l )
                    #print( 'start_phrase {} found, add_line'.format(start_phrase), add_line )
                else:
                    continue
            else:
                match = regex_stop.match( l )
                if match:
                    if endpoints: temp.append( l )
                    blocks.append(temp)
                    add_line = False
                    #print( 'stop_phrase {} found, add_line'.format(stop_phrase), add_line )
                    #print( 'updated blocks', blocks )
                else:
                    #print( 'adding line to temp', l )
                    temp.append( l )
        return blocks

    def set_start_timestamp( self, start_timestamp ):
        """ 
        Sets the initial timestamp for the debug file that will prevent messages prior to that 
        moment from being returned and worked with.  Useful for avoiding constant reuse of timestamp 
        filtering.
        """
        if isinstance( start_timestamp, datetime.datetime ):
            self.start_timestamp = start_timestamp
        else:
            self.start_timestamp = None
            print( "Starting point NOT set.  Please input a datetime.datetime object." )
            
    def set_end_timestamp( self, end_timestamp ):
        """ 
        Sets the end timestamp for the debug file that will prevent messages after that 
        moment from being returned and worked with.  Useful for avoiding constant reuse of timestamp 
        filtering.
        """
        if isinstance( end_timestamp, datetime.datetime ):
            self.end_timestamp = end_timestamp
        else:
            self.end_timestamp = None
            print( "Ending point NOT set.  Please input a datetime.datetime object." )

    def read_line( self, line ):
        """ Matches the line pattern and extracts the relevant components.  Replaces date with datetime obj."""
        m = DEBUG_REGEX.match( line )
        if m:
            data = m.groupdict()
            data['timestamp'] = self.get_timestamp( data['timestamp'], 
                                                    start_timestamp=self.start_timestamp, 
                                                    end_timestamp=self.end_timestamp,
                                                    )
            return data
        else:
            return {}
        
    def filter_lines( self, lines , start_timestamp=None ):
        """ 
        Filter out lines before the given start timestamp, which are probably from a previous run
        because the debug log is written to serially and not necessarily restarted each time a new
        experiment is started.
        
        Also works without the input timestamp if we set it with set_start_timestamp.
        """
        # If no starttimestamp is passed in, try using the attribute self.start_timestamp
        if start_timestamp is None: start_timestamp = self.start_timestamp

        # Add routine to ignore '--' lines in the grep output.
        lines = [ line for line in lines if line != '--' ]
        if isinstance( start_timestamp, datetime.datetime ):
            # filter all_lines by lines that have timestamp after the official experiment start.
            try:
                filtered = [ l for l in lines if self.read_line( l )['timestamp'] > start_timestamp ]
                return filtered
            except KeyError:
                return []
        else:
            print( "Not doing any filtering.  Please input a datetime.datetime object." )
            return lines
        
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
    def get_timestamp( date_string, start_timestamp=None, end_timestamp=None ):
        """ Extract the timestamp from debug log line, convert to datetime object, and add in a year """
        # NOTE: No Year in the debuglog timestamps
        stamp = datetime.datetime.strptime( date_string.strip() , "%b %d %H:%M:%S" )
        # Below is where we add in a year
        # First try to use the start and end timestamps pulled from explog since they have the year -- B.P. 17Sep2019
        if start_timestamp:
            # try matching the start timestamp month if it exists
            if stamp.month == start_timestamp.month: 
                return stamp.replace( start_timestamp.year )
            # try matching the end timestamp month if it exists
            elif end_timestamp:  
                if stamp.month == end_timestamp.month: 
                    return stamp.replace( end_timestamp.year )
        # Maybe all we have is an end_timestamp...not sure why, but it's possible
        elif end_timestamp:  
            if stamp.month == end_timestamp.month: 
                return stamp.replace( end_timestamp.year )
        else:
            # Fallback method --> not guaranteed to work on or about New Year's Eve
            return stamp.replace( TODAY.year )
    
    
class ValkyrieDebug( DebugLog ):
    """ Specific class for parsing the Valkyire Debug log for workflow timing and status messages. """
    
    def parallel_grep( self , start_timestamp=None ):
        """ Merges all phrases for grepping into a single operation, stores to self.all_lines """
        # try using attribute self.start_timestamp is start_timestamp is None
        if start_timestamp is None: start_timestamp = self.start_timestamp

        greps = [ 'do_',        # for modules
                  'planStatus', # for high level timing
                  ': peStatus', # for a typo.
                  'start magnetic isp', # for mag to seq timing, possibly
                  ]
        
        self.all_lines = self.search_many( *greps )
        
        if start_timestamp:
            # filter all_lines by lines that have timestamp after the official experiment start.
            self.all_lines = self.filter_lines( self.all_lines, start_timestamp )
        
        
    def detect_modules( self ):
        """ Reads log for workflow components, allowing detection of e2e runs. """
        # Primary search criteria is 'do_'
        # Let's assume that if we end up with a run report, we are actually doing sequencing...?
        modules = { 'libprep'   : False,
                    'harpoon'   : False,
                    'magloading': False,
                    'coca'      : False,
                    'sequencing': True  }
        
        #if hasattr( self, 'all_lines' ):
        #    lines = [ line for line in self.all_lines if 'do_' in line ]
        #else:
        #    lines = self.search( 'do_' )
        
        lines = self.search( 'do_' )
            
        parsed  = [ self.read_line( line ) for line in lines ]
        
        conditions = [ ('libprep'   , ['libprep'] ),
                       ('harpoon'   , ['harpoon'] ),
                       ('magloading', ['magneticLoading'] ),
                       ('coca'      , ['coca'] ),]
                       #('sequencing', ['sequencing'] ) ]
        
        for key, words in conditions:
            print( 'Searching for {} . . .'.format( key ) )
            for line in parsed:
                m = re.match( '''.*do_(?P<module>[\w]+)\s(?P<active>[\w]+)''', line['message'] )
                if m:
                    module = m.groupdict()['module']
                    active = m.groupdict()['active']
                    if module in words:
                        modules[ key ] = active.lower() == 'true'
                        print( '. . . {}'.format( active ) )
                # Previous method
                #message_words = line['message'].split()
                #if set(words).issubset( set(message_words) ):
                #    modules[ key ] = 'true' in [ w.lower() for w in message_words ]

        self.modules = modules

        print( 'summary' )
        for k in ['libprep','harpoon','magloading','coca','sequencing']:
            print( '{}:\t{}'.format( k  , modules[k] ) )
    
    
    def get_overall_timing( self ):
        if hasattr( self, 'all_lines' ):
            #lines = [ line for line in self.all_lines if 'planstatus' in line.lower() ]
            lines = [ line for line in self.all_lines if 'planstatus' in line.lower() or 'pestatus' in line.lower()]
        else:
            # Bugfix.  Looks like someone accidentally overwrote planStatus with peStatus
            lines = self.search_many( 'planStatus', ': peStatus' )
            
        parsed     = [ self.read_line( line ) for line in lines ]
        timing     = {}
        conditions = [ ('review',['Review']),
                       ('library_start',['Library','Started']),
                       ('library_end',['Library','Completed']),
                       ('templating_start',['Templating','Started']),
                       ('templating_end',['Templating','Completed']),
                       ('sequencing_start',['Sequencing','Started']),
                       ('sequencing_end',['Sequencing','Completed']) ]
        
        for key,words in conditions:
            timing[ key ] = None
            for line in parsed:
                message_words = [ w.replace(',','') for w in line['message'].split() ]
                if set(words).issubset( set( message_words ) ):
                    timing[ key ] = line['timestamp']
                    
        # Check for if they are all blank.
        if not any( timing.values() ):
            timing['sequencing_start'] = self.start_timestamp # Faster than getattr and now initialized to None
            timing['sequencing_end']   = self.end_timestamp
            
        self.timing = timing
        
        # PW:  At one point, I thought this was a good idea.  Instead, I want runs that are not easily detected
        #      as having run modules to be categorized as "unknown" runs rather than muddying "Sequencing only"
        
        # Update modules in case this was a tricky run that was manually loaded with emPCR, for instance:
        #if self.modules['sequencing'] == False:
        #    if self.timing['sequencing_start'] and self.timing['sequencing_end']:
        #        print( 'Detected sequencing start/end times and updating modules to include sequencing!' )
        #        self.modules['sequencing'] == True
                
    
    def plot_workflow( self, savepath='' ):
        # Set colors for the workflow
        COLORS = {'lib' : 'blue',
                  'temp': 'green',
                  'seq' : 'darkcyan' }
        
        if self.modules['libprep']:
            lib   = (self.timing['library_end'] - self.timing['library_start']).total_seconds() / 3600.
            dead3 = (self.timing['templating_start'] - self.timing['library_end']).total_seconds() / 3600.
        else:
            lib   = 0
            dead3 = 0
            
        if self.modules['harpoon'] or self.modules['magloading'] or self.modules['coca']:
            temp  = (self.timing['templating_end'] - self.timing['templating_start']).total_seconds() / 3600.
            dead4 = (self.timing['sequencing_start'] - self.timing['templating_end']).total_seconds() / 3600.
        else:
            temp  = 0
            dead4 = 0

        try:
            seq   = (self.timing['sequencing_end'] - self.timing['sequencing_start']).total_seconds() / 3600.
        except TypeError:
            print( 'Error reading timing details from overall timing. Unable to plot workflow timing.' )
            timing_metrics = { 'total'     : 0,
                               'libprep'   : lib ,
                               'templating': temp,
                               'sequencing': 0 ,
                               'dead_time' : dead3 + dead4 ,
            }
            return timing_metrics
        
        fig = plt.figure( figsize=(8,2) )
        ax  = fig.add_subplot( 111 )
        last = 0
        
        # Library preparation time
        if lib > 0:
            ax.barh( 0.5 , lib , 0.5, left=0, color=COLORS['lib'], alpha=0.4, align='center' )
            ax.text( lib/2., 0.5, 'Library Prep\n{}'.format( self.get_hms(lib) ), color=COLORS['lib'],
                     ha='center', va='center', fontsize=10 )
            last += lib
            
            # Need dead time
            ax.barh( 0.5 , dead3 , 0.5, left=last, color='grey', alpha=0.4, align='center' )
            last += dead3
        
        # Templating time
        if temp > 0:
            ax.barh( 0.5 , temp, 0.5, left=last, color=COLORS['temp'], alpha=0.4, align='center' )
            ax.text( last + temp/2., 0.5, 'Templating\n{}'.format( self.get_hms( temp ) ),
                     color=COLORS['temp'], ha='center', va='center', fontsize=10  )
            last += temp
            
            # Dead time again
            ax.barh( 0.5 , dead4 , 0.5, left=last, color='grey', alpha=0.4, align='center' )
            last += dead4
        
        # Sequencing time
        ax.barh( 0.5 , seq , 0.5, left=last, color=COLORS['seq'], alpha=0.4, align='center' )
        ax.text( last + seq/2. , 0.5, 'Seq.\n{}'.format( self.get_hms( seq ) ), color=COLORS['seq'],
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
    

    def detect_workflow_version( self ):
        """
        Reads through debug log to identify the version of the workflow scripts.
        """
        # Initialize values
        workflow_version = { 'working_directory': None,
                             'version': None, 
                             'branch': None,
                             'commit': None
                             }
        
        version_lines = self.search( 'workflowVersion:' )
        if version_lines:
            line = self.read_line( version_lines[0] )
            m    = WF_REGEX.match( line['message'] )
            if m:
                info = m.groupdict()
                workflow_version['working_directory'] = info['wd']
                workflow_version['version']           = info['version']
                if 'git' in info['version']:
                    # This is a branch and we need to record
                    gm = GIT_RE.match( info['version'] )
                    if gm:
                        git_info                   = gm.groupdict()
                        workflow_version['commit'] = git_info['commit']
                        workflow_version['branch'] = git_info['branch']
                        
        return workflow_version

    def detect_init( self ):
        """ Detects if initialization happened during the run, and if so, returns timing details. """
        timing = {}
        lines  = [line for line in self.search_many( 'script_init.py &&' , 'script_init_cancel.py' ) if 'running command' in line]
        
        for line in lines:
            parsed    = self.read_line( line )
            msg_lower = parsed['message'].lower()
            
            if 'script_init.py' in msg_lower:
                timing['start'] = parsed['timestamp']
            elif 'script_init_cancel.py' in msg_lower:
                timing['end']   = parsed['timestamp']
                
        if set(['start','end']).issubset( set(timing.keys()) ):
            # We have the right entries to do the calculations we need
            timing['duration'] = float( (timing['end'] - timing['start']).seconds / 3600. )

        return timing

    def detect_postchip_clean( self ):
        """ Detects if the postchipclean routine was run.  This occurs once all lanes on a chip are spent. """
        timing = {}
        lines  = self.search_many( '''.*Starting thread.*script_postchipclean.py.*''',
                                   '''script_postchipclean_cancel.py''' )
        
        for line in lines:
            parsed    = self.read_line( line )
            msg_lower = parsed['message'].lower()
            
            if 'clean.py' in msg_lower:
                timing['start'] = parsed['timestamp']
            elif '_cancel.py' in msg_lower:
                timing['end']   = parsed['timestamp']
                
        if set(['start','end']).issubset( set(timing.keys()) ):
            # We have the right entries to do the calculations we need
            timing['duration'] = float( (timing['end'] - timing['start']).seconds / 3600. )
            
        return timing

    def detect_postrun_clean( self ):
        """ 
        Detects if the postrun clean routine was run.  This clean is a subset of the postchipclean routine.
        Only cleans the lanes used in this run.  Time required scales with number of lanes used.
        """
        timing = {}
        lines  = self.search_many( '''Script_PostRunClean.txt''',
                                   '''PostRunClean''', context=5 )
        
        for line in lines:
            parsed    = self.read_line( line )
            msg_lower = parsed['message'].lower()
            
            if 'openscriptdirfile' in msg_lower:
                timing['start'] = parsed['timestamp']
            elif 'experiment complete' in msg_lower:
                timing['end']   = parsed['timestamp']
                
        if set(['start','end']).issubset( set(timing.keys()) ):
            # We have the right entries to do the calculations we need
            timing['duration'] = float( (timing['end'] - timing['start']).seconds / 3600. )
            
        return timing

