import os, re, datetime

from . import chiptype as ct # These are allowed because they are similarly lite
from .efuse import Efuse     # These are allowed because they are simillary lite

class Explog( object ):
    def __init__( self, path=None, body=None ):
        # Load the explog contents
        if path is not None:
            self.init_path( path )
        elif body is not None:
            self.init_body( body )

        self.metrics = {}
        self.parse()

    ##############################################################
    # Setup functions                                            #
    ##############################################################
    def init_path( self, path ):
        ''' Load the explog from file '''
        self.path    = path
        self.lines   = []

        # Find log file, default to explog_final
        try:
            self.log = os.path.join( self.path, 'explog_final.txt' )
            self.load()
            self.found = True
            self.final = True
            return
        except ( OSError, IOError ):
            pass
        try:
            self.log = os.path.join( self.path, 'explog.txt' )
            self.load()
            self.found = True
            self.final = False
            return
        except ( OSError, IOError ):
            pass
        try:
            self.log = self.path
            self.load()
            self.found = True
            self.final = None
            return
        except ( OSError, IOError ):
            pass
        self.found = False
        self.final = None

        if not self.found:
            print( 'Error!  No explog files found at given path:\n\t%s' % self.path )

    def init_body( self, body ):
        ''' Load the explog from the input text string '''
        self.path  = '.'
        lines = []
        self.lines = body.split( '\n' )
        # 5/7/2019 - I'm not sure what these are doing here, but they seem to have been
        # added on 10/9/2017 (3774333c636), and today it seems like a really bad idea, 
        # since it breaks up the efuse
        #self.lines = []
        #for line in lines:
        #    self.lines += line.split( ',' )
        
        self.found = True
        self.final = None

    def load( self ):
        ''' opens file and reads in the lines '''
        with open( self.log , 'r' ) as f:
            self.lines = f.readlines()
        
    ##############################################################
    # Parser main functions                                      #
    ##############################################################
    def parse( self ):
        # TODO: iterate over metrics.
        # when splitting, only split on the first :
        self.get_chiptype()

        # Get the correct metric list
        if self.chiptype.series.lower() in ['pgm']:
            self.assigner( self.PGM_METRICS, self.metrics, self.find )
        elif self.chiptype.series.lower() in [ 'proton', 's5' ]:
            self.assigner( self.METRICS, self.metrics, self.find )
        else:
            raise NotImplementedError( 'unknown chip series: {series}'.format( self.chiptype ) )
        
        # Parse the Efuse
        if self.chiptype.series.lower() in [ 'proton', 's5' ]:
            self.parse_efuse()

        # Assign attributes
        # Jan 2020 - 'DynamicRange' is the correct DR key, it was a datacollect bug that confused us.  - PW
        # Feb 2020 - Not so much.  The above was isolated to Valkyrie.  On other systems (560, for instance)
        #                the AfterBF number is right.  *sigh*  - PW
        #if self.final:
        #    self.DR = self.metrics['DynamicRange']
        #elif not self.final:
        #    self.DR = self.metrics['DynamicRangeAfterBF']
        self.DR = self.metrics['DynamicRangeAfterBF']
        
        self.isRaptor   = self.metrics['Platform'].lower() in [ 's5', 's5xl']
        self.isValkyrie = self.metrics['Platform'].lower() in [ 'valkyrie' ]
        
        # Advanced parsers
        self.get_run_time( )
        self.parse_advscriptfeatures()
        self.read_errors()
        self.get_flows() 
        self.get_multilane()

    def find( self , s , default=None ):
        ''' scans self.lines for a specific set of text. '''
        value = default
        for line in self.lines:
            if s in line:
                value = line.split(':',1)[1]
                value = value.strip()
                #value = value.replace(' ' ,'')
                #value = value.replace('\n','')
                break
        return value

    def assigner( self, metrics, dest, extractor ):
        ''' Assigns formated metrics into "dest", using safe encapsulation
            to ensure errors result in assignment of default
            operates on "dest" in place
           
        Inputs: 
            metrics: list of tuples, each formated as:
                       ( destination dictionary key, 
                         source "key" (or attribute), 
                         default, 
                         operation to format the extracted source value )
            dest:   dictionary to place results
            extractor: function (or lambda) who's input is the "source key" and output is a raw value
        '''
        for metric in metrics:
            local, key, default, method = metric
            if method == int:
                method = lambda l: int(float(l))
            if key is None:
                dest[local] = default
            elif key == 'CHIPTYPE':
                try:
                    dest[local] = method( self.chiptype )
                except:
                    dest[local] = default
                    print( 'Error processing CHIPTYPE method: {} -> {}'.format( key, local ) )
            else:
                try:
                    value = extractor ( key )
                    dest[local] = method(value)
                except Exception as e1:
                    try:
                        dest[local] = method(self, value)
                    except Exception as e2:
                        dest[local] = default
                        print( 'Error processing field: {} -> {}'.format( key, local ) )
                        #raise e1

    @property
    def raw_dict( self ):
        ''' 
        Return the explog as a raw dictionary object, avoiding any 
        formating assumptions
        '''
        blocks  = re.compile( r'^((block_\d+)|(thumbnail_\d+))$' )

        output = {}
        sections = [ 'ExperimentErrorLog', 'ExperimentInfoLog' ]
        section = None
        misc = 0

        for line in self.lines:
            if section and line.strip() and ( line.lstrip() == line ):
                # We were in an indented section
                # But now we have reached an unindented line
                # we must have reached the end of the section
                section = None
            try:
                key, val = line.split( ':', 1 )
            except:
                line = line.strip().rstrip( ':' )
                if not line:
                    continue
                key   = 'msg {}'.format( misc )
                misc += 1
                val   = line

            key = key.strip()
            val = val.strip()

            if blocks.match( key ):
                if section != 'blocks':
                    section = 'blocks'
                    output[section] = {}
                    method = 'add'
            elif key == 'ExperimentInfoLog':
                section = key
                output[section] = {}
                method = 'add'
                continue
            elif key == 'ExperimentErrorLog':
                section = key
                output[section] = []
                method = 'append'
                continue


            if section:
                if method == 'add':
                    output[section][key] = val
                elif method == 'append': 
                    output[section].append( line.strip() )
            else:
                output[key] = val
        return output

    ##############################################################
    # Complex imports, requiring multi-line or advanced matching #
    ##############################################################
    def read_errors( self ):
        ''' Reads explog for errors '''
        def process_matches( matches, includefiles=False ):
            locs = sorted( list( set( [ m[1] for m in matches ] ) ) )
            summary = {}
            if not locs: 
                return summary
            for loc in locs:
                files = list( set( [ m[0] for m in matches if m[1] == loc ] ) )
                isbeadfind = ( 'beadfind_pre_0001' in files ) or ( 'beadfind_pre_0003' in files )
                files.sort( key=filesorter )
                summary[loc] = { 'instances': len(files), 
                                 'first':     files[0],
                                 'beadfind':  isbeadfind, }
                if includefiles:
                    summary[loc]['files'] = files
                    
            files = list( set( [ m[0] for m in matches ] ) )
            isbeadfind = ( 'beadfind_pre_0001' in files ) or ( 'beadfind_pre_0003' in files )
            files.sort( key=filesorter )
            summary['total'] = { 'instances': len(files), 
                                 'first':     files[0],
                                 'beadfind':  isbeadfind, }
            if includefiles:
                summary['total']['files'] = files
            return summary
        
        def filesorter( filename ):
            if '_' in filename:
                section = '_'.join(filename.split( '_' )[:-1])
                try:
                    index   = int( filename.split( '_' )[-1] )
                except ValueError:
                    # Very non standard acquisitions
                    section = filename.split('.')[0]
                    index = 0
            else:
                section = filename.split('.')[0]
                index   = 0
            sections = [ 'R4', 'R3', 'R2', 'R1', 'W1', 'beadfind_pre', 'prerun', 'extraG', 'acq' ]
            try:
                sid = sections.index( section )
            except:
                sid = -1
            return sid*10000+index
        
        start       = False
        self.errors = []
        linklosses  = []
        regionslips = []
        
        regionslip = re.compile( 'Region Slip on file (\w{,20})\.dat 0x([a-z0-9]*)' )
        linkloss   = re.compile( 'Link lock loss on file (\w{,20})\.dat 0x([a-z0-9]*)' )
        
        for line in self.lines:
            if 'ExperimentErrorLog:' in line:
                try:
                    index = self.lines.index( 'ExperimentErrorLog: \n' )
                except ValueError:
                    # This would occur if we fed text into the class' body input
                    try:
                        index = self.lines.index( 'ExperimentErrorLog: ' )
                    except ValueError:
                        print( 'Error processing errors, could not find beginning of error section.' )
                        return None
                    
                for line in self.lines[(index+1):]:
                    # Handle link losses and region slips separately from other alarms and warnings.
                    rs = regionslip.match( line.strip() )
                    ll = linkloss.match  ( line.strip() )
                    if rs:
                        regionslips.append( rs.group(1,2) )
                    elif ll:
                        linklosses.append ( ll.group(1,2) )
                    elif line.strip() not in ['','ExpLog_Done']:
                        self.errors.append( line.strip() )
                        
        self.regionslips = process_matches( regionslips )
        self.linklosses  = process_matches( linklosses  )

        # Process other specific errors
        # Defaults
        self.metrics['Valkyrie_Dev_Scripts'] = False
        
        for error in self.errors:
            # Valkyrie development scripts
            if 'ALARM: Workflow link exists' in error:
                self.metrics['Valkyrie_Dev_Scripts'] = True
                
    def parse_advscriptfeatures( self ):
        ''' There can be multiple advanced script features '''
        self.AdvScriptFeatures = []
        for line in self.lines:
            if 'AdvScriptFeaturesName:' in line:
                self.AdvScriptFeatures.append( line.split('AdvScriptFeaturesName:')[1].strip() )

    def get_chiptype( self ):
        ''' Get the chip type, initially trying from TsChipType '''
        TsChipType = self.find('TsChipType')
        try:
            if not TsChipType:
                self.chiptype = ct.ChipType( self.find('ChipType') )
            else:
                self.chiptype = ct.ChipType( TsChipType )
        except:
            self.chiptype = ct.ChipType( 'unknown' )

        if self.chiptype.type in ['P0','510','520','521','521v2','530','530v2']:
            self.metrics['ChipType'] = 'Proton 0'
        elif self.chiptype.type in ['P1','540','540v2','541']:
            self.metrics['ChipType'] = 'Proton I'
        elif self.chiptype.type in ['P2']:
            self.metrics['ChipType'] = 'Proton II'
        elif self.chiptype.type in ['550','P3']:
            self.metrics['ChipType'] = '550 | P2.3'
        else:
            self.metrics['ChipType'] = self.chiptype.type
            
        self.metrics['chiptype_name'] = self.chiptype.name

    def parse_efuse( self ):
        fuse = self.metrics.get( 'Efuse', '' )
        ef = Efuse( fuse )
        self.assigner( self.EFUSE, self.metrics, lambda k: getattr(ef,k) )

    def get_flows( self ):
        ''' Determine number of flows...and discern from the 'OrigFlows' line in proton explogs '''
        for line in self.lines:
            if 'Flows:' in line:
                if 'Orig' not in line:
                    self.flows = int( line.split(':')[1] )
                    return

    def get_multilane( self, recalculate_chiptype=True ):
        ''' Determine if this is really a multilane chip '''
        if self.chiptype.series.lower() in ['pgm']:
            # No such thing as a multilane PGM...yet
            self.is_multilane = False
            return
        if self.isValkyrie:
            # Valkyrie is always multilane
            self.is_multilane = True
        else:
            # Could be a modified instrument.
            # If all lanes are active, then it's likely a full chip
            # If <4 lanes are active, it's a multilane
            lanefields = [ 'LanesActive1', 'LanesActive2', 'LanesActive3', 'LanesActive4' ]
            is_multilane = 0
            for field in lanefields:
                if self.metrics.get( field ):
                    is_multilane += 1
            if is_multilane == len(lanefields):
                # Probably a full chip. not actually a multilane.
                is_multilane = 0
            self.is_multilane = bool( is_multilane )

        if self.is_multilane and recalculate_chiptype and 'Val' not in self.chiptype.name:
            try:
                chiptype = ct.ChipType( 'Val' + self.chiptype.name )
                print( 'setting chiptype to multilane {}'.format( chiptype.name ) )
                self.chiptype = chiptype
            except:
                pass
            
    def get_run_time( self ):
        """ 
        This function reads in Start Time and End Time and takes the difference.  Returns Hours.
        Used (hopefully) to help identify end-to-end Valkyrie runs.
        """
        start_str = self.find( 'Start Time' )
        end_str   = self.find( 'End Time'   )
        print( 'Start Time from explog: {}'.format(start_str))
        print(   'End Time from explog: {}'.format( end_str ))
        
        try:
            start = datetime.datetime.strptime( start_str , '%m/%d/%Y %H:%M:%S' )
        except:
            start = None
        try:    
            end   = datetime.datetime.strptime( end_str   , '%m/%d/%Y %H:%M:%S' )
        except:
            end   = None
        
        # Leave these as strings so that they are json serializable.
        self.metrics['start_time'] = start_str
        self.metrics['end_time']   = end_str
            
        # Assign datetime objects as explog attributes for future reference and no need for recalling strptime
        self.start_time = start
        self.end_time   = end
        
        if start and end:
            delta     = end - start
            run_hours = (24. * delta.days) + (delta.seconds / 3600.)
        else:
            run_hours = 0.
            
        # Note that anything > 5 hours on Valkyrie probably means end-to-end
        self.metrics['run_time']   = float( '{:.1f}'.format( run_hours ) )
        return None
    
    ##############################################################
    # Metric-specific formaters                                  #
    ##############################################################
    def calc_oversampling( self, v ):
        if float(v) == 625:
            return '1x'
        elif float(v) == 1176:
            return '2x'
        raise ValueError( 'unknown frequency: %s' % v )

    def calc_framerate( self, v ):
        if float(v) == 625:
            return 15.0
        elif float(v) == 1176:
            return 30.0
        raise ValueError( 'unknown frequency: %s' % v )
    def calc_framerate_pgm( self, v ):
        if float(v):
            return 1./float(v)
        raise ValueError( 'Cannot calculate frame rate from frame time: %s' % v )

    def parse_hist( self, v ):
        return [ int(i) for i in v.strip().split() ]

    def parse_cgc( self, v ):
        return [ float(i) for i in v.strip().split() ]

    def chiprows( self, v ):
        ''' This gets a special filter because the default backup depends on self.chiptype 
        i.e. We want to query the explog, but if there is an error, we default to back.
             If we only wanted a a dependence on self.chiptype but did not want to query
             explog, there are other ways to handle this '''
        if v is None:
            return self.chiptype.chipR
        else:
            return int( float( v ) ) 

    def chipcols( self, v ):
        ''' This gets a special filter because the default backup depends on self.chiptype 
        i.e. We want to query the explog, but if there is an error, we default to back.
             If we only wanted a a dependence on self.chiptype but did not want to query
             explog, there are other ways to handle this '''
        if v is None:
            return self.chiptype.chipC
        else:
            return int( float( v ) )

    def get_rtn( self , v ):
        ''' This pulls the row temporal noise metric from the 'ChipNoiseInfo' metric line '''
        match_string = r'(?P<na>.+RTN:)(?P<rtn>[0-9\.]+)'
        m = re.match( match_string , v )
        if m:
            return float( m.groupdict()['rtn'] )
        else:
            return 0.0


    def make_bool( self, v ):
        ''' Robustly convert the specified value to boolean '''
        if isinstance( v, list ):
            if len(v) == 1:
                return make_float( v[0] )
        if v in ( True, False, None ):
            return v
        try:
            vstr = v.lower()
            maps = { 'true': True, 
                     'false': False, 
                     'null': None, 
                     'none': None, 
                     'yes': True, 
                     'no': False }
            return maps[vstr]
        except AttributeError:
            pass
        return bool(v)

    ##############################################################
    # Lookup tables                                              #
    ##############################################################
    # These need to be defined after the function definitions so that functions are available
    # local, key, default, dtype
    METRICS = ( (                           'Efuse',              'Chip Efuse:',    '',   str ), 
                (                        'Platform',                 'Platform', 'n/a',   str ), 
                (                     'ChipBarcode',              'ChipBarcode', 'n/a',   str ), 
                (                     'ChipVersion',              'ChipVersion', 'n/a',   str ),
                (                   'ChipTSVersion',            'ChipTSVersion', 'n/a',   str ),
                (                      'TsChipType',               'TsChipType', 'n/a',   str ),
                (                    'ChipMajorRev',             'ChipMajorRev', 'n/a',   str ),
                (                    'ChipMinorRev',             'ChipMinorRev', 'n/a',   str ),
                (                    'ChipType_raw',                 'ChipType', 'n/a',   str ),
                (                            'Rows',                     'Rows',  None, chiprows ),
                (                         'Columns',                  'Columns',  None, chipcols ),
                (             'AnalogSupplyVoltage',                     'vdda',   0.0, float ),
                (            'DigitalSupplyVoltage',                     'vddd',   0.0, float ),
                (             'OutputSupplyVoltage',                     'vddo',   0.0, float ),
                (                            'vdda',                     'vdda',   0.0, float ),
                (                            'vddd',                     'vddd',   0.0, float ),
                (                            'vddo',                     'vddo',   0.0, float ),
                (                            'facc',                     'facc',   0.0, float ),
                (                             'dac',                     'dac:',   0.0, float ),
                (                  'FluidPotential',                      'dac',   0.0, lambda l: int( float(l) * 1000 ) ),
                ( 'PreSeqRunReferenceElectrodeMean',                      'dac',   0.0, lambda l: int( float(l) * 1000 ) ),
                (                        'ChipMode',                 'ChipMode',     0,   int ),
                (                    'DynamicRange',            'DynamicRange:',   0.0, lambda l: float(l)*1000. ),
                (               'DynamicRangeForBF',        'DynamicRangeForBF',   0.0, lambda l: float(l)*1000. ),
                (             'DynamicRangeAfterBF',      'DynamicRangeAfterBF',   0.0, lambda l: float(l)*1000. ),
                (              'DynamicRangeActual',       'DynamicRangeActual',   0.0, lambda l: float(l)*1000. ),
                (                      'RangeStart',              'RangeStart:',     0,   int ),
                (                  'RangeStart_pre',          'RangeStart_pre:',     0,   int ),
                (                 'RangeStart_post',         'RangeStart_post:',     0,   int ),
                (                             'DSS',              'RangeStart:',     0,   int ),
                (                       'ChipNoise',                'ChipNoise',   0.0, float ),
                (                        'RowNoise',                 'RowNoise',   0.0, float ),
                (                        'ChipGain',                 'ChipGain',   0.0, float ),
                (                     'CalGainStep',              'CalGainStep',   0.0, float ),
                (        'Pixels out of range High', 'Pixels out of range High',     0,   int ),
                (         'Pixels out of range Low',  'Pixels out of range Low',     0,   int ),
                (                 'Pixels in range',          'Pixels in range',     0,   int ),
                (                'Pixels with gain',         'Pixels with gain',     0,   int ),
                (                   'GainCurveVref',            'GainCurveVref',    [], parse_cgc ),
                (                   'GainCurveGain',            'gainCurveGain',    [], parse_cgc ),
                (                  'Cal image Hist',           'Cal image Hist',    [], parse_hist),
                (                   'Cal Chip Hist',            'Cal Chip Hist',    [], parse_hist),
                (                        'FpgaFreq',                 'FpgaFreq',   0.0, float ), 
                (                        'ChipFreq',                 'ChipFreq',   0.0, float ), 
                (                  'ClockFrequency',                 'ChipFreq',   0.0, float ), 
                (                    'Oversampling',                 'ChipFreq', 'n/a', calc_oversampling ),
                (                       'FrameRate',                 'ChipFreq',   0.0, calc_framerate ),
                (                      'DeviceName',               'DeviceName', 'n/a', str ),
                (                  'ReleaseVersion',          'Release_version', 'n/a', str ),
                (              'DatacollectVersion',      'Datacollect_version',     0, str ),
                (                 'LiveViewVersion',         'LiveView_version',     0, int ), 
                (                  'ScriptsVersion',           'Script_version', 'n/a', str ),
                (                  'RFIDMgrVersion',          'rfidmgr_version',     0, int ),
                (                 'GraphicsVersion',         'Graphics_version',     0, int ),
                (                       'OSVersion',               'OS_version',     0, int ),
                (                      'RSMVersion',              'RSM_version',     0, int ),
                (                      'OIAVersion',              'OIA_version',     0, int ),
                (               'ReaderFPGAVersion',      'Reader FPGA_version', 'n/a', str ),
                (                  'MuxFPGAVersion',         'Mux FPGA version',     0, int ),
                (                'ValveFPGAVersion',       'Valve FPGA_version', 'n/a', str ),
                (                    'FlowRateMain',             'FlowRateMain',   0.0, float ),
                (                    'FlowRateChip',             'FlowRateChip',   0.0, float ),
                (                      'OverSample',               'OverSample', 'n/a', str ),
                (                       'OverClock',                'OverClock', 'n/a', str ),
                (                        'ChipTemp',                'ChipTemp:',   0.0, float ),
                (                       'ChipTemp0',                'ChipTemp0',   0.0, float ),
                (                       'ChipTemp1',                'ChipTemp1',   0.0, float ),
                (                       'ChipTemp2',                'ChipTemp2',   0.0, float ),
                (                       'ChipTemp3',                'ChipTemp3',   0.0, float ),
                (                   'ChipNoiseInfo',           'ChipNoiseInfo:',    '', str ),
                (                'RowTemporalNoise',           'ChipNoiseInfo:',   0.0, get_rtn ),
                (                    'LanesActive1',             'LanesActive1',  True, make_bool ), 
                (                    'LanesActive2',             'LanesActive2',  True, make_bool ), 
                (                    'LanesActive3',             'LanesActive3',  True, make_bool ), 
                (                    'LanesActive4',             'LanesActive4',  True, make_bool ), 
                (                     'LanesAssay1',              'LanesAssay1',  'n/a', str ), 
                (                     'LanesAssay2',              'LanesAssay2',  'n/a', str ), 
                (                     'LanesAssay3',              'LanesAssay3',  'n/a', str ), 
                (                     'LanesAssay4',              'LanesAssay4',  'n/a', str ), 
                (                          'doInit',                   'doInit',  True, make_bool ), 
                (                   'doLibraryPrep',            'doLibraryPrep',  True, make_bool ), 
                (                       'doHarpoon',                'doHarpoon',  True, make_bool ), 
                (                'doTemplatingPrep',         'doTemplatingPrep',  True, make_bool ), 
                (                  'doPostLibClean',           'doPostLibClean',  True, make_bool ), 
                (                 'doParallelClean',          'doParallelClean',  True, make_bool ), 
                (                   'doVacuumClean',            'doVacuumClean',  True, make_bool ), 
                (                 'doPostChipClean',          'doPostChipClean',  True, make_bool ), 
                (                    'postRunClean',             'postRunClean',  True, make_bool ), 
                (                      'flow_order',                'Image Map', 'n/a',   str ), 
                (                'reseq_flow_order',                'Reseq Map', 'n/a',   str ), 
                (                  'doResequencing',           'doResequencing',  True, make_bool ), 
              )

    PGM_METRICS =( (                           'PGMHW',              'PGM HW:',     0, float ),
                   (                        'Platform',              'PGM HW:', 'n/a', lambda l: 'PGM {0}'.format(l) ), 
                   (                        'ChipType',             'CHIPTYPE', 'n/a', lambda l: l.type ),   # CHIPTYPE is automatically interpreted to pass chiptype into the formater
                   (                     'ChipVersion',             'CHIPTYPE', 'n/a', lambda l: l.name[3:]),
                   (                            'Rows',             'CHIPTYPE',     0, lambda l: l.chipR),
                   (                         'Columns',             'CHIPTYPE',     0, lambda l: l.chipC),
                   # There is no efuse info on PGM chips.     
                   (                       'CMOSLotId',                   None, "n/a", str ),
                   (                   'AssemblyLotId',                   None, "n/a", str ),
                   (                        'FlowCell',                   None, "n/a", str ),
                   (                  'PackageTestBin',                   None, "n/a", str ),
                   (              'PackageTestSoftBin',                   None, "n/a", str ),
                   (                         'Comment',                   None, "n/a", str ),
                   # Save operation metrics
                   (             'AnalogSupplyVoltage',                   None,  3.30, float ),
                   (            'DigitalSupplyVoltage',                   None,  3.30, float ),
                   (             'OutputSupplyVoltage',                   None,  3.30, float ),
                   (                  'ClockFrequency',            'Frequency',     0, float ),
                   (                    'Oversampling',           'Oversample', 'n/a', str ),
                   # Read in reference electrode voltage
                   (                             'dac',       'Ref Electrode:',     0, lambda l: float(l[:5]) ),
                   (                  'FluidPotential',       'Ref Electrode:',     0, lambda l: int(1000*float(l[:5])) ),
                   ( 'PreSeqRunReferenceElectrodeMean',       'Ref Electrode:',     0, lambda l: int(1000*float(l[:5])) ),
                   (                       'FrameTime',           'Frame Time',     0, float ), 
                   (                       'FrameRate',           'Frame Time',     0, calc_framerate_pgm ),
                   (                    'DynamicRange',                  None,  236.0, float ),
                   (               'DynamicRangeForBF',                  None,  236.0, float ),
                   (             'DynamicRangeAfterBF',                  None,  236.0, float ),
                   (                'ChipThermometer0',                  None,    0.0, float ),
                   (                'ChipThermometer1',                  None,    0.0, float ),
                   (                'ChipThermometer2',                  None,    0.0, float ),
                   (                'ChipThermometer3',                  None,    0.0, float ),
                   (             'ChipThermometerMean',                  None,    0.0, float ),
                   (                      'DeviceName',                  None,  'n/a', str ),
                   (                    'PGMSWRelease',      'PGM SW Release',  'n/a', str ),
                   (              'DatacollectVersion', 'Datacollect version',      0, int ),
                   (                 'LiveViewVersion',    'LiveView version',      0, int ),
                   (                  'ScriptsVersion',      'Script version',  'n/a', str ),
                   (                 'GraphicsVersion',    'Graphics version',      0, int ),
                   (                       'OSVersion',          'OS version',      0, int ),
                   (                 'FirmwareVersion',    'Firmware version',      0, int ),
                   (                     'FPGAVersion',        'FPGA version',      0, int ),
                   (                   'DriverVersion',      'Driver version',      0, int ),
                   (                    'BoardVersion',       'Board version',  'n/a', str ),
                   (                     'KernelBuild',        'Kernel Build',  'n/a', str ),
                 )

    #                   Metrics Dict, Efuse Class Attribute, default, operation
    EFUSE = ( (          'CMOSLotId',                 'lot',   'n/a',   str ),
              (            'WaferId',               'wafer',       0,   int ),
              (             'WaferX',                   'x',       0,   int ),
              (             'WaferY',                   'y',       0,   int ),
              (      'AssemblyLotId',            'assembly',   'n/a',   str ),
              (           'FlowCell',            'flowcell',   'n/a',   str ),
              (      'PackageTestId',                'part',       0,   int ),
              (     'PackageTestBin',             'hardbin',       0,   int ),
              ( 'PackageTestSoftBin',             'softbin',       0,   int ),
              (           'DryNoise',               'noise',       0, float ),
              (            'Comment',             'comment',   'n/a',   str ),
              (        'ChipBarcode',             'barcode',   'n/a',   str ),
              (      'EfuseChipType',            'chiptype',   'n/a',   str ),
            )

ExpLog = Explog
