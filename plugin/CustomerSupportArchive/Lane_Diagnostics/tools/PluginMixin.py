import os, sys, re
from . import json2 as json

from . import chipcal, chiptype, explog

import numpy as np

class PluginMixin( object ):
    """
    This is a collection of functions and methods that are common to most if not all of our plugins.
    
    ToDo: Define a series of multilane methods for determination (without loading gain?)
    """

    def init_plugin( self ):
        """ 
        This is the primary method that will initialize common inputs and read commonly used datafiles. 
        """
        # Initialize the metrics that will ultimately be saved in results.json
        self.metrics = {}
        
        # Pull in relevant information from the run
        self.prepare_startplugin    ( )
        self.get_barcodes           ( )
        self.detect_bc_ladder       ( )
        self.read_explog            ( )
        self.read_ion_params        ( )
        self.get_flows              ( ) # This overwrites the value in prepare_startplugin
        # These need to happen after we get self.explog
        self.find_die_area          ( )
        self.check_explog_multilane ( ) #sets self.explog_is_multilane, self.explog_lanes_active
        
        # Set a bunch of convenience attributes for file existence:
        self.has_fc_bfmask          = os.path.exists( os.path.join( self.sigproc_dir , 'analysis.bfmask.bin' ) )
        self.has_block_bfmask       = os.path.exists( os.path.join( self.sigproc_dir , 'block_X0_Y0' , 'analysis.bfmask.bin' ) )
        self.has_bfmask_stats       = os.path.exists( os.path.join( self.sigproc_dir , 'analysis.bfmask.stats' ) )
        self.has_bfmask             = self.has_fc_bfmask or self.has_block_bfmask
        self.has_chipcal            = self.find_chipcal( )
        self.has_explog             = self.find_explog ( final=False )
        self.has_explog_final       = self.find_explog ( final=True )
        self.has_rawdata            = self.find_rawdata( )
        self.has_reference          = ( self.ion_params['referenceName'] != "" )
        self.has_nomatch_bam        = os.path.exists( os.path.join( self.basecaller_dir, 'nomatch_rawlib.basecaller.bam' ) )

        # Set convenience attributes for chemistry
        self.isCOCA                 = bool( self.ion_params['reverse_primer_dict']['chemistryType'] == 'avalanche' )
               
        # This stored the nomatch bam filepath for later processing if it exists
        if self.has_nomatch_bam:
            self.nomatch_bam_filepath = os.path.join( self.basecaller_dir, 'nomatch_rawlib.basecaller.bam' )
        else:
            self.nomatch_bam_filepath = None

        # Report on file status
        self.get_file_status( )
        
        # Determine and Validate the chiptype 
        try:
            self.determine_chiptype( )
        except ValueError:
            # Something went heinously wrong.  Let's exit as gracefully as possible.
            block_html = os.path.join( self.results_dir , '{}_block.html'.format( self.__class__.__name__ ) )
            with open( block_html , 'w' ) as f:
                msg = "Unable to find explog files, chip block directories, or calibration files in order to ascertain the chiptype for this plugin."
                print( msg )
                f.write( '<html><body><p><em>{}</em></p></body></html>'.format( msg ) )

    def check_explog_multilane( self ):
        ''' Check the explog values for LanesActiveX
                set     self.explog_is_multilane = boolean
                        self.explog_lanes_active = dict( lane_{}: boolean )
        '''
        # initialize explog_is_multilane to False
        explog_is_multilane = False
        explog_lanes_active = {}
        # loop through lanes, check if explog says active        
        for i in range( 1,5 ):
            try:               
                active = self.explog.metrics.get( 'LanesActive{}'.format(i), False )
            except (AttributeError, KeyError):
                # Default to False if indeterminate
                active = False
            # Keys are str(lane_num), values are bools
            explog_lanes_active.update( {'lane_{}'.format(i):active} )

            # If at least one lane is active, set explog_is_multilane = True
            #   Do nothing if false
            if active: explog_is_multilane = True
        # Assign attributes
        self.explog_is_multilane = explog_is_multilane
        self.explog_lanes_active = explog_lanes_active

    def detect_bc_ladder( self ):
        """
        Inspects the barcodes for this run to detect PQ / 560 read length ladder libraries.
        """
        self.pq_indices      = [11,12,13,14,15]
        self.pq_ladder    = {'IonXpress_011' : 110 ,
                             'IonXpress_012' : 120 ,
                             'IonXpress_013' : 130 ,
                             'IonXpress_014' : 140 ,
                             'IonXpress_015' : 150 }
        
        self.s560_indices    = [11,12,13,14,15,16,7,8,9,2]
        self.s560_ladder  = {'IonXpress_011' : 110 ,
                             'IonXpress_012' : 120 ,
                             'IonXpress_013' : 130 ,
                             'IonXpress_014' : 140 ,
                             'IonXpress_015' : 150 ,
                             'IonXpress_016' : 160 ,
                             'IonXpress_007' : 170 ,
                             'IonXpress_008' : 180 ,
                             'IonXpress_009' : 190 ,
                             'IonXpress_002' : 200 }
        
        self.has_pq_ladder   = False
        self.has_s560_ladder = False
        
        try:
            run_indices      = [ int( self.barcodes[b]['barcode_index'] ) for b in self.barcodes ]
        except KeyError:
            print( 'Run does not appear to have any barcodes; it clearly does not have a bc ladder.' )
            run_indices      = [ ]
            
        # Previously used .issubset method, however, this fails for many-bc runs that happen to include the ladder barcodes.
        # Changed to exact. from: if set( self.s560_indices ).issubset( set(run_indices) ):
        if set( self.s560_indices ) == set(run_indices):
            # This is a 560 library ladder
            self.has_s560_ladder = True
            print( 'Identified 10-barcode S560 library ladder.' )
        elif set( self.pq_indices ) == set(run_indices):
            # This is a PQ library ladder
            self.has_pq_ladder = True
            print( 'Identified 5-barcode PQ library ladder.' )
        else:
            print( 'No PQ or 560 barcode ladders found.' )
            
        return None
        
    def determine_chiptype( self ):
        """ Robust way to handle chip type determination """
        blockdir     = self.sigproc_dir
        use_blockdir = False
        chiprc       = None
        
        # Find a way to extract chiprc
        if hasattr( self , 'explog' ):
            # Let's pull rows and cols from there.
            chiprc  = ( self.explog.metrics['Rows'] , self.explog.metrics['Columns'] )
            
        try:
            # Start with what TS gives us.  Is that in chips.csv?
            ct = chiptype.ChipType( self.chip_type )
            print( 'TS chiptype resulted in chiptype object of type: {}'.format( ct.name ) )
        except KeyError:
            # Ok, it's not in chips.csv.  Let's get something from chiprc.
            if chiprc != None:
                ct = chiptype.get_ct_from_rc( *chiprc )
            else:
                # Ok, it seems that explog didn't work for some reason.
                try:
                    ct = chiptype.get_ct_from_dir( blockdir )
                    print( 'Warning!  TS chip type not found in chips.csv.  Chiptype derived from block directories to be: {}'.format( ct.name ) )
                    use_blockdir = True
                except IndexError:
                    print( 'Warning!  Could not find explog file or find blockdirs (is this a thumbnail?)' )
                    print( 'Now attempting to load chip R and C from the calibration files.' )
                    try:
                        cc         = chipcal.ChipCal  ( self.calibration_dir , chiptype=self.chip_type )
                        rows, cols = cc.load_offset_rc( )
                        chiprc     = ( rows , cols )
                        ct         = chiptype.get_ct_from_rc( rows, cols )
                    except:
                        raise ValueError( 'Error!  Unable to read chiptype from cal files.' )
                    
        # Now we can validate something.
        if use_blockdir:
            val_ct = chiptype.validate_chiptype( ct , blockdir=blockdir )
        else:
            val_ct = chiptype.validate_chiptype( ct , rc=chiprc )
            
        print( 'ChipType validated to be {0.name}'.format( val_ct ) )
        self.ct = val_ct
        
        return None

    def exit_on_missing_files( self, fc_bfmask=False, block_bfmask=False, bfmask_stats=False, bfmask=False, chipcal=False, explog=False, explog_final=False, rawdata=False, reference=False, nomatch_bam=False ):
        ''' If a file is required for analysis, set the kwarg=True.  
                The has_<file> attribute will be checked and if it is False, the plugin will exit with a message '''
        kwargs = locals()
        for kw, val in kwargs.items():
            if kw != 'self' and val and not getattr( self, 'has_' + kw ):
                # EXIT NOW!!!!!
                print( 'Exiting analysis.  Required file {} is missing'.format( kw ) )
                block_html = os.path.join( self.results_dir , '{}_block.html'.format( self.__class__.__name__ ) )
                with open( block_html , 'w' ) as f:
                    msg = "Unable to find required file {}.  Exiting gracefully without performing analysis.".format( kw )
                    f.write( '<html><body><p><em>{}</em></p></body></html>'.format( msg ) )
                sys.exit(0)

    def find_chipcal( self ):
        """ 
        Tests if chip calibration files (using gain) still exist in this run.
        """
        # Triggers off of gain because this is always loaded no matter what, even for edge analysis applications.
        gain_files = [ 'gainimage.dat' , 'gainImage0.dat', 'gainImage2.dat', 'gainImage3.dat' ]
        found      = False
        
        for gf in gain_files:
            if os.path.exists( os.path.join( self.calibration_dir , gf ) ):
                found = True
                break

        return found
    
    def find_die_area( self ):
        """ Finds the die area based on efuse x,y """
        try:
            x = self.explog.metrics['WaferX']
            y = self.explog.metrics['WaferY']
        except (AttributeError, KeyError):
            print( 'Unable to determine die area' )
            return
        refmap  = np.array( [ [0,0,1,1,1,1,1,0,0],
                              [0,1,2,2,2,2,2,1,0],
                              [1,2,3,3,3,3,3,2,1],
                              [1,2,3,3,3,3,3,2,1],
                              [1,2,3,3,3,3,3,2,1],
                              [0,1,2,2,2,2,2,1,0],
                              [0,0,1,1,1,1,1,0,0]  ] , dtype=np.int8 )
        
        names = ['unknown','edge','outer','center']
        area  = names[ refmap[ int(x) , int(y) ] ]
        
        self.explog.metrics['Area'] = area
        
        return None
    
    def find_explog( self , final=False ):
        """
        Tests if explog file exists.  Final=False doesn't care which.
        """
        found = False

        if hasattr( self , 'explog' ):
            if final:
                if self.explog.final:
                    found = True
            else:
                found = True

        return found
    
    def find_rawdata( self ):
        """ Does a quick check to see if raw data has been deleted already. """
        first_flow = 'acq_0000.dat'
        # 11/8/2018 - STP - Removed noPCA detection since that is chiptype dependent (and thumbnail dependent)
            
        if self.thumbnail:
            self.acq_dir = self.raw_data_dir
        else:
            self.acq_dir = os.path.join( self.raw_data_dir , 'thumbnail' )
            
        return os.path.exists( os.path.join( self.acq_dir , first_flow ) )

    def get_barcodes( self , min_read_count=1000 ):
        """ Reads in barcodes associated with the run """
        self.barcodes = {}
        allbc         = {}
        
        try:
            with open( 'barcodes.json','r' ) as f:
                allbc = json.load(f)
        except:
            try:
                with open(os.path.join(self.results_dir, 'barcodes.json'),'r') as f:
                    allbc = json.load(f) 
            except IOError:
                print( 'Warning!  barcodes.json file not found!' )
            except ValueError:
                print( 'Warning!  There appear to be no barcodes in this run!' )
            
        # NOTE: THIS DOES NOT PICK UP nomatch BARCODES
        for b in allbc:
            # Ignore barely-found barcodes as noise
            if int(allbc[b]['read_count']) > min_read_count:
                self.barcodes[b] = allbc[b]

        return None
    
    def get_file_status( self ):
        """ Prints to output the status of typical filetypes that might be needed for the plugin. """
        def get_msg( fname , found ):
            """ Helper function to print a simple status for a file type and if it's found or not. """
            if found:
                msg = '{} successfully found.'.format( fname )
            else:
                msg = 'Warning!  {} was not found!'.format( fname )

            return msg

        filenames = ['FC beadfind mask', 'Block beadfind mask', 'Beadfind mask STATS', 'Raw calibration data' , 'explog.txt' , 'explog_final.txt' ,
                     'Raw acquisition data', 'NoMatch Bam']
        info      = [self.has_fc_bfmask, self.has_block_bfmask, self.has_bfmask_stats, self.has_chipcal , self.has_explog, self.has_explog_final,
                     self.has_rawdata, self.has_nomatch_bam ]
        for fn, found in zip( filenames , info ):
            print( get_msg( fn , found ) )

        return None

    def get_flows( self ):
        ''' Get the number of flows, paying attention to custom analysis arguments, explog, and actual number of files '''
        # First check how many files exist
        if self.find_rawdata():
            filelist = os.listdir( self.acq_dir )
            regex = re.compile( r'^acq_(\d+)' )
            filelist = [ f for f in filelist if regex.search(f) ]
            filelist.sort( reverse=True )
            filelist.sort( key=lambda f: len(f), reverse=True )
            try:
                max_acq = int( regex.search( filelist[0] ).groups()[0] ) + 1
                print( 'Detected {} acquisition flow files'.format( max_acq ) )
            except:
                max_acq = 0
        else:
            max_acq = 0
        # next check explog
        try:
            exp_flows = self.explog.flows
        except:
            exp_flows = None
        print( 'Detected {} flows from explog'.format( exp_flows ) )
        # next check how many flows were analyzed
        basecallerargs = self.ion_params.get( 'basecallerArgs', '' )
        match = re.search( 'flowlimit=(\d+)', basecallerargs )
        if match:
            basecaller_flows = int( match.groups()[0] )
            print( 'Detected {} flows from basecaller'.format( basecaller_flows ) )
        else:
            basecaller_flows = None
        # Pick the safest value
        selected = None
        if basecaller_flows is None:
            if exp_flows is None:
                selected = max_acq
            elif max_acq == 0:
                selected = exp_flows
            else:
                selected = min( max_acq, exp_flows )
        else:
            if exp_flows is None:
                if max_acq == 0:
                    selected = basecaller_flows
                else:
                    selected = min( basecaller_flows, max_acq )
            else:
                if max_acq == 0:
                    selected = min( basecaller_flows, exp_flows )
                else:
                    selected = min( basecaller_flows, exp_flows, max_acq )
        print( 'Determined {} flows present'.format( selected ) )
        self.flows = selected
        return selected

    def get_array_height_and_width( self ):
        # initialize height and width to None
        height  = None
        width   = None
        # check if we have a file to parse
        if self.has_bfmask_stats:
            # look for the height and width values
            regex_height    = re.compile( 'Height = (\d+)' )
            regex_width     = re.compile( 'Width = (\d+)' )

            with open( os.path.join( self.sigproc_dir, 'analysis.bfmask.stats' ) ) as file:
                for line in file.readlines():
                    if not height:
                        match = regex_height.match( line )
                        if match:
                            groups = match.groups()
                            height = int( groups[0] )
                    if not width:
                        match = regex_width.match( line )
                        if match:
                            groups = match.groups()
                            width = int( groups[0] )
                    if height and width: 
                        break
        else:
            # just return Nones for the values
            pass
        return {'height':height, 'width':width}

    def has_noPCA_files( self ):
        """ Does a quick check to see if noPCA files exist """
        if not self.find_rawdata():
            return False
        first_flow = 'acq_0000.dat_noPCA'
        return os.path.exists( os.path.join( self.acq_dir , first_flow ) )

    # According to the documentation, the startplugin.json file data is automatically loaded in the class
    # and can be accessed via self.startplugin.  This code will try that first and if not try to load the file.
    def prepare_startplugin( self ):
        """ 
        Reads in namespace-like variables from startplugin.json, unless it's already a class attribute.
        Prepares common attributes of interest that we commonly use.
        """
        self.results_dir = os.environ['TSP_FILEPATH_PLUGIN_DIR']
        # if getattr( self , 'startplugin' , None ):
        #     print( 'Using startplugin class attribute.' )
        # else:
        print( 'Reading startplugin.json . . .' )
        try:
            with open('startplugin.json','r') as fh:
                self.startplugin = json.load(fh)
                
            print ( ' . . . startplugin.json successfully read and parsed.\n' )
        except:
            try: 
                with open(os.path.join(self.results_dir, 'startplugin.json'),'r') as fh:
                    self.startplugin = json.load(fh)
                
                print ( ' . . . startplugin.json successfully read and parsed.\n' )
            except:
                print ( "Error reading startplugin.json!" )
                return None
                
        # Define some needed variables
        self.plugin_dir    = self.startplugin['runinfo']['plugin_dir'  ] # This is the directory your plugin code lives in
        
        # For RUO TS, the following is True:
        #     On thumbnails, this is the 'thumbnail' directory
        #     On full chip, this is the calibration directory (thumbnail lives within there)
        #
        # For Valkyrie TS, this directory is not the thumbnail . . . it is where calibration files live
        self.raw_data_dir  = self.startplugin['runinfo']['raw_data_dir']
        
        self.analysis_dir   = self.startplugin['runinfo']['analysis_dir'] # Main directory for results from a sequencing run
        # self.results_dir    = self.startplugin['runinfo']['results_dir' ] # *** Save images, files, results.json, and html files within this directory ***
        self.sigproc_dir    = self.startplugin['runinfo']['sigproc_dir' ] # Holds pipeline files like bfmask and block directories from analysis
        self.basecaller_dir = self.startplugin['runinfo']['basecaller_dir' ] # Holds barcode nomatch .bam file and a variety of pngs and .sam files 
        self.flows          = self.startplugin['expmeta']['run_flows'   ]
        self.chip_type      = self.startplugin['expmeta']['chiptype'    ]
        
        self.thumbnail     = ( self.startplugin['runplugin']['run_type'] == 'thumbnail' )
        
        # This actually isn't the efuse.  Let's not use this for now.
        # self.efuse        = self.startplugin['expmeta']['chipBarcode' ]
        
        # This may be useful in other code. . .
        if 'thumbnail' in self.raw_data_dir:
            self.calibration_dir = self.raw_data_dir.split('thumbnail')[0]
        else:
            self.calibration_dir = self.raw_data_dir
            
        return None
    
    def read_explog( self , accept_initial=False ):
        """
        This function trys a number of methods through which to identify where the explog file is and parse it.
        
        The preference is to use the explog_final.txt data to have access to sensor data, but use of the
        accept_initial flag will allow explog.txt to be used as a last resort, if it is found.
        """
        def attempt_load( path , final=True ):
            """ 
            Attempts to load the explog (_final by default) at a given path. """
            found     = False
            file_name = 'explog_final.txt'
            if not final:
                file_name = 'explog.txt'
                
            fp = os.path.join( path , file_name )
            if os.path.exists( fp ):
                self.explog = explog.Explog( path=path )
                found = True
                
            return found
        
        self.explog_dir = None
        
        paths = [ self.raw_data_dir , os.path.dirname( self.raw_data_dir ) , self.analysis_dir ]
        names = [ 'raw_data_dir' , 'Parent Dir of raw_data_dir' , 'analysis_dir' ]
        
        # Loop through acceptable paths to find explog.  Break if we found it.
        for path, name in zip( paths, names ):
            if attempt_load( path , True ):
                print( 'Found explog_final.txt in {}!'.format( name ) )
                self.explog_dir = path
                break
        else:    
            # If we allow accept and haven't found explog_final, let's try a last ditch effort for explog.txt
            print( 'Unable to find explog_final.txt.' )
            if accept_initial:
                print( 'Attempting to find explog.txt . . .' )
                for path, name in zip( paths, names ):
                    if attempt_load( path , False ):
                        print( 'Found explog.txt in {}!'.format( name ) )
                        self.explog_dir = path
                        break
        
        return None
    
    def read_ion_params( self ):
        """
        Looks at ion_params_00.json for requisite info.
        """
        # Read ion_params_00.json
        pfile = os.path.join( self.analysis_dir , 'ion_params_00.json' )
        if not os.path.exists( pfile ):
            print( 'Error!  ion_params_00.json file not found.' )
            return None
        
        with open( pfile , 'r' ) as p:
            self.ion_params = json.load( p )
            
        self.plan_rl = int( self.ion_params['plan'].get( 'libraryReadLength' , 0 ) )
 
    def validate_lane_active( self, lane_id, detected ):
        ''' This function takes in a lane_id (string or int) and boolean (detected) and compares to the explog_active value 
                The function will return a boolean for valid or invalid
        '''
        explog_active = self.explog_lanes_active['lane_{}'.format(lane_id)]
        if not explog_active and detected:
            return False
        return True          

    def write_metrics( self ):
        """
        Saves self.metrics as the requisite results.json file for later scraping and analysis.
        """
        if self.metrics:
            with open( os.path.join( self.results_dir , 'results.json' ) , 'w' ) as results_json:
                json.dump( self.metrics , results_json )
        else:
            print( 'Warning!  No results.json file written because metrics appear to be empty. . .' )

