#!/usr/bin/env python
# Copyright (C) 2019 Ion Torrent Systems, Inc. All Rights Reserved

from ion.plugin import *

import sys, os, datetime, time, json, re, csv, math, glob, textwrap
import numpy as np
import time # delete later
import matplotlib
matplotlib.use( 'agg' ) # to make sure this is set for before another module doesn't set it
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# Set default cmap to jet, which is viridis on newer TS software (matplotlib 2.1)
matplotlib.rcParams['image.cmap'] = 'jet'

# Load image tools to resize deck images, if requested.
from PIL import Image

# Import our tools
from tools.PluginMixin import PluginMixin
from tools.debug_reader import ValkyrieDebug
get_hms = ValkyrieDebug.get_hms
from tools import html as H
from tools import sparklines
from tools.libpreplog import LibPrepLog 
from collections import OrderedDict

# Required for lane wetting detection
#from skimage.measure import compare_ssim
#import cv2


LOG_REGEX = re.compile( '''summary.(?P<timestamp>[\w\-_:]+).log''' )

class ValkyrieWorkflow( IonPlugin , PluginMixin ):
    """ 
    Phil Waggoner
    
    Plugin to analyze timing of Valkyrie workflow for run tracking and performance monitoring.
    
    Latest Update | CN: v1.1.32 added additional try/except in pipPrepTest section for csa. Also for csa,
                        updated path block html uses to find flow_spark.svg
    Latest Update | CN: v1.1.30 bugfix for when robot waste is very clogged, bugfix for when there are None values in flow data. bugfix in pinch measure      
                        empty libPrepLog no longer registers as a plugin error- just says not found. 
    Latest Update | CN: Added pipette pickup tip errors, alert and metrics, added is_integrated metric      
    Latest Update | CN: Added pipette alert and metrics for pipette pressure tests     
    Latest Update | CN: Added pipette timedout errors to pipette error section     
    Latest Update | CN: separated bottom tube errors by pipette. Added pipette serial numbers.     
    Latest Update | CN: added pipPress plots    
    Latest Update | CN: Bug-proof plugin. Search for !! to find changes.    
    Latest Update | CN: Only make symlink to tube bottom log csv if link does not already exist.    
    Latest Update | CN: Flow rate sparkline will include reseq flows, if reseq run on ValkTS.    
    Latest Update | CN: added CW and CW+MW flow rate median and std to results.json   
    Latest Update | CN: added libpreplog stuff   
    """
    version       = "1.1.32" # must also change the version number in launch function
    allow_autorun = True
    
    runTypes      = [ RunType.THUMB , RunType.FULLCHIP , RunType.COMPOSITE ]
    
    def launch( self ):
        print( "Start Plugin" )
        self.init_plugin( )
       
        self.version = "1.1.32"
        print('Version {}'.format(self.version))

        self.metrics['software_version'] = str(self.explog.metrics['ReleaseVersion'])
        self.metrics['dev_scripts_used'] = self.explog.metrics['Valkyrie_Dev_Scripts']
        self.metrics['datacollect']      = str(self.explog.metrics['DatacollectVersion'])
        self.metrics['scripts_version']  = str(self.explog.metrics['ScriptsVersion'])
        
        self.metrics['plugin_error'] = False # initialize value

        try:
            if self.purification_dir:
                print('Found purification path from startplugin.json: {}'.format(self.purification_dir))
                self.metrics['is_integrated'] = True
            else:
                self.metrics['is_integrated'] = False
        except:
            print('!! Unable to determine if this is an integrated run')

        # Set up debug_log path
        self.debug_log = os.path.join( self.calibration_dir , 'debug' )
        
        # Exit plugin if we're missing required data after waiting a while. Have to do this on RUO server and ValkTS since dbg folder may also be missing 
        for delay in [(20,False),(20,False),(10,False),(10,False),(1,True)]:
            not_a_valk, missing_debug = self.is_missing_data( final_check=delay[1] )
            if not_a_valk:
                print('We are on a valkyrie TS but this is not a valkyrie run. This should not be possible!')
                sys.exit(0)
            if missing_debug:
                if delay[1]:
                    sys.exit(0) # if debug is missing after the final delay, we give up and exit
                print( 'Debug not found. Waiting {} minutes before checking again.'.format(delay[0]) )
                time.sleep(delay[0]*60) # debug is missing, but we will wait and try again 
            else:
                break
        
        # Identify if we are on RUO or Valkryie by presence of dbg folder - where deck and chip images live
        dbg = os.path.join( self.calibration_dir , 'dbg' )
        if os.path.exists( dbg ):
            self.valkyrie_analysis = True
            self.dbg = dbg
            print( 'Detected that analysis is being done on a Valkyrie!  Images located at {}!'.format( self.dbg ) )
        elif os.path.exists( os.path.join( self.raw_data_dir, 'dbg' ) ):
            self.valkyrie_analysis = False
            self.dbg = os.path.join( self.raw_data_dir, 'dbg' )
            print( 'Detected that analysis is being done on an RUO TS!  Images located at {}!'.format(  self.dbg ) )
        else:
            print( 'Unable to find debug images!' )
            self.dbg = self.raw_data_dir
        
        # Set do_reseq variable, used in various areas. Eventually save as metric?  
        try:
            self.do_reseq = self.explog.metrics['doResequencing']
            self.reseq_flow_order = self.explog.metrics['reseq_flow_order'].lower() 
        except:
            self.do_reseq = False
            self.reseq_flow_order = None
        
        self.prepare_chip_images( )
        self.prepare_deck_images( scale_factor=2 )
        self.analyze_flow_rate  ( )
        
        # Initialize debug log reader.  Send in end time as well in case of weird seq only runs.
        self.expt_start          = self.explog.start_time
        print('FROM DEBUG', self.expt_start, self.explog.end_time )
        self.debug               = ValkyrieDebug( self.debug_log, self.expt_start, self.explog.end_time )
        self.metrics['workflow'] = self.debug.detect_workflow_version( )
        
        # Detect modules and, if we do not detect any, assume this is a mock run and exit the plugin.
        self.debug.detect_modules( )
        
        # Define a minimum flow threshold for mock run detection
        self.mock_flow_threshold = 200
        if self.is_mock_run( ):
            #print('actually, lets continue')
            sys.exit(0)
            
        ###########################################################################
        # Calculate Initialization, 'postrun,' and Analysis Pipeline Timing
        ###########################################################################
        
        self.init_timing = self.debug.detect_init( )
        self.metrics['init_done'] = False
        if self.init_timing:
            print( 'Detected that the instrument was initialized this run, taking {:.1f} hours.'.format( self.init_timing.get('duration',0) ) )
            self.metrics['init_done'] = True
            
        self.postchip_clean_timing = self.debug.detect_postchip_clean( )
        if self.postchip_clean_timing:
            print( 'Detected that PostChipClean took place for this run, taking {:.1f} hours.'.format( self.postchip_clean_timing.get('duration',0) ) )
            
        self.postrun_clean_timing = self.debug.detect_postrun_clean( )
        if self.postrun_clean_timing:
            print( 'Detected that PostRunClean took place for this run, taking {:.1f} hours.'.format( self.postrun_clean_timing.get('duration',0) ) )
            
        # Read pipeline info for post-OIA data - need to add summary of sample timing to self.analysis_timing
        try:
            self.analysis_timing = self.analyze_pipeline( )
            print('analysis_timing: ')
            print( self.analysis_timing )
            
            self.sample_timing, self.sample_plugin_timing   = self.analyze_samples ( )
            
            print('sample_timing: ')
            print( self.sample_timing )
            
            if self.sample_timing:
                self.analysis_timing['Samples'] = self.sample_timing
                self.analysis_timing['Sample-Level Plugins'] = self.sample_plugin_timing
        except:
            print('!! Something went wrong when analyzing summary logs. Skipping')
            self.metrics['plugin_error'] = True
            
        # If we found OIA data, let's analyze and add to the analysis_timing dictionary.
        oia_timing = os.path.join( self.raw_data_dir, 'onboard_results', 'sigproc_results', 'timing.txt' )
        if os.path.exists( oia_timing ):
            print( 'Found OIA timing.txt.  Now analyzing.' )
            try:
                self.oia_timing = Timing( oia_timing )
                self.oia_timing.make_detailed_plot( self.results_dir )
                self.analysis_timing['OIA'] = self.oia_timing.overall
                print( 'OIA timing:' )
                print( self.oia_timing.overall )
            except:
                print('!! Something went wrong when analyzing OIA data. Skipping analysis')
                self.metrics['plugin_error'] = True
            
        
        ###########################################################################
        
        # Analyze Workflow Timing -- Now incorporating ScriptStatus.csv.
        self.analyze_workflow_timing(self.do_reseq )
        
        ###########################################################################
        # Analyze files for finding clogs in sequencing lines 
        ###########################################################################
        self.doPostChip = self.explog.metrics['doPostChipClean'] 
        self.debugInfo = DebugInfo( self.calibration_dir, self.results_dir, self.expt_start, self.debug.all_lines )
        if self.debugInfo.foundConicalClogCheck:
            self.metrics['conical_clog_check'] = self.debugInfo.ccc_metrics  # conical clog check happens in PostRunClean and PostChipClean
            if self.debugInfo.postChipClean:
                self.metrics['pcc'] = self.debugInfo.pcc_metrics             # save all metrics generated from all other postChipClean tests
        if self.debugInfo.plugin_error:
            self.metrics['plugin_error'] = True
        
        
        self.flows = FlowInfo( self.calibration_dir, self.results_dir, self.explog.metrics['flow_order'].lower(), self.do_reseq, self.reseq_flow_order, true_active=self.true_active )
        if self.flows.hasFlowData:
            self.metrics['flow_rate'].update( self.flows.flow_metrics )
        if self.flows.plugin_error:
            self.metrics['plugin_error'] = True
       
        # New- libPrepLog
        if self.ss_file_found: 
            self.libpreplog = LibPrepLogCSV( self.calibration_dir, self.results_dir, self.expt_start, self.ss.data, seq_end=self.seq_end  )
            if self.libpreplog.hasData:
                self.metrics['libpreplog'] = self.libpreplog.lpl_metrics 
        else:
            print('Skipping analysis of libpreplog since ScriptStatus.csv not found')
            self.libpreplog = LibPrepLogCSV( self.calibration_dir, self.results_dir, self.expt_start )
        
        if self.libpreplog.plugin_error:
            self.metrics['plugin_error'] = True

        # Analyze new Pipette Tests
        self.pipPresTests = PipettePresTests( self.calibration_dir, self.results_dir )
        if self.pipPresTests.plugin_error:
            self.metrics['plugin_error'] = True
        elif self.pipPresTests.found_pipPres:
            self.metrics['pipPresTest'] = self.pipPresTests.results 

        # Analyze Tube Bottom locations
        self.analyze_tube_bottoms( )
       
        # Look for blocked tips
        self.count_blocked_tips( )

        # Analyze vacuum log files
        self.analyze_vacuum_logs( )

        # Analyze pipette behavior
        self.analyze_pipette_behavior( )
        
        # Find pipette errors such as er52
        self.find_pipette_errors( ) # function to find er52 and other pipette errors

        # Search for instances in which pipette failed to pickup tip
        self.find_pipette_failed_to_pickup_tip( )
        
        # Check if plugin completed before postrun cleans were complete
        self.did_plugin_run_too_early( )
        # self.message = '' when PostRun data was found and the plugin did not run too early
        
        # Find pipette serial numbers
        self.find_pipette_serialNumbers( )

        # Create graphic of which modules were used....Or just make an html table.
        self.write_block_html( )
        
        # If PostRun data is missing, wait for it to appear. If it appears, rerun parts of the plugin
        if self.csa:
            check_after_x_mins = [10,10,10,10,10,10,10,10,10,10,10,10] # input the number of minutes to delay and for how many iterations
        else: 
            check_after_x_mins = [10,10,10,10,10,10,10]
        rerun = False # set to true if we find PostRun after waiting
        if self.message == '': 
            self.write_metrics( )
            print( "Plugin Complete." )
        else:
            for delay in check_after_x_mins:
                print( '_______________Waiting {} minutes for data to transfer to TS.'.format(delay) )
                time.sleep(delay*60)
                print( 'Check if postRun has ended....' )
                self.analyze_workflow_timing( self.do_reseq ) # checks the ScriptStatus file for postRun complete
                self.did_plugin_run_too_early( )
                if self.message == '':
                    print('_______________PostRun has been found. Stop waiting.')
                    rerun = True
                    break
                else:
                    print('_______________PostRun still not found.')
            if rerun:
                print('_______________Re-running parts of ValkyrieWorkflow....')
                # Analyze vacuum log files - should now find PostLibClean vac logs
                self.analyze_vacuum_logs( )
                ###########################################################################
                # Analyze files for finding clogs in sequencing lines 
                ###########################################################################
                self.doPostChip = self.explog.metrics['doPostChipClean'] 
                self.debugInfo = DebugInfo( self.calibration_dir, self.results_dir, self.expt_start, self.debug.all_lines )
                if (self.debugInfo.foundConicalClogCheck):
                    self.metrics['conical_clog_check'] = self.debugInfo.ccc_metrics  # conical clog check happens in PostRunClean and PostChipClean
                    if self.debugInfo.postChipClean:
                        self.metrics['pcc'] = self.debugInfo.pcc_metrics             # save all metrics generated from all other postChipClean tests
                self.flows = FlowInfo( self.calibration_dir, self.results_dir, self.explog.metrics['flow_order'].lower(), self.do_reseq, self.reseq_flow_order, true_active=self.true_active )
                if self.flows.hasFlowData:
                    self.metrics['flow_rate'].update( self.flows.flow_metrics )
                self.write_block_html( )
            else:
                print( '_______________PostRun was never found... moving on' )
            
            # Complete plugin 
            self.write_metrics( )
            print( "Plugin Complete." )
        
    def analyze_pipeline( self ):
        """ Finds the appropriate summary.log file and reads in analysis modules and timing. """
        # The summary.__.log files live in the analysis dir.
        log, log2 = self.get_first_log( self.analysis_dir )
        if log:
            # If we're here, it's time to read in the information.
            summary_log = SummaryLog( os.path.join( self.analysis_dir, log ) )
            return summary_log.timing
        else:
            return {}
    
    def analyze_samples( self ):
        """ Processes through sample log files and determines sample analysis timing details. """
        sample_analysis_logs = []
        sample_plugin_logs = []
        log_dict = {}
        # Parse log files
        for sample_name in self.sample_dirs:
            sample_dir = os.path.join( self.analysis_dir, self.sample_dirs[sample_name] )
            log, log2 = self.get_first_log( sample_dir )
            if log:
                log_dict[sample_name]={}
                sl_obj = SummaryLog( os.path.join( sample_dir, log ), sample_name )
                log_dict[sample_name]['analysis_log'] = sl_obj
                sample_analysis_logs.append( sl_obj ) 
                print('log {} for sample {} in dir {}'.format( log, sample_name, dir ) )
            if log2:
                sl_obj = SummaryLog( os.path.join( sample_dir, log2 ), sample_name )
                log_dict[sample_name]['plugin_log'] = sl_obj
                sample_plugin_logs.append( sl_obj ) 
        # Exit if we found no samples
        if not sample_analysis_logs:
            print( 'Error!  No samples found.  This is embarassing . . .' )
            print( 'sample dirs: {}'.format(self.sample_dirs) )
            return {}, {}
        
        #try:
        # Sort the logs by earliest analysis start time
        sample_analysis_logs = sorted( sample_analysis_logs, key=lambda x: x.get_start_time() )
        sample_plugin_logs = sorted( sample_plugin_logs, key=lambda x: x.get_start_time() ) 
        timing_sample_analysis = { 'start' : sample_analysis_logs[0].get_start_time(),
                   'end'   : sorted( [s.get_end_time() for s in sample_analysis_logs] )[-1] }
        timing_sample_plugins = { 'start' : sample_plugin_logs[0].get_start_time(),
                   'end'   : sorted( [s.get_end_time() for s in sample_plugin_logs] )[-1] }
        
        
        print('sample plugin end: {}'.format( sorted( [s.get_end_time() for s in sample_plugin_logs] )[-1] ) )
        
        # Measure duration CN: only uses the input, but happens to be a method of the summary log class 
        timing_sample_analysis['duration'] = sample_analysis_logs[0].get_duration( timing_sample_analysis )
        timing_sample_plugins['duration'] = sample_plugin_logs[0].get_duration( timing_sample_plugins )
        
        
        #except:
        #    print('something went wrong with sample duration analysis...')
        #    return {}
        
        # Plot the data separately
        # CN: find a way to add in the plugin time- perhaps using the log_dict ? 
        def plot_sample( s, y ):
            """ Currently done to plot in minutes """
            info    = [ (k, s.timing[k]['start'], s.timing[k]['duration']) for k in s.timing ]
            procs   = sorted( info, key=lambda x: x[1] )
            labels  = [p[0] for p in procs]
            patches = []
            cm      = matplotlib.cm.Set3
            colors  = [cm(i) for i in np.linspace(0,1,12)]
            for i,proc in enumerate(procs):
                left  = (proc[1] - procs[0][1]).total_seconds() / 60.
                patch = plt.barh( y+i, proc[2]*60., height=0.8, left=left, align='center', color=colors[i],
                                  label=proc[0], zorder=3 )
                patches.append( patch )
                
            return y+i, labels, patches
        
        def plot_sample_all( sample_dict, y ):
            """ 
            Currently done to plot in minutes. Altered from plot_sample, this function uses the log_dict
            to show sample analysis timing and sample plugin timing from summary logs 1 and 2, respectivel.
            Should be find even when there is no second summary log (not tested)."""
            s_a = sample_dict['analysis_log']
            s_p = sample_dict['plugin_log']
            info_analysis  = [ (k, s_a.timing[k]['start'], s_a.timing[k]['duration']) for k in s_a.timing ]
            try:
                info_plugins   = [ (k, s_p.timing[k]['start'], s_p.timing[k]['duration']) for k in s_p.timing ]
            except:
                info_plugins = [] # incase there is no second summary file found
            
            info = info_analysis + info_plugins 
            procs   = sorted( info, key=lambda x: x[1] )
            labels  = [p[0] for p in procs]
            patches = []
            cm      = matplotlib.cm.Set3
            colors  = [cm(i) for i in np.linspace(0,1,12)]
            for i,proc in enumerate(procs):
                left  = (proc[1] - procs[0][1]).total_seconds() / 60.
                patch = plt.barh( y+i, proc[2]*60., height=0.8, left=left, align='center', color=colors[i],
                                  label=proc[0], zorder=3 )
                patches.append( patch )
                
            return y+i, labels, patches
        
        # This is where the magic happens
        plt.figure ( figsize=(12,4) )
        plt.subplot( 121 )
        yticktups = []
        y         = 0
        space     = 2
        
        sorted_sample_names = log_dict.keys()
        sorted_sample_names.sort()
        
        #for sample in sample_analysis_logs:
        for sample_name in sorted_sample_names:
            #last_y, labels, patches = plot_sample( sample, y )
            last_y, labels, patches = plot_sample_all( log_dict[sample_name], y )
            #yticktups.append( ((last_y-y)/2. + y,sample.name) )
            yticktups.append( ((last_y-y)/2. + y,sample_name) )
            y = last_y + space
            plt.axhline( y =(last_y+space/2) , color='grey', ls='-' )
            
        plt.legend( handles=patches, bbox_to_anchor=(1.1,1), loc=2, borderaxespad=0. )
        yt,yl = zip(*yticktups)
        plt.yticks( yt, yl )
        plt.ylim  ( -1, y-space/2 )
        plt.xlabel( 'Time (Minutes)' )
        plt.grid( axis='x', ls=':', color='grey', zorder=0 )
        plt.tight_layout()
        plt.savefig( os.path.join( self.results_dir, 'sample_analysis_timing.png' ) )
        plt.close( )
        
        # Pass out a timing heirarchy to be leveraged in the overall timing plot.
        return timing_sample_analysis, timing_sample_plugins
    
    def get_summary_logs( self, log_dir ):
        """ Helper method to get summary log files.  Typically we want the first one created. """
        logs      = [l for l in os.listdir( log_dir ) if LOG_REGEX.match( l ) ]
        return sorted( logs )
    
    def get_first_log( self, log_dir ):
        """ 
        Wrapper on get_summary_logs to give the first log, usually what we want.
        CN update: first log gives sample analysis timing, second log gives timing for sample-level 
        plugin execution. Lets get the first two since for certain runs, the plugin execution 
        takes a significant amount of time and need to be shown in the timing analysis plot. 
        """
        logs = self.get_summary_logs( log_dir )
        try:
            log1 = logs[0]
        except IndexError:
            log1 = ''
        try:
            log2 = logs[1]
        except IndexError:
            log2 = ''
        return log1, log2
        
    def find_debug_images( self, regex , scale_factor=1 , convert_to_jpg=False ):
        """ Reusable routine to find images from the dbg folder and copy to plugin output dir """
        
        def resize( source, dest, scale_factor ):
            """ Quick resize of images using PIL.  Source and dest need to include image name. """
            original  = Image.open( source )
            new_size  = ( int( original.size[0]/scale_factor ), int( original.size[1]/scale_factor ) )
            try:
                small     = original.resize( new_size , Image.LANCZOS )
            except AttributeError:
                # Older versions of PIL don't have LANCZOS, use ANTIALIAS
                small     = original.resize( new_size , Image.ANTIALIAS )
            small.save( dest , 'PNG' )
            return None
        
        def make_jpg( source, dest ):
            """ Convert image to jpg with max quality. """
            original = Image.open( source )
            original.save( dest.replace('.png', '.jpg'), 'JPEG', quality=95 )
            return None
        
        # Sanity check -- if scale_factor is not int...
        try:
            scale_factor = int( scale_factor )
        except:
            scale_factor = 1
            
        images = []
        for f in os.listdir( self.dbg ):
            m = regex.match( f )
            if m:
                matchdict         = m.groupdict()
                matchdict['file'] = f
                matchdict['dt']   = datetime.datetime( *[int(d) for d in matchdict['dt'].split('_')] )
                images.append( matchdict )
                
                orig_path = os.path.join( self.dbg, f )
                new_path  = os.path.join( self.results_dir, f )
                
                # Reduce image size if requested.  We will ignore requests to "blow up" the image.
                if scale_factor > 1:
                    resize( orig_path, new_path, scale_factor )
                else:
                    os.system( 'cp {} {}'.format( orig_path , new_path ) )

                # Convert to jpg if requested.  jpg used to retain some quality and higher resolution but one
                #   which will not be transferred into the CSA.  used for deck images only.
                if convert_to_jpg:
                    make_jpg( orig_path , new_path )
                    
        return images
    
    def prepare_chip_images( self ):
        """ Searches for and prepares html page for chip images taken at different points during the run. """
        ci_re = re.compile( """chipImage_(?P<process>[\D]+)_(?P<dt>[\d_]+)_(?P<lane>[1-4]{1}).jpg""" )
        chip_images = self.find_debug_images( ci_re )
        chip_images = sorted( chip_images, key=lambda x: (x['dt'], x['lane']) )
        # Save a boolean to test if we had any images.  Will use in block html.
        self.found_chip_images = len( chip_images ) > 0
        
        if not self.found_chip_images:
            print( 'No debug chip images were found!  Skipping html page creation.' )
            return None
        
        # Start on html page.
        doc    = H.Document( )
        
        styles = H.Style( )
        styles.add_css  ( 'img', {'max-width': '100%',
                                  'width'    : '100%' } )
        doc.add         ( styles )
        
        header = H.Header  ( 'Chip Images Collected During the Workflow' , 3 )
        doc.add( header )
        doc.add( H.Paragraph(
            'Each column of lane images is taken while centering that particular lane within the ADC window.' ) )
        doc.add( H.Break() )
        
        # Table time
        image_arr = [ chip_images[i:i+4] for i in range(0, len(chip_images), 4)]
        table     = H.Table( border='0' )
        row       = H.TableRow( )
        for title in ['&nbsp','Lane 1','Lane 2', 'Lane 3', 'Lane 4']:
            row.add_cell( H.TableCell( title, True, width='20%', align='center' ) )
        table.add_row( row )
        
        for img_row in image_arr:
            row   = H.TableRow ( )
            label = H.TableCell( textwrap.dedent( """\
            <p>{}</p>
            <p>{}</p>
            """.format( img_row[0]['process'], img_row[0]['dt'].strftime( '%d %b %Y | %X' ))),
                                                  True, width='20%', align='center' )
            row.add_cell( label )
            for img in img_row:
                # This was previous method
                #row.add_cell( H.TableCell( H.Image( img['file'] ).as_link() ) )
                
                # Now show low res image but link to full res image in rawdata.
                full_img = os.path.join( self.dbg , img['file'] )
                link     = H.Link( full_img )
                link.add_body( H.Image( img['file'] ) )
                row.add_cell ( H.TableCell( link ) )
                
            table.add_row( row )
            
        doc.add( table )
        with open( os.path.join( self.results_dir, 'chipImages.html' ), 'w' ) as f:
            f.write( str( doc ) )
        
        # Lane wetting detection - compare Harpoon_start to Run_start for all unused lanes
        #self.lane_wetting_detection( image_arr ) # incomplete, do not run

    def lane_wetting_detection( self, chip_images ):
        '''Adapted from Shawn W.'s script'''
        print( 'chip_images' )
        print( chip_images )
        # First determine the lowest active lane
        smallest_active_lane = 1
        for lane in [1,2,3,4]:
            if self.explog.metrics['LanesActive{}'.format(lane)]:
                smallest_active_lane = lane
        print( 'smallest_active_lane', smallest_active_lane )
        for lane in [1,2,3,4]:
            if lane < smallest_active_lane:
                continue
            left_boundary = int(800*(lane-1)/4)
            right_boundary = int(800*(lane)/4)
            print(lane, left_boundary, right_boundary)
            # select the Harpoon_start and Run_start images
            run_start_fileName  = chip_images[0][lane-1]['file']
            harp_start_fileName = chip_images[1][lane-1]['file']
            print( run_start_fileName , harp_start_fileName )
            #run_start_original = cv2.imread( run_start_fileName )
            #harp_start_original = cv2.imread( harp_start_fileName )

            #if run_start_original is not None and harp_start_original is not None:
                # crop
            #    before = run_start_original[0:600,left_boundary:right_boundary]
            #    after  = harp_start_original[0:600,left_boundary:right_boundary]
            


    def prepare_deck_images( self , scale_factor=1 ):
        """ 
        Searches for and prepares html page for deck images taken during the run. 
        scale factor is a divisor for the side of the images that resizes them.  
        -  e.g. a scale_factor of 2 will reduce w and h by 50%, leading to a 4x reduction in size.
        """
        di_re       = re.compile( """deckImage_(?P<process>[\w]+)_(?P<site>[0-9]{1})_(?P<dt>[\d_]+).png""" )
        deck_images = self.find_debug_images( di_re , scale_factor=scale_factor , convert_to_jpg=True )
        deck_images = sorted( deck_images, key=lambda x: (x['dt'], x['site']) )
        
        # Save a boolean to test if we had any images.  Will use in block html.
        self.found_deck_images = len( deck_images ) > 0
        
        if not self.found_deck_images:
            print( 'No debug deck images were found!  Skipping html page creation.' )
            return None
            
        # Start on html page.
        doc    = H.Document( )
        
        styles = H.Style( )
        styles.add_css  ( 'img', {'max-width': '100%',
                                  'width'    : '100%' } )
        doc.add         ( styles )
        
        header = H.Header  ( 'Deck Images Collected During the Workflow' , 3 )
        doc.add( header )
        doc.add( H.Paragraph(
            'Each row of images is taken at different points during the workflow from each of the cameras.' ) )
        doc.add( H.Break() )
        
        # Table time
        image_arr = [ deck_images[i:i+3] for i in range(0, len(deck_images), 3)]
        table     = H.Table( border='0' )
        row       = H.TableRow( )
        for title in ['&nbsp','Camera 1','Camera 2', 'Camera 3']:
            row.add_cell( H.TableCell( title, True, width='25%', align='center' ) )
        table.add_row( row )
        
        for img_row in image_arr:
            row   = H.TableRow ( )
            label = H.TableCell( textwrap.dedent( """\
            <p>{}</p>
            <p>{}</p>
            """.format( img_row[0]['process'], img_row[0]['dt'].strftime( '%d %b %Y | %X' ) ) ),
                                                  True, width='25%', align='center' )
            row.add_cell( label )
            for img in img_row:
                row.add_cell( H.TableCell( H.Image( img['file'] ).as_link() ) )
                
            table.add_row( row )
            
        doc.add( table )
        with open( os.path.join( self.results_dir, 'deckImages.html' ), 'w' ) as f:
            f.write( str( doc ) )
            
    def analyze_flow_rate( self , flow_rate_margin=0.1 ):
        """ 
        Analyzes the flow rate from the instrument depending on the number of lanes really being run 
        By default, we set the acceptable range within 10% of the target flow rate (flow_rate_margin)
        """
        # Initialize metrics
        
        
        # First we have to do some detective work and make sure we know how many lanes are active.
        # For now, let's try using the explog_final.json file
        exp = {}
        try:
            with open( os.path.join( self.raw_data_dir , 'explog_final.json' ), 'r' ) as f:
                exp = json.load( f , strict=False )
        except:
            with open( os.path.join( self.calibration_dir , 'explog_final.json' ), 'r' ) as f:
                exp = json.load( f , strict=False )
                
        if not exp:
            print( "Could not find the explog_final.json file in order to count true number of active lanes." )
            print( "As such, skipping detection of flow rate deviations." )
            return None

        # Calculate the target based on true active.  self.explog will be fake on DualAnalysis
        def is_active( string ):
            """ Helper function to deal with yes/no in the LanesActive# field. """
            return string.lower() in ['yes','true','y']
        
        try:
            true_active = sum( [ is_active( exp['LanesActive{}'.format(i)] ) for i in [1,2,3,4] ] )
            print( 'Detected the true number of active lanes: {:d} . . .'.format( true_active ) )
            self.true_active = true_active # will need in FlowInfo class 
            target_flow = 48. * true_active
            flow_range  = target_flow * np.array( (1.0-flow_rate_margin, 1.0+flow_rate_margin ), float )
            
            # Let's get only the data we want from the text explog with flow data
            x        = self.explog.flowax[   self.explog.seq_flow_mask ]
            data     = self.explog.flowrate[ self.explog.seq_flow_mask ]
            outliers = np.logical_or( data > flow_range.max() , data < flow_range.min() )
            if outliers.any():
                first = np.where( outliers )[0][0]
                print( 'Found {} flow rate outliers!  First deviant flow: acq_{:04d}!'.format( outliers.sum(), first ))
            else:
                first = None
                
            if true_active == 0:
                avg_per_lane = data.mean()
            else:
                avg_per_lane = data.mean() / true_active
            print('Length of flow rate array: {}'.format(len(x)))
            if self.do_reseq:
                flow_total = len(x)
            else:
                flow_total = float(self.explog.flows)
            print( 'Average flow rate: {:.1f} uL/s'.format( data.mean() ) )
            flow_rate_metrics = { 'mean'         : data.mean() ,
                                  'std'          : data.std() ,
                                  'outliers'     : outliers.sum() ,
                                  'perc_outliers': 100. * float(outliers.sum()) / flow_total ,
                                  'first'        : first ,
                                  'avg_per_lane' : avg_per_lane , 
                                  'true_active'  : true_active }
            
            self.metrics['flow_rate'] = flow_rate_metrics
            
            # Also, make a sparkline.
            fig   = plt.figure( figsize=(8,0.5) )
            gs    = gridspec.GridSpec( 1, 1, left=0.0, right=1.0, bottom=0.05, top=0.95 )
            spark = sparklines.Sparkline( fig, gs[0], 8 )
            
            label = sparklines.Label( r'$\overline{Q}=%.1f \  \frac{\mu L}{s}$' % data.mean() , width=3 , fontsize=16 )
            spark.add_label( label, 'left' )
            
            # If we have outliers, let's color code it red.
            if outliers.sum() > 0:
                color = 'red'
            else:
                color = 'green'
            num   = sparklines.Label( r'$%d$' % outliers.sum() , width=1 , fontsize=16 , color=color )
            spark.add_label( num , 'right' )
            spark.create_subgrid()
            spark.ax.axis   ( 'off' )
            spark.ax.plot   ( x , data , '-' , linewidth=0.75 )
            if self.do_reseq:
                num_flows = int(self.explog.flows)
                spark.ax.fill_between( np.arange(num_flows), flow_range[0] , flow_range[1], color='green' , alpha=0.4, linewidth=0.0 )
                spark.ax.fill_between( np.arange(num_flows+13,x[-1],1), flow_range[0] , flow_range[1], color='green' , alpha=0.4, linewidth=0.0 )
            else:
                spark.ax.axhspan( flow_range[0] , flow_range[1] , color='green' , alpha=0.4 )
            spark.ax.axhline( target_flow , ls='--', color='green', linewidth=0.5 )
            spark.draw      ( )
            
            fig.savefig( os.path.join( self.results_dir , 'flow_spark.svg' ), format='svg' )
        except:
            print('!! Something went wrong when making flow rate sparkline') 
            self.metrics['plugin_error'] = True
            
    def analyze_workflow_timing( self, do_reseq=False ):
        """ 
        Does original large scale analysis.
        Now also does analysis based on ScriptStatus.csv, if the file is available.
        """
        self.debug.parallel_grep  ( )

        # Get timing and adjust if needed.
        self.debug.get_overall_timing(reseq=do_reseq)
        
        if not self.debug.modules['libprep']:
            # Let's deal with mag2seq
            if (not self.debug.modules['harpoon']) and self.debug.modules['magloading']:
                if self.debug.timing['templating_start'] == None:
                    self.debug.timing['templating_start'] = self.expt_start
                    
                # If there's no templating end, then we can't measure dead time....I think.
                if self.debug.timing['templating_end'] == None:
                    self.debug.timing['templating_end'] = self.debug.timing['sequencing_start']
                    
        if self.debug.timing['sequencing_end'] == None:
            self.debug.timing['sequencing_end'] = self.explog.end_time 

        # Plot workflow and gather timing metrics
        timing_metrics = self.debug.plot_workflow( os.path.join( self.results_dir, 'workflow_timing.png' ) )
        
        # Save some metrics to results.json
        self.metrics['timing' ]     = timing_metrics
        
        # Set up default values
        self.metrics['run_type'] = 'unknown'
        run_types = [ 'end_to_end', 'harp_to_seq', 'mag_to_seq', 'coca_to_seq', 'mag_to_temp', 'seq_only' ]
        for rt in run_types:
            self.metrics[ rt ] = False
            
        mods = self.debug.modules # shortcut
        self.metrics['modules'] = mods
        #if all( mods.values() ): # no longer works since resequencing was added as a module
        if all( [ mods['libprep'], mods['harpoon'], mods['magloading'], mods['coca'], mods['sequencing'] ] ):
            self.metrics['end_to_end']  = True
            self.metrics['run_type']    = 'End to End'
        elif all([ mods['harpoon'], mods['magloading'], mods['coca'], mods['sequencing'] ]):
            self.metrics['harp_to_seq'] = True
            self.metrics['run_type']    = 'Harpoon to Seq.'
        elif all([ mods['magloading'], mods['coca'], mods['sequencing'] ]):
            self.metrics['mag_to_seq']  = True
            self.metrics['run_type']    = 'MagLoading to Seq.'
        elif all([ mods['coca'], mods['sequencing'] ]):
            self.metrics['coca_to_seq'] = True
            self.metrics['run_type']    = 'COCA to Seq.'
        elif all([ mods['magloading'], mods['coca'] ]):
            self.metrics['mag_to_temp'] = True
            self.metrics['run_type']    = 'MagLoading to Templating'
        elif mods['sequencing'] and not any([ mods['libprep'], mods['harpoon'], mods['magloading'], mods['coca'] ]): # change all to any
            print('Classifying run as sequencing only')
            self.metrics['seq_only'] = True
            self.metrics['run_type'] = 'Sequencing Only'
            
        # ScriptStatus lives in "rawdata" folder with calibration files, not acq files.
        ss_file = os.path.join( self.calibration_dir, 'ScriptStatus.csv' )
        if not os.path.exists( ss_file ):
            print( 'Error finding ScriptStatus.csv file!' )
            self.ss_file_found = False
            return None
        else:
            self.ss_file_found = True
            
        try:
            # Add in postrun timing before we update_data
            self.ss = ScriptStatus( ss_file, self.explog.start_time )
            self.ss.read()
            self.ss.update_data( )
            self.ss.add_overall_timing( self.debug.timing, self.metrics['run_type'], reseq=do_reseq )
            self.ss.add_postrun_modules( self.postchip_clean_timing, self.postrun_clean_timing )
            
            # Make figures
            self.ss.submodule_pareto( self.results_dir, count=5 )
            self.ss.full_timing_plot( self.results_dir , self.init_timing , self.analysis_timing, reseq=do_reseq )
            
            # Save metrics for durations, used tips.  Some of these are copied to the overall results.json
            if mods['resequencing']:
                seq_done = self.ss.get_relative_time( self.ss.data['resequencing']['overall']['end'] )
                self.seq_end = self.ss.data['resequencing']['overall']['end'] # for libPrepLog
            else:
                seq_done = self.ss.get_relative_time( self.ss.data['sequencing']['overall']['end'] )
                self.seq_end = self.ss.data['sequencing']['overall']['end'] # for libPrepLog
            
            ss_metrics = { 'used_tips': 0 }
            
            # The other metrics required for the program are time to basecaller completion and sample analysis compl.
            self.metrics['time_to_seq_done']     = seq_done
            try:
                self.metrics['time_to_postrun_done'] = self.ss.get_relative_time(self.ss.data['postrun']['overall']['end'])
            except: # no POSTRUN
                self.metrics['time_to_postrun_done'] = None
            if self.analysis_timing:
                try:
                    self.metrics['time_to_basecaller_done'] = self.ss.get_relative_time( self.analysis_timing['BaseCallingActor']['end'] )
                except:
                    self.metrics['time_to_basecaller_done'] = None # for when BaseCallingActor fails
                try:
                    self.metrics['time_to_samples_done'] = self.ss.get_relative_time( self.analysis_timing['Samples']['end'] )
                except:
                    self.metrics['time_to_samples_done'] = None # for case where sample summary files are missing. 
            for module in self.ss.data:
                m           = self.ss.data[module]
                module_time = m['overall']['duration']
                module_tips = m['overall']['used_tips']
                ss_metrics[module]       = { 'duration': module_time, 'used_tips': module_tips }
                ss_metrics['used_tips'] += module_tips
                # Add submodules
                if 'submodules' in m:
                    for sm in m['submodules']:
                        sm_time = m['submodules'][sm]['duration']
                        sm_tips = m['submodules'][sm].get('used_tips',0)
                        ss_metrics[module][sm] = {'duration': sm_time, 'used_tips': sm_tips }
                        
            self.metrics['script_status'] = ss_metrics
        except:
            print('!! Something went wrong when analyzing the ScriptStatus file or making timing plot. Skipping analysis')
            self.ss_file_found = False
            self.metrics['plugin_error'] = True
            return None
                    
    def analyze_tube_bottoms( self ):
        """ 
        Analyzes the debug csv that checks tubes for their bottom location (only occurs when running BottomFind) 
        """
        print('Analyzing tube bottom log')
        self.metrics['bottomlog'] = {}
        missed    = [] # original metrics. total, both pipettes together
        bent_tips = []
       
        # New metrics separated by pipette id
        pip1_all        = [] # might be useful to know total number of bottom finds per pipette
        pip1_missed     = []
        pip1_bent_tips  = []
        pip2_all        = []
        pip2_missed     = []
        pip2_bent_tips  = []
        
        try: 
            tbl = os.path.join( self.calibration_dir, 'TubeBottomLog.csv' )
            if os.path.exists( tbl ):
                self.has_tube_bottom_log = True
                with open( tbl, 'r' ) as f:
                    fnames = ['tube','pipette','missed_bottom','bent_tip','zcal','bottom_found'] 
                    reader = csv.DictReader( f, fieldnames=fnames )
                    for row in reader:
                        # Skip over rows if they are from the 'liq_waste_01' tube until Shawn fixes this line.
                        if row['tube'] == 'liq_waste_01':
                            continue
                        
                        try:
                            # This line ignores other lines in the CSV that do not have a zcal value.
                            _ = float( row['zcal'] )
                            
                            # Old metrics: not counting warnings, but only the most extreme.
                            if 'Bent tip' in row['bent_tip']:
                                bent_tips.append( row['tube'] )
                            if 'Not reaching bottom' in row['missed_bottom']:
                                missed.append( row['tube'] )
                            
                            # New metrics looking at each attempt in each line  
                            attempts = row['bottom_found'].replace('[','').replace(']','')
                            attempts = [float(attempt) for attempt in attempts.split()]
                            for attempt in attempts:
                                if attempt < -2:
                                    if row['pipette']=='1':
                                        pip1_missed.append( row['tube'] )
                                    else:
                                        pip2_missed.append( row['tube'] )
                                if attempt > 2:
                                    if row['pipette']=='1':
                                        pip1_bent_tips.append( row['tube'] )
                                    else:
                                        pip2_bent_tips.append( row['tube'] )
                                # To get total
                                if row['pipette']=='1':
                                    pip1_all.append( row['tube'] )
                                else:
                                    pip2_all.append( row['tube'] )

                        except( ValueError, TypeError ):
                            # Don't care, must not have been a row with a real zcal value and thus other useful info.
                            pass
                        
                # Summarize metrics
                self.metrics['bottomlog'] = { 'missed_bottom'          : ', '.join( missed ),
                                              'missed_bottom_count'    : len( missed ),
                                              'bent_tips'              : ', '.join( bent_tips ),
                                              'bent_tips_count'        : len( bent_tips ),
                                              'missed_bottom_p1'       : ', '.join( pip1_missed ),
                                              'missed_bottom_p1_count' : len( pip1_missed ), 
                                              'bent_tips_p1'           : ', '.join( pip1_bent_tips ),
                                              'bent_tips_p1_count'     : len( pip1_bent_tips ) , 
                                              'total_p1'               : len( pip1_all ) ,
                                              'missed_bottom_p2'       : ', '.join( pip2_missed ),
                                              'missed_bottom_p2_count' : len( pip2_missed ), 
                                              'bent_tips_p2'           : ', '.join( pip2_bent_tips ),
                                              'bent_tips_p2_count'     : len( pip2_bent_tips ) , 
                                              'total_p2'               : len( pip2_all ) ,
                                              }
                print('> 2 mm above zcal (missed bottom)')
                print('Pipette 1 :  {}'.format(self.metrics['bottomlog']['missed_bottom_p1_count']))
                print('Pipette 2 :  {}'.format(self.metrics['bottomlog']['missed_bottom_p2_count']))
                
                print('> 2 mm below zcal (bent tip)')
                print('Pipette 1 :  {}'.format(self.metrics['bottomlog']['bent_tips_p1_count']))
                print('Pipette 2 :  {}'.format(self.metrics['bottomlog']['bent_tips_p2_count']))
                
                print('Total tube bottom finds')
                print('Pipette 1 :  {}'.format(self.metrics['bottomlog']['total_p1']))
                print('Pipette 2 :  {}'.format(self.metrics['bottomlog']['total_p2']))
                
                # Make a symlink to the raw file for easy access
                if not os.path.exists( os.path.join( self.results_dir, 'TubeBottomLog.csv' ) ):
                    os.symlink( tbl , os.path.join( self.results_dir, 'TubeBottomLog.csv' ) )
                else:
                    print( 'symlink TubeBottomLog.csv already exists.' )
            else:
                self.has_tube_bottom_log = False
                print( 'Could not find the TubeBottomLog.csv file.  Skipping analysis.' )
        except:
            print('!! Something went wrong when analyzing TubeBottomLov.csv. Skipping analysis')
            self.has_tube_bottom_log = False
            self.metrics['plugin_error'] = True

    def find_pipette_failed_to_pickup_tip( self ):
        '''
        Search for er75, which means pipette failed to pick up a tip. After 5 attempts, pipette will try tip at different location 
        '''
        ############################## Define helper functions
        def new_tip_loc( matches ):
            if len(matches) >= 2:
                if 'tip_loc' in matches[-2]: # matches[-1] is the one we are currently evaluating
                    return True              # 'UNABLE TO PICKUP TIP' line is the last line printed before moving to a different chip location
                time_delta = ( get_timestamp(matches[-1]) - get_timestamp(matches[-2]) ).seconds
                print('       time_delta: {}'.format(time_delta))
                if time_delta < 4:
                    return False # if timestamp is within 4 seconds of previous, then assume the attempt is on the same tip location
                else:
                    return True 
            else:
                return False # to handle first er75 event 
        
        def get_timestamp( match_dict ):
            mon = match_dict['mon']
            day = match_dict['day']
            time = match_dict['time']
            timestring = '{} {} {} {}'.format( datetime.datetime.now().year, mon, day, time ) 
            return datetime.datetime.strptime( timestring, '%Y %b %d %H:%M:%S' )

        def get_previous( matches ):
            ''' matches[-1] corresponds to the pipette failured attempt in the new location. We want info from the previous '''
            if len(matches) < 2: # handle case where there was only one failed attempt and we are looking at that one failed attempt outside the loop.
                tip_loc = 'Unknown'
                pip_id  = matches[-1]['pipette_id']
                return tip_loc, pip_id
            if 'tip_loc' in matches[-2]:            # we should expect to have at least 5 entries in matches if this is true
                tip_loc = matches[-2]['tip_loc']    # we only know the tip loc when 'UNABLE' line is printed after it gives up on that tip location
                pip_id  = matches[-3]['pipette_id']         # pipette ID will not be in 'UNABLE' line, so must look to the previous
            else:
                tip_loc = 'Unknown'
                pip_id = matches[-2]['pipette_id']
            return tip_loc, pip_id
        ##############################
        print('Searching for er75 or er62, which means pipette failed to pickup tip')
        try:
            self.search_for_tip_pickup_errors = True
            regex_GTid   = re.compile( r"/var/log/[\w.]+:(?P<mon>[\w]+)[ ]+(?P<day>[\d]+)[ ](?P<time>[\d:]+) [\w:.\-_ ]+ pipette recv id = (?P<pipette_id>[\d]), ret = GTid(?P<number>[\d]+)er(?P<error>[^0]+)" ) 
            regex_giveup = re.compile( r"/var/log/[\w.]+:(?P<mon>[\w]+)[ ]+(?P<day>[\d]+)[ ](?P<time>[\d:]+) [\w:.\-_ ]+ UNABLE TO PICKUP TIP AT (?P<tip_loc>[\w_]+).  RETRYING AT A DIFFERENT LOCATION" )
            matches  = [] # list of groupdicts for all lines that match
            tip_locs = [] 
            attempt_count  = 0
            lines = self.debug.all_lines
            for line in lines:
                match = regex_GTid.match( line )
                match_giveup = regex_giveup.match( line )
                if match:
                    print(line)
                    matches.append(match.groupdict())
                    if new_tip_loc( matches ):
                        print('       Failed attempt is at a New Tip Loc. Log previous attempt set')
                        try:
                            tip_loc, pip_id = get_previous( matches )  
                            tip_loc_dict = {'tip_loc': tip_loc, 'failed_attempts': attempt_count, 'pipette': pip_id }
                            print('       {}'.format(tip_loc_dict))
                            tip_locs.append( tip_loc_dict ) 
                        except:
                            print('       Something went wrong when searching for pip_id and tip_loc for previous attempt set. Skipping...')
                        attempt_count = 1 # reset attempt counter to 1 
                    else:
                        attempt_count += 1
                if match_giveup:
                    print(line)
                    matches.append( match_giveup.groupdict() )

            # Handle very last failed attempt
            if len(matches) > 0:
                print('Lets log the very last failed attempt group')
                tip_loc, pip_id = get_previous( matches )
                tip_loc_dict = {'tip_loc': tip_loc, 'failed_attempts': attempt_count, 'pipette': pip_id }
                print('       {}'.format(tip_loc_dict))
                tip_locs.append( tip_loc_dict )
            
            # Save metrics to display
            self.struggle_to_pickup_tips = {}
            self.unable_to_pickup_tips = {}
            for id in ['1','2']:
                num_attempts = [] # only cases where pipette was eventually successful
                unable_to_pickup = []
                for tip_loc in tip_locs:
                    if tip_loc['pipette']==id:
                        if tip_loc['tip_loc']=='Unknown':
                            num_attempts.append(tip_loc['failed_attempts'])
                        else:
                            unable_to_pickup.append(tip_loc['tip_loc'])
                        
                num_attempts = np.asarray(num_attempts)
                self.struggle_to_pickup_tips['pipette_{}'.format(id)] = {'tip_loc_count':len(num_attempts),'failed_attempt_avg':'{:.1f}'.format(np.average(num_attempts))}
                self.unable_to_pickup_tips['pipette_{}'.format(id)] = unable_to_pickup
            print(self.struggle_to_pickup_tips)
            print(self.unable_to_pickup_tips)

            self.metrics['tip_pickup_errors'] = {'p1':{}, 'p2':{}}
            for id in ['1','2']:
                self.metrics['tip_pickup_errors']['p{}'.format(id)]['struggle_count'] = self.struggle_to_pickup_tips['pipette_{}'.format(id)]['tip_loc_count']
                self.metrics['tip_pickup_errors']['p{}'.format(id)]['struggle_attempt_avg'] = self.struggle_to_pickup_tips['pipette_{}'.format(id)]['failed_attempt_avg']
                self.metrics['tip_pickup_errors']['p{}'.format(id)]['unable_count'] = len(self.unable_to_pickup_tips['pipette_{}'.format(id)])
                self.metrics['tip_pickup_errors']['p{}'.format(id)]['unable_locs'] = ', '.join(self.unable_to_pickup_tips['pipette_{}'.format(id)])
        except:
            print('!! Something went wrong when searching for tip pickup errors, such as er75 and er62. Skipping Analysis.')
            self.search_for_tip_pickup_errors = False
            self.metrics['plugin_error'] = True

    def find_pipette_errors( self ):
        '''
        Function to determine if either pipette experienced an ERROR 52. This error means that the pipette did not complete its aspiration.
        The pipette may not have aspirated or dispensed everything it was supposed to. When one er52 happens, more are likely to follow.
        Now adding in other general pipette errors.
        Oct-2020 adding pipette timeout errors
        '''
        print('Seaching for er52 and other pipette errors in debug log...')
        er52 = {'pipette_1':0, 'pipette_2':0} # initiate dictionary containing times of er52 errors - now only used to make results.json
        pipette_errors = { 'pipette_1':{'count':0, 'messages':[]}, 'pipette_2':{'count':0, 'messages':[]} } # for all pipette errors, including er52. Use in block html
        self.search_for_pipette_errors = True 
        try:
            general_error_regex = re.compile( r"[\w:./\-_ ]+ ValueError: ERROR: (?P<function>[\w]+) pipette (?P<pipette_id>[\d]) returned error = (?P<error_str>[\w:\- !=<>.]+)"  ) # does not identify er52
            timeout_to_error_regex = re.compile( r"[\w:./\-_ ]+ pipette timedout waiting for response (?P<function>[\w,]+) id = (?P<pipette_id>[\d]), pipCmd = (?P<pipCmd>[\w]+)"  ) # only for pipette timout errors
            timeout_from_error_regex = re.compile( r"[\w:./\-_ ]+ pipette timedout waiting for response (?P<function>[\w,]+) id = (?P<pipette_id>[\d]) for response to (?P<pipCmd>[\w]+)"  ) # only for pipette timout errors
            print('About to do parellel grep for pipette errors...') 
            #self.debug.parallel_grep( filter_on_starttime=False ) # just to test
            lines = self.debug.all_lines
            for line in lines:
                # First check if line is from er52
                if ('er52' in line) and ('pipette response' in line):
                    print( 'Line flagged for er52: {}'.format(line) )
                    try:
                        pipette_id = int(line.split('=')[1].split(',')[0])
                        er52['pipette_{}'.format(pipette_id)]+=1
                        message = 'er52'
                        pipette_errors['pipette_{}'.format(pipette_id)]['count']+=1
                        pipette_errors['pipette_{}'.format(pipette_id)]['messages'].append( message )
                    except:
                        print('WARNING: found er52 in line above, however could not identify pipette id')
                
                if 'ValueError' in line:
                    print( 'Line flagged for pipette error: {}'.format(line) )

                # See if the line matches the regex for the format of the generic pipette error    
                match = general_error_regex.match( line )
                if match:
                    function = match.groupdict()['function']
                    message = match.groupdict()['error_str']
                    pipette_id = str(match.groupdict()['pipette_id'])
                    pipette_errors['pipette_{}'.format(pipette_id)]['count']+=1
                    pipette_errors['pipette_{}'.format(pipette_id)]['messages'].append( '{}: {}'.format(function,message) )
                
                if 'timedout' in line:
                    match_to   = timeout_to_error_regex.match( line ) # two different types of timedout lines require two different regex and message
                    match_from = timeout_from_error_regex.match( line )
                    if match_to:
                        match = match_to
                        pipCmd =  match.groupdict()['pipCmd'][0:4]
                        if pipCmd in ['RFid','ZIid']:
                            pass # fine with these types 
                        else:
                            print('Timedout line: {}'.format(line))
                            pipette_id = str(match.groupdict()['pipette_id'])
                            message = 'pipette timedout waiting for response {} pipCmd = {}'.format(match.groupdict()['function'].replace(',',''),match.groupdict()['pipCmd'])
                            pipette_errors['pipette_{}'.format(pipette_id)]['count']+=1
                            pipette_errors['pipette_{}'.format(pipette_id)]['messages'].append( '{}'.format(message) )
                    if match_from:
                        match = match_from
                        pipCmd =  match.groupdict()['pipCmd'][0:4]
                        if pipCmd in ['RFid','ZIid']:
                            pass
                        else:
                            print('Timedout line: {}'.format(line))
                            pipette_id = str(match.groupdict()['pipette_id'])
                            message = 'pipette timedout waiting for response {} pip for response to {}'.format(match.groupdict()['function'].replace(',',''),match.groupdict()['pipCmd'])
                            pipette_errors['pipette_{}'.format(pipette_id)]['count']+=1
                            pipette_errors['pipette_{}'.format(pipette_id)]['messages'].append( '{}'.format(message) )
            
            self.metrics['er52'] = er52

            # All this just to get the messages into the right format for results.json
            p1_count = pipette_errors['pipette_1']['count']
            p2_count = pipette_errors['pipette_2']['count']
            p1_mes   = ', '.join( pipette_errors['pipette_1']['messages'] )
            p2_mes   = ', '.join( pipette_errors['pipette_2']['messages'] )
            self.pipette_errors = { 'pipette_1': {'count': p1_count, 'messages': p1_mes}, 
                                    'pipette_2': {'count': p2_count, 'messages': p2_mes} }
            self.metrics['pipette_errors'] = self.pipette_errors
        except:
            print('!! Something went wrong when searching for pipette errors, such as er52. Skipping Analysis.')
            self.search_for_pipette_errors = False
            self.metrics['plugin_error'] = True

    def find_pipette_serialNumbers( self ):
        '''
        Searches debug for pipette serial numbers.
        '''
        print('Searching for pipette serial numbers...')
        sns = {'1': None, '2':None}
        self.metrics['serial_number_pip1']= sns['1'] # will update if found
        self.metrics['serial_number_pip2']= sns['2']
        try:
            regex = re.compile( r'[\w:./\-_ ]+ python: pipette: recv id = (?P<id>[\d]), ret = (?P<part1>[\w\-]+)/(?P<part2>[\w]+)/(?P<serial>[\d]+)')
            # need to grep whole debug since lines we need are before start of run
            print('About to do parellel grep for pipette serial numbers...') 
            self.debug.parallel_grep( filter_on_starttime=False )
            lines = self.debug.all_lines
            for i,line in enumerate(lines):
                match = regex.match(line)
                if match:
                    matchdict = match.groupdict()
                    sns[matchdict['id']] = matchdict['serial'] 

            self.metrics['serial_number_pip1']= sns['1']
            self.metrics['serial_number_pip2']= sns['2']
            print(sns)
        except:
            print('!! Something went wrong when searching for pipette serial numbers.')
            self.metrics['plugin_error'] = True

    def analyze_pipette_behavior( self ):
        """ Reads the debug log to extract the time pipettes spend on mixing steps
            and the order that they were used
        
            As of 16Sep2019 -- only looking at 'mix well to denature' step in 
                'Denature Harpooned Beads from MyOnes' process.
                There are 2 'mix well' steps in this process.
        
            Data is stored as 
                pipette_mixing -> block<block_num> -> p<pipette_num>_<use> -> {target:..., elapsed:.., timestamp:.., etc.}
        
            This yields the timing associated with each use of the pipettes for each block of processing.
            That information should be sufficient to tie a pipette's timing to a specific lane.
        """
        print( '-------STARTING: Analyzing Pipette Behavior-------' )
        try:
            # initialize metric storage
            pipette_behavior = {}
            
            # get harpooon behavior
            pipette_behavior['harpoon_mixing'] = self.analyze_pipette_behavior_HARPOON()
            # get magloading
            pipette_behavior['foam_scrape'] = self.analyze_pipette_behavior_FOAM_SCRAPE()
            
            # store metrics
            self.metrics['pipette_behavior'] = pipette_behavior

            print( '-------COMPLETE: Analyzing Pipette Behavior-------' )
        except:
            print( '!! Something went wrong when analyzing pipette behavoir. Skipping Analysis.')
            self.metrics['plugin_error'] = True

    #############################################
    #   BEGIN -- HELPERS FOR PIPETTE BEHAVIOR   #
    #############################################
    
    def analyze_pipette_behavior_HARPOON( self ):
        # Check if harpoon module was completed
        # If not --> stop further analysis
        print( 'Starting -- Harpoon analysis')
        if not self.debug.modules['harpoon']: return

        # Get the relevant lines from the debug log
        # --> For the harpoon module, it will be two blocks between 'mix well to denature' and 'ditch tip'
        process_start   = r'.*#Process: \*\*\* mix well to denature.*'
        process_stop    = r'.*#Process: \*\*\* ditch tips.*'
        relevant_lines  = [ r'Mixing: Pipette [0-9] Aspirate',
                            r'Mix: totalTime',
                            ]
        
        # Get blocks of lines for further parsing
        blocks = self.debug.search_blocks( process_start, process_stop, *relevant_lines, case_sensitive=True )
        print( 'blocks', blocks )
        print( 'Complete -- Harpoon analysis')

        return self.parse_pipette_mixing_time_blocks( blocks )
    
    
    def analyze_pipette_behavior_FOAM_SCRAPE( self ):
        # Check if harpoon module was completed
        # If not --> stop further analysis

        print( 'Starting -- Foam Scrape analysis')
        if not self.debug.modules['magloading']: return
        
        # Get the relevant lines from the debug log
        # --> For the harpoon module, it will be two blocks between 'mix well to denature' and 'ditch tip'
        process_start   = r'.*#Process:---Foam generation and foam.*'
        process_stop    = r'.*#Process:---60/40 AB/IPA.*'
        relevant_lines  = [ r'Mixing: Pipette [0-9]',
                            r'Mix: totalTime',
                            r'Dispense: totalTime',
                            ]
        
        # Get blocks of lines for further parsing
        blocks = self.debug.search_blocks( process_start, process_stop, *relevant_lines, case_sensitive=True )
        print( 'blocks', blocks )
        print( 'Complete -- Foam Scrape analysis')

        return self.parse_pipette_mixing_time_blocks( blocks )
    
    # NOTE: This seems to handle most of what we want from pipette timing
    def parse_pipette_mixing_time_blocks( self, blocks ):
        ''' 
        Takes in blocks for pipette mixing times and outputs a dictionary
        
        NOTE:  Requires the following grep strings in the generation of the blocks
        
        r'Mixing: Pipette [0-9]'
        r'Mix: totalTime'
        '''
        # parse each block for pipette timing information
        regex_mixing    = re.compile( r'.*: Mixing: Pipette (?P<pipette>[0-9]) (?P<type>\w+) (?P<output_volume>[0-9]+)ul and mix (?P<mix_volume>[0-9]+)ul for (?P<num_cycles>[0-9]+) cycles at (?P<mix_rate>[0-9]+)ul/sec.*' )
        dispense_timing = re.compile( r'.*: Dispense: totalTime = (?P<totalTime>[0-9]+[.][0-9]+) estTime = (?P<estTime>[0-9]+[.][0-9]+), elapsedTime = (?P<elapsedTime>[0-9]+[.][0-9]+).*' )
        mix_timing      = re.compile( r'.*: Mix: totalTime = (?P<totalTime>[0-9]+[.][0-9]+) estTime = (?P<estTime>[0-9]+[.][0-9]+), elapsedTime = (?P<elapsedTime>[0-9]+[.][0-9]+).*' )
        
        block_dict = {}
        for i,b in enumerate( blocks ):
            block_name = 'block{}'.format(i)
            block_dict[block_name]={}
            use = {'A':{'1':0,'2':0},'D':{'1':0,'2':0},}
            for j, line in enumerate(b):
                data = self.debug.read_line( line )
                if data and 'Mixing' in data['message']:
                    out_dict = {}
                    # get the pipette number and iterate use
                    mix_match   = regex_mixing.match( data['message'] )
                    mix_dict    = {}
                    if mix_match: mix_dict = mix_match.groupdict()
                    # try to get pipette number and make pipette_name with use appened
                    try: pnum = str(mix_dict.pop('pipette'))
                    except KeyError: continue
                    
                    # get the timestamp from the data line
                    timestamp = data['timestamp']
                    # get the timing values
                    try:
                        time_data = self.debug.read_line( b[j+1] )
                    except IndexError:
                        # Missing data --> try to continue
                        continue
                    
                    time_message = time_data['message']
                    if 'Dispense' in time_message:  
                        regex_timing = dispense_timing
                        type_letter = 'D'
                    else:                           
                        regex_timing = mix_timing
                        type_letter = 'A'

                    use[type_letter][pnum]+=1
                    pipette_name = 'p{}_{}{}'.format( pnum, type_letter, use[type_letter][pnum] )
                    for key, item in mix_dict.items():
                        try:                out_dict[key] = float(item)
                        except ValueError:  out_dict[key] = str(item)

                    time_match = regex_timing.match( time_message )
                    if time_match:
                        for key, item in time_match.groupdict().items():
                            out_dict[key] = float( item )
                    # store values
                    out_dict['timestamp'] = timestamp.strftime( '%m/%d/%Y %H:%M:%S' )
                    block_dict[block_name][pipette_name] = out_dict
        return block_dict
    
    #############################################
    #   END -- HELPERS FOR PIPETTE BEHAVIOR     #
    #############################################

    def count_blocked_tips( self ):
        """ Reads the debug log to identify how many tips were blocked during piercing and count them up. """
        
        try:
            # Initialize variables
            self.metrics['blocked_tips'] = {}
            self.searched_for_blocked_tips = True 

            # Pipette 1 and 2 blockages here come from blockages detected after piercing.
            # Warnings are when we are forced to use a blocked tip for a step and has a different error message.
            pipette_1 = []
            pipette_2 = []
            warnings  = []

            # Look for piercing-related blockages
            lines     = self.debug.search_many( 'blocked after piercing' , 'GetContainerGeometry' )
            lines     = [ line for line in lines if 'AddExpEntry' not in line ]
            for i, line in enumerate( lines ):
                data = self.debug.read_line( line )
                if data:
                    if 'blocked' in data['message']:
                        # Grab tube from previous line
                        previous = self.debug.read_line( lines[ i-1 ] )['message'].strip()
                        #if 'blocked' in previous:
                        #    # Avoid double counting due to new ExpEntry Alarms
                        #    previous = self.debug.read_line( lines[ i-2 ] )['message'].strip()
                        
                        tube = previous.split()[2]
                        m    = re.match( r'.+?tip (?P<tip>[\d]{1}).+?', line )
                        if m:
                            tip = int( m.groupdict()['tip'])
                            if tip == 1:
                                pipette_1.append( tube )
                            elif tip == 2:
                                pipette_2.append( tube )
                                
                            print( "Found blocked tip on pipette {} after piercing {}!".format( tip, tube ) )
                            
            # Look for warnings where we had to use a blocked tip anyway:
            lines = self.debug.search( 'using a blocked tip' , after=1 )
            for i, line in enumerate(lines):
                data = self.debug.read_line( line )
                if data:
                    if 'blocked' in data['message'].lower():
                        # Read which well is being accessed in the next line of the file
                        tube = self.debug.read_line( lines[ i+1 ] )['message'].strip().split()[2]
                        
                        warnings.append( tube )
            
            self.metrics['blocked_tips'] = { 'p1_count'  : len( pipette_1 ),
                                             'p2_count'  : len( pipette_2 ),
                                             'p1_tubes'  : ', '.join(pipette_1) ,
                                             'p2_tubes'  : ', '.join(pipette_2) ,
                                             'used_blocked_tips'      : ', '.join(warnings) ,
                                             'used_blocked_tips_count': len( warnings )
                                             }
            return None
        except:
            print('!! Something went wrong when looking for blocked tips. Skipping Analysis.')
            self.searched_for_blocked_tips = False
            self.metrics['plugin_error'] = True

    def analyze_vacuum_logs( self ):
        ''' Reads the vacuum log csv files'''
        # metrics that will be written to json file and eventually saved to database
        self.metrics['vacLog'] = { 'lane 1':{}, 'lane 2':{}, 'lane 3':{}, 'lane 4':{}, 'lane 5':{}, 'lane 0':{}  } 
        
        lanes = ['lane 1', 'lane 2', 'lane 3', 'lane 4', 'lane 5', 'lane 0']
        real_lanes = ['lane 1', 'lane 2', 'lane 3', 'lane 4']
        #try:
        if True:
            v5 = os.path.join( self.calibration_dir, 'vacuum_data_lane5.csv' )
            vlog = os.path.join( self.calibration_dir, 'vacuum_log.csv' )
            
            # Regardless of which lane was run, the log for lane 5 (robot waste) will always be present
            if os.path.exists( v5 ) or os.path.exists( vlog ):
                self.has_vacuum_logs = True
                print('Vacuum Logs Found')
            else:
                self.has_vacuum_logs = False
            
            for lane in lanes:
                vl = os.path.join( self.calibration_dir, 'vacuum_data_lane{}.csv'.format(lane.split(' ')[1] ) )
                if os.path.exists( vl ):
                    print( 'Analyzing {} vacuum log...'.format(lane) )
                    process_list = self.extract_vacLog_data( vl )
                    process_list, metrics_dict = self.detect_leaks_clogs( process_list, lane )
                    self.metrics['vacLog'][lane] = metrics_dict
                    print('Total of {} processes for {}'.format(len(process_list) ,lane))
                    self.save_vacuum_plots( process_list, metrics_dict )
                else:
                    print('{} vacuum log not found.'.format(lane))
                    self.metrics['vacLog'][lane]['log_found']=False
                    self.metrics['vacLog'][lane]['postLib_found']=False 
            
            # If there are no vacuum logs in any lane, set self.has_vacuum_logs False
            if not any( [self.metrics['vacLog'][lane]['log_found'] for lane in real_lanes ] ) :
                self.has_vacuum_logs = False
        #except:
        #    print( '!! Something went wrong when analyzing the vacuum logs. Skipping analysis')
        #    self.has_vacuum_logs = False
        #    self.metrics['plugin_error'] = True
                    
    def extract_vacLog_data( self, filepath ):
        '''Create empty array that will contain a list of dictionaries, one dict for each process'''
        all_process_list = []
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            list_csv = list(csv_reader)
        
        # Iterate over each row in the file, setting some vales and finding start and end indices 
        all_process_start = []
        all_process_end = []
        for i,row in enumerate(list_csv):
            if i==len(list_csv)-1:
                all_process_end.append(i)
            if row[0]=='Comment':
                if i>0:
                    all_process_end.append(i-1)
                else:
                    pass
                process_dict = {}
               
                # Determine if the process actually happened during the run, not before
                processStartTime = list_csv[i+1][1]
                datetime_processStartTime = datetime.datetime.strptime(list_csv[i+1][1],'%Y_%m_%d-%H:%M:%S')
                if datetime_processStartTime > self.expt_start:
                    during_run = True 
                else:
                    during_run = False
                
                # Save values taken from process header
                process_dict['StartTime'] = processStartTime.replace('-','').replace(':','').replace('_','') 
                process_dict['Process']=row[1] # This is the comment in the vacuum log file
                process_dict['Lane']=list_csv[i+1][3]
                process_dict['Process during run?'] = during_run
                process_dict['Target Pressure']=float( list_csv[i+1][5] )
                process_dict['Initial Pressure']=float( list_csv[i+1][7] )
                
                # Determine which type of process it is
                if row[1].split(':')[0]=='CLOGCHECK':
                    if float(list_csv[i+1][17])==0:
                        process_dict['Type']='ClogCheck Slow'
                    else:
                        process_dict['Type']='ClogCheck Fast'
                elif row[1].split(':')[0]=='LEAKCHECK':
                    if float(list_csv[i+1][17])==0:
                        process_dict['Type']='LeakCheck Slow'
                    else:
                        process_dict['Type']='LeakCheck Fast'    
                elif row[1].split(':')[0] in ['MAGLOAD','TEMPLATE']:
                    process_dict['Type']='VacuumDry'
                # For newest version of postlibclean, added 21-Jun-2019
                elif 'POSTLIBCLEAN' or 'POSTRUN' or 'PostLibClean' in row[1]:
                    process_dict['Type']='VacuumDry'
                    if 'LEAKCHECK' in row[1]:
                        if float(list_csv[i+1][17])==0:
                            process_dict['Type']='LeakCheck Slow'
                        else:
                            process_dict['Type']='LeakCheck Fast'
                    if 'CLOGCHECK' in row[1]:
                        if float(list_csv[i+1][17])==0:
                            process_dict['Type']='ClogCheck Slow'
                        else:
                            process_dict['Type']='ClogCheck Fast'
                else:
                    process_dict['Type']='Unknown'
                    print(row[1])

                # Fill in additional header values appropriate to process type
                if process_dict['Type'] in ['ClogCheck Slow', 'ClogCheck Fast', 'LeakCheck Slow', 'LeakCheck Fast']:
                    process_dict['PassFail Threshold']=float( list_csv[i+1][9] ) # use for checking errors
                    process_dict['TestTime']=float( list_csv[i+1][11] ) # might not be necessary
                    process_dict['HoldTime']=float( list_csv[i+1][15] ) # might not be necessary

                all_process_start.append(i)
                all_process_list.append(process_dict)

        # Only include processes that occured after run start time
        process_start = []
        process_end = []
        process_list = []
        for i, process_dict in enumerate(all_process_list):
            if process_dict['Process during run?']:
                process_start.append(all_process_start[i])
                process_end.append(all_process_end[i])
                process_list.append(all_process_list[i])
            else:
                print('Skipping vacuum step {}.'.format( process_dict['Process']) )
        
        #Iterate over each process. Get arrays of time, pressure, and other info depending on process type 
        for i, start in enumerate(process_start):
            end = process_end[i]
            process = list_csv[start+2:end+1] # array of rows from a single process
            if process_list[i]['Type'] in ['VacuumDry']:
                self.analyze_process_VacuumDry(process, process_list[i])
            elif process_list[i]['Type'] in ['ClogCheck Slow', 'ClogCheck Fast']:
                self.analyze_process_ClogCheck(process, process_list[i])
            elif process_list[i]['Type'] in ['LeakCheck Slow', 'LeakCheck Fast']:
                self.analyze_process_LeakCheck(process, process_list[i])
            elif process_list[i]['Type']=='Unknown':
                self.analyze_process_Unknown(process, process_list[i])
            else:
                self.analyze_process_Unknown(process, process_list[i])
        return( process_list )
   
    def analyze_process_VacuumDry(self, process, process_dict):
        '''Handles VacuumDry steps from MAGLOADING, TEMPLATING, and POSTRUN'''
        
        print('Analyzing {} for {}'.format(process_dict['Type'],process_dict['Process']) )
        time_temp = [] # array of time values
        pressure_temp = [] # array of pressure values
        pumping_to_goal_temp = [] # boolean array for if pumping to target/initial pressure
        venting_to_target_temp = [] # for new PostRun
        pumping_to_maintain_target_temp = [] # boolean array for if pumping to maintain target pressure   
        pump_start_time = None # set initial value incase we don't find a pump start time

        pump_has_started = False # changed to True if conditions are met
        pressure_reached_target = False # changed to True if conditions are met
        pressure_reached_initial = False # used only for PostRun in which initial is -8
        got_index = False
        
        
        for j, state in enumerate(process):
            time_temp.append(float(state[3]))
            pressure_temp.append(float(state[5]))     
            
            venting_to_target = 0
            pumping_to_goal = 0
            pumping_to_maintain_target = 0
            
            if state[1].split('_')[1]=='Pumping': # generalized for both VacuumDry and PostRun
                # Record the time the pump has started
                if pump_has_started==False:
                    pump_start_time = float(state[3])
                pump_has_started = True
                pumping_to_goal = 1
                if process_dict['Target Pressure'] < process_dict['Initial Pressure']: # only for normal VacuumDry
                    if float(state[5]) <= process_dict['Target Pressure']: 
                        # Record the time the pressure reached target
                        if pressure_reached_target==False:
                            pump_reached_target_time = float(state[3])
                            pump_reached_target_index = j
                        pressure_reached_target = True
                        # Now that the target pressure has been reached, pumping will be to maintain target pressure
                    if pressure_reached_target:
                        pumping_to_goal = 0
                        pumping_to_maintain_target = 1 
                else: # PostRun     
                    if float(state[5])<= process_dict['Initial Pressure']:
                        if pressure_reached_initial==False:
                            pump_reached_initial_time = float(state[3])
                            pump_reached_initial_index = j
                        pressure_reached_initial = True
                    if pressure_reached_initial: 
                        if float(state[5])>= process_dict['Target Pressure']: # venting through lane allowed pressure decrease
                            # Record the time pressure reached target
                            if pressure_reached_target==False:
                                pump_reached_target_time = float(state[3])
                                pump_reached_target_index = j
                            pressure_reached_target = True
                    if pressure_reached_target:
                        pumping_to_goal = 0
                        pumping_to_maintain_target = 1
                            
                                
            elif state[1]=='TargetPressure_Checking':
                if process[j-1][1].split('_')[1]=='Pumping': # generalized for both VacuumDry and PostRun
                    pumping_to_goal = 1
                    if pressure_reached_target ==True:
                        pumping_to_goal = 0
                        pumping_to_maintain_target = 1

            elif state[1]=='TargetPressure_Venting':
                if pressure_reached_target == False:
                    venting_to_target = 1
                 
            # Append to arrays
            pumping_to_goal_temp.append(pumping_to_goal)
            venting_to_target_temp.append(venting_to_target)
            pumping_to_maintain_target_temp.append(pumping_to_maintain_target)
            
            # Set the delay after valve opens we wait before collecting duty cycle. Necessary for steps with SDS.
            if process_dict['Type']=='VacuumDry':
                t = 4 # change from 10 to 4 for shorter target pressure hold times. 
            else:     # This is for PostRun. Two second delay
                t = 2
                
            # Determine index t after pressure reached target to later determine Duty Cycle
            if pressure_reached_target:
                if float(state[3]) >= (pump_reached_target_time + t):
                    if got_index==False:
                        t_after_target_index = j # need this index for later
                    got_index=True
        
        process_dict['time (s)']=time_temp
        process_dict['pressure']=pressure_temp
        process_dict['max_dt'] = max([fin-init for init, fin in zip(time_temp, time_temp[1:])]) # for width of bars in plots
            
        process_dict['pumping to goal']=pumping_to_goal_temp
        process_dict['venting to target']=venting_to_target_temp
        process_dict['pumping to maintain target']=pumping_to_maintain_target_temp
        process_dict['Pump Start Time'] = pump_start_time
        process_dict['Pressure Reached Target'] = pressure_reached_target
        try:
            process_dict['Duty Cycle at End']           = float(np.sum(pumping_to_maintain_target_temp[t_after_target_index:])) /float( len(pumping_to_maintain_target_temp[t_after_target_index:]))*100 
            process_dict['Amplitude of Oscillation']    = np.std(pressure_temp[t_after_target_index:])* math.sqrt(3)
            process_dict['Max Amp of Oscillation']      = (max(pressure_temp[pump_reached_target_index:t_after_target_index])-min(pressure_temp[pump_reached_target_index:t_after_target_index])) / 2.0
            process_dict['Integral of Abs. Osc.']       = np.absolute( np.array(pressure_temp[pump_reached_target_index:t_after_target_index]) ).sum()
        except:
            process_dict['Duty Cycle at End']           = 100 
            process_dict['Amplitude of Oscillation']    = None
            process_dict['Max Amp of Oscillation']      = None
            process_dict['Integral of Abs. Osc.']       = None
            
        if process_dict['Target Pressure'] <= process_dict['Initial Pressure']: # normal VacuumDry    
            if pressure_reached_target:
                process_dict['Time to Target'] = pump_reached_target_time - pump_start_time
                process_dict['Vent Open Time'] = pump_start_time
            else:
                if pump_start_time == None:
                    process_dict['Time to Target'] = 0 
                    process_dict['Duty Cycle at End'] = 0 
                    process_dict['Pressure Reached Target'] = True 
                    print('Did not find time to target. Assume pump never turned on since pressure was already lower than target')
                else:
                    process_dict['Time to Target'] = time_temp[-1] - pump_start_time # entire time is spends pumping

        else: # PostRun
            process_dict['Pressure Reached Initial'] =  pressure_reached_initial
            if pressure_reached_initial:
                process_dict['Time to Initial']=pump_reached_initial_time - pump_start_time
                process_dict['Vent Open Time'] = pump_reached_initial_time
                if pressure_reached_target:
                    process_dict['Time to Target'] = pump_reached_target_time - pump_reached_initial_time
                else:
                    process_dict['Time to Target']= time_temp[-1] - pump_reached_initial_time
            else:
                try:
                    process_dict['Time to Initial'] = time_temp[-1] - pump_start_time        
                except:
                    process_dict['Time to Initial'] = None
                    print('Unable to determine time to initial. Perhaps because we do not have a pump start time.')

    def analyze_process_ClogCheck( self, process, process_dict ):
        '''Handles all clog check steps that occur during PostLibClean'''

        print('Analyzing ClogCheck for {}'.format(process_dict['Process']) )
        time_temp = [] # array of time values
        pressure_temp = [] # array of pressure values
        pumping = []
        venting = []
        venting_started = False
        for j, state in enumerate(process):
            time_temp.append(float(state[3]))
            pressure_temp.append(float(state[5])) 
            pump_on = 0
            valve_open = 0
            if state[1].split('_')[1]=='Pumping':
                pump_on = 1
            if state[1]=='TargetPressure_Venting':
                valve_open = 1
                if venting_started == False:
                    vent_open_time = float(state[3])
                    end_of_hold_pressure = float(state[5])
                venting_started = True
                
            pumping.append(pump_on)
            venting.append(valve_open)
                   
        process_dict['time (s)']=time_temp
        process_dict['pressure']=pressure_temp   
        process_dict['max_dt'] = max([fin-init for init, fin in zip(time_temp, time_temp[1:])]) # for width of bars
        process_dict['pumping']=pumping
        process_dict['venting']=venting
        if venting_started:
            process_dict['vent time'] = time_temp[-1] - vent_open_time # This should be exactly the same as the TestTime 
            process_dict['Vent Open Time'] = vent_open_time
            process_dict['End of Hold Pressure'] = end_of_hold_pressure
        # Note that the initial pressure may never be reached if there is a leak. This should still be used to look for clog

    def analyze_process_LeakCheck( self, process, process_dict ):
        print('Analyzing LeakCheck for {}'.format(process_dict['Process']) )
        time_temp = [] # array of time values
        pressure_temp = [] # array of pressure values
        pumping = []
        holding = []
        holding_started = False
        for j, state in enumerate(process):
            time_temp.append(float(state[3]))
            pressure_temp.append(float(state[5])) 
            pump_on = 0
            hold = 0
            if state[1].split('_')[1]=='Pumping':
                pump_on = 1
            if state[1]=='TargetPressure_Holding':
               hold = 1 
               if holding_started == False:
                   start_of_hold_pressure = float(state[5])
                   holding_started = True
            pumping.append(pump_on)
            holding.append(hold)
                   
        process_dict['time (s)']=time_temp
        process_dict['pressure']=pressure_temp   
        process_dict['max_dt'] = max([fin-init for init, fin in zip(time_temp, time_temp[1:])]) # for width of bars
        process_dict['pumping']=pumping
        process_dict['holding']=holding
        process_dict['Start of Hold Pressure'] = start_of_hold_pressure

    def analyze_process_Unknown( self, process, process_dict ):
        print('Analyzing Unknown for {}'.format(process_dict['Process']) )
        time_temp = [] # array of time values
        pressure_temp = [] # array of pressure values
            
        for j, state in enumerate(process):
            time_temp.append(float(state[3]))
            pressure_temp.append(float(state[5]))     
           
        process_dict['time (s)']=time_temp
        process_dict['pressure']=pressure_temp   
        process_dict['max_dt'] = max([fin-init for init, fin in zip(time_temp, time_temp[1:])]) # for width of bars
        
    def detect_leaks_clogs( self, process_list, lane ):
        suspected_leaks         = []
        suspected_clogs         = []
        postLib_leaks           = []
        postLib_clogs           = []
        non_SDS_DC              = [] # do not include SDS. Do not include postRun for Lanes1,2,3,4
        non_SDS_amp             = [] # do not include SDS. Do not include postRun for Lanes1,2,3,4
        non_SDS_maxAmp          = [] # do not include SDS. Do not include postRun for Lanes1,2,3,4
        non_SDS_time_to_target  = []
        log_found               = False   # workflow vacuum logs- set to true once found
        postLib_found           = False
        maxAmp_foamScrape       = None
        intAbsOsc_foamScrape    = None
        postTemp_SDS_wash1_DC               = None
        postTemp_SDS_wash1_time_to_target   = None
        postTemp_SDS_wash1_AO               = None
        postTemp_SDS_wash2_DC               = None
        postTemp_SDS_wash2_time_to_target   = None
        postTemp_SDS_wash2_AO               = None
        temp_2xMeltOff_ABwash_DC               = None
        temp_2xMeltOff_ABwash_time_to_target   = None
        temp_2xMeltOff_ABwash_AO               = None
        temp_postMO_flush_DC               = None
        temp_postMO_flush_time_to_target   = None
        temp_postMO_flush_AO               = None
        
        for process_dict in process_list:
            # Want to save the VacuumDry steps around the meltoff 
            if process_dict['Process'] == 'TEMPLATE: PostTemplating SDS Wash 1':
                try:
                    postTemp_SDS_wash1_DC             = process_dict['Duty Cycle at End']
                    postTemp_SDS_wash1_time_to_target = process_dict['Time to Target']
                    print('iprocess: {}'.format(process_dict['Process']))
                except:
                    postTemp_SDS_wash1_DC               = None
                    postTemp_SDS_wash1_time_to_target   = None
            if process_dict['Process'] == 'TEMPLATE: PostTemplating SDS Wash 2':
                try:
                    postTemp_SDS_wash2_DC             = process_dict['Duty Cycle at End']
                    postTemp_SDS_wash2_time_to_target = process_dict['Time to Target']
                    print('iprocess: {}'.format(process_dict['Process']))
                except:
                    postTemp_SDS_wash2_DC               = None
                    postTemp_SDS_wash2_time_to_target   = None
            if process_dict['Process'] == 'TEMPLATE: 2xMeltOff and AB wash':
                try:
                    temp_2xMeltOff_ABwash_DC             = process_dict['Duty Cycle at End']
                    temp_2xMeltOff_ABwash_time_to_target = process_dict['Time to Target']
                    print('iprocess: {}'.format(process_dict['Process']))
                except:
                    temp_2xMeltOff_ABwash_DC               = None
                    temp_2xMeltOff_ABwash_time_to_target   = None
            if process_dict['Process'] == 'TEMPLATE: PostMO 60/40 Flush':
                try:
                    temp_postMO_flush_DC             = process_dict['Duty Cycle at End']
                    temp_postMO_flush_time_to_target = process_dict['Time to Target']
                    print('iprocess: {}'.format(process_dict['Process']))
                except:
                    temp_postMO_flush_DC               = None
                    temp_postMO_flush_time_to_target   = None
            # Want to save the max amplitude of the VacuumDry following foam scrape since this usually has a pressure spike. 
            # Want to save the max amplitude of the VacuumDry following foam scrape since this usually has a pressure spike. 
            if 'Foam Scrape' in process_dict['Process'] or 'foam' in process_dict['Process']:
                try:
                    maxAmp_foamScrape       = process_dict['Max Amp of Oscillation']
                    intAbsOsc_foamScrape    = process_dict['Integral of Abs. Osc.']
                except:
                    maxAmp_foamScrape       = None
                    intAbsOsc_foamScrape    = None
            # For problems during workflow, excluding SDS steps
            if (process_dict['Type']=='VacuumDry') and not ('POSTLIBCLEAN' in process_dict['Process']) and not ('SDS' in process_dict['Process']) and not ('POSTRUN' in process_dict['Process']):  
                log_found = True
                # Check for clog
                if process_dict['Pressure Reached Target']:
                    if process_dict['Duty Cycle at End']<= 8: # 8 chosen based on data from actual clogs. 
                        suspected_clogs.append(process_dict['Process'])
                    if process_dict['Duty Cycle at End']>= 30: # arbitrarily selected threshold, may lead to false alarms at high altitudes 
                        suspected_leaks.append(process_dict['Process'])
                    else:
                        # To track normal pump behavior. SDS steps are always weird due to bubbles.
                        # Leave out the clogged runs as well
                        non_SDS_DC.append(process_dict['Duty Cycle at End'])
                        non_SDS_amp.append(process_dict['Amplitude of Oscillation'])
                        non_SDS_maxAmp.append(process_dict['Max Amp of Oscillation'])
                        # For time-to-target, only use vac steps in which no lane is starting right after a robot waste dump.
                        if process_dict['Process'] in ['MAGLOAD: water wash for sucrose','MAGLOAD: AB wash and dry','TEMPLATE: PostMO 60/40 Flush']:
                            print('normal time to target in process: {}'.format(process_dict['Process']))
                            non_SDS_time_to_target.append(process_dict['Time to Target'])
                # Leak if pressure did not reach target, however only when target pressure is attainable 
                elif process_dict['Target Pressure']>-9:
                    suspected_leaks.append(process_dict['Process'])
            # For PostLibClean problems        
            elif process_dict['Type'] in ['ClogCheck Slow','ClogCheck Fast']:
                postLib_found = True
                if process_dict['pressure'][-1] < process_dict['End of Hold Pressure'] + process_dict['PassFail Threshold']:
                    postLib_clogs.append(process_dict['Process'])    
            elif process_dict['Type'] in ['LeakCheck Slow','LeakCheck Fast']:
                postLib_found = True
                if process_dict['pressure'][-1] > process_dict['Start of Hold Pressure'] + process_dict['PassFail Threshold']:
                    postLib_leaks.append(process_dict['Process'])    
            if (process_dict['Type']=='VacuumDry') and ( ('POSTLIBCLEAN' or 'POSTRUN') in process_dict['Process']) and not ('SDS' in process_dict['Process']) :
                postLib_found = True
                # Check for clog in bleed valve. Should never reach target pressure, unless clog is severe
                if process_dict['Lane']=='0':
                    if process_dict['pressure'][-1] < -1:
                        postLib_clogs.append(process_dict['Process'])
                elif process_dict['Pressure Reached Target']:
                    if process_dict['Duty Cycle at End']<= 8: # 8 chosen based on data from actual clogs. 
                        postLib_clogs.append(process_dict['Process'])
                    if process_dict['Duty Cycle at End']>= 30: # arbitrarily selected threshold, may lead to false alarms at high altitudes 
                        postLib_leaks.append(process_dict['Process'])
                # Leak if pressure did not reach target 
                else:
                    postLib_leaks.append(process_dict['Process'])
            else:
            # Handle other vacuum check processes that happen DURING the workflow 
                pass
        
        # Remove None values in calculation of the mean
        non_SDS_amp = [amp for amp in non_SDS_amp if amp != None]
        non_SDS_maxAmp = [Mamp for Mamp in non_SDS_maxAmp if Mamp != None]
        non_SDS_DC = [amp for amp in non_SDS_DC if amp != None]
        non_SDS_time_to_target = [amp for amp in non_SDS_time_to_target if amp != None]

        if log_found == False:
            print('No workflow vacuum logs were found.')
        if postLib_found == False:
            print('No postLibClean logs were found.') 
        
        if lane in ['lane 0','lane 5']:
            metrics_dict = {'postLib_abnormal_process_count'    : len(postLib_leaks)+len(postLib_clogs),
                            'postLib_suspected_leaks_count'     : len(postLib_leaks),
                            'postLib_suspected_clogs_count'     : len(postLib_clogs),
                            'postLib_leaks'                     : ', '.join(postLib_leaks),
                            'postLib_clogs'                     : ', '.join(postLib_clogs),
                            'postLib_found'                     : postLib_found ,
                            'log_found'                         : log_found, # not saved in chipdb, but used to make block html. expect to always be false 
                            }
        else:
            metrics_dict = {'abnormal_process_count'                : len(suspected_leaks)+len(suspected_clogs),
                            'suspected_leaks_count'                 : len(suspected_leaks),
                            'suspected_clogs_count'                 : len(suspected_clogs),
                            'suspected_leaks'                       : ', '.join(suspected_leaks),
                            'suspected_clogs'                       : ', '.join(suspected_clogs),
                            'postLib_abnormal_process_count'        : len(postLib_leaks)+len(postLib_clogs),
                            'postLib_suspected_leaks_count'         : len(postLib_leaks),
                            'postLib_suspected_clogs_count'         : len(postLib_clogs),
                            'postLib_leaks'                         : ', '.join(postLib_leaks),
                            'postLib_clogs'                         : ', '.join(postLib_clogs),
                            'normal_DC'                             : np.mean(non_SDS_DC),
                            'normal_amp'                            : np.mean(non_SDS_amp),
                            'normal_maxAmp'                         : np.mean(non_SDS_maxAmp),
                            'normal_time_to_target'                 : np.mean(non_SDS_time_to_target),
                            'postTemp_SDS_wash1_DC'                 : postTemp_SDS_wash1_DC,
                            'postTemp_SDS_wash1_time_to_target'     : postTemp_SDS_wash1_time_to_target,
                            'postTemp_SDS_wash2_DC'                 : postTemp_SDS_wash2_DC,
                            'postTemp_SDS_wash2_time_to_target'     : postTemp_SDS_wash2_time_to_target,
                            'meltOff_ABwash_DC'                     : temp_2xMeltOff_ABwash_DC,
                            'meltOff_ABwash_time_to_target'         : temp_2xMeltOff_ABwash_time_to_target,
                            'postMO_flush_DC'                       : temp_postMO_flush_DC,
                            'postMO_flush_time_to_target'           : temp_postMO_flush_time_to_target,
                            'log_found'                             : log_found, 
                            'postLib_found'                         : postLib_found ,
                            'maxAmp_foamScrape'                     : maxAmp_foamScrape,
                            'intAbsOsc_foamScrape'                  : intAbsOsc_foamScrape,
                            }

        return(process_list, metrics_dict)
    
    def save_vacuum_plots( self, process_list, metrics_dict ): 
        '''Iterate through each process and save each individual plot'''
        postRun = False
        process_count = 0
        for process in process_list:
            process_count += 1
            plt.figure( figsize=(2.7,2) )
            fontsize = 8.5
            dt = process['max_dt']
            
            if process['Type'] in ['VacuumDry']:
                initP = process['Initial Pressure']
                finalP = process['Target Pressure']
                t_pumpStart = process['Pump Start Time']
                try:
                    t_ventStart = process['Vent Open Time'] # Same as t_pumpStart for VacuumDry
                except: ## fix later
                    t_ventStart = 1

                ymax = 0.1
                ymin = min( min(finalP, initP), min(process['pressure']) )
            
                bar_height = (ymax-ymin)* 0.111
                bar_loc = ymin - (bar_height*1.5)
                
                x_loc = max(process['time (s)'])*3/5
                y_loc = 0 + process['Target Pressure']/4 
                plt.ylim([bar_loc, 0.1])
                
                # For bleed valve vacuum dry
                if process['Lane']=='0' and process['Target Pressure'] == process['Initial Pressure']:  
                    color = 'green' 
                    if process['pressure'][-1] < -1: # will not reach zero when clogged 
                        color = 'red'
                        plt.title('Suspected Clog', fontsize=fontsize, fontweight='bold') 
                elif process['Pressure Reached Target']:
                    color = 'green'
                    plt.text(x_loc,y_loc,' DC = {:.0f} %'.format(process['Duty Cycle at End']) , fontsize=fontsize)
                    if process['Amplitude of Oscillation']:
                        plt.text(x_loc,y_loc-1,' amp = {:.3f} psi'.format(process['Amplitude of Oscillation']) , fontsize=fontsize)
                    #if process['Max Amp of Oscillation']:
                    #    plt.text(x_loc,y_loc-2,' max amp = {:.3f} psi'.format(process['Max Amp of Oscillation']) , fontsize=fontsize)
                    if process['Duty Cycle at End'] <=8:
                        color = 'red'
                        plt.title('Suspected Clog', fontsize=fontsize, fontweight='bold')
                    if process['Duty Cycle at End'] >=30:
                        color = 'red'
                        plt.title('Suspected Leak', fontsize=fontsize, fontweight='bold')
                else:
                    if process['Target Pressure'] <= process['Initial Pressure']: 
                        # normal VacuumDry
                        color = 'red'
                        plt.title('Suspected Leak', fontsize=fontsize, fontweight='bold')
                    elif process['Target Pressure'] > process['Initial Pressure']: 
                        # new POSTRUN, clog when target pressure not reached
                        color = 'red'
                        if process['Pressure Reached Initial']:
                            # Clog if initial pressure was reached, but not target pressure
                            plt.title('Suspected Clog', fontsize=fontsize, fontweight='bold')
                        else:
                            # Leak if initial pressure was not reached either
                            plt.title('Suspected Leak', fontsize=fontsize, fontweight='bold')
                
                # Colored Bar Section
                plt.bar(process['time (s)'], np.asarray(process['pumping to maintain target'])*bar_height, bottom = bar_loc, width=dt, color='gray', edgecolor='none' )
                plt.bar(process['time (s)'], np.asarray(process['pumping to goal'])*bar_height, bottom = bar_loc, width=dt, color='orange', edgecolor='none' )
                if finalP <= initP: # VacuumDry (<) and old PostRuns (=)
                    if (process['Time to Target'] == None) or (t_pumpStart == None):
                        plt.text(2,bar_loc+bar_height/4,'Pump never turned on', fontsize=fontsize)
                    else:
                        plt.text(t_pumpStart+process['Time to Target']/20,bar_loc+bar_height/4,str( "{:.1f}".format(process['Time to Target'])+' s' ), fontsize=fontsize)            
                else: # new PostRun
                    if process['Time to Initial']>=5.5:
                        plt.text(t_pumpStart+process['Time to Initial']/20,bar_loc+bar_height/4,str( "{:.1f}".format(process['Time to Initial'])+' s' ), fontsize=fontsize)
                    plt.bar(process['time (s)'], np.asarray(process['venting to target'])*bar_height, bottom = bar_loc, width=dt, color='aquamarine', edgecolor='none' )
                    try:
                        plt.text(t_ventStart+process['Time to Target']/20,bar_loc+bar_height/4,str( "{:.1f}".format(process['Time to Target'])+' s' ), fontsize=fontsize)
                    except:
                        plt.text(t_ventStart+5/20,bar_loc+bar_height/4,'???', fontsize=fontsize )
                plt.yticks([0,process['Target Pressure']], fontsize=fontsize) # was 10
                plt.axhline(y=finalP, linestyle='-', color='gray')
                plt.axhline(y=initP, linestyle='--', color='gray')
                plt.xticks([])
                
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)
                

            elif process['Type'] in ['ClogCheck Slow', 'ClogCheck Fast']:
                initP = process['Initial Pressure'] # should be -6
                finalP = process['Target Pressure'] # is zero, however I want to change it to the threshold value
                try:
                    threshold = process['End of Hold Pressure'] + process['PassFail Threshold'] # use threshold if that is higher...
                    plt.axhline(y=threshold, linestyle='-', color='orange')
                except:
                    threshold = -12 # placeholder number, for when key does not exist

                ymax = max(math.ceil(max(process['pressure'])), threshold)
                ymin = min(min(process['pressure']), initP)

                # This is to prevent matplotlib from freaking out 
                if (ymax - ymin) <=2:
                    ymax += 2
                try:
                    if process['pressure'][-1] >= process['End of Hold Pressure'] + process['PassFail Threshold']:
                        color = 'green'
                    else:
                        color = 'red'
                        plt.title('Suspected Clog', fontsize=fontsize, fontweight='bold')
                except:
                    color = 'red'

                t_ventStart = process['Vent Open Time']
                bar_height = (ymax-ymin)* 0.111 
                bar_loc = ymin - (bar_height*1.5)
                plt.ylim([bar_loc, ymax])
                
                plt.bar(process['time (s)'], np.asarray(process['pumping'])*bar_height, bottom = bar_loc, width=dt, color='orange', edgecolor='none' )
                plt.bar(process['time (s)'], np.asarray(process['venting'])*bar_height, bottom = bar_loc, width=dt, color='aquamarine', edgecolor='none' )
                plt.text(t_ventStart+process['vent time']/4,bar_loc+bar_height/4, '{:.0f} s venting'.format(process['vent time']) , fontsize=fontsize)
                
                plt.yticks([math.ceil(initP),math.ceil(ymax)], fontsize=fontsize)
                #plt.axhline(y=finalP, linestyle='-', color='gray')
                plt.axhline(y=initP, linestyle='--', color='gray')
                plt.xticks([])
                
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)
            
            
            elif process['Type'] in ['LeakCheck Slow', 'LeakCheck Fast']:
                initP = process['Initial Pressure'] # should be -6
                finalP = process['Target Pressure'] # is zero, however I want to change it to the threshold value
                try:
                    threshold = process['Start of Hold Pressure'] + process['PassFail Threshold'] # use threshold if that is higher...
                    plt.axhline(y=threshold, linestyle='-', color='orange')
                except:
                    threshold = -12 # placeholder number, for when key does not exist

                ymax = max(math.ceil(max(process['pressure'])), threshold)
                ymin = min(min(process['pressure']), initP)

                # This is to prevent matplotlib from freaking out 
                if (ymax - ymin) <=2:
                    ymax += 2
                try:
                    if process['pressure'][-1] <= process['Start of Hold Pressure'] + process['PassFail Threshold']:
                        color = 'green'
                    else:
                        color = 'red'
                        plt.title('Suspected Leak', fontsize=fontsize, fontweight='bold')
                except:
                    color = 'red'

                bar_height = (ymax-ymin)* 0.111 
                bar_loc = ymin - (bar_height*1.5)
                plt.ylim([bar_loc, ymax])
                
                plt.bar(process['time (s)'], np.asarray(process['pumping'])*bar_height, bottom = bar_loc, width=dt, color='orange', edgecolor='none' )
                
                plt.yticks([math.ceil(initP),math.ceil(ymax)], fontsize=fontsize)
                #plt.axhline(y=finalP, linestyle='-', color='gray')
                plt.axhline(y=initP, linestyle='--', color='gray')
                plt.xticks([])
                
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)

            else:
                color = 'black'
                plt.yticks([0,-9, -12], fontsize=fontsize)
                plt.ylim([-12, 0.1])
            
            plt.plot( process['time (s)'],process['pressure'], '-', color=color)
            #plt.title(process['Process'], fontsize=10, fontweight="bold")
            
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            # Determine when postlibclean (also called postrun) begins 
            if 'PostLibClean' in process['Process']:
                postRun = True
            if 'POSTLIBCLEAN' in process['Process']:
                postRun = True
            if 'POSTRUN' in process['Process']:
                postRun = True
            
            
            # The figure name is used to reconstruct the process name in write_vacuum_html 
            fig_name = 'Lane{}_process_{}_{}.png'.format( process['Lane'], str(process_count), process['Process'].replace(':',' ').replace('/','-').replace('%','perc') )
            if postRun:
                fig_name = 'PostLib_Lane{}_process_{}_{}.png'.format( process['Lane'], process['StartTime'], process['Process'].replace(':',' ').replace('/','-').replace('%','perc') )
            try:
                plt.savefig( os.path.join(self.results_dir, fig_name) )
            except:
                print(fig_name, 'not able to be saved')
            plt.close( )

        # Empty figure as place holder
        fig = plt.figure(figsize=(2.7,2))
        ax = fig.add_subplot(1,1,1, axisbg='lightgray')
        #fig.set_facecolor('gray')
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        fig.savefig( os.path.join(self.results_dir, 'empty.png'))
            
    def is_mock_run( self ):
        """ 
        Checks to see if there were enough sequencing flows to matter.  
        If not, it assumes we found a mock run and sends an exit signal.
        Used to be module detection based, however, this was buggy and didn't always work.
        """
        answer = False

        #if not any( self.debug.modules.values() ):
        if self.explog.flows < self.mock_flow_threshold:
            answer = True
            m = 'This run appears to be a mock run, with only {:d} flows! Exiting . . .'.format(self.explog.flows)
            print( m )
            
            # Write trivial output to the block html
            doc = H.Document()
            msg = H.Header( m , 3 )
            doc.add( msg )
            with open( os.path.join( self.results_dir, 'ValkyrieWorkflow_block.html' ), 'w' ) as f:
                f.write( str( doc ) )
                
        return answer
    
    def is_missing_data( self, final_check=False):
        """ Checks for required data and returns a true if we are missing data, meaning we should immediately exit the plugin. """
        not_a_valk    = False # true if not a valkyrie run
        missing_debug = False # won't necessarily exit plugin is data is missing, may decide to wait for debug
        msg = ''
        # Let's make sure this is a Valkyrie run.
        if self.explog.metrics['Platform'].lower() == 'valkyrie':
            print( 'Confirmed that run is on a Valkyrie!  Proceeding with plugin.' )
        else:
            print( 'Exiting plugin as this run is not on a Valkyrie!' )
            not_a_valk = True
            
            # Write trivial output
            doc = H.Document()
            msg = H.Header( 'This run does not appear to be on a Valkryie.  Exiting plugin.', 3 )
            doc.add( msg )
            with open( os.path.join( self.results_dir, 'ValkyrieWorkflow_block.html' ), 'w' ) as f:
                f.write( str( doc ) )
            return not_a_valk, missing_debug
                
        # Now let's make sure we have access to the debug file.
        if os.path.exists( self.debug_log ):
            print( 'Successfully located debug log file.' )
            msg = H.Header( 'Found debug log file.', 3 )
        else:
            missing_debug = True
            if final_check:
                print( 'Exiting plugin since the debug log file was not found!' )
                msg = H.Header( 'Unable to find debug log file.  Exiting plugin.', 3 )
            else:
                print( 'Debug log file was not found!' )
                msg = H.Header( 'Unable to find debug log file.  Waiting to see if it appears....', 3 )
        # Write trivial output
        doc = H.Document()
        doc.add( msg )
        with open( os.path.join( self.results_dir, 'ValkyrieWorkflow_block.html' ), 'w' ) as f:
            f.write( str( doc ) )
                
        return not_a_valk, missing_debug
    
    def did_plugin_run_too_early( self ):
        """
        Plugins are launch once analysis is complete. However, postLibClean and PostRun or PostChipClean may run after analysis is complete.
        This function checks whether or not this is the case, and generates a message notifying the user that data is missing.
        This function is used to determine if ValkyrieWorkflow should wait and see if PostRun has finished. 
        No longer using Vacuum logs from PostLibClean since PostRun will always occur after PostLibClean.
        """
        # First, read the explog to see if user has selected PostLibClean, PostChipClean, or PostRunClean.
        doPostLib     = self.explog.metrics['doPostLibClean'] # now refers to the deck clean, usually happens in parallel (set by doParallelClean)
        doPostLibVac  = self.explog.metrics['doVacuumClean']  # clean the vacuum manifold during PostLib
        if doPostLibVac==None:
            print( 'no entry for doVacuumClean in explog.' )
            doPostLibVac = doPostLib  # will fail for earlier builds in which this does not yet exist since it was grouped with doPostLib
        doPostRun     = self.explog.metrics['postRunClean']  # note that now PostRun and PostChip are always BOTH true unless unselected
        doPostChip    = self.explog.metrics['doPostChipClean']

        # Print out what was selected for postLib, postRun, and postChip cleans
        print('PostLib Deck Clean: {}'.format(doPostLib)) 
        print('PostLib Vacuum Clean: {}'.format(doPostLibVac))  
        print('PostRun:{}'.format(doPostRun)) 
        print('PostChip: {}'.format(doPostChip))
       
        # Next, generate a warning message for when the plugin completed before post run cleans were complete.
        if self.ss_file_found:
            try:
                if self.ss.data['postrun']['overall']['end']==None:
                    found_postrun_end = False
                else:
                    found_postrun_end = True
            except:
                found_postrun_end = False
        else:
            found_postrun_end = False


        if (doPostChip or doPostRun) and not found_postrun_end:
                self.message = 'WARNING: missing postRun/postChip clean data.'
                print('Postrun end time not found.... likely that plugin was launched before postrun cleans were complete.')
        else:
            self.message = ''
        
        vac_log_robotWaste = os.path.join( self.calibration_dir, 'vacuum_data_lane5.csv' )
        os.path.exists( vac_log_robotWaste )
        if doPostLibVac and not os.path.exists( vac_log_robotWaste ):
                self.message = 'WARNING: missing postLib vacuum clean data.'
                print('PostLib Deck clean not found.... likely that plugin was launched before these were generated.')
        else:
            self.message = ''

    def write_block_html( self ):
        """ Creates a simple block html file for minimal viewing clutter. """
        doc = H.Document()
        doc.add( '<!DOCTYPE html>' )
        
        styles = H.Style()
        styles.add_css( 'td.active'  , { 'background-color': '#0f0' } )
        styles.add_css( 'td.inactive', { 'background-color': '#000' } )
        styles.add_css( 'td.info'   , { 'background-color': 'white',
                                         'color': '#000' } ) # used to be white (#fff) but I changed it to black
        styles.add_css( 'td.error'   , { 'background-color': '#FF4500',
                                         'font-weight': 'bold',
                                         'color': '#000' } ) # used to be white (#fff) but I changed it to black
        styles.add_css( 'td.flag'   , { 'background-color': '#FFAF33',
                                         'font-weight': 'bold',
                                         'color': '#000' } )
        styles.add_css( 'td.ok'      , { 'font-weight': 'bold',
                                         'color': '#0f0' } )
        styles.add_css( 'td.warning' , { 'font-weight': 'bold',
                                         'color': '#FF4500' } )
        
        styles.add_css( '.tooltip'   , { 'position': 'relative',
                                             'display' : 'inline-block',
                                             'border-bottom': '1px dotted black' } )
        
        styles.add_css( 'span.tooltip + div.tooltiptext' , { 'visibility': 'hidden',
                                                             'padding': '5px 10px' ,
                                                             'border-radius': '6px',
                                                             'background-color':'white',
                                                             'position': 'absolute',
                                                             'z-index': '1',
                                                             'color': '#000'} ) # color used to be red (#FF4500) but I changed it to black
        
        styles.add_css( 'span.tooltip:hover + div.tooltiptext' , { 'visibility': 'visible' } )
        
        doc.add( styles )
        
        table   = H.Table( border='0' )
        space   = H.TableCell( '&nbsp', width="5px" )
        
        for name,module in [('LibPrep','libprep'), ('Harpoon','harpoon'), ('MagLoading','magloading'),
                            ('COCA'   ,'coca'   ), ('Seq.','sequencing'),('ReSeq.','resequencing') ]:
            row   = H.TableRow()
            label = H.TableCell( name , width="80px", align='right' )
            row.add_cell( label )
            
            if self.debug.modules[module]:
                cls = "active"
            else:
                cls = "inactive"
                
            row.add_cell( space )
            
            status = H.TableCell( '&nbsp', width="20px" )
            status.attrs['class'] = cls
            row.add_cell( status )
            table.add_row( row )
            
        # Create a table of software versions
        vtable = H.Table( border='0' )
        
        row1   = H.TableRow( )
        row1.add_cell( H.TableCell( 'Software Version' , True , width="200px" , align='right' ) )
        row1.add_cell( space )
        row1.add_cell( H.TableCell( self.explog.metrics['ReleaseVersion'], width='80px', align='left' ) )

        row3   = H.TableRow( )
        row3.add_cell( H.TableCell( 'Datacollect Version' , True , width="200px" , align='right' ) )
        row3.add_cell( space )
        row3.add_cell( H.TableCell( str(self.explog.metrics['DatacollectVersion']), width='80px', align='left' ) )
        
        row4   = H.TableRow( )
        row4.add_cell( H.TableCell( 'Scripts Version' , True , width="200px" , align='right' ) )
        row4.add_cell( space )
        row4.add_cell( H.TableCell( str(self.explog.metrics['ScriptsVersion']), width='80px', align='left' ) )
        
        for r in [row1, row3, row4]:
            vtable.add_row( r )

        script_table = H.Table( border='0' )
        dev_scripts  = self.explog.metrics['Valkyrie_Dev_Scripts']
        row1 = H.TableRow( )
        row1.add_cell( H.TableCell( 'Dev. Scripts Used?' , True , width="200px" , align='right' ) )
        row1.add_cell( space )
        tf = H.TableCell( str(dev_scripts), width='80px', align='left' )
        if dev_scripts:
            tf.attrs['class'] = 'warning'
        else:
            tf.attrs['class'] = 'ok'
        row1.add_cell( tf )

        wf   = self.metrics['workflow']
        row2 = H.TableRow( )
        row2.add_cell( H.TableCell( 'Working Directory' , True , width="200px" , align='right' ) )
        row2.add_cell( space )
        if wf['working_directory']:
            row2.add_cell( H.TableCell( wf['working_directory'], width='80px', align='left' ) )
        else:
            row2.add_cell( H.TableCell( 'default', width='80px', align='left' ) )
            
        row3 = H.TableRow( )
        row3.add_cell( H.TableCell( 'git branch' , True , width="200px" , align='right' ) )
        row3.add_cell( space )
        if wf['branch']:
            row3.add_cell( H.TableCell( wf['branch'], width='80px', align='left' ) )
        else:
            row3.add_cell( H.TableCell( 'n/a', width='80px', align='left' ) )
            
        row4 = H.TableRow( )
        row4.add_cell( H.TableCell( 'git commit' , True , width="200px" , align='right' ) )
        row4.add_cell( space )
        if wf['commit']:
            # Let's start by showing the first 11 characters of the hash to match Stash/Bitbucket.
            row4.add_cell( H.TableCell( wf['commit'][:11], width='80px', align='left' ) )
        else:
            row4.add_cell( H.TableCell( 'n/a', width='80px', align='left' ) )
            
        for row in [row1,row2,row3,row4]:
            script_table.add_row( row )
            
        # Include Alarms and then flow rate analysis
        
        alarms   = H.Table( border='0', width='100%' )
        alarmrow = H.TR( )
        
        # Table
        # row per failure type, columns are label, colored box (green if ok, red if not), list of failed steps
        if self.searched_for_blocked_tips:
            blockage_table = H.Table( border='0' )
            space          = H.TableCell( '&nbsp', width="5px" )
            
            # Pipette 1
            row0   = H.TableRow( )
            row0.add_cell( H.TC( 'Tip Blockages', True , align='right' , width="125px") )
            row0.add_cell( space )
            row0.add_cell( H.TableCell( '&nbsp', width="20px" ) )
            
            row1   = H.TableRow( )
            label  = H.TableCell( 'Pipette 1', align='right' , width="125px")
            p1_num = self.metrics['blocked_tips']['p1_count']
            if p1_num == 0:
                alert = H.TableCell( '', width="20px" )
                alert.attrs['class'] = 'active'
            else:
                code  = textwrap.dedent("""\
                <span class="tooltip">%s</span>
                <div class="tooltiptext">%s</div>
                """ % ( str(p1_num),  self.metrics['blocked_tips']['p1_tubes']  ) )
                alert = H.TableCell( code , width="20px" , align='center' )
                alert.attrs['class'] = 'flag'
            for cell in [label, space, alert]:
                row1.add_cell( cell )
                
            # Pipette 2
            row2   = H.TableRow( )
            label  = H.TableCell( 'Pipette 2', align='right' , width="125px")
            space  = H.TableCell( '&nbsp', width="5px" )
            p2_num = self.metrics['blocked_tips']['p2_count']
            if p2_num == 0:
                alert = H.TableCell( '', width="20px" )
                alert.attrs['class'] = 'active'
            else:
                code  = textwrap.dedent("""\
                <span class="tooltip">%s</span>
                <div class="tooltiptext">%s</div>
                """ % ( str(p2_num),  self.metrics['blocked_tips']['p2_tubes']  ) )
                alert = H.TableCell( code , width="20px" , align='center' )
                alert.attrs['class'] = 'flag'
            #fails  = H.TableCell( self.metrics['blocked_tips']['p2_tubes'] , width='600px' )
            for cell in [label, space, alert]:
                row2.add_cell( cell )

            # Warnings (uses of known or suspected blocked tips)
            row3   = H.TableRow( )
            label  = H.TableCell( 'Used Blocked Tips', align='right' , width="125px")
            space  = H.TableCell( '&nbsp', width="5px" )
            bt_num = self.metrics['blocked_tips']['used_blocked_tips_count']
            if bt_num == 0:
                alert = H.TableCell( '', width="20px" )
                alert.attrs['class'] = 'active'
            else:
                code  = textwrap.dedent("""\
                <span class="tooltip">%s</span>
                <div class="tooltiptext">%s</div>
                """ % ( str(bt_num),  self.metrics['blocked_tips']['used_blocked_tips']  ) )
                alert = H.TableCell( code , width="20px" , align='center' )
                alert.attrs['class'] = 'error'
            for cell in [label, space, alert]:
                row3.add_cell( cell )
                
            blockage_table.add_row( row0 )
            blockage_table.add_row( row1 )
            blockage_table.add_row( row2 )
            blockage_table.add_row( row3 )
            
            alarmrow.add_cell( H.TC( blockage_table , width='12%' ) )
        else:
            alarmrow.add_cell( H.TC( '&nbsp', width='12%' ) )
            
        # Need to add info about the TubeBottomLog.csv in the block html, including link to csv
        if self.has_tube_bottom_log:
            # Table
            # row per failure type, columns are label, colored box (green if ok, red if not), list of failed steps
            tip_table = H.Table( border='0' )
            space     = H.TableCell( '&nbsp', width="5px" )
            
            row0   = H.TableRow( )
            row0.add_cell( H.TC( 'Tube Bottom Log', True , align='right' , width="130px") )
            for pipette in ['1','2']:
                row0.add_cell( space )
                try:
                    pip_sn  = textwrap.dedent("""\
                    <span class="tooltip">%s</span>
                    <div class="tooltiptext">%s</div>
                    """ % ( 'P{}'.format(pipette), 'pip serial#: {}'.format(self.metrics['serial_number_pip{}'.format(pipette)]) ) )
                    pip_id = H.TableCell( pip_sn , width="20px" , align='center' )
                    pip_id.attrs['class'] = 'info'
                    row0.add_cell( pip_id )
                except:
                    row0.add_cell( H.TC( 'P{}'.format(pipette), False, align='left', width="20px" ) )


            # Missed Bottom - changed nomenclature based on feedback from Shawn.
            row1   = H.TableRow( )
            label  = H.TableCell( '> 2 mm <em>above</em> zcal', align='right' , width="130px")
            row1.add_cell(label)
            
            for pipette in [1,2]:
                row1.add_cell(space)
                mb_num = self.metrics['bottomlog']['missed_bottom_p{}_count'.format(pipette)]
                if mb_num == 0:
                    alert = H.TableCell( '', width="20px" )
                    alert.attrs['class'] = 'active'
                else:
                    code  = textwrap.dedent("""\
                    <span class="tooltip">%s</span>
                    <div class="tooltiptext">%s</div>
                    """ % ( str(mb_num), self.metrics['bottomlog']['missed_bottom_p{}'.format(pipette)] ) )
                    alert = H.TableCell( code , width="20px" , align='center' )
                    alert.attrs['class'] = 'error'
                row1.add_cell(alert)
                
            # Bent Tube - changed nomenclature based on feedback from Shawn.
            row2   = H.TableRow( )
            label  = H.TableCell( '> 2 mm <em>below</em> zcal', align='right' , width="130px")
            row2.add_cell(label)
            for pipette in [1,2]:
                row2.add_cell(space) 
                bt_num = self.metrics['bottomlog']['bent_tips_p{}_count'.format(pipette)]
                if bt_num == 0:
                    alert = H.TableCell( '', width="20px" )
                    alert.attrs['class'] = 'active'
                else:
                    code  = textwrap.dedent("""\
                    <span class="tooltip">%s</span>
                    <div class="tooltiptext">%s</div>
                    """ % ( str(bt_num), self.metrics['bottomlog']['bent_tips_p{}'.format(pipette)] ) )
                    alert = H.TableCell( code , width="20px" , align='center' )
                    alert.attrs['class'] = 'error'
                row2.add_cell(alert)
                
            tip_table.add_row( row0 )
            tip_table.add_row( row1 )
            tip_table.add_row( row2 )
            
            alarmrow.add_cell( H.TC( tip_table, width='13%' ) )
        else:
            alarmrow.add_cell( H.TC( '&nbsp', width='13%' ) )
        
        
        # Pipette Pressure Tests
        if self.pipPresTests.found_pipPres:
            # Table
            # row per failure type, columns are label, colored box (green if ok, red if not), list of failed steps
            ppt_table = H.Table( border='0' )
            space     = H.TableCell( '&nbsp', width="5px" )
            
            row0   = H.TableRow( )
            row0.add_cell( H.TC( 'PipPres Tests', True , align='right' , width="110px") )
            for pipette in ['1','2']:
                row0.add_cell( space )
                row0.add_cell( H.TC( 'P{}'.format(pipette), False, align='left', width="20px" ) )
            ppt_table.add_row( row0 )

            for type in [('waterWell_fail','water well'),('inCoupler_fail','coupler'),('vacTest_fail','vacuum')]:
                row   = H.TableRow( )
                label  = H.TableCell( type[1], align='right' , width="110px")
                row.add_cell(label)
            
                for pipette in [1,2]:
                    row.add_cell(space)
                    num = self.pipPresTests.results[type[0]]['p{}_count'.format(pipette)] 
                    if num > 0:
                        code  = textwrap.dedent("""\
                        <span class="tooltip">%s</span>
                        <div class="tooltiptext">%s</div>
                        """ % ( str(num), self.pipPresTests.results[type[0]]['p{}'.format(pipette)]) )
                        alert = H.TableCell( code , width="20px" , align='center' )
                        alert.attrs['class'] = 'error'
                    elif self.pipPresTests.results[type[0]]['did_test']:
                        alert = H.TableCell( '', width="20px" )
                        alert.attrs['class'] = 'active'
                    else:
                        alert = H.TableCell( '', width="20px" )
                        alert.attrs['class'] = 'inactive' 
                    row.add_cell(alert)
                ppt_table.add_row( row )
            
            alarmrow.add_cell( H.TC( ppt_table , width='11%' ) )
        else:
            alarmrow.add_cell( H.TC( '&nbsp', width='11%' ) )
        
        # Table for Pipette Malfunction errors and/or pipette tip pickup errors            
        er52_table  = H.Table( border='0' )
        space       = H.TableCell( '&nbsp', width="5px" )
        row0   = H.TableRow( )
        row0.add_cell( H.TC( 'Pipette Errors', True , align='right' , width="150px") )
        for pipette in ['1','2']:
            row0.add_cell( space )
            row0.add_cell( H.TC( 'P{}'.format(pipette), False, align='left', width="15px" ) )
        er52_table.add_row( row0 )
            
        if self.search_for_pipette_errors:
            # First Row for pipette errors er52/timeout/other
            row1 = H.TableRow( )
            label  = H.TableCell( 'er52/timeout/other', align='right' , width="150px")
            row1.add_cell( label )
            for id in [1,2]:
                row1.add_cell(space)
                count = self.pipette_errors['pipette_{}'.format(id)]['count']
                alert = H.TableCell( '', width="20px" )
                if count == 0:
                    alert = H.TableCell( '', width="20px" )
                    alert.attrs['class'] = 'active'
                else:
                    code  = textwrap.dedent("""\
                    <span class="tooltip">%s</span>
                    <div class="tooltiptext">%s</div>
                    """ % ( str(count), self.pipette_errors['pipette_{}'.format(id)]['messages'] ) )
                    alert = H.TableCell( code , width="20px" , align='center' )
                    alert.attrs['class'] = 'error'
                row1.add_cell( alert )
            er52_table.add_row( row1 )
        if self.search_for_tip_pickup_errors:
            row_struggle = H.TableRow( )
            label  = H.TableCell( 'struggle to pickup tip', align='right' , width="150px")
            row_struggle.add_cell(label)
            for id in [1,2]:
                row_struggle.add_cell(space)
                count = self.struggle_to_pickup_tips['pipette_{}'.format(id)]['tip_loc_count'] 
                alert = H.TableCell( '', width="20px" )
                if count == 0:
                    alert = H.TableCell( '', width="20px" )
                    alert.attrs['class'] = 'active'
                else:
                    message = 'Averge number of failed attempts per tip location: {}'.format( self.struggle_to_pickup_tips['pipette_{}'.format(id)]['failed_attempt_avg'] ) 
                    code  = textwrap.dedent("""\
                    <span class="tooltip">%s</span>
                    <div class="tooltiptext">%s</div>
                    """ % ( str(count), message ) )
                    alert = H.TableCell( code , width="20px" , align='center' )
                    alert.attrs['class'] = 'error'
                row_struggle.add_cell( alert )
            er52_table.add_row( row_struggle )
            
            row_unable = H.TableRow( )
            label  = H.TableCell( 'unable to pickup tip', align='right' , width="150px")
            row_unable.add_cell(label)
            for id in [1,2]:
                row_unable.add_cell(space)
                count = len(self.unable_to_pickup_tips['pipette_{}'.format(id)]) 
                alert = H.TableCell( '', width="20px" )
                if count == 0:
                    alert = H.TableCell( '', width="20px" )
                    alert.attrs['class'] = 'active'
                else:
                    message = ', '.join( self.unable_to_pickup_tips['pipette_{}'.format(id)] ) 
                    code  = textwrap.dedent("""\
                    <span class="tooltip">%s</span>
                    <div class="tooltiptext">%s</div>
                    """ % ( str(count), message ) )
                    alert = H.TableCell( code , width="20px" , align='center' )
                    alert.attrs['class'] = 'error'
                row_unable.add_cell( alert )
            er52_table.add_row( row_unable )


        if self.search_for_pipette_errors or self.search_for_tip_pickup_errors:
            alarmrow.add_cell( H.TC( er52_table , width='14%' ) )
        else:
            alarmrow.add_cell( H.TC( '&nbsp', width='14%' ) )

            
        
        # Errors and Warnings for vacuum logs    
        if self.has_vacuum_logs:
            # Table
            # row per lane, columns are label, colored box (green if ok, red if not, black if not used), number of failed processes 
            vacLog_table = H.Table( border='0' )
            space        = H.TableCell( '&nbsp', width="5px" )
            row0         = H.TR( )
            row0.add_cell( H.TC( 'Vacuum Log Report', True , align='right' , width="100px") )
            row0.add_cell( space )
            #row0.add_cell( H.TableCell( '&nbsp', width="20px" ) )
            row0.add_cell( H.TC( 'WF', False, align='right', width="20px" ) )
            row0.add_cell( space )
            row0.add_cell( H.TC( 'PL', False, align='left', width="20px" ) )
            vacLog_table.add_row( row0 )
            
            lanes = ['lane 1','lane 2','lane 3','lane 4','lane 5','lane 0']
            lane_names = ['lane 1','lane 2','lane 3','lane 4','robot waste','bleed valve']
            # Workflow
            for lane,lane_name in zip(lanes,lane_names):
                Vrow1   = H.TableRow( )
                label   = H.TableCell( lane_name, align='right' , width="100px")
                # workflow alerts, only vacuum logs from lanes are in workflow
                if self.metrics['vacLog'][lane]['log_found']:
                    abnormal_num = self.metrics['vacLog'][lane]['abnormal_process_count']
                    if abnormal_num == 0:
                        wf_alert = H.TableCell( '', width="20px" )
                        wf_alert.attrs['class'] = 'active'
                    elif abnormal_num < 2: # one will just be flagged
                        abnormal_processes = self.metrics['vacLog'][lane]['suspected_leaks'] + self.metrics['vacLog'][lane]['suspected_clogs']
                        code  = textwrap.dedent("""\
                        <span class="tooltip">%s</span>
                        <div class="tooltiptext">%s</div>
                        """ % ( str(abnormal_num), abnormal_processes  ))
                        wf_alert = H.TableCell( code , width="20px" , align='center' )
                        wf_alert.attrs['class'] = 'flag'
                    else:
                        abnormal_processes = self.metrics['vacLog'][lane]['suspected_leaks'] + self.metrics['vacLog'][lane]['suspected_clogs']
                        code  = textwrap.dedent("""\
                        <span class="tooltip">%s</span>
                        <div class="tooltiptext">%s</div>
                        """ % ( str(abnormal_num), abnormal_processes  ))
                        wf_alert = H.TableCell( code , width="20px" , align='center' )
                        wf_alert.attrs['class'] = 'error'
                else:
                    wf_alert = H.TableCell('', width="20px" )
                    wf_alert.attrs['class'] = 'inactive'
                # PostLibClean alerts, lanes, robot waste, and bleed valve 
                if self.metrics['vacLog'][lane]['postLib_found'] :# having issues here
                    abnormal_num = self.metrics['vacLog'][lane]['postLib_abnormal_process_count']
                    if abnormal_num == 0:
                        postLib_alert = H.TableCell( '', width="20px" )
                        postLib_alert.attrs['class'] = 'active'
                    elif abnormal_num < 2:
                        abnormal_processes = self.metrics['vacLog'][lane]['postLib_leaks'] + self.metrics['vacLog'][lane]['postLib_clogs']
                        code  = textwrap.dedent("""\
                        <span class="tooltip">%s</span>
                        <div class="tooltiptext">%s</div>
                        """ % ( str(abnormal_num), abnormal_processes  ))
                        postLib_alert = H.TableCell( code , width="20px" , align='center' )
                        postLib_alert.attrs['class'] = 'flag'
                    else:
                        abnormal_processes = self.metrics['vacLog'][lane]['postLib_leaks'] + self.metrics['vacLog'][lane]['postLib_clogs']
                        code  = textwrap.dedent("""\
                        <span class="tooltip">%s</span>
                        <div class="tooltiptext">%s</div>
                        """ % ( str(abnormal_num), abnormal_processes  ))
                        postLib_alert = H.TableCell( code , width="20px" , align='center' )
                        postLib_alert.attrs['class'] = 'error'
                else:
                    postLib_alert = H.TableCell('', width="20px" )
                    postLib_alert.attrs['class'] = 'inactive'
                
                for cell in [label, space, wf_alert, space, postLib_alert]:
                    Vrow1.add_cell( cell )
                vacLog_table.add_row( Vrow1 )
            
                
            # Write the vacuum html files
            self.write_vacuum_html( )
            try:
                self.write_vacuum_postlib_html( )
            except:
                print('Error when trying to create vacuum_postlib_html') ###
                
            alarmrow.add_cell( H.TC( vacLog_table, width='12%' ) )
        else:
            alarmrow.add_cell( H.TC( '&nbsp', width='12%' ) )
        
        #if os.path.exists( 'flow_spark.svg' ):
        if os.path.exists( os.path.join( self.results_dir, 'flow_spark.svg' ) ):
            flow_img = H.Image( os.path.join( 'flow_spark.svg' ), width='100%' )
            fail4    = H.TC( flow_img.as_link() )
            alarmrow.add_cell( fail4 )
        alarms.add_row( alarmrow )
        
        # Section for PostRun and PostChip clean Warnings & Alarms
        if self.debugInfo.hasDebugInfo:
            pr_alarms   = H.Table( border='0', width='90%' )
            pr_row = H.TR( )
            if self.debugInfo.foundConicalClogCheck:  
                # Conical clog check table
                ccc_table = H.Table( border='0' )
                space = H.TableCell( '&nbsp', width="5px" )
                row0 = H.TR( ) 
                row0.add_cell( H.TC( 'Conical Clog Check', True , align='right' , width="100px") )
                row0.add_cell( space )
                ccc_table.add_row( row0 )
                for conical in ['W1','RG','RC','RA','RT']:
                    ccc_row = H.TableRow( )
                    label   = H.TableCell( conical, align='right' , width="50px")
                    ccc_alert = H.TableCell( '', width="20px" )
                    if self.debugInfo.ccc_metrics['conical'][conical]['is_clogged']:
                        # clog in this conical
                        ccc_alert.attrs['class'] = 'error'
                    elif self.debugInfo.ccc_metrics['conical'][conical]['is_clogged']==None:
                        # test not reliable due to low W3 flow
                        ccc_alert.attrs['class'] = 'inactive'
                    else:
                        # no clog in this conical
                        ccc_alert.attrs['class'] = 'active'
                        print('no clog in conical {}'.format(conical))
                    for cell in [label,space,ccc_alert]:
                        ccc_row.add_cell( cell )
                    ccc_table.add_row( ccc_row )
                # Add one more row for W3 flow rate check
                ccc_row = H.TableRow( )
                label   = H.TableCell( 'W3 flow', align='right' , width="50px")
                ccc_alert = H.TableCell( '', width="20px" ) 
                if self.debugInfo.ccc_metrics['low_W3']==None:
                    ccc_alert.attrs['class'] = 'error'
                elif self.debugInfo.ccc_metrics['low_W3']:
                    ccc_alert.attrs['class'] = 'flag'
                else:
                    ccc_alert.attrs['class'] = 'active'
                for cell in [label,space,ccc_alert]:
                    ccc_row.add_cell( cell )
                ccc_table.add_row( ccc_row )
                pr_row.add_cell( H.TC( ccc_table, width='10%') )


            # PostChipClean alarm table- only build if we have postChipClean data to analyze
            if self.debugInfo.postChipClean:
                # sequencing line W3 flow rate check. 
                flow_table = H.Table( border='0' )
                space = H.TableCell( '&nbsp', width="5px" )
                row0 = H.TR( )
                row0.add_cell( H.TC( 'W3 Flow Rate', True , align='right' , width="100px") )
                row0.add_cell( space ) 
                row0.add_cell( H.TC( 'MW', False, align='right', width="20px" ) )
                row0.add_cell( space ) 
                row0.add_cell( H.TC( 'CW', False, align='right', width="20px" ) )
                flow_table.add_row( row0 )
                for lane in ['1','2','3','4']:
                    rowX = H.TableRow( )
                    label   = H.TableCell( 'lane {}'.format(lane), align='right' , width="100px")
                    cells = [label] # start building the cells array that will be added to the row
                    for line in ['PM','PC']:
                        cells.append( space )
                        note = ''
                        # Build message in red box
                        if self.debugInfo.pcc_seq_line_clog_check['{}{}'.format(line,lane)]['flowRate'] < 90:
                            note = 'slightly low W3 flow rate' 
                            color = 'flag'
                        if self.debugInfo.pcc_seq_line_clog_check['{}{}'.format(line,lane)]['flowRate'] < 80:
                            note = 'low W3 flow rate' 
                            color = 'error'
                        # Make alert
                        if len(note)>1:
                            code  = textwrap.dedent("""\
                            <span class="tooltip">%s</span>
                            <div class="tooltiptext">%s</div>
                            """ % ( '!', note  ))
                            alert = H.TableCell( code , width="20px", align='center' )
                            alert.attrs['class'] = color 
                        else:
                            alert = H.TableCell( '', width="20px" )
                            alert.attrs['class'] = 'active'
                        cells.append( alert )
                    for cell in cells:
                        rowX.add_cell( cell )
                    flow_table.add_row( rowX )
                #pr_row.add_cell( H.TC( flow_table, width='15%') )
                
                # Seq line clog check and stuck valve check
                slc_table = H.Table( border='0' )
                space = H.TableCell( '&nbsp', width="5px" )
                row0 = H.TR( )
                row0.add_cell( H.TC( 'Clog Check', True , align='right' , width="100px") )
                row0.add_cell( space ) 
                row0.add_cell( H.TC( 'MW', False, align='right', width="20px" ) )
                row0.add_cell( space ) 
                row0.add_cell( H.TC( 'CW', False, align='right', width="20px" ) )
                slc_table.add_row( row0 )
                for lane in ['1','2','3','4']:
                    rowX = H.TableRow( )
                    label   = H.TableCell( 'lane {}'.format(lane), align='right' , width="100px")
                    cells = [label] # start building the cells array that will be added to the row
                    for line in ['PM','PC']:
                        cells.append( space )
                        note_list = []
                        # Build message in red box
                        attempts = self.debugInfo.pcc_seq_line_clog_check['{}{}'.format(line,lane)]['attempts']
                        if self.debugInfo.pcc_stuck_valves_test != None:
                            if self.debugInfo.pcc_stuck_valves_test['{}{}'.format(line,lane)]['is_stuck']:
                                note_list.append( 'detected stuck valve' )
                                color = 'error'
                                message = '!'
                        if self.debugInfo.pcc_seq_line_clog_check['{}{}'.format(line,lane)]['clog_cleared']:
                            note_list.append( 'clog cleared after {} attempts'.format( attempts ) )
                            color = 'flag'
                            message = str(attempts)
                        if self.debugInfo.pcc_seq_line_clog_check['{}{}'.format(line,lane)]['is_clogged']:
                            note_list.append( 'clog remained after {} attempts'.format( attempts ) )
                            color = 'error'
                            message = str(attempts)
                        # Make alert
                        if len(note_list)>0:
                            notes = ', '.join(x for x in note_list)  
                            code  = textwrap.dedent("""\
                            <span class="tooltip">%s</span>
                            <div class="tooltiptext">%s</div>
                            """ % ( message, notes  ))
                            alert = H.TableCell( code , width="20px", align='center' )
                            alert.attrs['class'] = color 
                        else:
                            alert = H.TableCell( '', width="20px" )
                            alert.attrs['class'] = 'active'
                        cells.append( alert )
                    for cell in cells:
                        rowX.add_cell( cell )
                    slc_table.add_row( rowX )
                #pr_row.add_cell( H.TC( slc_table, width='15%') )
                
                # Build dumping reagents check table
                dump_table = H.Table( border='0' )
                space = H.TableCell( '&nbsp', width="3px" ) # how can I make this space smaller?
                row0 = H.TR( )
                row0.add_cell( H.TC( 'Empty Conicals', True , align='right' , width="100px") )
                row0.add_cell( space )
                for conical in [ 'W1','RG', 'RC', 'RA', 'RT' ]:
                    row0.add_cell( H.TC( conical, False, align='right', width="20px" ) )
                    row0.add_cell( space )
                dump_table.add_row( row0 )
                reagents = [('unused reagents', 'unused reagents'),
                               ('W3 wash', 'W3'),
                               ('nuc cart. res.', 'nuc cartridge residual'),]
                for tup in reagents:
                    row = H.TableRow( )
                    label = H.TableCell( tup[0], align='right' , width="200px") 
                    cells = [label]
                    # Only have box clored red if nuc cartride residual was not completely dumped
                    if tup[0]=='nuc cart. res.':
                        color = 'error'
                    else:
                        color = 'flag'
                    for conical in [ 'W1','RG', 'RC', 'RA', 'RT' ]:
                        cells.append(space)
                        alert = H.TableCell( '', width="20px" )
                        if self.debugInfo.pcc_dump_data[tup[1]][conical]['empty']:
                            alert.attrs['class'] = 'active'
                        else:
                            alert.attrs['class'] = color
                        cells.append(alert)
                    for cell in cells:
                        row.add_cell( cell )
                    dump_table.add_row( row )
                #pr_row.add_cell( H.TC( dump_table, width='15%') )
            
                # Mainifold Leak Check table
                if self.debugInfo.pcc_manifoldLeakCheck != None:
                    man_table = H.Table( border='0' ) 
                    space = H.TableCell( '&nbsp', width="5px" )
                    row0 = H.TR( )
                    row0.add_cell( H.TC( 'Manifold Leak Check', True , align='right' , width="100px") )
                    row0.add_cell( space )
                    man_table.add_row( row0 )
                    for man in self.debugInfo.pcc_manifoldLeakCheck:
                        man_row = H.TableRow( )
                        label   = H.TableCell( man, align='right' , width="50px")
                        man_alert = H.TableCell( '', width="20px" )
                        if self.debugInfo.pcc_manifoldLeakCheck[man]['found_leak']:
                            # True means there was a leak
                            man_alert.attrs['class'] = 'error'
                        else:
                            # no clog in this conical
                            man_alert.attrs['class'] = 'active'
                        for cell in [label,space,man_alert]:
                            man_row.add_cell( cell )
                        man_table.add_row( man_row )
                
                # Add all tables to pr_row - not sure why I don't seem to have control over the positioning
                pr_row.add_cell( H.TC( flow_table, width='10%') )
                pr_row.add_cell( H.TC( slc_table, width='10%') )
                pr_row.add_cell( H.TC( dump_table, width='10%') )
                if self.debugInfo.pcc_manifoldLeakCheck != None:
                    pr_row.add_cell( H.TC( man_table, width='10%') )
            
            pr_alarms.add_row( pr_row ) 
            

        # Actually build the document. Start with message to re-run plugin if necessary. 
        if self.message:
            print('self.message is true')
            doc.add( H.Header( self.message , 4 , style='color:red;') )
        
        # Run type
        doc.add( H.Header( 'Run Type: {}'.format( self.metrics['run_type'] ) , 4 ) )
        
        # Finish span_table depending on if we found vac log files
        links = H.List( ordered=False )

        if self.has_tube_bottom_log:
            tbl_l = H.Link( 'TubeBottomLog.csv' )
            tbl_l.add_body( 'Link to TubeBottomLog.csv' )
            links.add_item( H.ListItem( tbl_l ) )
        else:
            links.add_item( H.ListItem( 'No TubeBottomLog Found' ) )
            
        if self.has_vacuum_logs:
            l = H.Link( 'Workflow_VacuumLog.html' )
            l.add_body( 'Workflow_VacuumLog.html' )
            links.add_item( H.ListItem( l ) )
        else:
            links.add_item( H.ListItem( 'No VacuumLog Found' ) )
            
        if os.path.exists( os.path.join( self.results_dir, 'PostLib_VacuumLog.html' ) ):
            l = H.Link( 'PostLib_VacuumLog.html' )
            l.add_body( 'PostLib_VacuumLog.html' )
            links.add_item( H.ListItem( l ) )
        else:
            links.add_item( H.ListItem( 'No PostLib_VacuumLog Found' ) )
            
        if self.found_chip_images:
            ci = H.Link( 'chipImages.html' )
            ci.add_body( 'chipImages.html' )
            links.add_item( H.ListItem( ci ) )
        else:
            links.add_item( H.ListItem( 'No chipImages found' ) )
            
        if self.found_deck_images:
            di = H.Link( 'deckImages.html' )
            di.add_body( 'deckImages.html' )
            links.add_item( H.ListItem( di ) )
        else:
            links.add_item( H.ListItem( 'No deckImages found' ) )

        if self.debugInfo.hasDebugInfo:
            if self.debugInfo.pinchMeas:
                sp = H.Link('Pinch_Clearing.html')
                sp.add_body( 'Pinch_Clearing.html' )
                links.add_item( H.ListItem( sp ) )
            if (self.debugInfo.foundConicalClogCheck):
                ccc = H.Link('ConicalClogCheck.html' )
                ccc.add_body('ConicalClogCheck.html' )
                links.add_item( H.ListItem( ccc ) )
            if self.debugInfo.postChipClean:
                slc = H.Link('PostChipClean.html' ) 
                slc.add_body('PostChipClean.html' )
                links.add_item( H.ListItem( slc ) )
        else:
            links.add_item( H.ListItem( 'DebugInfo.json not found' ) )

        if self.flows.hasFlowData:
            fd = H.Link('Detailed_Flow_Data.html')
            fd.add_body( 'Detailed_Flow_Data.html' )
            links.add_item( H.ListItem( fd ) )
        else:
            links.add_item( H.ListItem( 'No flow data found' ) )
        
        if self.libpreplog.hasLog:
            if self.libpreplog.hasData:
                lp = H.Link('libPrep_log.html')
                lp.add_body( 'libPrep_log.html' )
                links.add_item( H.ListItem( lp ) )
            else:
                links.add_item( H.ListItem( 'libPrep_log.csv is empty.' ) )
        else:
            links.add_item( H.ListItem( 'libPrep_log.csv not found.' ) )
        
        if self.pipPresTests.found_pipPres:
            ppt = H.Link('PipettePressureTests.html')
            ppt.add_body('PipettePressureTests.html')
            links.add_item( H.ListItem( ppt ) )
        else:
            links.add_item( H.ListItem( 'PipPress directory not found' ) )

        # Add link to oia_timing if it was made
        if os.path.exists( os.path.join( self.results_dir, 'oia_timing.png' ) ):
            oia = H.Link( 'oia_timing.png' )
            oia.add_body( 'OIA timing by block' )
            links.add_item( H.ListItem( oia ) )
            
        # Add link to sample analysis timing if it was made
        if os.path.exists( os.path.join( self.results_dir, 'sample_analysis_timing.png' ) ):
            sat = H.Link( 'sample_analysis_timing.png' )
            sat.add_body( 'Sample Analysis Timing' )
            links.add_item( H.ListItem( sat ) )
            
        span_table = H.Table( border='0', width='100%' )
        row        = H.TableRow()
        left       = H.TableCell( table , width='10%' )
        mid_left   = H.TableCell( vtable , width='20%' )
        mid_right  = H.TableCell( script_table , width='50%' )
        right      = H.TableCell( links, width='20%' )
        
        for cell in [left, mid_left, mid_right, right]:
            row.add_cell( cell )
            
        span_table.add_row( row )
        
        doc.add( span_table )
        doc.add( H.HR() )
        doc.add( H.Header( 'Warnings & Alarms' , 4 ) )
        doc.add( alarms )
        
        if self.debugInfo.hasDebugInfo:
            # PostChipClean section
            if (self.doPostChip and self.debugInfo.postChipClean):
                doc.add( H.HR() ) 
                doc.add( H.Header( 'PostChipClean Warnings & Alarms' , 4 ) )
                doc.add( pr_alarms )
        
            # PostRunClean section
            if self.debugInfo.postRunClean:
                doc.add( H.HR() ) 
                doc.add( H.Header( 'PostRunClean Warnings & Alarms' , 4 ) )
                doc.add( pr_alarms )
             
        # Now for the timing analysis
        doc.add( H.HR() )
        doc.add( H.H( 'Timing Analysis', 3 ) )
        
        if os.path.exists( os.path.join( self.results_dir, 'detailed_workflow_timing.png' ) ):
            wf = H.Image( 'detailed_workflow_timing.png' , width="67%" )
        else:
            wf = H.Image( os.path.join( 'workflow_timing.png' ), width="40%" )
            
        doc.add( wf.as_link() )
        
        if os.path.exists( os.path.join( self.results_dir, 'workflow_pareto.png' ) ):
            doc.add( H.Image( 'workflow_pareto.png' , width='500px' ).as_link() )
            
        if self.csa:
            doc.add( H.Paragraph( 'ValkyrieWorkflow v{}'.format(self.version) , style='font-style: italic;') )
        
        with open( os.path.join( self.results_dir, 'ValkyrieWorkflow_block.html' ), 'w' ) as f:
            f.write( str( doc ) )
                        
    def write_vacuum_html( self ):
        '''Creates html page to display figures for each vacuum process during the workflow'''
        doc = H.Document()
        doc.add( H.Header('Vacuum Logs', 3))
       
        # Determine which lanes have vacuum log figures
        lanes = []
        num_processes = 0
        for lane in ['1','2','3','4']:
            findfile = glob.glob( self.results_dir + '/Lane{}_process*.png'.format(lane) )
            if len(findfile)>0:
                num_processes = len(findfile)
                lanes.append(lane)
                
        table = H.Table( border = '0' ) 
        
        # First row- headings
        row = H.TableRow()
        width_fig = "200px"
        width_process = "400px"
        
        for lane in lanes:
            lane_header = H.Header('Lane {}'.format(lane),3 )
            lane_label =  H.TableCell( lane_header, True, align='center', width=width_fig  ) 
            row.add_cell( lane_label )
        row.add_cell( H.TableCell( '', True, align='center',width=width_fig ) )
        table.add_row( row )
        
        # Following rows, figures
        for i in range(1,num_processes+1):
            fig_row = H.TableRow()
            for lane in lanes:
                findfile = glob.glob( self.results_dir + '/Lane{}_process_{}_*.png'.format(lane,i))[0]
                fileName = findfile.split('/')[-1]
                img = H.Image( os.path.join( fileName ))# remove width='100%' since it makes the figures fuzzy..
                img_cell = H.TableCell( img.as_link() )
                fig_row.add_cell( img_cell )
            process_name = H.Header(fileName.split('_')[3].split('.png')[0].replace('-','/').replace('perc','%'), 4)
            process_name_cell = H.TableCell( process_name, True, align='left', width=width_process )
            fig_row.add_cell( process_name_cell )
            table.add_row( fig_row )

        doc.add( table )

        with open( os.path.join( self.results_dir, 'Workflow_VacuumLog.html' ), 'w' ) as f:
            f.write( str(doc) )

    def write_vacuum_postlib_html( self ):
        '''Creates html page to display figures for each vacuum process during postrun'''
        doc = H.Document()
        doc.add( H.Header('PostLibClean Vacuum Logs', 3))
      
        # Empty figure as place holder
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(1,1,1)
        #ax.set_facecolor('lightgray')
        fig.set_facecolor('lightgray')
        plt.xticks([])
        plt.yticks([])

        # Determine which lanes have vacuum log figures
        lanes = []
        process_number = {'1':[],'2':[],'3':[], '4':[], '5':[], '0':[]}
        for lane in ['1','2','3','4','5','0']:
            findfile = glob.glob( self.results_dir + '/PostLib_Lane{}_process*.png'.format(lane) )
            if len(findfile)>0:
                #num_processes = len(findfile)
                lanes.append(lane)
            for file in findfile:    
                processname = file.split('/')[-1]
                numbers = [int(s) for s in processname.split('_') if s.isdigit() ]
                process_number[lane].append(numbers[0])
        table = H.Table( border = '0' ) 
        
        # First row- headings
        row = H.TableRow()
        width_fig = "200px"
        width_process = "400px"
       
        # Headers
        for lane in lanes:
            lane_header = H.Header('Lane {}'.format(lane),3 )
            if lane == '5':
                lane_header = H.Header('Robot Waste',3 )
            if lane == '0':
                lane_header = H.Header('Bleed Valve',3 )
            lane_label =  H.TableCell( lane_header, True, align='center', width=width_fig  ) 
            row.add_cell( lane_label )
        row.add_cell( H.TableCell( '', True, align='center',width=width_fig ) )
        table.add_row( row )
        
        # This section is for the case where postrun steps are skipped for a particular lane due to clog
        # Iterage through processes. 
        num_proc = 0
        for lane in ['1','2','3','4','5','0']:
            num_proc += len(process_number[lane])
        
        #  Proceeds if there were PostLib processes found 
        if num_proc:
            # Get list of process names from .png files, including lane 5 (Robot waste) and lane 0 (bleed valve) 
            process_name_list = []
            findfiles = glob.glob( self.results_dir + '/PostLib_Lane*_process_*_*.png')
            fileNames = [findfile.split('/')[-1] for findfile in findfiles]
            # List of tuples where the first element is the process name, the second the process start time
            process_tuple = [(fileName.split('_')[4].split('.png')[0],int(fileName.split('_')[3]), fileName.split('_')[1].split('e')[1] ) for fileName in fileNames]
            # Sort by the process start time
            sortedNames = sorted(process_tuple, key=lambda x: x[1])
            # remove duplicates while retaining the order. Only want the first element of each tuple (the names)
            distinct_sortedNames = []
            for i, tup in enumerate(sortedNames):
                process_name = tup[0]
                if i==0:
                    distinct_sortedNames.append(process_name)
                    continue
                if process_name != sortedNames[i-1][0]:
                    distinct_sortedNames.append(process_name)
            
            # build dictionary where each lane has a corresponding array of process names.
            # Only works if names of processes are distinct in each lane. 
            process_name_dict = {}
            for lane in lanes:
                process_name_dict[lane]=[]
                for process_name in distinct_sortedNames:
                    timestamp = '0000' # added to array when that process did not occur in that lane
                    for tuple in sortedNames:
                        if tuple[0]==process_name and tuple[2]==lane:
                            timestamp = tuple[1]
                            if timestamp in process_name_dict[lane]:
                                timestamp = '0001'
                            else:
                                break
                    if timestamp in process_name_dict[lane]:
                        timestamp = '0000'
                    process_name_dict[lane].append(timestamp) # if process was not found

            for i, process_name in enumerate( distinct_sortedNames ):
                fig_row = H.TableRow()
                for lane in lanes:
                    timestamp = process_name_dict[lane][i]
                    try:
                        findfile = glob.glob( self.results_dir + '/PostLib_Lane{}_process_{}_*{}.png'.format(lane,timestamp, process_name))[0]
                        fileName = findfile.split('/')[-1]
                        img = H.Image( os.path.join( fileName ))# remove width='100%' since it makes the figures fuzzy..
                        img_cell = H.TableCell( img.as_link() )
                    except:
                        img_cell = H.TableCell(H.Image(os.path.join( 'empty.png' )))
                    fig_row.add_cell( img_cell )
                process_name = H.Header(process_name.replace('-','/').replace('perc','%'), 4)
                process_name_cell = H.TableCell( process_name, True, align='left', width=width_process )
                fig_row.add_cell( process_name_cell )
                table.add_row( fig_row )
                
        doc.add( table )
        
        # Adding a process check to avoid creating the html file if we don't have any of this data.
        # There are likely better implementations of this but will leave for now. - PW
        if (lanes == []) and (num_proc == 0):
            print( 'No PostLib Vacuum steps were found!  Skipping creation of the html file . . .' )
            return None
        
        with open( os.path.join( self.results_dir, 'PostLib_VacuumLog.html' ), 'w' ) as f:
            f.write( str(doc) )

class DebugInfo:
    '''Class for reading and interacting with DebugInfo.json'''
    def __init__(self, filepath, outdir,expt_start, debug_lines):
        self.expt_start = expt_start
        self.filepath = filepath
        self.outdir   = outdir
        self.plugin_error = False # set to false, change to true if error
        
        try:
            self.debug_lines = debug_lines # used to check if conical clog check was skipped due to low W3 flow 
            print('loading DebugInfo.json...')
            self.pinchMeas     = False # set initial value
            self.postChipClean = False # set initial value
            self.postRunClean  = False # set initial value
            self.foundConicalClogCheck =  False # set initial value
            debugInfo_raw = {}
            try:
                with open( os.path.join( self.filepath , 'DebugInfo.json' ), 'r' ) as f:
                    debugInfo_raw = json.load( f, strict = False )
            except:
                print('DebugInfo.json file not found')
                print('Skipping analysis of DebugInfo.json')
                self.hasDebugInfo = False
                return None
            self.hasDebugInfo = True
            # Next extract pinch clearing measurements, group data by name and lane, make plots, and generate html page.
            pinch_data = self.extract_pinch_data( debugInfo_raw )
            if self.pinchMeas:
                self.plot_pinch_data( pinch_data )
                self.write_pinch_data_html( pinch_data )
            # Next handle postRunClean - only care about the conical clog check, not so much the start and end time of PostRun
            pr_ccc_data = self.extract_postRunClean( debugInfo_raw )
            
            # Next handle postChipClean
            pc_data, pc_ccc_data  = self.extract_postChipClean( debugInfo_raw ) # sets self.postChipClean
            if self.postChipClean:
                print('We found real post chip clean data.')
                # Check chip and main waste lines for clogs. Includes 'FlowRateCheck' and 'Checking for Clogs'  
                # Checking for Clogs: PC1-1, PM1-4. PM = pressure main valve, PC = pressure chip valve. These air valves control the pressure in the pinch regulators, which controls flow
                self.postChip_check_seq_lines_for_clogs( pc_data ) # generates metrics and plots
                self.postChip_pinchManifoldStuckValve( pc_data )   # generates metrics and plots
                self.postChip_dumpConicals( pc_data ) # generates metrics and plots - Confirm reagent dumping was sucessful. Pressure must drop to 1 below highest recorded pressure. First unused reagents, then W3, then nu cartridge residual
                self.postChip_manifoldLeakCheck( pc_data ) # generates metrics and plots
                self.write_PCC_html( )
            
            if self.foundConicalClogCheck: # will be true if found in either postRun or postChip
                if pc_ccc_data != None:
                    print('CCC data from PostChipClean')
                    ccc_data = pc_ccc_data
                elif pr_ccc_data != None:
                    print('CCC data from PostRunClean')
                    ccc_data = pr_ccc_data
                if ccc_data['W3 only']==None:
                    print('Conical Clog Check skipped due to low W3 flow. Skipping Conical Clog Check plot genration')
                else:
                    self.plot_ccc_data( ccc_data )
                    self.write_conical_clog_check_html( ) # keep as pr html, since postChipClean html might look different 
                self.conical_clog_check_metrics = self.build_ccc_data_metric_dict( ccc_data )
        except:
            print('!! Something went wrong when analyzing DebugInfo.json containing postChipClean data. Skipping analysis.')
            self.hasDebugInfo = False
            self.foundConicalClogCheck = False
            self.plugin_error = True
            return None
            
    def extract_pinch_data( self, debugInfo_all ):
        '''Pulls out pinch data from list of dictionaries from DebugInfo.json (input into function as debugInfo_all). Organizes by lane.'''
        name_list_all = []
        for obj in debugInfo_all['objs']:
            if not obj['name'] in ['end','(null)','PostChipClean','PostRunClean','fluidicsAtCust']: # to isolate dictionaries related to pinch measurements
                name_list_all.append(obj['name'])  # for example, a name might be 'RP-Pinch MW4 Clearing' 
        name_list = set(name_list_all) 
        print('name_list: {}'.format(name_list))

        pinch_dict = {}
        keys = ['timestamp','flowRate', 'manPres', 'regPres', 'tankPres'] # timestamp is first since it is used to determine if a new name is needed for reSeq prerun
        lanes_all = []
        for name in name_list:
            found_first_point = False # boolean used to save first time point for pinch measure of each name. Use to determine if there is a reSeq prerun
            include_lane = True       # set to false if timestamp is before experiment start time. Will not include this lane in dict used for plotting
            pinch_dict[name] = { key: [] for key in keys }
            for point in debugInfo_all['objs']:
                if point['name']==name:
                    dict_name = name # dict_name will change if reseq PreRun pinch measure points are found
                    for key in keys:
                        if key =='timestamp': # will be the first key
                            dt_timestamp = datetime.datetime.strptime( point[key] , '%m_%d_%Y_%H:%M:%S' )
                            if dt_timestamp < self.expt_start:
                                include_lane = False
                                print('time stamp {} is before exp start {}. Ignoring'.format(dt_timestamp,self.expt_start))
                                break 
                            if found_first_point:
                                if (dt_timestamp-first_timestamp).total_seconds() > 200:
                                    dict_name = name.replace('PreRun','PreRun_ReSeq')# update the name to include ReSeq
                                    try:
                                        pinch_dict[dict_name][key] 
                                    except:
                                        pinch_dict[dict_name] = { key: [] for key in keys } # need to create the dictionary since it does not yet exist
                            pinch_dict[dict_name][key].append( dt_timestamp )
                            if not found_first_point:
                                first_timestamp = dt_timestamp # use this to determine if there is a large time jump between one PreRun point and the next. If time jump, then the second PreRun is for reseq
                                found_first_point = True
                                try:
                                    lanes_all.append( name.split('CW')[1].split(' ')[0] )
                                except:
                                    pass
                        else:
                            pinch_dict[dict_name][key].append(point[key])
        lanes = list(set(lanes_all)) # convert to list to enable indexing
        if len(lanes)>0:
            self.pinchMeas = True
        else:
            print('Did not find any lanes that underwent pinch measurement')
            return None
        
        org_by_lane = {'Lane {}'.format(lane):{} for lane in lanes }
        for lane in lanes:
            for name in pinch_dict: 
                if name.split('W')[1][0]==lane:
                    org_by_lane['Lane {}'.format(lane)][name.replace(lane, '')] = pinch_dict[name]
                    try:
                        org_by_lane['Lane {}'.format(lane)][name.replace(lane, '')]['start time'] = min( pinch_dict[name]['timestamp'] )
                    except:
                        pass
        # Now determine the order that the pinch measurements occured, and generate a list of names in that order. Will be used in html
        stages = []
        lane_dict = org_by_lane['Lane {}'.format(lanes[0])] #just use the first lane since it doesn't matter
        for key in lane_dict:
            stages.append((key.split(' ')[0], lane_dict[key]['start time']  ))
        sorted_stages = sorted(stages, key=lambda x:x[1])
        seen = set()
        self.stages =  [x[0] for x in sorted_stages if not ( x[0] in seen or seen.add(x[0]) )]  # Remove duplicate stage names while retaining order

        return org_by_lane

    def extract_postRunClean( self, debugInfo_all ):
        for obj in debugInfo_all['objs']:
            pr_data = None
            if obj['name']=='PostRunClean':
                pr_start = datetime.datetime.strptime( obj['StartTime'], "%Y_%m_%d-%H:%M:%S" )
                try:
                    obj['endTime']
                except:
                    print('Found PostRunClean dictionary with no end time. Skip.')
                    continue

                if obj['StartTime']==obj['endTime']:
                    print('Start time and end time are the same for this PostRunClean dictionary. Assume it is fake.')
                elif pr_start < self.expt_start: 
                    print('PostRun Start time is before experiment start time. Skip analysis.')
                else:
                    print('found some PostRunClean data')
                    try: 
                        obj['checkForClogsInConicals']
                        print('conical clog check was done in postRun')
                        pr_data = obj
                        self.postRunClean  = True  # set to true if there are any PostRunClean stuff 
                    except:
                        print('conical clog check not done during postRun')
                        
        ccc_data = self.extract_ccc( pr_data )
        return ccc_data

    def extract_postChipClean( self, debugInfo_all ):
        pc_data = None
        for obj in debugInfo_all['objs']:
            if obj['name']=='PostChipClean':
                pcc_start = datetime.datetime.strptime( obj['StartTime'], "%Y_%m_%d-%H:%M:%S" )
                try: 
                    obj['endTime']
                except:
                    print('PostChipClean dictionary found, however it has no end time. Skip.')
                    continue
                if obj['StartTime']==obj['endTime']:
                    print('Start time and end time are the same for this PostChipClean dictionary. Assume it is fake.')
                elif pcc_start < self.expt_start:
                    print('PostChip Start time is before experiment start time. Skip analysis.')
                else:
                    self.postChipClean = True
                    pc_data = obj
                    break
        
        ccc_data = self.extract_ccc( pc_data ) # this is treated differently from other postChip checks since it also happens during postRun, therefore functions are shared
        return pc_data, ccc_data
    
    def postChip_check_seq_lines_for_clogs( self, pc_data ):
        seq_lines = {'PM1': {}, 'PM2': {}, 'PM3': {}, 'PM4': {}, 'PC1': {}, 'PC2': {}, 'PC3': {}, 'PC4': {}}
        
        for line in seq_lines:
            seq_lines[line] = {'clogcheck_pres':[],'clogcheck_limit':None, 'press_pass':'green', 'flowRate':None, 'flowRate_pass':'green', 'is_clogged': False, 'clog_cleared': False, 'attempts': 0 }
            # First get flow rate
            flowRateChk_key = '{}W {}'.format(line[1], line)
            seq_lines[line]['flowRate'] = pc_data['FlowRateCheck'][flowRateChk_key]
            if seq_lines[line]['flowRate'] > 90:
                plot_color = 'green'
            elif seq_lines[line]['flowRate'] > 80:
                plot_color = 'orange'
            else:
                plot_color = 'red'  
            seq_lines[line]['flowRate_pass'] = plot_color
            # Then pressure
            seq_lines[line]['clogcheck_limit'] = pc_data['Checking for Clogs'][line]['limit']
            for stage in ['start','middle','end']:
                seq_lines[line]['clogcheck_pres'].append( pc_data['Checking for Clogs'][line][stage] )
            # If a clog was detected during 'Checking for clogs', then clogCheck will happen. 
            if line=='PC3':
                print(seq_lines[line])
            if 'clogCheck' in pc_data.keys():
                for check in pc_data['clogCheck']['objs']:
                    if check['name']==line:
                        seq_lines[line]['attempts'] = check['interval']+1
                        updated_pres = []
                        for stage in ['start','middle','end']:
                            updated_pres.append( check[stage] )
                        seq_lines[line]['clogcheck_pres'] = updated_pres
                        
            # determine final is_clogged status and press_pass color
            if (seq_lines[line]['clogcheck_pres'][1] - seq_lines[line]['clogcheck_pres'][2]) > seq_lines[line]['clogcheck_limit']:
                plot_color = 'green'
                if seq_lines[line]['attempts'] > 0:
                    seq_lines[line]['clog_cleared'] = True
            else:
                plot_color = 'red'
                seq_lines[line]['is_clogged']=True
            seq_lines[line]['press_pass'] = plot_color
        
        self.pcc_seq_line_clog_check = seq_lines # later used to build alarm in main block html
        
        # Save metrics
        # for each lane, flowRate_MW, pinchPresTest_is_clogged_MW 
        metrics_dict = {} 
        for lane in [1,2,3,4]:
            metrics_dict['lane_{}'.format(lane)]={}
            for waste in [('CW','PC{}'.format(lane)), ('MW','PM{}'.format(lane))]:
                metrics_dict['lane_{}'.format(lane)][waste[0]]={}
                metrics_dict['lane_{}'.format(lane)][waste[0]]['flowRate'] = seq_lines[waste[1]]['flowRate']
                metrics_dict['lane_{}'.format(lane)][waste[0]]['is_clogged'] = seq_lines[waste[1]]['is_clogged']
                metrics_dict['lane_{}'.format(lane)][waste[0]]['clog_cleared'] = seq_lines[waste[1]]['clog_cleared']
                metrics_dict['lane_{}'.format(lane)][waste[0]]['clog_clearing_attempts'] = seq_lines[waste[1]]['attempts']
        self.pcc_metrics = metrics_dict

        # make the plot
        fig = plt.figure(figsize=(8,4))
        PM1_FR = plt.subplot2grid((2,8), (1,0))
        PM2_FR = plt.subplot2grid((2,8), (1,1), sharey=PM1_FR)
        PM3_FR = plt.subplot2grid((2,8), (1,2), sharey=PM1_FR)
        PM4_FR = plt.subplot2grid((2,8), (1,3), sharey=PM1_FR)
        PC1_FR = plt.subplot2grid((2,8), (1,4), sharey=PM1_FR)
        PC2_FR = plt.subplot2grid((2,8), (1,5), sharey=PM1_FR)
        PC3_FR = plt.subplot2grid((2,8), (1,6), sharey=PM1_FR)
        PC4_FR = plt.subplot2grid((2,8), (1,7), sharey=PM1_FR)
        
        PM1_pres = plt.subplot2grid((2,8), (0,0))
        PM2_pres = plt.subplot2grid((2,8), (0,1), sharey=PM1_pres)
        PM3_pres = plt.subplot2grid((2,8), (0,2), sharey=PM1_pres)
        PM4_pres = plt.subplot2grid((2,8), (0,3), sharey=PM1_pres)
        PC1_pres = plt.subplot2grid((2,8), (0,4), sharey=PM1_pres)
        PC2_pres = plt.subplot2grid((2,8), (0,5), sharey=PM1_pres)
        PC3_pres = plt.subplot2grid((2,8), (0,6), sharey=PM1_pres)
        PC4_pres = plt.subplot2grid((2,8), (0,7), sharey=PM1_pres)

        PM4_pres.set_title('Sequencing Line Clog Check', fontsize=11)
        PM1_FR.set_ylabel('flowRate', fontsize=11)
        PM1_pres.set_ylabel('pinch pressure', fontsize=11)

        for axes in [(PM1_FR, 'PM1'),(PM2_FR, 'PM2'),(PM3_FR, 'PM3'),(PM4_FR, 'PM4'), (PC1_FR, 'PC1'),(PC2_FR, 'PC2'),(PC3_FR, 'PC3'),(PC4_FR, 'PC4')]:
            axes[0].scatter(0,seq_lines[axes[1]]['flowRate'],color=seq_lines[axes[1]]['flowRate_pass'],s=50) 
            if axes[1]=='PM1':
                plt.setp(axes[0].get_yticklabels(),fontsize = 10)
            else:
                plt.setp(axes[0].get_yticklabels(),visible=False)
            axes[0].xaxis.set_ticks_position('none')   
            axes[0].xaxis.set_ticks([0])
            axes[0].set_xticklabels([axes[1]],fontsize=10)
        
        # Plotting and formatting
        max_pres = []
        for axes in [(PM1_pres, 'PM1'),(PM2_pres, 'PM2'),(PM3_pres, 'PM3'),(PM4_pres, 'PM4'), (PC1_pres, 'PC1'),(PC2_pres, 'PC2'),(PC3_pres, 'PC3'),(PC4_pres, 'PC4')]:
            axes[0].plot([0,1,2],seq_lines[axes[1]]['clogcheck_pres'],color=seq_lines[axes[1]]['press_pass']) 
            if axes[1]=='PM1':
                plt.setp(axes[0].get_yticklabels(),fontsize = 10)
                plt.setp(axes[0].get_xticklabels(),visible=False)
            else:
                plt.setp(axes[0].get_yticklabels(),visible=False)
                plt.setp(axes[0].get_xticklabels(),visible=False)
            axes[0].set_xlim([-1,3])
            axes[0].xaxis.set_ticks_position('none')   
            # get max pressure and add to max_pres list. will be used to determine location of N={} label for clearing attempts. all because axes.get_ylim() is returning nonsense
            max_pres.append( max(seq_lines[axes[1]]['clogcheck_pres']) )
        
        # Add i=N to pressure plots in which clog clearing attempts were made
        PM1_pres.set_ylim( PM1_pres.get_ylim()[0], max(max_pres)+0.1 ) # so that there is space for the N= 
        for axes in [(PM1_pres, 'PM1'),(PM2_pres, 'PM2'),(PM3_pres, 'PM3'),(PM4_pres, 'PM4'), (PC1_pres, 'PC1'),(PC2_pres, 'PC2'),(PC3_pres, 'PC3'),(PC4_pres, 'PC4')]:
            if seq_lines[axes[1]]['attempts']>0:
                text = 'N={}'.format(seq_lines[axes[1]]['attempts'])
                loc = max(max_pres)
                axes[0].text(0,loc,text)
            
        fig.subplots_adjust(hspace=0.1, wspace=0)
        
        fig.savefig( os.path.join( self.outdir, 'seq_line_clog_check.png'),format='png',bbox_inches='tight')
        plt.close()

    def postChip_pinchManifoldStuckValve( self, pc_data ):
        '''
        Test each pinch regulator for stuck valves. Pressurize, then open valve to confirm pressure drops. 
        '''
        try:
            pc_data['Pinch Manifold Stuck valve test']
        except:
            print('Pinch manifold stuck valve test not present in PostChipClean.')
            self.pcc_stuck_valves_test = None
            return

        pinchvalves = {'PM1': {}, 'PM2': {}, 'PM3': {}, 'PM4': {}, 'PC1': {}, 'PC2': {}, 'PC3': {}, 'PC4': {}}
        for valve in pinchvalves:
            pinchvalves[valve] = { 'pressure':[], 'limit':None, 'is_stuck': False, 'color':'green' }
            pres_start = pc_data['Pinch Manifold Stuck valve test'][valve]['start']
            pres_end = pc_data['Pinch Manifold Stuck valve test'][valve]['end']
            limit = pc_data['Pinch Manifold Stuck valve test'][valve]['limit']
            pinchvalves[valve]['pressure'].append( pres_start ) 
            pinchvalves[valve]['pressure'].append( pres_end ) 
            pinchvalves[valve]['limit'] = limit
            if pres_end + limit < pres_start:
                pinchvalves[valve]['is_stuck'] = False
                pinchvalves[valve]['color'] = 'green'
            else:
                pinchvalves[valve]['is_stuck'] = True 
                pinchvalves[valve]['color'] = 'red'
        
        self.pcc_stuck_valves_test = pinchvalves # use later in block html

        # Add is_stuck metric to self.pcc_metrics generated in postChip_check_seq_lines_for_clogs
        for lane in [1,2,3,4]:
            for valve in [('CW','PC{}'.format(lane)), ('MW','PM{}'.format(lane))]:
                self.pcc_metrics['lane_{}'.format(lane)][valve[0]]['is_stuck'] = pinchvalves[valve[1]]['is_stuck']
        # make separate plot, however group results into Seq line clog check since the categories are the same.
        fig = plt.figure(figsize=(8,2))
        pm1 = plt.subplot2grid((1,8), (0,0))
        pm2 = plt.subplot2grid((1,8), (0,1), sharey=pm1)
        pm3 = plt.subplot2grid((1,8), (0,2), sharey=pm1)
        pm4 = plt.subplot2grid((1,8), (0,3), sharey=pm1)
        pc1 = plt.subplot2grid((1,8), (0,4), sharey=pm1)
        pc2 = plt.subplot2grid((1,8), (0,5), sharey=pm1)
        pc3 = plt.subplot2grid((1,8), (0,6), sharey=pm1)
        pc4 = plt.subplot2grid((1,8), (0,7), sharey=pm1)
        
        for axes in [(pm1,'PM1'),(pm2,'PM2'),(pm3,'PM3'),(pm4,'PM4'),(pc1,'PC1'),(pc2,'PC2'),(pc3,'PC3'),(pc4,'PC4')]:
            axes[0].plot([0,1],pinchvalves[axes[1]]['pressure'],'-o', color=pinchvalves[axes[1]]['color'])
            axes[0].xaxis.set_ticks([0.5]) 
            axes[0].set_xticklabels([axes[1]], fontsize=10)
            axes[0].xaxis.set_ticks_position('none')
            if axes[1]!='PM1':
                plt.setp(axes[0].get_yticklabels(),visible=False)
            axes[0].set_xlim([-0.5,1.5])
        
        pm4.set_title('Pinch Manifold Stuck Valve Check', fontsize=11)
        plt.setp(pm1.get_yticklabels(),fontsize = 10)
        pm1.set_ylabel('pressure', fontsize=11)
        
        fig.subplots_adjust(wspace=0)
        fig.savefig( os.path.join( self.outdir, 'postChip_pinchStuckValveTest.png'),format='png',bbox_inches='tight')
        plt.close()

    def postChip_dumpConicals( self, pc_data ):
        dump = {'unused reagents' :{'debugKeys':['Dumping all unused reagents', 'all unused reagents'], 'data':{}}, # different keys for different TS versions. Going forward, the first one should only be found
                'W3'  :{'debugKeys':['Dumping all W3 in conicals', 'all W3 in conicals'], 'data':{}},
                'nuc cartridge residual':{'debugKeys':['Dumping nuc cartridge residual', 'nuc catridge residual', 'Dumping nuc catridge residual'], 'data':{}}
                }
        dump_data = {}
        for reagent in dump: 
            nuc_dict = {}
            for key_option in dump[reagent]['debugKeys']:
                try:
                    nuc_dict = pc_data[key_option]
                    break
                except:
                    print('{} not sucessful'.format(key_option))
            dump_data[reagent] = {'W1':{'pres':[],'empty':False}, 'RG':{'pres':[],'empty':False}, 'RC':{'pres':[],'empty':False}, 'RA':{'pres':[],'empty':False}, 'RT':{'pres':[],'empty':False}}
            for nuc in dump_data[reagent]:
                startPres   = nuc_dict['{}_ALL'.format(nuc)]['startPres']
                highestPres = nuc_dict['{}_ALL'.format(nuc)]['highestPres']
                endPres     = nuc_dict['{}_ALL'.format(nuc)]['endPres']
                dump_data[reagent][nuc]['pres'].append( startPres )
                dump_data[reagent][nuc]['pres'].append( highestPres )
                dump_data[reagent][nuc]['pres'].append( endPres )
                if endPres + 1 < highestPres:
                    dump_data[reagent][nuc]['empty']=True
        self.pcc_dump_data = dump_data # use later in html 

        # Save metrics
        for conical in [ 'W1','RG', 'RC', 'RA', 'RT' ]:
            self.pcc_metrics[conical]={}
            for reagent in [('unused_reagents','unused reagents'),('W3','W3'),('nuc_cart_res','nuc cartridge residual')]:
                self.pcc_metrics[conical][reagent[0]]={}
                self.pcc_metrics[conical][reagent[0]]['not_emptied'] = not dump_data[reagent[1]][conical]['empty']

        fig = plt.figure(figsize=(8,2.5))
        nucs = [ 'W1','RG', 'RC', 'RA', 'RT' ] # get list of nucs
        unused = plt.subplot2grid((1,3), (0,0))
        W3     = plt.subplot2grid((1,3), (0,1), sharey=unused)
        res    = plt.subplot2grid((1,3), (0,2), sharey=unused)
        nuc_colors = {'W1':'mediumpurple', 'RG': 'dimgray', 'RC': 'royalblue', 'RA': 'mediumseagreen', 'RT': 'indianred'}
        
        for nuc in nucs:
            unused.plot([0,1,2],dump_data['unused reagents'][nuc]['pres'], '-o', color = nuc_colors[nuc]) 
            W3.plot([0,1,2],dump_data['W3'][nuc]['pres'], '-o', color = nuc_colors[nuc]) 
            res.plot([0,1,2],dump_data['nuc cartridge residual'][nuc]['pres'],'-o', color = nuc_colors[nuc], label=nuc) 
        # legend
        res.legend(loc=(1.05,0.3), fontsize='small',frameon=False)

        # Figure formatting
        unused.xaxis.set_ticks([0,1,2])
        W3.xaxis.set_ticks([0,1,2])
        res.xaxis.set_ticks([0,1,2])
        xlabel = ['start', 'highest', 'end']
        unused.set_xticklabels(xlabel, fontsize=10)
        W3.set_xticklabels(xlabel, fontsize=10)
        res.set_xticklabels(xlabel, fontsize=10)
       
        unused.set_title('1. unused reagents', fontsize=11)
        W3.set_title('2. W3 rinse', fontsize=11)
        res.set_title('3. nuc cart. residual', fontsize=11)

        unused.set_ylabel('pressure', fontsize=11)
        plt.setp(unused.get_yticklabels(),fontsize=10)
        plt.setp(W3.get_yticklabels(),visible=False)
        plt.setp(res.get_yticklabels(),visible=False)
     
        unused.set_xlim([-0.5,2.5])
        W3.set_xlim([-0.5,2.5])
        res.set_xlim([-0.5,2.5])

        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        fig.savefig( os.path.join( self.outdir, 'postChip_dumpConicals.png'),format='png',bbox_inches='tight')
        plt.close()

    def postChip_manifoldLeakCheck( self, pc_data ):
        possible_manifolds = {'PRV':{'debugkey':'PRV leak test'}, 'Pinch':{'debugkey':'Pinch Manifold Leak test'},'REG2':{'debugkey':'REG2 leak test'},'Conical':{'debugkey':'conicalPressureTest'}}
        manifolds = {}
        for man in possible_manifolds:
            if possible_manifolds[man]['debugkey'] in pc_data:
                print( possible_manifolds[man]['debugkey'] )
                possible_manifolds[man]['limit'] = pc_data[possible_manifolds[man]['debugkey']]['limit']
                possible_manifolds[man]['found_leak'] = not (pc_data[possible_manifolds[man]['debugkey']]['state']=='passed')
                possible_manifolds[man]['pressure']=[]
                for stage in ['start','end']:
                    possible_manifolds[man]['pressure'].append( pc_data[possible_manifolds[man]['debugkey']][stage] )
                manifolds[man] = possible_manifolds[man]
            else:
                print('{} not present in PostChipClean'.format(possible_manifolds[man]['debugkey']))
        print('manifoldLeakCheck: {}'.format(manifolds))
        #manifolds = possible_manifolds
        if manifolds == {}:
            self.pcc_manifoldLeakCheck = None
        else:
            self.pcc_manifoldLeakCheck = manifolds # use later in block html

    def extract_ccc( self, data_raw ):
        '''
        Re-organizes conical clog check data and determines pass/fail. Conical clog check is done during both postChipClean and postRunClean
        '''
        # Set thresholds for W3 for num lanes 1,2,3,4. Each tuple is (value below which W3 flow rate is flagged, lower end of normal, upper end of normal). 
        # All values set by CTN based on ~85 runs
        W3_thresholds = [ (75, 100, 130) , (150, 200, 240), (225, 250, 290), (300, 315, 390) ]
        
        ccc_data = {'conical':['W1','RG','RC','RA','RT'],'name':[],'W3 and conical':[],'W3 only':None, 'low W3': False, 'num lanes':None, 'W3 thresholds':None, 'ratio':[],'ratioLimit':0.79, 'is clogged':[]} # initialize dictionary of conical clog check data
        
        # First check if ConicalClogCheck was skipped due to W3 below 50 uL/s
        print('Check debug log to see if ConicalClogCheck was skipped due to W3 below 50 uL/s...')
        for line in self.debug_lines:
            if 'W3 failed' in line:
                print(line)
                ccc_data['low W3'] = None # None for red in block_html, since True will be yellow
                ccc_data['is_clogged']=[None,None,None,None,None,None] # use to make blocks black in block_html
                self.foundConicalClogCheck = True # not really true- I suppose
                return ccc_data

        if data_raw:
            print('Now extracting conical clog check data')
            for conical in ccc_data['conical']:
                for key in data_raw['checkForClogsInConicals']: # iterate over keys, which are names of conicals
                    self.foundConicalClogCheck = True
                    if key[:2]==conical:
                        con_data = data_raw['checkForClogsInConicals'][key]
                        ccc_data['name'].append( con_data['name'] )
                        ccc_data['W3 and conical'].append( con_data['FR'] )
                        ccc_data['W3 only'] = con_data['WFR'] # same for all conicals, measured once at the beginning of test with no valves from conicals open and after flow through WL valves has stabilized
                        ccc_data['ratio'].append( con_data['FRRatio'] )
                        if ccc_data['W3 only']<20:
                            ccc_data['is clogged'].append(None) # W3 flow is very low, and therefore test is not reliable
                            print('W3 flow is very low')
                        else:
                            ccc_data['is clogged'].append( con_data['FRRatio']  > ccc_data['ratioLimit']  ) # Pass = True, Fail = False
                        print( 'ratio {} and ratioLimit {} for conical {}'.format(con_data['FRRatio'],ccc_data['ratioLimit'],conical) )
                        
            ccc_data['num lanes'] = len( [ int(name[2])for name in ccc_data['name'][0].split(',')] ) # normal W3 flow depends on number of lanes in use. Maybe different from active lanes.
            ccc_data['W3 thresholds'] = W3_thresholds[ ccc_data['num lanes']-1 ] # set equal to touple corresponding to number of lanes used in test
            if ccc_data['W3 only'] < ccc_data['W3 thresholds'][0]: # first value of tuple gives lower limit of W3 flow rate that is not flagged as too low
                ccc_data['low W3'] = True
        else:
            ccc_data = None
        return ccc_data
    
    def plot_pinch_data( self, pinch_data ):
        '''Generates plot of pinch cal data from reagent-prime and pre-run'''
        last_lane = max(pinch_data.keys())
        for lane in pinch_data.keys():
            for stage in self.stages:
                if stage=='(null)':
                    continue
                fig = plt.figure(figsize = (4,4))
                flow = plt.subplot2grid((2,1), (0,0))
                pres = plt.subplot2grid((2,1), (1,0), sharex=flow)
            
                try:
                    base_MW = pinch_data[lane]['{} MW Clearing'.format(stage)]
                    base_CW = pinch_data[lane]['{} CW Clearing'.format(stage)]
                    time_0 = min(base_MW['timestamp'] + base_CW['timestamp'])
                except:
                    fig.savefig( os.path.join( self.outdir , 'pinch_{}_{}.png'.format( stage,lane.replace(' ','_') ) ), format='png',bbox_inches='tight' )
                    return

                CW_diff = [time-time_0 for time in base_CW['timestamp']] # convert to relative time with the first point set to time 0
                MW_diff = [time-time_0 for time in base_MW['timestamp']]
                CW_time = [time_point.total_seconds() for time_point in CW_diff]
                MW_time = [time_point.total_seconds() for time_point in MW_diff]
                flow_MW, = flow.plot(MW_time, base_MW['flowRate'], 'o', color='gray', label='Main Waste')
                flow_CW, = flow.plot(CW_time, base_CW['flowRate'], 'o', color='green', label='Chip Waste')
                man_MW,  = pres.plot(MW_time, base_MW['manPres'], 'o', color='red', label='Mainifold')
                man_CW,  = pres.plot(CW_time, base_CW['manPres'], 'o', color='red')
                reg_MW,  = pres.plot(MW_time, base_MW['regPres'], 'o', color='blue', label='Regulator')
                reg_CW,  = pres.plot(CW_time, base_CW['regPres'], 'o', color='blue')
                tank_MW, = pres.plot(MW_time, base_MW['tankPres'], 'o', color='black', label='Tank')
                tank_CW, = pres.plot(CW_time, base_CW['tankPres'], 'o', color='black')
    
                flow.set_ylabel('FlowRate')
                pres.set_ylabel('Pressure')
                pres.set_xlabel('seconds')
                flow.locator_params('y',nbins=5)
                pres.locator_params('y',nbins=5)
                # Leave out title since labels will be in table of html
                #flow.set_title('{} {}'.format(lane ,stage), loc='center', fontsize=14)
                if lane==last_lane:
                    flow.legend(loc=(1.05,0.5),fontsize='small',numpoints=1,frameon=False)
                    pres.legend(loc=(1.05,0.5),fontsize='small',numpoints=1,frameon=False)
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.1)
                plt.setp(flow.get_xticklabels(),visible=False)
                plt.setp(flow.get_yticklabels(),fontsize=10)
                plt.setp(pres.get_yticklabels(),fontsize=10)
                plt.setp(pres.get_xticklabels(),fontsize=10)
                fig.savefig( os.path.join( self.outdir , 'pinch_{}_{}.png'.format( stage,lane.replace(' ','_') ) ), format='png',bbox_inches='tight' )

    def plot_ccc_data(self, ccc_data):
        fig = plt.figure(figsize = (4.5,4.5))
        FR_W3_conical = plt.subplot2grid((3,1), (0,1))
        FR_ratio      = plt.subplot2grid((3,1), (2,0), sharex=FR_W3_conical)
        
        # Determine the marker colors. 
        nuc_colors  = ['mediumpurple','gray','royalblue','mediumseagreen', 'indianred']
        ratio_colors = []
        ratio_marker = []
        last_pass = -1 # use for the legend
        last_fail = -1
        last_unknown = -1
        for i, status in enumerate(ccc_data['is clogged']):
            if status==None:
                ratio_colors.append('black') # test is not reliable due to very low W3 flow
                ratio_marker.append('o')
                last_unknown = i
            elif status:
                ratio_colors.append('red') # suspected to be clogged
                ratio_marker.append('o')
                last_fail = i
            else:
                ratio_colors.append('green')
                ratio_marker.append('o')
                last_pass = i
        # Scatter plot so we can have a different color for each marker
        for i in range(len(ccc_data['ratio'])):
            FR_W3_conical.scatter(i,ccc_data['W3 and conical'][i],  color=nuc_colors[i], s=50)
            if i==last_pass:
                FR_ratio.scatter(i,ccc_data['ratio'][i],  color=ratio_colors[i], marker=ratio_marker[i], s=50, label='pass')
            elif i == last_fail:
                FR_ratio.scatter(i,ccc_data['ratio'][i],  color=ratio_colors[i], marker=ratio_marker[i], s=50, label='fail')
            elif i == last_unknown:
                FR_ratio.scatter(i,ccc_data['ratio'][i],  color=ratio_colors[i], marker=ratio_marker[i], s=50, label='status unknown')
            else:
                FR_ratio.scatter(i,ccc_data['ratio'][i],  color=ratio_colors[i], marker=ratio_marker[i], s=50)
        
        FR_W3_conical.axhline(y=ccc_data['W3 only'], color='black', linestyle='-', label='W3 only')
        FR_ratio.axhline(y=ccc_data['ratioLimit'], color='black', linestyle='--', label='pass/fail limit')

            
        FR_ratio.set_xticklabels(ccc_data['conical'])    
        FR_ratio.xaxis.set_ticks(np.arange(0,5,1))
            
        lanes = [ int(name[2])for name in ccc_data['name'][0].split(',')]
        FR_W3_conical.set_title('Flow through lanes {}'.format(lanes), loc='center', fontsize=14)

        FR_W3_conical.set_ylabel('W3 + conical')
        FR_ratio.set_ylabel('Ratio')
        
        # Whole lot of figure formatting 
        FR_ratio.set_ylim([min(0.6,min(ccc_data['ratio'])),1])
        FR_ratio.set_xlim([-0.5,4.5])
        
        low = ccc_data['W3 thresholds'][1]
        high = ccc_data['W3 thresholds'][2] 
        FR_W3_conical.fill_between(np.arange(-1,6,1),low,high,color='green',alpha='0.2') # need to make legend for this
        if ccc_data['low W3']:
            FR_W3_conical.text(0.25,0.01,'WARNING: low W3 flow', color='red', horizontalalignment='left', verticalalignment='bottom', transform=FR_W3_conical.transAxes,fontsize=12)

        FR_W3_conical.ticklabel_format(axis='y',style='plain',useOffset=False)
        
        FR_W3_conical.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        FR_ratio.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        FR_W3_conical.locator_params('y',nbins=5)
        FR_ratio.locator_params('y',nbins=5)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.12)
        plt.setp(FR_W3_conical.get_xticklabels(),visible=False)
                
        plt.setp(FR_W3_conical.get_yticklabels(),fontsize=10)
        plt.setp(FR_ratio.get_yticklabels(),fontsize=10)
        plt.setp(FR_ratio.get_xticklabels(),fontsize=10)
        
        # Build custom legend --- NOT WORKING, not in use.
        ratio_legend_elements=[
                        Line2D([0],[0],color='black', label='pass/fail limit',linestyle='--'),
                        Line2D([0],[0], marker='o',color='w', markerfacecolor='red', markersize=8, label='fail'),
                        Line2D([0],[0], marker='o',color='w', markerfacecolor='green', markersize=8,label='pass'),
                              ]

        FR_W3_conical.legend(loc=(1.05,0.5),fontsize='small',numpoints=1,frameon=False)
        FR_ratio.legend(loc=(1.05,0.35),fontsize='small',scatterpoints=1,frameon=False)
        #FR_ratio.legend(ratio_legend_elements,loc=(1.05,0.5),fontsize='small',numpoints=1,frameon=False)
        #fig.savefig( os.path.join( self.outdir, 'postRun_clogcheck.png'),format='png',bbox_extra_artists=ldg,bbox_inches='tight')
        fig.savefig( os.path.join( self.outdir, 'conical_clog_check.png'),format='png',bbox_inches='tight')

    def write_pinch_data_html( self, pinch_data ):
        doc = H.Document()
        doc.add( H.Header('Pinch Clearing Measurements',2) )
        
        stages = [stage.split('-')[0] for stage in self.stages]
        lanes = pinch_data.keys()
        lanes.sort()
        table = H.Table( border = '0' )
        
        # No need for lane headings since it is in the figure title, however stage (RP-Pinch or PreRun-Pinch) is not in figure.
        # Lane Headings
        row_header = H.TableRow()
        row_header.add_cell( H.TableCell('',True,align='center') )
        for lane in lanes:
            lane_header = H.Header(lane, 2)
            lane_header_cell = H.TableCell(lane_header, True, align='center')
            row_header.add_cell( lane_header_cell )
        table.add_row( row_header )

        for stage in stages:
            row = H.TableRow()
            stage_name = H.Header(stage.replace('RP','Reagent Prime'),2)
            stage_name_cell = H.TableCell( stage_name, True, align='right' )
            row.add_cell( stage_name_cell ) # Rotate?
            for lane in lanes:
                #img = H.Image( os.path.join( self.outdir, 'pinch_{}_{}.png'.format(stage,lane.replace(' ','_') ) )
                img = H.Image( 'pinch_{}_{}.png'.format('{}-Pinch'.format(stage),lane.replace(' ','_') ) )
                img_cell = H.TableCell( img.as_link() )
                row.add_cell( img_cell )
            table.add_row( row )
        doc.add( table )
        
        with open ( os.path.join( self.outdir, 'Pinch_Clearing.html'), 'w') as f:
            f.write( str(doc) )

    def write_conical_clog_check_html( self ):
        doc = H.Document()
        doc.add( H.Header('Conical Clog Check',2) )
        
        table = H.Table( border = '0' )
        
        # Description
        row_des = H.TableRow()
        description = ('Test during postChipClean to look for clogs in the conical lines. First, W3 is' +  
                      ' flowed through all used lanes with conical valves closed. Then for each' +
                      ' conical, W3 and nucs are flowed through all used lanes. Since the flow sensor' +
                      ' only measures W3 flow, the measured flow rate will be lower when nucs are' +
                      ' flowing. If the flow rate is similar with and without nuc flow (resulting in a' +
                      ' high flow rate ratio), a clog may be present.')
        des_cell = H.TableCell( H.Paragraph(description), width='600px' ) 
        row_des.add_cell( des_cell )
        table.add_row( row_des )

        # Figure
        row = H.TableRow()
        img = H.Image('conical_clog_check.png')
        img_cell = H.TableCell( img.as_link() )
        row.add_cell( img_cell )
        table.add_row( row )
        doc.add( table )
        
        with open ( os.path.join( self.outdir, 'ConicalClogCheck.html'), 'w') as f:
            f.write( str(doc) )

    def write_PCC_html( self ):
        doc = H.Document()
        doc.add( H.Header('PostChipClean Checks',2) )
        
        slc_table = H.Table( border = '0' )
        # Figure: sequencing line clog check
        row = H.TableRow()
        img = H.Image('seq_line_clog_check.png')
        img_cell = H.TableCell( img.as_link() )
        row.add_cell( img_cell )
        slc_table.add_row( row )
        # Description
        row_des = H.TableRow()
        description = ('Before PostChipClean: check for clogs using the W3 flow rate. Possible clog if the flow rate is'+
                        ' less than 90. Even if there is a clog here, it may be removed during PostChipClean.\n'+
                        'After PostChipClean: check for clogs using the pinch pressure.'+
                        ' If the final pressure is at least 0.5 less than the middle pressure,'+
                        ' there is no clog. Green = no clog.')   
        des_cell = H.TableCell( H.Paragraph(description), width='600px' ) 
        row_des.add_cell( des_cell )
        slc_table.add_row( row_des )
        doc.add( slc_table ) 
        doc.add( H.HR() )

        
        if self.pcc_stuck_valves_test !=None:
            pinch_table = H.Table( border = '0' )
            # Figure: pinch manifold stuck valve test 
            row = H.TableRow()
            img = H.Image('postChip_pinchStuckValveTest.png')
            img_cell = H.TableCell( img.as_link() )
            row.add_cell( img_cell )
            pinch_table.add_row( row )
            # Description
            row_des = H.TableRow()
            description = ('Check if pinch manifold valves are stuck. Valves are not stuck if'+
                            ' the pressure drops by at least 2 psi after the valves are open.')
            des_cell = H.TableCell( H.Paragraph(description), width='600px' )
            row_des.add_cell( des_cell )
            pinch_table.add_row( row_des )
            doc.add( pinch_table )
            doc.add( H.HR() )
        
        dump_table = H.Table( border = '0' )
        # Figure: Dumping Reagents from conicals
        row = H.TableRow()
        img = H.Image('postChip_dumpConicals.png')
        img_cell = H.TableCell( img.as_link() )
        row.add_cell( img_cell )
        dump_table.add_row( row )
        # Description
        row_des_dump = H.TableRow()
        description = ('Test to confirm that the nuc conicals have emptied after each conical dumping procedure.'+
                        ' The conicals are pressurized (close to 12 psi), then valves are opened to allow flow.'+
                        ' The pressure will drop only if the conicals are empty. Test ends once the final pressure'+
                        ' is at least 1 psi below highest recorded pressure, or times out.')
        des_cell = H.TableCell( H.Paragraph(description), width='600px' )
        row_des_dump.add_cell( des_cell )
        dump_table.add_row( row_des_dump )
        doc.add( dump_table )
        doc.add( H.HR() )
        
        with open ( os.path.join( self.outdir, 'PostChipClean.html'), 'w') as f:
            f.write( str(doc) )

    def build_ccc_data_metric_dict( self, ccc_data ):
        n_lanes = ccc_data['num lanes']
        metric_dict = { 'W3_flowRate': ccc_data['W3 only'], 'low_W3': ccc_data['low W3'], 'num_lanes': n_lanes, 'conical':{} }
        if ccc_data['W3 only']==None:
            metric_dict['low W3']=True
        for i, conical in enumerate(ccc_data['conical']):
            if ccc_data['W3 only']==None:
                metric_dict['conical'][conical]={}
                metric_dict['conical'][conical]['flowRate'] = None
                metric_dict['conical'][conical]['flowRate_ratio'] = None
                metric_dict['conical'][conical]['is_clogged'] = None
            else:
                metric_dict['conical'][conical]={}
                metric_dict['conical'][conical]['flowRate'] = ccc_data['W3 and conical'][i]
                metric_dict['conical'][conical]['flowRate_ratio'] = ccc_data['ratio'][i]
                metric_dict['conical'][conical]['is_clogged'] = ccc_data['is clogged'][i]  
        self.ccc_metrics = metric_dict
        return metric_dict

class FlowInfo:
    '''
    Class for reading and interacting with pressureInfo.json and flowRateInfo.json. 
    Grouped together since they should be displayed and analyzed together.
    Either analyze both files, or none.
    '''
    def __init__(self, filepath, outdir, flow_order, do_reseq, reseq_flow_order, true_active=None):
        self.filepath = filepath
        print('FILEPATH: {}'.format(self.filepath))
        self.outdir   = outdir
        self.do_reseq = do_reseq
        self.flow_order = flow_order
        self.reseq_flow_order = reseq_flow_order
        print('flow order: {}'.format(flow_order) )
        self.true_active = true_active
        self.plugin_error = False
       
        try:
            # If resequencing happened according to explog, look for flow data files in reseq folder
            self.reseqfilepath = os.path.join( self.filepath, 'reseq' ) 
            print(self.filepath)
            print(self.reseqfilepath)
            if self.do_reseq:
                print('reseq flow order: {}'.format(reseq_flow_order) )
                print('loading reseq pressureInfo.json...')
                reseq_pres_data_raw = {}
                try:
                    with open( os.path.join( self.reseqfilepath , 'pressureInfo.json' ), 'r' ) as f:
                        reseq_pres_data_raw = json.load( f, strict = False )
                except:
                    print('reseq pressureInfo.json file not found')
                    print('----Skipping analysis of reseq flow data.')
                    self.do_reseq = False
                print('loading reseq flowRateInfo.json...')
                reseq_flowR_data_raw = {}
                try:
                    with open( os.path.join( self.reseqfilepath , 'flowRateInfo.json' ), 'r' ) as f:
                        reseq_flowR_data_raw = json.load( f, strict = False )
                except:
                    print('reseq flowRateInfo.json file not found')
                    print('----Skipping analysis of reseq flow data.')
                    self.do_reseq = False
            
            # Look for flow data files
            print('loading pressureInfo.json...')
            pres_data_raw = {}
            try:
                with open( os.path.join( self.filepath , 'pressureInfo.json' ), 'r' ) as f:
                    pres_data_raw = json.load( f, strict = False )
            except:
                print('pressureInfo.json file not found')
                print('----Skipping analysis of all flow data.')
                self.hasFlowData = False
                return None
            print('loading flowRateInfo.json...')
            flowR_data_raw = {}
            try:
                with open( os.path.join( self.filepath , 'flowRateInfo.json' ), 'r' ) as f:
                    flowR_data_raw = json.load( f, strict = False )
            except:
                print('flowRateInfo.json file not found')
                print('Skipping analysis of all flow data.')
                self.hasFlowData = False
                return None
            self.hasFlowData = True
            
            # Next, group by name, make plots, and generate html page.
            flow_data_s, flow_data_f, median_end_time = self.combine_flow_data( pres_data_raw, flowR_data_raw, self.flow_order  )
            if self.do_reseq:
                reseq_flow_data_s, reseq_flow_data_f, reseq_median_end_time = self.combine_flow_data( reseq_pres_data_raw, reseq_flowR_data_raw, self.reseq_flow_order, reseq=True  )
                self.plot_flow_data( reseq_flow_data_s, reseq_flow_data_f, reseq_median_end_time, reseq=True )
                self.plot_per_flow( reseq_flow_data_f, reseq=True )
            
            self.plot_flow_data( flow_data_s, flow_data_f, median_end_time )
            self.plot_per_flow( flow_data_f )
            self.write_flow_data_html( flow_data_s ) 
        except:
            print( '!! Something went wrong when analyzing detailed flow data. Skipping Analysis')
            self.hasFlowData = False
            self.plugin_error = True
            return None

    def combine_flow_data( self, pres_data_raw, flowR_data_raw, flow_order, reseq=False ):
        flow_data_s = [] # list of dictionaries. Each has flowrate and pressure per sec for each flow
        flows = [ int(key.split('flow')[1]) for key in flowR_data_raw.keys() if key != 'end' ]
        flows.sort()
        end_time = [] # will be used to determine if 11s or 13s flow scripts are being used
        for flow in flows:
            flow_dict={}
            flow_dict['flow'] = flow 
            flow_dict['flowRate'] = flowR_data_raw['flow{}'.format(flow)]
            flow_dict['manPres']  = [ point[0] for point in pres_data_raw['flow{}'.format(flow)] ]
            flow_dict['regPres']  = [ point[1] for point in pres_data_raw['flow{}'.format(flow)] ]
            flow_dict['tankPres'] = [ point[2] for point in pres_data_raw['flow{}'.format(flow)] ]
            flow_dict['time']     = [ index*0.1 for index,fl in enumerate(flow_dict['flowRate']) ]
            end_time.append( 0.1 * len(flow_dict['flowRate']) )
            flow_data_s.append(flow_dict)
        # Next, build dictionary of flowrate per flow
        # Use the final time to determine if we are using 13s flow or 11s flow.
        print('What do the end times look like?')
        print('min {}'.format(min(end_time)))
        print('med {}'.format(np.median(end_time)))
        print('max {}'.format(max(end_time)))
        #if np.median(end_time) > 12.5:
        if min(end_time) >  12.5:
            print('Looks like we are using the 13s flow script based on end_time median of {}'.format(np.median(end_time)))
            flow_data_f = {'flow_order':[], 'staging':{'time':2.0,'flowRate':[]}, 'CW+MW':{'time':5.0,'flowRate':[]}, 'CW':{'time':12.0,'flowRate':[]}}
        else:
            print('Looks like we are using the 11s flow script based on end_time median of {}'.format(np.median(end_time)))
            flow_data_f = {'flow_order':[], 'staging':{'time':1.6,'flowRate':[]}, 'CW+MW':{'time':4.3,'flowRate':[]}, 'CW':{'time':10.0,'flowRate':[]}}

        for j,key in enumerate(flow_data_f):
            if key=='flow_order':
                break
            # determine the index that corresponds to the correct time for staging, CW+MW, or CW. Will be used later in plotting
            for i,time in enumerate(flow_dict['time']):
                if time >= flow_data_f[key]['time']:
                    flow_data_f[key]['index']=i
                    break
            print('index: {} at time {} for requested time {}'.format(i-1, time, flow_data_f[key]['time']) )
            for k,flow in enumerate(flows):  # use the index since the first flow might be 1 or might be 0 --> how often is it 1? looks like flow 0 is g, not t
                #print(k)
                try:
                    flow_data_f[key]['flowRate'].append( flow_data_s[k]['flowRate'][i-1] )
                except: 
                    flow_data_f[key]['flowRate'].append( None ) # to handle case where somewhere in between 11s flow and 13s flow is chosen.  
                if j == 0: # only need to do this one time. 
                    if flow==0:
                        flow_data_f['flow_order'].append( 'g' )  
                        print('assigning flow0 to g')
                    else:
                        flow_data_f['flow_order'].append( flow_order[(flow-1) % len(flow_order) ] )

        # need to generate metrics for staging flow rate
        stag_by_nuc = {'g': [] ,'c':[],'a':[],'t':[] }
        for i, nuc in enumerate(flow_data_f['flow_order']):
            stag_by_nuc[nuc].append(flow_data_f['staging']['flowRate'][i])
        metrics_dict = {'g':{}, 'c':{}, 'a':{}, 't':{} }
        for nuc in metrics_dict:
            metrics_dict[nuc]['staging_median'] = np.median( stag_by_nuc[nuc]  )
        print('Staging flow rates: {}'.format(metrics_dict))
        
        # CW and MW+CW flow rates will not depend on the nuc. 
        print('len of flow_data_f CW: {}, first value {}'.format( len(flow_data_f['CW']['flowRate']),flow_data_f['CW']['flowRate'][0] ))
        metrics_dict['CW_median'] = np.median( flow_data_f['CW']['flowRate'] )
        metrics_dict['CW_MW_median'] = np.median( flow_data_f['CW+MW']['flowRate'] )
        try:
            metrics_dict['CW_std']    = np.std( flow_data_f['CW']['flowRate'] )
            metrics_dict['CW_MW_std']    = np.std( flow_data_f['CW+MW']['flowRate'] )
        except:
            print('Removing nones before trying to find standard deviation of CW and CW_MW flow rate')
            CW_FR    = [val for val in flow_data_f['CW']['flowRate'] if val != None]
            CW_MW_FR = [val for val in flow_data_f['CW+MW']['flowRate'] if val != None]
            print('len of flow_data_f no none CW: {}'.format( len(CW_FR) ) )
            print('len of flow_data_f no none CWMW: {}'.format( len(CW_MW_FR) ) )
            metrics_dict['CW_std']    = np.std( CW_FR )
            metrics_dict['CW_MW_std']    = np.std( CW_MW_FR )
        if not reseq:
            self.flow_metrics = metrics_dict   

        return flow_data_s, flow_data_f, np.median( end_time ) 

    def plot_flow_data( self, flow_data, flow_data_f, median_end_time, reseq=False ):
        print('Plotting flow data...')
        flow_order = flow_data_f['flow_order']

        fig = plt.figure(figsize = (5,9.2))
        manPres  = plt.subplot2grid((6,1),(0,0))
        regPres  = plt.subplot2grid((6,1),(1,0), sharex=manPres)
        tankPres = plt.subplot2grid((6,1),(2,0), sharex=manPres)    
        flow = plt.subplot2grid((6,1), (3,5), sharex= manPres)
        nuc_colors = {'g': 'dimgray', 'c': 'royalblue', 'a': 'mediumseagreen', 't': 'indianred'}
        for i,flow_dict in enumerate(flow_data):
            time = flow_dict['time']
            line_color = nuc_colors[ flow_order[i] ] # set the color to match the nuc 
            if i in [0,1,2,3]:
                flow.plot(time, flow_dict['flowRate'], color=line_color, label=flow_order[i] )
            flow.plot(time, flow_dict['flowRate'], color=line_color)
            manPres.plot(time, flow_dict['manPres'], color=line_color)
            regPres.plot(time, flow_dict['regPres'],  color=line_color)
            tankPres.plot(time, flow_dict['tankPres'],  color=line_color)
        flow.set_ylabel('W2 FlowRate', fontsize=12)
        manPres.set_ylabel('Man. Pres.', fontsize=11)
        regPres.set_ylabel('Reg. Pres.', fontsize=11)
        tankPres.set_ylabel('Tank Pres.', fontsize=11)
        flow.set_xlabel('seconds', fontsize=11)
        flow.set_xlim([0,median_end_time+0.5])
        flow.set_ylim([-15,flow.get_ylim()[1]])

        flow.locator_params('y',nbins=6)
        
        # Add vertical lines corresponding to times selected for the flow rate vs flow plot
        for stage in ['staging','CW+MW','CW']:
            time = flow_data_f[stage]['time']
            flow.axvline(time, -15, flow.get_ylim()[1], linestyle='--',color='black')

        if reseq:
            savename = 'flowRate_Pressure_allFlows_RESEQ.png'
            title = 'All {} Reseq Flows'.format(len(flow_data))
        else:
            savename = 'flowRate_Pressure_allFlows.png'
            title = 'All {} Flows'.format(len(flow_data))
        
        manPres.set_title(title, loc='center', fontsize=12)
        flow.legend(loc='upper right',fontsize='small',frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.12)

        for ax in [manPres, regPres, tankPres]:
            plt.setp(ax.get_xticklabels(),visible=False)
            plt.setp(ax.get_yticklabels(),fontsize=10)
            ax.locator_params('y',nbins=4)
        
        plt.setp(flow.get_yticklabels(),fontsize=10)
        plt.setp(flow.get_xticklabels(),fontsize=10)
        fig.savefig( os.path.join( self.outdir, savename),format='png',bbox_inches='tight') 
        plt.close()

    def plot_per_flow( self, flow_data_f, lines=False, reseq=False ):
        '''lines=True in order to connect all flows with the same nuc with a line.'''
        fig = plt.figure(figsize = (10,9)) 
        CW    = plt.subplot2grid((3,1), (0,0))
        CW_MW = plt.subplot2grid((3,1), (1,0), sharex=CW)
        stage = plt.subplot2grid((3,1), (2,0), sharex=CW) 
        
        nuc_colors = {'g': 'dimgray', 'c': 'royalblue', 'a': 'mediumseagreen', 't': 'indianred'}

        # Make thresholds dictionary- for active lanes from 0 to 4. None if no value available.
        # CW+MW is target +/- 10 percent. CW and staging are based on normal values from recent runs
        CWMW_low = [None]
        CWMW_high = [None]
        for num_active in [1,2,3,4]:
            CWMW_low.append( .9*num_active*48. )
            CWMW_high.append( 1.1*num_active*48. )
        
        thresh = {'staging': {'low':[None,26,50,None,90], 'high':[None,33,62,None,115]}, 
                  'CW'     : {'low':[None,17,35,None,74], 'high':[None,20,38,None,80]},
                  'CW+MW'  : {'low':CWMW_low, 'high':CWMW_high},
                 }
        
        if lines:
            # Reorganize data, sorting by nuc
            each_by_nuc = {}
            for key in ['staging','CW+MW','CW']:
                by_nuc = {'g': {'flow':[], 'flowRate':[]},'c':{'flow':[], 'flowRate':[]},'a':{'flow':[], 'flowRate':[]},'t':{'flow':[], 'flowRate':[]} }
                for i, nuc in enumerate(flow_data_f['flow_order']):
                    by_nuc[nuc]['flow'].append(i)
                    by_nuc[nuc]['flowRate'].append(flow_data_f[key]['flowRate'][i])
                each_by_nuc[key]=by_nuc
            # Then plot
            for nuc in nuc_colors:
                CW.plot( each_by_nuc['CW'][nuc]['flow'], each_by_nuc['CW'][nuc]['flowRate'],color=nuc_colors[nuc] )
                CW_MW.plot( each_by_nuc['CW+MW'][nuc]['flow'], each_by_nuc['CW+MW'][nuc]['flowRate'],color=nuc_colors[nuc] )
                stage.plot( each_by_nuc['staging'][nuc]['flow'], each_by_nuc['staging'][nuc]['flowRate'],color=nuc_colors[nuc], label=nuc )
        else:
            # Scatter plot so each nuc can have its own color
            stage.plot( np.arange(len(flow_data_f['flow_order'])) , flow_data_f['staging']['flowRate'], color='goldenrod',zorder=1, linewidth=0.5 )
            CW.plot( np.arange(len(flow_data_f['flow_order'])) , flow_data_f['CW']['flowRate'], color='goldenrod',zorder=1, linewidth=0.5 )
            CW_MW.plot( np.arange(len(flow_data_f['flow_order'])) , flow_data_f['CW+MW']['flowRate'], color='goldenrod',zorder=1, linewidth=0.5 )
            for i, nuc in enumerate(flow_data_f['flow_order']):
                CW.scatter(i, flow_data_f['CW']['flowRate'][i], color=nuc_colors[nuc], s=10,zorder=2 )
                CW_MW.scatter(i, flow_data_f['CW+MW']['flowRate'][i], color=nuc_colors[nuc], s=10,zorder=2 )
                if i in [0,1,2,3]:
                    stage.scatter(i, flow_data_f['staging']['flowRate'][i], color=nuc_colors[nuc], s=10, label=nuc, zorder=2 )
                else:
                    stage.scatter(i, flow_data_f['staging']['flowRate'][i], color=nuc_colors[nuc], s=10, zorder=2 )
        
        # figure formatting 
        stage.set_xlabel('flow number',fontsize=11)
        stage.set_ylabel('staging (t= {} s)'.format(flow_data_f['staging']['time']),fontsize=11)
        CW.set_ylabel('CW (t= {} s)'.format(flow_data_f['CW']['time']),fontsize=11)
        CW_MW.set_ylabel('CW+MW (t= {} s)'.format(flow_data_f['CW+MW']['time']),fontsize=11)
        stage.set_xlim(-5,len(flow_data_f['flow_order'])+5)
        stage.locator_params('y',nbins=5)
        CW.locator_params('y',nbins=5)
        CW_MW.locator_params('y',nbins=5)

        for key, ax in zip(['staging','CW','CW+MW'],[stage,CW,CW_MW]):
            low  = thresh[key]['low'][self.true_active]
            high = thresh[key]['high'][self.true_active]
            if low and high:
                ax.fill_between(np.arange(len(flow_data_f['flow_order'])),low,high, color='green',alpha=0.2,zorder=0, linewidth=0.0)
        
        #plt.xlim(0,40)
        if reseq:
            savename = 'flowRate_vs_flow_scatter_RESEQ.png'
            title = 'Measured W2 Flow Rate for Reseq'
        else:
            savename = 'flowRate_vs_flow_scatter.png'
            title = 'Measured W2 Flow Rate'

        stage.legend(loc=(1.05,1.5), fontsize='small',scatterpoints=1,frameon=False)
        CW.set_title(title, loc='center', fontsize=12)

        fig.subplots_adjust(hspace=0.12)
        plt.setp(CW.get_xticklabels(),visible=False)
        plt.setp(CW_MW.get_xticklabels(),visible=False)
        plt.setp(stage.get_xticklabels(),fontsize=10)
        plt.setp(stage.get_yticklabels(),fontsize=10)
        plt.setp(CW.get_yticklabels(),fontsize=10)
        plt.setp(CW_MW.get_yticklabels(),fontsize=10)
        
        fig.savefig( os.path.join( self.outdir, savename),format='png',bbox_inches='tight')
        plt.close()

    def write_flow_data_html( self, flow_data ):
        doc = H.Document()
        #doc.add( H.Header('Flow Data',2) )
        table = H.Table( border = '0' ) 
        row_des = H.TableRow()
        description = ('CW = flow W2 through chip waste, MW = flow W2 through main waste, '+
                        'staging = flow W2 through chip waste and nuc through main waste.'+
                        ' Abnormally high W2 flow during staging may indicate low flow from nuc conical.')
        des_cell = H.TableCell( H.Paragraph(description), width='600px' ) 
        row_des.add_cell( des_cell )
        table.add_row( row_des )
        
        row_header = H.TableRow() 
        header_cell = H.TableCell(H.Header('Sequencing Flow Data',2))
        row_header.add_cell( header_cell )
        table.add_row( row_header )

        row = H.TableRow()
        img_vflow = H.Image( 'flowRate_vs_flow_scatter.png' )
        img_cell_vflow = H.TableCell( img_vflow.as_link() )
        row.add_cell( img_cell_vflow )
        space = H.TableCell( '&nbsp', width="50px" )
        row.add_cell( space )
        img_vtime = H.Image( 'flowRate_Pressure_allFlows.png' )
        img_cell_vtime = H.TableCell( img_vtime.as_link() )
        row.add_cell( img_cell_vtime )
        table.add_row( row )
        
        if self.do_reseq:
            row_header = H.TableRow() 
            header_cell = H.TableCell(H.Header('Resequencing Flow Data',2))
            row_header.add_cell( header_cell )
            table.add_row( row_header )

            row = H.TableRow()
            img_vflow = H.Image( 'flowRate_vs_flow_scatter_RESEQ.png' )
            img_cell_vflow = H.TableCell( img_vflow.as_link() )
            row.add_cell( img_cell_vflow )
            space = H.TableCell( '&nbsp', width="50px" )
            row.add_cell( space )
            img_vtime = H.Image( 'flowRate_Pressure_allFlows_RESEQ.png' )
            img_cell_vtime = H.TableCell( img_vtime.as_link() )
            row.add_cell( img_cell_vtime )
            table.add_row( row )
        
        doc.add( table )

        with open ( os.path.join( self.outdir, 'Detailed_Flow_Data.html'), 'w') as f:
            f.write( str(doc) )

class LibPrepLogCSV:
    def __init__(self, filepath, outdir, expt_start, module_timing=None, seq_end=None):
        self.expt_start = expt_start
        self.filepath = filepath 
        self.outdir   = outdir
        self.plugin_error = False
        try:
            lpl = LibPrepLog( path=self.filepath )
            self.hasLog   = lpl.found
            self.hasData  = lpl.has_data 

            self.lpl_metrics = {} # start with it empty, populate if we find metrics we want
            if not self.hasData:
                print('LibPrepLog missing data. Has log = {}'.format(self.hasLog))
                return None 
            if module_timing==None:
                self.hasLog = False # even though we actually have it, we don't want to analyze without module_timing data
                return None
            
            # Make keys for libpreplog_metrics
            self.save_headers = ['PCRHeatSink','Heatsink1','Heatsink2','Ambient1','Ambient2','Ambient3']
            for hd in self.save_headers:
                self.lpl_metrics[hd]={}
            
            # Parse module_timing dict from ScriptStatus to get module times and colors
            module_time_colors = self.build_module_dict( module_timing )

            data_trunc = self.truncate_data( lpl.data, expt_start, seq_end=seq_end )
            self.figure_names = []
            self.figure_names_at = [] # amplify target figures
            for header in lpl.header:
                if header=='time':
                    continue
                self.figure_names.append( self.plot_temperature( data_trunc, header, module_time_colors ) )
                if 'libprep' in module_time_colors.keys():
                    if 'amplify target' in module_time_colors['libprep'].keys():
                        self.figure_names_at.append( self.plot_temperature_at( data_trunc, header, module_time_colors ) )
            
            self.write_libPrep_log_html()
        except:
            self.hasLog = False
            print('!! Something went wrong when analyzing LibPrepLog. Skipping Analysis')
            self.plugin_error = True
            return None
    
    def build_module_dict( self, module_timing ):
        mods   = ['libprep','templating','sequencing','resequencing']
        colors = ['blue','green','darkcyan','darkmagenta']
        module_time_colors = {}
        for mod, color in zip(mods,colors):
            try:
                start = module_timing[mod]['overall']['start'] 
                end = module_timing[mod]['overall']['end']
            except:
                print('LibPrepLog Colors: Missing start or end time for {} module.'.format(mod))
                continue
            if (start==None) or (end==None): 
                continue
            module_time_colors[mod]={}
            module_time_colors[mod]['color']=color
            module_time_colors[mod]['start']=start
            module_time_colors[mod]['end']=end
            module_time_colors[mod]['submodule_start'] = [] # use this to add lines for all other submodules in the workflow
            module_time_colors[mod]['submodule_end'] = []
            for submod in module_timing[mod]['submodules']:
                module_time_colors[mod]['submodule_start'].append(module_timing[mod]['submodules'][submod]['start'])
                module_time_colors[mod]['submodule_end'].append(module_timing[mod]['submodules'][submod]['end'])
            if mod == 'libprep':
                try:
                    at_start = module_timing[mod]['submodules']['Ampliseq amplify target']['start']
                    at_end   = module_timing[mod]['submodules']['Ampliseq amplify target']['end']
                except:
                    try:
                        at_start = module_timing[mod]['submodules']['Ampliseq HD amplify target']['start']
                        at_end   = module_timing[mod]['submodules']['Ampliseq HD amplify target']['end']
                    except:
                        print('amplify target submodule not found')
                        continue
                module_time_colors[mod]['amplify target']={}
                module_time_colors[mod]['amplify target']['start'] = at_start
                module_time_colors[mod]['amplify target']['end'] = at_end
        return module_time_colors
    
    def truncate_data( self, data, expt_start, seq_end ):
        ''' we only care about the data during the run'''
        # default/starting indicies will be first and last
        start_index = 0
        end_index = len(data['time'])-1  
        for i,time in enumerate( data['time'] ):
            try:
                if time > expt_start:
                    start_index = i
                    break
            except:
                print('something is wrong with this time point {} at index {}'.format(time,i))
        if seq_end:
            for i,time in enumerate( data['time'] ):
                try:
                    if time > seq_end:
                        end_index = i
                        break
                except:
                    print('something is wrong with this time point {} at index {}'.format(time,i))
        print('TRUNCATE libpreplog start_index: {}'.format(start_index))
        print('TRUNCATE libpreplog   end_index: {}'.format(end_index))
        print('TRUNCATE libpreplog         len: {}'.format(len(data['time'])))
        data_trunc = {}
        for header in data.keys():
            if end_index:
                trunc=data[header][start_index:end_index]
            else:
                trunc=data[header][start_index:]
            data_trunc[header] = trunc
        return data_trunc

    def plot_temperature_at( self, data, header, module_time_colors ): 
        # plot enlarged amplify target 
        time = data['time']
        fig = plt.figure( figsize=(10,2) )
        ax = fig.add_subplot(1,1,1)

        # only include amplify target region in plot  
        at_start = module_time_colors['libprep']['amplify target']['start']
        at_end   = module_time_colors['libprep']['amplify target']['end']
        
        i_start = 0
        i_end = len(time)-1
        for i,t in enumerate( data['time'] ):
            if t > at_start:
                i_start = i
                break
        for i,t in enumerate( data['time'] ):
            if t > at_end:
                i_end = i
                break
        data_at = data[header][i_start:i_end]
        time    = data['time'][i_start:i_end] 
        
        ax.plot( time, data_at, color='black' )
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel( header )
        ax.set_xlabel( 'time' )
        
        ax.set_xlim([time[0], time[-1]])
        bot, top = ax.get_ylim()
        spacer = (top - bot)*0.05
        ax.set_ylim([bot-spacer, top+spacer])
        
        # Add text label for amplify target
        ax.text( at_start + (at_end-at_start)/2 - datetime.timedelta(hours=0.1), top+spacer*1.2, 'Amplify Target', color='black', fontweight='bold', fontsize=8)
        
        savename = 'libPrepLog_'+header+'_amplifytarget.png'
        fig.savefig( os.path.join( self.outdir, savename),format='png',bbox_inches='tight')
        plt.close()
        
        # Here is a good place to save the metrics for the amplify target module
        if header in self.save_headers:
            self.lpl_metrics[header]['amplify_target'] = {}
            # calculate avg, q2, and iqr then save
            try:
                self.lpl_metrics[header]['amplify_target']['q2']   = np.median(data_at)
            except:
                print('Unable to determine q2 of {} for amplify target submodule'.format(header))
            try:
                self.lpl_metrics[header]['amplify_target']['mean'] = data_at.mean()
            except:
                print('Unable to determine mean of {} for amplify target submodule'.format(header))
            try:
                self.lpl_metrics[header]['amplify_target']['std']  = data_at.std()
            except:
                print('Unable to determine std of {} for amplify target submodule'.format(header))
        return savename 
    
    def plot_temperature( self, data, header, module_time_colors ): 
        time = data['time']
        fig = plt.figure( figsize=(10,2) )
        ax = fig.add_subplot(1,1,1)
        ax.plot( time, data[header], color='black' )
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel( header )
        ax.set_xlabel( 'time' )
        x_min = time[0]
        try:
            x_max = max(time[-1],module_time_colors['sequencing']['end'])
        except:
            x_max = time[-1]
        ax.set_xlim([x_min,x_max])
        # Set the y lims to have some space at the top
        bot, top = ax.get_ylim()
        spacer = (top - bot)*0.1
        ax.set_ylim([bot, top+spacer])

        # use ss.self.data module timing to highlight modules  
        for mod in module_time_colors:
            mod_start = module_time_colors[mod]['start']
            mod_end   = module_time_colors[mod]['end']
            mod_color = module_time_colors[mod]['color']
            if 'amplify target' in module_time_colors[mod].keys():
                # calculate lable shift of amplify target label
                at_start = module_time_colors[mod]['amplify target']['start']
                at_end   = module_time_colors[mod]['amplify target']['end']
                at_hours = (at_end-at_start).total_seconds() / 3600.0
                run_hours= (x_max-x_min).total_seconds() / 3600.0
                label_hours = 0.625 * run_hours / 5.5 # approximate length of Amplify Target label in hours
                if label_hours > at_hours:
                    offset = datetime.timedelta(hours=0)
                else:
                    offset = datetime.timedelta(hours=(at_hours-label_hours)/2.0)
                # want this region to stand out within the overall libprep module
                ax.axvspan(mod_start,at_start,facecolor=mod_color, alpha=0.5)
                ax.axvspan(at_end,mod_end,facecolor=mod_color, alpha=0.5)
                ax.axvspan(at_start,at_end,facecolor=mod_color, alpha=0.7)
                ax.text(at_start+offset ,top, 'Amplify Target', color='white',fontweight='bold',fontsize=8)
            else:
                ax.axvspan(mod_start,mod_end,facecolor=mod_color, alpha=0.4)
            # Add module text labels
            ax.text( mod_start + (mod_end-mod_start)/2 - datetime.timedelta(hours=0.6), top+spacer*1.1, mod, color='black', fontweight='bold', fontsize=8)
            for sm_start, sm_end in zip( module_time_colors[mod]['submodule_start'],module_time_colors[mod]['submodule_end']):
                ax.axvline(x=sm_start, color=mod_color, zorder=1,linestyle='-',linewidth=0.1)
                ax.axvline(x=sm_end, color=mod_color, zorder=1,linestyle='-',linewidth=0.1) # should usually overlap with start of next one
        
        savename = 'libPrepLog_'+header+'.png'
        fig.savefig( os.path.join( self.outdir, savename),format='png',bbox_inches='tight')
        plt.close()
        return savename 

    def write_libPrep_log_html( self ):
        doc = H.Document()
        doc.add( H.Header('LibPrepLog Temperature Data',2) )
        table = H.Table( border = '0' ) 
        at = False # at for amplify target
        if len(self.figure_names_at)>0:
            at = True
        for i, figure in enumerate( self.figure_names ):
            row = H.TableRow()
            image = H.Image( figure )
            img_cell = H.TableCell( image.as_link() , align='right')
            row.add_cell( img_cell )
            if at:
                image_at    = H.Image( self.figure_names_at[i] )
                img_cell_at = H.TableCell( image_at.as_link() , align='right')
                row.add_cell( img_cell_at )
            table.add_row( row )

        doc.add( table )

        with open ( os.path.join( self.outdir, 'libPrep_log.html'), 'w') as f:
            f.write( str(doc) )

class PipettePresTests:
    def __init__(self, resultdir, outdir, lanes=[1,2,3,4] ):
        self.filepath = os.path.join(resultdir, 'pipPres' ) 
        self.outdir   = outdir
        self.lanes = lanes # active lanes, or perhaps I should look for all
        self.found_pipPres = True
        self.plugin_error = False
        try:
            if not os.path.exists( self.filepath ):
                self.found_pipPres = False
                return None
            # dict for metrics
            self.results = { 'waterWell_fail': {'p1_count':0, 'p1':[], 'p2_count':0, 'p2':[], 'did_test': False},
                                    'inCoupler_fail': {'p1_count':0, 'p1':[], 'p2_count':0, 'p2':[], 'did_test': False},
                                      'vacTest_fail': {'p1_count':0, 'p1':[], 'p2_count':0, 'p2':[], 'did_test': False},
                                    }

            # Generate graphs from the three tests.
            self.lane_colors = ['black','orangered','forestgreen','mediumblue']
            try:
                self.graphVacPressures()
            except:
                print('!! Something went wrong when plotting the pipPres vacuum pressures')
                self.plugin_error = True
            try:
                self.graphPipPressure('pipPressure')
            except:
                print('!! Something went wrong when plotting the pipPres pip pressures')
                self.plugin_error = True
            try:
                self.graphPipPressure('pipInCoupler')
            except:
                print('!! Something went wrong when plotting the pipPres pipInCoupler pressures')
                self.plugin_error = True

            # convert fail list into comma delimited string
            for key in self.results:
                for pip in [1,2]:
                    self.results[key]['p{}'.format(pip)] = ', '.join( self.results[key]['p{}'.format(pip)] )
            print(self.results)
            
            self.write_pipPres_html()
        except:
            print('!! Something went wrong in PipettePresTests class. Skipping...') 
            self.plugin_error = True
            self.found_pipPres = False

    def graphPipPressure( self, basename ):
        fig = plt.figure(figsize=(5,3))
        pip1 = plt.subplot2grid((1,2),(0,0))
        pip2 = plt.subplot2grid((1,2),(0,1),sharey=pip1)
        
        for lane,color in zip([1,2,3,4],self.lane_colors):
            for pip,ax in zip([1,2],[pip1,pip2]):
                for attempt,alpha in zip( [0,1], [1,0.5] ):
                    filename = '{}_lane{}_try{}_PIP{}.csv'.format(basename,lane,attempt,pip)
                    t = []
                    p = []
                    try:
                        with open(os.path.join(self.filepath,filename)) as csv_file:
                            csv_reader = csv.reader(csv_file, delimiter=',')
                            list_csv = list(csv_reader)
                        p0 = float(list_csv[0][0])
                        for i,line in enumerate(list_csv):
                            t.append(i)
                            p.append(float(line[0])-p0)
                        label = 'L{}_try{}'.format(lane,attempt)
                        ax.plot(t,p,color=color,alpha=alpha,label=label)
                        self.checkPipPresTest( p, pip, label, basename ) 
                    except:
                        pass
        pip2.legend(loc=(1.05,0.2))
        pip1.set_ylabel('pipette pressure reading')
        pip1.set_title('Pipette 1')
        pip2.set_title('Pipette 2')
        plt.setp(pip2.get_yticklabels(),visible=False)

        for ax in [pip1,pip2]:
            ax.set_xlabel('index')
            ax.set_axisbelow(True)
            ax.grid(which='major',linewidth='0.2',color='gray')

        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        fig.savefig( os.path.join( self.outdir, 'pipPres_{}.png'.format(basename) ),format='png',bbox_inches='tight')
        plt.close()

    def checkPipPresTest( self, trace, pipette, trace_name, basename ):
        '''
        for pipPressure: Pass if point 500 is less than -70. 
        for pipInCoupler: pass if final point is less than -500
        '''
        if len(trace) < 10:
            print('Skipping {} {} because array length is only {} points.'.format(basename,trace_name,len(trace)))
            return

        if basename=='pipPressure':
            thresh = -70
            value = trace[500]
            type = 'waterWell_fail'
        elif basename=='pipInCoupler':
            thresh = -500
            value = trace[-1]
            type = 'inCoupler_fail'
            print('{} {} array length {} last point {}'.format(basename,trace_name,len(trace),trace[-1]))
        elif basename=='vac':
            thresh = -5.5
            value = trace[-1]
            type = 'vacTest_fail'
        self.results[type]['did_test'] = True 
        
        if value>thresh:
           self.results[type]['p{}_count'.format(pipette)]+=1
           self.results[type]['p{}'.format(pipette)].append(trace_name)

    def graphVacPressures( self ):
        fig = plt.figure(figsize=(5,3))
        pip1 = plt.subplot2grid((1,2),(0,0))
        pip2 = plt.subplot2grid((1,2),(0,1),sharey=pip1)
        
        for lane,color in zip([1,2,3,4],self.lane_colors):
            for pip,ax in zip([1,2],[pip1,pip2]):
                filename = 'Vacuum_PressureDataLANE{}_PIP{}.csv'.format(lane,pip)
                t = []
                p = []
                try:
                    with open(os.path.join(self.filepath,filename)) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        list_csv = list(csv_reader)
                    for i,line in enumerate(list_csv):
                        t.append(float(line[3]))
                        p.append(float(line[5]))
                    label = 'L{}'
                    ax.plot(t,p,color=color,label=label.format(lane))
                    self.checkPipPresTest( p, pip, label, 'vac' )
                except:
                    pass
        pip2.legend(loc=(1.05,0.3))
        pip1.set_ylabel('vacuum pressure reading (PSI)')
        pip1.set_title('Pipette 1')
        pip2.set_title('Pipette 2')
        plt.setp(pip2.get_yticklabels(),visible=False)

        for ax in [pip1,pip2]:
            ax.set_xlabel('seconds')
            ax.set_axisbelow(True)
            ax.grid(which='major',linewidth='0.2',color='gray')

        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        fig.savefig( os.path.join( self.outdir, 'pipPres_vacPressures.png'),format='png',bbox_inches='tight')
        plt.close()
    
    def write_pipPres_html( self ):
        doc = H.Document()
        doc.add( H.Header('Pipette Pressure Tests',2) )
        doc.add( H.HR() )
        
        images  = ['pipPres_pipPressure.png','pipPres_pipInCoupler.png','pipPres_vacPressures.png']
        headers = [ 'Pipette Pressure: aspirate from water well',
                    'Pipette Pressure: aspirate from chip coupler',
                    'Vacuum Pressure: test vacuum leak caused by pipette' ]
        
        des_pipPressure  = ( 'Pressure curves are captured when each pipette aspirates from the water well during'+
                             ' post run deck clean. Two aspirations are performed per active lane per pipette. The pressure'
                             ' curves are baseline corrected by setting the first point equal to zero. A bad '+
                             ' pipette will have a pressure reading that does not significantly deviate from baseline.')   
        
        des_pipInCoupler = ( 'Pressure curves are captured when each pipette aspirates from each active lane of'+
                             ' the chip coupler. Two aspirations are performed per active lane per pipette. The pressure'
                             ' curves are baseline corrected by setting the first point equal to zero. If the tip attatchment'+
                             ' is good, the aspirate will stop prematurely because the pipette thinks there is a blockage.'+   
                             ' For a bad tip attatchement, the pressure deviation from baseline will be small and the pressure'+
                             ' will return to baseline upon completing the aspiration.')   
            
        des_vacPressures  = ( 'The pipete tip is inserted into chip coupler and vac pressure is monitored as'+
                              ' the vac is pumped down to -6PSI. For a good pipette, pressure should remain'+
                              ' close to -6PSI. A bad tip will act as a leak the vac pressure will drift'+
                              ' towards baseline after reaching the target pressure of -6PSI.')   

        descriptions = [ des_pipPressure, des_pipInCoupler, des_vacPressures ]

        for (im, header, des) in zip(images,headers,descriptions):
            if os.path.exists(os.path.join( self.outdir, im)):
                doc.add( H.Header(header,3) )
                vacpres_table = H.Table( border = '0' )
                row = H.TableRow()
                img = H.Image(im)
                img_cell = H.TableCell( img.as_link() )
                row.add_cell( img_cell )
                vacpres_table.add_row( row )
                # Description
                row_des = H.TableRow()
                des_cell = H.TableCell( H.Paragraph(des), width='600px' ) 
                row_des.add_cell( des_cell )
                vacpres_table.add_row( row_des )
                doc.add( vacpres_table ) 
                doc.add( H.HR() )
            
        with open ( os.path.join( self.outdir, 'PipettePressureTests.html'), 'w') as f:
            f.write( str(doc) )

class SummaryLog:
    """ Class for reading and interacting with summary log files on Valkyrie TS """
    def __init__( self, filepath, name='' ):
        self.filepath = filepath
        self.timing   = {}
        if name:
            self.name = name
        
        self.read( )
        self.calc_durations( )
        
    def read( self ):
        log_start = None
        summary_regex = re.compile( '''(?P<type>[\w]+)\s+(?P<ts>[\s\w:,]+)\s+\-\s+(?P<msg>.*)''' )
        with open( self.filepath , 'r' ) as f:
            for line in f.readlines():
                m = summary_regex.match( line )
                if m:
                    x         = m.groupdict()
                    timestamp = datetime.datetime.strptime( x['ts'].split(',')[0], "%d %b %Y %H:%M:%S" )
                    
                    # If this is the first time through the log file, record the first timestamp as the assumed first step timing.
                    if log_start == None:
                        log_start = timestamp
                    
                    if x['type'] == 'INFO':
                        if 'starting' in x['msg'].lower():
                            module = x['msg'].replace('.','').split()[1]
                            if module not in self.timing:
                                self.timing[module] = {}
                            self.timing[module]['start'] = timestamp
                        if 'finished' in x['msg'].lower():
                            module = x['msg'].replace('.','').split()[0]
                            if module not in self.timing:
                                self.timing[module] = {}
                            self.timing[module]['end']   = timestamp
                            
                            # Check to see that the module started.
                            # If not, assume it was first and use the log_start timestamp as the start.
                            if 'start' not in self.timing[module].keys():
                                self.timing[module]['start'] = log_start
                                
    def calc_durations( self ):
        for module in self.timing:
            self.timing[module]['duration'] = self.get_duration( self.timing[module] )
            
    def get_start_time( self ):
        """ Returns the start time of the first module. """
        try:
            start_time = sorted( [self.timing[m]['start'] for m in self.timing] )[0]
        except:
            print('Unable to find start time. Here is the timing dict:')
            start_time = None
            print(self.timing)
        return None
    
    def get_end_time( self ):
        """ Returns the end time of the last module. """
        total_end = []
        for m in self.timing:
            # in case there are m's with no end time
            try:
                total_end.append( self.timing[m]['end'] )
            except:
                print('{} does not have an end time'.format(m))

        #return sorted( [self.timing[m]['end'] for m in self.timing] )[-1]
        return sorted( total_end  )[-1]
    
    @staticmethod
    def get_duration( sm_dict ):
        """ returns duration in hours """
        try:
            return float( (sm_dict['end'] - sm_dict['start']).total_seconds() / 3600. )
        except( TypeError, KeyError ):
            return 0.
        
class ScriptStatus:
    def __init__( self, filepath , true_start ):
        """
        Initializes the script status object.  true_start is the explog start time that is the TRUE reference start
        """
        self.filepath   = filepath
        self.start_time = true_start
        print( true_start )
        self.set_fieldnames( )
        
        # Initialize timing.  Will be a dictionary of dictionaries.
        # - Each will have 'overall' key but will then have a dictionary of submodules.
        # - Each module will have keys: start, end, tips_start, tips_end, duration, used_tips
        # Note that we will want to add reagent prime back in sometime.
        self.data = { 'libprep' :   { 'overall': {} , 'submodules': {} },
                      'templating': { 'overall': {} , 'submodules': {} },
                      'analysis':   { 'overall': {} , 'submodules': {} },
                      'sequencing': { 'overall': {} , 'submodules': {} },
                      'resequencing': { 'overall': {} , 'submodules': {} },
                      'postrun':    { 'overall': {} , 'submodules': {} },
        }
        
        # Initialize attributes
        self.parallelPostLib = None
        
    def set_fieldnames( self, *fieldnames ):
        if fieldnames:
            self.fieldnames = fieldnames
        else:
            self.fieldnames = ['time','elapsed_hours','module','submodule','status','used_tips']
            
            
    def update_data( self ):
        """ Runs through the modules and calculates durations and used tips if possible. """
        def get_duration( sm_dict ):
            """ returns duration in hours """
            try:
                return float( (sm_dict['end'] - sm_dict['start']).seconds / 3600. )
            except( TypeError, KeyError ):
                return 0.
        
        def get_used_tips( sm_dict ):
            """ returns tips used in submodule """
            try:
                return int( sm_dict['tips_end'] - sm_dict['tips_start'] )
            except( TypeError, KeyError ):
                return 0
            
        for module in self.data:
            if module in ['sequencing']:
                continue
            for submodule in self.data[module]['submodules']:
                sm = self.data[module]['submodules'][submodule]
                sm['duration']  = get_duration ( sm )
                sm['used_tips'] = get_used_tips( sm )
                
        # Update used tips per module in overall
        for module in self.data:
            o = self.data[module]['overall']
            if module in ['sequencing']:
                o['used_tips'] = get_used_tips( o )
            else:
                o['used_tips'] = 0
                for sm in self.data[module]['submodules']:
                    o['used_tips'] += self.data[module]['submodules'][sm]['used_tips']
                    
                    
    def parse_line( self, line ):
        """ Parses each line of the csv file into useful information. """
        module_ref = { 'library preparation': 'libprep' , 'templating': 'templating', 'post-run': 'postrun' }
        
        # Read in and create datetime timestamp
        timestamp = datetime.datetime.strptime( line['time'], "%Y_%m_%d-%H:%M:%S" )
        if timestamp < self.start_time:
            print('skipping submodule in ScriptStatus file since it occured before true start time')
            return None

        # Get module.  Ignore if it isn't part of the desired information. (e.g. 'run')
        try:
            module = module_ref[ line['module'].strip() ]
        except KeyError:
            # Let's get out of here.
            return None
        
        # If we made it here, something useful occurred.  Let's update self.data
        # First check if submodule exists yet.
        submodule = line['submodule'].strip()
        start     = line['status'].strip() == 'started'
        end       = line['status'].strip() == 'completed'
        
        # Lets make resequencing a proper module
        if module in ['templating']:
            if submodule == 'Resequencing':
                module = 'resequencing'
                submodule = 'templating' # should line up with resequencing prep in debug get_overall_timing 
        
        if module in ['postrun']:
            # The postrun info in ScriptStatus is the PostLibClean
            if submodule == 'main':
                submodule = 'postlibclean'
            else:
                submodule = 'postlibclean {}'.format(submodule)
                self.parallelPostLib = True
            
        if module in ['sequencing']:
            # There are no sequencing submodules yet
            pass
        elif submodule not in self.data[module]['submodules']:
            self.data[module]['submodules'][submodule] = { 'start': None,
                                                           'end': None,
                                                           'tips_start': 0,
                                                           'tips_end': 0   }
            
        # Now that we're sure the submodule is in the dictionary, let's populate info from this row
        if module in ['sequencing']:
            sm = self.data[module]['overall']
        else:
            sm = self.data[module]['submodules'][submodule]
            
        if start:
            sm['start']      = timestamp
            sm['tips_start'] = int( line['used_tips'].strip() )
            if module in ['postrun']:
                self.data[module]['overall']['start'] = timestamp  # not accurate when parallel postLibClean is enabled. Update later in code, where seq end is already defined
            
        if end:
            sm['end']        = timestamp
            sm['tips_end']   = int( line['used_tips'].strip() )
            if module in ['postrun']:
                self.data[module]['overall']['end'] = timestamp
            
        return None
    
        
    def add_overall_timing( self , timing_dict, run_type,reseq=False ):
        """ Adds ValkyrieDebug.timing to the self.data for the 'true' module start/end times. """
        # This is a mess... 
        for k in timing_dict:
            if '_' in k:
                m, ts = k.split('_')
                start = ts=='start'
                if m == 'library':
                    module = 'libprep'
                else:
                    module = m
                   
                if module.split('-')[0]=='resequencing':
                    submodule = module.split('-')[1]
                    module = 'resequencing'
                    if 'sequencing' not in self.data[module]['submodules']:
                        print('Adding submodule {} to module {} in self.data'.format(submodule,module))
                        self.data[module]['submodules']['sequencing']      = { 'start': None,
                                                                              'end': None,
                                                                              'tips_start': 0,
                                                                              'tips_end': 0   }
                    if 'prerun chipcal' not in self.data[module]['submodules']:
                        print('Adding submodule {} to module {} in self.data'.format(submodule,module))
                        self.data[module]['submodules']['prerun chipcal']      = { 'start': None,
                                                                              'end': None,
                                                                              'tips_start': 0,
                                                                              'tips_end': 0   }
                    if submodule=='templating':
                        if start:
                            self.data[module]['overall']['start'] = timing_dict[k]
                    if submodule=='sequencing':
                        if start:
                            self.data[module]['submodules'][submodule]['start']=timing_dict[k]
                        else:
                            self.data[module]['overall']['end'] = timing_dict[k]
                            self.data[module]['submodules'][submodule]['end']=timing_dict[k]

                
                if module == 'sequencing':
                    # Create initial values if needed
                    if module not in self.data[module]['submodules']:
                        self.data[module]['submodules']['sequencing']     = { 'start': None,
                                                                              'end': None,
                                                                              'tips_start': 0,
                                                                              'tips_end': 0   }
                        self.data[module]['submodules']['prerun chipcal'] = { 'start': None,
                                                                              'end': None,
                                                                              'tips_start': 0,
                                                                              'tips_end': 0   }
                if start:
                    if m == 'sequencing':
                        if run_type == 'Sequencing Only':
                            print('sequencing only run, start time is {}'.format(self.start_time))
                            self.data[module]['overall']['start'] = self.start_time
                            self.data[module]['submodules']['prerun chipcal']['start'] = self.start_time
                        else:
                            seq_start = timing_dict['templating_end']
                            self.data[module]['overall']['start'] = seq_start
                            self.data[module]['submodules']['prerun chipcal']['start'] = seq_start
                            
                        # Update submodules
                        self.data[module]['submodules']['sequencing']['start']   = timing_dict[k]
                        self.data[module]['submodules']['prerun chipcal']['end'] = timing_dict[k] # this is wrong for reseq runs since seq start is overwritten
                    else:
                        self.data[module]['overall']['start'] = timing_dict[k]
                else:
                    if m == 'sequencing':
                        self.data[module]['overall']['end'] = timing_dict[k]
                        self.data[module]['submodules']['sequencing']['end'] = timing_dict[k]
                    else:
                        self.data[module]['overall']['end'] = timing_dict[k]
        
        # For reseq runs, must correct the seq-seq end time. For now, this is the best I can do. 
        if reseq:
            if self.data['sequencing']['submodules']['sequencing']['end'] == self.data['resequencing']['submodules']['sequencing']['end']:
                print('sequencing end time is the same as resequencing end time. Switching it to resequencing templating start time')
                self.data['sequencing']['submodules']['sequencing']['end'] = self.data['resequencing']['submodules']['templating']['start']
                self.data['sequencing']['overall']['end'] = self.data['resequencing']['submodules']['templating']['start']
            self.data['resequencing']['submodules']['prerun chipcal']['start'] = self.data['resequencing']['submodules']['templating']['end']
            self.data['resequencing']['submodules']['prerun chipcal']['end'] = self.data['resequencing']['submodules']['sequencing']['start']
            # get resequencing submodule durations
            for sm in self.data['resequencing']['submodules']:
                o = self.data['resequencing']['submodules'][sm]
                try:
                    o['duration'] = float( (o['end'] - o['start']).seconds / 3600. )
                except( KeyError, TypeError ):
                    o['duration'] = 0.

        # Update overall durations
        for module in self.data:
            o = self.data[module]['overall']
            try:
                o['duration'] = float( (o['end'] - o['start']).seconds / 3600. )
            except( KeyError, TypeError ):
                o['duration'] = 0.
                
        # Get sequencing submodule durations
        for sm in self.data['sequencing']['submodules']:
            o = self.data['sequencing']['submodules'][sm]
            try:
                o['duration'] = float( (o['end'] - o['start']).seconds / 3600. )
            except( KeyError, TypeError ):
                o['duration'] = 0.
        
        print('FINAL SELF.DATA: {}'.format(self.data)) 

    def add_postrun_modules( self, postchip, postrun ):
        """ Adds postchip or postrun timing modules to the overall timing.  Should only ever be either-or. """
        if postchip and postrun:
            print( "ERROR!  Somehow we detected both postchip and postrun cleans, this should not be possible!" )
            print( " -- No updates made to module timing!" )
            return None
        
        # Add the appropriate submodule then update overall information for postrun module
        o = self.data['postrun']['overall']
        if postchip:
            self.data['postrun']['submodules']['postchip'] = postchip
            o['end'] = postchip['end']
        elif postrun:
            self.data['postrun']['submodules']['postrun']  = postrun
            try:
                o['end'] = postrun['end']
            except:
                o['end'] = None
                print("Postrun end time not found?!")
            
        try:
            o['duration'] = float( (o['end'] - o['start']).seconds / 3600. )
        except( KeyError, TypeError ):
            o['duration'] = 0.
            print("Setting duration to zero.")
            
            
    def read( self ):
        """ Reads the csv file """
        with open( self.filepath , 'r' ) as ss:
            reader = csv.DictReader( ss, fieldnames=self.fieldnames )
            for line in reader:
                # Skip the first line
                if line['time'].strip() == 'time':
                    continue
                self.parse_line( line )

        # Now skipping this so we can update the post-run information prior to updating the data.
        # Update the data and add delta fields.
        # self.update_data( )
        
        
    def get_relative_time( self, timestamp ):
        """ Returns the relative time in hours since the official start time, explog.start_time """
        #if timestamp == None:
        #    print('timestamp is none')
        #    return 0 
        
        try:
            rel_time = (timestamp - self.start_time).total_seconds() / 3600. # convert to hours
        except:  
            timestamp = datetime.datetime.strptime(timestamp, '%m/%d/%Y %H:%M:%S') 
            rel_time = (timestamp - self.start_time).total_seconds() / 3600.
        
        # The year of module timestamps may be wrong around the new year since the year is guessed in debug_reader get_timestamp
        # Here we will check if the relative time is excessive, and recalculate the rel_time if it is.
        if (timestamp - self.start_time).days > 365:
            corrected_timestamp =  timestamp.replace(self.start_time.year) 
            rel_time = self.get_relative_time(corrected_timestamp)
        
        return rel_time 
    
    
    def get_submodule_times( self ):
        """ Helper function to extract submodule timing. """
        abbrevs = { 'libprep': 'Lib',
                    'templating': 'Temp',
                    'sequencing': 'Seq',
                    'resequencing': 'ReSeq',
                    'analysis':'',
                    'postrun':'Postrun'}
        
        submod_tuples = []
        
        # Get submodule times
        for module in self.data:
            abb = abbrevs[module]
            if self.data[module]['submodules']:
                for sm in self.data[module]['submodules']:
                    sm_data = self.data[module]['submodules'][sm]
                    
                    if abb:
                        label = ':'.join( [abb,sm] )
                    else:
                        label = sm
                        
                    try:
                        submod_tuples.append( ( label, sm_data['duration'] ) )
                    except:
                        submod_tuples.append( ( label, 0 ) ) # when there is no end time... but where is this set?  
                        sm_data['duration'] = 0 # i don't know where this is set- but seems like it is needed
            else:
                try:
                    submod_tuples.append( ( abb, self.data[module]['overall']['duration'] ) )
                except:
                    submod_tuples.append( ( abb, 0 ) )  # when there is no end time...
                
        return submod_tuples
                
        
    def submodule_pareto( self , outdir , count=5 ):
        """ Creates pareto of submodules (or sequencing/postrun) for duration. """
        pareto = sorted( self.get_submodule_times(), key=lambda x: x[1], reverse=True )
        pareto = [tuple for tuple in pareto if tuple[1]>0 ] # remove submodules with zero duration
        print(pareto)
        if len(pareto) > count:
            pareto = pareto[:count]
            
        # Simple plot.
        labels, times = zip(*pareto)
        # Let's use cleanse_submodule function
        labels = [ self.cleanse_submodule( label ) for label in labels ] 

        x     = np.arange( len( pareto ) )
        width = 0.6
        
        plt.figure( )
        plt.bar   ( x, times, width=width , align='center' )
        plt.xticks( x , labels , fontsize=10 )
        plt.xlim  ( -0.5, count-0.5 )
        ylims = plt.ylim()
        plt.ylim  ( 0 , ylims[1] + 0.25 )
        #plt.xlabel( 'Module' )
        plt.ylabel( 'Duration (hours)' )
        
        for i, time in enumerate( times ):
            plt.text( i , time + 0.1 , get_hms( time ) , ha='center', va='center', fontsize=10 )
            
        plt.title       ( 'Submodule Duration Pareto' )
        plt.tight_layout( )
        plt.savefig     ( os.path.join( outdir, 'workflow_pareto.png' ) )
        plt.close       ( )
        
    @staticmethod
    def cleanse_submodule( name ):
        """ Helper function to abbreviate workflow names """
        # Define list of search, replace string tuples
        name = name.title()
        mods = [ ( 'Amplification', 'Amp.' ),
                 ( 'Ampliseq '    , ''     ),
                 ( ' - '          , ' '    ),
                 ( 'Universal'    , 'Univ.'),
                 ( 'Reverse Transcriptase', 'RT' ),
                 ( 'Contamination', 'Cont.' ),
                 ( 'Resuspension' , 'Resusp.' ),
                 ( 'Sequencing'   , 'Seq.' ),
                 ('Coca'          , 'COCA'),
                 ('Hd'            , ''  ), # remove HD from submodule name
                 ('Rna'           , 'RNA' ),
                 ('Rt'            , 'RT'  ),
                 ('Udg'           , 'UDG' ),
                 ( ' '            , '\n'  ),
                 ('Oia'           , 'OIA' ),
                 ( 'actor'        , ''    ),
                 ( 'Basecalling'  , 'Base\nCaller' ),
                 ( 'Barcodecrosstalk' , 'BC\nxTalk' ),
                 ( 'qcmetric'     , ' QC' ),
                 ( 'Postlibclean' , 'PostLib' ),
                 ( 'Postrunclean' , 'PostRun' ),
                 ( 'Postchipclean', 'PostChip' ),
                 ( 'Samples'      , 'Sample Analysis' ),
                 ( 'Sample-Level Plugins'      , 'Sample Plugins' ),
                 ]
        
        for (s, r) in mods:
            name = name.replace( s , r )
            
        return name
    
    
    def full_timing_plot( self , outdir , init_timing={} , analysis_timing={}, reseq=False ):
        """ Test plot showing all submodules under the main module plot... """
        fig = plt.figure( figsize=(12,4) )
        ax  = fig.add_subplot( 111 )
        upper  = True
        mods   = ['libprep','templating','sequencing','resequencing','postrun']
        colors = ['blue','green','darkcyan','darkmagenta','orange']
        height = 1.5
        
        if init_timing or analysis_timing or self.parallelPostLib:
            # We need to account for a second bar for init/analysis
            y     = 4.75
            par_y = 1.75
            
            if analysis_timing and self.parallelPostLib:
                # Prevent parallelPostLib and analysis blocks from overlapping. 
                at_y  = 1    # move analysis block much lower
                par_y = 2.75 # to have enough space, more parallel blocks closer to main block
            else:
                at_y = par_y # have init and analysis at the same height when no parallel postLibClean
            
            # Make room for legend if we need it for analysis modules.
            if not analysis_timing:
                ylims = [-1.5, 7]
            else:
                ylims = [-3, 7]
                
            mod_y = -1.25
        else:
            y     = 1.75
            ylims = [-0.25, 3.25]
            mod_y = 0
         
        # Update start of postRun module to be when seq ends with parallel postLibClean
        if self.parallelPostLib:
            if reseq:
                self.data['postrun']['overall']['start'] = self.data['resequencing']['overall']['end']
            else:
                self.data['postrun']['overall']['start'] = self.data['sequencing']['overall']['end']

        for module,color in zip( mods, colors ):
            print( '------' )
            print( module )
            if not self.data[module]['overall'].get('start',{}):
                continue
            # Shade background and add text
            o     = self.data[module]['overall']
            start = self.get_relative_time( o['start'] )
            try:
                end   = self.get_relative_time( o['end'] )
                mid   = start + (end-start)/2
            except: # necessary when there is module end time is None. Happens when plugin runs too early
                end = start + 200 # lets see what happens with this
                mid = start + 100
                print( 'Missing end for module {}'.format(module))
            ax.axvspan( start, end, color=color, alpha=0.4 )
            ax.text   ( mid , mod_y, '{}\n{}'.format( module.title(), get_hms( o['duration'] ) ), weight='bold',
                        ha='center', va='bottom', color=color, fontsize=10 )
            
            # Create submodule bars if they exist
            if 'submodules' in self.data[module]:
                sm    = self.data[module]['submodules']
                print(sm, module)
                #info  = [ (self.cleanse_submodule( k ), sm[k]['start'], sm[k]['duration'] ) for k in sm ]
                info = []
                for k in sm:
                    try:
                        info.append( (self.cleanse_submodule( k ), sm[k]['start'], sm[k]['duration']) )
                    except:
                        print('unable to process submodule {}'.format(k))
                try:
                    procs = sorted( info, key=lambda x: x[1] )
                except: 
                    print('Encountered an issue when trying to sort submodules by start time. Not sorting')
                    procs = info
                for proc in procs:
                    # Let's skip the module if the duration is 0.
                    if proc[2] == 0:
                        print( 'Skipping submodule {} since it has {} duration . . .'.format(sm, proc[2]) )
                        continue
                     
                    left = self.get_relative_time( proc[1] )
                    
                    # Move postrun submodules in parallel block with parallel PostLib. 
                    if self.parallelPostLib and module == 'postrun':
                        y_ = par_y
                    else:
                        y_ = y
                    
                    ax.barh( y_, proc[2], height=height, left=left, align='center',
                         color=color, edgecolor='black', zorder=3 )
                    
                    # If process is too close to start of module, lets bump it forward by 30 minutes
                    label_x_pos = left + proc[2]/2.
                    point_x_pos = left + proc[2]/2.
                    #if (end - label_x_pos ) < 0.5 and proc[2] <= 0.6:
                        #label_x_pos -= 0.25
                    #    ha = 'right'
                    #elif (label_x_pos < 1 or (label_x_pos - start) < 1) and (proc[2] <= 0.6):
                    #    label_x_pos += 0.25
                    #    ha = 'left'
                    if proc[2] <= 0.6:
                        label_x_pos += 0.25
                        ha = 'left'
                    else:
                        ha = 'center'
                    # If process is > 0.66 hour, put label inside bar.
                    if proc[2] > 0.66:
                        ax.text( label_x_pos, y_, proc[0], ha='center', va='center', fontsize=8,
                                  color='white', weight='normal' )
                    else:
                        arrows = dict(facecolor='black', width=0.75, headwidth=3, shrink=0.1, frac=0.1 ) # headlength used to be frac=0.1
                        if proc[0] in ['PostLib\nDeck\nClean','Postrun']:
                            upper = False  # for parallel deck clean, put to the side so it does not overlap with seq submodule 
                            if proc[0]=='Postrun':
                                ha = 'left'
                        elif proc[0] in ['PostLib\nVacuum\nClean']:
                            ha = 'left'
                        if upper:
                            if proc[0]=='Templating':
                                extra = 0.5
                            elif proc[0]=='Prerun\nChipcal':
                                extra = -0.2
                            else:
                                extra = 0
                            ax.annotate( proc[0], xy=(point_x_pos,y_+height/2.), xycoords='data',
                                         xytext=(label_x_pos, y_+height/2.+0.5+extra), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='bottom', fontsize=8 , weight='normal' )
                            #if self.parallelPostLib and module=='postrun':
                            #    upper = False  ### Gets in the way when there is parallel postLibClean 
                        else:
                            ax.annotate( proc[0], xy=(point_x_pos,y_), xycoords='data',
                                         xytext=(label_x_pos, y_+0.5), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='top', fontsize=8 , weight='normal' )
                            #ax.annotate( proc[0], xy=(point_x_pos,y_-height/2.), xycoords='data',
                            #             xytext=(label_x_pos, y_-height/2.-0.5), textcoords='data',
                            #             arrowprops=arrows,
                            #             ha=ha, va='top', fontsize=8 , weight='normal' )
                            upper = True
                        print(proc[0], label_x_pos, proc[2], ha)
                            
        seq_done = self.get_relative_time( self.data['sequencing']['overall']['end'] )
        
        # Run time defined as time to the end of the PostWhateverClean
        #runtime  = seq_done + self.data['postrun']['overall']['duration']
        try: # When no POSTRUN
            runtime  = self.get_relative_time( self.data['postrun']['overall']['end'] )
        except:
            runtime = None

        # Set timing and analysis boxes to grey?
        par_color = 'gray'
        if init_timing:
            # Plot init bar
            init_start = self.get_relative_time( init_timing['start'] )
            init_width = init_timing['duration']
            ax.barh( par_y, init_width, height=height, left=init_start, align='center',
                     color=par_color, edgecolor='black', zorder=3 , alpha=0.75 )
            ax.text( (init_start + init_width)/2., par_y, 'Init', ha='center', va='center', fontsize=8,
                     color='white', weight='normal' )
            
        if analysis_timing:
            # Plot analysis, again alternating labels above/below.
            upper = True
            info  = [ (self.cleanse_submodule( k ), analysis_timing[k]['start'], analysis_timing[k]['duration'] ) for k in analysis_timing ]
            print( 'submodules in analysis timing:')
            for i in info:
                print(i)
            
            print( 'removing submodules with no timestamp...')
            info_not_none = [e for e in info if e[1] is not None]
                
            procs = sorted( info_not_none, key=lambda x: x[1] )
            
            # Need to create value for analysis completion and sample analysis completion.
            final_step      = procs[len(procs)-1]
            analysis_done   = self.get_relative_time( final_step[1] ) + final_step[2]
            try:
                basecaller_done = self.get_relative_time( analysis_timing['BaseCallingActor']['end'] )
            except:
                print('BaseCallingActor end time not found... perhaps BaseCalling failed')
            samples_found = True
            try: 
                samples_done    = self.get_relative_time( analysis_timing['Samples']['end'] )
            except:
                samples_found = False
                print('Samples not found....')
            # Measure finish time, either postrun or analysis
            el_fin  = max( analysis_done, runtime )
            if analysis_done > runtime:
                ax.axvspan( runtime, analysis_done, color='gray', alpha=0.3 )
    
            # Setup other details, including a general "analysis" label
            patches = []
            colors  = [matplotlib.cm.Set1(i) for i in np.linspace(0,1,9)]
            ax.text( self.get_relative_time(procs[0][1])-0.1, at_y, 'Analysis', ha='right', va='center',
                     fontsize=10, color='black', weight='bold' )
            
            for i,proc in enumerate(procs):
                left = self.get_relative_time( proc[1] )
                patch = ax.barh( at_y, proc[2], height=1, left=left, align='center', color=colors[i],
                                 edgecolor='black', zorder=3 , alpha=0.75, label=proc[0].replace('\n',' ') )
                patches.append( patch )
                
                if False:
                    # If process is too close to the beginning or end of the run, lets bump it back by 30 minutes
                    label_x_pos = left + proc[2]/2.
                    point_x_pos = left + proc[2]/2.
                    ha = 'center'
                    if (label_x_pos < 1 or (label_x_pos - start) < 1) and (proc[2] <= 0.6):
                        label_x_pos -= 0.25
                        ha = 'right'
                    elif (el_fin - label_x_pos) < 0.5:
                        label_x_pos -= 0.25
                        ha = 'right'
                    else:
                        ha = 'center'
                        
                    # If process is > 30 minutes, put label insize bar.
                    if proc[2] > 0.5:
                        ax.text( label_x_pos, at_y, proc[0], ha='center', va='center', fontsize=8,
                                  color='white', weight='normal' )
                    else:
                        arrows = dict(facecolor='black', width=0.75, headwidth=3, shrink=0.1, frac=0.1 )
                        if upper:
                            ax.annotate( proc[0], xy=(point_x_pos,at_y+height/2.), xycoords='data',
                                         xytext=(label_x_pos, at_y+height/2.+0.5), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='bottom', fontsize=8 , weight='normal' )
                            upper = False
                        else:
                            ax.annotate( proc[0], xy=(point_x_pos,at_y-height/2.), xycoords='data',
                                         xytext=(label_x_pos, at_y-height/2.-0.5), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='top', fontsize=8 , weight='normal' )
                            upper = True
                else:
                    ax.legend( handles=patches, loc='lower center', fontsize=10, ncol=len(patches), frameon=True, facecolor='whitesmoke', framealpha=1 )
        else:
            el_fin = runtime
            
        # Set xlim as maximum process ending
        ax.set_xlim  ( 0, el_fin )
        ax.set_xticks( np.arange( 0, el_fin, 1 ) )
        
        ax.set_xlabel( 'Run Time (hours)' )
        ax.set_yticks( [] )
        ax.set_ylim( *ylims )
        try: # When no POSTRUN
            ttl = 'PostRun Complete: {}'.format( get_hms(runtime) )
        except:
            ttl = 'PostRun not found.' 
        if analysis_timing:
            try:
                ttl += '  |  BaseCaller Complete: {}'.format( get_hms(basecaller_done) )
            except:
                ttl += '  |  BaseCaller End not found.'
            if samples_found:
                ttl += '  |  Sample Analysis Complete: {}'.format( get_hms(samples_done) )
            else:
                ttl += '  |  Sample Analysis Missing.'
        ax.set_title( ttl )
        ax.grid  ( which='major', axis='x', ls=':', color='grey', zorder=0 )
        ax.grid  ( which='minor', axis='x', ls=':', color='grey', zorder=0, alpha=0.5 )
        fig.tight_layout( )
        fig.savefig( os.path.join( outdir, 'detailed_workflow_timing.png' ) )
        plt.close  ( )

class Timing( object ):
    """ Class for reading the timing.txt file that is output from the onboard_results for OIA. """
    timing_regex = re.compile( """TIMING\s(?P<run>[\w_\-\.]+)\s(?P<block>[\w_]+)\s(?P<chunk>[\w\-]+)\s(?P<duration>[\w\-]+)\s(?P<threadid>[\w]+)\s"(?P<start>[0-9\-\s:]+)"\s"(?P<stop>[0-9\-\s:]+)"\s(?P<returncode>[0-9]+)""" )
    
    def __init__( self, filepath ):
        self.filepath = filepath
        self.data = []
        self.block_timing = {}
        self.read( )
        self.analyze( )
        
    def read( self ):
        with open( self.filepath, 'r' ) as f:
            for line in f.readlines():
                parsed = self.parse_line( line )
                if parsed:
                    self.data.append( parsed )
                    
    def analyze( self ):
        """ Runs through data to pile up information and do high level analysis. """
        overall_start = None
        overall_stop   = None
        for d in self.data:
            b = d['block']
            if b not in self.block_timing:
                self.block_timing[b] = { 'start': d['start'] , 'stop': d['stop'] }
            else:
                self.block_timing[b]['start'] = min( self.block_timing[b]['start'], d['start'])
                self.block_timing[b]['stop']  = max( self.block_timing[b]['stop'] , d['stop'] )
                
            if overall_start:
                overall_start = min( overall_start, d['start'] )
            else:
                overall_start = d['start']
                
            if overall_stop:
                overall_stop  = max( overall_stop,  d['stop'] )
            else:
                overall_stop  = d['stop']
                
        self.overall = { 'start': overall_start, 'stop': overall_stop, }
        self.overall['duration'] = self.get_duration( self.overall )
        
        for block in self.block_timing:
            self.block_timing[block]['duration'] = self.get_duration( self.block_timing[block] )
        
    def parse_line( self, line ):
        """ Reads the line and returns a dictionary if it's a match. """
        m = self.timing_regex.match( line )
        x = {}
        if m:
            x = m.groupdict()
            
            # decode timestamps and calc duration
            x['start']    = self.get_timestamp( x['start'] )
            x['stop']     = self.get_timestamp( x['stop'] )
            x['duration'] = self.get_duration ( x )
            
        return x
    
    def make_detailed_plot( self , outdir ):
        """ Creates a detailed plot showing analysis time on a per-block basis. """
        sk = sorted( self.block_timing.keys(),
                     key=lambda x: ( int(x.split('_')[0][1:]),int(x.split('_')[1][1:]) )[::-1])
        
        plt.figure( figsize=(10,10) )
        for d in self.data:
            _ = plt.barh( sk.index(d['block']), d['duration'], height=0.8,
                          left=(d['start']-self.overall['start']).total_seconds()/3600., align='center' )
            
        n_blocks = len( sk )
        plt.ylim  ( -0.5, n_blocks - 0.5 )
        plt.yticks( range( n_blocks ), sk, fontsize=6 )
        plt.xlabel( 'Relative time (hr)' )
        plt.title ( 'OIA Timing | Total Duration: {:.2f} Hours'.format( self.overall['duration'] ) )
        plt.savefig( os.path.join( outdir, 'oia_timing.png' ) )
        plt.close()
        
    @staticmethod
    def get_duration( line_dict ):
        """ returns duration in hours """
        try:
            return float( (line_dict['stop'] - line_dict['start']).total_seconds() / 3600. )
        except( TypeError, KeyError ):
            return 0.
    
    @staticmethod
    def get_timestamp( dts ):
        """ reads in timestamp from timing file and converts to datetime object. """
        return datetime.datetime.strptime( dts, "%Y-%m-%d %H:%M:%S" )
    
    
if __name__ == "__main__":
    PluginCLI()
