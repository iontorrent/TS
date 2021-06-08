#!/usr/bin/env python
# Copyright (C) 2019 Ion Torrent Systems, Inc. All Rights Reserved

from ion.plugin import *

import sys, os, datetime, time, json, re, csv, math, glob, textwrap
import numpy as np

import matplotlib
matplotlib.use( 'agg' ) # to make sure this is set for before another module doesn't set it
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set default cmap to jet, which is viridis on newer TS software (matplotlib 2.1)
matplotlib.rcParams['image.cmap'] = 'jet'

# Load image tools to resize deck images, if requested.
from PIL import Image

# Import our tools
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.join(parentdir, 'autoCal')) 
print os.path.join(parentdir, 'autoCal')

from tools.PluginMixin import PluginMixin
from tools.debug_reader import ValkyrieDebug
get_hms = ValkyrieDebug.get_hms
from tools import html as H
from tools import sparklines

LOG_REGEX = re.compile( '''summary.(?P<timestamp>[\w\-_:]+).log''' )

class ValkyrieWorkflow( IonPlugin , PluginMixin ):
    """ 
    Phil Waggoner
    
    Plugin to analyze timing of Valkyrie workflow for run tracking and performance monitoring.
    
    Latest Update | CN: Added maxAmp_foamScrape and normal_maxAmp metrics.
                  | CN: Added warning message for when plugin is launched before post run cleans complete.
                  | PW: Fixed a bug in regex for git branch/commit info.
                  | CN: Move check for instrument platform before analyze_flow_rate to avoid crashing on proton runs.
                  | PW: Fixed bug in regex for OIA_TIMING that did not expect a period in a run name.
                  | CN: Bugfix to ensure plugin works when there is no postrun.
                  | PW: Bugfix to ensure plugin works on RUO Pipeline.
                  | PW: Added initialization and pipeline/analysis timing.
                  | BP: Added pipette timing metrics
                  | BP: minor uprev of version to better control relaunch and get plugins up to speed.
                  | 0.9.11  | BP: fixed formatting bugs for date and floats
                  | 0.9.12  | BP: uprevved for version control
                  | 0.9.13  | BP: improved foam scrape detection, and block parsing
                  | 0.9.14  | BP: uprevved for version control
    """
    version       = "0.9.14"
    allow_autorun = True
    
    runTypes      = [ RunType.THUMB , RunType.FULLCHIP , RunType.COMPOSITE ]
    
    def launch( self ):
        print( "Start Plugin" )
        self.init_plugin( )
        
        self.metrics['software_version'] = str(self.explog.metrics['ReleaseVersion'])
        self.metrics['dev_scripts_used'] = self.explog.metrics['Valkyrie_Dev_Scripts']
        self.metrics['datacollect']      = str(self.explog.metrics['DatacollectVersion'])
        self.metrics['scripts_version']  = str(self.explog.metrics['ScriptsVersion'])
        
        # Set up debug_log path
        self.debug_log = os.path.join( self.calibration_dir , 'debug' )
        
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
            
        # Exit plugin if we're missing required data
        if self.is_missing_data( ):
            sys.exit(0)
        
        self.prepare_chip_images( )
        self.prepare_deck_images( scale_factor=2 )
        self.analyze_flow_rate  ( )
        
        # Exit plugin if we're missing required data
        #if self.is_missing_data( ):
        #    sys.exit(0)
            
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
        self.analysis_timing = self.analyze_pipeline( )
        print( self.analysis_timing )
        
        self.sample_timing   = self.analyze_samples ( )
        print( self.sample_timing )
        
        if self.sample_timing:
            self.analysis_timing['Samples'] = self.sample_timing
        
        # If we found OIA data, let's analyze and add to the analysis_timing dictionary.
        oia_timing = os.path.join( self.raw_data_dir, 'onboard_results', 'sigproc_results', 'timing.txt' )
        if os.path.exists( oia_timing ):
            print( 'Found OIA timing.txt.  Now analyzing.' )
            self.oia_timing = Timing( oia_timing )
            self.oia_timing.make_detailed_plot( self.results_dir )
            self.analysis_timing['OIA'] = self.oia_timing.overall
            print( 'OIA timing:' )
            print( self.oia_timing.overall )
            
        ###########################################################################
        
        # Analyze Workflow Timing -- Now incorporating ScriptStatus.csv.
        self.analyze_workflow_timing( )
        
        # Analyze Tube Bottom locations
        self.analyze_tube_bottoms( )
        
        # Look for blocked tips
        self.count_blocked_tips( )

        # Analyze vacuum log files
        self.analyze_vacuum_logs( )

        # Analyze pipette behavior
        self.analyze_pipette_behavior( )

        # Check if plugin completed before postrun cleans were complete
        self.did_plugin_run_too_early( )

        # Create graphic of which modules were used....Or just make an html table.
        self.write_block_html( )
        
        self.write_metrics( )
        print( "Plugin Complete." )
        
    def analyze_pipeline( self ):
        """ Finds the appropriate summary.log file and reads in analysis modules and timing. """
        # The summary.__.log files live in the analysis dir.
        log = self.get_first_log( self.analysis_dir )
        if log:
            # If we're here, it's time to read in the information.
            summary_log = SummaryLog( os.path.join( self.analysis_dir, log ) )
            return summary_log.timing
        else:
            return {}
    
    def analyze_samples( self ):
        """ Processes through sample log files and determines sample analysis timing details. """
        sample_logs = []
        # Parse log files
        for sample_name in self.sample_dirs:
            sample_dir = os.path.join( self.analysis_dir, self.sample_dirs[sample_name] )
            log = self.get_first_log( sample_dir )
            if log:
                sample_logs.append( SummaryLog( os.path.join( sample_dir, log ), sample_name ) )
                
        # Exit if we found no samples
        if not sample_logs:
            print( 'Error!  No samples found.  This is embarassing . . .' )
            return {}
        
        # Sort the logs by earliest analysis start time
        sample_logs = sorted( sample_logs, key=lambda x: x.get_start_time() )
        timing = { 'start' : sample_logs[0].get_start_time(),
                   'end'   : sorted( [s.get_end_time() for s in sample_logs] )[-1] }
        
        # Measure duration
        timing['duration'] = sample_logs[0].get_duration( timing )
        
        # Plot the data separately
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
        
        # This is where the magic happens
        plt.figure ( figsize=(12,4) )
        plt.subplot( 121 )
        yticktups = []
        y         = 0
        space     = 2
        for sample in sample_logs:
            last_y, labels, patches = plot_sample( sample, y )
            yticktups.append( ((last_y-y)/2. + y,sample.name) )
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
        return timing
    
    def get_summary_logs( self, log_dir ):
        """ Helper method to get summary log files.  Typically we want the first one created. """
        logs      = [l for l in os.listdir( log_dir ) if LOG_REGEX.match( l ) ]
        return sorted( logs )
    
    def get_first_log( self, log_dir ):
        """ Wrapper on get_summary_logs to give the first log, usually what we want. """
        logs = self.get_summary_logs( log_dir )
        try:
            log = logs[0]
        except IndexError:
            log = ''
        return log
        
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
        
        true_active = sum( [ is_active( exp['LanesActive{}'.format(i)] ) for i in [1,2,3,4] ] )
        print( 'Detected the true number of active lanes: {:d} . . .'.format( true_active ) )
        
        target_flow = 48. * true_active
        flow_range  = target_flow * np.array( (1.0-flow_rate_margin, 1.0+flow_rate_margin ), float )
        
        # Let's get only the data we want from the text explog with flow data
        x        = self.explog.flowax[   self.explog.flowax > 0 ]
        data     = self.explog.flowrate[ self.explog.flowax > 0 ]
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
            
        print( 'Average flow rate: {:.1f} uL/s'.format( data.mean() ) )
        flow_rate_metrics = { 'mean'         : data.mean() ,
                              'std'          : data.std() ,
                              'outliers'     : outliers.sum() ,
                              'perc_outliers': 100. * float(outliers.sum()) / float(self.explog.flows) ,
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
        spark.ax.axhspan( flow_range[0] , flow_range[1] , color='green' , alpha=0.4 )
        spark.ax.axhline( target_flow , ls='--', color='green', linewidth=0.5 )
        spark.draw      ( )
        
        fig.savefig( os.path.join( self.results_dir , 'flow_spark.svg' ), format='svg' )
        
    def analyze_workflow_timing( self ):
        """ 
        Does original large scale analysis.
        Now also does analysis based on ScriptStatus.csv, if the file is available.
        """
        self.debug.parallel_grep  ( )

        # Get timing and adjust if needed.
        self.debug.get_overall_timing()
        
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
        if all( mods.values() ):
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
        elif mods['sequencing'] and not all([ mods['libprep'], mods['harpoon'], mods['magloading'], mods['coca'] ]):
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
            
        # Add in postrun timing before we update_data
        self.ss = ScriptStatus( ss_file, self.explog.start_time )
        self.ss.read()
        self.ss.add_postrun_modules( self.postchip_clean_timing, self.postrun_clean_timing )
        self.ss.update_data( )
        self.ss.add_overall_timing( self.debug.timing )
        
        # Make figures
        self.ss.submodule_pareto( self.results_dir, count=5 )
        self.ss.full_timing_plot( self.results_dir , self.init_timing , self.analysis_timing )
        
        # Save metrics for durations, used tips.  Some of these are copied to the overall results.json
        seq_done   = self.ss.get_relative_time( self.ss.data['sequencing']['overall']['end'] )
        ss_metrics = { 'used_tips': 0 }
        
        # The other metrics required for the program are time to basecaller completion and sample analysis compl.
        self.metrics['time_to_seq_done']     = seq_done
        try:
            self.metrics['time_to_postrun_done'] = self.ss.get_relative_time(self.ss.data['postrun']['overall']['end'])
        except: # no POSTRUN
            self.metrics['time_to_postrun_done'] = None
        if self.analysis_timing:
            self.metrics['time_to_basecaller_done'] = self.ss.get_relative_time( self.analysis_timing['BaseCallingActor']['end'] )
            self.metrics['time_to_samples_done'] = self.ss.get_relative_time( self.analysis_timing['Samples']['end'] )
        
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
                
    def analyze_tube_bottoms( self ):
        """ 
        Analyzes the debug csv that checks tubes for their bottom location (only occurs when running BottomFind) 
        """
        self.metrics['bottomlog'] = {}
        missed    = []
        bent_tips = []
        
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
                        
                        # Change to not count all levels of warnings, but only the most extreme.
                        if 'Bent tip' in row['bent_tip']:
                            bent_tips.append( row['tube'] )
                            
                        # Change to not count all levels of warnings, but only the most extreme.
                        if 'Not reaching bottom' in row['missed_bottom']:
                            missed.append( row['tube'] )
                    except( ValueError, TypeError ):
                        # Don't care, must not have been a row with a real zcal value and thus other useful info.
                        pass
                    
            # Summarize metrics
            self.metrics['bottomlog'] = { 'missed_bottom'       : ', '.join( missed ),
                                          'missed_bottom_count' : len( missed ),
                                          'bent_tips'           : ', '.join( bent_tips ),
                                          'bent_tips_count'     : len( bent_tips )
                                          }
            
            # Make a symlink to the raw file for easy access
            os.symlink( tbl , os.path.join( self.results_dir, 'TubeBottomLog.csv' ) )
        else:
            self.has_tube_bottom_log = False
            print( 'Could not find the TubeBottomLog.csv file.  Skipping analysis.' )

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
        # initialize metric storage
        pipette_behavior = {}
        
        # get harpooon behavior
        pipette_behavior['harpoon_mixing'] = self.analyze_pipette_behavior_HARPOON()
        # get magloading
        pipette_behavior['magloading'] = self.analyze_pipette_behavior_MAGLOADING()
        
        # store metrics
        self.metrics['pipette_behavior'] = pipette_behavior

        print( '-------COMPLETE: Analyzing Pipette Behavior-------' )

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
    
    
    def analyze_pipette_behavior_MAGLOADING( self ):
        # Check if harpoon module was completed
        # If not --> stop further analysis

        print( 'Starting -- Magloading analysis')
        if not self.debug.modules['magloading']: return
        
        # Get the relevant lines from the debug log
        # --> For the harpoon module, it will be two blocks between 'mix well to denature' and 'ditch tip'
        process_start   = r'.*#Process:---Foam generation and foam.*'
        process_stop    = r'.*#Process:---60/40 AB/IPA.*'
        relevant_lines  = [ r'Mixing: Pipette [0-9] Aspirate',
                            r'Mix: totalTime',
                            ]
        
        # Get blocks of lines for further parsing
        blocks = self.debug.search_blocks( process_start, process_stop, *relevant_lines, case_sensitive=True )
        print( 'blocks', blocks )
        print( 'Complete -- Magloading analysis')

        return self.parse_pipette_mixing_time_blocks( blocks )
    
    # NOTE: This seems to handle most of what we want from pipette timing
    def parse_pipette_mixing_time_blocks( self, blocks ):
        ''' 
        Takes in blocks for pipette mixing times and outputs a dictionary
        
        NOTE:  Requires the following grep strings in the generation of the blocks
        
        r'Mixing: Pipette [0-9] Aspirate' 
        r'Mix: totalTime'
        '''
        # parse each block for pipette timing information
        regex_mixing = re.compile( r'.*: Mixing: Pipette (?P<pipette>[0-9]).*' )
        regex_timing = re.compile( r'.*: Mix: totalTime = (?P<totalTime>[0-9]+[.][0-9]+) estTime = (?P<estTime>[0-9]+[.][0-9]+), elapsedTime = (?P<elapsedTime>[0-9]+[.][0-9]+).*' )
        
        block_dict = {}
        for i,b in enumerate( blocks ):
            block_name = 'block{}'.format(i)
            block_dict[block_name]={}
            use = {'1':0,'2':0}
            for j, line in enumerate(b):
                data = self.debug.read_line( line )
                if data and 'Mixing' in data['message']:
                    # get the pipette number and iterate use
                    mix_match   = regex_mixing.match( data['message'] )
                    mix_dict    = {}
                    if mix_match: mix_dict = mix_match.groupdict()
                    # try to get pipette number and make pipette_name with use appened
                    try: pnum = str(mix_dict['pipette'])
                    except KeyError: continue
                    use[pnum]+=1
                    pipette_name = 'p{}_{}'.format( pnum, use[pnum] )
                    
                    # get the timestamp from the data line
                    timestamp = data['timestamp']
                    # get the timing values
                    try:
                        time_data = self.debug.read_line( b[j+1] )
                    except IndexError:
                        # Missing data --> try to continue
                        continue
                    time_match = regex_timing.match( time_data['message'] )
                    time_dict = {}
                    if time_match:
                        for key, item in time_match.groupdict().items():
                            time_dict[key] = float( item )
                    # store values
                    time_dict['timestamp'] = timestamp.strftime( '%m/%d/%Y %H:%M:%S' )
                    block_dict[block_name][pipette_name] = time_dict
        return block_dict
    
    #############################################
    #   END -- HELPERS FOR PIPETTE BEHAVIOR     #
    #############################################

    def count_blocked_tips( self ):
        """ Reads the debug log to identify how many tips were blocked during piercing and count them up. """
        # Initialize variables
        self.metrics['blocked_tips'] = {}

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
                                         'p1_tubes'  : pipette_1 ,
                                         'p2_tubes'  : pipette_2 ,
                                         'used_blocked_tips'      : warnings ,
                                         'used_blocked_tips_count': len( warnings )
                                         }
        return None

    def analyze_vacuum_logs( self ):
        ''' Reads the vacuum log csv files'''
        # metrics that will be written to json file and eventually saved to database
        self.metrics['vacLog'] = { 'lane 1':{}, 'lane 2':{}, 'lane 3':{}, 'lane 4':{}, 'lane 5':{}, 'lane 0':{}  } 
        
        lanes = ['lane 1', 'lane 2', 'lane 3', 'lane 4', 'lane 5', 'lane 0']

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
                process_list, metrics_dict = self.detect_leaks_clogs( process_list )
                self.metrics['vacLog'][lane] = metrics_dict
                print('Total of {} processes for {}'.format(len(process_list) ,lane))
                self.save_vacuum_plots( process_list, metrics_dict )
            else:
                print('{} vacuum log not found.'.format(lane))
                self.metrics['vacLog'][lane]['log_found']=False
                self.metrics['vacLog'][lane]['postLib_found']=False 
        
        # If there are no vacuum logs in any lane, set self.has_vacuum_logs False
        if not any( [self.metrics['vacLog'][lane]['log_found'] for lane in lanes ] ) :
            self.has_vacuum_logs = False
                
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
            process_dict['Pressure Reached Target Time'] = pressure_reached_target_time
        except:
            pass
        try:
            process_dict['Duty Cycle at End'] = float(np.sum(pumping_to_maintain_target_temp[t_after_target_index:])) /float( len(pumping_to_maintain_target_temp[t_after_target_index:]))*100 
            process_dict['Amplitude of Oscillation'] = np.std(pressure_temp[t_after_target_index:])* math.sqrt(3)
            process_dict['Max Amp of Oscillation'] = (max(pressure_temp[pump_reached_target_index:t_after_target_index])-min(pressure_temp[pump_reached_target_index:t_after_target_index])) / 2.0
        except:
            process_dict['Duty Cycle at End'] = 100 
            process_dict['Amplitude of Oscillation'] = None
            process_dict['Max Amp of Oscillation'] = None
            
        if process_dict['Target Pressure'] <= process_dict['Initial Pressure']: # normal VacuumDry    
            if pressure_reached_target:
                process_dict['Time to Target'] = pump_reached_target_time - pump_start_time
                process_dict['Vent Open Time'] = pump_start_time
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
                process_dict['Time to Initial'] = time_temp[-1] - pump_start_time        

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
        
    def detect_leaks_clogs( self, process_list ):
        suspected_leaks = []
        suspected_clogs = []
        postLib_leaks = []
        postLib_clogs = []
        non_SDS_DC = [] # do not include SDS. Do not include postRun for Lanes1,2,3,4
        non_SDS_amp = [] # do not include SDS. Do not include postRun for Lanes1,2,3,4
        non_SDS_maxAmp = [] # do not include SDS. Do not include postRun for Lanes1,2,3,4
        non_SDS_time_to_target = []
        log_found = False   # workflow vacuum logs- set to true once found
        postLib_found = False
        maxAmp_foamScrape = None
        for process_dict in process_list:
            # Want to save the max amplitude of the VacuumDry following foam scrape since this usually has a pressure spike. 
            if 'Foam Scrape' in process_dict['Process']:
                try:
                    maxAmp_foamScrape = process_dict['Max Amp of Oscillation']
                except:
                    maxAmp_foamScrape = None
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
        
        metrics_dict = {'abnormal_process_count': len(suspected_leaks)+len(suspected_clogs),
                        'suspected_leaks_count' : len(suspected_leaks),
                        'suspected_clogs_count': len(suspected_clogs),
                        'suspected_leaks' : suspected_leaks,
                        'suspected_clogs': suspected_clogs,
                        'postLib_abnormal_process_count': len(postLib_leaks)+len(postLib_clogs),
                        'postLib_suspected_leaks_count' : len(postLib_leaks),
                        'postLib_suspected_clogs_count': len(postLib_clogs),
                        'postLib_leaks' : postLib_leaks,
                        'postLib_clogs': postLib_clogs,
                        'normal_DC': np.mean(non_SDS_DC),
                        'normal_amp': np.mean(non_SDS_amp),
                        'normal_maxAmp': np.mean(non_SDS_maxAmp),
                        'normal_time_to_target': np.mean(non_SDS_time_to_target),
                        'log_found': log_found, 
                        'postLib_found': postLib_found ,
                        'maxAmp_foamScrape': maxAmp_foamScrape }

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
    
    def is_missing_data( self ):
        """ Checks for required data and returns a true if we are missing data, meaning we should immediately exit the plugin. """
        answer = False
        
        # Let's make sure this is a Valkyrie run.
        if self.explog.metrics['Platform'].lower() == 'valkyrie':
            print( 'Confirmed that run is on a Valkyrie!  Proceeding with plugin.' )
        else:
            print( 'Exiting plugin as this run is not on a Valkyrie!' )
            answer = True
            
            # Write trivial output
            doc = H.Document()
            msg = H.Header( 'This run does not appear to be on a Valkryie.  Exiting plugin.', 3 )
            doc.add( msg )
            with open( os.path.join( self.results_dir, 'ValkyrieWorkflow_block.html' ), 'w' ) as f:
                f.write( str( doc ) )
                
        # Now let's make sure we have access to the debug file.
        if os.path.exists( self.debug_log ):
            print( 'Successfully located debug log file.' )
        else:
            print( 'Exiting plugin since the debug log file was not found!' )
            answer = True
            
            # Write trivial output
            doc = H.Document()
            msg = H.Header( 'Unable to find debug log file.  Exiting plugin.', 3 )
            doc.add( msg )
            with open( os.path.join( self.results_dir, 'ValkyrieWorkflow_block.html' ), 'w' ) as f:
                f.write( str( doc ) )
                
        return answer
    
    def did_plugin_run_too_early( self ):
        """
        Plugins are launch once analysis is complete. However, postLibClean and PostRun or PostChipClean may run after analysis is complete.
        This function checks whether or not this is the case, and generates a message to add to the html notifying the user to re-launch.
        """
        # First, read the explog to see if user has selected PostLibClean, PostChipClean, or PostRunClean.
        doPostLib  = self.explog.metrics['doPostLibClean']
        doPostRun  = self.explog.metrics['postRunClean']
        doPostChip = self.explog.metrics['doPostChipClean']

        # Print out what was selected for postLib, postRun, and postChip cleans
        print('PostLib: {}'.format(doPostLib))
        print('PostRun:{}'.format(doPostRun))
        print('PostChip: {}'.format(doPostChip))
        
        # Next, generate a warning message for when the plugin completed before post run cleans were complete.
        if (doPostChip or doPostRun) and not self.metrics['time_to_postrun_done']:
            self.message = 'WARNING: Plugin was launched before postrun cleans were complete. Relaunch plugin to see complete timing analysis (applies to TS only)'
        # Change message slightly if postLibClean was missing.  
        elif doPostLib and not self.metrics['vacLog']['lane 5']['postLib_found']:
            print(doPostLib, self.metrics['vacLog']['lane 5']['postLib_found'])
            self.message = 'WARNING: Plugin was launched before postrun cleans were complete. Relaunch plugin to see complete timing analysis and postLibClean vacuum log data (applies to TS only).'
        else:
            self.message = ''
    
    def write_block_html( self ):
        """ Creates a simple block html file for minimal viewing clutter. """
        doc = H.Document()
        doc.add( '<!DOCTYPE html>' )
        
        styles = H.Style()
        styles.add_css( 'td.active'  , { 'background-color': '#0f0' } )
        styles.add_css( 'td.inactive', { 'background-color': '#000' } )
        styles.add_css( 'td.error'   , { 'background-color': '#FF4500',
                                         'font-weight': 'bold',
                                         'color': '#fff' } )
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
                                                             'color': '#FF4500'} )
        
        styles.add_css( 'span.tooltip:hover + div.tooltiptext' , { 'visibility': 'visible' } )
        
        doc.add( styles )
        
        table   = H.Table( border='0' )
        space   = H.TableCell( '&nbsp', width="5px" )
        
        for name,module in [('LibPrep','libprep'), ('Harpoon','harpoon'), ('MagLoading','magloading'),
                            ('COCA'   ,'coca'   ), ('Seq.','sequencing') ]:
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
            """ % ( str(p1_num), ', '.join( self.metrics['blocked_tips']['p1_tubes'] ) ) )
            alert = H.TableCell( code , width="20px" , align='center' )
            alert.attrs['class'] = 'error'
        #fails  = H.TableCell( self.metrics['blocked_tips']['p1_tubes'] , width='600px' )
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
            """ % ( str(p2_num), ', '.join( self.metrics['blocked_tips']['p2_tubes'] ) ) )
            alert = H.TableCell( code , width="20px" , align='center' )
            alert.attrs['class'] = 'error'
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
            """ % ( str(bt_num), ', '.join( self.metrics['blocked_tips']['used_blocked_tips'] ) ) )
            alert = H.TableCell( code , width="20px" , align='center' )
            alert.attrs['class'] = 'error'
        for cell in [label, space, alert]:
            row3.add_cell( cell )
            
        blockage_table.add_row( row0 )
        blockage_table.add_row( row1 )
        blockage_table.add_row( row2 )
        blockage_table.add_row( row3 )
        
        alarmrow.add_cell( H.TC( blockage_table , width='20%' ) )
        
        # Need to add info about the TubeBottomLog.csv in the block html, including link to csv
        if self.has_tube_bottom_log:
            # Table
            # row per failure type, columns are label, colored box (green if ok, red if not), list of failed steps
            tip_table = H.Table( border='0' )
            space     = H.TableCell( '&nbsp', width="5px" )
            
            row0   = H.TableRow( )
            row0.add_cell( H.TC( 'Tube Bottom Log', True , align='right' , width="125px") )
            row0.add_cell( space )
            row0.add_cell( H.TableCell( '&nbsp', width="20px" ) )
            
            # Missed Bottom - changed nomenclature based on feedback from Shawn.
            row1   = H.TableRow( )
            label  = H.TableCell( '> 2 mm <em>above</em> zcal', align='right' , width="125px")
            mb_num = self.metrics['bottomlog']['missed_bottom_count']
            if mb_num == 0:
                alert = H.TableCell( '', width="20px" )
                alert.attrs['class'] = 'active'
            else:
                code  = textwrap.dedent("""\
                <span class="tooltip">%s</span>
                <div class="tooltiptext">%s</div>
                """ % ( str(mb_num), self.metrics['bottomlog']['missed_bottom'] ) )
                alert = H.TableCell( code , width="20px" , align='center' )
                alert.attrs['class'] = 'error'
            #fails  = H.TableCell( self.metrics['bottomlog']['missed_bottom'] , width='600px' )
            for cell in [label, space, alert]:
                row1.add_cell( cell )
                
            # Bent Tube - changed nomenclature based on feedback from Shawn.
            row2   = H.TableRow( )
            label  = H.TableCell( '> 2 mm <em>below</em> zcal', align='right' , width="125px")
            space  = H.TableCell( '&nbsp', width="5px" )
            bt_num = self.metrics['bottomlog']['bent_tips_count']
            if bt_num == 0:
                alert = H.TableCell( '', width="20px" )
                alert.attrs['class'] = 'active'
            else:
                code  = textwrap.dedent("""\
                <span class="tooltip">%s</span>
                <div class="tooltiptext">%s</div>
                """ % ( str(bt_num), self.metrics['bottomlog']['bent_tips'] ) )
                alert = H.TableCell( code , width="20px" , align='center' )
                alert.attrs['class'] = 'error'
            #fails  = H.TableCell( self.metrics['bottomlog']['bent_tips'] , width='600px' )
            for cell in [label, space, alert]:
                row2.add_cell( cell )
                
            tip_table.add_row( row0 )
            tip_table.add_row( row1 )
            tip_table.add_row( row2 )
            
            alarmrow.add_cell( H.TC( tip_table, width='20%' ) )
        else:
            alarmrow.add_cell( H.TC( '&nbsp', width='20%' ) )
            
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
            
            # Workflow
            for lane in [('lane 1', 'lane 1'),( 'lane 2','lane 2'),( 'lane 3','lane 3'),( 'lane 4','lane 4'), ('lane 5','robot waste'), ('lane 0','bleed valve')]:
                Vrow1   = H.TableRow( )
                label   = H.TableCell( lane[1], align='right' , width="100px")
                # workflow alerts, only vacuum logs from lanes are in workflow
                if self.metrics['vacLog'][lane[0]]['log_found']:
                    abnormal_num = self.metrics['vacLog'][lane[0]]['abnormal_process_count']
                    if abnormal_num == 0:
                        wf_alert = H.TableCell( '', width="20px" )
                        wf_alert.attrs['class'] = 'active'
                    else:
                        abnormal_processes = ', '.join(self.metrics['vacLog'][lane[0]]['suspected_leaks']) + ', '.join(self.metrics['vacLog'][lane[0]]['suspected_clogs'])
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
                if self.metrics['vacLog'][lane[0]]['postLib_found'] :# having issues here
                    abnormal_num = self.metrics['vacLog'][lane[0]]['postLib_abnormal_process_count']
                    if abnormal_num == 0:
                        postLib_alert = H.TableCell( '', width="20px" )
                        postLib_alert.attrs['class'] = 'active'
                    else:
                        abnormal_processes = ', '.join(self.metrics['vacLog'][lane[0]]['postLib_leaks']) + ', '.join(self.metrics['vacLog'][lane[0]]['postLib_clogs'])
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
                
            alarmrow.add_cell( H.TC( vacLog_table, width='20%' ) )
        else:
            alarmrow.add_cell( H.TC( '&nbsp', width='20%' ) )
            
        flow_img = H.Image( os.path.join( 'flow_spark.svg' ), width='100%' )
        fail4    = H.TC( flow_img.as_link() )
        alarmrow.add_cell( fail4 )
        alarms.add_row( alarmrow )

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
        return sorted( [self.timing[m]['start'] for m in self.timing] )[0]
    
    def get_end_time( self ):
        """ Returns the end time of the last module. """
        return sorted( [self.timing[m]['end'] for m in self.timing] )[-1]
    
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
        self.set_fieldnames( )
        
        # Initialize timing.  Will be a dictionary of dictionaries.
        # - Each will have 'overall' key but will then have a dictionary of submodules.
        # - Each module will have keys: start, end, tips_start, tips_end, duration, used_tips
        # Note that we will want to add reagent prime back in sometime.
        self.data = { 'libprep' :   { 'overall': {} , 'submodules': {} },
                      'templating': { 'overall': {} , 'submodules': {} },
                      'analysis':   { 'overall': {} , 'submodules': {} },
                      'sequencing': { 'overall': {} , 'submodules': {} },
                      'postrun':    { 'overall': {} , 'submodules': {} },
        }
        
        
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

        if module in ['postrun']:
            # The postrun info in ScriptStatus is the PostLibClean
            submodule = 'postlibclean'
            
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
                self.data[module]['overall']['start'] = timestamp
            
        if end:
            sm['end']        = timestamp
            sm['tips_end']   = int( line['used_tips'].strip() )
            if module in ['postrun']:
                self.data[module]['overall']['end'] = timestamp
            
        return None
    
        
    def add_overall_timing( self , timing_dict ):
        """ Adds ValkyrieDebug.timing to the self.data for the 'true' module start/end times. """
        for k in timing_dict:
            if '_' in k:
                m, ts = k.split('_')
                start = ts=='start'
                if m == 'library':
                    module = 'libprep'
                else:
                    module = m
                    
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
                        try:
                            seq_start = timing_dict['templating_end']
                            self.data[module]['overall']['start'] = seq_start
                            self.data[module]['submodules']['prerun chipcal']['start'] = seq_start
                        except KeyError:
                            # Sequencing only run
                            self.data[module]['overall']['start'] = self.start_time
                            self.data[module]['submodules']['prerun chipcal']['start'] = self.start_time
                            
                        # Update submodules
                        self.data[module]['submodules']['sequencing']['start']   = timing_dict[k]
                        self.data[module]['submodules']['prerun chipcal']['end'] = timing_dict[k]
                    else:
                        self.data[module]['overall']['start'] = timing_dict[k]
                else:
                    if m == 'sequencing':
                        self.data[module]['overall']['end'] = timing_dict[k]
                        self.data[module]['submodules']['sequencing']['end'] = timing_dict[k]
                    else:
                        self.data[module]['overall']['end'] = timing_dict[k]
                        
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
        try:
            rel_time = (timestamp - self.start_time).total_seconds() / 3600.
        except:  
            timestamp = datetime.datetime.strptime(timestamp, '%m/%d/%Y %H:%M:%S') 
            rel_time = (timestamp - self.start_time).total_seconds() / 3600.

        return rel_time 
    
    
    def get_submodule_times( self ):
        """ Helper function to extract submodule timing. """
        abbrevs = { 'libprep': 'Lib',
                    'templating': 'Temp',
                    'sequencing': 'Seq',
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
                        
                    submod_tuples.append( ( label, sm_data['duration'] ) )
            else:
                submod_tuples.append( ( abb, self.data[module]['overall']['duration'] ) )
                
        return submod_tuples
                
        
    def submodule_pareto( self , outdir , count=5 ):
        """ Creates pareto of submodules (or sequencing/postrun) for duration. """
        pareto = sorted( self.get_submodule_times(), key=lambda x: x[1], reverse=True )
        if len(pareto) > count:
            pareto = pareto[:count]
            
        # Simple plot.
        labels, times = zip(*pareto)
        labels = [ label.replace( 'Ampliseq', 'AS' ).replace(' ', '\n' ) for label in labels ]
        
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
                 ('Hd'            , 'HD'  ),
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
                 ]
        
        for (s, r) in mods:
            name = name.replace( s , r )
            
        return name
    
    
    def full_timing_plot( self , outdir , init_timing={} , analysis_timing={} ):
        """ Test plot showing all submodules under the main module plot... """
        fig = plt.figure( figsize=(12,4) )
        ax  = fig.add_subplot( 111 )
        upper  = True
        mods   = ['libprep','templating','sequencing','postrun']
        colors = ['blue','green','darkcyan','orange']
        height = 1.5
        
        if init_timing or analysis_timing:
            # We need to account for a second bar for init/analysis
            y     = 4.75
            par_y = 1.75

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
            
        for module,color in zip( mods, colors ):
            print( module )
            print( self.data[module] )
            print( self.data[module]['overall'] )
            print( '------' )
            if not self.data[module]['overall'].get('start',{}):
                continue
            
            # Shade background and add text
            o     = self.data[module]['overall']
            start = self.get_relative_time( o['start'] )
            end   = self.get_relative_time( o['end'] )
            mid   = start + (end-start)/2
            ax.axvspan( start, end, color=color, alpha=0.4 )
            ax.text   ( mid , mod_y, '{}\n{}'.format( module.title(), get_hms( o['duration'] ) ), weight='bold',
                        ha='center', va='bottom', color=color, fontsize=10 )
            
            # Create submodule bars if they exist
            if 'submodules' in self.data[module]:
                sm    = self.data[module]['submodules']
                info  = [ (self.cleanse_submodule( k ), sm[k]['start'], sm[k]['duration'] ) for k in sm ]
                procs = sorted( info, key=lambda x: x[1] )
                for proc in procs:
                    # Let's skip the module if the duration is 0.
                    if proc[2] == 0:
                        print( 'Skipping module with zero duration . . .' )
                        continue
                    
                    left = self.get_relative_time( proc[1] )
                    ax.barh( y, proc[2], height=height, left=left, align='center',
                             color=color, edgecolor='black', zorder=3 )
                    
                    # If process is too close to start of module, lets bump it forward by 30 minutes
                    label_x_pos = left + proc[2]/2.
                    point_x_pos = left + proc[2]/2.
                    if (end - label_x_pos ) < 0.5 and proc[2] <= 0.6:
                        label_x_pos -= 0.25
                        ha = 'right'
                    elif (label_x_pos < 1 or (label_x_pos - start) < 1) and (proc[2] <= 0.6):
                        label_x_pos += 0.25
                        ha = 'left'
                    else:
                        ha = 'center'
                        
                    # If process is > 0.66 hour, put label inside bar.
                    if proc[2] > 0.66:
                        ax.text( label_x_pos, y, proc[0], ha='center', va='center', fontsize=8,
                                  color='white', weight='normal' )
                    else:
                        arrows = dict(facecolor='black', width=0.75, headwidth=3, shrink=0.1, frac=0.1 )
                        if upper:
                            ax.annotate( proc[0], xy=(point_x_pos,y+height/2.), xycoords='data',
                                         xytext=(label_x_pos, y+height/2.+0.5), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='bottom', fontsize=8 , weight='normal' )
                            upper = False
                        else:
                            ax.annotate( proc[0], xy=(point_x_pos,y-height/2.), xycoords='data',
                                         xytext=(label_x_pos, y-height/2.-0.5), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='top', fontsize=8 , weight='normal' )
                            upper = True
                            
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
            for i in info:
                print( i )
                
            procs = sorted( info, key=lambda x: x[1] )
            
            # Need to create value for analysis completion and sample analysis completion.
            final_step      = procs[len(procs)-1]
            analysis_done   = self.get_relative_time( final_step[1] ) + final_step[2]
            samples_done    = self.get_relative_time( analysis_timing['Samples']['end'] )
            basecaller_done = self.get_relative_time( analysis_timing['BaseCallingActor']['end'] )
            
            # Measure finish time, either postrun or analysis
            el_fin  = max( analysis_done, runtime )

            # Setup other details, including a general "analysis" label
            patches = []
            colors  = [matplotlib.cm.Set1(i) for i in np.linspace(0,1,9)]
            ax.text( self.get_relative_time(procs[0][1])-0.1, par_y, 'Analysis', ha='right', va='center',
                     fontsize=10, color='black', weight='bold' )
            
            for i,proc in enumerate(procs):
                left = self.get_relative_time( proc[1] )
                patch = ax.barh( par_y, proc[2], height=height, left=left, align='center', color=colors[i],
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
                        ax.text( label_x_pos, par_y, proc[0], ha='center', va='center', fontsize=8,
                                  color='white', weight='normal' )
                    else:
                        arrows = dict(facecolor='black', width=0.75, headwidth=3, shrink=0.1, frac=0.1 )
                        if upper:
                            ax.annotate( proc[0], xy=(point_x_pos,par_y+height/2.), xycoords='data',
                                         xytext=(label_x_pos, par_y+height/2.+0.5), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='bottom', fontsize=8 , weight='normal' )
                            upper = False
                        else:
                            ax.annotate( proc[0], xy=(point_x_pos,par_y-height/2.), xycoords='data',
                                         xytext=(label_x_pos, par_y-height/2.-0.5), textcoords='data',
                                         arrowprops=arrows,
                                         ha=ha, va='top', fontsize=8 , weight='normal' )
                            upper = True
                else:
                    ax.legend( handles=patches, loc='lower center', fontsize=10, ncol=len(patches) )
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
            ttl += '  |  BaseCaller Complete: {}'.format( get_hms(basecaller_done) )
            ttl += '  |  Sample Analysis Complete: {}'.format( get_hms(samples_done) )
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
