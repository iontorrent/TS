#!/usr/bin/env python
# Copyright (C) 2018 Ion Torrent Systems, Inc. All Rights Reserved
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, textwrap
import numpy as np
import numpy.ma as ma
import json
import time
import re
import urllib
moduleDir = os.path.abspath( os.path.dirname( __file__ ))

from ion.plugin import *
from tools import chipcal, chiptype, explog, PluginMixin
import subprocess
import pybam
import bfmask

# Import tools for reading h5 files (ionstats output)
import h5py

# Import multilane plotting tools
from multilane_plot import MultilanePlot

# Import separator.spatial.h5 processing
from tools.SeparatorSpatialProcessing import SeparatorSpatialMetricExtractor as SSME

# For pybam -- so that data fits into int16 or uint16 for non-barcoded reads
NBC = 2**15-1

class Lane_Diagnostics( IonPlugin, PluginMixin.PluginMixin ):
    ''' 
    Plugin to assist in debugging and understanding multilane sequencing runs and their chips
    
    Now managed by Brennan Pursley
    
    Latest updates | 26Jun2019  | PluginMixin bugfix
    Latest updates | 17Apr2019  | **MAJOR**
                                | v 2.0.0
                                | Added aligned metrics from ionstats json
                                |   --> created AQ class for processing
                                | Renamed aligned-->bcmatch
                                | Removed q7 metrics
                                | Overhauled codebase -- except for pybam sections
                                |   --> added sections and broke out functionality to smaller methods
                                | Added disclaimer to html
                                | Added below_thresh metrics
                                | Improved nomatch calculations
                                | Added below_thresh to total
                                | v 2.0.1
                                | debug of nonbarcoded -- no barcode name key error
                                | debug of below_thresh values -- bad naming
                                | v 2.0.2
                                | added bcmatch length and local length metrics
                                | v 2.0.3
                                | added barcode-bcmatch length
                    | 15Jul2019 | updated tools with +/- inf support
                    | 23Jul2019 | **MAJOR**
                                | new logic for explog_lanes_active in tools
                    | 02Aug2019 | uprevved tools
                    | 02Aug2019*| Added badppf metrics and images
                    | 23Sep2019*| Added sunrise metrics
                    | 18Feb2020 | v2.3.6    | Added self.csa check on writing normalduck table
                                            | Changed warning message about active lanes
                    | 19Feb2020 | v2.3.7    | Fixed write_no_explog_msg bug
                    | 19Feb2020 | v2.3.8    | Tools update for csa
                    | 25Feb2020 | v2.3.9    | Tools update, implement results_dir in SSME
                    | 18May2020 | v2.3.10   | Added RunType.COMPOSITE
                    | 24Jun2020 | v2.3.11   | Handle missing barcode filepaths
                    | 30Jun2020 | v2.3.12   | Handle sigproc_dir seq folder
                    | 17Aug2020 | v2.3.13   | chipcal determine_lane debug
                                | v2.3.14   | commented out any saving of .dat files

    '''
    version       = "2.3.14"
    allow_autorun = True
    
    runtypes      = [ RunType.THUMB , RunType.FULLCHIP, RunType.COMPOSITE ]
 
    def launch( self ):
        # Get metadata
        print('Plugin Start')
        self.init_plugin()
        
        # This function makes use of values from init_plugin to gracefully exit if files are missing.
        self.exit_on_missing_files( chipcal=True, rawdata=True )

        self.metrics.update( { 'ChipType' : self.chip_type , 'lane_1':{'barcodes':{}}, 'lane_2':{'barcodes':{}}, 'lane_3':{'barcodes':{}}, 'lane_4':{'barcodes':{}} } )
        
        # Print debug output
        print('\n')
        print('Plugin Dir: {}'.format(   self.plugin_dir   ))
        print('Raw Data Dir: {}'.format( self.raw_data_dir ))
        print('Analysis Dir: {}'.format( self.analysis_dir ))
        print('Results Dir: {}'.format(  self.results_dir  ))
        print('Run Flows: {}'.format(    self.flows        ))
        print('Chiptype: {}'.format(     self.chip_type    ))
        print('\n')
               
        # Try to determine gain. Shortcut to exit if calibration gain file doesn't exist.
        self.determine_lane( )
      
        if self.is_multilane:
            # Initial lane analysis and communication with chipdb.ite
            if not self.csa:
               self.call_normalduck( )

            # Analyze multilane flowcell placement
            self.analyze_fca( )

            # Loading and per lane bfmask metrics
            self.analyze_lane_loading( )
            
            # Analyze raw-read-accuracy and, if they exist, aligned metrics
            # ...if barcoded, analyze per-barcode metrics
            self.analyze_ionstats( )

            # Analyze non-aligned metrics if they exist
            # ...if barcoded, analyze per-barcode metrics
            self.analyze_regular_pybam( )
            
            # Analyze nomatch if it exists
            self.analyze_nomatch_pybam( )

            # Calculate total reads for all lanes
            self.calc_total_reads( )

            # Analyze separator.spatial.h5 files
            self.analyze_separator_spatial()
            
            # Create outputs
            self.write_metrics( )
        else:
            print( 'Aborting analysis -- Not a multilane chip.' )
            
        ##################################################
        # Current Progress Marker
        ##################################################
        
        self.write_block_html     ( )
        self.write_html           ( )
        self.write_lane_html_files( )
        
        print( 'Plugin complete.' )
        
        sys.exit(0)
 
#############################
#   PREAMBLE FUNCTIONS      #
#############################

    def call_normalduck( self ):
        """ Contacts normalduck database for information on this chip and other relevant runs. """
        if not self.has_explog:
            print( 'Skipping call to Normalduck --> no explog' )
            return None
        print( 'Calling NormalDuck . . .' )
        info   = { 'lot'   : self.explog.metrics['CMOSLotId'],
                   'wafer' : self.explog.metrics['WaferId'],
                   'part'  : self.explog.metrics['PackageTestId'] }
        params = urllib.urlencode( info )
        f      = urllib.urlopen  ( 'http://chipdb.ite/valkyrie/lanefinder?{}'.format( params ) )
        print( 'http://chipdb.ite/valkyrie/lanefinder?{}'.format( params ) )
        
        try:
            response = json.loads( f.read() )
            response['error'] = False
            print( response )
        except ValueError:
            # There is a problem with a field preventing it's proper querying.  Server freaked out.
            response = {'error':True}
            
        # Here we pull out and parse data coming back from the server, storing in a list of experiments more or less.
        self.normalduck_data = []
        if not response['error']:
            if response['found']:
                self.normalduck_permalink = response.get( 'permalink' , None )
                for expt in response['data']:
                    row = ( expt['lane_1'] , expt['lane_2'] , expt['lane_3'] , expt['lane_4'] , expt['tn'] , expt['fc'] , expt['date'] , expt['instrument'] )
                    self.normalduck_data.append( row )
            else:
                # There is no evidence of this chip on the DB yet.  Need to give a canned response.
                print('Chip not found on ChipDB!  This is likely the first time this chip has been run.')

        print( ' . . . Communication with NormalDuck complete.\n' )
  
    def determine_lane( self , gain_lane_threshold=500. ):
        ''' 
        Determines which lane(s) where run based off of the gain file
        Currently default gain_lane_threshold is same as chosen in tools/chipcal.py::determine_lane()
        '''
        print('Determining active lanes . . .')
        
        #Initialize the lane information with null values for error reporting
        self.lane      = [None, None, None, None]
        
        # Load gain and determine reference pixels using a gain cutoff. *From chipDiagnostics
        if self.has_chipcal:
            #NOTE: This sets the class global attribute cc
            self.cc = chipcal.ChipCal( self.calibration_dir , self.chip_type , self.results_dir )
            self.cc.load_gain  ( )
            self.cc.find_refpix( )
            self.cc.determine_lane( gain_threshold=gain_lane_threshold )
        else:
            print( 'Unable to find calibration gain file.  It is likely that the file has been deleted.' )
            self.is_multilane = False
            return None
        
        #NOTE: This sets the class global attributes of is_multilane
        self.is_multilane = bool(self.cc.is_multilane) # For similarity with ChipDB
        
        # Address what to do if this is thumbnail.
        if self.thumbnail:
            #NOTE: This sets the class global attribute blocksize
            self.blocksize = [10,10]
        else:
            self.blocksize = [self.cc.chiptype.miniR , self.cc.chiptype.miniC]
        
        # Previously, we counted number of fluidically addressable wells for each lane here
        # ChipCal.determine_lane() now does this.
        self.unused_and_active_lanes = []
        for i in range(1,5):
            lane                                = 'lane_{}'.format(i)
            is_active                           = bool( getattr(self.cc , lane) )
            self.lane[i-1]                      = is_active
            self.metrics[lane]['active']        = self.lane[i-1]
            
            if self.has_explog:
                #NOTE: If explog is not found, unused_and_active_lanes will be empty
                #       and the explog_active metrics will not be stored
                explog_active                       = self.explog_lanes_active[lane]
                self.metrics[lane]['explog_active'] = explog_active          

                # Validate active lane
                if not self.validate_lane_active( i, is_active ):
                    self.unused_and_active_lanes.append( i )

            # for access from class object
            setattr( self , 'lane_{}_active'.format( i ) , self.lane[i-1] ) 
            
            # For backwards compatability.
            k               = 'fluidically_addressable_wells_count_lane{}'.format(str(i))
            self.metrics[k] = float( self.cc.lane_metrics[lane].get( 'addressable_wells', 0 ) )
            self.metrics[lane]['fluidically_addressable_wells'] = self.metrics[k]

        if self.unused_and_active_lanes:
            print( 'WARNING: Lanes {} are active but not processed in this plugin analysis'.format( ','.join( str(x) for x in self.unused_and_active_lanes ) ) )
            
        # Add other metrics
        others = { 'Which_Lane' : self.lane, 'If_Multilane' : self.is_multilane }
        self.metrics.update( others )
        
        print(' . . . Active lanes determined.\n' )
        return None

#################################
#   ANALYZE FLOWCELL ACCURACY   #
#################################

    def analyze_fca( self ):
        """ Uses chip gain to measure accuracy of flowcell placement.  Leverages code from tools/chipcal. """
        fca_metrics = self.cc.measure_flowcell_placement( outdir=self.results_dir )

        # Update metrics.
        if fca_metrics:
            for k in fca_metrics:
                self.metrics[k]['fca'] = fca_metrics[k]
    
#############################
#   ANALYZE LANE LOADING    #
#############################

    def analyze_lane_loading( self ):
        '''Analyze beadfind metrics. Also, find if there are bubbles in the ignore bin'''
        self.lane_median_loading = [0,0,0,0]

        if self.has_fc_bfmask:
            fc_bf_mask = self.fc_bfmask_path
        else:
            print( 'It appears that analysis.bfmask.bin has already been deleted, or was never created.  Skipping!')
            
        if self.is_multilane:
            print('\nStarting lane loading analysis . . .')
            
            # Import bfmask.BeadfindMask and run with its canned analysis. 
            self.bf = bfmask.BeadfindMask( fc_bf_mask )
            
            # Do lane specific loading heatmaps
            self.lane_loading_lane_heatmaps()
                
            # Create basic plot of median loading per lane.
            self.lane_loading_basic_median_plot()
            
            ### Start detailed metric analysis here. ###
            
            # Calculate mask array for fluidically active and *NOT* ignored wells.
            self.bf.select_mask('ignore')
            if self.thumbnail:
                ignore = self.bf.current_mask
                self.bf.select_mask('pinned')
                pinned = self.bf.current_mask
                self.bf.select_mask('ignore')
                unignored_active = np.logical_not( np.logical_and( pinned , ignore ) )
            else:
                unignored_active = np.logical_and( self.cc.active , np.logical_not( self.bf.current_mask )  )
            
            # Analyze Bead Mask
            self.lane_loading_plots_and_metrics( 'bead', 'Loading Density', 'Loading', unignored_active )
                    
            # Analyze Filtpass Mask
            self.lane_loading_plots_and_metrics( 'filtpass', 'Filtpass Density', 'Wells Passing Filters', unignored_active )
                    
            # Analyze Useless Wells
            self.lane_loading_plots_and_metrics( 'useless', 'Useless Density', 'Useless Wells -- Empty + Badkey', unignored_active )

            # Analyze Badppf Wells
            self.lane_loading_plots_and_metrics( 'badppf', 'Badppf Density', 'Badppf', unignored_active )
                   
            print ( '. . . Loading analysis complete.' )            
        else:
            print ('Skipping loading analysis, as this was not detected to be a multilane chip!' )
            
        return None

    def lane_loading_lane_heatmaps( self ):
        ''' Make lane specific heat maps for loading '''
        self.bf.select_mask( 'bead' )
        if self.thumbnail:
            x = self.bf.block_reshape( self.bf.current_mask , [10,10] )
        else:
            x = self.bf.block_reshape( self.bf.current_mask , [self.bf.chiptype.miniR,self.bf.chiptype.miniC] )
        for lane_id, lane, active in self.iterlanes():
            idx = lane_id - 1
            array = 100. * np.array(x , float).mean(2)[:,x.shape[1]/4*idx:x.shape[1]/4*(idx+1)]
            clims = [0,100]
            w     = 0.94
            h     = 2.5
            fig   = plt.figure( figsize=(w,h) )
            ax    = fig.add_subplot(111)
            plt.axis  ( 'off' )
            plt.imshow( array , origin='lower' , aspect='equal' , interpolation='nearest' , clim=clims )
            
            # Set extent of image to exactly that of the chip image
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fname  = '{}_{}_heatmap.png'.format( self.bf.current_mask_name, lane )
            plt.savefig( os.path.join( self.results_dir , fname ) , bbox_inches=extent )
            plt.close  ( )
            
            self.lane_median_loading[idx]       = np.median(array)
            self.metrics[ lane ]['p50_loading'] = self.lane_median_loading[idx]

    def lane_loading_basic_median_plot( self ):
        ''' Create basic plot of median loading per lane. '''
        plt.figure ( )
        plt.plot   ( [1,2,3,4] , self.lane_median_loading , 'o' )
        plt.ylabel ( 'Lane Median Loading %' )
        # plt.xticks ( [1,2,3,4] )
        plt.xlabel ( 'Lane' )
        plt.xlim   ( (0.5, 4.5) )
        plt.title  ( 'Lane Median Loading %')
        plt.grid   ( )
        plt.savefig( os.path.join( self.results_dir , 'Lane_Median_Loading.png' ) )
        plt.close  ( )

    def lane_loading_plots_and_metrics( self, mask, title, xlabel, unignored_active ):
        ''' Takes mask and processes plots and data '''
        self.bf.select_mask( mask )
        # assign mask to local variable
        current_mask   = self.bf.current_mask
        # special handling for useless mask
        if mask=='useless':
            if self.thumbnail:
                density = 100. * BlockReshape ( current_mask , self.blocksize , unignored_active ).get_mean()
            else:
                density = 100. * BlockReshape ( current_mask , self.blocksize , self.cc.refpix ).get_mean()
        else:
            density = 100. * BlockReshape ( current_mask , self.blocksize ).get_mean()
        # set up plotting
        mplot   = MultilanePlot( density , title, xlabel, '%', clims=[0,100])
        # special handling for bead mask
        title_all           = 'multilane_plot_all_{}.png'
        title_all_rescaled  = 'multilane_plot_all_{}_RESCALED.png'
        if mask == 'bead':
            mplot.plot_all( os.path.join( self.results_dir , title_all.format('loading') ) )
        elif mask == 'badppf':
            mid_val = np.median( density[density>0] )
            low  = max( mid_val-10, 0 )
            high = min( mid_val+10, 100 )
            mplot.update_clims( [low,high,] )
            mplot.plot_all( os.path.join( self.results_dir , title_all_rescaled.format('badppf') ) )
            mplot.update_clims( [0,100,] )
            mplot.plot_all( os.path.join( self.results_dir , title_all.format('badppf') ) )
        else:
            mplot.plot_all( os.path.join( self.results_dir , title_all.format(mask) ) )
        
        # lane_metric is used in loop below
        if mask   == 'bead':     
            lane_metric     = 'lane_beads'
            percent_metric  = 'loading'
        elif mask == 'filtpass': 
            lane_metric     = 'filtpass_beads'
            percent_metric  = 'percent_filtpass'
        elif mask == 'useless': 
            lane_metric     = 'useless_wells'
            percent_metric  = 'percent_useless'
        elif mask == 'badppf': 
            lane_metric     = 'badppf_wells'
            percent_metric  = 'percent_badppf'
        else:                    
            lane_metric     = 'active_wells_sum_metric'
            percent_metric  = 'active_wells_percent_metric'
        midline_metric = '{}_midline'.format( percent_metric )
        midline_top    = '{}_midline_top'.format( percent_metric )
        midline_bot    = '{}_midline_bot'.format( percent_metric )

        # extract and plot lane metrics
        title_one           = 'multilane_plot_lane_{}_{}.png'
        title_one_rescaled  = 'multilane_plot_lane_{}_{}_RESCALED.png'
        nrows = current_mask.shape[0]
        rows_top = slice( int(nrows*100/200), int(nrows*101/200) )
        rows_bot = slice( int(nrows* 99/200), int(nrows*100/200) )
        for (lane_id, lane, active) in self.iterlanes():
            if active:
                # special handling for bead mask
                if mask=='bead':
                    mplot.plot_one( lane_id , os.path.join( self.results_dir , title_one.format(lane_id, 'loading') ) )
                elif mask=='badppf':
                    slice_   = mplot.lane_slice(lane_id)
                    mid_val = np.median( slice_[slice_>0] )
                    low  = max( mid_val-15, 0 )
                    high = min( mid_val+15, 100 )
                    mplot.update_clims( [low,high,] )
                    mplot.plot_one( lane_id , os.path.join( self.results_dir , title_one_rescaled.format(lane_id, mask) ) )
                    mplot.update_clims( [0,100,] )
                    mplot.plot_one( lane_id , os.path.join( self.results_dir , title_one.format(lane_id, mask) ) )
                else:
                    mplot.plot_one( lane_id , os.path.join( self.results_dir , title_one.format(lane_id, mask) ) )
                lane_array    = self.get_lane_slice( current_mask , lane_id )
                active_wells  = self.get_lane_slice( unignored_active , lane_id )
                
                lane_sum = lane_array[ active_wells ].sum()

                self.metrics[lane][ lane_metric ] = int( lane_sum )

                # special processing cases
                if mask == 'filtpass':
                    div     = float( self.metrics[lane][ 'lane_beads' ] )
                    div_top = float( self.metrics[lane][ 'loading_midline_top' ] ) * active_wells[rows_top].sum()
                    div_bot = float( self.metrics[lane][ 'loading_midline_bot' ] ) * active_wells[rows_bot].sum()
                else:
                    div     = float( active_wells.sum() )
                    div_top = float( active_wells[rows_top].sum() )
                    div_bot = float( active_wells[rows_bot].sum() )

                self.metrics[lane][percent_metric] = 100. * float( lane_sum ) / div

                # Sunrise detection
                perc_top = 100. * float(lane_array[rows_top].sum()) / div_top
                perc_bot = 100. * float(lane_array[rows_bot].sum()) / div_bot
                midline_delta = perc_bot- perc_top

                self.metrics[lane][midline_metric] = midline_delta
                self.metrics[lane][midline_top]    = perc_top
                self.metrics[lane][midline_bot]    = perc_bot


#########################
#   ANALYZE IONSTATS    #
#########################

    def analyze_ionstats( self ):
        ''' Analyze raw-read-accuracy and, if they exist, aligned metrics
            ...if barcoded, analyze per-barcode metrics
        '''
        print( '\nStarting Ionstats Analysis . . .' )

        bc_path_list = [bc['bam_filepath'] for bc in self.barcodes.values()]
        # Trim list to just aligned bams --> non-aligned live in basecaller_dir
        bc_path_list = [ p for p in bc_path_list if self.basecaller_dir not in p ]

        # Check if bc_path_list is populated
        if len(bc_path_list)>0:
            # define an h5 file output path for raw-read accuracy analysis
            h5_file     = os.path.join( self.results_dir , 'ionstats_error_summary.h5' )
            # define a json file output path for aligned metrics analysis
            json_file   = os.path.join( self.results_dir , 'ionstats_alignment.json' )
            
            ### BUILD THE IONSTATS CALL ###
            base_cmd = 'ionstats alignment --chip-origin 0,0 '
            # Handle thumbnail case
            if self.thumbnail:
                base_cmd += '--chip-dim 1200,800 --subregion-dim 10,10 '
            else:
                base_cmd += '--chip-dim {0.chipC},{0.chipR} --subregion-dim {0.miniC},{0.miniR} '.format(self.cc.chiptype)
            # Settings for processing behavior    
            base_cmd += '--evaluate-hp --skip-rg-suffix .nomatch --n-flow {} --max-subregion-hp 6 '.format(self.flows)
            # Settings for output behavior
            base_cmd += '--output-h5 {} --output {} '.format( h5_file, json_file )

            ### FIRST: process all aligned bams ###
            self.ionstats_processing_full( base_cmd, bc_path_list, h5_file, json_file )
            ### SECOND: individually process barcode bams ###
            self.ionstats_processing_barcodes( base_cmd, bc_path_list, h5_file, json_file )

        else:
            print( '\nAborting Ionstats Analysis -- No aligned bam files.' )
            return None
               
    def ionstats_processing_full( self, base_cmd, bc_path_list, h5_file, json_file ):
        """ FULL Ionstats Processing.  This code leveraged from RRA_Spatial Plugin - thanks Charlene!"""
        #NOTE:  Requires base ionstats input command 
        full_cmd = base_cmd + '-i {} '.format( ','.join( bc_path_list ) )
        print( '\n Ionstats full call:\n' )
        print( full_cmd )
        self.ionstats_call( full_cmd )
        # Extract RRA values from h5 file
        ct = self.cc.chiptype
        if os.path.exists( h5_file ):
            start_time = time.time()
            if self.thumbnail: rra = RRA( h5_file , [800 , 1200] , [10 , 10] )
            else:              rra = RRA( h5_file , [ct.chipR , ct.chipC] , [ct.miniR , ct.miniC] )
            self.ionstats_rra_plots_and_metrics( rra )
            # MOVE h5 file for troubleshooting
            self.ionstats_move_file( h5_file, 'ionstats_FULL_error_summary.h5' )
            print( ' . . . FULL Ionstats H5 Processing Completed in {:0.1f} seconds.'.format( (time.time()-start_time) ) )
        else:
            print( ' . . . Error with running ionstats.  FULL Ionstats H5 Processing failed.\n' )

        # Extract AQ values from json file
        if os.path.exists( json_file ):
            start_time = time.time()
            if self.thumbnail: aq = AQ( json_file , [800 , 1200] , [10 , 10] )
            else:              aq = AQ( json_file , [ct.chipR , ct.chipC] , [ct.miniR , ct.miniC] )
            self.ionstats_aq_metrics( aq )
            # MOVE json file for troubleshooting
            self.ionstats_move_file( json_file, 'ionstats_FULL_alignment.json' )
            print( ' . . . FULL Ionstats JSON Processing Completed in {:0.1f} seconds.'.format( (time.time()-start_time) ) )
        else:
            print( ' . . . Error with running ionstats.  FULL Ionstats JSON Processing failed.\n' )

        return None
 
    def ionstats_processing_barcodes( self, base_cmd, bc_path_list, h5_file, json_file ):
        """ BARCODE Ionstats Processing.  This code leveraged from RRA_Spatial Plugin - thanks Charlene!"""
        # First we have to run ionstats again.  Let's build the command
        print( '\nStarting BARCODE Ionstats Processing . . .' )
        for key, bc in self.barcodes.items():
            fpath = bc['bam_filepath']
            if (fpath not in bc_path_list) or (key=='nonbarcoded'):
                continue
            # If something makes it through without a barcode name, handle here (nonbarcoded should be handled above)
            try:             
                bc_name = bc['barcode_name']
            except KeyError: 
                print( '\nNO BARCODE NAME -- filepath: {}'.format( fpath ) )
                continue

            bc_cmd  = base_cmd + '-i {} '.format( fpath )
            print( 'Barcode {} ionstats call:'.format(bc_name) )
            print( bc_cmd )
            self.ionstats_call( bc_cmd )
            # Analyze the file and make some plots -- first arbitrary clims, then fixed?.
            ct = self.cc.chiptype
            if os.path.exists( h5_file ):
                start_time = time.time()
                if self.thumbnail: rra = RRA( h5_file , [800 , 1200] , [10 , 10] )
                else:              rra = RRA( h5_file , [ct.chipR , ct.chipC] , [ct.miniR , ct.miniC] )
                self.ionstats_rra_plots_and_metrics( rra, barcode=bc_name )
                # DELETE h5 file
                self.ionstats_delete_file( h5_file )
                print( ' . . . BARCODE Ionstats H5 Processing Completed in {:0.1f} seconds.'.format( (time.time()-start_time) ) )
            else:
                print( ' . . . Error with running ionstats.  BARCODE Ionstats Processing failed.\n' )
            # Extract AQ values from json file
            if os.path.exists( json_file ):
                start_time = time.time()
                if self.thumbnail: aq = AQ( json_file , [800 , 1200] , [10 , 10] )
                else:              aq = AQ( json_file , [ct.chipR , ct.chipC] , [ct.miniR , ct.miniC] )              
                self.ionstats_aq_metrics( aq, barcode=bc_name )
                # DELETE json file for troubleshooting
                self.ionstats_delete_file( json_file )
                print( ' . . . BARCODE Ionstats JSON Processing Completed in {:0.1f} seconds.'.format( (time.time()-start_time) ) )
            else:
                print( ' . . . Error with running ionstats.  BARCODE Ionstats JSON Processing failed.\n' )
        return None

    def ionstats_call( self, cmd ):
        ''' Launch ionstats '''
        start_time = time.time()
        subprocess.call( cmd , shell=True )
        print( 'ionstats completed in {:0.1f} seconds.'.format( (time.time()-start_time) ) )

    def ionstats_delete_file( self, file ):
        ''' Delete file '''
        r_cmd  = 'rm '
        r_cmd += file
        subprocess.call( r_cmd , shell=True )

    def ionstats_move_file( self, file, new_name ):
        ''' Move file for troubleshooting '''
        mv_cmd  = 'mv '
        mv_cmd += file
        mv_cmd += ' '
        mv_cmd += os.path.join( self.results_dir, new_name )
        subprocess.call( mv_cmd , shell=True )

    def ionstats_rra_plots_and_metrics( self, rra, barcode=None ):
        ''' Output relevant RRA plots and store metrics '''
        title = 'Local Raw Read Accuracy'
        if barcode:  title += '\n{}'.format( barcode )
        rra_plot = MultilanePlot( rra.rra , title , 'RRA' , '%' , clims=[95,100] , bin_scale=25 )
        title = None

        fname = 'multilane_plot_all_rra'
        if barcode: fname += '_{}'.format( barcode )
        fname += '.png'
        rra_plot.plot_all( os.path.join( self.results_dir , fname ) )
        fname = None

        for (lane_id, lane, active) in self.iterlanes():
            if active:
                fname = 'multilane_plot_lane_{}_rra'.format( lane_id )
                if barcode: fname += '_{}'.format( barcode )
                fname += '.png'       
                rra_plot.plot_one( lane_id , os.path.join( self.results_dir , fname ) )
                fname = None

                local_rra_dict = {'local_accuracy': self.get_lane_metrics( rra.rra , lane_id , 0 ) }
                if not barcode:
                    self.metrics[lane].update( local_rra_dict )
                else:
                    try:
                        self.metrics[lane]['barcodes'][barcode].update( local_rra_dict )
                    except KeyError:
                        self.metrics[lane]['barcodes'].update({barcode:local_rra_dict}) 
        
        # Change scaling and fix limits --> replot with 'fixed' suffix
        rra_plot.bin_scale = 50
        rra_plot.update_clims( [98,100] )
        
        fname = 'multilane_plot_all_rra'
        if barcode: fname += '_{}'.format( barcode )
        fname += '_fixed.png'
        rra_plot.plot_all( os.path.join( self.results_dir , fname ) )
        fname = None

        for (lane_id, lane, active) in self.iterlanes():
            if active:
                fname = 'multilane_plot_lane_{}_rra'.format( lane_id )
                if barcode: fname += '_{}'.format( barcode )
                fname += '_fixed.png'       
                rra_plot.plot_one( lane_id , os.path.join( self.results_dir , fname ) )
                fname = None

    def ionstats_aq_metrics( self, aq, barcode=None ):
        ''' Store AQ metrics.  Lengths are weighted by reads per region. '''
        # Create flow-based readlength scale
        for (lane_id, lane, active) in self.iterlanes():
            if active:
                aq20_metrics_dict = {'aq20_length':self.get_lane_metrics( aq.aq20_length , lane_id , 0 , reads_weight=aq.aq20_reads ),
                                    'aq20_reads' : self.get_lane_slice(aq.aq20_reads, lane_id).sum(),
                                    }
                aq07_metrics_dict = {'aq07_length':self.get_lane_metrics( aq.aq07_length , lane_id , 0 , reads_weight=aq.aq07_reads ),
                                    'aq07_reads' : self.get_lane_slice(aq.aq07_reads, lane_id).sum(),
                                    }

                if not barcode:
                    self.metrics[lane].update( aq20_metrics_dict )
                    self.metrics[lane].update( aq07_metrics_dict )
                else:
                    try:
                        self.metrics[lane]['barcodes'][barcode].update(aq20_metrics_dict)
                    except KeyError:
                        self.metrics[lane]['barcodes'].update({barcode:aq20_metrics_dict})
                    try:
                        self.metrics[lane]['barcodes'][barcode].update(aq07_metrics_dict)
                    except KeyError:
                        self.metrics[lane]['barcodes'].update({barcode:aq07_metrics_dict})

#############################
#   ANALYZE PYBAM FILES     #
#############################

    def analyze_regular_pybam( self ):
        '''get lane specific results using pybam. 
        Call Ionstats to generate region specific seuqencing metrics'''

        #NOTE:  Total reads == Raw + NoMatch (if exists)
        #           Total reads caclulated elswhere after attempting to process NoMatch bam

        #Initialize outputs
        self.lane_block_aq20_results={1:{},2:{},3:{},4:{}}   #TO DO: Fill this in
        self.lane_aq20_results      ={1:{},2:{},3:{},4:{}}
        
        if self.is_multilane:
            print ( '\nStarting Lane Aligned Results . . .' )
            print ( 'Full chip Detected, starting pybam' )
        else:
            print ( 'Aborting bam file analysis -- this is not a multilane chip.' )
            return None
        
        quality = [ -1, 0, 7, 20 ] # -1: barcode mask
                                   #  0: read length
                                   #  7: q7 length
                                   # 20: q20 length
        if self.thumbnail == True:
            shape = [800, 1200, len(quality)]
        else:
            shape = [self.cc.rows, self.cc.cols, len(quality)]

        # If barcodes are present, then do not merge the non-barcode reads
        # If only non-barcoded reads are present, then read those
        # Check if non-barcoded reads are present
        self.nobc = NBC in [ bc.get( 'barcode_index', NBC ) for bc in self.barcodes.values() ]
        # Check if barcoded reads are present
        self.bc   = any( [ bc.get( 'barcode_index', NBC ) != NBC for bc in self.barcodes.values() ] )
        self.merge_nbc = self.nobc and not self.bc

        qualmat = self.load_barcode_bams_into_qualmat( shape, quality )           

        #qualmat[:,:,0].astype( np.int16 ).tofile( 'barcode_mask.dat' )
        self.bcmatch_length   = qualmat[:,:,1].astype( np.int16 )
        self.bcmatch_reads    = qualmat[:,:,1].astype( np.bool  )
        self.q20Len       = qualmat[:,:,3].astype( np.int16 )

        # save q20Len for posterity
        # NOTE: commented out for CSA use 17 Aug 2020 B.P.
        #self.save_q20Len()
        
        # For barcodes, look at just the locations and lengths, but keep them separate
        qualmat = qualmat[:,:,:2]
        qualmat[:,:,1] = 0
        quality = [ -1, 0 ] # -1: barcode mask
                            #  0: read length
        if not( self.merge_nbc ):
            # make a mask of non-barcoded reads
            for bc in self.barcodes.values():
                fill     = bc.get( 'barcode_index', NBC )
                if ( fill == NBC ):
                    filename = bc['bam_filepath']
                    print( 'procesing new bamfile:' )
                    print( filename )
                    pybam.loadquals( filename, quality, qualmat, fill=fill )

            #qualmat[:,:,1].astype( np.int16 ).tofile( 'nbc_length.dat' )
            #qualmat[:,:,1].astype( np.bool ).tofile( 'nbc_totalreads.dat' )
            
        self.bcmask = qualmat[:,:,0].astype( np.int16 )
              
        ##################################################
        # BC-Match Length
        ##################################################
        
        # Create a flow-based readlength scale
        flowlims, vmr_flowlims = self.create_flow_based_readlength_scale( )
            
        # non-local-averaged data (pure histograms of individual well read lengths)
        rl = np.zeros( self.bcmatch_length.shape )
        rl[ self.bcmatch_length > 25 ] = self.bcmatch_length[ self.bcmatch_length > 25 ]
        rl_hists = MultilanePlot( rl , 'BC-Match Read Length' , 'Read Length' , units='bp' , clims=flowlims )
        rl_hists.plot_histograms_only( os.path.join( self.results_dir, 'multilane_plot_bcmatch_rl_histograms.png' ) )

        bcmrl = BlockReshape( self.bcmatch_length, self.blocksize, self.bcmatch_length <= 25 )
 
        for (i, lane, active) in self.iterlanes():
            if active:
                self.metrics[lane]['bcmatch_length']              = self.get_lane_metrics( self.bcmatch_length , i, 25 )
                self.metrics[lane]['bcmatch_length']['localmean'] = self.get_lane_metrics( bcmrl.get_mean() , i , 0 )             

                rl_hists.plot_single_lane_histogram_only( i, os.path.join( self.results_dir, 'multilane_plot_lane_{}_bcmatch_rl_histogram.png'.format(i) ) )

                if self.bc:
                    for bc in self.barcodes.values():
                        bc_name = bc['barcode_name']
                        bc_ind  = bc['barcode_index']
                        try:
                            self.metrics[lane]['barcodes'][bc_name].update( {'bcmatch_length': self.get_lane_metrics( self.bcmatch_length , i , 25, add_vmr=True, bc_ind=bc_ind, bcmask = self.bcmask ) } )
                        except KeyError:
                            self.metrics[lane]['barcodes'].update( { bc_name: {'bcmatch_length': self.get_lane_metrics( self.bcmatch_length , i , 25, add_vmr=True, bc_ind=bc_ind, bcmask = self.bcmask ) } } )
        del rl
        
        ##################################################
        # Reads (bcmatch and Q20)
        ##################################################
        
        bcm_reads = np.zeros( self.bcmatch_reads.shape )
        bcm_reads[ self.bcmatch_length>25 ] = self.bcmatch_reads[ self.bcmatch_length>25 ]

        below_thresh_bcm_reads = np.zeros( self.bcmatch_reads.shape )
        below_thresh_bcm_reads[ self.bcmatch_length<=25 ] = self.bcmatch_reads[ self.bcmatch_length<=25 ]

        ar_block = 100. * BlockReshape( bcm_reads , self.blocksize ).get_mean()
        ar = MultilanePlot( ar_block , 'BC-Match Read Density' , 'Wells with BC-Match Reads' , '%' , 
                            clims=None , bin_scale=2 )
        ar.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_bcmatch_read_density.png' ) )
        
        for (i, lane, active) in self.iterlanes():
            if active:
                ar.plot_one( i , os.path.join( self.results_dir , 'multilane_plot_lane_{}_bcmatch_read_density.png'.format(i) ) )
                bcm_reads_slice                         = self.get_lane_slice( bcm_reads, i )
                self.metrics[lane]['bcmatch_reads']     = bcm_reads_slice.sum()

                below_thresh_bcm_reads_slice                        = self.get_lane_slice( below_thresh_bcm_reads, i )
                self.metrics[lane]['below_thresh_bcmatch_reads']    = below_thresh_bcm_reads_slice.sum()

                self.metrics[lane]['local_bcmatch_reads']   = self.get_lane_metrics( ar_block , i , 0 )
                # get barcode dependent values per lane for bcmatch_reads
                if self.bc:
                    bc_lane_slice = self.get_lane_slice( self.bcmask, i )
                    for bc in self.barcodes.values():
                        bc_name = bc['barcode_name']
                        bc_ind  = bc['barcode_index']
                        try:
                            self.metrics[lane]['barcodes'][bc_name].update( {'bcmatch_reads':bcm_reads_slice[bc_lane_slice==bc_ind].sum()} )
                        except KeyError:
                            self.metrics[lane]['barcodes'].update( { bc_name: {'bcmatch_reads':bcm_reads_slice[bc_lane_slice==bc_ind].sum()} } )          
                
        q20_read_density = 100. * BlockReshape( (self.q20Len > 25) , self.blocksize ).get_mean()
        q20_reads = MultilanePlot( q20_read_density , 'Q20 Read Density' , 'Wells with Q20 Reads' , '%' , 
                                   clims=None , bin_scale=2)
        q20_reads.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_q20_read_density.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                q20_reads.plot_one( i , os.path.join( self.results_dir , 
                                                      'multilane_plot_lane_{}_q20_read_density.png'.format(i) ) )
                self.metrics[lane]['local_q20_reads'] = self.get_lane_metrics( q20_read_density  , i , 0 )
                q20reads_lane_slice = self.get_lane_slice( (self.q20Len > 25), i )
                self.metrics[lane]['q20_reads'] = q20reads_lane_slice.sum()

                if self.bc:
                    for bc in self.barcodes.values():
                        bc_name = bc['barcode_name']
                        bc_ind  = bc['barcode_index']
                        try:
                            self.metrics[lane]['barcodes'][bc_name].update( {'q20_reads':q20reads_lane_slice[bc_lane_slice==bc_ind].sum()} )
                        except KeyError:
                            self.metrics[lane]['barcodes'].update( { bc_name: {'q20_reads':q20reads_lane_slice[bc_lane_slice==bc_ind].sum()} } )          

        # This section only useful if we have the beadfind mask file.
        if self.has_fc_bfmask:
            # loaded beads conversion into q20 reads.
            self.bf.select_mask( 'bead' )
            beads             = self.bf.current_mask
            bead_count        = BlockReshape( beads , self.blocksize ).get_sum()
            q20_read_count    = BlockReshape( (self.q20Len > 25 ) , self.blocksize ).get_sum()
            
            # Get conversions and deal with division by zero
            msk               = (bead_count == 0)
            bead_count[ msk ] = 1
            conversions       = 100. * q20_read_count / bead_count
            conversions[ msk ]= 0
            
            conv = MultilanePlot( conversions , 'Q20 Reads / Bead Conversions' , 'Q20 Reads/Beads' , '%' , 
                                  clims=[0,100] , bin_scale=2 )
            conv.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_q20_read_bead_conversions.png' ) )
            for (i, lane, active) in self.iterlanes():
                if active:
                    conv.plot_one( i , os.path.join( self.results_dir,
                                                     'multilane_plot_lane_{}_q20_read_bead_conversions.png'.format(i)) )
                    self.metrics[lane]['bead_to_q20'] = 100.*self.metrics[lane]['q20_reads']/self.get_lane_slice( beads , i ).sum()
                    self.metrics[lane]['local_bead_to_q20'] = self.get_lane_metrics( conversions , i , 0 )
        else:
            print( 'Skipping conversion analysis since no beadfind mask file was found.' )
            
        ##################################################
        # Q20 Read Length
        ##################################################
        # Define a few global metrics and colorbars
        self.metrics['q20_mrl'] = ( self.q20Len[ self.q20Len > 25 ] ).mean()
        
        q20            = BlockReshape( self.q20Len , self.blocksize , self.q20Len <= 25 )
        local_q20_mean = q20.get_mean( )
        local_q20_vmr  = q20.get_vmr ( )
        
        # Q20 Multilane Plots
        # non-local-averaged data (pure histograms of individual well read lengths)
        masked_q20 = np.zeros( self.q20Len.shape , np.int16 )
        masked_q20[ self.q20Len > 25 ] = self.q20Len[ self.q20Len > 25 ]
        q20_rl_hists = MultilanePlot( masked_q20 , 'Q20 Read Length' , 'Q20 RL' , 'bp' , clims=flowlims )
        q20_rl_hists.plot_histograms_only( os.path.join( self.results_dir , 'multilane_plot_q20_rl_histograms.png' ))

        for (i, lane, active) in self.iterlanes():
            if active:
                q20_rl_hists.plot_single_lane_histogram_only( i, os.path.join( self.results_dir, 'multilane_plot_lane_{}_q20_rl_histogram.png'.format(i) ) )

        del masked_q20
        
        # Autoscaled
        fc_q20 = MultilanePlot( local_q20_mean, 'Q20 Mean Read Length', 'Q20 MRL', 'bp', clims=None, bin_scale=1 ) 
        fc_q20.plot_all       ( os.path.join( self.results_dir , 'multilane_plot_all_q20_mrl.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                fc_q20.plot_one( i , os.path.join(self.results_dir,'multilane_plot_lane_{}_q20_mrl.png'.format(i)) )
            
        # Scaled +/- 50
        # Note that these clims will be normalized to +/- 100 bp of mean read length across all lanes.p
        clim_center = np.nan_to_num( 10. * np.floor( self.metrics['q20_mrl'] / 10. ) )
        print( 'clim_center', clim_center )
        cmax        = clim_center + 50
        cmin        = clim_center - 50
        if cmin < 0:
            cmin = 0
        q20_mrl_clims = [cmin,cmax]        
        fc_q20.update_clims( q20_mrl_clims )
        fc_q20.plot_all    ( os.path.join( self.results_dir , 'multilane_plot_all_q20_mrl_centered.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                fc_q20.plot_one( i , os.path.join( self.results_dir , 'multilane_plot_lane_{}_q20_mrl_centered.png'.format(i) ) )
            
        # Flow-based scale
        fc_q20.update_clims( flowlims )
        fc_q20.plot_all    ( os.path.join( self.results_dir , 'multilane_plot_all_q20_mrl_flow_limits.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                fc_q20.plot_one( i , os.path.join( self.results_dir , 'multilane_plot_lane_{}_q20_mrl_flow_limits.png'.format(i) ) )
            
        # Variance-Mean-Ratio Plots
        # Autoscaled and then fixed scale
        vmr = MultilanePlot( local_q20_vmr , 'Q20 RL Variance-Mean-Ratio' , 'Q20 VMR' , 'bp' , clims=None  )
        vmr.plot_all       ( os.path.join( self.results_dir , 'multilane_plot_all_q20_vmr.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                vmr.plot_one( i , os.path.join( self.results_dir , 'multilane_plot_lane_{}_q20_vmr.png'.format(i) ) )
                
        # Single Fixed Scale
        vmr.update_clims( [0,80] )
        vmr.plot_all    ( os.path.join( self.results_dir , 'multilane_plot_all_q20_vmr_fixed_scale.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                vmr.plot_one( i , os.path.join( self.results_dir , 
                                                'multilane_plot_lane_{}_q20_vmr_fixed_scale.png'.format(i) ) )
                
        # Flow-based limits
        vmr.update_clims( vmr_flowlims )
        vmr.plot_all    ( os.path.join( self.results_dir , 'multilane_plot_all_q20_vmr_flow_limits.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                vmr.plot_one( i , os.path.join( self.results_dir , 
                                                'multilane_plot_lane_{}_q20_vmr_flow_limits.png'.format(i) ) )
        
        for (lane_id, lane, active) in self.iterlanes():
            if active:
                # This is a lane slice of Q20 RL for this lane, used for "global" lane metrics.
                self.metrics[lane]['q20_length'] = self.get_lane_metrics( self.q20Len , lane_id , 25, add_vmr=True )
                
                if self.bc:
                    for bc in self.barcodes.values():
                        bc_name = bc['barcode_name']
                        bc_ind  = bc['barcode_index']
                        try:
                            self.metrics[lane]['barcodes'][bc_name].update( {'q20_length': self.get_lane_metrics( self.q20Len , lane_id , 25, add_vmr=True, bc_ind=bc_ind, bcmask = self.bcmask ) } )
                        except KeyError:
                            self.metrics[lane]['barcodes'].update( { bc_name: {'q20_length': self.get_lane_metrics( self.q20Len , lane_id , 25, add_vmr=True, bc_ind=bc_ind, bcmask = self.bcmask ) } } )
               
                # This is local mean, which goes into a heatmap and VMR calculations.
                self.metrics[lane]['q20_length']['localmean'] = self.get_lane_metrics( local_q20_mean , lane_id , 0)
 
                # Make per-lane spatial maps.  While wafermappable, probably won't use that way.
                lane_local_q20 = self.get_lane_slice( local_q20_mean , lane_id )
                self.multilane_wafermap( lane_local_q20 , lane , 'q20mrl' , clims=q20_mrl_clims , transpose=False )
                self.multilane_wafermap( lane_local_q20 , lane , 'q20mrl' , clims=q20_mrl_clims , transpose=True )
                
                # This is *local* variance-mean-ratio
                self.metrics[lane]['q20_length']['localvmr'] = self.get_lane_metrics( local_q20_vmr , lane_id , 0 )
                
                # Make per-lane spatial maps.  While wafermappable, probably won't use that way.
                lane_local_vmr = self.get_lane_slice( local_q20_vmr , lane_id )
                self.multilane_wafermap( lane_local_vmr , lane , 'q20vmr' , clims=vmr_flowlims , transpose=False )
                self.multilane_wafermap( lane_local_vmr , lane , 'q20vmr' , clims=vmr_flowlims , transpose=True )
                
        print(' . . . Finished Lane Alignment Results.\n')
        return None

    def analyze_nomatch_pybam( self ):
        '''  get lane specific no_match results using pybam. '''
        
        if self.is_multilane:
            print ( '\nStarting Lane NoMatch Results . . .' )
            print ( 'Full chip Detected, starting pybam' )
        else:
            print ( 'Aborting bam file analysis -- this is not a multilane chip.\n' )
            return None
        
        if not self.has_nomatch_bam:
            print ( 'Aborting bam file analysis -- nomatch bamfile does not exist.' )
            return None
        
        quality = [ -1, 0 ] # -1: barcode mask
                            #  0: read length
        if self.thumbnail == True:
            shape = [800, 1200, len(quality)]
        else:
            shape = [self.cc.rows, self.cc.cols, len(quality)]
            
        qualmat = np.zeros( shape, dtype=np.uint16 ) # pybam requires uint16
             
        filename = self.nomatch_bam_filepath
        print( 'procesing new bamfile:' )
        print( filename )

        fill = 1
        pybam.loadquals( filename, quality, qualmat, fill=fill )
        
        print( 'qualmat shape', qualmat.shape )
        
        # TODO: For some reason, qualmat loses data.  For now, just pad it
        qualmat_t = qualmat
        qualmat   = np.zeros( shape, dtype=np.uint16 ) # pybam requires uint16

        mrow = min( qualmat.shape[0], qualmat_t.shape[0] )
        mcol = min( qualmat.shape[1], qualmat_t.shape[1] )
        qualmat[0:mrow,0:mcol] = qualmat_t[0:mrow,0:mcol]

        #qualmat[:,:,0].astype( np.int16 ).tofile( 'barcode_mask.dat' )
        self.nomatch_length = qualmat[:,:,1].astype( np.int16 )
        self.nomatch_reads  = qualmat[:,:,1].astype( np.bool  )
                
        ##################################################
        # Reads (NoMatch)
        ##################################################

        nm_reads = np.zeros( self.nomatch_reads.shape )
        nm_reads[ self.nomatch_length>25 ] = self.nomatch_reads[ self.nomatch_length>25 ]

        below_thresh_nm_reads = np.zeros( self.nomatch_reads.shape )
        below_thresh_nm_reads[ self.nomatch_length<=25 ] = self.nomatch_reads[ self.nomatch_length<=25 ]

        
        nmr_block = 100. * BlockReshape( nm_reads , self.blocksize ).get_mean()
        nmr = MultilanePlot( nmr_block , 'NoMatch Read Density' , 'Wells with NoMatch Reads' , '%' , 
                            clims=None , bin_scale=2 )
        nmr.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_nomatch_read_density.png' ) )
        
        for (i, lane, active) in self.iterlanes():
            if active:
                nmr.plot_one( i , os.path.join( self.results_dir , 
                                               'multilane_plot_lane_{}_nomatch_read_density.png'.format(i) ) )
                self.metrics[lane]['nomatch_reads']                 = self.get_lane_slice  ( nm_reads , i ).sum()
                self.metrics[lane]['below_thresh_nomatch_reads']    = self.get_lane_slice  ( below_thresh_nm_reads , i ).sum()
                self.metrics[lane]['local_nomatch_reads']           = self.get_lane_metrics( nm_reads , i , 0 )             

        del nmr_block

        ##################################################
        # NoMatch Length
        ##################################################
            
        # non-local-averaged data (pure histograms of individual well read lengths)
        nmrl = BlockReshape( self.nomatch_length, self.blocksize, self.nomatch_length <= 25 )
 
        for (i, lane, active) in self.iterlanes():
            if active:
                self.metrics[lane]['nomatch_length']              = self.get_lane_metrics( self.nomatch_length , i, 25 )
                self.metrics[lane]['nomatch_length']['localmean'] = self.get_lane_metrics( nmrl.get_mean() , i , 0 )             

        del nmrl

    #########################################
    #   LANE_ALIGNED HELPER FUNCTIONS       #
    #########################################

    def load_barcode_bams_into_qualmat( self, shape, quality ):
        ''' Loads barcode bams into the qualmat matrix and returns qualmat '''
        qualmat = np.zeros( shape, dtype=np.uint16 ) # pybam requires uint16
        
        for bc in self.barcodes.values():
            filename = bc['bam_filepath']
            if not filename or filename == '':
                print( '!!! Filepath does not exist !!!\nDouble-check barcodes.json.  Moving on to next barcode' )
                continue
            print( 'procesing new bamfile:' )
            print( filename )
            fill     = bc.get( 'barcode_index', NBC )
            if ( fill == NBC ) and ( not self.merge_nbc ):
                print( 'Skipping merge of non-barcoded reads since barcoded reads are present' )
                continue
            pybam.loadquals( filename, quality, qualmat, fill=fill )

        print( 'qualmat shape', qualmat.shape )
        
        # TODO: For some reason, qualmat loses data.  For now, just pad it
        qualmat_t = qualmat
        qualmat   = np.zeros( shape, dtype=np.uint16 ) # pybam requires uint16

        mrow = min( qualmat.shape[0], qualmat_t.shape[0] )
        mcol = min( qualmat.shape[1], qualmat_t.shape[1] )
        qualmat[0:mrow,0:mcol] = qualmat_t[0:mrow,0:mcol]

        return qualmat

    def save_q20Len( self ):
        for (i, lane, active) in self.iterlanes():
            lane_q20 = None
            if active:
                lane_q20 = self.get_lane_slice( self.q20Len , i )
                #lane_q20.tofile( os.path.join( self.results_dir , '{}_q20_length.dat'.format( lane ) ) )
                if lane_q20 is not None:                
                    with open( os.path.join( self.results_dir , 'lane_array_size.txt' ) , 'w' ) as f:
                        f.write( '{0}\t{1}'.format( *lane_q20.shape ) )

#####################################
#   ANALYZE SEPARATOR SPATIAL DATA  #
#####################################

    def analyze_separator_spatial( self ):
        '''Analyze separator.spatial.h5 files for multilane metrics.'''
        if self.is_multilane:
            print ( '\nStarting analysis of separator.spatial.h5 files. . .' )
        else:
            print ( 'Aborting SSME analysis -- this is not a multilane chip.' )
            return None

        hw_dict = self.get_array_height_and_width()

        ssme = SSME( self.sigproc_dir, results_dir=self.results_dir, height=hw_dict['height'], width=hw_dict['width'], thumbnail=self.thumbnail )

        self.ssme_files_found = ssme.files_found

        if self.ssme_files_found:
            print( 'separator.spatial.h5 files found' )
            # make fullchip mask
            ssme.make_fullchip_mask()
            ssme.plot_and_save_fullchip_mask()

            selected_metrics = {'snr'               :{'units':'',       'clims':(0,25),     'bin_scale':1, 'alias': 'SNR'},
                                'tau_e'             :{'units':'frames', 'clims':(0,20),     'bin_scale':1, 'alias': 'Tau E'},
                                'tau_b'             :{'units':'frames', 'clims':(0,20),     'bin_scale':1, 'alias': 'Tau B'},
                                'peak_sig'          :{'units':'counts', 'clims':(25,150),   'bin_scale':1, 'alias': 'Key Signal'},
                                'buff_clust_conf'   :{'units':'',       'clims':(0,1),      'bin_scale':50, 'alias': 'Buff. Clust. Conf.'},
                                'sig_clust_conf'    :{'units':'',       'clims':(0,1),      'bin_scale':50, 'alias': 'Sig. Clust. Conf.'},
                                }
            metric_names = selected_metrics.keys()

            for metric_name in metric_names:
                # populate associated values
                units       = selected_metrics[metric_name]['units']
                clims       = selected_metrics[metric_name]['clims']
                bin_scale   = selected_metrics[metric_name]['bin_scale']
                alias       = selected_metrics[metric_name]['alias']
                # backup case
                if alias is None:
                    alias = metric_name

                # create metric array
                ssme.build_array( metric_name )

                if ssme.metric_array is not None:
                    print( metric_name + ': array built' )
                    # plot then save plot of masked array (0's for masked region)
                    ssme.plot_and_save_masked_metric_array()
                    # define data with 0's for masked region
                    data = ssme.metric_array * ssme.fullchip_mask
                    # create multilane plot object
                    ar = MultilanePlot( data, '{}'.format( alias ), alias, units=units, clims=clims, bin_scale=bin_scale )
                    #NOTE: use metric_name for file saving
                    ar.plot_all( os.path.join( self.results_dir, 'multilane_plot_all_{}.png'.format( metric_name ) ) )
                    # iterate through active lanes.  process stats and figures
                    for (i, lane_name, active) in self.iterlanes():
                        if active:
                            if clims:
                                lower_lim = clims[0]
                            else:
                                lower_lim = 0
                            self.metrics[lane_name][metric_name] = self.get_lane_metrics( data, i, lower_lim=lower_lim )
                            #NOTE: use metric name for file saving
                            ar.plot_one( i, os.path.join( self.results_dir, 
                                                            'multilane_plot_lane_{i}_{name}.png'.format( i=i, name=metric_name ) ) )
                else:
                    print( metric_name + ': DID NOT BUILD ARRAY' )

        else:
            print( 'Aborting SSME analysis -- no separator.spatial.h5 files found.' )
            return None
           
#############################
#   CALCULATE TOTAL READS   #
#############################

    def calc_total_reads ( self ):
        ''' Stores total_reads as the sum of bcmatch_reads and nomatch_reads.  Uses np.nan_to_num to avoid nans in sum '''
        for (i, lane, active) in self.iterlanes():
            if active:
                if self.has_nomatch_bam:
                    self.metrics[lane]['total_reads'] = np.nan_to_num( self.metrics[lane]['bcmatch_reads'] ) + np.nan_to_num( self.metrics[lane]['below_thresh_bcmatch_reads'] ) + np.nan_to_num( self.metrics[lane]['nomatch_reads'] ) + np.nan_to_num( self.metrics[lane]['below_thresh_nomatch_reads'] ) 
                else:
                    self.metrics[lane]['total_reads'] = np.nan_to_num( self.metrics[lane]['bcmatch_reads'] ) + np.nan_to_num( self.metrics[lane]['below_thresh_bcmatch_reads'] ) 
                    
#########################################
#       GENERAL HELPER FUNCTIONS        #
#########################################

    def create_flow_based_readlength_scale( self ):
        ''' Creates flow depenedent scales to minimize white space '''
        if self.flows < 400:
            flowlims     = [ 20,150]
            vmr_flowlims = [  0, 20]
        elif (self.flows >= 400) and (self.flows < 750 ):
            flowlims     = [ 20,300]
            vmr_flowlims = [ 20, 40]
        elif (self.flows >= 750) and (self.flows < 1000 ):
            flowlims     = [ 20,500]
            vmr_flowlims = [ 40, 80]
        elif self.flows >= 1000:
            flowlims     = [ 20,800]
            vmr_flowlims = [ 40,100]
        else:
            flowlims     = [ 20,800]
            vmr_flowlims = [  0,100]
        return ( flowlims, vmr_flowlims, )

    def get_lane_slice( self , data , lane_number , lane_width=None ):
        """ 
        Takes a data array and returns data from only the lane of interest. 
        lane_number is 1-indexed.
        """
        if lane_width == None:
            lane_width = data.shape[1] / 4
        cs = slice( lane_width*(lane_number-1) , lane_width*(lane_number) )
        return data[:,cs]

    def get_lane_metrics( self , data , lane_number , lower_lim=0 , add_vmr=False , lane_width=None, bc_ind=None, bcmask=None, reads_weight=None ):
        """ Creates a dictionary of mean, q2, and std of a lane, masking out values above a lower_lim (unless is None). 
            **reads_weight is an INTEGER input that allows mini/micro-block processing for AQ20, etc values to be parsed into full lane values**
        """
        lane_data = self.get_lane_slice( data , lane_number , lane_width )
        if bc_ind is not None and bcmask is not None:
            bcmask_lane_slice = self.get_lane_slice( bcmask, lane_number, lane_width )
            temp = np.zeros( lane_data.shape )
            try:
                temp[ bcmask_lane_slice == bc_ind ] = lane_data[ bcmask_lane_slice == bc_ind ] 
            except IndexError:
                # This barcode does not have q20 reads so lane_data = temp = 0's
                pass
            lane_data = temp
        
        # filter by lower_lim if it exists
        if lower_lim == None:
            masked = lane_data
        else:
            masked = lane_data[ lane_data > lower_lim ]
        
        # calculate weighted metrics -- else regular
        if reads_weight is not None:
            weight = self.get_lane_slice( reads_weight, lane_number, lane_width )
            if lower_lim is not None:
                weight = weight[ lane_data > lower_lim ]
            tot = weight.sum()
            if tot > 0:
                # build new array with weights
                lw_temp = []
                for w, m in zip( weight, masked ):
                    lw_temp += [ m for i in range(int(w)) ]
                lw_temp = np.array( lw_temp )
                # calculate metrics from temp
                # NOTE:  a weighted std CANNOT be properly calculated from available data --> no std
                #           --we will store super pixel weighted std for future reference
                metrics = { 'mean': lw_temp.mean(), 'q2': np.median( lw_temp ), 'spr_pix_wstd' : lw_temp.std() }
            else:
                # use np.nan instead of None for consistent processing
                metrics = { 'mean': np.nan , 'q2': np.nan , 'std' : np.nan }
        else:
            metrics = { 'mean': masked.mean() , 'q2': np.median( masked ) , 'std' : masked.std() }
        
        if add_vmr and reads_weight is None:
            try:
                vmr = float( masked.var() ) / float( masked.mean() )
            except ZeroDivisionError:
                vmr = 0.
            metrics['vmr'] = vmr
            
        return metrics
        
    def iterlanes( self ):
        """ 
        Handy iterator to cycle through lane information and active states.  To be reused quite often. 
        returns lane number (1-4), lane name (for metric parsing), and  boolean for lane activity.
        """
        for i in range(1,5):
            name = 'lane_{}'.format(i)
            if self.has_explog:
                if self.explog_lanes_active[name]:
                    active = True
                else:
                    active = False
            else:
                active = self.lane[i-1]
            yield ( i , name , active )

    def multilane_wafermap( self , array , lane , metric , clims=None , transpose=True ):
        """ 
        Plots a series of wafermap spatial plots when given the full chip data array. 
        Does not create an image if the lane was inactive.
        """
        # Determine proper image size for data
        if transpose:
            array  = array.T
            img_name = '{}_{}_wafermap.png'.format( lane , metric )
        else:
            img_name = '{}_{}_wafermap_nonT.png'.format( lane , metric )
            
        w,h    = matplotlib.figure.figaspect( array )
        aspect = float(w)/float(h)
        dpi    = float( matplotlib.rcParams['figure.dpi'] )
        
        if transpose:
            w = (210. / dpi)
            h = w / aspect
            
        fig = plt.figure( figsize=(w,h) )
        ax  = fig.add_subplot(111)
        plt.axis  ( 'off' )
        if clims==None:
            plt.imshow( array , origin='lower' , aspect='equal' , interpolation='nearest' )
        else:
            plt.imshow( array , origin='lower' , aspect='equal' , interpolation='nearest' , clim=clims )
            
        # Set extent of image to exactly that of the chip image
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig( os.path.join( self.results_dir , img_name ) , bbox_inches=extent )
        plt.close  ( )

#########################
#   HTML PROCESSING     #
#########################
    
    # DISPLAY ON MAIN REPORT SCREEN
    def write_block_html( self ):
        ''' Writes html file to be output on main report screen '''
        html = os.path.join( self.results_dir , 'Lane_Diagnostics_block.html' ) 
        
        block  = textwrap.dedent('''\
                 <html><head><title>Lane Diagnostics</title></head>
                 <body>
                 <style type="text/css">table              {border-collapse: collapse;}</style>
                 <style type="text/css">tr:nth-child(even) {background-color: #DDD;}</style>
                 <style type="text/css">td                 {border: 1px solid black; text-align: center; }</style>
                 <style type="text/css">td.used            {background-color: #0f0;}</style>
                 <style type="text/css">td.unused          {background-color: #000;}</style>
        ''')
        
        if not self.has_chipcal:
            block += textwrap.dedent('''\
                <p><em>The calibration files were not found for this run.  Attempted to contact ChipDB and exit.</em></p>
                <hr>''')
            if self.has_explog:
                block += self.write_normalduck_table( )
            else:
                block += self.write_no_explog_msg()
            block += textwrap.dedent('''</body></html>''')
        elif self.is_multilane==False:
            block  = textwrap.dedent('''<p>This is a Fullchip Run, No Lane Diagnostics Needed</p>
                </body></html>''') 
        else:
            active_lanes = []
            for (lane_id, lane, active) in self.iterlanes():
                if active:
                    active_lanes.append( lane )
                    
            if len(active_lanes) == 1:
                l = active_lanes[0]
                imgs = ['multilane_plot_{}_loading.png'.format(l) ,
                        'multilane_plot_{}_q20_mrl.png'.format(l) ,
                        'multilane_plot_{}_q20_vmr_flow_limits.png'.format(l) ,
                        'multilane_plot_{}_rra_fixed.png'.format(l) ]
                if self.ssme_files_found:
                    imgs += [ 'multilane_plot_{}_peak_sig.png'.format(l),
                              'multilane_plot_{}_snr.png'.format(l) ]
            else:
                imgs = ['multilane_plot_all_loading.png',
                        'multilane_plot_all_q20_mrl.png',
                        'multilane_plot_all_q20_vmr_flow_limits.png' ,
                        'multilane_plot_all_rra_fixed.png']
                if self.ssme_files_found:
                    imgs += [ 'multilane_plot_all_peak_sig.png',
                              'multilane_plot_all_snr.png']
            if self.unused_and_active_lanes:
                block += textwrap.dedent( '''\
                        <font color='red'> <strong>WARNING: Lanes {} are active but not processed in this plugin analysis</strong></font>
                        <br>
                        '''.format( ','.join( str(x) for x in self.unused_and_active_lanes ) ) )
 
            block += textwrap.dedent('''\
            <p> <em>DISCLAIMER: <br>
                Aligned and raw-read accuracy metrics are from call to ionstats. <br>
                All other metrics are generated by SDAT algorithms. </em></p>
            ''')

            block += textwrap.dedent('''\
            <h3>Lane Diagnostics Report</h3>
            ''')
            for i, img in enumerate( imgs ):
                block += textwrap.dedent(''' <a href="{0}"><img src="{0}" width="20%" /></a> '''.format(img) )
            block += textwrap.dedent('''\
            <hr>
            ''')
            if self.has_explog and not self.csa: 
                block += self.write_normalduck_table( )
            elif self.has_explog:
                pass
            else:
                block += self.write_no_explog_msg()
            block += textwrap.dedent(''' </body></html> ''')
            
        with open( html , 'w' ) as f:
            f.write( block )

    def write_no_explog_msg( self ):
        ''' writes message saying explog was not found '''
        block = textwrap.dedent( '''\
                <p><em> The explog file was not found for this run.  Certain features have been skipped.</em></p>
                ''' )
        return block

    def write_normalduck_table( self ):
        def make_lane_td( used ):
            td = '<td class="{}"></td>'
            if used:
                return td.format( 'used' )
            else:
                return td.format( 'unused' )
                
        def link( addr ):
            return '<a href="{0}">{0}</a>'.format( addr )
            
        block = ''
        # Add table for normalduck data of all runs on this chip
        block += '<h3>Other Sequencing Experiments Using This Chip</h3>'
        if self.normalduck_data:
            block += '''<p><em>For future reference, this chip's run information can always be found at the following link: </em><a href="{0}">{0}</a></p><br>'''.format( self.normalduck_permalink )
            block += '<table width="100%"><tr><th>Date</th><th>Instrument</th><th>L1</th><th>L2</th><th>L3</th><th>L4</th><th>Thumbnail</th><th>Full Chip</th></tr>'
            for row in self.normalduck_data:
                r = '<td>{}</td><td>{}</td>'.format( row[6] , row[7] )
                for lane in row[0:4]:
                    r += make_lane_td( lane )

                r     += '<td>{}</td><td>{}</td>'.format( link( row[4] ) , link( row[5] ) )
                block += '<tr>{}</tr>'.format( r )
            block += ''' </table> '''
            block += '''<p><em>Note that recent runs may take up to a day to appear here and active lanes could take up to two days to be updated.  Please check the permalink above again in the near future for the most up to date information.</em></p>'''
        else:
            block += '<br><p><em>It appears that this is the first run using this chip.  No information was found on ChipDB.</em></p>'
        return block      
                
    def write_lane_html_files( self ):
        """ Creates a mirror of the main page, more or less, for single lanes. Table is omitted."""
        if self.has_chipcal and self.is_multilane:
            for (i,lane,active) in self.iterlanes():
                if active:
                    html  = os.path.join( self.results_dir , 'Lane_{}.html'.format( i ) )
                    
                    start = ''' 
                    <html>
                    <head>
                    <title>Lane Diagnostics</title>
                    <h1><center>Lane Diagnostics | Lane {} Analysis</center></h1>'''.format( i )
                    start += '''
                    </head>
                    <body>
                    <style type="text/css">tr.sh {background-color: #eee; }</style>
                    <hr>
                    '''
                    
                    start += self.write_image_array( lane=i )
                    start += '</body></html>'
                    
                    with open( html , 'w' ) as f:
                        f.write( start )
                        
    def write_html( self ):
        ''' Writes full html for report output . . . but only for multilane chips when calibration files exist'''
        if self.has_chipcal and self.is_multilane:
            html  = os.path.join( self.results_dir , 'Lane_Diagnostics.html' )
            
            start = ''' 
            <html>
            <head>
            <title>Lane Diagnostics</title>
            <h1><center>Lane Diagnostics</center></h1>
            </head>
            <body>
            <style type="text/css">tr.sh {background-color: #eee; }</style>
            <hr>
            '''
            # Add a summary table of metrics by lane.
            if self.thumbnail:
                # wd = Well divider
                wd    = 1000.
                units = 'K'
            else:
                wd    = 1000000.
                units = 'M'
            
            # Set up Metrics Types
            # Title, list of tuples: 
            #             ( Name , Metric subset <e.g. 'local_bcmatch_reads'> , metric , formatter , divider )
            mtypes  = [('BeadFind Metrics'          , '%'   , 
                        [ ( 'Loading'               , None  , 'loading',  '{:.1f}', None ) , 
                          ( 'FiltPass'              , None  , 'percent_filtpass', '{:.1f}', None ) ,
                          ( 'Useless (Empty+Badkey)', None  , 'percent_useless' , '{:.1f}', None ) ] ) ,
                       ('Q20 Read Length' , 'bp' , 
                        [ ( 'Median (Q2)' , 'q20_length' , 'q2'  ,     '{:.0f}' , None ) ,
                          ( 'Mean'        , 'q20_length' , 'mean',     '{:.0f}' , None ) ,
                          ( 'SD'          , 'q20_length' , 'std' ,     '{:.0f}' , None ) ,
                          ( 'VMR'         , 'q20_length' , 'vmr' ,     '{:.0f}' , None ) ] ) ,
                       ('AQ20 Read Length', 'bp' ,
                        [ ( 'Median (Q2)' , 'aq20_length', 'q2',     '{:.0f}' , None ) ,
                          ( 'Mean'        , 'aq20_length', 'mean',   '{:.0f}' , None ) , ] ) ,
                       ('Reads' , units , 
                        [ ( 'Total'       , None    , 'total_reads'                 ,   '{:.1f}' , wd ) ,
                          ( 'BC-Match'    , None    , 'bcmatch_reads'               ,   '{:.1f}' , wd ) ,
                          ( 'No-Match'    , None    , 'nomatch_reads'               ,   '{:.1f}' , wd ) ,
                          ( 'BT-BCm'      , None    , 'below_thresh_bcmatch_reads'  ,   '{:.1f}' , wd ) ,
                          ( 'BT-NOm'      , None    , 'below_thresh_nomatch_reads'  ,   '{:.1f}' , wd ) ,
                          ( ' --- '       , None    , ''                            ,   '{}'     , None ) ,
                          ( 'Q20*'        , None    , 'q20_reads'                   ,   '{:.1f}' , wd ) ,
                          ( 'AQ20'        , None    , 'aq20_reads'                  ,   '{:.1f}' , wd ) , ] ) ,

                       ('Raw Read Accuracy [Local]', '%' ,
                        [ ( 'Median (Q2)' , 'local_accuracy' , 'q2'  , '{:.2f}' , None ) ,
                          ( 'Mean'        , 'local_accuracy' , 'mean', '{:.2f}' , None ) ,
                          ( 'SD'          , 'local_accuracy' , 'std' , '{:.2f}' , None ) ] ),
                       ('Key Signal [separator_spatial]', 'counts',
                        [ ( 'Median (Q2)' , 'peak_sig' , 'q2'  , '{:.0f}' , None ) ,
                          ( 'Mean'        , 'peak_sig' , 'mean', '{:.0f}' , None ) ,
                          ( 'SD'          , 'peak_sig' , 'std' , '{:.0f}' , None ) ] ),
                       ('SNR [separator_spatial]', '',
                        [ ( 'Median (Q2)' , 'snr' , 'q2'  , '{:.1f}' , None ) ,
                          ( 'Mean'        , 'snr' , 'mean', '{:.1f}' , None ) ,
                          ( 'SD'          , 'snr' , 'std' , '{:.1f}' , None ) ] ),
                       ('Sig. Clust. Conf. [separator_spatial]', '',
                        [ ( 'Median (Q2)' , 'sig_clust_conf' , 'q2'  , '{:.2f}' , None ) ,
                          ( 'Mean'        , 'sig_clust_conf' , 'mean', '{:.2f}' , None ) ,
                          ( 'SD'          , 'sig_clust_conf' , 'std' , '{:.2f}' , None ) ] ),
                       ('Tau E [separator_spatial]', 'frames',
                        [ ( 'Median (Q2)' , 'tau_e' , 'q2'  , '{:.1f}' , None ) ,
                          ( 'Mean'        , 'tau_e' , 'mean', '{:.1f}' , None ) ,
                          ( 'SD'          , 'tau_e' , 'std' , '{:.1f}' , None ) ] ),
                       ('Tau B [separator_spatial]', 'frames',
                        [ ( 'Median (Q2)' , 'tau_b' , 'q2'  , '{:.1f}' , None ) ,
                          ( 'Mean'        , 'tau_b' , 'mean', '{:.1f}' , None ) ,
                          ( 'SD'          , 'tau_b' , 'std' , '{:.1f}' , None ) ] ),
                       ('Buff. Clust. Conf. [separator_spatial]', '',
                        [ ( 'Median (Q2)' , 'buff_clust_conf' , 'q2'  , '{:.2f}' , None ) ,
                          ( 'Mean'        , 'buff_clust_conf' , 'mean', '{:.2f}' , None ) ,
                          ( 'SD'          , 'buff_clust_conf' , 'std' , '{:.2f}' , None ) ] ),                 

                   ]
            
            blank_row = '<tr><td width="40%">&nbsp</td><td width="15%"></td><td width="15%"></td><td width="15%"></td><td width="15%"></td></tr>'
            suffix    = ''
            if not self.has_fc_bfmask:
                # We need to ignore beadfind metrics part of the table
                mtypes = mtypes[1:]
                suffix = '<p><em>Beadfind Mask File does not exist.  Skipping that part of this table.</em></p>'
                
            table = '<table border="0" cellspacing="0" width="100%">'
            for section , unit , rows in mtypes:
                table += '<tr class="sh"><td width="40%">{0}</td><td width="15%">({1})</td>'.format( section,unit )
                table += '<td width="15%"></td><td width="15%"></td><td width="15%"></td></tr>'
                table += '<td width="40%">Lane</td>'
                for i in [1,2,3,4]:
                    table += '<td width="15%">{}</td>'.format(i)
                table += '</tr>'
                table += blank_row
                print( '\n Making table of metrics' )
                for (name, subset, metric, fmt, divider) in rows:
                    print( 'name: {}'.format( name ) )
                    print( 'subset: {}'.format( subset ) )
                    print( 'metric: {}'.format( metric ) )
                    print( 'format: {}'.format( fmt ) )
                    print( 'divider: {}'.format( divider ) )
                    row = '<tr><td width="40%">{}</td>'.format( name )
                    for lane in ['lane_{}'.format(l) for l in [1,2,3,4]]:
                        if self.metrics[lane]['active']:
                            try:
                                if subset:
                                    val = self.metrics[lane][subset].get( metric , 0. )
                                elif metric == '':
                                    val = '-'
                                else:
                                    val = self.metrics[lane].get( metric , 0. )
                                if divider:
                                    if val == 0:
                                        pass
                                    else:
                                        val /= divider
                                if val == '-':
                                    value = val
                                else:
                                    value = fmt.format( val )
                            except KeyError:
                                value = '-'
                        else:
                            value = '-'
                        row += '<td width="15%">{}</td>'.format( value )
                    row   += '</tr>'
                    table += row
                # Insert a blank spacer row.
                table += blank_row
            table += '</table>'
            table += suffix
            
            # Add Metric Table and Read Length Histograms
            start += '<h2>Summary Metrics and Read Length Histograms</h2>'
            start += '<table border="0" cellspacing="0" width="100%"><tr>'
            start += '<tr><td width="40%">{}</td>'.format( table )
            for img in ['multilane_plot_bcmatch_rl_histograms.png','multilane_plot_q20_rl_histograms.png']:
                start += '<td width="30%"><a href="{0}"><img src="{0}" width="100%" /></a></td>'.format(img)
            start += '</tr></table><hr>'
            
            # Add array of full chip images
            start += self.write_image_array( )
            
            #for i in sorted(os.listdir(self.results_dir)):
            #    if '.png' in i:
            #        start+='''<a href="{}"><img src="{}" width="20%" /></a>'''.format(i,i)
            start += '''            
                </body></html>
                '''
            
            with open( html , 'w' ) as f:
                f.write( start )
        return None
        
    def write_image_array( self , lane=None ):
        """ 
        Creates a canned array format of images based on type and limits set.  
        If lane is None, fullchip images are used.
        """
        if lane in [1,2,3,4]:
            itype = 'lane_{}'.format( lane )
        else:
            itype = 'all'
            
        # Define images and order
        mp_path  = 'multilane_plot_{}_{}.png'
        bf_imgs  = [ mp_path.format(itype,i) for i in ['loading','filtpass','useless'] ]
        q20mrl   = [ mp_path.format(itype,i) for i in ['q20_mrl','q20_mrl_flow_limits','q20_mrl_centered'] ]
        q20vmr   = [ mp_path.format(itype,i) for i in ['q20_vmr','q20_vmr_flow_limits','q20_vmr_fixed_scale'] ]
        reads    = [ mp_path.format(itype,i) for i in ['bcmatch_read_density','q20_read_density','q20_read_bead_conversions'] ]
        rra      = [ mp_path.format(itype,i) for i in ['rra','rra_fixed'] ]
        badppf   = [ mp_path.format(itype,i) for i in ['badppf','badppf_RESCALED'] ]
        sepspa1  = [ mp_path.format(itype,i) for i in ['peak_sig', 'snr', 'sig_clust_conf'] ]
        sepspa2  = [ mp_path.format(itype,i) for i in ['tau_e', 'tau_b', 'buff_clust_conf'] ]
        rra.append( '' )
        
        # Prepare html text
        sections = ['Beadfind Analysis','Q20 Read Length','Q20 Variance-Mean-Ratio','Read Data','Raw Read Accuracy', 'Badppf']
        images   = [ bf_imgs , q20mrl , q20vmr , reads , rra, badppf ]
        labels   = [ [] ,
                     ['Autoscaled Limits','Flow-based Limits','Mean +/- 50 bp Limits'] ,
                     ['Autoscaled Limits','Flow-based Limits','Fixed Scale Limits'] ,
                     [] ,
                     ['Autoscaled Limits','Fixed Limits',''],
                     ['Fixed Scale Limits', 'Mean-based Limits', '' ] ,
                     ]
        if self.ssme_files_found:
            sections += ['Separator Spatial Metrics - 1', 'Separator Spatial Metrics - 2']
            images += [sepspa1, sepspa2]
            labels += [[],[],]

        html     = ''
        
        for header,labels, imgs in zip( sections , labels , images ):
            html += '<h2>{}</h2>'.format( header )
            html += '<table border="0" cellspacing="0" width="100%">'
            if labels != []:
                html += '<tr>'
                for l in labels:
                    html += '<th><center>{}</center></th>'.format( l )
                html += '</tr>'
            html += '<tr>'
            for image in imgs:
                if image == '':
                    html += '<td width="25%"></td>'
                else:
                    html += '<td width="25%"><a href="{0}"><img src="{0}" width="100%" /></a></td>'.format(image)
            html += '</tr></table>'
            html += '<hr>'
        
        return html

#################################
#       SUPPORTING CLASSES      #
#################################

class BlockReshape( object ):
    """ Blocksize should be defined using chipcal.chiptype.miniR and .miniC """
    def __init__( self , data , blocksize , mask=None ):
        ''' Important note: mask is for pixels what we want to ignore!  Like pinned, e.g. '''
        self.set_data( data, blocksize, mask )
        
    def block_reshape( self , data , blocksize ):
        ''' Does series of reshapes and transpositions to get into miniblocks. '''
        rows , cols = data.shape
        blockR = rows / blocksize[0]
        blockC = cols / blocksize[1]
        return data.reshape(rows , blockC , -1 ).transpose((1,0,2)).reshape(blockC,blockR,-1).transpose((1,0,2))
    
    def get_ma( self ):
        if hasattr( self , 'data' ):
            reshaped = self.block_reshape( self.data , self.blocksize )
            remasked = np.array( self.block_reshape( self.mask , self.blocksize ) , bool )
            return ma.masked_array( reshaped , remasked )
        else:
            print( 'Error!  Can not reshape and mask data until set_data has been called!' )
            return None
        
    def set_data( self , data , blocksize , mask ):
        ''' Important note: mask is for pixels what we want to ignore!  Like pinned, e.g. '''
        self.data      = np.array( data , dtype=float )
        self.blocksize = blocksize
        
        # Again, masked pixels are ones we want to ignore.   If using all, we need np.zeros.
        try:
            if not mask:
                self.mask = np.zeros( self.data.shape )
            else:
                self.mask = mask
        except ValueError:
            if not mask.any():
                self.mask = np.zeros( self.data.shape )
            else:
                self.mask = mask
                
        self.reshaped = self.get_ma( )
        
    def get_mean( self ):
        if hasattr( self , 'reshaped' ):
            return self.reshaped.mean(2).data
        
    def get_std( self ):
        if hasattr( self , 'reshaped' ):
            return self.reshaped.std(2).data
        
    def get_sum( self ):
        if hasattr( self , 'reshaped' ):
            return self.reshaped.sum(2).data
        
    def get_var( self ):
        if hasattr( self , 'reshaped' ):
            return self.reshaped.var(2).data
        
    def get_vmr( self ):
        ''' 
        Calculates variance-mean-ratio . . . sigma^2 / mean 
        Note that here we are going to define superpixels with zero means to be 0 VMR, not np.nan.
        '''
        if hasattr( self , 'reshaped' ):
            means      = self.get_mean()
            msk        = (means == 0)
            means[msk] = 1
            variance   = self.get_var()
            vmr        = variance / means
            vmr[msk]   = 0
            return vmr
        
class RRA:
    origin_re = re.compile( 'origin=(?P<col>[0-9]+),(?P<row>[0-9]+)-' )
    
    def __init__( self , fname , chipRC , blocksize ):
        self.fname     = fname
        self.chipRC    = chipRC
        self.blocksize = blocksize
        self.dims      = [ chipRC[0]/blocksize[0] , chipRC[1]/blocksize[1] ]
        self.rra       = np.zeros( self.dims , np.float64 )
        
        self.load( self.fname )
        
    def load( self , fname ):
        if os.path.exists( fname ):
            self.data = h5py.File( fname , 'r' )
            for region in self.data['per_region']:
                r,c           = self.get_index( region )
                self.rra[r,c] = self.calc_rra ( self.data['per_region'][region] )
        else:
            print('Error!  File does not exist.')
            
    def get_coord( self , region_name ):
        if self.origin_re.match( region_name ):
            col = int( self.origin_re.match( region_name ).group( 'col' ) )
            row = int( self.origin_re.match( region_name ).group( 'row' ) )
            return [row , col]
        else:
            return []
            
    def get_index( self , region_name ):
        coords = self.get_coord( region_name )
        if coords:
            row = coords[0] / self.blocksize[0]
            col = coords[1] / self.blocksize[1]
            return (row,col)
        else:
            return ()
        
    @staticmethod
    def calc_rra( group ):
        """ f['per_region'][region] is a 7 member group. """
        rra     = 0.
        errors  = float( group['n_err'][:][0] )
        aligned = float( group['n_aligned'][:][0] )
        if aligned > 0:
            rra = 100. * (1.0 - (errors / aligned))
            
        return rra

class AQ:
    region_re = re.compile( '\((?P<col>[0-9]+),(?P<row>[0-9]+)\)' )
    
    def __init__( self , fname , chipRC , blocksize ):
        self.fname          = fname
        self.chipRC         = chipRC
        self.blocksize      = blocksize
        self.dims           = [ chipRC[0]/blocksize[0] , chipRC[1]/blocksize[1] ]

        self.aq20_reads     = np.zeros( self.dims , np.float64 )
        self.aq20_length    = np.zeros( self.dims , np.float64 )
        self.aq07_reads     = np.zeros( self.dims , np.float64 )
        self.aq07_length    = np.zeros( self.dims , np.float64 )
        
        self.load( self.fname )
        
    def load( self , fname ):
        ''' Data comes from ionastats_alignment.json '''
        if os.path.exists( fname ):
            with open( fname, 'r' ) as file:
                self.data = json.loads( file.read() )
            for region, values in self.data['Regional'].items():
                r,c             = self.get_index( region )
                # Available regional quality keys are AQ7,10,17,20,30,47
                #   with subkeys max_read_length, mean_read_length (int), num_bases, num_reads
                # AQ20
                reads, mrl = self.calc_aq_vals( values, 'AQ20' )
                self.aq20_reads[r,c]    = reads
                self.aq20_length[r,c]   = mrl
                # AQ7
                reads, mrl = self.calc_aq_vals( values, 'AQ7' )
                self.aq07_reads[r,c]    = reads
                self.aq07_length[r,c]   = mrl
        else:
            print('Error!  File does not exist.')
            
    def get_coord( self , region_name ):
        if self.region_re.match( region_name ):
            col = int( self.region_re.match( region_name ).group( 'col' ) )
            row = int( self.region_re.match( region_name ).group( 'row' ) )
            return [row , col]
        else:
            return []
            
    def get_index( self , region_name ):
        coords = self.get_coord( region_name )
        if coords:
            row = coords[0] / self.blocksize[0]
            col = coords[1] / self.blocksize[1]
            return (row,col)
        else:
            return ()
        
    @staticmethod
    def calc_aq_vals( values, quality ):
        ''' Extract bases and reads. Calculate mean read length (mrl) '''
        bases   = values[quality]['num_bases'] 
        reads   = values[quality]['num_reads']
        if reads>0: mrl = bases/reads
        else:       mrl = 0
        return (reads, mrl,)


if __name__ == "__main__":
    PluginCLI()
