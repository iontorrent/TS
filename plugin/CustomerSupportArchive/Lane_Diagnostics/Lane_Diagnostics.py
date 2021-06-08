#!/usr/bin/env python
# Copyright (C) 2018 Ion Torrent Systems, Inc. All Rights Reserved
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, inspect
import numpy as np
import numpy.ma as ma
import json
import time
import re
import urllib
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.join(parentdir, 'autoCal'))

from ion.plugin import *
from tools import chipcal, chiptype, explog
import subprocess
import pybam
import bfmask

# Import tools for reading h5 files (ionstats output)
import h5py

# Import multilane plotting tools
from multilane_plot import MultilanePlot

# importe separator.spatial.h5 processing
from tools.SeparatorSpatialProcessing import SeparatorSpatialMetricExtractor as SSME

# For pybam -- so that data fits into int16 or uint16 for non-barcoded reads
NBC = 2**15-1

class Lane_Diagnostics( IonPlugin ):
    ''' 
    Plugin to assist in debugging and understanding multilane sequencing runs and their chips
    
    Now managed by Phil Waggoner
    
    Latest updates | Fixed bugs so that the plugin now works on Valkryie TS.
                   | Updated tools to deal with GX5 chips.
                   | Added SeparatorSpatial Metrics
    '''
    version       = "1.4.7"
    allow_autorun = True
    
    runtypes      = [ RunType.THUMB , RunType.FULLCHIP ]
    
    def launch( self ):
        # Get metadata
        print('Plugin Start')
        self.initialize( )

        self.metrics.update( { 'ChipType' : self.chip_type , 'lane_1':{}, 'lane_2':{}, 'lane_3':{}, 'lane_4':{} } )
        self.find_die_area( )
        #self.get_plugin_json( )
        #self.get_barcodes()
        #self.explog = Explog( self.raw_data_dir )
        
        # Print debug output
        print('\n')
        print('Plugin Dir: {}'.format(   self.plugin_dir   ))
        print('Raw Data Dir: {}'.format( self.raw_data_dir ))
        print('Analysis Dir: {}'.format( self.analysis_dir ))
        print('Results Dir: {}'.format(  self.results_dir  ))
        print('Run Flows: {}'.format(    self.flows        ))
        print('Chiptype: {}'.format(     self.chip_type    ))
        print('\n')
        
        # Initial lane analysis and communication with chipdb.ite
        #self.call_normalduck     ( )
        
        # Try to determine gain. Shortcut to exit if calibration gain file doesn't exist.
        self.determine_lane      ( )
        
        if self.cal_exists and self.is_multilane:
            # Analyze multilane flowcell placement
            self.analyze_fca( )
            
            # Loading and per lane bfmask metrics
        
            self.analyze_lane_loading( )
        
        
            # Create heatmaps, means, histograms, and conversion metrics -- per lane.
            self.lane_aligned_results_pybam ( )
            
            # New section for RRA?
            self.analyze_rra( )

            # Analyze separator.spatial.h5 files
            self.analyze_separator_spatial()
            
            # Create outputs
            self.save_json       ( )
    
        else:
            print( 'Aborting analysis -- not multilane OR cal does not exist' )
            
        ##################################################
        # Current Progress Marker
        ##################################################
        
        self.write_block_html     ( )
        self.write_html           ( )
        self.write_lane_html_files( )
        
        print( 'Plugin complete.' )
        
        sys.exit(0)
        
    def analyze_fca( self ):
        """ Uses chip gain to measure accuracy of flowcell placement.  Leverages code from tools/chipcal. """
        fca_metrics = self.cc.measure_flowcell_placement( outdir=self.results_dir )

        # Update metrics.
        if fca_metrics:
            for k in fca_metrics:
                self.metrics[k]['fca'] = fca_metrics[k]
                
    def analyze_rra( self ):
        """ Raw Read Accuracy Analysis.  This code leveraged from RRA_Spatial Plugin - thanks Charlene!"""
        # First we have to run ionstats again.  Let's build the command
        print( 'Starting Raw Read Accuracy Analysis . . .' )
        h5_file = os.path.join( self.results_dir , 'ionstats_error_summary.h5' )
        cmd = 'ionstats alignment --chip-origin 0,0 '
        if self.thumbnail:
            cmd += '--chip-dim 1200,800 --subregion-dim 10,10 '
        else:
            cmd += '--chip-dim {0.chipC},{0.chipR} --subregion-dim {0.miniC},{0.miniR} '.format(self.cc.chiptype)
            
        cmd += '--evaluate-hp --skip-rg-suffix .nomatch --n-flow {} --max-subregion-hp 6 '.format(self.flows)
        cmd += '--output-h5 {} '.format( h5_file )
        cmd += '-i {} '.format( ','.join( [ bc['bam_filepath'] for bc in self.barcodes.values() ] ) )
        
        print( 'ionstats call:' )
        print( cmd )
        
        start_time = time.time()
        subprocess.call( cmd , shell=True )
        print( 'ionstats completed in {:0.1f} seconds.'.format( (time.time()-start_time) ) )
        
        # Analyze the file and make some plots -- first arbitrary clims, then fixed?.
        ct = self.cc.chiptype
        if os.path.exists( h5_file ):
            if self.thumbnail:
                self.rra = RRA( h5_file , [800 , 1200] , [10 , 10] )
            else:
                self.rra = RRA( h5_file , [ct.chipR , ct.chipC] , [ct.miniR , ct.miniC] )
            
            rra_plot = MultilanePlot( self.rra.rra , 'Local Raw Read Accuracy' , 'RRA' , '%' , clims=[95,100] , 
                                      bin_scale=25 )
            rra_plot.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_rra.png' ) )
            for (lane_id, lane, active) in self.iterlanes():
                if active:
                    rra_plot.plot_one( lane_id , os.path.join( self.results_dir ,
                                                               'multilane_plot_lane_{}_rra.png'.format(lane_id) ) )
                    self.metrics[lane]['local_accuracy'] = self.get_lane_metrics( self.rra.rra , lane_id , 0 )
                    
            rra_plot.bin_scale = 50
            rra_plot.update_clims( [98,100] )
            rra_plot.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_rra_fixed.png' ) )
            for (lane_id, lane, active) in self.iterlanes():
                if active:
                    rra_plot.plot_one( lane_id , os.path.join( self.results_dir ,
                                                               'multilane_plot_lane_{}_rra_fixed.png'.format(lane_id) ) )
            print( ' . . . Raw Read Accuracy Analysis Completed.' )
        else:
            print( ' . . . Error with running ionstats.  Raw Read Accuracy analysis failed.\n' )
        return None
        
    def analyze_lane_loading( self ):
        '''Analyze beadfind metrics. Also, find if there are bubbles in the ignore bin'''
        self.lane_median_loading = [0,0,0,0]
        fc_bf_mask               = os.path.join( self.sigproc_dir , 'analysis.bfmask.bin' )
        
        if not os.path.exists( fc_bf_mask ):
            print( 'It appears that analysis.bfmask.bin has already been deleted, or was never created.  Skipping!')
            self.has_bfmask = False
            
        elif self.is_multilane:
            print('Starting lane loading analysis . . .')
            self.has_bfmask = True
            
            # Import bfmask.BeadfindMask and run with its canned analysis. 
            self.bf = bfmask.BeadfindMask( fc_bf_mask )
            
            # Do lane specific loading heatmaps
            self.bf.select_mask( 'bead' )
            if self.thumbnail:
                x = self.bf.block_reshape( self.bf.current_mask , [10,10] )
            else:
                x = self.bf.block_reshape( self.bf.current_mask , [self.bf.chiptype.miniR,self.bf.chiptype.miniC] )
            for idx,i in enumerate(self.lane):
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
                fname  = '%s_lane_%s_heatmap.png' % ( self.bf.current_mask_name,str(idx+1) )
                plt.savefig( os.path.join( self.results_dir , fname ) , bbox_inches=extent )
                plt.close  ( )
                
                self.lane_median_loading[idx]                          = np.median(array)
                self.metrics['lane_{}'.format( idx+1 )]['p50_loading'] = self.lane_median_loading[idx]
                
            # Create basic plot of median loading per lane.
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
            
            # Start detailed metric analysis here.
            
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
            self.bf.select_mask( 'bead' )
            beads        = self.bf.current_mask
            bead_loading = 100. * BlockReshape ( beads , self.blocksize ).get_mean()
            lane_loading = MultilanePlot( bead_loading , 'Loading Density', 'Loading', '%', clims=[0,100] )
            lane_loading.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_loading.png' ) )
            for (lane_id, lane, active) in self.iterlanes():
                if active:
                    lane_loading.plot_one( lane_id, os.path.join( self.results_dir , 'multilane_plot_lane_{}_loading.png'.format(lane_id) ) )
                    bead_array   = self.get_lane_slice( beads , lane_id )
                    active_wells = self.get_lane_slice( unignored_active , lane_id )
                    
                    lane_beads   = bead_array[ active_wells ].sum()
                    loading      = 100. * float( lane_beads ) / float( active_wells.sum() )
                    self.metrics[lane]['lane_beads'] = int( lane_beads )
                    self.metrics[lane]['loading']    = loading
                    
            # Analyze Filtpass Mask
            self.bf.select_mask( 'filtpass' )
            filtpass   = self.bf.current_mask
            fp_density = 100. * BlockReshape ( filtpass , self.blocksize ).get_mean()
            lane_fpd   = MultilanePlot( fp_density , 'Filtpass Density', 'Wells Passing Filters', '%',clims=[0,100])
            lane_fpd.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_filtpass.png' ) )
            for (lane_id, lane, active) in self.iterlanes():
                if active:
                    lane_fpd.plot_one( lane_id , os.path.join( self.results_dir , 'multilane_plot_lane_{}_filtpass.png'.format(lane_id) ) )
                    filtpass_array = self.get_lane_slice( filtpass , lane_id )
                    active_wells   = self.get_lane_slice( unignored_active , lane_id )
                    
                    lane_pass    = filtpass_array[ active_wells ].sum()
                    self.metrics[lane]['filtpass_beads']   = int( lane_pass )
                    self.metrics[lane]['percent_filtpass'] = 100. * float( lane_pass ) / float( self.metrics[lane]['lane_beads'] )
                    
            # Analyze Useless Wells
            self.bf.select_mask( 'useless' )
            useless         = self.bf.current_mask
            if self.thumbnail:
                useless_density = 100. * BlockReshape ( useless , self.blocksize , unignored_active ).get_mean()
            else:
                useless_density = 100. * BlockReshape ( useless , self.blocksize , self.cc.refpix ).get_mean()
            lane_ud         = MultilanePlot( useless_density , 'Useless Density', 'Useless Wells','%',clims=[0,100] )
            lane_ud.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_useless.png' ) )
            for (lane_id, lane, active) in self.iterlanes():
                if active:
                    lane_ud.plot_one( lane_id , os.path.join( self.results_dir , 'multilane_plot_lane_{}_useless.png'.format(lane_id) ))
                    useless_wells = self.get_lane_slice( useless , lane_id )
                    active_wells  = self.get_lane_slice( unignored_active , lane_id )
                    
                    uwc = useless_wells[ active_wells ].sum()
                    self.metrics[lane]['useless_wells']   = int( uwc )
                    self.metrics[lane]['percent_useless'] = 100. * float( uwc ) / float( active_wells.sum() )
                    
            print ( '. . . Loading analysis complete.' )
            
        else:
            print ('Skipping loading analysis, as this was not detected to be a multilane chip!' )
            
        return None
       
    def analyze_separator_spatial( self ):
        '''Analyze separator.spatial.h5 files for multilane metrics.'''
        if self.is_multilane:
            print ( 'Starting analysis of separator.spatial.h5 files. . .' )
        else:
            print ( 'Aborting analysis -- this is not a multilane chip.' )
            return None

        ssme = SSME( self.sigproc_dir )

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
            print( 'Aborting analysis -- no separator.spatial.h5 files found.' )
            return None

    def call_normalduck( self ):
        """ Contacts normalduck database for information on this chip and other relevant runs. """
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
        
        #Initialize the lane information will null values for error reporting
        self.lane      = [None, None, None, None]
        self.lane_chip = None
        
        # Load gain and determine reference pixels using a gain cutoff. *From chipDiagnostics
        try:
            self.cc = chipcal.ChipCal( self.caldir , self.chip_type , self.results_dir )
            self.cc.load_gain  ( )
            self.cc.find_refpix( )
            dirName = os.path.basename(self.analysis_dir)
            active_lanes = dirName[8:].split('_')
            active_lanes = map(int, active_lanes)
            self.cc.determine_lane( gain_lane_threshold, 50, active_lanes= active_lanes)
            self.cal_exists = True
        except Exception, e:
            print('Failed to upload to ftp: '+ str(e))

            print( 'Unable to find calibration gain file.  It is likely that the file has been deleted.' )
            self.cal_exists   = False
            self.is_multilane = False
            self.lane_chip    = False
            return None
        
        self.lane_chip    = bool(self.cc.is_multilane)
        self.is_multilane = bool(self.cc.is_multilane) # For similarity with ChipDB
        
        # Address what to do if this is thumbnail.
        if self.thumbnail:
            self.blocksize = [10,10]
        else:
            self.blocksize = [self.cc.chiptype.miniR , self.cc.chiptype.miniC]
        
        # Previously, we counted number of fluidically addressable wells for each lane here
        # ChipCal.determine_lane() now does this.
        for i in range(1,5):
            lane                         = 'lane_{}'.format(i)
            is_active                    = bool( getattr(self.cc , lane) )
            self.lane[i-1]               = is_active
            self.metrics[lane]['active'] = self.lane[i-1]
            
            # for access from class object
            setattr( self , 'lane_{}_active'.format( i ) , self.lane[i-1] ) 
            
            # For backwards compatability.
            k               = 'fluidically_addressable_wells_count_lane{}'.format(str(i))
            self.metrics[k] = float( self.cc.lane_metrics[lane].get( 'addressable_wells', 0 ) )
            self.metrics[lane]['fluidically_addressable_wells'] = self.metrics[k]
            
        # Add other metrics
        others = { 'Which_Lane' : self.lane, 'If_Multilane' : self.lane_chip }
        self.metrics.update( others )
        
        # Plot for debugging purposes.
        cols = int( self.explog.metrics['Columns'] )
        data = ma.masked_array( self.cc.gain , self.cc.gain==0. ).mean( axis=0 ).data
        data[ np.isnan( data ) ] = 0

        plt.figure  ( )
        plt.plot    ( range( cols ) , data )
        plt.ylabel  ( 'Average_Gain (mV/V)' )
        plt.xlim    ( 0 , cols )
        plt.ylim    ( 0 , 1200 )
        plt.axhline ( y=gain_lane_threshold, color='r', linestyle='-')
        plt.title   ( 'Average Gain [masking wells with 0 gain]' )
        plt.grid    ( )
        plt.savefig ( os.path.join( self.results_dir , 'Lane_gain_avg.png' ) )
        plt.close   ( )

        print(' . . . Active lanes determined.\n' )
        return None
    
    def get_lane_slice( self , data , lane_number , lane_width=None ):
        """ 
        Takes a data array and returns data from only the lane of interest. 
        lane_number is 1-indexed.
        """
        if lane_width == None:
            lane_width = data.shape[1] / 4
        cs = slice( lane_width*(lane_number-1) , lane_width*(lane_number) )
        return data[:,cs]

    def get_lane_metrics( self , data , lane_number , lower_lim=0 , add_vmr=False , lane_width=None ):
        """ Creates a dictionary of mean, q2, and std of a lane, masking out values above a lower_lim (unless is None). """
        lane_data = self.get_lane_slice( data , lane_number , lane_width )
        if lower_lim == None:
            masked = lane_data
        else:
            masked = lane_data[ lane_data > lower_lim ]
            
        metrics = { 'mean': masked.mean() , 'q2': np.median( masked ) , 'std' : masked.std() }
        
        if add_vmr:
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
            yield ( i , name , getattr( self , '{}_active'.format( name ) , False ) )
            
    def lane_aligned_results_pybam( self ):
        '''get lane specific aligned results using pybam. 
        Call Ionstats to generate region specific alignes seuqencing metrics'''
        #Initialize outputs
        self.lane_block_aq20_results={1:{},2:{},3:{},4:{}}   #TO DO: Fill this in
        self.lane_aq20_results      ={1:{},2:{},3:{},4:{}}
        
        if self.is_multilane:
            print ( 'Starting Lane Aligned Results . . .' )
            print ( 'Full chip Detected, starting pybam' )
        else:
            print ( 'Aborting bam file analysis -- this is not a multilane chip.' )
            return None
        
        # Eventually need to get rid of this try statement and be more intelligent about error handling.
        quality = [ -1, 0, 7, 20 ] # -1: barcode mask
                                   #  0: read length
                                   #  7: q7 length
                                   # 20: q20 length
        if self.thumbnail == True:
            shape = [800, 1200, len(quality)]
        else:
            shape = [self.cc.rows, self.cc.cols, len(quality)]
            
        qualmat = np.zeros( shape, dtype=np.uint16 ) # pybam requires uint16
        
        # If barcodes are present, then do not merge the non-barcode reads
        # If only non-barcoded reads are present, then read those
        # Check if non-barcoded reads are present
        nobc = NBC in [ bc.get( 'barcode_index', NBC ) for bc in self.barcodes.values() ]
        # Check if barcoded reads are present
        bc   = any( [ bc.get( 'barcode_index', NBC ) != NBC for bc in self.barcodes.values() ] )
        merge_nbc = nobc and not bc
        
        for bc in self.barcodes.values():
            filename = bc['bam_filepath']
            print( 'procesing new bamfile:' )
            print( filename )
            fill     = bc.get( 'barcode_index', NBC )
            if ( fill == NBC ) and ( not merge_nbc ):
                print( 'Skipping merge of non-barcoded reads since barcoded reads are present' )
                continue
            pybam.loadquals( filename, quality, qualmat, fill=fill )
            
        print( 'qualmat shape', qualmat.shape )
        
        # TODO: For some reason, qualmat looses data.  For now, just pad it
        qualmat_t = qualmat
        qualmat   = np.zeros( shape, dtype=np.uint16 ) # pybam requires uint16

        mrow = min( qualmat.shape[0], qualmat_t.shape[0] )
        mcol = min( qualmat.shape[1], qualmat_t.shape[1] )
        qualmat[0:mrow,0:mcol] = qualmat_t[0:mrow,0:mcol]

        #qualmat[:,:,0].astype( np.int16 ).tofile( 'barcode_mask.dat' )
        self.aligned_length = qualmat[:,:,1].astype( np.int16 )
        self.reads          = qualmat[:,:,1].astype( np.bool  )
        self.q7Len          = qualmat[:,:,2].astype( np.int16 )
        self.q20Len         = qualmat[:,:,3].astype( np.int16 )

        # save q20Len for posterity
        for (i, lane, active) in self.iterlanes():
            if active:
                lane_q20 = self.get_lane_slice( self.q20Len , i )
                lane_q20.tofile( os.path.join( self.results_dir , '{}_q20_length.dat'.format( lane ) ) )
                
        with open( os.path.join( self.results_dir , 'lane_array_size.txt' ) , 'w' ) as f:
            f.write( '{0}\t{1}'.format( *lane_q20.shape ) )
        
        # For barcodes, look at just the locations and lengths, but keep them separate
        qualmat = qualmat[:,:,:2]
        qualmat[:,:,1] = 0
        quality = [ -1, 0 ] # -1: barcode mask
                            #  0: read length
        if not( merge_nbc ):
            # make a mask of non-barcoded reads
            for bc in self.barcodes.values():
                fill     = bc.get( 'barcode_index', NBC )
                if ( fill == NBC ):
                    filename = bc['bam_filepath']
                    print( 'procesing new bamfile:' )
                    print( filename )
                    pybam.loadquals( filename, quality, qualmat, fill=fill )

            qualmat[:,:,1].astype( np.int16 ).tofile( os.path.join( self.results_dir ,'nbc_aligned_length.dat') )
            qualmat[:,:,1].astype( np.bool ).tofile( os.path.join( self.results_dir ,'nbc_totalreads.dat') )
            
        self.bcmask = qualmat[:,:,0].astype( np.int16 )
        
        # Plots ARE DONE
        # length -- read length histogram per lane.
        # reads  -- heatmap of aligned and q20 reads
        # q20    -- mean read length heatmap, VMR, q20_reads/reads conversion, RL histogram?
        
        # Metrics per lane NEED DEFINED
        # length -- mean, std
        # reads  -- aligned reads, % (aligned reads / active wells)
        # q7     -- q7 reads, length:[mean, median, std, iqr, global_vmr, local_vmr:[mean, std]]
        # q20    -- q20reads, length:[mean, median, std, iqr, global_vmr, local_vmr:[mean, std]]
        
        # global comparisons to beads?  q20 reads / beads, aligned_reads / beads ?
        
        ##################################################
        # Aligned Length
        ##################################################
        
        # Create a flow-based readlength scale
        if self.flows < 400:
            flowlims     = [ 20,150]
            vmr_flowlims = [  0, 20]
        elif (self.flows >= 400) and (self.flows < 750 ):
            flowlims     = [ 20,300]
            vmr_flowlims = [ 20, 40]
        elif (self.flows >= 750) and (self.flows < 1000 ):
            flowlims     = [ 20,400]
            vmr_flowlims = [ 40, 80]
        elif self.flows >= 1000:
            flowlims     = [ 20,800]
            vmr_flowlims = [ 40,100]
        else:
            flowlims     = [ 20,800]
            vmr_flowlims = [  0,100]
            
        # non-local-averaged data (pure histograms of individual well read lengths)
        rl = np.zeros( self.aligned_length.shape )
        rl[ self.aligned_length >= 20 ] = self.aligned_length[ self.aligned_length >= 20 ]
        rl_hists = MultilanePlot( rl , 'Aligned Read Length' , 'Read Length' , units='bp' , clims=flowlims )
        rl_hists.plot_histograms_only( os.path.join( self.results_dir, 'multilane_plot_aligned_rl_histograms.png' ) )
 
        for (i, lane, active) in self.iterlanes():
            if active:
                rl_hists.plot_single_lane_histogram_only( i, os.path.join( self.results_dir, 'multilane_plot_lane_{}_aligned_rl_histogram.png'.format(i) ) )

        del rl
        
        ##################################################
        # Reads (aligned and Q20)
        ##################################################
        
        aligned_reads = 100. * BlockReshape( self.reads , self.blocksize ).get_mean()
        ar = MultilanePlot( aligned_reads , 'Aligned Read Density' , 'Wells with Aligned Reads' , '%' , 
                            clims=None , bin_scale=2 )
        ar.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_aligned_read_density.png' ) )
        
        for (i, lane, active) in self.iterlanes():
            if active:
                ar.plot_one( i , os.path.join( self.results_dir , 
                                               'multilane_plot_lane_{}_aligned_read_density.png'.format(i) ) )
                self.metrics[lane]['aligned_reads']       = self.get_lane_slice  ( self.reads , i ).sum()
                self.metrics[lane]['local_aligned_reads'] = self.get_lane_metrics( aligned_reads , i , 0 )
                
        q20_read_density = 100. * BlockReshape( (self.q20Len > 20) , self.blocksize ).get_mean()
        q20_reads = MultilanePlot( q20_read_density , 'Q20 Read Density' , 'Wells with Q20 Reads' , '%' , 
                                   clims=None , bin_scale=2)
        q20_reads.plot_all( os.path.join( self.results_dir , 'multilane_plot_all_q20_read_density.png' ) )
        for (i, lane, active) in self.iterlanes():
            if active:
                q20_reads.plot_one( i , os.path.join( self.results_dir , 
                                                      'multilane_plot_lane_{}_q20_read_density.png'.format(i) ) )
                self.metrics[lane]['q20_reads']       = self.get_lane_slice  ( (self.q20Len > 20), i ).sum()
                self.metrics[lane]['local_q20_reads'] = self.get_lane_metrics( q20_read_density  , i , 0 )
                
        # This section only useful if we have the beadfind mask file.
        if self.has_bfmask:
            # loaded beads conversion into q20 reads.
            self.bf.select_mask( 'bead' )
            beads             = self.bf.current_mask
            bead_count        = BlockReshape( beads , self.blocksize ).get_sum()
            q20_read_count    = BlockReshape( (self.q20Len > 20 ) , self.blocksize ).get_sum()
            
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
        self.metrics['q20_mrl'] = ( self.q20Len[ self.q20Len > 20 ] ).mean()
        
        q20            = BlockReshape( self.q20Len , self.blocksize , self.q20Len < 20 )
        local_q20_mean = q20.get_mean( )
        local_q20_vmr  = q20.get_vmr ( )
        
        # Q20 Multilane Plots
        # non-local-averaged data (pure histograms of individual well read lengths)
        masked_q20 = np.zeros( self.q20Len.shape , np.int16 )
        masked_q20[ self.q20Len >= 20 ] = self.q20Len[ self.q20Len >= 20 ]
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
                self.metrics[lane]['q20_length'] = self.get_lane_metrics( self.q20Len , lane_id , 20, add_vmr=True )
                
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
                
        # At some point, maybe bring back these plots.  But simple bar charts aren't my favorite, let alone 4 point scatters.
        if False:
            # Mask out zeros for statistics reasons
            self.masked_q20Len_full=np.ma.masked_where(self.q20Len == 0, self.q20Len)
            
            #plot the figure for aq20mean(Only for debug. Remove this later and integrate into plotting structure )
            plt.figure ()
            values=[]
            x_values=[]
            #go lane by lane and fine the mean value of the aligned reads
            for i in range(4):
                current_value=np.ma.mean(self.masked_q20Len_full[:,self.masked_q20Len_full.shape[1]/4*i:self.masked_q20Len_full.shape[1]/4*(i+1)])
                values.append(current_value)
                x_values.append(i+1)
                if np.ma.is_masked(current_value)==True:
                    current_value=0
                self.lane_aq20_results[i+1].update({'mean_read_length':current_value})
            plt.plot   ( x_values,values , 'o')
            plt.ylabel ( 'AQ20mean (bp)' )
            
            # Could use full range of DSS but normally isn't needed. Let's have a dynamic range for now.
            plt.xticks([1,2,3,4])
            plt.xlim( (0.5, 4.5) )
            plt.title  ( 'AQ20mean by Lane')
            plt.grid   ( )
            plt.savefig( os.path.join( self.results_dir , 'Lane_aq20mean.png' ) )
            plt.close  ( )
            
            #Now plot figure for aligned reads
            plt.figure ()
            values=[]
            x_values=[]
            for i in range(4):
                current_value=np.ma.sum(self.masked_q20Len_full[:,self.masked_q20Len_full.shape[1]/4*i:self.masked_q20Len_full.shape[1]/4*(i+1)])
                values.append(current_value)
                x_values.append(i+1)
                if np.ma.is_masked(current_value)==True:
                    current_value=0
                self.lane_aq20_results[i+1].update({'num_reads':current_value})
            plt.plot   ( x_values,values , 'o')
            plt.ylabel ( 'IonExpressBarcode sum of all aq20 Aligned reads' )
            plt.xticks([1,2,3,4])
            plt.xlim( (0.5, 4.5) )
            plt.title  ( 'aq20 Aligned Reads by Lane')
            plt.grid   ( )
            plt.savefig( os.path.join( self.results_dir , 'Lane_aq20reads.png' ) )
            plt.close  ( )

            #Now plot spatial for mean reads
            plt.figure ()
            plt.imshow(self.masked_q20Len_full, vmin=np.ma.median(self.masked_q20Len_full)*0.5, vmax=np.ma.median(self.masked_q20Len_full)*1.5 )
            plt.colorbar()
            plt.title  ( 'Spatial aq20 Reads')
            plt.savefig( os.path.join( self.results_dir , 'LSpatial_aq20reads.png' ) )
            plt.close  ( )
            
            self.metrics.update({'lane_1_aq20mean':self.lane_aq20_results[1]['mean_read_length'],
            'lane_2_aq20mean':self.lane_aq20_results[2]['mean_read_length'],
            'lane_3_aq20mean':self.lane_aq20_results[3]['mean_read_length'],
            'lane_4_aq20mean':self.lane_aq20_results[4]['mean_read_length']})
            
        print(' . . . Finished Lane Alignment Results.\n')
        return None

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
        
    def find_die_area( self ):
        #from chipDiagnostics, possibly remove later to save time
        """ Finds the die area based on efuse x,y """
        x = self.explog.metrics['WaferX']
        y = self.explog.metrics['WaferY']
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
        
    def save_json( self ):
        ''' Saves metrics to json file  '''
        dump_dict={}
        dump_dict.update(self.metrics)
        with open( os.path.join( self.results_dir , 'results.json' ) , 'w' ) as f:
            json.dump( dump_dict,f)
            
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
        try:
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

        except:
            pass
        return block
        
    def write_block_html( self ):
        ''' Writes html file to be output on main report screen '''
        html = os.path.join( self.results_dir , 'Lane_Diagnostics_block.html' ) 
        
        block  = '''
                 <html><head><title>Lane Diagnostics</title></head>
                 <body>
                 <style type="text/css">table              {border-collapse: collapse;}</style>
                 <style type="text/css">tr:nth-child(even) {background-color: #DDD;}</style>
                 <style type="text/css">td                 {border: 1px solid black; text-align: center; }</style>
                 <style type="text/css">td.used            {background-color: #0f0;}</style>
                 <style type="text/css">td.unused          {background-color: #000;}</style>
        '''
        
        if not self.cal_exists:
            block += '''
                <p><em>The calibration files were not found for this run.  Attempted to contact ChipDB and exit.</em></p>
                <hr>'''
            block += self.write_normalduck_table( )
            block += '''</body></html>''' 
        elif self.lane_chip==False:
            block  = '''<p>This is a Fullchip Run, No Lane Diagnostics Needed</p>
                </body></html>''' 
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

            block += '''
            <h3>Lane Diagnostics Report</h3>
            '''
            for i, img in enumerate( imgs ):
                block += ''' <a href="{0}"><img src="{0}" width="20%" /></a> '''.format(img)
            block += '''
            <hr>
            '''
            
            block += self.write_normalduck_table( )
            block += ''' </body></html> '''
            
        with open( html , 'w' ) as f:
            f.write( block )
                
    def write_lane_html_files( self ):
        """ Creates a mirror of the main page, more or less, for single lanes. Table is omitted."""
        if self.cal_exists and self.is_multilane:
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
        if self.cal_exists and self.is_multilane:
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
            #             ( Name , Metric subset <e.g. 'local_aligned_reads'> , metric , formatter , divider )
            mtypes  = [('BeadFind Metrics' , '%' , 
                        [ ( 'Loading'     , None , 'loading',          '{:.1f}', None ) , 
                          ( 'FiltPass'    , None , 'percent_filtpass', '{:.1f}', None ) ,
                          ( 'Useless'     , None , 'percent_useless' , '{:.1f}', None ) ] ) ,
                       ('Q20 Read Length' , 'bp' , 
                        [ ( 'Median (Q2)' , 'q20_length' , 'q2'  ,     '{:.0f}' , None ) ,
                          ( 'Mean'        , 'q20_length' , 'mean',     '{:.0f}' , None ) ,
                          ( 'SD'          , 'q20_length' , 'std' ,     '{:.0f}' , None ) ,
                          ( 'VMR'         , 'q20_length' , 'vmr' ,     '{:.0f}' , None ) ] ) ,
                       ('Reads' , units , 
                        [ ( 'Aligned'     , None , 'aligned_reads' ,   '{:.1f}' , wd ) ,
                          ( 'Q20'         , None , 'q20_reads'     ,   '{:.1f}' , wd ) ] ) ,
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
            if not self.has_bfmask:
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
                for (name, subset, metric, fmt, divider) in rows:
                    row = '<tr><td width="40%">{}</td>'.format( name )
                    for lane in ['lane_{}'.format(l) for l in [1,2,3,4]]:
                        if self.metrics[lane]['active']:
                            try:
                                if subset:
                                    val = self.metrics[lane][subset].get( metric , 0. )
                                else:
                                    val = self.metrics[lane].get( metric , 0. )
                                if divider:
                                    if val == 0:
                                        pass
                                    else:
                                        val /= divider
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
            for img in ['multilane_plot_aligned_rl_histograms.png','multilane_plot_q20_rl_histograms.png']:
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
        reads    = [ mp_path.format(itype,i) for i in ['aligned_read_density','q20_read_density','q20_read_bead_conversions'] ]
        rra      = [ mp_path.format(itype,i) for i in ['rra','rra_fixed'] ]
        sepspa1  = [ mp_path.format(itype,i) for i in ['peak_sig', 'snr', 'sig_clust_conf'] ]
        sepspa2  = [ mp_path.format(itype,i) for i in ['tau_e', 'tau_b', 'buff_clust_conf'] ]
        rra.append( '' )
        
        # Prepare html text
        sections = ['Beadfind Analysis','Q20 Read Length','Q20 Variance-Mean-Ratio','Read Data','Raw Read Accuracy']
        images   = [ bf_imgs , q20mrl , q20vmr , reads , rra ]
        labels   = [ [] ,
                     ['Autoscaled Limits','Flow-based Limits','Mean +/- 50 bp Limits'] ,
                     ['Autoscaled Limits','Flow-based Limits','Fixed Scale Limits'] ,
                     [] ,
                     ['Autoscaled Limits','Fixed Limits',''],
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
        
    def initialize( self ):
        """ 
        New method meant to unify parameter detection, from startplugin.json, explog, etc. and validate chiptype.
        """
        print( 'Initializing plugin data . . .' )
        self.metrics = {}
        self.get_plugin_json( )
        self.get_barcodes   ( )
        self.read_explog    ( ) # Saves as self.explog
        
        # Deal with chiptype
        try:
            self.determine_chiptype( )
            print( ' . . . Initialization completed and chiptype confirmed.' )
        except ValueError:
            # Something went heinously wrong.  Let's exit as gracefully as possible.
            block_html = os.path.join( self.results_dir , '{}_block.html'.format( self.__class__.__name__ ) )
            with open( block_html , 'w' ) as f:
                msg = "Unable to find explog files, chip block directories, or calibration files in order to ascertain the chiptype for this plugin.  Exiting gracefully without performing analysis."
                f.write( '<html><body><p><em>{}</em></p></body></html>'.format( msg ) )
            sys.exit( 0 )
            
    def get_barcodes( self ):
        """ Reads in barcodes associated with the run """
        self.barcodes = {}
        with open( self.results_dir+'/barcodes.json','r' ) as f:
            allbc = json.load(f)
            
        for b in allbc:
            # Ignore barely-found barcodes as noise
            if int(allbc[b]['read_count']) > 1000:
                self.barcodes[b] = allbc[b]
                
        return None
        
    def get_plugin_json( self ):
        """ Reads in namespace-like variables from startplugin.json. Blatenly stolen from chipDiagnostics"""
        # print( 'Reading startplugin.json . . .' )
        # if not getattr(self, 'startpluginjson', None ):
        #     try:
        #         with open('startplugin.json','r') as fh:
        #             self.startpluginjson = json.load(fh)
        #     except:
        #         self.log.exception("Error reading start plugin json")
                
        # Define some needed variables
        self.plugin_dir   = os.environ['DIRNAME']
        self.raw_data_dir = os.environ['RAW_DATA_DIR']
        self.analysis_dir = os.environ['ANALYSIS_DIR']
        self.results_dir  = os.environ['TSP_FILEPATH_PLUGIN_DIR']
        self.sigproc_dir  = os.environ['SIGPROC_DIR']
        if not os.path.exists(self.sigproc_dir):
            self.sigproc_dir  = self.analysis_dir + '/rawdata/onboard_results/sigproc_results'
        self.flows        = os.environ['TSP_NUM_FLOWS']
        self.chip_type    = os.environ['TSP_CHIPTYPE']
        self.efuse        = os.environ['TSP_CHIPBARCODE']
        
        if 'FC:' in self.efuse:
            if ',' in self.efuse.split('FC:')[1]:
                self.flowcell_version=self.efuse.split('FC:')[1].split(',')[0]
        else:
            self.flowcell_version='Flowcell Version Not Available'

        # This may be useful in other code. . . 
        if 'thumbnail' in self.raw_data_dir:
            self.thumbnail = True
            self.caldir    = self.raw_data_dir.split('thumbnail')[0]
        else:
            self.thumbnail = False
            self.caldir    = self.raw_data_dir
            
        print ( 'Chip type discovered: %s' % self.chip_type )
        print ( ' . . . startplugin.json successfully read and parsed.\n' )
        return None   

    def read_explog( self ):
        # Test for 'explog_final.txt' in raw_data_dir.  If not there, check one directory up.
        if os.path.exists( os.path.join( self.analysis_dir, 'explog.txt.csa' ) ):
            self.explog = explog.Explog( path=os.path.join( self.analysis_dir, 'explog.txt.csa' ))
            print( 'Found explog_final.txt in analysis_dir' )
        elif os.path.exists( os.path.join( os.path.dirname(self.raw_data_dir) , 'explog_final.txt' ) ):
            self.explog = explog.Explog( path=os.path.dirname(self.raw_data_dir) )
            print( 'Found explog_final.txt in os.path.dirname(raw_data_dir)!' )
        elif os.path.exists( os.path.join( self.raw_data_dir , 'explog_final.txt' ) ):
            self.explog = explog.Explog( path=self.raw_data_dir )
            print( 'Found explog_final.txt in raw_data_dir!' )
        else:
            print( 'Error!  Could not find explog_final.txt' )
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
                        cc         = chipcal.ChipCal  ( self.raw_data_dir , chiptype=self.chip_type )
                        rows, cols = cc.load_offset_rc( )
                        chiprc     = ( rows , cols)
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
    
    ####################################################################################################
    # Depreciated methods and analyses that are not likely to be used, but are retained for posterity.
    ####################################################################################################
    
    def old_bfmask_analysis( self ):
        """ 
        It is not currently clear if the following analysis is accurate or useful. 
        At present it is excluded from the plugin.    - Phil
        """
        #New in 0.2.4: Sum of low loading in each area
        self.bf.select_mask( 'bead' )
        for idx,i in enumerate(self.lane):
            #create beaded array
            x = self.bf.block_reshape( self.bf.current_mask , 
            [self.bf.chiptype.miniR,self.bf.chiptype.miniC] )
            beadarray = 100. * np.array(x , float).mean(2)[:,x.shape[1]/4*idx:x.shape[1]/4*(idx+1)]

            plt.imshow(beadarray)
            plt.title  ( 'beadarray lane {}'.format(idx+1) )
            plt.savefig( os.path.join( self.results_dir , 'beadarr-lane{}.png'.format(str(idx).format(str(idx+1)))))
            plt.close()
            #create masked gain array
            x = self.bf.block_reshape( self.cc.gain , 
            [self.bf.chiptype.miniR,self.bf.chiptype.miniC] )
            gainarray = np.array(x , float).mean(2)[:,x.shape[1]/4*idx:x.shape[1]/4*(idx+1)]
            mgainarray=np.ma.masked_where(gainarray<800,gainarray)
            plt.imshow(mgainarray)
            plt.title('mgainarray lane {}'.format(idx+1))
            plt.colorbar()
            plt.savefig( os.path.join( self.results_dir , 'mgainarr-lane{}.png'.format(str(idx+1))))
            plt.close()

            #use the gian array to analyze beaded array
            mbeadedarray=np.ma.masked_where(gainarray<800,beadarray)
            plt.imshow(mbeadedarray)
            plt.title('mbeadedarray lane {}'.format(idx+1))
            plt.colorbar()
            plt.savefig( os.path.join( self.results_dir , 'mbeadarr-lane{}.png'.format(str(idx+1))))
            plt.close()

            #find areas of low loading based off of median loading
            p50_loading=np.median(mbeadedarray.data)
            low_loading_cutoff=20
            if p50_loading>low_loading_cutoff:
                m_low_loading_array=np.ma.masked_where(mbeadedarray>low_loading_cutoff,mbeadedarray)
                plt.imshow(m_low_loading_array)
                plt.title('m_low_loading_array lane {}'.format(idx+1))
                plt.colorbar()
                plt.savefig( os.path.join( self.results_dir , 'm_low_loading_array-lane{}.png'.format(str(idx+1))))
                plt.close()

            #left side into 0.5 in
            cropped_array=mbeadedarray[mbeadedarray.shape[0]*0.1:mbeadedarray.shape[0]*0.9,
                          0:mbeadedarray.shape[1]*0.5]
            leftcount=0
            for row in cropped_array:
                temp_row_count=0
                for element in row:
                    # only work on non masked elements
                    if np.ma.is_masked(element)==False:
                        #figure out if the element is low loading (bubble)
                        if element<low_loading_cutoff:
                            temp_row_count+=1
                        #stop iterating elements when we hit a high loading area
                        # this indicates that the bubble is finished. 
                        elif element>p50_loading-p50_loading*0.35:
                            leftcount=leftcount+temp_row_count
                            break
            plt.imshow(cropped_array,interpolation='none',vmin=0,vmax=20)
            plt.title('left cropped_array lane {}'.format(idx+1))
            plt.colorbar()
            plt.savefig( os.path.join( self.results_dir , 'leftside-lane{}.png'.format(str(idx+1))))
            plt.close()
            self.metrics.update({'loading_bubble_count_leftside_lane{}'.format(str(idx+1)):leftcount})


            #right side into 0.25 in
            cropped_array=np.fliplr(mbeadedarray)[mbeadedarray.shape[0]*0.1:mbeadedarray.shape[0]*0.9,
                          0:mbeadedarray.shape[1]*0.5]
            rightcount=0
            for row in cropped_array:
                temp_row_count=0
                for element in row:
                    # only work on non masked elements
                    if np.ma.is_masked(element)==False:
                        #figure out if the element is low loading (bubble)
                        if element<low_loading_cutoff:
                            temp_row_count+=1
                        #stop iterating elements when we hit a high loading area
                        # this indicates that the bubble is finished. 
                        elif element>p50_loading-p50_loading*0.35:
                            rightcount=rightcount+temp_row_count
                            break
            plt.imshow(cropped_array,interpolation='none',vmin=0,vmax=20)
            plt.title('right cropped_array lane {}'.format(idx+1))
            plt.colorbar()
            plt.savefig( os.path.join( self.results_dir , 'rightside-lane{}.png'.format(str(idx+1))))
            plt.close()
            self.metrics.update({'loading_bubble_count_rightside_lane{}'.format(str(idx+1)):rightcount})

            #bottom side into 0.10 in
            rolled_array=np.rollaxis(mbeadedarray,1)
            cropped_array=rolled_array[:,0:rolled_array.shape[0]*0.1]
            bottomcount=0
            for row in cropped_array:
                temp_row_count=0
                for element in row:
                    # only work on non masked elements
                    if np.ma.is_masked(element)==False:
                        #figure out if the element is low loading (bubble)
                        if element<low_loading_cutoff:
                            temp_row_count+=1
                        #stop iterating elements when we hit a high loading area
                        # this indicates that the bubble is finished. 
                        elif element>p50_loading-p50_loading*0.35:
                            bottomcount=bottomcount+temp_row_count
                            break
            plt.imshow(cropped_array,interpolation='none',vmin=0,vmax=20)
            plt.title('bottom cropped_array lane {}'.format(idx+1))
            plt.colorbar()
            plt.savefig( os.path.join( self.results_dir , 'bottomside-lane{}.png'.format(str(idx+1))))
            plt.close()
            self.metrics.update({'loading_bubble_count_bottomside_lane{}'.format(str(idx+1)):bottomcount})

            #top side into 0.10 in
            rolled_array=np.rollaxis(mbeadedarray,1)
            cropped_array=np.fliplr(rolled_array)[:,0:rolled_array.shape[0]*0.1]
            topcount=0
            for row in cropped_array:
                temp_row_count=0
                for element in row:
                    # only work on non masked elements
                    if np.ma.is_masked(element)==False:
                        #figure out if the element is low loading (bubble)
                        if element<low_loading_cutoff:
                            temp_row_count+=1
                        #stop iterating elements when we hit a high loading area
                        # this indicates that the bubble is finished. 
                        elif element>p50_loading-p50_loading*0.35:
                            topcount=topcount+temp_row_count
                            break
            plt.imshow(cropped_array,interpolation='none',vmin=0,vmax=20)
            plt.title('top cropped_array lane {}'.format(idx+1))
            plt.colorbar()
            plt.savefig( os.path.join( self.results_dir , 'topside-lane{}.png'.format(str(idx+1))))
            plt.close()
            self.metrics.update({'loading_bubble_count_topside_lane{}'.format(str(idx+1)):topcount})

            #Now add bubble pixels up
            self.metrics.update({'loading_bubble_count_total_lane{}'.format(str(idx+1)):leftcount+rightcount+topcount+bottomcount})

        #Add all lanes bubble counts together for a full chip count
        self.metrics.update({'loading_bubble_count_total_alllanes':self.metrics['loading_bubble_count_total_lane1']
                                                                 + self.metrics['loading_bubble_count_total_lane2']
                                                                 + self.metrics['loading_bubble_count_total_lane3']
                                                                 + self.metrics['loading_bubble_count_total_lane4']})

        # "This code never really worked as well as I wanted." - Creighton
        # Behavior of ignore mask was unpredictable at best.
        #New in 0.2.3: sum of ignore bin 
        temp_dict={}
        self.bf.select_mask( 'ignore' )
        self.bf_ignore_sum=np.sum(self.bf.current_mask)
        lane_dict={'lane_1':1,'lane_2':2,'lane_3':3,'lane_4':4}
        region_dict={'inlet':[0,0.1],'middle':[0.1,0.9], 'outlet':[0.9,1], 'full_lane':[0,1]}
        if self.lane_chip==True:
            for lane in lane_dict.keys():
                if self.lane[lane_dict[lane]-1]==True:
                    for region in region_dict.keys():
                        temp_arr=self.bf.current_mask
                        if region=='middle':
                            #only look at the edge
                            cropped_arr1=temp_arr[region_dict[region][0]*temp_arr.shape[0]:region_dict[region][1]*temp_arr.shape[0],temp_arr.shape[1]/4*(lane_dict[lane]-1):temp_arr.shape[1]/4*(lane_dict[lane])]
                            left_arr_sum=np.sum(cropped_arr1[:,0:cropped_arr1.shape[0]/4*1])
                            right_arr_sum=np.sum(cropped_arr1[:,cropped_arr1.shape[0]/4*3:])
                            temp_dict.update({lane+'_'+'bfignore_sum'+'_'+region:left_arr_sum+right_arr_sum})
                        else:
                            cropped_arr=temp_arr[region_dict[region][0]*temp_arr.shape[0]:region_dict[region][1]*temp_arr.shape[0],temp_arr.shape[1]/4*(lane_dict[lane]-1):temp_arr.shape[1]/4*(lane_dict[lane])]
                            temp_dict.update({lane+'_'+'bfignore_sum'+'_'+region:np.sum(cropped_arr)})
        self.metrics.update(temp_dict)
        return None
        
    def update_lane_data( self ):
        '''
        checks which arrays are available and calculates metrics on it on a per lane basis
        Phil Note:  This is mostly covered in chipDiagnostics now.  Only unique feature here is looking at regions of performance in inlet, outlet, left, right.
        '''
        lane_dict={'lane_1':1,'lane_2':2,'lane_3':3,'lane_4':4}
        metric_dict={'noise':'noise', 'gain':'gain', 'offset':'offset'}
        #Check to make sure everyhting is loaded first
        for metric in metric_dict.keys():
            if not hasattr( self.cc , metric):
                print( 'Loading {} . . .'.format(metric) )
                eval('self.cc.load_{}()'.format(metric))
                print( ' . . . done.' )
        region_dict={'inlet':[0,0.33333],'middle':[0.333333,0.66666], 'outlet':[0.666666,1], 'full_lane':[0,1]}
        
        stats_dict={'q1':1,'q10':10,'q25':25,'q50':50,'q75':75,'q90':90,'q99':99, 
        'iqr':'custom', 'average':'np_attr', 'std':'np_attr'}
        #initialize the final array
        temp_dict={}
        for lane in lane_dict.keys():
            for metric in metric_dict.keys():
                for region in region_dict.keys():
                    for stat in stats_dict.keys():
                        temp_dict.update({lane+'_'+metric+'_'+region+'_'+stat:None})
        if self.lane_chip==True:
            for lane in lane_dict.keys():
                if self.lane[lane_dict[lane]-1]==True:
                    for region in region_dict.keys():
                        for metric in metric_dict.keys():
                            #print('Starting ', lane, region, metric)
                            temp_arr=getattr(self.cc, metric_dict[metric])
                            cropped_arr=temp_arr[region_dict[region][0]*temp_arr.shape[0]:region_dict[region][1]*temp_arr.shape[0],temp_arr.shape[1]/4*(lane_dict[lane]-1):temp_arr.shape[1]/4*(lane_dict[lane])]
                            #print('finished array cropping')
                            p_list=[]
                            for stat in stats_dict.keys():
                                #print ('Doing ', stat)
                                if type(stats_dict[stat])==int:
                                    p_list.append(stats_dict[stat])
                                elif stats_dict[stat]=='np_attr':
                                    temp_dict.update({lane+'_'+metric+'_'+region+'_'+stat:getattr(np,stat)(cropped_arr)})
                            #print ('Startign percentile')
                            p_results=np.percentile(cropped_arr,p_list)
                            p_dict=dict(zip(p_list, p_results))
                            for p in p_dict.keys():
                                temp_dict.update({lane+'_'+metric+'_'+region+'_'+'q'+str(p):p_dict[p]})
                            #print ('Starting unique: IQR')
                            temp_dict.update({lane+'_'+metric+'_'+region+'_'+'iqr':p_dict[75]-p_dict[25]})
                            ##print ('Starting unique: mode')
                            #temp_dict.update({lane+'-'+metric+'-'+region+'-'+'mode':stats.mode(cropped_arr)})
                            #print ('Finished setting') 

                            
        else:
            print ('No Lanes, not applicable')
            
        self.metrics.update(temp_dict)

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
    
if __name__ == "__main__":
    PluginCLI()

