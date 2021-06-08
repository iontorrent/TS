#!/usr/bin/env python
# Copyright (C) 2018 Ion Torrent Systems, Inc. All Rights Reserved
#
# Plugin written by Brennan Pursley 11Apr2019

# Torrent specific imports
from ion.plugin import *

# Generally useful libraries
import numpy as np
import numpy.ma as ma
import os, sys, textwrap
import time, re
import subprocess, urllib, h5py
import warnings, copy
deepcopy = copy.deepcopy

import traceback

# Image Processing
import scipy.ndimage as ndimage
from scipy.ndimage import morphology as morph

# Plotting tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import prctile
from matplotlib.gridspec import GridSpec

# Load modules from our tools
from tools import chipcal, chiptype, explog
from tools import datfile, stats, thumbnail
from tools import bfmask, thumbnail
from tools.PluginMixin import PluginMixin

# import multilane plotting tools
from multilane_plot import MultilanePlot

import pybam

NBC = 2**15-1 # so that data fits into int16 or uint16

class NucStepSpatialV2( IonPlugin, PluginMixin ):
    '''
    Plugin to extract metrics regarding bead sticking

    Brennan Pursley (2019)

    Updates     | 07Jan2020 |   v1.9.0  | Added ffc metrics
                            |   v1.9.1  | Debugging and added ffc figures to front display
                | 08Jan2020 |   v1.9.2  | Added ffc images to subpage
                                        | Simplified logic for making html
                            |   v1.9.3  | Bugfixes
                            |   v1.9.4  | Bugfix for inlet-->outlet ordering for ffc
                            |   v1.9.5  | Added display of key nuc metrics on ffc
                            |   v1.9.6  | Streamlined html logic
                | 23Jan2020 |   v1.9.7  | fixed bug in color coding of GCAT traces
                | 10Feb2020 |   v1.10.0 | Adding in prerun flow analysis
                            |   v1.10.1 | Bugfixes + aesthetic improvements
                | 11Feb2020 |   v1.10.2 | Removed npy output files to reduce size
                                        | set find_min=True for prerun metrics for more appropriate fwhm
                | 13Feb2020 |   v1.10.3 | updated tools to get CSA functionality
                | 14Feb2020 |   v1.10.5 | send all output to results_dir
                | 18Feb2020 |   v1.10.6 | turned off saving of .dat files
                | 05Mar2020 |   v1.10.7 | bug fix for empty np.where()
                | 19Mar2020 |   v1.10.8 | Added in spike analysis for G nuc
                | 20Mar2020 |   v1.10.9 | bugfixes, logic improvements, and plot of spike height
                            |   v1.10.10| added spike plateau, fixed logic bug in removing lists of data
                                        | fixed bug in logic for plateau
                            |   v1.10.11| fixed logic errors, added more spike plots
                | 15May2020 |   v1.11.0 | added relative_key_spatial image generator
                                        | --> commented out most code for quick debug
                                v1.11.1 | Generates relative_key_spatial image with link on plugin output
                | 19May2020 |   v1.11.2 | Trim number of output metrics
                            |   v1.11.3 | improved trimming of metrics
                | 01Jul2020 |   v1.11.4 | handled sigproc/seq folder
                | 07Aug2020 |   v1.12.0 | improve LowT image
                                        | add LowT metrics
                            |   v1.12.1 | bugfixes
                | 13Aug2020 |   v1.12.3 | updated tools for GX7 thumbnail and improved PluginMixin
                | 17Aug2020 |   v1.12.4 | updated tools for updated chipcal.determine_lane
                | 19Aug2020 |   v1.12.5 | fixed ylims bug
                            |   v1.12.6 | logic update for ylims
                            |   v1.13.0 | Changed display images
    '''

    version         = "1.13.0"
    allow_autorun   = True
    runtypes        = [ RunType.THUMB ]

    def launch( self ):
        #NOTE: (un)comment for troubleshooting
        #warnings.filterwarnings('error')
        self.init_plugin()

        self.exit_on_missing_files( rawdata=True )

        if self.explog_is_multilane:
            self.metrics.update( { 'ChipType' : self.chip_type , 'lane_1':{}, 'lane_2':{}, 'lane_3':{}, 'lane_4':{} } )
        else:
            self.metrics.update( { 'ChipType' : self.chip_type } )

        #NOTE: This sets the class global attribute blocksize
        self.blocksize = [10,10]
        
        # Evaluate normalized step height behavior for beadsticking
        for isspa in [True,False]:
            self.load_thumbnail_data( isspa=isspa )        
            self.nss_processing( isspa=isspa )
            if not isspa:
                # Evaluate raw nucstep quality from 0-mer flows
                self.nuc_quality_check()
                self.nuc_spikes()
            else:
                # generate relative key images spatial plot
                # Low-T Screen
                try:
                    self.relative_peak_spatial()
                except Exception as e:
                    print( '!!! ERROR:  Relative Peak Spatial Image Was Not Generated !!!\n{}\n'.format(e) )
                    traceback.print_exc()
            self.cleanup( isspa=isspa )
            #TODO: Get this working
            #self.bf_processing( isspa=isspa )
        # Checkpoint writing of metrics before attempting pybam
        self.write_metrics()

        # Evaluate read quality metrics vs stuck bead locations
        self.get_pybam_array()
        self.pybam_vs_dist_from_stuck()
        self.pybam_vs_nss_height()
        # checkpoint of writing metrics before continuing
        self.write_metrics()


        self.write_html()
        # write something to the block html no matter what
        self.write_block_html()
        self.write_spatial_key_html()

        sys.exit(0)

#################################
#       CORE FUNCTIONS          #
#################################

    def cleanup( self, isspa=True, things_to_delete=None ):
        if things_to_delete is None:
            things_to_delete = [t for t in dir(self) if 'aq_' in t or 'bf_' in t ] 
        print( 'Things to delete\n{}'.format( things_to_delete ) )

        for thing in things_to_delete:
            if isspa and not '_spa' in thing:
                continue
            try:    delattr( self, thing )
            except: 
                print( 'Deletion of {} did not work'.format(thing) )
                pass

    def get_pybam_array( self ):
        ''' Runs pybam to extract reads and various qualities of readlength '''
        #NOTE Only works with regular thumbnail
        if self.has_reference:
            quality = [ -1, 0, 7, 20 ] # -1: barcode mask
                                       #  0: read length
                                       #  7: q7 length
                                       # 20: q20 length
        else:
            quality = [ -1, 0]
            
        # Plugin only runs on thumbnail
        shape = [ 800, 1200, len(quality) ]

        qualmat = np.zeros( shape, dtype=np.uint16 ) # pybam requires uint16
        print( 'qualmat shape', qualmat.shape )

        # If barcodes are present, then do not merge the non-barcode reads
        # If only non-barcoded reads are present, then read those
        # Check if non-barcoded reads are present
        nobc = NBC in [ bc.get( 'barcode_index', NBC ) for bc in self.barcodes.values() ]
        # Check if barcoded reads are present
        bc   = any( [ bc.get( 'barcode_index', NBC ) != NBC for bc in self.barcodes.values() ] )
        merge_nbc = nobc and not bc
        
        for bc in self.barcodes.values():
            filename = bc['bam_filepath']
            if not os.path.exists( filename ):
                print( 'WARNING!  Bamfile not found for barcode {}! Path: {}'.format( bc['barcode_name'],
                                                                                      filename ) )
                continue
            print( 'procesing new bamfile:' )
            print( filename )
            fill = bc.get( 'barcode_index', NBC )
            if ( fill == NBC ) and ( not merge_nbc ):
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

        self.aligned_length = qualmat[:,:,1].astype( np.int16 )
        #self.aligned_length.tofile( 'aligned_length.dat' )

        self.totalreads = qualmat[:,:,1].astype( np.bool )
        #self.totalreads.tofile( 'totalreads.dat' )

        # These only work if there's a reference
        if self.has_reference:
            self.q7_length = qualmat[:,:,2].astype( np.int16 )
            #self.q7_length.tofile( 'q7_length.dat' )
            self.q20_length = qualmat[:,:,3].astype( np.int16 )
            #self.q20_length.tofile( 'q20_length.dat' )
        else:
            self.q7_length = np.zeros( self.aligned_length.shape )
            self.q20_length = np.zeros( self.aligned_length.shape )
            
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
                    
            self.nbc_aligned_length = qualmat[:,:,1].astype( np.int16 )
            #self.nbc_aligned_length.tofile( 'nbc_aligned_length.dat' )
            self.nbc_totalreads = qualmat[:,:,1].astype( np.bool )
            #self.nbc_totalreads.tofile( 'nbc_totalreads.dat' )

        self.barcode_mask = qualmat[:,:,0].astype( np.int16 )
        #self.barcode_mask.tofile( 'barcode_mask.dat' )

    def load_datfile( self, aq_num=6, isspa=True, ftype='aq' ):
        ''' Handles loading, masking, and attribute setting of datfiles '''
        if ftype == 'aq':
            filename = 'acq_{:04d}.dat'.format(aq_num)
        elif ftype == 'bf':
            filename = 'beadfind_pre_{:04d}.dat'.format(aq_num)
        elif ftype == 'pr':
            filename = 'prerun_{:04d}.dat'.format( aq_num )
        else:
            print( 'Wrong Filetype' )
            raise
        if isspa:   filename += '_spa'
        path = os.path.join( self.acq_dir, filename )
        print( 'acquisition file: {}'.format( path ) )
        
        #####   AQ_DF PROCESSING    #####
        if isspa:   chiptype = self.ct.tn_spa
        else:       chiptype = self.ct.thumbnail
        aq_df = datfile.DatFile( filename=path, norm=True, fcmask=False, dr=self.explog.DR, chiptype=chiptype )
        print( 'aq_df rows, cols, frames', aq_df.rows, aq_df.cols, aq_df.frames )
        print( 'aq_df blockRC', aq_df.chiptype.blockR, aq_df.chiptype.blockC )
        
        # If processing the beadfind file, need to extract active pixels
        if ftype=='bf':
            print( 'bf data shape: {}'.format( aq_df.data.shape ) )
            print( 'bf isbasic: {}'.format( aq_df.isbasic ) )
            # NOTE:  Active already filters/excludes for pinned
            active = aq_df.measure_active_pixels()
            print( 'active_shape: {}'.format( active.shape ) )
            self._set_attr( 'active', active, isspa )
            self.output_image( active.astype(np.int), 'Active', 'active', isspa=isspa )
            #self.save_array( 'active', isspa )
            
            # Fill in the gaps inside active lanes 
            #--> could be important as bead sticking might make some wells look inactive
            filled_active = morph.binary_fill_holes( active )
            self._set_attr( 'filled_active', filled_active, isspa )
            self.output_image( active.astype(np.int), 'Filled Active', 'filled_active', isspa=isspa )
            #self.save_array( 'filled_active', isspa )

            # set the pinned attribute
            self._set_attr( 'pinned', aq_df.measure_pinned(), isspa )

            # extract and store bf_slope
            self.measure_val( 'slope', aq_df, ftype, aq_num, isspa )
        elif (ftype=='aq' or ftype=='pr') and not isspa:
            if self.has_bfmask:
                # need to set pinned
                aq_df.pinned = self._get_attr( 'pinned', isspa )
                # need to set active pixels
                aq_df.actpix = self._get_attr( 'active', isspa )
                # need to set this or else active pix will be overwritten
                aq_df.active_thresh = aq_df._active_thresh = 0
                # if isspa, get values for nuc_quality_check
                # NOTE: mask needs to be what we don't want --> NOT desired
                # bead wells
                avgtrace_bead = aq_df.measure_avgtrace( micro=False, mask= ~self.bead )
                print( 'avgtrace shape bead {}'.format( avgtrace_bead.shape ) )
                self._set_attr( '{}_{:04d}_avg_trace_bead'.format(ftype, aq_num), avgtrace_bead, isspa )
                #self.save_array( '{}_{:04d}_avg_trace_bead'.format(ftype, aq_num), isspa )
                # empty wells
                avgtrace_empty = aq_df.measure_avgtrace( micro=False, mask= ~self.empty, redo=True )
                print( 'avgtrace shape empty {}'.format( avgtrace_empty.shape ) )
                self._set_attr( '{}_{:04d}_avg_trace_empty'.format(ftype, aq_num), avgtrace_empty, isspa )
                #self.save_array( '{}_{:04d}_avg_trace_empty'.format(ftype, aq_num), isspa )
        elif ftype=='aq' and isspa:
            pass
        else:
            raise

        ## Always Measure height
        self.measure_val( 'height', aq_df, ftype, aq_num, isspa )

        del aq_df

    def load_thumbnail_data( self, isspa=True ):
        ''' Load and mask arrays of regular thumbnail or spatial data '''
        if isspa:
            print( '\nLoading spatial thumbnail data . . .')
        else:
            fc_bf_mask  = self.fc_bfmask_path
            if not self.has_fc_bfmask:
                self.has_bfmask = False
                print( '\nIt appears that analysis.bfmask.bin has already been deleted, or was never created.  Skipping associated processing.')
            else:
                print( '\nSetting up desired bead mask' )
                # Import bfmask.BeadfindMask and run with its canned analysis. 
                self.bf = bfmask.BeadfindMask( fc_bf_mask )
                # set up bf masking
                self.bf.select_mask('bead')
                self.bead = self.bf.current_mask
                if self.bead.shape != (1200,800,):
                    self.bead = thumbnail.get_thumbnail( self.bead, chiptype=self.ct )
                self.bf.select_mask( 'empty' )
                self.empty = self.bf.current_mask
                if self.empty.shape != (1200,800,):
                    self.empty = thumbnail.get_thumbnail( self.empty, chiptype=self.ct )
            print( '\nLoading thumbnail data . . .' )
        
        #####   BF_DF PROCESSING    #####
        #NOTE MUST HAPPEN FIRST TO GET active AND filled_active
        self.load_datfile( aq_num=1, isspa=isspa, ftype='bf' )

        #####   AQ_DF PROCESSING    #####
        # Use aq_num 6 for beadsticking analysis
        # Use aq_nums 1,3,4, and 6 for 0-mer nuc analysis
        # Use aq_nums 0,2,5, and 7 for 1-mer nuc analysis
        if not isspa:   aq_list = [i for i in range(8)]
        else:       aq_list = [6]
        for i in aq_list:
            self.load_datfile( aq_num=i, isspa=isspa, ftype='aq' )
        if not isspa:
            for i in aq_list:
                self.load_datfile( aq_num=i, isspa=isspa, ftype='pr' )

        print( '. . . loading is complete' )

    def measure_val( self, val, aq_df, ftype, aq_num, isspa ):
        ''' perform all value measurements of arrays '''
        filled_active = self._get_attr( 'filled_active', isspa )

        if ftype == 'aq' or ftype=='pr':
            # need to set active pixels
            aq_df.actpix = self._get_attr( 'active', isspa )
            # need to set this or else active pix will be overwritten
            aq_df.active_thresh = aq_df._active_thresh = 0
            # set aq_df_condition and save
            aq_df_condition = np.logical_and( filled_active.astype(bool), ~aq_df.pinned.astype(bool) )
        elif ftype == 'bf':
            aq_df_condition = np.logical_and( filled_active.astype(bool), ~aq_df.pinned.astype(bool) ) 
        else:
            raise
        # Store filtering condition
        self._set_attr( '{}_{:04d}_df_condition'.format(ftype, aq_num), aq_df_condition, isspa )
        self.output_image( 
                aq_df_condition, 
                '{} {:04d} {} Filtering Condition'.format(ftype.upper(), aq_num, val[0].upper()+val[1:] ), 
                '{}_{:04d}_{}_filtering_condition'.format(ftype, aq_num, val), 
                isspa=isspa )
        #self.save_array( '{}_{:04d}_df_condition'.format(ftype, aq_num), isspa )

        # generate value array before masking
        if   val == 'height':   temp = aq_df.measure_height()
        #elif val == 't0'    :   temp = aq_df.measure_t0( )
        elif val == 'slope' :   temp = aq_df.measure_slope()

        print( 'temp shape {}'.format(temp.shape) )
        print( 'aq_df_condition shape {}'.format( aq_df_condition.shape ) )

        # mask value array and store
        val_array   = np.zeros( temp.shape )
        val_array[aq_df_condition]  = temp[ aq_df_condition ]
        self._set_attr( '{}_{:04d}_{}'.format(ftype, aq_num, val), val_array, isspa )
        print( '{} {} shape: {}'.format( 
            ftype,
            val,
            val_array.shape )
            )
        self.output_image( 
                val_array, 
                '{} {:04d} {}'.format(ftype.upper(), aq_num, val.upper()), 
                '{}_{:04d}_{}'.format(ftype, aq_num, val), 
                isspa=isspa )
        #self.save_array( '{}_{:04d}_{}'.format(ftype, aq_num, val), isspa )

    
#####################################
#       Beadsticking Analysis       #
#####################################

    # Core beadsticking calculation
    def nss_processing( self, isspa=True ):
        # Modified to make use of our height calculation and masking
        print( '\nStarting nss_processing . . .' )
        # Handle warnings like an error
        warnings.filterwarnings('error')
        # define normalization of an imgslice
        def norm_imgslice( imgslice ):
            for c in range(120):
                for r in range(80):
                    imgReg = imgslice[r*10:(r+1)*10, c*10:(c+1)*10]
                    try:
                        const = np.percentile( imgReg, 90 ) #imgReg[imgReg > prctile(imgReg, 95)].mean()
                        imgslice[r*10:(r+1)*10, c*10:(c+1)*10] = imgslice[r*10:(r+1)*10, c*10:(c+1)*10]/const
                    except RuntimeWarning:
                        # Nothing to do since we aren't modifying the array
                        pass
            # remove nans and infs
            return self.remove_nans_infs( imgslice )
        
        # Save for possible future use
        # normalize imgslice by raw gain_thumb --> 0's where gain == 0
        imgslice = self._get_attr( 'aq_0006_height', isspa )
        
        if isspa:
            # exctract proper thumbnail for raw gain (no thresholding)
            self.cc = chipcal.ChipCal( self.calibration_dir , self.ct.name , self.results_dir )
            self.cc.load_gain( )
            gain_thumb = thumbnail.get_thumbnail( self.cc.gain, chiptype=self.ct, spa=isspa )
            gt_title = 'Gain Thumbnail'
            gt_file = 'gain_thumb'
            self.output_image( gain_thumb, gt_title, gt_file, isspa=isspa )
            self.cc.close_gain()
            print( imgslice.shape, gain_thumb.shape )
            imgslice = np.divide( imgslice, gain_thumb, out=np.zeros( imgslice.shape ), where=gain_thumb>0 )
            print( imgslice.shape )
            imgslice = self.remove_nans_infs( imgslice )

        # Normalize and rescale imgslice by 100
        imgslice = 100.*norm_imgslice( imgslice )
        #imgslice = 100.*norm_imgslice( self._get_attr( 'aq_0006_height', isspa ) )
        self._set_attr( 'nss_height', imgslice, isspa )
        #self.save_array( 'nss_height', isspa )

        title       = 'Nuc Step Height'
        filename    = 'NSS_normImage'
        vlims       = (0,100,)
        self.output_image( imgslice, title, filename, vlims=vlims, isspa=isspa )

        self.output_image( imgslice, 'NSS Extremes', 'NSS_extremes', vlims=(100,200,), isspa=isspa )

        if self.explog_is_multilane:
            self.extract_and_plot_lane_metrics( imgslice, 'nss_step_height', 'Nuc Step Height', 'Height', clims=(0,100,), local_clims=(0,100,), localstd_clims=(0,50,), isspa=isspa, integral_cutoffs=[10,25,50,75], extreme_high=100 )
        else:
            self.extract_and_plot_ffc_metrics ( imgslice, 'nss_step_height', 'Nuc Step Height', 'Height', clims=(0,100,), local_clims=(0,100,), localstd_clims=(0,50,), isspa=isspa, integral_cutoffs=[10,25,50,75], extreme_high=100 )
        
        # Reset warning handling
        warnings.filterwarnings('default')
        print( '. . . nss_processing is complete\n' )

    # Read quality from pybam data
    def extract_reads_and_q20( self, mask=None ):
        ''' Gather reads, wells, read density, q20_reads, and q20_mean from masked arrays '''
        base_zeros = np.zeros(self.totalreads.shape)
        base_ones  = np.ones( self.totalreads.shape)
        if mask is None:
            mask = np.array(base_ones)
        mask = mask.astype( np.bool )
        m_reads     = np.array( base_zeros )
        m_q20       = np.array( base_zeros )
        m_q20_reads = np.array( base_zeros )

        m_reads[mask==True] = self.totalreads[mask==True]

        condition        = np.logical_and( mask==True, self.q20_length>25 )
        m_q20[condition] = self.q20_length[condition]

        m_q20_reads      = m_q20.astype( np.bool )

        data = {}

        if self.explog_is_multilane:
            for (i, lane, active) in self.iterlanes():
                if active:
                    data[lane] = {}

                    mask_slice      = self.get_lane_slice( mask.astype(np.int) , i )
                    mread_slice     = self.get_lane_slice( m_reads             , i )
                    mq20_slice      = self.get_lane_slice( m_q20               , i )
                    mq20_read_slice = self.get_lane_slice( m_q20_reads         , i )

                    reads = mread_slice.sum()
                    wells = mask_slice.sum()
                    if wells>0: read_density = reads/ np.float( wells )
                    else:       read_density = 0
                    q20_reads = mq20_read_slice.sum()
                    if q20_reads>0: q20_mean = mq20_slice.sum()/q20_reads
                    else:           q20_mean = 0

                    data[lane].update( {'reads':reads, 
                                        'wells':wells, 
                                        'read_density':read_density, 
                                        'q20_reads':q20_reads, 
                                        'q20_mean':q20_mean } )
        else:
            reads = m_reads.sum()
            wells = mask.astype(np.int).sum()
            if wells>0: read_density = reads/np.float(wells)
            else:       read_density = 0
            q20_reads = m_q20_reads.sum()
            if q20_reads>0: q20_mean = m_q20.sum()/np.float(q20_reads)
            else:           q20_mean = 0

            data.update( {'reads':reads, 
                          'wells':wells, 
                          'read_density':read_density, 
                          'q20_reads':q20_reads, 
                          'q20_mean':q20_mean } )
        return data

    # Pybam output processing
    def pybam_vs_dist_from_stuck( self ):
        ''' Generates reads and q20 length vs distance from stuck beads '''

        print( '\nStarting pybam_vs_dist_from_stuck . . .' )
        
        label    = 'pybam_vs_dist'
        ref_name = 'dist_from_stuck'
        
        data = self.pversus_prep_data_storage( label=label, ref_name=ref_name )

        mask, _ = self.pversus_make_dist_masks( old_mask=None, initial=True )
        # calculate values where stuck beads are
        if not mask.any():
            print( 'No Stuck Beads!!!\n pybam_vs_dist_from_stuck is complete!' )  
            # save the empty dicts
            # NOTE Trim pversus data from saving
            #self.pversus_save( data )
            return
        # store rings as different integer values, plot as one array
        ring_image = np.array( mask.astype(np.int) )

        self.pversus_update_vals( data, 0, mask, 'area', label=label, ref_name=ref_name, ref_cutoff=10 )
        # ring has same mask as area since it's just the stuck bead locations
        self.pversus_update_vals( data, 0, mask, 'ring', label=label, ref_name=ref_name, ref_cutoff=10 )       
        # get values up to 10 wells away from stuck bead
        for i in range( 1, 11 ):
            # store old mask
            old_mask = mask
            # make new mask from old
            mask, ring_mask = self.pversus_make_dist_masks( old_mask=old_mask, dist=i )
            self.pversus_update_vals( data, i, mask     , 'area', label=label, ref_name=ref_name, ref_cutoff=10 )       
            self.pversus_update_vals( data, i, ring_mask, 'ring', label=label, ref_name=ref_name, ref_cutoff=10 )       
            # need to multiply ring_mask by i+1 to keep all vals non-zero
            ring_image += (i+1)*ring_mask.astype(int)
        self.output_image( ring_image, 'Masks -- Distance From Stuck', 'masks_dist_from_stuck', vlims=(0,12,), isspa=False )
        # save all the metrics
        self.pversus_make_plots( data, label=label, ref_name=ref_name )
        # Purge lists from data on save
        # NOTE Trim pversus data from saving
        #self.pversus_save( data )
        print( '. . . pybam_vs_dist_from_stuck is complete!' )

    def pybam_vs_nss_height( self ):
        ''' Generates reads and q20 length vs bins of NSS Int25 values '''       
        print( '\nStarting pybam_vs_nss_height . . .' )

        label    = 'pybam_vs_nss_height'
        ref_name = 'nss_height'
        
        data = self.pversus_prep_data_storage( label=label, ref_name=ref_name )
        # initialize old_mask
        old_mask = None
        # get values up to Int25=100 in steps of 5
        for cutoff in [ 5*i for i in range(1,21)]:
            # make new mask from old
            mask, ring_mask = self.pversus_make_nss_height_masks( cutoff, old_mask=old_mask )
            # store old mask
            old_mask = mask
            self.pversus_update_vals( data, cutoff, mask,      'area', label=label, ref_name=ref_name, ref_cutoff=100 )
            self.pversus_update_vals( data, cutoff, ring_mask, 'ring', label=label, ref_name=ref_name, ref_cutoff=100 )
            # need to multiply ring_mask by i+1 to keep all vals non-zero
            if cutoff == 5:
                ring_image = cutoff*np.array( mask.astype(np.int) )
            else:
                ring_image += cutoff*ring_mask.astype(int)
        self.output_image( ring_image, 'Masks -- NSS Height (steps of 5)', 'masks_nss_height_steps_of_5', vlims=(0,100,), isspa=False )
        # save all the metrics
        # NOTE Trim pversus data from saving
        #self.pversus_save( data )
        self.pversus_make_plots( data, label=label, ref_name=ref_name )
        print( '. . . pybam_vs_nss_height is complete!' )

    def pversus_prep_data_storage( self, label='pybam_vs_dist', ref_name='dist_from_stuck' ):
        ''' Helpfer function for pybam_vs ---
            Preps data dict
        '''
        temp = {'reads':[],
                'wells':[],
                'read_density':[],
                'q20_reads':[],
                'q20_mean':[],}
        area_temp = deepcopy( temp )
        area_temp.update( {'integrated_'+ref_name:[]} )
        ring_temp = deepcopy( temp )
        ring_temp.update( {ref_name:[]} )
        data = {}
        if self.explog_is_multilane:
            for (i, lane, active) in self.iterlanes():
                if active:
                    data.update( {lane:{label:{'area':deepcopy(area_temp),'ring':deepcopy(ring_temp)} } } )
        else:
            data.update( {label:{'area':deepcopy(area_temp),'ring':deepcopy(ring_temp)} } )
        print( 'prepped data storage: ', data )
        return data

    def pversus_make_plots( self, data, label=None, ref_name=None ):
        for dtype in ['area','ring']:
            for y_name in ['read_density', 'q20_mean']:
                if y_name == 'read_density': ylims = (0,0.6,)
                elif y_name == 'q20_mean':   ylims = self.create_flow_based_readlength_scale()
                else:                        ylims = None
                if dtype=='area': x_name = 'integrated_'+ref_name
                else:            x_name = ref_name
                if self.explog_is_multilane:
                    all_xs      = []
                    all_ys      = []
                    all_lanes   = []

                    xlabel = x_name
                    ylabel = y_name

                    for (i, lane, active) in self.iterlanes():
                        if active:
                            x = data[lane][label][dtype][x_name]
                            all_xs.append(x)
                            y = data[lane][label][dtype][y_name]
                            all_ys.append(y)

                            all_lanes.append( lane )

                            title = '{} | {} vs {}\n{}'.format(lane, y_name, x_name, dtype)
                            filename = 'multilane_{}_{}_vs_{}_{}'.format(lane, y_name, x_name, dtype)
                            self.make_fig( x, y, xlabel, ylabel, title, filename, ylims=ylims )
                            self.make_fig( x, y, xlabel, ylabel, title, filename+'_AUTOSCALE' )

                    title = '{} | {} vs {}\n{}'.format( 'All Lanes', y_name, x_name, dtype)
                    filename = 'multilane_{}_{}_vs_{}_{}'.format('all', y_name, x_name, dtype)
                    self.make_fig( all_xs, all_ys, xlabel, ylabel, title, filename, ylims=ylims, legend_list=all_lanes, plot_all=True )
                    self.make_fig( all_xs, all_ys, xlabel, ylabel, title, filename+'_AUTOSCALE', legend_list=all_lanes, plot_all=True )
                else:
                    x = data[label][dtype][x_name]
                    y = data[label][dtype][y_name]
                    xlabel = x_name
                    ylabel = y_name
                    title = '{} | {} vs {}\n{}'.format('FFC', y_name, x_name, dtype)
                    filename = '{}_{}_vs_{}_{}'.format('ffc', y_name, x_name, dtype)
                    self.make_fig( x, y, xlabel, ylabel, title, filename, ylims=ylims )
                    self.make_fig( x, y, xlabel, ylabel, title, filename+'_AUTOSCALE' )
    
    def pversus_update_vals( self, data, ref_val, mask, dtype, label='pybam_vs_dist', ref_name='dist_from_stuck', ref_cutoff=5 ):
        ''' Helper function for pybam_vs --- 
            Stores data as lists vs ref_val
            Stores individual values up to dist 3
        '''
        if dtype == 'area': ref_name = 'integrated_'+ref_name
        # store ref_vals
        if self.explog_is_multilane:
            for (i, lane, active) in self.iterlanes():
                if active:
                    data[lane][label][dtype][ref_name].append(ref_val)
        else:
            data[label][dtype][ref_name].append(ref_val)
        # store everything else
        vals = self.extract_reads_and_q20( mask=mask )
        if self.explog_is_multilane:
            for (i, lane, active) in self.iterlanes():
                if active:
                    for key, val in vals[lane].items():
                        data[lane][label][dtype][key].append(val)
                        if ref_val<=ref_cutoff:
                            data[lane][label][dtype]['{}_at{:02d}'.format(key,ref_val)]=val

        else:
            for key, val in vals.items():
                data[label][dtype][key].append(val)
                if ref_val<=ref_cutoff:
                    data[label][dtype]['{}_at{:02d}'.format(key,ref_val)]=val

    def pversus_make_dist_masks( self, old_mask=None, initial=False, dist=None ):
        ''' Helper function for pybam_vs --- 
            Makes new masks from old_mask
        '''
        if initial:
            # simplify using height
            h = self.nss_height      
            # define stuck mask
            stuck = np.zeros( h.shape )
            stuck[ h<25 ] = h[ h<25 ]
            mask = stuck.astype( bool )
            ring_mask = None
        else:
            #NOTE To properly deal with edge behavior, the following algorithm was developed
            # 1) break TN array into original 100x100 blocks
            #   A) make a new_mask array of zeros in the shape of old mask
            # 2) for each block with a dist > 1 do the following (for dist == 1, no further processing needed -> if bead is off the block, it has to be at least 1 well away)
            #   A) make edge ring with thickness of dist
            #   B) subtract old_mask block from edge ring
            #   C) set all vals < 0 in edge ring to 0
            #   D) dilate old mask block (it hasn't been altered till this point)
            #   E) subtract the edge ring from the dilated old_mask block and save as new mask block
            #   F) set all vals < 0 in new mask block to 0
            # 3) recombine all the new mask blocks into a new mask
            #   A) as new blocks come in, fill the corresponding rows and cols
            # 4) make a ring_mask by subtracting the old_mask from the new mask
            square = ndimage.generate_binary_structure( 2, 2 )
            def block_processing( rows, cols ):
                # makes use of globally defined dist, old_mask, and square
                # will also make use of 100x100 dimensions for block
                old_block = old_mask[ rows[0]:rows[1], cols[0]:cols[1] ].astype( np.int )
                if dist <= 1:
                    return morph.binary_dilation( old_block, square )
                else:
                    edge = np.zeros( old_block.shape )
                    edge[:dist,:] = 1
                    edge[:,:dist] = 1
                    edge[100-dist:101,:] = 1
                    edge[:,100-dist:101] = 1
                    edge = edge - old_block
                    edge[edge<0] = 0
                    new_block = morph.binary_dilation( old_block, square )
                    new_block = new_block - edge
                    new_block[new_block<0] = 0
                    return new_block
            mask = np.zeros( old_mask.shape )
            for i in [ 100*x for x in range(7) ]:
                for j in [ 100*x for x in range(11) ]:
                    rows = (i,i+99,)
                    cols = (j,j+99,)
                    mask[ rows[0]:rows[1], cols[0]:cols[1] ] = block_processing( rows, cols )
            criteria = np.logical_and( mask, self.filled_active )
            mask[~criteria] = np.zeros(mask.shape)[~criteria]
            mask = mask.astype(np.bool)
            ring_mask = (mask.astype(np.int) - old_mask.astype(np.int)).astype(np.bool)
        return (mask, ring_mask,)

    def pversus_make_nss_height_masks( self, cutoff, old_mask=None ):
        ''' Helper function for pybam_vs --- 
            Makes new masks from old_mask
        '''
        h = self.nss_height      
        mask = np.zeros( h.shape )
        mask[ h<cutoff ] = h[ h<cutoff ]
        mask = mask.astype( bool )
        if old_mask is None:
            # simplify using height
            # define stuck mask
            ring_mask = None
            return ( mask, ring_mask, )
        else:
            criteria = np.logical_and( mask, self.filled_active )
            mask[~criteria] = np.zeros(mask.shape)[~criteria]
            ring_mask = (mask.astype(np.int) - old_mask.astype(np.int)).astype(np.bool)
            return (mask, ring_mask,)

    def pversus_save( self, data ):
        ''' Helper fuction for pybam_vs ---
            Stores data in self.metrics
        '''
        subnames =  ['reads','wells','read_density','q20_reads','q20_mean']
        subnames += ['integrated_dist_from_stuck', 'dist_from_stuck']
        subnames += ['integrated_nss_height', 'nss_height']
        
        def trim( t, d ):
            # remove lists
            for k, items in d.items():
                for k2, items2 in items.items():
                    for s in subnames:
                        try:
                            _ = temp[k][k2].pop(s)
                        except KeyError:
                            continue
            return temp

        if self.explog_is_multilane:
            for (i, lane, active) in self.iterlanes():
                if active:
                    temp = deepcopy( data[lane] )
                    self.metrics[lane].update( trim(temp,data[lane]) )
        else: 
            temp = deepcopy( data )
            self.metrics.update( trim(temp,data) )

#####################################
#       Buffering Analysis          #
#####################################

    #TODO: This needs work.  Can't figure out why it isn't running yet.
    def bf_processing( self, isspa=True ):
        ''' Extract images and metrics from Beadfind 0001 height and slope arrays '''
        warnings.filterwarnings('error')
        print( '\nStarting bf_height processing . . .' )
        if isspa:   image = self.bf_0001_height_spa
        else:       image = self.bf_0001_height
        if self.explog_is_multilane:
            self.extract_and_plot_lane_metrics( image, 'bf_0001_step_height', 'BF 0001 Height', 'Height', isspa=isspa )
        else:
            self.extract_and_plot_ffc_metrics ( image, 'bf_0001_step_height', 'BF 0001 Height', 'Height', isspa=isspa )
        print( '. . . bf_height processing is complete.' )

        print( '\nStarting bf_slope processing . . .' )
        if isspa:   image = self.bf_0001_slope_spa
        else:       image = self.bf_0001_slope
        if self.explog_is_multilane:        
            self.extract_and_plot_lane_metrics( image, 'bf_0001_slope', 'BF 0001 Slope', 'Slope', isspa=isspa )
        else:
            self.extract_and_plot_ffc_metrics ( image, 'bf_0001_slope', 'BF 0001 Slope', 'Slope', isspa=isspa )
        print( '. . . bf_slope processing is complete.' )
        warnings.filterwarnings('default')

#####################################
#       Nuc Quality Analysis        #
#####################################

    def nuc_spikes( self ):
        def find_G_traces():
            flow_order  = self.explog.metrics['flow_order']
            flow_order  = [ f.lower() for f in flow_order ]
            lo          = len(flow_order)
            num_flows   = self.explog.flows
            targets     = [50*(i+1) for i in range(int(num_flows/50))]

            aqs = []
            for t in targets:
                found = False
                look  = t%lo
                shift = 1
                while not found:
                    if look - shift < 0: 
                        look + lo
                    if flow_order[ look - shift ] == 'g':
                        aqs.append( t-shift )
                        found = True
                    shift += 1
            return aqs

        def raw_trace( raw, dtype='bead'):
            f = 'aq_{:04d}_avg_trace_{}'
            return self._get_attr( f.format(raw, dtype), isspa=False )

        # Determine which flows are G and make list
        gtraces = find_G_traces()
        for g in gtraces:
            self.load_datfile( g, isspa=False, ftype='aq' )

        def flow_metrics( data, tn_blockR, tn_blockC ):
            print( 'data shape {}'.format( data.shape ) )
            temp_d   = data[tn_blockR][tn_blockC]
            peak_height  = temp_d.max()
            try:    loc    = int( min( np.where( temp_d==peak_height )[0] ) )
            except: loc    = None

            try:    plateau = temp_d[-20:].mean()
            except: plateau = None

            try:    shift        = temp_d - plateau
            except: shift        = None
            
            try:    spike_height = np.max(shift)
            except: spike_height = None

            try:    
                test = shift-(spike_height/2.)
                fwhm = len( np.where( test>0 )[0] )
            except: 
                fwhm = None

            return {'peak_height':peak_height, 'loc':loc, 'spike_height':spike_height, 'spike_fwhm':fwhm, 'plateau':plateau }

        def plot_spike_metric( spkd, metric='spike_height' ):
            xlabel      = 'Avg Block {} Idx (inlet->outlet)'
            if 'fwhm' in metric or 'loc' in metric:    
                unit='frames'
            else:                   
                unit='counts'
            ylabel      = '{} ({})'.format(metric, unit)
            title       = '{}\nvs. Avg. Block {} Idx'.format(metric, '{}')
            filename    = 'spikes__{}_vs_avg_block_idx.png'.format(metric)

            if not self.explog_is_multilane:
                xlabel = xlabel.format('Col')
                title = title.format('Col')

                flows  = sorted( spkd.keys() )
                x = []
                y = []
                l = []
                for f in flows:
                    xi, xticks, yi = zip( *[ (i,k,v[metric],) for i,(k,v,) in enumerate( sorted( spkd[f].items() ) ) ] )
                    x.append( xi )
                    y.append( yi )
                    l.append( f )
                self.make_fig( x, y, xlabel, ylabel, title, filename, legend_list=l, plot_all=True )

            #if lane is not None: 
            #    pd.reverse()
            #    xlabel  = xlabel.format( 'Row' )
            #    title   = 'Lane {}\n'.format(lane) + title.format( 'Row' )
            #    filename = 'lane_{}_'.format(lane) + filename
            

        # prepend G 0mer raw trace
        gtraces = [3] + gtraces
        # extract spike data
        spike_data = {}
        for g in gtraces:
            tk = 'G{:04d}'.format(g)
            data = raw_trace( g, dtype='bead' )
            if self.explog_is_multilane:
                for (i, lane, active) in self.iterlanes():
                    print( 'processing lane {} -- {}'.format( i, active ) )
                    if active:
                        try:
                            spike_data[lane].update( {tk:{}} )
                        except KeyError:
                            spike_data[lane] = {tk:{}}
                        tn_blockC = i*3 - 2
                        for tn_blockR in range(8):
                            loc = 'bR_{:02d}'.format(tn_blockR)
                            spike_data[lane][tk][loc] = flow_metrics( data, tn_blockR, tn_blockC )
            else:
                tn_blockR = 3
                spike_data.update( {tk:{}} )
                for tn_blockC in range(12):
                    loc = 'bC_{:02d}'.format(tn_blockC)
                    spike_data[tk][loc] = flow_metrics( data, tn_blockR, tn_blockC )
        self.metrics['spike_data'] = spike_data
        plot_spike_metric( spike_data, metric='spike_height' )
        plot_spike_metric( spike_data, metric='plateau' )
        plot_spike_metric( spike_data, metric='peak_height' )
        plot_spike_metric( spike_data, metric='loc' )
        plot_spike_metric( spike_data, metric='spike_fwhm' )

    def nuc_quality_check( self ):
        ''' records the step height and relative peak pH time of the 0-mer nuc steps 
            
            0-mer flows are
            1:A, 3:G, 4:T, and 6:C

            1-mer flows are
            0:T, 2:C, 5:A, and 7:G
        '''

        def raw_trace( raw, dtype='bead', prerun=False ):
            if prerun:  f = 'pr_{:04d}_avg_trace_{}'
            else:       f = 'aq_{:04d}_avg_trace_{}'
            return self._get_attr( f.format(raw, dtype),   isspa=False )

        def raw_diff( raw, prerun=False ):
            rb = raw_trace( raw, 'bead',  prerun=prerun )
            re = raw_trace( raw, 'empty', prerun=prerun )
            return (rb - re)

        def nuc_diff( onemer, zeromer ):
            f = 'aq_{:04d}_avg_trace_{}'
            odiff = raw_diff( onemer )
            zdiff = raw_diff( zeromer )
            return odiff - zdiff

        def nuc_dict( onemer, zeromer ):
            return {'key':nuc_diff( onemer, zeromer ), 
                    'one':raw_diff(onemer), 
                    'zero':raw_diff(zeromer), 
                    'raw_bead_one':   raw_trace( onemer,  dtype='bead'),
                    'raw_bead_zero':  raw_trace( zeromer, dtype='bead'),                  
                    'raw_empty_one':  raw_trace( onemer,  dtype='empty'),
                    'raw_empty_zero': raw_trace( zeromer, dtype='empty'),
                    }
        def prerun_dict( fnum ):
            return {    'prerun_bead':  raw_trace( fnum, dtype='bead',  prerun=True ),
                        'prerun_empty': raw_trace( fnum, dtype='empty', prerun=True ),
                        'prerun_diff':  raw_diff( fnum, prerun=True ),
                        }


        print( 'calculating nuc_diffs' )
        data = {
                'G': nuc_dict( 7, 3 ),
                'C': nuc_dict( 2, 6 ),
                'A': nuc_dict( 5, 1 ),
                'T': nuc_dict( 0, 4 ),
            }
        data.update( {'p{}'.format(i):prerun_dict(i) for i in range(8)} )

        nuc_keys = ['G','C','A','T']

        def flow_metrics( nkey, dtype, tn_blockR, tn_blockC, find_min=False ):
            print( 'data shape {}'.format( data[nkey][dtype].shape ) )
            temp_nkey   = data[nkey][dtype][tn_blockR][tn_blockC]
            height  = temp_nkey.max()
            depth   = temp_nkey.min()
            if find_min:
                try:    peak_loc    = int( min( np.where( temp_nkey==depth )[0] ) )
                except: peak_loc    = None

                try:    shift       = temp_nkey - depth/2.
                except: shift       = None

                try:    fwhm        = len( np.where(shift<0)[0] )
                except: fwhm        = None
            else:
                try:    peak_loc    = int( min( np.where( temp_nkey==height )[0] ) )
                except: peak_loc    = None

                try:    shift       = temp_nkey - height/2.
                except: shift       = None

                try:    fwhm        = len( np.where(shift>0)[0] )
                except: fwhm        = None

            return (temp_nkey, {'height':height, 'depth':depth, 'peak_loc':peak_loc, 'fwhm':fwhm },)

        def process( tn_blockR, tn_blockC, loc, lane=None ):
            temp_nucs = {}

            rs = 100*tn_blockR
            re = 100*(tn_blockR + 1)
            cs = 100*tn_blockC
            ce = 100*(tn_blockC + 1)

            bead_slice = np.array( self.bead[rs:re,cs:ce] )
            empty_slice = np.array( self.empty[rs:re,cs:ce] )

            print( '\n bead slice {}'.format( bead_slice.shape ) )
            print( '\n empty slice {}'.format( empty_slice.shape ) )

            num_beads   = bead_slice.sum()
            num_empty   = empty_slice.sum()

            temp_nucs = {   'num_beads': num_beads,
                            'num_empty': num_empty,
                            }

            title           =   '{} {dtype}\n'\
                                'Averaged over Thumbnail Block [r({}:{}), c({}:{})]\n'\
                                '({} beads, {} empty)'.format( loc,rs,re,cs,ce,num_beads,num_empty, dtype='{}')
            filename        =   '{}_{dtype}_flows.png'.format( loc, dtype='{}' )
            fname_ascale    =   '{}_{dtype}_flows_AUTOSCALED.png'.format( loc, dtype='{}' )


            if lane is not None:
                title    = 'Lane {} '.format(lane) + title 
                filename = 'lane_{}_'.format(lane) + filename

            def mpd( dname, tname, fkey ):
                return { dname : {  'data'          :[],
                                    'title'         : title.format(tname),
                                    'filename'      : filename.format(fkey),
                                    'fname_ascale'  : fname_ascale  } }

            temp_plot_info = {}
            temp_plot_info.update( mpd( 'key',  'Key (1mer-0mer)\nBeads - Empties', 'key'   ) )
            temp_plot_info.update( mpd( 'one',  'Key 1mer\nBeads - Empties',        '1mer'  ) )
            temp_plot_info.update( mpd( 'zero', 'Key 0mer\nBeads - Empties',        '0mer'  ) )
            temp_plot_info.update( mpd( 'raw_bead_zero',    'Raw Bead 0mer Flows',    'raw_bead_0mer' ) )
            temp_plot_info.update( mpd( 'raw_bead_one',     'Raw Bead 1mer Flows',    'raw_bead_1mer' ) )
            temp_plot_info.update( mpd( 'raw_empty_zero',   'Raw Empty 0mer Flows',   'raw_empty_0mer') )
            temp_plot_info.update( mpd( 'raw_empty_one',    'Raw Empty 1mer Flows',   'raw_empty_1mer' ) )                    
            temp_plot_info.update( mpd( 'prerun_bead',      'Pre-Run Bead Flows',     'prerun_bead' ) )                    
            temp_plot_info.update( mpd( 'prerun_empty',     'Pre-Run Empty Flows',    'prerun_empty' ) )                    
            temp_plot_info.update( mpd( 'prerun_diff',      'Pre-Run Diff\nBeads - Empties',     'prerun_diff' ) )                    

            key_height      = []
            key_depth       = []
            key_peak_loc    = []
            key_fwhm        = []
            for nkey in nuc_keys:
                temp_nucs[nkey] = {}
                for dtype in ['key','one','zero','raw_bead_zero', 'raw_bead_one', 'raw_empty_zero', 'raw_empty_one']:
                    if dtype == 'zero':
                        sig, flow = flow_metrics( nkey, dtype, tn_blockR, tn_blockC, find_min=True )
                    else:
                        sig, flow = flow_metrics( nkey, dtype, tn_blockR, tn_blockC )

                        if dtype == 'key':
                            key_height.append(      flow['height'] )
                            key_depth.append(       flow['depth'] )
                            key_peak_loc.append(    flow['peak_loc'] )
                            key_fwhm.append(        flow['fwhm'] )

                    temp_plot_info[dtype]['data'].append( [ [t for t in range(len(sig))],  sig,    nkey, ] )

                    # NOTE Trim metrics that are being stored
                    if 'raw' in dtype:
                        _ = flow.pop( 'depth' )
                        _ = flow.pop( 'fwhm' )
                        _ = flow.pop( 'peak_loc' )
                    elif dtype=='zero':
                        _ = flow.pop( 'height' )
                        _ = flow.pop( 'peak_loc' )
                        _ = flow.pop( 'fwhm' )
                    elif dtype=='one':
                        continue
                    temp_nucs[nkey][dtype] = flow

            temp_nucs.update( { 'rel_diff_key_height'       : self.diff_calc( key_height ),
                                'rel_diff_key_depth'        : self.diff_calc( key_depth ),
                                'rel_diff_key_peak_loc'     : self.diff_calc( key_peak_loc,  rel_min=True ),
                                'raw_diff_key_peak_loc'     : self.diff_calc( key_peak_loc,  raw=True ),
                                'rel_diff_key_fwhm'         : self.diff_calc( key_fwhm,      rel_min=True ),
                                'raw_diff_key_fwhm'         : self.diff_calc( key_fwhm,      raw=True ),
                                'scaled_rel_diff_key_fwhm'  : self.diff_calc( key_fwhm,      rel_min=True, height_refs=key_height ),
                                            } )

            for fnum in range(8):
                pkey = 'p{}'.format(fnum)
                temp_nucs[pkey] = {}
                for dtype in ['prerun_bead', 'prerun_empty', 'prerun_diff']:
                    sig, flow = flow_metrics( pkey, dtype, tn_blockR, tn_blockC, find_min=True )
                    temp_plot_info[dtype]['data'].append( [ [t for t in range(len(sig))],  sig,     pkey, ] )
                    # NOTE Only store data for prerun_diff
                    if dtype == 'prerun_diff':
                        temp_nucs[pkey][dtype] = flow

            return {'nucs':{loc:temp_nucs}, 'plot_info':{loc:temp_plot_info} }


        def plot_flows( ldata, ylims, title, filename, nuc_colors=True ):
            x,y,legend_list = zip(*ldata)
            self.make_fig( x, y, 'Frames', 'Counts', title, filename, legend_list=legend_list, 
                            plot_all=True, nuc_colors=nuc_colors, figsize=(6,6,), ylims=ylims )

        plot_info = {}
        if self.explog_is_multilane:
            for (i, lane, active) in self.iterlanes():
                print( 'processing lane {} -- {}'.format( lane, active ) )
                if active:
                    tn_blockC = i*3 - 2
                    nucs = {}
                    plot_info[lane] = {}
                    for tn_blockR in range(8):
                        loc = 'bR_{}'.format(tn_blockR)
                        out = process( tn_blockR, tn_blockC, loc, lane=i )
                        plot_info[lane].update( out['plot_info'] )
                        #NOTE: Only keep data for a limited number of blocks
                        if tn_blockR in (1,4,6,):
                            nucs.update( out['nucs'] )
                        
                    # Store metrics for a given lane
                    self.metrics[lane]['nucs'] = nucs

            # set up plotting limits across lanes and blocks
            lane_ylims = { x:[0,0] for x in ['key','one','zero','raw_bead_zero','raw_bead_one',
                'raw_empty_zero','raw_empty_one', 'prerun_bead','prerun_empty','prerun_diff' ] }
            
            for lane, litems in plot_info.items():
                for loc, bitems in litems.items():
                    for dtype, info in bitems.items():
                        clim        = lane_ylims[dtype]
                        _, y, _     = zip(*info['data'])
                        nlim        = self.get_lims( y )
                        if (clim[0] is not None) and (nlim[0] is not None) and (clim[0] > nlim[0]):  
                            lane_ylims[dtype][0] = nlim[0]
                        if (clim[1] is not None) and (nlim[1] is not None) and (clim[1] < nlim[1]): 
                            lane_ylims[dtype][1] = nlim[1]

            for lane, litems in plot_info.items():
                for loc, bitems in litems.items():
                    for dtype, info in bitems.items():
                        if 'prerun' in dtype:   nuc_colors = False
                        else:                   nuc_colors = True
                        plot_flows( info['data'], lane_ylims[dtype], info['title'], info['filename'], nuc_colors=nuc_colors )
                        # Autoscale plots
                        plot_flows( info['data'], [None,None,], info['title']+'\nAUTOSCALE', info['fname_ascale'], nuc_colors=nuc_colors )

        else:
            tn_blockR = 3
            nucs = {}
            plot_info = {}
            for tn_blockC in range(12):
                loc = 'bC_{}'.format(tn_blockC)
                out = process( tn_blockR, tn_blockC, loc, lane=None )
                plot_info.update( out['plot_info'] )
                #NOTE Only keep metrics for a limited number of blocks
                if tn_blockC in (1,4,7,10,): 
                    nucs.update( out['nucs'] )
                
            # Store metrics
            self.metrics['nucs'] = nucs

            # set up plotting limits across blocks
            ffc_ylims = { x:[0,0] for x in ['key','one','zero','raw_bead_zero','raw_bead_one',
                'raw_empty_zero','raw_empty_one', 'prerun_bead','prerun_empty','prerun_diff' ] }
            
            for loc, bitems in plot_info.items():
                for dtype, info in bitems.items():
                    clim        = ffc_ylims[dtype]
                    _, y, _     = zip(*info['data'])
                    nlim        = self.get_lims( y )
                    if (clim[0] is not None) and (nlim[0] is not None) and (clim[0] > nlim[0]): 
                        ffc_ylims[dtype][0] = nlim[0]
                    if (clim[1] is not None) and (nlim[1] is not None) and (clim[1] < nlim[1]): 
                        ffc_ylims[dtype][1] = nlim[1]

            for loc, bitems in plot_info.items():
                for dtype, info in bitems.items():
                    if 'prerun' in dtype:   nuc_colors = False
                    else:                   nuc_colors = True
                    plot_flows( info['data'], ffc_ylims[dtype], info['title'], info['filename'], nuc_colors=nuc_colors )
                    # Autoscale plots
                    plot_flows( info['data'], ffc_ylims[dtype], info['title']+'\nAUTOSCALE', info['fname_ascale'], nuc_colors=nuc_colors )

    def relative_peak_spatial( self ):
        ''' generates a spatial plot of 1mer flow - 0mer flow peak heights of one nuc 
            relative to all 
            
            No bead/empty filtering --> just 1mer - 0mer flow with max recorded

            Note: ignore values < 0
        '''
        print( '--- STARTING relative_peak_spatial --- ' )

        aq_dfs = []
        base_path = 'acq_{:04d}.dat_spa'
        for i in range(8):
            path = os.path.join( self.acq_dir, base_path.format(i) )
            aq_dfs.append( datfile.DatFile( filename=path, norm=True, 
                            fcmask=False, dr=self.explog.DR, chiptype=self.ct.tn_spa ).data )


        # 0-mer flows are
        # 1:A, 3:G, 4:T, and 6:C
        #
        # 1-mer flows are
        # 0:T, 2:C, 5:A, and 7:G

        # Get the max height of each trace for 1mer-0mer
        nucs = {}
        nucs['G'] = np.nan_to_num(aq_dfs[7]-aq_dfs[3]).max(2)
        nucs['C'] = np.nan_to_num(aq_dfs[2]-aq_dfs[6]).max(2)
        nucs['A'] = np.nan_to_num(aq_dfs[5]-aq_dfs[1]).max(2)
        nucs['T'] = np.nan_to_num(aq_dfs[0]-aq_dfs[4]).max(2)

        # For later iterations since you can't iterate AND change a dict at the same time
        nuc_keys = list( nucs.keys() )

        # clean up aq_dfs (don't need anymore)
        del aq_dfs

        # trim to just active pixels and set negatives from g,c,a,t to 0
        filled_active = self._get_attr( 'filled_active', True )
        for n in nuc_keys:
            nucs[n][~filled_active] = 0
            temp = nucs[n].copy()
            temp[ temp<0 ] = 0
            nucs[n] = temp 

        def make_mask( x ):
            mask = np.zeros( x.shape )
            mask[x<=0] = 1
            return mask.astype( bool )

        means   = {}
        stds    = {}
        for n in nuc_keys:
            block = BlockReshape( nucs[n], self.blocksize, make_mask( nucs[n] ) )
            means[n] = block.get_mean()
            stds[n]  = block.get_std()

        # normalize to percentile of mean values
        test = []
        for n in nuc_keys:
            temp = means[n]
            test += list( temp[temp>0] )
        scale = 100./np.percentile( np.array( test ), 90 )

        for n in nuc_keys:
            # means
            temp        = means[n]
            means[n]    = scale*temp
            # stds
            temp        = stds[n]
            stds[n]     = scale*temp

        print( 'Generating metrics and multilane figs' )
        # Generate metrics
        if self.explog_is_multilane:
            for n in nuc_keys:
                self.extract_and_plot_lane_metrics( nucs[n], '{}_norm_max_diff'.format(n), '{} Norm. Max 1mer-0mer'.format(n), 'Norm. Max 1mer-0mer', clims=(0,100,), local_clims=(0,100,), localstd_clims=(0,50,), isspa=True, integral_cutoffs=[10,25,50,75], scale=scale )
        else:
            self.extract_and_plot_ffc_metrics( nucs[n], '{}_norm_max_diff'.format(n), '{} Norm. Max 1mer-0mer'.format(n), 'Norm. Max 1mer-0mer', clims=(0,100,), local_clims=(0,100,), localstd_clims=(0,50,), isspa=True, integral_cutoffs=[10,25,50,75], scale=scale )

        print( 'Generating overview figures' )
        # Generate Figures
        for data, label in zip( [means, stds], ['Heights', 'Variability'] ):

            # make output figure
            fig, axs = plt.subplots( 2, 2 )
            vmax = 100
            if label=='Variability':
                vmax = 50
            # Make this one first to leverage for the colorbar
            # NOTE: Need T,C,A,G flow ordering
            for ax, n in zip( axs.flat, ['T','C','A','G'] ) :
                im = ax.imshow( data[n], vmin=0, vmax=vmax, origin='lower', interpolation='nearest' )
                ax.set_title( '{} 1mer-0mer'.format( n ) )
                ax.get_xaxis().set_visible( False )
                ax.get_yaxis().set_visible( False )

            fig.subplots_adjust( bottom=0.05, top=0.85, left=0.05, right=0.95 )
            #fig.colorbar( im, cax=fig.add_axes( [0.85,0.15,0.05,0.7] ), shrink=0.6 )
            fig.colorbar( im, ax=axs[:,:], location='right', shrink=0.6 )

            plt.suptitle( 'Spatial 1mer-0mer Key Flow {}\nNormalized to 90th Percentile Height\nArranged by Incorporation Order (T-C-A-G)\n'.format(label) )
            filename = 'spatial_relative_keys{}.png'
            if label == 'Heights':
                filename = filename.format( '' )
            else:
                filename = filename.format( '_'+label )
            fig.savefig( os.path.join( self.results_dir, filename ) )  # save the figure to file
            plt.close()


#########################################
#       METRIC & PLOTTING TOOLS         #
#########################################


    def diff_calc( self, a, rel_min=False, raw=False, height_refs=None ):
        try:        
            if raw:
                d = max(a) - min(a)
            elif height_refs is not None:
                temp = [ float(x)/float(y) for x,y in zip(a,height_refs) if (x is not None) and (y is not None) and (y > 0) ]
                if rel_min:
                    d = 100*float( max(temp) - min(temp) )/float(min(temp))
                else:
                    d = 100*float( max(temp) - min(a))/float(max(temp))
            elif rel_min:
                d = 100*float( max(a) - min(a))/float(min(a))
            else:
                d = 100*float( max(a) - min(a))/float(max(a))
        except:
            d = None
        return d

    def extract_and_plot_ffc_metrics( self, metric_array, metric_name, title, xlabel, units=None, clims=None, local_clims=None, localstd_clims=None, bin_scale=2, isspa=False, do_local=True, do_localstd=True, integral_cutoffs=None, extreme_high=None, mask=None, scale=None ):
        ''' Generically makes regular and local metrics, along with associated plots '''
        print( 'Getting metrics and plots for {}'.format( metric_name ) )

        fname               = 'ffc_{}_density.png'
        local_fname         = 'ffc_{}_density_local.png'        
        localstd_fname      = 'ffc_{}_density_localstd.png'

        if isspa:
            title       += ' -- Spa'
            metric_name += '_spa'

        if mask is None:
            mask = np.logical_not( self._get_attr( 'filled_active', isspa ) )

        if scale is not None:
            metric_array = scale*metric_array

        base_ffc_block   = BlockReshape( metric_array , self.blocksize, mask )
        ffc              = FullFlowcellPlot( metric_array,  title+' (FULL)' ,    xlabel, units=units,   clims=clims ,           bin_scale=bin_scale )
        ffc.plot_heat_and_hist           ( os.path.join( self.results_dir , fname.format( metric_name )          ) )
        self.metrics[metric_name]                 = self.get_metrics( metric_array,  lane_number=None, lower_lim=0, add_vmr=True, integral_cutoffs=integral_cutoffs, extreme_high=extreme_high )#[10,25,50,75] )
        
        if do_local:
            ffc_block        = base_ffc_block.get_mean()
            ffc_local        = FullFlowcellPlot( ffc_block ,    title+' (local)' ,   xlabel, None,          clims=local_clims,      bin_scale=bin_scale )
            ffc_local.plot_heat_and_hist     ( os.path.join( self.results_dir , local_fname.format( metric_name )    ) )
            self.metrics[metric_name]['local']        = self.get_metrics( ffc_block,     lane_number=None, lower_lim=0, add_vmr=True )
            try:
                # std_ratio is the local mean std (variability of local means) divided by the global std (global variability)
                std_ratio = self.metrics[metric_name]['local']['std'] / self.metrics[metric_name]['std']
                self.metrics[metric_name]['std_ratio'] = std_ratio
            except TypeError:
                self.metrics[metric_name]['std_ratio'] = None
            except KeyError:
                print( 'no {} metrics'.format( metric_name ) )

        if do_localstd:
            ffc_block_std    = base_ffc_block.get_std()
            ffc_localstd     = FullFlowcellPlot( ffc_block_std, title+' (localstd)', xlabel, None,          clims=localstd_clims,   bin_scale=bin_scale )
            ffc_localstd.plot_heat_and_hist  ( os.path.join( self.results_dir , localstd_fname.format( metric_name ) ) )
            self.metrics[metric_name]['localstd']     = self.get_metrics( ffc_block_std, lane_number=None, lower_lim=0, add_vmr=True )

    def extract_and_plot_lane_metrics( self, metric_array, metric_name, title, xlabel, units=None, clims=None, local_clims=None, localstd_clims=None, bin_scale=2, isspa=False, do_local=True, do_localstd=True, integral_cutoffs=None, extreme_high=None, mask=None, scale=None ):
        ''' Generically makes regular and local metrics, along with associated plots '''
        print( 'Getting metrics and plots for {}'.format( metric_name ) )

        all_fname               = 'multilane_all_{}_density'
        all_local_fname         = 'multilane_all_{}_density_local'        
        all_localstd_fname      = 'multilane_all_{}_density_localstd'

        lane_fname              = 'multilane_lane_{}_{}_density'
        lane_local_fname        = 'multilane_lane_{}_{}_density_local'
        lane_localstd_fname     = 'multilane_lane_{}_{}_density_localstd'

        if isspa:
            title       += ' -- Spa'
            metric_name += '_spa'

        all_fname           += '.png'
        all_local_fname     += '.png'
        all_localstd_fname  += '.png'
        lane_fname          += '.png'
        lane_local_fname    += '.png'
        lane_localstd_fname += '.png'

        if mask is None:
            mask = np.logical_not( self._get_attr( 'filled_active', isspa ) )

        if scale is not None:
            metric_array = scale*metric_array

        base_mp_block   = BlockReshape( metric_array , self.blocksize, mask )
        mp              = MultilanePlot( metric_array, title+' (FULL)' , xlabel , units , clims=clims , bin_scale=bin_scale )
        mp.plot_all( os.path.join( self.results_dir , all_fname.format( metric_name ) ) )

        if do_local:    
            mp_block        = base_mp_block.get_mean()
            mp_local        = MultilanePlot( mp_block ,     title+' (local)'    , xlabel , None , clims=local_clims   , bin_scale=bin_scale )
            mp_local.plot_all   ( os.path.join( self.results_dir , all_local_fname.format   ( metric_name ) ) )
        if do_localstd: 
            mp_block_std    = base_mp_block.get_std()
            mp_localstd     = MultilanePlot( mp_block_std , title+' (localstd)' , xlabel , None , clims=localstd_clims, bin_scale=bin_scale )
            mp_localstd.plot_all( os.path.join( self.results_dir , all_localstd_fname.format( metric_name ) ) )

        for (i, lane, active) in self.iterlanes():
            if active:
                mp.plot_one( i , os.path.join( self.results_dir , lane_fname.format(i, metric_name) ) )
                self.metrics[lane][metric_name] = self.get_metrics( metric_array, lane_number=i, lower_lim=0, add_vmr=True, integral_cutoffs=integral_cutoffs, extreme_high=extreme_high )#[10,25,50,75] )
                if do_local:
                    mp_local.plot_one( i , os.path.join( self.results_dir , lane_local_fname.format(i, metric_name) ) )
                    self.metrics[lane][metric_name]['local'] = self.get_metrics( mp_block,     lane_number=i, lower_lim=0, add_vmr=True )
                    try:
                        # std_ratio is the local mean std (variability of local means) divided by the global std (global variability)
                        std_ratio = self.metrics[lane][metric_name]['local']['std'] / self.metrics[lane][metric_name]['std']
                        self.metrics[lane][metric_name]['std_ratio'] = std_ratio
                    except TypeError: 
                        self.metrics[lane][metric_name]['std_ratio'] = None
                    except KeyError:
                        print( 'no {} metrics for {}'.format(metric_name, lane) )
                if do_localstd:
                    mp_localstd.plot_one( i , os.path.join( self.results_dir , lane_localstd_fname.format(i, metric_name) ) )
                    self.metrics[lane][metric_name]['localstd'] = self.get_metrics( mp_block_std, lane_number=i, lower_lim=0, add_vmr=True )

    def get_lims( self, y ):
        # default ylims
        ylims = [None, None,]

        ay = np.array(y)

        test = np.nanmax( ay )
        if   test < 10:     ylims[1] = 11
        elif test < 20:     ylims[1] = 22
        elif test < 50:     ylims[1] = 55
        elif test < 100:    ylims[1] = 110
        elif test < 200:    ylims[1] = 220
        elif test < 500:    ylims[1] = 550
        elif test < 1000:   ylims[1] = 1100
        elif test < 8000:   ylims[1] = 8000
        else:               ylims[1] = None
        
        test = np.nanmin( ay )
        if   test > -10:    ylims[0] = -11
        elif test > -20:    ylims[0] = -22
        elif test > -50:    ylims[0] = -55
        elif test > -100:   ylims[0] = -110
        elif test > -200:   ylims[0] = -220
        elif test > -500:   ylims[0] = -550
        elif test > -1000:  ylims[0] = -1100
        elif test > -8000:  ylims[0] = -8000
        else:               ylims[0] = None

        return ylims

    def make_fig( self, x, y, xlabel, ylabel, title, filename, ylims=None, legend_list=None, plot_all=False, nuc_colors=False, figsize=(12,8,) ):
        # Output figure
        fig = plt.figure(figsize=figsize)
        if plot_all:
            cmap = plt.get_cmap( 'jet' )
            colors = [cmap(i) for i in np.linspace(0,1,len(x))]
            for i, (xi, yi, li,) in enumerate( zip(x,y,legend_list) ): 
                if nuc_colors:
                    # Nucs GCAT coloring --> G:k, C:b, A:g, T:r
                    nc = {'G':'k','C':'b','A':'g','T':'r'}
                    plt.plot( xi, yi, nc[li], label=li )
                    plt.legend(loc='best', ncol=1)
                else:
                    plt.plot( xi, yi, label=li, color=colors[i] )
                    plt.legend(loc='best', ncol=2)
        else:
            plt.plot( x, y )
        plt.xlabel( xlabel )
        plt.ylabel( ylabel )
        plt.title ( title  )
        plt.ylim( ylims )
        plt.tight_layout()
        fig.savefig( os.path.join( self.results_dir, filename) )   # save the figure to file
        plt.close()

    def output_image( self, data, title, filename, vlims=None, isspa=False ):      
        # Adjust title and filename dependent upon spa
        if isspa:
            title    += ' -- Spa'
            filename += '_spa'
        filename += '.png'

        print( 'Outputting image {}'.format( title ) )

        # Output figure
        fig = plt.figure(figsize=(12,8))
        if vlims:
            if vlims[0]: vmin = vlims[0]
            else:        vmin = None
            if vlims[1]: vmax = vlims[1]
            else:        vmax = None
        else:
            vmin = vmax = None
        plt.imshow(data, vmin=vmin, vmax=vmax, origin = 'lower', interpolation='none')
        plt.colorbar()
        plt.title(title)
        fig.savefig( os.path.join( self.results_dir, filename) )   # save the figure to file
        plt.close()

#################################
#       PLUGIN DISPLAY          #
#################################

    def write_block_html( self ):
        '''Writes html file that is output to main report screen.'''
        html = os.path.join( self.results_dir, 'NucStepSpatialV2_block.html' )

        width = '300px'
        img_width = '300px'
        nuc_img_width = '200px'

        block = '''
        <html><head><title>NucStepSpatialV2</title></head>
        <body>
        <style type="text/css">table                {border=collapse: collapse;}</style>
        <style type="text/css">tr:nth-child(even)   {background-color: #DDD;}</style>
        <style type="text/css">td                   {border: 1px solid black; text-align: center; }</style>
        '''
        
        block += '''<p> As of version 1.3.0, the normalization algorithm has been updated to 
                        account for gain variation as well as extreme outliers (pinned low/high) </p>
                    <p> As of version 1.6.0, the Rel Diff FWHM calculation is max-min/min </p>
                '''

        if not self.explog_is_multilane:
            # Beadsticking Images
            imgs = [ 'ffc_nss_step_height_spa_density.png',
                     'ffc_read_density_vs_nss_height_ring_AUTOSCALE.png',
                     'ffc_read_density_vs_dist_from_stuck_ring_AUTOSCALE.png'
                    ]
            for img in imgs:
                block += '''
                <a href="{0}"><img src="{0}" width="{w}" /></a> 
                '''.format( img, w=img_width )

            # NUC Images
            block += '''<div>'''
            imgs = [ 'bC_1_key_flows.png', 'bC_1_raw_bead_1mer_flows.png', 'bC_1_raw_bead_0mer_flows.png' ]
            for img in imgs:
                block += '''
                <a href="{0}"><img src="{0}" width="{w}" /></a> 
                '''.format( img, w=nuc_img_width )
            block += '''</div>'''


            block += '''
            <div width:{w};">
            '''.format(w=width )

            block += '''
            <table>
                <tr>
                    <th> Metric </th>
                    <th> Int75 </th>
                    <th> Int50 </th>
                    <th> Int25 </th>
                    <th> Int10 </th>
                </tr>
            '''

            selected_metrics = [('nss_step_height_spa'  ,'NSS Height Spa',),
                                ]

            for key, name in selected_metrics:
                block += '''
                <tr> 
                    <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td>
                </tr>
               '''.format(      name,
                                *['{:.2f}'.format( self.metrics[key][m]) for m in ['Int75','Int50','Int25','Int10'] ]
                                )

            # Nuc Metrics
            block += '''

                <tr> <th colspan="5" height="10px"> </th></tr>
                <tr> <th colspan="5" height="10px"> </th></tr>

                <tr> 
                    <th colspan="3"> r.600:700, c.MidLane </th> <th colspan="2">  </th>
                </tr>'''

            nuc_metrics = [ ('rel_diff_key_height'   , 'Rel Diff Pk Sig (%)', ),
                            ('raw_diff_key_fwhm'     , 'RAW Diff FWHM (frames)', ),
                            ('raw_diff_key_peak_loc' , 'RAW Diff Pk Loc (frames)',), ]

            for key, name in nuc_metrics:
                try:
                    block += '''
                    <tr> 
                        <td colspan="3">{}</td> <td colspan="2">{}</td> 
                    </tr>
                    '''.format(      name,
                                    '{:.1f}'.format( self.metrics['nucs']['bC_1'][key] ),
                                    )
                except Exception as e:
                    print( '!!! Problem with getting value !!!\n{}'.format(e) )
                    block += '''
                    <tr> 
                        <td colspan="3">{}</td> <td colspan="2">{}</td>  
                    </tr>
                    '''.format(      name,
                                    '---',
                                    )

            block += ''''</table></div>'''
        else:
            # Beadsticking Images
            block += '''<div>'''
            imgs = [ 'multilane_all_nss_step_height_spa_density.png',]
            for img in imgs:
                block += '''
                <a href="{0}"><img src="{0}" width="{w}" /></a> 
                '''.format( img, w=img_width )
            block += '''</div>'''

            block += '''<div>'''
            imgs = [ 'multilane_all_{}_norm_max_diff_spa_density_local.png'.format( nuc ) for nuc in ['T','C','A','G'] ]
            for img in imgs:
                block += '''
                <a href="{0}"><img src="{0}" width="{w}" /></a> 
                '''.format( img, w=img_width )
            block += '''</div>'''

            # NUC Images
            block += '''<div>'''
            imgs = [ 'lane_{}_bR_6_key_flows.png'.format( i ) for i, _, active in self.iterlanes() if active ]
            for img in imgs:
                block += '''
                <a href="{0}"><img src="{0}" width="{w}" /></a> 
                '''.format( img, w=nuc_img_width )
            block += '''</div>'''

            block += ''' <hr> '''

            for i, lane, active in self.iterlanes():
                if i==1 or i ==3:
                    block += '''<span>'''

                if not active:
                    block += '''
                    <div style="display:inline-block;width:{w};">
                        <h3> Lane {} </h3>
                    </div>
                    '''.format(i, w=width)
                else:
                    block += '''
                    <div style="display:inline-block;width:{w};">
                    <h3> Lane {} </h3>
                    '''.format(i, w=width )

                    # Beadsticking Metrics
                    block += '''
                    <table>
                        <tr> 
                            <th> Metric </th> <th> Int75 </th> <th> Int50 </th> <th> Int25 </th> <th> Int10 </th>
                        </tr>
                    '''

                    selected_metrics = [('nss_step_height_spa'  ,'NSS Height Spa',),
                                        ]

                    for key, name in selected_metrics:
                        try:
                            block += '''
                            <tr> 
                                <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> 
                            </tr>
                            '''.format(      name,
                                            *['{:.1f}'.format( self.metrics[lane][key][m] ) for m in ['Int75','Int50','Int25','Int10'] ]
                                            )
                        except:
                            block += '''
                            <tr> 
                                <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> 
                            </tr>
                            '''.format(      name,
                                            *['---' for i in range(4) ]
                                            )

                    # Nuc Metrics
                    block += '''

                        <tr> <th colspan="5" height="10px"> </th></tr>
                        <tr> <th colspan="5" height="10px"> </th></tr>

                        <tr> 
                            <th colspan="3"> r.600:700, c.MidLane </th> <th colspan="2">  </th>
                        </tr>'''

                    nuc_metrics = [ ('rel_diff_key_height'   , 'Rel Diff Pk Sig (%)', ),
                                    ('raw_diff_key_fwhm'     , 'RAW Diff FWHM (frames)', ),
                                    ('raw_diff_key_peak_loc' , 'RAW Diff Pk Loc (frames)',), ]

                    for key, name in nuc_metrics:
                        try:
                            block += '''
                            <tr> 
                                <td colspan="3">{}</td> <td colspan="2">{}</td> 
                            </tr>
                            '''.format(      name,
                                            '{:.1f}'.format( self.metrics[lane]['nucs']['bR_6'][key] ),
                                            )
                        except Exception as e:
                            print( '!!! Problem with getting value !!!\n{}'.format(e) )
                            block += '''
                            <tr> 
                                <td colspan="3">{}</td> <td colspan="2">{}</td>  
                            </tr>
                            '''.format(      name,
                                            '---',
                                            )

                    block += ''' 
                    </table> 
                    '''

                    block += '''</div>'''
                if i == 2 or i == 4:
                    block += '''</span>'''
            block += '''</body></html>'''

        with open( html, 'w' ) as f:
            f.write( block )

    def write_html( self ):
        '''Writes html file that is output on a subpage linked to main report screen.'''
        html = os.path.join( self.results_dir, 'NucStepSpatialV2.html' )

        nuc_img_width = '125px'

        block = '''
        <html><head><title>NucStepSpatialV2</title></head>
        <body>
        <style type="text/css">table                {border=collapse: collapse;}</style>
        <style type="text/css">tr:nth-child(even)   {background-color: #DDD;}</style>
        <style type="text/css">td                   {border: 1px solid black; text-align: center; }</style>
        '''

        if not self.explog_is_multilane:
            block += ''' <p> <span> '''
            block += '''
            <div style="display:inline-block;width:45%;">
            <a href="{sp}"><img src="{sp}" width="15%" /></a>
            <a href="{tn}"><img src="{tn}" width="15%" /></a>
            <a href="{sp_l}"><img src="{sp_l}" width="15%" /></a>
            <a href="{tn_l}"><img src="{tn_l}" width="15%" /></a>
            <a href="{sp_lstd}"><img src="{sp_lstd}" width="15%" /></a>
            <a href="{tn_lstd}"><img src="{tn_lstd}" width="15%" /></a>
            </div>
            '''.format( sp      = 'ffc_nss_step_height_spa_density.png',
                        tn      = 'ffc_nss_step_height_density.png',
                        sp_l    = 'ffc_nss_step_height_spa_density_local.png',
                        tn_l    = 'ffc_nss_step_height_density_local.png',
                        sp_lstd = 'ffc_nss_step_height_spa_density_localstd.png',
                        tn_lstd = 'ffc_nss_step_height_density_localstd.png',
                    )
            
            block += '''
            <div style="display:inline-block;width:45%;">
            <table>
                <tr>
                    <th> Metric </th>
                    <th> Mean </th>
                    <th> Q2 </th>
                    <th> STD </th>
                    <th> IQR </th>
                    <th> VMR </th>
                    <th> Q1 </th>
                    <th> P10 </th>
                    <th> P50-P10 </th>
                    <th> P25-P10 </th>
                </tr>
            '''

            selected_metrics = [('nss_step_height_spa'  ,'NSS Height Spa',),
                                ('nss_step_height'  ,'NSS Height',),
                                ]

            gen_stats = ['mean','q2','std','iqr','vmr','q1','P10','d_p50_p10','d_p25_p10']

            for key, name in selected_metrics:
                try:
                    block += '''
                    <tr> 
                        <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td>
                    </tr>
                    '''.format(      name,
                                    *['{:.2f}'.format( self.metrics[key][m] ) for m in gen_stats ]
                                    )
                except (KeyError, ValueError):
                    pass

                try:
                    block += '''
                    <tr> 
                        <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td>
                    </tr>
                    '''.format(      name + '--local',
                                    *['{:.2f}'.format( self.metrics[key]['local'][m] ) for m in gen_stats ]
                                    )
                except (KeyError, ValueError): 
                    pass

                try:
                    block += '''
                    <tr> 
                        <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td>
                    </tr>
                    '''.format(      name + '--localstd',
                                    *['{:.2f}'.format( self.metrics[key]['localstd'][m] ) for m in gen_stats ]
                                    )
                except (KeyError, ValueError): 
                    pass
            block += ''' 
            </table> 
            '''

            block += ''' </div>'''
            block += '''</span></p>'''
            
            block += ''' <p> <span> '''
            block += '''
            <table> '''

            block += '''<tr><th colspan=2> Inlet (left) --> Outlet (right) </th></tr>'''

            img_info = [ ('Key Flows', 'key_flows',),
                     ('1mer Flows', '1mer_flows',),
                     ('0mer Flows', '0mer_flows',),
                     ('RAW Bead 1mer Flows',  'raw_bead_1mer_flows',),
                     ('RAW Bead 0mer Flows',  'raw_bead_0mer_flows',),
                     ('RAW Empty 1mer Flows', 'raw_empty_1mer_flows',),
                     ('RAW Empty 0mer Flows', 'raw_empty_0mer_flows',),
                     ('Prerun Diff Flows',    'prerun_diff_flows',),
                     ('Prerun Bead Flows',    'prerun_bead_flows',),
                     ('Prerun Empty Flows',   'prerun_empty_flows',),
                    ]

            for name, fname in img_info:
                block += '''<tr><th>{name}</th><td>'''.format(name=name)
                imgs = [ 'bC_{}_{fname}.png'.format( j, fname=fname ) for j in range(12) ]
                for img in imgs:
                    block += '''
                    <a href="{0}"><img src="{0}" width="{w}" /></a> 
                    '''.format( img, w=nuc_img_width )
                block += '''</td></tr>'''

            block += ''' </table>'''
            block += '''</span></p>'''           

            block += ''' </body></html> '''

        else:
            for i, lane, active in self.iterlanes():
                if active:
                    block += ''' <p> <span> '''
                    block += '''
                    <div style="display:inline-block;width:45%;">
                    <h3> Lane {lane} </h3>
                    <a href="{sp}"><img src="{sp}" width="15%" /></a>
                    <a href="{tn}"><img src="{tn}" width="15%" /></a>
                    <a href="{sp_l}"><img src="{sp_l}" width="15%" /></a>
                    <a href="{tn_l}"><img src="{tn_l}" width="15%" /></a>
                    <a href="{sp_lstd}"><img src="{sp_lstd}" width="15%" /></a>
                    <a href="{tn_lstd}"><img src="{tn_lstd}" width="15%" /></a>
                    </div>
                    '''.format( lane = i,
                                sp      = 'multilane_lane_{}_nss_step_height_spa_density.png'.format(i),
                                tn      = 'multilane_lane_{}_nss_step_height_density.png'.format(i),
                                sp_l    = 'multilane_lane_{}_nss_step_height_spa_density_local.png'.format(i),
                                tn_l    = 'multilane_lane_{}_nss_step_height_density_local.png'.format(i),
                                sp_lstd = 'multilane_lane_{}_nss_step_height_spa_density_localstd.png'.format(i),
                                tn_lstd = 'multilane_lane_{}_nss_step_height_density_localstd.png'.format(i),
                            )
                    
                    block += '''
                    <div style="display:inline-block;width:45%;">
                    <table>
                        <tr>
                            <th> Metric </th>
                            <th> Mean </th>
                            <th> Q2 </th>
                            <th> STD </th>
                            <th> IQR </th>
                            <th> VMR </th>
                            <th> Q1 </th>
                            <th> P10 </th>
                            <th> P50-P10 </th>
                            <th> P25-P10 </th>
                        </tr>
                    '''.format(i)

                    selected_metrics = [('nss_step_height_spa'  ,'NSS Height Spa',),
                                        ('nss_step_height'  ,'NSS Height',),
                                        ]

                    gen_stats = ['mean','q2','std','iqr','vmr','q1','P10','d_p50_p10','d_p25_p10']

                    for key, name in selected_metrics:
                        try:
                            block += '''
                            <tr> 
                                <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td>
                            </tr>
                            '''.format(      name,
                                            *['{:.2f}'.format( self.metrics[lane][key][m] ) for m in gen_stats ]
                                            )
                        except (KeyError, ValueError):
                            pass

                        try:
                            block += '''
                            <tr> 
                                <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td>
                            </tr>
                            '''.format(      name + '--local',
                                            *['{:.2f}'.format( self.metrics[lane][key]['local'][m] ) for m in gen_stats ]
                                            )
                        except (KeyError, ValueError): 
                            pass

                        try:
                            block += '''
                            <tr> 
                                <td>{}</td> <td>{}</td>  <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td>
                            </tr>
                            '''.format(      name + '--localstd',
                                            *['{:.2f}'.format( self.metrics[lane][key]['localstd'][m] ) for m in gen_stats ]
                                            )
                        except (KeyError, ValueError): 
                            pass
                    block += ''' 
                    </table> 
                    '''

                    block += ''' </div>'''
                    block += '''</span></p>'''
                    
                    block += ''' <p> <span> '''
                    block += '''
                    <table> '''

                    block += '''<tr><th colspan=2> Inlet (left) --> Outlet (right) </th></tr>'''

                    img_info = [ ('Key Flows', 'key_flows',),
                             ('1mer Flows', '1mer_flows',),
                             ('0mer Flows', '0mer_flows',),
                             ('RAW Bead 1mer Flows',  'raw_bead_1mer_flows',),
                             ('RAW Bead 0mer Flows',  'raw_bead_0mer_flows',),
                             ('RAW Empty 1mer Flows', 'raw_empty_1mer_flows',),
                             ('RAW Empty 0mer Flows', 'raw_empty_0mer_flows',),
                             ('Prerun Diff Flows',    'prerun_diff_flows',),
                             ('Prerun Bead Flows',    'prerun_bead_flows',),
                             ('Prerun Empty Flows',   'prerun_empty_flows',),
                            ]
                    for name, fname in img_info:
                        block += '''<tr><th>{name}</th><td>'''.format(name=name)
                        imgs = [ 'lane_{}_bR_{}_{fname}.png'.format( i, j, fname=fname ) for j in range(7, -1, -1) ]
                        for img in imgs:
                            block += '''
                            <a href="{0}"><img src="{0}" width="{w}" /></a> 
                            '''.format( img, w=nuc_img_width )
                        block += '''</td></tr>'''

                    block += ''' </table>'''
                    block += '''</span></p>'''
                    

        block += ''' </body></html> '''

        with open( html, 'w' ) as f:
            f.write( block )

    def write_spatial_key_html( self ):
        '''Writes html file that is output on a subpage linked to main report screen.'''
        html = os.path.join( self.results_dir, 'SpatialKeyHeights.html' )


        block = '''
        <html><head><title>NucStepSpatialV2</title></head>
        <body>
        '''

        block += '''
            <div>
            <a href="{sk}"><img src="{sk}" width="50%" /></a>
            </div>
            '''.format( sk='spatial_relative_keys.png' )

        block += '''
            <div>
            <a href="{sk}"><img src="{sk}" width="50%" /></a>
            </div>
            '''.format( sk='spatial_relative_keys_Variability.png' )

        block += ''' </body></html> '''
        with open( html, 'w' ) as f:
            f.write( block )

#########################
#   HELPER FUNCTIONS    #
#########################

    def _set_attr( self, attr_name, val, isspa ):
        if isspa: attr_name += '_spa'
        setattr( self, attr_name, val )

    def _get_attr( self, attr_name, isspa ):
        if isspa:  attr_name += '_spa'
        return getattr( self, attr_name )

    def create_flow_based_readlength_scale( self ):
        ''' Creates flow depenedent scales to minimize white space '''
        if self.flows < 400:
            flowlims     = [ 0,150]
        elif (self.flows >= 400) and (self.flows < 750 ):
            flowlims     = [ 0,300]
        elif (self.flows >= 750) and (self.flows < 1000 ):
            flowlims     = [ 0,500]
        elif self.flows >= 1000:
            flowlims     = [ 0,800]
        else:
            flowlims     = [ 0,800]
        return flowlims

    def save_array( self, name, isspa ):
        if isspa:   file_name = name + '_spa'
        else:       file_name = name
        np.save( os.path.join( self.results_dir, file_name ) , self._get_attr( name, isspa ) )

    def get_lane_slice( self , data , lane_number , lane_width=None ):
        """ 
        Takes a data array and returns data from only the lane of interest. 
        lane_number is 1-indexed.
        """
        if lane_width == None:
            lane_width = data.shape[1] / 4
        cs = slice( lane_width*(lane_number-1) , lane_width*(lane_number) )
        return data[:,cs]

    def get_metrics( self , data , lane_number=None , lower_lim=0, add_vmr=False , lane_width=None, integral_cutoffs=None, extreme_high=None ):
        """ Creates a dictionary of mean, q2, and std of a lane, masking out values above a lower_lim (unless is None). 
            
            integral_cutoffs must be a list of integers

            extreme_high must be a number
        """
        if lane_number is not None:
            data = self.get_lane_slice( data , lane_number , lane_width )

        # filter by lower_lim if it exists
        if lower_lim == None:                           
            masked = data
        else:
            masked = data[ data > lower_lim ]
        print( 'masked data has shape {}'.format( masked.shape ) )
        # if masked has no elements, just return an empty dictionary
        if not masked.any():
            dummy = stats.named_stats( np.array([0]) )
            metrics = { key:None for key in dummy.keys() }
            metrics.update( {'d_p50_p10': None, 'd_p25_p10': None} )
            if integral_cutoffs:
                metrics.update( { 'Int{}'.format(int(c)):None for c in integral_cutoffs } )
            if add_vmr:
                metrics['vmr'] = None
        else:
            metrics = stats.named_stats( masked )
            p50 = metrics['q2']
            p25 = metrics['q1']
            p10 = metrics['P10']
            metrics.update( {'d_p50_p10': p50-p10, 'd_p25_p10': p25-p10} )
            #metrics = { 'mean': masked.mean() , 'q2': np.median( masked ) , 'std' : masked.std() }
            
            if add_vmr:
                try:
                    vmr = float( masked.var() ) / float( masked.mean() )
                except ZeroDivisionError:
                    vmr = 0.
                metrics['vmr'] = vmr

            # Integral Metrics using integer cutoffs
            # Reported as a percent of total wells
            if integral_cutoffs:
                total = masked[ masked != np.array(None) ].size
                for c in integral_cutoffs:
                    c = int(c)
                    metrics.update( { 'Int{}'.format(c): 100*np.float( masked[ np.logical_and( masked<c, masked != np.array(None) ) ].size )/ np.float(total) } )
            if extreme_high:
                total       = masked[ masked != np.array(None) ].size
                eh_count    = masked[ np.logical_and( masked>extreme_high, masked != np.array(None) ) ].size 
                metrics.update( { 'extreme_high'    : eh_count,
                                  'extreme_high_pct': np.float( eh_count / np.float(total) ) ,
                                  } )
        return metrics
        
    def iterlanes( self ):
        """ 
        Handy iterator to cycle through lane information and active states.  To be reused quite often. 
        returns lane number (1-4), lane name (for metric parsing), and  boolean for lane activity.
        """
        for i in range(1,5):
            name = 'lane_{}'.format(i)
            if self.explog_lanes_active[name]:
                active = True
            else:
                active = False
            yield ( i , name , active )

    def remove_nans_infs( self, data ):
        temp_z = np.zeros( data.shape )
        condition = ~np.logical_or( np.isnan( data ), np.isinf( data ) )
        # set nans and infs to 0
        temp_z[ condition ] = data[ condition ]
        return temp_z

class FullFlowcellPlot( object ):
    def __init__( self , data , title , metric_label , units='' , clims=None , cmap=matplotlib.cm.nipy_spectral , bin_scale=1 ):
        self.data         = data
        self.title        = title
        self.metric_label = metric_label
        self.units        = units
        self.bin_scale    = bin_scale
        
        # self.cmap is the color plan, and used in imshow.  self.cm is the cmap mapped onto our given limits.
        self.cmap         = cmap
        
        self.update_clims( clims )
        
        # Set other secretly configurable properties
        self.figsize = (12,8)
        
    def update_clims( self , clims ):
        if not clims:
            _ = self.calc_clims()
        else:
            self.clims = clims
            
        self.calc_bins( )
        self.set_cm( self.cmap )
        return None
        
    def update_cmap( self , cmap ):
        self.cmap = cmap
        self.set_cm( self.cmap )
        
    def get_xlabel( self ):
        if self.units != '':
            xl = '{} ({})'.format( self.metric_label , self.units )
        else:
            xl = '{}'.format( self.metric_label )
        return xl
    
    def plot_heat_and_hist( self , figpath=None , figsize=None):
        data = self.data
            
        if figsize:
            fig     = plt.figure( figsize=figsize )
        else:
            fig     = plt.figure( figsize=(6,8) )
        spatial = self.subplot_to_grid( fig , (2,1) , (0,0) )
        spahist = self.subplot_to_grid( fig , (2,1) , (1,0) )
        
        # Spatial plot
        im = spatial.imshow ( data , interpolation='nearest', origin='lower', clim=self.clims , cmap=self.cmap )
        spatial.set_xticks  ( [] )
        spatial.set_yticks  ( [] )
        spatial.set_title   ( 'Full-Flowcell | {}'.format( self.title ) )
        
        # Histogram
        n, bins, patches = spahist.hist( data[data > 0] , bins=self.bins, zorder=0 )
        colors = self.cm.to_rgba( bins[:-1] + 0.5 * np.diff(bins) )
        for i in range( len(patches) ):
            patches[i].set_facecolor( colors[i,:] )
            patches[i].set_edgecolor( colors[i,:] )
            
        # X-axis config
        xt = np.arange      ( self.clims[0] , self.clims[1]+0.1 , float(self.clims[1]-self.clims[0])/4. )
        spahist.set_xlim       ( self.clims )
        spahist.set_xticks     ( xt )
        spahist.set_xticklabels( [ '{:.1f}'.format(x) for x in xt ] )
        spahist.set_xlabel     ( self.get_xlabel() )
        
        avg = data[ data > 0 ].mean()
        q2  = np.median( data[ data > 0 ] )
        
        if self.units == '%':
            _   = spahist.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f}{}'.format( avg, self.units ) )
            _   = spahist.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f}{}'.format( q2 , self.units ) )
        else:
            _   = spahist.axvline( avg , ls='-',  color='blue' ,  alpha=0.5 , label='Mean: {:.1f} {}'.format( avg, self.units ) )
            _   = spahist.axvline( q2  , ls='--', color='black',  alpha=0.5 , label='Q2: {:.1f} {}'.format( q2 , self.units ) )
        
        spahist.legend        ( loc='best' , fontsize=10 )
        spahist.grid          ( ':' , color='grey' , zorder=3 )
        plt.tight_layout   ( )
        plt.subplots_adjust( hspace=0, wspace=0 , top=0.94 )
        if not figpath:
            plt.show( )
        else:
            plt.savefig( figpath )
            plt.close  ( )
            
    def calc_clims( self ):
        print( 'data shape', self.data.shape )
        valid = self.data[ self.data > 0 ]
        print( 'valid shape (i.e. >0)', valid.shape )
        if valid.shape == (0,):
            self.clims = [-1,1]
            return self.clims
        m     = valid.mean()
        sigma = valid.std()
        skew  = (valid.max() - m) - (m - valid.min())
        if skew < (-3 * sigma):
            low  = m - 4.5 * sigma
            high = m + 1.5 * sigma
        elif skew > (3 * sigma):
            low  = m - 1.5 * sigma
            high = m + 4.5 * sigma
        else:
            low  = m - 3 * sigma
            high = m + 3 * sigma
        zmin = 10. * ( np.floor( low / 10. ) )
        zmax = 10. * ( np.ceil ( high / 10. ) )
        if zmin < 0:
            zmin = 0
        self.clims = [zmin , zmax]
        return [zmin,zmax]
    
    def calc_bins ( self , bin_scale=None ):
        a , b     = self.clims
        if bin_scale:
            self.bin_scale = bin_scale
        
        # Auto bin scale scaling
        if self.bin_scale == 1:
            if (b-a) <= 20:
                print( 'Adjusting bin_scale to 4 due to small clims.' )
                self.bin_scale = 4
            elif (b-a) > 20 and (b-a) <= 50:
                print( 'Adjusting bin_scale to 2 due to modest clims.' )
                self.bin_scale = 2
        #print( 'bin_scale', self.bin_scale ) 
        #print( 'a, b', a, b )
        self.bins = np.linspace( a , b , self.bin_scale*(b-a) + 1 )
        
    def set_cm( self , cmap ):
        self.cm = self.create_cmap( self.clims , cmap )
        
    @staticmethod
    def create_cmap( lims , cmap ):
        cm = matplotlib.cm.ScalarMappable( cmap=cmap )
        cm.set_clim( *lims )
        return cm

    @staticmethod
    def subplot_to_grid( fig , shape, loc, rowspan=1, colspan=1 ):
        """ A mirror function of matplotlib.pyplot.subplot2grid that doesn't try to import tkinter """
        gridspec    = GridSpec( shape[0], shape[1])
        subplotspec = gridspec.new_subplotspec(loc, rowspan, colspan)
        axis        = fig.add_subplot( subplotspec )
        return axis

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


if __name__ == "__main__":
    PluginCLI()
