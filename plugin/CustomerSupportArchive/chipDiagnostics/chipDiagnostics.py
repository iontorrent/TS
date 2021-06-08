#!/usr/bin/env python
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

from ion.plugin import *

import sys, os, time, subprocess, textwrap
import numpy as np
from scipy import ndimage
import traceback

import matplotlib
matplotlib.use( 'agg' ) # to make sure this is set for before another module doesn't set it
import matplotlib.pyplot as plt

# Set default cmap to jet, which is viridis on newer TS software (matplotlib 2.1)
matplotlib.rcParams['image.cmap'] = 'jet'

# Import our tools
import html
import kickback
from cdtools import NoisyOffset
from tools import misc, chipcal
from tools.core import chiptype
from tools.explog import Explog
from tools.imtools import GBI
from tools.stats import calc_iqr
from tools import json2 as json

# Bring in edge analyzer
from tools import edge_analyzer as ea
from tools.PluginMixin import PluginMixin

class chipDiagnostics( IonPlugin , PluginMixin ):
    """ 
    Phil Waggoner
    
    Plugin to analyze core chip performance metrics.
    
    Latest Update | Reverted to DynamicRangeAfterBF as the core DR to use for offset analysis based on behavior
                  |   from all chip types, including those not on Valkyrie.
                  | Bugfix where tools directory was not committed and included in chipDiagnostics with DR fix.
                  | Major update due to correction for selecting proper DynamicRange from explog.  Version 4.0 
                      was mistakenly upgraded to use the wrong value when in reality there was a datacollect bug.
                  | v4.0.3 | B.P. | Uprevved version number for improved installation tracking (4.0.2 was used 
                      in debug and had installation issue on tardis.ite)
                  | v6.1.1 | B.P.   | updated chipcal.determine_lane algorithm
    """
    version       = "6.1.1"
    allow_autorun = True
    
    runTypes      = [ RunType.THUMB , RunType.FULLCHIP , RunType.COMPOSITE ]
    
    def launch( self ):
        # Replace previous calls with PluginMixin.init_plugin()
        # Moved find_die_area method into PluginMixin
        self.init_plugin( )
        
        # Analyze calibration and create related outputs
        if self.has_chipcal:
            self.analyze_cal         ( )
            self.calibration         ( )
            self.pixel_uniformity    ( )
            self.instrument          ( )
            
            # Leverages new EdgeAnalyzer Class
            self.analyze_edge_effects( ) 
            self.edge_analysis       ( )
            
            self.compile_json        ( )
            self.edge_effects        ( ) # Traditional column-averaged "edge effects"
            self.sneaky_clusters     ( )
            self.multilane_analysis  ( )
            
            # Analyze Noise vs. Offset data (i.e. conversion noise)
            # Update 2017-10-12: Removing conversion noise from analysis for now.
            #                    Utility limited and currently not worth the computational time for the plugin.
            #                    - Phil
            #self.analyze_conversion( )
            #self.conversion_noise  ( )
            print('Conversion noise analysis turned off in version 2.3.  If desired, please contact Phil Waggoner.')
            
        if self.has_rawdata:
            self.analyze_kickback    ( )
            
            # Compile metrics and save results.json
            self.compile_json      ( )
            
        self.block_html        ( )
        
        print( 'Plugin complete.' )
        
    def analyze_cal( self ):
        ''' loads and does basic calibration analysis '''
        #self.cc = chipcal.ChipCal( self.calibration_dir , self.chip_type , self.results_dir )
        self.cc = chipcal.ChipCal( self.calibration_dir , self.ct.name , self.results_dir )
        self.cc.check_chip_rc    ( self.explog.metrics['Rows'] , self.explog.metrics['Columns'] )
        
        # Set up metrics dictionary for wafermap spatial plot limits.
        self.wafermap_lims = {}
        
        def process_lims( lims ):
            ''' turns a limit list/tuple into a dummy dict. '''
            return {'low': lims[0] , 'high': lims[1] }

        # Setup some useful limit constants for wafermap spatials
        gain_localstd_lims   = [0,200]
        noise_localstd_lims  = [0, 80]
        offset_localstd_lims = [0, 60]
        
        # Load gain and determine reference pixels using a gain cutoff.
        self.cc.load_gain       ( )
        self.cc.determine_lane  ( ) # Need to do this as soon as possible.
        self.cc.find_refpix     ( )
        
        # Moved up load_offset so that we can determine pinned pixels ahead of analyzing gain and ignoring pinned pixels . . .
        print( 'DR detected to be %s' % self.explog.DR )
        self.cc.offset_lims = [ 0, self.explog.DR ] # This call gets overwritten in .load_offset . . . . .
        self.cc.load_offset   ( DR=self.explog.DR , pinned_low_cutoff=500 , pinned_high_cutoff=15883 )
        
        self.analyze_property   ( 'gain' )
        
        self.cc.wafermap_spatial( self.cc.gain[::self.cc.rows/600,::self.cc.cols/800],'gain',self.cc.gain_lims )
        self.cc.wafermap_spatial( self.cc.gain_localstd , 'gain_localstd' , gain_localstd_lims )
        self.wafermap_lims['gain']          = process_lims( self.cc.gain_lims )
        self.wafermap_lims['gain_localstd'] = process_lims( gain_localstd_lims )
        
        # Create reference-pixel-motivated gain plots with different limits than the above
        self.cc.low_gain_spatial  ( [0,500] )
        self.cc.low_gain_histogram( [0,500] )
        
        # Load noise and prepare metrics.  Try to define a universal noise scale of 0-400 uV.
        self.cc.load_noise    ( )
        if self.ct.series == 'pgm':
            self.cc.noise_lims = [0,150]
        else:
            self.cc.noise_lims = [0,400]
        self.analyze_property ( 'noise' )
        self.cc.edge_noise    ( )
        
        self.cc.wafermap_spatial( self.cc.noise[::self.cc.rows/600,::self.cc.cols/800],'noise',self.cc.noise_lims )
        self.cc.wafermap_spatial( self.cc.noise_localstd , 'noise_localstd' , noise_localstd_lims )
        self.wafermap_lims['noise']          = process_lims( self.cc.noise_lims )
        self.wafermap_lims['noise_localstd'] = process_lims( noise_localstd_lims )
        
        # Load offset.  Make sure to get real DR.  This should be read in from explog through other methods.
        self.analyze_property  ( 'offset' )
        
        self.cc.pinned_heatmaps( )
        self.cc.pinned_heatmaps( hd=True ) # New in version 4.0.0
        self.wafermap_lims['perc_pinned_low_hs']   = process_lims( [0,  5] )
        self.wafermap_lims['perc_pinned_low']      = process_lims( [0, 20] )
        self.wafermap_lims['perc_pinned_low_full'] = process_lims( [0,100] )
        
        self.cc.wafermap_spatial(self.cc.offset[::self.cc.rows/600,::self.cc.cols/800],'offset',self.cc.offset_lims)
        self.cc.wafermap_spatial( self.cc.offset_localstd , 'offset_localstd' , offset_localstd_lims )
        self.wafermap_lims['offset']          = process_lims( self.cc.offset_lims )
        self.wafermap_lims['offset_localstd'] = process_lims( offset_localstd_lims )
        
        # Analyze sneaky clusters
        self.cc.analyze_sneaky_clusters( )
        
        # Test for multilane chip and if it is, then make some plots
        if self.cc.is_multilane:
            for met in ['noise','gain','offset']:
                metrics = [met, '{}_localstd'.format( met ), '{}_true_localstd'.format( met ), '{}_true_localstd_hd'.format( met )]
                for m in metrics:
                    self.cc.calc_metrics_by_lane( m )
                    self.cc.multilane_boxplot   ( m )
                    self.cc.calc_pinned_metrics_by_lane( )
                
            # These guys use default limits
            self.cc.multilane_wafermap_spatial( 'noise'  )
            self.cc.multilane_wafermap_spatial( 'gain'   )
            self.cc.multilane_wafermap_spatial( 'offset' )
            self.cc.multilane_wafermap_spatial( 'noise' , transpose=False )
            self.cc.multilane_wafermap_spatial( 'gain'  , transpose=False )
            self.cc.multilane_wafermap_spatial( 'offset', transpose=False )
            
            # Special limits
            for m, lims in [('noise',noise_localstd_lims), ('gain',gain_localstd_lims), ('offset',offset_localstd_lims)]:
                for prefix,suffix in [ ('',''), ('true_',''), ('true_','_hd')]:
                    self.cc.multilane_wafermap_spatial( '{}_{}localstd{}'.format(m, prefix, suffix), lims )
                    self.cc.multilane_wafermap_spatial( '{}_{}localstd{}'.format(m, prefix, suffix), lims, transpose=False )
                    
            # Make the suite of pinned pixel density wafermap images of each lane at the scales of 0 to 5, 20, or 100%.
            self.cc.multilane_pinned_heatmaps( hd=False )
            self.cc.multilane_pinned_heatmaps( hd=True  )
            
    def analyze_conversion( self ):
        ''' 
        Uses calibration data to look for correlation of noise, offset, and pixel conversions 
        Capability remains to slice data additional ways in the future, e.g. with single halfs/rows/cols/rois/etc.
        See Noisyoffset.set_<row/col>_slice for more.
        '''
        # Attempt to set noise limits by noise mean
        noise_lims = [0,400]
        noise_q2   = self.cc.metrics['noise_q2']
        if noise_q2 < 100:
            noise_lims = [0,150]
        elif (noise_q2 >= 100) and (noise_q2 < 200):
            noise_lims = [50,250]
        elif (noise_q2 >= 200) and (noise_q2 < 300):
            noise_lims = [150,350]
        elif (noise_q2 >= 300) and (noise_q2 < 400):
            noise_lims = [250,450]
            
        self.no = NoisyOffset( self.cc.noise , self.cc.offset , DR=self.explog.DR )
        self.no.noise_lims = noise_lims
        
        # First analyze full chip
        self.no.analyze( self.results_dir )
        
        # Now analyze by quadrant
        #
        # *** I realize this isn't Cartesian style.  Sorry! *** - Phil
        #
        # (0,0) Origin in lower left
        #------#------# 
        #  Q4  |  Q3  #
        #------M------#
        #  Q1  |  Q2  #
        #------#------# Software row zero
        
        for Q in [1,2,3,4]:
            self.no.analyze( self.results_dir , Q )
            
        # Serialize json first
        misc.serialize( self.no.metrics )
        
        # Write out json file
        with open( os.path.join( self.results_dir , 'noise_vs_offset.json' ) , 'w' ) as f:
            json.dump( self.no.metrics , f )
            
    def analyze_edge_effects( self ):
        """ Leverages EdgeAnalyzer class to see radial calibration effects. """
        edge = ea.EdgeAnalyzer( self.cc.active , self.cc.is_multilane )
        edge.parse_chip()

        # Spatial map of ring masks.
        edge.plot_rings( os.path.join( self.results_dir , 'ring_mask_spatial.png' ) )
        
        # Ringplots . . . would like to get noise, offset, gain.  Average and Stdev?
        x , noise_q2   = edge.make_ringplot( self.cc.noise , ylabel='Median Chip Noise (uV)' ,
                                             imgpath=os.path.join( self.results_dir , 'ringplot_noise_q2.png' ) ,
                                             operation = np.median )
        
        _ , noise_iqr  = edge.make_ringplot( self.cc.noise , ylabel='Noise IQR (uV)' ,
                                             imgpath=os.path.join( self.results_dir , 'ringplot_noise_iqr.png' ) ,
                                             operation = calc_iqr )
        
        _ , offset_q2  = edge.make_ringplot( self.cc.offset , ylabel='Pixel Offset Q2 (mV)' ,
                                             imgpath=os.path.join( self.results_dir , 'ringplot_offset_q2.png' ) ,
                                             operation = np.median )
        
        _ , offset_iqr = edge.make_ringplot( self.cc.offset , ylabel='Pixel Offset IQR (mV)' ,
                                             imgpath=os.path.join( self.results_dir , 'ringplot_offset_iqr.png' ) ,
                                             operation = calc_iqr )
        
        _ , gain_q2    = edge.make_ringplot( self.cc.gain , ylabel='Gain Q2 (mV/V)' ,
                                             imgpath=os.path.join( self.results_dir , 'ringplot_gain_q2.png' ) ,
                                             operation = np.median )
        
        _ , gain_iqr   = edge.make_ringplot( self.cc.gain , ylabel='Gain IQR (mV/V)' ,
                                             imgpath=os.path.join( self.results_dir , 'ringplot_gain_iqr.png' ) ,
                                             operation = calc_iqr )
        
        # Now we need to calculate some metrics for the results.json file.
        edge_metrics = { }
        
        # Add in how many pixels exist in each ring.  New method iter_rings adds center of chip.
        edge_metrics['pixels'] = {}
        for i,ring in edge.iter_rings():
            edge_metrics['pixels'][ring] = edge.ring_pixel_count[i]
            
        # Right now, let's save metrics for each interval.
        for i, ring in edge.iter_rings():
            if i == 0:
                edge_metrics.update( { 'noise_q2'  : {} , 'noise_iqr': {} , 'offset_q2' : {} ,
                                       'offset_iqr': {} , 'gain_q2'  : {} , 'gain_iqr'  : {} } )
            else:
                edge_metrics['noise_q2'  ][ring] = float( noise_q2[i-1]  )
                edge_metrics['noise_iqr' ][ring] = float( noise_iqr[i-1] )
                edge_metrics['offset_q2' ][ring] = float( offset_q2[i-1] )
                edge_metrics['offset_iqr'][ring] = float( offset_iqr[i-1])
                edge_metrics['gain_q2'   ][ring] = float( gain_q2[i-1]   )
                edge_metrics['gain_iqr'  ][ring] = float( gain_iqr[i-1]  )
                
        misc.serialize( edge_metrics )
        self.edge_metrics = edge_metrics
        
    def analyze_kickback( self ):
        ''' Analyze the kickback '''
        try:
            kbip = kickback.KickbackIonPlugin()
            kbip.startplugin = self.startplugin
            kbip.results_dir = self.results_dir
            kbip.outjson     = 'kickback.json'

            # Tell kickback plugin where raw acquisitions really are (raw_tndata_dir is last used here)
            # kbip.raw_tndata_dir = self.acq_dir
            try:
                kbip.launch()
            except SystemExit:
                # Ignore the sys.exit at the end of the script
                pass
            self.results['kickback'] = kbip.results
        except:
            print( traceback.format_exc() )
            print( 'ERROR IN KICKBACK ANALYSIS' )


    def analyze_property( self , prop ):
        m = prop.lower()
        
        self.cc.calc_metrics ( m , ignore_upper_limit=True )
        self.cc.histogram    ( m , rescale_ylim=True ) # Added kwarg in version 4.0.0 to eliminate refpix/other lanes from dominating ylims
        if m == 'noise' and self.ct.series == 'pgm':
            self.cc.noise = GBI( self.cc.noise[:,:,np.newaxis] , self.cc.noise < 1 , 10 ).squeeze()
            self.cc.noise[ np.isnan( self.cc.noise ) ] = 0.
            
        self.cc.spatial      ( m )
        self.cc.colavg       ( m )
        self.cc.plot_colavg  ( m )
        self.cc.plot_colavg  ( m , True )
        
        # Turning off this old method in version 4.0.0 - Phil
        # self.cc.diff_img     ( m )
        # self.cc.plot_diff_img( m )
        
        ##################################################
        # Generate (spam) superpixel attributes
        ##################################################
        # This is the historical default
        self.cc.superpixel_analysis( m ) 
        self.cc.local_std    ( m )
        self.cc.local_avg    ( m )
        self.cc.diff_img_hd  ( m )
        
        # This is the better way to do it (ignore_pinned=True)
        self.cc.superpixel_analysis( m, ignore_pinned=True )
        self.cc.local_std    ( m, ignore_pinned=True )
        
        # This is the best way to do it (ignore_pinned=True and hd=True)
        self.cc.superpixel_analysis( m, ignore_pinned=True, hd=True ) 
        self.cc.local_std    ( m, ignore_pinned=True, hd=True )
        self.cc.local_avg    ( m, ignore_pinned=True )
        self.cc.diff_img_hd  ( m, ignore_pinned=True )
        ##################################################
        
        self.cc.save_json    ( m )
        return None
    
    def calibration( self ):
        """ 
        Takes the place of previous calibration script.  
        Also should be run after creating self.explog and running self.analyze_cal(). 
        """
        if not hasattr( self , 'explog' ):
            print( 'ERROR!  Have not yet loaded explog.  Attempting to load . . .' )
            self.explog = Explog( self.raw_data_dir )
            
        if not hasattr( self , 'cc' ):
            print( 'ERROR!  We have not run calibration analysis yet, cannot make html page yet . . .' )
            return None
            
        # Create plots from self.explog.  These will not do anything for PGM runs.
        self.explog.chip_temp_plot ( self.results_dir )
        self.explog.dss_plot       ( self.results_dir )
        
        # Create HTML output
        link_styles = [ '<style>a:link    { color: #000; text-decoration:none;      font-weight=bold; }</style>',
                        '<style>a:visited { color: #000; text-decoration:none;      font-weight=bold; }</style>',
                        '<style>a:hover   { color: #333; text-decoration:underline; font-weight=bold; }</style>',
                        '<style>a:active  { color: #333; text-decoration:underline; font-weight=bold; }</style>']
        cal = html.HTML( os.path.join( self.results_dir , 'calibration.html' ) )
        cal.make_header( 'Calibration' , use_courier=True , styles=link_styles )
        
        #########################
        # Pedigree Section
        #########################
        pedigree = html.table( )
        pedigree.add_row( ['<font size=4, face="Arial">Pedigree</font>',''] , [50,50] )
        pedigree.add_row( ['<br>',''] , [50,50] )
        
        # Design
        pedigree.add_row( ['Design',''] , [50,50] , cl='d0' )
        labels = ['Type','Version','Rows','Columns']
        keys   = ['ChipType','ChipVersion','Rows','Columns']
        for pair in zip( labels , keys ):
            pedigree.add_row( [ pair[0] , self.explog.metrics[ pair[1] ] ] , [50,50] )
        pedigree.add_row( ['<br>',''] )
        
        # Manufacture
        pedigree.add_row( ['Manufacture',''] , [50,50] , cl='d0' )
        labels = ['CMOS Lot','Wafer','Wafer X','Wafer Y','Assembly Lot','Flow Cell','Barcode']
        keys   = ['CMOSLotId','WaferId','WaferX','WaferY','AssemblyLotId','FlowCell','ChipBarcode']
        for pair in zip( labels , keys ):
            if pair[1] == 'CMOSLotId':
                # Let's link to chipdb.
                link = '<a href="http://chipdb.ite/lots/{0}/">{0}</a>'.format( self.explog.metrics[ pair[1] ] )
                pedigree.add_row( [ pair[0] , link ] , [50,50] )
            elif pair[1] == 'WaferId':
                url  = "http://chipdb.ite/lots/{0[CMOSLotId]}/{0[WaferId]}/".format( self.explog.metrics )
                link = '<a href="{0}">{1}</a>'.format( url , self.explog.metrics[ pair[1] ] )
                pedigree.add_row( [ pair[0] , link ] , [50,50] )
            else:
                pedigree.add_row( [ pair[0] , self.explog.metrics[ pair[1] ] ] , [50,50] )
        pedigree.add_row( ['<br>',''] )
        
        # Package Test (only for proton)
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            pedigree.add_row( ['Package Test',''] , [50,50] , cl='d0' )
            labels = ['Id','Bin','Soft Bin','Comment']
            keys   = ['PackageTestId','PackageTestBin','PackageTestSoftBin','Comment']
            for pair in zip( labels , keys ):
                pedigree.add_row( [ pair[0] , self.explog.metrics[ pair[1] ] ] , [50,50] )
            pedigree.add_row( ['<br>',''] )
        
        # Add source info.  Eventually need to add error handling for when explog isn't found.
        pedigree.add_row( ['<font size=2 face="Arial">Source: %s</font>' % os.path.basename( self.explog.log ),
                           ''] , [50,50] )
        
        cal.add( pedigree.get_table() )
        cal.add( '<hr>' )
        
        #########################
        # Operation Section
        #########################
        operation = html.table( )
        operation.add_row( ['<font size=4, face="Arial">Operation</font>',''] )
        operation.add_row( ['<br>',''] )
        
        # Power Supply
        operation.add_row( ['Power Supply',''] , [50,50] , cl='d0' )
        labels = ['Analog','Digital','Output']
        keys   = ['%sSupplyVoltage' % l for l in labels]
        for pair in zip( labels , keys ):
            operation.add_row( [ pair[0] , '%1.2f V' % self.explog.metrics[ pair[1] ] ] , [50,50] )
        operation.add_row( ['<br>',''] )
        
        # Clock
        operation.add_row( ['Clock',''] , [50,50] , cl='d0' )
        labels = ['Frequency','Oversampling','Frame Rate']
        fmt    = ['%d MHz' , '%s' , '%d fps' ]
        keys   = ['ClockFrequency','Oversampling','FrameRate']
        for pair in zip( labels , fmt , keys ):
            operation.add_row( [ pair[0] , pair[1] % self.explog.metrics[ pair[2] ] ] , [50,50] )
        operation.add_row( ['<br>',''] )
        
        # Dynamic Range
        operation.add_row( ['Dynamic Range',''] , [50,50] , cl='d0' )
        labels = ['Pre-Beadfind','For Beadfind','Post-Beadfind']
        keys   = ['DynamicRange','DynamicRangeForBF','DynamicRangeAfterBF']
        for pair in zip( labels , keys ):
            operation.add_row( [ pair[0] , '%d mV' % self.explog.metrics[ pair[1] ] ] , [50,50] )
            
        # Add row for "Actual" field that is present on Proton, S5
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            operation.add_row( ['Actual' , '%d mV' % self.explog.metrics['DynamicRangeActual'] ] , [50,50] )
        else:
            # In the PGM case, we're done until the calibration metrics section, so tie in source.
            operation.add_row( ['<br>',''] )
            operation.add_row( ['<font size=2 face="Arial">Source: %s</font>' % os.path.basename( self.explog.log ),
                                '' ] )
            
        cal.add( operation.get_table() )
        
        # Now we need to add in DAC and Temperature sections for Proton/S5
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            #########################
            # DAC table
            #########################
            dac = html.table( )
            dac.add_row( ['DAC','','',''] , cl='d0' )
            
            # Metrics section
            dac.add_row( ['Mean' , '%0.0f counts' % self.explog.metrics['ChipDACMean'] , '' , '' ] )
            dac.add_row( ['SD'   , '%0.0f counts' % self.explog.metrics['ChipDACSD']   , '' , '' ] ) 
            dac.add_row( ['90%' , '%0.0f counts' % self.explog.metrics['ChipDAC90']   , '' , '' ] )
            
            dac_metrics = dac.get_table()
            dac_section = [ dac_metrics , html.image_link( 'chip_dac.png' ) ]
            
            #########################
            # Temperature table
            #########################
            ttl   = html.table( )
            ttl.add_row( ['Temperature'] , cl='d0' )
            ttl.add_row( ['<font size=2 face="Arial" color="red">[Measured values may not be accurate; thermometer testing and characterization in progress]</font>'] , cl='d0' )
            title = ttl.get_table()
            
            tmets = html.table( )
            # tmets.add_row( ['Temperature <font size=2 face="Arial" color="red">[Measured values may not be accurate; thermometer testing and characterization in progress]</font>','','',''] , cl='d0' )
            
            tmets.add_row( ['Thermometer 1','','Thermometer 2',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['ChipThermometer1Mean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['ChipThermometer2Mean'] ] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['ChipThermometer1SD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['ChipThermometer2SD'] ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['ChipThermometer190'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['ChipThermometer290'] ] )
            tmets.add_row( ['<br>','','',''] )
            tmets.add_row( ['Thermometer 3','','Thermometer 4',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['ChipThermometer3Mean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['ChipThermometer4Mean'] ] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['ChipThermometer3SD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['ChipThermometer4SD'] ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['ChipThermometer390'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['ChipThermometer490'] ] )
            tmets.add_row( ['<br>','','',''] )
            tmets.add_row( ['Average','','',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['ChipThermometerAverageMean'] ,'',''] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['ChipThermometerAverageSD'] , '' , '' ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['ChipThermometerAverage90']  ,'','' ] )
            
            temp_metrics = title + tmets.get_table()
            temp_section = [ temp_metrics , html.image_link( 'chip_thermometer_temperature.png' ) ]
            
            # Add these tables to the calibration html
            non_pgm = html.table( )
            cols   = [67,33]
            non_pgm.add_row( dac_section , cols )
            non_pgm.add_row( temp_section, cols )
            cal.add( non_pgm.get_table() )
            cal.add( '<br>' )
            cal.add( '<p><font size=2 face="Arial">Source: %s</font></p>' % os.path.basename( self.explog.log ) )
            
        # Merge files together again and add divider befor we get to calibration metrics.
        cal.add( '<hr>' )
        
        ##############################
        # Calibration Data
        ##############################
        suffixes = ['mean','std','P90','q1','q2','q3']

        # Gain...start with metrics
        gm     = html.table( )
        labels = ['Mean','SD','90%','Q1','Q2','Q3']
        values = ['gain_%s' % suffix for suffix in suffixes]
        
        gm.add_row( ['Calibration Gain',''] , cl='d0' )
        for pair in zip( labels , values ):
            gm.add_row( [ pair[0] , '%4.0f mV/V' % self.cc.metrics[ pair[1] ] ] )
            
        gain_metrics = gm.get_table()
        
        gain = html.table()
        gain.add_row( ['<font size=4, face="Arial">Gain</font>' , '' , '' ] )
        gain.add_row( ['<br>','',''] )
        gain.add_row( [gain_metrics,html.image_link('gain_spatial.png'),html.image_link('gain_histogram.png') ] )
        gain.add_row( [ '<font size=2 face="Arial">Source: %s</font>' % os.path.basename( self.cc.gainfile ) ,
                        '' ,'' ] )
        
        # Do we want to add the chip gain vs. vref plot here?  Maybe not yet.
        
        cal.add( gain.get_table() )
        cal.add( '<hr>' )
        
        # Noise
        # I am making a conscious decision to leave noise in uV and stop reporting in DN. - PW 25 Jan 2017
        nm     = html.table( )
        labels = ['Mean','SD','90%','Q1','Q2','Q3']
        values = ['noise_%s' % suffix for suffix in suffixes]
        
        nm.add_row( ['Calibration Noise',''] , cl='d0' )
        for pair in zip( labels , values ):
            nm.add_row( [ pair[0] , '%3.0f uV' % self.cc.metrics[ pair[1] ] ] )
            
        noise_metrics = nm.get_table()
        
        noise = html.table()
        noise.add_row( ['<font size=4, face="Arial">Noise</font>' , '' , '' ] )
        noise.add_row( ['<br>','',''] )
        noise.add_row( [noise_metrics,html.image_link('noise_spatial.png'),html.image_link('noise_histogram.png')])
        noise.add_row( ['<font size=2 face="Arial">Source: %s</font>' % os.path.basename( self.cc.noisefile ) ,
                        '' , ''] )
        
        cal.add( noise.get_table() )
        cal.add( '<hr>' )
        
        # Offset, Pinned pixels, and Vref
        om     = html.table( )
        
        # Add IQR to this group
        labels = ['Mean','SD','90%','Q1','Q2','Q3','IQR']
        values = ['offset_%s' % suffix for suffix in suffixes]
        values.append( 'offset_iqr' )
        
        om.add_row( ['Relative Offset',''] , cl='d0' )
        for pair in zip( labels , values ):
            om.add_row( [ pair[0] , '%3.0f mV' % self.cc.metrics[ pair[1] ] ] )
            
        # Now add a few more fun items.
        om.add_row( ['<br>',''] )
        om.add_row( ['Pixels in Range', self.cc.metrics[ 'PixelInRange' ] ] )
        om.add_row( ['Non-addressible Pixels', self.cc.metrics[ 'PixelInactive' ] ] )
        om.add_row( ['Pinned Low', '%d  |  (%0.2f%%)'  % ( self.cc.metrics[ 'PixelLow' ] ,
                                                           self.cc.metrics[ 'PercPinnedLow' ] ) ] )
        om.add_row( ['Pinned High', '%d  |  (%0.2f%%)' % ( self.cc.metrics[ 'PixelHigh' ] ,
                                                           self.cc.metrics[ 'PercPinnedHigh'] ) ] )
        om.add_row( ['<br>',''] )
        om.add_row( ['Reference Electrode', '%d mV' % self.explog.metrics[ 'FluidPotential' ] ] )
        om.add_row( ['<br>',''] )
        
        offset_metrics = om.get_table()
        
        offset = html.table()
        offset.add_row( ['<font size=4, face="Arial">Relative Offset</font>' , '' , '' ] )
        offset.add_row( ['<br>','',''] )
        offset.add_row( [offset_metrics,html.image_link('pix_offset_spatial.png'),
                         html.image_link('pix_offset_histogram.png') ] )
        
        cal.add( offset.get_table() )
        cal.add( '<p><font size=2 face="Arial">Source: %s | %s </font></p>' % ( os.path.basename(self.cc.pixfile),
                                                                                os.path.basename(self.explog.log)))
        cal.add( '<hr>' )
        
        # Add footer
        cal.make_footer( )
        
        # Write calibration.html!
        cal.write( )
        return None

    def sneaky_clusters( self ):
        """
        Creates sneaky_clusters.html for analysis of sneaky cluster defect.  Predominantly a P0 thing.
        """
        output = html.HTML( os.path.join( self.results_dir , 'SneakyClusters.html' ) )
        output.make_header( 'SneakyClusters' , use_courier=True )
        
        images = html.table( )
        images.add_row( [html.image_link('sneaky_cluster_plot.png'),
                         html.image_link('sneaky_clusters_bad_superpixels.png')] , [50,50] )
        images.add_row( [html.image_link('perc_pinned_low_spatial.png'),
                         html.image_link('offset_localstd_nopins_spatial.png')] , [50,50] )
            
        output.add        ( images.get_table( ) )
        output.make_footer( )
        output.write      ( )
        return None
    
    def edge_analysis( self ):
        """ Makes edge_analysis.html file. """
        # Write HTML output
        with open( os.path.join( self.results_dir , 'edge_analysis.html' ) , 'w' ) as f:
            f.write( '<html><head><title>EdgeAnalyzer</title></head><body><h2>Chip Calibration Edge Analysis</h2><br>\n')
            f.write( textwrap.dedent( '''\
                       <h3>Important Notes</h3>
                       <table border="0" cellpadding="0" width="100%%">
                        <tr>
                         <td width="70%%">
                       <ul>
                        <li>This analysis is based on perceived active pixels using a mask of pixels with gain > 500 mV/V.</li>
                        <li>A series of masks are made by eroding this active pixel mask by N pixels.  These masks are then applied to study pixel behavior in each "ring."</li>
                        <li>We note that larger bubbles can start to interfere with this particular algorithm, however, we believe this to be more robust than row averages across the chip center.</li>
                        <li>For reference, a spatial map of the masks used is shown below to the right.</li>
                       </ul>
                         </td>
                       ''' ) )
            f.write( '<td width="30%%"><a href="{0}"><img src="{0}" width="100%%" /></a></td>'.format( 'ring_mask_spatial.png' ) )
            f.write( textwrap.dedent( '''\
                        </tr>
                       </table>
                       <br>
                       ''') )
            
            f.write( '<table border="1" cellpadding="0" width="100%%">\n' )
            images = [ ( 'ringplot_noise_q2.png'   , 'noise_spatial.png' ) ,
                       ( 'ringplot_noise_iqr.png'  , 'noise_localstd_spatial.png') ,
                       ( 'ringplot_offset_q2.png'  , 'pix_offset_spatial.png' ) , 
                       ( 'ringplot_offset_iqr.png' , 'offset_localstd_spatial.png' ) ,
                       ( 'ringplot_gain_q2.png'    , 'gain_spatial.png' ) ,
                       ( 'ringplot_gain_iqr.png'   , 'gain_localstd_spatial.png' )
                       ]
                       
            for x , y in images:
                row  = '<tr><td width="70%%"><a href="{0}"><img src="{0}" width="100%%" /></a></td>'.format(x)
                if y == '':
                    row += '<td width="30%%"></td></tr>'
                else:
                    row += '<td width="30%%"><a href="{0}"><img src="{0}" width="100%%" /></a></td></tr>'.format(y)
                f.write( row )
                
            f.write( '</table></body></html>' )
            
    def block_html( self ):
        """ Writes a trivial block HTML file for the given chip. """
        block = html.HTML( os.path.join( self.results_dir , 'chipDiagnostics_block.html' ) )
        info  = '<html><body><hr>'
        
        if not self.has_chipcal:
            info += '<p><em>WARNING!  Chip Calibration files not found!  Chipcal analyses skipped . . .</em></p>'
            info += '<br>'
            
        if not self.has_rawdata:
            info += '<p><em>WARNING!  Raw acquisition files not found!  Kickback analysis skipped . . .</em></p>'
            info += '<br>'
        
        info += '<h4>Potentially Useful ChipDB Links</h4>'
        info += '<table border="0" cellspacing="0" width="20%">'
        info += '<tr><td width="50%"><b>Lot Report:</b></td><td width="50%"><a href="http://chipdb.ite/lots/{0}/">{0}</a></td></tr>'.format( self.explog.metrics['CMOSLotId'] )
        info += '<tr><td width="50%"><b>Wafer Report:</b></td><td width="50%"><a href="http://chipdb.ite/lots/{0[CMOSLotId]}/{0[WaferId]}/">W{0[WaferId]:02d}</a></td></tr></table><br>'.format( self.explog.metrics )
        
        if self.has_chipcal:
            if self.cc.is_multilane:
                info += '<p><em>This is a multilane chip.  Find more info on it at its <a href="http://chipdb.ite/valkyrie/chipfinder/?barcode={0[ChipBarcode]}" target="_blank">unique page</a> on chipdb.ite.</em></p>'.format( self.explog.metrics )
        info += '</body></html>'
        block.add        ( info )
        block.write      ( )

    def edge_effects( self ):
        """ 
        This function plots column average plots for chipcal metrics.
        Brings in functionality previously located in edgeEffects::noise.py.
        """
        output = html.HTML( os.path.join( self.results_dir , 'edgeEffects.html' ) )
        output.make_header( 'edgeEffects' , use_courier=True )
        
        images = html.table( )
        images.add_row( ['Column average plot','Errorbar column average plot'],[50,50] , th=True )
        for m in ['noise','offset','gain']:
            images.add_row( [html.image_link('{}_colavg.png'.format(m)),html.image_link('{}_colavg_errorbar.png'.format(m))] , [50,50] )
            
        output.add        ( images.get_table( ) )
        output.make_footer( )
        output.write      ( )
        
    def find_refpix( self , gain_cutoff=500 ):
        ''' This doesn't need to be explicitly called, it's also in self.cc.find_refpix. '''
        if not hasattr( self.cc , 'gain' ):
            print( "Error!  Have not yet loaded gain.  Please load and try again." )
            return None
        
        # Create binary footprint for binary_opening operation
        footprint = np.zeros((5,5))
        footprint[1:4,:] = 1
        footprint[:,1:4] = 1
        mask = ndimage.morphology.binary_opening( self.cc.gain < gain_cutoff , structure=footprint , iterations=2 )
        
        # Correct for binary_opening false Falses at extreme corners.
        mask[ 0:2 , 0:2 ] = True
        mask[ 0:2 ,-2:  ] = True
        mask[-2:  , 0:2 ] = True
        mask[-2:  ,-2:  ] = True
        
        self.active = ~mask
        self.refpix = mask

    def instrument( self ):
        """ Takes the place of previous instrument script.  Needs to be run after creating self.explog. """
        if not hasattr( self , 'explog' ):
            print( 'ERROR!  Have not yet loaded explog.  Attempting to load . . .' )
            self.explog = Explog( self.raw_data_dir )
            
        # Create plots from self.explog
        self.explog.pressure_plot ( self.results_dir )
        self.explog.inst_temp_plot( self.results_dir )
        self.explog.cpu_temp_plot ( self.results_dir )
        self.explog.fpga_temp_plot( self.results_dir )
        
        # Create HTML output
        instr = html.HTML( os.path.join( self.results_dir , 'instrument.html' ) )
        instr.make_header( 'Instrument' , use_courier=True )
        
        # Trivial start table
        device = html.table( )
        device.add_row( ['<font size=4, face="Arial">Hardware</font>',''] , [50,50] )
        device.add_row( ['<br>',''] )
        device.add_row( ['Device Name' , self.explog.metrics['DeviceName'] ] , [50,50] , cl='d0')
        instr.add( device.get_table() )
        instr.add( '<br>' )
        
        #########################
        # Temperature table
        #########################
        # Temperature title
        ttl = html.table( )
        ttl.add_row( ['Temperature'] , cl='d0' )
        temp_ttl = ttl.get_table()
        
        # Metrics section
        tmets = html.table( )
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            tmets.add_row( ['Chip Bay','','Cooler',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['ChipBayTemperatureMean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['CoolerTemperatureMean'] ] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['ChipBayTemperatureSD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['CoolerTemperatureSD'] ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['ChipBayTemperature90'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['CoolerTemperature90'] ] )
            tmets.add_row( ['<br>','','',''] )
            tmets.add_row( ['Ambient 1','','Ambient 2',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['Ambient1TemperatureMean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['Ambient2TemperatureMean'] ] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['Ambient1TemperatureSD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['Ambient2TemperatureSD'] ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['Ambient1Temperature90'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['Ambient2Temperature90'] ] )
            
        elif float( self.explog.metrics['PGMHW'] ) == 1.0:
            tmets.add_row( ['Instrument','','Chip',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['InstrumentTemperatureMean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['ChipTemperatureMean'] ] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['InstrumentTemperatureSD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['ChipTemperatureSD'] ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['InstrumentTemperature90'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['ChipTemperature90'] ] )
            
        elif float( self.explog.metrics['PGMHW'] ) == 1.1:
            tmets.add_row( ['Instrument','','Chip',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['InstrumentTemperatureMean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['ChipTemperatureMean'] ] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['InstrumentTemperatureSD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['ChipTemperatureSD'] ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['InstrumentTemperature90'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['ChipTemperature90'] ] )
            tmets.add_row( ['<br>','','',''] )
            tmets.add_row( ['Restrictor','','Heatsink',''] )
            tmets.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['RestrictorTemperatureMean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['HeatsinkTemperatureMean'] ] )
            tmets.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['RestrictorTemperatureSD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['HeatsinkTemperatureSD'] ] )
            tmets.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['RestrictorTemperature90'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['HeatsinkTemperature90'] ] )
            
        temp_metrics = tmets.get_table()
        
        temperature = [ (temp_ttl + '\n' + temp_metrics) , html.image_link( 'instrument_temperature.png' ) ]
        
        #########################
        # Pressure table
        #########################
        # Pressure title
        ttl = html.table( )
        ttl.add_row( ['Pressure'] , cl='d0' )
        pressure_ttl = ttl.get_table()
        
        # Metrics section
        pmets = html.table( )
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            pmets.add_row( ['Regulator','','Manifold',''] )
            pmets.add_row( ['Mean' , '%0.1f psi' % self.explog.metrics['RegulatorPressureMean'] , 
                           'Mean' , '%0.1f psi' % self.explog.metrics['ManifoldPressureMean'] ] )
            pmets.add_row( ['SD' , '%0.1f psi' % self.explog.metrics['RegulatorPressureSD'] , 
                           'SD' , '%0.1f psi' % self.explog.metrics['ManifoldPressureSD'] ] )
            pmets.add_row( ['90%' , '%0.1f psi' % self.explog.metrics['RegulatorPressure90'] , 
                           '90%' , '%0.1f psi' % self.explog.metrics['ManifoldPressure90'] ] )
        else:
            pmets.add_row( ['Mean' , '%0.1f psi' % self.explog.metrics['PressureMean'] , '' , '' ] )
            pmets.add_row( ['SD'   , '%0.1f psi' % self.explog.metrics['PressureSD']   , '' , '' ] ) 
            pmets.add_row( ['90%' , '%0.1f psi' % self.explog.metrics['Pressure90']   , '' , '' ] )
            
        pressure_metrics = pmets.get_table()
        
        pressure = [ (pressure_ttl + '\n' + pressure_metrics) , html.image_link( 'instrument_pressure.png' ) ]
        
        #########################
        # CPU table
        #########################
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            # CPU Temp title
            ttl = html.table( )
            ttl.add_row( ['CPU Temperature'] , cl='d0' )
            cpu_ttl = ttl.get_table()
            
            # Metrics section
            cpu = html.table( )
            cpu.add_row( ['CPU 1','','CPU 2',''] )
            cpu.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['CPU1TemperatureMean'] , 
                          'Mean' , '%0.1f degC' % self.explog.metrics['CPU2TemperatureMean'] ] )
            cpu.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['CPU1TemperatureSD'] , 
                          'SD' , '%0.1f degC' % self.explog.metrics['CPU2TemperatureSD'] ] )
            cpu.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['CPU1Temperature90'] , 
                          '90%' , '%0.1f degC' % self.explog.metrics['CPU2Temperature90'] ] )
            cpu_metrics = cpu.get_table()
            
            cpu_temp = [ (cpu_ttl + '\n' + cpu_metrics) , html.image_link( 'instrument_cpu_temperature.png' ) ]
        
        #########################
        # FPGA table
        #########################
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            # FPGA Temp title
            ttl = html.table( )
            ttl.add_row( ['FPGA Temperature'] , cl='d0' )
            fpga_ttl = ttl.get_table()
            
            # Metrics section
            fpga = html.table( )
            fpga.add_row( ['FPGA 1','','FPGA 2',''] )
            fpga.add_row( ['Mean' , '%0.1f degC' % self.explog.metrics['FPGA1TemperatureMean'] , 
                           'Mean' , '%0.1f degC' % self.explog.metrics['FPGA2TemperatureMean'] ] )
            fpga.add_row( ['SD' , '%0.1f degC' % self.explog.metrics['FPGA1TemperatureSD'] , 
                           'SD' , '%0.1f degC' % self.explog.metrics['FPGA2TemperatureSD'] ] )
            fpga.add_row( ['90%' , '%0.1f degC' % self.explog.metrics['FPGA1Temperature90'] , 
                           '90%' , '%0.1f degC' % self.explog.metrics['FPGA2Temperature90'] ] )
            fpga_metrics = fpga.get_table()
            
            fpga_temp = [ (fpga_ttl + '\n' + fpga_metrics) , html.image_link( 'instrument_fpga_temperature.png' ) ]
            
        # Create main data table
        data   = html.table( )
        cols   = [67,33]
        data.add_row( temperature , cols )
        data.add_row( pressure    , cols )
        
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            data.add_row( cpu_temp    , cols )
            data.add_row( fpga_temp   , cols )
        
        instr.add( data.get_table() )
        
        # Add source comment
        instr.add( '<p><font size=2 face="Arial">Source: %s</font></p>' % os.path.basename( self.explog.log ))
        instr.add( '<hr>' )
        
        # Make Zebra table of Software information.
        if self.explog.chiptype.series.lower() in ['proton','s5']:
            labels = ['Datacollect Version','LiveView Version','Scripts Version','Graphics Version',
                      'OS Version','RSM Version','OIA Version','Reader FPGA Version','Mux FPGA Version',
                      'Valve FPGA Version' ]
        else:
            labels = ['PGM SW Release','Datacollect Version','LiveView Version','Scripts Version',
                      'Graphics Version','OS Version','Firmware Version','FPGA Version','Driver Version',
                      'Board Version','Kernel Build']
        values = [ self.explog.metrics[x.replace( ' ','' )] for x in labels ]
        
        instr.add( '<p><font size=4, face="Arial">Software</font><p>' )
        instr.add( '<br>' )
        
        software = html.table( zebra=True )
        # software.add_row( ['<font size=4, face="Arial">Software</font>',''] , [50,50] )
        for pair in zip( labels , values ):
            software.add_row( pair , [50,50] )
        
        instr.add( software.get_table() )
        instr.add( '<p><font size=2 face="Arial">Source: %s</font></p>' % os.path.basename( self.explog.log ))
        instr.add( '<hr>' )
        
        # Add footer
        instr.make_footer( )
        
        # Write instrument.html!
        instr.write( )
        return None
        
    def compile_json( self ):
        ''' 
        Compiles json files from sub analyses.  
        Can be called over and over to recompile and resave the json file.
        '''
        csv     = False
        results = {}
        jsonout = os.path.join( self.results_dir , 'results.json' )
        
        # Delete json file if it currently exists.
        if os.path.exists( jsonout ):
            os.remove( jsonout )
        
        json_files = ['noise.json','gain.json','pix.json','noise_vs_offset.json', 'kickback.json']
        
        for js in json_files:
            jsfile = os.path.join( self.results_dir , js )
            if os.path.exists( jsfile ):
                try:
                    with open( jsfile , 'r' ) as f:
                        loaded = json.load( f )

                        key = js.split('.')[0]
                        if key == 'pix':
                            key = 'offset'
                        results[key] = {}
                        for met in loaded:
                            if key == met.split('_')[0]:
                                # This now handles getting rid of noise_true_noise_localstd to true_localstd
                                # But also handles going from noise_q2 to just q2
                                new_metric = met.replace('{}_'.format( key ),'' )
                                results[key][new_metric] = loaded[met]
                            else:
                                results[key][met] = loaded[met]
                except:
                    print 'Error reading %s' % js

        # Add in dynamic range used in the analysis....for posterity
        results['used_dynamic_range'] = self.explog.DR
        
        # Add in results from pinned pixels
        pinned_metrics = ['PixelLow','PixelHigh','PixelInactive','PixelInRange','PixelCount','PercPinnedLow',
                          'PercPinnedHigh','PinnedLowThreshold','PinnedHighThreshold','PercPinned',
                          'PercPinned_TB_Diff']
        for m in pinned_metrics:
            results['offset'][m] = self.cc.metrics[m]
            
        # Add in results from explog
        results['explog'] = self.explog.metrics
        
        # Add in wafermap limits
        results['wafermap_lims'] = self.wafermap_lims
        
        # Add in sneaky clusters
        results[ 'sneaky_superpixel_count'  ] = self.cc.metrics[ 'sneaky_superpixel_count' ]
        results[ 'sneaky_superpixel_matrix' ] = self.cc.metrics[ 'sneaky_superpixel_matrix' ]
        results[ 'perc_pinned_thresholds'   ] = self.cc.metrics[ 'perc_pinned_thresholds' ]
        results[ 'offset_local_thresholds'  ] = self.cc.metrics[ 'offset_local_thresholds' ]
        
        # Add multilane analysis metrics
        for m in ['is_multilane','lane_1','lane_2','lane_3','lane_4']:
            results[ m ] = getattr( self.cc , m , False )
            
        results['lane_metrics'] = self.cc.lane_metrics
        
        # Add local_pinned metrics
        for met in ['pinned_low', 'pinned_high', 'pinned']:
            for n in ['_all','']:
                # Skip non-HD metrics for now.
                # for suffix in ['_hd','']:
                for suffix in ['_hd']:
                    metric = 'local_{}{}{}'.format( met, n, suffix ) 
                    results[metric] = self.cc.metrics.get( metric, {} )
        
        # Add in edge_analyzer metrics
        results['edge_metrics'] = self.edge_metrics
        
        print 'Writing results.json file . . .'
        misc.serialize( results )
        with open( jsonout , 'w' ) as f:
            json.dump( results , f )
            
        if csv:
            print 'Writing csv file: ' + csv_path
            with open( csv_path, 'w' ) as f:
                keys = sorted( results.keys() )
                for key in keys:
                    f.write( '%s, ' % key )
                f.write('\n')
                for key in keys:
                    f.write( '%s, ' % results[key] )
                    
    def pixel_uniformity( self ):
        """ Creates pixel uniformity output page just as is done for PixelUniformity Plugin """

        # Change this for version 4.0.0 to focus on only the true_<metric>_hd metrics.
        def metric_table( metric_prefix , title ):
            ''' local function to create a metric table '''
            output = html.table( )
            output.add_row( [title,''] , [70,30] , cl='d0' )
            names = ['Median (Q2)','IQR','Mode','Stdev','90th Percentile']
            mets  = ['q2','iqr','mode','std','P90']
            for ( name , met ) in zip( names , mets ):
                output.add_row( [ name,'%0.0f' % self.cc.metrics['%s_%s' % (metric_prefix,met)] ] , [70,30] )
            return output.get_table()
        
        pu = html.HTML( os.path.join( self.results_dir , 'PixelUniformity.html' ) )
        pu.make_header( 'Pixel Uniformity' ,  use_courier=True )
        
        # General metrics and info
        toptable = html.table()
        toptable.add_row( ['Chip Info','&nbsp'] , [50,50] , cl='d0' )
        toptable.add_row( ['Lot',self.explog.metrics['CMOSLotId']] , [50,50] )
        toptable.add_row( ['Wafer',self.explog.metrics['WaferId']] , [50,50] )
        coords = '(%d,%d)' % (self.explog.metrics['WaferX'],self.explog.metrics['WaferY'])
        toptable.add_row( ['(X,Y)', coords                       ] , [50,50] )
        toptable.add_row( ['Area',self.explog.metrics['Area']    ] , [50,50] )
        pu.add( toptable.get_table() )
        pu.add( '<br><hr><br>'       )
        
        # Offset Localstd table
        # 4 sections: metric table, spatial, colavg, histogram
        #ol_mets = html.table( )
        #ol_mets.add_row( ['Offset Local Stdev',''] , [70,30] , cl='d0' )
        #names = ['Median (Q2)','IQR','Mode','Stdev','90th Percentile']
        #mets  = ['median','iqr','mode','std','P90']
        #for ( name , met ) in zip( names , mets ):
        #    ol_mets.add_row( [name,'%d' % self.cc.metrics['offset_localstd_%s' % met]] , [70,30] )

        types = ['offset','gain','noise']
        try:
            offset_local = metric_table( 'offset_true_localstd_hd' , 'Offset True Local Stdev' )
            gain_local   = metric_table( 'gain_true_localstd_hd'   , 'Gain True Local Stdev' )
            noise_local  = metric_table( 'noise_true_localstd_hd'  , 'Noise True Local Stdev' )
            spatials  = ['{}_true_localstd_hd_spatial.png'.format(t) for t in types ]
            colavgs   = ['{}_true_localstd_hd_colavg.png'.format(t) for t in types ]
            histograms= ['{}_true_localstd_hd_histogram.png'.format(t) for t in types ]
        except KeyError:
            offset_local = metric_table( 'offset_localstd' , 'Offset Local Stdev' )
            gain_local   = metric_table( 'gain_localstd'   , 'Gain Local Stdev' )
            noise_local  = metric_table( 'noise_localstd'  , 'Noise Local Stdev' )
            spatials  = ['{}_localstd_spatial.png'.format(t) for t in types ]
            colavgs   = ['{}_localstd_colavg.png'.format(t) for t in types ]
            histograms= ['{}_localstd_histogram.png'.format(t) for t in types ]
            
        main = html.table( )
        w    = [25,25,25,25]
        main.add_row( ['Metrics','Spatial Map','Column Average','Histogram'] , w , th=True )
        
        metric_tables = [ offset_local , gain_local , noise_local ]
        
        for (a,b,c,d) in zip( metric_tables , spatials , colavgs , histograms ):
            main.add_row( [ a , html.image_link( b ) , html.image_link( c ) , html.image_link( d ) ] , w )

        # Add perc pinned pixel plots
        main.add_row( [ '<center>% Pixels Pinned Low</center>' , 
                        html.image_link( 'perc_pinned_low_hs_spatial_hd.png' ) ,
                        html.image_link( 'perc_pinned_low_full_spatial_hd.png' ) , 
                        html.image_link( 'perc_pinned_low_histogram_hd.png' ) ] , w )
        main.add_row( [ '<center>% Pixels Pinned High</center>' , 
                        html.image_link( 'perc_pinned_high_hs_spatial_hd.png' ) ,
                        html.image_link( 'perc_pinned_high_full_spatial_hd.png' ) , 
                        html.image_link( 'perc_pinned_high_histogram_hd.png' ) ] , w )
        main.add_row( [ '<center>Total % Pixels Pinned</center>' , 
                        html.image_link( 'perc_pinned_hs_spatial_hd.png' ) ,
                        html.image_link( 'perc_pinned_full_spatial_hd.png' ) , 
                        html.image_link( 'perc_pinned_histogram_hd.png' ) ] , w )
        
        pu.add( '<h2>Local Standard Deviation Analysis</h2>' )
        pu.add( main.get_table() )
        pu.add( '<br><hr><br>' )
        
        # Diff images
        pu.add( '<h2>Difference images</h2>' )
        diff_img = html.table( )
        #diff_img.add_row( [ html.image_link('%s_diff_img.png' % x ) for x in ['offset','gain','noise']],[33,33,33])
        diff_img.add_row([html.image_link('true_%s_diff_img_hd.png' % x ) for x in ['offset','gain','noise']],[33,33,33])
        pu.add( diff_img.get_table() )
        
        pu.make_footer( )
        pu.write      ( )
        
        return None
        
    def conversion_noise( self ):
        '''
        creates HTML output page for looking at noise potentially caused by simultaneous pixel conversion in ramp
        requires self.analyze_conversion( ) to be run.
        '''
        if not hasattr( self , 'no' ):
            print( 'ERROR!  Have not yet analyzed conversion noise.  Attempting to run . . .' )
            self.analyze_conversion( )

        # Prepare list of figure file names
        prefixes  = ['fc','q1','q2','q3','q4']
        cchists   = [ '%s_no_cchist.png' % prefix for prefix in prefixes ]
        bincounts = [ '%s_noise_vs_bincount.png' % prefix for prefix in prefixes ]
        fittedbc  = [ '%s_noise_vs_bincount_fitted.png' % prefix for prefix in prefixes ]
        nvo       = [ '%s_noise_vs_offset.png' % prefix for prefix in prefixes ]
        nvo_fixed = [ '%s_noise_vs_offset_fixed.png' % prefix for prefix in prefixes ]

        cn = html.HTML( os.path.join( self.results_dir , 'conversion_noise.html' ) )
        cn.make_header( 'Conversion Noise' , use_courier=True )

        # Make brief description of this analysis.
        cn.add( '<h2>Analysis Introduction and Overview</h2>' )
        cn.add( textwrap.dedent( '''\
                  <p><em>Our goal here is to analyze chip noise and determine if a pixel's noise is correlated to its pixel offset voltage.  While it may not be immediately clear, the hypothesis for this correlation is rooted in how the analog-to-digital converter (ADC) converts pixel voltage into a digital signal that is then sent off-chip.  Each column is read out for every row, and the conversion event happens more or less when a particular pixel's voltage is equal to the voltage in a predefined downward "ramp" voltage used for comparision.</em></p>
                  <p><em>In particular, we are concerned that if many, many pixels have similar voltages and "convert" at the same time, there would be some extra noise added into the pixel noise due to disturbance of the ramp signal, which is the same signal shared across many columns.  Results are plotted by full chip as well as by quadrant.</em></p>
                  <br>
                  <hr>
                  ''' ) )

        # Table of all plots
        cn.add( '<h2>Plots by Chip Region</h2>' )
        by_region = html.table( )
        w         = [12,22,22,22,22]
        row_labels= [ '<center><b>%s</b></center>' % prefix.upper() for prefix in prefixes ]
        by_region.add_row( ['Region','Color-coded Histogram','Noise vs. Offset - Fixed Y Scale','Noise vs. Bincount',
                            'Fitted Noise vs. Bincount'] , w , True , th=True )
        for i in range(5):
            by_region.add_row( [ row_labels[i] , html.image_link( cchists[i] ) , html.image_link( nvo_fixed[i] ) ,
                                 html.image_link( bincounts[i] ) , html.image_link( fittedbc[i] ) ] , w )
        cn.add( by_region.get_table() )

        # Now add quadrant plots
        cn.add( '<br><hr><br>' )
        cn.add( '<h2>Quadrant Plots</h2>' )
        cn.add( '<p><em>Note that quadrants are plotted with software origin at lower left (row=0) and that region is defined as quadrant #1.  The quadrants then proceed counter-clockwise from there.</em></p>' )
        
        plotnames = ['Color-coded Histogram','Noise vs. Offset','Noise vs. Offset - Fixed Y Scale','Noise vs. Bincount','Fitted Noise vs. Bincount']
        plotlists = [ cchists , nvo , nvo_fixed , bincounts , fittedbc ]
        for j in range(len(plotnames)):
            cn.add( '<center><h3>%s</h3></center>' % plotnames[j] )
            
            qp = html.table( width=60 , border=1 )
            qp.add_row( [ html.image_link( plotlists[j][4] ) , html.image_link( plotlists[j][3] ) ] , [50,50] )
            qp.add_row( [ html.image_link( plotlists[j][1] ) , html.image_link( plotlists[j][2] ) ] , [50,50] )
            cn.add( '<center>%s</center>' % qp.get_table() )
            cn.add( '<br>' )
            
        # Add a fit metric table
        cn.add( '<hr><br>' )
        cn.add( '<h2>Conversion Noise Linear Fit Data</h2>' )
        
        fits = html.table( zebra=True )
        fitw = [40,12,12,12,12,12]
        fits.add_row( ['Region','Slope * 10^-6','Intercept','R-squared','P-Value','Std. Error'] , fitw , th=True )
        for m in range(5):
            fields = [ row_labels[m] ]
            for metric in ['slope','intercept','rsq','pval','std_err']:
                fields.append( '<center>%.2f</center>' % self.no.metrics[ '%s_noise_vs_bincount_%s' % ( prefixes[m] , metric ) ] )
                
            fits.add_row( fields , fitw )

        cn.add( fits.get_table() )
        
        # Write HTML
        cn.make_footer( )
        cn.write      ( )
        
        return None
        
    def multilane_analysis( self ):
        """ Creates HTML page for multilane analysis, only runs if the chip is actually multilane. """
        # Define several handy helper functionsn
        def rotated_img( imgpath , cls='transpose' , width=100 ):
            ''' Returns code for displaying an image also as a link '''
            text = '<a href="%s"><img class="%s" src="%s" width="%d%%" /></a>' % ( imgpath, cls , imgpath , width )
            return text
        
        def lane_img( imgpath , height=100 ):
            ''' Returns code for displaying an image also as a link '''
            #text = '<a href="%s"><img src="%s" height="%d%%" /></a>' % ( imgpath, imgpath , height )
            text = '<a href="%s"><img src="%s" width="%d%%" /></a>' % ( imgpath, imgpath , height )
            return text
        
        def get_label( metric ):
            """ creates a fancy label for a given metric """
            m = metric.lower()
            
            special = {'std': 'SD' , 'q2': 'Median (Q2)' , 'p90': '90th Percentile' , 'iqr': 'IQR' }
            
            if metric in special:
                return special[m]
            else:
                return m.title()
            
        def iter_lanes( ):
            """ 
            Helper iterator for looping through lanes. 
            returns lane number, its name, and if it's active.
            """
            for i in range(1,5):
                name = 'lane_{}'.format(i)
                yield ( i , name , getattr( self.cc , name ) )
                
        def create_section( chip_metric , units , metrics , fmt ):
            """
            chip_metric = noise, gain, offset_localstd, for instance
            units is the string for units of metric of interest, e.g. uV for noise
            metrics are a list of metrics wanting displayed.  empty strings are interpreted as row skips.
            fmt is a string formatter (for %-based formatting)
            """
            # This defines the total table for this main chip metric
            widths     = [40,5,5,5,5,5,5,30]
            section    = html.table()
            section.add_row( ['<font size=4, face="Arial">%s</font>' % chip_metric.title(),'','','','','','',''], widths )
            section.add_row( ['<br>','','','','','','',''] , widths )
            
            section_data_row = ['','','','','','','',html.image_link('multilane_{}_boxplot.png'.format(chip_metric) ) ]
            
            # TODO
            # - decide where to put units.  Don't want a unit spam.  Favorite options are in header or extra column
            
            # This defines the little metric table that lives within the above table
            met_widths = [40,15,15,15,15]
            met_table  = html.table()
            met_table.add_row( ['Pixel %s' % chip_metric.title() , units , '' , '' , '' ], met_widths, cl='d0')
            met_table.add_row( ['Lane','1','2','3','4'] , met_widths )
            
            # Handle pulling metrics for each lane
            # Metrics are saved in self.lane_metrics['lane_#'][metric]['Q2'], for example
            for metric in metrics:
                if metric == '':
                    # This is to be skipped.   
                    met_table.add_row( ['&nbsp','','','',''] , met_widths )
                    continue
                metric_row = [ get_label(metric) ]
                for i, lane, active in iter_lanes( ):
                    if active:
                        # Get the metric
                        if chip_metric in self.cc.lane_metrics[lane]:
                            val = self.cc.lane_metrics[lane][chip_metric][metric]
                        else:
                            val = 0.
                            
                        metric_row.append( fmt % val )
                    else:
                        metric_row.append( '-' ) # trying this on for size.  Don't want a million "n/a"s
                        
                met_table.add_row( metric_row , met_widths )
                
            # If this is gain, let's add addressible wells here, in MPixels
            if chip_metric.lower() == 'gain':
                # First add another spacer row
                met_table.add_row( ['&nbsp','','','',''] , met_widths )
                metric_row = [ 'Addressable Wells (M)' ]
                met_name   = 'addressable_wells'
                for i, lane, _ in iter_lanes( ):
                    if met_name in self.cc.lane_metrics[lane]:
                        val ='{:.1f}'.format( self.cc.lane_metrics[lane][met_name] / 1e6 )
                    else:
                        val = '0'
                    metric_row.append( val )
                    
                met_table.add_row( metric_row , met_widths )
                
            # Add metric table to the section
            section_data_row[0] = met_table.get_table()
            
            # Add the relevant wafermap image
            for i, lane, active in iter_lanes( ):
                if active:
                    # We are using i+1 here because an extra 5% width td spacer is used on each side of the images.
                    imgpath             = '{}_{}_wafermap_nonT.png'.format( lane , chip_metric )
                    section_data_row[i+1] = '<center>{}</center>'.format( lane_img( imgpath ) )
                else:
                    # This is an inactive lane
                    section_data_row[i+1] = '&nbsp'
                    
            # Complete the section
            section.add_row( section_data_row , widths )
            
            if 'gain' in chip_metric:
                f = self.cc.gainfile
            elif 'noise' in chip_metric:
                f = self.cc.noisefile
            elif 'offset' in chip_metric:
                f = self.cc.pixfile
            footer = '<font size=2 face="Arial">Source: %s</font>' % os.path.basename( f )
            section.add_row( [ footer , '', '', '', '', '', '', '' ] , widths )
            
            return section.get_table()
        
        # OK, let's actually start this one
        if not self.cc.is_multilane:
            print( 'This chip was not identified as a multilane chip.  Skipping multilane analysis display.' )
            return None
        
        valkyrie = html.HTML( os.path.join( self.results_dir , 'multilane_analysis.html' ) )
        
        # Make header, but also add a flip and rotate image tool.
        transpose = '<style type="text/css">img.transpose { transform: rotate(-90deg) scaleX(-1); -webkit-transform: rotate(-90deg) scaleX(-1); }</style>'
        valkyrie.make_header( 'Multilane Analysis' , use_courier=True , styles = [transpose] )

        # Now let's add all the sections
        
        # Conscious decision to only show most important metrics
        # use of a '' will force loop to put in an empty table row, useful to separate quartiles from means
        metrics = ['','q2','iqr','','P90','','mean','std']
        
        for chip_metric in ['gain','gain_true_localstd_hd','noise','noise_true_localstd_hd','offset','offset_true_localstd_hd']:
            if 'gain' in chip_metric:
                units = self.cc.gain_units
                fmt   = '%4.0f'
            elif 'noise' in chip_metric:
                units = self.cc.noise_units
                fmt   = '%3.0f'
            elif 'offset' in chip_metric:
                units = self.cc.offset_units
                fmt   = '%3.0f'
            else:
                units = ''
                fmt   = '%s'
                
            valkyrie.add( create_section( chip_metric , units , metrics , fmt ) )
            valkyrie.add( '<hr>' )
            
            # add a line separator between major metrics
            #if 'localstd' in chip_metric:
            #    valkyrie.add( '<hr>' )
            #else:
            #    valkyrie.add( '<br>' )
            #    valkyrie.add( '<p>&nbsp</p>' )
            #    valkyrie.add( '<br>' )
                
        valkyrie.make_footer( )
        valkyrie.write( )
        return None
    
    def output( self ):
        pass
    
    def report( self ):
        pass
    
    def metric( self ):
        pass
    
if __name__ == "__main__":
    PluginCLI()
