#!/usr/bin/env python
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
import os, sys, textwrap
import json

# Import custom data parsing and plotting module
from tzero import TZero
from tools import explog, chiptype, misc
from tools.PluginMixin import PluginMixin

from ion.plugin import *

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def image_link( imgpath , width=100 ):
    ''' Returns code for displaying an image also as a link '''
    text = '<a href="%s"><img src="%s" width="%d%%" /></a>' % ( imgpath, imgpath , width )
    return text

class autoCal( IonPlugin , PluginMixin ):
    """ 
    Plugin for visualization of instrument & chip performance
    Author:      Phil Waggoner
    Last Updates: 
    
    13 May 2019 | Added ReleaseVersion and RFIDMgrVersion, as well as scraping if development Valkyrie Workflow Scripts were used.
    22 Apr 2019 | Bugfix to explog code to catch erroneous explog_final.txt files.
    03 Apr 2019 | Bugfix to ensure we try to get more Valkyrie data first from per-flow data.
    01 Apr 2019 | Added new Valkyrie flow rate and temperature data, plus explog lane active info.
    31 Mar 2019 | Updated explog_final flow_data match for new valkyrie pattern
    15 Jul 2019 | Updated tools for +/- inf handling
    24 Jan 2020 | Tools update for CSA
    16 Mar 2020 | Fixed dual gain curve for handling standard cal
    10 Aug 2020 | Updated to handle resequencing runs on Valkyrie.
    17 AUg 2020 | Updated tools
    """
    version       = "2.1.1"
    allow_autorun = True
    
    runtypes      = [ RunType.THUMB , RunType.FULLCHIP ]
    
    def launch( self ):
        # Get metadata
        print('Start')
        self.init_plugin( )
        
        # print 'raw data dir: ' + self.raw_data_dir
        # print 'analysis dir: ' + self.analysis_dir
        
        # Analyze Data
        # Explog final is read in by default in the init_plugin function.
        if not self.explog:
            print( 'No explog files found, aborting plugin.' )
            sys.exit(0)
        elif self.explog.final:
            # Make the requisite plots and finnagle other metrics
            self.explog.make_all_plots  ( self.results_dir )
            self.derive_synclink_metrics( )
            
            # Calculate run time.  This is only populated if we have the explog_final.txt file.
            self.metrics['run_time'] = self.explog.get_run_time( )
        else:
            print( 'Only the initial explog file was found!  Skipping most plotting and synclink metrics...' )
            self.explog.gain_curve( self.results_dir )
            
        # Overwrite explog.metrics['chiptype_name'] with validated chiptype.
        self.explog.metrics['chiptype_name'] = self.ct.name
        
        # For the case of 560 chips, let's grab the initial explog to find out if we did post-run calibration
        self.dual_gain_curve( )
        
        # In the case that we have a reseq run, we should plot the two gain curves, just for fun.
        if self.explog.is_reseq:
            self.reseq_gain_curve( )
            
        self.analyze_T0( )
        self.write_html( )
        
        # Write out results.json after combining T0 and explog metrics.
        print( 'Writing results.json' )
        self.metrics.update( self.explog.metrics )

        # Account for reseq gain curves.  Historically, reseq gain curves have been ignored and the GainCurve[Gain\Vref] are from the initial run.
        # To keep that convention, we need to swap and create new metrics.
        # May eventually want additional initial-seq run metrics here, but it's not clear if any of them would be useful at this juncture.
        if self.explog.is_reseq:
            # Seq gain curve
            self.metrics['GainCurveVref'] = self.first_explog.metrics['GainCurveVref']
            self.metrics['GainCurveGain'] = self.first_explog.metrics['GainCurveGain']
            
            # Reseq gain curve
            self.metrics['Reseq_GainCurveVref'] = self.explog.metrics['GainCurveVref']
            self.metrics['Reseq_GainCurveGain'] = self.explog.metrics['GainCurveGain']
            
        misc.serialize( self.metrics )
        with open( os.path.join( self.results_dir , 'results.json' ) , 'w' ) as f:
            json.dump( self.metrics , f )
            
        print( 'Plugin complete.' )

    def analyze_T0( self ):
        ''' wrapper function that handles analyzing tzero from the run and, if applicable, resequencing. '''
        # Analyze the T0 data from seq and reseq runs, if applicable.
        initial_t0 = self.analyze_debugT0( use_reseq=False )
        if self.explog.is_reseq:
            reseq_t0 = self.analyze_debugT0( use_reseq=True )
            
            # Make a diff image as well
            T0_diff = reseq_t0.t0 - initial_t0.t0
            plt.figure  ( )
            plt.imshow  ( T0_diff , origin='lower', interpolation='nearest' , clim=[-5,5], cmap='seismic' )
            plt.xlabel  ( 'Column Block' )
            plt.ylabel  ( 'Row Block' )
            plt.title   ( 'Different in Estimated T0 (Frame) [Reseq-Seq]' )
            plt.colorbar( shrink=0.7 )
            plt.savefig ( os.path.join( self.results_dir , 't0_difference_plot.png' ) )
            plt.close   ( )

            # Make an additional html file that simply shows the t0 plots side by side.
            html = textwrap.dedent( '''\
            <html><body>
            <table border="0" cellspacing="0" cellpadding="0">
            <tr>
            <td width="33%%">Initial run</td>
            <td width="33%%">Resequencing</td>
            <td width="33%%">Delta (Reseq - Seq)</td>
            </tr>
            <tr>
            <td width="33%%"><a href="t0_spatial.png"><img src="t0_spatial.png" width="100%%" /></a></td>
            <td width="33%%"><a href="reseq/t0_spatial.png"><img src="reseq/t0_spatial.png" width="100%%" /></a></td>
            <td width="33%%"><a href="t0_difference_plot.png"><img src="t0_difference_plot.png" width="100%%" /></a></td>
            </tr>
            </table>
            </body></html>''' )
            with open( os.path.join( self.results_dir, 't0_reseq.html' ), 'w' ) as f:
                f.write( html )
                
    def analyze_debugT0( self, use_reseq=False ):
        ''' runs T0/vfc subanalysis '''
        if use_reseq:
            debug_file = os.path.join( self.raw_data_dir, 'reseq', 'T0Estimate_dbg_final.json' )
            outdir     = os.path.join( self.results_dir, 'reseq' )
            if not os.path.exists( outdir ):
                os.mkdir( outdir )
        else:
            outdir = self.results_dir
            # Check if it's a thumbnail:
            if os.path.basename( self.raw_data_dir ) == 'thumbnail':
                debug_file = os.path.join( os.path.dirname( self.raw_data_dir ) , 'T0Estimate_dbg_final.json' )
            else:
                debug_file = os.path.join( self.raw_data_dir , 'T0Estimate_dbg_final.json' )
            
        self.tzero_errors = []
        if os.path.exists( debug_file ):
            print('Load T0 Debug JSON.')
            tz         = TZero( debug_file , outdir )
            
            print('Plot data and create html.')
            tz.t0_spatial_plot ( )
            tz.cycle_regions   ( )
            tz.close_html      ( )
            tz.write_html      ( )
            
            # Copy debug file here for later use.
            os.system( 'cp %s %s' % (debug_file , os.path.join(outdir , 'T0Estimate_dbg_final.json')) )
            
            # add summary of errors found to the scroll box if we found any. (right now only for initial run)
            if not use_reseq:
                if len( tz.warnings['fake_t0'] ) == 0:
                    front_porch_count = 0
                else:
                    front_porch_count = len( list( set( [ e.keys()[0] for e in tz.warnings['fake_t0'] ] )))
                    
                if (len( tz.warnings['early'] ) > 0) or (len( tz.warnings['late'] ) > 0) or (front_porch_count > 0):
                    self.tzero_errors.append( '<h3>T0 Debug Errors:</h3>' )
                    self.tzero_errors.append( 'Total early T0 warnings: <b>%d</b>' % len( tz.warnings['early'] ) )
                    self.tzero_errors.append( 'Total late T0 warnings: <b>%d</b>' % len( tz.warnings['late'] ) )
                    self.tzero_errors.append( 'Total front porch warnings: <b>%d</b>' % front_porch_count )
                    
                # Save errors/warnings to metrics, which will later be stored in results.json.
                self.metrics['tzero_warning_early']       = len( tz.warnings['early'] )
                self.metrics['tzero_warning_late']        = len( tz.warnings['late']  )
                self.metrics['tzero_warning_front_porch'] = front_porch_count
            return tz
        else:
            print( 'Skipping T0 analysis, T0Estimate_dbg_final.json file not found.' )
            if not use_reseq:
                self.metrics['tzero_warning_early']       = int( 0 )
                self.metrics['tzero_warning_late']        = int( 0 )
                self.metrics['tzero_warning_front_porch'] = int( 0 )
            return None
                
    def derive_synclink_metrics( self ):
        """ Pulls in and synthesizes synclink metrics from self.explog.metrics """
        ll = self.explog.linklosses
        rs = self.explog.regionslips
        
        self.metrics[  'linkloss_regions'] = ll.get('regions', int(0) )
        self.metrics['regionslip_regions'] = rs.get('regions', int(0) )
        
        if 'total' in ll:
            self.metrics['linkloss_flows'  ]   = int( ll['total'].get( 'instances' , 0 ) )
        else:
            self.metrics['linkloss_flows'  ]   = int( 0 )
            
        if 'total' in rs:
            self.metrics['regionslip_flows'  ] = int( rs['total'].get( 'instances' , 0 ) )
        else:
            self.metrics['regionslip_flows'  ] = int( 0 )
            
        self.metrics['linklosses' ] = ll
        self.metrics['regionslips'] = rs
        
    def dual_gain_curve( self ):
        """ If we have a post-run calibration, let's try to make a dual gain curve plot and overwrite gain_curve"""
        # explog.txt will have the gain curve from the initial calibration
        # explog_final.txt will have the post-run gain curve.
        if (self.explog.final and self.explog.found) and self.has_explog:
            if os.path.exists( os.path.join( self.explog_dir , 'explog.txt' ) ):
                self.initial_explog = explog.Explog( os.path.join( self.explog_dir   , 'explog.txt' ) )
            elif os.path.exists( os.path.join( self.analysis_dir , 'explog.txt' ) ):
                self.initial_explog = explog.Explog( os.path.join( self.analysis_dir , 'explog.txt' ) )
            else:
                return None

        # Save metrics in explog for later use . . .
        self.metrics['PostRun_GainCurveGain'] = self.initial_explog.metrics['GainCurveGain']
        self.metrics['PostRun_GainCurveVref'] = self.initial_explog.metrics['GainCurveVref']
        
        vref_initial = np.array( [ x for x in self.initial_explog.metrics['GainCurveVref'] if x != 0. ] )
        gain_initial = np.array( [ x for x in self.initial_explog.metrics['GainCurveGain'] if x != 0. ] )
        
        vref_final   = np.array( [ x for x in self.explog.metrics['GainCurveVref'] if x != 0. ] )
        gain_final   = np.array( [ x for x in self.explog.metrics['GainCurveGain'] if x != 0. ] )
        
        if np.all( vref_initial == vref_final ) and np.all( gain_initial == gain_final ):
            print( "Post run gain curve does not appear to have been performed.  Skipping." )
        elif ( vref_initial.size == 1 ) or ( vref_final.size == 1):
            print( "Post run cal appears to have been performed but a gain curve was not generated during one of the cals.  Skipping dual gain curve." )
        else:
            print( "Post run gain curve detected!  Creating new plot." )
            
            # Move the auto-created gain curve file to a new name with _final
            os.rename( os.path.join( self.results_dir , 'gain_curve.png' ) ,
                       os.path.join( self.results_dir , 'gain_curve_final.png' ) )
            
            plt.figure ( )
            plt.plot   ( vref_initial , gain_initial , 'bo-' , label="Pre-run"  )
            plt.plot   ( vref_final   , gain_final   , 'ro-' , label="Post-run" )
            plt.xlabel ( 'Reference Electrode Voltage (V)' )
            plt.ylabel ( 'Chip Gain (V/V)' )
            plt.ylim   ( 0 , 1.25 )
            
            # Selected Vref at beginning
            plt.axvline( self.initial_explog.metrics['dac'] , ls='--' , color='blue' )
            plt.text   ( self.initial_explog.metrics['dac'] + 0.025 , 0.1 ,
                         'PreRun Vref=%1.2f V' % self.initial_explog.metrics['dac'] ,
                         fontsize=10 , weight='semibold' , color='blue' , rotation=90 , va='bottom' )

            # Selected Vref at the end
            plt.axvline( self.explog.metrics['dac'] , ls='--' , color='red' )
            plt.text   ( self.explog.metrics['dac'] - 0.04 , 0.1 ,
                         'PostRun Vref=%1.2f V' % self.explog.metrics['dac'] ,
                         fontsize=10 , weight='semibold' , color='red' , rotation=90 , va='bottom' )
            
            # Calculate how much VREF has shifted from beginning to end of the run.
            vref_shift = float( self.explog.metrics['dac'] ) - float( self.initial_explog.metrics['dac'] )
            self.metrics['vref_shift'] = vref_shift
            
            plt.title  ( 'Calibration Gain Curve | Vref Shift: {:.2f}'.format( vref_shift ) )
            plt.grid   ( )
            plt.legend ( loc='best' )
            plt.savefig( os.path.join( self.results_dir , 'gain_curve.png' ) )
            plt.close  ( )

            # Calculate the gain falloff
            dac0 = self.initial_explog.metrics['dac'] 
            gcv0 = self.initial_explog.metrics['GainCurveVref']
            gcg0 = self.initial_explog.metrics['GainCurveGain']
            dacf = self.explog.metrics['dac'] 
            gcvf = self.explog.metrics['GainCurveVref']
            gcgf = self.explog.metrics['GainCurveGain']

            def interp( val, xp, yp ):
                print( val, xp, yp )
                if val in xp:
                    print( 'found val in x-list' )
                    return yp[xp.index(val)]
                else:
                    print( 'attempting narrow interpolation' )
                    xl = xp[0]
                    yl = yp[0]
                    # Figure out where we crossed vref
                    for xr, yr in zip( xp[1:], yp[1:] ):
                        if ( xl - val ) * ( xr - val ) < 0:
                            print( val, [xl,xr], [yl,yr] )
                            return np.interp( val, [xl,xr], [yl,yr] )
                        xl, yl = xr, yr
                print( 'falling back to wide interpolation' )
                return np.interp( val, xp, yp )
            gain_init       = interp( dac0, gcv0, gcg0 )
            gain_final      = interp( dacf, gcvf, gcgf )
            gain_projected  = interp( dac0, gcvf, gcgf )
            gain_loss       = gain_projected - gain_init
            gain_loss_perc  = 100. * gain_loss / gain_init 
            gain_loss_ideal = gain_final - gain_init
            self.metrics['gain_max_init']        = gain_init
            self.metrics['gain_max_final']       = gain_final
            self.metrics['gain_projected_final'] = gain_projected
            self.metrics['gain_loss']            = gain_loss
            self.metrics['gain_loss_perc']       = gain_loss_perc
            self.metrics['gain_loss_ideal']      = gain_loss_ideal

    def reseq_gain_curve( self ):
        ''' creates new gain curve plot with both chip calibrations. '''
        self.first_explog = explog.Explog( self.raw_data_dir )
        
        os.rename( os.path.join( self.results_dir , 'gain_curve.png' ) ,
                   os.path.join( self.results_dir , 'gain_curve_reseq.png' ) )
        
        # Create initial gain curve and rename
        self.first_explog.gain_curve( self.results_dir )
        os.rename( os.path.join( self.results_dir , 'gain_curve.png' ) ,
                   os.path.join( self.results_dir , 'gain_curve_seq.png' ) )
        
        def clean_data( eo ):
            # We have the data and can make a plot!
            vref   = np.array( [ x for x in eo.metrics['GainCurveVref'] if x != 0. ] )
            gain   = np.array( [ x for x in eo.metrics['GainCurveGain'] if x != 0. ] )
            
            # The following code added for Valkyrie alphas.  Not a bad idea really.
            if (not vref.any()) and (not gain.any()):
                print( 'Gain curve information was empty or all zeroes!  Skipping gain plot.' )
                return None
            
            return vref, gain
            
        plt.figure ( )
        colors = ['blue','green']
        first_log  = clean_data( self.first_explog )
        second_log = clean_data( self.explog )
                
        if first_log is not None:
            plt.plot   ( first_log[0],  first_log[1] ,  'bo-', label='1st Run' )
            plt.axvline( self.first_explog.metrics['dac'] , ls='--' , color='blue', label='1st Vref' )
        if second_log is not None:
            plt.plot   ( second_log[0], second_log[1] , 'go-', label='Reseq' )
            plt.axvline( self.explog.metrics['dac'] , ls='--' , color='green', label='2nd Vref' )
        
        plt.xlabel ( 'Reference Electrode Voltage (V)' )
        plt.ylabel ( 'Chip Gain (V/V)' )
        plt.title  ( 'Calibration Gain Curve' )
        plt.ylim   ( 0 , 1.2 )
        plt.grid   ( )
        plt.legend ( )
        plt.savefig( os.path.join( self.results_dir , 'gain_curve.png' ) )
        plt.close  ( )
        
    def write_html( self ):
        """ Creates html and block html files """
        html = textwrap.dedent('''\
        <html>
        <head><title>autoCal</title></head>
        <body>
        <style type="text/css">div.scroll {max-height:800px; overflow-y:auto; overflow-x:hidden; border: 0px solid black; padding: 5px 5px 0px 25px; float:left; }</style>
        <style type="text/css">div.plots {overflow-y:hidden; overflow-x:hidden; padding: 0px; }</style>
        <style type="text/css">tr.shade {background-color: #eee; }</style>
        <style type="text/css">td.top {vertical-align:top; }</style>
        <style>table.link {border: 1px solid black; border-collapse: collapse; padding: 0px; table-layout: fixed; text-align: center; vertical-align:middle; }</style>
        <style>th.link, td.link {border: 1px solid black; }</style>
        ''' )
        if self.ct.series == 'pgm':
            width  = 50
            images = [['instrument_temperature.png','instrument_pressure.png']]
        else:
            width  = 33
            images = [['chip_dac.png','dc_offset.png','flow_duration.png'],
                      ['instrument_temperature.png','instrument_cpu_temperature.png','instrument_fpga_temperature.png'],
                      ['instrument_pressure.png','pinch_regulators.png','gain_curve.png'],
                      ['valkyrie_flowrate.png','valkyrie_flowtemp.png','']]
            
        # Loop through creation of a table. . . Need to split 75/25 with a div for scrollbar.
        html += '<table cellspacing="0" width="100%%"><tr><td width="70%%">'
        html += '<div class="plots">'
        html += '<table cellspacing="0" width="100%%">'
        for i in images:
            html += '<tr>'
            for pic in i:
                if not os.path.exists( os.path.join( self.results_dir , pic ) ):
                    pic = ''
                if pic == '':
                    html += '<td width="%s%%">&nbsp</td>' % width
                else:
                    html += '<td width="%s%%">%s</td>' % ( width , image_link( pic ))
            html += '</tr>'
        html += '</table>\n</div>'
        if not self.explog.flowax.any():
            html += '<p><em>Error making plots!  Flow data was not found in the explog_final.txt file!</em></p>'
        html += '</td>'
        
        # Add in error section
        html += textwrap.dedent('''\
        <td class="top" width="30%%"><br><h3><b>Reported Errors:</b></h3>
        <div class="scroll">
        <p align="left">''' )
        
        # First do link loss and region slip parsing
        lines = []
        # Linkloss
        not_regions = ['link','summary_state','summary_fails','regions']
        if 'total' in self.metrics['linklosses']:
            regions = [ k for k in self.metrics['linklosses'].keys() if k not in not_regions ]
            regions.sort()
            lines += [ '<p>LINKLOSS</p>', 
                       '<table class="link">', 
                       '  <tr>', 
                       '    <th class="link">Location</th>', 
                       '    <th class="link"># Flows</th>', 
                       '    <th class="link">First Flow</th>', 
                       '  <tr>' ]
            for region in regions:
                lines += [ '  <tr>',
                           '    <th class="link">%s</th>' % region, 
                           '    <td class="link">%s</td>' % self.metrics['linklosses'][region]['instances'],
                           '    <td class="link">%s</td>' % self.metrics['linklosses'][region]['first'],
                           '  </tr>' ]
            lines.append( '</table>' )
            lines.append( '<br>'     )

        # Regionslip
        if 'total' in self.metrics['regionslips']:
            regions = [ k for k in self.metrics['regionslips'].keys() if k not in not_regions ]
            regions.sort()
            lines += [ '<p>REGIONSLIP</p>', 
                       '<table class="link">', 
                       '  <tr>', 
                       '    <th class="link">Location</th>', 
                       '    <th class="link"># Flows</th>', 
                       '    <th class="link">First Flow</th>', 
                       '  <tr>' ]
            for region in regions:
                lines += [ '  <tr>',
                           '    <th class="link">%s</th>' % region, 
                           '    <td class="link">%s</td>' % self.metrics['regionslips'][region]['instances'],
                           '    <td class="link">%s</td>' % self.metrics['regionslips'][region]['first'],
                           '  </tr>' ]
            lines.append( '</table>' )
            lines.append( '<br>'     )

        if lines != []:
            html += '<center>'
            html += '\n'.join( lines )
            html += '</center>'
            
        # Then handle other explog errors
        for error in self.explog.errors:
            html += error + '<br>'
            
        # Then add in tzero errors if file is found.
        if self.tzero_errors != []:
            html += '</p>'
            html += '<br><p>'
            for error in self.tzero_errors:
                html += error + '<br>'
            html += '</p><br>'
            
        # Add in additional "errors"
        rtn = self.explog.metrics['RowTemporalNoise']
        if rtn > 200:
            html += '<p>Warning!  Row Noise (%.1f) is over 200!</p>' % rtn
        elif rtn > 100:
            html += '<p>Warning!  Row Noise (%.1f) is over 100!</p>' % rtn
            
        amb1 = self.explog.metrics['Ambient1TemperatureMean']
        if amb1 > 35:
            html += '<p>Warning!  Ambient Temp 1 (%.1f C) is quite high, > 35 C!</p><br>' % amb1
        if amb1 > 30:
            html += '<p>Warning!  Ambient Temp 1 (%.1f C) is high, > 30 C!</p><br>' % amb1
            
        html += '</div></td></tr></table><br><hr>'
        
        # Create Info Section.
        shade = True
        labels= ['Chip Type','Datacollect Version','LiveView Version','Scripts Version','Platform',
                 'Main Flow Rate', 'Chip Flow Rate','Chip Noise Info','OverSample','Row Noise',
                 'Mean Ambient1 Temp.', 'Run Time (hours)']
        keys  = ['chiptype_name','DatacollectVersion','LiveViewVersion','ScriptsVersion','Platform','FlowRateMain',
                 'FlowRateChip','ChipNoiseInfo','OverSample','RowTemporalNoise','Ambient1TemperatureMean',
                 'run_time']
        
        html += '<h3><b>Info:</b></h3>'
        html += '<table width="50%%" cellspacing="0" border="0">'
        
        for (l,k) in zip(labels,keys):
            if k in self.explog.metrics:
                if shade:
                    tr    = '<tr class="shade">'
                else:
                    tr    = '<tr>'
                    
                shade = not shade
                if k == 'Ambient1TemperatureMean':
                    html += '''%s<td width="30%%">%s</td><td width="70%%">%.1f</td></tr>\n''' % (tr,l,self.explog.metrics[k])
                else:
                    html += '''%s<td width="30%%">%s</td><td width="70%%">%s</td></tr>\n''' % (tr,l,self.explog.metrics[k])

        # Valkyrie-specific
        if self.explog.metrics['Platform'].lower() == 'valkyrie':
            if 'run_time' in self.metrics:
                if shade:
                    tr    = '<tr class="shade">'
                else:
                    tr    = '<tr>'
                html += '''{}<td width="30%%">End-to-end?</td><td width="70%%">{}</td></tr>\n'''.format( tr, str(self.explog.metrics['run_time'] > 8) )
                
        if self.ct.series == 'pgm':
            if self.explog.AdvScriptFeatures != []:
                for adv in self.explog.AdvScriptFeatures:
                    if shade:
                        tr    = '<tr class="shade">'
                    else:
                        tr    = '<tr>'
                        
                    shade = not shade
                    html += '''%s<td width="30%%">AdvScriptFeature</td><td width="70%%">%s</td></tr>\n''' % (tr,adv)
                    
        html += '</table>'
        html += '</body></html>'

        # Save both standard and block html here.
        with open( os.path.join( self.results_dir , 'autoCal.html' ) , 'w' ) as f:
            f.write( html )
            
        with open( os.path.join( self.results_dir , 'autoCal_block.html' ) , 'w' ) as g:
            g.write( html )
        
    def output( self ):
        pass
    
    def report( self ):
        pass
    
    def metric( self ):
        pass
    
if __name__ == "__main__":
    PluginCLI()
