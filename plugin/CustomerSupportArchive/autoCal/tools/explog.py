import datetime, os, re

import numpy as np
import matplotlib
matplotlib.use('agg', warn=False)
import matplotlib.pyplot as plt
import scipy.stats as stats

from .core import explog as explog_lite
from links import Links


class Explog( explog_lite.Explog ):
    ''' Class to read and deal with explog files 
    This extends the raw parser to perform analytics and plots
    '''
    def parse( self ):
        super( Explog, self ).parse()
        self.get_flow_data()
        self.analyze_links( )
        
    def analyze_links( self ):
        # Define dictionary keys that are not actual chip regions (links)
        not_regions = ['link','summary_state','summary_fails', 'total', 'regions']
        
        # First, regionslips.
        rs = Links.from_nested_dict( self.regionslips, regionslips=True )
        rs.calc_metrics()
        self.regionslips.update( rs.metrics )
        
        if 'total' in self.regionslips:
            self.regionslips['regions'] = len( [x for x in self.regionslips if x not in not_regions ] )
        else:
            self.regionslips['regions'] = int( 0 )
            
        # Then linklosses.
        ll = Links.from_nested_dict( self.linklosses, regionslips=False )
        ll.calc_metrics()
        self.linklosses.update( ll.metrics )
        
        if 'total' in self.linklosses:
            self.linklosses['regions'] = len( [x for x in self.linklosses if x not in not_regions ] )
        else:
            self.linklosses['regions'] = int( 0 )
            
    def calc_flow_metrics( self , array , key_root ):
        if array.any():
            self.metrics['{}Mean'.format( key_root )] = array.mean()
            self.metrics['{}SD'.format(   key_root )] = array.std()
            self.metrics['{}90'.format(   key_root )] = float( stats.scoreatpercentile( array , 90 ) )
        else:
            self.metrics['{}Mean'.format( key_root )] = 0.0
            self.metrics['{}SD'.format(   key_root )] = 0.0
            self.metrics['{}90'.format(   key_root )] = 0.0
        return None
        
    def get_flow_data( self ):
        ''' 
        Gets data of pressure, dss, etc. on a per-flow basis 
        This works for PGM and Proton using self.chiptype.series triggering.
        '''
        if self.chiptype.series.lower() in ['pgm']:
            # PGM explog scraping
            # Note that the match string is different for 1.0 and 1.1 PGM Hardware...more on 1.1
            if float( self.metrics['PGMHW'] ) == 1.0:
                match_string = r'(?P<flow>[0-9]+)(?P<na0>\)) (?P<pressure>[0-9\.]+) (?P<tempinst>[0-9\.]+) (?P<tempchip>[0-9\.]+) (?P<na1>\(0x[0-9A-Fa-f]+) (?P<na2>0x[0-9A-Fa-f]+) (?P<na3>0x[0-9A-Fa-f]+) (?P<na4>0x[0-9A-Fa-f]+\))'
            elif float( self.metrics['PGMHW'] ) == 1.1:
                match_string = r'(?P<flow>[0-9]+)(?P<na0>\)) (?P<pressure>[0-9\.]+) (?P<tempinst>[0-9\.]+) (?P<temprest>[0-9\.]+) (?P<tempsink>[0-9\.]+) (?P<tempchip>[0-9\.]+) (?P<na1>\(0x[0-9A-Fa-f]+) (?P<na2>0x[0-9A-Fa-f]+) (?P<na3>0x[0-9A-Fa-f]+) (?P<na4>0x[0-9A-Fa-f]+\))'
                
            flow     = []
            pressure = []
            tempinst = []
            temprest = [] # (Flow) Restrictor T, PGM 1.1 only.
            tempsink = [] # Chip Heat Sink, PGM 1.1 only.
            tempchip = []
            
            for line in self.lines:
                m = re.match( match_string , line.strip() )
                if (m != None):
                    m = m.groupdict()
                    
                    flow.append( int( m['flow'] ) )
                    pressure.append( float( m['pressure'] ) )
                    tempinst.append( float( m['tempinst'] ) )
                    tempchip.append( float( m['tempchip'] ) )
                    if float( self.metrics['PGMHW'] ) == 1.1:
                        temprest.append( float( m['temprest'] ) )
                        tempsink.append( float( m['tempsink'] ) )
                    
            self.flowax   = np.array( flow , np.int16 )[::-1]
            self.pressure = np.array( pressure , float )[::-1]
            self.tempinst = np.array( tempinst , float )[::-1]
            self.temprest = np.array( temprest , float )[::-1]
            self.tempsink = np.array( tempsink , float )[::-1]
            self.tempchip = np.array( tempchip , float )[::-1]
            
            # Save metrics
            self.calc_flow_metrics( self.pressure , 'Pressure' )
            self.calc_flow_metrics( self.tempinst , 'InstrumentTemperature' )
            self.calc_flow_metrics( self.temprest , 'RestrictorTemperature' )
            self.calc_flow_metrics( self.tempsink , 'HeatsinkTemperature' )
            self.calc_flow_metrics( self.tempchip , 'ChipTemperature' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            # This is getting out of hand.
            valk_match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pc1>[0-9\.\-]+) (?P<pc2>[0-9\.\-]+) (?P<pc3>[0-9\.\-]+) (?P<pc4>[0-9\.\-]+) (?P<pm1>[0-9\.\-]+) (?P<pm2>[0-9\.\-]+) (?P<pm3>[0-9\.\-]+) (?P<pm4>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)(?P<na999>.+Vref=)(?P<vref>[0-9\.\-]+)'
            valk_string_2 = r'(?P<flow>.+?): Pressure=(?P<p0>[\d.-]+) (?P<p1>[\d.-]+) Temp=(?P<t0>[\d.-]+) (?P<t1>[\d.-]+) (?P<t2>[\d.-]+) (?P<t3>[\d.-]+) dac_start_sig=(?P<dac>[\d.-]+) avg=(?P<dc>[\d.-]+) time=(?P<t>[\d:]+) fpgaTemp=(?P<t4>[\d.-]+) (?P<t5>[\d.-]+) chipTemp=(?P<t6>[\d.-]+) (?P<t7>[\d.-]+) (?P<t8>[\d.-]+) (?P<t9>[\d.-]+) (?P<t10>[\d.-]+)[\w\s=]+cpuTemp=(?P<t11>[\d.-]+) (?P<t12>[\d.-]+) heater=[\d.-]* cooler=[\d.-]* gpuTemp=(?P<t13>[\d.-]+) diskPerFree=(?P<dpf>[\d.-]*) FACC_Offset=(?P<faccOff>[\d.-]*).+FACC=(?P<facc>[\d.-]*).+Pinch=(?P<pc1>[\d.-]*) (?P<pc2>[\d.-]*) (?P<pc3>[\d.-]*) (?P<pc4>[\d.-]*) (?P<pm1>[\d.-]*) (?P<pm2>[\d.-]*) (?P<pm3>[\d.-]*) (?P<pm4>[\d.-]*).+FR=(?P<flowrate>[\d.-]*).+FTemp=(?P<flowtemp>[\d.-]*).+Vref=(?P<vref>[\d.-]*)'
            if self.isRaptor:
                # S5/S5XL
                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+)(?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)(?P<na9>.+diskPerFree=)(?P<dpf>[0-9\.\-]+)(?P<na10>.+FACC_Offset=)(?P<faccOff>[0-9\.\-]+)(?P<na11>.+FACC=)(?P<facc>[0-9\.\-]+)(?P<na12>.+Pinch=)(?P<pinch1>[0-9\.\-]+) (?P<pinch2>[0-9\.\-]+)(?P<na13>.+ManTemp=)(?P<manT>[0-9\.\-]+)'
            else:
                # Proton
                match_string = r'(?P<flow>.+?:)(?P<na0> Pressure=)(?P<p0>[0-9\.]+) (?P<p1>[0-9\.]+) (?P<na1>Temp=)(?P<t0>[0-9\.\-]+) (?P<t1>[0-9\.\-]+) (?P<t2>[0-9\.\-]+) (?P<t3>[0-9\.\-]+) (?P<na2>dac_start_sig=)(?P<dac>[0-9\.\-]+) (?P<na3>avg=)(?P<dc>[0-9\.\-]+) (?P<na4>time=)(?P<t>[0-9:]+) (?P<na5>fpgaTemp=)(?P<t4>[0-9\.\-]+) (?P<t5>[0-9\.\-]+) (?P<na6>chipTemp=)(?P<t6>[0-9\.\-]+) (?P<t7>[0-9\.\-]+) (?P<t8>[0-9\.\-]+) (?P<t9>[0-9\.\-]+) (?P<t10>[0-9\.\-]+) (?P<na7>.+cpuTemp=)(?P<t11>[0-9\.\-]+) (?P<t12>[0-9\.\-]+) (?P<na8>.+gpuTemp=)(?P<t13>[0-9\.\-]+)'
                
                
            isfirst = True
            p_reg   = [] # Regulator pressure
            p_mani  = [] # manifold pressure
            bayT    = [] # chip bay temperature
            ambT1   = [] # ambient temperature 1 (not to be used on Proton.)
            coolT   = [] # cooler temperature
            ambT2   = [] # ambient temperature 2 ( The real ambient temperature on Proton, S5.
            manH    = [] # RAPTOR: Manifold heater (t0)
            tec     = [] # RAPTOR: TEC (t1)
            wasteT  = [] # RAPTOR: waste line temp (t2)
            dss     = [] # dac_start_sig
            dc      = []
            time    = [] # time
            fpgaT1  = [] # fpga temperature 1
            fpgaT2  = [] # fpga temperature 2
            meanT   = []
            temp1   = []
            temp2   = []
            temp3   = []
            temp4   = []
            cpuT1   = [] # cpu temperature 1
            cpuT2   = [] # cpu temperature 2
            gpuT    = [] # gpu temperature
            cwa     = [] # chip waste pinch regulator pressure (S5 only)
            mwa     = [] # main waste pinch regulator pressure (S5 only)
            manT    = [] # manifold temperature (S5 only)
            vref    = [] # Only on recent datacollects, where vref is allowed to move with drift.

            # Valkyrie-specific items.  These chip/main pinch regulators go with each of the four lanes.
            pc1   = []
            pc2   = []
            pc3   = []
            pc4   = []
            pm1   = []
            pm2   = []
            pm3   = []
            pm4   = []
            flowrate = []
            flowtemp = []
            
            # February 2018 - Important note!
            # MarkB has added a vref element to the explog file.  to support versions w/o the vref, we will add
            #   the regex expression only if we find 'Vref' in the line.
            # Adding 'Vref' to the above match strings will fail if previous software (DC) is run and
            #   no vref is found in the lines.
            for line in self.lines:
                # April 2019 - switching order to look for latest valkyrie string first to get more data.
                m = re.match( valk_string_2, line.strip() )
                if (m != None):
                    self.isValkyrie = True
                    self.isRaptor   = False
                elif re.match( valk_match_string , line.strip() ):
                    m = re.match( valk_match_string , line.strip() )
                    self.isValkyrie = True
                    self.isRaptor   = False
                else:
                    m = re.match( match_string , line.strip() )
                    if (m != None):
                        # Check here for vref in line, so that we ignore other lines that reference 'Vref'
                        # This should only occur once, then next loop iteration, Vref will be in match_string.
                        if 'Vref' in line and 'Vref' not in match_string:
                            match_string += r'(?P<na999>.+Vref=)(?P<vref>[0-9\.\-]+)'
                            m = re.match( match_string , line.strip() )
                            
                if (m != None):
                    m = m.groupdict()
                    
                    # Deal with time
                    d = datetime.datetime.strptime( m['t'] , '%H:%M:%S' )
                    if isfirst:
                        isfirst = False
                        d0      = d
                        
                    if d < d0:
                        d += datetime.timedelta(days=1)
                        
                    tdelta     = d - d0
                    delta_time = tdelta.seconds
                    
                    # Apply values
                    p_reg.append ( float( m['p0' ] ) )
                    p_mani.append( float( m['p1' ] ) )
                    dss.append   (   int( m['dac'] ) )
                    dc.append    (   int( m['dc']  ) )
                    time.append  (   int(delta_time) )
                    fpgaT1.append( 5.0/9.0 * (float( m['t4'] ) - 32.0 ) )
                    fpgaT2.append( 5.0/9.0 * (float( m['t5'] ) - 32.0 ) )
                    meanT.append ( float( m['t6' ] ) )
                    temp1.append ( float( m['t7' ] ) )
                    temp2.append ( float( m['t8' ] ) )
                    temp3.append ( float( m['t9' ] ) )
                    temp4.append ( float( m['t10'] ) )
                    cpuT1.append ( float( m['t11'] ) )
                    cpuT2.append ( float( m['t12'] ) )
                    gpuT.append  (   int( m['t13'] ) )
                    
                    # The 4 temperatures, t0, t1, t2, and t3 are different on Proton and S5
                    if self.isRaptor:
                        manH.append  ( float( m['t0' ] ) ) # Manifold Heater
                        wasteT.append( float( m['t1' ] ) ) # Waste
                        tec.append   ( float( m['t2' ] ) ) # TEC 
                        ambT2.append ( float( m['t3' ] ) ) # Ambient
                    elif self.isValkyrie:
                        manH.append  ( float( m['t0' ] ) ) # Manifold Heater
                        wasteT.append( float( m['t2' ] ) ) # Waste
                        tec.append   ( float( m['t1' ] ) ) # TEC 
                        ambT2.append ( float( m['t3' ] ) ) # Ambient
                    else:
                        # This is how it used to be -- correct for proton.
                        bayT.append  ( float( m['t0' ] ) ) 
                        ambT1.append ( float( m['t1' ] ) ) # Poorly attached to waste lines.  Sort of ambient
                        coolT.append ( float( m['t2' ] ) ) # No longer used
                        ambT2.append ( float( m['t3' ] ) ) # This is the important ambient temperature on proton
                        
                    if self.isRaptor:
                        cwa.append ( float( m['pinch1'] ) )
                        mwa.append ( float( m['pinch2'] ) )
                        manT.append(   int( m['manT']   ) )
                    elif self.isValkyrie:
                        pc1.append( float( m['pc1'] ) )
                        pc2.append( float( m['pc2'] ) )
                        pc3.append( float( m['pc3'] ) )
                        pc4.append( float( m['pc4'] ) )
                        pm1.append( float( m['pm1'] ) )
                        pm2.append( float( m['pm2'] ) )
                        pm3.append( float( m['pm3'] ) )
                        pm4.append( float( m['pm4'] ) )

                        # Not all valkyrie runs have this befor a certain datacollect.
                        try:
                            flowrate.append( float( m['flowrate'] ) )
                            flowtemp.append( float( m['flowtemp'] ) )
                        except KeyError:
                            pass
                            
                    if 'vref' in m.keys():
                        vref.append( float( m['vref']   ) )
                    
            self.p_reg = np.array( p_reg , float )
            self.p_mani= np.array( p_mani, float )
            self.bayT  = np.array( bayT  , float )
            self.ambT1 = np.array( ambT1 , float )
            self.coolT = np.array( coolT , float )
            self.ambT2 = np.array( ambT2 , float )
            self.ambT  = self.ambT2
            self.manH  = np.array( manH  , float )
            self.tec   = np.array( tec   , float )
            self.wasteT= np.array( wasteT, float )
            self.dss   = np.array( dss   , np.int16 )
            self.dc    = np.array( dc    , np.int16 )
            self.time  = np.array( time  , np.int16 )
            self.fpgaT1= np.array( fpgaT1, float )
            self.fpgaT2= np.array( fpgaT2, float )
            self.meanT = np.array( meanT , float )
            self.temp1 = np.array( temp1 , float )
            self.temp2 = np.array( temp2 , float )
            self.temp3 = np.array( temp3 , float )
            self.temp4 = np.array( temp4 , float )
            self.cpuT1 = np.array( cpuT1 , float )
            self.cpuT2 = np.array( cpuT2 , float )
            self.gpuT  = np.array( gpuT  , np.int16 )
            self.cwa   = np.array( cwa   , float )
            self.mwa   = np.array( mwa   , float )
            self.manT  = np.array( manT  , np.int16 )
            self.vref  = np.array( vref  , float )
            
            self.pc1   = np.array( pc1 , float )
            self.pc2   = np.array( pc2 , float )
            self.pc3   = np.array( pc3 , float )
            self.pc4   = np.array( pc4 , float )
            self.pm1   = np.array( pm1 , float )
            self.pm2   = np.array( pm2 , float )
            self.pm3   = np.array( pm3 , float )
            self.pm4   = np.array( pm4 , float )
            self.flowrate = np.array( flowrate , float )
            self.flowtemp = np.array( flowtemp , float )
            
            # Define x-axis based on flow number, with prerun flows being negative...an alternative to time axis.
            allflows = len(self.dss)
            self.flowax = np.arange( allflows ) - ( allflows-self.flows )
            
            # Instrument pressure metrics
            self.calc_flow_metrics( self.p_reg , 'RegulatorPressure'  )
            self.calc_flow_metrics( self.p_mani, 'ManifoldPressure'   )
            self.calc_flow_metrics( self.bayT  , 'ChipBayTemperature' )
            self.calc_flow_metrics( self.ambT1 , 'Ambient1Temperature')
            self.calc_flow_metrics( self.coolT , 'CoolerTemperature'  )
            
            # Duplicate ambient2 metrics as straight up ambient
            self.calc_flow_metrics( self.ambT2 , 'Ambient2Temperature')
            self.calc_flow_metrics( self.ambT2 , 'AmbientTemperature' )
            
            self.calc_flow_metrics( self.tec   , 'TECTemperature'     )
            self.calc_flow_metrics( self.manH  , 'ManifoldHeater'     )
            self.calc_flow_metrics( self.wasteT, 'WasteTemperature'   )
            self.calc_flow_metrics( self.dss   , 'ChipDAC'            )
            self.calc_flow_metrics( self.fpgaT1, 'FPGA1Temperature'   )
            self.calc_flow_metrics( self.fpgaT2, 'FPGA2Temperature'   )
            self.calc_flow_metrics( self.temp1 , 'ChipThermometer1'   )
            self.calc_flow_metrics( self.temp2 , 'ChipThermometer2'   )
            self.calc_flow_metrics( self.temp3 , 'ChipThermometer3'   )
            self.calc_flow_metrics( self.temp4 , 'ChipThermometer4'   )
            self.calc_flow_metrics( self.meanT , 'ChipThermometerAverage' )
            self.calc_flow_metrics( self.cpuT1 , 'CPU1Temperature'    )
            self.calc_flow_metrics( self.cpuT2 , 'CPU2Temperature'    )
            self.calc_flow_metrics( self.gpuT  , 'GPUTemperature'     )
            
            self.calc_flow_metrics( self.flowrate, 'ValkyrieFlowRate' )
            self.calc_flow_metrics( self.flowtemp, 'ValkyrieFlowTemp' )
            
        return None
            
    def dss_plot( self , outdir ):
        ''' plots DSS with prerun flows as "negative" flows '''
        if self.chiptype.series.lower() in ['pgm']:
            print ( 'Skipping! This plot (dss_plot) cannot be created for PGM.' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            # Decide on if Vref is recorded -- if so, make a slightly different plot.
            if self.vref.any():
                fig, ax1 = plt.subplots()
                ax1.plot       ( self.flowax , self.dss , 'b-' )
                ax1.set_ylabel ( 'dac_start_sig' , color='b' )
                ax1.tick_params( 'y' , colors='b' )
                ax1.set_title  ( 'dac_start_sig' )
                ax1.set_xlabel ( 'Flows (negative values are beadfind and prerun flows)')
                ax1.set_xlim   ( self.flowax[0]-5 , self.flowax[-1] )
                
                # Define vref axis as second y-axis
                ax2 = ax1.twinx()
                ax2.plot       ( self.flowax , self.vref , 'g-' )
                ax2.set_ylabel ( 'Reference Electrode (V)' , color='g' )
                ax2.tick_params( 'y' , colors='g' )
                ax2.set_xlim   ( self.flowax[0]-5 , self.flowax[-1] )
                
                ax1.grid       ( ) # This only works if default ticks are at same positions along axes.
                
                fig.tight_layout( )
                plt.savefig( os.path.join( outdir , 'chip_dac.png' ) )
                plt.close  ( )
            else:
                plt.figure ()
                plt.plot   ( self.flowax , self.dss , '-' )
                plt.ylabel ( 'dac_start_sig' )
                
                # Could use full range of DSS but normally isn't needed. Let's have a dynamic range for now.
                # plt.ylim   ( 0 , 4096 )
                ymin,ymax  = plt.ylim()
                plt.ylim   ( 0.9*ymin , 1.1*ymax )
                plt.title  ( 'dac_start_sig' )
                plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
                plt.xlim   ( self.flowax[0]-5 , self.flowax[-1] )
                # plt.axvline( x=0 , ls='-' , color='black' )
                plt.grid   ( )
                # plt.savefig( os.path.join( outdir , 'dss_plot.png' ) )
                plt.savefig( os.path.join( outdir , 'chip_dac.png' ) )
                plt.close  ( )
                
            return None

    def dc_offset_plot( self , outdir ):
        ''' Plots mean DC offset traces vs. flow '''
        if self.chiptype.series.lower() in ['pgm']:
            print ( 'Skipping! This plot (dc_offset) cannot be created for PGM.' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            plt.figure ( )
            plt.plot   ( self.flowax , self.dc , label='DC Offset' )
            plt.xlim   ( self.flowax[0]-5 , self.flowax[-1] )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'DC Offset (DN14 Counts)' )
            plt.grid   ( )
            plt.title  ( 'DC Offset' )
            ymin,ymax  = plt.ylim()
            plt.ylim   ( 0.9*ymin , 1.1*ymax )
            plt.legend ( )
            plt.savefig( os.path.join( outdir , 'dc_offset.png' ) )
            plt.close  ( )

    def cumulative_offset_plot( self , outdir ):
        ''' 
        Plots cumulative offset starting from acq0.
        NB: I have no idea what this actually means.  I adapted from autoCal plugin code, which 
            seems to make some funky assumptions about DSS. - Phil
        
        MarkB thinks it's bogus too.  Won't make it by default anymore.
        '''
        if self.chiptype.series.lower() in ['pgm']:
            print ( 'Skipping! This plot (cumulative_offset) cannot be created for PGM.' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            # Do the wacky massaging of numbers
            start = np.where( self.flowax == 0 )[0][0]
            dc    = self.dc[start:]
            dss   = self.dss[start:]
            ddc   = np.diff( dc )
            idx   = np.where( np.diff( dss ) != 0 )[0]
            if (idx[-1]+1) == len( ddc ):
                idx = idx[:-1]
            
            ddc[idx+1] = ddc[idx]
            dc_new     = dc[0] + np.cumsum( ddc )
            dc_range   = dc_new.ptp()
            self.metrics['dc_range'] = dc_range
            
            plt.figure ( )
            plt.plot   ( self.flowax[start:-1] , dc_new )
            plt.xlim   ( -5 , self.flowax[-1]+5 )
            plt.xlabel ( 'Flow' )
            plt.ylabel ( 'Cumulative DC Offset (DN14 Counts)' )
            plt.grid   ( )
            #ymin,ymax  = plt.ylim()
            #plt.ylim   ( 0.9*ymin , 1.1*ymax )
            plt.title  ( 'Cumulative DC Offset | DC Range: %d' % dc_range )
            plt.savefig( os.path.join( outdir , 'dc_range.png' ) )
            plt.close  ( )
            
    def flow_duration_plot( self , outdir ):
        ''' Plots mean DC offset traces vs. flow '''
        if self.chiptype.series.lower() in ['pgm']:
            print ( 'Skipping! This plot (flow_duration) cannot be created for PGM.' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            plt.figure ( )
            plt.plot   ( self.flowax[:-1] , np.diff( self.time ) , label='Flow Duration' )
            plt.xlim   ( self.flowax[0]-5 , self.flowax[-1] )
            
            # 21 Feb 2019
            # Pre-run flows do not matter.  Let's fix the y scale to zoom in on the area of interest.
            plt.ylim   ( 0 , 45 )
            
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'Time (s)' )
            plt.grid   ( )
            plt.title  ( 'Flow Duration' )
            ymin,ymax  = plt.ylim()
            plt.ylim   ( 0.9*ymin , 1.1*ymax )
            plt.legend ( )
            plt.savefig( os.path.join( outdir , 'flow_duration.png' ) )
            plt.close  ( )

    def gain_curve( self , outdir ):
        ''' 
        Plots gain curve from calibration if it was written in explog/explog_final.
        Requires Datacollect >= 3544
        '''
        if (self.metrics['GainCurveVref'] != []) and (self.metrics['GainCurveGain'] != []):
            # We have the data and can make a plot!
            vref   = np.array( [ x for x in self.metrics['GainCurveVref'] if x != 0. ] )
            gain   = np.array( [ x for x in self.metrics['GainCurveGain'] if x != 0. ] )
            
            # The following code added for Valkyrie alphas.  Not a bad idea really.
            if (not vref.any()) and (not gain.any()):
                print( 'Gain curve information was empty or all zeroes!  Skipping gain plot.' )
                return None
                
            max_ix = np.where( gain == gain.max() )
            
            plt.figure ( )
            plt.plot   ( vref , gain , 'o-' )
            plt.xlabel ( 'Reference Electrode Voltage (V)' )
            plt.ylabel ( 'Chip Gain (V/V)' )
            plt.title  ( 'Calibration Gain Curve' )
            plt.ylim   ( 0 , 1.2 )
            xlims = plt.xlim()
            mid   = np.array( xlims ).mean()
            #plt.text   ( vref[max_ix] , gain[max_ix]+0.05 , 'Gain = %1.2f at VREF = %1.2f' % (gain[max_ix] , vref[max_ix] ) , ha='left' , fontsize=12 )
            plt.text   ( mid , 0.1 , 'Max Gain = %1.2f at VREF = %1.2f V' % (gain[max_ix] , vref[max_ix] ) , ha='center' , fontsize=12 , weight='semibold' , bbox={'facecolor':'blue', 'alpha':0.25, 'pad':10} )
            plt.axvline( self.metrics['dac'] , ymin=0.167 , ls='--' , color='red' )
            plt.text   ( self.metrics['dac'] + 0.025 , 0.21 , 'Selected DAC=%1.2f V' % self.metrics['dac'] ,
                         fontsize=10 , weight='semibold' , color='red' , rotation=90 , va='bottom' )
            plt.grid   ( )
            plt.savefig( os.path.join( outdir , 'gain_curve.png' ) )
            plt.close  ( )
            
    def pinch_plot( self , outdir ):
        ''' Plots pinch regulator pressures '''
        if self.isRaptor:
            plt.figure ( )
            plt.plot   ( self.flowax , self.mwa , label='Main waste' )
            plt.plot   ( self.flowax , self.cwa , label='Chip waste' )
            plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'Pressure (psi)' )
            plt.grid   ( )
            plt.title  ( 'Pinch Regulator Pressure' )
            ymin,ymax  = plt.ylim()
            plt.ylim   ( 0.9*ymin , 1.1*ymax )
            plt.legend ( )
            plt.savefig( os.path.join( outdir , 'pinch_regulators.png' ) )
            plt.close  ( )
        elif self.isValkyrie:
            plt.figure ( )
            for lane, c in zip( [1,2,3,4], ['red','orange','green','blue'] ):
                plt.plot( self.flowax , getattr( self, 'pm{}'.format(lane), [] ) , ls='-'  , color=c , label='L{} Main'.format(lane) )
                plt.plot( self.flowax , getattr( self, 'pc{}'.format(lane), [] ) , ls='--' , color=c , label='L{} Chip'.format(lane) )
            plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'Pressure (psi)' )
            plt.grid   ( )
            plt.title  ( 'Pinch Regulator Pressure' )
            ymin,ymax  = plt.ylim()
            plt.ylim   ( 0.9*ymin , 1.1*ymax )
            plt.legend ( ncol=4 , fontsize=10 , loc='upper center' )
            plt.savefig( os.path.join( outdir , 'pinch_regulators.png' ) )
            plt.close  ( )
        else:
            print ( 'Skipping! This plot (pinch_plot) cannot be created for PGM or Proton' )
            
    def pressure_plot( self , outdir ):
        ''' Plots instrument pressures '''
        plt.figure ()
        
        if self.chiptype.series.lower() in ['pgm']:
            plt.plot   ( self.flowax , self.pressure , label='Pressure' ) 
            plt.xlabel ( 'Flow' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            plt.plot   ( self.flowax , self.p_reg  , label='Regulator' )
            plt.plot   ( self.flowax , self.p_mani , label='Manifold'  )
            plt.xlim   ( self.flowax[0]-5 , self.flowax[-1] )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            
        plt.grid   ( True )
        plt.title  ( 'Instrument Pressure' )
        plt.ylabel ( 'Pressure (psi)' )
        ymin,ymax  = plt.ylim()
        plt.ylim   ( 0 , 1.5*ymax )
        plt.legend ( )
        plt.savefig( os.path.join( outdir , 'instrument_pressure.png' ) )
        plt.close  ( )
        
        return None
        
    def chip_temp_plot( self , outdir ):
        ''' Creates chip temperature plot '''
        if self.chiptype.series.lower() in ['pgm']:
            print ( 'Skipping! This plot (chip_temp) cannot be created for PGM.' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            plt.figure ( )
            plt.plot   ( self.flowax , self.temp1 , label='Thermometer 1' )
            plt.plot   ( self.flowax , self.temp2 , label='Thermometer 2' )
            plt.plot   ( self.flowax , self.temp3 , label='Thermometer 3' )
            plt.plot   ( self.flowax , self.temp4 , label='Thermometer 4' )
            plt.ylabel ( 'Temperature (degC)' )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.grid   ( True )
            plt.title  ( 'Chip Temperature' )
            # plt.axvline( x=0 )
            plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            ymin,ymax  = plt.ylim()
            plt.ylim   ( 0.7*ymin , 1.3*ymax )
            plt.legend ( loc=0 )
            plt.savefig( os.path.join( outdir , 'chip_thermometer_temperature.png' ) )
            plt.close  ( )
            
        return None
        
    def cpu_temp_plot( self , outdir ):
        ''' Plots CPU temperature plot '''
        if self.chiptype.series.lower() in ['pgm']:
            print ( 'Skipping! This plot (cpu_temp) cannot be created for PGM.' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            plt.figure ()
            plt.plot   ( self.flowax , self.cpuT1 , label='CPU 1' )
            plt.plot   ( self.flowax , self.cpuT2 , label='CPU 2' )
            plt.plot   ( self.flowax , self.gpuT  , label='GPU'   )
            plt.grid   ( True )
            plt.title  ( 'CPU Temperature' )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'Temperature (degC)' )
            plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            ymin, ymax = plt.ylim()
            plt.ylim   ( 0.8*ymin , 1.2*ymax )
            plt.legend ( loc=0 )
            plt.savefig( os.path.join( outdir , 'instrument_cpu_temperature.png' ) )
            plt.close  ( )
        
        return None
        
    def fpga_temp_plot( self , outdir ):
        ''' Plots FPGA temperature plot '''
        if self.chiptype.series.lower() in ['pgm']:
            print ( 'Skipping! This plot (fpga_temp) cannot be created for PGM.' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            plt.figure ()
            plt.plot   ( self.flowax , self.fpgaT1 , label='FPGA 1' )
            plt.plot   ( self.flowax , self.fpgaT2 , label='FPGA 2' )
            plt.grid   ( True )
            plt.title  ( 'FPGA Temperature' )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'Temperature (degC)' )
            plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            ymin, ymax = plt.ylim()
            plt.ylim   ( 0.8*ymin , 1.2*ymax )
            plt.legend ( loc=0 )
            plt.savefig( os.path.join( outdir , 'instrument_fpga_temperature.png' ) )
            plt.close  ( )
            
        return None
        
    def inst_temp_plot( self , outdir ):
        ''' Plots instrument temperature plot '''
        plt.figure ()
        
        if self.chiptype.series.lower() in ['pgm']:
            if float( self.metrics['PGMHW'] ) == 1.0:
                plt.plot( self.flowax , self.tempinst , label='Instrument'    )
                plt.plot( self.flowax , self.tempchip , label='Chip'          )
            elif float( self.metrics['PGMHW'] ) == 1.1:
                plt.plot( self.flowax , self.tempinst , label='Instrument'    )
                plt.plot( self.flowax , self.temprest , label='Restrictor'    )
                plt.plot( self.flowax , self.tempsink , label='Chip Heatsink' )
                plt.plot( self.flowax , self.tempchip , label='Chip'          )
                
            plt.xlabel  ( 'Flow' )
            
        elif self.chiptype.series.lower() in ['proton','s5','s5xl']:
            if self.isRaptor:
                plt.plot   ( self.flowax , self.manH  , label='Manifold Heater'  ) # t0
                plt.plot   ( self.flowax , self.wasteT, label='Waste' ) # t1
                plt.plot   ( self.flowax , self.tec   , label='TEC'    ) # t2
                plt.plot   ( self.flowax , self.ambT2 , label='Ambient' ) # t3
                plt.plot   ( self.flowax , self.manT  , label='Manifold' ) # manT
            elif self.isValkyrie:
                plt.plot   ( self.flowax , self.manH  , label='Manifold Heater'  ) # t0
                plt.plot   ( self.flowax , self.wasteT, label='Waste' ) # t1
                plt.plot   ( self.flowax , self.tec   , label='TEC'    ) # t2
                plt.plot   ( self.flowax , self.ambT2 , label='Ambient' ) # t3
            else:
                plt.plot   ( self.flowax , self.bayT  , label='Chip Bay'  )
                #plt.plot   ( self.flowax , self.coolT , label='Under Chip') - This is unused now.
                # plt.plot   ( self.flowax , self.ambT1 , label='Ambient 1' ) - this is unreliable
                plt.plot   ( self.flowax , self.ambT2 , label='Ambient' )
                plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            
        plt.xlim   ( self.flowax[0] , self.flowax[-1] )
        plt.grid   ( True )
        plt.title  ( 'Instrument Temperature' )
        plt.ylabel ( 'Temperature (degC)' )
        ymin, ymax = plt.ylim()
        plt.ylim   ( 0.8*ymin , 1.2*ymax )
        plt.legend ( loc=0 )
        plt.savefig( os.path.join( outdir , 'instrument_temperature.png' ) )
        plt.close  ( )
        
        return None

    def flowrate_plot( self, outdir ):
        """ 
        Plots the flowrate for the run -- only for the Valkyrie platform.
        The nominal rate is 48 * the number of lanes run.
        """
        if self.isValkyrie and any(self.flowrate):
            plt.figure ( )
            plt.plot( self.flowax , self.flowrate , ls='-' , color='blue' )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'Flow Rate (uL/s)' )
            plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            plt.ylim   ( 0, 210 )
            
            i = 1
            for y in range(48,200,48):
                plt.axhline( y, ls='--', color='green' )
                plt.text   ( self.flowax[-1]-5, y+5, '{} lane target'.format( i ), fontsize=8,
                             va='center', ha='right', color='green' )
                i += 1

            plt.yticks ( range(0, 200, 48 ) )
            plt.grid   ( axis='x' )
            plt.title  ( 'Valkyrie Flow Rate' )
            plt.savefig( os.path.join( outdir , 'valkyrie_flowrate.png' ) )
            plt.close  ( )
        else:
            print ( 'Skipping! Flowrate data does not exist for this run.' )
        
    def flowtemp_plot( self, outdir ):
        """ 
        Plots the flow temperature for the run -- only for the Valkyrie platform.
        """
        if self.isValkyrie and any(self.flowrate):
            plt.figure ( )
            plt.plot( self.flowax , self.flowtemp , ls='-' , color='red' )
            plt.xlabel ( 'Flows (negative values are beadfind and prerun flows)')
            plt.ylabel ( 'Flow Temperature (C)' )
            plt.xlim   ( self.flowax[0] , self.flowax[-1] )
            plt.grid   ( )
            plt.title  ( 'Valkyrie Flow Temperature' )
            plt.savefig( os.path.join( outdir , 'valkyrie_flowtemp.png' ) )
            plt.close  ( )
        else:
            print ( 'Skipping! Flow temperature data does not exist for this run.' )
        
    def make_all_plots( self , outdir ):
        self.dss_plot              ( outdir )
        self.dc_offset_plot        ( outdir )
        self.flow_duration_plot    ( outdir )
        self.pinch_plot            ( outdir )
        self.pressure_plot         ( outdir )
        self.chip_temp_plot        ( outdir )
        self.cpu_temp_plot         ( outdir )
        self.fpga_temp_plot        ( outdir )
        self.inst_temp_plot        ( outdir )
        self.gain_curve            ( outdir )
        self.flowrate_plot         ( outdir )
        self.flowtemp_plot         ( outdir )
        # Removing this plot because it's probably bogus.  At the very least it is misleading.
        #self.cumulative_offset_plot( outdir )
    

