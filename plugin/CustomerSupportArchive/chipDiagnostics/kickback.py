import matplotlib
matplotlib.use( 'agg', warn=False )
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ion.plugin import *
from tools.chiptype import ChipType
from tools.datfile import DatFile
from tools.stats import named_stats
from tools.chip import load_mask
from tools.explog import Explog
import os, json, sys
import numpy as np
import textwrap

# Set default cmap to jet, which is viridis on newer TS software (matplotlib 2.1)
matplotlib.rcParams['image.cmap'] = 'jet'

moduleDir = os.path.abspath( os.path.dirname( __file__ ) )
#TODO:
## Need to calculate the average row potential without zeroing out the signal
# Need to report kickback heights at 3 key locations (how to do on Valkyrie?)
## Need to report Peak-to-peak heights on average row potential plots 
## What about S5 where signals are reversed?

class KickbackAnalyzer( object ):
    ''' class for analyzing kickback properties on a single acquisition (spatial thumbnail)'''
    # These are defined on the half chip
    FC_REGIONS = { 'inlet':   ( slice( 300, 400 ), slice( 100, 200   ) ), 
                   'outlet':  ( slice( 300, 400 ), slice( 1000, 1100 ) ), 
                   'outedge': ( slice(  50, 100 ), slice( 1000, 1100 ) ), 
                 }

    LANE_REGIONS = { 'inlet':   ( slice( 650, 750 ), slice( 100, 150 ) ), 
                     'outlet':  ( slice(  50, 100 ), slice( 100, 150 ) ), 
                     'outedge': ( slice( 125, 200 ), slice(  25,  75 ) ), 
                   }
    LANES = { 1: slice(   0,  300 ), 
              2: slice( 300,  600 ),
              3: slice( 600,  900 ), 
              4: slice( 900, 1200 ), 
            }

    def __init__( self, filename, dr, chiptype, outdir='.', lanes=None ):
        ''' filename - path to spatial thumbnail dat file
            dr       - effective (gain corrected) dynamic range, in V
            chiptype - ChipType-compatible specification
            outdir   - Directory to output files (must already exist
            lanes    - Boolean list of active lanes.  If lanes are None, it's assumed to be a full chip
        '''
        self.filename = filename
        self.dr       = dr
        self.outdir   = outdir
        self.lanes    = lanes

        # Load the chip info
        self.chiptype = ChipType( chiptype, tn='spa' )
        self.transposed = self.chiptype.transpose

        # Load the dat file
        self.datfile  = DatFile( filename, chiptype=self.chiptype, norm=True )
        self.pinned   = self.datfile.pinned
        self.datanorm = self.datfile.data
        self.dataabs  = self.datanorm + self.datfile.offset_norm[:,:,np.newaxis]
        self.isbasic  = self.datfile.isbasic

        self.results  = {}
        self.warnings = []

    def analyze( self ):
        ''' Do the stock analysis '''
        self.analyze_heights()
        self.analyze_driving_potential( normed=True )
        self.analyze_driving_potential( normed=False )
        self.analyze_average_potential()
        self.mark_regions()
        self.plot_regional_traces()

    ###########################
    # Core analysis functions #
    ###########################
    def analyze_average_potential( self ):
        ''' Analyze the average potential per frame, across the entire chip '''
        # Plot the average frame signal
        data = self.dataabs
        frameavg = data.reshape((-1,data.shape[-1])).mean( axis=0 )
        framerate = self.datfile.framerate
        frame_t  = np.arange( frameavg.size )/framerate
        plt.figure()
        plt.plot( frame_t, frameavg )
        plt.xlabel( 'time (s)' )
        plt.ylabel( 'Average Frame Potential (DN14)' )
        plt.title( 'Average Potential' )
        self.savefig( 'framepotential.png' )
        plt.close()

    def analyze_driving_potential( self, normed=False ):
        ''' Calculate the approximate potential which drives kickback '''
        # Get the right data
        if normed:
            data = self.datanorm
            zero_fn = 'zeroed_'
            zero = ' (zeroed)'
        else:
            data = self.dataabs
            zero_fn = ''
            zero = ''

        # Calculate the average row potential, as it is read out
        rowavg = data.astype(np.float).mean( axis=1 )
        rows = rowavg.shape[0]
        # Average top and bottom, from center to edge
        rowavg_ud = np.flipud( rowavg[:rows/2,:] ) + rowavg[rows/2:,:] 
        # Calculate the min and max in each frame
        rowavg_max = rowavg_ud.max( axis=0 )
        rowavg_min = rowavg_ud.min( axis=0 )
        fp = rowavg_ud.T.flatten()

        # Calculate the time for each row
        framerate = self.datfile.framerate
        rowrate = framerate*(rows/2)
        time = np.arange( fp.size )/rowrate

        # Plot the full series
        plt.figure()
        plt.plot( time, fp )
        plt.xlabel( 'time (s)' )
        plt.ylabel( 'Average Row Potential (DN14)' )
        plt.title( 'Driving Potential{}'.format(zero) )
        self.savefig( 'fluidpotential_{}full.png'.format( zero_fn ) )
        # Plot approximatly 4 frames
        plt.xlim( 3, 3.3 )
        self.savefig( 'fluidpotential_{}3s.png'.format( zero_fn ) )
        plt.close()

        # Calculate the peak-to-peak amplitude
        p2p = rowavg_max - rowavg_min
        p2p.sort()
        # Throw out any p2p that are less than 50% of the median 
        # These are likely compression or zeroing artifacts
        med = p2p[p2p.size/2]
        p2p = p2p[p2p>med/2.]
        stats = named_stats( p2p, name='{}p2p'.format( zero_fn ) )
        self.results.update( stats )

    def analyze_heights( self ):
        ''' Analyze the actual kickback '''
        # Calculate the kickback and W1 step size
        if self.isbasic:
            kickback      = self.datanorm.max( axis=2 )
            kickback_time = self.datanorm.argmax( axis=2 )
            stepsize      = -self.datanorm.min( axis=2 )
            stepsize_time = self.datanorm.argmin( axis=2 )
        else:
            kickback      = -self.datanorm.min( axis=2 )
            kickback_time = self.datanorm.argmin( axis=2 )
            stepsize      = self.datanorm.max( axis=2 )
            stepsize_time = self.datanorm.argmax( axis=2 )

        # Calculate the kickback (DN14) in key regions of the chip
        try:
            self.calc_regional_kickback( kickback, 'kickback' )
        except:
            self.warn( 'Unable to calculate regional kickback properties' )

        # Analyze the kickback (DN14)
        kb_lims = self.get_outlier_lims( kickback ) # Dynamicly calculate limits, excluding refpix and pinned
        kb_lims[0] = 0
        self.plot_spatial( kickback, 'kickback_free.png', title='Kickback (DN14)', clim=kb_lims )
        self.plot_spatial( kickback, 'kickback_fixed.png', title='Kickback (DN14)', clim=[0, 200] )
        self.plot_spatial( kickback_time, 'kickback_t0.png', title='Kickback T0', clim=[0, 30] )
        self.do_hist( kickback, [0, 250],     'kickback' )

        # Analyze the step size(DN14)
        ss_lims = self.get_outlier_lims( stepsize ) # Dynamicly calculate limits, excluding refpix and pinned
        ss_lims[0] = 0
        self.plot_spatial( stepsize, 'stepsize_free.png', title='W1 Step Size (DN14)', clim=ss_lims )
        self.plot_spatial( stepsize, 'stepsize_fixed.png', title='W1 Step Size (DN14)', clim=[2500, 5000] )
        self.plot_spatial( stepsize_time, 'maxstep_t0.png', title='Time to max step', clim=[0, 105] )
        self.do_hist( stepsize, [2500, 5000], 'stepsize' )

        # Convert to mV and redo the histograms for stepsize and kickback
        mvcts = 1000.*self.dr/(2**14)
        self.do_hist( kickback*mvcts, [0, 10],     'kickback_mV' )
        self.do_hist( stepsize*mvcts, [50, 200], 'stepsize_mV' )

        # Now normalize the kickback by the median step size and convert to %
        ss_q2 = self.results['stepsize_q2']
        kickback_norm = kickback.astype( np.float )/ss_q2 * 100.
        kb_lims = self.get_outlier_lims( kickback_norm )
        kb_lims[0] = 0
        self.plot_spatial( kickback_norm, 'kickback_rel_free.png', title='Relative Kickback %', clim=kb_lims )
        self.plot_spatial( kickback_norm, 'kickback_rel_fixed.png', title='Relative Kickback %', clim=[0, 5] )
        self.do_hist( kickback_norm, [0, 10],     'kickback_rel' )

        # Calculate the kickback (%) in key regions of the chip
        try:
            self.calc_regional_kickback( kickback_norm, 'kickback_norm' )
        except:
            self.warn( 'Unable to calculate regional kickback properties' )

    def mark_regions( self ):
        ''' Create a figure marking the specific regions of the chip '''
        # load the array mask ( active=1, inactive=0 )
        if not self.lanes:
            gluesafe = self.load_mask().astype( np.int )
            rows, cols = self.FC_REGIONS['inlet']
            gluesafe[rows,cols] = 2
            rows, cols = self.FC_REGIONS['outlet']
            gluesafe[rows,cols] = 3
            rows, cols = self.FC_REGIONS['outedge']
            gluesafe[rows,cols] = 4
            gluesafe[400:] = np.flipud( gluesafe[:400] )
        else:
            # Valkyrie chip. Haven't figured this out yet
            gluesafe = self.load_mask().astype( np.int )
            lanes = []
            for i, active in enumerate( self.lanes ):
                lanedata = self.get_lane( gluesafe, i+1 )
                if active:
                    rows, cols = self.LANE_REGIONS['inlet']
                    lanedata[rows,cols] = 2
                    rows, cols = self.LANE_REGIONS['outlet']
                    lanedata[rows,cols] = 3
                    rows, cols = self.LANE_REGIONS['outedge']
                    lanedata[rows,cols] = 4
                    lanedata[:,150:] = np.fliplr( lanedata[:, :150] )
                lanes.append( lanedata )
            gluesafe = np.hstack( lanes )

        plt.figure()
        colors = ( 'black', 'white', 'blue', 'green', 'red' )
        cm = LinearSegmentedColormap.from_list( 'regions', colors )
        plt.imshow( gluesafe, origin='lower', cmap=cm )
        plt.title( 'Averaging regions' )
        self.savefig( 'regions.png' )
        plt.close()

    def plot_regional_traces( self ):
        ''' Plot the trace in each "region" '''
        data = self.datanorm
        regions = [ 'inlet', 'outlet', 'outedge' ]

        framerate = self.datfile.framerate
        frame_t  = np.arange( data.shape[-1])/framerate

        lines =  ['-', '--', ':', '-.']
        colors = [ 'b', 'g', 'r' ]

        plt.figure()
        ax = plt.subplot(111)
        if self.lanes:
            for i, active in enumerate( self.lanes ):
                if active:
                    lane = i+1
                    for j, reg in enumerate( regions) :
                        rdata = self.get_region( self.get_lane( data, lane ), reg )
                        trace = rdata.reshape( -1, rdata.shape[-1] ).mean( axis=0 ) 
                        lbl = '{}({})'.format( reg, lane )
                        ls = colors[j]+lines[i]
                        plt.plot( frame_t, trace, ls, label=lbl )
        else:
            for reg in regions:
                rdata = self.get_region( data, reg )
                trace = rdata.reshape( -1, rdata.shape[-1] ).mean( axis=0 ) 
                plt.plot( frame_t, trace, label=reg )
        if self.lanes and sum(self.lanes)>1:
            # Too many legend entries.  Need to move the legend out of the figure
            box = ax.get_position()
            ax.set_position( [ box.x0, box.y0, box.width*0.8, box.height ] )
            ax.legend( loc='center left', bbox_to_anchor=(1, 0.5) )
        elif self.isbasic:
            loc = 'upper right'
            plt.legend( loc=loc )
        else:
            loc = 'lower right'
            plt.legend( loc=loc )
        plt.title( 'Average Signal per Region' )
        plt.xlabel( 'Time (s)' )
        plt.ylabel( 'Signal (DN14)' )
        self.savefig( 'regiontraces.png' )
        plt.close()

    ###########################
    # Helper        functions #
    ###########################
    def calc_regional_kickback( self, kickback, name='' ):
        ''' Calculate the kickback in key areas of the chip '''
        if name[-1] != '_':
            name += '_'
            
        regions = {}
        if not self.lanes:
            regions[name+'inlet']   = self.get_region( kickback, 'inlet'  ).mean()
            regions[name+'outlet']  = self.get_region( kickback, 'outlet' ).mean()
            regions[name+'outedge'] = self.get_region( kickback, 'outedge' ).mean()
        else:
            inlet   = []
            outlet  = []
            outedge = []
            for i, active in enumerate( self.lanes ):
                lane = i+1
                if active:
                    kb = self.get_region( self.get_lane( kickback, lane ), 'inlet' ).mean()
                    inlet.append( kb )
                    rg = '{}inlet_lane{}'.format( name, lane )
                    regions[rg] = kb
                    kb = self.get_region( self.get_lane( kickback, lane ), 'outlet' ).mean()
                    outlet.append( kb )
                    rg = '{}outlet_lane{}'.format( name, lane )
                    regions[rg] = kb
                    kb = self.get_region( self.get_lane( kickback, lane ), 'outedge' ).mean()
                    outedge.append( kb )
                    rg = '{}outedge_lane{}'.format( name, lane )
                    regions[rg] = kb
            regions[name+'inlet']   = np.mean( inlet )
            regions[name+'outlet']  = np.mean( outlet )
            regions[name+'outedge'] = np.mean( outedge )

        self.results.update( regions )

    def do_hist( self, data, lims, name ):
        ''' Calculate a histogram, masking bad pixels '''
        flat = self.flatten( data )
        stats = named_stats( flat, name=name, histDir=self.outdir, histlims=lims )
        self.results.update( stats )

    def flatten( self, data ):
        ''' Flatten the data, removing pinned pixels and "reference" pixels '''
        gluesafe = self.load_mask()
        if gluesafe is None:
            goodpix = ~self.pinned
        else:
            goodpix = np.logical_and( gluesafe, ~self.pinned )
        flat = data[goodpix]
        return flat

    def get_outlier_lims( self, data ):
        ''' Get the limits, removing outliers '''
        flat = self.flatten( data )
        flat.sort()
        outlier = flat.size/1000
        return [ flat[outlier], flat[-outlier] ]

    def get_lane( self, data, lane ):
        ''' Extract the specified lane '''
        return data[:,self.LANES[lane]]

    def get_region( self, data, region ):
        ''' Extract pre-defined regions of the chip
            For multilane chips, input data must be specific to a lane
        '''
        if not self.lanes:
            rows, cols = self.FC_REGIONS[region]
            if data.shape[0] == 400:
                # Data is already folded
                return data[rows, cols ]
            top = data[rows,cols]
            bot = np.flipud( np.flipud(data)[rows,cols] )
            return np.vstack( ( top, bot ) )
        else:
            rows, cols = self.LANE_REGIONS[region]
            if data.shape[1] == 150:
                # Data is already folded
                return data[rows, cols ]
            left  = data[rows,cols]
            right = np.fliplr( np.fliplr(data)[rows,cols] )
            return np.hstack( ( left, right ) )

    def load_mask( self ):
        ''' load the appropriate mask (defined in chiptype).  
            Ideally, the gluesafe mask is loaded, but if that isn't defined, 
            use the flowcell mask instead
        '''
        try:
            return self.mask
        except AttributeError:
            pass

        try:
            mask = self.chiptype.gluesafe
        except AttributeError:
            self.warn( 'Unable to load gluesafe mask due to unknown chip type' )
            return None

        filename = '{}/tools/dats/{}.omask'.format( moduleDir, mask )
        try:
            print( 'loading gluesafe mask: {}'.format( filename ) )
            mask = load_mask( filename, self.chiptype )
            self.mask = mask
            return mask
        except:
            self.warn( 'Error reading gluesafe mask' )

        # OK couldn't get the gluesafe.  Try the regular mask
        self.warn( 'Attempting to load the flowcell mask instead of the preferred gluesafe mask' )
        try:
            mask = self.chiptype.flowcell
        except AttributeError:
            self.warn( 'Unable to load flowcell mask due to unknown chip type' )
            return None

        filename = '{}/tools/dats/{}.omask'.format( moduleDir, mask )
        try:
            print( 'loading flowcell mask: {}'.format( filename ) )
            mask = load_mask( filename, self.chiptype )
            self.mask = mask
            return mask
        except:
            self.warn( 'Error reading flowcell mask' )
            return None

    def plot_spatial( self, data, savename, clim=None, title=None ):
        ''' Make a spatial plot of the data and save to disk '''
        plt.figure()
        plt.imshow( data, origin='lower', interpolation='nearest' )
        plt.clim( clim )
        plt.colorbar()
        plt.title( title )
        self.savefig( savename )
        plt.close()

    def savefig( self, fn ):
        ''' Save the figure to the output directory '''
        sn = os.path.join( self.outdir, fn )
        plt.savefig( sn )

    def warn( self, msg ):
        ''' log a warning '''
        self.warnings.append( msg )
        print( 'WARNING! {}'.format( msg ) )

class KickbackIonPlugin( object ):
    ''' Self contained plugin object but not actually an IonPlugin object.
        If you want to turn this into its own plugin, just make a subclass:
            class kickback( IonPlugin, KickbackIonPlugin ): pass
        Analyze the fluidic kickback.  
        Kickback is an electrical parasitic coupling of the average row potential 
          back into the fluidic potential.
        It generally has the effect of creating a signal rise (oposite to an incoming pH change)
          at the outlet of the chip prior to the arival of fluid
        
        Scott Parker, 8/2/2018
        scott.t.parker@thermofisher.com'''
    version = "1.1.1"
    runTypes = [ RunType.THUMB ]

    outjson     = 'results.json'
    results_dir = '.'

    def launch( self ):
        ''' Execute the plugin '''
        self.results = { 'warning': [] }
        self.read_info()
        
        filename = '{}/beadfind_pre_0003.dat_spa'.format( self.raw_tndata_dir )
        if not os.path.exists( filename ):
            # Perhaps it's in the thumbnail directory and this is a full chip report?
            filename = '{}/thumbnail/beadfind_pre_0003.dat_spa'.format( self.raw_tndata_dir )
            
        if self.is_multilane:
            lanes = self.lanes
        else:
            lanes = None
        print( lanes )
        analyzer = KickbackAnalyzer( filename, self.dr_eff, self.chiptype, outdir=self.results_dir, lanes=lanes )
        # Do stock analyses
        try:
            analyzer.analyze()
        finally:
            # Copy the results
            self.results.update( analyzer.results )
            self.results['warning'].append( analyzer.warnings )

            # make the output
            json.dump( self.results, open( os.path.join( self.results_dir, self.outjson ), 'w' ) )
            try:
                BlockHtml( self.results, outdir=self.results_dir )
            except:
                print( 'error making block html' )
                raise
            try:
                Html( self.results, outdir=self.results_dir )
            except:
                print( 'error making main html' )
                raise

        sys.exit(0)

    def read_info( self ):
        """ Reads in namespace-like variables from startplugin.json """

        self.raw_tndata_dir = self.startplugin['runinfo']['raw_data_dir']

        # Read dynamic range from the explog
        self.log = Explog( path=self.raw_tndata_dir )
        if not self.log.found:
            raise IOError( 'Could not load log file ' )
        print( 'Read ExpLog from {}'.format( self.log.log ) )
        self.DR_BF = self.log.metrics['DynamicRangeForBF']
        self.gain  = self.log.metrics['ChipGain']
        if not self.gain:
            self.warn('Gain read in as zero! Assuming gain=1')
            self.gain = 1
        self.dr_eff = self.DR_BF / self.gain
        self.results['dynamic_range'] = self.DR_BF
        self.results['gain']          = self.gain
        self.results['effective_dr']  = self.dr_eff

        # Read the chip type
        try:
            self.chiptype = ChipType( name=self.log.chiptype.name, tn='spa' )
        except:
            self.chiptype = None
            self.warn( 'Unable to detect chip type' )

        self.is_multilane = self.log.is_multilane
        self.lanes = [ self.log.metrics.get(l, False) for l in [ 'LanesActive1', 'LanesActive2', 'LanesActive3', 'LanesActive4' ] ]

    def warn( self, msg ):
        ''' log a warning '''
        self.results['warning'].append( msg )
        print( 'WARNING! {}'.format( msg ) )

class HtmlBase( object ):
    def __init__( self, results, outdir='.' ):
        self.results = results
        self.outdir  = '.'
        self.make_html()

    @staticmethod
    def img_link( img, width=300 ):
        if width:
            wt = 'width="{}"'.format( width )
        else:
            wt = '' 
        txt = '''<a href="{img}"><img {width} src="{img}"/></a>'''.format( img=img, width=wt )
        return txt

    @staticmethod
    def link( dest, content=None ):
        if content is None:
            content = dest
        return '''<a href="{dest}">{content}</a>'''.format( dest=dest, content=content )

    @staticmethod
    def datatable( data ):
        body = '<table>\n'
        for row in data:
            body += '  <tr><th>{0}</th><td>{1}</td></tr>\n'.format( *row)
        body += '</table>\n'
        return body

    @classmethod
    def blanktable( cls, rows, cols ):
        body = [ '<table>' ]
        for row in range(rows):
            fmt = 'r{}_'.format( row ) + 'c{}'
            body += [ cls.blankrow( cols, fmt=fmt ) ]
        body += [ '</table>' ]
        return '\n'.join( body ) + '\n'

    @staticmethod
    def blankrow( cols, fmt='c{}' ):
        body = [ '<tr>' ]
        for col in range(cols):
            cellname = fmt.format( col )
            cellname = '{' + cellname + '}'
            body += [ '  <td>', 
                      '    ' + cellname, 
                      '  </td>' ]
        body += [ '</tr>' ]
        return '\n'.join( body ) + '\n'

    def make_html( self ):
        pass

    def result( self, k, fmt='{}', default='' ):
        try:
            val = self.results[k]
            return fmt.format( val )
        except:
            return default

    def image_list( self ):
        files = os.listdir( self.outdir )
        files = [ f for f in files if '.png' in f ]
        files.sort()
        body = [ '<h2>List of all images</h2>', 
                 '<ul>', ]
        for fn in files:
            body.append( '<li>'+self.link(fn)+'</li>' )
        body.append( '</ul>' )
        return '\n'.join( body ) + '\n'

    def resultslink( self ):
        return '<br>\n' + self.link( 'results.json' ) + '\n<br>\n'

class BlockHtml(HtmlBase):
    def make_html( self ):
        body = self.blanktable(1,4)
        data = ( ( 'Inlet:',  self.result( 'kickback_norm_inlet',   fmt='{:0.2f}%' ) ), 
                 ( 'Outlet:', self.result( 'kickback_norm_outlet',  fmt='{:0.2f}%' ) ), 
                 ( 'Edge:',   self.result( 'kickback_norm_outedge', fmt='{:0.2f}%' ) ), 
                 ( '90%:',    self.result( 'kickback_rel_P90',      fmt='{:0.2f}%' ) ), 
               )
        cell1 = self.datatable( data )
        cell2 = self.img_link( 'kickback_rel_fixed.png' )
        cell3 = self.img_link( 'kickback_t0.png' )
        cell4 = self.img_link( 'fluidpotential_full.png' )
        body = body.format( r0_c0=cell1, r0_c1=cell2, r0_c2=cell3, r0_c3=cell4 )
        with open( os.path.join( self.outdir, 'kickback_block.html' ), 'w' ) as f:
            f.write( body )
 
class Html(HtmlBase):
    images = ( #'fluidpotential_3s', 
               #'fluidpotential_full', 
               'fluidpotential_zeroed_3s', 
               'fluidpotential_zeroed_full', 
               'framepotential', 
               'kickback_fixed', 
               'kickback_free', 
               'kickback_histogram', 
               'kickback_mV_histogram', 
               'kickback_rel_fixed', 
               'kickback_rel_free', 
               'kickback_rel_histogram', 
               'kickback_t0', 
               'maxstep_t0', 
               'stepsize_fixed', 
               'stepsize_free', 
               'stepsize_histogram', 
               'stepsize_mV_histogram', )

    def make_html( self ):
        body = ''
        body += self.resultslink()
        body += '<table>\n'
        row = self.blankrow( 4 )

        # Make the kickback table row
        data = ( ( 'Inlet:',  self.result( 'kickback_norm_inlet',   fmt='{:0.2f}%' ) ), 
                 ( 'Outlet:', self.result( 'kickback_norm_outlet',  fmt='{:0.2f}%' ) ), 
                 ( 'Edge:',   self.result( 'kickback_norm_outedge', fmt='{:0.2f}%' ) ), 
                 ( '90%:',    self.result( 'kickback_rel_P90',      fmt='{:0.2f}%' ) ), 
               )
        content = { 'c0': '<h2>Kickback</h2>\n' + self.datatable( data ),
                    'c1': self.img_link( 'kickback_rel_fixed.png' ), 
                    'c2': self.img_link( 'kickback_rel_histogram.png' ),
                    'c3': self.img_link( 'kickback_t0.png' ), }
        body += row.format( **content )

        # make the regional trace plots
        data = ( ( 'Inlet:',  self.result( 'kickback_norm_inlet',   fmt='{:0.2f}%' ) ), 
                 ( 'Outlet:', self.result( 'kickback_norm_outlet',  fmt='{:0.2f}%' ) ), 
                 ( 'Edge:',   self.result( 'kickback_norm_outedge', fmt='{:0.2f}%' ) ), 
                 ( '90%:',    self.result( 'kickback_rel_P90',      fmt='{:0.2f}%' ) ), 
               )
        content = { 'c0': '',
                    'c1': self.img_link( 'regions.png' ), 
                    'c2': self.img_link( 'regiontraces.png' ),
                    'c3': '' }
        body += row.format( **content )


        # Make the stepsize row
        data = ( ( 'q2:',  self.result( 'stepsize_mV_q2',  fmt='{:0.1f}' ) ), 
                 ( 'iqr:', self.result( 'stepsize_mV_iqr', fmt='{:0.1f}' ) ), 
                 ( '90%:', self.result( 'stepsize_mV_P90', fmt='{:0.1f}' ) ), 
               )
        content = { 'c0': '<h2>W1 Stepsize (mV)</h2>\n' + self.datatable( data ),
                    'c1': self.img_link( 'stepsize_fixed.png' ), 
                    'c2': self.img_link( 'stepsize_histogram.png' ),
                    'c3': self.img_link( 'maxstep_t0.png' ), }
        body += row.format( **content )

        # Make the avgpotential row
        data = ( ( 'q2:',  self.result( 'p2p_q2',  fmt='{:0.0f}' ) ), 
                 ( 'iqr:', self.result( 'p2p_iqr', fmt='{:0.0f}' ) ), 
                 ( '90%:', self.result( 'p2p_P90', fmt='{:0.0f}' ) ), 
               )
        content = { 'c0': '<h2>Peak to Peak (DN14)</h2>\n' + self.datatable( data ),
                    'c1': self.img_link( 'fluidpotential_full.png' ), 
                    'c2': self.img_link( 'fluidpotential_3s.png' ),
                    'c3': self.img_link( 'framepotential.png' ), }
        body += row.format( **content )

        body += '</table>\n'

        body += self.image_list()

        with open( os.path.join( self.outdir, 'kickback.html' ), 'w' ) as f:
            f.write( body )
