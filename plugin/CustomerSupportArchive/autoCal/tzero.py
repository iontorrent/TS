import argparse
import sys, os, json, re
import numpy as np
import matplotlib
matplotlib.use( 'agg' )
import matplotlib.pyplot as plt
import textwrap

class TZero:
    def __init__( self , jsonfile , outputdir ):
        ''' loads t0 debug json file and parses it. '''
        if os.path.exists( jsonfile ):
            with open( jsonfile , 'r' ) as f:
                self.raw = json.load( f )
                
            self.info = self.raw['FrameInfo']
            # Need to parse and detect r,c blocks
            region_pattern = re.compile( r'Region_X(\d+)_Y(\d+)' )
            self.regions   = [ region_pattern.match( x ).groups() for x in self.info ]
            cols , rows    = zip(*self.regions)
            self.rows      = sorted( list(set( rows )) , key=int )
            self.cols      = sorted( list(set( cols )) , key=int )
            
            # Create output folder if needed
            self.outputdir = outputdir
            if not os.path.exists( outputdir ):
                os.mkdir( outputdir )
                
            self.imgdir = os.path.join( outputdir , 'plots' )
            if not os.path.exists( self.imgdir ):
                os.mkdir( self.imgdir )
                
            self.img_relpath = 'plots'
            
            # Create container for averaging warnings
            self.warnings = { 'early' : [] , 'late' : [] , 'fake_t0' : [] }
            
            # Start HTML
            self.init_html( )
        else:
            raise IOError( 'Error!  File not found.' )
        
    def close_html( self ):
        if os.path.exists( os.path.join( self.outputdir , 't0_spatial.png' ) ):
            self.html += '<a href="t0_spatial.png"><img src="t0_spatial.png" width="50%%" /></a>'
        self.html += '</body></html>'
        
    def cycle_regions( self , make_html=True ):
        def img_link( imgpath , width=100 ):
            ''' Returns code for displaying an image also as a link '''
            text = '<a href="%s"><img src="%s" width="%d%%" /></a>' % ( imgpath, imgpath , width )
            return text
        
        if make_html:
            #self.html += '<table border="0" cellspacing="0" width="100%%">'
            self.html += '<table border="1" cellspacing="0" cellpadding="4">'
            for row in self.rows[::-1]:
                tr = '<tr>'
                for col in self.cols:
                    region   = 'Region_X%s_Y%s' % ( col , row )
                    self.region_plot( region )
                    img_path = os.path.join( self.img_relpath , '%s.png' % region )
                    if self.false_start( region ):
                        tr += '<td class="early">' + img_link( img_path ) + '</td>'
                    elif self.too_late( region ):
                        tr += '<td class="late">'  + img_link( img_path ) + '</td>'
                    elif self.fake_t0( region ):
                        tr += '<td class="fake">'  + img_link( img_path ) + '</td>'
                    else:
                        tr += '<td>'  + img_link( img_path ) + '</td>'
                        
                tr += '</tr>'
                self.html += tr + '\n'
                
            self.html += '</table>'
        else:
            for row in self.rows:
                for col in self.cols:
                    self.region_plot( 'Region_X%s_Y%s' % ( col , row ) )

    def fake_t0( self , region_name ):
        ''' compares the acquisition trace to the estimated T0 to see if there is a fake T0 (e.g. "Half Moon Bay" issue) in the trace. '''
        region = self.info[ region_name ]
        
        # We are looking for too much signal in trace2.
        index  = int( region['estimate T0'] )
        acq    = np.array( region['trace2'] )
        height = acq.max( )

        if height == 0:
            # There are no traces here.
            return False
        
        # Method 1: acquistion trace at T0 has >= 10% of max height
        metric = 100. * float(acq[index]) / height
        error1 = ( metric >= 10. )
        if error1:
            print( 'WARNING! %s: Acquisition trace has abnormal signal at estimated T0 of %.1f%% of max height.' % (region_name,metric) )
            self.warnings['fake_t0'].append( { region_name : 'signal:%.1f%%' % metric } )
            
        # Method 2: acquisition trace has a high slope before T0....high = 5?
        if index == 0:
            error2 = False
        else:
            slope  = np.diff( acq )
            slmax  = slope[:index].max( )
            error2 = ( slmax > 5 ) 
            if error2:
                print( 'WARNING! %s: Acquisition trace has abnormal positive slope before estimated T0 of %.1f.' % (region_name,slmax) )
                self.warnings['fake_t0'].append( { region_name : 'max_slope:%.1f' % slmax } )
                
        return (error1 or error2)
    
    def false_start( self , region_name ):
        ''' Compares estimated T0 with post-t0-vfc to check for starting T0 too early on averaged frames '''
        region = self.info[ region_name ]
        
        # Estimated T0 can be used as the index for the vfc trace
        index  = int( region['estimate T0'] )
        avgN   = int( region['Post-t0-vfc'][index] )
        error  = ( avgN != 1 )
        if error:
            print( 'WARNING! %s: T0 estimated at averaged frame (avgN = %d at frame %d)' % (region_name,avgN,index))
            self.warnings['early'].append( { region_name : avgN } )
        return error

    def init_html( self ):
        self.html = textwrap.dedent( '''\
                 <html>
                 <head><title>T0 Debug</title></head>
                 <body>
                 <style type="text/css">td.good {background-color:green; }</style>
                 <style type="text/css">td.early {background-color:red; }</style>
                 <style type="text/css">td.late {background-color:orange; }</style>
                 <style type="text/css">td.fake {background-color:magenta; }</style>
                 <h3><center>T0 Estimate Debug Output</center></h3>
                 '''
                 )
        return None
    
    def region_plot( self , region_name ):
        ''' Plots data from each frame '''
        region = self.info[ region_name ]
        fig , ax1 = plt.subplots()
        sgn = np.sign( np.mean(region['trace0']) )
        ax1.plot       ( sgn * np.array(region['trace0']) / 10. , '-' , label='BF1 / 10'   )
        sgn = np.sign( np.mean(region['trace1']) )
        ax1.plot       ( sgn * np.array(region['trace1']) / 10. , '-' , label='BF3 / 10'   )
        sgn = np.sign( np.mean(region['trace2']) )
        ax1.plot       ( sgn * np.array(region['trace2']) , '-' , label='Final Acq' )
        ax1.axvline    ( int(region['estimate T0']) , ls='--' , color='orange' , lw=2.0 )
        ax1.text       ( int(region['estimate T0']) +1 , 480 , 'Estimated T0' , horizontalalignment='left' ,
                         weight='semibold' , fontsize=10 , color='orange' , rotation=90 , va='top' )
        ax1.set_ylabel ( 'Signal Counts' )
        ax1.set_ylim   ( -50 , 650 )
        ax1.set_yticks ( np.arange( 0 , 601 , 100 ) )
        ax1.set_xlabel ( 'Frames' )
        ax1.set_title  ( region_name )
        ax1.grid       ( )
        ax1.legend     ( loc='upper left' , fontsize=10 )
        
        ax2 = ax1.twinx()
        ax2.plot       ( region['Pre-t0-vfc'] , '-' , color='grey' , lw=1.5 , label='Pre-t0 vfc'  )
        ax2.plot       ( region['Post-t0-vfc'], '-' , color='black', lw=2.0 , label='Post-t0 vfc' )
        ax2.set_ylabel ( 'Number of Averaged Frames' )
        ax2.set_ylim   ( -1 , 13 )
        ax2.set_yticks ( np.arange(0,13,2) )
        ax2.legend     ( loc='lower right' , fontsize=10 )
        
        fig.tight_layout()
        #plt.show()
        outpath = os.path.join( self.imgdir , '%s.png' % region_name )
        plt.savefig( outpath )
        plt.close  ( )
        return None
        
    def save_metrics( self ):
        ''' Saves a few metrics to results.json '''
        self.metrics = { 'early_t0_count': len( self.warnings['early'] ) ,
                         'late_t0_count' : len( self.warnings['late']  ) ,
                         'warnings'      : self.warnings }

        with open( os.path.join( self.outputdir , 'results.json' ) , 'w' ) as f:
            json.dump( self.metrics , f )
        
    def t0_spatial_plot( self ):
        ''' '''
        self.t0 = np.zeros( (8,12) , np.int16 )
        for i in range(len(self.rows)):
            for j in range(len(self.cols)):
                region       = 'Region_X%s_Y%s' % ( self.cols[j] , self.rows[i] )
                self.t0[i,j] = int( self.info[ region ]['estimate T0'] )
                
        plt.figure  ( )
        plt.imshow  ( self.t0 , origin='lower', interpolation='nearest' , clim=[10,30] )
        plt.xlabel  ( 'Column Block' )
        plt.ylabel  ( 'Row Block' )
        plt.title   ( 'Estimated T0 (Frame)' )
        plt.colorbar( shrink=0.7 )
        plt.savefig ( os.path.join( self.outputdir , 't0_spatial.png' ) )
        plt.close   ( )
    
    def too_late( self , region_name ):
        ''' Compares estimated T0 with post-t0-vfc to check for starting T0 too late '''
        region = self.info[ region_name ]
        
        # find first frame of high-speed data (excluding very first frame) and then the sixth frame afterwards
        ix     = np.where( np.array( region['Post-t0-vfc'] ) == 1 )[0]
        start  = ix[0]
        if start == 0:
            # Skip this one
            start = ix[1]
            
        delta = region['estimate T0'] - start
        
        # Point out an error if T0 is estimated 6 or more frames after the high speed region starts
        error  = (  delta > 5 )
        if error:
            print( 'WARNING! %s: T0 estimated %s frames after uncompressed frames start!' % (region_name,delta) )
            self.warnings['late'].append( { region_name : delta } )
        return error

    def write_block_html( self ):
        ''' Creates block html output '''
        if len( self.warnings['fake_t0'] ) == 0:
            front_porch_count = 0
        else:
            front_porch_count = len( list( set( [ e.keys()[0] for e in self.warnings['fake_t0'] ] )))
        
        self.block = textwrap.dedent( '''\
                <html><body>
                <table border="0" cellspacing="0" cellpadding="0">
                 <tr>
                  <td width="30%%"><a href="t0_spatial.png"><img src="t0_spatial.png" width="100%%" /></a></td>
                  <td width="70%%"><h3>T0 Warnings:</h3>
                <p>Total false start (early T0) warnings: <b>%d</b></p>
                <p>Total late T0 warnings: <b>%d</b></p>
                <p>Total front porch warnings: <b>%d</b></p></td>
                 </tr>
                </table>
                </body></html>''' % ( len( self.warnings['early'] ) , len( self.warnings['late'] ) , front_porch_count ) )
        
        with open( os.path.join( self.outputdir , 'debugT0_block.html' ) , 'w' ) as f:
            f.write( self.block )
            
    def write_html( self ):
        with open( os.path.join( self.outputdir , 'tzero_debug.html' ) , 'w' ) as f:
            f.write( self.html )
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser( "Tool for visualizing T0 debug output" )

    # Inputs
    parser.add_argument( '-f' , '--json-file' , dest='jsonfile' , required=True , help='Full path to debug json file, usually called T0Estimate_dbg_final.json' )
    parser.add_argument( '-o' , '--outputdir' , dest='outputdir', default=None , help='path to save image files and write html, defaults to directory containing json file.' )
    
    args = parser.parse_args()
    
    if args.outputdir == None:
        args.outputdir = os.path.dirname( args.jsonfile )
        
    if not os.path.exists( args.outputdir ):
        os.mkdir( args.outputdir )
        
    tz = TZero( args.jsonfile , args.outputdir )
    tz.cycle_regions( )
    tz.close_html   ( )
    tz.write_html   ( )
    
    print( 'HTML file created at %s.' % os.path.join( tz.outputdir , 'tzero_debug.html' ) )

