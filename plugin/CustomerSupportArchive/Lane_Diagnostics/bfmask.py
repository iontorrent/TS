import sys, os, re
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.append('/home/phil/code')
from tools.core import chiptype as ct

class BeadfindMask:
    """ New and improved class for reading beadfind files """
    def __init__( self , bfmask_file , chiptype=None , fromblocks=False ):
        self.file       = bfmask_file
        self.dir        = os.path.dirname( self.file )
        self.fromblocks = fromblocks
        
        # Try to determine if we have to try from blocks:
        if not os.path.exists( bfmask_file ):
            print( 'Warning!  Full chip bfmask file not found.  Will try to pull from blocks.' )
            self.fromblocks = True
        else:
            self.fromblocks = fromblocks
            
        if chiptype:
            self.chiptype = ct.ChipType( chiptype )
            self.rows     = self.chiptype.chipR
            self.cols     = self.chiptype.chipC
        else:
            self.chiptype = chiptype
            
        self.masks      = ( ('empty'       , 2**0 , '__gt__' , 0 ),
                            ('bead'        , 2**1 , '__gt__' , 0 ),
                            ('live'        , 2**2 , '__gt__' , 0 ),
                            ('dud'         , 2**3 , '__gt__' , 0 ),
                            ('reference'   , 2**4 , '__gt__' , 0 ),
                            ('tf'          , 2**5 , '__gt__' , 0 ),
                            ('lib'         , 2**6 , '__gt__' , 0 ),
                            ('pinned'      , 2**7 , '__gt__' , 0 ),
                            ('ignore'      , 2**8 , '__gt__' , 0 ),
                            ('washout'     , 2**9 , '__gt__' , 0 ),
                            ('exclude'     , 2**10, '__gt__' , 0 ),
                            ('keypass'     , 2**11, '__gt__' , 0 ),
                            ('badkey'      , 2**12, '__gt__' , 0 ),
                            ('short'       , 2**13, '__gt__' , 0 ),
                            ('badppf'      , 2**14, '__gt__' , 0 ),
                            ('badresidual' , 2**15, '__gt__' , 0 ),
                            ('filtered'    , 61440, '__gt__' , 0 ),
                            ('filtpass'    , 61504, '__eq__' , 64),
                            ('goodkey'     ,  4098, '__eq__' , 2 ),
                            ('useless'     ,  4097, '__gt__' , 0 ) )
        
        # Initialize other variables....
        self.defined_masks     = [ msk[0] for msk in self.masks ]
        self.current_mask_name = None
        self.metrics           = {}
        self.failure           = False

        # Automate loading of the bfmask file.
        self.load ( )
        
    def analyze( self , outdir ):
        """
        Normal edgeEffects analysis of bfmask files
        """
        # Go ahead and analyze all defined masks
        for mask in self.defined_masks:
            self.process_mask( mask , outdir )
            
        # Create overlay plots
        self.overlay_plot( outdir , top=True  )
        self.overlay_plot( outdir , top=False )
        
        # Save loading by block to file for Yating.
        # I don't think this is used much anymore; also I'm going to ignore masking pins for now.
        # Also, I'm not going to save dat files anymore.  They're in results.json.
        self.select_mask( 'bead' )
        loading_by_block = self.block_reshape( self.current_mask , [self.chiptype.blockR,self.chiptype.blockC] )
        np.array( loading_by_block , float ).mean(2).tofile( os.path.join(outdir,'loading_by_block.dat') )

        return None
    
    def column_average_plot( self , path='' ):
        """
        Plots full chip version of column average.
        Replaces previous "linear(...)" method.
        """
        x , data = self.process_column_average( full_chip=True )
        plt.plot    ( x , 100. * data , '-' )
        plt.title   ( 'bfmask.{} per row (%)'.format( self.current_mask_name ) )
        plt.xlabel  ( 'Row' )
        plt.ylabel  ( 'Percent pixels in {} mask'.format( self.current_mask_name ) )
        
        if path == '':
            plt.show( )
        else:
            plt.savefig( os.path.join( path , '{}_column_avg.png'.format(self.current_mask_name) ))
            plt.close  ( )
            
        return None

    def edge_plot( self , path='' , top=False ):
        x , data = self.process_column_average( top )
        
        if top:
            side = 'top'
        else:
            side = 'bottom'
            
        plt.figure  ( )
        plt.plot    ( x , 100. * data , '-' )
        plt.title   ( 'bfmask.{0} per row at {1} of chip (%)'.format( self.current_mask_name , side ) )
        plt.xlabel  ( 'Rows from %s of chip' % side )
        plt.ylabel  ( 'Percent pixels in %s mask' % self.current_mask_name )
        
        # Assign metrics
        if top:
            self.metrics['{}_cross_top'.format(self.current_mask_name)]    = self.chiptype.blockR - self.get_crossing_point(x[::-1],data) # Have to undo auto-reversal
            self.metrics['{}_integral_top'.format(self.current_mask_name)] = np.trapz( data , x[::-1] ) # Have to undo auto-reversal 
        else:
            self.metrics['{}_cross_bot'.format(self.current_mask_name)]    = self.get_crossing_point(x,data)
            self.metrics['{}_integral_bot'.format(self.current_mask_name)] = np.trapz( data , x ) 
            
        if path == '':
            plt.show( )
        else:
            plt.savefig( os.path.join( path , '{0}_{1}_edge.png'.format(self.current_mask_name,side)))
            plt.close  ( )
            
        return None

    def get_column_average( self ):
        if self.current_mask_name is None:
            print( 'Wait!  You have not selected a mask yet!' )
            return None
        
        R , C       = self.chiptype.blockR , self.chiptype.blockC
        column_data = np.array( self.current_mask[:,3*C:9*C] , float ).mean(1)
        
        # Store column average for later
        setattr( self , '{}_column_average'.format( self.current_mask_name ) , column_data )
        return None

    def heatmap( self , path='' , mask=None ):
        if mask is not None:
            self.select_mask( mask )
            
        x = self.block_reshape( self.current_mask , [self.chiptype.miniR,self.chiptype.miniC] )
        plt.figure  ( )
        plt.imshow  ( 100. * np.array(x , float).mean(2), interpolation='nearest', origin='lower', clim=[0,100])
        plt.xticks  ( np.arange(0,x.shape[1]+1,x.shape[1]/6) , np.arange(0,self.cols+1,self.cols/6) )
        plt.yticks  ( np.arange(0,x.shape[0]+1,x.shape[0]/4) , np.arange(0,self.cols+1,self.cols/4) )
        plt.title   ( 'bfmask.%s' % self.current_mask_name )
        plt.colorbar( shrink=0.7 )
        
        # Display or not depending on the path
        if path == '':
            plt.show( )
        else:
            plt.savefig( os.path.join( path , '%s_heatmap.png' % ( self.current_mask_name ) ) )
            plt.close  ( )
            
        return None

    def load( self ):
        """ Loads bfmask.bin file """
        if self.fromblocks:
            # Read from block dirs, if those files are found.
            # This code is pretty untested. - Phil
            dirlist    = os.listdir( self.dir )
            seq_block  = re.compile( r'block_X[0-9]+_Y[0-9]+' )
            blocknames = [ d for d in dirlist if seq_block.match( d ) ]
            
            rowblocks  = sorted(list(set( [int(y.split('_')[2][1:]) for y in blocknames ] )))
            colblocks  = sorted(list(set( [int(x.split('_')[1][1:]) for x in blocknames ] )))
            
            (r,c)      = ( rowblocks[1] , colblocks[1] )
            if not os.path.exists( os.path.join(self.dir , blocknames[0] , 'analysis.bfmask.bin') ):
                print('ERROR!  Cannot find block bfmask files.' )
                self.failure = True
                return None
            
            if not self.chiptype:
                self.chiptype = ct.get_ct_from_dir( self.dir )
                self.rows     = self.chiptype.chipR
                self.cols     = self.chiptype.chipC
                
            self.data = np.zeros( (self.chiptype.chipR,self.chiptype.chipC) , np.int16 )
            for row in rowblocks:
                for col in colblocks:
                    f = os.path.join(self.dir,'block_X{0}_Y{1}'.format(col,row),'analysis.bfmask.bin')
                    raw = np.fromfile( f , dtype=np.dtype('u2') )
                    rowslice = slice( row , row+r )
                    colslice = slice( col , col+c )
                    self.data[rowslice,colslice] = raw[4:].reshape(r,c)
        else:
            # Use full chip file
            raw  = np.fromfile( self.file , dtype=np.dtype('u2') )
            
            # Read header to extract rows, cols.
            hdr  = raw[0:4]
            self.rows = hdr[0]
            self.cols = hdr[2]
            self.data = raw[4:].reshape(( self.rows , self.cols ))
            
            if not self.chiptype:
                self.chiptype = ct.get_ct_from_rc( self.rows , self.cols )
        
        # Initialize "current" mask
        self.current_mask      = np.zeros( (self.rows,self.cols) , dtype=bool )
        self.current_mask_name = None
        
        return None
    
    def odds_ratio( self , path='' , top=False , mask='goodkey' , smooth=True ):
        """ 
        Plots edge loading behavior at top or bottom (default) of chip using odds ratio 
        Odds ratio is defined as ratio of probability of (x) to probability of not(x) more or less...
        Note that previous code saved these images as 'goodkey_or_top/bot.png'
        """
        x , data = self.process_column_average( top , mask=mask )
        OR       = data / (1-data)
        
        plt.figure ( )
        if smooth:
            plt.plot   ( x[:-2] , np.convolve( OR , np.ones(5,)/5. )[2:-4] )
        else:
            plt.plot   ( x , OR )
            
        if top:
            side = 'top'
            self.metrics['{}_OR_cross_top'.format(mask)] = self.chiptype.blockR - self.get_crossing_point(x[::-1],OR,4) # Have to undo auto-reversal 
            if mask == 'goodkey':
                self.metrics['goodkey_OR_top'] = list( OR )
                plt.axvline( self.metrics['goodkey_OR_cross_top'] , ls=':' , color='black' )
                plt.axhline( 4 , ls=':' , color='grey' )
                plt.text   ( self.metrics['goodkey_OR_cross_top']+20 , 0.2 * OR.max() , 'Crosses x=0.8\nAt row {}'.format(self.metrics['goodkey_OR_cross_top']) )
        else:
            side = 'bottom'
            self.metrics['{}_OR_cross_bot'.format(mask)] = self.get_crossing_point(x,OR,4)
            if mask == 'goodkey':
                self.metrics['goodkey_OR_bot'] = list( OR )
                plt.axvline( self.metrics['goodkey_OR_cross_bot'] , ls=':' , color='black' )
                plt.axhline( 4 , ls=':' , color='grey' )
                plt.text   ( self.metrics['goodkey_OR_cross_bot']+20 , 0.2 * OR.max() , 'Crosses x=0.8\nAt row {}'.format(self.metrics['goodkey_OR_cross_bot']) )
                
        plt.ylabel ( 'Odds ratio for {} mask'.format(mask) )
        plt.title  ( 'Odds ratio per row at %s of chip (%%)' % ( side ) )
        plt.xlabel ( 'Rows from %s of chip' % side )
        
        if path == '':
            plt.show   ( )
        else:
            plt.savefig( os.path.join( path , '{0}_{1}_edge_OR.png'.format(mask,side) ))
            plt.close  ( ) 
            
        return None
    
    def overlay_plot( self , path='' , top=False ):
        """ 
        Special plot that creates bead, badkey, empty, and bead-badkey on same plot.
        """
        x , bead   = self.process_column_average( top=top , mask='bead'   )
        _ , badkey = self.process_column_average( top=top , mask='badkey' )
        _ , empty  = self.process_column_average( top=top , mask='empty'  )
        
        if top:
            side = 'top'
            if 'xaxis' not in self.metrics:
                self.metrics['xaxis']  = list( x )
            self.metrics['bead_top' ]  = list( bead  )
            self.metrics['empty_top']  = list( empty )
            self.metrics['badkey_top'] = list( badkey )
        else:
            side = 'bottom'
            self.metrics['xaxis']      = list( x )
            self.metrics['bead_bot' ]  = list( bead   )
            self.metrics['empty_bot']  = list( empty  )
            self.metrics['badkey_bot'] = list( badkey )
            
        plt.figure( )
        plt.plot  ( x , 100.*bead          , '-' , color='blue'   , label='Beads' )
        plt.plot  ( x , 100.*(bead-badkey) , '-' , color='green'  , label='Beads-BadKey' )
        plt.plot  ( x , 100.*empty         , '-' , color='red'    , label='Empty' )
        plt.plot  ( x , 100.*badkey        , '-' , color='orange' , label='BadKey' )
        plt.legend( loc='center right' )
        plt.ylim  ( 0 , 100 )
        plt.xlabel( 'Rows from edge'   )
        plt.ylabel( 'Percent of wells' )
        plt.title ( 'Overlay plot for chip {}'.format(side) )
        plt.grid  ( )
        
        if path == '':
            plt.show ( )
        else:
            plt.savefig( os.path.join( path , '{}_overlay.png'.format(side) ) )
            plt.close  ( )
            
        return None
    
    def plot_mask( self , path='' , mask=None ):
        if mask is not None:
            self.select_mask( mask )

        # Note -- to save on memory, let's plot every 6 pixels.
        plt.figure  ( )
        plt.imshow  ( self.current_mask[::6,::6], interpolation='nearest', origin='lower', clim=[0,1] )
        plt.title   ( 'bfmask.%s' % self.current_mask_name )
        plt.xticks  ( np.arange(0,self.cols+1,self.cols/6) )
        plt.yticks  ( np.arange(0,self.cols+1,self.cols/4) )

        if path == '':
            plt.show( )
        else:
            plt.savefig( os.path.join( path , '%s_mask.png' % ( self.current_mask_name ) ) )
            plt.close  ( )

        return None
    
    def process_column_average( self , top=False , full_chip=False , mask=None ):
        """ 
        Averages column average data by microblock and returns desired region of interest.
        full_chip overrides top input.
        """
        if mask:
            try:
                colavg = getattr( self , '{}_column_average'.format( mask ) )
            except AttributeError:
                print( 'Error!  Mask {} has not been selected yet.'.format( mask ) )
                return None,None
        elif self.current_mask_name:
            colavg = getattr( self , '{}_column_average'.format( self.current_mask_name ) )
        else:
            print( 'Wait!  You have not selected a mask yet!' )
            return None, None
        
        rows_to_avg = self.chiptype.microR
        if full_chip:
            data = colavg.reshape(-1,rows_to_avg).mean(1)
            x    = np.arange     ( 0 , self.chiptype.chipR , rows_to_avg )
        else:
            x = np.arange( 0 , self.chiptype.blockR , rows_to_avg )
            if top:
                data = colavg[-self.chiptype.blockR:].reshape(-1,rows_to_avg).mean(1)
                x    = x[::-1] # We reverse the x axis in the plot to make it "rows from glue"
            else:
                data = colavg[:self.chiptype.blockR].reshape(-1,rows_to_avg).mean(1)
                
        return x , data
    
    def process_mask( self , mask , outdir ):
        """
        Canned analysis for a given mask.  Saves plots to file.
        """
        # Check that mask exists
        if mask not in self.defined_masks:
            print( 'ERROR!  Requested {} mask is not defined!'.format( mask ) )
        else:
            # Load mask (which also creates column average)
            self.select_mask        (  mask  )
            self.heatmap            ( outdir )
            self.column_average_plot( outdir ) 
            
            # Make edge plots and define mean metric
            self.edge_plot          ( outdir , top=True  )
            self.edge_plot          ( outdir , top=False )
            self.metrics['{}_cross_mean'.format(mask)]    = float( self.metrics['{}_cross_top'.format(mask)] + self.metrics['{}_cross_bot'.format(mask)] ) / 2.
            self.metrics['{}_integral_mean'.format(mask)] = float( self.metrics['{}_integral_top'.format(mask)] + self.metrics['{}_integral_bot'.format(mask)] ) / 2.
            
            if mask in ['bead','goodkey']:
                self.odds_ratio     ( outdir , top=True  , mask=mask )
                self.odds_ratio     ( outdir , top=False , mask=mask )
                if mask == 'goodkey':
                    self.metrics['goodkey_OR_cross_mean'] = float( self.metrics['goodkey_OR_cross_top'] + self.metrics['goodkey_OR_cross_bot'] ) / 2.
                    self.metrics['bead_OR_cross_mean']    = float( self.metrics['bead_OR_cross_top'] + self.metrics['bead_OR_cross_bot'] ) / 2.
        return None
    
    def select_mask( self , mask_name ):
        found = False
        for mask in self.masks:
            if mask_name == mask[0]:
                # This is a little tricky.
                # 
                # Filtered gives a bitwise_and of anything in the last 4 bits which means
                #  [2**12 + 2**13 + 2**14 + 2**15] = 61140
                #  bitwise_and( self.data , 61140 ) will return value > 0 if the well is filtered
                #  in any of these bits.
                #
                # For filterpass wells, these would not havee any bits flipped from 12-15.  But,
                #   they would have the lib bit flipped (6).  So we add 2**6 to 61140.  However,
                #   in this case we only want to find those wells where the bitwise_and
                #   evaluates to 2**12 == 64, meaning they are library but do not have bits 12-15
                #   flipped.  This is why 61504 is stored.
                #
                # This is equivalent to the previous statment: np.logical_and( lib, ~filtered)
                # self.current_mask = np.bitwise_and( self.data , mask[1] ) == 64
                
                found = True
                self.current_mask_name = mask[0]
                self.current_mask = getattr(np.bitwise_and(self.data, mask[1]), mask[2])(mask[3])
                print( 'Mask changed to %s' % self.current_mask_name )
                self.get_column_average( )
                
        if not found:
            print( 'Error! Requested mask [%s] is not present in self.masks.' % mask_name )
            print( 'Current mask remains: %s' % self.current_mask_name )
            
        return None
    
    @staticmethod
    def block_reshape( data , blocksize ):
        """ Copy of block reshape functionality. """
        rows, cols = data.shape
        numR = rows/blocksize[0]
        numC = cols/blocksize[1]
        return data.reshape(rows,numC,-1).transpose((1,0,2)).reshape(numC,numR,-1).transpose((1,0,2))
    
    @staticmethod
    def get_crossing_point( xdata , data , cross=0.5 , norm=False , smooth=5 ):
        # Do the smoothing operation
        trim     = (smooth-1) / 2
        smoothed = np.convolve( data , np.ones(smooth,)/float(smooth) )[trim:-trim]
        
        # Normalize if desired.
        if norm:
            smoothed = smoothed / smoothed.max()
            
        # Shift data down by crossing point and take absolute value
        #shifted  = np.abs( smoothed - cross )
        shifted  = np.abs( data - cross )
        index    = np.argmin(shifted)
        
        #if index == len(xdata):
            # Guess that this means loading is excellent.  So then set index = 0.
            #index = 0
        
        return xdata[index]
