'''
Place to store depreciated functions and classes
Items stored in here should list their origin and date added

The only function which should be run here is 'exists' which checks if the file is 
depreciated and raises an attribute error if it is

This file should not contain import statements, since functions shouldn't be run here.
This is the code graveyard
'''

# ecc_analysis.py - DataBlock : 7/28/2015
def _get_datfile( self, filename, prefix='', bt_method='maxslope' ):
    ''' 
    Reads the datfile and writes the BufferTest result to file.  
    returns ecc_data
    Also saves self.slopes and self.pinned
    '''
    basic = True if bt_method.lower() == 'threshold' else None  # Hopefully historical v2 analysis is the only method which uses threhold.

    filename = os.path.join( self.rawdatadir, filename )
    start = time.time()
    ecc_data = DatFile(filename, isbasic=basic, chiptype=self.chiptype)
    start = time.time()
    bt = ecc_data.BufferTest( method=bt_method )
    self.slopes = np.array( bt['slopes'], dtype=np.float ) * self.chiptype.uvcts * self.framerate
    self.pinned = ecc_data.pinned
    start = time.time()
    ebf = ecc_data.BeadFind()

    # If slopes isn't present, none of these are either.  Save them
    start = time.time()
    self.write_dat( 'pinned',   prefix=prefix, data=ecc_data.pinned )
    self.write_dat( 't0',       prefix=prefix, data=bt['t0'] )
    self.write_dat( 'scales',   prefix=prefix, data=bt['scales'] )
    self.write_dat( 'actpix',   prefix=prefix, data=bt['actpix'] )
    self.write_dat( 'gaincorr', prefix=prefix, data=bt['gaincorr'] )
    self.write_dat( 'gain_iqr', prefix=prefix, data=bt['gain_iqr'] )
    self.write_dat( 'ebfvals',  prefix=prefix, data=ebf['bfmat'] )
    self.write_dat( 'slopes',   prefix=prefix )

    return ecc_data

# ecc_analysis.py - DataBlock : 7/28/2015
def apply_bt( self, bt, prefix='' ):
    ''' takes the buffertest results and applys it to the current object '''
    if prefix:
        prefix += '_'

    # Transfer metrics directly over
    metrics = [ 't0', 'scales', 'actpix', 'gaincorr', 'gain_iqr' ]
    for m in metrics:
        try:
            setattr( self, '%s%s' % ( prefix, m ), bt[m] )
        except ( AttributeError, KeyError ):
            pass

    # rescale the slopes when copying
    slopes = np.array( bt['slopes'], dtype=np.float ) * self.chiptype.uvcts * self.framerate
    setattr( self, '%s%s' % ( prefix, 'slopes' ), slopes )

# ecc_analysis.py - DataBlock : 7/28/2015
def calculate_buffering_v2( self ):
    ''' 
    Calculate chip buffering using ECCv2 methods (threshold slopes and saved flow correction) 
    Saves dat-files condensed metrics
    '''
    raise AttributeError( 'DataBlock.calculate_buffering_v2 is depreciated' )
    # Calculate the buffering
    self.annotate( 'Calculating ECCv2 buffering' )

    # TODO: This is mascarading as a P2 correction, but PCA doesn't work on any TN
    datfilename = 'W1_step.dat'
    start = time.time()
    if self.chiptype.name == 'P2.2.2' and self.chiptype.tn == 'self':
        datfilename = 'W1_step.dat_noPCA'
    elif self.chiptype.tn == 'self':
        self.annotate( "WARNING: using W1_step.dat on a thumbnail.  If this is a PCA file, results will be bad" )

    # Read the slopes from W1_step.dat to self.slopes
    if self.force:
        self._get_datfile( datfilename, prefix='W1_v2', bt_method='threshold' )
    else:
        try:
            self.read_dat( 'slopes', prefix='W1_v2', set_attr=True )
        except IOError:
            self._get_datfile( datfilename, prefix='W1_v2', bt_method='threshold' )
    #self.annotate( ' Time to _get_datfile:  %.1f s.' % ( time.time() - start ) )
    

    # Read the flow correction
    start = time.time()
    fc = self.flowcorr_fromfile()
    #self.annotate( ' Time to load flowcorr:  %.1f s.' % ( time.time() - start ) )
    # Calculate buffering
    start = time.time()
    self.buffering = np.divide( 10000*fc, -self.slopes/1000 ) # slopes/1000 converts uV to mV
    #self.annotate( ' Time to calculate buffering:  %.1f s.' % ( time.time() - start ) )

    # Set this analysis as current
    start = time.time()
    self.write_dat( 'buffering', 'W1_v2' )
    #self.annotate( ' Time to write buffering:  %.1f s.' % ( time.time() - start ) )

# ecc_analysis.py - DataBlock : 7/28/2015
def calculate_buffering_v21( self ):
        ''' Calculate chip buffering using ECCv2 flow correction and ECCv3 slopes '''
        raise AttributeError( 'DataBlock.calculate_buffering_v21 is depreciated' )
        # Calculate the buffering
        self.annotate( 'Calculating ECCv2.1 buffering' )

        datfilename = 'W1_step.dat'
        start = time.time()
        if self.chiptype.name == 'P2.2.2' and self.chiptype.tn == 'self':
            datfilename = 'W1_step.dat_noPCA'
        elif self.chiptype.tn == 'self':
            self.annotate( "WARNING: using W1_step.dat on a thumbnail.  If this is a PCA file, results will be bad" )

        # Read the slopes from W1_step.dat
        if self.force:
            self._get_datfile( datfilename, prefix='W1_v3')
        else:
            try:
                self.read_dat( 'slopes', prefix='W1_v3', set_attr=True )
            except IOError:
                self._get_datfile( datfilename, prefix='W1_v3')

        # Read the flow correction
        fc = self.flowcorr_fromfile()
        # Calculate Buffering
        self.buffering = np.divide( -10000*fc, self.slopes/1000 ) 

        # Set this analysis as current
        self.buffering = self.buffering
        self.write_dat( 'buffering', 'W1_v21' )

# ecc_analysis.py - DataBlock : 7/28/2015
def calculate_buffering_v22( self ):
    ''' Calculate chip buffering using ECCv3 flow correction and ECCv3 slopes on a W1 flow '''
    raise AttributeError( 'DataBlock.calculate_buffering_v22 is depreciated' )
    # Calculate the buffering
    self.annotate( 'Calculating ECCv2.2 buffering' )

    # Read the flow correction
    try:
        fc = self.flowcorr_frombuffer( flow_file='T_step.dat' )
    except IOError:
        self.annotate( '...No flowcorr data present. Skipping V2.2 analysis' )
        return

    datfilename = 'W1_step.dat'
    start = time.time()
    if self.chiptype.name == 'P2.2.2' and self.chiptype.tn == 'self':
        datfilename = 'W1_step.dat_noPCA'
    elif self.chiptype.tn == 'self':
        self.annotate( "WARNING: using W1_step.dat on a thumbnail.  If this is a PCA file, results will be bad" )

    # Read the slopes from W1_step.dat
    if self.force:
        # Do the full analysis
        self._get_datfile( datfilename, prefix='W1_v3')
    else:
        try:
            # Try to just read the slopes data
            self.read_dat( 'slopes', prefix='W1_v3', set_attr=True )
        except IOError:
            self._get_datfile( datfilename, prefix='W1_v3')

    # Calculate the buffering
    self.buffering = np.divide( -10000*fc, self.slopes ) 
    
    # Set this analysis as current
    self.buffering = self.buffering
    self.write_dat( 'buffering', 'W1_v22' )

# ecc_analysis.py - DataBlock : 7/28/2015
def calculate_buffering( self, flow_file='W1_step.dat', fc_file=None, load=True ):
    """
    Generic function to determine the chip buffering from the specified flow file

    the flowcorrection file is specified by fc_file.
    If fc_file is None, only slopes are calculated from flow_file
    if fc_file is 'file', the saved flow-correction file is used
    if otherwise, the specified aquisition file is used for flow-correction

    if load=True, we will attempt to use existing slopes data before running a buffertest
    """
    raise AttributeError( 'DataBlock.calculate_buffering is depreciated' )
    self.annotate( 'Calculating block slopes' )

    filename = os.path.join(self.rawdatadir, flow_file)
    ecc_data = DatFile(filename, chiptype=self.chiptype)
    self.set_rc(ecc_data)
    bt = ecc_data.BufferTest()
    
    if self.analyses['v2']:
        fc = self.flowcorr_fromfile()
    elif fc_file is None:
        fc = np.ones( bt['slopes'].shape )
    else:
        fc = self.flowcorr_frombuffer(fc_file)

    slopes = np.array( bt['slopes'], dtype=np.float ) * self.chiptype.uvcts * self.framerate
    self.slopes = np.divide( slopes, fc ) 

    if fc_file is not None:
        self.buffering = np.divide( 10000, -self.slopes ) 

# ecc_analysis.py - DataBlock : 7/28/2015
def compare_flowcorr( self, blocks=False ):
    """
    Plots the flowcorr against the slopes data.
    Creates two files: a 2d histogram and a point map
    if blocks is set to true, then it breaks the result down by block and writes a unified HTML table
    """

    if not ( hasattr( self, 'flowcorr') and hasattr( self, 'slopes' ) ):
        self.annotate( 'Cannot compare slopes and flowcorr', 0 )
        return

    if blocks and isinstance( self, ( ECC_Analysis, Thumbnail ) ):
        # Setup the limits for each block
        lims = []
        for x in np.arange( 0, self.chiptype.chipC, self.macroC ):
            for y in np.arange( 0, self.chiptype.chipR, self.macroR ):
                lims.append( ( y, y+self.macroR, x, x+self.macroR) )
        # Setup directory to save each file
        dirname = 'slopes_flowcorr'
        system.makedir( os.path.join( self.analysisdir, dirname ) )
    else:
        lims = [ ( None, None, None, None ) ]
        dirname = ''

    # Calculate the medians only once
    flowcorr_norm = np.median( self.flowcorr )
    slopes_norm   = np.median( self.slopes )

    for l in lims:
        # Get the data range
        xdata = self.flowcorr[ l[0]:l[1], l[2]:l[3] ].flatten() / flowcorr_norm
        ydata = self.slopes[ l[0]:l[1], l[2]:l[3] ].flatten() / slopes_norm

        # Calculate the blockname
        if blocks:
            blockname = 'block_X%i_Y%i_' % ( l[2], l[0] )
        else:
            blockname = ''

        # Make the point plot
        filename = '%s/%s/%s%s.png' % ( self.analysisdir, dirname, blockname, 'slopes_flowcorr_raw' )
        f = plt.figure( facecolor='w' )
        plt.plot( xdata, ydata, '.', markersize=1 )
        plt.xlabel( 'flowcorr/median' )
        plt.ylabel( 'slopes/median' )
        plt.xlim( (0,3) )
        plt.ylim( (0,3) )
        f.savefig( filename )
        plt.close( f )

        filename = '%s/%s/%s%s.png' % ( self.analysisdir, dirname, blockname, 'slopes_flowcorr' )
        bins = np.arange(0,2.01,0.05)
        H, x, y = np.histogram2d( ydata, xdata, bins=bins )
        f = plt.figure()
        extent = [0, 2, 0, 2]
        plt.imshow( H, origin='lower', aspect='equal', interpolation='nearest', extent=extent )
        plt.xlabel( 'flowcorr/median' )
        plt.ylabel( 'slopes/median' )
        f.savefig( filename )
        plt.close( f )
    
    if blocks:
        self._make_slopes_flowcorr_html()
    
# ecc_analysis.py - DataBlock : 7/28/2015
def get_minirc(self,block):
    """
    Gets miniR and miniC from the block if it hasn't already been assigned
    """
    if not hasattr(self,'miniR'):
        self.miniR = block.miniR
        self.miniC = block.miniC

# ecc_analysis.py - DataBlock : 7/28/2015
def load_buffering(self): 
    """
    Loads buffering data from file (With older file support)
    """
    warnings.warn( 'This version supports older files', DepreciationWarning )
    try:
        if self.analyses['v2']:
            if isinstance( self, ECC_Analysis ):
                basename = 'fc_buffering_gc.dat'
            else:
                basename = 'fc_buffering.dat'
            filename = os.path.join( self.analysisdir, basename )
            self.buffering = np.fromfile( filename, count=self.chiptype.chipR*self.chiptype.chipC, dtype=np.int16 ).reshape( self.chiptype.chipR, self.chiptype.chipC ).astype(np.float)/10.
        else:
            filename = os.path.join(self.analysisdir,'buffering.dat')
            #self.buffering = np.fromfile(filename,dtype=np.int16)/10
            self.buffering = np.fromfile(filename,dtype=np.float32)
            self.buffering = self.buffering.reshape(self.chiptype.chipR,self.chiptype.chipC)
    except IOError:
        self.annotate( 'Buffering data not present.', 0 )
    except AttributeError:
        self.annotate( 'Could not determine shape of loaded buffering data', 0 )

# ecc_analysis.py - DataBlock : 7/28/2015
def _make_slopes_flowcorr_html( self ):
    """ 
    Makes a single html page with blocks arranged side-by-side.  
    This only works for Thumbnails or ECC_Analysis instances
    """

    self.annotate( 'Writing html output page' )
    
    basename = 'slopes_flowcorr'
    html = '%s/%s.html'   % ( self.analysisdir, basename )
    with open( html , 'w' ) as f:
        # Page header
        f.write( '<html><head>%s</head><body>\n' % 'Slopes (Y) vs Flowcorr (X)' )

        for raw in [ '', '_raw']:
            # Table header 
            f.write( '<table border="1" cellpadding="0" width="100%%">\n' )
            # Loading Table
            try:
                width = 100/self.Xblocks
                for y in reversed( np.arange( 0, self.chiptype.chipR, self.macroR ) ):
                    row = '<tr>\n'
                    for x in np.arange( 0, self.chiptype.chipC, self.macroC ):
                        filename = '%s/block_X%i_Y%i_%s%s.png' % ( basename, x, y, basename, raw )
                        row  += '  <td width="%i%%"><a href="%s"><img src="%s" width="100%%" /></a></td>\n' % ( width, filename, filename )
                    row += '</tr>\n'
                    f.write (row)
            except AttributeError:
                pass
            f.write( '</table><br>' )

        # End of page
        f.write( '</body></html>' )

# ecc_analysis.py - DataBlock : 7/28/2015
def norm_buffering( self ):
    """ sets self.norm_buffering_scale, which is the scale factor to scale buffering to have a median = 1 """
    self.norm_buffering_scale = np.median( self.buffering )

# ecc_analysis.py - DataBlock : 7/28/2015
def open_cal_gain( self ):
    """
    Opens up gain files for calibration data.
    Please use chip name [ "314R" , "P0" , "P2" ] for the chiptype input rather than '900' or 'P1.2.18'
    NOTE: This code writes out gain as a float, in units of V/V, standard for each chip type.

    data is written to self.gain
    """
    img = np.zeros( 0 )
    if self.type in [ '314R' , '316D' , '316E' , '318B' , '318C' , '318G' ]:
        gainfile = os.path.join( self.rawdatadir , 'gainimage.dat' )
    else:
        gainfile = os.path.join( self.rawdatadir , 'gainImage0.dat' )
        if not os.path.exists( gainfile ):
            gainfile = os.path.join( self.rawdatadir , 'gainImage2.dat' )

    if os.path.exists( gainfile ):
        with open( gainfile , 'r' ) as f:
            if self.type in ['314R']:
                img  = np.fromfile( f , count=1152*1280 , dtype=np.dtype('>H')).reshape( 1152 , 1280 , order='C')
                img  = np.array( img , dtype=float ) / 69. / 50.
            elif self.type in ['316D' , '316E']:
                img  = np.fromfile( f , count=2640*2736 , dtype=np.dtype('>H')).reshape( 2640 , 2736 , order='C')
                img  = np.array( img , dtype=float ) / 69. / 50.
            elif self.type in ['318B' , '318C' , '318G']:
                img  = np.fromfile( f , count=3792*3392 , dtype=np.dtype('>H')).reshape( 3792 , 3392 , order='C')
                img  = np.array( img , dtype=float ) / 69. / 50.
            elif self.type in ['P0' , 'P1' , 'P2' , '900' ]:
                hdr  = np.fromfile (f, count=4, dtype=np.dtype('<u4'))
                cols = hdr[2]
                rows = hdr[3]
                img  = np.fromfile (f, count=rows*cols, dtype=np.dtype('<u2')).reshape (rows, cols, order='C')
                # Note:  In other code this was / ( 4 * 1023 ).  Seems weird like it should be 4096 not 4092.
                # Leaving it at 4092 for now.
                img  = np.array( img , dtype=float ) / 4092.
            else:
                self.annotate( 'ERROR: Unknown chip type supplied.', 0 )
    else:
        self.annotate( 'ERROR: Gain file does not exist.', 0 )
        
    self.gain = img

# ecc_analysis.py - DataBlock : 7/28/2015
def open_cal_noise( self ):
    """
    Adapted from cal.py in order to simply load data from calibration noise .dat file for post processing.
    saves noise to self.noise
    """
    fpath = '%s/NoisePic0.dat' % self.rawdatadir 
    
    if not os.path.exists( fpath ):
        self.annotate( 'File does not exist! (%s)' % fpath, 0 )
        return None
    else:
        # get noise data
        f       = open( fpath , 'r' )
        hdr     = np.fromfile ( f, count=4, dtype=np.dtype('<u4') )
        cols    = hdr[2]
        rows    = hdr[3]
        img_fc  = np.fromfile ( f, count=rows*cols, dtype=np.dtype('<u2') ).reshape( rows, cols, order='C' )
        img_fc  = np.asarray ( img_fc, dtype=np.dtype('<u2') )
        
        badcols = np.zeros(( rows , cols ), dtype=bool )
        
        m = np.logical_and( (img_fc > 0) , (img_fc < 600) )
        for r in range(8):
            for c in range(12):
                rws     = (rows / 8 ) * np.array(( r , r + 1 ))
                cls     = (cols / 12) * np.array(( c , c + 1 ))
                roi     = img_fc[rws[0]:rws[1],cls[0]:cls[1]]
                
                # Find bad columns
                colavg    = np.mean( roi , axis = 0 )
                badcolmsk = np.tile( colavg >= ( 2*colavg.std() + colavg.mean() ) , ((rows/8),1) )
                
                badcols[ rws[0]:rws[1] , cls[0]:cls[1] ] = badcolmsk
        f.close ( )

        self.noise = chipcal.PI_noise( img_fc , badcols )

# ecc_analysis.py - DataBlock : 7/28/2015
def open_cal_offest( self ):
    """
    Opens up pixel offset files for calibration data.
    Please use chip name [ "314R" , "P0" , "P2" ] for the chiptype input rather than '900' or 'P1.2.18'
    NOTE: This code writes out offset as a float, in units of mV, standard for each chip type.

    output is written to self.pix_offset
    """
    img = np.zeros( 0 )
    if self.type in [ '314R' , '316D' , '316E' , '318B' , '318C' , '318G' ]:
        pixfile = os.path.join( self.rawdatadir , 'piximage.dat' )
    else:
        pixfile = os.path.join( self.rawdatadir , 'PixImage0.dat' )
        if not os.path.exists( pixfile ):
            pixfile = os.path.join( self.rawdatadir , 'PixImage2.dat' )
            if not os.path.exists( pixfile ):
                pixfile = os.path.join( self.rawdatadir , 'PixImage3.dat' )

    if os.path.exists( pixfile ):
        with open( pixfile , 'r' ) as f:
            if self.type in ['314R']:
                img  = np.fromfile( f , count=1152*1280 , dtype=np.dtype('>H')).reshape( 1152 , 1280 , order='C')
                img  = np.array( img , dtype=float ) / 69.
            elif self.type in ['316D' , '316E']:
                img  = np.fromfile( f , count=2640*2736 , dtype=np.dtype('>H')).reshape( 2640 , 2736 , order='C')
                img  = np.array( img , dtype=float ) / 69.
            elif self.type in ['318B' , '318C' , '318G']:
                img  = np.fromfile( f , count=3792*3392 , dtype=np.dtype('>H')).reshape( 3792 , 3392 , order='C')
                img  = np.array( img , dtype=float ) / 69.
            elif self.type in ['P0' , 'P1' , 'P2' , '900' ]:
                hdr  = np.fromfile (f, count=4, dtype=np.dtype('<u4'))
                cols = hdr[2]
                rows = hdr[3]
                img  = np.fromfile (f, count=rows*cols, dtype=np.dtype('<u2')).reshape (rows, cols, order='C')
                img /= 4
                img  = np.array( img , dtype=float ) * 400. / 4092.
            else:
                self.annotate( 'ERROR: Unknown chip type supplied.', 0 )
    else:
        self.annotate( 'ERROR: Pixel offset file does not exist.', 0 )
        
    self.pix_offset = img

# ecc_analysis.py - DataBlock : 7/28/2015
def open_cal_noise( self ):
    """
    Adapted from cal.py in order to simply load data from calibration noise .dat file for post processing.
      - Should only be used for analysis NOT done on TS.
    Saves noise to self.noise
    """
    fpath = '%s/noise.dat' % self.rawdatadir
    
    if not os.path.exists( fpath ):
        self.annotate('File does not exist! (%s)' % fpath, 0)
        return None
    else:
        # get noise data
        f         = open( fpath , 'r' )
        img_fc    = np.fromfile( f, dtype=np.int16 ).reshape( self.chiptype.chipR, self.chiptype.chipC )
        badcols   = np.zeros(( self.chiptype.chipR , self.chiptype.chipC ), dtype=bool )
        
        # Find bad columns
        colavg    = np.mean( img_fc , axis = 0 )
        badcolmsk = np.tile( colavg >= ( 2*colavg.std() + colavg.mean() ) , (rows,1) )
        badcols   = badcolmsk
        f.close   ( )

        self.noise = PI_noise( img_fc , badcols )

# ecc_analysis.py - DataBlock : 7/28/2015
def plot_all(self,scale=None):
    """
    Plots all available data on the current block
    """
    self.logging('Making plots')
    self.compare_flowcorr()
    self.plot_buffering(scale)
    self.plot_flowcorr(scale)
    self.plot_flowcorr_sign(scale)
    self.plot_slopes(scale)
    self.plot_slopes_sign(scale)

# ecc_analysis.py - DataBlock : 7/28/2015
def plot_buffering(self,scale=None):
    """
    Plots the buffering map
    """
    warnings.warn( 'plot_buffering is depreciated.  Use plot_spatial instead', DepreciationWarning )
    try:
        extent = [0, self.buffering.shape[1], 0, self.buffering.shape[0]]
        data = plotting.downsample(self.buffering,scale=scale)
    except AttributeError:
        self.annotate( 'Not plotting buffering data', 2 )
        return

    dpi = 100

    f = plt.figure( dpi=dpi,
                    facecolor='w' )
    
    plt.imshow(data,aspect='equal',origin='lower',interpolation='nearest',extent=extent)
    plt.clim([0,10])
    plt.colorbar()

    filename = os.path.join(self.analysisdir,'buffering.png')
    f.savefig(filename)
    plt.close(f)

def plot_colnoise( self, fc=False ):
    ''' 
    Plots the column noise at either the block or full chip level
    '''
    plt.figure()
    plt.plot( self.colnoise.std(1), 'o-' )
    plt.ylim( 0, 15 )
    plt.xlim( 0, self.colnoise.shape[0] )
    plt.ylabel( 'Column Noise (DN14 counts)' )
    plt.xlabel( 'Column' )
    plt.savefig( os.path.join( self.analysisdir, 'colnoise.png' ) )
    plt.close()

# ecc_analysis.py - DataBlock : 7/28/2015
def plot_differential_image( self, data, lims=None, rmode=None, metric='', testnum=0 ):
    ''' Plot differential spatial plot '''
    # Not bothering with diff and rmode-based plots in new edge class method.
    # If interest increases again, it can be revisited.

    lims, units = self.metric_plot_params( metric )

    # If image is too big,  have to skip datapoints...
    scale = 1
    # Pick a comon scale factor to preserve the aspect ratio
    for dim in data.shape:
        if dim >= 10000:
            # Maximum data size will be 1800 pixels in any dimension
            scale = max( scale, dim/1800 )
    # Plot the data
    extent = [0, data.shape[1]-1, 0, data.shape[0]-1] 
    plt.imshow   ( data[::scale,::scale] , extent=extent , interpolation='nearest' , origin='lower' , clim=lims )
    plt.colorbar( )
    plt.title   ( 'Differential %s for ECC %s | Mode = %.1f %s' % ( metric , testnum , rmode , units ) )
    plt.xlabel  ( 'Rows' )
    plt.ylabel  ( 'Columns' )
    plt.savefig ( '%s/%s_diff_spatial.png' % ( self.testdir , metric ))
    plt.close   ( )

# ecc_analysis.py - DataBlock : 7/28/2015
def plot_flowcorr(self,scale=None):
    """
    Plots the flowcorr map
    """
    warnings.warn( 'plot_flowcorr is depreciated.  Use plot_spatial instead', DepreciationWarning )
    try:
        extent = [0, self.flowcorr.shape[1], 0, self.flowcorr.shape[0]]
        data = plotting.downsample(self.flowcorr,scale=scale)
    except AttributeError:
        self.annotate( 'Not plotting flowcorr data', 2 )
        return

    dpi = 100

    f = plt.figure( dpi=dpi,
                    facecolor='w' )
    
    plt.imshow(data,aspect='equal',origin='lower',interpolation='nearest',extent=extent)
    plt.clim([0,500])
    plt.colorbar()

    filename = os.path.join(self.analysisdir,'flowcorr.png')
    f.savefig(filename)
    plt.close(f)
        
# ecc_analysis.py - DataBlock : 7/28/2015
def plot_flowcorr_sign(self,scale=None):
    """
    Plots the flowcorr map, indicating positive and negative regions
    """
    warnings.warn( 'plot_flowcorr_sign is depreciated.  Use plot_spatial instead', DepreciationWarning )
    try:
        extent = [0, self.flowcorr.shape[1], 0, self.flowcorr.shape[0]]
        data = np.zeros(self.flowcorr.shape)
        data[self.flowcorr>0] = 1
        data[self.flowcorr<0] = -1
        data = plotting.downsample(data,scale=scale,subsample=True)
    except AttributeError:
        return

    dpi = 100

    f = plt.figure( dpi=dpi,
                    facecolor='w' )
    
    cmap = colors.ListedColormap(['black','red','white'])
    
    plt.imshow(data,aspect='equal',origin='lower',cmap=cmap,interpolation='nearest',extent=extent)
    plt.clim([-1.5,1.5])
    plt.colorbar(ticks=[-1,0,1])

    filename = os.path.join(self.analysisdir,'flowcorr_sign.png')
    f.savefig(filename)
    plt.close(f)

# ecc_analysis.py - DataBlock : 7/28/2015
def plot_slopes(self,scale=None):
    """
    Plots the slopes map
    """
    warnings.warn( 'plot_slopes is depreciated.  Use plot_spatial instead', DepreciationWarning )
    try:
        extent = [0, self.slopes.shape[1], 0, self.slopes.shape[0]]
        data = plotting.downsample(self.slopes,scale=scale)
    except AttributeError:
        self.annotate( 'Not plotting slopes data', 2 )
        return

    dpi = 100

    f = plt.figure( dpi=dpi,
                    facecolor='w' )
    
    plt.imshow(data,aspect='equal',origin='lower',interpolation='nearest',extent=extent)
    if np.median(self.slopes) > 0:
        clim = (0,5000)
    else:
        clim = (-5000,0)
    plt.clim(clim)
    plt.colorbar()

    filename = os.path.join(self.analysisdir,'slopes.png')
    f.savefig(filename)
    plt.close(f)

# ecc_analysis.py - DataBlock : 7/28/2015
def plot_slopes_sign(self,scale=None):
    """
    Plots the slopes map, indicating positive and negative regions
    """
    warnings.warn( 'plot_slopes_sign is depreciated.  Use plot_spatial instead', DepreciationWarning )
    try:
        extent = [0, self.slopes.shape[1], 0, self.slopes.shape[0]]
        data = np.zeros(self.slopes.shape)
        data[self.slopes>0] = 1
        data[self.slopes<0] = -1
        data = plotting.downsample(data,scale=scale,subsample=True)
    except AttributeError:
        return

    dpi = 100

    f = plt.figure( dpi=dpi,
                    facecolor='w' )
    
    cmap = colors.ListedColormap(['black','red','white'])
    
    plt.imshow(data,aspect='equal',origin='lower',cmap=cmap,interpolation='nearest',extent=extent)
    plt.clim([-1.5,1.5])
    plt.colorbar(ticks=[-1,0,1])

    filename = os.path.join(self.analysisdir,'slopes_sign.png')
    f.savefig(filename)
    plt.close(f)

# ecc_analysis.py - DataBlock : 7/28/2015
def read_dats( self, prefix='' ):
    for metric in dp.datprops:
        self.read_dat( metric, prefix )

# ecc_analysis.py - DataBlock : 7/28/2015
def rescale_data( self, data, metric=None, tofile=False ):
    ''' Scales data from saved units to real units in place '''
    if metric in [ 'buffering', 'bf_std', 'qsnr', 'gaincorr', 'gain_iqr', 'phslopes', 'buffering_gc', 'buffering_gsc' ]:
        data /= 10.
    elif metric in [ 'ebfvals' ]:
        data *= self.chiptype.uvcts
    return data

# ecc_analysis.py - ECC_Info : 7/28/2015
def ScanDir( self ):
    ''' depreciated.  use scan_dir instead '''
    warnings.warn( 'ScanDir is depreciated.  Use scan_dir instead', DeprecationWarning )
    self.scan_dir()

# ecc_analysis.py - DataBlock : 7/28/2015
def stich_buffering(self): 
    rc = (self.chiptype.chipR,self.chiptype.chipC)
    for d in self.directories:
        # Try statement allows us to skip thumbnails and any other folder
        try:
            c, r = self.xy_from_block(d)
        except TypeError:
            continue

        # Load the block data
        dirname = os.path.join(self.analysisdir,d)
        block = DataBlock(rawdatadir=dirname, analysisdir=dirname)
        block.load_all()

        
        self.array_insert(self.slopes,   block.slopes,   (r,c))
        try:
            self.array_insert(self.buffering,block.buffering,(r,c))
            self.array_insert(self.flowcorr, block.flowcorr, (r,c))
        except AttributeError:
            self.annotate( 'Skipping buffering and/or flowcorr', 1 )

# ecc_analysis.py - DataBlock : 7/28/2015
def stich_dat(self, filename, datformat=None ): 
    """ 
    function for stiching together an arbitray dat file. 
    This does not stich raw dat files from an instrument. 
    if datformat is unspecifed, we try to guess
    """
    rc = (self.chiptype.chipR,self.chiptype.chipC)
    data = np.zeros( rc, datformat )
    for d in self.directories:
        # Try statement allows us to skip thumbnails and any other folder
        try:
            c, r = self.xy_from_block(d)
        except TypeError:
            continue

        # Load the block data
        block = os.path.join( self.rawdatadir, filename )
        
        self.array_insert( data, block, (r,c) )
    return data

# ecc_analysis.py - DataBlock : 7/28/2015
def write_dats( self, prefix='' ):
    for metric in dp.datprops:
        self.write_dat( metric, prefix )

# ecc_analysis.py - DataBlock : 7/28/2015
def xy_from_block(self,blockname):
    """
    Parses a block name and extracts the X and Y offset.
    returns x, y
    """

    try:
        blockParts = blockname.split('_')
        if blockParts[0].lower() == 'block':
            blockParts.pop(0)
        X = int(blockParts[0][1:])
        Y = int(blockParts[1][1:])

        return X, Y
    except:
        return -1, -1 

# files.py: 7/12/2017
# Replaced with more complete class as stand-alone from chipDiagnostics
class Explog:    
    """ 
    Class to handle explog files.
    if arg is a path (i.e. contains '/explog.txt' or 'explog_final.txt'), it will load file.
    otherwise, arg should be the raw text of the explog file (i.e. single string with '\n' still in place.
    """
    def __init__( self , arg ):
        if ('explog.txt' in arg) or ('explog_final.txt' in arg):
            # This means we have a path
            with open( arg , 'r' ) as f:
                self.lines = f.read().splitlines()
        else:
            # This means we have a string of data.
            self.lines = arg.splitlines()
            
    def parse( self ):
        # Define a dummy class
        class dummy: pass
        
        # Set defaults
        advopt = 'AdvScriptFeaturesName'
        setattr( self , 'ScriptOpt' , [] )
        setattr( self , 'HardDisk' , [] )
        err    = 'ExperimentErrorLog'
        setattr( self, 'Errors' , [] )
        setattr( self, 'Blocks' , [] )
        setattr( self, 'Params' , dummy() )
        setattr( self, 'fields', {} )
        
        for l in self.lines:
            if 'Experiment Name' in l:
                temp = l.replace('%s: ' % l.split(':')[0] , '')
                temp = temp[:7]
                while '-' in temp[-1]:
                    temp = temp[:-1]
                setattr( self , 'runName' , temp )
            elif 'Start Time:' in l:
                try:
                    setattr( self , 'Start_Time' ,  l.replace('%s:' % l.split(':')[0] , '') )
                    if '/' in l:
                        # Proton case
                        self.rundate , self.runtime        = self.Start_Time.strip().split(' ')
                        self.month , self.day , self.year  = self.rundate.split('/')
                        self.hour , self.minute , self.sec = self.runtime.split(':')
                    else:
                        # PGM case
                        _ , m , self.day , self.runtime , self.year = self.Start_Time.strip().split(' ')
                        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                        for i in range(len(months)):
                            if m == months[i]:
                                self.month = str( i+1 )
                        self.rundate = '%s/%s/%s' % ( self.month , self.day , self.year )
                        self.hour , self.minute , self.sec = self.runtime.split(':')
                    # Set ecc-like timestamp
                    self.timestamp = '%s_%s_%s_%s_%s' % ( self.year , self.month , self.day , self.hour , self.minute )
                except:
                    pass
            elif 'Chip Efuse:' in l:
                self.efuse = chip.EfuseClass( l.replace('Chip Efuse:','') )
            elif 'ChipHistogramInfo:' in l:
                setattr( self , 'ChipHistogramInfo' , l.replace('ChipHistogramInfo:','') )
            elif '%s:' % advopt in l:
                self.ScriptOpt.append( l.split(':')[1].strip() )
            elif ('block_' in l) or ('thumbnail_' in l):
                attrname = l.split(':')[0]
                attrdata = l.replace('%s:' % attrname , '' ).split(', ')
                dims     = attrdata[0:4]
                end      = (attrdata.pop()).strip()
                misc     = attrdata[4:]
                rinfo    = dummy()
                rinfo.x , rinfo.y , rinfo.cols , rinfo.rows = ( 0 , 0 , 0 , 0 )
                if 'thumbnail_' in l:
                    rinfo.id = 'thumbnail'
                else:
                    rinfo.id = l.split(':')[0].split('_')[1]
                for d in dims:
                    if 'X' in d:
                        rinfo.x    = int( d.replace('X','') )
                    if 'Y' in d:
                        rinfo.y    = int( d.replace('Y','') )
                    if 'W' in d:
                        rinfo.cols = int( d.replace('W','') )
                    if 'H' in d:
                        rinfo.rows = int( d.replace('H','') )
                for m in misc:
                    setattr( rinfo , m.split(':')[0] , m.split(':')[1] )
                lastbit = end.split(' ')
                setattr( rinfo , lastbit[0].split(':')[0] , lastbit[0].split(':')[1] )
                setattr( rinfo , lastbit[1].split('=')[0] , int( lastbit[1].split('=')[1] ))
                setattr( rinfo , lastbit[2].split('=')[0] , int( lastbit[2].split('=')[1] ))
                # We finally got there... setting something like self.block_035
                if 'block_' in l:
                    self.Blocks.append( rinfo )
                else:
                    setattr( self , 'thumbnail' , rinfo )
            elif 'CalGainStep:' in l:
                self.cal_gain_step = float( l.split(':')[1].strip() )
            elif 'ChipType:' in l:
                self.chip_type = l.split(':')[1].strip()
            elif 'ChipNoiseInfo:' in l:
                setattr( self , 'ChipNoiseInfo' , dummy() )
                items = l.replace('ChipNoiseInfo:','').split(' ')
                for i in items:
                    if len( i.split(':') ) >= 2 :
                        setattr( self.ChipNoiseInfo , i.split(':')[0] , i.split(':')[1] )
            elif 'HardDisk:' in l:
                name = l.split(':')[0]
                rest = l.replace( '%s:' % name , '' ).strip()
                disk      = dummy()
                disk.path , diskinfo = rest.split(',')
                for entry in diskinfo.split(' '):
                    if len( entry.split(':') ) >= 2:
                        setattr( disk , entry.split(':')[0] , entry.split(':')[1] )
                self.HardDisk.append( disk ) 
            elif 'ALARM:' in l:
                self.Errors.append( l.split(':')[1].strip() )
            elif ('OverSample:' in l) or ('ChipBarCode:' in l) or ('ChipFreq:' in l) or ('ChipVersion:' in l) or ('ECC_Enabled' in l) or ('DeviceName:' in l):
                setattr( self , l.split(':')[0].replace(' ','_') , l.split(':')[1].strip() )
            else:
                if len(l.split(':')) == 2:
                    setattr( self.Params , l.split(':')[0].replace(' ','_') , l.split(':')[1].strip() )
                elif len(l.split('::')) == 2:
                    setattr( self.Params , l.split('::')[0].replace(' ','_') , l.split('::')[1].strip() )
                elif len(l.split(':')) == 1:
                    pass
                else:
                    if l.split(':')[0] not in ['Kernel Build' , ''] and not re.match( r'\s*((prerun)|(acq)|(extraG)|(beadfind_pre))_[0-9]*\.dat', l ) :
                        print( 'Explog parsing error:  Unknown case with multiple colons.  Attempting to parse:\n\t%s' % l )
                    setattr( self.Params , l.split(':')[0] , l.replace( '%s:' % l.split(':')[0] , '' ) )
            # Generic parser
            try:
                parts = l.split( ':' )
                name  = parts[0]
                value = ':'.join( parts[1:] )
            except IndexError:
                continue
            name  = name.strip()
            value = value.strip()
            if value.lower() == 'no':
                value = False
            elif value.lower() == 'yes':
                value = True
            else:
                try:
                    value = float( value )
                    if float( int( value ) ) == value:
                        value = int(value)
                except ValueError:
                    pass
            self.fields[name] = value

        # Set expected common parameters
        # Noise ( in uV! )
        if hasattr( self.Params , 'ChipNoise' ):
            self.noise = float( getattr( self.Params , 'ChipNoise' ) )
        elif hasattr( self.Params , 'Noise' ):
            self.noise   = float( getattr( self.Params , 'Noise' ) ) * 14.4
            self.noise90 = float( getattr( self.Params , 'Noise_90pct' ) ) * 14.4
            
        # Gain
        if hasattr( self.Params , 'ChipGain' ):
            self.gain = float ( getattr( self.Params , 'ChipGain' ) )
        elif hasattr( self.Params , 'Gain' ):
            self.gain = float ( getattr( self.Params , 'Gain' ) )
            
