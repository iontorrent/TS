''' 
Functions for analyzing DatFiles
'''
import os, sys, time
import warnings
import numpy as np
from . import stats, imtools, vfc
from . import chiptype as ct
from .annotate import Capturing
from .misc import flatten

moduleDir = os.path.abspath( os.path.dirname( __file__ ) )
allow_missing = True
# Suppress all divide by zero and other ridiculous warnings
np.seterr( 'ignore' )


try:
    from .di_loader import deinterlace as di
except ImportError as e:
    print( 'WARNING! Unable to import deinterlace' )
    warnings.warn(str(e),ImportWarning)

# Note: for wettest datfiles, you may want to use the DatFile class from the 
# wettest repo.  It is a subclass of DatFile, but the class variables
# (e.g. runway, normFrames, ...) are set differently

class DatFile( object ):
    # Used during __init__
    normFrames    = 10    # Number of initial frames for normalizing self.data

    # Used during measurements
    active_thresh = 500   # Minimum variance required to call a pixel "active"
    runway        = 10    # Number of ideally level frames at start of acquisition (this is used for gain and noise calculation)
    mesa          = 10    # Number of ideally level frames at end of acquisition

    noiseframewarning = False

    # Internal flags
    _vfc          = False # indicates if VFC is detected
    def __init__(self, filename=None, norm=True, fcmask=True, isbasic=None, chiptype=None, dr=400, vfc=False, normframes=None, pinned_low_cutoff=None, pinned_high_cutoff=None ):
        '''
        Generic function that will load an arbitrarily sized acquisition file.
        Returns array size details, pixel data, and pinned well information.
        
        ** If norm=True, then it will also sets all traces to begin at zero the same way TorrentExplorer 
        ** and TorrentR do.  This would be important if looking for pixel offset information.

        isbasic sets options for handling acidic or basic steps. This only acts when MakeBasic is called
            if True, assumes the step is basic
            if False, it assumes the step is acidic and negates the data during analysis (but not loading)
            if None, no assumptions are made and decisions are made on the micro-block.  This is less stable against bubbles

        dr specifies the dynamic range in mV.  Many properties are converted to mV or uV before exporting using this value
        
        Properties:
        .rows
        .cols
        .frames
        .timestamps (in milliseconds)
        .data
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if normframes is None:
            normframes = self.normFrames

        # Options store function options to ensure that the correct value is returned
        self.options    = {}

        self.norm       = norm
        self.isbasic    = isbasic
        if chiptype is None:
            chiptype = 'dummy'
        chiptype = ct.ChipType( chiptype ) # This works, even if chiptype is already a chiptype
        self.chiptype   = chiptype  
        self.uvcts      = dr*1000/float(2**14)
        self.vfcenabled = vfc
        self.dr         = dr

        self.filename   = filename

        if not os.path.exists( filename ):
            self.data = np.zeros(0)
            msg = 'File %s not found.' % filename
            if allow_missing:
                print( msg )
                return
            raise IOError( msg )

        # Read in .dat file
        with Capturing() as output: # Capture the C-output and move it to python
            acq = di.deinterlace_c ( filename )
        for o in output:
            if o.strip():
                print( o )

        if acq.data is None:
            raise IOError( "Error reading datfile %s" % filename )

        self.cols         = acq.cols
        self.rows         = acq.rows
        self.frames       = acq.frames
        self.timestamps   = acq.timestamps
        self.uncompFrames = acq.uncompFrames

        # Get the time stamps
        self.GetFrames()
        
        # Reshape data array from (frame,R,C) to (R,C,frame)
        img = np.rollaxis ( acq.data , 0 , 3 )
        img = np.array    ( img , dtype='i2' )
        self.data         = img
        
        # Add new property to the class defining pixels that are pinned.
        if pinned_high_cutoff is None:
            pinned_high_cutoff = 16370
        if pinned_low_cutoff is None:
            pinned_low_cutoff = 10
        self.pinned_high = np.any( img > pinned_high_cutoff , axis=2 )
        self.pinned_low  = np.any( img < pinned_low_cutoff,     axis=2 )
        self.pinned      = self.pinned_high | self.pinned_low

        if self.norm==True:
            # Normalize all pixels to start their signals at 0 before the flow begins.
            self.offset_norm = self.measure_offset( maxframe=normframes )
            # Added a method to broadcast background subtraction and cut 5 secs from 7.5 
            # DatFile opening time.....woohoo!!  - Phil 10/7/13
            self.data -= self.offset_norm[ : , : , np.newaxis ]

        # Determine chip type from image size
        self.ParseDatType()

        # Check if the step is acidic or basic
        if self.isbasic is None:
            self.CheckBasic()

        # Check if VFC was enabled:
        if self.data.shape[2] > 8:
            if not self.vfcenabled:
                if self.is_Thumbnail:
                    if ( self.data[:,:,0] == self.data[:,:,7] ).all():
                        if self.norm: 
                            first = self.offset_norm
                        else:
                            first = self.data[:,:,0]
                        if not ( first == 0 ).all():
                            print( 'WARNING! VFC was detected but not requested.  Settting vfcenabled=True' )
                            self.vfcenabled = True

    ##################################################
    # Analysis Functions                             #
    ##################################################
    '''
    Some of these functions generate variables which might be needed
    by other functions.  For example the slopes calculation depends
    on t0 and should save t0 to self.  Therefore, each of these functions
    should first attempt to return the corresponding attribute 
    before launching their own analysis
    These should be simple functions, producing 1 variable.  Larger functions
    should be called below
    These functions also should not require any inputs
    '''
    def measure_active_thresh( self ):
        ''' returns the active threshold for buffering calulations '''
        return self.active_thresh

    def measure_active_pixels( self ):
        ''' Measures the active pixels using the default threshold '''
        try:
            return self.actpix
        except AttributeError:
            self.CalculateActivePixels()
            return self.actpix

    def measure_avgtrace( self, micro=False, mask=None, redo=False ):
        '''
        NOTE:  mask takes in a numpy array of what you want to block
                EX: if you want to mask out empty wells (i.e. only have data from beads)
                        --> mask=empty
        '''
        try:
            if self._avgtrace_micro == micro and not redo: 
                return self.avgtrace
            else:
                raise
        except:
            return self.AvgTrace( micro=micro, mask=mask, redo=redo )

    def measure_average( self ):
        ''' 
        Takes the average of all frames to return an average frame
        The value returned is the absolute value (not normalized) (in counts)

        Scott Parker 7/17/2015
        '''
        try:
            return self.average
        except AttributeError:
            if self._vfc:
                self.average = np.average( self.data, axis=2, weights=self.vfcprofile )
            else:
                self.average = self.data.mean( axis=2 )
            return self.average

    def measure_bfgain( self ):
        ''' 
        Measures the beadfind gain, returning a result in mV/V

        Scott Parker 7/17/2015
        '''
        try:
            return self.bfgain
        except AttributeError:
            self.BeadfindGainTest()
            return self.bfgain

    def measure_bfgain_std( self, hd=False ):
        ''' 
        Measures the local standard deviation of the beadfind "gain" (in mV/V)

        Scott Parker 7/17/2015
        '''
        try:
            return self.bfgain_std
        except AttributeError:
            self.BeadfindGainTest()
            self.bfgain_std = self.LocalVar( self.bfgain, var='std', hd=hd, dname='bfgain' )
            return self.bfgain_std

    def measure_bfgain_iqr( self, hd=False, mask=None ):
        ''' 
        Measures the local standard deviation of the beadfind "gain" (in mV/V)
        If mask is undefined, it defaults to excluding the pinned mask
        Otherwise, mask must be a boolean aray matching data size

        Scott Parker 7/17/2015
        '''
        try:
            return self.bfgain_iqr
        except AttributeError:
            self.BeadfindGainTest()
            #self.bfgain_iqr = self.LocalVar( self.bfgain, var='iqr', hd=hd, dname='bfgain' )
            if mask is None:
                mask = np.logical_or( self.pinned, self.bfgain==0 )
            if hd:
                output = stats.chip_uniformity( self.bfgain, self.chiptype, block='micro', exclude=mask )
            else:
                output = stats.chip_uniformity( self.bfgain, self.chiptype, block='mini', exclude=mask )
            self.bfgain_iqr = output['blocks_iqr']
            return self.bfgain_iqr

    def measure_bf_signal( self, nndist=None ):
        ''' 
        Measures the beadfind signal (max deviation from neighbors) 

        Scott Parker 7/17/2015
        '''
        try:
            return self.bfmat
        except AttributeError:
            self.BeadFind( nndist=nndist )
            return self.bfmat

    def measure_bf_signal_std( self, nndist=None ):
        ''' 
        Measures the local variation of the beadfind signal

        Scott Parker 7/17/2015
        '''
        try:
            return self.bfmat_std
        except AttributeError:
            self.BeadFind( nndist=nndist )
            self.bfmat_std = self.LocalVar( self.bfmat, var='std', hd=hd, dname='bfmat' )
            return self.bfmat_std

    def measure_bf_signal_iqr( self, nndist=None ):
        ''' 
        Measures the local variation of the beadfind signal

        Scott Parker 7/17/2015
        '''
        try:
            return self.bfmat_iqr
        except AttributeError:
            self.BeadFind( nndist=nndist )
            #self.bfmat_iqr = self.LocalVar( self.bfmat, var='iqr', hd=hd, dname='bfmat' )
            output = stats.chip_uniformity( self.bfmat, self.chiptype, block='micro', exclude=self.pinned )
            self.bfmat_iqr = output['blocks_iqr']
            return self.bfmat_iqr

    def measure_colmeans( self, half=None, frames=slice(None) ):
        '''Measures the mean column values'''
        # Colmeans are calcualted by measure_colnoise, which does try/except and options parsing
        self.measure_colnoise( half=half, frames=frames )
        if half == 'top':
            return self.colmeans_top
        elif half == 'bottom':
            return self.colmeans_bottom
        else:
            return self.colmeans

    def measure_colmeans_top( self, frames=slice(None) ):
        return self.measure_colmeans( half='top', frames=frames )

    def measure_colmeans_bottom( self, frames=slice(None) ):
        return self.measure_colmeans( half='bottom', frames=frames )

    def measure_colmeans_runway( self, frames=None ): 
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_colmeans( frames=slice(None,frames) )

    def measure_colmeans_top_runway( self, frames=None ):
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_colmeans_top( frames=slice(None,frames) )

    def measure_colmeans_bottom_runway( self, frames=None ):
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_colmeans_bottom( frames=slice(None,frames) )

    def measure_colnoise( self, half=None, frames=slice(None) ):
        ''' 
        Measures the CORRELATED column noise in uV
        This will not work as well on spatial thumbnails because it goes across the whole chip

        Scott Parker 7/17/2015
        '''
        try:
            if self.options['measure_colnoise:frames'] == frames:
                if half == 'top':
                    return self.colnoise_top
                elif half == 'bottom':
                    return self.colnoise_bottom
                else:
                    return self.colnoise
        except:
            pass
        self.options['measure_colnoise:frames'] = frames
        colnoise, colmeans = self.RCnoise( axis='col', driftcorrect=True, frames=frames )
        if half == 'top':
            self.colnoise_top = colnoise
            self.colmeans_top = colmeans
        elif half == 'bottom':
            self.colnoise_bottom = colnoise
            self.colmeans_bottom = colmeans
        else:
            self.colnoise = colnoise
            self.colmeans = colmeans
        print( 'raw colmeans shape: %s' % ( colmeans.shape, ) )
        return colnoise

    def measure_colnoise_runway( self, frames=None ):
        ''' Measures the column correlated noise for the runway '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_colnoise( frames=slice(None, frames) )

    def measure_colnoise_bottom( self, frames=slice(None) ):
        ''' 
        Measures the CORRELATED column noise in uV
        This only works on the bottom half of the array, making it suitable for thumbnails
        '''
        return self.measure_colnoise( half='bottom', frames=frames )

    def measure_colnoise_bottom_runway( self, frames=None ):
        ''' 
        Measures the CORRELATED column noise in uV
        This only works on the bottom half of the array, making it suitable for thumbnails
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_colnoise_bottom( frames=slice(None,frames) )

    def measure_colnoise_top( self, frames=slice(None) ):
        ''' 
        Measures the CORRELATED column noise in uV
        This only works on the top half of the array, making it suitable for thumbnails

        Scott Parker 7/17/2015
        '''
        return self.measure_colnoise( half='top', frames=frames )

    def measure_colnoise_top_runway( self, frames=None ):
        ''' 
        Measures the CORRELATED column noise in uV
        This only works on the top half of the array, making it suitable for thumbnails

        Scott Parker 7/17/2015
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_colnoise_top( frames=slice(None,frames) )

    def measure_drift( self, frames=slice(None,None), bkgsubtract=False ):
        ''' 
        Calculates the signal drift across all frames (in counts)
        Fits a linear function to the data and returns the slope.

        # 6/7/2017 - Overwrote ECC measure_drift.  It did not seem to be used
        '''
        try:
            if self.options['measure_drift:frames'] == frames:
                return self.drift
            else:
                raise ValueError
        except:
            pass

        self.options['measure_drift:frames'] = frames
        vfcslice = self.VFCslice(frames)
        data = self.data[:,:,vfcslice]
        if bkgsubtract:
            region_rows = int( self.rows / self.miniR )
            region_cols = int( self.cols / self.miniC )

            self.drift = self.data.shape[:-1]

            for r in range( region_rows ):
                for c in range( region_cols ):
                    rws = slice( r*self.miniR, (r+1)*self.miniR )
                    cls = slice( c*self.miniC, (c+1)*self.miniC )

                    roi    = data[ rws, cls ]
                    badpx  = np.zeros( roi.shape[:-1], dtype=np.bool )   # All pixels are good

                    nnimg = roi - imtools.GBI( roi, badpx, 10 )

                    nnimg = nnimg.reshape( -1, nnimg.shape[-1]).T

                    x = self.inds[vfcslice]
                    A = np.vstack( [ x, np.ones(x.shape) ] ).T
                    fit = np.linalg.lstsq( A, nnimg )
                    self.drift[rws,cls] = fit[0][0].reshape( self.roi.shape[:-1] )
        else:
            data = data.reshape(-1,data.shape[-1]).T
            x = self.inds[vfcslice]
            A = np.vstack( [ x, np.ones(x.shape) ] ).T
            fit = np.linalg.lstsq( A, data )
            self.drift = fit[0][0].reshape( self.data.shape[:-1] )
        return self.drift

    def measure_drift_mesa( self, frames=None ):
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.mesa

        try:
            if self.options['measure_drift_mesa:frames'] == frames:
                return self.drift_mesa
            else:
                raise ValueError
        except:
            pass
        self.drift_mesa = self.measure_drift( frames=slice(-frames,None) )
        return self.drift_mesa

    def measure_drift_runway( self, frames=None ):
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        try:
            if self.options['measure_drift_runway:frames'] == frames:
                return self.drift_runway
            else:
                raise ValueError
        except:
            pass
        self.drift_runway = self.measure_drift( frames=slice(None,frames) )
        return self.drift_runway

    def measure_driftavg( self, frames=slice(None,None) ):
        ''' Measures a single value for drift of the entire array '''
        try:
            if self.options['measure_driftavg:frames'] == frames:
                return self.driftavg
            else:
                raise ValueError
        except:
            pass

        self.options['measure_driftavg:frames'] = frames
        vfcslice = self.VFCslice(frames)
        data = self.data[:,:,vfcslice]
        data = data.reshape(-1,data.shape[-1])
        data = data[~self.pinned.flatten().astype(bool),:]
        data = data.mean( axis=0 )
        x = self.inds[vfcslice]
        A = np.vstack( [ x, np.ones(x.shape) ] ).T
        fit = np.linalg.lstsq( A, data )
        self.driftavg = fit[0][0]
        return self.driftavg

    def measure_driftavg_mesa( self, frames=None ):
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.mesa

        try:
            if self.options['measure_driftavg_mesa:frames'] == frames:
                return self.driftavg_mesa
            else:
                raise ValueError
        except:
            pass
        self.driftavg_mesa = self.measure_driftavg( frames=slice(-frames,None) )
        return self.driftavg_mesa

    def measure_driftavg_runway( self, frames=None ):
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        try:
            if self.options['measure_driftavg_runway:frames'] == frames:
                return self.driftavg_runway
            else:
                raise ValueError
        except:
            pass
        self.driftavg_runway = self.measure_driftavg( frames=slice(None,frames) )
        return self.driftavg_runway

    def measure_drift_iqr( self, frames=slice(None,None), bkgsubtract=False, mask=None ):
        ''' 
        Measures the local IQR of the drift (in DN14/frame)
        If mask is undefined, it defaults to excluding the pinned mask
        Otherwise, mask must be a boolean aray matching data size

        Scott Parker 7/17/2015
        '''
        try:
            return self.drift_iqr
        except AttributeError:
            self.measure_drift( frames=frames, bkgsubtract=bkgsubtract )
            #self.bfgain_iqr = self.LocalVar( self.bfgain, var='iqr', hd=hd, dname='bfgain' )
            if mask is None:
            #    mask = np.logical_or( self.pinned, self.drift_iqr==0 )
                mask = self.pinned
            output = stats.chip_uniformity( self.drift, self.chiptype, block='mini', exclude=mask )
            self.drift_iqr = output['blocks_iqr']
            return self.drift_iqr

    def measure_gain_counts( self, stepsize=None ):
        ''' Measures the drift-corrected gain in counts.  If a step size (in mV) is provided, the gain will be returned. Units are always mV/V '''
        drift_mesa   = self.measure_driftavg_mesa()
        drift_runway = self.measure_driftavg_runway()
        runway_t = self.inds[ self.VFCslice( slice( None, self.runway ) ) ].mean()
        mesa_t   = self.inds[ self.VFCslice( slice( -self.mesa,  None ) ) ].mean()
        avgdrift = np.mean( ( drift_mesa, drift_runway ) )
        deltat = mesa_t - runway_t
        correct = avgdrift*deltat

        gain_diff = -(self.measure_plateau() - correct)
        if stepsize:
            return gain_diff * ( self.uvcts ) / ( stepsize )  # uV/mV OR mV/V
        return gain_diff

    def measure_height( self ):
        ''' 
        Measures the height of the nuc step (in counts )
        Right now, this works using a simple max/min algorithm
        This is the old GetScales function, generalized to the full chip

        Scott Parker 7/17/2015
        '''
        try:
            return self.height
        except AttributeError:
            ''' Gets the step height for the image per well '''
            # TODO: Need to make this work for non-normalized data
            if (self.isbasic is None) or (not self.norm):
                # This shouldn't normally happen, but let's allow the contingency and make it positive
                height = np.max( self.data, axis=2 ) - np.min( self.data, axis=2 )
            elif self.isbasic:
                height = np.min( self.data, axis=2 )
            elif not self.isbasic:
                height = np.max( self.data, axis=2 )
            self.height = height

            return self.height

    def measure_mesa( self ):
        ''' Returns the frame ID assumed for the mesa '''
        return self.mesa

    def measure_noise( self, frames=slice(None, None), bkgsubtract=False ):
        ''' 
        This measures the drift-corrected noise (in uV)
        Algorithm is:
        rms = sqrt( sum_i( ( ( F_(i+1) - F_i ) / 2 )^2 ) / frames )
        Frames is specified as a slice

        setting bkgsubtract=True does a regional background (drift) subtraction before
        calculating the drft-corrected noise

        Scott Parker 7/17/2015
        '''
        try:
            if ( self.options['measure_noise:frames'] == frames and
                 self.options['measure_noise:bkgsubtract'] == bkgsubtract ):
                return self.noise
            else:
                raise ValueError
        except:
            self.options['measure_noise:frames'] = frames
            self.options['measure_noise:bkgsubtract'] = bkgsubtract
            if self._vfc or self.vfcenabled:
                # Don't have to worry about doing background subtraction here, 
                # even when GetUncompressedFrames( common=False ) because
                # VFC differences occur at block boundaries and an integer number
                # of ROIs fit into each block
                data = self.GetUncompressedFrames( frames=frames )
            else:
                data = self.data[:,:,frames]

            if bkgsubtract:
                region_rows = int( self.rows / self.miniR )
                region_cols = int( self.cols / self.miniC )

                self.noise = np.zeros( self.data.shape[:-1] )
                for r in range( region_rows ):
                    for c in range( region_cols ):
                        rws = slice( r*self.miniR, (r+1)*self.miniR )
                        cls = slice( c*self.miniC, (c+1)*self.miniC )

                        roi    = data[ rws, cls ]
                        badpx  = np.zeros( roi.shape[:-1], dtype=np.bool )   # All pixels are good

                        nnimg = roi - imtools.GBI( roi, badpx, 10 )

                        self.noise[rws,cls] = self.DriftFreeNoise( nnimg )
            else:
                self.noise  = self.DriftFreeNoise( data )

            self.noise *= self.uvcts
            return self.noise

    def measure_noise_runway( self, frames=None ):
        ''' Calculates the noise (drift corrected) for the runway portion of the trace '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_noise( frames=slice( None, frames ) )

    def measure_noiseavg( self, frames=slice(None,None) ):
        ''' Measures the average noise of the entire array, assuming time and space averaging are equivilant '''
        try:
            if self.options['measure_noiseavg:frames'] == frames:
                return self.noise
        except:
            pass
        self.options['measure_noiseavg:frames'] = frames
        if self._vfc or self.vfcenabled:
            data = self.GetUncompressedFrames( frames=frames )
        else:
            data = self.data[:,:,frames]

    def measure_norm( self ):
        ''' Returns the normalization offset factor for the raw data '''
        try:
            return self.offset_norm
        except AttributeError:
            return np.zeros( self.data.shape[:2] )

    def measure_num_uncomp( self, frames=slice(None) ):
        ''' Returns the number of uncompressed frames '''
        try:
            if self.options['measure_num_uncomp'] == frames:
                return self.num_uncompressed_frames
        except:
            pass
        self.options['measure_num_uncomp'] = frames
        self.num_uncompressed_frames = self.GetUncompressedFrames( frames ).shape[-1]
        return self.num_uncompressed_frames

    def measure_num_uncomp_runway( self, frames=None ):
        ''' Returns the number of uncompressed frames in the runway '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_num_uncomp( frames=slice(None,frames) )

    def measure_offset( self, maxframe=None ):
        ''' 
        Measures the pixel offsets by the average of the first N frames, correcting for VFC
        offsets are in counts

        Scott Parker 7/17/2015
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if maxframe is None:
            maxframe = self.normFrames

        try:
            if self.options['measure_offset:maxframes'] == maxframe:
                return self.bkg_offsets
            else:
                raise AttributeError 
        except:
            self.options['measure_offset:maxframes'] = maxframe
            bkg = np.mean ( self.data[:,:,self.inds<maxframe] , axis=2 ) + self.measure_norm()
            bkg = np.array( bkg , dtype='i2' ) 
            self.bkg_offsets = bkg
            return self.bkg_offsets

    def measure_offset_iqr( self, maxframe=None, mask=None ):
        ''' 
        Measures the local IQR of the drift (in DN14/frame)
        If mask is undefined, it defaults to excluding the pinned mask
        Otherwise, mask must be a boolean aray matching data size

        Scott Parker 7/17/2015
        '''
        try:
            return self.offset_iqr
        except AttributeError:
            offsets = self.measure_offset( maxframe=maxframe )
            if mask is None:
                mask = self.pinned
            output = stats.chip_uniformity( offsets, self.chiptype, block='mini', exclude=mask )
            self.offset_iqr = output['blocks_iqr']
            return self.offset_iqr

    def measure_pHpoint( self ):
        ''' 
        Returns the equilibrium pH value in absolute counts
        This is currently very lazy because it just takes the last frame

        Scott Parker 7/17/2015
        '''
        try:
            return self.pHpoint
        except AttributeError:
            last = self.data[:,:,-1]
            self.pHpoint = last + self.measure_norm()
            return self.pHpoint

    def measure_plateau( self, frames=None ):
        ''' Returns the plateau value (average of last 10 frames ) '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.mesa

        try:
            if self.options['measure_plateau:frames'] == frames:
                return self.plateau
            else:
                raise AttributeError
        except:
            self.options['measure_plateau:frames'] = frames
            self.plateau = self.data[:,:,-frames:].mean(axis=2)
            return self.plateau

    def measure_pinned( self ):
        ''' 
        returns the pined pixels 
        this is always calculated in init so no need to try/except
        '''
        return self.pinned

    def measure_pinned_high( self ):
        ''' 
        returns the high pined pixels 
        this is always calculated in init so no need to try/except
        '''
        return self.pinned_high

    def measure_pinned_low( self ):
        ''' 
        returns the low pined pixels 
        this is always calculated in init so no need to try/except
        '''
        return self.pinned_low

    def measure_pinned_runway( self, frames=None):
        ''' 
        returns the pined pixel mask from the first specified frames
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        try:
            if self.options['measure_pinned_runway:frames'] == frames:
                return self.pinned_runway
            else:
                raise AttributeError
        except:
            self.options['measure_pinned_runway:frames'] = frames
            data = self.data + self.measure_norm()[:,:,np.newaxis]
            self.pinned_runway_high = np.any( data[:,:,:frames] > 16370 , axis=2 ) 
            self.pinned_runway_low  = np.any( data[:,:,:frames] < 10 , axis=2 ) 
            self.pinned_runway = self.pinned_runway_high | self.pinned_runway_low
            return self.pinned_runway

    def measure_pinned_runway_high( self, frames=None ):
        ''' 
        returns the high pined pixel mask from the first specified frames
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        try:
            return self.pinned_runway_high
        except:
            self.measure_pinned_runway( frames )
            return self.pinned_runway_high

    def measure_pinned_runway_low( self, frames=None ):
        ''' 
        returns the low pined pixel mask from the first specified frames
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        try:
            return self.pinned_runway_low
        except:
            self.measure_pinned_runway( frames )
            return self.pinned_runway_low

    def measure_raw_noise( self, frames=slice( None, None ), bkgsubtract=False ):
        ''' 
        Measures the raw (not drift corrected) noise (in uV)
        This is just a standard deviation

        Scott Parker 7/17/2015
        '''
        try:
            if ( self.options['measure_raw_noise:frames'] == frames and
                 self.options['measure_raw_noise:bkgsubtract'] == bkgsubtract ):
                return self.raw_noise
        except:
            pass
        self.options['measure_raw_noise:frames'] = frames
        self.options['measure_raw_noise:bkgsubtract'] = bkgsubtract
        if self._vfc or self.vfcenabled:
            # Don't have to worry about doing background subtraction here, 
            # even when GetUncompressedFrames( common=False ) because
            # VFC differences occur at block boundaries and an integer number
            # of ROIs fit into each block
            data = self.GetUncompressedFrames( frames=frames )
        else:
            data = self.data[:,:,frames]

        if bkgsubtract:
            region_rows = int( self.rows / self.miniR )
            region_cols = int( self.cols / self.miniC )

            self.raw_noise = np.zeros( self.data.shape[:-1] )
            for r in range( region_rows ):
                for c in range( region_cols ):
                    rws = slice( r*self.miniR, (r+1)*self.miniR )
                    cls = slice( c*self.miniC, (c+1)*self.miniC )

                    roi    = data[ rws, cls ]
                    badpx  = np.zeros( roi.shape[:-1], dtype=np.bool )   # All pixels are good

                    nnimg = roi - imtools.GBI( roi, badpx, 10 )

                    self.raw_noise[rws,cls] = nnimg.std( axis=2 )
        else:
            self.raw_noise = data.std( axis=2 )
        # Convert to uV
        self.raw_noise *= self.uvcts
        return self.raw_noise

    def measure_raw_noise_runway( self, frames=None ):
        ''' 
        Calculates the raw noise for the runway portion of the trace
        '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_raw_noise( frames=slice( None, frames ) )

    def measure_rowmeans( self, frames=slice(None) ):
        ''' 
        Measures the CORRELATED row noise (in uV)

        Scott Parker 7/17/2015
        '''
        self.measure_rownoise( frames=frames )
        return self.rowmeans

    def measure_rowmeans_runway( self, frames=None ):
        ''''''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        return self.measure_rowmeans( frames=slice(None, frames ) )

    def measure_rownoise( self, frames=slice(None) ):
        ''' Measures the CORRELATED row noise (in uV) '''
        try:
            if self.options['measure_rownoise:frames'] == frames:
                return self.rownoise
        except:
            pass
        self.rownoise, self.rowmeans = self.RCnoise( axis='row', driftcorrect=True, frames=frames )
        return self.rownoise

    def measure_rownoise_runway( self, frames=None ):
        ''' Measures the correlated row noise (in uV) for the runway '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if frames is None:
            frames = self.runway

        try:
            if self.options['measure_rownoise_runway:frames'] == frames:
                return self.rownoise_runway
        except:
            pass
        self.rownoise_runway = self.measure_rownoise( frames = slice( None, frames ) )
        return self.rownoise_runway

    def measure_runway( self ):
        ''' Returns the frame ID for the runway '''
        return self.runway

    def measure_runway_mesa_delta( self, rw=runway, ms=mesa ):
        ''' Returns the number of frames between the runway and the mesa '''
        # Apply defaults.  I have to do it this way because defining in the function definition
        # does not allow these values to be changed when subclassing
        if rw is None:
            rw = self.runway
        if ms is None:
            ms = self.mesa

        runway_t = self.inds[ self.VFCslice( slice( None,  rw ) ) ].mean()
        mesa_t   = self.inds[ self.VFCslice( slice( -ms, None ) ) ].mean()
        deltat = mesa_t - runway_t
        return deltat

    def measure_slope( self, method='avgslope' ):
        ''' 
        Measures the maximum slope for each nuc trace.
        This uses the BufferTest function to generate slopes
        Units are uV/second

        Scott Parker 7/17/2015
        '''
        param = 'slopes_%s' % method.lower()
        try:
            return getattr( self, param )
        except AttributeError:
            self.BufferTest( method=method )
            return getattr( self, param )

    def measure_slope_avgframes( self, method='avgslope' ):
        ''' 
        Measures the number of frames used to calculate the slope by BufferTest
        Units are frames

        Scott Parker 12/16/2015
        '''
        param = 'slope_avgframes_%s' % method.lower()
        try:
            return getattr( self, param )
        except AttributeError:
            self.BufferTest( method=method )
            return getattr( self, param )

    def measure_t0( self, method='avgslope' ):
        ''' 
        Measures the maximum slope for a typical nuc trace
        This uses the BufferTest function to generate t0
        units are frames

        Scott Parker 7/17/2015
        '''
        param = 't0_%s' % method.lower()
        try:
            return getattr( self, param )
        except AttributeError:
            self.BufferTest( method=method )
            return getattr( self, param )

    ##################################################
    # (Empty) Bead find                              #
    ##################################################
    def BeadFind( self, active_thresh=None, nndist=None ):
        '''
        Generic beadfind algorithm adapted from Todd Rearick's Thumbnail beadfind matlab script
        This calculates the maximum deviation of the well from the neighbor-subtracted background
        
        saves:
            self.bfmat (units are uV)
        '''
        try:
            return { 'bfmat' : self.bfmat , 
                     'time'  : 0 }
        except AttributeError:
            pass

        # Start timing algorithm
        start_time = time.time()

        if nndist is None:
            nndist = 10

        # Pull the global threshold if not provided
        if active_thresh is None:
            active_thresh = self.active_thresh
        
        # Find active pixels (to self.actpix)
        self.CalculateActivePixels( active_thresh=active_thresh )

        # Calculate BeadfindGainTest for the entire array (to set self.bfgain, self.bfgain_t0, self.bfgain_t1)
        self.BeadfindGainTest()

        bfmat  = np.zeros(self.pinned.shape)
        bfgain = np.zeros(self.pinned.shape)
        
        rows = int( self.rows / self.miniR )
        cols = int( self.cols / self.miniC )

        nogoodpix_count = 0
        badpixel_count  = 0
        for r in range(rows):
            for c in range(cols):
                rws = slice( r*self.miniR, (r+1)*self.miniR )
                cls = slice( c*self.miniC, (c+1)*self.miniC )
                goodpx = self.actpix[ rws, cls ]
                
                # For cleanliness, define the ROI for this tn block here
                #NOTE change type to float64 for future scaling operations
                roi    = self.data[ rws, cls, : ].astype( np.float64 )
                
                # Account for the far reaches of the chip that have wacky beadfind signals:
                if ( self.bfgain_t1[r,c] - self.bfgain_t0[r,c] ) <= 0 or goodpx.sum() < np.ceil( 0.05 * self.miniR * self.miniC ):
                    goodpx = np.zeros( goodpx.shape , dtype=bool )
                else:
                    # Trim ROI to minimize frames needed for calc assuming bf signal comes between (t0-5) and t1
                    roi = roi[ : , : , ( self.bfgain_t0[r,c]-5 ) : self.bfgain_t1[r,c] ]
                    
                if not goodpx.any():
                    nogoodpix_count += 1
                else:
                    gain = self.bfgain[rws,cls].astype( np.float64 )/1000.
                    roi = roi.astype( np.float64) / gain[:,:,np.newaxis]
                    #nn_gain = 1.
                    #NOTE GBI requires int16 type as input for appropriate behavior
                    nnimg = roi - imtools.GBI( roi.astype( np.int16 ) , ~goodpx , nndist , nn_gain=1 ).astype( np.float64 )     
                    nnimg[ np.isnan(nnimg) ] = 0
                    nnimg[ ~goodpx         ] = 0
                        
                    # blank anything with an excessive standard deviation
                    s3 = np.std ( nnimg , axis = 2 , ddof = 1 )
                    oddpixelthresh = stats.scoreatpercentile(s3.flatten(),99)
                    if ( oddpixelthresh > 2 * np.mean( s3 ) ):
                        oddpixels = s3 >= oddpixelthresh;
                        goodpx = goodpx & ~oddpixels
                        nnimg[ oddpixels ] = 0

                    # Note to self -- eventually need to add a new if goodpx.sum() < 500 here
                    try:
                        vspread = np.std ( nnimg.reshape( -1 , nnimg.shape[2] ) , axis = 0 , ddof = 1 )
                    except ( ValueError ):
                        badpixel_count += 1
                    else:
                        maxdif  = np.argmax ( vspread )
                        bfval   = nnimg[ : , : , maxdif ]
                        bfval[ np.isnan(bfval) ] = 0
                        
                        # Output gain correction and beadfind value matrices for further analysis
                        bfmat  [ rws , cls ] = goodpx * bfval

        if nogoodpix_count:
            print(  '...No good pixels were found in %i block%s and were skipped.' % ( nogoodpix_count, 's' if nogoodpix_count > 1 else '' ) )
        if badpixel_count:
            print( '...Error in %i block%s due to bad pixels and were skipped' % ( badpixel_count, 's' if badpixel_count > 1 else '' ) )

        exectime = (time.time() - start_time)
        # Convert units 
        bfmat *= self.uvcts
        # Save results
        self.bfmat = bfmat
        return { 'bfmat' : bfmat , 'time' : exectime }
            
    ##################################################
    # Buffer test (slopes and t0)                    #
    ##################################################
    def BufferTest(self, method='MaxSlope', active_thresh=None ):
        """
        This function is analogous to the BeadFind function but will calcualte buffering data.
        The only input needed is the acquisition metafile.
        slopes are in units of uV/second
        """
        try:
            return { 'slopes'    : getattr( self, 'slopes_%s' % method.lower() ) , 
                     't0'        : getattr( self, 't0_%s'     % method.lower() ) , 
                     'time'      : 0, 
                     'avgframes' : getattr( self, 'slope_avgframes_%s' % method.lower() ) }
        except AttributeError:
            pass

        # Start timing algorithm
        start_time = time.time()

        # Pull the global threshold if not provided
        if active_thresh is None:
            active_thresh = self.active_thresh
        
        # Find active pixels (to self.actpix)
        self.CalculateActivePixels( active_thresh=active_thresh )

        
        if self.miniR == 0 and self.miniC == 0:
            raise ValueError( 'Error! dat file is of unexpected size' )

        # Calculate the number of miniblocks
        rows = int( self.rows / self.miniR )
        cols = int( self.cols / self.miniC )
        
        # Initialize data arrays
        slopes    = np.zeros(self.pinned.shape)  # This has 1 value per well
        if method.lower() == 'maxsloperaw':
            tzeros   = np.zeros(self.pinned.shape)  # This has 1 value per well
        else:
            tzeros   = np.zeros (( rows, cols ))    # This has 1 value per miniblock
        if method.lower() == 'avgslope':
            avgframes = np.zeros (( rows, cols ))   # This has 1 value per miniblock
        elif method.lower() == 'maxslopeavg':
            avgframes = np.array( [[3]] )    #  kind of
        else:
            avgframes = np.array( [[1]] )    

        # Analyze each miniblock
        start = time.time()
        for r in range(rows):
            rws = slice( r*self.miniR, (r+1)*self.miniR )
            for c in range(cols):
                cls = slice( c*self.miniC, (c+1)*self.miniC )
                
                # These are assumed to be good pixels (i.e. NOT pinned)
                goodpx = self.actpix[ rws, cls ]
                # For cleanliness, define the ROI for this block here
                roi    = self.data[ rws, cls, : ]

                if goodpx.any():
                    # Recall, BeadfindSlope asks for pinned, not goodpx...
                    if method.lower() == 'threshold':
                        bufferdata = self.BeadfindSlope_Threshold(roi, ~goodpx, self.chiptype.startframe )
                    elif method.lower() == 'avgslope':
                        bufferdata = self.BeadfindSlope_AvgSlope(roi, ~goodpx, self.chiptype.startframe )
                        avgframes[ r, c ] = bufferdata['avgframes']
                    elif method.lower() == 'maxslopeavg':
                        bufferdata = self.BeadfindSlope_MaxSlope(roi, ~goodpx, self.chiptype.startframe, avg=True)
                    elif method.lower() == 'maxslope':
                        bufferdata = self.BeadfindSlope_MaxSlope(roi, ~goodpx, self.chiptype.startframe, avg=False )
                    elif method.lower() == 'maxsloperaw':
                        bufferdata = self.BeadfindSlope_MaxSlopeRaw(roi, ~goodpx )
                    else:
                        raise ValueError('Unknown beadfind method specified. (%s)' % method.lower())
                    slopes [ rws , cls ] = bufferdata['slopes']
                    if method.lower() == 'maxsloperaw':
                        tzeros [ rws, cls ] = bufferdata['t0']
                    else:
                        tzeros [ r,c ] = bufferdata['t0']

        # Convert units
        slopes *= self.uvcts     # DN14/frame -> uV/frame
        slopes *= self.framerate # uV/sec

        # mask pixels
        slopes[ ~self.actpix ] = np.nan
        exectime = (time.time() - start_time)

        # Save results
        setattr( self, 'slopes_%s'          % method.lower(), slopes )
        setattr( self, 't0_%s'              % method.lower(), tzeros )
        setattr( self, 'slope_avgframes_%s' % method.lower(), avgframes )
        buffertest = { 'slopes' : slopes , 't0' : tzeros , 'time' : exectime  }
        buffertest = { 'slopes' : slopes , 
                       't0' : tzeros , 
                       'time' : exectime, 
                       'avgframes': avgframes  }
        return buffertest

    def BeadfindSlope_AvgSlope(self, image , pinned , startframe=15 ):
        '''
        This is where the heavy lifting of BufferTest is performed.  
        Slopes are in units of counts/frame
        Slopes are called based on the MaxSlope method

        Algorithm to fit a linear slope to a given trace, using simple linear regression.
        The algorithm detects whether the step is acidic or basic and then converts the data to a basic step
        
        startframe input added to support 3-series data compression that starts much earlier that frame 15.
        method specifies which way to calculate the slope.  
        '''
        ### TODO:When VFC is on, it's necessary to check the averaged frames are not compressed higher than 2 frames. 
        ### So far with Datacollect 3527(vfc3.txt), we don't need to worried about it on 550 chips.

        # Calculate average trace
        avgtrace = stats.masked_avg( image , pinned )
        avgtrace = self.MakeBasic(avgtrace)

        # Define search parameter to pull out linear part of the beadfind curves
        init_value = avgtrace[0]
        min_value = np.min(avgtrace)
        threshold = (min_value+init_value)/4*3
        # Get the allowed window
        allowed = avgtrace>threshold
        #allowed[:startframe] = 0   # changed to allow for VFC
        allowed[ self.inds<startframe ] = 0

        # Calculate the first derivative
        d_avgtrace = np.diff(avgtrace)
        d_avgtrace /= np.diff( self.inds ).astype(np.float)
        # Get the maximum value the fast way
        t0_ind = d_avgtrace.argmin()
        t0 = self.inds[t0_ind]
        # Make sure value is allowed.  If not, take the slower route and iterate over the array
        if not allowed[t0_ind]:
            t0 = 0
            min_value = np.Infinity
            for ind in range(len(d_avgtrace)):
                if allowed[ind+1]:
                    if d_avgtrace[ind] < min_value:
                        t0_ind = ind+1
                        t0 = self.inds[t0_ind]
                        min_value = d_avgtrace[ind]
        
        h1 = ( avgtrace.max() - avgtrace.min() )
        # Need to be insensitive to dynamic range.  Average Points is probably driven by uV noise so let's convert to that
        huv = h1 * self.uvcts
        # TODO: This is more hardcoded that I would like
        if self.chiptype.name == '550':
            avgpoints = (1-20)/(22000.-1800.)*(huv-22000)+1
        else:
            avgpoints = (1-20)/(31300.-5200.)*(huv-31300)+1
        avgpoints = max( 1, avgpoints )
        avgpoints = min( avgpoints, 20 )
        avgpoints = int(avgpoints)

        if t0 != 0:
            first = t0_ind-avgpoints/2
            while ~allowed[first]:
                first += 1
            last = first + avgpoints + 1
            if last < image.shape[2]:
                slopes  = image[:,:,last] - image[:,:,first]
                slopes /= self.inds[last] - self.inds[first]
                slopes *= ~pinned
            else:
                slopes  = np.zeros( pinned.shape )
        else:
            slopes  = np.zeros( pinned.shape )
        
        # Save variables
        return { 'slopes' : slopes , 't0' : t0, 'avgframes' : avgpoints }

    def BeadfindSlope_MaxSlope(self, image , pinned , startframe=15, avg=False):
        '''
        This is where the heavy lifting of BufferTest is performed.  
        Slopes are in units of counts/frame
        Slopes are called based on the MaxSlope method

        Algorithm to fit a linear slope to a given trace, using simple linear regression.
        The algorithm detects whether the step is acidic or basic and then converts the data to a basic step
        
        startframe input added to support 3-series data compression that starts much earlier that frame 15.
        method specifies which way to calculate the slope.  

        Specifying avg does some time-averaging around the max slope
               t-1:  1/6
               t:    1/2
               t+1:  1/3
        Emperically, this gives similar slope values to the Threshold method but the t0 pooints will probably be later
        '''
        # Calculate average trace
        avgtrace = stats.masked_avg( image , pinned )
        avgtrace = self.MakeBasic(avgtrace)

        # Define search parameter to pull out linear part of the beadfind curves
        init_value = avgtrace[0]
        min_value = np.min(avgtrace)
        threshold = (min_value+init_value)/4*3
        # Get the allowed window
        allowed = avgtrace>threshold
        #allowed[:startframe] = 0   # changed to allow for VFC
        allowed[ self.inds<startframe ] = 0

        # Calculate the first derivative
        d_avgtrace = np.diff(avgtrace)
        d_avgtrace /= np.diff( self.inds ).astype(np.float)
        # Get the maximum value the fast way
        t0_ind = d_avgtrace.argmin()
        t0 = self.inds[t0_ind]
        # Make sure value is allowed.  If not, take the slower route and iterate over the array
        if not allowed[t0_ind]:
            t0 = 0
            min_value = np.Infinity
            for ind in range(len(d_avgtrace)):
                if allowed[ind+1]:
                    if d_avgtrace[ind] < min_value:
                        t0_ind = ind+1
                        t0 = self.inds[t0_ind]
                        min_value = d_avgtrace[ind]
        
        if (t0 != 0 and t0 < len(avgtrace)-1):
            trimmed = image[:,:,t0_ind-1:t0_ind+3]
            all_slopes = np.diff(trimmed,axis=2)
            # Correct for VFC
            all_slopes /= np.diff(self.inds[t0_ind-1:t0_ind+3])[None,None,:]
            if avg:
                # This weights the slopes at t0-1:t0:t0+1 with a 1:3:2 ratio
                slopes = (all_slopes[:,:,0]/6 + 
                          all_slopes[:,:,1]/2 + 
                          all_slopes[:,:,2]/3) * ~pinned
            else:
                slopes = all_slopes[:,:,1] * ~pinned
        else:
            slopes  = np.zeros( pinned.shape )
        
        # Save variables
        return { 'slopes' : slopes , 't0' : t0 }

    def BeadfindSlope_MaxSlopeRaw(self, image , pinned , startframe=15 ):
        '''
        This is where the heavy lifting of BufferTest is performed.  
        Slopes are in units of counts/frame
        Slopes are called based on the MaxSlope method on an individual pixel level

        Since analysis is done on the pixel level, there is no reason do do this on the block level except
        to preserve code compatibility with the other methods
        
        startframe input added to support 3-series data compression that starts much earlier that frame 15.

        returns slopes in DN14/frame
        '''
        # Calculate the first derivative
        d_trace = np.diff( image, axis=2 )
        d_trace = self.MakeBasic( d_trace )
        d_trace /= np.diff( self.inds ).astype( np.float )

        # Get the maximum value the fast way
        t0_ind = d_trace.argmin( axis=2 )
        t0 = self.inds[t0_ind]
        t0 *= ~pinned

        # Get the slopes
        slopes = d_trace.min( axis=2 )
        slopes *= ~pinned

        # Save variables
        return { 'slopes' : slopes , 't0' : t0 }

    def BeadfindSlope_Threshold(self, image , pinned , startframe=15):
        '''
        This is the OLD definition which underestimates the slope

        Algorithm to fit a linear slope to a given trace, using simple linear regression.
        The algorithm detects whether the step is acidic or basic and then converts the data to a basic step
        
        startframe input added to support 3-series data compression that starts much earlier that frame 15.
        method specifies which way to calculate the slope.  
        '''
        # Calculate average trace
        avgtrace = stats.masked_avg( image , pinned )
        avgtrace = self.MakeBasic(avgtrace)

        # Define search parameter to pull out linear part of the beadfind curves
        first = 0.05 * np.min(avgtrace)
        
        f = image.shape[2]-2    
        
        t0 = 0
        t1 = 0
        # Start looking after startframe
        #i = startframe
        i = (np.where( self.inds>=startframe ) )[0][0]    # this fixes time warp 
        while i < f and t0 == 0:       # TODO: Should this be different for higher framerates?
        # Check for steep edge of signal
            if avgtrace[i] < first:
                t0 = i
                #t0 = self.inds[i] #TODO: this fixes timewarp.  Thre need to be more changes below!
                # Check for super steep beadfinds.  Some go from zero to -6000 in a frame...
                if avgtrace[i] < (5 * first):
                    t0 = i - 1
                t1 = t0 + 2
            i = i + 1
            if i == f:
                break
            
        if (t0 != 0 and t1 != 0 ):
            trimmed = image[ : , : , t0:(t1 + 1) ]
            x       = np.hstack([ np.arange( t0 , t1 + 1 ).reshape(-1,1) , np.array(( 1 , 1 , 1 )).reshape(-1,1) ])
            fits    = np.linalg.lstsq( x , np.transpose(trimmed.reshape(-1,trimmed.shape[2])))[0]
            slopes  = fits[0].reshape( pinned.shape ) * ~pinned
        else:
            slopes  = np.zeros( pinned.shape )

        # Save variables
        return { 'slopes' : slopes , 't0' : self.inds[t0] , 't1' : self.inds[t1] }

    ##################################################
    # Beadfind Gain Calculations                     #
    ##################################################
    def BeadfindGainTest( self ):
        ''' 
        This used to be CalculateBeadfindGainNorm
        Calculate BeadfindGain on miniblocks for the entire chip 
        Saves 
           self.bfgain    (mV/V)
           self.bfgain_t0 (frames)
           self.bfgain_t1 (frames)
           self.gaincorr  (mV/V; this is a duplicate of bfgain)
        If these variables already exist, BeadfindGainNorm is not recalculated
        '''
        try:
            return ( self.bfgain, self.bfgain, self.bfgain_t0, self.bfgain_t1 )
        except AttributeError:
            pass

        # Get the active pixels using the default.  This should already be calculated and is
        # just to make sure the necessary variables exist
        self.CalculateActivePixels()

        # Get the size of the block-averaged data
        rows = int( self.rows / self.miniR )
        cols = int( self.cols / self.miniC )

        # Initialize arrays
        self.bfgain    = np.zeros( (self.rows, self.cols ) )
        self.bfgain_t0 = np.zeros( (rows, cols ), dtype=np.int )
        self.bfgain_t1 = np.zeros( (rows, cols ), dtype=np.int )

        # Loop over miniblocks
        for r in range(rows):
            rws = slice( r*self.miniR, (r+1)*self.miniR )
            for c in range(cols):
                cls = slice( c*self.miniC, (c+1)*self.miniC )
                goodpx = self.actpix[ rws, cls ]
                roi    = self.data[ rws, cls, : ]

                bfg    = self.BeadfindGain( roi , ~goodpx )
                self.bfgain[ rws, cls ] = bfg['bfgain']
                self.bfgain_t0[ r, c]  = bfg['t0']
                self.bfgain_t1[ r, c]  = bfg['t1']

        # Convert units and assign to "familiar" variables
        self.bfgain *= self.actpix  # Mask pixels 
        self.bfgain *= 1000         # Convert from uV to mV
        # Should set this for compatibility
        self.gaincorr = self.bfgain

    def BeadfindGain( self, image=None, pinned=None):
        '''
        This does the heavy lifting of BeadfindGainTest, working on each individual miniblock
        This was largley derived from BeadfindGainNorm
        Algorithm to normalize beadfind data to average trace, returning beadfind "gain" 
        If you don't specify image and pinned, it runs on the full data block

        bfGain is the slope of the best fit line of the equilibrium values between 
        the average trace and the individual wells, 
        '''
        if image is None:
            image = self.data
        if pinned is None:
            pinned = self.pinned

        # Calculate average trace
        avgtrace = stats.masked_avg( image , pinned )
        avgtrace = self.MakeBasic( avgtrace )

        # Define search parameters to trim out beadfind-relative (steep) part of the curves
        first = 0.05 * np.min(avgtrace)
        last  = 0.95 * np.min(avgtrace)
        
        f = image.shape[2]
        
        # Find t0 and t1 which define the edges of the flat portions at the start and end of the trace
        t0 = 0
        t1 = 0
        for i in range(f):
            # Check for steep edge of signal
            if ( avgtrace[i] < first and t0 == 0 ):
                t0 = i
            
            # Check for nearly leveled out signal
            if ( avgtrace[i] < last  and t1 == 0 ):
                    t1 = i
                
            # Calculate the gain
            if (t0 != 0 and t1 != 0 ):
                    trimmed = np.dstack(( image[ : , : , 0:t0 ] , image[ : , : , t1:(f - 1)] ))
                    avgtrim = np.concatenate(( avgtrace[  0:t0 ] , avgtrace[ t1:(f - 1)] )).reshape(-1,1)
                    fits    = np.linalg.lstsq( avgtrim.reshape(-1,1) , np.transpose(trimmed.reshape(-1,trimmed.shape[2])))
                    bfgain  = fits[0].reshape( pinned.shape ) * ~pinned
                    break
            else:
                    bfgain  = np.ones( pinned.shape ) * ~pinned

        gain = { 'bfgain' : bfgain , 't0' : t0 , 't1' : t1 }
        return gain

    ##################################################
    # Active pixels                                  #
    ##################################################
    def CalculateActivePixels( self, active_thresh=None):
        ''' 
        Determines the active pixels in self.data, saving to self.actpix
        Active pixels are determined as those whose standard devaiation
        are > the specified threshold

        if self.actpix exists, this calculation is not repeated
        This should save about 5.5 seconds per subsequent call on a P2 block
        '''
        # Pull the global threshold if not provided
        if active_thresh is None:
            active_thresh = self.active_thresh

        # Check if actpix was already calculated
        try:
            # Make sure the same active_thresh was used for the last calculation
            if self._active_thresh == active_thresh:
                return self.actpix
        except AttributeError:
            pass

        self._active_thresh = active_thresh
        simg   = np.std( self.data, axis = 2 , ddof = 1)
        self.actpix = simg > self._active_thresh 
        pins   = np.array( self.pinned , dtype = bool )
        self.actpix = self.actpix & ~pins

    ##################################################
    # Average Trace                                  #
    ##################################################
    def AvgTrace( self, micro=False, mask=None, redo=False ):
        """
        Calculates the average trace for each block. For a block, this will reaturn a [1,1,N] array.  For a thumbnail, this returns an [8,12,N] array

        NOTE:  mask takes in a numpy array of what you want to block
                EX: if you want to mask out empty wells (i.e. only have data from beads)
                        --> mask=empty
        """
        try:
            # cached setting for last calculation
            if self._avgtrace_micro == micro and not redo:
                return self.avgtrace
        except AttributeError:
            self._avgtrace_micro = micro
            pass

        if mask is not None:
            mask = ~np.logical_and( self.actpix, ~mask )
        else:
            mask = ~self.actpix

        # Start timing algorithm
        start_time = time.time()

        # Find active pixels (to self.actpix)
        self.CalculateActivePixels()

        if self.chiptype.blockR == 0 and self.chiptype.blockC == 0:
            raise ValueError( 'Error! dat file is of unexpected size' )

        # choose regular blocks or micro blocks for avg trace
        if micro:
            blockR = self.chiptype.microR
            blockC = self.chiptype.microC
        else:
            blockR = self.chiptype.blockR
            blockC = self.chiptype.blockC

        # Calculate the number of block
        rows = int( self.rows / blockR )
        cols = int( self.cols / blockC )
        
        # Initialize data arrays
        self.avgtrace = np.zeros( (rows, cols, self.data.shape[2] ) )

        # Analyze each block
        start = time.time()
        for r in range(rows):
            rws = slice( r*blockR, (r+1)*blockR )
            for c in range(cols):
                cls = slice( c*blockC, (c+1)*blockC )
                
                self.avgtrace[r,c,:] = stats.masked_avg( self.data[rws,cls,:], mask[rws,cls] )

        return self.avgtrace

    ##################################################
    # Noise algorithms                               #
    ##################################################
    def RCnoise( self, axis='row', driftcorrect=True, half='', frames=slice(None) ):
        ''' 
        Measures the CORRELATED row or column noise ( in uV )
        Return is a noise value PER row or column
        It averages the entire row or column and then measures the temporal noise of the average value
        Column noise means noise of the column.  This likely represents ADC noise
        Row noise means noise of the row.

        For (spatial) thumbnails, the top and bottom analysis blocks are removed, as well as the left and right 2 blocks
        '''
        if axis.lower() == 'col':
            axis = 0
        elif axis.lower() == 'row':
            axis = 1
        else:
            raise ValueError( 'RCnoise axis input is invalid.  It must be "row" or "col"' )

        cols = slice( None, None )
        if half.lower() == 'top':
            rows = slice( self.data.shape[0]/2, -self.data.shape[0]/8)
        elif half.lower() == 'bottom':
            rows = slice( self.data.shape[0]/8, self.data.shape[0]/2 )
        elif self.chiptype.spatn == 'self':
            rows = slice( self.data.shape[0]/8, -self.data.shape[0]/8 )
            cols = slice( self.data.shape[1]/6, -self.data.shape[0]/6 )
        else:
            rows = slice( None, None )

        
        if self._vfc:
            # VFC detected while loading the data.  This will always be OK because the entire array has the same compression.  
            # Different blocks however will have different compressions. 
            avg = self.GetUncompressedFrames( frames=frames )[rows,cols,:].mean( axis=axis )
        elif self.vfcenabled and self.is_Thumbnail:
            print( 'WARNING! VFC detected on a thumbnail.  If VFC is different for different rows, these results are unreliable' )
            avg = self.GetUncompressedFrames( frames=frames )[rows,cols,:].mean( axis=axis )
        else:
            avg = self.data[rows,cols,frames].mean( axis=axis )

        if driftcorrect:
            # Calculate the local drift correction (within a chip block)
            blocksize = ( self.chiptype.blockR, 1 ) # TODO this may fail at some point (e.g. 521)
            blocks = self.block_reshape( avg, blocksize )
            block_means = blocks.mean( axis=2 ).reshape( -1, avg.shape[-1] )
            # Apply the drift correction back onto the data
            avg_dc = avg - self.block_unreshape( block_means, blocksize )

            # Calculate the noise of the row means
            noise  = avg_dc.std( axis=1 )
        else:
            noise  = avg.std( axis=1 )
        return noise * self.uvcts, avg * self.uvcts

    def DriftFreeNoise( self, data, arr=True ):
        ''' 
        This measures the drift-corrected noise (in counts!) in the final dimension of data
        It is essentially a series of 2-frame measurements and is similar to the 1/(N-1) version 
        of the standard deviation for drift free data.

        Algorithm is:
        rms = 1/sqrt(2) * sqrt( sum_i( ( ( F_(i+1) - F_i ) / 2 )^2 ) / frames )

        This calculation breaks down for low N (<20), so for lower N, it is recommended to set arr=True
        This also does a series of 2-frame measurements, but reports a single noise value per frame 
        (assmes data is ergodic).

        Scott Parker 7/17/2015
        '''
        # Calculate a series of 2-frame noises
        if data.shape[-1] <= 20 and not self.noiseframewarning:
            print( 'WARNING!  Noise measurements for <= 20 are systematically lower than the actual noise.  Recommend setting arr=True' )
            self.noiseframewarning = True
        numdims = len( data.shape )
        noise = np.diff( data )
        axis = numdims - 1

        noise  /= 2
        noise **= 2

        # Calculate the rms
        if arr:
            rms   = noise.mean( axis=axis )
            rms **= 0.5
            rms  *= np.sqrt(2)  # This corrects for the frame averaging

        else:
            rms   = noise.reshape( -1, noise.shape[-1] ).mean( axis=0 )
            rms **= 0.5
            rms  *= np.sqrt(2)
        return rms

    def fftNoise( self, frames=slice(None, None), rows=slice(None, None), cols=slice(None, None), bkgsubtract=False ):
        '''
        This uses np.fft algorithm to get fft noise. This only works for tn/spa data
        '''
        # TODO: This loops over the entire array and then trims.  Maybe a little inefficient
        data = self.GetUncompressedFrames( frames=frames, common=False )
        # Do the FFT on one pixel to get dims
        fft1 = np.fft.rfft( data[0,0,:] )

        region_rows = int( self.rows / self.miniR )
        region_cols = int( self.cols / self.miniC )

        fft = np.zeros( ( data.shape[0], data.shape[1], fft1.size ) )
        for r in range( region_rows ):
            for c in range( region_cols ):
                rws = slice( r*self.miniR, (r+1)*self.miniR )
                cls = slice( c*self.miniC, (c+1)*self.miniC )

                goodpx = ~self.pinned[ rws, cls ]
                roi    = data[ rws, cls ]

                if not goodpx.any():
                    continue
                if bkgsubtract:
                    nnimg = roi - imtools.GBI( roi, ~goodpx, 10 )   # TODO: Skipping gain for now
                else:
                    nnimg = roi
                fft[rws,cls] = np.abs(np.fft.rfft( nnimg, axis=2 ))
        # Trim to limits
        fft = fft[rows, cols, :]
        # Flatten
        fft = np.reshape( fft, (-1,fft.shape[-1]), order='F' ) # TODO: not worying about pinned pixels here
        #fft = np.abs( fft )
        fft = fft.mean( axis=0 )
        print( fft.shape, 'fft shape' )
        ### sampling frequency in Hz
        freq = 15 #*2. # STP: Why *2?
        f = np.fft.rfftfreq(32, d=1./freq)
        print( f.shape )
       
        return fft, f


    ##################################################
    # Utilities                                      #
    ##################################################
    def block_reshape( self, data, blocksize ):
        rows, cols = data.shape
        numR = rows/blocksize[0]
        numC = cols/blocksize[1]
        return data.reshape(rows , numC , -1 ).transpose((1,0,2)).reshape(numC,numR,-1).transpose((1,0,2))

    def block_unreshape( self, data, blocksize ):
        numR, numC = data.shape
        rows = numR*blocksize[0]
        cols = numC*blocksize[1]
        els  = np.prod( blocksize )
        data = np.tile( data, (els, 1, 1) ).transpose((1,2,0))
        return data.transpose((1,0,2)).reshape( numC, rows, -1 ).transpose((1,0,2)).reshape( rows, cols )

    def CheckBasic( self, startframe=15 ):
        ''' Does a global averaging to determine if the step is basic or acidic.  This might not do wel at the corner blocks '''
        avgtrace = stats.masked_avg( self.data, self.pinned )
        if startframe == 0:
            startval = trace[0]
        else:
            startval = np.mean(avgtrace[self.inds<startframe])
        if np.mean(avgtrace[self.inds>=startframe]) > startval:
            self.isbasic = False
        else:
            self.isbasic = True

    def GetFrames( self ):
        ''' converts timestamps to frame indexes, saving to self.inds '''
        # Load the time stamps
        t = np.array( self.timestamps )
        # Calcualte the difference from frame to frame
        dt = (t[1:]-t[:-1])
        # Normalize by time stamp for the first frame
        # Convert to the nearest integer multiple. 
        # Round makes sure we round up, but indicies must be integers
        dt_inds = np.round((dt)/float(t[0])).astype(np.int)
        # Convert time deltas into frame indicies
        self.inds = np.append( [0], np.cumsum( dt_inds ) )
        # Figure out if data is compressed or not
        self.vfcprofile = np.append( [1], dt_inds )
        self._vfc = len( set( dt_inds ) ) > 1
        self.vfc_uncompressed = sum( self.vfcprofile == 1 )
        if self._vfc and self.vfc_uncompressed != 33:
            print( 'WARNING! Number of uncompressed frames in VFC file has changed from expected value of 33.  Certain calculations (e.g. Noise) may be inacurate' )
        # Calculate the average framerate
        self.framerate = (dt_inds/dt.astype(np.float)).mean()*1000

    def LocalVar( self, data, var='std', hd=False, dname='' ):
        ''' 
        Returns the local variation of the data
        var[ 'std', 'iqr' ]:  Which metric of variation
        hd[ False, True]   :  Uses mini- or micro-blocks as the tile
        dname              :  data name (e.g. gaincorr) used for special limits while flatening data
        '''
        # TODO: This might be better integrated with stats.uniformity
        if hd:
            block = 'micro'
        else:
            block = 'mini'
        # Get the tile size
        blockR = getattr( self.chiptype, '%sR' % block.lower() )
        blockC = getattr( self.chiptype, '%sC' % block.lower() )

        # Get the size of the block-averaged data
        rows = int( self.rows / blockR )
        cols = int( self.cols / blockC )

        # Initialize arrays
        output = np.zeros (( rows, cols ))

        # Loop over tiles 
        for r in range(rows):
            rws = slice( r*blockR, (r+1)*blockR )
            for c in range(cols):
                cls = slice( c*blockC, (c+1)*blockC )

                roi   = data[ rws, cls ]
                flat  = flatten( roi, dname )
                if var == 'iqr':
                    percs = stats.percentiles( flat )
                    output[ r, c ] = percs['iqr']
                elif var == 'std':
                    output[ r, c ] = flat.std()
                else:
                    raise ValueError( 'Invalid variance: %s' % var )

        return output
        
    def MakeBasic(self, trace, startframe=15):
        """
        Takes a normalized trace and makes sure that the trace is basic (i.e. negative).  If it isn't, it makes it so
        """
        if self.isbasic == True:
            return trace
        elif self.isbasic == False:
            return -trace
        elif self.isbasic is None:
            # This case shouldn't happen unless you override __init__
            if startframe == 0:
                startval = trace[0]
            else:
                startval = np.mean(trace[self.inds<startframe])
            if np.mean(trace[self.inds>=startframe]) > startval:
                return -trace
            return trace
        else:
            raise ValueError( 'self.isbasic is invalid (%s)' % self.isbasic )

    def ParseDatType(self,fcmask=True):
        ''' 
        Determine chip type from the number of rows and columns
        '''
        if self.chiptype is None:
            self.chiptype    = ct.get_ct_from_rc( self.rows, self.cols )
        self.type         = self.chiptype.type
        self.is_Thumbnail = self.chiptype.tn == 'self' or self.chiptype.spatn == 'self'
        self.miniR        = self.chiptype.miniR
        self.miniC        = self.chiptype.miniC
        if self.chiptype.spatn == 'self':
            self.miniR = self.chiptype.microR
            self.miniC = self.chiptype.microC

        if fcmask:
            fc_path = '/results/python/eccserver/flowcell_%s.dat' % self.type
            if os.path.exists(fc_path):
                fc = np.fromfile( fc_path , dtype=bool ).reshape( self.rows , self.cols )
                self.data[ fc ] = 0

    def VFCslice( self, frameslice ):
        ''' Calculates the new slice in VFC time coordinates '''
        first = frameslice.start
        last  = frameslice.stop
        if first is not None:
            if first < 0:
                first += len( self.inds )
            try:
                first = [ i>=first for i in self.inds ].index(True)
            except ValueError:
                # No frameslice in range
                first = len(self.inds)
        if last is not None:
            if last < 0:
                last += len( self.inds )
            try:
                last = [ i>last for i in self.inds ].index(True)
            except ValueError:
                last = len(self.inds)
        return slice( first, last )

    def GetUncompressedFrames( self, frames=slice(None, None), common=True ):
        ''' 
        Returns an aray of uncompressed pseudo-frames
        If different amounts of uncompressed frames exist in different parts 
        of the block, extra frames will be removed so the same number can be 
        applied across the array
        If common = True, only real frames which are uncompressed across the entire array
        are returned instead of pseud-frames.  This will result in fewer frames

        Frames are in NON-VFC coordinates
        '''
        # Don't recalculate if we don't have to
        try:
            if ( self.options['GetUncompressedFrames:frames'] == frames & 
                 self.options['GetUncompressedFrames:common'] == common ):
                return self.uncompressed
        except:
            pass
        self.options['GetUncompressedFrames:frames'] = frames
        self.options['GetUncompressedFrames:common'] = common
        # Return the full array if VFC is not detected
        if not( self._vfc or self.vfcenabled ):
            self.uncompressed = self.data
            return self.uncompressed
        if common:
            try:
                vfcfile = os.path.join( os.path.dirname( self.filename ), '..', 'VFCProfile.txt' )
                first, last = vfc.read_common_frames( vfcfile )
                print( 'VFC Profile read from %s' % vfcfile )
                self.uncompressed_inds = np.arange( first, last ) 
            except IOError:
                try:
                    vfcfile = os.path.join( moduleDir, '..', 'dats', '%s.txt' % self.chiptype.vfc_profile )
                    first, last = vfc.read_common_frames( vfcfile )
                    print( 'VFC Profile read from %s' % vfcfile )
                except IOError:
                    try:
                        vfcfile = '/software/config/VFCProfile.txt' 
                        first, last = vfc.read_common_frames( vfcfile )
                        print( 'VFC Profile read from %s' % vfcfile )
                    except IOError:
                        print( 'WARNING! Unable to read VFC profile.  Returning ALL data' )
                        return self.data
            if frames.start is not None and frames.start > first:
                first = frames.start
            if frames.stop is not None and frames.stop < last:
                last = frames.stop
            if self.is_Thumbnail:
                self.uncompressed = self.data[:,:,first:last]
            else:
                keep = np.ones( self.inds.shape, dtype=np.bool )
                if frames.start is not None:
                    keep[self.inds<frames.start] = False
                if frames.stop is not None:
                    keep[self.inds>=frames.stop] = False
                keep[self.inds<first] = False
                keep[self.inds>=last] = False
                self.uncompressed = self.data[:,:,keep]
                self.uncompressed_keep = keep
        elif self.is_Thumbnail:
            # Return a pseudo-frame
            # Thumbnails need to read the VFC profile and map
            try:
                vfcfile = os.path.join( os.path.dirname( self.filename ), '..', 'VFCProfile.txt' )
                profile = vfc.read_profile( vfcfile )
                #self.uncompressed_inds = ( first, last )
                print(  'VFC Profile read from %s' % vfcfile )
            except IOError:
                try:
                    vfcfile = os.path.join( moduleDir, '..', 'dats', '%s.txt' % self.chiptype.vfc_profile )
                    profile = vfc.read_profile( vfcfile )
                    print(  'VFC Profile read from %s' % vfcfile )
                except IOError:
                    print( 'WARNING! Unable to read VFC profile.  Returning ALL data' )
                    return self.data
            # Remove registers in the profile which are associated with the thumbnail
            vfc.remove_tns( profile )
            keeps = {}
            for p in profile:
                #keeps[p] = vfc.expand_uncomp_mask( profile[p], tn=True )
                expanded = vfc.expand_uncomp_mask( profile[p], tn=True )
                keep     = np.zeros( len(expanded), dtype=np.bool )
                keep[frames] = expanded[frames]
                keeps[p] = keep

            numframes = vfc.unify_uncomp_mask( keeps )

            x, y = np.meshgrid( range( self.data.shape[1] ), range( self.data.shape[0] ) )
            reg = vfc.reg_from_rc( y, x )

            mask = np.array( [ keeps[r] for r in reg.flatten() ] ).flatten()
            uncomp = self.data.flatten()[mask]
            self.uncompressed = uncomp.reshape( self.data.shape[0], self.data.shape[1], numframes )
            # Indicies are different for each pixel.  Don't bother setting
            self.uncompressed_inds = None
        else:
            # Blocks already have the required information stored in self.vfcprofile
            #keep = vfc.uncomp_mask( self.vfcprofile )
            keep = np.array( vfc.uncomp_mask( self.vfcprofile ) )
            if frames.start is not None:
                keep[self.inds<frames.start] = False
            if frames.stop is not None:
                keep[self.inds>frames.stop] = False
            self.uncompressed = self.data[:,:,keep]
            self.uncompressed_inds = self.inds[keep]
        return self.uncompressed
