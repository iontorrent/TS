"""
Module for running manual chip calibration routines and subroutines, also leveraging direct instrument control.  
"""

from . import instrument
from . import datfile
from . import system
from . import proton_valves as pv
from . import raptor_valves as rv
import numpy as np
import time
np.seterr( 'ignore' )

verbose = False
sequencer = 'proton'

class CalibrationError( Exception ):
    def __init__( self, value ):
        self.value = value
    def __str__( self ):
        return repr( self.value ) 

def annotate( txt ):
    if verbose:
        print txt

def cal_sweeper( stop=True, valvestate=209680 ):
    ''' Centers the array by sweeping Vref and estimating the center 
    returns boolean pass/fail

    setting stop=False measures all values and returns 
        target  - vref target
        centers - mean array signal at each vref
        pinned  - fraction of pinned pixels at each vref
        vref    - vref for each measurement
    '''
    # Get valve position
    vs = instrument.get_valve_position()
    try:
        # First Chip image takes a while.  Start one now while we flow some more W2
        #instrument.capture_image( wait=False )
        # Start flowing W2.  TODO:  This is RAPTOR ONLY!
        annotate( 'Flowing W2' )
        instrument.set_valve_position( valvestate )
        instrument.wake_chip( wait=True )
        #time.sleep( 4 )

        centers = []
        pinned  = []
        step    = 0.05
        vref    = np.arange( 0.8, 2.001, step )
        # Sweep Vref and measure the array center, stopping when found
        last = 0
        target = None
        for v in vref:
            instrument.set_dac( v )
            time.sleep( 0.1 )
            bins = instrument.get_hist( capture=True )
            hm = hist_mean( bins )
            hp = hist_pins( bins )
            annotate( 'Vref = %0.2f | Center = %0.3f | Pinned = %0.3f' % ( v, hm, hp ) )
            if v != vref[0]: # I don't trust the first measurement to be real
                if last < 0.5 and hm >= 0.5: # Just crossed 0.5 
                    target = ( 0.5 - last ) / ( hm - last ) * ( step ) + v - step
                    if hp < 0.25 and stop:
                        annotate( 'Setting vref to %0.3f' % target )
                        instrument.set_dac( target )
                        instrument.set_valve_position( vs )
                        return True
            centers.append( hm )
            pinned.append( hp )
            last = hm


        if target:
            annotate( 'Setting vref to %0.3f' % target )
            instrument.set_dac( target )
        annotate( 'Returning valve state' )
        instrument.set_valve_position( vs )
        if stop:
            # If you made it here, we couldn't calibrate
            return False
        else:
            return target, centers, pinned, vref
    finally:
        instrument.set_valve_position( vs )

def is_centered( dist, lims=(0.4, 0.6), maxpins=0.25 ):
    ''' Checks if the array is centered '''
    hm = hist_mean( dist )
    hp = hist_pins( dist )
    return hp < maxpins and hm > lims[0] and hm < lims[1]

def hist_mean( dist, removepins=True):
    ''' Calculate the mean of the distribution (limits 0-1) '''
    lims = np.linspace( 0, 1, len(dist) )
    dist = np.array( dist )
    # Remove pinned wells
    if removepins:
        dist = np.array(dist[1:-1])
        lims = lims[1:-1]

    avgval = (dist*lims).sum()/dist.sum()
    if np.isnan( avgval ):
        avgval = int( dist[-1] > dist[0] )
    return avgval

def hist_pins( dist ):
    ''' Calculate fraction of pinned wells in the distribution '''
    pinned = (dist[0] + dist[-1])/float(sum(dist))
    return pinned

def quick_gain( mv_v=True ):
    ''' Does a quick and dirty gain measurement by only analyzing the shift in the histogram '''
    instrument.wake_chip( wait=True )
    dr   = instrument.get_dynamic_range()
    vref = instrument.get_dac()
    new_vref = vref + dr/6
    bins = instrument.get_hist( capture=True )
    hm_low = hist_mean( bins )
    instrument.set_dac( new_vref )
    time.sleep( 0.1 )
    bins = instrument.get_hist( capture=True )
    hm_high = hist_mean( bins )
    instrument.set_dac( vref )

    delta_hist = hm_high - hm_low
    gain = ( delta_hist * dr ) / ( new_vref - vref )
    if mv_v:
        gain *= 1000.
    return gain

def measure_gain():
    sto = instrument.get_save_thumbnail_only()
    dr = instrument.get_dynamic_range()
    dac = instrument.get_dac()
    stepsize = dr/6.

    try:
        instrument.set_save_thumbnail_only( True )
        wt, st = instrument.start_script( 'Script_NoiseScript', 'stp-gain', wait=False )
        time.sleep( 2.5 )
        instrument.set_dac( dac + stepsize )
        instrument.wait_for_script( 'Script_NoiseScript', wt, st )

        name = system.get_instrument_name()
        path = '/sw_results/%s-stp-gain/thumbnail/acq_0000.dat' % name
        df = datfile.DatFile( path, norm=True )
        height = df.measure_plateau()

        height_top = height[100:400,100:-100].flatten()
        height_top.sort()
        q2_top = height_top[ height_top.size/2 ]

        height_bot = height[400:-100,100:-100].flatten()
        height_bot.sort()
        q2_bot = height_bot[ height_bot.size/2 ]

        height = height[100:-100,100:-100].flatten()
        height.sort()
        q2 = height[ height.size/2 ]
    finally:
        instrument.set_dac( dac )
        instrument.set_save_thumbnail_only( sto )

    # Do unit conversions (to mV/V)
    q2     = ( float(q2)/(2**14)*(dr*1000) )     / ( stepsize )
    q2_top = ( float(q2)/(2**14)*(dr*1000) ) / ( stepsize )
    q2_bot = ( float(q2)/(2**14)*(dr*1000) ) / ( stepsize )
    return q2, ( q2_top, q2_bot )

def measure_noise():
    ''' Flows W2 and measures noise.  This assumes that the pinch regulators are already set up '''
    # Read the initial state
    sto = instrument.get_save_thumbnail_only()

    try:
        instrument.set_save_thumbnail_only( True )
        instrument.start_script( 'Script_NoiseScript', 'stp-noise' )

        dr = instrument.get_dynamic_range()*1000.
        name = system.get_instrument_name()
        path = '/sw_results/%s-stp-noise/thumbnail/acq_0000.dat' % name
        try:
            df = datfile.DatFile( path, dr=dr )
        except TypeError:
            df = datfile.DatFile( path )
            df.chiptype.uvcts = (dr*1000.)/float(2**14)
        noise = df.measure_noise()

        noise_top = noise[100:400,100:-100].flatten()
        noise_top.sort()
        q2_top = noise_top[ noise_top.size/2 ]

        noise_bot = noise[400:-100,100:-100].flatten()
        noise_bot.sort()
        q2_bot = noise_bot[ noise_bot.size/2 ]

        noise = noise[100:-100,100:-100].flatten()
        noise.sort()
        q2 = noise[ noise.size/2 ]
    finally:
        instrument.set_save_thumbnail_only( sto )
    return q2, ( q2_top, q2_bot )

def center_dss( maxitter=25, window=0.05, maxstep=500, minstep=100, dt=0.2, damp=1, __wake=False ):
    ''' This centers DSS on the current Vref.  
    It does not do any flow control '''
    annotate( 'Waking chip' )
    instrument.wake_chip( wait=True )
    if __wake:
        time.sleep( 2 )

    # This should pin array high
    annotate( 'Setting DSS to 0' )
    for i in range( 5 ):
        instrument.set_dac_start_sig( 0 )
        time.sleep( dt )
        bins = instrument.get_hist( capture=True )
        hm = hist_mean( bins )
        if ( hm > 0.5 ) or ( bins[-1] > sum( bins[:-1] ) ):
            break
        else:
            annotate( 'Array Center: %0.2f' % hm )
            annotate( 'Pinned high:  %i' % bins[-1] )
            annotate( 'Not high:     %i' % sum( bins[:-1] ) )
            if i == 4:
                if not __wake:
                    return center_dss( maxitter=maxitter, window=window, __wake=True )
                raise CalibrationError( 'Could not bring DSS into range' )

    # This should pin array low
    annotate( 'Setting DSS to Max' )
    for i in range( 5 ):
        instrument.set_dac_start_sig( 2**14 - 1 )
        time.sleep( dt )
        bins = instrument.get_hist( capture=True )
        hm = hist_mean( bins )
        if ( hm < 0.5 ) or ( bins[0] > sum( bins[1:] ) ):
            break
        else:
            annotate( 'Array Center: %0.2f' % hm )
            annotate( 'Pinned low:   %i' % bins[0] )
            annotate( 'Not low:      %i' % sum( bins[0:] ) )
            if i == 4:
                if __wake: 
                    return center_dss( maxitter=maxitter, window=window, __wake=False )
                raise CalibrationError( 'Could not bring DSS into range' )

    # OK now lets try to center the range
    center = hm
    annotate( 'Setting DSS to Middle' )
    dss = 2**13
    instrument.set_dac_start_sig( dss )
    time.sleep( dt )
    
    for iters in range( maxitter ):
        center, dss, done = step_dss_to_center( center, dss, window=window, maxstep=maxstep, 
                                                minstep=minstep, dt=dt, damp=damp )
        if done:
            return dss

    raise CalibrationError( 'Could not center the array in %i itterations' % maxitter )

def step_dss_to_center( last_center, last_dss, window=0.05, maxstep=500, minstep=100, dt=0.2, damp=1 ):
    ''' Move DSS to center, based on the previous position '''
    # If we are sufficiently far from being pined, throw out the pinned pixels
    remove_pins = ( last_center > 0.2 ) and ( last_center < 0.8 )

    # Measure the array center
    bins = instrument.get_hist( capture=True )
    center = hist_mean( bins, removepins=remove_pins )
    output = 'Array Center: %0.3f' % ( center )
    if not remove_pins:
        output += ' (pinned)'
    annotate( output )

    # Measure the current DSS
    dss = instrument.get_dac_start_sig()
    annotate( dss )

    # Check if we are centered
    if ( center > (0.5-window) ) and ( center < (0.5+window) ) and ( remove_pins ):
        return center, dss, True    # Centered

    if last_dss == dss:
        # Not going to go anywhere.  Pretend like we gave it a kick
        last_dss -= 5
    
    try:
        # Calcualte the nominal amount to move with a linear fit
        delta = int( ( dss - last_dss )/float(( center - last_center ))*( 0.5 - center ) )
        # Since a linear fit will overshoot, especially if the array was previously 
        # mostly pinned, damp the step size a bit, depending on how far from center you are
        adjust = (1-abs(center-0.5) )*damp
        delta = int( adjust * delta )
        if np.isnan( delta ):
            raise ValueError
    except ( ValueError, ZeroDivisionError ):
        # If the center didn't move, use a the largest allowable step
        delta = np.sign( center - 0.5 ) * maxstep

    # Make sure that we aren't stepping to far
    sign = np.sign( delta )
    if not sign:
        sign = dss - last_dss
    delta = sign * min( abs( delta ) , maxstep ) 
    # Make sure we are moving enough to not get caught on the edges
    if center < 0.2:
        delta = min( -minstep, delta )
    if center > 0.8:
        delta = max( minstep, delta )
    #if not( ( center > 0.2 ) and ( center < 0.8 ) ):
    #    delta = sign * max( abs( delta ), minstep )

    # Set the new dss
    new_dss = dss + delta
    instrument.set_dac_start_sig( new_dss ) 
    time.sleep( dt )

    return center, dss, False

def center_vref( maxitter=25, window=0.05, maxstep=0.2, minstep=.05, dt=0.2, damp=1, __wake=False ): 
    ''' This centers Vref on the current DSS.  
    It does not do any flow control '''

    annotate( 'Waking chip' )
    instrument.wake_chip( wait=True )
    if __wake:
        time.sleep( 2 )

    # This should pin array low 
    annotate( 'Setting vref to 0' )
    for i in range( 5 ):
        instrument.set_dac( 0 )
        time.sleep( dt )
        bins = instrument.get_hist( capture=True )
        hm = hist_mean( bins )
        if ( hm < 0.5 ) or ( bins[0] > sum( bins[1:] ) ):
            break
        else:
            annotate( 'Array Center: %0.2f' % hm )
            annotate( 'Pinned low:   %i' % bins[0] )
            annotate( 'Not low:      %i' % sum( bins[1:] ) )
            if i == 4:
                if not __wake:
                    return center_vref( maxitter=maxitter, window=window, __wake=True )
                raise CalibrationError( 'Could not bring Vref into range' )

    # This should pin array low
    annotate( 'Setting Vref to Max' )
    for i in range( 5 ):
        instrument.set_dac( 3.3 )
        time.sleep( dt )
        bins = instrument.get_hist( capture=True )
        hm = hist_mean( bins )
        if ( hm > 0.5 ) or ( bins[-1] > sum( bins[:-1] ) ):
            break
        else:
            annotate( 'Array Center: %0.2f' % hm )
            annotate( 'Pinned high:  %i' % bins[-1] )
            annotate( 'Not high:     %i' % sum( bins[:-1] ) )
            if i == 4:
                if not __wake: 
                    return center_vref( maxitter=maxitter, window=window, __wake=True )
                raise CalibrationError( 'Could not bring DSS into range' )

    # OK now lets try to center the range
    center = hm
    annotate( 'Setting Vref to 1.5' )
    vref = 1.5
    instrument.set_dac( vref )
    time.sleep( dt )
    
    for iters in range( maxitter ):
        center, vref, done, = step_vref_to_center( center, vref, window=window, maxstep=maxstep, 
                                                   minstep=minstep, dt=dt, damp=damp )
        if done:
            return vref

    raise CalibrationError( 'Could not center the array in %i itterations' % maxitter )

def step_vref_to_center( last_center, last_vref, window=0.05, maxstep=0.2, minstep=.05, dt=0.2, damp=1 ):
    ''' Move Vref to center, based on the previous position '''
    # If we are sufficiently far from being pined, throw out the pinned pixels
    remove_pins = ( last_center > 0.2 ) and ( last_center < 0.8 )

    # Measure the array center
    bins = instrument.get_hist( capture=True )
    center = hist_mean( bins, removepins=remove_pins )
    output = 'Array Center: %0.3f' % ( center )
    if not remove_pins:
        output += ' (pinned)'
    annotate( output )

    # Measure the current DSS
    vref = instrument.get_dac()
    annotate( vref )

    # Check if we are centered
    if ( center > (0.5-window) ) and ( center < (0.5+window) ) and ( remove_pins ):
        return center, vref, True    # Centered

    if last_vref == vref:
        # Not going to go anywhere.  Pretend like we gave it a kick
        last_vref -= 0.005
    
    try:
        # Calcualte the nominal amount to move with a linear fit
        delta = ( vref - last_vref )/float(( center - last_center ))*( 0.5 - center ) 
        # Since a linear fit will overshoot, especially if the array was previously 
        # mostly pinned, damp the step size a bit, depending on how far from center you are
        adjust = (1-abs(center-0.5) )*damp
        delta = adjust * delta 
        if np.isnan( delta ) or ( delta == 0 ):
            raise ValueError
    except ( ValueError, ZeroDivisionError ):
        # If the center didn't move, use a the largest allowable step
        delta = np.sign( 0.5 - center ) * maxstep

    # Make sure that we aren't stepping to far
    sign = np.sign( delta )
    if not sign:
        sign = vref - last_vref
    delta = sign * min( abs( delta ) , maxstep ) 
    # Make sure we are moving enough to not get caught on the edges
    if center < 0.2:
        delta = max( minstep, delta )
    if center > 0.8:
        delta = min( -minstep, delta )

    # Set the new vref
    new_vref = vref + delta
    instrument.set_dac( new_vref ) 
    time.sleep( dt )

    return center, vref, False

def manual_centering( window=None, dt=0.05 ):
    ''' Provides a monitoring tool for manually centering the window
    If window is specified (typically <= 0.5), the function will auto-exit when centered
    If not, you must exit with ctrl-C'''

    instrument.wake_chip( wait=True )

    if window is None:
        annotate( 'Press ctrl-C to stop centering'  )

    try:
        while True:
            # Measure the array center
            bins = instrument.get_hist( capture=True )
            center = hist_mean( bins, removepins=True )
            output = 'Array Center: %0.3f' % ( center )
            annotate( output )
            if ( window is not None ) and ( abs( center - 0.5 ) < window ):
                return 
    except KeyboardInterrupt:
        return

def valve_macro( macro ):
    ''' Calls the correct macro runner depending on target system '''
    if sequencer == 'proton': 
        pv.run_macro( macro )
    elif sequencer == 'raptor': 
        rv.run_macro( macro )
    else:
        raise ValueError( "'%s' is not a recognized sequencer" % sequencer )

def vref_gain_curve( low=0.8, high=2.2, delta=0.02, flow=40, fullgain=False, macro=None ):
    ''' Measures the Vref-gain curve 
    # Setting macro allows for override of the initial W2 priming.  
    # Format is defined according to raptor_valves.py or proton_valves.py
    '''
    vref = np.arange( low, high+float(delta)/2, delta )

    vs = instrument.get_valve_position()
    
    # Take an image to warm up
    instrument.capture_image()

    output = []
    if macro is not None:
        pass
    elif sequencer == 'proton':
        macro = ( (    3, [ 1, 13, 14, 24, 26 ], None ),  # W2 Fast for 3 seconds
                  ( max( flow-3, 1 ), [ 1, 13, 14, 24, 26 ], None ),  # W2 Slow for remainder of pre-flow
                  ( None, [ 1, 13, 14, 24, 26 ], None ) ) # W2 Slow Until Stopped
    elif sequencer == 'raptor': 
        macro = ( ( 10, 'W2P,W2L,REG,PRC,PRM,CWA,MWA', 8),  # Fast W2 flow to chip and waste
                  ( max( flow-12, 1 ), 'W2L,REG,PRC,CWA', 2 ), # Slow W2 flow to chip
                  ( None, 'W2P,W2L,CWA', 8 ), ) # Slow W2 flow to waste until stopped, maintaing W2 pressure
    else: 
        raise ValueError( "'%s' is not a recognized sequencer" % sequencer )

    try:
        # Pre-flow W2
        if flow is not None:
            annotate( 'Flowing W2' )
            valve_macro( macro )

        for v in vref:
            instrument.set_dac( v )
            time.sleep( 0.2 )
            try:
                valve_macro( [macro[-1]] ) # Refresh W2 flow
                dss = center_dss()
                quickgain = quick_gain()
                if fullgain:
                    gain_out = measure_gain()
                else:
                    gain_out = [None]
            except CalibrationError:
                dss = None
                quickgain = None
                gain_out = [None]

            output.append( { 'vref': v, 
                             'dss': dss, 
                             'gain': gain_out[0], 
                             'quickgain': quickgain } )

    except:
        raise
    finally:
        instrument.set_valve_position( vs )

    return output
