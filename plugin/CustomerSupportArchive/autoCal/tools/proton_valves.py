from . import instrument
import time
import atexit

valves = [ 'W2E',  # 1 - W2 to reference electrode to chip
           'AIR',  # 2 - Air to fluidics manifold
           'R1L',  # 3
           'C2L',  # 4
           'R2L',  # 5
           'C1L',  # 6
           'R3L',  # 7
           'W1L',  # 8
           'R4L',  # 9
           'W2L',  # 10
           'M2W'   # 11 - Manifold to waste
           'W3L',  # 12
           'CWS',  # 13 - Chip Waste Slow
           'CWF',  # 14
           'MWS',  # 15
           'MWF',  # 16
           '',     # 17
           '',     # 18
           '',     # 19
           '',     # 20
           '',     # 21
           'A2W',  # 22 - Air to Waste
           'GMP',  # 23 - Gas to Manifold Pressure
           'GSP',  # 24 - Gas Supply Pressure
           'W1P',  # 25
           'W2P',  # 26
           'W3P',  # 27
           'C1P',  # 28
           'C2P',  # 29
           'RNP',  # 30 - Reagent N pressure
           '',     # 31
           '',     # 32
           ]

def valvestate( cmd ):
    ''' Calculates the FPGA Valving State.  Input is either a list of valve numbers or string of valve names '''
    if isinstance( cmd, basestring ):
        cmd_valves = [ s.strip().upper() for s in cmd.split(',') ]
        cmd_valves = [ valves.index(c) for c in cmd_valves ]
    else:
        cmd_valves = [ c-1 for c in cmd ]

    valvestate = 0
    for c in cmd_valves:
        if c:
            valvestate += 2**c
    return valvestate

def set_valvestate( cmd ):
    ''' Calculate and set the valve position by name '''
    vs = valvestate( cmd )
    instrument.set_valve_position( vs )

def run_macro( macro ):
    ''' format ( ( time, valve, presure ), ( t, v, p ), ... ) 
    setting any parameter to none holds the value
    Pressure is changed after the valves are set
    '''

    for line in macro:
        if line[1] is not None:
            set_valvestate( line[1] )
        if line[2] is not None:
            instrument.set_pressure( line[2] )
        if line[0] is not None:
            time.sleep( line[0] )
    
# Note: these are convienence functions and may not always play well together or be the most 
# efficient for a larger script
def flow_w2( chip=True, main=False, flowtime=None, prime=True ):
    ''' Flow W2 to chip and/or main waste.
    If flowtime is unspecified, it will continue after this function completes '''
    macro = []
    macro.append( ( 5, [24,26,25,27,30], None ) ) # Pressurize the manifold
    if prime:
        valves = [24,26,25,27,30,1]
        if chip:
            valves += [ 13, 14 ]
        if main: 
            valves += [ 15, 16 ]
        macro.append( ( 8, valves, None ) ) # high speed Prime w2
    if prime:
        valves = [24,26,25,27,30,1]
        if chip:
            valves += [ 13 ]
        if main: 
            valves += [ 15 ]
        macro.append( ( flowtime, valves, None ) ) # slow speed flow
    if flowtime is not None:
        macro.append( ( None, [24,26,25,27,30], None ) ) # stop flow
    run_macro( macro )

def w1_step( flowtime=7 ):
    macro = []
    macro.append( ( 5, [24,26,25,27,30], None ) ) # Pressurize the manifold
    macro.append( ( 2, [24,26,25,27,30,1,13,14,15,16], None ) ) # High Speed Prime W2
    macro.append( ( 5, [24,26,25,27,30,1,8,13,14,15,16], None ) ) # High Speed Prime W1
    macro.append( ( 10, [24,26,25,27,30,1,8,13,15], None ) ) # Stage W1
    macro.append( ( flowtime, [24,26,25,27,30,8,13,15], None ) ) # Flow W1
    macro.append( ( 1, [24,26,25,27,30,1,8,13,15], None ) ) # Stage W1
    macro.append( ( 5, [24,26,25,27,30,1,13,15], None ) ) # Flow W2
    macro.append( ( None, [24,26,25,27,30], None ) ) # Pressurize the manifold
    run_macro( macro )

def stop():
    ''' Stop all valves/shutdown '''
    macro = ( ( None, [24,26,25,27,30], 10.5 ),  )
    run_macro( macro )

atexit.register( stop )
