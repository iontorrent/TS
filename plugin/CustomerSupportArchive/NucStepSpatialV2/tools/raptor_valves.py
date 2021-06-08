import os
import time, datetime
import atexit
import re

# Load modules for either datacollect or python control
try:
    try:
        from . import instrument
        _instrument_loaded = True
    except ValueError:
        import instrument
        _instrument_loaded = True
except ImportError:
    _instrument_loaded = False
if not os.getuid():
    try:
        import dev
        _dev_loaded = True
    except ImportError:
        _dev_loaded = False
if not _instrument_loaded and not _dev_loaded:
    raise ImportError( 'Unable to import instrument (datacollect control) or dev (python control) for raptor fluidics' )

# Load in the tank worker in case we decide to kill datacollect
try:
    from raptor_functions import TankWorker
except ImportError:
    pass

###################
## Typical Usage ##
###################
# Important Notes:
#   For extended flows, you may need to periodically refresh the W2 or pinch regulators
#   If you want to do this, it may be necessary to write your own flow scripts/macros
#   instead of using the pre-built functions
#   
# Case 1: Flow W2:
#   import raptor_valves as rv
#   tank = TankWorker()           # if killing datacollect
#   rv.set_mode( 'py' )           # for chips/python enviroment
#   rv.pressurize_w2()            # Build up pressure in W2
#   rv.set_pinch_reg( 3, 'main' ) # Set the flow rate for the main waste
#   rv.set_pinch_reg( 2, 'chip' ) # Set the flow rate for the chip waste
#   rv.flow_w2()                  # Flow W2 to chip only (see other options in command call)
#   <do stuff>
#   rv.stop()
#   tank.stop()                   # Turn off the tank worker pressure control


#################################################
## Define valve positions and mapping function ##
#################################################
valves = [ 'RGL',  # 1
           'RCL',  # 2
           'RAL',  # 3
           'RTL',  # 4
           'W2L',  # 5
           'W1L',  # 6
           'W3L',  # 7
           'AIR',  # 8
           'CWA',  # 9
           'MWA',  # 10
           '',     # 11
           '',     # 12
           'PRM',  # 13
           'PRC',  # 14
           'TNK',  # 15
           'VNT',  # 16
           'REG',  # 17
           'W2P',  # 18
           'SRP',  # 19
           'W3P',  # 20
           '',     # 21
           'SRV',  # 22 
           ]

def valvestate( cmd ):
    cmd_valves = [ s.strip().upper() for s in cmd.split(',') ]

    valvestate = 0
    for c in cmd_valves:
        if c:
            valvestate += 2**valves.index(c)
    return valvestate

################################################################
## Define control functions for python or datacollect control ##
################################################################
def _set_valvestate_dc( cmd ):
    ''' Calculate and set the valve position by name '''
    vs = valvestate( cmd )
    instrument.set_valve_position( vs )

def _set_pressure_dc( val ):
    instrument.set_pressure( val )

def _set_valvestate_py( cmd ):
    ''' Calculate and set the valve position by name '''
    vs = valvestate( cmd )
    dev.set( 'valve', vs )

def _set_pressure_py( val ):
    dev.set_pressure( val )

# Set the control functions, defaulting to datacollect
set_valvestate = _set_valvestate_dc
set_pressure   = _set_pressure_dc

# Set up a function to switch between modes
def set_control( mode = 'dc' ):
    global set_valvestate 
    global set_pressure
    if mode.lower().strip() in ( 'datacollect', 'dc', 'cmdcontrol' ):
        set_valvestate = _set_valvestate_dc
        set_pressure   = _set_pressure_dc
    if mode.lower().strip() in ( 'chip', 'python', 'py' ):
        set_valvestate = _set_valvestate_py
        set_pressure   = _set_pressure_py


#######################################################
## Provide convience functions for typical operation ##
#######################################################
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
def set_pinch_reg( pressure, reg, ptime=15, flow=True ):
    ''' Set the specified pinch regulator ("chip" or "main") to the specifed pressure '''
    reg_air = 'PR' + reg[0].upper()
    reg_liq = reg[0].upper() + 'WA'
    
    if flow:
        flow = 'W2L,' + reg_liq
    else:
        flow = ''

    macro = ( ( 2, 'VNT,SRP,SRV', None ), 
              ( 0.1, '', None ), 
              ( 1, None, pressure ), 
              ( ptime, 'REG,%s,%s' % ( reg_air, flow ), None ), 
              ( 0, '', None ) )

    run_macro( macro )

def pressurize_w2( pressure=8, ptime=15 ):
    ''' Pressurize the W2 bottle '''
    macro = ( ( 2, 'VNT,SRP,SRV', None ), 
              ( 0.1, '', None ), 
              ( 1, None, pressure ), 
              ( ptime, 'REG,W2P', None ) )
    run_macro( macro )

def pressurize_cartridge( pressure=8, ptime=15 ):
    ''' Pressurize the W2 bottle '''
    macro = ( ( 2, 'VNT,SRP,SRV', None ), 
              ( 0.1, '', None ), 
              ( 1, None, pressure ), 
              ( ptime, 'REG,SRP', None ) )
    run_macro( macro )

def flow_w2( chip=True, main=False, flowtime=None, w2_pressure=None ):
    ''' Flow W2 to chip and/or main waste.
    If flowtime is unspecified, it will continue after this function completes 
    '''
    flow = 'W2L'
    if chip:
        flow += ',CWA'
    if main: 
        flow += ',MWA'

    if flowtime is None:
        macro = ( ( None, flow, None ), )
    else:
        macro = ( ( flowtime, flow, None ),
                  ( None, '', None ) )
    run_macro( macro )

def stop():
    ''' Stop all valves/shutdown '''
    macro = ( ( None, '', 0 ),  )
    run_macro( macro )

########################################################
## Log parser for reading valve states from debug log ##
########################################################
def scanlog():
    ''' Returns binary valve states from /var/log/debug '''
    valving = re.compile( r'(\w+ \d+ \d+:\d+:\d+).*Valving Value:([a-f0-9]+)' )
    fpga    = re.compile( r'(\w+ \d+ \d+:\d+:\d+).*FpgaValvingState = ([0-9]+)' )
    valve_states = []
    for line in open( '/var/log/debug' ):
        if valving.search( line ):
            dt, valve = valving.search( line ).groups()
            valve = int( valve, 16 )
        elif fpga.search( line ):
            dt, valve = fpga.search( line ).groups()
            valve = int( valve )
        else:
            continue
        dt = datetime.datetime.strptime( dt, '%b %d %H:%M:%S' )
        valve = format( valve, '#024b' )[2:]
        valve_states.append( (dt, valve) )
    return valve_states

######################################################
## Make sure that we stop all flows when this exits ##
######################################################
atexit.register( stop )
