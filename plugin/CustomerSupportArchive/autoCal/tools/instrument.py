''' 
DO NOT PUT ANYTHING IN HERE THAT REQUIRES RUNNING AS ROOT! 
Module to interface with the instrument using datacollect

IMPORTANT: All calls to grep /var/log/debug MUST USE the -a flag as well to avoid hanging on binary junk
'''
import xml.etree.ElementTree as ET
import subprocess
import datetime, time
import os, sys
import numpy as np
import traceback
import re

verbose = 1
devnull = open( os.devnull, 'w' )

_disable_clamp_open = False

GENEXUS = 'GENEXUS' 
S5      = 'S5'
PROTON  = 'PROTON'

# Make sure that /var/log/debug is readable.  TODO: "sudo python" w/o password only works on raptor
if os.path.exists( '/var/log/debug' ):
    subprocess.call( '''sudo python -c "import os; os.system( 'chown syslog /var/log/debug; chgrp adm /var/log/debug; chmod 644 /var/log/debug' )"''', shell=True )

class cmdError( Exception ):
    pass

def _annotate( text, level=2 ):
    ''' 
    Prints the text based on the verbosity 
    0 is always shown.
    1 is usually shown
    2 is for optional text
    '''
    if level <= verbose:
        print text

def cmdcontrol( field, value=None, group=None ):
    # Need to be careful about booleans
    if value is True:
        value = 'true'
    elif value is False:
        value = 'false'
    elif value is None:
        value = ''

    if group:
        cmd = '/software/cmdControl SetSub %s %s %s' % ( group, field, value )
    else:
        cmd = '/software/cmdControl Set %s %s' % ( field, value )

    _annotate( cmd )
    return get_BBresp( cmd )

def cmdquery( group, field=None ):
    if field:
        cmd = '/software/cmdControl Query %s | grep %s' % ( group, field )
    else:
        cmd = '/software/cmdControl Query %s' % ( group )
    _annotate( cmd )
    p = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE, stderr=devnull )
    return p.communicate()[0].strip()

def cmdquery_val( *args, **kwargs ):
    ''' wrapper for cmdquery.  Extracts the value xml tag and returns it as a string
        raises a cmdError if no match was found
    '''
    resp = cmdquery( *args, **kwargs )
    regex = re.compile( '\<value\>(.*)\<\/value\>' )
    try:
        return regex.search( resp ).groups()[0]
    except AttributeError:
        raise cmdError

def chip_poweroff():
    cmdcontrol( 'wantChipEnabled', 0, 'WrCntrls' )

def cycle_chip_power():
    chip_poweroff()
    time.sleep( 3 )
    wake_chip()
    capture_image( timeout=20 )

def datacollect_running():
    cmd = "ps ax | grep datacollect"
    resp   = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return 'datacollect.avx' in resp

def capture_image( wait=True, timeout=10 ):
    ''' Captures a chip image.  This doesn't save it though, but it does write a histogram to explog '''
    cmdcontrol( 'ChipImage' )
        
    if wait:
        starttime = datetime.datetime.now()
        cmd = 'grep -a -e \<SET\>.*\<ChipImage\> -e "Acquisition Complete" /var/log/debug'
        while True:
            output = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE, stderr=devnull ).communicate()[0]
            # Make sure the output was recieved
            if ( '<SET>' not in output or
                 '<ChipImage>' not in output ):
                print output
                _annotate( 'ERROR unable to parse /var/log/debug', 0 )
                return False

            # Step backwards to find an 'Acquisition Complete' later than a Set ChipImage
            for line in reversed( output.split( '\n' ) ):
                if 'Acquisition Complete' in line:
                    return True
                elif '<SET>' in line and '<ChipImage>' in line:
                    break

            elapsed = (datetime.datetime.now() - starttime ).total_seconds()
            if elapsed > timeout:
                _annotate( 'ERROR timeout while waiting for ChipImage', 0 )
                return False
            time.sleep(0.1)

def chip_ready():
    cmd = "/software/cmdControl Query WrCntrls | grep ChipReady"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            return int( result.text )
        except ValueError:
            continue
    return 0

def close_cartridge( require_cartridge=True, wait=False ): 
    ''' 
    Closes the nuc cartridge.  Will wait up to 20 seconds for the process to complete
    Returns True/False if process was sucessful.
    If state is unknown or process is ongoing, returns None
    '''
    # I am so wary about a cartridge not eing fully installed that I am going to force 
    # it to always be present # STP 12/3/15
    require_cartridge=True
    # Check if the cartridge is actually open:
    rmh = is_reagent_motor_home()  # Returns True if motor is open
    rmc = is_reagent_motor_closed()
    rcp = is_reagent_cartridge_present()
    if rmc:
        # Cartridge is already closed
        return True
    if not ( rmc or rmh ):
        # motor is in an undetermined state
        _annotate( 'Unable to determine motor state.  It is neither open nor closed', 0 )
        return False

    if require_cartridge and not rcp:
        _annotate( 'Cartridge is not present!', 0 )
        return False

    # Close the clamp
    cmdcontrol( 'CloseCartridge', 1 )
    #NOTE: 6/29/2018
    #       These next two lines seem old as cmd isn't defined and cmdcontrol is doing the work
    #       Commented out as this is the source of errors for init and clean
    ##_annotate( cmd )
    ##subprocess.call( cmd, shell=True, stdout=devnull, stderr=devnull )

    # Wait and then make sure the chip motor is actually closing
    time.sleep( 2 )
    if is_reagent_motor_home():
        _annotate( 'Unable to close the clamp!', 0 )
        return False
    else:
        # Motor is not home (closed, moving, or stuck)
        pass

    if wait:
        timeout = 20
        ahora = _now() 
        while not is_reagent_motor_closed():
            elapsed = (_now() - ahora).total_seconds()
            if elapsed > timeout:
                _annotate( 'Reagent cartridge did not open in %i seconds' % ( int(timeout) ), 0 )
                return False
            time.sleep( 1 )

        return True

    return None

def close_clamp( wait=False ): 
    ''' 
    Closes the chip clamp.  Will wait up to 20 seconds for any chip ready signal 
    Returns True/False if process was sucessful.
    If state is unknown or process is ongoing, returns None
    '''
    # Check if the clamp is actually open:
    cmh = is_chip_motor_home()  # Returns True if motor is open
    if cmh:
        # Motor is home (open)
        # Close the clamp
        cmdcontrol( 'CloseClamp', 1 )
        # Wait and then make sure the chip motor is actually closing
        time.sleep( 2.5 )
        if is_chip_motor_home():
            _annotate( 'Unable to close the clamp!', 0 )
            return False
    else:
        # Motor is not home (closed, moving, or stuck)
        pass

    # In case the clamp is already closed, turn on the chip
    try:
        wake_chip( wait=False )
    except IOError:
        # This is generated if chip clamp is already open
        pass

    if wait:
        return wait_for_chip_ready( 1, timeout=20 )
    else:
        return None

def get_actual_dynamic_range():
    ''' Measures the chip version and writes to conditions '''
    # This is not the actual dynamic range!
    BBresp  = get_BBresp( "/software/cmdControl Query WrCntrls | grep DynamicRangeActual" )
    root    = parse_xml   ( BBresp )
    for info in root.iter('value'):
        return float(info.text)
    return 0

def get_active_genexus_lanes( as_int=False ):
    ''' Returns a list of lane numbers as integers 

        4-bit array, integer values
        --> all lanes = 15, no lanes = 0
        Lane 1 == 1
        Lane 2 == 2
        Lane 3 == 4
        Lane 4 == 8
    '''
    lane_bits  = int( cmdquery_val( 'Options', field='DbgLaneActive' ) )
    if as_int:
        return lane_bits

    temp    = lane_bits
    lanes   = []
    for x,l in zip([8,4,2,1],[4,3,2,1]):
        if temp >= x:
            lanes.append( l )
            temp -= x
        if temp == 0:
            break
    return sorted( lanes )


def get_BBresp( cmd, maxcount=30, platformCmd=False ):
    ''' Keeps asking cmdControl until it gets an answer '''
    count = 0
    while True:
        BBresp   = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]

        if platformCmd: source = 'platformCmd'
        else:           source = 'cmdControl'

        if BBresp is None:
            count += 1
            _annotate('Recieved invalid response from %s %i times' % (source, count), 0 )
            _annotate('...%s' % cmd, 0 )
            time.sleep(0.5)
        else:
            return BBresp
        if maxcount >= count:
            raise IOError( 'Unable to get a valid reponse from %s (%i tries) to the question: %s' % ( source, maxcount, cmd ) )

def get_cal_gain():
    ''' Returns the ExpInfo gain data '''
    cmd = '/software/cmdControl Query ExpInfo | grep ChipGain'
    BBresp = get_BBresp( cmd )
    resp   = parse_xml ( BBresp )
    value = resp.find('value').text
    return float( value )

def get_cal_noise():
    ''' Returns the ExpInfo for noise calibration '''
    cmd = '/software/cmdControl Query ExpInfo | grep -e ChipNoise -e RowNoise'
    BBresp = get_BBresp( cmd )
    resp   = parse_xml ( BBresp )
    results = {}
    for info in resp.iter( 'ExpInfo' ):
        name  = info.find('name').text
        value = info.find('value').text
        if name == 'ChipNoiseInfo':
            err = []
            for part in value.split():
                try:
                    cni_name, cni_value = part.strip().replace('=',':').split( ':' )
                except:
                    err.append( part )
                    continue
                try:
                    cni_value = float(cni_value)
                except:
                    pass
                results['ChipNoiseInfo_%s' % cni_name] = cni_value
            if err:
                results['ChipNoiseInfo_ParseError'] = err
        else:
            try:
                value = float(value)
            except:
                pass
            results[name] = value
    return results

def get_chip_clock():
    ''' Estimates the framerate '''
    BBresp = get_BBresp( "/software/cmdControl Query WrCntrls | grep ChipFreq" )
    resp         = parse_xml ( BBresp )
    freq = float( resp.find('value').text )

    return freq

def get_chip_temp( convert=True ):
    BBresp = get_BBresp( "/software/cmdControl Query ChipTemp" )
    try:
        temp = float( re.search( r'\<ChipTemp\>([0-9\.]*)\<\/ChipTemp\>', BBresp).groups()[0] )
    except:
        print 'Error reading chip temperature'
        return 0

    if convert:
        # see /software/testing/support/chip.py
        slope = 1/2. # C/dn8
        yint  = -35.  # offset
        temp = slope * temp + yint

    return temp

def get_chip_version():
    ''' Measures the chip version and writes to conditions '''
    BBresp  = get_BBresp( "/software/cmdControl Query WrCntrls | grep ChipTSVersion" )
    root    = parse_xml   ( BBresp )
    for info in root.iter('value'):
        return info.text
    return ''

def get_dac():
    ''' Gets the dac voltage '''
    BBresp = get_BBresp( r"/software/cmdControl Query WrCntrls | grep \>dac\<" )
    resp   = parse_xml ( BBresp )
    return float( resp.find('value').text )
get_vref = get_dac

def get_dac_start_sig():
    ''' Measures the chip version and writes to conditions '''
    BBresp  = get_BBresp( "/software/cmdControl Query WrCntrls | grep RangeStart" )
    resp    = parse_xml   ( BBresp )
    freq = float( resp.find('value').text )
    return freq
get_dss = get_dac_start_sig

def get_dynamic_range():
    ''' Measures the chip version and writes to conditions '''
    BBresp  = get_BBresp( "/software/cmdControl Query WrCntrls | grep DynamicRange" )
    root    = parse_xml   ( BBresp )
    for info in root.iter('value'):
        return float(info.text)
    return 0

def get_instrument_version():
    BBresp  = get_BBresp( "/software/cmdControl Query InstalledPackages" )
    BBresp  = BBresp.replace  ( 'REQ Query InstalledPackages' , '' )
    BBresp  = BBresp.replace  ( '\n\ndone\n'        , '' )
    root    = parse_xml   ( BBresp )
    data    = {}
    for info in root.iter('InstalledPackage'):
        try:
            key = info.find('name').text.strip().replace('\n',';').replace(',',';').replace(' ','_')
            if not key:
                continue
            version = info.find('version').text.strip().replace('\n',';').replace(',',';')
            data[key] = version
        except AttributeError:
            pass
    for info in root.iter('cpuInfo'):
        version = info.text.strip().replace('\n',';').replace(',',';')
        data['cpuInfo'] = version
    return data

def get_datacollect_version():
    info = get_instrument_version()
    for i in info:
        if 'datacollect' in i:
            return info[i]
    return info['datacollect']

def get_efuse():
    ''' Reads the efuse from ExpInfo '''
    BBresp  = get_BBresp( "/software/cmdControl Query ExpInfo | grep ChipEfuse" )
    root    = parse_xml   ( BBresp )
    for info in root.iter('value'):
        return info.text
    return ''

def get_framerate( baserate=15. ):
    ''' Estimates the framerate '''
    BBresp = get_BBresp( "/software/cmdControl Query Options | grep OverSample" )
    resp         = parse_xml ( BBresp )
    oversample   = float( resp.find('value').text )

    framerate = baserate * oversample
    return framerate
        
def get_hist( cal=False, capture=False ):
    ''' Gets the histogram from the expinfo.  If cal is true, then returns the calibration histogram and overrides the capture flag '''
    if cal:
        cmd = '/software/cmdControl Query ExpInfo | grep \<name\>calbins\<\/name\>'
    else:
        if capture:
            capture_image()
            time.sleep( 0.1 )
        cmd = '/software/cmdControl Query ExpInfo | grep \<name\>bins\<\/name\>'
    output = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE, stderr=devnull ).communicate()[0]

    hist = re.compile( '\<value\>([0-9 ]*)\<\/value\>' )
    bins = [ int(i) for i in hist.search(output).groups()[0].split() ]
    return bins

def get_manifold_pressure():
    resp = cmdquery( 'AllPressures' )
    regex = re.compile( '\<Manifold Pressure\>(.*)\<\/Manifold Pressure\>' )
    return float( regex.search( resp ).groups()[0] )

def get_platform():
    cmd = '/software/platformCmd'
    BBresp = get_BBresp( cmd, platformCmd=True )
    regex = re.compile( 'dc_log:platform = \< (.*) \>' )
    val = regex.search( BBresp ).groups()[0]
    val = val.lower()
    if 'valkyrie' in val or 'genexus' in val:
        return GENEXUS
    elif 's5ruo' in val:
        return S5
    elif 'proton' in val:
        return PROTON
    else:
        print( 'ERROR: Platform not found' )
        raise

def get_save_thumbnail_only():
    ''' Checks if only thumbail data is being saved to disk '''
    cmd = '/software/cmdControl Query WrCntrls | grep wantThumbnailToDisk'
    resp   = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    val = bool( int( re.search( r'\<value\>([0-9]*)\<\/value\>', resp ).groups()[0] ) )
    return val

def get_script_time( BBresp , line1 ):
    """This function parses xml output from the RunScript command to detect how long the Script will take to implement"""
    BBresp = BBresp.replace (line1,'')
    BBresp = BBresp.replace ('\n\ndone\n','')
    resp   = parse_xml  ( BBresp )
    for info in resp.iter('RunScript'):
        return np.ceil( float(info.text) )
    _annotate( 'Unable to determine script time', 0 )
    return 0

def get_scripts():
    """ Reads in list of scripts present on proton machine """
    # Query proton for script list and parse xml response
    BBresp = get_BBresp( "/software/cmdControl Query Scripts" )
    t      = BBresp.replace('REQ Query Scripts','')
    t      = t.replace('\n\ndone\n','')
    resp   = parse_xml ( t )
    for result in resp.iter('QUERY'):
        return result.text.split()

def get_valkyrie_lane():
    ''' Seems to be for proton retrofits --> only works for one lane at a time
        Use get_active_valk_lanes moving forward
        BP 14 Apr 2020
    '''
    resp = cmdquery( 'calAvgAlgo', group='Options' )
    regex = re.compile( '\<value\>(.*)\<\/value\>' )
    return regex.search( resp ).groups()[0]

def get_valve_position():
    ''' Gets the valve position '''
    cmd = '/software/cmdControl Query FpgaValvingState'
    resp   = subprocess.Popen( cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    vs = int( re.search( r'\<FpgaValvingState\>([0-9]*)\<\/FpgaValvingState\>', resp ).groups()[0] )
    return vs
    
def get_vfc():
    ''' checks if VFC is turned on '''
    BBresp = get_BBresp( "/software/cmdControl Query Options | grep VFC" )
    resp         = parse_xml ( BBresp )
    for result in resp.iter( 'value' ):
        if result.text.lower() == 'no':
            return False
        elif result.text.lower() == 'yes':
            return True
    return None

def is_chip_handle_engaged():
    ''' Checks if the manual chip handle is fully engaged '''
    #cmd = "/software/cmdControl Query WrCntrls | grep ChipClampMotorHome"
    cmd = "/software/cmdControl Query WrCntrls | grep ChipClampFullyEngaged"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            if result.text.lower() == 'no':
                return False
            elif result.text.lower() == 'yes':
                return True
        except ValueError:
            continue
    return None

def is_chip_motor_home():
    ''' Checks if the chip motor is home (open) '''
    cmd = "/software/cmdControl Query WrCntrls | grep ChipClampMotorHome"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            if result.text.lower() == 'no':
                return False
            elif result.text.lower() == 'yes':
                return True
        except ValueError:
            continue
    return None

def is_cal_done( starttime=None ):
    ''' 
    Checks if calibration is done using the new CalibrateInProgress query 
    starttime is only used for backwards compatibility for datacollect versions before this was implemented
    '''
    BBresp = get_BBresp( "/software/cmdControl Query CalibrateInProgress" )
    t      = BBresp.replace('REQ Query CalibrateInProgress','')
    t      = t.replace('\n\ndone\n','')
    resp   = parse_xml ( t )
    for result in resp.iter('CalibrateInProgress'):
        try:
            return not bool(int(result.text))
        except ValueError:
            continue
    if starttime is None:
        raise IOError( 'Unable to determine if calibration was completed.  You may need to provide a start time' )
    return is_cal_done_old( starttime )

def get_logtime( l ):
    ''' Parse the datetime from the debug log line.
        You usually want to wrap try/except ValueError
        in case the log line is non-standard
        '''
    return datetime.datetime.strptime(' '.join(l.split()[:3]),'%b %d %H:%M:%S')

def cmp_logtime( l, starttime ):
    ''' Checks if the log is later than the start time '''
    try:
        return get_logtime(l) > starttime
    except ValueError:
        return False

def is_cal_done_old( starttime ):
    """Interprets /var/log/debug to check if chip calibrate passed"""
    log = subprocess.Popen(["grep -a 'Type:ChipCalStatus: Done' /var/log/debug | tail -n 200" ], stdout=subprocess.PIPE, shell=True).communicate()[0]
    log = log.splitlines()
    log = [ l for l in log if cmp_logtime(l,starttime) ]
    if len(log)>0:
        return True
    else:
        return None

def is_door_closed():
    ''' Checks if the door is closed and locked '''
    closed = False
    locked = False

    cmd = "/software/cmdControl Query WrCntrls | grep DoorOpen"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            if result.text.lower() == 'no':
                closed = True
            elif result.text.lower() == 'yes':
                closed = False
        except ValueError:
            continue
        if closed:
            break

    cmd = "/software/cmdControl Query WrCntrls | grep DoorLock"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            if result.text.lower() == 'no':
                locked = False 
            elif result.text.lower() == 'yes':
                locked = True 
        except ValueError:
            continue
        if locked:
            break
    
    return closed and locked

def is_image_captured( imageName, starttime ):
    ''' Similar to is_script_done, but can be used to see if an image has been saved to disk, mid-script '''
    log = subprocess.Popen(["grep -a 'Closing' /var/log/debug | tail -n 200" ], stdout=subprocess.PIPE, shell=True).communicate()[0]
    if log == '':
        _annotate( 'Please check the log, it appears the flow script was not run', 0 )
        return False
    else:
        log = log.splitlines()
        # Reduce to only values after starttime
        log = [ l for l in log if cmp_logtime(l,starttime)  ]
        log = '\n'.join(log)
        if imageName in log:
            _annotate( '\n%s has been captured . . .' % imageName, 1 )
            return True
    return False


def is_reagent_cartridge_present():
    ''' Checks if the reagent cartridge is present '''
    cmd = "/software/cmdControl Query WrCntrls | grep ReagentClampCartridgePresent"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            if result.text.lower() == 'no':
                return False
            elif result.text.lower() == 'yes':
                return True
        except ValueError:
            continue
    return None
            
def is_reagent_motor_closed():
    ''' Checks if the reagent motor is closed (fully engaged) '''
    cmd = "/software/cmdControl Query WrCntrls | grep ReagentClampFullyEngaged"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            if result.text.lower() == 'no':
                return False
            elif result.text.lower() == 'yes':
                return True
        except ValueError:
            continue
    return None
            
def is_reagent_motor_home():
    ''' Checks if the reagent motor is home (open) '''
    cmd = "/software/cmdControl Query WrCntrls | grep ReagentClampMotorHome"
    _annotate( cmd )
    BBresp = get_BBresp( cmd )
    resp   = parse_xml  ( BBresp )
    for result in resp.iter('value'):
        try:
            if result.text.lower() == 'no':
                return False
            elif result.text.lower() == 'yes':
                return True
        except ValueError:
            continue
    return None

def is_script_done( scriptName, starttime ):
    """Interprets /var/log/debug to check if flow script completed"""
    log = subprocess.Popen(["grep -a 'Type:' /var/log/debug | tail -n 200" ], stdout=subprocess.PIPE, shell=True).communicate()[0]
    if log == '':
        _annotate( 'Please check the log, it appears the flow script was not run', 0 )
        return False
    else:
        log = log.splitlines()
        # Reduce to only values after starttime
        log = [ l for l in log if cmp_logtime(l,starttime) ]
        log = '\n'.join(log)
        if 'ValvingComplete' in log and 'Experiment Complete' in log:
            _annotate( '\n%s has now finished . . .' % scriptName, 1 )
            return True
    return False

def is_valving_done():
    ''' 
    Checks if any valving operations are still taking place 
    Returns True or False.  
    If cmdControl does not support the "ValvingRunning" command, then
    None is returned
    '''
    BBresp = get_BBresp( "/software/cmdControl Query ValvingRunning" )
    t      = BBresp.replace('REQ Query ValvingRunning','')
    t      = t.replace('\n\ndone\n','')
    resp   = parse_xml ( t )
    for result in resp.iter('ValvingRunning'):
        try:
            return not bool(int(result.text))
        except ValueError:
            continue
        return None
    
def lock_door( wait=False, timeout=None ): 
    ''' 
    Locks the door.  
    If wait is specified, it waits infinitly for the door to actually be closed
    Returns True/False if door was closed. If wait=False, returns None
    '''
    cmdcontrol( 'LockDoor', 1 )

    if wait or timeout:
        return wait_for_door_closed( timeout=timeout )
    else:
        return None

def open_cartridge( wait=False ): 
    ''' 
    Closes the nuc cartridge.  Will wait up to 20 seconds for the process to complete
    Returns True/False if process was sucessful.
    If state is unknown or process is ongoing, returns None
    '''
    # Check if the cartridge is actually open:
    rmh = is_reagent_motor_home()  # Returns True if motor is open
    rmc = is_reagent_motor_closed()
    if rmh:
        # Cartridge is already open
        return True
    if not ( rmc or rmh ):
        # motor is in an undetermined state
        _annotate( 'Unable to determine motor state.  It is neither open nor closed', 0 )
        return False

    wait_for_valving()
    # Open the clamp
    cmdcontrol( 'OpenCartridge', 1 )
    # Wait and then make sure the chip motor is actually opening
    wait_for_valving()
    time.sleep(5)
    if is_reagent_motor_closed():
        _annotate( 'Unable to open the cartridge clamp!', 0 )
        return False
    else:
        # Motor is not home (closed, moving, or stuck)
        pass

    if wait:
        timeout = 20
        ahora = _now() 
        while not is_reagent_motor_home():
            elapsed = (_now() - ahora).total_seconds()
            if elapsed > timeout:
                _annotate( 'Reagent cartridge did not close in %i seconds' % ( int(timeout) ), 0 )
                return False
            time.sleep( 1 )

        return True

    return None

def open_clamp( wait=False ):
    ''' 
    Opens the chip clamp.  Will wait up to 20 seconds to make sure clamp is opened if desired 
    Returns True or False if we know the process was sucessfull.  If state is unknown or ongoing, returns None
    '''
    if _disable_clamp_open:
        return 

    # make sure all valving processes are done before trying to open the clamp
    ivd = is_valving_done()
    if ivd is None:
        _annotate( 'Unable to determine if it is safe to open the clamp.  Please update datacollect', 0 )
        return False
    elif ivd == False:
        _annotate( 'It is not safe to open the clamp!', 0 )
        return False
    
    ## This didn't work for a few different script attempts.  There's 
    ## someting about how datacollect handles things that I'm missing.
    #
    ## Pressurize before opening the clamp
    #flowscript = 'Script_WT_Pressurize-Before-OpenClamp'
    #expname = 'Pressurize-Before-OpenClamp'
    #waittime, starttime = start_script( flowscript, expname, wait=False )
    #wait_for_script( flowscript, waittime, starttime )

    # Send the clamp open signal
    cmdcontrol( 'OpenClamp', 1 )

    # Wait for clamp to open
    if wait:
        timeout = 20
        ahora = _now() 
        while not is_chip_motor_home():
            elapsed = (_now() - ahora).total_seconds()
            if elapsed > timeout:
                _annotate( 'Chip clamp did not open in %i seconds' % ( int(timeout) ), 0 )
                return False
            time.sleep( 1 )

        return True

    return None

def open_door():
    ''' unlocks the door '''
    cmdcontrol( 'UnlockDoor', 1 )

def passed_cal():
    ''' Checks if chip cal passed by querying /var/log/debug '''
    # Find when calibrate started
    cmd = 'grep -a ChipCalibrateWorker\ started /var/log/debug'
    output = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE ).communicate()[0].strip()
    if not output:
        print 'ERROR! no calibration event detected'
        return False
    lines = output.split( '\n' )
    starttime = get_logtime( lines[-1] )

    # Find when calibrate finished
    cmd = "grep -a 'ChipCal:\ Passed\|ChipCal:\ Failed' /var/log/debug"
    output = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE ).communicate()[0].strip()
    if not output:
        # No result returned yet (or ever)
        return None
    lines = output.split( '\n' )
    endtime = get_logtime( lines[-1] )
    if endtime < starttime:
        # No result returned yet
        return None

    return ( 'Passed' in lines[-1] )

def parse_xml( text ):
    ''' Converts the xml output to grouped fields.
        This adds an outer set of tags if necessary '''
    try:
        return ET.fromstring   ( text )
    except ET.ParseError:    # This fixes the "junk after ..." commands.  Maybe this should be the default method [STP 12/16/2014]
        text = '<data>\n' + text + '</data>'
        try:
            return ET.fromstring   ( text )
        except:
            _annotate( 'ERROR PARSING XML', 0 )
            _annotate( '-'*30, 0 )
            _annotate( traceback.format_exc() )
            _annotate( '-'*30, 0 )
            raise

def set_active_genexus_lanes( value='1234' ):
    ''' Takes a string of lane numbers as input
        The string is parsed to determine which lanes should be set as active
        An integer (0-15) is generated based on the conversion table for the byte register
        The integer is then sent to the datacollect option DbgLaneActive

        4-bit array, integer values
        --> all lanes = 15, no lanes = 0
        Lane 1 == 1
        Lane 2 == 2
        Lane 3 == 4
        Lane 4 == 8
    '''
    lane_dict   = {'1':1,'2':2,'3':4,'4':8}

    if isinstance( value, type([]) ) or isinstance( value, type(tuple) ):
        value = ''.join([str(x) for x in value])
    elif isinstance( value, int ):
        value = str(value)

    # duplicate values would cause problems, might as well handle them
    lanes = set( [ v for v in value ] )
    if len(value)>len(lanes):
        print( 'Error: Duplicate Values' )
        return 0

    lane_bits  = 0
    for l in lanes:
        if l in '1234':
            lane_bits += lane_dict[l]
        else:
            print( 'Error: {} not an accepted lane'.format(l) )
            return 0
    cmdcontrol( 'DbgLaneActive', value=lane_bits, group='Options' )

def set_advCompression( value=None ):
    ''' Controls advanced (PCA?) compression '''
    # P2:  Default to ON.  Turn off with noAdvCompression = True
    # P1:  Default to OFF. Turn on with useAdvCompression = True

    if value is None:
        cmdcontrol( 'useAdvCompression', False, 'Options' )
        cmdcontrol( 'noAdvCompression',  False, 'Options' )
    elif value == True:
        cmdcontrol( 'useAdvCompression', True,  'Options' )
        cmdcontrol( 'noAdvCompression',  False, 'Options' )
    elif value == False:
        cmdcontrol( 'useAdvCompression', False, 'Options' )
        cmdcontrol( 'noAdvCompression',  True,  'Options' )

def set_chipmode( value=0 ):
    # Command seems to have ch
    if DATACOLLECT >= 3611: # maybe earlier
        cmdcontrol( 'ChipModeOverRide_p1', value, 'Options' )
        cmdcontrol( 'ChipModeOverRide_p2', value, 'Options' )
    else:
        cmdcontrol( 'ChipModeOverRide', value, 'Options' )

def set_dac( voltage ):
    ''' Sets the fluidic bias '''
    voltage = '{:0.6f}'.format( voltage )
    cmdcontrol( 'dac', voltage, 'WrCntrls' )
# Convienience rename
set_vref = set_dac

def set_dac_start_sig( value ):
    ''' Sets DSS '''
    cmdcontrol( 'RangeStart', value, 'WrCntrls' )
# Convienience rename
set_dss = set_dac_start_sig

def set_dr( target=0.3 ):
    ''' Sets the dynamic range (in V) '''
    cmdcontrol( 'DynamicRange', group='WrCntrls', value=target )

def set_efuse( value='' ):
    ''' Sets the efuse override'''
    cmdcontrol( 'OverrideEfuse', value, 'Options' )

def set_save_thumbnail_only( value=False ):
    ''' Only save thumbnail data to disk.  Not full chip. (datacollect > 3431) '''
    cmdcontrol( 'wantThumbnailToDisk', int(value), 'WrCntrls' )

def set_sampling( rate=None, high=None ):
    ''' Set sampling rates
    rate: integer for oversampling
    high: boolean for high sampling rate
    '''
    if high is not None:
        cmdcontrol( 'highSampleRate', str(high).lower(), 'Options' )
    if rate is not None:
        cmdcontrol( 'OverSample', int(rate), 'Options' )

def set_servicemode( value=True ):
    cmdcontrol( 'ExpertFeatures', value, 'Options' )

def set_p1_emulation( value=False ):
    ''' Emulate P2s as a P1 for readout '''
    cmdcontrol( 'EmulateP1.1', value, 'Options' )

def set_pinchreg_flowrate( reg, fc, rate ):
    ''' set flowrates on the pinch regulators
    reg:  main or chip
    fc:   F - Full Flowcell
          QF - Quarter Flowcell
    rate: integer flow rate (mL/min)
    '''
    field = 'VarReg_%s%s' % ( reg.capitalize(), fc.upper() )
    cmdcontrol( field, int(rate), 'Options' )

def set_pressure( pres ):
    ''' Sets the regulator pressure '''
    cmdcontrol( 'pres', '%0.1f'%pres, 'WrCntrls' )

def set_refpix( rpo ):
    ''' Sets the ref_pix_override property.
    Caution, this is only applied at chip-init.  it does 
    not change the real-time value '''
    cmdcontrol( 'ref_pix_override', group='Options', value=rpo )

def set_rfid_interlock( value=False ):
    cmdcontrol( 'checkRfid', value, 'Options' )

def set_t0Compression( value=None ):
    ''' Controls T0 compression '''
    if value is None:
        cmdcontrol( 'useT0Compression', False, 'Options' )
        cmdcontrol( 'noT0Compression',  False, 'Options' )
    elif value == True:
        cmdcontrol( 'useT0Compression', True,  'Options' )
        cmdcontrol( 'noT0Compression',  False, 'Options' )
    elif value == False:
        cmdcontrol( 'useT0Compression', False, 'Options' )
        cmdcontrol( 'noT0Compression',  True,  'Options' )

def set_temps( manifold=None, tec=None ):
    if manifold is not None:
        cmdcontrol( 'ManifoldHeaterTemperature', int(manifold), 'Options' )
    if tec is not None:
        cmdcontrol( 'ChipTECTemperature', int(tec), 'Options' )

def set_valkyrie_lane( lane ):
    ''' Seems to be for proton retrofits --> only one lane at a time 
        Use set_active_valk_lanes moving forward
        BP 14 Apr 2020
    '''
    cmdcontrol( 'calAvgAlgo', group='Options', value=lane )

def set_valve_position( code ):
    ''' Sets the valve position using the FpgaValvingState '''
    cmdcontrol( 'FpgaValvingState', int(code) )

def set_vfc( value=False ):
    ''' Turns VFC on or off '''
    cmdcontrol( 'VFC', value, 'Options' )

def set_waste_interlock( value=False ):
    cmdcontrol( 'checkWaste', value, 'Options' )

def start_datacollect():
    cmd = 'sudo /software/StartDatacollect'
    subprocess.call( cmd, shell=True )
    time.sleep( 10 )

def start_cal( flag, wait=True, keepalive=False ):
    ''' 
    Runs the specified calibration and waits for it to complete
    Flags:
        0      Fast Cal
        6      Standard Cal
        16     2 frame noise measurement and calculation
        32     1 frame aquisition ( saves to CalCaptureFrame )
        128    Dry Bit (use in conjunction with other cals) 
        2048   ExpCal
        32786  Keep-alive bit.  Keep the chip powered on after cal finishes
    '''
    # 9/2/2016 - Mark B says I need these two lines for P2 chips.  Otherwise the chip might not be ready and calibrate enters a "wierd state"
    wake_chip( wait=False )
    time.sleep(1)

    starttime = _starttime()
    if keepalive:
        bits = '{0:016b}'.format(flag)
        if not(int(bits[-16])):
        #bit = (16**3) * 8 #0x8000
        #if flag < bit: # Not the most robust
            if DATACOLLECT >= 3611:
                flag += (1<<(16-1))
    cmdcontrol( 'StartCalibrate', int(flag) )
    if wait:
        wait_for_cal( starttime )
    return starttime

def start_script( scriptName, expName='', N=1, wait=True ):
    starttime = _starttime()
    BBresp = get_BBresp( "/software/cmdControl Set RunScript %s %i %s" % ( scriptName, N, expName ) )
    ln1    = 'REQ Set RunScript %s:%i' % ( scriptName , N )
    if expName:
        ln1 += ':' + expName

    waittime = get_script_time( BBresp, ln1 )
    if not wait:
        return waittime, starttime

    wait_for_script( scriptName, waittime, starttime )

def wait_for_cal( starttime=None, maxtime=90 ):
    ''' 
    Waits for the calibration to complete, up to the max time (in seconds) 
    maxtime is only used if the we have to determine the calibration state from /var/log/debug
    '''
    if starttime is None:
        starttime = _now()

    while True:
        icd = is_cal_done( starttime )
        if icd:
            _annotate( '', 1 )
            return True
        elif icd is None:
            elapsed = (_now() - starttime).total_seconds()
            if elapsed > maxtime:
                _annotate( 'Calibration completion was not completed in the allocated time. Either it has failed, the datacollect syntax has changed, or a something has gone wrong', 0 )
                return False
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
        time.sleep( 5 )

def wait_for_chip_ready( value, timeout=40 ):
    ''' Waits for the chip to reach the minimum status '''
    ahora = _now() 
    while chip_ready() < value:
        elapsed = (_now() - ahora).total_seconds()
        if elapsed > timeout:
            _annotate( 'Chip did not reach ready status of %i in %i seconds' % ( int(value), int(timeout) ), 0 )
            return False
        time.sleep( 1 )

    return True

def wait_for_door_closed( timeout=None ):
    ''' Waits for the chip to reach the minimum status '''
    ahora = _now() 
    while not is_door_closed():
        if timeout is not None:
            elapsed = (_now() - ahora).total_seconds()
            if elapsed > timeout:
                _annotate( 'Door not closed in %i seconds' % ( int(timeout) ), 0 )
                return False
        time.sleep( 2 )

    return True

def wait_for_valving( timeout=20 ):
    ''' Waits for the chip to reach the minimum status '''
    ahora = _now() 
    while not is_valving_done():
        elapsed = (_now() - ahora).total_seconds()
        if elapsed > timeout:
            _annotate( 'Valving did not complet in %i seconds' % ( int(timeout) ), 0 )
            return False
        time.sleep( 1 )

    return True

def wait_for_script( scriptName, waittime, starttime, flex=60, interval=5 ):
    ''' Waits for the specifed script to complete.  Since the chiptype impacts the script time, this buffers
    for up to the specified flex time '''
    if interval <=0:
        print( 'interval must be an integer number of seconds greater than 0' )
        raise
    loops = int(flex/interval)

    _annotate( '\tThis script will complete in about %i seconds, please wait . . .' % waittime, 1 )
    _time_progress( waittime , starttime=starttime, start='\t' , current=scriptName)

    if is_script_done ( scriptName, starttime ):
        pass
    else:
        # This will loop for up to 1 minute
        for i in range( loops ):  # If you change this, don't forget to change the elif below as well:
            _annotate( '\tWaiting %s more seconds for flow script to complete . . .' % interval, 2 )
            _time_progress( float(interval) , starttime=starttime, start='\t' , current=scriptName)
            if is_script_done ( scriptName, starttime ):
                break
            elif i == loops-1:
                # Due to various datacollect versions and the typical robustness of flow scripts, removing the doublecheck for flow scripts to finish.
                _annotate( '\n\tCannot determine if flow script was completed.  \n\tIf using a modified datacollect, you can probably ignore this message.  Otherwise, be aware that it may or may not have finished.  If you are doing wideband noise script analysis, it could kill datacollect before acquisition is finished and kill off your last acquisition.............' , 0 )

def wake_chip( wait=True, keep_fully_on=True ):
    ''' Turns on the chip '''
    cr = chip_ready()
    if cr > 1:
        return
    elif cr == 0:
        raise IOError( 'Chip clamp is open' )
    if keep_fully_on and DATACOLLECT >= 3612:
        flag = 3
    else:
        flag = 1
    cmdcontrol( 'wantChipEnabled', flag, 'WrCntrls' )
    if wait:
        wait_for_chip_ready( 3 )

def _progress_bar( progress , start='' , current='' , spacer='' ):
    """ 
    Writes a progress bar to the terminal during ftp transfer 
    progress is a float between 0 and 1
    start lets you precede the line with something....like a \t constant
    current lets us tag an ending onto the line, e.g. showing which file or directory is being transferred
    """
    barLength = 20
    status    = ''
    tail      = ''
    if isinstance( progress , int ):
        progress = float( progress )
    if not isinstance( progress , float ):
        progress = 0.
        status   = "Error: progress var must be a float"
    if progress < 0:
        progress = 0.
        status   = "Halt...\r\n"
    if progress >= 1:
        progress = 1.
        status   = "Done...               \r\n"
    block = int( round( barLength*progress ) )
    if current != '':
        tail = ' | %s' % current
    if progress == 1.:
        tail   = ''
        spacer = ''
    text  = '\r{0}Percent:[{1}] {2:.0f}% {3}{4}{5}'.format( start , "#"*block + "-"*( barLength-block ), progress*100, status , tail , spacer )

    if verbose > 3:
        print( text )
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
    return None

def _progress_bar_basic():
    sys.stdout.write('.')
    sys.stdout.flush()

def _progress_bar_timed():
    sys.stdout.write('.')
    sys.stdout.flush()
    
def _now():
    return _starttime()

def _starttime():
    ''' Gets the starttime for the measurement '''
    starttime = datetime.datetime.now()
    return starttime.replace(year=1900)   # This is because string parsingin IsScriptDone automatically set year=1900

def _time_progress( wait_time , starttime=None, start='' , current='' , spacer='' , cal=False ):
    """ Runs ProgressBar with a timing input """
    if verbose > 2:
        if wait_time > 120:
            current += ' | ~%0.1f minutes' % ( wait_time/60. )
        else:
            current += ' | ~%0.1f total seconds' % wait_time
    for i in range( int(wait_time) ):
        # If we're running this interactively, we want to see progressbar, otherwise no!
        time.sleep( 1 )
        if verbose > 1:
            _progress_bar( float( i / ( wait_time ) ) , start , current , spacer )
        elif verbose == 1:
            _progress_bar_basic()

        if cal:
            if i in [ 29 , 44 , 59 , 74 , 89 , 104 ]:
                if is_cal_done( starttime ):
                    time.sleep( 4 )
                    _annotate( '\tCalibration done early!  Moving on . . .', 2 )
                    break

    _annotate( 'Verbosity: %i' % verbose )
    if verbose > 1 :
        _progress_bar( 1. , start , current )
    elif verbose == 1:
        _progress_bar_basic()

# Determine datacollect version so newer features can be implemented
def check_cmdcontrol():
    try:
        return not subprocess.call( ['/software/cmdControl', 'Query'] )
    except OSError:
        return False

_ctr = 0
_ctrlmt = 2
DATACOLLECT = 0
while _ctr < _ctrlmt:
    try:
        DATACOLLECT = int( get_datacollect_version() )
        _ctr = _ctrlmt
    except KeyError:
        if not check_cmdcontrol():
            print( 'WARNING: cmdControl not detected!' )
            break
        if not datacollect_running():
            start_datacollect()
        _ctr += 1
    except KeyboardInterrupt:
        raise
    except:
        print( 'ERROR READING DATACOLLECT VERSION' )
