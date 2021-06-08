'''
Module for performing logging operations
After you should usually overwrite logLevel and logFile with desired defaults
'''
try:
    from cStringIO import StringIO 
except ModuleNotFoundError:
    from io import StringIO
import os as _os
import csv as _csv
import sys as _sys
import ctypes as _ctypes
import tempfile as _tempfile
import subprocess as _subprocess
try:
    from .figures import messenger as _msg
except ImportError:
    print( 'Unable to import annotate.figures (used for wettest)' )
import datetime

################################################################################
# Define global constants -----------------------------------------------------#
################################################################################
logLevel = 1
logFile  = '' 
syslog   = ''

imgdest  = None
imgsrc   = None
imgname  = None  # for makefig

_libc     = _ctypes.CDLL( None )
_c_stdout = _ctypes.c_void_p.in_dll( _libc, 'stdout' )

broadcastUser = ''
broadcastTty  = ''

def setup( logLevel=None, logFile=None, syslog=None,
           imgdest=None, imgsrc=None, imgname=None,
           broadcastUser=None, broadcastTty=None ):
    ''' 
    1-line function to make setting global variables easier
    '''
    inputs = locals()
    thismodule = _sys.modules[__name__]
    for key, value in inputs.iteritems():
        if value is not None:
            # This modifies global values even though I didn't call global
            setattr( thismodule, key, value )
    
################################################################################
# Main annotation function ----------------------------------------------------#
################################################################################
def set_log( newLogLevel=None, newLogFile=None ):
    ''' 
    sets the logging parameters for the instance 
    '''
    global logLevel, logFile
    if newLogLevel is not None:
        logLevel = newLogLevel
    if newLogFile is not None:
        logFile = newLogFile

def write( text, level=1, _logLevel=None, _logFile=None, to_syslog=True ):
    """
    NOTE: IT IS USUALLY BETTER TO USE self.annotate TO AUTOMATICALLY SET _logLevel. IF NOT, THE MODULE LOG LEVEL WILL BE USED (999)
    This function both writes text to the terminal window (ala print) but also saves it to a log file.
    This calls global variables:
        logLevel (integer)
        logFile  (string)
    If level <= logLevel, the message is printed to screen.  
    if level = -1, the message is only printed to screen and not to file

    _logLevel and _logFile is used to override system defaults.  This is usually only used when calling annotate from a class which might have a different level.
    """
    if _logLevel is None:
        _logLevel = logLevel

    if level <= _logLevel:
        print ( text )
        
    if _logFile is None:
        _logFile = logFile

    # Write the text to the logFile, if set
    if _logFile and _logLevel>=0 and level is not None:
        try:
            log = open ( _logFile , 'a' )
            log.write  ( '%s\n' % text )
            log.close  ()
        except IOError:
            pass

    if syslog and to_syslog:
        try:
            with open( syslog, 'a' ) as log:
                ahora = datetime.datetime.now().strftime( '%Y-%m-%d %H:%M:%S - ' )
                log.write( ahora )
                log.write( text.strip().replace( '\n', '<br>' ) )
                log.write( '\n' )
        except IOError:
            pass

    return None

def write_list( arr, *args, **kwargs ):
    ''' Pass-through to (l)write, taking array input instead of string '''
    for txt in arr:
        write( txt, *args, **kwargs )

def broadcast( msg, user=None, tty=None ):
    ''' Sends the specified message to the user terminal '''
    broadcast_all = False
    if not user:
        user = broadcastUser
    if not tty:
        tty  = broadcastTty
    if not ( user and tty ):
        write( 'No user or tty specified for broadcasting:', 0 )
        broadcast_all = True
    write( msg, 0 )

    devnull = open( _os.devnull, 'w' )
    if broadcast_all:
        cmd = 'echo "%s" | wall' % ( msg )
    else:
        cmd = 'echo "%s" | write %s %s' % ( msg, user, tty )
    _subprocess.Popen( cmd, shell=True, stdout=devnull, stderr=devnull )

def input( msg ):
    ''' Pass-through for raw_input. '''
    resp = raw_input( msg )
    print( '' )
    return resp

################################################################################
# Image generation ------------------------------------------------------------#
################################################################################
def write_image( img, abspath=False, live=False ):
    ''' Writes the image by copying to the specified location.
    If live is selected, a blocking image is opened using EOG '''
    if img is None:
        return
    if imgdest is None:
        return
    if imgsrc is None or abspath or img[0]=='/':
        filename = img
    else:
        filename = _os.path.join( imgsrc, img )
    cmd = [ 'ln', '-s', filename, imgdest ]
    devnull = open( _os.devnull, 'w' )
    _subprocess.Popen( cmd, stdout=devnull, stderr=devnull )

    if live:
        cmd = [ 'eog', filename ]
        _subprocess.call( cmd, stdout=devnull, stderr=devnull )

def messenger( func, *args, **kwargs ):
    ''' Pass-through to messenger, saving the resultant image and applying write_image 
    func (string) is the name of the messenger function
    '''
    filename = imgname
    getattr( _msg, func )( *args, **kwargs ).save( filename )
    devnull = open( _os.devnull, 'w' )
    _subprocess.Popen( 'chown ionadmin:ionadmin %s' % filename, shell=True, stdout=devnull, stderr=devnull ) # hopefully no race conditions here
    write_image( filename )

################################################################################
# Annotation tools ------------------------------------------------------------#
################################################################################
class Capturing( list ):
    ''' 
    Context manager to capture standard out, including that from c-libraries

    Standard usage:
    with Capturing as output:
      BLOCK

    This translates to 
    output.__enter__()
    try:
        BLOCK
    finally:
        output.__exit__()

    Adapted from:
        http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    '''
    def __enter__( self ):
        ''' Run at beginning of with statement '''
        # The original fd to which stdout usually points
        # Usually this is 1 for POSIX systems
        self._original_stdout_fd = _sys.stdout.fileno()
        # Save a copy of the original stdout fd in _saved_stdout_fd
        self._saved_stdout_fd = _os.dup( self._original_stdout_fd )
        # Create a temporary file and redirect stdout to it
        self._tfile = _tempfile.TemporaryFile( mode='w+b' )
        self._redirect_stdout( self._tfile.fileno() )
        return self


    def __exit__( self, *args ):
        ''' Run at end of with statement '''
        # Send all output back to stdout instead of the temp file
        self._redirect_stdout( self._saved_stdout_fd )
        # Copy contents of temporary file to the given list
        self._tfile.flush()
        self._tfile.seek( 0, _os.SEEK_SET )
        try:
            self.extend(  self._tfile.read().split('\n') ) # Python 2
        except TypeError:
            self.extend(  self._tfile.read().decode().split('\n') ) # Python 3
        self._tfile.close()

    def _redirect_stdout( self, to_fd ):
        ''' Redirect stdout to the given file descriptor '''
        # Flush the C-level buffer stdout
        _libc.fflush( _c_stdout )
        # Flush and close _sys.stdout.  Also closes the file descriptor (fd)
        _sys.stdout.close()
        # Make _original_stdout_fd point to the same file as to_fd
        _os.dup2( to_fd, self._original_stdout_fd )
        # Create a new _sys.stdout that points to the redirected fd
        _sys.stdout = _os.fdopen( self._original_stdout_fd, 'w' ) 

    def write( self, level=1, indent=0, char=' ', _logLevel=None, _logFile=None ):
        '''
        Writes the contents of this to annotate.write_list, displaying 
        to output and possibly to file
        level:   Logging level
        indent:  Number os characters to indent
        char:    Character to use while indenting
        '''
        prefix = ''
        if indent:
            prefix = char*indent

        arr = [ prefix+v for v in self ]
        if len(arr):
            arr = arr[:-1]  # For some reason the last element is always empty
        write_list( arr, level=level, _logLevel=_logLevel, _logFile=_logFile )

class PyCapture( list ):
    ''' Python capture function (Wont'capture C output.  More compatible from GUI)'''
    def __enter__( self ):
        self._stdout = _sys.stdout
        _sys.stdout = self._stringio = StringIO()
        return self

    def __exit__( self, *args ):
        self.extend( self._stringio.getvalue().splitlines())
        _sys.stdout = self._stdout

    def write( self, level=1, indent=0, char=' ', _logLevel=None, _logFile=None ):
        '''
        Writes the contents of this to annotate.write_list, displaying 
        to output and possibly to file
        level:   Logging level
        indent:  Number os characters to indent
        char:    Character to use while indenting
        '''
        prefix = ''
        if indent:
            prefix = char*indent

        arr = [ prefix+v for v in self ]
        if len(arr):
            arr = arr[:-1]  # For some reason the last element is always empty
        write_list( arr, level=level, _logLevel=_logLevel, _logFile=_logFile )

