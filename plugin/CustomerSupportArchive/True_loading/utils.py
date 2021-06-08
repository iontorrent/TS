import sys
import os

from subprocess import call 

def log_silent(message):
    """ silent log """
    pass

def log(message):
    """ print log message to stout; change plugins"""
    sys.stderr.write( message + '\n' )
    sys.stderr.flush()


def log_file( message, fname ):
    """ appends a message into a file """
    with open(fname, 'a') as f:
        f.write( message + '\n' )

def error_and_exit(message):
    """ critical error 'handler' -- change to self.log.exception for plugins"""
    sys.stderr.write( "ERROR " + message + '\n' )
    sys.stderr.write( "  (exiting)\n" )
    sys.exit(1)


def check_filename(fname):
    """ Returns True is filename exists. Rases an exception/exits otherwise """
    try:
        with open(fname):
            return True
    except IOError as err:
        error_and_exit( "'%s' file does not exist" % (fname) ); 


def run_silent(command):
    """ executes a shell command line in a silent mode"""
    try:
        retcode = call(command, shell=True)
        if retcode !=0:
            log( "WARNING return code = %i" % retcode )
    except OSError:
        error_and_exit("executing '" + command + "'")

def run(command):
    """ executes a shell command line """
    log( command )
    run_silent( command )



def strip_extension(path):
    """ strip extension (if any) from the base path """
    if os.path.basename(path).find('.') > 0:
        return path[ :path.rfind('.') ]
    else:
        return path


def get_basename(path):
    """ strips extension from the filename """
    filename = os.path.basename(path)
    return filename[ :filename.rfind('.') ]
