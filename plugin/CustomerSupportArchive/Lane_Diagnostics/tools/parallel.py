'''
Functions for parallel computing
This has functions which run on the annotate framework
'''
import subprocess, os
try:
    from . import annotate
except:
    class annotate:
        @staticmethod
        def write( msg, level=1 ):
            print( msg )
from time import sleep, time

def stripe( commands, notes=None, numprocs=1, capture_output=True, timeout=None, proclimit_file=None, procactive_file=None ):
    ''' 
    Runs the specified commands specified in parallel 
    commands is a list of lists to input into subprocess.call
    Notes, if specified, is a list of text to display for each new command.
    printcommands determines if the command is printed as it executes

    proclimit_file is an optional argument specifiying a filename to monitor to dynamically throttle down the 
           number of utilized processes.  The contents of that file should be a single integer indicating
           number of processes
    procactive_file is an optional argument used to denote the number of currently open processes
    

    This uses annotate, so a log file can be specified with annotate.set_log to 
    save the output
    '''
    # Set up notes fields if unspecified
    if notes is None:
        notes = [ '' for c in commands ]

    # Run each command, maximzing the number of processes run
    processes = []
    try:
        proclimit = min( numprocs, int( open( proclimit_file ).read().strip() ) )
    except:
        proclimit = numprocs
    for (cmd,note) in zip( commands, notes ):
        if note:
            annotate.write( note, 2 )
        # OK, I'll also allow strings, but I won't like it
        try:
            isstring = isinstance( cmd, basestring ) # python2
        except:
            isstring = isinstance( cmd, str ) #python3
        if capture_output:
            output = subprocess.PIPE
        else:
            output = open( '/dev/null', 'w' )
        if isstring:
            processes.append( subprocess.Popen( cmd, shell=True, stdout=output, stderr=output ) )
            processes[-1].calling = '...COMMAND: %s' % cmd # We set this here, but it isn't output until after the analysis completes
            processes[-1].starttime = time()
        else:   # Run the command in list form
            processes.append( subprocess.Popen( cmd, stdout=output, stderr=output ) )
            processes[-1].calling = '\n...COMMAND: %s' % ' '.join(cmd) # We set this here, but it isn't output until after the analysis completes
            processes[-1].starttime = time()

        if procactive_file:
            with open( procactive_file, 'w' ) as paf:
                paf.write( str( len( processes ) ) )

        # Add another one if queue is not full.  
        if len(processes) < proclimit:
            continue

        # If full, wait for one to finish
        one_is_done = False
        # Check and remove completed tasks
        while not one_is_done:
            # Scan all processes to see if one is done
            ahora = time()
            for i in range(len(processes)):
                # True status means process is complete.  If incomplete, None is returned
                toolong = timeout and ( ahora - processes[i].starttime ) > timeout
                if processes[i].poll() is not None or toolong:
                    p = processes.pop(i)        # Pop doesn't work well with loops, so if one is found, stop looking and kick off the next one
                    annotate.write( p.calling, 2 )
                    if toolong:
                        try:
                            annotate.write( '  KILLING PROCESSES... Process did not complete in %i seconds' % timeout )
                            p.terminate()
                        except OSError:
                            annotate.write( '                       Process seems to already be dead'  )
                    if capture_output:
                        output = p.stdout.read()
                        if output.strip():
                            try:
                                outstr = output.replace( '\n', '\n    ' )
                            except TypeError:
                                outstr = output.decode('utf8').replace( '\n', '\n    ' )
                            annotate.write( '  ' + outstr, level=3 )
                    one_is_done = True
                    break   # for; also exits one_is_done
            if procactive_file:
                with open( procactive_file, 'w' ) as paf:
                    paf.write( str( len( processes ) ) )
            if proclimit_file:
                try:
                    proclimit = min( numprocs, int( open( proclimit_file ).read().strip() ) )
                    if proclimit <= len( processes ) and one_is_done:
                        # we finished, but a throttling request has been made, so stay in the one_is_done loop
                        annotate.write( 'Active throttling detected.  Not launching a new instance' )
                        one_is_done = False
                except:
                    # The proclimit file is currently not active.  stop throttling.
                    proclimit = numprocs

            # Wait to query again
            sleep( .1 )

    # Wait for remaining processes to finish
    while len( processes ):
        #Wait for one to finish
        one_is_done = False
        while not one_is_done:
            # Scan all processes to see if one is done
            ahora = time()
            for i in range(len(processes)):
                # True status means process is complete.  If incomplete, None is returned
                toolong = timeout and ( ahora - processes[i].starttime ) > timeout
                if processes[i].poll() is not None or toolong:
                    p = processes.pop(i)        # Pop doesn't work well with loops, so if one is found, stop looking and kick off the next one
                    annotate.write( p.calling, 2 )
                    if toolong:
                        try:
                            annotate.write( '  KILLING PROCESSES... Process did not complete in %i seconds' % timeout )
                            p.terminate()
                        except OSError:
                            annotate.write( '                       Process seems to already be dead'  )
                    if capture_output:
                        output = p.stdout.read()
                        if output.strip():
                            try:
                                outstr = output.replace( '\n', '\n    ' )
                            except TypeError:
                                outstr = output.decode('utf8').replace( '\n', '\n    ' )
                            annotate.write( '  ' + outstr, level=3 )
                    one_is_done = True
                    break   # while not one_is_done
            if procactive_file:
                with open( procactive_file, 'w' ) as paf:
                    paf.write( str( len( processes ) ) )
            # Wait to query again
            sleep( .1 )
        
    wait_for_procs(processes)
    if procactive_file:
        try:
            os.remove( procactive_file )
        except OSError:
            pass

def wait_for_procs( processes ):
    """ Waits for a list of processes to complete before going on. Useful for multithreading. """
    status = [ False ] * len( processes )
    while all( status ) == False:
        sleep( 1 )
        for j in range( len(processes) ):
            # True status means process is complete.  If incomplete, None is returned.
            status[j] = ( processes[j].poll() != None )
    
    return None
