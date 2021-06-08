import os, subprocess, time, argparse

class LockExists( Exception ):
    pass

class LockFile:
    ''' Create a lock file
        filename: the name of the lock file to create
        reserve:  Keep attempting to get the lock until you have it.  Otherwise, it raises an error
        procname: Name of the current process.  It is advisable to set this to the parent python process
        pid:      PID of the parent process.  This is normally auto-detected, so only set this if you know what you're doing
    '''
    filename = '' # you should normally override this
    procname = 'python' # You should normally set this to os.path.basename(__file__) of the parent processes
    reserve_interval = 1 # second
    pid = None
    user = None # Assign the lock file to the following user once its created (requires root access)
    group = None
    created = False

    def __init__( self, filename=None, reserve=False, procname=None, pid=None, onlycheck=False ):
        if filename is not None:
            self.filename = filename
        if procname is not None:
            self.procname = procname
        if pid is not None:
            self.pid = pid
        self.onlycheck = onlycheck
        self.created = False

        if not os.path.exists( self.filename ):
            self.create()
            return
        self.validate( reserve=reserve )

    def create( self ):
        if self.onlycheck: 
            return
        self.msg = '{} {}'.format( self.get_pid(), self.procname )
        with open( self.filename, 'w' ) as lock:
            lock.write( self.msg )
        self.created = True
        if self.user:
            cmd = [ 'chown', self.user, self.filename ]
            subprocess.call( cmd )
        if self.group:
            cmd = [ 'chgrp', self.group, self.filename ]
            subprocess.call( cmd )


    def validate( self, reserve=False ):
        with open( self.filename ) as lock:
            pid, program = lock.read().strip().split( ' ', 1 )
        if pid == str(self.pid) and program == self.procname:
            # Looks like we're trying to pick up a lock file that we lost the reference to
            return
        cmd = [ 'ps', pid ]
        resp = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]
        if program in resp:
            if reserve:
                time.sleep( self.reserve_interval )
                self.validate( reserve=reserve )
            else:
                raise LockExists( 'Lock file {} exists for {} ({})'.format( self.filename, program, pid ) )
        self.delete()
        self.create()

    def delete( self ):
        os.remove( self.filename )

    def get_pid( self ):
        if self.pid:
            return self.pid
        else:
            return os.getpid()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--pid', required=True, type=int )
    parser.add_argument( '--filename', required=True )
    parser.add_argument( '--procname', required=True )
    parser.add_argument( '--delete', action='store_true' )
    parser.add_argument( '--reserve', action='store_true' )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.delete:
        lock = LockFile( filename=args.filename, 
                         pid=args.pid, 
                         procname=args.procname,
                       )
        lock.delete()
    else:
        lock = LockFile( filename = args.filename, 
                         pid      = args.pid, 
                         procname = args.procname,
                         reserve  = args.reserve, 
                       )

