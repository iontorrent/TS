from functools import wraps
import time

def timeit( f ):
    ''' Decorator for measuring time to execute a function '''
    @wraps(f)
    def decorated( *args, **kwargs ):
        fname = f.__name__
        ahora = time.time()
        timer = Timer( 'TIMEIT ({})'.format( fname ) )
        try:
            return f( *args, **kwargs )
        finally:
            timer.elapsed()
    return decorated


class Timer:
    ''' Class for measuring time performance, both continously and interrupted '''
    def __init__( self, name='ELAPSED', pause=False ):
        ''' Create a new timer.
            Name:   This name will be displayed when printing output
            pause:  True/False - If true, initialize the timer, but don't start it 
        '''
        self.start    = time.time()
        self.lapstart = self.start
        self.name     = name
        self.total    = 0
        self.paused   = pause
        out( '{}: starting'.format( name ) )

    def elapsed( self, msg='' ):
        ''' Mark the time since one of the following:
              elapsed()
              total_elapsed()
              absolute_elapsed()
              unpause()
              __init__
            display the time, along with the timer name and optional message
        '''

        self._update()
        if msg:
            msg = ' ({})'.format( msg )
        out( '{}{}: {:0.2f} seconds'.format( self.name, msg, self.lap ) )

    def total_elapsed( self ):
        ''' Output the total UNPAUSED time  '''
        self._update()
        out( '{}: {:0.2f} seconds'.format( self.name, self.total ) )

    def absolute_elapsed( self ):
        ''' Output the total time SINCE INITIALIZATION '''
        ahora = time.time()
        total = ( ahora - self.start )
        out( '{}: {:0.2f} seconds'.format( self.name, total ) )

    def pause( self ):
        ''' Pause the timer '''
        self._update()
        self.paused = True

    def unpause( self ):
        ''' Unpause the timer '''
        if self.paused:
            ahora = time.time()
            self.lapstart = ahora
            self.paused = False

    def _update( self ):
        if self.paused:
            return
        ahora = time.time()
        self.lap = ( ahora - self.lapstart )
        self.total += self.lap
        self.lapstart = ahora


def out( msg ):
    ''' Basic print statement
        this can be overwritten by other modules by doing:
        tools.timers.out = <new out function>
    '''
    print( msg )
