import os
import subprocess

class ping:
    def __init__( self, server, count=1, auto=True, parse=False ):
        self.server = server
        self.count  = 1
        self.parse  = parse
        if auto:
            self.ping()

    def ping( self ):
        devnull = open( os.devnull, 'w' )
        cmd = ' '.join( [ 'ping', '-q', '-c', str(self.count), self.server ] )
        self.response = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=devnull, shell=True ).communicate()[0]
        self.found = bool(self.response)

        if self.parse:
            self._parse_response()

    def _parse_response( self ):
        self._get_ip()
        self._get_times()

    def _get_ip( self ):
        try:
            for line in self.response.split('\n'):
                if 'PING' not in line:
                    continue
                line = line.split('(')[1]
                line = line.split(')')[0]
                self.ip = line
        except IndexError:
            self.ip = None

    def _get_times( self ):
        try:
            for line in self.response.split('\n'):
                if 'min/avg/max/mdev' not in line:
                    continue
                times = line.split('=')[1].split()[0].split('/')
                self.mintime = float( times[0] )
                self.avgtime = float( times[1] )
                self.maxtime = float( times[2] )
                self.mdev    = float( times[3] )
                return
        except IndexError:
            pass
        self.mintime = None
        self.avgtime = None
        self.maxtime = None
        self.mdev    = None

def get_domain():
    ''' Guesses the current domain for transfering to servers 
    Appends the domain to the server if necessary '''
    domains = [ 'ite', 'itw', 'cbd', 'home' ]
    # read from resolv.conf
    resp = subprocess.Popen("more /etc/resolv.conf | grep 'domain'", stdout=subprocess.PIPE, shell=True).communicate()[0]
    if resp != '':
        domain = resp.split()[1]
        if domain in domains:
            return domain

    # Find the closest domain
    pings   = [ ping( 'ecc.' + d, parse=True ) for d in domains ]
    times   = [ 9999 if p.avgtime is None else p.avgtime for p in pings ]
    if min(times) != 9999:
        domain = domains[ times.index( min( times ) ) ]
        return domain
    raise OSError( 'Unable to determine domain' )

