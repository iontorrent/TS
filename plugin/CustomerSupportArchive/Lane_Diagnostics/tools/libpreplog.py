import datetime, os, re, csv

import numpy as np
import matplotlib
matplotlib.use( 'agg', warn=False )
import matplotlib.pyplot as plt
import scipy.stats as stats

class LibPrepLog( object ):
    def __init__( self, path=None ):
        # must be run first
        self.init_path( path )

        if self.found:
            self.load()
            self.parse()
        else:
            print( 'ERROR: the file {} was not found'.format( self.log ) )

    def init_path( self, path=None ):
        ''' Determine if file exists and load if it does '''
        # Initialize values
        self.path   = path
        self.lines  = []
        
        self.log    = None
        self.found  = False

        filename    = 'libPrep_log.csv'

        if self.path is None:
            print( 'ERROR: path is None.  Please specify a path where {} can be found'.format( filename ) )
            return
        
        # 
        self.log = os.path.join( self.path, filename )
        if os.path.exists( self.log ):
            self.found = True

    def load( self ):
        hdr     = []
        lines   = []
        with open( self.log, 'r' ) as csvfile:
            # Get header as it appears
            reader = csv.reader( csvfile )
            for i, row in enumerate( reader ):
                if i == 0:
                    hdr += row
                else:
                    lines.append( { k:v for k,v in zip(hdr,row) } )
        self.header = hdr
        self.lines  = lines
        
    def parse( self ):
        # Breakout lines into data
        def clean( key, l ):
            fmt     = '%Y-%m-%d %H:%M:%S'
            # List all temp metrics and remove 0 values later
            nozeros = ['Heatsink1', 'Heatsink2', 'PCRHeatSink']
            nozeros += [ x for x in self.header if 'Temp' in x ]
            nozeros += [ x for x in self.header if 'Ambient' in x ]
            try:
                val = l[key]
            except KeyError:
                return None

            try:
                if key == 'time':
                    return datetime.datetime.strptime( val, fmt ) 
                elif (float(val) <= 0.0 ) and (key in nozeros):
                    # Should not be any 0 C or lower temps
                    return None
                else:
                    return float( val )
            except (TypeError, ValueError,):
                return None
 
        data = {}
        keys = tuple( self.lines[0].keys() )

        for l in self.lines:
            # skip time == None
            if clean( 'time', l ) is None:
                continue
            for k in keys:
                val = clean( k, l )
                try:
                    data[k].append( val )
                except KeyError:
                    data[k] = [ val ]

        self.data = {}
        for k in keys:
            self.data[k] = np.array( data[k] )
