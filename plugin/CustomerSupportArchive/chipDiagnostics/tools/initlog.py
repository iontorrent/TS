import os, shutil
import re
from datetime import datetime

class InitLog:   
    """
    Class to handle, copy, and parse the InitLog
    """
    
    autochip = 'Unknown'
    seq_kit = ''
    kit_name = ''
    w2_formulation = 'Unknown'
    W1 = W2 = ''
    R1 = R2 = R3 = R4 = ''
    inittime = None

    def __init__(self,path="",dest='',parse=True):
        """
        Obtains a lock on the InitLog and parses it.

        If path is unspecified, init_log will look for any file named InitLog.txt, starting in the present working directory and then in /software/config

        if dest is specified, the initlog is copied to the destination with the path 'InitLog.txt'
        dest must be a folder
        """

        # Build the appropriate file name
        filename = path
        if path:
            if os.path.isdir(path):
                filename = os.path.join(path,'InitLog.txt')
        else:
            if os.path.isfile('InitLog.txt'):
                filename = os.path.join(os.getcwd(),path)
            else:
                filename = os.path.join('/software/config/InitLog.txt')

        # Make sure the file exists
        if os.path.exists(filename):
            self.filename = filename
        else:
            raise IOError('%s not found' % filename)

        # Copy the file to the experiment folder
        if dest:
            self.copy(dest)

        # Parse the initlog
        if parse:
            self.parse()

    def add_autoph_step(self,ind,step):
        """
        Grows self.autoph until the specified index is available, filling with 0. 
        At the specified index, step is added
        """
        while len(self.autoph) < ind+1:
            self.autoph.append(0)
        self.autoph[ind] = step

    def copy(self,dest='.'):
        """
        Copies the initlog to the specified directory
        dest must be a folder
        """
        dest = os.path.join(dest,'InitLog.txt')
        shutil.copy(self.filename, dest)
        self.filename = dest

    def parse(self):
        """
        Parses the entire file keeping the latest values for regent pH, time, autoph, etc...
        """
        for line in open(self.filename):
            # First extract chipefuse before it's mangled in the below:
            self.parse_chipefuse(line)
            rawline = line
            line = line.replace(':','')
            parts = line.split()
            # Order doen't matter here because only one of these should actually execute per line
            try:
                self.parse_autoph(parts)
                self.parse_date(rawline)
                self.parse_rawtrace(parts)
                self.parse_rawtrace(parts)
                self.parse_reagent(parts)
                self.parse_W1(line,parts)
                self.parse_chiptype(line)
                self.parse_kit(parts)
                self.parse_chipgain(line)
            except IndexError:
                continue

    def parse_autoph(self,parts):
        """
        Attempts to interpret thie line as an autoph step and saves to self
        """
        if parts[0] != 'AUTOPH':
            return
        ind = int(parts[2])
        if ind == 0:
            self.autoph = []
        step = {'start':  float(parts[4]),
                'end':    float(parts[6]),
                'volume': float(parts[8])}
        self.add_autoph_step(ind,step)
        
    def parse_date(self,line):
        """
        Attempts to interpret the line as a timestamp and saves it to a temporary variable
        """
        try:
            self._last_time = datetime.strptime( re.sub( r':*\s*$', '', line), "%a %b %d %H:%M:%S %Y" )
            if self.inittime is None:
                self.inittime = self._last_time
        except ValueError:
            pass

    def parse_chiptype( self, line ):
        '''
        Reads the efuse used for autopH
        '''
        if 'Chip Type' not in line:
            return
        parts = line.split()
        self.autochip = parts[-1]
        if self.autochip == 'BB1':
            self.autochip = 'Unknown'
            
    def parse_chipefuse( self , line ):
        if 'efuse:' in line:
            self.efuse = line.split('efuse:')[1].strip()

    def parse_chipgain( self , line ):
        if 'Chip gain=' in line:
            self.gain = float( line.split('=')[1] )
            
    def parse_kit(self, parts):
        """
        Attempts to interpret sequencing kit from InitLog 
        """
        if re.match( r'<SEQ.*>', parts[-1] ):
            self.seq_kit = parts[-1][1:-1]
            self.parse_dbSeqKit( self.seq_kit )

    def parse_dbSeqKit( self, kit ):
        """ 
        Attempts to read the database of sequncing kit names 
        """
        dbfile = '/software/config/dbSeqKitQuery.txt'
        try:
            for line in open(dbfile):
                parts = line.split(',')
                if parts[7] == kit:
                    kit_name = parts[2]
                    if 'v3' in kit_name:
                        w2 = 'v3'
                    elif 'v2' in kit_name:
                        w2 = 'v2'
                    elif kit_name == 'Ion PI IC 200 Seq Kit' or kit_name == 'Ion PI Hi-Q Sequencing 200 Kit':
                        w2 = 'Namso'
                    else:
                        w2 = 'Unknown'
                    self.kit_name = kit_name
                    self.w2_formulation = w2
        except:
            pass

    def parse_rawtrace(self,parts):
        """
        Rawtrace signifies the beginning of pH values, and thus the last timepoint is the measurement
        Let's save that value
        """
        if parts[0] == 'Rawtrace':
            try:
                self.time = self._last_time
            except:
                pass
    
    def parse_reagent(self,parts):
        """
        Parses the reagent pHs
        """
        if parts[0] == 'W2pH':
            self.W2 = float(parts[1])
        elif parts[0] == 'W1pH':
            self.W1 = float(parts[1])
        elif parts[0] == 'R1pH':
            self.R1 = float(parts[1])
        elif parts[0] == 'R2pH':
            self.R2 = float(parts[1])
        elif parts[0] == 'R3pH':
            self.R3 = float(parts[1])
        elif parts[0] == 'R4pH':
            self.R4 = float(parts[1])
        
    def parse_W1(self,line,parts):
        """
        Parses the lines for W1 flows at the end of an autoph step
        """
        if 'Dumping some W1 to waste' in line:
            self.W1toWaste = float(parts[5].split('=')[-1])
        if 'Diluting W1 with W2' in line:
            self.W1toW2 = float(parts[4].split('=')[-1])
        if 'W1 Step' in line:
            self.W1Step = int( parts[2] )

