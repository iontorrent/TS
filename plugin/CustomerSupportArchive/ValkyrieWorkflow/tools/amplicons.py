import csv, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re, requests, json

TS_USER  = 'ionadmin'
TS_PASS  = 'ionadmin'
TS_AUTH  = requests.auth.HTTPBasicAuth( TS_USER, TS_PASS )

METAL_RE = re.compile( r'.+?>(?P<object>[\w._-]*)</a></td>' )

class MetalMixin( object ):
    """ Container for handy TS metal navigation. """
    @staticmethod
    def get_url( url, auth=TS_AUTH ):
        """ Shortcut to get a url through requests using standard TS_AUTH.  Does not assume json or txt. """
        return requests.get( url, auth=auth )
    
    @staticmethod
    def find_plugin_dir( report_url, plugin_name, auth=TS_AUTH ):
        """ Given a report URL, find the path to the given plugin's metal plugin_out dir. """
        plugin_data = os.path.join( report_url, 'metal/plugin_out' )
        
        plugins = requests.get( plugin_data, auth=auth )
        for l in plugins.text.splitlines():
            if plugin_name in l:
                # This iterates through and for multiple overwritten plugins will end up with the latest plugin
                line = l.strip()
                #print( line )
                
        pattern = METAL_RE
        m = pattern.match( line )
        if m:
            plugin_dir = os.path.join( plugin_data , m.groupdict()['object'] )
            return plugin_dir
        else:
            print( 'Plugin Dir not found for {} at {}!'.format( plugin_name, report_url ) )
            return None

        
    @staticmethod
    def get_barcodes( plugin_url=None, report_url=None, plugin=None, auth=TS_AUTH ):
        if not plugin_url and (report_url and plugin):
            plugin_url = self.find_plugin_dir( report_url, plugin, auth=auth )
            
        if not plugin_url:
            print( 'Plugin Dir not found!' )
            return None
        
        if plugin=='coverageAnalysis_CB':
            bc_pattern = re.compile( r'(IonDual_[0-9]{4}-IonDual_[0-9]{4})' )
            print('using alternative method of finding barcodes') 
            all = requests.get( plugin_url , auth=auth )
            barcodes = []
            for l in all.text.splitlines():
                if 'href' in l:
                    d = l.split('"')[1].split('/')[-1]
                    if bc_pattern.match( d ):
                        barcodes.append( d )

        else:
            print('using regular method')
            bc_data  = requests.get( os.path.join( plugin_url, 'barcodes.json' ) , auth=auth )
            bc       = json.loads( bc_data.text )
            barcodes = bc.keys()
        return barcodes
    
    
    @staticmethod
    def find_plugin_file( ts_url, file_pattern, auth=TS_AUTH ):
        """ Takes a TS URL (within metal) and searches for files of a given pattern, e.g. 'amplicat.xls' """
        fline = None
        page  = requests.get( ts_url, auth=auth )
        for line in page.text.splitlines():
            if file_pattern in line:
                fline = line
                
        pattern = METAL_RE
        m = pattern.match( fline )
        if m:
            file_path = os.path.join( ts_url , m.groupdict()['object'] )
            #print( 'found {} file path {}'.format(file_pattern,file_path))
            return file_path
        else:
            print( 'File containing {} not found at {}!'.format( file_pattern, ts_url ) )
            return None


class CSVBase( MetalMixin ):
    """ Base class with methods for for loading a csv file and doing later analysis on it. """
    def __init__( self, csv_file, *args, **kwargs ):
        self.csv_file = csv_file
        self.args     = args
        self.kwargs   = kwargs

        
    def get_column( self, key, func=None ):
        """ gets a column of data and will apply func to the data to adjust the type, if supplied """
        if func:
            return [ func( d[key] ) for d in self.data ]
        else:
            return [ d[key] for d in self.data ]
        
        
    def read_csv( self , as_dict=False , delimiter=',' , lines=[], url=None ):
        """
        Reads the CSV file, either as a list of lists or a list of dicts and saves as generic self.data.
        Default delimiter is a comma (hey, it's a CSV) but might want to overwrite specifically in subclasses.
        """
        data = []

        # If lines are supplied, this ignores the self.csv_file
        if lines:
            csvfile = lines
        elif os.path.exists( self.csv_file ):
            #with open( self.csv_file , 'r' ) as csvfile:
            print('self.csv_file exists')
            csvfile = open( self.csv_file , 'r' )
        else:
            print( 'No lines supplied and csv file [{}] was not found!'.format( self.csv_file ) )
            return None
        
        if as_dict:
            reader = csv.DictReader( csvfile, delimiter=delimiter )
        else:
            reader = csv.reader( csvfile, delimiter=delimiter )
            
        for row in reader:
            try:
                data.append( row )
            except:
                print( 'Error appending row!  Attempting to print below:' )
                print( row )
                    
        self.data = data

        
    @classmethod
    def load( cls, csv_file, *args, **kwargs ):
        """ Helper method that automatically reads the csv [with class defaults] after initialization. """
        obj = cls( csv_file, *args, **kwargs )
        obj.read_csv( )
        return obj

    
    @classmethod
    def from_lines( cls, lines, *args, **kwargs ):
        """ Helper method that returns a class instance based on lines of text, likely from a url request. """
        obj = cls( None, *args, **kwargs )
        obj.read_csv( lines=lines )
        return obj
    
    
    @classmethod
    def from_url( cls, url, *args, **kwargs ):
        """ Helper method that returns a class instance based on given csv url. """
        auth    = kwargs.get('auth'     , TS_AUTH )
        delim   = kwargs.get('delimiter', '\t' ) # was ',' 
        as_dict = kwargs.get('as_dict'  , False )
        ans     = requests.get( url, auth=auth )
        lines   = ans.text.splitlines()
        
        obj     = cls( None, *args, **kwargs )
        obj.read_csv( as_dict=True, delimiter=delim, lines=lines ) # used to be as_dict=as_dict
        return obj

    
    @classmethod
    def get_per_barcode_files( cls, report_url, plugin_name, file_pattern, *args, **kwargs ):
        """ Tries to find all per-barcode files of given pattern in <plugin_name> on the given report_url. """
        auth       = kwargs.get('auth'     , TS_AUTH )
        files      = {}
        plugin_dir = cls.find_plugin_dir( report_url, plugin_name, auth=auth )
        barcodes   = cls.get_barcodes   ( plugin_url=plugin_dir, plugin=plugin_name, auth=auth )
        
        for barcode in barcodes:
            #print('------------------')
            bc_path = os.path.join( plugin_dir, barcode )
            #print('barcode path: {}'.format(bc_path))
            fp      = cls.find_plugin_file( bc_path, file_pattern, auth=auth )
            if fp:
                # have the path to the xls file- use it to make a class from url. Class will NOT contain csv that can be read 
                files[ barcode ] = cls.from_url( fp, *args, **kwargs )
            else:
                files[ barcode ] = None
                
        return files


class AmpliCat( CSVBase ):
    """
    Class for reading and analyzing amplicat.xls files from AmpliSeqCheckup_UMT, which come from the plugin's
    per-barcode link titled 'Download the amplicon read categories summary file' at the bottom of the page.
    """
    source = 'AmpliSeqCheckup_UMT'
    
    def read_csv( self , as_dict=True , delimiter='\t' , lines=[] ):
        """ Overwriting method by calling specific methods of the parent class with tab delimiter """
        super( AmpliCat, self).read_csv( as_dict=as_dict, delimiter=delimiter, lines=lines )
        
        
    def draw_histogram( self, ax, data, bins, **kwargs ):
        n, bins, patches = ax.hist( data, bins=bins, **kwargs )
        return n, bins, patches
    
    
    def annotate_fwd_bias( self , ax ):
        """ Common ways to annotate an axis for %forward reads and strand bias """
        ax.axvline( 30 , ls='-' , color='red'   )
        ax.axvline( 70 , ls='-' , color='red'   )
        ax.axvline( 50 , ls=':' , color='black' )

        xl = '% Forward Reads'
        if hasattr( self, 'source' ):
            xl += ' (source: {})'.format( self.source )
            
        ax.set_xlabel( xl )
        ax.set_ylabel( 'Target Amplicons' )
        
        # Determine where labels should go.
        ymin, ymax = ax.get_ylim()
        
        ypos = (ymax-ymin) * 0.6
        ax.text( 15 , ypos , 'Reverse Bias', ha='center' , color='red')
        ax.text( 85 , ypos , 'Forward Bias', ha='center' , color='red')
        
        
    def forward_bias_hist( self , prefix='' ):
        fbias = self.get_column( 'fwd_pc' , float )
        fig   = plt.figure( figsize=(12,6) )
        ax    = fig.add_subplot( 111 )
        ax.hist( fbias, bins=np.linspace( 0, 100, 101 ) )
        
        self.annotate_fwd_bias( ax )
        if prefix:
            ax.set_title ( '{}  | Histogram of Per-Amplicon Strand Bias [N={:d} Amplicons]'.format( prefix,
                                                                                                    len( fbias ) ))
        else:
            ax.set_title ( 'Histogram of Per-Amplicon Strand Bias [N={:d} Amplicons]'.format( len( fbias ) ) )
        fig.show( )
        
        
    @classmethod
    def from_url( cls, url, *args, **kwargs ):
        """ Helper method that returns a class instance based on given csv url. """
        return super( AmpliCat, cls ).from_url( url, as_dict=True, delimiter='\t' )
    
    
    @classmethod
    def get_per_barcode_files( cls, report_url ):
        """ Overwrites parent method with obvious plugin and file pattern """
        return super( AmpliCat, cls ).get_per_barcode_files( report_url, 'AmpliSeqCheckup_UMT', 'amplicat.xls' )
    
    
class CovAnalysis( AmpliCat ):
    """
    Class very similar to the amplicat class, meant to look at strand bias, but here we extract information
      from the coverageAnalysis plugin and the amplicon.cov.xls file.
    """
    source = 'coverageAnalysis'
    
    def read_csv( self, as_dict=True, delimiter='\t', lines=[] ):
        super( CovAnalysis, self ).read_csv( as_dict=as_dict, delimiter=delimiter, lines=lines )

        # Now we need to iterate and create the fwd_pc key in each row
        for row in self.data:
            forward_reads = float( row['fwd_reads'] )
            total_reads   = float( row['total_reads'] )
            if total_reads > 0:
                row['fwd_pc'] = 100. * forward_reads / total_reads
            else:
                row['fwd_pc'] = 0.
                
                
    @classmethod
    def get_per_barcode_files( cls, report_url ):
        """ Overwrites parent method with obvious plugin and file pattern """
        return super( AmpliCat, cls ).get_per_barcode_files( report_url, 'coverageAnalysis', 'amplicon.cov.xls' )
    
    
def paired_histogram( top, top_label, bot, bot_label, title='', figpath='' ):
    """ 
    Makes a paired histogram of strand bias given two objects of the AmpliCat class 
    Assumes that we have already created the objects and read the csvs.
    """
    # Set up figure
    fig   = plt.figure( figsize=(12,6) )
    ax1   = fig.add_subplot( 211 )
    ax2   = fig.add_subplot( 212 )
    bins  = np.linspace( 0, 100, 101 )
    
    top.draw_histogram   ( ax1, top.get_column( 'fwd_pc', float ), bins=bins, color='blue', label=top_label )
    ax1.legend( )
    ax1.set_title( title )
    
    bot.draw_histogram   ( ax2, bot.get_column( 'fwd_pc', float ), bins=bins, color='green', label=bot_label )
    ax2.legend( )
    
    # Equalize heights then add strand bias annotations ( because they depend on height of y-axis )
    ymax = max( ax1.get_ylim()[1], ax2.get_ylim()[1] )
    ax1.set_ylim( 0, ymax )
    ax2.set_ylim( 0, ymax )
    
    top.annotate_fwd_bias( ax1 )
    bot.annotate_fwd_bias( ax2 )
    
    fig.tight_layout( )
    if figpath:
        fig.savefig( figpath )
        print( 'Figure saved to: {}'.format( figpath ) )
        
    fig.show( )

    
