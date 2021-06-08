import os
import matplotlib.pyplot as plt
import numpy as np

from . import system
from . import chiptype as ct
from .datfile import DatFile

class Thumbnail():
    """
    Class for doing analysis of traces on thumbnails
    This should easily be generalized to datablocks
    """
    def __init__(self, rawdatadir='.', analysisdir='.', chiptype=None, framerate=15 ):
        """ 
        Initializes the block for analysis.
        usage:
            rawdatadir : Directory of data files.  
            analysisdir: Directory to save analysis.  
            chiptype   : Name of chip (e.g. P1.0.20)
            framerate  : Aquisition rate
        """
        
        self.rawdatadir  = rawdatadir
        self.analysisdir = analysisdir

        if isinstance( chiptype, ct.ChipType ):
            self.chiptype = chiptype
        else:
            self.chiptype = ct.ChipType( name=chiptype, tn=tn, blockdir=os.path.join( rawdatadir, '..' ) )
        self.framerate = framerate

        # Make sure the analysisdir exists
        system.makedir( analysisdir )

    def block_avg(self):
        """
        Averages the traces stored in self.data by block, resulting in a 8x12 array
        which is saved to self.avg
        """
        self.avg = np.zeros([8,12,self.data.shape[2]])
        for row in range(8):
            for col in range(12):
                self.avg[row,col,:] = self.data[row*100:(row+1)*100,col*100:(col+1)*100,:].mean(axis=0).mean(axis=0)
    
    def load_raw( self, flow_file ):
        """
        Loads the complete dat file
        """
        self.flow_file = flow_file
        filename = os.path.join( self.rawdatadir, flow_file )
        im = DatFile( filename, chiptype=self.chiptype )
        self.trace_data = im
        self.data = im.data
        
    def plot_block_traces( self, filename='' ):
        ''' Plots the average trace for all blocks '''
        # make sure traces exist
        if not hasattr( self, 'avg' ):
            self.block_avg()

        if filename:
            filename = os.path.join( self.analysisdir, filename )
        plot_block_traces( self.avg, filename )

    def plot_cols(self,avg=True,diff=False,row=None,legend='right',title='',filename=''):
        """
        Plots the average of each block along a particular row
        
        inputs: 
            avg:  Use the block-averaged traces instead of individual wells
            diff: Plots the delta value between frames instead of the actual frame value
            row:  Row to plot.  4 or 400 for avg = True or False, respectivly
            filename:  Name of the file.  Data will be stored in analysisdir
        """
        # Select the appropriate data 
        if avg:
            cols = range(12)
            if row is None:
                row = 4
            try:
                data = self.avg
            except AttributeError:
                self.block_avg()
                data = self.avg
        else:
            cols = np.arange(50,1200,100)
            if row is None:
                row  = 400
            data = self.data

        # Get a vector for the time-axis
        x = range(data.shape[2])
        if diff:
            x.pop()

        # Make the figure
        fig = plt.figure(facecolor='w')
        ax = plt.subplot(111)
        # Plot each column
        cm = plt.get_cmap( 'jet', len(cols) )
        for index in range( len( cols ) ):
            col = cols[index]
            if diff:
                y = np.diff( data[ row, col ] )
            else:
                y = data[ row, col ]
            ax.plot( x, y, label = 'col %s' % (col), color=cm(index) )
        # Set the legend
        box = ax.get_position()
        if legend=='bottom':
            ax.set_position([box.x0,    box.y0 + box.height*0.1,
                             box.width, box.height*0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=12)
        else:
            ax.set_position([box.x0,        box.y0,
                             box.width*0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Set the title
        plt.title(title)
        # Output the figure
        if filename:
            savename = os.path.join( self.analysisdir, filename )
            fig.savefig(savename)
        else:
            plt.show()

    def plot_timepoint_by_col(self,time=-1,avg=True,diff=False,row=None,legend='right',title='',filename=''):
        """
        Plots the trace at a particular point in time, in the given row, for each column
        
        inputs: 
            avg:  Use the block-averaged traces instead of individual wells
            diff: Plots the delta value between frames instead of the actual frame value
            row:  Row to plot.  4 or 400 for avg = True or False, respectivly
            filename:  Name of the file.  Data will be stored in self.analysisdir
        """
        # Select the appropriate data 
        if avg:
            cols = range(12)
            if row is None:
                row = 4
            try:
                data = self.avg
            except AttributeError:
                self.block_avg()
                data = self.avg
        else:
            cols = np.arange(50,1200,100)
            if row is None:
                row  = 400
            data = self.data

        # Get a vector for the column number
        x = range(data.shape[1])

        # Make the figure
        fig = plt.figure(facecolor='w')
        ax = plt.subplot(111)
        # Plot each column
        for col in cols:
            if diff:
                y = data[ row, :, time+1 ] - data[ row, :, time ]  
            else:
                y = data[ row, :, time ]  
            ax.plot( x, y, label = 'col %s' % (col))
        # Set the legend
        box = ax.get_position()
        if legend=='bottom':
            ax.set_position([box.x0,    box.y0 + box.height*0.1,
                             box.width, box.height*0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=12)
        else:
            ax.set_position([box.x0,        box.y0,
                             box.width*0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Set the title
        plt.title(title)
        # Output the figure
        if filename:
            savename = os.path.join( self.analysisdir, filename )
            fig.savefig(savename)
        else:
            plt.show()

def plot_tubes( thumbnails, col=1, row=4, outdir='.', files=None, title=None ):
    """
    Plots all Reagent tubes at the specified row and column in the specified column

    If files is specified (list if filenames including .dat), only the thumbnails listed in files will be plotted
    """
    
    # Pare down the file list
    if files:
        tn_files = [ tn.flow_file for tn in thumbnails ]
        tns = []
        for f in files:
            try:
                tns.append( thumbnails[ tn_files.index(f) ] )
            except ValueError:
                pass
    else:
        tns = thumbnails

    if not title:
        title = 'Reagent Flows'
    savename = title.replace( ' ', '_' )

    if len(tns):
        fn = os.path.join( outdir, '%s-X%i-Y%i.png' % ( savename, col, row ) )
        plot_reagents( tns, row=row, col=col, title=title, savename=fn )

        fn = os.path.join( outdir, '%s-Diff-X%i-Y%i.png' % ( savename, col, row ) )
        plot_reagents( tns, row=row, col=col, title='Delta %s' % title, savename=fn , diff=True )

def plot_reagents( thumbnails, row=None, col=None, avg=True, diff=False, legend='right', title='', savename='', names=None):
    """
    Plots the trace in the specified row and column for each of the thumbnail files

    Inputs:
        row, col:   Well to plot.  If average is set, defaults to (4, 1)  otherwise (400,100)

        avg:        Uses the average value
        diff:       plots frame to frame difference

        savename:   filename, including path if necessary to save the figure
        names:      Array of strings specifing legend names.  This must be the same size as thumbnails.  If unset, defaults to flow_file
    """

    # Set the row and columns
    if avg:
        if row is None:
            row = 4
        if col is None:
            col = 1
    else:
        if row is None:
            row = 400
        if col is None:
            col = 100

    # Make the figure
    fig = plt.figure(facecolor='w')
    ax = plt.subplot(111)

    # Plot each thumbnail file
    if names is None:
        names = [ tn.flow_file.split('.')[0] for tn in thumbnails ]
    cm = plt.get_cmap('jet',len(thumbnails))
    if len(thumbnails) > 10:
        showlabels = np.linspace( 0, len(thumbnails)-1, 10 ).astype(np.int) 
    else:
        showlabels = np.arange( len(thumbnails ) )
    for index in range( len( thumbnails ) ):
        tn = thumbnails[ index ]
        l  = names[ index ] if index in showlabels else None
        if avg:
            y = tn.avg[ row, col ]
        else:
            y = tn.data[ row, col ]

        x = range( tn.data.shape[2] )
        if diff:
            x.pop()
            y = np.diff(y)
        
        ax.plot( x, y, label=l, color=cm(index) )

    # Set the legend
    box = ax.get_position()
    if legend=='bottom':
        ax.set_position([box.x0,    box.y0 + box.height*0.1,
                         box.width, box.height*0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=12)
    else:
        ax.set_position([box.x0,        box.y0,
                         box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Set the title
    title += ': X%i Y%i' % (col, row)
    plt.title(title)
    # Output the figure
    if savename:
        fig.savefig(savename)
        plt.close(fig)
    else:
        fig.show()

def plot_block_traces( data, filename='' ):
    ''' 
    Plots the trace data for each x/y coordinate in data
    '''
    fig       = plt.figure( figsize=(13,9) )

    minval    = data[:].min()
    maxval    = data[:].max()
    diffval   = (maxval-minval)/10.
    increment = 10**np.ceil( np.log10( diffval ) )
    minval    = np.floor((minval/increment))*increment
    maxval    = np.ceil((maxval/increment))*increment
    
    cm = plt.get_cmap('jet',13)

    index = 0
    ROW = data.shape[0]
    COL = data.shape[1]
    for row in range(ROW):
        for col in range(COL):
            index += 1
            ax = fig.add_axes( [col/(COL+1.), (row+1)/(ROW+1.), 1/(COL+1.), 1/(ROW+1.) ] )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.plot( data[row,col,:], color=cm(index%cm.N) )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim( [0,120] )
            ax.set_ylim( [minval, maxval] )
    index = 0
    for row in range(ROW):
        ax = fig.add_axes( [COL/(COL+1.), (row+1)/(ROW+1.), 1/(COL+1.), 1/(ROW+1.)] )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for col in range(COL):
            index += 1
            plt.plot( data[row,col,:], color=cm(index%cm.N) )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim( [0,120] )
        ax.set_ylim( [minval, maxval] )
    for col in range(COL):
        index = col+1
        ax = fig.add_axes( [col/(COL+1.), 0, 1/(COL+1.), 1/(ROW+1.)] )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for row in range(ROW):
            plt.plot( data[row,col,:], color=cm(index%cm.N) )
            index += COL
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim( [0,120] )
        ax.set_ylim( [minval, maxval] )

    # Output the figure
    if filename:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()

