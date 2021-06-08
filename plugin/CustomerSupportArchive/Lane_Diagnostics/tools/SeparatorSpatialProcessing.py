import numpy as np
import h5py
import json
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import stats

class SeparatorSpatialMetricExtractor(object):
    '''This class will open all relevant h5 files, 
        run them through the file parser (separate class),
        and then build arrays to generate images and extract
        statistics.

        Available metrics are:
            t0, snr, mad, sd, bf_metric, tau_e, tau_b, peak_sig, trace_sd, 
            bead, empty, ignore, pinned, reference, tf, lib, dud, soft_filt, 
            buff_clust_conf, sig_clust_conf
    '''
    def __init__( self, sigproc_dir, filename='separator.spatial.h5', height=None, width=None, thumbnail=False ):
        # define a tuple of available metrics
        self.metric_names   = ( 't0', 'snr', 'mad', 'sd', 'bf_metric', 'tau_e', 'tau_b', 
                'peak_sig', 'trace_sd', 'bead', 'empty', 'ignore', 'pinned', 'reference',
                'tf', 'lib', 'dud', 'soft_filt', 'buff_clust_conf', 'sig_clust_conf', )
        # contour limits dictionary --> add tuple as (min, max)
        self.clims = {  'snr': (0,25,),
                        'peak_sig': (25,150,),
                        'tau_e': (0,20),
                        'tau_b': (0,20),
                        'buff_clust_conf': (0,1),
                        'sig_clust_conf': (0,1),
                    }

        self.titles = { 'snr': 'SNR',
                        'peak_sig':'Key Signal',
                        'tau_e':'Tau E',
                        'tau_b':'Tau B',
                        'buff_clust_conf':'Buff. Clust. Conf.',
                        'sig_clust_conf':'Sig. Clust. Conf.',
                    }

        # set array height and width values
        self.height                 = height
        self.width                  = width
        
        # store if fullchip
        self.thumbnail              = thumbnail
        
        # initialize file dependent information
        self.files_found            = False
        self.file_list              = None
        self.block_x_list           = None
        self.block_y_list           = None
        
        # initalize metric and metric arrays
        self.selected_metric        = None
        self.metric_array           = None
        self.metric_dict            = None

        # initialize fullchip mask
        self.fullchip_mask          = None

        # populate file depenent information
        self.find_files_in_blocks( sigproc_dir, filename=filename )

        
    def analyze_fullchip( self ):
        '''This applies the mask to the metric array and calculates statistics
        '''
        # reinitialize metric dict
        self.metric_dict = None
        if self.fullchip_mask is not None:
            # calculate statistics using uniformity function
            blocksize = stats.calc_blocksize( self.metric_array, nominal=(100,100) )
            print( '    Blocksize = '+str(blocksize) )
            temp_dict = stats.uniformity( self.metric_array, blocksize, exclude=~self.fullchip_mask, only_values=True, iqr=True, std=True )
            temp_dict.update( {'blocksize':blocksize} )

            # store temp_dict in metric_dict
            if temp_dict is not None:
                self.metric_dict = { self.selected_metric: temp_dict }

    def build_array( self, metric ):     
        ''' Build full array for a given metric from all available files 
        '''
        self.selected_metric = metric

        self.metric_array = None
        
        temp = None

        for file in self.file_list:
            if file['file_exists']:
                ssfp = SeparatorSpatialFileParser()
                ssfp.load_file( file['filepath'] )
                f_array = ssfp.make_metric_array( metric )
                f_rows, f_cols = f_array.shape
                #print( 'f_rows = '+str(f_rows) )
                #print( 'f_cols = '+str(f_cols) )
                if temp is None:
                    temp = np.zeros( (f_rows*(len( self.block_y_list )), f_cols*(len( self.block_x_list )),) )
                #print( 'temp.shape = '+ str(temp.shape) )
                # Numpy array indexes start at 0
                # To select a slice, the range must be from 'start' to 'end+1'
                #   --> start + length of range == end+1
                col_start = f_cols * ( self.block_x_list.index( file['x'] ) )
                col_stop  = f_cols + col_start
                row_start = f_rows * ( self.block_y_list.index( file['y'] ) )
                row_stop  = f_rows + row_start
                #print( 'row_start = '+str(row_start ) )
                #print( 'row_stop = '+str(row_stop ) )
                #print( 'col_start = '+str(col_start ) )
                #print( 'col_stop = '+str(col_stop ) )

                # insert slice into array
                temp[ row_start:row_stop, col_start:col_stop ] = f_array
                # delete ssfp instance
                del ssfp
      
        if temp is not None:
            self.metric_array = temp

    def find_files_in_blocks( self, sigproc_dir, filename='separator.spatial.h5' ):
        ''' Determine if files with name=filename exist within block directories
                store file_exists, filepath, x, and y in a dictionary for each file inside a list
                extract a list of x and y vals
                   --> order them and save as x_list, and y_list
                initialize row_size and col_size to None
                   --> populate later with row/col * len(x_list/y_list)
        '''
        keys = ( 'file_exists', 'filepath', 'x', 'y' )
        if sigproc_dir[-1] != '/':
            sigproc_dir += '/'
        temp_file_list = []
        temp_x_list = []
        temp_y_list = []
        
        # get names of folders (and files) in sigproc_dir
        folder_names = os.listdir( sigproc_dir )
        
        # cycle through names looking for block folders
        regex = re.compile( r'block_X(\d+)_Y(\d+)' )
        for folder_name in folder_names:
            # initialize file_exists
            file_exists = False
            match = regex.match( folder_name )
            if match:
                # if the block folder exists, populate the values corresponding to keys
                path = sigproc_dir + folder_name

                if filename in os.listdir( path ):
                    file_exists = True

                # make full filepath
                filepath = path
                if filepath[-1] != '/':
                    filepath += '/'
                filepath += filename

                groups = match.groups()
                x = int( groups[0] ) 
                y = int( groups[1] )

                temp_x_list.append( x )
                temp_y_list.append( y )

                # append the values to the temp_file_list
                vals = [file_exists, filepath, x, y]
                temp_file_list.append( { str(x[0]):x[1] for x in zip( keys, vals ) } )
            
        # remove duplicate values
        temp_x_list = list( set( temp_x_list ) )
        temp_y_list = list( set( temp_y_list ) )

        # sort the x and y values
        temp_x_list.sort()
        temp_y_list.sort()

        #NOTE: At this point, we have a sorted list of x and y vals, but we might not have them all.
        #   Should use actual array height and width if possible
        if self.height and self.width and not self.thumbnail: 
            delta_x = temp_x_list[1] - temp_x_list[0]
            delta_y = temp_y_list[1] - temp_y_list[0]

            def make_list( max, delta ):
                all_vals = []
                temp = 0
                while temp < max:
                    all_vals.append( temp )
                    temp += delta
                return all_vals
 
            all_x = make_list( self.width,  delta_x )
            all_y = make_list( self.height, delta_y )

            all_x.sort()
            all_y.sort()

            # store the x and y values as a tuple
            self.block_x_list = tuple( all_x )
            self.block_y_list = tuple( all_y )
        else:
            # store the x and y values as a tuple
            self.block_x_list = tuple( temp_x_list )
            self.block_y_list = tuple( temp_y_list )
            
        # store the file list
        self.file_list = temp_file_list

        if self.file_list:
            print( 'Found some files to process' )
            self.files_found = True
        else:
            print( '-------------------NO FILES WERE FOUND------------------------')
            self.files_found = False

    def make_fullchip_mask( self ):
        ''' separator_spatial applied some mask processing to the data already.
            This code uses 0's in snr as a mask for subsequent statistical analysis.
            snr appears to have already had the <pinned> data applied as a mask, as 
            well as the <ignore> data.
        '''
        # Build snr array
        self.build_array( 'snr' )
        if self.metric_array is not None:
            # make an array of ones (i.e. Trues)
            temp = np.ones( self.metric_array.shape, dtype=bool )
            # set fullchip_mask equal to the logical_and of temp and snr > 0
            #   selects only values where snr > 0
            self.fullchip_mask = np.logical_and( temp, self.metric_array > 0 )
        # reset selected_metric and metric_array
        self.selected_metric    = None
        self.metric_array       = None

    def plot_and_save_fullchip_mask( self ):
        if self.fullchip_mask is not None:
            plt.figure()
            plt.imshow( self.fullchip_mask, origin='lower', interpolation='nearest', clim=(0,1,) )
            plt.title( 'SeparatorSpatialMetrics -- Fullchip Mask' )
            plt.colorbar( shrink=0.7 )
            plt.xlabel( 'Superpixel Columns' )
            plt.ylabel( 'Superpixel Rows' )
            plt.savefig( 'FullchipMask.png' )
            plt.close()

    def plot_and_save_metric_array( self ):
        if self.metric_array is not None:
            try:
                clim = self.clims[ self.selected_metric ]
            except KeyError:
                pass

            try:
                title = self.titles[ self.selected_metric ]
            except KeyError:
                title=self.selected_metric

            plt.figure()
            if clim is not None:
                plt.imshow( self.metric_array, origin='lower', interpolation='nearest', clim=clim )
            else:
                plt.imshow( self.metric_array, origin='lower', interpolation='nearest' )
            plt.title( 'No Mask -- ' + title )
            plt.colorbar( shrink=0.7 )
            plt.xlabel( 'Superpixel Columns' )
            plt.ylabel( 'Superpixels Rows' )
            plt.savefig( self.selected_metric + '_NoMask.png' )
            plt.close()

    def plot_and_save_masked_metric_array( self ):
        if (self.metric_array is not None) and (self.fullchip_mask is not None):
            try:
                clim = self.clims[ self.selected_metric ]
            except KeyError:
                clim = None

            try:
                title = self.titles[ self.selected_metric ]
            except KeyError:
                title=self.selected_metric

            plt.figure()
            temp = self.metric_array*self.fullchip_mask
            if clim is not None:
                plt.imshow( temp, origin='lower', interpolation='nearest', clim=clim )
            else:
                plt.imshow( temp, origin='lower', interpolation='nearest' )
            plt.title( title )
            plt.colorbar( shrink=0.7 )
            plt.xlabel( 'Superpixel Columns' )
            plt.ylabel( 'Superpixels Rows' )
            plt.savefig( self.selected_metric + '_masked.png' )
            plt.close()


class SeparatorSpatialFileParser(object):
    ''' This class handles the parsing the spatial separator h5 file
        The file structure is as follows:
            When loaded, the file has two keys
                spatial_header, spatial_table
            The values associated with the keys are extracted by calling
                file[key].value
            spatial_header consists of 
                row_step, col_step -- integers
                    **These values determine the size of the superpixels in the array
                headers -- a list
                    **These are the headers for columns in spatial_table
                    row_start, row_end, col_start, col_end, t0, snr, mad, sd, bf_metric, tau_e,
                        tau_b, peak_sig, trace_sd, bead, empty, ignore, pinned, reference, 
                        tf, lib, dud, soft_filt, buff_clust_conf, sig_clust_conf
            spatial_table contains a numpy array
                Each row has values that correlate to the name in the list headers
    '''
                
    def __init__( self ):
        # These are the two key elements of the h5 file
        self.hdr_json   = None
        self.tbl        = None

        # These are stored inside the hdr_json
        #   row and col step determine the size of the superpixel
        #   tbl_hdrs label the values in each table column
        #       ** the table is indexed by row
        self.row_step   = None
        self.col_step   = None
        self.tbl_hdrs   = None

        self.min_row    = None
        self.max_row    = None
        self.min_col    = None
        self.max_col    = None

    def load_file( self, filepath ):
        # Open hfd5 file
        file = h5py.File( filepath, 'r' )

        # populate hdr_json and table
        self.hdr_json   = json.loads( file['spatial_header'].value )
        self.tbl      = file['spatial_table'].value

        # close h5 file
        file.close()

        # extract info about array
        self.extract_array_info()

        # rescale table indexes for further processing
        self.rescale_table_indexes()

    #############################################
    #   USED IN load_file -- DO NOT CALL TWICE  #
    #############################################

    def extract_array_info( self ):
        # extract row and col step
        self.row_step = self.hdr_json['row_step']
        self.col_step = self.hdr_json['col_step']
        
        # extract tbl_hdrs
        self.tbl_hdrs = self.hdr_json['headers']
        #print( 'tbl_hdrs = '+str(self.tbl_hdrs) )

        # extract min max col values
        self.min_row = int( min( self.get_col( 'row_start' ) ) )
        self.max_row = int( max( self.get_col( 'row_start' ) ) )
        self.min_col = int( min( self.get_col( 'col_start' ) ) )
        self.max_col = int( max( self.get_col( 'col_start' ) ) )

    def rescale_table_indexes(self):
        ''' This function 
                elminates the row_end and col_end indexes
                rescales the start indexes to be single integer steps for the superpixels
        '''
        # delete the row_end
        self.tbl = np.delete( self.tbl, 1, axis=1 )
        self.tbl_hdrs.pop(1)
        # delete the col_end
        self.tbl = np.delete( self.tbl, 2, axis=1 )
        self.tbl_hdrs.pop(2)
        # rescale the row_start and col_start (col_start now has index of 1)
        self.tbl[:,0] = (self.tbl[:,0] - self.min_row)/self.row_step
        self.tbl[:,1] = (self.tbl[:,1] - self.min_col)/self.col_step

    #################################
    #   CALL AFTER LOADING FILE     #
    #################################

    def make_metric_array( self, hdr_label ):
        num_rows = int( max( self.tbl[:,0]+1 ))
        num_cols = int( max( self.tbl[:,1]+1 ))
        temp = np.zeros( ( num_rows, num_cols, ) )
        for i, row in enumerate( self.tbl ):
            temp[ int(row[0]), int(row[1]) ] = self.get_val( i, hdr_label )
        return temp

    def get_val( self, row_ix, hdr_label ):
        try:
            return self.tbl[ row_ix, hdr_label ]
        except (ValueError, IndexError):
            pass

        try:
            return self.tbl[row_ix, self.tbl_hdrs.index( hdr_label ) ]
        except:
            raise

    def get_row( self, row_ix ):
        return self.tbl[row_ix,:]

    def get_col( self, hdr_label ):
        try:
            # return a value by index
            return self.tbl[ :, hdr_label ]
        except (ValueError, IndexError):
            pass

        try:
            # return a value by hdr_label
            return self.tbl[ :, self.tbl_hdrs.index( hdr_label ) ]
        except:
            raise


