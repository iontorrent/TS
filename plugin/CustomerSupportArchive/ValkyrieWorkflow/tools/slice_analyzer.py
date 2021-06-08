import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class SliceAnalyzer():
    def __init__( self, data, shape=(10656, 15456), bin_size=8, sample='' ):
        try:
            self.data = data.reshape( shape )
        except:
            raise
        self.shape = shape
        self.lane_width = None
        self.bin_size = bin_size
        #self.max = np.nan_to_num( np.nanmax( self.data[np.where(self.data>0)] ) )
        #print( self.max )
        
        self.sample = sample
        
        self.ffc   = {}
        self.multilane = {}
        self.calculated_lanes = []
    
    ################################
    #      DATA MANIPULATION       #
    ################################
        
    def trim( self, num_rows=1000 ):
        '''takes in an array and trims num_rows from the top and bottom
                e.g. data.shape = (10000,x) --> n_data.shape = (8000,x)
                                 data[1000] --> n_data[0]
                                 data[8999] --> n_data[7999]
        '''
        return self.data[ num_rows:(self.data.shape[0]-num_rows) ]

    def select_lane( self, data, lane=None ):
        if lane is not None:
            lane_width = np.int( data.shape[1]/4 )
            lane_idx = lane-1
            self.lane_width = lane_width
            return data[:, lane_idx*lane_width:lane*lane_width]
        
    def select_region( self, trim_rows=None, lane=None ):
        if trim_rows is not None:
            data = self.trim( trim_rows )
        else:
            data = self.data
        if lane is not None:
            data = self.select_lane( data, lane=lane )
        return data
    
    def store_values( self, values, lane=None ):
        if lane is not None:
            l_key = 'lane{}'.format(lane)
            try:
                self.multilane[l_key]
            except KeyError:
                self.multilane[l_key] = {}
            self.multilane[l_key].update( values )
        else:
            self.ffc.update( values )
        
    ############################
    #       CALCULATIONS       #
    ############################
    
    def binned_strips( self, bin_size=None, trim_rows=None, lane=None, threshold=None ):
        ''' takes in a bool data set and bins strips with dimensions of all rows by bin_size num of cols
            input:
            
            returns:
                binned_percent  = a percent calculated as sum of binned data / total num of entries in bin
                edges           = (l_edge, r_edge) tuple of indexes
                midpoint        = np.int( (l_edge + r_edge )/2 ) index
        '''
        if lane is not None:
            self.calculated_lanes.append(lane)
        data = self.select_region( trim_rows=trim_rows, lane=lane )
        
        if bin_size is None:
            bin_size = self.bin_size
        else:
            self.bin_size = bin_size
            
        num_rows = data.shape[0]
        num_cols = data.shape[1]
        binned = np.zeros( np.int( np.floor( num_cols/bin_size ) ) ) 
        masked_binned = np.zeros( binned.size )
        idxs = bin_size*np.arange( 0, binned.shape[0] )
        l_edge = 0 
        l_edge_found = False
        r_edge_max = len(binned) - 1
        r_edge = r_edge_max
        r_edge_found = False
        for i, idx in enumerate( idxs ):
            section = data[:,idx:(idx+bin_size)].flatten()
            # boolean array of nonzero elements
            masked_section = np.ma.masked_greater(section, 0)
            full_size = section.size
            if threshold is not None:
                # this operation flattens the array as well as removing values below threshold
                section = section[np.where(section>threshold)]
            # Measure size AFTER accounting for threshold
            size = section.size
            
            if size == 0:
                binned[i] == 0
            else:
                binned[i] = np.nan_to_num( np.nansum(section)/size )
            
            # statistics based on full size of array
            if full_size == 0:
                masked_binned[i]=0
            else:
                masked_binned[i] = 100*np.nan_to_num( np.nansum(masked_section.mask)/full_size )

            if l_edge_found==True and binned[i]<binned[l_edge] and idx<500:
                l_edge_found = False
                l_edge = 0
                
            if l_edge == 0 and binned[i] >= 0.0001 and l_edge_found == False:
                l_edge = i
                #if l_edge < 0:
                #    l_edge = 0
                l_edge_found = True
            if r_edge >= r_edge_max and binned[i] <= 0.0001 and r_edge_found==False and l_edge_found==True and idx>500:
                r_edge = i
                r_edge_found = True
            #print( i, r_edge_found )
            #print( r_edge, binned[i], l_edge_found, idx )
        midpoint = np.int( (l_edge + r_edge)/2 )
        #print( l_edge, r_edge, midpoint, data.shape )
        
        # this block performs the cumulative calculation for fromL and fromR using the edges found above
        c_idxs = bin_size*np.arange( 0, midpoint )
        c_bin_fromL = np.zeros( c_idxs.shape[0] )
        c_bin_fromR = np.zeros( c_idxs.shape[0] )
        l_idx = l_edge*bin_size
        r_idx = r_edge*bin_size
        #print( l_idx, r_idx )
        for i in range( c_idxs.size ):
            l_section = data[:,l_idx:(l_idx+((i+1)*bin_size))].flatten()
            r_section = data[:,r_idx:(r_idx-((i+1)*bin_size)):-1].flatten()
            if threshold:
                # This operation flattens the array as well as removing values below threshold
                l_section = l_section[np.where(l_section>threshold)]
                r_section = r_section[np.where(r_section>threshold)]
            # Measure size AFTER accounting for threshold
            l_size = l_section.size
            r_size = r_section.size
                
            if l_size == 0:
                c_bin_fromL[i] = 0
            else:
                c_bin_fromL[i] = np.nansum(l_section)/( l_size )
            
            if r_size == 0:
                c_bin_fromR[i] = 0
            else:
                c_bin_fromR[i] = np.nansum(r_section)/( r_size ) 
                
        # this block rescales the simple binning by a values about the midpoint
        #print( midpoint, l_edge, r_edge, bin_size )
        m_l_ind = bin_size*midpoint-np.int(np.floor(bin_size/2.))
        m_r_ind = bin_size*midpoint+np.int(np.floor(bin_size/2.))
        mid_section = data[:,m_l_ind:m_r_ind].flatten()
        mid_fullsize = mid_section.size
        masked_mid_section = np.ma.masked_greater(mid_section, 0)
        if threshold:
            mid_section = mid_section[np.where(mid_section>threshold)]
        bin_mid = np.nansum(mid_section)/(mid_fullsize)
        rescaled_simple = binned/bin_mid
        rescaled_masked = masked_binned*mid_fullsize/(100*np.nansum(masked_mid_section.mask))
                
        values = {'idxs':idxs, 
                  'binned_simple':binned, 
                  'masked_binned':masked_binned,
                  'c_idxs':c_idxs,
                  'c_bin_fromL':c_bin_fromL, 
                  'c_bin_fromR':c_bin_fromR,
                  'rescaled_simple':rescaled_simple,
                  'rescaled_masked':rescaled_masked,
                  'edges':(l_edge, r_edge), 
                  'midpoint':midpoint }
        
        self.store_values( values, lane=lane )
        # Don't need full data anymore so get rid of it
        self.data = None
 
    def relative_lane_perf_assymetry( self, lane, max_wells_from_edge=1500, start_i = 4 ):
        if lane is not None:
            l_key = 'lane{}'.format(lane)
            idxs = self.multilane[l_key]['idxs']
            edges = self.multilane[l_key]['edges']
            midpoint = self.multilane[l_key]['midpoint']
            rm = self.multilane[l_key]['rescaled_masked']
            width = self.lane_width
        else:
            idxs = self.ffc['idxs']
            edges = self.ffc['edges']
            midpoint = self.ffc['midpoint']
            rm = self.ffc['rescaled_masked']
            width = self.shape[1]
        wells_from_edge = 0
        rel_perf = []
        rel_idxs = []
        i = start_i
        while wells_from_edge < max_wells_from_edge:
            rel_perf.append(100*np.nan_to_num((rm[edges[0]+i] - rm[edges[1]-i])/(rm[edges[0]+i]+rm[edges[1]-i])))
            if lane == 4:
                wells_from_edge = width-idxs[edges[1]-i]
                rel_idxs.append(wells_from_edge)
            else:
                wells_from_edge = idxs[edges[0]+i]
                rel_idxs.append(wells_from_edge)
            i += 1
        values = {'rel_idxs':rel_idxs, 'rel_perf':rel_perf}
        self.store_values( values, lane=lane )
        return (rel_idxs, rel_perf )        
            
    ################################
    #         PLOTTING             #
    ################################
    
    def plot_profile( self, x, y, edges, width, graph_title='Profile', ylabel='', show_left=False, show_right=False, hline=None, relative=False ):
        plt.plot( x, y )
        # vertical lines
        vert_y = np.linspace( np.nanmin( y ), np.nanmax( y ))
        vert_x = np.ones( vert_y.size )
        if show_left:
            # left
            plt.plot( 100*vert_x, vert_y, 'r--', label='100 wells from edge' )
            plt.plot( 200*vert_x, vert_y, 'y--', label='200 wells from edge' )
            plt.plot( [], [], ' ', label='Edge Distance: '+str(edges[0])+' wells')
        if show_right:
            # right
            plt.plot( (self.lane_width - 100.)*vert_x, vert_y, 'r--', label='100 wells from edge' )
            plt.plot( (self.lane_width - 200.)*vert_x, vert_y, 'y--', label='200 wells from edge' )
            plt.plot( [], [], ' ', label='Edge Distance: '+str(width - edges[1])+' wells')
        if hline is not None:
            y_hline = np.ones(x.size)*hline
            plt.plot( x, y_hline, 'k--' )
        title = ''
        if self.sample:
            title += self.sample + ': '
        title += graph_title
        plt.title( title )
        if relative:
            xlabel = 'Wells from edge'
        else:
            xlabel = 'Well Index'
        xlabel += ' -- bins of {} columns'
        plt.xlabel( xlabel.format( self.bin_size ) )
        if ylabel is not None:
            plt.ylabel( ylabel )
        plt.legend()
        plt.tight_layout()
            
    def plot_simple_profile( self, lane=None, ylabel=None, show_left=False, show_right=False ):
        if lane is not None:
            l_key = 'lane{}'.format(lane)
            idxs = self.multilane[l_key]['idxs']
            binned_simple = self.multilane[l_key]['binned_simple']
            edges = self.multilane[l_key]['edges']
            width = self.lane_width
        else:
            idxs = ffc['idxs']
            binned_simple = self.ffc['binned_simple']
            edges = self.ffc['edges']
            width = self.shape[1]
        edges = (self.bin_size*edges[0], self.bin_size*edges[1])
        self.plot_profile( idxs, binned_simple, edges, width, graph_title='Simple Profile', ylabel=ylabel, show_left=show_left, show_right=show_right )        
            
    def plot_nonzero_profile( self, lane=None, ylabel=None, show_left=False, show_right=False ):
        if lane is not None:
            l_key = 'lane{}'.format(lane)
            idxs = self.multilane[l_key]['idxs']
            masked_binned = self.multilane[l_key]['masked_binned']
            edges = self.multilane[l_key]['edges']
            width = self.lane_width
        else:
            idxs = ffc['idxs']
            masked_binned = self.ffc['masked_binned']
            edges = self.ffc['edges']
            width = self.shape[1]
        edges = (self.bin_size*edges[0], self.bin_size*edges[1])
        self.plot_profile( idxs, masked_binned, edges, width, graph_title='Percent Non-zero', ylabel=ylabel, show_left=show_left, show_right=show_right )

    def plot_rescaled_nonzero_profile( self, lane=None, ylabel=None, show_left=False, show_right=False, hline=None, flip=False, relative=True ):
        if lane is not None:
            l_key = 'lane{}'.format(lane)
            idxs = self.multilane[l_key]['idxs']
            rescaled_masked = self.multilane[l_key]['rescaled_masked']
            edges = self.multilane[l_key]['edges']
            width = self.lane_width
        else:
            idxs = ffc['idxs']
            rescaled_masked = self.ffc['rescaled_masked']
            edges = self.ffc['edges']
            width = self.shape[1]
        edges = (self.bin_size*edges[0], self.bin_size*edges[1])
        if flip:
            idxs = np.flipud( idxs )
            #plt.gca().invert_xaxis()
            edges = ( width-edges[1], width-edges[0], )
            temp = show_right
            show_right = show_left
            show_left = temp
        self.plot_profile( idxs, rescaled_masked, edges, width, graph_title='Rescaled Non-zero', ylabel=ylabel, show_left=show_left, show_right=show_right, hline=hline, relative=relative )
              
    def plot_simple_LvsR( self, lane=None, ylabel=None ):
        if lane is not None:
            l_key = 'lane{}'.format(lane)
            idxs = self.multilane[l_key]['idxs']
            midpoint = self.multilane[l_key]['midpoint']
            binned_simple = self.multilane[l_key]['binned_simple']
        else:
            idxs = ffc['idxs']
            midpoint = self.ffc['midpoint']
            binned_simple = self.ffc['binned_simple']
        l_simp = binned_simple[midpoint:0:-1]
        r_simp = binned_simple[midpoint:]
        plt.plot( idxs[:l_simp.shape[0]], l_simp, label='Left' )
        plt.plot( idxs[:r_simp.shape[0]], r_simp, label='Right' )
        title = ''
        if self.sample:
            title += self.sample + ': '
        title += 'Comparison of L/R profiles'
        plt.title( title )
        plt.xlabel( 'Wells from midpoint of lane - bins of {} columns'.format( self.bin_size ) )
        if ylabel is not None:
            plt.ylabel( ylabel )
        plt.legend()
        plt.tight_layout()
        
    def plot_cumulative_LvsR( self, lane=None, ylabel=None ):
        if lane is not None:
            l_key = 'lane{}'.format(lane)
            c_idxs = self.multilane[l_key]['c_idxs']
            c_bin_fromL = self.multilane[l_key]['c_bin_fromL']
            c_bin_fromR = self.multilane[l_key]['c_bin_fromR']
        else:
            c_idxs = ffc['c_idxs']
            c_bin_fromL = self.ffc['c_bin_fromL']
            c_bin_fromR = self.ffc['c_bin_fromR']
        plt.plot( c_idxs, c_bin_fromL, label='Left' )
        plt.plot( c_idxs, c_bin_fromR, label='Right' )
        title = ''
        if self.sample:
            title += self.sample + ': '
        title += 'Cumulative from L/R'
        plt.title( title )
        plt.xlabel( 'Wells from edge of lane - bins of {} columns'.format( self.bin_size ) )
        if ylabel is not None:
            plt.ylabel( ylabel )
        plt.legend()
        plt.tight_layout()
