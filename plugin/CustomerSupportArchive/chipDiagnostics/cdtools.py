import sys, os
import numpy as np
import scipy.stats
import plotting

class NoisyOffset:
    def __init__( self , noise , offset , DR=400 , bins=101 ):
        self.noise      = noise
        self.offset     = offset
        self.DR         = DR
        self.bins       = bins
        self.metrics    = {}
        self.noise_lims = [50,250]
        
        # Assume that we are working from calibration data primarily, meaning that:
        # Offsets are in mV.
        # Noise is gain corrected and in uV.
        if DR < 500:
            self.offset_label = 'Pixel Offset (mV)'
        else:
            self.offset_label = 'Pixel Offset (DN)'
        self.noise_label = 'Gain-corrected Noise (uV)'
        
        self.row_slice = slice( None )
        self.col_slice = slice( None )
        
    def set_row_slice( self , start=None , end=None , step=None ):
        self.row_slice = slice( start , end , step )
        
    def set_col_slice( self , start=None , end=None , step=None ):
        self.col_slice = slice( start , end , step )
        
    def plot_cchist( self , noise_lims=None , title='Noise vs. Offset' , figname=None ):
        if noise_lims is not None:
            self.noise_lims = noise_lims
        
        self.cchist = plotting.ColorHist( )
        offset_data = self.offset[ self.row_slice , self.col_slice ]
        noise_data  = self.noise [ self.row_slice , self.col_slice ]
        
        self.cchist.set_data( offset_data.flatten() , title )
        # self.cchist.set_bins( 0 , 16383 , 101 , xl )
        self.cchist.set_bins( 0 , self.DR , self.bins , self.offset_label )
        self.cchist.set_colordata( noise_data.flatten() , self.noise_label , self.noise_lims )
        self.cchist.plot( figname=figname )
        
    def plot_by_count( self , fit=True , figname=None ):
        # Ignore the first and last bins which should be pinned pixels.
        count = self.cchist.n[1:-1]
        means = self.cchist.meanvals[1:-1]
        msk   = means > 10
        
        if fit:
            bybin      = plotting.Lines( )
            bybin.grid = True
            #msk   = self.cchist.meanvals > 10
            #bybin.add_series( self.cchist.n[msk] , self.cchist.meanvals[msk] , ls='' , marker='o' , label='Data' )
            #bybin.add_series( self.cchist.n[1:-1][msk[1:-1]] , self.cchist.meanvals[1:-1][msk[1:-1]] , ls='' , marker='o' , label='Data' )
            bybin.add_series( count[msk] , means[msk] , ls='' , marker='o' , label='Data' )
            bybin.ylabel = self.noise_label
            bybin.xlabel = 'Counts of pixels per bin'
            
            # Add in the fits
            # But first mask out what we don't want to see....not sure these assumptions will hold for all chips
            noise_mask = np.logical_and( means > self.noise_lims[0] , means < self.noise_lims[1] )
            count_mask = count > 100000
            mask       = np.logical_and( count_mask , noise_mask )

            #print len(mask)
            #print count[mask]
            #print means[mask]
            if mask.sum() > 0:
                slope, intercept, r, p, std_err = scipy.stats.linregress( count[mask] , means[mask] )

                x = np.arange( 0 , count[mask].max() , count[mask].max()/100. )
                bybin.add_series( x , (intercept + slope*x) , ls='-' , label='Fit [R^2 = %.2f]' % r**2 )

                bybin.title = 'Masked linear Fit: %.2f + %.2fe-6 * x | R^2 = %.2f' % ( intercept , slope*1e6 , r**2 )
            else:
                slope     = 0
                intercept = 0
                r         = 0
                p         = 1
                std_err   = 0
                bybin.title = 'Masked linear Fit'
                
            bybin.ylims = self.noise_lims
            bybin.plot( figname=figname )
            
            # Note, slope multiplied by 10^6 to get whole numbers.
            fitdata = {'slope':slope*1e6 , 'intercept':intercept , 'rsq':r**2 , 'pval':p , 'std_err':std_err}
            return fitdata
        
        else:
            bybin      = plotting.Plot( )
            bybin.grid = True
            bybin.set_x( count[msk] , 'Counts of pixels per bin' )
            bybin.set_y( means[msk] , self.noise_label )
            bybin.plot( figname=figname )
            
            # If we're not fitting anything, return None.
            return None

    def plot_by_bin( self , figname=None , fix_y=False ):
        nvo      = plotting.Plot( )
        nvo.grid = True
        nvo.set_x ( self.cchist.midpoints , self.offset_label )
        nvo.set_y ( self.cchist.meanvals  , self.noise_label )
        if fix_y:
            nvo.ylims = self.noise_lims
        nvo.plot  ( figname=figname )
        
    def analyze( self , outputdir , quadrant=None ):
        ''' 
        Analyze typical behaviors for either full chip or quadrants.
        quadrant values of 1,2,3,4 are accepted, otherwise will analyze full chip
        '''
        # Analyze by quadrant
        #
        # *** I realize this isn't Cartesian style.  Sorry! *** - Phil
        #
        # (0,0) Origin in lower left
        #------#------# 
        #  Q4  |  Q3  #
        #------M------#
        #  Q1  |  Q2  #
        #------#------# Software row zero
        
        # This assumes that the input data is full chip data, or in the least 2D data.
        M = [ self.noise.shape[0]/2 , self.noise.shape[1]/2 ]
        
        if (quadrant is not None) and int(quadrant) in [1,2,3,4]:
            Q = int(quadrant)
            if Q == 1:
                self.set_row_slice( end=M[0] )
                self.set_col_slice( end=M[1] )
            if Q == 2:
                self.set_row_slice( end=M[0] )
                self.set_col_slice( start=M[1] )
            if Q == 3:
                self.set_row_slice( start=M[0] )
                self.set_col_slice( start=M[1] )
            if Q == 4:
                self.set_row_slice( start=M[0] )
                self.set_col_slice( end=M[1] )
                
            self.plot_cchist  ( title='Q%d Noise vs. Offset' % Q, figname=os.path.join(outputdir,'q%d_no_cchist.png' % Q))
            met = self.plot_by_count( figname=os.path.join( outputdir , 'q%d_noise_vs_bincount_fitted.png' % Q ) )

            # Save fit metrics
            for metric in met:
                self.metrics['q%d_noise_vs_bincount_%s' % (Q,metric)] = met[metric]
            
            self.plot_by_count( fit=False , figname=os.path.join( outputdir , 'q%d_noise_vs_bincount.png' % Q ) )
            self.plot_by_bin  ( figname=os.path.join( outputdir , 'q%d_noise_vs_offset.png' % Q ) )
            self.plot_by_bin  ( figname=os.path.join( outputdir , 'q%d_noise_vs_offset_fixed.png' % Q ) , fix_y=True )
        else:
            self.plot_cchist  ( title='Full Chip Noise vs. Offset' , figname=os.path.join(outputdir,'fc_no_cchist.png') )
            met = self.plot_by_count( figname=os.path.join( outputdir , 'fc_noise_vs_bincount_fitted.png' ) )
            
            # Save fit metrics
            for metric in met:
                self.metrics['fc_noise_vs_bincount_%s' % metric] = met[metric]
                
            self.plot_by_count( fit=False , figname=os.path.join( outputdir , 'fc_noise_vs_bincount.png' ) )
            self.plot_by_bin  ( figname=os.path.join( outputdir , 'fc_noise_vs_offset.png'   ) )
            self.plot_by_bin  ( figname=os.path.join( outputdir , 'fc_noise_vs_offset_fixed.png'   ) , fix_y=True )

            # Save the data points from noise vs. offset for json-based scraping and comparison across runs later:
            self.metrics['fc_nvo_offsets'] = list( self.cchist.midpoints )
            self.metrics['fc_nvo_noises' ] = list( self.cchist.meanvals  )
