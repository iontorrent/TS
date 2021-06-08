import numpy as np
from PIL import Image
import matplotlib
import sys

class ImageParser:
    def __init__( self, filename, row_start_scalar=None, col_start_scalar=None ):
        ''' This class is meant to take in a filepath to an image, import as RGB using PIL,
            and then map the RGB values to single floats per pixel on a normalized 0-1 scale
            related to a matplotlib colormap

            Upon initialization, the image file will be imported and converted to RGB
            --> no other processing is automatic
            
            Subclasses should be created to handle specific cases of images
            --> initialization arguments include row_start_scalar and col_start_scalar
                --> if None, the search for the array begins in the center of the image
                --> floats between 0 and 1 can be chosen to pick a different relative start point

            A possible subclass to find colorbars could be defined through inheritence and 
            adjustment of start_scalars
        '''
        self.figbox = {}
        self.filename = filename
        # row start and col start can range from 0-1 for control of img find start
        self.row_start_scalar = row_start_scalar
        self.col_start_scalar = col_start_scalar
        # make sure image is RGB
        self.im = Image.open( filename ).convert("RGB")
        self.col_max = self.im.size[0]
        self.row_max = self.im.size[1]
        print( self.col_max, self.row_max )

    def _get_coordinates( self, borderless=False ):
        ''' determines the coordinates for the edge of the plot
            The algorithm works by searching for grayscale to define the edges
        
            input:  borderless = bool
                --> short-circuits algorithm if image is known to be borderless
                    --> otherwise algorithm will cycle through entire image dimensions

            output: None
                --> populates figbox dictionary with locations of image border
        '''
        # some images have no border and it would be nice to run them through the same pipeline
        if borderless:
            self.figbox['left']     = 0
            self.figbox['top']      = 0
            self.figbox['right']    = self.col_max
            self.figbox['bot']      = self.row_max
            return

        # Start in the middle of the image.  This will certainly be in the plot
        if self.row_start_scalar is not None:
            row_start = int( self.row_max*self.row_start_scalar )
        else:
            # start at midpoint
            row_start = int( self.row_max*0.5 )

        if self.col_start_scalar is not None:
            col_start = int( self.col_max*self.col_start_scalar )
        else:
            # start at midpoint
            col_start = int( self.col_max*0.5 )

        # Scan left to find the edge
        test = 0.9* ( 255**2 + 255**2 + 255**2 )
        for col in range( col_start, -1, -1 ):
            px = self.im.getpixel( ( col, row_start ) ) 
            t = px[0]**2 + px[1]**2 + px[2]**2
            if px[0]==px[1] and px[1]==px[2] and t<test:
                self.figbox['left'] = col + 1
                break
            elif col==0:
                self.figbox['left'] = col
        # Scan up to find the top
        for row in range( row_start, -1, -1 ):
            px = self.im.getpixel( ( col_start, row ) ) 
            t = px[0]**2 + px[1]**2 + px[2]**2
            if px[0]==px[1] and px[1]==px[2] and t<test:
                self.figbox['top'] = row + 1
                break
            elif row==0:
                self.figbox['top'] = row
        # Scan right to find the edge
        for col in range( col_start, self.col_max, 1 ):
            px = self.im.getpixel( ( col, row_start ) )
            t = px[0]**2 + px[1]**2 + px[2]**2
            if px[0]==px[1] and px[1]==px[2] and t<test and col>self.figbox['left']:
                self.figbox['right'] = col - 1
                break
            elif col==self.col_max-1:
                self.figbox['right'] = col
        # Scan down to find the bottom
        for row in range( row_start, self.row_max, 1 ):
            px = self.im.getpixel( ( col_start, row ) )
            t = px[0]**2 + px[1]**2 + px[2]**2
            if px[0]==px[1] and px[1]==px[2] and t<test and row>self.figbox['top']:
                self.figbox['bot'] = row - 1 
                break
            elif row==self.row_max-1:
                self.figbox['bot'] = row

    def _read_color_data( self, cmap='jet', N=256 ):
        ''' Reads the color data and interprets to values 
            
            input:  cmap    = matplotlib cmap name
                    N       = number of color levels to use for parsing image

            output: None
                --> generates attributes 'data' and 'subimage'
                    --> data is an np.array with single values (normalized 0-1) per pixel
                    --> subimage is a cropping of the input image to the data array (stored as PIL RGB image)
        '''
        # make a colormap image for reference
        # 1) get matplotlib colormap module
        cm          = matplotlib.cm
        # 2) make a len N array linear array of values between 0-1
        ref_range   = np.linspace(0,1,N)
        # 3) use the array to get 256 RGB colors spanning the range of a chosen colormap
        #   NOTE: multiply by 255 --> cmap rgb values are normalized 0-1, but we need 0-255 for 8-bit RGB
        #   NOTE: Also, we need to drop the 4th value in the cmap (alpha--> transparency)
        ref_rgb     = 255*np.array([ cm.get_cmap(name=cmap, lut=N)(v)[:3] for v in ref_range ], dtype=np.float)
        # 4) get the subimage (leave in PIL format)
        subim = self.im.crop((self.figbox['left'], self.figbox['top'], self.figbox['right'], self.figbox['bot']))
        # 5) Match colors of subim to ref_rgb using distance and assign values by index
        #   5a. define color assignment and a dictionary for a cache-like lookup to speed the processing
        color_cache = {}
        def color_assign( t ):
            # t = input target color
            try: 
                return color_cache[str(t)]
            except KeyError:
                distances = [ np.array([(t[i]-r[i])**2 for i in range(3)], dtype=np.float).sum() for r in ref_rgb ]
                color_cache[str(t)] = distances.index( min(distances) ) /np.float(N)
                return color_cache[str(t)]
        #   5b. define temp np.array of subim, loop through and assign colors
        temp = np.array( subim )
        data = np.zeros( temp.shape[:2] )
        for i, row in enumerate( temp[:] ):
            for j, val in enumerate( row[:] ):
                data[i][j] = color_assign(val)
        # 6) assign attributes data and subim
        self.data   = data
        self.subim  = subim

        
