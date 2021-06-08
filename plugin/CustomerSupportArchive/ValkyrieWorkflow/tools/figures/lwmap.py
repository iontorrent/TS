import Image, ImageDraw, ImageFont
import numpy as np
import os

moduleDir = os.path.abspath( os.path.dirname( __file__ ) )

def lotmap( data, waferlist=False, dataargs={}, plotargs={} ):
    ''' Plot all wafers represented in the data.
    waferlist = False: 
        data is a list of dictionaries with fields: 
          'wafer'
    waferlist = True:
        data is a list or set containing all wafers to plot
    '''

    boxsize   = plotargs.get( 'boxsize', 100 )
    linewidth = plotargs.get( 'linewidth', 3 )
    fontsize  = plotargs.get( 'fontsize', 25 )

    if waferlist:
        wafer_list = []
        for d in data:
            try:
                wafer_list.append( int( d ) )
            except TypeError:
                pass
    else:
        wafer_col  = dataargs.get( 'wafer', 'wafer' )
        wafer_list = []
        for d in data:
            try:
                wafer_list.append( int( d[wafer_col] ) )
            except TypeError:
                pass
        #wafer_list = [ int(d[wafer_col]) for d in data ]

    def to_xy( index0 ):
        x = index0%5
        y = index0/5
        return x, y

    im = Image.new( "RGB", (boxsize*5, boxsize*5), "red" )
    draw = ImageDraw.Draw( im )

    # Fill in boxes
    for wafer in range( 25 ):
        if (wafer+1) in wafer_list:
        #if data[wafer]:
            x, y = to_xy(wafer)
            draw.rectangle( ( ( x*boxsize, y*boxsize ), 
                              ( (x+1)*boxsize, (y+1)*boxsize ) ) , 
                            outline ="green", 
                            fill    = 'green' )


    # Draw the wafer numbers
    font = ImageFont.truetype( '%s/arial.ttf' % moduleDir, fontsize )
    for wafer in range( 25 ):
        x, y = to_xy(wafer)
        y = (y+0.5)*boxsize
        x = (x+0.5)*boxsize
        wfrstr = str( wafer + 1 )
        # Center the text
        w, h = draw.textsize( wfrstr, font=font )
        x -= w/2
        y -= h/2
        draw.text( (x, y), wfrstr, font=font )

    # Draw the borders
    for i in range(6):
        draw.line( ( 0, i*boxsize, im.size[0], i*boxsize ), fill='black', width=linewidth )
        draw.line( ( i*boxsize, 0, i*boxsize, im.size[1] ), fill='black', width=linewidth )

    return im

def wafermap( data, dataargs={}, plotargs={} ):
    imsize    = plotargs.get( 'boxsize', 500 )
    linewidth = plotargs.get( 'linewidth', 2 )
    fontsize  = plotargs.get( 'fontsize', 20 )
    show_nums = plotargs.get( 'show_numbers', True )

    xcol     = dataargs.get( 'xcol', 'x' )
    ycol     = dataargs.get( 'ycol', 'y' )

    diameter  = imsize*0.9
    radius    = diameter/2
    xc = imsize/2 
    yc = imsize/2
    if show_nums:
        xc += ( imsize - diameter )/4.
        yc -= ( imsize - diameter )/4.

    im = Image.new( "RGB", (imsize, imsize), "white" )
    draw = ImageDraw.Draw( im )

    font = ImageFont.truetype( '%s/arial.ttf' % moduleDir, fontsize )

    # Draw the wafer
    draw.ellipse( ( xc-radius-linewidth/1.5, yc-radius-linewidth/1.5, 
                    xc+radius+linewidth/1.5, yc+radius+linewidth/1.5 ), 
                  fill='black', outline=None )
    draw.ellipse( ( xc-radius+linewidth/1.5, yc-radius+linewidth/1.5, 
                    xc+radius-linewidth/1.5, yc+radius-linewidth/1.5 ), 
                  fill=(200,200,200), outline=None )
    # White-out the active die
    for x in range(3):
        dx = 24*(3.5-x)
        dy = 20*(4.5-(2-x))
        dy *= radius/100.
        dx *= radius/100.
        draw.rectangle( ( xc-dy, yc-dx, xc+dy, yc+dx ), fill='white' )

    # Mark measurements
    for row in data:
        try:
            x = int(row[xcol])
            y = int(row[ycol])
        except TypeError:
            continue
        lx = 24*(3.5-x)*radius/100. + yc
        ly = 20*(y-4.5)*radius/100. + xc
        ux = lx - 24*radius/100.
        uy = ly + 20*radius/100.
        draw.rectangle( (ly, lx, uy, ux), fill=(0,255,255) )

    # Draw the grid 
    # Chip grid and conventional grid coordinats are flipped
    for y in range(10):
        dy = 20*(4.5 - y)
        dx = np.sqrt(100**2 - dy**2)
        dy *= radius/100.
        dx *= radius/100.
        draw.line( ( xc+dy, yc-dx, xc+dy, yc+dx ), fill='black', width=linewidth )
    for x in range(8):
        dx = 24*(3.5 - x)
        dy = np.sqrt(100**2 - dx**2)
        dy *= radius/100.
        dx *= radius/100.
        draw.line( ( xc-dy, yc+dx, xc+dy, yc+dx ), fill='black', width=linewidth )

    # Number the grid
    if show_nums:
        for y in range(-1, 9):
            if y == -1:
                y += 0.5
                text = 'y = '
            else:
                text = str(y)
            dy = 20*(4. - y)*radius/100.
            xp = ( imsize + yc + radius)/2.
            yp = xc - dy
            w, h = draw.textsize( text, font=font )
            xp -= h/2
            yp -= w/2
            draw.text( (yp, xp), text, fill='black', font=font )
        for x in range(8):
            if x == 7:
                x -= 0.5
                text = 'x ='
            else:
                text = str(x)
            dx = 24*(x-3.)*radius/100.
            yp = (xc - radius)/2.
            xp = yc - dx
            w, h = draw.textsize( text, font=font )
            xp -= h/2
            yp -= w/2
            draw.text( (yp, xp), text, fill='black', font=font )

    return im

def lotwafermap( data, dataargs={}, plotargs={} ):
    ''' takes a list of data and plots the boolean presence '''

    boxsize   = plotargs.get( 'boxsize', 100 )
    imsize    = boxsize*5
    linewidth = plotargs.get( 'linewidth', 2 )
    fontsize  = plotargs.get( 'fontsize', 15 )

    dataargs_default = { 'xcol': 'x', 
                         'ycol': 'y', 
                         'wafer': 'wafer' }
    dataargs_default.update( dataargs )
    dataargs = dataargs_default

    xcol     = dataargs['xcol']
    ycol     = dataargs['ycol']
    wafer    = dataargs['wafer']

    im = Image.new( "RGB", (imsize, imsize), "white" )
    draw = ImageDraw.Draw( im )
    font = ImageFont.truetype( '%s/arial.ttf' % moduleDir, fontsize )

    # Plot each wafer
    miniargs = {'boxsize': boxsize, 'linewidth':1, 'show_numbers':False }
    for wafer_id in range( 25 ):
        wafer_data = []
        for d in data:
            try:
                if int(d[wafer]) == wafer_id + 1:
                    wafer_data.append(d)
            except TypeError:
                pass
        wafer_im   = wafermap( wafer_data, dataargs=dataargs, plotargs=miniargs )
        
        x = wafer_id%5
        y = wafer_id/5

        xp = x*boxsize
        yp = y*boxsize

        im.paste( wafer_im, (xp, yp) )

        # Add the wafer number
        wfrstr = str( wafer_id + 1 )
        xp += boxsize/50.
        yp += boxsize/50.
        draw.text( (xp, yp), wfrstr, font=font, fill='black' )

    # Draw the borders
    for i in range(6):
        draw.line( ( 0, i*boxsize, im.size[0], i*boxsize ), fill='black', width=linewidth )
        draw.line( ( i*boxsize, 0, i*boxsize, im.size[1] ), fill='black', width=linewidth )

    return im

if __name__ == '__main__':
    data = [ { 'wafer': 2, 'x': 2, 'y': 5 }, 
             { 'wafer': 2, 'x': 4, 'y': 0 } ]
    
    lotmap( data ).show()

    lotwafermap( data ).show()

