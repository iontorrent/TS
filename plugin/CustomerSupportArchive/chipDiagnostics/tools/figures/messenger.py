try:
    from PIL import Image, ImageDraw, ImageFont
except:
    import Image, ImageDraw, ImageFont
import os

moduleDir = os.path.abspath( os.path.dirname( __file__ ) )
verbose = False

def annotate( txt ):
    if verbose:
        print( txt )

def ceildev( a, b ):
    return -(-a / b )

def msgfig( msg, plotargs={} ):
    imsize    = plotargs.get( 'imsize', 450 ) # int, or ( w, h)
    fontcolor = plotargs.get( 'color', 'white' )
    bkg       = plotargs.get( 'bkg', 'black' )
    fontsize  = plotargs.get( 'fontsize', 100 )
    margin    = plotargs.get( 'margin', 45 )
    autowrap  = plotargs.get( 'autowrap', True )
    if not hasattr( imsize, '__iter__' ):
        imsize = ( imsize, imsize ) # width, height 

    im   = Image.new( "RGB", (imsize), bkg )
    draw = ImageDraw.Draw( im )

    def wrap( msg, font ):
        maxwidth  = imsize[0] - 2*margin
        maxheight = imsize[1] - 2*margin
        # Coarsly wrap
        linewidth, lineheight = draw.textsize( msg, font=font )
        lineheight += font.getoffset( msg )[1]

        # Check if wrapping is necessary
        if linewidth < maxwidth:
            annotate( 'No wrapping required' )
            return [ msg, ]
        
        # Set target width of line
        numlines = ceildev( linewidth, maxheight )
        targetwidth = linewidth / numlines
        annotate( 'Expected number of lines: %i' % numlines )
        annotate( 'Expected width: %i' % targetwidth )

        # Make sure we can wrap within bounds
        maxlines = maxheight / lineheight
        if numlines > maxlines:
            annotate( 'Wrap failed because too many lines are expected' )
            return False
    
        # Make sure each word fits
        words = msg.split()
        word_widths, word_heights = zip( *[ draw.textsize(w, font=font) for w in words ] )
        space_width, space_height = draw.textsize( ' ', font=font )
        for ww in word_widths:
            if ww > maxwidth:
                annotate( 'Wrap failed because a word is longer than max length' )
                return False


        output = []
        line      = ''
        linewidth = 0
        annotate( 'Word lengths: ' )
        annotate( zip( words, word_widths ) )
        for word, ww in zip( words, word_widths ):
            # Check if word can be added without going over limit
            if linewidth + ww > maxwidth:
                annotate( 'Wrapping to next line (over max): %s' % word )
                output.append( line )
                line      = ''
                linewidth = 0
            # Check if we are alread over target
            if linewidth > targetwidth:
                annotate( 'Wrapping to next line (over target): %s' % word )
                output.append( line )
                line      = ''
                linewidth = 0
            # Check if adding word moves us further from target
            delta0 = linewidth - targetwidth
            delta1 = delta0 + ww + space_width
            delta0 = abs( delta0 )
            delta1 = abs( delta1 )
            if delta0 < delta1:
                annotate( 'Wrapping to next line (delta): %s' % word )
                annotate( 'Deltas: (%i, %i) ' % ( delta0, delta1 ) )
                output.append( line )
                line      = ''
                linewidth = 0
            # Add the line
            if line:
                line += ' '
                linewidth += space_width
            line += word
            linewidth += ww
        output.append( line )


        # Finally make sure we haven't gone over any limits
        # Initialize the height by subtracting the initial offset since it will be removed
        total_height = -font.getoffset(msg)[1]
        for line in output:
            linewidth, lineheight = draw.textsize( line, font=font )
            lineheight += font.getoffset( line )[1]
            if linewidth > maxwidth:
                annotate( 'Wrap failed because line "%s" is longer (%i) than max width (%i)' % ( line, linewidth, maxwidth ) )
                return False
            total_height += lineheight
        if total_height > maxheight:
            annotate( 'Wrap failed because total height (%i) excedes max hight (%i)' % ( total_height, maxheight ) )
            return False

        annotate( 'Returning at end of wrap' )
        return output

    def writelines( txt_arr, font ):
        line_widths, line_heights = zip( *[ draw.textsize(line, font=font) for line in txt_arr ] )
        line_offsets = [ font.getoffset( line ) for line in txt_arr ]
        annotate( 'Line Widths: ' )
        annotate( line_widths )
        annotate( 'Line Heights: ' )
        annotate( line_heights )
        total_height = sum( line_heights ) + sum( [ lo[1] for  lo in line_offsets[1:] ] )
        annotate( 'Total Height: %i' % total_height )
        top = imsize[1] / 2 - total_height / 2 - ( line_offsets[0][1] * 2 )
        for line, lw, lh, lo in zip( txt_arr, line_widths, line_heights, line_offsets ):
            top += lo[1]
            left = imsize[0] / 2 - lw/2 - lo[0]
            annotate( 'Origin: (%i, %i)' % ( left, top ) )
            draw.text( (left, top), line, fill=fontcolor, font=font )
            top += lh

    fontstep = 2
    for current_font in reversed( range( 6, fontsize+1, fontstep ) ):
        annotate( 'Current Font: %i' % current_font )
        font = ImageFont.truetype( '%s/arial.ttf' % moduleDir, current_font )
        output = wrap( msg, font )
        if output:
            annotate( 'Output: ' )
            annotate( output )
            writelines( output, font )
            annotate( 'Midline: %i' % (imsize[1]/2) )

            return im
    return False

def banner_fig( msg, image, plotargs={} ):
    imsize       = plotargs.get( 'imsize', 450 ) # int, or ( w, h)
    bannerheight = plotargs.get( 'bannerheight', 150 )
    bannermargin = plotargs.get( 'bannermargin', 15 )
    if not hasattr( imsize, '__iter__' ):
        imsize = ( imsize, imsize ) # width, height 

    annotate( 'Output Size: %s' % str(imsize) )

    # Load the image and rescale it
    try:
        im = Image.open( image )
        annotate( 'Original Image Size: %s' % str(im.size) )
        annotate( 'Banner Height: %i' % bannerheight )
        maxheight = imsize[1] - bannerheight
        annotate( 'Max Height: %i' % maxheight )

        ar_orig  = float( im.size[1] ) / im.size[0]
        ar_avail = float( maxheight ) / imsize[0]
        if ar_orig >= ar_avail:
            # image is taller than is available
            imwidth = int( im.size[0] * maxheight / im.size[1] )
            imheight = maxheight
        else:
            # Image is wider than is available
            imheight = int( im.size[1] * imsize[0] / im.size[0] )
            imwidth  = imsize[0]
        newsize = ( imwidth, imheight )
        annotate( 'Target image size: %s' % str( newsize ) )
        if im.size != newsize:
            im = im.resize( newsize )
            annotate( 'New Image Size: %s' % str(im.size) )
    except:
        figname = image.split( '/' )[-1]
        imheight = imsize[1] - bannerheight
        errorargs = { 'imsize': ( imsize[0], imheight  ), 
                      'margin': bannermargin, 
                      'margin': 15,
                      'bkg': 'white', 
                      'color': 'black' }
        im = msgfig( figname, plotargs=errorargs )
        annotate( 'image sized from except statement' )


    # make the banner
    top = ( imsize[1] + bannerheight - imheight ) / 2
    left = imsize[0]/2 - imwidth/2
    bannerheight = max( imsize[1] - imheight, bannerheight )
    annotate( 'Image Top: %i' % top )
    annotate( 'Banner Height: %i' % bannerheight )

    bannerargs = { 'imsize': ( imsize[0], bannerheight ), 
                   'margin': bannermargin }
    banner = msgfig( msg, plotargs=bannerargs )
    annotate( 'Banner Size: %s' % str( banner.size ) )

    # Assemble the figure
    fig  = Image.new( "RGB", imsize, 'white' )
    fig.paste( im, (left, top ) )
    fig.paste( banner, (0,0) )

    return fig

def split_msgfig( top_msg, bot_msg, plotargs={} ):
    imsize       = plotargs.get( 'imsize', 450 ) # int, or ( w, h)
    top_bkg      = plotargs.get( 'top_bkg', 'black' )
    top_color    = plotargs.get( 'top_color', 'white' )
    bot_bkg      = plotargs.get( 'bot_bkg', 'red' )
    bot_color    = plotargs.get( 'bot_color', 'black' )

    if not hasattr( imsize, '__iter__' ):
        imsize = ( imsize, imsize ) # width, height 

    top_size = ( imsize[0], imsize[1]/2 )
    top_args = { 'bkg':    top_bkg, 
                 'color':  top_color, 
                 'imsize': top_size }
    top      = msgfig( top_msg, plotargs=top_args )

    bot_size = ( imsize[0], ceildev( imsize[1], 2 ) ) # Just to make sure we will the space from rounding errors
    bot_args = { 'bkg':    bot_bkg, 
                 'color':  bot_color, 
                 'imsize': bot_size }
    bot      = msgfig( bot_msg, plotargs=bot_args )

    # Assemble the figure
    fig  = Image.new( "RGB", imsize, 'white' )
    fig.paste( top, (0, 0) )
    fig.paste( bot, (0, top_size[1] ) )

    return fig

def test_error():
    ''' Makes the test error figure '''
    plotargs = { 'color': 'black', 
                 'bkg':   'red' }
    msg = 'TEST ERROR'
    return msgfig( msg, plotargs )

def chip_error():
    ''' Makes the chip error figure '''
    plotargs = { 'color': 'white', 
                 'bkg':   'red' }
    msg = 'CHIP ERROR'
    return msgfig( msg, plotargs )

def error():
    ''' Makes the error figure '''
    plotargs = { 'color': 'white', 
                 'bkg':   'red' }
    msg = 'ERROR'
    return msgfig( msg, plotargs )

def xfer_error():
    ''' Makes the error figure '''
    plotargs = { 'color': 'white', 
                 'bkg':   'red' }
    msg = 'ERROR transfering results to server'
    return msgfig( msg, plotargs )

def hard_bin( hb, bintype=None ):
    ''' Makes the hardbin figure '''
    #lut = { 1: ( (  0,255,  0), 'black' ),    # green
    #        2: ( (  0,102,255), 'white' ),  # darkish blue
    #        3: ( (  0,255,255), 'black' ),  # cyan, 
    #        4: ( (255,255,  0), 'black' ) } # fucia
    lut = { 1: ( (  0,255,  0), 'black' ) }    # green
    default = (  (128, 22, 32), 'white' ) # dark red
    if not hb:
        return error()
    colors = lut.get( hb, default )

    plotargs = { 'color': colors[1], 
                 'bkg':   colors[0], 
                 'fontsize': 200 }
    if bintype:
        msg = '%s %s' % ( bintype.upper(), hb )
    else:
        msg = str(hb)

    return msgfig( msg, plotargs )

def banner_hard_bin( bannermsg, hb, bintype=None, bannercolor='black' ):
    ''' Makes the hardbin figure 
    Banner will have white text on the specified background '''
    lut = { 1: ( (  0,255,  0), 'black' ),    # green
            2: ( (  0,102,255), 'white' ),  # darkish blue
            3: ( (  0,255,255), 'black' ),  # cyan, 
            4: ( (255,255,  0), 'black' ) } # fucia
    default = ( (255,255,204), 'black' )
    if not hb:
        return error()
    colors = lut.get( hb, default )

    bannerargs = { 'imsize': ( 450, 150  ), 
                   'margin': 15,
                   'bkg': bannercolor, 
                   'color': 'white' }
    im = msgfig( bannermsg, plotargs=bannerargs )

    plotargs = { 'imsize': ( 450, 300 ), 
                 'color': colors[1], 
                 'bkg':   colors[0], 
                 'fontsize': 200 }
    if bintype:
        msg = '%s %s' % ( bintype.upper(), hb )
    else:
        msg = str(hb)

    binfig = msgfig( msg, plotargs )

    # Assemble the figure
    fig  = Image.new( "RGB", (450, 450), 'white' )
    fig.paste( binfig, (0, 150) )
    fig.paste( im, (0,0) )
    
    return fig

def success():
    ''' Makes the sucess figure '''
    plotargs = { 'color': 'black', 
                 'bkg':   (0,255,0) }
    msg = 'SUCCESS'
    return msgfig( msg, plotargs )

if __name__ == '__main__':
    import pdb
    verbose = True
    im = banner_fig('Hello World', '/results/do/not/exist/dne.jpg').show()
