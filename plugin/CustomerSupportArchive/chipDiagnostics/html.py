import numpy as np

class HTML:
    """ Class for writing generic html output files.  Built to work with handy table formatting. """
    def __init__( self , filepath ):
        self.filepath = filepath
        
        # Initialize the text to eventually be written.  This will be a list of 'sections.'
        self.text     = []
        
    def add( self , htmltext ):
        ''' adds new section of html text to self.text. '''
        self.text.append( htmltext )
        
    def make_header( self , title , use_courier=False , head=True , styles=[]):
        self.courier = False
        
        header = '<html>\n'
        if head:
            header += '<head><title>%s</title></head>\n' % title
            
        header += '''<body>
<style type="text/css">tr.d0 {background-color: #eee;}</style>'''

        if styles != []:
            for s in styles:
                header += s

        header += '<font size=5, face="Arial"><center>%s</center></font>' % title
        
        if use_courier:
            self.courier = True
            header += '''<font face="Courier">'''
            
        header += '''<hr>''' 
        
        self.text.append( header )
        return None
    
    def make_footer( self ):
        footer = ''
        if self.courier:
            footer += '</font>\n'
        footer += '</body></html>'
        
        self.text.append( footer )
        
    def write( self ):
        with open( self.filepath , 'w' ) as f:
            for x in self.text:
                f.write( x )
        return None
    
class table:
    """ Class to make handy tables.  This is going to be hacked at first. """
    def __init__( self , width=100 , zebra=False , border=0 , cellpadding=None):
        if cellpadding is None:
            self.start = '<table border="%s" cellspacing="0" width="%d%%"><tbody>' % (border,width)
        else:
            self.start = '<table border="%s" cellpadding="%s" width="%d%%"><tbody>' % (border,cellpadding,width)
        self.end   = '</tbody></table>'
        self.zebra = zebra
        
        # If zebra is turned on, then we start with a shaded row and use the shaded property to alternate between
        # shaded/white rows.
        if self.zebra:
            self.shaded = True
            
        # Initialize table rows
        self.rows = []
        
    def add_row( self , td_list=[] , widths=[] , cl=None , th=False ):
        # Zebra table will override input class option
        if self.zebra:
            if self.shaded:
                cl = 'd0'
            else:
                cl = None
            self.shaded = not self.shaded
            
        if cl is None:
            row = '<tr>'
        else:
            row = '<tr class="%s">' % cl
            
        # handle widths
        if widths == []:
            # Use equally spaced sections based on length of td_list
            fields = len(td_list)
            widths = list( np.tile( int( np.floor( 100 / fields ) ) , fields ) )
            
        if len(widths) != len(td_list):
            print( 'Error!  List of TD values and widths are not the same!' )
            return None
        
        pairs = zip( widths , td_list )
        for p in pairs:
            if th:
                row += '<th width="%d%%">%s</th>' % (p)
            else:
                row += '<td width="%d%%">%s</td>' % (p)
        row += '</tr>'
        
        # Add to rows list
        self.rows.append( row )
        
        return None
            
    def get_table( self ):
        ''' returns text of html code for table '''
        table  = ''
        table += '%s\n' % self.start
        
        for r in self.rows:
            table += '%s\n' % r
            
        table += '%s\n' % self.end
        
        return table
    
def image_link( imgpath , width=100 ):
    ''' Returns code for displaying an image also as a link '''
    text = '<a href="%s"><img src="%s" width="%d%%" /></a>' % ( imgpath, imgpath , width )
    return text
