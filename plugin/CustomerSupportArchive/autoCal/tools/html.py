import traceback as tb

class classproperty( property ):
    ''' @classmethod + @property '''
    def __get__( self, cls, owner ):
        return classmethod( self.fget ).__get__( None, owner )()

class Tag:
    ''' Helper class for creating a tag.  Not usually called directly '''
    def __init__( self, tag, close='>', **attrs ):
        self.closer = close
        self.tag    = tag
        try:
            attrs['class'] = attrs.pop( 'class_' )
        except:
            pass
        self.attrs  = attrs
        
    def __str__( self ):
        out = '<{}'.format( self.tag )
        for key, val in self.attrs.items():
            if key in ( 'class', 'id' ):
                val = self.to_str( val )
            if val:
                out += ' {}="{}"'.format( key, val )
            else:
                out += ' {}'.format( key )
        out += self.closer
        return out
    
    @property
    def closetag( self ):
        return '</{}>'.format( self.tag )
    
    @staticmethod
    def to_str( val, sep=' ' ):
        ''' Safely convert either a list-like or string to a string '''
        try:
            val + ''
            return val
        except:
            # must be list-like
            return sep.join( val )   

class HtmlBlock( object ):
    ''' Base class.  Do not call this directly.  At minimum, lines needs to be overwritten and defined as a property '''
    def __init__( self, **attrs ):
        self.attrs = attrs
    
    @property
    def lines( self ):
        # Overwrite this method to return a list of lines
        raise NotImplementedError
    
    def __str__( self ):
        try:
            return '\n'.join( [ str(l) for l in self.lines ] )
        except:
            print( "Error writing to string" )
            print( self.lines )
            raise
    
    def append( self, prop, val ):
        try:
            getattr( self, prop ).append( val )
        except AttributeError:
            setattr( self, prop, [ val ] )
            
    @staticmethod
    def to_list( val ):
        ''' Safely convert a list-like or string to a list '''
        try:
            return val.lines
        except AttributeError:
            pass
        try:
            val + ''
            return [ val ]
        except TypeError:
            pass
        return list( val )

    def __add__( self, rhs ):
        try:
            return rhs + self.lines 
        except TypeError:
            raise TypeError( 'Cannot add {} to {}'.format( rhs.__class__, self.__class__ ) )

    def __radd__( self, lhs ):
        try:
            return lhs + self.lines
        except TypeError:
            raise TypeError( 'Cannot add {} to {}'.format( self.__class__, lhs.__class__ ) )

    def __iadd__( self, lhs ):
        try:
            lhs += self.lines
            return
        except TypeError:
            raise TypeError( 'Cannot add {} to {}'.format( self.__class__, lhs.__class__ ) )

class Document( HtmlBlock ):
    ''' Root level document.  All pages should start here and call .add to add elements '''
    def add( self, content ):
        self.append( 'content', content )

    @property
    def lines( self ):
        items = getattr( self, 'content', [] )
        lines = []
        for item in items:
            try:
                lines += item.lines
            except AttributeError:
                try: 
                    item + ''
                    lines += [ item ]
                except:
                    lines += list( item )
        return lines

class Div( Document ):
    ''' <div> tag '''
    def __init__( self, content=(), **attrs ):
        super(Div,self).__init__( **attrs )
        try:
            self.add( content + '' )
        except TypeError:
            for item in content:
                self.add( item )

    def add( self, content ):
        self.append( 'content', content )

    @property
    def lines( self ):
        items = getattr( self, 'content', [] )
        lines  = [ Tag('div', **self.attrs) ]
        for item in items:
            try:
                lines += item.lines
            except AttributeError:
                try: 
                    item + ''
                    lines += [ item ]
                except:
                    lines += list( item )
        lines += [ lines[0].closetag ]   
        return lines

class Table( HtmlBlock ):
    ''' <table> objects '''
    def __init__( self, rows=(), **attrs ):
        super(Table,self).__init__( **attrs )
        for row in rows:
            self.add_row( row )
            
    def add_row( self, row ):
        self.append( 'rows', row )
        
    @property
    def lines( self ):
        lines  = [ Tag('table', **self.attrs) ]
        try:
            lines += self.rows
        except AttributeError:
            pass
        lines += [ lines[0].closetag ]   
        return lines

    @classmethod
    def simple_table( cls, content, headerrow=True, headercol=True ):
        table = cls()
        for i, row in enumerate( content ):
            trow = TableRow()
            for j, col in enumerate( row ):
                header = ( headerrow and not i ) or ( headercol and not j )
                cell = TableCell( content, header=header )
                trow.add_cell( cell )
            table.add_row( trow )
        return table


    @classmethod
    def style_zebra_( cls, class_='', id='' ):
        ''' a style object to display with zebra formatting '''
        sel = 'table'
        if class_:
            sel += '.{}'.format( class_ )
        elif id:
            sel += '#{}'.format( id )

        style = Style()

        sels = '{s}, {s} tr, {s} th, {s} td'.format( s=sel )
        style.add_css( sels, { 'border': '1px', 
                               'border-collapse': 'collapse', } )
        sels = '{} tr:nth-child(even)'.format( sel )
        style.add_css( sels, { 'border': '1px', 
                               'border-collapse': 'collapse', } )
        return style

    def style_zebra( self ):
        ''' a style object to display with zebra formatting '''
        return self.style_zebra_( self.attrs.get( 'class', '' ), self.attrs.get( 'id', '' ) )

class TableRow( HtmlBlock ):
    ''' <tr> '''
    def __init__( self, cells=(), **attrs ):
        super(TableRow,self).__init__( **attrs )
        for cell in cells:
            self.add_cell( cell )
    
    def add_cell( self, cell ):
        self.append( 'cells', cell )   
    
    @property
    def lines( self ):
        lines  = [ Tag('tr', **self.attrs) ]
        try:
            lines += self.cells
        except AttributeError:
            pass
        lines += [ lines[0].closetag ] 
        return lines
TR = TableRow
        
class TableCell( HtmlBlock ):
    ''' <th> or <td>, selected by the header flag '''
    def __init__( self, body=(), header=False, **attrs ):
        super(TableCell,self).__init__( **attrs )
        self.header = header
        self.add_body( body )
        
    def add_body( self, body ):
        self.body = body
    
    @property
    def lines( self ):
        tag = 'th' if self.header else 'td'
        lines  = [ Tag(tag, **self.attrs) ]
        try:
            lines += self.to_list( self.body )
        except AttributeError:
            pass
        lines += [ lines[0].closetag ]  
        return lines
TC = TableCell
    
class List( HtmlBlock ):
    ''' <ul> or <ol> objects '''
    def __init__( self, items=(), ordered=False, **attrs ):
        super(List,self).__init__( **attrs )
        self.ordered = ordered
        for item in items:
            self.add_item( item )
    
    def add_item( self, item ):
        self.append( 'items', item )

    @classmethod
    def css_flat_( cls, ordered, class_='', id='' ):
        ''' Return CSS selector and styles to display as an inline list '''
        sel = cls._get_tag( ordered )
        if class_:
            sel += '.{}'.format( class_ )
        elif id:
            sel += '#{}'.format( id )
        sel += ' li'
        return sel, { 'display': 'inline' }

    def css_flat( self ):
        ''' Return CSS selector and styles to display as an inline list '''
        return self.css_flat_( self.ordered, self.attrs.get( 'class', '' ), self.attrs.get( 'id', '' ) )

    @staticmethod
    def _get_tag( ordered ):
        return 'ol' if ordered else 'ul'

    @property
    def _tag( self ):
        return self._get_tag( self.ordered )

    @property
    def lines( self ):
        lines  = [ Tag(self._tag, **self.attrs) ]
        try:
            lines += self.items
        except AttributeError:
            pass
        lines += [ lines[0].closetag ]  
        return lines  
    
class ListItem( HtmlBlock ):     
    ''' <li> objects '''
    def __init__( self, body='', **attrs ):
        super(ListItem,self).__init__( **attrs )
        self.add_body( body )

    def add_body( self, body ):
        self.body = body

    @property
    def lines( self ):
        lines  = [ Tag('li', **self.attrs) ]
        try:
            lines += self.to_list( self.body )
        except AttributeError:
            pass
        lines += [ lines[0].closetag ]  
        return lines      
LI = ListItem

class Link( HtmlBlock ):
    ''' <a> tag '''
    def __init__( self, href, **attrs ):
        super( Link, self ).__init__( href=href, **attrs )
        self.href = href
    
    def add_body( self, body ):
        self.body = body
    
    @property
    def lines( self ):
        lines  = [ Tag('a', **self.attrs) ]
        try:
            body = self.to_list( self.body )
        except AttributeError:
            body = []
        lines += body
        lines += [ lines[0].closetag ]  

        # Collapse to single line
        if len(body) <= 1:
            lines = [ ''.join([ str(l) for l in lines ]) ]
        return lines
A = Link
    
class Image( HtmlBlock ):
    ''' <img> tag.  
        Image.as_link will return the image wrapped in <a> tags '''

    def __init__( self, src, **attrs ):
        super( Image, self ).__init__( src=src, **attrs )
        self.src = src
    
    def as_link( self ):
        link = Link( self.src, target="blank" )
        link.add_body( self.lines )
        return link
    
    @property
    def lines( self ):
        return [ Tag( 'image', close='/>', **self.attrs ) ]

class Style( HtmlBlock ):
    ''' <style> block '''
    def add_css( self, selector, styles ):
        ''' Input CSS selector and dictionary of attributes '''
        self.append( 'css', ( selector, styles ) )

    @property
    def lines( self ):
        lines = [ Tag( 'style' ) ]
        css = getattr( self, 'css', [] )
        for selector, style in css:
            styles = [ '{}: {};'.format( k, v ) for k,v in style.items() ]
            lines += [ '{} {{ {}'.format( selector, styles[0] ) ]
            for s in styles[1:]:
                lines += [ ' '*(len(selector)+3) + s ]
            lines[-1] += ' }'
        lines += [ lines[0].closetag ]  
        return lines

    def make_pair( self, key, val ):
        return '{}: {};'.format( key, val )

class Header( HtmlBlock ):
    ''' <h#> tag '''
    def __init__( self, body='', h=1, **attrs ):
        super( Header, self ).__init__( **attrs )
        self.add_body( body )
        self.level = h

    def add_body( self, body ):
        self.body = body

    @property
    def lines( self ):
        lines  = [ Tag('h{}'.format( self.level ), **self.attrs) ]
        try:
            body = self.to_list( self.body )
        except AttributeError:
            body = []
        lines += body
        lines += [ lines[0].closetag ]  

        # Collapse to single line
        if len(body) <= 1:
            lines = [ ''.join([ str(l) for l in lines ]) ]
        return lines      

class HorizontalRule( HtmlBlock ):
    ''' <hr> tag '''
    def __init__( self, **attrs ):
        super( HorizontalRule, self ).__init__( **attrs )
        
    @property
    def lines( self ):
        lines  = [ Tag('hr', **self.attrs) ]
        return lines      

class Break( HtmlBlock ):
    ''' <br> tag '''
    def __init__( self, **attrs ):
        super( Break, self ).__init__( **attrs )
        
    @property
    def lines( self ):
        lines  = [ Tag('br', **self.attrs) ]
        return lines      

class Paragraph( HtmlBlock ):
    ''' <p> tag '''
    def __init__( self, body='', **attrs ):
        super( Paragraph, self ).__init__( **attrs )
        self.add_body( body )
        
    def add_body( self, body ):
        self.body = body
        
    @property
    def lines( self ):
        lines  = [ Tag('p', **self.attrs) ]
        try:
            body = self.to_list( self.body )
        except AttributeError:
            body = []
        lines += body
        lines += [ lines[0].closetag ]  
        
        # Collapse to single line
        if len(body) <= 1:
            lines = [ ''.join([ str(l) for l in lines ]) ]
        return lines      


H  = Header
HR = HorizontalRule
BR = Break
P  = Paragraph


