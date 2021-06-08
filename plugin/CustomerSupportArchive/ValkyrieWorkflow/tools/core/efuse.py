# python2 and python3 compatible
class EfuseClass:
    """ A class for parsing efuses """
    normal = ('lot','wafer','assembly','part','comment','flowcell','y' ,'x' ,'bin','noise','softbin','chiptype','barcode')
    codes  = ('L:' ,'W:'   ,'J:'      ,'P:'  ,'C:'     ,'F:'      ,'Y:','X:','B:' ,'N:'   ,'SB:',    'CT:',     'BC:' )

    def __init__( self , fuse ):
        self.text = fuse
        fields = fuse.split(',')
        consumed = []
        # First generically set attributes:
        for f in fields:
            try:
                setattr( self , f.split(':')[0] , f.split(':')[1] )
            except:
                pass
        # Set familiar name handles
        for i in range(len(self.normal)):
            for j, f in enumerate( fields ):
                # Set default values
                setattr( self , self.normal[i] , 'NA' )
                if self.codes[i] in f:
                    setattr( self , self.normal[i] , f.replace(self.codes[i],'') )
                    consumed.append( j )
                    break
        # Record anything that we don't already have a field for
        misc = []
        for j, f in enumerate( fields ):
            if j not in consumed:
                misc.append( f )
        self.misc = ','.join( misc )
        # Dummy proof
        self.X = self.x
        self.Y = self.y
        self.hardbin = self.bin
        self.fuse = fuse

Efuse = EfuseClass
