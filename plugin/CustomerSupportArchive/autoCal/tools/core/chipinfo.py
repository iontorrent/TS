
def kwdataclass( cls ):
    class KWDataClass( object ):
        def __init__( self, **kwargs ):
            self._data = kwargs.copy() # dictionary of input elements
            for k, v in self._data.values:
                setattr( self, k, v )

    return KWDataClass

def dataclass( cls ):
    class DataClass( object ):
        def __init__( self, names, vals ):
            self._data = dict( zip( names.copy(), vals.copy() ) )
            for k, v in self._data.values:
                setattr( self, k, v )

    return DataClass

class DataClass( object ):
    def __init__( self, names, vals ):
        self._data = dict( zip( names.copy(), vals.copy() ) )
        for k, v in self._data.values:
            setattr( self, k, v )

class DataTables:
    pass
datatables = DataTables()

# Reference Pixels
labels = ( 'ref_rows', 'ref_cols', 'ref_array' )
refpix = {
  '550_16': ( 3, 15, '550_16' ), 
  '550_3490': ( 0, 15, '550_3490' ), 
  '550_3491': ( 0, 15, '550_3491' ), 
  '550_3525': ( 0, 15, '550_3525' ), 
  '560_3610': ( 0, None, None ), 
  'P22': ( 8, 0, None ), 
  'none': ( None, None, None ), 
}
#refpix = { k:DataClass(labels, v) for k, v in refpix.items() }
refpix = { k:dict(zip(labels, v)) for k, v in refpix.items() }
datatables.refpix = refpix



# Flowfront
labels = ( 'flowstandard', 'ff_fmin', 'ff_fmax', 'ff_step', 'ff_snum', 'ff_tmin', 'ff_tmax', 'ff_tstep', )
flowfront = {
  '314':  ( None, 6, 10, 1, 5, 6, 10, 1 ), 
  '316':  ( None, 6, 24, 2, 7, 6, 24, 2 ), 
  '318':  ( None, 8, 26, 2, 7, 8, 26, 2 ),
  'P0r':  ( 'P1_flow_standard', 17, 23, None, None, 17, 26, None ), 
  'P0':   ( None, 16, 28, None, None, 16, 28, None ), 
  'P1':   ( None, 17, 30, None, None, 17, 40, None ),
  'P12':  ( None, 17, 30, None, None, 17, 30, None ),
  'P1a':  ( None, 18, 24, None, None, 17, 40, None ),
  'P221': ( None, 10, 22, None, None, 10, 22, None ),
  'PQ':   ( None, 16, 40, None, None, 16, 40, None ),
  'tn':   ( None, None, None, None, None, None, None, None ),
  'dummy': ( None, 0, 0, None, None, 0, 0, None ),
}
#flowfront = { k:DataClass(labels, v) for k, v in flowfront.items() }
flowfront = { k:dict(zip(labels, v)) for k, v in flowfront.items() }
datatables.flowfront = flowfront

# Dimensions
labels = ( 'chipR', 'chipC', 'blockR', 'blockC', 'miniR', 'miniC', 'microR', 'microC', 'transpose', 'burger', 'tn', 'spatn', 'numprocs', 'maxprocs' )
dims = { 
       '314': (  None,  None, 1152, 1280, 128, 128,   12,   16, False,    None,   None,  'spa',    1,    1, ),
       '316': (  None,  None, 2640, 2736, 120, 114, None, None, False,    None,   None,  'spa',    1,    1, ),
       '318': (  3792,  3392, 3792, 3392,  79,  64, None, None, False,    None,   None,  'spa',    1,    1, ),
       '510': (   864,  7680,  864,  640,  72,  80,   12,   16, False,   '530', 'P1tn',  'spa',   16,   16, ),
       '520': (  1728,  7680,  864,  640,  72,  80,   12,   16, False,   '530', 'P1tn',  'spa',   16,   16, ),
       'R20': (  1728,  7680,  216,  640,  36,  80,   12,   16, False,    None, 'P1tn',  'spa',   16,   16, ),
       '521': (  1728,  7680,  864,  640,  72,  80,   12,   16, False,    None, 'P1tn',  'spa',   16,   16, ),
     '521v2': (  1664,  7680,  832,  640,  64,  80,   13,   16, False,    None, 'P1tn',  'spa',   16,   16, ),
        'P0': (  5328,  7680,  666,  640,  74,  64,    9,   10, False,    None, 'P1tn',  'spa',   16,   16, ),
       '530': (  5312,  7680,  664,  640,  83,  80,    8,   10, False,    None, 'P1tn',  'spa',   16,   16, ),
        'P1': ( 10656, 15456, 1332, 1288, 111,  92,   12,   14, False,    None, 'P1tn',  'spa',   12,   16, ),
  'Valkyrie': ( 10656, 15456, 1332, 1288, 111,  92,   12,   14,  True,    None, 'P1tn',  'spa',   12,   16, ),
   '541_nar': (  5328, 15456,  666, 1288, 111,  92,    9,   14, False,    None, 'P1tn',  'spa',   12,   16, ),
   '541_280': (  9376, 15456, 1172, 1288, 293,  92,    4,   14, False,    None, 'P1tn',  'spa',   12,   16, ),
    '550_16': ( 14208, 20544, 1776, 1712, 111, 107,   12,   16, False,    None, 'P1tn',  'spa',    8,   16, ),
  '550_3490': ( 14032, 20544, 1754, 1712, 877, 107,    2,   16, False,    None, 'P1tn',  'spa',    8,   16, ),
  '550_3491': ( 14040, 20544, 1755, 1712, 117, 107,   13,   16, False,    None, 'P1tn',  'spa',    8,   16, ),
  '550_3525': ( 13632, 20544, 1704, 1712,  71, 107,   12,   16, False,    None, 'P1tn',  'spa',    8,   16, ),
  '560_3610': ( 18528, 30912, 2316, 2576, 193, 184,   12,   16, False,    None, 'P1tn',  'spa',    4,   16, ),
      'P221': ( 21312, 30912, 2664, 2576, 111,  92,   12,   14, False,    None, 'P1tn',  'spa',    4,   16, ),
      'P222': ( 21296, 30912, 2662, 2576, 121,  92,   11,   14, False,    None, 'P1tn',  'spa',    4,   16, ),
    'P222os': ( 17664, 30912, 2208, 2576, 184, 184,   12,   14, False,'550_16', 'P1tn',  'spa',    4,   16, ),
   'P222_16': ( 21296, 30912, 2662, 2576, 242, 184,   11,   14, False,    None, 'P1tn',  'spa',    4,   16, ),
     'P222a': ( 21072, 30912, 2634, 2576, 439, 184,    6,   14, False,    None, 'P1tn',  'spa',    4,   16, ),
     'P222b': ( 21072, 30912, 2634, 2576, 439, 368,    6,    8, False,    None, 'P1tn',  'spa',    4,   16, ),
       'P12': ( 10656, 15456, 1332, 1288, 111,  92,   12,   14, False,    None, 'P1tn',  'spa',   12,   16, ),
        'tn': (   800,  1200,  100,  100, 100, 100,   10,   10,  None,    None, 'self',   None, None, None, ),
       'spa': (   800,  1200,  100,  100, 100, 100,   10,   10,  None,    None,   None, 'self', None, None, ),
       's10': (    64,   120,    8,   10,   1,   1,    1,    1, False,    None, 'P1tn',  'spa',   16,   16, ),
       's12': (    96,   168,   12,   14,   1,   1,    1,    1, False,    None, 's1tn',  'spa',   12,   16, ),
       's22': (   176,   336,   22,   28,   1,   1,    1,    1, False,    None, 's1tn',  'spa',    6,   16, ),
      's1tn': (     8,    12,    1,    1,   1,   1, None, None, False,    None, 'self',   None, None, None, ),
     'dummy': (     0,     0,    0,    0,   0,   0,    0,    0, False,    None,   None,   None,    0,    0, ),
}
#dims = { k:DataClass(labels, v) for k, v in dims.items() }
dims = { k:dict(zip(labels, v)) for k, v in dims.items() }
datatables.dims = dims
# Do standard calculations
for k, row in dims.items():
    try:
        row['yBlocks']      = row['chipR'] / row['blockR']
        row['xBlocks']      = row['chipC'] / row['blockC']
        row['RBlocks']      = row['yBlocks']
        row['CBlocks']      = row['xBlocks']
    except:
        pass
    try:
        row['miniRblocks']  = row['chipR'] / row['miniR']
        row['miniCblocks']  = row['chipC'] / row['miniC']
        row['blockminiRblocks']  = row['blockR'] / row['miniR']
        row['blockminiCblocks']  = row['blockC'] / row['miniC']
    except:
        pass
    try:
        row['microRblocks'] = row['chipR'] / row['microR']
        row['microCblocks'] = row['chipC'] / row['microC']
        row['blockmicroRblocks'] = row['blockR'] / row['microR']
        row['blockmicroCblocks'] = row['blockC'] / row['microC']
    except:
        pass

# Buffering
labels = ( 'flowcorr_ecc', 'flowcorr_wt', 'startframe' )
buffering = {
  '314': ( 'flowcorr_314', None, 0 ), 
  '318': ( 'flowcorr_318', None, 0 ), 
  'P1':  ( 'fc_flowcorr', 'flowcorr_540_rev1', 15 ), 
  'R1020': ( 'fc_flowcorr_R1020', None, 15 ), 
  '521': ( 'fc_flowcorr_521', None, 15 ), 
  '521v2': ( 'fc_flowcorr_521v2', None, 15 ), 
  'Valkyrie': ( 'valkyrie_flowcorr_rev1', None, 15 ), 
  'super': ( 'fc_flowcorr', 'fc_flowcorr', 15 ), 
  'p1ecc': ( 'fc_flowcorr', None, 15 ), 
  'zero': ( None, None, 0 ), 
  'none': ( None, None, None ), 
}
#buffering = { k:DataClass(labels, v) for k, v in buffering.items() }
buffering = { k:dict(zip(labels, v)) for k, v in buffering.items() }
datatables.buffering = buffering

# Chip Info
labels = ( 'type', 'engine', 'wellsize', 'series' )
chipinfo = {
  '314': ( '314', None, 'PGM', 'pgm' ), 
  '316': ( '316', None, 'PGM', 'pgm' ), 
  '318': ( '318', None, 'PGM', 'pgm' ), 
  '510': ( '510', 1, 'P0', 'proton' ), 
  '520': ( '520', 1, 'P0', 'proton' ), 
  '521': ( '521', 1, 'P0', 'proton' ), 
  '521v2': ( '521v2', 2, 'P0', 'proton' ), 
  '530': ( '530', 1, 'P0', 'proton' ), 
  'P2.0': ( 'P0', 2, 'P0', 'proton' ), 
  '540': ( '540', 1, 'P1', 'proton' ), 
  'P2.1': ( 'P1', 2, 'P1', 'proton' ), 
  '541': ( '541', 1, 'P1', 'proton' ), 
  '541v2': ( '541', 2, 'P1', 'proton' ), 
  '550': ( '550', 2, '550', 'proton' ), 
  '560': ( '560', 2, '560', 'proton' ), 
  'P2': ( 'P2', 2, 'P2', 'proton' ), 
  'P1.2': ( 'P1.2', 1, 'P2', 'proton' ), 
  's10': ( 'P0', None, None, 'proton' ),
  's12': ( 'P1', None, None, 'proton' ),
  's22': ( 'P2', None, None, 'proton' ),
  'none': ( None, None, None, None ), 
  'dummy': ( 'unknown', None, None, 'proton'), 
}
#chipinfo = { k:DataClass(labels, v) for k, v in chipinfo.items() }
chipinfo = { k:dict(zip(labels, v)) for k, v in chipinfo.items() }
datatables.chipinfo = chipinfo

# Masks
labels = ( 'flowcell', 'gluesafe' )
masks = { 
  '314': ( 'flowcell_314', None ), 
  '316': ( 'flowcell_316', None ), 
  '318': ( 'flowcell_318', None ), 
  'P1':  ( 'p1', 'p1_gluesafe' ), 
  '521': ( '521', '521_gluesafe' ), 
  'v2':  ( 'p1_v2', 'p1_gluesafe_v2' ), 
  'valkyrie': ( 'valkyrie_rev1.T', None ), 
  '550_16': ( '550_16', 'p1_gluesafe' ), 
  '550': ( '550', 'p1_gluesafe' ), 
  #'spa': ( 'spa', None ), 
  'spa': ( None, None ), 
  'none': ( None, None ) 
}
#masks = { k:DataClass(labels, v) for k, v in masks.items() }
masks = { k:dict(zip(labels, v)) for k, v in masks.items() }
datatables.masks = masks

# ecc cal
labels = ( 'chipcal', 'dynamic_range', 'cal_timeout' )
ecc_cal = {
    '3-series': ( None, 400, None ), 
    'P1': ( 2048, 300, None ), 
    '541v2': ( 2048, 250, None ), 
    'PQ': ( 2048, 200, None ), 
    '560': ( 2048, 250, 750 ), 
    'none': ( None, None, None ), 
}
#ecc_cal = { k:DataClass(labels, v) for k, v in ecc_cal.items() }
ecc_cal = { k:dict(zip(labels, v)) for k, v in ecc_cal.items() }
datatables.ecc_cal = ecc_cal

# vfc
labels = ( 'vfc', 'vfc_profile' )
vfc = { 
  '3-series': ( False, None ), 
  'P1': ( False, 'vfc1' ), 
  '521': ( True, 'vfc1' ), 
  'Valkyrie': ( False, 'vfc4' ), 
  '550_16': ( True, 'vfc2' ), 
  '550_3490': ( False, 'vfc2' ), 
  '550_3525': ( False, 'vfc3' ), 
  'none': ( None, None ), 
  'dummy': ( False, None ), 
}
#vfc = { k:DataClass(labels, v) for k, v in vfc.items() }
vfc = { k:dict(zip(labels, v)) for k, v in vfc.items() }
datatables.vfc = vfc


# Aggregate
labels = ( 'name', 'prefered', 'chipinfo',     'dims',   'refpix',  'ecc_cal', 'buffering',    'masks',      'vfc', 'flowfront' )
chiptable = ( 
    (          '314',       True,      '314',      '314',     'none', '3-series',       '314',      '314', '3-series', '314',   ),
    (          '316',       True,      '316',      '316',     'none', '3-series',      'zero',      '316', '3-series', '316',   ),
    (          '318',       True,      '318',      '318',     'none', '3-series',       '318',      '318', '3-series', '318',   ),
    (          '510',      False,      '510',      '510',     'none',       'P1',        'P1',       'P1',       'P1', 'P0r',   ),
    (          '520',      False,      '520',      '520',     'none',       'P1',        'P1',       'P1',       'P1', 'P0r',   ),
    #(     'R1.0.20',      False,      '521',      'R20',     'none',       'P1',     'R1020',      '521',      '521', 'P0r',   ),
    (          '521',      False,      '521',      '521',     'none',       'P1',       '521',      '521',      '521', 'P0r',   ),
    (        '521v2',      False,    '521v2',    '521v2',     'none',       'P1',     '521v2',      '521',       'P1', 'P0r',   ),
    #(     'P1.0.19',      False,      '530',       'P0',     'none',       'P1',        'P1',       'P1',       'P1', 'P0',    ),
    (          '530',       True,      '530',      '530',     'none',       'P1',        'P1',       'v2',       'P1', 'P0',    ),
    (       'P2.0.1',      False,     'P2.0',       'P0',     'none',       'P1',        'P1',       'P1',       'P1', 'P0',    ),
    (          '540',       True,      '540',       'P1',     'none',       'P1',        'P1',       'v2',       'P1', 'P1',    ),
    (       'P2.1.1',      False,     'P2.1',       'P1',     'none',       'P1',        'P1',       'P1',       'P1', 'P1',    ),
    (          '541',      False,      '541',       'P1',     'none',       'P1',        'P1',       'P1',       'P1', 'P1',    ),
    ('Valkyrie_3528',      False,      '541', 'Valkyrie',     'none',       'P1',  'Valkyrie', 'valkyrie', 'Valkyrie', 'P1a',   ),
    (   'narrow_541',      False,      '541',  '541_nar',     'none',       'P1',     'p1ecc',       'P1',       'P1', 'P1',    ),
    (    '541v2_250',      False,    '541v2',       'P1',     'none',    '541v2',        'P1',       'P1',       'P1', 'P1',    ),
    (    '541v2_280',      False,    '541v2',  '541_280',     'none',    '541v2',        'P1',       'P1',       'P1', 'P1',    ),
    ( 'Val541v2_250',      False,    '541v2',       'P1',     'none',    '541v2',  'Valkyrie', 'valkyrie',       'P1', 'P1',    ),
    ( 'Val541v2_280',      False,    '541v2',  '541_280',     'none',    '541v2',  'Valkyrie', 'valkyrie',       'P1', 'P1',    ),
    (       '550_16',      False,      '550',   '550_16',   '550_16',       'PQ',        'P1',   '550_16',   '550_16', 'P1',    ),
    (     '550_3490',      False,      '550', '550_3490', '550_3490',       'PQ',        'P1',      '550', '550_3490', 'P1',    ),
    (     '550_3491',      False,      '550', '550_3491', '550_3491',       'PQ',        'P1',      '550', '550_3490', 'P1',    ),
    (     '550_3525',       True,      '550', '550_3525', '550_3525',       'PQ',        'P1',      '550', '550_3525', 'P1',    ),
    (     '560_3610',       True,      '560', '560_3610', '560_3610',      '560',        'P1',       'P1', '550_3490', 'P1',    ),
    #(      'P2.2.1',      False,       'P2',     'P221',      'P22',       'PQ',        'P1',       'P1', '550_3490', 'P221',  ),
    #(      'P2.2.2',      False,       'P2',     'P222',      'P22',       'PQ',        'P1',       'P1', '550_3490', 'PQ',    ),
    #(   'P2.2.2_os',      False,       'P2',   'P222os',      'P22',       'PQ',        'P1',       'P1', '550_3490', 'PQ',    ),
    (    'P2.2.2_16',      False,       'P2',  'P222_16',      'P22',       'PQ',        'P1',       'P1', '550_3490', 'PQ',    ),
    #(      'P2.2.2',      False,       'P2',    'P222a',      'P22',       'PQ',        'P1',       'P1', '550_3490', 'PQ',    ),
    (       'P2.2.2',       True,       'P2',    'P222b',      'P22',       'PQ',        'P1',       'P1', '550_3490', 'PQ',    ),
    (      'P1.2.18',      False,     'P1.2',      'P12',     'none',       'PQ',        'P1',       'P1',       'P1', 'P12',   ),
# Thumbnails
    (         'P1tn',       None,     'none',       'tn',     'none',     'none',      'none',     'none',     'none',  'tn', ),
    (          'spa',       None,     'none',      'spa',     'none',     'none',      'none',      'spa',     'none',  'tn', ),
# Dummy Entry
    (        'dummy',       None,    'dummy',    'dummy',     'none',     'none',      'zero',     'none',    'dummy', 'dummy', ),
# compressed superpixels
    (      's1.0.20',       None,      's10',      's10',     'none',     'none',     'super',     'none',    'dummy', 'P0',   ),
    (      's1.2.18',       None,      's12',      's12',     'none',     'none',     'super',     'none',    'dummy', 'P12',   ),
    (       's2.2.2',       None,      's22',      's22',     'none',     'none',     'super',     'none',    'dummy', 'PQ',    ),
    (         's1tn',       None,     'none',     's1tn',     'none',     'none',      'none',     'none',     'none', 'tn',  ),
)
def format_row( r, l ):
    data = dict(zip( l, r ))
    output = {}
    for k, v in data.items():
        try:
            output.update( getattr( datatables, k )[v] )
        except AttributeError:
            output[k] = v
        except KeyError:
            print( r[0], k, v )
            raise
    return output
chiptable = [ format_row( r, labels ) for r in chiptable ]
params = chiptable # for compatibility



aliases = {
    'Q1.0.20'    : 'R1.0.20',
    'P1.0.510.20': '510',
    '520v1'      : '520',
    'P1.0.520.20': '520',
    'P1.0.521.20': '521',
    '521v1'      : '521',
    '530v1'      : '530',
    'P1.0.20'    : '530',
    'P1v2'       : '540',
    'P1v3'       : '540',
    'P1.1.17'    : '540',
    '540v1'      : '540',
    'P1.1.541'   : '541',
    '541v1'      : '541',
    '541V1'      : '541',
    '541v2'      : '541v2_250',
    'GX5'        : '541v2_250',
    'GX5v2'      : '541v2_250',
    '540v2'      : '541v2_250',
    'Val541'     : 'Valkyrie_3528',
    'Val540'     : 'Valkyrie_3528',
    'P2.1.2'     : '541v2_250',
    'P2.3.1'     : '550_3525',
    'P2.3.2'     : '550_3525',
    '550v1'      : '550_3525',
    '550'        : '550_3525',
    '550v1'      : '550_3525',
    'GX7'        : '550_3525',
    'GX7v1'      : '550_3525',
    'P2.2'       : 'P2.2.2',
    'PQv1'       : 'P2.2.2',
    '560'        : '560_3610',
    'P2.2.4'     : '560_3610',
    '314R'       : '314', 
    '316D'       : '316', 
    '316E'       : '316', 
    '318B'       : '318',
    '318C'       : '318',
    '318G'       : '318',
    '---.Unknown': 'dummy', 
    'unknown'    : 'dummy', 
}
