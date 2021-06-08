import numpy as np
from numpy import float32, int32, int16, int8, dtype, uint8, uint16, iinfo, finfo
import os

verbose = 1

# DON'T FORGET TO INCREMENT THE VERION WHEN THE datprops TABLE CHANGES
version = 7
# datprops version is recorded in wettest at results.json['datprops_version'] (no version -> version 4 )
# datprops version is recorded in ecc at ecc_results.json['datprops_version'] (no version -> version 5 )

''' A summary of dat file properties. These are read by read and write functions '''
datprops = { 'actpix'                : { 'dtype' : bool,        'scale' : 1,    'a_dtype' : bool,    'a_scale' : 1,       'units' : '' },    # Pixels determined as active
             'avgtrace'              : { 'dtype' : float32,     'scale' : 1.,   'a_dtype' : float32, 'a_scale' : 1,       'units' : 'counts'  },
             'avgval'                : { 'dtype' : int16,       'scale' : 1.,   'a_dtype' : int8,    'a_scale' : 0.004,   'units' : 'counts'  },
             'bfgain'                : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.2,     'units' : 'mV/V'  },
             'bfgain_std'            : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 5.,      'units' : 'mV/V'  },
             'buffering'             : { 'dtype' : uint16,      'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : '' },    # Raw buffering values (for ECC)
             'buffering_gc'          : { 'dtype' : uint16,      'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : '' },    
             'buffering_wt'          : { 'dtype' : uint16,      'scale' : 1.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : '' },    # Raw buffering values for Wet Test
             'colnoise'              : { 'dtype' : uint16,      'scale' : 100., 'a_dtype' : uint8,   'a_scale' : 16.,     'units' : 'counts' },    
             'colnoise_raw'          : { 'dtype' : uint16,      'scale' : 10.,  'a_dtype' : uint16,  'a_scale' : 10.,     'units' : 'counts' },    
             'col_avg_coords'        : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int16,   'a_scale' : 1,       'units' : '' },    # Coordinates for column average files
             'col_avg_coords_e'      : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int16,   'a_scale' : 1,       'units' : '' },    # Coordinates for column average files
             'col_avg_coords_lstd'   : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int16,   'a_scale' : 1,       'units' : '' },    # Coordinates for column average files
             'col_avg_buffering_gc'  : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : '' },
             'col_avg_gain'          : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.2,     'units' : ''  },
             'col_avg_noise'         : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.4,     'units' : ''  },
             'col_avg_noise_localstd': { 'dtype' : float32,     'scale' : 1,    'a_dtype' : uint8,   'a_scale' : 0.4,     'units' : ''  },
             'col_avg_offset'        : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : ''  },
             'col_std_buffering_gc'  : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : ''  },
             'col_std_gain'          : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : ''  },
             'col_std_noise'         : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : ''  },
             'col_std_offset'        : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 1,       'units' : ''  },
             'diff'                  : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.5,     'units' : 'uV'    },
             'drift'                 : { 'dtype' : int16,       'scale' : 1.,   'a_dtype' : int8,    'a_scale' : 1,       'units' : 'counts'  },   
             'driftrate'             : { 'dtype' : float32,     'scale' : 1.,   'a_dtype' : float32, 'a_scale' : 1,       'units' : 'uV/sec'  },   
             'driftrate_iqr'         : { 'dtype' : float32,     'scale' : 1.,   'a_dtype' : float32, 'a_scale' : 1,       'units' : 'uV/sec'  },   
             'driftavg'              : { 'dtype' : float32,     'scale' : 1.,   'a_dtype' : float32, 'a_scale' : 1,       'units' : 'counts'  },   
             'ebfvals'               : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : int8,    'a_scale' : 0.5,     'units' : ''  },   # Empty beadfind values
             'ebfiqr'                : { 'dtype' : float32,     'scale' : 1.,   'a_dtype' : int8,    'a_scale' : 0.5,     'units' : ''  },   # Empty beadfind values
             'edge_noise_colavg'     : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : int8,    'a_scale' : -1.,     'units' : ''  },   
             'flowcorr'              : { 'dtype' : dtype('i2'), 'scale' : 1000.,'a_dtype' : int8,    'a_scale' : 250.,    'units' : ''  },   # unitless, average ~1;
             'frames'                : { 'dtype' : int8,        'scale' : 1,    'a_dtype' : int8,    'a_scale' : 1,       'units' : ''  },   # unitless, average ~1;
             'gain'                  : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.2,     'units' : 'mV/V'  },
             'gaincorr'              : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.2,     'units' : 'mV/V'  },
             'gain_corr_v5'          : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : uint8,   'a_scale' : 5.,      'units' : 'counts'  },
             'gain_iqr'              : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 5.,      'units' : 'mV/V'  },
             'gain_iqr_hd'           : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 5.,      'units' : 'mV/V'  },
             'height_corr'           : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : uint8,   'a_scale' : 5.,      'units' : 'counts'  },
             'high_res_gain_iqr'     : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 5.,      'units' : 'mV/V'  },
             'noise'                 : { 'dtype' : float32,     'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.5,     'units' : 'uV'    },
             'megabuffering'         : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : float32, 'a_scale' : 1,       'units' : '%'  },  # % standard deviation of the nuc step height
             'mx'                    : { 'dtype' : int8,        'scale' : 1,    'a_dtype' : uint8,   'a_scale' : 1,       'units' : 'counts'  },
             'nuc_height'            : { 'dtype' : int16,       'scale' : 5,    'a_dtype' : int8,    'a_scale' : 1,       'units' : ''  },       # Size of reagent step
             'gain1_offset'          : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int8,    'a_scale' : 0.5,     'units' : ''  },
             'offset'                : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int8,    'a_scale' : 0.5,     'units' : ''  },
             'offset_diff'           : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int8,    'a_scale' : 0.5,     'units' : ''  },
             'offset_noisetest'      : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : int8,    'a_scale' : 1,       'units' : 'mV' },
             'offset_colavg_dd'      : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : int8,    'a_scale' : 0.5,     'units' : ''  },
             'P1_flow_standard'      : { 'dtype' : int8,        'scale' : 1,    'a_dtype' : int8,    'a_scale' : 1,       'units' : ''  },
             'phpoint'               : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int8,    'a_scale' : 0.25,    'units' : 'counts'  }, # Plateau value from buffered pH traces
             'phslopes'              : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : int8,    'a_scale' : 2.5,     'units' : 'mV/pH'  },  # pH sensitivity
             'pinned'                : { 'dtype' : bool,        'scale' : 1,    'a_dtype' : bool,    'a_scale' : 1,       'units' : ''  },
             'raw_noise'             : { 'dtype' : float32,       'scale' : 10.,  'a_dtype' : uint8,   'a_scale' : 0.5,     'units' : 'uV'    },
             'rownoise'              : { 'dtype' : uint16,      'scale' : 100., 'a_dtype' : uint8,   'a_scale' : 16.,     'units' : 'counts'  },
             'rownoise_raw'          : { 'dtype' : uint16,      'scale' : 1.,   'a_dtype' : uint16,  'a_scale' : 10.,     'units' : 'counts'  },
             'row_avg_coords'        : { 'dtype' : int16,       'scale' : 1,    'a_dtype' : int16,   'a_scale' : 1,       'units' : ''  },
             'row_avg_noise'         : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : int8,    'a_scale' : 1,       'units' : ''  },
             'row_avg_buffering_gc'  : { 'dtype' : int16,       'scale' : 10.,  'a_dtype' : int8,    'a_scale' : 1,       'units' : ''  },
             'scales'                : { 'dtype' : int16,       'scale' : 5,    'a_dtype' : int8,    'a_scale' : 1,       'units' : ''  },       # Size of reagent step
             'slopes'                : { 'dtype' : uint16,      'scale' : -0.02,'a_dtype' : uint8,   'a_scale' : -0.0005, 'units' : 'uV/s'  },   # Slope of reagent flw.  eccv2 slopes were counts/frame   
                                                                                                                                                 # Changed scale from -1 to -0.1 on 3/21/2014.  
                                                                                                                                                 # Changed to 0.02 on 11/4/2015
             'slopes_wt'             : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : uint8,   'a_scale' : -0.0005, 'units' : 'uV/s'  },   # Wet Test slopes.  As of 6/6/2017, "slopes" was only used for ecc.  It was not used for wet test, so it can be safely renmaed
             't0'                    : { 'dtype' : int8,        'scale' : 1,    'a_dtype' : uint8,   'a_scale' : 1,       'units' : 'frames'  },
             'vary'                  : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : float32, 'a_scale' : 1,       'units' : ''  },       # col-averaged data, followed by coordinates
             'varx'                  : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : float32, 'a_scale' : 1,       'units' : ''  },       # row-averaged data, followed by coordinates
             'varxshape'             : { 'dtype' : int32,       'scale' : 1,    'a_dtype' : int32,   'a_scale' : 1,       'units' : ''  },
             'varyshape'             : { 'dtype' : int32,       'scale' : 1,    'a_dtype' : int32,   'a_scale' : 1,       'units' : ''  },
             'vs_height'             : { 'dtype' : int16,       'scale' : 5,    'a_dtype' : int8,    'a_scale' : 1,       'units' : ''  },      
             'vs_slopes'             : { 'dtype' : float32,     'scale' : 5,    'a_dtype' : float32, 'a_scale' : 1,       'units' : ''  },     
             'w2_noise'              : { 'dtype' : float32,     'scale' : 1,    'a_dtype' : float32, 'a_scale' : 1,       'units' : ''  },     
             }

# Set the limits for the data range based on the scale factor and the data type
for metric in datprops:
    # Do the standard props
    scale = datprops[metric]['scale']
    dtype = datprops[metric]['dtype']
    if dtype == bool:
        lims = [0, 1]
    else:
        try:
            minval = iinfo(dtype).min/scale
            maxval = iinfo(dtype).max/scale
            lims = sorted( [ minval, maxval ] )
        except ValueError:
            try:
                minval = finfo(dtype).min/scale
                maxval = finfo(dtype).max/scale
                lims = sorted( [ minval, maxval ] )
            except ValueError:
                lims = None
    datprops[metric]['limits'] = lims

    # Do the archived props
    scale = datprops[metric]['a_scale']
    dtype = datprops[metric]['a_dtype']
    if dtype == bool:
        lims = [0, 1]
    else:
        try:
            minval = iinfo(dtype).min/scale
            maxval = iinfo(dtype).max/scale
            lims = sorted( [ minval, maxval ] )
        except ValueError:
            try:
                minval = finfo(dtype).min/scale
                maxval = finfo(dtype).max/scale
                lims = sorted( [ minval, maxval ] )
            except ValueError:
                lims = None
    datprops[metric]['a_limits'] = lims

def get_dtype( metric, archive=False ):
    ''' Safely returns the data type from the lookup table '''
    if archive:
        typeflag = 'a_dtype'
    else:
        typeflag = 'dtype'

    try:
        dt = datprops[metric][typeflag]
    except KeyError:
        try:
            dt = datprops[os.path.basename(metric)][typeflag]
        except KeyError:
            annotate('WARNING! Unknown metric %s.  Assuming float32' % ( metric ) )
            dt = np.float32
    return dt

def get_scale( metric, archive=False ):
    ''' Safely returns the data type from the lookup table '''
    if archive:
        typeflag = 'a_scale'
    else:
        typeflag = 'scale'

    try:
        scale = datprops[metric][typeflag]
    except KeyError:
        try:
            scale = datprops[os.path.basename(metric)][typeflag]
        except KeyError:
            annotate('WARNING! Unknown metric %s [%s].  Assuming scale = 1' % ( metric, os.path.basename( metric ) ) )
            scale = 1
    return scale

def read_dat( filename, metric, dtype=None, scale=None, allow_archive=True, chiptype=None ):
    """
    Generic function to read dat files.  This rescales data but reshaping is only performed if chiptype is specified

    metric       : base name of metric to read (e.g. buffering_gc)
    dtype        : data type of the dat file.  Automatically pulled from datprops unless specified
    scale        : scaling factor for the dat file.  Automatically pulled from datprops unless specified
    allow_archive: read from .adat if .dat is missing
    chiptype     : a ChipType instance.  If this value is set, an attempt is made to reshape the data to its most probable size
    """
    # Select the correct filename and datprop table
    archivename = filename.replace('.dat','.adat') 
    if not os.path.exists( filename ) and os.path.exists( archivename ) and allow_archive:
        filename = archivename
        archive = True
    elif os.path.exists( filename ):
        archive = False
    else:
        raise IOError ('File %s is not present' % filename )

    # Select the correct data type, or use the provided one
    if dtype is None:
        dtype = get_dtype( metric, archive=archive )

    # Select the correct scale factor, or use the provided one
    if scale is None:
        scale = get_scale( metric, archive=archive )

    data = np.fromfile( filename, dtype=dtype )/scale
    if chiptype is not None:
        data = reshape( data, chiptype )

    return data

def reshape( data, chiptype, size=None ):
    """
    Reshapes the data to its most likely size based on the chiptype

    data     : a flat data array for reshaping
    chiptype : a ChipType instance
    size     : An optional tuple specifying the most likely (R,C)
    """
    # Build the list of possible sizes
    sizes = [ [ chiptype.chipR,                  chiptype.chipC ],
              [ chiptype.blockR,                 chiptype.blockC ],
              [ chiptype.chipR/chiptype.miniR,   chiptype.chipC/chiptype.miniC ],
              [ chiptype.chipR/chiptype.microR,  chiptype.chipC/chiptype.microC ], 
              [ chiptype.blockR/chiptype.miniR,  chiptype.blockC/chiptype.miniC ],
              [ chiptype.blockR/chiptype.microR, chiptype.blockC/chiptype.microC ],
              [ chiptype.blockR/chiptype.microR, chiptype.blockC/chiptype.microC ],
              [ chiptype.yBlocks,                chiptype.xBlocks ],
              [ chiptype.chipR,                  -1 ],
              [ chiptype.chipC,                  -1 ], 
              [ chiptype.blockR,                 -1 ],    # If you  know the data is going to match this, you might need to input it with the size parameter
              [ chiptype.blockC,                 -1 ] ]   # to avoid it being acidentally caught by [chipR,-1]

    # handles py3 division behavior which implicitly returns floats
    # NOTE: Needs to happen before checking if "size" is not none to pass in any size value
    sizes = [ (int(x[0]),int(x[1]),) for x in sizes ]

    # Prepend the list with the manual input size
    if size is not None:
        sizes = [size] + sizes

    # Try to reshape the array
    for possiblesize in sizes:
        try:
            data = data.reshape( possiblesize )
        except ValueError:
            continue 
        break

    return data

def write_dat( data, filename, metric, dtype=None, scale=None, archive=False):
    """
    Generic function to write dat files

    metric       : base name of metric to read (e.g. buffering_gc)
    dtype        : data type of the dat file.  Automatically pulled from datprops unless specified
    scale        : scaling factor for the dat file.  Automatically pulled from datprops unless specified
    archive      : write to .adat
    """
    archive = False
    if archive:
        samedat   = get_dtype( metric, archive=False ) == get_dtype( metric, archive=True )
        samescale = get_scale( metric, archive=False ) == get_scale( metric, archive=True )

        if not( samedat and samescale ):
            filename  = filename.replace('.dat', '.adat')
            archive = True

    if dtype is None:
        dtype = get_dtype( metric, archive=archive )

    if scale is None:
        scale = get_scale( metric, archive=archive )

    (np.array(data)*scale).astype( dtype ).tofile(filename)

def archive_dat( filename, metric ):
    ''' Reads converts the specified file to an archive file '''

    # Don't bother archiving already archived files
    if '.adat' in filename:
        return
    # This is largly to skip already archived files, since .dat files won't exist if the file has been archived already
    if not os.path.exists( filename ):
        return

    data = read_dat( filename, metric )
    write_dat( data, filename, metric, archive=True )
    os.remove( filename )

def annotate( msg, level=1 ):
    if level <= verbose:
        print( msg )
