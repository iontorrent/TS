# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
import argparse
import ConfigParser


def processParametersMerge(ppfilename, verbose):

    process_parameter_file = 'processParameters.txt'


    # Output file
    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')
    
    # Input file
    config_pp = ConfigParser.RawConfigParser()

    if verbose:
        print "Reading", ppfilename

    config_pp.read(ppfilename)
    version = config_pp.get('global','Version')
    build = config_pp.get('global','Build')
    svnrev = config_pp.get('global','SvnRev')
    runid = config_pp.get('global','RunId')
    datadirectory = config_pp.get('global','dataDirectory')
    chip = config_pp.get('global','Chip')
    floworder = config_pp.get('global','flowOrder')
    librarykey = config_pp.get('global','libraryKey')
    cyclesProcessed = config_pp.get('global','cyclesProcessed')
    framesProcessed = config_pp.get('global','framesProcessed')
    
    config_out.set('global','Version',version)
    config_out.set('global','Build',build)
    config_out.set('global','SvnRev',svnrev)
    config_out.set('global','RunId',runid)
    config_out.set('global','dataDirectory',datadirectory)
    config_out.set('global','Chip',chip)
    config_out.set('global','flowOrder',floworder)
    config_out.set('global','libraryKey',librarykey)
    config_out.set('global','cyclesProcessed',cyclesProcessed)
    config_out.set('global','framesProcessed',framesProcessed)
    
    with open(process_parameter_file, 'wb') as configfile:
        if verbose:
            print "Writing", process_parameter_file
        config_out.write(configfile)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('blockfolder', nargs=1)
    args = parser.parse_args()

    process_parameter_file = 'processParameters.txt'
    infile = os.path.join(args.blockfolder[0], process_parameter_file)
    processParametersMerge(infile, args.verbose)
