# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import argparse
import subprocess
from subprocess import CalledProcessError
import sys
import pylab
import numpy
import math
import os

import ConfigParser
import io


def merge(folder, infile, outfile, verbose, offset_str):

    infile = os.path.join(folder,infile)

    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(folder, 'processParameters.txt'))
    if offset_str == "use_blocks":
        size = config.get('global', 'Block')
    elif offset_str == "use_analysis_regions":
        size = config.get('global', 'Analysis Region')
    else:
        print "MaskMerge: ERROR: offset string not known"
        sys.exit(1)

    offset = size.split(',')
    offsetx = int(offset[0])
    offsety = int(offset[1])

    if verbose:
        print "MaskMerge: Reading "+str(infile)

    beadlist = numpy.loadtxt(infile, dtype='int', comments='#')


    # ignore block length
    WIDTH=int(beadlist[0,0])
    HEIGHT=int(beadlist[0,1])
    if verbose:
        print "MaskMerge: block size:", WIDTH, HEIGHT

    # add offset to current block data, ignore first column which contains the block size
    beadlist[1:,0]+=offsety
    beadlist[1:,1]+=offsetx

    if verbose:
        print "MaskMerge: Append block to "+str(outfile)+" with offsets x: "+str(offsetx)+" y: "+str(offsety)

    # append data
    f_handle = file(outfile, 'a')
    numpy.savetxt(f_handle, beadlist[1:,:], fmt='%1.1i')
    f_handle.close()

def main_merge(mask, blockfolder, verbose, offset_str):

    print "MaskMerge: MaskMerge started"

    INPUTFILE = mask+'.mask'
    OUTPUTFILE = mask+'.mask'
    if verbose:
        print "MaskMerge: masktype:",mask
        print "MaskMerge: in:",INPUTFILE
        print "MaskMerge: out:",OUTPUTFILE

    # add block data to outputfile
    for i,folder in enumerate(blockfolder):

        if i==0:
            if verbose:
                print "MaskMerge: write header"

            config = ConfigParser.RawConfigParser()
            config.read(os.path.join(folder, 'processParameters.txt'))
            chip = config.get('global', 'Chip')
            size = chip.split(',')
            sizex = size[0]
            sizey = size[1]

            if verbose:
                print "MaskMerge: chip size:",sizex,sizey
 
            # open to write file size
            f = open(OUTPUTFILE, 'w')
            f.write(sizex+" "+sizey+"\n")
            f.close()

        merge(folder,INPUTFILE,OUTPUTFILE,verbose,offset_str)

    if verbose:
        print "MaskMerge: write",OUTPUTFILE


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-m', '--mask', default='MaskBead', help='mask to be merged')
    parser.add_argument('-o', '--offset_str', dest='offset_str', default='use_blocks', help=' offset string')
    parser.add_argument('blockfolder', nargs='+')

    args = parser.parse_args()

    if args.verbose:
        print "MaskMerge:",args

    main_merge(args.mask, args.blockfolder, args.verbose, args.offset_str)
