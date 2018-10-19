#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from optparse import OptionParser
from merge_and_sort_vcf import merge_and_sort

def main():
    
    parser = OptionParser()
    parser.add_option('-i', '--input-vcf',      help='Input vcf file to be sorted', dest='input') 
    parser.add_option('-o', '--output-vcf',     help='Filename of output sorted vcf', dest='output')
    parser.add_option('-r', '--index-fai',      help='Reference genome index for chromosome order', dest='index') 
    (options, args) = parser.parse_args()

    if options.input is None:
        raise IOError('Input VCF not specified via -i or --input-vcf')
        return 1
    if options.output is None:
        raise IOError('Output VCF not specified via -o or -output-vcf')
        return 1
    if options.index is None:
        raise IOError('Reference genome index not specified via -r or --index-fai')
        return 1
    try:
        merge_and_sort([options.input, ], options.index, options.output)
    except:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
