#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from optparse import OptionParser

def main():
    
    parser = OptionParser()
    parser.add_option('-i', '--input-vcf',      help='Input vcf file to be sorted', dest='input') 
    parser.add_option('-o', '--output-vcf',     help='Filename of output sorted vcf', dest='output')
    parser.add_option('-r', '--index-fai',      help='Reference genome index for chromosome order', dest='index') 
    (options, args) = parser.parse_args()

    if options.input is None:
        sys.stderr.write('[sort_vcf.py] Error: --input-vcf not specified\n')
        return 1
    if options.output is None:
        sys.stderr.write('[sort_vcf.py] Error: --output-vcf not specified\n')
        return 1
    if options.index is None:
        sys.stderr.write('[sort_vcf.py] Error: --index-fai not specified\n')
        return 1

    # Step 1: Read index to establish chromosome order
    
    chr_order = []
    chr_vcf_entries = {}
    index_file = open(options.index,'r')
    for line in index_file:
        if not line:
            continue
        fields = line.split()
        chr_order.append(fields[0])
        chr_vcf_entries[fields[0]] = []
    index_file.close()
    print 'Chromosome order: ' + ' '.join(chr_order)

    input_file = open(options.input,'r')
    output_file = open(options.output,'w')
    
    for line in input_file:
        if not line:
            continue
        if line[0] == '#':
            output_file.write(line)
            continue
        fields = line.split()
        chr_vcf_entries[fields[0]].append((int(fields[1]),line))
    
    input_file.close()
    
    for chr in chr_order:
        chr_vcf_entries[chr].sort()
        output_file.writelines([line for idx,line in chr_vcf_entries[chr]])    
    
    output_file.close()


if __name__ == '__main__':
    sys.exit(main())
