#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option('-i', '--input-vcf',      help='Input vcf files (separated by ",") to be merged and sorted', dest='input') 
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
        merge_and_sort(options.input.split(','), options.index, options.output)
    except:
        return 1
    return 0

def merge_and_sort(input_files_list, fai_path, output_file):

    # Step 1: Read index to establish chromosome order
    chr_order = []
    with open(fai_path, 'r') as f_i:
        for line in f_i:
            if not line:
                continue
            fields = line.split()
            chr_order.append(fields[0])
    chr_vcf_entries = dict([(chrom, []) for chrom in chr_order])
    #print 'Chromosome order: ' + ' '.join(chr_order)

    # Step 2: Read all lines in the VCF files
    header_list = []
    for index, input_file in enumerate(input_files_list):
        with open(input_file, 'r') as f_r:
            for line in f_r:
                if not line:
                    continue
                if line.startswith('#'):
                    # Use the header of the first input VCF
                    if index == 0:
                        header_list.append(line)
                    continue
                fields = line.split('\t')
                chr_vcf_entries[fields[0]].append((int(fields[1]), line))
    # Step 3: Sort all lines and write to the output file
    with open(output_file, 'w') as f_w:
        f_w.writelines(header_list)
        for chrom in chr_order:
            chr_vcf_entries[chrom].sort()
            map(lambda zip_tuple : f_w.write(zip_tuple[1]), chr_vcf_entries[chrom])
    
if __name__ == '__main__':
    sys.exit(main())
