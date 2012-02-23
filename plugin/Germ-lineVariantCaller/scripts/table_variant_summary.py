#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from optparse import OptionParser
import string
import re
import gzip
import math

def check_option(parser, value, name):
    if None == value:
        print 'Option ' + name + ' required.\n'
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-v', '--vcf-file', help='the bgzip compressed VCF file', dest='vcf_file')
    parser.add_option('-r', '--results-json-file', help='the bgzip compressed VCF file', dest='results_file', default='results.json')
    parser.add_option('-q', '--min-qual', help='the minimum quality to return', dest='min_qual', default=20)
    parser.add_option('-Q', '--max-qual', help='the maximum quality to return', dest='max_qual', default=50)
    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.print_help()
        exit(1)
    check_option(parser, options.vcf_file, '-v')
    check_option(parser, options.results_file, '-v')
    check_option(parser, options.min_qual, '-q')
    check_option(parser, options.max_qual, '-Q')

counts = [[[0 for k in range(1+int(options.max_qual/10))] for j in range(2)] for i in range(3)]
total_counts = [[0 for j in range(2)] for i in range(3)]

f = gzip.open(options.vcf_file, 'r')
for line in f:
    # process
    if not re.search('^#', line):
        # tokenize
        tokens = line.split('\t')
        # ploidy
        gt = tokens[9][0:string.find(tokens[9], ':')]
        if re.search('0', gt):
            ploidy = 0 # het
        else:
            ploidy = 1 # hom
        # variant type
        varType = 2 # SNP by default
        if re.search('^INDEL', tokens[7]):
            # pick the first one
            ref = tokens[3]
            alt = tokens[4]
            if re.search(',', alt):
                alt = alt[0:string.find(alt, ',')]
            # get the type and indel length
            if len(alt) < len(ref):
                varType = 1 # deletion
            else:
                varType = 0 # insertion
        # qual
        qual = tokens[5]
        if options.max_qual < float(qual):
            qual = options.max_qual
        qual = int(float(qual)/10)
        total_counts[varType][ploidy] += 1
        for q in range(0, qual+1):
            counts[varType][ploidy][q] += 1
f.close()

varTypeDict = {0 : 'Insertions', 1 : 'Deletions', 2 : 'SNPs'}
varTypeDictLong = {0 : 'insertions', 1 : 'deletions', 2 : 'SNPs'}
ploidyDict = {0 : 'Het', 1 : 'Hom'} 
ploidyDictLong = {0 : 'heterozygous', 1 : 'homozygous'} 

# Creaete the table
stdout_str = ''
results_str = ''
stdout_str += '\t\t\t\t\t<table class=\"noheading\">\n'
stdout_str += '\t\t\t\t\t\t<col width=40px/>\n'
stdout_str += '\t\t\t\t\t\t<col width=25px/>\n'
for qual in range(int(options.min_qual/10), len(counts[0][0])):
    stdout_str += '\t\t\t\t\t\t<col width=40px/>\n'
stdout_str += '\t\t\t\t\t\t<tr>\n'
stdout_str += '\t\t\t\t\t\t<td></td>\n'
stdout_str += '\t\t\t\t\t\t<td colspan=' + str(len(range(int(options.min_qual/10), len(counts[0][0])))+1) + '>'
stdout_str += '<span class=\'tip\' title=\'The number of variants at a minimum quality\'><span class=\'tippy\'>Number of Variants</span></span></td></tr>\n'
stdout_str += '\t\t\t\t\t\t<tr><td><span class=\'tip\' title=\'The type of variant called\'><span class=\'tippy\'>Variant Type</span></span></td>\n'
stdout_str += '\t\t\t\t\t\t<td><span class=\'tip\' title=\'Total number of variant calls (may be greater than sum of other columns because variants with less than Q20 quality are not reported in subsequent columns)\'><span class=\'tippy\'>Total calls</span></span></td>\n'
for qual in range(int(options.min_qual/10), len(counts[0][0])):
    if 0 == qual:
        stdout_str += '\t\t\t\t\t\t<td><span class=\'tip\' title=\'All variants called\'><span class=\'tippy\'>All (Q' + str(qual * 10) + '+)</span></span></td>\n'
    else:
        prob = 100.0 * (1.0 - (math.pow(10.0, -1.0 * qual)))
        stdout_str += '\t\t\t\t\t\t<td><span class=\'tip\' title=\'All variants called with quality greater than or equal to ' \
                + str(prob) + '% (Q' + str(qual * 10) + '+)\'><span class=\'tippy\'>' \
                + str(prob) + '% (Q' + str(qual * 10) + '+)</span></span></td>\n'
stdout_str += '</tr>\n'
results_str += '{\n'
commaA = 0
for varType in range(3):
    for ploidy in range(2):
        stdout_str += '\t\t\t\t\t\t<tr><td>' \
                + '<span class=\'tip\' title=\'The number of ' + ploidyDictLong[ploidy] + ' ' + varTypeDictLong[varType]  \
                + ' at a given quality threshold\'><span class=\'tippy\'>' \
                + varTypeDict[varType] + ' (' + ploidyDict[ploidy] + ')</span></span></td>'
        stdout_str += '<td>' + str(total_counts[varType][ploidy])+ '</td>'
        if 0 < commaA:
            results_str += ',\n'
        results_str += '\t"' + ploidyDictLong[ploidy] + ' ' + varTypeDictLong[varType] + '" : {\n'
        results_str += '\t\t"Q0" : "' + str(total_counts[varType][ploidy])+ '"'
        for qual in range(int(options.min_qual/10), len(counts[varType][ploidy])):
            results_str += ',\n'
            stdout_str += '<td>' + str(counts[varType][ploidy][qual]) + '</td>'
            results_str += '\t\t"Q' + str(qual * 10) + '" : "' + str(counts[varType][ploidy][qual]) + '"'
        stdout_str += '</tr>\n'
        results_str += '\n\t}'
        commaA += 1
stdout_str += '\t\t\t\t\t</table>\n'
results_str += '\n}\n'

# Write these later so that we do not fail while writing one of these
sys.stdout.write(stdout_str)
f_results = open(options.results_file, 'w')
f_results.write(results_str)
f_results.close()
