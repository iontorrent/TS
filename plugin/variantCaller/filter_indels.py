#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import math
import os
import os.path
import getopt

def usage():
    global progName
    sys.stderr.write("Usage: %s [--min-bayesian-score=<float>] <input vcf> <output vcf>\n" % progName)

def main(argv):
    # arg processing
    global progName
    min_bayesian_score = 15.0
    try:
        opts, args = getopt.getopt( argv, "hm:", ["help", "min-bayesian-score="] )
    except getopt.GetoptError, msg:
        sys.stderr.write(msg)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-m", "--min-bayesian-score"):
            min_bayesian_score = float(arg)

    if(len(args) != 2):
        sys.stderr.write("Error: Invalid number of arguments\n")
        usage()
        sys.exit(1)
    invcf = args[0]
    if not os.path.exists(invcf):
        sys.stderr.write("No input vcf file found at: %s\n" % invcf)
        sys.exit(1)

    inf = open(invcf,'r')
    out = open(args[1],'w')
    for lines in inf:
        if lines[0]=='#':
            out.write(lines)
        else:
            attr={}
            fields = lines.split('\t')
            variant = 0
            info = fields[7].split(';')
            for items in info:
                key,val = items.split('=')
                attr[key]=val
            #var_freq = float(attr['Variants-freqs'].split(',')[0])*100.0
            #var_list = attr['Num-variant-reads'].split(',')
            #var_list = [int(x) for x in var_list];
            #var_cov = var_list[0]
            #ref_cov = int(attr['Num-spanning-ref-reads'])
            #total_cov = ref_cov + sum(var_list)
            #(plus,minus) = attr['Plus-minus-strand-counts'].split(',')[0].split('/')
            if len(fields[3]) == len(fields[4]):
                continue
            if float(attr['Bayesian_Score']) < float(min_bayesian_score):
                continue
            else:
                out.write(lines)
    inf.close()
    out.close()

if __name__ == '__main__':
    global progName
    progName = sys.argv[0]
    main(sys.argv[1:])

