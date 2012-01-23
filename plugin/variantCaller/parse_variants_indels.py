#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Convert variant-hunter VCF file tab-separated text file, with optional annotation BED file
# Args: input.var output(.xls) [annotation.bed]

import sys
import math
import gzip
import string
import re
from collections import defaultdict
import bisect

varTypeDict = {0 : 'INS', 1 : 'DEL', 2 : 'SNP', 3 : 'MNP'}
ploidyDict = {0 : 'Het', 1 : 'Hom', 2 : 'NC'} 

region_srt = defaultdict(lambda: defaultdict(defaultdict))
region_end = defaultdict(lambda: defaultdict(defaultdict))
region_ids = defaultdict(lambda: defaultdict(defaultdict))

def findgene(chrom,start):
    if( have_bed ):
        list_pos = bisect.bisect_right(region_srt[chrom],start,0,len(region_srt[chrom]))-1
        try:
            if( (len(region_end[chrom]) > 0) and (start <= region_end[chrom][list_pos]) ):
                return region_ids[chrom][list_pos]
        except:
            pass
    return ("N/A","N/A","N/A")

# Assumes a sorted BED file, as according to spec. and assumed elsewhere
have_bed = (len(sys.argv) >= 4) and (sys.argv[3] != "")
if( have_bed ):
    genes = open(sys.argv[3],'r')
    for lines in genes:
        if len(lines) == 0:
            continue
        tmp = lines.split('\t')
        chrom = tmp[0].strip()
        if chrom.startswith('track '):
            continue
        region_srt.setdefault(chrom,[]).append( int(tmp[1])+1 )
        region_end.setdefault(chrom,[]).append( int(tmp[2]) )
        if len(tmp) < 6:
            region_ids.setdefault(chrom,[]).append( 'N/A', 0, 'N/A' )
        else:
            region_ids.setdefault(chrom,[]).append( (tmp[3].strip(),tmp[len(tmp)-2].strip(),tmp[len(tmp)-1].strip()) )
    genes.close()

inf = open(sys.argv[1],'r')
out = open(sys.argv[2],'w')
out.write("Chromosome\tPosition\tGene Symb\tTarget ID\tVarType\tPloidy\tRef\tVariant\tVarFreq\tP-value\tCoverage\tRefCov\tVarCov\n")
for lines in inf:
    if lines[0]=='#':
        continue
    else:
        attr={}
        fields = lines.split('\t')
        info = fields[7].split(';')
        # variant type
        varType = 2 # SNP by default
        ref = fields[3]
        alt = fields[4]
        if re.search(',', alt):
            alt = alt[0:string.find(alt, ',')]
        # get the type and indel length
        if len(alt) < len(ref):
            varType = 1 # deletion
        else:
            if len(alt) > len(ref):
                varType = 0 # insertion
            else:
                if len(alt) > 1:
                   varType = 3 # MNP
                else:
                   continue
        # split out the calling attributes
        for items in info:
            key,val = items.split('=')
            attr[key]=val
        # get the vairiant ploidy
        var_type = attr['Zygosity'].split(',')[0]
        if var_type == 'HETEROZYGOUS':
            ploidy = 0
        elif var_type == 'HOMOZYGOUS':
            ploidy = 1
        else:
            ploidy = 2
        # get stats for 1st variant only
        var_freq = float(attr['Variants-freqs'].split(',')[0])*100.0
        var_list = attr['Num-variant-reads'].split(',')
        var_list = [int(x) for x in var_list];
        var_cov = var_list[0]
        ref_cov = int(attr['Num-spanning-ref-reads'])
        total_cov = ref_cov + sum(var_list)
        pval = math.pow(10,(-0.1*float(attr['Bayesian_Score'])))
        (region_id,aux_id,gene_name) = findgene( fields[0], int(fields[1]) )
        out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.2e\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,region_id,varTypeDict[varType],ploidyDict[ploidy],fields[3],fields[4],var_freq,pval,total_cov,ref_cov,var_cov))

inf.close()
out.close()
