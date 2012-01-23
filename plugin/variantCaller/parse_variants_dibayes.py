#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Convert diBayes VCF file tab-separated text file, with optional annotation BED file
# Args: input.var output(.xls) [annotation.bed]

import sys
import string
import math
import re
from collections import defaultdict
import bisect

varTypeDict = {0 : 'INS', 1 : 'DEL', 2 : 'SNP' , 3: 'MNP'}
ploidyDict = {0 : 'Het', 1 : 'Hom', 2 : 'NC' } 

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
        fields = lines.split('\t')
        info = fields[9].split(':')
        attr={}
        # ploidy
        gt = fields[9][0:string.find(fields[9], ':')]
        if re.search('0', gt):
            ploidy = 0 # het
        else:
            ploidy = 1 # hom
        # variant type
        varType = 2 # SNP by default
        if re.search('^INDEL', fields[7]):
                              # pick the first one
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
        for i in fields[7].split(';'):
            k,v = i.split('=')
            attr[k]=v
        if (float(fields[5]) > 2500.0):
            fields[5]='2500.0'
        qual = math.pow(10,(-0.1*float(fields[5])))
        (ref_cov,alt_cov) = info[5].split(',')[0:2]
        total_cov = info[4]
        var_freq = (float(alt_cov)*100.0)/(float(total_cov))
        (region_id,aux_id,gene_name) = findgene( fields[0], int(fields[1]) )    
        if (len(fields[4].split(',')) == 1):
            out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.2e\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,region_id,varTypeDict[varType],ploidyDict[ploidy],fields[3],fields[4],var_freq,qual,total_cov,ref_cov,alt_cov))
        
        else:
            out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.2e\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,region_id,varTypeDict[varType],ploidyDict[ploidy],fields[3],fields[4].split(',')[0],var_freq,qual,total_cov,ref_cov,alt_cov))
            alt2_cov = str(int(total_cov)-int(ref_cov)-int(alt_cov))
            var2_freq = (float(alt2_cov)*100.0)/float(total_cov)
            out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.2e\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,region_id,varTypeDict[varType],ploidyDict[ploidy],fields[3],fields[4].split(',')[1],var2_freq,qual,total_cov,ref_cov,alt2_cov))

inf.close()
out.close()
