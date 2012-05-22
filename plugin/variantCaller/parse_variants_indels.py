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
out.write("Chrom\tPosition\tGene Sym\tTarget ID\tType\tPloidy\tRef\tVariant\tVar Freq\tP-value\tCoverage\tRef Cov\tVar Cov\n")
for lines in inf:
    if lines[0]=='#':
        continue
    else:
        attr={}
        attr_info={}
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
        
        ref_cov = 0        
        var_cov = 0
        total_cov = 0
        
       # split out the INFO attributes
        for items in info:
            if len(items)>0:
               if "=" in items:
                  key,val = items.split('=')
                  attr[key]=val
                  
           # calculate strand-bias if info is available
                  if key == "Strand_Counts":
                     varstr = val.split(':')[1]
                     variants = varstr.split(',')
                     indel=""
                     if len(fields[3]) < len(fields[4]):
                        indel="+%s" % fields[4][1:len(fields[4])]
                     else:
                        indel="%iD" % (len(fields[3]) - len(fields[4]))
 
                     ii=0
                     ref_cov =  int(variants[ii+1]) + int(variants[ii+2])
                     for v in variants:
                        if ii % 3==0:                        
                            total_cov = total_cov + int(variants[ii+1]) + int(variants[ii+2])                        
                            if indel==v:
                               var_cov = int(variants[ii+1]) +  int(variants[ii+2])
                        ii=ii+1   
        
       # pval = math.pow(10,(-0.1*float(attr_info['Bayesian_Score'])))
        pval = math.pow(10,(-0.1*float(fields[5])))
        (region_id,aux_id,gene_name) = findgene( fields[0], int(fields[1]) )
                
        if len(fields) < 9:
           ploidy = 0
           var_freq = 0
           ref_cov = 0
           var_cov = 0
           total_cov = 0
        else:        
           info_def = fields[8].split(':')
           info_val = fields[9].split(':')                             
           # split out the GENOTYPE attributes
           ii=-1
           for items in info_def:
              ii=ii+1
              if len(items)>0:
                 attr[items]=info_val[ii]
           # get the vairiant ploidy
           var_type = attr['GT']
           if var_type == '0/1':
              ploidy = 0
           elif var_type == '1/1':
              ploidy = 1
           else:
              ploidy = 2
           # get stats for 1st variant only
           if total_cov <= 0:
              total_cov = 1
           var_freq = float(var_cov)/float(total_cov)*100.0
           if var_freq == 0:
              var_freq = float(attr['FA'].split(',')[0])*100.0
              var_cov = attr['AD'].split(',')[1]
              ref_cov = attr['AD'].split(',')[0]
              total_cov = attr['DP']
        
        out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%.2e\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,region_id,varTypeDict[varType],ploidyDict[ploidy],fields[3],fields[4],var_freq,pval,total_cov,ref_cov,var_cov))

inf.close()
out.close()
