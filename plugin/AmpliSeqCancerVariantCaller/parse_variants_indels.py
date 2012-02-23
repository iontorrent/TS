#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import math
import gzip
from collections import defaultdict
import bisect


def findgene(chrom,start,ids):
        pos_list = sorted(ids[chrom].keys())
        list_pos = bisect.bisect_right(pos_list,start) - 1
        if start <= ids[chrom][pos_list[list_pos]].keys():
                return ids[chrom][pos_list[list_pos]].items()[0][1]


geneid = defaultdict(lambda: defaultdict(defaultdict))

genes = open("/results/plugins/AmpliSeqCancerVariantCaller/bedfiles/400_hsm_gene_coords.txt",'r')
for lines in genes:
        tmp = lines.split()
        geneid[tmp[0]][int(tmp[1])][int(tmp[2])]=tmp[3].rstrip()

genes.close()



inf = gzip.open(sys.argv[1],'r')
out = open(sys.argv[2],'w')
		
for lines in inf:
	if lines[0]=='#':
		continue
	else:
                attr={}
		fields = lines.split('\t')
		info = fields[7].split(';')
                for items in info[:-1]:
                    key,val = items.split('=')
                    attr[key]=val
                # place holder
                var_freq = float(attr['Variants-freqs'].split(',')[0])*100.0
                total_cov = attr['Num-spanning-reads']
		var_list = attr['Num-variant-reads'].split(',')
		var_list = [int(x) for x in var_list];
		ref_cov = int(total_cov) - sum(var_list)
                var_cov = attr['Num-variant-reads'].split(',')[0]
		pval = math.pow(10,(-0.1*float(attr['Score'])))
                gene_name = findgene(fields[0],int(fields[1]),geneid)
		sys.stdout.write("\t\t\t\t\t\t<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td>%.3g</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (fields[0],fields[1],gene_name,fields[3],fields[4],var_freq,pval,total_cov,ref_cov,var_cov))
		out.write("%s\t%s\t%s\t%s\t%s\t%.2f\t%.3g\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,fields[3],fields[4],var_freq,pval,total_cov,ref_cov,var_cov))
inf.close()
out.close()
