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
out.write("Sequence Name\tPosition\tGene Name\tReference\tVariant\tVarFreq\tP-value\tTotal Coverage\tReference Coverage\tVariantCoverage\n")
		
for lines in inf:
	if lines[0]=='#':
		continue
	else:
		fields = lines.split('\t')
		info = fields[9].split(':')
		attr={}
		for i in fields[7].split(';'):
			k,v = i.split('=')
			attr[k]=v
		if (float(fields[5]) > 2500.0):
			fields[5]='2500.0'
		qual = math.pow(10,(-0.1*float(fields[5])))
		(ref_cov,alt_cov) = info[5].split(',')[0:2]
		total_cov = info[4]
		var_freq = (float(alt_cov)*100.0)/(float(total_cov))
		gene_name = findgene(fields[0],int(fields[1]),geneid)	
		if (len(fields[4].split(',')) == 1):
			sys.stdout.write("\t\t\t\t\t\t<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td>%.3g</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (fields[0],fields[1],gene_name,fields[3],fields[4],var_freq,qual,total_cov,ref_cov,alt_cov))
			out.write("%s\t%s\t%s\t%s\t%s\t%.2f\t%.3g\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,fields[3],fields[4],var_freq,qual,total_cov,ref_cov,alt_cov))
		
		else:
			sys.stdout.write("\t\t\t\t\t\t<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td>%.3g</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (fields[0],fields[1],gene_name,fields[3],fields[4].split(',')[0],var_freq,qual,total_cov,ref_cov,alt_cov))
			out.write("%s\t%s\t%s\t%s\t%s\t%.2f\t%.3g\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,fields[3],fields[4].split(',')[0],var_freq,qual,total_cov,ref_cov,alt_cov))
			alt2_cov = str(int(total_cov)-int(ref_cov)-int(alt_cov))
			var2_freq = (float(alt2_cov)*100.0)/float(total_cov)
			sys.stdout.write("\t\t\t\t\t\t<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.2f</td><td>%.3g</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (fields[0],fields[1],gene_name,fields[3],fields[4].split(',')[1],var2_freq,qual,total_cov,ref_cov,alt2_cov))
			out.write("%s\t%s\t%s\t%s\t%s\t%.2f\t%.3g\t%s\t%s\t%s\n" % (fields[0],fields[1],gene_name,fields[3],fields[4].split(',')[1],var2_freq,qual,total_cov,ref_cov,alt2_cov))


inf.close()
out.close()
