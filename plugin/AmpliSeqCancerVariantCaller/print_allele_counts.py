#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import bisect
from collections import defaultdict


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




inf = open(sys.argv[1],'r')
out = open(sys.argv[2],'w')
out.write("Sequence Name\tPosition\tGene Name\tReference\tTotal Coverage\tA\tC\tG\tT\tN\n")

for lines in inf:
    if lines[0]=='#':
        continue
    else:
        (contig,position,ref,cov,cov_A,cov_C,cov_G,cov_T,cov_N,cov_a,cov_c,cov_g,cov_t,cov_n) = lines.split('\t')
        gene_name = findgene(contig,int(position),geneid)
        sys.stdout.write("\t\t\t\t\t\t<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (contig,position,gene_name,ref,cov,cov_A,cov_C,cov_G,cov_T,cov_N))
	out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (contig,position,gene_name,ref,cov,cov_A,cov_C,cov_G,cov_T,cov_N))

inf.close()
