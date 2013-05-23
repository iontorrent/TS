#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import bisect
from collections import defaultdict
from operator import itemgetter

region_ids = defaultdict(lambda: defaultdict(defaultdict))
leftAlignedHotspotStarts = defaultdict(lambda: defaultdict(defaultdict))

def getLociDataNew(chrom, pos):
	if have_bed:
		ids = region_ids[chrom][pos]
		if ids != None and len(ids) > 0:
			return ids
		else:
    			return ("N/A","N/A","N/A")

# assumes a BED file sorted by chrom,start,end
have_bed = (len(sys.argv) >= 5) and (sys.argv[4] != "") and (sys.argv[3] != "")

if have_bed:
	leftAlignedGenes = open(sys.argv[3], 'r')
	for lines in leftAlignedGenes:
		if len(lines) == 0:
			continue
		tmp = lines.split('\t')
		chrom = tmp[0].strip()
        	if chrom.startswith('track '):
            		continue
        	# avoid 0-length BED regions (un-anchored inserts)
        	rsrt = int(tmp[1])+1
        	rend = int(tmp[2])
        	if rsrt > rend:
            		continue
		hotspotId = tmp[3].strip()
		leftAlignedHotspotStarts[hotspotId] = rsrt
	leftAlignedGenes.close()
		
if have_bed:
    genes = open(sys.argv[4],'r')
    for lines in genes:
        if len(lines) == 0:
            continue
        tmp = lines.split('\t')
        chrom = tmp[0].strip()
        if chrom.startswith('track '):
            continue
        # avoid 0-length BED regions (un-anchored inserts)
        rsrt = int(tmp[1])+1
        rend = int(tmp[2])
	hotspotId = 'N/A'
	if len(tmp) > 3 :
		hotspotId = tmp[3].strip()
		if(leftAlignedHotspotStarts[hotspotId] != rsrt):
			hotspotId = hotspotId + "*";
        if rsrt > rend:
            continue
        while (rsrt <= rend):
		if len(tmp) < 6:
            		region_ids[chrom].setdefault(rsrt,[]).append( ('N/A', 0, 'N/A'))
        	else:
            		region_ids[chrom].setdefault(rsrt,[]).append((tmp[len(tmp)-1].strip(),tmp[len(tmp)-2].strip(),hotspotId) )
		rsrt += 1
    genes.close()

# assumes input file has forward+reverse read coverage (Uppercase fields) and reverse coverage (Lowercase fields)
inf = open(sys.argv[1],'r')
out = open(sys.argv[2],'w')
out.write("Chrom\tPosition\tTarget ID\tHotSpot ID\tRef\tCov\tA Reads\tC Reads\tG Reads\tT Reads\t+Cov\t-Cov\tDeletions\n")
for lines in inf:
    if lines[0]=='#':
        continue
    else:
	regIds = []
	hotspotIds = []
        (contig,position,ref,cov,cov_A,cov_C,cov_G,cov_T,cov_D,cov_a,cov_c,cov_g,cov_t,cov_d) = lines.split('\t')
	startPosition = int(position)
	ids = getLociDataNew( contig, startPosition )
	for item in ids:
		if(not item[0] in regIds):
			regIds.append(item[0])
		if(not item[2] in hotspotIds):
			hotspotIds.append(item[2])
	#(region_id,aux_id,hotspot_id) = getLociData( contig, int(position) )
        # forward/reverse coverage does not include coverage by deletions
        cov_r = int(cov_a) + int(cov_c) + int(cov_g) + int(cov_t)
        cov_f = int(cov) - int(cov_D) - cov_r
        
	out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\n" % (contig,position, ";".join(regIds), ";".join(hotspotIds),ref,cov,cov_A,cov_C,cov_G,cov_T,cov_f,cov_r,cov_D))
out.close()
inf.close()
