#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# This script adds HS flag to INFO field to the variants in VCF file 
# that are in hotspots. Currently, not used in TS 2.0 or 2.2

import sys
import bisect
from collections import defaultdict
from operator import itemgetter

region_srt = defaultdict(lambda: defaultdict(defaultdict))
region_end = defaultdict(lambda: defaultdict(defaultdict))
region_ids = defaultdict(lambda: defaultdict(defaultdict))
sorted_end = defaultdict(lambda: defaultdict(defaultdict))
sorted_idx = defaultdict(lambda: defaultdict(defaultdict))

def getLociData(chrom,pos):
    """Find smallest loci containing pos and return its data tuple, or tuple of 'N/A'.
    If a tie on sizes take region where pos is closest to either end.
    If there is a tie for size and closest location, choose one with closer start pos.
    """
    if 1 == 1:
        try:
            # last loci start <= pos
            nr_srt = bisect.bisect_right(region_srt[chrom],pos)-1
            if nr_srt >= 0:
                # search back for the first loci start == pos
                if region_srt[chrom][nr_srt] == pos:
                    while nr_srt > 0:
                        if region_srt[chrom][nr_srt-1] != pos:
                            break
                        nr_srt -= 1
                    # by-pass loci end search if (the first) loci exactly matches pos
                    if region_end[chrom][nr_srt][0] == pos:
                        return 1
                # first loci end >= pos (must exist if start loci found)
                nr_end = bisect.bisect_left(sorted_end[chrom],pos)
                idx = sorted_idx[chrom][nr_end]
                # this happens for inner enclosed regions (s1,e1),(s2,e2) where s1<s2<e2<e1 and looking for s1 <= pos < s2
                if idx > nr_srt:
                    idx = nr_srt
                spos = region_srt[chrom][idx]
                epos = region_end[chrom][idx][0]
                best_idx = idx
                best_len = epos - spos
                min_dist = min( pos - spos, epos - pos )
                while idx < nr_srt:
                    idx += 1
                    spos = region_srt[chrom][idx]
                    epos = region_end[chrom][idx][0]
                    # check region in start-sorted list can contain pos
                    if epos >= pos:
                        rlen = epos - spos
                        rmin = min( pos - spos, epos - pos )
                        if (rlen < best_len) or (rlen == best_len and rmin <= min_dist):
                            best_idx = idx
                            best_len = rlen
                            min_dist = rmin
                return 1
        except:
            pass
    return 0

# assumes a BED file sorted by chrom,start,end
hotspots = open(sys.argv[1],'r')
invcf = open(sys.argv[2],'r')
outvcf = open(sys.argv[3],'w')
    
for lines in hotspots:
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
# tuple with original index used for region_end sorting
   region_srt.setdefault(chrom,[]).append(rsrt)
   region_end.setdefault(chrom,[]).append( (rend,len(region_srt[chrom])-1) )

hotspots.close()

for k in region_end.keys():
   sorted_end[k], sorted_idx[k] = zip( *sorted(region_end[k],key=itemgetter(0)) )

   
for lines in invcf:
   if lines[0]=='#':
      outvcf.write("%s"%lines)
   else:
      fields = lines.split('\t')
      flen = int(len(fields))
      if flen < 8:
         outvcf.write("%s"%lines)
      else:
          outstr = ""
          ii = 0
          for items in fields:
             outstr = outstr + items
             if ii == 7:
                if getLociData(fields[0],int(fields[1])) == 1:
                   outstr = outstr + ";HS"
             if ii != flen - 1:
                outstr = outstr + "\t"
             ii=ii+1
          outvcf.write("%s"%outstr)   

outvcf.close()
invcf.close()
