#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import bisect
from collections import defaultdict
from operator import itemgetter

region_srt = defaultdict(lambda: defaultdict(defaultdict))
region_end = defaultdict(lambda: defaultdict(defaultdict))
region_ids = defaultdict(lambda: defaultdict(defaultdict))
sorted_end = defaultdict(lambda: defaultdict(defaultdict))
sorted_idx = defaultdict(lambda: defaultdict(defaultdict))

minCoverage = 15
minFracHom = 0.8
minFracHet = 0.3

# GDM: My code here to match the raw allele output to allele table is overkill here.
# I assume a relic from early dev. that assumed the loci bed might be user defined.
# A lot is just to employ the python bisection methods for matching positions...

def getLociData(chrom,pos):
    """Find smallest loci containing pos and return its data tuple, or tuple of 'N/A'.
    If a tie on sizes take region where pos is closest to either end.
    If there is a tie for size and closest location, choose one with closer start pos.
    """
    if have_bed:
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
                        return region_ids[chrom][nr_srt]
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
                return region_ids[chrom][best_idx]
        except:
            pass
    return ("N/A","N/A","N/A")


def iupac_code(a,b):
    ab = b+a if a > b else a+b
    if( ab == "AC" ): return "M"
    if( ab == "AG" ): return "R"
    if( ab == "AT" ): return "W"
    if( ab == "CG" ): return "S"
    if( ab == "CT" ): return "Y"
    if( ab == "GT" ): return "K"
    return "?"


# output will be in the input loci order, so that missing data can be filled in
lociIDs = []
lociPos = {}

# assumes a BED file sorted by chrom,start,end
have_bed = (len(sys.argv) >= 4) and (sys.argv[3] != "")
if have_bed:
    genes = open(sys.argv[3],'r')
    for lines in genes:
        if len(lines) == 0:
            continue
        lines = lines.strip()
        tmp = lines.split('\t')
        chrom = tmp[0]
        if chrom.startswith('track '):
            continue
        # avoid 0-length BED regions (un-anchored inserts)
        rsrt = int(tmp[1])+1
        rend = int(tmp[2])
        if rsrt > rend:
            continue
        # redundancy here since this data is keyed from SNP ID vs. suited for loci matching
        hid = tmp[3].strip()
        sid = tmp[-1].strip()
        aux = tmp[-2].strip()
        lociIDs.append(sid)
        lociPos[sid] = "%s\t%d\t%s\t%s"%(chrom,rsrt,sid,hid)
        # tuple with original index used for region_end sorting
        region_srt.setdefault(chrom,[]).append(rsrt)
        region_end.setdefault(chrom,[]).append( (rend,len(region_srt[chrom])-1) )
        if len(tmp) < 6:
            region_ids.setdefault(chrom,[]).append( 'N/A', 0, 'N/A' )
        else:
            region_ids.setdefault(chrom,[]).append( (sid,aux,hid) )
    genes.close()
    for k in region_end.keys():
        sorted_end[k], sorted_idx[k] = zip( *sorted(region_end[k],key=itemgetter(0)) )


# assumes input file has forward+reverse read coverage (Uppercase fields) and reverse coverage (Lowercase fields)
inf = open(sys.argv[1],'r')
bcov = {}
major = ''
repLines = {}
for lines in inf:
    if lines[0]=='#':
        continue
    (contig,position,ref,cov,cov_A,cov_C,cov_G,cov_T,cov_D,cov_a,cov_c,cov_g,cov_t,cov_d) = lines.split('\t')
    (region_id,aux_id,hotspot_id) = getLociData( contig, int(position) )
    # forward/reverse coverage does not include coverage by deletions or adjacent inserts for 5.14
    # also as of 5.14 cov_D includes deletions plus inserts and cov_d is for the inserts alone
    cov_r = int(cov_a) + int(cov_c) + int(cov_g) + int(cov_t)
    cov_f = int(cov) - int(cov_D) - cov_r
    inss  = int(cov_d); # has extra CR
    dels  = int(cov_D) - inss
    # determine major/minor alleles for allele frequency
    bcov['A'] = int(cov_A)
    bcov['C'] = int(cov_C)
    bcov['G'] = int(cov_G)
    bcov['T'] = int(cov_T)
    bcov['D'] = dels
    scov = sorted(bcov.items(),key=itemgetter(1),reverse=True)
    major = scov[0][0]
    maj_r = scov[0][1]
    minor = scov[1][0]
    min_r = scov[1][1]
    # reported AF is always for top 2 alleles, even if one is a deletion so call is '?'
    af = 100.0 * maj_r / float(maj_r+min_r) if maj_r+min_r > 0 else 0
    # allele test denominator no longer considers ignored insert reads
    denom = float(int(cov)-inss)
    maj_f = maj_r / denom if denom > 0 else 0
    min_f = min_r / denom if denom > 0 else 0
    # make call based on simplistic rules
    call = '?'
    if denom >= minCoverage:
        if major != 'D' and maj_f+min_f >= minFracHom:
            if maj_f >= minFracHom:
                call = major
            elif minor != 'D' and min_f >= minFracHet:
                call = iupac_code(major,minor)
    # defer output
    repLines[region_id] = "%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\n" % (
        contig, position, region_id, hotspot_id, call, ref, af, cov,
        cov_A, cov_C, cov_G, cov_T, dels, inss, cov_f, cov_r )
inf.close()

# output in original SNP# order and fill in for missing mpileup data
out = open(sys.argv[2],'w')
out.write("Chrom\tPosition\tTarget ID\tTaqMan Assay ID\tCall\tRef\tAF\tCov\tA Reads\tC Reads\tG Reads\tT Reads\tDeletions\tInserts\tCov+\tCov-\n")
for sid in lociIDs:
    if sid in repLines:
        out.write(repLines[sid])
    else:
        out.write("%s\t?\t?\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n"%lociPos[sid])
out.close()

