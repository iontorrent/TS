#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import fileinput
import sys
import re

print "#\tContig\tPosition\tReference\tCoverage\tA\tC\tG\tT\tD\ta\tc\tg\tt\td"
cov = {}
for p in fileinput.input():
    tmp = p.split()
    contig = tmp[0]
    chrpos = int(tmp[1])
    ref_upper = tmp[2].upper()
    ref_lower = tmp[2].lower()
    coverage = int(tmp[3])
    bases = tmp[4]
    # test for unexpected reference bases - direct matches for non-ACGT reference should not happen
    good_ref = ('ACGT'.find(ref_upper) >= 0)
    # parse the string one character at a time - unexpected codes in pileup will produce a warning
    cov['A'] = cov['C'] = cov['G'] = cov['T'] = cov['D'] = 0
    cov['a'] = cov['c'] = cov['g'] = cov['t'] = cov['d'] = 0
    slen = len(bases)
    pos = 0
    while( pos < slen ):
        base = bases[pos]
        pos += 1
        if base == '^':
            pos += 1
        elif base == '.':
            if good_ref:
                cov[ref_upper] += 1
        elif base == ',':
            if good_ref:
                cov[ref_lower] += 1
        elif 'ACGTacgt'.find(base) >= 0:
            cov[base] += 1
        elif base == '+' or base == '-':
            insLen = 0
            while( pos < slen ):
                v = '0123456789'.find(bases[pos])
                if v < 0:
                    #if base == '-': reverse_del = bases[pos].islower(), etc. - IF anchor base is in BED file
                    pos += insLen
                    break
                insLen = v + insLen * 10
                pos += 1
        elif base == '*':
            # unfortuantely +/- strand is not tracked in mpileup nor reliably determined (unless anchor included in BED)
            cov['D'] += 1
        elif '$<>'.find(base) < 0:
            sys.stderr.write("WARNING: Unexpected pileup code %s at %s:%d\n" % (base,contig,chrpos))
    # Original format of output retained, since only total coverage is defined for deletions.
    # Hence, dependent code expecting this results file format does not need to be changed if this gets fixed later.
    print "%s\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % ( contig, chrpos, ref_upper, coverage,
        cov['A']+cov['a'], cov['C']+cov['c'], cov['G']+cov['g'], cov['T']+cov['t'], cov['D']+cov['d'],
        cov['a'], cov['c'], cov['g'], cov['t'], cov['d'] )

