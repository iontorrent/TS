#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import fileinput
import sys
import re

# output format option - 1 is original format
summedBaseCovOutput = 1

# option to revert to former parsing - 1 is for procesing pre-allele
twoBaseAlleles = 1
ignoreInsertReads = 1

def parsePileupStr(line,option=''):
    # parse a record from a pileup to set of alleles, ignoring base reads with post inserts
    # returns position, reference and decoded alleles, with 'D' for a deletion and 'd' for any base with a post-insert.
    # - optionally '^' may be used to indicate the next base is at the start of a read
    # option is used to align adjacent reads at a base position
    noEndReads        = (option == 'NO_END_READS')
    noMarkReadStarts  = (option != 'MARK_READ_STARTS')
    tmp = line.split()
    contig = tmp[0]
    chrpos = int(tmp[1])
    ref_upper = tmp[2].upper()
    ref_lower = tmp[2].lower()
    coverage = int(tmp[3])
    bases = tmp[4]
    # test for unexpected reference bases - direct matches for non-ACGT reference should not happen
    if 'ACGT'.find(ref_upper) < 0:
        ref_upper = ref_lower = 'x'
    # parse the string one character at a time - unexpected codes in pileup will produce a warning
    pstr = ''
    slen = len(bases)
    pos = 0
    base = 'x'
    while( pos < slen ):
        last_base = base
        base = bases[pos]
        pos += 1
        if base == '^':
            pos += 1
            # NOTE: in principle read starts should only occur at the ends of pileupes
            if noMarkReadStarts: base = 'x'
        elif base == '.':
            base = ref_upper
        elif base == ',':
            base = ref_lower
        elif base == '+' or base == '-':
            # discount previous base alignment if followed by insert
            if base == '+':
                if len(pstr): pstr = pstr[:-1]
                base = 'd'
            else:
                base = 'x'
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
            base = 'D'
        elif base == '$':
            if noEndReads and len(pstr): pstr = pstr[:-1]
            base = 'x'
        elif '<>'.find(base) >= 0:
            base = 'x'
        elif 'ACGTacgt'.find(base) < 0:
            sys.stderr.write("WARNING: Unexpected pileup code %s at %s:%d\n" % (base,contig,chrpos))
        if base != 'x':
            pstr += base
    return contig,chrpos,ref_upper,coverage,pstr


def countAlleles(seq,preSeq=""):
    # return an alleles counts dictionary for aligned sequences
    # preSeq, if defined, provides the complementary pre-allele to check for pre-inserts in read alignments
    # - reads that are 'd' in either are 'discounted' to the 'd' group.
    # - it is assumed no '^' markers are left in preSeq.
    # if seq contains read start annotation '^' then the following base is not aligned to the preSeq.
    # A warning is issued if seq and preSeq reads are otherwise not aligned (at the ends).
    cov = {'A':0,'C':0,'G':0,'T':0,'D':0,'a':0,'c':0,'g':0,'t':0,'d':0}
    pos = 0
    pPos = 0
    slen = len(seq)
    plen = len(preSeq)
    base = 'x'
    checkPreSeq = (preSeq != "")
    pLenWarn = True
    while( pos < slen ):
        base = seq[pos]
        pos += 1
        if base == '^':
            checkPreSeq = False
            continue
        pbase = ""
        if checkPreSeq:
            if pPos < plen:
                pbase = preSeq[pPos]
                pPos += 1
            elif pLenWarn:
                sys.stderr.write("WARNING: Read base %s at %d does not have a match with previous base read\n" % (base,pos))
                pLenWarn = False
        else:
            checkPreSeq = (preSeq != "")
        if pbase == 'd':
            cov['d'] += 1
        else:
            cov[base] += 1
    return cov

# Original format of output retained, since only total coverage is defined for deletions.
# Hence, dependent code expecting this results file format does not need to be changed if this gets fixed later.
# NOTE: 'd' is now used to track discarded reads, as of discovery of occaassional sequencing/alignment issues for SNP#3
print "#\tContig\tPosition\tReference\tCoverage\tA\tC\tG\tT\tD\ta\tc\tg\tt\td"

if not twoBaseAlleles:
    ignoreInsertReads = 0
preBase = twoBaseAlleles
preSeq  = ''
for line in fileinput.input():
    # for base before actual SNP collect just the score discounts
    if preBase:
        preBase = 0
        if ignoreInsertReads:
            contig,chrpos,ref,coverage,preSeq = parsePileupStr(line,'NO_END_READS')
    else:
        preBase = twoBaseAlleles
        contig,chrpos,ref,coverage,seq = parsePileupStr(line,'MARK_READ_STARTS')
        cov = countAlleles(seq,preSeq)
        if summedBaseCovOutput:
            cov['A'] += cov['a']
            cov['C'] += cov['c']
            cov['G'] += cov['g']
            cov['T'] += cov['t']
            cov['D'] += cov['d']
        print "%s\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % ( contig, chrpos, ref, coverage,
            cov['A'], cov['C'], cov['G'], cov['T'], cov['D'], cov['a'], cov['c'], cov['g'], cov['t'], cov['d'] )

