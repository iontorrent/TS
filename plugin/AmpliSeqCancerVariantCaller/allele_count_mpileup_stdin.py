#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import fileinput
import sys
import re

#out = open(sys.argv[1],'w')

print "#\tContig\tPosition\tReference\tCoverage\tA\tC\tG\tT\tN\ta\tc\tg\tt\tn"
bases=["A","C","G","T","N"]

for p in fileinput.input():
    
    cov = {}
    cov['A']=0
    cov['C']=0
    cov['G']=0
    cov['T']=0
    cov['N']=0
    cov['a']=0
    cov['c']=0
    cov['g']=0
    cov['t']=0
    cov['n']=0

    tmp = p.split()

    ref_base = tmp[2].upper()
    contig = tmp[0]

    pos = int(tmp[1])
    total_coverage = int(tmp[3])
    read_bases = tmp[4]
    # replace $ and ^ indicating begining and end of read
    read_bases = re.sub('\$|\^.','',read_bases)
    # replace < and > indicating read junctions
    read_bases = re.sub('\<|\>','',read_bases)
    # replace insertions
    struc_len=[]
    pattern=[]
    if re.search('\+',read_bases):
        for i in re.finditer('\+',read_bases):
            struc_len.append(int(i.start())+1)

        for i in struc_len:
            pattern.append('\+'+read_bases[i]+'[ACGTNacgtn]{'+read_bases[i]+'}')

        for i in pattern:
            read_bases=re.sub(i,'',read_bases)

    # replace deletions

    struc_len=[]
    pattern=[]
    if re.search('-',read_bases):
        for i in re.finditer('-',read_bases):
            struc_len.append(int(i.start())+1)

        for i in struc_len:
            pattern.append('-'+read_bases[i]+'[ACGTNacgtn]{'+read_bases[i]+'}')

        for i in pattern:
            read_bases=re.sub(i,'',read_bases)

         
         
#    read_bases = re.sub('\+[0-9]+[ACGTNacgtn]+','',read_bases)
    # replace deletions
 #   read_bases = re.sub('\-[0-9]+[ACGTNacgtn]+','',read_bases)


    if re.match('[ACGTN]',ref_base):
        cov[ref_base] = len(re.findall('\.',read_bases))
        cov[ref_base.lower()] = len(re.findall('\,',read_bases))
        for k in cov.keys():
            if not ((k == ref_base) or (k ==ref_base.lower())):
                cov[k] = len(re.findall(k,read_bases))


        print contig+"\t"+str(pos)+"\t"+ref_base+"\t"+str(total_coverage)+"\t"+str(cov['A']+cov['a'])+"\t"+str(cov['C']+cov['c'])+"\t"+str(cov['G']+cov['g'])+"\t"+str(cov['T']+cov['t'])+"\t"+str(cov['N']+cov['n'])+"\t"+str(cov['a'])+"\t"+str(cov['c'])+"\t"+str(cov['g'])+"\t"+str(cov['t'])+"\t"+str(cov['n'])
        
#        for k in bases:
#            if not (k==ref_base):
#                if ((cov[k]+cov[k.swapcase()]) >= 0.1*(cov[ref_base]+cov[ref_base.lower()])):
#                    out.write(contig+"\t"+str(pos)+"\t"+ref_base+"\t"+str(cov[ref_base]+cov[ref_base.lower()])+"\t"+str(cov[k]+cov[k.lower()])+"\n")
#out.close()
