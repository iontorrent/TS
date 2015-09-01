#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import bisect
from operator import itemgetter

def main():
    header = []
    genotypes = {}
    try:
        vcf = open(sys.argv[3], 'r')
    except:
        return
    
    for line in vcf:
        if (line.startswith("##fileformat=")): continue
        elif (line.startswith("##source=")): continue
        elif (line.startswith("##INFO=")): continue
        elif (line.startswith("##FILTER=")): continue
        elif (line.startswith("##FORMAT=")): continue
        elif (line.startswith("##")): 
            header.append(line)
            continue
        elif (line.startswith("#CHROM")): continue
        else:
            line = line.rstrip()
            cols = line.split("\t")
            key = cols[0] + ":" + cols[1]
            id = cols[2]
            ref = cols[3]
            alt = cols[4].split(',')
            filter = cols[6]
            info = cols[7].split(";")
            hotspot = ""
            if ("HS" in info): hotspot = id
            type = ""
            for element in info:
                if (element.startswith("TYPE=")):
                    type = element[5:]
                    break
            format = dict(zip(cols[8].split(':'),cols[9].split(':')))
                
            # get the vairiant ploidy
            genotype = format.get('GT','./.')
            genotype_parts = genotype.split('/')
    
            try:
                genotype1_int = int(genotype_parts[0])
                genotype2_int = int(genotype_parts[1])
        
                if genotype == '0/0':
                    ploidy = 'Ref'
                elif genotype1_int == genotype2_int:
                    ploidy = 'Hom'
                else:
                    ploidy = 'Het'
        
                alleles = [ref] + alt
                genotype_actual = alleles[genotype1_int] + '/' + alleles[genotype2_int]
        
            except:
                ploidy = 'NC'
                genotype_actual = genotype
                genotype1_int = None
                genotype2_int = None
            genotypes[key] = (genotype_actual, filter, hotspot, ref)
    vcf.close()
    header.append("##genotype is reported assuming diploid model\n")

    # assumes input file has forward+reverse read coverage (Uppercase fields) and reverse coverage (Lowercase fields)
    out = open(sys.argv[5],'w')
    for line in header:
        out.write(line)
    out.write("#BARCODE\tSAMPLE_NAME\tCHROM\tPOS\tHOTSPOT\tFILTER\tREF\tGENOTYPE\tA READS\tC READS\tG READS\tT READS\tDELETIONS\n")
    try:
        inf = open(sys.argv[4],'r')
        for lines in inf:
            if lines[0]=='#': continue
            (contig,position,ref,cov,cov_A,cov_C,cov_G,cov_T,cov_D,cov_a,cov_c,cov_g,cov_t,cov_d) = lines.split('\t')
            startPosition = int(position)
            key = contig + ":" + position
            if (key in genotypes): 
                out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (sys.argv[1],sys.argv[2],contig,position,genotypes[key][2],genotypes[key][1],genotypes[key][3],genotypes[key][0],cov_A,cov_C,cov_G,cov_T,cov_D))
        inf.close()
        out.close()
    except:
        out.close()

if __name__ == '__main__':
    main()
    