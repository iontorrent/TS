#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
from optparse import OptionParser



def main():
    
    parser = OptionParser()
    parser.add_option('-t', '--hotspot-txt',    help='Text file with OCP hotspot definitions (default:ocp_hotspot.txt)', dest='hotspot_txt', default='merged_ocp3_hotspots.20140507.txt')
    parser.add_option('-b', '--blacklist-bed',  help='Blacklist BED (default:ocp_blist.bed)', dest='blist_bed', default='blist.rev5.bed')
    parser.add_option('-o', '--output-bed',     help='Combined BED with hotspots + blacklist (default: ocp_hotspot_blist_generated.bed)', dest='out_bed', default='ocp_hotspot_blist_generated.bed')
    parser.add_option('-z', '--targets-bed',    help='Target Regions BED file (optional)', dest='targets_bed', default='OCP3.20140508.designed.bed')
    (options, args) = parser.parse_args()


    chr_order = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
        'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20',
        'chr21','chr22','chrX','chrY','chrM']

    entries_by_chr = dict([chr,[]] for chr in chr_order)
    hotspot_pos_by_chr = dict([chr,{}] for chr in chr_order)


    #
    # Step 0. If specified, preload the targets bed file
    #
    
    targets_by_chr = None
    
    if options.targets_bed:
        targets_by_chr = dict([chr,[]] for chr in chr_order)
    
        targets_bed = open(options.targets_bed, 'r')
        line_count = 0
        for targets_bed_line in targets_bed:
            line_count += 1
            if not targets_bed_line or targets_bed_line.startswith('track'):
                continue
            fields = targets_bed_line.split('\t')
            if len(fields) < 6:
                print "%s line %d: Target regions BED file has less than 6 columns - skipping" % (options.targets_bed, line_count)
                continue
            chr = fields[0]
            pos1 = int(fields[1])
            pos2 = int(fields[2])
            name = fields[3]
            targets_by_chr[chr].append([pos1,pos2,name,0])


    #
    # Step 1: Load and process the hotspot.txt file
    #

    hs_txt = open(options.hotspot_txt, 'r')
    line_count = 0
    missing_id_counter = 0
    for hs_txt_line in hs_txt:
        line_count += 1
        if hs_txt_line.startswith('CHROM'):
            continue
        if not hs_txt_line:
            continue
        
        fields = hs_txt_line.split('\t')
        if len(fields) < 4:
            print "%s line %d: Hotspot txt file has less than 4 columns - skipping" % (options.hotspot_txt, line_count)
            continue
        
        if fields[0] in chr_order:
            chr = fields[0]
        else:
            chr = 'chr' + fields[0]
            if chr not in chr_order:
                print "%s line %d: invalid chromosome name - skipping" % (options.hotspot_txt, line_count)
                continue
        
        pos = int(fields[1]) - 1
        ref = fields[3]
        alt = fields[4]
        #if len(fields) > 8 and fields[8]:
        id = fields[2]
        #else:
        if not id:
            missing_id_counter += 1
            id = 'missing_%d_%s' % (missing_id_counter, fields[5])
            print "%s line %d: variant without ID" % (options.hotspot_txt, line_count)
        
        if ref == '-' or ref == '.':
            ref = ''
        if alt == '-' or alt == '.':
            alt = ''
        while ref and alt and ref[0] == alt[0]:
            ref = ref[1:]
            alt = alt[1:]
            pos += 1
        
        my_target = 'NONE'
        
        if targets_by_chr:
            for target in targets_by_chr[chr]:
                if pos < target[0] or pos >= target[1]:
                    continue
                my_target = target[2]
                target[3] += 1
        
        if targets_by_chr and my_target == 'NONE':
            print "%s line %d: variant does not overlap any region" % (options.hotspot_txt, line_count)
        else:
            if pos in hotspot_pos_by_chr[chr]:
                hotspot_pos_by_chr[chr][pos].append(alt)
            else:
                hotspot_pos_by_chr[chr][pos] = [alt]
        
            for actual_id in id.split(';'):
                entries_by_chr[chr].append([pos,
                                            ('%s\t%d\t%d\t%s\tREF=%s;OBS=%s\t%s' % (chr,pos,pos+len(ref),actual_id,ref,alt,my_target))])
    
    hs_txt.close()
    
    #if targets_by_chr:
    #    for chr in chr_order:
    #        for target in targets_by_chr[chr]:
    #            if target[3] == 0:
    #                print "Warning: target %s %d %d %s has no hotspots" % (chr,target[0],target[1],target[2])

    
    


    #
    # Step 2: Load blacklist file
    #

    blist_bed = open(options.blist_bed, 'r')
    line_count = 0
    for blist_bed_line in blist_bed:
        line_count += 1
        
        fields = blist_bed_line.strip().split('\t')
        if len(fields) < 5:
            print "%s line %d: Blacklist BED file must have 6 columns - skipping" % (options.blist_bed, line_count)
            continue
        
        chr = fields[0]
        if chr not in chr_order:
            print "%s line %d: invalid chromosome name - skipping" % (options.blist_bed, line_count)
            continue
        
        pos = int(fields[1])
        ref = 'N'
        alt = 'N'
        sub_info = fields[4].split(';')
        for info in sub_info:
            infox = info.split("=")
            if infox[0] == 'REF':
                ref = infox[1]
            if infox[0] == 'OBS':
                alt = infox[1]
        
        #fields[3] = 'blist'
        
        if pos in hotspot_pos_by_chr[chr]:
            while alt in hotspot_pos_by_chr[chr][pos]:
                alt += 'C'
            
            print "%s line %d: Blacklist entry %s>%s overlaps a hotspot %s" % (options.blist_bed, line_count, ref, alt, 
                                                                               ','.join(hotspot_pos_by_chr[chr][pos]))
            #print blist_bed_line
            
            for info_idx,info in enumerate(sub_info):
                if info.startswith('OBS='):
                    sub_info[info_idx] = 'OBS='+alt
            fields[4] = ';'.join(sub_info)

        my_target = 'NONE'
        if targets_by_chr:
            for target in targets_by_chr[chr]:
                if pos < target[0] or pos >= target[1]:
                    continue
                my_target = target[2]

        if len(fields) == 5:
            fields.append(my_target)

        if targets_by_chr and my_target == 'NONE':
            pass
            #    print "%s line %d: blacklist does not overlap any region" % (options.blist_bed, line_count)
        else:
            entries_by_chr[chr].append([pos, '\t'.join(fields).strip()])
        
        
    
    blist_bed.close()
    
    
    
    #
    # Step 3: Save results
    #

    for chr in chr_order:
        entries_by_chr[chr].sort()

    out_bed = open(options.out_bed, 'w')
    out_bed.write('track name="Oncomine Cancer Panel" type=bedDetail\n')
    for chr in chr_order:
        for v2 in entries_by_chr[chr]:
            out_bed.write(v2[1] + '\n')
    
    out_bed.close()



if __name__ == '__main__':
    main()

