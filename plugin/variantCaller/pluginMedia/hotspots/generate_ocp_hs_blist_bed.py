#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import re
import collections
from argparse import ArgumentParser

import pprint
pp = pprint.PrettyPrinter(indent=2)

def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def main():

    parser = ArgumentParser()
    hotspot_arg = parser.add_mutually_exclusive_group(required=True)
    hotspot_arg.add_argument('-t', '--hotspot-txt',  help='Text file with OCP hotspot definitions', dest='hotspot_txt')
    hotspot_arg.add_argument('-s', '--hotspot-bed',  help='BED file with OCP hotspot definitions (new alternative to --hotspot-txt)', dest='hotspot_bed')
    parser.add_argument('-b', '--blacklist-bed',     help='Blacklist BED. If it contains AMPL_LEFT positions inside amplicon overlaps close to the ends of both amplicons, such AMPL_LEFT positions are not merged into the output. Instead, they are translated to TRIM_LEFT directives in a new target regions BED.', dest='blist_bed')
    parser.add_argument('-o', '--output-bed',        help='Combined BED with hotspots + blacklist (default: ocp_hotspot_blist_generated.bed)', dest='out_bed', default='ocp_hotspot_blist_generated.bed')
    parser.add_argument('-z', '--targets-bed',       help='Target Regions BED file. A new target BED file with TRIM_LEFT and TRIM_RIGHT directives will be created to take into account the edge errors discovered by the blacklist generator.', dest='targets_bed')
    options = parser.parse_args()


    chr_order = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
        'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20',
        'chr21','chr22','chrX','chrY','chrM']

    entries_by_chr = dict([chr,[]] for chr in chr_order)
    blacklist_entries_by_chr = dict([chr,[]] for chr in chr_order)
    hotspot_pos_by_chr = dict([chr,{}] for chr in chr_order)


    #
    # Step 0. If specified, preload the targets bed file
    #

    targets_by_chr = None
    targets_header = []

    if options.targets_bed:
        targets_by_chr = collections.OrderedDict([chr,[]] for chr in chr_order)

        targets_bed = open(options.targets_bed, 'r')
        line_count = 0
        for targets_bed_line in targets_bed:
            line_count += 1
            if not targets_bed_line or targets_bed_line.startswith('track'):
                targets_header.append(targets_bed_line)
                continue
            fields = targets_bed_line.rstrip().split('\t')
            if len(fields) < 6:
                print "%s line %d: Target regions BED file has less than 6 columns - skipping" % (options.targets_bed, line_count)
                continue
            chr = fields[0]
            pos1 = int(fields[1])
            pos2 = int(fields[2])
            name = fields[3]
            x = fields[4]
            info = fields[5]
            targets_by_chr[chr].append([pos1, pos2, name, x, info, 0, 0]) # First 0 to count hotspots per target
                                                                          # (not used, commented out below);
                                                                          # the second 0 is a placeholder for the TRIM_LEFT size
                                                                          #
                                                                          #
                                                                          # AMPL1: ---------------+++++----  <- overhang = 4
                                                                          # AMPL2:                LLLLL--------------------
                                                                          #                     ->    <- TRIM_LEFT = 5 (Ɐ overhang >= 2)
                                                                          #           (L = AMPL_LEFT position)

    #
    # Step 1: Load and process the hotspot.txt file
    #

    if options.hotspot_bed:
        hs_bed = open(options.hotspot_bed, 'r')
        line_count = 0
        missing_id_counter = 0
        for hs_bed_line in hs_bed:
            line_count += 1
            if not hs_bed_line or hs_bed_line.startswith('track'):
                continue
            if not hs_bed_line:
                continue

            fields = hs_bed_line.split('\t')
            if len(fields) < 5:
                print "%s line %d: Hotspot txt file has less than 4 columns - skipping" % (options.hotspot_txt, line_count)
                continue

            if fields[0] in chr_order:
                chr = fields[0]
            else:
                chr = 'chr' + fields[0]
                if chr not in chr_order:
                    print "%s line %d: invalid chromosome name - skipping" % (options.hotspot_txt, line_count)
                    continue

            # pos = int(fields[1]) - 1
            pos = int(fields[1]) # BED positions are 0-based
            endpos = int(fields[2])
            id = fields[3]
            var = fields[4]
            ref = 'N'
            alt = 'N'
            for expr in fields[4].split(';'):
                key, val = expr.partition("=")[::2]
                if key == 'REF':
                    ref = val
                if key == 'OBS':
                    alt = val

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
                    target[5] += 1

            if targets_by_chr and my_target == 'NONE':
                print "%s line %d: variant does not overlap any region" % (options.hotspot_txt, line_count)
            else:
                if pos in hotspot_pos_by_chr[chr]:
                    hotspot_pos_by_chr[chr][pos].append(alt)
                else:
                    hotspot_pos_by_chr[chr][pos] = [alt]

                # from hotspots.txt
                entries_by_chr[chr].append([pos, ('%s\t%d\t%d\t%s\tREF=%s;OBS=%s\t%s' % (chr, pos, endpos, id, ref, alt, my_target))])

        hs_bed.close()

    else: # hotspots.bed
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
                    target[5] += 1

            if targets_by_chr and my_target == 'NONE':
                print "%s line %d: variant does not overlap any region" % (options.hotspot_txt, line_count)
            else:
                if pos in hotspot_pos_by_chr[chr]:
                    hotspot_pos_by_chr[chr][pos].append(alt)
                else:
                    hotspot_pos_by_chr[chr][pos] = [alt]

                for actual_id in id.split(';'):

                    # from hotspots.bed
                    entries_by_chr[chr].append([pos, ('%s\t%d\t%d\t%s\tREF=%s;OBS=%s\t%s' % (chr,pos,pos+len(ref),actual_id,ref,alt,my_target))])

        hs_txt.close()

    #if targets_by_chr:
    #    for chr in chr_order:
    #        for target in targets_by_chr[chr]:
    #            if target[5] == 0:
    #                print "Warning: target %s %d %d %s has no hotspots" % (chr,target[0],target[1],target[2])



    #
    # Step 2: Load blacklist file
    #

    edge_blacklist = dict([chr, {}] for chr in chr_order)
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

        if pos in hotspot_pos_by_chr[chr]:
            #while alt in hotspot_pos_by_chr[chr][pos]:
            #    alt += 'C'

            print "%s:%s in %s line %d: Blacklist entry %s>%s overlaps a hotspot '%s'" % (chr, pos, options.blist_bed, line_count, ref, alt,
                                                                               ','.join(hotspot_pos_by_chr[chr][pos]))
            #print blist_bed_line

            for info_idx,info in enumerate(sub_info):
                if info.startswith('OBS='):
                    sub_info[info_idx] = 'OBS='+alt
            fields[4] = ';'.join(sub_info)

        my_target_name = 'NONE'
        my_target_index = None
        if targets_by_chr:
            for i, target in enumerate(targets_by_chr[chr]):
                if pos < target[0] or pos >= target[1]:
                    continue
                my_target_name = target[2]
                my_target_index = i

        if len(fields) == 5:
            fields.append(my_target_name)
        if len(fields) == 6 and fields[5] == '.':
            fields[5] = my_target_name

        if targets_by_chr and my_target_name == 'NONE':
            print "%s:%s in %s line %d: blacklisted position is not contained in any region" % (chr, pos, options.blist_bed, line_count)
        else:
            for tag in ['AMPL_LEFT', 'AMPL_RIGHT']:
                if fields[3].startswith(tag):
                    # Count <tag> occurrences in each position (in case there is ever more than one)
                    if pos not in edge_blacklist[chr]:
                        edge_blacklist[chr][pos] = {tag: 1}
                    else:
                        edge_blacklist[chr][pos][tag] += 1

            blacklist_entries_by_chr[chr].append([pos, '\t'.join(fields).strip()])

    blist_bed.close()

    # Find all amplicon overlaps and mark AMPL_LEFT positions in them
    for chr in chr_order:
        for index, target in enumerate(targets_by_chr[chr]):
            # Clean up any TRIM_* directives that may be there
            target[4] = re.sub(r'(;?TRIM_(LEFT|RIGHT)=\d+)+', '', target[4]);
            s1 = target[0]
            e1 = target[1]
            a1 = target[2]
            for index2 in range(index + 1, len(targets_by_chr[chr])):
                target2 = targets_by_chr[chr][index2]
                s2 = target2[0]
                e2 = target2[1]
                a2 = target2[2]
                if overlap([s1, e1], [s2, e2]):
                    blacklisted = False
                    rightmost_blacklisted = s2
                    #for pos in range(s2, e1 + 1):
                    for pos in range(s2, e1): # potential bug!
                        if pos in edge_blacklist[chr]:
                            if 'AMPL_LEFT' in edge_blacklist[chr][pos]:
                                blacklisted = True
                                rightmost_blacklisted = pos

                    overhang = e1 - 1 - rightmost_blacklisted
                    if blacklisted and overhang >= 2:
                        target2[6] = rightmost_blacklisted - s2 + 1 # Set TRIM_LEFT
                        # Rescan edge_blacklist and unset AMPL_LEFT
                        for pos in range(s2, e1 + 1):
                            if pos in edge_blacklist[chr]:
                                if 'AMPL_LEFT' in edge_blacklist[chr][pos]:
                                    edge_blacklist[chr][pos].pop('AMPL_LEFT')

                    sys.stdout.write((' ' * (6 - len(chr))) + chr + ':' + (' ' * (15 - len(a1))) + a1 + '  ' + (' ' * (10 - len(str(s1)))) + str(s1) + ' -----------------')
                    for pos in range (s2, e1 + 1):
                        char = '-'
                        if pos <= rightmost_blacklisted:
                            char = '+'
                        if pos in edge_blacklist[chr] and 'AMPL_RIGHT' in edge_blacklist[chr][pos]:
                            char = 'R';
                        sys.stdout.write(char)
                    print ' %d  overhang: %d' % (e1, overhang)
                    sys.stdout.write( (' ' * (22 - len(a2))) + a2 + (' ' * (29 - len(str(s2)))) + str(s2) + ' ')
                    for pos in range (s2, e1 + 1):
                        char = '-'
                        if pos in edge_blacklist[chr] and 'AMPL_RIGHT' in edge_blacklist[chr][pos]:
                            char = 'R';
                        if pos in edge_blacklist[chr] and 'AMPL_LEFT' in edge_blacklist[chr][pos]:
                            char = 'L';
                        if pos in edge_blacklist[chr] and 'AMPL_LEFT' in edge_blacklist[chr][pos] and 'AMPL_RIGHT' in edge_blacklist[chr][pos]:
                            char = '*';
                        sys.stdout.write(char)

                    print '---------------------------------- %d\n' % e2

    # Create a trimmed regions BED using the TRIM_LEFT flag set above (target[6)
    targets_bed_trimmed = open(re.sub(r'\.bed$', "-trimmed.bed", options.targets_bed), 'w')
    for line in targets_header:
        targets_bed_trimmed.write(line)

    for chr in chr_order:
        for target in targets_by_chr[chr]:
            if target[6]:
                target[4] = target[4] + ';TRIM_LEFT=%d' % target[6]
            targets_bed_trimmed.write('\t'.join([chr] + [str(x) for x in target[:-2]]) + '\n')

    targets_bed_trimmed.close()

    #
    # Step 3: Clean up the blacklist
    #
    for chr in chr_order:
        for entry in blacklist_entries_by_chr[chr]:
            pos = entry[0]
            type = entry[1].split('\t')[3].split('.')[0]
            if type == 'AMPL_LEFT':
                if pos in edge_blacklist[chr] and ('AMPL_LEFT' in edge_blacklist[chr][pos]):
                    entries_by_chr[chr].append(entry)
            elif type.startswith('SSE') or type == 'LHP':
                if not (pos in edge_blacklist[chr] and ('AMPL_LEFT' in edge_blacklist[chr][pos] or 'AMPL_RIGHT' in edge_blacklist[chr][pos])):
                    entries_by_chr[chr].append(entry)
            else:
                entries_by_chr[chr].append(entry)

    #
    # Step 4: Save results
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

