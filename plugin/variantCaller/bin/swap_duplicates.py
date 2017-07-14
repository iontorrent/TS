#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (C) 2017 Thermo Fisher Scientific Inc. All Rights Reserved
#
# this is a partial fix for cases where a hotspot is called REF or NOCALL yet
# the variant is actually present and called as de-novo in a separate vcf line.
# Affects repeat sequence indels where tmap may right align and
# the variant is correctly called as a de-novo but the hotspot is
# left aligned too far away to be discovered as the same variant and appears
# to be reference.  Limitations of this fix include:
# (1) the de-novo will be incorrectly reported as REF or NOCALL
# (2) the de-novo filter thresholds are applied to the hotspot
# All vcf lines affected have an extra INFO tag: UFR, see TVC 5.4 release notes
# Affects any version where this script is called in variant_caller_pipeline.py

import sys
import TvcVcfFile as vcf
import os
from Bio import SeqIO

def make_regions_ref(bed_file, genome):

    line_count = 0

    regions = {}
    for bed_line in open(bed_file, 'r'):
        line_count += 1
        if not bed_line or bed_line.startswith('track'):
            continue
        if not bed_line:
            continue

        fields = bed_line.split('\t')
        if len(fields) < 3:
            print "%s line %d: bed file has less than 3 columns - skipping" % bed_file, line_count
            continue

        contig = fields[0]
        pos = int(fields[1]) # BED positions are 0-based
        endpos = int(fields[2])
        if len(fields) == 3:
            idd = str(line_count)
        else:
            idd = fields[3]

        ref = genome[contig].seq[pos:endpos]
        regions[idd] = [contig, pos, str(ref)]

    return (regions)


def find_region(regions, contig, pos):
    for key, value in regions.items():
        if contig == value[0] and pos >= value[1] and pos < value[1] + len(value[2]):
            return key, value[1], value[2]
    return None, 0, 0

# find all lines with novel allele with genotype(s)
def find_novels(vcf_record, regions):
    ref = vcf_record['REF']
    alts = vcf_record['ALT']
    contig = vcf_record['CHROM']
    pos = vcf_record['POS'] - 1
    endpos = pos+len(ref)
    genotype=vcf_record['FORMAT'][0]['GT'].split('/')

    while '0' in genotype: genotype.remove('0')
    while '.' in genotype: genotype.remove('.')
    if not genotype:
        return []
    genotype = map(lambda x: x-1, map(int, genotype))
            
    region_id, region_start, region_seq = find_region(regions, contig, pos)

    omapalts = vcf_record['INFO']['OMAPALT']
    oids = vcf_record['INFO']['OID']

    left_pad = region_seq[0:pos-region_start]
    right_pad = region_seq[len(ref)+pos-region_start:]

    ret = []
    for gt, alt in enumerate([alts[i] for i in genotype]):
        for ii, allele in enumerate(omapalts):
            if allele == alt and oids[ii] == '.':
                alt_pad = str(left_pad + alt + right_pad)
                if left_pad + ref + right_pad == region_seq:
                    ret.append([alt_pad, ii])
    return ret



# find all lines with HS tag
def find_uncalled_hotspots(vcf_record, regions):
        
    if not 'HS' in vcf_record['INFO']:
        return []

    ref = vcf_record['REF']
    alts = vcf_record['ALT']
    contig = vcf_record['CHROM']
    pos = vcf_record['POS'] - 1
    endpos = pos+len(ref)
    genotype = vcf_record['FORMAT'][0]['GT'].split('/')

    while '0' in genotype: genotype.remove('0')
    while '.' in genotype: genotype.remove('.')
    if genotype:
        genotype = map(lambda x: x-1, map(int, genotype))
          
    region_id, region_start, region_seq = find_region(regions, contig, pos)

    omapalts = vcf_record['INFO']['OMAPALT']
    oids = vcf_record['INFO']['OID']
            
    # Find labeled OIDs not in the genotype
    ret = []
    for ii, alt in enumerate(alts):
        if genotype and ii+1 in genotype:
            continue
        ix = [i for i, allele in enumerate(omapalts) if allele == alt]
        for i in ix:
            if not oids[i] == '.':
                left_pad = region_seq[0:pos-region_start]
                right_pad = region_seq[len(ref)+pos-region_start:]
                alt_pad = str(left_pad + alt + right_pad)
                if left_pad + ref + right_pad == region_seq:
                    ret.append([alt_pad, i])
    return ret


# find all lines with HS tag
def find_called_hotspots(vcf_record, regions):
        
    if not 'HS' in vcf_record['INFO']:
        return []

    ref = vcf_record['REF']
    alts = vcf_record['ALT']
    contig = vcf_record['CHROM']
    pos = vcf_record['POS'] - 1
    endpos = pos+len(ref)
    genotype = vcf_record['FORMAT'][0]['GT'].split('/')

    while '0' in genotype: genotype.remove('0')
    while '.' in genotype: genotype.remove('.')
    if genotype:
        genotype = map(lambda x: x-1, map(int, genotype))
    else:
        return []
          
    region_id, region_start, region_seq = find_region(regions, contig, pos)

    omapalts = vcf_record['INFO']['OMAPALT']
    oids = vcf_record['INFO']['OID']
            
    # find labeled OIDs in the genotype
    ret = []
    for ii, alt in enumerate(alts):
        if ii not in genotype:
            continue
        ix = [i for i, allele in enumerate(omapalts) if allele == alt]
        for i in ix:
            for ii, alt in enumerate(alts):
                if genotype and ii in genotype:
                    if omapalts[i] == alt and not oids[i] == '.':
                        left_pad = region_seq[0:pos-region_start]
                        right_pad = region_seq[len(ref)+pos-region_start:]
                        alt_pad = str(left_pad + alt + right_pad)
                        if left_pad + ref + right_pad == region_seq:
                            ret.append([alt_pad, i])
    return ret
    
# in-place update of a novel with a hotspot allele
def swap_ids(alt_pad, novel_entry, hotspot_entry):

    hotspot_record = hotspot_entry[0]
    i_hs = hotspot_entry[1]

    novel_record = novel_entry[0]
    i_n = novel_entry[1]

    novel_oid = novel_record['INFO']['OID'][i_n]
    novel_oalt = novel_record['INFO']['OALT'][i_n]
    novel_opos = novel_record['INFO']['OPOS'][i_n]
    novel_oref = novel_record['INFO']['OREF'][i_n]

    hs_oid =  hotspot_record['INFO']['OID'][i_hs]

    # leave OMAPALT alone as it is the key into rest of the record

    # update novel_record from hotspot_record
    novel_record['INFO']['OID'][i_n] = hotspot_record['INFO']['OID'][i_hs]
    novel_record['INFO']['OALT'][i_n] = hotspot_record['INFO']['OALT'][i_hs]
    novel_record['INFO']['OPOS'][i_n] = hotspot_record['INFO']['OPOS'][i_hs]
    novel_record['INFO']['OREF'][i_n] = hotspot_record['INFO']['OREF'][i_hs]

    # update hotspot_record from saved novel_record values
    hotspot_record['INFO']['OID'][i_hs] = novel_oid
    hotspot_record['INFO']['OALT'][i_hs] = novel_oalt
    hotspot_record['INFO']['OPOS'][i_hs] = novel_opos
    hotspot_record['INFO']['OREF'][i_hs] = novel_oref

    # Fix hotspot_record ID field and INFO HS key
    hs_id =  hotspot_record['ID'].split(';')
    hs_id.remove(hs_oid)
    hotspot_record['ID'] = ';'.join(hs_id)
    if not hotspot_record['ID']:
        hotspot_record['ID'] = '.'
        hotspot_record['INFO'].pop('HS', None)

    # Fix novel_record ID field
    novel_id = novel_record['ID']
    if novel_record['ID'] == '.':
        novel_record['ID'] = ''
    else:
        novel_record['ID'] += ';'
    novel_record['ID'] += hs_oid

    ii_n = novel_record['ALT'].index(novel_record['INFO']['OMAPALT'][i_n])
    ii_hs = hotspot_record['ALT'].index(hotspot_record['INFO']['OMAPALT'][i_hs])

    # Add flag to hotspot_record
    hotspot_record['INFO']['UFR'] = ['.']*len(hotspot_record['ALT'])
    hotspot_record['INFO']['UFR'][ii_hs] = 'unknown'

    # Add flag to novel_record
    # filter_type_hotspot = hotspot_record['INFO']['TYPE'][ii_hs]
    novel_record['INFO']['UFR'] = ['.']*len(novel_record['ALT'])
    novel_record['INFO']['UFR'][ii_n] = 'hs'

    # Add HS to INFO
    novel_record['INFO']['HS'] = None


def fix_records(vcf_file, bed_file, out_vcf_file):
    novels = {}
    uncalled_hotspots = {}
    called_hotspots = {}
    other_vcf = []

    with vcf.TvcVcfFile(vcf_file, 'r') as f_vcf:
        f_vcf.set_bypass_size_check_tags(['FR',])

        ref = [x for x in f_vcf.vcf_double_hash_header if x.find('##reference=') == 0]
        if len(ref) == 1:
            genome_file = ref[0][len('##reference='):]
        else:
            print 'reference not found in this vcf: ' + vcf_file
            raise ValueError('reference not found in this vcf: ' + vcf_file)

        #genome =  SeqIO.to_dict(SeqIO.parse(open(sys.argv[3]), 'fasta'))
        genome =  SeqIO.to_dict(SeqIO.parse(open(genome_file), 'fasta'))

        regions = make_regions_ref(bed_file, genome)

        for vcf_record in f_vcf:

            # bug if multiple novels map to the same allele, but we will ignore this

            if not 'OID' in vcf_record['INFO']:
                other_vcf.append(vcf_record)
                continue

            for v in find_novels(vcf_record, regions):
                alt_pad = v[0]
                i = v[1]
                if alt_pad:
                    if alt_pad in novels:
                        novels[alt_pad].append([vcf_record, i])
                    else:
                        novels[alt_pad] = [[vcf_record, i]]

            # multiple hotspots may map to the same allele
            for v in find_uncalled_hotspots(vcf_record, regions):
                alt_pad = v[0]
                i = v[1]
                if alt_pad:
                    if alt_pad in uncalled_hotspots:
                        uncalled_hotspots[alt_pad].append([vcf_record, i])
                    else:
                        uncalled_hotspots[alt_pad] = [[vcf_record, i]]

            # multiple hotspots may map to the same allele
            for v in find_called_hotspots(vcf_record, regions):
                alt_pad = v[0]
                i = v[1]
                if alt_pad:
                    if alt_pad in called_hotspots:
                        called_hotspots[alt_pad].append([vcf_record, i])
                    else:
                        called_hotspots[alt_pad] = [[vcf_record, i]]

        swap_found = False

        for key in novels:
            if key in uncalled_hotspots:
                # there is one or more an uncalled hotspot matching a called variant
                # novel is 0/1, one or more hotspots match the allele
                # novel is 1/2, one or more hotspots match only one of the 2 alleles

                for hotspot_entry in uncalled_hotspots[key]:

                    # probably a bug in tvcutils merging that more than 1 duplicate novel entry has the same allele but given the limitations of this fix, we don't care, just take the first
                    novels_entry = novels[key][0]
                    swap_ids(key, novels_entry, hotspot_entry)
                    swap_found = True

        if swap_found:
            f_vcf_w = vcf.TvcVcfFile(out_vcf_file, 'w')
            # write header
            f_vcf_w.set_header(f_vcf)

            f_vcf_w.add_to_header('##INFO=<ID=UFR,Number=.,Type=String,Description="List of ALT alleles with Unexpected Filter thresholds">')
        
            for key, entry in novels.iteritems():
                for e in entry:
                    vcf_record = e[0]
                    if not ('HS' in vcf_record['INFO'] and vcf_record['ID'] == '.'):
                        f_vcf_w.write(vcf_record)            

            for key, entry in uncalled_hotspots.iteritems():
                for e in entry:
                    vcf_record = e[0]
                    if not ('HS' in vcf_record['INFO'] and vcf_record['ID'] == '.'):
                        f_vcf_w.write(vcf_record)            

            for key, entry in called_hotspots.iteritems():
                for e in entry:
                    vcf_record = e[0]
                    if not ('HS' in vcf_record['INFO'] and vcf_record['ID'] == '.'):
                        f_vcf_w.write(vcf_record)

            for vcf_record in other_vcf:
                f_vcf_w.write(vcf_record)

            f_vcf_w.uniq(True)
            f_vcf_w.close()

    return

if __name__ == '__main__':


    if len(sys.argv) < 4:
        status = 1
        print "Requires input vcf, input bed and output vcf files as an argument, if no swaps are found then output vcf is not produced"
        exit(status)

    regions = None
    fix_records(sys.argv[1], sys.argv[2], sys.argv[3])
    exit(0)
    

