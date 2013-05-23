#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import subprocess
import traceback
from optparse import OptionParser


def execute_output(cmd):
    try:
        process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
        return process.communicate()[0]
    except:
        traceback.print_exc()
        return ''


def get_chromosome_order(fasta_index):
    chr_order = []
    index_file = open(fasta_index,'r')
    for line in index_file:
        if line:
            chr_order.append(line.split()[0])
    index_file.close()
    print 'Chromosome order: ' + ' '.join(chr_order)
    return chr_order



def parse_vcf_line(input_vcf_line, fasta, allow_block_substitutions):
    
    vcf_record = {}
    
    fields = input_vcf_line.split('\t')
    vcf_record['chr'] = fields[0]
    vcf_record['pos'] = int(fields[1])
    vcf_record['ref'] = fields[3]
    #id = fields[2].split(',')
    vcf_record['alt'] = fields[4].split(',')

    vcf_record['opos'] = []
    vcf_record['oid'] = []
    vcf_record['oref'] = []
    vcf_record['oalt'] = []
    vcf_record['omapalt'] = []
    
    for info_field in fields[7].strip().split(';'):
        subfields = info_field.split('=')
        if len(subfields) != 2:
            continue
        if subfields[0] == 'OPOS':
            vcf_record['opos'] = subfields[1].split(',')
        elif subfields[0] == 'OID':
            vcf_record['oid'] = subfields[1].split(',')
        elif subfields[0] == 'OREF':
            vcf_record['oref'] = subfields[1].split(',')
        elif subfields[0] == 'OALT':
            vcf_record['oalt'] = subfields[1].split(',')

    if len(vcf_record['alt']) > 1:
        print "[postprocess_hotspot_vcf.py] WARNING: Ignoring record %s:%d with multiple alt alleles." % (vcf_record['chr'],vcf_record['pos'])
        return None

    num_orig = len(vcf_record['opos'])
    if len(vcf_record['oid']) != num_orig or len(vcf_record['oref']) != num_orig or len(vcf_record['oalt']) != num_orig :
        print "[postprocess_hotspot_vcf.py] WARNING: Ignoring record %s:%d with incorrect OPOS/OID/OREF/OALT tags" % (vcf_record['chr'],vcf_record['pos'])
        return None

    if len(vcf_record['oid']) != num_orig or len(vcf_record['oref']) != num_orig or len(vcf_record['oalt']) != num_orig :
        print "[postprocess_hotspot_vcf.py] WARNING: Ignoring record %s:%d with incorrect OPOS/OID/OREF/OALT tags" % (vcf_record['chr'],vcf_record['pos'])
        return None

    if allow_block_substitutions == False and len(vcf_record['ref']) > 1 and len(vcf_record['alt'][0]) > 1 and len(vcf_record['ref']) != len(vcf_record['alt'][0]):
        print "[postprocess_hotspot_vcf.py] WARNING: Ignoring record %s:%d with block substitution" % (vcf_record['chr'],vcf_record['pos'])
        return None

    fasta_ref = validate_reference_allele(vcf_record['chr'],vcf_record['pos'],vcf_record['ref'],fasta)
    if fasta_ref != vcf_record['ref']:
        print "[postprocess_hotspot_vcf.py] WARNING: Ignoring record %s:%d with incorrect reference allele (given=%s, expected=%s)" % (vcf_record['chr'],vcf_record['pos'],vcf_record['ref'],fasta_ref)
        return None
    
    vcf_record['omapalt'] = num_orig * vcf_record['alt']

    return vcf_record



def validate_reference_allele(chr,pos,ref,fasta):
    
    faidx_command  = 'samtools faidx'
    faidx_command += ' ' + fasta
    faidx_command += ' %s:%d-%d' % (chr, pos, pos+len(ref)-1)
    
    result = execute_output(faidx_command).splitlines()
    if len(result) > 1:
        return result[1]
    else:
        return None
        


def write_merged_vcf_record(output_vcf,overlapping_vcf_entries, allow_block_substitutions):

    ''' Step 1: Determine longest ref allele '''
    
    ref = ''    
    for vcf_entry in overlapping_vcf_entries:
        if len(ref) < len(vcf_entry['ref']):
            ref = vcf_entry['ref']

    chr = overlapping_vcf_entries[0]['chr']
    pos = overlapping_vcf_entries[0]['pos']
    alt = []
    opos = []
    oid = []
    oref = []
    oalt = []
    omapalt = []


    ''' Step 2: Determine the unified list of alt alleles, remove any new block substitutions '''
    
    for vcf_entry in overlapping_vcf_entries:
        ref_gap = ref[len(vcf_entry['ref']):]
        for local_alt in vcf_entry['alt']:
            xalt = local_alt + ref_gap
            if allow_block_substitutions == False and len(ref) > 1 and len(xalt) > 1 and len(ref) != len(xalt):
                print "[postprocess_hotspot_vcf.py] WARNING: Ignoring record %s:%d with block substitution" % (vcf_entry['chr'],vcf_entry['pos'])
                continue
            if xalt not in alt:
                alt.append(xalt)


    ''' Step 3: Patch ref, alt, omapalt to unify ref length '''

    for vcf_entry in overlapping_vcf_entries:
        ref_gap = ref[len(vcf_entry['ref']):]
        for local_opos,local_oid,local_oref,local_oalt,local_omapalt in zip(vcf_entry['opos'],vcf_entry['oid'],vcf_entry['oref'],vcf_entry['oalt'],vcf_entry['omapalt']):
            if local_omapalt+ref_gap in alt:
                opos.append(local_opos)
                oid.append(local_oid)
                oref.append(local_oref)
                oalt.append(local_oalt)
                omapalt.append(local_omapalt+ref_gap)
    
    
    ''' Step 4: Write out combined VCF entry '''
    
    output_vcf.write('%s\t%d\t%s\t%s\t%s\t.\t.\tOID=%s;OPOS=%s;OREF=%s;OALT=%s;OMAPALT=%s\n' % (
        chr, pos, '.', ref, ','.join(alt),
        ','.join(map(str,oid)),
        ','.join(opos),
        ','.join(oref),
        ','.join(oalt),
        ','.join(omapalt)
    ))
    
    

def main():

    ''' Postprocess Hotspot VCF requirements:
            - Take left-aligned hotspot VCF with OPOS,OID,OREF,OALT info fields
            - Sort the VCF
            - Remove entries with ref allele not matching the actual reference fasta
            - Merge entries with same chr and pos. Merge includes ALT alleles and O* info fields
            - Special handling for duplicate entries TBD
            
    '''


    parser = OptionParser()
    parser.add_option('-i', '--input-vcf',  help='Input VCF file', dest='input_vcf')
    parser.add_option('-o', '--output-vcf', help='Output VCF with collapsed duplicates and removed TEMPID field', dest='output_vcf')
    parser.add_option('-f', '--reference', help='Reference fasta file with index, for validating reference allele', dest='fasta')
    parser.add_option('-a', '--allow-block-substitutions', help='Disable block substitution filter', dest='allow_block_substitutions',
                       action="store_true", default=False)
    (options, args) = parser.parse_args()

    if options.input_vcf is None or options.output_vcf is None or options.fasta is None:
        parser.print_help()
        return 1


    ''' Step 1. Retrieve chromosome index  '''

    chr_order = get_chromosome_order(options.fasta + '.fai')


    ''' Step 2. Load input vcf entries into memory. Copy header lines to output vcf '''

    input_vcf = open(options.input_vcf,'r')
    output_vcf = open(options.output_vcf,'w')
    chr_vcf_entries = dict((chr,[]) for chr in chr_order)
    allow_block_substitutions = options.allow_block_substitutions
    
    for line in input_vcf:
        if not line:
            continue
        if line[0] == '#':
            if line.startswith('##allowBlockSubstitutions=true'):
                allow_block_substitutions = True
            else:
                output_vcf.write(line)
            continue
        vcf_record = parse_vcf_line(line, options.fasta, allow_block_substitutions)
        if vcf_record:
            chr_vcf_entries[vcf_record['chr']].append((vcf_record['pos'],vcf_record))
    
    input_vcf.close()
    

    ''' Step 3. Iterate by chromosomes, and then by sorted positions. Identify vcf lines starting at the same positions. '''
        
    for chr in chr_order:
        
        chr_vcf_entries[chr].sort()
        
        while chr_vcf_entries[chr]:
            current_pos,vcf_entry = chr_vcf_entries[chr].pop(0)
            overlapping_vcf_entries = [vcf_entry]
            while chr_vcf_entries[chr] and chr_vcf_entries[chr][0][0] == current_pos:
                current_pos,vcf_entry = chr_vcf_entries[chr].pop(0)
                overlapping_vcf_entries.append(vcf_entry)
            
            write_merged_vcf_record(output_vcf,overlapping_vcf_entries, allow_block_substitutions)
    
    output_vcf.close()
    
    return 0
        
    

if __name__ == "__main__":    
    exit(main())


