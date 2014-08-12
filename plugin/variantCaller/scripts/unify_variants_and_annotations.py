#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import subprocess
import bisect
import sys
import traceback
import json
from optparse import OptionParser

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return None



def get_chromosome_order(fasta_index):
    chr_order = []
    index_file = open(fasta_index,'r')
    for line in index_file:
        if line:
            chr_order.append(line.split()[0])
    index_file.close()
    print 'Chromosome order: ' + ' '.join(chr_order)
    return chr_order


def parse_header_line(line,header_lines,info_a_list,format_a_list):
    header_lines.append(line)
    
    if line.startswith('##INFO'):
        inner_line = line[8:-2].split(',')
        id = None
        is_A = False
        for field in inner_line:
            if field.startswith('ID='):
                id = field[3:]
            if field == 'Number=A':
                is_A = True
        if id and is_A:
            info_a_list.append(id)
            print '[unify_variants] Identified info_a field: ' + id
        return

    if line.startswith('##FORMAT'):
        inner_line = line[10:-2].split(',')
        id = None
        is_A = False
        for field in inner_line:
            if field.startswith('ID='):
                id = field[3:]
            if field == 'Number=A':
                is_A = True
        if id and is_A:
            format_a_list.append(id)
            print '[unify_variants] Identified format_a field: ' + id
    
    
        
        
    
    

def parse_vcf_line(line,info_a_list,format_a_list):
    
    fields = line.strip().split('\t')
    vcf_record = {}
    vcf_record['chr'] = fields[0]
    vcf_record['pos'] = int(fields[1])
    vcf_record['id']  = fields[2]
    vcf_record['ref'] = fields[3]
    vcf_record['alt'] = fields[4].split(',')
    vcf_record['qual'] = fields[5]
    vcf_record['filter'] = fields[6]
    
    vcf_record['info_order'] = []
    vcf_record['info'] = {}
    info_items = fields[7].split(';')
    for item in info_items:
        subitem = item.split('=',1)
        if len(subitem) == 0:
            continue
        if len(subitem) == 1:
            subitem.append('')
        vcf_record['info_order'].append(subitem[0])
        if subitem[0] in info_a_list:
            vcf_record['info'][subitem[0]] = subitem[1].split(',')
        else:
            vcf_record['info'][subitem[0]] = subitem[1]
    
    if len(fields) > 8:
        vcf_record['format_order'] = []
        vcf_record['format'] = {}
        for name,value in zip(fields[8].split(':'),fields[9].split(':')):
            vcf_record['format_order'].append(name)
            if name in format_a_list:
                vcf_record['format'][name] = value.split(',')
            else:
                vcf_record['format'][name] = value
    if len(fields) > 10:
	vcf_record['normal_format'] = {}    
        for name,value in zip(fields[8].split(':'),fields[10].split(':')):
            if name in format_a_list:
                vcf_record['normal_format'][name] = value.split(',')
            else:
                vcf_record['normal_format'][name] = value
    return vcf_record


def write_vcf_line(output_file,vcf_record):

    output_file.write('%s\t' % vcf_record['chr'])
    output_file.write('%d\t' % vcf_record['pos'])
    output_file.write('%s\t' % vcf_record['id'])
    output_file.write('%s\t' % vcf_record['ref'])
    output_file.write('%s\t' % (','.join(vcf_record['alt'])))
    output_file.write('%s\t' % vcf_record['qual'])
    output_file.write('%s\t' % vcf_record['filter'])
    
    for idx,info in enumerate(vcf_record['info_order']):
        if idx:
            output_file.write(';')
        output_file.write(info)
        value = vcf_record['info'][info]
        if value:
            if type(value) is list:
                output_file.write('=' + (','.join(value)))
            else:
                output_file.write('=' + value)

    if 'oid' in vcf_record:
        output_file.write(';OID=' + (','.join(vcf_record['oid'])))
        output_file.write(';OPOS=' + (','.join(vcf_record['opos'])))
        output_file.write(';OREF=' + (','.join(vcf_record['oref'])))
        output_file.write(';OALT=' + (','.join(vcf_record['oalt'])))
        output_file.write(';OMAPALT=' + (','.join(vcf_record['omapalt'])))

    if 'format' in vcf_record:
        output_file.write('\t%s\t' % ':'.join(vcf_record['format_order']))
        for idx,format in enumerate(vcf_record['format_order']):
            if idx:
                output_file.write(':')
            value = vcf_record['format'][format]
            if type(value) is list:
                output_file.write(','.join(value))
            else:
                output_file.write(value)
    
    if 'normal_format' in vcf_record:
        output_file.write('\t')
        for idx,format in enumerate(vcf_record['format_order']):
            if idx:
                output_file.write(':')
            value = vcf_record['normal_format'][format]
            if type(value) is list:
                output_file.write(','.join(value))
            else:
                output_file.write(value)
    output_file.write('\n')
    
            
    

def merge_hotspot_into_vcf(vcf_record, vcf_hotspot):
    ''' Requirements:
            - Original vcf alleles are all preserved in the original order. This automatically ensures GT stays valid.
            - Examine INFO and Format fields. Copy any new Number!=A field to original vcf. Generate "blank" field for new Number=A fields
            - For all hotspot alleles already present in vcf, copy values in Number=A fields if original field is blank
            - For all new hotspot alleles, append to vcf and to Number=A fields.
    '''

    ''' Combine ALT alleles, normalize ref allele, build hs-to-all map '''
    old_ref_extension = vcf_hotspot['ref'][len(vcf_record['ref']):]
    hs_ref_extension = vcf_record['ref'][len(vcf_hotspot['ref']):]
    vcf_record['ref'] += old_ref_extension
    for old_idx in range(len(vcf_record['alt'])):
        vcf_record['alt'][old_idx] += old_ref_extension
    map_hs_to_all = []
    for hs_idx,hs_alt in enumerate(vcf_hotspot['alt']):
        hs_alt += hs_ref_extension
        if hs_alt not in vcf_record['alt']:
            vcf_record['alt'].append(hs_alt)
        map_hs_to_all.append((hs_idx,vcf_record['alt'].index(hs_alt)))

        
    ''' Transfer INFO fields from vcf_hotspot to vcf_record '''
    for hs_info in vcf_hotspot['info_order']:
        if type(vcf_hotspot['info'][hs_info]) is list:              # Transfer A field
            if hs_info not in vcf_record['info_order']:
                vcf_record['info_order'].append(hs_info)
                vcf_record['info'][hs_info] = []
            vcf_record['info'][hs_info] += (len(vcf_record['alt'])-len(vcf_record['info'][hs_info])) * ['.']            
            for hs_map,all_map in map_hs_to_all:
                if vcf_record['info'][hs_info][all_map] in ['','.']:
                    vcf_record['info'][hs_info][all_map] = vcf_hotspot['info'][hs_info][hs_map]
        else:
            if hs_info not in vcf_record['info_order']:             # Transfer non-A field
                vcf_record['info_order'].append(hs_info)
                vcf_record['info'][hs_info] = vcf_hotspot['info'][hs_info]

    original_format_order = vcf_record['format_order'][:]
    ''' Transfer FORMAT fields from vcf_hotspot to vcf_record '''
    for hs_format in vcf_hotspot['format_order']:
        if type(vcf_hotspot['format'][hs_format]) is list:          # Transfer A field
            if hs_format not in vcf_record['format_order']:
                vcf_record['format_order'].append(hs_format)
                vcf_record['format'][hs_format] = []
            vcf_record['format'][hs_format] += (len(vcf_record['alt'])-len(vcf_record['format'][hs_format])) * ['.']            
            for hs_map,all_map in map_hs_to_all:
                if vcf_record['format'][hs_format][all_map] in ['','.'] and hs_map < len(vcf_hotspot['format'][hs_format]):
                    vcf_record['format'][hs_format][all_map] = vcf_hotspot['format'][hs_format][hs_map]
        else:
            if hs_format not in vcf_record['format_order']:         # Transfer non-A field
                vcf_record['format_order'].append(hs_format)
                vcf_record['format'][hs_format] = vcf_hotspot['format'][hs_format]
    if 'normal_format' in vcf_hotspot:
        for hs_normal_format in vcf_hotspot['format_order']:
            if type(vcf_hotspot['normal_format'][hs_normal_format]) is list:          # Transfer A field
                if hs_normal_format not in original_format_order:
                    vcf_record['normal_format'][hs_normal_format] = []
                vcf_record['normal_format'][hs_normal_format] += (len(vcf_record['alt'])-len(vcf_record['normal_format'][hs_normal_format])) * ['.']            
                for hs_map,all_map in map_hs_to_all:
                    if vcf_record['normal_format'][hs_normal_format][all_map] in ['','.'] and hs_map < len(vcf_hotspot['normal_format'][hs_normal_format]):
                        vcf_record['normal_format'][hs_normal_format][all_map] = vcf_hotspot['normal_format'][hs_normal_format][hs_map]
            else:
                if hs_normal_format not in original_format_order:         # Transfer non-A field
                    vcf_record['normal_format'][hs_normal_format] = vcf_hotspot['normal_format'][hs_normal_format]
    

def generate_novel_annotation(vcf_record):

    vcf_record['oid'] = []
    vcf_record['opos'] = []
    vcf_record['oref'] = []
    vcf_record['oalt'] = []
    vcf_record['omapalt'] = []

    for alt in vcf_record['alt']:
        opos = vcf_record['pos']
        oref = vcf_record['ref']
        oalt = alt
        while len(oref) >= 1 and len(oalt) >= 1 and oref[-1] == oalt[-1]:
            oref = oref[:-1]
            oalt = oalt[:-1]
        while len(oref) >= 1 and len(oalt) >= 1 and oref[0] == oalt[0]:
            oref = oref[1:]
            oalt = oalt[1:]
            opos += 1
        if not oref:
            oref = '-'
        if not oalt:
            oalt = '-'
        vcf_record['oid'].append('.')
        vcf_record['opos'].append(str(opos))
        vcf_record['oref'].append(oref)
        vcf_record['oalt'].append(oalt)
        vcf_record['omapalt'].append(alt)


def merge_annotation_into_vcf(vcf_record, vcf_annotation):

    all_oid     = vcf_annotation['info']['OID'].split(',')
    all_opos    = vcf_annotation['info']['OPOS'].split(',')
    all_oref    = vcf_annotation['info']['OREF'].split(',')
    all_oalt    = vcf_annotation['info']['OALT'].split(',')
    all_omapalt = vcf_annotation['info']['OMAPALT'].split(',')
    filtered_oid = []

    annotation_ref_extension = vcf_record['ref'][len(vcf_annotation['ref']):]
    record_ref_extension = 0
    if len(vcf_annotation['ref']) > len(vcf_record['ref']):
        record_ref_extension = len(vcf_annotation['ref'])-len(vcf_record['ref'])
    
    blacklist = dict(zip(vcf_annotation['alt'],vcf_annotation['info'].get('BSTRAND','.').split(',')))

    for oid,opos,oref,oalt,omapalt in zip(all_oid,all_opos,all_oref,all_oalt,all_omapalt):
        if blacklist.get(omapalt,'.') != '.':
            continue
        filtered_oid.append(oid)

        if record_ref_extension:
            if vcf_annotation['ref'][-record_ref_extension:] != omapalt[-record_ref_extension:]:
                print '[unify_variants] Hotspot annotation %s:%d, allele %s not eligible for shortening' % (vcf_record['chr'],vcf_record['pos'],omapalt)
                continue
            omapalt = omapalt[:-record_ref_extension]
        
        omapalt += annotation_ref_extension
        if omapalt in vcf_record['omapalt']:
            idx = vcf_record['omapalt'].index(omapalt)
        else:
            idx = None
        if idx is None:
            print '[unify_variants] Hotspot annotation %s:%d, allele %s not found in merged variant file' % (vcf_record['chr'],vcf_record['pos'],omapalt)
            continue
        
        if len(oref) >= 1 and len(oalt) >= 1 and oref[0] == oalt[0]:
            oref = oref[1:]
            oalt = oalt[1:]
            opos = str(int(opos) + 1)
        if not oref:
            oref = '-'
        if not oalt:
            oalt = '-'
        
        if vcf_record['oid'][idx] == '.':   # Minimal representation present, remove
            vcf_record['oid'][idx] = oid
            vcf_record['opos'][idx] = opos
            vcf_record['oref'][idx] = oref
            vcf_record['oalt'][idx] = oalt
            vcf_record['omapalt'][idx] = omapalt
        else:
            vcf_record['oid'].append(oid)
            vcf_record['opos'].append(opos)
            vcf_record['oref'].append(oref)
            vcf_record['oalt'].append(oalt)
            vcf_record['omapalt'].append(omapalt)

    if filtered_oid:
        vcf_record['id'] = ';'.join(filtered_oid)


def main():
    ''' Unify Variants and Annotations requirements:
            - Accept novel small variants from a tvc-generated vcf file
            
            - If IndelAssembly present, accept novel assembly indels from IndelAssembly-generated vcf file
            - Merge above two lists. If overlap encountered, remove IndelAssembly allele. Ensure result is sorted
            
            - If hotspots, accept hotspot calls from another tvc-generated vcf file
            - Merge hotspot calls with the de-novo calls. Merge alleles, info fields, format fields.
            
            - Populate OID,OREF,OALT,OPOS,OMAPALT using "minimal" representation
            - If hotspots, accept hotspot annotation vcf file
            - Substitute OID,OREF,OALT,OPOS,OMAPALT for all alleles in hotspot annotation file.
    '''

    parser = OptionParser()
    parser.add_option('-t', '--novel-tvc-vcf',          help='VCF file with novel tvc variants', dest='novel_tvc_vcf')
    parser.add_option('-i', '--novel-assembly-vcf',     help='VCF file with novel IndelAssembly variants', dest='novel_assembly_vcf')
    parser.add_option('-a', '--hotspot-annotation-vcf', help='VCF file with hotspot annotations', dest='hotspot_annotation_vcf')
    parser.add_option('-o', '--output-vcf',             help='Output VCF with unified and annotated variants', dest='output_vcf')
    parser.add_option('-r', '--index-fai',              help='Reference genome index for chromosome order', dest='index') 
    parser.add_option('-c', '--single-chromosome',      help='Process only one named chromosome', dest='single_chromosome') 
    parser.add_option('-j', '--tvc-metrics',            help='Metrics file generated by tvc', dest='tvc_metrics') 
    (options, args) = parser.parse_args()

    if options.novel_tvc_vcf is None or options.output_vcf is None:
        parser.print_help()
        return 1

    ''' Step 1. Retrieve chromosome index '''

    if options.single_chromosome:
        chr_order = [options.single_chromosome]
    else:
        if options.index is None:
            parser.print_help()
            return 1
        chr_order = get_chromosome_order(options.index)

    ''' Step 2. Load novel tvc file into memory and sort '''

    chr_vcf = dict((chr,[]) for chr in chr_order)
    chr_pos = dict((chr,[]) for chr in chr_order)
    header_lines = []
    info_a_list = []
    format_a_list = []
    
    input_vcf = open(options.novel_tvc_vcf,'r')
    line_number = 0
    for line in input_vcf:
        line_number += 1
        try:
            if not line:
                continue
            if line[0] == '#':
                parse_header_line(line,header_lines,info_a_list,format_a_list)
                continue
            vcf_record = parse_vcf_line(line,info_a_list,format_a_list)
            if vcf_record and vcf_record['chr'] in chr_order:
                chr_vcf[vcf_record['chr']].append((vcf_record['pos'],vcf_record))
        except:
            print "[unify_variants] Error while parsing %s line %d" % (options.novel_tvc_vcf,line_number)
            raise
    input_vcf.close()

    for chr in chr_order:
        chr_vcf[chr].sort()
        chr_pos[chr] = [pos for (pos,x) in chr_vcf[chr]]


    ''' Step 3. Load assembly results, merge in. '''

    if options.novel_assembly_vcf:
        
        chr_vcf_assembly = dict((chr,[]) for chr in chr_order)
        input_vcf = open(options.novel_assembly_vcf,'r')
        for line in input_vcf:
            if not line or line[0] == '#':
                continue
            vcf_record = parse_vcf_line(line,info_a_list,format_a_list)
            if 'FR' not in vcf_record['info_order']:
                vcf_record['info_order'].append('FR')
                vcf_record['info']['FR'] = '.'
            if not vcf_record or vcf_record['chr'] not in chr_order:
                continue
            
            # Case 1: IndelAssembly vcf record does not overlap TVC record
            #         -> include IndelAssembly
            idx = index(chr_pos[vcf_record['chr']],vcf_record['pos'])
            if idx is None:
                chr_vcf_assembly[vcf_record['chr']].append((vcf_record['pos'],vcf_record))
                continue
            
            # Case 2: IndelAssembly overlaps TVC record, but TVC has 0/0 or ./. genotype
            #         -> merge both records, keep IndelAssembly genotype. What about FR?
            if chr_vcf[vcf_record['chr']][idx][1]['format']['GT'] in ['0/0', './.']:
                merge_hotspot_into_vcf(vcf_record, chr_vcf[vcf_record['chr']][idx][1])
                chr_vcf[vcf_record['chr']][idx] = (vcf_record['pos'],vcf_record)
                print "[unify_variants] Advanced merge of IndelAssembly variant %s:%d" % (vcf_record['chr'],vcf_record['pos'])
                continue
            
            # Case 3: IndelAssembly overlaps TVC record and TVC has genotype other than 0/0, ./.
            #         -> skip IndelAssembly altogether
            print "[unify_variants] Skipping IndelAssembly variant %s:%d" % (vcf_record['chr'],vcf_record['pos'])
            
        input_vcf.close()
    
        for chr in chr_order:
            chr_vcf[chr] += chr_vcf_assembly[chr]
            chr_vcf[chr].sort()
            chr_pos[chr] = [pos for (pos,x) in chr_vcf[chr]]
        
        del chr_vcf_assembly
        
    
    ''' Step 4. Load hotspot results, merge in. '''
    '''
    if options.hotspot_tvc_vcf:
        
        chr_vcf_hs_new = dict((chr,[]) for chr in chr_order)
        
        input_vcf = open(options.hotspot_tvc_vcf,'r')
        for line in input_vcf:
            if not line or line[0] == '#':
                continue
            vcf_record = parse_vcf_line(line,info_a_list,format_a_list)
            if vcf_record and vcf_record['chr'] in chr_order:
                idx = index(chr_pos[vcf_record['chr']],vcf_record['pos'])
                if idx is None:
                    chr_vcf_hs_new[vcf_record['chr']].append((vcf_record['pos'],vcf_record))
                else:
                    merge_hotspot_into_vcf(chr_vcf[vcf_record['chr']][idx][1], vcf_record)
        input_vcf.close()

        for chr in chr_order:
            chr_vcf[chr] += chr_vcf_hs_new[chr]
            chr_vcf[chr].sort()
            chr_pos[chr] = [pos for (pos,x) in chr_vcf[chr]]

        del chr_vcf_hs_new

    '''



    ''' Step 5. Compute minimal representation OID;OPOS;OREF;OALT values for all alleles '''
    
    for chr in chr_order:
        for pos,vcf_record in chr_vcf[chr]:
            generate_novel_annotation(vcf_record)
    
    
    ''' Step 6. Load hotspot annotation file, copy OID;OPOS;OREF;OALT annotation to all fields '''
    
    if options.hotspot_annotation_vcf:

        input_vcf = open(options.hotspot_annotation_vcf,'r')
        for line in input_vcf:
            if not line or line[0] == '#':
                continue
            vcf_record = parse_vcf_line(line,[],[])
            if vcf_record and vcf_record['chr'] in chr_order:
                idx = index(chr_pos[vcf_record['chr']],vcf_record['pos'])
                if idx is None:
                    if '.' in vcf_record['info'].get('BSTRAND','.'):
                        print '[unify_variants] Hotspot annotation %s:%d not found in merged variant file' % (vcf_record['chr'],vcf_record['pos'])
                        continue
                else:
                    merge_annotation_into_vcf(chr_vcf[vcf_record['chr']][idx][1], vcf_record)
        input_vcf.close()


    ''' Step 7. Save final set of records into output vcf '''
    
    output_file = open(options.output_vcf,'w')
    output_file.writelines(header_lines[0:-1])
    output_file.write('##INFO=<ID=OID,Number=.,Type=String,Description="List of original Hotspot IDs">\n')
    output_file.write('##INFO=<ID=OPOS,Number=.,Type=Integer,Description="List of original allele positions">\n')
    output_file.write('##INFO=<ID=OREF,Number=.,Type=String,Description="List of original reference bases">\n')
    output_file.write('##INFO=<ID=OALT,Number=.,Type=String,Description="List of original variant bases">\n')
    output_file.write('##INFO=<ID=OMAPALT,Number=.,Type=String,Description="Maps OID,OPOS,OREF,OALT entries to specific ALT alleles">\n')
    
    if options.tvc_metrics:
        json_file = open(options.tvc_metrics, 'r')
        tvc_metrics = json.load(json_file)
        json_file.close()
        try:
            output_file.write('##deamination_metric='+str(tvc_metrics['metrics']['deamination_metric'])+'\n')
        except:
            pass
    
    output_file.write(header_lines[-1])
    
    for chr in chr_order:
        for pos,vcf_record in chr_vcf[chr]:
            write_vcf_line(output_file,vcf_record)
    
    output_file.close()
    
    return 0


if __name__ == "__main__":    
    sys.exit(main())

