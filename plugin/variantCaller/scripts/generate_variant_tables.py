#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import bisect
import json
import traceback
import copy
from collections import defaultdict
from optparse import OptionParser

from matplotlib import use
use("Agg",warn=False)
import matplotlib.pyplot as plt
        
def write_alleles2_line(options, fid, **kwargs):
    header = kwargs.get('header', False)
    allele = kwargs.get('data',{})
    report_ppa = kwargs.get('report_ppa', False)
    report_subset = kwargs.get('report_subset', False)

    fid.write(('Chrom'                  if header else allele['chrom'])                 + '\t')
    fid.write(('Position'               if header else allele['pos'])                   + '\t')
    fid.write(('Ref'                    if header else allele['ref'])                   + '\t')
    fid.write(('Variant'                if header else allele['alt'])                   + '\t')
    fid.write(('Allele Call'            if header else allele['call'])                  + '\t')
    fid.write(('Filter'                 if header else allele['call_filter'])           + '\t')
    fid.write(('Frequency'              if header else allele['freq'])                  + '\t')
    if report_ppa:
        fid.write(('Possible Polyploidy Allele' \
                                        if header else allele['is_ppa'])                + '\t')
    fid.write(('Quality'                if header else allele['qual'])                  + '\t')
    fid.write(('Filter'                 if header else allele['qual_filter'])           + '\t')
    if (options.library_type == "tagseq"):
        fid.write(('LOD'                if header else allele['LOD'])                   + '\t')

    # Extra fields displayed only in Allele Search view
    fid.write(('Type'                   if header else allele['type'])                  + '\t')
    fid.write(('Allele Source'          if header else allele['source'])                + '\t')
    fid.write(('Allele Name'            if header else allele['name'])                  + '\t')
    fid.write(('Gene ID'                if header else allele['gene'])                  + '\t')
    fid.write(('Region Name'            if header else allele['submitted_region'])      + '\t')
    if report_subset:
        fid.write(('Subset Of'            if header else allele['subset_of'])      + '\t')

    fid.write(('VCF Position'           if header else allele['pos_vcf'])               + '\t')
    fid.write(('VCF Ref'                if header else allele['ref_vcf'])               + '\t')
    fid.write(('VCF Variant'            if header else allele['alt_vcf'])               + '\t')
    
    # Extra fields displayed only in Coverage Filters view

    if (options.library_type == "tagseq"):
        fid.write(('Read Cov'               if header else allele['read_cov'])       + '\t')
        fid.write(('Allele Read Cov'        if header else allele['allele_read_cov'])       + '\t')
        fid.write(('Allele Read Freq'       if header else allele['allele_read_freq'])       + '\t')
        fid.write(('Mol Coverage'           if header else allele['mol_coverage'])            + '\t')
        fid.write(('Filter'                 if header else allele['mol_coverage_filter'])            + '\t')
        fid.write(('Allele Mol Cov'         if header else allele['allele_mol_cov'])       + '\t')
        fid.write(('Filter'                 if header else allele['allele_mol_cov_filter'])+ '\t')
        fid.write(('Allele Mol Freq'        if header else allele['allele_mol_freq'])      + '\t')
        fid.write(('Filter'                 if header else allele['allele_mol_freq_filter']) + '\t')
    else:
        fid.write(('Original Coverage'      if header else allele['cov_total'])             + '\t')
        fid.write(('Coverage'               if header else allele['cov_total_downsampled']) + '\t')
        fid.write(('Filter'                 if header else allele['cov_total_filter'])      + '\t')        
        fid.write(('Coverage+'              if header else allele['cov_total_plus'])        + '\t')
        fid.write(('Filter'                 if header else allele['cov_total_plus_filter']) + '\t')
        fid.write(('Coverage-'              if header else allele['cov_total_minus'])       + '\t')
        fid.write(('Filter'                 if header else allele['cov_total_minus_filter'])+ '\t')
        fid.write(('Allele Cov'             if header else allele['cov_allele'])            + '\t')
        fid.write(('Allele Cov+'            if header else allele['cov_allele_plus'])       + '\t')
        fid.write(('Allele Cov-'            if header else allele['cov_allele_minus'])      + '\t')
    fid.write(('Strand Bias'            if header else allele['strand_bias'])           + '\t')
    fid.write(('Filter'                 if header else allele['strand_bias_filter'])    + '\t')

    # Extra fields displayed only in Quality Filters view
    fid.write(('Common Signal Shift'    if header else allele['rbi'])                   + '\t')
    fid.write(('Filter'                 if header else allele['rbi_filter'])            + '\t')
    fid.write(('Reference Signal Shift' if header else allele['refb'])                  + '\t')
    fid.write(('Filter'                 if header else allele['varb_filter'])           + '\t')
    fid.write(('Variant Signal Shift'   if header else allele['varb'])                  + '\t')
    fid.write(('Filter'                 if header else allele['varb_filter'])           + '\t')
    fid.write(('Relative Read Quality'  if header else allele['mlld'])                  + '\t')
    fid.write(('Filter'                 if header else allele['mlld_filter'])           + '\t')
    fid.write(('HP Length'              if header else allele['hp_length'])             + '\t')
    fid.write(('Filter'                 if header else allele['hp_length_filter'])      + '\t')
    fid.write(('Context Error+'         if header else allele['sse_plus'])              + '\t')
    fid.write(('Filter'                 if header else allele['sse_plus_filter'])       + '\t')
    fid.write(('Context Error-'         if header else allele['sse_minus'])             + '\t')
    fid.write(('Filter'                 if header else allele['sse_minus_filter'])      + '\t')
    fid.write(('Context Strand Bias'    if header else allele['sssb'])                  + '\t')
    fid.write(('Filter'                 if header else allele['sssb_filter'])           + '\t')
    
    # More fields to aid dealing with multiple samples
    fid.write(('Sample Name'            if header else allele['sample'])                + '\t')
    fid.write(('Barcode'                if header else allele['barcode'])               + '\t')
    fid.write(('Run Name'               if header else allele['run_name'])              + '\t')
    fid.write(('Allele'                 if header else allele['gene'] + " " + allele['name']) + '\t')
    if not header:
        temp_chr = allele['chrom']
        if temp_chr.startswith("chr") and len(temp_chr) == 4:
            temp_chr = "chr0" + temp_chr[3:]
        temp_pos = allele['pos'];
        while len(temp_pos) < 9:
            temp_pos = '0' + temp_pos
    fid.write(('Location'               if header else temp_chr + ":" + temp_pos) + '\n')


                    
    
        
def num_get(my_dict, my_key, default):
    try:
        return float(my_dict[my_key])
    except:
        return default


def num_get_list(my_dict, my_key, default):
    value = copy.copy(default)
    for idx,txt in enumerate(my_dict.get(my_key,'').split(',')):
        try:
            value[idx] = float(txt)
        except:
            pass
    return value



def main():
    
    parser = OptionParser()
    parser.add_option('-z', '--suppress-no-calls',      help='Suppress no calls [on/off]', dest='suppress_no_calls') 
    parser.add_option('-i', '--input-vcf',      help='Input vcf file to be sorted', dest='input') 
    parser.add_option('-r', '--region-bed',     help='Region bed file (optional)', dest='region') 
    parser.add_option('-B', '--barcode',        help='Barcode name (optional)', dest='barcode')
    parser.add_option('-R', '--run-name',       help='Run name (optional)', dest='run_name')
    parser.add_option('-s', '--hotspots',       help='Generate hotspot ID column', dest='hotspot', action="store_true", default=False)
    parser.add_option('-o', '--output-xls',     help='Variant table tab-delimited file', dest='output')
    parser.add_option('-a', '--alleles-xls',    help='Alleles table tab-delimited file', dest='alleles')
    parser.add_option('-b', '--alleles2-xls',   help='Extended alleles table tab-delimited file', dest='alleles2')
    parser.add_option('-c', '--concatenated-xls', help='Concatenated alleles table', dest='concatenated_xls')
    parser.add_option('-S', '--scatter-png',    help='Scatterplot of coverage vs. frequency for variants', dest='scatter_png')
    parser.add_option('-j', '--summary-json',   help='Variant summary in json file', dest='summary')
    parser.add_option('-l', '--library-type',   help='Library type', dest='library_type')
    (options, args) = parser.parse_args()

    if options.input is None:
        sys.stderr.write('[generate_variant_tables.py] Error: --input-vcf not specified\n')
        return 1
    if options.output is None:
        sys.stderr.write('[generate_variant_tables.py] Error: --output-xls not specified\n')
        return 1


    #
    # Step 1: Retrieve regions bed file
    #
    
    region_start = defaultdict(lambda: defaultdict(defaultdict))
    region_end = defaultdict(lambda: defaultdict(defaultdict))
    region_ids = defaultdict(lambda: defaultdict(defaultdict))
    
    if options.region:
        region_bed_file = open(options.region,'r')
        is_ion_version_4 = False
        for line in region_bed_file:
            if not line:
                continue
            if line.startswith('track '):
                if 'torrentSuiteVersion=3.6' in line or 'ionVersion=4.0' in line:
                    is_ion_version_4 = True
            fields = line.split('\t')
            if len(fields) < 6:
                continue
            chrom = fields[0].strip()
            region_start.setdefault(chrom,[]).append(int(fields[1]) + 1)
            region_end.setdefault(chrom,[]).append(int(fields[2]))
            if is_ion_version_4:
                gene_id = 'unknown'
                for subfield in fields[-1].strip().split(';'):
                    key_val = subfield.split('=')
                    if len(key_val) == 2 and key_val[0] == 'GENE_ID':
                        gene_id = key_val[1]
                        break
            else:
                gene_id = fields[-1].strip()
            region_ids.setdefault(chrom,[]).append((fields[3].strip(), gene_id))
        region_bed_file.close()
    
    
    #
    # Step 3: Convert and annotate vcf

    # Do I want to report ppa?
    is_report_ppa = False
    with open(options.input,'r') as input_vcf:
        for line in input_vcf:
            if not line:
                continue
            if not line.startswith("#"):
                break
            elif line.startswith('##INFO=<ID=PPA,'):
                is_report_ppa = True
                break
    # Do I want to report subset?
    is_report_subset = False
    with open(options.input,'r') as input_vcf:
        for line in input_vcf:
            if not line:
                continue
            if not line.startswith("#"):
                break
            elif line.startswith('##INFO=<ID=SUBSET,'):
                is_report_subset = True
                break

    input_vcf = open(options.input,'r')
    output_xls = open(options.output,'w')
    my_headers = ['Chrom', 'Position', 'Gene Sym', 'Target ID', 'Type', 'Zygosity', 'Genotype', 'Ref', 'Variant', 'Var Freq', 'Qual', 'Coverage', 'Ref Cov', 'Var Cov"']
    if options.hotspot:
        my_headers.append('HotSpot ID')
    output_xls.write('\t'.join(my_headers) + '\n')

    if options.alleles2:
        alleles2_xls = open(options.alleles2,'w')
        write_alleles2_line(options, alleles2_xls, header=True, report_ppa=is_report_ppa, report_subset=is_report_subset)

    if options.concatenated_xls:
        skip_header = os.path.exists(options.concatenated_xls)
        concatenated_xls = open(options.concatenated_xls,'a')
        if not skip_header:
            write_alleles2_line(options, concatenated_xls, header=True, report_ppa=is_report_ppa, report_subset=is_report_subset)

    observed_chr_order = []
    chr_calls_total = {}
    chr_calls_het_snp = {}
    chr_calls_hom_snp = {}
    chr_calls_het_indel = {}
    chr_calls_hom_indel = {}
    chr_calls_other = {}
    chr_calls_none = {}
    
    hotspot_total = 0
    hotspot_het_snp = 0
    hotspot_hom_snp = 0
    hotspot_het_indel = 0
    hotspot_hom_indel = 0
    hotspot_other = 0
    hotspot_none = 0

    hom_freq = []
    hom_cov = []
    het_freq = []
    het_cov = []
    abs_freq = []
    abs_cov = []
    nc_freq = []
    nc_cov = []
    
    novel_counter_for_subset = 0
    
    summary_json = {
        'sample_name' : '',
        'variants_total' : {
            'variants' : 0,
            'het_snps' : 0,
            'homo_snps' : 0,
            'het_indels' : 0,
            'homo_indels' : 0,
            'other' : 0,
            'no_call' : 0
        },
        'variants_by_chromosome' : [],
        'variants_by_call' : {
            'absent' : 0,
            'heterozygous' : 0,
            'homozygous' : 0,
            'no_call' : 0
        },
        'variants_by_source' : {
            'novel' : 0,
            'hotspot' : 0
        },
        'variants_by_type' : {
            'snp' : 0,
            'ins' : 0,
            'del' : 0,
            'mnp' : 0,
            'complex' : 0
        },
        'filters' : {
            'min_coverage' : { 'present':0, 'absent':0, 'filtered':0 },
            'min_cov_each_strand' : { 'present':0, 'absent':0, 'filtered':0 },
            'strand_bias' : { 'present':0, 'absent':0, 'filtered':0 },
            'beta_bias' : { 'present':0, 'absent':0, 'filtered':0 },
            'min_variant_score' : { 'present':0, 'absent':0, 'filtered':0 },
            'hp_max_length' : { 'present':0, 'absent':0, 'filtered':0 },
            'data_quality_stringency' : { 'present':0, 'absent':0, 'filtered':0 },
            'sse_one_strand' : { 'present':0, 'absent':0, 'filtered':0 },
            'sse_both_strands' : { 'present':0, 'absent':0, 'filtered':0 },
            'rejection' : { 'present':0, 'absent':0, 'filtered':0 },
            'filter_x_predictions' : { 'present':0, 'absent':0, 'filtered':0 },
            'filter_unusual_predictions' : { 'present':0, 'absent':0, 'filtered':0 }
        }
                    
    }
    
    #TODO: Use TvcVcfFile to parse the vcf records.   
    for line in input_vcf:
        if not line:
            continue

        if line.startswith('#CHROM'):
            fields = line.split('\t')
            if len(fields) > 9:
                sample_name = ""
                elements = fields[9].strip().split('.')
                for index in range(0, max(1, len(elements) - 2)):
                    if len(sample_name) > 0:
                        sample_name += "."
                    sample_name += elements[index]
                summary_json['sample_name'] = sample_name
        if line[0]=='#':
            continue

        fields = line.split('\t')
        # variant type
        chr = fields[0]
        pos = fields[1]
        id = fields[2]
        ref = fields[3]
        alt = fields[4].split(',')
        qual = fields[5]
        info = {}
        for item in fields[7].split(';'):
            subitem = item.split('=',1)
            if len(subitem) == 1:
                subitem.append('')
            if len(subitem) == 2:
                info[subitem[0]] = subitem[1]
        format = dict(zip(fields[8].split(':'),fields[9].split(':')))

        
        variant_type = 'Other' # SNP by default
        if len(ref) == 1 and len(alt[0]) == 1:
            variant_type = 'SNP'
        if len(alt[0]) < len(ref):
            variant_type = 'DEL'
        if len(alt[0]) > len(ref):
            variant_type = 'INS'
        if len(alt[0]) == len(ref) and len(alt[0]) > 1:
            variant_type = 'MNP'
        

        (region_id,gene_name) = ("N/A","N/A")
        if options.region and region_start[chr]:
            list_pos = bisect.bisect_right(region_start[chr], int(pos)) - 1
            if list_pos >= 0 and int(pos) <= region_end[chr][list_pos]:
                (region_id,gene_name) = region_ids[chr][list_pos]
                    
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

        if is_report_ppa:
            PPA = info.get('PPA', genotype).split('/')
        
        if is_report_subset:
            SUBSET = info.get('SUBSET', ','.join(['.']*len(alt))).split(',')

        # Coverages       
        LOD  = num_get_list(info, 'LOD', [1]*len(alt))
        DP   = num_get(info, 'DP', 0)
        AF   = num_get_list(info, 'AF', [0]*len(alt))
        RO   = num_get(info, 'RO', 0)
        FDP  = num_get(info, 'FDP', DP)
        FRO  = num_get(info, 'FRO', RO)
        AO   = num_get_list(info, 'AO', [0]*len(alt))
        FAO  = num_get_list(info, 'FAO', AO)
        SRF  = num_get(info, 'SRF', 0)
        FSRF = num_get(info, 'FSRF', SRF)
        SRR  = num_get(info, 'SRR', 0)
        FSRR = num_get(info, 'FSRR', SRR)
        SAF  = num_get_list(info, 'SAF', [0]*len(alt))
        FSAF = num_get_list(info, 'FSAF', SAF)
        SAR  = num_get_list(info, 'SAR', [0]*len(alt))
        FSAR = num_get_list(info, 'FSAR', SAR)
    
        # Outlier and/or heal-snps adjusted FDP and DP
        DP   = RO + sum(AO)
        FDP  = FRO + sum(FAO)
    
        #Fiters
        SSEN = num_get_list(info, 'SSEN', [0]*len(alt))
        SSEP = num_get_list(info, 'SSEP', [0]*len(alt))
        SSSB = num_get_list(info, 'SSSB', [0]*len(alt))
        STB  = num_get_list(info, 'STB', [0]*len(alt))
        SXB  = num_get_list(info, 'SXB', [0]*len(alt))
        RBI  = num_get_list(info, 'RBI', [0]*len(alt))
        REFB = num_get_list(info, 'REFB', [0]*len(alt))
        VARB = num_get_list(info, 'VARB', [0]*len(alt))
        HRUN = num_get_list(info, 'HRUN', [0]*len(alt))
        MLLD = num_get_list(info, 'MLLD', [0]*len(alt))
        FR   = info.get('FR','')
        
        # Write variants.xls
        total_cov = DP
        total_f_cov = FDP # In tvc, MRO = FRO, MAO = FAO
        var_freq = [100.0 * float(v)/float(total_cov) if total_cov > 0 else 0.0 for v in AO]
        # f_var_freq = [100.0 * float(v)/float(total_f_cov) if total_f_cov > 0 else 0.0 for v in FAO]
        f_var_freq = [100.0 * f for f in AF] # TS-14592        
        output_xls.write("%s\t%s\t" % (chr,pos)) # Chrom, Position
        output_xls.write("%s\t%s\t" % (gene_name,region_id)) # Gene Sym, Target ID
        output_xls.write("%s\t%s\t%s\t" % (variant_type,ploidy,genotype_actual)) # Type, Zygosity
        output_xls.write("%s\t%s\t" % (fields[3],fields[4])) # Ref, Variant
        output_xls.write("%s\t%s\t%s\t%s\t%s" % (sum(f_var_freq),qual,total_f_cov,FRO,sum(FAO)))

        if options.hotspot:
            hotspot_annotation = id
            if hotspot_annotation != '.':
                output_xls.write("\t"+hotspot_annotation)
            else:
                output_xls.write("\t---")

        output_xls.write("\n")
        # Write variants.xls done

        # Now handling alleles.xls
        all_oid     = info['OID'].split(',')
        all_opos    = info['OPOS'].split(',')
        all_oref    = info['OREF'].split(',')
        all_oalt    = info['OALT'].split(',')
        all_omapalt = info['OMAPALT'].split(',')
    
        # I am trying to save the FR field from the abuse of realignemnt and heal-snps.
        frs = [fr for fr in FR.split(',') if not (fr.startswith('SKIPREALIGNx') or fr.startswith('REALIGNEDx') or fr.startswith('HEALED'))]
        oid_dict = {}

        for oid,opos,oref,oalt,omapalt in zip(all_oid,all_opos,all_oref,all_oalt,all_omapalt):
            if omapalt not in alt:
                continue
            idx = alt.index(omapalt)
            key = str(opos) + ":" + oref + ":" + oalt + ":" + str(idx)
            if oid != ".":
                if key in oid_dict:
                    if oid_dict[key] != "":
                        oid_dict[key] += "," + oid
                    else:
                        oid_dict[key] = oid
                else:
                    oid_dict[key] = oid
            else:
                oid_dict[key] = ""
        
        if is_report_subset:
            allele_name_list = []
            for oid in all_oid:
                if oid in ['', '.']:
                    novel_counter_for_subset += 1
                    # I need allele name for subset.
                    allele_name_list.append('tvc.novel.%d' %novel_counter_for_subset)
                else:
                    allele_name_list.append(oid)
        
        for oid,opos,oref,oalt,omapalt in zip(all_oid,all_opos,all_oref,all_oalt,all_omapalt):
            if omapalt not in alt:
                continue
            idx = alt.index(omapalt)

            key = str(opos) + ":" + oref + ":" + oalt + ":" + str(idx)
            if key not in oid_dict:
                continue
            oid = oid_dict[key]
            del oid_dict[key]
            if oid == "":
                oid = "."

            if (options.suppress_no_calls == "on"):
                if oid == '.' and genotype1_int != (idx+1) and genotype2_int != (idx+1):
                    continue

            allele = {}
            allele['chrom']             = chr
            allele['pos']               = opos
            allele['ref']               = oref
            allele['alt']               = oalt
            allele['pos_vcf']           = pos
            allele['ref_vcf']           = ref
            allele['alt_vcf']           = alt[idx]
            allele['name']              = '---' if oid == '.' else oid
            allele['source']            = 'Novel' if oid == '.' else 'Hotspot'
            allele['gene']              = gene_name
            allele['submitted_region']  = region_id
            allele['qual']              = qual
            allele['sample']            = summary_json.get('sample_name','N/A')
            allele['barcode']           = options.barcode if options.barcode else 'N/A'
            allele['run_name']          = options.run_name if options.run_name else 'N/A'

            if is_report_ppa:
                allele['is_ppa'] = '1' if str(idx + 1) in PPA else '0' 
            
            if is_report_subset:
                allele['name'] = allele_name_list[idx]
                if SUBSET[idx] != '.':
                    my_super_sets = [allele_name_list[int(my_super_set_idx) - 1] for my_super_set_idx in SUBSET[idx].split('/')]
                    allele['subset_of'] = ','.join(my_super_sets)
                else:
                    allele['subset_of'] = '---'
                
            ref_len = len(oref.strip('-'))
            alt_len = len(oalt.strip('-'))
            if ref_len == 1 and alt_len == 1:
                allele['type'] = 'SNP'
                summary_json['variants_by_type']['snp'] += 1
            elif ref_len == alt_len:
                allele['type'] = 'MNP'
                summary_json['variants_by_type']['mnp'] += 1
            elif ref_len == 0:
                allele['type'] = 'INS'
                summary_json['variants_by_type']['ins'] += 1
            elif alt_len == 0:
                allele['type'] = 'DEL'
                summary_json['variants_by_type']['del'] += 1
            else:
                allele['type'] = 'COMPLEX'
                summary_json['variants_by_type']['complex'] += 1

            if genotype1_int is None or genotype2_int is None:
                allele['call'] = 'No Call'
                summary_json['variants_by_call']['no_call'] += 1
            elif genotype1_int == 0 and genotype2_int == 0:	
                allele['call'] = 'Absent'
                summary_json['variants_by_call']['absent'] += 1			
            elif genotype1_int == (idx+1) and genotype2_int == (idx+1):
                allele['call'] = 'Homozygous'
                summary_json['variants_by_call']['homozygous'] += 1
            elif genotype1_int == (idx+1) or genotype2_int == (idx+1):
                allele['call'] = 'Heterozygous'
                summary_json['variants_by_call']['heterozygous'] += 1
            else:
                allele['call'] = 'Absent'
                summary_json['variants_by_call']['absent'] += 1
            
            if oid == '.':
                summary_json['variants_by_source']['novel'] += 1
            else:
                summary_json['variants_by_source']['hotspot'] += 1

            # Again note that in tagseq, MDP = FDP, MAO = FAO, MRO = FRO            
            if (options.library_type == "tagseq"):
                allele['read_cov']                  = '%d'      % (DP)
                allele['allele_read_cov']           = '%d'      % (AO[idx])
                allele['allele_read_freq']          = '%1.3f'   % (100.0 * AO[idx] / DP if (DP) > 0.0 else 0.0)
                allele['mol_coverage']              = '%d'      % (FDP)
                allele['allele_mol_cov']            = '%d'      % (FAO[idx])
                allele['allele_mol_freq']           = '%1.3f'   % (100.0 * FAO[idx] / FDP if FDP > 0.0 else 0.0)
                # tagseq doesn't have variant merging problem.
                allele['freq']                      = '%1.3f'   % (100.0 * FAO[idx] / FDP if FDP > 0.0 else 0.0)
            else:
                allele['cov_total']                 = '%d'      % (DP)
                allele['cov_total_downsampled']     = '%d'      % (FDP)
                allele['cov_total_plus']            = '%d'      % (FSRF + sum(FSAF))
                allele['cov_total_minus']           = '%d'      % (FSRR + sum(FSAR))
                allele['cov_allele']                = '%d'      % (FAO[idx])
                allele['cov_allele_plus']           = '%d'      % (FSAF[idx])
                allele['cov_allele_minus']          = '%d'      % (FSAR[idx])
                #allele['freq']                      = '%1.1f'   % (100.0 * FAO[idx] / FDP if FDP > 0.0 else 0.0)            
                allele['freq']                      = '%1.1f'   % (100.0 * AF[idx]) # TS-14592

            allele['LOD']                       = '%1.2f'   % (100.0 * float(LOD[idx]))                
            allele['strand_bias']               = '%1.4f'   % (STB[idx])
            allele['beta_bias']                 = '%1.4f'   % (SXB[idx])
            allele['sse_plus']                  = '%1.4f'   % (SSEP[idx])
            allele['sse_minus']                 = '%1.4f'   % (SSEN[idx])
            allele['sssb']                      = '%1.4f'   % (SSSB[idx])
            allele['rbi']                       = '%1.4f'   % (RBI[idx])
            allele['refb']                      = '%1.4f'   % (REFB[idx])
            allele['varb']                      = '%1.4f'   % (VARB[idx])
            allele['mlld']                      = '%1.4f'   % (MLLD[idx])
            allele['hp_length']                 = '%d'      % (HRUN[idx])

            if oid != '.':
                allele_prefix = 'Hotspot'
            elif allele['type'] == 'SNP':
                allele_prefix = 'SNP'
            else:
                allele_prefix = 'INDEL'

            allele['cov_total_filter']      = '-'
            allele['cov_total_plus_filter'] = '-'
            allele['cov_total_minus_filter'] = '-'
            allele['strand_bias_filter'] = '-'
            allele['beta_bias_filter'] = '-'
            allele['qual_filter'] = '-'
            allele['hp_length_filter'] = '-'
            allele['mlld_filter'] = '-'
            allele['varb_filter'] = '-'
            allele['rbi_filter'] = '-'
            allele['sse_plus_filter'] = '-'
            allele['sse_minus_filter'] = '-'
            allele['sssb_filter'] = '-'
            allele['call_filter'] = '-'

            if len(alt) == len(frs):
                fr = frs[idx]
            else:
                fr = FR
            # Filters:
            allele['cov_total_plus_filter'] = ('Minimum coverage on either strand ('+allele_prefix+')') if 'PosCov' in fr or 'NODATA' in fr else '-'
            allele['cov_total_minus_filter'] = ('Minimum coverage on either strand ('+allele_prefix+')')  if 'NegCov' in fr or 'NODATA' in fr else '-'
            allele['strand_bias_filter'] = ('Maximum strand bias ('+allele_prefix+')')  if 'STDBIAS' in fr else '-'
            allele['beta_bias_filter'] = '-'
            allele['qual_filter'] = ('Minimum quality ('+allele_prefix+')')  if 'QualityScore' in fr else '-'
            allele['hp_length_filter'] = 'Maximum homopolymer length' if ('HPLEN' in fr or 'HPINSLEN' in fr or 'HPDELLEN' in fr) else '-'
            allele['mlld_filter'] = 'Minimum relative read quality' if 'STRINGENCY' in fr else '-'
            allele['varb_filter'] = 'Maximum reference/variant signal shift' if ('PREDICTIONRefSHIFTx' in fr or 'PREDICTIONVar' in fr) else '-'
            allele['rbi_filter'] = 'Maximum common signal shift' if 'PREDICTIONSHIFT' in fr else '-'
            allele['sse_plus_filter'] = '-'
            allele['sse_minus_filter'] = '-'
            allele['sssb_filter'] = '-'
            if 'PositiveSSE' in fr:
                allele['sse_plus_filter'] = 'Context error on one strand'
                allele['sssb_filter'] = 'Context error on one strand'
            if 'NegaitveSSE' in fr:
                allele['sse_minus_filter'] = 'Context error on one strand'
                allele['sssb_filter'] = 'Context error on one strand'
            if 'PredictedSSE' in fr:
                allele['sse_plus_filter'] = 'Context error on both strands'
                allele['sse_minus_filter'] = 'Context error on both strands'

            if (options.library_type == "tagseq"):
                allele['mol_coverage_filter']      = ('Minimum coverage ('+allele_prefix+')') if 'MINCOV' in fr or 'NODATA' in fr else '-'                
                if 'VARCOV<' in fr:
                    allele['allele_mol_cov_filter'] = 'Minimum variant mol coverage'
                elif 'VARCOV-TGSM<' in fr:
                    allele['allele_mol_cov_filter'] = 'Tag-similar variant mol'
                else:
                    allele['allele_mol_cov_filter'] = '-'
                allele['allele_mol_freq_filter'] = 'Beyond limit of detection' if 'AF<' in fr else '-'
            else:
                allele['cov_total_filter']      = ('Minimum coverage ('+allele_prefix+')') if 'MINCOV' in fr or 'NODATA' in fr else '-'

            filter_list = ['-']
            for k,v in allele.iteritems():
                if k.endswith('_filter') and v not in filter_list:
                    filter_list.append(v)
            if 'REJECTION' in fr:
                filter_list.append('Excess outlier reads')
    
            allele['call_filter'] = ','.join(filter_list[1:]) if len(filter_list) > 1 else '-'

            # Filter stats:
            if allele_prefix == 'Hotspot':
                if 'MINCOV' in fr or 'NODATA' in fr:
                    summary_json['filters']['min_coverage']['filtered'] += 1
                else:
                    summary_json['filters']['min_coverage']['present'] += 1

                if 'PosCov' in fr or 'NegCov' in fr or 'NODATA' in fr:
                    summary_json['filters']['min_cov_each_strand']['filtered'] += 1
                else:
                    summary_json['filters']['min_cov_each_strand']['present'] += 1
        
                if 'STDBIAS' in fr:
                    summary_json['filters']['strand_bias']['filtered'] += 1
                else:
                    summary_json['filters']['strand_bias']['present'] += 1
        
                if 'XBIAS' in fr:
                    summary_json['filters']['beta_bias']['filtered'] += 1
                else:
                    summary_json['filters']['beta_bias']['present'] += 1
        
                if 'QualityScore' in fr:
                    summary_json['filters']['min_variant_score']['filtered'] += 1
                else:
                    summary_json['filters']['min_variant_score']['present'] += 1

                if 'HPLEN' in fr:
                    summary_json['filters']['hp_max_length']['filtered'] += 1
                else:
                    summary_json['filters']['hp_max_length']['present'] += 1
        
                if 'STRINGENCY' in fr:
                    summary_json['filters']['data_quality_stringency']['filtered'] += 1
                else:
                    summary_json['filters']['data_quality_stringency']['present'] += 1
        
                if 'PositiveSSE' in fr or 'NegativeSSE' in fr:
                    summary_json['filters']['sse_one_strand']['filtered'] += 1
                else:
                    summary_json['filters']['sse_one_strand']['present'] += 1

                if 'PredictedSSE' in fr:
                    summary_json['filters']['sse_both_strands']['filtered'] += 1
                else:
                    summary_json['filters']['sse_both_strands']['present'] += 1

                if 'PREDICTIONHypSHIFT' in fr:
                    summary_json['filters']['filter_x_predictions']['filtered'] += 1
                else:
                    summary_json['filters']['filter_x_predictions']['present'] += 1

                if 'PREDICTIONSHIFT' in fr:
                    summary_json['filters']['filter_unusual_predictions']['filtered'] += 1
                else:
                    summary_json['filters']['filter_unusual_predictions']['present'] += 1
        
                if 'REJECTION' in fr:
                    summary_json['filters']['rejection']['filtered'] += 1
                else:
                    summary_json['filters']['rejection']['present'] += 1
                
            if options.alleles2:
                write_alleles2_line(options, alleles2_xls, data=allele, report_ppa=is_report_ppa,report_subset=is_report_subset)
            
            if options.concatenated_xls:
                write_alleles2_line(options, concatenated_xls, data=allele, report_ppa=is_report_ppa,report_subset=is_report_subset)
        
            
        
            if chr not in observed_chr_order:
                observed_chr_order.append(chr)
                chr_calls_total[chr] = 0
                chr_calls_het_snp[chr] = 0
                chr_calls_hom_snp[chr] = 0
                chr_calls_het_indel[chr] = 0
                chr_calls_hom_indel[chr] = 0
                chr_calls_other[chr] = 0
                chr_calls_none[chr] = 0
            
            if allele['call'] == 'Heterozygous' and allele['type'] == 'SNP':
                chr_calls_total[chr] += 1
                chr_calls_het_snp[chr] += 1
                if oid != '.':
                    hotspot_total += 1
                    hotspot_het_snp += 1
            elif allele['call'] == 'Homozygous' and allele['type'] == 'SNP':
                chr_calls_total[chr] += 1
                chr_calls_hom_snp[chr] += 1
                if oid != '.':
                    hotspot_total += 1
                    hotspot_hom_snp += 1
            elif allele['call'] == 'Heterozygous' and allele['type'] in ['INS','DEL']:
                chr_calls_total[chr] += 1
                chr_calls_het_indel[chr] += 1
                if oid != '.':
                    hotspot_total += 1
                    hotspot_het_indel += 1
            elif allele['call'] == 'Homozygous' and allele['type'] in ['INS','DEL']:
                chr_calls_total[chr] += 1
                chr_calls_hom_indel[chr] += 1
                if oid != '.':
                    hotspot_total += 1
                    hotspot_hom_indel += 1
            elif allele['call'] == 'Absent' or allele['call'] == 'No Call':
                chr_calls_none[chr] += 1
                if oid != '.':
                    hotspot_none += 1
            else:
                chr_calls_total[chr] += 1
                chr_calls_other[chr] += 1
                if oid != '.':
                    hotspot_total += 1
                    hotspot_other += 1

    input_vcf.close()
    output_xls.close()
    if options.alleles2:
        alleles2_xls.close()

    if options.concatenated_xls:
        concatenated_xls.close()

    for chr in observed_chr_order:
        summary_json['variants_total']['variants']      += chr_calls_total[chr]
        summary_json['variants_total']['het_snps']      += chr_calls_het_snp[chr]
        summary_json['variants_total']['homo_snps']     += chr_calls_hom_snp[chr]
        summary_json['variants_total']['het_indels']    += chr_calls_het_indel[chr]
        summary_json['variants_total']['homo_indels']   += chr_calls_hom_indel[chr]
        summary_json['variants_total']['other']         += chr_calls_other[chr]
        summary_json['variants_total']['no_call']       += chr_calls_none[chr]
        summary_json['variants_by_chromosome'].append({
            'chromosome'    : chr,
            'variants'      : chr_calls_total[chr],
            'het_snps'      : chr_calls_het_snp[chr],
            'homo_snps'     : chr_calls_hom_snp[chr],
            'het_indels'    : chr_calls_het_indel[chr],
            'homo_indels'   : chr_calls_hom_indel[chr],
            'other'         : chr_calls_other[chr],
            'no_call'       : chr_calls_none[chr]
        })

    if options.hotspot:
        summary_json['hotspots_total'] = {
            'variants'      : hotspot_total,
            'het_snps'      : hotspot_het_snp,
            'homo_snps'     : hotspot_hom_snp,
            'het_indels'    : hotspot_het_indel,
            'homo_indels'   : hotspot_hom_indel,
            'other'         : hotspot_other,
            'no_call'       : hotspot_none
        }

    if options.summary:
        summary_file = open(options.summary,'w')
        json.dump(summary_json,summary_file)
        summary_file.close()
        
        if options.scatter_png and options.alleles:
            try:
                fig = plt.figure(figsize=(10,5),dpi=100)
                ax = fig.add_subplot(111,frame_on=True)
                
                ax.plot(hom_freq,hom_cov,'.',markerfacecolor="red",markersize=10,markeredgewidth=0,label='Present (Hom)')
                ax.plot(het_freq,het_cov,'.',markerfacecolor="green",markersize=10,markeredgewidth=0,label='Present (Het)')
                ax.plot(abs_freq,abs_cov,'.',markerfacecolor="blue",markersize=10,markeredgewidth=0,label='Absent')
                ax.plot(nc_freq,nc_cov,'x', markeredgecolor="black",markersize=4,markeredgewidth=1.5,label='No Call')
                
                max_cov = max(max(hom_cov+[0]),max(het_cov+[0]),max(abs_cov+[0]),max(nc_cov+[0]))
                
                ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
                ax.set_xticklabels(['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
            
                ax.set_xlim(-5,105)
                ax.set_ylim(-0.05*max_cov,1.05*max_cov)
                ax.set_xlabel("Allele Frequency")
                ax.set_ylabel("Total Coverage")
                ax.xaxis.grid(color='0.8')
                ax.yaxis.grid(color='0.8')
                ax.set_axisbelow(True)
                ax.legend(loc=1,prop={'size':'small'},numpoints=1,title='Allele Calls')
                #fig.patch.set_alpha(0.0)
                fig.savefig(options.scatter_png)
        
            except:
                print 'Unable to generate plot %s' % options.scatter_png
                traceback.print_exc()


if __name__ == '__main__':
    sys.exit(main())


