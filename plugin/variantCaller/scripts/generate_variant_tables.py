#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import math
import bisect
import json
import traceback
from collections import defaultdict
from optparse import OptionParser

from matplotlib import use
use("Agg",warn=False)
import matplotlib.pyplot as plt


def main():
    
    parser = OptionParser()
    parser.add_option('-i', '--input-vcf',      help='Input vcf file to be sorted', dest='input') 
    parser.add_option('-r', '--region-bed',     help='Region bed file (optional)', dest='region') 
    parser.add_option('-s', '--hotspot-bed',    help='Hotspot bed file', dest='hotspot') 
    parser.add_option('-o', '--output-xls',     help='Variant table tab-delimited file', dest='output')
    parser.add_option('-a', '--alleles-xls',    help='Alleles table tab-delimited file', dest='alleles')
    parser.add_option('-c', '--chromosome-png', help='Bar plot of # variants per chromosome', dest='chrom_png')
    parser.add_option('-S', '--scatter-png',    help='Scatterplot of coverage vs. frequency for variants', dest='scatter_png')
    parser.add_option('-j', '--summary-json',   help='Variant summary in json file', dest='summary')
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
        for line in region_bed_file:
            if not line or line.startswith('track '):
                continue
            fields = line.split('\t')
            if len(fields) < 6:
                continue
            chrom = fields[0].strip()
            region_start.setdefault(chrom,[]).append(int(fields[1]) + 1)
            region_end.setdefault(chrom,[]).append(int(fields[2]))
            region_ids.setdefault(chrom,[]).append((fields[3].strip(), fields[-1].strip()))
        region_bed_file.close()
    
    
    #
    # Step 3: Convert and annotate vcf
    #
    
    input_vcf = open(options.input,'r')
    output_xls = open(options.output,'w')
    output_xls.write("Chrom\tPosition\tGene Sym\tTarget ID\tType\tZygosity\tGenotype\tRef\tVariant\tVar Freq\tP-value\tCoverage\tRef Cov\tVar Cov")
    if options.hotspot:
        output_xls.write("\tHotSpot ID")
    output_xls.write("\n")

    if options.alleles:
        alleles_xls = open(options.alleles,'w')
        alleles_xls.write('Chrom\tPosition\tAllele ID\tAllele Source\tRef\tVariant\tType\tAllele Call\tCall Details\tFreq\tAllele Cov\tDownsampled Cov\tTotal Cov\tVCF Record\n')

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
    
    sample_name = ''
    
    for line in input_vcf:
        if not line:
            continue

        if line.startswith('#CHROM'):
            fields = line.split('\t')
            if len(fields) > 9:
                sample_name = fields[9].strip()
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
            subitem = item.split('=')
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


        try:
            ref_cov = int(format['RO'])
        except:
            ref_cov = 0

        try:
            ref_cov2 = int(format['FRO'])
        except:
            ref_cov2 = ref_cov

        var_cov = [0] * len(alt)
        for alt_idx,var_cov_txt in enumerate(format.get('AO','').split(',')):
            try:
                var_cov[alt_idx] = int(var_cov_txt)
            except:
                pass
        var_cov2 = var_cov
        for alt_idx,var_cov_txt in enumerate(format.get('FAO','').split(',')):
            try:
                var_cov2[alt_idx] = int(var_cov_txt)
            except:
                pass

        total_cov = ref_cov + sum(var_cov)
        total_cov2 = ref_cov2 + sum(var_cov2)
        
        var_freq = []
        for v in var_cov:
            if total_cov > 0:
                var_freq.append(float(v)/float(total_cov)*100.0)
            else:
                var_freq.append(0.0)
        var_freq2 = []
        for v in var_cov2:
            if total_cov2 > 0:
                var_freq2.append(float(v)/float(total_cov2)*100.0)
            else:
                var_freq2.append(0.0)
        
        output_xls.write("%s\t%s\t" % (chr,pos)) # Chrom, Position
        output_xls.write("%s\t%s\t" % (gene_name,region_id)) # Gene Sym, Target ID
        output_xls.write("%s\t%s\t%s\t" % (variant_type,ploidy,genotype_actual)) # Type, Zygosity
        output_xls.write("%s\t%s\t" % (fields[3],fields[4])) # Ref, Variant
        output_xls.write("%s\t%s\t%s\t%s\t%s" % (sum(var_freq2),qual,total_cov2,ref_cov2,sum(var_cov2)))
        
        is_hotspot = 0
        if options.hotspot:
            hotspot_annotation = id
            if hotspot_annotation != '.':
                output_xls.write("\t"+hotspot_annotation)
                is_hotspot = 1
            else:
                output_xls.write("\t---")
        
        output_xls.write("\n")

        if options.alleles:
                        
            all_oid     = info['OID'].split(',')
            all_opos    = info['OPOS'].split(',')
            all_oref    = info['OREF'].split(',')
            all_oalt    = info['OALT'].split(',')
            all_omapalt = info['OMAPALT'].split(',')
            filtering_reason = info.get('FR','.').lstrip('.,')
            if not filtering_reason:
                filtering_reason = '---'
        
            for oid,opos,oref,oalt,omapalt in zip(all_oid,all_opos,all_oref,all_oalt,all_omapalt):
                if omapalt not in alt:
                    continue
                idx = alt.index(omapalt)
                
                xref = oref.strip('-')
                xalt = oalt.strip('-')
                
                alleles_xls.write('%s\t%s\t' % (chr,opos))
                if oid == '.':
                    alleles_xls.write('---\tNovel\t')
                else:
                    alleles_xls.write('%s\tHotspot\t' % oid)
                alleles_xls.write('%s\t%s\t' % (oref,oalt))
                if len(xref) == 1 and len(xalt) == 1:
                    alleles_xls.write('SNP\t')
                elif len(xref) == len(xalt):
                    alleles_xls.write('MNP\t')
                elif len(xref) == 0:
                    alleles_xls.write('INS\t')
                elif len(xalt) == 0:
                    alleles_xls.write('DEL\t')
                else:
                    alleles_xls.write('COMPLEX\t')

                if genotype1_int is None or genotype2_int is None:
                    alleles_xls.write('No Call\t%s\t' % filtering_reason)
                    nc_freq.append(var_freq2[idx])
                    nc_cov.append(total_cov)
                elif genotype1_int == (idx+1) and genotype2_int == (idx+1):
                    alleles_xls.write('Present\tHomozygous\t')
                    hom_freq.append(var_freq2[idx])
                    hom_cov.append(total_cov)
                elif genotype1_int == (idx+1) or genotype2_int == (idx+1):
                    alleles_xls.write('Present\tHeterozygous\t')
                    het_freq.append(var_freq2[idx])
                    het_cov.append(total_cov)
                else:
                    alleles_xls.write('Absent\t---\t')
                    abs_freq.append(var_freq2[idx])
                    abs_cov.append(total_cov)
                
                alleles_xls.write('%1.1f%%\t%d\t%d\t%d\t' % (var_freq2[idx],var_cov2[idx],total_cov2,total_cov))
                alleles_xls.write('%s:%s\n' % (chr,pos))
                
        
        if chr not in observed_chr_order:
            observed_chr_order.append(chr)
            chr_calls_total[chr] = 0
            chr_calls_het_snp[chr] = 0
            chr_calls_hom_snp[chr] = 0
            chr_calls_het_indel[chr] = 0
            chr_calls_hom_indel[chr] = 0
            chr_calls_other[chr] = 0
            chr_calls_none[chr] = 0
        
        if ploidy == 'Het' and variant_type == 'SNP':
            chr_calls_total[chr] += 1
            hotspot_total += is_hotspot
            chr_calls_het_snp[chr] += 1
            hotspot_het_snp += is_hotspot
        elif ploidy == 'Hom' and variant_type == 'SNP':
            chr_calls_total[chr] += 1
            hotspot_total += is_hotspot
            chr_calls_hom_snp[chr] += 1
            hotspot_hom_snp += is_hotspot
        elif ploidy == 'Het' and variant_type in ['INS','DEL']:
            chr_calls_total[chr] += 1
            hotspot_total += is_hotspot
            chr_calls_het_indel[chr] += 1
            hotspot_het_indel += is_hotspot
        elif ploidy == 'Hom' and variant_type in ['INS','DEL']:
            chr_calls_total[chr] += 1
            hotspot_total += is_hotspot
            chr_calls_hom_indel[chr] += 1
            hotspot_hom_indel += is_hotspot
        elif ploidy == 'Ref' or ploidy == 'NC':
            chr_calls_none[chr] += 1
            hotspot_none += is_hotspot
        else:
            chr_calls_total[chr] += 1
            hotspot_total += is_hotspot
            chr_calls_other[chr] += 1
            hotspot_other += is_hotspot
            
    input_vcf.close()
    output_xls.close()
    if options.alleles:
        alleles_xls.close()


    if options.summary:
        summary_json = {
            'sample_name' : sample_name,
            'variants_total' : {
                'variants' : 0,
                'het_snps' : 0,
                'homo_snps' : 0,
                'het_indels' : 0,
                'homo_indels' : 0,
                'other' : 0,
                'no_call' : 0
            },
            'variants_by_chromosome' : []
        }
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
        summary_file = open(options.summary,'w')
        json.dump(summary_json,summary_file)
        summary_file.close()

        if options.chrom_png:
            try:
                fig = plt.figure(figsize=(12,3.5),dpi=100)
                ax = fig.add_subplot(111,frame_on=False,position=[0.1,0.22,0.85,0.72])

                plot_chr = []
                plot_calls = []
                for idx,chr in enumerate(observed_chr_order):
                    if chr_calls_total[chr] > 0:
                        plot_chr.append(chr)
                        plot_calls.append(chr_calls_total[chr])

                plot_range = range(len(plot_chr))
                
                ax.bar(plot_range,plot_calls,color="#2D4782",linewidth=0,align='center')
                ax.set_xticks(plot_range)
                if len(plot_chr) < 6:
                    ax.set_xticklabels(plot_chr)
                else:
                    ax.set_xticklabels(plot_chr,rotation=45)
            
                ax.set_xlim(-0.9,len(plot_chr)-1.0+0.9)
                #ax.set_ylim(0,1.1*alt_max)
                ax.set_xlabel("Number of variant calls by chromosome")
                fig.patch.set_alpha(0.0)
                fig.savefig(options.chrom_png)
        
            except:
                print 'Unable to generate plot %s' % options.chrom_png
                traceback.print_exc()
        
        
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


