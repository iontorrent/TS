# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:06:01 2017

@author: ckoller

Filters the lines in a VCF file based on information in the target file lines
Both target file as well as VCF file are assuemd to be sorted

Filter:
- VCF lines that are outside any target region
- De-novo alleles in targets that are marked as HOTSPOT_ONLY in the info field

Overlapping target areas:
The properties of the leftmost target area in the overlap area are applied.

"""

import time
from ion.plugin.bedParser import *
# In lieu of a vcf library do a hack-job for quick turn around time

def filterVCFbyTarget(target_file, in_vcf_name, out_vcf_name):
    
    start_time = time.time()
    try:
        vcf_in  = open(in_vcf_name, 'r')
    except:
        print("FilterByTarget: Unable to read file %s" % in_vcf_name)
        return False
        
    try:
        vcf_out = open(out_vcf_name, 'w')
    except:
        print("FilterByTarget: Unable to write to file %s" % in_vcf_name)
        vcf_in.close()
        return False
    
    try:
        targets = open(target_file, 'r')        
        target_reader = IonBedReader(targets)
    except:
        print("FilterByTarget: Unable to read file %s" % in_vcf_name)
        vcf_out.close()
        vcf_in.close()
        return False
    
    # Accounting of filtering
    num_records_read = 0
    num_records_written = 0
    num_outside_target = 0
    num_denovo_filtered = 0
    
    mytarget = target_reader.readline()
    
    for line in vcf_in:
        
        # Transfer header and comment lines to output file
        if line.startswith('#'):
            vcf_out.write(line)
            continue
    
        #for record in vcf_reader:
        num_records_read += 1
        
        # Assume tab separated vcf format of
        #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT
        vcf_record = line.rstrip().split('\t')
        info_field = target_reader._parseInfoField(vcf_record[7])
        
        # VCF is a 1-based index
        # BED is a 0-based index, open ended interval
        in_target = target_reader.InTarget(mytarget, vcf_record[0], (int(vcf_record[1])-1))
        while mytarget and in_target>0:
            mytarget = target_reader.readline()
            in_target = target_reader.InTarget(mytarget, vcf_record[0], (int(vcf_record[1])-1))
            
        if mytarget and in_target==0:
            if 'HOTSPOTS_ONLY' in mytarget['info']:
                if 'HS' in info_field:
                    vcf_out.write(line)
                    num_records_written += 1
                else:
                    num_denovo_filtered += 1
            else:
                vcf_out.write(line)
                num_records_written += 1
        else:
            num_outside_target += 1
    
    targets.close()
    vcf_out.close()
    vcf_in.close()
    
    # And finally print a summary:
    print ('FilterByTarget: Wrote %d out of %d vcf records to %s. NumDeNovoFiltered=%d, NumOffTarget=%d Duration=%d' % \
        (num_records_written, num_records_read, out_vcf_name, num_denovo_filtered, num_outside_target, (time.time()-start_time)) )
    return  True
    

# ===============================================================

    
    
