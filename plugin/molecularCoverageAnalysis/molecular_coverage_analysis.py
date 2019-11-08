#!/usr/bin/python
# Copyright (C) 2019 Thermo Fisher Scientific, Inc. All Rights Reserved.
import pysam
import os
import sys
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        
import json
import string
import time

try:
    import lod
except ImportError:
    plugin_dir = os.path.dirname(os.path.realpath(__file__))
    try_lod_dir_list = [
        os.path.join(plugin_dir, 'bin'), # Inside the plugin
        '/results/plugins/variantCaller/bin', # TS default VC plugin bin
        os.path.join(os.path.dirname(plugin_dir), 'variantCaller/bin') # Assume plugin repo
    ]
    for try_lod_dir in try_lod_dir_list:
        try_lod_path = os.path.join(try_lod_dir, 'lod.py')
        if os.path.exists(try_lod_path):
            sys.path.append(try_lod_dir)
            break
    import lod
from optparse import OptionParser
from multiprocessing.pool import Pool as ThreadPool


#--------------------------------------------------------------------------------------------------------
# printtime
#--------------------------------------------------------------------------------------------------------
def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%a %Y-%m-%d %X %Z') + " ] " + message
    sys.stdout.flush()
    sys.stderr.flush()
    
#--------------------------------------------------------------------------------------------------------
# bed_line_to_basic_dict
#--------------------------------------------------------------------------------------------------------
def bed_line_to_basic_dict(bed_line):
    split_bed = bed_line.strip('\n').split('\t')
    info_text = split_bed[-1] if '=' in split_bed[-1] else '.'
    bed_dict = {
        'chrom': split_bed[0], 
        'chromStart': int(split_bed[1]), 
        'chromEnd': int(split_bed[2]), 
        'name': split_bed[3], 
        'region_len': int(split_bed[2]) - int(split_bed[1]),
        'info': info_text,
        'info_dict': {},
    }
    if info_text == '.':
        return bed_dict
    
    # Parse the info field
    for split_info in info_text.split(';'):
        split_by_equality = split_info.split('=')
        my_value = split_by_equality[1] if len(split_by_equality) > 1 else None
        try:
            my_value = float(my_value)
        except ValueError:
            pass
        bed_dict['info_dict'][split_by_equality[0]] = my_value
    
    return bed_dict

#--------------------------------------------------------------------------------------------------------
# Rev Comp
#--------------------------------------------------------------------------------------------------------
comp_trans = string.maketrans('ACGTNacgtn', 'TGCANtgcan')
def rev_comp(seq):
    # The fastest rev_comp that I have in python
    return seq.translate(comp_trans)[::-1]

#--------------------------------------------------------------------------------------------------------
# Handle the methods in the old pysam
#--------------------------------------------------------------------------------------------------------
# Could be slow
def has_tag_by_try(bam_read, key):
    try:
        bam_read.opt(key)
        return True
    except KeyError:
        pass
    return False

#--------------------------------------------------------------------------------------------------------
# Family Manager
#--------------------------------------------------------------------------------------------------------
class FamilyManager:
    def __init__(self, bam_path):
        # Open bam file
        self.__f_bam = pysam.AlignmentFile(bam_path, 'rb') if hasattr(pysam, 'AlignmentFile') else pysam.Samfile(bam_path, 'rb')

        # Handle the zt, yt in the BAM header.
        self.umt_structure = {}
        for tag_key in ('zt', 'yt'):
            try:
                self.umt_structure[tag_key] = self.__f_bam.header['RG'][0][tag_key]
            except KeyError:
                KeyError('No %s tag is detected in the RG of the BAM header.'%tag_key)
        
            # Uniqueness of ZT, YT?
            if len(set(map(lambda rg: rg[tag_key], self.__f_bam.header['RG']))) > 1:
                ValueError('Multiple values of %s being detected in the BAM header. Currently not supported.'%tag_key)

        # Error out multisample BAM
        if len(set(map(lambda rg: rg.get('SM', ''), self.__f_bam.header['RG']))) > 1:
            ValueError('Multi-sample being detected in the BAM header. Currently not supported.'%tag_key)

        # LOD manager
        self.__lod_manager = lod.LodManager()
        self.__lod_manager.do_smoothing(True)

        # Ignore ZR?
        self.set_ignore_zr(False)
        
        # Initialize tvc parameters
        self.set_tvc_param(None)
        
        # Handle the deprecated pysam.AlignedRead attributes
        self.__get_overlap = (lambda bam_read, pos_start, pos_end : (bam_read.get_overlap(pos_start, pos_end))) if hasattr(pysam.AlignedRead, 'get_overlap') else (lambda bam_read, pos_start, pos_end : (bam_read.overlap(pos_start, pos_end)))
        self.__get_tag = (lambda bam_read, key : (bam_read.get_key(key))) if hasattr(pysam.AlignedRead, 'get_key') else (lambda bam_read, key : (bam_read.opt(key)))
        self.__reference_start = (lambda bam_read : (bam_read.reference_start)) if hasattr(pysam.AlignedRead, 'reference_start') else (lambda bam_read : (bam_read.pos))
        self.__reference_end = (lambda bam_read : (bam_read.reference_end)) if hasattr(pysam.AlignedRead, 'reference_end') else (lambda bam_read : (bam_read.aend))
        self.__has_tag = (lambda bam_read, key : (bam_read.has_tag(key))) if hasattr(pysam.AlignedRead, 'has_tag') else has_tag_by_try
        self.__mapping_quality = (lambda bam_read : (bam_read.mapping_quality)) if hasattr(pysam.AlignedRead, 'mapping_quality') else (lambda bam_read : (bam_read.mapq))
        self.__cigartuples = (lambda bam_read : (bam_read.cigartuples)) if hasattr(pysam.AlignedRead, 'cigartuples') else (lambda bam_read : (bam_read.cigar))
        self.__query_alignment_sequence = (lambda bam_read : (bam_read.query_alignment_sequence)) if hasattr(pysam.AlignedRead, 'query_alignment_sequence') else (lambda bam_read : (bam_read.query))
        self.__query_name = (lambda bam_read : (bam_read.query_name)) if hasattr(pysam.AlignedRead, 'query_name') else (lambda bam_read : (bam_read.qname))
    
    def set_ignore_zr(self, flag):
        self.__ignore_zr = True if flag else False
        return self.__ignore_zr
    
    def __enter__(self):
        return self

    def close(self):
        self.__f_bam.close()
        return 0

    def __exit__(self, type, msg, traceback):
        return self.close()

    def get_tvc_param(self, key):
        return self.__tvc_param_dict[key]['value']
        
    def set_tvc_param(self, tvc_param_dict=None):
        if tvc_param_dict is None:
            # Default param dict
            self.__tvc_param_dict = {
                'tag_trim_method': {
                    'value': 'sloppy-trim',
                    'type': str,
                    'section': 'torrent_variant_caller'
                },
                'min_tag_fam_size': {
                    'value': 3,
                    'type': int,
                    'section': 'torrent_variant_caller'
                },
                'min_fam_per_strand_cov': {
                    'value': 0,
                    'type': int,
                    'section': 'torrent_variant_caller'
                },
                'hotspot_min_allele_freq': {
                    'value': 0.0005,
                    'type': float,
                    'section': 'torrent_variant_caller'
                },
                'hotspot_min_variant_score': {
                    'value': 3.0,
                    'type': float,
                    'section': 'torrent_variant_caller'
                },                
                'hotspot_min_var_coverage': {
                    'value': 3,
                    'type': int,
                    'section': 'torrent_variant_caller'
                },
                'min_callable_prob': {
                    'value': 0.98,
                    'type': float,
                    'section': 'torrent_variant_caller'
                },
                'min_cov_fraction': {
                    'value': 0.0,
                    'type': float,
                    'section': 'freebayes'
                },                                     
                'read_snp_limit':{
                    'value': 10,
                    'type': int,
                    'section': 'freebayes'                                  
                },
                'min_mapping_qv':{
                    'value': 4,
                    'type': int,
                    'section': 'freebayes'                                  
                },
                'read_mismatch_limit':{
                    'value': 0,
                    'type': int,
                    'section': 'freebayes'                                  
                },
                'read_max_mismatch_fraction':{
                    'value': 1.0,
                    'type': float,
                    'section': 'freebayes'                                  
                },            
            }
        else:
            for key, one_param_dict  in self.__tvc_param_dict.iteritems():
                value = tvc_param_dict.get(one_param_dict['section'], {}).get(key, None)
                if value is not None:
                    one_param_dict['value'] = one_param_dict['type'](value)
        
        self.__require_strict_umt = self.__tvc_param_dict['tag_trim_method']['value'].lower().startswith('strict')
        # Set lod parameters as well
        self.__lod_manager.set_parameters(dict([(key.replace('hotspot_', ''), my_value['value']) for key, my_value in self.__tvc_param_dict.iteritems()]))

    def is_read_filtered_out(self, bam_read):
        """
        Apply exactly the same read filtering rule as in TVC
        """
        # Don't use secondary or duplicated reads
        if bam_read.is_secondary or bam_read.is_duplicate:
            return True
        
        # Filter on mapq        
        if self.__mapping_quality(bam_read) < self.__tvc_param_dict['min_mapping_qv']['value']:
            return True
        
        # Get NM (Number of Mismatches)
        try:
            nm = self.__get_tag(bam_read, 'NM')
        except KeyError:
            # Shouldn't happen
            nm = 0

        # No mismatch, the rest of the filters all pass 
        if nm == 0:
            return False
        
        # Filtering on read_max_mismatch_fraction (usually not used)
        if self.__tvc_param_dict['read_max_mismatch_fraction']['value'] < 1.0:
            qlen = len(self.__query_alignment_sequence(bam_read)) 
            if nm > qlen * self.__tvc_param_dict['read_max_mismatch_fraction']['value']:
                return True
        
        # Now deal with read_snp_limit and read_mismatch_limit
        read_snp_limit = self.__tvc_param_dict['read_snp_limit']['value']
        read_mismatch_limit = self.__tvc_param_dict['read_mismatch_limit']['value']

        # Usually a read shall pass
        # Note that read_mismatch_limit = 0 => disable the filter.
        if nm <= min(read_mismatch_limit, read_snp_limit) if read_mismatch_limit > 0 else read_snp_limit:
            return False
        
        # Calculate number of SNPs and modified_nm (where one gap open counts 1)
        num_snps = nm
        modified_nm = nm
        for cigar_pair in self.__cigartuples(bam_read):
            # Is it INDEL?
            if cigar_pair[0] in (1, 2):
                num_snps -= cigar_pair[1]
                modified_nm -= (cigar_pair[1] - 1) # One gap counts one
        # Sanity check
        if not (0 <= num_snps <= modified_nm):
            raise(ValueError('Read "%s": Inconsistent Cigar="%s" and NM=%d .'%(self.__query_name(bam_read), bam_read.cigarstring, nm)))
        
        # Filtering on read_snp_limit
        if num_snps > read_snp_limit:
            return True
        
        # Filtering on read_mismatch_limit
        if modified_nm > read_mismatch_limit and read_mismatch_limit > 0:
            return True        
        
        # The read passes all filters.
        return False

    def get_fam_key(self, bam_read, region_list=None, search_around_idx=None):
        try:
            yt = self.__get_tag(bam_read, 'YT')            
            zt = self.__get_tag(bam_read, 'ZT')
        except KeyError:
            return None
        
        # Apply read filters as TVC does
        # Note that the target overrode "read_mismatch_limit" is not applied here, since TVC applies it for consensus BAM.
        if self.is_read_filtered_out(bam_read):
            return None
    
        # self.__has_tag(bam_read, 'ZK') could be slow (in try - except) in TagSeq BAM with old pysam...
        if self.__has_tag(bam_read, 'ZK'):
            fam_strand_key = 'B'
            if bam_read.is_reverse:        
                zt, yt = rev_comp(yt), rev_comp(zt)
        else:
            fam_strand_key = 'R' if bam_read.is_reverse else 'F'
    
        if (region_list is None) or (search_around_idx is None):
            return '%s+%s+%s' %(fam_strand_key, zt, yt)
        
        # Now handle super amplicon, as TVC does.
        #@TODO (?)
        # The breaking condition is not ideal if you have highly overlapping regions with small length and large length.
        # The right way to do it is to be guided by the merged region BED.
        # I didn't implement this way by considering the complexity of plugin, since the case is rare anyway.
        covered_target_idx_list = []
        # Skip checking search_around_idx since it should be covered.
        for target_idx in xrange(search_around_idx - 1, -1, -1):
            # Trivial case first
            if self.__reference_start(bam_read) >= region_list[target_idx]['chromEnd']:
                break
            elif self.is_bam_read_cover_the_region(bam_read, region_list[target_idx]):
                covered_target_idx_list.append(target_idx)
            else:
                 break        
        for target_idx in xrange(search_around_idx + 1, len(region_list)):
            # Trivial case first
            if self.__reference_end(bam_read) <= region_list[target_idx]['chromStart']:
                break
            if self.is_bam_read_cover_the_region(bam_read, region_list[target_idx]):
                covered_target_idx_list.append(target_idx)
            else:
                break
        
        # Most of the reads are not super amplicon. Don't bother sorting an empty list.
        if not covered_target_idx_list:
            return '%s+%s+%s' %(fam_strand_key, zt, yt)
        
        covered_target_idx_list.sort()
        
        # The covered regions are also part of the key for family identification, as TVC does. 
        return '%s+%s+%s+%s' %(fam_strand_key, zt, yt, ','.join(map(str, covered_target_idx_list)))

    def is_strict_tag(self, umt, tag_key):
        my_tag_structure = self.umt_structure[tag_key]
        # Sanity Check
        if len(umt) != len(my_tag_structure):
            raise ValueError('The length of the UMT %s does not match the length of the tag structure %s.'%(umt, self.umt_structure[tag_key]))
        return False if [False for tag_pair in zip(umt, my_tag_structure) if tag_pair[1] not in 'N%s' %tag_pair[0]] else True

    def is_func_fam(self, fam_key, fam_dict, ignore_min_fam_per_strand_cov=False):
        # Check min_tag_fam_size
        if fam_dict['all'] < self.__tvc_param_dict['min_tag_fam_size']['value']:
            return False
        
        # Check min_fam_per_strand_cov for Bi-dir UMT
        if fam_key.startswith('B+') and (not ignore_min_fam_per_strand_cov):
            if min(fam_dict['fwd'], fam_dict['rev']) < self.__tvc_param_dict['min_fam_per_strand_cov']['value']:
                return False

        # Finally check strictness if needed
        if self.__require_strict_umt:
            split_fam_key = fam_key.split('+')
            if not (self.is_strict_tag(split_fam_key[1], 'zt') and self.is_strict_tag(split_fam_key[2], 'yt')):
                return False
        
        # All pass. I am a functional family.
        return True


    def is_bam_read_cover_the_region(self, bam_read, region_dict):
        target_overlap = int(self.__get_overlap(bam_read, region_dict['chromStart'], region_dict['chromEnd']))
        return (target_overlap >= self.__tvc_param_dict['min_cov_fraction']['value'] * region_dict['region_len']) and target_overlap    

    def gen_fam_dict_one_region(self, region_dict, region_list=None, search_around_idx=None):
        """
        region_list=None, search_around_idx=None are for super amplicons.
        """
        all_fam_dict = {'B': {}, 'R': {}, 'F': {}, 'miss_tag': {'fwd': 0, 'rev': 0}}
        for bam_read in self.__f_bam.fetch(region_dict['chrom'], region_dict['chromStart'], region_dict['chromEnd']):
            # Must cover a certain portion of the region
            if not self.is_bam_read_cover_the_region(bam_read, region_dict):
                continue            

            # Generate fam_key
            fam_key = self.get_fam_key(bam_read, region_list, search_around_idx)

            # Get ZR for consensus BAM
            zr = 1
            # self.__get_tag(bam_read, 'ZR') could be slow (in catching exception) in rawlib.bam...
            # Therefore please ignore ZR if not using consensus BAM.
            if not self.__ignore_zr:
                try:
                    zr = self.__get_tag(bam_read, 'ZR')
                except KeyError:
                    pass

            # Add the read to the family
            # It is very important to use setdefault. Otherwise, try or check the existance the key could be slow.  
            fam_dict = all_fam_dict[fam_key[0]].setdefault(fam_key, {'fwd': 0, 'rev': 0}) if (fam_key is not None) else all_fam_dict['miss_tag']
            fam_dict['rev' if bam_read.is_reverse else 'fwd'] += zr

        # Following the same logic in TVC: I use the "type" (i.e., B, F, R) of the UMT that has the highest family coverage. Other "types" are thrown to "miss_tag"
        # I.e., I shall get all "B" families in ASHD, and I should get either R or F in TagSeq.
        key_order = ('B', 'F', 'R')
        fam_num_by_type = [len(all_fam_dict[key]) for key in key_order]
        # Ugly code (of using index) but it is okay since I only have 3 entries.
        # If no read at all, it will use "B". 
        arg_max_fam_depth_by_type = fam_num_by_type.index(max(fam_num_by_type))
        for idx, key in enumerate(key_order):
            if idx == arg_max_fam_depth_by_type:
                for fam_dict in all_fam_dict[key].itervalues():
                    fam_dict['all'] = fam_dict['fwd'] + fam_dict['rev']
            else:
                # Throw away those minor types to miss_tag
                for fam_dict in all_fam_dict.pop(key).itervalues():
                    all_fam_dict['miss_tag']['fwd'] += fam_dict['fwd']
                    all_fam_dict['miss_tag']['rev'] += fam_dict['rev']
                
        all_fam_dict['miss_tag']['all'] = all_fam_dict['miss_tag']['fwd'] + all_fam_dict['miss_tag']['rev']
        return all_fam_dict

    def reads_group_by_fam_size(self,fam_size): # roughly group reads based on fam size
	
        if fam_size >=1 and fam_size <=2 : return 'small' #  read depth is not enough if this group is high
	if fam_size >=3 and fam_size <=30 : return 'median' #  read depth is good if this group is high
	return 'large' #  read depth is over if this group is high, and suggest NO more real families can be found with higher read depth

    def gen_stats_one_region(self, all_fam_dict, region_dict=None):
        my_stats_dict = {
            'all_fam_size_hist': dict([(key, {0: 0}) for key in ('fwd_rev', 'fwd_only', 'rev_only', 'all')]),
            'strict_fam_size_hist': dict([(key, {1: 0}) for key in ('fwd_rev', 'fwd_only', 'rev_only', 'all')]),
            'raw_read_cov': dict([(key, 0) for key in ('fwd', 'rev', 'all')]),
            'func_fam_size_hist': {self.__tvc_param_dict['min_tag_fam_size']['value']: 0},
            'fwd_rev_fam_cov': 0,
	    'fwd_only_fam_cov':0,
	    'rev_only_fam_cov':0,
            'strict_func_fam_cov': 0,
            'strict_func_umt_rate': 0.0,
	    'average_fam_size' : 0.0,
            'strict_func_fam_size_hist': {self.__tvc_param_dict['min_tag_fam_size']['value']: 0},
            'strict_func_read_cov': {'fwd': 0, 'rev': 0, 'all': 0},
            'func_read_cov': {'fwd': 0, 'rev': 0, 'all': 0},
            'fam_read_cov': {'small': 0, 'median': 0, 'large': 0},
            'umt_strand_type': None,
            'lod': None
        }       
        
        # I don't allow mixing UMT strand type
        # all_fam_dict should only have two keys: "miss_tag" and the one I want.
        umt_strand_type_key = all_fam_dict.keys()
        umt_strand_type_key.remove('miss_tag')
        # Check point: I shall get 'B' or "R" or "F"
        assert(umt_strand_type_key in map(list, 'BRF'))
        my_stats_dict['umt_strand_type'] = umt_strand_type_key[0]
        
        fam_cov_no_strand_constraint = 0
        for fam_key, fam_dict in all_fam_dict[my_stats_dict['umt_strand_type']].iteritems():            
            my_fam_size = fam_dict['all']
            assert(my_fam_size > 0)
            my_fam_healthiness_key = 'fwd_rev'
            if fam_dict['fwd'] == 0:
                my_fam_healthiness_key = 'rev_only'
            elif fam_dict['rev'] == 0:
                my_fam_healthiness_key = 'fwd_only'

            # Add to raw read coverage
            my_stats_dict['raw_read_cov']['fwd'] += fam_dict['fwd']
            my_stats_dict['raw_read_cov']['rev'] += fam_dict['rev']
                
            # strictness of the UMT
            split_fam_key = fam_key.split('+')
            is_strict_umt = self.is_strict_tag(split_fam_key[1], 'zt') and self.is_strict_tag(split_fam_key[2], 'yt')
            
            # Add to  fam hist
            for strict_key in (['all', 'strict'] if is_strict_umt else ['all']):
		if(my_fam_size!=0) :
			group = self.reads_group_by_fam_size(my_fam_size)
			my_stats_dict['fam_read_cov'][group] +=my_fam_size
                try:
                    my_stats_dict['%s_fam_size_hist'%strict_key][my_fam_healthiness_key][my_fam_size] += 1
                except KeyError:
                    my_stats_dict['%s_fam_size_hist'%strict_key][my_fam_healthiness_key][my_fam_size] = 1
		if (strict_key == 'all' ) and (my_fam_size >= self.__tvc_param_dict['min_tag_fam_size']['value']):
		    my_stats_dict['%s_fam_cov'%my_fam_healthiness_key] +=1
            
                
            # Add to funct fam hist and func read cov
            is_func_fam = self.is_func_fam(fam_key, fam_dict)
		
            if is_func_fam:
                for strict_key in (['', 'strict_'] if is_strict_umt else ['']):
                    try:
                        my_stats_dict['%sfunc_fam_size_hist'%strict_key][my_fam_size] += 1
                    except KeyError:
                        my_stats_dict['%sfunc_fam_size_hist'%strict_key][my_fam_size] = 1
                    my_stats_dict['%sfunc_read_cov'%strict_key]['fwd'] += fam_dict['fwd']
                    my_stats_dict['%sfunc_read_cov'%strict_key]['rev'] += fam_dict['rev']
                fam_cov_no_strand_constraint += 1
            elif self.is_func_fam(fam_key, fam_dict, True):
                fam_cov_no_strand_constraint += 1
	    
          		

      
        # Handle the "miss_tag" reads:
        # family size of zero in the hist for "miss_tag" reads
        my_stats_dict['all_fam_size_hist']['fwd_only'][0] = all_fam_dict['miss_tag']['fwd']
        my_stats_dict['all_fam_size_hist']['rev_only'][0] = all_fam_dict['miss_tag']['rev']
        # Add "miss_tag" reads to raw_read_cov    
        my_stats_dict['raw_read_cov']['fwd'] += all_fam_dict['miss_tag']['fwd']
        my_stats_dict['raw_read_cov']['rev'] += all_fam_dict['miss_tag']['rev']  
        
        # The following stats do not need to be caluclated in the for loop:     
        for key in ('all_fam_size_hist', 'strict_fam_size_hist'):
            fam_size_set = set([])
            for strand_key in ('fwd_rev', 'fwd_only', 'rev_only'):
                fam_size_set.update(my_stats_dict[key][strand_key])
            for fam_size in fam_size_set:
                my_stats_dict[key]['all'][fam_size] = my_stats_dict[key]['fwd_rev'].get(fam_size, 0) + my_stats_dict[key]['fwd_only'].get(fam_size, 0) + my_stats_dict[key]['rev_only'].get(fam_size, 0)

        # calcuate average family size 
        total_reads = 0
        for k,v  in my_stats_dict['func_fam_size_hist'].iteritems():
	    total_reads = total_reads + k*v
	total_family = sum(my_stats_dict['func_fam_size_hist'].values())
	if total_family != 0 :
	    my_stats_dict['average_fam_size']=total_reads*1.0/total_family


        my_stats_dict['func_fam_cov'] = sum(my_stats_dict['func_fam_size_hist'].values())
        my_stats_dict['strict_func_fam_cov'] = sum(my_stats_dict['strict_func_fam_size_hist'].values())
        my_stats_dict['strict_func_umt_rate'] = 0.0 if my_stats_dict['func_fam_cov'] == 0 else ( float(my_stats_dict['strict_func_fam_cov'] ) / float(my_stats_dict['func_fam_cov']) )
        my_stats_dict['func_read_cov']['all'] = my_stats_dict['func_read_cov']['fwd'] + my_stats_dict['func_read_cov']['rev']
        my_stats_dict['strict_func_read_cov']['all'] = my_stats_dict['strict_func_read_cov']['fwd'] + my_stats_dict['strict_func_read_cov']['rev']        
        my_stats_dict['raw_read_cov']['all'] += (my_stats_dict['raw_read_cov']['fwd'] + my_stats_dict['raw_read_cov']['rev'])
        
        # Conversion efficiency
        my_stats_dict['r2f_conv_rate'] = dict([(strand_key, float(my_stats_dict['func_read_cov'][strand_key]) / float(raw_read_cov)) if raw_read_cov != 0 else (strand_key, 0.0) for strand_key, raw_read_cov in my_stats_dict['raw_read_cov'].iteritems()])

        # func fam coverage loss due to "min_fam_per_strand_cov"
        my_stats_dict['func_fam_cov_loss_due_to_strand'] = (1.0 - float(my_stats_dict['func_fam_cov']) / float(fam_cov_no_strand_constraint)) if fam_cov_no_strand_constraint != 0 else 0.0
        
        # LOD
        my_stats_dict['lod'] = self.__lod_manager.calculate_lod(my_stats_dict['func_fam_cov'])
        # Put the region in if needed
        if region_dict is not None:
            my_stats_dict.update(region_dict)

        # Add parameters
        my_stats_dict['param'] = dict([(param_key, param_dict['value']) for param_key, param_dict in  self.__tvc_param_dict.iteritems()])

        return my_stats_dict


def get_stats_for_xls(stats_one_region):
    one_region_stats_for_xls= {
        'raw_read_cov_all': None,
        'raw_read_cov_fwd': None,
        'raw_read_cov_rev': None,
        'func_read_cov_all': None,
        'func_read_cov_fwd': None,
        'func_read_cov_rev': None,
        'func_fam_cov': None,
        'func_fam_cov_loss_due_to_strand': None,
        'lod': None,
        'average_fam_size':None,
	'both_strands_fam_cov': None,
        'fwd_only_fam_cov': None,
        'rev_only_fam_cov': None,      
        'strict_func_fam_cov': None,
        'strict_func_umt_rate': None,
        'r2f_conv_rate_all': None,
        'r2f_conv_rate_rev': None,
        'r2f_conv_rate_fwd': None,	
        'per_cov_small_fam': None,	
        'per_cov_median_fam': None,	
        'per_cov_large_fam': None,	
    }
    
    # Add to stats_for_meta        
    for key in ('func_fam_cov', 'strict_func_fam_cov', 'lod', 'strict_func_umt_rate','fwd_only_fam_cov','rev_only_fam_cov','average_fam_size'):
        one_region_stats_for_xls[key] = stats_one_region[key]

    for key in ('raw_read_cov', 'func_read_cov'):
        for strand_key, cov_value in stats_one_region[key].iteritems():
            one_region_stats_for_xls['%s_%s'%(key, strand_key)] = cov_value             
    for strand_key in ('fwd', 'rev', 'all'):
        one_region_stats_for_xls['r2f_conv_rate_%s'%strand_key] = stats_one_region['r2f_conv_rate'][strand_key]*100

    for group_key in ('small','median','large'):
	total_read_cov = stats_one_region['fam_read_cov']['small']+stats_one_region['fam_read_cov']['median']+stats_one_region['fam_read_cov']['large']
        one_region_stats_for_xls['per_cov_%s_fam'%group_key] = 0 if total_read_cov == 0 else (100.0*stats_one_region['fam_read_cov'][group_key]/(stats_one_region['fam_read_cov']['small']+stats_one_region['fam_read_cov']['median']+stats_one_region['fam_read_cov']['large']))


    one_region_stats_for_xls['func_fam_cov_loss_due_to_strand'] = stats_one_region['func_fam_cov_loss_due_to_strand'] 
    one_region_stats_for_xls['both_strands_fam_cov'] = stats_one_region['fwd_rev_fam_cov']

    # reformat precision float value for output
    for key in ('average_fam_size','per_cov_small_fam','per_cov_median_fam','per_cov_large_fam'):
        one_region_stats_for_xls[key] = "%.1f"%(one_region_stats_for_xls[key])
    for key in ('strict_func_umt_rate','func_fam_cov_loss_due_to_strand'):
        one_region_stats_for_xls[key] = "%.3f"%(one_region_stats_for_xls[key])
    return one_region_stats_for_xls

#--------------------------------------------------------------------------------------------------------

def plot_stats(my_stat_dict, png_path_prefix='', fig_idx=1):
    # The figures are shared with all workers. Need to make sure they are independent.
    max_fam_size = max(my_stat_dict['all_fam_size_hist']['all'])
    x_cutoff = 40 # for "geq x_cutoff"
    is_need_x_geq = max_fam_size > x_cutoff
    x_ary = numpy.arange(min(x_cutoff, max_fam_size) + 1, dtype=int)
    
    color_dict_by_strand = {
        'fwd_rev': numpy.array((78, 153, 59))/255.0,
        'fwd_only': numpy.array((232, 150, 149))/255.0,
        'rev_only': numpy.array((150, 148, 233))/255.0 ,
    }
    
    # Figure 1 and Figure 2
    # Iterate over (Figure 1)"all_umt" and (Figure 2) "strict_umt"
    for fig_key in ['all', 'strict']:
        # Initialize the figure
        my_fig = plt.figure(fig_idx, figsize=(18,10))
        my_fig.hold(1)
        # Plot two subplots
        axes_list = [my_fig.add_subplot(subplot_key) for subplot_key in ('121', '122')]
    
        # Initialize the lists/arrays
        bottom_ary = numpy.zeros(len(x_ary), dtype=int)
        bottom_ccumsum_ary = numpy.zeros(len(x_ary), dtype=int)

        # Iterate over the bars in the same plot
        for strand_key in ('fwd_rev', 'fwd_only', 'rev_only'):
            # y_list is the hist
            y_ary = numpy.zeros(len(x_ary), dtype=int)
            for fam_size, fam_counts in my_stat_dict['%s_fam_size_hist'%fig_key][strand_key].iteritems():
                fam_size = int(fam_size)
                y_ary[min(fam_size, x_cutoff)] += fam_counts            
            
            # Plot for fam size hist (subplots in the first row)
            axes_list[0].hold(1)            
            axes_list[0].bar(x_ary, y_ary, 1, bottom=bottom_ary, color=color_dict_by_strand[strand_key], align='center', log=False)
            bottom_ary += y_ary

            # Plot for fam size CCDF  (subplots in the second row)
            ccumsum_y_ary = numpy.cumsum(y_ary[::-1])[::-1]
            axes_list[1].hold(1)                       
            axes_list[1].bar(x_ary, ccumsum_y_ary, 1, bottom=bottom_ccumsum_ary, color=color_dict_by_strand[strand_key], align='center', log=False)
            bottom_ccumsum_ary += ccumsum_y_ary
            
        # Finalize the figure
        strictness_text = 'all UMT' if fig_key == 'all' else 'strict UMT only'
        sup_title_text_list = [
            'Amplicon %s, %s:%d-%d' %(my_stat_dict['name'], my_stat_dict['chrom'], my_stat_dict['chromStart'] + 1, my_stat_dict['chromEnd']),
            'Families are generated from reads with %s and cover more than %d%s of the amplicon' %(strictness_text, int(100.0 * my_stat_dict['param']['min_cov_fraction']), '%')
        ]

        if my_stat_dict['all_fam_size_hist']['all'].get(0, 0):
            sup_title_text_list.append('\nReads with missing UMT are counted in family of size 0.')
        
        my_fig.suptitle('\n'.join(sup_title_text_list))

        # Finalize the subplots
        for subplot_idx, my_axes in enumerate(axes_list):
            xlim_1 = min(x_cutoff, max_fam_size)
            my_axes.set_xlim([-0.5, xlim_1 + 0.5])
            if xlim_1 > 10:
                xticks_list = range(0, min(x_cutoff, max_fam_size) + 1, 5)
            else:
                xticks_list = range(xlim_1 + 1)
            if is_need_x_geq and subplot_idx == 0:
                if x_cutoff not in xticks_list:
                    xticks_list.append(x_cutoff)
                xticks_labels = ['%d'%x for x in xticks_list]
                xticks_labels[xticks_list.index(x_cutoff)] = '$\geq$ %d' %x_cutoff
            else:
                xticks_labels = ['%d'%x for x in xticks_list]
            my_axes.set_xticks(xticks_list)
            my_axes.set_xticklabels(xticks_labels)
            my_axes.legend(['FWD & REV', 'FWD only', 'REV only'])        

            if subplot_idx == 0:
                title_text = "Family size histogram (%s)" %(strictness_text)
                ylabel_text = '#(families of size $x$)'
            else:
                title_text = "Complementary culmulative sum of the histogram (%s)"%strictness_text
                ylabel_text = '#(families of size $\geq x$)'
            my_axes.set_title(title_text)
            my_axes.set_ylabel(ylabel_text)
            my_axes.set_xlabel('$x$')
            my_axes.grid(1)
        my_fig.subplots_adjust(wspace=0.2, left=0.08, right=0.95, top=0.88, bottom=0.1)
        my_fig.savefig('%sfam_size_hist_%s_umt.png'%(png_path_prefix, fig_key))
        plt.close(fig_idx)


    # Figure 3: Functional family hist
    max_fam_size = max(my_stat_dict['func_fam_size_hist'])
    smallest_fam_size = max_fam_size
    is_need_x_geq = max_fam_size > x_cutoff
    x_ary = numpy.arange(min(x_cutoff, max_fam_size) + 1, dtype=int)
    
    # Initialize the figure

    my_fig = plt.figure(fig_idx, figsize=(18,10))
    my_fig.hold(1)
    # Plot two subplots
    axes_list = [my_fig.add_subplot(subplot_key) for subplot_key in ('121', '122')]


    # Initialize the lists/arrays
    y_strict_func_ary = numpy.zeros(len(x_ary), dtype=int)
    y_func_ary = numpy.zeros(len(x_ary), dtype=int)
    for fam_size, fam_counts in my_stat_dict['strict_func_fam_size_hist'].iteritems():
        fam_size = int(fam_size)
        y_strict_func_ary[min(fam_size, x_cutoff)] += fam_counts
    for fam_size, fam_counts in my_stat_dict['func_fam_size_hist'].iteritems():
        fam_size = int(fam_size)
        y_func_ary[min(fam_size, x_cutoff)] += fam_counts
        if fam_counts > 0 and fam_size < smallest_fam_size:
            smallest_fam_size = fam_size
            

    # Plot for fam size hist (subplots in the first row)
    axes_list[0].hold(1)            
    axes_list[0].bar(x_ary, y_func_ary, 1, color='silver', align='center', log=False)
    axes_list[0].bar(x_ary, y_strict_func_ary, 1, color='lightblue', align='center', log=False)


    # Plot for fam size CCDF  (subplots in the second row)
    ccumsum_y_func_ary = numpy.cumsum(y_func_ary[::-1])[::-1]
    ccumsum_y_strict_func_ary = numpy.cumsum(y_strict_func_ary[::-1])[::-1]

    axes_list[1].hold(1)    
    axes_list[1].bar(x_ary, ccumsum_y_func_ary, 1, color='silver', align='center', log=False)
    axes_list[1].bar(x_ary, ccumsum_y_strict_func_ary, 1, color='lightblue', align='center', log=False)
    
    # suptitle for the fig
    func_condition_text = ['A functinoal family must']
    abc_list = list('abc')
    if my_stat_dict['param']['tag_trim_method'].lower().startswith('strict'):
        func_condition_text.append('%s) have a strict UMT, and'%abc_list.pop(0))
    func_condition_text.append('%s) consist of at least %d reads,' %(abc_list.pop(0), my_stat_dict['param']['min_tag_fam_size']))
    if my_stat_dict['umt_strand_type'] == 'B' and my_stat_dict['param']['min_fam_per_strand_cov'] > 0:
        func_condition_text.append('and %s) at least %d read(s) on each strand,'%(abc_list.pop(0), my_stat_dict['param']['min_fam_per_strand_cov']))
    func_condition_text.append('where the reads must cover more that %d%s of the amplicon.' %(int(100.0 * my_stat_dict['param']['min_cov_fraction']), '%'))        
    sup_title_text_list = [
        'Amplicon %s, %s:%d-%d' %(my_stat_dict['name'], my_stat_dict['chrom'], my_stat_dict['chromStart'] + 1, my_stat_dict['chromEnd']),
        ' '.join(func_condition_text) + '.',
    ]
    my_fig.suptitle('\n'.join(sup_title_text_list))

    # Finalize the subplots
    for subplot_idx, my_axes in enumerate(axes_list):               
        xlim_1 = min(x_cutoff, max_fam_size)
        if xlim_1 > 10:
            xticks_list = range(0, min(x_cutoff, max_fam_size) + 1, 5)
        else:
            xticks_list = range(xlim_1 + 1)
        if is_need_x_geq and subplot_idx == 0:
            if x_cutoff not in xticks_list:
                xticks_list.append(x_cutoff)
            xticks_labels = ['%d'%x for x in xticks_list]
            xticks_labels[xticks_list.index(x_cutoff)] = '$\geq$ %d' %x_cutoff
        else:
            xticks_labels = ['%d'%x for x in xticks_list]
        if smallest_fam_size not in xticks_list:
            xticks_list.append(smallest_fam_size)
            xticks_labels.append('%d'%smallest_fam_size)
        my_axes.set_xticks(xticks_list)
        my_axes.set_xticklabels(xticks_labels)
        my_axes.set_xlim([smallest_fam_size - 0.5, xlim_1 + 0.5])        
        my_axes.legend(['Non-strict UMT', 'Strict UMT'])
        if subplot_idx == 0:
            title_text = "Functional family size histogram"
            ylabel_text = '#(families of size $x$)'
        else:
            title_text = "Complementary culmulative sum of the histogram"
            ylabel_text = '#(families of size $\geq x$)'
        my_axes.set_title(title_text)
        my_axes.set_ylabel(ylabel_text)
        my_axes.set_xlabel('$x$')
        my_axes.grid(1)
    my_fig.subplots_adjust(wspace=0.2, left=0.08, right=0.95, top=0.88, bottom=0.1)
    my_fig.savefig('%sfunc_fam_size_hist.png'%png_path_prefix)
    plt.close(fig_idx)

            
#--------------------------------------------------------------------------------------------------------

def get_default_tvc_param_dict():
    tvc_param_dict = {
        'torrent_variant_caller':{
            'tag_trim_method': 'sloppy-trim',
            'min_tag_fam_size': 3, 
            'min_fam_per_strand_cov': 1,
            'hotspot_min_variant_score': 3.0,
            'hotspot_min_allele_freq': 0.0005,
            'hotspot_min_var_coverage': 3,
            'min_callable_prob': 0.98,
        },
        'freebayes':{
            'min_cov_fraction': 0.9,
            'read_snp_limit': 10,
            'min_mapping_qv': 4,
            'read_mismatch_limit': 0,
            'read_max_mismatch_fraction': 1.0,
        }
    }
    return tvc_param_dict

#--------------------------------------------------------------------------------------------------------
def dump_to_json(my_data, json_path):
    with open(json_path, 'w') as f_json:
        json.dump(my_data, f_json, indent=4, sort_keys=True)

#--------------------------------------------------------------------------------------------------------
def dump_stats_xls(stats_for_xls_list, region_list, xls_path):

    assert(len(region_list) == len(stats_for_xls_list))
  #  header_list = [
  #      'contig_id', 'contig_srt', 'contig_end', 'region_id', 'attributes',
  #      'func_fam_cov', 'lod', 'strict_func_umt_rate', 'func_fam_cov_loss_due_to_strand',
  #      'raw_read_cov_all', 'func_read_cov_all', 'r2f_conv_rate_all',
  #      'raw_read_cov_fwd', 'func_read_cov_fwd', 'r2f_conv_rate_fwd',
  #      'raw_read_cov_rev', 'func_read_cov_rev', 'r2f_conv_rate_rev',
  #  ]
    # the names exposed to the customer are all "mol" instead of "fam"
    header_list = [
         'contig_id', 'contig_srt', 'contig_end', 'region_id', 'attributes',
         'func_mol_cov', 'lod', 'strict_func_umt_rate', 'func_mol_cov_loss_due_to_strand',
         'fwd_only_mol_cov', 'rev_only_mol_cov', 'both_strands_mol_cov','perc_functional_reads','reads_per_func_mol','perc_to_mol_(<3_reads)','perc_to_mol_(>=3_&<30_reads)','perc_to_mol_(>=30_reads)'
    ]
    
    with open(xls_path, 'w') as f_xls:
        f_xls.writelines(['\t'.join(header_list), '\n'])
        for region_dict, one_region_stats_for_xls in zip(region_list, stats_for_xls_list):
            lod = one_region_stats_for_xls['lod']
            line_list = [
                region_dict['chrom'], region_dict['chromStart'] + 1, region_dict['chromEnd'], region_dict['name'], region_dict['info'],
                one_region_stats_for_xls['func_fam_cov'], lod if lod is not None else 'N/A', one_region_stats_for_xls['strict_func_umt_rate'], one_region_stats_for_xls['func_fam_cov_loss_due_to_strand'],
                one_region_stats_for_xls['fwd_only_fam_cov'], one_region_stats_for_xls['rev_only_fam_cov'], one_region_stats_for_xls['both_strands_fam_cov'],one_region_stats_for_xls['r2f_conv_rate_all'],
                one_region_stats_for_xls['average_fam_size'],
		one_region_stats_for_xls['per_cov_small_fam'],one_region_stats_for_xls['per_cov_median_fam'],one_region_stats_for_xls['per_cov_large_fam'],
                #one_region_stats_for_xls['raw_read_cov_fwd'], one_refamgion_stats_for_xls['func_read_cov_fwd'], one_region_stats_for_xls['r2f_conv_rate_fwd'],
                #one_region_stats_for_xls['raw_read_cov_rev'], one_region_stats_for_xls['func_read_cov_rev'], one_region_stats_for_xls['r2f_conv_rate_rev'],
            ]
            f_xls.writelines(['\t'.join(map(str, line_list)), '\n'])

#--------------------------------------------------------------------------------------------------------
def my_percentile_nearest(sorted_list, percentile, do_reverse=False):
    assert(0 <= percentile <= 100 and sorted_list)
    if do_reverse:
        percentile = 100.0 - percentile
    precentile_idx = int(numpy.round((len(sorted_list) - 1) * percentile / 100.0))
    return sorted_list[precentile_idx]
#
def gen_meta_stats(stats_for_xls_list):
    precentil_idx_dict = {
        'worst': 0, 
        'bottom_20': 20, 
        'median': 50, 
        'top_20': 80, 
        'best': 100,
    }
    coverage_metric_keys = (
        'func_fam_cov', 'lod', 'strict_func_umt_rate', 'func_fam_cov_loss_due_to_strand',
        'raw_read_cov_all', 'func_read_cov_all', 'r2f_conv_rate_all',
        'raw_read_cov_fwd', 'func_read_cov_fwd', 'r2f_conv_rate_fwd',
        'raw_read_cov_rev', 'func_read_cov_rev', 'r2f_conv_rate_rev'
    )
    # Initialize meta_stat_dict by assuming no coverage at all.
    meta_stats_dict = dict([(coverage_metric, dict([(key, 0 if coverage_metric != 'lod' else 'N/A') for key in precentil_idx_dict])) for coverage_metric in coverage_metric_keys])

    if not stats_for_xls_list:
        return meta_stats_dict
    
    # stats_for_xls_list is a list of dict
    # Convert it to dict of list
    dict_of_list = dict([(coverage_metric_key, [region_stats_dict[coverage_metric_key] for region_stats_dict in stats_for_xls_list]) for coverage_metric_key in coverage_metric_keys])

    for coverage_metric_key, coverage_metric_list in dict_of_list.iteritems():
        if coverage_metric_key == 'lod':
            # convert None to 2.0 just for the sorting of LOD (None is worse than 1.0 in LOD)
            # That's one reason that I use "nearest" for percentile calculation
            coverage_metric_list = [2.0 if v is None else v for v in coverage_metric_list]
        # Sort the list 
        coverage_metric_list.sort()
        meta_stats_dict[coverage_metric_key] = dict([(stat_key, my_percentile_nearest(coverage_metric_list, percentile, coverage_metric_key in ('lod', 'func_fam_cov_loss_due_to_strand'))) for stat_key, percentile in precentil_idx_dict.iteritems()])
    
    # Handle LOD if there is no coverage in the amplicon 
    for key in precentil_idx_dict:
        if meta_stats_dict['lod'][key] == 2.0:
            meta_stats_dict['lod'][key] = 'N/A'

    return meta_stats_dict
    

def mol_coverage_analysis_worker(input_dict):
    bam_path = input_dict['bam_path']
    region_start_idx = input_dict['region_start_idx']
    region_end_idx = input_dict['region_end_idx']
    # The worker is responsible to process region_list[region_start_idx:region_end_idx]
    # I input the entire region_list because of the handling of super amplicons, as TVC does.
    region_list = input_dict['region_list']
    tvc_param_dict = input_dict['tvc_param_dict']
    output_dir = input_dict['output_dir']
    details_dir = os.path.join(output_dir, '%s.details'%input_dict['output_prefix'])
    output_prefix = input_dict['output_prefix']
    ignore_zr = input_dict['ignore_zr']
    make_plots = input_dict['make_plots']

    # Initialize stats_list
    stats_for_xls_list = []
    
    # Family Manager
    with FamilyManager(bam_path) as umt_manager:
        umt_manager.set_tvc_param(tvc_param_dict)
        umt_manager.set_ignore_zr(ignore_zr)
        # Iterate over all regions
        for region_idx, region_dict in enumerate(region_list[region_start_idx:region_end_idx]):
            file_prefix_one_region = '%s.%s_%d-%d'%(output_prefix, region_dict['chrom'], region_dict['chromStart'], region_dict['chromEnd'])
            # Do family classification
            all_fam_dict = umt_manager.gen_fam_dict_one_region(region_dict, region_list, region_idx)
    
            # Analyze the stats for the region
            stats_one_region = umt_manager.gen_stats_one_region(all_fam_dict, region_dict)
            
            # Dump stats_one_region to json
            stats_one_region_json_path = os.path.join(details_dir, '%s.stats.json' %(file_prefix_one_region))
            dump_to_json(stats_one_region, stats_one_region_json_path)
            with open(stats_one_region_json_path, 'r') as f_json:
                stats_one_region = json.load(f_json)
               
            # Make a plot
            
            # CZB: Could be slow. Need a solution to speed up.
            if make_plots:
                png_path_prefix = os.path.join(details_dir, '%s' %(file_prefix_one_region))          
                plot_stats(stats_one_region, png_path_prefix, input_dict['worker_id'] + 1)
             
            # For xls and meta stats
            stats_for_xls_list.append(get_stats_for_xls(stats_one_region))

            # Track the status of the worker
#            print('    - Worker #%d: %d/%d completed' %(input_dict['worker_id'], region_idx + 1, len(region_list[region_start_idx:region_end_idx])))

    return stats_for_xls_list
                    
#--------------------------------------------------------------------------------------------------------

def main(input_dict):
    bam_path = input_dict['bam_path']
    bed_path = input_dict['bed_path']
    num_threads = input_dict['num_threads']
    tvc_param_dict = input_dict['tvc_param_dict']
    output_dir = input_dict['output_dir']
    output_prefix = input_dict['output_prefix']
    ignore_zr = input_dict['ignore_zr']
    make_plots = input_dict['make_plots']

    printtime('Start analyzing coverage related metrics for UMT.')
    print('    - BAM file: %s'%bam_path)
    print('    - BED file: %s'%bed_path)

    # mkdir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    details_dir = os.path.join(output_dir, '%s.details'%output_prefix)
    if not os.path.isdir(details_dir):
        os.mkdir(details_dir)    
    
    # Parse BED regions
    with open(bed_path, 'r') as f_bed:
        region_list = tuple([bed_line_to_basic_dict(line) for line in f_bed if not line.startswith('track')])
    # Create job for the workers
    num_workers = num_threads
    num_workers = min(num_workers, len(region_list))
    num_regions_per_worker = len(region_list) / num_workers
    num_extra_regions = len(region_list) % num_workers 
    worker_start_idx_list = numpy.cumsum([0] + [num_regions_per_worker + 1 if work_id < num_extra_regions else num_regions_per_worker for work_id in xrange(num_workers)])
    
    printtime('Processing %d regions with %d threads (# of regions per thread = %d%s) '%(len(region_list), num_workers, num_regions_per_worker, ' or %d'%(num_regions_per_worker + 1) if num_extra_regions else ''))    
    
    # Prepare input dict for the workers    
    input_dict_list = [{
            'worker_id': worker_id, 
            'bam_path': bam_path, 
            'region_list': region_list, 
            'region_start_idx': worker_start_idx_list[worker_id], 
            'region_end_idx': worker_start_idx_list[worker_id + 1],
            'tvc_param_dict': tvc_param_dict, 
            'ignore_zr': ignore_zr, 
            'output_dir': output_dir, 
            'output_prefix': output_prefix,
            'make_plots': make_plots,
        } for worker_id in xrange(num_workers)]

    # pool for the workers
    pool = ThreadPool(num_workers)
    worker_output_list = pool.map(mol_coverage_analysis_worker, input_dict_list)
    stats_for_xls_list = [item for sublist in worker_output_list for item in sublist]

    # Dump xls
    xls_path = os.path.join(output_dir, '%s.amplicon.cov.xls'%output_prefix)
    dump_stats_xls(stats_for_xls_list, region_list, xls_path)
    printtime('Detailed stats XLS file successfully saved to %s'%xls_path)
    
    # Meta stats
    meta_stats_dict = gen_meta_stats(stats_for_xls_list)
    meta_stats_json_path = os.path.join(output_dir, '%s.cov.stats.json'%output_prefix)
    dump_to_json(meta_stats_dict, meta_stats_json_path)
    printtime('Meta stats JSON file successfully saved to %s'%meta_stats_json_path)

    # Dump used tvc parameter
    tvc_param_json_path = os.path.join(output_dir, 'local_parameters_coaverageAnalysisUMT.json')
    dump_to_json(tvc_param_dict, tvc_param_json_path)

    printtime('The analysis is completed successfully!')
    
#--------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Option Parser
    parser = OptionParser("Coverage Analysis for UMT.")
    parser.add_option('-b', '--bam-file',         help='Input BAM file', dest='bam_file')
    parser.add_option('-r', '--bed-file',         help='The designed region BED file', dest='bed_file')
    parser.add_option('-t', '--num-threads',      help='Number of threads to be used [Default=8]', dest='num_threads')
    parser.add_option('-O', '--output-dir',       help='Output directory [Default=current directory]', dest='output_dir')
    parser.add_option('-o', '--output-prefix',    help='The prefix that will be added to the name of the output files [Default=""]', dest='output_prefix')
    parser.add_option('-p', '--make-plots',       help='Make plots of the family size histogram [Default=0]', dest='make_plots')    
    parser.add_option('-z', '--ignore-zr',        help='Ignore ZR tag of the BAM reads. Set to 1 if it is not a consensus BAM. [Default=0]', dest='ignore_zr')    
    parser.add_option('-j', '--tvc-json',         help='(Optional) TVC parameter json file (for determining the functionality of families, etc).', dest='tvc_json')

    (options, args) = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        print('(Note): If tvc json file is no provided or the parameter is not specified, the following default parameters will be used:')
        for key, value in get_default_tvc_param_dict().iteritems():
            print("    '%s': %s" %(key, str(value)))
        sys.exit(0)

    input_dict = {}
    
    # BAM file
    if options.bam_file is None:
        raise(IOError("No input BAM file provided."))
        sys.exit(1)
    else:
        input_dict['bam_path'] = options.bam_file

    # BED file   
    if options.bed_file is None:
        raise(IOError("No input BED file provided."))
        sys.exit(1)
    else:
        input_dict['bed_path'] = options.bed_file

    # num_threads
    input_dict['num_threads'] = 8 if options.num_threads is None else int(options.num_threads)

    # ignore_zr ?
    input_dict['ignore_zr'] = 0 if options.ignore_zr is None else int(options.ignore_zr)

    # make_plots ?
    input_dict['make_plots'] = 0 if options.make_plots is None else int(options.make_plots)

    # output dir
    input_dict['output_dir'] = os.getcwd() if options.output_dir is None else options.output_dir
        
    # output prefix
    input_dict['output_prefix'] = '' if options.output_prefix is None else options.output_prefix
           
        
    # tvc json
    tvc_param_dict = get_default_tvc_param_dict()
    if options.tvc_json is None:
        input_tvc_param_dict = {"torrent_variant_caller": {}, "freebayes": {}}
    else:
        with open(options.tvc_json, 'r') as f_json:
            input_tvc_param_dict = json.load(f_json)
    for section_key in ('torrent_variant_caller', 'freebayes'):
        if section_key not in input_tvc_param_dict:
            raise(KeyError('The key "%s" is not found in the JSON file "%s".' %(section_key, options.tvc_json)))
        for param_key in tvc_param_dict[section_key]:
            tvc_param_dict[section_key][param_key] = input_tvc_param_dict[section_key].get(param_key, tvc_param_dict[section_key][param_key])
    input_dict['tvc_param_dict'] = tvc_param_dict
    
    # main function
    main(input_dict)
