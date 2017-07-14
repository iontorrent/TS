#!/usr/bin/python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

# Test all the json files are formatted correctly
# check for typos
#
# TVC 4.6 version
# update to make sure all documented parameters are present
#
# example use:
# for f in `ls -1 *.json`; do python test_format.py $f; done

import sys
import json
import subprocess

sections={"torrent_variant_caller": [],
          "long_indel_assembler": [],
          "freebayes": []
          }

sections["torrent_variant_caller"]=[
                                    # allele classification parameters
                                    "use_fd_param",
                                    "min_ratio_for_fd",                                        
                                    "indel_as_hpindel",                                     
                                    # Allele specific parameters
                                    ## indel parameters
                                    "indel_min_allele_freq",
                                    "indel_min_variant_score",
                                    "indel_min_coverage",
                                    "indel_min_cov_each_strand",   
                                    "indel_strand_bias", 
                                    "indel_strand_bias_pval",                                    
                                    ## snp parameters
                                    "snp_min_allele_freq",
                                    "snp_min_variant_score",
                                    "snp_min_coverage",
                                    "snp_min_cov_each_strand",   
                                    "snp_strand_bias", 
                                    "snp_strand_bias_pval",
                                    ## mnp parameters
                                    "mnp_min_allele_freq",
                                    "mnp_min_variant_score",
                                    "mnp_min_coverage",
                                    "mnp_min_cov_each_strand",   
                                    "mnp_strand_bias", 
                                    "mnp_strand_bias_pval",
                                    ## hotspot parameters
                                    "hotspot_min_allele_freq",
                                    "hotspot_min_variant_score",
                                    "hotspot_min_coverage",
                                    "hotspot_min_cov_each_strand",   
                                    "hotspot_strand_bias", 
                                    "hotspot_strand_bias_pval",
                                    # SNP/MNP realignment parameters 
                                    "do_snp_realignment",
                                    "do_mnp_realignment",
                                    "realignment_threshold",                                       
                                    # Flow evaluation parameters
                                    "downsample_to_coverage",
                                    "heavy_tailed",
                                    "outlier_probability",
                                    "prediction_precision",
                                    "min_detail_level_for_fast_scan",
                                    "max_flows_to_test",                                    
                                    "suppress_recalibration",
                                    # HP length filters
                                    "hp_max_length",
                                    "hp_indel_hrun",
                                    "hp_ins_len",
                                    "hp_del_len",
                                    # Flow evaluation filters
                                    "data_quality_stringency",
                                    "filter_unusual_predictions",
                                    "filter_deletion_predictions",
                                    "filter_insertion_predictions",
                                    # Position bias filters
                                    "use_position_bias",
                                    "position_bias",
                                    "position_bias_pval",
                                    "position_bias_ref_fraction",                                    
                                    # SSE filters
                                    "error_motifs",                                    
                                    "sse_prob_threshold",
                                    # Others
                                    "report_ppa",
                                    ]
sections["long_indel_assembler"]=["kmer_len",
                                  "min_var_freq",                                  
                                  "min_var_count",
                                  "short_suffix_match",
                                  "min_indel_size",
                                  "max_hp_length",
                                  "relative_strand_bias",
                                  "output_mnv",
                                  ]
sections["freebayes"]=["allow_indels",
                       "allow_snps",
                       "allow_mnps",
                       "allow_complex",
                       "gen_min_alt_allele_freq",                       
                       "gen_min_indel_alt_allele_freq",
                       "gen_min_coverage",
                       "min_mapping_qv",
                       "read_snp_limit",
                       "read_max_mismatch_fraction",                       
                       ]

def tagseq_param(my_sections):
    my_sections['long_indel_assembler'] = []
    if 'torrent_variant_caller' not in my_sections:
        my_sections["torrent_variant_caller"] = []
    my_sections["torrent_variant_caller"] += ['min_tag_fam_size',
                                              'indel_func_size_offset',
                                              'tag_trim_method',
                                              'snp_min_var_coverage',
                                              'indel_min_var_coverage',
                                              'mnp_min_var_coverage',
                                              'hotspot_min_var_coverage',
                                              'fd_nonsnp_min_var_cov',
                                              'tag_sim_max_cov',
                                              'use_lod_filter',
                                              'lod_multiplier',
                                              'try_few_restart_freq',
                                              ]
    if 'freebayes' not in my_sections:
        my_sections["freebayes"] = []    
    my_sections['freebayes'] += ['read_mismatch_limit',
                                 'min_cov_fraction',
                                 ]

def main(argv):
    if not argv:
        inputfile = ''
    else:
        inputfile = argv[0]
    
    with open(inputfile,'r') if len(inputfile) >1 else sys.stdin as json_file:
        try:
            d = json.load(json_file)
        except:
            print "Failure to load %s" % (inputfile)
            raise
        
        # Is it a tagseq parameter file?
        library_list = [lib.lower().replace(' ', '_') for lib in d.get('meta', {}).get('compatibility', {}).get('library', [])]
        if 'tagseq' in library_list or 'tag_sequencing' in library_list or 'tag_seq' in library_list:
             tagseq_param(sections)
        
        # test for typos
        error = False
        for section, val in sections.iteritems():
            try:
                x = d[section]
            except:
                print  "Error: %s does not contain section %s" % (inputfile, section)
                error = True
                continue
            for v in val:
                try:
                    vv = x[v]
                except:
                    print  "Error: %s does not contain key %s" % (inputfile, v)
                    error = True
                    continue
                del x[v]

            if len(x) > 0:
                print "Error: %s contains an unmatched key %s" % (inputfile, x.keys())
                error = True

    if not error:
        if len(inputfile)>1:
            print "No error in %s" % inputfile
        else:
            print "No error"

if __name__ == "__main__":
    main(sys.argv[1:])









