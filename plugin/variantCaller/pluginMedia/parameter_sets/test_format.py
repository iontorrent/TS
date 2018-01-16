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
import os

required_parameters = dict([(key, []) for key in ['torrent_variant_caller', 'long_indel_assembler', 'freebayes']]) 
required_parameters["torrent_variant_caller"] = [
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
required_parameters["long_indel_assembler"]=["kmer_len",
                                  "min_var_freq",                                  
                                  "min_var_count",
                                  "short_suffix_match",
                                  "min_indel_size",
                                  "max_hp_length",
                                  "relative_strand_bias",
                                  "output_mnv",
                                  ]
required_parameters["freebayes"]=["allow_indels",
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

def tagseq_param(my_required_parameters):
    my_required_parameters['long_indel_assembler'] = []
    my_required_parameters["torrent_variant_caller"] += ['min_tag_fam_size',
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
    my_required_parameters['freebayes'] += ['read_mismatch_limit',
                                            'min_cov_fraction',
                                            ]

def convert_to_type(my_value, my_type):
    if my_type == 'Integer':
        if isinstance(my_value, float):
            raise TypeError('%f is a float, not an int number.' %my_value)
        return int(my_value)
    if my_type == 'Float':
        return float(my_value)
    if my_type == 'String':
        return str(my_value)
    if my_type == 'Bool':
        if my_value in [True, 1, '1']:
            return True
        if my_value in [False, 0, '0']:
            return False
        # CZB: Note that tvc (i.e., json.h) does not recognize 'true', 'True', 'false', 'False', 'on', 'off' in parameter json.
        raise TypeError('"%s" is not a valid Boolean value. Please use either 0 or 1.'%str(my_value))
    if my_type == 'Vector':
        # Note json decodes str as unicode
        if isinstance(my_value, (str, unicode)):
            my_value = str(my_value) # unicode cleanup
            if not (my_value.startswith('[') and my_value.endswith(']')):
                raise TypeError('"%s": Not a valid stringified vector. Please use, e.g., [1,2,3].' %str(my_value))
            if not my_value[1:-1]:
                return []
            return my_value[1:-1].split(',')
        if isinstance(my_value, (list, tuple)):
            return my_value
        raise ValueError('"%s": Invalid vector'%str(my_value))
    raise TypeError('Unknown Type %s'%my_type)
    
def check_range(my_converted_value, my_format_dict):
    if 'range' not in my_format_dict:
        return
    split_range = my_format_dict['range'].replace(' ', '').split(',')
    lower_bound = split_range[0][1:]
    lower_operator = split_range[0][0]
    assert(lower_operator in ['[', '(']) 
    upper_bound = split_range[1][:-1]
    upper_operator = split_range[1][-1]
    assert(upper_operator in [']', ')']) 

    if lower_bound != '-inf':
        lower_bound = convert_to_type(lower_bound, my_format_dict['type'])
        if (lower_operator == '[' and my_converted_value < lower_bound) or (lower_operator == '(' and my_converted_value <= lower_bound):
            raise ValueError('The value %s is not in the interval %s'%(str(my_converted_value), my_format_dict['range']))
        
    if upper_bound != 'inf':
        upper_bound = convert_to_type(upper_bound, my_format_dict['type'])
        if (upper_operator == ']' and my_converted_value > upper_bound) or (upper_operator == ')' and my_converted_value >= upper_bound):
            raise ValueError('The value %s is not in the interval %s'%(str(my_converted_value), my_format_dict['range']))

def check_allowed_values(my_converted_value, my_format_dict):
    if 'in' not in my_format_dict:
        return
    if my_converted_value not in my_format_dict['in']:
        raise ValueError('The value %s is not in %s' %(str(my_converted_value), str(my_format_dict['in'])))

def check_one_parameter(my_value, my_format_dict):
    my_value = convert_to_type(my_value, my_format_dict['type'])
    check_range(my_value, my_format_dict)
    check_allowed_values(my_value, my_format_dict)

def custom_check(my_param_json):
    # Custom check for hp_indel_hrun, hp_del_len, hp_ins_len
    hp_indel_hrun = my_param_json.get('torrent_variant_caller', {}).get('hp_indel_hrun', None)
    hp_del_len = my_param_json.get('torrent_variant_caller', {}).get('hp_del_len', None)
    hp_ins_len = my_param_json.get('torrent_variant_caller', {}).get('hp_ins_len', None)
    # None of them is in the parameter file
    if hp_indel_hrun == hp_del_len == hp_ins_len == None:
        return
    # They must be specified simultaneously.
    if None in [hp_indel_hrun, hp_del_len, hp_ins_len]:
        raise KeyError('The parameters "hp_indel_hrun", "hp_del_len", "hp_ins_len" must be specified in the parameter file simultaneously.')
       
    hp_indel_hrun = map(int, convert_to_type(hp_indel_hrun, 'Vector'))
    hp_del_len = map(int, convert_to_type(hp_del_len, 'Vector'))
    hp_ins_len = map(int, convert_to_type(hp_ins_len, 'Vector'))
    
    # Their length must be the same
    if len(hp_indel_hrun) != len(hp_del_len) or len(hp_del_len) != len(hp_ins_len):
        raise ValueError('The parameters "hp_indel_hrun", "hp_del_len", "hp_ins_len" must have the same length.')
    
    if len(hp_indel_hrun) == len(hp_del_len) == len(hp_ins_len) == 0:
        return
    
    # They must be non-negative
    if min(hp_indel_hrun) < 0:
        raise ValueError('min(hp_indel_hrun) can not be negative.')
    if min(hp_del_len) < 0:
        raise ValueError('min(hp_del_len) can not be negative.')
    if min(hp_ins_len) < 0:
        raise ValueError('min(hp_ins_len) can not be negative.')

def check_one_parameter_file(param_json_path, description_json_path = None):
    print("+ Checking the parameter json file: %s"%param_json_path)
    
    if description_json_path is None:
        description_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'configs/description.json')
    print('  - Loading description json: %s' %description_json_path)
    with open(description_json_path, 'rb') as f_json:
        description_json = json.load(f_json)
        
    print('  - Loading parameter json: %s' %param_json_path)
    with open(param_json_path, 'rb') as f_json:
        param_json = json.load(f_json)

    # Is it a tagseq parameter file?
    library_list = [lib.lower().replace(' ', '_') for lib in param_json.get('meta', {}).get('compatibility', {}).get('library', [])]
    if 'tagseq' in library_list or 'tag_sequencing' in library_list or 'tag_seq' in library_list:
         print('  - The parameter file %s is for Tagseq.' %param_json_path)
         tagseq_param(required_parameters)    

    error_list = []
    # First check the parameter file has all parameters required.
    for section_key, section_value in required_parameters.iteritems():
        if section_key not in param_json:
            error_list.append('The section "%s" is not in %s' %(section_key, param_json_path))
            continue
        for param_key in section_value:
            if param_key not in param_json[section_key]:
                error_list.append('The parameter "%s.%s" is required but not specified in' %(section_key, param_key, param_json_path))

    # Then check the parameters in the parameter file
    for section_key in required_parameters.keys():
        for param_key, param_value in param_json[section_key].iteritems():
            if param_key not in description_json[section_key]:
                # The parameters in the generic parameter file must be in description.json
                error_list.append('The parameter "%s.%s" is not specified in %s' %(section_key, param_key, description_json_path))
            else:
                try:
                    check_one_parameter(param_value, description_json[section_key][param_key])                    
                except Exception as e:
                    error_list.append('The parameter "%s.%s": %s' %(section_key, param_key, e.message))
    try:
        custom_check(param_json)
    except Exception as e:
        error_list.append(e.message)
    
    print('  + Checking the parameters in %s'%param_json_path)
    if error_list:
        for error_msg in error_list:
            print('    - ERROR: %s'%error_msg)
    else:
        print('    - No error.')
    return error_list

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputfile = sys.argv[1]
    else:
        inputfile = sys.stdin
    check_one_parameter_file(inputfile)
