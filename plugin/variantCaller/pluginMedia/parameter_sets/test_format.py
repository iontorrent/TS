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

sections["torrent_variant_caller"]=["data_quality_stringency",
                                    "hp_max_length",
                                    "filter_unusual_predictions",
                                    "filter_insertion_predictions",
                                    "filter_deletion_predictions", 
                                    "indel_as_hpindel",
                                    "snp_min_cov_each_strand",
                                    "snp_min_variant_score",
                                    "snp_min_allele_freq",
                                    "snp_min_coverage",
                                    "snp_strand_bias", 
                                    "snp_strand_bias_pval",
                                    "mnp_min_cov_each_strand",
                                    "mnp_min_variant_score",
                                    "mnp_min_allele_freq",
                                    "mnp_min_coverage",
                                    "mnp_strand_bias", 
                                    "mnp_strand_bias_pval",
                                    "indel_min_cov_each_strand",
                                    "indel_min_variant_score",
                                    "indel_min_allele_freq",
                                    "indel_min_coverage",
                                    "indel_strand_bias",
                                    "indel_strand_bias_pval",
                                    "hotspot_min_cov_each_strand",
                                    "hotspot_min_variant_score",
                                    "hotspot_min_allele_freq",
                                    "hotspot_min_coverage",
                                    "hotspot_strand_bias",
                                    "hotspot_strand_bias_pval",
                                    "downsample_to_coverage",
                                    "outlier_probability",
                                    "do_snp_realignment",
                                    "do_mnp_realignment",
                                    "realignment_threshold",
                                    "use_position_bias",
                                    "position_bias",
                                    "position_bias_pval",
                                    "position_bias_ref_fraction",
                                    "sse_prob_threshold",
                                    "prediction_precision",
                                    "heavy_tailed",
                                    "suppress_recalibration"]
sections["long_indel_assembler"]=["kmer_len",
                                  "min_var_count",
                                  "short_suffix_match",
                                  "min_indel_size",
                                  "max_hp_length",
                                  "min_var_freq",
                                  "relative_strand_bias",
                                  "output_mnv"]
sections["freebayes"]=["allow_indels",
                       "allow_snps",
                       "allow_mnps",
                       "allow_complex",
                       "min_mapping_qv",
                       "read_snp_limit",
                       "gen_min_alt_allele_freq",
                       "read_max_mismatch_fraction",
                       "gen_min_indel_alt_allele_freq",
                       "gen_min_coverage"]

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









