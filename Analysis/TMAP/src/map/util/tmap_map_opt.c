/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <config.h>
#include <assert.h>
#include "../../util/tmap_alloc.h"
#include "../../util/tmap_error.h"
#include "../../util/tmap_sort.h"
#include "../../io/tmap_file.h"
#include "../../seq/tmap_seq.h"
#include "../../util/tmap_progress.h"
#include "../../util/tmap_definitions.h"
#include "tmap_map_opt.h"

// For sorting the @RG fields
static int32_t 
tmap_map_opt_sort_rg_convert(const char *val) {
    if(strlen(val) < 2) return -1;
    else if('I' == val[0] && 'D' == val[1]) return 0;
    else if('C' == val[0] && 'N' == val[1]) return 1;
    else if('D' == val[0] && 'S' == val[1]) return 2;
    else if('D' == val[0] && 'T' == val[1]) return 3;
    else if('F' == val[0] && 'O' == val[1]) return 4;
    else if('K' == val[0] && 'S' == val[1]) return 5;
    else if('L' == val[0] && 'B' == val[1]) return 6;
    else if('P' == val[0] && 'G' == val[1]) return 7;
    else if('P' == val[0] && 'I' == val[1]) return 8;
    else if('P' == val[0] && 'L' == val[1]) return 9;
    else if('P' == val[0] && 'U' == val[1]) return 10;
    else if('S' == val[0] && 'M' == val[1]) return 11;
    else return -1;
}
#define tmap_map_opt_sort_rg_lt(_a, _b) (tmap_map_opt_sort_rg_convert(_a) < tmap_map_opt_sort_rg_convert(_b))
typedef char* tmap_map_opt_sort_rg_t;
TMAP_SORT_INIT(tmap_map_opt_sort_rg, tmap_map_opt_sort_rg_t, tmap_map_opt_sort_rg_lt)
  
static char *tmap_map_opt_input_types[] = {"INT", "FLOAT", "NUM", "FILE", "STRING", "NONE"};

// int32_t print function
#define __tmap_map_opt_option_print_func_int_init(_name) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s%d%s]%s", KMAG, KYEL, opt->_name, KMAG, KNRM); \
  }

// int32_t or message if default is used print function
#define __tmap_map_opt_option_print_func_int_or_msg_init(_name, _defval, _defmsg) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      if (opt->_name == _defval) \
          tmap_file_fprintf (tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, _defmsg, KMAG, KNRM); \
      else \
          tmap_file_fprintf (tmap_file_stderr, "%s[%s%d%s]%s", KMAG, KYEL, opt->_name, KMAG, KNRM); \
  }

  // int64_t print function
#define __tmap_map_opt_option_print_func_int64_init(_name) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s%ld%s]%s", KMAG, KYEL, opt->_name, KMAG, KNRM); \
  }

// int64_t or message if default is used print function
#define __tmap_map_opt_option_print_func_int64_or_msg_init(_name, _defval, _defmsg) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      if (opt->_name == _defval) \
          tmap_file_fprintf (tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, _defmsg, KMAG, KNRM); \
      else \
          tmap_file_fprintf (tmap_file_stderr, "%s[%s%ld%s]%s", KMAG, KYEL, opt->_name, KMAG, KNRM); \
  }

  // double print function
#define __tmap_map_opt_option_print_func_double_init(_name) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s%g%s]%s", KMAG, KYEL, opt->_name, KMAG, KNRM); \
  }

// double or message if default is used print function
#define __tmap_map_opt_option_print_func_double_or_msg_init(_name, _defval, _defmsg) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      if (opt->_name == _defval) \
          tmap_file_fprintf (tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, _defmsg, KMAG, KNRM); \
      else \
          tmap_file_fprintf (tmap_file_stderr, "%s[%s%g%s]%s", KMAG, KYEL, opt->_name, KMAG, KNRM); \
  }

// char array print function
#define __tmap_map_opt_option_print_func_chars_init(_name, _null_msg) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, (NULL == opt->_name) ? _null_msg : opt->_name, KMAG, KNRM); \
  } \

// int32_t print function
#define __tmap_map_opt_option_print_func_int_autodetected_init(_name, _detected) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      tmap_file_fprintf(tmap_file_stderr, "%s(%s%s%s)%s %s[%s%d%s]%s", KMAG, KYEL, (0 == opt->_detected) ? "user set" : "autodetect", \
                        KMAG, KNRM, KMAG, KYEL, \
                        opt->_name, KMAG, KNRM); \
  }

// true/false 
#define __tmap_map_opt_option_print_func_tf_init(_name) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      /* tmap_file_fprintf(tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, (1 == opt->_name) ? "true" : "false", KMAG, KNRM); */ \
  }


  // verbosity
#define __tmap_map_opt_option_print_func_verbosity_init() \
  static void tmap_map_opt_option_print_func_verbosity(void *arg) { \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, (1 == tmap_progress_get_verbosity()) ? "true" : "false", KMAG, KNRM); \
  }

// compression
#define __tmap_map_opt_option_print_func_compr_init(_name, _var, _compr) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, (_compr == opt->_var) ? "using" : "not using", KMAG, KNRM); \
  }

// number/probability
#define __tmap_map_opt_option_print_func_np_init(_name_num, _name_prob) \
  static void tmap_map_opt_option_print_func_##_name_num(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      if(opt->_name_num < 0) { \
          tmap_file_fprintf(tmap_file_stderr, "%s[probability: %s%lf%s]%s", KMAG, KYEL, opt->_name_prob, KMAG, KNRM); \
      } \
      else { \
          tmap_file_fprintf(tmap_file_stderr, "%s[number: %s%d%s]%s", KMAG, KYEL, opt->_name_num, KMAG, KNRM); \
      } \
  }


// number/fraction
#define __tmap_map_opt_option_print_func_nf_init(_name_num, _name_frac) \
  static void tmap_map_opt_option_print_func_##_name_num(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      if(opt->_name_num < 0) { \
          tmap_file_fprintf(tmap_file_stderr, "%s[fraction: %s%lf%s]%s", KMAG, KYEL, opt->_name_frac, KMAG, KNRM); \
      } \
      else { \
          tmap_file_fprintf(tmap_file_stderr, "%s[number: %s%d%s]%s", KMAG, KYEL, opt->_name_num, KMAG, KNRM); \
      } \
  }

// array of character arrays (i.e. list of file names)
#define __tmap_map_opt_option_print_func_char_array_init(_name_array, _name_length, _default) \
  static void tmap_map_opt_option_print_func_##_name_array(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      int32_t i; \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s", KMAG, KYEL); \
      if(0 == opt->_name_length) tmap_file_fprintf(tmap_file_stderr, _default); \
      else { \
          for(i=0;i<opt->_name_length;i++) { \
              if(0 < i) tmap_file_fprintf(tmap_file_stderr, ","); \
              tmap_file_fprintf(tmap_file_stderr, "%s", opt->_name_array[i]); \
          } \
      } \
      tmap_file_fprintf(tmap_file_stderr, "%s]%s", KMAG, KNRM); \
  }

// reads format
#define __tmap_map_opt_option_print_func_reads_format_init(_name) \
  static void tmap_map_opt_option_print_func_##_name(void *arg) { \
      tmap_map_opt_t *opt = (tmap_map_opt_t*)arg; \
      char *reads_format = tmap_get_reads_file_format_string(opt->_name); \
      tmap_file_fprintf(tmap_file_stderr, "%s[%s%s%s]%s", KMAG, KYEL, reads_format, KMAG, KNRM); \
      free(reads_format); \
  }

/*
 * Define the print functions for each opt.
 */
// global options
__tmap_map_opt_option_print_func_chars_init(fn_fasta, "not using")
__tmap_map_opt_option_print_func_chars_init(bed_file, "not using")
__tmap_map_opt_option_print_func_char_array_init(fn_reads, fn_reads_num, "stdin")
__tmap_map_opt_option_print_func_reads_format_init(reads_format)
__tmap_map_opt_option_print_func_chars_init(fn_sam, "stdout")
__tmap_map_opt_option_print_func_int64_init(bam_start_vfo)
__tmap_map_opt_option_print_func_int64_init(bam_end_vfo)
__tmap_map_opt_option_print_func_tf_init(use_param_ovr)
__tmap_map_opt_option_print_func_tf_init(ovr_candeval)
__tmap_map_opt_option_print_func_tf_init(use_bed_in_end_repair)
__tmap_map_opt_option_print_func_tf_init(use_bed_in_mapq)
__tmap_map_opt_option_print_func_tf_init(use_bed_read_ends_stat)
__tmap_map_opt_option_print_func_int_init(amplicon_scope)
__tmap_map_opt_option_print_func_int_init(score_match)
__tmap_map_opt_option_print_func_int_init(pen_mm)
__tmap_map_opt_option_print_func_int_init(pen_gapo)
__tmap_map_opt_option_print_func_int_init(pen_gape)
__tmap_map_opt_option_print_func_int_init(pen_gapl)
__tmap_map_opt_option_print_func_int_init(gapl_len)
__tmap_map_opt_option_print_func_int_init(bw)
__tmap_map_opt_option_print_func_int_init(softclip_type)
__tmap_map_opt_option_print_func_int_init(dup_window)
__tmap_map_opt_option_print_func_int_init(max_seed_band)
__tmap_map_opt_option_print_func_tf_init(unroll_banding)
__tmap_map_opt_option_print_func_double_init(long_hit_mult)
__tmap_map_opt_option_print_func_int_init(score_thr)
__tmap_map_opt_option_print_func_int_init(reads_queue_size)
__tmap_map_opt_option_print_func_int_autodetected_init(num_threads, num_threads_autodetected)
__tmap_map_opt_option_print_func_int_init(aln_output_mode)
__tmap_map_opt_option_print_func_char_array_init(sam_rg, sam_rg_num, "not using")
__tmap_map_opt_option_print_func_tf_init(bidirectional)
__tmap_map_opt_option_print_func_tf_init(seq_eq)
__tmap_map_opt_option_print_func_tf_init(ignore_rg_sam_tags)
__tmap_map_opt_option_print_func_tf_init(rand_read_name)
__tmap_map_opt_option_print_func_tf_init(prefix_exclude)
__tmap_map_opt_option_print_func_tf_init(suffix_exclude)
__tmap_map_opt_option_print_func_tf_init(use_new_QV)
__tmap_map_opt_option_print_func_compr_init(input_compr_gz, input_compr, TMAP_FILE_GZ_COMPRESSION)
__tmap_map_opt_option_print_func_compr_init(input_compr_bz2, input_compr, TMAP_FILE_BZ2_COMPRESSION)
__tmap_map_opt_option_print_func_int_init(output_type)
__tmap_map_opt_option_print_func_int_init(end_repair)
__tmap_map_opt_option_print_func_int_init(max_one_large_indel_rescue)
__tmap_map_opt_option_print_func_int_init(min_anchor_large_indel_rescue)
__tmap_map_opt_option_print_func_int_init(amplicon_overrun)
__tmap_map_opt_option_print_func_int_init(max_adapter_bases_for_soft_clipping)
__tmap_map_opt_option_print_func_tf_init(end_repair_5_prime_softclip)
__tmap_map_opt_option_print_func_int_init(repair_min_freq)
__tmap_map_opt_option_print_func_int_init(repair_min_count)
__tmap_map_opt_option_print_func_int_init(repair_min_adapter)
__tmap_map_opt_option_print_func_int_init(repair_max_overhang)
__tmap_map_opt_option_print_func_double_init(repair_identity_drop_limit)
__tmap_map_opt_option_print_func_int_init(repair_max_primer_zone_dist)
__tmap_map_opt_option_print_func_int_init(repair_clip_ext)



__tmap_map_opt_option_print_func_int_init(shm_key)
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
__tmap_map_opt_option_print_func_double_init(sample_reads)
#endif
__tmap_map_opt_option_print_func_int_init(vsw_type)
__tmap_map_opt_option_print_func_int_init(vsw_fallback)
__tmap_map_opt_option_print_func_tf_init(confirm_vsw_corr)
__tmap_map_opt_option_print_func_tf_init(candidate_ext)
__tmap_map_opt_option_print_func_tf_init(correct_failed_vsw)
__tmap_map_opt_option_print_func_tf_init(use_nvsw_on_nonstd_bases)

__tmap_map_opt_option_print_func_verbosity_init()

__tmap_map_opt_option_print_func_tf_init(do_realign)
__tmap_map_opt_option_print_func_int_init(realign_mat_score)
__tmap_map_opt_option_print_func_int_init(realign_mis_score)
__tmap_map_opt_option_print_func_int_init(realign_gip_score)
__tmap_map_opt_option_print_func_int_init(realign_gep_score)
__tmap_map_opt_option_print_func_int_init(realign_bandwidth)
__tmap_map_opt_option_print_func_int_init(realign_cliptype)
__tmap_map_opt_option_print_func_int_init(realign_maxlen)
__tmap_map_opt_option_print_func_int_init(realign_maxclip)

__tmap_map_opt_option_print_func_tf_init(report_stats)
__tmap_map_opt_option_print_func_chars_init(realign_log, "")
// __tmap_map_opt_option_print_func_tf_init(log_text_als)

// end tandem repeat clip
__tmap_map_opt_option_print_func_tf_init(do_repeat_clip)
// __tmap_map_opt_option_print_func_int_init(repclip_overlap)
__tmap_map_opt_option_print_func_tf_init(repclip_continuation)

__tmap_map_opt_option_print_func_int_init(cigar_sanity_check)


// context-dependent gaps
__tmap_map_opt_option_print_func_tf_init(do_hp_weight)
__tmap_map_opt_option_print_func_int_init(gap_scale_mode)
__tmap_map_opt_option_print_func_double_init(context_mat_score)
__tmap_map_opt_option_print_func_double_init(context_mis_score)
__tmap_map_opt_option_print_func_double_init(context_gip_score)
__tmap_map_opt_option_print_func_double_init(context_gep_score)
__tmap_map_opt_option_print_func_int_init(context_extra_bandwidth)
__tmap_map_opt_option_print_func_int_init(debug_log)
// __tmap_map_opt_option_print_func_int_init(context_noclip)

// alignment length filtering
__tmap_map_opt_option_print_func_int_or_msg_init(min_al_len, MIN_AL_LEN_NOCHECK_SPECIAL, "Not filtered")
__tmap_map_opt_option_print_func_double_or_msg_init(min_al_cov, MIN_AL_COVERAGE_NOCHECK_SPECIAL, "Not filtered")
__tmap_map_opt_option_print_func_double_or_msg_init(min_identity, MIN_AL_IDENTITY_NOCHECK_SPECIAL, "Not filtered")

// flowspace
__tmap_map_opt_option_print_func_int_init(fscore)
__tmap_map_opt_option_print_func_tf_init(softclip_key)
__tmap_map_opt_option_print_func_tf_init(sam_flowspace_tags)
__tmap_map_opt_option_print_func_tf_init(ignore_flowgram)
__tmap_map_opt_option_print_func_tf_init(aln_flowspace)
// pairing
__tmap_map_opt_option_print_func_int_init(pairing)
__tmap_map_opt_option_print_func_int_init(strandedness)
__tmap_map_opt_option_print_func_int_init(positioning)
__tmap_map_opt_option_print_func_double_init(ins_size_mean)
__tmap_map_opt_option_print_func_double_init(ins_size_std)
__tmap_map_opt_option_print_func_double_init(ins_size_std_max_num)
__tmap_map_opt_option_print_func_double_init(ins_size_outlier_bound)
__tmap_map_opt_option_print_func_int_init(ins_size_min_mapq)
__tmap_map_opt_option_print_func_tf_init(read_rescue)
__tmap_map_opt_option_print_func_double_init(read_rescue_std_num)
__tmap_map_opt_option_print_func_int_init(read_rescue_mapq_thr)
// map1/map2/map3 options, but specific to each
__tmap_map_opt_option_print_func_int_init(min_seq_len)
__tmap_map_opt_option_print_func_int_init(max_seq_len)
// map1/map3 options
__tmap_map_opt_option_print_func_int_init(seed_length)
// map2/map3 options
__tmap_map_opt_option_print_func_int_init(max_seed_hits)
// map3/map4 options
__tmap_map_opt_option_print_func_double_init(hit_frac)
__tmap_map_opt_option_print_func_int_init(seed_step)
// map1 options
__tmap_map_opt_option_print_func_int_init(seed_max_diff)
__tmap_map_opt_option_print_func_int_init(seed2_length)
__tmap_map_opt_option_print_func_np_init(max_diff, max_diff_fnr)
__tmap_map_opt_option_print_func_double_init(max_err_rate)
__tmap_map_opt_option_print_func_nf_init(max_mm, max_mm_frac)
__tmap_map_opt_option_print_func_nf_init(max_gapo, max_gapo_frac)
__tmap_map_opt_option_print_func_nf_init(max_gape, max_gape_frac)
__tmap_map_opt_option_print_func_int_init(max_cals_del)
__tmap_map_opt_option_print_func_int_init(indel_ends_bound)
__tmap_map_opt_option_print_func_int_init(max_best_cals)
__tmap_map_opt_option_print_func_int_init(max_entries)
// map2 options
__tmap_map_opt_option_print_func_double_init(length_coef)
__tmap_map_opt_option_print_func_int_init(max_seed_intv)
__tmap_map_opt_option_print_func_int_init(z_best)
__tmap_map_opt_option_print_func_int_init(seeds_rev)
__tmap_map_opt_option_print_func_tf_init(narrow_rmdup)
__tmap_map_opt_option_print_func_int_init(max_chain_gap)
// map3 options
__tmap_map_opt_option_print_func_int_init(hp_diff)
__tmap_map_opt_option_print_func_tf_init(fwd_search)
__tmap_map_opt_option_print_func_double_init(skip_seed_frac)
// map4 options
__tmap_map_opt_option_print_func_int_init(min_seed_length)
__tmap_map_opt_option_print_func_int_init(max_seed_length)
__tmap_map_opt_option_print_func_double_init(max_seed_length_adj_coef)
__tmap_map_opt_option_print_func_int_init(max_iwidth)
__tmap_map_opt_option_print_func_int_init(max_repr)
__tmap_map_opt_option_print_func_tf_init(rand_repr)
__tmap_map_opt_option_print_func_tf_init(use_min)
// mapvsw options
// stage options
__tmap_map_opt_option_print_func_int_init(stage_score_thr)
__tmap_map_opt_option_print_func_int_init(stage_mapq_thr)
__tmap_map_opt_option_print_func_tf_init(stage_keep_all)
__tmap_map_opt_option_print_func_double_init(stage_seed_freqc)
__tmap_map_opt_option_print_func_double_init(stage_seed_freqc_group_frac)
__tmap_map_opt_option_print_func_int_init(stage_seed_freqc_rand_repr)
__tmap_map_opt_option_print_func_int_init(stage_seed_freqc_min_groups)
__tmap_map_opt_option_print_func_int_init(stage_seed_max_length)

static int32_t
tmap_map_opt_option_flag_length(tmap_map_opt_option_t *opt)
{
  int32_t flag_length = 0;
  if(0 < opt->option.val) flag_length += 3; // dash, char, comma
  if(NULL != opt->name) flag_length += 2 + strlen(opt->name);
  return flag_length;
}

static int32_t 
tmap_map_opt_option_type_length(tmap_map_opt_option_t *opt)
{
  int32_t type_length = 0;
  if(TMAP_MAP_OPT_TYPE_NONE != opt->type) {
      type_length = strlen(tmap_map_opt_input_types[opt->type]);
  }
  return type_length;
}

static void
tmap_map_opt_option_print(tmap_map_opt_option_t *opt, tmap_map_opt_t *parent_opt)
{
  int32_t i, j, flag_length, type_length, length_to_description = 0;
  static char *spacer = "    ";

  flag_length = tmap_map_opt_option_flag_length(opt);
  type_length = tmap_map_opt_option_type_length(opt);

  if(NULL == opt->option.name) {
      tmap_error("option did not have a name", Exit, OutOfRange);
  }

  // spacer
  length_to_description += tmap_file_fprintf(tmap_file_stderr, spacer);
  // short flag, if available
  if(0 < opt->option.val) {
      length_to_description += tmap_file_fprintf(tmap_file_stderr, "%s-%c,%s", KCYN, (char)opt->option.val, KNRM);
  }
  // long flag
  length_to_description += tmap_file_fprintf(tmap_file_stderr, "%s--%s%s", KCYN, opt->option.name, KNRM);
  if(NULL != parent_opt) {
      for(i=flag_length;i< parent_opt->options->max_flag_length;i++) {
          length_to_description += tmap_file_fprintf(tmap_file_stderr, " ");
      }
  }
  length_to_description += tmap_file_fprintf(tmap_file_stderr, " ");
  // type
  length_to_description += tmap_file_fprintf(tmap_file_stderr, "%s%s%s", KWHT,
                    (TMAP_MAP_OPT_TYPE_NONE == opt->type) ? "" : tmap_map_opt_input_types[opt->type], KNRM);
  if(NULL != parent_opt) {
      for(i=type_length;i<parent_opt->options->max_type_length;i++) {
          length_to_description += tmap_file_fprintf(tmap_file_stderr, " ");
      }
  }
  // spacer
  length_to_description += tmap_file_fprintf(tmap_file_stderr, spacer);
  // description
  tmap_file_fprintf(tmap_file_stderr, "%s%s%s", KGRN,
                    opt->description, KNRM);
  // value, if available
  if(NULL != opt->print_func && NULL != parent_opt) {
      tmap_file_fprintf(tmap_file_stderr, " ");
      opt->print_func(parent_opt);
  }
  tmap_file_fprintf(tmap_file_stderr, "\n");
  // multi-option description
  if(NULL != opt->multi_options) {
      for(i=0;NULL != opt->multi_options[i];i++) {
          // spacers
          for(j=0;j<length_to_description;j++) {
              tmap_file_fprintf(tmap_file_stderr, " ");
          }
          tmap_file_fprintf(tmap_file_stderr, spacer);
          tmap_file_fprintf(tmap_file_stderr, "%s%s%s\n", KGRN, opt->multi_options[i], KNRM);
      }
  }
}

static tmap_map_opt_options_t *
tmap_map_opt_options_init()
{
  return tmap_calloc(1, sizeof(tmap_map_opt_options_t), "");
}

/*
  tmap_map_opt_options_add(opt->options, "seed-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the k-mer length to seed CALs (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_seed_length,
                           TMAP_MAP_ALGO_MAP1 | TMAP_MAP_ALGO_MAP3);
*/

static void
tmap_map_opt_options_add(tmap_map_opt_options_t *options, const char *name, 
                         int32_t has_arg, int32_t *flag, int32_t val, int32_t type, 
                         char *description, char **multi_options,
                         tmap_map_opt_option_print_t print_func,
                         int32_t algos)
{
  int32_t flag_length = 0, type_length = 0;
  while(options->mem <= options->n) {
      if(0 == options->mem) {
          options->mem = 4;
      }
      else {
          options->mem *= 2;
      }
      options->options = tmap_realloc(options->options, sizeof(tmap_map_opt_option_t) * options->mem, "options->options");
  }
  
  if(NULL == name) {
      tmap_error("option did not have a name", Exit, OutOfRange);
  }
  if(NULL == description) {
      tmap_error("option did not have a description", Exit, OutOfRange);
  }
  options->options[options->n].name = tmap_strdup(name);
  options->options[options->n].option.name = options->options[options->n].name;
  options->options[options->n].option.has_arg = has_arg;
  options->options[options->n].option.flag = flag;
  options->options[options->n].option.val = val;
  options->options[options->n].type = type;
  options->options[options->n].description = (NULL == description) ? NULL : tmap_strdup(description);
  options->options[options->n].multi_options = multi_options;
  options->options[options->n].print_func = (NULL == print_func) ? NULL : print_func;
  options->options[options->n].algos = algos;

  flag_length = tmap_map_opt_option_flag_length(&options->options[options->n]);
  if(options->max_flag_length < flag_length) {
      options->max_flag_length = flag_length;
  }

  type_length = tmap_map_opt_option_type_length(&options->options[options->n]);
  if(options->max_type_length < type_length) {
      options->max_type_length = type_length;
  }

  options->n++;
}

static void
tmap_map_opt_options_destroy(tmap_map_opt_options_t *options)
{
  int32_t i;
  for(i=0;i<options->n;i++) {
      free(options->options[i].name);
      free(options->options[i].description);
  }
  free(options->options);
  free(options);
}

static void
tmap_map_opt_init_helper(tmap_map_opt_t *opt)
{
  static char *softclipping_type[] = {
      "0 - allow on the left and right portion of the read",
      "1 - allow on the left portion of the read",
      "2 - allow on the right portion of the read",
      "3 - do not allow soft-clipping",
      NULL};
  static char *aln_output_mode[] = {"0 - unique best hits",
      "1 - random best hit",
      "2 - all best hits",
      "3 - all alignments",
      NULL};
  static char *vsw_type[] = {
      "NB: currently only #1, #2, #4, and #6 have been tested",
      "1 - lh3/ksw.c/nh13",
      "2 - simple VSW",
      "3 - SHRiMP2 VSW [not working]",
      "4 - Psyho (Top Coder #1)",
      "5 - ACRush (Top Coder #2)",
      "6 - folsena (Top Coder #3)",
      "7 - logicmachine (Top Coder #4)",
      "8 - venco (Top Coder #5) [not working]",
      "9 - Bladze (Top Coder #6)",
      "10 - ngthuydiem (Top Coder #7) [Farrar cut-and-paste]",
      NULL};
  static char *realignment_clip_type[] = {
      "0 - global realignment",
      "1 - semiglobal (can start/end anywhere in reference)",
      "2 - semiglobal with soft clip on bead side of a read",
      "3 - semiglobal with soft clip on key side of a read",
      "4 - local (semiglobal with soft clip on both sides of a read",
      NULL};
  static char *gap_scale_modes [] = {"0 - no scaling",
      "1 - scale gap extension cost",
      "2 - scale gap initiation and gap extension costs",
      NULL};
  static char *output_type[] = {"0 - SAM", "1 - BAM (compressed)", "2 - BAM (uncompressed)", NULL};
  static char *pairing[] = {"0 - no pairing is to be performed", "1 - mate pairs (-S 0 -P 1)", "2 - paired end (-S 1 -P 0)", NULL};
  static char *strandedness[] = {"0 - same strand", "1 - opposite strand", NULL};
  static char *positioning[] = {"0 - read one before read two", "1 - read two before read one", NULL};
  static char *end_repair[] = {"0 - disable", "1 - prefer mismatches", "2 - prefer indels", ">2 - specify %% Mismatch above which to trim end alignment", NULL};
  static char *sanity_check_outcome [] = {
      "0 - do not perform sanity check",
      "1 - perform content checks, print warning to stderr; do not check alignment compatibility or scores",
      "2 - perform content and alignment compatibility checks, print warnings to stderr; do not check alignment scores",
      "3 - perform all checks, print warnings to stderr",
      "4 - perform content checks, exit on error",
      "5 - perform content checks, exit on error; warn if processed alignment is incompatible with raw one",
      "6 - perform content and alignment compatibility checks, exit on error",
      "7 - perform content checks, exit on error; warn if processed alignment is incompatible with raw one or if score is suspicious",
      "8 - perform content and alignment compatibility checks, exit on error; warn if score is suspicious",
      "9 - perform all checks, exit if any of them fails",
      NULL};

  opt->options = tmap_map_opt_options_init();

  // global options
  tmap_map_opt_options_add(opt->options, "fn-fasta", required_argument, 0, 'f', 
                           TMAP_MAP_OPT_TYPE_FILE,
                           "FASTA reference file name",
                           NULL,
                           tmap_map_opt_option_print_func_fn_fasta,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "bed-file", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_FILE,
                           "bed file for AmpliSeq",
                           NULL,
                           tmap_map_opt_option_print_func_bed_file,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "fn-reads", required_argument, 0, 'r', 
                           TMAP_MAP_OPT_TYPE_FILE,
                           "the reads file name", 
                           NULL,
                           tmap_map_opt_option_print_func_fn_reads,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "reads-format", required_argument, 0, 'i', 
                           TMAP_MAP_OPT_TYPE_STRING,
                           "the reads file format (fastq|fq|fasta|fa|sff|sam|bam)",
                           NULL,
                           tmap_map_opt_option_print_func_reads_format,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "fn-sam", required_argument, 0, 's', 
                           TMAP_MAP_OPT_TYPE_FILE,
                           "the SAM file name",
                           NULL,
                           tmap_map_opt_option_print_func_fn_sam,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "bam-start-vfo", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "sets starting virtual file offsets that limit the range of BAM reads that will be processed",
                           NULL,
                           tmap_map_opt_option_print_func_bam_start_vfo,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "bam-end-vfo", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "sets ending virtual file offsets that limit the range of BAM reads that will be processed",
                           NULL,
                           tmap_map_opt_option_print_func_bam_end_vfo,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "par-ovr", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "use location-specific parameters overriding if provided in BED file",
                           NULL,
                           tmap_map_opt_option_print_func_use_param_ovr,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "ovr-candeval", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "use location-specific parameters overriding at candidate evaluation stage",
                           NULL,
                           tmap_map_opt_option_print_func_ovr_candeval,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "no-bed-er", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "use of amplicon edge coordinates from BED file in end repair",
                           NULL,
                           tmap_map_opt_option_print_func_use_bed_in_end_repair,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "no-bed-mapq", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "use of amplicon edge coordinates from BED file in map quality calculation",
                           NULL,
                           tmap_map_opt_option_print_func_use_bed_in_mapq,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "repair", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "use read ends statistics for REPAiR (read-end position alignment repair) if provided in BED file",
                           NULL,
                           tmap_map_opt_option_print_func_use_bed_read_ends_stat,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "ampl-scope", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "read padding used for amplicon search",
                           NULL,
                           tmap_map_opt_option_print_func_amplicon_scope,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "score-match", required_argument, 0, 'A', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "score for a match",
                           NULL,
                           tmap_map_opt_option_print_func_score_match,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "pen-mismatch", required_argument, 0, 'M', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the mismatch penalty",
                           NULL,
                           tmap_map_opt_option_print_func_pen_mm,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "pen-gap-open", required_argument, 0, 'O', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the indel start penalty",
                           NULL,
                           tmap_map_opt_option_print_func_pen_gapo,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "pen-gap-extension", required_argument, 0, 'E', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the indel extension penalty",
                           NULL,
                           tmap_map_opt_option_print_func_pen_gape,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "pen-gap-long", required_argument, 0, 'G', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the long indel penalty",
                           NULL,
                           tmap_map_opt_option_print_func_pen_gapl,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "gap-long-length", required_argument, 0, 'K', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the number of extra bases to add when searching for long indels",
                           NULL,
                           tmap_map_opt_option_print_func_gapl_len,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "band-width", required_argument, 0, 'w', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the band width",
                           NULL,
                           tmap_map_opt_option_print_func_bw,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "softclip-type", required_argument, 0, 'g', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the soft-clipping type",
                           softclipping_type,
                           tmap_map_opt_option_print_func_softclip_type,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "duplicate-window", required_argument, 0, 'W', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "remove duplicate alignments within this bp window (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_dup_window,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "max-seed-band", required_argument, 0, 'B', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the window of bases in which to group seeds",
                           NULL,
                           tmap_map_opt_option_print_func_max_seed_band,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "unroll-banding", no_argument, 0, 'U', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "unroll the grouped seeds from banding if multiple alignments are found",
                           NULL,
                           tmap_map_opt_option_print_func_unroll_banding,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "long-hit-mult", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the multiplier of the query length for a minimum target length to identify a repetitive group",
                           NULL,
                           tmap_map_opt_option_print_func_long_hit_mult,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "score-thres", required_argument, 0, 'T', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "score threshold divided by the match score",
                           NULL,
                           tmap_map_opt_option_print_func_score_thr,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "reads-queue-size", required_argument, 0, 'q', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the queue size for the reads (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_reads_queue_size,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "num-threads", required_argument, 0, 'n', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the number of threads",
                           NULL,
                           tmap_map_opt_option_print_func_num_threads,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "aln-output-mode", required_argument, 0, 'a', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "output filter",
                           aln_output_mode,
                           tmap_map_opt_option_print_func_aln_output_mode,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "sam-read-group", required_argument, 0, 'R', 
                           TMAP_MAP_OPT_TYPE_STRING,
                           "the RG tags to add to the SAM header",
                           NULL,
                           tmap_map_opt_option_print_func_sam_rg,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "bidirectional", no_argument, 0, 'D', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "specifies the input reads are to be annotated as bidirectional",
                           NULL,
                           tmap_map_opt_option_print_func_bidirectional,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "use-seq-equal", no_argument, 0, 'I', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "specifies to use the '=' symbol in the SEQ field",
                           NULL,
                           tmap_map_opt_option_print_func_seq_eq,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "ignore-rg-from-sam", no_argument, 0, 'C', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "specifies to not use the RG header and RG record tags in the SAM file",
                           NULL,
                           tmap_map_opt_option_print_func_ignore_rg_sam_tags,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "rand-read-name", no_argument, 0, 'u', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "specifies to randomize based on the read name",
                           NULL,
                           tmap_map_opt_option_print_func_rand_read_name,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "prefix-exclude", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "specify how many letters of prefix of name to be excluded when do randomize by name",
                           NULL,
                           tmap_map_opt_option_print_func_prefix_exclude,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "suffix-exclude", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "specify how many letters of suffix of name to be excluded when do randomize by name",
                           NULL,
                           tmap_map_opt_option_print_func_suffix_exclude,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "newQV", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "specify whether to use new mapping QV formula",
                           NULL,
                           tmap_map_opt_option_print_func_use_new_QV,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "input-gz", no_argument, 0, 'z', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "the input is gz (gzip) compressed",
                           NULL,
                           tmap_map_opt_option_print_func_input_compr_gz,
                           TMAP_MAP_ALGO_GLOBAL);
#ifndef DISABLE_BZ2
  tmap_map_opt_options_add(opt->options, "input-bz2", no_argument, 0, 'j', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "the input is bz2 (bzip2) compressed",
                           NULL,
                           tmap_map_opt_option_print_func_input_compr_bz2,
                           TMAP_MAP_ALGO_GLOBAL);
#endif
  tmap_map_opt_options_add(opt->options, "output-type", required_argument, 0, 'o', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the output type",
                           output_type,
                           tmap_map_opt_option_print_func_output_type,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "end-repair", required_argument, 0, 0 /* no short flag */, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "specifies to perform end repair",
                           end_repair,
                           tmap_map_opt_option_print_func_end_repair,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "max-one-large-indel-rescue", required_argument, 0, 0 /* no short flag */,
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum indel size to rescue with one large indel algorithm",
                           NULL, 
                           tmap_map_opt_option_print_func_max_one_large_indel_rescue,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "min-anchor-large-indel-rescue", required_argument, 0, 0 /* no short flag */,
                           TMAP_MAP_OPT_TYPE_INT,
                           "the minimum anchor  size to rescue with one large indel algorithm",
                           NULL,
                           tmap_map_opt_option_print_func_min_anchor_large_indel_rescue,
                           TMAP_MAP_ALGO_GLOBAL);
   tmap_map_opt_options_add(opt->options, "max-amplicon-overrun-large-indel-rescue", required_argument, 0, 0 /* no short flag */,
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum number of bases allowed for a read to overrun the end of amplicon",
                           NULL,
                           tmap_map_opt_option_print_func_amplicon_overrun,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "max-adapter-bases-for-soft-clipping", required_argument, 0, 'J',
                           TMAP_MAP_OPT_TYPE_INT,
                           "specifies to perform 3' soft-clipping (via -g) if at most this # of adapter bases were found (ZB tag)",
                           NULL,
                           tmap_map_opt_option_print_func_max_adapter_bases_for_soft_clipping,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "er-no5clip", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "disable 5' soft clipping by end-repair",
                           NULL,
                           tmap_map_opt_option_print_func_end_repair_5_prime_softclip,
                           TMAP_MAP_ALGO_GLOBAL);

  tmap_map_opt_options_add(opt->options, "repair-min-freq", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "minimal frequency sum for REPAiR (read-end position alignment repair) ",
                           NULL,
                           tmap_map_opt_option_print_func_repair_min_freq,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "repair-min-count", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "minimal read count for REPAiR (read-end position alignment repair) ",
                           NULL,
                           tmap_map_opt_option_print_func_repair_min_count,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "repair-min-adapter", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "Minimal adapter size (ZB tag) for REPAiR (read-end position alignment repair) ",
                           NULL,
                           tmap_map_opt_option_print_func_repair_min_adapter,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "repair-max-overhang", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "maximal distance from the template end to the amplicon end (ampl_len - ZA) for REPAiR (read-end position alignment repair) ",
                           NULL,
                           tmap_map_opt_option_print_func_repair_max_overhang,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "repair-identity-drop-limit", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the identity score of the newly aligned zone should be above IDENTITY_DROP_LIMIT*(removed_portion_identity) ",
                           NULL,
                           tmap_map_opt_option_print_func_repair_identity_drop_limit,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "repair-max-primer-zone-dist", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "maximal number of errors in the primer zone (between amplicon end and the read end if read end is inside amplicon)",
                           NULL,
                           tmap_map_opt_option_print_func_repair_max_primer_zone_dist,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "repair-clip-ext", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "minimal number of bases to extend the clip in REPAiR",
                           NULL,
                           tmap_map_opt_option_print_func_repair_clip_ext,
                           TMAP_MAP_ALGO_GLOBAL);

  tmap_map_opt_options_add(opt->options, "shared-memory-key", required_argument, 0, 'k', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "use shared memory with the following key",
                           NULL,
                           tmap_map_opt_option_print_func_shm_key,
                           TMAP_MAP_ALGO_GLOBAL);
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  tmap_map_opt_options_add(opt->options, "sample-reads", required_argument, 0, 'x',
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "sample the reads at this fraction",
                           NULL,
                           tmap_map_opt_option_print_func_sample_reads,
                           TMAP_MAP_ALGO_GLOBAL);
#endif
  tmap_map_opt_options_add(opt->options, "vsw-type", required_argument, 0, 'H',
                           TMAP_MAP_OPT_TYPE_INT,
                           "the vectorized smith-waterman algorithm (very untested)",
                           vsw_type,
                           tmap_map_opt_option_print_func_vsw_type,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "vsw-fallback", required_argument, 0, 'H',
                           TMAP_MAP_OPT_TYPE_INT,
                           "the fallback vectorized smith-waterman algorithm",
                           NULL,
                           tmap_map_opt_option_print_func_vsw_fallback,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "confirm-vsw-corr", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "confirm corrections of assymetric VSW scores with standard SW",
                           NULL,
                           tmap_map_opt_option_print_func_confirm_vsw_corr,
                           TMAP_MAP_ALGO_GLOBAL);
 tmap_map_opt_options_add(opt->options, "no-candidate-ext", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "do not extend reference zone for partially aligned candidates",
                           NULL,
                           tmap_map_opt_option_print_func_candidate_ext,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "no-vsw-fallback", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "do not use fallback alignment method if primary method fails",
                           NULL,
                           tmap_map_opt_option_print_func_correct_failed_vsw,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "no-nonstd-bases", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "do not process candidate locations containing non-standard nucleotides",
                           NULL,
                           tmap_map_opt_option_print_func_use_nvsw_on_nonstd_bases,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "help", no_argument, 0, 'h', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "print this message",
                           NULL,
                           NULL,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "verbose", no_argument, 0, 'v', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "print verbose progress information",
                           NULL,
                           tmap_map_opt_option_print_func_verbosity,
                           TMAP_MAP_ALGO_GLOBAL);
  
  // realignment options
  tmap_map_opt_options_add(opt->options, "do-realign", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "perform realignment of the found matches",
                           NULL,
                           tmap_map_opt_option_print_func_do_realign,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-mat", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment match score",
                           NULL,
                           tmap_map_opt_option_print_func_realign_mat_score,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-mis", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment mismatch score",
                           NULL,
                           tmap_map_opt_option_print_func_realign_mis_score,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-gip", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment gap opening score",
                           NULL,
                           tmap_map_opt_option_print_func_realign_gip_score,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-gep", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment gap extension score",
                           NULL,
                           tmap_map_opt_option_print_func_realign_gep_score,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-bw", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment bandwidth",
                           NULL,
                           tmap_map_opt_option_print_func_realign_bandwidth,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-clip", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment clip type",
                           realignment_clip_type,
                           tmap_map_opt_option_print_func_realign_cliptype,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-maxlen", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment maximal supported sequence length",
                           NULL,
                           tmap_map_opt_option_print_func_realign_maxlen,
                           TMAP_MAP_ALGO_GLOBAL);
  tmap_map_opt_options_add(opt->options, "r-maxclip", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "realignment maximal supported clip length",
                           NULL,
                           tmap_map_opt_option_print_func_realign_maxclip,
                           TMAP_MAP_ALGO_GLOBAL);

  tmap_map_opt_options_add(opt->options, "stats", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "report processing statistics",
                           NULL,
                           tmap_map_opt_option_print_func_report_stats,
                           TMAP_MAP_ALGO_GLOBAL);

  tmap_map_opt_options_add(opt->options, "log", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FILE,
                           "the realignment log file name",
                           NULL,
                           tmap_map_opt_option_print_func_realign_log,
                           TMAP_MAP_ALGO_GLOBAL);
/*
  tmap_map_opt_options_add(opt->options, "text-als", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "include textual alignments in realignment log",
                           NULL,
                           tmap_map_opt_option_print_func_log_text_als,
                           TMAP_MAP_ALGO_GLOBAL);
*/
  // alignment end repeat clipping
  tmap_map_opt_options_add(opt->options, "do-repeat-clip", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "clip tandem repeats from the alignment 3' ends",
                           NULL,
                           tmap_map_opt_option_print_func_do_repeat_clip,
                           TMAP_MAP_ALGO_GLOBAL);

  tmap_map_opt_options_add(opt->options, "repclip-cont", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "clip only repeats continued into the reference for at least one period",
                           NULL,
                           tmap_map_opt_option_print_func_repclip_continuation,
                           TMAP_MAP_ALGO_GLOBAL);



  tmap_map_opt_options_add(opt->options, "cigar-sanity-check", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "perform diagnostics sanity check on all generated alignments",
                           sanity_check_outcome,
                           tmap_map_opt_option_print_func_cigar_sanity_check,
                           TMAP_MAP_ALGO_GLOBAL);


  // context-dependent indel weights
  tmap_map_opt_options_add(opt->options, "context", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "realign with context-dependent gap scores",
                           NULL,
                           tmap_map_opt_option_print_func_do_hp_weight,
                           TMAP_MAP_ALGO_GLOBAL);
  // hp gap scaling mode
  tmap_map_opt_options_add(opt->options, "gap-scale", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "gaps over homopolymers score scale mode",
                           gap_scale_modes,
                           tmap_map_opt_option_print_func_gap_scale_mode,
                           TMAP_MAP_ALGO_GLOBAL);
  // context match score
  tmap_map_opt_options_add(opt->options, "c-mat", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "context match score",
                           NULL,
                           tmap_map_opt_option_print_func_context_mat_score,
                           TMAP_MAP_ALGO_GLOBAL);
  // context mismatch penalty
  tmap_map_opt_options_add(opt->options, "c-mis", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "context mismatch score",
                           NULL,
                           tmap_map_opt_option_print_func_context_mis_score,
                           TMAP_MAP_ALGO_GLOBAL);
  // context gap initiation penalty
  tmap_map_opt_options_add(opt->options, "c-gip", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "context gap opening score",
                           NULL,
                           tmap_map_opt_option_print_func_context_gip_score,
                           TMAP_MAP_ALGO_GLOBAL);
  // context gap extension penalty
  tmap_map_opt_options_add(opt->options, "c-gep", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "context gap extension score",
                           NULL,
                           tmap_map_opt_option_print_func_context_gep_score,
                           TMAP_MAP_ALGO_GLOBAL);
  // context realignment extra bandwidth
  tmap_map_opt_options_add(opt->options, "c-bw", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "context bandwidth",
                           NULL,
                           tmap_map_opt_option_print_func_context_extra_bandwidth,
                           TMAP_MAP_ALGO_GLOBAL);
  // context debug log
  tmap_map_opt_options_add(opt->options, "debug-log", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "log alignment post-processing details into the log file",
                           NULL,
                           tmap_map_opt_option_print_func_debug_log,
                           TMAP_MAP_ALGO_GLOBAL);

  // filtering by alignment length
  tmap_map_opt_options_add(opt->options, "min-al-len", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "Filter out alignments shorter then given length",
                           NULL,
                           tmap_map_opt_option_print_func_min_al_len,
                           TMAP_MAP_ALGO_GLOBAL);
  // filtering by alignment coverage
  tmap_map_opt_options_add(opt->options, "min-cov", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "Filter out alignments which cover less then given fraction of total read length",
                           NULL,
                           tmap_map_opt_option_print_func_min_al_cov,
                           TMAP_MAP_ALGO_GLOBAL);
  // filtering by alignment identity
  tmap_map_opt_options_add(opt->options, "min-iden", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "Filter out alignments whith identity fraction below given one",
                           NULL,
                           tmap_map_opt_option_print_func_min_identity,
                           TMAP_MAP_ALGO_GLOBAL);
  
  // flowspace options
  tmap_map_opt_options_add(opt->options, "pen-flow-error", required_argument, 0, 'X', 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the flow score penalty",
                           NULL,
                           tmap_map_opt_option_print_func_fscore,
                           TMAP_MAP_ALGO_FLOWSPACE);
  tmap_map_opt_options_add(opt->options, "softclip-key", no_argument, 0, 'y', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "soft clip only the last base of the key",
                           NULL,
                           tmap_map_opt_option_print_func_softclip_key,
                           TMAP_MAP_ALGO_FLOWSPACE);
  tmap_map_opt_options_add(opt->options, "sam-flowspace-tags", no_argument, 0, 'Y', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "include flow space specific SAM tags when available",
                           NULL,
                           tmap_map_opt_option_print_func_sam_flowspace_tags,
                           TMAP_MAP_ALGO_FLOWSPACE);
  tmap_map_opt_options_add(opt->options, "ignore-flowgram", no_argument, 0, 'N', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "do not use the flowgram, otherwise use the flowgram when available",
                           NULL,
                           tmap_map_opt_option_print_func_ignore_flowgram,
                           TMAP_MAP_ALGO_FLOWSPACE);
  tmap_map_opt_options_add(opt->options, "final-flowspace", no_argument, 0, 'F', 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "produce the final alignment in flow space",
                           NULL,
                           tmap_map_opt_option_print_func_aln_flowspace,
                           TMAP_MAP_ALGO_FLOWSPACE);

  // pairing options
  tmap_map_opt_options_add(opt->options, "pairing", required_argument, 0, 'Q',
                           TMAP_MAP_OPT_TYPE_INT,
                           "the insert pairing",
                           pairing,
                           tmap_map_opt_option_print_func_pairing,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "strandedness", required_argument, 0, 'S',
                           TMAP_MAP_OPT_TYPE_INT,
                           "the insert strandedness",
                           strandedness,
                           tmap_map_opt_option_print_func_strandedness,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "positioning", required_argument, 0, 'P',
                           TMAP_MAP_OPT_TYPE_INT,
                           "the insert positioning when on the same strand (-S 0)",
                           positioning,
                           tmap_map_opt_option_print_func_positioning,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "ins-size-mean", required_argument, 0, 'b',
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the mean insert size",
                           NULL,
                           tmap_map_opt_option_print_func_ins_size_mean,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "ins-size-std", required_argument, 0, 'c',
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the insert size standard deviation",
                           NULL,
                           tmap_map_opt_option_print_func_ins_size_std,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "ins-size-std-max-num", required_argument, 0, 'd',
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the insert size maximum standard deviation",
                           NULL,
                           tmap_map_opt_option_print_func_ins_size_std_max_num,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "ins-size-outlier-bound", required_argument, 0, 'p',
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the insert size 25/75 quartile outlier bound",
                           NULL,
                           tmap_map_opt_option_print_func_ins_size_outlier_bound,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "ins-size-min-mapq", required_argument, 0, 't',
                           TMAP_MAP_OPT_TYPE_INT,
                           "the minimum mapping quality to consider for computing the insert size",
                           NULL,
                           tmap_map_opt_option_print_func_ins_size_min_mapq,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "read-rescue", no_argument, 0, 'L',
                           TMAP_MAP_OPT_TYPE_NONE,
                           "perform read rescue",
                           NULL,
                           tmap_map_opt_option_print_func_read_rescue,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "read-rescue-std-num", required_argument, 0, 'l',
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the number of standard deviations around the mean insert size to perform read rescue",
                           NULL,
                           tmap_map_opt_option_print_func_read_rescue_std_num,
                           TMAP_MAP_ALGO_PAIRING);
  tmap_map_opt_options_add(opt->options, "read-rescue-mapq-thr", required_argument, 0, 'm',
                           TMAP_MAP_OPT_TYPE_INT,
                           "mapping quality threshold for read rescue",
                           NULL,
                           tmap_map_opt_option_print_func_read_rescue_mapq_thr,
                           TMAP_MAP_ALGO_PAIRING);

  // map1/map3 options
  tmap_map_opt_options_add(opt->options, "seed-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the k-mer length to seed CALs (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_seed_length,
                           TMAP_MAP_ALGO_MAP1 | TMAP_MAP_ALGO_MAP3);

  // map2/map3 options
  tmap_map_opt_options_add(opt->options, "max-seed-hits", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum number of hits returned by a seed",
                           NULL,
                           tmap_map_opt_option_print_func_max_seed_hits,
                           TMAP_MAP_ALGO_MAP2 | TMAP_MAP_ALGO_MAP3);

  // map3/map4 options
  tmap_map_opt_options_add(opt->options, "hit-frac", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the fraction of seed positions that are under the maximum",
                           NULL,
                           tmap_map_opt_option_print_func_hit_frac,
                           TMAP_MAP_ALGO_MAP3 | TMAP_MAP_ALGO_MAP4);
  tmap_map_opt_options_add(opt->options, "seed-step", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the number of bases to increase the seed for each seed increase iteration (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_seed_step,
                           TMAP_MAP_ALGO_MAP3 | TMAP_MAP_ALGO_MAP4);

  // map1 options
  tmap_map_opt_options_add(opt->options, "seed-max-diff", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "maximum number of edits in the seed",
                           NULL,
                           tmap_map_opt_option_print_func_seed_max_diff,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "seed2-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the secondary seed length (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_seed2_length,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-diff", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NUM,
                           "maximum number of edits or false-negative probability assuming the maximum error rate",
                           NULL,
                           tmap_map_opt_option_print_func_max_diff,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-error-rate", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the assumed per-base maximum error rate",
                           NULL,
                           tmap_map_opt_option_print_func_max_err_rate,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-mismatches", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NUM,
                           "maximum number of or (read length) fraction of mismatches",
                           NULL,
                           tmap_map_opt_option_print_func_max_mm,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-gap-opens", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NUM,
                           "maximum number of or (read length) fraction of indel starts",
                           NULL,
                           tmap_map_opt_option_print_func_max_gapo,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-gap-extensions", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NUM,
                           "maximum number of or (read length) fraction of indel extensions",
                           NULL,
                           tmap_map_opt_option_print_func_max_gape,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-cals-deletion", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum number of CALs to extend a deletion ",
                           NULL,
                           tmap_map_opt_option_print_func_max_cals_del,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "indel-ends-bound", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "number of bps from the end of the read ",
                           NULL,
                           tmap_map_opt_option_print_func_indel_ends_bound,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-best-cals", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "optimal CALs have been found ",
                           NULL,
                           tmap_map_opt_option_print_func_max_best_cals,
                           TMAP_MAP_ALGO_MAP1);
  tmap_map_opt_options_add(opt->options, "max-nodes", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "maximum number of alignment nodes ",
                           NULL,
                           tmap_map_opt_option_print_func_max_entries,
                           TMAP_MAP_ALGO_MAP1);

  // map2 options
  tmap_map_opt_options_add(opt->options, "length-coef", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
"coefficient of length-threshold adjustment",
                           NULL,
                           tmap_map_opt_option_print_func_length_coef,
                           TMAP_MAP_ALGO_MAP2);
  tmap_map_opt_options_add(opt->options, "max-seed-intv", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
"maximum seeding interval size",
                           NULL,
                           tmap_map_opt_option_print_func_max_seed_intv,
                           TMAP_MAP_ALGO_MAP2);
  tmap_map_opt_options_add(opt->options, "z-best", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum number of top-scoring nodes to keep on each iteration",
                           NULL,
                           tmap_map_opt_option_print_func_z_best,
                           TMAP_MAP_ALGO_MAP2);
  tmap_map_opt_options_add(opt->options, "seeds-rev", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
"# seeds to trigger reverse alignment",
                           NULL,
                           tmap_map_opt_option_print_func_seeds_rev,
                           TMAP_MAP_ALGO_MAP2);
  tmap_map_opt_options_add(opt->options, "narrow-rmdup", no_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_NONE,
                           "remove duplicates for narrow SA hits",
                           NULL,
                           tmap_map_opt_option_print_func_narrow_rmdup,
                           TMAP_MAP_ALGO_MAP2);
  tmap_map_opt_options_add(opt->options, "max-chain-gap", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "maximum gap size during chaining",
                           NULL,
                           tmap_map_opt_option_print_func_max_chain_gap,
                           TMAP_MAP_ALGO_MAP2);

  // map3 options
  tmap_map_opt_options_add(opt->options, "hp-diff", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "single homopolymer error difference for enumeration",
                           NULL,
                           tmap_map_opt_option_print_func_hp_diff,
                           TMAP_MAP_ALGO_MAP3);
  tmap_map_opt_options_add(opt->options, "fwd-search", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "use forward search instead of a reverse search",
                           NULL,
                           tmap_map_opt_option_print_func_fwd_search,
                           TMAP_MAP_ALGO_MAP3);
  tmap_map_opt_options_add(opt->options, "skip-seed-frac", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the fraction of a seed to skip when a lookup succeeds",
                           NULL,
                           tmap_map_opt_option_print_func_skip_seed_frac,
                           TMAP_MAP_ALGO_MAP3);
  // map4
  tmap_map_opt_options_add(opt->options, "min-seed-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the minimum seed length to accept hits (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_min_seed_length,
                           TMAP_MAP_ALGO_MAP4);
  tmap_map_opt_options_add(opt->options, "max-seed-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum seed length to accept hits",
                           NULL,
                           tmap_map_opt_option_print_func_max_seed_length,
                           TMAP_MAP_ALGO_MAP4);
  tmap_map_opt_options_add(opt->options, "max-seed-length-adj-coef", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the maximum seed length adjustment coefficient (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_max_seed_length_adj_coef,
                           TMAP_MAP_ALGO_MAP4);
  tmap_map_opt_options_add(opt->options, "max-iwidth", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum interval size to accept a hit",
                           NULL,
                           tmap_map_opt_option_print_func_max_iwidth,
                           TMAP_MAP_ALGO_MAP4);
  tmap_map_opt_options_add(opt->options, "max-repr", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum representitive hits for repetitive hits",
                           NULL,
                           tmap_map_opt_option_print_func_max_repr,
                           TMAP_MAP_ALGO_MAP4);
  tmap_map_opt_options_add(opt->options, "rand-repr", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "choose the representitive hits randomly, otherwise uniformly",
                           NULL,
                           tmap_map_opt_option_print_func_rand_repr,
                           TMAP_MAP_ALGO_MAP4);
  tmap_map_opt_options_add(opt->options, "use-min", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "when seed stepping, try seeding when at least the minimum seed length is present, otherwise maximum",
                           NULL,
                           tmap_map_opt_option_print_func_use_min,
                           TMAP_MAP_ALGO_MAP4);

  // mapvsw options
  // None

  // map1/map2/map3 options, but specific to each
  tmap_map_opt_options_add(opt->options, "min-seq-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the minimum sequence length to examine (-1 to disable)",
                           NULL,
                           tmap_map_opt_option_print_func_min_seq_len,
                           ~(TMAP_MAP_ALGO_MAPALL | TMAP_MAP_ALGO_STAGE));
  tmap_map_opt_options_add(opt->options, "max-seq-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the maximum sequence length to examine",
                           NULL,
                           tmap_map_opt_option_print_func_max_seq_len,
                           ~(TMAP_MAP_ALGO_MAPALL | TMAP_MAP_ALGO_STAGE));

  // stage options
  tmap_map_opt_options_add(opt->options, "stage-score-thres", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "score threshold for the stage divided by the match score",
                           NULL,
                           tmap_map_opt_option_print_func_stage_score_thr,
                           TMAP_MAP_ALGO_STAGE);
  tmap_map_opt_options_add(opt->options, "stage-mapq-thres", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "mapping quality threshold for the stage divided by the match score",
                           NULL,
                           tmap_map_opt_option_print_func_stage_mapq_thr,
                           TMAP_MAP_ALGO_STAGE);
  tmap_map_opt_options_add(opt->options, "stage-keep-all", no_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_NONE,
                           "keep mappings from the first stage for the next stage",
                           NULL,
                           tmap_map_opt_option_print_func_stage_keep_all,
                           TMAP_MAP_ALGO_STAGE);
  tmap_map_opt_options_add(opt->options, "stage-seed-freq-cutoff", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "the minimum frequency of a seed to be considered for mapping",
                           NULL,
                           tmap_map_opt_option_print_func_stage_seed_freqc,
                           TMAP_MAP_ALGO_STAGE);
  tmap_map_opt_options_add(opt->options, "stage-seed-freq-cutoff-group-frac", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_FLOAT,
                           "if more than this fraction of groups were filtered, keep representative hits",
                           NULL,
                           tmap_map_opt_option_print_func_stage_seed_freqc_group_frac,
                           TMAP_MAP_ALGO_STAGE);
  tmap_map_opt_options_add(opt->options, "stage-seed-freq-cutoff-rand-repr", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "the number of representative hits to keep",
                           NULL,
                           tmap_map_opt_option_print_func_stage_seed_freqc_rand_repr,
                           TMAP_MAP_ALGO_STAGE);
  tmap_map_opt_options_add(opt->options, "stage-seed-freq-cutoff-min-groups", required_argument, 0, 0,
                           TMAP_MAP_OPT_TYPE_INT,
                           "the minimum of groups required after the filter has been applied, otherwise iteratively reduce the filter",
                           NULL,
                           tmap_map_opt_option_print_func_stage_seed_freqc_min_groups,
                           TMAP_MAP_ALGO_STAGE);
  tmap_map_opt_options_add(opt->options, "stage-seed-max-length", required_argument, 0, 0, 
                           TMAP_MAP_OPT_TYPE_INT,
                           "the length of the prefix of the read to consider during seeding",
                           NULL,
                           tmap_map_opt_option_print_func_stage_seed_max_length,
                           TMAP_MAP_ALGO_STAGE);

  /*
  // Prints out all single-flag command line options
  int i, c;
  for(c=1;c<256;c++) {
      for(i=0;i<opt->options->n;i++) {
          if(c == opt->options->options[i].option.val) {
              fputc((char)c, stderr);
              if(required_argument == opt->options->options[i].option.has_arg) {
                  fputc(':', stderr);
              }
          }
      }
  }
  fprintf(stderr, "\n");
  */
  
  /*
  // Prints out all command line options
  int i, c;
  for(c=0;c<256;c++) {
      for(i=0;i<opt->options->n;i++) {
          if(c == opt->options->options[i].option.val) {
              if(0 < opt->options->options[i].option.val) {
                  tmap_file_fprintf(tmap_file_stderr, "-%c,", (char)opt->options->options[i].option.val);
              }
              // long flag
              tmap_file_fprintf(tmap_file_stderr, "--%s\n", opt->options->options[i].option.name);
          }
      }
  }
  fprintf(stderr, "\n");
  */
}

tmap_map_opt_t *
tmap_map_opt_init(int32_t algo_id)
{
  tmap_map_opt_t *opt = NULL;

  opt = tmap_calloc(1, sizeof(tmap_map_opt_t), "opt");

  // internal data
  opt->algo_id = algo_id;
  opt->algo_stage = -1;
  opt->argv = NULL;
  opt->argc = -1;

  // global options 
  opt->fn_fasta = NULL;
  opt->fn_reads = NULL;
  opt->fn_reads_num = 0;
  opt->reads_format = TMAP_READS_FORMAT_UNKNOWN;
  opt->fn_sam = NULL;
  opt->bam_start_vfo = 0;
  opt->bam_end_vfo = 0;
  opt->use_param_ovr = 0;
  opt->ovr_candeval = 0;
  opt->use_bed_in_end_repair = 1;
  opt->use_bed_in_mapq = 1;
  opt->use_bed_read_ends_stat = 0;
  opt->amplicon_scope = TMAP_MAP_OPT_AMPLICON_SCOPE;
  opt->score_match = TMAP_MAP_OPT_SCORE_MATCH;
  opt->pen_mm = TMAP_MAP_OPT_PEN_MM;
  opt->pen_gapo = TMAP_MAP_OPT_PEN_GAPO;
  opt->pen_gape = TMAP_MAP_OPT_PEN_GAPE;
  opt->pen_gapl = TMAP_MAP_OPT_PEN_GAPL;
  opt->gapl_len = 50;
  opt->bw = 50; 
  opt->softclip_type = TMAP_MAP_OPT_SOFT_CLIP_RIGHT;
  opt->dup_window = 128;
  opt->max_seed_band = 15;
  opt->unroll_banding = 0;
  opt->long_hit_mult = 3.0;
  opt->score_thr = 8;
  opt->reads_queue_size = 65535;
  opt->num_threads = tmap_detect_cpus();
  opt->num_threads_autodetected = 1;
  opt->aln_output_mode = TMAP_MAP_OPT_ALN_MODE_RAND_BEST;
  opt->sam_rg = NULL;
  opt->sam_rg_num = 0;
  opt->bidirectional = 0;
  opt->seq_eq = 0;
  opt->ignore_rg_sam_tags = 0;
  opt->rand_read_name = 0;
  opt->input_compr = TMAP_FILE_NO_COMPRESSION;
  opt->output_type = 0;
  opt->end_repair = 0;
  opt->max_one_large_indel_rescue = 30;
  opt->min_anchor_large_indel_rescue = 6;
  opt->amplicon_overrun = 6;
  opt->max_adapter_bases_for_soft_clipping = INT32_MAX;
  opt->end_repair_5_prime_softclip = 1;
  opt->repair_min_freq = 97;
  opt->repair_min_count = 0;
  opt->repair_min_adapter = 0;
  opt->repair_max_overhang = 16;
  opt->repair_identity_drop_limit = 0.6;
  opt->repair_max_primer_zone_dist = 1;
  opt->repair_clip_ext = 12;
  opt->shm_key = 0;
  opt->min_seq_len = -1;
  opt->max_seq_len = -1;
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  opt->sample_reads = 1.0;
#endif
  opt->vsw_type = 4;
  opt->vsw_fallback = 1;
  opt->confirm_vsw_corr = 0;
  opt->candidate_ext = 1;
  opt->correct_failed_vsw = 1;
  opt->use_nvsw_on_nonstd_bases = 1;

  opt->do_realign = 0;
  opt->realign_mat_score = TMAP_MAP_OPT_REALIGN_SCORE_MATCH;
  opt->realign_mis_score = TMAP_MAP_OPT_REALIGN_SCORE_MM;
  opt->realign_gip_score = TMAP_MAP_OPT_REALIGN_SCORE_GO;
  opt->realign_gep_score = TMAP_MAP_OPT_REALIGN_SCORE_GE;
  opt->realign_bandwidth = TMAP_MAP_OPT_REALIGN_BW;
  opt->realign_cliptype = 2;
  opt->realign_maxlen = TMAP_MAP_OPT_REALIGN_MAXLEN;
  opt->realign_maxclip = TMAP_MAP_OPT_REALIGN_MAXCLIP;
  opt->report_stats = 0;
  opt->realign_log = NULL;
  // tail repeat clipping
  opt->do_repeat_clip = 0;
  opt->repclip_continuation = 0;

  opt->cigar_sanity_check = TMAP_MAP_SANITY_NONE;
  
  // context dependent gap scores
  opt->do_hp_weight = 0;
  opt->gap_scale_mode = TMAP_CONTEXT_GAP_SCALE_GEP;
  opt->context_mat_score = TMAP_MAP_OPT_CONTEXT_SCORE_MATCH;
  opt->context_mis_score = TMAP_MAP_OPT_CONTEXT_SCORE_MM;
  opt->context_gip_score = TMAP_MAP_OPT_CONTEXT_SCORE_GO;
  opt->context_gep_score = TMAP_MAP_OPT_CONTEXT_SCORE_GE;
  opt->context_extra_bandwidth = TMAP_MAP_OPT_CONTEXT_BW_EXTRA;

  opt->debug_log = 0;

  // alignment length filtering
  opt->min_al_len = MIN_AL_LEN_NOCHECK_SPECIAL;
  opt->min_al_cov = MIN_AL_COVERAGE_NOCHECK_SPECIAL;
  opt->min_identity = MIN_AL_IDENTITY_NOCHECK_SPECIAL;

  // flowspace options
  opt->fscore = TMAP_MAP_OPT_FSCORE;
  opt->softclip_key = 0;
  opt->sam_flowspace_tags = 0;
  opt->ignore_flowgram = 0;
  opt->aln_flowspace = 0;

  // pairing options
  opt->pairing = 0;
  opt->strandedness = -1;
  opt->positioning = -1;
  opt->ins_size_mean = -1.0;
  opt->ins_size_std = -1.0;
  opt->ins_size_std_max_num = 2.0;
  opt->ins_size_outlier_bound = 2.0;
  opt->ins_size_min_mapq = 20;
  opt->read_rescue = 0;
  opt->read_rescue_std_num = -1.0;
  opt->read_rescue_mapq_thr = 0;

  switch(algo_id) {
    case TMAP_MAP_ALGO_MAP1:
      // map1
      opt->seed_length = 32;
      opt->seed_length_set = 1;
      opt->seed_max_diff = 2;
      opt->seed2_length = 48;
      opt->max_diff = -1; opt->max_diff_fnr = 0.04;
      opt->max_mm = 3; opt->max_mm_frac = -1.;
      opt->max_err_rate = 0.02;
      opt->max_gapo = 1; opt->max_gapo_frac = -1.;
      opt->max_gape = 6; opt->max_gape_frac = -1.;
      opt->max_cals_del = 10;
      opt->indel_ends_bound = 5;
      opt->max_best_cals = 32;
      opt->max_entries = 2000000;
      break;
    case TMAP_MAP_ALGO_MAP2:
      // map2
      //opt->mask_level = 0.50; 
      opt->length_coef = 5.5f;
      opt->max_seed_intv = 6; 
      opt->z_best = 1; 
      opt->seeds_rev = 5;
      opt->narrow_rmdup = 0;
      opt->max_seed_hits = 1024;
      opt->max_chain_gap= 10000;
      break;
    case TMAP_MAP_ALGO_MAP3:
      // map3
      opt->seed_length = -1;
      opt->seed_length_set = 0;
      opt->max_seed_hits = 20;
      opt->hp_diff = 0;
      opt->hit_frac = 0.20;
      opt->seed_step = 8;
      opt->fwd_search = 0;
      opt->skip_seed_frac = 0.2;
      break;
    case TMAP_MAP_ALGO_MAP4:
      // map4
      opt->min_seed_length = -1;
      opt->max_seed_length = 48; 
      opt->max_seed_length_adj_coef = 2.0; 
      opt->hit_frac = 0.20;
      opt->seed_step = 8;
      opt->max_iwidth = 20;
      opt->max_repr = 3;
      opt->rand_repr = 0;
      opt->use_min = 0;
      break;
    case TMAP_MAP_ALGO_MAPVSW:
      // mapvsw
      break;
    case TMAP_MAP_ALGO_STAGE:
      // stage
      opt->stage_score_thr = 8;
      opt->stage_mapq_thr = 23; // 0.5% error
      opt->stage_keep_all = 0;
      opt->stage_seed_freqc = 0.0; //all-pass filter as default
      opt->stage_seed_freqc_group_frac = 0.9; 
      opt->stage_seed_freqc_rand_repr = 2; 
      opt->stage_seed_freqc_min_groups = 1; 
      opt->stage_seed_max_length = -1;
      break;
    default:
      break;
  }

  // build options for parsing and printing
  tmap_map_opt_init_helper(opt);

  opt->sub_opts = NULL;
  opt->num_sub_opts = 0;
  opt->bed_file = NULL;

  return opt;
}

tmap_map_opt_t*
tmap_map_opt_add_sub_opt(tmap_map_opt_t *opt, int32_t algo_id)
{
  opt->num_sub_opts++;
  opt->sub_opts = tmap_realloc(opt->sub_opts, opt->num_sub_opts * sizeof(tmap_map_opt_t*), "opt->sub_opts");
  opt->sub_opts[opt->num_sub_opts-1] = tmap_map_opt_init(algo_id);
  // copy global options
  tmap_map_opt_copy_global(opt->sub_opts[opt->num_sub_opts-1], opt);
  return opt->sub_opts[opt->num_sub_opts-1];
}

void
tmap_map_opt_destroy(tmap_map_opt_t *opt)
{
  int32_t i;

  free(opt->fn_fasta);
  free(opt->bed_file);
  for(i=0;i<opt->fn_reads_num;i++) {
      free(opt->fn_reads[i]); 
  }
  free(opt->fn_reads);
  free(opt->fn_sam);
  for(i=0;i<opt->sam_rg_num;i++) {
      free(opt->sam_rg[i]);
  }
  free(opt->sam_rg);
  
  free (opt->realign_log);

  for(i=0;i<opt->num_sub_opts;i++) {
      tmap_map_opt_destroy(opt->sub_opts[i]);
  }
  free(opt->sub_opts);

  // destroy options for parsing and printing
  tmap_map_opt_options_destroy(opt->options);

  free(opt);
}

static void
tmap_map_opt_usage_algo(tmap_map_opt_t *opt, int32_t stage)
{
  int32_t i;
  if(opt->algo_id & TMAP_MAP_ALGO_MAPALL) {
      return; // NB: there are no MAPALL specific options
  }
  else if(opt->algo_id & TMAP_MAP_ALGO_STAGE) {
      tmap_file_fprintf(tmap_file_stderr, "\n%sstage%d options: [stage options] [algorithm [algorithm options]]+%s\n", KBLDRED, stage, KNRM);
  }
  else if(stage < 0) {
      tmap_file_fprintf(tmap_file_stderr, "\n%s%s options (optional):%s\n", KBLDRED, tmap_algo_id_to_name(opt->algo_id), KNRM);
  }
  else {
      tmap_file_fprintf(tmap_file_stderr, "\n%s%s stage%d options (optional):%s\n", KBLDRED, tmap_algo_id_to_name(opt->algo_id), stage, KNRM);
  }
  for(i=0;i<opt->options->n;i++) {
      tmap_map_opt_option_t *o = &opt->options->options[i];

      if(0 < (o->algos & opt->algo_id)) {
          tmap_map_opt_option_print(o, opt);
      }
  }
}

int
tmap_map_opt_usage(tmap_map_opt_t *opt)
{
  int32_t i, prev_stage;

  tmap_version(opt->argc, opt->argv);
  
  // print global options
  if(opt->algo_id == TMAP_MAP_ALGO_MAPALL) {
      tmap_file_fprintf(tmap_file_stderr, "\n%s%s %s[%sglobal options%s]%s %s[%sflowspace options%s]%s %s[%sstage%s[%s0-9%s]%s+ %s[%sstage options%s]%s %s[%salgorithm %s[%salgorithm options%s]%s%s]%s+%s]%s+%s\n", 
                        KBLDRED,
                        tmap_algo_id_to_name(opt->algo_id),
                        KMAG, KWHT, KMAG, KBLDRED, KMAG, KWHT, KMAG, KBLDRED,
                        KMAG, KWHT, KMAG, KBLDRED, KMAG, KWHT, KMAG, KBLDRED,
                        KMAG, KWHT, KMAG, KBLDRED, KMAG, KWHT, KMAG, KBLDRED,
                        KMAG, KWHT, KMAG, KBLDRED, KMAG, KWHT, KMAG, KBLDRED,
                        KMAG, KWHT, KMAG, KBLDRED, KMAG, KWHT, KMAG, KBLDRED,
                        KMAG, KWHT, KMAG, KBLDRED, KMAG, KWHT, KMAG, KBLDRED,
                        KMAG, KWHT, KMAG, KBLDRED, KMAG, KWHT, KMAG, KBLDRED,
                        KNRM);
  }
  else {
      tmap_file_fprintf(tmap_file_stderr, "%sUsage: %s %s [global options] [flowspace options] [%s options]%s\n", 
                        KBLDRED,
                        PACKAGE, 
                        tmap_algo_id_to_name(opt->algo_id),
                        tmap_algo_id_to_name(opt->algo_id),
                        KBLDRED);
  }
  tmap_file_fprintf(tmap_file_stderr, "\n");
  tmap_file_fprintf(tmap_file_stderr, "%sglobal options:%s\n", KBLDRED, KNRM);
  for(i=0;i<opt->options->n;i++) {
      tmap_map_opt_option_t *o = &opt->options->options[i];

      if(o->algos == TMAP_MAP_ALGO_GLOBAL) {
          tmap_map_opt_option_print(o, opt);
      }
  }

  // print flowspace options
  tmap_file_fprintf(tmap_file_stderr, "\n");
  tmap_file_fprintf(tmap_file_stderr, "%sflowspace options:%s\n", KBLDRED, KNRM);
  for(i=0;i<opt->options->n;i++) {
      tmap_map_opt_option_t *o = &opt->options->options[i];

      if(o->algos == TMAP_MAP_ALGO_FLOWSPACE) {
          tmap_map_opt_option_print(o, opt);
      }
  }

  // print pairing options
  tmap_file_fprintf(tmap_file_stderr, "\n");
  tmap_file_fprintf(tmap_file_stderr, "%spairing options:%s\n", KBLDRED, KNRM);
  for(i=0;i<opt->options->n;i++) {
      tmap_map_opt_option_t *o = &opt->options->options[i];

      if(o->algos == TMAP_MAP_ALGO_PAIRING) {
          tmap_map_opt_option_print(o, opt);
      }
  }

  // print algorithm specific options
  for(i=0,prev_stage=-1;i<opt->num_sub_opts;i++) {
      // print the stage
      if(opt->sub_opts[i]->algo_stage != prev_stage) {
          prev_stage = opt->sub_opts[i]->algo_stage;
          // print the stage
          /*
          //tmap_file_fprintf(tmap_file_stderr, "\nstage%d options:\n", prev_stage);
          for(j=0;j<opt->options->n;j++) {
              tmap_map_opt_option_t *o = &opt->options->options[j];
              if(o->algos == TMAP_MAP_ALGO_STAGE) {
                  tmap_map_opt_option_print(o, opt);
              }
          }
          */
      }
      
      tmap_map_opt_usage_algo(opt->sub_opts[i], opt->sub_opts[i]->algo_stage);
  }
  tmap_map_opt_usage_algo(opt, -1);

  tmap_map_opt_destroy(opt);

  return 1;
}

static void
tmap_map_opt_add_tabbed(char ***dest, int32_t *dest_num, char *str)
{
  int32_t i, j, l;

  l = strlen(str);
  i = j = 0;
  while(i < l) {
      j = i;
      while(j < l && '\t' != str[j]) {
          j++;
      }
      // add
      (*dest_num)++;
      (*dest) = tmap_realloc((*dest), sizeof(char*) * (*dest_num), "(*dest)");
      (*dest)[(*dest_num)-1] = tmap_malloc(sizeof(char) * (j - i + 1), "(*dest)[(*dest_num)-1]");
      strncpy((*dest)[(*dest_num)-1], str + i, (j - i));
      (*dest)[(*dest_num)-1][(j-i)] = '\0';
      tmap_chomp((*dest)[(*dest_num)-1]); // remove trailing white spaces
      i = j + 1;
  }
}

int32_t
tmap_map_opt_parse(int argc, char *argv[], tmap_map_opt_t *opt)
{
  int i, c, option_index, val = 0;
  char *getopt_format = NULL; 
  int32_t getopt_format_mem = 0;
  int32_t getopt_format_len = 1;
  struct option *options = NULL;

  // set options passed in
  opt->argc = argc; opt->argv = argv;
  
  if(argc == optind) {
      // no need to parse
      //fprintf(stderr, "\n[opt_parse] argc==optind.  no need to parse\n");
      return 1;
  }

  /*
  fprintf(stderr, "\nargc: %d optind: %d\n", argc, optind);
  for(i=optind;i<argc;i++) {
      fprintf(stderr, "[opt_parse] i=%d argv[i]=%s\n", i, argv[i]);
  }
  */
  

  // allocate
  options = tmap_calloc(1, sizeof(struct option) * opt->options->n, "options");

  // format
  getopt_format_len = 0;
  getopt_format_mem = 4;
  getopt_format = tmap_calloc(getopt_format_mem, sizeof(char) * getopt_format_mem, "getopt_format"); 

  // shallow copy
  for(i=0;i<opt->options->n;i++) {
      options[i] = opt->options->options[i].option; 
      if(0 != options[i].val) {
          while(getopt_format_mem < getopt_format_len + 4) {
              getopt_format_mem <<= 1;
              getopt_format = tmap_realloc(getopt_format, sizeof(char) * getopt_format_mem, "getopt_format"); 
          }
          getopt_format[getopt_format_len] = (char)(options[i].val);
          getopt_format_len++;
          getopt_format[getopt_format_len] = '\0';
          if(no_argument != options[i].has_arg) {
              getopt_format[getopt_format_len] = ':';
              getopt_format_len++;
          }
      }
  }
  getopt_format[getopt_format_len] = '\0';

  // check algorithm
  switch(opt->algo_id) {
    case TMAP_MAP_ALGO_MAP1:
    case TMAP_MAP_ALGO_MAP2:
    case TMAP_MAP_ALGO_MAP3:
    case TMAP_MAP_ALGO_MAP4:
    case TMAP_MAP_ALGO_MAPVSW:
    case TMAP_MAP_ALGO_STAGE:
    case TMAP_MAP_ALGO_MAPALL:
      break;
    default:
      tmap_error("unrecognized algorithm", Exit, OutOfRange);
      break;
  }

  while((c = getopt_long(argc, argv, getopt_format, options, &option_index)) >= 0) {
      // Global options
      if(c == '?') {
          break;
      }
      else if(0 == c && 0 == strcmp("bam-start-vfo", options[option_index].name)) {
          opt->bam_start_vfo = strtol(optarg,NULL,0);
      }
      else if(0 == c && 0 == strcmp("bam-end-vfo", options[option_index].name)) {
          opt->bam_end_vfo = strtol(optarg,NULL,0);
      }
      else if(0 == c && 0 == strcmp("par-ovr", options[option_index].name)) {
          opt->use_param_ovr = 1;
      }
      else if(0 == c && 0 == strcmp("ovr-candeval", options[option_index].name)) {
          opt->ovr_candeval = 1;
      }
      else if((0 == c && 0 == strcmp("no-bed-er", options[option_index].name))) {
          opt->use_bed_in_end_repair = 0;
      }
      else if((0 == c && 0 == strcmp("no-bed-mapq", options[option_index].name))) {
          opt->use_bed_in_mapq = 0;
      }
      else if((0 == c && 0 == strcmp("repair", options[option_index].name))) {
          opt->use_bed_read_ends_stat = 1;
      }
      else if (0 == c && 0 == strcmp("ampl-scope", options[option_index].name)) {
          opt->amplicon_scope = atoi(optarg);
      }
      else if(c == 'A' || (0 == c && 0 == strcmp("score-match", options[option_index].name))) {
          opt->score_match = atoi(optarg);
      }
      else if(c == 'B' || (0 == c && 0 == strcmp("max-seed-band", options[option_index].name))) {
          opt->max_seed_band = atoi(optarg);
      }
      else if(c == 'C' || (0 == c && 0 == strcmp("ignore-rg-from-sam", options[option_index].name))) {
          opt->ignore_rg_sam_tags = 1;
      }
      else if(c == 'D' || (0 == c && 0 == strcmp("bidirectional", options[option_index].name))) {
          opt->bidirectional = 1;
      }
      else if(c == 'E' || (0 == c && 0 == strcmp("pen-gap-extension", options[option_index].name))) {
          opt->pen_gape = atoi(optarg);
      }
      else if(c == 'G' || (0 == c && 0 == strcmp("pen-gap-long", options[option_index].name))) {
          opt->pen_gapl = atoi(optarg);
      }
      else if(c == 'H' || (0 == c && 0 == strcmp("vsw-type", options[option_index].name))) {
          opt->vsw_type = atoi(optarg); 
      }
      else if(0 == c && 0 == strcmp("vsw-fallback", options[option_index].name)) {
          opt->vsw_fallback = atoi(optarg); 
      }
      else if(0 == c && 0 == strcmp("confirm-vsw-corr", options[option_index].name)) {
          opt->confirm_vsw_corr = 1;
      }
      else if(0 == c && 0 == strcmp("no-candidate-ext", options[option_index].name)) {
          opt->candidate_ext = 0;
      }
      else if(0 == c && 0 == strcmp("no-vsw-fallback", options[option_index].name)) {
          opt->correct_failed_vsw = 0;
      }
      else if(0 == c && 0 == strcmp("no-nonstd-bases", options[option_index].name)) {
          opt->use_nvsw_on_nonstd_bases = 0;
      }
      else if(c == 'I' || (0 == c && 0 == strcmp("use-seq-equal", options[option_index].name))) {
          opt->seq_eq = 1;
      }
      else if(c == 'J' || (0 == c && 0 == strcmp("max-adapter-bases-for-soft-clipping", options[option_index].name))) {
          opt->max_adapter_bases_for_soft_clipping = atoi(optarg);
      }
      else if(c == 'K' || (0 == c && 0 == strcmp("gap-long-length", options[option_index].name))) {
          opt->gapl_len = atoi(optarg);
      }
      else if(c == 'M' || (0 == c && 0 == strcmp("pen-mismatch", options[option_index].name))) {
          opt->pen_mm = atoi(optarg);
      }
      else if(c == 'O' || (0 == c && 0 == strcmp("pen-gap-open", options[option_index].name))) {
          opt->pen_gapo = atoi(optarg);
      }
      else if(c == 'R' || (0 == c && 0 == strcmp("sam-read-group", options[option_index].name))) {
          tmap_map_opt_add_tabbed(&opt->sam_rg, &opt->sam_rg_num, optarg);
      }
      else if(c == 'T' || (0 == c && 0 == strcmp("score-thres", options[option_index].name))) {
          opt->score_thr = atoi(optarg);
      }
      else if(c == 'U' || (0 == c && 0 == strcmp("unroll-banding", options[option_index].name))) {
          opt->unroll_banding = 1;
      }
      else if(c == 'W' || (0 == c && 0 == strcmp("duplicate-window", options[option_index].name))) {
          opt->dup_window = atoi(optarg);
      }
      else if(c == 'a' || (0 == c && 0 == strcmp("aln-output-mode", options[option_index].name))) {
          opt->aln_output_mode = atoi(optarg);
      }
      else if(c == 'f' || (0 == c && 0 == strcmp("fn-fasta", options[option_index].name))) {
          free(opt->fn_fasta);
          opt->fn_fasta = tmap_strdup(optarg);
      }
      else if((0 == c && 0 == strcmp("bed-file", options[option_index].name))) {
          free(opt->bed_file);
          opt->bed_file = tmap_strdup(optarg);
      }
      else if(c == 'g' || (0 == c && 0 == strcmp("softclip-type", options[option_index].name))) {
          opt->softclip_type = atoi(optarg);
      }
      else if(c == 'h' || (0 == c && 0 == strcmp("help", options[option_index].name))) {
          break;
      }
      else if(c == 'i' || (0 == c && 0 == strcmp("reads-format", options[option_index].name))) {
          opt->reads_format = tmap_get_reads_file_format_int(optarg);
      }
      else if(c == 'j' || (0 == c && 0 == strcmp("input-bz2", options[option_index].name))) {
          opt->input_compr = TMAP_FILE_BZ2_COMPRESSION;
          for(i=0;i<opt->fn_reads_num;i++) {
              tmap_get_reads_file_format_from_fn_int(opt->fn_reads[i], &opt->reads_format, &opt->input_compr);
          }
      }
      else if(c == 'k' || (0 == c && 0 == strcmp("shared-memory-key", options[option_index].name))) {
          opt->shm_key = atoi(optarg);
      }
      else if(c == 'o' || (0 == c && 0 == strcmp("output-type", options[option_index].name))) {
          opt->output_type = atoi(optarg);
      }
      else if(c == 'n' || (0 == c && 0 == strcmp("num-threads", options[option_index].name))) {
          opt->num_threads = atoi(optarg);
          opt->num_threads_autodetected = 0;
      }
      else if(c == 'q' || (0 == c && 0 == strcmp("reads-queue-size", options[option_index].name))) {
          opt->reads_queue_size = atoi(optarg);
      }
      else if(c == 'r' || (0 == c && 0 == strcmp("fn-reads", options[option_index].name))) {
          opt->fn_reads_num++;
          opt->fn_reads = tmap_realloc(opt->fn_reads, sizeof(char*) * opt->fn_reads_num, "opt->fn_reads");
          opt->fn_reads[opt->fn_reads_num-1] = tmap_strdup(optarg); 
          tmap_get_reads_file_format_from_fn_int(opt->fn_reads[opt->fn_reads_num-1], &opt->reads_format, &opt->input_compr);
      }
      else if(c == 's' || (0 == c && 0 == strcmp("fn-sam", options[option_index].name))) {
          free(opt->fn_sam);
          opt->fn_sam = tmap_strdup(optarg);
      }
      else if(c == 'u' || (0 == c && 0 == strcmp("rand-read-name", options[option_index].name))) {
          opt->rand_read_name = 1;
      } 
      else if(0 == c && 0 == strcmp("prefix-exclude",  options[option_index].name)) {
	  opt->prefix_exclude = atoi(optarg);
      } 
      else if(0 == c && 0 == strcmp("suffix-exclude",  options[option_index].name)) {
          opt->suffix_exclude = atoi(optarg);
      } 
      else if(0 == c && 0 == strcmp("newQV",  options[option_index].name)) {
          opt->use_new_QV = atoi(optarg);
      }
      else if(c == 'v' || (0 == c && 0 == strcmp("verbose", options[option_index].name))) {
          tmap_progress_set_verbosity(1);
      }
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
      else if(c == 'x' || (0 == c && 0 == strcmp("sample-reads", options[option_index].name))) {
          opt->sample_reads = atof(optarg); 
      }
#endif
      else if(c == 'w' || (0 == c && 0 == strcmp("band-width", options[option_index].name))) {
          opt->bw = atoi(optarg);
      }
      else if(c == 'z' || (0 == c && 0 == strcmp("input-gz", options[option_index].name))) {
          opt->input_compr = TMAP_FILE_GZ_COMPRESSION;
          for(i=0;i<opt->fn_reads_num;i++) {
              tmap_get_reads_file_format_from_fn_int(opt->fn_reads[i], &opt->reads_format, &opt->input_compr);
          }
      }
      else if(0 == c && 0 == strcmp("long-hit-mult", options[option_index].name)) {
          opt->long_hit_mult = atof(optarg);
      }
      else if(0 == c && 0 == strcmp("end-repair", options[option_index].name)) {
          opt->end_repair = atoi(optarg);
      }
      else if(0 == c && 0 == strcmp("max-one-large-indel-rescue", options[option_index].name)) {
          opt->max_one_large_indel_rescue = atoi(optarg);
      } else if(0 == c && 0 == strcmp("min-anchor-large-indel-rescue", options[option_index].name)) {
          opt->min_anchor_large_indel_rescue = atoi(optarg);
      } else if(0 == c && 0 == strcmp("max-amplicon-overrun-large-indel-rescue", options[option_index].name)) {
          opt->amplicon_overrun = atoi(optarg);
      }
      else if (0 == c && 0 == strcmp ("er-no5clip", options [option_index].name)) {
          opt->end_repair_5_prime_softclip = 0;
      } else if(0 == c && 0 == strcmp("repair-min-freq", options[option_index].name)) {
          opt->repair_min_freq = atoi(optarg);
      } else if(0 == c && 0 == strcmp("repair-min-count", options[option_index].name)) {
          opt->repair_min_count = atoi(optarg);
      } else if(0 == c && 0 == strcmp("repair-min-adapter", options[option_index].name)) {
          opt->repair_min_adapter = atoi(optarg);
      } else if(0 == c && 0 == strcmp("repair-max-overhang", options[option_index].name)) {
          opt->repair_max_overhang = atoi(optarg);
      } else if(0 == c && 0 == strcmp("repair-identity-drop-limit", options[option_index].name)) {
          opt->repair_identity_drop_limit = atof(optarg);
      } else if(0 == c && 0 == strcmp("repair-max-primer-zone-dist", options[option_index].name)) {
          opt->repair_max_primer_zone_dist = atoi(optarg);
      } else if(0 == c && 0 == strcmp("repair-clip-ext", options[option_index].name)) {
          opt->repair_clip_ext = atoi(optarg);
      }
      // End of global options

      // realignment options
      else if (0 == c && 0 == strcmp ("do-realign", options [option_index].name)) {
          opt->do_realign = 1;
      }
      else if (0 == c && 0 == strcmp ("r-mat", options [option_index].name)) {
          opt->realign_mat_score = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("r-mis", options [option_index].name)) {
          opt->realign_mis_score = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("r-gip", options [option_index].name)) {
          opt->realign_gip_score = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("r-gep", options [option_index].name)) {
          opt->realign_gep_score = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("r-bw", options [option_index].name)) {
          opt->realign_bandwidth = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("r-clip", options [option_index].name)) {
          opt->realign_cliptype = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("r-maxlen", options [option_index].name)) {
          opt->realign_maxlen = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("r-maxclip", options [option_index].name)) {
          opt->realign_maxclip = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("stats", options [option_index].name)) {
          opt->report_stats = 1;
      }
      else if (0 == c && 0 == strcmp ("log", options [option_index].name)) {
          free(opt->realign_log);
          opt->realign_log = tmap_strdup(optarg);
      }
//      else if (0 == c && 0 == strcmp ("text-als", options [option_index].name)) {
//          opt->log_text_als = 1;
//      }
      // tail-repeat clipping
      else if (0 == c && 0 == strcmp ("do-repeat-clip", options [option_index].name)) {
          opt->do_repeat_clip = 1;
      }
      else if (0 == c && 0 == strcmp ("repclip-cont", options [option_index].name)) {
          opt->repclip_continuation = 1;
      }
      else if (0 == c && 0 == strcmp ("cigar-sanity-check", options [option_index].name)) {
          opt->cigar_sanity_check = atoi (optarg);
      }
      // context-dependent gap scoring
      else if (0 == c && 0 == strcmp ("context", options [option_index].name)) {
          opt->do_hp_weight = 1;
      }
      else if (0 == c && 0 == strcmp ("gap-scale", options [option_index].name)) {
          opt->gap_scale_mode = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("c-mat", options [option_index].name)) {
          opt->context_mat_score = atof (optarg);
      }
      else if (0 == c && 0 == strcmp ("c-mis", options [option_index].name)) {
          opt->context_mis_score = atof (optarg);
      }
      else if (0 == c && 0 == strcmp ("c-gip", options [option_index].name)) {
          opt->context_gip_score = atof (optarg);
      }
      else if (0 == c && 0 == strcmp ("c-gep", options [option_index].name)) {
          opt->context_gep_score = atof (optarg);
      }
      else if (0 == c && 0 == strcmp ("c-bw", options [option_index].name)) {
          opt->context_extra_bandwidth = atoi (optarg);
      }
      else if (0 == c && 0 == strcmp ("debug-log", options [option_index].name)) {
          opt->debug_log = 1;
      }
      // filtering by alignment length
      else if (0 == c && 0 == strcmp ("min-al-len", options [option_index].name)) {
          opt->min_al_len = atoi (optarg);
      }
      // filtering by alignemnt coverage
      else if (0 == c && 0 == strcmp ("min-cov", options [option_index].name)) {
          opt->min_al_cov = atof (optarg);
      }
      // filtering by match identity
      else if (0 == c && 0 == strcmp ("min-iden", options [option_index].name)) {
          opt->min_identity = atof (optarg);
      }
      // Flowspace options
      else if(c == 'F' || (0 == c && 0 == strcmp("final-flowspace", options[option_index].name))) {
          opt->aln_flowspace = 1;
      }
      else if(c == 'N' || (0 == c && 0 == strcmp("use-flowgram", options[option_index].name))) {
          opt->ignore_flowgram = 1;
      }
      else if(c == 'X' || (0 == c && 0 == strcmp("pen-flow-error", options[option_index].name))) {
          opt->fscore = atoi(optarg);
      }
      else if(c == 'Y' || (0 == c && 0 == strcmp("sam-flowspace-tags", options[option_index].name))) {
          opt->sam_flowspace_tags = 1;
      }
      else if(c == 'y' || (0 == c && 0 == strcmp("softclip-key", options[option_index].name))) {
          opt->softclip_key = 1;
      }
      // End of flowspace options
      // Pairing options
      else if(c == 'Q' || (0 == c && 0 == strcmp("pairing", options[option_index].name))) {
          opt->pairing = atoi(optarg);
          if(1 == opt->pairing) {
              opt->strandedness = 0;
              opt->positioning = 1;
          }
          else if(2 == opt->pairing) {
              opt->strandedness = 1;
              opt->positioning = 0;
          }
      }
      else if(c == 'S' || (0 == c && 0 == strcmp("strandedness", options[option_index].name))) {
          opt->strandedness = atoi(optarg);
      }
      else if(c == 'P' || (0 == c && 0 == strcmp("positioning", options[option_index].name))) {
          opt->positioning = atoi(optarg);
      }
      else if(c == 'b' || (0 == c && 0 == strcmp("ins-size-mean", options[option_index].name))) {
          opt->ins_size_mean = atof(optarg);
      }
      else if(c == 'c' || (0 == c && 0 == strcmp("ins-size-std", options[option_index].name))) {
          opt->ins_size_std = atof(optarg);
      }
      else if(c == 'd' || (0 == c && 0 == strcmp("ins-size-std-max-num", options[option_index].name))) {
          opt->ins_size_std_max_num = atof(optarg);
      }
      else if(c == 'L' || (0 == c && 0 == strcmp("read-rescue", options[option_index].name))) {
          opt->read_rescue = 1;
      }
      else if(c == 'l' || (0 == c && 0 == strcmp("read-rescue-std-num", options[option_index].name))) {
          opt->read_rescue_std_num = atof(optarg);
      }
      else if(c == 'm' || (0 == c && 0 == strcmp("read-rescue-mapq-thr", options[option_index].name))) {
          opt->read_rescue_mapq_thr = atoi(optarg);
      }
      else if(c == 'p' || (0 == c && 0 == strcmp("ins-size-outlier-bound", options[option_index].name))) {
          opt->ins_size_outlier_bound = atof(optarg);
      }
      else if(c == 't' || (0 == c && 0 == strcmp("ins-size-max-mapq", options[option_index].name))) {
          opt->ins_size_min_mapq = atoi(optarg);
      }
      // End of pairing options 
      // End single flag options
      else if(0 != c) {
          tmap_bug();
      }
      // MAP1/MAP2/MAP3/MAPVSW
      else if(0 == strcmp("min-seq-length", options[option_index].name) && (opt->algo_id == TMAP_MAP_ALGO_MAP1 || opt->algo_id == TMAP_MAP_ALGO_MAP2 || opt->algo_id == TMAP_MAP_ALGO_MAP4
                                                                            || opt->algo_id == TMAP_MAP_ALGO_MAP3 || opt->algo_id == TMAP_MAP_ALGO_MAPVSW)) {
          opt->min_seq_len = atoi(optarg);
      }
      else if(0 == strcmp("max-seq-length", options[option_index].name) && (opt->algo_id == TMAP_MAP_ALGO_MAP1 || opt->algo_id == TMAP_MAP_ALGO_MAP2 || opt->algo_id == TMAP_MAP_ALGO_MAP4
                                                                            || opt->algo_id == TMAP_MAP_ALGO_MAP3 || opt->algo_id == TMAP_MAP_ALGO_MAPVSW)) {
          opt->max_seq_len = atoi(optarg);
      }
      // MAP1/MAP3
      else if(0 == strcmp("seed-length", options[option_index].name) && (opt->algo_id == TMAP_MAP_ALGO_MAP1 || opt->algo_id == TMAP_MAP_ALGO_MAP3)) {
          opt->seed_length = atoi(optarg);
          opt->seed_length_set = 1;
      }
      // MAP2/MAP3
      else if(0 == strcmp("max-seed-hits", options[option_index].name) && (opt->algo_id == TMAP_MAP_ALGO_MAP2 || opt->algo_id == TMAP_MAP_ALGO_MAP3)) {
          opt->max_seed_hits = atoi(optarg);
      }
      // MAP3/MAP4
      else if(0 == strcmp("hit-frac", options[option_index].name) && (opt->algo_id == TMAP_MAP_ALGO_MAP3 || opt->algo_id == TMAP_MAP_ALGO_MAP4)) {
          opt->hit_frac = atof(optarg);
      }
      else if(0 == strcmp("seed-step", options[option_index].name) && (opt->algo_id == TMAP_MAP_ALGO_MAP3 || opt->algo_id == TMAP_MAP_ALGO_MAP4)) {
          opt->seed_step = atoi(optarg);
      }
      // MAP1
      else if(0 == strcmp("seed-max-diff", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          opt->seed_max_diff = atoi(optarg);
      }
      else if(0 == strcmp("seed2-length", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          opt->seed2_length = atoi(optarg);
      }
      else if(0 == strcmp("max-diff", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          if(NULL != strstr(optarg, ".")) opt->max_diff = -1, opt->max_diff_fnr = atof(optarg);
          else opt->max_diff = atoi(optarg), opt->max_diff_fnr = -1.0;
      }
      else if(0 == strcmp("max-error-rate", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          opt->max_err_rate = atof(optarg);
      }
      else if(0 == strcmp("max-mismatches", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          if(NULL != strstr(optarg, ".")) opt->max_mm = -1, opt->max_mm_frac = atof(optarg);
          else opt->max_mm = atoi(optarg), opt->max_mm_frac = -1.0;
      }
      else if(0 == strcmp("max-gap-opens", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          if(NULL != strstr(optarg, ".")) opt->max_gapo = -1, opt->max_gapo_frac = atof(optarg);
          else opt->max_gapo = atoi(optarg), opt->max_gapo_frac = -1.0;
      }
      else if(0 == strcmp("max-gap-extensions", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          if(NULL != strstr(optarg, ".")) opt->max_gape = -1, opt->max_gape_frac = atof(optarg);
          else opt->max_gape = atoi(optarg), opt->max_gape_frac = -1.0;
      }
      else if(0 == strcmp("max-cals-deletion", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          opt->max_cals_del = atoi(optarg);
      }
      else if(0 == strcmp("indel-ends-bound", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          opt->indel_ends_bound = atoi(optarg);
      }
      else if(0 == strcmp("max-best-cals", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          opt->max_best_cals = atoi(optarg);
      }
      else if(0 == strcmp("max-nodes", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP1) {
          opt->max_entries = atoi(optarg);
      }
      // MAP2
      else if(0 == strcmp("length-coef", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP2) {
          opt->length_coef = atof(optarg);
      }
      else if(0 == strcmp("max-seed-intv", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP2) {
          opt->max_seed_intv = atoi(optarg);
      }
      else if(0 == strcmp("z-best", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP2) {
          opt->z_best= atoi(optarg);
      }
      else if(0 == strcmp("seeds-rev", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP2) {
          opt->seeds_rev = atoi(optarg);
      }
      else if(0 == strcmp("narrow-rmdup", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP2) {
          opt->narrow_rmdup = 1;
      }
      else if(0 == strcmp("max-chain-gap", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP2) {
          opt->max_chain_gap = atoi(optarg);
      }
      // MAP 3
      else if(0 == strcmp("hp-diff", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP3) {
          opt->hp_diff = atoi(optarg);
      }
      else if(0 == strcmp("fwd-search", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP3) {
          opt->fwd_search = 1;
      }
      else if(0 == strcmp("skip-seed-frac", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP3) {
          opt->skip_seed_frac = atof(optarg);
      }
      // MAP 4
      else if(0 == strcmp("min-seed-length", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP4) {
          opt->min_seed_length = atoi(optarg);
      }
      else if(0 == strcmp("max-seed-length", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP4) {
          opt->max_seed_length = atoi(optarg);
      }
      else if(0 == strcmp("max-seed-length-adj-coef", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP4) {
          opt->max_seed_length_adj_coef = atof(optarg);
      }
      else if(0 == strcmp("max-iwidth", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP4) {
          opt->max_iwidth = atoi(optarg);
      }
      else if(0 == strcmp("max-repr", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP4) {
          opt->max_repr = atoi(optarg);
      }
      else if(0 == strcmp("rand-repr", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP4) {
          opt->rand_repr = 1;
      }
      else if(0 == strcmp("use-min", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_MAP4) {
          opt->use_min = 1;
      }
      // STAGE
      else if(0 == strcmp("stage-score-thres", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_score_thr = atoi(optarg);
      }
      else if(0 == strcmp("stage-mapq-thres", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_mapq_thr = atoi(optarg);
      }
      else if(0 == strcmp("stage-keep-all", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_keep_all = 1;
      }
      else if(0 == strcmp("stage-seed-freq-cutoff", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_seed_freqc = atof(optarg);
      }
      else if(0 == strcmp("stage-seed-freq-cutoff-group-frac", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_seed_freqc_group_frac = atof(optarg);
      }
      else if(0 == strcmp("stage-seed-freq-cutoff-rand-repr", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_seed_freqc_rand_repr = atoi(optarg);
      }
      else if(0 == strcmp("stage-seed-freq-cutoff-min-groups", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_seed_freqc_min_groups = atoi(optarg);
      }
      else if(0 == strcmp("stage-seed-max-length", options[option_index].name) && opt->algo_id == TMAP_MAP_ALGO_STAGE) {
          opt->stage_seed_max_length = atoi(optarg);
      }
      // MAPALL
      else {
          break;
      }
      if(argc == optind) {
          val = 1;
      }
  }
  if(optind < argc) {
      val = 0;
      tmap_file_fprintf(tmap_file_stderr, "non-option command-line-elements:\n");
      while(optind < argc) {
          tmap_file_fprintf(tmap_file_stderr, "%s", argv[optind]);
          i = -1;
          if(opt->algo_id == TMAP_MAP_ALGO_MAPALL) {
              i = tmap_algo_name_to_id(argv[optind]);
              if(0 <= i) {
                  tmap_file_fprintf(tmap_file_stderr, ": recognized an algorithm name, did you forget to include the stage parameter?\n");
              }
          }
          if(i < 0) {
              tmap_file_fprintf(tmap_file_stderr, ": unknown command line option\n");
          }
          optind++;
      }
  }
  free(options);
  free(getopt_format);
  return val;
}

static int32_t
tmap_map_opt_file_check_with_null(char *fn1, char *fn2)
{
  if(NULL == fn1 && NULL == fn2) {
      return 0;
  }
  else if((NULL == fn1 && NULL != fn2)
          || (NULL != fn1 && NULL == fn2)) {
      return 1;
  }
  else if(0 != strcmp(fn1, fn2)) {
      return 1;
  }
  return 0;
}

// check that the global and flowspace options match the algorithm specific global options
void
tmap_map_opt_check_global(tmap_map_opt_t *opt_a, tmap_map_opt_t *opt_b) 
{
    int32_t i;
    // global
    if(0 != tmap_map_opt_file_check_with_null(opt_a->fn_fasta, opt_b->fn_fasta)) {
        tmap_error("option -f was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->fn_reads_num != opt_b->fn_reads_num) {
        tmap_error("option -r was specified outside of the common options", Exit, CommandLineArgument);
    }
    for(i=0;i<opt_a->fn_reads_num;i++) {
        if(0 != tmap_map_opt_file_check_with_null(opt_a->fn_reads[i], opt_b->fn_reads[i])) {
            tmap_error("option -r was specified outside of the common options", Exit, CommandLineArgument);
        }
    }
    if(0 != tmap_map_opt_file_check_with_null(opt_a->fn_sam, opt_b->fn_sam)) {
        tmap_error("option -s was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->reads_format != opt_b->reads_format) {
        tmap_error("option -i was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->bam_start_vfo != opt_b->bam_start_vfo) {
        tmap_error("option --bam-start-vfo was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->bam_end_vfo != opt_b->bam_end_vfo) {
        tmap_error("option --bam-end-vfo was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->use_param_ovr != opt_b->use_param_ovr) {
        tmap_error("option --par-ovr was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->score_match != opt_b->score_match) {
        tmap_error("option -A was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->pen_mm != opt_b->pen_mm) {
        tmap_error("option -M was specified outside of the common options", Warn, CommandLineArgument);
    }
    if(opt_a->pen_gapo != opt_b->pen_gapo) {
        tmap_error("option -O was specified outside of the common options", Warn, CommandLineArgument);
    }
    if(opt_a->pen_gape != opt_b->pen_gape) {
        tmap_error("option -E was specified outside of the common options", Warn, CommandLineArgument);
    }
    if(opt_a->pen_gapl != opt_b->pen_gapl) {
        tmap_error("option -G was specified outside of the common options", Warn, CommandLineArgument);
    }
    if(opt_a->gapl_len != opt_b->gapl_len) {
        tmap_error("option -K was specified outside of the common options", Warn, CommandLineArgument);
    }
    if(opt_a->bw != opt_b->bw) {
        tmap_error("option -w was specified outside of the common options", Warn, CommandLineArgument);
    }
    if(opt_a->softclip_type != opt_b->softclip_type) {
        tmap_error("option -g was specified outside of the common options", Warn, CommandLineArgument);
    }
    if(opt_a->dup_window != opt_b->dup_window) {
        tmap_error("option -W was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->max_seed_band != opt_b->max_seed_band) {
        tmap_error("option -B was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->unroll_banding != opt_b->unroll_banding) {
        tmap_error("option -U was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->score_thr != opt_b->score_thr) {
       tmap_error("option -T was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->reads_queue_size != opt_b->reads_queue_size) {
        tmap_error("option -q was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->num_threads != opt_b->num_threads) {
        tmap_error("option -n was specified outside of the common options", Exit, CommandLineArgument);
    }
    /* NB: "aln_output_mode" or "-a" may be modified by mapall */
    if(opt_a->sam_rg_num != opt_b->sam_rg_num) {
        tmap_error("option -R was specified outside of the common options", Exit, CommandLineArgument);
    }
    for(i=0;i<opt_a->sam_rg_num;i++) {
        if(0 != tmap_map_opt_file_check_with_null(opt_a->sam_rg[i], opt_b->sam_rg[i])) {
            tmap_error("option -R was specified outside of the common options", Exit, CommandLineArgument);
        }
    }
    if(opt_a->bidirectional != opt_b->bidirectional) {
        tmap_error("option -D was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->seq_eq != opt_b->seq_eq) {
        tmap_error("option -I was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->input_compr != opt_b->input_compr) {
        tmap_error("option -j or -z was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->output_type != opt_b->output_type) {
        tmap_error("option -o was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->shm_key != opt_b->shm_key) {
        tmap_error("option -k was specified outside of the common options", Exit, CommandLineArgument);
    }
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
    if(opt_a->sample_reads != opt_b->sample_reads) {
        tmap_error("option -x was specified outside of the common options", Exit, CommandLineArgument);
    }
#endif
    if(opt_a->vsw_type != opt_b->vsw_type) {
        tmap_error("option -H was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->vsw_fallback != opt_b->vsw_fallback) {
        tmap_error("option --vsw-fallback was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->long_hit_mult != opt_b->long_hit_mult) {
        tmap_error("option --long-hit-mult was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->end_repair != opt_b->end_repair) {
        tmap_error("option --end-repair was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->max_one_large_indel_rescue != opt_b->max_one_large_indel_rescue) {
        tmap_error("option --max-one-large-indel-rescue was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->min_anchor_large_indel_rescue != opt_b->min_anchor_large_indel_rescue) {
        tmap_error("option --min-anchor-large-indel-rescue was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->amplicon_overrun != opt_b->amplicon_overrun) {
        tmap_error("option --min-anchor-large-indel-rescue was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->max_adapter_bases_for_soft_clipping != opt_b->max_adapter_bases_for_soft_clipping) {
        tmap_error("option --max-adapter-bases-for-soft-clipping was specified outside of the common options", Exit, CommandLineArgument);
    }
    // flowspace
    if(opt_a->fscore != opt_b->fscore) {
        tmap_error("option -X was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->softclip_key != opt_b->softclip_key) {
        tmap_error("option -y was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->sam_flowspace_tags != opt_b->sam_flowspace_tags) {
        tmap_error("option -Y was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->ignore_flowgram != opt_b->ignore_flowgram) {
        tmap_error("option -S was specified outside of the common options", Exit, CommandLineArgument);
    }
    if(opt_a->aln_flowspace != opt_b->aln_flowspace) {
        tmap_error("option -F was specified outside of the common options", Exit, CommandLineArgument);
    }
    // pairing
    if(opt_a->pairing != opt_b->pairing) {
        tmap_error("option -Q was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->strandedness != opt_b->strandedness) {
        tmap_error("option -S was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->positioning != opt_b->positioning) {
        tmap_error("option -P was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->ins_size_mean != opt_b->ins_size_mean) {
        tmap_error("option -b specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->ins_size_std != opt_b->ins_size_std) {
        tmap_error("option -c was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->ins_size_std_max_num != opt_b->ins_size_std_max_num) {
        tmap_error("option -d was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->ins_size_outlier_bound != opt_b->ins_size_outlier_bound) {
        tmap_error("option -p was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->ins_size_min_mapq != opt_b->ins_size_min_mapq) {
        tmap_error("option -t was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->read_rescue != opt_b->read_rescue) {
        tmap_error("option -L was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->read_rescue_std_num != opt_b->read_rescue_std_num) {
        tmap_error("option -l was specified outside the common options", Exit, CommandLineArgument);
    }
    if(opt_a->read_rescue_mapq_thr != opt_b->read_rescue_mapq_thr) {
        tmap_error("option -m was specified outside the common options", Exit, CommandLineArgument);
    }
}

void
tmap_map_opt_check_stage(tmap_map_opt_t *opt_a, tmap_map_opt_t *opt_b) 
{
  if(opt_a->stage_score_thr != opt_b->stage_score_thr) {
      tmap_error("option --stage-score-thres specified outside of stage options", Exit, CommandLineArgument);
  }
  if(opt_a->stage_mapq_thr != opt_b->stage_mapq_thr) {
      tmap_error("option --stage-mapq-thres specified outside of stage options", Exit, CommandLineArgument);
  }
  if(opt_a->stage_keep_all != opt_b->stage_keep_all) {
      tmap_error("option --stage-keep-all specified outside of stage options", Exit, CommandLineArgument);
  }
  if(opt_a->stage_seed_freqc != opt_b->stage_seed_freqc) {
      tmap_error("option --stage-seed-freqc specified outside of stage options", Exit, CommandLineArgument);
  }
  if(opt_a->stage_seed_freqc_group_frac != opt_b->stage_seed_freqc_group_frac) {
      tmap_error("option --stage-seed-freqc-group-frac specified outside of stage options", Exit, CommandLineArgument);
  }
  if(opt_a->stage_seed_freqc_rand_repr != opt_b->stage_seed_freqc_rand_repr) {
      tmap_error("option --stage-seed-freqc-rand-repr specified outside of stage options", Exit, CommandLineArgument);
  }
  if(opt_a->stage_seed_freqc_min_groups != opt_b->stage_seed_freqc_min_groups) {
      tmap_error("option --stage-seed-freqc-min-groups specified outside of stage options", Exit, CommandLineArgument);
  }
  if(opt_a->stage_seed_max_length != opt_b->stage_seed_max_length) {
      tmap_error("option --stage-score-thres specified outside of stage options", Exit, CommandLineArgument);
  }
}

void
tmap_map_opt_check(tmap_map_opt_t *opt)
{
  // global 
  if(NULL == opt->fn_fasta && 0 == opt->shm_key) {
      tmap_error("option -f or option -k must be specified", Exit, CommandLineArgument);
  }
  else if(NULL != opt->fn_fasta && 0 < opt->shm_key) {
      tmap_error("option -f and option -k may not be specified together", Exit, CommandLineArgument);
  }
  if(0 == opt->fn_reads_num && TMAP_READS_FORMAT_UNKNOWN == opt->reads_format) {
      tmap_error("option -r or option -i must be specified", Exit, CommandLineArgument);
  }
  else if(1 < opt->fn_reads_num) {
      if(1 == opt->sam_flowspace_tags) {
          tmap_error("options -1 and -2 cannot be used with -Y", Exit, CommandLineArgument);
      }
      else if(0 != opt->pairing) {
          if(opt->strandedness < 0 || 1 < opt->strandedness) {
              tmap_error("option -S was not specified", Exit, CommandLineArgument);
          }
          else if(opt->positioning < 0 || 1 < opt->positioning) {
              tmap_error("option -P was not specified", Exit, CommandLineArgument);
          }
          else if(1 == opt->read_rescue) {
              if(opt->read_rescue_std_num < 0) {
                  tmap_error("option -l was not specified", Exit, CommandLineArgument);
              }
              tmap_error_cmd_check_int(opt->read_rescue_mapq_thr, 0, 255, "-m");
          }
      }
      // OK
  }
  if(TMAP_READS_FORMAT_UNKNOWN == opt->reads_format) {
      tmap_error("the reads format (-r/-i) was unrecognized", Exit, CommandLineArgument);
  }
  tmap_error_cmd_check_int64(opt->bam_start_vfo, 0, INT64_MAX, "--bam-start-vfo");
  tmap_error_cmd_check_int64(opt->bam_end_vfo, 0, INT64_MAX, "--bam-end-vfo");
  tmap_error_cmd_check_int(opt->use_param_ovr, 0, 1, "--par-ovr");
  tmap_error_cmd_check_int(opt->ovr_candeval, 0, 1, "--ovr-candeval");
  tmap_error_cmd_check_int(opt->confirm_vsw_corr, 0, 1, "--confirm-vsw-corr");
  tmap_error_cmd_check_int(opt->candidate_ext, 0, 1, "--no-candidate-ext");
  tmap_error_cmd_check_int(opt->correct_failed_vsw, 0, 1, "--no-err-fallback");
  tmap_error_cmd_check_int(opt->use_nvsw_on_nonstd_bases, 0, 1, "--no-nonstd-bases");
  tmap_error_cmd_check_int(opt->use_bed_in_end_repair, 0, 1, "--no-bed-er");
  tmap_error_cmd_check_int(opt->use_bed_in_mapq, 0, 1, "--no-bed-mapq");
  tmap_error_cmd_check_int(opt->use_bed_read_ends_stat, 0, 1, "--repair");
  tmap_error_cmd_check_int(opt->amplicon_scope, 0, INT32_MAX, "--ampl-scope");
  tmap_error_cmd_check_int(opt->score_match, 1, INT32_MAX, "-A");
  tmap_error_cmd_check_int(opt->pen_mm, 1, INT32_MAX, "-M");
  tmap_error_cmd_check_int(opt->pen_gapo, 1, INT32_MAX, "-O");
  tmap_error_cmd_check_int(opt->pen_gape, 1, INT32_MAX, "-E");
  if(-1 != opt->pen_gapl) tmap_error_cmd_check_int(opt->pen_gapl, 1, INT32_MAX, "-G");
  if(1 != opt->gapl_len) tmap_error_cmd_check_int(opt->gapl_len, 1, INT32_MAX, "-E");
  tmap_error_cmd_check_int(opt->bw, 0, INT32_MAX, "-w");
  tmap_error_cmd_check_int(opt->softclip_type, 0, 3, "-g");
  tmap_error_cmd_check_int(opt->softclip_key, 0, 1, "-y");
  tmap_error_cmd_check_int(opt->dup_window, -1, INT32_MAX, "-W");
  tmap_error_cmd_check_int(opt->max_seed_band, 1, INT32_MAX, "-B");
  tmap_error_cmd_check_int(opt->unroll_banding, 0, 1, "-U");
  tmap_error_cmd_check_int(opt->long_hit_mult, INT32_MIN, INT32_MAX, "--long-hit-mult");
  tmap_error_cmd_check_int(opt->score_thr, INT32_MIN, INT32_MAX, "-T");
  if(-1 != opt->reads_queue_size) tmap_error_cmd_check_int(opt->reads_queue_size, 1, INT32_MAX, "-q");
  tmap_error_cmd_check_int(opt->num_threads, 1, INT32_MAX, "-n");
  tmap_error_cmd_check_int(opt->aln_output_mode, 0, 3, "-a");
  // SAM RG
  if(0 < opt->sam_rg_num) {
      // sort them
      tmap_sort_introsort(tmap_map_opt_sort_rg, opt->sam_rg_num, opt->sam_rg);
  }
  tmap_error_cmd_check_int(opt->bidirectional, 0, 1, "-D");
  tmap_error_cmd_check_int(opt->seq_eq, 0, 1, "-I");
  tmap_error_cmd_check_int(opt->ignore_rg_sam_tags, 0, 1, "-C");
  /*
  if(0 == opt->ignore_rg_sam_tags && NULL != opt->sam_rg) {
      tmap_error("Must use -C with -R", Exit, CommandLineArgument);
  }
  */
  tmap_error_cmd_check_int(opt->rand_read_name, 0, 1, "-u");
  tmap_error_cmd_check_int(opt->output_type, 0, 2, "-o");
  tmap_error_cmd_check_int(opt->end_repair, 0, 100, "--end-repair");
  tmap_error_cmd_check_int(opt->max_adapter_bases_for_soft_clipping, 0, INT32_MAX, "max-adapter-bases-for-soft-clipping");
  tmap_error_cmd_check_int(opt->end_repair_5_prime_softclip, 0, 1, "--er-no5clip");

  tmap_error_cmd_check_int(opt->repair_min_freq, 0, INT32_MAX, "repair-min-freq");
  tmap_error_cmd_check_int(opt->repair_min_count, 0, INT32_MAX, "repair-min-count");
  tmap_error_cmd_check_int(opt->repair_min_adapter, 0, INT32_MAX, "repair-min-freq");
  tmap_error_cmd_check_int(opt->repair_max_overhang, 0, INT32_MAX, "repair-max-overhang");
  tmap_error_cmd_check_double(opt->repair_identity_drop_limit, 0.0, 1.0, "repair-identity-drop-limit");
  tmap_error_cmd_check_int(opt->repair_max_primer_zone_dist, 0, INT32_MAX, "repair-max-primer-zone-dist");
  tmap_error_cmd_check_int(opt->repair_clip_ext, 0, INT32_MAX, "repair-clip-ext");

#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  tmap_error_cmd_check_int(opt->sample_reads, 0, 1, "-x");
#endif
  tmap_error_cmd_check_int(opt->vsw_type, 1, 10, "-H");
  // Warn users
  switch(opt->vsw_type) {
    case 1:
    case 2:
    case 4:
    case 6:
      break;
    default:
      tmap_error("the option -H value has not been extensively tested; proceed with caution", Warn, CommandLineArgument);
      break;
  }
  tmap_error_cmd_check_int(opt->vsw_fallback, 1, 10, "-H");
  // Warn users
  switch(opt->vsw_fallback) {
    case 1:
    case 2:
    case 4:
    case 6:
      break;
    default:
      tmap_error("the option --vsw-fallback value has not been extensively tested; proceed with caution", Warn, CommandLineArgument);
      break;
  }
  // flowspace options
  tmap_error_cmd_check_int(opt->fscore, 0, INT32_MAX, "-X");
  tmap_error_cmd_check_int(opt->softclip_key, 0, 1, "-y");
  tmap_error_cmd_check_int(opt->sam_flowspace_tags, 0, 1, "-Y");
  tmap_error_cmd_check_int(opt->ignore_flowgram, 0, 1, "-S");
  tmap_error_cmd_check_int(opt->aln_flowspace, 0, 1, "-F");

  // pairing
  tmap_error_cmd_check_int(opt->pairing, 0, 2, "-Q");
  tmap_error_cmd_check_int(opt->strandedness, -1, 1, "-S");
  tmap_error_cmd_check_int(opt->positioning, -1, 1, "-P");
  tmap_error_cmd_check_int(opt->ins_size_mean+0.99, 0, INT32_MAX, "-b");
  tmap_error_cmd_check_int(opt->ins_size_std+0.99, 0, INT32_MAX, "-c");
  tmap_error_cmd_check_int(opt->ins_size_std_max_num+0.99, 0, INT32_MAX, "-d");
  tmap_error_cmd_check_int(opt->ins_size_outlier_bound+0.99, 0, INT32_MAX, "-p");
  tmap_error_cmd_check_int(opt->ins_size_min_mapq, 0, INT32_MAX, "-t");
  tmap_error_cmd_check_int(opt->read_rescue, 0, 1, "-L");
  tmap_error_cmd_check_int(opt->read_rescue_std_num+0.99, 0, INT32_MAX, "-l");
  tmap_error_cmd_check_int(opt->read_rescue_mapq_thr, 0, INT32_MAX, "-m");

  // realignment
  tmap_error_cmd_check_int(opt->do_realign, 0, 1, "--do-realign");
  tmap_error_cmd_check_int(opt->realign_mat_score, -127, 128, "--r-mat");
  tmap_error_cmd_check_int(opt->realign_mis_score, -127, 128, "--r-mis");
  tmap_error_cmd_check_int(opt->realign_gip_score, -127, 128, "--r-gip");
  tmap_error_cmd_check_int(opt->realign_gep_score, -127, 128, "--r-gep");
  tmap_error_cmd_check_int(opt->realign_bandwidth, 0, 256, "--r-bw");
  tmap_error_cmd_check_int(opt->realign_cliptype, 0, 4, "--r-clip");
  tmap_error_cmd_check_int(opt->realign_maxlen, 0, 4096, "--r-maxlen");
  tmap_error_cmd_check_int(opt->realign_maxclip, 0, 256, "--r-maxclip");

  // stats report  
  tmap_error_cmd_check_int(opt->report_stats, 0, 1, "--stats");

  // tail repeat clip
  tmap_error_cmd_check_int (opt->do_repeat_clip, 0, 1, "--do-repeat-clip");
  tmap_error_cmd_check_int (opt->repclip_continuation, 0, 1, "--repclip_cont");

  tmap_error_cmd_check_int (opt->cigar_sanity_check, 0, TMAP_MAP_SANITY_LASTVAL, "--cigar-sanity-check");

  // context dependent gap scores
  tmap_error_cmd_check_int (opt->do_hp_weight, 0, 1, "--context");
  tmap_error_cmd_check_int (opt->gap_scale_mode, 0, 2, "--hpscale");
  tmap_error_cmd_check_int (opt->context_mat_score, INT32_MIN, INT32_MAX, "--c-mat");
  tmap_error_cmd_check_int (opt->context_mis_score, INT32_MIN, INT32_MAX, "--c-mis");
  tmap_error_cmd_check_int (opt->context_gip_score, INT32_MIN, INT32_MAX, "--c-gip");
  tmap_error_cmd_check_int (opt->context_gep_score, INT32_MIN, INT32_MAX, "--c-gep");
  tmap_error_cmd_check_int (opt->context_extra_bandwidth, 0, 256, "--c-bw");
  tmap_error_cmd_check_int (opt->debug_log, 0, 1, "--debug-log");

  // DK: check if still required. warning: the logging is thread-safe, but locking file for the duration of alignment may cause deadlock.
  if (opt->num_threads > 1 && opt->debug_log)
      tmap_error ("Debug logging is available only in single-threaded mode", Exit, CommandLineArgument);
  if (!opt->realign_log && opt->debug_log)
      tmap_error ("Debug logging is available only when realign log file is specified", Exit, CommandLineArgument);

  // alignment length filtering
  tmap_error_cmd_check_int_x (opt->min_al_len, 0, INT32_MAX, MIN_AL_LEN_NOCHECK_SPECIAL, "--min-al-len");
  tmap_error_cmd_check_double_x (opt->min_al_cov, 0.0, 1.0, MIN_AL_COVERAGE_NOCHECK_SPECIAL, "--min-cov");
  assert (MIN_AL_IDENTITY_NOCHECK_SPECIAL == -DBL_MAX); // if this does not hold, next line should be un-commented instead of subsequenct one
  // tmap_error_cmd_check_double_x (opt->min_identity, -DBL_MAX, 1.0, MIN_AL_IDENTITY_NOCHECK_SPECIAL, "--min-iden");
  tmap_error_cmd_check_double (opt->min_identity, -DBL_MAX, 1.0, "--min-iden");


  // stage/algorithm options
  if(-1 != opt->min_seq_len) tmap_error_cmd_check_int(opt->min_seq_len, 1, INT32_MAX, "--min-seq-length");
  if(-1 != opt->max_seq_len) tmap_error_cmd_check_int(opt->max_seq_len, 1, INT32_MAX, "--max-seq-length");
  if(-1 != opt->min_seq_len && -1 != opt->max_seq_len && opt->max_seq_len < opt->min_seq_len) {
      tmap_error("The minimum sequence length must be less than the maximum sequence length (--min-seq-length and --max-seq-length)", Exit, CommandLineArgument);
  }
  switch(opt->algo_id) {
    case TMAP_MAP_ALGO_MAP1: // map1 options
      if(-1 != opt->seed_length) tmap_error_cmd_check_int(opt->seed_length, 1, INT32_MAX, "--seed-length");
      tmap_error_cmd_check_int(opt->seed_max_diff, 0, INT32_MAX, "--seed-max-diff");
      if(-1 != opt->seed2_length) tmap_error_cmd_check_int(opt->seed2_length, 1, INT32_MAX, "--seed2-length");
      if(-1 != opt->seed_length && -1 != opt->seed2_length) {
          tmap_error_cmd_check_int(opt->seed_length, 1, opt->seed2_length, "The secondary seed length (--seed2-length) must be greater than the primary seed length (--seed-length)");
      }
      tmap_error_cmd_check_int((opt->max_diff_fnr < 0) ? opt->max_diff: (int32_t)opt->max_diff_fnr, 0, INT32_MAX, "--max-diff");
      tmap_error_cmd_check_int((int32_t)opt->max_err_rate, 0, INT32_MAX, "--max-error-rate");
      // this will take care of the case where they are both < 0
      tmap_error_cmd_check_int((opt->max_mm_frac < 0) ? opt->max_mm : (int32_t)opt->max_mm_frac, 0, INT32_MAX, "--max-mismatches");
      // this will take care of the case where they are both < 0
      tmap_error_cmd_check_int((opt->max_gapo_frac < 0) ? opt->max_gapo : (int32_t)opt->max_gapo_frac, 0, INT32_MAX, "--max-gap-opens");
      // this will take care of the case where they are both < 0
      tmap_error_cmd_check_int((opt->max_gape_frac < 0) ? opt->max_gape : (int32_t)opt->max_gape_frac, 0, INT32_MAX, "--max-gap-extensions");
      tmap_error_cmd_check_int(opt->max_cals_del, 1, INT32_MAX, "--max-cals-deletion");
      tmap_error_cmd_check_int(opt->indel_ends_bound, 0, INT32_MAX, "--indels-ends-bound");
      tmap_error_cmd_check_int(opt->max_best_cals, 0, INT32_MAX, "--max-best-cals");
      tmap_error_cmd_check_int(opt->max_entries, 1, INT32_MAX, "--max-nodes");
      break;
    case TMAP_MAP_ALGO_MAP2:
      //tmap_error_cmd_check_int(opt->mask_level, 0, 1, "-m");
      tmap_error_cmd_check_int(opt->length_coef, 0, INT32_MAX, "--length-coef");
      tmap_error_cmd_check_int(opt->max_seed_intv, 0, INT32_MAX, "--max-seed-intv");
      tmap_error_cmd_check_int(opt->z_best, 1, INT32_MAX, "--z-best");
      tmap_error_cmd_check_int(opt->seeds_rev, 0, INT32_MAX, "--seeds-rev");
      tmap_error_cmd_check_int(opt->narrow_rmdup, 0, 1, "--narrow-rmdup");
      tmap_error_cmd_check_int(opt->max_seed_hits, 1, INT32_MAX, "--max-seed-hits");
      tmap_error_cmd_check_int(opt->max_chain_gap, 0, INT32_MAX, "--max-chain-gap");
      break;
    case TMAP_MAP_ALGO_MAP3:
      if(-1 != opt->seed_length) tmap_error_cmd_check_int(opt->seed_length, 1, INT32_MAX, "--seed-length");
      tmap_error_cmd_check_int(opt->max_seed_hits, 1, INT32_MAX, "--max-seed-hits");
      tmap_error_cmd_check_int(opt->hp_diff, 0, INT32_MAX, "--hp-diff");
      tmap_error_cmd_check_int(opt->hit_frac, 0, 1, "--hit-frac");
      if(-1 != opt->seed_step) tmap_error_cmd_check_int(opt->seed_step, 1, INT32_MAX, "--seed-step");
      tmap_error_cmd_check_int(opt->skip_seed_frac, 0, 1, "--skip-seed-frac");
      break;
    case TMAP_MAP_ALGO_MAP4:
      if(-1 != opt->min_seed_length) {
          tmap_error_cmd_check_int(opt->min_seed_length, 1, INT32_MAX, "--min-seed-length");
          if(opt->max_seed_length < opt->min_seed_length) {
              tmap_error("--max-seed-length is less than --min-seed-length", Exit, CommandLineArgument);
          }
      }
      if(-1 != opt->max_seed_length) tmap_error_cmd_check_int(opt->max_seed_length, 1, INT32_MAX, "--max-seed-length");
      if(0 < opt->max_seed_length_adj_coef) tmap_error_cmd_check_int(opt->max_seed_length_adj_coef, 0, INT32_MAX, "--max-seed-length-adj-coef");
      tmap_error_cmd_check_int(opt->hit_frac, 0, 1, "--hit-frac");
      if(-1 != opt->seed_step) tmap_error_cmd_check_int(opt->seed_step, 1, INT32_MAX, "--seed-step");
      tmap_error_cmd_check_int(opt->max_iwidth, 0, INT32_MAX, "--max-iwidth");
      tmap_error_cmd_check_int(opt->max_repr, 0, INT32_MAX, "--max-repr");
      tmap_error_cmd_check_int(opt->rand_repr, 0, 1, "--rand-repr");
      tmap_error_cmd_check_int(opt->use_min, 0, 1, "--use-min");
      break;
    case TMAP_MAP_ALGO_MAPALL:
      if(0 == opt->num_sub_opts) {
          tmap_error("no stages/algorithms given", Exit, CommandLineArgument);
      }
      break;
    case TMAP_MAP_ALGO_STAGE:
      // stage options
      tmap_error_cmd_check_int(opt->stage_score_thr, INT32_MIN, INT32_MAX, "--stage-score-thres");
      tmap_error_cmd_check_int(opt->stage_mapq_thr, 0, 255, "--stage-mapq-thres");
      tmap_error_cmd_check_int(opt->stage_keep_all, 0, 1, "--stage-keep-all");
      tmap_error_cmd_check_int(opt->stage_seed_freqc, 0.0, 1.0, "--stage-seed-freq-cutoff");
      tmap_error_cmd_check_int(opt->stage_seed_freqc_group_frac, 0.0, 1.0, "--stage-seed-freq-cutoff-group-frac");
      tmap_error_cmd_check_int(opt->stage_seed_freqc_rand_repr, 0, INT32_MAX, "--stage-seed-freq-cutoff-rand-repr");
      tmap_error_cmd_check_int(opt->stage_seed_freqc_min_groups, 0, INT32_MAX, "--stage-seed-freq-cutoff-min-groups");
      if(-1 != opt->stage_seed_max_length) tmap_error_cmd_check_int(opt->stage_seed_max_length, 1, INT32_MAX, "--stage-max-seed-length");
      break;
    default:
      break;
  }
  /*
  // check sub-options
  for(i=0;i<opt->num_sub_opts;i++) {
      // check mapping algorithm specific options
      tmap_map_opt_check(opt->sub_opts[i]);

      // check that common values match other opt values
      tmap_map_opt_check_common(opt, opt->sub_opts[i]);
  }
  */
}

void
tmap_map_opt_copy_global(tmap_map_opt_t *opt_dest, tmap_map_opt_t *opt_src)
{
    int i;

    // global options
    opt_dest->fn_fasta = tmap_strdup(opt_src->fn_fasta);
    opt_dest->bed_file = tmap_strdup(opt_src->bed_file);
    opt_dest->fn_reads_num = opt_src->fn_reads_num;
    opt_dest->fn_reads = tmap_malloc(sizeof(char*)*opt_dest->fn_reads_num, "opt_dest->fn_reads");
    for(i=0;i<opt_dest->fn_reads_num;i++) {
        opt_dest->fn_reads[i] = tmap_strdup(opt_src->fn_reads[i]);
    }
    opt_dest->reads_format = opt_src->reads_format;
    opt_dest->fn_sam = tmap_strdup(opt_src->fn_sam);
    opt_dest->bam_start_vfo = opt_src->bam_start_vfo;
    opt_dest->bam_end_vfo = opt_src->bam_end_vfo;
    opt_dest->use_param_ovr = opt_src->use_param_ovr;
    opt_dest->confirm_vsw_corr = opt_src->confirm_vsw_corr;
    opt_dest->candidate_ext = opt_src->candidate_ext;
    opt_dest->correct_failed_vsw = opt_src->correct_failed_vsw;
    opt_dest->use_nvsw_on_nonstd_bases= opt_src->use_nvsw_on_nonstd_bases;
    opt_dest->ovr_candeval = opt_src->ovr_candeval;
    opt_dest->use_bed_in_end_repair = opt_src->use_bed_in_end_repair;
    opt_dest->use_bed_in_mapq = opt_src->use_bed_in_mapq;
    opt_dest->use_bed_read_ends_stat = opt_src->use_bed_read_ends_stat;
    opt_dest->amplicon_scope = opt_src->amplicon_scope;
    opt_dest->score_match = opt_src->score_match;
    opt_dest->pen_mm = opt_src->pen_mm;
    opt_dest->pen_gapo = opt_src->pen_gapo;
    opt_dest->pen_gape = opt_src->pen_gape;
    opt_dest->pen_gapl = opt_src->pen_gapl;
    opt_dest->gapl_len = opt_src->gapl_len;
    opt_dest->use_new_QV = opt_src->use_new_QV;
    opt_dest->prefix_exclude =  opt_src->prefix_exclude;
    opt_dest->suffix_exclude = opt_src->suffix_exclude;
    opt_dest->bw = opt_src->bw;
    opt_dest->softclip_type = opt_src->softclip_type;
    opt_dest->dup_window = opt_src->dup_window;
    opt_dest->max_seed_band = opt_src->max_seed_band;
    opt_dest->unroll_banding = opt_src->unroll_banding;
    opt_dest->long_hit_mult = opt_src->long_hit_mult;
    opt_dest->score_thr = opt_src->score_thr;
    opt_dest->reads_queue_size = opt_src->reads_queue_size;
    opt_dest->num_threads = opt_src->num_threads;
    if(TMAP_MAP_ALGO_STAGE == opt_dest->algo_id) {
        opt_dest->aln_output_mode = opt_src->aln_output_mode;
    }
    else {
        opt_dest->aln_output_mode = TMAP_MAP_OPT_ALN_MODE_ALL;
    }
    opt_dest->sam_rg_num = opt_src->sam_rg_num; 
    opt_dest->sam_rg = tmap_malloc(sizeof(char*)*opt_dest->sam_rg_num, "opt_dest->sam_rg");
    for(i=0;i<opt_dest->sam_rg_num;i++) {
        opt_dest->sam_rg[i] = tmap_strdup(opt_src->sam_rg[i]);
    }
    opt_dest->bidirectional = opt_src->bidirectional;
    opt_dest->seq_eq = opt_src->seq_eq;
    opt_dest->ignore_rg_sam_tags = opt_src->ignore_rg_sam_tags;
    opt_dest->rand_read_name = opt_src->rand_read_name;
    opt_dest->input_compr = opt_src->input_compr;
    opt_dest->output_type = opt_src->output_type;
    opt_dest->end_repair = opt_src->end_repair;
    opt_dest->max_one_large_indel_rescue = opt_src->max_one_large_indel_rescue;
    opt_dest->min_anchor_large_indel_rescue = opt_src->min_anchor_large_indel_rescue;
    opt_dest->amplicon_overrun = opt_src->amplicon_overrun;
    opt_dest->max_adapter_bases_for_soft_clipping = opt_src->max_adapter_bases_for_soft_clipping;
    opt_dest->end_repair_5_prime_softclip = opt_src->end_repair_5_prime_softclip;
    opt_dest->repair_min_freq = opt_src->repair_min_freq;
    opt_dest->repair_min_count = opt_src->repair_min_count;
    opt_dest->repair_min_adapter = opt_src->repair_min_adapter;
    opt_dest->repair_max_overhang = opt_src->repair_max_overhang;
    opt_dest->repair_identity_drop_limit = opt_src->repair_identity_drop_limit;
    opt_dest->repair_max_primer_zone_dist = opt_src->repair_max_primer_zone_dist;
    opt_dest->repair_clip_ext = opt_src->repair_clip_ext;
    opt_dest->shm_key = opt_src->shm_key;
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
    opt_dest->sample_reads = opt_src->sample_reads;
#endif
    opt_dest->vsw_type = opt_src->vsw_type;
    opt_dest->vsw_fallback = opt_src->vsw_fallback;

    // realignment control
    opt_dest->do_realign = opt_src->do_realign;
    opt_dest->realign_mat_score = opt_src->realign_mat_score;
    opt_dest->realign_mis_score = opt_src->realign_mis_score;
    opt_dest->realign_gip_score = opt_src->realign_gip_score;
    opt_dest->realign_gep_score = opt_src->realign_gep_score;
    opt_dest->realign_bandwidth = opt_src->realign_bandwidth;
    opt_dest->realign_cliptype = opt_src->realign_cliptype;

    // context control
    opt_dest->do_hp_weight = opt_src->do_hp_weight;
    opt_dest->gap_scale_mode = opt_src->gap_scale_mode;
    opt_dest->context_mat_score = opt_src->context_mat_score;
    opt_dest->context_mis_score = opt_src->context_mis_score;
    opt_dest->context_gip_score = opt_src->context_gip_score;
    opt_dest->context_gep_score = opt_src->context_gep_score;
    opt_dest->context_extra_bandwidth = opt_src->context_extra_bandwidth;

    // repeat tail clipping
    opt_dest->do_repeat_clip = opt_src->do_repeat_clip;
    opt_dest->repclip_continuation = opt_src->repclip_continuation;

    // sanity checking
    opt_dest->cigar_sanity_check = opt_src->cigar_sanity_check;

    opt_dest->debug_log = opt_src->debug_log;
    opt_dest->cigar_sanity_check = opt_src->cigar_sanity_check;

    // flowspace options
    opt_dest->fscore = opt_src->fscore;
    opt_dest->softclip_key = opt_src->softclip_key;
    opt_dest->sam_flowspace_tags = opt_src->sam_flowspace_tags;
    opt_dest->ignore_flowgram = opt_src->ignore_flowgram;
    opt_dest->aln_flowspace = opt_src->aln_flowspace;

    // pairing
    opt_dest->pairing = opt_src->pairing;
    opt_dest->strandedness = opt_src->strandedness;
    opt_dest->positioning = opt_src->positioning;
    opt_dest->ins_size_mean = opt_src->ins_size_mean;
    opt_dest->ins_size_std = opt_src->ins_size_std;
    opt_dest->ins_size_std_max_num = opt_src->ins_size_std_max_num;
    opt_dest->ins_size_outlier_bound = opt_src->ins_size_outlier_bound;
    opt_dest->ins_size_min_mapq = opt_src->ins_size_min_mapq;
    opt_dest->read_rescue = opt_src->read_rescue;
    opt_dest->read_rescue_std_num = opt_src->read_rescue_std_num;
    opt_dest->read_rescue_mapq_thr = opt_src->read_rescue_mapq_thr;
}

void
tmap_map_opt_copy_stage(tmap_map_opt_t *opt_dest, tmap_map_opt_t *opt_src)
{
  opt_dest->stage_score_thr = opt_src->stage_score_thr;
  opt_dest->stage_mapq_thr = opt_src->stage_mapq_thr;
  opt_dest->stage_keep_all = opt_src->stage_keep_all;
  opt_dest->stage_seed_freqc = opt_src->stage_seed_freqc;
  opt_dest->stage_seed_freqc_group_frac = opt_src->stage_seed_freqc_group_frac;
  opt_dest->stage_seed_freqc_rand_repr = opt_src->stage_seed_freqc_rand_repr;
  opt_dest->stage_seed_freqc_min_groups = opt_src->stage_seed_freqc_min_groups;
  opt_dest->stage_seed_max_length = opt_src->stage_seed_max_length;
}

void
tmap_map_opt_print(tmap_map_opt_t *opt)
{
  int32_t i;
  fprintf(stderr, "algo_id=%d\n", opt->algo_id);
  fprintf(stderr, "fn_fasta=%s\n", opt->fn_fasta);
  for(i=0;i<opt->fn_reads_num;i++) {
      if(0 < i) fprintf(stderr, ",");
      fprintf(stderr, "fn_reads=%s", opt->fn_reads[i]);
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "fn_sam=%s\n", opt->fn_sam);
  fprintf(stderr, "reads_format=%d\n", opt->reads_format);
  fprintf(stderr, "bam_start_vfo=%ld\n", opt->bam_start_vfo);
  fprintf(stderr, "bam_end_vfo=%ld\n", opt->bam_end_vfo);
  fprintf(stderr, "use_param_ovr=%d\n", opt->use_param_ovr);
  fprintf(stderr, "ovr_candeval=%d\n", opt->ovr_candeval);
  fprintf(stderr, "use_bed_in_end_repair=%d\n", opt->use_bed_in_end_repair);
  fprintf(stderr, "use_bed_in_mapq=%d\n", opt->use_bed_in_mapq);
  fprintf(stderr, "use_bed_read_ends_stat=%d\n", opt->use_bed_read_ends_stat);
  fprintf(stderr, "amplicon_scope=%d\n", opt->amplicon_scope);
  fprintf(stderr, "score_match=%d\n", opt->score_match);
  fprintf(stderr, "pen_mm=%d\n", opt->pen_mm);
  fprintf(stderr, "pen_gapo=%d\n", opt->pen_gapo);
  fprintf(stderr, "pen_gape=%d\n", opt->pen_gape);
  fprintf(stderr, "pen_gapl=%d\n", opt->pen_gapl);
  fprintf(stderr, "gapl_len=%d\n", opt->gapl_len);
  fprintf(stderr, "fscore=%d\n", opt->fscore);
  fprintf(stderr, "bw=%d\n", opt->bw);
  fprintf(stderr, "softclip_type=%d\n", opt->softclip_type);
  fprintf(stderr, "softclip_key=%d\n", opt->softclip_key);
  fprintf(stderr, "dup_window=%d\n", opt->dup_window);
  fprintf(stderr, "max_seed_band=%d\n", opt->max_seed_band);
  fprintf(stderr, "unroll_banding=%d\n", opt->unroll_banding);
  fprintf(stderr, "long_hit_mult=%lf\n", opt->long_hit_mult);
  fprintf(stderr, "score_thr=%d\n", opt->score_thr);
  fprintf(stderr, "reads_queue_size=%d\n", opt->reads_queue_size);
  fprintf(stderr, "num_threads=%d\n", opt->num_threads);
  fprintf(stderr, "aln_output_mode=%d\n", opt->aln_output_mode);
  for(i=0;i<opt->sam_rg_num;i++) {
      if(0 < i) fprintf(stderr, ",");
      fprintf(stderr, "sam_rg=%s", opt->sam_rg[i]);
  }
  fprintf(stderr, "bidirectional=%d\n", opt->bidirectional);
  fprintf(stderr, "seq_eq=%d\n", opt->seq_eq);
  fprintf(stderr, "ignore_rg_sam_tags=%d\n", opt->ignore_rg_sam_tags);
  fprintf(stderr, "rand_read_name=%d\n", opt->rand_read_name);
  fprintf(stderr, "sam_flowspace_tags=%d\n", opt->sam_flowspace_tags);
  fprintf(stderr, "ignore_flowgram=%d\n", opt->ignore_flowgram);
  fprintf(stderr, "aln_flowspace=%d\n", opt->aln_flowspace);
  fprintf(stderr, "input_compr=%d\n", opt->input_compr);
  fprintf(stderr, "output_type=%d\n", opt->output_type);
  fprintf(stderr, "end_repair=%d\n", opt->end_repair);
  fprintf(stderr, "max-one-large-indel-rescue=%d\n", opt->max_one_large_indel_rescue);
  fprintf(stderr, "min-anchor-large-indel-rescue=%d\n",opt->min_anchor_large_indel_rescue);
  fprintf(stderr, "max-amplicon-overrun-indel-rescue=%d", opt->amplicon_overrun);
  fprintf(stderr, "max_adapter_bases_for_soft_clipping=%d\n", opt->max_adapter_bases_for_soft_clipping);
  fprintf(stderr, "end_repair_5_prime_softclip=%d\n", opt->end_repair_5_prime_softclip);
  fprintf(stderr, "repair_min_freq=%d", opt->repair_min_freq);
  fprintf(stderr, "repair_min_count=%d", opt->repair_min_count);
  fprintf(stderr, "repair_min_adapter=%d", opt->repair_min_adapter);
  fprintf(stderr, "repair_max_overhang=%d", opt->repair_max_overhang);
  fprintf(stderr, "repair_identity_drop_limit=%f", opt->repair_identity_drop_limit);
  fprintf(stderr, "repair_max_primer_zone_dist=%d", opt->repair_max_primer_zone_dist);
  fprintf(stderr, "repair_clip_ext=%d", opt->repair_clip_ext);
  fprintf(stderr, "shm_key=%d\n", (int)opt->shm_key);
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  fprintf(stderr, "sample_reads=%lf\n", opt->sample_reads);
#endif
  fprintf(stderr, "vsw_type=%d\n", opt->vsw_type);
  fprintf(stderr, "vsw_fallback=%d\n", opt->vsw_fallback);
  fprintf(stderr, "confirm_vsw_corr=%d\n", opt->confirm_vsw_corr);
  fprintf(stderr, "candidate_ext=%d\n", opt->candidate_ext);
  fprintf(stderr, "correct_failed_vsw=%d\n", opt->correct_failed_vsw);
  fprintf(stderr, "use_nvsw_on_nonstd_bases=%d\n", opt->use_nvsw_on_nonstd_bases);
  fprintf(stderr, "min_seq_len=%d\n", opt->min_seq_len);
  fprintf(stderr, "max_seq_len=%d\n", opt->max_seq_len);
  fprintf(stderr, "seed_length=%d\n", opt->seed_length);
  fprintf(stderr, "seed_length_set=%d\n", opt->seed_length_set);
  fprintf(stderr, "seed_max_diff=%d\n", opt->seed_max_diff);
  fprintf(stderr, "seed2_length=%d\n", opt->seed2_length);
  fprintf(stderr, "max_diff=%d\n", opt->max_diff);
  fprintf(stderr, "max_diff_fnr=%lf\n", opt->max_diff_fnr);
  fprintf(stderr, "max_err_rate=%lf\n", opt->max_err_rate);
  fprintf(stderr, "max_mm=%d\n", opt->max_mm);
  fprintf(stderr, "max_mm_frac=%lf\n", opt->max_mm_frac);
  fprintf(stderr, "max_gapo=%d\n", opt->max_gapo);
  fprintf(stderr, "max_gapo_frac=%lf\n", opt->max_gapo_frac);
  fprintf(stderr, "max_gape=%d\n", opt->max_gape);
  fprintf(stderr, "max_gape_frac=%lf\n", opt->max_gape_frac);
  fprintf(stderr, "max_cals_del=%d\n", opt->max_cals_del);
  fprintf(stderr, "indel_ends_bound=%d\n", opt->indel_ends_bound);
  fprintf(stderr, "max_best_cals=%d\n", opt->max_best_cals);
  fprintf(stderr, "max_entries=%d\n", opt->max_entries);
  fprintf(stderr, "length_coef=%lf\n", opt->length_coef);
  fprintf(stderr, "max_seed_intv=%d\n", opt->max_seed_intv);
  fprintf(stderr, "z_best=%d\n", opt->z_best);
  fprintf(stderr, "seeds_rev=%d\n", opt->seeds_rev);
  fprintf(stderr, "narrow_rmdup=%d\n", opt->narrow_rmdup);
  fprintf(stderr, "max_seed_hits=%d\n", opt->max_seed_hits);
  fprintf(stderr, "max_chain_gap=%d\n", opt->max_chain_gap);
  fprintf(stderr, "narrow_rmdup=%d\n", opt->narrow_rmdup);
  fprintf(stderr, "hp_diff=%d\n", opt->hp_diff);
  fprintf(stderr, "hit_frac=%lf\n", opt->hit_frac);
  fprintf(stderr, "seed_step=%d\n", opt->seed_step);
  fprintf(stderr, "fwd_search=%d\n", opt->fwd_search);
  fprintf(stderr, "skip_seed_frac=%lf\n", opt->skip_seed_frac);
  fprintf(stderr, "stage_score_thr=%d\n", opt->stage_score_thr);
  fprintf(stderr, "stage_mapq_thr=%d\n", opt->stage_mapq_thr);
  fprintf(stderr, "stage_keep_all=%d\n", opt->stage_keep_all);
  fprintf(stderr, "stage_seed_freqc=%.2f\n", opt->stage_seed_freqc);
  fprintf(stderr, "stage_seed_freqc_group_frac=%.2f\n", opt->stage_seed_freqc_group_frac);
  fprintf(stderr, "stage_seed_freqc_rand_repr=%d\n", opt->stage_seed_freqc_rand_repr);
  fprintf(stderr, "stage_seed_freqc_min_groups=%d\n", opt->stage_seed_freqc_min_groups);
  fprintf(stderr, "stage_seed_max_length=%d\n", opt->stage_seed_max_length);
}
