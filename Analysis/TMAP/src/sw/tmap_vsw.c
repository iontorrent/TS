/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/* The MIT License

   Copyright (c) 2011 by Attractive Chaos <attractor@live.co.uk>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   */

#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include <config.h>
#include "../util/tmap_alloc.h"
#include "../util/tmap_error.h"
#include "../util/tmap_definitions.h"
#include "tmap_sw.h"
#include "tmap_vsw_definitions.h"
#include "tmap_vsw.h"

tmap_vsw_t*
tmap_vsw_init (const uint8_t *query, int32_t qlen,
                    int32_t query_start_clip, int32_t query_end_clip,
                    int32_t type,
                    int32_t fallback_type,
                    tmap_vsw_opt_t *opt)
{
    tmap_vsw_t *vsw = NULL;
    vsw = tmap_calloc (1, sizeof (tmap_vsw_t), "vsw");
    vsw->type = type;
    vsw->fallback_type = fallback_type;
    vsw->query_start_clip = query_start_clip;
    vsw->query_end_clip = query_end_clip;
    vsw->opt = opt;
    vsw->algorithm = tmap_vsw_wrapper_init (type);
    vsw->algorithm_fallback = tmap_vsw_wrapper_init (fallback_type);
    vsw->algorithm_default = tmap_vsw_wrapper_init (VSW_DEFAULT_METHOD);
    return vsw;
}

void tmap_vsw_set_params (tmap_vsw_t* vsw,
                          int32_t query_start_clip,
                          int32_t query_end_clip,
                          tmap_vsw_opt_t* opt )
{
    vsw->query_start_clip = query_start_clip;
    vsw->query_end_clip = query_end_clip;
    *(vsw->opt) = *opt;
}

void
tmap_vsw_destroy (tmap_vsw_t *vsw)
{
    if (NULL == vsw) 
        return;
    tmap_vsw_wrapper_destroy (vsw->algorithm);
    tmap_vsw_wrapper_destroy (vsw->algorithm_fallback);
    tmap_vsw_wrapper_destroy (vsw->algorithm_default);
    free (vsw);
}

#ifdef TMAP_VSW_DEBUG_CMP
static void
tmap_vsw_process_compare (tmap_vsw_t *vsw,
                         const uint8_t *query, int32_t qlen,
                         uint8_t *target, int32_t tlen, 
                         tmap_vsw_result_t *result,
                         int32_t *overflow, int32_t score_thr, int32_t dir)
{
  int32_t query_end_type0, target_end_type0, n_best_type0, score_type0;
  int32_t query_end_type1, target_end_type1, n_best_type1, score_type1;
  int32_t cmp_non_vsw = 0;

  query_end_type0 = target_end_type0 = n_best_type0 = score_type0 = 0;
  query_end_type1 = target_end_type1 = n_best_type1 = score_type1 = 0;

  // baseline
  if (0 == cmp_non_vsw) 
  {
      tmap_vsw_wrapper_process (vsw->algorithm_default,
                               target, tlen, 
                               query, qlen, 
                               vsw->opt->score_match,
                               -vsw->opt->pen_mm,
                               -vsw->opt->pen_gapo,
                               -vsw->opt->pen_gape,
                               dir, 
                               vsw->query_start_clip, vsw->query_end_clip, 
                               &score_type0, &target_end_type0, &query_end_type0, &n_best_type0);
  }
  else 
  {
      tmap_sw_param_t ap;
      tmap_sw_path_t path [1024];
      int32_t path_len;
      int32_t i, matrix [25];
      ap.matrix=matrix;
      for (i=0;i<25;i++) { 
          ap.matrix [i] = -(vsw->opt)->pen_mm; 
      } 
      for (i=0;i<4;i++) { 
          ap.matrix [i * 5 + i] = vsw->opt->score_match; 
      } 
      ap.gap_open = vsw->opt->pen_gapo; ap.gap_ext = vsw->opt->pen_gape; 
      ap.gap_end = vsw->opt->pen_gape; 
      ap.row = 5; 
      score_type0 = tmap_sw_clipping_core ((uint8_t*)target, tlen, (uint8_t*)query, qlen, &ap,
                                    vsw->query_start_clip, vsw->query_end_clip, 
                                    path, &path_len, dir);
  }
  // current
  tmap_vsw_wrapper_process (vsw->algorithm,
                           target, tlen, 
                           query, qlen, 
                           vsw->opt->score_match,
                           -vsw->opt->pen_mm,
                           -vsw->opt->pen_gapo,
                           -vsw->opt->pen_gape,
                           dir, 
                           vsw->query_start_clip, vsw->query_end_clip, 
                           &score_type1, &target_end_type1, &query_end_type1, &n_best_type1);

  if (0 != cmp_non_vsw) 
  {
      n_best_type0 = n_best_type1;
      target_end_type0 = target_end_type1;
      query_end_type0 = query_end_type1;
  }

  if (tlen <= target_end_type0) tmap_bug ();
  if (qlen <= query_end_type0) tmap_bug ();
  /*
  if (tlen <= target_end_type1) tmap_bug ();
  if (qlen <= query_end_type1) tmap_bug ();
  */

  if (score_type0 != score_type1
     || target_end_type0 != target_end_type1
     || query_end_type0 != query_end_type1
     || n_best_type0 != n_best_type1) 
  {
      int32_t i;
      fprintf (stderr, "in %s dir=%d\n", __func__, dir);
      fprintf (stderr, "query_start_clip=%d\n", vsw->query_start_clip);
      fprintf (stderr, "query_end_clip=%d\n", vsw->query_end_clip);
      fprintf (stderr, "qlen=%d tlen=%d\n", qlen, tlen);
      for (i=0;i<qlen;i++) 
      {
          fputc ("ACGTN"[query [i]], stderr);
      }
      fputc ('\n', stderr);
      for (i=0;i<tlen;i++) 
      {
          fputc ("ACGTN"[target [i]], stderr);
      }
      fputc ('\n', stderr);
      fprintf (stderr, "tlen=%d qlen=%d score=[%d,%d] target_end=[%d,%d] query_end=[%d,%d] n_best=[%d,%d]\n",
              tlen, qlen,
              score_type0, score_type1,
              target_end_type0, target_end_type1,
              query_end_type0, query_end_type1,
              n_best_type0, n_best_type1);
      do 
      {
          tmap_sw_param_t ap;
          tmap_sw_path_t *path = NULL;
          int32_t i, matrix [25], n_cigar, score, path_len;
          uint32_t *cigar = NULL;
          ap.matrix=matrix;
          for (i=0;i<25;i++) { 
              ap.matrix [i] = -(vsw->opt)->pen_mm; 
          } 
          for (i=0;i<4;i++) { 
              ap.matrix [i * 5 + i] = vsw->opt->score_match; 
          } 
          ap.gap_open = vsw->opt->pen_gapo; ap.gap_ext = vsw->opt->pen_gape; 
          ap.gap_end = vsw->opt->pen_gape; 
          ap.row = 5; 
          path_len = 0;
          path = tmap_calloc (1024, sizeof (tmap_sw_path_t), "path");
          score = tmap_sw_clipping_core ((uint8_t*)target, tlen, (uint8_t*)query, qlen, &ap,
                                        vsw->query_start_clip, vsw->query_end_clip, 
                                        path, &path_len, dir);
          // print out the path
          cigar = tmap_sw_path2cigar (path, path_len, &n_cigar);
          fprintf (stderr, "tmap_sw_clipping_core score=%d\n", score);
          for (i=0;i<n_cigar;i++) 
          {
              fprintf (stderr, "%d%c", cigar [i]>>4, "MIDNSHP"[cigar [i]&0xf]);
          }
          fputc ('\n', stderr);
          free (path);
          free (cigar);
      } while (0);
      // try the opposite direction
      // baseline
      tmap_vsw_wrapper_process (vsw->algorithm_default,
                               target, tlen, 
                               query, qlen, 
                               vsw->opt->score_match,
                               -vsw->opt->pen_mm,
                               -vsw->opt->pen_gapo,
                               -vsw->opt->pen_gape,
                               1-dir, 
                               vsw->query_start_clip, vsw->query_end_clip, 
                               &score_type0, &target_end_type0, &query_end_type0, &n_best_type0);
      fprintf (stderr, "baseline opposite reverse tlen=%d qlen=%d score=[%d,%d] target_end=[%d,%d] query_end=[%d,%d] n_best=[%d,%d]\n",
              tlen, qlen,
              score_type0, score_type1,
              target_end_type0, target_end_type1,
              query_end_type0, query_end_type1,
              n_best_type0, n_best_type1);
      // top coder
      for (i=0;i<tlen;i++) fputc ("ACGTN"[target [i]], stderr);
      fputc ('\t', stderr);
      for (i=0;i<qlen;i++) fputc ("ACGTN"[query [i]], stderr);
      fputc ('\t', stderr);
      fprintf (stderr, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
              vsw->query_start_clip, vsw->query_end_clip, 
              vsw->opt->score_match,
              -vsw->opt->pen_mm,
              -vsw->opt->pen_gapo,
              -vsw->opt->pen_gape,
              dir,
              -1, -1, -1, -1);
      fprintf (stderr, "Error: algorithms produced different results!\n");
      exit (1);
  }
}
#endif

static int32_t
tmap_vsw_process (tmap_vsw_t *vsw,
              const uint8_t *query, int32_t qlen,
              uint8_t *target, int32_t tlen, 
              tmap_vsw_result_t *result,
              int32_t *overflow, int32_t score_thr, 
              int32_t is_rev, int32_t direction,
              uint32_t confirm_conflicts,
              uint32_t fallback_failed,
              uint32_t fallback_noniupac,
              tmap_map_stats_t* stat)
{
    int32_t found_forward = 1, query_end, target_end, n_best, score = INT32_MIN, do_not_process = 0;
    // TODO: check potential overflow
    // TODO: check that gap penalties will not result in an overflow
    // TODO: check that the max/min alignment score do not result in an overflow
    int32_t vect_used = 1;
    int32_t fit_error = 0;

#ifdef TMAP_VSW_DEBUG
    if (1 == is_rev) 
    {
        int i;
        fprintf (stderr, "in %s is_rev=%d\n", __func__, is_rev);
        fprintf (stderr, "query_start_clip=%d\n", vsw->query_start_clip);
        fprintf (stderr, "query_end_clip=%d\n", vsw->query_end_clip);
        fprintf (stderr, "qlen=%d tlen=%d\n", qlen, tlen);
        for (i=0;i<qlen;i++) 
        {
            fputc ("ACGTN"[query [i]], stderr);
        }
        fputc ('\n', stderr);
        for (i=0;i<tlen;i++) 
        {
            fputc ("ACGTN"[target [i]], stderr);
        }
        fputc ('\n', stderr);
    }
#endif

    // DVK: vsw score asymmetry bug: 
    // TopCoder 4 ("Psyco") implementation is broken when there is a base not within [0-3] range in the sequence.
    // we'll detect such case and call default vsw (non-vectorized) for now.
    uint32_t non_std_base = 0;
    {
        int32_t i;
        for (i = 0; i != qlen && !non_std_base; ++i)
            non_std_base = (query [i] > 3);
        for (i = 0; i < tlen && !non_std_base; ++i) 
            non_std_base = (target [i] > 3); 
    }
    if (stat && non_std_base && vsw->type == 4)
    {
        if (is_rev) 
            ++stat->nonstd_base_fallbacks_rev;
        else
            ++stat->nonstd_base_fallbacks_fwd;
    }

#ifdef FORCE_VSW_4_NONSTD_BASE_ERROR_PROPAGATION // do not enable for production :)
    non_std_base = 0;
#else
    if (non_std_base && vsw->type == 4 && !fallback_noniupac)
        do_not_process = 1;
        // non_std_base = 0; // for now force vectorized processing with currently elected method. May need a way to specify different fallback method later
#endif 

    // update based on current problem
    query_end = target_end = n_best = 0;
    if (NULL != overflow) (*overflow) = 0;
    int32_t alignment_ok = 0;

    if (!do_not_process)
    {
        if (tlen <= tmap_vsw_wrapper_get_max_tlen (vsw->algorithm)
            && qlen <= tmap_vsw_wrapper_get_max_qlen (vsw->algorithm)
            && !(non_std_base && vsw->type == 4)) // DVK: TopCoder4 does not handle non-standard bases properly
        {
            ++ stat->vect_sw_calls;
            vect_used = 1;
            tmap_vsw_wrapper_process (vsw->algorithm,
                                    target, tlen, 
                                    query, qlen, 
                                    vsw->opt->score_match,
                                    -vsw->opt->pen_mm,
                                    -vsw->opt->pen_gapo,
                                    -vsw->opt->pen_gape,
                                    direction, 
                                    vsw->query_start_clip, vsw->query_end_clip, 
                                    &score, &target_end, &query_end, &n_best, &fit_error);
    #ifdef TMAP_VSW_DEBUG_CMP
            tmap_vsw_process_compare (vsw,
                                    query, qlen,
                                    target, tlen, 
                                    result,
                                    overflow, score_thr, direction);
    #endif
            if (fit_error)
            {
                alignment_ok = 0;
                ++ stat->vswfails;
            }
            else 
                alignment_ok = 1;
        }
        if (!alignment_ok && (fallback_failed || non_std_base)) // try fallback
        {
            if (vsw->fallback_type != vsw->type 
                && tlen <= tmap_vsw_wrapper_get_max_tlen (vsw->algorithm_fallback)
                && qlen <= tmap_vsw_wrapper_get_max_qlen (vsw->algorithm_fallback)
                && !(non_std_base && vsw->fallback_type == 4))
            {
                ++ stat->fallback_vsw_calls;
                fit_error = 0;
                vect_used = 1;
                tmap_vsw_wrapper_process (vsw->algorithm_fallback,
                                        target, tlen, 
                                        query, qlen, 
                                        vsw->opt->score_match,
                                        -vsw->opt->pen_mm,
                                        -vsw->opt->pen_gapo,
                                        -vsw->opt->pen_gape,
                                        direction, 
                                        vsw->query_start_clip, vsw->query_end_clip, 
                                        &score, &target_end, &query_end, &n_best, &fit_error);
                if (fit_error)
                {
                    alignment_ok = 0;
                    ++ stat->vswfails;
                }
                else
                    alignment_ok = 1;
            }
        }
        if (!alignment_ok && fallback_failed)
        { // try the default
            if (vsw->type != VSW_DEFAULT_METHOD && vsw->fallback_type != VSW_DEFAULT_METHOD)
            {
                ++ stat->fallback_sw_calls;
                fit_error = 0;
                vect_used = 0;
                tmap_vsw_wrapper_process (vsw->algorithm_default,
                                        target, tlen, 
                                        query, qlen, 
                                        vsw->opt->score_match,
                                        -vsw->opt->pen_mm,
                                        -vsw->opt->pen_gapo,
                                        -vsw->opt->pen_gape,
                                        direction, 
                                        vsw->query_start_clip, vsw->query_end_clip, 
                                        &score, &target_end, &query_end, &n_best, &fit_error);
                if (fit_error)
                    alignment_ok = 0; // irrelevant with current default SW implementation
                else
                    alignment_ok = 1; 
            }
        }
    }

    if (!alignment_ok || score < score_thr || 0 == n_best) 
    {
        query_end = target_end = -1;
        n_best = 0;
        score = INT32_MIN;
    }
    if (!alignment_ok && !do_not_process)
        ++stat->totswfails;

    if (0 == is_rev) 
    {

        result->query_end = query_end;
        result->target_end = target_end;
        result->n_best = n_best;
        result->score_fwd = score;

        // check forward results
        if (!alignment_ok 
            || result->score_fwd < score_thr
            || ((result->query_end == result->query_start || result->target_end == result->target_start) && result->score_fwd <= 0)
            || n_best <= 0)
        {
            // return if we found no legal/good forward results
            result->query_end = result->query_start = 0;
            result->target_end = result->target_start = 0;
            result->score_fwd = result->score_rev = INT16_MIN;
            result->n_best = 0;
            return INT32_MIN;
        }
        else if (-1 == result->query_end) 
        {
            tmap_bug ();
        }

        result->query_start = result->target_start = 0;
        result->score_rev = INT16_MIN;

#ifdef TMAP_VSW_DEBUG
        fprintf (stderr, "result->score_fwd=%d result->score_rev=%d\n",
                result->score_fwd, result->score_rev);
        fprintf (stderr, "{?-%d] {?-%d}\n",
                result->query_end,
                result->target_end);
#endif

        return result->score_fwd;
    }
    else 
    {
        result->query_start = qlen - query_end - 1;
        result->target_start = tlen - target_end - 1;
        result->n_best = n_best;
        result->score_rev = score;

#ifdef TMAP_VSW_DEBUG
        fprintf (stderr, "is_rev=%d result->score_fwd=%d result->score_rev=%d\n",
                is_rev, result->score_fwd, result->score_rev);
#endif

        // check reverse results
        if (!alignment_ok) 
        {
            result->query_end = result->query_start = 0;
            result->target_end = result->target_start = 0;
            result->score_fwd = result->score_rev = INT16_MIN;
            result->n_best = 0;
            return INT32_MIN; 
        }
        else if (result->score_fwd != result->score_rev) 
        { // something went wrong... FIXME
            // re-process both with the default, warn the user if initial score could not be confirmed
            // use the default
            if (stat)
            {
                if (vect_used)
                    ++stat->vswfails;
                ++stat->totswfails;
            }
            if (vsw->type != VSW_DEFAULT_METHOD && fallback_failed) 
            {
                int32_t def_vsw_score = INT_MIN;
                stat->fallback_sw_calls ++;
                tmap_vsw_wrapper_process (vsw->algorithm_default,
                                        target, tlen, 
                                        query, qlen, 
                                        vsw->opt->score_match,
                                        -vsw->opt->pen_mm,
                                        -vsw->opt->pen_gapo,
                                        -vsw->opt->pen_gape,
                                        direction, 
                                        vsw->query_start_clip, vsw->query_end_clip, 
                                        &def_vsw_score, &target_end, &query_end, &n_best, overflow);

                // DVK: trust the default method, but optionally confirm with the non-vectorized SW
                // LATER: (maybe) re-run forward algnment here to re-confirm score
                uint32_t accept_correction = 1;
                if (confirm_conflicts)
                {
                    tmap_sw_param_t ap;
                    int32_t i, matrix [25];
                    ap.matrix=matrix;
                    for ( i = 0; i < 25; ++i) 
                        ap.matrix [i] = -(vsw->opt)->pen_mm; 
                    for (i = 0; i < 4; ++i) 
                    { 
                        ap.matrix [i*5 + i] = vsw->opt->score_match; 
                    } 
                    ap.gap_open = vsw->opt->pen_gapo; 
                    ap.gap_ext = vsw->opt->pen_gape; 
                    ap.gap_end = vsw->opt->pen_gape; 
                    ap.row = 5;
                    int32_t sw_score = tmap_sw_clipping_core ((uint8_t*)target, tlen, (uint8_t*)query, qlen, &ap,
                                            vsw->query_start_clip, vsw->query_end_clip, 
                                            NULL, NULL, direction);
                    if (sw_score != def_vsw_score)
                    {
                        accept_correction = 0;
                        tmap_warning ("Unrecoverable internal error: VSW alignment score asymmetry. Forward score =  %d, reverse score = %d, default (\"gold-standard\") VSW score = %d, non-vectorized score = %d.\nCandidate ignored, read may be unmapped or mapped improperly", result->score_fwd, result->score_rev, def_vsw_score, sw_score);
                    }
                }
                if (accept_correction)
                {
                    // tmap_warning ("Correctable internal error: VSW alignment score assymetry. Forward score = %d, reverse score = %d,\n default (\"gold-standard\") VSW score = %d. Non-vectorized method matches reverse result; using confirmed score", result->score_fwd, result->score_rev, def_vsw_score);
                    if (stat)
                        ++stat->asymmetric_scores_corrected;
                    result->query_start = qlen - query_end - 1;
                    result->target_start = tlen - target_end - 1;
                    result->n_best = n_best;
                    result->score_rev = def_vsw_score;
                    result->score_fwd = def_vsw_score;
                    return score;
                }
            }
            else
               tmap_warning ("Unrecoverable internal error: VSW alignment score asymmetry in default method (#%d). Forward score =  %d, reverse score = %d\nCandidate ignored, read may be unmapped or mapped improperly", result->score_fwd, result->score_rev);
            // ignore candidate with conflict (warning already issued)
            if (stat)
            {
                ++stat->asymmetric_scores_failed;
                ++stat->totswfails;
            }
            result->query_end = result->query_start = 0;
            result->target_end = result->target_start = 0;
            result->score_fwd = result->score_rev = INT16_MIN;
            result->n_best = 0;
            return INT32_MIN;
        }
        else
        {
            if (stat)
                ++stat->symmetric_scores;
        }
        return result->score_fwd;
    }
}

int32_t
tmap_vsw_process_fwd (tmap_vsw_t *vsw,
              const uint8_t *query, int32_t qlen,
              uint8_t *target, int32_t tlen, 
              tmap_vsw_result_t *result,
              int32_t *overflow, int32_t score_thr, int32_t direction, 
              uint32_t confirm_conflicts, 
              uint32_t fallback_failed,
              uint32_t fallback_noniupac,
              tmap_map_stats_t* stat)
{
  return tmap_vsw_process (vsw, query, qlen, target, tlen, result, overflow, score_thr, 0, direction, confirm_conflicts, fallback_failed, fallback_noniupac, stat);
}

int32_t
tmap_vsw_process_rev (tmap_vsw_t *vsw,
              const uint8_t *query, int32_t qlen,
              uint8_t *target, int32_t tlen, 
              tmap_vsw_result_t *result,
              int32_t *overflow, int32_t score_thr, int32_t direction, 
              uint32_t confirm_conflicts, 
              uint32_t fallback_failed,
              uint32_t fallback_noniupac,
              tmap_map_stats_t* stat)
{
  return tmap_vsw_process (vsw, query, qlen, target, tlen, result, overflow, score_thr, 1, direction, confirm_conflicts, fallback_failed, fallback_noniupac, stat);
}
