/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <config.h>
#include <time.h>
#include <assert.h>
#ifdef HAVE_LIBPTHREAD
#include <pthread.h>
#endif
#include <stdio.h>
#include "../util/tmap_error.h"
#include "../util/tmap_alloc.h"
#include "../util/tmap_definitions.h"
#include "../util/tmap_progress.h"
#include "../util/tmap_sam_convert.h"
#include "../util/tmap_sort.h"
#include "../util/tmap_rand.h"
#include "../util/tmap_hash.h"
#include "../seq/tmap_seq.h"
#include "../index/tmap_refseq.h"
#include "../index/tmap_bwt_gen.h"
#include "../index/tmap_bwt.h"
#include "../index/tmap_bwt_match.h"
#include "../index/tmap_bwt_match_hash.h"
#include "../index/tmap_sa.h"
#include "../index/tmap_index.h"
#include "../io/tmap_seqs_io.h"
#include "../server/tmap_shm.h"
#include "../sw/tmap_fsw.h"
#include "../sw/tmap_sw.h"
#include "util/tmap_map_stats.h"
#include "util/tmap_map_util.h"
#include "pairing/tmap_map_pairing.h"

#include "tmap_map_driver.h"
#include "../realign/realign_c_util.h"

#define PARANOID_TESTS 0


// NB: do not turn these on, as they do not currently improve run time. They
// could be useful if many duplicate lookups are performed and the hash
// retrieval was fast...
//#define TMAP_DRIVER_USE_HASH 1
//#define TMAP_DRIVER_CLEAR_HASH_PER_READ 1

#define __tmap_map_sam_sort_score_lt(a, b) ((a).score > (b).score)
TMAP_SORT_INIT(tmap_map_sam_sort_score, tmap_map_sam_t, __tmap_map_sam_sort_score_lt)

// sorts integers
TMAP_SORT_INIT(tmap_map_driver_sort_isize, int32_t, tmap_sort_lt_generic);

static void
tmap_map_driver_do_init(tmap_map_driver_t *driver, tmap_refseq_t *refseq)
{
  int32_t i, j;
  for(i=0;i<driver->num_stages;i++) {
      tmap_map_driver_stage_t *stage = driver->stages[i];
      for(j=0;j<stage->num_algorithms;j++) {
          tmap_map_driver_algorithm_t *algorithm = stage->algorithms[j];
          if(NULL != algorithm->func_init 
             && 0 != algorithm->func_init(&algorithm->data, refseq, algorithm->opt)) {
              tmap_error("the thread function could not be initialized", Exit, OutOfRange);
          }
      }
  }
}

static void
tmap_map_driver_do_cleanup(tmap_map_driver_t *driver)
{
  int32_t i, j;
  for(i=0;i<driver->num_stages;i++) {
      tmap_map_driver_stage_t *stage = driver->stages[i];
      for(j=0;j<stage->num_algorithms;j++) {
          tmap_map_driver_algorithm_t *algorithm = stage->algorithms[j];
          if(NULL != algorithm->func_cleanup 
             && 0 != algorithm->func_cleanup(&algorithm->data)) {
              tmap_error("the thread function could not be cleaned up", Exit, OutOfRange);
          }
      }
  }
}

static void
tmap_map_driver_do_threads_init(tmap_map_driver_t *driver, 
                                int32_t tid)
{
  int32_t i, j;
  for(i=0;i<driver->num_stages;i++) {
      tmap_map_driver_stage_t *stage = driver->stages[i];
      for(j=0;j<stage->num_algorithms;j++) {
          tmap_map_driver_algorithm_t *algorithm = stage->algorithms[j];
          if(NULL != algorithm->func_thread_init 
             && 0 != algorithm->func_thread_init(&algorithm->thread_data[tid], 
                                                 algorithm->opt)) {
              tmap_error("the thread function could not be initialized", Exit, OutOfRange);
          }
      }
  }
}

static void
tmap_map_driver_do_threads_cleanup(tmap_map_driver_t *driver, int32_t tid)
{
  int32_t i, j;
  for(i=0;i<driver->num_stages;i++) {
      tmap_map_driver_stage_t *stage = driver->stages[i];
      for(j=0;j<stage->num_algorithms;j++) {
          tmap_map_driver_algorithm_t *algorithm = stage->algorithms[j];
          if(NULL != algorithm->func_thread_cleanup 
             && 0 != algorithm->func_thread_cleanup(&algorithm->thread_data[tid], algorithm->opt)) {
              tmap_error("the thread function could not be cleaned up", Exit, OutOfRange);
          }
      }
  }
}

static void
tmap_map_driver_init_seqs(tmap_seq_t **seqs, tmap_seq_t *seq, int32_t max_length)
{
  int32_t i;
  // init seqs
  for(i=0;i<4;i++) {
      // TODO: only if necessary
      seqs[i] = tmap_seq_clone(seq); // clone the sequence 
      // modify the length before reversing or reverse complimenting 
      if(0 < max_length && max_length < tmap_seq_get_bases_length(seq)) { // NB: does not modify quality string or other meta data
          tmap_seq_get_bases(seqs[i])->l = max_length;
          tmap_seq_get_bases(seqs[i])->s[max_length] = '\0';
      }
      switch(i) {
        case 0: // forward
          break;
        case 1: // reverse compliment
          tmap_seq_reverse_compliment(seqs[i]); break;
        case 2: // reverse
          tmap_seq_reverse(seqs[i]); break;
        case 3: // compliment
          tmap_seq_compliment(seqs[i]); break;
      }
      tmap_seq_to_int(seqs[i]); // convert to integers
  }
}

/*
 DVK - helpers for tail repeat finding
 To be:
 1) moved to separate place
 2) changed to linear-time implementation (using KMP preprocessing)
*/

#define mymin(a,b) ((a)<(b)?(a):(b))
#define mymax(a,b) ((a)<(b)?(b):(a))


/* this function finds maximal length of the peridicity suffix in text that match prefix in pattern
   returns first position in a text that belongs to such periodicity position */
unsigned tailrep_cont (unsigned pattlen, const char* pattern, unsigned textlen, const char* text)
{
    unsigned rep_beg = textlen;
    unsigned period;
    if (!textlen || !pattlen)
        return rep_beg;
    for (period = 1; period != pattlen; ++period)
    {
        unsigned ppos = period - 1;
        unsigned tpos = textlen - 1;
        uint8_t reached_beg = 0;
        for (;;)
        {
            if (text [tpos] != pattern [ppos])
                break;
            if (!tpos)
            {
                reached_beg = 1;
                break;
            }
            --tpos;
            if (!ppos)
                ppos = period - 1;
            else
                --ppos;
        }
        if (reached_beg)
        {
            rep_beg = 0;
            break;
        }
        if (tpos + period > textlen) // this enforces whatever is less then full period in read to stay unclipped
            continue;
        rep_beg = mymin (rep_beg, tpos+1);
    }
    return rep_beg;
}

/* this function finds maximal length of the periodicity suffix in text
   returns first position in a text that belongs to such periodicity position */
unsigned tailrep (unsigned textlen, const char* text, unsigned max_period)
{
    unsigned rep_beg = textlen;
    unsigned period;
    if (!textlen || !max_period)
        return rep_beg;
    for (period = 1; period != max_period; ++period)
    {
        if (textlen < period + 1)
            break;
        unsigned ppos = textlen - 1;
        unsigned tpos = textlen - period - 1;
        uint8_t reached_beg = 0;
        for (;;)
        {
            if (text [tpos] != text [ppos])
                break;
            if (!tpos)
            {
                reached_beg = 1;
                break;
            }
            --tpos;
            if (ppos == textlen - period)
                ppos = textlen - 1;
            else
                --ppos;
        }
        if (reached_beg)
        {
            rep_beg = 0;
            break;
        }
        if (tpos + 2*period > textlen)
             continue;
        rep_beg = mymin (rep_beg, tpos+1);
    }
    return rep_beg;
}

// puts all bases starting from trim_at positions into the tail clip zone;
// resizes the passed in cigar buffer if necessary, returns new cigar length
unsigned trim_cigar_right (uint32_t** cigar_buff, unsigned orig_cigar_sz, unsigned qry_len, unsigned trim_at)
{
    uint32_t* cigar = *cigar_buff;
    // copy the beginning of the cigar until query position reaches q_al_end;
    // put the reminder into the soft-clip (no matter if there was one already;
    // append the hard-clip if it was there (does it appear there at all?)

    // check presence of hard-clip at the end, remember it
    uint32_t end_hard_clip = 0;
    if (orig_cigar_sz && bam_cigar_op (cigar [orig_cigar_sz - 1]) == BAM_CHARD_CLIP)
        end_hard_clip = cigar [orig_cigar_sz - 1];

    // count number of operations in the transformed CIGAR
    int opno, op = -1, tail_beg = 0, prev_tail_beg = 0, constype;
    int new_soft_clip_idx = -1, new_hard_clip_idx = -1;
    int unclipped = 0;
    if (trim_at) // 0922 - do not enter loop if trimming at start
    {
        for (opno = 0; ; ++opno)  
        {
            if (opno == orig_cigar_sz) // 0922 just in case: added condition, which never should happen
            {
                fprintf (stderr, "Internal Cigar trimming error\n");
                tmap_bug ();
            }
            constype = bam_cigar_type (cigar [opno]);
            prev_tail_beg = tail_beg;
            if (constype & CONSUME_QRY) 
                tail_beg += bam_cigar_oplen (cigar [opno]);
            op = bam_cigar_op (cigar [opno]);
            if (op != BAM_CHARD_CLIP && op != BAM_CSOFT_CLIP && op != BAM_CPAD)
                unclipped = 1;
            if (trim_at <= tail_beg) 
                break;
        }
    }
    if (unclipped == 0)
        return 0;
    unsigned new_tail_soft_clip = 0;
    if (trim_at < tail_beg)  // replace last cigar op with just a piece of it
    {
        // trim the extent of current cigar operation
        uint32_t op = bam_cigar_op (cigar [opno]);
        cigar [opno] = op | ((trim_at - prev_tail_beg) << BAM_CIGAR_SHIFT);
        new_tail_soft_clip += tail_beg - trim_at;
    }   // otherwise, do not trim - keep entire last op in the cigar
    else
    {
        // if trimmed at the edge of cigar operator, 
        // iteratively remove any non-match operators starting from the tail one, incrementing clip_extension by the consumed read length
        // if reached beginning of cigar, return 0 (reject entire read)
        while (bam_cigar_op (cigar [opno]) != BAM_CMATCH)
        {
            if (bam_cigar_type (cigar [opno]) & 1) // consumes query
                new_tail_soft_clip += bam_cigar_oplen (cigar [opno]);
            if (opno == 0)
                return 0;
            -- opno;
        }
    }
    opno ++;
    new_tail_soft_clip += qry_len - tail_beg;
    // position for soft_clip;
    if (new_tail_soft_clip)
        new_soft_clip_idx = opno ++;
    if (end_hard_clip)
        new_hard_clip_idx = opno ++;
    if (opno > orig_cigar_sz)
        cigar = *cigar_buff = (uint32_t*) tmap_realloc (cigar, sizeof (uint32_t) * opno, "cigar_buff");
    if (new_soft_clip_idx != -1)
        cigar [new_soft_clip_idx] = BAM_CSOFT_CLIP | (new_tail_soft_clip << BAM_CIGAR_SHIFT);
    if (new_hard_clip_idx != -1)
        cigar [new_hard_clip_idx] = end_hard_clip;
    return opno;
}

unsigned trim_cigar_left (uint32_t** cigar_buff, unsigned orig_cigar_sz, unsigned trim_at, unsigned* ref_rep_off_p)
{
    uint32_t* cigar = *cigar_buff;
    // unconditionally add the left hard-clip if any;
    // skip the beginning of the cigar until query position reaches q_al_end;
    // put skipped zone into the soft-clip (no matter if there was one already);
    // add remining CIGAR segments (with first one possibly reduced)

    unsigned op_idx = 0; // index of current cigar operation
    unsigned op_end = 0; // end of current cigar operation in (reversed!) query coordinate (this function is called for reverse matches only)
                         // equal to the start position of NEXT item in cigar
    uint8_t in_lclip = 1; // flag indicating that we are still in the left clipping region (respective to the reference and cigar direction)
    uint8_t lhard_present = 0; // flag indicating that left hard clip is present
    uint8_t lsoft_present = 0; // flag indicating that left soft clip is present
    uint32_t op, op1; // cigar operation
    uint32_t oplen; // number of bases in cigar operation
    uint32_t constype; // ref/query consuming type, as returned by bam_cigar_type
    unsigned new_cigar_sz = 0; // number of items in new scigar (after trimming)

    *ref_rep_off_p = 0;

    // skip to the beginning of non-trimmed operations
    if (trim_at) 
    {
        for (;; ++op_idx)  
        {
            // track whether we are in clipped or non-clipped (aligned) portion of a read
            op = bam_cigar_op (cigar [op_idx]);
            if (in_lclip)
            {
                if (op == BAM_CHARD_CLIP)
                {
                    if (lhard_present)
                        return 0; // alignment already fully trimmed or invalid
                    lhard_present = 1, ++new_cigar_sz;
                }
                else if (op == BAM_CSOFT_CLIP)
                {
                    if (lsoft_present)
                        return 0; // alignment already fully trimmed or invalid
                    lsoft_present = 1, ++new_cigar_sz;
                }
                else if (op != BAM_CPAD)
                    in_lclip = 0;
            }

            if (!in_lclip && (op == BAM_CHARD_CLIP || op == BAM_CSOFT_CLIP || op == BAM_CPAD))
                return 0; // reached the clipped tail of the alignment but not the trim position

            // update op_beg and op_end to coordinates of current cigar item
            constype = bam_cigar_type (cigar [op_idx]);
            oplen = bam_cigar_oplen (cigar [op_idx]);
            if (constype & CONSUME_QRY) 
                op_end += oplen;

            if (constype & CONSUME_REF)
            {
                *ref_rep_off_p += oplen;
                if (op_end > trim_at) // can happen only on M(or X/=) operation, so safe
                    *ref_rep_off_p -= (op_end - trim_at);
            }

            if (op_end >= trim_at) // reached point at or beyond trim position
            // if (op_end > trim_at) // reached point at or beyond trim position
                break;


#if PARANOID_TESTS
            if (op_idx == orig_cigar_sz) // should never rich the end!
            {
                fprintf (stderr, "Internal Cigar trimming error: trim_cigar_left reached cigar end\n");
                tmap_bug ();
            }
#else
            assert (op_idx < orig_cigar_sz);
#endif
        }
    }
    if (in_lclip) // new clip is covered by already present one; no need for changes
        return orig_cigar_sz;

#if PARANOID_TESTS
    if (op_end < trim_at) // this is abberant case, should not happen
    {
        fprintf (stderr, "Internal Cigar trimming error: trim_cigar_left: trim position seem to be beyond the end of the alignment\n");
        tmap_bug ();
        // return 0;
    }
#else
    assert (op_end >= trim_at);
#endif

    if (op_end == trim_at)
    {
        // if trimmed at the edge of cigar operator, 
        // iteratively remove any non-match operators starting from the head one, incrementing clip_extension by the consumed read length
        // if reached beginning of cigar, return 0 (reject entire read)
        while (op_idx + 1 < orig_cigar_sz && bam_cigar_op (cigar [op_idx + 1]) != BAM_CMATCH)
        {
            ++ op_idx;
            constype = bam_cigar_type (cigar [op_idx]);
            oplen = bam_cigar_oplen (cigar [op_idx]);
            if (constype & CONSUME_QRY) // consumes query
            {
                unsigned ol = oplen;
                trim_at += ol;
                op_end  += ol;
            }
            if (constype & CONSUME_REF) // consumes reference
                *ref_rep_off_p += oplen;
        }
    }

    // now we are guaranteed to be inside or at the right edge of the non-clipping cigar operator (at the index op_idx)

    // finish computing new cigar length
    if (!lsoft_present) // need space for soft clip operator if one not present
        ++new_cigar_sz;
    if (op_end > trim_at)  // need space for the right part of partially split operator
        ++new_cigar_sz;
    else // check if there are non-clipping operations left
    {
        if ((op_idx + 1 == orig_cigar_sz) ||  // there are no operations left
            ((op1 = bam_cigar_op (cigar [op_idx + 1])) == BAM_CSOFT_CLIP || op1 == BAM_CHARD_CLIP || op1 == BAM_CPAD))
            return 0;
        // fprintf (stderr, "op_idx = %d, orig_cigar_size = %d, op_end = %d, trim_at = %d\n", op_idx, orig_cigar_sz, op_end, trim_at);

        // tmap_bug ();
    }

    assert (orig_cigar_sz > op_idx); // just in case - check that we did not go beyond the cigar size
    new_cigar_sz += orig_cigar_sz - op_idx - 1; // add remining cigar operators count

    // now as we know the new cigar size, create it and fill
    // assume there is enough stack space
    uint32_t* new_cigar = (uint32_t*) alloca (new_cigar_sz * sizeof (uint32_t));
    unsigned new_cigar_idx = 0;
    // add hardclip if there
    if (lhard_present)
        new_cigar [new_cigar_idx] = cigar [new_cigar_idx], ++new_cigar_idx;
    // soft clip anything before shift
    new_cigar [new_cigar_idx] = BAM_CSOFT_CLIP | (trim_at << BAM_CIGAR_SHIFT), ++new_cigar_idx;
    // if there is split operation, add the reminder of it
    if (op_end > trim_at)
        new_cigar [new_cigar_idx] = op | ((op_end - trim_at) << BAM_CIGAR_SHIFT), ++new_cigar_idx;

    // add the reminder if any
#if PARANOID_TESTS
    if (new_cigar_sz != new_cigar_idx + (orig_cigar_sz - 1 - op_idx))
    {
        fprintf (stderr, "Internal Cigar trimming error: trim_cigar_left: new cigar size seem to be computed improperly\n");
        tmap_bug ();
    }
#else
    assert (new_cigar_sz == new_cigar_idx + (orig_cigar_sz - 1 - op_idx));
#endif
    if (op_idx + 1 != orig_cigar_sz)
        memcpy (new_cigar + new_cigar_idx, cigar+op_idx+1, (orig_cigar_sz - 1 - op_idx) * sizeof (uint32_t));

    // now check if there is a need to expand the space
    if (new_cigar_sz > orig_cigar_sz)
        cigar = *cigar_buff = (uint32_t*) tmap_realloc (cigar, new_cigar_sz * sizeof (uint32_t), "cigar_buff");
    // copy newly formed cigar into cigar space
    memcpy (cigar, new_cigar, new_cigar_sz * sizeof (uint32_t));

    return new_cigar_sz;
}
unsigned trim_cigar (uint32_t** cigar_buff, unsigned orig_cigar_sz, unsigned qry_len, unsigned trim_at, uint8_t forward, unsigned* ref_rep_off_p)
{
    // trim_at is in query forward coordinates. Cigar is in reference forward coordinates.
    if (forward)
        return trim_cigar_right (cigar_buff, orig_cigar_sz, qry_len, trim_at);
    else
        return trim_cigar_left (cigar_buff, orig_cigar_sz, qry_len - trim_at, ref_rep_off_p);
}

static uint8_t fully_clipped (uint32_t* cigar, int32_t n_cigar)
{
    // check if there are any matches in cigar
    // (less conservative would be a check for any reference consuming op, but since mapping that consumes reference with no match is senseless, we'll consider it no-map as well.)
    int32_t idx;
    for (idx = 0; idx != n_cigar; ++idx)
        if (TMAP_SW_CIGAR_OP (cigar [idx]) == BAM_CMATCH)
            return 0;
    return 1;
}

void realign_read (tmap_seq_t* qryseq, tmap_refseq_t* refseq, tmap_map_sams_t* sams, struct RealignProxy* realigner, tmap_map_stats_t *stat, uint32_t log_text_als)
{
        // extract query
    const char* qryname = tmap_seq_get_name (qryseq)->s;
    tmap_string_t* qrybases = tmap_seq_get_bases (qryseq);
    const char* qry = qrybases->s;
    unsigned matchidx;
    for (matchidx = 0; matchidx < sams->n; ++matchidx)
    {
        // extract packed cigar and it's length
        tmap_map_sam_t* match = sams->sams + matchidx;
        unsigned orig_cigar_sz = match->n_cigar;
        uint32_t* orig_cigar = tmap_calloc (orig_cigar_sz, sizeof (uint32_t), "orig_cigar");
        memcpy (orig_cigar, match->cigar, match->n_cigar * sizeof (uint32_t));
        // extract seqid 
        unsigned ref_id = match->seqid;
        // extract offset
        unsigned ref_off = match->pos;
        // extract read direction in alignment
        uint8_t forward = (match->strand == 0) ? 1 : 0;
        // make inverse/complement copy of the query if direction is reverse
        if (!forward)
        {
            char* qry_rev = qry_mem (realigner, qrybases->l+1);
            memcpy (qry_rev, qry, qrybases->l);
            qry_rev [qrybases->l] = 0;
            tmap_reverse_compliment (qry_rev, (int32_t) qrybases->l);
            qry = qry_rev;
        }
        // compute alignment and subject lengths
        unsigned q_len_cigar, r_len_cigar;
        seq_lens_from_bin_cigar (orig_cigar, orig_cigar_sz, &q_len_cigar, &r_len_cigar);
        uint8_t* ref = (uint8_t*) ref_mem (realigner, r_len_cigar);
        int32_t converted_cnt;
        // extract reference. This returns reference sequence in UNPACKED but BINARY CONVERTED form - values of 0-4, one byte per base!
        tmap_refseq_subseq2 (refseq, ref_id+1, ref_off+1, ref_off + r_len_cigar, ref, 0, &converted_cnt);
        // convert reference to ascii format
        {
            int ii;
            for (ii = 0; ii < r_len_cigar; ++ii) 
            {
                if (ref [ii] >= sizeof (tmap_iupac_int_to_char)) 
                    tmap_bug ();
                ref [ii] = tmap_iupac_int_to_char [ref [ii]];
            }
            ref [r_len_cigar] = 0;
        }
        // get name for reporting
        const char* ref_name = refseq->annos [ref_id].name->s;
        // prepare variables to hold results
        uint32_t* cigar_dest;
        unsigned cigar_dest_sz;
        unsigned new_ref_offset, new_qry_offset, new_ref_len, new_qry_len;
        ++(stat->num_realign_invocations);
        uint64_t apo = stat->num_realign_already_perfect;
        uint64_t nco = stat->num_realign_not_clipped;
        uint64_t swfo = stat->num_realign_sw_failures;
        uint64_t ucfo = stat->num_realign_unclip_failures;
        const char* al_proc_class = "UNKNOWN";
        // compute the alignment
        uint8_t al_mod = 0;
        if (realigner_compute_alignment (realigner, 
            qry, 
            qrybases->l,
            (const char*) ref, 
            r_len_cigar,
            ref_off, 
            forward, 
            orig_cigar, 
            orig_cigar_sz, 
            &cigar_dest, 
            &cigar_dest_sz, 
            &new_ref_offset, 
            &new_qry_offset, 
            &new_ref_len,
            &new_qry_len,
            &(stat->num_realign_already_perfect),
            &(stat->num_realign_not_clipped),
            &(stat->num_realign_sw_failures), 
            &(stat->num_realign_unclip_failures)))
        {
            // fix alignment end softclip - the realignment may skip some of it
            int32_t sc_adj = qrybases->l - new_qry_offset - new_qry_len;
            if (sc_adj) do
            {
                if ((cigar_dest_sz != 0) && (TMAP_SW_CIGAR_OP (cigar_dest [cigar_dest_sz - 1]) == BAM_CSOFT_CLIP))
                {
                    sc_adj += TMAP_SW_CIGAR_LENGTH (cigar_dest [cigar_dest_sz - 1]);
                    TMAP_SW_CIGAR_STORE (cigar_dest [cigar_dest_sz - 1], BAM_CSOFT_CLIP, sc_adj);
                    break;
                }
                ++cigar_dest_sz;
                cigar_dest = tmap_realloc (cigar_dest, cigar_dest_sz * sizeof (*(match->cigar)), "cigar_dest");
                TMAP_SW_CIGAR_STORE (cigar_dest [cigar_dest_sz - 1], BAM_CSOFT_CLIP, sc_adj);
            }
            while (0);
            // check if changes were introduced
            if (new_ref_offset != ref_off || orig_cigar_sz != cigar_dest_sz || memcmp (orig_cigar, cigar_dest, orig_cigar_sz * sizeof (*orig_cigar)))
            {
                ++(stat->num_realign_changed);
                if (new_ref_offset != ref_off)
                {
                    // log shifted alignment
                    ++(stat->num_realign_shifted);
                    if (nco == stat->num_realign_not_clipped) 
                        al_proc_class = "MOD-SHIFT";
                    else
                        al_proc_class = "MOD-SHIFT NOCLIP";
                }
                else
                {
                    // log changed alignment
                    if (nco == stat->num_realign_not_clipped) 
                        al_proc_class = "MOD";
                    else
                        al_proc_class = "MOD NOCLIP";
                }
                al_mod = 1;
                // pack the alignment back into the originating structure (unallocate memory taken by the old one)
                // TODO implement more intelligent alignment memory management, avoid heap operations when possible
                free (match->cigar);
                match->cigar = cigar_dest;
                match->n_cigar = cigar_dest_sz;
                match->pos = new_ref_offset;
                // we need to update match->result since edge indel salvage relies on it
                match->result.query_start = new_qry_offset;
                match->result.query_end = new_qry_offset + new_qry_len - 1;
                match->result.target_start = 0; // adjusted!new_ref_offset - ref_off;
                match->result.target_end = new_ref_len - 1;
                match->target_len = new_ref_len;
            }
            else
            {
                // log unchanged alignment
                free (cigar_dest);
                if (nco == stat->num_realign_not_clipped) 
                    al_proc_class = "UNMOD";
                else
                    al_proc_class = "UNMOD NOCLIP";
            }
        }
        else
        {
            // log reason why alignment was not processed
            if (swfo != stat->num_realign_sw_failures)
                al_proc_class = "SWERR";
            else if (ucfo != stat->num_realign_unclip_failures)
                al_proc_class = "UNCLIPERR";
            else if (apo != stat->num_realign_already_perfect)
                al_proc_class = "PERFECT";
        }
        // log 
        tmap_log_record_begin ();
        tmap_log ("REALIGN: %s(%s) vs %s:%d %s. ", qryname, (forward ? "FWD":"REV"), ref_name, ref_off, al_proc_class);
        if (al_mod)
        {
            tmap_log ("orig (%d op) at %d: ", orig_cigar_sz, ref_off);
            cigar_log (orig_cigar, orig_cigar_sz);
            tmap_log ("; new (%d op) at %d: ", cigar_dest_sz, new_ref_offset);
            cigar_log (cigar_dest, cigar_dest_sz);

            if (log_text_als)
            {
                tmap_map_log_text_align ("  Before realignment:\n", orig_cigar, orig_cigar_sz, qry, qrybases->l, forward, (const char*) ref, ref_off);
                tmap_map_log_text_align ("  After realignment:\n", match->cigar, match->n_cigar, qry, qrybases->l, forward, (const char*) (ref + (new_ref_offset - ref_off)), new_ref_offset);
            }
        }
        tmap_log ("\n");
        tmap_log_record_end ();
        free (orig_cigar);
    }
}

void context_align_read (tmap_seq_t* qryseq, tmap_refseq_t* refseq, tmap_map_sams_t* sams, struct RealignProxy* context, tmap_map_stats_t *stat, int32_t log_text_als)
{
    // context_align_read (qryseq, refseq, sams)
    // extract query
    const char* qryname = tmap_seq_get_name (qryseq)->s;

    tmap_string_t* qrybases = tmap_seq_get_bases (qryseq);
    const char* qry = qrybases->s;
    uint32_t qry_len = qrybases->l;
    unsigned matchidx;
    for (matchidx = 0; matchidx < sams->n; ++matchidx)
    {
        // extract packed cigar and it's length
        tmap_map_sam_t* match = sams->sams + matchidx;
        uint32_t* orig_cigar = match->cigar;
        unsigned orig_cigar_sz = match->n_cigar;
        // extract seqid 
        unsigned ref_id = match->seqid;
        // extract offset
        unsigned ref_off = match->pos;
        // extract read direction in alignment
        uint8_t forward = (match->strand == 0) ? 1 : 0;
        // make inverse/complement copy of the query if direction is reverse
        if (!forward)
        {
            char* qry_rev = qry_mem (context, qrybases->l+1);
            memcpy (qry_rev, qry, qrybases->l);
            qry_rev [qrybases->l] = 0;
            tmap_reverse_compliment (qry_rev, (int32_t) qrybases->l);
            qry = qry_rev;
        }
        // compute alignment and subject lengths
        unsigned q_len_cigar, r_len_cigar;
        seq_lens_from_bin_cigar (orig_cigar, orig_cigar_sz, &q_len_cigar, &r_len_cigar);
        uint8_t* ref = (uint8_t*) ref_mem (context, r_len_cigar+1);
        int32_t converted_cnt;
        // extract reference. This returns reference sequence in UNPACKED but BINARY CONVERTED form - values of 0-4, one byte per base!
        if (ref != tmap_refseq_subseq2 (refseq, ref_id+1, ref_off+1, ref_off + r_len_cigar, ref, 0, &converted_cnt))
            tmap_bug ();
        // convert reference to ascii format
        {
            int ii;
            for(ii = 0; ii < r_len_cigar; ++ii) 
            {
                if (ref [ii] >= sizeof (tmap_iupac_int_to_char)) 
                    tmap_bug ();
                ref [ii] = tmap_iupac_int_to_char [ref [ii]];
            }
            ref [r_len_cigar] = 0;
        }
        const char* ref_name = refseq->annos [ref_id].name->s;
        // prepare variables to hold results
        uint32_t* cigar_dest;
        unsigned cigar_dest_sz;
        unsigned new_ref_offset, new_qry_offset, new_ref_len, new_qry_len;
        ++(stat->num_hpcost_invocations);
        // compute the alignment

        if (realigner_compute_alignment (context, 
            qry, 
            qry_len,
            (const char*) ref, 
            r_len_cigar,
            ref_off, 
            forward, 
            orig_cigar, 
            orig_cigar_sz, 
            &cigar_dest, 
            &cigar_dest_sz, 
            &new_ref_offset, 
            &new_qry_offset,
            &new_ref_len,
            &new_qry_len,
            NULL,
            NULL,
            NULL, 
            NULL))
        {
            // check if changes were introduced
            if (new_ref_offset != ref_off || orig_cigar_sz != cigar_dest_sz || memcmp (orig_cigar, cigar_dest, orig_cigar_sz*sizeof (*orig_cigar)))
            {
                if (tmap_log_enabled ())
                {
                    tmap_log_record_begin ();
                    tmap_log ("CONTEXT-GAP: %s(%s) vs %s:%d MODIFIED: ", qryname, (forward ? "FWD":"REV"), ref_name, ref_off);
                    tmap_log ("orig (%d op) at %d: ", orig_cigar_sz, ref_off);
                    cigar_log (orig_cigar, orig_cigar_sz);
                    tmap_log ("; new (%d op) at %d: ", cigar_dest_sz, new_ref_offset);
                    cigar_log (cigar_dest, cigar_dest_sz);
                    tmap_log ("\n");

                    if (log_text_als)
                    {
                        tmap_map_log_text_align ("  Before context realignment:\n", orig_cigar, orig_cigar_sz, qry, qrybases->l, forward, (const char*) ref, ref_off);
                        tmap_map_log_text_align ("  After context realignment:\n", cigar_dest, cigar_dest_sz, qry, qrybases->l, forward, (const char*) (ref + (new_ref_offset - ref_off)), new_ref_offset);
                    }
                    tmap_log_record_end ();
                }

                ++(stat->num_hpcost_modified);
                if (new_ref_offset != ref_off)
                    // log shifted alignment
                    ++(stat->num_hpcost_shifted);
                // pack the alignment back into the originating structure (unallocate memory taken by the old one)
                if (orig_cigar_sz < cigar_dest_sz)
                {
                    free (match->cigar);
                    match->cigar = (uint32_t*) tmap_malloc (sizeof (uint32_t) * cigar_dest_sz, "cigar_dest");
                }
                match->n_cigar = cigar_dest_sz;
                memcpy (match->cigar, cigar_dest, sizeof (uint32_t) * cigar_dest_sz);
                sams->sams [matchidx].pos = new_ref_offset;
                // we need to update match->result since edge indel salvage relies on it
                match->result.query_start = new_qry_offset;
                match->result.query_end = new_qry_offset + new_qry_len - 1;
                match->result.target_start = 0; // adjusted!new_ref_offset - ref_off;
                match->result.target_end = new_ref_len - 1;
                match->target_len = new_ref_len;

            }
            else
                tmap_log_s ("CONTEXT-GAP: %s(%s) vs %s:%d UNCHANGED\n", qryname, (forward ? "FWD":"REV"), ref_name, ref_off);
        }
        else
        {
            ++(stat->num_hpcost_skipped);
            tmap_log_s ("CONTEXT-GAP: %s(%s) vs %s:%d SKIPPED\n", qryname, (forward ? "FWD":"REV"), ref_name, ref_off);
        }
    }

}

const unsigned MAX_PERIOD = 10;

void tail_repeats_clip_read (tmap_seq_t* qryseq, tmap_refseq_t* refseq, tmap_map_sams_t* sams, struct RealignProxy* realigner, tmap_map_stats_t *stat, int32_t repclip_continuation)
{
    // extract query
    const char* qryname = tmap_seq_get_name (qryseq)->s;
    tmap_string_t* qrybases = tmap_seq_get_bases (qryseq);
    const char* qry = qrybases->s;
    unsigned filtered = 0; // some of the alignments could be completely skipped

    unsigned matchidx;
    for (matchidx = 0; matchidx < sams->n; ++matchidx)
    {
        // if necessary, prepare slot for filling in the sam
        if (filtered != matchidx)
        { 
            // if we had rejected some of matches, the tail should shrink. (avoid reallocation, just swap all internal pointers. Otherwise it'll be too heavy, and it'll be better to skip them during BAM writing, which is logically more cumbersome)
            tmap_map_sam_t tmp = sams->sams [filtered]; 
            sams->sams [filtered] = sams->sams [matchidx];
            sams->sams [matchidx] = tmp;
        }
        // extract packed cigar and it's length
        uint32_t* orig_cigar = sams->sams [filtered].cigar;
        unsigned orig_cigar_sz = sams->sams [filtered].n_cigar;
        // extract seqid 
        unsigned ref_id = sams->sams [filtered].seqid;
        // extract offset
        unsigned ref_off = sams->sams [filtered].pos;
        // extract read direction in alignment
        uint8_t forward = (sams->sams [filtered].strand == 0) ? 1 : 0;
        // compute alignment and subject lengths
        unsigned q_al_beg, q_al_end, r_al_beg, r_al_end;
        alignment_bounds_from_bin_cigar (orig_cigar, orig_cigar_sz, forward, qrybases->l, &q_al_beg, &q_al_end, &r_al_beg, &r_al_end);
        // check if this read had hit the adapter (even one-base hit makes the repeat clipping invalid)
        if (tmap_sam_get_zb (qryseq->data.sam) > 0 && tmap_sam_get_za (qryseq->data.sam) <= q_al_end)
        {
            filtered ++;
            continue;
        }
        const char* ref_name = refseq->annos [ref_id].name->s;
        unsigned qry_rep_off;
        if (repclip_continuation)
        {
            // extract reference 'tail' (continuation). This returns reference sequence in UNPACKED but BINARY CONVERTED form - values of 0-4, one byte per base!
            int32_t ref_tail_beg = forward ? (ref_off + r_al_end) : ((MAX_PERIOD < ref_off + r_al_beg) ? (ref_off + r_al_beg - MAX_PERIOD) : 0);
            int32_t ref_tail_end = forward ? mymin (refseq->annos [ref_id].len, ref_tail_beg + MAX_PERIOD) : (ref_off + r_al_beg); //inclusive 1-based - same as exclusive 0-based
            int32_t ref_tail_len = ref_tail_end - ref_tail_beg;
            if (!ref_tail_len)
            {
                filtered ++;
                continue;
            }
            uint8_t* ref = (uint8_t*) ref_mem (realigner, ref_tail_len);
            int32_t converted_cnt;
            tmap_refseq_subseq2 (refseq, ref_id+1, ref_tail_beg+1, ref_tail_end, ref, 0, &converted_cnt);
            // convert reference to ascii format
            {
                int ii;
                for(ii = 0; ii < ref_tail_len; ++ii) 
                {
                    if (ref [ii] >= sizeof (tmap_iupac_int_to_char)) 
                    {
                        fprintf (stderr, "Invalid base: %d at position %d in reference %s:%d-%d\n", ref [ii], ii, ref_name, ref_tail_beg+1, ref_tail_end);
                        tmap_bug ();
                    }
                    ref [ii] = tmap_iupac_int_to_char [ref [ii]];
                }
                ref [ref_tail_len] = 0;
            }
            // inverse/complement for reverse match
            if (!forward)
                tmap_reverse_compliment ((char*) ref, ref_tail_len);
            // check if there is a tail periodicity extending by at least one period into reference overhang; 
            // tailrep returns 0 if entire (unclipped) alignment gets trimmed
            qry_rep_off = tailrep_cont (ref_tail_len, (char*) ref, q_al_end, qry);
        }
        else
        {
            qry_rep_off = tailrep (q_al_end, qry, MAX_PERIOD);
        }
        // qry_rep_off is where ON THE READ (in read's forward direction) the clip should start.
        ++ stat->num_seen_tailclipped;
        stat->bases_seen_tailclipped += q_al_end;
        // soft-clip longest one, if any
        if (qry_rep_off != q_al_end)
        {
            // save original cigar for reporting
            uint32_t* orig_cigar = alloca (sizeof (uint32_t) * orig_cigar_sz);
            memcpy (orig_cigar, sams->sams [filtered].cigar, sizeof (uint32_t) * orig_cigar_sz);
            unsigned ref_rep_off = 0;
            unsigned trimmed_cigar_sz = trim_cigar (&(sams->sams [filtered].cigar), orig_cigar_sz, qrybases->l, qry_rep_off, forward, &ref_rep_off);
            // if (trimmed_cigar_sz)
            if (trimmed_cigar_sz && !fully_clipped (sams->sams [filtered].cigar, trimmed_cigar_sz))
            {
                sams->sams [filtered].n_cigar = trimmed_cigar_sz;
                if (!forward)
                    sams->sams [filtered].pos += ref_rep_off;

                tmap_log_record_begin ();
                tmap_log ("TAIL_REP_TRIM: %s(%s) vs %s:%d TRIMMED %d bases. ", qryname, (forward ? "FWD":"REV"), ref_name, ref_off, q_al_end - qry_rep_off);
                tmap_log ("orig (%d op) at %d: ", orig_cigar_sz, ref_off);
                cigar_log (orig_cigar, orig_cigar_sz);
                tmap_log ("; new (%d op) at %d: ", trimmed_cigar_sz, sams->sams [filtered].pos);
                cigar_log (sams->sams [filtered].cigar, trimmed_cigar_sz);
                tmap_log ("\n");
                tmap_log_record_end ();

                stat->bases_tailclipped += q_al_end - qry_rep_off; // with respect to the read
                filtered ++;
            }
            else
            {
                tmap_log_s ("TAIL_REP_TRIM: %s(%s) vs %s:%d REMOVED (FULLY TRIMMED)\n", qryname, (forward ? "FWD":"REV"), ref_name, ref_off);
                stat->bases_tailclipped += q_al_end;
                stat->bases_fully_tailclipped += q_al_end;
                ++ stat->num_fully_tailclipped;
                // do not increment filtered; skip this 
            }
            ++ stat->num_tailclipped;
        }
        else
        {
            tmap_log_s ("TAIL_REP_TRIM: %s(%s) vs %s:%d UNCHANGED\n", qryname, (forward ? "FWD":"REV"), ref_name, ref_off);
            ++ filtered;
        }
    } // end alignment processing
    sams->n = filtered;
}

void realign_reads (tmap_seqs_t *seqs_buffer, tmap_refseq_t* refseq, tmap_map_record_t* record, struct RealignProxy* realigner, tmap_map_stats_t *stat, int32_t log_text_als)
{
    unsigned seqidx;
    for (seqidx = 0; seqidx < seqs_buffer->n; ++seqidx) 
    {
        tmap_map_sams_t* sams = record->sams [seqidx];
        tmap_seq_t* qryseq = seqs_buffer->seqs [seqidx];
        realign_read (qryseq, refseq, sams, realigner, stat, log_text_als);
    }
}

void context_align_reads (tmap_seqs_t *seqs_buffer, tmap_refseq_t* refseq, tmap_map_record_t* record, struct RealignProxy* realigner, tmap_map_stats_t *stat, int32_t log_text_als)
{
    unsigned seqidx;
    for (seqidx = 0; seqidx < seqs_buffer->n; ++seqidx) 
    {
        tmap_map_sams_t* sams = record->sams [seqidx];
        tmap_seq_t* qryseq = seqs_buffer->seqs [seqidx];
        context_align_read (qryseq, refseq, sams, realigner, stat, log_text_als);
    }
}

void tail_repeat_clip_reads (tmap_seqs_t *seqs_buffer, tmap_refseq_t* refseq, tmap_map_record_t* record, struct RealignProxy* realigner, tmap_map_stats_t *stat, int32_t repclip_continuation)
{
    unsigned seqidx;
    for (seqidx = 0; seqidx < seqs_buffer->n; ++seqidx) 
    {
        tmap_map_sams_t* sams = record->sams [seqidx];
        tmap_seq_t* qryseq = seqs_buffer->seqs [seqidx];
        tail_repeats_clip_read (qryseq, refseq, sams, realigner, stat, repclip_continuation);
    }
}

void
tmap_map_driver_core_worker(sam_header_t *sam_header,
                            tmap_seqs_t **seqs_buffer, 
                            tmap_map_record_t **records, 
                            tmap_map_bams_t **bams,
                            int32_t seqs_buffer_length,
                            int32_t *buffer_idx,
                            tmap_index_t *index,
                            tmap_map_driver_t *driver,
                            tmap_map_stats_t *stat,
                            tmap_rand_t *rand,
                            // DVK - realigner
                            struct RealignProxy* realigner,
                            struct RealignProxy* context,
                            int32_t do_pairing,
                            int32_t tid)
{
    int32_t i, j, k, low = 0;
    int32_t found;
    tmap_seq_t*** seqs = NULL;
    tmap_bwt_match_hash_t* hash=NULL;
    int32_t max_num_ends = 0;

    // common memory resource for all target fragments
    // common memory resource for WS traceback paths
    tmap_sw_path_t *path_buf = NULL; // buffer for traceback path
    int32_t path_buf_sz = 0;         // used portion and allocated size of traceback path. 

    #ifdef TMAP_DRIVER_USE_HASH
    // init the occurence hash
    hash = tmap_bwt_match_hash_init (); 
    #endif

    // init memory
    max_num_ends = 2;
    seqs = tmap_malloc (sizeof (tmap_seq_t**) * max_num_ends, "seqs");
    for (i = 0; i < max_num_ends; i++)
        seqs [i] = tmap_calloc (4, sizeof (tmap_seq_t*), "seqs[i]");

    // init target sequence cache
    ref_buf_t target;
    target_cache_init (&target);

    // initialize thread data
    tmap_map_driver_do_threads_init (driver, tid);

    // Go through the buffer
    while (low < seqs_buffer_length) 
    {
        if (tid == (low % driver->opt->num_threads)) 
        {
            tmap_map_stats_t *stage_stat = NULL;
            tmap_map_record_t *record_prev = NULL;
            int32_t num_ends;

            #ifdef TMAP_DRIVER_USE_HASH
            #ifdef TMAP_DRIVER_CLEAR_HASH_PER_READ
            // TODO: should we hash each read, or across the thread?
            tmap_bwt_match_hash_clear (hash);
            #endif
            #endif

            num_ends = seqs_buffer [low]->n;
            if(max_num_ends < num_ends) 
            {
                seqs = tmap_realloc (seqs, sizeof (tmap_seq_t**) * num_ends, "seqs");
                while(max_num_ends < num_ends) 
                {
                    seqs [max_num_ends] = tmap_calloc(4, sizeof(tmap_seq_t*), "seqs[max_num_ends]");
                    max_num_ends++;
                }
                max_num_ends = num_ends;
            }
            // re-initialize the random seed
            if(driver->opt->rand_read_name)
                tmap_rand_reinit(rand, tmap_hash_str_hash_func_exc(tmap_seq_get_name(seqs_buffer[low]->seqs[0])->s, driver->opt->prefix_exclude, driver->opt->suffix_exclude));

            // init
            for(i = 0; i < num_ends; i++) 
            {
                tmap_map_driver_init_seqs (seqs [i], seqs_buffer [low]->seqs [i], -1);
                if (NULL != stat) stat->num_reads++;
            }

            // init records
            records [low] = tmap_map_record_init (num_ends);

            // go through each stage
            for (i = 0; i < driver->num_stages; i++) 
            { // for each stage

                tmap_map_driver_stage_t *stage = driver->stages [i];
                tmap_sw_param_t sw_par;
                // stage may have special sw parameters
                tmap_map_util_populate_sw_par (&sw_par, stage->opt);

                // stage stats
                stage_stat = tmap_map_stats_init ();

                // seed
                for (j = 0; j < num_ends; j++) 
                {   // for each end
                    tmap_seq_t** stage_seqs = NULL;
                    // should we seed using the whole read?
                    if(0 < stage->opt->stage_seed_max_length && stage->opt->stage_seed_max_length < tmap_seq_get_bases_length (seqs [j][0])) 
                    {
                        stage_seqs = tmap_calloc (4, sizeof (tmap_seq_t*), "seqs[i]");
                        tmap_map_driver_init_seqs (stage_seqs, seqs_buffer [low]->seqs [i], stage->opt->stage_seed_max_length);
                    }
                    else 
                        stage_seqs = seqs [j];
                    for (k = 0; k < stage->num_algorithms; k++) 
                    { // for each algorithm
                        tmap_map_driver_algorithm_t *algorithm = stage->algorithms [k];
                        tmap_map_sams_t *sams = NULL;
                        if (i + 1 != algorithm->opt->algo_stage) 
                            tmap_bug();
                        // map
                        sams = algorithm->func_thread_map (&algorithm->thread_data [tid], stage_seqs, index, hash, rand, algorithm->opt);
                        if (NULL == sams) 
                            tmap_error ("the thread function did not return a mapping", Exit, OutOfRange);
                        // append
                        tmap_map_sams_merge (records [low]->sams [j], sams);
                        // destroy
                        tmap_map_sams_destroy (sams);
                    }
                    stage_stat->num_after_seeding += records [low]->sams [j]->n;
                    if (0 < stage->opt->stage_seed_max_length && stage->opt->stage_seed_max_length < tmap_seq_get_bases_length (seqs [j][0])) 
                    {
                        // free
                        for (j = 0; j < 4; j++) 
                        {
                          tmap_seq_destroy (stage_seqs [j]);
                          stage_seqs [j] = NULL;
                        }
                    }
                    stage_seqs = NULL; // do not use
                }

                // keep mappings for subsequent stages or restore mappings from
                // previous stages
                if (1 == stage->opt->stage_keep_all) 
                {
                    // merge from the previous stage
                    if (0 < i) 
                    {
                        tmap_map_record_merge (records [low], record_prev);
                        // destroy the record
                        tmap_map_record_destroy (record_prev);
                        record_prev = NULL;
                    }

                    // keep for the next stage
                    if (i < driver->num_stages - 1) // more stages left
                        record_prev = tmap_map_record_clone (records [low]);
                }

                // generate scores with smith waterman
                for (j = 0; j < num_ends; j++) 
                {   // for each end
                    records [low]->sams [j] = tmap_map_util_sw_gen_score (index->refseq, seqs_buffer [low]->seqs [j], records [low]->sams [j], seqs [j], rand, stage->opt, &k);
                    stage_stat->num_after_scoring += records [low]->sams [j]->n;
                    stage_stat->num_after_grouping += k;
                }

                // remove duplicates
                for (j = 0; j < num_ends; j++) 
                {   // for each end
                    tmap_map_util_remove_duplicates (records [low]->sams [j], stage->opt->dup_window, rand);
                    stage_stat->num_after_rmdup += records [low]->sams [j]->n;
                }

                // (single-end) mapping quality
                for (j = 0;j < num_ends; j++) // for each end
                    driver->func_mapq (records [low]->sams [j], tmap_seq_get_bases_length (seqs [j][0]), stage->opt, index->refseq);


                // filter if we have more stages
                if (i < driver->num_stages-1) 
                {
                    for (j = 0; j < num_ends; j++) // for each end
                        tmap_map_sams_filter2 (records [low]->sams [j], stage->opt->stage_score_thr, stage->opt->stage_mapq_thr);
                }

                if (0 == do_pairing && 0 <= driver->opt->strandedness && 0 <= driver->opt->positioning
                 && 2 == num_ends && 0 < records [low]->sams [0]->n && 0 < records [low]->sams [1]->n) 
                {   // pairs of reads!
                    // read rescue
                    if (1 == stage->opt->read_rescue) 
                    {
                        int32_t flag = tmap_map_pairing_read_rescue (index->refseq, 
                                                                  seqs_buffer [low]->seqs[0], seqs_buffer [low]->seqs [1],
                                                                  records [low]->sams [0], records [low]->sams [1],
                                                                  seqs [0], seqs [1],
                                                                  rand, stage->opt);
                        // recalculate mapping qualities if necessary
                        if (0 < (flag & 0x1))  // first end was rescued
                            driver->func_mapq (records [low]->sams [0], tmap_seq_get_bases_length (seqs [0][0]), stage->opt, index->refseq);
                        if(0 < (flag & 0x2)) // second end was rescued
                            driver->func_mapq (records [low]->sams [1], tmap_seq_get_bases_length (seqs [1][0]), stage->opt, index->refseq);
                    }
                    // pick pairs
                    tmap_map_pairing_pick_pairs (records [low]->sams [0], records [low]->sams [1],
                                              seqs [0][0], seqs [1][0],
                                              rand, stage->opt);
                }
                else 
                {
                    // choose alignments
                    for (j = 0; j < num_ends; ++j) 
                    { // for each end
                        tmap_map_sams_filter1 (records [low]->sams [j], stage->opt->aln_output_mode, TMAP_MAP_ALGO_NONE, rand);
                        stage_stat->num_after_filter += records [low]->sams [j]->n;
                    }
                }

                // generate and post-process alignments
                found = 0;
                for (j = 0; j < num_ends; ++j)  // for each end
                { 
                    tmap_map_sams_t* sams = records [low]->sams [j];
                    tmap_seq_t* seq = seqs_buffer [low]->seqs [j];

                    //if (0 == strcmp (seq->data.sam->name->s, "K0BK3:04849:06835"))
                    //    tmap_progress_print2 ("here");
                    tmap_seq_t** seq_variants = seqs [j];
                    // if there are no mappings, continue
                    if (!sams->n)
                        continue;
                    // find alignment starqts
                    sams = records [low]->sams [j] = tmap_map_util_find_align_starts 
                    (
                        index->refseq,      // reference server
                        sams,               // initial rough mapping 
                        seq,                // read
                        seq_variants,       // array of size 4 that contains pre-computed inverse / complement combinations
                        stage->opt,         // stage parameters
                        &target,            // target cache control structure
                        stat
                    );
                    if (!sams->n) 
                        continue;
                    // reference alignment
                    tmap_map_util_align 
                    (
                        index->refseq,      // reference server
                        sams,               // mappings to compute alignments for
                        seq_variants,       // array of size 4 that contains pre-computed inverse / complement combinations
                        &target,            // target cache control structure
                        &path_buf,          // buffer for traceback path
                        &path_buf_sz,       // used portion and allocated size of traceback path. 
                        &sw_par,            // Smith-Waterman scoring parameters
                        stat
                    );
                    // realign in flowspace or with context, if requested
                    if (driver->opt->aln_flowspace)
                        // NB: seqs_buffer should have its key sequence if 0 < key_seq_len
                        tmap_map_util_fsw 
                        (
                            seq, 
                            sams, 
                            index->refseq, 
                            driver->opt->bw, 
                            driver->opt->softclip_type, 
                            driver->opt->score_thr,
                            driver->opt->score_match, 
                            driver->opt->pen_mm, 
                            driver->opt->pen_gapo,
                            driver->opt->pen_gape, 
                            driver->opt->fscore, 
                            1 - driver->opt->ignore_flowgram,
                            stat
                        );
                    else if (driver->opt->do_hp_weight)
                        context_align_read 
                        (
                            seq, 
                            index->refseq, 
                            sams, 
                            context, 
                            stat, 
                            driver->opt->log_text_als
                        );
                    // perform anti-dyslexic realignment if enabled
                    if (driver->opt->do_realign)
                    {
                        realign_read 
                        (
                            seq, 
                            index->refseq, 
                            sams, 
                            realigner, 
                            stat, 
                            driver->opt->log_text_als
                        );
                    }
                    // salvage ends
                    if (0 < driver->opt->pen_gapl)
                        tmap_map_util_salvage_edge_indels 
                        (
                            index->refseq,      // reference server
                            sams,               // mappings to compute alignments for
                            seq_variants,       // array of size 4 that contains pre-computed inverse / complement combinations
                            stage->opt,         // tmap parameters
                            &sw_par,            // Smith-Waterman scoring parameters
                            &target,            // target cache control structure
                            &path_buf,          // buffer for traceback path
                            &path_buf_sz,        // used portion and allocated size of traceback path. 
                            stat
                        );
                    // trim key
                    if (1 == stage->opt->softclip_key)
                        tmap_map_util_trim_key 
                        (
                            sams,               // mappings to compute alignments for
                            seq,                // read
                            seq_variants,       // array of size 4 that contains pre-computed inverse / complement combinations
                            index->refseq,      // reference server
                            &target,            // target cache control structure
                            stat
                        );
                    // end repair
                    if (stage->opt->end_repair)
                        tmap_map_util_end_repair_bulk 
                        (
                            index->refseq,      // reference server
                            sams,               // mappings to compute alignments for
                            seq,                // read
                            seq_variants,       // array of size 4 that contains pre-computed inverse / complement combinations
                            stage->opt,         // tmap parameters
                            &target,            // target cache control structure
                            &path_buf,          // buffer for traceback path
                            &path_buf_sz,       // used portion and allocated size of traceback path. 
                            stat
                        );
                    // explicitly fix 5' softclip is required
                    if (!stage->opt->end_repair_5_prime_softclip && (stage->opt->softclip_type == 2 || stage->opt->softclip_type == 3))
                    {
                        int i;
                        for (i = 0; i != sams->n; ++i)
                        {
                            tmap_map_util_remove_5_prime_softclip 
                            (
                                index->refseq,
                                sams->sams + i,
                                seq,
                                seq_variants,
                                &target,
                                &path_buf,
                                &path_buf_sz,
                                &sw_par,
                                stage->opt,
                                stat
                            );
                        }
                    }
                    if (stage->opt->cigar_sanity_check)  // do this before tail repeat clipping as the latter does not update alignment box. TODO: update box in tail clip and move this to the very end of alignment post processing
                    {
                        int i;
                        for (i = 0; i != sams->n; ++i)
                        {
                            cigar_sanity_check 
                            (
                                index->refseq,
                                sams->sams + i,
                                seq,
                                seq_variants,
                                &target,
                                stage->opt
                            );
                        }
                    }
                    // clip repeats
                    if (driver->opt->do_repeat_clip)
                        tail_repeats_clip_read 
                        (
                            seq, 
                            index->refseq, 
                            sams, 
                            realigner, 
                            stat, 
                            driver->opt->repclip_continuation
                        );

                    stage_stat->num_with_mapping++;
                    found = 1;
                }

                // TODO
                // if paired, update pairing score based on target start?

                // did we find any mappings?
                if (found) 
                {   // yes
                    if (stat) 
                        tmap_map_stats_add (stat, stage_stat);
                    tmap_map_stats_destroy (stage_stat);
                    break;
                }
                else 
                {   // no
                    tmap_map_record_destroy (records [low]);
                    // re-init
                    records [low] = tmap_map_record_init (num_ends);
                }
                tmap_map_stats_destroy (stage_stat);
            }

            // only convert to BAM and destroy the records if we are not trying to
            // estimate the pairing parameters
            if (0 == do_pairing) 
            {
                // convert the record to bam
                if (1 == seqs_buffer [low]->n) 
                {
                    bams [low] = tmap_map_bams_init (1); 
                    bams [low]->bams [0] = tmap_map_sams_print (seqs_buffer [low]->seqs [0], index->refseq, records [low]->sams [0], 
                                                           0, NULL, driver->opt->sam_flowspace_tags, driver->opt->bidirectional, driver->opt->seq_eq, driver->opt->min_al_len, driver->opt->min_al_cov, driver->opt->min_identity, driver->opt->score_match, &(stat->num_filtered_als));
                }
                else 
                {
                    bams [low] = tmap_map_bams_init (seqs_buffer [low]->n);
                    for (j = 0; j < seqs_buffer [low]->n; j++) 
                        bams [low]->bams [j] = tmap_map_sams_print (seqs_buffer [low]->seqs [j], index->refseq, records [low]->sams [j],
                                                               (0 == j) ? 1 : ((seqs_buffer [low]->n-1 == j) ? 2 : 0),
                                                               records [low]->sams[(j+1) % seqs_buffer [low]->n], 
                                                               driver->opt->sam_flowspace_tags, driver->opt->bidirectional, driver->opt->seq_eq, driver->opt->min_al_len, driver->opt->min_al_cov, driver->opt->min_identity, driver->opt->score_match, &(stat->num_filtered_als));
                }
                // free alignments, for space
                tmap_map_record_destroy (records [low]); 
                records [low] = NULL;
            }
            // free seqs
            for (i = 0; i < num_ends; i++) 
            {
                for (j = 0; j < 4; j++) 
                {
                    tmap_seq_destroy (seqs [i][j]);
                    seqs [i][j] = NULL;
                }
            }
            tmap_map_record_destroy (record_prev);
        }
        // next
        (*buffer_idx) = low;
        low++;
    }
    (*buffer_idx) = seqs_buffer_length;

    // free thread variables
    for (i = 0; i < max_num_ends; i++) 
        free (seqs [i]);
    free (seqs);
    free (path_buf);
    target_cache_free (&target);

    // cleanup
    tmap_map_driver_do_threads_cleanup (driver, tid);
    #ifdef TMAP_DRIVER_USE_HASH
    // free hash
    tmap_bwt_match_hash_destroy(hash);
    #endif
}

void *
tmap_map_driver_core_thread_worker(void *arg)
{
  tmap_map_driver_thread_data_t *thread_data = (tmap_map_driver_thread_data_t*)arg;

  tmap_map_driver_core_worker(thread_data->sam_header, thread_data->seqs_buffer, thread_data->records, thread_data->bams, 
                              thread_data->seqs_buffer_length, thread_data->buffer_idx, thread_data->index, thread_data->driver, 
                              thread_data->stat, thread_data->rand, /* DVK - realigner */ thread_data->realigner, thread_data->context, thread_data->do_pairing, thread_data->tid);

  return arg;
}

//static inline void
static int32_t
tmap_map_driver_create_threads(sam_header_t *header,
                               tmap_seqs_t **seqs_buffer, 
                               tmap_map_record_t **records, 
                               tmap_map_bams_t **bams,
                               int32_t seqs_buffer_length,
                               tmap_index_t *index,
                               tmap_map_driver_t *driver,
                               tmap_map_stats_t *stat,
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
                               tmap_rand_t *rand_core,
#endif
#ifdef HAVE_LIBPTHREAD
                               pthread_attr_t **attr,
                               pthread_t **threads,
                               tmap_map_driver_thread_data_t **thread_data,
                               tmap_rand_t **rand,
                               tmap_map_stats_t **stats,
                               struct RealignProxy** realigner,
                               struct RealignProxy** context,
#endif
                               int32_t do_pairing)
{
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  int32_t j;
#endif
  int32_t buffer_idx; // buffer index for processing data with a single thread

#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  // sample reads
  {
      int32_t i;
      if(driver->opt->sample_reads < 1) {
          for(i=j=0;i<seqs_buffer_length;i++) {
              if(driver->opt->sample_reads < tmap_rand_get(rand_core)) continue; // skip
              if(j < i) {
                  tmap_seqs_t *seqs;
                  seqs = seqs_buffer[j];
                  seqs_buffer[j] = seqs_buffer[i]; 
                  seqs_buffer[i] = seqs;
              }
              j++;
          }
          tmap_progress_print2("sampling %d out of %d [%.2lf%%]", j, seqs_buffer_length, 100.0*j/(double)seqs_buffer_length);
          seqs_buffer_length = j;
          if(0 == seqs_buffer_length) return 0;
      }
  }
#endif

  // do alignment
#ifdef HAVE_LIBPTHREAD
  if(1 == driver->opt->num_threads) {
      buffer_idx = 0;
      tmap_map_driver_core_worker(header, seqs_buffer, records, bams, 
                                  seqs_buffer_length, &buffer_idx, index, driver, stat, rand[0], realigner [0], context [0], do_pairing, 0);
  }
  else {
      int32_t i;
      (*attr) = tmap_calloc(1, sizeof(pthread_attr_t), "(*attr)");
      pthread_attr_init((*attr));
      pthread_attr_setdetachstate((*attr), PTHREAD_CREATE_JOINABLE);

      (*threads) = tmap_calloc(driver->opt->num_threads, sizeof(pthread_t), "(*threads)");
      (*thread_data) = tmap_calloc(driver->opt->num_threads, sizeof(tmap_map_driver_thread_data_t), "(*thread_data)");

      // create threads
      for(i=0;i<driver->opt->num_threads;i++) {
          (*thread_data)[i].sam_header = header;
          (*thread_data)[i].seqs_buffer = seqs_buffer;
          (*thread_data)[i].seqs_buffer_length = seqs_buffer_length;
          (*thread_data)[i].buffer_idx = tmap_calloc(1, sizeof(int32_t), "(*thread_data)[i].buffer_id");
          (*thread_data)[i].records = records;
          (*thread_data)[i].bams = bams;
          (*thread_data)[i].index = index;
          (*thread_data)[i].driver = driver;
          if(NULL != stats) (*thread_data)[i].stat = stats[i];
          else (*thread_data)[i].stat = NULL;
          (*thread_data)[i].rand = rand[i];
          // DVK - realigner
          (*thread_data)[i].realigner = realigner [i];
          (*thread_data)[i].context = context [i];
          (*thread_data)[i].do_pairing = do_pairing;
          (*thread_data)[i].tid = i;
          if(0 != pthread_create(&(*threads)[i], (*attr), tmap_map_driver_core_thread_worker, &(*thread_data)[i])) {
              tmap_error("error creating threads", Exit, ThreadError);
          }
      }
  }
#else 
  buffer_idx = 0;
  tmap_map_driver_core_worker(header, seqs_buffer, records, bams, 
                              seqs_buffer_length, &buffer_idx, index, driver, stat, rand, do_pairing, 0);
#endif
  return 1;
}

static int32_t
tmap_map_driver_infer_pairing(tmap_seqs_io_t *io_in,
                              sam_header_t *header,
                              tmap_seqs_t **seqs_buffer, 
                              tmap_map_record_t **records, 
                              tmap_map_bams_t **bams,
                              int32_t seqs_buffer_length,
                              int32_t reads_queue_size,
                              tmap_index_t *index,
                              tmap_map_driver_t *driver,
                              tmap_map_stats_t *stat,
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
                              tmap_rand_t *rand_core,
#endif
#ifdef HAVE_LIBPTHREAD
                              pthread_attr_t **attr,
                              pthread_t **threads,
                              tmap_map_driver_thread_data_t **thread_data,
                              tmap_rand_t **rand,
                              tmap_map_stats_t **stats,
                              // DVK - realigner
                              struct RealignProxy** realigner,
                              struct RealignProxy** context
#endif
                              ) // NB: just so that the function definition is clean
{
  int32_t i, isize_num = 0, tmp;
  int32_t *isize = NULL;
  int32_t p25, p50, p75;
  int32_t max_len = 0;

  // check if we should do pairing
  if(driver->opt->strandedness < 0 || driver->opt->positioning < 0 || !(driver->opt->ins_size_std < 0)) return 0;

  // NB: infers from the first chunk of reads
  tmap_progress_print("inferring pairing parameters");
  tmap_progress_print("loading reads");
  seqs_buffer_length = tmap_seqs_io_read_buffer(io_in, seqs_buffer, reads_queue_size, header);
  tmap_progress_print2("loaded %d reads", seqs_buffer_length);
  if(0 == seqs_buffer_length) return 0;

  // holds the insert sizes
  isize = tmap_malloc(sizeof(int32_t) * seqs_buffer_length, "isize");

  // TODO: check that he data is paired...
  // TODO: check that we choose only the best scoring alignment
  // create the threads
  if(0 == tmap_map_driver_create_threads(header, seqs_buffer, records, 
                                         bams, seqs_buffer_length, index, driver, NULL,
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
                                         rand_core,
#endif
#ifdef HAVE_LIBPTHREAD
                                         attr, threads, thread_data, rand, NULL, realigner, context,
#endif
                                         1)) {
      return 0;
  }

  // estimate pairing parameters
  for(i=0;i<seqs_buffer_length;i++) {
#ifdef HAVE_LIBPTHREAD
      if(1 < driver->opt->num_threads) {
          // NB: we will write data as threads process the data.  This is to
          // facilitate SAM/BAM writing, which may be slow, especially for
          // BAM.
          int32_t tid = (i % driver->opt->num_threads);
          while((*(*thread_data)[tid].buffer_idx) <= i) {
              usleep(1000*1000); // sleep
          }
      }
#endif
      // only for paired ends
      if(NULL != records[i]
         && 2 == records[i]->n 
         && 1 == records[i]->sams[0]->n 
         && 1 == records[i]->sams[1]->n
         && driver->opt->ins_size_min_mapq <= records[i]->sams[0]->sams[0].mapq

         && driver->opt->ins_size_min_mapq <= records[i]->sams[1]->sams[0].mapq) {
          tmap_map_sam_t *one = NULL;
          tmap_map_sam_t *two = NULL;
          int32_t strand_diff, position_diff;
          one = &records[i]->sams[0]->sams[0]; 
          two = &records[i]->sams[1]->sams[0]; 
          // get strand difference
          strand_diff = tmap_map_pairing_get_strand_diff(one, two, driver->opt->strandedness);
          if(1 == strand_diff && one->seqid == two->seqid) { // strand OK, same contig
              // get the position difference
              position_diff = tmap_map_pairing_get_position_diff(one, two, 
                                                                 tmap_seq_get_bases_length(seqs_buffer[i]->seqs[0]),
                                                                 tmap_seq_get_bases_length(seqs_buffer[i]->seqs[1]),
                                                                 driver->opt->strandedness, driver->opt->positioning);
              /*
              fprintf(stderr, "%s: %d %d\n",
                      tmap_seq_get_name(seqs_buffer[i]->seqs[0])->s,
                      strand_diff,
                      position_diff);
              */
              isize[isize_num++] = position_diff;
              if(max_len < tmap_seq_get_bases_length(seqs_buffer[i]->seqs[0])) max_len = tmap_seq_get_bases_length(seqs_buffer[i]->seqs[0]);
              if(max_len < tmap_seq_get_bases_length(seqs_buffer[i]->seqs[1])) max_len = tmap_seq_get_bases_length(seqs_buffer[i]->seqs[1]);
          }
      }
      // destroy the record
      tmap_map_record_destroy(records[i]); 
      records[i] = NULL;
  }

#ifdef HAVE_LIBPTHREAD
  // join threads
  if(1 < driver->opt->num_threads) {
      // join threads
      for(i=0;i<driver->opt->num_threads;i++) {
          if(0 != pthread_join((*threads)[i], NULL)) {
              tmap_error("error joining threads", Exit, ThreadError);
          }
          // free the buffer index
          free((*thread_data)[i].buffer_idx);
          (*thread_data)[i].buffer_idx = NULL;
      }
      free((*threads)); (*threads) = NULL;
      free(*(thread_data)); (*thread_data) = NULL;
      free((*attr)); (*attr) = NULL;
  }
#endif

  if(isize_num < 8) {
      tmap_error("failed to infer the insert size distribution (too few reads): turning pairing off", Warn, OutOfRange);
      driver->opt->pairing = -1;
      return seqs_buffer_length;
  }

  // print the # of pairs we are using to infer the insert size distribution
  tmap_progress_print("inferring the insert size distribution from %d high-quality pairs", isize_num);

  // sort
  tmap_sort_introsort(tmap_map_driver_sort_isize, isize_num, isize);

  // get 25/50/75 percentile
  p25 = isize[(int32_t)(.25 * isize_num + .499)];
  p50 = isize[(int32_t)(.50 * isize_num + .499)];
  p75 = isize[(int32_t)(.75 * isize_num + .499)];

  int32_t low, high;
  double avg, std;
  int32_t n;

  // get the lower boundary for computing hte mean and standard deviation
  tmp = (int32_t)(p25 - driver->opt->ins_size_outlier_bound * (p75 - p25) + .499);
  low = tmp > max_len ? tmp : max_len; // ensure at least the size of the read (TODO: overlapping?)
  if(low < 1) low = 1;
  // get the upper boundary for computing hte mean and standard deviation
  high = (int32_t)(p75 + driver->opt->ins_size_outlier_bound * (p75 - p25) + .499);
  tmap_progress_print("(25, 50, 75) percentile: (%d, %d, %d)", p25, p50, p75);
  tmap_progress_print("low and high boundaries for computing mean and standard deviation: (%d, %d)", low, high);
  // mean
  for(i=n=0, avg=0.0;i<isize_num;i++) {
      if(low <= isize[i] && isize[i] <= high) {
          avg += isize[i];
          n++;
      }
  }
  avg /= n;
  // std. dev
  for(i=0, std=0.0;i<isize_num;i++) {
      if(low <= isize[i] && isize[i] <= high) {
          std += (isize[i] - avg) * (isize[i] - avg);
      }
  }
  std = sqrt(std / n);
  tmap_progress_print("mean and std.dev: (%.2f, %.2f)", avg, std);
  // update
  driver->opt->ins_size_mean = avg;
  driver->opt->ins_size_std = std;

  // Calculate low/high boundaries for proper pairs
  /*
  tmp = (int32_t)(p25 - (driver->opt->ins_size_outlier_bound+1.0) * (p75 - p25) + .499);
  low = tmp > max_len ? tmp : max_len; // ensure at least the size of the read (TODO: overlapping?)
  if (low < 1) low = 1;
  high = (int32_t)(p75 + (driver->opt->ins_size_outlier_bound+1.0) * (p75 - p25) + .499);
  // bound
  if (low > avg - driver->opt->ins_size_std_max_num * (driver->opt->ins_size_outlier_bound+2.0)) low = (int32_t)(avg - driver->opt->ins_size_std_max_num * (driver->opt->ins_size_outlier_bound+2.0) + .499);
  low = tmp > max_len ? tmp : max_len; // ensure at least the size of the read (TODO: overlapping?)
  if (high < avg - driver->opt->ins_size_std_max_num * (driver->opt->ins_size_outlier_bound+2.0)) high = (int32_t)(avg + driver->opt->ins_size_std_max_num * (driver->opt->ins_size_outlier_bound+2.0) + .499);
  */
  low = avg - (std * driver->opt->ins_size_std_max_num);
  high = avg + (std * driver->opt->ins_size_std_max_num);
  tmap_progress_print("low and high boundaries for proper pairs: (%d, %d)", low, high);

  // free
  free(isize);

  // return the # of sequences loaded into the buffer
  return seqs_buffer_length;
}

#ifdef HAVE_LIBPTHREAD
typedef struct {
    tmap_seqs_io_t *io_in;
    tmap_sam_io_t *io_out;
    tmap_seqs_t **seqs_buffer;
    int32_t seqs_buffer_length;
    int32_t reads_queue_size;
} tmap_map_driver_thread_io_data_t;

static void *
tmap_map_driver_thread_io_worker (void *arg)
{
  tmap_map_driver_thread_io_data_t *d = (tmap_map_driver_thread_io_data_t*) arg;
  d->seqs_buffer_length = tmap_seqs_io_read_buffer (d->io_in, d->seqs_buffer, d->reads_queue_size, d->io_out->fp->header->header);
  return d;
}
#endif



void 
tmap_map_driver_core (tmap_map_driver_t *driver)
{
  uint32_t i, j, k, n_reads_processed = 0; // # of reads processed
  int32_t seqs_buffer_length = 0; // # of reads read in
  int32_t seqs_loaded = 0; // 1 if the seq_buffer is loaded, 0 otherwse
  tmap_seqs_io_t *io_in = NULL; // input file(s)
  tmap_sam_io_t *io_out = NULL; // output file
  tmap_seqs_t **seqs_buffer = NULL; // buffer for the reads
#ifdef HAVE_LIBPTHREAD
  tmap_seqs_t **seqs_buffer_next = NULL; // buffer for the reads
#endif
  tmap_map_record_t **records=NULL; // buffer for the mapped data
  tmap_map_bams_t **bams=NULL;// buffer for the mapped BAM data
  tmap_index_t *index = NULL; // reference indes
  tmap_map_stats_t *stat = NULL; // alignment statistics
#ifdef HAVE_LIBPTHREAD
  pthread_attr_t *attr = NULL;
  pthread_attr_t attr_io;
  pthread_t *threads = NULL;
  pthread_t *thread_io = NULL;
  tmap_map_driver_thread_data_t *thread_data=NULL;
  tmap_map_driver_thread_io_data_t thread_io_data;
  tmap_rand_t **rand = NULL; // random # generator for each thread
  tmap_map_stats_t **stats = NULL; // alignment statistics for each thread
#else
  tmap_rand_t *rand = NULL; // random # generator
#endif

  time_t start_time = time (NULL);

  if (driver->opt->report_stats)
      tmap_file_stdout = tmap_file_fdopen(fileno(stdout), "wb", TMAP_FILE_NO_COMPRESSION);

// DVK - realignment
#ifdef HAVE_LIBPTHREAD
  struct RealignProxy** realigner = NULL;
  struct RealignProxy** context = NULL;
#else
  struct RealignProxy* realigner = NULL;
  struct RealignProxy* context = NULL;
#endif


#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  tmap_rand_t *rand_core = tmap_rand_init(13); // random # generator for sampling
#endif
  int32_t seq_type, reads_queue_size; // read type, read queue size
  bam_header_t *header = NULL; // BAM Header

  /*
  if(NULL == driver->opt->fn_reads) {
      tmap_progress_set_verbosity(0); 
  }
  */

  tmap_progress_print("running with %d threads (%s)",
                       driver->opt->num_threads,
                       (0 == driver->opt->num_threads_autodetected) ? "user set" : "autodetected");
  
  // print out the algorithms and stages
  for(i=0;i<driver->num_stages;i++) {
      for(j=0;j<driver->stages[i]->num_algorithms;j++) {
          tmap_progress_print("%s will be run in stage %d", 
                               tmap_algo_id_to_name(driver->stages[i]->algorithms[j]->opt->algo_id),
                               driver->stages[i]->algorithms[j]->opt->algo_stage);
      }
  }

  // open the reads file for reading
  // NB: may have no fns (streaming in)
  seq_type = tmap_reads_format_to_seq_type(driver->opt->reads_format); 
  io_in = tmap_seqs_io_init(driver->opt->fn_reads, driver->opt->fn_reads_num, seq_type, driver->opt->input_compr,
                            driver->opt->bam_start_vfo, driver->opt->bam_end_vfo);

  // get the index
  index = tmap_index_init(driver->opt->fn_fasta, driver->opt->shm_key);

  // initialize the driver->options and print any relevant information
  tmap_map_driver_do_init(driver, index->refseq);
  if (!tmap_refseq_read_bed(index->refseq, driver->opt->bed_file)) 
    tmap_error ("Bed file read error", Exit, OutOfRange );

  // allocate the buffer
  if(-1 == driver->opt->reads_queue_size) {
      reads_queue_size = 1;
  }
  else {
      reads_queue_size = driver->opt->reads_queue_size;
  }
  seqs_buffer = tmap_malloc(sizeof(tmap_seqs_t*)*reads_queue_size, "seqs_buffer");
#ifdef HAVE_LIBPTHREAD
  seqs_buffer_next = tmap_malloc(sizeof(tmap_seqs_t*)*reads_queue_size, "seqs_buffer_next");
#endif
  for(i=0;i<reads_queue_size;i++) { // initialize the buffer
      seqs_buffer[i] = tmap_seqs_init(seq_type);
#ifdef HAVE_LIBPTHREAD
      seqs_buffer_next[i] = tmap_seqs_init(seq_type);
#endif
  }
  records = tmap_malloc(sizeof(tmap_map_record_t*)*reads_queue_size, "records");
  bams = tmap_malloc(sizeof(tmap_map_bams_t*)*reads_queue_size, "bams");

  stat = tmap_map_stats_init();
#ifdef HAVE_LIBPTHREAD
  stats = tmap_malloc(driver->opt->num_threads * sizeof(tmap_map_stats_t*), "stats");
  rand = tmap_malloc(driver->opt->num_threads * sizeof(tmap_rand_t*), "rand");
  for(i=0;i<driver->opt->num_threads;i++) {
      stats[i] = tmap_map_stats_init();
      rand[i] = tmap_rand_init(i);
  }
#else
  rand = tmap_rand_init(13);
#endif

    // DVK - thread-safe logging
    FILE* logfile = NULL;
    {
        if (driver->opt->realign_log)
        {
            logfile = fopen (driver->opt->realign_log, "w");
            if (!logfile)
                tmap_error ("logfile", Exit, OpenFileError );
            tmap_log_enable (logfile);
        }
        else
            tmap_log_disable ();
    }

    // DVK - realigner  
    // Note: this needs to be initialized only if --do-realign is specified, 
    // !!! or if --do-repeat-clip is specified, as repeat clipping uses some of the structures in realigner for data holding
    {
#ifdef HAVE_LIBPTHREAD
        realigner = tmap_malloc (driver->opt->num_threads * sizeof (struct RealignProxy*), "realigner");
        context = tmap_malloc (driver->opt->num_threads * sizeof (struct RealignProxy*), "context");
        for (i = 0; i != driver->opt->num_threads;  ++i)
        {
            realigner [i] = realigner_create ();
            realigner_set_scores (realigner [i], driver->opt->realign_mat_score, driver->opt->realign_mis_score, driver->opt->realign_gip_score, driver->opt->realign_gep_score);
            realigner_set_bandwidth (realigner [i], driver->opt->realign_bandwidth);
            realigner_set_clipping (realigner [i], (enum CLIPTYPE) driver->opt->realign_cliptype);

            context [i] = context_aligner_create ();
            realigner_set_scores (context [i], driver->opt->context_mat_score, driver->opt->context_mis_score, -driver->opt->context_gip_score, -driver->opt->context_gep_score);
            realigner_set_bandwidth (context [i], driver->opt->context_extra_bandwidth);
            realigner_set_gap_scale_mode (context [i], driver->opt->gap_scale_mode);
            realigner_set_debug (context [i], driver->opt->context_debug_log);
            // WARNING! not a thread-safve operation! Enable only in single-threaded mode!
            if (driver->opt->context_debug_log && logfile)
                realigner_set_log (context [i], fileno (logfile));
        }
    #else
        realigner = realigner_create ();
        realigner_set_scores (realigner, driver->opt->realign_mat_score, driver->opt->realign_mis_score, driver->opt->realign_gip_score, driver->opt->realign_gep_score);
        realigner_set_bandwidth (realigner, driver->opt->realign_bandwidth);
        realigner_set_clipping (realigner, (enum CLIPTYPE) driver->opt->realign_cliptype);

        context = context_aligner_create ();
        realigner_set_scores (context, driver->opt->context_mat_score, driver->opt->context_mis_score, driver->opt->context_gip_score, driver->opt->context_gep_score);
        realigner_set_bandwidth (context, driver->opt->context_extra_bandwidth);
        realigner_set_gap_scale_mode (context, driver->opt->gap_scale_mode);
        realigner_set_debug (context, driver->opt->context_debug_log);
        if (driver->opt->context_debug_log && logfile)
            realigner_set_log (context, fileno (logfile));
#endif
    }


  // BAM Header
  header = tmap_seqs_io_to_bam_header(index->refseq, io_in, 
                                      driver->opt->sam_rg, driver->opt->sam_rg_num,
                                      driver->opt->argc, driver->opt->argv);

  // open the output file
  switch(driver->opt->output_type) {
    case 0: // SAM
      io_out = tmap_sam_io_init2((NULL == driver->opt->fn_sam) ? "-" : driver->opt->fn_sam, "wh", header); 
      break;
    case 1:
      io_out = tmap_sam_io_init2((NULL == driver->opt->fn_sam) ? "-" : driver->opt->fn_sam, "wb", header); 
      break;
    case 2:
      io_out = tmap_sam_io_init2((NULL == driver->opt->fn_sam) ? "-" : driver->opt->fn_sam, "wbu", header); 
      break;
    default:
      tmap_bug();
  }

  // destroy the BAM Header
  bam_header_destroy(header);
  header = NULL;

  // pairing
  seqs_buffer_length = tmap_map_driver_infer_pairing(io_in, io_out->fp->header->header, seqs_buffer, records, 
                                                     bams, seqs_buffer_length, reads_queue_size,
                                                     index, driver, stat,
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
                                                     rand_core,
#endif
#ifdef HAVE_LIBPTHREAD
                                                     &attr, &threads, &thread_data, rand, stats, realigner, context
#endif
                                                     );
  if(0 == seqs_buffer_length) {
      tmap_progress_print("loading reads");
      seqs_buffer_length = tmap_seqs_io_read_buffer(io_in, seqs_buffer, reads_queue_size, io_out->fp->header->header);
      tmap_progress_print2("loaded %d reads", seqs_buffer_length);
  }
  seqs_loaded = 1;

  // main processing loop
  tmap_progress_print("processing reads");
  while(1) {
      // get the reads
      if(0 == seqs_loaded) { 
          tmap_progress_print("loading reads");
#ifdef HAVE_LIBPTHREAD
          // join the thread that loads in the reads
          if(0 != pthread_join((*thread_io), NULL)) {
              tmap_error("error joining IO thread", Exit, ThreadError);
          }
          free(thread_io);
          thread_io = NULL;
          // swap buffers
          seqs_buffer_length = thread_io_data.seqs_buffer_length;
          seqs_buffer_next = seqs_buffer; // temporarily store this here
          seqs_buffer = thread_io_data.seqs_buffer; 
          thread_io_data.seqs_buffer = seqs_buffer_next;
#else
          seqs_buffer_length = tmap_seqs_io_read_buffer (io_in, seqs_buffer, reads_queue_size, io_out->fp->header->header);
#endif
          seqs_loaded = 1;
          tmap_progress_print2("loaded %d reads", seqs_buffer_length);
      }
      if(0 == seqs_buffer_length) { // are there any more?
          break;
      }

#ifdef HAVE_LIBPTHREAD
      // launch a new thread that loads in the reads 
      pthread_attr_init(&attr_io);
      pthread_attr_setdetachstate(&attr_io, PTHREAD_CREATE_JOINABLE);
      thread_io = tmap_malloc(sizeof(pthread_t), "thread_io");
      thread_io_data.io_in = io_in;
      thread_io_data.io_out = io_out;
      thread_io_data.seqs_buffer_length = 0;
      thread_io_data.reads_queue_size = reads_queue_size;
      thread_io_data.seqs_buffer = seqs_buffer_next;
      if(0 != pthread_create(thread_io, &attr_io, tmap_map_driver_thread_io_worker, &thread_io_data)) {
          tmap_error("error creating threads", Exit, ThreadError);
      }
#endif

      // create the threads
      if(0 == tmap_map_driver_create_threads(io_out->fp->header->header, seqs_buffer, records, 
                                             bams, seqs_buffer_length, index, driver, stat,
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
                                             rand_core,
#endif
#ifdef HAVE_LIBPTHREAD
                                             &attr, &threads, &thread_data, rand, stats, realigner, context,
#endif
                                             0)) {
          break;
      }

      /*
      if(-1 != driver->opt->reads_queue_size) {
          tmap_progress_print("writing alignments");
      }
      */

      // write data
      for(i=0;i<seqs_buffer_length;i++) {
#ifdef HAVE_LIBPTHREAD
          if(1 < driver->opt->num_threads) {
              // NB: we will write data as threads process the data.  This is to
              // facilitate SAM/BAM writing, which may be slow, especially for
              // BAM.
              int32_t tid = (i % driver->opt->num_threads);
              while((*thread_data[tid].buffer_idx) <= i) {
                  usleep(1000*1000); // sleep
              }
          }
#endif
          // write
          for(j=0;j<bams[i]->n;j++) { // for each end
              for(k=0;k<bams[i]->bams[j]->n;k++) { // for each hit
                  bam1_t *b = NULL;
                  b = bams[i]->bams[j]->bams[k]; // that's a lot of BAMs
                  if(NULL == b) tmap_bug();
                  if(samwrite(io_out->fp, b) <= 0) {
                      tmap_error("Error writing the SAM file", Exit, WriteFileError);
                  }
              }
          }
          tmap_map_bams_destroy(bams[i]);
          bams[i] = NULL;
      }

#ifdef HAVE_LIBPTHREAD
      // join threads
      if(1 < driver->opt->num_threads) {
          // join threads
          for(i=0;i<driver->opt->num_threads;i++) {
              if(0 != pthread_join(threads[i], NULL)) {
                  tmap_error("error joining threads", Exit, ThreadError);
              }
              // add the stats
              tmap_map_stats_add (stat, stats[i]);
              tmap_map_stats_zero (stats[i]);
              // free the buffer index
              free(thread_data[i].buffer_idx);
          }
          free(threads); threads = NULL;
          free(thread_data); thread_data = NULL;
          free(attr); attr = NULL;
      }
#endif
      // TODO: should we flush when writing SAM and processing one read at a time?

        // print statistics
        n_reads_processed += seqs_buffer_length;
        if (-1 != driver->opt->reads_queue_size) 
        {
            tmap_progress_print2("processed %d reads", n_reads_processed);
            tmap_progress_print2("stats [%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf]",
                                stat->num_with_mapping * 100.0 / (double)stat->num_reads,
                                stat->num_after_seeding/(double)stat->num_with_mapping,
                                stat->num_after_grouping/(double)stat->num_with_mapping,
                                stat->num_after_scoring/(double)stat->num_with_mapping,
                                stat->num_after_rmdup/(double)stat->num_with_mapping,
                                stat->num_after_filter/(double)stat->num_with_mapping);
            if (driver->opt->do_repeat_clip)
            {
                tmap_progress_print2("total %llu, mapped %llu, clipped %llu, rejected %llu, %.2f%% bases]",
                                    stat->num_reads,
                                    stat->num_with_mapping,
                                    stat->num_tailclipped,
                                    stat->num_fully_tailclipped,
                                    ((double) stat->bases_tailclipped) * 100 / stat->bases_seen_tailclipped);
            }
        }
        seqs_loaded = 0;
    }

    if(-1 == driver->opt->reads_queue_size) 
    {
        tmap_progress_print2("processed %d reads", n_reads_processed);
        tmap_progress_print2("stats [%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf]",
                            stat->num_with_mapping * 100.0 / (double)stat->num_reads,
                            stat->num_after_seeding/(double)stat->num_with_mapping,
                            stat->num_after_grouping/(double)stat->num_with_mapping,
                            stat->num_after_scoring/(double)stat->num_with_mapping,
                            stat->num_after_rmdup/(double)stat->num_with_mapping,
                            stat->num_after_filter/(double)stat->num_with_mapping);
    }

    // OUTPUT STATS
    if (driver->opt->report_stats)
    {
        time_t end_time = time (NULL);
        tmap_file_printf ("\nMapping completed in %d seconds.\n", end_time-start_time);
        tmap_file_printf ("                    Total reads:  %llu\n",         stat->num_reads);
        tmap_file_printf ("                   Mapped reads:  %llu (%.2f%%)\n", stat->num_with_mapping, stat->num_with_mapping * 100.0 / (double)stat->num_reads);
        tmap_file_printf ("                  After seeding:  %llu\n", stat->num_after_seeding);
        tmap_file_printf ("                 After grouping:  %llu\n", stat->num_after_grouping);
        tmap_file_printf ("                  After scoring:  %llu\n", stat->num_after_scoring);
        tmap_file_printf ("             After dups removal:  %llu\n", stat->num_after_rmdup);
        tmap_file_printf ("                After filtering:  %llu\n", stat->num_after_filter);

        if (!driver->opt->do_realign)
            tmap_file_printf (  "No realignment perormed\n");
        else
        {
            tmap_file_printf ("Realignment statistics:\n");
            tmap_file_printf ("        Realignment invocations:  %llu\n", stat->num_realign_invocations);
            tmap_file_printf ("           Not realigned (good):  %llu\n");
            if (stat->num_realign_sw_failures)
                tmap_file_printf ("             Algorithm failures:  %llu\n", stat->num_realign_sw_failures);
            if (stat->num_realign_unclip_failures)
                tmap_file_printf ("           Un-clipping failures:  %llu\n", stat->num_realign_sw_failures);
            tmap_file_printf ("             Altered alignments:  %llu", stat->num_realign_changed);
            if (stat->num_realign_invocations)
                tmap_file_printf (" (%.2f%% total)", ((double) stat->num_realign_changed) * 100 / stat->num_realign_invocations);
            tmap_file_printf ("\n");
                tmap_file_printf ("              Altered positions:  %llu", stat->num_realign_shifted);
            if (stat->num_realign_shifted)
                tmap_file_printf (" (%.2f%% total, %.2f%% changed)", ((double) stat->num_realign_shifted) * 100 / stat->num_realign_invocations, ((double) stat->num_realign_shifted) * 100 / stat->num_realign_changed);
            tmap_file_printf ("\n");
        }

        if (!driver->opt->do_hp_weight)
            tmap_file_printf ("No realignment with context-dependent gap cost performed\n");
        else
        {
            tmap_file_printf ("Context-dependent realignment statistics:\n");
            tmap_file_printf ("                    Invocations:  %llu\n", stat->num_hpcost_invocations);
            tmap_file_printf ("             Skipped (too long):  %llu\n", stat->num_hpcost_skipped);
            tmap_file_printf ("             Altered alignments:  %llu", stat->num_hpcost_modified);
            if (stat->num_hpcost_invocations)
                tmap_file_printf (" (%.2f%% total)", ((double) stat->num_hpcost_modified) * 100 / stat->num_hpcost_invocations);
            if (stat->num_hpcost_invocations - stat->num_hpcost_skipped)
            {
                double percent = ((double) stat->num_hpcost_modified * 100) / (stat->num_hpcost_invocations - stat->num_hpcost_skipped);
                tmap_file_printf (" (%.2f%% realigned)", percent);
            }
            tmap_file_printf ("\n");
            tmap_file_printf ("              Altered positions:  %llu\n", stat->num_hpcost_shifted);
        }

        if (!(0 < driver->opt->pen_gapl))
            tmap_file_printf ("No edge long indel salvage performed\n");
        else
        {
            tmap_file_printf ("Long tail indels salvage       : %llu reads\n", stat->reads_salvaged);
            tmap_file_printf ("                                  %8s %8s %8s %8s", "5'fwd", "3'fwd", "5'rev", "3'rev\n");
            tmap_file_printf ("             Salvaged read ends:  %8llu %8llu %8llu %8llu\n",
                              stat->num_salvaged [F5P], 
                              stat->num_salvaged [F3P], 
                              stat->num_salvaged [R5P], 
                              stat->num_salvaged [R3P]);
            tmap_file_printf ("            Average query bases:  %8.1f %8.1f %8.1f %8.1f\n", 
                              stat->num_salvaged [F5P]?(((double) stat->bases_salvaged_qry [F5P]) / stat->num_salvaged [F5P]):0., 
                              stat->num_salvaged [F3P]?(((double) stat->bases_salvaged_qry [F3P]) / stat->num_salvaged [F3P]):0., 
                              stat->num_salvaged [R5P]?(((double) stat->bases_salvaged_qry [R5P]) / stat->num_salvaged [R5P]):0.,
                              stat->num_salvaged [R3P]?(((double) stat->bases_salvaged_qry [R3P]) / stat->num_salvaged [R3P]):0.);
            tmap_file_printf ("        Average reference bases:  %8.1f %8.1f %8.1f %8.1f\n", 
                              stat->num_salvaged [F5P]?(((double) stat->bases_salvaged_ref [F5P]) / stat->num_salvaged [F5P]):0., 
                              stat->num_salvaged [F3P]?(((double) stat->bases_salvaged_ref [F3P]) / stat->num_salvaged [F3P]):0., 
                              stat->num_salvaged [R5P]?(((double) stat->bases_salvaged_ref [R5P]) / stat->num_salvaged [R5P]):0.,
                              stat->num_salvaged [R3P]?(((double) stat->bases_salvaged_ref [R3P]) / stat->num_salvaged [R3P]):0.);
            tmap_file_printf ("           Average score change:  %8.1f %8.1f %8.1f %8.1f\n", 
                              stat->num_salvaged [F5P]?(((double) stat->score_salvaged_total [F5P]) / stat->num_salvaged [F5P]):0., 
                              stat->num_salvaged [F3P]?(((double) stat->score_salvaged_total [F3P]) / stat->num_salvaged [F3P]):0., 
                              stat->num_salvaged [R5P]?(((double) stat->score_salvaged_total [R5P]) / stat->num_salvaged [R5P]):0.,
                              stat->num_salvaged [R3P]?(((double) stat->score_salvaged_total [R3P]) / stat->num_salvaged [R3P]):0.);
        }

        if (!driver->opt->end_repair)
            tmap_file_printf ("No end repair performed\n");
        else if (driver->opt->end_repair <= 2)
            tmap_file_printf ("\"Old-style\" end repair performed, no statistics collected\n");
        else
        {
            tmap_file_printf ("End repair                     : %d reads softclipped, %d extended.\n", stat->reads_end_repair_clipped, stat->reads_end_repair_extended);
            tmap_file_printf ("                                  %8s %8s %8s %8s\n", "5'fwd", "3'fwd", "5'rev", "3'rev");
            tmap_file_printf ("              Clipped read ends:  %8llu %8llu %8llu %8llu\n",
                              stat->num_end_repair_clipped [F5P], 
                              stat->num_end_repair_clipped [F3P], 
                              stat->num_end_repair_clipped [R5P], 
                              stat->num_end_repair_clipped [R3P]);
            tmap_file_printf ("          Average bases clipped:  %8.1f %8.1f %8.1f %8.1f\n", 
                              stat->num_end_repair_clipped [F5P]?(((double) stat->bases_end_repair_clipped [F5P]) / stat->num_end_repair_clipped [F5P]):0., 
                              stat->num_end_repair_clipped [F3P]?(((double) stat->bases_end_repair_clipped [F3P]) / stat->num_end_repair_clipped [F3P]):0., 
                              stat->num_end_repair_clipped [R5P]?(((double) stat->bases_end_repair_clipped [R5P]) / stat->num_end_repair_clipped [R5P]):0.,
                              stat->num_end_repair_clipped [R3P]?(((double) stat->bases_end_repair_clipped [R3P]) / stat->num_end_repair_clipped [R3P]):0.);
            tmap_file_printf ("             Extended read ends:  %8llu %8llu %8llu %8llu\n",
                              stat->num_end_repair_extended [F5P], 
                              stat->num_end_repair_extended [F3P], 
                              stat->num_end_repair_extended [R5P], 
                              stat->num_end_repair_extended [R3P]);
            tmap_file_printf ("         Average bases extended:  %8.1f %8.1f %8.1f %8.1f\n", 
                              stat->num_end_repair_extended [F5P]?(((double) stat->bases_end_repair_extended [F5P]) / stat->num_end_repair_extended [F5P]):0., 
                              stat->num_end_repair_extended [F3P]?(((double) stat->bases_end_repair_extended [F3P]) / stat->num_end_repair_extended [F3P]):0., 
                              stat->num_end_repair_extended [R5P]?(((double) stat->bases_end_repair_extended [R5P]) / stat->num_end_repair_extended [R5P]):0.,
                              stat->num_end_repair_extended [R3P]?(((double) stat->bases_end_repair_extended [R3P]) / stat->num_end_repair_extended [R3P]):0.);
            tmap_file_printf ("        Average indels inserted:  %8.1f %8.1f %8.1f %8.1f\n", 
                              stat->num_end_repair_extended [F5P]?(((double) stat->total_end_repair_indel [F5P]) / stat->num_end_repair_extended [F5P]):0., 
                              stat->num_end_repair_extended [F3P]?(((double) stat->total_end_repair_indel [F3P]) / stat->num_end_repair_extended [F3P]):0., 
                              stat->num_end_repair_extended [R5P]?(((double) stat->total_end_repair_indel [R5P]) / stat->num_end_repair_extended [R5P]):0.,
                              stat->num_end_repair_extended [R3P]?(((double) stat->total_end_repair_indel [R3P]) / stat->num_end_repair_extended [R3P]):0.);
        }
        if (driver->opt->end_repair)
        {
            if (driver->opt->end_repair_5_prime_softclip)
                tmap_file_printf ("5' soft-clipping is allowed for end repair, no softclip removal performed\n");
            else if  (driver->opt->softclip_type != 2 && driver->opt->softclip_type != 3)
                tmap_file_printf ("5' soft-clipping is explicitly allowed through option \"-g %d\" on command line, no softclip removal performed\n", driver->opt->softclip_type);
            else
            {
            tmap_file_printf ("5' softclip removed on %llu reads\n", stat->num_5_softclips [0] + stat->num_5_softclips [1]);
            tmap_file_printf ("                                  %8s %8s\n", "fwd", "rev");
            tmap_file_printf ("            Recovered read ends:  %8llu %8llu\n",
                            stat->num_5_softclips [0], 
                            stat->num_5_softclips [1]);
            tmap_file_printf ("  Average query bases recovered:  %8.1f %8.1f\n", 
                            stat->num_5_softclips [0]?(((double) stat->bases_5_softclips_qry [0]) / stat->num_5_softclips [0]):0., 
                            stat->num_5_softclips [1]?(((double) stat->bases_5_softclips_qry [1]) / stat->num_5_softclips [1]):0.);
            tmap_file_printf ("    Average ref bases recovered:  %8.1f %8.1f\n", 
                            stat->num_5_softclips [0]?(((double) stat->bases_5_softclips_ref [0]) / stat->num_5_softclips [0]):0., 
                            stat->num_5_softclips [1]?(((double) stat->bases_5_softclips_ref [1]) / stat->num_5_softclips [1]):0.);
            tmap_file_printf ("        Average recovered score:  %8.1f %8.1f\n", 
                            stat->num_5_softclips [0]?(((double) stat->score_5_softclips_total [0]) / stat->num_5_softclips [0]):0., 
                            stat->num_5_softclips [1]?(((double) stat->score_5_softclips_total [1]) / stat->num_5_softclips [1]):0.); 
            }
        }

        if (!driver->opt->do_repeat_clip)
            tmap_file_printf ("No tail repeat clipping performed\n");
        else
        {
            tmap_file_printf (  " Alignments tail-clipped: %llu", stat->num_tailclipped);
            if (stat->num_seen_tailclipped)
                tmap_file_printf (                              " (%.2f%% seen)", ((double) stat->num_tailclipped) * 100 / stat->num_seen_tailclipped);
            tmap_file_printf ("\n");
            tmap_file_printf (  "      Tail-clipped bases: %llu", stat->bases_tailclipped);
            if (stat->bases_seen_tailclipped)
                tmap_file_printf (                              " (%.2f%% seen)", ((double) stat->bases_tailclipped) * 100 / stat->bases_seen_tailclipped);
            tmap_file_printf ("\n");
            tmap_file_printf (  "Completely clipped reads: %llu", stat->num_fully_tailclipped);
            if (stat->num_seen_tailclipped)
                tmap_file_printf (                              " (%.2f%% clipped)", ((double) stat->num_fully_tailclipped) * 100 / stat->num_tailclipped);
            tmap_file_printf (", contain %llu bases", stat->bases_fully_tailclipped);
            if (stat->bases_tailclipped)
                tmap_file_printf (" (%.2f%% clipped)", ((double) stat->bases_fully_tailclipped) * 100 / stat->bases_tailclipped);
            tmap_file_printf ("\n");
            if (stat->num_seen_tailclipped)
            {
                tmap_file_printf (    "   Average bases clipped:\n");
                tmap_file_printf (    "                    per read: %.1f\n", ((double) stat->bases_tailclipped) / stat->num_seen_tailclipped);
                if (stat->num_tailclipped)
                    tmap_file_printf ("            per clipped read: %.1f\n", ((double) stat->bases_tailclipped) / stat->num_tailclipped);
                if (stat->num_fully_tailclipped)
                    tmap_file_printf ("      per fully clipped read: %.1f\n", ((double) stat->bases_fully_tailclipped) / stat->num_fully_tailclipped);
            }
        }
        if (!driver->opt->min_al_len && !driver->opt->min_al_cov && !driver->opt->min_identity)
            tmap_file_printf ("No alignment filtering performed\n");
        else
        {
            tmap_file_printf (  "  Filtered alignments: %llu\n", stat->num_filtered_als);
        }
  }


  tmap_progress_print2("cleaning up");

  // cleanup the algorithm persistent data
  tmap_map_driver_do_cleanup(driver);

  // close the input/output
  tmap_sam_io_destroy(io_out);

  // free memory
  tmap_index_destroy(index);
  tmap_seqs_io_destroy(io_in);
  for(i=0;i<reads_queue_size;i++) {
      tmap_seqs_destroy(seqs_buffer[i]);
#ifdef HAVE_LIBPTHREAD
      tmap_seqs_destroy(seqs_buffer_next[i]);
#endif
  }
  free(seqs_buffer);
#ifdef HAVE_LIBPTHREAD
  free(seqs_buffer_next);
#endif

// DVK - realigner
#ifdef HAVE_LIBPTHREAD
  for (i = 0; i != driver->opt->num_threads; ++i)
  {
      realigner_destroy (realigner [i]);
      realigner_destroy (context [i]);
  }
  free (realigner);
  free (context);

#else
  realigner_destroy (realigner);
  realigner_destroy (context);
#endif

  tmap_log_disable ();
  if (logfile)
    fclose (logfile);

  free(records);
  free(bams);
  tmap_map_stats_destroy(stat);
#ifdef HAVE_LIBPTHREAD
  for(i=0;i<driver->opt->num_threads;i++) {
      tmap_map_stats_destroy(stats[i]);
      tmap_rand_destroy(rand[i]);
  }
  free(stats);
  free(rand);
  free(thread_io);
#else
  tmap_rand_destroy(rand);
#endif
#ifdef ENABLE_TMAP_DEBUG_FUNCTIONS
  tmap_rand_destroy(rand_core);
#endif

  // if (driver->opt->report_stats)
  //    tmap_file_fclose (tmap_file_stdout);
}

/* MAIN API */

tmap_map_driver_algorithm_t*
tmap_map_driver_algorithm_init(tmap_map_driver_func_init func_init,
                    tmap_map_driver_func_thread_init func_thread_init,
                    tmap_map_driver_func_thread_map func_thread_map,
                    tmap_map_driver_func_thread_cleanup func_thread_cleanup,
                    tmap_map_driver_func_cleanup func_cleanup,
                    tmap_map_opt_t *opt)
{
  tmap_map_driver_algorithm_t *algorithm = NULL;

  // the only necessary function is func_thread_map
  if(func_thread_map == NULL ) {
      tmap_error("func_thread_map cannot be null", Exit, OutOfRange);
  }

  algorithm = tmap_calloc(1, sizeof(tmap_map_driver_algorithm_t), "algorithm");
  algorithm->func_init = func_init;
  algorithm->func_thread_init = func_thread_init;
  algorithm->func_thread_map = func_thread_map;
  algorithm->func_thread_cleanup = func_thread_cleanup;
  algorithm->func_cleanup = func_cleanup;
  algorithm->opt = opt;
  algorithm->data = NULL;
  algorithm->thread_data = tmap_calloc(opt->num_threads, sizeof(void*), "algorithm->thread_data");
  return algorithm;
}

void
tmap_map_driver_algorithm_destroy(tmap_map_driver_algorithm_t *algorithm)
{
  free(algorithm->thread_data);
  free(algorithm);
}

tmap_map_driver_stage_t*
tmap_map_driver_stage_init(int32_t stage)
{
  tmap_map_driver_stage_t *s = NULL;
  s = tmap_calloc(1, sizeof(tmap_map_driver_stage_t), "stage");
  s->stage = stage;
  s->opt = tmap_map_opt_init(TMAP_MAP_ALGO_STAGE);
  s->opt->algo_stage = stage;
  return s;
}

void
tmap_map_driver_stage_add(tmap_map_driver_stage_t *s,
                    tmap_map_driver_func_init func_init,
                    tmap_map_driver_func_thread_init func_thread_init,
                    tmap_map_driver_func_thread_map func_thread_map,
                    tmap_map_driver_func_thread_cleanup func_thread_cleanup,
                    tmap_map_driver_func_cleanup func_cleanup,
                    tmap_map_opt_t *opt)
{
  // check against stage options
  tmap_map_opt_check_stage(s->opt, opt);
  s->num_algorithms++;
  s->algorithms = tmap_realloc(s->algorithms, sizeof(tmap_map_driver_algorithm_t*) * s->num_algorithms, "s->algorithms");
  s->algorithms[s->num_algorithms-1] = tmap_map_driver_algorithm_init(func_init, func_thread_init, func_thread_map,
                                                                      func_thread_cleanup, func_cleanup, opt);
}

void
tmap_map_driver_stage_destroy(tmap_map_driver_stage_t *stage)
{
  int32_t i;
  for(i=0;i<stage->num_algorithms;i++) {
      tmap_map_driver_algorithm_destroy(stage->algorithms[i]);
  }
  tmap_map_opt_destroy(stage->opt);
  free(stage->algorithms);
  free(stage);
}

tmap_map_driver_t*
tmap_map_driver_init(int32_t algo_id, tmap_map_driver_func_mapq func_mapq)
{
  tmap_map_driver_t *driver = NULL;
  driver = tmap_calloc(1, sizeof(tmap_map_driver_t), "driver");
  driver->func_mapq = func_mapq;
  driver->opt = tmap_map_opt_init(algo_id);
  return driver;
}

void
tmap_map_driver_add(tmap_map_driver_t *driver,
                    tmap_map_driver_func_init func_init,
                    tmap_map_driver_func_thread_init func_thread_init,
                    tmap_map_driver_func_thread_map func_thread_map,
                    tmap_map_driver_func_thread_cleanup func_thread_cleanup,
                    tmap_map_driver_func_cleanup func_cleanup,
                    tmap_map_opt_t *opt)
{
  // make more stages
  if(driver->num_stages < opt->algo_stage) {
      driver->stages = tmap_realloc(driver->stages, sizeof(tmap_map_driver_stage_t*) * opt->algo_stage, "driver->stages");
      while(driver->num_stages < opt->algo_stage) {
          driver->num_stages++;
          driver->stages[driver->num_stages-1] = tmap_map_driver_stage_init(driver->num_stages);
          // copy global options into this stage
          tmap_map_opt_copy_global(driver->stages[driver->num_stages-1]->opt, driver->opt);
          // copy stage options into this stage
          tmap_map_opt_copy_stage(driver->stages[driver->num_stages-1]->opt, opt);
      }
  }

  // check options
  tmap_map_opt_check(opt);

  // check against global options
  tmap_map_opt_check_global(driver->opt, opt);

  // add to the stage
  tmap_map_driver_stage_add(driver->stages[opt->algo_stage-1],
                    func_init,
                    func_thread_init,
                    func_thread_map,
                    func_thread_cleanup,
                    func_cleanup,
                    opt);
}

void
tmap_map_driver_run(tmap_map_driver_t *driver)
{
  tmap_map_driver_core(driver);
}

void
tmap_map_driver_destroy(tmap_map_driver_t *driver)
{
  int32_t i;
  for(i=0;i<driver->num_stages;i++) {
      tmap_map_driver_stage_destroy(driver->stages[i]);
  }
  tmap_map_opt_destroy(driver->opt);
  free(driver->stages);
  free(driver);
}
