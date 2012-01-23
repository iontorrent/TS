/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "sff.h"
#include "fastq.h"
#include "sff_definitions.h"


fq_t *sff2fq (sff_t *sff)
{
	fq_t *fq = NULL;
	if (NULL == (fq = fq_init())){
        return fq;
    }
    
	int clip_left_index = 0;
	int clip_right_index = 0;
	clip_left_index = max (1, max (sff_clip_qual_left (sff), sff_clip_adapter_left (sff)));
	clip_right_index = min ((sff_clip_qual_right (sff) == 0 ? sff_n_bases(sff):sff_clip_qual_right(sff)),
							(sff_clip_adapter_right(sff) == 0 ? sff_n_bases(sff):sff_clip_adapter_right(sff)));
    // NB: clip_right_index and clip_left_offset now form a 0-based, half-open range
    const int clip_left_offset = clip_left_index - 1;
    assert(clip_left_offset >= 0);
    // If right clip precedes left, clip is invalid and truncated to 0-length read
    const int clip_seqlen = (clip_right_index >= clip_left_offset) ? clip_right_index - clip_left_offset : 0;
    assert(clip_seqlen >= 0);
    assert(clip_seqlen <= strlen(sff_bases(sff)));

    //copy name
    fq->name = strdup (sff_name(sff));

    //copy sequence
    fq->seq = (char *) malloc (strlen (sff_bases(sff))+1);
    strncpy (fq->seq, sff_bases(sff)+clip_left_offset, clip_seqlen);
    fq->seq[clip_seqlen] = '\0';
       
    //copy quality
    char *qual = qual_to_fastq (sff_quality(sff));
    fq->qual = (char *) malloc (strlen (sff_quality(sff))+1);
    strncpy (fq->qual, qual+clip_left_offset, clip_seqlen);
    fq->qual[clip_seqlen] = '\0';
    free (qual);
    
    //set length
    fq->l = strlen (fq->seq);
    
    return fq;
}

fq_t *fq_init ()
{
	fq_t *fq = NULL;
    fq = (fq_t *) malloc (sizeof (fq_t));
    if (fq == NULL)
    {
    	fprintf (stderr, "malloc: %s\n", strerror(errno));
    	return fq;
    }
    fq->name = NULL;
    fq->seq = NULL;
    fq->qual = NULL;
    fq->l = 0;
    
	return fq;
}

void fq_destroy (fq_t *fq)
{
	if (fq != NULL) {
    	if (fq->name) free (fq->name);
        if (fq->seq) free (fq->seq);
        if (fq->qual) free (fq->qual);
        free (fq);
        fq = NULL;
    }
}



char *qual_to_fastq (char *quality)
{
	char *fastqual = NULL;
	unsigned int b;
    fastqual = (char *) malloc (strlen(quality)+1);
	for (b=0;b<strlen(quality);b++)
		fastqual[b] = QualToFastQ(quality[b],DEFAULT_QUAL_OFFSET);
	fastqual[strlen(quality)] = '\0';
	return fastqual;
}

char QualToFastQ(int q, int offset)
{
	// from http://maq.sourceforge.net/fastq.shtml
	char c = (q <= 93 ? q : 93) + offset;
	return c;
}
