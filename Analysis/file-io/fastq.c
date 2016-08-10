/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#include "fastq.h"


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
