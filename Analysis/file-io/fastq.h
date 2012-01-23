/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FASTQ_H
#define FASTQ_H

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#include "sff_definitions.h"

typedef struct {
	char *name;
    char *seq;
    char *qual;
    int l;
} fq_t;

#define DEFAULT_QUAL_OFFSET 33

#ifdef __cplusplus
extern "C" {
#endif

    /*!
	    Create fastq structure from sff honoring the clip values in SFF
        @param	sff	pointer to SFF to convert
        @return		pointer to fastq
     */
    fq_t *sff2fq (sff_t *sff);

    /*!
	    @return		a pointer to the empty fq
     */
    fq_t *fq_init ();

    /*!
	    @param	fq	pointer to fq to destroy
     */
    void fq_destroy (fq_t *fq);

	/*!
    	Convert integer quality score string to char string
        @param	quality	pointer to string
        @return		pointer to string
     */
    char *qual_to_fastq (char *quality);
    
	/*!
    	Convert an integer quality score to a char
        @param	q	integer to convert
        @param	offset	character offset
     */
    char QualToFastQ(int q, int offset);

#ifdef __cplusplus
}
#endif
#endif // FASTQ_H
