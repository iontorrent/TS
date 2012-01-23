/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FASTQ_FILE_H
#define FASTQ_FILE_H

#include <stdlib.h>
#include <stdio.h>

#include "fastq.h"

#ifdef __cplusplus
extern "C" {
#endif

	/*!
    	Open file pointer
        @param	filename	pointer to filename to open
        @return		pointer to file descriptor
     */
	FILE *fq_file_open (char *filename);
    
    /*!
	    Write fastq read to file descriptor
        @param	fd	a file pointer to which to write
        @param	fq	pointer to the fastq to write
        @return		the number of bytes written
        */
    int fq_write (FILE *fd, fq_t *fq);
    
#ifdef __cplusplus
}
#endif
#endif // FASTQ_FILE_H
