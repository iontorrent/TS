/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "fastq_file.h"

int fq_write (FILE *fd, fq_t *fq)
{
	int count = 0;
    
    fprintf (fd, "@");
    fprintf (fd, "%s\n", fq->name);
    fprintf (fd, "%s\n", fq->seq);
    fprintf (fd, "+\n");
    fprintf (fd, "%s\n", fq->qual);
    
	return count;
}

FILE *fq_file_open (char *filename)
{
	FILE *fq_fd = NULL;
    if (NULL == (fq_fd = fopen (filename, "wb")))
    {
    	fprintf (stderr, "%s: %s\n", filename, strerror(errno));
		exit (EXIT_FAILURE);
    }
    return fq_fd;
}
