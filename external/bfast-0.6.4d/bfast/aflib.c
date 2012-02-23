#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <bzlib.h>
#include <zlib.h>
#include <ctype.h>
#include <config.h>
#include "aflib.h"

void AFILE_print_error(char *message)
{
	fprintf(stderr, "%s\n", message);
	exit(1);
}

AFILE *AFILE_afopen(const char* path, const char *mode, int32_t compression) 
{
	AFILE *afp = NULL;
	if(NULL == strchr(mode, 'r') && NULL == strchr(mode, 'w')) {
		AFILE_print_error("Improper mode");
	}

	afp = calloc(1, sizeof(AFILE));
	if(NULL == afp) AFILE_print_error("Could not allocate memory\n");
	afp->fp=NULL;
#ifndef DISABLE_BZ2 
	afp->bz2=NULL;
	afp->n_unused=0;
#endif
	afp->gz=NULL;
	afp->c=compression;

	switch(afp->c) {
		case AFILE_NO_COMPRESSION:
			afp->fp = fopen(path, mode);
			if(NULL == afp->fp) {
				free(afp); 
				return NULL;
			}
			break;
#ifndef DISABLE_BZ2 
		case AFILE_BZ2_COMPRESSION:
			afp->fp = fopen(path, mode);
			if(NULL == afp->fp) {
				free(afp); 
				return NULL;
			}
			if(NULL != strchr(mode, 'r')) {
				afp->open_type = AFILE_BZ2_READ;
				afp->bz2 = BZ2_bzReadOpen(&afp->bzerror, afp->fp, 0, 0, afp->unused, afp->n_unused);
				if(NULL == afp->bz2) {
					free(afp); 
					return NULL;
				}
			}
			else {
				afp->open_type = AFILE_BZ2_WRITE;
				// 900K blockSize100k
				// 30 workFactor
				afp->bz2 = BZ2_bzWriteOpen(&afp->bzerror, afp->fp, 9, 0, 30); 
			}
			break;
#endif
		case AFILE_GZ_COMPRESSION:
			afp->gz = gzopen(path, mode);
			if(NULL == afp->gz) {
				free(afp); 
				return NULL;
			}
			break;
		default:
			AFILE_print_error("Could not recognize compresssion\n");
			break;
	}

	return afp;
}

AFILE *AFILE_afdopen(int filedes, const char *mode, int32_t compression) 
{
	AFILE *afp = NULL;
	if(NULL == strchr(mode, 'r') && NULL == strchr(mode, 'w')) {
		AFILE_print_error("Improper mode");
	}

	afp = calloc(1, sizeof(AFILE));
	if(NULL == afp) AFILE_print_error("Could not allocate memory\n");
	afp->fp=NULL;
#ifndef DISABLE_BZ2 
	afp->bz2=NULL;
	afp->n_unused=0;
#endif
	afp->gz=NULL;
	afp->c=compression;

	switch(afp->c) {
		case AFILE_NO_COMPRESSION:
			afp->fp = fdopen(filedes, mode);
			if(NULL == afp->fp) {
				free(afp); 
				return NULL;
			}
			break;
#ifndef DISABLE_BZ2 
		case AFILE_BZ2_COMPRESSION:
			afp->fp = fdopen(filedes, mode);
			if(NULL == afp->fp) {
				free(afp); 
				return NULL;
			}
			if(NULL != strchr(mode, 'r')) {
				afp->open_type = AFILE_BZ2_READ;
				afp->bz2 = BZ2_bzReadOpen(&afp->bzerror, afp->fp, 0, 0, afp->unused, afp->n_unused);
				if(NULL == afp->bz2) {
					free(afp); 
					return NULL;
				}
			}
			else {
				afp->open_type = AFILE_BZ2_WRITE;
				// 900K blockSize100k
				// 30 workFactor
				afp->bz2 = BZ2_bzWriteOpen(&afp->bzerror, afp->fp, 9, 0, 30); 
			}
#endif
		case AFILE_GZ_COMPRESSION:
			afp->gz = gzdopen(filedes, mode);
			if(NULL == afp->gz) {
				free(afp); 
				return NULL;
			}
			break;
		default:
			AFILE_print_error("Could not recognize compresssion\n");
			break;
	}

	return afp;
}

void AFILE_afclose(AFILE *afp) 
{
	switch(afp->c) {
		case AFILE_NO_COMPRESSION:
#ifndef DISABLE_BZ2 
		case AFILE_BZ2_COMPRESSION:
			if(AFILE_BZ2_WRITE == afp->open_type) {
				BZ2_bzWriteClose(&afp->bzerror, afp->bz2, 0, NULL, NULL);
				if(afp->bzerror == BZ_IO_ERROR) {
					AFILE_print_error("Could not close the stream after writing");
				}
			}
			fclose(afp->fp);
			break;
#endif
		case AFILE_GZ_COMPRESSION:
			gzclose(afp->gz);
			break;
		default:
			AFILE_print_error("Could not recognize compresssion\n");
			break;
	}

	free(afp);
}


size_t AFILE_afread(void *ptr, size_t size, size_t count, AFILE *afp) 
{
#ifndef DISABLE_BZ2 
	int32_t nbuf=0, i;
	void *unused_tmp_void=NULL;
	char *unused_tmp=NULL;
#endif

	switch(afp->c) {
		case AFILE_NO_COMPRESSION:
			return fread(ptr, size, count, afp->fp);
#ifndef DISABLE_BZ2 
		case AFILE_BZ2_COMPRESSION:
			while(0 == nbuf && 
					!(BZ_STREAM_END == afp->bzerror && 0 == afp->n_unused && feof(afp->fp))) {
				nbuf = BZ2_bzRead(&afp->bzerror, afp->bz2, ptr, size*count);
				if(BZ_OK == afp->bzerror) {
					// return # of bytes
					return nbuf;
				}
				else if(BZ_STREAM_END == afp->bzerror) {
					// Get unused
					BZ2_bzReadGetUnused(&afp->bzerror, afp->bz2, &unused_tmp_void, &afp->n_unused);
					if(BZ_OK != afp->bzerror) AFILE_print_error("Could not BZ2_bzReadGetUnused"); 
					unused_tmp = (char*)unused_tmp_void;
					for(i=0;i<afp->n_unused;i++) {
						afp->unused[i] = unused_tmp[i];
					}
					// Close
					BZ2_bzReadClose(&afp->bzerror, afp->bz2);
					if(BZ_OK != afp->bzerror) AFILE_print_error("Could not BZ2_bzReadClose"); 
					afp->bzerror = BZ_STREAM_END; // set to the stream end for next call to this function
					// Open again if necessary
					if(0 == afp->n_unused && feof(afp->fp)) {
						return nbuf;
					}
					else {
						afp->bz2 = BZ2_bzReadOpen(&afp->bzerror, afp->fp, 0, 0, afp->unused, afp->n_unused);
						if(NULL == afp->bz2) AFILE_print_error("Could not open file");
					}
				}
				else {
					fprintf(stderr, "nbuf = %d\n", nbuf);
					fprintf(stderr, "afp->bzerror = %d\n", afp->bzerror);
					AFILE_print_error("Could not read");
				}
			}
			return nbuf;
#endif
		case AFILE_GZ_COMPRESSION:
			return gzread(afp->gz, ptr, size*count);
			break;
		default:
			AFILE_print_error("Could not recognize compresssion\n");
			break;
	}
	return 0;
}

int AFILE_afread2(AFILE *afp, void *ptr, unsigned int len)
{
	return AFILE_afread(ptr, sizeof(char), (size_t)len, afp);
}

size_t AFILE_afwrite(void *ptr, size_t size, size_t count, AFILE *afp) 
{
	switch(afp->c) {
		case AFILE_NO_COMPRESSION:
			return fwrite(ptr, size, count, afp->fp);
#ifndef DISABLE_BZ2 
		case AFILE_BZ2_COMPRESSION:
			return BZ2_bzwrite(afp->bz2, ptr, size*count);
#endif
		case AFILE_GZ_COMPRESSION:
			return gzwrite(afp->gz, ptr, size*count);
		default:
			AFILE_print_error("Could not recognize compresssion\n");
			break;
	}
	return 0;
}
