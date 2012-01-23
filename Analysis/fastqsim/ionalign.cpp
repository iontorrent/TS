/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>

#include "../fstrcmp.h"

static char *refSeq = NULL;
static int refLen = 0;

void FreeGenome()
{
        if (refSeq != NULL)
                free(refSeq);
        refSeq = NULL;
}

void LoadGenome(char *fileName)
{
        if (refSeq != NULL)
                FreeGenome();

        FILE *fp = fopen(fileName, "r");
        if (fp) {
                fseek(fp, 0, SEEK_END);
                int bytes = ftell(fp);
                refSeq = (char *)malloc(bytes);
                char *ptr = refSeq;
                fseek(fp, 0, SEEK_SET);
                char line[512];
                fgets(line, sizeof(line), fp);
                while (fgets(line, sizeof(line), fp)) {
                        int len = strlen(line);
                        while (len > 0 && (line[len-1] == '\r' || line[len-1] == '\n')) { // strip new lines & carriage returns
                                line[len-1] = 0;
                                len--;
                        }
                        memcpy(ptr, line, len);
                        ptr += len;
                        refLen += len;
                }
                *ptr = 0;
                fclose(fp);
                printf("Loaded reference genome with %d bases\n", refLen);
        }
}

int align(char *line, int len)
{
	if (strstr(refSeq, line))
		return 0;
	else
		return len;

	int i;
	double result;
	int edit1, edit2;
	double minimum = 0.8; // strings so dissimilar that we stop wasting time
	for(i=0;i<refLen-len;i++) {
		result = fstrcmp(&refSeq[i], len, line, len, minimum, &edit1, &edit2);
	}
	int sum = edit1 + edit2;
	return sum;
}

int main(int argc, char *argv[])
{
	// load up the ref genome
	LoadGenome("CP000948.fna");

	// loop through our fastq reads, one at a time, and attempt to align each
	char line[MAX_PATH_LENGTH];
	int len;
	FILE *fp = fopen(argv[1], "r");
	int alignCount = 0;
	int readCount = 0;
	if (fp) {
		while(fgets(line, sizeof(line), fp)) {
			if (line[0] == '@') {
				fgets(line, sizeof(line), fp);
				len = strlen(line);
				len--;
				line[len] = 0; // remove the return char
				int ret = align(line, len);
				if (ret < 3) {
					alignCount++;
				}
				readCount++;
				if (readCount % 1 == 0) {
					printf(".");
					fflush(stdout);
				}
			}
		}
		fclose(fp);
	}

	printf("Aligned %d of %d reads\n", alignCount, readCount);
}

