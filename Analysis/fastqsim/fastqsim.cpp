/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// simple tool used to generate a bunch of fastq formatted reads
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static char *refSeq = NULL;
static int refLen = 0;
char *LibraryKey = "TCAG";
int LibraryKeyLen = 4;


double random(double val)
{
        double r = rand() / (double)RAND_MAX;
        return r * val;
}

// gauss_random - gaussian distributed random number generator
// sigma - the standard deviation, use 1.0 for gauss
double gauss_random(double sigma)
{
        double x, y, r2;
        do {
                x = -1.0 + random(2.0);
                y = -1.0 + random(2.0);
                r2 = x*x + y*y;
        } while (r2 > 1.0 || r2 == 0);
        return sigma * y * sqrt(-2.0 * log(r2)/r2);
}

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
                assert(fgets(line, sizeof(line), fp));
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

char *GetLibrarySequence(int offset, int len, double indelProb, int maxErrors, bool includeKey)
{
        static char seq[10000];

        if ((offset+len) >= refLen)
                offset = refLen-len;

	int i;
	int k = LibraryKeyLen;
	if (includeKey)
		k = 0;
	char base;
	int numErrors = 0;
	for(i=0;i<len;i++) {
		double prob = random(1.0);
		if (k < LibraryKeyLen) {
			base = LibraryKey[k];
			if ((prob < indelProb) && (numErrors < maxErrors)) { // del
				k += 2;
				numErrors++;
			} else if ((prob > (1.0-indelProb)) && (numErrors < maxErrors)) { // insert
				k = k;
				numErrors++;
			} else { // normal
				k++;
			}
		} else {
			base = refSeq[offset];
			if ((prob < indelProb) && (numErrors < maxErrors)){ // del
				offset += 2;
				numErrors++;
			} else if ((prob > (1.0-indelProb)) && (numErrors < maxErrors)){ // insert
				offset = offset;
				numErrors++;
			} else { // normal
				offset++;
			}
		}
		seq[i] = base;
	}
	seq[i] = 0; // NULL-terminate the string

        return seq;
}

char QualToFastQ(int q)
{
        // from http://maq.sourceforge.net/fastq.shtml
        char c = (q <= 93 ? q : 93) + 33;
        return c;
}

int FastQToQual(char c)
{
        return c - 33;
}

//int main(int argc, char *argv[])
int main ()
{
	// load up our genome
	// LoadGenome("CP000948.fna");
	LoadGenome("NC_008253.fna");

	int numReads = 10000;
	const int readLenMean = 35;
	const double indelProbPerBase = 1.0/readLenMean;
	const int maxErrors = 1;

	FILE *fpq = fopen("simreads.fnq", "w");

	int i;
	for(i=0;i<numReads;i++) {
		int base, len;
		len = readLenMean + (int)(10.0 * gauss_random(1.0) + 0.5);
		if (len > 4) {
			if (len > 40)
				len = 40;
			// write the entry name
			fprintf(fpq, "@%s_%05d\n", "sim", i);
			// generate a simulated sequence
			char *read = GetLibrarySequence((rand()%refLen), len, indelProbPerBase, maxErrors, false);
			// write out the sequence
			for(base=0;base<len;base++)
				fprintf(fpq, "%c", read[base]);
			// write out the quality line
			fprintf(fpq, "\n+\n");
			for(base=0;base<len;base++)
				fprintf(fpq, "%c", QualToFastQ(17));
			fprintf(fpq, "\n");
		}
	}

	fclose(fpq);
}

