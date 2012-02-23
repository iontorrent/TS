/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// basic sff reader

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../file-io/ion_util.h"
#include "IonVersion.h"

#include "sff.h"
#include "sff_file.h"
#include "sff_header.h"
#include "sff_read_header.h"
#include "sff_read.h"

#define DEFAULT_QUAL_OFFSET 33

#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

int usage_read ()
{
	fprintf (stderr, "\n");
  fprintf (stderr, "SFFRead - Converts SFF formatted file to FASTQ formatted file.\n");
  fprintf (stderr, "\n");
  fprintf (stderr, "Usage: %s read [OPTIONS] filename\n", "SFFUtils");
  fprintf (stderr, "\n");
  fprintf (stderr, "Options:\n");
	fprintf (stderr, "   -a\tPrints all reads to stdout.  By default, only first ten reads are printed to stdout.\n");
	fprintf (stderr, "   -R\tPrint the flow values of the read with this Row value (Note: you must also specify the -C option).\n");
	fprintf (stderr, "   -C\tPrint the flow values of the read with this Column value (Note: you must also specify the -R option).\n");
	fprintf (stderr, "   -q\tWrite all reads to the specified filename in fastq format. ('SFFRead -q keypass.fastq rawlib.sff').\n");
	fprintf (stderr, "   -c\tForce clipping of the key sequence from fastq file output.  Note there is no way to override the clipping values set in SFF file.\n");
	fprintf (stderr, "   -s\tOffset to apply to quality scores (Default = %d).\n", DEFAULT_QUAL_OFFSET);
	fprintf (stderr, "   -d\tAdditional debug info printed to stdout.\n");
	fprintf (stderr, "   -k\tSpecify the key pattern (Default is TCAG)  Please use all capital letters.\n");
	fprintf (stderr, "   -L\tLegacy format read name in fastq file output, ie. r10|c100.\n");
	fprintf (stderr, "   -u\tDo not trim adapters from the reads (ignores sff adapter clip values, still uses quality clip)\n");
	fprintf (stderr, "   -v\tPrint version information and exit.\n");
	fprintf (stderr, "\n");
	return 1;
}


char QualToFastQ(int q, int offset)
{
	// from http://maq.sourceforge.net/fastq.shtml
	char c = (q <= 93 ? q : 93) + offset;
	return c;
}

int FastQToQual(char c, int qual_offset)
{
	return c - qual_offset;
}

static char *hackkey = "TCAG";
static int hackkeylen = 4;

int main_read(int argc, char *argv[])
{
	char	*fastqFileName = NULL;
	char	*sffFileName = NULL;
	bool	forceClip = false;
	bool	keyPass = false;
	bool	allReads = false;
	int	readCol = -1;
	int	readRow = -1;
	bool	findRead = false;
	int row, col;
	int numKeypassedReads = 0;
	int qual_offset = DEFAULT_QUAL_OFFSET;
	bool legacyFASTQName = false;	// enable if you want r10|c100 format name in fastq file
	bool debug = false;
	bool legacyReadName = false;
	bool adapterTrim = true;
	
	// process command-line args
	int argcc = 1;
	while (argcc < argc) {
		if (argv[argcc][0] == '-') {
			switch (argv[argcc][1]) {
				case 'a': // output all reads
					allReads = true;
				break;

				case 'R': // report read at row & column
					argcc++;	
					readRow = atoi(argv[argcc]);
				break;

				case 'C': // report read at row & column
					argcc++;	
					readCol = atoi(argv[argcc]);
				break;

				case 'q': // convert to fastq
					argcc++;
					fastqFileName = argv[argcc];
				break;

				case 'c': // force qual clip left to 5
					forceClip = true;
				break;

				case 's':	// Offset to apply to quality scores
					argcc++;
					qual_offset = atoi(argv[argcc]);
					if(qual_offset==0) {
						fprintf (stderr, "-s option should specify a nonzero quality offset\n");
						return 1;
					}
				break;
				
				case 'k': // force keypass
					keyPass = true;
					argcc++;
					hackkey = argv[argcc];
					hackkeylen = strlen(hackkey);
				break;
			
				case 'L':	// don't record name of read in comment
					legacyFASTQName = true;
				break;
			
				case 'd':	// enable debug print outs
					debug = true;
				break;
			
			
			  case 'u':	// prevent read clipping
					adapterTrim = false;
				break;
			
				case 'v':	// version info
					fprintf (stdout, "%s", IonVersion::GetFullVersion("SFFRead").c_str());
					return 0;
				break;

        case 'h': // help info
        default:
          return usage_read();
			}
		}
		else {
			sffFileName = argv[argcc];
		}
		argcc++;
	}

	if (!sffFileName)
    return usage_read();

	if (readCol > -1 && readRow > -1) {
		findRead = true;
		allReads = true;// makes it search all reads
	}

    sff_file_t* sff_file_in = NULL;
    sff_file_in = sff_fopen(sffFileName, "rb", NULL, NULL);

	if (sff_file_in) {
		if (!findRead && !fastqFileName) {
			printf("Reading file: %s\n", sffFileName);
            sff_header_print(stdout, sff_file_in->header);
		}

		// -- read the reads
		int numReads;
		if (allReads) {
			numReads = sff_file_in->header->n_reads;
		}
		else {
			numReads = (sff_file_in->header->n_reads < 10 ? sff_file_in->header->n_reads:10);
		}
		FILE *fpq = NULL;
		if (fastqFileName) {
			numReads = sff_file_in->header->n_reads;
			fpq = fopen(fastqFileName, "w");
			if (!fpq){
				perror (fastqFileName);
				exit (1);
			}
		}

		for(int i=0;i<numReads;i++) {
            sff_read_header_t* rh = sff_read_header_read(sff_file_in->fp);
            sff_read_t* rr = sff_read_read(sff_file_in->fp, sff_file_in->header, rh);

			// optional - ignore the left & right adapter clipping by simply setting these values to 0
			if (!adapterTrim) {
				rh->clip_adapter_left = 0;
				rh->clip_adapter_right = 0;
			}

			if (!fpq && !findRead) {
				printf("Read header length: %d\n", rh->rheader_length);
				printf("Read name length: %d\n", rh->name_length);
			}

			
			// Extract the row and column popsition info for this read
            if (1 != ion_readname_to_rowcol(rh->name->s, &row, &col)) {
                fprintf (stderr, "Error parsing read name: '%s'\n", rh->name->s);
                continue;
            }
            if(1 == ion_readname_legacy(rh->name->s)) {
                legacyReadName = true;
			}
			else {
                legacyReadName = false;
			}
			

			if (!fpq && !findRead) {
				printf("Read: %s (r%05d|c%05d) has %d bases\n",
						(rh->name_length > 0 ? rh->name->s : "NONAME"),
						row, col,
						rh->n_bases);
				printf("Clip left: %d qual: %d right: %d qual: %d\n",
					rh->clip_adapter_left, rh->clip_qual_left,
					rh->clip_adapter_right, rh->clip_qual_right);
				printf("Flowgram values:\n");
			}

			if (findRead) {
				if (row == readRow && col == readCol) {
					//printf("Ionogram: ");
					int i;
					for(i=0;i<sff_file_in->header->flow_length;i++) {
						printf("%.2lf ", (double)(rr->flowgram[i])/100.0);
					}
					printf("\n");
					
					//// now print the bases - all the bases, not clipped!
					//// these bases correspond to the raw flowgram data. in essence
					//for (int b=0;b<r.number_of_bases;b++)
					//	fprintf(stdout, "%c", bases[b]);
					//fprintf(stdout, "\n");
				}
			}
			else if (fpq) {
				bool ok = true;
				if (keyPass) {
					// if (r.number_of_bases > h.key_length) {
					if ((int)rh->n_bases > hackkeylen) {
						int b;
						// for(b=0;b<h.key_length;b++) {
						for(b=0;b<hackkeylen;b++) {
							// if (key_sequence[b] != bases[b]) {
							if (hackkey[b] != rr->bases->s[b]) {
								ok = false;
								break;
							}
						}
					} else
						ok = false; // not long enough
				}

				int clip_left_index = 0;
				int clip_right_index = 0;
				if (ok) {
					//numKeypassedReads++;
					
					// If force-clip option is set, we want to ensure the key gets trimmed
					if (forceClip && rh->clip_adapter_left < 4)
						rh->clip_adapter_left = hackkeylen+1;

					clip_left_index = max (1, max (rh->clip_qual_left, rh->clip_adapter_left));
					clip_right_index = min ((rh->clip_qual_right == 0 ? rh->n_bases:rh->clip_qual_right),
											(rh->clip_adapter_right == 0 ? rh->n_bases:rh->clip_adapter_right));
					if (debug)
						fprintf (stdout, "debug clip: left = %d right = %d\n", clip_left_index, clip_right_index);
					numKeypassedReads++;
					if (clip_left_index > clip_right_index)
						// Suppress output of zero-mer reads (left > right)
						ok = false;
				}
				if (ok) {
					//print id string
					if (legacyFASTQName) {
						fprintf (fpq, "@r%d|c%d\n", row, col);
					}
					else {
						if (legacyReadName){
							//Override legacy name
							char runId[6] = {'\0'};
							strncpy (runId, &rh->name->s[7], 5);
							fprintf (fpq, "@%s:%d:%d\n", runId, row, col);
						}
						else {
							//Copy name verbatim
							fprintf (fpq, "@%s\n", rh->name->s);
						}
					}
						
					//print bases
					for (int b=clip_left_index-1;b<clip_right_index;b++)
						fprintf(fpq, "%c", rr->bases->s[b]);
					fprintf(fpq, "\n");
					//print '+'
					fprintf(fpq, "+\n");
					//print quality scores
					for (int b=clip_left_index-1;b<clip_right_index;b++)
						fprintf(fpq, "%c", QualToFastQ((int)(rr->quality->s[b]),qual_offset));
					fprintf(fpq, "\n");
				}
			}
			else {
				int f;
				for(f=0;f<sff_file_in->header->flow_length;f++)
					printf("%d ", rr->flowgram[f]);
				printf("\nFlow index per base:\n");
				unsigned int b;
				for(b=0;b<rh->n_bases;b++)
					printf("%d ", rr->flow_index[b]);
				printf("\nBases called:\n");
				for(b=0;b<rh->n_bases;b++)
					printf("%c", rr->bases->s[b]);
				printf("\nQuality scores:\n");
				for(b=0;b<rh->n_bases;b++)
					printf("%d ", rr->quality->s[b]);
				printf("\nDone with this read\n\n");
			}

            sff_read_header_destroy(rh);
            sff_read_destroy(rr);
		}

		//	debug print - keypass reads written to the fastq file
		if (fpq) {
			fprintf (stdout, "Keypass Reads(%s) = %d\n", hackkey, numKeypassedReads);
			fprintf (stdout, "Total Reads = %d\n", numReads);
			fprintf (stdout, "Percentage = %.2f%%\n", ((float) numKeypassedReads/ (float) numReads) * 100.0);
		}
        sff_fclose(sff_file_in);
		if (fpq)
			fclose(fpq);
	}
	else {
		perror (sffFileName);
		return 1;
	}

	return 0;
}
