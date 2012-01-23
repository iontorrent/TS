/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// SFF signal extract tool for sam files
// (c) 2009
// $Rev: $
//      $Date: $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

#include "SFFWrapper.h"
#include "Mask.h"
#include "LinuxCompat.h"
#include "file-io/ion_util.h"
#include "Histogram.h"
#include "Utils.h"

char *flowOrder = (char *)"TACG";

int GenerateIonogram(const char *_seq, int *ionogram)
{
	char seq[512];

	unsigned int i;
	int len = 0;
	for(i=0;i<strlen(_seq);i++) {
		if (seq[i] != '-') {
			seq[len] = _seq[i];
			len++;
		}
	}

        int flows = 0;
        int bases = 0;
        while (flows < 800 && bases < len) {
                ionogram[flows] = 0;
                while (flowOrder[flows%4] == seq[bases] && bases < len) {
                        ionogram[flows]++;
                        bases++;
                }
                flows++;
        }
        return flows;
}

int main(int argc, char *argv[])
{
	// set some defaults
	int maxFlows = 100;
	bool reportErrors = true;
	int verbose = 0;
	char *sffFile = (char *)"rawlib.sff";
	char *resultsPath = (char *)".";
	int whichQ = 0; // 0 is Q7, 1 is Q10, 2 is Q17, 3 is Q20, 4 is Q47
	int minQLen = 21;
	bool showBest = false;

	double *values[8];
	int vnum[8];
	memset(vnum, 0, sizeof(vnum));

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'n': // num flows
				argcc++;
				maxFlows = atoi(argv[argcc]);
			break;

			case 'e': // ignore errors
				reportErrors = false;
			break;

			case 's':
				argcc++;
				sffFile = argv[argcc];
			break;

                        case 'r':
                                argcc++;
                                resultsPath = argv[argcc];
                        break;

			case 'q':
				argcc++;
				whichQ = atoi(argv[argcc]);
				argcc++;
				minQLen = atoi(argv[argcc]);
			break;

			case 'v':
				verbose++;	
			break;

			case 'b':
				showBest = true;
			break;
		}
		argcc++;
	}

	// load up our mask file
	Mask mask(1,1);
	char maskPath[MAX_PATH_LENGTH];
	sprintf(maskPath, "%s/bfmask.bin", resultsPath);
	mask.SetMask(maskPath);

	// load up our SFF file
	SFFWrapper sff;
	sff.OpenForRead("", sffFile);
	if (maxFlows > sff.GetHeader()->flow_length)
		maxFlows = sff.GetHeader()->flow_length;
	if (maxFlows == 0)
		maxFlows = sff.GetHeader()->flow_length;

	unsigned int i;
        for(i=0;i<8;i++) {
                values[i] = new double[100000];
        }


	// parse the SAM file and pull out the row/col values we want to process
	char samFile[MAX_PATH_LENGTH];
	sprintf(samFile, "%s/Default.sam.parsed", resultsPath);
	unsigned long *chip = new unsigned long[mask.W() * mask.H()];
	memset(chip, 0, sizeof(unsigned long) * mask.W() * mask.H());
	FILE *fp = fopen(samFile, "r");
	char buf[16384];
	int numReads = 0;

	int paramCol[8]; // we are looking for 4 column headers, but not sure what columns they are located in, this array will store those column indexes
	for(int i=0;i<8;i++)
		paramCol[i] = -1;

	if (fp) {
		// read header and process to learn the column indexed we care about
		if (fgets(buf, sizeof(buf), fp)) {
			int buflen = strlen(buf);
			if (buflen > 0)
				buf[buflen-1] = 0; // remove the return line
			char *ptr = strtok(buf, "\t");
			int heading = 0;
			while (ptr) {
				if (strcmp(ptr, "q7Len") == 0)
					paramCol[0] = heading;
				if (strcmp(ptr, "q10Len") == 0)
					paramCol[1] = heading;
				if (strcmp(ptr, "q17Len") == 0)
					paramCol[2] = heading;
				if (strcmp(ptr, "q20Len") == 0 )
					paramCol[3] = heading;
				if (strcmp(ptr, "q47Len") == 0 )
					paramCol[4] = heading;
				if (strcmp(ptr, "qDNA.a") == 0)
					paramCol[5] = heading;
				if (strcmp(ptr, "match.a") == 0)
					paramCol[6] = heading;
				if (strcmp(ptr, "tDNA.a") == 0)
					paramCol[7] = heading;

				ptr = strtok(NULL, "\t");
				heading++;
			}
		}

		if (verbose > 0) {
			for(int i=0;i<8;i++) {
				printf("Param %d is at header %d\n", i, paramCol[i]);
			}
		}

		int line = 1;
		long curpos = ftell(fp);
		while (fgets(buf, sizeof(buf), fp)) {
			// get the row/col
			// Read Name Format of: VWXYZ:<row>:<column>
			//     VWXYZ is runId; row is y axis position; column is x axis position
			int row = 0, col = 0;
			int st = sscanf (buf, "%*5c:%d:%d", &row, &col);
			if (st != 2) {
				// fprintf (stderr, "Error parsing read name: '%s'\n", buf);
				// continue;
				st = sscanf (buf, "r%d|c%d", &row, &col);
				if (st != 2) {
					fprintf (stderr, "Error parsing read name: '%s'\n", buf);
					continue;
				}
			}

			// get the 5 params
			char *ptr = strtok(buf, "\t");
			int heading = 0;
			int q[5] = {0};
			while (ptr) {
				for(int param=0;param<5;param++) { // only looking for our 5 Q score headings here
					if (paramCol[param] == heading)
						q[param] = atoi(ptr);
				}
				ptr = strtok(NULL, "\t");
				heading++;
			}

			// test to see if our desired param exceeds the min Q length
			if (q[whichQ] >= minQLen) {
				// chip[col+row*mask.W()] = line; // store the line so we can get back to it fast
				chip[col+row*mask.W()] = curpos; // store the file position so we can get back to it fast
				numReads++;
			}
			line++;
			curpos = ftell(fp);
		}
		fclose(fp);
	} else {
		printf("SAM file not found?\n");
	}

	if (verbose)
		printf("Processing %d reads\n", numReads);

	if (numReads == 0) {
		printf("No reads found matching input criteria of q: %d len: %d\n", whichQ, minQLen);
		exit(0); // nothing to process
	}

	// loop through the sam file, for reads that pass our test, add Ionogram to signal stats

	// we will also evaluate the quality of each read and dump out the best read based on RMSE of 1-mers
	int bestRow = -1, bestCol = -1;
	double bestRMSE = -1.0;

	const sff_t *data;
	int x, y;
	int flow;
	fp = fopen(samFile, "r");
	for(i=0;i<sff.GetHeader()->n_reads;i++) {
		if (verbose)
		if ((i % (sff.GetHeader()->n_reads/100)) == 0) {
			printf(".");
			fflush(stdout);
		}

		data = sff.LoadEntry(i);
                if(1 != ion_readname_to_xy(sff_name(data), &x, &y)) {
                    fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(data));
                    continue;
		}
		// see if this X-Y is one we care about
		bool found = false;
		if (chip[x+y*mask.W()] > 0) {
				/*
				unsigned int li;
				for(li=0;li<chip[x+y*mask.W()];li++)
					assert(fgets(buf, sizeof(buf), fp)); // skip this line
				*/
				fseek(fp, chip[x+y*mask.W()], SEEK_SET);
				assert(fgets(buf, sizeof(buf), fp)); // reads this line
				int row, col;
				int st = sscanf (buf, "%*5c:%d:%d", &row, &col);
				if (st ==2 && row == y && col == x) {
					found = true;

					// evaluate how good the read is
					int base;
					double baseVal, rmse = 0.0;
					int rmseCount = 0;
					for(flow=0;flow<maxFlows;flow++) {
						baseVal = sff_flowgram(data)[flow] / 100.0;
						base = (int)(baseVal + 0.5);
						if (base == 1 || base == 2 || base == 3) {
							double deltaBase = baseVal - base;
							rmse += deltaBase*deltaBase;
							rmseCount++;
						}
					}
					if (rmseCount > 0) {
						rmse /= rmseCount;
						if (bestRMSE == -1.0 || rmse < bestRMSE) {
							bestRMSE = rmse;
							bestRow = row;
							bestCol = col;
						}
					}
				}
		}

		if (found) {

			char *ptr = strtok(buf, "\t");
			char *ref = NULL, *seq = NULL, *bars = NULL;
			int heading = 0;
			int q[5] = {0};
			while (ptr) {
				for(int param=0;param<5;param++) { // only looking for our 5 Q score headings here
					if (paramCol[param] == heading)
						q[param] = atoi(ptr);
				}
				if (paramCol[5] == heading)
					seq = ptr;
				if (paramCol[6] == heading)
					bars = ptr;
				if (paramCol[7] == heading)
					ref = ptr;
				ptr = strtok(NULL, "\t");
				heading++;
			}
				
			if (verbose > 1) {
				printf("r%d|c%d %d %d %d %d %d\n", y, x, q[0], q[1], q[2], q[3], q[4]);
				printf("Seq: %s\nBar: %s\nRef: %s\n", seq, bars, ref);
			}

				int flowOffset = 8;
				int firstbase = (int)(sff_flowgram(data)[7]/100.0+0.5);
				if (firstbase > 1)
					flowOffset = 4;
				bool foundN = (strchr(ref, 'N') != NULL ? true : false);
				// grab Ionogram values and add to histogram
				int totalBases = 0;
				int base2;
				if (!foundN)
				for(flow=0;flow<maxFlows;flow++) {
					// base - this is the number of bases we thought the flow was
					// val - this is the measured & corrected signal we ended up with
					// base2 - this is the true base count for the aligned read at this flow
					// Calculating can't be done in flowspace since missing a base (reporting 0-mer) can offset the conversion to flowspace by 4
					// So, comparisions are done in read space, driven by the homopolymer calls
					// Examples with TACG flow order:
					//   TCCT
					//   TC-T
					//   we measured 1.1, 0.1, 1.6, 0.2, 0.9 and called as 10201 or TCCT in flowspace
					//   but we see that our 1.6 belongs in the 1-mer bin, not the 2-mer bin, so in order to find the correct bin, only exact matches are used
					// Example 2:
					//   TGA
					//   T-A
					//   here, we have over-called the 0-mer G as a 1-mer G.  Converting to flowspace on each read yields:
					//     1001100 for our read, and
					//     1100 for the reference, so it would be bad to compare these two in flow space, as a shift of 4 flows has occured

					double val = sff_flowgram(data)[flow]/100.0;
					int base = (int)(val+0.5);
					if (base < 0) base = 0;
					if (flow < 8) // key not included in reads, so take right from SFF (fine since they are assumed to have been keypassed)
						base2 = base;
					else {
						base2 = 0;
						int k;
						// overcalls only add to base2 the number of true matches in the ref, so base2 is lower
						for(k=0;k<base;k++) {
							if (seq[totalBases+k] == ref[totalBases+k])
								base2++;
						}
						totalBases += base;
						// undercalls increase base2 to the true number of bases we should have called
						while(seq[totalBases] == '-') {
							base2++;
							totalBases++;
						}
					}
					bool isErr = (base2 != base);
					bool skipThisOne = false;
					if (!reportErrors && isErr) 
						skipThisOne = true;
					if (base2 < 8 && !skipThisOne) {
						if (vnum[base2] < 100000) {
							values[base2][vnum[base2]] = val;
							vnum[base2]++;
						}

						if (verbose > 2) {
							if (isErr) {
								printf("Called a %d-mer as a %d-mer at flow %d\n", base2, base, flow);
								printf("Seq: %s\nRef: %s\n", seq, ref);
							}
						}
					}
				}
		}
	}
        sff.Close();
	fclose(fp);

	for(i=0;i<8;i++) {
		int num;
		printf("%d = ", i);
		for(num=0;num<vnum[i];num++)
			printf("%.5lf ", values[i][num]);
		printf("\n");
		delete [] values[i];
	}

	if (showBest)
		printf("Best read at %d,%d with %.5lf\n", bestRow, bestCol, bestRMSE);

	delete [] chip;
}

