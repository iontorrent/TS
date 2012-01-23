/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// SFF signal extract tool
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

char *flowOrder = "TACG";

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
	int row = -1, col = -1;
	int dumpMode = 2;
	char *rowColFile = "JBOutforMel.txt";
	int maxFlows = 100;
	bool reportErrors = true;
	int verbose = 0;

	double *values[8];
	int vnum[8];
	memset(vnum, 0, sizeof(vnum));

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'r': //find this row
				argcc++;
				row = atoi(argv[argcc]);
			break;

			case 'c': //find this col
				argcc++;
				col = atoi(argv[argcc]);
			break;

			case 'd': //dump mode
				argcc++;
				dumpMode = atoi(argv[argcc]);
			break;

			case 'f': //row col file
				argcc++;
				rowColFile = argv[argcc];
			break;

			case 'n': // num flows
				argcc++;
				maxFlows = atoi(argv[argcc]);
			break;

			case 'e': // ignore errors
				reportErrors = false;
			break;

			case 'v':
				verbose++;	
			break;
		}
		argcc++;
	}

	// load up our mask file
	Mask mask(1,1);
	mask.SetMask("bfmask.bin");

	// load up our SFF file
	SFFWrapper sff;
	sff.OpenForRead(".", "rawlib.sff");

	Histogram *h[8];
	unsigned int i;
	for(i=0;i<8;i++) {
		h[i] = new Histogram(1001, -1.0, 8.0);
		values[i] = new double[100000];
	}


	const sff_t *data;
	int x, y;
	int flow;
	for(i=0;i<sff.GetHeader()->n_reads;i++) {
		data = sff.LoadEntry(i);
		
                if (1 != ion_readname_to_xy(sff_name(data), &x, &y)) {
                    fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(data));
                    continue;
		}
		// see if this X-Y is one we care about
		FILE *fp = fopen(rowColFile, "r");
		char buf[512];
		bool found = false;
		char seq[512];
		char ref[512];
		//int refIonogram[1000];
		while (!found && fgets(buf, sizeof(buf), fp)) {
			int row, col;
			sscanf(buf, "(%d,%d", &row, &col);
			int base2;
			if (row == y && col == x) {
				assert(fgets(buf, sizeof(buf), fp));
				sscanf(buf, "%s", seq);
				assert(fgets(buf, sizeof(buf), fp));
				sscanf(buf, "%s", ref);
				printf("Found row %d col %d\n", row, col);
				// int numFlows = GenerateIonogram(ref, refIonogram);
				int flowOffset = 8;
				int firstbase = (int)(sff_flowgram(data)[7]/100.0+0.5);
				if (firstbase > 1)
					flowOffset = 4;
				bool foundN = (strchr(ref, 'N') != NULL ? true : false);
				// grab Ionogram values and add to histogram
				int totalBases = 0;
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
						h[base2]->Add(val);
						if (vnum[base2] < 100000) {
							values[base2][vnum[base2]] = val;
							vnum[base2]++;
						}

						if (verbose) {
							if (isErr) {
								printf("Called a %d-mer as a %d-mer at flow %d\n", base2, base, flow);
								printf("Seq: %s\nRef: %s\n", seq, ref);
							}
						}
					}
				}
				found = true;
			} else {
				assert(fgets(buf, sizeof(buf), fp));
				assert(fgets(buf, sizeof(buf), fp));
			}
		}
		fclose(fp);
	}
	sff.Close();

	char name[256];
	if (dumpMode < 3)
	for(i=0;i<8;i++) {
		sprintf(name, "Sig_%d.txt", i);
		h[i]->Dump(name, dumpMode);
		delete h[i];
		delete values[i];
	}

	if (dumpMode == 3) {
		for(i=0;i<8;i++) {
			int num;
			for(num=0;num<vnum[i];num++)
				printf("%.5lf ", values[i][num]);
			printf("\n");
		}
	}
}

