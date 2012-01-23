/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// SFF stats tool
// (c) 2009
// $Rev: $
//      $Date: $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "SFFWrapper.h"
#include "Mask.h"
#include "file-io/ion_util.h"
#include "LinuxCompat.h"

int main(int argc, char *argv[])
{
	// set some defaults
	int startFlow = 9;
	int endFlow = 50;
	int nuc = -1;
	int row = -1, col = -1;

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 's': // start avg on this flow
				argcc++;
				startFlow = atoi(argv[argcc]);
			break;

			case 'e': // end avg on this flow
				argcc++;
				endFlow = atoi(argv[argcc]);
			break;

			case 'n': // just use this nuc for the key
				argcc++;
				nuc = atoi(argv[argcc]);
			break;

			case 'r': //find this row
				argcc++;
				row = atoi(argv[argcc]);
			break;

			case 'c': //find this col
				argcc++;
				col = atoi(argv[argcc]);
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

	// first stat - compare the avg key signal with the signal over some number of non-key flows
	// goal is to check for mono-clonality of the library beads
	// so loop through all reads, if read XY matches our mask for library bead, then process
	unsigned int i;
	const sff_t *data;
	int x = 0;
	int y = 0;
	int beadCount = 0;
	int seqCount = 0;
	int seqZeroCount = 0;
	double keyAvg = 0.0;
	double seqAvg = 0.0;
	double keyZeroAvg = 0.0;
	double seqZeroAvg = 0.0;
	int flow;
	bool ok = true;
	for(i=0;i<sff.GetHeader()->n_reads;i++) {
		data = sff.LoadEntry(i);
                if(1 != ion_readname_to_rowcol(sff_name(data), &row, &col)) {
                    fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(data));
                    continue;
		}
		if (row > -1 && col > -1) {
			if (row == y && col == x)
				ok = true;
			else
				ok = false;
		}
		if (ok && mask.Match(x, y, MaskLib)) {
			beadCount++;
			// get avg in key (not sure if I want to subtract avg 0-mer signal to be fair?)
			double flowAvg = 0.0;
			if (nuc == -1) {
				flowAvg += sff_flowgram(data)[0];
				flowAvg += sff_flowgram(data)[2];
				flowAvg += sff_flowgram(data)[5];
				keyAvg += flowAvg/3.0;
			} else {
				flowAvg = sff_flowgram(data)[nuc];
				keyAvg += flowAvg;
			}

			// get the avg 0-mer
			flowAvg = 0.0;
			if (nuc == -1) {
				flowAvg += sff_flowgram(data)[1];
				flowAvg += sff_flowgram(data)[3];
				flowAvg += sff_flowgram(data)[4];
				flowAvg += sff_flowgram(data)[6];
				keyZeroAvg += flowAvg/4.0;
			} else {
				int nuc2;
				if (nuc < 4) nuc2 = nuc + 4; else nuc2 = nuc- 4;
				flowAvg = sff_flowgram(data)[nuc2];
				keyZeroAvg += flowAvg;
			}

			// get avg for next N flows - counting only 1-mers
			flowAvg = 0.0;
			int count = 0;
			double flowZeroAvg = 0.0;
			int zeroCount = 0;
			for(flow=startFlow;flow<=endFlow;flow++) {
				if ((sff_flowgram(data)[flow] > 49) && (sff_flowgram(data)[flow] < 151)) {
					flowAvg += sff_flowgram(data)[flow];
					count++;
				} else if (sff_flowgram(data)[flow] <= 49) {
					flowZeroAvg += sff_flowgram(data)[flow];
					zeroCount++;
				}
			}
			if (count > 0) {
				seqAvg += flowAvg / (double)count;
				seqCount++;
			}
			if (zeroCount > 0) {
				seqZeroAvg += flowZeroAvg / (double)zeroCount;
				seqZeroCount++;
			}
		} else {
			if (ok && row > -1 && col > -1)
				printf("Error?  This row/col was not marked as a library bead?\n");
		}
	}
	if (beadCount > 0) {
		keyAvg /= beadCount;
		if (seqCount)
			seqAvg /= seqCount;
		keyZeroAvg /= beadCount;
		if (seqZeroCount)
			seqZeroAvg /= seqZeroCount;

		printf("Beads: %d  Key avg: %.4lf Zero avg: %.4lf  Seq beads: %d  Seq avg: %.4lf Zero avg: %.4lf\n",
			beadCount, keyAvg, keyZeroAvg, seqCount, seqAvg, seqZeroAvg);
	} else {
		printf("Error?  This row/col was not found in SFF file?\n");
	}
        sff.Close();
}

