/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// SFF query tool - looks for 'interesting' reads
// (c) 2010
// $Rev: $
//      $Date: $
//

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

#include "SFFWrapper.h"
#include "LinuxCompat.h"
#include "file-io/ion_util.h"
#include "Utils.h"

struct TopWell {
	int	x;
	int	y;
	char	*seq;
	char	*bars;
	char	*ref;
	char	*ionogram;
	double	metric;
};

struct ListItem {
	int lineNumber;
	int topNumber;
};

int sortMyList(const void *a, const void *b)
{
	ListItem *la = (ListItem *)a;
	ListItem *lb = (ListItem *)b;
	return la->lineNumber - lb->lineNumber;
}

int main(int argc, char *argv[])
{
	// set some defaults
	int verbose = 0;
	char *sffFile = (char *)"rawlib.sff";
	char *resultsPath = (char *)".";

	// default to only consider 100+bp perfect reads
	int minQLen = 100;
	int whichQ = 4; // 0=Q7, 1=Q10, 2=Q17, 3=Q20, 4=Q47

	int topCount = 100; // default to outputing the top 100
	int maxFlow = 200; // only look at the first 200 flows for now - 0 will allow use of all flows in an sff file

	int topType = 1; // 1 is top N pretty, 2 is top 'whichQ' (so default is perfect)

	// process cmd line args
	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
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

			case 'n':
				argcc++;
				topCount = atoi(argv[argcc]);
			break;

			case 'm':
				argcc++;
				maxFlow = atoi(argv[argcc]);
			break;

			case 't':
				argcc++;
				topType = atoi(argv[argcc]);
			break;

			case 'v':
				verbose++;	
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
	sff.OpenForRead(resultsPath, sffFile);

	// parse the SAM file and pull out the row/col values we want to process
	char samFile[MAX_PATH_LENGTH];
	sprintf(samFile, "%s/Default.sam.parsed", resultsPath);
	unsigned int *readMetric = new unsigned int[mask.W() * mask.H()];
	unsigned int *readLine = new unsigned int[mask.W() * mask.H()];
	memset(readMetric, 0, sizeof(int) * mask.W() * mask.H());
	memset(readLine, 0, sizeof(int) * mask.W() * mask.H());
	FILE *fp = fopen(samFile, "r");
	char buf[16384];
	int numReads = 0;

	int paramCol[8]; // we are looking for column headers, but not sure what columns they are located in, this array will store those column indexes
	int i;
	for(i=0;i<8;i++)
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
			char *bars;
			while (ptr) {
				for(int param=0;param<5;param++) { // only looking for our 5 Q score headings here
					if (paramCol[param] == heading)
						q[param] = atoi(ptr);
				}
				if (paramCol[6] == heading)
					bars = ptr;
				ptr = strtok(NULL, "\t");
				heading++;
			}

			// test to see if our desired param exceeds the min Q length
			if (q[whichQ] >= minQLen) {
				readMetric[col+row*mask.W()] = q[whichQ]; // store the metric found for this read
				readLine[col+row*mask.W()] = line; // store the line in the SAM file that we found this read at in case we need to get back to it fast
				numReads++;
			}
			line++;
		}
		fclose(fp);
	}

	if (verbose)
		printf("Processing %d reads\n", numReads);

	// we will keep a list of the top N reads here in the topWell var
	// we will insert new entries into the list such that the list is always sorted with the best (lowest metric) as the first element
	TopWell topWell[topCount];
	memset(topWell, 0, sizeof(TopWell) * topCount);
	for(i=0;i<topCount;i++) {
		topWell[i].metric = 9999999999.0;
		topWell[i].ionogram = NULL;
	}
	int numTop = 0;

	if (verbose > 0)
		printf("SFF file entries each contain %d flows.\n", sff.GetHeader()->flow_length);

	if (maxFlow > sff.GetHeader()->flow_length || maxFlow == 0)
		maxFlow = sff.GetHeader()->flow_length;

	char *ionogram;
	ionogram = new char[sff.GetHeader()->flow_length * 6 + 2]; // want to make sure the text array can store the 5 digit numbers plus a space between each plus the NULL terminator

	const sff_t *data;
	int x, y;
	for(i=0;i<(int)sff.GetHeader()->n_reads;i++) {
		if (verbose) {
			if ((i % (sff.GetHeader()->n_reads/100)) == 0) {
				printf(".");
				fflush(stdout);
			}
		}

		data = sff.LoadEntry(i);
                if(1 != ion_readname_to_xy(sff_name(data), &x, &y)) {
                    fprintf (stderr, "Error parsing read name: '%s'\n", sff_name(data));
                    continue;
		}
		// see if this X-Y is one we care about
		if (readMetric[x+y*mask.W()] > 0) {
			double metric = 0.0;
			int flow;
			double baseVal;
			if (topType == 1) {
				double rmse = 0.0;
				int rmseCount = 0;
				int base;
				int testMax = 200;
				if (maxFlow < testMax)
					testMax = maxFlow;
				for(flow=0;flow<testMax;flow++) {
					baseVal = sff_flowgram(data)[flow] / 100.0;
					base = (int)(baseVal + 0.5);
					if (base == 1 || base == 2 || base == 3) { // we only care about how the 1-mers thru 3-mers look
						double deltaBase = baseVal - base;
						rmse += deltaBase*deltaBase;
						rmseCount++;
					}
				}
				if (rmseCount > 0)
					metric = rmse/rmseCount;
			} else if (topType == 2) {
				metric = 10000 - readMetric[x+y*mask.W()]; // sort of a hack since we minimize the metric when sorting and such
			}

			// find the location where we need to insert
			int j;
			for(j=0;j<numTop;j++) {
				if (metric < topWell[j].metric) {
					break;
				}
			}
			// j contains the insert point, shift the rest of the entries down, and add the new one at 'j'
			if (j < topCount) {
				int k;
				int startNumForShift = numTop;
				if (startNumForShift >= topCount)
					startNumForShift--; // don't want to write past the end of our array!
				for(k=numTop;k>j;k--) {
					topWell[k] = topWell[k-1];
				}
				topWell[j].x = x;
				topWell[j].y = y;
				topWell[j].metric = metric;
				// if (topWell[j].ionogram)
					// free(topWell[j].ionogram);
				ionogram[0] = NULL;
				// for(flow=0;flow<sff.GetHeader()->flow_length;flow++) {
				for(flow=0;flow<maxFlow;flow++) {
					char buf[12];
					baseVal = sff_flowgram(data)[flow] / 100.0;
					if (baseVal > 29.999)
						baseVal = 29.999;
					sprintf(buf, "%.2lf ", baseVal);
					strcat(ionogram, buf);
				}
				topWell[j].ionogram = strdup(ionogram);
				if (numTop < topCount)
					numTop++;
			}
		}
	}
        sff.Close();

	// go back to the Default.sam.parsed file and grab the sequence info for the top 100 reads
	// so we only go through this list once, make a new list of the lines we found reads at, and sort the list by those line numbers
	ListItem list[numTop];
	for(int lineN=0;lineN<numTop;lineN++) {
		list[lineN].lineNumber = readLine[topWell[lineN].x + topWell[lineN].y*mask.W()];
		list[lineN].topNumber=lineN;
	}
	qsort(list, numTop, sizeof(ListItem), sortMyList);

	int next = 0;
	int last = -1; // neg 1 to force header row skip too
	fp = fopen(samFile, "r");
	if (fp) {
		for(next=0;next<numTop;next++) {
			int skipThisMany = list[next].lineNumber - last;
			last = list[next].lineNumber;
			int li;
			for(li=0;li<skipThisMany;li++)
				assert(fgets(buf, sizeof(buf), fp)); // skip this line

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

			int len = q[whichQ]; // MGD - not sure if this is correct, but intent is to only show length of read out to the Q score
			seq[len] = 0;
			bars[len] = 0;
			ref[len] = 0;
			topWell[list[next].topNumber].seq = strdup(seq);
			topWell[list[next].topNumber].bars = strdup(bars);
			topWell[list[next].topNumber].ref = strdup(ref);
		}

		fclose(fp);
	}

	// dump the query info for the top 100 reads
	printf("Top pretty reads:\n");
	for(i=0;i<numTop;i++) {
		// I may want to only dump seq, bars, ref up to the first error, so only show perfect portion
		printf("X: %d Y: %d RMSE: %.5lf\n", topWell[i].x, topWell[i].y, topWell[i].metric);
		printf("%s\n", topWell[i].seq);
		printf("%s\n", topWell[i].bars);
		printf("%s\n", topWell[i].ref);
		printf("%s\n", topWell[i].ionogram);
	}

	delete [] readMetric;
	delete [] readLine;
	delete [] ionogram;
}

