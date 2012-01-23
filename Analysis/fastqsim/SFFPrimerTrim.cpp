/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// Adapter Clipper

#include <stdio.h>
#include <inttypes.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <assert.h>
#include <cfloat>

#include "../ByteSwapUtils.h"

#define MAGIC ('.' | ('s' << 8) | ('f' << 16) | ('f' << 24))
#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

struct CommonHeader {
	uint32_t	magic_number;
	char		version[4];
	uint64_t	index_offset;
	uint32_t	index_length;
	uint32_t	number_of_reads;
	uint16_t	header_length;
	uint16_t	key_length;
	uint16_t	number_of_flows_per_read;
	uint8_t		flowgram_format_code;
};

struct ReadHeader {
	uint16_t	read_header_length;
	uint16_t	name_length;
	uint32_t	number_of_bases;
	uint16_t	clip_qual_left;
	uint16_t	clip_qual_right;
	uint16_t	clip_adapter_left;
	uint16_t	clip_adapter_right;
};

int main(int argc, char *argv[])
{
	FILE *inSFF = NULL;
	FILE *outSFF = NULL;
    int n = 0; //number elements read
	char *inFileName = NULL;
	char outFileName[512] = {"\0"};
	int numReads;
	int numFoundSeq = 0;
	int numPassFOM = 0;
	int *corr_histo = NULL;
	//int keyFlows = 8;
	//int adapterFlows = 0;
	bool debugflag = false;
	double fom = 0.85;
	int minPrimerLen = 10;
	// TODO: hardcoded, single primer
	char *primer = NULL;
	
	char		name[256];
	uint16_t	*flowgram_values; // [NUMBER_OF_FLOWS_PER_READ];
	uint8_t		*flow_index_per_base; // * number_of_bases;
	char		*bases; // * number_of_bases;
	uint8_t		*quality_scores; // * number_of_bases;

	char		*flow_chars;
	char		*key_sequence;
	
	// Parse command line arguments
	int argcc = 1;
	while (argcc < argc) {
		if (argv[argcc][0] == '-') {
			switch (argv[argcc][1]) {
				
				case 's':	// define barcode string
					argcc++;
					primer = strdup (argv[argcc]);
				break;
				
				case 'd':	// print debug info
					debugflag = true;
				break;
			
				case 'f':	// goodness of fit threshold
					argcc++;
					fom = atof (argv[argcc]);
				break;
			
				case 'l':	// minimum length of primer to match
					argcc++;
					minPrimerLen = atof (argv[argcc]);
				break;
			
				default:
					fprintf (stderr, "Unknown option %s\n", argv[argcc]);
					exit (1);
				break;
			}
		}
		else {
			inFileName = argv[argcc];
		}
		argcc++;
	}
	
	if (!inFileName) {
		fprintf (stdout, "No input file specified\n");
		fprintf (stdout, "Usage: %s [-g #][-l #][-d]\n", argv[0]);
		fprintf (stdout, "\t-s Specify primer string (CTGAGACTGCCAAGGCACACAGGGGATAGG).\n");
		fprintf (stdout, "\t-f Specify acceptance threshold (0.85).\n");
		fprintf (stdout, "\t-l Specify minimum length of primer to match (10).\n");
		fprintf (stdout, "\t-d Prints debug information.\n");
		exit (1);
	}
	
	// No primer passed in from command line so set up a default
	if (primer == NULL) {
		primer = strdup ("CTGAGACTGCCAAGGCACACAGGGGATAGG");
	}
	
	if (corr_histo)
		free (corr_histo);
	corr_histo = (int *) malloc (sizeof(int) * (strlen(primer)+1));
	memset(corr_histo,0,sizeof(int)*(strlen(primer)+1));
	
	//Create output filename from input filename
	snprintf (outFileName, 512, "%s/PT_%s", dirname(inFileName), inFileName);
	
	//Open the SFF file
	inSFF = fopen(inFileName, "rb");
	if (!inSFF) {
		perror (inFileName);
		exit (1);
	}
	//Open the outputSFF file
	outSFF = fopen(outFileName, "wb");
	if (!outSFF) {
		perror (outFileName);
		exit (1);
	}
	
	// Read the input file header
	CommonHeader h;
	n = fread(&h, 31, 1, inSFF);
    assert(n==1);
	
	//Copy the header to write the output file
	CommonHeader ch_out;
	ch_out.magic_number = h.magic_number;
	ch_out.version[0] = 0;
	ch_out.version[1] = 0;
	ch_out.version[2] = 0;
	ch_out.version[3] = 1;
	ch_out.index_offset = h.index_offset;
	ch_out.index_length = h.index_length;
	ch_out.number_of_reads = h.number_of_reads;
	ch_out.header_length = h.header_length;
	ch_out.key_length = h.key_length;
	ch_out.number_of_flows_per_read = h.number_of_flows_per_read;
	
	ByteSwap8(h.index_offset);
	ByteSwap4(h.index_length);
	ByteSwap4(h.number_of_reads);
	ByteSwap2(h.header_length);
	ByteSwap2(h.key_length);
	ByteSwap2(h.number_of_flows_per_read);
	flow_chars = (char *)malloc(h.number_of_flows_per_read);
	key_sequence = (char *)malloc(h.key_length);
	n = fread(flow_chars, h.number_of_flows_per_read, 1, inSFF);
    assert(n==1);
	n = fread(key_sequence, h.key_length, 1, inSFF);
    assert(n==1);
	int padBytes = (8-((31 + h.number_of_flows_per_read + h.key_length) & 7));
	char padData[8];
	if (padBytes > 0) {
		n = fread(padData, padBytes, 1, inSFF);
		assert(n==1);
	}
	
	if (true) {
		//DEBUG
		printf("Magic:	%u	%s\n", h.magic_number, (h.magic_number == MAGIC ? "Yes" : "No"));
		printf("Header length: %hu\n", h.header_length);
		printf("Version: %d%d%d%d\n", h.version[0], h.version[1], h.version[2], h.version[3]);
		printf("Index offset: %lu  length: %u\n", h.index_offset, h.index_length);
		printf("Number of reads: %u\n", h.number_of_reads);
		printf("Key length: %u\n", h.key_length);
		printf("Flows per read: %hu\n", h.number_of_flows_per_read);
		printf("Flowgram format: %hhu\n", h.flowgram_format_code);
		printf ("End of Header\n\n");
	}
		
	// Write the header of the output SFF
	char pad[8];
	memset(pad, 0, sizeof(pad));
	int bytes = 31;
	fwrite (&ch_out, bytes, 1, outSFF);
	for(int i=0;i<h.number_of_flows_per_read;i++) {
		fwrite(&flow_chars[i%4], 1, 1, outSFF);
		bytes++;
	}

	fwrite(key_sequence, 1, 4, outSFF);
	bytes += 4;

	padBytes = (8 - (bytes & 0x7)) & 0x7;
	if (padBytes > 0)
		fwrite(pad, padBytes, 1, outSFF);

	// Prepare to process all the reads
	numReads = h.number_of_reads;
	
	flowgram_values = (uint16_t *)malloc(sizeof(uint16_t) * h.number_of_flows_per_read);
	int maxBases = h.number_of_flows_per_read * 100; // problems if ever a 10-mer hits every flow!
	flow_index_per_base = (uint8_t *)malloc(sizeof(uint8_t) * maxBases);
	bases = (char *)malloc(maxBases);
	quality_scores = (uint8_t *)malloc(sizeof(uint8_t) * maxBases);
	
	//Loop thru the reads
	for (int i=0;i<numReads;i++) {
	//for (int i=0;i<10;i++) {
		// Read read header
		ReadHeader r;
		n = fread(&r, 16,  1, inSFF);
        assert(n==1);

		ByteSwap2(r.read_header_length);
		ByteSwap4(r.number_of_bases);
		ByteSwap2(r.name_length);
		ByteSwap2(r.clip_qual_left);
		ByteSwap2(r.clip_qual_right);
		ByteSwap2(r.clip_adapter_left);
		ByteSwap2(r.clip_adapter_right);
		
		if (r.name_length > 0) {
			n = fread(name, r.name_length, 1, inSFF);
			assert(n==1);
            name[r.name_length] = '\0';
		}
		
		int readPadLength = ((8 - ((16 + r.name_length) & 7)))%8;
		if (readPadLength > 0) {
			n = fread(padData, readPadLength, 1, inSFF);
			assert(n==1);
		}

		n = fread(flowgram_values, h.number_of_flows_per_read, sizeof(uint16_t), inSFF);
		assert(n==sizeof(uint16_t));
		n = fread(flow_index_per_base, r.number_of_bases, sizeof(uint8_t), inSFF);
		assert(n==sizeof(uint8_t));
		n = fread(bases, r.number_of_bases, 1, inSFF);
		assert(n==1);
		bases[r.number_of_bases] = '\0';
		n = fread(quality_scores, r.number_of_bases, sizeof(uint8_t), inSFF);
		assert(n==sizeof(uint8_t));

		int bytesRead = h.number_of_flows_per_read * sizeof(uint16_t) + 3 * r.number_of_bases;
		readPadLength = (8 - (bytesRead & 7))%8;
		if (readPadLength > 0) {
			n = fread(padData, readPadLength, 1, inSFF);
			assert(n==1);
		}
		
		int f;
		if (false) {
			// DEBUG
			printf("Read: %s has %d bases\n",
							(r.name_length > 0 ? name : "NONAME"),
							r.number_of_bases);
			//printf("Read header length: %d\n", r.read_header_length);
			printf("Clip left: %d qual: %d right: %d qual: %d\n",
						r.clip_adapter_left, r.clip_qual_left,
						r.clip_adapter_right, r.clip_qual_right);
			printf("Flowgram bases:\n");
			for(f=0;f<h.number_of_flows_per_read;f++)
				printf("%d ", (int) floor (ByteSwap2(flowgram_values[f])/100.0 + 0.5));
			printf("\n");
			//printf("\nFlow index per base:\n");
			unsigned int b;
			//for(b=0;b<r.number_of_bases;b++)
			//	printf("%d ", flow_index_per_base[b]);
			printf("Bases called:\n");
			for(b=0;b<r.number_of_bases;b++)
				printf("%c", bases[b]);
			//printf("\nQuality scores:\n");
			//for(b=0;b<r.number_of_bases;b++)
			//	printf("%d ", quality_scores[b]);
		} else {
			for(f=0;f<h.number_of_flows_per_read;f++)
				ByteSwap2(flowgram_values[f]);
		}
		
		
		int primerStart = 0;
		
		/*
		// First try using brute force and system calls
		// Search for exact match on entire primer string; then start lopping off
		// bases from the end and matching.
		int matchLen = strlen (primer);
		char *ptr = NULL;
		while (matchLen > 7)
		{
			primer[matchLen] = '\0';
			if ((ptr = strstr (bases, primer)) != NULL) {
				// Got a match
				primerStart = ptr - &bases[0];
				fprintf (stdout, "%s\n", name);
				if (debugflag) {
					fprintf (stdout, "Pointer match at %d for length %d (%d) total %d\n", primerStart, matchLen, primerStart + matchLen,r.number_of_bases);
					for(int b=0;b<r.number_of_bases;b++)
						printf("%c", bases[b]);
					printf ("\n");
					for(int b=0;b<strlen(primer);b++)
						printf("%c", primer[b]);
					printf ("\n");
				}
				
				break;
			}
			matchLen--;
		}
		*/
		
		int primerLen = strlen(primer);
		int sPos = 0;
		int correct = 0;
		double *correctness = (double *) malloc (sizeof(double) * r.number_of_bases);
		
	#if 0
		// Loop thru starting positions starting at left going right
		int endPos = r.number_of_bases - primerLen;
		for (int i=sPos;i<endPos;i++) {
			// Loop thru the flows
			for (int flow=0;flow<primerLen;flow++) {
				if (primer[flow] == bases[flow+i]) {
					correct++;
				}
			}
					
			correctness[i] = (double) correct/ (double) primerLen;
			//fprintf (stdout, "%d/%d ", correct, primerLen);
			correct = 0;
		}
		//fprintf (stdout, "\n");
		
		double max;
		int matchingIndex = 0;
		
		for (int i=sPos;i<endPos;i++) {
			if (i == sPos || correctness[i] > max) {
				max = correctness[i];
				matchingIndex = i;
			}
		}
	#else
		// Loop thru starting positions starting at right and going left
		int endPos = r.number_of_bases - minPrimerLen;
		for (int i=endPos;i >= sPos;i--) {
			primerLen = ((r.number_of_bases-i) < (strlen(primer)) ? (r.number_of_bases-i):strlen(primer));
			// Loop thru the flows
			for (int flow=0;flow<primerLen;flow++) {
				if (primer[flow] == bases[flow+i]) {
					correct++;
				}
			}
			
			if (correct > 0 && primerLen > 0)
				correctness[i] = (double) correct/ (double) primerLen;
			else
				correctness[i] = 0;
			correct = 0;
		}
		
		double max = DBL_MIN;
		int matchingIndex = 0;
		
		for (int i=0;i<endPos;i++) {
			if (i == 0 || correctness[i] > max) {
				max = correctness[i];
				matchingIndex = i;
			}
		}
	#endif
		numFoundSeq++;
		// Caveat:
		// There is a discrepancy here in the recorded correctness histogram.
		// At the end of the read, the primer length is variable.  but, we only
		// report on statistics for the full primer length at the end of the
		// execution.  So, someof the reads which compare with high percentage
		// might not actually compare over the full lenght of the primer.
		corr_histo[(int)floor(max*strlen(primer))]++;
		
		if (max >= fom) {
			numPassFOM++;
			primerStart = matchingIndex;
			if (debugflag) {
				fprintf (stdout, "%s\n", name);
				fprintf (stdout, "Matching Index = %d (%0.2lf)\n", matchingIndex, max);
				for (int i=0;i<matchingIndex;i++)
					fprintf (stdout, " ");
				fprintf (stdout, "%s\n%s\n", primer, bases);
			}
		}
		free (correctness);

		
		// Set the right side clipping value
		r.clip_qual_right = primerStart;	// the meat of the subject
		//fprintf (stdout, "r.clip_qual_right = %d\n", r.clip_qual_right);
		
		
		/* This is the unique code for barcode clipping from SFFClipper */
		//
		//// Trim the key and the adapter
		//int numFlowsToTrim = keyFlows + adapterFlows;
		//// numBasesToTrim = basesInKey + basesInAdapter;
		//int numBasesToTrim = 0;
		//for (int j=0;j<numFlowsToTrim;j++)
		//	numBasesToTrim += (int) floor (flowgram_values[j]/100.0 + 0.5); //relies on debug print to byteswap!
		//
		//// TODO: make this generic for the final base of the key
		//int k = 4;
		//while (bases[k++] == 'G')
		//	numBasesToTrim -= 1;
		//
		//if (debugflag)
		//	printf("\nNumBases = %d\n", numBasesToTrim);
		//
		//
		//r.clip_qual_left = numBasesToTrim + 1;	//The essence of this tool.
		//
		/* End of unique code for barcode clipping */
		
		
		
		
		//
		//	Update the output file
		//
		int nameLen = r.name_length;
		int numBasesCalled = r.number_of_bases;
		
		// write the header
		ByteSwap2(r.read_header_length);
		ByteSwap4(r.number_of_bases);
		ByteSwap2(r.name_length);
		ByteSwap2(r.clip_qual_left);
		ByteSwap2(r.clip_qual_right);
		ByteSwap2(r.clip_adapter_left);
		ByteSwap2(r.clip_adapter_right);
		fwrite (&r, 16, 1, outSFF);
		
		fwrite(name, nameLen, 1, outSFF);
		int writePadLength = (8 - (nameLen & 7)) & 7;
		if (writePadLength)
			fwrite(padData, writePadLength, 1, outSFF);

		for (f=0;f<h.number_of_flows_per_read;f++)
			ByteSwap2(flowgram_values[f]);
		fwrite(flowgram_values, h.number_of_flows_per_read, sizeof(uint16_t), outSFF);

		fwrite(flow_index_per_base, numBasesCalled, sizeof(uint8_t), outSFF);

		fwrite(bases, numBasesCalled, 1, outSFF);

		fwrite(quality_scores, numBasesCalled, sizeof(uint8_t), outSFF);

		int bytesWritten = h.number_of_flows_per_read * sizeof(uint16_t) + 3 * numBasesCalled;
		writePadLength = (8 - (bytesWritten & 7)) & 7;
		if (writePadLength)
			fwrite(padData, writePadLength, 1, outSFF);
	}
	
	// Print statistics to stdout
	fprintf (stdout, "\n=====================================================\n");
	fprintf (stdout, "Primer String: %s\n", primer);
	fprintf (stdout, "%15s: %10d\n", "Total Reads", numReads);
	//fprintf (stdout, "%15s: %10d\n", "with Adapter", numFoundSeq);
	fprintf (stdout, "%15s: %10d\n", "passing FOM", numPassFOM);
	fprintf (stdout, "Acceptance threshold: %0.2lf%%\n", fom);
	/*fprintf (stdout, "Search indices: %d thru %d\n", sPos, endPos-1);*/
	for (int i = 0; i <= (int)strlen(primer);i++) {
		fprintf (stdout, "[%2d/%2d %6.1lf%%] %5d/%d %6.2lf%%\n", i, (int) strlen(primer), ((double)i/(double)strlen(primer))*100.0,corr_histo[i], numFoundSeq, ((double)corr_histo[i]/(double)numFoundSeq)*100.0);
		//fprintf (stdout, "%d\n", corr_histo[i]);
	}
	
	//Cleanup
	fclose (inSFF);
	fclose (outSFF);
	free (flow_index_per_base);
	free (bases);
	free (quality_scores);
	free (flowgram_values);
	free (flow_chars);
	free (corr_histo);
	free (primer);
	free (key_sequence);
	
	return 0;
}
