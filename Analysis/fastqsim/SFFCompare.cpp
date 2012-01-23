/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// compares two sff files
// currently compares number of reads in each and then the flowgram values
// for matching row column coordinates.
/* To compile:
g++ -g -Wall -o SFFCompare SFFCompare.cpp ../file-io/ion_error.c ../file-io/ion_util.c
*/

#include <stdio.h>
#include <inttypes.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include "../file-io/file_util.h"

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

	char		*flow_chars;
	char		*key_sequence;

struct ReadHeader {
	uint16_t	read_header_length;
	uint16_t	name_length;
	uint32_t	number_of_bases;
	uint16_t	clip_qual_left;
	uint16_t	clip_qual_right;
	uint16_t	clip_adapter_left;
	uint16_t	clip_adapter_right;
};


bool readCommonHeader (FILE *fp, CommonHeader *h)
{
	//CommonHeader h;
	fread(h, 31, 1, fp);
	ByteSwap8(h->index_offset);
	ByteSwap4(h->index_length);
	ByteSwap4(h->number_of_reads);
	ByteSwap2(h->header_length);
	ByteSwap2(h->key_length);
	ByteSwap2(h->number_of_flows_per_read);
	if (false) {
		printf("Magic:	%u	%s\n", h->magic_number, (h->magic_number == MAGIC ? "Yes" : "No"));
		printf("Header length: %hu\n", h->header_length);
		printf("Version: %d%d%d%d\n", h->version[0], h->version[1], h->version[2], h->version[3]);
		printf("Index offset: %lu  length: %u\n", h->index_offset, h->index_length);
		printf("Number of reads: %u\n", h->number_of_reads);
		printf("Key length: %u\n", h->key_length);
		printf("Flows per read: %hu\n", h->number_of_flows_per_read);
		printf("Flowgram format: %hhu\n", h->flowgram_format_code);
	}
	char *flow_chars = (char *)malloc(h->number_of_flows_per_read);
	char *key_sequence = (char *)malloc(h->key_length);
	fread(flow_chars, h->number_of_flows_per_read, 1, fp);
	fread(key_sequence, h->key_length, 1, fp);
	int i;
	if (false) {
		printf("Key sequence: ");
		for(i=0;i<h->key_length;i++)
			printf("%c", key_sequence[i]);

		printf("\nFlow chars:\n");
		for(i=0;i<h->number_of_flows_per_read;i++)
			printf("%c", flow_chars[i]);
		printf("\n");
	}
	int padBytes = (8-((31 + h->number_of_flows_per_read + h->key_length) & 7));
	char padData[8];
	fread(padData, padBytes, 1, fp);
	
	free (flow_chars);
	free (key_sequence);
	return (false);
}

bool readNextRead (FILE *fp, ReadHeader *r, uint16_t *flowgram_values, int numFlows, int keyLength, char *name)
{
	char padData[8];
	
	//char		name[256];
	//uint16_t	*flowgram_values; // [NUMBER_OF_FLOWS_PER_READ];
	uint8_t		*flow_index_per_base; // * number_of_bases;
	char		*bases; // * number_of_bases;
	uint8_t		*quality_scores; // * number_of_bases;
	
	//*flowgram_values = (uint16_t *)malloc(sizeof(uint16_t) * numFlows);
	int maxBases = numFlows * 100; // problems if ever a 10-mer hits every flow!
	quality_scores = (uint8_t *)malloc(sizeof(uint8_t) * maxBases);
	flow_index_per_base = (uint8_t *)malloc(sizeof(uint8_t) * maxBases);
	bases = (char *)malloc(maxBases);
	
	//ReadHeader r;
	if (fread(r, 16,  1, fp) != 1) {
		fprintf (stderr, "Error reading read header\n");
	}

	ByteSwap2(r->read_header_length);
	ByteSwap4(r->number_of_bases);
	ByteSwap2(r->name_length);
	ByteSwap2(r->clip_qual_left);
	ByteSwap2(r->clip_qual_right);
	ByteSwap2(r->clip_adapter_left);
	ByteSwap2(r->clip_adapter_right);

	if (false) {
		printf("Read header length: %d\n", r->read_header_length);
		printf("Read name length: %d\n", r->name_length);
	}

	if (r->name_length > 0) {
		fread(name, r->name_length, 1, fp);
		name[r->name_length] = 0; // so we can easily print it
	}
	
	int readPadLength = ((8 - ((16 + r->name_length) & 7)))%8;
	fread(padData, readPadLength, 1, fp);
	
	fread(flowgram_values, numFlows, sizeof(uint16_t), fp);
	fread(flow_index_per_base, r->number_of_bases, sizeof(uint8_t), fp);
	fread(bases, r->number_of_bases, 1, fp);
	fread(quality_scores, r->number_of_bases, sizeof(uint8_t), fp);

	int bytesRead = numFlows * sizeof(uint16_t) + 3 * r->number_of_bases;
	readPadLength = (8 - (bytesRead & 7))%8;
	fread(padData, readPadLength, 1, fp);
	
	free (quality_scores);
	free (flow_index_per_base);
	//free (flowgram_values);
	free (bases);
	
	return (false);
}

bool getCoordinates (ReadHeader *r, char *name, int *X, int *Y)
{
  if(1 != ion_readname_to_xy(name, X, Y)) {
      fprintf (stderr, "Error parsing read name: '%s'\n", name);
      continue;
  }
  return (false);
}

bool readThisRead (FILE *fp, int x, int y, ReadHeader *rh, uint16_t *fgValues)
{
	//fprintf (stdout, "Looking for %d %d\n", x, y);
	//Brute force, start from beginning and read every read, comparing row col
	rewind (fp);
	CommonHeader h;
	readCommonHeader (fp, &h);
	char name[256];
	
	//loop through each read and compare the row col.  stop on match
	for (unsigned int i=0;i<h.number_of_reads;i++) {
		
		memset(rh, 0, sizeof(ReadHeader));
		
		//read next read in first file
		readNextRead (fp, rh, fgValues,
					  h.number_of_flows_per_read,
					  h.key_length,
					  name);
		int X1=0, Y1=0;
		getCoordinates (rh, name, &X1, &Y1);
		
		if (X1 == x && Y1 == y) {
			return (true);
		}
		
	}
	return (false);
}

int main(int argc, char *argv[])
{
	char	*sff1FileName = NULL;
	char	*sff2FileName = NULL;

    sff1FileName = argv[1];
    sff2FileName = argv[2];
    
	if (!sff1FileName || !sff2FileName) {
		printf("Usage: SFFCompare <first sff> <second sff>\n");
		exit(1);
	}

	FILE *fp1 = NULL;
	FILE *fp2 = NULL;
	fp1 = fopen(sff1FileName, "rb");
    fp2 = fopen(sff2FileName, "rb");
    
    if (fp1 == NULL || fp2 == NULL) {
        fprintf (stderr, "Error reading a file\n");
        exit (1);
    }
    
    //Open first sff and second sff
    //Use first sff for source of reads
    //Read each read in seqeunce from first, search second sff for matching
    //row|column and compare flowgram values.
    CommonHeader h1;
    CommonHeader h2;
	
	readCommonHeader (fp1, &h1);
	readCommonHeader (fp2, &h2);
    
    if (h1.number_of_reads == h2.number_of_reads) {
        fprintf (stdout, "Number of reads match\n");
    }
    else {
        fprintf (stdout, "Number of reads DO NOT match:\n");
        fprintf (stdout, "\t%s has %d reads\n", sff1FileName, h1.number_of_reads);
        fprintf (stdout, "\t%s has %d reads\n", sff2FileName, h2.number_of_reads);
    }
	
	char name[256];
	ReadHeader r1;
	ReadHeader r2;
	int X1, Y1;
	uint16_t *fgValues1 = (uint16_t *) malloc (h1.number_of_flows_per_read * sizeof (uint16_t));
	uint16_t *fgValues2 = (uint16_t *) malloc (h1.number_of_flows_per_read * sizeof (uint16_t));
	
	for (unsigned int i=0;i<h1.number_of_reads;i++) {
		
		memset(&r1, 0, sizeof(ReadHeader));
		memset(&r2, 0, sizeof(ReadHeader));
		
		//read next read in first file
		readNextRead (fp1, &r1, fgValues1,
					  h1.number_of_flows_per_read,
					  h1.key_length,
					  name);
		getCoordinates (&r1, name, &X1, &Y1);
		
		//DEBUG
		//fprintf (stdout, "X = %d Y = %d\n", X1, Y1);
		
		//search for same row|col in second file
		if (readThisRead (fp2, X1, Y1, &r2, fgValues2)) {
			//Coordinates match
			//fprintf (stdout, "\r%d", i);
			//compare flow values
			for (int x=0;x<h1.number_of_flows_per_read;x++) {
				if (fgValues1[x] != fgValues2[x]) {
					fprintf (stdout, "Mismatch:%d\n",i);
					//lines with a difference get printed to stdout
					for (int g=0;g<h1.number_of_flows_per_read;g++) {
						fprintf (stdout, "%d ", fgValues1[g]);
					}
					fprintf (stdout, "\n");
					for (int g=0;g<h1.number_of_flows_per_read;g++) {
						fprintf (stdout, "%d ", fgValues2[g]);
					}
					fprintf (stdout, "\n");
					break;
				}
			}
		}
		else {
			fprintf (stdout, "\n%d No match\n",i);
		}
	}
	
	if (fgValues1)
		free (fgValues1);
	if (fgValues2)
		free (fgValues2);
    fclose(fp1);
    fclose(fp2);
    return(0);
}
