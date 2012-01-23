/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 *	Purpose: Take a bead mask and an exclude mask generated from Torrent Explorer
 *	and output a full-chip sized binary Mask file.
 *
 *	Need to know:
 *		- Full chip size
 *		- filename of bead mask
 *		- filename of exclude mask
 *
 *	NOTE!!!
 *		The raw mask file from Torrent Explorer generates comma delimited data.
 *		This must be stripped first!
 *
 *		sed -i '3~1s/,//g' <maskfilename>
 *
 *	will remove all the commas starting with the third line
 *	
 */


#include <stdio.h>
#include <string.h>
#include <getopt.h>	// for getopt_long
#include <stdlib.h>
#include <assert.h>
#include "Mask.h"
#include "Utils.h"

struct ChipType {
	char *label;
	int h;
	int w;
};
static struct ChipType ChipTable[] = {
	{"314",	1152,	1280},
	{"316",	2640,	2736},
	{"324",	1152,	1280},
	{NULL,	0,		0}
};

void Trim(char *buf)
{
	int len = strlen(buf);
	while (len > 0 && (buf[len-1] == '\r' || buf[len-1] == '\n'))
		len--;
	buf[len] = 0;
}

int main(int argc, char *argv[])
{
	char *beadMaskFile = NULL;
	char *excludeMaskFile = NULL;
	struct ChipType *myChipType = NULL;

	int c;
	int option_index = 0;
	static struct option long_options[] =
		{
			{"chiptype",				required_argument,	NULL,	'c'},
			{"beadmask",				required_argument,	NULL,	'b'},
			{"excludemask",				required_argument,	NULL,	'e'},
			{"version",					no_argument,		NULL,	'v'},
			{NULL, 0, NULL, 0}
		};
		
	while ((c = getopt_long (argc, argv, "c:b:e:v", long_options, &option_index)) != -1)
	{
		switch (c)
		{
            case (0):
                if (long_options[option_index].flag != 0)
                    break;
				
				break;
			case ('c'):
				for (int n=0;ChipTable[n].label != NULL;n++) {
					if (strncmp (optarg, ChipTable[n].label,3) == 0) {
						myChipType = &ChipTable[n];
					}
				}
				if (!myChipType) {
					fprintf (stderr, "Error: Unknown chip type '%s'\n", optarg);
					fprintf (stderr, "Valid chip types are:\n");
					for (int n=0;ChipTable[n].label != NULL;n++)
						fprintf (stderr, "\t\"%s\"\n", ChipTable[n].label);
					fprintf (stderr, "Exiting.\n");
					exit (1);
				}
				break;
			case ('b'):
				beadMaskFile = strdup (optarg);
				break;
			case ('e'):
				excludeMaskFile = strdup (optarg);
				break;
			case ('v'):
				fprintf (stdout, "%s - Version 0.1\n", argv[0]);
				exit (0);
				break;
			default:
				fprintf (stderr, "We are broken:\n");
				exit (1);
				break;
		}
	}
	
	fprintf (stdout, "=========================================================================\n");
	fprintf (stdout,"\n");
	fprintf (stdout, "NOTE!!!\n");
	fprintf (stdout,"Currently need to strip commas out of the mask files from Torrent Explorer\n");
	fprintf (stdout,"\n");
	fprintf (stdout,"Do this:\n");
	fprintf (stdout,"\n\tsed -i '3~1s/,//g' <maskfilename>\n");
	fprintf (stdout,"\n");
	fprintf (stdout,"You will get an output file named: <maskfilename>.bin\n");
	fprintf (stdout,"\n");
	fprintf (stdout, "=========================================================================\n");
	fprintf (stdout,"\n");
	
	if (!beadMaskFile) {
		fprintf (stderr, "Error: Please specify a beadmask filename (-b <filename>)\n");
		exit (0);
	}
	
	if (!excludeMaskFile) {
		fprintf (stderr, "Error: Please specify an exclude mask filename (-e <filename>)\n");
		exit (0);
	}
	
	if (!myChipType) {
		fprintf (stderr, "Using default chip type: %s width=%d height=%d\n", ChipTable[0].label,ChipTable[0].w,ChipTable[0].h);
		myChipType = &ChipTable[0];
	}
				
	assert (myChipType);
	assert (beadMaskFile);
	assert (excludeMaskFile);

	//--- mask file format
	//---	First Line: x-origin,y-origin
	//---	Second Line: x-length,y-length
	//---	Followed by y number of lines, each line x long
	
	FILE *fpbead = fopen(beadMaskFile, "r");
	FILE *fpexclude = fopen(excludeMaskFile, "r");
	
	if (!fpbead) {
		perror (beadMaskFile);
		exit (1);
	}
	
	if (!fpexclude) {
		perror (excludeMaskFile);
		exit (1);
	}
	
	//---	header information parsing
	float bmxo, bmyo;
	int bmxl, bmyl;
	float emxo, emyo;
	int emxl, emyl;
	int stat;
	char *sstat;
	
	stat = fscanf (fpbead, "%f,%f\n", &bmxo, &bmyo);	// beadmask x origin and beadmask y origin
	stat = fscanf (fpbead, "%d,%d\n", &bmxl, &bmyl);	// beadmask x length and beadmask y length
	stat = fscanf (fpexclude, "%f,%f\n", &emxo, &emyo);	// beadmask x origin and beadmask y origin
	stat = fscanf (fpexclude, "%d,%d\n", &emxl, &emyl);	// beadmask x length and beadmask y length
	
	if (bmxo != emxo || bmyo != emyo) {
		fprintf (stderr, "Error: Origin of the bead mask and exclude mask is not identical\n");
		fprintf (stderr, "Exiting.\n");
		exit (1);
	}
	if (bmxl != emxl || bmyl != emyl) {
		fprintf (stderr, "Error: Dimension(s) of the bead mask and exclude mask is(are) not identical\n");
		fprintf (stderr, "Exiting.\n");
		exit (1);
	}
	
	//---	Initialize a full-chip Mask (All wells are MaskEmpty)
	Mask fullChipMask(myChipType->w, myChipType->h);
	
	//---	Mark all wells outside region MaskExclude
	for (int y=0;y<myChipType->h;y++) {
		for (int x=0;x<myChipType->w;x++) {
			if (y < bmyo){
				fullChipMask[x+y*myChipType->w] = MaskExclude;
			}
			else if (y > (bmyo + bmyl)){
				fullChipMask[x+y*myChipType->w] = MaskExclude;
			}
			else {
				if (x < bmxo) {
					fullChipMask[x+y*myChipType->w] = MaskExclude;
				}
				else if (x > (bmxo + bmxl)) {
					fullChipMask[x+y*myChipType->w] = MaskExclude;
				}
				else {
					// This is the central region of interest zone.
				}
			}
		}
	}
	
	char buf[16384];
	
	//---	Loop thru the regional area of mask and fill in exclude data
	for (int y=bmyo;y<(bmyo+bmyl);y++){
		sstat = fgets(buf, sizeof(buf), fpexclude);
		Trim(buf);
		for (int idx=0, x=bmxo;x<(bmxo+bmxl);x++,idx++){
			if (buf[idx] == '1')
				fullChipMask[x+y*myChipType->w] = MaskExclude;
				//fprintf (stdout, "Exclude (%d,%d)\n", x, y);
		}
	}
	
	//---	Loop thru the regional area of mask and fill in bead data
	for (int y=bmyo;y<(bmyo+bmyl);y++){
		sstat = fgets(buf, sizeof(buf), fpbead);
		Trim(buf);
		for (int idx=0, x=bmxo;x<(bmxo+bmxl);x++,idx++){
			if (buf[idx] == '1') {
				fullChipMask[x+y*myChipType->w] = MaskBead;
				fprintf (stdout, "Bead (%d,%d)\n", x, y);
			}
		}
	}
	
	char maskName[MAX_PATH_LENGTH];
	strcpy(maskName, beadMaskFile);
	char *ptr = strrchr(maskName, '.');
	if (ptr) {
		strcpy(ptr+1, "bin");
	} else {
		strcat(maskName, ".bin");
	}

	printf("Writing out mask: %s\n", maskName);
	fullChipMask.WriteRaw(maskName);
	
	fclose (fpbead);
	fclose (fpexclude);
}

