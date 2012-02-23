#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <config.h>
#include <unistd.h>
#include "BLibDefinitions.h"
#include "AlignedRead.h"
#include "AlignedReadConvert.h"
#include "BError.h"
#include "BLib.h"

#define Name "bfast brg2fasta"
#define BRG2FASTA_FASTA_LINE_LENGTH 60
#define BAFCONVERT_ROTATE_NUM 100000

/* Prints the reference genome in FASTA format.
 * */

int BfastBRG2Fasta(int argc, char *argv[])
{
	char rgFileName[MAX_FILENAME_LENGTH]="\0";
	char fastaFileName[MAX_FILENAME_LENGTH]="\0";
	RGBinary rg;
	int32_t space = NTSpace;

	if(2 == argc) {
		strcpy(rgFileName, argv[1]);
		/* Infer the space */
		strcpy(fastaFileName, rgFileName);
		assert(0 < strlen(rgFileName) - strlen(BFAST_RG_FILE_EXTENSION) - 1);
		fastaFileName[strlen(rgFileName) - strlen(BFAST_RG_FILE_EXTENSION) - 1] = '\0'; // remove file extension
		assert(strlen(SPACENAME(NTSpace)) == strlen(SPACENAME(ColorSpace))); // must hold for the next comparison to work
		assert(0 < strlen(fastaFileName)-2);
		if(0 == strcmp(SPACENAME(NTSpace), fastaFileName + (strlen(fastaFileName)-2))) {
			space = NTSpace;
		}
		else { 
			space = ColorSpace;
		}
		assert(0 < strlen(fastaFileName) - strlen(SPACENAME(space)) - 1);
		fastaFileName[strlen(fastaFileName) - strlen(SPACENAME(space)) - 1]='\0'; // remove space name

		/* Read the BRG */
		RGBinaryReadBinary(&rg,
				space,
				fastaFileName);
		/* Unpack */
		RGBinaryUnPack(&rg);

		int32_t i, j, ctr;
		for(i=0;i<rg.numContigs;i++) {
			fprintf(stdout, ">%s\n",
					rg.contigs[i].contigName);
			for(j=ctr=0;j<rg.contigs[i].sequenceLength;j++) {
				putchar(rg.contigs[i].sequence[j]);
				ctr++;
				if(BRG2FASTA_FASTA_LINE_LENGTH < ctr) {
					putchar('\n');
					ctr=0;
				}
			}
			if(0 < ctr) {
				putchar('\n');
			}
		}

		RGBinaryDelete(&rg);

		fprintf(stderr, "Terminating successfully!\n");
	}
	else {
		fprintf(stderr, "\nUsage:%s <bfast reference genome file>\n", Name);
		fprintf(stderr, "\nsend bugs to %s\n",
				PACKAGE_BUGREPORT);
	}
	return 0;
}
