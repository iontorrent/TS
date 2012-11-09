/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include "BarCode.h"

char *IonSet1 = \
"file_id IonSet1\n"
"score_mode 1\n"
"score_cutoff 2.0\n"
"barcode 1,1,TACTCACGATA,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 2,2,TCGTGTCGCAC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 3,3,TGATGATTGCC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 4,4,TCGATAATCTT,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 5,5,TCTTACACCAC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 6,6,TAGCCAAGTAC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 7,7,TGACATTACTT,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 8,8,TGCCTTACCGC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 9,9,TACCGAGGCAC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 10,10,TGCAAGCCTTC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 11,11,TACATTACATC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 12,12,TCAAGCACCGC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 13,13,TAGCTTACCGC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 14,14,TCATGATCAAC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 15,15,TGACCGCATCC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n"
"barcode 16,16,TGGTGTAGCAC,CTGCTGTACGGCCAAGGCGT,,none,11,none\n";

char *IonSet2 = \
"file_id IonSet2\n"
"score_mode 1\n"
"score_cutoff 2.0\n"
"barcode 1,1,T,,,,none,1,none\n"
"barcode 2,2,AA,,,,none,2,none\n";

// legacy format
char *IonSet3 = \
"file_id IonSet2\n"
"barcode 1,T,,,,none,1,none\n"
"barcode 2,AA,,,,none,2,none\n";

char *IonXpress = \
"file_id IonXpress\n"
"score_mode 0\n"
"score_cutoff 0.0\n"
"barcode 16,IonXpress_16,TCTGGATGAC,GAT,,none,10,none\n"
"barcode 15,IonXpress_15,TCTAGAGGTC,GAT,,none,10,none\n"
"barcode 14,IonXpress_14,TTGGAGTGTC,GAT,,none,10,none\n"
"barcode 13,IonXpress_13,TCTAACGGAC,GAT,,none,10,none\n"
"barcode 12,IonXpress_12,TAGGTGGTTC,GAT,,none,10,none\n"
"barcode 11,IonXpress_11,TCCTCGAATC,GAT,,none,10,none\n"
"barcode 10,IonXpress_10,CTGACCGAAC,GAT,,none,10,none\n"
"barcode 9,IonXpress_9,TGAGCGGAAC,GAT,,none,10,none\n"
"barcode 8,IonXpress_8,TTCCGATAAC,GAT,,none,10,none\n"
"barcode 7,IonXpress_7,TTCGTGATTC,GAT,,none,10,none\n"
"barcode 6,IonXpress_6,CTGCAAGTTC,GAT,,none,10,none\n"
"barcode 5,IonXpress_5,CAGAAGGAAC,GAT,,none,10,none\n"
"barcode 4,IonXpress_4,TACCAAGATC,GAT,,none,10,none\n"
"barcode 3,IonXpress_3,AAGAGGATTC,GAT,,none,10,none\n"
"barcode 2,IonXpress_2,TAAGGAGAAC,GAT,,none,10,none\n"
"barcode 1,IonXpress_1,CTAAGGTAAC,GAT,,none,10,none";


int seq2flow(char *seq, unsigned short *flowVals, char *flowOrder)
{
	int numFlowVals = 0;
	int base = 0;
	int len = strlen(seq);
	int flowOrderLen = strlen(flowOrder);
	int flow = 0;
	unsigned short seed[3] = {1,1,0};	
	
	while (base < len) {
		flowVals[flow] = 0;
		while (flowOrder[flow%flowOrderLen] == seq[base] && base < len) {
//			flowVals[flow] += 100;
      flowVals[flow]+= 100 + 20*erand48(seed); //add some noise
		  base++;
		}
		flow++;
	}
	return flow;
}

int main(int argc, char *argv[])
{
	// make a barcode list file
	FILE *fp = fopen("TestBarcodeFile.txt", "w");
	//fprintf(fp, "%s", IonSet2);
	//fprintf(fp, "%s", IonSet3);
	fprintf(fp, "%s", IonXpress);
	fclose(fp);

	barcode bc;
	bool rtbug = true;
	bc.SetRTDebug (rtbug);

	char *samba = "TACGTACGTCTGAGCATCGATCGATGTACAGC";
	bc.SetFlowOrder(samba);

	bc.ReadBarCodeFile("TestBarcodeFile.txt");

	char *seq[10];	
/*
	seq[0] = "TCAGTCTTACACCACCTGCTGTACGGCCAAGGCGTCCGGGCCCAAATTT"; // barcode 5 + CCGGGCCCAAATTT
	seq[1] = "TCAGTCTTACACCACCTGCTGTACGGCCAAGGCGTTGAACGGACTGACT"; // barcode 5 + TGAACGGACTGACT
	seq[2] = "TCAGTCTTACACCACTGCTGTACGGCCAAGGCGTTGAACGGACTGACT"; // barcode 5 w/ undercall on C in adapter + TGAACGGACTGACT
	seq[3] = "TCAGTCTTACACACCTGCTGTACGGCCAAGGCGTTGAACGGACTGACT"; // barcode 5 w/ undercall on C in barcode (shared by adapter) + TGAACGGACTGACT
	seq[4] = "TCAGTCTACACCACCTGCTGTACGGCCAAGGCGTTGAACGGACTGACT"; // barcode 5 w/ undercall on T in barcode + TGAACGGACTGACT
	seq[5] = "TCAGTCTACACCACCTGCTGTACGGCCAAGGCGTCCGGGCCCAAATTT"; // barcode 5 w/ undercall on T in barcode + CCGGGCCCAAATTT
	seq[6] = "TCAGGGTGTAGCACCTGCTGTACGGCCAAGGCGTCCGGGCCCAAATTT"; // barcode 16 w/ undercall on first T in barcode + CCGGGCCCAAATTT
	seq[7] = "TCAGTGGTTGTTAGCACCTGCTGTACGGCCAAGGCGTCCGGGCCCAAATTT"; // barcode 16 w/ two overcalls on T + CCGGGCCCAAATTT
*/

/* test cases specific to IonSet3
	seq[0] = "TCAGTGAACGGACTGACT"; // barcode MGD-1 - key + one base, barcode right-clip should be 6
	seq[1] = "TCAGAAGAACGGACTGACT"; // barcode MGD-2 - key + two bases, barcode right-clip should be 7
*/

	int seqv[] = {1,0,1,0,0,1,0,1,0,1,0,0,3,2,0,2,0,1,1,1,1,0,0,2,0,1,1,0,0,0,1,0,1,1,0,1,0,0,2,1};
	char *flowOrder = samba;
	int len = sizeof(seqv) / sizeof(int);
	char localbuf0[256];
	seq[0] = localbuf0;
	int base = 0;
	for(int flow=0;flow<len;flow++) {
		int bases = seqv[flow];
		while(bases) {
			localbuf0[base] = flowOrder[flow%strlen(flowOrder)];
			bases--;
			base++;
		}
	}
	localbuf0[base] = 0; // NULL-terminate it

	// seq[0] = "TCAGCAAAGGAACAAGATGTGTAGCCG"; // barcode BC5 miss-classified as BC2 due to missing 'G', fragment = CCGGGCCCAAATTT
	int numTestCases = 1;

//	char *flowOrder = "TACG";
	unsigned short flowVals[1000];

	int numFlowVals;
	bcmatch *bcm;

	int mode = 1; // 0
	double value = 2.0; // 0.5;

	// bc.SetScoreMode(mode);
	// bc.SetScoreCutoff(value);

	for (int i=0; i<numTestCases; i++) {
		printf("Sequence seq:%d %s\n", i, seq[i]);
		numFlowVals = seq2flow(seq[i], flowVals, samba);
		bcm = bc.flowSpaceTrim(flowVals, numFlowVals);
		if (bcm) {
			printf("Barcode: %s  clip: %d %d\n\n", bcm->matching_code, bcm->bc_left, bcm->bc_right);
			free(bcm->matching_code);
			free(bcm);
		} else {
			printf("Barcode: not found?\n\n");
		}
	}
}

