#include <stdio.h>
#include <string.h>
#include <config.h>
#include <stdint.h>
#include "BError.h"
#include "Main.h"

static int usage()
{
	fprintf(stderr, "\n");
	fprintf(stderr, "BFAST:   the blat-like fast accurate search tool\n");
#ifdef GIT_REV
	fprintf(stderr, "Version: %s git:%s\n", PACKAGE_VERSION, GIT_REV);
#else
	fprintf(stderr, "Version: %s\n", PACKAGE_VERSION);
#endif
	fprintf(stderr, "Contact: %s\n\n", PACKAGE_BUGREPORT);
	fprintf(stderr, "Usage:   bfast <command> [options]\n\n"); 
	fprintf(stderr, "Pre-processing:\n");
	fprintf(stderr, "         fasta2brg\n");
	fprintf(stderr, "         index\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Alignment:\n");
	fprintf(stderr, "         match\n");
	fprintf(stderr, "         localalign\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Post-processing:\n");
	fprintf(stderr, "         postprocess\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "File Conversion:\n");
	fprintf(stderr, "         bafconvert\n");
	fprintf(stderr, "         header\n");
	fprintf(stderr, "         bmfconvert\n");
	fprintf(stderr, "         brg2fasta\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Easy Alignment:\n");
	fprintf(stderr, "         easyalign\n");
	return 1;
}

int main(int argc, char *argv[])
{
	if(argc < 2) return usage();
	else if (0 == strcmp("fasta2brg", argv[1])) return BfastFasta2BRG(argc-1, argv+1);
	else if (0 == strcmp("index", argv[1])) return BfastIndex(argc-1, argv+1);
	else if (0 == strcmp("match", argv[1])) return BfastMatch(argc-1, argv+1);
	else if (0 == strcmp("localalign", argv[1])) return BfastLocalAlign(argc-1, argv+1);
	else if (0 == strcmp("postprocess", argv[1])) return BfastPostProcess(argc-1, argv+1);
	else if (0 == strcmp("bafconvert", argv[1])) return BfastBAFConvert(argc-1, argv+1);
	else if (0 == strcmp("header", argv[1])) return BfastHeader(argc-1, argv+1);
	else if (0 == strcmp("bmfconvert", argv[1])) return BfastBMFConvert(argc-1, argv+1);
	else if (0 == strcmp("brg2fasta", argv[1])) return BfastBRG2Fasta(argc-1, argv+1);
	else if (0 == strcmp("easyalign", argv[1])) return BfastAlign(argc-1, argv+1);
	else {
		PrintError("bfast", argv[1], "Unknown command", Exit, OutOfRange);
	}
	return 0;
}

