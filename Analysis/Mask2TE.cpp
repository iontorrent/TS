/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// Converter: Mask list to TE mask format.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
int main (int argc, char *argv[])
{

	if (argc != 2)
		return 1;
	
	FILE *fp = NULL;
	FILE *out = NULL;
	char *outFile = {"mask1"};

	fp = fopen (argv[1], "rb");
	if (fp == NULL)
		return 1;
	out = fopen (outFile, "wb");
	
	int row, col;
	int w, h;
	int n = 0;
	
	n = fscanf (fp, "%d %d\n", &w, &h);
	assert (n == 2);
	
	int *array = (int *) malloc(sizeof(int) * w * h);
	memset(array,0,sizeof(int) * w * h);
	
	while (!feof(fp)) {
		n = fscanf (fp, "%d %d\n", &row, &col);
		array[row*w + col] = 1;
	}
	
	fprintf (out, "0,0\n");	// origin of maks area, default to whole-chip
	fprintf (out, "%d,%d\n", w,h);	// width and height of region
	
	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++) {
			fprintf (out, "%d", array[x+y*w]);
			if (x+1 != w)
				fprintf (out, ",");
		}
		fprintf (out, "\n");
	}

    if(fp!=NULL) {
        fclose(fp);
    }
    if(out!=NULL) {
        fclose(out);
    }

    free(array);
    
	return (0);
}
