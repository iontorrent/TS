#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "BError.h"
#include "BLibDefinitions.h"
#include "RGIndexLayout.h"

/* TODO */
void RGIndexLayoutCreate(char *mask, int32_t hashWidth, int32_t depth, RGIndexLayout *layout)
{
	char *FnName="RGIndexLayoutCreate";
	int i;

	assert(NULL != mask);

	layout->depth = depth;
	layout->width = strlen(mask);
	layout->mask=NULL;
	layout->mask = malloc(sizeof(int32_t)*(layout->width));
	if(NULL==layout->mask) {
		PrintError(FnName, "layout->mask", "Could not allocate memory", Exit, MallocMemory);
	}
	layout->hashWidth=hashWidth;
	if(RGINDEXLAYOUT_MAX_HASH_WIDTH < layout->hashWidth) {
		PrintError(FnName, "layout->hashWidth", "The hash width was too large", Exit, OutOfRange);
	}
	/* Copy over from temp mask */
	layout->keysize=0;
	for(i=0;i<layout->width;i++) {
		switch(mask[i]) {
			case '0':
				layout->mask[i] = 0;
				break;
			case '1':
				layout->mask[i] = 1;
				layout->keysize++;
				break;
			default:
				PrintError(FnName, NULL, "Could not understand mask", Exit, OutOfRange);
		}
	}
	/* Check mask begins and ends with a one */
	if(layout->mask[0] == 0) {
		PrintError(FnName, NULL, "Layout must begin with a one", Exit, OutOfRange);
	}
	if(layout->mask[layout->width-1] == 0) {
		PrintError(FnName, NULL, "Layout must begin with a one", Exit, OutOfRange);
	}
	/* Check that key-size is greater than or equalt to the hash width */
	if(layout->keysize < layout->hashWidth + layout->depth) {
		PrintError(FnName, NULL, "Hash width + depth are greater than the key size", Exit, OutOfRange);
	}
}

void RGIndexLayoutDelete(RGIndexLayout *layout)
{
	free(layout->mask);
	layout->mask=NULL;;
}
