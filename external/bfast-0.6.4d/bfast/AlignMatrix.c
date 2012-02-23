#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "BLibDefinitions.h"
#include "BError.h"
#include "AlignMatrix.h"

/* This will destroy any data present */
void AlignMatrixReallocate(AlignMatrix *m, int32_t nrow, int32_t ncol)
{
	char *FnName="AlignMatrixReallocate";
	int32_t i, prevNRow, prevNCol;

	assert(0 < nrow);
	assert(0 < ncol);
	assert(nrow < SEQUENCE_LENGTH);
	assert(ncol < SEQUENCE_LENGTH);

	prevNRow = m->nrow;
	prevNCol = m->ncol;

	/* rows */
	for(i=nrow;i<prevNRow;i++) { // free old rows
		free(m->cells[i]);
	}
	m->cells = realloc(m->cells, sizeof(AlignMatrixCell*)*nrow);
	if(NULL == m->cells) {
		PrintError(FnName, "m->cells", "Could not reallocate memory", Exit, ReallocMemory);
	}
	for(i=prevNRow;i<nrow;i++) { // initialize new rows
		m->cells[i] = malloc(sizeof(AlignMatrixCell)*m->ncol); 
		if(NULL == m->cells[i]) {
			PrintError(FnName, "m->cells[i]", "Could not allocate memory", Exit, MallocMemory);
		}
	}
	m->nrow = nrow;

	/* cols */
	for(i=0;i<m->nrow;i++) {
		m->cells[i] = realloc(m->cells[i], sizeof(AlignMatrixCell)*ncol); 
		if(NULL == m->cells[i]) {
			PrintError(FnName, "m->cells[i]", "Could not reallocate memory", Exit, ReallocMemory);
		}
	}
	m->ncol = ncol;
}

void AlignMatrixInitialize(AlignMatrix *m)
{
	m->cells=NULL;
	m->nrow=m->ncol=0;
}

void AlignMatrixFree(AlignMatrix *m)
{
	int32_t i;
	for(i=0;i<m->nrow;i++) {
		free(m->cells[i]);
	}
	free(m->cells);
	AlignMatrixInitialize(m);
}
