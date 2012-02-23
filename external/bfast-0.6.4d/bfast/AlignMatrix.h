#ifndef ALIGNMATRIX_H_
#define ALIGNMATRIX_H_

/* NT Matrix only uses the first element in the array */
typedef struct {
	int32_t score[ALPHABET_SIZE+1];
	int32_t from[ALPHABET_SIZE+1];
	int32_t length[ALPHABET_SIZE+1];
	/* Color space specific items */
	//int8_t colorError[ALPHABET_SIZE+1];
} AlignMatrixSubCell;

typedef struct {
	AlignMatrixSubCell h; // deletion
	AlignMatrixSubCell s; // match/mismatch 
	AlignMatrixSubCell v; // insertion 
} AlignMatrixCell;

typedef struct {
	AlignMatrixCell **cells;
	int32_t nrow;
	int32_t ncol;
} AlignMatrix;

void AlignMatrixInitialize(AlignMatrix*);
void AlignMatrixFree(AlignMatrix*);
void AlignMatrixReallocate(AlignMatrix*, int32_t, int32_t);

#endif
