#ifndef ALIGNNTSPACE_H_
#define ALIGNNTSPACE_H_
#include "BLibDefinitions.h"
#include "AlignMatrix.h"
#include "Align.h"

int32_t AlignNTSpaceUngapped(char*, char*, int, char*, int, int, ScoringMatrix*, AlignedEntry*, int, int32_t, char);
void AlignNTSpaceGappedBounded(char*, int, char*, int, ScoringMatrix*, AlignedEntry*, AlignMatrix*, int32_t, char, int32_t, int32_t);
void AlignNTSpaceGappedConstrained(char*, char*, int, char*, int, ScoringMatrix*, AlignedEntry*, AlignMatrix*, int32_t, int32_t, int32_t, int32_t, char);
void AlignNTSpaceRecoverAlignmentFromMatrix(AlignedEntry*, AlignMatrix*, char*, int, char*, int, int32_t, int32_t, int, int32_t, char, int);
void AlignNTSpaceInitializeAtStart(AlignMatrix*, ScoringMatrix*, int32_t, int32_t);
void AlignNTSpaceInitializeToExtend(AlignMatrix*, ScoringMatrix*, int32_t, int32_t, int32_t, int32_t);
void AlignNTSpaceFillInCell(char*, int32_t, char*, int32_t, ScoringMatrix*, AlignMatrix*, int32_t, int32_t, int32_t, int32_t);
#endif
