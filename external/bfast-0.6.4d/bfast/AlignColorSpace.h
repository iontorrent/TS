#ifndef ALIGNCOLORSPACE_H_
#define ALIGNCOLORSPACE_H_
#include "AlignedEntry.h"
#include "BLibDefinitions.h"

int32_t AlignColorSpaceUngapped(char*, char*, int, char*, int, int, ScoringMatrix*, AlignedEntry*, int, int32_t, char);
void AlignColorSpaceUngappedGetBest(ScoringMatrix*, int32_t, char, char, int32_t, int32_t, int32_t, int32_t*, int32_t*, char*);
void AlignColorSpaceGappedBounded(char*, int, char*, int, ScoringMatrix*, AlignedEntry*, AlignMatrix*, int32_t, char, int32_t, int32_t);
void AlignColorSpaceGappedConstrained(char*, char*, int, char*, int, ScoringMatrix*, AlignedEntry*, AlignMatrix*, int32_t, int32_t, int32_t, int32_t, char);
void AlignColorSpaceRecoverAlignmentFromMatrix(AlignedEntry*, AlignMatrix*, char*, int, char*, int, int32_t, int32_t, int, int32_t, char, int, int);

void AlignColorSpaceInitializeAtStart(char*, AlignMatrix*, ScoringMatrix*, int32_t, int32_t, int32_t, char);
void AlignColorSpaceInitializeToExtend(char*, AlignMatrix*, ScoringMatrix*, int32_t, int32_t, int32_t, int32_t, int32_t);
void AlignColorSpaceFillInCell(char*, int32_t, char*, int32_t, ScoringMatrix*, AlignMatrix*, int32_t, int32_t, char, int32_t, int32_t, int32_t);
int32_t AlignColorSpaceGetAlphabetSize(char*, int32_t, char*, int32_t);


#endif
