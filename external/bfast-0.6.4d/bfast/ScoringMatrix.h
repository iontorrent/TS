#ifndef SCORINGMATRIX_H_
#define SCORINGMATRIX_H_

#include "BLibDefinitions.h"

int32_t ScoringMatrixGetNTScore(char, char, ScoringMatrix*);
int32_t ScoringMatrixGetColorScore(char, char, ScoringMatrix*);

int ScoringMatrixRead(char*, ScoringMatrix*, int);
void ScoringMatrixInitialize(ScoringMatrix*);
int32_t ScoringMatrixCheck(ScoringMatrix*, int32_t);

#endif
