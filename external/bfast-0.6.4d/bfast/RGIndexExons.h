#ifndef RGINDEXEXONS_H_
#define RGINDEXEXONS_H_

#include "BLibDefinitions.h"

void RGIndexExonsRead(char*, RGIndexExons*);
int RGIndexExonsWithin(RGIndexExons*, uint32_t, uint32_t, uint32_t, uint32_t);
void RGIndexExonsInitialize(RGIndexExons*);
void RGIndexExonsDelete(RGIndexExons*);

#endif
