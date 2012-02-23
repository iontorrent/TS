#ifndef RGRANGES_H_
#define RGRANGES_H_

#include <stdio.h>
#include "BLibDefinitions.h"

void RGRangesCopyToRGMatch(RGRanges*, RGIndex*, RGMatch*, int32_t, int32_t);
void RGRangesAllocate(RGRanges*, int32_t);
void RGRangesReallocate(RGRanges*, int32_t);
void RGRangesFree(RGRanges*);
void RGRangesInitialize(RGRanges*);

#endif

