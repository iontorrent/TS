/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SEQUTILS_H
#define SEQUTILS_H

#include <string>
#include <vector>

extern int isBaseArray[256];

inline bool isBase(int x)
{
    return isBaseArray[x];
}

// Find next HP block in seq, on or after position end.
void NextHP(const std::string& seq, int& begin, int& end, int& hpLen);

// For each base in seq, find the flow on which that base should
// incorporate. Store results in flowNum.

typedef std::vector<int> FlowSeq;

int ReadFlowNum(FlowSeq& flowNum, const std::string& flowOrder, const std::string& seq);

#endif // SEQUTILS_H

