/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include "SeqUtils.h"

using namespace std;

int isBaseArray[256];

struct isBaseArrayInit {
    isBaseArrayInit(){
        fill_n(isBaseArray, 256, 0);
        isBaseArray['a'] = 1;
        isBaseArray['A'] = 1;
        isBaseArray['c'] = 1;
        isBaseArray['C'] = 1;
        isBaseArray['g'] = 1;
        isBaseArray['G'] = 1;
        isBaseArray['t'] = 1;
        isBaseArray['T'] = 1;
    }
};

static isBaseArrayInit isBaseArrayInitObject; 

void NextHP(const string& seq, int& begin, int& end, int& hpLen)
{
    // First advance begin to start of next HP block:
    int seqLen = seq.length();
    for(begin=end; begin<seqLen and not isBase(seq[begin]); ++begin)
        ;

    // Find the end of this block, ignoring any insertions ('-'):
    hpLen = 0;
    end   = begin;
    char base = seq[begin];
    while(end < seqLen){
        if(seq[end] == base)
            ++hpLen;
        else if(seq[end] == '-')
            ;
        else
            break;
        ++end;
    }
}

inline int SkipInsertions(FlowSeq& flowNum, int seqPos, int seqLen, int flow, const string& seq)
{
    while(seq[seqPos]=='-' and seqPos<seqLen) 
        flowNum[seqPos++] = flow;

    return seqPos;
}

int ReadFlowNum(FlowSeq& flowNum, const string& flowOrder, const string& seq)
{
    int cycleLen = flowOrder.length();
    int seqLen   = seq.length();
    int seqPos   = 0;
    int flow     = 0;

    flowNum.resize(seqLen);
    for(; seqPos<seqLen; ++flow){
        seqPos = SkipInsertions(flowNum, seqPos, seqLen, flow, seq);
        char nuc = flowOrder[flow%cycleLen];
        while(seqPos<seqLen and (seq[seqPos]==nuc or seq[seqPos]=='-')){
            flowNum[seqPos] = flow;
            ++seqPos;
        }
    }
    
    SkipInsertions(flowNum, seqPos, seqLen, flow, seq);

    return flow;
}

