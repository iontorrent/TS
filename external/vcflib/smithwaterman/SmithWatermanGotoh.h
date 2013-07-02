#pragma once

#include <iostream>
#include <algorithm>
#include <memory>
//#include "Alignment.h"
#include "Mosaik.h"
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <string>
#include "disorder.h"
#include "Repeats.h"
#include "LeftAlign.h"

using namespace std;

#define MOSAIK_NUM_NUCLEOTIDES 26
#define GAP '-'

class CSmithWatermanGotoh {
public:
    // constructor
    CSmithWatermanGotoh(float matchScore, float mismatchScore, float gapOpenPenalty, float gapExtendPenalty);
    // destructor
    ~CSmithWatermanGotoh(void);
    // aligns the query sequence to the reference using the Smith Waterman Gotoh algorithm
    void Align(unsigned int& referenceAl, string& cigarAl, const string& s1, const string& s2);
    // enables homo-polymer scoring
    void EnableHomoPolymerGapPenalty(float hpGapOpenPenalty);
    // enables non-repeat gap open penalty
    void EnableEntropyGapPenalty(float enGapOpenPenalty);
    // enables repeat gap extension penalty
    void EnableRepeatGapExtensionPenalty(float rGapExtensionPenalty, float rMaxGapRepeatExtensionPenaltyFactor = 10);
private:
    // creates a simple scoring matrix to align the nucleotides and the ambiguity code N
    void CreateScoringMatrix(void);
    // corrects the homopolymer gap order for forward alignments
    void CorrectHomopolymerGapOrder(const unsigned int numBases, const unsigned int numMismatches);
    // returns the maximum floating point number
    static inline float MaxFloats(const float& a, const float& b, const float& c);
    // our simple scoring matrix
    float mScoringMatrix[MOSAIK_NUM_NUCLEOTIDES][MOSAIK_NUM_NUCLEOTIDES];
    // keep track of maximum initialized sizes
    unsigned int mCurrentMatrixSize;
    unsigned int mCurrentAnchorSize;
    unsigned int mCurrentQuerySize;
    unsigned int mCurrentAQSumSize;
    // define our traceback directions
    // N.B. This used to be defined as an enum, but gcc doesn't like being told
    // which storage class to use
    const static char Directions_STOP;
    const static char Directions_LEFT;
    const static char Directions_DIAGONAL;
    const static char Directions_UP;
    // repeat structure determination
    const static int repeat_size_max;
    // define scoring constants
    const float mMatchScore;
    const float mMismatchScore;
    const float mGapOpenPenalty;
    const float mGapExtendPenalty;
    // store the backtrace pointers
    char* mPointers;
    // store the vertical gap sizes - assuming gaps are not longer than 32768 bases long
    short* mSizesOfVerticalGaps;
    // store the horizontal gap sizes - assuming gaps are not longer than 32768 bases long
    short* mSizesOfHorizontalGaps;	
    // score if xi aligns to a gap after yi
    float* mQueryGapScores;
    // best score of alignment x1...xi to y1...yi
    float* mBestScores;
    // our reversed alignment
    char* mReversedAnchor;
    char* mReversedQuery;
    // define static constants
    static const float FLOAT_NEGATIVE_INFINITY;
    // toggles the use of the homo-polymer gap open penalty
    bool mUseHomoPolymerGapOpenPenalty;
    // specifies the homo-polymer gap open penalty
    float mHomoPolymerGapOpenPenalty;
    // toggles the use of the entropy gap open penalty
    bool mUseEntropyGapOpenPenalty;
    // specifies the entropy gap open penalty (multiplier)
    float mEntropyGapOpenPenalty;
    // toggles the use of the repeat gap extension penalty
    bool mUseRepeatGapExtensionPenalty;
    // specifies the repeat gap extension penalty
    float mRepeatGapExtensionPenalty;
    // specifies the max repeat gap extension penalty
    float mMaxRepeatGapExtensionPenalty;
};

// returns the maximum floating point number
inline float CSmithWatermanGotoh::MaxFloats(const float& a, const float& b, const float& c) {
    float max = 0.0f;
    if(a > max) max = a;
    if(b > max) max = b;
    if(c > max) max = c;
    return max;
}
