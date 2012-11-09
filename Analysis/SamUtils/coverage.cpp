/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "api/BamMultiReader.h"
#include "api/BamAlignment.h"

using namespace std;
using namespace BamTools;

/* Complie with

   g++ -O3 -I $TS/external/bamtools/include -L$TS/external/bamtools/lib coverage.cpp -lbamtools -o coverage
*/

typedef vector<int64_t> Histogram;

void PrintHist(Histogram& hist);

bool GetNextAlignment(BamAlignment& al, BamMultiReader& reader, int32_t refID)
{
    bool good = reader.GetNextAlignmentCore(al);
    if(not good or al.RefID != refID)
        return false;
    else if(not al.IsMapped() or al.IsDuplicate() or al.IsFailedQC())
        return GetNextAlignment(al, reader, refID);
    else
        return true;
}

void CountDepth(Histogram& hist, BamMultiReader& reader, BamAlignment& al, int32_t refID, int64_t refLen)
{
    bool moreReads = (al.RefID == refID);

    int32_t maxReadLen = 1000;
    vector<int64_t> readEnds(maxReadLen);

    int64_t depth = 0;
    for(int64_t pos=0; pos<refLen; ++pos){
        while(moreReads and al.Position == pos){
            ++depth;
            assert(al.GetEndPosition() - pos < maxReadLen);
            ++readEnds[al.GetEndPosition() % maxReadLen];
            moreReads = GetNextAlignment(al, reader, refID);
        }
        depth -= readEnds[pos % maxReadLen];
        assert(depth >= 0);
        readEnds[pos % maxReadLen] = 0;
        if(depth >= hist.size())
            hist.resize(2 * depth);
        ++hist[depth];
    }
}

int main(int argc, char* argv[])
{
    vector<string> bamFiles(argc-1);
    for(int i=1; i<argc; ++i)
        bamFiles[i-1] = argv[i];

    BamMultiReader reader;
    reader.Open(bamFiles);

    int32_t refID    = 0;
    int32_t maxDepth = 10;
    Histogram hist(maxDepth);
    const RefVector& refData = reader.GetReferenceData();
    assert(not refData.empty());
    BamAlignment al;
    bool moreReads = GetNextAlignment(al, reader, 0);
    for(RefVector::const_iterator r=refData.begin(); r!=refData.end(); ++r, ++refID){
        cout << setw(12) << r->RefLength << "\t" << r->RefName << endl;
        cerr << setw(12) << r->RefLength << "\t" << r->RefName << endl;
        CountDepth(hist, reader, al, refID, r->RefLength);
    }
    PrintHist(hist);
}

void PrintHist(Histogram& hist)
{
    int32_t maxDepth = hist.size();
    while(maxDepth >= 0 and hist[--maxDepth] == 0)
        ;

    for(int depth=0; depth<=maxDepth; ++depth)
        cout << setw(6) << depth << setw(12) << hist[depth] << endl;
    cout << endl;
}

