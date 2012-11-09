/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iomanip>
#include <string>
#include "api/BamMultiReader.h"
#include "api/BamWriter.h"
#include "api/BamAlignment.h"
#include "api/SamHeader.h"

using namespace std;
using namespace BamTools;

/* Complie with

   g++ -O3 -I $TS/external/bamtools/include -L$TS/external/bamtools/lib bam_merge.cpp -lbamtools -o bam_merge */

int main(int argc, char* argv[])
{
    vector<string> bamFile;
    for(int i=1; i<argc; ++i)
        bamFile.push_back(argv[i]);

    string outFile = "combined.bam";

    BamMultiReader reader;
    reader.Open(bamFile);

    SamHeader header = reader.GetHeader();
    RefVector refs   = reader.GetReferenceData();

    BamWriter writer;
    writer.SetNumThreads(8);
    writer.Open(outFile, header, refs);
    assert(writer.IsOpen());

    BamAlignment al;
    size_t numReads = 0;
    while(reader.GetNextAlignment(al)){
        writer.SaveAlignment(al);
        if(++numReads % 10000 == 0)
            cerr << setw(12) << numReads << '\r';
    }
    cerr << setw(12) << numReads << endl;
    cerr << "done!"  << endl;
}

