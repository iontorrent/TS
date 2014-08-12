/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <list>
#include <string>
#include <vector>

#include "boost/multi_array.hpp"
#include "hdf5.h"

#include "BAMReader.h"
#include "BAMUtils.h"
#include "SeqUtils.h"

using namespace std;

void FillRefFlowNum(FlowSeq& refFlowNum, int hpBegin, int hpLen, int flow)
{
    fill_n(refFlowNum.begin()+hpBegin, hpLen, flow);
}

void PrintRefHP(char nuc, int hpBegin, int hpLen, int flow)
{
    cout << setw(4) << hpBegin
         << setw(4) << nuc
         << setw(4) << hpLen
         << setw(4) << flow
         << endl;
}

void MapUnaligned(
    FlowSeq&      refFlowNum,
    list<int>&    unaligned,
    const string& refSeq,
    int           prevFlow,
    int           flow,
    const string& flowOrder
)
{
    int cycleLen = flowOrder.length();

    if(not unaligned.empty()){
        for(int f=prevFlow; f<=flow and not unaligned.empty(); ++f){
            char nuc = flowOrder[f%cycleLen];
            if(refSeq[unaligned.front()] == nuc){
                int hpStart = unaligned.front();
                int hpLen   = 0;
                do {
                    unaligned.pop_front();
                    ++hpLen;
                } while(not unaligned.empty() and unaligned.front() == nuc);
                //PrintRefHP(nuc, hpStart, hpLen, f);
                FillRefFlowNum(refFlowNum, hpStart, hpLen, f);
            }
        }

        unaligned.clear();
    }
}

void RefFlowNum(
    FlowSeq&       refFlowNum,
    int            keyLen,
    const string&  flowOrder,
    const string&  readSeq,
    const string&  matchSeq,
    const string&  refSeq,
    const FlowSeq& readFlowNum
)
{
    int refHP0   = 0;
    int refHP1   = 0;
    int refHPLen = 0;
    int prevFlow = -1;
    int refLen   = refSeq.length();

    refFlowNum.resize(refLen);
    list<int> unaligned;
    while(refHP1 < refLen){
        NextHP(refSeq, refHP0, refHP1, refHPLen);

        int flow = -1;
        for(int pos=refHP0; pos<refHP1; ++pos){
            if(matchSeq[pos] == '|'){
                assert(refSeq[refHP0] == readSeq[pos]);
                flow = readFlowNum[pos+keyLen];
                break;
            }
        }

        if(flow >= 0){
            MapUnaligned(refFlowNum, unaligned, refSeq, prevFlow, flow, flowOrder);
            //PrintRefHP(refSeq[refHP0], refHP0, refHPLen, flow);
            FillRefFlowNum(refFlowNum, refHP0, refHP1-refHP0, flow);
            prevFlow = flow;
        }else{
            unaligned.push_back(refHP0);
        }
    }
}

void PrintFlows(const string& seq, const FlowSeq& flowNum) 
{
    for(size_t i=0; i<seq.length(); ++i){
        cout << setw(3) << i
             << setw(3) << seq[i]
             << setw(6) << flowNum[i]
             << endl;
    }
}

typedef vector<int> Ionogram;

void MakeIonogram(Ionogram& ion, int numFlows, const string& seq, const FlowSeq& flowNum)
{
    ion.resize(numFlows, 0);
    assert(seq.length() <= flowNum.size());
    int len = flowNum.size();

    for(int i=0; i<len; ++i){
       if(seq[i] != '-')
           ++ion[flowNum[i]];
    }
}

void PrintAlignedIonograms(
    const Ionogram& readIon,
    const Ionogram& refIon,
    const int       numFlows,
    const string&   flowOrder)
{
    for(int flow=0; flow<numFlows; ++flow){
        char nuc   = flowOrder[flow%flowOrder.length()];
        char label = readIon[flow] == refIon[flow] ? ' ' : '*';
        cout << setw(4) << flow
             << setw(4) << nuc
             << setw(6) << readIon[flow]
             << setw(6) << refIon[flow]
             << setw(2) << label
             << endl;
    }
}

typedef boost::multi_array<int,3> CountArray;

void PrintCounts(const CountArray& counts, int numFlows, int maxKMer)
{
    for(int flow=8; flow<numFlows; ++flow){
        cout << setw(4) << flow << endl;
        for(int k0=0; k0<=maxKMer; ++k0){
            for(int k1=0; k1<=maxKMer; ++k1)
                cout << setw(12) << counts[flow][k0][k1];
            cout << endl;
        }
        cout << endl;
    }
}

void WriteCountsHDF5(const CountArray& counts, int numFlows, int maxKMer)
{
    hid_t file_id = H5Fcreate("FlowErr.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert(file_id >= 0);

    hsize_t dims[3];
    dims[0] = numFlows;
    dims[1] = maxKMer + 1;
    dims[2] = maxKMer + 1;

    hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
    assert(dataspace_id >= 0);

    hid_t dataset_id = H5Dcreate(file_id, "/dset", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dataset_id >= 0);

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, counts.data());
    assert(status >= 0);

    status = H5Dclose(dataset_id);
    assert(status >= 0);

    status = H5Sclose(dataspace_id);
    assert(status >= 0);

    status = H5Fclose(file_id);
    assert(status >= 0);
}
int Trim3Prime(const string& seq, int numBases)
{
    int alignPos = seq.length();
    for(string::const_reverse_iterator i=seq.rbegin(); i!=seq.rend() and numBases; ++i, --alignPos){
        if(isBase(*i))
            --numBases;
    }

    return alignPos;
}

int main(int argc, char* argv[])
{
    assert(argc == 7);
    string bamFile(argv[1]);
    string flowOrder(argv[2]);
    string key(argv[3]);
    int    numFlows  = strtol(argv[4], 0, 10);
    int    maxKMer   = strtol(argv[5], 0, 10);
    int    trimBases = strtol(argv[6], 0, 10);

    int       numReads = 0;
    BAMReader reader(bamFile);
    reader.open();
    CountArray counts(boost::extents[numFlows][maxKMer+1][maxKMer+1]);
    for(BAMReader::iterator i=reader.get_iterator(); i.good(); i.next(), ++numReads){
        BAMRead  read = i.get();
        BAMUtils util(read);
        string readSeq  = util.get_qdna();
        string matchSeq = util.get_matcha();
        string refSeq   = util.get_tdna();

        int alignPos = Trim3Prime(readSeq, trimBases);
        readSeq.resize(alignPos);
        matchSeq.resize(alignPos);
        refSeq.resize(alignPos);

        if(readSeq.empty()) continue;
        //cout << setw(12) << readSeq  << endl;
        //cout << setw(12) << matchSeq << endl;
        //cout << setw(12) << refSeq   << endl;

        FlowSeq readFlowNum;
        string  seq = key + readSeq;
        ReadFlowNum(readFlowNum, flowOrder, seq);
        
        FlowSeq refFlowNum;
        RefFlowNum(refFlowNum, key.length(), flowOrder, readSeq, matchSeq, refSeq, readFlowNum);
        // temporary hack (needs to be fixed in RefFlowNum):
        readFlowNum.erase(readFlowNum.begin(), readFlowNum.begin()+key.length());

        //cout << "read:" << endl;
        //PrintFlows(readSeq, readFlowNum);

        //cout << "ref:" << endl;
        //PrintFlows(refSeq, refFlowNum);

        Ionogram readIon;
        Ionogram refIon;
        MakeIonogram(readIon, numFlows, readSeq, readFlowNum);
        MakeIonogram(refIon,  numFlows, refSeq,  refFlowNum);

        //PrintAlignedIonograms(readIon, refIon, numFlows, flowOrder);

        assert(not readFlowNum.empty());
        assert(not refFlowNum.empty());
        int lastFlow = max(readFlowNum.back(), refFlowNum.back());
        for(int flow=8; flow<=lastFlow; ++flow){
            int kRead = min(readIon[flow], maxKMer);
            int kRef  = min(refIon[flow],  maxKMer);
            ++counts[flow][kRead][kRef];
        }

        //cout << endl;


        if(numReads % 1000 == 0)
            cerr << setw(12) << numReads << '\r';

        //if(numReads > 1000)
        //    break;
    }

    cerr << endl << "done" << endl;

    //PrintCounts(counts, numFlows, maxKMer);

    WriteCountsHDF5(counts, numFlows, maxKMer);
}


