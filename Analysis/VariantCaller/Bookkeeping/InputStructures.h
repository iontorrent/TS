/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     InputStructures.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef INPUTSTRUCTURES_H
#define INPUTSTRUCTURES_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include "api/api_global.h"
#include "api/BamAux.h"
#include "api/BamConstants.h"
#include "api/BamReader.h"
#include "api/BamMultiReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/SamReadGroup.h"
#include "api/SamReadGroupDictionary.h"
#include "api/SamSequence.h"
#include "api/SamSequenceDictionary.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <levmar.h>
#include <Variant.h>
#include <errno.h>

#include "MiscUtil.h"

#include <time.h>
#include "BaseCallerUtils.h"
#include "../Splice/ErrorMotifs.h"
#include "ExtendParameters.h"

using namespace std;
using namespace BamTools;
using namespace ion;

// TODO: Check if merging of Parameters and InputStructures makes sense
// No: it does not make sense - Parameters keeps input values, InputStructures keeps active data provoked by those values

class LiveFiles{
  public:
    ofstream outVCFFile;
    ofstream filterVCFFile;
    ofstream consensusFile;
    
    string file_base_name;
    time_t start_time;
    
    LiveFiles();
    void SetBaseName(ExtendParameters &parameters);
    void ActiveConsensus(ExtendParameters &parameters);
    void ActiveFilterVCF(ExtendParameters &parameters);
    void ActiveOutputVCF(ExtendParameters &parameters);
    void ActivateFiles(ExtendParameters &parameters);
    void ActiveDiagnostic(ExtendParameters &parameters);
    void ActiveOutputDir(ExtendParameters &parameters);
    void ShutDown();
};

//Input Structures
class InputStructures {
  public:

    map<string,string> reference_contigs;

    vector<string> sequences;
    vector<string> refSequences;
    map<string,int> sequence_to_bam_ref_id;
    vector<string> sampleList;
    map<string, string> readGroupToSampleNames;

    FlowOrder      treePhaserFlowOrder;
    string         flowOrder ;
    uint16_t       nFlows;
    vector<string> bamFlowOrderVector;
    uint16_t       min_map_qv;

    bool flowSigPresent;
    string flowKey;
    KeySequence key;
    
    int DEBUG;
    BamMultiReader bamMultiReader;
    SamHeader samHeader;
        
    TIonMotifSet ErrorMotifs;

    
    InputStructures();
    //~InputStructures();
    void BringUpReferenceData(ExtendParameters &parameters);
    string & ReturnReferenceContigSequence(vcf::Variant ** current_variant);
    void ShiftLocalBamReaderToCorrectBamPosition(BamTools::BamMultiReader &local_bamReader, vcf::Variant **current_variant);
    void bam_initialize(vector<string> bams /*, string inputBAMIndex*/);
    void read_fasta(string, map<string, string> &);
    void read_error_motifs(string & fname){ErrorMotifs.load_from_file(fname.c_str());};

};




#endif //INPUTSTRUCTURES_H
