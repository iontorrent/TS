/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ThreadedVariantQueue.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#ifndef THREADEDVARIANTQUEUE_H
#define THREADEDVARIANTQUEUE_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include <utility>
#include "api/api_global.h"
#include "api/BamAux.h"
#include "api/BamConstants.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/SamReadGroup.h"
#include "api/SamReadGroupDictionary.h"
#include "api/SamSequence.h"
#include "api/SamSequenceDictionary.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <time.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <levmar.h>
#include <Variant.h>
#include <Fasta.h>
#include <Allele.h>
#include <Sample.h>
#include <AlleleParser.h>
#include <Utility.h>
#include <Genotype.h>
#include <BedReader.h>
#include <ResultData.h>

#include "peakestimator.h"
#include "stats.h"

#include "CountingSemaphore.h"
#include "HypothesisEvaluator.h"

#include "FlowDist.h"
#include "InputStructures.h"
#include "AlignmentAssist.h"
#include "HandleVariant.h"
#include "ExtendParameters.h"
#include "MiscUtil.h"
#include "VcfFormat.h"
#include "CandidateVariantGeneration.h"

using namespace std;
using namespace BamTools;
using namespace ion;

class variantThreadInfo; // forward declaration to avoid the loop

class MasterTracker {
  public:
    long int totalVariants;
    long int totalHPVariants;
    long int threadCounter;
    time_t master_timer;

    // parameters defining our threading pattern
    int max_records_per_thread;
    int max_threads_available;
//create two couting semaphores to keep track of max number of threads currently running and max number of total threads opened
    semaphore_t max_threads_sem;
    // threads write in order of creation assuming sorted vcf file
    semaphore_t output_queue_sem;
    // track our fun threads as they happen
    std::vector<pthread_t*> thread_tracker;

    MasterTracker() {
      totalVariants = 0;
      totalHPVariants = 0;
      threadCounter = 0;
      max_records_per_thread = 1000; // a default value
      max_threads_available = 1; // a default value
      init_semaphore(&max_threads_sem, 0);
      init_semaphore(&output_queue_sem, 0);
    };
    void WaitForFreeThread();
    void WaitForZeroThreads();
    void DeleteThreads();
    void Finalize();
    void StartJob(variantThreadInfo *variant_thread_job_spec, bool FINAL_JOB, int DEBUG);
};

// each individual job
class variantThreadInfo {
  public:
    variantThreadInfo(): threadID(0), records_in_thread(0), weight_of_thread(0), heartbeat_done(false), variantArray(NULL), parameters(NULL), global_context(NULL), outVCFStream(NULL), filterVCFStream(NULL), thread_master(NULL) {
    };
    variantThreadInfo(ofstream &outVCFFile, ofstream &filterVCFFile, ExtendParameters *inparameters, InputStructures *_global_context, MasterTracker &all_thread_master) {
      // new job goes on new thread ID
      all_thread_master.threadCounter++; // side effects bad(!)
      threadID = all_thread_master.threadCounter;

      variantArray = new vcf::Variant*[all_thread_master.max_records_per_thread];
      parameters = inparameters;
      outVCFStream = &outVCFFile;
      filterVCFStream = &filterVCFFile;
      global_context = _global_context;
      thread_master = &all_thread_master;
      records_in_thread = 0;
      weight_of_thread = 0;
      time(&local_timer); //when do I start this variant clock ticking
      heartbeat_done = false;

    };
    ~variantThreadInfo() {
      // destructor
      if (variantArray != NULL)
        delete[] variantArray;
      // this thread goes away when the job does (technically one instant after...)
      down(&thread_master->max_threads_sem);
    }
    void PushVariantOntoJob(vcf::Variant *variant) {
      variantArray[records_in_thread++] =  variant;
      // "weight of thread" = expected read depth ~ compute load
      weight_of_thread += CalculateWeightOfVariant(variant);
      
      HeartBeatIn(variant);
    }
    void HeartBeatOut(vcf::Variant *current_variant);
    void HeartBeatIn(vcf::Variant *variant) {
      if (!heartbeat_done) {
        // give me a little sign that you are alive and doing something with variants
        // natural to start when we start a new job
        double cur_sec = difftime(local_timer, thread_master->master_timer);
        cout << "Diastole: " << (variant)->sequenceName << " " << variant->position << " CurSec: " << cur_sec << endl;
        heartbeat_done = true;
      }
    }
    void WriteOutputAndCleanUp();
    void WriteVariants();
    void EchoThread();
    bool ReadyForAnotherVariant();
    void OpenThreadBamReader(BamTools::BamMultiReader &bamMultiReader);
    void WaitForMyNumberToComeUp();
    void TagNextThread();

    // stuff I'm tracking for my master...
    int threadID;
    int records_in_thread;
    int weight_of_thread;
    time_t local_timer;
    bool heartbeat_done;

    vcf::Variant ** variantArray;
    ExtendParameters * parameters;
    InputStructures * global_context;
    ofstream * outVCFStream;
    ofstream * filterVCFStream;
    MasterTracker *thread_master; // all hail the one true master
};


class VariantJobServer {
  public:

    MasterTracker all_thread_master;
    vcf::Variant * variant;
    bool isHotSpot;
    variantThreadInfo * variant_thread_job_spec;

    VariantJobServer() {
      variant = NULL;
      variant_thread_job_spec = NULL;
      isHotSpot = false;
      time(&all_thread_master.master_timer);
    };
    void PushCurVariantOntoJobs(ofstream &outVCFFile,
                                ofstream &filterVCFFile,
                                vcf::VariantCallFile &vcfFile,
                                InputStructures &global_context,
                                ExtendParameters *parameters);
    void KillMeNow(int DEBUG);
    void NewVariant(vcf::VariantCallFile &vcfFile);
    void SetupJobServer(ExtendParameters *parameters);
    ~VariantJobServer();
};


// the actual worker function
void *ProcessSetOfVariantsWorker(void *ptr);
// the guy that orchestrates the dance
void ThreadedVariantCaller(ofstream &outVCFFile, ofstream &filterVCFFile, ofstream &consensusFile, InputStructures &global_context, ExtendParameters *parameters);
void justProcessInputVCFCandidates(CandidateGenerationHelper &candidate_generator, ExtendParameters *parameters);

#endif // THREADEDVARIANTQUEUE_H
