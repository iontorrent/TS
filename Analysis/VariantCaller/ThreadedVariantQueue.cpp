/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ThreadedVariantQueue.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#include "ThreadedVariantQueue.h"


// variantThreadInfo specifies the job a single thread is doing

void variantThreadInfo::OpenThreadBamReader(BamTools::BamMultiReader &bamMultiReader) {
    if (!bamMultiReader.Open(parameters->bams)) {
        cerr << "FATAL ERROR: fail to open bam file/files"  << endl;
        exit(-1);
    }
    bamMultiReader.LocateIndexes();

}

bool variantThreadInfo::ReadyForAnotherVariant() {
    bool test_for_max_variants = (records_in_thread < thread_master->max_records_per_thread);
    // fudge factor for expected read depth per variant
    // in case our variants decide to be a lot "heavier" than our threading specifies, adapt compute load
    const int EXPECTED_DEPTH_OF_VARIANT = 200;
    bool test_for_max_weight = (weight_of_thread< (thread_master->max_records_per_thread * EXPECTED_DEPTH_OF_VARIANT));

    return(test_for_max_variants & test_for_max_weight);
}

void variantThreadInfo::EchoThread() {
    if (global_context->DEBUG)
        fprintf(stdout, "Processing ThreadID = %d maxRecords in Thread = %d \n", threadID, records_in_thread);
}

void variantThreadInfo::HeartBeatOut(vcf::Variant *current_variant) {
    if (heartbeat_done) {
        // when we write the heartbeat variant out to disk
        time_t final_timer;
        time(&final_timer);
        double sec_time;
        sec_time = difftime(final_timer, local_timer);
        double cur_time = difftime(final_timer, thread_master->master_timer);
        cout << "Systole: " << current_variant->sequenceName << " " << current_variant->position << " ThreadTime: " << sec_time << " Variants " << records_in_thread << " CurSec: " << cur_time << endl;
        heartbeat_done = false;
    }
}

void variantThreadInfo::WriteVariants() {
    int counter = 0;
    vcf::Variant * current_variant;
//  bool heartbeat_done = false;

    while (counter < records_in_thread) {

        current_variant = variantArray[counter++];

        if (current_variant == NULL) {
            fprintf(stdout, "Variant Array in thread returned null for Counter = %d \n", counter - 1);
            exit(-1);
        }
        HeartBeatOut(current_variant);

        if (current_variant->isFiltered && !current_variant->isHotSpot && parameters->my_controls.suppress_no_calls)
            (*filterVCFStream) << *current_variant << endl;
        else
            (*outVCFStream) << *current_variant << endl;
        delete current_variant;
    }
}

void variantThreadInfo::WaitForMyNumberToComeUp() {
    while (thread_master->output_queue_sem.count < (threadID - 1)) {
        pthread_cond_wait(&(thread_master->output_queue_sem.cond), &(thread_master->output_queue_sem.mutex));
    }
}

void variantThreadInfo::TagNextThread() {
    //up the output_queue semaphore to next thread id, so the next thread can start writing to VCF file
    if (global_context->DEBUG)
        fprintf(stdout, "trying up output_queue_sem \n");
    up(&thread_master->output_queue_sem);

    if (global_context->DEBUG)
        fprintf(stdout, "finished up output_queue_sem \n");

    if (global_context->DEBUG)
        fprintf(stdout, "Finished write to VCF file from thead ID = %d, output queue sem count = %d , max threads sem count = %d\n", threadID, thread_master->output_queue_sem.count, thread_master->max_threads_sem.count);

}

void variantThreadInfo::WriteOutputAndCleanUp() {
    //now check if this thread is ready to write to VCF, if so write and exit, else sleep
    WaitForMyNumberToComeUp();

    if (thread_master->output_queue_sem.count == (threadID - 1)) {
        //loop thru all the VCF records and write to appropriate VCF output file
        if (global_context->DEBUG)
            fprintf(stdout, "Starting to write to VCF file from thread ID = %d, total threads sem count = %d, max threads sem count = %d \n", threadID, thread_master->output_queue_sem.count, thread_master->max_threads_sem.count);
        WriteVariants();
    }

    TagNextThread();
}

// Master tracker tracks our universe of threads

void MasterTracker::WaitForFreeThread() {
    //TO-DO
    //if not sleep for 5 secs or until the num threads becomes less than MAX THREADS
    while (max_threads_sem.count >= max_threads_available) {
        pthread_cond_wait(&(max_threads_sem.cond), &(max_threads_sem.mutex));
    }
}

void MasterTracker::DeleteThreads() {
    std::vector<pthread_t*>::iterator thread_vector_iterator;
    for (thread_vector_iterator = thread_tracker.begin(); thread_vector_iterator < thread_tracker.end(); thread_vector_iterator++) {
        delete *thread_vector_iterator;
    }
}

void MasterTracker::StartJob(variantThreadInfo *variant_thread_job_spec, bool FINAL_JOB, int DEBUG) {

    if (DEBUG)
        fprintf(stdout, "Starting Thread ID = %d, varint count = %d, max threads sem count = %d \n", (int)threadCounter, variant_thread_job_spec->records_in_thread, max_threads_sem.count);

    pthread_t *thread;
    thread = new pthread_t();
    int thread_ret_value = pthread_create(thread, NULL, &ProcessSetOfVariantsWorker, (void *)variant_thread_job_spec);
    if (thread_ret_value) {
        fprintf(stderr, "FATAL: Unable to create thread - Return Value = %d \n", thread_ret_value);
        exit(-1);
    }
    //pthread join
    thread_tracker.push_back(thread);

    if (DEBUG)
        fprintf(stdout, "Starting Thread ID = %d, trying to up max_threads_sem \n", (int)threadCounter);

    up(&max_threads_sem);
    //pthread_join(*thread, NULL);

    if (DEBUG)
        fprintf(stdout, "Finished starting Thread ID = %d, readPairs count = %d, max threads sem count = %d \n", (int)threadCounter, variant_thread_job_spec->records_in_thread, max_threads_sem.count);
    if (DEBUG & FINAL_JOB)
        fprintf(stdout, "Finished starting Final Thread ID = %d, readPairs count = %d, max threads sem count = %d \n", (int)threadCounter, variant_thread_job_spec->records_in_thread, max_threads_sem.count);
}

void MasterTracker::WaitForZeroThreads() {
    //join all the threads
    while (max_threads_sem.count > 0) {
        pthread_cond_wait(&(max_threads_sem.cond), &(max_threads_sem.mutex));
    }
}

void MasterTracker::Finalize() {
    WaitForZeroThreads();
    //delete all the thread objects created
    DeleteThreads();
}

// Down here are the two important functions

void *ProcessSetOfVariantsWorker(void *ptr) {

    variantThreadInfo       *variant_thread_ptr = static_cast<variantThreadInfo *>(ptr);
    PersistingThreadObjects  thread_objects(*(variant_thread_ptr->global_context));

    //BamTools::BamMultiReader bamMultiReader;

    variant_thread_ptr->OpenThreadBamReader(thread_objects.bamMultiReader);
    variant_thread_ptr->EchoThread();

    string prevSequenceName = "";
    string currentSequenceName = "";
    //string local_contig_sequence = "";
    int variant_counter = 0;
    while (variant_counter < variant_thread_ptr->records_in_thread) {

        vcf::Variant **current_variant = &(variant_thread_ptr->variantArray[variant_counter++]);


        if (*current_variant == NULL) {
            fprintf(stdout, "Variant Array in thread returned null for Variant Counter = %d \n", variant_counter - 1);
            exit(-1);
        }
        currentSequenceName = (*current_variant)->sequenceName;
        if (currentSequenceName.compare(prevSequenceName) != 0) {

            prevSequenceName = currentSequenceName;
            thread_objects.local_contig_sequence = variant_thread_ptr->global_context->ReturnReferenceContigSequence(current_variant);

        }

        if (thread_objects.local_contig_sequence.empty()) {
            cerr << "FATAL: Reference sequence for Contig " << (*current_variant)->sequenceName << " , not found in reference fasta file " << endl;
            exit(-1);
        }
        // separate queuing of variants from >actual work< of calling variants
        DoWorkForOneVariant(thread_objects, current_variant, variant_thread_ptr->parameters, variant_thread_ptr->global_context);
    }

    //bamReader.close();
    variant_thread_ptr->WriteOutputAndCleanUp();

    delete variant_thread_ptr; // also calls destructor which tells semaphore to decrement available thread-count
    pthread_exit((void *)0);
}



void VariantJobServer::NewVariant(vcf::VariantCallFile &vcfFile) {
    variant = new vcf::Variant(vcfFile);  //candidate_generator.parser->variantCallFile
    isHotSpot = false;
};


void VariantJobServer::PushCurVariantOntoJobs(ofstream &outVCFFile,
        ofstream &filterVCFFile,
        vcf::VariantCallFile &vcfFile,
        InputStructures &global_context,
        ExtendParameters *parameters) {
    variant->isHotSpot = isHotSpot;  // see TODO above


    //@TODO: the logic here is slightly more complex than optimal
    // need a new job, create one
    if (variant_thread_job_spec == NULL) {
        variant_thread_job_spec = new variantThreadInfo(outVCFFile, filterVCFFile, parameters, &global_context, all_thread_master);
    }
   // if job can take another variant, go for it
    if (variant_thread_job_spec->ReadyForAnotherVariant()) {
        variant_thread_job_spec->PushVariantOntoJob(variant);
    }
    else {
      // ready to fire off the job
        //check for the number of running threads and if less than MAX create new thread and pass the job
        all_thread_master.WaitForFreeThread();

        if (all_thread_master.max_threads_sem.count < all_thread_master.max_threads_available) {
            all_thread_master.StartJob(variant_thread_job_spec, false, global_context.DEBUG);
// generate a new job to handle the variant we were passed
            variant_thread_job_spec = NULL;
            variant_thread_job_spec = new variantThreadInfo(outVCFFile, filterVCFFile, parameters, &global_context, all_thread_master);
            variant_thread_job_spec->PushVariantOntoJob(variant);

        }
        else {
            cerr << "FATAL: thought I had a thread available but I didn't" << endl;
            exit(-1);
        }
        //Added for 3.6.1 Patch to clear thread resource after they terminate. Need to move this to use thread pools instead.
        RefreshThreadPool();

    }
// get a new variant to handle the next case
    NewVariant(vcfFile);
}

void VariantJobServer::RefreshThreadPool() {
    if (all_thread_master.thread_tracker.size() > (size_t) 3*all_thread_master.max_threads_available) {
        std::vector<pthread_t*>::iterator thread_itr = all_thread_master.thread_tracker.begin();
        int counter = 0;
        int rc = 0;
        while (counter < all_thread_master.max_threads_available && thread_itr != all_thread_master.thread_tracker.end()) {
            rc = pthread_join(**thread_itr, NULL);
            if (rc) {
                cerr << "FATAL: pthread_join returned non-zero return code " << rc << endl;
                exit(-1);
            }
            //remove the thread reference since it has completed successfully to release it's resources
            delete *thread_itr;
            thread_itr = all_thread_master.thread_tracker.erase(thread_itr);
            counter++;
        }
    }
}

void VariantJobServer::KillMeNow(int DEBUG) {
    // just one last job before I die
    if (variant_thread_job_spec != NULL) {
        all_thread_master.StartJob(variant_thread_job_spec, true, DEBUG);
        variant_thread_job_spec = NULL;
    }
    // left over variant
    if (variant != NULL) {
        delete variant; // not used, terminate out
        variant = NULL;
    }
    // exit gracefully
    all_thread_master.Finalize();
}

void VariantJobServer::SetupJobServer(ExtendParameters *parameters) {
    all_thread_master.max_records_per_thread = parameters->program_flow.nVariantsPerThread;
    all_thread_master.max_threads_available = parameters->program_flow.nThreads;

}

VariantJobServer::~VariantJobServer() {
    if (variant_thread_job_spec != NULL) {
        delete variant_thread_job_spec;
    }
    if (variant != NULL)
        delete variant;

}

// my_job_server consumes a stream of variants generated in some way
// candy_wrapper produces a new variant each time it is called
// the idea is to isolate consumers and producers cleanly
// so that we don't need to know anything at all about where these variants are coming from
void ThreadedVariantCaller(ofstream &outVCFFile, ofstream &filterVCFFile,  InputStructures &global_context, ExtendParameters *parameters) {

  // set up consumer of variants
    VariantJobServer my_job_server;
    my_job_server.SetupJobServer(parameters);

    // set up producer of variants
    CandidateStreamWrapper candy_wrapper;
    candy_wrapper.SetUp(outVCFFile, filterVCFFile, global_context, parameters);

    my_job_server.NewVariant(candy_wrapper.candidate_generator.parser->variantCallFile);

    while (candy_wrapper.ReturnNextVariantStream(my_job_server.isHotSpot,my_job_server.variant, parameters)) {
        my_job_server.PushCurVariantOntoJobs(outVCFFile, filterVCFFile, candy_wrapper.candidate_generator.parser->variantCallFile, global_context, parameters);
    }
    
    my_job_server.KillMeNow(global_context.DEBUG);

}


