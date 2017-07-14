/* Copyright (C) 2015 Life Technologies Corporation, a part of Thermo Fisher Scientific, Inc. All Rights Reserved. */

//! @file     Calibration.cpp
//! @ingroup  Calibration
//! @brief    Calibration. Executable to train algorithms for adjusting signal intensity and base calls.


#include "api/BamMultiReader.h"
#include <algorithm>
#include <random>
#include "IonVersion.h"
#include "Utils.h"
#include "json/json.h"

#include "CalibrationHelper.h"
#include "HistogramCalibration.h"
#include "LinearCalibrationModel.h"

using namespace std;
using namespace BamTools;

void * CalibrationWorker(void *input);
int ExecuteThreadedCalibrationTraining(CalibrationContext &calib_context);

// ----------------------------------------------------------------
//! @brief    Write Calibration startup info to json structure.
//! @ingroup  Calibration

void DumpStartingStateOfProgram (int argc, const char *argv[], time_t analysis_start_time, Json::Value &json)
{
  char my_host_name[128] = { 0 };
  gethostname (my_host_name, 128);
  string command_line = argv[0];
  for (int i = 1; i < argc; i++) {
    command_line += " ";
    command_line += argv[i];
  }

  cout << "---------------------------------------------------------------------" << endl;
  cout << "Calibration " << IonVersion::GetVersion() << "." << IonVersion::GetRelease()
       << " (" << IonVersion::GetGitHash() << ") (" << IonVersion::GetBuildNum() << ")" << endl;
  cout << "Command line = " << command_line << endl;

  json["host_name"]    = my_host_name;
  json["start_time"]   = get_time_iso_string(analysis_start_time);
  json["version"]      = IonVersion::GetVersion() + "." + IonVersion::GetRelease();
  json["git_hash"]     = IonVersion::GetGitHash();
  json["build_number"] = IonVersion::GetBuildNum();
  json["command_line"] = command_line;
}


// ------------------------------------------------------------------

void SaveJson(const Json::Value & json, const string& filename_json)
{
  ofstream out(filename_json.c_str(), ios::out);
  if (out.good())
    out << json.toStyledString();
  else
    cerr << "Unable to write JSON file " << filename_json;
}

// ------------------------------------------------------------------

void PrintHelp_CalModules()
{
  HistogramCalibration::PrintHelp_Training();
  cout << endl;
  LinearCalibrationModel::PrintHelp_Training();
  cout << "---------------------------------------------------------------------" << endl;
  exit(EXIT_SUCCESS);
}


// =======================================================================

int main (int argc, const char *argv[])
{
  time_t program_start_time;
  time(&program_start_time);
  Json::Value calibration_json(Json::objectValue);
  DumpStartingStateOfProgram (argc,argv,program_start_time, calibration_json["Calibration"]);

  //
  // Step 1. Process command line options
  //

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);

  CalibrationContext calib_context;
  if (not calib_context.InitializeFromOpts(opts)){
    PrintHelp_CalModules();
  }

  HistogramCalibration master_histogram(opts, calib_context);
  calib_context.hist_calibration_master = &master_histogram;

  LinearCalibrationModel master_linear_model(opts, calib_context);
  calib_context.linear_model_master = &master_linear_model;

  opts.CheckNoLeftovers();

  //
  // Step 2. Execute threaded calibration
  //
  int calibration_thread_time = 0;

  if (calib_context.successive_fit) {

    // first train linear model
    if (master_linear_model.DoTraining()) {
      int l_thread_time = 0;
      for (int i_iteration=0; i_iteration<calib_context.num_train_iterations; i_iteration++) {
        cout << " -Training Iteration " << i_iteration+1;
        l_thread_time = ExecuteThreadedCalibrationTraining(calib_context);

        // Activate master linear model after every round of training
        master_linear_model.CreateCalibrationModel(false); // make linear model
        master_linear_model.SetModelGainsAndOffsets(); // expand for use in basecalling

        calibration_thread_time += l_thread_time;
        calib_context.bam_reader.Rewind(); // reset all files for another pass
        cout << " Duration = " << l_thread_time << endl;
      }
    }

    // Then apply it during polish model training
    if (master_histogram.DoTraining()) {
      calib_context.local_fit_linear_model = false;
      calib_context.local_fit_polish_model = true;
      calibration_thread_time += ExecuteThreadedCalibrationTraining(calib_context);
    }
  }
  else {
    // Single pass in which both models are fit jointly
    calibration_thread_time=ExecuteThreadedCalibrationTraining(calib_context);
  }


  //
  // Step 3. Create models, write output, and close modules
  //

  // Linear Model
  if (master_linear_model.CreateCalibrationModel())
    master_linear_model.ExportModelToJson(calibration_json["LinearModel"], "");

  // HP histogram calibration
  if (master_histogram.CreateCalibrationModel())
    master_histogram.ExportModelToJson(calibration_json["HPHistogram"]);


  // Transfer stuff from calibration context and close bam reader
  calib_context.Close(calibration_json["Calibration"]);

  time_t program_end_time;
  time(&program_end_time);

  calibration_json["Calibration"]["end_time"] = get_time_iso_string(program_end_time);
  calibration_json["Calibration"]["total_duration"] = (Json::Int)difftime(program_end_time,program_start_time);
  calibration_json["Calibration"]["calibration_duration"] = (Json::Int)calibration_thread_time;

  SaveJson(calibration_json, calib_context.filename_json);
  return EXIT_SUCCESS;
}

// --------------------------------------------------------------------------

int ExecuteThreadedCalibrationTraining(CalibrationContext &calib_context){

  time_t calibration_start_time;
  time(&calibration_start_time);

  calib_context.num_model_reads  = 0;
  calib_context.num_model_writes = 0;
  calib_context.wait_to_read_model  = false;
  calib_context.wait_to_write_model = true;

  pthread_mutex_init(&calib_context.read_mutex,  NULL);
  pthread_mutex_init(&calib_context.write_mutex, NULL);
  pthread_cond_init(&calib_context.model_read_cond, NULL);
  pthread_cond_init(&calib_context.model_write_cond, NULL);

  pthread_t worker_id[calib_context.num_threads];
  for (unsigned int worker = 0; worker < calib_context.num_threads; worker++)
    if (pthread_create(&worker_id[worker], NULL, CalibrationWorker, &calib_context)) {
      cerr << "Calibration ERROR: Problem starting thread" << endl;
      exit (EXIT_FAILURE);
    }

  for (unsigned int worker = 0; worker < calib_context.num_threads; worker++)
    pthread_join(worker_id[worker], NULL);

  pthread_mutex_destroy(&calib_context.read_mutex);
  pthread_mutex_destroy(&calib_context.write_mutex);
  pthread_cond_destroy(&calib_context.model_read_cond);
  pthread_cond_destroy(&calib_context.model_write_cond);

  time_t calibration_end_time;
  time(&calibration_end_time);
  return difftime(calibration_end_time,calibration_start_time);
}



// --------------------------------------------------------------------------


void * CalibrationWorker(void *input)
{

  CalibrationContext& calib_context = *static_cast<CalibrationContext*>(input);

  // *** Initialize Modules

  vector<BamAlignment> bam_alignments;
  bam_alignments.reserve(calib_context.num_reads_per_thread);
  ReadAlignmentInfo read_alignment;
  read_alignment.SetSize(calib_context.max_num_flows);
  bool update_bam_stats = (calib_context.bam_reader.NumPasses() == 0);

  // Random number generator & index vector
  std::default_random_engine rand_engine(calib_context.rand_seed);
  vector<unsigned> index_vec(calib_context.num_reads_per_thread);
  for (unsigned int id=0; id < calib_context.num_reads_per_thread; ++id)
    index_vec[id] = id;

  vector<DPTreephaser> treephaser_vector;
  for (unsigned int iFO=0; iFO < calib_context.flow_order_vector.size(); iFO++) {
    DPTreephaser      dpTreephaser(calib_context.flow_order_vector.at(iFO));
    treephaser_vector.push_back(dpTreephaser);
  }

  HistogramCalibration    hist_calibration_local(*calib_context.hist_calibration_master);
  LinearCalibrationModel  linear_cal_model_local(*calib_context.linear_model_master); // Training data for current batch
  LinearCalibrationModel  linear_model_cal_sim  (*calib_context.linear_model_master); // state of master to date (changing for blind)


  // *** Process reads

  while (true) {

    hist_calibration_local.CleanSlate();
    linear_cal_model_local.CleanSlate();
    unsigned long num_useful_reads = 0;

    // Step 0 *** Wait for permission to read most recent master model
    //            Blind fit only! Non-blind is limited to one training iteration.

    if (calib_context.blind_fit and calib_context.local_fit_linear_model){
      pthread_mutex_lock(&calib_context.write_mutex);

      while(calib_context.wait_to_read_model)
        pthread_cond_wait (&calib_context.model_read_cond, &calib_context.write_mutex);

      linear_model_cal_sim.CopyTrainingData(*calib_context.linear_model_master);
      calib_context.num_model_reads++;
      if (calib_context.num_model_reads == calib_context.num_threads){
        calib_context.num_model_reads = 0;
        calib_context.wait_to_read_model = true;
        calib_context.wait_to_write_model = false;
        pthread_cond_broadcast(&calib_context.model_write_cond);
      }

      pthread_mutex_unlock(&calib_context.write_mutex);

      // Activate local copy of linear model
      linear_model_cal_sim.CreateCalibrationModel(false);
      linear_model_cal_sim.SetModelGainsAndOffsets();
      if (not update_bam_stats)
        std::shuffle ( index_vec.begin(), index_vec.end(), rand_engine);
    }

    // Step 1 *** load a number of reads from the BAM files

    bam_alignments.clear();
    BamAlignment new_alignment;
    bool have_alignment = true;

    pthread_mutex_lock(&calib_context.read_mutex);

    // we may have an unsorted BAM with a large chunk of unmapped reads somewhere in the middle
    while((bam_alignments.size() < calib_context.num_reads_per_thread) and have_alignment) {

      have_alignment = calib_context.bam_reader.GetNextAlignmentCore(new_alignment);
      // Only log read stats during first pass through BAM
      if (have_alignment) {
        if (update_bam_stats)
          calib_context.num_reads_in_bam++;

        if (new_alignment.IsMapped()) {
          if (update_bam_stats)
            calib_context.num_mapped_reads++;

          if(new_alignment.MapQuality >= calib_context.min_mapping_qv) {
            if (update_bam_stats)
              calib_context.num_loaded_reads++;
            bam_alignments.push_back(new_alignment);
          }
        }
        else if (calib_context.load_unmapped) {
          if (update_bam_stats)
            calib_context.num_loaded_reads++;
          bam_alignments.push_back(new_alignment);
        }
      }
    }

    if ((not have_alignment) and (bam_alignments.size() == 0)) {

      pthread_mutex_unlock(&calib_context.read_mutex);

      pthread_mutex_lock(&calib_context.write_mutex);
      calib_context.num_model_writes++;
      if (calib_context.num_model_writes == calib_context.num_threads){
        calib_context.num_model_writes = 0;
        calib_context.wait_to_read_model = false;
        calib_context.wait_to_write_model = true;
        pthread_cond_broadcast(&calib_context.model_read_cond);
      }
      pthread_mutex_unlock(&calib_context.write_mutex);

      return NULL;
    }

    pthread_mutex_unlock(&calib_context.read_mutex);


    // Step 2 *** Iterate over individual reads and extract information

    for (unsigned int idx=0; idx<bam_alignments.size(); idx++) {

      unsigned int iRead = index_vec.at(idx);
      if (iRead >= bam_alignments.size())
        continue;

      // Unpack alignment related information & generate predictions
      bam_alignments[iRead].BuildCharData();
      read_alignment.UnpackReadInfo(&bam_alignments[iRead], treephaser_vector, calib_context);
      read_alignment.UnpackAlignmentInfo(calib_context);
      read_alignment.GeneratePredictions(treephaser_vector, linear_model_cal_sim);

      if (read_alignment.is_filtered)
        continue; // No need to waste time.

      // *** Pass read info to different modules so they can extract whatever info they need

      num_useful_reads++;
      if (calib_context.local_fit_polish_model)
        hist_calibration_local.AddTrainingRead(read_alignment, calib_context);

      if (calib_context.local_fit_linear_model){
        if (calib_context.blind_fit)
          linear_cal_model_local.AddBlindTrainingRead(read_alignment,linear_model_cal_sim);
        else
          linear_cal_model_local.AddTrainingRead(read_alignment,linear_model_cal_sim);
      }
    }


    // Step 3 *** After read processing finished, aggregate the collected information.

    pthread_mutex_lock(&calib_context.write_mutex);

    if (calib_context.blind_fit and calib_context.local_fit_linear_model){
      while(calib_context.wait_to_write_model)
        pthread_cond_wait (&calib_context.model_write_cond, &calib_context.write_mutex);

      calib_context.num_model_writes++;
    }

    if (update_bam_stats)
      calib_context.num_useful_reads += num_useful_reads;
    if (calib_context.local_fit_polish_model)
      calib_context.hist_calibration_master->AccumulateHistData(hist_calibration_local);
    if (calib_context.local_fit_linear_model)
      calib_context.linear_model_master->AccumulateTrainingData(linear_cal_model_local);

    if (calib_context.num_model_writes == calib_context.num_threads){
      calib_context.num_model_writes = 0;
      calib_context.wait_to_read_model = false;
      calib_context.wait_to_write_model = true;
      pthread_cond_broadcast(&calib_context.model_read_cond);
    }

    pthread_mutex_unlock(&calib_context.write_mutex);

  } // -- end while loop
}
