/* Copyright (C) 2015 Life Technologies Corporation, a part of Thermo Fisher Scientific, Inc. All Rights Reserved. */

//! @file     Calibration.cpp
//! @ingroup  Calibration
//! @brief    Calibration. Executable to train algorithms for adjusting signal intensity and base calls.


#include "api/BamMultiReader.h"
#include "IonVersion.h"
#include "Utils.h"
#include "json/json.h"

#include "CalibrationHelper.h"
#include "HistogramCalibration.h"
#include "LinearCalibrationModel.h"

using namespace std;
using namespace BamTools;

void * CalibrationWorker(void *input);

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
    cout << "Calibration " << IonVersion::GetVersion() << "-" << IonVersion::GetRelease()
         << " (" << IonVersion::GetGitHash() << ")" << endl;
    cout << "Command line = " << command_line << endl;

    json["host_name"]    = my_host_name;
    json["start_time"]   = get_time_iso_string(analysis_start_time);
    json["version"]      = IonVersion::GetVersion() + "-" + IonVersion::GetRelease();
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

  time_t calibration_start_time;
  time(&calibration_start_time);

  pthread_mutex_init(&calib_context.read_mutex,  NULL);
  pthread_mutex_init(&calib_context.write_mutex, NULL);

  pthread_t worker_id[calib_context.num_threads];
  for (int worker = 0; worker < calib_context.num_threads; worker++)
  if (pthread_create(&worker_id[worker], NULL, CalibrationWorker, &calib_context)) {
    cerr << "Calibration ERROR: Problem starting thread" << endl;
    exit (EXIT_FAILURE);
  }

  for (int worker = 0; worker < calib_context.num_threads; worker++)
    pthread_join(worker_id[worker], NULL);

  pthread_mutex_destroy(&calib_context.read_mutex);
  pthread_mutex_destroy(&calib_context.write_mutex);

  time_t calibration_end_time;
  time(&calibration_end_time);


  //
  // Step 3. Create models, write output, and close modules
  //

  // HP histogram calibration
  if (master_histogram.CreateCalibrationModel())
    master_histogram.ExportModelToJson(calibration_json["HPHistogram"]);

  // Linear Model
  if (master_linear_model.CreateCalibrationModel())
    master_linear_model.ExportModelToJson(calibration_json["LinearModel"], "");


  // Transfer stuff from calibration context and close bam reader
  calib_context.Close(calibration_json["Calibration"]);

  time_t program_end_time;
  time(&program_end_time);

  calibration_json["Calibration"]["end_time"] = get_time_iso_string(program_end_time);
  calibration_json["Calibration"]["total_duration"] = (Json::Int)difftime(program_end_time,program_start_time);
  calibration_json["Calibration"]["calibration_duration"] = (Json::Int)difftime(calibration_end_time,calibration_start_time);

  SaveJson(calibration_json, calib_context.filename_json);
  return EXIT_SUCCESS;
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

  vector<DPTreephaser> treephaser_vector;
  for (unsigned int iFO=0; iFO < calib_context.flow_order_vector.size(); iFO++) {
    DPTreephaser      dpTreephaser(calib_context.flow_order_vector.at(iFO));
    treephaser_vector.push_back(dpTreephaser);
  }

  HistogramCalibration    hist_calibration_local(*calib_context.hist_calibration_master);
  LinearCalibrationModel  linear_cal_model_local(*calib_context.linear_model_master);


  // *** Process reads

  while (true) {

    // Step 1 *** load a number of reads from the BAM files

	hist_calibration_local.CleanSlate();
	linear_cal_model_local.CleanSlate();

	unsigned long num_useful_reads = 0;

    pthread_mutex_lock(&calib_context.read_mutex);

    bam_alignments.clear();
    BamAlignment new_alignment;
    bool have_alignment = true;

    // we may have an unsorted BAM with a large chunk of unmapped reads somewhere in the middle
    while((bam_alignments.size() < calib_context.num_reads_per_thread) and have_alignment) {

      have_alignment = calib_context.bam_reader.GetNextAlignment(new_alignment);
      if (have_alignment) {
        calib_context.num_reads_in_bam++;

        if (new_alignment.IsMapped()) {
          calib_context.num_mapped_reads++;

          if(new_alignment.MapQuality >= calib_context.min_mapping_qv) {
            calib_context.num_loaded_reads++;
            bam_alignments.push_back(new_alignment);
          }
        }
        else if (calib_context.load_unmapped) {
          calib_context.num_loaded_reads++;
          bam_alignments.push_back(new_alignment);
        }
      }
    }

    if ((not have_alignment) and (bam_alignments.size() == 0)) {
      pthread_mutex_unlock(&calib_context.read_mutex);
      return NULL;
    }

    pthread_mutex_unlock(&calib_context.read_mutex);


    // Step 2 *** Iterate over individual reads and extract information

    for (unsigned int iRead=0; iRead<bam_alignments.size(); iRead++) {

      // Unpack alignment related information & generate predictions
      read_alignment.UnpackReadInfo(&bam_alignments[iRead], calib_context);
      read_alignment.UnpackAlignmentInfo(calib_context, calib_context.debug);
      read_alignment.GeneratePredictions(treephaser_vector, calib_context);

      if (read_alignment.is_filtered)
        continue; // No need to waste time.

      // *** Pass read info to different modules so they can extract whatever info they need

      num_useful_reads++;

      hist_calibration_local.AddTrainingRead(read_alignment);
      linear_cal_model_local.AddTrainingRead(read_alignment);
    }


    // Step 3 *** After read processing finished, aggregate the collected information.

    pthread_mutex_lock(&calib_context.write_mutex);

    calib_context.num_useful_reads += num_useful_reads;
    calib_context.hist_calibration_master->AccumulateHistData(hist_calibration_local);
    calib_context.linear_model_master->AccumulateTrainingData(linear_cal_model_local);

    pthread_mutex_unlock(&calib_context.write_mutex);

  } // -- end while loop
}
