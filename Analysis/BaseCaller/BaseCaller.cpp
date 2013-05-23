/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//! @defgroup BaseCaller BaseCaller executable
//! @brief    BaseCaller executable is responsible for phasing estimation, dephasing, filtering, and QV calculation

//! @file     BaseCaller.cpp
//! @ingroup  BaseCaller
//! @brief    BaseCaller main source

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <iomanip>
#include <algorithm>
#include <string>

#include "json/json.h"
#include "MaskSample.h"
#include "LinuxCompat.h"
#include "IonErr.h"
#include "OptArgs.h"
#include "IonVersion.h"
#include "file-io/ion_util.h"
#include "Utils.h"
#include "RawWells.h"

#include "BarcodeDatasets.h"
#include "BarcodeClassifier.h"
#include "OrderedDatasetWriter.h"
#include "TreephaserSSE.h"
#include "PhaseEstimator.h"
#include "PerBaseQual.h"
#include "BaseCallerFilters.h"
#include "BaseCallerMetricSaver.h"
#include "BaseCallerRecalibration.h"
#include "RecalibrationModel.h"

using namespace std;


//! @brief    Information needed by BaseCaller worker threads
//! @ingroup  BaseCaller

struct BaseCallerContext {

  // General run parameters
  string                    run_id;                 //!< Run ID string, prepended to each read name
  string                    keynormalizer;          //!< Name of selected key normalization algorithm
  string                    dephaser;               //!< Name of selected dephasing algorithm
  string                    filename_wells;         //!< Filename of the input wells file
  int                       chip_size_y;            //!< Chip height in wells
  int                       chip_size_x;            //!< Chip width in wells
  int                       region_size_y;          //!< Wells hdf5 dataset chunk height
  int                       region_size_x;          //!< Wells hdf5 dataset chunk width
  ion::FlowOrder            flow_order;             //!< Flow order object, also stores number of flows
  vector<KeySequence>       keys;                   //!< Info about key sequences in use by library and TFs
  string                    flow_signals_type;      //!< The flow signal type: "default" - Normalized and phased, "wells" - Raw values (unnormalized and not dephased), "key-normalized" - Key normalized and not dephased, "adaptive-normalized" - Adaptive normalized and not dephased, and "unclipped" - Normalized and phased but unclipped.
  string                    output_directory;       //!< Root directory for all output files
  int                       block_row_offset;       //!< Offset added to read names
  int                       block_col_offset;       //!< Offset added to read names
  int                       extra_trim_left;        //!< Number of additional insert bases past key and barcode to be trimmed
  bool                      process_tfs;            //!< If set to false, TF-related BAM will not be generated
  int                       windowSize;             //!< Normalization window size

  // Important outside entities accessed by BaseCaller
  Mask                      *mask;                  //!< Beadfind and filtering outcomes for wells
  BaseCallerFilters         *filters;               //!< Filter configuration and stats
  PhaseEstimator            estimator;              //!< Phasing estimation results
  PerBaseQual               quality_generator;      //!< Base phred quality value generator
  vector<int>               class_map;              //!< What to do with each well
  BaseCallerMetricSaver     *metric_saver;          //!< Saves requested metrics to an hdf5
  BarcodeClassifier         *barcodes;              //!< Barcode detection and trimming
  BaseCallerRecalibration   recalibration;          //!< Base call and signal adjustment algorithm
  RecalibrationModel        recalModel;             //!< Model estimation of simulated predictions and observed measurements

  // Threaded processing
  pthread_mutex_t           mutex;                  //!< Shared read/write mutex for BaseCaller worker threads
  int                       next_region;            //!< Number of next region that needs processing by a worker
  int                       next_begin_x;           //!< Starting X coordinate of next region
  int                       next_begin_y;           //!< Starting Y coordinate of next region

  // Basecalling results saved here
  OrderedDatasetWriter      lib_writer;                 //!< Writer object for library BAMs
  OrderedDatasetWriter      tf_writer;                  //!< Writer object for test fragment BAMs
  set<unsigned int>         unfiltered_set;             //!< Indicates which wells are to be saved to unfiltered BAMs
  OrderedDatasetWriter      unfiltered_writer;          //!< Writer object for unfiltered BAMs for a random subset of library reads
  OrderedDatasetWriter      unfiltered_trimmed_writer;  //!< Writer object for unfiltered trimmed BAMs for a random subset of library reads
};


void * BasecallerWorker(void *input);



//! @brief    Print BaseCaller version with figlet.
//! @ingroup  BaseCaller

void BaseCallerSalute()
{
  char banner[256];
  sprintf (banner, "/usr/bin/figlet -m0 BaseCaller %s 2>/dev/null", IonVersion::GetVersion().c_str());
  if (system (banner))
    fprintf (stdout, "BaseCaller %s\n", IonVersion::GetVersion().c_str()); // figlet did not execute
}

//! @brief    Print BaseCaller startup info, also write it to json structure.
//! @ingroup  BaseCaller

void DumpStartingStateOfProgram (int argc, const char *argv[], time_t analysis_start_time, Json::Value &json)
{
  char my_host_name[128] = { 0 };
  gethostname (my_host_name, 128);
  string command_line;
  printf ("\n");
  printf ("Hostname = %s\n", my_host_name);
  printf ("Start Time = %s", ctime (&analysis_start_time));
  printf ("Version = %s-%s (%s) (%s)\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
      IonVersion::GetSvnRev().c_str(), IonVersion::GetBuildNum().c_str());
  printf ("Command line = ");
  for (int i = 0; i < argc; i++) {
    if (i)
      command_line += " ";
    command_line += argv[i];
    printf ("%s ", argv[i]);
  }
  printf ("\n");
  fflush (NULL);

  json["host_name"] = my_host_name;
  json["start_time"] = get_time_iso_string(analysis_start_time);
  json["version"] = IonVersion::GetVersion() + "-" + IonVersion::GetRelease().c_str();
  json["svn_revision"] = IonVersion::GetSvnRev();
  json["build_number"] = IonVersion::GetBuildNum();
  json["command_line"] = command_line;
}


//! @brief    Print BaseCaller usage.
//! @ingroup  BaseCaller

void PrintHelp()
{
  printf ("\n");
  printf ("Usage: BaseCaller [options] --input-dir=DIR\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -h,--help                             print this help message and exit\n");
  printf ("  -v,--version                          print version and exit\n");
  printf ("  -i,--input-dir             DIRECTORY  input files directory [required option]\n");
  printf ("     --wells                 FILE       input wells file [input-dir/1.wells]\n");
  printf ("     --mask                  FILE       input mask file [input-dir/analysis.bfmask.bin]\n");
  printf ("  -o,--output-dir            DIRECTORY  results directory [current dir]\n");
  printf ("     --lib-key               STRING     library key sequence [TCAG]\n");
  printf ("     --tf-key                STRING     test fragment key sequence [ATCG]\n");
  printf ("     --flow-order            STRING     flow order [retrieved from wells file]\n");
  printf ("     --run-id                STRING     read name prefix [hashed input dir name]\n");
  printf ("  -n,--num-threads           INT        number of worker threads [2*numcores]\n");
  printf ("  -f,--flowlimit             INT        basecall only first n flows [all flows]\n");
  printf ("  -r,--rows                  INT-INT    basecall only a range of rows [all rows]\n");
  printf ("  -c,--cols                  INT-INT    basecall only a range of columns [all columns]\n");
  printf ("     --region-size           INTxINT    wells processing chunk size [50x50]\n");
  printf ("     --num-unfiltered        INT        number of subsampled unfiltered reads [100000]\n");
  printf ("     --keynormalizer         STRING     key normalization algorithm [keynorm-old]\n");
  printf ("     --dephaser              STRING     dephasing algorithm [treephaser-sse]\n");
  printf ("     --window-size           INT        normalization window size (%d-%d) [%d]\n", kMinWindowSize_, kMaxWindowSize_, DPTreephaser::kWindowSizeDefault_);
  printf ("     --flow-signals-type     STRING     select content of FZ tag [none]\n");
  printf ("                                          \"none\" - FZ not generated\n");
  printf ("                                          \"wells\" - Raw values (unnormalized and not dephased)\n");
  printf ("                                          \"key-normalized\" - Key normalized and not dephased\n");
  printf ("                                          \"adaptive-normalized\" - Adaptive normalized and not dephased\n");
  printf ("                                          \"residual\" - Measurement-prediction residual\n");
  printf ("                                          \"scaled-residual\" - Scaled measurement-prediction residual\n");
  printf ("     --block-row-offset      INT        Offset added to read coordinates [0]\n");
  printf ("     --block-col-offset      INT        Offset added to read coordinates [0]\n");
  printf ("     --extra-trim-left       INT        Number of additional bases after key and barcode to remove from each read [0]\n");
  printf ("     --calibration-training  INT        Generate training set of INT reads. No TFs, no unfiltered sets. 0=off [0]\n");
  printf ("     --calibration-file      FILE       Enable recalibration using tables from provided file [off]\n");
  printf ("     --model-file            FILE       Enable recalibration using model from provided file [off]\n");
  printf ("     --phase-estimation-file FILE       Enable reusing phase estimation from provided file [off]\n");
  printf ("\n");

  BaseCallerFilters::PrintHelp();
  PhaseEstimator::PrintHelp();
  PerBaseQual::PrintHelp();
  BarcodeClassifier::PrintHelp();
  BaseCallerMetricSaver::PrintHelp();

  exit (EXIT_SUCCESS);
}

//! @brief    Verify path exists and if it does, canonicalize it
//! @ingroup  BaseCaller

void ValidateAndCanonicalizePath(string &path)
{
  char *real_path = realpath (path.c_str(), NULL);
  if (real_path == NULL) {
    perror(path.c_str());
    exit(EXIT_FAILURE);
  }
  path = real_path;
  free(real_path);
}

//! @brief    Verify path exists and if it does, canonicalize it. If it doesn't, try fallback location
//! @ingroup  BaseCaller

void ValidateAndCanonicalizePath(string &path, const string& backup_path)
{
  char *real_path = realpath (path.c_str(), NULL);
  if (real_path != NULL) {
    path = real_path;
    free(real_path);
    return;
  }
  perror(path.c_str());
  printf("%s: inaccessible, trying alternative location\n", path.c_str());
  real_path = realpath (backup_path.c_str(), NULL);
  if (real_path != NULL) {
    path = real_path;
    free(real_path);
    return;
  }
  perror(backup_path.c_str());
  exit(EXIT_FAILURE);
}

//! @brief    Shortcut: Print message with time elapsed from start
//! @ingroup  BaseCaller

void ReportState(time_t analysis_start_time, char *my_state)
{
  time_t analysis_current_time;
  time(&analysis_current_time);
  fprintf(stdout, "\n%s: Elapsed: %.1lf minutes, Timestamp: %s\n", my_state,
      difftime(analysis_current_time, analysis_start_time) / 60,
      ctime (&analysis_current_time));
}

//! @brief    Shortcut: save json value to a file.
//! @ingroup  BaseCaller

void SaveJson(const Json::Value & json, const string& filename_json)
{
  ofstream out(filename_json.c_str(), ios::out);
  if (out.good())
    out << json.toStyledString();
  else
    ION_WARN("Unable to write JSON file " + filename_json);
}

void SaveBaseCallerProgress(int percent_complete, const string& output_directory)
{
  string filename_json = output_directory+"/progress_basecaller.json";
  Json::Value progress_json(Json::objectValue);
  progress_json["percent_complete"] = percent_complete;
  SaveJson(progress_json, filename_json);
}




//! @brief    Main function for BaseCaller executable
//! @ingroup  BaseCaller

int main (int argc, const char *argv[])
{
  BaseCallerSalute();

  time_t analysis_start_time;
  time(&analysis_start_time);

  Json::Value basecaller_json(Json::objectValue);
  DumpStartingStateOfProgram (argc,argv,analysis_start_time, basecaller_json["BaseCaller"]);

  /*---   Parse command line options  ---*/

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);

  if (opts.GetFirstBoolean('h', "help", false) or argc == 1)
    PrintHelp();

  if (opts.GetFirstBoolean('v', "version", false)) {
    fprintf (stdout, "%s", IonVersion::GetFullVersion ("BaseCaller").c_str());
    exit (EXIT_SUCCESS);
  }


  // Establish datasets


  // Command line processing *** Directories and file locations

  string input_directory        = opts.GetFirstString ('i', "input-dir", ".");
  string output_directory       = opts.GetFirstString ('o', "output-dir", ".");

  string barcodes_directory   = output_directory + "/bc_files";
  string unfiltered_untrimmed_directory   = output_directory + "/unfiltered.untrimmed";
  string unfiltered_trimmed_directory   = output_directory + "/unfiltered.trimmed";
  string unfiltered_untrimmed_barcodes_directory   = output_directory + "/unfiltered.untrimmed/bc_files";
  string unfiltered_trimmed_barcodes_directory   = output_directory + "/unfiltered.trimmed/bc_files";

  CreateResultsFolder ((char*)output_directory.c_str());
  CreateResultsFolder ((char*)barcodes_directory.c_str());
  CreateResultsFolder ((char*)unfiltered_untrimmed_directory.c_str());
  CreateResultsFolder ((char*)unfiltered_trimmed_directory.c_str());
  CreateResultsFolder ((char*)unfiltered_untrimmed_barcodes_directory.c_str());
  CreateResultsFolder ((char*)unfiltered_trimmed_barcodes_directory.c_str());

  ValidateAndCanonicalizePath(input_directory);
  ValidateAndCanonicalizePath(output_directory);
  ValidateAndCanonicalizePath(barcodes_directory);
  ValidateAndCanonicalizePath(unfiltered_untrimmed_directory);
  ValidateAndCanonicalizePath(unfiltered_trimmed_directory);
  ValidateAndCanonicalizePath(unfiltered_untrimmed_barcodes_directory);
  ValidateAndCanonicalizePath(unfiltered_trimmed_barcodes_directory);

  string filename_wells         = opts.GetFirstString ('-', "wells", input_directory + "/1.wells");
  string filename_mask          = opts.GetFirstString ('-', "mask", input_directory + "/analysis.bfmask.bin");

  ValidateAndCanonicalizePath(filename_wells);
  ValidateAndCanonicalizePath(filename_mask, input_directory + "/bfmask.bin");

  string filename_filter_mask   = output_directory + "/bfmask.bin";
  string filename_json          = output_directory + "/BaseCaller.json";

  printf("\n");
  printf("Input files summary:\n");
  printf("     --input-dir %s\n", input_directory.c_str());
  printf("         --wells %s\n", filename_wells.c_str());
  printf("          --mask %s\n", filename_mask.c_str());
  printf("\n");
  printf("Output directories summary:\n");
  printf("    --output-dir %s\n", output_directory.c_str());
  printf("              bc %s\n", barcodes_directory.c_str());
  printf("        unf.untr %s\n", unfiltered_untrimmed_directory.c_str());
  printf("     unf.untr.bc %s\n", unfiltered_untrimmed_barcodes_directory.c_str());
  printf("          unf.tr %s\n", unfiltered_trimmed_directory.c_str());
  printf("       unf.tr.bc %s\n", unfiltered_trimmed_barcodes_directory.c_str());
  printf("\n");


  // Command line processing *** Various options that need cleanup

  BaseCallerContext bc;
  bc.output_directory = output_directory;

  char default_run_id[6]; // Create a run identifier from full output directory string
  ion_run_to_readname (default_run_id, (char*)output_directory.c_str(), output_directory.length());

  bc.run_id                     = opts.GetFirstString ('-', "run-id", default_run_id);
  bc.dephaser                   = opts.GetFirstString ('-', "dephaser", "treephaser-sse");
  bc.keynormalizer              = opts.GetFirstString ('-', "keynormalizer", "keynorm-old");
  int num_threads               = opts.GetFirstInt    ('n', "num-threads", max(2*numCores(), 4));
  int num_unfiltered            = opts.GetFirstInt    ('-', "num-unfiltered", 100000);
  bc.flow_signals_type          = opts.GetFirstString ('-', "flow-signals-type", "none");
  bc.block_row_offset           = opts.GetFirstInt    ('-', "block-row-offset", 0);
  bc.block_col_offset           = opts.GetFirstInt    ('-', "block-col-offset", 0);
  bc.extra_trim_left            = opts.GetFirstInt    ('-', "extra-trim-left", 0);
  int calibration_training      = opts.GetFirstInt    ('-', "calibration-training", 0);
  string phase_file_name        = opts.GetFirstString ('s', "phase-estimation-file", "");
  bc.windowSize                 = opts.GetFirstInt    ('-', "window-size", DPTreephaser::kWindowSizeDefault_);

  bc.process_tfs = true;
  int subsample_library = -1;

  if (calibration_training > 0) {
    printf (" ======================================================================================\n");
    printf (" ===== BaseCaller will only generate training set (up to %d reads) for Recalibration !\n", calibration_training);
    printf (" ======================================================================================\n\n");
    bc.process_tfs = false;
    subsample_library = calibration_training;
    num_unfiltered = 0;
    bc.flow_signals_type = "scaled-residual";
  }

  // Dataset setup

  BarcodeDatasets datasets(opts, bc.run_id);
  datasets.GenerateFilenames("basecaller_bam",".basecaller.bam");

  printf("Datasets summary (Library):\n");
  printf("   datasets.json %s/datasets_basecaller.json\n", output_directory.c_str());
  for (int ds = 0; ds < datasets.num_datasets(); ++ds) {
    printf("            %3d: %s   Contains read groups: ", ds+1, datasets.dataset(ds)["basecaller_bam"].asCString());
    for (int bc = 0; bc < (int)datasets.dataset(ds)["read_groups"].size(); ++bc)
      printf("%s ", datasets.dataset(ds)["read_groups"][bc].asCString());
    printf("\n");
  }
  printf("\n");

  BarcodeDatasets datasets_tf(bc.run_id);
  if (bc.process_tfs) {
    datasets_tf.dataset(0)["file_prefix"] = "rawtf";
    datasets_tf.dataset(0)["dataset_name"] = "Test Fragments";
    datasets_tf.read_group(0)["description"] = "Test Fragments";
    datasets_tf.GenerateFilenames("basecaller_bam",".basecaller.bam");

    printf("Datasets summary (TF):\n");
    printf("   datasets.json %s/datasets_tf.json\n", output_directory.c_str());
    for (int ds = 0; ds < datasets_tf.num_datasets(); ++ds) {
      printf("            %3d: %s   Contains read groups: ", ds+1, datasets_tf.dataset(ds)["basecaller_bam"].asCString());
      for (int bc = 0; bc < (int)datasets_tf.dataset(ds)["read_groups"].size(); ++bc)
        printf("%s ", datasets_tf.dataset(ds)["read_groups"][bc].asCString());
      printf("\n");
    }
  } else
    printf("TF basecalling disabled\n");
  printf("\n");

  BarcodeDatasets datasets_unfiltered_untrimmed(datasets);
  BarcodeDatasets datasets_unfiltered_trimmed(datasets);

  // Command line processing *** Options that have default values retrieved from wells or mask files

  RawWells wells ("", filename_wells.c_str());
  if (!wells.OpenMetaData()) {
    fprintf (stderr, "Failed to retrieve metadata from %s\n", filename_wells.c_str());
    exit (EXIT_FAILURE);
  }
  Mask mask (1, 1);
  if (mask.SetMask (filename_mask.c_str()))
    exit (EXIT_FAILURE);

  string chip_type = "unknown";
  if (wells.KeyExists("ChipType"))
    wells.GetValue("ChipType", chip_type);

  bc.region_size_x = 50; //! @todo Get default chip size from wells reader
  bc.region_size_y = 50;
  string arg_region_size        = opts.GetFirstString ('-', "region-size", "");
  if (!arg_region_size.empty()) {
    if (2 != sscanf (arg_region_size.c_str(), "%dx%d", &bc.region_size_x, &bc.region_size_y)) {
      fprintf (stderr, "Option Error: region-size %s\n", arg_region_size.c_str());
      exit (EXIT_FAILURE);
    }
  }

  bc.flow_order.SetFlowOrder(     opts.GetFirstString ('-', "flow-order", wells.FlowOrder()),
                                  opts.GetFirstInt    ('f', "flowlimit", wells.NumFlows()));
  if (bc.flow_order.num_flows() > (int)wells.NumFlows())
    bc.flow_order.SetNumFlows(wells.NumFlows());
  assert (bc.flow_order.is_ok());

  string lib_key                = opts.GetFirstString ('-', "lib-key", "TCAG"); //! @todo Get default key from wells
  string tf_key                 = opts.GetFirstString ('-', "tf-key", "ATCG");
  lib_key                       = opts.GetFirstString ('-', "librarykey", lib_key);   // Backward compatible opts
  tf_key                        = opts.GetFirstString ('-', "tfkey", tf_key);
  bc.keys.resize(2);
  bc.keys[0].Set(bc.flow_order, lib_key, "lib");
  bc.keys[1].Set(bc.flow_order, tf_key, "tf");

  bc.chip_size_y = mask.H();
  bc.chip_size_x = mask.W();
  unsigned int subset_begin_x = 0;
  unsigned int subset_begin_y = 0;
  unsigned int subset_end_x = bc.chip_size_x;
  unsigned int subset_end_y = bc.chip_size_y;
  string arg_subset_rows        = opts.GetFirstString ('r', "rows", "");
  string arg_subset_cols        = opts.GetFirstString ('c', "cols", "");
  if (!arg_subset_rows.empty()) {
    if (2 != sscanf (arg_subset_rows.c_str(), "%u-%u", &subset_begin_y, &subset_end_y)) {
      fprintf (stderr, "Option Error: rows %s\n", arg_subset_rows.c_str());
      exit (EXIT_FAILURE);
    }
  }
  if (!arg_subset_cols.empty()) {
    if (2 != sscanf (arg_subset_cols.c_str(), "%u-%u", &subset_begin_x, &subset_end_x)) {
      fprintf (stderr, "Option Error: rows %s\n", arg_subset_cols.c_str());
      exit (EXIT_FAILURE);
    }
  }
  subset_end_x = min(subset_end_x, (unsigned int)bc.chip_size_x);
  subset_end_y = min(subset_end_y, (unsigned int)bc.chip_size_y);
  if (!arg_subset_rows.empty() or !arg_subset_cols.empty())
    printf("Processing chip subregion %u-%u x %u-%u\n", subset_begin_x, subset_end_x, subset_begin_y, subset_end_y);




  bc.class_map.assign(bc.chip_size_x*bc.chip_size_y, -1);
  for (unsigned int y = subset_begin_y; y < subset_end_y; ++y) {
    for (unsigned int x = subset_begin_x; x < subset_end_x; ++x) {
      if (mask.Match(x, y, MaskLib))
        bc.class_map[x + y * bc.chip_size_x] = 0;
      if (mask.Match(x, y, MaskTF) and bc.process_tfs)
        bc.class_map[x + y * bc.chip_size_x] = 1;
    }
  }

  // If we are in library subsampling mode, remove excess reads from bc.class_map
  if (subsample_library > 0) {
    vector<int> new_class_map(bc.chip_size_x*bc.chip_size_y, -1);
    MaskSample<unsigned int> random_lib(mask, MaskLib, (MaskType)(MaskFilteredBadResidual|MaskFilteredBadPPF|MaskFilteredBadKey), subsample_library);
    vector<unsigned int> & values = random_lib.Sample();
    for (int idx = 0; idx < (int)values.size(); ++idx)
      new_class_map[values[idx]] = bc.class_map[values[idx]];
    bc.class_map.swap(new_class_map);
  }

  bc.mask = &mask;
  bc.filename_wells = filename_wells;

  BaseCallerFilters filters(opts, bc.flow_order, bc.keys, mask);
  bc.filters = &filters;
  bc.estimator.InitializeFromOptArgs(opts);
  bc.recalibration.Initialize(opts, bc.flow_order);
  bc.recalModel.Initialize(opts);

  int num_regions_x = (bc.chip_size_x +  bc.region_size_x - 1) / bc.region_size_x;
  int num_regions_y = (bc.chip_size_y +  bc.region_size_y - 1) / bc.region_size_y;

  BarcodeClassifier barcodes(opts, datasets, bc.flow_order, bc.keys, output_directory, bc.chip_size_x, bc.chip_size_y);
  bc.barcodes = &barcodes;

  // initialize the per base quality score generator
  bc.quality_generator.Init(opts, chip_type, output_directory,bc.recalibration.is_enabled());

  BaseCallerMetricSaver metric_saver(opts, bc.chip_size_x, bc.chip_size_y, bc.flow_order.num_flows(),
      bc.region_size_x, bc.region_size_y, output_directory);
  bc.metric_saver = &metric_saver;

  // Command line parsing officially over. Detect unknown options.
  opts.CheckNoLeftovers();


  // Save some run info into our handy json file

  basecaller_json["BaseCaller"]["run_id"] = bc.run_id;
  basecaller_json["BaseCaller"]["flow_order"] = bc.flow_order.str();
  basecaller_json["BaseCaller"]["num_flows"] = bc.flow_order.num_flows();
  basecaller_json["BaseCaller"]["lib_key"] =  bc.keys[0].bases();
  basecaller_json["BaseCaller"]["tf_key"] =  bc.keys[1].bases();
  basecaller_json["BaseCaller"]["chip_type"] = chip_type;
  basecaller_json["BaseCaller"]["input_dir"] = input_directory;
  basecaller_json["BaseCaller"]["output_dir"] = output_directory;
  basecaller_json["BaseCaller"]["filename_wells"] = filename_wells;
  basecaller_json["BaseCaller"]["filename_mask"] = filename_mask;
  basecaller_json["BaseCaller"]["num_threads"] = num_threads;
  basecaller_json["BaseCaller"]["dephaser"] = bc.dephaser;
  basecaller_json["BaseCaller"]["keynormalizer"] = bc.keynormalizer;
  basecaller_json["BaseCaller"]["block_row_offset"] = bc.block_row_offset;
  basecaller_json["BaseCaller"]["block_col_offset"] = bc.block_col_offset;
  basecaller_json["BaseCaller"]["block_row_size"] = bc.chip_size_y;
  basecaller_json["BaseCaller"]["block_col_size"] = bc.chip_size_x;
  SaveJson(basecaller_json, filename_json);

  SaveBaseCallerProgress(0, output_directory);





  MemUsage("RawWellsBasecalling");

  // Find distribution of clonal reads for use in read filtering:
  filters.TrainClonalFilter(output_directory, wells, num_unfiltered, mask);

  MemUsage("ClonalPopulation");
  ReportState(analysis_start_time,"Polyclonal Filter Training Complete");

  // Library CF/IE/DR parameter estimation
  MemUsage("BeforePhaseEstimation");
  bool isLoaded = false;
  if(!phase_file_name.empty()){
	cout << "\nLoad phase estimation from " << phase_file_name << endl;
    isLoaded = bc.estimator.LoadPhaseEstimationTrainSubset(phase_file_name, &mask, bc.region_size_x, bc.region_size_y);
  }
  if(!isLoaded){
      wells.OpenForIncrementalRead();
      bc.estimator.DoPhaseEstimation(&wells, &mask, bc.flow_order, bc.keys, bc.region_size_x, bc.region_size_y, num_threads == 1);
      wells.Close();
  }

  bc.estimator.ExportResultsToJson(basecaller_json["Phasing"]);
  bc.estimator.ExportTrainSubsetToJson(basecaller_json["TrainSubset"]);

  SaveJson(basecaller_json, filename_json);
  SaveBaseCallerProgress(10, output_directory);  // Phase estimation assumed to be 10% of the work

  bc.barcodes->BuildPredictedSignals(bc.estimator.GetAverageCF(), bc.estimator.GetAverageIE(), bc.estimator.GetAverageDR());

  MemUsage("AfterPhaseEstimation");

  ReportState(analysis_start_time,"Phase Parameter Estimation Complete");

  MemUsage("BeforeBasecalling");


  //
  // Step 1. Open wells and output BAM files
  //

  bc.lib_writer.Open(output_directory, datasets, num_regions_x*num_regions_y, bc.flow_order, bc.keys[0].bases(),
      "BaseCaller",
      basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
      basecaller_json["BaseCaller"]["command_line"].asString(),
      basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString(),
      false);

  if (bc.process_tfs) {
    bc.tf_writer.Open(output_directory, datasets_tf, num_regions_x*num_regions_y, bc.flow_order, bc.keys[1].bases(),
        "BaseCaller",
        basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
        basecaller_json["BaseCaller"]["command_line"].asString(),
        basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString(),
        false);
  }

  //! @todo Random subset should also respect options -r and -c
  if (num_unfiltered > 0) {
    MaskSample<unsigned int> random_lib(mask, MaskLib, num_unfiltered);
    bc.unfiltered_set.insert(random_lib.Sample().begin(), random_lib.Sample().end());
  }
  if (!bc.unfiltered_set.empty()) {

    bc.unfiltered_writer.Open(unfiltered_untrimmed_directory, datasets_unfiltered_untrimmed, num_regions_x*num_regions_y, bc.flow_order, bc.keys[0].bases(),
        "BaseCaller",
        basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
        basecaller_json["BaseCaller"]["command_line"].asString(),
        basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString(),
        true);

    bc.unfiltered_trimmed_writer.Open(unfiltered_trimmed_directory, datasets_unfiltered_trimmed, num_regions_x*num_regions_y, bc.flow_order, bc.keys[0].bases(),
        "BaseCaller",
        basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
        basecaller_json["BaseCaller"]["command_line"].asString(),
        basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString(),
        true);
  }

  //
  // Step 3. Open miscellaneous results files
  //

  //
  // Step 4. Execute threaded basecalling
  //

  bc.next_region = 0;
  bc.next_begin_x = 0;
  bc.next_begin_y = 0;

  time_t basecall_start_time;
  time(&basecall_start_time);

  pthread_mutex_init(&bc.mutex, NULL);

  pthread_t worker_id[num_threads];
  for (int worker = 0; worker < num_threads; worker++)
    if (pthread_create(&worker_id[worker], NULL, BasecallerWorker, &bc)) {
      printf("*Error* - problem starting thread\n");
      exit (EXIT_FAILURE);
    }

  for (int worker = 0; worker < num_threads; worker++)
    pthread_join(worker_id[worker], NULL);

  pthread_mutex_destroy(&bc.mutex);

  time_t basecall_end_time;
  time(&basecall_end_time);


  //
  // Step 5. Close files and print out some statistics
  //

  printf("\n\nBASECALLING: called %d of %u wells in %1.0lf seconds with %d threads\n\n",
      filters.NumWellsCalled(), (subset_end_y-subset_begin_y)*(subset_end_x-subset_begin_x),
      difftime(basecall_end_time,basecall_start_time), num_threads);

  bc.lib_writer.Close(datasets, "Library");
  if (bc.process_tfs)
    bc.tf_writer.Close(datasets_tf, "Test Fragments");

  filters.TransferFilteringResultsToMask(mask);

  if (!bc.unfiltered_set.empty()) {

    ofstream filter_status;

    string filter_status_filename = unfiltered_untrimmed_directory + string("/filterStatus.txt");
    filter_status.open(filter_status_filename.c_str());
    filter_status << "col" << "\t" << "row" << "\t" << "highRes" << "\t" << "valid" << endl;
    for (set<unsigned int>::iterator I = bc.unfiltered_set.begin(); I != bc.unfiltered_set.end(); I++) {
      int x = (*I) % bc.chip_size_x;
      int y = (*I) / bc.chip_size_x;
      filter_status << x << "\t" << y;
      filter_status << "\t" << (int) mask.Match(x, y, MaskFilteredBadResidual); // Must happen after filters transferred to mask
      filter_status << "\t" << (int) mask.Match(x, y, MaskKeypass);
      filter_status << endl;
    }
    filter_status.close();

    filter_status_filename = unfiltered_trimmed_directory + string("/filterStatus.txt");
    filter_status.open(filter_status_filename.c_str());
    filter_status << "col" << "\t" << "row" << "\t" << "highRes" << "\t" << "valid" << endl;
    for (set<unsigned int>::iterator I = bc.unfiltered_set.begin(); I != bc.unfiltered_set.end(); I++) {
      int x = (*I) % bc.chip_size_x;
      int y = (*I) / bc.chip_size_x;
      filter_status << x << "\t" << y;
      filter_status << "\t" << (int) mask.Match(x, y, MaskFilteredBadResidual); // Must happen after filters transferred to mask
      filter_status << "\t" << (int) mask.Match(x, y, MaskKeypass);
      filter_status << endl;
    }
    filter_status.close();

    bc.unfiltered_writer.Close(datasets_unfiltered_untrimmed);
    bc.unfiltered_trimmed_writer.Close(datasets_unfiltered_trimmed);

    datasets_unfiltered_untrimmed.SaveJson(unfiltered_untrimmed_directory+"/datasets_basecaller.json");
    datasets_unfiltered_trimmed.SaveJson(unfiltered_trimmed_directory+"/datasets_basecaller.json");
  }

  metric_saver.Close();

  barcodes.Close(datasets);

  datasets.SaveJson(output_directory+"/datasets_basecaller.json");

  if (bc.process_tfs)
    datasets_tf.SaveJson(output_directory+"/datasets_tf.json");

  // Generate BaseCaller.json

  bc.lib_writer.SaveFilteringStats(basecaller_json, "lib", true);
  if (bc.process_tfs)
    bc.tf_writer.SaveFilteringStats(basecaller_json, "tf", false);

  time_t analysis_end_time;
  time(&analysis_end_time);

  basecaller_json["BaseCaller"]["end_time"] = get_time_iso_string(analysis_end_time);
  basecaller_json["BaseCaller"]["total_duration"] = (int)difftime(analysis_end_time,analysis_start_time);
  basecaller_json["BaseCaller"]["basecalling_duration"] = (int)difftime(basecall_end_time,basecall_start_time);

  basecaller_json["Filtering"]["qv_histogram"] = Json::arrayValue;
  for (int qv = 0; qv < 50; ++qv)
    basecaller_json["Filtering"]["qv_histogram"][qv] = (Json::UInt64)bc.lib_writer.qv_histogram()[qv];

  SaveJson(basecaller_json, filename_json);
  SaveBaseCallerProgress(100, output_directory);

  mask.WriteRaw (filename_filter_mask.c_str());
  mask.validateMask();

  MemUsage("AfterBasecalling");
  ReportState(analysis_start_time,"Basecalling Complete");

  return EXIT_SUCCESS;
}


//! @brief      Main code for BaseCaller worker thread
//! @ingroup    BaseCaller
//! @param[in]  input  Pointer to BaseCallerContext.

void * BasecallerWorker(void *input)
{
  BaseCallerContext& bc = *static_cast<BaseCallerContext*>(input);

  RawWells wells ("", bc.filename_wells.c_str());
  pthread_mutex_lock(&bc.mutex);
  wells.OpenForIncrementalRead();
  pthread_mutex_unlock(&bc.mutex);

  vector<float> residual(bc.flow_order.num_flows(), 0);
  vector<float> scaled_residual(bc.flow_order.num_flows(), 0);
  vector<float> wells_measurements(bc.flow_order.num_flows(), 0);
  vector<float> local_noise(bc.flow_order.num_flows(), 0);
  vector<float> minus_noise_overlap(bc.flow_order.num_flows(), 0);
  vector<float> homopolymer_rank(bc.flow_order.num_flows(), 0);
  vector<float> neighborhood_noise(bc.flow_order.num_flows(), 0);
  vector<float> phasing_parameters(3);
  vector<uint16_t>  flowgram(bc.flow_order.num_flows());
  vector<int16_t>   flowgram2(bc.flow_order.num_flows());
  vector<int16_t> filtering_details(13,0);

  vector<uint8_t>   quality(3*bc.flow_order.num_flows());
  vector<int>       base_to_flow (3*bc.flow_order.num_flows());             //!< Flow of in-phase incorporation of each base.

  DPTreephaser treephaser(bc.flow_order, bc.windowSize);
  TreephaserSSE treephaser_sse(bc.flow_order, bc.windowSize);


  while (true) {

    //
    // Step 1. Retrieve next unprocessed region
    //

    pthread_mutex_lock(&bc.mutex);

    if (bc.next_begin_y >= bc.chip_size_y) {
      wells.Close();
      pthread_mutex_unlock(&bc.mutex);
      return NULL;
    }

    int current_region = bc.next_region++;
    int begin_x = bc.next_begin_x;
    int begin_y = bc.next_begin_y;
    int end_x = min(begin_x + bc.region_size_x, bc.chip_size_x);
    int end_y = min(begin_y + bc.region_size_y, bc.chip_size_y);
    bc.next_begin_x += bc.region_size_x;
    if (bc.next_begin_x >= bc.chip_size_x) {
      bc.next_begin_x = 0;
      bc.next_begin_y += bc.region_size_y;
    }

    int num_usable_wells = 0;
    for (int y = begin_y; y < end_y; ++y)
      for (int x = begin_x; x < end_x; ++x)
        if (bc.class_map[x + y * bc.chip_size_x] >= 0)
          num_usable_wells++;

    if      (begin_x == 0)            printf("\n% 5d/% 5d: ", begin_y, bc.chip_size_y);
    if      (num_usable_wells ==   0) printf("  ");
    else if (num_usable_wells <  750) printf(". ");
    else if (num_usable_wells < 1500) printf("o ");
    else if (num_usable_wells < 2250) printf("# ");
    else                              printf("##");
    fflush(NULL);

    if (begin_x == 0)
      SaveBaseCallerProgress(10 + (80*begin_y)/bc.chip_size_y, bc.output_directory);

    pthread_mutex_unlock(&bc.mutex);

    // Process the data
    deque<ProcessedRead> lib_reads;
    deque<ProcessedRead> tf_reads;
    deque<ProcessedRead> unfiltered_reads;
    deque<ProcessedRead> unfiltered_trimmed_reads;

    if (num_usable_wells == 0) { // There is nothing in this region. Don't even bother reading it
      bc.lib_writer.WriteRegion(current_region,lib_reads);
      if (bc.process_tfs)
        bc.tf_writer.WriteRegion(current_region,tf_reads);
      if (!bc.unfiltered_set.empty()) {
        bc.unfiltered_writer.WriteRegion(current_region,unfiltered_reads);
        bc.unfiltered_trimmed_writer.WriteRegion(current_region,unfiltered_trimmed_reads);
      }
      continue;
    }

    wells.SetChunk(begin_y, end_y-begin_y, begin_x, end_x-begin_x, 0, bc.flow_order.num_flows());
    wells.ReadWells();

    for (int y = begin_y; y < end_y; ++y)
    for (int x = begin_x; x < end_x; ++x) {   // Loop over wells within current region

      //
      // Step 2. Retrieve additional information needed to process this read
      //

      unsigned int read_index = x + y * bc.chip_size_x;
      int read_class = bc.class_map[read_index];
      if (read_class < 0)
        continue;
      bool is_random_unfiltered = bc.unfiltered_set.count(read_index) > 0;

      bc.filters->SetValid(read_index); // Presume valid until some filter proves otherwise

      if (read_class == 0)
        lib_reads.push_back(ProcessedRead());
      else
        tf_reads.push_back(ProcessedRead());

      ProcessedRead& processed_read = (read_class==0) ? lib_reads.back() : tf_reads.back();

      if (read_class == 0)
        processed_read.read_group_index = bc.barcodes->no_barcode_read_group_;
      else
        processed_read.read_group_index = 0;

      // Respect filter decisions from Background Model
      if (bc.mask->Match(read_index, MaskFilteredBadResidual))
        bc.filters->SetBkgmodelHighPPF(read_index, processed_read.filter);

      if (bc.mask->Match(read_index, MaskFilteredBadPPF))
        bc.filters->SetBkgmodelPolyclonal(read_index, processed_read.filter);

      if (bc.mask->Match(read_index, MaskFilteredBadKey))
        bc.filters->SetBkgmodelFailedKeypass(read_index, processed_read.filter);

      if (!is_random_unfiltered and !bc.filters->IsValid(read_index)) // No reason to waste more time
        continue;

      float cf = bc.estimator.GetWellCF(x,y);
      float ie = bc.estimator.GetWellIE(x,y);
      float dr = bc.estimator.GetWellDR(x,y);

      for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow)
        wells_measurements[flow] = wells.At(y,x,flow);

      // Sanity check. If there are NaNs in this read, print warning
      vector<int> nanflow;
      for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow) {
        if (!isnan(wells_measurements[flow]))
          continue;
        wells_measurements[flow] = 0;
        nanflow.push_back(flow);
      }
      if(nanflow.size() > 0) {
        fprintf(stderr, "ERROR: BaseCaller read NaNs from wells file, x=%d y=%d flow=%d", x, y, nanflow[0]);
        for(unsigned int flow=1; flow < nanflow.size(); flow++) {
          fprintf(stderr, ",%d", nanflow[flow]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
      }

      //
      // Step 3. Perform base calling and quality value calculation
      //

      BasecallerRead read;
      if (bc.keynormalizer == "keynorm-new") {
        read.SetDataAndKeyNormalizeNew(&wells_measurements[0], wells_measurements.size(), bc.keys[read_class].flows(), bc.keys[read_class].flows_length() - 1, true);

      } else { // if (bc.keynormalizer == "keynorm-old") {
        read.SetDataAndKeyNormalize(&wells_measurements[0], wells_measurements.size(), bc.keys[read_class].flows(), bc.keys[read_class].flows_length() - 1);

      }

      bc.filters->FilterHighPPFAndPolyclonal (read_index, read_class, processed_read.filter, read.raw_measurements);
      if (!is_random_unfiltered and !bc.filters->IsValid(read_index))// No reason to waste more time
        continue;

      // Execute the iterative solving-normalization routine
      if (bc.dephaser == "treephaser-sse") {
        treephaser_sse.SetModelParameters(cf, ie);
        vector<vector<vector<float> > > * aPtr = 0;
        vector<vector<vector<float> > > * bPtr = 0;
        if(bc.recalModel.is_enabled()){
          aPtr = bc.recalModel.getAs(x+bc.block_col_offset, y+bc.block_row_offset);
          bPtr = bc.recalModel.getBs(x+bc.block_col_offset, y+bc.block_row_offset);
          if(aPtr != 0 && bPtr != 0)
            treephaser_sse.SetAsBs(aPtr, bPtr, true);
        }
        treephaser_sse.NormalizeAndSolve(read);
        treephaser.SetModelParameters(cf, ie, 0);//to remove
      } else if (bc.dephaser == "dp-treephaser") {
        treephaser.SetModelParameters(cf, ie, dr);
        treephaser.NormalizeAndSolve4(read, bc.flow_order.num_flows());
        treephaser.ComputeQVmetrics(read);
      } else if (bc.dephaser == "treephaser-adaptive") {
        treephaser.SetModelParameters(cf, ie, 0);
        treephaser.NormalizeAndSolve3(read, bc.flow_order.num_flows()); // Adaptive normalization
        treephaser.ComputeQVmetrics(read);
      } else { //if (bc.dephaser == "treephaser-swan") {
        treephaser.SetModelParameters(cf, ie, dr);
        vector<vector<vector<float> > > * aPtr = 0;
        vector<vector<vector<float> > > * bPtr = 0;
        if(bc.recalModel.is_enabled()){
          aPtr = bc.recalModel.getAs(x+bc.block_col_offset, y+bc.block_row_offset);
          bPtr = bc.recalModel.getBs(x+bc.block_col_offset, y+bc.block_row_offset);
//          printf("a: %6.4f; b: %6.4f\n", (*aPtr)[0][0][1], (*bPtr)[0][0][1]);
          treephaser.SetAsBs(*aPtr, *bPtr, true);
        }
        treephaser.NormalizeAndSolve5(read, bc.flow_order.num_flows()); // sliding window adaptive normalization
        treephaser.ComputeQVmetrics(read);
      }

      // If recalibration is enabled, generate adjusted sequence and normalized_measurements, and recompute QV metrics

//      BasecallerRead readDP = read;
      if (bc.recalibration.is_enabled()) {
        bc.recalibration.CalibrateRead(x+bc.block_col_offset,y+bc.block_row_offset,read.sequence, read.normalized_measurements, read.prediction, read.state_inphase);
        if(bc.dephaser == "treephaser-sse")
          treephaser_sse.ComputeQVmetrics(read);
        else
          treephaser.ComputeQVmetrics(read); // also generates updated read.prediction
      }

      // Misc data management: Generate residual, scaled_residual
      for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow) {
        residual[flow] = read.normalized_measurements[flow] - read.prediction[flow];
        scaled_residual[flow] = residual[flow] / read.state_inphase[flow];
      }

      //delay call of ComputeQVmetrics so that state_inphase would be from treephaserSSE consistently
      if (!bc.recalibration.is_enabled() && bc.dephaser == "treephaser-sse") {
        treephaser_sse.ComputeQVmetrics(read);
      }

      // Misc data management: Put base calls in proper string form

      processed_read.filter.n_bases = read.sequence.size();
      processed_read.filter.is_called = true;

      // Misc data management: Generate base_to_flow

      base_to_flow.clear();
      base_to_flow.reserve(processed_read.filter.n_bases);
      for (int base = 0, flow = 0; base < processed_read.filter.n_bases; ++base) {
        while (flow < bc.flow_order.num_flows() and read.sequence[base] != bc.flow_order[flow])
          flow++;
        base_to_flow.push_back(flow);
      }


      // Misc data management: Populate some trivial read properties

      char read_name[256];
      sprintf(read_name, "%s:%05d:%05d", bc.run_id.c_str(), bc.block_row_offset + y, bc.block_col_offset + x);
      processed_read.bam.Name = read_name;
      processed_read.bam.SetIsMapped(false);

      phasing_parameters[0] = cf;
      phasing_parameters[1] = ie;
      phasing_parameters[2] = dr;
      processed_read.bam.AddTag("ZP", phasing_parameters);


      // Calculation of quality values
      // Predictor 1 - Treephaser residual penalty
      // Predictor 2 - Local noise/flowalign - 'noise' in the input base's measured val.  Noise is max[abs(val - round(val))] within +-1 BASES
      // Predictor 3 - Read Noise/Overlap - mean & stdev of the 0-mers & 1-mers in the read
      // Predictor 3 (new) - Beverly Events
      // Predictor 4 - Transformed homopolymer length
      // Predictor 5 - Treephaser: Penalty indicating deletion after the called base
      // Predictor 6 - Neighborhood noise - mean of 'noise' +-5 BASES around a base.  Noise is mean{abs(val - round(val))}

      int num_predictor_bases = min(bc.flow_order.num_flows(), processed_read.filter.n_bases);

      PerBaseQual::PredictorLocalNoise(local_noise, num_predictor_bases, base_to_flow, read.normalized_measurements, read.prediction);
      PerBaseQual::PredictorNeighborhoodNoise(neighborhood_noise, num_predictor_bases, base_to_flow, read.normalized_measurements, read.prediction);
      //PerBaseQual::PredictorNoiseOverlap(minus_noise_overlap, num_predictor_bases, read.normalized_measurements, read.prediction);
      PerBaseQual::PredictorBeverlyEvents(minus_noise_overlap, num_predictor_bases, base_to_flow, scaled_residual);
      PerBaseQual::PredictorHomopolymerRank(homopolymer_rank, num_predictor_bases, read.sequence);

      quality.clear();
      bc.quality_generator.GenerateBaseQualities(processed_read.bam.Name, processed_read.filter.n_bases, bc.flow_order.num_flows(),
          read.penalty_residual, local_noise, minus_noise_overlap, // <- predictors 1,2,3
          homopolymer_rank, read.penalty_mismatch, neighborhood_noise, // <- predictors 4,5,6
          base_to_flow, quality,
          read.additive_correction,
          read.multiplicative_correction,
          read.state_inphase);

      //
      // Step 4a. Barcode classification of library reads
      //

      if (processed_read.filter.n_bases_filtered == -1)
        processed_read.filter.n_bases_filtered = processed_read.filter.n_bases;

      processed_read.filter.n_bases_key = min(bc.keys[read_class].bases_length(), processed_read.filter.n_bases);
      processed_read.filter.n_bases_prefix = processed_read.filter.n_bases_key;

      if (read_class == 0)
        bc.barcodes->ClassifyAndTrimBarcode(read_index, processed_read, read, base_to_flow);

      //
      // Step 4b. Custom mod: Trim extra bases after key and barcode. Make it look like barcode trimming.
      //

      if (bc.extra_trim_left > 0)
        processed_read.filter.n_bases_prefix = min(processed_read.filter.n_bases_prefix + bc.extra_trim_left, processed_read.filter.n_bases);


      //
      // Step 4. Calculate/save read metrics and apply filters
      //

      bc.filters->FilterZeroBases     (read_index, read_class, processed_read.filter);
      bc.filters->FilterShortRead     (read_index, read_class, processed_read.filter);
      bc.filters->FilterFailedKeypass (read_index, read_class, processed_read.filter, read.sequence);
      bc.filters->FilterHighResidual  (read_index, read_class, processed_read.filter, residual);
      bc.filters->FilterBeverly       (read_index, read_class, processed_read.filter, scaled_residual, base_to_flow);
      bc.filters->TrimAdapter         (read_index, read_class, processed_read, scaled_residual, base_to_flow, treephaser, read);
      bc.filters->TrimQuality         (read_index, read_class, processed_read.filter, quality);
      bc.filters->TrimAvalanche       (read_index, read_class, processed_read.filter, quality);

      //! New mechanism for dumping potentially useful metrics.
      if (bc.metric_saver->save_anything() and (is_random_unfiltered or !bc.metric_saver->save_subset_only())) {
        pthread_mutex_lock(&bc.mutex);

        bc.metric_saver->SaveRawMeasurements          (y,x,read.raw_measurements);
        bc.metric_saver->SaveAdditiveCorrection       (y,x,read.additive_correction);
        bc.metric_saver->SaveMultiplicativeCorrection (y,x,read.multiplicative_correction);
        bc.metric_saver->SaveNormalizedMeasurements   (y,x,read.normalized_measurements);
        bc.metric_saver->SavePrediction               (y,x,read.prediction);
        bc.metric_saver->SaveStateInphase             (y,x,read.state_inphase);
        bc.metric_saver->SaveStateTotal               (y,x,read.state_total);
        bc.metric_saver->SavePenaltyResidual          (y,x,read.penalty_residual);
        bc.metric_saver->SavePenaltyMismatch          (y,x,read.penalty_mismatch);
        bc.metric_saver->SaveLocalNoise               (y,x,local_noise);
        bc.metric_saver->SaveNoiseOverlap             (y,x,minus_noise_overlap);
        bc.metric_saver->SaveHomopolymerRank          (y,x,homopolymer_rank);
        bc.metric_saver->SaveNeighborhoodNoise        (y,x,neighborhood_noise);

        pthread_mutex_unlock(&bc.mutex);
      }


      //
      // Step 4b. Add flow signal information to ZM tag in BAM record.
      //

      flowgram2.clear();
      int max_flow = min(bc.flow_order.num_flows(),16);
      if (processed_read.filter.n_bases_filtered > 0)
        max_flow = min(bc.flow_order.num_flows(), base_to_flow[processed_read.filter.n_bases_filtered-1] + 16);

      for (int flow = 0; flow < max_flow; ++flow)
        flowgram2.push_back(2*(int16_t)(128*read.normalized_measurements[flow]));
      processed_read.bam.AddTag("ZM", flowgram2);
      //flowgram2.push_back(1*(int16_t)(256*read.normalized_measurements[flow]));
      //flowgram2.push_back(2*(int16_t)(128*read.normalized_measurements[flow]));
      //flowgram2.push_back(4*(int16_t)(64*read.normalized_measurements[flow]));
      //flowgram2.push_back(8*(int16_t)(32*read.normalized_measurements[flow]));

      //
      // Step 4c. Populate FZ tag in BAM record.
      //

      flowgram.clear();
      if (bc.flow_signals_type == "wells") {
        for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow)
          flowgram.push_back(max(0,(int)(100.0*wells_measurements[flow]+0.5)));
        processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

      } else if (bc.flow_signals_type == "key-normalized") {
        for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow)
          flowgram.push_back(max(0,(int)(100.0*read.raw_measurements[flow]+0.5)));
        processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

      } else if (bc.flow_signals_type == "adaptive-normalized") {
        for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow)
          flowgram.push_back(max(0,(int)(100.0*read.normalized_measurements[flow]+0.5)));
        processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

      } else if (bc.flow_signals_type == "residual") {
        for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow)
          flowgram.push_back(max(0,(int)(1000 + 100*residual[flow])));
        processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

      } else if (bc.flow_signals_type == "scaled-residual") { // This settings is necessary part of calibration training
        for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow){
          //between 0 and 98
          float adjustment = min(0.49f, max(-0.49f, scaled_residual[flow]));
          flowgram.push_back(max(0,(int)(49.5 + 100*adjustment)));
        }
        processed_read.bam.AddTag("FZ", flowgram);
      }

      //
      // Step 5. Pass basecalled reads to appropriate writers
      //

      if (processed_read.filter.n_bases > 0) {
        processed_read.bam.QueryBases.reserve(processed_read.filter.n_bases);
        processed_read.bam.Qualities.reserve(processed_read.filter.n_bases);
        for (int base = processed_read.filter.n_bases_prefix; base < processed_read.filter.n_bases_filtered; ++base) {
          processed_read.bam.QueryBases.push_back(read.sequence[base]);
          processed_read.bam.Qualities.push_back(quality[base] + 33);
        }
        processed_read.bam.AddTag("ZF","i", base_to_flow[processed_read.filter.n_bases_prefix]);
      } else
        processed_read.bam.AddTag("ZF","i", 0);



      if (is_random_unfiltered) { // Lib, random
        unfiltered_trimmed_reads.push_back(processed_read);
        unfiltered_reads.push_back(processed_read);

        ProcessedRead& untrimmed_read = unfiltered_reads.back();

        processed_read.filter.GenerateZDVector(filtering_details);
        untrimmed_read.bam.AddTag("ZD", filtering_details);

        if (processed_read.filter.n_bases > 0) {
          untrimmed_read.bam.QueryBases.reserve(processed_read.filter.n_bases);
          untrimmed_read.bam.Qualities.reserve(processed_read.filter.n_bases);
          for (int base = max(processed_read.filter.n_bases_filtered,processed_read.filter.n_bases_prefix); base < processed_read.filter.n_bases; ++base) {
            untrimmed_read.bam.QueryBases.push_back(read.sequence[base]);
            untrimmed_read.bam.Qualities.push_back(quality[base] + 33);
          }
        }

        // Temporary workaround: provide fake FZ tag for unfiltered.trimmed and unfiltered.untrimmed sets.
        if (bc.flow_signals_type == "none") {
          flowgram.assign(1,0);
          unfiltered_reads.back().bam.AddTag("FZ", flowgram);
          unfiltered_trimmed_reads.back().bam.AddTag("FZ", flowgram);
        }


        // If this read was supposed to have "early filtering", make sure we emulate that here
        if (processed_read.filter.n_bases_after_bkgmodel_bad_key >= 0 or
            processed_read.filter.n_bases_after_bkgmodel_high_ppf >= 0 or
            processed_read.filter.n_bases_after_bkgmodel_polyclonal >= 0 or
            processed_read.filter.n_bases_after_high_ppf >= 0 or
            processed_read.filter.n_bases_after_polyclonal >= 0)
          processed_read.filter.n_bases = -1;
      }
    }

    bc.lib_writer.WriteRegion(current_region,lib_reads);
    if (bc.process_tfs)
      bc.tf_writer.WriteRegion(current_region,tf_reads);
    if (!bc.unfiltered_set.empty()) {
      bc.unfiltered_writer.WriteRegion(current_region,unfiltered_reads);
      bc.unfiltered_trimmed_writer.WriteRegion(current_region,unfiltered_trimmed_reads);
    }
  }
}





