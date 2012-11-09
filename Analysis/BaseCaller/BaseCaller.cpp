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
#include "Stats.h"
#include "IonErr.h"
#include "PhaseEstimator.h"
#include "BaseCallerFilters.h"
#include "OptArgs.h"
#include "IonVersion.h"
#include "file-io/ion_util.h"
#include "Utils.h"
#include "RawWells.h"
#include "PerBaseQual.h"
#include "BaseCallerMetricSaver.h"
#include "TreephaserSSE.h"
#include "BarcodeClassifier.h"
#include "BarcodeDatasets.h"
#include "OrderedDatasetWriter.h"

#include "dbgmem.h"

using namespace std;


//! @brief    Information needed by BaseCaller worker threads
//! @ingroup  BaseCaller

struct BaseCallerContext {

  // General run parameters
  string                    run_id;                 //!< Run ID string, prepended to each read name
  string                    dephaser;               //!< Name of selected dephasing algorithm
  string                    filename_wells;         //!< Filename of the input wells file
  int                       chip_size_y;            //!< Chip height in wells
  int                       chip_size_x;            //!< Chip width in wells
  int                       region_size_y;          //!< Wells hdf5 dataset chunk height
  int                       region_size_x;          //!< Wells hdf5 dataset chunk width
  ion::FlowOrder            flow_order;             //!< Flow order object, also stores number of flows
  vector<KeySequence>       keys;                   //!< Info about key sequences in use by library and TFs
  string                    flow_signals_type;      //!< The flow signal type: "default" - Normalized and phased, "wells" - Raw values (unnormalized and not dephased), "key-normalized" - Key normalized and not dephased, "adaptive-normalized" - Adaptive normalized and not dephased, and "unclipped" - Normalized and phased but unclipped.
  string                    output_directory;
  int                       block_row_offset;       //!< Offset added to read names
  int                       block_col_offset;       //!< Offset added to read names

  // Important outside entities accessed by BaseCaller
  Mask                      *mask;                  //!< Beadfind and filtering outcomes for wells
  BaseCallerFilters         *filters;               //!< Filter configuration and stats
  PhaseEstimator            estimator;              //!< Phasing estimation results
  PerBaseQual               quality_generator;      //!< Base phred quality value generator
  vector<int>               class_map;              //!< What to do with each well
  BaseCallerMetricSaver     *metric_saver;          //!< Saves requested metrics to an hdf5
  BarcodeClassifier         *barcodes;              //!< Barcode detection and trimming

  // Threaded processing
  pthread_mutex_t           mutex;                  //!< Shared read/write mutex for BaseCaller worker threads
  int                       next_region;            //!< Number of next region that needs processing by a worker
  int                       next_begin_x;           //!< Starting X coordinate of next region
  int                       next_begin_y;           //!< Starting Y coordinate of next region

  // Basecalling results saved here
  //OrderedRegionSFFWriter    lib_sff;                //!< Writer object for library SFF
  OrderedDatasetWriter      lib_sff;
  OrderedDatasetWriter      tf_sff;                 //!< Writer object for test fragment SFF
  set<unsigned int>         unfiltered_set;         //!< Indicates which wells are to be saved to unfiltered SFF
  OrderedDatasetWriter      unfiltered_sff;         //!< Writer object for unfiltered SFF for a random subset of library reads
  OrderedDatasetWriter      unfiltered_trimmed_sff; //!< Writer object for unfiltered trimmed SFF for a random subset of library reads
  RawWells                  *residual_file;         //!< Cafie residual writer object. Can be NULL if residual file not needed.
  FILE                      *well_stat_file;        //!< WellStats.txt file handle. Can be NULL if well stats file not needed.
};


void * BasecallerWorker(void *input);
FILE * OpenWellStatFile(const string& well_stat_file);
void WriteWellStatFileEntry(FILE *well_stat_file, const KeySequence& key,
    SFFEntry & sff_entry, BasecallerRead & read, const vector<float> & residual,
    int x, int y, double cf, double ie, double dr, bool clonal);



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
  printf ("     --dephaser              STRING     dephasing algorithm [treephaser-sse]\n");
  printf ("     --cafie-residuals       on/off     generate cafie residuals file [off]\n");
  printf ("     --well-stat-file        on/off     generate wells stats file [off]\n");
  printf ("     --flow-signals-type     STRING     the flow signals type [default]\n");
  printf ("                                          \"default\" - Normalized and phased\n");
  printf ("                                          \"wells\" - Raw values (unnormalized and not dephased)\n");
  printf ("                                          \"key-normalized\" - Key normalized and not dephased\n");
  printf ("                                          \"adaptive-normalized\" - Adaptive normalized and not dephased\n");
  printf ("                                          \"unclipped\" - Normalized and phased but unclipped\n");
  printf ("     --block-row-offset      INT        Offset added to read coordinates [0]\n");
  printf ("     --block-col-offset      INT        Offset added to read coordinates [0]\n");
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
  fprintf(stdout, "\n%s: Elapsed: %.1lf minutes\n\n", my_state, difftime(analysis_current_time, analysis_start_time) / 60);
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


#ifdef _DEBUG
void memstatus (void)
{
  memdump();
  dbgmemClose();
}
#endif /* _DEBUG */





//! @brief    Main function for BaseCaller executable
//! @ingroup  BaseCaller

int main (int argc, const char *argv[])
{
  BaseCallerSalute();

#ifdef _DEBUG
  atexit (memstatus);
  dbgmemInit();
#endif /* _DEBUG */

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

  string filename_lib_sff       = output_directory + "/rawlib.sff";
  string filename_tf_sff        = output_directory + "/rawtf.sff";
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
  int num_threads               = opts.GetFirstInt    ('n', "num-threads", max(2*numCores(), 4));
  bool generate_well_stat_file  = opts.GetFirstBoolean('-', "well-stat-file", false);
  bool generate_cafie_residual  = opts.GetFirstBoolean('-', "cafie-residuals", false);
  int num_unfiltered            = opts.GetFirstInt    ('-', "num-unfiltered", 100000);
  bc.flow_signals_type          = opts.GetFirstString ('-', "flow-signals-type", "default");
  bc.block_row_offset           = opts.GetFirstInt    ('-', "block-row-offset", 0);
  bc.block_col_offset           = opts.GetFirstInt    ('-', "block-col-offset", 0);

  // Dataset setup

  BarcodeDatasets datasets(opts, bc.run_id);
  datasets.GenerateFilenames("basecaller_bam",".basecaller.bam");

  BarcodeDatasets datasets_tf(bc.run_id);
  datasets_tf.dataset(0)["file_prefix"] = "rawtf";
  datasets_tf.dataset(0)["dataset_name"] = "Test Fragments";
  datasets_tf.read_group(0)["description"] = "Test Fragments";
  datasets_tf.GenerateFilenames("basecaller_bam",".basecaller.bam");

  printf("Datasets summary (Library):\n");
  printf("   datasets.json %s/datasets_basecaller.json\n", output_directory.c_str());
  for (int ds = 0; ds < datasets.num_datasets(); ++ds) {
    printf("            %3d: %s   Contains read groups: ", ds+1, datasets.dataset(ds)["basecaller_bam"].asCString());
    for (int bc = 0; bc < (int)datasets.dataset(ds)["read_groups"].size(); ++bc)
      printf("%s ", datasets.dataset(ds)["read_groups"][bc].asCString());
    printf("\n");
  }
  printf("\n");

  printf("Datasets summary (TF):\n");
  printf("   datasets.json %s/datasets_tf.json\n", output_directory.c_str());
  for (int ds = 0; ds < datasets_tf.num_datasets(); ++ds) {
    printf("            %3d: %s   Contains read groups: ", ds+1, datasets_tf.dataset(ds)["basecaller_bam"].asCString());
    for (int bc = 0; bc < (int)datasets_tf.dataset(ds)["read_groups"].size(); ++bc)
      printf("%s ", datasets_tf.dataset(ds)["read_groups"][bc].asCString());
    printf("\n");
  }
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
  string arg_region_size          = opts.GetFirstString ('-', "region-size", "");
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
      if (mask.Match(x, y, MaskTF))
        bc.class_map[x + y * bc.chip_size_x] = 1;
    }
  }


  bc.mask = &mask;
  bc.filename_wells = filename_wells;

  BaseCallerFilters filters(opts, bc.flow_order, bc.keys, mask);
  bc.filters = &filters;
  bc.estimator.InitializeFromOptArgs(opts);

  int num_regions_x = (bc.chip_size_x +  bc.region_size_x - 1) / bc.region_size_x;
  int num_regions_y = (bc.chip_size_y +  bc.region_size_y - 1) / bc.region_size_y;

  BarcodeClassifier barcodes(opts, datasets, bc.flow_order, bc.keys, output_directory, bc.chip_size_x, bc.chip_size_y);
  bc.barcodes = &barcodes;

  // initialize the per base quality score generator
  bc.quality_generator.Init(opts, chip_type, output_directory);

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
  wells.OpenForIncrementalRead();
  bc.estimator.DoPhaseEstimation(&wells, &mask, bc.flow_order, bc.keys, bc.region_size_x, bc.region_size_y, num_threads == 1);
  wells.Close();

  bc.estimator.ExportResultsToJson(basecaller_json["Phasing"]);
  SaveJson(basecaller_json, filename_json);
  SaveBaseCallerProgress(10, output_directory);  // Phase estimation assumed to be 10% of the work


  MemUsage("AfterPhaseEstimation");

  ReportState(analysis_start_time,"Phase Parameter Estimation Complete");

  MemUsage("BeforeBasecalling");


  //
  // Step 1. Open wells and sff files
  //

  bc.lib_sff.Open(output_directory, datasets, num_regions_x*num_regions_y, bc.flow_order, bc.keys[0].bases(),
      "BaseCaller",
      basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
      basecaller_json["BaseCaller"]["command_line"].asString(),
      basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString());

  bc.tf_sff.Open(output_directory, datasets_tf, num_regions_x*num_regions_y, bc.flow_order, bc.keys[1].bases(),
      "BaseCaller",
      basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
      basecaller_json["BaseCaller"]["command_line"].asString(),
      basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString());

  //! @todo Random subset should also respect options -r and -c
  MaskSample<unsigned int> random_lib(mask, MaskLib, num_unfiltered);
  bc.unfiltered_set.insert(random_lib.Sample().begin(), random_lib.Sample().end());
  if (!bc.unfiltered_set.empty()) {

    bc.unfiltered_sff.Open(unfiltered_untrimmed_directory, datasets_unfiltered_untrimmed, num_regions_x*num_regions_y, bc.flow_order, bc.keys[0].bases(),
        "BaseCaller",
        basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
        basecaller_json["BaseCaller"]["command_line"].asString(),
        basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString());

    bc.unfiltered_trimmed_sff.Open(unfiltered_trimmed_directory, datasets_unfiltered_trimmed, num_regions_x*num_regions_y, bc.flow_order, bc.keys[0].bases(),
        "BaseCaller",
        basecaller_json["BaseCaller"]["version"].asString() + "/" + basecaller_json["BaseCaller"]["svn_revision"].asString(),
        basecaller_json["BaseCaller"]["command_line"].asString(),
        basecaller_json["BaseCaller"]["start_time"].asString(), basecaller_json["BaseCaller"]["chip_type"].asString());
  }

  //
  // Step 3. Open miscellaneous results files
  //

  // Set up phase residual file
  bc.residual_file = NULL;
  if (generate_cafie_residual) {
    bc.residual_file = new RawWells(output_directory.c_str(), "1.cafie-residuals");
    bc.residual_file->CreateEmpty(bc.flow_order.num_flows(), bc.flow_order.c_str(), bc.chip_size_y, bc.chip_size_x);
    bc.residual_file->OpenForWrite();
    bc.residual_file->SetChunk(0, bc.chip_size_y, 0, bc.chip_size_x, 0, bc.flow_order.num_flows());
  }

  // Set up wellStats file (if necessary)
  bc.well_stat_file = NULL;
  if (generate_well_stat_file)
    bc.well_stat_file = OpenWellStatFile(output_directory + "/wellStats.txt");

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

  printf("\n\nBASECALLING: called %d of %u wells in %1.0lf seconds with %d threads\n",
      filters.NumWellsCalled(), (subset_end_y-subset_begin_y)*(subset_end_x-subset_begin_x),
      difftime(basecall_end_time,basecall_start_time), num_threads);

  bc.lib_sff.Close(datasets);
  bc.tf_sff.Close(datasets_tf);

  // Close files
  if (bc.well_stat_file)
    fclose(bc.well_stat_file);

  if(bc.residual_file) {
    bc.residual_file->WriteWells();
    bc.residual_file->WriteRanks();
    bc.residual_file->WriteInfo();
    bc.residual_file->Close();
    delete bc.residual_file;
  }

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

    bc.unfiltered_sff.Close(datasets_unfiltered_untrimmed, true);
    bc.unfiltered_trimmed_sff.Close(datasets_unfiltered_trimmed, true);

    datasets_unfiltered_untrimmed.SaveJson(unfiltered_untrimmed_directory+"/datasets_basecaller.json");
    datasets_unfiltered_trimmed.SaveJson(unfiltered_trimmed_directory+"/datasets_basecaller.json");
  }

  metric_saver.Close();

  barcodes.Close(datasets);

  datasets.SaveJson(output_directory+"/datasets_basecaller.json");

  datasets_tf.SaveJson(output_directory+"/datasets_tf.json");

  // Generate BaseCaller.json

  filters.GenerateFilteringStatistics(basecaller_json, mask);
  time_t analysis_end_time;
  time(&analysis_end_time);

  basecaller_json["BaseCaller"]["end_time"] = get_time_iso_string(analysis_end_time);
  basecaller_json["BaseCaller"]["total_duration"] = (int)difftime(analysis_end_time,analysis_start_time);
  basecaller_json["BaseCaller"]["basecalling_duration"] = (int)difftime(basecall_end_time,basecall_start_time);

  basecaller_json["Filtering"]["qv_histogram"] = Json::arrayValue;
  for (int qv = 0; qv < 50; ++qv)
    basecaller_json["Filtering"]["qv_histogram"][qv] = (Json::UInt64)bc.lib_sff.qv_histogram()[qv];

  SaveJson(basecaller_json, filename_json);
  SaveBaseCallerProgress(100, output_directory);

  MemUsage("AfterBasecalling");

  ReportState(analysis_start_time,"Basecalling Complete");

  mask.WriteRaw (filename_filter_mask.c_str());
  mask.validateMask();

  ReportState (analysis_start_time,"Analysis (from wells file) Complete");

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
  vector<float> corrected_ionogram(bc.flow_order.num_flows(), 0);
  vector<float> flow_values(bc.flow_order.num_flows(), 0);
  vector<float> local_noise(bc.flow_order.num_flows(), 0);
  vector<float> minus_noise_overlap(bc.flow_order.num_flows(), 0);
  vector<float> homopolymer_rank(bc.flow_order.num_flows(), 0);
  vector<float> neighborhood_noise(bc.flow_order.num_flows(), 0);

  DPTreephaser treephaser(bc.flow_order);
  TreephaserSSE treephaser_sse(bc.flow_order);


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
    deque<SFFEntry> lib_reads;
    deque<SFFEntry> tf_reads;
    deque<SFFEntry> unfiltered_reads;
    deque<SFFEntry> unfiltered_trimmed_reads;

    if (num_usable_wells == 0) { // There is nothing in this region. Don't even bother reading it
      bc.lib_sff.WriteRegion(current_region,lib_reads);
      bc.tf_sff.WriteRegion(current_region,tf_reads);
      if (!bc.unfiltered_set.empty()) {
        bc.unfiltered_sff.WriteRegion(current_region,unfiltered_reads);
        bc.unfiltered_trimmed_sff.WriteRegion(current_region,unfiltered_trimmed_reads);
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

      // Respect filter decisions from Background Model
      if (bc.mask->Match(read_index, MaskFilteredBadResidual))
        bc.filters->SetBkgmodelHighPPF(read_index);

      if (bc.mask->Match(read_index, MaskFilteredBadPPF))
        bc.filters->SetBkgmodelPolyclonal(read_index);

      if (bc.mask->Match(read_index, MaskFilteredBadKey))
        bc.filters->SetBkgmodelFailedKeypass(read_index);

      if (!is_random_unfiltered and !bc.filters->IsValid(read_index)) // No reason to waste more time
          continue;

      float cf = bc.estimator.GetWellCF(x,y);
      float ie = bc.estimator.GetWellIE(x,y);
      float dr = bc.estimator.GetWellDR(x,y);

      for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow)
        flow_values[flow] = wells.At(y,x,flow);

      // Sanity check. If there are NaNs in this read, print warning
      vector<int> nanflow;
      for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow) {
        if (!isnan(flow_values[flow]))
          continue;
        flow_values[flow] = 0;
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
      read.SetDataAndKeyNormalize(&flow_values[0], flow_values.size(), bc.keys[read_class].flows(), bc.keys[read_class].flows_length() - 1);

      bc.filters->FilterHighPPFAndPolyclonal (read_index, read_class, read.raw_measurements);
      if (!is_random_unfiltered and !bc.filters->IsValid(read_index)) // No reason to waste more time
          continue;


      // Execute the iterative solving-normalization routine
      //! @todo Reuse treephaser & cafie parameters for an entire region

      if (bc.dephaser == "treephaser-sse") {
        treephaser_sse.SetModelParameters(cf, ie);
        treephaser_sse.NormalizeAndSolve(read);
        treephaser.SetModelParameters(cf, ie, 0);
        treephaser.ComputeQVmetrics(read);

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
        treephaser.NormalizeAndSolve5(read, bc.flow_order.num_flows()); // sliding window adaptive normalization
        treephaser.ComputeQVmetrics(read);
      }


      SFFEntry sff_entry;
      char read_name[256];
      sprintf(read_name, "%s:%05d:%05d", bc.run_id.c_str(), bc.block_row_offset + y, bc.block_col_offset + x);
      sff_entry.name = read_name;
      sff_entry.clip_qual_left = bc.keys[read_class].bases_length() + 1;
      sff_entry.clip_qual_right = 0;
      sff_entry.clip_adapter_left = 0;
      sff_entry.clip_adapter_right = 0;
      sff_entry.clip_adapter_flow = -1;
      sff_entry.flowgram.resize(bc.flow_order.num_flows());
      sff_entry.n_bases = 0;
      sff_entry.barcode_id = 0;
      sff_entry.barcode_n_errors = 0;

      for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow) {
        residual[flow] = read.normalized_measurements[flow] - read.prediction[flow];
        float adjustment = residual[flow] / read.state_inphase[flow];
        adjustment = min(0.49f, max(-0.49f, adjustment));
        corrected_ionogram[flow] = max(0.0f, read.solution[flow] + adjustment);
        sff_entry.flowgram[flow] = (int)(corrected_ionogram[flow]*100.0+0.5);
        sff_entry.n_bases += read.solution[flow];
      }

      // Fix left clip if have fewer bases than are supposed to be in the key
      if(sff_entry.clip_qual_left > (sff_entry.n_bases+1))
        sff_entry.clip_qual_left = sff_entry.n_bases+1;

      sff_entry.flow_index.reserve(sff_entry.n_bases);
      sff_entry.bases.reserve(sff_entry.n_bases);
      sff_entry.quality.reserve(sff_entry.n_bases);

      vector<int> base_to_flow;
      base_to_flow.reserve(sff_entry.n_bases);

      unsigned int prev_used_flow = 0;
      for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow) {
        for (int hp = 0; hp < read.solution[flow]; hp++) {
          sff_entry.flow_index.push_back(1 + flow - prev_used_flow);
          base_to_flow.push_back(flow);
          sff_entry.bases.push_back(bc.flow_order[flow]);
          prev_used_flow = flow + 1;
        }
      }


      // Calculation of quality values
      // Predictor 1 - Treephaser residual penalty
      // Predictor 2 - Local noise/flowalign - 'noise' in the input base's measured val.  Noise is max[abs(val - round(val))] within +-1 BASES
      // Predictor 3 - Read Noise/Overlap - mean & stdev of the 0-mers & 1-mers in the read
      // Predictor 4 - Transformed homopolymer length
      // Predictor 5 - Treephaser: Penalty indicating deletion after the called base
      // Predictor 6 - Neighborhood noise - mean of 'noise' +-5 BASES around a base.  Noise is mean{abs(val - round(val))}

      int num_predictor_bases = min(bc.flow_order.num_flows(), sff_entry.n_bases);

      PerBaseQual::PredictorLocalNoise(local_noise, num_predictor_bases, base_to_flow, corrected_ionogram);
      PerBaseQual::PredictorNoiseOverlap(minus_noise_overlap, num_predictor_bases, corrected_ionogram);
      PerBaseQual::PredictorHomopolymerRank(homopolymer_rank, num_predictor_bases, sff_entry.flow_index);
      PerBaseQual::PredictorNeighborhoodNoise(neighborhood_noise, num_predictor_bases, base_to_flow, corrected_ionogram);

      bc.quality_generator.GenerateBaseQualities(sff_entry.name, sff_entry.n_bases, bc.flow_order.num_flows(),
          read.penalty_residual, local_noise, minus_noise_overlap, // <- predictors 1,2,3
          homopolymer_rank, read.penalty_mismatch, neighborhood_noise, // <- predictors 4,5,6
          base_to_flow, sff_entry.quality,
          read.additive_correction,
          read.multiplicative_correction,
          read.state_inphase);

      //
      // Step 4a. Barcode classification of library reads
      //

      if (read_class == 0)
        bc.barcodes->ClassifyAndTrimBarcode(read_index, sff_entry);

      //
      // Step 4. Calculate/save read metrics and apply filters
      //

      bc.filters->SetReadLength       (read_index, sff_entry);
      bc.filters->FilterZeroBases     (read_index, read_class, sff_entry);
      bc.filters->FilterShortRead     (read_index, read_class, sff_entry);
      bc.filters->FilterFailedKeypass (read_index, read_class, read.solution);
      bc.filters->FilterHighResidual  (read_index, read_class, residual);
      bc.filters->FilterBeverly       (read_index, read_class, read, sff_entry); // Also trims clipQualRight
      bc.filters->TrimAdapter         (read_index, read_class, sff_entry);
      bc.filters->TrimQuality         (read_index, read_class, sff_entry);

      //! New mechanism for dumping potentially useful metrics.
      //! @todo use this to replace cafie residual
      if (bc.metric_saver->save_anything() and (is_random_unfiltered or !bc.metric_saver->save_subset_only())) {
        pthread_mutex_lock(&bc.mutex);

        bc.metric_saver->SaveRawMeasurements          (y,x,read.raw_measurements);
        bc.metric_saver->SaveAdditiveCorrection       (y,x,read.additive_correction);
        bc.metric_saver->SaveMultiplicativeCorrection (y,x,read.multiplicative_correction);
        bc.metric_saver->SaveNormalizedMeasurements   (y,x,read.normalized_measurements);
        bc.metric_saver->SavePrediction               (y,x,read.prediction);
        bc.metric_saver->SaveSolution                 (y,x,read.solution);
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


      if (bc.well_stat_file or bc.residual_file) {
        //! @todo This extra information dump needs justification/modernization
        //!       Might be useful for filter/trimmer development. Should it be part of BaseCallerFilters?
        //!       Perhaps hdf5-based WellStats? WellStats+cafieResidual in one file?

        pthread_mutex_lock(&bc.mutex);
        if (bc.well_stat_file) {
          WriteWellStatFileEntry(bc.well_stat_file, bc.keys[read_class], sff_entry, read, residual,
            x, y, cf, ie, dr, !bc.filters->IsPolyclonal(read_index));
        }
        if (bc.residual_file) {
          for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow)
            bc.residual_file->WriteFlowgram(flow, x, y, residual[flow]);
        }
        pthread_mutex_unlock(&bc.mutex);
      }

      //
      // Step 4b. Use pre-dephased but normalized signals.
      // 
      if ("default" != bc.flow_signals_type) {
        for (int flow = 0; flow < bc.flow_order.num_flows(); ++flow) {
          if ("wells" == bc.flow_signals_type) sff_entry.flowgram[flow] = (int)(100.0*flow_values[flow]+0.5);
          else if ("key-normalized" == bc.flow_signals_type) sff_entry.flowgram[flow] = (int)(100.0*read.raw_measurements[flow]+0.5);
          else if ("adaptive-normalized" == bc.flow_signals_type) sff_entry.flowgram[flow] = (int)(100.0*read.normalized_measurements[flow]+0.5);
          else if ("unclipped" == bc.flow_signals_type) {
            float adjustment = residual[flow] / read.state_inphase[flow];
            corrected_ionogram[flow] = max(0.0f, read.solution[flow] + adjustment);
            sff_entry.flowgram[flow] = (int)(corrected_ionogram[flow]*100.0+0.5);
          } else break; // ignore
        }
      } 

      //
      // Step 5. Save the basecalling results to appropriate sff files
      //

      if (read_class == 0) { // Lib
        if (is_random_unfiltered) { // Lib, random
          if (bc.filters->IsValid(read_index)) { // Lib, random, valid
            lib_reads.push_back(sff_entry);
          }
          unfiltered_trimmed_reads.push_back(sff_entry);

          unfiltered_reads.push_back(SFFEntry());
          sff_entry.clip_adapter_right = 0;     // Strip trimming info before writing to random sff
          sff_entry.clip_qual_right = 0;
          sff_entry.swap(unfiltered_reads.back());

        } else { // Lib, not random
          if (bc.filters->IsValid(read_index)) { // Lib, not random, valid
            lib_reads.push_back(SFFEntry());
            sff_entry.swap(lib_reads.back());
          }
        }
      } else {  // TF
        if (bc.filters->IsValid(read_index)) { // TF, valid
          tf_reads.push_back(SFFEntry());
          sff_entry.swap(tf_reads.back());
        }
      }
    }

    bc.lib_sff.WriteRegion(current_region,lib_reads);
    bc.tf_sff.WriteRegion(current_region,tf_reads);
    if (!bc.unfiltered_set.empty()) {
      bc.unfiltered_sff.WriteRegion(current_region,unfiltered_reads);
      bc.unfiltered_trimmed_sff.WriteRegion(current_region,unfiltered_trimmed_reads);
    }
  }
}


//! @brief    Open WellStats.txt file for writing and write out the header
//! @ingroup  BaseCaller

FILE * OpenWellStatFile(const string &well_stat_filename)
{
  FILE *well_stat_file = NULL;
  fopen_s(&well_stat_file, well_stat_filename.c_str(), "wb");
  if (!well_stat_file) {
    perror(well_stat_filename.c_str());
    return NULL;
  }
  fprintf(well_stat_file,
      "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
      "col", "row", "isTF", "isLib", "isDud", "isAmbg", "nCall",
      "cf", "ie", "dr", "keySNR", "keySD", "keySig", "oneSig",
      "zeroSig", "ppf", "isClonal", "medAbsRes", "multiplier");

  return well_stat_file;
}

//! @brief    Generate and write a WellStats.txt record
//! @ingroup  BaseCaller

void WriteWellStatFileEntry(FILE *well_stat_file, const KeySequence& key,
    SFFEntry & sff_entry, BasecallerRead & read, const vector<float> & residual,
    int x, int y, double cf, double ie, double dr, bool clonal)
{
  double median_abs_residual = BaseCallerFilters::MedianAbsoluteCafieResidual(residual, 60);
  vector<float>::const_iterator first = read.raw_measurements.begin() + mixed_first_flow();
  vector<float>::const_iterator last  = read.raw_measurements.begin() + mixed_last_flow();
  float ppf = percent_positive(first, last);

  double zeromers[key.flows_length()-1];
  double onemers[key.flows_length()-1];
  int num_zeromers = 0;
  int num_onemers = 0;

  for(int i=0; i<(key.flows_length()-1); i++) {
    if(key[i] == 0)
      zeromers[num_zeromers++] = read.normalized_measurements[i];
    else if(key[i] == 1)
      onemers[num_onemers++] = read.normalized_measurements[i];
  }

  double min_stdev      = 0.01;
  double zeromer_sig    = ionStats::median(zeromers,num_zeromers);
  double zeromer_stdev  = std::max(min_stdev,ionStats::sd(zeromers,num_zeromers));
  double onemer_sig     = ionStats::median(onemers,num_onemers);
  double onemer_stdev   = std::max(min_stdev,ionStats::sd(onemers,num_onemers));
  double key_sig        = onemer_sig - zeromer_sig;
  double key_stdev      = sqrt(pow(zeromer_stdev,2) + pow(onemer_stdev,2));
  double key_snr        = key_sig / key_stdev;

  // Write a line of results - make sure this stays in sync with the header line written by OpenWellStatFile()
  fprintf(well_stat_file,
      "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%1.4f\t%1.4f\t%1.5f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%d\t%1.3f\t%1.3f\n",
      x, y, key.name()=="tf", key.name()=="lib", 0, 0, sff_entry.n_bases,
      cf, ie, dr, key_snr, key_stdev, key_sig, onemer_sig,
      zeromer_sig, ppf, (int)clonal, median_abs_residual, read.key_normalizer);
}




