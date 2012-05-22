/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <iomanip>
#include <algorithm>
#include <string>
#include "json/json.h"

#include "BaseCaller.h"
#include "MaskSample.h"
#include "LinuxCompat.h"
#include "Stats.h"
#include "IonErr.h"
#include "DPTreephaser.h"
#include "PhaseEstimator.h"
#include "OptArgs.h"

#include "dbgmem.h"

using namespace std;


void BaseCaller_salute()
{
  char banner[256];
  sprintf (banner, "/usr/bin/figlet -m0 BaseCaller %s 2>/dev/null", IonVersion::GetVersion().c_str());
  if (system (banner))
    fprintf (stdout, "BaseCaller %s\n", IonVersion::GetVersion().c_str()); // figlet did not execute
}

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
  json["start_time"] = ctime (&analysis_start_time);
  json["version"] = IonVersion::GetVersion() + "-" + IonVersion::GetRelease().c_str();
  json["svn_revision"] = IonVersion::GetSvnRev();
  json["build_number"] = IonVersion::GetBuildNum();
  json["command_line"] = command_line;
}



void PrintHelp()
{
  fprintf (stdout, "\n");
  fprintf (stdout, "Usage: BaseCaller [options] --input-dir=DIR\n");
  fprintf (stdout, "\n");
  fprintf (stdout, "General options:\n");
  fprintf (stdout, "  -h,--help                             print this help message and exit\n");
  fprintf (stdout, "  -v,--version                          print version and exit\n");
  fprintf (stdout, "  -i,--input-dir             DIRECTORY  input files directory [required option]\n");
  fprintf (stdout, "     --wells                 FILE       input wells file [input-dir/1.wells]\n");
  fprintf (stdout, "     --mask                  FILE       input mask file [input-dir/analysis.bfmask.bin]\n");
  fprintf (stdout, "  -o,--output-dir            DIRECTORY  results directory [current dir]\n");
  fprintf (stdout, "     --lib-key               STRING     library key sequence [TCAG]\n");
  fprintf (stdout, "     --tf-key                STRING     test fragment key sequence [ATCG]\n");
  fprintf (stdout, "     --flow-order            STRING     flow order [retrieved from wells file]\n");
  fprintf (stdout, "     --run-id                STRING     read name prefix [hashed input dir name]\n");
  fprintf (stdout, "  -n,--num-threads           INT        number of worker threads [2*numcores]\n");
  fprintf (stdout, "  -f,--flowlimit             INT        basecall only first n flows [all flows]\n");
  fprintf (stdout, "  -r,--rows                  INT-INT    basecall only a range of rows [all rows]\n");
  fprintf (stdout, "  -c,--cols                  INT-INT    basecall only a range of columns [all columns]\n");
  fprintf (stdout, "     --region-size           INTxINT    wells processing chunk size [50x50]\n");
  fprintf (stdout, "     --num-unfiltered        INT        number of subsampled unfiltered reads [100000]\n");
  fprintf (stdout, "     --dephaser              STRING     dephasing algorithm [treephaser-swan]\n");
  fprintf (stdout, "     --phred-table-file      FILE       predictor / quality value file [system default]\n");
  fprintf (stdout, "     --cafie-residuals       on/off     generate cafie residuals file [off]\n");
  fprintf (stdout, "     --well-stat-file        on/off     generate wells stats file [off]\n");
  fprintf (stdout, "\n");

  BaseCallerFilters::PrintHelp();

  fprintf (stdout, "Phasing estimation options:\n");
  fprintf (stdout, "     --phasing-estimator     STRING     phasing estimation algorithm [spatial-refiner]\n");
  fprintf (stdout, "     --libcf-ie-dr           cf,ie,dr   don't estimate phasing and use specified values [not using]\n");
  fprintf (stdout, "  -R,--phasing-regions       INTxINT    number of phasing regions (ignored by spatial-refiner) [12x13]\n");
  fprintf (stdout, "\n");

  exit (EXIT_SUCCESS);
}

void * BasecallerWorkerWrapper(void *input)
{
  static_cast<BaseCaller*>(input)->BasecallerWorker();
  return NULL;
}

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

void ReportState(time_t analysis_start_time, char *my_state)
{
  time_t analysis_current_time;
  time(&analysis_current_time);
  fprintf(stdout, "\n%s: Elapsed: %.1lf minutes\n\n", my_state, difftime(analysis_current_time, analysis_start_time) / 60);
}

void SaveJson(const Json::Value & json, const string& filename_json)
{
  ofstream out(filename_json.c_str(), ios::out);
  if (out.good())
    out << json.toStyledString();
  else
    ION_WARN("Unable to write JSON file " + filename_json);
}

#ifdef _DEBUG
void memstatus (void)
{
  memdump();
  dbgmemClose();
}
#endif /* _DEBUG */

/*************************************************************************************************
 *  Start of Main Function
 ************************************************************************************************/

int main (int argc, const char *argv[])
{
  BaseCaller_salute();

#ifdef _DEBUG
  atexit (memstatus);
  dbgmemInit();
#endif /* _DEBUG */

  time_t analysis_start_time;
  time(&analysis_start_time);

  Json::Value basecaller_json(Json::objectValue);
  DumpStartingStateOfProgram (argc,argv,analysis_start_time, basecaller_json["BaseCaller"]);

  updateProgress (WELL_TO_IMAGE); // TODO: Make pipeline do it
  updateProgress (IMAGE_TO_SIGNAL);


  /*---   Parse command line options  ---*/

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);

  if (opts.GetFirstBoolean('h', "help", false) or argc == 1)
    PrintHelp();

  if (opts.GetFirstBoolean('v', "version", false)) {
    fprintf (stdout, "%s", IonVersion::GetFullVersion ("BaseCaller").c_str());
    exit (EXIT_SUCCESS);
  }


  // Command line processing *** Directories and file locations

  string input_directory        = opts.GetFirstString ('i', "input-dir", ".");
  string output_directory       = opts.GetFirstString ('o', "output-dir", ".");
  string unfiltered_directory   = output_directory + "/unfiltered";

  CreateResultsFolder ((char*)output_directory.c_str());
  CreateResultsFolder ((char*)unfiltered_directory.c_str());

  ValidateAndCanonicalizePath(output_directory);
  ValidateAndCanonicalizePath(unfiltered_directory);
  ValidateAndCanonicalizePath(input_directory);

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
  printf("Output files summary:\n");
  printf("    --output-dir %s\n", output_directory.c_str());
  printf("       --lib-sff %s\n", filename_lib_sff.c_str());
  printf("        --tf-sff %s\n", filename_tf_sff.c_str());
  printf("   --filter-mask %s\n", filename_filter_mask.c_str());
  printf("          json : %s\n", filename_json.c_str());
  printf("unfiltered-dir : %s\n", unfiltered_directory.c_str());
  printf("\n");


  // Command line processing *** Various options that need cleanup

  BaseCaller basecaller;

  char default_run_id[6]; // Create a run identifier from full output directory string
  ion_run_to_readname (default_run_id, (char*)output_directory.c_str(), output_directory.length());

  basecaller.run_id             = opts.GetFirstString ('-', "run-id", default_run_id);
  basecaller.dephaser           = opts.GetFirstString ('-', "dephaser", "treephaser-swan");
  basecaller.phred_table_file   = opts.GetFirstString ('-', "phred-table-file", "");
  int num_threads               = opts.GetFirstInt    ('n', "num-threads", max(2*numCores(), 4));
  bool generate_well_stat_file  = opts.GetFirstBoolean('-', "well-stat-file", false);
  bool generate_cafie_residual  = opts.GetFirstBoolean('-', "cafie-residuals", false);
  int num_unfiltered            = opts.GetFirstInt    ('-', "num-unfiltered", 100000);

  printf("Run ID: %s\n", basecaller.run_id.c_str());


  // Command line processing *** Options that have default values retrieved from wells or mask files

  RawWells rawWells ("", filename_wells.c_str());
  if (!rawWells.OpenMetaData()) {
    fprintf (stderr, "Failed to retrieve metadata from %s\n", filename_wells.c_str());
    exit (EXIT_FAILURE);
  }
  Mask mask (1, 1);
  if (mask.SetMask (filename_mask.c_str()))
    exit (EXIT_FAILURE);

  if (rawWells.KeyExists("ChipType")) {
    string chipType;
    rawWells.GetValue("ChipType", chipType);
    ChipIdDecoder::SetGlobalChipId(chipType.c_str());
  }

  basecaller.region_size_x = 50; // TODO: Get from wells reader
  basecaller.region_size_y = 50; // TODO: Get from wells reader
  string argRegionSize          = opts.GetFirstString ('-', "region-size", "");
  if (!argRegionSize.empty()) {
    if (2 != sscanf (argRegionSize.c_str(), "%dx%d", &basecaller.region_size_x, &basecaller.region_size_y)) {
      fprintf (stderr, "Option Error: region-size %s\n", argRegionSize.c_str());
      exit (EXIT_FAILURE);
    }
  }

  string flowOrder              = opts.GetFirstString ('-', "flow-order", rawWells.FlowOrder());
  basecaller.num_flows          = opts.GetFirstInt    ('f', "flowlimit", rawWells.NumFlows());
  basecaller.num_flows = min(basecaller.num_flows, (int)rawWells.NumFlows());
  assert (!flowOrder.empty());
  assert (basecaller.num_flows > 0);

  string lib_key                = opts.GetFirstString ('-', "lib-key", "TCAG"); // TODO: default from wells?
  string tf_key                 = opts.GetFirstString ('-', "tf-key", "ATCG");
  lib_key                       = opts.GetFirstString ('-', "librarykey", lib_key);   // Backward compatible opts
  tf_key                        = opts.GetFirstString ('-', "tfkey", tf_key);
  basecaller.keys.resize(2);
  basecaller.keys[0].Set(flowOrder, lib_key, "lib");
  basecaller.keys[1].Set(flowOrder, tf_key, "tf");

  basecaller.chip_size_y = mask.H();
  basecaller.chip_size_x = mask.W();
  unsigned int subset_begin_x = 0;
  unsigned int subset_begin_y = 0;
  unsigned int subset_end_x = basecaller.chip_size_x;
  unsigned int subset_end_y = basecaller.chip_size_y;
  string argSubsetRows          = opts.GetFirstString ('r', "rows", "");
  string argSubsetCols          = opts.GetFirstString ('c', "cols", "");
  if (!argSubsetRows.empty()) {
    if (2 != sscanf (argSubsetRows.c_str(), "%u-%u", &subset_begin_y, &subset_end_y)) {
      fprintf (stderr, "Option Error: rows %s\n", argSubsetRows.c_str());
      exit (EXIT_FAILURE);
    }
  }
  if (!argSubsetCols.empty()) {
    if (2 != sscanf (argSubsetCols.c_str(), "%u-%u", &subset_begin_x, &subset_end_x)) {
      fprintf (stderr, "Option Error: rows %s\n", argSubsetCols.c_str());
      exit (EXIT_FAILURE);
    }
  }
  subset_end_x = min(subset_end_x, (unsigned int)basecaller.chip_size_x);
  subset_end_y = min(subset_end_y, (unsigned int)basecaller.chip_size_y);
  if (!argSubsetRows.empty() or !argSubsetCols.empty())
    printf("Processing chip subregion %u-%u x %u-%u\n", subset_begin_x, subset_end_x, subset_begin_y, subset_end_y);

  basecaller.class_map.assign(basecaller.chip_size_x*basecaller.chip_size_y, -1);
  for (unsigned int y = subset_begin_y; y < subset_end_y; ++y) {
    for (unsigned int x = subset_begin_x; x < subset_end_x; ++x) {
      if (mask.Match(x, y, MaskLib))
        basecaller.class_map[x + y * basecaller.chip_size_x] = 0;
      if (mask.Match(x, y, MaskTF))
        basecaller.class_map[x + y * basecaller.chip_size_x] = 1;
    }
  }


  basecaller.mask = &mask;
  basecaller.flow_order = flowOrder;
  basecaller.chip_id = ChipIdDecoder::GetGlobalChipId();
  basecaller.output_directory = output_directory;
  basecaller.filename_wells = filename_wells;

  BaseCallerFilters filters(opts, flowOrder, basecaller.num_flows, basecaller.keys, &mask);
  basecaller.filters = &filters;
  basecaller.estimator.InitializeFromOptArgs(opts, flowOrder, basecaller.num_flows, basecaller.keys);

  // Command line parsing officially over. Detect unknown options.
  opts.CheckNoLeftovers();


  // Save some run info into our handy json file

  basecaller_json["BaseCaller"]["run_id"] = basecaller.run_id;
  basecaller_json["BaseCaller"]["flow_order"] = flowOrder;
  basecaller_json["BaseCaller"]["lib_key"] =  basecaller.keys[0].bases();
  basecaller_json["BaseCaller"]["tf_key"] =  basecaller.keys[1].bases();
  basecaller_json["BaseCaller"]["num_flows"] = basecaller.num_flows;
  basecaller_json["BaseCaller"]["chip_id"] = basecaller.chip_id;
  basecaller_json["BaseCaller"]["input_dir"] = input_directory;
  basecaller_json["BaseCaller"]["output_dir"] = output_directory;
  basecaller_json["BaseCaller"]["filename_wells"] = filename_wells;
  basecaller_json["BaseCaller"]["filename_mask"] = filename_mask;
  basecaller_json["BaseCaller"]["flow_order"] = flowOrder;
  basecaller_json["BaseCaller"]["num_threads"] = num_threads;
  basecaller_json["BaseCaller"]["dephaser"] = basecaller.dephaser;
  SaveJson(basecaller_json, filename_json);



  MemUsage("RawWellsBasecalling");

  // Find distribution of clonal reads for use in read filtering:
  filters.FindClonalPopulation(output_directory, &rawWells, num_unfiltered);

  MemUsage("ClonalPopulation");
  ReportState(analysis_start_time,"Polyclonal Filter Training Complete");

  // Library CF/IE/DR parameter estimation
  MemUsage("BeforePhaseEstimation");
  rawWells.OpenForIncrementalRead();
  basecaller.estimator.DoPhaseEstimation(&rawWells, &mask, basecaller.region_size_x, basecaller.region_size_y, num_threads == 1);
  rawWells.Close();

  basecaller.estimator.ExportResultsToJson(basecaller_json["Phasing"]);
  SaveJson(basecaller_json, filename_json);

  MemUsage("AfterPhaseEstimation");

  ReportState(analysis_start_time,"Phase Parameter Estimation Complete");

  MemUsage("BeforeBasecalling");


  //
  // Step 1. Open wells and sff files
  //

  int num_regions_x = (basecaller.chip_size_x +  basecaller.region_size_x - 1) / basecaller.region_size_x;
  int num_regions_y = (basecaller.chip_size_y +  basecaller.region_size_y - 1) / basecaller.region_size_y;

  basecaller.lib_sff.OpenForWrite(filename_lib_sff, num_regions_x*num_regions_y,
                                  basecaller.num_flows, flowOrder, basecaller.keys[0].bases());
  basecaller.tf_sff.OpenForWrite(filename_tf_sff, num_regions_x*num_regions_y,
                                  basecaller.num_flows, flowOrder, basecaller.keys[1].bases());

  // TODO: Random subset should also respect options -r and -c
  MaskSample<well_index_t> randomLib(mask, MaskLib, num_unfiltered);
  basecaller.unfiltered_set.insert(randomLib.Sample().begin(), randomLib.Sample().end());
  if (!basecaller.unfiltered_set.empty()) {
    string full_path_to_unfiltered_sff = unfiltered_directory + "/" + basecaller.run_id + ".lib.unfiltered.untrimmed.sff";
    string full_path_to_unfiltered_trimmed_sff = unfiltered_directory + "/" + basecaller.run_id + ".lib.unfiltered.trimmed.sff";
    basecaller.unfiltered_sff.OpenForWrite(full_path_to_unfiltered_sff, num_regions_x*num_regions_y,
                                  basecaller.num_flows, flowOrder, basecaller.keys[0].bases());
    basecaller.unfiltered_trimmed_sff.OpenForWrite(full_path_to_unfiltered_trimmed_sff, num_regions_x*num_regions_y,
                                  basecaller.num_flows, flowOrder, basecaller.keys[0].bases());
  }

  //
  // Step 3. Open miscellaneous results files
  //

  // Set up phase residual file
  basecaller.residual_file = NULL;
  if (generate_cafie_residual) {
    basecaller.residual_file = new RawWells(output_directory.c_str(), "1.cafie-residuals");
    basecaller.residual_file->CreateEmpty(basecaller.num_flows, flowOrder.c_str(), basecaller.chip_size_y, basecaller.chip_size_x);
    basecaller.residual_file->OpenForWrite();
    basecaller.residual_file->SetChunk(0, basecaller.chip_size_y, 0, basecaller.chip_size_x, 0, basecaller.num_flows);
  }

  // Set up wellStats file (if necessary)
  basecaller.well_stat_file = NULL;
  if (generate_well_stat_file)
    basecaller.well_stat_file = OpenWellStatFile(output_directory + "/wellStats.txt");

  //
  // Step 4. Execute threaded basecalling
  //

  basecaller.next_region = 0;
  basecaller.next_begin_x = 0;
  basecaller.next_begin_y = 0;

  time_t startBasecall;
  time(&startBasecall);

  pthread_mutex_init(&basecaller.mutex, NULL);

  pthread_t worker_id[num_threads];
  for (int iWorker = 0; iWorker < num_threads; iWorker++)
    if (pthread_create(&worker_id[iWorker], NULL, BasecallerWorkerWrapper, &basecaller)) {
      printf("*Error* - problem starting thread\n");
      exit (EXIT_FAILURE);
    }

  for (int iWorker = 0; iWorker < num_threads; iWorker++)
    pthread_join(worker_id[iWorker], NULL);

  pthread_mutex_destroy(&basecaller.mutex);

  time_t endBasecall;
  time(&endBasecall);

  //
  // Step 5. Close files and print out some statistics
  //

  printf("\n\nBASECALLING: called %d of %u wells in %1.0lf seconds with %d threads\n",
      filters.getNumWellsCalled(), (subset_end_y-subset_begin_y)*(subset_end_x-subset_begin_x),
      difftime(endBasecall,startBasecall), num_threads);

  basecaller.lib_sff.Close();
  printf("Generated library SFF with %d reads\n", basecaller.lib_sff.NumReads());
  basecaller.tf_sff.Close();
  printf("Generated TF SFF with %d reads\n", basecaller.tf_sff.NumReads());

  // Close files
  if (basecaller.well_stat_file)
    fclose(basecaller.well_stat_file);

  if(basecaller.residual_file) {
    basecaller.residual_file->WriteWells();
    basecaller.residual_file->WriteRanks();
    basecaller.residual_file->WriteInfo();
    basecaller.residual_file->Close();
    delete basecaller.residual_file;
  }

  filters.TransferFilteringResultsToMask(&mask);

  if (!basecaller.unfiltered_set.empty()) {

    string filterStatusFileName = unfiltered_directory + string("/") + basecaller.run_id + string(".filterStatus.txt");
    ofstream filterStatus;
    filterStatus.open(filterStatusFileName.c_str());
    filterStatus << "col" << "\t" << "row" << "\t" << "highRes" << "\t" << "valid" << endl;

    for (set<well_index_t>::iterator I = basecaller.unfiltered_set.begin(); I != basecaller.unfiltered_set.end(); I++) {
      int x = (*I) % basecaller.chip_size_x;
      int y = (*I) / basecaller.chip_size_x;
      filterStatus << x << "\t" << y;
      filterStatus << "\t" << (int) mask.Match(x, y, MaskFilteredBadResidual); // Must happen after filters transferred to mask
      filterStatus << "\t" << (int) mask.Match(x, y, MaskKeypass);
      filterStatus << endl;
    }

    filterStatus.close();

    basecaller.unfiltered_sff.Close();
    basecaller.unfiltered_trimmed_sff.Close();
    printf("Generated random unfiltered library SFF with %d reads\n", basecaller.unfiltered_sff.NumReads());
    printf("Generated random unfiltered trimmed library SFF with %d reads\n", basecaller.unfiltered_trimmed_sff.NumReads());
  }


  // Generate BaseCaller.json

  filters.GenerateFilteringStatistics(basecaller_json["BeadSummary"]);
  time_t analysis_end_time;
  time(&analysis_end_time);
  basecaller_json["BaseCaller"]["end_time"] = ctime (&analysis_end_time);
  basecaller_json["BaseCaller"]["total_duration"] = (int)difftime(analysis_end_time,analysis_start_time);
  basecaller_json["BaseCaller"]["basecalling_duration"] = (int)difftime(endBasecall,startBasecall);
  SaveJson(basecaller_json, filename_json);

  MemUsage("AfterBasecalling");

  ReportState(analysis_start_time,"Basecalling Complete");

  mask.WriteRaw (filename_filter_mask.c_str());
  mask.validateMask();

  ReportState (analysis_start_time,"Analysis (from wells file) Complete");

  return EXIT_SUCCESS;
}




void BaseCaller::BasecallerWorker()
{
  PerBaseQual pbq;    // initialize the per base quality score generator
  if (!pbq.Init(chip_id, output_directory, flow_order, phred_table_file))
    ION_ABORT("*Error* - perBaseQualInit failed");

  RawWells wells ("", filename_wells.c_str());
  pthread_mutex_lock(&mutex);
  wells.OpenForIncrementalRead();
  pthread_mutex_unlock(&mutex);

  weight_vec_t residual(num_flows, 0);
  weight_vec_t corrected_ionogram(num_flows, 0);
  vector<float> flow_values(num_flows, 0);

  while (true) {

    //
    // Step 1. Retrieve next unprocessed region
    //

    pthread_mutex_lock(&mutex);

    if (next_begin_y >= chip_size_y) {
      wells.Close();
      pthread_mutex_unlock(&mutex);
      return;
    }
    int current_region = next_region++;
    int begin_x = next_begin_x;
    int begin_y = next_begin_y;
    int end_x = min(begin_x + region_size_x, chip_size_x);
    int end_y = min(begin_y + region_size_y, chip_size_y);
    next_begin_x += region_size_x;
    if (next_begin_x >= chip_size_x) {
      next_begin_x = 0;
      next_begin_y += region_size_y;
    }

    int num_usable_wells = 0;
    for (int y = begin_y; y < end_y; ++y)
      for (int x = begin_x; x < end_x; ++x)
        if (class_map[x + y * chip_size_x] >= 0)
          num_usable_wells++;

    if      (begin_x == 0)            printf("\n% 5d/% 5d: ", begin_y, chip_size_y);
    if      (num_usable_wells ==   0) printf("  ");
    else if (num_usable_wells <  750) printf(". ");
    else if (num_usable_wells < 1500) printf("o ");
    else if (num_usable_wells < 2250) printf("# ");
    else                              printf("##");
    fflush(NULL);

    pthread_mutex_unlock(&mutex);

    // Process the data
    deque<SFFWriterWell> unfiltered_reads;
    deque<SFFWriterWell> unfiltered_trimmed_reads;
    deque<SFFWriterWell> lib_reads;
    deque<SFFWriterWell> tf_reads;

    if (num_usable_wells == 0) { // There is nothing in this region. Don't even bother reading it
      lib_sff.WriteRegion(current_region,lib_reads);
      tf_sff.WriteRegion(current_region,tf_reads);
      if (!unfiltered_set.empty()) {
        unfiltered_sff.WriteRegion(current_region,unfiltered_reads);
        unfiltered_trimmed_sff.WriteRegion(current_region,unfiltered_trimmed_reads);
      }
      continue;
    }

    wells.SetChunk(begin_y, end_y-begin_y, begin_x, end_x-begin_x, 0, num_flows);
    wells.ReadWells();

    for (int y = begin_y; y < end_y; ++y)
    for (int x = begin_x; x < end_x; ++x) {   // Loop over wells within current region

      //
      // Step 2. Retrieve additional information needed to process this read
      //

      well_index_t read_index = x + y * chip_size_x;
      int read_class = class_map[read_index];
      if (read_class < 0)
        continue;
      bool is_random_unfiltered = unfiltered_set.count(read_index) > 0;

      filters->markReadAsValid(read_index); // Presume valid until some filter proves otherwise

      // Respect filter decisions from Background Model
      if (mask->Match(read_index, MaskFilteredBadResidual))
        filters->forceBkgmodelHighPPF(read_index);

      if (mask->Match(read_index, MaskFilteredBadPPF))
        filters->forceBkgmodelPolyclonal(read_index);

      if (mask->Match(read_index, MaskFilteredBadKey))
        filters->forceBkgmodelFailedKeypass(read_index);

      if (!is_random_unfiltered and !filters->isValid(read_index)) // No reason to waste more time
          continue;

      float cf = estimator.getCF(x,y);
      float ie = estimator.getIE(x,y);
      float dr = estimator.getDR(x,y);

      for (int flow = 0; flow < num_flows; ++flow)
        flow_values[flow] = wells.At(y,x,flow);

      // Sanity check. If there are NaNs in this read, print warning
      vector<int> nanflow;
      for (int flow = 0; flow < num_flows; ++flow) {
        if (!isnan(flow_values[flow]))
          continue;
        flow_values[flow] = 0;
        nanflow.push_back(flow);
      }
      if(nanflow.size() > 0) {
        fprintf(stderr, "ERROR: BaseCaller read NaNs from wells file, x=%d y=%d flow=%d", x, y, nanflow[0]);
        for(unsigned int iFlow=1; iFlow < nanflow.size(); iFlow++) {
          fprintf(stderr, ",%d", nanflow[iFlow]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
      }

      //
      // Step 3. Perform base calling and quality value calculation
      //

      BasecallerRead read;
      read.SetDataAndKeyNormalize(&flow_values[0], num_flows, keys[read_class].flows(), keys[read_class].flows_length() - 1);

      filters->tryFilteringHighPPFAndPolyclonal (read_index, read_class, read.measurements);
      if (!is_random_unfiltered and !filters->isValid(read_index)) // No reason to waste more time
          continue;

      // TODO: Reuse treephaser & cafie parameters for an entire region
      DPTreephaser treephaser(flow_order.c_str(), num_flows, 8);

      if (dephaser == "dp-treephaser")
        treephaser.SetModelParameters(cf, ie, dr);
      else
        treephaser.SetModelParameters(cf, ie, 0); // Adaptive normalization

      // Execute the iterative solving-normalization routine
      if (dephaser == "dp-treephaser")
        treephaser.NormalizeAndSolve4(read, num_flows);
      else if (dephaser == "treephaser-adaptive")
        treephaser.NormalizeAndSolve3(read, num_flows); // Adaptive normalization
      else
        treephaser.NormalizeAndSolve5(read, num_flows); // sliding window adaptive normalization

      // one more pass to get quality metrics
      treephaser.ComputeQVmetrics(read);

      SFFWriterWell sff_entry;
      stringstream read_name;
      read_name << run_id << ":" << y << ":" << x;
      sff_entry.name = read_name.str();
      sff_entry.clipQualLeft = keys[read_class].bases_length() + 1;
      sff_entry.clipQualRight = 0;
      sff_entry.clipAdapterLeft = 0;
      sff_entry.clipAdapterRight = 0;
      sff_entry.flowIonogram.resize(num_flows);
      sff_entry.numBases = 0;

      for (int flow = 0; flow < num_flows; ++flow) {
        residual[flow] = read.normalizedMeasurements[flow] - read.prediction[flow];
        float adjustment = residual[flow] / treephaser.oneMerHeight[flow];
        adjustment = min(0.49f, max(-0.49f, adjustment));
        corrected_ionogram[flow] = max(0.0f, read.solution[flow] + adjustment);
        sff_entry.flowIonogram[flow] = (int)(corrected_ionogram[flow]*100.0+0.5);
        sff_entry.numBases += read.solution[flow];
      }

      // Fix left clip if have fewer bases than are supposed to be in the key
      if(sff_entry.clipQualLeft > (sff_entry.numBases+1))
        sff_entry.clipQualLeft = sff_entry.numBases+1;

      sff_entry.baseFlowIndex.reserve(sff_entry.numBases);
      sff_entry.baseCalls.reserve(sff_entry.numBases);
      sff_entry.baseQVs.reserve(sff_entry.numBases);

      unsigned int prev_used_flow = 0;
      for (int flow = 0; flow < num_flows; flow++) {
        for (int hp = 0; hp < read.solution[flow]; hp++) {
          sff_entry.baseFlowIndex.push_back(1 + flow - prev_used_flow);
          sff_entry.baseCalls.push_back(flow_order[flow % flow_order.length()]);
          prev_used_flow = flow + 1;
        }
      }

      // Calculation of quality values
      // TODO: refactor me and make me thread safe so that the same pbq object can be used across threads
      pbq.setWellName(sff_entry.name);
      pbq.GenerateQualityPerBaseTreephaser(treephaser.penaltyResidual, treephaser.penaltyMismatch,
          corrected_ionogram, residual, sff_entry.baseFlowIndex);
      pbq.GetQualities(sff_entry.baseQVs);

      //
      // Step 4. Calculate/save read metrics and apply filters
      //

      filters->tryFilteringZeroBases     (read_index, read_class, sff_entry);
      filters->tryFilteringShortRead     (read_index, read_class, sff_entry);
      filters->tryFilteringFailedKeypass (read_index, read_class, read.solution);
      filters->tryFilteringHighResidual  (read_index, read_class, residual);
      filters->tryFilteringBeverly       (read_index, read_class, read, sff_entry); // Also trims clipQualRight
      filters->tryTrimmingAdapter        (read_index, read_class, sff_entry);
      filters->tryTrimmingQuality        (read_index, read_class, sff_entry);

      if (well_stat_file or residual_file) {
        // TODO: This extra information dump needs justification/modernization
        //       Might be useful for filter/trimmer development. Should it be part of BaseCallerFilters?
        //       Perhaps hdf5-based WellStats? WellStats+cafieResidual in one file?

        pthread_mutex_lock(&mutex);
        if (well_stat_file) {
          WriteWellStatFileEntry(well_stat_file, keys[read_class], sff_entry, read, residual,
            x, y, cf, ie, dr, !filters->isPolyclonal(read_index));
        }
        if (residual_file) {
          for (int iFlow = 0; iFlow < num_flows; iFlow++)
            residual_file->WriteFlowgram(iFlow, x, y, residual[iFlow]);
        }
        pthread_mutex_unlock(&mutex);
      }

      //
      // Step 5. Save the basecalling results to appropriate sff files
      //

      if (read_class == 0) { // Lib
        if (is_random_unfiltered) { // Lib, random
          if (filters->isValid(read_index)) { // Lib, random, valid
            lib_reads.push_back(SFFWriterWell());
            sff_entry.copyTo(lib_reads.back());
          }
          unfiltered_trimmed_reads.push_back(SFFWriterWell());
          sff_entry.copyTo(unfiltered_trimmed_reads.back());

          unfiltered_reads.push_back(SFFWriterWell());
          sff_entry.clipAdapterRight = 0;     // Strip trimming info before writing to random sff
          sff_entry.clipQualRight = 0;
          sff_entry.moveTo(unfiltered_reads.back());

        } else { // Lib, not random
          if (filters->isValid(read_index)) { // Lib, not random, valid
            lib_reads.push_back(SFFWriterWell());
            sff_entry.moveTo(lib_reads.back());
          }
        }
      } else {  // TF
        if (filters->isValid(read_index)) { // TF, valid
          tf_reads.push_back(SFFWriterWell());
          sff_entry.moveTo(tf_reads.back());
        }
      }
    }

    lib_sff.WriteRegion(current_region,lib_reads);
    tf_sff.WriteRegion(current_region,tf_reads);
    if (!unfiltered_set.empty()) {
      unfiltered_sff.WriteRegion(current_region,unfiltered_reads);
      unfiltered_trimmed_sff.WriteRegion(current_region,unfiltered_trimmed_reads);
    }
  }
}



FILE * OpenWellStatFile(const string &wellStatFile)
{
  FILE *wellStatFileFP = NULL;
  fopen_s(&wellStatFileFP, wellStatFile.c_str(), "wb");
  if (!wellStatFileFP) {
    perror(wellStatFile.c_str());
    return NULL;
  }
  fprintf(wellStatFileFP,
      "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
      "col", "row", "isTF", "isLib", "isDud", "isAmbg", "nCall",
      "cf", "ie", "dr", "keySNR", "keySD", "keySig", "oneSig",
      "zeroSig", "ppf", "isClonal", "medAbsRes", "multiplier");

  return wellStatFileFP;
}


void WriteWellStatFileEntry(FILE *wellStatFileFP, const KeySequence& key,
    SFFWriterWell & readResults, BasecallerRead & read, weight_vec_t & residual,
    int x, int y, double cf, double ie, double dr, bool clonal)
{
  double medAbsResidual = BaseCallerFilters::getMedianAbsoluteCafieResidual(residual, CAFIE_RESIDUAL_FLOWS_N);
  vector<float>::const_iterator first = read.measurements.begin() + mixed_first_flow();
  vector<float>::const_iterator last  = read.measurements.begin() + mixed_last_flow();
  float ppf = percent_positive(first, last);

  double zeroMer[key.flows_length()-1];
  double oneMer[key.flows_length()-1];
  int nZeroMer = 0;
  int nOneMer = 0;

  for(int i=0; i<(key.flows_length()-1); i++) {
    if(key[i] == 0)
      zeroMer[nZeroMer++] = read.normalizedMeasurements[i];
    else if(key[i] == 1)
      oneMer[nOneMer++] = read.normalizedMeasurements[i];
  }

  double minSD=0.01;
  double zeroMerSig = ionStats::median(zeroMer,nZeroMer);
  double zeroMerSD  = std::max(minSD,ionStats::sd(zeroMer,nZeroMer));
  double oneMerSig  = ionStats::median(oneMer,nOneMer);
  double oneMerSD   = std::max(minSD,ionStats::sd(oneMer,nOneMer));
  double keySig     = oneMerSig - zeroMerSig;
  double keySD      = sqrt(pow(zeroMerSD,2) + pow(oneMerSD,2));
  double keySNR =  keySig / keySD;

  // Write a line of results - make sure this stays in sync with the header line written by OpenWellStatFile()
  fprintf(wellStatFileFP,
      "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%1.4f\t%1.4f\t%1.5f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%1.3f\t%d\t%1.3f\t%1.3f\n",
      x, y, key.name()=="tf", key.name()=="lib", 0, 0, readResults.numBases,
      cf, ie, dr, keySNR, keySD, keySig, oneMerSig,
      zeroMerSig, ppf, (int)clonal, medAbsResidual, read.keyNormalizer);
}




