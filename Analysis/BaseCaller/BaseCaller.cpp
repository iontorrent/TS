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
#include <algorithm>
#include <string>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <deque>
#include <iostream>
#include <set>
#include <vector>

#include "json/json.h"
#include "MaskSample.h"
#include "LinuxCompat.h"
#include "IonErr.h"
#include "OptArgs.h"
#include "IonVersion.h"
#include "file-io/ion_util.h"
#include "Utils.h"
#include "RawWells.h"

#include "BaseCallerUtils.h"
#include "DPTreephaser.h"
#include "BarcodeDatasets.h"
#include "BarcodeClassifier.h"
#include "OrderedDatasetWriter.h"
#include "TreephaserSSE.h"
#include "PhaseEstimator.h"
#include "PerBaseQual.h"
#include "BaseCallerFilters.h"
#include "BaseCallerMetricSaver.h"
#include "HistogramCalibration.h"
#include "LinearCalibrationModel.h"
#include "MolecularTagTrimmer.h"

#include "BaseCallerParameters.h"

using namespace std;


void * BasecallerWorker(void *input);


// ----------------------------------------------------------------
//! @brief    Print BaseCaller version with figlet.
//! @ingroup  BaseCaller

void BaseCallerSalute()
{
    char banner[256];
    sprintf (banner, "/usr/bin/figlet -m0 BaseCaller %s 2>/dev/null", IonVersion::GetVersion().c_str());
    if (system (banner))
        fprintf (stdout, "BaseCaller %s\n", IonVersion::GetVersion().c_str()); // figlet did not execute
}

// ----------------------------------------------------------------
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
    printf ("Version = %s.%s (%s) (%s)\n",
            IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
            IonVersion::GetGitHash().c_str(), IonVersion::GetBuildNum().c_str());
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
    json["git_hash"] = IonVersion::GetGitHash();
    json["build_number"] = IonVersion::GetBuildNum();
    json["command_line"] = command_line;
}

// ----------------------------------------------------------------
//! @brief    Shortcut: Print message with time elapsed from start
//! @ingroup  BaseCaller

void ReportState(time_t analysis_start_time, const char *my_state)
{
    time_t analysis_current_time;
    time(&analysis_current_time);
    fprintf(stdout, "\n%s: Elapsed: %.1lf minutes, Timestamp: %s\n", my_state,
            difftime(analysis_current_time, analysis_start_time) / 60,
            ctime (&analysis_current_time));
}

// ----------------------------------------------------------------
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

void JsonToCommentLine(const Json::Value & json, vector<string> &comments) {

  Json::FastWriter writer;
  string str = writer.write(json);
  // trim unwanted newline added by writer
  int last_char = str.size()-1;
  if (last_char>=0 and str[last_char]=='\n') {
    str.erase(last_char,1);
  }
  comments.push_back(str);
}

// ----------------------------------------------------------------
void SaveBaseCallerProgress(int percent_complete, const string& output_directory)
{
    string filename_json = output_directory+"/progress_basecaller.json";
    Json::Value progress_json(Json::objectValue);
    progress_json["percent_complete"] = percent_complete;
    SaveJson(progress_json, filename_json);
}


// --------------------------------------------------------------------------
// Function sets the read class for each well and sub-samples if desired

void ClassifyAndSampleWells(BaseCallerContext & bc, const BCwellSampling & SamplingOpts)
{
    ReservoirSample<unsigned int> downsampled_subset(SamplingOpts.downsample_size, 2);
    ReservoirSample<unsigned int> unfiltered_subset(SamplingOpts.num_unfiltered, 1);
    bool eval_all_libWells = SamplingOpts.downsample_size == 0 or SamplingOpts.have_calib_panel;
    bc.class_map.assign((unsigned int)(bc.chip_subset.GetChipSizeX()*bc.chip_subset.GetChipSizeY()), -1);

    // First iteration over wells to sample them and/or assign read class
    for (int y = bc.chip_subset.GetBeginY(); y < bc.chip_subset.GetEndY(); ++y) {
      for (int x = bc.chip_subset.GetBeginX(); x < bc.chip_subset.GetEndX(); ++x) {
    	if (bc.mask->Match(x, y, MaskLib)) {
          // Unfiltered set contains a selection of randomly selected library beads
          if (SamplingOpts.num_unfiltered>0)
            unfiltered_subset.Add(x + y * bc.chip_subset.GetChipSizeX());
          // For calibration set we exclude already filtered reads
          if (SamplingOpts.downsample_size>0 and (not bc.mask->Match(x, y, SamplingOpts.MaskNotWanted)))
            downsampled_subset.Add(x + y * bc.chip_subset.GetChipSizeX());
          if (eval_all_libWells)
            bc.class_map[x + y * bc.chip_subset.GetChipSizeX()] = 0;
        }
        if (bc.mask->Match(x, y, MaskTF) and bc.process_tfs) {
          if (SamplingOpts.downsample_size>0)
            downsampled_subset.Add(x + y * bc.chip_subset.GetChipSizeX());
          else
            bc.class_map[x + y * bc.chip_subset.GetChipSizeX()] = 1;
        }
      }
    }
    downsampled_subset.Finished();
    unfiltered_subset.Finished();

    // Another pass over the read class map to set our sampled subsets
    if (SamplingOpts.num_unfiltered > 0)
      bc.unfiltered_set.insert(unfiltered_subset.GetData().begin(), unfiltered_subset.GetData().end());
    if (SamplingOpts.downsample_size > 0) {
      for (size_t idx=0; idx<downsampled_subset.GetCount(); idx++) {
    	// Mark random calibration sample to be used in addition to calibration panel reads
        if (SamplingOpts.have_calib_panel)
          bc.class_map.at(downsampled_subset.GetVal(idx)) = 2;
        else if (bc.mask->Match(downsampled_subset.GetVal(idx), MaskTF))
          bc.class_map.at(downsampled_subset.GetVal(idx)) = 1;
        else
          bc.class_map.at(downsampled_subset.GetVal(idx)) = 0;
      }
    }
    // Print Summary:
    unsigned int sum_tf_wells    = 0;
    unsigned int sum_lib_wells   = 0;
    unsigned int sum_calib_wells = 0;
    for (unsigned int idx=0; idx<bc.class_map.size(); idx++) {
      if (bc.class_map.at(idx) == 0) sum_lib_wells++;
      if (bc.class_map.at(idx) == 1) sum_tf_wells++;
      if (bc.class_map.at(idx) == 2) sum_calib_wells++;
    }
    cout << "Bead classification summary:" << endl;
    cout << " - Total num. wells   : " << bc.class_map.size() << endl;
    cout << " - Num. Library wells : " << sum_lib_wells       << endl;
    cout << " - Num. Test fragments: " << sum_tf_wells        << endl;
    cout << " - Num. calib. wells  : " << sum_calib_wells     << endl;
};

// --------------------------------------------------------------------------

void scaleup_flowgram(const vector<float>& sigIn, vector<int16_t>& sigOut, int max_flow)
{
    int safe_max = min(max_flow, (int)sigIn.size());
	if (sigOut.size() != (size_t)safe_max)
      sigOut.resize(safe_max);
    for (int flow = 0; flow < safe_max; ++flow){
        int v = 128*sigIn[flow];
        // handle overflow
        if (v < -16383 or v > 16383) {
            v = min(max(-16383,v), 16383);
        }
        sigOut[flow] = (int16_t)(2*v); // faster than sigOut.push_back()
    }
}


void make_base_to_flow(const vector<char>& sequence,ion::FlowOrder& flow_order, vector<int>& base_to_flow,vector<int>& flow_to_base,int num_flows)
{
    int nBases = sequence.size();
    base_to_flow.resize(nBases);
    flow_to_base.assign(num_flows,-1);

    for (int base = 0, flow = 0; base < nBases; ++base) {
        while (flow < num_flows and sequence.at(base) != flow_order[flow])
            flow++;
        base_to_flow.at(base) = flow;
        flow_to_base.at(flow) = base;
    }
}

// --------------------------------------------------------------------------
//! @brief    Main function for BaseCaller executable Mark: XXX
//! @ingroup  BaseCaller


int main (int argc, const char *argv[])
{
    BaseCallerSalute();

    time_t analysis_start_time;
    time(&analysis_start_time);

    Json::Value basecaller_json(Json::objectValue);
    DumpStartingStateOfProgram (argc,argv,analysis_start_time, basecaller_json["BaseCaller"]);
    Json::Value basecaller_bam_comments(Json::objectValue); // Comment lines are json structures
    vector<string> bam_comments; // Every entry in the vector is a comment line in the BAM

    //
    // Step 1. Process Command Line Options & Initialize Modules
    //

    BaseCallerParameters bc_params;
    OptArgs opts, null_opts;
    opts.ParseCmdLine(argc, argv);

    if (opts.GetFirstBoolean('h', "help", false) or argc == 1)
    	bc_params.PrintHelp();
    if (opts.GetFirstBoolean('v', "version", false)) {
        fprintf (stdout, "%s", IonVersion::GetFullVersion ("BaseCaller").c_str());
        exit (EXIT_SUCCESS);
    }

    // Command line processing *** Main directories and file locations first
    bc_params.InitializeFilesFromOptArgs(opts);
    bc_params.InitContextVarsFromOptArgs(opts);

    // Command line processing *** Options that have default values retrieved from wells or mask files
    RawWells wells ("", bc_params.GetFiles().filename_wells.c_str());
    if (!wells.OpenMetaData()) {
        fprintf (stderr, "Failed to retrieve metadata from %s\n", bc_params.GetFiles().filename_wells.c_str());
        exit (EXIT_FAILURE);
    }
    unsigned int chunk_rows, chunk_cols, chunk_flows;
    wells.GetH5ChunkSize(chunk_rows, chunk_cols, chunk_flows);
    Mask mask (1, 1);
    if (mask.SetMask (bc_params.GetFiles().filename_mask.c_str()))
        exit (EXIT_FAILURE);

    string chip_type = "unknown";
    if (wells.KeyExists("ChipType"))
        wells.GetValue("ChipType", chip_type);

    if((!chip_type.empty()) && chip_type[0] == 'P') chip_type[0] = 'p';

    // Command line processing *** Various general option and opts to classify and sample wells
    BaseCallerContext bc;
    bc.mask = &mask;
    bc.SetKeyAndFlowOrder(opts, wells.FlowOrder(), wells.NumFlows());
    bc.chip_subset.InitializeChipSubsetFromOptArgs(opts, mask.W(), mask.H(), chunk_cols, chunk_rows);

    // Sampling options may reset command line arguments & change context
    bc_params.InitializeSamplingFromOptArgs(opts, bc.chip_subset.NumWells());
    bc_params.SetBaseCallerContextVars(bc);

    // --------- Stand alone phase estimation and exit ---------------------------------------
    bc.estimator.InitializeFromOptArgs(opts, bc.chip_subset, bc.keynormalizer);

    if (bc_params.JustPhaseEstimation()) {
      wells.OpenForIncrementalRead();
      bc.estimator.DoPhaseEstimation(&wells, &mask, bc.flow_order, bc.keys, bc_params.NumThreads());
      wells.Close();

      bc.estimator.ExportResultsToJson(basecaller_json["Phasing"]);
      bc.estimator.ExportTrainSubsetToJson(basecaller_json["TrainSubset"]);

      time_t analysis_end_time;
      time(&analysis_end_time);
      basecaller_json["BaseCaller"]["end_time"] = get_time_iso_string(analysis_end_time);
      basecaller_json["BaseCaller"]["total_duration"] = (int)difftime(analysis_end_time,analysis_start_time);
      SaveJson(basecaller_json, bc_params.GetFiles().filename_phase);

      ReportState(analysis_start_time,"Phase Parameter Estimation Complete");
      ReportState(analysis_start_time,"Basecalling Complete");
      exit(EXIT_SUCCESS);
    }
    // --------- Or do phase estimation after booting up all modules & before base calling ---

    // *** Setup for different datasets
    BarcodeDatasets datasets_calibration(bc.run_id, bc_params.GetFiles().calibration_panel_file);
    datasets_calibration.SetIonControl(bc.run_id);
    datasets_calibration.GenerateFilenames("IonControl","basecaller_bam",".basecaller.bam",bc_params.GetFiles().output_directory);

    BarcodeDatasets datasets(bc.run_id, bc_params.GetFiles().lib_datasets_file);
    // Check if any of the template barcodes is equal to a control barcode
    if (datasets_calibration.DatasetInUse())
      datasets.RemoveControlBarcodes(datasets_calibration.json());
    datasets.GenerateFilenames("Library","basecaller_bam",".basecaller.bam",bc_params.GetFiles().output_directory);

    BarcodeDatasets datasets_tf(bc.run_id);
    datasets_tf.SetTF(bc.process_tfs);
    datasets_tf.GenerateFilenames("TF","basecaller_bam",".basecaller.bam",bc_params.GetFiles().output_directory);

    BarcodeDatasets datasets_unfiltered_untrimmed(datasets);
    BarcodeDatasets datasets_unfiltered_trimmed(datasets);


    // *** Initialize remaining modules of BaseCallerContext
    basecaller_bam_comments["BaseCallerComments"]["MasterKey"] = bc.run_id;

    BaseCallerFilters filters(opts, basecaller_bam_comments["BaseCallerComments"], bc.flow_order, bc.keys, mask);
    bc.filters = &filters;

    BaseCallerMetricSaver metric_saver(opts, bc.chip_subset.GetChipSizeX(), bc.chip_subset.GetChipSizeY(), bc.flow_order.num_flows(),
                                bc.chip_subset.GetRegionSizeX(), bc.chip_subset.GetRegionSizeY(), bc_params.GetFiles().output_directory);
    bc.metric_saver = &metric_saver;

    // Calibration modules
    HistogramCalibration hist_calibration(opts, bc.flow_order);
    bc.histogram_calibration = &hist_calibration;

    LinearCalibrationModel linear_calibration_model(opts, bam_comments, bc.run_id, bc.chip_subset, &bc.flow_order);
    bc.linear_cal_model = &linear_calibration_model;

    // initialize the per base quality score generator - dependent on calibration
    bc.quality_generator.Init(opts, chip_type, bc_params.GetFiles().input_directory, bc_params.GetFiles().output_directory, hist_calibration.is_enabled());

    // Barcode classification
    BarcodeClassifier barcodes(opts, datasets, bc.flow_order, bc.keys, bc_params.GetFiles().output_directory,
    		                   bc.chip_subset.GetChipSizeX(), bc.chip_subset.GetChipSizeY());
    bc.barcodes = &barcodes;
    // Make sure calibration barcodes are initialized with default parameters
    BarcodeClassifier calibration_barcodes(null_opts, datasets_calibration, bc.flow_order, bc.keys,
                          bc_params.GetFiles().output_directory, bc.chip_subset.GetChipSizeX(), bc.chip_subset.GetChipSizeY());
    bc.calibration_barcodes = &calibration_barcodes;

    // Molecular tag identification & trimming
    MolecularTagTrimmer tag_trimmer;
    tag_trimmer.InitializeFromJson(MolecularTagTrimmer::ReadOpts(opts), datasets.read_groups(), barcodes.TrimBarcodes());
    bc.tag_trimmer = &tag_trimmer;

    // Command line parsing officially over. Detect unknown options.
    opts.CheckNoLeftovers();

    // Save some run info into our handy json file
    bc_params.SaveParamsToJson(basecaller_json, bc, chip_type);
    SaveBaseCallerProgress(0, bc_params.GetFiles().output_directory);

    MemUsage("RawWellsBasecalling");


    //
    // Step 2. Filter training and do phase estimation
    //

    // Classify wells subsets to be processed / ignored during base calling
    ClassifyAndSampleWells(bc, bc_params.GetSamplingOpts());

    // Find distribution of clonal reads for use in read filtering:
    filters.TrainClonalFilter(bc_params.GetFiles().output_directory, wells, mask);
    MemUsage("ClonalPopulation");
    ReportState(analysis_start_time,"Polyclonal Filter Training Complete");

    // Library phasing parameter estimation
    if (not bc.estimator.HaveEstimates()) {
        MemUsage("BeforePhaseEstimation");

        wells.OpenForIncrementalRead();
        bc.estimator.DoPhaseEstimation(&wells, &mask, bc.flow_order, bc.keys, bc_params.NumThreads());
        wells.Close();
        MemUsage("AfterPhaseEstimation");
    }
    bc.estimator.ExportResultsToJson(basecaller_json["Phasing"]);
    bc.estimator.ExportTrainSubsetToJson(basecaller_json["TrainSubset"]);

    SaveJson(basecaller_json, bc_params.GetFiles().filename_json);
    SaveBaseCallerProgress(10, bc_params.GetFiles().output_directory);  // Phase estimation assumed to be 10% of the work
    ReportState(analysis_start_time,"Phase Parameter Estimation Complete");


    // Initialize Barcode Classifier(s) - dependent on phase estimates
    bc.barcodes->BuildPredictedSignals(bc.estimator.GetAverageCF(), bc.estimator.GetAverageIE(), bc.estimator.GetAverageDR());
    bc.calibration_barcodes->BuildPredictedSignals(bc.estimator.GetAverageCF(), bc.estimator.GetAverageIE(), bc.estimator.GetAverageDR());

    MemUsage("BeforeBasecalling");


    //
    // Step 3. Open wells and output BAM files & initialize writers
    //
    JsonToCommentLine(basecaller_bam_comments, bam_comments);

    // Library data set writer - always
    bc.lib_writer.Open(bc_params.GetFiles().output_directory, datasets, 0, bc.chip_subset.NumRegions(),
                 bc.flow_order, bc.keys[0].bases(), filters.GetLibBeadAdapters(), bc_params.NumBamWriterThreads(),
                 basecaller_json, bam_comments, tag_trimmer, barcodes.TrimBarcodes(), bc_params.CompressOutputBam());

    // Calibration reads data set writer - if applicable
    if (bc.have_calibration_panel)
      bc.calib_writer.Open(bc_params.GetFiles().output_directory, datasets_calibration, 0, bc.chip_subset.NumRegions(),
                     bc.flow_order, bc.keys[0].bases(), filters.GetLibBeadAdapters(), bc_params.NumBamWriterThreads(),
                     basecaller_json, bam_comments, tag_trimmer, barcodes.TrimBarcodes(), bc_params.CompressOutputBam());

    // Test fragments data set writer - if applicable
    if (bc.process_tfs)
      bc.tf_writer.Open(bc_params.GetFiles().output_directory, datasets_tf, 1, bc.chip_subset.NumRegions(),
                  bc.flow_order, bc.keys[1].bases(), filters.GetTFBeadAdapters(), bc_params.NumBamWriterThreads(),
                  basecaller_json, bam_comments, tag_trimmer, barcodes.TrimBarcodes(), bc_params.CompressOutputBam());

    // Unfiltered / unfiltered untrimmed data set writers - if applicable
    if (!bc.unfiltered_set.empty()) {
    	bc.unfiltered_writer.Open(bc_params.GetFiles().unfiltered_untrimmed_directory, datasets_unfiltered_untrimmed, -1,
                      bc.chip_subset.NumRegions(), bc.flow_order, bc.keys[0].bases(), filters.GetLibBeadAdapters(),
                      bc_params.NumBamWriterThreads(), basecaller_json, bam_comments, tag_trimmer, barcodes.TrimBarcodes(),
                      bc_params.CompressOutputBam());

        bc.unfiltered_trimmed_writer.Open(bc_params.GetFiles().unfiltered_trimmed_directory, datasets_unfiltered_trimmed, -1,
                              bc.chip_subset.NumRegions(), bc.flow_order, bc.keys[0].bases(), filters.GetLibBeadAdapters(),
                              bc_params.NumBamWriterThreads(), basecaller_json, bam_comments, tag_trimmer, barcodes.TrimBarcodes(),
                              bc_params.CompressOutputBam());
    }

    //
    // Step 4. Execute threaded basecalling
    //

    time_t basecall_start_time;
    time(&basecall_start_time);

    pthread_mutex_init(&bc.mutex, NULL);

    pthread_t worker_id[bc_params.NumThreads()];
    for (int worker = 0; worker < bc_params.NumThreads(); worker++)
        if (pthread_create(&worker_id[worker], NULL, BasecallerWorker, &bc)) {
            printf("*Error* - problem starting thread\n");
            exit (EXIT_FAILURE);
        }

    for (int worker = 0; worker < bc_params.NumThreads(); worker++)
        pthread_join(worker_id[worker], NULL);

    pthread_mutex_destroy(&bc.mutex);

    time_t basecall_end_time;
    time(&basecall_end_time);


    //
    // Step 5. Close files and print out some statistics
    //

    printf("\n\nBASECALLING: called %d of %u wells in %1.0lf seconds with %d threads\n\n",
           filters.NumWellsCalled(), bc.chip_subset.NumWells(),
           difftime(basecall_end_time,basecall_start_time), bc_params.NumThreads());

    bc.lib_writer.Close(datasets, "Library");
    if (bc.have_calibration_panel)
    	bc.calib_writer.Close(datasets_calibration, "IonControl");
    if (bc.process_tfs)
        bc.tf_writer.Close(datasets_tf, "Test Fragments");

    filters.TransferFilteringResultsToMask(mask);

    if (!bc.unfiltered_set.empty()) {

        // Must happen after filters transferred to mask
        bc.WriteUnfilteredFilterStatus(bc_params.GetFiles());

        bc.unfiltered_writer.Close(datasets_unfiltered_untrimmed);
        bc.unfiltered_trimmed_writer.Close(datasets_unfiltered_trimmed);

        datasets_unfiltered_untrimmed.SaveJson(bc_params.GetFiles().unfiltered_untrimmed_directory+"/datasets_basecaller.json");
        datasets_unfiltered_trimmed.SaveJson(bc_params.GetFiles().unfiltered_trimmed_directory+"/datasets_basecaller.json");
    }

    metric_saver.Close();
    barcodes.Close(datasets);
    calibration_barcodes.Close(datasets_calibration);
    if (bc.have_calibration_panel) {
      datasets.json()["IonControl"]["datasets"] = datasets_calibration.json()["datasets"];
      datasets.json()["IonControl"]["read_groups"] = datasets_calibration.read_groups();
    }
    datasets.SaveJson(bc_params.GetFiles().output_directory+"/datasets_basecaller.json");
    if (bc.process_tfs)
        datasets_tf.SaveJson(bc_params.GetFiles().output_directory+"/datasets_tf.json");

    // Generate BaseCaller.json

    bc.lib_writer.SaveFilteringStats(basecaller_json, "lib", true);
    if (bc.have_calibration_panel)
      bc.calib_writer.SaveFilteringStats(basecaller_json, "control", false);
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

    SaveJson(basecaller_json, bc_params.GetFiles().filename_json);
    SaveBaseCallerProgress(100, bc_params.GetFiles().output_directory);

    mask.WriteRaw (bc_params.GetFiles().filename_filter_mask.c_str());
    mask.validateMask();

    MemUsage("AfterBasecalling");
    ReportState(analysis_start_time,"Basecalling Complete");

    return EXIT_SUCCESS;
}

// ----------------------------------------------------------------
//! @brief      Main code for BaseCaller worker thread Mark: XXX
//! @ingroup    BaseCaller
//! @param[in]  input  Pointer to BaseCallerContext.

void * BasecallerWorker(void *input)
{
    BaseCallerContext& bc = *static_cast<BaseCallerContext*>(input);

    RawWells wells ("", bc.filename_wells.c_str());
    pthread_mutex_lock(&bc.mutex);
    wells.OpenForIncrementalRead();
    pthread_mutex_unlock(&bc.mutex);

    WellsNormalization wells_norm(&bc.flow_order, bc.wells_norm_method);
    wells_norm.SetWells(&wells, bc.mask);
    int num_flows =  bc.flow_order.num_flows();

    vector<float>     residual(num_flows, 0);
    vector<float>     scaled_residual(num_flows, 0);
    vector<float>     wells_measurements(num_flows, 0);
    vector<float>     local_noise(num_flows, 0);
    vector<float>     minus_noise_overlap(num_flows, 0);
    vector<float>     homopolymer_rank(num_flows, 0);
    vector<float>     neighborhood_noise(num_flows, 0);
    vector<float>     phasing_parameters(3);
    vector<uint16_t>  flowgram(num_flows);
    vector<int16_t>   flowgram2(num_flows);
    vector<int16_t>   filtering_details(13,0);
    vector<uint8_t>   quality(3*num_flows);
    vector<int>       base_to_flow (3*num_flows);             //!< Flow of in-phase incorporation of each base.
    vector<int>       flow_to_base (num_flows,-1);            //!< base pos of each flow, -1 if not available

    DPTreephaser      treephaser(bc.flow_order, bc.windowSize);
    treephaser.SetStateProgression(bc.diagonal_state_prog);
    treephaser.SkipRecalDuringNormalization(bc.skip_recal_during_norm);
    
    // SSE treephaser definition. XXX
#if defined( __SSE3__ )
    TreephaserSSE treephaser_sse(bc.flow_order, bc.windowSize);
    treephaser_sse.SkipRecalDuringNormalization(bc.skip_recal_during_norm);
#endif


    while (true) {

        //
        // Step 1. Retrieve next unprocessed region
        //

        pthread_mutex_lock(&bc.mutex);

        int current_region, begin_x, begin_y, end_x, end_y;
        if (not bc.chip_subset.GetCurrentRegionAndIncrement(current_region, begin_x, end_x, begin_y, end_y)) {
           wells.Close();
           pthread_mutex_unlock(&bc.mutex);
           return NULL;
        }

        int num_usable_wells = 0;
        for (int y = begin_y; y < end_y; ++y)
            for (int x = begin_x; x < end_x; ++x)
                if (bc.class_map[x + y * bc.chip_subset.GetChipSizeX()] >= 0)
                    num_usable_wells++;

        if      (begin_x == 0)            printf("\n% 5d/% 5d: ", begin_y, bc.chip_subset.GetChipSizeY());
        if      (num_usable_wells ==   0) printf("  ");
        else if (num_usable_wells <  750) printf(". ");
        else if (num_usable_wells < 1500) printf("o ");
        else if (num_usable_wells < 2250) printf("# ");
        else                              printf("##");
        fflush(NULL);

        if (begin_x == 0)
            SaveBaseCallerProgress(10 + (80*begin_y)/bc.chip_subset.GetChipSizeY(), bc.output_directory);

        pthread_mutex_unlock(&bc.mutex);

        // Process the data
        deque<ProcessedRead> lib_reads;                // Collection of template library reads
        deque<ProcessedRead> tf_reads;                 // Collection of test fragment reads
        deque<ProcessedRead> calib_reads;              // Collection of calibration library reads
        deque<ProcessedRead> unfiltered_reads;         // Random subset of lib_reads
        deque<ProcessedRead> unfiltered_trimmed_reads; // Random subset of lib_reads

        if (num_usable_wells == 0) { // There is nothing in this region. Don't even bother reading it
            bc.lib_writer.WriteRegion(current_region, lib_reads);
            if (bc.have_calibration_panel)
                bc.calib_writer.WriteRegion(current_region, calib_reads);
            if (bc.process_tfs)
                bc.tf_writer.WriteRegion(current_region, tf_reads);
            if (!bc.unfiltered_set.empty()) {
                bc.unfiltered_writer.WriteRegion(current_region,unfiltered_reads);
                bc.unfiltered_trimmed_writer.WriteRegion(current_region,unfiltered_trimmed_reads);
            }
            continue;
        }

        wells.SetChunk(begin_y, end_y-begin_y, begin_x, end_x-begin_x, 0, num_flows);
        wells.ReadWells();
        wells_norm.CorrectSignalBias(bc.keys);
        wells_norm.DoKeyNormalization(bc.keys);


        for (int y = begin_y; y < end_y; ++y)
            for (int x = begin_x; x < end_x; ++x) {   // Loop over wells within current region

                //
                // Step 2. Retrieve additional information needed to process this read
                //

                unsigned int read_index = x + y * bc.chip_subset.GetChipSizeX();
                int read_class = bc.class_map[read_index];
                if (read_class < 0)
                    continue;
                bool is_random_calibration_read = false;
                if (read_class == 2){
                  is_random_calibration_read = true;
                  read_class = 0; // Calibration reads are library beads;
                }
                bool is_random_unfiltered  = bc.unfiltered_set.count(read_index) > 0;

                if (not is_random_unfiltered and bc.only_process_unfiltered_set)
                  continue;

                bc.filters->SetValid(read_index); // Presume valid until some filter proves otherwise

                if (read_class == 0)
                    lib_reads.push_back(ProcessedRead(bc.barcodes->NoBarcodeReadGroup()));
                else
                    tf_reads.push_back(ProcessedRead(0));
                ProcessedRead& processed_read = (read_class==0) ? lib_reads.back() : tf_reads.back();

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

                for (int flow = 0; flow < num_flows; ++flow)
                    wells_measurements[flow] = wells.At(y,x,flow);

                // Sanity check. If there are NaNs in this read, print warning
                vector<int> nanflow;
                for (int flow = 0; flow < num_flows; ++flow) {
                    if (!isnan(wells_measurements[flow]))
                        continue;
                    wells_measurements[flow] = 0;
                    nanflow.push_back(flow);
                }
                if (nanflow.size() > 0) {
                    fprintf(stderr, "ERROR: BaseCaller read NaNs from wells file, x=%d y=%d flow=%d", x, y, nanflow[0]);
                    for (unsigned int flow=1; flow < nanflow.size(); flow++) {
                        fprintf(stderr, ",%d", nanflow[flow]);
                    }
                    fprintf(stderr, "\n");
                    fflush(stderr);
                }

                //
                // Step 3. Perform base calling and quality value calculation
                //

                BasecallerRead read;
                bool key_pass = true;
                if (bc.keynormalizer == "adaptive") {
                  key_pass = read.SetDataAndKeyNormalizeNew(&wells_measurements[0], wells_measurements.size(), bc.keys[read_class].flows(), bc.keys[read_class].flows_length() - 1, false);
                }
                else if (bc.keynormalizer == "off") {
                  key_pass = read.SetDataAndKeyPass(wells_measurements, wells_measurements.size(), bc.keys[read_class].flows(), bc.keys[read_class].flows_length() - 1);
                }
                else { // if (bc.keynormalizer == "default") {
                  key_pass = read.SetDataAndKeyNormalize(&wells_measurements[0], wells_measurements.size(), bc.keys[read_class].flows(), bc.keys[read_class].flows_length() - 1);
                }

                // Get rid of outliers quickly
                bc.filters->FilterHighPPFAndPolyclonal (read_index, read_class, processed_read.filter, read.raw_measurements);
                if (not key_pass)
                  bc.filters->FilterFailedKeypass (read_index, read_class, processed_read.filter, read.sequence);
                if (!is_random_unfiltered and !bc.filters->IsValid(read_index)) // No reason to waste more time
                  continue;

                // Check if this read is either from the calibration panel or from the random calibration set
                if(bc.calibration_training and bc.have_calibration_panel) {
                  if (!is_random_calibration_read and !bc.calibration_barcodes->MatchesBarcodeSignal(read)) {
                	bc.filters->SetFiltered(read_index, read_class, processed_read.filter); // Set as filtered
                    continue;  // And move on along
                  }
                }

                // Equal calibration opportunity for everybody! (except TFs!)
                const vector<vector<vector<float> > > * aPtr = 0;
                const vector<vector<vector<float> > > * bPtr = 0;
                if (bc.linear_cal_model->is_enabled() && read_class == 0) { //do not recalibrate TFs
                  aPtr = bc.linear_cal_model->getAs(x+bc.chip_subset.GetColOffset(), y+bc.chip_subset.GetRowOffset());
                  bPtr = bc.linear_cal_model->getBs(x+bc.chip_subset.GetColOffset(), y+bc.chip_subset.GetRowOffset());
                }

                // Execute the iterative solving-normalization routine - switch by specified algorithm
                // Structure code by SSE or CPP code call

                bool compute_base_calls = true;
                bool calibrate_read = bc.histogram_calibration->is_enabled();
                calibrate_read = calibrate_read and (read_class == 0 or (bc.calibrate_TFs and read_class == 1));

                treephaser.SetAsBs(aPtr, bPtr); // Set/delete recalibration model for this read
                // Set up CPP base caller - Adapter trimming always uses cpp treephaser
                if (bc.skip_droop or bc.sse_dephaser)
                  treephaser.SetModelParameters(cf, ie);
                else
                  treephaser.SetModelParameters(cf, ie, dr);

                // Execute vectorized basecaller version XXX
#if defined( __SSE3__ )
                if (bc.sse_dephaser) {
                  treephaser_sse.SetAsBs(aPtr, bPtr);  // Set/delete recalibration model for this read
                  treephaser_sse.SetModelParameters(cf, ie); // SSE version has no hook for droop.

                  if (bc.dephaser == "treephaser-sse")
                    treephaser_sse.NormalizeAndSolve(read);
                  else // bc.dephaser == "treephaser-solve" Solving without normalization
                    treephaser_sse.SolveRead(read, 0, num_flows);

                  // Store debug info if desired and calibration enabled
                  if (bc.debug_normalization_bam and bc.histogram_calibration->is_enabled())
                    read.not_calibrated_measurements = read.normalized_measurements;

                  // Calibrate library reads
                  if (calibrate_read) {
                    bc.histogram_calibration->PolishRead(bc.flow_order, x+bc.chip_subset.GetColOffset(), y+bc.chip_subset.GetRowOffset(), read, bc.linear_cal_model);
                  }

                  // Compute QV metrics on pot. calibrated sequence
                  // Generate base_to_flow before ComputeQVmetrics. But not too early, otherwise the sequence is not ready
                  make_base_to_flow(read.sequence,bc.flow_order,base_to_flow,flow_to_base,num_flows);

                  treephaser_sse.ComputeQVmetrics_flow(read,flow_to_base,bc.flow_predictors_);
                  compute_base_calls = false;
                }
#endif

                // Use CPP code version if we didn't already use vectorized code
                if (compute_base_calls){
                  if (bc.dephaser == "dp-treephaser") {
                    // Single parameter gain estimation
                    treephaser.NormalizeAndSolve_GainNorm(read, num_flows);
                  } else if (bc.dephaser == "treephaser-adaptive") {
                    // Adaptive nortmalization - resolving read from start in each iteration
                    treephaser.NormalizeAndSolve_Adaptive(read, num_flows);
                  } else { //if (bc.dephaser == "treephaser-swan") {
                    // Default corresponding to (approximately) what the sse version is doing
                    // Adaptive normalization - sliding window without resolving start
                    treephaser.NormalizeAndSolve_SWnorm(read, num_flows);
                  }
                  
                  // Store debug info if desired and calibration enabled
                  if (bc.debug_normalization_bam and bc.histogram_calibration->is_enabled())
                    read.not_calibrated_measurements = read.normalized_measurements;

                  // Calibrate library reads
                  if (calibrate_read) {
                    treephaser.Simulate(read, num_flows, true);
                    bc.histogram_calibration->PolishRead(bc.flow_order, x+bc.chip_subset.GetColOffset(), y+bc.chip_subset.GetRowOffset(), read, bc.linear_cal_model);
                  }

                  // Compute QV metrics on pot. calibrated sequence
                  // Generate base_to_flow before ComputeQVmetrics. But not too early, otherwise the sequence is not ready
                  make_base_to_flow(read.sequence,bc.flow_order,base_to_flow,flow_to_base,num_flows);

                  treephaser.ComputeQVmetrics_flow(read,flow_to_base,bc.flow_predictors_);
                }

                // Misc data management: Generate residual, scaled_residual
                for (int flow = 0; flow < num_flows; ++flow) {
                    residual[flow] = read.normalized_measurements[flow] - read.prediction[flow];
                    scaled_residual[flow] = residual[flow] / read.state_inphase[flow];
                }

                // Misc data management: Put base calls in proper string form
                processed_read.filter.CalledRead(read.sequence.size());

                // Misc data management: Populate some trivial read properties

                char read_name[256];
                sprintf(read_name, "%s:%05d:%05d", bc.run_id.c_str(), bc.chip_subset.GetRowOffset() + y, bc.chip_subset.GetColOffset() + x);
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

                //char short_name[20]; // x:y only
                //sprintf(short_name, "%05d:%05d", bc.chip_subset.GetRowOffset() + y, bc.chip_subset.GetColOffset() + x);
                //bc.quality_generator.GenerateBaseQualities(short_name, processed_read.filter.n_bases, num_flows,
                vector<float> homopolymer_rank_flow(num_flows, 0);
                bool use_flow_predictors = bc.flow_predictors_;
                if (bc.quality_generator.toSavePredictors()) {
                    int num_predictor_bases = min(num_flows, processed_read.filter.n_bases);
                    PerBaseQual::PredictorLocalNoise(local_noise, num_predictor_bases, base_to_flow, read.normalized_measurements, read.prediction,use_flow_predictors);
                    PerBaseQual::PredictorNeighborhoodNoise(neighborhood_noise, num_predictor_bases, base_to_flow, read.normalized_measurements, read.prediction,use_flow_predictors);
                    //PerBaseQual::PredictorNoiseOverlap(minus_noise_overlap, num_predictor_bases, read.normalized_measurements, read.prediction,use_flow_predictors);
                    PerBaseQual::PredictorBeverlyEvents(minus_noise_overlap, num_predictor_bases, base_to_flow, scaled_residual,use_flow_predictors);
                    PerBaseQual::PredictorHomopolymerRank(homopolymer_rank, num_predictor_bases, read.sequence, homopolymer_rank_flow, flow_to_base, use_flow_predictors);

                    bc.quality_generator.DumpPredictors(read_name, processed_read.filter.n_bases, num_flows,
                            read.penalty_residual, local_noise, minus_noise_overlap, // <- predictors 1,2,3
                            homopolymer_rank, read.penalty_mismatch, neighborhood_noise, // <- predictors 4,5,6
                            base_to_flow, quality,
                            read.additive_correction,
                            read.multiplicative_correction,
                            read.state_inphase,
                            read.penalty_residual_flow,
                            read.penalty_mismatch_flow,
                            homopolymer_rank_flow,
                            flow_to_base,
                            use_flow_predictors);
                }

                // 8/18/2016, ew: still use base-predictors to predict quality scores for now
                use_flow_predictors = false;
                bool redo_genPred = ((! bc.quality_generator.toSavePredictors()) || bc.flow_predictors_) ? true:false;
                //redo_genPred = true;
                if (redo_genPred) {
					int num_predictor_bases = min(num_flows, processed_read.filter.n_bases);
                    PerBaseQual::PredictorLocalNoise(local_noise, num_predictor_bases, base_to_flow, read.normalized_measurements, read.prediction,use_flow_predictors);
                    PerBaseQual::PredictorNeighborhoodNoise(neighborhood_noise, num_predictor_bases, base_to_flow, read.normalized_measurements, read.prediction,use_flow_predictors);
                    //PerBaseQual::PredictorNoiseOverlap(minus_noise_overlap, num_predictor_bases, read.normalized_measurements, read.prediction,use_flow_predictors);
                    PerBaseQual::PredictorBeverlyEvents(minus_noise_overlap, num_predictor_bases, base_to_flow, scaled_residual,use_flow_predictors);
                    PerBaseQual::PredictorHomopolymerRank(homopolymer_rank, num_predictor_bases, read.sequence, homopolymer_rank_flow, flow_to_base, use_flow_predictors);
                }
                //bc.quality_generator.GenerateBaseQualities(processed_read.bam.Name, processed_read.filter.n_bases, num_flows,
                bc.quality_generator.GenerateBaseQualities(read_name, processed_read.filter.n_bases, num_flows,
                        read.penalty_residual, local_noise, minus_noise_overlap, // <- predictors 1,2,3
                        homopolymer_rank, read.penalty_mismatch, neighborhood_noise, // <- predictors 4,5,6
                        base_to_flow, quality,
                        read.additive_correction,
                        read.multiplicative_correction,
                        read.state_inphase,
                        read.penalty_residual_flow,
                        read.penalty_mismatch_flow,
                        homopolymer_rank_flow,
                        flow_to_base,
                        use_flow_predictors);

                //
                // Step 4a. Barcode classification of library reads
                //

                bc.filters->TrimKeySequence(bc.keys[read_class].bases_length(), processed_read.filter);

                if (read_class == 0)
                {   // Library beads - first separate out calibration barcodes
                	processed_read.read_group_index = -1;
                	if (bc.have_calibration_panel){
                	  bc.calibration_barcodes->ClassifyAndTrimBarcode(read_index, processed_read, read, base_to_flow);
                	  processed_read.is_control_barcode = (processed_read.read_group_index >= 0);
                	}

                	// *** Processing for standard library beads only
                    if (processed_read.read_group_index < 0){
                      bc.barcodes->ClassifyAndTrimBarcode(read_index, processed_read, read, base_to_flow);
                      bc.filters->TrimPrefixTag          (read_index, read_class, processed_read, read.sequence, bc.tag_trimmer);
                      bc.filters->TrimExtraLeft          (read_class, processed_read, read.sequence); // Only those guys get an extra trim left
                    }
                }


                //
                // Step 4. Calculate/save read metrics and apply filters
                // The order of the filtering/trimming operations actually matters.

                // Filters completely remove a read from the BAM
                bc.filters->FilterZeroBases     (read_index, read_class, processed_read.filter);
                bc.filters->FilterShortRead     (read_index, read_class, processed_read.filter);
                bc.filters->FilterFailedKeypass (read_index, read_class, processed_read.filter, read.sequence);
                bc.filters->FilterHighResidual  (read_index, read_class, processed_read.filter, residual);
                //bc.filters->FilterBeverly       (read_index, read_class, processed_read.filter, scaled_residual, base_to_flow);
                bc.filters->FilterQuality       (read_index, read_class, processed_read.filter, quality);

                // Read trimming shortens a read or remove it if it's too short after trimming
                bc.filters->TrimAdapter         (read_index, read_class, processed_read, scaled_residual, base_to_flow, treephaser, read);
                bc.filters->TrimSuffixTag       (read_index, read_class, processed_read, read.sequence, bc.tag_trimmer);
                bc.filters->TrimExtraRight      (read_index, read_class, processed_read, read.sequence);
                bc.filters->TrimQuality         (read_index, read_class, processed_read.filter, quality);

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

                int max_flow = num_flows;
                if (bc.trim_zm){
                  if (processed_read.filter.n_bases_filtered > 0)
                    max_flow = min(num_flows, base_to_flow[processed_read.filter.n_bases_filtered-1] + 16);
                  else
                    max_flow = min(num_flows,16);
                }

                scaleup_flowgram(read.normalized_measurements,flowgram2,max_flow);
                processed_read.bam.AddTag("ZM", flowgram2);
                if (bc.debug_normalization_bam) {
                    scaleup_flowgram(read.additive_correction,flowgram2,max_flow);
                    processed_read.bam.AddTag("Ya", flowgram2);
                    scaleup_flowgram(read.multiplicative_correction,flowgram2,max_flow);
                    processed_read.bam.AddTag("Yb", flowgram2);
                    scaleup_flowgram(read.raw_measurements,flowgram2,max_flow);
                    processed_read.bam.AddTag("Yw", flowgram2);
                    if (bc.histogram_calibration->is_enabled()) {
                      scaleup_flowgram(read.not_calibrated_measurements,flowgram2,max_flow);
                      processed_read.bam.AddTag("Yx", flowgram2);
                    }
                }

                //
                // Step 4c. Populate FZ tag in BAM record.
                //

                flowgram.clear();
                if (bc.flow_signals_type == "wells") {
                    for (int flow = 0; flow < num_flows; ++flow)
                        flowgram.push_back(max(0,(int)(100.0*wells_measurements[flow]+0.5)));
                    processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

                } else if (bc.flow_signals_type == "key-normalized") {
                    for (int flow = 0; flow < num_flows; ++flow)
                        flowgram.push_back(max(0,(int)(100.0*read.raw_measurements[flow]+0.5)));
                    processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

                } else if (bc.flow_signals_type == "adaptive-normalized") {
                    for (int flow = 0; flow < num_flows; ++flow)
                        flowgram.push_back(max(0,(int)(100.0*read.normalized_measurements[flow]+0.5)));
                    processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

                } else if (bc.flow_signals_type == "residual") {
                    for (int flow = 0; flow < num_flows; ++flow)
                        flowgram.push_back(max(0,(int)(1000 + 100*residual[flow])));
                    processed_read.bam.AddTag("FZ", flowgram); // Will be phased out soon

                } else if (bc.flow_signals_type == "scaled-residual") { // This settings is necessary part of calibration training
                    for (int flow = 0; flow < num_flows; ++flow) {
                        //between 0 and 98
                        float adjustment = min(0.49f, max(-0.49f, scaled_residual[flow]));
                        flowgram.push_back(max(0,(int)(49.5 + 100*adjustment)));
                    }
                    processed_read.bam.AddTag("FZ", flowgram);
                }

                //
                // Step 5. Pass basecalled reads to appropriate writers
                //

                // Create BAM entries
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

                // Randomly selected library beads - excluding calibration reads
                if (is_random_unfiltered and (not processed_read.is_control_barcode)) {
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

                // Move read from lib_reads stack to calib_reads if necessary
                // This invalidates the processed_read reference and needs to be at the very end
                if (processed_read.is_control_barcode) {
                  calib_reads.push_back(processed_read);
                  lib_reads.pop_back();
                }
            }

        bc.lib_writer.WriteRegion(current_region, lib_reads);
        if (bc.have_calibration_panel)
            bc.calib_writer.WriteRegion(current_region, calib_reads);
        if (bc.process_tfs)
            bc.tf_writer.WriteRegion(current_region, tf_reads);
        if (!bc.unfiltered_set.empty()) {
            bc.unfiltered_writer.WriteRegion(current_region,unfiltered_reads);
            bc.unfiltered_trimmed_writer.WriteRegion(current_region,unfiltered_trimmed_reads);
        }
    }
}


