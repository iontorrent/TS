/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     BaseCallerParameters.h
//! @ingroup  BaseCaller
//! @brief    Command line option reading and storage for BaseCaller modules

#include "BaseCallerParameters.h"
#include "MolecularTagTrimmer.h"


void SaveJson(const Json::Value & json, const string& filename_json); // Borrow from Basecaller.cpp

void ValidateAndCanonicalizePath(string &path)
{
    char *real_path = realpath (path.c_str(), NULL);
    if (real_path == NULL) {
        perror(path.c_str());
        exit(EXIT_FAILURE);
    }
    path = real_path;
    free(real_path);
};

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
};

// ==================================================================

bool BaseCallerContext::SetKeyAndFlowOrder(OptArgs& opts, const char * FlowOrder, const int NumFlows)
{
    flow_order.SetFlowOrder( opts.GetFirstString ('-', "flow-order", FlowOrder),
                             opts.GetFirstInt    ('f', "flowlimit", NumFlows));
    if (flow_order.num_flows() > NumFlows)
      flow_order.SetNumFlows(NumFlows);
    assert(flow_order.is_ok());

    string lib_key                = opts.GetFirstString ('-', "lib-key", "TCAG"); //! @todo Get default key from wells
    string tf_key                 = opts.GetFirstString ('-', "tf-key", "ATCG");
    lib_key                       = opts.GetFirstString ('-', "librarykey", lib_key);   // Backward compatible opts
    tf_key                        = opts.GetFirstString ('-', "tfkey", tf_key);
    keys.resize(2);
    keys[0].Set(flow_order, lib_key, "lib");
    keys[1].Set(flow_order, tf_key, "tf");
    return true;
};

void BaseCallerContext::ClassifyAndSampleWells(const BCwellSampling & SamplingOpts)
{
    ReservoirSample<unsigned int> downsampled_subset(SamplingOpts.downsample_size, 2);
    ReservoirSample<unsigned int> unfiltered_subset(SamplingOpts.num_unfiltered, 1);
    bool eval_all_libWells = SamplingOpts.downsample_size == 0 or SamplingOpts.have_calib_panel;

    // First iteration over wells to sample them and/or assign read class
    for (int y = chip_subset.GetBeginY(); y < chip_subset.GetEndY(); ++y) {
      for (int x = chip_subset.GetBeginX(); x < chip_subset.GetEndX(); ++x) {

        int well_index  = x + y * chip_subset.GetChipSizeX();
        if (read_class_map->ClassMatch(x, y, MapLibrary)) {
          // For calibration training set / downsampling set we exclude already filtered reads
          // We however mark all filtered reads as output well to preserve correct accounting of filtered reads
          // Unfiltered set contains a selection of randomly selected output library beads

          if (SamplingOpts.downsample_size>0) {
            if (not read_class_map->ClassMatch(x, y, MapFiltered))
              downsampled_subset.Add(well_index);
            else {
              read_class_map->setClassType(well_index, MapOutputWell);
              if (SamplingOpts.num_unfiltered>0)
                unfiltered_subset.Add(well_index);
            }
          }
          else if (SamplingOpts.num_unfiltered>0){
            unfiltered_subset.Add(well_index);
          }
          if (eval_all_libWells)
            read_class_map->setClassType(well_index, MapOutputWell);
        }

        if (read_class_map->ClassMatch(x, y, MapTF) and process_tfs) {
          if (SamplingOpts.downsample_size>0)
            downsampled_subset.Add(well_index);
          else
            read_class_map->setClassType(well_index, MapOutputWell);
        }
      }
    }

    // Another pass over the read class map to set our sampled subsets
    downsampled_subset.Finished();
    if (SamplingOpts.downsample_size > 0) {
      for (size_t idx=0; idx<downsampled_subset.GetCount(); idx++) {
        read_class_map->setClassType(downsampled_subset.GetVal(idx), MapOutputWell);
        // Add to random unfiltered set
        if (SamplingOpts.num_unfiltered>0)
          unfiltered_subset.Add(downsampled_subset.GetVal(idx));
        // Mark random calibration sample to be used in addition to calibration panel reads
        if (SamplingOpts.have_calib_panel)
          read_class_map->setClassType(downsampled_subset.GetVal(idx), MapCalibration);
      }
    }

    unfiltered_subset.Finished();
    if (SamplingOpts.num_unfiltered > 0){
      unfiltered_set.insert(unfiltered_subset.GetData().begin(), unfiltered_subset.GetData().end());
      for (size_t idx=0; idx<unfiltered_subset.GetCount(); idx++) {
        read_class_map->setClassType(unfiltered_subset.GetVal(idx), MapOutputWell);
        read_class_map->setClassType(unfiltered_subset.GetVal(idx), MapUnfiltered);
      }
    }

    // Print Summary:
    unsigned int sum_tf_wells    = 0;
    unsigned int sum_lib_wells   = 0;
    unsigned int sum_calib_wells = 0;
    unsigned int sum_output_wells= 0;


    for (unsigned int idx=0; idx<read_class_map->getNumWells(); idx++) {
      if      (read_class_map->ClassMatch(idx, MapOutputWell))  ++sum_output_wells;
      if      (read_class_map->ClassMatch(idx, MapCalibration)) ++sum_calib_wells;
      else if (read_class_map->ClassMatch(idx, MapLibrary))     ++sum_lib_wells;
      else if (read_class_map->ClassMatch(idx, MapTF))          ++sum_tf_wells;
    }
    cout << "Bead classification summary:" << endl;
    cout << " - Total num. wells      : " << read_class_map->getNumWells() << endl;
    cout << " - Num. Library wells    : " << sum_lib_wells       << endl;
    cout << " - Num. Test fragments   : " << sum_tf_wells        << endl;
    cout << " - Num. calib. wells     : " << sum_calib_wells     << endl;
    cout << " - Num. unfiltered wells : " << unfiltered_set.size() << endl;
    cout << " - Num. output wells     : " << sum_output_wells    << endl;
};


bool BaseCallerContext::WriteUnfilteredFilterStatus(const BaseCallerFiles & bc_files) {

    ofstream filter_status;

    string filter_status_filename = bc_files.unfiltered_untrimmed_directory + string("/filterStatus.txt");
    filter_status.open(filter_status_filename.c_str());
    filter_status << "col" << "\t" << "row" << "\t" << "highRes" << "\t" << "valid" << endl;
    for (set<unsigned int>::iterator I = unfiltered_set.begin(); I != unfiltered_set.end(); ++I) {
        int x = (*I) % chip_subset.GetChipSizeX();
        int y = (*I) / chip_subset.GetChipSizeX();
        filter_status << x << "\t" << y;
        filter_status << "\t" << (int) read_class_map->MaskMatch(x, y, MaskFilteredBadResidual); // Must happen after filters transferred to mask
        filter_status << "\t" << (int) read_class_map->MaskMatch(x, y, MaskKeypass);
        filter_status << endl;
    }
    filter_status.close();

    filter_status_filename = bc_files.unfiltered_trimmed_directory + string("/filterStatus.txt");
    filter_status.open(filter_status_filename.c_str());
    filter_status << "col" << "\t" << "row" << "\t" << "highRes" << "\t" << "valid" << endl;
    for (set<unsigned int>::iterator I = unfiltered_set.begin(); I != unfiltered_set.end(); ++I) {
        int x = (*I) % chip_subset.GetChipSizeX();
        int y = (*I) / chip_subset.GetChipSizeX();
        filter_status << x << "\t" << y;
        filter_status << "\t" << (int) read_class_map->MaskMatch(x, y, MaskFilteredBadResidual); // Must happen after filters transferred to mask
        filter_status << "\t" << (int) read_class_map->MaskMatch(x, y, MaskKeypass);
        filter_status << endl;
    }
    filter_status.close();

    return true;
};

// ==================================================================

//! @brief    Print BaseCaller usage.
//! @ingroup  BaseCaller

void BaseCallerParameters::PrintHelp()
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
    printf ("     --compress-bam          BOOL       Output compressed / uncompressed BAM [true]\n");
    printf ("  -f,--flowlimit             INT        basecall only first n flows [all flows]\n");
    printf ("     --keynormalizer         STRING     key normalization algorithm [gain]\n");
    printf ("     --wells-normalization   STRING     normalize wells signal and correct for signal bias [off]/on/keyOnly/signalBiasOnly/pinZero\n");
#if defined( __SSE3__ )
    printf ("     --dephaser              STRING     dephasing algorithm e.g. treephaser-sse, treephaser-solve, dp-treephaser, treephaser-adaptive, treephaser-swan  [treephaser-sse]\n");
#else
    printf ("     --dephaser              STRING     dephasing algorithm e.g. dp-treephaser, treephaser-adaptive, treephaser-swan [treephaser-swan]\n");
#endif
    printf ("     --window-size           INT        normalization window size (%d-%d) [%d]\n", DPTreephaser::kMinWindowSize_, DPTreephaser::kMaxWindowSize_, DPTreephaser::kWindowSizeDefault_);
    printf ("     --flow-signals-type     STRING     select content of FZ tag [none]\n");
    printf ("                                          \"none\" - FZ not generated\n");
    printf ("                                          \"wells\" - Raw values (unnormalized and not dephased)\n");
    printf ("                                          \"key-normalized\" - Key normalized and not dephased\n");
    printf ("                                          \"adaptive-normalized\" - Adaptive normalized and not dephased\n");
    printf ("                                          \"residual\" - Measurement-prediction residual\n");
    printf ("                                          \"scaled-residual\" - Scaled measurement-prediction residual\n");
    printf ("     --compress-multi-taps   BOOL       Compress the signal from adjacent multi-tap flows [false]\n");
    printf ("     --num-unfiltered        INT        number of subsampled unfiltered reads [100000]\n");
    printf ("     --only-process-unfiltered-set   on/off   Only save reads that would also go to unfiltered BAMs. [off]\n");
    printf ("\n");
    printf ("Chip/Block division:\n");
    printf ("  -r,--rows                  INT-INT    subset of rows to be processed [all rows]\n");
    printf ("  -c,--cols                  INT-INT    subset of columns to be processed [all columns]\n");
    printf ("     --region-size           INT,INT    region size (x,y) for processing [50x50]\n");
    printf ("     --block-offset          INT,INT    region offset (x,y) added to read coordinates\n");
    printf ("     --downsample-fraction   FLOAT      Only save a fraction of generated reads. 1.0 saves all reads. [1.0]\n");
    printf ("     --downsample-size       INT        Only save up to 'downsample-size' reads. [0]\n");
    printf ("\n");
    printf ("Calibration Options:\n");
    printf ("     --calibration-training  INT        Generate training set of INT reads. No TFs, no unfiltered sets. -1=off [-1]\n");
    printf ("     --calibration-panel     FILE       Datasets json for calibration panel reads to be used for training [off]\n");
    printf ("     --calibration-json      FILE       Enable Calibration using models from provided json file [off]\n");
    printf ("     --model-file            FILE       Legacy text input file for LinearModelCalibration [off]\n");
    printf ("     --calibration-file      FILE       Legacy text input file for HistogramCalibration [off]\n");
    printf ("     --calibrate-tfs         FILE       Calibrate test fragment reads [off]\n");
    printf ("\n");
    printf ("Inline Control Options:\n");
    printf ("     --inline-control        BOOL       Enable inline control [off]\n");
    printf ("     --inlinecontrol-reference       FILE       Fasta file for inline control\n");
    printf ("\n");
    printf ("Debug Options:\n");
    printf ("     --debug-normalization-bam  BOOL       Output debug data to the bam tags Ya, Yb, Yw, and Yx, for the adaptive offset, adaptive slope, well-normalized measurements, and not-calibrated measurements [off]\n");
    printf ("\n");

    BaseCallerFilters::PrintHelp();
    PhaseEstimator::PrintHelp();
    PerBaseQual::PrintHelp();
    BarcodeClassifier::PrintHelp();
#ifndef DATAVIEWER
    EndBarcodeClassifier::PrintHelp();
#endif
    MolecularTagTrimmer::PrintHelp(false);
    BaseCallerMetricSaver::PrintHelp();

    exit (EXIT_SUCCESS);
};


// ----------------------------------------------------------------------
// XXX Input options

bool BaseCallerParameters::InitializeFilesFromOptArgs(OptArgs& opts)
{
    // Vectorized wells input to load&Analyze multiple wells files
    // If explicitly specified --wells and --mask arguments need to contain the full paths
    // Default names for wells is 1.wells in teh respective input directories and
    // default names for mask is analysis.bfmask.bin

    bc_files.input_directory        = opts.GetFirstStringVector ('i', "input-dir", ".");
    bc_files.filename_wells         = opts.GetFirstStringVector ('-', "wells", "");
    bc_files.filename_mask          = opts.GetFirstStringVector ('-', "mask", "");

    // Read block offset
    // Coordinate offset is addressable in row,col and x,y format, with x,y taking precedence
    int block_offset_x = opts.GetFirstInt    ('-', "block-col-offset", 0);
    int block_offset_y = opts.GetFirstInt    ('-', "block-row-offset", 0);
    std::stringstream default_opt_val;
    default_opt_val << block_offset_x << ',' << block_offset_y;
    std::vector<int> arg_block_offset  = opts.GetFirstIntVector ('-', "block-offset", default_opt_val.str(), ',');
    if (arg_block_offset.size() != 2) {
      std::cerr << "BaseCaller Option Error: argument 'block-offset' needs to be 2 comma separated values <Int>,<Int>" << std::endl;
      exit (EXIT_FAILURE);
    }
    block_offset_x = arg_block_offset.at(0);
    block_offset_y = arg_block_offset.at(1);
    std::stringstream block;
    block << "/block_X" << block_offset_x << "_Y" << block_offset_y;

    if (bc_files.filename_wells.empty() and bc_files.filename_mask.empty())
    {
      for (unsigned int i=0; i<bc_files.input_directory.size(); ++i){
    	string input_org(bc_files.input_directory[i]);
    	bc_files.input_directory[i] += block.str();
    	cout << "Trying to read raw data from " << bc_files.input_directory[i]
    	     << " or " << input_org << endl;
        ValidateAndCanonicalizePath(bc_files.input_directory[i], input_org);
        bc_files.filename_wells.push_back(bc_files.input_directory[i] + "/1.wells");
        bc_files.filename_mask.push_back(bc_files.input_directory[i] + "/analysis.bfmask.bin");
      }
    }
    else if (bc_files.filename_wells.size() != bc_files.filename_mask.size())
    {
      cerr << "ERROR: Options --wells and --mask need to be vectors of the same length." << endl;
      exit(EXIT_FAILURE);
    }

    for (unsigned int i=0; i<bc_files.filename_wells.size(); ++i){
      ValidateAndCanonicalizePath(bc_files.filename_wells[i]);
      ValidateAndCanonicalizePath(bc_files.filename_mask[i], bc_files.input_directory[i] + "/bfmask.bin");
    }

    bc_files.output_directory       = opts.GetFirstString ('o', "output-dir", ".");
    bc_files.unfiltered_untrimmed_directory = bc_files.output_directory + "/unfiltered.untrimmed";
    bc_files.unfiltered_trimmed_directory   = bc_files.output_directory + "/unfiltered.trimmed";

    CreateResultsFolder ((char*)bc_files.output_directory.c_str());
    CreateResultsFolder ((char*)bc_files.unfiltered_untrimmed_directory.c_str());
    CreateResultsFolder ((char*)bc_files.unfiltered_trimmed_directory.c_str());


    ValidateAndCanonicalizePath(bc_files.output_directory);
    ValidateAndCanonicalizePath(bc_files.unfiltered_untrimmed_directory);
    ValidateAndCanonicalizePath(bc_files.unfiltered_trimmed_directory);

    bc_files.filename_filter_mask   = bc_files.output_directory + "/basecaller.bfmask.bin";
    bc_files.filename_json          = bc_files.output_directory + "/BaseCaller.json";
    bc_files.filename_phase         = bc_files.output_directory + "/PhaseEstimates.json";

    printf("\n");
    printf("Input files summary:\n");
    printf("     --input-dir %s\n", bc_files.input_directory[0].c_str());
    for (unsigned int i=1; i<bc_files.input_directory.size(); ++i)
      cout<< ',' << bc_files.input_directory[i] <<endl;
    printf("         --wells %s\n", bc_files.filename_wells[0].c_str());
    for (unsigned int i=1; i<bc_files.filename_wells.size(); ++i)
          cout<< ',' << bc_files.filename_wells[i] <<endl;
    printf("          --mask %s\n", bc_files.filename_mask[0].c_str());
    for (unsigned int i=1; i<bc_files.filename_mask.size(); ++i)
          cout<< ',' << bc_files.filename_mask[i] <<endl;
    printf("\n");
    printf("Output directories summary:\n");
    printf("    --output-dir %s\n", bc_files.output_directory.c_str());
    printf("        unf.untr %s\n", bc_files.unfiltered_untrimmed_directory.c_str());
    printf("          unf.tr %s\n", bc_files.unfiltered_trimmed_directory.c_str());
    printf("\n");

    bc_files.lib_datasets_file      = opts.GetFirstString ('-', "datasets", "");
    bc_files.calibration_panel_file = opts.GetFirstString ('-', "calibration-panel", "");
    bc_files.inline_control_reference_file = opts.GetFirstString ('-', "inlinecontrol-reference", "/opt/ion/config/inline_controls_ref.fasta");

    if (not bc_files.lib_datasets_file.empty())
      ValidateAndCanonicalizePath(bc_files.lib_datasets_file);
    if (not bc_files.calibration_panel_file.empty())
      ValidateAndCanonicalizePath(bc_files.calibration_panel_file);

    string structure_file = opts.GetFirstString ('-', "read-structure-file", "/opt/ion/config/StructureMetaInfo.json");
    string app_name = opts.GetFirstString ('-', "read-structure", "");
    Json::Value structure_in;

    if (not app_name.empty()) {

      ifstream in(structure_file.c_str(), ifstream::in);
      if (!in.good()) {
        cerr << "ERROR: Opening file " << structure_file << " unsuccessful. Aborting" << endl;
        exit(EXIT_FAILURE);
      }
      in >> structure_in;

      if (not structure_in.isMember(app_name)){
        cerr << "ERROR: Structure file " << structure_file << " does not contain a member " << app_name << endl;
        exit(EXIT_FAILURE);
      }
      else {
        if (structure_in[app_name].isMember("adapter")) // XXX Dummy for now
          bc_files.read_structure["adapter"] = getNormString(structure_in[app_name]["adapter"].asString());
        if (structure_in[app_name].isMember("handle")){
          bc_files.read_structure["handle"] = NormalizeDictStructure(structure_in[app_name]["handle"]);
          cout << "Structure handle : " << structure_in[app_name]["handle"].toStyledString() << endl;
        }
        if (structure_in[app_name].isMember("tag")){
          bc_files.read_structure["tag"] = NormalizeDictStructure(structure_in[app_name]["tag"]);
          cout << "Structure tag    : " << structure_in[app_name]["tag"].toStyledString() << endl;
        }
      }
    }

    bc_files.ignore_washouts          = opts.GetFirstBoolean('-', "ignore_washouts", false);
    bc_files.options_set = true;
    return true;
};

// ----------------------------------------------------------------------

bool BaseCallerParameters::InitContextVarsFromOptArgs(OptArgs& opts){

    assert(bc_files.options_set);
    char default_run_id[6]; // Create a run identifier from full output directory string
    ion_run_to_readname (default_run_id, (char*)bc_files.output_directory.c_str(), bc_files.output_directory.length());
    context_vars.run_id                      = opts.GetFirstString ('-', "run-id", default_run_id);
    num_threads_                             = opts.GetFirstInt    ('n', "num-threads", max(2*numCores(), 4));
    num_bamwriter_threads_                   = opts.GetFirstInt    ('-', "num-threads-bamwriter", 0);
    compress_output_bam_                     = opts.GetFirstBoolean('-', "compress-bam", true);

    context_vars.flow_signals_type           = opts.GetFirstString ('-', "flow-signals-type", "none");
    context_vars.only_process_unfiltered_set = opts.GetFirstBoolean('-', "only-process-unfiltered-set", false);
    context_vars.flow_predictors_            = opts.GetFirstBoolean('-', "flow-predictors", false);

    // Treephaser options
#if defined( __SSE3__ )
    context_vars.dephaser                    = opts.GetFirstString ('-', "dephaser", "treephaser-sse");
#else
    context_vars.dephaser                    = opts.GetFirstString ('-', "dephaser", "treephaser-swan");
#endif
    context_vars.keynormalizer               = opts.GetFirstString ('-', "keynormalizer", "gain");
    context_vars.windowSize                  = opts.GetFirstInt    ('-', "window-size", DPTreephaser::kWindowSizeDefault_);
    context_vars.skip_droop                  = opts.GetFirstBoolean('-', "skip-droop", true); // cpp basecaller only
    context_vars.skip_recal_during_norm      = opts.GetFirstBoolean('-', "skip-recal-during-normalization", false);
    context_vars.diagonal_state_prog         = opts.GetFirstBoolean('-', "diagonal-state-prog", false);
    context_vars.wells_norm_method           = opts.GetFirstString ('-', "wells-normalization", "off");
    context_vars.just_phase_estimation       = opts.GetFirstBoolean('-', "just-phase-estimation", false);
    context_vars.calibrate_TFs               = opts.GetFirstBoolean('-', "calibrate-tfs", false);
    context_vars.trim_zm                     = opts.GetFirstBoolean('-', "trim-zm", true);
    context_vars.compress_multi_taps         = opts.GetFirstBoolean('-', "compress-multi-taps", false);

    context_vars.inline_control              = opts.GetFirstBoolean('-', "inline-control", false);
    if (context_vars.inline_control and not bc_files.inline_control_reference_file.empty()){
      ValidateAndCanonicalizePath(bc_files.inline_control_reference_file);
    }

    // debug options
    context_vars.debug_normalization_bam     = opts.GetFirstBoolean ('-', "debug-normalization-bam", false);

    // Not every combination of options is possible here:
    if (context_vars.diagonal_state_prog and context_vars.dephaser != "treephaser-swan") {
      cout << " === BaseCaller Option Incompatibility: Using dephaser treephaser-swan with diagonal state progression instead of "
           << context_vars.dephaser << endl;
      context_vars.dephaser = "treephaser-swan";
    }

    context_vars.process_tfs      = true;
    context_vars.options_set      = true;
    return true;
};

// ----------------------------------------------------------------------

bool BaseCallerParameters::InitializeSamplingFromOptArgs(OptArgs& opts, const int num_lib_wells)
{
	  assert(context_vars.options_set);

    // If we are just doing phase estimation none of the options matter, so don't spam output
	  if (context_vars.just_phase_estimation){
	    sampling_opts.options_set = true;
	    return true;
	  }

    sampling_opts.num_unfiltered           = opts.GetFirstInt    ('-', "num-unfiltered", 100000);
    sampling_opts.downsample_size          = opts.GetFirstInt    ('-', "downsample-size", 0);
    sampling_opts.downsample_fraction      = opts.GetFirstDouble ('-', "downsample-fraction", 1.0);

    sampling_opts.calibration_training     = opts.GetFirstInt    ('-', "calibration-training", -1);
    sampling_opts.have_calib_panel         = (not bc_files.calibration_panel_file.empty());

    // Reconcile parameters downsample_size and downsample_fraction
    bool downsample = sampling_opts.downsample_size > 0 or sampling_opts.downsample_fraction < 1.0;
    if (sampling_opts.downsample_fraction < 1.0) {
      if (sampling_opts.downsample_size == 0)
        sampling_opts.downsample_size = (int)((float)num_lib_wells*sampling_opts.downsample_fraction);
      else {
        cerr << " === BaseCaller Option Incompatibility: Specify either downsample-size or downsample-fraction. Aborting!" << endl;
        exit(EXIT_FAILURE);
      }
    }
    if (downsample)
      cout << "Downsampling activated: Randomly choosing " << sampling_opts.downsample_size << " reads on this chip." << endl;

    // Calibration training requires additional changes & overwrites command line options
    if (sampling_opts.calibration_training >= 0) {
      if (context_vars.diagonal_state_prog) {
        cerr << " === BaseCaller Option Incompatibility: Calibration training not supported for diagonal state progression. Aborting!" << endl;
                exit(EXIT_FAILURE);
      }
      if (sampling_opts.downsample_size>0)
        sampling_opts.calibration_training = min(sampling_opts.calibration_training, sampling_opts.downsample_size);

      sampling_opts.downsample_size  = max(sampling_opts.calibration_training, 0);
      sampling_opts.num_unfiltered   = 0;
      context_vars.process_tfs       = false;
      cout << "=== BaseCaller Calibration Training ===" << endl;
      cout << " - Generating a training set up to " << sampling_opts.downsample_size << " randomly selected reads." << endl;
      if (sampling_opts.have_calib_panel)
        cout << " - Adding calibration panel reads specified in " << bc_files.calibration_panel_file << endl;
      cout << endl;
    }

	  sampling_opts.options_set = true;
    return true;
};

// ----------------------------------------------------------------------

bool BaseCallerParameters::SetBaseCallerContextVars(BaseCallerContext & bc)
{
    // General run parameters
	assert(sampling_opts.options_set);


    bc.filename_wells         = bc_files.filename_wells;
    bc.output_directory       = bc_files.output_directory;

    bc.run_id                 = context_vars.run_id;
    bc.flow_signals_type      = context_vars.flow_signals_type;
    bc.process_tfs            = context_vars.process_tfs;
    bc.have_calibration_panel = sampling_opts.have_calib_panel;
    bc.calibration_training   = (sampling_opts.calibration_training >= 0);
    bc.only_process_unfiltered_set = context_vars.only_process_unfiltered_set;

    bc.wells_norm_method      = context_vars.wells_norm_method;
    bc.keynormalizer          = context_vars.keynormalizer;
    bc.dephaser               = context_vars.dephaser;
    bc.sse_dephaser           = (bc.dephaser == "treephaser-sse" or bc.dephaser == "treephaser-solve");
    bc.windowSize             = context_vars.windowSize;
    bc.diagonal_state_prog    = context_vars.diagonal_state_prog;
    bc.skip_droop             = context_vars.skip_droop;
    bc.skip_recal_during_norm = context_vars.skip_recal_during_norm;
    bc.calibrate_TFs          = context_vars.calibrate_TFs;
    bc.trim_zm                = context_vars.trim_zm;
    bc.compress_multi_taps    = context_vars.compress_multi_taps;

    bc.flow_predictors_       = context_vars.flow_predictors_;
    bc.inline_control         = context_vars.inline_control;
    // debug options
    bc.debug_normalization_bam              = context_vars.debug_normalization_bam;
    return true;
};

// ----------------------------------------------------------------------

Json::Value BaseCallerParameters::NormalizeDictStructure(Json::Value structure)
{
  vector<string> keys = structure.getMemberNames();
  for (unsigned int k=0; k<keys.size(); ++k){
    string normStr(getNormString(structure[keys[k]].asString()));
    structure[keys[k]] = normStr;
  }
  return structure;
}

// ----------------------------------------------------------------------

bool BaseCallerParameters::SaveParamsToJson(Json::Value& basecaller_json, const BaseCallerContext& bc, const string& chip_type)
{
    basecaller_json["BaseCaller"]["run_id"] = bc.run_id;
    basecaller_json["BaseCaller"]["flow_order"] = bc.flow_order.str();
    basecaller_json["BaseCaller"]["num_flows"] = bc.flow_order.num_flows();
    basecaller_json["BaseCaller"]["lib_key"] =  bc.keys[0].bases();
    basecaller_json["BaseCaller"]["tf_key"] =  bc.keys[1].bases();
    basecaller_json["BaseCaller"]["chip_type"] = chip_type;
    basecaller_json["BaseCaller"]["output_dir"] = bc_files.output_directory;

    for (unsigned int i=0; i<bc_files.input_directory.size(); ++i)
      basecaller_json["BaseCaller"]["input_dir"][i] = bc_files.input_directory[i];
    for (unsigned int i=0; i<bc_files.input_directory.size(); ++i)
      basecaller_json["BaseCaller"]["filename_wells"][i] = bc_files.filename_wells[i];
    for (unsigned int i=0; i<bc_files.input_directory.size(); ++i)
      basecaller_json["BaseCaller"]["filename_mask"][i] = bc_files.filename_mask[i];

    basecaller_json["BaseCaller"]["num_threads"] = num_threads_;
    basecaller_json["BaseCaller"]["dephaser"] = bc.dephaser;
    basecaller_json["BaseCaller"]["keynormalizer"] = bc.keynormalizer;
    basecaller_json["BaseCaller"]["block_row_offset"] = bc.chip_subset.GetRowOffset();
    basecaller_json["BaseCaller"]["block_col_offset"] = bc.chip_subset.GetColOffset();
    basecaller_json["BaseCaller"]["block_row_size"] = bc.chip_subset.GetChipSizeY();
    basecaller_json["BaseCaller"]["block_col_size"] = bc.chip_subset.GetChipSizeX();
    basecaller_json["BaseCaller"]["inline_control"] = bc.inline_control;
    basecaller_json["BaseCaller"]["inline_control_reference"] = bc_files.inline_control_reference_file;
    SaveJson(basecaller_json, bc_files.filename_json);
    return true;
};
