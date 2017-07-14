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

bool BaseCallerContext::WriteUnfilteredFilterStatus(const BaseCallerFiles & bc_files) {

    ofstream filter_status;

    string filter_status_filename = bc_files.unfiltered_untrimmed_directory + string("/filterStatus.txt");
    filter_status.open(filter_status_filename.c_str());
    filter_status << "col" << "\t" << "row" << "\t" << "highRes" << "\t" << "valid" << endl;
    for (set<unsigned int>::iterator I = unfiltered_set.begin(); I != unfiltered_set.end(); ++I) {
        int x = (*I) % chip_subset.GetChipSizeX();
        int y = (*I) / chip_subset.GetChipSizeX();
        filter_status << x << "\t" << y;
        filter_status << "\t" << (int) mask->Match(x, y, MaskFilteredBadResidual); // Must happen after filters transferred to mask
        filter_status << "\t" << (int) mask->Match(x, y, MaskKeypass);
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
        filter_status << "\t" << (int) mask->Match(x, y, MaskFilteredBadResidual); // Must happen after filters transferred to mask
        filter_status << "\t" << (int) mask->Match(x, y, MaskKeypass);
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
    printf ("Debug Options:\n");
    printf ("     --debug-normalization-bam  BOOL       Output debug data to the bam tags Ya, Yb, Yw, and Yx, for the adaptive offset, adaptive slope, well-normalized measurements, and not-calibrated measurements [off]\n");
    printf ("\n");

    BaseCallerFilters::PrintHelp();
    PhaseEstimator::PrintHelp();
    PerBaseQual::PrintHelp();
    BarcodeClassifier::PrintHelp();
    MolecularTagTrimmer::PrintHelp(false);
    BaseCallerMetricSaver::PrintHelp();

    exit (EXIT_SUCCESS);
};


// ----------------------------------------------------------------------

bool BaseCallerParameters::InitializeFilesFromOptArgs(OptArgs& opts)
{
    bc_files.input_directory        = opts.GetFirstString ('i', "input-dir", ".");
    bc_files.output_directory       = opts.GetFirstString ('o', "output-dir", ".");
    bc_files.unfiltered_untrimmed_directory = bc_files.output_directory + "/unfiltered.untrimmed";
    bc_files.unfiltered_trimmed_directory   = bc_files.output_directory + "/unfiltered.trimmed";

    CreateResultsFolder ((char*)bc_files.output_directory.c_str());
    CreateResultsFolder ((char*)bc_files.unfiltered_untrimmed_directory.c_str());
    CreateResultsFolder ((char*)bc_files.unfiltered_trimmed_directory.c_str());

    ValidateAndCanonicalizePath(bc_files.input_directory);
    ValidateAndCanonicalizePath(bc_files.output_directory);
    ValidateAndCanonicalizePath(bc_files.unfiltered_untrimmed_directory);
    ValidateAndCanonicalizePath(bc_files.unfiltered_trimmed_directory);

    bc_files.filename_wells         = opts.GetFirstString ('-', "wells", bc_files.input_directory + "/1.wells");
    bc_files.filename_mask          = opts.GetFirstString ('-', "mask", bc_files.input_directory + "/analysis.bfmask.bin");

    ValidateAndCanonicalizePath(bc_files.filename_wells);
    ValidateAndCanonicalizePath(bc_files.filename_mask, bc_files.input_directory + "/bfmask.bin");

    bc_files.filename_filter_mask   = bc_files.output_directory + "/bfmask.bin";
    bc_files.filename_json          = bc_files.output_directory + "/BaseCaller.json";
    bc_files.filename_phase         = bc_files.output_directory + "/PhaseEstimates.json";

    printf("\n");
    printf("Input files summary:\n");
    printf("     --input-dir %s\n", bc_files.input_directory.c_str());
    printf("         --wells %s\n", bc_files.filename_wells.c_str());
    printf("          --mask %s\n", bc_files.filename_mask.c_str());
    printf("\n");
    printf("Output directories summary:\n");
    printf("    --output-dir %s\n", bc_files.output_directory.c_str());
    printf("        unf.untr %s\n", bc_files.unfiltered_untrimmed_directory.c_str());
    printf("          unf.tr %s\n", bc_files.unfiltered_trimmed_directory.c_str());
    printf("\n");

    bc_files.lib_datasets_file      = opts.GetFirstString ('-', "datasets", "");
    bc_files.calibration_panel_file = opts.GetFirstString ('-', "calibration-panel", "");
    if (not bc_files.lib_datasets_file.empty())
      ValidateAndCanonicalizePath(bc_files.lib_datasets_file);
    if (not bc_files.calibration_panel_file.empty())
      ValidateAndCanonicalizePath(bc_files.calibration_panel_file);

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

bool BaseCallerParameters::InitializeSamplingFromOptArgs(OptArgs& opts, const int num_wells)
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
    sampling_opts.MaskNotWanted            = MaskNone;

    // Reconcile parameters downsample_size and downsample_fraction
    bool downsample = sampling_opts.downsample_size > 0 or sampling_opts.downsample_fraction < 1.0;
    if (sampling_opts.downsample_fraction < 1.0) {
      if (sampling_opts.downsample_size == 0)
    	sampling_opts.downsample_size = (int)((float)num_wells*sampling_opts.downsample_fraction);
      else
        sampling_opts.downsample_size = min(sampling_opts.downsample_size, (int)((float)num_wells*sampling_opts.downsample_fraction));
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
      sampling_opts.MaskNotWanted    = (MaskType)(MaskFilteredBadResidual|MaskFilteredBadPPF|MaskFilteredBadKey);
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

    bc.flow_predictors_       = context_vars.flow_predictors_;
    // debug options
    bc.debug_normalization_bam              = context_vars.debug_normalization_bam;
    return true;
};

// ----------------------------------------------------------------------

bool BaseCallerParameters::SaveParamsToJson(Json::Value& basecaller_json, const BaseCallerContext& bc, const string& chip_type)
{
    basecaller_json["BaseCaller"]["run_id"] = bc.run_id;
    basecaller_json["BaseCaller"]["flow_order"] = bc.flow_order.str();
    basecaller_json["BaseCaller"]["num_flows"] = bc.flow_order.num_flows();
    basecaller_json["BaseCaller"]["lib_key"] =  bc.keys[0].bases();
    basecaller_json["BaseCaller"]["tf_key"] =  bc.keys[1].bases();
    basecaller_json["BaseCaller"]["chip_type"] = chip_type;
    basecaller_json["BaseCaller"]["input_dir"] = bc_files.input_directory;
    basecaller_json["BaseCaller"]["output_dir"] = bc_files.output_directory;
    basecaller_json["BaseCaller"]["filename_wells"] = bc_files.filename_wells;
    basecaller_json["BaseCaller"]["filename_mask"] = bc_files.filename_mask;
    basecaller_json["BaseCaller"]["num_threads"] = num_threads_;
    basecaller_json["BaseCaller"]["dephaser"] = bc.dephaser;
    basecaller_json["BaseCaller"]["keynormalizer"] = bc.keynormalizer;
    basecaller_json["BaseCaller"]["block_row_offset"] = bc.chip_subset.GetRowOffset();
    basecaller_json["BaseCaller"]["block_col_offset"] = bc.chip_subset.GetColOffset();
    basecaller_json["BaseCaller"]["block_row_size"] = bc.chip_subset.GetChipSizeY();
    basecaller_json["BaseCaller"]["block_col_size"] = bc.chip_subset.GetChipSizeX();
    SaveJson(basecaller_json, bc_files.filename_json);
    return true;
};
