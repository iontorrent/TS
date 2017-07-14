/* Copyright (C) 2016 Thermo Fisher Scientific, Inc. All Rights Reserved */

#include "ConsensusParameters.h"
#include "MolecularTagTrimmer.h"
#include "MiscUtil.h"
#include <fenv.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <errno.h>

using namespace std;


void ConsensusHelp() {
  printf("Usage: consensus [options]\n");
  printf("\n");

  printf("General options:\n");
  printf("  -h,--help                                         print this help message and exit\n");
  printf("  -v,--version                                      print version and exit\n");
  printf("  -n,--num-threads                      INT         number of worker threads [4]\n");
  printf("     --parameters-file                  FILE        json file with algorithm control parameters [optional]\n");
  printf("\n");

  printf("Inputs:\n");
  printf("  -r,--reference                        FILE        reference fasta file [required]\n");
  printf("  -b,--input-bam                        FILE        bam file with mapped reads [required]\n");
  printf("  -t,--target-file                      FILE        only process targets in this bed file [required]\n");
  printf("\n");

  printf("Outputs:\n");
  printf("  -O,--output-dir                       DIRECTORY   base directory for all output files [current dir]\n");
  printf("     --consensus-bam                    FILE        save processed consensus reads to the BAM file [required]\n");

  printf("\n");

  MolecularTagTrimmer::PrintHelp(true);
  printf("\n");

  printf("BaseCaller options:\n");
  printf("     --suppress-recalibration           on/off      suppress homopolymer recalibration [on].\n");
  printf("     --use-sse-basecaller               on/off      switch to use the vectorized version of the basecaller [on].\n");
  printf("\n");


  printf("Read filtering options:\n");
  printf("  -M,--min-mapping-qv                   INT         do not use reads with mapping quality below this [4]\n");
  printf("  -U,--read-snp-limit                   INT         do not use reads with number of snps above this [10]\n");
  printf("     --read-mismatch-limit              INT         do not use reads with number of mismatches (where 1 gap open counts 1) above this value (0 to disable this option) [0]\n");
  printf("     --min-cov-fraction                 FLOAT       do not use reads with fraction of covering the best assigned (unmerged) target region below this [0.0]\n");
  printf("\n");

  printf("Consensus read output options:\n");
  printf("     --need-3-end-adapter               on/off      do not output consensus reads w/o 3\" adapter found [off]\n");
  printf("     --filter-qt-reads                  on/off      do not output quality-trimmed consensus reads [off]\n");
  printf("     --filter-single-read-consensus     on/off      do not output single-read consensus [off]\n");
  printf("\n");

  printf("Debug:\n");
  printf("     --skip-consensus                   on/off      skip all calculations for consensus; output targets_depth.txt only [off]\n");
  printf("\n");

}

ConsensusParameters::ConsensusParameters(int argc, char** argv)
{
  // i/o parameters:
  fasta = "";                // -f --fasta-reference
  targets = "";              // -t --targets
  outputDir = "";
  consensus_bam = "";


  //OptArgs opts;
  opts.ParseCmdLine(argc, (const char**)argv);

  if (argc == 1) {
    ConsensusHelp();
    exit(0);
  }
  if (opts.GetFirstBoolean('v', "version", false)) {
    exit(0);
  }
  if (opts.GetFirstBoolean('h', "help", false)) {
    ConsensusHelp();
    exit(0);
  }

  // enable floating point exceptions during program execution
  if (opts.GetFirstBoolean('-', "float-exceptions", true)) {
    cout << "consensus: Floating point exceptions enabled." << endl;
    feraiseexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  }

  Json::Value tvc_params(Json::objectValue);
  Json::Value freebayes_params(Json::objectValue);
  Json::Value params_meta(Json::objectValue);
  ParametersFromJSON(opts, tvc_params, freebayes_params, params_meta);

  prefixExclusion =  opts.GetFirstInt('-', "prefix-exclude", 6);
  cerr << "prefix-exclude = " <<  prefixExclusion << endl;

  params_meta_name = params_meta.get("name",string()).asString();
  params_meta_details = params_meta.get("configuration_name", string()).asString();
  string external_file = params_meta.get("external_file", string()).asString();
  if (not external_file.empty()) {
    if (not params_meta_details.empty()) {params_meta_details += ", ";}
    params_meta_details += external_file;
  }
  string repository_id = params_meta.get("repository_id",string()).asString();
  string ts_version = params_meta.get("ts_version","").asString();
  if (not repository_id.empty()) {
    if (not params_meta_details.empty())
      params_meta_details += ", ";
    params_meta_details += repository_id;
  }
  if (not ts_version.empty()) {
    if (not params_meta_details.empty())
      params_meta_details += ", ";
    params_meta_details += "TS version: ";
    params_meta_details += ts_version;
  }

  // Retrieve the parameters that consensus needed
  program_flow.nThreads                 = RetrieveParameterInt   (opts, tvc_params, 'n', "num-threads", 4);
  program_flow.suppress_recalibration   = RetrieveParameterBool  (opts, tvc_params, '-', "suppress-recalibration", true);
#ifdef __SSE3__
  program_flow.use_SSE_basecaller       = RetrieveParameterBool  (opts, tvc_params, '-', "use-sse-basecaller", true);
#else
  program_flow.use_SSE_basecaller       = RetrieveParameterBool  (opts, tvc_params, '-', "use-sse-basecaller", false);
#endif
  min_mapping_qv                        = RetrieveParameterInt   (opts, freebayes_params, 'M', "min-mapping-qv", 4);
  min_cov_fraction                      = RetrieveParameterDouble(opts, freebayes_params, '-', "min-cov-fraction", 0.0f);
  read_mismatch_limit                   = RetrieveParameterInt   (opts, freebayes_params, '-', "read-mismatch-limit", 0);
  read_snp_limit                        = RetrieveParameterInt   (opts, freebayes_params, 'U', "read-snp-limit", 10);
  need_3_end_adapter                    = opts.GetFirstBoolean('-', "need-3-end-adapter", false);
  filter_qt_reads                       = opts.GetFirstBoolean('-', "filter-qt-reads", false);
  filter_single_read_consensus          = opts.GetFirstBoolean('-', "filter-single-read-consensus", false);
  skip_consensus                        = opts.GetFirstBoolean('-', "skip-consensus", false);

  SetMolecularTagTrimmerOpt(tvc_params);

  cout << endl;

  // Check limit
  CheckParameterLowerBound<int>       ("num-threads",         program_flow.nThreads,        1);
  CheckParameterLowerUpperBound<float>("min-cov-fraction",    min_cov_fraction,             0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("read-snp-limit",      read_snp_limit,               0);
  CheckParameterLowerBound<int>       ("min-mapping-qv",      min_mapping_qv,               0);
  CheckParameterLowerBound<int>       ("min-mapping-qv",      min_mapping_qv,               0);
  CheckParameterLowerBound<int>       ("read-mismatch-limit", read_mismatch_limit,          0);
  CheckParameterLowerUpperBound<int>  ("tag-trim-method",    tag_trimmer_parameters.tag_trim_method, 0, 2);
  CheckParameterLowerBound<int>       ("min-tag-fam-size",   tag_trimmer_parameters.min_family_size, 1);
  cout << endl;

  SetupFileIO(opts);
}

void ConsensusParameters::SetupFileIO(OptArgs &opts)
{
  // Reference fasta
  fasta                                 = opts.GetFirstString('r', "reference", "");
  if (fasta.empty()) {
    cerr << "FATAL ERROR: Reference file not specified via -r" << endl;
    exit(-1);
  }
  ValidateAndCanonicalizePath(fasta);

  // Region bed
  targets                               = opts.GetFirstString('t', "target-file", "");
  if (targets.empty()) {
	cerr << "FATAL ERROR: Target file not specified!" << endl;
	exit(-1);
  }
  ValidateAndCanonicalizePath(targets);

  // Input bam
  opts.GetOption(bams, "", 'b', "input-bam");
  if (bams.empty()) {
    cerr << "FATAL ERROR: BAM file not specified via -b" << endl;
    exit(-1);
  }
  for (unsigned int i_bam = 0; i_bam < bams.size(); ++i_bam)
    ValidateAndCanonicalizePath(bams[i_bam]);

  // Output dir
  outputDir                             = opts.GetFirstString('O', "output-dir", ".");
  mkpath(outputDir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); // XXX nope!
  ValidateAndCanonicalizePath(outputDir);

  // output consensus bam
  string dir = "";
  if (skip_consensus){
	  consensus_bam = "";
	  dir = outputDir + "/";
  }else{
	  consensus_bam                         = opts.GetFirstString('-', "consensus-bam", "");
	  consensus_bam = outputDir + "/" + consensus_bam;
	  dir = consensus_bam.substr(0, consensus_bam.find_last_of("/\\"));
  }
  mkpath(dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  ValidateAndCanonicalizePath(dir);

  // Options for multisample. Currently not supported.
  /*
  sampleName                            = opts.GetFirstString('g', "sample-name", "");
  force_sample_name                     = opts.GetFirstString('-', "force-sample-name", "");
  */
}
