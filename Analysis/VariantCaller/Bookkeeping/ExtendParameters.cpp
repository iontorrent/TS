/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ExtendParameters.h"
#include <iomanip>

using namespace std;


void VariantCallerHelp() {
  printf("Usage: tvc [options]\n");
  printf("\n");

  printf("General options:\n");
  printf("  -h,--help                                         print this help message and exit\n");
  printf("  -v,--version                                      print version and exit\n");
  printf("  -n,--num-threads                      INT         number of worker threads [2]\n");
  printf("  -N,--num-variants-per-thread          INT         worker thread batch size [500]\n");
  printf("     --parameters-file                  FILE        json file with algorithm control parameters [optional]\n");
  printf("\n");

  printf("Inputs:\n");
  printf("  -r,--reference                        FILE        reference fasta file [required]\n");
  printf("  -b,--input-bam                        FILE        bam file with mapped reads [required]\n");
  printf("  -g,--sample-name                      STRING      sample for which variants are called (In case of input BAM files with multiple samples) [optional if there is only one sample]\n");
  printf("  -t,--target-file                      FILE        only process targets in this bed file [optional]\n");
  printf("  -R --region                           STRING      only process <chrom>:<start_position>-<end_position> [optional] \n") ;
  printf("  -D,--downsample-to-coverage           INT         ?? [2000]\n");
  printf("     --model-file                       FILE        HP recalibration model input file.\n");
  printf("     --recal-model-hp-thres             INT         Lower threshold for HP recalibration.\n");
  printf("\n");

  printf("Outputs:\n");
  printf("  -O,--output-dir                       DIRECTORY   base directory for all output files [current dir]\n");
  printf("  -o,--output-vcf                       FILE        vcf file with variant calling results [required]\n");
  printf("     --suppress-reference-genotypes     on/off      write reference calls into the filtered variants vcf [on]\n");
  printf("     --suppress-no-calls                on/off      write filtered variants into the filtered variants vcf [on]\n");
  printf("     --suppress-nocall-genotypes        on/off      do not report a genotype for filtered variants [on]\n");
  printf("\n");

  printf("Variant candidate generation (FreeBayes):\n");
  printf("     --allow-snps                       on/off      allow generation of snp candidates [on]\n");
  printf("     --allow-indels                     on/off      allow generation of indel candidates [on]\n");
  printf("     --allow-mnps                       on/off      allow generation of mnp candidates [on]\n");
  printf("     --allow-complex                    on/off      allow generation of block substitution candidates [off]\n");
  printf("     --max-complex-gap                  INT         maximum number of reference alleles between two alternate alleles to allow merging of the alternate alleles [1]\n");
  printf("     --use-reference-allele             on/off      generate bogus alt=ref candidates for all non-variant locations [off]\n");
  //printf("     --left-align-indels                on/off      Experimental: Left align Indels in candidate generation [off]\n");
  printf("  -m,--use-best-n-alleles               INT         maximum number of snp alleles [2]\n");
  printf("  -M,--min-mapping-qv                   INT         do not use reads with mapping quality below this [4]\n");
  //printf("  -q,--min-base-qv                      INT         do not use reads with base quality below this [4]\n");
  printf("  -U,--read-snp-limit                   INT         do not use reads with number of snps above this [10]\n");
  printf("  -z,--read-max-mismatch-fraction       FLOAT       do not use reads with fraction of mismatches above this [1.0]\n");
  printf("     --gen-min-alt-allele-freq          FLOAT       minimum required alt allele frequency to generate a candidate [0.2]\n");
  printf("     --gen-min-indel-alt-allele-freq    FLOAT       minimum required alt allele frequency to generate a homopolymer indel candidate [0.2]\n");
  printf("     --gen-min-coverage                 INT         minimum required coverage to generate a candidate [6]\n");
  printf("\n");

  printf("External variant candidates:\n");
  printf("  -c,--input-vcf                        FILE        vcf.gz file (+.tbi) with additional candidate variant locations and alleles [optional]\n");
  printf("     --process-input-positions-only     on/off      only generate candidates at locations from input-vcf [off]\n");
  printf("     --use-input-allele-only            on/off      only consider provided alleles for locations in input-vcf [off]\n");
  printf("\n");

  printf("Variant candidate scoring options:\n");
  printf("     --min-delta-for-flow               FLOAT       minimum prediction delta for scoring flows [0.1]\n");
  printf("     --max-flows-to-test                INT         maximum number of scoring flows [10]\n");
  printf("     --outlier-probability              FLOAT       probability for outlier reads [0.01]\n");
  printf("     --heavy-tailed                     INT         degrees of freedom in t-dist modeling signal residual heavy tail [3]\n");
  printf("     --suppress-recalibration           on/off      Suppress homopolymer recalibration [on].\n");
  printf("     --do-snp-realignment               on/off      Realign reads in the vicinity of candidate snp variants [on].\n");
  printf("\n");

  printf("Advanced variant candidate scoring options:\n");
  printf("     --use-sse-basecaller               on/off      Switch to use the vectorized version of the basecaller [on].\n");
  printf("     --resolve-clipped-bases            on/off      If 'true', the basecaller is used to solve soft clipped bases [off].\n");
  printf("     --prediction-precision             FLOAT       prior weight in bias estimator [30.0]\n");
  printf("     --shift-likelihood-penalty         FLOAT       penalize log-likelihood for solutions involving large systematic bias [0.3]\n");
  printf("     --minimum-sigma-prior              FLOAT       prior variance per data point, constant [0.085]\n");
  printf("     --slope-sigma-prior                FLOAT       prior rate of increase of variance over minimum by signal [0.0084]\n");
  printf("     --sigma-prior-weight               FLOAT       weight of prior estimate of variance compared to observations [1.0]\n");
  printf("     --k-zero                           FLOAT       variance increase for adding systematic bias [3.0]\n");
  printf("     --sse-relative-safety-level        FLOAT       dampen strand bias detection for SSE events for low coverage [0.025]\n");
  printf("     --tune-sbias                       FLOAT       dampen strand bias detection for low coverage [0.01]\n");
  printf("\n");

  printf("Variant filtering:\n");
  // Filters depending on the variant type
  printf("  -k,--snp-min-coverage                 INT         filter out snps with total coverage below this [6]\n");
  printf("  -C,--snp-min-cov-each-strand          INT         filter out snps with coverage on either strand below this [1]\n");
  printf("  -B,--snp-min-variant-score            FLOAT       filter out snps with QUAL score below this [2.5]\n");
  printf("  -s,--snp-strand-bias                  FLOAT       filter out snps with strand bias above this [0.95]\n");
  printf("  -A,--snp-min-allele-freq              FLOAT       minimum required alt allele frequency for non-reference snp calls [0.2]\n");
  printf("     --indel-min-coverage               INT         filter out indels with total coverage below this [30]\n");
  printf("     --indel-min-cov-each-strand        INT         filter out indels with coverage on either strand below this [1]\n");
  printf("     --indel-min-variant-score          FLOAT       filter out indels with QUAL score below this [2.5]\n");
  printf("  -S,--indel-strand-bias                FLOAT       filter out indels with strand bias above this [0.85]\n");
  printf("     --indel-min-allele-freq            FLOAT       minimum required alt allele frequency for non-reference indel call [0.2]\n");
  printf("     --hotspot-min-coverage             INT         filter out hotspot variants with total coverage below this [6]\n");
  printf("     --hotspot-min-cov-each-strand      INT         filter out hotspot variants with coverage on either strand below this [1]\n");
  printf("     --hotspot-min-variant-score        FLOAT       filter out hotspot variants with QUAL score below this [2.5]\n");
  printf("     --hotspot-strand-bias              FLOAT       filter out hotspot variants with strand bias above this [0.95]\n");
  printf("  -H,--hotspot-min-allele-freq          FLOAT       minimum required alt allele frequency for non-reference hotspot variant call [0.2]\n");
  // Filters not depending on the variant score
  printf("  -L,--hp-max-length                    INT         filter out indels in homopolymers above this [8]\n");
  printf("  -e,--error-motifs                     FILE        table of systematic error motifs and their error rates [optional]\n");
  printf("     --sse-prob-threshold               FLOAT       filter out variants in motifs with error rates above this [0.2]\n");
  printf("     --min-ratio-reads-non-sse-strand   FLOAT       minimum required alt allele frequency for variants with error motifs on opposite strand [0.2]\n");
  // These filters depend on scoring
  printf("     --data-quality-stringency          FLOAT       minimum mean log-likelihood delta per read [4.0]\n");
  printf("     --read-rejection-threshold         FLOAT       filter variants where large numbers of reads are rejected as outliers [0.5]\n");
  printf("     --filter-unusual-predictions       FLOAT       posterior log likelihood threshold for accepting bias estimate [0.3]\n");
  printf("     --filter-deletion-predictions      FLOAT       check post-evaluation systematic bias in deletions [100.0]\n");
  printf("     --filter-insertion-predictions     FLOAT       check post-evaluation systematic bias in insertions [100.0]\n");
  printf("     --heal-snps                        on/off      suppress in/dels not participating in diploid variant genotypes if the genotype contains a SNP [on].\n");
  printf("\n");

  printf("Debugging:\n");
  printf("  -d,--debug                            INT         (0/1/2) display extra debug messages [0]\n");
  printf("  -3,--diagnostic-input-vcf             FILE        (devel) list of candidate variant locations and alleles [optional]\n");
  printf("     --skip-candidate-generation        on/off      (devel) bypass candidate generation and use diagnostic-input-vcf [off]\n");
  printf("     --do-json-diagnostic               on/off      (devel) dump internal state to json file (uses much more time/memory/disk) [off]\n");
  printf("     --do-minimal-diagnostic            on/off      (devel) provide minimal read information for called variants [off]\n");
  printf("     --override-limits                  on/off      (devel) disable limit-check on input parameters [off].\n");
  //printf("  -G,--reference-sample                 STRING      ?? [optional]\n");
  printf("\n");
}







ControlCallAndFilters::ControlCallAndFilters() {
  // all defaults handled by sub-filters
  data_quality_stringency = 4.0f;  // phred-score for this variant per read
  read_rejection_threshold = 0.5f; // half the reads gone, filter this
  //xbias_tune = 0.005f;
  sbias_tune = 0.5f;
  downSampleCoverage = 2000;
  RandSeed = 631;
  // wanted by downstream
  suppress_reference_genotypes = true;
  suppress_nocall_genotypes = true;
  suppress_no_calls = true;
  heal_snps = true;
}

ProgramControlSettings::ProgramControlSettings() {
  nVariantsPerThread = 1000;
  nThreads = 1;
  DEBUG = 0;

  use_SSE_basecaller = true;
  rich_json_diagnostic = false;
  minimal_diagnostic = false;
  json_plot_dir = "./json_diagnostic/";
  inputPositionsOnly = false;
  skipCandidateGeneration = false;
  suppress_recalibration = true;
  do_snp_realignment = true;
  resolve_clipped_bases = false;
}

ExtendParameters::ExtendParameters(void) : Parameters() {

  vcfProvided = false;
  sseMotifsProvided = false;
  //Input files
  inputBAM = "";
  outputDir = ".";

  candidateVCFFileName = "";
  recalModelHPThres = 4;

}

int GetParamsInt(Json::Value& json, const string& key, int default_value) {
  if (not json.isMember(key))
    return default_value;
  if (json[key].isString())
    return atoi(json[key].asCString());
  return json[key].asInt();
}

double GetParamsDbl(Json::Value& json, const string& key, double default_value) {
  if (not json.isMember(key))
    return default_value;
  if (json[key].isString())
    return atof(json[key].asCString());
  return json[key].asDouble();
}


int RetrieveParameterInt(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, int default_value)
{
  string long_name_underscores = long_name_hyphens;
  for (unsigned int i = 0; i < long_name_underscores.size(); ++i)
    if (long_name_underscores[i] == '-')
      long_name_underscores[i] = '_';

  int value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    if (json[long_name_underscores].isString())
      value = atoi(json[long_name_underscores].asCString());
    else
      value = json[long_name_underscores].asInt();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstInt(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (integer, " << source << ")" << endl;
  return value;
}

double RetrieveParameterDouble(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, double default_value)
{
  string long_name_underscores = long_name_hyphens;
  for (unsigned int i = 0; i < long_name_underscores.size(); ++i)
    if (long_name_underscores[i] == '-')
      long_name_underscores[i] = '_';

  double value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    if (json[long_name_underscores].isString())
      value = atof(json[long_name_underscores].asCString());
    else
      value = json[long_name_underscores].asDouble();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstDouble(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (double,  " << source << ")" << endl;
  return value;
}


bool RetrieveParameterBool(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, bool default_value)
{
  string long_name_underscores = long_name_hyphens;
  for (unsigned int i = 0; i < long_name_underscores.size(); ++i)
    if (long_name_underscores[i] == '-')
      long_name_underscores[i] = '_';

  bool value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    if (json[long_name_underscores].isString())
      value = atoi(json[long_name_underscores].asCString());
    else
      value = json[long_name_underscores].asInt();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstBoolean(short_name, long_name_hyphens, value);
    source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << (value ? "true" : "false") << " (boolean, " << source << ")" << endl;
  return value;
}

// =============================================================================

void EnsembleEvalTuningParameters::SetOpts(OptArgs &opts, Json::Value& tvc_params) {

  max_flows_to_test                     = RetrieveParameterInt   (opts, tvc_params, '-', "max-flows-to-test", 10);
  min_delta_for_flow                    = RetrieveParameterDouble(opts, tvc_params, '-', "min-delta-for-flow", 0.1);

  prediction_precision                  = RetrieveParameterDouble(opts, tvc_params, '-', "prediction-precision", 30.0);
  outlier_prob                          = RetrieveParameterDouble(opts, tvc_params, '-', "outlier-probability", 0.01);
  germline_prior_strength               = RetrieveParameterDouble(opts, tvc_params, '-', "germline-prior-strength", 0.0f);
  heavy_tailed                          = RetrieveParameterInt   (opts, tvc_params, '-', "heavy-tailed", 3);
  
  filter_unusual_predictions            = RetrieveParameterDouble(opts, tvc_params, '-', "filter-unusual-predictions", 0.3f);
  soft_clip_bias_checker                = RetrieveParameterDouble(opts, tvc_params, '-', "soft-clip-bias-checker", 0.1f);
  filter_deletion_bias                  = RetrieveParameterDouble(opts, tvc_params, '-', "filter-deletion-predictions", 100.0f);
  filter_insertion_bias                 = RetrieveParameterDouble(opts, tvc_params, '-', "filter-insertion-predictions", 100.0f);

  // shouldn't majorly affect anything, but still expose parameters for completeness
  pseudo_sigma_base                     = RetrieveParameterDouble(opts, tvc_params, '-', "shift-likelihood-penalty", 0.3f);
  magic_sigma_base                      = RetrieveParameterDouble(opts, tvc_params, '-', "minimum-sigma-prior", 0.085f);
  magic_sigma_slope                     = RetrieveParameterDouble(opts, tvc_params, '-', "slope-sigma-prior", 0.0084f);
  sigma_prior_weight                    = RetrieveParameterDouble(opts, tvc_params, '-', "sigma-prior-weight", 1.0f);
  k_zero                                = RetrieveParameterDouble(opts, tvc_params, '-', "k-zero", 3.0f); // add variance from cluster shifts

}

void EnsembleEvalTuningParameters::CheckParameterLimits() {

  CheckParameterLowerUpperBound<int>  ("max-flows-to-test",       max_flows_to_test,       1,     100);
  CheckParameterLowerUpperBound<float>("min-delta-for-flow",      min_delta_for_flow,      0.01f, 0.5f);
  CheckParameterLowerBound<float>     ("prediction-precision",    prediction_precision,    0.1f);
  CheckParameterLowerUpperBound<float>("outlier-probability",     outlier_prob,            0.0f,  1.0f);
  CheckParameterLowerUpperBound<float>("germline-prior-strength", germline_prior_strength, 0.0f,  1000.0f);
  CheckParameterLowerBound<int>       ("heavy-tailed",            heavy_tailed,            1);

  CheckParameterLowerBound<float>     ("filter-unusual-predictions",    filter_unusual_predictions,   0.0f);
  CheckParameterLowerUpperBound<float>("soft-clip-bias-checker",        soft_clip_bias_checker, 0.0f, 1.0f);
  CheckParameterLowerBound<float>     ("filter-deletion-predictions",   filter_deletion_bias,         0.0f);
  CheckParameterLowerBound<float>     ("filter-insertion-predictions",  filter_insertion_bias,        0.0f);

  CheckParameterLowerBound<float>     ("shift-likelihood-penalty",  pseudo_sigma_base,    0.01f);
  CheckParameterLowerBound<float>     ("minimum-sigma-prior",       magic_sigma_base,     0.01f);
  CheckParameterLowerBound<float>     ("slope-sigma-prior",         magic_sigma_slope,    0.0f);
  CheckParameterLowerBound<float>     ("sigma-prior-weight",        sigma_prior_weight,   0.01f);
  CheckParameterLowerBound<float>     ("k-zero",                    k_zero,               0.0f);
}

// ============================================================================

void ClassifyFilters::SetOpts(OptArgs &opts, Json::Value & tvc_params) {

  hp_max_length                         = RetrieveParameterInt   (opts, tvc_params, 'L', "hp-max-length", 8);
  sseProbThreshold                      = RetrieveParameterDouble(opts, tvc_params, '-', "sse-prob-threshold", 0.2);
  minRatioReadsOnNonErrorStrand         = RetrieveParameterDouble(opts, tvc_params, '-', "min-ratio-reads-non-sse-strand", 0.2);
  sse_relative_safety_level             = RetrieveParameterDouble(opts, tvc_params, '-', "sse-relative-safety-level", 0.025);
  // min ratio of reads supporting variant on non-sse strand for variant to be called

}

void ClassifyFilters::CheckParameterLimits() {

  CheckParameterLowerBound<int>       ("hp-max-length",        hp_max_length,        1);
  CheckParameterLowerUpperBound<float>("sse-prob-threshold",   sseProbThreshold, 0.0f, 1.0f);
  CheckParameterLowerUpperBound<float>("min-ratio-reads-non-sse-strand",   minRatioReadsOnNonErrorStrand, 0.0f, 1.0f);
  CheckParameterLowerUpperBound<float>("sse-relative-safety-level",   sse_relative_safety_level, 0.0f, 1.0f);

}

// ===========================================================================

void ControlCallAndFilters::CheckParameterLimits() {

  filter_variant.CheckParameterLimits();
  CheckParameterLowerBound<float>     ("data-quality-stringency",  data_quality_stringency,  0.0f);
  CheckParameterLowerUpperBound<float>("read-rejection-threshold", read_rejection_threshold, 0.0f, 1.0f);
  CheckParameterLowerUpperBound<int>  ("downsample-to-coverage",   downSampleCoverage,       20, 10000);

 // CheckParameterLowerUpperBound<float>("tune-xbias",      xbias_tune,     0.001f, 1000.0f);
  CheckParameterLowerUpperBound<float>("tune-sbias",      sbias_tune,     0.001f, 1000.0f);

  CheckParameterLowerBound<int>       ("snp-min-cov-each-strand",  filter_snps.min_cov_each_strand, 0);
  CheckParameterLowerBound<float>     ("snp-min-variant-score",    filter_snps.min_quality_score,   0.0f);
  CheckParameterLowerUpperBound<float>("snp-min-allele-freq",      filter_snps.min_allele_freq,     0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("snp-min-coverage",         filter_snps.min_cov,             0);
  CheckParameterLowerUpperBound<float>("snp-strand-bias",          filter_snps.strand_bias_threshold,  0.5f, 1.0f);
//  CheckParameterLowerBound<float>     ("snp-beta-bias",            filter_snps.beta_bias_filter,    0.0f);

  CheckParameterLowerBound<int>       ("indel-min-cov-each-strand",  filter_hp_indel.min_cov_each_strand, 0);
  CheckParameterLowerBound<float>     ("indel-min-variant-score",    filter_hp_indel.min_quality_score,   0.0f);
  CheckParameterLowerUpperBound<float>("indel-min-allele-freq",      filter_hp_indel.min_allele_freq,     0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("indel-min-coverage",         filter_hp_indel.min_cov,             0);
  CheckParameterLowerUpperBound<float>("indel-strand-bias",          filter_hp_indel.strand_bias_threshold,  0.5f, 1.0f);
//  CheckParameterLowerBound<float>     ("indel-beta-bias",            filter_hp_indel.beta_bias_filter,    0.0f);

  CheckParameterLowerBound<int>       ("hotspot-min-cov-each-strand",  filter_hotspot.min_cov_each_strand, 0);
  CheckParameterLowerBound<float>     ("hotspot-min-variant-score",    filter_hotspot.min_quality_score,   0.0f);
  CheckParameterLowerUpperBound<float>("hotspot-min-allele-freq",      filter_hotspot.min_allele_freq,     0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("hotspot-min-coverage",         filter_hotspot.min_cov,             0);
  CheckParameterLowerUpperBound<float>("hotspot-strand-bias",          filter_hotspot.strand_bias_threshold,  0.5f, 1.0f);
//  CheckParameterLowerBound<float>     ("hotspot-beta-bias",            filter_hotspot.beta_bias_filter,    0.0f);
}

// ------------------------------------------------------

void ControlCallAndFilters::SetOpts(OptArgs &opts, Json::Value& tvc_params) {

  filter_variant.SetOpts(opts, tvc_params);
  RandSeed = 631;    // Not exposed to user at this point

  // catchall filter parameter to be used to filter any generic predictive model of quality
  data_quality_stringency               = RetrieveParameterDouble(opts, tvc_params, '-', "data-quality-stringency",4.0f);
  // if we reject half the reads from evaluator, something badly wrong with this position
  read_rejection_threshold              = RetrieveParameterDouble(opts, tvc_params, '-', "read-rejection-threshold",0.5f);
  downSampleCoverage                    = RetrieveParameterInt   (opts, tvc_params, '-', "downsample-to-coverage", 2000);
  
  //xbias_tune                            = RetrieveParameterDouble(opts, tvc_params, '-', "tune-xbias", 0.005f);
  sbias_tune                            = RetrieveParameterDouble(opts, tvc_params, '-', "tune-sbias", 0.01f);

  suppress_reference_genotypes          = RetrieveParameterBool   (opts, tvc_params, '-', "suppress-reference-genotypes", true);
  suppress_nocall_genotypes             = RetrieveParameterBool   (opts, tvc_params, '-', "suppress-nocall-genotypes", true);
  suppress_no_calls                     = RetrieveParameterBool   (opts, tvc_params, '-', "suppress-no-calls", true);

  heal_snps                             = RetrieveParameterBool   (opts, tvc_params, '-', "heal-snps", true);

  // SNPS are my usual variants
  filter_snps.min_cov_each_strand       = RetrieveParameterInt   (opts, tvc_params, 'C', "snp-min-cov-each-strand", 0);
  filter_snps.min_quality_score         = RetrieveParameterDouble(opts, tvc_params, 'B', "snp-min-variant-score", 10.0);
  filter_snps.min_allele_freq           = RetrieveParameterDouble(opts, tvc_params, 'A', "snp-min-allele-freq", 0.2);
  filter_snps.min_cov                   = RetrieveParameterInt   (opts, tvc_params, 'k', "snp-min-coverage", 6);
  filter_snps.strand_bias_threshold     = RetrieveParameterDouble(opts, tvc_params, 's', "snp-strand-bias", 0.95);
//  filter_snps.beta_bias_filter          = RetrieveParameterDouble(opts, tvc_params, '-', "snp-beta-bias", 8.0f);

  // hp_indels are more complex
  filter_hp_indel.min_cov_each_strand   = RetrieveParameterInt   (opts, tvc_params, '-', "indel-min-cov-each-strand", 1);
  filter_hp_indel.min_quality_score     = RetrieveParameterDouble(opts, tvc_params, '-', "indel-min-variant-score", 10.0);
  filter_hp_indel.min_allele_freq       = RetrieveParameterDouble(opts, tvc_params, '-', "indel-min-allele-freq", 0.2);
  filter_hp_indel.min_cov               = RetrieveParameterInt   (opts, tvc_params, '-', "indel-min-coverage", 15);
  filter_hp_indel.strand_bias_threshold = RetrieveParameterDouble(opts, tvc_params, 'S', "indel-strand-bias", 0.85);
//  filter_hp_indel.beta_bias_filter      = RetrieveParameterDouble(opts, tvc_params, '-', "indel-beta-bias", 8.0f);


  // derive hotspots by default from SNPs
  // override from command line or json
  filter_hotspot.min_cov_each_strand    = RetrieveParameterInt   (opts, tvc_params, '-', "hotspot-min-cov-each-strand", filter_snps.min_cov_each_strand);
  filter_hotspot.min_quality_score      = RetrieveParameterDouble(opts, tvc_params, '-', "hotspot-min-variant-score", filter_snps.min_quality_score);
  filter_hotspot.min_allele_freq        = RetrieveParameterDouble(opts, tvc_params, 'H', "hotspot-min-allele-freq", filter_snps.min_allele_freq);
  filter_hotspot.min_cov                = RetrieveParameterInt   (opts, tvc_params, '-', "hotspot-min-coverage", filter_snps.min_cov);
  filter_hotspot.strand_bias_threshold  = RetrieveParameterDouble(opts, tvc_params, '-', "hotspot-strand-bias", filter_snps.strand_bias_threshold);
//  filter_hotspot.beta_bias_filter       = RetrieveParameterDouble(opts, tvc_params, '-', "hotspot-beta-bias", filter_snps.beta_bias_filter);

}

// =============================================================================

void ProgramControlSettings::CheckParameterLimits() {

  CheckParameterLowerUpperBound<int>  ("num-threads",              nThreads,             1, 128);
  CheckParameterLowerUpperBound<int>  ("num-variants-per-thread",  nVariantsPerThread,   1, 10000);
}

void ProgramControlSettings::SetOpts(OptArgs &opts, Json::Value &tvc_params) {

  DEBUG                                 = opts.GetFirstInt   ('d', "debug", 0);
  nThreads                              = RetrieveParameterInt   (opts, tvc_params, 'n', "num-threads", 12);
  nVariantsPerThread                    = RetrieveParameterInt   (opts, tvc_params, 'N', "num-variants-per-thread", 250);
  use_SSE_basecaller                    = RetrieveParameterBool  (opts, tvc_params, '-', "use-sse-basecaller", true);
  // decide diagnostic
  rich_json_diagnostic                  = RetrieveParameterBool  (opts, tvc_params, '-', "do-json-diagnostic", false);
  minimal_diagnostic                    = RetrieveParameterBool  (opts, tvc_params, '-', "do-minimal-diagnostic", false);


  inputPositionsOnly                    = RetrieveParameterBool  (opts, tvc_params, '-', "process-input-positions-only", false);
  skipCandidateGeneration               = RetrieveParameterBool  (opts, tvc_params, '-', "skip-candidate-generation", false);
  suppress_recalibration                = RetrieveParameterBool  (opts, tvc_params, '-', "suppress-recalibration", true);
  do_snp_realignment                    = RetrieveParameterBool  (opts, tvc_params, '-', "do-snp-realignment", true);
  resolve_clipped_bases                 = RetrieveParameterBool  (opts, tvc_params, '-', "resolve-clipped-bases", false);
}

// ===========================================================================

void ExtendParameters::SetupFileIO(OptArgs &opts) {
  // freeBayes slot
  fasta                                 = opts.GetFirstString('r', "reference", "");
  if (fasta.empty()) {
    cerr << "Fatal ERROR: Reference file not specified via -r" << endl;
    exit(1);
  }

  // freeBayes slot
  variantPriorsFile                     = opts.GetFirstString('c', "input-vcf", "");
  vcfProvided = true;
  if (variantPriorsFile.empty()) {
    vcfProvided = false;
    cerr << "INFO: No input VCF (Hotspot) file specified via -c,--input-vcf" << endl;
    //exit(1);
  }

  //THIS IS FOR R&D USE ONLY
  candidateVCFFileName                  = opts.GetFirstString('3', "diagnostic-input-vcf", "");
  if (candidateVCFFileName.empty())
    program_flow.skipCandidateGeneration = false;

  sseMotifsFileName                     = opts.GetFirstString('e', "error-motifs", "");
  sseMotifsProvided = true;
  if (sseMotifsFileName.empty()) {
    sseMotifsProvided = false;
    cerr << "INFO: Systematic error motif file not specified via -e" << endl;
    //exit(1);
  }

  opts.GetOption(bams, "", 'b', "input-bam");
  if (bams.empty()) {
    cerr << "FATAL ERROR: BAM file not specified via -b" << endl;
    exit(-1);
  }
  outputDir                             = opts.GetFirstString('O', "output-dir", ".");
  // freeBayes slot
  outputFile                            = opts.GetFirstString('o', "output-vcf", "");

  if (outputFile.empty()) {
    cerr << "Fatal ERROR: Output VCF filename not specified via -o" << endl;
    exit(1);
  }

  opts.GetOption(regions, "", 'R', "region");
  if (!candidateVCFFileName.empty() && !variantPriorsFile.empty()) {
    cerr << "Fatal ERROR: Both Hotspot VCF file and diagnostic VCF were specified at the same time" << endl;
    cerr << "Fatal ERROR: For diagnostic use please specify only <diagnostic-input-vcf> " << endl;
    exit(1);
  }
  else if (!candidateVCFFileName.empty()) {
    //pass on the diagnostics VCF file to freebayes for allele generation.
    variantPriorsFile = candidateVCFFileName;
  }
}

// ------------------------------------------------------------

void ExtendParameters::SetFreeBayesParameters(OptArgs &opts, Json::Value& fb_params) {
  // FreeBayes parameters
  // primarily used in candidate generation

  targets                               = opts.GetFirstString('t', "target-file", "");

  allowIndels                           = RetrieveParameterBool  (opts, fb_params, '-', "allow-indels", true);
  allowSNPs                             = RetrieveParameterBool  (opts, fb_params, '-', "allow-snps", true);
  allowMNPs                             = RetrieveParameterBool  (opts, fb_params, '-', "allow-mnps", true);
  allowComplex                          = RetrieveParameterBool  (opts, fb_params, '-', "allow-complex", false);
  leftAlignIndels                       = RetrieveParameterBool  (opts, fb_params, '-', "left-align-indels", false);
  if (leftAlignIndels) {
    cout << "WARNING: left-align-indels is an experimental feature." << endl;
  }
  
  useBestNAlleles                       = RetrieveParameterInt   (opts, fb_params, 'm', "use-best-n-alleles", 2);
  forceRefAllele                        = RetrieveParameterBool  (opts, fb_params, '-', "use-reference-allele", false);
  onlyUseInputAlleles                   = RetrieveParameterBool  (opts, fb_params, '-', "use-input-allele-only", false);
  MQL0                                  = RetrieveParameterInt   (opts, fb_params, 'M', "min-mapping-qv", 4);
  //BQL0                                  = RetrieveParameterInt   (opts, fb_params, 'q', "min-base-qv", 4);
  BQL0                                  = 0; // Do not use base qv filter!
  readSnpLimit                          = RetrieveParameterInt   (opts, fb_params, 'U', "read-snp-limit", 10);
  readMaxMismatchFraction               = RetrieveParameterDouble(opts, fb_params, 'z', "read-max-mismatch-fraction", 1.0);
  maxComplexGap                         = RetrieveParameterInt   (opts, fb_params, '!', "max-complex-gap", 1);
  // more FreeBayes parameters derived from other parameters
  if (forceRefAllele)
    useRefAllele = true;
  // read from json or command line, otherwise default to snp frequency
  minAltFraction                        = RetrieveParameterDouble(opts, fb_params, '-', "gen-min-alt-allele-freq", my_controls.filter_snps.min_allele_freq);
  minCoverage                           = RetrieveParameterInt   (opts, fb_params, '-', "gen-min-coverage", my_controls.filter_snps.min_cov);
  minIndelAltFraction                   = RetrieveParameterDouble(opts, fb_params, '-', "gen-min-indel-alt-allele-freq", my_controls.filter_hp_indel.min_allele_freq);
  //set up debug levels

  if (program_flow.DEBUG > 0)
    debug = true;
  if (program_flow.DEBUG > 1)
    debug2 = true;

  if (program_flow.inputPositionsOnly) {
    processInputPositionsOnly = true;
  }

  if (!vcfProvided && (processInputPositionsOnly || onlyUseInputAlleles) ) {
    cerr << "FATAL ERROR: Parameter error - Process-input-positions-only: " << processInputPositionsOnly << " use-input-allele-only: " << onlyUseInputAlleles << " :  Specified without Input VCF File " << endl;
    exit(-1);
  }
}

// ------------------------------------------------------------

void ExtendParameters::ParametersFromJSON(OptArgs &opts, Json::Value &tvc_params, Json::Value &freebayes_params) {
  string parameters_file                = opts.GetFirstString('-', "parameters-file", "");
  Json::Value parameters_json(Json::objectValue);
  if (not parameters_file.empty()) {
    ifstream in(parameters_file.c_str(), ifstream::in);
    if (!in.good()) {
      fprintf(stderr, "[tvc] ERROR: cannot open %s\n", parameters_file.c_str());
    }
    else {
      in >> parameters_json;
      in.close();
      if (parameters_json.isMember("pluginconfig"))
        parameters_json = parameters_json["pluginconfig"];
      tvc_params = parameters_json.get("torrent_variant_caller", Json::objectValue);
      freebayes_params = parameters_json.get("freebayes", Json::objectValue);
    }
  }
}

// ------------------------------------------------------------

void ExtendParameters::CheckParameterLimits() {
  // Check in the order they were set
  my_controls.CheckParameterLimits();
  my_eval_control.CheckParameterLimits();
  program_flow.CheckParameterLimits();

  // Checking FreeBayes parameters
  CheckParameterLowerUpperBound<int>  ("use-best-n-alleles",         useBestNAlleles,         1,    20);
  CheckParameterLowerBound<int>       ("min-mapping-qv",             MQL0,                    0);
  CheckParameterLowerBound<int>       ("read-snp-limit",             readSnpLimit,            0);
  CheckParameterLowerUpperBound<float>("read-max-mismatch-fraction", readMaxMismatchFraction, 0.0f, 1.0f);

  CheckParameterLowerUpperBound<long double>("gen-min-alt-allele-freq",       minAltFraction,      0.0, 1.0);
  CheckParameterLowerBound<int>             ("gen-min-coverage",              minCoverage,         0);
  CheckParameterLowerUpperBound<long double>("gen-min-indel-alt-allele-freq", minIndelAltFraction, 0.0, 1.0);

}


// ------------------------------------------------------------

ExtendParameters::ExtendParameters(int argc, char** argv)  {

  //OptArgs opts;
  opts.ParseCmdLine(argc, (const char**)argv);

  if (argc == 1) {
    VariantCallerHelp();
    exit(0);
  }
  if (opts.GetFirstBoolean('v', "version", false)) {
    exit(0);
  }
  if (opts.GetFirstBoolean('h', "help", false)) {
    VariantCallerHelp();
    exit(0);
  }

  Json::Value tvc_params(Json::objectValue);
  Json::Value freebayes_params(Json::objectValue);
  ParametersFromJSON(opts, tvc_params, freebayes_params);

  SetupFileIO(opts);

  sampleName                            = opts.GetFirstString('g', "sample-name", "");
  //referenceSampleName                   = opts.GetFirstString('G', "reference-sample", "");

  my_controls.SetOpts(opts, tvc_params);
  my_eval_control.SetOpts(opts, tvc_params);
  program_flow.SetOpts(opts, tvc_params);

  // Dummy lines for HP recalibration
  recal_model_file_name = opts.GetFirstString ('-', "model-file", "");
  recalModelHPThres = opts.GetFirstInt('-', "recal-model-hp-thres", 4);

  SetFreeBayesParameters(opts, freebayes_params);
  bool overrideLimits          = RetrieveParameterBool  (opts, tvc_params, '-', "override-limits", false);

  opts.CheckNoLeftovers();
  // Sanity checks on input variables once all are set.
  if (!overrideLimits)
    CheckParameterLimits();

}

