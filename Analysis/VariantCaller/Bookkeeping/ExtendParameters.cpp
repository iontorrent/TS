/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ExtendParameters.h"
#include <fenv.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <errno.h>
#include "MiscUtil.h"

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
  printf("     --do-indel-assembly                on/off      do indel assembly to call long indel variants [on]\n");
  printf("\n");

  printf("Inputs:\n");
  printf("  -r,--reference                        FILE        reference fasta file [required]\n");
  printf("  -b,--input-bam                        FILE        bam file with mapped reads [required]\n");
  printf("  -g,--sample-name                      STRING      sample for which variants are called (In case of input BAM files with multiple samples) [optional if there is only one sample]\n");
  printf("     --force-sample-name                STRING      force all read groups to have this sample name [off]\n");
  printf("  -t,--target-file                      FILE        only process targets in this bed file [optional]\n");
  printf("     --trim-ampliseq-primers            on/off      match reads to targets and trim the ends that reach outside them [off]\n");
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
  printf("  -m,--use-best-n-alleles               INT         maximum number of snp alleles [2]\n");
  printf("  -M,--min-mapping-qv                   INT         do not use reads with mapping quality below this [4]\n");
  printf("  -U,--read-snp-limit                   INT         do not use reads with number of snps above this [10]\n");
  printf("  -z,--read-max-mismatch-fraction       FLOAT       do not use reads with fraction of mismatches above this [1.0]\n");
  printf("     --gen-min-alt-allele-freq          FLOAT       minimum required alt allele frequency to generate a candidate [0.2]\n");
  printf("     --gen-min-indel-alt-allele-freq    FLOAT       minimum required alt allele frequency to generate a homopolymer indel candidate [0.2]\n");
  printf("     --gen-min-coverage                 INT         minimum required coverage to generate a candidate [6]\n");
  printf("     --merge-variant-lookahead          INT         how many bases ahead to merge nearby variant to form correct haplotype [3, 0 if not allow complex]\n");
  printf("\n");

  printf("External variant candidates:\n");
  printf("  -l,--blacklist-vcf                    FILE        vcf.gz file (+.tbi) with blacklist candidate variant locations and alleles [optional]\n");
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
  printf("     --do-mnp-realignment               on/off      Realign reads in the vicinity of candidate mnp variants [do-snp-realignment].\n");
  printf("     --realignment-threshold            FLOAT       Max. allowed fraction of reads where realignment causes an alignment change [1.0].\n");
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
  printf("     --max-detail-level                 INT         number of evaluated frequencies for a given hypothesis, reduce for very high coverage, set to zero to disable this option [0]\n");
  printf("     --min-detail-level-for-fast-scan   INT         minimum detail level to trigger the fast scan for log-posterior of frequencies [2500]\n");
  printf("     --try-few-restart-freq             on/off      speed up the evaluator by trying fewer initial guesses of allele frequency [off].\n");

  printf("\n");

  printf("Variant filtering:\n");
  // Filters depending on the variant type
  printf("  -k,--snp-min-coverage                 INT         filter out snps with total coverage below this [6]\n");
  printf("  -C,--snp-min-cov-each-strand          INT         filter out snps with coverage on either strand below this [0]\n");
  printf("  -B,--snp-min-variant-score            FLOAT       filter out snps with QUAL score below this [10.0]\n");
  printf("  -s,--snp-strand-bias                  FLOAT       filter out snps with strand bias above this [0.95] given strand bias > snp-strand-bias\n");
  printf("     --snp-strand-bias-pval             FLOAT       filter out snps with pval below this [1.0] given pval < snp-strand-bias-pval\n");
  //  printf("  -s,--snp-strand-bias                FLOAT       filter out snps with strand bias above this [0.95]\n");
  printf("  -A,--snp-min-allele-freq              FLOAT       minimum required alt allele frequency for non-reference snp calls [0.2]\n");
  printf("     --snp-min-var-coverage             INT         filter out snps with variant allele coverage below this [0]\n");

  printf("     --mnp-min-coverage                 INT         filter out mnps with total coverage below this [snp-min-coverage]\n");
  printf("     --mnp-min-cov-each-strand          INT         filter out mnps with coverage on either strand below this, [snp-min-cov-each-strand]\n");
  printf("     --mnp-min-variant-score            FLOAT       filter out mnps with QUAL score below this [snp-min-variant-score]\n");
  printf("     --mnp-strand-bias                  FLOAT       filter out mnps with strand bias above this [snp-strand-bias] given strand bias > mnp-strand-bias\n");
  printf("     --mnp-strand-bias-pval             FLOAT       filter out mnps with pval below this [snp-strand-bias-pval] given pval < mnp-strand-bias-pval\n");
  printf("     --mnp-min-allele-freq              FLOAT       minimum required alt allele frequency for non-reference mnp calls [snp-min-allele-freq]\n");
  printf("     --mnp-min-var-coverage             INT         filter out mnps with variant allele coverage below this [snp-min-var-coverage]\n");

  printf("     --indel-min-coverage               INT         filter out indels with total coverage below this [30]\n");
  printf("     --indel-min-cov-each-strand        INT         filter out indels with coverage on either strand below this [1]\n");
  printf("     --indel-min-variant-score          FLOAT       filter out indels with QUAL score below this [10.0]\n");
  printf("  -S,--indel-strand-bias                FLOAT       filter out indels with strand bias above this [0.95] given strand bias > indel-strand-bias\n");
  printf("     --indel-strand-bias-pval           FLOAT       filter out indels with pval below this [1.0] given pval < indel-strand-bias-pval\n");
  //  printf("  -S,--indel-strand-bias                FLOAT       filter out indels with strand bias above this [0.85]\n");
  printf("     --indel-min-allele-freq            FLOAT       minimum required alt allele frequency for non-reference indel call [0.2]\n");
  printf("     --indel-min-var-coverage           INT         filter out indels with variant allele coverage below this [snp-min-var-coverage]\n");

  printf("     --hotspot-min-coverage             INT         filter out hotspot variants with total coverage below this [6]\n");
  printf("     --hotspot-min-cov-each-strand      INT         filter out hotspot variants with coverage on either strand below this [snp-min-cov-each-strand]\n");
  printf("     --hotspot-min-variant-score        FLOAT       filter out hotspot variants with QUAL score below this [snp-min-variant-score]\n");
  printf("     --hotspot-strand-bias              FLOAT       filter out hotspot variants with strand bias above this [0.95] given strand bias > hotspot-strand-bias\n");
  printf("     --hotspot-strand-bias-pval         FLOAT       filter out hotspot variants with pval below this [1.0] given pval < hotspot-strand-bias-pval\n");
  //  printf("     --hotspot-strand-bias              FLOAT       filter out hotspot variants with strand bias above this [0.95]\n");
  printf("  -H,--hotspot-min-allele-freq          FLOAT       minimum required alt allele frequency for non-reference hotspot variant call [0.2]\n");
  printf("     --hotspot-min-var-coverage         INT         filter out hotspot variants with variant allele coverage below this [snp-min-var-coverage]\n");

  // Filters not depending on the variant score
  printf("  -L,--hp-max-length                    INT         filter out indels in homopolymers above this [8]\n");
  printf("  -e,--error-motifs                     FILE        table of systematic error motifs and their error rates [optional]\n");
  printf("     --sse-prob-threshold               FLOAT       filter out variants in motifs with error rates above this [0.2]\n");
  printf("     --min-ratio-reads-non-sse-strand   FLOAT       minimum required alt allele frequency for variants with error motifs on opposite strand [0.2]\n");
  printf("     --indel-as-hpindel                 on/off      apply indel filters to non HP indels [off]\n");
  printf("     --use-lod-filter                   on/off      enable the Limit Of Detection (LOD) filter [off]\n");
  printf("     --lod-multiplier                   FLOAT       multiplier of LOD for filtering out variants with low alt allele frequency [0.6]\n");

  // position-bias filter
  printf("\nPosition bias variant filtering:\n");
  printf("     --use-position-bias                on/off      enable the position bias filter [off]\n");
  printf("     --position-bias                    FLOAT       filter out variants with position bias relative to soft clip ends in reads > position-bias [0.75]\n");
  printf("     --position-bias-pval               FLOAT       filter out if position bias above position-bias given pval < position-bias-pval [0.05]\n");
  printf("     --position-bias-ref-fraction       FLOAT       skip position bias filter if (reference read count)/(reference + alt allele read count) <= to this [0.05]\n");
  // These filters depend on scoring
  printf("\nFilters that depend on scoring across alleles:\n");
  printf("     --data-quality-stringency          FLOAT       minimum mean log-likelihood delta per read [4.0]\n");
  printf("     --read-rejection-threshold         FLOAT       filter variants where large numbers of reads are rejected as outliers [0.5]\n");
  printf("     --filter-unusual-predictions       FLOAT       posterior log likelihood threshold for accepting bias estimate [0.3]\n");
  printf("     --filter-deletion-predictions      FLOAT       check post-evaluation systematic bias in deletions; a high value like 100 effectively turns off this filter [100.0]\n");
  printf("     --filter-insertion-predictions     FLOAT       check post-evaluation systematic bias in insertions; a high value like 100 effectively turns off this filter [100.0]\n");
  printf("\n");
  printf("     --heal-snps                        on/off      suppress in/dels not participating in diploid variant genotypes if the genotype contains a SNP or MNP [on].\n");
  printf("\n");

  MolecularTagTrimmer::PrintHelp(true);

  printf("Debugging:\n");
  printf("  -d,--debug                            INT         (0/1/2) display extra debug messages [0]\n");
  printf("     --do-json-diagnostic               on/off      (devel) dump internal state to json file (uses much more time/memory/disk) [off]\n");
  printf("     --postprocessed-bam                FILE        (devel) save tvc-processed reads to an (unsorted) BAM file [optional]\n");
  printf("     --do-minimal-diagnostic            on/off      (devel) provide minimal read information for called variants [off]\n");
  printf("     --override-limits                  on/off      (devel) disable limit-check on input parameters [off].\n");
  printf("     --output-multi-min-allele-freq     on/off      output the inference results for multiple min-allele-freq in vcf [off].\n");
  printf("     --snp-multi-min-allele-freq        FLOAT VECTOR  multiple min-allele-freq for snp calls [0.05,0.1,0.15,0.2].\n");
  printf("     --mnp-multi-min-allele-freq        FLOAT VECTOR  multiple min-allele-freq for mnp calls [snp-multi-min-allele-freq].\n");
  printf("     --indel-multi-min-allele-freq      FLOAT VECTOR  multiple min-allele-freq for indel calls [0.05,0.1,0.15,0.2].\n");
  printf("     --hotspot-multi-min-allele-freq    FLOAT VECTOR  multiple min-allele-freq for hotspot calls [0.05,0.1,0.15,0.2].\n");
  printf("\n");
}


ControlCallAndFilters::ControlCallAndFilters() {
  // all defaults handled by sub-filters
  data_quality_stringency = 4.0f;  // phred-score for this variant per read
  read_rejection_threshold = 0.5f; // half the reads gone, filter this

  use_position_bias = false;
  position_bias_ref_fraction = 0.05;  // FRO/(FRO+FAO)
  position_bias = 0.75f;              // position bias
  position_bias_pval = 0.05f;         // pval for observed > threshold

  use_lod_filter = false;
  lod_multiplier = 0.6f;

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
  do_indel_assembly = true;

#ifdef __SSE3__
  use_SSE_basecaller = true;
#else
  use_SSE_basecaller = false;
#endif
  rich_json_diagnostic = false;
  minimal_diagnostic = false;
  json_plot_dir = "./json_diagnostic/";
  inputPositionsOnly = false;
  suppress_recalibration = true;
  resolve_clipped_bases = false;

  is_multi_min_allele_freq = false;
  snp_multi_min_allele_freq.clear();
  mnp_multi_min_allele_freq.clear();
  indel_multi_min_allele_freq.clear();
  hotspot_multi_min_allele_freq.clear();
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

void ConvertToUpper(string &s){
    for(unsigned int i = 0; i<s.size(); ++i)
        s[i] = toupper(s[i]);
}

string RetrieveParameterString(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value)
{
  string long_name_underscores = long_name_hyphens;
  for (unsigned int i = 0; i < long_name_underscores.size(); ++i)
    if (long_name_underscores[i] == '-')
      long_name_underscores[i] = '_';

  string value = default_value;
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
    value = json[long_name_underscores].asString();
    source = "parameters json file";
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
    value = opts.GetFirstString(short_name, long_name_hyphens, value);
    source = "command line option";
  }
  ConvertToUpper(value);
  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (string,  " << source << ")" << endl;
  return value;
}

string GetRidOfDomainAndHyphens(const string& name)
{
	string s(name);
	int index = name.rfind("::");
	if(index > -1)
	{
		s = name.substr(index + 2, name.length() - index - 2);
	}

	for (unsigned int i = 0; i < s.size(); ++i)
	{
		if (s[i] == '-')
		s[i] = '_';
	}

	return s;
}

void Split(const string& s, char c, vector<string>& v)
{
  v.clear();
  if (s != "") {
    string::size_type i = 0;
    string::size_type j = s.find (c);
    if (j == string::npos)
    {
      v.push_back (s);
    }
    else
    {
      while (j != string::npos)
      {
        v.push_back (s.substr (i, j-i));
        i = ++j;
        j = s.find (c,j);

        if (j == string::npos)
        {
          v.push_back (s.substr (i, s.length()));
        }
      }
    }
  }
}

int RetrieveParameterVectorFloat(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, const string& default_value, vector<float>& ret_vector)
{
  string long_name_underscores = GetRidOfDomainAndHyphens(long_name_hyphens);
  string value = default_value;

  if(value.length() > 0)
  {
	  vector<string> words;
	  Split(value,',',words);
	  ret_vector.clear();
	  for (size_t i = 0; i < words.size(); i++) {
		char *end;
		int err = errno;
		errno = 0;
		ret_vector.push_back(strtod(words[i].c_str(), &end));
		if (errno != 0 || *end != '\0') {
		  cout << "Error converting: " + words[i] + " to an float for option: " + long_name_hyphens << endl;
		  return errno;
		}
		errno = err;
	  }
  }
  string source = "builtin default";

  if (json.isMember(long_name_underscores)) {
	  ret_vector.clear();
	  size_t sz = json[long_name_underscores].size();
	  char buf[1000];
      if(sz > 0)
	  {
          if(sz == 1)
          {
              if(json[long_name_underscores][0].isString())
              {
                  ret_vector.push_back(atof(json[long_name_underscores][0].asCString()));
                  value = json[long_name_underscores][0].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][0].asFloat());
                  sprintf(buf, "%f", ret_vector[0]);
                  value = buf;
              }
          }
          else
          {
              value = "";
              for(int i = 0; i < (int)sz - 1; i++)
              {
                  if(json[long_name_underscores][i].isString())
                  {
                      ret_vector.push_back(atof(json[long_name_underscores][i].asCString()));
                      value += json[long_name_underscores][i].asCString();
                      value += ",";
                  }
                  else
                  {
                      ret_vector.push_back(json[long_name_underscores][i].asFloat());
                      sprintf(buf, "%f,", ret_vector[i]);
                      string ss = buf;
                      value += ss;
                  }
              }

              if(json[long_name_underscores][(int)sz - 1].isString())
              {
                  ret_vector.push_back(atof(json[long_name_underscores][(int)sz - 1].asCString()));
                  value += json[long_name_underscores][(int)sz - 1].asCString();
              }
              else
              {
                  ret_vector.push_back(json[long_name_underscores][(int)sz - 1].asFloat());
                  sprintf(buf, "%f", ret_vector[(int)sz - 1]);
                  string ss = buf;
                  value += ss;
              }
          }
          source = "parameters json file";
      }
  }

  if (opts.HasOption(short_name, long_name_hyphens)) {
	  ret_vector.clear();
	  vector<double> ret_vector2;
	  opts.GetOption(ret_vector2, default_value, short_name, long_name_hyphens);
	  for(size_t i = 0; i < ret_vector2.size(); i++)
	  {
		  ret_vector.push_back((float)ret_vector2[i]);
	  }

	  char buf[1000];
	  if(ret_vector.empty())
	  {
	      cout << "Error setting: there is no value set for option: " + long_name_hyphens << endl;
		  return 1;
	  }
	  else if(ret_vector.size() == 1)
	  {
		  sprintf(buf, "%f", ret_vector[0]);
		  value = buf;
	  }
	  else
	  {
		  value = "";
		  for(size_t i = 0; i < ret_vector.size() - 1; i++) {
			  sprintf(buf, "%f,", ret_vector[i]);
			  string ss = buf;
			  value += ss;
		  }
		  sprintf(buf, "%f", ret_vector[ret_vector.size() - 1]);
		  string ss = buf;
		  value += ss;
	  }
	  source = "command line option";
  }

  cout << setw(35) << long_name_hyphens << " = " << setw(10) << value << " (float,  " << source << ")" << endl;
  return 0;
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
  max_detail_level                      = RetrieveParameterInt(opts, tvc_params, '-', "max-detail-level", 0);
  min_detail_level_for_fast_scan        = RetrieveParameterInt(opts, tvc_params, '-', "min-detail-level-for-fast-scan", 2500);
  try_few_restart_freq                  = RetrieveParameterBool(opts, tvc_params, '-', "try-few-restart-freq", false);

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
  CheckParameterLowerUpperBound<float>("outlier-probability",     outlier_prob,            0.0000001f,  1.0f); // extremely low outlier_prob causes floating exception
  CheckParameterLowerUpperBound<float>("germline-prior-strength", germline_prior_strength, 0.0f,  1000.0f);
  CheckParameterLowerBound<int>       ("heavy-tailed",            heavy_tailed,            1);

  CheckParameterLowerBound<float>     ("filter-unusual-predictions",    filter_unusual_predictions,   0.0f);
  CheckParameterLowerUpperBound<float>("soft-clip-bias-checker",        soft_clip_bias_checker, 0.0f, 1.0f);
  CheckParameterLowerBound<float>     ("filter-deletion-predictions",   filter_deletion_bias,         0.0f);
  CheckParameterLowerBound<float>     ("filter-insertion-predictions",  filter_insertion_bias,        0.0f);
  CheckParameterLowerUpperBound<int>  ("max-detail-level",    max_detail_level,   0, 10000);
  CheckParameterLowerBound<int>       ("min-detail-level-for-fast-scan",min_detail_level_for_fast_scan,   0);


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
  do_snp_realignment                    = RetrieveParameterBool  (opts, tvc_params, '-', "do-snp-realignment", true);
  do_mnp_realignment                    = RetrieveParameterBool  (opts, tvc_params, '-', "do-mnp-realignment", do_snp_realignment);
  realignment_threshold                 = RetrieveParameterDouble(opts, tvc_params, '-', "realignment-threshold", 1.0);

  indel_as_hpindel               = RetrieveParameterBool  (opts, tvc_params, '-', "indel-as-hpindel", false);
}

void ClassifyFilters::CheckParameterLimits() {

  CheckParameterLowerBound<int>       ("hp-max-length",        hp_max_length,        1);
  CheckParameterLowerUpperBound<float>("sse-prob-threshold",   sseProbThreshold, 0.0f, 1.0f);
  CheckParameterLowerUpperBound<float>("min-ratio-reads-non-sse-strand",   minRatioReadsOnNonErrorStrand, 0.0f, 1.0f);
  CheckParameterLowerUpperBound<float>("sse-relative-safety-level",   sse_relative_safety_level, 0.0f, 1.0f);
  CheckParameterLowerUpperBound<float>("realignment-threshold",   realignment_threshold, 0.0f, 1.0f);

}

// ===========================================================================

void ControlCallAndFilters::CheckParameterLimits() {

  filter_variant.CheckParameterLimits();
  CheckParameterLowerBound<float>     ("data-quality-stringency",  data_quality_stringency,  0.0f);
  CheckParameterLowerUpperBound<float>("read-rejection-threshold", read_rejection_threshold, 0.0f, 1.0f);
  CheckParameterLowerUpperBound<int>  ("downsample-to-coverage",   downSampleCoverage,       20, 100000);
  CheckParameterLowerUpperBound<float>("position-bias-ref-fraction",position_bias_ref_fraction,  0.0f, 1.0f);
  CheckParameterLowerUpperBound<float>("position-bias",            position_bias,  0.0f, 1.0f);
  CheckParameterLowerUpperBound<float>("position-bias-pval",       position_bias_pval,  0.0f, 1.0f);
 // CheckParameterLowerUpperBound<float>("tune-xbias",      xbias_tune,     0.001f, 1000.0f);
  CheckParameterLowerUpperBound<float>("tune-sbias",      sbias_tune,     0.001f, 1000.0f);
  CheckParameterLowerBound<float>     ("lod-multiplier",           lod_multiplier,         0.6f);

  CheckParameterLowerBound<int>       ("snp-min-cov-each-strand",  filter_snps.min_cov_each_strand, 0);
  CheckParameterLowerBound<float>     ("snp-min-variant-score",    filter_snps.min_quality_score,   0.0f);
  CheckParameterLowerUpperBound<float>("snp-min-allele-freq",      filter_snps.min_allele_freq,     0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("snp-min-coverage",         filter_snps.min_cov,             0);
  CheckParameterLowerUpperBound<float>("snp-strand-bias",          filter_snps.strand_bias_threshold,  0.5f, 1.0f);
  CheckParameterLowerUpperBound<float>("snp-strand-bias-pval",     filter_snps.strand_bias_pval_threshold,  0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("snp-min-var-coverage",     filter_snps.min_var_cov,         0);
//  CheckParameterLowerBound<float>     ("snp-beta-bias",            filter_snps.beta_bias_filter,    0.0f);

  CheckParameterLowerBound<int>       ("mnp-min-cov-each-strand",  filter_mnp.min_cov_each_strand, 0);
  CheckParameterLowerBound<float>     ("mnp-min-variant-score",    filter_mnp.min_quality_score,   0.0f);
  CheckParameterLowerUpperBound<float>("mnp-min-allele-freq",      filter_mnp.min_allele_freq,     0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("mnp-min-coverage",         filter_mnp.min_cov,             0);
  CheckParameterLowerUpperBound<float>("mnp-strand-bias",          filter_mnp.strand_bias_threshold,  0.5f, 1.0f);
  CheckParameterLowerUpperBound<float>("mnp-strand-bias-pval",     filter_mnp.strand_bias_pval_threshold,  0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("mnp-min-var-coverage",     filter_mnp.min_var_cov,         0);

  CheckParameterLowerBound<int>       ("indel-min-cov-each-strand",  filter_hp_indel.min_cov_each_strand, 0);
  CheckParameterLowerBound<float>     ("indel-min-variant-score",    filter_hp_indel.min_quality_score,   0.0f);
  CheckParameterLowerUpperBound<float>("indel-min-allele-freq",      filter_hp_indel.min_allele_freq,     0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("indel-min-coverage",         filter_hp_indel.min_cov,             0);
  CheckParameterLowerUpperBound<float>("indel-strand-bias",          filter_hp_indel.strand_bias_threshold,  0.5f, 1.0f);
  CheckParameterLowerUpperBound<float>("indel-strand-bias-pval",     filter_hp_indel.strand_bias_pval_threshold,  0.0f, 1.0f);
//  CheckParameterLowerBound<float>     ("indel-beta-bias",            filter_hp_indel.beta_bias_filter,    0.0f);
  CheckParameterLowerBound<int>       ("indel-min-var-coverage",     filter_hp_indel.min_var_cov,         0);

  CheckParameterLowerBound<int>       ("hotspot-min-cov-each-strand",  filter_hotspot.min_cov_each_strand, 0);
  CheckParameterLowerBound<float>     ("hotspot-min-variant-score",    filter_hotspot.min_quality_score,   0.0f);
  CheckParameterLowerUpperBound<float>("hotspot-min-allele-freq",      filter_hotspot.min_allele_freq,     0.0f, 1.0f);
  CheckParameterLowerBound<int>       ("hotspot-min-coverage",         filter_hotspot.min_cov,             0);
  CheckParameterLowerUpperBound<float>("hotspot-strand-bias",          filter_hotspot.strand_bias_threshold,  0.5f, 1.0f);
  CheckParameterLowerUpperBound<float>("hotspot-strand-bias-pval",     filter_hotspot.strand_bias_pval_threshold,  0.0f, 1.0f);
//  CheckParameterLowerBound<float>     ("hotspot-beta-bias",            filter_hotspot.beta_bias_filter,    0.0f);
  CheckParameterLowerBound<int>       ("hotspot-min-var-coverage",     filter_hotspot.min_var_cov,         0);
}

// ------------------------------------------------------

void ControlCallAndFilters::SetOpts(OptArgs &opts, Json::Value& tvc_params) {

  filter_variant.SetOpts(opts, tvc_params);
  RandSeed = 631;    // Not exposed to user at this point

  // catchall filter parameter to be used to filter any generic predictive model of quality
  data_quality_stringency               = RetrieveParameterDouble(opts, tvc_params, '-', "data-quality-stringency",4.0f);
  // if we reject half the reads from evaluator, something badly wrong with this position
  read_rejection_threshold              = RetrieveParameterDouble(opts, tvc_params, '-', "read-rejection-threshold",0.5f);

  use_position_bias                     = RetrieveParameterBool(opts, tvc_params, '-', "use-position-bias", false);
  position_bias_ref_fraction            = RetrieveParameterDouble(opts, tvc_params, '-', "position-bias-ref-fraction",0.05f);
  position_bias                         = RetrieveParameterDouble(opts, tvc_params, '-', "position-bias",0.75f);
  position_bias_pval                    = RetrieveParameterDouble(opts, tvc_params, '-', "position-bias-pval",0.05f);

  use_lod_filter                        = RetrieveParameterBool(opts, tvc_params, '-', "use-lod-filter", false);
  lod_multiplier                        = RetrieveParameterDouble(opts, tvc_params, '-', "lod-multiplier",0.6f);

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
  filter_snps.strand_bias_pval_threshold= RetrieveParameterDouble(opts, tvc_params, 's', "snp-strand-bias-pval", 1.0);
  filter_snps.min_var_cov               = RetrieveParameterInt   (opts, tvc_params, '-', "snp-min-var-coverage", 0);

  filter_mnp.min_cov_each_strand    = RetrieveParameterInt   (opts, tvc_params, '-', "mnp-min-cov-each-strand", filter_snps.min_cov_each_strand);
  filter_mnp.min_quality_score      = RetrieveParameterDouble(opts, tvc_params, '-', "mnp-min-variant-score", filter_snps.min_quality_score);
  filter_mnp.min_allele_freq        = RetrieveParameterDouble(opts, tvc_params, 'H', "mnp-min-allele-freq", filter_snps.min_allele_freq);
  filter_mnp.min_cov                = RetrieveParameterInt   (opts, tvc_params, '-', "mnp-min-coverage", filter_snps.min_cov);
  filter_mnp.strand_bias_threshold  = RetrieveParameterDouble(opts, tvc_params, '-', "mnp-strand-bias", filter_snps.strand_bias_threshold);
  filter_mnp.strand_bias_pval_threshold= RetrieveParameterDouble(opts, tvc_params, 's', "mnp-strand-bias-pval", filter_snps.strand_bias_pval_threshold);
  filter_mnp.min_var_cov            = RetrieveParameterInt   (opts, tvc_params, '-', "mnp-min-var-coverage", filter_snps.min_var_cov);

  // hp_indels are more complex
  filter_hp_indel.min_cov_each_strand   = RetrieveParameterInt   (opts, tvc_params, '-', "indel-min-cov-each-strand", 1);
  filter_hp_indel.min_quality_score     = RetrieveParameterDouble(opts, tvc_params, '-', "indel-min-variant-score", 10.0);
  filter_hp_indel.min_allele_freq       = RetrieveParameterDouble(opts, tvc_params, '-', "indel-min-allele-freq", 0.2);
  filter_hp_indel.min_cov               = RetrieveParameterInt   (opts, tvc_params, '-', "indel-min-coverage", 15);
  filter_hp_indel.strand_bias_threshold = RetrieveParameterDouble(opts, tvc_params, 'S', "indel-strand-bias", 0.85);
  filter_hp_indel.strand_bias_pval_threshold= RetrieveParameterDouble(opts, tvc_params, 's', "indel-strand-bias-pval", 1.0);
  filter_hp_indel.min_var_cov           = RetrieveParameterInt   (opts, tvc_params, '-', "indel-min-var-coverage", filter_snps.min_var_cov);
  // derive hotspots by default from SNPs
  // override from command line or json
  filter_hotspot.min_cov_each_strand    = RetrieveParameterInt   (opts, tvc_params, '-', "hotspot-min-cov-each-strand", filter_snps.min_cov_each_strand);
  filter_hotspot.min_quality_score      = RetrieveParameterDouble(opts, tvc_params, '-', "hotspot-min-variant-score", filter_snps.min_quality_score);
  filter_hotspot.min_allele_freq        = RetrieveParameterDouble(opts, tvc_params, 'H', "hotspot-min-allele-freq", filter_snps.min_allele_freq);
  filter_hotspot.min_cov                = RetrieveParameterInt   (opts, tvc_params, '-', "hotspot-min-coverage", filter_snps.min_cov);
  filter_hotspot.strand_bias_threshold  = RetrieveParameterDouble(opts, tvc_params, '-', "hotspot-strand-bias", filter_snps.strand_bias_threshold);
  filter_hotspot.strand_bias_pval_threshold= RetrieveParameterDouble(opts, tvc_params, 's', "hotspot-strand-bias-pval", filter_snps.strand_bias_pval_threshold);
  filter_hotspot.min_var_cov            = RetrieveParameterInt   (opts, tvc_params, '-', "hotspot-min-var-coverage", filter_snps.min_var_cov);

}

// =============================================================================

void ProgramControlSettings::CheckParameterLimits() {

  CheckParameterLowerUpperBound<int>  ("num-threads",              nThreads,             1, 128);
  CheckParameterLowerUpperBound<int>  ("num-variants-per-thread",  nVariantsPerThread,   1, 10000);
  for(unsigned int i_freq = 0; i_freq < snp_multi_min_allele_freq.size(); ++i_freq){
	  string identifier = "multi-min-allele-freq[" + convertToString(i_freq) + "]";
	  CheckParameterLowerUpperBound<float>  (identifier, snp_multi_min_allele_freq[i_freq], 0.0f, 1.0f);
  }
  for(unsigned int i_freq = 0; i_freq < mnp_multi_min_allele_freq.size(); ++i_freq){
	  string identifier = "mnp-multi-min-allele-freq[" + convertToString(i_freq) + "]";
	  CheckParameterLowerUpperBound<float>  (identifier, mnp_multi_min_allele_freq[i_freq], 0.0f, 1.0f);
  }
  for(unsigned int i_freq = 0; i_freq < indel_multi_min_allele_freq.size(); ++i_freq){
	  string identifier = "indel-multi-min-allele-freq[" + convertToString(i_freq) + "]";
	  CheckParameterLowerUpperBound<float>  (identifier, indel_multi_min_allele_freq[i_freq], 0.0f, 1.0f);
  }
  for(unsigned int i_freq = 0; i_freq < hotspot_multi_min_allele_freq.size(); ++i_freq){
	  string identifier = "hotspot-multi-min-allele-freq[" + convertToString(i_freq) + "]";
	  CheckParameterLowerUpperBound<float>  (identifier, hotspot_multi_min_allele_freq[i_freq], 0.0f, 1.0f);
  }
}

void ProgramControlSettings::SetOpts(OptArgs &opts, Json::Value &tvc_params) {

  DEBUG                                 = opts.GetFirstInt   ('d', "debug", 0);
  nThreads                              = RetrieveParameterInt   (opts, tvc_params, 'n', "num-threads", 12);
  nVariantsPerThread                    = RetrieveParameterInt   (opts, tvc_params, 'N', "num-variants-per-thread", 250);
#ifdef __SSE3__
  use_SSE_basecaller                    = RetrieveParameterBool  (opts, tvc_params, '-', "use-sse-basecaller", true);
#else
  use_SSE_basecaller                    = RetrieveParameterBool  (opts, tvc_params, '-', "use-sse-basecaller", false);
#endif
  // decide diagnostic
  rich_json_diagnostic                  = RetrieveParameterBool  (opts, tvc_params, '-', "do-json-diagnostic", false);
  minimal_diagnostic                    = RetrieveParameterBool  (opts, tvc_params, '-', "do-minimal-diagnostic", false);

  inputPositionsOnly                    = RetrieveParameterBool  (opts, tvc_params, '-', "process-input-positions-only", false);
  suppress_recalibration                = RetrieveParameterBool  (opts, tvc_params, '-', "suppress-recalibration", true);
  resolve_clipped_bases                 = RetrieveParameterBool  (opts, tvc_params, '-', "resolve-clipped-bases", false);

  use_SSE_basecaller                    = RetrieveParameterBool  (opts, tvc_params, '-', "use-sse-basecaller", true);

  do_indel_assembly                     = RetrieveParameterBool  (opts, tvc_params, '-', "do-indel-assembly", true);

  RetrieveParameterVectorFloat(opts, tvc_params, '-', "snp-multi-min-allele-freq", "0.05,0.1,0.15,0.2", snp_multi_min_allele_freq);
  string snp_multi_min_allele_freq_str = "";
  for(unsigned i_freq = 0; i_freq < snp_multi_min_allele_freq.size(); ++i_freq){
	  snp_multi_min_allele_freq_str += convertToString(snp_multi_min_allele_freq[i_freq]);
	  if(i_freq < snp_multi_min_allele_freq.size() - 1){
		  snp_multi_min_allele_freq_str += ",";
	  }
  }
  RetrieveParameterVectorFloat(opts, tvc_params, '-', "mnp-multi-min-allele-freq", snp_multi_min_allele_freq_str, mnp_multi_min_allele_freq);
  RetrieveParameterVectorFloat(opts, tvc_params, '-', "indel-multi-min-allele-freq", "0.05,0.1,0.15,0.2", indel_multi_min_allele_freq);
  RetrieveParameterVectorFloat(opts, tvc_params, '-', "hotspot-multi-min-allele-freq", "0.05,0.1,0.15,0.2", hotspot_multi_min_allele_freq);
}

// ===========================================================================

bool ExtendParameters::ValidateAndCanonicalizePath(string &path)
{
  char *real_path = realpath (path.c_str(), NULL);
  if (real_path == NULL) {
    perror(path.c_str());
    exit(EXIT_FAILURE);
  }
  path = real_path;
  free(real_path);
  return true;
}

int mkpath(std::string s,mode_t mode)
{
    size_t pre=0,pos;
    std::string dir;
    int mdret = 0;

    if(s[s.size()-1]!='/'){
        // force trailing / so we can handle everything in loop
        s+='/';
    }

    while((pos=s.find_first_of('/',pre))!=std::string::npos){
        dir=s.substr(0,pos++);
        pre=pos;
        if(dir.size()==0) continue; // if leading / first time is 0 length
        if((mdret=mkdir(dir.c_str(),mode)) && errno!=EEXIST){
            return mdret;
        }
    }
    return mdret;
}

void ExtendParameters::SetupFileIO(OptArgs &opts, Json::Value& tvc_params) {
  // freeBayes slot
  fasta                                 = opts.GetFirstString('r', "reference", "");
  if (fasta.empty()) {
    cerr << "Fatal ERROR: Reference file not specified via -r" << endl;
    exit(1);
  }
  ValidateAndCanonicalizePath(fasta);

  // freeBayes slot
  blacklistFile                     = opts.GetFirstString('l', "blacklist-vcf", "");
  if (blacklistFile.empty()) {
	cerr << "INFO: No blacklist VCF file specified via -l,--blacklist-vcf" << endl;
  }
  else
	ValidateAndCanonicalizePath(blacklistFile);
  variantPriorsFile                     = opts.GetFirstString('c', "input-vcf", "");
  if (variantPriorsFile.empty()) {
    cerr << "INFO: No input VCF (Hotspot) file specified via -c,--input-vcf" << endl;
  }
  else
	ValidateAndCanonicalizePath(variantPriorsFile);

  sseMotifsFileName                     = opts.GetFirstString('e', "error-motifs", "");
  sseMotifsProvided = true;
  if (sseMotifsFileName.empty()) {
    sseMotifsProvided = false;
    cerr << "INFO: Systematic error motif file not specified via -e" << endl;
  }
  else {
    cout << "INFO: Using error-motifs file:\t" << sseMotifsFileName << endl;
    ValidateAndCanonicalizePath(sseMotifsFileName);
  }

  opts.GetOption(bams, "", 'b', "input-bam");
  if (bams.empty()) {
    cerr << "FATAL ERROR: BAM file not specified via -b" << endl;
    exit(-1);
  }
  for (unsigned int i_bam = 0; i_bam < bams.size(); ++i_bam)
    ValidateAndCanonicalizePath(bams[i_bam]);

  outputDir                             = opts.GetFirstString('O', "output-dir", ".");
  mkpath(outputDir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  ValidateAndCanonicalizePath(outputDir);

  outputFile                            = opts.GetFirstString('o', "output-vcf", "");
  if (outputFile.empty()) {
    cerr << "Fatal ERROR: Output VCF filename not specified via -o" << endl;
    exit(1);
  }

  // Are those file names?
  postprocessed_bam                     = opts.GetFirstString('-', "postprocessed-bam", "");
  sampleName                            = opts.GetFirstString('g', "sample-name", "");
  force_sample_name                     = opts.GetFirstString('-', "force-sample-name", "");

}

// ------------------------------------------------------------

void ExtendParameters::SetFreeBayesParameters(OptArgs &opts, Json::Value& fb_params) {
  // FreeBayes parameters
  // primarily used in candidate generation

  targets                               = opts.GetFirstString('t', "target-file", "");
  trim_ampliseq_primers                 = opts.GetFirstBoolean('-', "trim-ampliseq-primers", false);
  if (targets.empty() and trim_ampliseq_primers) {
    cerr << "ERROR: --trim-ampliseq-primers enabled but no --target-file provided" << endl;
    exit(1);
  }

  allowIndels                           = RetrieveParameterBool  (opts, fb_params, '-', "allow-indels", true);
  allowSNPs                             = RetrieveParameterBool  (opts, fb_params, '-', "allow-snps", true);
  allowMNPs                             = RetrieveParameterBool  (opts, fb_params, '-', "allow-mnps", true);
  allowComplex                          = RetrieveParameterBool  (opts, fb_params, '-', "allow-complex", false);
  mergeLookAhead			= RetrieveParameterInt   (opts, fb_params, '-', "merge-variant-lookahead", 3);
  
  // deprecated:
  // leftAlignIndels                       = RetrieveParameterBool  (opts, fb_params, '-', "left-align-indels", false);
  RetrieveParameterBool  (opts, fb_params, '-', "left-align-indels", false);
  
  //useBestNAlleles = 0;
  useBestNAlleles                       = RetrieveParameterInt   (opts, fb_params, 'm', "use-best-n-alleles", 2);
  onlyUseInputAlleles                   = RetrieveParameterBool  (opts, fb_params, '-', "use-input-allele-only", false);
  min_mapping_qv                        = RetrieveParameterInt   (opts, fb_params, 'M', "min-mapping-qv", 4);
  read_snp_limit                        = RetrieveParameterInt   (opts, fb_params, 'U', "read-snp-limit", 10);
  readMaxMismatchFraction               = RetrieveParameterDouble(opts, fb_params, 'z', "read-max-mismatch-fraction", 1.0);
  maxComplexGap                         = RetrieveParameterInt   (opts, fb_params, '!', "max-complex-gap", 1);
  // read from json or command line, otherwise default to snp frequency
  minAltFraction                        = RetrieveParameterDouble(opts, fb_params, '-', "gen-min-alt-allele-freq", my_controls.filter_snps.min_allele_freq);
  minCoverage                           = RetrieveParameterInt   (opts, fb_params, '-', "gen-min-coverage", my_controls.filter_snps.min_cov);
  minIndelAltFraction                   = RetrieveParameterDouble(opts, fb_params, '-', "gen-min-indel-alt-allele-freq", my_controls.filter_hp_indel.min_allele_freq);
  //set up debug levels

  if (program_flow.DEBUG > 0)
    debug = true;

  if (program_flow.inputPositionsOnly) {
    processInputPositionsOnly = true;
  }

  if (variantPriorsFile.empty() && (processInputPositionsOnly || onlyUseInputAlleles) ) {
    cerr << "ERROR: Parameter error - Process-input-positions-only: " << processInputPositionsOnly << " use-input-allele-only: " << onlyUseInputAlleles << " :  Specified without Input VCF File " << endl;
    exit(1);
  }
}

// ------------------------------------------------------------

void ExtendParameters::ParametersFromJSON(OptArgs &opts, Json::Value &tvc_params, Json::Value &freebayes_params, Json::Value &params_meta) {
  string parameters_file                = opts.GetFirstString('-', "parameters-file", "");
  Json::Value parameters_json(Json::objectValue);
  if (not parameters_file.empty()) {
    ifstream in(parameters_file.c_str(), ifstream::in);

    if (!in.good()) {
      fprintf(stderr, "[tvc] FATAL ERROR: cannot open %s\n", parameters_file.c_str());
      exit(-1);
    }
    
    // This line can cause seg-faults if we don't have a json file at the input
    in >> parameters_json;
    in.close();
    if (parameters_json.isMember("pluginconfig"))
      parameters_json = parameters_json["pluginconfig"];

    tvc_params = parameters_json.get("torrent_variant_caller", Json::objectValue);
    freebayes_params = parameters_json.get("freebayes", Json::objectValue);
    params_meta = parameters_json.get("meta", Json::objectValue);
  }
}

// ------------------------------------------------------------

void ExtendParameters::CheckParameterLimits() {
  // Check in the order they were set
  my_controls.CheckParameterLimits();
  my_eval_control.CheckParameterLimits();
  program_flow.CheckParameterLimits();

  // Checking FreeBayes parameters
  CheckParameterLowerUpperBound<int>  ("use-best-n-alleles",         useBestNAlleles,              0,    20);
  CheckParameterLowerBound<int>       ("min-mapping-qv",             min_mapping_qv,               0);
  CheckParameterLowerBound<int>       ("read-snp-limit",             read_snp_limit,               0);
  CheckParameterLowerUpperBound<float>("read-max-mismatch-fraction", readMaxMismatchFraction,      0.0f, 1.0f);

  CheckParameterLowerUpperBound<long double>("gen-min-alt-allele-freq",       minAltFraction,      0.0, 1.0);
  CheckParameterLowerBound<int>             ("gen-min-coverage",              minCoverage,         0);
  CheckParameterLowerUpperBound<long double>("gen-min-indel-alt-allele-freq", minIndelAltFraction, 0.0, 1.0);

}


// ------------------------------------------------------------

ExtendParameters::ExtendParameters(int argc, char** argv)
{
  // i/o parameters:
  fasta = "";                // -f --fasta-reference
  targets = "";              // -t --targets
  outputFile = "";

  // operation parameters
  trim_ampliseq_primers = false;
  useDuplicateReads = false;      // -E --use-duplicate-reads
  useBestNAlleles = 0;         // -n --use-best-n-alleles
  allowIndels = true;            // -i --no-indels
  allowMNPs = true;            // -X --no-mnps
  allowSNPs = true;          // -I --no-snps
  allowComplex = false;
  maxComplexGap = 3;
  onlyUseInputAlleles = false;
  min_mapping_qv = 0;                    // -m --min-mapping-quality
  readMaxMismatchFraction = 1.0;    //  -z --read-max-mismatch-fraction
  read_snp_limit = 10000000;       // -$ --read-snp-limit
  minAltFraction = 0.2;  // require 20% of reads from sample to be supporting the same alternate to consider
  minIndelAltFraction = 0.2;
  minAltCount = 2; // require 2 reads in same sample call
  minAltTotal = 1;
  minCoverage = 0;
  debug = false;
  multisample = false;

  processInputPositionsOnly = false;

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

  // enable floating point exceptions during program execution
  if (opts.GetFirstBoolean('-', "float-exceptions", true)) {
    cout << "TVC: Floating point exceptions enabled." << endl;
    feraiseexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  } //*/

  Json::Value tvc_params(Json::objectValue);
  Json::Value freebayes_params(Json::objectValue);
  Json::Value params_meta(Json::objectValue);
  ParametersFromJSON(opts, tvc_params, freebayes_params, params_meta);

  SetupFileIO(opts, tvc_params);

  my_controls.SetOpts(opts, tvc_params);
  my_eval_control.SetOpts(opts, tvc_params);
  program_flow.SetOpts(opts, tvc_params);
  tag_trimmer_parameters = MolecularTagTrimmer::ReadOpts(opts);

  // Preserve the data for all flows if we want to do rich diagnostic
  // Otherwise we only keep the data for test flows
  my_eval_control.preserve_full_data = program_flow.rich_json_diagnostic;

  // Dummy lines for HP recalibration
  recal_model_file_name = opts.GetFirstString ('-', "model-file", "");
  recalModelHPThres = opts.GetFirstInt('-', "recal-model-hp-thres", 4);

  prefixExclusion =  opts.GetFirstInt('-', "prefix-exclude", 6);
  cerr << "prefix-exclude = " <<  prefixExclusion << endl;

  SetFreeBayesParameters(opts, freebayes_params);
  bool overrideLimits          = RetrieveParameterBool  (opts, tvc_params, '-', "override-limits", false);

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


  //  opts.CheckNoLeftovers();
  // Sanity checks on input variables once all are set.
  if (!overrideLimits)
    CheckParameterLimits();

}

// check whether parameter contains only the characters in context
bool CheckParameterStringContext(string identifier, string &parameter, const string &contains_only, const string &default_value){
	bool is_ok = true; // for the case where parameter is an empty string. Will return true if parameter is empty.

	cout << "Check parameter " << identifier << ": "
			<< parameter << " contains only " << contains_only << "? ";

	for(unsigned int i = 0; i < parameter.size(); ++i){
		is_ok = false;
		for(unsigned int j = 0; j < contains_only.size(); ++j){
			if(parameter[i] == contains_only[j]){
				is_ok = true;
				break;
			}
		}
		if(not is_ok){
			break;
		}
	}
	if(is_ok){
		cout<< "OK!";
	}else{
		cout<< "The parameter "<< identifier
				<< " should only contain the characters in " << contains_only <<". "
				<< "Using the default value " << default_value <<" instead!";
	}
	cout << endl;
	return is_ok;
}


