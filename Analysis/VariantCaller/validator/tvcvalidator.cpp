/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <math.h>
#include <glob.h>
#include <map>

#include "OptArgs.h"
#include "IonVersion.h"

#include <Variant.h>

using namespace std;


class VariantCallerResults {
public:
  void load_vcf(string& input_vcf_filename) {
    vcf::VariantCallFile input_vcf;
    input_vcf.parseSamples = true;
    input_vcf.open(input_vcf_filename);
    vcf_header = input_vcf.header;
    variants.clear();
    variants.push_back(vcf::Variant(input_vcf));
    while (input_vcf.getNextVariant(variants.back()))
      variants.push_back(vcf::Variant(input_vcf));
    variants.pop_back();
  }

  string vcf_header;
  deque<vcf::Variant> variants;
};


class ValidatorTruth {
public:
  ValidatorTruth() {
    num_variants = 0;
  }

  string          truth_filename_;

  int             num_variants;
  vector<string>  chr;
  vector<long>    position1;
  vector<long>    position2;
  vector<string>  ref;
  vector<string>  alt1;
  vector<string>  alt2;


  void ReadTruthFile(const string& truth_filename)
  {
    truth_filename_ = truth_filename;

    ifstream truth_reader(truth_filename.c_str());
    if (!truth_reader.is_open()) {
      cout << "Error opening file: " << truth_filename << endl;
      return;
    }

    while (not truth_reader.eof()) {

      string line;
      getline(truth_reader, line);
      if (line.empty() or line[0] == '#')
        continue;

      istringstream linestream(line);
      string token;
      string ref_allele;
      string alt_allele;

      getline(linestream, token, '\t');
      chr.push_back(token);

      getline(linestream, token, '\t');
      position1.push_back(atoi(token.c_str()));

      getline(linestream, token, '\t');
      position2.push_back(atoi(token.c_str()));

      getline(linestream, ref_allele, '\t');

      getline(linestream, alt_allele, '\t');

      ref.push_back(ref_allele);

      // Parsing alt_allele

      size_t found = alt_allele.find_first_of("/", 0);
      if (found == string::npos) {   //  - if no '/', it is homozygous (empty alt2)
        alt1.push_back(alt_allele);
        alt2.push_back("");

      } else {                       //  - if '/' present, it is heterozygous (alt1, alt2).
        alt1.push_back(alt_allele.substr(0, found));
        alt2.push_back(alt_allele.substr(found+1));
      }

      num_variants++;
    }
  }



  void CompareToCalls(VariantCallerResults& results_vcf)
  {
    vector<int>  map_call_to_truth(results_vcf.variants.size(), -1);
    vector<int>  map_truth_to_call(num_variants, -1);
    vector<bool> truth_perfect_match(num_variants, false);
    vector<bool> call_perfect_match(results_vcf.variants.size(), false);

    // Iterate over calls and match them to the truth table

    for (int truth_idx = 0; truth_idx < num_variants; ++truth_idx) {

      int best_distance = 10;

      for (int call_idx = 0; call_idx < (int)results_vcf.variants.size(); ++call_idx) {

        if (call_perfect_match[call_idx])
          continue;

        vcf::Variant& variant = results_vcf.variants[call_idx];

        string& genotype = variant.samples.begin()->second["GT"][0];

        if (variant.sequenceName != chr[truth_idx])
          continue;

        variant.getGenotypeIndexesDiploid();


        if (variant.position == position2[truth_idx] and
            variant.ref == ref[truth_idx] and
            variant.alt[0] == alt1[truth_idx] and
            alt2[truth_idx].empty() and
            genotype == "1/1") {
          map_truth_to_call[truth_idx] = call_idx;
          truth_perfect_match[truth_idx] = true;
          call_perfect_match[call_idx] = true;
          map_call_to_truth[call_idx] = truth_idx;
          break;
        }

        if (variant.position == position2[truth_idx] and
            variant.ref == ref[truth_idx] and
            variant.alt[0] == alt1[truth_idx] and
            variant.ref == alt2[truth_idx] and
            genotype == "0/1") {
          map_truth_to_call[truth_idx] = call_idx;
          truth_perfect_match[truth_idx] = true;
          call_perfect_match[call_idx] = true;
          map_call_to_truth[call_idx] = truth_idx;
          break;
        }

        if (variant.position == position2[truth_idx] and
            variant.ref == ref[truth_idx] and
            variant.alt[0] == alt2[truth_idx] and
            variant.ref == alt1[truth_idx] and
            genotype == "0/1") {
          map_truth_to_call[truth_idx] = call_idx;
          truth_perfect_match[truth_idx] = true;
          call_perfect_match[call_idx] = true;
          map_call_to_truth[call_idx] = truth_idx;
          break;
        }



        int distance = abs(variant.position - position2[truth_idx]);
        if (distance < best_distance) {
          map_truth_to_call[truth_idx] = call_idx;
          map_call_to_truth[call_idx] = truth_idx;
          best_distance = distance;
        }
      }



      /*
      printf("Truth: %s:%d ref=%s alt1=%s alt2=%s  ",
          chr[truth_idx].c_str(), (int)position2[truth_idx], ref[truth_idx].c_str(),
          alt1[truth_idx].c_str(), alt2[truth_idx].c_str());

      if (map_truth_to_call[truth_idx] >= 0 and truth_perfect_match[truth_idx])
        printf("- perfect match\n");

      if (map_truth_to_call[truth_idx] >= 0 and not truth_perfect_match[truth_idx]) {
        int call_idx = map_truth_to_call[truth_idx];
        vcf::Variant& variant = results_vcf.variants[call_idx];
        string& genotype = variant.samples.begin()->second["GT"][0];
        printf("- approximate match p=%d ref=%s alt=%s gen=%s\n",
            (int)variant.position, variant.ref.c_str(), variant.alt[0].c_str(), genotype.c_str());
      }

      if (map_truth_to_call[truth_idx] < 0)
        printf("- X\n");
      */
    }

    int false_positives_hard = 0;
    int false_positives_soft = 0;
    int false_negatives_hard = 0;
    int false_negatives_soft = 0;
    int true_positives = 0;

    for (int truth_idx = 0; truth_idx < num_variants; ++truth_idx) {
      if (map_truth_to_call[truth_idx] >= 0 and truth_perfect_match[truth_idx])
        true_positives++;
      if (map_truth_to_call[truth_idx] >= 0 and not truth_perfect_match[truth_idx])
        false_negatives_soft++;
      if (map_truth_to_call[truth_idx] < 0)
        false_negatives_hard++;
    }

    for (int call_idx = 0; call_idx < (int)results_vcf.variants.size(); ++call_idx) {
      if (map_call_to_truth[call_idx] >= 0 and not call_perfect_match[call_idx])
        false_positives_soft++;
      if (map_call_to_truth[call_idx] < 0)
        false_positives_hard++;
    }

    printf("Strict: TP =% 6d FP =% 6d FN =% 6d ; Weak: TP =% 6d FP =% 6d FN =% 6d ; File %s (%d)\n",
        true_positives, false_positives_hard+false_positives_soft, false_negatives_hard+false_negatives_soft,
        true_positives+false_positives_soft, false_positives_hard, false_negatives_hard,
        truth_filename_.c_str(), num_variants);

    /*
    // Variant type: SNP
    if (variant.ref.length() == variant.alt[0].length()) {
    }

    // Variant type: Insertion
    if (variant.ref.length() < variant.alt[0].length()) {
    }

    // Variant type: Deletion
    if (variant.ref.length() > variant.alt[0].length()) {
    }
    */
  }


};








void VariantValidatorHelp()
{
  printf ("Usage: tvcvalidator [options]\n");
  printf ("\n");
  printf ("General options:\n");
  printf ("  -h,--help                                    print this help message and exit\n");
  printf ("  -v,--version                                 print version and exit\n");
  printf ("  -c,--input-vcf                   FILE        vcf file with candidate variant locations [required]\n");
  printf ("  -t,--truth-file                  FILE        file with expected variants in bed or vcf format\n");
  printf ("  -d,--truth-dir                   DIRECTORY   location of multiple truth bed files [/results/plugins/validateVariantCaller/files]\n");
  printf ("\n");
}


int main(int argc, const char* argv[])
{
  printf ("tvcvalidator %s-%s (%s) - Prototype tvc validation tool\n\n",
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetSvnRev().c_str());

  if (argc == 1) {
    VariantValidatorHelp();
    return 1;
  }

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);

  if (opts.GetFirstBoolean('v', "version", false)) {
    return 0;
  }
  if (opts.GetFirstBoolean('h', "help", false)) {
    VariantValidatorHelp();
    return 0;
  }

  string input_vcf_filename = opts.GetFirstString ('i', "input-vcf", "");
  string truth_filename = opts.GetFirstString ('t', "truth-file", "");
  string truth_dir = opts.GetFirstString ('d', "truth-dir", "/results/plugins/validateVariantCaller/files");

  // TODO: reference optional, only used to verify reference allele in input-vcf and truth files
  //string reference_filename = opts.GetFirstString ('r', "reference", "");

  opts.CheckNoLeftovers();


  //
  // Step 1. Load input VCF file into memory
  //

  if (input_vcf_filename.empty()) {
    VariantValidatorHelp();
    cerr << "ERROR: Input VCF file not specified " << endl;
    return 1;
  }

  VariantCallerResults results_vcf;
  results_vcf.load_vcf(input_vcf_filename);
  printf("Loaded VCF %s with %d variant calls\n", input_vcf_filename.c_str(), (int)results_vcf.variants.size());



  //
  // Step 2. Parse truth files, compare them to the input vcf, and compute match scores
  //

  if (not truth_filename.empty()) {
    ValidatorTruth truth;
    truth.ReadTruthFile(truth_filename);
    truth.CompareToCalls(results_vcf);
    return 0;
  }

  truth_dir += "/*.bed";
  glob_t glob_result;
  glob(truth_dir.c_str(), GLOB_TILDE, NULL, &glob_result);
  for(unsigned int i = 0; i < glob_result.gl_pathc; ++i) {

    ValidatorTruth truth;
    truth.ReadTruthFile(string(glob_result.gl_pathv[i]));
    truth.CompareToCalls(results_vcf);

  }
  globfree(&glob_result);


  return 0;
}





