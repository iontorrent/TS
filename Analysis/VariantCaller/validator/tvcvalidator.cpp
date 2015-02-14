/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <math.h>
#include <glob.h>
#include <map>
#include <set>
#include <algorithm>
#include <iomanip>


#include "OptArgs.h"
#include "IonVersion.h"

#include <Variant.h>
#include "ReferenceReader.h"

using namespace std;


// TODO list
//  - Accept BED file to limit scope
//  - Accept filtered VCF for extra reporting
//  - Generate detailed, per-variant report in excel friendly format
//    - Basic info for strict TP
//    - Detailed info for weak TP, also check for strict/weak matches in filtered VCF
//    - Detailed info for FN, also check for strict/weak matches in filtered VCF
//    - Detailed info for FP
//
//  - Support SNP in MNP matching
//  - Support suboptimal allele representations



struct ValidatorTarget {
  int         chr;
  int         begin;
  int         end;
};

void PopulateValidatorTarget(vector<ValidatorTarget>& targets, const ReferenceReader& ref_reader,
    const string& targets_bed_filename)
{
  if (targets_bed_filename.empty()) {
    for (int chr = 0; chr < ref_reader.chr_count(); ++chr) {
      targets.push_back(ValidatorTarget());
      ValidatorTarget& target = targets.back();
      target.begin = 0;
      target.end = ref_reader.chr_size(chr);
      target.chr = chr;
    }
    return;
  }

  FILE *bed_file = fopen(targets_bed_filename.c_str(), "r");
  if (not bed_file) {
    cerr << "ERROR: Unable to open target file " << targets_bed_filename << " : " << strerror(errno) << endl;
    exit(1);
  }

  char line[4096];
  char chr_name[4096];
  int begin;
  int end;
  char region_name[4096];
  int line_number = 0;

  while (fgets(line, 4096, bed_file)) {
    ++line_number;

    if (strncmp(line,"track",5) == 0)
      continue;

    int num_fields = sscanf(line, "%s\t%d\t%d", chr_name, &begin, &end);
    if (num_fields == 0)
      continue;
    if (num_fields < 3) {
      cerr << "ERROR: Failed to parse target file line " << line_number << endl;
      exit(1);
    }

    targets.push_back(ValidatorTarget());
    ValidatorTarget& target = targets.back();
    target.begin = begin;
    target.end = end;
    target.chr = ref_reader.chr_idx(chr_name);

    if (target.chr < 0) {
      cerr << "ERROR: Target region " << " (" << chr_name << ":" << begin << "-" << end << ")"
           << " has unrecognized chromosome name" << endl;
      exit(1);
    }

    if (begin < 0 || end > ref_reader.chr_size(target.chr)) {
      cerr << "ERROR: Target region " << " (" << chr_name << ":" << begin << "-" << end << ")"
           << " is outside of reference sequence bounds ("
           << chr_name << ":0-" << ref_reader.chr_size(target.chr) << ")" << endl;
      exit(1);
    }
    if (end < begin) {
      cerr << "ERROR: Target region " << " (" << chr_name << ":" << begin << "-" << end << ")"
           << " has inverted coordinates" << endl;
      exit(1);
    }
  }

  fclose(bed_file);

  if (targets.empty()) {
    cerr << "ERROR: No targets loaded from " << targets_bed_filename
         << " after parsing " << line_number << " lines" << endl;
    exit(1);
  }
  printf ("Loaded target file %s with %d targets\n", targets_bed_filename.c_str(), (int)targets.size());
}





struct TruthSetAllele;

struct VariantCallerAllele {
  VariantCallerAllele() : chr(-1), pos(-1), alt_idx(-1), call(-1), match(NULL), match_status(0) {}
  int               chr;
  int               pos;
  string            ref;
  string            alt;
  int               alt_idx;
  int               call;    // 0-absent, 1-het, 2-homo
  string            fr;
  vcf::Variant      raw;

  TruthSetAllele*   match;
  int               match_status;     // 0-unmatched, 1-approx, 2-allele match, 3-allele & genotype match
};


void LoadVariantCallerResults(deque<VariantCallerAllele>& variants, string& input_vcf_filename,
    const ReferenceReader& ref_reader, bool only_called)
{
  vcf::VariantCallFile input_vcf;
  input_vcf.parseSamples = true;
  input_vcf.open(input_vcf_filename);
  variants.clear();
  vcf::Variant input_variant(input_vcf);
  while (input_vcf.getNextVariant(input_variant)) {

    int chr = ref_reader.chr_idx(input_variant.sequenceName.c_str());
    if (chr < 0)
      continue;
    char g1 = '.';
    char g2 = '.';
    string& genotype = input_variant.samples.begin()->second["GT"][0];
    if (genotype.size() == 3) {
      g1 = genotype[0];
      g2 = genotype[2];
    }

    for (int i = 0; i < (int)input_variant.alt.size(); ++i) {
      variants.push_back(VariantCallerAllele());
      variants.back().chr = chr;
      variants.back().pos = input_variant.position-1;
      variants.back().ref = input_variant.ref;
      variants.back().alt = input_variant.alt[i];
      variants.back().alt_idx = i;
      variants.back().call = 0;
      if (i+'1' == g1)
        variants.back().call++;
      if (i+'1' == g2)
        variants.back().call++;
      if (variants.back().call == 0 and only_called) {
        variants.pop_back();
        continue;
      }
      map<string,vector<string> >:: iterator I = input_variant.info.find("FR");
      if (I != input_variant.info.end()) {
        for (int j = 0; j < (int)I->second.size(); ++j) {
          if (j) variants.back().fr += ",";
          variants.back().fr += I->second[j];
        }
      }
      variants.back().raw = input_variant;
    }
  }

  printf ("Loaded variant file %s (total %d)\n", input_vcf_filename.c_str(), (int)variants.size());
}





struct TruthSetAllele {
  TruthSetAllele() : chr(-1), pos(-1), expected_call(0), match(NULL), match_status(0) {}

  // Allele info and expected call
  int           chr;
  int           pos;
  string        ref;
  string        alt;
  int           expected_call;    // 0-absent, 1-het, 2-homo

  // Annotations
  set<int>    annotations;

  // Matching results
  VariantCallerAllele *match;        // filtered or unfiltered vcf record
  int match_status;           // 0 - no match, 1 - filtered match, 2 - weak match,
                              // 3 - position match, 4 - position+allele match, 5 - position+allele+genotype match
};


// Comparator
bool CompareAlleles(const TruthSetAllele* a, const TruthSetAllele* b)
{
  if (a->chr < b->chr)
    return true;
  if (a->chr > b->chr)
    return false;
  if (a->pos < b->pos)
    return true;
  if (a->pos > b->pos)
    return false;
  if (a->ref.size() < b->ref.size())
    return true;
  if (a->ref.size() > b->ref.size())
    return false;
  return a->alt < b->alt;
}


class ValidatorTruth {
public:
  ValidatorTruth() : ref_reader_(NULL), targets_(NULL) {}

  const ReferenceReader*          ref_reader_;
  const vector<ValidatorTarget>*  targets_;
  vector<TruthSetAllele>          truth;
  vector<string>                  annotation_names;


  void Initialize(const ReferenceReader& ref_reader, const vector<ValidatorTarget>& targets)
  {
    ref_reader_ = &ref_reader;
    targets_ = &targets;
  }


  int TargetCheckHelper(int chr, int pos, int recent_target)
  {
    if (chr < (*targets_)[recent_target].chr)
      return -1;
    if (chr > (*targets_)[recent_target].chr)
      return 1;
    if (pos < (*targets_)[recent_target].begin)
      return -1;
    if (pos >= (*targets_)[recent_target].end)
      return 1;
    return 0;
  }

  bool IsWithinTarget(int chr, int pos, int& recent_target)
  {
    int first_direction = TargetCheckHelper(chr, pos, recent_target);
    if (first_direction == 0)
      return true;
    while (true) {
      recent_target += first_direction;
      if (recent_target < 0 or recent_target >= (int)targets_->size()) {
        recent_target -= first_direction;
        return false;
      }
      int direction = TargetCheckHelper(chr, pos, recent_target);
      if (direction == 0)
        return true;
      if (direction != first_direction)
        return false;
    }
  }


  void AddTruthFile(const string& truth_filename)
  {
    int recent_target = 0;

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

      getline(linestream, token, '\t');
      int chr = ref_reader_->chr_idx(token.c_str());
      assert (chr >= 0);

      getline(linestream, token, '\t');
      int pos = atoi(token.c_str());

      if (not IsWithinTarget(chr, pos, recent_target))
        continue;

      getline(linestream, token, '\t');

      string ref_allele;
      getline(linestream, ref_allele, '\t');

      // Parsing alt_allele
      string alt_allele;
      getline(linestream, alt_allele, '\t');

      size_t found = alt_allele.find_first_of("/", 0);
      if (found == string::npos) {   //  - if no '/', it is homozygous (empty alt2)

        TruthSetAllele allele;
        allele.chr = chr;
        allele.pos = pos;
        allele.ref = ref_allele;
        allele.alt = alt_allele;
        allele.expected_call = 2;
        truth.push_back(allele);

      } else {                       //  - if '/' present, it is heterozygous (alt1, alt2).
        string alt;
        alt = alt_allele.substr(0, found);
        if (alt != ref_allele) {
          TruthSetAllele allele;
          allele.chr = chr;
          allele.pos = pos;
          allele.ref = ref_allele;
          allele.alt = alt;
          allele.expected_call = 1;
          truth.push_back(allele);
        }
        alt = alt_allele.substr(found+1);
        if (alt != ref_allele) {
          TruthSetAllele allele;
          allele.chr = chr;
          allele.pos = pos;
          allele.ref = ref_allele;
          allele.alt = alt;
          allele.expected_call = 1;
          truth.push_back(allele);
        }
      }
    }
    printf ("Loaded truth set %s (total %d)\n", truth_filename.c_str(), (int)truth.size());
  }




  void AddAnnotationFile(const string& annotation_filename)
  {
    int recent_target = 0;

    ifstream annotation_reader(annotation_filename.c_str());
    if (!annotation_reader.is_open()) {
      cout << "Error opening file: " << annotation_filename << endl;
      return;
    }

    while (not annotation_reader.eof()) {

      string line;
      getline(annotation_reader, line);
      if (line.empty() or line[0] == '#')
        continue;

      istringstream linestream(line);
      TruthSetAllele allele;
      string token;
      char chr_buffer[256], ref_buffer[256],alt_buffer[256],annotation_buffer[4096];

      if (6 != sscanf(line.c_str(), "%s\t%d\t%s\t%s\t%d\t%s", chr_buffer, &allele.pos, ref_buffer, alt_buffer, &allele.expected_call, annotation_buffer)) {
        cerr << "Misformatted annotation line: " << line << endl;
        continue;
      }
      allele.chr = ref_reader_->chr_idx(chr_buffer);
      assert (allele.chr >= 0);
      allele.pos -= 1;
      if (not IsWithinTarget(allele.chr, allele.pos, recent_target))
        continue;
      allele.ref = ref_buffer;
      allele.alt = alt_buffer;

      istringstream linestream2(annotation_buffer);
      while (true) {
        token.clear();
        getline(linestream2, token, ',');
        if (token.empty())
          break;
        int annotation_idx = -1;
        for (int idx = 0; idx < (int)annotation_names.size(); ++idx) {
          if (annotation_names[idx] == token) {
            annotation_idx = idx;
            break;
          }
        }
        if (annotation_idx == -1) {
          annotation_idx = annotation_names.size();
          annotation_names.push_back(token);
        }
        allele.annotations.insert(annotation_idx);
      }

      truth.push_back(allele);
    }
    printf ("Loaded annotations %s\n", annotation_filename.c_str());
  }


  void CombineAndSort()
  {
    vector<TruthSetAllele*> sorted_truth;
    sorted_truth.reserve(truth.size());
    for (unsigned int idx = 0; idx < truth.size(); ++idx)
      sorted_truth.push_back(&truth[idx]);
    sort(sorted_truth.begin(), sorted_truth.end(), CompareAlleles);

    vector<TruthSetAllele> new_truth;
    new_truth.reserve(truth.size());
    for (unsigned int idx = 0; idx < truth.size(); ++idx) {
      TruthSetAllele& allele = *sorted_truth[idx];
      if (idx and new_truth.back().chr == allele.chr and new_truth.back().pos == allele.pos
          and new_truth.back().ref == allele.ref and new_truth.back().alt == allele.alt) {
        if (new_truth.back().expected_call == -1)
          new_truth.back().expected_call = allele.expected_call;
        else if (new_truth.back().expected_call != allele.expected_call) {
          cout << "WARNING: Inconsistent expected call for " << ref_reader_->chr_str(allele.chr) << ":" << allele.pos+1
              << "  - " << new_truth.back().expected_call << " vs " << allele.expected_call << endl;
        }
        new_truth.back().annotations.insert(allele.annotations.begin(),allele.annotations.end());
        continue;
      }
      new_truth.push_back(allele);
    }

    truth.swap(new_truth);
  }



  int AdvancedAlleleComparator(const TruthSetAllele& truth, const VariantCallerAllele& call)
  {

    if (truth.chr != call.chr)
      return 0;

    if (truth.pos == call.pos and truth.ref == call.ref and truth.alt == call.alt)
      return 3;


    //
    // Check if truth is a snp contained within mnp call
    //

    if (truth.ref.length() == 1 and truth.alt.length() == 1 and call.ref.length() == call.alt.length()) {
      int relative_pos = truth.pos - call.pos;
      if (relative_pos >= 0 and relative_pos < (int)call.alt.length() and call.alt[relative_pos] == truth.alt[0])
        return 2;
    }




    //
    // Minimize and left-align both alleles, then compare
    //


    // Minimize and left-align truth

    int truth_pos = truth.pos;
    int truth_ref_length = truth.ref.length();
    deque<char> truth_alt(truth.alt.begin(),truth.alt.end());

    ReferenceReader::iterator ref_back = ref_reader_->iter(truth.chr,truth_pos+truth_ref_length-1);
    while (not truth_alt.empty() and truth_ref_length and *ref_back == truth_alt.back()) {
      --truth_ref_length;
      --ref_back;
      truth_alt.pop_back();
    }

    ReferenceReader::iterator ref_front = ref_reader_->iter(truth.chr,truth_pos);
    while (not truth_alt.empty() and truth_ref_length and *ref_front == truth_alt.front()) {
      --truth_ref_length;
      ++ref_front;
      ++truth_pos;
      truth_alt.pop_front();
    }

    if (truth_ref_length == 0) {  // Attempt insertion left alignment
      --ref_front;
      while (truth_pos and *ref_front == truth_alt.back()) {
        truth_alt.push_front(truth_alt.back());
        truth_alt.pop_back();
        --ref_front;
        --truth_pos;
      }
    }

    if (truth_alt.empty()) {  // Attempt deletion left alignment
      --ref_front;
      while (truth_pos and *ref_front == *ref_back) {
        --ref_front;
        --ref_back;
        --truth_pos;
      }
    }

    // Minimize and left align call

    int call_pos = call.pos;
    int call_ref_length = call.ref.length();
    deque<char> call_alt(call.alt.begin(),call.alt.end());

    ref_back = ref_reader_->iter(call.chr,call_pos+call_ref_length-1);
    while (not call_alt.empty() and call_ref_length and *ref_back == call_alt.back()) {
      --call_ref_length;
      --ref_back;
      call_alt.pop_back();
    }

    ref_front = ref_reader_->iter(call.chr,call_pos);
    while (not call_alt.empty() and call_ref_length and *ref_front == call_alt.front()) {
      --call_ref_length;
      ++ref_front;
      ++call_pos;
      call_alt.pop_front();
    }

    if (call_ref_length == 0) {  // Attempt insertion left alignment
      if (call_alt.empty()) {
        cerr << "ERROR: Left alignment failure for call: " << ref_reader_->chr_str(call.chr) << ":" << call.pos+1
            << " " << call.ref << "/" << call.alt << endl;
        exit(1);


      }
      --ref_front;
      while (call_pos and *ref_front == call_alt.back()) {
        call_alt.push_front(call_alt.back());
        call_alt.pop_back();
        --ref_front;
        --call_pos;
      }
    }

    if (call_alt.empty()) {  // Attempt deletion left alignment
      --ref_front;
      while (call_pos and *ref_front == *ref_back) {
        --ref_front;
        --ref_back;
        --call_pos;
      }
    }

    if (truth_pos == call_pos and truth_ref_length == call_ref_length and truth_alt == call_alt)
      return 1;

    return 0;
  }



  void CompareToCalls(deque<VariantCallerAllele>& results_vcf, deque<VariantCallerAllele>& filtered_vcf)
  {

    deque<VariantCallerAllele>::iterator filtered_window = filtered_vcf.begin();
    deque<VariantCallerAllele>::iterator results_window = results_vcf.begin();

    // Iterate over calls and match them to the truth table

    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {

      TruthSetAllele& allele = truth[truth_idx];

      // Advance search window starts

      for (; filtered_window != filtered_vcf.end(); ++filtered_window) {
        if (filtered_window->chr > allele.chr)
          break;
        if (filtered_window->chr == allele.chr and filtered_window->pos > allele.pos-22)
          break;
      }

      for (; results_window != results_vcf.end(); ++results_window) {
        if (results_window->chr > allele.chr)
          break;
        if (results_window->chr == allele.chr and results_window->pos > allele.pos-22)
          break;
      }

      int best_distance = 10;

      // Match to filtered entries

      for (deque<VariantCallerAllele>::iterator filtered = filtered_window; filtered != filtered_vcf.end(); ++filtered) {
        if (filtered->chr > allele.chr)
          break;
        if (filtered->chr == allele.chr and filtered->pos > allele.pos+22)
          break;

        int distance = abs(filtered->pos - allele.pos);
        if (distance < best_distance) {
          allele.match = &(*filtered);
          allele.match_status = 1;
          best_distance = distance;
        }
      }

      // Match to called entries

      for (deque<VariantCallerAllele>::iterator results = results_window; results != results_vcf.end(); ++results) {
        if (results->chr > allele.chr)
          break;
        if (results->chr == allele.chr and results->pos > allele.pos+22)
          break;

        //if (results->match_status >= 4)
        //  continue;


        if (AdvancedAlleleComparator(allele,*results) > 0) {
          allele.match = &(*results);
          results->match = &allele;
          allele.match_status = 4;
          results->match_status = 4;
          if (results->call == allele.expected_call) {
            allele.match_status = 5;
            results->match_status = 5;
          }
          break;
        }

        if (best_distance == 0 or allele.expected_call == 0)
          continue;
        int distance = abs(results->pos - allele.pos);

        if (distance == 0) {
          allele.match = &(*results);
          allele.match_status = 3;
          if (results->match_status < 3) {
            results->match_status = 3;
            results->match = &allele;
          }
          best_distance = distance;
        }

        if (distance < best_distance) {
          allele.match = &(*results);
          allele.match_status = 2;
          if (results->match_status < 2) {
            results->match_status = 2;
            results->match = &allele;
          }
          best_distance = distance;
        }
      }

    }


    //
    // Calculate matching statistics
    //

    vector<int> match_status_count(6,0);

    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      if (truth[truth_idx].expected_call > 0)
        match_status_count[truth[truth_idx].match_status]++;
    }

    printf ("Truth set summary:\n");
    printf (" - [5] Perfect match                  :% 6d\n", match_status_count[5]);
    printf (" - [4] Match allele, but not genotype :% 6d\n", match_status_count[4]);
    printf (" - [3] Match position but not allele  :% 6d\n", match_status_count[3]);
    printf (" - [2] Approximate position           :% 6d\n", match_status_count[2]);
    printf (" - [1] Match to filtered record       :% 6d\n", match_status_count[1]);
    printf (" - [0] No match                       :% 6d\n", match_status_count[0]);
    printf("\n");

    cout << "New false negatives:" << endl;

    // Dump [4] entries
    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      TruthSetAllele& allele = truth[truth_idx];
      if (allele.expected_call == 0)
        continue;
      if (allele.match_status != 4)
        continue;
      if (not allele.annotations.empty())
        continue;
      printf("[4] Truth %s:%d %s>%s  Expected#: %d Called: %d", ref_reader_->chr(allele.chr), allele.pos+1,
          allele.ref.c_str(), allele.alt.c_str(), allele.expected_call, allele.match->call);
      printf("  Annotations:");
      for (set<int>::iterator I = allele.annotations.begin(); I != allele.annotations.end(); ++I) {
        if (I != allele.annotations.begin())
          printf (",");
        printf("%s", annotation_names[*I].c_str());
      }
      printf("\n");
    }

    // Dump [3] entries
    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      TruthSetAllele& allele = truth[truth_idx];
      if (allele.expected_call == 0)
        continue;
      if (allele.match_status != 3)
        continue;
      if (not allele.annotations.empty())
        continue;
      printf("[3] Truth %s:%d %s>%s  Expected#: %d  Found %s:%d %s>", ref_reader_->chr(allele.chr), allele.pos+1,
          allele.ref.c_str(), allele.alt.c_str(), allele.expected_call, ref_reader_->chr(allele.match->chr), (int)allele.match->pos+1, allele.match->ref.c_str());
      for (int idx = 0; idx < (int)allele.match->raw.alt.size(); ++idx) {
        if (idx)
          printf (",");
        printf("%s", allele.match->raw.alt[idx].c_str());
      }
      printf(" with call %s", allele.match->raw.samples.begin()->second["GT"][0].c_str());
      printf("  Annotations:");
      for (set<int>::iterator I = allele.annotations.begin(); I != allele.annotations.end(); ++I) {
        if (I != allele.annotations.begin())
          printf (",");
        printf("%s", annotation_names[*I].c_str());
      }
      printf("\n");
    }

    // Dump [2] entries
    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      TruthSetAllele& allele = truth[truth_idx];
      if (allele.expected_call == 0)
        continue;
      if (allele.match_status != 2)
        continue;
      if (not allele.annotations.empty())
        continue;
      printf("[2] Truth %s:%d %s>%s  Expected#: %d  Found %s:%d %s>", ref_reader_->chr(allele.chr), allele.pos+1,
          allele.ref.c_str(), allele.alt.c_str(), allele.expected_call, allele.match->raw.sequenceName.c_str(), (int)allele.match->raw.position, allele.match->raw.ref.c_str());
      for (int idx = 0; idx < (int)allele.match->raw.alt.size(); ++idx) {
        if (idx)
          printf (",");
        printf("%s", allele.match->raw.alt[idx].c_str());
      }
      printf(" with call %s", allele.match->raw.samples.begin()->second["GT"][0].c_str());
      printf("  Annotations:");
      for (set<int>::iterator I = allele.annotations.begin(); I != allele.annotations.end(); ++I) {
        if (I != allele.annotations.begin())
          printf (",");
        printf("%s", annotation_names[*I].c_str());
      }
      printf("\n");
    }

    // Dump [1] entries
    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      TruthSetAllele& allele = truth[truth_idx];
      if (allele.expected_call == 0)
        continue;
      if (allele.match_status != 1)
        continue;
      if (not allele.annotations.empty())
        continue;
      printf("[1] Truth %s:%d %s>%s  Expected#: %d  Found filtererd %s:%d %s>", ref_reader_->chr(allele.chr), allele.pos+1,
          allele.ref.c_str(), allele.alt.c_str(), allele.expected_call, allele.match->raw.sequenceName.c_str(), (int)allele.match->raw.position, allele.match->raw.ref.c_str());
      for (int idx = 0; idx < (int)allele.match->raw.alt.size(); ++idx) {
        if (idx)
          printf (",");
        printf("%s", allele.match->raw.alt[idx].c_str());
      }
      printf("  Annotations:");
      for (set<int>::iterator I = allele.annotations.begin(); I != allele.annotations.end(); ++I) {
        if (I != allele.annotations.begin())
          printf (",");
        printf("%s", annotation_names[*I].c_str());
      }
      printf("\n");
    }

    // Dump [0] entries
    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      TruthSetAllele& allele = truth[truth_idx];
      if (allele.expected_call == 0)
        continue;
      if (allele.match_status != 0)
        continue;
      if (not allele.annotations.empty())
        continue;
      printf("[0] Truth %s:%d %s>%s  Expected#: %d  No match", ref_reader_->chr(allele.chr), allele.pos+1,
          allele.ref.c_str(), allele.alt.c_str(), allele.expected_call);
      printf("  Annotations:");
      for (set<int>::iterator I = allele.annotations.begin(); I != allele.annotations.end(); ++I) {
        if (I != allele.annotations.begin())
          printf (",");
        printf("%s", annotation_names[*I].c_str());
      }
      printf("\n");
    }



    // False positive report

    vector<int> call_status_count(6,0);
    for (deque<VariantCallerAllele>::iterator results = results_vcf.begin(); results != results_vcf.end(); ++results) {
      call_status_count[results->match_status]++;
    }

    printf ("Variant Calls summary:\n");
    printf (" - [5] Perfect match                  :% 6d\n", call_status_count[5]);
    printf (" - [4] Match allele, but not genotype :% 6d\n", call_status_count[4]);
    printf (" - [3] Match position but not allele  :% 6d\n", call_status_count[3]);
    printf (" - [2] Approximate position           :% 6d\n", call_status_count[2]);
    printf (" - [0] No match                       :% 6d\n", call_status_count[0]);
    printf("\n");

    cout << "New false positives:" << endl;

    for (deque<VariantCallerAllele>::iterator results = results_vcf.begin(); results != results_vcf.end(); ++results) {
      /*
      if (results->match_status >= 1 and results->match->expected_call == 0) {
        cout << ref_reader_->chr(results->chr) << ":" << results->pos+1 << " - ";
        printf("  Annotations:");
        for (set<int>::iterator I = results->match->annotations.begin(); I != results->match->annotations.end(); ++I) {
          if (I != results->match->annotations.begin())
            cout << ",";
          cout << annotation_names[*I];
        }
        cout << endl;
      }
      */
      if (results->match_status >= 4)
        continue;
      cout << ref_reader_->chr(results->chr) << ":" << results->pos+1 << " - " << results->raw << endl;
    }
    cout << endl;

    //////


    // Dump category-specific stats
    vector<vector<int> > annotation_stats(annotation_names.size()+1, vector<int>(6,0));
    int truth_positive_total = 0;
    int truth_positive_correct = 0;
    int truth_positive_partial = 0;

    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      TruthSetAllele& allele = truth[truth_idx];
      if (allele.expected_call == 0)
        continue;
      truth_positive_total++;
      if (allele.match_status == 5)
        truth_positive_correct++;
      if (allele.match_status == 4)
        truth_positive_partial++;
      for (set<int>::iterator I = allele.annotations.begin(); I != allele.annotations.end(); ++I)
        annotation_stats[*I][allele.match_status]++;
      if (allele.annotations.empty())
        annotation_stats.back()[allele.match_status]++;
    }

    cout << setw(30) << "Truth set annotations";
    cout << setw(10) << "FN";
    cout << setw(10) << "Partial";
    cout << setw(10) << "TP";
    cout << setw(10) << "Total" << endl;

    for (int idx = 0; idx < (int)annotation_names.size(); ++idx) {
      int total = annotation_stats[idx][0]+annotation_stats[idx][1]+annotation_stats[idx][2]+annotation_stats[idx][3]+annotation_stats[idx][4] + annotation_stats[idx][5];
      if (total == 0)
        continue;
      cout << setw(30) << annotation_names[idx];
      cout << setw(10) << total - annotation_stats[idx][5] - annotation_stats[idx][4];
      cout << setw(10) << annotation_stats[idx][4];
      cout << setw(10) << annotation_stats[idx][5];
      cout << setw(10) << total  << endl;
    }
    cout << "             -------------------------------------------------" << endl;
    int idx = annotation_names.size();
    cout << setw(30) << "With annotation";
    cout << setw(10) << truth_positive_total-truth_positive_correct-truth_positive_partial - (annotation_stats[idx][0]+annotation_stats[idx][1]+annotation_stats[idx][2]+annotation_stats[idx][3]);
    cout << setw(10) << truth_positive_partial - annotation_stats[idx][4];
    cout << setw(10) << truth_positive_correct - annotation_stats[idx][5];
    cout << setw(10) << truth_positive_total - (annotation_stats[idx][0]+annotation_stats[idx][1]+annotation_stats[idx][2]+annotation_stats[idx][3]+annotation_stats[idx][4]+annotation_stats[idx][5]);
    cout << endl;

    cout << setw(30) << "Without annotation";
    cout << setw(10) << annotation_stats[idx][0]+annotation_stats[idx][1]+annotation_stats[idx][2]+annotation_stats[idx][3];
    cout << setw(10) << annotation_stats[idx][4];
    cout << setw(10) << annotation_stats[idx][5];
    cout << setw(10) << annotation_stats[idx][0]+annotation_stats[idx][1]+annotation_stats[idx][2]+annotation_stats[idx][3]+annotation_stats[idx][4]+annotation_stats[idx][5];
    cout << endl;

    cout << setw(30) << "Total";
    cout << setw(10) << truth_positive_total-truth_positive_correct - truth_positive_partial;
    cout << setw(10) << truth_positive_partial;
    cout << setw(10) << truth_positive_correct;
    cout << setw(10) << truth_positive_total;
    cout << endl;
    cout << endl;



    // Dump category-specific stats
    annotation_stats.assign(annotation_names.size()+1, vector<int>(6,0));
    int truth_negative_total = 0;
    int truth_negative_match = 0;

    for (int truth_idx = 0; truth_idx < (int)truth.size(); ++truth_idx) {
      TruthSetAllele& allele = truth[truth_idx];
      if (allele.expected_call != 0)
        continue;
      truth_negative_total++;
      if (allele.match_status == 4)
        truth_negative_match++;
      for (set<int>::iterator I = allele.annotations.begin(); I != allele.annotations.end(); ++I)
        annotation_stats[*I][allele.match_status]++;
      if (allele.annotations.empty())
        annotation_stats.back()[allele.match_status]++;
    }

    cout << setw(30) << "Common failure annotations";
    cout << setw(10) << "FP";
    cout << setw(10) << "TN";
    cout << setw(10) << "Total" << endl;

    for (int idx = 0; idx < (int)annotation_names.size(); ++idx) {
      int total = annotation_stats[idx][0]+annotation_stats[idx][1]+annotation_stats[idx][2]+annotation_stats[idx][3]+annotation_stats[idx][4] + annotation_stats[idx][5];
      if (total == 0)
        continue;
      cout << setw(30) << annotation_names[idx];
      cout << setw(10) << annotation_stats[idx][4];
      cout << setw(10) << total - annotation_stats[idx][4];
      cout << setw(10) << total;
      cout << endl;
    }
    cout << "             -------------------------------------------------" << endl;
    cout << setw(30) << "With annotation";
    cout << setw(10) << truth_negative_match;
    cout << setw(10) << truth_negative_total - truth_negative_match;
    cout << setw(10) << truth_negative_total;
    cout << endl;


    int num_novel_fp = 0;
    int num_total_fp = 0;
    for (deque<VariantCallerAllele>::iterator results = results_vcf.begin(); results != results_vcf.end(); ++results) {
      // Skip correct calls
      if (results->match_status == 5)
        continue;
      if (results->match_status == 4 and results->match->expected_call > 0)
        continue;
      num_total_fp++;
      // Skip annotated FPs
      if (results->match_status == 4 and results->match->expected_call == 0)
        continue;
      num_novel_fp++;
    }

    cout << setw(30) << "Without annotation";
    cout << setw(10) << num_novel_fp;
    cout << endl;
    cout << setw(30) << "Total";
    cout << setw(10) << num_total_fp;
    cout << endl;
    cout << endl;

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
      IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(), IonVersion::GetGitHash().c_str());

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

  string fasta_filename= opts.GetFirstString ('-', "reference", "/results/referenceLibrary/tmap-f3/hg19/hg19.fasta");
  string targets_bed_filename= opts.GetFirstString ('-', "targets", "");


  string input_vcf_filename = opts.GetFirstString ('i', "input-vcf", "");
  string filtered_vcf_filename = opts.GetFirstString ('f', "filtered-vcf", "");

  //string truth_filename = opts.GetFirstString ('t', "truth-file", "");

  string annotation_filename = opts.GetFirstString ('-', "annotation", "../annotations.txt");

  string truth_filename1 = "/results/plugins/validateVariantCaller/files/NA12878_NIST_NoChrY_SNP.bed";
  string truth_filename2 = "/results/plugins/validateVariantCaller/files/NA12878_NIST_NoChrY_indel.bed";


  opts.CheckNoLeftovers();

  ReferenceReader ref_reader;
  ref_reader.Initialize(fasta_filename);

  vector<ValidatorTarget> targets;
  PopulateValidatorTarget(targets, ref_reader, targets_bed_filename);


  ValidatorTruth truth;
  truth.Initialize(ref_reader, targets);
  truth.AddTruthFile(truth_filename1);
  truth.AddTruthFile(truth_filename2);
  truth.AddAnnotationFile(annotation_filename);
  truth.CombineAndSort();



  //
  // Step 1. Load input VCF file into memory
  //

  if (input_vcf_filename.empty()) {
    VariantValidatorHelp();
    cerr << "ERROR: Input VCF file not specified " << endl;
    return 1;
  }

  deque<VariantCallerAllele> results_vcf;
  LoadVariantCallerResults(results_vcf, input_vcf_filename, ref_reader, true);

  deque<VariantCallerAllele> filtered_vcf;
  LoadVariantCallerResults(filtered_vcf, filtered_vcf_filename, ref_reader, false);

  //
  // Step 2. Parse truth files, compare them to the input vcf, and compute match scores
  //

  truth.CompareToCalls(results_vcf, filtered_vcf);


  return 0;
}





