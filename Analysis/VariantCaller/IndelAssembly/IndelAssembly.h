/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef INDELASSEMBLY_H
#define INDELASSEMBLY_H

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <fstream>
#include <iomanip>
#include <deque>
#include <map>

#include <tr1/unordered_set>

#include "IonVersion.h"
#include "OptArgs.h"
#include "api/BamMultiReader.h"
#include "api/BamWriter.h"
#include "json/json.h"
#include "ReferenceReader.h"
#include "TargetsManager.h"
#include "SampleManager.h"



// Example call:
// java -Xmx8G -cp /results/plugins/variantCaller/share/TVC/jar/
//  -jar /results/plugins/variantCaller/share/TVC/jar/GenomeAnalysisTK.jar
//  -T IndelAssembly
//  -R /results/referenceLibrary/tmap-f3/hg19/hg19.fasta
//  -I IonXpress_001_rawlib.bam
//  -L "/results/uploads/BED/1108/hg19/merged/plain/CHP2.20131001.designed.bed"
//  -o indel_assembly.vcf
//  -S SILENT
//  -U ALL
//  -filterMBQ
//  --short_suffix_match 5   --output_mnv 0   --min_var_count 5   --min_var_freq 0.15
//  --min_indel_size 4   --max_hp_length 8   --relative_strand_bias 0.8   --kmer_len 19

using namespace std;
using namespace std::tr1;
using namespace BamTools;



void IndelAssemblyHelp();

bool ValidateAndCanonicalizePath(string &path);

//int RetrieveParameterInt_x(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, int default_value);

//double RetrieveParameterDouble_x(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, double default_value);

class IndelAssemblyArgs {
public:
  IndelAssemblyArgs() {}
  IndelAssemblyArgs(int argc, char* argv[]);

void setDepthFile(const string& str);
void setReference(const string& str);
void setBams(vector<string>& v);
void setTargetFile(const string& str);
void setOutputVcf(const string& str);
void setParametersFile(const string& str);

void setSampleName(const string& str);

void processParameters(OptArgs& opts);

  string parameters_file;
  string reference;
  vector<string> bams;
  string target_file;
  string output_vcf;
  string sample_name;
  string force_sample_name;

  int read_limit;
  int kmer_len;
  int min_var_count;
  int short_suffix_match;
  int min_indel_size;
  int max_hp_length;
  double min_var_freq;
  double min_var_score;
  double relative_strand_bias;
  int output_mnv;
  bool multisample;
};



class CoverageBySample {
public:
  CoverageBySample() {}
  CoverageBySample(int num_samples) ;
  void Clear(int num_samples);
  void Increment(int strand, int sample, int num_samples) ;
  void Absorb(const CoverageBySample& other);

  void Min(const CoverageBySample& other);

  void Max(const CoverageBySample& other);

  int Total() const ;
  int TotalByStrand(int strand) const;
  int Sample(int sample) const ;
  int SampleByStrand(int strand, int sample) const ;
  const vector<int>& operator[](int strand) const ;
private:
  vector<int> cov_by_sample[2];
};


class Spectrum {
public:
  struct TKmer {
    int freq;
    int pos_in_reference;
    CoverageBySample cov_by_sample;
    TKmer() : freq(-1), pos_in_reference(-1) {}

    void Increment(int strand, int sample, int num_samples, bool is_primary_sample) {
      cov_by_sample.Increment(strand, sample, num_samples);
      if (is_primary_sample)
        freq++;
    }
    void Absorb(const TKmer& other) {
      cov_by_sample.Absorb(other.cov_by_sample);
      freq += other.freq;
    }
  };

  struct base_counts {
    int count1, count2;
    char key1, key2;
    base_counts (char k1, int c1, char k2, int c2) : count1(c1), count2(c2), key1(k1), key2(k2) {}
  };

  struct TVarCall {
    string varSeq;
    int startPos;
    int endPos;
    CoverageBySample varCov;
    bool repeatDetected;
    int lastPos;
  };

  map<string, TKmer> spectrum;
  int KMER_LEN;
  bool isERROR_INS;
  int num_samples;


  Spectrum(int kmerlen, int _num_samples)
      : KMER_LEN(kmerlen), isERROR_INS(false), num_samples(_num_samples) {}

  void add(const string& sequence, int strand, int sample, bool is_primary);
  static int getRepeatFreeKmer(const string& reference, int kmer_len);
  int getCounts(const string& kmer);
  int getPosInReference(const string& kmer);
  base_counts max2pairs(const string& kmer);
  void updateReferenceKmers(int shift);
  bool KmerPresent(const string& kmerstr);
  bool KmerPresent(const map<string,TKmer>::iterator& kmer);
  string DetectLeftAnchor(const string& reference, int minCount, int shortSuffix);
  bool isCorrectionEligible(const string& prevKmer, char fixBase, char errorBase);
  bool ApplyCorrection(const string& prevKmer, char fixBase, char errorBase);
  string advanceOnMaxPath(string startKmer, int stepsAhead);
  bool getPath(const string& anchorKMer, int minCount, int WINDOW_PREFIX, TVarCall& results);
  int getKMER_LEN();
};





class IndelAssembly {
public:
  IndelAssembly(IndelAssemblyArgs *_options, ReferenceReader *_reference_reader, SampleManager *_sample_manager, TargetsManager *_targets_manager) ;
  ~IndelAssembly() {pthread_mutex_destroy(&mutexmap);}

  struct Coverage {
    int soft_clip[2];
    int indel[2];
    CoverageBySample total;
    Coverage(int num_samples) {
      Clear(num_samples);
    }
    void Clear(int num_samples) {
      soft_clip[0] = soft_clip[1] = indel[0] = indel[1] = 0;
      total.Clear(num_samples);
    }
  };


  IndelAssemblyArgs *options;
  ReferenceReader *reference_reader;
  SampleManager *sample_manager;
  TargetsManager *targets_manager;
  
  ofstream out;

  const static int WINDOW_PREFIX = 300; // set it to max read length
  const static int WINDOW_SIZE = 2*WINDOW_PREFIX;
  int MIN_VAR_COUNT;
  double VAR_FREQ;
  size_t READ_LIMIT;
  int KMER_LEN;                         // fixed for now
  int SHORT_SUFFIX_MATCH;
  double RELATIVE_STRAND_BIAS;
  bool ASSEMBLE_SOFTCLIPS_ONLY;
  bool SKIP_MNV;
  int MIN_INDEL_SIZE;
  int MAX_HP_LEN;

  deque<Coverage> coverage;

  int curLeft;
  int curChrom;
  int softclip_event_start[2];
  int softclip_event_length[2];
  int indel_event_last[2];
  int assemStart;
  int assemVarCov_positive;
  int assemVarCov_negative;
  CoverageBySample assembly_total_cov;  

  deque<BamAlignment> ReadsBuffer;
  deque<BamAlignment> pre_buffer;

  pthread_mutex_t mutexmap;

  struct VarInfo {
    int contig;
    int pos;
    string ref;
    string var;
    VarInfo(int _contig, int _pos, const string& _ref, const string& _var)
      : contig(_contig), pos(_pos), ref(_ref), var(_var) {}
  };
  deque<VarInfo> calledVariants;

  bool processRead(BamAlignment& alignment, vector<MergedTarget>::iterator& indel_target);
  int getSoftEnd(BamAlignment& alignment);
  int getSoftStart(BamAlignment& alignment);
  void SetReferencePoint(BamAlignment& read);
  void map(BamAlignment& read);
  void onTraversalDone(bool do_assembly);
  void cleanCounts();
  void shiftCounts(int delta);
  void DetectCandidateRegions(int wsize);
  bool passFilter();
  void BuildKMerSpectrum(Spectrum& spectrum, int assemStart, int assemLength);
  void SegmentAssembly(int assemStart, int assemLength) ;
  int DetectIndel (int genStart, string reference, Spectrum& spectrum);
  void PrintVCF(const string& refwindow, const Spectrum::TVarCall& v, int contig, int pos,
                string ref, string var,
                int varLen, int type, int qual);
  void OutputVcfHeader();
  void AddCounts(BamAlignment& read);
};

#endif //INDELASSEMBLY_H










