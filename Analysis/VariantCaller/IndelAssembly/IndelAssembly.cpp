/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */


#include "IndelAssembly.h"

const int IndelAssembly::WINDOW_SIZE;

void IndelAssemblyHelp() {
  printf("Usage: tvcassembly [options]\n");
  printf("\n");

  printf("General options:\n");
  printf("  -h,--help                                         print this help message and exit\n");
  printf("  -v,--version                                      print version and exit\n");
  printf("  -n,--num-threads                      INT         number of worker threads [2]\n");
  printf("     --parameters-file                  FILE        json file with algorithm control parameters [optional]\n");
  printf("  -r,--reference                        FILE        reference fasta file [required]\n");
  printf("  -b,--input-bam                        FILE        bam file with mapped reads [required]\n");
  printf("  -d,--depth-file                       FILE        output depth file [optional]\n");
  printf("  -t,--target-file                      FILE        only process targets in this bed file [optional]\n");
  printf("  -o,--output-vcf                       FILE        vcf file with variant calling results [required]\n");
  printf("  -g,--sample-name                      STRING      sample for which variants are called (In case of input BAM files with multiple samples) [optional if there is only one sample]\n");
  printf("     --force-sample-name                STRING      force all read groups to have this sample name [off]\n");


  printf("\n");

  printf("     --read_limit                       INT         The maxmimum number of non-primary reads that can be held in memory [1000000].\n");
  printf("     --kmer_len                         INT         (klen) Size of the smallest k-mer used in assembly [19].\n");
  printf("     --min_var_count                    INT         (mincount) Minimum support for a variant to be evaluated [5].\n");
  printf("     --short_suffix_match               INT         (ssm) Minimum sequence match on both sides of the variant [5].\n");
  printf("     --min_indel_size                   INT         (mis) Minimum size indel reported by assembly [4].\n");
  printf("     --max_hp_length                    INT         (maxhp) Variants containing HP larger than this are not reported [8].\n");
  printf("     --min_var_freq                     FLOAT       (minfreq) Minimum frequency of the variant to be reported [0.15].\n");
  printf("     --min_var_score                    FLOAT       (minscore) Minimum score of the variant to be reported [10].\n");
  printf("     --relative_strand_bias             FLOAT       (stbias) Variants with strand bias above this are not reported [0.80].\n");
  printf("     --output_mnv                       INT         (mnv) Output multi-nucleotide variants assembled in a region with multiple adjacent variants [0].\n");

  printf("\n");
}

bool ValidateAndCanonicalizePath(string &path)
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

int RetrieveParameterInt_x(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, int default_value)
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

double RetrieveParameterDouble_x(OptArgs &opts, Json::Value& json, char short_name, const string& long_name_hyphens, double default_value)
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

  IndelAssemblyArgs::IndelAssemblyArgs(int argc, char* argv[]) {

    OptArgs opts;
    opts.ParseCmdLine(argc, (const char**)argv);

    if (argc == 1) {
      IndelAssemblyHelp();
      exit(0);
    }
    if (opts.GetFirstBoolean('v', "version", false)) {
      exit(0);
    }
    if (opts.GetFirstBoolean('h', "help", false)) {
      IndelAssemblyHelp();
      exit(0);
    }

    reference = opts.GetFirstString('r',"reference", "");

    opts.GetOption(bams, "", 'b', "input-bam");
    if (bams.empty()) {
      cout << "ERROR: Argument --input-bam is required\n";
      exit(-1);
    }
    for (unsigned int i_bam = 0; i_bam < bams.size(); ++i_bam)
      ValidateAndCanonicalizePath(bams[i_bam]);


    target_file = opts.GetFirstString('t',"target-file", "");
    output_vcf = opts.GetFirstString('o',"output-vcf", "");

    sample_name = opts.GetFirstString('g',"sample-name", "");
	multisample = false;
	if (sample_name == "") {multisample = true;}
    force_sample_name = opts.GetFirstString('-',"force-sample-name", "");

    if (reference.empty()) {
      cout << "ERROR: Argument --reference is required\n";
      exit(1);
    }
    if (output_vcf.empty()) {
      cout << "ERROR: Argument --output-vcf is required\n";
      exit(1);
    }

    parameters_file = opts.GetFirstString('-', "parameters-file", "");
	processParameters(opts);
}

void IndelAssemblyArgs::processParameters(OptArgs& opts) {
    Json::Value assembly_params(Json::objectValue);
    if (!parameters_file.empty()) {
      Json::Value parameters_json(Json::objectValue);
      ifstream in(parameters_file.c_str(), ifstream::in);
      if (!in.good()) {
        fprintf(stderr, "[tvc] FATAL ERROR: cannot open %s\n", parameters_file.c_str());
        exit(-1);
      }
      in >> parameters_json;
      in.close();
      if (parameters_json.isMember("pluginconfig"))
        parameters_json = parameters_json["pluginconfig"];
      assembly_params = parameters_json.get("long_indel_assembler", Json::objectValue);
    }

    read_limit = RetrieveParameterInt_x (opts, assembly_params, '-',"read-limit", 1000000);
    kmer_len = RetrieveParameterInt_x (opts, assembly_params, '-',"kmer-len", 19);
    min_var_count = RetrieveParameterInt_x (opts, assembly_params, '-',"min-var-count", 5);
    short_suffix_match = RetrieveParameterInt_x (opts, assembly_params, '-',"short-suffix-match", 5);
    min_indel_size = RetrieveParameterInt_x (opts, assembly_params, '-',"min-indel-size", 4);
    max_hp_length = RetrieveParameterInt_x (opts, assembly_params, '-',"max-hp-length", 8);
    min_var_freq = RetrieveParameterDouble_x(opts, assembly_params, '-',"min-var-freq", 0.15);
    min_var_score = RetrieveParameterDouble_x(opts, assembly_params, '-',"min-var-score", 10);
    relative_strand_bias = RetrieveParameterDouble_x(opts, assembly_params, '-',"relative-strand-bias", 0.80);
    output_mnv = RetrieveParameterInt_x (opts, assembly_params, '-',"output-mnv", 0);

    opts.CheckNoLeftovers();
  }
  
  void IndelAssemblyArgs::setReference(const string& str) {
      reference = str;
      if (reference.empty()) {
        cout << "ERROR: Argument --reference is required\n";
        exit(1);
      }
  }

  void IndelAssemblyArgs::setBams(vector<string>& v) {
      if (v.empty()) {
        cout << "ERROR: Argument --input-bam is required\n";
        exit(-1);
      }
      for (unsigned int i_bam = 0; i_bam < v.size(); ++i_bam)
        ValidateAndCanonicalizePath(v[i_bam]);
      bams = v;
  }
  void IndelAssemblyArgs::setTargetFile(const string& str) {target_file = str;}
  void IndelAssemblyArgs::setOutputVcf(const string& str) {
      output_vcf = str;
      if (output_vcf.empty()) {
        cout << "ERROR: Argument --output-vcf is required\n";
        exit(1);
      }
  }
  void IndelAssemblyArgs::setParametersFile(const string& str) {parameters_file = str;}

  void IndelAssemblyArgs::setSampleName(const string& str) {
      sample_name = str;
      multisample = false;
      if (sample_name == "") {multisample = true;}
  }

  CoverageBySample::CoverageBySample(int num_samples) {
    if ((int)cov_by_sample[0].size() != num_samples) {
      cov_by_sample[0].resize(num_samples, 0);
      cov_by_sample[1].resize(num_samples, 0);
    }
  }

  void CoverageBySample::Clear(int num_samples) {
    cov_by_sample[0].assign(num_samples,0);
    cov_by_sample[1].assign(num_samples,0);
  }

  void CoverageBySample::Increment(int strand, int sample, int num_samples) {
    if ((int)cov_by_sample[0].size() != num_samples) {
      cov_by_sample[0].resize(num_samples, 0);
      cov_by_sample[1].resize(num_samples, 0);
    }
    cov_by_sample[strand][sample]++;
  }
  void CoverageBySample::Absorb(const CoverageBySample& other) {
    if (cov_by_sample[0].size() < other.cov_by_sample[0].size()) {
      cov_by_sample[0].resize(other.cov_by_sample[0].size(), 0);
      cov_by_sample[1].resize(other.cov_by_sample[0].size(), 0);
    }
    for (int sample = 0; sample < (int)cov_by_sample[0].size(); ++sample) {
      cov_by_sample[0][sample] += other.cov_by_sample[0][sample];
      cov_by_sample[1][sample] += other.cov_by_sample[1][sample];
    }
  }

  void CoverageBySample::Min(const CoverageBySample& other) {
    if (cov_by_sample[0].size() < other.cov_by_sample[0].size()) {
      cov_by_sample[0].resize(other.cov_by_sample[0].size(), 0);
      cov_by_sample[1].resize(other.cov_by_sample[0].size(), 0);
    }
    for (int sample = 0; sample < (int)cov_by_sample[0].size(); ++sample) {
      cov_by_sample[0][sample] = min(cov_by_sample[0][sample], other.cov_by_sample[0][sample]);
      cov_by_sample[1][sample] = min(cov_by_sample[1][sample], other.cov_by_sample[1][sample]);
    }
  }

  void CoverageBySample::Max(const CoverageBySample& other) {
    if (cov_by_sample[0].size() < other.cov_by_sample[0].size()) {
      cov_by_sample[0].resize(other.cov_by_sample[0].size(), 0);
      cov_by_sample[1].resize(other.cov_by_sample[0].size(), 0);
    }
    for (int sample = 0; sample < (int)cov_by_sample[0].size(); ++sample) {
      cov_by_sample[0][sample] = max(cov_by_sample[0][sample], other.cov_by_sample[0][sample]);
      cov_by_sample[1][sample] = max(cov_by_sample[1][sample], other.cov_by_sample[1][sample]);
    }
  }

  int CoverageBySample::Total() const {
    int total = 0;
    for (int sample = 0; sample < (int)cov_by_sample[0].size(); ++sample)
      total += cov_by_sample[0][sample] + cov_by_sample[1][sample];
    return total;
  }
  int CoverageBySample::TotalByStrand(int strand) const {
    int total = 0;
    for (int sample = 0; sample < (int)cov_by_sample[0].size(); ++sample)
      total += cov_by_sample[strand][sample];
    return total;
  }
  int CoverageBySample::Sample(int sample) const {
    return cov_by_sample[0][sample] + cov_by_sample[1][sample];
  }
  int CoverageBySample::SampleByStrand(int strand, int sample) const {
    return cov_by_sample[strand][sample];
  }
  const vector<int>& CoverageBySample::operator[](int strand) const {
    return cov_by_sample[strand];
  }

  void Spectrum::add(const string& sequence, int strand, int sample, bool is_primary) {
    for(int x = 0; x <= (int)sequence.length() - KMER_LEN; x++) {
      TKmer& kmer = spectrum[sequence.substr(x,KMER_LEN)];
      kmer.Increment(strand, sample, num_samples, is_primary);
    }
  }


  int Spectrum::getRepeatFreeKmer(const string& reference, int kmer_len) {
    unordered_set<string> ref_spectrum;
    int kmer_max = 3 * kmer_len;
    for (; kmer_len < kmer_max; ++kmer_len) {
      bool has_repeat = false;
      for (int i = 0; i < (int)reference.length()-kmer_len; ++i) {
        string kseq = reference.substr(i, kmer_len);
        if(ref_spectrum.count(kseq)) {
          has_repeat = true;
          break;
        }
        ref_spectrum.insert(kseq);
      }
      if (!has_repeat)
        break;
    }
    return kmer_len;
  }

  int Spectrum::getCounts(const string& kmer) {
    if (spectrum.find(kmer) != spectrum.end())
      return spectrum[kmer].freq;
    return 0;
  }

  int Spectrum::getPosInReference(const string& kmer) {
    if (spectrum.find(kmer) != spectrum.end())
      return spectrum[kmer].pos_in_reference;
    return -1;
  }

  Spectrum::base_counts Spectrum::max2pairs(const string& kmer) {

    int a = getCounts(kmer + 'A');
    int c = getCounts(kmer + 'C');
    int g = getCounts(kmer + 'G');
    int t = getCounts(kmer + 'T');

    int max4 = max(a,max(c,max(g,t)));
    if (a==max4) { if(c>=g) { if(c>=t) return base_counts('A',a,'C',c); else return base_counts('A',a,'T',t); }
                   else     { if(g>=t) return base_counts('A',a,'G',g); else return base_counts('A',a,'T',t); } }
    if (c==max4) { if(a>=g) { if(a>=t) return base_counts('C',c,'A',a); else return base_counts('C',c,'T',t); }
                   else     { if(g>=t) return base_counts('C',c,'G',g); else return base_counts('C',c,'T',t); } }
    if (g==max4) { if(c>=a) { if(c>=t) return base_counts('G',g,'C',c); else return base_counts('G',g,'T',t); }
                   else     { if(a>=t) return base_counts('G',g,'A',a); else return base_counts('G',g,'T',t); } }
                   if(c>=g) { if(c>=a) return base_counts('T',t,'C',c); else return base_counts('T',t,'A',a); }
                   else     { if(g>=a) return base_counts('T',t,'G',g); else return base_counts('T',t,'A',a); }
  }

  void Spectrum::updateReferenceKmers(int shift) {
    for (map<string, TKmer>::iterator kmer = spectrum.begin(); kmer != spectrum.end(); ++kmer)
      kmer->second.pos_in_reference = max(kmer->second.pos_in_reference - shift, -1);
  }


  bool Spectrum::KmerPresent(const string& kmerstr) {
    map<string,TKmer>::iterator kmer = spectrum.find(kmerstr);
    if (kmer  == spectrum.end())
      return false;
    return kmer->second.freq >= 0;
  }

  bool Spectrum::KmerPresent(const map<string,TKmer>::iterator& kmer) {
    if (kmer  == spectrum.end())
      return false;
    return kmer->second.freq >= 0;
  }


  string Spectrum::DetectLeftAnchor(const string& reference, int minCount, int shortSuffix) {

    string anchorKMer;
    int firstAnchor = -1;

    for (int i = 0; i <= (int)reference.length()-KMER_LEN; ++i) {
      string refKMer = reference.substr(i,KMER_LEN);
      if(!KmerPresent(refKMer))
        continue;

      spectrum[refKMer].pos_in_reference = i;

      if (firstAnchor == -1 || i-firstAnchor == 1) {

        string otherKmer = refKMer.substr(0,refKMer.length()-1);
        base_counts m2p = max2pairs(otherKmer);

        int nCandPath = 0;
        if (m2p.count1 > minCount)
          nCandPath++;
        if (m2p.count2 > minCount)
          nCandPath++;

        if(firstAnchor != -1 && nCandPath==2 && isCorrectionEligible(anchorKMer,m2p.key1,m2p.key2)) {
          if(ApplyCorrection(anchorKMer,m2p.key1,m2p.key2))
            return "REPEAT";
          nCandPath = 1;
        }

        if(nCandPath==1 && refKMer == otherKmer+m2p.key1) {
          firstAnchor = i;
          anchorKMer = refKMer;
        }
      }
    }

    if( ((int)reference.length() - firstAnchor - 1 - KMER_LEN) < shortSuffix || firstAnchor == -1)
      return "NULL";
    else
      return anchorKMer;
  }

  bool Spectrum::isCorrectionEligible(const string& prevKmer, char fixBase, char errorBase) {

    //check if error is in HP (remove this condition when to consider other types of errors)
    if (prevKmer[prevKmer.length()-1] == fixBase || prevKmer[prevKmer.length()-1] == errorBase) {

      string fixedKmer = prevKmer.substr(1) + fixBase;
      string fixedNext = fixedKmer.substr(1) + errorBase;
      string errorKmer = prevKmer.substr(1) + errorBase;

      if (!KmerPresent(errorKmer) || !KmerPresent(fixedKmer))
        return false;

      int countError = spectrum[errorKmer].freq;
      int countFixed = spectrum[fixedKmer].freq;

      if(/*countError <= 0.1*(countError+countFixed) ||*/ KmerPresent(fixedNext)) {
        string seqPostError = advanceOnMaxPath(errorKmer,5);
        string seqPostFix = advanceOnMaxPath(fixedKmer,5);
        if (seqPostError.empty() || seqPostFix.empty())
          return false;
        if (seqPostFix.find(seqPostError.substr(1)) == 0) {
          isERROR_INS = true;
          return true;
        }
        if(seqPostError.find(seqPostFix.substr(1)) == 0) {
          isERROR_INS = false;
          return true;
        }
      }
    }
    return false;
  }


  bool Spectrum::ApplyCorrection(const string& prevKmer, char fixBase, char errorBase) {

    string errSeq = prevKmer.substr(1) + errorBase;
    string fixSeq;
    string extendingSeq = advanceOnMaxPath(errSeq,KMER_LEN);
    map<string,TKmer>::iterator errKmer = spectrum.find(errSeq);
    map<string,TKmer>::iterator fixKmer = spectrum.end();

    if (isERROR_INS) {
      fixSeq = prevKmer;
      fixKmer = spectrum.find(fixSeq);
      if(KmerPresent(fixKmer) && KmerPresent(errKmer)) {
        fixKmer->second.Absorb(errKmer->second);
        errKmer->second.freq = -1;
       }
    } else {
      fixSeq = prevKmer.substr(1)+fixBase;
      fixKmer = spectrum.find(fixSeq);
      if (KmerPresent(fixKmer) && KmerPresent(errKmer))
        fixKmer->second.Absorb(errKmer->second);
      fixSeq = fixSeq.substr(1) + errorBase;
      fixKmer = spectrum.find(fixSeq);
      if (KmerPresent(fixKmer) && KmerPresent(errKmer)){
        fixKmer->second.Absorb(errKmer->second);
        errKmer->second.freq = -1;
      }
    }

    for (int i = 0; i < (int)extendingSeq.length(); ++i) {
      errSeq = errSeq.substr(1) + extendingSeq[i];
      fixSeq = fixSeq.substr(1) + extendingSeq[i];
      errKmer = spectrum.find(errSeq);
      fixKmer = spectrum.find(fixSeq);
      if (KmerPresent(fixKmer) && KmerPresent(errKmer)) {
        if (errSeq == fixSeq || fixKmer->second.freq == -1)
          return true;
        fixKmer->second.Absorb(errKmer->second);
        errKmer->second.freq = -1;
      }
    }
    return false; // returns true when a repeat is detected
  }


  string Spectrum::advanceOnMaxPath(string startKmer, int stepsAhead) {

    string varSeq;
    varSeq.reserve(stepsAhead);
    while ((int)varSeq.length() < stepsAhead) {
      startKmer = startKmer.substr(1);
      base_counts m2p = max2pairs(startKmer);
      if (m2p.count1 <= 1)
        break;
      startKmer += m2p.key1;
      varSeq += m2p.key1;
    }
    return varSeq;
  }

  bool Spectrum::getPath(const string& anchorKMer, int minCount, int WINDOW_PREFIX, TVarCall& results) {

    results.startPos = spectrum[anchorKMer].pos_in_reference + KMER_LEN;
    results.varSeq.clear();
    string nextKmer;
    string prevKmer = anchorKMer;
    string tmpKmer;
    results.varCov.Clear(num_samples);

    while ((int)results.varSeq.length() < WINDOW_PREFIX) {
      nextKmer = prevKmer.substr(1);
      base_counts m2p = max2pairs(nextKmer);
      int nCandPath = (m2p.count1 > minCount ? 1 : 0) + (m2p.count2 > minCount ? 1 : 0);
      if (nCandPath == 0)
        break;

      tmpKmer = nextKmer + m2p.key1;
      results.endPos = spectrum[tmpKmer].pos_in_reference;

      // if we have 2 candidates and we picked reference then change it to variant
      if (nCandPath == 2 && results.endPos > -1 && results.varSeq.empty()) {
        m2p.key1 = m2p.key2;
        m2p.count1 = m2p.count2;
        nCandPath = 1;
        tmpKmer = nextKmer + m2p.key1;
        results.endPos = spectrum[tmpKmer].pos_in_reference;
      }

      if (results.endPos > -1 && results.varSeq.empty()) {
        results.varCov.Clear(num_samples);
        results.lastPos = 0;
        results.repeatDetected = true;
        return false;
      }

      if (results.endPos >= results.startPos) {
        results.lastPos = results.endPos;
        results.repeatDetected = false;
        return true;
      }

      if(results.endPos >= -1)
        spectrum[tmpKmer].pos_in_reference = -2;

      else if((results.endPos==-2 || results.varSeq.empty()) && nCandPath == 2) {
        tmpKmer = nextKmer + m2p.key2;
        results.endPos = spectrum[tmpKmer].pos_in_reference;
        if (results.endPos >= results.startPos) {
          results.lastPos = results.endPos;
          results.repeatDetected = false;
          return true;
        } else if (results.endPos==-1)
          spectrum[tmpKmer].pos_in_reference = -2;
        else if (results.endPos==-2) {
          results.varCov.Clear(num_samples);
          results.lastPos = 0;
          results.repeatDetected = true;
          return false;
        }
        m2p.key1 = m2p.key2;
        m2p.count1 = m2p.count2;
        nCandPath = 1;

      } else {
        results.varCov.Clear(num_samples);
        results.lastPos = 0;
        results.repeatDetected = true;
        return false;
      }

      if (nCandPath == 2 && isCorrectionEligible(prevKmer,m2p.key1,m2p.key2)) {
        if (ApplyCorrection(prevKmer,m2p.key1,m2p.key2)) {
          results.varCov.Clear(num_samples);
          results.lastPos = 0;
          results.repeatDetected = true;
          return false;
        }
      }

      if (results.varSeq.empty())
        results.varCov = spectrum[tmpKmer].cov_by_sample;
      else
        results.varCov.Min(spectrum[tmpKmer].cov_by_sample);
      results.varSeq += m2p.key1;
      prevKmer = tmpKmer;
    }

    results.endPos = 0;
    results.lastPos = results.startPos + 1;
    results.repeatDetected = false;
    return true;
  }

  int Spectrum::getKMER_LEN() {
    return KMER_LEN;
  }

  bool IndelAssembly::processRead(BamAlignment& alignment, vector<MergedTarget>::iterator& current_target) {
    if (!alignment.IsMapped()) {
      return true;
    }

    // back up a couple of targets
    while (current_target != targets_manager->merged.end() && (alignment.RefID > current_target->chr || (alignment.RefID == current_target->chr && alignment.Position >= current_target->end))) {
        ++current_target;
    }

    if (current_target == targets_manager->merged.end()) {
        return false;
    }

    if (alignment.RefID < current_target->chr || (alignment.RefID == current_target->chr && alignment.GetEndPosition() <= current_target->begin)) {
        return true;
    }
    pthread_mutex_lock (&mutexmap);
    map(alignment);
    pthread_mutex_unlock (&mutexmap);

    return true;
  }

  IndelAssembly::IndelAssembly(IndelAssemblyArgs *_options, ReferenceReader *_reference_reader, SampleManager *_sample_manager, TargetsManager *_targets_manager) {
	pthread_mutex_init(&mutexmap, NULL);
    options = _options;
    reference_reader = _reference_reader;
    sample_manager = _sample_manager;
	targets_manager = _targets_manager;

    MIN_VAR_COUNT = options->min_var_count;
    VAR_FREQ = options->min_var_freq;
	READ_LIMIT = options->read_limit;
    KMER_LEN = options->kmer_len;                         // fixed for now
    SHORT_SUFFIX_MATCH = options->short_suffix_match;
    RELATIVE_STRAND_BIAS = options->relative_strand_bias;
    ASSEMBLE_SOFTCLIPS_ONLY = false;
    SKIP_MNV = (options->output_mnv == 0);
    MIN_INDEL_SIZE = options->min_indel_size;
    MAX_HP_LEN = options->max_hp_length;

    curChrom = -1;
    curLeft = -1;
    softclip_event_start[0] = softclip_event_start[1] = 0;
    softclip_event_length[0] = softclip_event_length[1] = 0;
    indel_event_last[0] = indel_event_last[1] = 0;
    assemStart = 0;
    assemVarCov_positive = 0;
    assemVarCov_negative = 0;
    assembly_total_cov.Clear(sample_manager->num_samples_);
    coverage.resize(WINDOW_SIZE, sample_manager->num_samples_);

    out.open(options->output_vcf.c_str());
    OutputVcfHeader();
  }
  
  int IndelAssembly::getSoftEnd(BamAlignment& alignment) {

    int position = alignment.Position + 1;
    bool left_clip_skipped = false;

    for(int j = 0; j < (int)alignment.CigarData.size(); ++j) {
      char cgo = alignment.CigarData[j].Type;
      int cgl = alignment.CigarData[j].Length;

      if ((cgo == 'H' || cgo == 'S') && !left_clip_skipped)
        continue;
      left_clip_skipped = true;

      if (cgo == 'H')
        break;
      if (cgo == 'S' || cgo == 'M' || cgo == 'D' || cgo == 'N')
        position += cgl;
    }

    return position;
  }



  int IndelAssembly::getSoftStart(BamAlignment& alignment) {

    int position = alignment.Position + 1;
    for(vector<CigarOp>::const_iterator cigar = alignment.CigarData.begin(); cigar != alignment.CigarData.end(); ++cigar) {
      if (cigar->Type == 'H')
        continue;
      if (cigar->Type == 'S') {
        position -= cigar->Length;
        continue;
      }
      break;
    }
    return position;
  }


  void IndelAssembly::SetReferencePoint(BamAlignment& read) {
    curChrom = read.RefID;
    curLeft = max(getSoftStart(read) - WINDOW_PREFIX, 0);
    softclip_event_start[0] = softclip_event_start[1] = 0;
    softclip_event_length[0] = softclip_event_length[1] = 0;
    indel_event_last[0] = indel_event_last[1] = 0;
    assemStart =  0;
    assemVarCov_positive = assemVarCov_negative = 0;
    assembly_total_cov.Clear(sample_manager->num_samples_);
  }


  void IndelAssembly::map(BamAlignment& read) {
    int sample;
    bool is_primary;
    if (!sample_manager->IdentifySample(read, sample, is_primary))
      return;
    if (!is_primary) {
      if (pre_buffer.size() == READ_LIMIT) {
        cerr << endl << "FATAL ERROR: IndelAssembly::map() pre_buffer overflow. You do not have enough primary sample reads in the BAM. Try increasing the read-limit parameter." << endl;
        exit(1);
      }
      else {
        pre_buffer.push_back(read);
      }
      return;
    }
    int delta = WINDOW_SIZE; // signal start of a new chromosome
    if (curChrom == read.RefID)
      delta  = getSoftEnd(read) - curLeft - WINDOW_SIZE;

    // Window is about to shift forward
    if(delta > 0) {

      DetectCandidateRegions(min(delta,WINDOW_SIZE));

      if (delta >= WINDOW_SIZE) { // Big shift
        SetReferencePoint(read);
        cleanCounts();
        ReadsBuffer.clear();
      } else { // Small shift
        shiftCounts(delta);
        curLeft += delta;
        while(!ReadsBuffer.empty() && getSoftEnd(ReadsBuffer.front()) <= curLeft)
          ReadsBuffer.pop_front();
      }
    }
	
    while (!pre_buffer.empty()) {
      if (getSoftEnd(pre_buffer.front()) > curLeft) {
        ReadsBuffer.push_back(pre_buffer.front());

        AddCounts(pre_buffer.front());
      }
      pre_buffer.pop_front();
    }

    ReadsBuffer.push_back(read);
    AddCounts(read);
  }

  void IndelAssembly::cleanCounts() {
    for (deque<Coverage>::iterator I = coverage.begin(); I != coverage.end(); ++I)
      I->Clear(sample_manager->num_samples_);
  }

  void IndelAssembly::shiftCounts(int delta) {
    coverage.erase(coverage.begin(), coverage.begin()+delta);
    coverage.resize(WINDOW_SIZE, sample_manager->num_samples_);
  }



  void IndelAssembly::DetectCandidateRegions(int wsize) {

    if (curChrom < 0)
      return;

    for(int i = 0; i < wsize; ++i) {

      for (int strand = 0; strand <= 1; ++strand) {
        //tracking soft-clips on positive strand
        if(coverage[i].soft_clip[strand] > MIN_VAR_COUNT) {
          if(softclip_event_start[strand] == 0)
            softclip_event_start[strand] = curLeft + i; // begin tracking
          softclip_event_length[strand] = curLeft + i - softclip_event_start[strand]; // continue tracking

        } else if(curLeft + i - softclip_event_start[strand] - softclip_event_length[strand] > KMER_LEN // stop tracking
            && (i+softclip_event_length[strand] >= WINDOW_SIZE
                || coverage[i+softclip_event_length[strand]].soft_clip[1-strand] < MIN_VAR_COUNT))
          softclip_event_start[strand] = softclip_event_length[strand] = 0;

        if (!ASSEMBLE_SOFTCLIPS_ONLY) {
          //tracking indels on positive strand
          if (coverage[i].indel[strand] > MIN_VAR_COUNT
              || (coverage[i].total.SampleByStrand(strand,sample_manager->primary_sample_) > 2*MIN_VAR_COUNT && coverage[i].indel[strand] >= 2*VAR_FREQ*coverage[i].total.SampleByStrand(strand,sample_manager->primary_sample_))
              || (coverage[i].total.SampleByStrand(strand,sample_manager->primary_sample_) > MIN_VAR_COUNT/2 && coverage[i].indel[strand] >= 4*VAR_FREQ*coverage[i].total.SampleByStrand(strand,sample_manager->primary_sample_)))
            indel_event_last[strand] = curLeft + i;

          else if(curLeft + i - indel_event_last[strand] > KMER_LEN)
            indel_event_last[strand] = 0;
        }
      }


      // start assembly tracking if any event observed
      if(assemStart == 0) {
        if (softclip_event_length[0] > KMER_LEN || softclip_event_length[1] > KMER_LEN)
          assemStart = curLeft + i - KMER_LEN;
        else if (indel_event_last[0] + indel_event_last[1] > 0)
          assemStart = curLeft + i;
      }

      // accumulate assembly coverage stats
      if(assemStart > 0) {
        assemVarCov_positive = max(assemVarCov_positive, ((softclip_event_length[0] > KMER_LEN) ? coverage[i].soft_clip[0] : 0 ) + coverage[i].indel[0]);
        assemVarCov_negative = max(assemVarCov_negative, ((softclip_event_length[1] > KMER_LEN) ? coverage[i].soft_clip[1] : 0 ) + coverage[i].indel[1]);
        assembly_total_cov.Max(coverage[i].total);
      }
      else {
        assembly_total_cov = coverage[i].total;
      }

      int assemLen = assemStart > 0 ? curLeft + i - assemStart : 0;
      if (assemLen > WINDOW_PREFIX
          || softclip_event_start[0] + softclip_event_start[1] + indel_event_last[0] + indel_event_last[1] == 0
          || i == wsize-1) {

        if(assemLen > 0) {
          if(assemLen == KMER_LEN+1 && assemStart - curLeft >=0) {  //clear single indel case
            assemVarCov_positive = coverage[assemStart - curLeft].soft_clip[0] + coverage[assemStart - curLeft].indel[0];
            assemVarCov_negative = coverage[assemStart - curLeft].soft_clip[1] + coverage[assemStart - curLeft].indel[1];
          }
          if(passFilter())
            SegmentAssembly(assemStart, assemLen);
        }
        assemStart = assemVarCov_positive = assemVarCov_negative = 0;
        assembly_total_cov.Clear(sample_manager->num_samples_);
      }
    }
  }

  bool IndelAssembly::passFilter() {
    if ((assemVarCov_positive + assemVarCov_negative) < VAR_FREQ*assembly_total_cov.Sample(sample_manager->primary_sample_)) //(assemTotCov_positive + assemTotCov_negative))
      return false;
    if (assembly_total_cov.SampleByStrand(0,sample_manager->primary_sample_) > 3 && assembly_total_cov.SampleByStrand(1,sample_manager->primary_sample_) > 3) {
      float factor1 = assemVarCov_negative * assembly_total_cov.SampleByStrand(0,sample_manager->primary_sample_);
      float factor2 = assemVarCov_positive * assembly_total_cov.SampleByStrand(1,sample_manager->primary_sample_);
      return (max(factor1,factor2) / (factor1+factor2)) < RELATIVE_STRAND_BIAS;
    }
    return true;
  }




  void IndelAssembly::BuildKMerSpectrum(Spectrum& spectrum, int assemStart, int assemLength) {

    for(int i = 0; i < (int)ReadsBuffer.size(); ++i) {
      BamAlignment& read = ReadsBuffer[i];

      int strand = read.IsReverseStrand() ? 1 : 0;
      int sample;
      bool is_primary;
      if (!sample_manager->IdentifySample(read, sample, is_primary))
        continue;

      int read_assem_start = assemStart - getSoftStart(read);
      int read_pos = 0;
      int lastIncluded = 0;
      int prev_cgl = 0;

      for(int j = 0; j < (int)read.CigarData.size() && read_pos-read_assem_start < assemLength; ++j) {
        char cgo = read.CigarData[j].Type;
        int cgl = read.CigarData[j].Length;

        if(cgo == 'S' || ((cgo == 'I' || cgo == 'D') && cgl>2)) {

          if (read_pos + cgl > read_assem_start)  {

            int startPos = max(read_pos, read_assem_start) - ((cgo == 'S') ? (KMER_LEN+10) : prev_cgl);

            int stopPos = read_pos + cgl + ((j == 0 && cgo == 'S') ? cgl : assemLength) + KMER_LEN + 1;

            if (lastIncluded < stopPos - KMER_LEN) {
              int seqStart = max(startPos, lastIncluded);
              int seqStop = min(stopPos, (int)read.QueryBases.length());

              if(seqStart >= (int)read.QueryBases.length() || seqStart >= seqStop - KMER_LEN)
                break;

              spectrum.add(read.QueryBases.substr(seqStart, seqStop - seqStart), strand, sample, is_primary);

              if (stopPos >= (int)read.QueryBases.length())
                break;
              lastIncluded = stopPos - KMER_LEN;
              if(lastIncluded < 0)
                lastIncluded = 0;
            }
          }
        }
        if(cgo != 'D' && cgo != 'H')
          read_pos += cgl;
        prev_cgl = cgl;
      }
    }
  }





  void IndelAssembly::SegmentAssembly(int assemStart, int assemLength) {

    if (assemStart >= (int)reference_reader->chr_size(curChrom))
      return;

    //cout << "SegmentAssembly(" << assemStart << "," << assemLength << ",chr="<< curChrom <<",nreads=" << ReadsBuffer.size() << ")\n";

    int KMER_EXT = 3*KMER_LEN;
    int kmerlen = KMER_LEN;

    int genStart = max(assemStart-KMER_EXT, 1);
    int genStop = min(assemStart + assemLength + KMER_EXT + 1, (int)reference_reader->chr_size(curChrom));
    string reference = reference_reader->substr(curChrom, genStart-1, genStop-genStart+1);

    // auto detect the size of k-mer to guarantee uniqueness
    kmerlen = Spectrum::getRepeatFreeKmer(reference, kmerlen);


    // the loop is created to support re-assembly in case of repeat detection
    while(kmerlen <= KMER_EXT) {

      Spectrum spectrum(kmerlen, sample_manager->num_samples_);

      BuildKMerSpectrum(spectrum, assemStart, assemLength);
      int repeatSegment = DetectIndel(genStart, reference, spectrum);
      if(repeatSegment == -1)
        break;

      int delta = repeatSegment - genStart;
      genStart += delta;
      assemStart += delta;
      assemLength -= delta;
      if (assemLength <= 0)
        break;
      kmerlen += KMER_LEN/2;
      reference = reference.substr(delta);

    }
  }


  int IndelAssembly::DetectIndel (int genStart, string reference, Spectrum& spectrum) {
    int cutFreq = (int)(0.1*(assemVarCov_positive + assemVarCov_negative));    // check this later
    if(cutFreq<MIN_VAR_COUNT)
      cutFreq = MIN_VAR_COUNT;

    int kmerlen = spectrum.getKMER_LEN();

    for(bool keepAssembling = true; keepAssembling;) { // assemble multiple events in a single window
      keepAssembling = false;
      string anchorKMer = spectrum.DetectLeftAnchor(reference,cutFreq,SHORT_SUFFIX_MATCH);
      if (anchorKMer == "NULL")
        break;
      if (anchorKMer == "REPEAT")
        return genStart;  // attempt to re-assemble with larger k-mer

      Spectrum::TVarCall var;
      bool no_repeats = spectrum.getPath(anchorKMer, cutFreq, WINDOW_PREFIX, var);

      if(!no_repeats)
        return genStart;  // attempt to re-assemble with larger k-mer

      if (!var.varSeq.empty() &&
          var.varCov.Sample(sample_manager->primary_sample_) >= MIN_VAR_COUNT &&
          var.varCov.Sample(sample_manager->primary_sample_) >= VAR_FREQ * assembly_total_cov.Sample(sample_manager->primary_sample_)) {

        if(var.endPos > 0) {


          if((int)var.varSeq.length() >= kmerlen-1 && (int)var.varSeq.length() - kmerlen + 1 > var.endPos - var.startPos) {
             // INSERTION  type 0 or 4 for MNV
             PrintVCF(reference, var, curChrom, genStart + var.startPos - 1,
                      reference.substr(var.startPos - 1, var.endPos-var.startPos + (var.endPos-var.startPos > 0 ? 1:0) + 1),
                      anchorKMer.substr(anchorKMer.length()-1) + var.varSeq.substr(0, var.varSeq.length() - kmerlen + (var.endPos - var.startPos > 0 ? 2:1)),
                      var.varSeq.length() - kmerlen + 1,
                      var.endPos - var.startPos > 0 ? 4 : 0,
                      50);

          } else if((int)var.varSeq.length() - kmerlen + 1 <= var.endPos - var.startPos) {
            // DELETION type 1 or 5 for MNV
            if((int)var.varSeq.length() + 1 - kmerlen >= 0) {
              PrintVCF(reference, var, curChrom, genStart + var.startPos - 1,
                       reference.substr(var.startPos - 1, var.endPos+(var.varSeq.length() - kmerlen + 1 > 0 ? 1:0) - (var.startPos - 1)),
                       reference.substr(var.startPos - 1, 1) + ((var.varSeq.length()  - kmerlen + 1 > 0) ? (var.varSeq.substr(0,var.varSeq.length()-kmerlen+1) + reference.substr(var.endPos,1)):""),
                       var.endPos  - var.startPos,
                       var.varSeq.length()  - kmerlen + 1 > 0 ? 5 : 1,
                       50);
            } else {
              PrintVCF(reference, var, curChrom, genStart + var.startPos - 1,
                       reference.substr(var.startPos - 1, var.endPos + kmerlen - var.varSeq.length()-1 - (var.startPos - 1)),
                       reference.substr(var.startPos - 1, 1),
                       var.endPos  - var.startPos + kmerlen - var.varSeq.length() - 1,
                       1,
                       50);
            }
          }
        } else if(var.endPos==0 && (int)var.varSeq.length() >= SHORT_SUFFIX_MATCH) {
          string shortSuffix = var.varSeq.substr(var.varSeq.length()-SHORT_SUFFIX_MATCH);
          string refSuffix = reference.substr(var.startPos);
          string shortRef = refSuffix.substr(0, min(kmerlen,(int)refSuffix.length()));
          string preffix = var.varSeq.substr(0, min(kmerlen,(int)var.varSeq.length()));

          size_t x = ((int)preffix.length() > SHORT_SUFFIX_MATCH) ? refSuffix.find(preffix) : string::npos;

          if (x > 0 && x != string::npos) {
            // DELETION type 3
            // to-do: support MNV
            PrintVCF(reference, var, curChrom, genStart + var.startPos - 1,
                     reference.substr(var.startPos - 1, 1 + x),
                     reference.substr(var.startPos - 1, 1),
                     x,
                     3,
                     (75+2*(2*preffix.length()-kmerlen/5))/10);

          } else if(shortRef.find(shortSuffix) != string::npos) {
            int delta = 0;
            int sufsz =  SHORT_SUFFIX_MATCH;
            int nummatch = 0;
            while (shortRef.find(shortSuffix) != string::npos) {
              nummatch = 0;
              for(int i = 0; (i+sufsz) < (int)shortRef.length() && (int)shortSuffix.length() == sufsz && nummatch < 2; i++) {
                if(shortRef.substr(i,sufsz) == shortSuffix) {
                  delta = i;
                  nummatch++;
                }
              }
              if(nummatch < 2)
                break;
              sufsz += 3;
              if((int)var.varSeq.length() > sufsz)
                shortSuffix = var.varSeq.substr(var.varSeq.length() - sufsz);
            }
            if(nummatch == 1) {
              // this is very simplistic decision, overcall/undercall can produce unexpected result, sequence comparison is required at this step
              // do it later  , produce an MNV
              if((int)var.varSeq.length() < delta + sufsz) {
                // DELETION type 3
                PrintVCF(reference, var, curChrom, genStart + var.startPos - 1,
                         reference.substr(var.startPos - 1, delta + sufsz - var.varSeq.length() + 1),
                         reference.substr(var.startPos - 1, 1),
                         delta + sufsz - var.varSeq.length(),
                         3,
                         1);
              } else {
                // INSERTION  type 2
                PrintVCF(reference, var, curChrom, genStart + var.startPos - 1,
                         anchorKMer.substr(anchorKMer.length()-1),
                         anchorKMer.substr(anchorKMer.length()-1) + var.varSeq.substr(0, var.varSeq.length() - delta - sufsz),
                         var.varSeq.length() - delta - sufsz,
                         2,
                         2);
              }
            }
          }
        }
      }

      if(var.lastPos > 0 && var.lastPos <= (int)reference.length()- kmerlen - SHORT_SUFFIX_MATCH && var.lastPos <= (int)reference.length() - 2*kmerlen) {
        reference = reference.substr(var.lastPos);
        genStart += var.lastPos;
        spectrum.updateReferenceKmers(var.lastPos);
        keepAssembling = true;
      }
    }
    return -1;
  }



  void IndelAssembly::PrintVCF(const string& refwindow, const Spectrum::TVarCall& v, int contig, int pos,
                string ref, string var,
                int varLen, int type, int qual) {

    if (varLen < MIN_INDEL_SIZE)
      return;
    if (SKIP_MNV && type > 3)
      return; // do not produce MNV

    int varCounts = v.varCov.Sample(sample_manager->primary_sample_);
    int totCounts = max(assembly_total_cov.Sample(sample_manager->primary_sample_), 1);
    int refCounts = max(totCounts - varCounts, 0);

    // Left Align the variant

    if(type < 4) {
      while (pos > 1) {
        if (ref[ref.length()-1] != var[var.length()-1])
          break;
        char pad = reference_reader->base(contig, pos-2);
        ref = pad + ref.substr(0, ref.length()-1);
        var = pad + var.substr(0, var.length()-1);
        --pos;
      }
    }

    string hpMaxString = refwindow.substr(max(v.startPos-1-MAX_HP_LEN, 0), v.startPos-1 - max(v.startPos-1-MAX_HP_LEN, 0))
        + var + refwindow.substr(v.startPos + ref.length() - 1, min(v.startPos + ref.length() - 1 + MAX_HP_LEN, refwindow.length()) - (v.startPos + ref.length() - 1));

    bool hpMax = false;
    // max HP length check
    if (MAX_HP_LEN < 20) {
      if(hpMaxString.find("AAAAAAAAAAAAAAAAAAAA", 0, MAX_HP_LEN) != string::npos)
        hpMax = true;
      else if(hpMaxString.find("CCCCCCCCCCCCCCCCCCCC", 0, MAX_HP_LEN) != string::npos)
        hpMax = true;
      else if(hpMaxString.find("GGGGGGGGGGGGGGGGGGGG", 0, MAX_HP_LEN) != string::npos)
        hpMax = true;
      else if(hpMaxString.find("TTTTTTTTTTTTTTTTTTTT", 0, MAX_HP_LEN) != string::npos)
        hpMax = true;
    }
    if (hpMax)
      qual = 0;

    if (varCounts < 30)
      qual = qual / (30-varCounts);
    if (qual < options->min_var_score)
      return;


    float varfq = (totCounts <= varCounts) ? 1.0f : ((float)varCounts)/((float)totCounts);
    float reffq = (totCounts <= refCounts) ? 1.0f : ((float)refCounts)/((float)totCounts);
    string genotype = (reffq > 0.2) ? "0/1" : "1/1";



    // Ensure the same variant is not reported twice
    int i = calledVariants.size() - 1;
    while(i > -1 && calledVariants[i].contig == contig && abs(pos - calledVariants[i].pos) < 300) {
      if(calledVariants[i].pos == pos &&  calledVariants[i].ref == ref && calledVariants[i].var == var)
        return;
      i--;
    }
    for(int j = 0; j <= i; j++)
      calledVariants.pop_front();
    calledVariants.push_back(VarInfo(contig, pos, ref, var));

    out << reference_reader->chr(contig) << "\t"
        << pos << "\t"
        << "." << "\t"
        << ref << "\t"
        << var << "\t"
        << qual << "\t"
        << "PASS" << "\t"
        << "AO=" << v.varCov.Total() << ";"
        << "DP=" << assembly_total_cov.Total() << ";"
        << "LEN=" << varLen << ";"
        << "RO=" << max(assembly_total_cov.Total() - v.varCov.Total(), 0) << ";"
        << "SAF=" << v.varCov.TotalByStrand(0) << ";"
        << "SAR=" << v.varCov.TotalByStrand(1) << ";"
        << "SRF=" << max(assembly_total_cov.TotalByStrand(0)-v.varCov.TotalByStrand(0), 0) << ";"
        << "SRR=" << max(assembly_total_cov.TotalByStrand(1)-v.varCov.TotalByStrand(1), 0) << ";"
        << "AF=" << (assembly_total_cov.Total() ? v.varCov.Total()/(float)assembly_total_cov.Total() : 0) <<";"
        << "TYPE=" << (type>3?"mnv":(ref.length()>var.length() ? "del" : (ref.length()<var.length() ? "ins" : "snp"))) << "\t"
        << "GT:GQ:DP:RO:AO:SAF:SAR:SRF:SRR:AF"
        << "\t" << genotype << ":99:"
        << assembly_total_cov.Sample(sample_manager->primary_sample_) << ":"
        << max(assembly_total_cov.Sample(sample_manager->primary_sample_) - v.varCov.Sample(sample_manager->primary_sample_), 0) << ":"
        << v.varCov.Sample(sample_manager->primary_sample_) << ":"
        << v.varCov.SampleByStrand(0, sample_manager->primary_sample_) << ":"
        << v.varCov.SampleByStrand(1, sample_manager->primary_sample_) << ":"
        << max(assembly_total_cov.SampleByStrand(0, sample_manager->primary_sample_) - v.varCov.SampleByStrand(0, sample_manager->primary_sample_), 0) << ":"
        << max(assembly_total_cov.SampleByStrand(1, sample_manager->primary_sample_) - v.varCov.SampleByStrand(1, sample_manager->primary_sample_), 0) << ":"
        << (assembly_total_cov.Sample(sample_manager->primary_sample_) ? v.varCov.Sample(sample_manager->primary_sample_)/(float)assembly_total_cov.Sample(sample_manager->primary_sample_) : 0);

    for (int sample = 0; sample < sample_manager->num_samples_; ++sample) {
      if (sample == sample_manager->primary_sample_)
        continue;
      if (options->multisample) {
          int varCounts = v.varCov.Sample(sample);
          int totCounts = max(assembly_total_cov.Sample(sample), 1);
          int refCounts = max(totCounts - varCounts, 0);
          float reffq = (totCounts <= refCounts) ? 1.0f : ((float)refCounts)/((float)totCounts);
          string genotype = (reffq > 0.2) ? "0/1" : "1/1";
          out << "\t" << genotype << ":99:"
              << assembly_total_cov.Sample(sample) << ":"
              << max(assembly_total_cov.Sample(sample) - v.varCov.Sample(sample), 0) << ":"
              << v.varCov.Sample(sample) << ":"
              << v.varCov.SampleByStrand(0, sample) << ":"
              << v.varCov.SampleByStrand(1, sample) << ":"
              << max(assembly_total_cov.SampleByStrand(0, sample) - v.varCov.SampleByStrand(0, sample), 0) << ":"
              << max(assembly_total_cov.SampleByStrand(1, sample) - v.varCov.SampleByStrand(1, sample), 0) << ":"
              << (assembly_total_cov.Sample(sample) ? v.varCov.Sample(sample)/(float)assembly_total_cov.Sample(sample) : 0);
      }
      else {
          out << "\t./.:0:"
              << assembly_total_cov.Sample(sample) << ":"
              << max(assembly_total_cov.Sample(sample) - v.varCov.Sample(sample), 0) << ":"
              << v.varCov.Sample(sample) << ":"
              << v.varCov.SampleByStrand(0, sample) << ":"
              << v.varCov.SampleByStrand(1, sample) << ":"
              << max(assembly_total_cov.SampleByStrand(0, sample) - v.varCov.SampleByStrand(0, sample), 0) << ":"
              << max(assembly_total_cov.SampleByStrand(1, sample) - v.varCov.SampleByStrand(1, sample), 0) << ":"
              << (assembly_total_cov.Sample(sample) ? v.varCov.Sample(sample)/(float)assembly_total_cov.Sample(sample) : 0);
      }
    }

    out << "\n";
  }


  void IndelAssembly::OutputVcfHeader() {
    out << "##fileformat=VCFv4.1\n";
    out << "##source=\"tvcassembly "<<IonVersion::GetVersion()<<"-"<<IonVersion::GetRelease()<<" ("<<IonVersion::GetGitHash()<<")\"\n";
    out << "##reference=" << options->reference << "\n";
    out << "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total read depth at the locus\">\n";
    out << "##INFO=<ID=RO,Number=1,Type=Integer,Description=\"Reference allele observations\">\n";
    out << "##INFO=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observations\">\n";
    out << "##INFO=<ID=SRF,Number=1,Type=Integer,Description=\"Number of reference observations on the forward strand\">\n";
    out << "##INFO=<ID=SRR,Number=1,Type=Integer,Description=\"Number of reference observations on the reverse strand\">\n";
    out << "##INFO=<ID=SAF,Number=A,Type=Integer,Description=\"Alternate allele observations on the forward strand\">\n";
    out << "##INFO=<ID=SAR,Number=A,Type=Integer,Description=\"Alternate allele observations on the reverse strand\">\n";
    out << "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator observation counts\">\n";
    out << "##INFO=<ID=TYPE,Number=A,Type=String,Description=\"The type of allele, either snp, mnp, ins, del, or complex.\">\n";
    out << "##INFO=<ID=LEN,Number=A,Type=Integer,Description=\"Allele length\">\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality, the Phred-scaled marginal (or unconditional) probability of the called genotype\">\n";
    out << "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n";
    out << "##FORMAT=<ID=RO,Number=1,Type=Integer,Description=\"Reference allele observation count\">\n";
    out << "##FORMAT=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observation count\">\n";
    out << "##FORMAT=<ID=SRF,Number=1,Type=Integer,Description=\"Number of reference observations on the forward strand\">\n";
    out << "##FORMAT=<ID=SRR,Number=1,Type=Integer,Description=\"Number of reference observations on the reverse strand\">\n";
    out << "##FORMAT=<ID=SAF,Number=A,Type=Integer,Description=\"Alternate allele observations on the forward strand\">\n";
    out << "##FORMAT=<ID=SAR,Number=A,Type=Integer,Description=\"Alternate allele observations on the reverse strand\">\n";
    out << "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator observation counts\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" << sample_manager->sample_names_[sample_manager->primary_sample_];
    for (int sample = 0; sample < sample_manager->num_samples_; ++sample)
      if (sample != sample_manager->primary_sample_)
        out << "\t" << sample_manager->sample_names_[sample];
    out <<"\n";
  }


  void IndelAssembly::AddCounts(BamAlignment& read) {
    int position_in_window = getSoftStart(read) - curLeft;
    int position_in_read = 0;

    if (position_in_window < 0) {
      //  out.println("A read started upfront window! Increase WINDOW_PREFIX. Read skipped.");
      return;
    }

    int strand = read.IsReverseStrand() ? 1 : 0;
    int sample;
    bool is_primary;
    if (!sample_manager->IdentifySample(read, sample, is_primary))
      return;
  
    bool after_match = false;
    for (vector<CigarOp>::const_iterator cigar = read.CigarData.begin(); cigar != read.CigarData.end(); ++cigar) {

      if (cigar->Type == 'S') {

        if (is_primary) {
          if (after_match) {
            for(int j = position_in_window; j <  position_in_window + min((int)cigar->Length,4*KMER_LEN) && j < WINDOW_SIZE; ++j)
              coverage[j].soft_clip[strand]++;
          } else {
            for(int j = position_in_window+max((int)cigar->Length - 4*KMER_LEN, 0); j < position_in_window + (int)cigar->Length && j < WINDOW_SIZE; ++j)
              coverage[j].soft_clip[strand]++;
          }
        }

        position_in_window += cigar->Length;
        position_in_read += cigar->Length;

      } else if (cigar->Type == 'I') {
        if (position_in_window < WINDOW_SIZE && is_primary)
          coverage[position_in_window].indel[strand]++;
        position_in_read += cigar->Length;

      } else if (cigar->Type == 'D' || cigar->Type == 'M' || cigar->Type == 'N') {
        if(cigar->Type == 'D' && position_in_window < WINDOW_SIZE && is_primary)
          coverage[position_in_window].indel[strand]++;

        for (int j = cigar->Length + position_in_window-1, k = position_in_read+cigar->Length-1; j>=position_in_window && k>=position_in_read && j<WINDOW_SIZE; j--,k--)
          coverage[j].total.Increment(strand, sample, sample_manager->num_samples_);

        position_in_window += cigar->Length;
        if(cigar->Type != 'D')
          position_in_read += cigar->Length;

        after_match = true;
      }
    }
  }
  
// -----------------------------------------------------------------------------
// The function takes a boolean signalling whether assembly output is desired

void IndelAssembly::onTraversalDone(bool do_assembly) {
  if (do_assembly)
    DetectCandidateRegions(WINDOW_SIZE);

  out.close();
}
  
