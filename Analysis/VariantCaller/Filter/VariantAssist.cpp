/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VariantAssist.cpp
//! @ingroup  VariantCaller
//! @brief    Helpful routines for output of variants

#include "VariantAssist.h"

void InsertBayesianScoreTag(vcf::Variant ** candidate_variant, float BayesianScore) {
  stringstream bayss;
  bayss << BayesianScore;
  vector<string> bayInfoString(1);
  bayInfoString.push_back(bayss.str());
  pair<map<string, vector<string> >::iterator, bool> ret;
  ret = (*candidate_variant)->info.insert(pair<string,vector<string> >("Bayesian_score",bayInfoString));
  if (ret.second == false) {
    cerr << "ERROR: Failed to Insert INFO tag Bayesian score in VCF" << endl;
    //exit(-1);
  }
}

void InsertGenericInfoTag(vcf::Variant ** candidate_variant, bool isNoCallVariant, string &noCallReason, string &infoss) {
  //vector<string> hpInfoString(1);
  //hpInfoString.push_back(infoss);
  if (isNoCallVariant){
    (*candidate_variant)->info["FR"].push_back(noCallReason);
  }
  // even if .nocall. append other reasons for this occurrence to be filtered
    // always have an FR tag no matter what to make parsing easy
    if (infoss.empty() ) infoss = "."; //make sure string is not empty even if the variant is not filtered
      (*candidate_variant)->info["FR"].push_back(infoss);
}

// if, for example, missing data
void NullGenotypeAllSamples(vcf::Variant ** candidate_variant){
    vector<string> sampleNames = (*candidate_variant)->sampleNames;

  for (vector<string>::iterator its = sampleNames.begin(); its != sampleNames.end(); ++its) {
    string& sampleName = *its;
    map<string, vector<string> >& sampleOutput = (*candidate_variant)->samples[sampleName];
      sampleOutput["GT"].push_back("./.");
      sampleOutput["GQ"].push_back(convertToString(0));
  }
}


void StoreGenotypeForOneSample(vcf::Variant ** candidate_variant, bool isNoCall, string &my_sample_name, string &my_genotype, float genotype_quality) {
  vector<string> sampleNames = (*candidate_variant)->sampleNames;

  for (vector<string>::iterator its = sampleNames.begin(); its != sampleNames.end(); ++its) {
    string& sampleName = *its;
    //cout << "VariantAssist: SampleName = " << sampleName << " my_sample = " << my_sample_name << endl;
    map<string, vector<string> >& sampleOutput = (*candidate_variant)->samples[sampleName];
  map<string, vector<string> >::iterator it;
  it = sampleOutput.find("GT");
  if (it != sampleOutput.end())    sampleOutput["GT"].clear();
  it = sampleOutput.find("GQ");
   if (it != sampleOutput.end())     sampleOutput["GQ"].clear();
   
    if (sampleName.compare(my_sample_name) == 0) { //sample of interest
      //cout << "isNocall " << isNoCall << " genotype = " << my_genotype << endl;
      if (isNoCall) {
        sampleOutput["GT"].push_back("./.");
        sampleOutput["GQ"].push_back(convertToString(100));
        //cout << "Storing No-call Genotype = " << "./." << endl;
      } else {
        sampleOutput["GT"].push_back(my_genotype);
        //cout << "Storing Genotype = " << my_genotype << endl;
        sampleOutput["GQ"].push_back(convertToString(genotype_quality));
      }
    } else { //for all other samples in BAM file just make a no-call at this point.
      sampleOutput["GT"].push_back("./.");
      sampleOutput["GQ"].push_back(convertToString(0));
    }
    //cout <<"VariantAssist: total genotypes = " << sampleOutput["GT"].size() << endl;
  }

}

void SetFilteredStatus(vcf::Variant ** candidate_variant, bool isNoCall, bool isFiltered, bool suppress_no_calls) {
  if (isNoCall ) {
    (*candidate_variant)->filter = "NOCALL" ;
    (*candidate_variant)->isFiltered = suppress_no_calls;
  } else {
    if (isFiltered) {
      (*candidate_variant)->filter = "FAIL" ;
      (*candidate_variant)->isFiltered = true;
    } else {
      (*candidate_variant)->filter = "PASS";
      (*candidate_variant)->isFiltered = false;
    }
  }
}

bool PrefilterSummaryStats(VariantBook &summary_stats, ControlCallAndFilters &my_controls, bool *isFiltered, string *filterReason, stringstream &infoss) {
  stringstream filterReasonStr;
  if (summary_stats.getStrandBias(my_controls.sbias_tune) > my_controls.filter_hp_indel.strand_bias_threshold) {
    *isFiltered = true;
    filterReasonStr << "STDBIAS>";
    filterReasonStr << my_controls.filter_hp_indel.strand_bias_threshold;
    //infoss << "FAIL-" << *filterReason;
    //return(true);
  }
  if (summary_stats.getPlusDepth() < my_controls.filter_hp_indel.min_cov_each_strand || summary_stats.getNegDepth() < my_controls.filter_hp_indel.min_cov_each_strand) {
    *isFiltered = true;
    filterReasonStr << "MINCOVEACHSTRAND<" ;
    filterReasonStr << my_controls.filter_hp_indel.min_cov_each_strand;
    //infoss << "FAIL-" << *filterReason;
    //return(true);
  }

  if (summary_stats.getAlleleFreq() < my_controls.filter_hp_indel.min_allele_freq/2) { //prefilter only when allele freq is less than half the min req else allow evaluation to proceed
    *isFiltered = true;
    filterReasonStr << "MinAlleleFreq<";
    filterReasonStr << my_controls.filter_hp_indel.min_allele_freq;
  }

  if (summary_stats.getDepth() < my_controls.filter_hp_indel.min_cov) {
    *isFiltered = true;
    filterReasonStr << "MINCOV<";
    filterReasonStr << my_controls.filter_hp_indel.min_cov;
    //infoss << "FAIL-" << *filterReason;
    //return(true);
  }

  *filterReason = filterReasonStr.str();
  if (*isFiltered)
    infoss << "FAIL: " << *filterReason;
  // hit no prefilters
  return(*isFiltered);
}

uint16_t VariantBook::getDepth() {
  return depth;
};

uint16_t VariantBook::getRefAllele(){
  return depth-plusVariant-negVariant;
};

uint16_t VariantBook::getVarAllele(){
  return plusVariant+negVariant;
};

uint16_t VariantBook::getPlusDepth() {
  return plusDepth;
};

uint16_t VariantBook::getNegDepth() {
  return (depth-plusDepth);
};

uint16_t VariantBook::getPlusBaseVariant() {
  return plusBaseVariant;
};

uint16_t VariantBook::getNegBaseVariant() {
  return negBaseVariant;
};

uint16_t VariantBook::getPlusRef(){
  return(plusDepth-plusVariant);
}

uint16_t VariantBook::getNegRef(){
  return(depth-plusDepth-negVariant);
}

uint16_t VariantBook::getPlusBaseDepth() {
  return plusBaseDepth;
};

uint16_t VariantBook::getNegBaseDepth() {
  return negBaseDepth;
};

void VariantBook::setBasePlusDepth(uint16_t _plusBaseDepth) {
  plusBaseDepth = _plusBaseDepth;
};

void VariantBook::setBaseNegDepth(uint16_t _negBaseDepth) {
  negBaseDepth = _negBaseDepth;
};

void VariantBook::setBasePlusVariant(uint16_t _plusBaseVariant) {
  plusBaseVariant = _plusBaseVariant;
};

void VariantBook::setBaseNegVariant(uint16_t _negBaseVariant) {
  negBaseVariant = _negBaseVariant;
};

void VariantBook::incrementPlusMean(int flowValue) {
  plusMean += flowValue;
};

void VariantBook::incrementNegMean(int flowValue) {
  negMean += flowValue;
};

double VariantBook::getPlusMean() {
  if (plusDepth ==0)
    return 0;

  return ((float)plusMean)/plusDepth;
};
void VariantBook::setDepth(uint16_t dep) {
  depth = dep;
};

void VariantBook::incrementDepth() {
  depth++;
};

void VariantBook::incrementPlusDepth() {
  plusDepth++;
};

void VariantBook::setPlusDepth(uint16_t dep) {
  plusDepth = dep;
};

double VariantBook::getNegMean() {
  if (negMean == 0)
    return 0;

  return ((float)negMean)/(depth-plusDepth);
};
void VariantBook::incrementPlusVariant() {
  plusVariant++;
}

uint16_t VariantBook::getPlusVariant() {
  return plusVariant;
}

void VariantBook::incrementNegVariant() {
  negVariant++;
}

uint16_t VariantBook::getNegVariant() {
  return negVariant;
}

float VariantBook::getAlleleFreq() {
  if (depth == 0)
    return 0.0;
  else
    return ((float)(plusVariant+negVariant))/depth;
}

// pretend beta distribution on each strand
// pretend quantity of interest is variant-frequency difference between strands
// use variance from beta to determine significance
// but we don't care about small deviations even if significant, so add a default variance to prevent going to infinity
// do the usual mild prior to avoid blowing up at zero observations
float ComputeXBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float var_zero){
  float mean_plus = (plus_var+0.5f)/(plus_depth+1.0f);
  float mean_minus = (neg_var+0.5f)/(neg_depth+1.0f);
  float var_plus = (plus_var+0.5f)*(plus_depth-plus_var+0.5f)/((plus_depth+1.0f)*(plus_depth+1.0f)*(plus_depth+2.0f));
  float var_minus = (neg_var+0.5f)*(neg_depth-neg_var+0.5f)/((neg_depth+1.0f)*(neg_depth+1.0f)*(neg_depth+2.0f));
  
  // squared difference in mean frequency, divided by variance of quantity, inflated by potential minimal variance
  return((mean_plus-mean_minus)*(mean_plus-mean_minus)/(var_plus+var_minus+var_zero));
}

float ComputeTunedXBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float proportion_zero){
  float mean_plus = (plus_var+0.5f)/(plus_depth+1.0f);
  float mean_minus = (neg_var+0.5f)/(neg_depth+1.0f);
  float var_plus = (plus_var+0.5f)*(plus_depth-plus_var+0.5f)/((plus_depth+1.0f)*(plus_depth+1.0f)*(plus_depth+2.0f));
  float var_minus = (neg_var+0.5f)*(neg_depth-neg_var+0.5f)/((neg_depth+1.0f)*(neg_depth+1.0f)*(neg_depth+2.0f));
  
  float mean_zero = (plus_var+neg_var+0.5f)/(plus_depth+neg_depth+1.0f);
  float var_zero = proportion_zero*mean_zero*proportion_zero*(1.0f-mean_zero); // variance proportional to frequency to handle near-somatic cases
  
  // squared difference in mean frequency, divided by variance of quantity, inflated by potential minimal variance
  return((mean_plus-mean_minus)*(mean_plus-mean_minus)/(var_plus+var_minus+var_zero));
}


float ComputeStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth){
  float strand_bias;
    long int num = max((long int)plus_var*neg_depth, (long int)neg_var*plus_depth);
  long int denum = (long int)plus_var*neg_depth + (long int)neg_var*plus_depth;
  if (denum == 0)
    strand_bias = 0.0f; //if there is no coverage on one of the strands then there is no bias.
  else {
    strand_bias = (float)num/denum;
  }
  return(strand_bias);
}

float ComputeTransformStrandBias(long int plus_var, long int plus_depth, long int neg_var, long int neg_depth, float tune_fish){
  // trying to compute f_plus/f_neg
  //  transform to (1-x)/(1+x) to bring range between -1,1
  // take absolute value for filtering
  // tune_fish shoudl be something like 0.5 on standard grounds
  if (tune_fish<0.001f)
    tune_fish = 0.001f;
  
  // need to handle safety factor properly
  // "if we had tune_fish extra alleles of both reference and alternate for each strand, how would we distribute them across the strands preserving the depth ratio we see"
  // because constant values get hit by depth ratio when looking at 0/0
  // to give tune_fish expected alleles on each strand, given constant depth
  float relative_tune_fish = 1.0f- (plus_depth+neg_depth+2.0f*tune_fish)/(plus_depth+neg_depth+4.0f*tune_fish);
  
  // expected extra alleles to see on positive and negative strand
  float expected_positive = relative_tune_fish * plus_depth; 
  float expected_negative = relative_tune_fish * neg_depth;  
  
  // bias calculation based on observed counts plus expected safety level
  float pos_val = (plus_var + expected_positive) * (neg_depth + 2.0f*expected_negative);
  float neg_val = (neg_var + expected_negative) * (plus_depth + 2.0f*expected_positive);
  float strand_bias = (pos_val - neg_val)/(pos_val + neg_val+0.001f);  // what if depth on one strand is 0, then of course we can't detect any bias
  
//  cout << plus_var << "\t" << plus_depth << "\t" << neg_var << "\t" << neg_depth << "\t" << tune_fish << "\t" << strand_bias << endl;
  
  return(strand_bias);
}


// revised strand bias
// runs from 0 = unbiased
// to 1 = total bias to one strand or the other
// with a safety value this time to prevent "1 allele" from causing enormous strand bias
float VariantBook::getStrandBias(float tune_bias){
  float full_strand_bias = fabs(ComputeTransformStrandBias(plusVariant, getPlusDepth(), negVariant, getNegDepth(), tune_bias));
  //@TODO: revert to full range parameters
  // this ugly transformation makes the range 0.5-1, just like the old strand bias
  // so we don't have to change parameter files
  float old_style_strand_bias = (full_strand_bias+1.0f)/2.0f;
  return(old_style_strand_bias);
}


/*float VariantBook::getStrandBias() {
  strandBias = ComputeStrandBias(plusVariant,getPlusDepth(),negVariant,getNegDepth());
  
  return strandBias;
}*/

float VariantBook::getBaseStrandBias() {
  float stdBias = ComputeStrandBias(plusBaseVariant,getPlusBaseDepth(),negBaseVariant,getNegBaseDepth());

  return stdBias;
}

float VariantBook::getRefStrandBias() {
  float refBias = ComputeStrandBias(getPlusDepth()-plusVariant,getPlusDepth(),getNegDepth()-negVariant,getNegDepth());

  return refBias;

}

float VariantBook::GetXBias(float tune_xbias){
  return(ComputeTunedXBias(plusVariant,getPlusDepth(),negVariant,getNegDepth(), tune_xbias));
}

void VariantBook::setAltAlleleFreq(float altFreq) {
  altAlleleFreq = altFreq;
  isAltAlleleFreqSet = true;
}

float VariantBook::getAltAlleleFreq() {
  if (depth == 0)
    return 0;
  else
    if (isAltAlleleFreqSet)
      return altAlleleFreq;

  return ((float)(plusVariant+negVariant))/depth;
}

int VariantBook::StatsCallGenotype(float threshold) {

  float altAlleleFreq = getAltAlleleFreq();
  if ((altAlleleFreq>=threshold) & (altAlleleFreq<(1.0f-threshold)))
    return 1;
  if (altAlleleFreq>=(1.0f-threshold))
    return(2);
  return(0);
}

void VariantBook::UpdateSummaryStats(bool strand, bool variant_evidence, int tracking_val) {
  incrementDepth();
  if (strand) {
    incrementPlusDepth();
    incrementPlusMean(tracking_val);
    if (variant_evidence)
      incrementPlusVariant();
  } else {
    //flowDist->incrementNegDepth();
    incrementNegMean(tracking_val);
    if (variant_evidence)
      incrementNegVariant();
  }
}
