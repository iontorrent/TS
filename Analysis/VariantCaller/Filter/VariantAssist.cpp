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
  if (summary_stats.getStrandBias() > my_controls.filter_hp_indel.strand_bias_threshold) {
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


float VariantBook::getStrandBias() {
  strandBias = ComputeStrandBias(plusVariant,getPlusDepth(),negVariant,getNegDepth());
  
/*  long int num = max((long int)plusVariant*getNegDepth(), (long int)negVariant*getPlusDepth());
  long int denum = (long int)plusVariant*getNegDepth() + (long int)negVariant*getPlusDepth();
  if (denum == 0)
    strandBias = 1.0;
  else {
    strandBias = (float)num/denum;
  }
  if (DEBUG) {
    cout << "STDBIAS Calculation : RefPosition = " << "refPosition" << " PlusVariant = " << plusVariant << " NegVariant = " << negVariant << "Depth = " << getDepth() << " Plus Depth = " << getPlusDepth() << " Neg Depth = " << getNegDepth() ;
    cout << " Numerator = " << num << " denominator = " << denum << " stdbias = " << strandBias << " PlusMean = " << getPlusMean() << " NegMean = " << getNegMean() << endl;
  }*/
  return strandBias;
}

float VariantBook::getBaseStrandBias() {
  float stdBias = ComputeStrandBias(plusBaseVariant,getPlusBaseDepth(),negBaseVariant,getNegBaseDepth());
/*  float stdBias = 0;
  long int num =  max((long int)plusBaseVariant*getNegBaseDepth(), (long int)negBaseVariant*getPlusBaseDepth());
  long int denum = (long int)(plusBaseVariant)*getNegBaseDepth() + (long int)(negBaseVariant)*getPlusBaseDepth();
  if (denum == 0)
    stdBias = 1.0;
  else {
    stdBias = (float)num/denum;
  }
  if (DEBUG) {
    cout << "Basespace STDBIAS Calculation : RefPosition = " << "refPosition" << " PlusBaseVariant = " << plusBaseVariant << " NegBaseVariant = " << negBaseVariant << "Depth = " << getDepth() << " Plus Depth = " << getPlusBaseDepth() << " Neg Depth = " << getNegBaseDepth() ;
    cout << " Numerator = " << num << " denominator = " << denum << " stdbias = " << stdBias << " PlusMean = " << getPlusMean() << " NegMean = " << getNegMean() << endl;
  }*/
  return stdBias;
}

float VariantBook::getRefStrandBias() {
  float refBias = ComputeStrandBias(getPlusDepth()-plusVariant,getPlusDepth(),getNegDepth()-negVariant,getNegDepth());
/*  float refBias = 0;
  long int num = max((long int)(getPlusDepth()-plusVariant)*getNegDepth(), (long int)(getNegDepth()-negVariant)*getPlusDepth());
  long int denum = (long int)(getPlusDepth()-plusVariant)*getNegDepth() + (long int)(getNegDepth()-negVariant)*getPlusDepth();
  if (denum == 0)
    refBias = 1.0;
  else {
    refBias = (float)num/denum;
  }
  if (DEBUG) {
    cout << "REF STDBIAS Calculation : RefPosition = " << "refPosition" << " PlusVariant = " << plusVariant << " NegVariant = " << negVariant << "Depth = " << getDepth() << " Plus Depth = " << getPlusDepth() << " Neg Depth = " << getNegDepth() ;
    cout << " Numerator = " << num << " denominator = " << denum << " stdbias = " << refBias << " PlusMean = " << getPlusMean() << " NegMean = " << getNegMean() << endl;
  }*/
  return refBias;

}

float VariantBook::GetXBias(float var_zero){
  return(ComputeXBias(plusVariant,getPlusDepth(),negVariant,getNegDepth(), var_zero));
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
