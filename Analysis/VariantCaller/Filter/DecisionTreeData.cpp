/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DecisionTreeData.h"

float FreqThresholdByType(AlleleIdentity &variant_identity, ControlCallAndFilters &my_controls) {
  float retval = my_controls.filter_snps.min_allele_freq;
  if (variant_identity.status.isHotSpot) {
    retval = my_controls.filter_hotspot.min_allele_freq;
  }
  else if (variant_identity.ActAsSNP()) {
    retval = my_controls.filter_snps.min_allele_freq;
  }
  else if (variant_identity.ActAsHPIndel()) {
    retval = my_controls.filter_hp_indel.min_allele_freq;
  }
  return(retval);
}


void DecisionTreeData::SetupFromMultiAllele(MultiAlleleVariantIdentity &_multi_allele) {
  multi_allele = _multi_allele;
  summary_stats_vector.resize(multi_allele.allele_identity_vector.size());
  summary_info_vector.resize(multi_allele.allele_identity_vector.size());
}

void DecisionTreeData::SetupSummaryStatsFromCandidate(vcf::Variant **candidate_variant) {
// mirror multiflowdist setup for good or ill
  vector<string> fwdRefObservation = (*candidate_variant)->info["SRF"];
  vector<string> revRefObservation = (*candidate_variant)->info["SRR"];
  vector<string> fwdAltObservation = (*candidate_variant)->info["SAF"];
  vector<string> revAltObservation = (*candidate_variant)->info["SAR"];
  int fwdRef = 0;
  int revRef = 0;
  int fwdDepth = 0;
  int revDepth = 0;
  vector<int> fwdAlt;
  vector<int> revAlt;
  uint8_t totalAlts = (*candidate_variant)->alt.size();

  if (fwdRefObservation.size() > 0)
    fwdRef = atoi(fwdRefObservation.at(0).c_str());

  if (revRefObservation.size() > 0)
    revRef = atoi(revRefObservation.at(0).c_str());

  if (fwdAltObservation.size() == totalAlts && revAltObservation.size() == totalAlts) {
    for (uint8_t i = 0; i < totalAlts; i++) {
      fwdAlt.push_back(atoi(fwdAltObservation.at(i).c_str()));
      revAlt.push_back(atoi(revAltObservation.at(i).c_str()));
    }
  }

  fwdDepth = fwdRef;
  revDepth = revRef;
  for (uint8_t i = 0; i < totalAlts; i++) {
    fwdDepth += fwdAlt.at(i);
    revDepth += revAlt.at(i);
  }

  summary_stats_vector.resize(totalAlts);
  for (uint8_t i = 0; i < totalAlts; i++) {
    summary_stats_vector[i].setBasePlusDepth(fwdDepth);
    summary_stats_vector[i].setBaseNegDepth(revDepth);
    summary_stats_vector[i].setBasePlusVariant(fwdAlt.at(i));
    summary_stats_vector[i].setBaseNegVariant(revAlt.at(i));
  }
  // everything else assumed to be set up
}

void DecisionTreeData::FilterReferenceCalls(int _allele) {
  // I cannot believe I have to do this to make validator happy
  // turn this off when we get gVCF setup!!!!
  if (summary_info_vector[_allele].genotype_call == 0) {
    summary_info_vector[_allele].isFiltered = true;
    summary_info_vector[_allele].filterReason  += "REFERENCECALL";
  }
};

void DecisionTreeData::FilterNoCalls(bool isNoCall, int _allele) {
  // I cannot believe I have to do this to make validator happy
  // turn this off when we get gVCF setup!!!!
  if (isNoCall) {
    summary_info_vector[_allele].isFiltered = true;
    summary_info_vector[_allele].filterReason = "NOCALL";
  }
};


void DecisionTreeData::FilterOnStrandBias(float threshold, int _allele) {
  if (summary_stats_vector[_allele].getStrandBias() > threshold) {
    summary_info_vector[_allele].isFiltered = true;
    stringstream filterReasonStr;
    filterReasonStr << "STDBIAS > ";
    filterReasonStr << threshold;
    summary_info_vector[_allele].filterReason = filterReasonStr.str();
  }
}

void DecisionTreeData::FilterOnMinimumCoverage(int min_cov_each_strand,  int _allele) {
  if (summary_stats_vector[_allele].getPlusDepth() < min_cov_each_strand || summary_stats_vector[_allele].getNegDepth() < min_cov_each_strand) {
    summary_info_vector[_allele].isFiltered = true;
    stringstream filterReasonStr;
    filterReasonStr << "MINCOVEACHSTRAND < " ;
    filterReasonStr << min_cov_each_strand;
    summary_info_vector[_allele].filterReason = filterReasonStr.str();
  }

}


void DecisionTreeData::FilterOnQualityScore(float min_quality_score, int _allele) {
  // can I reject the ref strongly

  if (summary_info_vector[_allele].alleleScore < min_quality_score) {
    summary_info_vector[_allele].isFiltered = true;
    stringstream filterReasonStr;
    filterReasonStr << "QUALITYSCORE<" ;
    filterReasonStr << min_quality_score;
    summary_info_vector[_allele].filterReason += filterReasonStr.str();
  }
}

void DecisionTreeData::DoFilter(ControlCallAndFilters &my_filters, int _allele) {
  // what things need to be done
  // and why
  //@TODO: should refer to control filters, but doesn't yet
  FilterReferenceCalls(_allele);
  FilterOnStrandBias(my_filters.filter_snps.strand_bias_threshold, _allele);
  FilterOnMinimumCoverage(my_filters.filter_snps.min_cov_each_strand, _allele);
  FilterOnQualityScore(my_filters.filter_snps.min_quality_score, _allele);
};


void DecisionTreeData::OverrideFilter(string & _filter_reason, int _allele) {
  // force a specific filtering operation based on some other data
  summary_info_vector[_allele].isFiltered = true;
  summary_info_vector[_allele].filterReason += _filter_reason;
}



float RetrieveRBITagValue(vcf::Variant *current_variant){
  
    map<string, vector<string> >::iterator it;
    float weight;
    
  it = current_variant->info.find("RBI");
  if (it != current_variant->info.end())
    weight = atof(current_variant->info.at("RBI")[0].c_str()); // or is this current sample ident?
  else weight = 0.0f;
  return(weight);
}


//@TODO: move this into DecisionTree and read from the tag BLL instead
void DecisionTreeData::SpecializedFilterFromBiasVariables(vcf::Variant *current_variant, float bias_threshold, int _allele) {
  float test_bias = 0.0f;
  
  if (bias_threshold<0)
    test_bias = 0.0f;  // obsolete
  else
    test_bias = RetrieveRBITagValue(current_variant);

  if (test_bias>bias_threshold) {
    stringstream filterReasonStr;
    filterReasonStr << "UNUSUALBIAS" ;
    filterReasonStr << test_bias;
    string my_tmp_string = filterReasonStr.str();
    OverrideFilter(my_tmp_string, _allele);
  }
}


// warning: no white-space allowed in filter reason
void FilterByBasicThresholds(stringstream &s, VariantBook &l_summary_stats,
                             VariantOutputInfo &l_summary_info,
                             BasicFilters &basic_filter, float tune_xbias) {

  if (l_summary_stats.getStrandBias() > basic_filter.strand_bias_threshold) {
    s << "STDBIAS" <<  l_summary_stats.getStrandBias() << ">" << basic_filter.strand_bias_threshold ;
    l_summary_info.isFiltered = true;
  }
  if (l_summary_stats.GetXBias(tune_xbias) > basic_filter.beta_bias_filter) {
    s << "XBIAS" <<  l_summary_stats.GetXBias(tune_xbias) << ">" << basic_filter.beta_bias_filter ;
    l_summary_info.isFiltered = true;
  }

// base strand bias would be a pre-filter, not a filter
  /*  if (l_summary_stats.getBaseStrandBias() > basic_filter.strand_bias_threshold) {
      s << "BaseSTDBIAS" << l_summary_stats.getBaseStrandBias() << " > " << basic_filter.strand_bias_threshold ;
      l_summary_info.isFiltered = true;
    }*/

// this is used in determining >genotype< which may be reference
// we should not filter out calls, especially in hotspots
  /*  if (l_summary_stats.getAltAlleleFreq() < basic_filter.min_allele_freq) {
      s << "ALLELEFREQ" << l_summary_stats.getAltAlleleFreq() <<  " < " << basic_filter.min_allele_freq ;
      l_summary_info.isFiltered = true;
    } */

  if (l_summary_info.alleleScore < basic_filter.min_quality_score) {
    s << "QualityScore<" << basic_filter.min_quality_score ;
    l_summary_info.isFiltered = true;
  }

  if (l_summary_stats.getPlusDepth() < basic_filter.min_cov_each_strand || l_summary_stats.getNegDepth() < basic_filter.min_cov_each_strand) {
    l_summary_info.isFiltered = true;
    s << "PosCov<" << l_summary_stats.getPlusDepth() << "NegCov<" << l_summary_stats.getNegDepth() ;
  }

  if (l_summary_stats.getDepth() < basic_filter.min_cov) {
    l_summary_info.isFiltered = true;
    s << "MINCOV<" << basic_filter.min_cov ;
  }
}


void DecisionTreeData::FilterOneAllele(VariantBook &l_summary_stats, VariantOutputInfo &l_summary_info, AlleleIdentity &l_variant_identity, ControlCallAndFilters &my_filters) {
  stringstream s;
  //check if the allele was prefiltered
  //cout << "is pre filtered = " << (*_summary_info).isFiltered << " is ref = " << _variant_identity.status.isReferenceCall << endl;
  //cout << " Qual score = " << (*_summary_info).alleleScore << endl;
  //cout << l_variant_identity.status.isSNP << " " << l_variant_identity.status.isIndel << " " << l_variant_identity.status.isHPIndel << endl;
//  if ((l_summary_info).isFiltered) {
    //DO WE NEED TO DO ANYTHING IF ALREADY FILTERED
//  } else {
    //common to all variant types
    
    // even if filtered already, apply later filters so we find out >all< of the filters.
    
    if (l_variant_identity.status.isReferenceCall) {
      s << "ReferenceCall" ;
      l_summary_info.isFiltered = true;
    }
    //cout << "inside else " << endl;
    //filter values specific to SNPs, MNVs and Non Homopolymer Indels
    if (l_variant_identity.status.isHotSpot) {
      // hot spot overrides
      FilterByBasicThresholds(s, l_summary_stats, l_summary_info, my_filters.filter_hotspot, tune_xbias);
    }
    else if (l_variant_identity.ActAsSNP()) {
      //cout << "inside snp flow " << endl;
      FilterByBasicThresholds(s, l_summary_stats, l_summary_info, my_filters.filter_snps, tune_xbias);

    }//end if SNP or MNV
    else
      if (l_variant_identity.ActAsHPIndel()) {

        FilterByBasicThresholds(s, l_summary_stats, l_summary_info, my_filters.filter_hp_indel, tune_xbias);

        // if we haven't already turned it into a no-call, check if we need to filter
        if ((l_variant_identity.ref_hp_length > my_filters.filter_variant.hp_max_length) and (l_variant_identity.status.isIndel) and (!l_variant_identity.status.isNoCallVariant)) {
          l_summary_info.isFiltered = true;
          s << "HOMOPOLYMERLENGTH" << l_variant_identity.ref_hp_length << ">" << my_filters.filter_variant.hp_max_length ;
        }
      }
    l_summary_info.filterReason += s.str(); // append reason(s) for filtration

//  } //end else

}

/* Method for FlowDistEvaluator branch to loop thru alleles and filter ones that fail the filter condition*/
void DecisionTreeData::FilterAlleles(ControlCallAndFilters &my_filters) {
  //cout << "In filter alleles " << endl;
  int numAlleles = summary_stats_vector.size();

  for (int i = 0; i < numAlleles; i++) {

    FilterOneAllele(summary_stats_vector[i], summary_info_vector[i], multi_allele.allele_identity_vector[i], my_filters);

  } //end loop thru alleles

}

void DecisionTreeData::AccumulateFilteredAlleles(){
   int numAlleles = summary_stats_vector.size();
  VariantBook _summary_stats;
  VariantOutputInfo _summary_info;
  AlleleIdentity _variant_identity;
   for (int i=0; i<numAlleles; i++){
     _summary_stats = summary_stats_vector.at(i);
    _summary_info = summary_info_vector.at(i);
    _variant_identity = multi_allele.allele_identity_vector.at(i);

    if (_summary_info.isFiltered)
      filteredAllelesIndex.push_back(i);
   }
}

string DecisionTreeData::AnyNoCallsMeansAllFiltered(){
   int numAlleles = summary_stats_vector.size();
  VariantBook _summary_stats;
  VariantOutputInfo _summary_info;
  AlleleIdentity _variant_identity;
  string noCallReason;
    for (int i=0; i<numAlleles; i++){
     _summary_stats = summary_stats_vector.at(i);
    _summary_info = summary_info_vector.at(i);
    _variant_identity = multi_allele.allele_identity_vector.at(i);
  
     if (_variant_identity.status.isNoCallVariant) {
      best_variant_filtered = true;
      noCallReason = _variant_identity.filterReason;
      
    }
  }
  return(noCallReason);
}

void DecisionTreeData::FindBestAlleleByScore(){
  
   int numAlleles = summary_stats_vector.size();
  VariantBook _summary_stats;
  VariantOutputInfo _summary_info;
  AlleleIdentity _variant_identity;
   best_allele_index = 0;
  float maxScore = 0.0;
  
   for (int i = 0; i < numAlleles; i++) {
    _summary_stats = summary_stats_vector.at(i);
    _summary_info = summary_info_vector.at(i);
    _variant_identity = multi_allele.allele_identity_vector.at(i);

    if (_summary_info.alleleScore > maxScore &&  !_summary_info.isFiltered) {
      best_allele_index = i;
      maxScore = _summary_info.alleleScore;

    }

  }
  best_allele_set = true;
}

void DecisionTreeData::BestSNPsSuppressInDels(){
  //now if the best allele is a SNP and one or more Indel alleles present at the same position
  //then remove all the indel alleles.
  //This is done mainly to represent SNPs at the exact position and not have to represent it as MNV.
  //EXAMPLE REF = CA Alt = C, CC. IF C->A SNP is true then we want to move the allele representation to REF = C, Alt = A which is a more standard representation.
  if (isBestAlleleSNP) {
    //loop thru all the alleles and filter all Indel alleles which will be later removed from alts.
   int numAlleles = summary_stats_vector.size();
  VariantBook _summary_stats;
  VariantOutputInfo _summary_info;
  AlleleIdentity _variant_identity;
    for (int counter = 0; counter < numAlleles; counter++) {
      _summary_info = summary_info_vector.at(counter);
      _variant_identity = multi_allele.allele_identity_vector.at(counter);


      if (_variant_identity.status.isIndel && !_summary_info.isFiltered) { //if it is Indel allele and not already filtered
        summary_info_vector.at(counter).isFiltered = true;
        filteredAllelesIndex.push_back(counter);
      }

    }
  }
}

void DecisionTreeData::FindBestAlleleIdentity(){
  
    AlleleIdentity _variant_identity = multi_allele.allele_identity_vector.at(best_allele_index);;
        if (_variant_identity.status.isSNP || _variant_identity.status.isMNV)
        isBestAlleleSNP = true;
      else
        isBestAlleleSNP = false;
}

void DecisionTreeData::DetectAllFiltered(){
    //if all alleles are filtered
  if (filteredAllelesIndex.size() == summary_stats_vector.size()) {
    best_variant_filtered = true;
    if (summary_info_vector.size() >= 1)
      best_filter_reason = summary_info_vector.at(0).filterReason;
  }
}

void DecisionTreeData::DetectBestAlleleFiltered(string &noCallReason){
    //finally if best_variant_filtered because of any one of the alt. alleles being a NOCALL set the best allele to  be filtered.
  if (best_variant_filtered) {
    summary_info_vector.at(best_allele_index).isFiltered = true;
    multi_allele.allele_identity_vector.at(best_allele_index).filterReason = noCallReason;
    multi_allele.allele_identity_vector.at(best_allele_index).status.isNoCallVariant = true;
  }
}

void DecisionTreeData::FindBestAllele() {
 
  AccumulateFilteredAlleles();
  
  string noCallReason = AnyNoCallsMeansAllFiltered();
  
  if (!best_allele_set)
    FindBestAlleleByScore();
     
  FindBestAlleleIdentity();

  BestSNPsSuppressInDels();

  DetectAllFiltered();

  DetectBestAlleleFiltered(noCallReason);

}

void DecisionTreeData::SetLocalGenotypeCallFromStats(float threshold) {
  VariantBook _summary_stats;
  VariantOutputInfo _summary_info;
  int numAltAlleles = summary_stats_vector.size();
  for (int i = 0; i < numAltAlleles; i++) {
    _summary_stats = summary_stats_vector.at(i);
    // if we >have not< set genotype call already
    if (summary_info_vector[i].genotype_call < 0) {
      summary_info_vector[i].genotype_call = _summary_stats.StatsCallGenotype(threshold); // should be parameter
    }
  }
}

 

void DecisionTreeData::InformationTagOnFilter(vcf::Variant ** candidate_variant, int _best_allele_index, string sampleName) {
  // store tagged filter quantities
  //only do best allele for this guy
  (*candidate_variant)->info["SSEP"].push_back(convertToString(multi_allele.allele_identity_vector[_best_allele_index].sse_prob_positive_strand));
  (*candidate_variant)->info["SSEN"].push_back(convertToString(multi_allele.allele_identity_vector[_best_allele_index].sse_prob_negative_strand));
  (*candidate_variant)->info["STB"].push_back(convertToString(summary_stats_vector[_best_allele_index].getStrandBias()));
  (*candidate_variant)->info["SXB"].push_back(convertToString(summary_stats_vector[_best_allele_index].GetXBias(tune_xbias)));  // variance zero = 0.1^2
  
  // depth by allele statements
  // complex with multialleles
  
  // each read is either an outlier, ref, or one of the alternates
  int total_depth;
  total_depth = summary_stats_vector[_best_allele_index].getRefAllele(); // same across all summary-stats objects
  for (unsigned int ia=0; ia<summary_stats_vector.size(); ia++){
    total_depth += summary_stats_vector[ia].getVarAllele();
  }
  
  (*candidate_variant)->info["FDP"].push_back(convertToString(total_depth));
  // ref is invariant across all alleles
  (*candidate_variant)->info["FRO"].push_back(convertToString(summary_stats_vector[_best_allele_index].getRefAllele()));
  (*candidate_variant)->info["FSRF"].push_back(convertToString(summary_stats_vector[_best_allele_index].getPlusRef()));
  (*candidate_variant)->info["FSRR"].push_back(convertToString(summary_stats_vector[_best_allele_index].getNegRef()));
  // alternate allele count varies by allele
  for (unsigned int ia=0; ia< summary_stats_vector.size(); ia++){
    
    (*candidate_variant)->info["FAO"].push_back(convertToString(summary_stats_vector[ia].getVarAllele()));
    (*candidate_variant)->info["FSAF"].push_back(convertToString(summary_stats_vector[ia].getPlusVariant()));
    (*candidate_variant)->info["FSAR"].push_back(convertToString(summary_stats_vector[ia].getNegVariant()));
  }
  
  if (!sampleName.empty()) {
      map<string, vector<string> >& sampleOutput = (*candidate_variant)->samples[sampleName];
      sampleOutput["FDP"].push_back(convertToString(total_depth));
      sampleOutput["FRO"].push_back(convertToString(summary_stats_vector[_best_allele_index].getRefAllele()));
      sampleOutput["FSRF"].push_back(convertToString(summary_stats_vector[_best_allele_index].getPlusRef()));
      sampleOutput["FSRR"].push_back(convertToString(summary_stats_vector[_best_allele_index].getNegRef()));
      for (unsigned int ia=0; ia< summary_stats_vector.size(); ia++){
        sampleOutput["FAO"].push_back(convertToString(summary_stats_vector[ia].getVarAllele()));
        sampleOutput["FSAF"].push_back(convertToString(summary_stats_vector[ia].getPlusVariant()));
        sampleOutput["FSAR"].push_back(convertToString(summary_stats_vector[ia].getNegVariant()));
      }
  }

  // hrun fill in
  ClearVal(*candidate_variant, "HRUN");
  for (unsigned int ia=0; ia<multi_allele.allele_identity_vector.size(); ia++){
    (*candidate_variant)->info["HRUN"].push_back(convertToString(multi_allele.allele_identity_vector[ia].ref_hp_length));
  }

};


// this is only complicated because
// we may prefilter the variant in some way
// or we may have multiple alleles
// in the case of a diploid, we should reduce nicely to 0/0, 0/1, 1/1

bool DecisionTreeData::SetGenotype(vcf::Variant ** candidate_variant, ExtendParameters *parameters, float gt_quality) {

  if (best_allele_index >= (int)summary_info_vector.size()) {
    cerr << "FATAL ERROR: Chosen Allele index is out of bounds - allele index = " << best_allele_index << " total number of alleles = " << summary_info_vector.size() << endl;
    exit(-1);
  }

  AlleleIdentity   _variant_identity = multi_allele.allele_identity_vector.at(best_allele_index);

  string genotype_string = GenotypeFromStatus(candidate_variant, parameters);
  StoreGenotypeForOneSample(candidate_variant, _variant_identity.status.isNoCallVariant, parameters->sampleName, genotype_string, gt_quality);

  if (genotype_string == "0/0") {
    return(true);
  }
  else
    return(false);
}

// prefilter for some statuses, then try to get the genotype string
string DecisionTreeData::GenotypeFromStatus(vcf::Variant **candidate_variant, ExtendParameters *parameters) {

  AlleleIdentity _variant_identity = multi_allele.allele_identity_vector.at(best_allele_index);
  //  _summary_stats = summary_stats_vector.at(best_allele_index);

  //if all alleles are filter set genotype to no call
  string genotypestring = "";

  if (best_variant_filtered & genotypestring.empty()) {
    genotypestring  = "./.";
  }

  if (_variant_identity.status.isNoCallVariant & genotypestring.empty())  {
    genotypestring  = "./.";
  }

  if (_variant_identity.status.isReferenceCall & genotypestring.empty()) {
    genotypestring = "0/0";
  }
  //now check for all possible genotype alleles

  if (genotypestring.empty()) {
    vector<int> allowedGenotypes;
    bool refAlleleFound = AllowedGenotypesFromSummary(allowedGenotypes);
    genotypestring = GenotypeStringFromAlleles(allowedGenotypes, refAlleleFound);
  }
  return(genotypestring);
}

// check across all alleles
bool DecisionTreeData::AllowedGenotypesFromSummary(std::vector<int> &allowedGenotypes) {

  bool refAlleleFound = false;
  VariantOutputInfo _summary_info;
  int numAltAlleles = summary_info_vector.size();

  //first check if Reference allele is present
  for (int i = 0; i < numAltAlleles; i++) {
    _summary_info = summary_info_vector.at(i);
    if (!_summary_info.isFiltered) { //if not filtered add to allowed genotype alleles
      // only push back if there >is< a alternate call here
      if (_summary_info.genotype_call > 0)
        allowedGenotypes.push_back(i + 1); //i+1 as first alt allele is numbered 1 and ref allele is numbered 0
      // if a reference call is made, add ref allele
      if (_summary_info.genotype_call < 2) // use summary info here because may supply genotype call from specialized routine elsewhere
        refAlleleFound = true;
    }
  }
  if (refAlleleFound)
    allowedGenotypes.insert(allowedGenotypes.begin(), 0); //add reference to front of allowed genotypes.
  return(refAlleleFound);
}

// manufacture from possibly complicated allele structure
string DecisionTreeData::GenotypeStringFromAlleles(std::vector<int> &allowedGenotypes, bool refAlleleFound) {
  //cout << " Allowed Genotypes size = " << allowedGenotypes.size() << " ref allele found = " << refAlleleFound << endl;
  //cout << " Genotype allele = " << allowedGenotypes.at(0) << endl;
  //HERE WE ASSUME A DIPLOID GENOME
  stringstream genotypestream;
  if (allowedGenotypes.size() > 2) {
    //set the genotype based on best allele found
    if (refAlleleFound)
      genotypestream << 0 << "/" << best_allele_index + 1; //+1 is used since first alt allele is numbered 1 in VCF
    else
      genotypestream << best_allele_index + 1 << "/" << best_allele_index + 1;
  }
  else { //size of allowed genotypes is either 1 or 2.
    if (allowedGenotypes.size() == 1)
      genotypestream << allowedGenotypes.at(0) << "/" << allowedGenotypes.at(0);
    else
      if (allowedGenotypes.size() == 2)
        genotypestream << allowedGenotypes.at(0) << "/" << allowedGenotypes.at(1);
      else {
        cerr << "FATAL ERROR: Invalid number of allowed alleles found - " << allowedGenotypes.size() << endl;
        exit(-1);
      }
  }
  return(genotypestream.str());
}

void DecisionTreeData::StoreMaximumAlleleInVariants(vcf::Variant ** candidate_variant, ExtendParameters *parameters) {
  VariantBook _summary_stats;
  VariantOutputInfo _summary_info;
  AlleleIdentity _variant_identity;
  int bestAlleleIndex = best_allele_index;
  if (bestAlleleIndex >= (int)summary_stats_vector.size()) {
    cerr << "FATAL ERROR: Chosen Allele index is out of bounds - allele index = " << bestAlleleIndex << " total number of alleles = " << summary_stats_vector.size() << endl;
    exit(-1);
  }
  _summary_stats = summary_stats_vector.at(bestAlleleIndex);
  _summary_info = summary_info_vector.at(bestAlleleIndex);
  _variant_identity = multi_allele.allele_identity_vector.at(bestAlleleIndex);

  (*candidate_variant)->quality = _summary_info.alleleScore;
  float gt_quality = _summary_info.gt_quality_score;
  //cout << "Best Allele Index = " << bestAlleleIndex << " summary_info is filtered = " << _summary_info.isFiltered << endl;

  SetFilteredStatus(candidate_variant, _variant_identity.status.isNoCallVariant, _summary_info.isFiltered, parameters->my_controls.suppress_no_calls);
  //InsertBayesianScoreTag(candidate_variant, flowDist->summary_info.alleleScore);
  InsertGenericInfoTag(candidate_variant, _variant_identity.status.isNoCallVariant, _variant_identity.filterReason, _summary_info.filterReason);


  //still need to set genotype

  bool reference_genotype = SetGenotype(candidate_variant, parameters, gt_quality);

  // and maybe suppress unwanted reference calls
  if (reference_genotype & !_variant_identity.status.isHotSpot & parameters->my_controls.suppress_reference_genotypes) {
    SetFilteredStatus(candidate_variant, _variant_identity.status.isNoCallVariant, true, parameters->my_controls.suppress_no_calls); // used to suppress reference genotypes if not hot spot
    string my_suppression_reason = _summary_info.filterReason + "SUPPRESSREFERENCECALL";
    InsertGenericInfoTag(candidate_variant, _variant_identity.status.isNoCallVariant, _variant_identity.filterReason, my_suppression_reason);
  }
}

// this only needs to know candidate variant, nothing else
void AdjustAlleles(vcf::Variant ** candidate_variant) {
  vector<string> types = (*candidate_variant)->info["TYPE"];
  string refAllele = (*candidate_variant)->ref;
  vector<string> alts = (*candidate_variant)->alt;
  long int position = (*candidate_variant)->position;
  bool snpPosFound = false;
  string altAllele = alts.at(0);
  string newRefAllele;
  string newAltAllele;
  //nothing to do if there are multiple allels

  if (types.size() != 1)
    return;
  else {
    if ((types.at(0)).compare("snp") == 0 && refAllele.length() > 1 && refAllele.length() == altAllele.length())  {
      //need to adjust position only in cases where SNP is represent as MNV due to haplotyping - REF= TTC ALT = TTT
      for (size_t i = 0; i < refAllele.length(); i++) {
        if (refAllele.at(i) != altAllele.at(i)) {
          snpPosFound = true;
          newRefAllele = refAllele.substr(i, 1);
          newAltAllele = altAllele.substr(i, 1);
          break;
        }
        position++;
      }
      //change the ref and alt allele and position of the variant to get to a more traditional snp representation
      if (snpPosFound) {
        (*candidate_variant)->position = position;
        (*candidate_variant)->ref = newRefAllele;
        (*candidate_variant)->alt.at(0) = newAltAllele;
      }

    }
  }

}


void AdjustFDPForRemovedAlleles(vcf::Variant ** candidate_variant, int filtered_allele_index, string sampleName){
  
  // first do the "info" tag as it is easier to find
    map<string, vector<string> >::iterator it;
    vcf::Variant *current_variant = *candidate_variant;
    int total_depth=0;
    
  it = current_variant->info.find("FDP");
  if (it != current_variant->info.end())
    total_depth = atoi(current_variant->info.at("FDP")[0].c_str()); // or is this current sample ident?
  
  int allele_depth = 0;
  it = current_variant->info.find("FAO");
  if (it != current_variant->info.end())
    allele_depth = atoi(current_variant->info.at("FAO")[filtered_allele_index].c_str()); 

  total_depth -= allele_depth;
  if (total_depth<0)
    total_depth = 0; // how can this happen?  
  
  ClearVal(*candidate_variant, "FDP");
  (*candidate_variant)->info["FDP"].push_back(convertToString(total_depth));
  
  if (!sampleName.empty()) {
      map<string, vector<string> >& sampleOutput = (*candidate_variant)->samples[sampleName];
      sampleOutput["FDP"].clear();
      sampleOutput["FDP"].push_back(convertToString(total_depth));
  }
}


void DecisionTreeData::RemoveFilteredAlleles(vcf::Variant ** candidate_variant, string &sample_name) {
  //now that all possible alt. alleles are evaluated decide on which allele is most likely and remove any that
  //that does'nt pass score threshold. Determine Genotype based on alleles that have evidence.
  (*candidate_variant)->updateAlleleIndexes();
  vector<string> originalAltAlleles = (*candidate_variant)->alt;
  if (summary_stats_vector.size() > 1  &&
      summary_stats_vector.size() > filteredAllelesIndex.size()  //remove only when number of alleles more than number of filtered alleles
      && !(*candidate_variant)->isHotSpot) { //dont remove alleles if it is a HOT SPOT position as alleles might have been provided by the user.
    //remove filtered alleles with no support
    string altStr;
    int index;
    for (size_t i = 0; i < filteredAllelesIndex.size(); i++) {
            
      index = filteredAllelesIndex.at(i);
      //generate allele index before removing alleles
      altStr = originalAltAlleles[index];
      //altStr = (*candidate_variant)->alt[index];
      // Note: need to update index for adjustments
      //AdjustFDPForRemovedAlleles(candidate_variant, index, sample_name);
      //cout << "Removed Fitered allele: index = " << index << " allele = " << altStr << endl;
      (*candidate_variant)->removeAlt(altStr);
      (*candidate_variant)->updateAlleleIndexes();
    }
  }
}

// once we have standard data format across all alleles, the common filters can execute.
void DecisionTreeData::DecisionTreeOutputToVariant(vcf::Variant ** candidate_variant, ExtendParameters *parameters) {
  FilterAlleles(parameters->my_controls);
  FindBestAllele();
  InformationTagOnFilter(candidate_variant, best_allele_index, parameters->sampleName);
  StoreMaximumAlleleInVariants(candidate_variant, parameters);
  if (!best_variant_filtered && isBestAlleleSNP) { //currently we are removing other filtered alleles if the best allele is a SNP
    RemoveFilteredAlleles(candidate_variant, parameters->sampleName);
    AdjustAlleles(candidate_variant);
  }
  FilterOnInformationTag(candidate_variant,parameters->my_controls.data_quality_stringency, parameters->my_controls.suppress_no_calls);
}


void AutoFailTheCandidate(vcf::Variant **candidate_variant, bool suppress_no_calls) {
  (*candidate_variant)->quality = 0.0f;
  NullInfoFields(*candidate_variant); // no information, destroy any spurious entries, add all needed tags
  SetFilteredStatus(candidate_variant, true, true, suppress_no_calls); // is this correct?
  string my_reason = "NODATA";
  bool isNoCall = false;
  string noCallReason = ".";
  InsertGenericInfoTag(candidate_variant, isNoCall, noCallReason, my_reason);
  NullGenotypeAllSamples(candidate_variant);
};

float RetrieveQualityTagValue(vcf::Variant *current_variant){
  
    map<string, vector<string> >::iterator it;
    float weight;
    
  it = current_variant->info.find("MLLD");
  if (it != current_variant->info.end())
    weight = atof(current_variant->info.at("MLLD")[0].c_str()); // or is this current sample ident?
  else weight = 0.0f;
  return(weight);
}

// fill this in with whatever we are really going to filter
void FilterOnInformationTag(vcf::Variant **candidate_variant, float data_quality_stringency, bool suppress_no_calls) {
  //(*candidate_variant)->info["MXFD"]
  string noCallReason = "";
  float filter_on_min_quality = RetrieveQualityTagValue(*candidate_variant);
  // turn this off right now
  if ((data_quality_stringency > filter_on_min_quality) & true) {
    (*candidate_variant)->quality = 0.0f;
    SetFilteredStatus(candidate_variant, true, true, suppress_no_calls);
    string my_reason = "STRINGENCY";
    InsertGenericInfoTag(candidate_variant, false, noCallReason , my_reason);
  }
}
