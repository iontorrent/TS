/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DecisionTreeData.h"

void AutoFailTheCandidate(vcf::Variant **candidate_variant, bool suppress_no_calls) {
  (*candidate_variant)->quality = 0.0f;
  NullInfoFields(*candidate_variant); // no information, destroy any spurious entries, add all needed tags
  NullGenotypeAllSamples(candidate_variant);
  NullFilterReason(candidate_variant);
  string my_reason = "NODATA";
  AddFilterReason(candidate_variant, my_reason);
  SetFilteredStatus(candidate_variant, true);
}



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


string EvaluatedGenotype::GenotypeAsString(){
  stringstream tmp_g;
  tmp_g << genotype_component.at(0) << "/" << genotype_component.at(1);
  return(tmp_g.str());
}

bool EvaluatedGenotype::IsReference(){
  if ((genotype_component.at(0)==0) & (genotype_component.at(1)==0))
    return(true);
  else
    return(false);
}



void PushAlleleCountsOntoStringMaps(map<string, vector<string> >& my_map, MultiBook &all_summary_stats){
  my_map["FDP"].push_back(convertToString(all_summary_stats.TotalCount(-1)));

  my_map["FRO"].push_back(convertToString(all_summary_stats.GetAlleleCount(-1,0)));
  my_map["FSRF"].push_back(convertToString(all_summary_stats.GetAlleleCount(0,0)));
  my_map["FSRR"].push_back(convertToString(all_summary_stats.GetAlleleCount(1,0)));

  // alternate allele count varies by allele
  for ( int i_alt=0; i_alt< all_summary_stats.NumAltAlleles(); i_alt++){

    my_map["FAO"].push_back(convertToString(all_summary_stats.GetAlleleCount(-1,i_alt+1)));
    my_map["FSAF"].push_back(convertToString(all_summary_stats.GetAlleleCount(0,i_alt+1)));
    my_map["FSAR"].push_back(convertToString(all_summary_stats.GetAlleleCount(1,i_alt+1)));
  }
}

float ComputeBaseStrandBiasForSSE(float relative_safety_level, vcf::Variant ** candidate_variant, unsigned _altAlleleIndex){
  unsigned alt_counts_positive = atoi((*candidate_variant)->info.at("SAF")[_altAlleleIndex].c_str());
  unsigned alt_counts_negative = atoi((*candidate_variant)->info.at("SAR")[_altAlleleIndex].c_str());
  // always only one ref count
 unsigned ref_counts_positive = atoi((*candidate_variant)->info.at("SRF")[0].c_str());
 unsigned ref_counts_negative = atoi((*candidate_variant)->info.at("SRR")[0].c_str());

  // remember to trap zero-count div by zero here with safety value
  float safety_val = 0.5f;  // the usual "half-count" to avoid zero
  unsigned total_depth = alt_counts_positive + alt_counts_negative + ref_counts_positive + ref_counts_negative;
  float relative_safety_val = safety_val + relative_safety_level * total_depth;

  float strand_ratio = ComputeTransformStrandBias(alt_counts_positive, alt_counts_positive+ref_counts_positive, alt_counts_negative, alt_counts_negative+ref_counts_negative, relative_safety_val);

  return(strand_ratio);
}


void SuppressReferenceCalls(vcf::Variant **candidate_variant, ExtendParameters *parameters, bool reference_genotype){
  if (reference_genotype & !(*candidate_variant)->isHotSpot & parameters->my_controls.suppress_reference_genotypes) {
    SetFilteredStatus(candidate_variant, true); // used to suppress reference genotypes if not hot spot
    string my_suppression_reason = "SUPPRESSREFERENCECALL";
    AddFilterReason(candidate_variant, my_suppression_reason);
  }
}

//*** Below here is actual decision tree data ***/


void DecisionTreeData::SetupFromMultiAllele(MultiAlleleVariantIdentity &_multi_allele) {
  multi_allele = _multi_allele;
//  summary_stats_vector.resize(multi_allele.allele_identity_vector.size());
  all_summary_stats.Allocate(multi_allele.allele_identity_vector.size()+1); // ref plus num alternate alleles
  summary_info_vector.resize(multi_allele.allele_identity_vector.size());
}

void DecisionTreeData::OverrideFilter(string & _filter_reason, int _allele) {
  // force a specific filtering operation based on some other data
  summary_info_vector[_allele].isFiltered = true;
  summary_info_vector[_allele].filterReason.push_back( _filter_reason);
}

// warning: no white-space allowed in filter reason
void FilterByBasicThresholds( int i_alt, MultiBook &m_summary_stats,
                             VariantOutputInfo &l_summary_info,
                             BasicFilters &basic_filter, float tune_xbias, float tune_sbias) {

  if (l_summary_info.variant_qual_score < basic_filter.min_quality_score) {
    string my_reason = "QualityScore<";
    my_reason += convertToString(basic_filter.min_quality_score);
    l_summary_info.filterReason.push_back(my_reason);
    l_summary_info.isFiltered = true;
  }

  if (m_summary_stats.GetDepth(-1, i_alt) < basic_filter.min_cov) {
     l_summary_info.isFiltered = true;
     string my_reason = "MINCOV<";
     my_reason += convertToString(basic_filter.min_cov);
     l_summary_info.filterReason.push_back(my_reason);
   }
  bool pos_cov = m_summary_stats.GetDepth(0, i_alt) < basic_filter.min_cov_each_strand;
  if (pos_cov){
    l_summary_info.isFiltered = true;
    string my_reason = "PosCov<";
    my_reason += convertToString(basic_filter.min_cov_each_strand);
    l_summary_info.filterReason.push_back(my_reason);
  }
  bool neg_cov = m_summary_stats.GetDepth(1, i_alt) < basic_filter.min_cov_each_strand;
  if ( neg_cov ) {
    l_summary_info.isFiltered = true;
    string my_reason = "NegCov<";
    my_reason +=convertToString( basic_filter.min_cov_each_strand);
    l_summary_info.filterReason.push_back(my_reason);
   }

  if (m_summary_stats.OldStrandBias(i_alt, tune_sbias) > basic_filter.strand_bias_threshold) {
     string my_reason = "STDBIAS";
     my_reason += convertToString( m_summary_stats.OldStrandBias(i_alt, tune_sbias));
     my_reason += ">";
     my_reason += convertToString(basic_filter.strand_bias_threshold);
     l_summary_info.filterReason.push_back(my_reason);

     l_summary_info.isFiltered = true;
   }

 /*  if (m_summary_stats.GetXBias(i_alt, tune_xbias) > basic_filter.beta_bias_filter) {
     string my_reason = "XBIAS";
     my_reason += convertToString( m_summary_stats.GetXBias(i_alt, tune_sbias));
     my_reason += ">";
     my_reason += convertToString(basic_filter.beta_bias_filter);
     l_summary_info.filterReason.push_back(my_reason);
     l_summary_info.isFiltered = true;
   }*/

}


void DecisionTreeData::FilterOneAllele(int i_alt, VariantOutputInfo &l_summary_info, AlleleIdentity &l_variant_identity, ControlCallAndFilters &my_filters) {

   // if some reason from the identity to filter it
  if (l_variant_identity.status.isProblematicAllele){
    l_summary_info.isFiltered = true;
   l_summary_info.filterReason.push_back(l_variant_identity.filterReason);
  }

    //filter values specific to SNPs, MNVs and Non Homopolymer Indels
    if (l_variant_identity.status.isHotSpot) {
      // hot spot overrides
      FilterByBasicThresholds( i_alt, all_summary_stats, l_summary_info, my_filters.filter_hotspot, tune_xbias, tune_sbias);
    }
    else if (l_variant_identity.ActAsSNP()) {
      //cout << "inside snp flow " << endl;
      FilterByBasicThresholds(i_alt, all_summary_stats, l_summary_info, my_filters.filter_snps, tune_xbias, tune_sbias);

    }//end if SNP or MNV
    else
      if (l_variant_identity.ActAsHPIndel()) {

        FilterByBasicThresholds( i_alt, all_summary_stats, l_summary_info, my_filters.filter_hp_indel, tune_xbias, tune_sbias);

      }

}

/* Method  to loop thru alleles and filter ones that fail the filter condition*/
void DecisionTreeData::FilterAlleles(ControlCallAndFilters &my_filters) {
  for (int i_alt = 0; i_alt < all_summary_stats.NumAltAlleles(); i_alt++) {
    FilterOneAllele(i_alt, summary_info_vector[i_alt], multi_allele.allele_identity_vector[i_alt], my_filters);
  } //end loop thru alleles
}


///***** heal snps here *****//

void DecisionTreeData::AccumulateFilteredAlleles(){
   int numAlleles = summary_info_vector.size();
  VariantOutputInfo _summary_info;
   for (int i=0; i<numAlleles; i++){
    _summary_info = summary_info_vector.at(i);

    if (_summary_info.isFiltered)
      filteredAllelesIndex.push_back(i);
   }
}


void DecisionTreeData::BestSNPsSuppressInDels(bool heal_snps){
  //now if the best allele is a SNP and one or more Indel alleles present at the same position
  //then remove all the indel alleles.
  //This is done mainly to represent SNPs at the exact position and not have to represent it as MNV.
  //EXAMPLE REF = CA Alt = C, CC. IF C->A SNP is true then we want to move the allele representation to REF = C, Alt = A which is a more standard representation.
  if (isBestAlleleSNP & heal_snps) {
    //loop thru all the alleles and filter all Indel alleles which will be later removed from alts.
   int numAlleles = summary_info_vector.size();

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


void DecisionTreeData::SimplifySNPsIfNeeded(vcf::Variant **candidate_variant, ExtendParameters *parameters){

  FindBestAlleleIdentity();
  AccumulateFilteredAlleles();

  if (!best_variant_filtered && (isBestAlleleSNP & parameters->my_controls.heal_snps)) { //currently we are removing other filtered alleles if the best allele is a SNP

    BestSNPsSuppressInDels(parameters->my_controls.heal_snps);
    // see if any genotype-components are needed here
    // if so, cannot remove them
    // unwilling to assume "deletions" or "insertions" are really reference if they are noticeable
    bool cannot_adjust = false;
    if (eval_genotype.genotype_already_set){
      for (unsigned int i_ndx=0; i_ndx<eval_genotype.genotype_component.size(); i_ndx++){
        int i_comp = eval_genotype.genotype_component.at(i_ndx);
        if (i_comp>0){
          if (summary_info_vector.at(i_comp-1).isFiltered)
            cannot_adjust = true;  // because it is part of the best diploid genotype
        }
      }
    }
    // and therefore will give a nonsense genotype if we do adjust
    if (!cannot_adjust){
      RemoveFilteredAlleles(candidate_variant, filteredAllelesIndex);
      AdjustAlleles(candidate_variant);
    }
  }

}

//********* heal snps done ****////

void DetectSSEForNoCall(VariantOutputInfo &l_summary_info, AlleleIdentity &var_identity, float sseProbThreshold, float minRatioReadsOnNonErrorStrand, float base_strand_bias,vcf::Variant ** candidate_variant, unsigned _altAlleleIndex) {

  if (var_identity.sse_prob_positive_strand >= sseProbThreshold && var_identity.sse_prob_negative_strand >= sseProbThreshold) {
      l_summary_info.isFiltered = true;
      string my_reason = "NOCALLxPredictedSSE";
      l_summary_info.filterReason.push_back(my_reason);
  }
  else {
    // use the >original< counts to determine whether we were affected by this problem
    //float strand_ratio = ComputeBaseStrandBiasForSSE(relative_safety_level, candidate_variant, _altAlleleIndex);

    float transform_threshold = (1-minRatioReadsOnNonErrorStrand)/(1+minRatioReadsOnNonErrorStrand);
    bool pos_strand_bias_reflects_SSE = (base_strand_bias > transform_threshold); // more extreme than we like
    bool neg_strand_bias_reflects_SSE = (base_strand_bias < -transform_threshold); // more extreme
//    // note: this breaks down at low allele counts
//    float positive_ratio = (alt_counts_positive+safety_val) / (alt_counts_positive + alt_counts_negative + safety_val);
//    float negative_ratio = (alt_counts_negative+safety_val) / (alt_counts_positive + alt_counts_negative + safety_val);
//    bool pos_strand_bias_reflects_SSE = (negative_ratio < minRatioReadsOnNonErrorStrand);
//    bool neg_strand_bias_reflects_SSE = (positive_ratio < minRatioReadsOnNonErrorStrand);
    if (var_identity.sse_prob_positive_strand >= sseProbThreshold &&  pos_strand_bias_reflects_SSE) {
      l_summary_info.isFiltered = true;
      string my_reason = "NOCALLxPositiveSSE";
      l_summary_info.filterReason.push_back(my_reason);
    }

    if (var_identity.sse_prob_negative_strand >= sseProbThreshold && neg_strand_bias_reflects_SSE) {
      l_summary_info.isFiltered = true;
      string my_reason = "NOCALLxNegativeSSE";
      l_summary_info.filterReason.push_back(my_reason);
    }
  }
  // cout << alt_counts_positive << "\t" << alt_counts_negative << "\t" << ref_counts_positive << "\t" << ref_counts_negative << endl;
}

void DecisionTreeData::FilterSSE(vcf::Variant **candidate_variant,ClassifyFilters &filter_variant){
  for (unsigned int i_allele=0; i_allele<multi_allele.allele_identity_vector.size(); i_allele++){

    // change for 4.0:  store all allele information for multiallele clean filter application after VCF
    int _alt_allele_index = i_allele;
    float base_strand_bias = ComputeBaseStrandBiasForSSE(filter_variant.sse_relative_safety_level, candidate_variant, _alt_allele_index);
    (*candidate_variant)->info["SSSB"].push_back(convertToString(base_strand_bias));
    (*candidate_variant)->info["SSEP"].push_back(convertToString(multi_allele.allele_identity_vector[_alt_allele_index].sse_prob_positive_strand));
    (*candidate_variant)->info["SSEN"].push_back(convertToString(multi_allele.allele_identity_vector[_alt_allele_index].sse_prob_negative_strand));

    //@TODO: make sure this takes information from the tags in candidate variant and nowhere else
    // which forces us to be honest and only use information in the output
    DetectSSEForNoCall(summary_info_vector[i_allele],
                       multi_allele.allele_identity_vector[i_allele],
                       filter_variant.sseProbThreshold,
                       filter_variant.minRatioReadsOnNonErrorStrand, base_strand_bias, candidate_variant, i_allele);
}
}

void DecisionTreeData::AddStrandBiasTags(vcf::Variant **candidate_variant){
  for ( int i_allele=0; i_allele<all_summary_stats.NumAltAlleles(); i_allele++){
   (*candidate_variant)->info["STB"].push_back(convertToString(all_summary_stats.OldStrandBias(i_allele, tune_sbias)));
//    (*candidate_variant)->info["SXB"].push_back(convertToString(all_summary_stats.GetXBias(i_allele,tune_xbias)));  // variance zero = 0.1^2
    }
}

void DecisionTreeData::AddCountInformationTags(vcf::Variant ** candidate_variant, string &sampleName) {
  // store tagged filter quantities

  AddStrandBiasTags(candidate_variant);
    // depth by allele statements
  // complex with multialleles

   map<string, vector<string> >& infoOutput = (*candidate_variant)->info;
   PushAlleleCountsOntoStringMaps(infoOutput,all_summary_stats);

   // testing this field for filtering
   infoOutput["FXX"].push_back(convertToString(all_summary_stats.GetFailedReadRatio()));

  if (!sampleName.empty()) {
      map<string, vector<string> >& sampleOutput = (*candidate_variant)->samples[sampleName];
      PushAlleleCountsOntoStringMaps(sampleOutput, all_summary_stats);
  }

  // hrun fill in
  ClearVal(*candidate_variant, "HRUN");
  for (unsigned int ia=0; ia<multi_allele.allele_identity_vector.size(); ia++){
    (*candidate_variant)->info["HRUN"].push_back(convertToString(multi_allele.allele_identity_vector[ia].ref_hp_length));
  }
}

void SetQualityByDepth(vcf::Variant **candidate_variant){
  float raw_qual_score= (*candidate_variant)->quality;
  unsigned int scan_read_depth = atoi((*candidate_variant)->info.at("FDP")[0].c_str());
  // factor of 4 to put on similar scale to other data outputs.
  // depends on rounding of log-likleihood changes
  (*candidate_variant)->info["QD"].push_back(convertToString(4.0*raw_qual_score/scan_read_depth));
}

void DecisionTreeData::GenotypeFromEvaluator(vcf::Variant ** candidate_variant, ExtendParameters *parameters){
  (*candidate_variant)->quality = eval_genotype.evaluated_variant_quality;

  string genotype_string = eval_genotype.GenotypeAsString();
  reference_genotype = eval_genotype.IsReference();

  StoreGenotypeForOneSample(candidate_variant, parameters->sampleName, genotype_string, eval_genotype.evaluated_genotype_quality);
}


// fill this in with whatever we are really going to filter
void DecisionTreeData::FilterOnStringency(vcf::Variant **candidate_variant, float data_quality_stringency, int _check_allele_index) {


  float filter_on_min_quality = RetrieveQualityTagValue(*candidate_variant, "MLLD", _check_allele_index);

  if ((data_quality_stringency > filter_on_min_quality)) {

    string my_reason = "STRINGENCY";
   OverrideFilter( my_reason, _check_allele_index);
  }
}

void DecisionTreeData::FilterOnSpecialTags(vcf::Variant ** candidate_variant, ExtendParameters *parameters){
    // separate my control here
  for (unsigned int _alt_allele_index = 0; _alt_allele_index < multi_allele.allele_identity_vector.size(); _alt_allele_index++) {
     // if something is strange here
    SpecializedFilterFromLatentVariables(multi_allele.variant,  parameters->my_eval_control.filter_unusual_predictions, _alt_allele_index); // unusual filters
    SpecializedFilterFromHypothesisBias(multi_allele.variant, multi_allele.allele_identity_vector[_alt_allele_index], parameters->my_eval_control.filter_deletion_bias, parameters->my_eval_control.filter_insertion_bias, _alt_allele_index);
    FilterOnStringency(candidate_variant, parameters->my_controls.data_quality_stringency, _alt_allele_index);
  } 
}

void DecisionTreeData::SpecializedFilterFromLatentVariables( vcf::Variant ** candidate_variant, float bias_radius, int _allele) {

  float bias_threshold;
  // likelihood threshold
  if (bias_radius < 0.0f)
    bias_threshold = 100.0f; // oops, wrong variable - should always be positive
  else
    bias_threshold = bias_radius; // fine now

//  float radius_bias = hypothesis_stack.cur_state.bias_generator.RadiusOfBias();
   float radius_bias = RetrieveQualityTagValue(*candidate_variant, "RBI", _allele);

  if (radius_bias > bias_threshold) {
    stringstream filterReasonStr;
    filterReasonStr << "PREDICTIONSHIFTx" ;
    filterReasonStr << radius_bias;
    string my_tmp_string = filterReasonStr.str();
    OverrideFilter(my_tmp_string, _allele);
  }
}

void DecisionTreeData::FilterAlleleHypothesisBias( float ref_bias, float var_bias, float threshold_bias, int _allele) {
      bool ref_bad = (ref_bias > 0 && fabs(ref_bias) > threshold_bias);  // not certain this one is in the correct direction for filtering
      bool var_bad = (var_bias > 0 && fabs(var_bias) > threshold_bias);

      // the ith variant allele is problematic
         if (var_bad){
           stringstream filterReasonStr;
          filterReasonStr << "PREDICTIONVar";
          filterReasonStr << _allele+1;
           filterReasonStr << "SHIFTx" ;
          filterReasonStr << var_bias;
          string my_tmp_string = filterReasonStr.str();
          OverrideFilter(my_tmp_string, _allele);
        }
      // the reference is problematicly shifted relative to this allele
        if (ref_bad){
          stringstream filterReasonStr;
          filterReasonStr << "PREDICTIONRefSHIFTx" ;
          filterReasonStr << ref_bias;
          string my_tmp_string = filterReasonStr.str();
          OverrideFilter(my_tmp_string, _allele);
        }

}


void DecisionTreeData::SpecializedFilterFromHypothesisBias(vcf::Variant ** candidate_variant, AlleleIdentity allele_identity, float deletion_bias, float insertion_bias, int _allele) {

//  float ref_bias = hypothesis_stack.cur_state.bias_generator.latent_bias_v[0];
//  float var_bias = hypothesis_stack.cur_state.bias_generator.latent_bias_v[1];
   float ref_bias = RetrieveQualityTagValue(*candidate_variant, "REFB", _allele);
   float var_bias = RetrieveQualityTagValue(*candidate_variant, "VARB", _allele);

  if (allele_identity.status.isHPIndel) {
    if (allele_identity.status.isDeletion) {
      FilterAlleleHypothesisBias( ref_bias, var_bias, deletion_bias, _allele);
    }
    else if (allele_identity.status.isInsertion) {
      FilterAlleleHypothesisBias( ref_bias, var_bias, insertion_bias, _allele);
    }
  }
}

void FilterOnReadRejectionRate(vcf::Variant **candidate_variant, float read_rejection_threshold){
  float observed_read_rejection = RetrieveQualityTagValue(*candidate_variant, "FXX",0);
  if (observed_read_rejection>read_rejection_threshold){
    SetFilteredStatus(candidate_variant, true);
    string my_reason = "REJECTION";
    AddFilterReason(candidate_variant, my_reason);
  }
}



void DecisionTreeData::AggregateFilterInformation(vcf::Variant ** candidate_variant, ExtendParameters *parameters){
  // complete the decision tree for SSE using observed counts in base space
  // adds tags to the file
  FilterSSE(candidate_variant, parameters->my_controls.filter_variant);

  FilterOnSpecialTags(candidate_variant, parameters);

  FilterAlleles(parameters->my_controls);

  AddCountInformationTags(candidate_variant, parameters->sampleName);

}




void DecisionTreeData::GenotypeAlleleFilterMyCandidate(vcf::Variant  **candidate_variant, ExtendParameters *parameters){
  // only filter if  ref/variant and variant is filtered, or if variant/variant and both variants are filtered

  // 1) do I have an allele escaping filtration?

  vector<int> filter_triggered; // only if we're filtered do we iterate through this
    bool no_filter = false;
    for (unsigned int i_ndx=0; i_ndx<eval_genotype.genotype_component.size(); i_ndx++){
      int i_comp = eval_genotype.genotype_component.at(i_ndx);
      // if any allele involved in the genotype escapes
      if (i_comp>0){
        int alt_allele = i_comp-1;
        if (!summary_info_vector.at(alt_allele).isFiltered){
         no_filter = true;  // an allele escapes
        }else
          filter_triggered.push_back(alt_allele); // this allele should be reported if no-one escapes

      } else {
        // if reference, does best alternate allele escape the filters?
        if (!summary_info_vector.at(best_allele_index).isFiltered){
          no_filter = true;  // the best alternate escapes!
        } else
         filter_triggered.push_back(best_allele_index);  // or it doesn't escape and needs to be reported
      }
    }

    // unique reasons for filtration
    // should be at most 2: one for each non-ref allele involved in the genotype or one if reference/reference
    if (filter_triggered.size()>0){
      std::sort(filter_triggered.begin(), filter_triggered.end());
    // because now I'm removing adjacent elements that are duplicates
      std::vector<int>::iterator it;
     it = std::unique(filter_triggered.begin(), filter_triggered.end());
      filter_triggered.resize(std::distance(filter_triggered.begin(), it));
    // and one last sort
      std::sort(filter_triggered.begin(), filter_triggered.end());
    }

    // if no-one escaped
    if (!no_filter){


      // report the reasons for everyone who didn't escape
      for (unsigned int i_ndx=0; i_ndx<filter_triggered.size(); i_ndx++){
        // who is triggered?
        int bad_variant = filter_triggered.at(i_ndx);
        VariantOutputInfo _summary_info;

        _summary_info = summary_info_vector.at(bad_variant);
        SetFilteredStatus(candidate_variant,true);

        // if any normal reasons for filtering this allele
        if (_summary_info.isFiltered)
          for (unsigned int i_reason=0; i_reason<_summary_info.filterReason.size(); i_reason++)
            AddFilterReason(candidate_variant, _summary_info.filterReason.at(i_reason));
      }
    }
}

void DecisionTreeData::FilterMyCandidate(vcf::Variant  **candidate_variant, ExtendParameters *parameters){

  // any allele involved in the genotype may trigger a filter
    GenotypeAlleleFilterMyCandidate(candidate_variant, parameters);

    // whole candidate reason
     FilterOnReadRejectionRate(candidate_variant, parameters->my_controls.read_rejection_threshold);
}



void DecisionTreeData::FillInFiltersAtEnd(vcf::Variant ** candidate_variant, ExtendParameters *parameters){
  // now we fill in the filters that are triggered
  // start with a blank "." as the first filter reason so we always have this tag and a value
  NullFilterReason(candidate_variant);
  SetFilteredStatus(candidate_variant, false);  // not filtered at this point (!)

  // candidate alleles contribute to possible filtration
  FilterMyCandidate(candidate_variant, parameters);

  SimplifySNPsIfNeeded(candidate_variant, parameters);

  // if the genotype is a reference call, and we want to suppress it, make sure it will be in the filtered file
  SuppressReferenceCalls(candidate_variant, parameters,  reference_genotype);

  if (parameters->my_controls.suppress_nocall_genotypes)
    DetectAndSetFilteredGenotype(candidate_variant, parameters->sampleName);
}


// once we have standard data format across all alleles, the common filters can execute.
// this is just horrible at the moment
// need to clean this code up badly

/*
0) A variant entry accumulates all relevant data before being filtered:
0a) including genotype from evaluator  (diploid genotype: best pair of alleles)
0b) counts of alleles
0c) the QUAL (ref vs all alleles) is computed using the minimum min-variant-frequency for all variants
0d) min-variant-frequency is for safety set at at least one read/depth

1) A variant entry is filtered means that the column FILTER is NOCALL instead of PASS.
1a) if --suppress-nocalls is true, all non-hotspot filtered entries are placed in a filtered vcf
1b) if --suppress-nocall-genotypes is true, all filtered entries have the genotype replaced by ./.
1c) if --heal-snps is true, and the genotype only includes SNP alleles, we remove all other alleles from the vcf entry and simplify the representation.

2) A variant is filtered when:
2a) genotype is 0/0 and --suppress-reference-genotypes is true
2b) genotype is 0/0 and the best alternate allele triggers a filter [cannot trust reference vs best alternate allele]
2c) genotype is 0/X and X triggers a filter [cannot trust X]
2d) genotype is X/Y and both X and Y trigger a filter [cannot trust X/Y]
2e) too many reads have been rejected from this variant [cannot trust remaining frequencies]
2g) there are no reads found for the variant location [NODATA]

3) An allele triggers a filter when:
3z) the allele is identical to reference (not a variant)
3y) the QUAL score is below the min-variant-score
3a) if coverage (ref+allele) is too low in total (low quality of data)
3b) if coverage (ref+allele) is insufficient on either strand (low quality of data)
3c) if strand bias is triggered for this allele vs reference (model violation)
3cc) if the strand-beta-bias is triggered for this allele vs reference (model violation)
3d) The allele fails STRINGENCY against reference (general signal/noise)
3e) The allele has a PREDICTIONSHIFT (systematic mismatch between predictions and measurements in basecalling)
3f) if in/del, if the allele reference or var cluster is mis-centered with respect to the ref-allele direction (PredictionRefShift or PredictionVarNshift = systematic problem)
3g) if del, if the SSE filters are triggered (deletion entries may be errors)
3h) if in/del if HPLEN is too large (signal/noise)
*/
void DecisionTreeData::DecisionTreeOutputToVariant(vcf::Variant ** candidate_variant, ExtendParameters *parameters) {

  AggregateFilterInformation(candidate_variant, parameters); // step 0 above

    GenotypeFromEvaluator(candidate_variant, parameters);  // step 0
    // add a derived tag from the QUAL field and the depth counts
    SetQualityByDepth(candidate_variant);

  // no actual filters should be filled in yet, just all the information needed for filtering
  FillInFiltersAtEnd(candidate_variant, parameters);
}
