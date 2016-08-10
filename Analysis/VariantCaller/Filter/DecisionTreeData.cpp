/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DecisionTreeData.h"

void RemoveVcfInfo(vcf::Variant &variant, const vector<string> &info_fields, const string &sample_name, int sample_index)
{
	for(unsigned int i_info = 0; i_info < info_fields.size(); ++i_info){
		variant.info.erase(info_fields[i_info]);
		variant.format.erase(std::remove(variant.format.begin(), variant.format.end(), info_fields[i_info]), variant.format.end());
	    if (sample_index >= 0) {
	        variant.samples[sample_name].erase(info_fields[i_info]);
	    }
	}
}

void AutoFailTheCandidate(vcf::Variant &candidate_variant, bool use_position_bias, const string &sample_name, bool use_molecular_tag = false) {
  candidate_variant.quality = 0.0f;
  NullInfoFields(candidate_variant, use_position_bias); // no information, destroy any spurious entries, add all needed tags
  NullGenotypeAllSamples(candidate_variant);
  NullFilterReason(candidate_variant, sample_name);
  string my_reason = "NODATA";
  AddFilterReason(candidate_variant, my_reason, sample_name);
  SetFilteredStatus(candidate_variant, true);

  if(not use_molecular_tag){
  	  RemoveVcfInfo(candidate_variant, vector<string>({"MDP", "MRO", "MAO", "MAF"}), sample_name, 0);
  }
}



float FreqThresholdByType(AlleleIdentity &variant_identity, const ControlCallAndFilters &my_controls,
    const VariantSpecificParams& variant_specific_params)
{
  if (variant_specific_params.min_allele_freq_override)
    return variant_specific_params.min_allele_freq;
  if (variant_identity.status.isHotSpot)
    return my_controls.filter_hotspot.min_allele_freq;
  if (variant_identity.ActAsSNP())
    return my_controls.filter_snps.min_allele_freq;
  if (variant_identity.ActAsMNP())
    return my_controls.filter_mnp.min_allele_freq;
  if (variant_identity.ActAsHPIndel())
    return my_controls.filter_hp_indel.min_allele_freq;

  return my_controls.filter_snps.min_allele_freq;
}


string EvaluatedGenotype::GenotypeAsString(){
  stringstream tmp_g;
  tmp_g << genotype_component[0] << "/" << genotype_component[1];
  return(tmp_g.str());
}

bool EvaluatedGenotype::IsReference(){
  if ((genotype_component[0]==0) & (genotype_component[1]==0))
    return(true);
  else
    return(false);
}

void PushValueOntoStringMaps(vector<string> &tag, int index, double value) {
	for (int i = tag.size() - 1; i < index; ++i) {tag.push_back("0");}
	double d = atof(tag[index].c_str());
	d += value;
	tag[index] = convertToString(d);
}

void PushAlleleCountsOntoStringMaps(map<string, vector<string> >& my_map, MultiBook &all_summary_stats, bool info){
  unsigned mdp = atoi(my_map["MDP"][0].c_str());
  PushValueOntoStringMaps(my_map["FDP"], 0, all_summary_stats.TotalCount(-1));
  PushValueOntoStringMaps(my_map["FRO"], 0, all_summary_stats.GetAlleleCount(-1,0));
  PushValueOntoStringMaps(my_map["FSRF"], 0, all_summary_stats.GetAlleleCount(0,0));
  PushValueOntoStringMaps(my_map["FSRR"], 0, all_summary_stats.GetAlleleCount(1,0));

  // alternate allele count varies by allele
  for ( int i_alt=0; i_alt< all_summary_stats.NumAltAlleles(); i_alt++){
    unsigned mao = atoi(my_map["MAO"][i_alt].c_str());
    PushValueOntoStringMaps(my_map["FAO"], i_alt, all_summary_stats.GetAlleleCount(-1,i_alt+1));
    PushValueOntoStringMaps(my_map["FSAF"], i_alt, all_summary_stats.GetAlleleCount(0,i_alt+1));
    PushValueOntoStringMaps(my_map["FSAR"], i_alt, all_summary_stats.GetAlleleCount(1,i_alt+1));
	//if (!info) {
      if (all_summary_stats.TotalCount(-1) > 0)
        my_map["AF"].push_back(convertToString((double)all_summary_stats.GetAlleleCount(-1,i_alt+1) / (double)all_summary_stats.TotalCount(-1)));
      else
        my_map["AF"].push_back(convertToString(0));
      if (mdp > 0)
        my_map["MAF"].push_back(convertToString((double)mao / (double)mdp));
      else
        my_map["MAF"].push_back(convertToString(0));
    //}
  }
}

float ComputeBaseStrandBiasForSSE(float relative_safety_level, vcf::Variant & candidate_variant, unsigned _altAlleleIndex, const string &sample_name){
  unsigned alt_counts_positive = atoi(candidate_variant.samples[sample_name]["SAF"][_altAlleleIndex].c_str());
  unsigned alt_counts_negative = atoi(candidate_variant.samples[sample_name]["SAR"][_altAlleleIndex].c_str());
  // always only one ref count
 unsigned ref_counts_positive = atoi(candidate_variant.samples[sample_name]["SRF"][0].c_str());
 unsigned ref_counts_negative = atoi(candidate_variant.samples[sample_name]["SRR"][0].c_str());

  // remember to trap zero-count div by zero here with safety value
  float safety_val = 0.5f;  // the usual "half-count" to avoid zero
  unsigned total_depth = alt_counts_positive + alt_counts_negative + ref_counts_positive + ref_counts_negative;
  float relative_safety_val = safety_val + relative_safety_level * total_depth;

  float strand_ratio = ComputeTransformStrandBias(alt_counts_positive, alt_counts_positive+ref_counts_positive, alt_counts_negative, alt_counts_negative+ref_counts_negative, relative_safety_val);

  return(strand_ratio);
}


void SuppressReferenceCalls(vcf::Variant &candidate_variant, const ExtendParameters &parameters, bool reference_genotype, const string &sample_name){
  if (reference_genotype & !candidate_variant.isHotSpot & parameters.my_controls.suppress_reference_genotypes) {
    SetFilteredStatus(candidate_variant, true); // used to suppress reference genotypes if not hot spot
    string my_suppression_reason = "SUPPRESSREFERENCECALL";
    AddFilterReason(candidate_variant, my_suppression_reason, sample_name);
  }
}

//*** Below here is actual decision tree data ***/


void DecisionTreeData::SetupFromMultiAllele(const EnsembleEval &my_ensemble) {
  //multi_allele = _multi_allele;
  allele_identity_vector = my_ensemble.allele_identity_vector;
  info_fields = my_ensemble.info_fields;
//  summary_stats_vector.resize(multi_allele.allele_identity_vector.size());
  all_summary_stats.Allocate(allele_identity_vector.size()+1); // ref plus num alternate alleles
  summary_info_vector.resize(allele_identity_vector.size());
}

void DecisionTreeData::OverrideFilter(vcf::Variant &candidate_variant, string & _filter_reason, int _allele, const string &sample_name) {
  // force a specific filtering operation based on some other data
  summary_info_vector[_allele].isFiltered = true;
  summary_info_vector[_allele].filterReason.push_back( _filter_reason);
}

// warning: no white-space allowed in filter reason
void FilterByBasicThresholds(const string& allele, int i_alt, MultiBook &m_summary_stats,
                             VariantOutputInfo &l_summary_info,
                             const BasicFilters &basic_filter, float tune_xbias, float tune_sbias,
                             const VariantSpecificParams& variant_specific_params,
							 bool is_reference_call)
{
  int effective_min_quality_score = basic_filter.min_quality_score;
  if (variant_specific_params.min_variant_score_override)
    effective_min_quality_score = variant_specific_params.min_variant_score;
  if (l_summary_info.variant_qual_score < effective_min_quality_score) {
    string my_reason = "QualityScore<";
    my_reason += convertToString(effective_min_quality_score);
    l_summary_info.filterReason.push_back(my_reason);
    l_summary_info.isFiltered = true;
  }

  int effective_min_cov = basic_filter.min_cov;
  if (variant_specific_params.min_coverage_override)
    effective_min_cov = variant_specific_params.min_coverage;

  if (m_summary_stats.GetDepth(-1, i_alt) < effective_min_cov) {
    l_summary_info.isFiltered = true;
    string my_reason = "MINCOV<";
    my_reason += convertToString(effective_min_cov);
    l_summary_info.filterReason.push_back(my_reason);
  }

  int effective_min_var_cov = basic_filter.min_var_cov;
  // Filter out a variant allele if the variant coverage (FAO) is too low.
  // Don't apply the variant coverage filter if we are making a reference call.
  // This is because, by applying this filter, a reference call will be filtered out if any of the variant allele fails to pass the filter, which does not make sense.
  if ((not is_reference_call) and m_summary_stats.GetAlleleCount(-1, i_alt + 1) < effective_min_var_cov) {
    l_summary_info.isFiltered = true;
    string my_reason = "VARCOV<";
    my_reason += convertToString(effective_min_var_cov);
    l_summary_info.filterReason.push_back(my_reason);
  }

  int effective_min_cov_each_strand = basic_filter.min_cov_each_strand;
  if (variant_specific_params.min_coverage_each_strand_override)
    effective_min_cov_each_strand = variant_specific_params.min_coverage_each_strand;
  bool pos_cov = m_summary_stats.GetDepth(0, i_alt) < effective_min_cov_each_strand;
  if (pos_cov){
    l_summary_info.isFiltered = true;
    string my_reason = "PosCov<";
    my_reason += convertToString(effective_min_cov_each_strand);
    l_summary_info.filterReason.push_back(my_reason);
  }
  bool neg_cov = m_summary_stats.GetDepth(1, i_alt) < effective_min_cov_each_strand;
  if (neg_cov) {
    l_summary_info.isFiltered = true;
    string my_reason = "NegCov<";
    my_reason +=convertToString(effective_min_cov_each_strand);
    l_summary_info.filterReason.push_back(my_reason);
   }

  float effective_strand_bias_thr = basic_filter.strand_bias_threshold;
  if (variant_specific_params.strand_bias_override)
    effective_strand_bias_thr = variant_specific_params.strand_bias;

  float effective_strand_bias_pval_thr = basic_filter.strand_bias_pval_threshold;
  if (variant_specific_params.strand_bias_pval_override)
    effective_strand_bias_thr = variant_specific_params.strand_bias_pval;

  float strand_bias = m_summary_stats.OldStrandBias(i_alt, tune_sbias);
  float strand_bias_pval = m_summary_stats.StrandBiasPval(i_alt, tune_sbias);
  if (strand_bias > effective_strand_bias_thr &&
      strand_bias_pval <= effective_strand_bias_pval_thr) {
     string my_reason = "STDBIAS";
     my_reason += convertToString(strand_bias);
     my_reason += ">";
     my_reason += convertToString(effective_strand_bias_thr);
     l_summary_info.filterReason.push_back(my_reason);

     string my_reason1 = "STDBIASPVAL";
     my_reason1 += convertToString( strand_bias_pval );
     my_reason1 += "<";
     my_reason1 += convertToString(effective_strand_bias_pval_thr);
     l_summary_info.filterReason.push_back(my_reason1);

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


float CalculateLod(int total_count, int min_var_cov){
	float lod = 1.0f;
	if(total_count > 0)
		lod = ((float) min_var_cov - 0.5f)/((float) total_count);
	return lod;
}

// Do some rounding for lod
float CalculateOutputLod(float lod){
	float output_lod = 0.0f;
	if(((int)(lod * 10000.0f)) % 10 >= 5){
		output_lod = (float)((int)((lod + 0.0005f) * 1000.0f)) * 0.001f;
	}
	else{
		output_lod = (float)(((int)(lod * 1000.0f))* 10 + 5) * 0.0001f;
	}
	if(output_lod < 0.0001f){
		output_lod = 0.0001f;
	}
	else if(output_lod > 100.0f){
		output_lod = 100.0f;
	}
	return output_lod;
}

void DecisionTreeData::AddLodTags(vcf::Variant &candidate_variant, const string &sample_name, const ExtendParameters &parameters){
	// Currently don't output LOD if use_lod_filter = false
	if(not parameters.my_controls.use_lod_filter){
		return;
	}
	int total_count = all_summary_stats.TotalCount(-1);
	candidate_variant.info["LOD"].clear();
	for (int i_allele = 0; i_allele < all_summary_stats.NumAltAlleles(); ++i_allele){
		int min_var_cov = 0;
		//filter values specific to SNPs, MNVs and Non Homopolymer Indels
		if (allele_identity_vector[i_allele].status.isHotSpot) {
			min_var_cov = parameters.my_controls.filter_hotspot.min_var_cov;
		}
		else if (allele_identity_vector[i_allele].ActAsSNP()) {
		    min_var_cov = parameters.my_controls.filter_snps.min_var_cov;
		}
		else if (allele_identity_vector[i_allele].ActAsMNP()) {
		    min_var_cov = parameters.my_controls.filter_mnp.min_var_cov;

		} // end if MNV
		else if (allele_identity_vector[i_allele].ActAsHPIndel()) {
			 min_var_cov = parameters.my_controls.filter_hp_indel.min_var_cov;
		}

		float lod = CalculateLod(total_count, min_var_cov);
		float output_lod = CalculateOutputLod(lod);
		string val = convertToString(output_lod);
		candidate_variant.info["LOD"].push_back(val);
	}
}

void DecisionTreeData::FilterOnLod(const string& allele, int i_alt, MultiBook &m_summary_stats, VariantOutputInfo &l_summary_info, const ControlCallAndFilters &my_filters, const BasicFilters &basic_filter, bool is_reference_call)
{
	if((not my_filters.use_lod_filter) or is_reference_call){
		return;
	}
	int min_var_cov = basic_filter.min_var_cov;
	int total_count = m_summary_stats.TotalCount(-1);
    float lod = CalculateLod(total_count, min_var_cov);
    float min_allele_freq = basic_filter.min_allele_freq;
    float af = (float) m_summary_stats.GetAlleleCount(-1, i_alt + 1) / (float)(total_count);

    // The LOD filter first filter out the alt allele if af < min_allele_freq
    if(af < min_allele_freq){
        string my_reason = "";
    	l_summary_info.isFiltered = true;
    	my_reason += "AF";
    	my_reason += convertToString(af);
    	my_reason += "<";
    	my_reason += convertToString(min_allele_freq);
    	l_summary_info.filterReason.push_back(my_reason);
    }
    // The LOD filter then filter out the alt allele if af < lod_multiplier * lod
    if(af < my_filters.lod_multiplier * lod){
        string my_reason = "";
    	l_summary_info.isFiltered = true;
    	my_reason += "AF";
    	my_reason += convertToString(af);
    	my_reason += "<";
    	my_reason += convertToString(my_filters.lod_multiplier);
    	my_reason += "x";
    	my_reason += convertToString(lod);
    	l_summary_info.filterReason.push_back(my_reason);
    }
}


void DecisionTreeData::FilterOnPositionBias(const string& allele, int i_alt, MultiBook &m_summary_stats,
						VariantOutputInfo &l_summary_info,
						const ControlCallAndFilters &my_filters,
						const VariantSpecificParams& variant_specific_params)
{
  if (!my_filters.use_position_bias) {
    return;
  }

  float effective_position_bias_pval_thr = my_filters.position_bias_pval;
  if (variant_specific_params.position_bias_pval_override)
    effective_position_bias_pval_thr = variant_specific_params.position_bias_pval;

  float effective_position_bias_thr = my_filters.position_bias;
  if (variant_specific_params.position_bias_override)
    effective_position_bias_thr = variant_specific_params.position_bias;

  i_alt = i_alt + 1;   // confusing, ref is 0, but iterating from 0 excluding ref

  float position_bias = m_summary_stats.PositionBias(i_alt);
  float position_bias_pval = m_summary_stats.GetPositionBiasPval(i_alt);

  // low fraction of ref reads is not associated with real position bias
  unsigned int ref_count = m_summary_stats.GetAlleleCount(-1,0);      //FRO
  unsigned int var_count = m_summary_stats.GetAlleleCount(-1,i_alt);  //FAO
  float ref_fraction = ((float)ref_count) / ((float)ref_count + (float)var_count + 0.1);

  if ( ref_fraction <= my_filters.position_bias_ref_fraction ) {
    return;
  }

  if ((position_bias_pval < effective_position_bias_pval_thr) &&
      (position_bias > effective_position_bias_thr)){
    string my_reason = "POSBIAS";
    my_reason += convertToString(position_bias);
    my_reason += ">";
    my_reason += convertToString(effective_position_bias_thr);
    l_summary_info.filterReason.push_back(my_reason);
    
    string my_reason1 = "POSBIASPVAL";
    my_reason1 += convertToString(position_bias_pval);
    my_reason1 += "<";
    my_reason1 += convertToString(effective_position_bias_pval_thr);
    l_summary_info.filterReason.push_back(my_reason1);

    l_summary_info.isFiltered = true;
  }
}


void DecisionTreeData::FilterOneAllele(int i_alt, VariantOutputInfo &l_summary_info, AlleleIdentity &l_variant_identity,
    const ControlCallAndFilters &my_filters, const VariantSpecificParams& variant_specific_params)
{
  bool is_reference_call = eval_genotype.IsReference();

  // if some reason from the identity to filter it
  if (l_variant_identity.status.isProblematicAllele){
    l_summary_info.isFiltered = true;
    for (unsigned int i_reason=0; i_reason<l_variant_identity.filterReasons.size(); i_reason++)
      l_summary_info.filterReason.push_back(l_variant_identity.filterReasons.at(i_reason));
  }

  BasicFilters const *my_basic_filter = NULL;
  //filter values specific to SNPs, MNVs and Non Homopolymer Indels
  if (l_variant_identity.status.isHotSpot) {
    // hot spot overrides
	  my_basic_filter = &(my_filters.filter_hotspot);
  }
  else if (l_variant_identity.ActAsSNP()) {
    //cout << "inside snp flow " << endl;
    my_basic_filter = &(my_filters.filter_snps);

  }//end if SNP
  else if (l_variant_identity.ActAsMNP()) {
    //cout << "inside mnp flow " << endl;
	my_basic_filter = &(my_filters.filter_mnp);

  } // end if MNV
  else if (l_variant_identity.ActAsHPIndel()) {
	  my_basic_filter = &(my_filters.filter_hp_indel);
  }

  FilterByBasicThresholds(l_variant_identity.altAllele, i_alt, all_summary_stats, l_summary_info, *my_basic_filter, tune_xbias, tune_sbias, variant_specific_params, is_reference_call);


  FilterOnPositionBias(l_variant_identity.altAllele, i_alt, all_summary_stats, l_summary_info, my_filters, variant_specific_params);
  // Dima's LOD filter
  FilterOnLod(l_variant_identity.altAllele, i_alt, all_summary_stats, l_summary_info, my_filters, *my_basic_filter, is_reference_call);
}

/* Method  to loop thru alleles and filter ones that fail the filter condition*/
void DecisionTreeData::FilterAlleles(const ControlCallAndFilters &my_filters, const vector<VariantSpecificParams>& variant_specific_params)
{
  for (int i_alt = 0; i_alt < all_summary_stats.NumAltAlleles(); i_alt++) {
    FilterOneAllele(i_alt, summary_info_vector[i_alt], allele_identity_vector[i_alt], my_filters, variant_specific_params[i_alt]);
  } //end loop thru alleles
}


///***** heal snps here *****//

void DecisionTreeData::AccumulateFilteredAlleles(){
   int numAlleles = summary_info_vector.size();
  VariantOutputInfo _summary_info;
   for (int i=0; i<numAlleles; i++){
    _summary_info = summary_info_vector[i];

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
      _summary_info = summary_info_vector[counter];
      _variant_identity = allele_identity_vector[counter];


      if (_variant_identity.status.isIndel && !_summary_info.isFiltered) { //if it is Indel allele and not already filtered
        summary_info_vector[counter].isFiltered = true;
        filteredAllelesIndex.push_back(counter);
      }

    }
  }
}

void DecisionTreeData::FindBestAlleleIdentity(){
  
    AlleleIdentity _variant_identity = allele_identity_vector[best_allele_index];
        if (_variant_identity.status.isSNP || _variant_identity.status.isMNV)
        isBestAlleleSNP = true;
      else
        isBestAlleleSNP = false;
}


void DecisionTreeData::SimplifySNPsIfNeeded(VariantCandidate &candidate_variant, const ExtendParameters &parameters, const string &sample_name){

  FindBestAlleleIdentity();
  AccumulateFilteredAlleles();

  if (!best_variant_filtered && (isBestAlleleSNP & parameters.my_controls.heal_snps)) { //currently we are removing other filtered alleles if the best allele is a SNP

    BestSNPsSuppressInDels(parameters.my_controls.heal_snps);
    // see if any genotype-components are needed here
    // if so, cannot remove them
    // unwilling to assume "deletions" or "insertions" are really reference if they are noticeable
    bool cannot_adjust = false;
    if (eval_genotype.genotype_already_set){
      for (unsigned int i_ndx=0; i_ndx<eval_genotype.genotype_component.size(); i_ndx++){
        int i_comp = eval_genotype.genotype_component[i_ndx];
        if (i_comp>0){
          if (summary_info_vector[i_comp-1].isFiltered)
            cannot_adjust = true;  // because it is part of the best diploid genotype
        }
      }
    }
    // and therefore will give a nonsense genotype if we do adjust
    if (!cannot_adjust){
      RemoveFilteredAlleles(candidate_variant.variant, filteredAllelesIndex, sample_name);
      AdjustAlleles(candidate_variant.variant, candidate_variant.position_upper_bound);
    }
  }

}

//********* heal snps done ****////

void DetectSSEForNoCall(VariantOutputInfo &l_summary_info, AlleleIdentity &var_identity, float sseProbThreshold, float minRatioReadsOnNonErrorStrand, float base_strand_bias,vcf::Variant & candidate_variant, unsigned _altAlleleIndex) {

  if (var_identity.sse_prob_positive_strand > sseProbThreshold && var_identity.sse_prob_negative_strand > sseProbThreshold) {
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
    if (var_identity.sse_prob_positive_strand > sseProbThreshold &&  pos_strand_bias_reflects_SSE) {
      l_summary_info.isFiltered = true;
      string my_reason = "NOCALLxPositiveSSE";
      l_summary_info.filterReason.push_back(my_reason);
    }

    if (var_identity.sse_prob_negative_strand > sseProbThreshold && neg_strand_bias_reflects_SSE) {
      l_summary_info.isFiltered = true;
      string my_reason = "NOCALLxNegativeSSE";
      l_summary_info.filterReason.push_back(my_reason);
    }
  }
  // cout << alt_counts_positive << "\t" << alt_counts_negative << "\t" << ref_counts_positive << "\t" << ref_counts_negative << endl;
}


void DecisionTreeData::FilterBlackList(const vector<VariantSpecificParams>& variant_specific_params)
{

  return; // disable pending algo changes, revert to 4.2 behavior

  for (unsigned int i_allele=0; i_allele<allele_identity_vector.size(); i_allele++) {

    VariantOutputInfo &l_summary_info = summary_info_vector[i_allele];
   
    float coverage = all_summary_stats.GetAlleleCount(-1,i_allele+1);
    float coverage_fwd = all_summary_stats.GetAlleleCount(0,i_allele+1);
    float coverage_rev = all_summary_stats.GetAlleleCount(1,i_allele+1);
   
    char black_list_strand = variant_specific_params[i_allele].black_strand;
    if( coverage > 0) {
      if(black_list_strand == 'F') { 
	if( (coverage_fwd / coverage) > .7) {
	  l_summary_info.isFiltered = true;
	  string my_reason = "NOCALLxLowQualityForwardStrand";
	  l_summary_info.filterReason.push_back(my_reason);
	}
      } else if(black_list_strand == 'R') {
	if( (coverage_rev / coverage) > .7 ) {
	  l_summary_info.isFiltered = true;
	  string my_reason = "NOCALLxLowQualityReverseStrand";
	  l_summary_info.filterReason.push_back(my_reason);
	}
      } else if(black_list_strand == 'B') {
	l_summary_info.isFiltered = true;
	string my_reason = "NOCALLxLowQualityBothStrand";
	l_summary_info.filterReason.push_back(my_reason);
      }
    }
  }
}


void DecisionTreeData::FilterSSE(vcf::Variant &candidate_variant, const ClassifyFilters &filter_variant, const vector<VariantSpecificParams>& variant_specific_params, const string &sample_name)
{
  for (unsigned int i_allele=0; i_allele<allele_identity_vector.size(); i_allele++) {

    // change for 4.0:  store all allele information for multiallele clean filter application after VCF
    int _alt_allele_index = i_allele;
    float base_strand_bias = ComputeBaseStrandBiasForSSE(filter_variant.sse_relative_safety_level, candidate_variant, _alt_allele_index, sample_name);
    candidate_variant.info["SSSB"].push_back(convertToString(base_strand_bias));
    candidate_variant.info["SSEP"].push_back(convertToString(allele_identity_vector[_alt_allele_index].sse_prob_positive_strand));
    candidate_variant.info["SSEN"].push_back(convertToString(allele_identity_vector[_alt_allele_index].sse_prob_negative_strand));

    //@TODO: make sure this takes information from the tags in candidate variant and nowhere else
    // which forces us to be honest and only use information in the output
    DetectSSEForNoCall(summary_info_vector[i_allele],
                       allele_identity_vector[i_allele],
                       variant_specific_params[_alt_allele_index].sse_prob_threshold_override ?
                           variant_specific_params[_alt_allele_index].sse_prob_threshold : filter_variant.sseProbThreshold,
                       filter_variant.minRatioReadsOnNonErrorStrand, base_strand_bias, candidate_variant, i_allele);
}
}

void DecisionTreeData::AddStrandBiasTags(vcf::Variant &candidate_variant, const string &sample_name){
  for ( int i_allele=0; i_allele<all_summary_stats.NumAltAlleles(); i_allele++){
    // ignore the ref allele, by convention allele 0, increment done all_summary_stats
    candidate_variant.info["STB"].push_back(convertToString(all_summary_stats.OldStrandBias(i_allele, tune_sbias)));
    candidate_variant.info["STBP"].push_back(convertToString(all_summary_stats.StrandBiasPval(i_allele, tune_sbias)));
//    (*candidate_variant)->info["SXB"].push_back(convertToString(all_summary_stats.GetXBias(i_allele,tune_xbias)));  // variance zero = 0.1^2
    }
}

inline bool isinitialized (float x){
  return ( x != -1);
}
void DecisionTreeData::AddPositionBiasTags(vcf::Variant &candidate_variant, const string &sample_name)
{
  candidate_variant.info["PB"].clear();
  candidate_variant.info["PBP"].clear();
  for ( int i_allele=0; i_allele<all_summary_stats.NumAltAlleles(); i_allele++){
    // ignore the ref allele, by convention allele 0
    int i_alt = i_allele+1;
    float v = all_summary_stats.GetPositionBias(i_alt);
    if ( isinitialized(v) ) {
      string val = convertToString(v);
      candidate_variant.info["PB"].push_back(val);
    }
    else {
      candidate_variant.info["PB"].push_back(".");	
    }
    float v1 = all_summary_stats.GetPositionBiasPval(i_alt);
    if ( isinitialized(v1) ) {
      string val1 = convertToString(all_summary_stats.GetPositionBiasPval(i_alt));
      candidate_variant.info["PBP"].push_back(val1);
    }
    else {
      candidate_variant.info["PBP"].push_back(".");	
    }
  }
}

void DecisionTreeData::AddCountInformationTags(vcf::Variant & candidate_variant, const string &sample_name, const ExtendParameters &parameters) {
  // store tagged filter quantities

  AddStrandBiasTags(candidate_variant, sample_name);
    // depth by allele statements
  // complex with multialleles

  AddPositionBiasTags(candidate_variant, sample_name);
  // Add Dima's LOD tag
  AddLodTags(candidate_variant, sample_name, parameters);

   map<string, vector<string> >& infoOutput = candidate_variant.info;
   PushAlleleCountsOntoStringMaps(infoOutput,all_summary_stats, true);

   infoOutput["FXX"].push_back(convertToString(all_summary_stats.GetFailedReadRatio()));
   
  if (!sample_name.empty()) {
      map<string, vector<string> >& sampleOutput = candidate_variant.samples[sample_name];
      PushAlleleCountsOntoStringMaps(sampleOutput, all_summary_stats, false);
  }

  // hrun fill in
  ClearVal(candidate_variant, "HRUN");
  for (unsigned int ia=0; ia<allele_identity_vector.size(); ia++){
    candidate_variant.info["HRUN"].push_back(convertToString(allele_identity_vector[ia].ref_hp_length));
  }
}

void SetQualityByDepth(vcf::Variant &candidate_variant, map<string, float>& variant_quality, const string &sample_name){
  float raw_qual_score = variant_quality[sample_name];
  unsigned int scan_read_depth = atoi(candidate_variant.samples[sample_name]["FDP"][0].c_str());
  // factor of 4 to put on similar scale to other data outputs.
  // depends on rounding of log-likleihood changes
  if (scan_read_depth > 0)
    candidate_variant.info["QD"].push_back(convertToString(4.0*raw_qual_score/scan_read_depth));
  else
    candidate_variant.info["QD"].push_back(convertToString(0));
}

void DecisionTreeData::GenotypeFromEvaluator(vcf::Variant & candidate_variant, const ExtendParameters &parameters, const string &sample_name){
  if ((candidate_variant.quality == 0) or (eval_genotype.evaluated_variant_quality < candidate_variant.quality)) {
    candidate_variant.quality = eval_genotype.evaluated_variant_quality;
  }
  variant_quality[sample_name] = eval_genotype.evaluated_variant_quality;

  string genotype_string = eval_genotype.GenotypeAsString();
  reference_genotype = eval_genotype.IsReference();

  StoreGenotypeForOneSample(candidate_variant, sample_name, genotype_string, eval_genotype.evaluated_genotype_quality, parameters.multisample);
}


// fill this in with whatever we are really going to filter
void DecisionTreeData::FilterOnStringency(vcf::Variant &candidate_variant, const float data_quality_stringency, int _check_allele_index, const string &sample_name) {


  float filter_on_min_quality = RetrieveQualityTagValue(candidate_variant, "MLLD", _check_allele_index);

  if ((data_quality_stringency > filter_on_min_quality)) {

    string my_reason = "STRINGENCY";
   OverrideFilter(candidate_variant, my_reason, _check_allele_index, sample_name);
  }
}

void DecisionTreeData::FilterOnSpecialTags(vcf::Variant & candidate_variant, const ExtendParameters &parameters,
    const vector<VariantSpecificParams>& variant_specific_params, const string &sample_name)
{
  // separate my control here
  /*
  float max_bias = 0.0f;
  for (unsigned int _alt_allele_index = 0; _alt_allele_index < allele_identity_vector.size(); _alt_allele_index++) {
     float bias =  RetrieveQualityTagValue(candidate_variant, "RBI", _alt_allele_index);
     if (abs(bias) > max_bias) max_bias = abs(bias);
  }
  */
  for (unsigned int _alt_allele_index = 0; _alt_allele_index < allele_identity_vector.size(); _alt_allele_index++) {
     // if something is strange here
    /*  Not to do this on allele, revert to 4.6*/
    SpecializedFilterFromLatentVariables(*(variant),  variant_specific_params[_alt_allele_index].filter_unusual_predictions_override ?
        variant_specific_params[_alt_allele_index].filter_unusual_predictions : parameters.my_eval_control.filter_unusual_predictions, _alt_allele_index, sample_name); // unusual filters
    // ZZ: Per Earl, the correct filter for bias is max of all the bias in each allele (directional), and check that with the threshold.
    // No allele overriding.
    /*
    if (max_bias > parameters.my_eval_control.filter_unusual_predictions) { 

      stringstream filterReasonStr;
      filterReasonStr << "PREDICTIONSHIFTx" ;
      filterReasonStr << max_bias;
      string my_tmp_string = filterReasonStr.str();
      OverrideFilter(my_tmp_string, _alt_allele_index);
    }
    */

    SpecializedFilterFromHypothesisBias(*(variant), allele_identity_vector[_alt_allele_index],
        variant_specific_params[_alt_allele_index].filter_deletion_predictions_override ?
            variant_specific_params[_alt_allele_index].filter_deletion_predictions : parameters.my_eval_control.filter_deletion_bias,
        variant_specific_params[_alt_allele_index].filter_insertion_predictions_override ?
            variant_specific_params[_alt_allele_index].filter_insertion_predictions : parameters.my_eval_control.filter_insertion_bias,
        _alt_allele_index, sample_name);

    float effective_data_quality_stringency = parameters.my_controls.data_quality_stringency;
    if (variant_specific_params[_alt_allele_index].data_quality_stringency_override)
      effective_data_quality_stringency = variant_specific_params[_alt_allele_index].data_quality_stringency;
    FilterOnStringency(candidate_variant, effective_data_quality_stringency, _alt_allele_index, sample_name);
  } 
}

void DecisionTreeData::SpecializedFilterFromLatentVariables(vcf::Variant & candidate_variant, const float bias_radius, int _allele, const string &sample_name) {

  float bias_threshold;
  // likelihood threshold
  if (bias_radius < 0.0f)
    bias_threshold = 100.0f; // oops, wrong variable - should always be positive
  else
    bias_threshold = bias_radius; // fine now

//  float radius_bias = hypothesis_stack.cur_state.bias_generator.RadiusOfBias();
   float radius_bias = RetrieveQualityTagValue(candidate_variant, "RBI", _allele);
//   cout << "RBI checking " << radius_bias << " threshold " << bias_threshold  << " allele " << _allele << endl; // ZZ

  if (radius_bias > bias_threshold) {
    stringstream filterReasonStr;
    filterReasonStr << "PREDICTIONSHIFTx" ;
    filterReasonStr << radius_bias;
    string my_tmp_string = filterReasonStr.str();
    OverrideFilter(candidate_variant, my_tmp_string, _allele, sample_name);
  }
}

void DecisionTreeData::FilterAlleleHypothesisBias(vcf::Variant & candidate_variant, float ref_bias, float var_bias, float threshold_bias, int _allele, const string &sample_name) {
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
          OverrideFilter(candidate_variant, my_tmp_string, _allele, sample_name);
        }
      // the reference is problematicly shifted relative to this allele
        if (ref_bad){
          stringstream filterReasonStr;
          filterReasonStr << "PREDICTIONRefSHIFTx" ;
          filterReasonStr << ref_bias;
          string my_tmp_string = filterReasonStr.str();
          OverrideFilter(candidate_variant, my_tmp_string, _allele, sample_name);
        }

}


void DecisionTreeData::SpecializedFilterFromHypothesisBias(vcf::Variant & candidate_variant, AlleleIdentity allele_identity, const float deletion_bias, const float insertion_bias, int _allele, const string &sample_name)
{

//  float ref_bias = hypothesis_stack.cur_state.bias_generator.latent_bias_v[0];
//  float var_bias = hypothesis_stack.cur_state.bias_generator.latent_bias_v[1];
  float ref_bias = RetrieveQualityTagValue(candidate_variant, "REFB", _allele);
  float var_bias = RetrieveQualityTagValue(candidate_variant, "VARB", _allele);

  if (allele_identity.ActAsHPIndel()) {
    if (allele_identity.status.isDeletion) {
      FilterAlleleHypothesisBias(candidate_variant, ref_bias, var_bias, deletion_bias, _allele, sample_name);
    }
    else if (allele_identity.status.isInsertion) {
      FilterAlleleHypothesisBias(candidate_variant, ref_bias, var_bias, insertion_bias, _allele, sample_name);
    }
  }
}

void FilterOnReadRejectionRate(vcf::Variant &candidate_variant, float read_rejection_threshold, const string &sample_name){
  float observed_read_rejection = RetrieveQualityTagValue(candidate_variant, "FXX",0);
  if (observed_read_rejection>read_rejection_threshold){
    SetFilteredStatus(candidate_variant, true);
    string my_reason = "REJECTION";
    AddFilterReason(candidate_variant, my_reason, sample_name);
  }
}



void DecisionTreeData::AggregateFilterInformation(vcf::Variant & candidate_variant,
    const vector<VariantSpecificParams>& variant_specific_params, const ExtendParameters &parameters, const string &sample_name)
{
  // complete the decision tree for SSE using observed counts in base space
  // adds tags to the file
  FilterBlackList(variant_specific_params);
  
  FilterSSE(candidate_variant, parameters.my_controls.filter_variant, variant_specific_params, sample_name);

  FilterOnSpecialTags(candidate_variant, parameters, variant_specific_params, sample_name);

  FilterAlleles(parameters.my_controls, variant_specific_params);

  AddCountInformationTags(candidate_variant, sample_name, parameters);

}




void DecisionTreeData::GenotypeAlleleFilterMyCandidate(vcf::Variant  &candidate_variant, const ExtendParameters &parameters, const string &sample_name){
  // only filter if  ref/variant and variant is filtered, or if variant/variant and both variants are filtered

  // 1) do I have an allele escaping filtration?

  vector<int> filter_triggered; // only if we're filtered do we iterate through this
    bool no_filter = false;
    for (unsigned int i_ndx=0; i_ndx<eval_genotype.genotype_component.size(); i_ndx++){
      int i_comp = eval_genotype.genotype_component[i_ndx];
      // if any allele involved in the genotype escapes
      if (i_comp>0){
        int alt_allele = i_comp-1;
        if (!summary_info_vector[alt_allele].isFiltered){
         no_filter = true;  // an allele escapes
        }else
          filter_triggered.push_back(alt_allele); // this allele should be reported if no-one escapes

      } else {
        // if reference, does best alternate allele escape the filters?
        if (!summary_info_vector[best_allele_index].isFiltered){
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

    // Abuse FR tag to write out info fields
    for (unsigned int i_info=0; i_info<info_fields.size(); i_info++)
      AddInfoReason(candidate_variant, info_fields.at(i_info), sample_name);

    // if no-one escaped
    if (!no_filter){

      // report the reasons for everyone who didn't escape
      for (unsigned int i_ndx=0; i_ndx<filter_triggered.size(); i_ndx++){
        // who is triggered?
        int bad_variant = filter_triggered[i_ndx];
        VariantOutputInfo _summary_info;

        _summary_info = summary_info_vector[bad_variant];
        SetFilteredStatus(candidate_variant,true);

        // if any normal reasons for filtering this allele
        if (_summary_info.isFiltered)
          for (unsigned int i_reason=0; i_reason<_summary_info.filterReason.size(); i_reason++)
            AddFilterReason(candidate_variant, _summary_info.filterReason[i_reason], sample_name, bad_variant);
      }
    }
}

void DecisionTreeData::FilterMyCandidate(vcf::Variant &candidate_variant, const ExtendParameters &parameters, const string &sample_name)
{
  // any allele involved in the genotype may trigger a filter
  GenotypeAlleleFilterMyCandidate(candidate_variant, parameters, sample_name);

  // whole candidate reason
  FilterOnReadRejectionRate(candidate_variant, parameters.my_controls.read_rejection_threshold, sample_name);
}



void DecisionTreeData::FillInFiltersAtEnd(VariantCandidate &candidate_variant, const ExtendParameters &parameters, const string &sample_name)
{
  // now we fill in the filters that are triggered
  // start with a blank "." as the first filter reason so we always have this tag and a value
  NullFilterReason(candidate_variant.variant, sample_name);
  SetFilteredStatus(candidate_variant.variant, false);  // not filtered at this point (!)

  // candidate alleles contribute to possible filtration
  FilterMyCandidate(candidate_variant.variant, parameters, sample_name);

  SimplifySNPsIfNeeded(candidate_variant, parameters, sample_name);

  // if the genotype is a reference call, and we want to suppress it, make sure it will be in the filtered file
  SuppressReferenceCalls(candidate_variant.variant, parameters,  reference_genotype, sample_name);

  if (parameters.my_controls.suppress_nocall_genotypes)
	  DetectAndSetFilteredGenotype(candidate_variant.variant, variant_quality, sample_name);
}


// once we have standard data format across all alleles, the common filters can execute.
// this is just horrible at the moment
// need to clean this code up badly/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DECISIONTREEDATA_H
#define DECISIONTREEDATA_H


#include "api/BamReader.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>

#include "ClassifyVariant.h"
#include "StackEngine.h"
#include "VcfFormat.h"

class EvaluatedGenotype{
public:
  bool genotype_already_set;
  float evaluated_genotype_quality;
  float evaluated_variant_quality;
  vector<int> genotype_component;

  EvaluatedGenotype(){
    genotype_already_set = false;
    evaluated_genotype_quality = 0.0f;
    evaluated_variant_quality = 0.0f;
    genotype_component.assign(2,0); // 0/0 = reference call

  };
  string GenotypeAsString();
  bool IsReference();
};

// all the data needed to make a decision for filtration
// characterize the variant, and the outcome from whatever evaluator we use
class DecisionTreeData {
  public:

    vcf::Variant * variant;                         //!< VCF record of this variant position
    vector<AlleleIdentity> allele_identity_vector;  //!< Detailed information for each candidate allele
    vector<string>         info_fields;             //!< Additional information to be printed out in vcf FR tag

    MultiBook all_summary_stats;

    vector<VariantOutputInfo> summary_info_vector;

    vector<int> filteredAllelesIndex;

	map<string, float> variant_quality;


    bool best_variant_filtered;

    bool best_allele_set;
    int best_allele_index;
    bool isBestAlleleSNP;
    bool reference_genotype;

    EvaluatedGenotype eval_genotype;

    float tune_xbias; // not tuned, removed from filters
    float tune_sbias;

    DecisionTreeData(vcf::Variant &candidate_variant) /*: multi_allele(candidate_variant)*/ {
      variant = &candidate_variant;
      best_allele_set = false;
      best_allele_index = 0;
      best_variant_filtered=false;
      isBestAlleleSNP = false;
      reference_genotype = false;


      tune_xbias = 0.005f; // tune calculation of chi-square bias = proportioinal variance by frequency
      tune_sbias = 0.5f; // safety factor for small allele counts for transformed strand bias
    };

    void OverrideFilter(vcf::Variant &candidate_variant, string & _filter_reason, int _allele, const string &sample_name);
    void FilterOneAllele(int i_alt,VariantOutputInfo &l_summary_info,
                         AlleleIdentity &l_variant_identity, const ControlCallAndFilters &my_filters,
                         const VariantSpecificParams& variant_specific_params);
    void FilterAlleles(const ControlCallAndFilters &my_filters, const vector<VariantSpecificParams>& variant_specific_params);

    void AccumulateFilteredAlleles();

    void BestSNPsSuppressInDels(bool heal_snps);
    void FindBestAlleleIdentity();
    void FindBestAlleleByScore();

    void GenotypeFromBestAlleleIndex(vcf::Variant &candidate_variant, const ExtendParameters &parameters);
    void GenotypeFromEvaluator(vcf::Variant &candidate_variant, const ExtendParameters &parameters, const string &sample_name);

    void FilterMyCandidate(vcf::Variant &candidate_variant, const ExtendParameters &parameters, const string &sample_name);
    void BestAlleleFilterMyCandidate(vcf::Variant &candidate_variant, const ExtendParameters &parameters);
    void GenotypeAlleleFilterMyCandidate(vcf::Variant &candidate_variant, const ExtendParameters &parameters, const string &sample_name);

    void SimplifySNPsIfNeeded(VariantCandidate &candidate_variant, const ExtendParameters &parameters, const string &sample_name);


    bool SetGenotype(vcf::Variant &candidate_variant, const ExtendParameters &parameters, float gt_quality);
    void DecisionTreeOutputToVariant(VariantCandidate &candidate_variant, const ExtendParameters &parameters, int sample_index);

    void AggregateFilterInformation(vcf::Variant &candidate_variant, const vector<VariantSpecificParams>& variant_specific_params, const ExtendParameters &parameters, const string &sample_name);
    void FillInFiltersAtEnd(VariantCandidate &candidate_variant,const ExtendParameters &parameters, const string &sample_name);



    void SetupFromMultiAllele(const EnsembleEval &my_ensemble);
    void AddStrandBiasTags(vcf::Variant &candidate_variant, const string &sample_name);
    void AddPositionBiasTags(vcf::Variant &candidate_variant, const string &sample_name);
    void AddLodTags(vcf::Variant &candidate_variant, const string &sample_name, const ExtendParameters &parameters);
    void AddCountInformationTags(vcf::Variant &candidate_variant, const string &sampleName, const ExtendParameters &parameters);
    string GenotypeStringFromAlleles(std::vector<int> &allowedGenotypes, bool refAlleleFound);
    bool AllowedGenotypesFromSummary(std::vector<int> &allowedGenotypes);
    string GenotypeFromStatus(vcf::Variant &candidate_variant, const ExtendParameters &parameters);
    void SpecializedFilterFromLatentVariables(vcf::Variant &candidate_variant, const float bias_radius, int _allele, const string &sample_name);
    void SpecializedFilterFromHypothesisBias(vcf::Variant &candidate_variant, AlleleIdentity allele_identity, const float deletion_bias, const float insertion_bias, int _allele, const string &sample_name);
    void FilterAlleleHypothesisBias(vcf::Variant &candidate_variant, float ref_bias, float var_bias, float threshold_bias, int _allele, const string &sample_name);
    void FilterOnSpecialTags(vcf::Variant &candidate_variant, const ExtendParameters &parameters, const vector<VariantSpecificParams>& variant_specific_params, const string &sample_name);
    void FilterOnStringency(vcf::Variant &candidate_variant, const float data_quality_stringency,  int _check_allele_index, const string &sample_name);
    void FilterOnPositionBias(const string& allele, int i_alt, MultiBook &m_summary_stats, VariantOutputInfo &l_summary_info, const ControlCallAndFilters &my_filters, const VariantSpecificParams& variant_specific_params);
    void FilterOnLod(const string& allele, int i_alt, MultiBook &m_summary_stats, VariantOutputInfo &l_summary_info, const ControlCallAndFilters &my_filters, const BasicFilters &basic_filter, bool is_reference_call);
    void FilterBlackList(const vector<VariantSpecificParams>& variant_specific_params);
    void FilterSSE(vcf::Variant &candidate_variant, const ClassifyFilters &filter_variant, const vector<VariantSpecificParams>& variant_specific_params, const string &sample_name);
};
void FilterByBasicThresholds(stringstream &s, int i_alt, MultiBook &m_summary_stats,
                             VariantOutputInfo &l_summary_info,
                             const BasicFilters &basic_filter, float tune_xbias, float tune_bias, bool is_reference_call);

void AutoFailTheCandidate(vcf::Variant &candidate_variant, bool use_position_bias, const string &sample_name);
float FreqThresholdByType(AlleleIdentity &variant_identity, const ControlCallAndFilters &my_controls, const VariantSpecificParams& variant_specific_params);
void DetectSSEForNoCall(AlleleIdentity &var_identity, float sseProbThreshold, float minRatioReadsOnNonErrorStrand, float relative_safety_level, vcf::Variant &candidate_variant, unsigned _altAlleIndex);
void SetQualityByDepth(vcf::Variant &candidate_variant, map<string, float>& variant_quality, const string &sample_name);

#endif // DECISIONTREEDATA_H



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
void DecisionTreeData::DecisionTreeOutputToVariant(VariantCandidate &candidate_variant, const ExtendParameters &parameters, int sample_index)
{
  string sample_name = "";
  if (sample_index >= 0) {sample_name = candidate_variant.variant.sampleNames[sample_index];}
  AggregateFilterInformation(candidate_variant.variant, candidate_variant.variant_specific_params, parameters, sample_name); // step 0 above

  GenotypeFromEvaluator(candidate_variant.variant, parameters, sample_name);  // step 0
  // add a derived tag from the QUAL field and the depth counts
  SetQualityByDepth(candidate_variant.variant, variant_quality, sample_name);

  // no actual filters should be filled in yet, just all the information needed for filtering
  FillInFiltersAtEnd(candidate_variant, parameters, sample_name);
}
