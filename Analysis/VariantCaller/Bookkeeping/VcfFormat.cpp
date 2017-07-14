/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VcfFormat.cpp
//! @ingroup  VariantCaller
//! @brief    Vcf file formatting & info tags

#include "VcfFormat.h"
#include "MiscUtil.h"
#include "ExtendParameters.h"
#include "IonVersion.h"


// current date string in YYYYMMDD format
string dateStr()
{
  time_t rawtime;
  struct tm* timeinfo;
  char buffer[80];
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 80, "%Y%m%d", timeinfo);
  return string(buffer);
}


string tvc_get_time_iso_string(time_t time)
{
  char time_buffer[1024];
  strftime(time_buffer, 1024, "%Y-%m-%dT%H:%M:%S", localtime(&time));
  return string(time_buffer);
}


string getVCFHeader(const ExtendParameters *parameters, ReferenceReader& ref_reader, const vector<string>& sample_list, int primary_sample, bool use_molecular_tag)
{
  stringstream headerss;
  headerss
  << "##fileformat=VCFv4.1" << endl
  << "##fileDate=" << dateStr() << endl
  << "##fileUTCtime=" << tvc_get_time_iso_string(time(NULL)) << endl
  << "##source=\"tvc " << IonVersion::GetVersion() << "-" << IonVersion::GetRelease() << " (" << IonVersion::GetGitHash() << ") - Torrent Variant Caller\"" << endl;

  if (not parameters->params_meta_name.empty())
    headerss << "##parametersName=\"" << parameters->params_meta_name << "\"" << endl;
  if (not parameters->params_meta_details.empty())
    headerss << "##parametersDetails=\"" << parameters->params_meta_details << "\"" << endl;

  if (not parameters->basecaller_version.empty())
    headerss << "##basecallerVersion=\"" << parameters->basecaller_version << "\"" << endl;
  if (not parameters->tmap_version.empty())
    headerss << "##tmapVersion=\"" << parameters->tmap_version << "\"" << endl;

  headerss << "##reference=" << parameters->fasta << endl;
  if (parameters->fasta == "GRCh38.p2") {headerss << "##masked_reference=GRCh38.p2.mask1" << endl;}

  string ref_filename = ref_reader.get_filename();
  size_t pos = ref_filename.rfind(".");
  if (pos != string::npos) {ref_filename = ref_filename.substr(0, pos);}
  pos = ref_filename.rfind("/");
  string mask_file = "maskfile_donot_remove.bed";
  if (pos != string::npos) {
	mask_file =  ref_filename.substr(0, pos+1)+mask_file;
	ref_filename = ref_filename.substr(pos + 1);
  }
  FILE *fp = fopen(mask_file.c_str(), "r");
  if (fp) {
    //read the header
    char line[1000];
    if (fgets(line, sizeof line, fp) and line[0] == '#') {
	char first[1000];
	sscanf(line, "%s", first);
	string tmp(first+1);
	headerss << "##maskVersion=" << tmp << endl;
    }
	fclose(fp);
  }
  string chr_name;
  long chr_size;
  for (int index = 0; (index < ref_reader.chr_count()); ++index) {
    chr_name = ref_reader.chr_str(index);
    chr_size = ref_reader.chr_size(index);
    headerss << "##contig=<ID=" << chr_name << ",length=" << chr_size << ",assembly=" << ref_filename << ">" << endl;
  }
  
  headerss << "##phasing=none" << endl
  // << "##commandline=\"" << parameters.commandline << "\"" << endl
  << "##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of samples with data\">" << endl

  << "##INFO=<ID=HS,Number=0,Type=Flag,Description=\"Indicate it is at a hot spot\">" << endl
  
  << "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total read depth at the locus\">" << endl
  << "##INFO=<ID=RO,Number=1,Type=Integer,Description=\"Reference allele observations\">" << endl
  << "##INFO=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observations\">" << endl
  << "##INFO=<ID=SRF,Number=1,Type=Integer,Description=\"Number of reference observations on the forward strand\">" << endl
  << "##INFO=<ID=SRR,Number=1,Type=Integer,Description=\"Number of reference observations on the reverse strand\">" << endl
  << "##INFO=<ID=SAF,Number=A,Type=Integer,Description=\"Alternate allele observations on the forward strand\">" << endl
  << "##INFO=<ID=SAR,Number=A,Type=Integer,Description=\"Alternate allele observations on the reverse strand\">" << endl
  
  << "##INFO=<ID=FDP,Number=1,Type=Integer,Description=\"Flow Evaluator read depth at the locus\">" << endl
  << "##INFO=<ID=FRO,Number=1,Type=Integer,Description=\"Flow Evaluator Reference allele observations\">" << endl
  << "##INFO=<ID=FAO,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations\">" << endl
  << "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator observation counts\">" << endl

  << "##INFO=<ID=FSRF,Number=1,Type=Integer,Description=\"Flow Evaluator Reference observations on the forward strand\">" << endl
  << "##INFO=<ID=FSRR,Number=1,Type=Integer,Description=\"Flow Evaluator Reference observations on the reverse strand\">" << endl
  << "##INFO=<ID=FSAF,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the forward strand\">" << endl
  << "##INFO=<ID=FSAR,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the reverse strand\">" << endl

  << "##INFO=<ID=TYPE,Number=A,Type=String,Description=\"The type of allele, either snp, mnp, ins, del, or complex.\">" << endl
  
  << "##INFO=<ID=LEN,Number=A,Type=Integer,Description=\"allele length\">" << endl
  << "##INFO=<ID=HRUN,Number=A,Type=Integer,Description=\"Run length: the number of consecutive repeats of the alternate allele in the reference genome\">" << endl
//  << "##INFO=<ID=SXB,Number=A,Type=Float,Description=\"Experimental strand bias based on approximate bayesian score for difference in frequency.\">" << endl
//  << "##INFO=<ID=MXFD,Number=1,Type=Float,Description=\"Mean maximum discrimination per read.\">" << endl;
//  << "##INFO=<ID=MFDT,Number=1,Type=Float,Description=\"Mean flows per read distinguishing variant above threshold.\">" << endl

  << "##INFO=<ID=MLLD,Number=A,Type=Float,Description=\"Mean log-likelihood delta per read.\">" << endl
  << "##INFO=<ID=FWDB,Number=A,Type=Float,Description=\"Forward strand bias in prediction.\">" << endl
  << "##INFO=<ID=REVB,Number=A,Type=Float,Description=\"Reverse strand bias in prediction.\">" << endl
  << "##INFO=<ID=REFB,Number=A,Type=Float,Description=\"Reference Hypothesis bias in prediction.\">" << endl
  << "##INFO=<ID=VARB,Number=A,Type=Float,Description=\"Variant Hypothesis bias in prediction.\">" << endl
  << "##INFO=<ID=STB,Number=A,Type=Float,Description=\"Strand bias in variant relative to reference.\">" << endl
  << "##INFO=<ID=STBP,Number=A,Type=Float,Description=\"Pval of Strand bias in variant relative to reference.\">" << endl
  << "##INFO=<ID=RBI,Number=A,Type=Float,Description=\"Distance of bias parameters from zero.\">" << endl
  << "##INFO=<ID=QD,Number=1,Type=Float,Description=\"QualityByDepth as 4*QUAL/FDP (analogous to GATK)\">" << endl
  << "##INFO=<ID=FXX,Number=1,Type=Float,Description=\"Flow Evaluator failed read ratio\">" << endl
  << "##INFO=<ID=FR,Number=.,Type=String,Description=\"Reason why the variant was filtered.\">" << endl
  << "##INFO=<ID=INFO,Number=.,Type=String,Description=\"Information about variant realignment and healing.\">" << endl
  << "##INFO=<ID=SSSB,Number=A,Type=Float,Description=\"Strand-specific strand bias for allele.\">" << endl
  << "##INFO=<ID=SSEN,Number=A,Type=Float,Description=\"Strand-specific-error prediction on negative strand.\">" << endl
  << "##INFO=<ID=SSEP,Number=A,Type=Float,Description=\"Strand-specific-error prediction on positive strand.\">" << endl
  << "##INFO=<ID=PB,Number=A,Type=Float,Description=\"Bias of relative variant position in reference reads versus variant reads. Equals Mann-Whitney U rho statistic P(Y>X)+0.5P(Y=X)\">" << endl
  << "##INFO=<ID=PBP,Number=A,Type=Float,Description=\"Pval of relative variant position in reference reads versus variant reads.  Related to GATK ReadPosRankSumTest\">" << endl
  << "##INFO=<ID=FDVR,Number=A,Type=Integer,Description=\"Level of Flow Disruption of the alternative allele versus reference.\">" << endl
  << "##INFO=<ID=SUBSET,Number=.,Type=String,Description=\"1-based index in ALT list of genotyped allele(s) that are a strict superset\">" << endl;

  // If we want to output multiple min-allele-freq
  if (parameters->program_flow.is_multi_min_allele_freq){
	  int multi_min_allele_freq_size = (int) parameters->program_flow.multi_min_allele_freq.size();
	  headerss
	    << "##INFO=<ID=MUQUAL,Number="<< multi_min_allele_freq_size << ",Type=Float,Description=\"QUAL scores for vector of min-allele-freq=(";
	  for (int i_maf = 0; i_maf < multi_min_allele_freq_size; ++i_maf){
		  headerss << parameters->program_flow.multi_min_allele_freq[i_maf] << (i_maf < (multi_min_allele_freq_size - 1) ? ",": "");
	  }
	  headerss << ").\">" << endl
	           << "##INFO=<ID=MUGT,Number=" << multi_min_allele_freq_size << ",Type=String,Description=\"Genotypes for the vector min-allele-freq specified.\">" << endl
	           << "##INFO=<ID=MUGQ,Number=" << multi_min_allele_freq_size << ",Type=Integer,Description=\"Genotype quality scores for the vector min-allele-freq specified.\">" << endl;
  }
  if (parameters->output_allele_cigar) {
	headerss << "##INFO=<ID=CIGAR,Number=A,Type=String,Description=\"Cigar to align reference to alternative allele.\">" << endl;
  }
  if (parameters->my_controls.use_lod_filter){
	  headerss << "##INFO=<ID=LOD,Number=A,Type=Float,Description=\"Limit of Detection at genomic location.\">" << endl;
  }
  if (use_molecular_tag){
	  headerss << "##INFO=<ID=MDP,Number=1,Type=Integer,Description=\"Total molecular depth at the locus\">" << endl
	 	       << "##INFO=<ID=MRO,Number=1,Type=Integer,Description=\"Reference allele molecular observations\">" << endl
	 	       << "##INFO=<ID=MAO,Number=A,Type=Integer,Description=\"Alternate allele molecular observations\">" << endl
	 	       << "##INFO=<ID=MAF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator molecular observation counts\">" << endl
	           << "##INFO=<ID=TGSM,Number=A,Type=Integer,Description=\"Number of additional families that may be falsely generated.\">" << endl
	           << "##INFO=<ID=VFSH,Number=1,Type=String,Description=\"The family size histogram of the variant, zipped by the pair (family size, family counts).\">" << endl;
  }
  if (parameters->my_controls.report_ppa){
	  headerss << "##INFO=<ID=PPA,Number=1,Type=String,Description=\"Possible Polyplody Alleles (PPA).\">" << endl;
  }
  if (parameters->my_controls.disable_filters){
	  headerss << "##INFO=<ID=BAI,Number=1,Type=Integer,Description=\"The 0-based index of the best alt allele.\">" << endl
	           << "##INFO=<ID=BAP,Number=2,Type=Integer,Description=\"The Best Allele Pair.\">" << endl
	           << "##INFO=<ID=FDBAP,Number=1,Type=Integer,Description=\"The level of Flow Disruption between the best allele pair.\">" << endl
	           << "##INFO=<ID=AAHPINDEL,Number=A,Type=Integer,Description=\"1: the alt allele Act As HP-INDEL; 0: otherwise.\">" << endl
	           << "##INFO=<ID=ISHPINDEL,Number=A,Type=Integer,Description=\"1: the alt allele is HP-INDEL; 0: otherwise.\">" << endl
	  	  	   << "##INFO=<ID=PARAM,Number=A,Type=String,Description=\"If not use FD, indicates which parameter set is used by the alt alleles.\">" << endl
    	  	   << "##INFO=<ID=FDPARAM,Number=A,Type=String,Description=\"If use FD, indicates which parameter set is used by the alt alleles.\">" << endl;
  }


  headerss << "##FILTER=<ID=NOCALL,Description=\"Generic filter. Filtering details stored in FR info tag.\">" << endl;

  headerss << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">" << endl
  << "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality, the Phred-scaled marginal (or unconditional) probability of the called genotype\">" << endl
  << "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">" << endl
  << "##FORMAT=<ID=RO,Number=1,Type=Integer,Description=\"Reference allele observation count\">" << endl
  << "##FORMAT=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observation count\">" << endl;

  if(use_molecular_tag){
      headerss << "##FORMAT=<ID=MDP,Number=1,Type=Integer,Description=\"Total molecular depth at the locus\">" << endl
      << "##FORMAT=<ID=MRO,Number=1,Type=Integer,Description=\"Reference allele molecular observation\">" << endl
      << "##FORMAT=<ID=MAO,Number=A,Type=Integer,Description=\"Alternate allele molecular observations\">" << endl
      << "##FORMAT=<ID=MAF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator molecular observation counts\">" << endl;
  }

  headerss << "##FORMAT=<ID=SRF,Number=1,Type=Integer,Description=\"Number of reference observations on the forward strand\">" << endl
  << "##FORMAT=<ID=SRR,Number=1,Type=Integer,Description=\"Number of reference observations on the reverse strand\">" << endl
  << "##FORMAT=<ID=SAF,Number=A,Type=Integer,Description=\"Alternate allele observations on the forward strand\">" << endl
  << "##FORMAT=<ID=SAR,Number=A,Type=Integer,Description=\"Alternate allele observations on the reverse strand\">" << endl

  << "##FORMAT=<ID=FDP,Number=1,Type=Integer,Description=\"Flow Evaluator Read Depth\">" << endl
  << "##FORMAT=<ID=FRO,Number=1,Type=Integer,Description=\"Flow Evaluator Reference allele observation count\">" << endl

  << "##FORMAT=<ID=FAO,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observation count\">" << endl
  << "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator observation counts\">" << endl
  << "##FORMAT=<ID=FSRF,Number=1,Type=Integer,Description=\"Flow Evaluator reference observations on the forward strand\">" << endl
  << "##FORMAT=<ID=FSRR,Number=1,Type=Integer,Description=\"Flow Evaluator reference observations on the reverse strand\">" << endl
  << "##FORMAT=<ID=FSAF,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the forward strand\">" << endl
  << "##FORMAT=<ID=FSAR,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the reverse strand\">" << endl;

  headerss << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
  // Ensure primary sample is always in the first column (IR req)
  headerss << "\t" << sample_list[primary_sample];
  for (size_t i = 0; i < sample_list.size(); i++)
    if (i != (size_t)primary_sample)
      headerss << "\t" << sample_list[i];

  return headerss.str();
}

void ClearVal(vcf::Variant &var, const char *clear_me, const string &sample_name){
    map<string, vector<string> >::iterator it;
      it = var.samples[sample_name].find(clear_me);
  if (it != var.samples[sample_name].end())
    var.samples[sample_name][clear_me].clear();
};

void ClearVal(vcf::Variant &var, const char *clear_me){
    map<string, vector<string> >::iterator it;
      it = var.info.find(clear_me);
  if (it != var.info.end())
    var.info[clear_me].clear();
};


//clear all the info tags, in case of a HotSpot VCF react Info tags might contain prior values
void clearInfoTags(vcf::Variant &var) {
  const vector<string> tag_to_clear =
  	  {"RO", "AO", "MDP", "MAO", "MRO", "MAF", "SAF", "SAR", "SRF", "SRR",
	   "DP", "RBI", "HRUN", "SSSB", "SSEN", "SSEP", "STB", "STBP", "PBP", "PB",
	   "FDP", "FRO", "FAO", "FSRF", "FSRR", "FSAF", "FSAR", "FXX", "QD", "TGSM",
	   "PPA", "VFSH", "MUQUAL", "MUGT", "MUGQ", "MLLD", "LOD"};
  for (vector<string>::const_iterator tag_it = tag_to_clear.begin(); tag_it != tag_to_clear.end(); ++tag_it)
	  ClearVal(var, tag_it->c_str());
}

void NullInfoFields(vcf::Variant &var, bool use_position_bias, bool use_molecular_tag){
    int num_alt = (int) var.alt.size();
	clearInfoTags(var);
	const vector<string> num_alt_tags_to_zero =
   	   {"AO", "SAF", "SAR", "AF", "FSAF", "FSAR", "HRUN", "RBI",
   	    "FWDB", "REVB", "REFB", "VARB", "SSSB", "SSEN", "SSEP", "MLLD", "AF"};
	for (vector<string>::const_iterator tag_it = num_alt_tags_to_zero.begin(); tag_it != num_alt_tags_to_zero.end(); ++tag_it){
		var.info[*tag_it] = vector<string>(num_alt, "0");
	}
	var.info["STB"] = vector<string>(num_alt, "0.5");
	var.info["STBP"] = vector<string>(num_alt, "1");

	const vector<string> tags_to_zero =
   	   {"DP", "RO", "SRF", "SRR", "FDP", "FRO", "FSRF", "FSRR", "FXX", "QD"};
	for (vector<string>::const_iterator tag_it = tags_to_zero.begin(); tag_it != tags_to_zero.end(); ++tag_it){
		var.info[*tag_it] = {"0"};
	}
	if (use_position_bias) {
		var.info["PB"] = vector<string>(num_alt, "0.5");
		var.info["PBP"] = vector<string>(num_alt, "1");
	}
	if (use_molecular_tag) {
		var.info["MDP"] = {"0"};
		var.info["MRO"] = {"0"};
		var.info["MAO"] = vector<string>(num_alt, "0");
		var.info["MAF"] = vector<string>(num_alt, "0");
	}
}

 // set up format string
void SetUpFormatString(vcf::Variant &var) {
	var.format = {"GT", "GQ", "MDP", "MRO", "MAO", "MAF", "DP",
			      "FDP", "RO", "FRO", "AO", "FAO", "AF", "SAR",
				  "SAF", "SRF", "SRR", "FSAR", "FSAF", "FSRF", "FSRR"};
}

int CalculateWeightOfVariant(vcf::Variant &current_variant){
  
    map<string, vector<string> >::iterator it;
    int weight;
    
  it = current_variant.info.find("DP");
  if (it != current_variant.info.end())
    weight = atoi(current_variant.info["DP"][0].c_str()); // or is this current sample ident?
  else weight = 1;
  return(weight);
}

float RetrieveQualityTagValue(vcf::Variant &current_variant, const string &tag_wanted, int _allele_index, const string& sample_name){
  
    map<string, vector<string> >::iterator it;
    float weight;
    
  it = current_variant.samples[sample_name].find(tag_wanted);
  if (it != current_variant.samples[sample_name].end()){
    // if the index is valid...
    if (current_variant.samples[sample_name][tag_wanted].size()> (unsigned int) _allele_index)
      weight = atof(current_variant.samples[sample_name][tag_wanted][_allele_index].c_str()); // or is this current sample ident?
      else
        weight = 0.0f;
  }
  else weight = 0.0f;
  return(weight);
}

float RetrieveQualityTagValue(vcf::Variant &current_variant, const string &tag_wanted, int _allele_index){
  
    map<string, vector<string> >::iterator it;
    float weight;
    
  it = current_variant.info.find(tag_wanted);
  if (it != current_variant.info.end()){
    // if the index is valid...
    if (current_variant.info[tag_wanted].size()> (unsigned int) _allele_index)
      weight = atof(current_variant.info[tag_wanted][_allele_index].c_str()); // or is this current sample ident?
      else
        weight = 0.0f;
  }
  else weight = 0.0f;
  return(weight);
}
// XXX

void NullFilterReason(vcf::Variant &candidate_variant, const string &sample_name){
	candidate_variant.info["FR"] = vector<string>(candidate_variant.alt.size(), ".");
}

void AddFilterReason(vcf::Variant &candidate_variant, string &additional_reason, const string &sample_name){
	for (unsigned int alt_allele_index = 0; alt_allele_index < candidate_variant.alt.size(); ++alt_allele_index){
        AddFilterReason(candidate_variant, additional_reason, sample_name, alt_allele_index);
  	}
}

void AddFilterReason(vcf::Variant &candidate_variant, string &additional_reason, const string &sample_name, unsigned int alt_allele_index){
  while (candidate_variant.info["FR"].size() < (alt_allele_index + 1)) {
    candidate_variant.info["FR"].push_back(".");
  }
  candidate_variant.info["FR"][alt_allele_index] += "&" + additional_reason;
}

void AddInfoReason(vcf::Variant &candidate_variant, string &additional_reason, const string &sample_name){
  candidate_variant.info["FR"].push_back(additional_reason);
}

// if, for example, missing data
void NullGenotypeAllSamples(vcf::Variant & candidate_variant, bool use_molecular_tag)
{
  vector<string>& sampleNames = candidate_variant.sampleNames;
  const vector<string> keys_to_zero = {"GQ", "FDP", "FRO", "FSRF", "FSRR"};
  const vector<string> keys_to_zeros_of_alt = {"FAO", "AF", "FSAF", "FSAR"};
  unsigned int num_alt = candidate_variant.alt.size();
  for (vector<string>::iterator sample_it = sampleNames.begin(); sample_it != sampleNames.end(); ++sample_it) {
      map<string, vector<string> >& sampleOutput = candidate_variant.samples[*sample_it];

      for (vector<string>::const_iterator key_it = keys_to_zero.begin(); key_it != keys_to_zero.end(); ++key_it){
    	  sampleOutput[*key_it] = {"0"};
      }
      for (vector<string>::const_iterator key_it = keys_to_zeros_of_alt.begin(); key_it != keys_to_zeros_of_alt.end(); ++key_it){
    	  sampleOutput[*key_it] = vector<string>(num_alt, "0");
      }
      sampleOutput["GT"] = {"./."};

      if (use_molecular_tag){
    	  sampleOutput["MDP"] = {"0"};
    	  sampleOutput["MRO"] = {"0"};
    	  sampleOutput["MAO"] = vector<string>(num_alt, "0");
    	  sampleOutput["MAF"] = vector<string>(num_alt, "0");
      }
  }
}

void OverwriteGenotypeForOneSample(vcf::Variant &candidate_variant, const string &my_sample_name, string &my_genotype, float genotype_quality){
	map<string, vector<string> >& sampleOutput = candidate_variant.samples[my_sample_name];
	sampleOutput["GT"] = {my_genotype};
	sampleOutput["GQ"] = {convertToString((int) genotype_quality)};
}

void DetectAndSetFilteredGenotype(vcf::Variant &candidate_variant, map<string, float>& variant_quality, const string &sample_name){
	if (candidate_variant.isFiltered){
	    string no_call_genotype = "./.";
	    float original_quality = variant_quality[sample_name];
	    OverwriteGenotypeForOneSample(candidate_variant, sample_name, no_call_genotype, original_quality);
	}
}


void StoreGenotypeForOneSample(vcf::Variant &candidate_variant, const string &sample_name, string &my_genotype, float genotype_quality, bool multisample) {
  vector<string> sampleNames = candidate_variant.sampleNames;

  if (multisample) {
    map<string, vector<string> >& sampleOutput = candidate_variant.samples[sample_name];
    sampleOutput["GT"] = {my_genotype};
    sampleOutput["GQ"] = {convertToString((int) genotype_quality)};
  }
  else {
    for (vector<string>::iterator its = sampleNames.begin(); its != sampleNames.end(); ++its) {
	  string& sampleName = *its;
	  map<string, vector<string> >& sampleOutput = candidate_variant.samples[sampleName];
	  if (sampleName.compare(sample_name) == 0) { //sample of interest
	  // if no-call, will reset this entry as a final step, but until then, give me my genotype
		sampleOutput["GT"] = {my_genotype};
		//cout << "Storing Genotype = " << my_genotype << endl;
		sampleOutput["GQ"] = {convertToString((int) genotype_quality)};
	  }
	  else{ //for all other samples in BAM file just make a no-call at this point.
	    sampleOutput["GT"] = {"./."};
	    sampleOutput["GQ"] = {"0"};
	  }
    }
  }
}

void SetFilteredStatus(vcf::Variant &candidate_variant, bool isFiltered) {
   // filtering only sets the column to no-call
  // choice to put  in filtered file is handled by writing it out
  // genotype is modified by genotype
    if (isFiltered) {
      candidate_variant.filter = "NOCALL";
      candidate_variant.isFiltered = true;
    } else {
      candidate_variant.filter = "PASS";
      candidate_variant.isFiltered = false;
    }
}



void AdjustFDPForRemovedAlleles(vcf::Variant &candidate_variant, int filtered_allele_index, string sampleName)
{
  // first do the "info" tag as it is easier to find
  map<string, vector<string> >::iterator it;
  int total_depth=0;

  it = candidate_variant.info.find("FDP");
  if (it != candidate_variant.info.end())
    total_depth = atoi(candidate_variant.info["FDP"][0].c_str()); // or is this current sample ident?

  int allele_depth = 0;
  it = candidate_variant.info.find("FAO");
  if (it != candidate_variant.info.end())
    allele_depth = atoi(candidate_variant.info["FAO"][filtered_allele_index].c_str());

  total_depth -= allele_depth;
  if (total_depth<0)
    total_depth = 0; // how can this happen?

  ClearVal(candidate_variant, "FDP");
  candidate_variant.info["FDP"].push_back(convertToString(total_depth));

  if (!sampleName.empty()) {
      map<string, vector<string> >& sampleOutput = candidate_variant.samples[sampleName];
      sampleOutput["FDP"].clear();
      sampleOutput["FDP"].push_back(convertToString(total_depth));
      it = candidate_variant.info.find("MDP");
      if (it != candidate_variant.info.end()){
    	  candidate_variant.info["MDP"] = candidate_variant.info["FDP"];
      }
      it = sampleOutput.find("MDP");
      if (it != sampleOutput.end()){
    	  sampleOutput["MDP"] = sampleOutput["FDP"];
      }
  }
}


void RemoveFilteredAlleles(vcf::Variant &candidate_variant, vector<int> &filtered_alleles_index, const string &sample_name) {
  //now that all possible alt. alleles are evaluated decide on which allele is most likely and remove any that
  //that does'nt pass score threshold. Determine Genotype based on alleles that have evidence.
  candidate_variant.updateAlleleIndexes();
  string my_healing_glow = "HEALED";
  vector<string> originalAltAlleles = candidate_variant.alt;
  if (originalAltAlleles.size() > 1  &&
      originalAltAlleles.size() > filtered_alleles_index.size()  //remove only when number of alleles more than number of filtered alleles
      && !candidate_variant.isHotSpot) { //dont remove alleles if it is a HOT SPOT position as alleles might have been provided by the user.
    //remove filtered alleles with no support
    string altStr;
    int index;
    for (size_t i = 0; i <filtered_alleles_index.size(); i++) {

      index = filtered_alleles_index[i];
      //generate allele index before removing alleles
      altStr = originalAltAlleles[index];
      // specify what the alleles removed are
      //my_healing_glow = "HEALED" + altStr;

      //altStr = (*candidate_variant)->alt[index];
      // Note: need to update index for adjustments
      //AdjustFDPForRemovedAlleles(candidate_variant, index, sample_name);
      //cout << "Removed Fitered allele: index = " << index << " allele = " << altStr << endl;
      // @TODO: removeAlt wrecks the genotype as well
      // fix so we don't remove genotype components.

      candidate_variant.removeAlt(altStr);
      candidate_variant.updateAlleleIndexes();
      // if we are deleting alleles, indicate data potentially damaged at this location
      AddInfoReason(candidate_variant, my_healing_glow, sample_name);
    }
  }
}

// this only needs to know candidate variant, nothing else
void AdjustAlleles(vcf::Variant &candidate_variant, int position_upper_bound)
{
  vector<string>& types  = candidate_variant.info["TYPE"];
  string& refAllele      = candidate_variant.ref;
  vector<string>& alts   = candidate_variant.alt;
  int position     = candidate_variant.position;
  int max_trim = refAllele.length();
  if (position_upper_bound)
    max_trim = min(max_trim,position_upper_bound - position);
  string& altAllele = alts[0];
  //nothing to do if there are multiple allels

  if (types.size() != 1)
    return;

  if ((types[0]).compare("snp") == 0 && refAllele.length() > 1 && refAllele.length() == altAllele.length())  {
    //need to adjust position only in cases where SNP is represent as MNV due to haplotyping - REF= TTC ALT = TTT
    for (int i = 0; i < max_trim; ++i) {
      if (refAllele[i] != altAllele[i]) {
        candidate_variant.position = position;
        candidate_variant.ref = refAllele.substr(i, 1);
        candidate_variant.alt[0] = altAllele.substr(i, 1);
        break;
      }
      position++;
    }

  }

}

