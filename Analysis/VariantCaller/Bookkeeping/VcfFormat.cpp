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


string get_time_iso_string(time_t time)
{
  char time_buffer[1024];
  strftime(time_buffer, 1024, "%Y-%m-%dT%H:%M:%S", localtime(&time));
  return string(time_buffer);
}


string getVCFHeader(const ExtendParameters *parameters, const vector<string>& sample_list, int primary_sample)
{
  stringstream headerss;
  headerss
  << "##fileformat=VCFv4.1" << endl
  << "##fileDate=" << dateStr() << endl
  << "##fileUTCtime=" << get_time_iso_string(time(NULL)) << endl
  << "##source=\"tvc " << IonVersion::GetVersion() << "-" << IonVersion::GetRelease() << " (" << IonVersion::GetGitHash() << ") - Torrent Variant Caller\"" << endl;

  if (not parameters->params_meta_name.empty())
    headerss << "##parametersName=\"" << parameters->params_meta_name << "\"" << endl;
  if (not parameters->params_meta_details.empty())
    headerss << "##parametersDetails=\"" << parameters->params_meta_details << "\"" << endl;

  if (not parameters->basecaller_version.empty())
    headerss << "##basecallerVersion=\"" << parameters->basecaller_version << "\"" << endl;
  if (not parameters->tmap_version.empty())
    headerss << "##tmapVersion=\"" << parameters->tmap_version << "\"" << endl;

  headerss
  << "##reference=" << parameters->fasta << endl
  << "##phasing=none" << endl
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
  << "##INFO=<ID=QD,Number=1,Type=Float,Description=\"QualityByDepth as 4*QUAL/FDP (analogous to GATK)\">" << endl

  << "##INFO=<ID=FSRF,Number=1,Type=Integer,Description=\"Flow Evaluator Reference observations on the forward strand\">" << endl
  << "##INFO=<ID=FSRR,Number=1,Type=Integer,Description=\"Flow Evaluator Reference observations on the reverse strand\">" << endl
  << "##INFO=<ID=FSAF,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the forward strand\">" << endl
  << "##INFO=<ID=FSAR,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the reverse strand\">" << endl
  << "##INFO=<ID=FXX,Number=1,Type=Float,Description=\"Flow Evaluator failed read ratio\">" << endl

  << "##INFO=<ID=TYPE,Number=A,Type=String,Description=\"The type of allele, either snp, mnp, ins, del, or complex.\">" << endl
  
  << "##INFO=<ID=LEN,Number=A,Type=Integer,Description=\"allele length\">" << endl
  << "##INFO=<ID=HRUN,Number=A,Type=Integer,Description=\"Run length: the number of consecutive repeats of the alternate allele in the reference genome\">" << endl
  
  << "##INFO=<ID=FR,Number=.,Type=String,Description=\"Reason why the variant was filtered.\">" << endl
  
//  << "##INFO=<ID=BLL,Number=1,Type=Float,Description=\"Log-likelihood of bias parameters under prior.\">" << endl
  << "##INFO=<ID=RBI,Number=A,Type=Float,Description=\"Distance of bias parameters from zero.\">" << endl
  << "##INFO=<ID=FWDB,Number=A,Type=Float,Description=\"Forward strand bias in prediction.\">" << endl
  << "##INFO=<ID=REVB,Number=A,Type=Float,Description=\"Reverse strand bias in prediction.\">" << endl
  << "##INFO=<ID=REFB,Number=A,Type=Float,Description=\"Reference Hypothesis bias in prediction.\">" << endl
  << "##INFO=<ID=VARB,Number=A,Type=Float,Description=\"Variant Hypothesis bias in prediction.\">" << endl

  << "##INFO=<ID=SSSB,Number=A,Type=Float,Description=\"Strand-specific strand bias for allele.\">" << endl
  << "##INFO=<ID=SSEN,Number=A,Type=Float,Description=\"Strand-specific-error prediction on negative strand.\">" << endl
  << "##INFO=<ID=SSEP,Number=A,Type=Float,Description=\"Strand-specific-error prediction on positive strand.\">" << endl
  << "##INFO=<ID=STB,Number=A,Type=Float,Description=\"Strand bias in variant relative to reference.\">" << endl
  << "##INFO=<ID=STBP,Number=A,Type=Float,Description=\"Pval of Strand bias in variant relative to reference.\">" << endl
  << "##INFO=<ID=PB,Number=A,Type=Float,Description=\"Bias of relative variant position in reference reads versus variant reads. Equals Mann-Whitney U rho statistic P(Y>X)+0.5P(Y=X)\">" << endl
  << "##INFO=<ID=PBP,Number=A,Type=Float,Description=\"Pval of relative variant position in reference reads versus variant reads.  Related to GATK ReadPosRankSumTest\">" << endl
//  << "##INFO=<ID=SXB,Number=A,Type=Float,Description=\"Experimental strand bias based on approximate bayesian score for difference in frequency.\">" << endl
  
  << "##INFO=<ID=MLLD,Number=A,Type=Float,Description=\"Mean log-likelihood delta per read.\">" << endl;
//  << "##INFO=<ID=MXFD,Number=1,Type=Float,Description=\"Mean maximum discrimination per read.\">" << endl;
//  << "##INFO=<ID=MFDT,Number=1,Type=Float,Description=\"Mean flows per read distinguishing variant above threshold.\">" << endl


  headerss << "##FILTER=<ID=NOCALL,Description=\"Generic filter. Filtering details stored in FR info tag.\">" << endl;

  headerss << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">" << endl
  << "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality, the Phred-scaled marginal (or unconditional) probability of the called genotype\">" << endl
  << "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">" << endl
  << "##FORMAT=<ID=RO,Number=1,Type=Integer,Description=\"Reference allele observation count\">" << endl

  << "##FORMAT=<ID=AO,Number=A,Type=Integer,Description=\"Alternate allele observation count\">" << endl
  << "##FORMAT=<ID=SRF,Number=1,Type=Integer,Description=\"Number of reference observations on the forward strand\">" << endl
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
    << "##FORMAT=<ID=FSAR,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the reverse strand\">" << endl
    << "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency based on Flow Evaluator observation counts\">" << endl

  << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
  // Ensure primary sample is always in the first column (IR req)
  headerss << "\t" << sample_list[primary_sample];
  for (size_t i = 0; i < sample_list.size(); i++)
    if (i != (size_t)primary_sample)
      headerss << "\t" << sample_list[i];



  return headerss.str();
}

void ClearVal(vcf::Variant &var, const char *clear_me){
    map<string, vector<string> >::iterator it;
      it = var.info.find(clear_me);
  if (it != var.info.end())
    var.info[clear_me].clear();
};


//clear all the info tags, in case of a HotSpot VCF react Info tags might contain prior values
void clearInfoTags(vcf::Variant &var) {
  map<string, vector<string> >::iterator it;

  it = var.info.find("RO");
  if (it != var.info.end())
    var.info["RO"].clear();

  it = var.info.find("AO");
  if (it != var.info.end())
    var.info["AO"].clear();

  it = var.info.find("SAF");
  if (it != var.info.end())
    var.info["SAF"].clear();

  it = var.info.find("SAR");
  if (it != var.info.end())
    var.info["SAR"].clear();

  it = var.info.find("SRF");
  if (it != var.info.end())
    var.info["SRF"].clear();

  it = var.info.find("SRR");
  if (it != var.info.end())
    var.info["SRR"].clear();

  it = var.info.find("DP");
  if (it != var.info.end())
      var.info["DP"].clear();

  
  it = var.info.find("RBI");
  if (it != var.info.end())
      var.info["RBI"].clear();

  it = var.info.find("HRUN");
  if (it != var.info.end())
      var.info["HRUN"].clear();

  
  it = var.info.find("MLLD");
  if (it != var.info.end())
      var.info["MLLD"].clear();

  //SSE compute
  ClearVal(var,"SSSB");
  ClearVal(var,"SSEN");
  ClearVal(var,"SSEP");
  
  it = var.info.find("STB");
  if (it != var.info.end())
    var.info["STB"].clear();

  it = var.info.find("STBP");
  if (it != var.info.end())
    var.info["STBP"].clear();

  it = var.info.find("PBP");
  if (it != var.info.end())
    var.info["PBP"].clear();

/*  it = var.info.find("SXB");
  if (it != var.info.end())
    var.info["SXB"].clear();*/

  ClearVal(var,"FDP");
  ClearVal(var,"FRO");
  ClearVal(var,"FAO");
  //ClearVal(var,"JUNK");
  ClearVal(var,"AF");
  ClearVal(var,"FSRF");
  ClearVal(var,"FSRR");
  ClearVal(var,"FSAF");
  ClearVal(var,"FSAR");
  ClearVal(var,"FXX");
  ClearVal(var,"QD");
}

void NullInfoFields(vcf::Variant &var, bool use_position_bias){
   clearInfoTags(var);
   for (vector<string>::iterator I = var.alt.begin(); I != var.alt.end(); ++I) {
     var.info["AO"].push_back(convertToString(0));
     var.info["SAF"].push_back(convertToString(0));
     var.info["SAR"].push_back(convertToString(0));
     var.info["FAO"].push_back(convertToString(0));
     var.info["AF"].push_back(convertToString(0));
     var.info["FSAF"].push_back(convertToString(0));
     var.info["FSAR"].push_back(convertToString(0));
     var.info["HRUN"].push_back(convertToString(0));
     var.info["RBI"].push_back(convertToString(0));
     var.info["FWDB"].push_back(convertToString(0));
     var.info["REVB"].push_back(convertToString(0));
     var.info["REFB"].push_back(convertToString(0));
     var.info["VARB"].push_back(convertToString(0));
     var.info["SSSB"].push_back(convertToString(0));
     var.info["SSEN"].push_back(convertToString(0));
     var.info["SSEP"].push_back(convertToString(0));
     var.info["STB"].push_back(convertToString(0));
     var.info["STBP"].push_back(convertToString(0));
     if(use_position_bias) {
    	 var.info["PB"].push_back(convertToString(0.5f));
    	 var.info["PBP"].push_back(convertToString(1.0f));
     }
     var.info["MLLD"].push_back(convertToString(0));
   }
   var.info["DP"].push_back(convertToString(0));
   var.info["RO"].push_back(convertToString(0));
   var.info["SRF"].push_back(convertToString(0));
   var.info["SRR"].push_back(convertToString(0));
   
   var.info["FDP"].push_back(convertToString(0));
   var.info["FRO"].push_back(convertToString(0));
   var.info["FSRF"].push_back(convertToString(0));
   var.info["FSRR"].push_back(convertToString(0));
   var.info["FXX"].push_back(convertToString(0.0));
   var.info["QD"].push_back(convertToString(0.0));
   
}

 // set up format string
void SetUpFormatString(vcf::Variant &var) {
  var.format.clear();
  var.format.push_back("GT");
  var.format.push_back("GQ");
  // XXX
  var.format.push_back("DP");
  var.format.push_back("FDP");
  var.format.push_back("RO");
  var.format.push_back("FRO");
  var.format.push_back("AO");
  var.format.push_back("FAO");
  //var.format.push_back("JUNK");
  var.format.push_back("AF");
  var.format.push_back("SAR");
  var.format.push_back("SAF");
  var.format.push_back("SRF");
  var.format.push_back("SRR");
  var.format.push_back("FSAR");
  var.format.push_back("FSAF");
  var.format.push_back("FSRF");
  var.format.push_back("FSRR");
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

void NullFilterReason(vcf::Variant &candidate_variant){
  ClearVal(candidate_variant, "FR");
  string my_null_filter = ".";
      AddFilterReason(candidate_variant, my_null_filter);
}

void AddFilterReason(vcf::Variant &candidate_variant, string &additional_reason){
  candidate_variant.info["FR"].push_back(additional_reason);
}

// if, for example, missing data
void NullGenotypeAllSamples(vcf::Variant & candidate_variant)
{
  vector<string>& sampleNames = candidate_variant.sampleNames;

  for (vector<string>::iterator its = sampleNames.begin(); its != sampleNames.end(); ++its) {
    string& sampleName = *its;
    map<string, vector<string> >& sampleOutput = candidate_variant.samples[sampleName];
      sampleOutput["GT"].push_back("./.");
      sampleOutput["GQ"].push_back(convertToString(0));
      sampleOutput["FDP"].push_back(convertToString(0));
      sampleOutput["FRO"].push_back(convertToString(0));
      sampleOutput["FSRF"].push_back(convertToString(0));
      sampleOutput["FSRR"].push_back(convertToString(0));
      for (vector<string>::iterator I = candidate_variant.alt.begin(); I != candidate_variant.alt.end(); ++I) {
        sampleOutput["FAO"].push_back(convertToString(0));
        sampleOutput["AF"].push_back(convertToString(0));
        sampleOutput["FSAF"].push_back(convertToString(0));
        sampleOutput["FSAR"].push_back(convertToString(0));
      }
  }
}

void OverwriteGenotypeForOneSample(vcf::Variant &candidate_variant, const string &my_sample_name, string &my_genotype, float genotype_quality){
  // will create entry if one does not exist

  map<string, vector<string> >& sampleOutput = candidate_variant.samples[my_sample_name];
  // clear existing values
  map<string, vector<string> >::iterator it;
  it = sampleOutput.find("GT");
  if (it != sampleOutput.end())    sampleOutput["GT"].clear();
  it = sampleOutput.find("GQ");
 if (it != sampleOutput.end())     sampleOutput["GQ"].clear();


      sampleOutput["GT"].push_back(my_genotype);
      // genotype quality should be an "int"
      int local_quality = genotype_quality;
      //cout << "Storing Genotype = " << my_genotype << endl;
      sampleOutput["GQ"].push_back(convertToString(local_quality));

}

void DetectAndSetFilteredGenotype(vcf::Variant &candidate_variant, const string &sampleName){
if (candidate_variant.isFiltered){
  string no_call_genotype = "./.";
  float original_quality = candidate_variant.quality;
  OverwriteGenotypeForOneSample(candidate_variant, sampleName, no_call_genotype, original_quality);
}
}


void StoreGenotypeForOneSample(vcf::Variant &candidate_variant, const string &my_sample_name, string &my_genotype, float genotype_quality) {
  vector<string> sampleNames = candidate_variant.sampleNames;

  for (vector<string>::iterator its = sampleNames.begin(); its != sampleNames.end(); ++its) {
    string& sampleName = *its;
    //cout << "VariantAssist: SampleName = " << sampleName << " my_sample = " << my_sample_name << endl;
    map<string, vector<string> >& sampleOutput = candidate_variant.samples[sampleName];
  map<string, vector<string> >::iterator it;
  it = sampleOutput.find("GT");
  if (it != sampleOutput.end())    sampleOutput["GT"].clear();
  it = sampleOutput.find("GQ");
   if (it != sampleOutput.end())     sampleOutput["GQ"].clear();

    if (sampleName.compare(my_sample_name) == 0) { //sample of interest
      //cout << "isNocall " << isNoCall << " genotype = " << my_genotype << endl;

      // if no-call, will reset this entry as a final step, but until then, give me my genotype
        sampleOutput["GT"].push_back(my_genotype);
        //cout << "Storing Genotype = " << my_genotype << endl;
        int local_quality = genotype_quality;
        sampleOutput["GQ"].push_back(convertToString(local_quality));

    } else { //for all other samples in BAM file just make a no-call at this point.
      sampleOutput["GT"].push_back("./.");
      sampleOutput["GQ"].push_back(convertToString(0));
    }
    //cout <<"VariantAssist: total genotypes = " << sampleOutput["GT"].size() << endl;
  }

}

void SetFilteredStatus(vcf::Variant &candidate_variant, bool isFiltered) {
   // filtering only sets the column to no-call
  // choice to put  in filtered file is handled by writing it out
  // genotype is modified by genotype
    if (isFiltered) {
      candidate_variant.filter = "NOCALL" ;
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
  }
}


void RemoveFilteredAlleles(vcf::Variant &candidate_variant, vector<int> &filtered_alleles_index) {
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
      AddFilterReason(candidate_variant, my_healing_glow);
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

