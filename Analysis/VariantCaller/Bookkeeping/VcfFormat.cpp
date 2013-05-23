/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     VcfFormat.cpp
//! @ingroup  VariantCaller
//! @brief    Vcf file formatting & info tags

#include "VcfFormat.h"
#include "CandidateVariantGeneration.h"
#include "ExtendParameters.h"



string getVCFHeader(ExtendParameters *parameters, CandidateGenerationHelper &candidate_generator) {
  stringstream headerss;
  headerss << "##fileformat=VCFv4.1" << endl
  << "##fileDate=" << dateStr() << endl
  << "##source=Torrent Unified Variant Caller (Extension of freeBayes) " << endl
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
  
  << "##INFO=<ID=FSRF,Number=1,Type=Integer,Description=\"Flow Evaluator Reference observations on the forward strand\">" << endl
  << "##INFO=<ID=FSRR,Number=1,Type=Integer,Description=\"Flow Evaluator Reference observations on the reverse strand\">" << endl
  << "##INFO=<ID=FSAF,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the forward strand\">" << endl
  << "##INFO=<ID=FSAR,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the reverse strand\">" << endl
  
  << "##INFO=<ID=TYPE,Number=A,Type=String,Description=\"The type of allele, either snp, mnp, ins, del, or complex.\">" << endl
  
  << "##INFO=<ID=LEN,Number=A,Type=Integer,Description=\"allele length\">" << endl
  << "##INFO=<ID=HRUN,Number=A,Type=Integer,Description=\"Run length: the number of consecutive repeats of the alternate allele in the reference genome\">" << endl
  
  << "##INFO=<ID=FR,Number=1,Type=String,Description=\"Reason why the variant was filtered.\">" << endl
  << "##INFO=<ID=NR,Number=1,Type=String,Description=\"Reason why the variant is a No-Call.\">" << endl
  
//  << "##INFO=<ID=BLL,Number=1,Type=Float,Description=\"Log-likelihood of bias parameters under prior.\">" << endl
  << "##INFO=<ID=RBI,Number=1,Type=Float,Description=\"Distance of bias parameters from zero.\">" << endl
  << "##INFO=<ID=FWDB,Number=1,Type=Float,Description=\"Forward strand bias in prediction.\">" << endl
  << "##INFO=<ID=REVB,Number=1,Type=Float,Description=\"Reverse strand bias in prediction.\">" << endl
  << "##INFO=<ID=REFB,Number=1,Type=Float,Description=\"Reference Hypothesis bias in prediction.\">" << endl
  << "##INFO=<ID=VARB,Number=1,Type=Float,Description=\"Variant Hypothesis bias in prediction.\">" << endl
  << "##INFO=<ID=SSEN,Number=1,Type=Float,Description=\"Strand-specific-error prediction on negative strand.\">" << endl
  << "##INFO=<ID=SSEP,Number=1,Type=Float,Description=\"Strand-specific-error prediction on positive strand.\">" << endl
  
  << "##INFO=<ID=STB,Number=1,Type=Float,Description=\"Strand bias in variant relative to reference.\">" << endl
  << "##INFO=<ID=SXB,Number=1,Type=Float,Description=\"Experimental strand bias based on approximate bayesian score for difference in frequency.\">" << endl
  
  << "##INFO=<ID=MLLD,Number=1,Type=Float,Description=\"Mean log-likelihood delta per read.\">" << endl;
//  << "##INFO=<ID=MXFD,Number=1,Type=Float,Description=\"Mean maximum discrimination per read.\">" << endl;
//  << "##INFO=<ID=MFDT,Number=1,Type=Float,Description=\"Mean flows per read distinguishing variant above threshold.\">" << endl

  headerss << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">" << endl
  << "##FORMAT=<ID=GQ,Number=1,Type=Float,Description=\"Genotype Quality, the Phred-scaled marginal (or unconditional) probability of the called genotype\">" << endl
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
    << "##FORMAT=<ID=FSRF,Number=1,Type=Integer,Description=\"Flow Evaluator reference observations on the forward strand\">" << endl
    << "##FORMAT=<ID=FSRR,Number=1,Type=Integer,Description=\"Flow Evaluator reference observations on the reverse strand\">" << endl
    << "##FORMAT=<ID=FSAF,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the forward strand\">" << endl
    << "##FORMAT=<ID=FSAR,Number=A,Type=Integer,Description=\"Flow Evaluator Alternate allele observations on the reverse strand\">" << endl

  << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
  for (size_t i = 0; i < candidate_generator.parser->sampleList.size(); i++)
    headerss << "\t" << candidate_generator.parser->sampleList.at(i) ;



  return headerss.str();
}

void ClearVal(vcf::Variant *var, const char *clear_me){
    map<string, vector<string> >::iterator it;
      it = var->info.find(clear_me);
  if (it != var->info.end())
    var->info[clear_me].clear();
};


//clear all the info tags, in case of a HotSpot VCF react Info tags might contain prior values
void clearInfoTags(vcf::Variant *var) {
  map<string, vector<string> >::iterator it;

  it = var->info.find("RO");
  if (it != var->info.end())
    var->info["RO"].clear();

  it = var->info.find("AO");
  if (it != var->info.end())
    var->info["AO"].clear();

  it = var->info.find("SAF");
  if (it != var->info.end())
    var->info["SAF"].clear();

  it = var->info.find("SAR");
  if (it != var->info.end())
    var->info["SAR"].clear();

  it = var->info.find("SRF");
  if (it != var->info.end())
    var->info["SRF"].clear();

  it = var->info.find("SRR");
  if (it != var->info.end())
    var->info["SRR"].clear();

  it = var->info.find("DP");
  if (it != var->info.end())
      var->info["DP"].clear();

//  ClearVal("BLL");
  
  it = var->info.find("RBI");
  if (it != var->info.end())
      var->info["RBI"].clear();

  it = var->info.find("HRUN");
  if (it != var->info.end())
      var->info["HRUN"].clear();

//  ClearVal("MFDT")

  
  it = var->info.find("MLLD");
  if (it != var->info.end())
      var->info["MLLD"].clear();
  
//  ClearVal("MXFD");
  
  it = var->info.find("SSEN");
  if (it != var->info.end())
      var->info["SSEN"].clear();

  it = var->info.find("SSEP");
  if (it != var->info.end())
    var->info["SSEP"].clear();

  it = var->info.find("STB");
  if (it != var->info.end())
    var->info["STB"].clear();

  it = var->info.find("SXB");
  if (it != var->info.end())
    var->info["SXB"].clear();

  ClearVal(var,"FDP");
  ClearVal(var,"FRO");
  ClearVal(var,"FAO");
  ClearVal(var,"FSRF");
  ClearVal(var,"FSRR");
  ClearVal(var,"FSAF");
  ClearVal(var,"FSAR");
}

void NullInfoFields(vcf::Variant *var){
   clearInfoTags(var);
   var->info["RO"].push_back(convertToString(0));
   var->info["AO"].push_back(convertToString(0));
   var->info["SAF"].push_back(convertToString(0));
   var->info["SRF"].push_back(convertToString(0));
   var->info["SAR"].push_back(convertToString(0));
   var->info["SRR"].push_back(convertToString(0));
   var->info["DP"].push_back(convertToString(0));
   
   var->info["FDP"].push_back(convertToString(0));
   var->info["FRO"].push_back(convertToString(0));
   var->info["FAO"].push_back(convertToString(0));
   var->info["FSRF"].push_back(convertToString(0));
   var->info["FSRR"].push_back(convertToString(0));
   var->info["FSAF"].push_back(convertToString(0));
   var->info["FSAR"].push_back(convertToString(0));
  
   
   var->info["HRUN"].push_back(convertToString(0));
   
   var->info["SSEN"].push_back(convertToString(0));
   var->info["SSEP"].push_back(convertToString(0));
   
    var->info["STB"].push_back(convertToString(0));
    var->info["SXB"].push_back(convertToString(0));
  
//   var->info["MFDT"].push_back(convertToString(0));
//   var->info["MXFD"].push_back(convertToString(0));
   var->info["MLLD"].push_back(convertToString(0)); 
   
//   var->info["BLL"].push_back(convertToString(0));
   var->info["RBI"].push_back(convertToString(0));
   var->info["FWDB"].push_back(convertToString(0));
   var->info["REVB"].push_back(convertToString(0));
   var->info["REFB"].push_back(convertToString(0));
   var->info["VARB"].push_back(convertToString(0));
}

 // set up format string
void SetUpFormatString(vcf::Variant *var) {
  var->format.clear();
  var->format.push_back("GT");
  var->format.push_back("GQ");
  // XXX
  var->format.push_back("DP");
  var->format.push_back("FDP");
  var->format.push_back("RO");
  var->format.push_back("FRO");
  var->format.push_back("AO");
  var->format.push_back("FAO");
  var->format.push_back("SAR");
  var->format.push_back("SAF");
  var->format.push_back("SRF");
  var->format.push_back("SRR");
  var->format.push_back("FSAR");
  var->format.push_back("FSAF");
  var->format.push_back("FSRF");
  var->format.push_back("FSRR");
}


int CalculateWeightOfVariant(vcf::Variant *current_variant){
  
    map<string, vector<string> >::iterator it;
    int weight;
    
  it = current_variant->info.find("DP");
  if (it != current_variant->info.end())
    weight = atoi(current_variant->info.at("DP")[0].c_str()); // or is this current sample ident?
  else weight = 1;
  return(weight);
}
