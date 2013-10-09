/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     CandidateVariantGeneration.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#include "CandidateVariantGeneration.h"


// Note: diagnostic HACK
void justProcessInputVCFCandidates(CandidateGenerationHelper &candidate_generator, ExtendParameters *parameters) {


    // just take from input file
    vcf::VariantCallFile vcfFile;
    vcfFile.open(parameters->candidateVCFFileName);
    vcfFile.parseSamples = false;

    //my_job_server.NewVariant(vcfFile);
    vcf::Variant variant(vcfFile);
    long int position = 0;
    long int startPos = 0;
    long int stopPos = 0;
    string contigName = "";

    //clear the BED targets and populate new targets that span just the variant positions
    candidate_generator.parser->targets.clear();
    while (vcfFile.getNextVariant(variant)) {
        position = variant.position;
        contigName = variant.sequenceName;
        startPos = position - 10;
        stopPos = position + 10;
        //range check
        if (candidate_generator.parser->targets.size() > 0) {
            BedTarget * prevbd = &(candidate_generator.parser->targets[candidate_generator.parser->targets.size()-1]);
            if (contigName.compare(prevbd->seq) == 0 && (startPos <= prevbd->right) && (startPos > prevbd->left)) {
                prevbd->right = stopPos;
            }
            else {
                BedTarget bd(contigName,
                             startPos,
                             stopPos);
                candidate_generator.parser->targets.push_back(bd);
            }
        }
        else {
            BedTarget bd(contigName,
                         startPos,
                         stopPos);
            candidate_generator.parser->targets.push_back(bd);
        }

    }
}


void CandidateStreamWrapper::ResetForNextPosition() {
    doing_hotspot = false;
    doing_generated = false;
}

void CandidateStreamWrapper::SetUp(ofstream &outVCFFile, ofstream &filterVCFFile,  InputStructures &global_context, ExtendParameters *parameters) {
    candidate_generator.SetupCandidateGeneration(global_context, parameters);

    if (parameters->output == "vcf") {
        string headerstr = getVCFHeader(parameters, candidate_generator);
        outVCFFile << headerstr << endl;
        filterVCFFile << headerstr << endl;
        candidate_generator.parser->variantCallFile.parseHeader(headerstr);
    }

    if (parameters->program_flow.skipCandidateGeneration) {
        //PURELY R&D branch to be used only for diagnostic purposes, we remove all targets and generate new targets spanning just the input variants
        justProcessInputVCFCandidates(candidate_generator, parameters);
    }
}

// if we need to start returning hotspots, return the first variant
// if we're in the middle, return the next variant
// if we're done, return false
bool CandidateStreamWrapper::ReturnNextHotSpot(bool &isHotSpot, vcf::Variant *variant) {
    if (!doing_hotspot && checkHotSpotsSpanningHaploBases && candidate_generator.parser->inputVariantsWithinHaploBases.size() > 0) {
        doing_hotspot = true;
        ith_variant = 0;
    }
    if (doing_hotspot) {
        if (ith_variant<candidate_generator.parser->inputVariantsWithinHaploBases.size()) {
            isHotSpot = true;
            fillInHotSpotVariant(candidate_generator.parser, candidate_generator.samples, variant, candidate_generator.parser->inputVariantsWithinHaploBases.at(ith_variant));
            ith_variant++;
        } else {
            doing_hotspot = false;
            checkHotSpotsSpanningHaploBases = false;
            candidate_generator.parser->inputVariantsWithinHaploBases.clear();
        }
    }
    return(doing_hotspot);
}

// if I'm ready to return a generated variant
// try the variant
//
bool CandidateStreamWrapper::ReturnNextGeneratedVariant(bool &isHotSpot, vcf::Variant *variant, ExtendParameters *parameters) {
    if (!doing_generated) {
        doing_generated = true; // only generate one variant per location
        if (!generateCandidateVariant(candidate_generator.parser, candidate_generator.samples, variant, isHotSpot, parameters, candidate_generator.allowedAlleleTypes))
        {
            //even if this position is not a valid candidate the candidate alleles might have spanned a hotspot position
            if (candidate_generator.parser->lastHaplotypeLength > 1)
                checkHotSpotsSpanningHaploBases = true;
            return(false);
        }

        if (candidate_generator.parser->lastHaplotypeLength > 1)
            checkHotSpotsSpanningHaploBases = true;
        return(true);
    } else
        return(false);
}

// at a position, return the next variant
// if hotspots needed, return them
// if no hotspot needed return generated
bool CandidateStreamWrapper::ReturnNextLocalVariant(bool &isHotSpot, vcf::Variant *variant, ExtendParameters *parameters) {
    if (ReturnNextHotSpot(isHotSpot, variant))
        return(true);
    else
        if (ReturnNextGeneratedVariant(isHotSpot, variant, parameters))
            return(true);
    // reset state
    return(false);
}

// returns true with a new variant
// returns false if ran out of variants to generate
bool CandidateStreamWrapper::ReturnNextVariantStream(bool &isHotSpot, vcf::Variant *variant, ExtendParameters *parameters) {
    // call next local variant, get a hotspot, return true
    // call next local variant, don't get a hotspot, generate fresh
    if (ReturnNextLocalVariant(isHotSpot, variant, parameters))
        return(true);
    // if didn't return either type of variant
    // reset
    ResetForNextPosition();
    while (candidate_generator.parser->getNextAlleles(candidate_generator.samples, candidate_generator.allowedAlleleTypes)) {
        // at neew location, generate candidate
        if (ReturnNextLocalVariant(isHotSpot, variant, parameters))
            return(true);
        // if failed to generate a candidate, of course we're not doing anything
        ResetForNextPosition();
    }
    return(false); // all done
}

CandidateGenerationHelper::~CandidateGenerationHelper(){
  if (parser!=NULL)
    delete parser;
  
}

void CandidateGenerationHelper::SetupCandidateGeneration(InputStructures &global_context, ExtendParameters *parameters) {
  //Now loop thru the VCF file for all the variants and accumulate the statistics
  /*if (vcfProvided) {
   vcf::VariantCallFile * vcfFile = new vcf::VariantCallFile();
   vcfFile->open(parameters->variantPriorsFile);
   vcfFile->parseSamples = false;
  } */

  freeParameters = *parameters;

  for (int i = 0 ; i < (int)parameters->bams.size(); i++)
    freeParameters.bams.push_back(parameters->bams.at(i));


  parser = new AlleleParser(freeParameters);

  SetupSamples(parameters, global_context);
  //ostream& out = *(parser->output);

  SetAllowedAlleles(parameters);

}

void CandidateGenerationHelper::SetupSamples(ExtendParameters *parameters, InputStructures &global_context) {
  bool inBam = false;
  //bool inReadGroup = false;
  global_context.sampleList = parser->sampleList;
  global_context.readGroupToSampleNames = parser->readGroupToSampleNames;

  //now check if there are multiple samples in the BAM file and if so user should provide a sampleName to process
  if (global_context.sampleList.size() == 1 && parameters->sampleName.empty()) {
    parameters->sampleName = global_context.sampleList[0];
  } else
    if (global_context.sampleList.size() > 1 && parameters->sampleName.empty())  {
      cerr << "FATAL: Multiple Samples found in BAM file/s provided. Torrent Variant Caller currently supports variant calling on only one sample. " << endl;
      cerr << "FATAL: Please provide a sample name to process using -g parameter. " << endl;
      cerr << "FATAL: Other samples in input BAM files can be used as Reference sample using -G parameter. " << endl;
      exit(-1);
    }

  if (!parameters->sampleName.empty())  {   //check the sample provided is found in BAM file
    for (vector<string>::const_iterator s = global_context.sampleList.begin(); s != global_context.sampleList.end(); ++s) {

      if (parameters->sampleName.compare(*s) == 0) {
        inBam = true;
        break;
      }

    }

    if (!inBam) {
      cerr << "FATAL: Sample " << parameters->sampleName << " provided using -g option " <<
           " is not listed in the header of BAM file(s) "
            << endl;
      exit(-1);
    }
    //now find the read group ID associated with this sample name
    for (map<string, string>::const_iterator p = global_context.readGroupToSampleNames.begin(); p != global_context.readGroupToSampleNames.end(); ++p) {
      if (parameters->sampleName == p->second) {
        parameters->ReadGroupIDVector.push_back( p->first);
        //break;
      }
    }
    //cout << "Parameters ReadGroupID = " << parameters->ReadGroupID << endl;
    if (parameters->ReadGroupIDVector.empty()) {
      cerr << "FATAL: Sample " << parameters->sampleName << " provided using -g option " <<
           " is not associated with any Read Group listed in the header of BAM file(s) "
            << endl;
      exit(-1);
    }

  }
}

void CandidateGenerationHelper::SetAllowedAlleles(ExtendParameters *parameters) {
  // this can be uncommented to force operation on a specific set of genotypes
  vector<Allele> allGenotypeAlleles;
  allGenotypeAlleles.push_back(genotypeAllele(ALLELE_GENOTYPE, "A", 1));
  allGenotypeAlleles.push_back(genotypeAllele(ALLELE_GENOTYPE, "T", 1));
  allGenotypeAlleles.push_back(genotypeAllele(ALLELE_GENOTYPE, "G", 1));
  allGenotypeAlleles.push_back(genotypeAllele(ALLELE_GENOTYPE, "C", 1));

  allowedAlleleTypes = ALLELE_REFERENCE;
  if (parameters->allowSNPs) {
    allowedAlleleTypes |= ALLELE_SNP;
  }
  if (parameters->allowIndels) {
    allowedAlleleTypes |= ALLELE_INSERTION;
    allowedAlleleTypes |= ALLELE_DELETION;
  }
  if (parameters->allowMNPs) {
    allowedAlleleTypes |= ALLELE_MNP;
  }
  if (parameters->allowComplex) {
    allowedAlleleTypes |= ALLELE_COMPLEX;
  }
}

bool fillInHotSpotVariant(AlleleParser * parser, Samples &samples, vcf::Variant * var, vcf::Variant hsVar) {

    var->sequenceName = hsVar.sequenceName;

    var->position = hsVar.position;

    //copy ref and alt alleles from hotspot variant object
    var->ref = hsVar.ref;

    for (size_t i = 0; i < hsVar.alt.size(); i++)
      var->alt.push_back(hsVar.alt.at(i));

    int coverage = countAlleles(samples);
    var->id = ".";
    var->filter = ".";

    var->quality = 0.0;

    SetUpFormatString(var);

    //clear all the info tags before populating them
    clearInfoTags(var);

    //pair<int,int> refObservationCountByStrand = getObservationCount(samples, referenceBase);
    var->info["RO"].push_back(convertToString(coverage));
    var->info["SRF"].push_back(convertToString(0));
    var->info["SRR"].push_back(convertToString(0));

    pair<int,int> altObservationCountByStrand;
    vector<pair<int, int> > altObservationCountByStrandVector;

    for (vector<string>::iterator aa = var->alt.begin(); aa != var->alt.end(); ++aa) {

      string altbase = *aa;
      altObservationCountByStrandVector.push_back(make_pair(0,0));


      if (var->ref.length() > altbase.length()) {
        var->info["TYPE"].push_back("del");

      }
      else if (var->ref.length() < altbase.length()) {
          var->info["TYPE"].push_back("ins");
      }
      else if (var->ref.length() == 1 && var->ref.length() == altbase.length()) {
              var->info["TYPE"].push_back("snp");

      }
      else if (var->ref.length() == altbase.length()) {
                var->info["TYPE"].push_back("mnp");
      }

      var->info["LEN"].push_back(convert(altbase.length()));

      var->info["AO"].push_back(convertToString(0));
      var->info["SAF"].push_back(convertToString(0));
      var->info["SAR"].push_back(convertToString(0));
      var->info["HRUN"].push_back(convertToString(parser->homopolymerRunLeft(altbase) + 1 + parser->homopolymerRunRight(altbase)));
    }

    var->infoFlags["HS"] = true;

    //insert depth
    var->info["DP"].push_back(convertToString(coverage));
    vector<string> sampleNames = parser->sampleList;

    map<string,int> coverageSample = countAllelesBySample(samples);
    //map<string, vector<pair<int, int> > > altAlleleCountsBySample = getObservationCountBySample(samples, altAlleles);
    //map<string, pair<int, int> > refObservationsBySample = getObservationCountBySample(samples, referenceBase);
    int sampleCov;
    pair<int,int> sampleRefObs;
    vector<pair<int,int> > altAlleleObsSample;
    for (vector<string>::iterator its = sampleNames.begin(); its != sampleNames.end(); ++its) {
      string& sampleName = *its;
      sampleCov = coverageSample[sampleName];
      //sampleRefObs = refObservationsBySample[sampleName];
      //altAlleleObsSample = altAlleleCountsBySample[sampleName];
      //var->sampleNames.push_back(sampleName);
      //var->outputSampleNames.push_back(sampleName);
      map<string, vector<string> >& sampleOutput = var->samples[sampleName];
      sampleOutput["DP"].push_back(convertToString(sampleCov));
      sampleOutput["RO"].push_back(convertToString(sampleCov));
      sampleOutput["SRF"].push_back(convertToString(0));
      sampleOutput["SRR"].push_back(convertToString(0));
      for (vector<pair<int,int> >::iterator obsitr = altAlleleObsSample.begin(); obsitr != altAlleleObsSample.end(); ++obsitr) {
        sampleOutput["AO"].push_back(convertToString(0));
        sampleOutput["SAF"].push_back(convertToString(0));
        sampleOutput["SAR"].push_back(convertToString(0));
      }



    }
    return true;
}

//@TODO: is this  a method for CandidateGenerationHelper?
bool generateCandidateVariant(AlleleParser * parser, Samples &samples, vcf::Variant * var,  bool &isHotSpot, ExtendParameters * parameters, int allowedAlleleTypes) {
  string cb = parser->currentReferenceBaseString();
  Allele nullAllele = genotypeAllele(ALLELE_NULL, "N", 1, "1N");

  if (cb != "A" && cb != "T" && cb != "C" && cb != "G") {
    return false;
  }
  if (!parser->inTarget()) {
    return false;
  }

  int coverage = countAlleles(samples);

  if (parameters->program_flow.DEBUG)
    cerr << "position: " << parser->currentSequenceName << ":" << (long unsigned int) parser->currentPosition + 1 << " coverage: " << coverage << endl;

  if (!parser->hasInputVariantAllelesAtCurrentPosition()) {
    isHotSpot = false;
    if (parameters->program_flow.inputPositionsOnly) //skip all non-hot-spot positions
      return false;
    // skips 0-coverage regions
    if (parameters->program_flow.DEBUG)
      cerr << "coverage " << parser->currentSequenceName << ":" << parser->currentPosition << " == " << coverage << endl;

    if (coverage == 0 || coverage <parameters->minCoverage) {
      return false;
    }

    if (!sufficientAlternateObservations(samples,  parameters->sampleName, parameters->minAltCount,  parameters->minAltFraction)) {
      if (parameters->program_flow.DEBUG)
        cerr << "Insufficient alt alleles" << endl;
      return false;
    }


  } else {
    isHotSpot = true;
  }
  vector<string> sampleListPlusRef;

  for (vector<string>::iterator s = parser->sampleList.begin(); s != parser->sampleList.end(); ++s) {
    sampleListPlusRef.push_back(*s);
  }
  if (parameters->useRefAllele)
    sampleListPlusRef.push_back(parser->currentSequenceName);

  // establish genotype alleles using input filters
  map<string, vector<Allele*> > alleleGroups;
  groupAlleles(samples, alleleGroups);
  //DEBUG2("grouped alleles by equivalence");

  vector<Allele> genotypeAlleles = parser->genotypeAlleles(alleleGroups, samples, parameters->onlyUseInputAlleles);

  // always include the reference allele as a possible genotype, even when we don't include it by default
  if (!parameters->useRefAllele) {
    vector<Allele> refAlleleVector;
    refAlleleVector.push_back(genotypeAllele(ALLELE_REFERENCE, string(1, parser->currentReferenceBase), 1, "1M"));
    genotypeAlleles = alleleUnion(genotypeAlleles, refAlleleVector);
  }

  // build haplotype alleles matching the current longest allele (often will do nothing)
  // this will adjust genotypeAlleles if changes are made
  //haplotype alleles are generated only when we are processing denovo variants
  if (!parameters->onlyUseInputAlleles)
    parser->buildHaplotypeAlleles(genotypeAlleles, samples, alleleGroups, allowedAlleleTypes);

  // always include the reference allele as a possible genotype, even when we don't include it by default
  if (!parameters->useRefAllele) {
    vector<Allele> refAlleleVector;
    refAlleleVector.push_back(genotypeAllele(ALLELE_REFERENCE, string(1, parser->currentReferenceBase), 1, "1M"));
    genotypeAlleles = alleleUnion(genotypeAlleles, refAlleleVector);
  }


  // re-calculate coverage, as this could change now that we've built haplotype alleles
  //coverage = countAlleles(samples);
  if (genotypeAlleles.size() <= 1) { // if we have only one viable allele, we don't have evidence for variation at this site
    // DEBUG2("no alternate genotype alleles passed filters at " << parser->currentSequenceName << ":" << parser->currentPosition);
    return false;
  }
  // add the null genotype
  if (parameters->excludeUnobservedGenotypes && genotypeAlleles.size() > 2) {
    genotypeAlleles.push_back(nullAllele);
  }
  // for each possible ploidy in the dataset, generate all possible genotypes
  //vector<int> ploidies = parser->currentPloidies(samples);
  //map<int, vector<Genotype> > genotypesByPloidy = getGenotypesByPloidy(ploidies, genotypeAlleles);
  Results results;

  map<string, int> inputAlleleCounts;

  GenotypeCombo bestCombo; // = NULL;

  string referenceBase = parser->currentReferenceBaseString();

  map<string, int> repeats;
  if (parameters->showReferenceRepeats) {
    repeats = parser->repeatCounts(parser->currentSequencePosition(), parser->currentSequence, 12);
  }

  vector<Allele> alts;
  if (parameters->onlyUseInputAlleles ||parameters->reportAllHaplotypeAlleles) {
    //alts = genotypeAlleles;
    for (vector<Allele>::iterator a = genotypeAlleles.begin(); a != genotypeAlleles.end(); ++a) {
      if (!a->isReference()) {
        alts.push_back(*a);
      }
    }
  } else {
    // get the unique alternate alleles in this combo, sorted by frequency in the combo
    vector<pair<Allele, int> > alternates = alternateAlleles(bestCombo, referenceBase);
    for (vector<pair<Allele, int> >::iterator a = alternates.begin(); a != alternates.end(); ++a) {
      Allele& alt = a->first;
      if (!alt.isNull())
        alts.push_back(alt);
    }
    // if there are no alternate alleles in the best combo, use the genotype alleles
    // XXX ...
    if (alts.empty()) {
      for (vector<Allele>::iterator a = genotypeAlleles.begin(); a != genotypeAlleles.end(); ++a) {
        if (!a->isReference()) {
          alts.push_back(*a);
        }
      }
    }
  }

  long int referencePosition = (long int) parser->currentPosition; // 0-based

  // remove alt alleles
  vector<Allele> altAlleles;
  for (vector<Allele>::iterator aa = alts.begin(); aa != alts.end(); ++aa) {
    if (!aa->isNull()) {
      altAlleles.push_back(*aa);
    }
  }

  // adjust reference position, reference sequence, and alt alleles by
  // stripping invariant bases off the beginning and end of the alt alleles
  // first we find the minimum start and end matches
  vector<Allele> adjustedAltAlleles;
  if (parameters->onlyUseInputAlleles) {
    //if we are using only input alleles then report them as is, no need to adjust the alleles and their position
    adjustedAltAlleles = altAlleles;
  }
  else {
    int minStartMatch = 0;
    int minEndMatch = 0;
    for (vector<Allele>::iterator aa = altAlleles.begin(); aa != altAlleles.end(); ++aa) {
      vector<pair<int, string> > cigar = splitCigar(aa->cigar);
      int startmatch = 0;
      int endmatch = 0;
      if (cigar.front().second == "M") {
        startmatch = cigar.front().first;
      }
      if (cigar.back().second == "M") {
        endmatch = cigar.back().first;
      }
      // check excludes complex alleles of the form, e.g. 1X3I
      if (cigar.size() > 1 && cigar.front().second == "M" && (cigar.at(1).second == "D" || cigar.at(1).second == "I")) {
        startmatch -= 1; // require at least one base flanking deletions
      }
      if (aa == altAlleles.begin()) {
        minStartMatch = startmatch;
        minEndMatch = endmatch;
      } else {
        minStartMatch = min(minStartMatch, startmatch);
        minEndMatch = min(minEndMatch, endmatch);
      }
    }
    // if either is non-zero, we have to adjust cigars and alternate sequences to be printed
    // this is done solely for reporting, so the altAlleles structure is used
    // for stats generation out of the ML genotype combination
    map<string, string> adjustedCigar;
    if (minStartMatch || minEndMatch) {
      for (vector<Allele>::iterator aa = altAlleles.begin(); aa != altAlleles.end(); ++aa) {
        // subtract the minStartMatch and minEndMatch bases from the allele start and end
        adjustedAltAlleles.push_back(*aa);
        Allele& allele = adjustedAltAlleles.back();
        vector<pair<int, string> > cigar = splitCigar(allele.cigar);
        // TODO clean this up by writing a wrapper for Allele::subtract() (?)
        if (cigar.front().second == "M") {
          cigar.front().first -= minStartMatch;
          allele.alternateSequence = allele.alternateSequence.substr(minStartMatch);
        }
        if (cigar.back().second == "M") {
          cigar.back().first -= minEndMatch;
          allele.alternateSequence = allele.alternateSequence.substr(0, allele.alternateSequence.size() - minEndMatch);
        }
        allele.cigar = joinCigar(cigar);
        allele.position += minStartMatch;
        allele.referenceLength -= minStartMatch + minEndMatch;
        adjustedCigar[aa->base()] = allele.cigar;
      }
      referencePosition += minStartMatch;
    } else {
      adjustedAltAlleles = altAlleles;
      for (vector<Allele>::iterator aa = altAlleles.begin(); aa != altAlleles.end(); ++aa) {
        adjustedCigar[aa->base()] = aa->cigar;
      }
    }

  }//end if useInputAllelesOnly

  var->ref = parser->referenceSubstr(referencePosition, 1);

  // the reference sequence should be able to encompass all events at the site, +1bp on the left
  for (vector<Allele>::iterator aa = adjustedAltAlleles.begin(); aa != adjustedAltAlleles.end(); ++aa) {

    Allele& altAllele = *aa;
    switch (altAllele.type) {
      case ALLELE_SNP:
      case ALLELE_REFERENCE:
      case ALLELE_MNP:
        if (var->ref.size() < altAllele.referenceLength) {
          var->ref = parser->referenceSubstr(referencePosition, altAllele.referenceLength);
        }
        break;
      case ALLELE_DELETION:
        // extend the reference sequence
        if (var->ref.size() < altAllele.referenceLength) {
          var->ref = parser->referenceSubstr(referencePosition, altAllele.referenceLength);
        }
        break;
      case ALLELE_INSERTION:
        break;
      case ALLELE_COMPLEX:
        if (var->ref.size() < altAllele.referenceLength) {
          var->ref = parser->referenceSubstr(referencePosition, altAllele.referenceLength);
        }
        break;
      default:
        cerr << "Unhandled allele type: " << altAllele.typeStr() << endl;
        break;
    }

  }



  for (vector<Allele>::iterator aa = adjustedAltAlleles.begin(); aa != adjustedAltAlleles.end(); ++aa) {
    Allele& altAllele = *aa;
    string altseq;
    switch (altAllele.type) {
      case ALLELE_REFERENCE:
        break;
      case ALLELE_SNP:
      case ALLELE_MNP:
        altseq = var->ref;
        altseq.replace(0, altAllele.alternateSequence.size(), altAllele.alternateSequence);
        var->alt.push_back(altseq);
        break;
      case ALLELE_DELETION:
      case ALLELE_INSERTION: // XXX is this correct???
      case ALLELE_COMPLEX:
        var->alt.push_back(altAllele.alternateSequence);
        break;
      default:
        cerr << "Unhandled allele type: " << altAllele.typeStr() << endl;
        break;
    }
  }

  //HACK to account for Duplicate alleles caused by Hotspot alleles not being represented the same as freebayes alleles
      //Real fix is to change the represent hotspot alleles so we can detect duplicates within freebayes
      vector<string>::iterator a1 = var->alt.begin();
      vector<Allele>::iterator a2 = altAlleles.begin();
      map<string, string> tempAltStrings;
      map<string, string>::iterator mapItr;
      string altstr;
      while( a1 != var->alt.end() &&  a2 != altAlleles.end() )
      {
        altstr = (*a1);

        mapItr = tempAltStrings.find(altstr);
        if (mapItr == tempAltStrings.end() ) //alt allele sequence not found
        {
          tempAltStrings[altstr] = "found";
          a1++;
          a2++;
        }
        else { //altstr is a duplicate and remove it from adjust and alt alleles vector
          a1 = var->alt.erase(a1);
          a2 = altAlleles.erase(a2);
          cerr << "Duplicate allele found in Variant position " <<  referencePosition + 1 << " duplicate allele = " << altstr << endl;
        }

      }

  assert(!var->ref.empty());
  for (vector<string>::iterator a = var->alt.begin(); a != var->alt.end(); ++a) {
    assert(!a->empty());
    if (*a == var->ref) {
      cerr << "variant at " << parser->currentSequenceName << ":" << referencePosition + 1 << endl;
      cerr << "alt is the same as the reference" << endl;
      cerr << *a << " == " << var->ref << endl;
    }
  }
  var->sequenceName = parser->currentSequenceName;
  // XXX this should be the position of the matching reference haplotype
  var->position = referencePosition + 1;
  var->id = ".";
  var->filter = ".";
  // XXX this should be the size of the maximum deletion + 1bp on the left end
  var->quality = 0.0;

  SetUpFormatString(var);
 
  //clear all the info tags before populating them
  clearInfoTags(var);

  pair<int,int> refObservationCountByStrand = getObservationCount(samples, referenceBase);
  var->info["RO"].push_back(convertToString(refObservationCountByStrand.first + refObservationCountByStrand.second));
  var->info["SRF"].push_back(convertToString(refObservationCountByStrand.first));
  var->info["SRR"].push_back(convertToString(refObservationCountByStrand.second));

  pair<int,int> altObservationCountByStrand;
  vector<pair<int, int> > altObservationCountByStrandVector;

  for (vector<Allele>::iterator aa = altAlleles.begin(); aa != altAlleles.end(); ++aa) {

    Allele& altAllele = *aa;
    string altbase = altAllele.base();
    altObservationCountByStrand = getObservationCount(samples, altbase);
    altObservationCountByStrandVector.push_back(altObservationCountByStrand);

    if (altAllele.type == ALLELE_DELETION) {
      var->info["TYPE"].push_back("del");

    } else
      if (altAllele.type == ALLELE_INSERTION) {
        var->info["TYPE"].push_back("ins");
      } else
        if (altAllele.type == ALLELE_COMPLEX) {
          var->info["TYPE"].push_back("complex");
        } else
          if (altAllele.type == ALLELE_SNP) {
            var->info["TYPE"].push_back("snp");

          } else
            if (altAllele.type == ALLELE_MNP) {
              var->info["TYPE"].push_back("mnp");
            } else {
              cout << "What is this?" << endl;
              cout << altAllele.type << endl;
              cout << altAllele << endl;
            }
    var->info["LEN"].push_back(convert(altAllele.length));

    var->info["AO"].push_back(convertToString(altObservationCountByStrand.first + altObservationCountByStrand.second));
    var->info["SAF"].push_back(convertToString(altObservationCountByStrand.first));
    var->info["SAR"].push_back(convertToString(altObservationCountByStrand.second));
    var->info["HRUN"].push_back(convertToString(parser->homopolymerRunLeft(altbase) + 1 + parser->homopolymerRunRight(altbase)));
  }

  if (isHotSpot)
    var->infoFlags["HS"] = true;
  //insert depth
  var->info["DP"].push_back(convertToString(coverage));
  vector<string> sampleNames = parser->sampleList;
  // samples
  //now calculate coverage (DP), referece observations (RO) and alt allele counts (AO) for each sample for format field
  map<string,int> coverageSample = countAllelesBySample(samples);
  map<string, vector<pair<int, int> > > altAlleleCountsBySample = getObservationCountBySample(samples, altAlleles);
  map<string, pair<int, int> > refObservationsBySample = getObservationCountBySample(samples, referenceBase);
  int sampleCov;
  pair<int,int> sampleRefObs;
  vector<pair<int,int> > altAlleleObsSample;
  for (vector<string>::iterator its = sampleNames.begin(); its != sampleNames.end(); ++its) {
    string& sampleName = *its;
    sampleCov = coverageSample[sampleName];
    sampleRefObs = refObservationsBySample[sampleName];
    altAlleleObsSample = altAlleleCountsBySample[sampleName];
    //var->sampleNames.push_back(sampleName);
    //var->outputSampleNames.push_back(sampleName);
    map<string, vector<string> >& sampleOutput = var->samples[sampleName];
    sampleOutput["DP"].push_back(convertToString(sampleCov));
    sampleOutput["RO"].push_back(convertToString(sampleRefObs.first + sampleRefObs.second));
    sampleOutput["SRF"].push_back(convertToString(sampleRefObs.first));
    sampleOutput["SRR"].push_back(convertToString(sampleRefObs.second));
    for (vector<pair<int,int> >::iterator obsitr = altAlleleObsSample.begin(); obsitr != altAlleleObsSample.end(); ++obsitr) {
      sampleOutput["AO"].push_back(convertToString((*obsitr).first + (*obsitr).second));
      sampleOutput["SAF"].push_back(convertToString((*obsitr).first));
      sampleOutput["SAR"].push_back(convertToString((*obsitr).second));
    }



  }
  return true;

}
