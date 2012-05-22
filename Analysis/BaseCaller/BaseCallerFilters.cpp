/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdio.h>
#include <iomanip>
#include "BaseCaller.h"
#include "LinuxCompat.h"
#include "Stats.h"
#include "IonErr.h"

#include "SFFTrim/QScoreTrim.h"
#include "SFFTrim/adapter_searcher.h"

using namespace std;

enum FilteringOutcomes {
  kUninitialized,
  kPassed,

  kFilterZeroBases,
  kFilterShortRead,
  kFilterFailedKeypass,
  kFilterHighPPF,
  kFilterPolyclonal,
  kFilterHighResidual,
  kFilterBeverly,

  kBkgmodelHighPPF,
  kBkgmodelPolyclonal,
  kBkgmodelFailedKeypass,

  kFilteredShortAdapterTrim,
  kFilteredShortQualityTrim
};

class ThousandsSeparator : public numpunct<char> {
protected:
    string do_grouping() const { return "\03"; }
};



void BaseCallerFilters::PrintHelp()
{
  fprintf (stdout, "Filtering and trimming options:\n");
  fprintf (stdout, "  -k,--keypass-filter        on/off     apply keypass filter [on]\n");
  fprintf (stdout, "     --clonal-filter-solve   on/off     apply polyclonal filter [off]\n");
  fprintf (stdout, "     --clonal-filter-tf      on/off     apply polyclonal filter to TFs [off]\n");
  fprintf (stdout, "     --clonal-filter-train   on/off     train polyclonal filter even when filter is disabled [off]\n");
  fprintf (stdout, "     --min-read-length       INT        apply minimum read length filter [8]\n");
  fprintf (stdout, "     --cr-filter             on/off     apply cafie residual filter [off]\n");
  fprintf (stdout, "     --cr-filter-tf          on/off     apply cafie residual filter to TFs [off]\n");
  fprintf (stdout, "     --cr-filter-max-value   FLOAT      cafie residual filter threshold [0.8]\n");
  fprintf (stdout, "     --beverly-filter        FLOAT,FLOAT,INT/off  (filter_ratio,trim_ratio,min_length)\n");
  fprintf (stdout, "                                        apply Beverly filter/trimmer [0.03,0.03,8]\n");
  fprintf (stdout, "\n");
  fprintf (stdout, "Options adapted from SFFTrim:\n");
  fprintf (stdout, "     --trim-adapter          STRING     reverse complement of adapter sequence [ATCACCGACTGCCCATAGAGAGGCTGAGAC]\n");
  fprintf (stdout, "     --trim-adapter-cutoff   FLOAT      cutoff for adapter trimming [0.0 = disabled]\n");
  fprintf (stdout, "     --trim-adapter-pick-closest on/off use closest candidate match, rather than longest [off]\n");
  fprintf (stdout, "     --trim-qual-window-size INT        window size for quality trimming [30]\n");
  fprintf (stdout, "     --trim-qual-cutoff      FLOAT      cutoff for quality trimming [100.0 = disabled]\n");
  fprintf (stdout, "     --trim-min-read-len     INT        reads trimmed shorter than this are omitted from output [8]\n");
  fprintf (stdout, "     --bead-summary          on/off     generate bead summary file [off]\n");
  fprintf (stdout, "\n");
}


BaseCallerFilters::BaseCallerFilters(OptArgs& opts,
    const string& _flowOrder, int _numFlows, const vector<KeySequence>& _keys, Mask *_maskPtr)
{
  flowOrder = _flowOrder;

  keypassFilter                   = opts.GetFirstBoolean('k', "keypass-filter", true);
  percentPositiveFlowsFilterTFs   = opts.GetFirstBoolean('-', "clonal-filter-tf", false);
  clonalFilterTraining            = opts.GetFirstBoolean('-', "clonal-filter-train", false);
  clonalFilterSolving             = opts.GetFirstBoolean('-', "clonal-filter-solve", false);
  minReadLength                   = opts.GetFirstInt    ('-', "min-read-length", 8);
  cafieResFilterCalling           = opts.GetFirstBoolean('-', "cr-filter", false);
  cafieResFilterTFs               = opts.GetFirstBoolean('-', "cr-filter-tf", false);
  generate_bead_summary_          = opts.GetFirstBoolean('-', "bead-summary", false);

  // TODO: get this to work right. May require "unwound" flow order, so incompatible with current wells.FlowOrder()
  //flt_control.cafieResMaxValueByFlowOrder[std::string ("TACG") ] = 0.06;  // regular flow order
  //flt_control.cafieResMaxValueByFlowOrder[std::string ("TACGTACGTCTGAGCATCGATCGATGTACAGC") ] = 0.08;  // xdb flow order

  cafieResMaxValue = opts.GetFirstDouble('-',  "cr-filter-max-value", 0.08);

  // SFFTrim options
  trim_adapter = opts.GetFirstString('-', "trim-adapter", "ATCACCGACTGCCCATAGAGAGGCTGAGAC");
  trim_adapter_cutoff = opts.GetFirstDouble('-', "trim-adapter-cutoff", 0.0);
  trim_adapter_closest = opts.GetFirstBoolean('-', "trim-adapter-pick-closest", false);
  trim_qual_wsize = opts.GetFirstInt('-', "trim-qual-window-size", 30);
  trim_qual_cutoff = opts.GetFirstDouble('-', "trim-qual-cutoff", 100.0);
  trim_min_read_len = opts.GetFirstInt('-', "trim-min-read-len", 8);


  // Validate options

  if (minReadLength < 1) {
    fprintf (stderr, "Option Error: min-read-length must specify a positive value (%d invalid).\n", minReadLength);
    exit (EXIT_FAILURE);
  }
  if (cafieResMaxValue <= 0) {
    fprintf (stderr, "Option Error: cr-filter-max-value must specify a positive value (%lf invalid).\n", cafieResMaxValue);
    exit (EXIT_FAILURE);
  }

  keys = _keys;
  numClasses = keys.size();

  assert(numClasses == 2);
  classFilterPolyclonal.resize(numClasses);
  classFilterPolyclonal[0] = clonalFilterSolving;
  classFilterPolyclonal[1] = clonalFilterSolving && percentPositiveFlowsFilterTFs;
  classFilterHighResidual.resize(numClasses);
  classFilterHighResidual[0] = cafieResFilterCalling;
  classFilterHighResidual[1] = cafieResFilterCalling && cafieResFilterTFs;


  string filter_beverly_args      = opts.GetFirstString('-', "beverly-filter", "0.03,0.03,8");
  if (filter_beverly_args == "off") {
    filter_beverly_enabled_ = false; // Nothing, really
    printf("Beverly filter: disabled, use --beverly-filter=filter_ratio,trim_ratio,min_length\n");

  } else {
    int stat = sscanf (filter_beverly_args.c_str(), "%f,%f,%d",
        &filter_beverly_filter_ratio_,
        &filter_beverly_trim_ratio_,
        &filter_beverly_min_read_length_);
    if (stat != 3) {
      fprintf (stderr, "Option Error: beverly-filter %s\n", filter_beverly_args.c_str());
      fprintf (stderr, "Usage: --beverly-filter=filter_ratio,trim_ratio,min_length\n");
      exit (EXIT_FAILURE);
    }
    filter_beverly_enabled_ = true;
    printf("Beverly filter: enabled, use --beverly-filter=off to disable\n");
    printf("Beverly filter: filter_ratio = %1.5f\n", filter_beverly_filter_ratio_);
    printf("Beverly filter: trim_ratio = %1.5f\n", filter_beverly_trim_ratio_);
    printf("Beverly filter: min_length = %d\n", filter_beverly_min_read_length_);
  }

  maskPtr = _maskPtr;
  numFlows = _numFlows;

  filterMask.assign(maskPtr->H()*maskPtr->W(), kUninitialized);
}


void BaseCallerFilters::FindClonalPopulation(const string& outputDirectory, RawWells *wellsPtr, int nUnfilteredLib)
{
  if (clonalFilterSolving or clonalFilterTraining) {
    wellsPtr->OpenForIncrementalRead();
    vector<int> keyIonogram(keys[0].flows(), keys[0].flows()+keys[0].flows_length());
    filter_counts counts;
    int nlib = maskPtr->GetCount(static_cast<MaskType> (MaskLib));
    counts._nsamp = min(nlib, nUnfilteredLib); // In the future, a parameter separate from nUnfilteredLib
    make_filter(clonalPopulation, counts, *maskPtr, *wellsPtr, keyIonogram);
    cout << counts << endl;
    wellsPtr->Close();
  }
}




void BaseCallerFilters::TransferFilteringResultsToMask(Mask *myMask)
{
  assert(myMask->H()*myMask->W() == (int)filterMask.size());

  for (size_t idx = 0; idx < filterMask.size(); idx++) {

    (*myMask)[idx] &= MaskAll - MaskFilteredBadPPF - MaskFilteredShort - MaskFilteredBadKey - MaskFilteredBadResidual - MaskKeypass;

    switch (filterMask[idx]) {
      case kPassed:                     (*myMask)[idx] |= MaskKeypass; break;

      case kFilterZeroBases:            (*myMask)[idx] |= MaskFilteredShort; break;
      case kFilterShortRead:            (*myMask)[idx] |= MaskFilteredShort; break;
      case kFilterFailedKeypass:        (*myMask)[idx] |= MaskFilteredBadKey; break;
      case kFilterHighPPF:              (*myMask)[idx] |= MaskFilteredBadPPF; break;
      case kFilterPolyclonal:           (*myMask)[idx] |= MaskFilteredBadPPF; break;
      case kFilterHighResidual:         (*myMask)[idx] |= MaskFilteredBadResidual; break;
      case kFilterBeverly:              (*myMask)[idx] |= MaskFilteredBadResidual; break;

      case kBkgmodelFailedKeypass:      (*myMask)[idx] |= MaskFilteredBadKey; break;
      case kBkgmodelHighPPF:            (*myMask)[idx] |= MaskFilteredBadPPF; break;
      case kBkgmodelPolyclonal:         (*myMask)[idx] |= MaskFilteredBadPPF; break;

      case kFilteredShortAdapterTrim:   (*myMask)[idx] |= MaskFilteredShort; break;
      case kFilteredShortQualityTrim:   (*myMask)[idx] |= MaskFilteredShort; break;
    }
  }
}


int BaseCallerFilters::getNumWellsCalled()
{
  int cntTotal = 0;

  for (size_t idx = 0; idx < filterMask.size(); idx++)
    if (filterMask[idx] != kUninitialized)
      cntTotal++;

  return cntTotal;
}

void BaseCallerFilters::GenerateFilteringStatistics(Json::Value &filterSummary)
{
  vector<int> cntValid(numClasses, 0);
  vector<int> cntZeroBases(numClasses, 0);
  vector<int> cntShortRead(numClasses, 0);
  vector<int> cntFailedKeypass(numClasses, 0);
  vector<int> cntHighPPF(numClasses, 0);
  vector<int> cntPolyclonal(numClasses, 0);
  vector<int> cntHighResidual(numClasses, 0);
  vector<int> cntBeverly(numClasses, 0);

  vector<int> cntBkgmodelFailedKeypass(numClasses, 0);
  vector<int> cntBkgmodelHighPPF(numClasses, 0);
  vector<int> cntBkgmodelPolyclonal(numClasses, 0);

  vector<int> cntShortAdapterTrim(numClasses, 0);
  vector<int> cntShortQualityTrim(numClasses, 0);

  vector<int> cntTotal(numClasses, 0);

  for (size_t idx = 0; idx < filterMask.size(); idx++) {
    int iClass = maskPtr->Match(idx, MaskLib) ? 0 : 1;  // Dependency on mask...

    if (filterMask[idx] != kUninitialized)
      cntTotal[iClass]++;

    switch (filterMask[idx]) {
      case kPassed:                     cntValid[iClass]++; break;

      case kFilterZeroBases:            cntZeroBases[iClass]++; break;
      case kFilterShortRead:            cntShortRead[iClass]++; break;
      case kFilterFailedKeypass:        cntFailedKeypass[iClass]++; break;
      case kFilterHighPPF:              cntHighPPF[iClass]++; break;
      case kFilterPolyclonal:           cntPolyclonal[iClass]++; break;
      case kFilterHighResidual:         cntHighResidual[iClass]++; break;
      case kFilterBeverly:              cntBeverly[iClass]++; break;

      case kBkgmodelFailedKeypass:      cntBkgmodelFailedKeypass[iClass]++; break;
      case kBkgmodelHighPPF:            cntBkgmodelHighPPF[iClass]++; break;
      case kBkgmodelPolyclonal:         cntBkgmodelPolyclonal[iClass]++; break;

      case kFilteredShortAdapterTrim:   cntShortAdapterTrim[iClass]++; break;
      case kFilteredShortQualityTrim:   cntShortQualityTrim[iClass]++; break;
    }
  }

  ostringstream table;
  table.imbue(locale(table.getloc(), new ThousandsSeparator));

  table << endl;
  table << setw(25) << " ";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << (keys[iClass].name() + " (" + keys[iClass].bases() + ")");
  table << endl;

  table << setw(25) << "Examined wells";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << cntTotal[iClass] ;
  table << endl;

  table << setw(26) << " ";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << "------------";
  table << endl;

  table << setw(25) << "BkgModel:   High PPF";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntBkgmodelHighPPF[iClass] ;
  table << endl;

  table << setw(25) << "BkgModel: Polyclonal";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntBkgmodelPolyclonal[iClass] ;
  table << endl;

  table << setw(25) << "BkgModel:    Bad key";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntBkgmodelFailedKeypass[iClass] ;
  table << endl;

  table << setw(25) << "High PPF";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntHighPPF[iClass] ;
  table << endl;

  table << setw(25) << "Polyclonal";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntPolyclonal[iClass] ;
  table << endl;

  table << setw(25) << "Zero bases";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntZeroBases[iClass] ;
  table << endl;

  table << setw(25) << "Short read";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntShortRead[iClass] ;
  table << endl;

  table << setw(25) << "Bad key";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntFailedKeypass[iClass] ;
  table << endl;

  table << setw(25) << "High residual";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntHighResidual[iClass] ;
  table << endl;

  table << setw(25) << "Beverly filter";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntBeverly[iClass] ;
  table << endl;

  table << setw(25) << "Short after adapter trim";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntShortAdapterTrim[iClass] ;
  table << endl;

  table << setw(25) << "Short after quality trim";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << -cntShortQualityTrim[iClass] ;
  table << endl;

  table << setw(26) << " ";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << "------------";
  table << endl;

  table << setw(25) << "Valid reads saved to SFF";
  for (int iClass = 0; iClass < numClasses; iClass++)
    table << setw(15) << cntValid[iClass] ;
  table << endl;
  table << endl;

  cout << table.str();

  for (int iClass = 0; iClass < numClasses; iClass++) {
    filterSummary[keys[iClass].name()]["key"]         = keys[iClass].bases();
    filterSummary[keys[iClass].name()]["polyclonal"]  = cntPolyclonal[iClass] + cntBkgmodelPolyclonal[iClass];
    filterSummary[keys[iClass].name()]["highPPF"]     = cntHighPPF[iClass] + cntBkgmodelHighPPF[iClass];
    filterSummary[keys[iClass].name()]["zero"]        = cntZeroBases[iClass];
    filterSummary[keys[iClass].name()]["short"]       = cntShortRead[iClass] + cntShortAdapterTrim[iClass] + cntShortQualityTrim[iClass];
    filterSummary[keys[iClass].name()]["badKey"]      = cntFailedKeypass[iClass] + cntBkgmodelFailedKeypass[iClass];
    filterSummary[keys[iClass].name()]["highRes"]     = cntHighResidual[iClass] + cntBeverly[iClass];
    filterSummary[keys[iClass].name()]["valid"]       = cntValid[iClass];
  }


  if (generate_bead_summary_) {

    ofstream outFile;
    outFile.open("beadSummary.filtered.txt");
    if(outFile.fail()) {
      ION_WARN(string("Unable to open output bead summary file ") + "beadSummary.filtered.txt" + string(" for write"));
      return;
    }

    string delim = "\t";
    outFile << "class" << delim;
    outFile << "key" << delim;
    outFile << "polyclonal" << delim;
    outFile << "highPPF" << delim;
    outFile << "zero" << delim;
    outFile << "short" << delim;
    outFile << "badKey" << delim;
    outFile << "highRes" << delim;
    outFile << "clipAdapter" << delim;
    outFile << "clipQual" << delim;
    outFile << "valid" << endl;

    for (int iClass = 0; iClass < numClasses; iClass++) {

      outFile << keys[iClass].name() << delim
          << keys[iClass].bases() << delim
          << (cntPolyclonal[iClass] + cntBkgmodelPolyclonal[iClass]) << delim
          << (cntHighPPF[iClass] + cntBkgmodelHighPPF[iClass]) << delim
          << (cntZeroBases[iClass]) << delim
          << (cntShortRead[iClass]) << delim
          << (cntFailedKeypass[iClass] + cntBkgmodelFailedKeypass[iClass]) << delim
          << (cntHighResidual[iClass] + cntBeverly[iClass]) << delim
          << cntShortAdapterTrim[iClass] << delim
          << cntShortQualityTrim[iClass] << delim
          << cntValid[iClass]  << endl;
    }
  }
}



void BaseCallerFilters::markReadAsValid(int readIndex)
{
  filterMask[readIndex] = kPassed;
}

bool BaseCallerFilters::isValid(int readIndex)
{
  return filterMask[readIndex] == kPassed;
}

bool BaseCallerFilters::isPolyclonal(int readIndex)
{
  return filterMask[readIndex] == kFilterPolyclonal;
}




void BaseCallerFilters::forceBkgmodelHighPPF(int readIndex)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;
  filterMask[readIndex] = kBkgmodelHighPPF;
}

void BaseCallerFilters::forceBkgmodelPolyclonal(int readIndex)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;
  filterMask[readIndex] = kBkgmodelPolyclonal;
}

void BaseCallerFilters::forceBkgmodelFailedKeypass(int readIndex)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;
  filterMask[readIndex] = kBkgmodelFailedKeypass;
}


void BaseCallerFilters::tryFilteringHighPPFAndPolyclonal (int readIndex, int iClass, const vector<float>& measurements)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;

  if (!classFilterPolyclonal[iClass])  // Filter disabled?
    return;

  if (clonalFilterSolving) {
    vector<float>::const_iterator first = measurements.begin() + mixed_first_flow();
    vector<float>::const_iterator last  = measurements.begin() + mixed_last_flow();
    float ppf = percent_positive(first, last);
    float ssq = sum_fractional_part(first, last);

    if(ppf > mixed_ppf_cutoff())
      filterMask[readIndex] = kFilterHighPPF;
    else if(!clonalPopulation.is_clonal(ppf, ssq))
      filterMask[readIndex] = kFilterPolyclonal;
  }
}


void BaseCallerFilters::tryFilteringZeroBases(int readIndex, int iClass, const SFFWriterWell& readResults)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;

  if(readResults.numBases == 0)
    filterMask[readIndex] = kFilterZeroBases;
}


void BaseCallerFilters::tryFilteringShortRead(int readIndex, int iClass, const SFFWriterWell& readResults)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;

  if(readResults.numBases < minReadLength)
    filterMask[readIndex] = kFilterShortRead;
}



void BaseCallerFilters::tryFilteringFailedKeypass(int readIndex, int iClass, const vector<char> &solution)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;

  if(!keypassFilter)  // Filter disabled?
    return;

  for (int iFlow = 0; iFlow < (keys[iClass].flows_length()-1); iFlow++)
    if (keys[iClass][iFlow] != solution[iFlow])
      filterMask[readIndex] = kFilterFailedKeypass;
  // keypass last flow
  if (keys[iClass][keys[iClass].flows_length()-1] > solution[keys[iClass].flows_length()-1])
    filterMask[readIndex] = kFilterFailedKeypass;
}



void BaseCallerFilters::tryFilteringHighResidual(int readIndex, int iClass, const vector<float>& residual)
{
  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;

  if(!classFilterHighResidual[iClass])  // Filter disabled?
    return;

  double medAbsResidual = getMedianAbsoluteCafieResidual(residual, CAFIE_RESIDUAL_FLOWS_N);

  if(medAbsResidual > cafieResMaxValue)
    filterMask[readIndex] = kFilterHighResidual;
}

void BaseCallerFilters::tryFilteringBeverly(int readIndex, int iClass, const BasecallerRead &read, SFFWriterWell& readResults)
{

  bool reject = false;

  if (filter_beverly_enabled_ and iClass == 0) {    // What about random reads? What about TFs?

    int num_onemers = 0;
    int num_twomers = 0;
    int num_extreme_onemers = 0;
    int num_extreme_twomers = 0;
    int num_bases_seen = 0;
    int max_trim_bases = 0;

    for (int flow = 0; flow < numFlows; flow++) {

      if (read.solution[flow] == 1) {
        num_onemers++;
        if (readResults.flowIonogram[flow] <= 59 or readResults.flowIonogram[flow] >= 140)
          num_extreme_onemers++;
      }

      if (read.solution[flow] == 2) {
        num_twomers++;
        if (readResults.flowIonogram[flow] <= 159 or readResults.flowIonogram[flow] >= 240)
          num_extreme_twomers++;
      }

      num_bases_seen += read.solution[flow];
      if (num_extreme_onemers <= num_onemers * filter_beverly_trim_ratio_)
        max_trim_bases = num_bases_seen;
    }

    if ((num_extreme_onemers + num_extreme_twomers) > (num_onemers + num_twomers) * filter_beverly_filter_ratio_) {
      if (max_trim_bases > filter_beverly_min_read_length_)
        readResults.clipQualRight = max_trim_bases;
      else
        reject = true;
    }
  }

  if (filterMask[readIndex] != kPassed) // Already filtered out?
    return;

  if(reject)
    filterMask[readIndex] = kFilterBeverly;
}




double BaseCallerFilters::getMedianAbsoluteCafieResidual(const vector<float> &residual, unsigned int nFlowsToAssess)
{
  double medAbsCafieRes = 0;

  unsigned int nFlow = min(residual.size(), (size_t) nFlowsToAssess);
  if (nFlow > 0) {
    vector<double> absoluteResid(nFlow);
    for (unsigned int iFlow = 0; iFlow < nFlow; iFlow++) {
      absoluteResid[iFlow] = abs(residual[iFlow]);
    }
    medAbsCafieRes = ionStats::median(absoluteResid);
  }

  return medAbsCafieRes;
}

//
// SFFTrim integration starts here
//




void BaseCallerFilters::tryTrimmingAdapter(int readIndex, int iClass, SFFWriterWell& readResults)
{
  if(trim_adapter_cutoff <= 0.0)  // Zero means disabled
    return;

  if (iClass != 0)  // Hardcoded: Don't trim TFs
    return;

  adapter_searcher as(flowOrder, keys[iClass].bases(), trim_adapter);
  int num_matches = as.find_matches_sff(&readResults.flowIonogram[0], numFlows, trim_adapter_cutoff);
  if(num_matches <= 0)
    return; // Adapter not found

  adapter_searcher::match match;
  if(trim_adapter_closest)
    match = as.pick_closest();
  else
    match = as.pick_longest();

  uint16_t clip_adapter_right = as.flow2pos(readResults.baseFlowIndex, readResults.baseCalls, readResults.numBases, match._flow);

  if (clip_adapter_right == 0)
    return;

  if (readResults.clipAdapterRight == 0 or clip_adapter_right < readResults.clipAdapterRight) {
    readResults.clipAdapterRight = clip_adapter_right; //Trim

    if (filterMask[readIndex] != kPassed) // Already filtered out?
      return;

    int trim_length = clip_adapter_right - max(1,max(readResults.clipAdapterLeft,readResults.clipQualLeft)) + 1;

    if (trim_length < trim_min_read_len) // Adapter trimming led to filtering
      filterMask[readIndex] = kFilteredShortAdapterTrim;
  }
}



void BaseCallerFilters::tryTrimmingQuality(int readIndex, int iClass, SFFWriterWell& readResults)
{
  if(trim_qual_cutoff >= 100.0)   // 100.0 or more means disabled
    return;

  if (iClass != 0)  // Hardcoded: Don't trim TFs
    return;


  uint8_t  *qbeg = &readResults.baseQVs[0];
  uint8_t  *qend = qbeg + readResults.numBases;
  uint8_t  *clip = QualTrim(qbeg, qend, trim_qual_cutoff, trim_qual_wsize);

  uint16_t clip_qual_right = clip - qbeg;

  if (readResults.clipQualRight == 0 or clip_qual_right < readResults.clipQualRight) {
    readResults.clipQualRight = clip_qual_right; //Trim

    if (filterMask[readIndex] != kPassed) // Already filtered out?
      return;

    int trim_length = clip_qual_right - max(1,max(readResults.clipAdapterLeft,readResults.clipQualLeft)) + 1;

    if (trim_length < trim_min_read_len) // Qualit trimming led to filtering
      filterMask[readIndex] = kFilteredShortQualityTrim;
  }
}



/*
void GenerateBeadSummary(const options& opt, int32_t cnt, int32_t nReads, int32_t dropped_by_adapter, int32_t dropped_by_qual)
{

  // Step 1. Load BaseCaller.json

  ifstream inJsonFile;
  inJsonFile.open(opt.basecaller_json.c_str());
  if(!inJsonFile.good()) {
    ION_WARN(string("Unable to open input bead summary file ") + opt.basecaller_json + string(" for read"));
    return;
  }
  Json::Value BaseCallerJson;
  inJsonFile >> BaseCallerJson;
  inJsonFile.close();

  Json::Value Summary = BaseCallerJson["BeadSummary"];

  // Step 2. Prepare to write trimmed BeadSummary file

  ofstream outFile;
  if(opt.out_bead_summary == "") {
    ION_WARN(string("Output bead summary file not specified, unable to update bead summary with reads trimmed to zero length"));
    return;
  } else {
    outFile.open(opt.out_bead_summary.c_str());
    if(outFile.fail()) {
      ION_WARN(string("Unable to open output bead summary file ") + opt.out_bead_summary + string(" for write"));
      return;
    }
  }

  // Step 3. Write header

*/












