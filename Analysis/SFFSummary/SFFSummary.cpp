/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "SFFSummary.h"

using namespace std;

SFFSummary::SFFSummary() {

  qual.resize(0);
  errThreshold.resize(0);
  readLength.resize(0);
  minReadLength.resize(0);

  nReads=0;
  nReadsByQual.clear();
  nReadsByQualLength.clear();
  nReadsByLength.clear();
  nBasesByQual.clear();
  maxLengthByQual.clear();
  meanLengthByQual.clear();
  keepPerReadData=false;
  reportPredictedQlen=false;
  readName.resize(0);
  perReadQualLength.resize(0);
  perReadLength.resize(0);

}

void SFFSummary::summaryStatInit(void) {
  // Initialize summary statistics
  nReads         = 0;
  for(unsigned int iQual=0; iQual < qual.size(); iQual++) {
    nReadsByQual[qual[iQual]]     = 0;
    nBasesByQual[qual[iQual]]     = 0;
    maxLengthByQual[qual[iQual]]  = 0;
    meanLengthByQual[qual[iQual]] = 0;
    for(unsigned int iLength=0; iLength < readLength.size(); iLength++)
      nReadsByQualLength[qual[iQual]][readLength[iLength]] = 0;
  }
  for(unsigned int iLength=0; iLength < readLength.size(); iLength++)
    nReadsByLength[readLength[iLength]] = 0;
}

void SFFSummary::summaryStatUpdate(vector <uint16_t> qScore, string rName) {

  nReads++;

  // Determine lengths at different error rates
  vector <unsigned int> thisReadLen;
  qualToErrLength(qScore, errThreshold, thisReadLen);

  // Collect summary stats
  double weight1,weight2;
  for(unsigned int iQual=0; iQual < qual.size(); iQual++) {

    // Confirm the read meets minimum length at whatever qScore we are filtering
    if( (minReadLength[iQual] > 0) && (thisReadLen[iQual] < minReadLength[iQual]) )
      continue;

    nReadsByQual[qual[iQual]] += 1;

    // Update number of bases attaining the quality score
    for(uint16_t iBase=0;  iBase < qScore.size(); iBase++) {
      if(qScore[iBase] >= qual[iQual])
        nBasesByQual[qual[iQual]] += 1;
    }

    // Update max length
    maxLengthByQual[qual[iQual]] = std::max(maxLengthByQual[qual[iQual]], thisReadLen[iQual]);

    // Update mean length
    weight1 = (double) (nReadsByQual[qual[iQual]]-1) / (double) nReadsByQual[qual[iQual]];
    weight2 = (double) 1.0 / (double) nReadsByQual[qual[iQual]];
    meanLengthByQual[qual[iQual]] = weight1 * meanLengthByQual[qual[iQual]] + weight2 * thisReadLen[iQual];

    // Update number of read with qLength exceeding each threshold
    for(uint16_t iLength=0;  iLength < readLength.size(); iLength++) {
      if(thisReadLen[iQual] >= readLength[iLength])
        nReadsByQualLength[qual[iQual]][readLength[iLength]] += 1;
    }
  }

  unsigned int trimmedLen = qScore.size();
  for(uint16_t iLength=0;  iLength < readLength.size(); iLength++) {
    if(trimmedLen >= readLength[iLength])
      nReadsByLength[readLength[iLength]] += 1;
  }

  if(keepPerReadData) {
    readName.push_back(rName);
    perReadQualLength.push_back(thisReadLen);
    perReadLength.push_back(trimmedLen);
  }
}


void SFFSummary::summarizeFromSffFile(string sffFile, vector <uint16_t> &_qual, vector <unsigned int> &_readLength, vector <unsigned int> &_minReadLength, bool _keepPerReadData) {
  qual            = _qual;
  readLength      = _readLength;
  minReadLength   = _minReadLength;
  keepPerReadData = _keepPerReadData;

  vector <uint16_t> qScore;

  // It would be more graceful to sort qual and minReadLength vectors, for now we just test & throw.
  for(unsigned int iQual=1; iQual < qual.size(); iQual++)
    if(qual[iQual] < qual[iQual-1])
      throw("quality thresholds should be specified in ascending order");

  // Convert quality threhsolds to error rate threhsolds
  phredToErr(qual,    errThreshold);

  summaryStatInit();
  sff_file_t *sff_file_in = sff_fopen(sffFile.c_str(), "rb", NULL, NULL);
  sff_t *sff = NULL;

  while(NULL != (sff = sff_read(sff_file_in))) {
    uint16_t clipped_start  = sff_clipped_read_left(sff)-1;
    uint16_t clipped_stop   = sff_clipped_read_right(sff);
    uint16_t clipped_length = clipped_stop-clipped_start;

    string rName = "";
    ion_string_t *tempName = sff->rheader->name;
    for(uint16_t iChar=0;  iChar < tempName->l; iChar++)
      rName += tempName->s[iChar];

    // Make a vector of qScores for the clipped portion
    ion_string_t *quality = sff->read->quality;
    assert(quality->l >= clipped_stop);
    qScore.resize(clipped_length);
    for(uint16_t iBase=clipped_start,qBase=0;  iBase < clipped_stop; iBase++,qBase++)
      qScore[qBase] = (uint16_t) quality->s[iBase];

    // Update the summary statistics
    summaryStatUpdate(qScore,rName);

    sff_destroy(sff);
  }
  sff_fclose(sff_file_in);

}

void SFFSummary::qualToErrLength(vector <uint16_t> &qScore, double errThreshold, unsigned int &readLen) {
  vector <double> vErrThreshold;
  vErrThreshold.push_back(errThreshold);
  vector <unsigned int> vReadLen;
  qualToErrLength(qScore, vErrThreshold, vReadLen);
  readLen = vReadLen[0];
}

void SFFSummary::qualToErrLength(vector <uint16_t> &qScore, vector <double> &errThreshold, vector <unsigned int> &readLen) {
  unsigned int nBases=qScore.size();
  unsigned int nErrThreshold=errThreshold.size();
  readLen.resize(nErrThreshold);
  readLen.assign(nErrThreshold,0);

  vector <double> cumulativeErrorRate;
  cumulativeErrorRate.resize(nBases);

  if(nBases > 0 && nErrThreshold > 0) {
    phredToErr(qScore[0],cumulativeErrorRate[0]);
    double errProbability;
    for(unsigned int iBase=1; iBase<nBases; iBase++) {
      phredToErr(qScore[iBase],errProbability);
      cumulativeErrorRate[iBase] = ((cumulativeErrorRate[iBase-1] * (double)(iBase)) + errProbability) / (double) (iBase+1);
    }

    for(unsigned int iBase=nBases-1,iThreshold=0; ; iBase--) {
      while( (iThreshold < nErrThreshold) && (cumulativeErrorRate[iBase] <= errThreshold[iThreshold]) ) {
        readLen[iThreshold] = iBase+1;
        iThreshold++;
      }
      if(iBase==0 || iThreshold==nErrThreshold)
        break;
    }
  }
}

void SFFSummary::phredToErr(vector <uint16_t> &qScore, vector <double> &errorRate) {
  unsigned int n = qScore.size();
  errorRate.resize(n);
  for(unsigned int i=0; i<n; i++)
    phredToErr(qScore[i],errorRate[i]);
}

void SFFSummary::phredToErr(uint16_t qScore, double &errorRate) {
  double exponent = ((double) qScore) / -10.0;
  errorRate = pow(10.0,exponent);
}

void SFFSummary::writeTSV(std::ostream &out) {
  for(unsigned int iQual=0; iQual < qual.size(); iQual++) {
    // Number of bases attaining the quality score
    out << "Number of Bases at Q" << qual[iQual] << " = " << nBasesByQual[qual[iQual]] << endl;
    out << "Number of Reads at Q" << qual[iQual] << " = " << nReadsByQual[qual[iQual]] << endl;
    // max and average lengths
    out << "Max Read Length at Q" << qual[iQual] << " = " << maxLengthByQual[qual[iQual]] << endl;
    out << "Mean Read Length at Q" << qual[iQual] << " = " << setiosflags(ios::fixed) << setprecision(1) << meanLengthByQual[qual[iQual]] << endl;
    // Number of reads with qLength exceeding each threshold
    for(uint16_t iLength=0;  iLength < readLength.size(); iLength++)
      out << "Number of " << readLength[iLength] << "BP Reads at Q" << qual[iQual] << " = " << nReadsByQualLength[qual[iQual]][readLength[iLength]] << endl;
  }
  // Number of trimmed reads exceeding each threshold
  for(uint16_t iLength=0;  iLength < readLength.size(); iLength++)
    out << "Number of " << readLength[iLength] << "BP Reads = " << nReadsByLength[readLength[iLength]] << endl;
}

void SFFSummary::writePrettyText(std::ostream &out) {
  unsigned int w1=15;
  unsigned int w2=12;

  
  // Header
  out << setw(w1) << left << "Statistic";
  for(unsigned int iQual=0; iQual < qual.size(); iQual++) {
    stringstream temp;
    temp << qual[iQual];
    string qString = "Q" + temp.str();
    out << setw(w2) << right << qString;
  }
  out << endl;

  // Number of bases attaining the quality score
  out << setw(w1) << left << "bases";
  for(unsigned int iQual=0; iQual < qual.size(); iQual++)
    out << setw(w2) << right << nBasesByQual[qual[iQual]];
  out << endl;

  // Number of reads attaining the quality score
  out << setw(w1) << left << "reads";
  for(unsigned int iQual=0; iQual < qual.size(); iQual++)
    out << setw(w2) << right << nReadsByQual[qual[iQual]];
  out << endl;

  // Max read length attaining the quality score
  out << setw(w1) << left << "maxLen";
  for(unsigned int iQual=0; iQual < qual.size(); iQual++)
    out << setw(w2) << right << maxLengthByQual[qual[iQual]];
  out << endl;

  // Mean read length attaining the quality score
  out << setw(w1) << left << "meanLen";
  for(unsigned int iQual=0; iQual < qual.size(); iQual++)
    out << setw(w2) << right << setiosflags(ios::fixed) << setprecision(1) << meanLengthByQual[qual[iQual]];
  out << endl;

  // Number of reads with qLength exceeding each threshold
  for(uint16_t iLength=0;  iLength < readLength.size(); iLength++) {
    stringstream temp;
    temp << readLength[iLength];
    string lString = temp.str() + "bp reads by qual";
    out << setw(w1) << left << lString;
    for(unsigned int iQual=0; iQual < qual.size(); iQual++)
      out << setw(w2) << right << nReadsByQualLength[qual[iQual]][readLength[iLength]];
    out << endl;
  }

  // Number of reads with trimmed length exceeding each threshold
  for(uint16_t iLength=0;  iLength < readLength.size(); iLength++) {
    stringstream temp;
    temp << readLength[iLength];
    string lString = temp.str() + "bp reads";
    out << setw(w1) << left << lString;
    out << setw(w2) << right << nReadsByLength[readLength[iLength]];
    out << endl;
  }
}

void SFFSummary::writePerReadData(std::ostream &out) {

  // Write Header
  out << "read\ttrimLen";
  if(reportPredictedQlen) {
    for(unsigned int iQual=0; iQual < qual.size(); iQual++)
      out << "\tQ" << qual[iQual] << "Len";
  }
  out << endl;

  // Write Data
  for(unsigned int iRead=0; iRead < readName.size(); iRead++) {
    out << readName[iRead];
    out << "\t" << perReadLength[iRead];
    if(reportPredictedQlen) {
      for(unsigned int iQual=0; iQual < qual.size(); iQual++)
        out << "\t" << perReadQualLength[iRead][iQual];
    }
    out << endl;
  }

}
