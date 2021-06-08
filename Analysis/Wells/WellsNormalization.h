/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

// Dummy class to insert and test wells normalization procedures

#ifndef WELLSNORMALIZATION_H
#define WELLSNORMALIZATION_H

#include <string>
#include <cmath>
#include <vector>
#include <ctime>

#include "RawWells.h"
#include "ReadClassMap.h"
#include "BaseCallerUtils.h"
#include "DPTreephaser.h"
#include "IonErr.h"

enum nuc {T,A,C,G};


// mask class of storing which wells
class ProcessingMask{
public:
	ProcessingMask(const unsigned int numRows, const unsigned int numCols);
	int SetMaskSize(const unsigned int numRows, const unsigned int numCols);
	bool Get(const unsigned int row, const unsigned int col) const;
	int Set(const unsigned int row, const unsigned int col, const bool value);
	unsigned int NumRows() const;
	unsigned int NumCols() const;
protected:
	vector< vector <bool> > mask_;
	unsigned int numRows_;
	unsigned int numCols_;

};


// data and functions for classifying 0 and 1
class Mark01Data{
public:
	Mark01Data(const size_t rowStart, const size_t numRows, const size_t colStart, const size_t numCols, const size_t flowStart, const size_t numFlows, RawWells const *wells, ReadClassMap const *rcm);
	int Classify(const ProcessingMask *mask, const unsigned int numMem, const unsigned int goodFlowStart, const unsigned int goodFlowEnd);

	// get flow classification (0,1,2)
	int& category(const size_t row, const size_t col, const size_t flow);
	// get residual from expected
	float& residual(const size_t row, const size_t col);
	// regional mean signal for a category.  (0 or 1)
	float MeanSignal(const int cat, const size_t flow, const ProcessingMask *mask);
	// regional fraction of 0 or 1-mers
	float Fraction(const int cat, const size_t flow, const ProcessingMask *mask);

	// initialize object
	void Initialize(const size_t rowStart, const size_t numRows, const size_t colStart, const size_t numCols, const size_t flowStart, const size_t numFlows, RawWells const *wells, ReadClassMap const *rcm);

protected:
	  vector<vector< vector<int> > > m01_;  		  // flow classified as 0, 1 and 2 (2-mer or higher)
	  vector<vector<float> > m01_res_;  	  // residual 1 per well
	  size_t rowStart_;
	  size_t numRows_;
	  size_t colStart_;
	  size_t numCols_;
	  size_t flowStart_;
	  size_t numFlows_;
	  RawWells const        *wells_;                  //!< Wells file reader
	  ReadClassMap const    *rcm_;                    //!< Beadfind and filtering outcomes for wells
	  unsigned int numWellsPassFilter;


};



//! @brief    Class to implement various wells file normalization methods
//!           Modifies wells file content after loading it from file
//!           and before any other BaseCaller method processes it
//! @ingroup  BaseCaller

class WellsNormalization
{
public:

  WellsNormalization();

  //! @brief   Constructor
  //! @param[in]   flow_order    Pointer to flow order object for this run
  //! @param[in]   norm_method   String specifying the normalization method to be used
  WellsNormalization(ion::FlowOrder const  *flow_order,
                     const string &norm_method);

  ~WellsNormalization();

  void SetFlowOrder(ion::FlowOrder const  *flow_order,
                    const string &norm_method);

  //! @brief   Set pointers to Wells and mask data for a particular thread
  //! @param[in]   wells         Pointer to RawWells object for a particular thread
  //! @param[in]   mask          Pointer to Mask object for a particular thread
  bool  SetWells(RawWells *wells, ReadClassMap const *rcm, unsigned int wells_file_index);

  //! @brief Key normalizes the values in the current wells file chunk
  //! @param[in]   keys          Vector of key sequence objects
  void  DoKeyNormalization(const vector<KeySequence> & keys);


  //! @brief Correct signal bias in 1.wells file.
  //! @param[in]    keys          Vector of key sequence objects
  void CorrectSignalBias(const vector<KeySequence> & keys);





protected:

  //! @brief   Determine if a read  is marked is filtered in mask
  //! @param[in]   x,y           x,y coordinates of read on chip
  bool  is_filtered(int x, int y) const;

  bool  is_filtered_libOnly(int x, int y) const;

  //! @brief update mask of all unfiltered wells
  //! @param[in]
  unsigned int UpdateMaskAll();


  void CorrectFlowOffset(const vector<float> sig0, const vector<float> fract0, const vector<float> sig1, const vector<float> fract1, const unsigned int winEachSide, const unsigned int startFlow, const ProcessingMask* pmask, string method);

  void CorrectNucOffset(const vector<float> sig0, const vector<float> fract0, unsigned int nucOffsetStartFlow, unsigned int nucOffsetEndFlow, ProcessingMask * pmask);

  void BalanceNucStrength(unsigned int nucStrengthStartFlow, unsigned int nucStrengthEndFlow);

  //! @brief update mask of good wells
  //! @param[in]
  unsigned int UpdateMaskGood(const double goodResidualThreshold, const bool useResidualMeanAsThreshold);


  double SubtractInvdividualWellZero(const size_t flowZeroStart, const size_t flowZeroStop);

  //! @brief   Update 01 classification data m01data_
  unsigned int Update01(const unsigned int numMem, const unsigned int goodFlowStart, const unsigned int goodFlowEnd);

  //! @brief   Find regional 0 and 1-mer signals, and fractions of 0-mer and 1-mer per flow based on classification stored in m01data_
  void Find01(vector<float>& sig0, vector<float>& sig1, vector<float>& fract0, vector<float>& fract1);

  //! Use maskAll. Intended to be used in CorrectSignalBias.
  //! @param[in]   keys          Vector of key sequence objects
  void NormalizeKeySignal(const vector<KeySequence> & keys, const ProcessingMask * mask);

  //! @brief   delete mark01 data
  void DeleteMark01Data();


  string                 norm_method_;            //!< Switch for normalization method
  bool                   is_disabled_;            //!< Switch to tell module to not do anything
  bool 					 doKeyNorm_;
  bool                   doSignalBiasCorr_;
  ion::FlowOrder const  *flow_order_;             //!< Flow order object
  RawWells              *wells_;                  //!< Wells file reader
  ReadClassMap const    *rcm_;                    //!< Beadfind and filtering outcomes for wells
  unsigned int           wells_index_;
  Mark01Data            *m01data_;                //!< classification for 0 and 1
  ProcessingMask        *maskAll_;                //!< mask of all wells needs to be normalized.
  ProcessingMask        *maskGood_;// mask of good wells

  vector<float> sig0Target_, sig1Target_;					// target 0-mer signal
  vector<float> offsetFactor_, scaleFactor_;	// offset subtracted from each flow
  bool DEBUG_;									// debugging flag
  double offsetT_, offsetA_, offsetC_, offsetG_, offsetAvg_;
  unsigned int nucOffsetStartFlowUsed_, nucOffsetEndFlowUsed_;
  double nucStrengthT_, nucStrengthA_, nucStrengthC_, nucStrengthG_;
  // SubtractIndividualWellZero debugging info
  unsigned int zeroFlowStartUsed_, zeroFlowStopUsed_, numWellsSubtractedZero_;
  double avgZeroSubtracted_, minZeroSubtracted_, maxZeroSubtracted_;
  unsigned int numPassFilter_, numGood_;
  bool pinZero_; // pin median zero-mer to zero in flow offset correction.
  string flowCorrectMethod_;


};


#endif // WELLSNORMALIZATION_H
