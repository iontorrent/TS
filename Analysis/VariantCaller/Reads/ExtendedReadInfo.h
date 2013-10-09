/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     ExtendedReadInfo.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection


#ifndef EXTENDEDREADINFO_H
#define EXTENDEDREADINFO_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include "api/api_global.h"
#include "api/BamAux.h"
#include "api/BamConstants.h"
#include "api/BamReader.h"
#include "api/SamHeader.h"
#include "api/BamAlignment.h"
#include "api/SamReadGroup.h"
#include "api/SamReadGroupDictionary.h"
#include "api/SamSequence.h"
#include "api/SamSequenceDictionary.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <Variant.h>


#include "InputStructures.h"
#include "MiscUtil.h"


using namespace std;
using namespace BamTools;
using namespace ion;

class ExtendedReadInfo{
  public:
  BamTools::BamAlignment     alignment;         //!< BamTools Alignment Information
  bool                       is_forward_strand; //!< Indicates whether read is from the forward or reverse strand
  string                     read_bases;        //!< Read sequence as base called (minus hard but including soft clips)
  string                     ref_aln;           //!< Gap padded read sequence
  string                     seq_aln;           //!< Gap padded reference sequence
  string                     pretty_aln;        //!< pretty alignment string displaying matches, insertions, deletions
  vector<float>              measurementValue;  //!< The measurement values for this read
  vector<int>                flowIndex;         //!< Main Incorporating flow for each base
  vector<float>              phase_params;      //!< cf, ie, droop parameters of this read
  int                        leftSC;            //!< Number of soft clipped bases at the start of the alignment
  int                        rightSC;           //!< Number of soft clipped bases at the end of the alignment
  int                        start_flow;        //!< Flow corresponding to the first base in read_seq
  int                        start_pos;         //!< Start position of the alignment as reported in BAM
  bool                       is_happy_read;
  string                     runid;             //!< Identify the run from which this read came: used to find run-specific parameters
  vector<int>                well_rowcol;       //!< 2 element int vector 0-based row, col in that order mapping to row,col in chip
  unsigned short             map_quality;        //! MapQuality as reported in BAM

  ExtendedReadInfo(){
    Default();
  };

  ExtendedReadInfo(int nFlows){
    Default();
    measurementValue.resize(nFlows,0);
    ref_aln.reserve(nFlows);
    seq_aln.reserve(nFlows);
    pretty_aln.reserve(nFlows);
  };

  void Default(){
    phase_params.resize(3,0);
    leftSC = rightSC = 0;
    is_forward_strand = true;
    start_flow = 0;
    start_pos = 0;
    is_happy_read = false;
    map_quality = 0;
  };

  //! @brief  Populates object variables
  bool UnpackThisRead(const InputStructures &global_context, const string &local_contig_sequence, int DEBUG);

  //! @ brief  Loading BAM tags into internal variables
  void GetUsefulTags(int DEBUG);

  //! @brief  Sets member variables containing alignment information
  //! @brief [in]  local_contig_sequence    reference sequence
  //! @brief [in]  aln_start_position       start position of the alignment
  bool UnpackAlignmentInfo(const string &local_contig_sequence, unsigned int aln_start_position);

  //! @brief  Populates object members flowIndex and read_seq
  bool CreateFlowIndex(const string &flowOrder);

  //! @brief  Increases the start flow to match the flow of the first aligned base
  void IncreaseStartFlow();

  unsigned int GetStartSC();

  unsigned int GetEndSC();

  //! @brief Has this read been nicely unpacked and is it useful?
  // Moved all functionality into StackPlus object
  //bool CheckHappyRead(InputStructures &global_context, int variant_start_pos, int DEBUG);
};

#endif //EXTENDEDREADINFO_H
