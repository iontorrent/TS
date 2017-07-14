/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * ClonalFilterWrapper.h
 *
 *  Created on: Oct 21, 2014
 *      Author: jakob siegel
 */

#ifndef CLONALFILTERWRAPPER_H_
#define CLONALFILTERWRAPPER_H_

#include <deque>
#include <vector>
#include "ClonalFilter/polyclonal_filter.h"
#include "LayoutTranslator.h"



class ClonalFilterWrapper
{
  deque<int>   col;
  deque<int>   row;
  deque<float> ppf;
  deque<float> ssq;
  deque<float> nrm;

  LayoutCubeWithRegions<unsigned short> * pBeadStateMask;
  LayoutCubeWithRegions<float> * pClonalFilterCube;
  unsigned short * pBfMask;

protected:
  void SaveH5(const char*  fname, const vector<int16_t>& row, const vector<int16_t>& col, const vector<float>&   ppf, const vector<float>&   ssq,  const vector<float>&   nrm);

public:

  ClonalFilterWrapper(unsigned short * bfMaks, LayoutCubeWithRegions<unsigned short> & BeadStateMask, LayoutCubeWithRegions<float> & ClonalFilterCube);
  void DumpPPFSSQ(const char * results_folder);
  void ApplyClonalFilter (  const PolyclonalFilterOpts & opts);
  void UpdateMask();

  void DumpPPFSSQtoH5(const char * results_folder);
};



#endif /* CLONALFILTERWRAPPER_H_ */
