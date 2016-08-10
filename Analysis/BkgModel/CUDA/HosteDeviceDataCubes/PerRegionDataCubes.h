/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * PerRegionDataCubes.h
 *
 *  Created on: Sep 22, 2015
 *      Author: Jakob Siegel
 */

#ifndef PERREGIONDATACUBES_H_
#define PERREGIONDATACUBES_H_


#include "LayoutCubeRegionsTemplate.h"
#include "DeviceParamDefines.h"
#include "ImgRegParams.h"

struct BkgModelWorkInfo;

class RegionalParamCubeClass : public LayoutCubeWithRegions<PerFlowParamsRegion>
{

protected:
  void translateHostToCube(BkgModelWorkInfo* pBase_bkinfo);
  void translateCubeToHost(BkgModelWorkInfo* pBase_bkinfo);

public:
  RegionalParamCubeClass(ImgRegParams iP, MemoryType mtype, vector<size_t>& sizeBytes  ):LayoutCubeWithRegions<PerFlowParamsRegion>(iP.getGridParam(),1, mtype, sizeBytes){};
  RegionalParamCubeClass(ImgRegParams iP, MemoryType mtype ):LayoutCubeWithRegions<PerFlowParamsRegion>(iP, Bp_NUM_PARAMS, mtype){};
  RegionalParamCubeClass(const RegionalParamCubeClass & that, MemoryType mtype ):LayoutCubeWithRegions<PerFlowParamsRegion>(that, mtype){};
  virtual ~RegionalParamCubeClass(){};

  void init(BkgModelWorkInfo* pBase_bkinfo);
  void reinjectHostStructures(BkgModelWorkInfo* pBase_bkinfo);

};



#endif /* PERREGIONDATACUBES_H_ */

