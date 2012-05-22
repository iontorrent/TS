/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BASECALLERCONTROLOPTS_H
#define BASECALLERCONTROLOPTS_H

#include <vector>
#include <string>
#include <map>
#include <set>
#include "Region.h"
#include "IonVersion.h"
#include "Utils.h"


class CafieControlOpts{
  public:
    int singleCoreCafie;
    double LibcfOverride;
    double LibieOverride;
    double LibdrOverride;
    std::string libPhaseEstimator;
    std::string basecaller;
    int cfiedrRegionsX, cfiedrRegionsY;
    int cfiedrRegionSizeX, cfiedrRegionSizeY;
    int blockSizeX, blockSizeY;
    int numCafieSolveFlows;

    int doCafieResidual;
    char *basecallSubsetFile;
    std::set< std::pair <unsigned short,unsigned short> > basecallSubset;

    // should this be in system context?
    std::string phredTableFile;

    void DefaultCAFIEControl();
    void EchoDerivedChipParams(int chip_len_x, int chip_len_y);
    ~CafieControlOpts();
};

void readBasecallSubsetFile (char *basecallSubsetFile, std::set< std::pair <unsigned short,unsigned short> > &basecallSubset);


#endif // BASECALLERCONTROLOPTS_H