/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADFINDCONTROLOPTS_H
#define BEADFINDCONTROLOPTS_H
#include <vector>
#include <string>
#include <map>
#include <set>
#include "Region.h"
#include "IonVersion.h"
#include "Utils.h"
#include "OptBase.h"

class BeadfindControlOpts{
  public:
    double bfMinLiveRatio;
    double bfMinLiveLibSnr;
    double bfMinLiveTfSnr;
    double bfTfFilterQuantile;
    double bfLibFilterQuantile;
    int skipBeadfindSdRecover;
    int beadfindThumbnail; // Is this a thumbnail chip where we need to skip smoothing across regions?
    bool beadfindSmoothTrace;
    std::string filterNoisyCols;
    char *beadMaskFile;
    std::string exclusionMaskFile;
    bool maskFileCategorized;
    char bfFileBase[MAX_PATH_LENGTH];
    char preRunbfFileBase[MAX_PATH_LENGTH];
    bool noduds;
    bool beadfindUseSepRef; // should we just use the reference wells the separator uses or go for expanded set?
    int bfOutputDebug;
    float bfMult;
    bool sdAsBf;
    bool useBeadfindGainCorrection;
    bool useDatacollectGainCorrectionFile;
    int useSignalReference;
    bool useSignalReferenceSet;
    std::string beadfindType;
    std::string bfType; // signal or buffer
    std::string bfDat;
    std::string bfBgDat;
    bool SINGLEBF;
    bool BF_ADVANCED;
    int numThreads;
    float minTfPeakMax;
    float minLibPeakMax;
    bool blobFilter;
    std::string doubleTapFlows;
    int predictFlowStart;
    int predictFlowEnd;
    int meshStepX;
    int meshStepY;
    std::vector<float> beadfindAcqThreshold;
    std::vector<float> beadfindBfThreshold;
    void DefaultBeadfindControl();
    void PrintHelp();
    void SetOpts(OptArgs &opts, Json::Value& json_params);
    void SetThumbnail(bool tn);
    ~BeadfindControlOpts();
};


#endif // BEADFINDCONTROLOPTS_H
