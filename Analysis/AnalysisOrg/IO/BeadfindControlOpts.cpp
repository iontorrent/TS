/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BeadfindControlOpts.h"
#include "Utils.h"
void BeadfindControlOpts::DefaultBeadfindControl()
{
  //maxNumKeyFlows = 0;
  //minNumKeyFlows = 99;
  bfMinLiveRatio = .0001;
  bfMinLiveLibSnr = 4;
  bfMinLiveTfSnr = -1;
  bfTfFilterQuantile = 1;
  bfLibFilterQuantile = 1;
  skipBeadfindSdRecover = 1;
  beadfindThumbnail = 0;
  beadfindLagOneFilt = 0;
  beadMaskFile = NULL;
  maskFileCategorized = 0;
  sprintf (bfFileBase, "beadfind_post_0003.dat");
  sprintf (preRunbfFileBase, "beadfind_pre_0003.dat");
  BF_ADVANCED = true;
  SINGLEBF = true;
  noduds = 0;
  beadfindUseSepRef = 0;
  numThreads = -1;
  minTfPeakMax = 40.0f;
  minLibPeakMax = 15.0f;
  bfOutputDebug = false;
  bfMult = 1.0;
  sdAsBf = true;
  gainCorrection = true;
  useSignalReference = 1;
  useSignalReferenceSet = 0;
  blobFilter = false;
  if (isInternalServer()) {
    bfOutputDebug = 2;
  }
  beadfindType = "differential";
  predictFlowStart = -1;
  predictFlowEnd = -1;
}

BeadfindControlOpts::~BeadfindControlOpts()
{
  if (beadMaskFile)
    free (beadMaskFile);
}
