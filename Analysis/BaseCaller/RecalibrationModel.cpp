/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     RecalibrationModel.cpp
//! @ingroup  BaseCaller
//! @brief    RecalibrationModel. Model estimation between simulated predictions and observed measurements

#include "RecalibrationModel.h"

#include <string>
#include <fstream>
#include <stdio.h>
#include <SystemMagicDefines.h>

int NuctoInt(char nuc) {
  switch(nuc) {
  case 'A': return 0;
  case 'C': return 1;
  case 'G': return 2;
  default: return 3;
  }
}

RecalibrationModel::RecalibrationModel():stratification_(NULL)
{
  is_enabled_ = false;
  max_hp_calibrated_ = 0;
}


RecalibrationModel::~RecalibrationModel()
{
  if(stratification_!=NULL) delete stratification_;
}


void RecalibrationModel::Initialize(OptArgs& opts)
{
  is_enabled_ = false;

  string model_file_name = opts.GetFirstString ('s', "model-file", "");
  if(model_file_name.empty()) {
    printf("RecalibrationModel: disabled\n\n");
    return;
  }

  ifstream model_file;
  model_file.open(model_file_name.c_str());
  if (model_file.fail()) {
    printf("RecalibrationModel: disabled (cannot open %s)\n\n", model_file_name.c_str());
    model_file.close();
    return;
  }

  int recalModelHPThres = opts.GetFirstInt('-', "recal-model-hp-thres", 4);

  string comment_line;
  getline(model_file, comment_line); //skip the comment time

  int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, max_hp_calibrated;
  model_file >> flowStart >> flowEnd >> flowSpan >> xMin >> xMax >> xSpan >> yMin >> yMax >> ySpan >>  max_hp_calibrated;
  stratification_ = new RegionStratification(xMin, xMax, xSpan, yMin, yMax, ySpan);
  //calculate number of partitions and initialize the stratifiedAs and stratifiedBs
  const int numRegionStratifications = stratification_->xCuts * stratification_->yCuts;
  const int numFlows = flowEnd - flowStart + 1;
  const int numHPs = MAX_HPXLEN + 1; //max_hp_calibrated + 1;
  const int numNucs = 4;
  stratifiedAs.resize(numRegionStratifications);
  stratifiedBs.resize(numRegionStratifications);
  for(int ind = 0; ind < numRegionStratifications; ++ind){
    stratifiedAs[ind].resize(numFlows);
    stratifiedBs[ind].resize(numFlows);
    for(int flowInd = 0; flowInd < numFlows; flowInd++){
      stratifiedAs[ind][flowInd].resize(numNucs);
      stratifiedBs[ind][flowInd].resize(numNucs);
      for(int nucInd = 0; nucInd < numNucs; ++nucInd){
        stratifiedAs[ind][flowInd][nucInd].assign(numHPs, 1.0);
        stratifiedBs[ind][flowInd][nucInd].assign(numHPs, 0.0);
      }
    }
  }

  //TODO: parse model_file into stratifiedAs and stratifiedBs
  float paramA, paramB;
  int refHP;
  char flowBase;
  while(model_file.good()){
    model_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> refHP >> paramA >> paramB;
    //populate it to stratifiedAs and startifiedBs
    int nucInd = NuctoInt(flowBase);
    int offsetRegion = (yMin - stratification_->yMin)/stratification_->ySpan + (xMin - stratification_->xMin)/stratification_->xSpan * stratification_->yCuts;
    //boundary check
    for(int flowInd = flowStart; flowInd < flowEnd; ++flowInd){
      if(refHP < recalModelHPThres -1) continue;
      stratifiedAs[offsetRegion][flowInd][nucInd][refHP] = paramA;
      stratifiedBs[offsetRegion][flowInd][nucInd][refHP] = paramB;
    }

  }

  model_file.close();

  printf("Recalibration: enabled (using calibration file %s)\n\n", model_file_name.c_str());
  is_enabled_ = true;
  if(recalModelHPThres > MAX_HPXLEN) is_enabled_ = false;
}

vector<vector<vector<float> > > * RecalibrationModel::getAs(int x, int y)
{
  if(!is_enabled_){
    return 0;
  }
  int offsetRegion = (y - stratification_->yMin)/stratification_->ySpan + (x - stratification_->xMin)/stratification_->xSpan * stratification_->yCuts;
  //dimension checking
  if(offsetRegion < 0 || offsetRegion >= (int)stratifiedAs.size())
    return 0;
  else
    return &(stratifiedAs[offsetRegion]);
}

 vector<vector<vector<float> > > * RecalibrationModel::getBs(int x, int y)
{
  if(!is_enabled_){
    return 0;
  }
  int offsetRegion = (y - stratification_->yMin)/stratification_->ySpan + (x - stratification_->xMin)/stratification_->xSpan * stratification_->yCuts;
  if(offsetRegion < 0 || offsetRegion >= (int)stratifiedBs.size())
    return 0;
  else
    return &(stratifiedBs[offsetRegion]);
}










