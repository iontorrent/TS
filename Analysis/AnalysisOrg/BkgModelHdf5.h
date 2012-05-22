/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BKGMODELHDF5_H
#define BKGMODELHDF5_H

#include <vector>
#include "CommandLineOpts.h"
#include "ImageSpecClass.h"
#include "DataCube.h"
#include "H5File.h"
#include "BkgFitterTracker.h"
#include "BkgDataPointers.h"


class BkgParamH5 {
 public:
  BkgParamH5();
  ~BkgParamH5() { Close(); }

  void DumpBkgModelRegionInfoH5(BkgFitterTracker &GlobalFitter, int flow, bool last_flow);
  void DumpBkgModelBeadFblkInfoH5(BkgFitterTracker &GlobalFitter, int flow, bool last_flow);
  void IncrementalWriteParam(DataCube<float> &cube, H5DataSet *set, int flow, int saveWellsFrequency,int numFlows);
  string Init(SystemContext &sys_context,SpatialContext &loc_context, int numFlows, ImageSpecClass &my_image_spec);
  void IncrementalWrite(BkgFitterTracker &GlobalFitter, int flow, int numFlows, bool last_flow);
  void Close();


public: // should be private eventually
  // space to store the data w/ WriteBeadParameterstoDataCubes
  // pointers need to be saved & passed to WriteBeadParameterstoDataCubes
  DataCube<float> amplMultiplier; // variables A1-Ann in BkgModelBeadData.nnnn.txt
  DataCube<float> beadDC; // variables A1-Ann in BkgModelBeadData.nnnn.txt
  DataCube<float> kRateMultiplier; // variables M1-Mnn in BkgModelBeadData.nnnn.txt
  DataCube<float> bgResidualError;
  DataCube<float> beadInitParam;
  DataCube<float> emptyOnceParam;
  DataCube<float> emphasisParam;
  DataCube<float> darkOnceParam;
  arma::Mat<float> darknessParam;
  arma::Mat<float> regionInitParam;
  arma::Mat<float> beadFblk_avgErr;
  arma::Mat<int> beadFblk_clonal;
  arma::Mat<int> beadFblk_corrupt;

  //regionalParams & regionalParamsExtra don't need to be sotred, because their values are already in the BkgModel
  // or could be calculated directly
  arma::Mat<float> regionalParams;
  arma::Mat<float> regionalParamsExtra;
  arma::Mat<int> regionOffset;
  arma::Mat<float> beadDC_bg; // bg_beadDC

  // pointers to the dataset (DS) in the .h5 file
  // need to call ptr->WriteDataCube() to write the above DataCubes to the .h5 file
  H5DataSet *amplDS;
  H5DataSet *beadDCDS;
  H5DataSet *beadDC_bgDS;
  H5DataSet *kRateDS;
  H5DataSet *resErrorDS;
  H5DataSet *beadInitParamDS;
  H5DataSet *darkOnceParamDS;
  H5DataSet *darknessParamDS;
  H5DataSet *emptyOnceParamDS;
  H5DataSet *regionInitParamDS;
  H5DataSet *regionOffsetDS;

  vector<H5DataSet * > regionParamDS;
  vector<H5DataSet * > regionParamDSExtra;
  vector<H5DataSet * > emphasisParamDS;

  vector<H5DataSet * > beadFblk_avgErrDS;
  vector<H5DataSet * > beadFblk_clonalDS;
  vector<H5DataSet * > beadFblk_corruptDS;

  BkgDataPointers ptrs;

  void close_Vector(vector<H5DataSet * > &vec);
  void savePointers();

  std::string getFilename() {return hgBgDbgFile;}

private:
  H5File h5BgDbg;
  std::string hgBgDbgFile;

};

#endif // BKGMODELHDF5_H
