/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PROCESSIMAGETOWELL_H
#define PROCESSIMAGETOWELL_H

#include <string>
#include <vector>
#include "CommandLineOpts.h"
#include "Mask.h"
#include "RawWells.h"
#include "Region.h"
#include "SeqList.h"
#include "TrackProgress.h"
#include "SlicedPrequel.h"
#include "ImageSpecClass.h"
#include "OptBase.h"
#include "BkgFitterTracker.h"
#include "ComplexMask.h"


class WriteFlowDataClass;

class ImageToWells{

  //outside references
  OptArgs &opts;
  CommandLineOpts &inception_state;
  SeqListClass &my_keys;
  TrackProgress &my_progress;
  ImageSpecClass &my_image_spec;
  SlicedPrequel &my_prequel_setup;


  //internal members
  Region            WholeChip;
  BkgFitterTracker  GlobalFitter;
  ComplexMask       FromBeadfindMask;

  //dynamic members:
  ImageTracker * ptrMyImgSet;
  ChunkyWells * ptrRawWells;

  WriteFlowDataClass * ptrWriteFlowData;


public:

  ImageToWells(
      OptArgs &refOpts,
      CommandLineOpts &refInception_state,
      Json::Value &refJson_params,
      SeqListClass &refMy_keys,
      TrackProgress &refMy_progress,
      ImageSpecClass &refMy_image_spec,
      SlicedPrequel &refMy_prequel_setup);

  ~ImageToWells();

  void SetUpThreadedSignalProcessing(Json::Value &refJson_params);
  void DoThreadedSignalProcessing();
  void FinalizeThreadedSignalProcessing();


protected:

  //reads start from serialization file or performs isolated Beadfind if no file available
  void GetRestartData(Json::Value &json_params);
  void StoreRestartData();

  //Initializes bf mask and pinned state or reads it from file if restart

  void InitMask();
  void SetupDirectoriesAndFiles();


  void InitGlobalFitter( Json::Value &json_params);
  void CreateAndInitRawWells();
  void CreateAndInitImageTracker();
  void InitFlowDataWriter();

  void AllocateAndStartUpGlobalFitter();


  void ExecuteFlowBlockSignalProcessing();
  void ExecuteFlowByFlowSignalProcessing();

  void DoClonalFilter(int flow);


  void DestroyFlowDataWriter();



  void MoveWellsFileAndFinalize();




  //helpers

  const FlowBlockSequence & getFlowBlockSeq() { return inception_state.bkg_control.signal_chunks.flow_block_sequence; }
  //isRestart returns true if executed from beadfind or from later flow
  bool isRestart(){ return !(inception_state.bkg_control.signal_chunks.restart_from.empty());}
  bool doSerialize(){ return !(inception_state.bkg_control.signal_chunks.restart_next.empty());}
  bool useFlowDataWriter();
  bool isLastFlow(int flow){ return ( flow ) == ( inception_state.flow_context.GetNumFlows()- 1 ); }

  const std::string getWellsFile(){return (string(inception_state.sys_context.wellsFilePath) + "/" + inception_state.sys_context.wellsFileName);}



};



#endif // PROCESSIMAGETOWELL_H
