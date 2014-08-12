/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef CUDASTREAMTEMPLATE_H
#define CUDASTREAMTEMPLATE_H

#include "StreamManager.h"


class TemplateStream : public cudaSimpleStreamExecutionUnit
{
  //Execution config
  static int _bpb;   // beads per block
  static int _l1type; // 0:sm=l1, 1:sm>l1, 2:sm<l1


  //host memory


  // device memory


  // pointers pointing into scratch


public:

  TemplateStream(streamResources * res, WorkerInfoQueueItem item);
  ~TemplateStream();


  // Implementation of virtual interface functions from base class
  bool InitJob();
  void ExecuteJob(); // execute async functions
  int handleResults(); // non async results handling
  void printStatus();

  // config setter, might need more of those if more than one kernel

  int getBeadsPerBlock();
  int getL1Setting();

  // static member function needed for setup and pre allocation
  static void setBeadsPerBLock(int bpb);
  static void setL1Setting(int type); // 0:sm=l1, 1:sm>l1, 2:sm<l1
  static void printSettings();
  static size_t getMaxHostMem();
  static size_t getMaxDeviceMem();


private:

  void cleanUp();
  int l1DefaultSetting(); //runtime decision on best l1 setting depending on device

 //Implementation of actual stream execution
  void prepareInputs();  // not async stuff
  void copyToDevice();
  void executeKernel();
  void copyToHost();

  bool ValidJob();

};



#endif // CUDASTREAMTEMPLATE_H
