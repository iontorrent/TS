/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RAWWELLSWRITEJOB_H
#define RAWWELLSWRITEJOB_H

#include "PJob.h"
#include "SignalProcessingMasterFitter.h"


class RawWellsWriteJob : public PJob
{
  public:
    RawWellsWriteJob(SignalProcessingMasterFitter *fitter):PJob(), _fitter(fitter) 
    {}

    ~RawWellsWriteJob(){}

    void Run();
    void setCurFlow(int flow) { _curFlow = flow; }
    void setCurAmpBuffer(float *buf) { _ampBuffer = buf; }

  private: 
    int _curFlow;
    float *_ampBuffer;
    SignalProcessingMasterFitter *_fitter;  
};

void RawWellsWriteJob::Run()
{
  // All the metadata like bead x and y coord, region start row and col, and block width should be available from 
  // the way we create jobs for individual regions and the state mask for all the beads of the block should provide
  // filtering information or amplitude should be curated based on state flags before buffer is passed in to
  // write to raw wells
  if (_fitter) {
  
    int blockW = _fitter->GetGlobalStage().bfmask->W(); 
    int numLBeads = _fitter->region_data->my_beads.numLBeads;
    RawWells *wellsFileHandle = _fitter->GetGlobalStage().getRawWellsHandle();   
    for (int i=0; i<numLBeads; ++i) {
      int x = _fitter->region_data->my_beads.params_nn[i].x + _fitter->region_data->region->col;
      int y = _fitter->region_data->my_beads.params_nn[i].y + _fitter->region_data->region->row;

      float val = _ampBuffer[y*blockW + x];
      if (!wellsFileHandle->GetSaveAsUShort()) {
        val *= _fitter->region_data->my_beads.params_nn[i].Copies;
      }
      

      if ( _fitter->region_data->my_beads.params_nn[i].my_state->pinned ||
         _fitter->region_data->my_beads.params_nn[i].my_state->corrupt) {
        val = 0;  
      }

      if (wellsFileHandle->GetSaveCopies())
        wellsFileHandle->WriteFlowgram(_curFlow, x, y, val, _fitter->region_data->my_beads.params_nn[i].Copies);
      else 
        wellsFileHandle->WriteFlowgram(_curFlow, x, y, val);
    }
  }
}

#endif // RAWWELLSWRITEJOB_H
