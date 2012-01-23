/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef STANDALONE_H
#define STANDALONE_H

#include <stdlib.h>


class stand_alone_buffer{
public:
    // buffers for stand-alone API operation
    float   *fg;
    float   *bg;
    float   *feval;
    float   *isig;
    float   *pf;
    
    stand_alone_buffer(){
    fg    = NULL;
    bg    = NULL;
    feval = NULL;
    isig  = NULL;
    pf    = NULL;
  };
  void allocate(int bead_flow_t){
    if (fg    == NULL) fg    = new float[bead_flow_t];
    if (bg    == NULL) bg    = new float[bead_flow_t];
    if (feval == NULL) feval = new float[bead_flow_t];
    if (isig  == NULL) isig  = new float[bead_flow_t];
    if (pf    == NULL) pf    = new float[bead_flow_t];
  };
  void Delete(void){
    if (fg    == NULL) delete[] fg;
    if (bg    == NULL) delete[] bg;
    if (feval == NULL) delete[] feval;
    if (isig  == NULL) delete[] isig;
    if (pf    == NULL) delete[] pf;
};
  ~stand_alone_buffer(){Delete();};
};


#endif // STANDALONE_H