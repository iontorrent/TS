/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ComplexMask.h"

ComplexMask::ComplexMask()
{
  my_mask = NULL;
  pinnedInFlow = NULL;
}

void ComplexMask::InitPinnedInFlow ( int numFlows )
{
  if ( pinnedInFlow == NULL ) {

    pinnedInFlow = new PinnedInFlow ( my_mask, numFlows );

    pinnedInFlow->Initialize ( my_mask );
  }
}

void ComplexMask::InitMask()
{
  my_mask = new Mask(1,1); // goofy that we have to create a dummy mask before we do anything real.
}

ComplexMask::~ComplexMask()
{
  if (my_mask!=NULL) delete my_mask;
  my_mask = NULL;
  
   if ( pinnedInFlow != NULL ) delete pinnedInFlow;
   pinnedInFlow=NULL;
}