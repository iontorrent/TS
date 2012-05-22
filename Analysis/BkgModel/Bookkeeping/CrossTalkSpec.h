/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CROSSTALKSPEC_H
#define CROSSTALKSPEC_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <vector>
#include <math.h>

// this specifies the effect of nearest neighbors on the well known as "crosstalk"
// we have a global background trace for a region
// this specifies the ways the neighbors contribute to a "crosstalk" trace added to the global trace
// neighbors affect the cross-talk trace by
// a) providing new hydrogen ions at a rate determined by the properties & amplitude of the neighbor
// b) buffering the hydrogen ions in the box above the well in some means
// c) the residence time/buffering of the bulk determines how fast we contribute to this measure
// This would be easier if we had an "inferred bulk" trace already instead of an "average case empty" 

class CrossTalkSpecification
{
  public:
    
    int nei_affected;
    // coordinates of neighbor being summed
    int *cx;
    int *cy;
    // how strong the influence of this neighbor is
    float *mix;
    // how much to delay the nuc rise
    float *delay;
    
    // Earl's reformulated parameters
    float *multiplier;
    float *tau_top;
    float *tau_fluid;
    
    // resistance to change in the 'box' above the well
    // includes rate of flow
    // and conductance/buffering of bulk fluid
    float tau_bulk;
    // global scaling for the strength of neighbor effect
    float cbulk;
    // reduction/increase for number of neighbors
    // this is oddly constructed to move in the opposite direction
    // so it should be re-examined from scratch
    float nscale;
    // are we hex_packed?
    bool hex_packed;
    bool three_series; // are we one clas of chips
    //do we do this at all?
    bool do_xtalk_correction;
    
    
    void Allocate(int size);
    void DeAllocate();
    void SetStandardGrid();
    void SetNewQuadGrid();
    void SetHexGrid();
    void SetAggressiveHexGrid();
    void SetNewHexGrid();
    float ClawBackBuffering(int nei_total);
    void ReadCrossTalkFromFile(char *);
    // different chips unfortunately have different layouts
    void NeighborByGridPhase(int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase);
    void NeighborByGridPhaseBB(int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase);
    void NeighborByChipType(int &ncx, int &ncy, int bead_rx, int bead_ry, int nei_idx, int region_x, int region_y);
    
    void BootUpXtalkSpec(bool can_do_xtalk_correction, char *chipType, char *xtalk_name);
    // missing functions
    // should be able to read in this effect from a file
    // should be able to assign this by region to vary it across the chip
    
    CrossTalkSpecification();
    ~CrossTalkSpecification();
};


#endif // CROSSTALKSPEC_H
