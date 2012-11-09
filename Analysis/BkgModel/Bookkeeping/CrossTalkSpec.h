/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CROSSTALKSPEC_H
#define CROSSTALKSPEC_H

#include <vector>
#include "Serialization.h"

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
    std::vector<int> cx;
    std::vector<int> cy;

    // how strong the influence of this neighbor is
    std::vector<float> mix;

    // how much to delay the nuc rise
    std::vector<float> delay;

    // Earl's reformulated parameters
    std::vector<float> multiplier;
    std::vector<float> tau_top;
    std::vector<float> tau_fluid;

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
    bool simple_model;
    bool rescale_flag;


    void Allocate ( int size );
    void SetStandardGrid();
    void SetNewQuadGrid();
    void SetHexGrid();
    void SetAggressiveHexGrid();
    void SetNewHexGrid();
    float ClawBackBuffering ( int nei_total );
    void ReadCrossTalkFromFile ( const char * );
    // different chips unfortunately have different layouts
    void NeighborByGridPhase ( int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase );
    void NeighborByGridPhaseBB ( int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase );
    void NeighborByChipType ( int &ncx, int &ncy, int bead_rx, int bead_ry, int nei_idx, int region_x, int region_y );

    void BootUpXtalkSpec ( bool can_do_xtalk_correction, const char *chipType, const char *xtalk_name );
    // missing functions
    // should be able to read in this effect from a file
    // should be able to assign this by region to vary it across the chip

    CrossTalkSpecification();

  private:
    // Boost serialization support:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize ( Archive& ar, const unsigned int version )
    {
      fprintf ( stdout, "Serialize CrossTalkSpec ... " );
      ar
      & nei_affected
      & cx
      & cy
      & mix
      & delay
      & multiplier
      & tau_top
      & tau_fluid
      & tau_bulk
      & cbulk
      & nscale
      & hex_packed
      & three_series
      & do_xtalk_correction
      & simple_model
      & rescale_flag;
      fprintf ( stdout, "done\n" );
    }
};


#endif // CROSSTALKSPEC_H

