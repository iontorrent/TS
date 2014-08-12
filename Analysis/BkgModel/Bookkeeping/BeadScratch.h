/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BEADSCRATCH_H
#define BEADSCRATCH_H


#include "BkgMagicDefines.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "FlowBuffer.h"
#include "BeadParams.h"
#include "DiffEqModel.h"
#include "TimeCompression.h"
#include "MathOptim.h"
#include "RegionTracker.h"
#include "BkgTrace.h"
#include "EmptyTrace.h"


// hold temporary values for a bead
class BeadScratchSpace{
    // No copying, please.
    BeadScratchSpace( const BeadScratchSpace & );
    BeadScratchSpace & operator=( const BeadScratchSpace & );
  public:
    // temporary storage for fitting algorithm used to hold
    // functional evaluation and partial derivative terms
    float   *scratchSpace;
    // pointer into the scratch location where the function evaluation is held
    float   *fval;
    // separate cache for incorporation trace
    float   *ival;
    // xtalk is held in this separate cache
    float   *cur_xtflux_block;
    // emphasis vector for all flows for this bead
    float   *custom_emphasis;
    float   *custom_emphasis_scale;
    int *WhichEmphasis; // current emphasis indexing used in creating custom_emphasis
    // hold current traces for bead with whatever correction has been done
    float *observed;
    float *shifted_bkg;
    int cur_shift;


    // some buffer sizes for mnemonic purposes
    int   bead_flow_t;
    int   npts; // number of time points from time-compression
    
    BeadScratchSpace();
    ~BeadScratchSpace();

    void  Allocate(int npts,int num_derivatives, int flow_block_size);
    void  ResetXtalkToZero() const;
    void  FillEmphasis(int *my_emphasis, float *source_emphasis[], 
        const std::vector<float>& source_emphasis_scale, int flow_block_size);
    void  SetEmphasis(float *Ampl, int max_emphasis, int flow_block_size);
    void  CreateEmphasis(float *source_emphasis[], const std::vector<float>& source_emphasis_scale,
                         int flow_block_size );
    void  FillObserved(const BkgTrace &my_trace,int ibd, int flow_block_size) const;
    void  FillShiftedBkg(const EmptyTrace &emptytrace, float tshift, 
                         const TimeCompression &time_c, bool force_fill, int flow_block_size);
    float CalculateFitError(float *per_flow_output, int flow_block_size );
    void  MultiFlowReturnResiduals(float *y_minus_f);
    void  MultiFlowReturnFval(float *out, int numfb);

 private:
    int num_derivatives;

    // Serialization section
    friend class boost::serialization::access;
    template<typename Archive>
      void save(Archive& ar, const unsigned version) const {
      //      assert(custom_emphasis_scale != NULL);
      //      assert(WhichEmphasis != NULL);
      ar &
	      bead_flow_t &
	      npts &
	      num_derivatives;

      //      assert(npts == 0 || (custom_emphasis_scale != NULL && WhichEmphasis != NULL));
      int start = -1;
      if (npts > 0 && custom_emphasis_scale != NULL && WhichEmphasis != NULL) {
        start = bead_flow_t / npts - 1;
      }
      for( int i = start ; i >= 0 ; --i ) {
        ar & custom_emphasis_scale[i]
           & WhichEmphasis[i];
      }
    }

    template<typename Archive>
      void load(Archive& ar, const unsigned version) {
      ar & 
	      bead_flow_t &
	      npts &
	      num_derivatives;
      if (npts > 0) {
        Allocate( npts, num_derivatives, bead_flow_t / npts);
        for( int i = bead_flow_t / npts - 1 ; i >= 0 ; --i ) {
          ar & custom_emphasis_scale[i]
            & WhichEmphasis[i];
        }
      }
      else {
        if (custom_emphasis_scale != NULL) {
          delete [] custom_emphasis_scale;
          custom_emphasis_scale = NULL;
        }
        if (WhichEmphasis != NULL) {
          delete [] WhichEmphasis;
          WhichEmphasis = NULL;
        }
      }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};



#endif // BEADSCRATCH_H
