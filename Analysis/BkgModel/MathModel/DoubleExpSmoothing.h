/* Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DOUBLEEXPSMOOTHING_H
#define DOUBLEEXPSMOOTHING_H

#include "RegionParams.h"

class DoubleExpSmoothing {
  // Alpha and gamma values for double exponential smoothing.
  // See http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc433.htm for details.
  //
  // A pair of controlling constants.
  float alpha;
  float gamma;

  // Previous values for history.
  float prevB;
  float prevS;

  // The first time someone tries to use us for real, we'll need to initialize B.
  bool prevBInitialized;

  // The function that we call to gain access to a reg_params parameter.
  float * (reg_params::*access_fn)();

  // moving average
  int flowCount;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & alpha
        & gamma
        & prevB
        & prevS
        & prevBInitialized;
  }
public:

  // This is primarily for serialization (sigh)
  void SetAccessFn( float * (reg_params::*_access_fn)() )
  {
    access_fn = _access_fn;
  }

  // This is also for serialization.
  DoubleExpSmoothing()
  {
    alpha = gamma = prevB = prevS = 0.f;
    access_fn = 0;
    prevBInitialized = false;
    flowCount = 1;
  }

  // When we first create one of these, we should know some basic parameters.
  DoubleExpSmoothing( float _alpha, float _gamma, float * (reg_params::*_access_fn)() )
  {
    alpha = _alpha;
    gamma = _gamma;
    access_fn = _access_fn;
    prevBInitialized = false;
    flowCount = 1;
  }

  // The first time we use one of these, we'll have to initialize it.
  void Initialize( reg_params * rp )
  {
    prevS = *(rp->*access_fn)();
    prevBInitialized = false;
  }

  // When we use this smoother, we update our previous values.
  void Smooth( reg_params *rp )
  {
    // An extra step to initialize things.
    if ( ! prevBInitialized )
    {
      // This routine is here because, theoretically, one can initialize B with a slope factor.
      // We tried that... it didn't really work well for our case. 
      // I leave the one-liner here, in case someone wants to try again, someday.
      // prevB = (*(rp->*access_fn)() - prevS);
      prevBInitialized = true;
      prevB = 0.f;
    }

    // Follow the math to calculate new S and B values.
    float y = *(rp->*access_fn)();
    float newS = alpha * y                + ( 1.f - alpha ) * ( prevS + prevB );
    float newB = gamma * ( newS - prevS ) + ( 1.f - gamma ) * prevB;

    *(rp->*access_fn)() = prevS = newS;
    prevB = newB;
  }

  void CMA( reg_params *rp )
  {
    float y = *(rp->*access_fn)();
    *(rp->*access_fn)() = (y + flowCount*prevS) / (flowCount+1);
    prevS = *(rp->*access_fn)();
    flowCount++;
    std::cout << "FLow: " << flowCount << std::endl;
  }

};

#endif // DOUBLEEXPSMOOTHING_H
