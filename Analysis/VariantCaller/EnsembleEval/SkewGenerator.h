/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef SKEWGENERATOR_H
#define SKEWGENERATOR_H


#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "CrossHypotheses.h"
#include "ShortStack.h"

// handle the case in which we have significant lopsidedness of measurements
class BasicSkewGenerator{
  public:
    vector<float> latent_skew;
    float dampened_skew;
    
    vector<float> skew_up;
    vector<float> skew_down;
    
    BasicSkewGenerator(){
      latent_skew.assign(2, 1.0f);

      skew_up.assign(2, 0.0f);
      skew_down.assign(2, 0.0f);
      dampened_skew = 30.0f; // identical to dampened bias
    };
    void GenerateSkew(CrossHypotheses &my_cross);
    void AddOneUpdateForHypothesis(int strand_key, float responsibility, vector<int> &test_flow, vector<float> &residuals);
    void AddCrossUpdate(CrossHypotheses &my_cross);
    
    void DoLatentUpdate();
    void ResetUpdate();
     // skew estimates
  void UpdateSkewGenerator(ShortStack &total_theory);
  void UpdateSkewEstimates(ShortStack &total_theory);
  void DoStepForSkew(ShortStack &total_theory);

    
};


#endif // SKEWGENERATOR_H
