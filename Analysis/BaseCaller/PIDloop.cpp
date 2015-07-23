/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     PIDloop.cpp
//! @ingroup  BaseCaller
//! @brief    PIDloop.  Implementation of PID control loop

#include "PIDloop.h"

const float PIDloop::kSaturation = 0.25f;

//! @brief  Default constructor
PIDloop::PIDloop()
    : kP_(1.0f),
      kI_(0.0f),
      kD_(0.0f)
{
    Initialize(0.0f);
}

//! @brief  Constructor
//! @param[in] gainP    Proportional gain coefficient
//! @param[in] gainI    Integral gain coefficient
//! @param[in] gainD    Derivative gain coefficient
//! @param[in] initVal  Initial y(t) loop value
PIDloop::PIDloop(const float gainP, const float gainI, const float gainD, const float initVal)
    : kP_(gainP),
      kI_(gainI),
      kD_(gainD)
{
    Initialize(initVal);
}
    
//! @brief  Initialize the PID loop to a known starting value
void PIDloop::Initialize(const float initVal)
{
    yt_   = initVal;
    dt_   = 1.0f;
    et_   = 0.0f;
    st_   = 0.0f;
    iVal_ = 0.0f;
    dVal_ = 0.0f;
}

//! @brief  Iterate one time step
//! @param[in] val  Next time-ordered sample to filter
//! @return Next time-ordered filtered sample
float PIDloop::Step(const float val)
{
    float et = val - yt_;

    // apply limiting to error signal (anti-windup guard)
    if (et < -kSaturation)
        et = -kSaturation;
    else if (et > kSaturation)
        et = kSaturation;

    iVal_ += et;  // integral
    dVal_  = (et - et_) / dt_;  // derivative
    yt_   += kP_ * et + kI_ * iVal_ + kD_ * dVal_;

    // update for next time
    et_    = et;
    dt_    = 1.0f;
    st_    = val;

    return yt_;
}

//! @brief  Iterate one time step without input
//! @return Next time-ordered filtered sample (extrapolated)
float PIDloop::Step()
{
    float et = st_ - yt_;

    // apply limiting to error signal (anti-windup guard)
    if (et < -kSaturation)
        et = -kSaturation;
    else if (et > kSaturation)
        et = kSaturation;

    yt_   += kP_ * et;
    dt_   += 1.0f;
    iVal_ /= 2.0f;  // anti-windup

    return yt_;
}
