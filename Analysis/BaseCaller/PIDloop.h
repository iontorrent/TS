/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     PIDloop.h
//! @ingroup  BaseCaller
//! @brief    PIDloop.  Implementation of PID control loop

#ifndef PIDLOOP_H
#define PIDLOOP_H

class PIDloop
{
public:
    //! @brief  Default constructor
    PIDloop();

    //! @brief  Constructor
    //! @param[in] gainP    Proportional gain coefficient
    //! @param[in] gainI    Integral gain coefficient
    //! @param[in] gainD    Derivative gain coefficient
    //! @param[in] initVal  Initial y(t) loop value
    PIDloop(const float gainP, const float gainI = 0.0f, const float gainD = 0.0f, const float initVal = 0.0f);
    
    //! @brief  Initialize the PID loop to a known starting value
    void Initialize(const float initVal = 0.0f);

    //! @brief  Iterate one time step
    //! @param[in] val  Next time-ordered sample to filter
    //! @return Next time-ordered filtered sample
    float Step(const float val);

    //! @brief  Iterate one time step without input
    //! @return Next time-ordered filtered sample (extrapolated)
    float Step();

protected:
    float   kP_;
    float   kI_;
    float   kD_;
    float   yt_;
    float   dt_;        // time between error terms
    float   et_;        // error term e(t)
    float   st_;        // previous step input value
    float   iVal_;      // integral e(t)
    float   dVal_;      // derivative e(t)

    const static float kSaturation;
};


#endif  // PIDLOOP_H
