/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ParallelDFT.h"
#include <stdio.h>
#include <math.h>

#ifndef PI
#define PI (3.141592653589793f)
#endif  // PI


ParallelDFT::ParallelDFT(unsigned int n_pts)
{
    SetPts(n_pts);
}


ParallelDFT::~ParallelDFT()
{
    m_twiddlesI.clear();
    m_twiddlesQ.clear();
    m_twiddlesQneg.clear();
}


int ParallelDFT::SetPts(unsigned int n_pts)
{
    if (n_pts != m_pts)
    {
        // Only update if changing size
        m_twiddlesI.resize(n_pts);
        m_twiddlesQ.resize(n_pts);
        m_twiddlesQneg.resize(n_pts);
        m_pts = n_pts;
        if (n_pts)
        {
            float kTwiddleScale = -2.0f*PI/static_cast<float>(n_pts);
            float fIdxVal;
            // Generate twiddle values
            for (unsigned int idx = 0; idx < n_pts; ++idx)
            {
                fIdxVal = static_cast<float>(idx)*kTwiddleScale;
                m_twiddlesI[idx]    = cos(fIdxVal);
                m_twiddlesQ[idx]    = sin(fIdxVal);
                m_twiddlesQneg[idx] = -m_twiddlesQ[idx];  // quicker than re-computing sin(-fIdxVal)
            }
        }
    }

    return static_cast<int>(m_pts);
}


/***
    Calculate a partial DFT on a pure real input signal.
*/
int ParallelDFT::PartialDFT(unsigned int n_freqOffset,
                            unsigned int n_numFreqs,
                            const float* srcData,
                            float*       dstDataI,
                            float*       dstDataQ)
{
    int retVal = 0;
    unsigned int freqIdx, timeIdx, maxFreq;
    register int   idx;
    register float tmpI, tmpQ;

    if (dstDataI && dstDataQ)
    {
        maxFreq = n_freqOffset + n_numFreqs;
        for (freqIdx = n_freqOffset; freqIdx < maxFreq; ++freqIdx)
        {
            tmpI = 0.0f;
            tmpQ = 0.0f;
            // Here's where we can get some parallel processing happening with SSE/AVX
            for (timeIdx = 0; timeIdx < m_pts; ++timeIdx)
            {
                idx   = (freqIdx * timeIdx) % m_pts;
                tmpI += srcData[timeIdx] * m_twiddlesI[idx];
                tmpQ += srcData[timeIdx] * m_twiddlesQ[idx];
            }
            dstDataI[retVal] = tmpI;
            dstDataQ[retVal] = tmpQ;
            ++retVal;
        }
    }

    return retVal;
}


/***
    Calculate an inverse DFT on a partial set of complex frequency elements.
    The BIG assumption here is that we're never feeding more than half the
    spectrum back in to the IDFT, which allows us to exploit the symmetry
    of the Fourier transofrm when expecting a real result, avoiding a lot
    of computation.
*/
int ParallelDFT::PartialIDFT(unsigned int n_freqOffset,
                             unsigned int n_numFreqs,
                             float*       dstData,
                             const float* srcDataI,
                             const float* srcDataQ)
{
    int retVal = 0;
    unsigned int freqIdx, timeIdx, maxFreq;
    register int   idx;
    register float tmp, scale;

    if (dstData && n_numFreqs)
    {
        maxFreq = n_freqOffset + n_numFreqs;
        scale = 1.0f / static_cast<float>(n_numFreqs);  // (1/N)*(N/n_numFreqs)
        for (timeIdx = 0; timeIdx < m_pts; ++timeIdx)
        {
            tmp = 0.0f;
            // Here's where we can get some parallel processing happening with SSE/AVX
            for (freqIdx = n_freqOffset; freqIdx < maxFreq; ++freqIdx)
            {
                idx  = (freqIdx * timeIdx) % m_pts;
                // We assume the result is going to be pure real (complex parts will cancel), so only calculate real components
                tmp += srcDataI[freqIdx] * m_twiddlesI[idx] - srcDataQ[freqIdx] * m_twiddlesQneg[idx];
            }
            dstData[retVal] = scale * tmp;
            ++retVal;
        }
    }

    return retVal;
}

