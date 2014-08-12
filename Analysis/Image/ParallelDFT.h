/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PARALLELDFT_H
#define PARALLELDFT_H

#include <vector>


/***
 Discrete Fourier Transform implementation optimized to process only frequency
 elements that will actually be required by the DFC algorithm.  Initial
 implementation is not optimal, but contains comments where SSE/AVX instructions
 can be substituted for performance enhancement.
*/
class ParallelDFT
{
public:

    ParallelDFT() { m_pts = 0; m_twiddlesI.clear(); m_twiddlesQ.clear(); m_twiddlesQneg.clear(); }

    ParallelDFT(unsigned int n_pts);
    
    ~ParallelDFT();

    unsigned int GetPts() { return m_pts; }

    int SetPts(unsigned int n_pts);

    /***
     Calculate a partial DFT on a pure real input signal.
    */
    int PartialDFT(unsigned int n_freqOffset,
                   unsigned int n_numFreqs,
                   const float* srcData,
                   float*       dstDataI,
                   float*       dstDataQ);

    /***
     Calculate an inverse DFT on a partial set of complex frequency elements.
    */
    int PartialIDFT(unsigned int n_freqOffset,
                    unsigned int n_numFreqs,
                    float*       dstData,
                    const float* srcDataI,
                    const float* srcDataQ);

private:
    unsigned int        m_pts;
    std::vector<float>  m_twiddlesI;
    std::vector<float>  m_twiddlesQ;
    std::vector<float>  m_twiddlesQneg;

};


#endif //PARALLELDFT_H
