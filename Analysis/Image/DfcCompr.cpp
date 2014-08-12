/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <algorithm>
#include "DfcCompr.h"
//#include "Utils.h"
#include <malloc.h>
#include <cmath>
// DFC doesn't use anything from the Eigen library
//#define EIGEN_USE_MKL_ALL 1
//#include <Eigen/Dense>
//#include <Eigen/LU>

using namespace std;
//using namespace Eigen;


/***
 The DFT engine needs to be kept in sync with any changes to the number of frames.
*/
int DfcCompr::SetNumFrames(int frames)
{
    if ((frames > 0) && (frames != n_frames))
    {
        n_frames = frames;
        DFT.SetPts(frames);
    }

    return n_frames;
}


/***
 Range check the new window length value and generate the new window coefficients.
 Window lengths should be relatively short.
*/
int DfcCompr::SetWindowLength(int length)
{
    if ((length >= 0) && (length < (n_frames/4)) && (length != winParamLen))
    {
        winParamLen = length;
        GenerateWindow();
    }

    return winParamLen;
}


/***
 Range check the new window alpha value and generat the new window coefficients.
*/
float DfcCompr::SetWindowAlpha(float alpha)
{
    if ((alpha >= 0.0f) && (alpha <= 3.0f) && (alpha != winParamAlpha))
    {
        winParamAlpha = alpha;
        GenerateWindow();
    }

    return winParamAlpha;
}


/***
 Generate a new set of Gaussian window coefficients based on the current set of
 window length and alpha configuration parameters.
*/
void DfcCompr::GenerateWindow()
{
    // setup vectors to store compression and decompression windows
    windowCompress.resize(winParamLen*2);
    windowExpand.resize(winParamLen*2);

    float halfN = static_cast<float>(winParamLen);
    float n     = 0.5f-halfN;  // starting at -(N-1)/2
    int   idx, maxIdx;
    float an_sq, val, oneOnVal;

    maxIdx = (winParamLen * 2) - 1;
    for (idx = 0; idx < winParamLen; ++idx)
    {
        an_sq    = winParamAlpha * (n / halfN);
        val      = exp((-0.5f) * (an_sq * an_sq));
        oneOnVal = 1.0f / val;
        n       += 1.0f;  // setup n for next loop iteration
        windowCompress[idx]        = val;
        windowCompress[maxIdx-idx] = val;  // exploit window symmetry
        windowExpand[idx]          = oneOnVal;
        windowExpand[maxIdx-idx]   = oneOnVal;
    }

}


/***
  Decompose the image data into a set of basis vectors and each well's projection
  onto them.
  @param n_wells - number of wells in the image patch
  @param n_frame - number of frames in this image patch.
  @param image - Same order as RawImage structure. Individual wells of data in frame, row, col major order so
                 value(row_i,col_j,frame_k) = image[row_i * ncol + col_j + (nrow * ncol * frame_k)]
  @param n_sample_wells - number of wells in the sample of the patch
  @param image_sample - sample of image above for vectors and such
  @param compressed - output of a lossy compressed patch
*/
void DfcCompr::LossyCompress(float *image)
{
    //////////////////////////////////////////
    // Temporary scratch_pad memory allocation
    if (scratch_pad == NULL) scratch_pad = new float[n_wells*n_basis*2];
    //////////////////////////////////////////

    int wellIdx, frameIdx;
    float tmpWell[n_frames];
    register float tmpSum;
    register int offset, winIdx;
    float * __restrict dftPtrI = &scratch_pad[0];
    float * __restrict dftPtrQ = &scratch_pad[n_wells*n_basis];
    float deltaMagI[n_basis];
    float deltaMagQ[n_basis];

    // Initialize keyframe
    for (frameIdx = 0; frameIdx < n_basis; ++frameIdx)
    {
        keyFrameI[frameIdx] = 0.0f;
        keyFrameQ[frameIdx] = 0.0f;
    }

    for (wellIdx = 0, offset = 0; wellIdx < n_wells; ++wellIdx)
    {
        // Step 1a - extract the next well from the image cube
        tmpSum = 0.0f;
        winIdx = 0;
        for (frameIdx = 0; frameIdx < n_frames; ++frameIdx)
        {
            tmpWell[frameIdx] = image[(frameIdx*n_wells)+wellIdx];
            tmpSum           += image[(frameIdx*n_wells)+wellIdx];
        }
        // Step 1b - remove DC offset and apply windowing to each well
        tmpSum /= static_cast<float>(n_frames);  // DC offset
        // Apply front part of window
        for (frameIdx = 0; frameIdx < winParamLen; ++frameIdx)
        {
            tmpWell[frameIdx] -= tmpSum;
            tmpWell[frameIdx] *= windowCompress[winIdx++];
        }
        // Middle part (window is unity)
        for (; frameIdx < (n_frames-winParamLen); ++frameIdx)
            tmpWell[frameIdx] -= tmpSum;
        // Apply back part of window
        for (; frameIdx < n_frames; ++frameIdx)
        {
            tmpWell[frameIdx] -= tmpSum;
            tmpWell[frameIdx] *= windowCompress[winIdx++];
        }
        // Step 2 - Compute partial DFT
        // store for deltas and accumulate for keyframe
        winIdx = n_basis * wellIdx;
        DFT.PartialDFT(1, static_cast<unsigned int>(n_basis), &tmpWell[0], &dftPtrI[winIdx], &dftPtrQ[winIdx]);
        for (frameIdx = 0; frameIdx < n_basis; ++frameIdx, ++offset)
        {
            keyFrameI[frameIdx] += dftPtrI[offset];
            keyFrameQ[frameIdx] += dftPtrQ[offset];
        }
    }

    // Step 3 - Convert accumulated spectrum into mean keyframe and initialize
    // maximum delta magnitude vector
    for (frameIdx = 0; frameIdx < n_basis; ++frameIdx)
    {
        keyFrameI[frameIdx] /= static_cast<float>(n_wells);
        keyFrameQ[frameIdx] /= static_cast<float>(n_wells);
        deltaMagI[frameIdx]  = 0.0f;
        deltaMagQ[frameIdx]  = 0.0f;
    }

    // Step 4 - Calculate the correlation statistics and emphasis vector needed to
    // populate the bits per frequency element vector
    Emphasis();

    // Step 5 - Convert DFT data from absolute to raw delta, keeping maximum
    // component magnitude observed for each frequency component along the way
    for (wellIdx = 0, offset = 0; wellIdx < n_wells; ++wellIdx)
        // May be able to use SSE/AVX vector acceleration here to process multiple frequency elements in parallel
        for (frameIdx = 0; frameIdx < n_basis; ++frameIdx, ++offset)
        {
            dftPtrI[offset] -= keyFrameI[frameIdx];
            dftPtrQ[offset] -= keyFrameQ[frameIdx];
            if (deltaMagI[frameIdx] < abs(dftPtrI[offset])) deltaMagI[frameIdx] = abs(dftPtrI[offset]);  // update maximum observed real magnitude vector
            if (deltaMagQ[frameIdx] < abs(dftPtrQ[offset])) deltaMagQ[frameIdx] = abs(dftPtrQ[offset]);  // update maximum observed imaginary magnitude vector
        }

    // Populate the scale vectors
    for (frameIdx = 0; frameIdx < n_basis; ++frameIdx)
    {
        tmpSum = static_cast<float>((1<<static_cast<int>(bitsPerFreq[frameIdx]-1))-1);
        scaleVectorI[frameIdx] = deltaMagI[frameIdx] / tmpSum;
        scaleVectorQ[frameIdx] = deltaMagQ[frameIdx] / tmpSum;
    }

    // Step 6 - Quantize the spectrum delta values and pack into output vectors
    for (wellIdx = 0, offset = 0; wellIdx < n_wells; ++wellIdx)
        // May be able to use SSE/AVX vector acceleration here to process multiple frequency elements in parallel
        for (frameIdx = 0; frameIdx < n_basis; ++frameIdx, ++offset)
        {
            deltaI[offset] = static_cast<short>(round(dftPtrI[offset]/scaleVectorI[frameIdx]));
            deltaQ[offset] = static_cast<short>(round(dftPtrQ[offset]/scaleVectorQ[frameIdx]));
        }

    ////////////////////////////////////////////
    // Temporary scratch_pad memory deallocation
    delete[] scratch_pad; scratch_pad = NULL;
    ////////////////////////////////////////////
}


/***
*/
void DfcCompr::Emphasis()
{
    float kernelI[n_basis];
    float kernelQ[n_basis];
    float *dftPtrI = &scratch_pad[0];
    float *dftPtrQ = &scratch_pad[n_wells*n_basis];
    float *corrI = new float[n_wells*n_basis];  // probably a good idea to make this static or semi-static
    float *corrQ = new float[n_wells*n_basis];  // probably a good idea to make this static or semi-static
    register int offset, freqIdx;
    int   wellIdx;
    float meanMag[n_basis];
    float meanAng[n_basis];
    float varMag[n_basis];
    float varAng[n_basis];
    register float delta;
    float emphasisVector[n_basis];
    register int bitVal;

    // Generate correlation kernel from keyframe and initialize statistics vectors
    for (freqIdx = 0; freqIdx < n_basis; ++freqIdx)
    {
        kernelI[freqIdx] = keyFrameI[freqIdx];
        kernelQ[freqIdx] = -keyFrameQ[freqIdx];
        meanMag[freqIdx] = 0.0f;
        meanAng[freqIdx] = 0.0f;
        varMag[freqIdx]  = 0.0f;
        varAng[freqIdx]  = 0.0f;
    }

    // Correlate each well spectrum with the complex conjugate of the mean well spectrum
    for (wellIdx = 0, offset = 0; wellIdx < n_wells; ++wellIdx)
        // May be able to vectorize the correlation process
        for (freqIdx = 0; freqIdx < n_basis; ++freqIdx, ++offset)
        {
            // DFT data        ==> (a+ib)
            // kernel          ==> (x+ib)
            // correlated data ==> (a+ib)(x+iy) = ax+iay+ibx-by = (ax-by)+i(ay+bx)
            corrI[offset] = dftPtrI[offset]*kernelI[freqIdx]-dftPtrQ[offset]*kernelQ[freqIdx];  // (ax-by)
            corrQ[offset] = dftPtrI[offset]*kernelQ[freqIdx]+dftPtrQ[offset]*kernelI[freqIdx];  // (ay+bx)
        }

    // Find mean of magnitude and angle of each frequency element
    for (wellIdx = 0, offset = 0; wellIdx < n_wells; ++wellIdx)
        // May be able to vectorize the frequency elements
        for (freqIdx = 0; freqIdx < n_basis; ++freqIdx, ++offset)
        {
            meanMag[freqIdx] += sqrt((corrI[offset]*corrI[offset])+(corrQ[offset]*corrQ[offset]));
            meanAng[freqIdx] += atan2(corrQ[offset], corrI[offset]);
        }
    for (freqIdx = 0; freqIdx < n_basis; ++freqIdx)
    {
        meanMag[freqIdx] /= static_cast<float>(n_wells);
        meanAng[freqIdx] /= static_cast<float>(n_wells);
    }

    // Find standard deviation (now that we have the mean)
    for (wellIdx = 0, offset = 0; wellIdx < n_wells; ++wellIdx)
        // May be able to vectorize the frequency elements
        for (freqIdx = 0; freqIdx < n_basis; ++freqIdx, ++offset)
        {
            delta = sqrt((corrI[offset]*corrI[offset])+(corrQ[offset]*corrQ[offset])) - meanMag[freqIdx];
            varMag[freqIdx] += delta * delta;  // accumulate magnitude variance
            delta = atan2(corrQ[offset], corrI[offset]) - meanAng[freqIdx];
            varAng[freqIdx] += delta * delta;  // accumulate angle variance
        }

    // Finished with correlation data (don't need to release once we make this block of memory static/semi-static)
    delete[] corrI;
    delete[] corrQ;

    // Convert accumulated variance to raw emphasis (original Matlab calculation
    // uses standard deviation rather than variance, but that uses an additional
    // sqrt step that has been optimized out here)
    for (freqIdx = 0, delta = 0.0f; freqIdx < n_basis; ++freqIdx)
    {
        emphasisVector[freqIdx] = sqrt((varMag[freqIdx]*varAng[freqIdx])/static_cast<float>(n_wells));
        if (emphasisVector[freqIdx] > delta)
            delta = emphasisVector[freqIdx];  // keep the maximum value for normalization
    }
    
    // Normalize the emphasis vector and convert it to bits per frequency element
    for (freqIdx = 0; freqIdx < n_basis; ++freqIdx)
    {
        // calculate the number of magnitude bits to allocate (log will always return zero or a negative value)
        bitVal = static_cast<int>(static_cast<float>(n_maxBits)+ceil(log2(emphasisVector[freqIdx]/delta)));
        if (bitVal < n_minBits) bitVal = n_minBits;  // ensure at least the minimum number of magnitude bits are allocated
        ++bitVal;  // add the sign bit
        bitsPerFreq[freqIdx] = static_cast<unsigned char>(bitVal);
    }
}


/***
 Reconstruct the image data from the compressed frequency domain vectors.
 @param n_wells - number of wells in the image patch
 @param n_frame - number of frames in this image patch.
 @param image   - Same order as RawImage structure. Individual wells of data in frame, row, col major order so
                  value(row_i,col_j,frame_k) = image[row_i * ncol + col_j + (nrow * ncol * frame_k)]
*/
void DfcCompr::LossyUncompress(float *image)
{
    float wellBufI[n_basis];
    float wellBufQ[n_basis];
    float tmpBuf[n_frames];
    int   wellIdx, frameIdx;
    register int offset, winIdx;

    for (wellIdx = 0, offset = 0; wellIdx < n_wells; ++wellIdx)
    {
        // Reconstruct partial spectrum from keyframe, delta and scale information
        for (frameIdx = 0; frameIdx < n_basis; ++frameIdx, ++offset)
        {
            wellBufI[frameIdx] = keyFrameI[frameIdx] + scaleVectorI[frameIdx]*static_cast<float>(deltaI[offset]);
            wellBufQ[frameIdx] = keyFrameQ[frameIdx] + scaleVectorQ[frameIdx]*static_cast<float>(deltaQ[offset]);
        }
        // Reconstruct time domain signal from partial spectrum using IDFT
        DFT.PartialIDFT(1, n_basis, &tmpBuf[0], &wellBufI[0], &wellBufQ[0]);
        // Store reconstructed values into the image cube
        for (frameIdx = 0, winIdx = 0; frameIdx < winParamLen; ++frameIdx)
            image[(frameIdx*n_wells)+wellIdx] = tmpBuf[frameIdx] * windowExpand[winIdx++];
        for (; frameIdx < (n_frames-winParamLen); ++frameIdx)
            image[(frameIdx*n_wells)+wellIdx] = tmpBuf[frameIdx];
        for (; frameIdx < n_frames; ++frameIdx)
            image[(frameIdx*n_wells)+wellIdx] = tmpBuf[frameIdx] * windowExpand[winIdx++];
    }
}


/*****************************************************************************/

/***
 Map the memory blocks and call LossyCompress.
*/
void DfcComprWrapper::LossyCompress(float *image)
{
    SetupPointers();
    dfc.LossyCompress(image);
};


/***
 Map the memory blocks and call LossyUncompress.
*/
void DfcComprWrapper::LossyUncompress(float *image)
{
    SetupPointers();
    dfc.LossyUncompress(image);
};


/***
 Do some evil pointer arithmetic to map the DFC header and payload data vectors
 into the basis_vectors and coefficients memory blocks.
*/
void DfcComprWrapper::SetupPointers()
{
    void *hdrPtr = reinterpret_cast<void *>(basis_vectors);
    void *dtrPtr = reinterpret_cast<void *>(coefficients);
    int   offset = sizeof(float) * dfc.n_basis;

    // Map the header vectors into the basis_vectors memory block
    dfc.keyFrameI    = reinterpret_cast<float *>(hdrPtr);
    dfc.keyFrameQ    = reinterpret_cast<float *>(hdrPtr) + offset;
    dfc.scaleVectorI = reinterpret_cast<float *>(hdrPtr) + 2*offset;
    dfc.scaleVectorQ = reinterpret_cast<float *>(hdrPtr) + 3*offset;
    dfc.bitsPerFreq  = reinterpret_cast<unsigned char *>(hdrPtr) + 4*offset;

    // Map the delta data into the coefficients memory block
    offset     = sizeof(short) * dfc.n_basis * dfc.n_wells;
    dfc.deltaI = reinterpret_cast<short *>(dtrPtr);
    dfc.deltaQ = reinterpret_cast<short *>(dtrPtr) + offset;
}
