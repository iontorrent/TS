/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

#include <float.h>

#include "Utils.h"
#include "ComparatorNoiseCorrector.h"

void ComparatorNoiseCorrector::CorrectComparatorNoise(RawImage *raw, Mask *mask, bool verbose,bool aggressive_correction, bool beadfind_image)
{
   CorrectComparatorNoise(raw->image,raw->rows,raw->cols,raw->frames,mask,verbose,aggressive_correction,beadfind_image);
}

void ComparatorNoiseCorrector::CorrectComparatorNoise(short *image, int rows, int cols, int frames, Mask *mask, bool verbose, bool aggressive_correction, bool beadfind_image)
{

   if (aggressive_correction)
      NNSpan = 4;
   else
      NNSpan = 1;

   if (!beadfind_image)
   {
      CorrectComparatorNoise_internal(image, rows, cols, frames,  mask, verbose, aggressive_correction);

      if (aggressive_correction)
      {
         int blk_size = 96;
         int sub_blocks = rows / blk_size;

         if (blk_size * sub_blocks < rows) sub_blocks++;

         for (int blk = 0; blk < sub_blocks; blk++)
         {
            int row_start = blk * blk_size;
            int row_end = (blk + 1) * blk_size;

            if (blk == sub_blocks - 1) row_end = rows;

            CorrectComparatorNoise_internal(image, rows, cols, frames,  mask, verbose, aggressive_correction, row_start, row_end);
         }
      }
   }
   else
   {
      // trick correction into only doing hf noise correction as this is all that works for beadfind images
      CorrectComparatorNoise_internal(image, rows, cols, frames,  mask, verbose, aggressive_correction, 0, rows);
   }

}

void ComparatorNoiseCorrector::CorrectComparatorNoise_internal(short *image, int rows, int cols, int frames, Mask *mask, bool verbose, bool aggressive_correction, int row_start,int row_end)
{
  float comparator_rms[cols*2];
  int comparator_mask[cols*2];
  float comparator_hf_rms[cols*2];
  int comparator_hf_mask[cols*2];
  float pcomp[frames];
  int c_avg_num[cols*4];
  int x,y,frame,i;
  int phase;
  int frameStride = rows * cols;
  bool hfonly = false;
  Allocate(cols*4*frames);
  memset(mComparator_sigs,0,sizeof(float) * cols*4*frames);
  memset(mComparator_noise,0,sizeof(float) * cols*2*frames);
  memset(c_avg_num,0,sizeof(c_avg_num));
  memset(comparator_mask,0,sizeof(comparator_mask));
  memset(comparator_hf_mask,0,sizeof(comparator_hf_mask));

  if (row_start == -1)
  {
	row_start = 0;
	row_end = rows;
	hfonly = false;
  }
  else
	hfonly = true;

  // first, create the average comparator signals
  // making sure to avoid pinned pixels
  i=row_start*cols;
  for ( y=row_start;y<row_end;y++ ) {
    for ( x=0;x<cols;x++ ) {
      int cndx;
      int frame;
      float *cptr;

      // if this pixel is pinned..skip it
      if ( (( *mask ) [i] & MaskPinned)==0 )
      {
        // figure out which comparator this pixel belongs to
        // since we don't know the phase just yet, we first split each column
        // up into 4 separate comparator signals, even though
        // there are really only 2 of them
        cndx = 4*x + (y&0x3);
        
        // get a pointer to where we will build the comparator signal average
        cptr = mComparator_sigs + cndx*frames;
  
        // add this pixels' data in
        for ( frame=0;frame<frames;frame++ )
          cptr[frame] += image[frame*frameStride+i];
  
        // count how many we added
        c_avg_num[cndx]++;
      }

      i++;
    }
  }

  // divide by the number to make a proper average
  for ( int cndx=0;cndx < cols*4;cndx++ )
  {
    float *cptr;

    if ( c_avg_num[cndx] == 0 )
      continue;

    // get a pointer to where we will build the comparator signal average
    cptr = mComparator_sigs + cndx*frames;

    // divide by corresponding count, extreme case: divide by zero if all pixels are pinned
    if(c_avg_num[cndx] != 0){
      for ( frame=0;frame<frames;frame++ ){
        cptr[frame] /= c_avg_num[cndx];
      }
    }
  }

  // subtract DC offset from average comparator signals
  for ( int cndx=0;cndx < cols*4;cndx++ )
  {
    float *cptr;
    float dc = 0.0f;

    // get a pointer to where we will build the comparator signal average
    cptr = mComparator_sigs + cndx*frames;

    for ( frame=0;frame<frames;frame++ )
      dc += cptr[frame];

    dc /= frames;

    // subtract dc offset
    for ( frame=0;frame<frames;frame++ )
      cptr[frame] -= dc;
  }

  // now figure out which pair of signals go together
  // this function also combines pairs of signals accordingly
  // from this point forward, there are only cols*2 signals to deal with
  phase = DiscoverComparatorPhase(mComparator_sigs,c_avg_num,cols*4,frames);

  //special case of rms==0, assign phase to -1
  if(phase != -1){
      // mask comparators that don't contain any un-pinned pixels
      for ( int cndx=0;cndx < cols*2;cndx++ )
      {
        if ( c_avg_num[cndx] == 0 )
        {
          comparator_mask[cndx] = 1;
          continue;
        }
      }

      // now neighbor-subtract the comparator signals
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,comparator_mask,NNSpan,cols*2,frames);

      // measure noise in the neighbor-subtracted signals
      CalcComparatorSigRMS(comparator_rms,mComparator_noise,cols*2,frames);

      // find the noisiest 10%
      MaskIQR(comparator_mask,comparator_rms,cols*2);

      // neighbor-subtract again...avoiding noisiest 10%
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,comparator_mask,NNSpan,cols*2,frames);

      // measure noise in the neighbor-subtracted signals
      CalcComparatorSigRMS(comparator_rms,mComparator_noise,cols*2,frames);

      // reset comparator_mask
      memset(comparator_mask,0,sizeof(comparator_mask));
      for ( int cndx=0;cndx < cols*2;cndx++ )
      {
        if ( c_avg_num[cndx] == 0 )
        {
          comparator_mask[cndx] = 1;
        }
      }
      MaskIQR(comparator_mask,comparator_rms,cols*2, verbose);

      // neighbor-subtract again...avoiding noisiest 10%
      NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,comparator_mask,NNSpan,cols*2,frames);

      if (aggressive_correction)
      {
         // Newly added stuff.
             // subtracts some of what we detect as comparator noise from neighbors before forming the nn average
             // this cleans things up a little

         // make another set of noise signals that have been run through a high-pass filter
         // filter low frequency noise out of noise signals
         memcpy(mComparator_hf_noise,mComparator_noise,sizeof(float)*cols*2*frames);
         HighPassFilter(mComparator_hf_noise,cols*2,frames,10);

         // neighbor-subtract again...now with some rejection of what we think the noise is
         NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,comparator_mask,NNSpan,cols*2,frames,mComparator_hf_noise);

         // measure noise in the neighbor-subtracted signals
         CalcComparatorSigRMS(comparator_rms,mComparator_noise,cols*2,frames);

         // reset comparator_mask
         memset(comparator_mask,0,sizeof(int)*cols*2);
         for ( int cndx=0;cndx < cols*2;cndx++ )
         {
           if ( c_avg_num[cndx] == 0 )
           {
             comparator_mask[cndx] = 1;
           }
         }
   //      MaskIQR(comparator_mask,comparator_rms,cols*2, verbose);
         MaskUsingDynamicStdCutoff(comparator_mask,comparator_rms,cols*2,1.0f);

         // even if some comparators didn't make the cut with the raw noise signal
         // we can correct more agressively if we put the noise signal through the high pass filter

         // redo the high-pass fitler
         memcpy(mComparator_hf_noise,mComparator_noise,sizeof(float)*cols*2*frames);
         // get first principal component
         GetPrincComp(pcomp,mComparator_hf_noise,comparator_mask,cols*2,frames);
         FilterUsingPrincComp(mComparator_hf_noise,pcomp,cols*2,frames);

         // measure high frequency noise
         CalcComparatorSigRMS(comparator_hf_rms,mComparator_hf_noise,cols*2,frames);

         for ( int cndx=0;cndx < cols*2;cndx++ )
         {
           if ( c_avg_num[cndx] == 0 )
           {
             comparator_hf_mask[cndx] = 1;
           }
         }
         MaskUsingDynamicStdCutoff(comparator_hf_mask,comparator_hf_rms,cols*2,2.0f);
      }

      // blanks comparator signal averages that didn't have any pixels to average (probably redundant)
      for ( int cndx=0;cndx < cols*2;cndx++ )
      {
        float *cptr;

        if ( c_avg_num[cndx] == 0 ) {
          // get a pointer to where we will build the comparator signal average
          cptr = mComparator_sigs + cndx*frames;
          memset(cptr,0,sizeof(float[frames]));
        }
      }

      // now subtract each neighbor-subtracted comparator signal from the
      // pixels that are connected to that comparator
      i=row_start*cols;
      for ( y=row_start;y<row_end;y++ ) {
        for ( x=0;x<cols;x++ ) {
          int cndx;
          int frame;
          float *cptr;

          // if this pixel is pinned..skip it
          if ( (( *mask ) [i] & MaskPinned)==0 )
          {
            // figure out which comparator this pixel belongs to
            cndx = 2*x;
            if (( (y&0x3) == (0+phase) ) || ( (y&0x3) == (1+phase) ))
                cndx++;

            //only perform correction on noisy comparators;
            if(comparator_mask[cndx] && !hfonly){
              // get a pointer to where we will build the comparator signal average
              cptr = mComparator_noise + cndx*frames;
              // subtract nn comparator signal from this pixel's data
              for ( frame=0;frame<frames;frame++ )
                image[frame*frameStride+i] -= cptr[frame];
            } else if (comparator_hf_mask[cndx])
            {
              // get a pointer to where we will build the comparator signal average
              cptr = mComparator_hf_noise + cndx*frames;
              // subtract nn comparator signal from this pixel's data
              for ( frame=0;frame<frames;frame++ )
                image[frame*frameStride+i] -= cptr[frame];
            }
          }

          i++;
        }
      }
  }
}

void ComparatorNoiseCorrector::CorrectComparatorNoiseThumbnail(RawImage *raw,Mask *mask, int regionXSize, int regionYSize, bool verbose) {
  CorrectComparatorNoiseThumbnail(raw->image, raw->rows, raw->cols, raw->frames, mask, regionXSize, regionYSize, verbose);
}

void ComparatorNoiseCorrector::CorrectComparatorNoiseThumbnail(short *image, int rows, int cols, int frames, Mask *mask, int regionXSize, int regionYSize, bool verbose)
{
  int frameStride = rows * cols;
  time_t cnc_start;
  time ( &cnc_start );
  MemUsage ( "Starting Comparator Noise Correction" );
  if( cols%regionXSize != 0 || rows%regionYSize != 0){
    //skip correction
    fprintf (stdout, "Region sizes are not compatible with image(%d x %d): %d x %d", rows, cols, regionYSize, regionXSize);
  }
  int nXPatches = cols / regionXSize;
  int nYPatches = rows / regionYSize;

  // float *mComparator_sigs;
  // float *mComparator_noise;
  float comparator_rms[regionXSize*2];
  int comparator_mask[regionXSize*2];
  int c_avg_num[regionXSize*4];
  int phase;

  Allocate(cols * 4 * frames);
  for(int pRow = 0; pRow < nYPatches; pRow++){
    for(int pCol = 0; pCol < nXPatches; pCol++){
        if(verbose)
          fprintf (stdout, "Patch y: %d, Patch x: %d\n", pRow, pCol);
        memset(mComparator_sigs,0,sizeof(float) * regionXSize*4*frames);
        memset(mComparator_noise,0,sizeof(float) * regionXSize*2*frames);
        memset(c_avg_num,0,sizeof(c_avg_num));
        memset(comparator_mask,0,sizeof(comparator_mask));

        // first, create the average comparator signals
        // making sure to avoid pinned pixels

        for ( int y=0;y<regionYSize;y++ ) {
          for(int x=0;x<regionXSize;x++ ) {
            // if this pixel is pinned..skip it
            int imgInd = cols * (y+pRow*regionYSize) + x + pCol*regionXSize;
            if ( (( *mask ) [imgInd] & MaskPinned)==0 )
            {
              // figure out which comparator this pixel belongs to
              // since we don't know the phase just yet, we first split each column
              // up into 4 separate comparator signals, even though
              // there are really only 2 of them
              int cndx = 4*x + (y&0x3);

              // get a pointer to where we will build the comparator signal average
              float *cptr = mComparator_sigs + cndx*frames;

              // add this pixels' data in
              for (int frame=0;frame<frames;frame++ )
                cptr[frame] += image[frame*frameStride+imgInd];

              // count how many we added
              c_avg_num[cndx]++;
            }
          }
        }

        // divide by the number to make a proper average
        for ( int cndx=0;cndx < regionXSize*4;cndx++ )
        {
          if ( c_avg_num[cndx] == 0 )
            continue;

          // get a pointer to where we will build the comparator signal average
          float *cptr = mComparator_sigs + cndx*frames;

          // divide by corresponding count, extreme case: divide by zero if all pixels are pinned
          if(c_avg_num[cndx] != 0){
            for (int frame=0;frame<frames;frame++ ){
              cptr[frame] /= c_avg_num[cndx];
            }
          }
        }

        // subtract DC offset from average comparator signals
        for ( int cndx=0;cndx < regionXSize*4;cndx++ )
        {
          float *cptr;
          float dc = 0.0f;

          // get a pointer to where we will build the comparator signal average
          cptr = mComparator_sigs + cndx*frames;

          for (int frame=0;frame<frames;frame++ )
            dc += cptr[frame];

          dc /= frames;

          // subtract dc offset
          for (int frame=0;frame<frames;frame++ )
            cptr[frame] -= dc;
        }

        // now figure out which pair of signals go together
        // this function also combines pairs of signals accordingly
        // from this point forward, there are only cols*2 signals to deal with
        phase = DiscoverComparatorPhase(mComparator_sigs,c_avg_num,regionXSize*4,frames);

        if(phase == -1){
//          fprintf (stdout, "Comparator Noise Correction skipped\n");
          continue;
        }

        // mask comparators that don't contain any un-pinned pixels
        for ( int cndx=0;cndx < regionXSize*2;cndx++ )
        {
          if ( c_avg_num[cndx] == 0 )
          {
            comparator_mask[cndx] = 1;
            continue;
          }
        }

        // now neighbor-subtract the comparator signals
        NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,comparator_mask,NNSpan, regionXSize*2,frames);

        // measure noise in the neighbor-subtracted signals
        CalcComparatorSigRMS(comparator_rms,mComparator_noise,regionXSize*2,frames);

        // find the noisiest 10%
        MaskIQR(comparator_mask,comparator_rms,regionXSize*2);

        // neighbor-subtract again...avoiding noisiest 10%
        NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,comparator_mask,NNSpan,regionXSize*2,frames);

        // measure noise in the neighbor-subtracted signals
        CalcComparatorSigRMS(comparator_rms,mComparator_noise,regionXSize*2,frames);

        // reset comparator_mask
        memset(comparator_mask,0,sizeof(comparator_mask));
        for ( int cndx=0;cndx < regionXSize*2;cndx++ )
        {
          if ( c_avg_num[cndx] == 0 )
          {
            comparator_mask[cndx] = 1;
          }
        }
        MaskIQR(comparator_mask,comparator_rms,regionXSize*2, verbose);

        // neighbor-subtract again...avoiding noisiest 10%
        NNSubtractComparatorSigs(mComparator_noise,mComparator_sigs,comparator_mask,NNSpan,regionXSize*2,frames);

        for ( int cndx=0;cndx < regionXSize*2;cndx++ )
        {
          float *cptr;

          if ( c_avg_num[cndx] == 0 ) {
            // get a pointer to where we will build the comparator signal average
            cptr = mComparator_sigs + cndx*frames;
            memset(cptr,0,sizeof(float[frames]));
          }
        }

        // now subtract each neighbor-subtracted comparator signal from the
        // pixels that are connected to that comparator
        for ( int y=0;y<regionYSize;y++ ) {
          for(int x=0;x<regionXSize;x++ ) {
            int imgInd = cols * (y+pRow*regionYSize) + x + pCol*regionXSize;

            // if this pixel is pinned..skip it
            if ( (( *mask ) [imgInd] & MaskPinned)==0 )
            {
              // figure out which comparator this pixel belongs to
              int cndx = 2*x;
              if (( (y&0x3) == (0+phase) ) || ( (y&0x3) == (1+phase) ))
                  cndx++;

              //only perform correction on noisy comparators;
              if(comparator_mask[cndx]){
                // get a pointer to where we will build the comparator signal average
                float *cptr = mComparator_noise + cndx*frames;
                // subtract nn comparator signal from this pixel's data
                for ( int frame=0;frame<frames;frame++ )
                  image[frame*frameStride+imgInd] -= cptr[frame];
              }
            }
          }
        }
    }
  }

  MemUsage ( "After Comparator Noise Correction" );
  time_t cnc_end;
  time ( &cnc_end );

  fprintf (stdout, "Comparator Noise Correction: %0.3lf sec.\n", difftime(cnc_end, cnc_start));
}


int ComparatorNoiseCorrector::DiscoverComparatorPhase(float *psigs,int *c_avg_num,int n_comparators,int nframes)
{
  float phase_rms[2];
  int phase;

  for ( phase = 0;phase < 2;phase++ )
  {
    phase_rms[phase] = 0.0f;
    int rms_num = 0;

    for ( int i=0;i < n_comparators;i+=4 )
    {
      float *cptr_1a;
      float *cptr_1b;
      float *cptr_2a;
      float *cptr_2b;
      float rms_sum = 0.0f;

      // have to skip any columns that have all pinned pixels in any subset-average
//      if (( c_avg_num[i] == 0 ) && ( c_avg_num[i] == 1 ) && ( c_avg_num[i] == 2 ) && ( c_avg_num[i] == 3 ))
      if (( c_avg_num[i] == 0 ) && ( c_avg_num[i + 1] == 0 ) && ( c_avg_num[i + 2] == 0 ) && ( c_avg_num[i + 3] == 0 )){
//        fprintf (stdout, "Noisy column: %d; Comparator: %d.\n", i/4, i&0x3);
        continue;
      }

      // get a pointers to the comparator signals
      if ( phase==0 ) {
        cptr_1a = psigs + (i+2)*nframes;
        cptr_1b = psigs + (i+3)*nframes;
        cptr_2a = psigs + (i+0)*nframes;
        cptr_2b = psigs + (i+1)*nframes;
      }
      else
      {
        cptr_1a = psigs + (i+0)*nframes;
        cptr_1b = psigs + (i+3)*nframes;
        cptr_2a = psigs + (i+1)*nframes;
        cptr_2b = psigs + (i+2)*nframes;
      }

      for ( int frame=0;frame < nframes;frame++ )
      {
        rms_sum += (cptr_1a[frame]-cptr_1b[frame])*(cptr_1a[frame]-cptr_1b[frame]);
        rms_sum += (cptr_2a[frame]-cptr_2b[frame])*(cptr_2a[frame]-cptr_2b[frame]);
      }
      phase_rms[phase] += rms_sum;
      rms_num++;
    }

    //make them comparable between different runs
    if(rms_num != 0){
        phase_rms[phase] /= (2*rms_num);
    }
  }
  
  if (phase_rms[0] == 0 || phase_rms[1] == 0){
    return -1; //special tag to indicate case of 0 rms
  }

  if ( phase_rms[0] < phase_rms[1] )
    phase = 0;
  else
    phase = 1;

  //get phase_rms values to check how reliable it is
//  fprintf (stdout, "Phase: %d; RMS Phase Calcs = %f vs %f\n", phase, phase_rms[0], phase_rms[1]);

  // now combine signals according to the detected phase
  int cndx=0;
  for ( int i=0;i < n_comparators;i+=4 )
  {
    int ndx[4];
    float *cptr_1a;
    float *cptr_1b;
    float *cptr_2a;
    float *cptr_2b;
    int num_1a,num_1b,num_2a,num_2b;
    float *cptr_1;
    float *cptr_2;
    int num1;
    int num2;
    float scale1;
    float scale2;

    // get a pointers to the comparator signals
    if ( phase==0 ) {
      ndx[0] = i+2;
      ndx[1] = i+3;
      ndx[2] = i+0;
      ndx[3] = i+1;
    }
    else
    {
      ndx[0] = i+0;
      ndx[1] = i+3;
      ndx[2] = i+1;
      ndx[3] = i+2;
    }
    cptr_1a = psigs + ndx[0]*nframes;
    cptr_1b = psigs + ndx[1]*nframes;
    cptr_2a = psigs + ndx[2]*nframes;
    cptr_2b = psigs + ndx[3]*nframes;
    num_1a = c_avg_num[ndx[0]];
    num_1b = c_avg_num[ndx[1]];
    num_2a = c_avg_num[ndx[2]];
    num_2b = c_avg_num[ndx[3]];

    num1 = num_1a+num_1b;
    num2 = num_2a+num_2b;

    cptr_1 = psigs + (cndx+0)*nframes;
    cptr_2 = psigs + (cndx+1)*nframes;

    if ( num1 > 0 )
      scale1 = 1.0f/((float)num1);
    else
      scale1 = 0.0f;

    if ( num2 > 0 )
      scale2 = 1.0f/((float)num2);
    else
      scale2 = 0.0f;

    for ( int frame=0;frame < nframes;frame++ )
    {
      // beware...we are doing this in place...need to be careful
      float sum1 = scale1*(cptr_1a[frame]*num_1a+cptr_1b[frame]*num_1b);
      float sum2 = scale2*(cptr_2a[frame]*num_2a+cptr_2b[frame]*num_2b);

      cptr_1[frame] = sum1;
      cptr_2[frame] = sum2;
    }

    c_avg_num[cndx+0] = num1;
    c_avg_num[cndx+1] = num2;
    cndx+=2;
  }

  return phase;
}

// now neighbor-subtract the comparator signals
void ComparatorNoiseCorrector::NNSubtractComparatorSigs(float *pnn,float *psigs,int *mask,int span,int n_comparators,int nframes,float *hfnoise)
{
  float nn_avg[nframes];
  float zero_sig[nframes];
  
  memset(zero_sig,0,sizeof(zero_sig));

  for ( int i=0;i < n_comparators;i++ )
  {
    int nn_cnt=0;
    float *cptr;
    float *chfptr;
    float *nncptr;
    float centroid = 0.0f;
    int i_c0 = i & ~0x1;
    memset(nn_avg,0,sizeof(nn_avg));

	// in case we weren't provided with high frequency noise correction for NNs, use all zeros instead
	chfptr = zero_sig;
    
    // rounding down the starting point and adding one to the rhs properly centers
    // the neighbor average about the central column...except in cases where columns are
    // masked within the neighborhood.

    //same column but the other comparator
    int offset[2] = {1, 0};
    int theOtherCompInd = i_c0 + offset[i-i_c0];
    if(!mask[theOtherCompInd]){
        // get a pointer to the comparator signal
        cptr = psigs + theOtherCompInd *nframes;

        if (hfnoise != NULL)
            chfptr = hfnoise + theOtherCompInd *nframes;

        // add it to the average
        for ( int frame=0;frame < nframes;frame++ )
          nn_avg[frame] += cptr[frame] - chfptr[frame];

        nn_cnt++;
        centroid += theOtherCompInd;
    }

    for(int s = 1, cndx = 0; s <= span; s++){
      //i_c0 is even number
      //odd
      if(!(i_c0 - 2*s + 1 < 0  || i_c0 + 2*s + 1 >= n_comparators || mask[i_c0 - 2*s + 1] || mask[i_c0 + 2*s + 1])) {
          // get a pointer to the comparator signal
          cndx = i_c0 - 2*s + 1;
          cptr = psigs + cndx*nframes;

          if (hfnoise != NULL)
              chfptr = hfnoise + cndx*nframes;
          // add it to the average
          for ( int frame=0;frame < nframes;frame++ )
            nn_avg[frame] += cptr[frame] - chfptr[frame];
          nn_cnt++;
          centroid += cndx;
          cndx = i_c0 + 2*s + 1;
          cptr = psigs + cndx*nframes;

          if (hfnoise != NULL)
              chfptr = hfnoise + cndx*nframes;
          // add it to the average
          for ( int frame=0;frame < nframes;frame++ )
            nn_avg[frame] += cptr[frame] - chfptr[frame];
          nn_cnt++;
          centroid += cndx;
      }

      //even, symmetric
      if(!(i_c0 - 2*s < 0  || i_c0 + 2*s >= n_comparators || mask[i_c0 - 2*s] || mask[i_c0 + 2*s])) {
          cndx = i_c0 - 2*s;
          cptr = psigs + cndx*nframes;

          if (hfnoise != NULL)
              chfptr = hfnoise + cndx*nframes;
          // add it to the average
          for ( int frame=0;frame < nframes;frame++ )
            nn_avg[frame] += cptr[frame] - chfptr[frame];
          nn_cnt++;
          centroid += cndx;
          cndx = i_c0 + 2*s;
          cptr = psigs + cndx*nframes;

          if (hfnoise != NULL)
              chfptr = hfnoise + cndx*nframes;
          // add it to the average
          for ( int frame=0;frame < nframes;frame++ )
            nn_avg[frame] += cptr[frame] - chfptr[frame];
          nn_cnt++;
          centroid += cndx;
      }
    }

    if (( nn_cnt > 0 ) )
    {
      for ( int frame=0;frame < nframes;frame++ )
        nn_avg[frame] /= nn_cnt;

      // now subtract the neighbor average
      cptr = psigs + i*nframes;
      nncptr = pnn + i*nframes;
      for ( int frame=0;frame < nframes;frame++ )
        nncptr[frame] = cptr[frame] - nn_avg[frame];
    }
    else
    {
//      fprintf (stdout, "Default noise of 0 is set: %d\n", i);
      // not a good set of neighbors to use...just blank the correction
      // signal and do nothing.
      nncptr = pnn + i*nframes;
      for ( int frame=0;frame < nframes;frame++ )
        nncptr[frame] = 0.0f;
    }
  }
}

void ComparatorNoiseCorrector::HighPassFilter(float *pnn,int n_comparators,int nframes,int span)
{
  float trc_scratch[nframes];

  for ( int i=0;i < n_comparators;i++ )
  {
    float *cptr;

    // get a pointer to the comparator noise signal
    cptr = pnn + i*nframes;
    
    // make smooth version of noise signal
    for (int j=0;j < nframes;j++)
    {
        float sum = 0.0f;
        int cnt = 0;
        
        for (int k=(j-span);k <=(j+span);k++)
        {
            if ((k >= 0) && (k < nframes) && (k!=j))
            {
                cnt++;
                sum += cptr[k];
            }
        }
        
        trc_scratch[j] = sum/cnt;
    }
    
    // now subtract off the smoothed signal to eliminate low frequency
    // components, most of which are residual background effects that the
    // neighbor subtraction algorithm doesn't completely fitler out
    // this unfortunately does also somtimes eliminate some real comparator
    // noise...
    for (int j=0;j < nframes;j++)
        cptr[j] -= trc_scratch[j];
  }
}

// measure noise in the neighbor-subtracted signals
void ComparatorNoiseCorrector::CalcComparatorSigRMS(float *prms,float *pnn,int n_comparators,int nframes)
{
  for ( int i=0;i < n_comparators;i++ )
  {
    float *cptr;
    float rms_sum = 0.0f;

    // get a pointer to the comparator signal
    cptr = pnn + i*nframes;

    // add it to the average
    for ( int frame=0;frame < nframes;frame++ )
      rms_sum += cptr[frame]*cptr[frame];

    prms[i] = sqrt(rms_sum/nframes);
//    fprintf (stdout, "RMS of Comparator %d: %f\n", i, prms[i]);
  }
}

// find the noisiest 10%
void ComparatorNoiseCorrector::MaskAbove90thPercentile(int *mask,float *prms,int n_comparators)
{
  float rms_sort[n_comparators];
  int i;

  memcpy(rms_sort,prms,sizeof(rms_sort));

  // sort the top 10%
  for ( i=0;i < (n_comparators/10);i++ )
  {
    for ( int j=i;j < n_comparators;j++ )
    {
      if ( rms_sort[j] > rms_sort[i] )
      {
        float tmp = rms_sort[j];
        rms_sort[j] = rms_sort[i];
        rms_sort[i] = tmp;
      }
    }
  }

  float rms_thresh = rms_sort[i-1];

//  printf("**************************** comparator noise threshold = %f\n",rms_thresh);

  for ( i=0;i < n_comparators;i++ )
  {
    if ( prms[i] >= rms_thresh )
      mask[i] = 1;
  }
}

// find the noisiest 10%
void ComparatorNoiseCorrector::MaskIQR(int *mask,float *prms,int n_comparators, bool verbose)
{
  float rms_sort[n_comparators];
  int i;

  memcpy(rms_sort,prms,sizeof(rms_sort));

  std::sort(rms_sort, rms_sort+n_comparators);

  float rms_thresh = rms_sort[n_comparators * 3 / 4 - 1] + 2.5 * (rms_sort[n_comparators * 3/4 - 1] - rms_sort[n_comparators * 1/4 - 1]) ;

  int noisyCount = 0;
  if(verbose)
    fprintf (stdout, "Noisy comparators:");
  for ( i=0;i < n_comparators;i++ )
  {
    if ( prms[i] >= rms_thresh ){
      mask[i] = 1;
      noisyCount ++;
      if(verbose)
        fprintf (stdout, " %d", i);
    }
  }
  if(verbose)
    fprintf (stdout, "\n");
//  fprintf (stdout, "\n%d noisy comparators; threshold: %f\n", noisyCount, rms_thresh);
}

void ComparatorNoiseCorrector::MaskUsingDynamicStdCutoff(int *mask,float *prms,int n_comparators, float std_mult, bool verbose)
{
    float mean_rms;
    float std_rms;
    int cnt;
    float rms_threshold = 1E+20f;
    int i;
    
    for (int iter=0;iter < 2;iter++)
    {
        mean_rms = 0.0f;
        std_rms = 0.0f;
        cnt = 0;
        
        for (i=0;i < n_comparators;i++ )
        {
            if ((mask[i] == 0) && (prms[i] < rms_threshold))
            {
                mean_rms += prms[i];
                cnt++;
            }
        }

        // if not enough to analyze...just bail
        if (cnt < 10)
            return;

        mean_rms /= cnt;

        for (i=0;i < n_comparators;i++ )
        {
            if ((mask[i] == 0) && (prms[i] < rms_threshold))
                std_rms += (prms[i]-mean_rms)*(prms[i]-mean_rms);
        }
        
        std_rms = sqrt(std_rms/cnt);
        
        rms_threshold = mean_rms + std_mult*std_rms;
    }

//  printf("hf rms threshold = %f\n",rms_threshold);
    
  int noisyCount = 0;
  if(verbose)
    fprintf (stdout, "Noisy comparators:");
    
    // now set the mask according to the threshold
    for (i=0;i < n_comparators;i++ )
    {
        if ((mask[i] == 0) && (prms[i] >= rms_threshold))
        {
            mask[i] = 1;
            noisyCount ++;
            if(verbose)
              fprintf (stdout, " %d", i);
        }
    }
  if(verbose)
    fprintf (stdout, "\n");
}

// simple iterative formula that is good for getting the first principal component
void ComparatorNoiseCorrector::GetPrincComp(float *pcomp,float *pnn,int *mask,int n_comparators,int nframes)
{
	float ptmp[nframes];
	float ttmp[nframes];
	float residual;
	
	for (int i=0;i < nframes;i++)
	{
		ptmp[i] = rand();
		ttmp[i] = 0.0f;
	}
	
	memset(pcomp,0,sizeof(float)*nframes);
	
	residual = FLT_MAX;
	
	while(residual > 0.001)
	{
		memset(ttmp,0,sizeof(float)*nframes);

		for (int i=0;i < n_comparators;i++)
		{
			float sum=0.0f;
			float *cptr = pnn + i*nframes;
			
			if (mask[i] == 0)
			{
				for (int j=0;j < nframes;j++)
					sum += ptmp[j]*cptr[j];
				
				for (int j=0;j < nframes;j++)
					ttmp[j] += cptr[j]*sum;
			}
		}
				
		float tmag = 0.0f;
		for (int i=0;i < nframes;i++)
			tmag += ttmp[i]*ttmp[i];
		
		tmag = sqrt(tmag/nframes);

		for (int i=0;i < nframes;i++)
			pcomp[i] = ttmp[i]/tmag;
			
		residual = 0.0f;
		for (int i=0;i < nframes;i++)
			residual += (pcomp[i]-ptmp[i])*(pcomp[i]-ptmp[i]);
		residual = sqrt(residual/nframes);
		memcpy(ptmp,pcomp,sizeof(float)*nframes);
	}
}

void ComparatorNoiseCorrector::FilterUsingPrincComp(float *pnn,float *pcomp,int n_comparators,int nframes)
{
    float pdotp = 0.0f;
    for (int i=0;i < nframes;i++)
        pdotp += pcomp[i]*pcomp[i];
    
    for (int i=0;i < n_comparators;i++)
    {
        float sum=0.0f;
    	float *cptr = pnn + i*nframes;
        
        for (int j=0;j < nframes;j++)
            sum += cptr[j]*pcomp[j];
        
        float scale = sum/pdotp;
        
        for (int j=0;j < nframes;j++)
            cptr[j] -= pcomp[j]*scale;
    }
}




