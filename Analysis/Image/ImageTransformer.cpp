/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageTransformer.h"
#include "LSRowImageProcessor.h"
#include "Image.h"
#include "IonErr.h"
#include "deInterlace.h"
#include "Vecs.h"
#include <malloc.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>
#include <fstream>
#include <iostream>
#include "ChannelXTCorrection.h"
#include "ChipIdDecoder.h"
#include "RawImage.h"

//Initialize chipSubRegion in Image class
#define MAX_GAIN_CORRECT 16383
#define BF_PIXEL_GAIN_WINDOW 20
#define BF_PIXEL_MIN_GAIN 0.8f
#define BF_PIXEL_MAX_GAIN 1.35f
#define BF_PIXEL_DEFAULT_GAIN 1.0f
#define BF_PIXEL_START 4
#define BF_GAIN_ROW_WIDTH 3
#define BF_GAIN_COL_WIDTH 6
Region ImageCropping::chipSubRegion(0,0,0,0);
int ImageCropping::cropped_region_offset_x = 0;
int ImageCropping::cropped_region_offset_y = 0;


#define CHANXT_VEC_SIZE    8
#define CHANXT_VEC_SIZE_B 32

typedef float ChanVecf_t __attribute__ ((vector_size (CHANXT_VEC_SIZE_B)));
typedef union{
  ChanVecf_t V;
  float A[CHANXT_VEC_SIZE];
}ChanVecf_u;


// this is perhaps a prime candidate for something that is a json set of parameters to read in
// inversion vector + coordinates of the offsets
// we do only a single pass because the second-order correction is too small to notice

static float chan_xt_vect_316[DEFAULT_VECT_LEN]      = {0.0029,-0.0632,-0.0511,1.1114,0.0000,0.0000,0.0000};
static float *default_316_xt_vectors[] = {chan_xt_vect_316};

static float chan_xt_vect_318_even[DEFAULT_VECT_LEN] = {0.0132,-0.1511,-0.0131,1.1076,0.0404,0.0013,0.0018};
static float chan_xt_vect_318_odd[DEFAULT_VECT_LEN]  = {0.0356,-0.1787,-0.1732,1.3311,-0.0085,-0.0066,0.0001};
static float *default_318_xt_vectors[] = {chan_xt_vect_318_even, chan_xt_vect_318_even,chan_xt_vect_318_even, chan_xt_vect_318_even,
                                          chan_xt_vect_318_odd, chan_xt_vect_318_odd,chan_xt_vect_318_odd, chan_xt_vect_318_odd
                                         };
int ImageTransformer::chan_xt_column_offset[DEFAULT_VECT_LEN]   = {-12,-8,-4,0,4,8,12};
char ImageTransformer::PCATest[128]={0};

ChipXtVectArrayType ImageTransformer::default_chip_xt_vect_array[] = {
  {ChipId316, {default_316_xt_vectors, 1, DEFAULT_VECT_LEN, chan_xt_column_offset} },
  {ChipId318, {default_318_xt_vectors, 8, DEFAULT_VECT_LEN, chan_xt_column_offset} },
  {ChipIdUnknown, {NULL, 0,0,NULL} },
};

ChannelXTCorrectionDescriptor ImageTransformer::selected_chip_xt_vectors = {NULL, 0,0,NULL};

int ImageTransformer::dump_XTvects_to_file=1; // we'll flip this after the first time we write to disk what the vectors are

// XTChannelCorrect:
// For the 316 and 318, corrects cross-talk due to incomplete analog settling within the 316 and 318, and also
// residual uncorrected incomplete setting at the output of the devices.
// works along each row of the image and corrects cross talk that occurs
// within a single acquisition channel (every fourth pixel is the same channel on the 316/318)
// This method has no effect on the 314.
// The following NOTE is out-of-date, something like this may come back sometime
// NOTE:  A side-effect of this method is that the data for all pinned pixels will be replaced with
// the average of the surrounding neighbor pixels.  This helps limit the spread of invalid data in the pinned
// pixels to neighboring wells
// void Image::XTChannelCorrect (Mask *mask)
void ImageTransformer::XTChannelCorrect(RawImage *raw,
                                        const char *experimentName) {

  float **vects = NULL;
  int nvects = 0;
  int *col_offset = NULL;
  int vector_len;
  int frame, row, col, vn;
  short *pfrm, *prow;
  int i, lc;
  uint32_t vndx;

  // If no correction has been configured for (by a call to CalibrateChannelXTCorrection), the try to find the default
  // correction using the chip id as a guide.
  if (selected_chip_xt_vectors.xt_vector_ptrs == NULL)
    for (int nchip = 0;
         default_chip_xt_vect_array[nchip].id != ChipIdUnknown; nchip++)
      if (default_chip_xt_vect_array[nchip].id
          == ChipIdDecoder::GetGlobalChipId()) {
        memcpy(&selected_chip_xt_vectors,
               &(default_chip_xt_vect_array[nchip].descr),
               sizeof(selected_chip_xt_vectors));
        break;
      }

  // if the chip type is unsupported, silently return and do nothing
  if (selected_chip_xt_vectors.xt_vector_ptrs == NULL)
    return;

  vects = selected_chip_xt_vectors.xt_vector_ptrs;
  nvects = selected_chip_xt_vectors.num_vectors;
  col_offset = selected_chip_xt_vectors.vector_indicies;
  vector_len = selected_chip_xt_vectors.vector_len;

  // fill in pinned pixels with average of surrounding valid wells
  //BackgroundCorrect(mask, MaskPinned, (MaskType)(MaskAll & ~MaskPinned & ~MaskExclude),0,5,false,false,true);
  if ((raw->cols % 8) != 0) {
    short tmp[raw->cols];
    float *vect;
    int ndx;
    float sum;

    for (frame = 0; frame < raw->frames; frame++) {
      pfrm = &(raw->image[frame * raw->frameStride]);
      for (row = 0; row < raw->rows; row++) {
        prow = pfrm + row * raw->cols;
        for (col = 0; col < raw->cols; col++) {
          vndx = ((col + ImageCropping::cropped_region_offset_x)
                  % nvects);
          vect = vects[vndx];

          sum = 0.0;
          for (vn = 0; vn < vector_len; vn++) {
            ndx = col + col_offset[vn];
            if ((ndx >= 0) && (ndx < raw->cols))
              sum += prow[ndx] * vect[vn];
          }
          tmp[col] = (short) (sum);
        }
        // copy result back into the image
        memcpy(prow, tmp, sizeof(short[raw->cols]));
      }
    }

  } else {
    //#define XT_TEST_CODE
#ifdef XT_TEST_CODE
    short *tstImg = (short *)malloc(raw->rows*raw->cols*2*raw->frames);
    memcpy(tstImg,raw->image,raw->rows*raw->cols*2*raw->frames);
    {
      short tmp[raw->cols];
      float *vect;
      int ndx;
      float sum;

      for ( frame = 0;frame < raw->frames;frame++ ) {
        pfrm = & ( tstImg[frame*raw->frameStride] );
        for ( row = 0;row < raw->rows;row++ ) {
          prow = pfrm + row*raw->cols;
          for ( col = 0;col < raw->cols;col++ ) {
            vndx = ( ( col+ImageCropping::cropped_region_offset_x ) % nvects );
            vect = vects[vndx];

            sum = 0.0;
            for ( vn = 0;vn < vector_len;vn++ ) {
              ndx = col + col_offset[vn];
              if ( ( ndx >= 0 ) && ( ndx < raw->cols ) )
                sum += prow[ndx]*vect[vn];
            }
            tmp[col] = ( short ) ( sum );
          }
          // copy result back into the image
          memcpy ( prow,tmp,sizeof ( short[raw->cols] ) );
        }
      }
    }
#endif
    {
      ChanVecf_u vectsV[8][7];
      ChanVecf_u Avect[4], Svect[4];
      uint32_t j;

      for (vn = 0; vn < nvects; vn++) {
        for (i = 0; i < 8; i++) {
          if (i < vector_len)
            vectsV[vn][6].A[i] = vects[vn][i];
          else
            vectsV[vn][6].A[i] = 0;
        }
      }
      for (vn = 0; vn < nvects; vn++) {
        for (j = 0; j < 6; j++) {
          for (i = 0; i < 7; i++) {
            vectsV[vn][j].A[i] = vectsV[vn][6].A[(i + 7 - (j + 1))
                % 7];
          }
          vectsV[vn][j].A[7] = 0;
        }
      }

#ifdef XT_TEST_CODE
      static int doneOnce=0;
      if(!doneOnce)
      {
        doneOnce=1;
        for(vn=0;vn<nvects;vn++)
        {
          printf("vn=%d\n",vn);
          for(j=0;j<7;j++)
          {
            printf(" %d(%d) %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",vn,j,
                   vectsV[vn][j].A[0],vectsV[vn][j].A[1],vectsV[vn][j].A[2],
                vectsV[vn][j].A[3],vectsV[vn][j].A[4],vectsV[vn][j].A[5],
                vectsV[vn][j].A[6],vectsV[vn][j].A[7]);
          }
        }

      }
#endif
      for (frame = 0; frame < raw->frames; frame++) {
        pfrm = &(raw->image[frame * raw->frameStride]);
        for (row = 0; row < raw->rows; row++) {
          prow = pfrm + row * raw->cols;

          // prime the Avect values
          for (lc = 0; lc < 4; lc++) {
            Avect[lc].A[0] = 0; // -12
            Avect[lc].A[1] = 0; // -8
            Avect[lc].A[2] = 0; // -4
            Avect[lc].A[3] = prow[lc]; //  0
            Avect[lc].A[4] = prow[lc + 4]; //  4
            Avect[lc].A[5] = prow[lc + 8]; //  8
            Avect[lc].A[6] = 0; //  12
            Avect[lc].A[7] = 0;
          }
          for (col = 0, j = 6; col < raw->cols;
               col += 4, j = ((j + 1) % 7)) {

            // fill in the last values...
            if ((col + 16) <= raw->cols) {
              for (lc = 0; lc < 4; lc++)
                Avect[lc].A[j] = prow[col + 12 + lc];
            } else {
              for (lc = 0; lc < 4; lc++)
                Avect[lc].A[j] = 0.0f;
            }

            for (lc = 0; lc < 4; lc++) {
              Svect[lc].V =
                  Avect[lc].V
                  * vectsV[((col
                             + ImageCropping::cropped_region_offset_x)
                            + lc) % nvects][j].V; // apply the vector

              prow[col + lc] = Svect[lc].A[0] + Svect[lc].A[1]
                  + Svect[lc].A[2] + Svect[lc].A[3]
                  + Svect[lc].A[4] + Svect[lc].A[5]
                  + Svect[lc].A[6]/* + Svect[lc].A[7]*/;
            }
          }
        }
      }
    }

#ifdef XT_TEST_CODE
    {
      short *pTstFrm, *pTstRow;

      // test that we did the right thing...
      for (frame = 0; frame < raw->frames; frame++) {
        pfrm = &(raw->image[frame * raw->frameStride]);
        pTstFrm = &(tstImg[frame * raw->frameStride]);
        for (row = 0; row < raw->rows; row++) {
          prow = pfrm + row * raw->cols;
          pTstRow = pTstFrm + row * raw->cols;
          for (col = 0; col < raw->cols; col++) {
            if (pTstRow[col] > (prow[col]+1) ||
                pTstRow[col] < (prow[col]-1) )
              printf("%s: frame=%d row=%d col=%d   tst=%d img=%d\n",__FUNCTION__,frame,row,col,pTstRow[col],prow[col]);
          }
        }
      }
      free(tstImg);
    }
#endif
  }
  //Dump XT vectors to file
  if (dump_XTvects_to_file) {
    char xtfname[512];
    sprintf(xtfname, "%s/cross_talk_vectors.txt", experimentName);
    FILE* xtfile = fopen(xtfname, "wt");

    if (xtfile != NULL) {
      //write vector length and number of vectors on top
      fprintf(xtfile, "%d\t%d\n", vector_len, nvects);
      //write offsets in single line
      for (int nl = 0; nl < vector_len; nl++)
        fprintf(xtfile, "%d\t", col_offset[nl]);
      fprintf(xtfile, "\n");
      //write vectors tab-separated one line per vector
      for (int vndx = 0; vndx < nvects; vndx++) {
        for (int vn = 0; vn < vector_len; vn++)
          fprintf(xtfile, "%4.6f\t", vects[vndx][vn]);
        fprintf(xtfile, "\n");
      }
      fclose(xtfile);
    }
    dump_XTvects_to_file = 0;
  }
}

ChannelXTCorrection *ImageTransformer::custom_correction_data = NULL;


#define RETRY_INTERVAL 15 // 15 seconds wait time.
#define TOTAL_TIMEOUT 3600 // 1 hr before giving up.



// checks to see if the special lsrowimage.dat file exists in the experiment directory.  If it does,
// this image is used to generate custom channel correction coefficients.  If not, the method silently
// returns (and subsequent analysis uses the default correction).
void ImageTransformer::CalibrateChannelXTCorrection ( const char *exp_dir,const char *filename, bool wait_for_prerun )
{
  // only allow this to be done once
  if ( custom_correction_data != NULL )
    return;

  // LSRowImageProcessor can generate a correction for the 314, but application of the correction is much more
  // difficult than for 316/318, and the expected benefit is not as high, so for now...we're skipping the 314
  if ( !ChipIdDecoder::IsLargePGMChip() )
    return;

  int len = strlen ( exp_dir ) +strlen ( filename ) + 2;
  char full_fname[len];

  sprintf ( full_fname,"%s/%s",exp_dir,filename );

  if ( wait_for_prerun ) {
    std::string preRun = exp_dir;
    preRun = preRun + "/prerun_0000.dat";
    std::string acq0 = exp_dir;
    acq0 = acq0 + "/acq_0000.dat";

    uint32_t waitTime = RETRY_INTERVAL;
    int32_t timeOut = TOTAL_TIMEOUT;
    //--- Wait up to 3600 seconds for a file to be available
    bool okToProceed = false;
    while ( timeOut > 0 ) {
      //--- do our checkpoint files exist?
      if ( isFile ( preRun.c_str() ) || isFile ( acq0.c_str() ) ) {
        okToProceed = true;
        break;
      }
      fprintf ( stdout, "Waiting to load crosstalk params in %s\n",  full_fname );
      sleep ( waitTime );
      timeOut -= waitTime;
    }
    if ( !okToProceed ) {
      ION_ABORT ( "Couldn't find gateway files for: " + ToStr ( full_fname ) );
    }
    // We got the files we expected so if the xtalk file isn't there then warn.
    if ( !isFile ( full_fname ) ) {
      ION_WARN ( "Didn't find xtalk file: " + ToStr ( full_fname ) );
    }
  }
  LSRowImageProcessor lsrowproc;
  custom_correction_data = lsrowproc.GenerateCorrection ( full_fname );
  if ( custom_correction_data != NULL )
    selected_chip_xt_vectors = custom_correction_data->GetCorrectionDescriptor();

}



// gain correction

float *ImageTransformer::gain_correction = NULL;


void ImageTransformer::GainCorrectImage(RawImage *raw)
{
  if ((raw->cols % VEC8_SIZE) == 0)
  {
    int lw=raw->rows*raw->cols/VEC8_SIZE;
    v8f_u  val;
    short int *imagePtr=raw->image;
    v8f_u *gainPtr;
    int frame,idx;

    // the alligned case
    for (frame=0;frame < raw->frames;frame++)
    {
      gainPtr = (v8f_u *)gain_correction;
      for (idx = 0;idx < lw;idx++)
      {
        LD_VEC8S_CVT_VEC8F(imagePtr,val);

        val.V *= gainPtr->V;

#ifdef __AVX__
        v8f_u max;
        max.V=LD_VEC8F(MAX_GAIN_CORRECT);
        val.V = __builtin_ia32_minps256(val.V,max.V); // cap at MAX_GAIN_CORRECT
#else
        for(int k=0;k<VEC8_SIZE;k++)
        {
          if(val.A[k] > MAX_GAIN_CORRECT)
            val.A[k] = MAX_GAIN_CORRECT;
        }
#endif
        CVT_VEC8F_VEC8S((*(v8s_u *)imagePtr),val);

        imagePtr+=VEC8_SIZE;
        gainPtr++;
      }
    }
  }
  else
  {
    for (int row = 0;row < raw->rows;row++)
    {
      for (int col = 0;col < raw->cols;col++)
      {
        float gain = gain_correction[row*raw->cols + col];
        short *prow = raw->image + row*raw->cols + col;

        for (int frame=0;frame < raw->frames;frame++)
        {
          float val = *(prow+frame*raw->frameStride);
          val *= gain;
          if (val > MAX_GAIN_CORRECT) val = MAX_GAIN_CORRECT;

          *(prow+frame*raw->frameStride) = (short)(val);
        }
      }
    }
  }

}


float CalculatePixelGain(float *my_trc,float *reference_trc,int min_val_frame, int raw_frames)
{
  float asq = 0.0;
  float axb = 0.0;
  float gain = 1.0f;


  for (int i=0;i < raw_frames;i++)
  {
    if (((i >= BF_PIXEL_START) && (i < min_val_frame - BF_PIXEL_GAIN_WINDOW)) | (i > min_val_frame))
      continue;

    asq += reference_trc[i]*reference_trc[i];
    axb += my_trc[i]*reference_trc[i];
  }
  if (axb != 0.0f )
    gain = asq/axb;

  // cap gain to reasonable values
  if (gain < BF_PIXEL_MIN_GAIN)
    gain = BF_PIXEL_MIN_GAIN;

  if (gain > BF_PIXEL_MAX_GAIN)
    gain = BF_PIXEL_MAX_GAIN;

  return gain;
}


// uses a beadfind flow to compute the gain of each pixel
// this can be used as a correction for all future images
void ImageTransformer::GainCalculationFromBeadfind(Mask *mask, RawImage *raw)
{
  //  (void)mask;
  float avg_trc[raw->frames];
  float my_trc[raw->frames];

  if (gain_correction == NULL)
    gain_correction = (float *)memalign(VEC8F_SIZE_B,sizeof(float)*raw->rows*raw->cols);

  if ((raw->imageState & IMAGESTATE_GainCorrected) != 0)
  { // disable gain correction
    for(int i=0;i<raw->rows*raw->cols;i++)
      gain_correction[i]=BF_PIXEL_DEFAULT_GAIN;
    return;
  }

  for (int row = 0;row < raw->rows;row++)
  {
    for (int col = 0;col < raw->cols;col++)
    {
      // don't bother calculating for pinned pixels
      if (mask->Match (col,row,(MaskType) (MaskPinned | MaskExclude | MaskIgnore)))
      {
        gain_correction[row*raw->cols + col] = 1.0f;
        continue;
      }

      int min_val_frame = 0;
      int min_val = 32000;

      bool no_neighbors = true;
      for (int frame = 0;frame < raw->frames;frame++)
      {
        float nnavg = 0.0;
        int avgsum = 0;
        for (int nr=row-3;nr <= row+3;nr++)
        {
          if ((nr >= 0) && (nr < raw->rows))
          {
            short *prow = raw->image + nr*raw->cols + frame*raw->frameStride;
            for (int nc=col-6;nc <= col+6;nc++)
            {
              // skip the column containing the bead being measured
              // as there is significant common-mode gain non-uniformity
              // within the column
              if ((nc >= 0) && (nc < raw->cols) && (nc != col))
              {
                if (!mask->Match (nc,nr,(MaskType) (MaskPinned | MaskExclude | MaskIgnore)))
                {
                  nnavg += prow[nc];
                  avgsum++;
                }
              }
            }

            if (nr == row)
            {
              my_trc[frame] = prow[col];
              if (prow[col] < min_val)
              {
                min_val = prow[col];
                min_val_frame = frame;
              }
            }
          }
        }
        if (avgsum>0)
        {
          avg_trc[frame] = nnavg / avgsum;
          no_neighbors = false;
        }
      }
      if (!no_neighbors)
        gain_correction[row*raw->cols + col] = CalculatePixelGain(my_trc,avg_trc,min_val_frame,raw->frames);
      else
        gain_correction[row*raw->cols + col] = BF_PIXEL_DEFAULT_GAIN;
    }
  }
}

// uses a beadfind flow to compute the gain of each pixel
// this can be used as a correction for all future images
// optimized version of above function
void ImageTransformer::GainCalculationFromBeadfindFaster(Mask *mask, RawImage *raw)
{
  fprintf(stdout, "gain_range: %.3f %.3f\n", BF_PIXEL_MIN_GAIN, BF_PIXEL_MAX_GAIN);
  if (gain_correction == NULL)
    gain_correction = (float *)memalign(VEC8F_SIZE_B,sizeof(float)*raw->rows*raw->cols);

  /* just set gain to 1.0f effectively disabling gain. */
  if ((raw->imageState & IMAGESTATE_GainCorrected) != 0)
  { // disable gain correction
    float *__restrict g_start = gain_correction;
    float *__restrict g_end = g_start + raw->frameStride;
    while (g_start != g_end) {
      *g_start++ = BF_PIXEL_DEFAULT_GAIN;
    }
    return;
  }

  /* Data cube for cumulative sum for calculating averages fast. 
     Note the padding by 1 row and colum for easy code flow */
  int64_t *__restrict cum_sum = (int64_t *)memalign(VEC8F_SIZE_B, sizeof(int64_t) * (size_t)(raw->cols +1) * (raw->rows + 1) * raw->frames);
  assert(cum_sum);
  size_t cum_sum_size = (size_t)(raw->cols +1) * (raw->rows + 1) * raw->frames;
  memset(cum_sum, 0, sizeof(int64_t) * cum_sum_size); // zero out
  int cs_frame_stride = (raw->cols + 1) * (raw->rows + 1);

  /* Mask of the cumulative number of good wells so we know denominator of average also padded. */
  int *__restrict num_good_wells = (int *) memalign(VEC8F_SIZE_B, sizeof(int) * cs_frame_stride);
  assert(num_good_wells);
  memset(num_good_wells, 0, sizeof(int) * cs_frame_stride);

  /* Data cube for our averages */
  float *__restrict nn_avg = (float *) memalign(VEC8F_SIZE_B, sizeof(float) * (size_t) raw->frameStride * raw->frames);
  assert(nn_avg);

  /* Summary statistics for regression */
  float *__restrict sum_stats = (float *) memalign(VEC8F_SIZE_B, sizeof(float) * 2 * raw->frameStride);
  assert(sum_stats);

  /* Matrix for minimum values */
  float *__restrict trace_min = (float *) memalign(VEC8F_SIZE_B, sizeof(float) * (size_t) raw->frameStride);
  assert(trace_min);
  float *__restrict tm_start = trace_min;
  float *__restrict tm_end= trace_min + raw->frameStride;
  float max_value = std::numeric_limits<float>::max();
  while(tm_start != tm_end) {
    *tm_start++ = max_value;
  }

  /* Matrix for minimum frame */
  int *__restrict trace_min_frame = (int *) memalign(VEC8F_SIZE_B, sizeof(int) * (size_t) raw->frameStride);
  assert(trace_min_frame);
  memset(trace_min, 0, sizeof(int) * (size_t) raw->frameStride);

  /*
   Algorithmic trick - Instead of recalculating the average for each well we're going
   to calculate the cumulative sum of every sub matrix and then use the difference
   in the cumulative sums to get sum for a region we want to average over. 
   
   Let M(x,y) be our original frame matrix and C(x,y) be the cumulative sum matrix
   then C(x,y) = M(x,y) + C(x-1,y) + C(x,y-1) - C(x-1,y-1)

   Original matrix  Cumulative Sum matrix
   9  10  11  12    15  33  54  78
   5   6   7   8     6  14  24  36
   1   2   3   4     1   3   6  10

   Then once we want the average for a region, say x = 1..2 and y = 1..2
   Avg({1:2},{1:2}) = C(2,2) - C(0,2) - C(2,0) + C(0,0)
                    = 54 - 6 - 15 + 1
                    = 34
   Which does equal 10 + 11 + 6 + 7 = 34 from the original matrix without having to iterate
   over and over to get the average for each region.

   Only additional issue is that we need to zero out pinned and bad wells and thus
   keep track of how many wells are actually used in a region for the cumulative sum.
   we'll do the same trick now just keeping track of good wells in a submatrix

   csugnet
  */
  /* Calculate the cumulative sum of images for each frame. */
  enum MaskType mask_ignore = (MaskType) (MaskPinned | MaskExclude | MaskIgnore);
  int64_t *__restrict cs_cur, *__restrict cs_prev; 
  const unsigned short *__restrict p_mask = mask->GetMask();
  for (int fIx = 0; fIx < raw->frames; fIx++) {
    for (int rowIx = 0; rowIx < raw->rows; rowIx++) {
      short *__restrict col_ptr = &raw->image[fIx * raw->frameStride + rowIx * raw->cols];
      short *__restrict col_ptr_end = col_ptr + raw->cols;
      const unsigned short *__restrict c_mask = p_mask + rowIx * raw->cols;
      cs_prev = cum_sum + (fIx * cs_frame_stride + rowIx * (raw->cols+1)); // +1 due to padding
      cs_cur = cs_prev + raw->cols + 1; // pointing at zero so needs to be incremented before assignment
      int64_t value;
      while(col_ptr != col_ptr_end) {
        value = *col_ptr++;
        if ((*c_mask++ & mask_ignore) != 0) { value = 0.0f; }
        value -= *cs_prev++;
        value += *cs_cur++ + *cs_prev;
        *cs_cur = value;
      }
    }
  }

  /* Calculate the cumulative sum of good wells similar to above. */
  for (int rowIx = 0; rowIx < raw->rows; rowIx++) {
    const unsigned short *__restrict c_mask = p_mask + rowIx * raw->cols;
    int *__restrict g_prev = num_good_wells + (rowIx * (raw->cols+1));
    int *__restrict g_cur = g_prev + raw->cols + 1;
    for (int colIx = 0; colIx < raw->cols; colIx++) {
      int good = 1;
      if ((*c_mask++ & mask_ignore) != 0) { good = 0; }
      int x =  *g_cur++ + good - *g_prev++;
      x += *g_prev;
      *g_cur = x;
    }
  }
  
  /* Go through each frame and calculate the avg for each well. */
  for (int fIx = 0; fIx < raw->frames; fIx++) {
    int64_t *__restrict cum_sum_frame = cum_sum + (fIx * cs_frame_stride);
    float *__restrict nn_avg_frame = nn_avg + (fIx * raw->frameStride);
    float *__restrict trace_min_cur = trace_min;
    short *__restrict img_cur = raw->image + fIx * raw->frameStride;
    int *__restrict trace_min_frame_cur = trace_min_frame;
    int cs_col_size = raw->cols + 1;
    int q1, q2,q3,q4;
    for (int rowIx = 0; rowIx < raw->rows; rowIx++) {
      for (int colIx = 0; colIx < raw->cols; colIx++) {
        int r_start = std::max(-1,rowIx - 4); // -1 ok for calc of edge conditions
        int r_end = std::min(rowIx+3, raw->rows-1);
        int c_start = std::max(-1, colIx - 7); // -1 ok for calc of edge conditions
        int c_end = colIx-1;
        int64_t sum1 = 0.0f, sum2 = 0.0f;
        int count1 = 0, count2 = 0;

        /* average for wells in columns left of our well. */
        if (c_end >= 0) {
          q1 = (r_end + 1) * cs_col_size + c_end + 1;
          q2 = (r_start +1) * cs_col_size + c_end + 1;
          q3 = (r_end+1) * cs_col_size + c_start + 1;
          q4 = (r_start+1) * cs_col_size + c_start + 1;
          count1 = num_good_wells[q1] - num_good_wells[q2] - num_good_wells[q3] + num_good_wells[q4];
          sum1 = cum_sum_frame[q1] - cum_sum_frame[q2] - cum_sum_frame[q3] + cum_sum_frame[q4];
        }

        /* average for wells in column right of our well. */
        c_start = colIx;
        c_end = std::min(raw->cols-1,colIx + 6);
        if (c_start < raw->cols) {
          q1 = (r_end + 1) * cs_col_size + c_end+1;
          q2 = (r_start +1) * cs_col_size + c_end + 1;
          q3 = (r_end+1) * cs_col_size + c_start + 1;
          q4 = (r_start+1) * cs_col_size + c_start + 1;
          count2 = num_good_wells[q1] - num_good_wells[q2] - num_good_wells[q3] + num_good_wells[q4];
          sum2 = cum_sum_frame[q1] - cum_sum_frame[q2] - cum_sum_frame[q3] + cum_sum_frame[q4];
        }

        /* combine them if we got some good wells */
        if (count1 + count2 > 0) {
          *nn_avg_frame = ((float)sum1 + sum2)/(count1 + count2);
          if (*trace_min_cur > *img_cur) {
            *trace_min_cur = *img_cur;
            *trace_min_frame_cur = fIx;
          }
        }
        else {
          *nn_avg_frame = 0.0f;
        }
        img_cur++;
        nn_avg_frame++;
        trace_min_cur++;
        trace_min_frame_cur++;
      }
    }
  }

  /* set up our summary statistics. */
  memset(sum_stats, 0, sizeof(float) * raw->frameStride * 2);
  float *__restrict xx = sum_stats;
  float *__restrict xy = xx + raw->frameStride;

  /* Calculate the summary statistics for regression fit */
  for (int fIx = 0; fIx < raw->frames; fIx++) {
    float * __restrict frame_nn_start = nn_avg + fIx * raw->frameStride;
    float * __restrict frame_nn_end = frame_nn_start + raw->frameStride;
    short * __restrict image_frame = raw->image + fIx * raw->frameStride;
    int *__restrict min_frame = trace_min_frame;
    float *__restrict xx_start = xx;
    float *__restrict xy_start = xy;
    const unsigned short *__restrict mask_start = mask->GetMask();
    while (frame_nn_start != frame_nn_end) {
      /* the window we fit on comes from the historical function, not sure of the reasoning. */
      if ((fIx < BF_PIXEL_START || ((fIx >= *min_frame - BF_PIXEL_GAIN_WINDOW) && fIx <= *min_frame)) && ( (*mask_start & mask_ignore) == 0)) {
        *xx_start += *frame_nn_start * *frame_nn_start;
        *xy_start += *frame_nn_start * *image_frame;
      }
      image_frame++;
      xx_start++;
      xy_start++;
      min_frame++;
      mask_start++;
      frame_nn_start++;
    }
  }

  /* Calculate final version of gain. */
  float *__restrict xx_start = xx;
  float *__restrict xx_end = xx + raw->frameStride;
  float *__restrict xy_start = xy;
  const unsigned short *__restrict mask_start = mask->GetMask();
  float *__restrict gain_start = gain_correction;
  int num_pin_low_gain = 0, num_pin_high_gain = 0;
  while(xx_start != xx_end) {
    if ((*mask_start & mask_ignore) == 0 && *xy_start != 0.0f) {
      *gain_start = *xx_start / *xy_start;
      if (*gain_start < BF_PIXEL_MIN_GAIN) {
        *gain_start =  BF_PIXEL_MIN_GAIN;
        num_pin_low_gain++;
      }
      if (*gain_start > BF_PIXEL_MAX_GAIN) {
        *gain_start = BF_PIXEL_MAX_GAIN;
        num_pin_high_gain++;
      }
    }
    else {
      *gain_start = 1.0;
    }
    gain_start++;
    mask_start++;
    xx_start++;
    xy_start++;
  }
  fprintf(stdout, "Gain threshold low %d (%.3f) high %d (%.3f)\n", 
          num_pin_low_gain, (float)num_pin_low_gain/raw->frameStride,
          num_pin_high_gain, (float)num_pin_high_gain/raw->frameStride);
  free(cum_sum);
  free(num_good_wells);
  free(nn_avg);
  free(sum_stats);
  free(trace_min);
  free(trace_min_frame);

}

// uses a beadfind flow to compute the gain of each pixel
// this can be used as a correction for all future images
// optimized version of above function
void ImageTransformer::GainCalculationFromBeadfindFasterSave(RawImage *raw, Mask *mask, char *bad_wells, 
                                                             int row_step, int col_step, ImageNNAvg *imageNN) {

  fprintf(stdout, "gain_range: %.3f %.3f\n", BF_PIXEL_MIN_GAIN, BF_PIXEL_MAX_GAIN);
  if (gain_correction == NULL)
    gain_correction = (float *)memalign(VEC8F_SIZE_B,sizeof(float)*raw->rows*raw->cols);

  /* just set gain to 1.0f effectively disabling gain. */
  if ((raw->imageState & IMAGESTATE_GainCorrected) != 0)
  { // disable gain correction
    float *__restrict g_start = gain_correction;
    float *__restrict g_end = g_start + raw->frameStride;
    while (g_start != g_end) {
      *g_start++ = BF_PIXEL_DEFAULT_GAIN;
    }
    return;
  }

  imageNN->Init(raw->rows, raw->cols, raw->frames);
  imageNN->CalcCumulativeSum(raw->image, mask, bad_wells);

  /* Summary statistics for regression */
  float *__restrict sum_stats = (float *) memalign(VEC8F_SIZE_B, sizeof(float) * 2 * raw->frameStride);
  assert(sum_stats);

  /* Matrix for minimum values */
  float *__restrict trace_min = (float *) memalign(VEC8F_SIZE_B, sizeof(float) * (size_t) raw->frameStride);
  assert(trace_min);
  float *__restrict tm_start = trace_min;
  float *__restrict tm_end= trace_min + raw->frameStride;
  float max_value = std::numeric_limits<float>::max();
  while(tm_start != tm_end) {
    *tm_start++ = max_value;
  }

  /* Matrix for minimum frame */
  int *__restrict trace_min_frame = (int *) memalign(VEC8F_SIZE_B, sizeof(int) * raw->frameStride);
  assert(trace_min_frame);
  memset(trace_min_frame, 0, sizeof(int) * raw->frameStride);

  imageNN->CalcNNAvgAndMinFrame(row_step, col_step, BF_GAIN_ROW_WIDTH, BF_GAIN_COL_WIDTH,
                                trace_min, trace_min_frame,
                                raw->image, false);
  const float *__restrict nn_avg = imageNN->GetNNAvgImagePtr();
  /* set up our summary statistics. */
  memset(sum_stats, 0, sizeof(float) * raw->frameStride * 2);
  float *__restrict xx = sum_stats;
  float *__restrict xy = xx + raw->frameStride;
  enum MaskType mask_ignore = (MaskType) (MaskPinned | MaskExclude | MaskIgnore);
  /* Calculate the summary statistics for regression fit */
  for (int fIx = 0; fIx < raw->frames; fIx++) {
    const float * __restrict frame_nn_start = nn_avg + fIx * raw->frameStride;
    const float * __restrict frame_nn_end = frame_nn_start + raw->frameStride;
    short * __restrict image_frame = raw->image + fIx * raw->frameStride;
    int *__restrict min_frame = trace_min_frame;
    float *__restrict xx_start = xx;
    float *__restrict xy_start = xy;
    const unsigned short *__restrict mask_start = mask->GetMask();
    while (frame_nn_start != frame_nn_end) {
      /* the window we fit on comes from the historical function, not sure of the reasoning. */
      if ((fIx < BF_PIXEL_START || ((fIx >= *min_frame - BF_PIXEL_GAIN_WINDOW) && fIx <= *min_frame)) && ( (*mask_start & mask_ignore) == 0)) {
        *xx_start += *frame_nn_start * *frame_nn_start;
        *xy_start += *frame_nn_start * *image_frame;
      }
      image_frame++;
      xx_start++;
      xy_start++;
      min_frame++;
      mask_start++;
      frame_nn_start++;
    }
  }

  /* Calculate final version of gain. */
  float *__restrict xx_start = xx;
  float *__restrict xx_end = xx + raw->frameStride;
  float *__restrict xy_start = xy;
  const unsigned short *__restrict mask_start = mask->GetMask();
  float *__restrict gain_start = gain_correction;
  int num_pin_low_gain = 0, num_pin_high_gain = 0;
  while(xx_start != xx_end) {
    if ((*mask_start & mask_ignore) == 0 && *xy_start != 0.0f) {
      *gain_start = *xx_start / *xy_start;
      if (*gain_start < BF_PIXEL_MIN_GAIN) {
        *gain_start =  BF_PIXEL_MIN_GAIN;
        num_pin_low_gain++;
      }
      if (*gain_start > BF_PIXEL_MAX_GAIN) {
        *gain_start = BF_PIXEL_MAX_GAIN;
        num_pin_high_gain++;
      }
    }
    else {
      *gain_start = 1.0;
    }
    gain_start++;
    mask_start++;
    xx_start++;
    xy_start++;
  }
  fprintf(stdout, "Gain threshold low %d (%.3f) high %d (%.3f)\n", 
          num_pin_low_gain, (float)num_pin_low_gain/raw->frameStride,
          num_pin_high_gain, (float)num_pin_high_gain/raw->frameStride);
  free(sum_stats);
  free(trace_min);
  free(trace_min_frame);
}

//@TODO:  Bad to have side effects on every image load from now on
// should be >explicit< pass of gain correction information to image loader
void ImageTransformer::CalculateGainCorrectionFromBeadfindFlow (char *_datDir, bool gain_debug_output)
{
  // calculate gain of each well from the beadfind and use for correction of all images thereafter
  Mask *gainCalcMask;
  std::string datdir = _datDir;
  std::string preBeadFind = datdir + "/beadfind_pre_0003.dat";
  Image bfImg;
  bfImg.SetImgLoadImmediate (false);
  bool loaded = bfImg.LoadRaw (preBeadFind.c_str());
  if (!loaded)
  {
    ION_ABORT ("*Error* - No beadfind file found, did beadfind run? are files transferred?  (" + preBeadFind + ")");
  }


  gainCalcMask = new Mask (bfImg.GetCols(),bfImg.GetRows());

  bfImg.FilterForPinned (gainCalcMask, MaskEmpty, false);
  bfImg.SetMeanOfFramesToZero (1,3);
  CalculateGainCorrectionFromBeadfindFlow(gain_debug_output, bfImg, *gainCalcMask);

  bfImg.Close();
  delete gainCalcMask;
}

void ImageTransformer::CalculateGainCorrectionFromBeadfindFlow (bool gain_debug_output, Image &bfImg, Mask &mask)
{
  // calculate gain of each well from the beadfind and use for correction of all images thereafter


  //@TODO:  implicit global variable->explicit global variable-> explicit variable
  //  GainCalculationFromBeadfind (gainCalcMask,bfImg.raw);
  GainCalculationFromBeadfindFaster (&mask,bfImg.raw);
  printf ("Computed gain for each pixel using beadind image\n");
  if (gain_debug_output) {
    DumpTextGain(bfImg.GetCols(),bfImg.GetRows());
  }
}


void ImageTransformer::DumpTextGain(int _image_cols, int _image_rows)
{
  FILE *gainfile = fopen ("gainimg.txt","w");

  for (int row=0;row < _image_rows;row++)
  {
    int col;

    for (col=0;col < (_image_cols-1); col++)
      fprintf (gainfile,"%7.5f\t",getPixelGain (row,col,_image_cols));

    fprintf (gainfile,"%7.5f\n",getPixelGain (row,col,_image_cols));
  }
  fclose (gainfile);
}


#ifndef BB_DC
struct lsrow_header {
    uint32_t magic;
    uint32_t version;
    uint32_t rows;
    uint32_t cols;
};
#define LSROWIMAGE_MAGIC_VALUE    0xFF115E3A
#endif

// load Gain.lsr
bool ImageTransformer::ReadDataCollectGainCorrection(
  const std::string &dataPath,
  unsigned int rows,
  unsigned int cols) {

  size_t dataSize = sizeof(float)*rows*cols; 
  if (gain_correction == NULL)
    gain_correction = (float*)malloc(dataSize);

  ifstream gainFile;
  gainFile.open(dataPath.c_str(),ios::in | ios::binary);

  if (!gainFile.is_open()) {
    std::cout << "DataCollect gain: unable to open " << dataPath.c_str() << std::endl;
  }
  else {
    struct lsrow_header hdr;
    gainFile.read((char*)&hdr,sizeof(lsrow_header));

    if (!gainFile.good()) {
      std::cout << "DataCollect gain: header error in " << dataPath.c_str() << std::endl;
    }   
    else if (hdr.magic != LSROWIMAGE_MAGIC_VALUE){
      std::cout << "DataCollect gain: header error in " << dataPath.c_str() << std::endl;
    }
    else if (hdr.rows != rows || hdr.cols != cols){
      std::cout << "DataCollect gain: unexpected dimensions rows:" << hdr.rows << " & cols:" <<  hdr.cols << std::endl;
    }
    else{
      std::cout << "DataCollect gain: loading rows:" << hdr.rows << " & cols:" << hdr.cols << std::endl;

	  gainFile.read((char*)gain_correction, dataSize);

	  if (gainFile.good()) {

 	    std::cout << "DataCollect gain: successfully loaded " << dataPath.c_str() << std::endl;

 	    gainFile.close();
	    return true;
	  }

	}

    gainFile.close();
  }


  std::cout << "DataCollect gain: loading errors, using 1.0 gain for all wells" << std::endl; 
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      gain_correction[r*cols + c] = 1.0f;
    }
  }

  return false;
}
