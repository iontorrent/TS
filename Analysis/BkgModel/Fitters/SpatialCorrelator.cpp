/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SpatialCorrelator.h"
#include "DNTPRiseModel.h"
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <float.h>
#include <vector>
#include <assert.h>
#include "LinuxCompat.h"
#include "SignalProcessingMasterFitter.h"
#include "RawWells.h"
#include "MathOptim.h"
#include "mixed.h"
#include "BkgDataPointers.h"


#define NN_SPAN_X (2)
#define NN_SPAN_Y (2)

float nn_even_col_map_defaultX[] = {
  0.000,0.000,0.010,0.000,0.000,
  0.015,0.031,0.131,0.026,0.015,
  0.026,0.127,0.000,0.111,0.025,
  0.015,0.117,0.133,0.110,0.013,
  0.000,0.021,0.010,0.020,0.000,
};

float nn_odd_col_map_defaultX[] = {
  0.000,0.031,0.010,0.026,0.000,
  0.015,0.127,0.131,0.111,0.015,
  0.026,0.117,0.000,0.110,0.025,
  0.015,0.021,0.133,0.020,0.013,
  0.000,0.000,0.010,0.000,0.000,
};

#define MEASURE_SPAN_X  (2)
#define MEASURE_SPAN_Y  (2)


SpatialCorrelator::SpatialCorrelator (SignalProcessingMasterFitter &_bkg) :
    bkg (_bkg)
{
  // do nothing special

  nn_odd_col_map = NULL;
  nn_even_col_map = NULL;
  avg_corr = 0.0f;

  region = NULL;

  Defaults();

}

void SpatialCorrelator::Defaults()
{
  region = bkg.region_data->region;
  
  nn_odd_col_map = new float[25];
  nn_even_col_map = new float[25];
  avg_corr = -0.1f;

  memcpy(nn_odd_col_map,nn_odd_col_map_defaultX,sizeof(float[25]));
  memcpy(nn_even_col_map,nn_even_col_map_defaultX,sizeof(float[25]));
}

SpatialCorrelator::~SpatialCorrelator()
{
  delete[] nn_odd_col_map;
  delete[] nn_even_col_map;
}


int EvenPutIndexMap[] = {
  -1, -1,  7, -1, -1,
   0,  3,  8, 11, 15,
   1,  4, -1, 12, 16,
   2,  5,  9, 13, 17,
  -1,  6, 10, 14, -1,
};

int OddPutIndexMap[] = {
  -1,  3,  7, 11, -1,
   0,  4,  8, 12, 15,
   1,  5, -1, 13, 16,
   2,  6,  9, 14, 17,
  -1, -1, 10, -1, -1,
};

void SpatialCorrelator::MeasureConvolution(int *prev_same_nuc_tbl,int *next_same_nuc_tbl)
{
  arma::Mat<double> lhs;
  arma::Mat<double> rhs;
  arma::Mat<double> vect;
  arma::Mat<double> lhs_build;
  arma::Mat<double> ccoeff;
  int mat_size = 19;
  float *ampl_map;
  double *result;
  int prev,next;


  result = new double[(MEASURE_SPAN_X*2+1)*(MEASURE_SPAN_Y*2+1)];
  memset(result,0,sizeof(double[(MEASURE_SPAN_X*2+1)*(MEASURE_SPAN_Y*2+1)]));

  ampl_map = new float[region->w*region->h];

  memset(ampl_map,0,sizeof(float[region->w*region->h]));

  lhs.zeros(mat_size,mat_size);
  rhs.zeros(mat_size,1);
  vect.zeros(1,mat_size);

//int ntest = 2000;

  for (int fnum=0;fnum < NUMFB;fnum++)
  {
    float flow_avg = 0.0f;
    prev = prev_same_nuc_tbl[fnum];
    next = next_same_nuc_tbl[fnum];

    flow_avg = MakeSignalMap(ampl_map,fnum); // warning: are we normalizing by number of beads or total area?

    // find wells that are good 0-mers in this flow
    for (int ibd=0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
    {
      if ((bkg.region_data->my_beads.params_nn[ibd].Ampl[fnum] < 0.5) &&
          (bkg.region_data->my_beads.params_nn[ibd].Ampl[prev] < 0.5) &&
          (bkg.region_data->my_beads.params_nn[ibd].Ampl[next] < 0.5))
      {
        int row,col;
        row = bkg.region_data->my_beads.params_nn[ibd].y;
        col = bkg.region_data->my_beads.params_nn[ibd].x;

        if ((row - MEASURE_SPAN_Y)>=0  && (row + MEASURE_SPAN_Y)<bkg.region_data->region->h &&
            (col - MEASURE_SPAN_X)>=0  && (col + MEASURE_SPAN_X)<bkg.region_data->region->w)
        {
          int *valmap;
          if (row&0x1)
            valmap = OddPutIndexMap;
          else
            valmap = EvenPutIndexMap;

          int valmapndx = 0;
          for (int dr=-MEASURE_SPAN_Y;dr <= MEASURE_SPAN_Y;dr++)
          {
            for (int dc=-MEASURE_SPAN_X;dc <= MEASURE_SPAN_X;dc++)
            {
              int ndx = valmap[valmapndx];

              if (ndx != -1)
                vect.at(ndx) = ampl_map[(row+dr)*region->w+col+dc];

              valmapndx++;
            }
          }

          vect.at(mat_size-1) = flow_avg;
          lhs_build = trans(vect) * vect;
          lhs = lhs + lhs_build;
          rhs = rhs + trans(vect) * ampl_map[row*region->w+col];

//          if (ntest == 0)
//          {
//            vect.print("vect =");
//            lhs_build.print("lhs_build = ");
//            lhs.print("lhs = ");
//            rhs.print("rhs = ");
//            exit(-1);
//          }
//          ntest--;
        }
      }
    }
  }


  // solve the equation for the correlation
  ccoeff = solve(lhs,rhs);

  memset(nn_odd_col_map,0,sizeof(float[25]));
  memset(nn_even_col_map,0,sizeof(float[25]));
  for (int i=0;i < (MEASURE_SPAN_X*2+1)*(MEASURE_SPAN_Y*2+1);i++)
  {
    int ndx = EvenPutIndexMap[i];

    if (ndx != -1)
    {
      nn_odd_col_map[i] = (float)(ccoeff.at(ndx));
    }
  }

  for (int i=0;i < (MEASURE_SPAN_X*2+1)*(MEASURE_SPAN_Y*2+1);i++)
  {
    int ndx = OddPutIndexMap[i];

    if (ndx != -1)
    {
      nn_even_col_map[i] = (float)(ccoeff.at(ndx));
    }
  }


  avg_corr = ccoeff.at(mat_size-1);

  if ((region->row == 0) && (region->col == 0))
  {
    for (int i=0;i < (MEASURE_SPAN_X*2+1)*(MEASURE_SPAN_Y*2+1);i++)
    {
      int ndx = EvenPutIndexMap[i];

      if (ndx != -1)
      {
        result[i] = ccoeff.at(ndx);
      }
    }
    for (int r=0;r < (MEASURE_SPAN_Y*2+1);r++)
    {
      for (int c=0;c < (MEASURE_SPAN_X*2+1);c++)
      {
        printf("%5.3lf ",result[r*(MEASURE_SPAN_X*2+1)+c]);
      }
      printf("\n");
    }
    printf("Avg correlation = %5.3lf\n",ccoeff.at(mat_size-1));
  }

  delete [] result;
  delete [] ampl_map;
}

float SpatialCorrelator::MakeSignalMap(float *ampl_map, int fnum)
{
  Region *region = bkg.region_data->region;
  float region_mean_sig = 0.0f;
  
  memset(ampl_map,0,sizeof(float[region->w*region->h]));
  for (int ibd=0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    int row,col;
    bead_params *tbead = &bkg.region_data->my_beads.params_nn[ibd];
    row = tbead->y;
    col = tbead->x;

    float bead_sig = tbead->Copies*tbead->Ampl[fnum];
    ampl_map[row*region->w + col] = bead_sig;


    region_mean_sig += bead_sig;
  }

  region_mean_sig /= (region->w*region->h);
  return(region_mean_sig);
};

void SpatialCorrelator::AmplitudeCorrectAllFlows()
{
  for (int fnum=0; fnum<NUMFB; fnum++)
    NNAmplCorrect(fnum);
}

float modulate_effect_by_flow(float start_frac, float flow_num, float offset)
{
  float approach_one_rate = flow_num/(flow_num+offset);
  return ( (1.0f-start_frac) * approach_one_rate + start_frac);
}

void SpatialCorrelator::NNAmplCorrect(int fnum)
{
  // make a 2-d map
  //..which is kinda what the 1.wells is...but I don't care right now
  float *ampl_map;
  int NucId;

  ampl_map = new float[region->w*region->h];

  float region_mean_sig = MakeSignalMap(ampl_map,fnum);

  NucId = bkg.region_data->my_flow.flow_ndx_map[fnum];
  float const_frac    = 0.25f;
  float const_frac_ref = 0.86f;
  float flow_num = bkg.region_data->my_flow.buff_flow[fnum];
  float cscale    =  modulate_effect_by_flow(const_frac,     flow_num, 32.0f);
  float cscale_ref = modulate_effect_by_flow(const_frac_ref, flow_num, 32.0f);
  float etbR;
  reg_params *my_rp = &bkg.region_data->my_regions.rp;
  
  for (int ibd=0;ibd < bkg.region_data->my_beads.numLBeads;ibd++)
  {
    int row,col;
    bead_params *tbead = &bkg.region_data->my_beads.params_nn[ibd];
    row = tbead->y;
    col = tbead->x;

    float bead_corrector = UnweaveMap(ampl_map, row, col, region_mean_sig);

    etbR = AdjustEmptyToBeadRatioForFlow(tbead->R,my_rp,NucId,flow_num);
    // "the contribution from the neighbors discounted by the dampening effect of buffering
    // "the contribution already accounted for from the reference wells used
    // "which sense the mean region signal
    // "all on the scale set by the number of copies in this bead
    // plus some mysterious values
    // the whole effect phased in slowly over flows to maximum
    tbead->Ampl[fnum] -= 1.425f* ( (0.33f*etbR*bead_corrector*cscale/tbead->Copies) -
                                          ( (0.14f/const_frac_ref) *cscale_ref*region_mean_sig/tbead->Copies));
    if (tbead->Ampl[fnum]!=tbead->Ampl[fnum])
    {
      printf("NAN: corrected to zero at %d %d %d\n", row,col,fnum);
      tbead->Ampl[fnum] = 0.0f;
    }
  }

  delete [] ampl_map;
}


float SpatialCorrelator::UnweaveMap(float *ampl_map, int row, int col, float default_signal)
{
    float sum = 0.0f;
    float *coeffs;
    float coeff_sum = 0.0f;
    float coeff;

    int lr,lc;

    if (col & 1)
      //coeffs = nn_odd_col_map;
      coeffs = nn_even_col_map;
    else
      //coeffs = nn_even_col_map;
      coeffs = nn_odd_col_map;

    for (int r=row-NN_SPAN_Y;r<=(row+NN_SPAN_Y);r++)
    {
      lr = r-row+NN_SPAN_Y;
      for (int c=col-NN_SPAN_X;c<=(col+NN_SPAN_X);c++)
      {
        lc = c-col+NN_SPAN_X;
        coeff = coeffs[lr*(2*NN_SPAN_X+1)+lc];
        
        if ((r < 0) || (r>=region->h) || (c < 0) || (c>=region->w))
        {
          // if we are on the edge of the region...as a stand-in for actual data
          // use the region average signal
          sum += default_signal*coeff;
        }
        else
        {
          sum += ampl_map[r*region->w+c]*coeff;
        }

        coeff_sum += coeff;
      }
    }

    sum /= coeff_sum;
    return(sum);
}
