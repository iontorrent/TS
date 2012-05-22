/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DNTPRiseModel.h"

DntpRiseModel::DntpRiseModel(int _npts,float _C,float *_tvals,int _sub_steps)
{
    npts = _npts;
    tvals = _tvals;
    C = _C;
    i_start = 0;
    sub_steps = _sub_steps;
    splineflag = true; // we're using splineflag
}

static float nuc_rise_prototype[] = {
0.000000,0.000797,0.001724,0.002770,0.003926,0.005183,0.006530,0.007960,
0.009461,0.011025,0.012641,0.014369,0.016265,0.018317,0.020511,0.022835,
0.025278,0.027826,0.030467,0.033188,0.035977,0.038901,0.042022,0.045320,
0.048772,0.052358,0.056057,0.059849,0.063712,0.067625,0.071567,0.075586,
0.079733,0.083995,0.088359,0.092812,0.097340,0.101931,0.106570,0.111245,
0.115941,0.120688,0.125515,0.130413,0.135374,0.140390,0.145451,0.150549,
0.155676,0.160822,0.165979,0.171164,0.176397,0.181671,0.186980,0.192318,
0.197679,0.203058,0.208449,0.213844,0.219239,0.224645,0.230075,0.235525,
0.240989,0.246464,0.251946,0.257430,0.262912,0.268388,0.273853,0.279316,
0.284785,0.290258,0.295732,0.301203,0.306669,0.312125,0.317569,0.322997,
0.328407,0.333805,0.339198,0.344583,0.349957,0.355320,0.360667,0.365997,
0.371307,0.376595,0.381859,0.387104,0.392334,0.397549,0.402746,0.407924,
0.413080,0.418213,0.423322,0.428403,0.433456,0.438485,0.443493,0.448481,
0.453446,0.458386,0.463301,0.468189,0.473049,0.477878,0.482677,0.487448,
0.492195,0.496917,0.501613,0.506282,0.510923,0.515535,0.520116,0.524665,
0.529182,0.533669,0.538130,0.542564,0.546970,0.551348,0.555695,0.560012,
0.564298,0.568552,0.572772,0.576962,0.581124,0.585259,0.589365,0.593441,
0.597487,0.601502,0.605486,0.609438,0.613356,0.617245,0.621105,0.624937,
0.628741,0.632515,0.636259,0.639973,0.643656,0.647307,0.650926,0.654516,
0.658077,0.661612,0.665117,0.668595,0.672043,0.675461,0.678849,0.682207,
0.685534,0.688832,0.692103,0.695348,0.698565,0.701755,0.704916,0.708050,
0.711154,0.714229,0.717275,0.720293,0.723286,0.726253,0.729194,0.732109,
0.734997,0.737858,0.740692,0.743498,0.746277,0.749029,0.751757,0.754461,
0.757141,0.759795,0.762424,0.765028,0.767607,0.770159,0.772686,0.775188,
0.777667,0.780124,0.782557,0.784967,0.787354,0.789717,0.792056,0.794371,
0.796662,0.798930,0.801177,0.803402,0.805606,0.807788,0.809949,0.812088,
0.814204,0.816298,0.818370,0.820420,0.822451,0.824463,0.826454,0.828425,
0.830376,0.832307,0.834217,0.836107,0.837976,0.839826,0.841658,0.843471,
0.845266,0.847043,0.848801,0.850541,0.852261,0.853963,0.855646,0.857311,
0.858959,0.860591,0.862206,0.863804,0.865385,0.866949,0.868495,0.870025,
0.871537,0.873033,0.874514,0.875979,0.877429,0.878864,0.880283,0.881686,
0.883074,0.884446,0.885803,0.887144,0.888472,0.889786,0.891085,0.892371,
0.893643,0.894900,0.896143,0.897372,0.898587,0.899788,0.900977,0.902153,
0.903316,0.904466,0.905604,0.906729,0.907841,0.908940,0.910026,0.911100,
0.912162,0.913213,0.914253,0.915281,0.916297,0.917302,0.918295,0.919277,
0.920246,0.921205,0.922153,0.923091,0.924019,0.924936,0.925843,0.926739,
0.927625,0.928501,0.929365,0.930220,0.931066,0.931902,0.932729,0.933546,
0.934354,0.935153,0.935942,0.936721,0.937492,0.938253,0.939006,0.939750,
0.940486,0.941214,0.941933,0.942644,0.943346,0.944040,0.944725,0.945402,
0.946072,0.946734,0.947388,0.948035,0.948675,0.949306,0.949931,0.950547,
0.951156,0.951758,0.952353,0.952941,0.953522,0.954097,0.954665,0.955226,
0.955780,0.956327,0.956868,0.957402,0.957930,0.958452,0.958968,0.959478,
0.959982,0.960480,0.960972,0.961457,0.961937,0.962410,0.962879,0.963342,
0.963799,0.964251,0.964698,0.965139,0.965575,0.966005,0.966430,0.966850,
0.967265,0.967675,0.968080,0.968481,0.968877,0.969267,0.969653,0.970034,
0.970411,0.970782,0.971150,0.971513,0.971872,0.972226,0.972576,0.972922,
0.973264,0.973601,0.973934,0.974262,0.974587,0.974909,0.975226,0.975539,
0.975849,0.976155,0.976457,0.976755,0.977049,0.977340,0.977627,0.977911,
0.978192,0.978469,0.978742,0.979013,0.979279,0.979543,0.979803,0.980060,
0.980313,0.980564,0.980812,0.981057,0.981298,0.981537,0.981772,0.982005,
0.982235,0.982461,0.982685,0.982907,0.983125,0.983341,0.983555,0.983765,
0.983973,0.984178,0.984381,0.984581,0.984778,0.984974,0.985167,0.985357,
0.985545,0.985731,0.985914,0.986095,0.986274,0.986450,0.986625,0.986797,
0.986967,0.987135,0.987300,0.987464,0.987626,0.987785,0.987943,0.988098,
0.988252,0.988403,0.988553,0.988701,0.988847,0.988991,0.989134,0.989274,
0.989413,0.989550,0.989685,0.989819,0.989951,0.990081,0.990209,0.990336,
0.990462,0.990586,0.990708,0.990828,0.990947,0.991065,0.991181,0.991295,
0.991409,0.991520,0.991631,0.991740,0.991847,0.991953,0.992058,0.992161,
0.992263,0.992364,0.992464,0.992562,0.992659,0.992755,0.992849,0.992942,
0.993035,0.993126,0.993215,0.993304,0.993392,0.993478,0.993563,0.993648,
0.993731,0.993812,0.993893,0.993973,0.994052,0.994130,0.994207,0.994283,
0.994358,0.994432,0.994505,0.994577,0.994648,0.994718,0.994788,0.994856,
0.994924,0.994991,0.995056,0.995121,0.995185,0.995249,0.995311,0.995373,
0.995434,0.995494,0.995553,0.995612,0.995670,0.995727,0.995783,0.995838,
0.995893,0.995947,0.996001,0.996054,0.996106,0.996157,0.996208,0.996258,
0.996307,0.996356,0.996404,0.996452,0.996499,0.996545,0.996591,0.996636,
0.996681,0.996724,0.996768,0.996811,0.996856,0.996902,0.996948,0.996992,
0.997034,0.997073,0.997109,0.997140,0.997165,0.997184,0.997200,0.997212,
0.997222,0.997232,0.997243,0.997256,0.997272,0.997292,0.997317,0.997380,
0.997507,0.997690,0.997924,0.998201,0.998515,0.998858,0.999225,0.999608,
1.000000};

static int nuc_rise_protoype_len = (sizeof(nuc_rise_prototype) / sizeof(nuc_rise_prototype[0]));

#define MEASURED_NUC_RISE_SIGMA_SCALE_FACTOR (7.0)

float MeasuredNucRiseInterpolation(float x)
{
	int left, right;
	float frac;
	float ret;

    // scale it
    x *= 10.0f;

    // shift it to adjust for mid nuc location
    x = x + 115.0f;

    // anything less than or equal to 0 should be 0.0
    if (x <= 0.0f)
        return 0.0f;

	left = (int)x;    // left-most point in the lookup table
	right = left + 1; // right-most point in the lookup table

	// both left and right points are inside the table...interpolate between them
	if ((left >= 0) && (right < nuc_rise_protoype_len))
	{
		frac = (x - left);
		ret = (1.0f - frac) * nuc_rise_prototype[left] + frac * nuc_rise_prototype[right];
	}
	else
        ret = 1.0f;

    return(ret);
}

// time warp the measured nuc rise function to make stretched or squashed versions for anywhere in the chip
int MeasuredNucRiseFunction(float *output, int npts, float *frame_times, int sub_steps, float C, float t_mid_nuc, float sigma)
{
    int ndx = 0;
    float tlast = 0.0f;
    float last_nuc_value = 0.0f;
    int i_start = 0;
    bool i_start_uninitialized = true;

    // this scaling factor makes the time-warp of the measured rise function roughly match a sigmoid curve
    // with the same value of sigma..so expected values for this parameter change as little as possible
    sigma /= MEASURED_NUC_RISE_SIGMA_SCALE_FACTOR;

    memset(output,0,sizeof(float[npts*sub_steps]));
    for (int i=0;(i < npts) && (last_nuc_value < 0.999f*C);i++)
    {
        // get the frame number of this data point (might be fractional because this point could be
        // the average of several frames of data.  This number is the average time of all the averaged
        // data points
        float t=frame_times[i];

        for (int st=1;st <= sub_steps;st++)
        {
            float tnew = tlast+(t-tlast)*(float)st/sub_steps;
            float erfdt = (tnew-t_mid_nuc)/sigma;

            last_nuc_value = C*MeasuredNucRiseInterpolation(erfdt);

            output[ndx++] = last_nuc_value;
        }

        if (i_start_uninitialized && (last_nuc_value > 0.0f))//MIN_PROC_THRESHOLD))
        {
            i_start = i;
            i_start_uninitialized = false;
        }

        tlast = t;
    }

    for (;ndx < sub_steps*npts;ndx++)
        output[ndx] = C;

    return(i_start);
}

// spline with one knot
int SplineRiseFunction(float *output, int npts, float *frame_times, int sub_steps, float C, float t_mid_nuc, float sigma, float tangent_zero, float tangent_one)
{
    int ndx = 0;
    float tlast = 0;
    float last_nuc_value = 0.0;
    float scaled_dt = -1.0;
    float my_sigma = 3*sigma; // bring back into range for ERF
    
    bool i_start_uninitialized = true;
     int i_start = 0;  // always a legal value here

    memset(output,0,sizeof(float[npts*sub_steps]));
    
    for (int i=0;(i < npts) && (scaled_dt<1.0f);i++)
    {
        // get the frame number of this data point (might be fractional because this point could be
        // the average of several frames of data.  This number is the average time of all the averaged
        // data points
        float t=frame_times[i];

        for (int st=1;st <= sub_steps;st++)
        {
            float tnew = tlast+(t-tlast)*(float)st/sub_steps;
            scaled_dt = (tnew-t_mid_nuc)/my_sigma +0.5f;

            if ((scaled_dt>0))
            {
              float scaled_dt_square = scaled_dt*scaled_dt;
              float scaled_dt_minus = scaled_dt-1;
              last_nuc_value = scaled_dt_square*(3.0f-2.0f*scaled_dt); //spline! with zero tangents at start and end points
              last_nuc_value += scaled_dt_square*scaled_dt_minus * tangent_one; // finishing tangent, tangent at zero = 0
              last_nuc_value += scaled_dt*scaled_dt_minus*scaled_dt_minus * tangent_zero; // tangent at start, tangent at 1 = 0
              
              // scale up to finish at C
              last_nuc_value *= C;
              if (scaled_dt>1)
                last_nuc_value = C;
            }
            output[ndx++] = last_nuc_value;
        }

        // first time point where we have a nonzero time
        if (i_start_uninitialized && (scaled_dt>0.0f))
        {
            i_start = i;
            i_start_uninitialized=false; // now we have a true value here
        }

        tlast = t;
    }
    // if needed, can do a spline decrease to handle wash-off at end, but generally we're done with the reaction by then
    // so may be fancier than we need

    for (;ndx < sub_steps*npts;ndx++)
        output[ndx] = C;

    if (i_start_uninitialized)
      printf("ERROR_FINDING_I_START: t_mid_nuc: %f sigma: %f\n", t_mid_nuc, sigma);
    return(i_start);
}

// can probably delete this
int SigmaXRiseFunction(float *output,int npts, float *frame_times, int sub_steps, float C, float t_mid_nuc,float sigma)
{
    int ndx = 0;
    float tlast = 0.0f;
    float last_nuc_value = 0.0f;
    int i_start = 0;
    bool i_start_uninitialized = true;

    memset(output,0,sizeof(float[npts*sub_steps]));
    for (int i=0;(i < npts) && (last_nuc_value < 0.999f*C);i++)
    {
        // get the frame number of this data point (might be fractional because this point could be
        // the average of several frames of data.  This number is the average time of all the averaged
        // data points
        float t=frame_times[i];

        for (int st=1;st <= sub_steps;st++)
        {
            float tnew = tlast+(t-tlast)*(float)st/sub_steps;
            float erfdt = (tnew-t_mid_nuc)/sigma;

            if (erfdt >= -3.0f)
                last_nuc_value = C*(1.0+ErfApprox(erfdt))/2.0f;

            output[ndx++] = last_nuc_value;
        }

        if (i_start_uninitialized && (last_nuc_value >= MIN_PROC_THRESHOLD))
        {
            i_start = i;
            i_start_uninitialized = false;
        }

        tlast = t;
    }

    for (;ndx < sub_steps*npts;ndx++)
        output[ndx] = C;
    
    if (i_start_uninitialized)
      printf("ERROR_FINDING_I_START: t_mid_nuc: %f sigma: %f\n", t_mid_nuc, sigma);
    
    return(i_start);
}

// coment this out to enable the new nuc rise function based on measured curve (more accurate)
#define SPLINE_RISE_FUNCTION

int SigmaRiseFunction(float *output,int npts, float *frame_times, int sub_steps, float C, float t_mid_nuc,float sigma, bool splineflag)
{
if (splineflag)
    return(SplineRiseFunction(output,npts,frame_times,sub_steps,C,t_mid_nuc,sigma,0.0f,0.0f));
else
    return(MeasuredNucRiseFunction(output,npts,frame_times,sub_steps,C,t_mid_nuc,sigma));
}

int DntpRiseModel::CalcCDntpTop(float *output,float t_mid_nuc,float sigma)
{
    return(SigmaRiseFunction(output,npts,tvals,sub_steps,C,t_mid_nuc,sigma, splineflag));
}

