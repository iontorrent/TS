/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TraceCurry.h"

void TraceCurry::SingleFlowIncorporationTrace (float A,float *fval)
{
    // temporary
    //p->my_state.hits_by_flow[fnum]++;
    eval_count++;
    MathModel::RedTrace (fval, ivalPtr, npts, deltaFrameSeconds, deltaFrame, c_dntp_top_pc, sub_steps, i_start, C, A, SP, kr, kmax, d, molecules_to_micromolar_conversion, sens, gain, tauB, math_poiss, reg_p->hydrogenModelType);
}

void TraceCurry::SingleFlowIncorporationTrace (float A,float kmult,float *fval)
{
    //p->my_state.hits_by_flow[fnum]++;
    float tkr = region_kr*kmult; // technically would be a side effect if I reset kr.
    eval_count++;
    MathModel::RedTrace (fval, ivalPtr, npts, deltaFrameSeconds, deltaFrame, c_dntp_top_pc, sub_steps, i_start, C, A, SP, tkr, kmax, d, molecules_to_micromolar_conversion, sens, gain, tauB, math_poiss, reg_p->hydrogenModelType);
}

void TraceCurry::IntegrateRedObserved(float *red, float *red_obs)
{
    MathModel::IntegrateRedFromRedTraceObserved (red,red_obs, npts, i_start, deltaFrame, tauB);
}

void TraceCurry::ErrorSignal(float *obs,float *fit, float *posptr, float *negptr)
{

    float err[npts];
    float timetotal=0.0f,positive=0.0f, negative=0.0f;
    bool posflag=true;
    for (int k=0; k<npts;k++){ err[k]=(obs[k]-fit[k])*deltaFrame[k];}
    int k=npts-1;
    while  (k>=0){
       timetotal+=deltaFrame[k];
       if (err[k]>0 && posflag){ positive+=err[k];}
       else if (err[k]<0) {posflag =false;}
       if ((err[k]<0)&& (!posflag)) {negative+=err[k];};
       k--;
     }
     *posptr=positive/timetotal;
     *negptr=-negative/timetotal;
}


void TraceCurry::smooth_kern(float *out, float *in, int npts)
{
    const int order = 5;
    const float coeff[order] = {0.0269,0.2334,0.4794,0.2334,0.0269};
    //float kern3[3] = {0.1968, 0.6063, 0.1968};
    //float kern5[5] = {-0.0857,0.3429,0.4857,0.3429,-0.0857};
    //float kern7[7] = {0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006};
    const int half = (order-1)/2;
    for (int n=0; n<half; n++)
        out[n] = in[n];
    for (int n=npts-half; n<npts; n++)
        out[n] = in[n];
    for (int n=half; n<npts-half; n++) {
        float sum = 0;
        for (int i=0; i<order; i++){
            sum += coeff[i]*in[n-half+i];
        }
        out[n] = sum;
    }
}


void TraceCurry::savitzky_golay(float *out, float *in, int npts)
{
    const int order = 7;
    const float coeff[order] = {-0.0952381, 0.14285714, 0.28571429, 0.33333333, 0.28571429, 0.14285714, -0.0952381};
    const int half = (order-1)/2;
    for (int n=0; n<half; n++)
        out[n] = in[n];
    for (int n=npts-half; n<npts; n++)
        out[n] = in[n];
    for (int n=half; n<npts-half; n++) {
        float sum = 0;
        for (int i=0; i<order; i++){
            sum += coeff[i]*in[n-half+i];
        }
        out[n] = sum;
    }
}


int TraceCurry::maxIndex(float *trace, int npts) // to be replaced with min()
{
    if (npts<=0)
        return (0);
    int tmax = 0;
    float ymax = trace[0];
    for (int i=1; i<npts; i++) {
        if (ymax < trace[i]) {
            ymax = trace[i];
            tmax = i;
        }
    }
    return (tmax);
}


int TraceCurry::minIndex(float *trace, int npts) // to be replaced with min()
{
    if (npts<=0)
        return (0);
    int tmin = 0;
    float ymin = trace[0];
    for (int i=1; i<npts; i++) {
        if (ymin > trace[i]) {
            ymin = trace[i];
            tmin = i;
        }
    }
    return (tmin);
}


int TraceCurry::minIndex_local(float *in, int npts)
{
    if (npts<=0)
        return (0);
    int valley = 0;
    for (int i = 1; i < npts; i++) {
        if (in[i]<in[i-1])
            valley = i;
        else
            break;
    }
    return (valley);
}


double TraceCurry::regression_slope(vector<float>& Y, vector<float>& X)
{
    // http://easycalculation.com/statistics/learn-regression.php
    // simple calculation for 1D X, not the matrix form
    // callculate the slope only, no intercept
    assert ((X.size()==Y.size()) && X.size()>0);
    int nData = X.size();
    if (nData<=1)
        return (0);
    double sumX = 0;
    double sumY = 0;
    double sumXX = 0;
    double sumXY = 0;
    for (int i=0; i<nData; i++)
    {
        double x = X[i];
        double y = Y[i];
        sumX += x;
        sumY += y;
        sumXX += x*x;
        sumXY += x*y;
        //cout << i << "\t" << y << "\t" << x << endl << flush;
    }

    double slope = 0;
    try
    {
    slope = (nData*sumXY - sumX*sumY) / (nData*sumXX - sumX*sumX);
    }
    catch (...)
    {
        //slope = 0.5; // the middle of the range [0-1]
        slope = 0; // slope>-0.04 will be changed to default -1/tauB in GuessLogTaub
    }
    return slope;
}


float TraceCurry::slope_logy(float *trace,int slen)
{
    if (slen <= 1)
        return (0);

    vector<float> x(slen);
    vector<float> y(slen);
    float ymin = trace[0];
    for (int i=1; i<slen; i++)
    {
        if (ymin > y[i])
            ymin = y[i];
    }

    for (int i=0; i<slen; i++)
    {
        y[i] = log(trace[i]-ymin+1);
        x[i] = i;
    }
    // use linear regression to find the slope
    float slope = regression_slope(y,x); // doesn't seem to be accurate??
    return (slope);
}


float TraceCurry::boundTaub(float t,float minTaub,float maxTaub)
{
    if (t<minTaub) t = minTaub;
    else if (t>maxTaub) t = maxTaub;
    return (t);
}


float TraceCurry::boundTaub_mid(float t,float minTaub,float maxTaub,float midTaub)
{
    if (t<minTaub) t = midTaub;
    else if (t>maxTaub) t = midTaub;
    return (t);
}


float TraceCurry::boundTaub_nuc(float t,float minTaub,float maxTaub,float default_tauB_bead,float default_tauB_region)
{
    if (! validTaub(t,minTaub,maxTaub))
    {
        if (validTaub(default_tauB_bead,minTaub,maxTaub))
            t = default_tauB_bead;
        else if (validTaub(default_tauB_region,minTaub,maxTaub))
            t = default_tauB_region;
        else
            t = boundTaub(t,minTaub,maxTaub);
    }
    return (t);
}


float TraceCurry::find_slope_logy_regress(float *trc_smooth,int deltax)
{
    if (peak>=lastx){
        return (0);
    }
    int dx = minIndex(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 compressed timeframes
    //int dx = minIndex_local(&trc_smooth[peak],lastx-peak+1); // exclude the last 5 timeframes
    if (dx<deltax) {
        valley = peak + dx;
        return (0);
    }
    // slope_logy()
    bot = peak + dx;
    int slen = dx+1;
    vector<float> x(slen);
    vector<float> y(slen);
    for (int i=0; i<slen; i++)
    {
        y[i] = log(trc_smooth[peak+i]-ymin+1);
        x[i] = i;
    }
    // use linear regression to find the slope
    float slope = regression_slope(y,x); // doesn't seem to be accurate??
    return (slope);
}


float TraceCurry::find_slope_logy_valley(float *trc_smooth,int deltax)
{
    if (peak>=lastx){
        return (0);
    }
    int dx = minIndex(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 compressed timeframes
    //int dx = minIndex_local(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 comparessed timeframes
    //float slope = (dx > 0) ? slope_logy(&trc_smooth[peak],dx+1) : 0.00001;
    if (dx<deltax) {
        valley = peak + dx;
        return (0);
    }
    else {
        bot = peak + dx;
        float y0 = log(trc_smooth[peak]-ymin+1);
        float y1 = log(trc_smooth[peak+dx]-ymin+1);
        float dy = y1-y0;
        float slope = dy/dx;
        return (slope);
    }
}


float TraceCurry::log_slope(float *trc_smooth,int start,int end,float ymin)
{
    if (end<=start)
        return (0);
    //int tmin = minIndex(trc_smooth, npts);
    //float ymin = trc_smooth[tmin];
    //float y0 = log(trc_smooth[start]-ymin+1);
    //float y1 = log(trc_smooth[end]-ymin+1);
    float y0 = log(trc_smooth[start]-ymin+1);
    float y1 = log(trc_smooth[end]-ymin+1);
    float slope = (y1 - y0) / float(end-start);
    return (slope);
}


void TraceCurry::set_peak(float *trace,int npts,int deltax)
{
    firstx = 6;
    lastx = 29; // last uncompressed timeframe
    peak = maxIndex(&trace[firstx],lastx-firstx+1) + firstx; // better than corrected
    valley = minIndex(trace, npts);
    ymin = trace[valley];
    top =  peak;
    bot = peak + deltax;
}


float TraceCurry::find_slope_logy_seg6(float *trc_smooth, int deltax)
{
    if (peak>=lastx) {
        return (0);
    }
    int dx = minIndex(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 compressed timeframes
    //int dx = minIndex_local(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 comparessed timeframes
    //float slope = (dx > 0) ? slope_logy(&trc_smooth[peak],dx+1) : 0.00001;
    if (dx<1) {
        bot = peak;
        return (0);
    }
    else if (dx<deltax) {
        bot = peak+dx;
        float slope = log_slope(trc_smooth,peak,bot,ymin);
        return (slope);
    }
    else {
        int start = peak;
        int end = peak+deltax;
        bot = end;
        float slope = log_slope(trc_smooth,start++,end++,ymin);
        /*
        int r = p->y+region->row;
        int c = p->x+region->col;
        bool is_xyflow = (r==303 && c==124 && flow==15) ? true:false;
        if (is_xyflow) {
            std::cout << "r==303 && c==124 && flow==15" << std::endl << std::flush;
            std::cout << "peak:" << peak << " valley:" << valley << std::endl << std::flush;
            std::cout << "top:" << top << " bot:" << bot << " slope:" << slope << std::endl << std::flush;
            for (int i=0; i<lastx; i++)
                std::cout << trc_smooth[i] << " ";
            std::cout << std::endl << std::flush;
        }
        */
        int startmax = peak+3;
        while (start<startmax && end<lastx) {
            float s = log_slope(trc_smooth,start,end,ymin);
            if (s<slope) {
                slope = s;
                top = start++;
                bot = end++;
                //if (is_xyflow) std::cout << "top:" << top << " bot:" << bot << " slope:" << slope << std::endl << std::flush;
            }
            else
                break;
        }
        return (slope);
    }
}


float TraceCurry::find_slope_logy_min(float *trc_smooth, int deltax)
{
    if (peak>=lastx){
        return (0);
    }
    int dx = minIndex(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 compressed timeframes
    //int dx = minIndex_local(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 comparessed timeframes
    //float slope = (dx > 0) ? slope_logy(&trc_smooth[peak],dx+1) : 0.00001;
    if (dx<deltax) {
        return (0);
    }
    else {
        float y0 = log(trc_smooth[peak]-ymin+1);
        float y1 = log(trc_smooth[peak+1]-ymin+1);
        float slope = y1 - y0;
        top = peak;
        bot = peak + 1;
        for (int i=2; i<=dx; i++) {
            y1 = log(trc_smooth[peak+i]-ymin+1);
            float ts = (y1-y0)/i;
            if (slope > ts) { // min
                slope = ts;
                bot = peak + i;
            }
        }
        return (slope);
    }
}


float TraceCurry::find_slope_logy_max(float *trc_smooth, int deltax)
{
    if (peak>=lastx){
        valley = lastx;
        return (0);
    }
    //int dx = minIndex(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 compressed timeframes
    int dx = lastx - peak;
    //int dx = minIndex_local(&trc_smooth[peak],lastx-peak+1); // exclude the last 18 comparessed timeframes
    //float slope = (dx > 0) ? slope_logy(&trc_smooth[peak],dx+1) : 0.00001;
    if (dx<deltax) {
        return (0);
    }
    else {
        float y0 = log(trc_smooth[peak]-ymin+1);
        float y1 = log(trc_smooth[peak+1]-ymin+1);
        float slope = y1 - y0;
        top = peak;
        bot = peak + 1;
        for (int i=2; i<=dx; i++) {
            y1 = log(trc_smooth[peak+i]-ymin+1);
            float ts = (y1-y0)/i;
            if (slope < ts) { // max
                slope = ts;
                bot = peak + i;
            }
        }
        return (slope);
    }
}


void TraceCurry::GuessLogTaub(float *signal_corrected, BeadParams *p, int fnum, reg_params *rp)
{
    //assert (npts>5);
    float trc_smooth[npts];
    // gaussian smooth (blur filter)
    smooth_kern(trc_smooth, signal_corrected, npts);
    //savitzky_golay(trc_smooth, signal_corrected, npts);
    set_peak(trc_smooth,npts);
    //set_peak(signal_corrected,npts);

    //float slope = find_slope_logy_regress(trc_smooth);
    //float slope = find_slope_logy_valley(trc_smooth);
    //float slope = find_slope_logy_max(trc_smooth);
    //float slope = find_slope_logy_min(trc_smooth);
    //float slope = find_slope_logy_seg6(trc_smooth);
    //float slope = find_slope_logy_seg6(signal_corrected);
    float slope = find_slope_logy_seg6(trc_smooth);
    float t = (slope>=0) ? 100000 : -1/slope;
    //p->tauB_nuc[NucID] = tauB = boundTaub_mid(t,rp->min_tauB,rp->max_tauB,rp->mid_tauB);
    //p->tauB_nuc[NucID] = tauB = boundTaub_nuc(t,rp->min_tauB,rp->max_tauB,p->tauB_nuc[NucID],tauB);
    //p->tauB_nuc[NucID] = tauB = boundTaub(t,rp->min_tauB,rp->max_tauB);
    tauB = boundTaub(t,rp->min_tauB,rp->max_tauB);
    //p->tauB[fnum] = tauB; // saved later in SingleFlowFit.cpp, don't need to save here??
    //p->peak[fnum] = peak;
    //p->valley[fnum] = valley;
    //p->top[fnum] = top;
    //p->bot[fnum] = bot;
}


float TraceCurry::GuessAmplitude(float *red_obs)
{
    float red_guess[npts];
    IntegrateRedObserved(red_guess,red_obs);
    float offset = SP*sens;
    int test_pts = i_start+reg_p->nuc_shape.nuc_flow_span*0.75;  // nuc arrives at i_start, halfway through is a good guess for when we're done
    if (test_pts>npts-1)
        test_pts = npts-1;
    float a_guess = (red_guess[test_pts]-red_guess[i_start])/offset;  // take the difference to normalize out anything happening before incorporation
  return(a_guess);
}

void TraceCurry::SetWellRegionParams (struct BeadParams *_p,struct reg_params *_rp,int _fnum,
                                      int _nnum,int _flow,
                                      int _i_start,float *_c_dntp_top)
{
    p = _p;
    reg_p = _rp;
    fnum = _fnum;
    NucID = _nnum;
    flow = _flow;


    // since this uses a library function..and the parameters involved aren't fit
    // it's helpful to compute this once and not in the model function
    SP = (float) (COPYMULTIPLIER * p->Copies) *pow (reg_p->CopyDrift,flow);

    etbR = reg_p->AdjustEmptyToBeadRatioForFlow (p->R, p->Ampl[fnum], p->Copies, p->phi, NucID, flow);
    tauB = reg_p->ComputeTauBfromEmptyUsingRegionLinearModel (etbR);

    sens = reg_p->sens*SENSMULTIPLIER;
    molecules_to_micromolar_conversion = reg_p->molecules_to_micromolar_conversion;
    d = reg_p->d[NucID]*p->dmult;
    kr = reg_p->krate[NucID]*p->kmult[fnum];
    region_kr = reg_p->krate[NucID];
    kmax = reg_p->kmax[NucID];
    C = reg_p->nuc_shape.C[NucID]; // setting a single flow
    gain = p->gain;
    // it is now necessary to have these precomputed somewhere else
    i_start = _i_start;
    c_dntp_top_pc = _c_dntp_top;
}

// what we're >really< setting
void TraceCurry::SetContextParams(int _i_start, float *c_dntp_top, int _sub_steps, float _C, float _SP, float _region_kr, float _kmax, float _d, float _sens, float _gain, float _tauB)
{
    i_start = _i_start;
    c_dntp_top_pc = c_dntp_top;
    // assume I directly know the parameters and don't have to recompute them
    C = _C;
    SP = _SP;
    region_kr = _region_kr;
    kr = region_kr;
    kmax = _kmax;
    d = _d;
    sens = _sens;
    gain = _gain;
    tauB = _tauB;
    sub_steps = _sub_steps;
}

TraceCurry::TraceCurry()
{
    ivalPtr = NULL;
    npts = 0;
    deltaFrame = NULL;
    deltaFrameSeconds = NULL;
    c_dntp_top_pc = NULL;
    math_poiss = NULL;
    p = NULL;
    reg_p = NULL;
    fnum = 0;
    NucID = 0;
    flow = 0;
    SP = 0.0f;
    etbR = 0.0f;
    tauB = 0.0f;
    sens = 0.0f;
    molecules_to_micromolar_conversion = 0.0f;
    d = 0.0f;
    kr = 0.0f;
    region_kr = 0.0f;
    kmax = 0.0f;
    C = 0.0f;
    i_start = 0;
    gain = 0.0f;
    eval_count = 0;
    sub_steps = ISIG_SUB_STEPS_SINGLE_FLOW;
    firstx = 6;
    lastx = 29; // last uncompressed timeframe
}

// constructor
void   TraceCurry::Allocate (int _len,float *_deltaFrame, float *_deltaFrameSeconds, PoissonCDFApproxMemo *_math_poiss)
{
    ivalPtr = new float[_len];
    npts = _len;
    deltaFrame = _deltaFrame;
    deltaFrameSeconds = _deltaFrameSeconds;
    c_dntp_top_pc = NULL;
    math_poiss = _math_poiss;
}

TraceCurry::~TraceCurry()
{
    if (ivalPtr!=NULL)
        delete[] ivalPtr;
}
