/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <assert.h>
#include <math.h>
#include <sstream>
#include <vector>
#include "EmptyTrace.h"
#include "BkgTrace.h"
#include "TraceClassifier.h"
#include <cmath>

using namespace std;


typedef pair<int,int> mypair;
bool cmpPairAsend (const mypair& a, const mypair& b) { return a.first < b.first; }
bool cmpPairDesend (const mypair& a, const mypair& b) { return a.first > b.first; }


template<class T>
string formatVector(const vector<T>& v, string msg)
{
    stringstream ss;
    int nV = v.size();
    for (int n=0; n<nV; n++)
    {
        ss << " " << v[n];
    }
    msg += ss.str();
    return msg;
}


string formatVector(const float *v, int sz, string msg)
{
    stringstream ss;
    for (int n=0; n<sz; n++)
    {
        ss << " " << v[n];
    }
    msg += ss.str();
    return msg;
}


template<class T>
T max(vector<T>& data)
{
    assert(data.size()>0);
    T vMax = data[0];
    for (size_t n=0; n<data.size(); n++)
    {
        if ((data[n]) > vMax)
            vMax = data[n];
    }
    return vMax;
}


template<class T>
T min(vector<T>& data)
{
    assert(data.size()>0);
    T vMin = data[0];
    for (size_t n=0; n<data.size(); n++)
    {
        if ((data[n]) < vMin)
            vMin = data[n];
    }
    return vMin;
}


template<class T>
T sort_pair(const vector<T> & data, vector<int> & Index, bool ascending=true)
{
    // A vector of a pair which will contain the sorted value and its index in the original array
    vector<pair<T,int> > myPair;
    myPair.resize(data.size());
    for(int i=0;i<(int)myPair.size();++i)
    {
        myPair[i].first = data[i];
        myPair[i].second = i;
    }
    if (ascending)
        sort(myPair.begin(),myPair.end(),cmpPairAsend);
    else
        sort(myPair.begin(),myPair.end(),cmpPairDesend);

    Index.resize(data.size());
    for(size_t i = 0; i < Index.size(); ++i) Index[i] = myPair[i].second;

    return myPair[0].first;
}


template<class T>
int findLowestValue_index(vector<T>& v, bool ascending=true)
{
    vector<int> index;
    T minV = sort_pair(v,index,ascending);

    vector<T> candidates(0);
    int nV = v.size();
    for (int n=0; n<nV; n++)
    {
        if (v[n] == minV)
            candidates.push_back(v[n]);
    }

    return (candidates.size() > 0 ? candidates[0] : 0);
}


template<class T>
T middle(vector<T>& v)
{
    int sz = v.size();
    assert(sz>0);
    if (sz<=2)
        return v[sz/2];
    else
    {
        vector<int> index;
        sort_pair(v,index);
        return v[index[sz/2]];
    }

}


template<class T>
T sum(T *data, int nData)
{
    T total = 0;
    for (int i=0; i<nData; i++)
        total += data[i];
    return total;
}


template<class T>
T sum(vector<T>& data)
{
    int nData = data.size();
    T total = 0;
    for (int i=0; i<nData; i++)
        total += data[i];
    return total;
}


template<class T>
T mean(T *data, int nData)
{
    T total = sum(data,nData);
    return total/nData;
}


template<class T>
T mean(vector<T>& data)
{
    int nData = data.size();
    T total = sum(data);
    return total/nData;
}


template<class T>
T stdev (T *data, int nData)
{
    T mu = mean(data,nData);
    T sd = 0;

    for (int i = 0; i < nData; i++)
    {
        float d = data[i] - mu;
        sd += d*d;
    }

    if (nData>1)
        sd /= (nData-1);
    sd = sqrt(sd);

    return sd;
}


template<class T>
T stdev (vector<T>& data)
{
    int nData = data.size();
    T mu = mean(data);
    T sd = 0;

    for (int i = 0; i < nData; i++)
    {
        float d = data[i] - mu;
        sd += d*d;
    }

    if (nData>1)
        sd /= (nData-1);
    sd = sqrt(sd);

    return sd;
}



template<class T>
T rootmeansquare (T *data, int nData)
{
    T ss = 0;
    for (int i = 0; i < nData; i++)
    {
        float d = data[i];
        ss += d*d;
    }

    if (nData>0)
        ss = sqrt(ss/nData);
    ss = sqrt(ss);
    return ss;
}


template<class T>
T rootmeansquare (vector<T>& data)
{
    int nData = data.size();
    T ss = 0;
    for (int i = 0; i < nData; i++)
    {
        float d = data[i];
        ss += d*d;
    }

    if (nData>0)
        ss = sqrt(ss/nData);
    ss = sqrt(ss);
    return ss;
}


template<class T>
void smooth(vector<T>& data, vector<T>& dataOut, int win=1)
{
    int nData = data.size();
    dataOut = data;
    for (int i=win; i<nData-win; i++)
    {
        vector <T> v;
        v.resize(win*2);
        for (int n=0; n<win*2; n++)
            v[n] = data[i-win+n];
        dataOut[i] = mean(v);
    }
}



float TraceClassifier::computeKurtosis(float *sig, int nSig)
{
    if (nSig<=1)
        return (0.0);

    float mu = mean(sig,nSig);
    float sd = stdev(sig,nSig);
    float kurt = 0;
    for (int n=0; n<nSig; n++)
    {
        float a = sig[n] - mu;
        kurt += a*a*a*a;
    }

    kurt /= (nSig-1)*sd*sd*sd*sd;
    kurt -= 3;

    return kurt;
}



int TraceClassifier::findValleyBeforePeak(const vector<int>&valleys,int peak)
{
    int nValleys = valleys.size();
    int v0 = 0;
    for (int n=0; n<nValleys; n++)
    {
        if (valleys[n]<peak)
            v0 = valleys[n];
        else
            break;
    }
    return v0;
}


int TraceClassifier::findValleyAfterPeak(const vector<int>&valleys,int peak)
{
    int nValleys = valleys.size();
    int v0 = peak+1;
    for (int n=0; n<nValleys; n++)
    {
        if (valleys[n]>peak)
        {
            v0 = valleys[n];
            break;
        }
    }
    return v0;
}


void TraceClassifier::findPeakBoarders(const vector<int>&valleys,int peak,vector<int>& boarders)
{
   boarders.resize(2);
   if (peak<=0)
   {
       boarders[0] = 0;
       boarders[1] = findValleyAfterPeak(valleys,peak);
   }
   else if (peak>=mNumBins-1)
   {
       boarders[1] = mNumBins-1;
       boarders[0] = findValleyBeforePeak(valleys,peak);
   }
   else
   {
       boarders[0] = findValleyBeforePeak(valleys,peak);
       boarders[1] = findValleyAfterPeak(valleys,peak);
   }
}



void TraceClassifier::findClosestNeighbors(const vector<int>&peaks,int peak,vector<int>&neighbors)
{
    neighbors.resize(0);
    if (peaks.size() <= 1)
        return;

    int nPeaks = peaks.size();
    int iNeg = 0;
    int iPos = nPeaks-1;
    int dNeg = -mNumBins;
    int dPos = mNumBins;
    bool foundNeg = false;
    bool foundPos = false;

    for (int n=0; n<nPeaks; n++)
    {
        if (peaks[n]==peak)
            continue;
        int dist = peaks[n] - peak;
        if (dist>0 && dist<dPos)
        {
            iPos = n;
            dPos = dist;
            foundPos = true;
        }
        else if (dist<0 && abs(dist)<abs(dNeg))
        {
            iNeg = n;
            dNeg = dist;
            foundNeg = true;
        }
    }

    if (foundNeg && peaks[iNeg] != peak)
        neighbors.push_back(peaks[iNeg]);
    if (foundPos && peaks[iPos] != peak)
        neighbors.push_back(peaks[iPos]);
}



float TraceClassifier::computeMinFs(const vector<int>&peaks, vector<int>& valleys)
{
    int nPeaks = peaks.size();
    mFstat.resize(nPeaks);
    if (nPeaks<=1)
        return (0.0);

    for (int n=0; n<nPeaks; n++)
    {
        vector <int> pkbd;
        findPeakBoarders(valleys,peaks[n],pkbd);
        //cerr << flowRegionStr() << "findPeakBoarders peak=" << peaks[n] << formatVector(pkbd," boarders=") <<endl << flush;
        vector<int> neighbors;
        findClosestNeighbors(peaks,peaks[n],neighbors);
        //cerr << formatVector(neighbors,flowRegionStr()+" findClosestNeighbors peak="+val2str(peaks[n])) << endl << flush;
        int nbs = neighbors.size();
        if (nbs<1)
            mFstat[n] = 0.0;
        else
        {
            vector<float> fs(nbs);
            for (int i=0; i<nbs; i++)
            {
                vector <int> bd;
                findPeakBoarders(valleys,neighbors[i],bd);
                float f = anova(bd,pkbd); // fstat for each peak/neighbor
                fs[i] = f;
                //cerr << flowRegionStr()+" computeMinFs peak="+val2str(peaks[n])+" neighbor="+val2str(neighbors[i])+ " f="+val2str(f) << endl << flush;
            }
            float mfs = fs[0];
            if (nbs>1)
            {
                float mfs = min(fs[0],fs[1]);
                if (mfs <= 0.0)
                    mfs = max(fs[0],fs[1]);
            }
            mFstat[n] = mfs;
        }
    }

    vector<float> tmpstat(mFstat);
    sort(tmpstat.begin(), tmpstat.end());
    float minFs = tmpstat[0];

    return minFs;
}



template<class T>
bool isBetween(T v, T v1, T v2)
{
    if (v1 > v2)
        swap(v1,v2);

    if ((v>= v1 && v<=v2) || (v>= v2 && v<=v1))
        return true;
    else
        return false;
}



void range(int iL, int iR, vector<int>& idx)
{
    if (iR<iL)
        swap(iR,iL);
    int sz = abs(iR-iL)+1;
    idx.resize(sz);
    for (int n=0; n<sz; n++)
        idx[n] = iL+n;
}



template<class T>
T moment(vector<T>& hist, int iL, int iR, int order=1)
{
    vector<int> idx;
    range(iL,iR,idx);
    T m = 0;
    int iStart = idx[0];
    for (int i=0; i<(int)idx.size(); i++)
    {
        if (order == 0)
            m += hist[i];
        else{
            int distance = abs(i-iStart)+1;
            if (order > 1)
                distance = pow(distance,order);
            m += hist[i] * distance;
        }
    }
    return m;
}


float regression_slope(vector<float>& Y, vector<float>& X)
{
    // http://easycalculation.com/statistics/learn-regression.php
    // simple calculation for 1D X, not the matrix form
    // callculate the slope only, no intercept
    assert (X.size()==Y.size());
    int nData = X.size();
    float sumX = 0;
    float sumY = 0;
    float sumXX = 0;
    float sumXY = 0;
    for (int i=0; i<nData; i++)
    {
        float x = X[i];
        float y = Y[i];
        sumX += x;
        sumY += y;
        sumXX += x*x;
        sumXY += x*y;
        //cout << i << "\t" << y << "\t" << x << endl << flush;
    }

    float slope = 0;
    try
    {
    slope = (nData*sumXY - sumX*sumY) / (nData*sumXX - sumX*sumX);
    }
    catch (...)
    {
        slope = 0.5; // the middle of the range [0-1]
    }

    //cout << "slope=" << slope << endl << flush;     exit(1);

    return slope;
}


// Compare function for qsort in Descending order
static int compareDescend (const void *v1, const void *v2)
{
  float val1 = * (float *) v1;
  float val2 = * (float *) v2;

  if (val1 < val2)
    return 1;
  else if (val2 < val1)
    return -1;
  else
    return 0;
}


bool hasSubstr(string str, string sub)
{
    size_t found = str.find(sub);
    return ( (found != string::npos) ? true : false);
}


void range(string in, vector<int>& out)
{
    vector<string> tokens;
    split(in,':',tokens);
    assert(tokens.size()==2);
    vector<int>tmp;
    for (int n=0; n<(int)tokens.size(); n++)
        tmp.push_back(atoi(tokens[n].c_str()));
    for (int i=tmp[0]; i<=tmp[1]; i++)
        out.push_back(i);;
}


bool TraceClassifier::flow_is_on(int flow)
{
    for (int n=0; n<(int)flows_on.size(); n++)
        if (flows_on[n]==flow)
            return true;
    return false;
}



bool TraceClassifier::needTraceClassifier(int flow)
{
    if (notEnoughEmptyWells())
        return false;

    if (mForceAllFlows)
        return true;

    if (mDoAllFlows) {
         cout << get_decision_metrics() << " delta_too_low()=" << delta_too_low() << " cv_too_low()=" << cv_too_low() << endl << flush;
        return ((delta_too_low() || cv_too_low())? false:true);
        //return ((delta_too_low() || rms_too_low()  || cv_too_low())? false:true);
        //return ((delta_too_low())? false:true);
    }
    else
        return (flow_is_on(flow) ? true:false);

}


bool TraceClassifier::needTopPicks() // for cv/snr calculation only
{
    if (mNumEmptyWells<=0)
        return false;

    if (mForceAllFlows)
        return false;

    //return mDoAllFlows ? true:false;
    return mDoAllFlows ? true:true; // for TopPicks and generate the cv

}


int TraceClassifier::parse_flowsOpt(const string &flowsOpt, vector<int>& flows_on)
{
    if (flowsOpt.length()==0)
        return 0;

    mDoAllFlows = false;
    mForceAllFlows = false;
    flows_on.resize(0);
    if (hasSubstr(flowsOpt,"all"))
    {
        mForceAllFlows = true;
    }
    else
    {
         if (hasSubstr(flowsOpt,","))
         {
             vector<string> tokens;
             split(flowsOpt,',',tokens);

             for (int n=0; n<(int)tokens.size(); n++)
             {
                 if (hasSubstr(tokens[n],":"))
                 {
                     range(tokens[n],flows_on);
                 }
                 else
                 {
                     flows_on.push_back(atoi(tokens[n].c_str()));
                 }
             }
         }
         else if (hasSubstr(flowsOpt,":"))
         {
             range(flowsOpt,flows_on);
         }
    }

    if (flows_on.size()==0)
    {
        mDoAllFlows = true;
    }
    else
    {
        for (int n=0; n<(int)flows_on.size(); n++)
            flows_on[n]--; // change to 0-index
    }

    cout << "TraceClassifier::parse_flowsOpt... flow=" << flow << formatVector(flows_on," flows_on=") << " mDoAllFlows=" << mDoAllFlows << " mForceAllFlows=" << mForceAllFlows << endl << flush;
    return flowsOpt.length();

}



void TraceClassifier::copyAvgTrace(float *bPtr)
{
    assert(avgTrace.size() == (size_t)imgFrames);
    for (int frame=0; frame<imgFrames; frame++)
        bPtr[frame] = avgTrace[frame];
}


void TraceClassifier::copyAvgTrace(vector<float>& bPtr)
{
    assert(avgTrace.size() == (size_t)imgFrames);
    for (int frame=0; frame<imgFrames; frame++)
        bPtr[frame] = avgTrace[frame];
}


int TraceClassifier::fitEmptyTrace()
{
    int nSig = 0;

    // 1.0 find mNumEmptyWells, hi/lo/sum of the traces
    mNumEmptyWells = generateAvgTrace(true); // true=emptyonly

    if (!notEnoughEmptyWells())
    {
        nSig = fitBaseline(true);
    }
    //cerr <<  "fitEmptyTrace..."+flowRegionStr() << endl << flush;
    return (nSig);
}



int TraceClassifier::fitBaseline(bool emptyonly)
{
    assert (mNumWellsInRegion > 0);
    int nWells = emptyonly ? mNumEmptyWells : mNumWellsInRegion;
    assert(nWells>0);

    copyAvgTrace(avgTrace_old);
    ///--------------------------------------------------------------------------------------------------------
    /// find out the starting/ending frames (x0,x1) from the diff
    ///--------------------------------------------------------------------------------------------------------

    findBaseFrames();

    // interpolate the baseline between mSigFrame_start & mSigFrame_end
    float dydx = (avgTrace_old[mSigFrame_end] - avgTrace_old[mSigFrame_start]) / (mSigFrame_end - mSigFrame_start);

    for (int n=mSigFrame_start+1; n<mSigFrame_end; n++)
    {
        avgTrace[n] = avgTrace_old[n] + dydx * (n-mSigFrame_start);
    }

    return (nWells);
}


void TraceClassifier::findBaseFrames()
{
    vector<float> dYY_sorted;
    dYY_sorted = dYY;
    sort(dYY_sorted.begin(),dYY_sorted.end());

    int iThresh = mBaselineFraction * imgFrames;

    float thresh = dYY_sorted[iThresh];
    for (int n=0; n<imgFrames; n++)
        baseframes[n] = dYY[n] < thresh ? true:false;

    vector<float> ySmooth;
    smooth(lo,ySmooth);
    findLocalMin(ySmooth,mValleys);
    findLocalMax(ySmooth,mPeaks);
    int nValleys = mValleys.size();
    mSigFrame_start= mValleys[0];
    int ith=1;
    while (mValleys[ith]<mSigFrame_start_max && ith<nValleys)
        mSigFrame_start = mValleys[ith++];

    while (mSigFrame_start<mSigFrame_start_max)
        if (baseframes[mSigFrame_start+1])
            mSigFrame_start++;

    mSigFrame_end = imgFrames - 1;
    while (baseframes[mSigFrame_end-1])
        mSigFrame_end--;

    int extra = (mSigFrame_end- mSigFrame_start) - (imgFrames - mSigFrame_win);
    if (extra > 0)
        mSigFrame_end -= extra;

}




int TraceClassifier::computeClusterMean(int HiLo)
{
    mHiLo = HiLo;

    int nSig = 0;

    // 1.0 find mNumEmptyWells, hi/lo/sum of the traces
    //evalEmptyWells();
    mNumEmptyWells = generateAvgTrace(true); // true=emptyonly
    //cout <<  "computeClusterMean..." << flowRegionStr() << formatVector(avgTrace," avgTrace before classifier") << endl << flush;
    //-------------------------------------------------------------------------------------------
    // check normalized snr to see whether TraceClassifier is needed or not
    //-------------------------------------------------------------------------------------------
    //if (mNumEmptyWells>0)
    if (needTopPicks())
    {
        findTopPicks();
        //compute_avgTrace_snr();
        compute_avgTrace_cv(); // cv is checked in needTraceClassifier()
    }
    //else
    //    setPicks(true); // default


    if (!needTraceClassifier(flow))
    {
        cout << "TraceClassifier not called"<< get_decision_metrics() << endl << flush;
        return (nSig);
    }
    else {
        cout << "TraceClassifier called" << get_decision_metrics() << endl << flush;
    }

    // 1.1 check to see if clustering is necessary
    // return if the distribution is ok???
    // avgTrace regression slope is < 0.5??? then do cluster analysis???

    // 2.0 cluster the traces

    // 2.1 copy the raw empty traces
    //copyAllEmptyTraces();
    // not enough memory to do copyAllEmptyTraces()
    // do calculation on the fly for two passes instead

    // 2.2 cluster analysis
    if (noEmptyWells())
    {
        //nSig = generateLowTrace(); // use the low envelope as the empty trace
        //generateAvgTrace(false); // false=all wells
        //nSig = clusterTraces(false); // use all traces to do clustering
        nSig = 0; // force emptyTrace->GenerateAverageEmptyTrace() to be called!!!

        // 3.0 result printing
        //cout << "computeClusterMean warning: mNumEmptyWells < mNumEmptyWells_min: " << mNumEmptyWells << "<" << mNumEmptyWells_min << flowRegionStr() << mThresholdMethod << " cv=" << get_cv() << endl << flush;
        //cout << "computeClusterMean... " << mTruePercent << "% of " << nSig << " all region wells are used in clusterTraces()" << flowRegionStr() << mThresholdMethod << " cv=" << get_cv() << endl << flush;
        cout << "computeClusterMean... clusterTraces(false) called to calculate the avgTrace from all wells -> thresh=" << get_threshold() << endl << flush;
    }
    else if (notEnoughEmptyWells())
    {
        //generateAvgTrace(false); // false=all wells
        //nSig = clusterTraces(false); // use all traces to do clustering
        nSig = 0; // force emptyTrace->GenerateAverageEmptyTrace() to be called!!!
    }
    else
    {
        nSig = clusterTraces(true);

        // 3.0 result printing
        //cout << "computeClusterMean found mNumTrueEmpties=" << mNumTrueEmpties << " out of " << mNumEmptyWells << flowRegionStr() << mThresholdMethod << " method" << endl << flush;
        cout << "computeClusterMean... " << mTruePercent << "% of " << nSig << " true empty wells are used in clusterTraces()" << flowRegionStr() << mThresholdMethod << " cv=" << get_cv() << endl << flush;
        cout << "computeClusterMean... clusterTraces(true) called to calculate the avgTrace from true empties -> thresh=" << get_threshold() << endl << flush;

    }
    //cout <<  "computeClusterMean..." << flowRegionStr() << formatVector(avgTrace," avgTrace after classifier") << endl << flush;
    cerr <<  "computeClusterMean..."+flowRegionStr()+ " threshIdx=" +val2str(mThreshIdx)+formatVector(hist," histogram") << endl << flush;
    return (nSig);
}



///---------------------------------------------------------------------------------------------------------
/// below are all private functions
///---------------------------------------------------------------------------------------------------------

void TraceClassifier::copyToAvgTrace(float *bPtr)
{
    assert(avgTrace.size() == (size_t)imgFrames);
    for (int frame=0; frame<imgFrames; frame++)
        avgTrace[frame] = bPtr[frame];
}


void TraceClassifier::copyToAvgTrace(vector<float>& bPtr)
{
    assert(avgTrace.size() == (size_t)imgFrames);
    for (int frame=0; frame<imgFrames; frame++)
        avgTrace[frame] = bPtr[frame];
}


int TraceClassifier::clusterTraces(bool emptyonly)
{
    assert (mNumWellsInRegion > 0);

    ///--------------------------------------------------------------------------------------------------------
    // avgTrace is done in generateAvgTrace() already;
    // save what's in avgTrace in case need to return it if mNumTrueEmpties < 50?
    ///--------------------------------------------------------------------------------------------------------

    copyAvgTrace(avgTrace_old);


    ///--------------------------------------------------------------------------------------------------------
    /// use histogram/PVT/BMT method to find the true empty traces
    ///--------------------------------------------------------------------------------------------------------
    // make the histogram from all the traces
    // 1.0 find the hi/lo (envelope) points in each time frame
    // get_HiLo_fromEmptyWells();

    // 1.1 do linear regression to find the signal coefficient
    int nWells = emptyonly ? mNumEmptyWells : mNumWellsInRegion;
    assert(nWells>0);
    float sig[nWells];

    int nSig = cluster_regression(sig,emptyonly);

    //mKurt = computeKurtosis(sig,nSig);
    //mKurtAvailable = true;
    //cout << "clusterTraces... " << get_decision_metrics() << endl << flush;

    /*
      // make PVT more robust instead checking kurtosis or other things
    if (kurt_too_high())
    {
        cout << "clusterTraces... kurtosis too high, aborting and use avgTrace_old" << get_decision_metrics() << endl << flush;
        return (0);
    }
    */

    // 1.2 use the thresholding methods to find signal threshold for true empties
    switch (mModel)
    {
    case 2:
        mThreshold = findThreshold(sig,nSig,"BMT");
        break;
    case 1:
    default:
        mThreshold = findThreshold(sig,nSig,"PVT",emptyonly);
        break;
    }


    averageTrueEmpties(sig,nSig,mThreshold,emptyonly);

    return mNumTrueEmpties;
}



void TraceClassifier::copyToWorkTrace (float *tmp_shifted)
{
    // shift the background by DEFAULT_FRAME_TSHIFT frames automatically: must be non-negative
    // "tshift=DEFAULT_FRAME_TSHIFT compensates for this exactly"
    int kount = 0;
    for (int frame=DEFAULT_FRAME_TSHIFT;frame<imgFrames;frame++,kount++)
        workTrace[kount] = tmp_shifted[frame];
    // assume same value after shifting
    for (; kount<imgFrames; kount++)
        workTrace[kount] = workTrace[kount-1];
    /*
    for (int n=0; n<imgFrames-DEFAULT_FRAME_TSHIFT; n++)
        workTrace[n] = tmp_shifted[n];
    for (int n=imgFrames-DEFAULT_FRAME_TSHIFT; n<imgFrames; n++)
        workTrace[n] = workTrace[n-1];
    */
}


bool TraceClassifier::isEmptyWell(int ax, int ay)
{
    bool isRef = bfmask->Match (ax,ay,referenceMask);
    bool isIgnoreOrAmbig = bfmask->Match (ax,ay, (MaskType) (MaskIgnore));
    bool isUnpinned = ! (pinnedInFlow->IsPinned (flow, bfmask->ToIndex (ay, ax)));
    bool isEmpty = (isRef & isUnpinned & ~isIgnoreOrAmbig) ? true:false;
    return isEmpty;
}


void TraceClassifier::zeroWorkTrace()
{
    for (int n=0; n<imgFrames; n++)
    {
        trace_sum[n]= 0;
    }
}


void TraceClassifier::sumWorkTrace()
{
    for (int n=0; n<imgFrames; n++)
    {
        trace_sum[n] += workTrace[n];
    }
}

void TraceClassifier::hiloWorkTrace()
{
    for (int n=0; n<imgFrames; n++)
    {
        if (hi[n] < workTrace[n]) hi[n] = workTrace[n];
        if (lo[n] > workTrace[n]) lo[n] = workTrace[n];
    }
}


void TraceClassifier::loWorkTrace()
{
    for (int n=0; n<imgFrames; n++)
    {
        //if (hi[n] < workTrace[n]) hi[n] = workTrace[n];
        if (lo[n] > workTrace[n]) lo[n] = workTrace[n];
    }
}


void TraceClassifier::calcAvgTrace(int nTotal)
{
    assert(nTotal>0);

    float factor = 1.0 / float(nTotal);
    for (int n=0; n<imgFrames; n++)
    {
        avgTrace[n] = trace_sum[n] * factor;
    }
}


void TraceClassifier::setAvgTraceDefault()
{
    for (int n=0; n<imgFrames; n++)
    {
        avgTrace[n] = 0;
    }
}


void TraceClassifier::copyLowTrace()
{
    for (int n=0; n<imgFrames; n++)
    {
        avgTrace[n] = lo[n];
    }
}


void TraceClassifier::compute_dYY()
{
    assert(dYY.size()>0);

    mTotalLo = 0;
    for (int n=0; n<imgFrames; n++)
    {
        dYY[n] = hi[n] - lo[n];
        mTotalLo += dYY[n];
    }
}


void TraceClassifier::setPicks(bool flag)
{
    for (int n=0; n<imgFrames; n++)
        picks[n] = flag;
}


void TraceClassifier::findTopPicks()
{
    vector<float>v(imgFrames);
    for (int n=0; n<imgFrames; n++)
        v[n] = dYY[n];
    vector<int> index;
    sort_pair(v,index,false); // descending
    int nStop = int(imgFrames * pick_ratio);
    int nTotal = 0;
    int nStart = imgFrames;
    for (int n=0; n<imgFrames; n++)
    {
        if (dYY[index[n]] > 0)
        {
            picks[index[n]] = true;
            nTotal++;
            if (nTotal >= nStop)
            {
                nStart = n+1;
                break;
            }
        }
        else
            picks[index[n]] = false;
    }
    for (int n=nStart; n<imgFrames; n++)
        picks[index[n]] = false;

    assert(picks.size()>0);
    mTotalLo = 0;
    for (int n=0; n<imgFrames; n++)
    {
        if (picks[n])
            mTotalLo += lo[n];
    }
}



float TraceClassifier::compute_avgTrace_snr()
{
    assert(dYY.size()>0);
    vector<float> data;
    //float totalDiff = 0;
    for (int n=0; n<imgFrames; n++)
    {
        if (!picks[n])
            continue;

        float d = avgTrace[n] - lo[n];
        data.push_back(d);
        //totalDiff += d;
    }
    float mu = mean_picked(lo);
    float sd = stdev(data);
    mSNR = (sd!=0) ? mu/sd : 0;

    /*
    mSNR = compute_snr(data);
    float amp = totalDiff;
    if (mTotalLo>0)
        amp /= mTotalLo;
    mSNR *= amp;
    */
    return mSNR;
}


float TraceClassifier::compute_avgTrace_cv()
{
    assert(dYY.size()>0);
    vector<float> data;
    //float totalDiff = 0;
    for (int n=0; n<imgFrames; n++)
    {
        if (!picks[n])
            continue;
        float d =avgTrace[n] - lo[n];
        data.push_back(d);
        //totalDiff += d;
    }
    float mu = mean_picked(lo);
    float delta = mean(data);
    mDelta = (mu!=0) ? delta/mu : 0;
    float sd = stdev(data);
    mCV = (mu!=0) ? sd/mu : 0;
    float rms = rootmeansquare(data);
    mRMS = (mu!=0) ? rms/mu : 0;

    /*
    mCV = compute_cv(data);
    float amp = totalDiff;
    if (mTotalLo>0)
        amp /= mTotalLo;
    mCV *= amp;
    */
    return mCV;
}


void TraceClassifier::copy_ShiftTrace_To_WorkTrace(int ax, int ay, int iWell)
{
    float tmp[imgFrames];         // scratch space used to hold un-frame-compressed data before shifting it
    float tmp_shifted[imgFrames]; // scratch space used to time-shift data before averaging/re-compressing

    TraceHelper::GetUncompressedTrace (tmp,img,ax,ay,imgFrames);
    //cout << "done GetUncompressedTrace..." << endl << flush;

    if (t0_map!=NULL)
    {
        TraceHelper::SpecialShiftTrace (tmp,tmp_shifted,imgFrames,t0_map[iWell]);
        //cout << "done SpecialShiftTrace..." << endl << flush;
        copyToWorkTrace (tmp_shifted);
        //cout << "done copyToWorkTrace..." << endl << flush;
    }
    else
    {
        //cout << "Alert: t0_map nonexistent\n" << endl << flush;
        copyToWorkTrace (tmp);
        //cout << "done copyToWorkTrace..." << endl << flush;
    }
    //cout << "done copy_ShiftTrace_To_WorkTrace..." << endl << flush;
}


float TraceClassifier::regress_workTrace()
{
    for (int n=0; n<imgFrames; n++)
    {
        dy[n] = workTrace[n] - lo[n];
    }
    float slope = regression_slope(dy,dYY);
    return slope;
}



bool TraceClassifier::avgCluster(int HiLo, bool emptyonly)
{
    assert (mTrueEmpties.size() != 0);
    int nY = region->h;
    int nX = region->w;

    // averaging the true empty traces
    int iWell = 0;
    int iEmpty = 0;
    int nTotal = 0;
    zeroWorkTrace();
    for (int y=0;y<nY;y++)
    {
        int ay = region->row + y;
        for (int x=0;x<nX; x++)
        {
            int ax = region->col + x;
            if (!emptyonly || isEmptyWell(ax,ay))
            //if (emptyWells[iWell++]==true)
            {
                switch (HiLo)
                {
                case 1:
                    if (!mTrueEmpties[iEmpty])
                    {
                        copy_ShiftTrace_To_WorkTrace(ax,ay,iWell);
                        sumWorkTrace();
                        nTotal++;
                    }
                case 0:
                default:
                    if (mTrueEmpties[iEmpty])
                    {
                        copy_ShiftTrace_To_WorkTrace(ax,ay,iWell);
                        sumWorkTrace();
                        nTotal++;
                    }
                }
                iEmpty++;
            }
            iWell++;
        }
    }

    if (nTotal>0)
    {
        calcAvgTrace(nTotal);
        return true;
    }
    else
    {
        cerr << get_decision_metrics() << " Warning: no true emptie (or all empties for HiLo=1) wells found, use whatever in the workTrace is used as avgTrace" << endl << flush;
        copyToAvgTrace(avgTrace_old); // used the avgTrace from evalEmptyWells();
		//for (int frame=0;frame < imgFrames;frame++)
        //    avgTrace[frame] = workTrace[frame]; // just use what ever is in there and give an alert
        return false;
    }

}




void TraceClassifier::makeHistogram(float *sig, int nSig, int nBins)
{
    assert (mNumBins > 1);
    float v[nSig];
    memcpy(v,sig,sizeof(float[nSig]));
    qsort (v, nSig, sizeof (float), compareDescend);

    float vMax = v[0];
    float vMin = v[nSig-1];
    float delta = (vMax-vMin)/(mNumBins-1);
    for (int n=0; n<mNumBins; n++)
    {
        bins[n] = vMin + delta * n;
        hist[n] = 0;
    }

    for (int i=0; i<nSig; i++)
    {
        for (int n=0; n<mNumBins; n++)
            if (sig[i] <= bins[n])
            {
                hist[n]++;
                break;
            }
        if (sig[i] > bins[mNumBins-1])
            hist[mNumBins-1]++;
    }

    // cumhist
    cumhist[0] = hist[0];
    int total = hist[0];
    for (int n=1; n<mNumBins; n++)
    {
        cumhist[n] = cumhist[n-1] + hist[n];
        total += hist[n];
    }
    float factor = 1.0 / float(total);
    for (int n=0; n<mNumBins; n++)
    {
        cumhist[n] *= factor;
        //cout << "cumhist[" << n << "]=" << cumhist[n] << endl << flush;
    }
}



float TraceClassifier::findThreshold(float *sig, int nSig, char* method, bool emptyonly)
{
    //if (nSig <=0 )        return mThreshold;
    if (nSig < mNumBins)        return mThreshold;

    // make histogram from sig[]
    makeHistogram(sig,nSig,mNumBins);

    // PVT method to find the threshold
    if (strcmp (method, "PVT") == 0)
    {
        mThresholdMethod = "PVT";
        threshold_PVT_method(emptyonly);
    }
    else
    {
        mThresholdMethod = "BMT";
        threshold_BMT_method();
    }

    return mThreshold;
}



int TraceClassifier::threshold_BMT_method(int order)
{
    // Balanced Moment Thresholding
    // Eugene Wang's own method

    int iL = -1;
    int iR = mNumBins;
    int iMid = iR/2;
    float wR = moment(hist,iMid,iR,order);
    float wL = moment(hist,iMid,iL,order);
    float preDiff = fabs(wR-wL);

    while (wL != wR)
    {
        if (wL > wR)
        {
            iMid--;
            if (iMid <=0)
            {
                iMid = 0;
                break;
            }
            else
            {
                wR = moment(hist,iMid,iR,order);
                wL = moment(hist,iMid,iL,order);
                float curDiff = fabs(wR-wL);
                if (wL<wR)
                {
                    if (preDiff < curDiff)
                        iMid = iMid+1;
                    break;
                }
                else
                    preDiff = curDiff;
            }
        }
        else
        {
            iMid++;
            if (iMid >= iR)
                iMid = mNumBins-1;
            else
            {
                wR = moment(hist,iMid,iR,order);
                wL = moment(hist,iMid,iL,order);
                float curDiff = fabs(wR-wL);
                if (wL>wR)
                {
                    if (preDiff < curDiff)
                        iMid = iMid-1;
                    break;
                }
                else
                    preDiff = curDiff;
            }
        }
    }

    // move to previous valley
    //iMid = findLeadingValley(iMid);

    // make sure that cumhist[mid] > mCumHist_min
    iMid = passMinCDF(iMid);

    string msg = "threshold bin at " + val2str(iMid) + " for hist:";
    cout << formatVector(hist, msg) << endl << flush;

    assert (iMid >= 0);
    assert (iMid < mNumBins);
    mThreshIdx = iMid;
    mThreshold = bins[iMid];
    return iMid;
}


int TraceClassifier::threshold_PVT_method(bool emptyonly)
{
    vector<int> peaks;
    vector<int> newPeaks;
    vector<int> valleys;
    vector<int> newValleys;
    vector<int> finalPeaks;
    vector<int> finalValleys;


    int mid = mNumBins / 2;
    findLocalMin(hist,valleys);
    mValleys = valleys;
    int nValleys = valleys.size();

    if (nValleys <= 0)
        mid = mNumBins-1 ;
    //else if (nValleys == 1)
    //    mid = valleys[0];
    else
    {
        findLocalMax(hist,peaks);
        removePeakZero(peaks);
        //cerr << formatPeakHist(peaks,flowRegionStr()+" peaks before reducePeaks") << endl << flush;

        mPeaks = peaks;
        findGaps(peaks,mGaps);
        int nPeaks = peaks.size();
        while (nPeaks > mMaxNumPeaks)
        {
            //cerr << flowRegionStr() << formatPeakHist(peaks,"peaks before reducePeaks") << endl << flush;
            //mergePeaks(peaks,newPeaks);
            reducePeaks(peaks,newPeaks,valleys,newValleys);
            //cerr << flowRegionStr() << formatPeakHist(newPeaks,"peaks after reducePeaks") << endl << flush;

            int nPeaksNew = newPeaks.size();
            //assert(nPeaksNew <= nPeaks);


            if (nPeaksNew < mMaxNumPeaks)
            {
                cout << "reducePeaks reduced peaks to < 2" << endl << flush;
                cout << formatPeakHist(peaks,"peaks before reducePeaks") << endl << flush;
                cout << formatPeakHist(newPeaks,"peaks after reducePeaks") << endl << flush;
                break;
            }
            else if (nPeaksNew >= nPeaks)
            {
                //cout << "reducePeaks cannot reduce peaks???" << endl << flush;
                //cout << formatPeakHist(peaks,"peaks before reducePeaks") << endl << flush;
                //cout << formatPeakHist(newPeaks,"peaks after reducePeaks") << endl << flush;
                break;
            }
            else
            {
                peaks = newPeaks;
                valleys = newValleys;
                nPeaks = peaks.size();
                //if (nPeaksNew==mMaxNumPeaks)                    break;
            }
            //cerr << formatPeakHist(peaks,flowRegionStr()+" peaks after reducePeaks") << endl << flush;
        }

        //nPeaks = peaks.size(); //?? is this necessary??
        if (nPeaks==0)
            mid = mNumBins-1;
        else
        {
            //sort_peaks_by_height(peaks, finalPeaks, false); // descending by hist
            //mid = findValley_FWHM(valleys,finalPeaks,emptyonly);
            finalPeaks = peaks;
            mid = findValley_FWHM(valleys,peaks,emptyonly);
            //mid = findFirstGround(finalPeaks);
            //mid = findFirstValley(valleys,finalPeaks,finalValleys);
            //mid = findCenterValley(valleys,finalPeaks,finalValleys);
            //mid = findLastValley(valleys,finalPeaks,finalValleys);
            //cout << flowRegionStr() << "threshold_PVT_method... theValley=" << mid << " hist=" << hist[mid] << formatVector(hist," hist:") << formatVector(finalPeaks," finalPeaks:") << endl << flush;
            //cout << flowRegionStr() << formatVector(finalValleys,"finalValleys after findValley:") << formatVector(finalPeaks," finalPeaks:") << endl << flush;
        }
    }

    // make sure that cumhist[mid] > mCumHist_min
    //mid = passMinCDF(mid,finalValleys); // moved to findValley_FWHW()

    string msg = " threshold=" + val2str(bins[mid]) + " at bin " + val2str(mid) + " hist=" +  val2str(hist[mid]) + ":";
    msg = flowRegionStr() + formatVector(finalPeaks," threshold_PVT_method finalPeaks:") + formatVector(hist, msg);
    cerr <<  msg << endl << flush;

    assert (mid >= 0);
    assert (mid < mNumBins);
    mThreshIdx = mid;
    mThreshold = bins[mid];
    return mid;
}


template<class T>
void TraceClassifier::findLocalMin(const vector<T>& v, vector<int>& valleys)
{
    int nBins = v.size();
    valleys.resize(0);
    if (nBins<=1)
        return;

    float vMax = v[0];
    bool wasHigher = false;

    for (int i=0; i<nBins; i++)
    {
        if (v[i] == v[i-1])
            continue;
        if (v[i] > v[i-1])
        {
            if (wasHigher)
            {
                valleys.push_back(i-1);
                vMax = v[i];
                wasHigher = false;
            }
            if (v[i] > vMax)
                vMax = v[i];
        }
        else
            wasHigher = true;
    }

}


template<class T>
void TraceClassifier::findLocalMax(const vector<T>& v, vector<int>& peaks)
{
    int nBins = v.size();
    peaks.resize(0);
    if (nBins<=1)
        return;

    float vMin = v[0];
    bool wasLower = true;

    for (int i=0; i<nBins; i++)
    {
        if (v[i] == v[i-1])
            continue;
        if (v[i] < v[i-1])
        {
            if (wasLower)
            {
                peaks.push_back(i-1);
                vMin = v[i];
                wasLower = false;
            }
            if (v[i] < vMin)
                vMin = v[i];
        }
        else
            wasLower = true;
    }
    if (wasLower)
        peaks.push_back(nBins-1);
}



int TraceClassifier::reducePeaks(const vector<int>& peaks, vector<int>& newPeaks, vector<int>& valleys, vector<int>& newValleys)
{
    int nPeaks = peaks.size();
    if (nPeaks < mMinFinalPeaks)
    {
        newPeaks = peaks;
        return nPeaks;
    }

    float minFs = computeMinFs(peaks,valleys);
    //cerr << formatVector(mFstat,flowRegionStr()+" reducePeaks: minFs="+val2str(minFs)+" mFstat:") << endl << flush;

    if (fstat_too_small(minFs))
    {
        removePeak_lowestFs(peaks,newPeaks,minFs,valleys,newValleys);
        nPeaks = newPeaks.size();
    }
    else
    {
        newPeaks = peaks;
        newValleys = valleys;
        return nPeaks;
    }

    return nPeaks;
}


int TraceClassifier::mergePeaks(const vector<int>& peaks, vector<int>& newPeaks)
{
    int nPeaks = peaks.size();
    if (nPeaks < mMinFinalPeaks)
    {
        newPeaks = peaks;
        return nPeaks;
    }

    //cout << "mergePeaks merging " << nPeaks << " peaks" << endl << flush;
    cerr << formatVector(peaks,"before mergePeaks:") << endl << flush;

    sort_peaks_by_height(peaks,newPeaks,false); // descending by hist
    //cout << formatPeakHist(newPeaks,"after sort_peaks_by_height:") << endl << flush;

    if (nPeaks == 3)
    {
        // find the one with the least moment from the reference/highest peak
        int refPeak = newPeaks[0];
        int d1 = abs(newPeaks[1] - refPeak);
        int d2 = abs(newPeaks[2] - refPeak);
        float m1 = hist[newPeaks[1]] * d1;
        float m2 = hist[newPeaks[2]] * d2;
        m1 *= d1;
        int idx2go = (m1<m2) ? 1 : 2;

        //cout << "erasing " << idx2go << endl << flush;
        //cout << formatPeakHist(newPeaks,"before newPeaks.erase:") << endl << flush;
        newPeaks.erase(newPeaks.begin() + idx2go);
        //cout << formatPeakHist(newPeaks,"after newPeaks.erase:") << endl << flush;
        return newPeaks.size();
    }

    vector<bool> available(nPeaks);
    vector<bool> kept(nPeaks);

    for (int n=0; n<nPeaks; n++)
    {
        available[n] = (newPeaks[n] == 0) ? false : true;
        kept[n] = false;
    }


    for (int n=0; n<nPeaks; n++)
    {
        if (available[n])
        {
            vector <int> neighbors;
            int refIdx = n;
            findNeighborsToElimiate(newPeaks,available,refIdx,neighbors);
            cerr << "findNeighborsToElimiate found "+val2str(neighbors.size())+" neighbors" << endl << flush;
            int togo = neighbors[0];
            int keep = neighbors[1];
            //cout << "togo=" << togo << ", keep=" << keep << endl << flush;

            if (togo>=0)
            {
                // process togo
                assert(togo<nPeaks);
                if (kept[togo])
                    continue;
                else
                    available[togo] = false;

                // process keep, only when togo >= 0
                if (keep != -1 && !kept[keep])
                    {
                        kept[keep] = true;
                        available[keep] = false;
                    }
            }
            else
            {
                // process (newPeaks[n] itself)
                kept[n] = true;
                available[n] = false;
            }
         }
    }

    vector<int> peaks2keep(0);
    for (int n=0; n<nPeaks; n++)
    {
        if (kept[n])
        {
            peaks2keep.push_back(newPeaks[n]);
        }
    }

    newPeaks = peaks2keep;
    nPeaks = newPeaks.size();
    cerr << formatVector(newPeaks,"after mergePeaks:") << endl << flush;
    return nPeaks;
}


int TraceClassifier::findFirstGround(const vector<int>& finalPeaks)
{
    int theValley = mNumBins-1;
    if (finalPeaks.size() < 1)
        return theValley;

    int n1 = finalPeaks[0];
    int n2 = mNumBins-1;
    if (finalPeaks.size() >= 2)
        n2 = finalPeaks[1];

    vector<int> index;
    findLowestHist(n1,n2,index);
    theValley = index[0];

    return theValley;
}



int TraceClassifier::findValley_FWHM(const vector<int>& valleys, const vector<int>& finalPeaks, bool emptyonly)
{
    int theValley = mNumBins-1;
    if (finalPeaks.size() < 1)
        return theValley;

    vector<int> finalValleys;
    findFinalValleys(finalPeaks,valleys,finalValleys); // finalVallyes.size()==finalPeaks.size()
    mFinalValleys = finalValleys;
    if (finalValleys.size() <= 0)
        return theValley;

    if (emptyonly==false) // all traces in the well
    {
        theValley = finalValleys[0]; // last finalValleys
        cout << flowRegionStr() << " findValley_FWHM(false)...theValley=" << theValley << endl << flush;
        return theValley;
    }

    //findGaps(finalPeaks,mGaps);
    int nValleys = finalValleys.size();
    //if ((nValleys == 3 && mNumEmptyWells>=mNumEmptyWells_min3) ||
    //(nValleys == 2 && mNumEmptyWells>=mNumEmptyWells_min2) ||
    // (nValleys == 1 && mNumEmptyWells>=mNumEmptyWells_min))
    if (useFinalValley(nValleys))
    {
        if (mModel==0)
            theValley = finalValleys[0];
        else
        {
            theValley = finalValleys[nValleys-1];
            if (nValleys>=2 && theValley==finalPeaks[nValleys-1])
                theValley = finalValleys[nValleys-2];
        }
        //theValley = finalValleys[0];
        //theValley = mValleys[mValleys.size()-1]; // last valley
        //theValley = mNumBins-1; // off.1
        //theValley = passMinCDF(theValley,mFinalValleys); // moved to findValley_FWHW()
        //cout << flowRegionStr() << " findValley_FWHM...theValley=" << theValley << formatVector(finalValleys," finalValleys:") << endl << flush;
    }
    /*
    else if (getLastGapSize() >= 2)
    {
        theValley = mValleys[mValleys.size()-1]; // last valley
        //theValley = finalValleys[finalValleys.size() - 1]; // last finalValleys
    }

    */
    else
    {
        //theValley = findLowerHist(n1,n2,maxHeight);
        //vector<int> index;
        //findLowestHist(n1,n2,index);
        //theValley = index[0];
        //theValley = valleys[valleys.size()-1];
        theValley = mNumBins-1; // off, use all of the empty traces
        //cerr << flowRegionStr()+" findValley_FWHM...lastValley="+val2str(theValley) << formatVector(valleys," valleys:") << formatVector(finalPeaks," finalPeaks:") << formatVector(hist," hist:") << endl << flush;
    }

    //cerr << flowRegionStr()+" findValley_FWHM... nValleys="+val2str(nValleys)+" theValley="+val2str(theValley)+" mNumEmptyWells="+val2str(mNumEmptyWells) << endl << flush;
    return theValley;
}



int TraceClassifier::findFirstValley(const vector<int>& valleys, int peak)
{
    int theValley = mNumBins-1;
    int nValleys = valleys.size();
    for (int i=0; i<nValleys; i++)
    {
        if (valleys[i] > peak)
        {
            theValley = valleys[i];
            break;
        }
    }
    return theValley;
}



int TraceClassifier::findFirstValley(const vector<int>& valleys, const vector<int>& finalPeaks, vector<int>& finalValleys)
{
    int theValley = mNumBins-1;
    if (finalPeaks.size() < 1)
        return theValley;

    int n1 = finalPeaks[0];
    int n2 = mNumBins-1;
    if (finalPeaks.size() >= 2)
        n2 = finalPeaks[1];

    finalValleys.resize(0);

    int nValleys = valleys.size();
    for (int i=0; i<nValleys; i++)
    {
        if (isBetween(valleys[i],n1,n2))
        {
            //cout << "found valley " << valleys[i] << " betweeen peaks " << n1 << " & " << n2 << endl << flush;
            finalValleys.push_back(valleys[i]);
        }
    }

    if (finalValleys.size() > 0)
        theValley = finalValleys[0]; // first valley

    return theValley;
}


int TraceClassifier::findCenterValley(const vector<int>& valleys, const vector<int>& finalPeaks, vector<int>& finalValleys)
{
    int theValley = (mNumBins)/2;
    int nPeaks = finalPeaks.size();
    if (nPeaks == 0)
        return theValley;
    else if (nPeaks == 1)
        return finalPeaks[0];

    int n1 = finalPeaks[0];
    int n2 = finalPeaks[1];

    finalValleys.resize(0);

    int nValleys = valleys.size();
    for (int i=0; i<nValleys; i++)
    {
        if (isBetween(valleys[i],n1,n2))
        {
            //cout << "found valley " << valleys[i] << " betweeen peaks " << n1 << " & " << n2 << endl << flush;
            finalValleys.push_back(valleys[i]);
        }
    }

    if (finalValleys.size() == 1)
        theValley = finalValleys[0];
    else
        theValley = finalValleys[finalValleys.size()/2]; // center valley

    return theValley;
}



int TraceClassifier::findLastValley(const vector<int>& valleys, const vector<int>& finalPeaks, vector<int>& finalValleys)
{
    int theValley = (mNumBins)/2;
    int nPeaks = finalPeaks.size();
    if (nPeaks == 0)
        return theValley;
    else if (nPeaks == 1)
        return finalPeaks[0];

    int n1 = finalPeaks[0];
    int n2 = finalPeaks[1];

    finalValleys.resize(0);

    int nValleys = valleys.size();
    for (int i=0; i<nValleys; i++)
    {
        if (isBetween(valleys[i],n1,n2))
        {
            //cout << "found valley " << valleys[i] << " betweeen peaks " << n1 << " & " << n2 << endl << flush;
            finalValleys.push_back(valleys[i]);
        }
    }

    if (finalValleys.size() == 1)
        theValley = finalValleys[0];
    else
        theValley = finalValleys[finalValleys.size()-1]; // last valley

    return theValley;
}



void TraceClassifier::findNeighborsToElimiate(const vector<int>& peaks, vector<bool>& available, int refIdx, vector<int>& neighbors)
{
    int nPeaks = peaks.size();

    vector<pair<int,int> > myPair;
    pair<int,int> pp;

    for (int i=0; i<nPeaks; i++)
        if (available[i] && peaks[i]!=refIdx)
        {
            int dist = abs(peaks[i]-peaks[refIdx]);
            pp.first = dist;
            pp.second = i; // the index, not the peak here!!!
            myPair.push_back(pp);
        }

    neighbors.resize(myPair.size());
    if (myPair.size() >= 1)
    {
        sort(myPair.begin(),myPair.end(),cmpPairAsend);
        for(size_t i = 0; i < neighbors.size(); ++i) neighbors[i] = myPair[i].second;
    }

    if (myPair.size() == 1)
    {
        neighbors.push_back(refIdx);
    }
    else if (myPair.size() == 0)
    {
        neighbors.push_back(-1);
        neighbors.push_back(-1);
    }
    /*
    else
    {
        // re-sort the two closeast neighbors by moment?? was by distance, good enough for now??
        // neighbors = leastMoment(peaks[refIdx],peaks[neighbors[:2]],hist);
        int nb = neighbors[0];
        if (hist[peaks[nb]] > hist[peaks[neighbors[refIdx]]])
        {
            neighbors.resize(2);
            neighbors[0] = refIdx;
            neighbors[1] = nb;
        }
        else if (hist[peaks[nb]] == hist[peaks[neighbors[refIdx]]])
        {
            vector<int> tmpVec(3);
            tmpVec[0] = peaks[nb];
            tmpVec[1] = peaks[neighbors[1]];
            tmpVec[2] = peaks[refIdx];
            int mid = middle(tmpVec);
            neighbors.resize(2);
            if (mid==peaks[refIdx])
            {
                neighbors[0] = refIdx;
                neighbors[1] = nb;
            }
            else
            {
                neighbors[0] = (mid==peaks[nb]) ? nb : neighbors[1];
                neighbors[1] = refIdx;
            }
        }
    }
    */

}




/*

void TraceClassifier::copySingleTrace (int offset, float *tmp_shifted)
{
    // shift the background by DEFAULT_FRAME_TSHIFT frames automatically: must be non-negative
    // "tshift=DEFAULT_FRAME_TSHIFT compensates for this exactly"
    int kount = offset;
    for (int frame=DEFAULT_FRAME_TSHIFT;frame<imgFrames;frame++)
    {
        emptyTraces[kount++] = tmp_shifted[frame];
    }
    // assume same value after shifting
    for (int n=0; n<DEFAULT_FRAME_TSHIFT; n++)
    {
        emptyTraces[kount] = emptyTraces[kount-1];
        kount++;
    }
}


void TraceClassifier::copyAllEmptyTraces()
{
    int nY = region->h;
    int nX = region->w;
    mNumWellsInRegion = nX * nY;

    // note: memory problem in multi-threading for each flow. Too much memory required???
    if (emptyTraces!=NULL) delete[] emptyTraces;
    cout << "copyAllEmptyTraces allocating memory " << mNumEmptyWells*imgFrames*sizeof(float) << endl << flush;
    emptyTraces = new float [mNumEmptyWells*imgFrames];
    assert (emptyTraces!=NULL);
    memset(emptyTraces,0,sizeof(float[mNumEmptyWells*imgFrames]));

    //if (emptyTraces.size() != (size_t) mNumWellsInRegion*imgFrames)
    //{
    //    cout << " emptyTraces.resize(" << mNumWellsInRegion*imgFrames << ")" << endl << flush;
    //    emptyTraces.resize(mNumWellsInRegion*imgFrames);
    //}
    //cout << " emptyTraces.size=" << emptyTraces.size() << endl << flush;
    //for (int n=0; n<(int)emptyTraces.size(); n++)
    //    emptyTraces[n] = 0;

    int iEmpty = 0;

    int iWell = 0;
    for (int y=0;y<nY;y++)
    {
        int ay = region->row + y;
        for (int x=0;x<nX; x++)
        {
            int ax = region->col + x;
            if (isEmptyWell(ax,ay))
            //if (emptyWells[iWell++]==true)
            {
                copy_ShiftTrace_To_WorkTrace(ax,ay,iWell);
                iEmpty++;
            }
            iWell++;
        }
    }

    mNumEmptyWells = iEmpty;
    //cout << "copyAllEmptyTraces copied " << iEmpty << " empty traces" << endl;

}


void TraceClassifier::get_HiLo_fromEmptyWells()
{
    float v[mNumEmptyWells];

    for (int x=0; x<imgFrames; x++)
    {
        for (int y=0; y<mNumEmptyWells; y++)
            v[y] = emptyTraces[y*imgFrames+x];
        // sort v to find the max/min
        try
        {
            qsort (v, mNumEmptyWells, sizeof (float), compareDescend);
        }
        catch (...)
        {
            cerr << "Error in qsort()" << endl << flush;
            exit(1);
        }

        hi[x] = v[0];
        lo[x] = v[mNumEmptyWells-1];
        dYY[x] = hi[x] - lo[x];
        //cout << "dYY: " << x << "\t" << hi[x] << "\t" << lo[x] << "\t" << dYY[x] << endl << flush;
    }
}



*/



string TraceClassifier::val2str(int val)
{
    stringstream ss;
    ss << val;
    return ss.str();
}


string TraceClassifier::val2str(size_t val)
{
    stringstream ss;
    ss << val;
    return ss.str();
}


string TraceClassifier::val2str(float val)
{
    stringstream ss;
    ss << val;
    return ss.str();
}


string TraceClassifier::formatPeakHist(const vector<int>& peaks, string msg)
{
    string s = formatVector(peaks,msg);
    stringstream ss;
    ss << ", histogram:";
    //int nPeaks = peaks.size();
    //for (int n=0; n<nPeaks; n++)
    //    ss << " " << hist[peaks[n]];
    for (int n=0; n<mNumBins; n++)
        ss << " " << hist[n];
    s += ss.str();
    return s;
}



void TraceClassifier::sort_peaks_by_height(const vector<int> & peaks, vector<int> & sortedPeaks, bool ascending)
{
    int nPeaks = peaks.size();
    vector<float>heights(nPeaks);
    for (int n=0; n<nPeaks; n++)
    {
        heights[n] = hist[peaks[n]];
    }

    vector<int> index;
    sort_pair(heights,index,ascending);

    sortedPeaks.resize(nPeaks);
    for (int n=0; n<nPeaks; n++)
    {
        sortedPeaks[n] = peaks[index[n]];
    }
}


int TraceClassifier::cluster_regression(float *sig, bool emptyonly)
{
    int iWell = 0;
    int iEmpty = 0;
    int nY = region->h;
    int nX = region->w;
    for (int y=0;y<nY;y++)
    {
        int ay = region->row + y;
        for (int x=0;x<nX; x++)
        {
            int ax = region->col + x;
            if (!emptyonly || isEmptyWell(ax,ay))
                //if (emptyWells[iWell]==true)
            {
                //cout << "copy_ShiftTrace_To_WorkTrace iWell=" << iWell << endl << flush;
                copy_ShiftTrace_To_WorkTrace(ax,ay,iWell);
                //if (isNoiseTrace()) continue;

                sig[iEmpty] = regress_workTrace();
                //cout << "regress_workTrace sig=" << sig[iEmpty] << endl << flush;
                iEmpty++;
            }
            iWell++;
        }
    }

    if (emptyonly)
    {
    if (mNumEmptyWells<=0)
        mNumEmptyWells = iEmpty;
    else
        assert(iEmpty==mNumEmptyWells);
    }
    return iEmpty;
}



int TraceClassifier::averageTrueEmpties(float *sig, int nSig, float thresh, bool emptyonly)
{
    // set trueEmpties for the ones < thresh
    int nTrueEmpties = setTrueEmpties(sig,nSig,thresh);
    mNumTrueEmpties = emptyonly ? nTrueEmpties : 0;
    mTrueRatio = float(nTrueEmpties) / nSig;
    mTruePercent = int(mTrueRatio*100);
    string TrueFalse = emptyonly ? " true empty wells " : " false empty wells ";
    cout << "computeClusterMean: nTrueEmpties=" << nTrueEmpties << " out of " << nSig << TrueFalse << get_decision_metrics() << endl << flush;
    cout << "thresh=" << thresh << endl << flush;

    //cout << "averageTrueEmpties: thresh=" << thresh << ", mNumTrueEmpties=" << mNumTrueEmpties << " out of " << mNumEmptyWells << endl << flush;
    ///--------------------------------------------------------------------------------------------------------
    /// check to see whether there's enough trueEmpties found
    ///--------------------------------------------------------------------------------------------------------
    if (hasNoEmpties(nTrueEmpties,nSig) || !hasEnoughEmpties(nTrueEmpties,nSig))
    {
        cout << "computeClusterMean warning: nTrueEmpties=" << nTrueEmpties << " out of " << nSig << TrueFalse << get_decision_metrics() << endl << flush;
        //for (int n=0; n<mNumEmptyWells; n++)
        //    cout << "sig[" << n << "]=" << sig[n] << " " << trueEmpties[n] << endl << flush;
        copyToAvgTrace(avgTrace_old); // used the avgTrace from evalEmptyWells();
    }
    else
        avgCluster(mHiLo,emptyonly);

    return (mNumTrueEmpties);
}



bool TraceClassifier::hasNoEmpties(int nTrueEmpties, int nEmpties)
{
    bool flag = ((nTrueEmpties<=0 && mHiLo==0) || (nTrueEmpties>=nEmpties && mHiLo==1)) ? true:false;
    return flag;
}


bool TraceClassifier::hasEnoughEmpties(int nTrueEmpties, int nEmpties)
{
    mTrueRatio = float(nTrueEmpties) / nEmpties;
    mTruePercent = int(mTrueRatio*100);

    bool enough = false;
    if (mHiLo==0)
        enough = (mTruePercent<mTruePercent_min || nTrueEmpties<mNumTrueEmpties_min) ? false:true;
    else
        enough = ( (mTruePercent>(100-mTruePercent_min)) || (nTrueEmpties>(nEmpties-mNumTrueEmpties_min)) ) ? false:true;
    return enough;
}


int TraceClassifier::passMinCDF(int mid)
{
    if (mid<0)
        mid=0;

    while (true)
    {
        if (cumhist[mid]>mCumHist_min || mid >= mNumBins-1)
            break;
        else
            mid++;
    }
    return mid;
}


int TraceClassifier::passMinCDF(int mid, vector<int>& valleys)
{
    if (mid>=mNumBins)
        return (mNumBins-1);

    if (mid<0)
        mid=0;

    int nValleys = valleys.size();
    for (int n=0; n<nValleys; n++)
    {
        int v = valleys[n];
        if (v < mid)
            continue;
        if (cumhist[v]>mCumHist_min || v >= mNumBins-1)
        {
            mid = v;
            break;
        }
    }
    return mid;
}


bool TraceClassifier::isNoiseTrace()
{
    bool ret = false;
    float sigMax = max(workTrace);
    float sigMin = min(workTrace);
    float diff = sigMax - sigMin;
    if (diff < mNoiseMax)
        ret = true;
    else
    {
        int iFrame = 0;
        for (int n=imgFrames-1; n>0; n--)
        {
            if (workTrace[n] >= sigMax)
            {
                iFrame = n;
                break;
            }
        }
        if (imgFrames - iFrame < mSigFrame_tail_min)
            ret = true;
    }
    return ret;
}


int TraceClassifier::generateAvgTrace(bool emptyonly)
{
    int nY = region->h;
    int nX = region->w;

    int numEmptyWells = 0;
    int iWell = 0;
    //mNoiseTraces.resize(0);

    for (int y=0;y<nY;y++)
    {
        int ay = region->row + y;
        for (int x=0;x<nX; x++)
        {
            int ax = region->col + x;
            if (!emptyonly || isEmptyWell(ax,ay))
            {
                //emptyWells[iWell] = 1; //1=IS_EMPTY, 2=TRUE_EMPTY
                copy_ShiftTrace_To_WorkTrace(ax,ay,iWell);
                //bool isNoise = isNoiseTrace()?true:false;
                //mNoiseTraces.push_back(isNoise);
                //if (isNoise) continue;

                sumWorkTrace();
                hiloWorkTrace();
                numEmptyWells++;
            }
            iWell++;
        }
    }
    if (numEmptyWells > 0)
    {
        compute_dYY();
        calcAvgTrace(numEmptyWells);
    }
    else
        setAvgTraceDefault();

    return numEmptyWells;
}



int TraceClassifier::generateLowTrace()
{
    int nY = region->h;
    int nX = region->w;

    //int numEmptyWells = 0;
    int iWell = 0;
    for (int y=0;y<nY;y++)
    {
        int ay = region->row + y;
        for (int x=0;x<nX; x++)
        {
            int ax = region->col + x;
            //if (!emptyonly || isEmptyWell(ax,ay))
            {
                //emptyWells[iWell] = 1; //1=IS_EMPTY, 2=TRUE_EMPTY
                copy_ShiftTrace_To_WorkTrace(ax,ay,iWell);
                //sumWorkTrace();
                //hiloWorkTrace();
                 loWorkTrace();
                //numEmptyWells++;
            }
            iWell++;
        }
    }
    copyLowTrace();

    return iWell;
}




float TraceClassifier::mean_picked(vector<float>& data)
{
    int nPicked = 0;
    vector<float> dd;
    for (int n=0; n<imgFrames; n++)
    {
        if (!picks[n])
            continue;
        dd.push_back(data[n]);
        nPicked++;
    }
    float retVal = (nPicked>0) ? mean(dd) : 0;
    return retVal;
}



string TraceClassifier::flowRegionStr()
{
    stringstream ss;
    ss << " flow=" << flow << " region(" << region->row << "," << region->col << ")";
    return ss.str();
}



string TraceClassifier::get_decision_metrics()
{
    stringstream ss;
    ss << flowRegionStr() << " mNumWellsInRegion=" << mNumWellsInRegion << " mNumEmptyWells=" << mNumEmptyWells << " mDoAllFlows=" << mDoAllFlows << " delta=" << get_delta() << " rms=" << get_rms() << " cv=" << get_cv();
    if (mKurtAvailable)
        ss << " kurtosis=" << get_kurtosis();

    return ss.str();
}



void TraceClassifier::removePeak_lowestFs(const vector<int>& peaks, vector<int>& newPeaks, float minFs, const vector<int>& valleys, vector<int>& newValleys)
{
    assert(mFstat.size() == peaks.size());
    int nPeaks = peaks.size();
    vector<int> candidates(0);
    for (int n=0; n<nPeaks; n++)
    {
        if (mFstat[n] == minFs)
            candidates.push_back(peaks[n]);
    }
    if (candidates.size()>0)
    {
        int pk2remove = candidates[0];
        if (candidates.size()>1)
        {
            if (hist[candidates[1]] < hist[candidates[0]])
                pk2remove = candidates[1];
        }
        newPeaks.resize(0);
        newValleys.resize(0);
        //int idx2remove = 0;
        for (int n=0; n<nPeaks; n++)
        {
            if (peaks[n] == pk2remove)
            {
                //idx2remove = n;
                //cout << "removePeak_lowestFs...skipping peak " << pk2remove << endl << flush;
                continue;
            }
            newPeaks.push_back(peaks[n]);

            int v = findFirstValley(valleys,peaks[n]); // better than findLastValley()
            /*
            int n2 = mNumBins - 1;
            if (n<nPeaks-1)
                n2 = n+1;
            int v = findLastValley(valleys,peaks[n],n2); // find the valley after the peak???
            */
            newValleys.push_back(v);
        }

        // remove the merged peak???


    }
    else
    {
        cout << "removePeak_lowestFs...found not candidates to remove" << endl << flush;
        newPeaks = peaks;

        // find new valleys here?? or just copy valleys
        newValleys.resize(0);
        for (int n=0; n<nPeaks; n++)
        {
            int v = findFirstValley(valleys,peaks[n]); // better than findLastValley()
            /*
            int n2 = mNumBins - 1;
            if (n<nPeaks-1)
                n2 = n+1;
            int v = findLastValley(valleys,peaks[n],n2); // find the valley after the peak???
            */

            newValleys.push_back(v);
        }

    }
    //cout << "remove_lowestFs..." << formatVector(peaks," peaks:") << formatVector(newPeaks," newPeaks:") << endl << flush;
}



float TraceClassifier::anova(const vector<int>& boarders1, const vector<int>& boarders2)
{
    vector<int> h(0);

    // within group variance
    for (int n=boarders1[0]; n<=boarders1[1]; n++)
        h.push_back(hist[n]);
    float mass1 = sum(h);
    float sse1 = computeVariance(h);
    if (sse1==0)
        sse1 = 0.5*0.5*mass1;

    h.resize(0);
    for (int n=boarders2[0]; n<=boarders2[1]; n++)
        h.push_back(hist[n]);
    float mass2 = sum(h);
    float sse2 = computeVariance(h);
    if (sse2==0)
        sse2 = 0.5*0.5*mass2;
    float sse = sse1 + sse2;

    // total variance
    int left = min(boarders1[0],boarders2[0]);
    int right = max(boarders1[1],boarders2[1]);
    h.resize(0);
    for (int n=left; n<=right; n++)
        h.push_back(hist[n]);
    float mass = sum(h);
    float ss = computeVariance(h);
    if (ss==0)
        ss = 0.5*0.5*mass;

    float fstat = (sse>0) ? ss/sse : 999;
    return fstat;
}



float TraceClassifier::computeVariance(const vector<int>& v)
{
    int nV = v.size();
    if (nV <= 1)
        return 0;

    float mass = 0;
    float mu = 0;
    float ss = 0;
    for (int n=0; n<nV; n++)
    {
        mass += v[n];
        mu += n*v[n];
        ss += n*n*v[n];
    }
    if (mass <= 0)
        return 0;

    mu /= mass;
    ss /= mass;
    ss -= mu*mu;

    return ss;
}


void TraceClassifier::removePeakZero(vector<int>& peaks)
{
    int nPeaks = peaks.size();
    for (int n=0; n<nPeaks; n++)
    {
        if (peaks[n] == 0)
        {
            peaks.erase(peaks.begin()+n);
            break;
        }
    }
}



int TraceClassifier::findLowestHist(int n1, int n2, vector<int>& index)
{
    int nHist = hist.size();
    if (n1 >= nHist)
        n1 = nHist-1;
    if (n2 >= nHist)
        n2 = nHist-1;
    if (n1>n2)
        swap(n1,n2);

    int lowest = hist[n1];
    index.resize(0);
    if (n1==n2)
        index.push_back(n1);
    else
    {
        vector<int> v;
        for (int n=n1; n<=n2; n++)
            v.push_back(hist[n]);
        sort(v.begin(),v.end());
        lowest = v[0];

        for (int n=n1; n<=n2; n++)
            if (hist[n]==lowest)
            {
                index.push_back(n);
                //cout << "findLowestHist...lowest=" << lowest << "=hist[" << n << "]=" << hist[n] << endl << flush;
            }
    }
    //cout << formatVector(index,"findLowestHist...") << endl << flush;
    return (lowest);
}



int TraceClassifier::findLowerHist(int n1, int n2, int maxHeight)
{
    if (n1==n2)
        return n1;
    int nHist = hist.size();
    if (n1 >= nHist)
        n1 = nHist-1;
    if (n2 >= nHist)
        n2 = nHist-1;
    if (n1>n2)
        swap(n1,n2);

    for (int n=n1; n<=n2; n++)
        if (hist[n] <= maxHeight)
        {
            //cout << flowRegionStr() << " findLowerHist... found " << n << endl << flush;
            return n;
        }

    // if could not find something <= maxHeight, use the lowest height
    //cout << flowRegionStr() << " calling findLowestHist..." << endl << flush;
    vector<int> index;
    findLowestHist(n1,n2,index);
    return (index[0]);

    return n2;
}


int TraceClassifier::findFinalValleys(const vector<int>& finalPeaks, const vector<int>& valleys, vector<int>&finalValleys)
{
    int nPeaks = finalPeaks.size();
    int nValleys = valleys.size();
    finalValleys.resize(0);

    for (int n=0; n<nPeaks-1; n++)
    {
        int n1 = finalPeaks[n];
        int n2 = finalPeaks[n+1];
        int theValley = 0;
        int maxHeight = int(ceil(hist[n1] * 0.5));
        int minPeakWidth = n1; // not n1-1, because zero-indexed
        for (int i=0; i<nValleys; i++)
        {
            if (isBetween(valleys[i],n1,n2) && (hist[valleys[i]] <= maxHeight) && (valleys[i] >= n1+minPeakWidth))
                //if (isBetween(valleys[i],n1,n2) && (hist[valleys[i]] <= maxHeight))
                //if (isBetween(valleys[i],n1,n2) && (valleys[i] >= n1+minPeakWidth))
            {
                //cerr << flowRegionStr()+" found valley "+val2str(valleys[i])+" betweeen peaks "+val2str(n1) +" & "+val2str(n2) << endl << flush;
                theValley = valleys[i];
                break; // the first valley after the peak
            }
        }
        if (theValley==0) // nothing found
            for (int i=0; i<nValleys; i++)
            {
                if (isBetween(valleys[i],n1,n2) && (hist[valleys[i]] <= maxHeight))
                {
                    //cerr << flowRegionStr()+" found valley "+val2str(valleys[i])+" betweeen peaks "+val2str(n1) +" & "+val2str(n2) << endl << flush;
                    theValley = valleys[i];
                    break; // the first valley after the peak
                }
            }
        if (theValley==0) // nothing found
            for (int i=0; i<nValleys; i++)
            {
                if (isBetween(valleys[i],n1,n2) && (valleys[i] >= n1+minPeakWidth))
                {
                    //cerr << flowRegionStr()+" found valley "+val2str(valleys[i])+" betweeen peaks "+val2str(n1) +" & "+val2str(n2) << endl << flush;
                    theValley = valleys[i];
                    break; // the first valley after the peak
                }
            }
        if (theValley==0) // nothing found
            for (int i=0; i<nValleys; i++)
            {
                if (isBetween(valleys[i],n1,n2))
                {
                    //cerr << flowRegionStr()+" found valley "+val2str(valleys[i])+" betweeen peaks "+val2str(n1) +" & "+val2str(n2) << endl << flush;
                    theValley = valleys[i];
                    break; // the first valley after the peak
                }
            }
        if (theValley==0) // nothing found
        {
            cerr << flowRegionStr()+" found NO valley betweeen peaks "+val2str(n1) +" & "+val2str(n2) << endl << flush;
            assert (theValley!=0);
        }
        finalValleys.push_back(theValley);
    }

    finalValleys.push_back(mNumBins-1);
    /*
    // use (mNumBins-1) as the last valley works better for our purposes
    if (nPeaks>0 && nValleys>0 && finalPeaks[nPeaks-1] < valleys[nValleys-1])
        finalValleys.push_back(valleys[nValleys-1]);
    else
        finalValleys.push_back(mNumBins-1);
    */

    //cerr << formatPeakHist(finalPeaks,flowRegionStr()+" findFinalValleys...finalPeaks") << endl << flush;
    //cerr << formatVector(finalValleys,flowRegionStr()+" findFinalValleys...finalValleys") << endl << flush;
	assert(finalPeaks.size() == finalValleys.size());

    return finalValleys.size();
}


int TraceClassifier::findLastValley(const vector<int>& valleys, int peak, int peak2)
{
    int theValley = mNumBins-1;
    int nValleys = valleys.size();
    for (int i=0; i<nValleys; i++)
    {
        if (valleys[i] > peak && valleys[i] < peak2)
        {
            theValley = valleys[i];
        }
    }
    return theValley;
}



void TraceClassifier::findGaps(const vector<int>& peaks, vector<int>& gaps)
{
    gaps.resize(0);
    int nPeaks = peaks.size();
    for (int n=0; n<nPeaks-1; n++)
    {
        int gap = 0;
        for (int i=peaks[n]+1; i<peaks[n+1]; i++)
            if (hist[i]==0)
                gap++;
        gaps.push_back(gap);
    }
    int gap = 0;
    for (int i=peaks[nPeaks-1]+1; i<(int)hist.size(); i++)
        if (hist[i]==0)
            gap++;
    gaps.push_back(gap);
}



int TraceClassifier::setTrueEmpties(float *sig, int nSig, float thresh)
{
    mTrueEmpties.resize(nSig);
    int nTrueEmpties = 0;
    for (int n=0; n<nSig; n++)
    {
        if (sig[n] <= thresh)
        {
            mTrueEmpties[n] = true;
            nTrueEmpties++;
        }
        else
            mTrueEmpties[n] = false;
    }
    return (nTrueEmpties);
}



TraceClassifier::TraceClassifier(Region *rg,PinnedInFlow *pf,Mask *bf,Image *im,int fl,int imf,MaskType rMask,std::vector<float>& t0,string flOpt, int model)
{
    region = rg;
    pinnedInFlow = pf;
    bfmask = bf;
    img = im;
    flow = fl;
    imgFrames = imf;
    referenceMask = rMask;
    t0_map = &t0[0];
    flowsOpt = flOpt;
    mModel = model;
    ///---------------------------------------------------------------------------------------------------------
    /// process the --bkg-avgEmpty-flows option
    ///---------------------------------------------------------------------------------------------------------
    mDoAllFlows = true;
    mForceAllFlows = false;

    parse_flowsOpt(flowsOpt,flows_on);

    mNumWellsInRegion = region->w * region->h;
    mNumEmptyWells = 0;
    mNumTrueEmpties = 0;

    mNoiseTraces.resize(0);

    // minimum for using the classifier!!!
    mSigFrame_start_max = 20;
    mSigFrame_tail_min = 10;
    mSigFrame_win = 40;
    mBaselineFraction = 0.5;

    mCumHist_min = 0.15;
    mTruePercent_min = (int) (mCumHist_min * 100);
    //mTruePercent_min = 10;
    pick_ratio = 0.333; // use only 1/3 top picks
    deltaMin = 0.2;
    rmsMin = 0.04;
    cvMin = 0.05;
    snrMin = 0.25;
    kurtMax = 0.5;
    mKurtAvailable = false;
    mMinFinalPeaks = 1;
    mMinFs = 2.0; //between-/within-group variance ratio

    assert(imgFrames > DEFAULT_FRAME_TSHIFT);
    assert(DEFAULT_FRAME_TSHIFT >= 0);

    hi.resize(imgFrames);
    lo.resize(imgFrames);
    dy.resize(imgFrames);
    dYY.resize(imgFrames);
    trace_sum.resize(imgFrames);
    picks.resize(imgFrames);
    workTrace.resize(imgFrames);
    avgTrace.resize(imgFrames);
    avgTrace_old.resize(imgFrames);
    baseframes.resize(imgFrames);
    for (int n=0; n<imgFrames; n++)
    {
        hi[n] = -10000; // make sure it is very low
        lo[n] =  10000; // make sure it is very high
        dYY[n] = trace_sum[n] = 0;
        picks[n] = true;
        baseframes[n] = true;
    }
    mTotalLo = 0;

    mHiLo = 0;
    mNumBins = 15;
    mMaxNumPeaks = 1;
    mThreshIdx = mNumBins - 1;
    bins.resize(mNumBins);
    hist.resize(mNumBins);
    cumhist.resize(mNumBins);
    mThreshold = 100;

    mValley2CallClassifier = 1; // used in findValley_FWHM()
    assert(mValley2CallClassifier>0);
    //cout << "TraceClassifier initialized for" << flowRegionStr() << endl << flush;

    mNumTrueEmpties_min = 1;
    // parameters used in useFinalValley()
    mNumEmptyWells_min = 11;
    mNumEmptyWells_min2 = 10;
    mNumEmptyWells_min3 = 9;
    mNoiseMax = 50;
    mNoiseMax = 50;

}



bool TraceClassifier::useFinalValley(int nValleys)
{
    bool ret = false;
    if ((nValleys == 1 && mNumEmptyWells>=mNumEmptyWells_min)  ||
        (nValleys == 2 && mNumEmptyWells>=mNumEmptyWells_min2) ||
        (nValleys == 3 && mNumEmptyWells>=mNumEmptyWells_min3))
        ret = true;
    return (ret);
}

