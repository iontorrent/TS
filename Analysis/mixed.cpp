/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cassert>
#include <cmath>
#include <algorithm>
#include <deque>
#include <functional>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <armadillo>
#include "Mask.h"
#include "RawWells.h"
#include "mixed.h"
#include "bivariate_gaussian.h"
#include "armadillo_utils.h"
#include "MaskSample.h"
#include "BeadTracker.h"

using namespace std;
using namespace arma;

typedef pair<int,int>   well_coord;
typedef set<well_coord> well_set;

void count_sample(filter_counts& counts, deque<float>& ppf, deque<float>& ssq, Mask& mask, RawWells& wells, const vector<int>& key_ionogram);
well_set sample_lib(Mask& mask, int nsamp);

void make_filter(clonal_filter& filter, filter_counts& counts, Mask& mask, RawWells& wells, const vector<int>& key_ionogram)
{
    // Make a clonality filter from a sample of reads from a RawWells file.
    // Record number of reads in sample that are caught by each filter.
    deque<float>  ppf;
    deque<float>  ssq;
    count_sample(counts, ppf, ssq, mask, wells, key_ionogram);
    make_filter(filter, counts, ppf, ssq);
}

void make_filter(clonal_filter& filter, filter_counts& counts, const deque<float>& ppf, const deque<float>& ssq)
{
    // Make a clonality filter from ppf and ssq for a sample of reads.
    // Record number of putative clonal and mixed reads in the sample.
    vec2  mean[2];
    mat22 sigma[2];
    vec2  alpha;
    bool  converged = fit_normals(mean, sigma, alpha, ppf, ssq);

    if(converged){
        bivariate_gaussian clonal(mean[0], sigma[0]);
        bivariate_gaussian mixed( mean[1], sigma[1]);
        filter = clonal_filter(clonal, mixed, mixed_ppf_cutoff(), converged);
    }

    if(converged){
        deque<float>::const_iterator p = ppf.begin();
        for(deque<float>::const_iterator s=ssq.begin(); s!=ssq.end(); ++p, ++s){
            if(filter.is_clonal(*p,*s))
                ++counts._nclonal;
        }
        counts._nmixed = ppf.size() - counts._nclonal;
    }
}

void count_sample(filter_counts& counts, deque<float>& ppf, deque<float>& ssq, Mask& mask, RawWells& wells, const vector<int>& key_ionogram)
{
    // Take sample of reads from a RawWells file, and apply some simple
    // filters to identify problem reads.
    // Record number of reads in sample, and number of reads caught by
    // each filter.
    well_set sample = sample_lib(mask, counts._nsamp);
    WellData data;
    unsigned int nflows = wells.NumFlows();
    vector<float> nrm(nflows);
    int flow0 = mixed_first_flow();
    int flow1 = mixed_last_flow();
    wells.ResetCurrentRegionWell();
    while(!wells.ReadNextRegionData(&data)){
        // Skip if this is not in the sample:
        well_coord wc(data.y, data.x);
        if(sample.find(wc) == sample.end())
            continue;

        // Skip wells with infinite signal:
        bool finite = all_finite(data.flowValues, data.flowValues+nflows);
        if(not finite){
            ++counts._ninf;
            continue;
        }

        // Key-normalize:
        float normalizer = ComputeNormalizerKeyFlows(data.flowValues, &key_ionogram[0], key_ionogram.size());
        transform(data.flowValues, data.flowValues+nflows, nrm.begin(), bind2nd(divides<float>(),normalizer));

        // Skip wells with bad key:
        bool good_key = key_is_good(nrm.begin(), key_ionogram.begin(), key_ionogram.end());
        if(not good_key){
            ++counts._nbad_key;
            continue;
        }

        // Skip possible super-mixed beads:
        float perc_pos = percent_positive(nrm.begin()+flow0, nrm.begin()+flow1);;
        if(perc_pos > mixed_ppf_cutoff()){
            ++counts._nsuper;
            continue;
        }

        // Record ppf and ssq:
        float sum_frac = sum_fractional_part(nrm.begin()+flow0, nrm.begin()+flow1);
        ppf.push_back(perc_pos);
        ssq.push_back(sum_frac);
    }
    assert(ppf.size() == ssq.size());
}

well_set sample_lib(Mask& mask, int nsamp)
{
    // Return a random sample of wells with library beads.
    MaskSample<uint32_t> lib_sample(mask, MaskLib, nsamp);

    well_set sample;
    int chip_width = mask.W();
    for(vector<uint32_t>::iterator i=lib_sample.Sample().begin(); i!=lib_sample.Sample().end(); ++i){
        int row = *i / chip_width;
        int col = *i % chip_width;
        sample.insert(make_pair(row,col));
    }

    return sample;
}


inline double square(double x)
{
    return x * x;
}

static bool test_convergence(const vec2 mean[2], const mat22& sgma, const vec2& alpha, const vec2 new_mean[2], const mat22& new_sgma, const vec2& new_alpha)
{
    double eps        = 1e-4;
    double mean_diff0 = max(max(new_mean[0] - mean[0]));
    double mean_diff1 = max(max(new_mean[1] - mean[1]));
    double sgma_diff  = max(max(new_sgma    - sgma));
    double alpha_diff = max(max(new_alpha   - alpha));
    double max_diff   = max(max(mean_diff0, mean_diff1), max(sgma_diff, alpha_diff));

    return max_diff < eps;
}

static void init(vec2 mean[2], mat22 sgma[2], vec2& alpha)
{
    // Following intializations are based on average over largne number of runs.
    // Average parameters for clonal population:
    mean[0] << 0.48811  << 1.61692;
    sgma[0] << 0.004836 << 0.032828 << endr
            << 0.032828 << 0.548305 << endr;

    // Average parameters for mixed population:
    mean[1] << 0.693371 << 4.397405;
    sgma[1] = sgma[0];

    // Start by assuming that clonal and mixed populations are equinumerous:
    alpha.fill(0.5);
}

bool fit_normals(vec2 mean[2], mat22 sgma[2], vec2& alpha, const deque<float>& ppf, const deque<float>& ssq)
{
    bool converged = false;
    
    try {
        // Initial guesses for two normal distributions:
        init(mean, sgma, alpha);

        int  max_iters = 20;
        for(int i=0; i<max_iters and not converged; ++i){
            // Re-estimate parameters for each distribution:
            int nsamp = ppf.size();
            vec2 sumw;
            sumw.fill(0.0);

            mat22 sum2;
            sum2.fill(0.0);

            vec2   sum1[2];
            sum1[0].fill(0.0);
            sum1[1].fill(0.0);

            if(not is_pos_def(sgma[0]) or not is_pos_def(sgma[1]))
                break;

            bivariate_gaussian clone_dist(mean[0], sgma[0]);
            bivariate_gaussian mixed_dist(mean[1], sgma[1]);

            // Accumulate weighted sums for re-estimating moments:
            for(int j=0; j<nsamp; ++j){
                // Skip reads outside the poisson range for ppf:
                if(mixed_ppf_cutoff() < ppf[j])
                    continue;

                // Each read gets two weights, reflecting the likelyhoods of that
                // read being clonal or mixed.
                vec2 x;
                x << ppf[j] << ssq[j];
                vec2 q;
                q[0] = alpha[0] * clone_dist.pdf(x);
                q[1] = alpha[1] * mixed_dist.pdf(x);
                vec2 w = q / sum(q);

                // Running sums for moments are weighted:
                sumw         += w;
                sum1[0]      += w[0] * x;
                sum1[1]      += w[1] * x;
                sum2.at(0,0) += w[0] * square(ppf[j] - mean[0][0]);
                sum2.at(0,1) += w[0] * (ppf[j] - mean[0][0]) * (ssq[j] - mean[0][1]);
                sum2.at(1,1) += w[0] * square(ssq[j] - mean[0][1]);
            }

            // New means:
            vec2 new_mean[2];
            for(int j=0; j<2; ++j)
                new_mean[j] = sum1[j] / sumw[j];

            // New covariance:
            mat22 new_sgma;
            new_sgma.at(0,0) = sum2.at(0,0) / sumw[0];
            new_sgma.at(0,1) = sum2.at(0,1) / sumw[0];
            new_sgma.at(1,0) = new_sgma.at(0,1);
            new_sgma.at(1,1) = sum2.at(1,1) / sumw[0];

            // New prior:
            vec2 new_alpha;
            new_alpha = sumw / nsamp;

            // Test for convergence:
            converged = test_convergence(mean, sgma[0], alpha, new_mean, new_sgma, new_alpha);

            // Update parameters, forcing covariances to be the same for both distributions:
            alpha   = new_alpha;
            mean[0] = new_mean[0];
            mean[1] = new_mean[1];
            sgma[0] = new_sgma;
            sgma[1] = new_sgma;

            // temp for debug:
            cout << "fit_normals itertation" << setw(4) << i << endl;
            cout << boolalpha << converged << endl;
            cout << mean[0] << endl;
            cout << mean[1] << endl;
            cout << sgma[0] << endl;
            cout << endl;
        }

        // Fallback position if failed to converge:
        if(not converged){
            init(mean, sgma, alpha);
            converged = true;
        }
    }catch(const exception& ex){
        converged = false;
        cerr << "exception thrown during fit_normals()" << endl;
        cerr << ex.what() << endl;
    }catch(...){
        converged = false;
        cerr << "unknown exception thrown during fit_normals()" << endl;
    }

    return converged;
}

ostream& operator<<(ostream& out, const filter_counts& c)
{
    out << setw(8) << "infinite" << setw(12) << c. _ninf    << endl
        << setw(8) << "bad-key"  << setw(12) << c._nbad_key << endl
        << setw(8) << "high-ppf" << setw(12) << c._nsuper   << endl
        << setw(8) << "mixed"    << setw(12) << c._nmixed   << endl
        << setw(8) << "clonal"   << setw(12) << c._nclonal  << endl
        << setw(8) << "samples"  << setw(12) << c._nsamp    << endl;

    return out;
}


