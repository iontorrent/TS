/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef BKGMODELCUDAKERNELS_H
#define BKGMODELCUDAKERNELS_H

#include "BkgModelCuda.h"
#include "MathOptimCuda.h"

//
// Constant Memory Symbols
//

#define USE_CUDA_ERF
#define USE_CUDA_EXP

__constant__ static float POISS_0_APPROX_TABLE_CUDA[sizeof(POISS_0_APPROX_TABLE)];
__constant__ static float POISS_1_APPROX_TABLE_CUDA[sizeof(POISS_1_APPROX_TABLE)];
__constant__ static float POISS_2_APPROX_TABLE_CUDA[sizeof(POISS_2_APPROX_TABLE)];
__constant__ static float POISS_3_APPROX_TABLE_CUDA[sizeof(POISS_3_APPROX_TABLE)];
__constant__ static float POISS_4_APPROX_TABLE_CUDA[sizeof(POISS_4_APPROX_TABLE)];
__constant__ static float POISS_5_APPROX_TABLE_CUDA[sizeof(POISS_5_APPROX_TABLE)];
__constant__ static float POISS_6_APPROX_TABLE_CUDA[sizeof(POISS_6_APPROX_TABLE)];
__constant__ static float POISS_7_APPROX_TABLE_CUDA[sizeof(POISS_7_APPROX_TABLE)];
__constant__ static float POISS_8_APPROX_TABLE_CUDA[sizeof(POISS_8_APPROX_TABLE)];

#ifndef USE_CUDA_ERF
__constant__ static float ERF_APPROX_TABLE_CUDA[sizeof(ERF_APPROX_TABLE)];
#endif

__constant__ static float CLONAL_CALL_SCALE_CUDA[12];

//
// Device Functions
//

template<int block_size, int mat_dim>
__device__
double ddot(int n, double* x, double* y )
{
    __shared__ double sum[block_size];

    // dot(x,y)
    sum[threadIdx.x] = 0;

    for (int i = 0; i < n; i += block_size)
    {
        int ii = i + threadIdx.x;
        if (ii < n)
            sum[threadIdx.x] += x[ii * mat_dim] * y[ii * mat_dim];
    }

    __syncthreads();

    if (block_size >= 64 && threadIdx.x < 32) sum[threadIdx.x] += sum[threadIdx.x + 32]; __syncthreads();
    if (block_size >= 32 && threadIdx.x < 16) sum[threadIdx.x] += sum[threadIdx.x + 16]; __syncthreads();
    if (block_size >= 16 && threadIdx.x <  8) sum[threadIdx.x] += sum[threadIdx.x +  8]; __syncthreads();
    if (block_size >=  8 && threadIdx.x <  4) sum[threadIdx.x] += sum[threadIdx.x +  4]; __syncthreads();
    if (block_size >=  4 && threadIdx.x <  2) sum[threadIdx.x] += sum[threadIdx.x +  2]; __syncthreads();
    if (block_size >=  2 && threadIdx.x <  1) sum[threadIdx.x] += sum[threadIdx.x +  1]; __syncthreads();

    return sum[0];
}

template<int block_size, int mat_dim>
__device__
void dgemv_n(int m, int n, double* a, double* x, double* y)
{
    // y = y - A*x
    for (int i = 0; i < m; i += block_size)
    {
        int ii = i + threadIdx.x;
        if (ii < m)
            for (int j = 0; j < n; j++)
                y[ii] = y[ii] - x[j * mat_dim] * a[ii + mat_dim * j];
    }
}

template<int block_size>
__device__
void dscal( int n, double alpha, double* x )
{
    // x = alpha * x
    for (int i = 0; i < n; i += block_size)
    {
        int ii = i + threadIdx.x;
        if (ii < n)
            x[ii] *= alpha;
    }
}

template<int block_size, int mat_dim>
__device__
void dtrsv_lnn(int n, double* a, double* x )
{
    __shared__ double temp;

    // Solve A * x = b
    for (int j = 0; j < n; j++)
    {
        if (threadIdx.x == 0)
        {
            x[j] = x[j] / a[j + j * mat_dim];
            temp = x[j];
        }

        __syncthreads();

        for (int i = j+1; i < n; i += block_size)
        {
            int ii = i + threadIdx.x;
            if (ii < n)
                x[ii] = x[ii] - temp * a[ii + j * mat_dim];
        }

        __syncthreads();
    }

}

template<int block_size, int mat_dim>
__device__
void dtrsv_ltn( int n, double* a, double* x )
{
    // Solve A' * x = b

    __shared__ double temp[block_size];
    for (int j = n-1; j >= 0; j--)
    {
        temp[threadIdx.x] = 0;

        for (int i = n; i > j+1; i -= block_size )
        {
            int ii = i - threadIdx.x;
            if ( ii > j+1 )
                temp[threadIdx.x] += a[ii + j * mat_dim] * x[ii];
        }

        __syncthreads();

        if (block_size >= 64 && threadIdx.x < 32) temp[threadIdx.x] += temp[threadIdx.x + 32]; __syncthreads();
        if (block_size >= 32 && threadIdx.x < 16) temp[threadIdx.x] += temp[threadIdx.x + 16]; __syncthreads();
        if (block_size >= 16 && threadIdx.x <  8) temp[threadIdx.x] += temp[threadIdx.x +  8]; __syncthreads();
        if (block_size >=  8 && threadIdx.x <  4) temp[threadIdx.x] += temp[threadIdx.x +  4]; __syncthreads();
        if (block_size >=  4 && threadIdx.x <  2) temp[threadIdx.x] += temp[threadIdx.x +  2]; __syncthreads();
        if (block_size >=  2 && threadIdx.x <  1) temp[threadIdx.x] += temp[threadIdx.x +  1]; __syncthreads();

        if ( threadIdx.x == 0 )
        {
            temp[0] = x[j] - temp[0];
            temp[0] = temp[0] / a[j + j * mat_dim];
            x[j] = temp[0];
        }

        __syncthreads();
    }

}


template <int block_size, int mat_dim>
__device__
void dposv( int n, double* a, double* x )
{
    // LTL factorization of lower positive definite matrix
    for (int j = 0; j < n; j++)
    {
        double dot = ddot<block_size, mat_dim> (j, &a[j], &a[j]);

        double ajj = a[(j + j * mat_dim)] - dot;
        ajj = sqrt(ajj);
        a[(j + j * mat_dim)] = ajj;

        if ( j+1 < n)
        {
            dgemv_n<block_size, mat_dim> ( n-j-1, j, &a[(j + 1)], &a[(j)], &a[((j + 1) + mat_dim * j)]);
            dscal<block_size> ( n-j-1, 1.0 / ajj, &a[(j + 1 + mat_dim * j)]);
        }
    }

    // Solve
    dtrsv_lnn<block_size, mat_dim> (n, a, x);
    dtrsv_ltn<block_size, mat_dim> (n, a, x);
}

template<typename T>
__device__
void clamp(T& x, T a, T b)
{
    // Clamps x between a and b
    x = (x < a ? a : (x > b ? b : x));
}

__device__
float poiss_cdf_approx(int n, float x)
{
    int left, right;
    float frac;
    float ret;
    int len;
    const float* ptr;

    switch (n)
    {
        case 0:
            ptr = POISS_0_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_0_APPROX_TABLE) / sizeof(float));
            x *= 20.0f;
            break;
        case 1:
            ptr = POISS_1_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_1_APPROX_TABLE) / sizeof(float));
            x *= 20.0f;
            break;
        case 2:
            ptr = POISS_2_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_2_APPROX_TABLE) / sizeof(float));
            x *= 20.0f;
            break;
        case 3:
            ptr = POISS_3_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_3_APPROX_TABLE) / sizeof(float));
            x *= 20.0f;
            break;
        case 4:
            ptr = POISS_4_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_4_APPROX_TABLE) / sizeof(float));
            x *= 10.0f;
            break;
        case 5:
            ptr = POISS_5_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_5_APPROX_TABLE) / sizeof(float));
            x *= 10.0f;
            break;
        case 6:
            ptr = POISS_6_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_6_APPROX_TABLE) / sizeof(float));
            x *= 10.0f;
            break;
        case 7:
            ptr = POISS_7_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_7_APPROX_TABLE) / sizeof(float));
            x *= 10.0f;
            break;
        case 8:
            ptr = POISS_8_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_8_APPROX_TABLE) / sizeof(float));
            x *= 10.0f;
            break;
        default:
            if (x < 0)
                return 0.0;
            else
                return (1.0 - (erf_approx((x - n - 1) / (sqrt(2 * (n + 1) * 0.88))) + 1) / 2);
    }

    left = (int) x; // left-most point in the lookup table
    right = left + 1; // right-most point in the lookup table

    // both left and right points are inside the table...interpolate between them
    if ((left >= 0) && (right < len))
    {
        frac = (float) (x - left);
        ret = (float) ((1 - frac) * ptr[left] + frac * ptr[right]);
    }
    else
    {
        if (left < 0)
            ret = (float) ptr[0];
        else
            ret = (float) (ptr[len - 1]);
    }

    return (ret);
}

__device__
float poiss_cdf_approx(int n, float x, float scale, int len, const float* ptr)
{
    int left, right;
    float frac;
    float ret;

    if (ptr != NULL) {
        x *= scale;
        left = (int) x; // left-most point in the lookup table
        right = left + 1; // right-most point in the lookup table

        // both left and right points are inside the table...interpolate between them
        if ((left >= 0) && (right < len))
        {
            frac = (float) (x - left);
            ret = (float) ((1 - frac) * ptr[left] + frac * ptr[right]);
        }
        else
        {
            if (left < 0)
                ret = (float) ptr[0];
            else
                ret = (float) (ptr[len - 1]);
        }
    }
    else {
        ret = x < 0 ? 0.0 : (1.0 - (erf_approx((x - n - 1) / (sqrt(2 * (n + 1) * 0.88))) + 1) / 2);
    }
    return ret;
}


__device__
const float*  precompute_pois_params(int n, int& len, float& scale) {
 
    const float* ptr = NULL;
    switch (n)
    {
        case 0:
            ptr = POISS_0_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_0_APPROX_TABLE) / sizeof(float));
            scale = 20.0f;
            break;
        case 1:
            ptr = POISS_1_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_1_APPROX_TABLE) / sizeof(float));
            scale = 20.0f;
            break;
        case 2:
            ptr = POISS_2_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_2_APPROX_TABLE) / sizeof(float));
            scale = 20.0f;
            break;
        case 3:
            ptr = POISS_3_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_3_APPROX_TABLE) / sizeof(float));
            scale = 20.0f;
            break;
        case 4:
            ptr = POISS_4_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_4_APPROX_TABLE) / sizeof(float));
            scale = 10.0f;
            break;
        case 5:
            ptr = POISS_5_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_5_APPROX_TABLE) / sizeof(float));
            scale = 10.0f;
            break;
        case 6:
            ptr = POISS_6_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_6_APPROX_TABLE) / sizeof(float));
            scale = 10.0f;
            break;
        case 7:
            ptr = POISS_7_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_7_APPROX_TABLE) / sizeof(float));
            scale = 10.0f;
            break;
        case 8:
            ptr = POISS_8_APPROX_TABLE_CUDA;
            len = (int) (sizeof(POISS_8_APPROX_TABLE) / sizeof(float));
            scale = 10.0f;
            break;
        default:
            ;
    }
    return ptr;
}

__device__
float erf_approx(float x)
{

#ifdef USE_CUDA_ERF
    return erf(x);
#else

    int left, right;
    float sign = 1.0;
    float frac;
    float ret;

    if (x < 0.0)
    {
        x = -x;
        sign = -1.0;
    }

    left = (int) (x * 100.0); // left-most point in the lookup table
    right = left + 1; // right-most point in the lookup table

    // both left and right points are inside the table...interpolate between them
    if ((left >= 0) && (right < (int) (sizeof(ERF_APPROX_TABLE) / sizeof(float))))
    {
        frac = (x * 100.0 - left);
        ret = (1 - frac) * ERF_APPROX_TABLE_CUDA[left] + frac * ERF_APPROX_TABLE_CUDA[right];
    }
    else
    {
        if (left < 0)
        ret = ERF_APPROX_TABLE_CUDA[0];
        else
        ret = 1.0;
    }

    return (ret * sign);

#endif

}

__device__
float exp_approx(float x, float* ExpApproxArray, int nElements)
{

#ifdef USE_CUDA_EXP
    return exp(x);
#else

    int left, right;
    float frac;
    float ret;

    if (x > 0)
    {
        return exp(x);
    }

    x = -x; // make the index positive

    left = (int) (x * 100.0); // left-most point in the lookup table
    right = left + 1; // right-most point in the lookup table

    // both left and right points are inside the table...interpolate between them
    if ((left >= 0) && (right < nElements))
    {
        frac = (x * 100.0 - left);
        ret = (1 - frac) * ExpApproxArray[left] + frac * ExpApproxArray[right];
    }
    else
    {
        if (left < 0)
        ret = ExpApproxArray[0];
        else
        ret = 0.0;
    }

    return (ret);

#endif

}

//
// Global Functions
//

__global__
void DfdgainStep_k(float* f, bead_params* p, float* EmphasisVectorByHomopolymer, int* active_bead_list, int num_fb, int num_pts, int num_steps, int step)
{
    int ibd = active_bead_list[blockIdx.x];
    int i = threadIdx.x;

    int stepOffset = step * num_pts * num_fb;
    f = &f[ibd * (num_fb * num_pts * num_steps)];

    for ( int fnum = 0; fnum < num_fb; fnum++ )
    {
        float* my_fval = f;
        float* my_output = f + stepOffset;

        float *src = &my_fval[num_pts * fnum];
        float *dst = &my_output[num_pts * fnum];
        float* em = &EmphasisVectorByHomopolymer[(p[ibd].WhichEmphasis[fnum])*num_pts];

        dst[i] = src[i] * em[i] / p[ibd].gain;
    }

}


__global__
void DfderrStep_k(float* f, bead_params* p, float* EmphasisVectorByHomopolymer, float* dark_matter_compensator, int* nucMap, int *active_bead_list, int num_fb, int num_pts, int num_steps, int step)
{
    int ibd = active_bead_list[blockIdx.x];
    int i = threadIdx.x;

    int stepOffset = step * num_pts * num_fb;
    f = &f[ibd * (num_fb * num_pts * num_steps)];

    for ( int fnum = 0; fnum < num_fb; fnum++ )
    {
        float* my_output = f + stepOffset;
        float* dst = &my_output[num_pts * fnum];
        float* em = &EmphasisVectorByHomopolymer[(p[ibd].WhichEmphasis[fnum])*num_pts];
	float* et = &dark_matter_compensator[nucMap[fnum]*num_pts];
        dst[i] = et[i] * em[i];
    }
}


template<int beads_per_block>
__global__
void YerrStep_k(float* f, bead_params* eval_params, float* EmphasisVectorByHomopolymer, float* EmphasisScale, int* active_bead_list, FG_BUFFER_TYPE* fg_buffers, float* residual, int num_fb, int num_pts, int num_steps, int num_beads, int step, int num_active_beads)
{
    // Bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if ( idx >= num_active_beads )
        return;

    // Bead index
    int ibd = active_bead_list[idx];

    // Thread index
    int fnum = threadIdx.x;

    // Shared memory used for reduction
    __shared__ float res[NUMFB][beads_per_block];
    __shared__ float scale[NUMFB][beads_per_block];
    __shared__ float mean_res[NUMFB][beads_per_block];

    // Per-bead / per-step output
    float* my_scratch_space = &f[ibd * (num_fb * num_pts * num_steps)];
    float* my_output = my_scratch_space + (step * (num_pts * num_fb));
    FG_BUFFER_TYPE *pfg = &fg_buffers[num_fb * num_pts * ibd] + num_pts * fnum;

    res[threadIdx.x][threadIdx.y] = 0.0;
    scale[threadIdx.x][threadIdx.y] = 0.0;
    mean_res[threadIdx.x][threadIdx.y] = 0.0;

    float *fv = my_scratch_space; // fval
    float *ye = my_output;

    int emndx = eval_params[ibd].WhichEmphasis[fnum];
    float* em = &EmphasisVectorByHomopolymer[emndx*num_pts];

    fv = my_scratch_space + (fnum * num_pts);
    ye = my_output + (fnum * num_pts);

    float rerr = 0.0;

    for (int i = 0; i < num_pts; i++)
    {
        float eval = ((float) (pfg[i]) - fv[i]);

	rerr += eval*eval;
        eval *= em[i];
        ye[i] = eval;
        res[threadIdx.x][threadIdx.y] += (eval * eval);
    }


    scale[threadIdx.x][threadIdx.y] += EmphasisScale[emndx];
    eval_params[ibd].rerr[threadIdx.x] = sqrt(rerr/(float)num_pts);
    mean_res[threadIdx.x][threadIdx.y] += eval_params[ibd].rerr[threadIdx.x]; // in v7

    __syncthreads();

    // Reduce to 10 elements
    if (threadIdx.x < 10 )
    {
        res[threadIdx.x][threadIdx.y] += res[ 10 + threadIdx.x][threadIdx.y];
        scale[threadIdx.x][threadIdx.y] += scale[ 10 + threadIdx.x ][threadIdx.y];
	mean_res[threadIdx.x][threadIdx.y] += mean_res[10 + threadIdx.x][threadIdx.y];
    }
    __syncthreads();

    // Reduce to 5 elements
    if (threadIdx.x < 5 )
    {
        res[threadIdx.x][threadIdx.y] += res[ 5 + threadIdx.x][threadIdx.y];
        scale[threadIdx.x][threadIdx.y] += scale[ 5 + threadIdx.x ][threadIdx.y];
	mean_res[threadIdx.x][threadIdx.y] += mean_res[5 + threadIdx.x][threadIdx.y];
    }
    __syncthreads();

    // Sum reduced elements
    if ( threadIdx.x == 0 )
    {
        res[0][threadIdx.y] += res[1][threadIdx.y];
        res[0][threadIdx.y] += res[2][threadIdx.y];
        res[0][threadIdx.y] += res[3][threadIdx.y];
        res[0][threadIdx.y] += res[4][threadIdx.y];

        scale[0][threadIdx.y] += scale[1][threadIdx.y];
        scale[0][threadIdx.y] += scale[2][threadIdx.y];
        scale[0][threadIdx.y] += scale[3][threadIdx.y];
        scale[0][threadIdx.y] += scale[4][threadIdx.y];

        mean_res[0][threadIdx.y] += mean_res[1][threadIdx.y];
        mean_res[0][threadIdx.y] += mean_res[2][threadIdx.y];
        mean_res[0][threadIdx.y] += mean_res[3][threadIdx.y];
        mean_res[0][threadIdx.y] += mean_res[4][threadIdx.y];

        residual[ibd] = sqrt( res[0][threadIdx.y] / scale[0][threadIdx.y] );
        mean_res[0][threadIdx.y] /= num_fb;
    }
    __syncthreads();

    eval_params[ibd].rerr[fnum] /= mean_res[0][threadIdx.y];     
}

template<bool add>
__global__
void DoStepDiff_k(bead_params* p, int p_offset, float diff, int* active_bead_list_cuda)
{
    // Bead indexes
    int ibd = active_bead_list_cuda[blockIdx.x];

    // Thread index
    int i = threadIdx.x;

    // Adjust pointers
    p = &p[ibd];

    // Offset pointer
    float* d = (float*) ((char*) p + p_offset);

    if (add)
        d[i] += diff;
    else
        d[i] -= diff;
}

template<bool add>
__global__
void DoStepDiffReg_k(reg_params* rp, int rp_offset, float diff)
{
    // Thread index
    int i = threadIdx.x;

    // Offset pointer
    float* d = (float*) ((char*) rp + rp_offset);

    if (add)
        d[i] += diff;
    else
        d[i] -= diff;
}

template<bool add>
__global__
void DoStepDiffNuc_k(reg_params* rp, int nuc_offset, float diff)
{
    // Thread index
    int i = threadIdx.x;

    // Offset pointer
    float* d = (float*) ((char*) &(rp->nuc_shape) + nuc_offset);

    if (add)
        d[i] += diff;
    else
        d[i] -= diff;
}

template<int beads_per_block>
__global__
void CalcPartialDerivWithEmphasis_k(float* p, float* pe, int len, float dp, int pelen, bead_params* eval_params, int* active_bead_list, int step, int num_fb, int num_pts, int num_steps, int num_active_beads)
{
    // Get indexes
    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (bead_index >= num_active_beads)
        return;

    int ibd = active_bead_list[bead_index];
    int i = threadIdx.x;
    eval_params += ibd;

    int beadOffset = ibd * num_fb * num_pts * num_steps;
    int stepOffset = step * num_pts * num_fb;
    for ( int fnum = 0; fnum < num_fb; fnum++ )
    {
        // Adjust pointers
        float* p1 = &p[beadOffset + fnum * num_pts];
        float* p2 = p1 + stepOffset;
        float* pev = &pe[eval_params->WhichEmphasis[fnum]*num_pts];

        p2[i] = (p2[i] - p1[i]) * pev[i] / dp;
    }
}

template<int mat_dim>
__global__
void SingularMatrixCheck_k( int* active_bead_list, int n, double* jtj, bool* cont_proc )
{
    int ibd = active_bead_list[blockIdx.x];

    jtj += ibd * mat_dim * mat_dim;

    for ( int i = 0; i < n; i++ )
        if ( jtj[ i*mat_dim+i ] == 0 ) {
            cont_proc[ibd] = true;
        }
}

template<int mat_dim>
__global__
void CopyMatrices_k(int* active_bead_list, double* jtj, double* rhs, double* jtj_lambda, double* delta, bead_params* eval_params, bead_params* params_nn)
{
    int ibd = active_bead_list[blockIdx.x];

    // Data offset
    int matOffset = ibd * mat_dim;
    delta += matOffset;
    rhs += matOffset;
    jtj_lambda += matOffset * mat_dim;
    jtj += matOffset * mat_dim;
    float* params_ptr = (float*) &eval_params[ibd];
    float* params_nn_ptr = (float*) &params_nn[ibd];

    // Rhs --> Delta
    delta[threadIdx.x] = rhs[threadIdx.x];

    // Jtj --> Jtj Lambda
    #pragma unroll
    for (int i = 0; i < mat_dim; i++)
        jtj_lambda[threadIdx.x + i * mat_dim] = jtj[threadIdx.x + i * mat_dim];

    // Params NN --> Params (this only works because the params structure is only floats!)
    for (int i = 0; i < sizeof(bead_params) / sizeof(float); i += mat_dim)
        if (threadIdx.x + i < sizeof(bead_params) / sizeof(float))
            params_ptr[threadIdx.x + i] = params_nn_ptr[threadIdx.x + i];
}

template<int mat_dim>
__global__
void AccumulateToRegionalMatrix_k(int* active_bead_list, int beads, double* jtj, double* rhs)
{
    bool isOdd = (beads % 2) != 0;

    int ibd1 = active_bead_list[ blockIdx.x ];
    int ibd2 = active_bead_list[ blockIdx.x + gridDim.x + ( isOdd ? 1 : 0 ) ];

    double* jtj1 = jtj + ibd1 * mat_dim * mat_dim;
    double* jtj2 = jtj + ibd2 * mat_dim * mat_dim;

    double* rhs1 = rhs + ibd1 * mat_dim;
    double* rhs2 = rhs + ibd2 * mat_dim;

    // Sum JTJ1 += JTJ2
    for (int ld = 0; ld < mat_dim; ld++)
        jtj1[ld * mat_dim + threadIdx.x] += jtj2[ld * mat_dim + threadIdx.x];

    // Sum RHS1 += RHS2
    rhs1[threadIdx.x] += rhs2[threadIdx.x];
}

template<int block_size, int beads_per_block>
__global__
void DotProductMatrixSums_k(double* sums, float* fval, int offset_1, int offset_2, int length, int* active_bead_list, int num_fb, int num_pts, int num_steps,
        int num_active_beads)
{
    // Shared memory buffer
    __shared__ double buffer[beads_per_block][block_size];

    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.y;
    if (bead_index >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[bead_index];
    int tid = threadIdx.x;

    // Adjust pointers
    fval += ibd * num_steps * num_fb * num_pts;
    float* f1 = fval + offset_1;
    float* f2 = fval + offset_2;

    buffer[threadIdx.y][tid] = 0;

    for (int chunk = 0; chunk < length; chunk += block_size)
    {
        int i = chunk + tid;
        if (i < length)
            buffer[threadIdx.y][tid] += double(f1[i] * f2[i]);
    }

    __syncthreads();

    // Reductions
    if (tid < 4) buffer[threadIdx.y][tid] += buffer[threadIdx.y][tid + 4]; __syncthreads();
    if (tid < 2) buffer[threadIdx.y][tid] += buffer[threadIdx.y][tid + 2]; __syncthreads();
    if (tid < 1) buffer[threadIdx.y][tid] += buffer[threadIdx.y][tid + 1]; __syncthreads();

    if (tid == 0)
        sums[ibd] += buffer[threadIdx.y][0];
}

template <int beads_per_block>
__global__
void DotProductMatrixSumsNew_k(double* sums, float* fval, int offset_1, int offset_2, int length, int* active_bead_list, int num_fb, int num_pts, int num_steps,
        int num_active_beads)
{
    // Shared memory buffer
    extern __shared__ double buffer[];

    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.y;
    if (bead_index >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[bead_index];
    int tid = threadIdx.x;

    // Adjust pointers
    fval += ibd * num_steps * num_fb * num_pts;
    float* f1 = fval + offset_1;
    float* f2 = fval + offset_2;

    buffer[tid] = double(f1[tid] * f2[tid]);

    //__syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        for (int i=0; i<length; ++i)
            sum += buffer[i];
        sums[ibd] += sum;

    }
}


template<int beads_per_block>
__global__
void WriteMatrixFromSums_k(AssyMatID matId, int row, int col, double* sum, int mat_dim, double* jtj, double* rhs, int* active_bead_list, int num_active_beads)
{
    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.x;

    // Bounds check
    if (bead_index >= num_active_beads)
        return;

    // Bead number
    int ibd = active_bead_list[bead_index];

    // Pointer offsets
    int matOffset = ibd * mat_dim;
    jtj += matOffset * mat_dim;
    rhs += matOffset;

    // Write to proper matrix
    if (matId == JTJ_MAT)
        jtj[ (col * mat_dim + row) ] = sum[ibd];
    else
        rhs[row] = sum[ibd];
}


__global__
void CalculateResidual_k(float* fval, bead_params* eval_params, float* ev, float* EmphasisScale, int* active_bead_list,
        FG_BUFFER_TYPE* fg_buffers, float* residual, int num_fb, int num_pts, int num_steps, int num_beads)
{
    // Bead index
    int ibd = active_bead_list[blockIdx.x];

    // Per bead pointer offset
    fval += ibd * (num_fb * num_pts * num_steps);

    // Shared memory used for reduction
    __shared__ float res[NUMFB];
    __shared__ float scale[NUMFB];
    __shared__ float mean_res[NUMFB];

    res[threadIdx.x] = 0.0;

    int fnum = threadIdx.x;

    FG_BUFFER_TYPE *pfg = &fg_buffers[ibd * num_fb * num_pts] + num_pts * fnum;
    float eval;
    int emndx = eval_params[ibd].WhichEmphasis[fnum];
    float *em = &ev[emndx*num_pts];
    float rerr = 0.0;

    fval += (fnum * num_pts);
    

    for (int i = 0; i < num_pts; i++)
    {
        eval = ((float) (pfg[i]) - fval[i]);

        rerr += eval*eval;
        eval = eval*em[i];
        res[fnum] += eval*eval;
    }


    eval_params[ibd].rerr[fnum] = sqrt(rerr/(float)num_pts);
    scale[fnum] = EmphasisScale[emndx];
    mean_res[fnum] = eval_params[ibd].rerr[fnum];

    __syncthreads();

    // Reduce to 10 elements
    if (threadIdx.x < 10 )
    {
        res[threadIdx.x] += res[ 10 + threadIdx.x];
        scale[threadIdx.x] += scale[ 10 + threadIdx.x ];
        mean_res[threadIdx.x] += mean_res[ 10 + threadIdx.x ];
    }
    __syncthreads();

    // Reduce to 5 elements
    if (threadIdx.x < 5 )
    {
        res[threadIdx.x] += res[ 5 + threadIdx.x];
        scale[threadIdx.x] += scale[ 5 + threadIdx.x ];
        mean_res[threadIdx.x] += mean_res[ 5 + threadIdx.x ];
    }
    __syncthreads();

    // Sum reduced elements
    if ( threadIdx.x == 0 )
    {
        res[0] += res[1];
        res[0] += res[2];
        res[0] += res[3];
        res[0] += res[4];

        scale[0] += scale[1];
        scale[0] += scale[2];
        scale[0] += scale[3];
        scale[0] += scale[4];
        
	mean_res[0] += mean_res[1];
        mean_res[0] += mean_res[2];
        mean_res[0] += mean_res[3];
        mean_res[0] += mean_res[4];

        residual[ibd] = sqrt( res[0] / scale[0] );
	mean_res[0] /= num_fb;
    }
    __syncthreads();

    eval_params[ibd].rerr[fnum] /= mean_res[0];
}

__global__
void CalculateRegionalResidual_k(float* fval, bead_params* eval_params, float* ev, float* EmphasisScale, int* active_bead_list,
        FG_BUFFER_TYPE* fg_buffers, float* residual, int num_fb, int num_pts, int num_steps, int num_beads)
{
    // Bead index
    int ibd = active_bead_list[blockIdx.x];

    // Per bead pointer offset
    fval += ibd * (num_fb * num_pts * num_steps);

    // Shared memory used for reduction
    __shared__ float res[NUMFB];
    __shared__ float scale[NUMFB];

    res[threadIdx.x] = 0.0;

    int fnum = threadIdx.x;

    FG_BUFFER_TYPE *pfg = &fg_buffers[ibd * num_fb * num_pts] + num_pts * fnum;
    float eval;
    int emndx = eval_params[ibd].WhichEmphasis[fnum];
    float *em = &ev[emndx*num_pts];

    fval += (fnum * num_pts);

    for (int i = 0; i < num_pts; i++)
    {
        eval = ((float) (pfg[i]) - fval[i])*em[i];
        res[fnum] += eval*eval;
    }
    scale[fnum] = EmphasisScale[emndx];

    __syncthreads();

    // Reduce to 10 elements
    if (threadIdx.x < 10 )
    {
        res[threadIdx.x] += res[ 10 + threadIdx.x];
        scale[threadIdx.x] += scale[ 10 + threadIdx.x ];
    }
    __syncthreads();

    // Reduce to 5 elements
    if (threadIdx.x < 5 )
    {
        res[threadIdx.x] += res[ 5 + threadIdx.x];
        scale[threadIdx.x] += scale[ 5 + threadIdx.x ];
    }
    __syncthreads();

    // Sum reduced elements
    if ( threadIdx.x == 0 )
    {
        res[0] += res[1];
        res[0] += res[2];
        res[0] += res[3];
        res[0] += res[4];

        scale[0] += scale[1];
        scale[0] += scale[2];
        scale[0] += scale[3];
        scale[0] += scale[4];

        residual[ibd] = sqrt( res[0] / scale[0] );
    }
}


template<int num_fb, int beads_per_block>
__global__
void MultiCycleNNModelFluxPulse_base_CUDA_k(float* fval, bead_params* p, reg_params* reg_p, float* ival, float* sbg, int* flow_ndx_map, float* delta_frame, float* dark_matter_compensator, int* buff_flow, int num_pts, int* active_bead_list, int step, int num_steps, int num_active_beads)
{
    extern __shared__ float smem_buffer[];

    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (bead_index >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[bead_index];

    // Adjust pointers
    int flowFrameProduct = num_fb * num_pts;
    fval += ibd * (flowFrameProduct * num_steps) + (step * flowFrameProduct);
    p += ibd;
    ival += ibd * flowFrameProduct;

    // Offset shared memory to this block's bead
    float* bead_smem = &smem_buffer[threadIdx.y * flowFrameProduct];

    // Coalesced input read from global memory
    for (int i = 0; i < flowFrameProduct; i += num_fb)
        bead_smem[i + threadIdx.x] = ival[i + threadIdx.x];
    __syncthreads();

    // per thread index
    int fnum = threadIdx.x;

    // constants used for solving background signal shape
    float aval = 0;

    // values for time shifting
    int intcall = ((int) (p->Ampl[fnum] + 0.5));
    clamp(intcall, 0, 5);

    float clonal_error_term = 0.0;
    if ((p->Ampl[fnum] < reg_p->restrict_clonal) && p->clonal_read && (fnum > KEY_LEN))
       clonal_error_term = fabs(p->Ampl[fnum] - intcall) * CLONAL_CALL_SCALE_CUDA[intcall];

    int nnum = flow_ndx_map[fnum];

    // calculate some constants used for this flow
    float gain = p->gain;
    float R = p->R * reg_p->NucModifyRatio[nnum] + reg_p->RatioDrift * buff_flow[fnum] * (1.0 - p->R*reg_p->NucModifyRatio[nnum])*0.001;
    float tau = (reg_p->tau_R_m*R+reg_p->tau_R_o);
    clamp(tau, (float)MINTAUB, (float)MAXTAUB);

    float dv = 0.0;
    float dv_rs = 0.0;
    float dt = -1.0;
    float dvn = 0.0;

    float* ve = &sbg[fnum * num_pts];
    float* smem = &bead_smem[fnum * num_pts];
    float darkness = reg_p->darkness[0];
    float* et = &dark_matter_compensator[nnum*num_pts];

    float scaleTau = 2.0 * tau;
    float Roffset = R - 1.0;
    float curSbgVal;
    float temp;
    for (int i = 0; i < num_pts; i++)
    {
        dt = delta_frame[i];
	aval = dt/scaleTau;

        // calculate new dv
        curSbgVal = ve[i];
        dvn = (smem[i] + Roffset * curSbgVal - dv_rs / tau - dv * aval) / (1 + aval);
        dv_rs += (dv + dvn) * dt * 0.5;
        dv = dvn;

        temp = (dv + curSbgVal) * gain + darkness * et[i];

        if (i < MAXCLONALMODIFYPOINTSERROR)
            temp += clonal_error_term * ((float) (i & 1) - 0.5) * 1600.0;

        smem[i] = temp;
    }

    __syncthreads();

    // Coalesced output write from global memory
    for (int i = 0; i < flowFrameProduct; i += num_fb)
        fval[i + threadIdx.x] = bead_smem[i + threadIdx.x];
}


template<int mat_dim, int block_size>
__global__
void FactorizeAndSolveMatrices_k(int* active_bead_list, int n, float* lambda, double* jtj_lambda, double* delta, int num_active_beads)
{
    // Bead index
    int ibd = active_bead_list[blockIdx.x];

    // Offset to this beads memory locations
    int matOffset = ibd * mat_dim;
    jtj_lambda += (mat_dim * matOffset);
    delta += matOffset;

    //__syncthreads();

    double lambdaVal = 1.0 + lambda[ibd];
    // Scale JTJ's diagonal by lambda
    for (int i = 0; i < n; i += block_size )
    {
        int ii = threadIdx.x + i;
        if ( ii < n )
            jtj_lambda[ (ii * mat_dim + ii) ] = jtj_lambda[ (ii * mat_dim + ii) ] * lambdaVal;
    }

    __syncthreads();

    // Cholesky factorization and solve
    dposv<block_size,mat_dim> ( n, jtj_lambda, delta );
}

template<int mat_dim, int block_size>
__global__
void FactorizeAndSolveRegionalMatrix_k(int n, float reg_lambda, double* jtj_lambda, double* delta)
{
    // Scale JTJ's diagonal by lambda
    for (int i = 0; i < n; i += block_size)
    {
        int ii = i + threadIdx.x;

        if (ii < n)
            jtj_lambda[ii * mat_dim + ii] = jtj_lambda[ii * mat_dim + ii] * double(1.0 + reg_lambda);
    }

    __syncthreads();

    // Cholesky factorization and solve
    dposv<block_size,mat_dim> ( n, jtj_lambda, delta );
}

template<int mat_dim>
__global__
void AdjustParameters_k(int* active_bead_list, bead_params* eval_params, bead_params* params_nn, int num_outputs, float* lambda, double* jtj_lambda, double* delta,
        delta_mat_output_line* output_list, bead_params* params_low, bead_params* params_high)
{
    // Get bead index
    int ibd = active_bead_list[blockIdx.x];

    // Offset to this beads memory locations
    delta += (mat_dim * ibd);

    // Adjust parameters
    float* eval_ptr = (float*) &eval_params[ibd];
    eval_ptr[ output_list[threadIdx.x].param_ndx ] += delta[ output_list[threadIdx.x].delta_ndx ];
    __syncthreads();

    // Clamp new parameters
    for (int i = 0; i < NUMFB; i++)
    {
        clamp(eval_params[ibd].Ampl[i], params_low[ibd].Ampl[i], params_high[ibd].Ampl[i]);
        clamp(eval_params[ibd].kmult[i], params_low[ibd].kmult[i], params_high[ibd].kmult[i]);
    }

    clamp(eval_params[ibd].R, params_low[ibd].R, params_high[ibd].R);

    clamp(eval_params[ibd].Copies, params_low[ibd].Copies, params_high[ibd].Copies);
    clamp(eval_params[ibd].gain, params_low[ibd].gain, params_high[ibd].gain);
    
    clamp(eval_params[ibd].dmult, params_low[ibd].dmult, params_high[ibd].dmult);
}

__global__
void AdjustRegionalParameters_k( int* active_bead_list, reg_params* rp, bead_params* params_nn, bead_params* params_low, bead_params* params_high, int num_fb )
{
    // Get bead index
    int ibd = active_bead_list[ blockIdx.x ];

    int i = 0;
    // Adjust parameters
    params_nn[ibd].R += rp->R;
    params_nn[ibd].Copies += rp->Copies;

    // Clamp new parameters
    for (i = 0; i < NUMFB; i++)
    {
        params_nn[ibd].Ampl[i] += rp->Ampl[i];
        clamp(params_nn[ibd].Ampl[i], params_low[ibd].Ampl[i], params_high[ibd].Ampl[i]);
        clamp(params_nn[ibd].kmult[i], params_low[ibd].kmult[i], params_high[ibd].kmult[i]);
    }

    clamp(params_nn[ibd].R, params_low[ibd].R, params_high[ibd].R);

    clamp(params_nn[ibd].Copies, params_low[ibd].Copies, params_high[ibd].Copies);
    clamp(params_nn[ibd].gain, params_low[ibd].gain, params_high[ibd].gain);

    clamp(params_nn[ibd].dmult, params_low[ibd].dmult, params_high[ibd].dmult);
}

__global__
void AdjustLambdaAndUpdateParameters_k(int* active_bead_list, bool* cont_proc, bool* well_completed, float* new_residual, float* residual,
        bead_params* params_nn, bead_params* eval_params, float* lambda, bool is_reg_fit, bool is_under_iter, int num_fb, int* fit_flag, int* req_flag )
{
    // Get bead index
    int ibd = active_bead_list[blockIdx.x];

    float achg = 0;

    if (new_residual[ibd] < residual[ibd])
    {
        for (int i = 0; i < num_fb; i++)
        {
            float chg = fabs(params_nn[ibd].Ampl[i] * params_nn[ibd].Copies - eval_params[ibd].Ampl[i] * eval_params[ibd].Copies);

            if (chg > achg)
                achg = chg;
        }

        params_nn[ibd] = eval_params[ibd];
        lambda[ibd] *= 0.10;

        if (lambda[ibd] < FLT_MIN)
            lambda[ibd] = FLT_MIN;

        residual[ibd] = new_residual[ibd];

        cont_proc[ibd] = true;
    }
    else
    {
        lambda[ibd] *= 10.0;
    }

    if ( (achg < 0.001) && (lambda[ibd] >= 1e10) )
    {
        // This block will contribute +1 to req_done
        req_flag[blockIdx.x] = 1;

        if ( !is_reg_fit || !is_under_iter )
        {
            // This block will contribute -1 to numFit
            fit_flag[blockIdx.x] = 1;

            well_completed[ibd] = true;
            cont_proc[ibd] = true;
        }
    }

    // if regional fitting...we can get stuck here if we can't improve until the next regional fit
    if ( is_reg_fit && is_under_iter && (lambda[ibd] >= 1e8) )
        cont_proc[ibd] = true;
}

template <int beads_per_block>
__global__
void MultiFlowComputeCumulativeIncorporationSignal_CUDA_k(bead_params* p, reg_params* reg_p, float* ivalPtr, int* flow_ndx_map, float* deltaFrame, int* buff_flow, int num_pts, int num_fb, int* active_bead_list, int num_active_beads, float* exp_approx_table_cuda, int exp_approx_table_size, float* nucRise, int* i_start)
{
    // Shared memory buffer
    extern __shared__ float smem[];

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;
    int nnum = flow_ndx_map[fnum];

    // Adjust pointers
    p += ibd;
    ivalPtr += ibd * (num_fb * num_pts);
    i_start += ibd*NUMNUC;
    float* bead_smem = &smem[ threadIdx.y * (num_fb * num_pts) ];


    float A;
    float c_dntp_sum;
    int ileft;
    int iright;
    float ifrac;

    // step 2
    float occ_l, occ_r;
    float totocc;
    float totgen;
    float pact;
    float c_dntp_bot;
    int i, st;

    // step 3
    float ldt;

    // step 4
    float c_dntp_top;
    float alpha;
    float c_dntp_int;
    float expval;
    float pact_new;
    float sign_and_sens;

    float SP = (float)(COPYMULTIPLIER * p->Copies)*pow(reg_p->CopyDrift,buff_flow[fnum]);

    float sens = reg_p->sens*SENSMULTIPLIER;
    float d = reg_p->d[nnum]*p->dmult;
    float kr = reg_p->krate[nnum] * p->kmult[fnum];
    float kmax = reg_p->kmax[nnum];
    float C = reg_p->nuc_shape.C;

    A = p->Ampl[fnum];
    sign_and_sens = sens;
    if (A < 0.0) {
	    A *= -1.0;
	    sign_and_sens *= -1.0;
    }
    else if (A > MAX_HPLEN) {
        A = MAX_HPLEN; 
    }

    // initialize diffusion/reaction simulation for this flow
    ileft = (int) A;
    iright = ileft + 1;
    ifrac = iright - A;
    occ_l = ifrac;
    occ_r = A - ileft;


    ileft--;
    iright--;

    if (ileft < 0)
    {
        occ_l = 0.0;
    }
    
    if (iright == MAX_HPLEN)
    {
        iright = ileft;
	occ_r = occ_l;
	occ_l = 0;
    }
   
    occ_l *= SP;
    occ_r *= SP;
    pact = occ_l + occ_r;
    totocc = SP*A; 
    totgen = totocc;

    c_dntp_bot = 0.0; // concentration of dNTP in the well
    c_dntp_top = 0.0;
    c_dntp_sum = 0.0;

    // some pre-computed things
    float c_dntp_bot_plus_kmax = 1.0/kmax;
    float last_tmp1 = 0.0;
    float last_tmp2 = 0.0;
    float c_dntp_fast_inc = kr*(C/ (C+kmax));

    // [dNTP] in the well after which we switch to simpler model
    float fast_start_threshold = 0.99*C;
    nucRise += ibd*NUMNUC*num_pts*ISIG_SUB_STEPS_MULTI_FLOW + nnum*num_pts*
	    ISIG_SUB_STEPS_MULTI_FLOW;
    int start = i_start[nnum];
    // first non-zero index of the computed [dNTP] array for this nucleotide
    int c_dntp_top_ndx = start*ISIG_SUB_STEPS_MULTI_FLOW;

    float rscale, lscale;
    int rlen, llen;
    float scaled_kr = kr*n_to_uM_conv;
    const float* rptr = precompute_pois_params(iright, rlen, rscale);
    const float* lptr = precompute_pois_params(ileft, llen, lscale);

    for(i=0; i<start; ++i)
	    bead_smem[fnum*num_pts + i] = 0;

    for (i = start; i < num_pts; i++)
    {
        if (totgen > 0.0) {
            ldt = deltaFrame[i]/FRAMESPERSEC;

	    // once the [dNTP] pretty much reaches full strength in the well
	    // the math becomes much simpler
	    if (c_dntp_bot > fast_start_threshold)
	    {
                c_dntp_sum += c_dntp_fast_inc * ldt;
		pact_new = poiss_cdf_approx(iright, c_dntp_sum, rscale, rlen, rptr) * occ_r;
		if (occ_l > 0.0)
			pact_new += poiss_cdf_approx(ileft, c_dntp_sum, lscale, llen, lptr) * occ_l;

		totgen -= ((pact + pact_new) / 2.0) * c_dntp_fast_inc * ldt;
		pact = pact_new;
	    }
	    else
	    {
	        ldt /= ISIG_SUB_STEPS_MULTI_FLOW;
		for (st = 1; (st <= ISIG_SUB_STEPS_MULTI_FLOW) && (totgen > 0.0); st++)
		{
		    c_dntp_top = nucRise[c_dntp_top_ndx++];

		    alpha = d + scaled_kr*pact*c_dntp_bot_plus_kmax;
		    expval = exp_approx(-alpha * ldt, exp_approx_table_cuda, exp_approx_table_size);

		    c_dntp_bot = c_dntp_bot*expval + d*c_dntp_top * (1 - expval) / alpha;

		    c_dntp_bot_plus_kmax = 1.0/(c_dntp_bot + kmax);
		    last_tmp1 = c_dntp_bot * c_dntp_bot_plus_kmax;
		    c_dntp_int = kr*(last_tmp2 + last_tmp1) *ldt/2.0;
		    last_tmp2 = last_tmp1;
		    c_dntp_sum += c_dntp_int;

		    // calculate new number of active polymerase
		    //pact_new = poiss_cdf_approx(iright,c_dntp_sum) * occ_r;
		    pact_new = poiss_cdf_approx(iright,c_dntp_sum,rscale,rlen,rptr) * occ_r;
		    if (occ_l > 0.0)
			    pact_new += poiss_cdf_approx(ileft,c_dntp_sum,lscale,llen,lptr) * occ_l;
			    //pact_new += poiss_cdf_approx(ileft,c_dntp_sum) * occ_l;

		    // this equation works the way we want if pact is the # of active pol in the well
		    // c_dntp_int is in uM-seconds, and dt is in frames @ 15fps
		    // it calculates the number of incorporations based on the average active polymerase
		    // and the average dntp concentration in the well during this time step.
		    // note c_dntp_int is the integral of the dntp concentration in the well during the
		    // time step, which is equal to the average [dntp] in the well times the timestep duration

		    totgen -= (pact+pact_new) * c_dntp_int/2.0;
		    pact = pact_new;
		}
	    }
	
            if (totgen < 0.0) totgen = 0.0;
        }
    
        bead_smem[fnum * num_pts + i] = sign_and_sens*(totocc-totgen);
    }

    __syncthreads();

    // Coalesced output write from global memory
    for (int i = 0; i < num_pts * num_fb; i += num_fb)
        ivalPtr[i + threadIdx.x] = bead_smem[i + threadIdx.x];
}


__global__
void MultiCycleNNModelFluxPulse_tshiftPartialDeriv_k(int* active_bead_list, int num_active_beads, float* fval, bead_params* p, reg_params* reg_p, float* ev, float* sbg, float* deltaFrame, int* buff_flow, int* flow_ndx_map, int num_fb, int num_pts, int num_steps, int step)
{
    // Get bead index
    int ibd = active_bead_list[blockIdx.x];

    // Adjust pointers
    fval += ibd * (num_fb * num_pts * num_steps) + (step * (num_pts * num_fb));
    p += ibd;

    int nnum = 0;
    int fnum;

    float *vb_out;

    float dv, dvn, dv_rs;
    float *ve;

    // constants used for solving background signal shape
    float tau;
    float afact_denom = 0;

    float gain = p->gain;

    float R;
    int i;
    float dt;

    fnum = threadIdx.x;

    float* em = &ev[(p->WhichEmphasis[fnum])*num_pts];

    nnum = flow_ndx_map[fnum];

    vb_out = fval + fnum * num_pts; // get ptr to start of the function evaluation for the current flow
    ve = &sbg[fnum * num_pts]; // get ptr to pre-shifted slope

    // calculate some constants used for this flow
    R = p->R * reg_p->NucModifyRatio[nnum] + reg_p->RatioDrift * buff_flow[fnum] * (1.0 - p->R*reg_p->NucModifyRatio[nnum])*0.001;
    tau = reg_p->tau_R_m*R + reg_p->tau_R_o;

    clamp(tau, (float)MINTAUB, (float)MAXTAUB);

    dv = 0.0;
    dv_rs = 0.0;
    dt = -1.0;

    for (i = 0; i < num_pts; i++)
    {
	    dt = deltaFrame[i];
	    afact_denom = 1.0 + dt / (2.0 * tau);

	    // calculate new dv
	    dvn = ((R - 1.0) * (ve[i]) - dv_rs / tau - dt * dv / (2.0 * tau)) / afact_denom;
	    dv_rs += (dv + dvn) * dt / 2.0;
	    dv = dvn;

	    vb_out[i] = (dv + ((ve[i]))) * gain * em[i];
    }
}

template <int beads_per_block>
__global__ void
InitializeArraysForBinarySearch_k(bool restart, bead_params* params_nn_cuda, float* ac, float* step, 
   int* active_bead_list, int num_beads, int num_fb, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int i = threadIdx.x;

    ac += num_fb*ibd;
    step += num_fb*ibd;
    params_nn_cuda += ibd;

    if (restart) {
        ac[i] = 0.5;
        step[i] = 1.0;        
    }
    else {
        ac[i] = params_nn_cuda->Ampl[i];
        step[i] = 0.02;
    }
    params_nn_cuda->WhichEmphasis[i] = 0;
}

template <int beads_per_block>
__global__ void
UpdateAmplitudeForEvaluation_k(int* active_bead_list, float* ac, bead_params* p, int num_beads, int num_fb, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;
    
    int ibd = active_bead_list[idx];
    int i = threadIdx.x;

    ac += num_fb*ibd;
    p += ibd;

    p->Ampl[i] = ac[i];
}

__global__
void ErrorForBinarySearch_k(float* fval, bead_params* eval_params, float* ev, int* active_bead_list,
        FG_BUFFER_TYPE* fg_buffers, float* err, int num_fb, int num_pts, int num_steps)
{

    extern __shared__ float sFval[];
    float *sfg = &sFval[num_fb*num_pts];

    // Bead index
    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    // Per bead pointer offset
    int beadOffset = ibd * num_fb * num_pts;
    int flowOffset = fnum*num_pts;
    fval += beadOffset * num_steps;
    err += ibd * num_fb;
    fg_buffers += beadOffset;
 
    int i;  

    // copy from global to shared
    for (i=0; i<num_pts*num_fb; i+=num_fb) {
        sFval[i + fnum] = fval[i + fnum];
        sfg[i + fnum] = fg_buffers[i + fnum];
    }
    __syncthreads();
 
    float eval;
    int emndx = eval_params[ibd].WhichEmphasis[fnum];
    float *em = &ev[emndx*num_pts];
    float flow_err = 0.0;

    float* shiftedFval = &sFval[flowOffset];
    float* shiftedFg = &sfg[flowOffset];
    for (i = 0; i < num_pts; i++)
    {
        eval = (shiftedFg[i] - shiftedFval[i])*em[i];
        flow_err += eval*eval;
    }
    err[fnum] = sqrt(flow_err/(float)num_pts);
}

__global__
void ErrorForBinarySearchNew_k(float* fval, bead_params* eval_params, float* ev, int* active_bead_list,
        FG_BUFFER_TYPE* fg_buffers, float* err, int num_fb, int num_pts, int num_steps)
{

    extern __shared__ float sFlowErr[];

    // Bead index
    int ibd = active_bead_list[blockIdx.x];

    // Per bead pointer offset
    int beadOffset = ibd * num_fb * num_pts;
    fval += beadOffset * num_steps;
    err += ibd * num_fb;
    fg_buffers += beadOffset;
    eval_params += ibd;
 
    int i;  

    float eval;
    int emndx, temp;
    float flow_err = 0.0;

    for (i = 0; i < num_fb; i++)
    {
        temp = i*num_pts + threadIdx.x;
        emndx = eval_params->WhichEmphasis[i]*num_pts + threadIdx.x;
        eval = (fg_buffers[temp] - fval[temp])*ev[emndx];
        sFlowErr[temp] = eval*eval;
    }
    __syncthreads();

    if (threadIdx.x < num_fb) {
      float* ptr = &sFlowErr[threadIdx.x*num_pts];
      for (i = 0; i < num_pts; i++) {
        flow_err += ptr[i];
      }
      err[threadIdx.x] = sqrt(flow_err/(float)num_pts);
    }
}


__global__ void
UpdateAp_k(float* ap, float* ac, int* active_bead_list, int num_fb) {

    int ibd = active_bead_list[blockIdx.x];
    int i = threadIdx.x;

    ap += num_fb*ibd;
    ac += num_fb*ibd;

    ap[i] = ac[i] + 0.00025;
}

__global__ void 
BinarySearchStepOne_k(float* ap, float* ac, float* ep, float* ec, float* step, 
              float min_a, float max_a, int* active_bead_list, int num_fb) {

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    ap += num_fb*ibd;
    ac += num_fb*ibd;
    ec += num_fb*ibd;
    ep += num_fb*ibd;
    step += num_fb*ibd;

    float slope = (ep[fnum] - ec[fnum]) * 4000.0;

    if (slope < 0.0)
        ap[fnum] = ac[fnum] + step[fnum];
    else
        ap[fnum] = ac[fnum] - step[fnum];

    if (ap[fnum] < min_a)
        ap[fnum] = min_a;
    if (ap[fnum] > max_a)
        ap[fnum] = max_a;

}

__global__ void 
BinarySearchStepTwo_k(float* ap, float* ac, float* ep, float* ec, float* step, 
              bool* cont, float min_step, int* active_bead_list, int num_fb) {

    extern __shared__ int done[];

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    ap += num_fb*ibd;
    ac += num_fb*ibd;
    ec += num_fb*ibd;
    ep += num_fb*ibd;
    step += num_fb*ibd;

    done[fnum] = 0;

    if (ep[fnum] < ec[fnum]) {
        ac[fnum] = ap[fnum];
        ec[fnum] = ep[fnum];
    }
    else {
        step[fnum] /= 2.0;
        if (step[fnum] < min_step) {
            done[fnum] = 1;
        }
    }

    __syncthreads();

    if (fnum < 10) {
        done[fnum] += done[fnum + 10];
    }
    __syncthreads();

    if (fnum < 5) {
        done[fnum] += done[fnum + 5];
    }
    __syncthreads();

    if (fnum == 0) {
        done[0] += done[1];
        done[0] += done[2];
        done[0] += done[3];
        done[0] += done[4];

        if (done[0] == num_fb) 
            cont[ibd] = true;
    }
  
}

__global__ void
UpdateAmplitudeAfterBinarySearch_k(bead_params* p, float* ac, int* active_bead_list, int num_fb) {

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    ac += num_fb*ibd;
    p += ibd;

    p->Ampl[fnum] = ac[fnum];
}

template <int beads_per_block>
__global__ void
DetermineSingleFlowFitType_k(bool relaxed_krate, bead_params* p, bool* fitType, float* krate_min, 
    float* krate_max, int* active_bead_list, int krateParams, int num_fb, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;

    fitType += num_fb*ibd;
    krate_max += krateParams*num_fb*ibd + fnum*krateParams;
    krate_min += krateParams*num_fb*ibd + fnum*krateParams;
    p += ibd;

    float fsig = p->Copies * p->Ampl[fnum];
    if (fsig > 2.0) {
       fitType[fnum] = true;
       krate_max[0] = MAX_HPLEN-1;
       krate_min[0] = MINAMPL;
       krate_max[1] = 1.75;
       krate_min[1] = 0.65;
       if (relaxed_krate) {
           krate_max[1] = 1.0;
           krate_min[1] = 0.1;
       }
    }
}

template <int beads_per_block>
__global__ void
SetWellRegionParams_k(bead_params* p, reg_params* rp, float* ampParams, 
    float* krateParams, float* lambda, int* active_bead_list,  
    float* SP, float* sens, float* tauB, int* flow_ndx_map, int* buff_flow, 
    int num_fb, int num_pts, int numAmpParams, int numKrateParams, int num_active_beads) {
    
    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;
    int nnum = flow_ndx_map[fnum]; 

    int beadOffset = ibd*num_fb;
    p += ibd;
    tauB += beadOffset; 
    sens += beadOffset; 
    SP += beadOffset;
    lambda += beadOffset;
    ampParams += (beadOffset + fnum) * numAmpParams;
    krateParams += (beadOffset + fnum) * numKrateParams;

    // initialize lambda
    lambda[fnum] = 1E-20;

    float Rval;
    float tau;
    // initialize params
    ampParams[0] = p->Ampl[fnum];   
    krateParams[0] = p->Ampl[fnum];   
    krateParams[1] = p->kmult[fnum];   

    float temp = p->R * rp->NucModifyRatio[nnum];
    Rval = temp + rp->RatioDrift * buff_flow[fnum] 
                             * (1.0-temp) / SCALEOFBUFFERINGCHANGE; 
    tau = (rp->tau_R_m*Rval + rp->tau_R_o); 
    clamp(tau, (float)MINTAUB, (float)MAXTAUB);

    SP[fnum] = (float)(COPYMULTIPLIER * p->Copies)*pow(rp->CopyDrift,buff_flow[fnum]);
    sens[fnum] = rp->sens *SENSMULTIPLIER;
 
    tauB[fnum] = tau;
}

__global__ void
SingleFlowBkgCorrect_k(bead_params* p, reg_params* rp, float* sbg, float* xtflux, float* dark_matter_compensator,
     float* fval, FG_BUFFER_TYPE* fgbuffers, float* deltaFrame, float* tauB, int* flow_ndx_map,
     int* buff_flow, int* active_bead_list, int num_fb, int num_pts) {

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;
    
    int nnum = flow_ndx_map[fnum]; 

    // running sums for integral terms
    float dv,dvn,dv_rs;

    int i;
    float dt, darkness;
    float* et;

    int beadOffset = ibd*num_fb*num_pts;
    int flowOffset = fnum*num_pts;

    p += ibd;
    sbg += flowOffset;
    fval += beadOffset + flowOffset;
    fgbuffers += beadOffset + flowOffset;
    xtflux += beadOffset + flowOffset;
    tauB += ibd*num_fb;
    
    darkness = rp->darkness[0];
    et = &dark_matter_compensator[nnum*num_pts];

    float gain = p->gain;
    float temp = p->R * rp->NucModifyRatio[nnum];
    float shiftRatio = (temp + (rp->RatioDrift * buff_flow[fnum] * (1.0-temp)) / SCALEOFBUFFERINGCHANGE) - 1.0; 
    float tau = tauB[fnum]; 
    float scaledTau = 2.0 * tau;
    dv = 0.0;
    dv_rs = 0.0;
    dt = -1.0;
    dvn = 0.0;

    float curSbgVal;
    float aval;
    for (i=0; i<num_pts; i++)
    {
        dt = deltaFrame[i];
        aval = dt/scaledTau;

	// calculate new dv
        curSbgVal = sbg[i]+xtflux[i];
	dvn = (shiftRatio*curSbgVal - dv_rs/tau - dv*aval) / (1.0 + aval);
	dv_rs += (dv+dvn)*dt*0.5;
	dv = dvn;

	fval[i] = ((float)(fgbuffers[i])-((dv+curSbgVal)*gain + darkness*et[i]));
    }
    
}

__global__ void
SetWeightVector_k(bead_params* p, float* ev, float* weight, float* wtScale, int* active_bead_list, 
    int num_fb, int num_pts, int num_ev) {

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    p += ibd;
    weight += ibd*num_fb*num_pts + fnum*num_pts;
    wtScale += ibd*num_fb;
   
#ifdef COPY_COUNT_EMP_SCALING
    float evSel = p->Ampl[fnum]*p->Copies/P_EMP_SCALE;
#else
    float evSel = p->Ampl[fnum];
#endif

    float val;
    float sum = 0.0;
    if (evSel < (num_ev - 1)) {
        int left = (int)evSel;
        float f1 = (left + 1.0 - evSel);
        if (left < 0) {
            left = 0;
            f1 = 1.0;
        } 
        float* ev1 = &ev[left*num_pts];
        float* ev2 = &ev[(left + 1)*num_pts];
        for (int i=0; i<num_pts; ++i) {
            val = f1 * ev1[i] + (1 - f1) * ev2[i];
            weight[i] = val;
            sum += val;
        }
    }
    else {
        float* ev1 = &ev[(num_ev - 1)*num_pts];
        for (int i=0; i<num_pts; ++i) {
            val =  ev1[i];
            weight[i] = val;
            sum += val;
        }
    } 
    wtScale[fnum] = sum;
}

template <int beads_per_block>
__global__ void
EvaluateFunc_beadblocks_k(bool* flowDone, bool* iterDone, bool* isKrate, bead_params* p, 
    reg_params* reg_p, float* f1, float* k1, float* fval, float* deltaFrame, 
    float* sens_cuda, float* tauB, float* SP, float* c_dntp_top_pc, int* i_start, 
    int* flow_ndx_map, int* buff_flow, int* active_bead_list, 
    float* exp_approx_table_cuda, int exp_approx_table_size, int num_fb, int num_pts, 
    int ampParams, int krateParams, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;
    
    int nnum = flow_ndx_map[fnum]; 

    int beadOffset = ibd*num_fb;
    int start = i_start[nnum];

    p += ibd;
    fval += (beadOffset + fnum)*num_pts;
    tauB += beadOffset;
    SP += beadOffset;
    sens_cuda += beadOffset;
    f1 += (beadOffset + fnum)*ampParams;
    k1 += (beadOffset + fnum)*krateParams;
    isKrate += beadOffset;
    c_dntp_top_pc += nnum*num_pts*ISIG_SUB_STEPS_SINGLE_FLOW; 
    flowDone += beadOffset;    
    iterDone += beadOffset;    

    if (flowDone[fnum] || iterDone[fnum])
        return;

    float A, kmult;
    float sign_and_sens = 1.0 * sens_cuda[fnum];
    if (isKrate[fnum] == true) {
        A = k1[0];
        kmult = k1[1];
    }
    else {
        A = f1[0];
        kmult = p->kmult[fnum];
    } 
    
    if (A < 0.0) {
        A *= -1.0;
        sign_and_sens *= -1.0;
    }
    else if (A > MAX_HPLEN) {
        A = MAX_HPLEN;
    }
     
    float tau = tauB[fnum];
 
    int ileft, iright;
    float ifrac;

    // step 2
    float occ_l,occ_r;
    float totocc;
    float totgen;
    float pact;
    float ival;
    int i;

    // step 3
    float ldt;
    int st;

    // step 4
    float alpha;
    float c_dntp_int;
    float expval;
    float pact_new;
    float C;

    // variables used for solving background signal shape
    float dv = 0.0;
    float dvn = 0.0;
    float dv_rs = 0.0;

    int c_dntp_top_ndx = ISIG_SUB_STEPS_SINGLE_FLOW*start;

    float c_dntp_sum = 0.0;
    float d = reg_p->d[nnum]*p->dmult;
    float kr = reg_p->krate[nnum]*kmult;
    float kmax = reg_p->kmax[nnum];
    float sp = SP[fnum];

    // initialize diffusion/reaction simulation for this flow
    ileft = (int) A;
    iright = ileft + 1;
    ifrac = iright - A;
    occ_l = ifrac;
    occ_r = A - ileft;


    ileft--;
    iright--;

    if (ileft < 0)
    {
        occ_l = 0.0;
    }
    
    if (iright == MAX_HPLEN)
    {
        iright = ileft;
	occ_r = occ_l;
	occ_l = 0;
    }
   
    occ_l *= sp;
    occ_r *= sp;
    pact = occ_l + occ_r;
    totocc = sp*A; 
    totgen = totocc;


    float rscale, lscale;
    int rlen, llen;
    const float* rptr = precompute_pois_params(iright, rlen, rscale);
    const float* lptr = precompute_pois_params(ileft, llen, lscale);


    float c_dntp_bot = 0.0; // concentration of dNTP in the well
    ival = 0.0;
    float c_dntp_top = 0.0;
    C = reg_p->nuc_shape.C;

    float c_dntp_bot_plus_kmax = 1.0/kmax;
    float last_tmp1 = 0.0;
    float last_tmp2 = 0.0;
    float c_dntp_fast_inc = kr*(C/ (C+kmax));
    float fast_start_threshold = 0.99*C;
    for (i=0; i<start; ++i)
        fval[i] = 0.0;

    float aval;
    float frame;
    float scaledTau = 2.0*tau;
    for (i=start;i < num_pts;i++)
    {
        frame = deltaFrame[i];
        if (totgen > 0.0)
	{
		// get the frame number of this data point (might be fractional because this point could be
		// the average of several frames of data.  This number is the average time of all the averaged
		// data points
		ldt = frame/FRAMESPERSEC;

		// once the [dNTP] pretty much reaches full strength in the well
		// the math becomes much simpler
		if (c_dntp_bot > fast_start_threshold)
		{
		    c_dntp_sum += c_dntp_fast_inc*ldt;

		    pact_new = poiss_cdf_approx(iright,c_dntp_sum,rscale,rlen,rptr)*occ_r;
		    if (occ_l > 0.0)
			    pact_new += poiss_cdf_approx(ileft,c_dntp_sum,lscale,llen,lptr)*occ_l;

		    totgen  -= ((pact+pact_new)*0.5)*c_dntp_fast_inc*ldt;
		    pact = pact_new;

		    //if (pf > totgen) pf = totgen;
		    //totgen -= pf;
		}
		else
		{
		    ldt /= ISIG_SUB_STEPS_SINGLE_FLOW;
		    for (st=1;(st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0);st++)
		    {
	                c_dntp_top = c_dntp_top_pc[c_dntp_top_ndx++];
			//if (c_dntp_top >= 0.0)
			//{
			    alpha = d+kr*pact*n_to_uM_conv*c_dntp_bot_plus_kmax;
			    expval = exp_approx(-alpha*ldt, exp_approx_table_cuda, exp_approx_table_size);
			    c_dntp_bot = c_dntp_bot*expval + d*c_dntp_top*(1-expval)/alpha;

                            c_dntp_bot_plus_kmax = 1.0/(c_dntp_bot + kmax);
                            last_tmp1 = c_dntp_bot * c_dntp_bot_plus_kmax;
                            c_dntp_int = kr*(last_tmp2 + last_tmp1) *ldt/2.0;
                            last_tmp2 = last_tmp1;
                            c_dntp_sum += c_dntp_int;

			    // calculate new number of active polymerase
			    pact_new = poiss_cdf_approx(iright,c_dntp_sum,rscale,rlen,rptr)*occ_r;
			    //pact_new = poiss_cdf_approx(iright,c_dntp_sum)*occ_r;
			    if (occ_l > 0.0)
				    pact_new += poiss_cdf_approx(ileft,c_dntp_sum,lscale,llen,lptr)*occ_l;

			    // this equation works the way we want if pact is the # of active pol in the well
			    // c_dntp_int is in uM-seconds, and dt is in frames @ 15fps
			    // it calculates the number of incorporations based on the average active polymerase
			    // and the average dntp concentration in the well during this time step.
			    // note c_dntp_int is the integral of the dntp concentration in the well during the
			    // time step, which is equal to the average [dntp] in the well times the timestep duration
			    totgen  -= (pact+pact_new) * c_dntp_int * 0.5;

			    //if (pf > totgen) pf = totgen;
			    //totgen -= pf;

			    pact = pact_new;
			//}
		    }
		}
          
                if (totgen < 0.0) totgen = 0.0;
		
	   }
                
           ival = (totocc-totgen)*sign_and_sens;

	// calculate the 'background' part (the accumulation/decay of the protons in the well
	// normally accounted for by the background calc)
	aval = frame/scaledTau;

	// calculate new dv
	dvn = (ival - dv_rs/tau - dv*aval) / (1.0+aval);
	dv_rs += (dv+dvn)*aval*tau;
	fval[i] = dv = dvn;
    }
}

__global__ void
CalcResidualForSingleFLowFit_k(bool* flowDone, float* fgbuffers, float* data, float* err_vec, float* r, 
    float* residualWeight, float* wtScale, int* active_bead_list, int num_fb, int num_pts, 
    int num_active_beads) {

    extern __shared__ float observed[];
    
    float* expected = &observed[num_fb*num_pts];

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    int flowOffset = fnum*num_pts;
    int beadOffset = ibd*num_fb*num_pts;
    int offset = ibd*num_fb;

    fgbuffers += beadOffset;
    data += beadOffset;
 
    int i;
    for (i=0; i<num_pts*num_fb; i+=num_fb) {
        expected[i + fnum] = data[i + fnum];
        observed[i + fnum] = fgbuffers[i + fnum];
    }
    __syncthreads();

    err_vec += beadOffset + flowOffset;
    residualWeight += beadOffset + flowOffset;
    r += offset;
    wtScale += offset;
    flowDone += offset;

    if (flowDone[fnum])
        return;

    double rVal = 0;
    double e;
    float resWt;
    float* sfg = &observed[flowOffset];
    float* sexp = &expected[flowOffset];
    for (i=0; i<num_pts; ++i) {
        resWt = residualWeight[i];
        e = resWt * (sfg[i] - sexp[i]);
        err_vec[i] = e;
        rVal += e*e;
    }
    
    r[fnum] = sqrt(rVal/(double)wtScale[fnum]);
}

__global__ void
CalcResidualForSingleFLowFitNew_k(bool* flowDone, float* fgbuffers, float* data, float* err_vec, float* r, 
    float* residualWeight, float* wtScale, int* active_bead_list, int num_fb, int num_pts, 
    int num_active_beads) {

    extern __shared__ double rvalShared[];
    
    int ibd = active_bead_list[blockIdx.x];

    int beadOffset = ibd*num_fb*num_pts;
    int offset = ibd*num_fb;

 
    int i;

    fgbuffers += beadOffset;
    data += beadOffset;
    err_vec += beadOffset;
    residualWeight += beadOffset;
    r += offset;
    wtScale += offset;
    flowDone += offset;


    double rVal = 0;
    double e;
    int temp;
    for (i=0; i<num_fb; ++i) {
        temp = i*num_pts + threadIdx.x; 
        if (flowDone[i]) {
            continue;
        }

        e = residualWeight[temp] * (fgbuffers[temp] - data[temp]);
        err_vec[temp] = e;
        rvalShared[temp] = e*e;
    }
  
    __syncthreads();
   
    if (threadIdx.x < num_fb) {
        if (flowDone[threadIdx.x]) 
            return;
        float wt = wtScale[threadIdx.x];
        double* ptr = &rvalShared[threadIdx.x*num_pts];
        for (i=0; i<num_pts; ++i) {
          rVal += ptr[i];
        }
        r[threadIdx.x] = sqrt(rVal/(double)wt);
    } 
}


template<int beads_per_block>
__global__ void
ComputeJacobianSingleFlow_k(float* jac, float* tmp, float* fval, float* weight, 
	int paramIdx, int numSingleFitParams, int* active_bead_list, 
        int num_fb, int num_pts, int num_active_beads) {
    
    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int i = threadIdx.x;

    int beadOffset = ibd*num_fb*num_pts;
    weight += beadOffset;
    fval += beadOffset;
    tmp += beadOffset;
    jac += beadOffset*numSingleFitParams;
   
    int fnum;
    int offset = 0; 
    int paramPtsProduct = paramIdx*num_pts + i;
    for (fnum=0; fnum < num_fb; ++fnum) {
        offset = fnum*num_pts + i;
        jac[fnum*num_pts*numSingleFitParams + paramPtsProduct] = 
            (weight[offset]*(tmp[offset] - 
                fval[offset]) * 1000);
    }
}

__global__ void
AdjustParamFitIter_k(bool krate, int paramIdx, float* f1, float* k1, int* active_bead_list, 
    int num_fb, int ampParams, int krateParams) {
  
    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    f1 += (ibd*num_fb + fnum)*ampParams;
    k1 += (ibd*num_fb + fnum)*krateParams;

    if (krate) {
        k1[paramIdx] += 0.001;
    }
    else {
        f1[paramIdx]+= 0.001;
    }
}

__global__ void
AdjustParamBackFitIter_k(bool krate, int paramIdx, float* f1, float* k1, float* f1_orig, float* k1_orig, 
    int* active_bead_list, int num_fb, int ampParams, int krateParams) {
  
    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    f1 += (ibd*num_fb + fnum)*ampParams;
    k1 += (ibd*num_fb + fnum)*krateParams;
    f1_orig += (ibd*num_fb + fnum)*ampParams;
    k1_orig += (ibd*num_fb + fnum)*krateParams;

    if (krate) {
        k1[paramIdx] = k1_orig[paramIdx];
    }
    else {
        f1[paramIdx] = f1_orig[paramIdx];
    }
}

template <int beads_per_block>
__global__ void
ComputeJTJSingleFlow_k(float* jac, double* jtj, bool* flowDone, 
        int ampParams, int krateParams, bool* isKrateFit, 
        int* active_bead_list, int num_fb, int num_pts, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;
    
    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb;
    jac += ((flowOffset + fnum)*num_pts)*(ampParams + krateParams);
    jtj += (flowOffset + fnum)*
        (ampParams*ampParams + krateParams*krateParams);
    flowDone += flowOffset;
    isKrateFit += flowOffset;

    if (flowDone[fnum]) 
        return;

        float* jac1;
        float* jac2;
        if (isKrateFit[fnum]) {
            jac += ampParams*num_pts;
            jtj += ampParams*ampParams;
            for (int i=0; i<krateParams; ++i) {
                jac1 = jac + i*num_pts;
                for (int j=0; j<krateParams; ++j) {
                    jac2 = jac + j*num_pts;
                    float val = 0; 
                    for (int k=0; k<num_pts; ++k) {
                         val += jac1[k] * jac2[k];
                    }
                    jtj[i*krateParams + j] = val;
                }
            }
        }
        else {
            for (int i=0; i<ampParams; ++i) {
                jac1 = jac + i*num_pts;
                for (int j=0; j<ampParams; ++j) {
                    jac2 = jac + j*num_pts;
                    float val = 0; 
                    for (int k=0; k<num_pts; ++k) {
                         val += jac1[k] * jac2[k];
                    }
                    jtj[i*ampParams + j] = val;
                }
            }

        }
}

template <int beads_per_block>
__global__ void
ComputeRHSSingleFlow_k(double* rhs, float* jac, float* err, bool* flowDone, 
        int ampParams, int krateParams, bool* isKrateFit, 
        int* active_bead_list, int num_fb, int num_pts, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb;
    jac += ((flowOffset + fnum)*num_pts)*(ampParams + krateParams);
    rhs += (flowOffset + fnum)*(ampParams + krateParams);
    err += (flowOffset + fnum)*num_pts;
    flowDone += flowOffset;
    isKrateFit += flowOffset;

    if (flowDone[fnum]) 
        return;

        float* jac1;
        if (isKrateFit[fnum]) {
            jac += ampParams*num_pts;
            rhs += ampParams;
            for (int i=0; i<krateParams; ++i) {
                jac1 = jac + i*num_pts;
                float val = 0; 
                for (int k=0; k<num_pts; ++k) {
                    val += jac1[k] * err[k];
                }
                rhs[i] = val;
            }
        }
        else {
            for (int i=0; i<ampParams; ++i) {
                jac1 = jac + i*num_pts;
                float val = 0; 
                for (int k=0; k<num_pts; ++k) {
                    val += jac1[k] * err[k];
                }
                rhs[i] = val;
            }
        }
}

template <int beads_per_block>
__global__ void
ComputeLHSWithLambdaSingleFLow_k(double* lhs, float* lambda, bool* flowDone, 
        bool* iterDone, int ampParams, int krateParams, bool* isKrateFit, 
        int* active_bead_list, int num_fb, int num_pts, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb;
    lhs += (flowOffset + fnum)*
        (ampParams*ampParams + krateParams*krateParams);
    lambda += flowOffset;
    flowDone += flowOffset;
    iterDone += flowOffset;
    isKrateFit += flowOffset;

    if (flowDone[fnum] || iterDone[fnum])
        return;

        if (isKrateFit[fnum]) {
            lhs += ampParams*ampParams;
            for (int i=0; i<krateParams; ++i) {
                lhs[i*krateParams + i] *= (double)(1 + lambda[fnum]);
            }
        }
        else {
            for (int i=0; i<ampParams; ++i) {
                lhs[i*ampParams + i] *= (double)(1 + lambda[fnum]);
            }
        }
}

__device__ void ReComputeLHS(double* lhs, double* jtj, float lambda, int numParams) {
  
    int matSize = numParams*numParams;
    for (int i=0; i<matSize; ++i) {
        lhs[i] = jtj[i];
    }
    for (int i=0; i<numParams; ++i) {
        lhs[i*numParams + i] *= (double)(1 + lambda);
    }
   
}

template <int beads_per_block>
__global__ void
SolveSingleFlow_k(double* lhs, double* rhs, double* jtj, double* delta, float* lambda, bool* flowDone, 
        bool* iterDone, int ampParams, int krateParams, bool* isKrateFit, 
        int* active_bead_list, int num_fb, int num_pts, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb;
    lhs += (flowOffset + fnum)*
        (ampParams*ampParams + krateParams*krateParams);
    jtj += (flowOffset + fnum)*
        (ampParams*ampParams + krateParams*krateParams);
    rhs += (flowOffset + fnum)*(ampParams + krateParams);
    delta += (flowOffset + fnum)*(ampParams + krateParams);
    lambda += flowOffset;
    flowDone += flowOffset;
    iterDone += flowOffset;
    isKrateFit += flowOffset;

    if (flowDone[fnum] || iterDone[fnum]) 
        return;

        float newLambda = lambda[fnum];   

        double localDel0, localDel1;
        if (isKrateFit[fnum]) {
            lhs += ampParams*ampParams;
            jtj += ampParams*ampParams;
            rhs += ampParams;
            delta += ampParams;
            
            double det = (lhs[0]*lhs[3] - lhs[1]*lhs[2]);
            localDel0 = (lhs[3]*rhs[0] - lhs[1]*rhs[1]) / det;
            localDel1 = (-lhs[2]*rhs[0] + lhs[0]*rhs[1]) / det;
            while ((localDel0 != localDel0 || localDel1 != localDel1)) {
                newLambda *= 10.0;
                if (newLambda > 1.0) {
                    lambda[fnum] = newLambda;
                    iterDone[fnum] = true;
                    return;
                }
                ReComputeLHS(lhs, jtj, newLambda, krateParams);
                det = (lhs[0]*lhs[3] - lhs[1]*lhs[2]);
                localDel0 = (lhs[3]*rhs[0] - lhs[1]*rhs[1]) / det;
                localDel1 = (-lhs[2]*rhs[0] + lhs[0]*rhs[1]) / det;
            }
            delta[0] = localDel0;
            delta[1] = localDel1;
        }
        else {
            localDel0 = rhs[0] / lhs[0];
            while (localDel0 != localDel0) {
                newLambda *= 10;
                if (newLambda > 10.0) {
                    lambda[fnum] = newLambda;
                    iterDone[fnum] = true;
                    return;
                }
                ReComputeLHS(lhs, jtj, newLambda, ampParams);
                localDel0 = rhs[0] / lhs[0];
            }        
            delta[0] = localDel0;    
        }
        lambda[fnum] = newLambda;
}

__global__ void
AdjustAndClampParamsSingleFlow_k(float* amp_params_new, float* krate_params_new, float* amp_params_max,
    float* amp_params_min, float* krate_params_max, float* krate_params_min, double* delta, 
    bool* flowDone, bool* iterDone, int ampParams, int krateParams, bool* isKrateFit, int* active_bead_list, 
    int num_fb, int num_pts) {

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb + fnum;
    int krateProduct = flowOffset*krateParams;

    amp_params_new += flowOffset*ampParams;
    krate_params_new += krateProduct;
    krate_params_min += krateProduct;
    krate_params_max += krateProduct;
    delta += (flowOffset)*(ampParams + krateParams);
    flowDone += ibd*num_fb;
    iterDone += ibd*num_fb;
    isKrateFit += ibd*num_fb;

    if (flowDone[fnum] || iterDone[fnum]) 
        return;

        if (isKrateFit[fnum]) {
            delta += ampParams;
            for (int i=0; i<krateParams; ++i) {
                krate_params_new[i] += delta[i];
                clamp(krate_params_new[i], krate_params_min[i], krate_params_max[i]);
            }
        }
        else {
            for (int i=0; i<ampParams; ++i) {
                amp_params_new[i] += delta[i];
                clamp(amp_params_new[i], amp_params_min[i], amp_params_max[i]);
            }
        }
}

__global__ void
CheckForIterationCompletion_k(bool* isKrate, bool* flowsToUpdate, bool* beadItr, bool* flowDone, 
    bool* iterDone, float* r1, float* r2, float* lambda, int* active_bead_list, 
    int num_fb) {

    extern __shared__ int done[];
    
    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb;
    r1 += flowOffset;
    r2 += flowOffset;
    lambda += flowOffset;
    isKrate += flowOffset;
    iterDone += flowOffset;
    flowDone += flowOffset;
    flowsToUpdate += flowOffset;

    done[fnum] = 0;

    float lambdaThreshold = 0.0;
    if (isKrate[fnum])
        lambdaThreshold = 1.0;
    else
        lambdaThreshold = 10.0;

    float lambda_val = lambda[fnum];
    if (iterDone[fnum] || flowDone[fnum])
        done[fnum] = 1;

    if (r2[fnum] < r1[fnum]) {
        iterDone[fnum] = true;
        flowsToUpdate[fnum] = true;
        done[fnum] = 1;
        lambda_val /= 10.0;
        if (lambda_val < FLT_MIN) 
            lambda_val = FLT_MIN;

        lambda[fnum] = lambda_val;
        r1[fnum] = r2[fnum];
    }
    else {
        lambda_val *= 10.0;
        if (lambda_val > lambdaThreshold) {
            iterDone[fnum] = true;
            done[fnum] = 1;
        }
        lambda[fnum] = lambda_val;
    }  
    
    __syncthreads();

    if (fnum < 10) {
        done[fnum] += done[fnum + 10];
    }
    __syncthreads();

    if (fnum < 5) {
        done[fnum] += done[fnum + 5];
    }
    __syncthreads();

    if (fnum == 0) {
        done[0] += done[1];
        done[0] += done[2];
        done[0] += done[3];
        done[0] += done[4];

        if (done[0] == num_fb) 
            beadItr[ibd] = true;
    }
}

__global__ void
DoneTest_k(double* delta, bool* isKrateFit, bool* flowDone, bool* cont_proc, int* active_bead_list, 
    int* done_cnt, int num_fb, int ampParams, int krateParams) {

    extern __shared__ int done[];

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb;
    delta += (flowOffset + fnum)*(ampParams + krateParams);
    isKrateFit += flowOffset;
    flowDone += flowOffset;
    done_cnt += flowOffset;

    done[fnum] = 0;

    if (flowDone[fnum]) {
        done[fnum] = 1;
    }
    else {
        if (isKrateFit[fnum]) {
            delta += ampParams;
        }
    
        if (delta[0]*delta[0] < 0.0000025) 
            done_cnt[fnum] += 1;
        else 
            done_cnt[fnum] = 0;

        if (done_cnt[fnum] > 1) {
            flowDone[fnum] = true;
            done[fnum] = 1;
        }
    }
    __syncthreads();

    if (fnum < 10) {
        done[fnum] += done[fnum + 10];
    }
    __syncthreads();

    if (fnum < 5) {
        done[fnum] += done[fnum + 5];
    }
    __syncthreads();

    if (fnum == 0) {
        done[0] += done[1];
        done[0] += done[2];
        done[0] += done[3];
        done[0] += done[4];

        if (done[0] == num_fb) 
            cont_proc[ibd] = true;
    }

}

__global__ void 
UpdateFinalParamsSingleFlow_k(bead_params* p, float* f1, float* k1, float* ev, float* weight, float* wtScale, 
    bool* isKrate, int* active_bead_list, int num_fb, int num_pts, int num_ev, int ampParams, int krateParams) {
    
    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb + fnum;
    p += ibd;
    f1 += flowOffset*ampParams;
    k1 += flowOffset*krateParams;
    weight += flowOffset*num_pts;
    isKrate += ibd*num_fb;
    wtScale += ibd*num_fb;

    if (isKrate[fnum]) {
        p->Ampl[fnum] = k1[0];
        p->kmult[fnum] = k1[1];
    }
    else {
        p->Ampl[fnum] = f1[0];
    }
    
    float* e = &ev[(num_ev - 1)*num_pts];
    float val;
    float sum = 0.0;
    for (int i=0; i<num_pts; ++i) {
        val = e[i];
        weight[i] = val;
        sum += val;
    } 
    wtScale[fnum] = sum;
}

__global__ void
UpdateParamsAndFlowValsAfterEachIter_k(bool* flowsToUpdate, bool* flowDone, 
    float* fval, float* tmp, float* f1, float* k1, float* f1_orig, float* k1_orig, 
    int* active_bead_list, int num_fb, int num_pts, int ampParams, 
    int krateParams) {

    int ibd = active_bead_list[blockIdx.x];
    int fnum = threadIdx.x;

    int flowOffset = ibd*num_fb + fnum;
    f1 += flowOffset*ampParams;
    k1 += flowOffset*krateParams;
    f1_orig += flowOffset*ampParams;
    k1_orig += flowOffset*krateParams;
    fval += flowOffset*num_pts;
    tmp += flowOffset*num_pts;
    flowsToUpdate += ibd*num_fb;
    flowDone += ibd*num_fb;

    if (flowDone[fnum] || !flowsToUpdate[fnum])
        return;

    for (int i=0; i<ampParams; ++i) {
        f1_orig[i] = f1[i];
    }
    for (int i=0; i<krateParams; ++i) {
        k1_orig[i] = k1[i];
    }
    
    for (int i=0; i<num_pts; ++i) {
        fval[i] = tmp[i];
    } 
}

template<int beads_per_block>
__global__ void CalcCDntpTop_k(float* nucRise, int* i_start, bead_params* p,
    reg_params* rp, float* frameNumber, int num_pts, int* active_bead_list, 
    int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.x;
    int nnum = threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];

    i_start += ibd*NUMNUC;
    nucRise += ibd*NUMNUC*num_pts*ISIG_SUB_STEPS_MULTI_FLOW + 
               nnum*num_pts*ISIG_SUB_STEPS_MULTI_FLOW;
    p += ibd;

    int ndx = 0;
    float tlast = 0;
    float last_nuc_value = 0.0;
    int start = -1;
    float C = rp->nuc_shape.C;
    float t_mid_nuc = rp->nuc_shape.t_mid_nuc + rp->nuc_shape.t_mid_nuc_delay[nnum]* (rp->nuc_shape.t_mid_nuc-VALVEOPENFRAME) /TZERODELAYMAGICSCALE; 
    float sigma = rp->nuc_shape.sigma * rp->nuc_shape.sigma_mult[nnum];
    float CProduct = C*0.999;
    float t, tnew, erfdt;
    int i, st;
    for (i=0; (i<num_pts) && (last_nuc_value < CProduct); i++) {

        t = frameNumber[i];

        for (st=1; st <= ISIG_SUB_STEPS_MULTI_FLOW; st++)
        {
            tnew = tlast+(t-tlast)*(float)st/ISIG_SUB_STEPS_MULTI_FLOW;
            erfdt = (tnew-t_mid_nuc)/sigma;

            if (erfdt >= -3)
                last_nuc_value = C*(1.0+erf_approx(erfdt))/2.0;

            nucRise[ndx++] = last_nuc_value;
        }

        if ((start == -1) && (last_nuc_value >= MIN_PROC_THRESHOLD))
            start = i;

        tlast = t;
    }

    for (;ndx < ISIG_SUB_STEPS_MULTI_FLOW*num_pts; ndx++)
        nucRise[ndx] = C;

    i_start[nnum] = start;
}

template<int num_fb, int beads_per_block>
__global__
void BlueSolveBackgroundTrace_k(bead_params* p, reg_params* reg_p, float* vb_out, float* blue_hydrogen, int* flow_ndx_map, float* delta_frame, int* buff_flow, int num_pts, int* active_bead_list, int num_active_beads)
{
    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (bead_index >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int fnum = threadIdx.x;

    // Adjust pointers
    p += ibd;
    vb_out += ibd*num_pts*num_fb + fnum * num_pts; 


    // constants used for solving background signal shape
    float aval = 0;

    int nnum = flow_ndx_map[fnum];

    // calculate some constants used for this flow
    float R = (p->R * reg_p->NucModifyRatio[nnum] + reg_p->RatioDrift * buff_flow[fnum] * (1.0 - p->R*reg_p->NucModifyRatio[nnum])*0.001);
    float shift_ratio = R - 1.0;
    float tau = (reg_p->tau_R_m*R+reg_p->tau_R_o);
    clamp(tau, (float)MINTAUB, (float)MAXTAUB);

    float dv = 0.0;
    float dv_rs = 0.0;
    float dt = -1.0;
    float dvn = 0.0;

    blue_hydrogen += fnum * num_pts;

    float scaleTau = tau * 2.0;
    float curSbgVal;
    for (int i = 0; i < num_pts; i++)
    {
        dt = delta_frame[i];
	aval = dt/scaleTau;

        // calculate new dv
        curSbgVal = blue_hydrogen[i];
        dvn = (shift_ratio * curSbgVal - dv_rs / tau - dv * aval) / (1.0 + aval);
        dv_rs += (dv + dvn) * dt * 0.5;
        dv = dvn;

        vb_out[i] = (dv + curSbgVal) ;

    }

}

__global__
void PostBlueSolveBackgroundTraceSteps_k(bead_params* p, reg_params* reg_p, float* vb_out, 
    float* dark_matter_compensator, int* flow_ndx_map, int num_pts, int num_fb, 
    int* active_bead_list, int num_active_beads)
{
    int bead_index = active_bead_list[blockIdx.x];

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int frame_num = threadIdx.x;

    // Adjust pointers
    p += ibd;
    vb_out += ibd*num_pts*num_fb; 


    float gain = p->gain;
    float darkness = reg_p->darkness[0];
    float temp;
    float* dark_matter_for_flow;
    int nnum;
    for (int i = 0; i < num_fb; i++)
    {
        nnum = flow_ndx_map[i];
        dark_matter_for_flow = &dark_matter_compensator[nnum*num_pts];
         
        temp = vb_out[i*num_pts + frame_num];
        vb_out[i*num_pts + frame_num] = temp * gain + darkness * dark_matter_for_flow[frame_num];
    }

}

__global__
void ObtainBackgroundCorrectedSignal_k(float* correctedSignal, FG_BUFFER_TYPE* fg_buffers, 
    float* ival, int num_pts, int num_fb, int* active_bead_list, int num_active_beads)
{
    int bead_index = active_bead_list[blockIdx.x];

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int frame_num = threadIdx.x;

    // Adjust pointers
    int offset = ibd*num_pts*num_fb;
    ival += offset; 
    fg_buffers += offset;
    correctedSignal += offset;

    int index;
    for (int i = 0; i < num_fb; i++)
    {
        index = i*num_pts + frame_num;
        correctedSignal[index] = (float)(fg_buffers[index]) - ival[index];
    }

}

template<int num_fb, int beads_per_block>
__global__
void RedSolveHydrogenFlowInWellAndAdjustedForGain_k(bead_params* p, reg_params* reg_p, float* vb_out, 
    float* red_hydrogen, int* i_start, int* flow_ndx_map, float* delta_frame, int* buff_flow, int num_pts, 
    int* active_bead_list, int num_active_beads)
{

    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (bead_index >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int fnum = threadIdx.x;

    // Adjust pointers
    p += ibd;
    i_start += ibd*NUMNUC;
    vb_out += ibd*num_pts*num_fb + fnum * num_pts; 
    red_hydrogen += ibd*num_pts*num_fb + fnum * num_pts;


    // constants used for solving background signal shape
    float aval;

    int nnum = flow_ndx_map[fnum];

    // calculate some constants used for this flow
    float gain = p->gain;
    float R = (p->R * reg_p->NucModifyRatio[nnum] + reg_p->RatioDrift * buff_flow[fnum] * (1.0 - p->R*reg_p->NucModifyRatio[nnum])*0.001);
    float tau = (reg_p->tau_R_m*R+reg_p->tau_R_o);
    clamp(tau, (float)MINTAUB, (float)MAXTAUB);

    float dv = 0.0;
    float dv_rs = 0.0;
    float dt;
    float dvn = 0.0;


    float scaleTau = tau * 2.0;
    for (int i = i_start[nnum]; i < num_pts; i++)
    {
        dt = delta_frame[i];
	aval = dt/scaleTau;

        // calculate new dv
        dvn = (red_hydrogen[i] - dv_rs / tau - dv * aval) / (1.0 + aval);
        dv_rs += (dv + dvn) * dt * 0.5;
        dv = dvn;

        vb_out[i] = dv * gain;
    }

}

__global__
void ProjectOnAmplitudeVector_k(float* ampVector, float* correctedSignal, float* model_trace, 
    int num_pts, int num_fb, int* active_bead_list, int num_active_beads)
{
    int bead_index = active_bead_list[blockIdx.x];

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int fnum = threadIdx.x;

    // Adjust pointers
    int offset = ibd*num_pts*num_fb + fnum*num_pts;
    correctedSignal += offset; 
    model_trace += offset;
    ampVector += ibd*num_fb;

    float val;
    float num = 0.0;
    float den = 0.0;
    float A;

    A = ampVector[fnum];
    for (int i = 0; i < num_pts; i++)
    {
        val = model_trace[i];
        num += correctedSignal[i] * val;
        den += val*val;
    }
    
    A *= (num/den);
    clamp(A, 0.001f, (float)MAX_HPLEN);

    ampVector[fnum] = A;
}

__global__
void UpdateProjectionAmplitude_k(bead_params* p, float* ampVector,  int num_fb, 
    int* active_bead_list, int num_active_beads)
{
    int bead_index = active_bead_list[blockIdx.x];

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int fnum = threadIdx.x;

    // Adjust pointers
    p += ibd;
    ampVector += ibd*num_fb;

    p->Ampl[fnum] = ampVector[fnum];
}

__global__
void InitializeProjectionSearchAmplitude_k(float* ampVector, int num_fb, 
    int* active_bead_list, int num_active_beads)
{
    int bead_index = active_bead_list[blockIdx.x];

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int fnum = threadIdx.x;

    // Adjust pointers
    ampVector += ibd*num_fb;

    ampVector[fnum] = 1;
}

template <int beads_per_block>
__global__
void ComputeCumulativeIncorporationHydrogensForProjection_k(bead_params* p, reg_params* reg_p, float* ivalPtr, float* Amp, int* flow_ndx_map, float* deltaFrame, int* buff_flow, int num_pts, int num_fb, int* active_bead_list, int num_active_beads, float* exp_approx_table_cuda, int exp_approx_table_size, float* nucRise, int* i_start)
{
    // Shared memory buffer
    extern __shared__ float smem[];

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;
    int nnum = flow_ndx_map[fnum];

    // Adjust pointers
    p += ibd;
    Amp += ibd*num_fb;
    ivalPtr += ibd * (num_fb * num_pts);
    i_start += ibd*NUMNUC;
    float* bead_smem = &smem[ threadIdx.y * (num_fb * num_pts) ];


    float A;
    float c_dntp_sum;
    int ileft;
    int iright;
    float ifrac;

    // step 2
    float occ_l, occ_r;
    float totocc;
    float totgen;
    float pact;
    float c_dntp_bot;
    int i, st;

    // step 3
    float ldt;

    // step 4
    float c_dntp_top;
    float alpha;
    float c_dntp_int;
    float expval;
    float pact_new;

    float SP = (float)(COPYMULTIPLIER * p->Copies)*pow(reg_p->CopyDrift,buff_flow[fnum]);
    float d = reg_p->d[nnum]*p->dmult;
    float sens = reg_p->sens*SENSMULTIPLIER;
    float kr = reg_p->krate[nnum] * p->kmult[fnum];
    float kmax = reg_p->kmax[nnum];

    A = Amp[fnum];
    if (A > MAX_HPLEN) 
	    A = MAX_HPLEN;


    // initialize diffusion/reaction simulation for this flow
    ileft = (int) A;
    iright = ileft + 1;
    ifrac = iright - A;
    occ_l = ifrac;
    occ_r = A - ileft;


    ileft--;
    iright--;

    if (ileft < 0)
    {
        occ_l = 0.0;
    }
    
    if (iright == MAX_HPLEN)
    {
        iright = ileft;
	occ_r = occ_l;
	occ_l = 0;
    }
   
    occ_l *= SP;
    occ_r *= SP;
    pact = occ_l + occ_r;
    totocc = SP*A; 
    totgen = totocc;

    c_dntp_bot = 0.0; // concentration of dNTP in the well
    c_dntp_top = 0.0;
    c_dntp_sum = 0.0;
    float C = reg_p->nuc_shape.C;

    // some pre-computed things
    float c_dntp_bot_plus_kmax = 1.0/kmax;
    float last_tmp1 = 0.0;
    float last_tmp2 = 0.0;
    float c_dntp_fast_inc = kr*(C/ (C+kmax));

    // [dNTP] in the well after which we switch to simpler model
    float fast_start_threshold = 0.99*C;
    nucRise += ibd*NUMNUC*num_pts*ISIG_SUB_STEPS_MULTI_FLOW + nnum*num_pts*
	    ISIG_SUB_STEPS_MULTI_FLOW;
    int start = i_start[nnum];
    // first non-zero index of the computed [dNTP] array for this nucleotide
    int c_dntp_top_ndx = start*ISIG_SUB_STEPS_MULTI_FLOW;

    float rscale, lscale;
    int rlen, llen;
    const float* rptr = precompute_pois_params(iright, rlen, rscale);
    const float* lptr = precompute_pois_params(ileft, llen, lscale);

    for(i=0; i<start; ++i)
	    bead_smem[fnum*num_pts + i] = 0;

    for (i = start; i < num_pts; i++)
    {
        if (totgen > 0.0) {
            ldt = deltaFrame[i]/FRAMESPERSEC;

	    // once the [dNTP] pretty much reaches full strength in the well
	    // the math becomes much simpler
	    if (c_dntp_bot > fast_start_threshold)
	    {
                c_dntp_sum += c_dntp_fast_inc * ldt;
		pact_new = poiss_cdf_approx(iright, c_dntp_sum, rscale, rlen, rptr) * occ_r;
		if (occ_l > 0.0)
			pact_new += poiss_cdf_approx(ileft, c_dntp_sum, lscale, llen, lptr) * occ_l;

		totgen -= ((pact + pact_new) / 2.0) * c_dntp_fast_inc * ldt;
		pact = pact_new;
	    }
	    else
	    {
	        ldt /= ISIG_SUB_STEPS_MULTI_FLOW;
		for (st = 1; (st <= ISIG_SUB_STEPS_MULTI_FLOW) && (totgen > 0.0); st++)
		{
		    c_dntp_top = nucRise[c_dntp_top_ndx++];

		    alpha = d + kr*pact*n_to_uM_conv*c_dntp_bot_plus_kmax;
		    expval = exp_approx(-alpha * ldt, exp_approx_table_cuda, exp_approx_table_size);

		    c_dntp_bot = c_dntp_bot*expval + d*c_dntp_top * (1 - expval) / alpha;

		    c_dntp_bot_plus_kmax = 1.0/(c_dntp_bot + kmax);
		    last_tmp1 = c_dntp_bot * c_dntp_bot_plus_kmax;
		    c_dntp_int = kr*(last_tmp2 + last_tmp1) *ldt/2.0;
		    last_tmp2 = last_tmp1;
		    c_dntp_sum += c_dntp_int;

		    // calculate new number of active polymerase
		    //pact_new = poiss_cdf_approx(iright,c_dntp_sum) * occ_r;
		    pact_new = poiss_cdf_approx(iright,c_dntp_sum,rscale,rlen,rptr) * occ_r;
		    if (occ_l > 0.0)
			    pact_new += poiss_cdf_approx(ileft,c_dntp_sum,lscale,llen,lptr) * occ_l;
			    //pact_new += poiss_cdf_approx(ileft,c_dntp_sum) * occ_l;

		    // this equation works the way we want if pact is the # of active pol in the well
		    // c_dntp_int is in uM-seconds, and dt is in frames @ 15fps
		    // it calculates the number of incorporations based on the average active polymerase
		    // and the average dntp concentration in the well during this time step.
		    // note c_dntp_int is the integral of the dntp concentration in the well during the
		    // time step, which is equal to the average [dntp] in the well times the timestep duration

		    totgen -= (pact+pact_new) * c_dntp_int/2.0;
		    pact = pact_new;
		}
	    }
	
            if (totgen < 0.0) totgen = 0.0;
        }
    
        bead_smem[fnum * num_pts + i] = sens*(totocc-totgen);
    }

    __syncthreads();

    // Coalesced output write from global memory
    for (int i = 0; i < num_pts * num_fb; i += num_fb)
        ivalPtr[i + threadIdx.x] = bead_smem[i + threadIdx.x];
}

template<int num_fb, int beads_per_block>
__global__
void RedHydrogenForXtalk_k(bead_params* p, reg_params* reg_p, float* vb_out, 
    float* red_hydrogen, float tau, int* i_start, float* delta_frame, int* flow_ndx_map,
    int num_pts, int* active_bead_list, int num_active_beads)
{

    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (bead_index >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[bead_index];

    // per thread index
    int fnum = threadIdx.x;

    // Adjust pointers
    p += ibd;
    i_start += ibd*NUMNUC;
    vb_out += ibd*num_pts*num_fb + fnum * num_pts; 
    red_hydrogen += ibd*num_pts*num_fb + fnum * num_pts;


    // constants used for solving background signal shape
    float aval;

    int nnum = flow_ndx_map[fnum];

    float dv = 0.0;
    float dv_rs = 0.0;
    float dt;
    float dvn = 0.0;


    float scaleTau = tau * 2.0;
    int i = i_start[nnum];
    for (; i < num_pts; i++)
    {
        dt = delta_frame[i];
	aval = dt/scaleTau;

        // calculate new dv
        dvn = (red_hydrogen[i] - dv_rs / tau - dv * aval) / (1.0 + aval);
        dv_rs += (dv + dvn) * dt * 0.5;
        vb_out[i] = dv = dvn;

    }

}

template <int block_size>
__global__ 
void DiminishIncorporationTraceForXtalk_k(float* incorp_trace, float* model_trace, 
    int len) {

    int id = blockIdx.x*block_size + threadIdx.x;
    if (id < len) {
        incorp_trace[id] -= model_trace[id];
    }
}

template <int block_size>
__global__ 
void ApplyXtalkMultiplier_k(float* model_trace, float multiplier, int len) {

    int id = blockIdx.x*block_size + threadIdx.x;
    if (id < len) {
        model_trace[id] *= multiplier;
    }
}

template <int beads_per_block>
__global__
void ComputeXtalkTraceForEveryBead_k(float* xtflux, float* nei_trace, int dataSize, int NeiSize, 
    int* numNeis, int* neiIdxMap, int* active_bead_list, int num_active_beads, int beadOffset) {

    int bead_index = (blockIdx.x * beads_per_block) + threadIdx.x;

    // Bounds check
    if (bead_index >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[bead_index];
   
    int neis = numNeis[ibd];
    
    xtflux += ibd*beadOffset;
    neiIdxMap += ibd*NeiSize*2;

    int i, j;
    int nei_bead_idx;
    int nei_idx;
    float* trace;
    for (i=0; i<neis; ++i) {
        nei_bead_idx = neiIdxMap[i*2 + 0];
        nei_idx = neiIdxMap[i*2 + 1];
        trace = nei_trace + nei_idx*dataSize + nei_bead_idx*beadOffset;
        for (j=0; j<beadOffset; ++j) {
           xtflux[j] += trace[j];
        }
    }

}

template <int beads_per_block>
__global__
void MultiFlowComputeCumulativeIncorporationSignalNew_CUDA_k(bead_params* p, reg_params* reg_p, float* ivalPtr, int* flow_ndx_map, float* deltaFrame, int* buff_flow, int num_pts, int num_fb, int* active_bead_list, int num_active_beads, float* exp_approx_table_cuda, int exp_approx_table_size, float* nucRise, int* i_start)
{
    // Shared memory buffer
    extern __shared__ float smem[];

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    // Bead index
    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;
    int nnum = flow_ndx_map[fnum];

    // Adjust pointers
    p += ibd;
    ivalPtr += ibd * (num_fb * num_pts);
    i_start += ibd*NUMNUC;
    float* bead_smem = &smem[ threadIdx.y * (num_fb * num_pts) ];


    float A;
    int ileft;
    int iright;
    float ifrac;

    // step 2
    float occ_l, occ_r;
    float totocc;
    float totgen;
    float pact;
    int i, st;

    // step 3
    float ldt;

    // step 4
    float pact_new;
    float sign_and_sens;

    float SP = (float)(COPYMULTIPLIER * p->Copies)*pow(reg_p->CopyDrift,buff_flow[fnum]);

    float sens = reg_p->sens*SENSMULTIPLIER;
    float d = reg_p->d[nnum]*p->dmult;
    float kr = reg_p->krate[nnum] * p->kmult[fnum];
    float kmax = reg_p->kmax[nnum];

    A = p->Ampl[fnum];
    sign_and_sens = sens;
    if (A < 0.0) {
            A *= -1.0;
            sign_and_sens *= -1.0;
    }
    else if (A > MAX_HPLEN) {
        A = MAX_HPLEN;
    }

    // initialize diffusion/reaction simulation for this flow
    ileft = (int) A;
    iright = ileft + 1;
    ifrac = iright - A;
    occ_l = ifrac;
    occ_r = A - ileft;


    ileft--;
    iright--;

    if (ileft < 0)
    {
        occ_l = 0.0;
    }

    if (iright == MAX_HPLEN)
    {
        iright = ileft;
        occ_r = occ_l;
        occ_l = 0;
    }

    occ_l *= SP;
    occ_r *= SP;
    pact = occ_l + occ_r;
    totocc = SP*A;
    totgen = totocc;

    float c_dntp_bot = 0.0; // concentration of dNTP in the well
    float c_dntp_top = 0.0;
    float c_dntp_sum = 0.0;
    float c_dntp_int;
    float c_dntp_old_rate = 0;
    float c_dntp_new_rate = 0;

    // [dNTP] in the well after which we switch to simpler model
    nucRise += ibd*NUMNUC*num_pts*ISIG_SUB_STEPS_MULTI_FLOW + nnum*num_pts*
            ISIG_SUB_STEPS_MULTI_FLOW;
    int start = i_start[nnum];

    // first non-zero index of the computed [dNTP] array for this nucleotide
    int c_dntp_top_ndx = start*ISIG_SUB_STEPS_MULTI_FLOW;
    float c_dntp_bot_plus_kmax = 1.0/kmax;

    float rscale, lscale;
    int rlen, llen;
    const float* rptr = precompute_pois_params(iright, rlen, rscale);
    const float* lptr = precompute_pois_params(ileft, llen, lscale);
    float scaled_kr = kr*n_to_uM_conv/d;
    float half_kr = kr * 0.5;

    for(i=0; i<start; ++i)
            bead_smem[fnum*num_pts + i] = 0;

    for (i = start; i < num_pts; i++)
    {
        if (totgen > 0.0)
        {
             ldt = deltaFrame[i]/FRAMESPERSEC;
             ldt = (ldt / ISIG_SUB_STEPS_MULTI_FLOW) * half_kr;
             for (st=1; (st <= ISIG_SUB_STEPS_MULTI_FLOW) && (totgen > 0.0);st++) {
                 c_dntp_top = nucRise[c_dntp_top_ndx];
		 c_dntp_top_ndx += 1;

		 // assume instantaneous equilibrium
		 c_dntp_old_rate = c_dntp_new_rate; 
		 c_dntp_bot = c_dntp_top/(1 + scaled_kr*pact*c_dntp_bot_plus_kmax);
		 c_dntp_bot_plus_kmax = 1.0/(c_dntp_bot + kmax);

		 c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
		 c_dntp_int = ldt*(c_dntp_new_rate+c_dntp_old_rate);
		 c_dntp_sum += c_dntp_int;

		 // calculate new number of active polymerase
		 pact_new = poiss_cdf_approx(iright,c_dntp_sum,rscale,rlen,rptr) * occ_r;
		 if (occ_l > 0.0)
			 pact_new += poiss_cdf_approx(ileft,c_dntp_sum,lscale,llen,lptr) * occ_l;

		 totgen -= ((pact+pact_new) / 2.0) * c_dntp_int;
		 pact = pact_new;
              }
               
              if (totgen < 0.0) totgen = 0.0;
         }
         bead_smem[fnum * num_pts + i] = sign_and_sens*(totocc-totgen);
     }
     __syncthreads();

    // Coalesced output write from global memory
    for (int i = 0; i < num_pts * num_fb; i += num_fb)
        ivalPtr[i + threadIdx.x] = bead_smem[i + threadIdx.x];
}

template <int beads_per_block>
__global__ void
EvaluateFunc_beadblocksNew_k(bool* flowDone, bool* iterDone, bool* isKrate, bead_params* p, 
    reg_params* reg_p, float* f1, float* k1, float* fval, float* deltaFrame, 
    float* sens_cuda, float* tauB, float* SP, float* c_dntp_top_pc, int* i_start, 
    int* flow_ndx_map, int* buff_flow, int* active_bead_list, 
    float* exp_approx_table_cuda, int exp_approx_table_size, int num_fb, int num_pts, 
    int ampParams, int krateParams, int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];
    int fnum = threadIdx.x;
    
    int nnum = flow_ndx_map[fnum]; 

    int beadOffset = ibd*num_fb;
    int start = i_start[nnum];

    p += ibd;
    fval += (beadOffset + fnum)*num_pts;
    tauB += beadOffset;
    SP += beadOffset;
    sens_cuda += beadOffset;
    f1 += (beadOffset + fnum)*ampParams;
    k1 += (beadOffset + fnum)*krateParams;
    isKrate += beadOffset;
    c_dntp_top_pc += nnum*num_pts*ISIG_SUB_STEPS_SINGLE_FLOW; 
    flowDone += beadOffset;    
    iterDone += beadOffset;    

    if (flowDone[fnum] || iterDone[fnum])
        return;

    float A, kmult;
    float sign_and_sens = 1.0 * sens_cuda[fnum];
    if (isKrate[fnum] == true) {
        A = k1[0];
        kmult = k1[1];
    }
    else {
        A = f1[0];
        kmult = p->kmult[fnum];
    } 
    
    if (A < 0.0) {
        A *= -1.0;
        sign_and_sens *= -1.0;
    }
    else if (A > MAX_HPLEN) {
        A = MAX_HPLEN;
    }
     
    float tau = tauB[fnum];
 
    int ileft, iright;
    float ifrac;

    // step 2
    float occ_l,occ_r;
    float totocc;
    float totgen;
    float pact;
    int i, st;

    // step 3
    float ldt;

    // step 4
    float c_dntp_int;
    float pact_new;

    // variables used for solving background signal shape
    float dv = 0.0;
    float dvn = 0.0;
    float dv_rs = 0.0;

    int c_dntp_top_ndx = ISIG_SUB_STEPS_SINGLE_FLOW*start;

    float d = reg_p->d[nnum]*p->dmult;
    float kr = reg_p->krate[nnum]*kmult;
    float kmax = reg_p->kmax[nnum];
    float sp = SP[fnum];

    // initialize diffusion/reaction simulation for this flow
    ileft = (int) A;
    iright = ileft + 1;
    ifrac = iright - A;
    occ_l = ifrac;
    occ_r = A - ileft;


    ileft--;
    iright--;

    if (ileft < 0)
    {
        occ_l = 0.0;
    }
    
    if (iright == MAX_HPLEN)
    {
        iright = ileft;
	occ_r = occ_l;
	occ_l = 0;
    }
   
    occ_l *= sp;
    occ_r *= sp;
    pact = occ_l + occ_r;
    totocc = sp*A; 
    totgen = totocc;


    float rscale, lscale;
    int rlen, llen;
    const float* rptr = precompute_pois_params(iright, rlen, rscale);
    const float* lptr = precompute_pois_params(ileft, llen, lscale);


    float c_dntp_bot = 0.0; // concentration of dNTP in the well
    float c_dntp_top = 0.0;
    float c_dntp_sum = 0.0;
    float c_dntp_old_rate = 0;
    float c_dntp_new_rate = 0;

    float c_dntp_bot_plus_kmax = 1.0/kmax;

    float aval;
    float frame;
    float scaledTau = 2.0*tau;
    float scaled_kr = kr*n_to_uM_conv/d;
    float half_kr = kr*0.5;
    for (i=start;i < num_pts;i++)
    {
        frame = deltaFrame[i];
        if (totgen > 0.0)
        {
             ldt = frame/FRAMESPERSEC;
             ldt = (ldt / ISIG_SUB_STEPS_SINGLE_FLOW) * half_kr;
             for (st=1; (st <= ISIG_SUB_STEPS_SINGLE_FLOW) && (totgen > 0.0);st++) {
                 c_dntp_top = c_dntp_top_pc[c_dntp_top_ndx];
		 c_dntp_top_ndx += 1;

		 // assume instantaneous equilibrium
		 c_dntp_old_rate = c_dntp_new_rate; 
		 c_dntp_bot = c_dntp_top/(1 + scaled_kr*pact*c_dntp_bot_plus_kmax);
		 c_dntp_bot_plus_kmax = 1.0/(c_dntp_bot + kmax);

		 c_dntp_new_rate = c_dntp_bot*c_dntp_bot_plus_kmax;
		 c_dntp_int = ldt*(c_dntp_new_rate+c_dntp_old_rate);
		 c_dntp_sum += c_dntp_int;

		 // calculate new number of active polymerase
		 pact_new = poiss_cdf_approx(iright,c_dntp_sum,rscale,rlen,rptr) * occ_r;
		 if (occ_l > 0.0)
			 pact_new += poiss_cdf_approx(ileft,c_dntp_sum,lscale,llen,lptr) * occ_l;

		 totgen -= ((pact+pact_new) / 2.0) * c_dntp_int;
		 pact = pact_new;
              }
               
              if (totgen < 0.0) totgen = 0.0;
         }
              
	// calculate the 'background' part (the accumulation/decay of the protons in the well
	// normally accounted for by the background calc)
	aval = frame/scaledTau;

	// calculate new dv
	dvn = ((totocc-totgen)*sign_and_sens - dv_rs/tau - dv*aval) / (1.0+aval);
	dv_rs += (dv+dvn)*aval*tau;
	fval[i] = dv = dvn;
    }
}

template<int beads_per_block>
__global__ void SplineFitDntpTop_k(float* nucRise, int* i_start, bead_params* p,
    reg_params* rp, float* frameNumber, int num_pts, int* active_bead_list, 
    int num_active_beads) {

    // Active bead list index
    int idx = (blockIdx.x * beads_per_block) + threadIdx.x;
    int nnum = threadIdx.y;

    // Bounds check
    if (idx >= num_active_beads)
        return;

    int ibd = active_bead_list[idx];

    i_start += ibd*NUMNUC;
    nucRise += ibd*NUMNUC*num_pts*ISIG_SUB_STEPS_MULTI_FLOW + 
               nnum*num_pts*ISIG_SUB_STEPS_MULTI_FLOW;
    p += ibd;

    int ndx = 0;
    float tlast = 0;
    float last_nuc_value = 0.0;
    int start = -1;
    float scaled_dt = -1.0;
    float C = rp->nuc_shape.C;
    float t_mid_nuc = rp->nuc_shape.t_mid_nuc+ rp->nuc_shape.t_mid_nuc_delay[nnum]* (rp->nuc_shape.t_mid_nuc-VALVEOPENFRAME) /TZERODELAYMAGICSCALE; 
    float sigma = rp->nuc_shape.sigma * rp->nuc_shape.sigma_mult[nnum] * 3;
    float t, tnew;
    int i, st;
    for (i=0; (i<num_pts) && (scaled_dt < 1); i++) {

        t = frameNumber[i];

        for (st=1; st <= ISIG_SUB_STEPS_MULTI_FLOW; st++)
        {
            tnew = tlast+(t-tlast)*(float)st/ISIG_SUB_STEPS_MULTI_FLOW;
            scaled_dt = (tnew-t_mid_nuc)/sigma + 0.5;

            if (scaled_dt > 0 && scaled_dt <= 1) {
                last_nuc_value = C*scaled_dt*scaled_dt*(3-2*scaled_dt);

            }
            else if (scaled_dt > 1) {
                    last_nuc_value = C;
            }
            nucRise[ndx++] = last_nuc_value;
        }

        if ((start == -1) && (scaled_dt > 0))
            start = i;

        tlast = t;
    }

    for (;ndx < ISIG_SUB_STEPS_MULTI_FLOW*num_pts; ndx++)
        nucRise[ndx] = C;

    i_start[nnum] = start;
}


#endif // BKGMODELCUDAKERNELS_H
