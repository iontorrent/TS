/*
 * Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef cuda_profiler_H
#define cuda_profiler_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Profiler Output Modes
 */
/*DEVICE_BUILTIN*/
typedef enum CUoutput_mode_st
{
    CU_OUT_KEY_VALUE_PAIR  = 0x00, //**< Output mode Key-Value pair format. */
    CU_OUT_CSV             = 0x01  //**< Output mode Comma separated values format. */
}CUoutput_mode;

/**
 * \brief Initialize the profiling::cuProfilerInitialize().
 *
 * cuProfilerInitialize is used to programmatically initialize the profiling. Using
 * this API user can specify config file, output file and output file format. This 
 * API is generally used to profile different set of counters by looping the kernel launch. configFile 
 * parameter can be used to load new set of counters for profiling.
 *
 * Configurations defined initially by environment variable settings are overwritten 
 * by cuProfilerInitialize().
 *
 * Limitation: Profiling APIs do not work when the application is running with any profiler tool. 
 * User must handle error CUDA_ERROR_PROFILER_DISABLED returned by profiler APIs if 
 * application is likely to be used with any profiler tool. 
 *
 * \param configFile - Name of the config file that lists the counters for profiling.
 * \param outputFile - Name of the outputFile where the profiling results will be stored.
 * \param outputMode - outputMode, can be CU_OUT_KEY_VALUE_PAIR or CU_OUT_CSV.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_PROFILER_DISABLED
 * \notefnerr
 *
 * \sa ::cuProfilerStart,
 * \sa ::cuProfilerStop
 */
CUresult CUDAAPI cuProfilerInitialize(const char *configFile, const char *outputFile, CUoutput_mode outputMode);

/**
 * \brief Start the profiling::cuProfilerStart().
 *
 * cuProfilerStart/cuProfilerStop is used to programmatically control the profiling duration. 
 * APIs give added benefit of controlling the profiling granularity i.e. allow profiling to be 
 * done only on selective pieces of code. 
 * 
 * cuProfilerStart() can also be used to selectively start profiling on a particular context even
 * when profiling is NOT ENABLED using environment variable. Profiling structures must be initialized
 * using cuProfilerInitialize() before making a call to cuProfilerStart().
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_PROFILER_DISABLED,
 * ::CUDA_ERROR_PROFILER_ALREADY_STARTED
 * \notefnerr
 *
 * \sa ::cuProfilerInitialize,
 * \sa ::cuProfilerStop
 */
CUresult CUDAAPI cuProfilerStart(void);

/**
 * \brief Stop the profiling::cuProfilerStop().
 *
 * This API can be used in conjunction with cuProfilerStart to selectively profile subsets of the 
 * CUDA program.
 * cuProfilerStop() can also be used to stop profiling for current context even when profiling is 
 * NOT ENABLED using environment variable. Profiling structures must be initialized using 
 * cuProfilerInitialize() before making a call to cuProfilerStop().
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_PROFILER_DISABLED,
 * ::CUDA_ERROR_PROFILER_ALREADY_STOPPED
 * \notefnerr
 *
 * \sa ::cuProfilerInitialize,
 * \sa ::cuProfilerStart
 */
CUresult CUDAAPI cuProfilerStop(void);

#ifdef __cplusplus
};
#endif

#endif

