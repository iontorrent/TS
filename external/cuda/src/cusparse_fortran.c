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

/*
 * This file contains example Fortran bindings for the CUSPARSE library, These
 * bindings have been tested with Intel Fortran 9.0 on 32-bit and 64-bit 
 * Windows, and with g77 3.4.5 on 32-bit and 64-bit Linux. They will likely
 * have to be adjusted for other Fortran compilers and platforms.
 */

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#if defined(__GNUC__)
#include <stdint.h>
#endif /* __GNUC__ */
#include "cuda_runtime.h" /* CUDA public header file     */
#include "cusparse.h"     /* CUSPARSE public header file */

#include "cusparse_fortran_common.h"
#include "cusparse_fortran.h"

/*---------------------------------------------------------------------------*/
/*------------------------- AUXILIARY FUNCTIONS -----------------------------*/
/*---------------------------------------------------------------------------*/
int CUDA_MALLOC(ptr_t *devPtr, int *size){
    void * tdevPtr = 0;
    int error = (int)cudaMalloc(&tdevPtr,(size_t)(*size));
    *devPtr = (ptr_t)tdevPtr;
    return error;
}
int CUDA_FREE(ptr_t *devPtr){
    return (int)cudaFree((void *)(*devPtr));
}
/* WARNING: 
   (i) notice that when passing dstPtr and srcPtr to cudaMemcpy,
   one of the parameters is dereferenced, while the other is not. This
   reflects the fact that for _FORT2C_ dstPtr was allocated by cudaMalloc
   in the wrapper, while srcPtr was allocated in the Fortran code; on the 
   other hand, for _C2FORT_ dstPtr was allocated in the Fortran code, while
   srcPtr was allocated by cudaMalloc in the wrapper. Users should be 
   careful and take great care of this distinction in their own code.
   (ii) there are two functions _INT and _REAL in order to avoid
   warnings from the Fortran compiler due to the fact that arguments of 
   different type are passed to the same function.
*/ 
int CUDA_MEMCPY_FORT2C_INT(ptr_t *dstPtr, const ptr_t *srcPtr, int *count, int *kind){
    return (int)cudaMemcpy((void *)(*dstPtr), (void *)srcPtr, (size_t)(*count), (enum cudaMemcpyKind)(*kind));
} 
int CUDA_MEMCPY_FORT2C_REAL(ptr_t *dstPtr, const ptr_t *srcPtr, int *count, int *kind){
    return (int)cudaMemcpy((void *)(*dstPtr), (void *)srcPtr, (size_t)(*count), (enum cudaMemcpyKind)(*kind));
} 
int CUDA_MEMCPY_C2FORT_INT(ptr_t *dstPtr, const ptr_t *srcPtr, int *count, int *kind){
    return (int)cudaMemcpy((void *)dstPtr, (void *)(*srcPtr), (size_t)(*count), (enum cudaMemcpyKind)(*kind));
} 
int CUDA_MEMCPY_C2FORT_REAL(ptr_t *dstPtr, const ptr_t *srcPtr, int *count, int *kind){
    return (int)cudaMemcpy((void *)dstPtr, (void *)(*srcPtr), (size_t)(*count), (enum cudaMemcpyKind)(*kind));
} 
int CUDA_MEMSET(ptr_t *devPtr, int *value, int *count){
    return (int)cudaMemset((void *)(*devPtr), *value, (size_t)(*count));  
}

void GET_SHIFTED_ADDRESS(ptr_t *originPtr, int *count, ptr_t *resultPtr){
    char * temp = (char *)(*originPtr);
    *resultPtr  = (ptr_t)(temp+(*count));
}

/*---------------------------------------------------------------------------*/
/*------------------------- CUSPARSE FUNCTIONS ------------------------------*/
/*---------------------------------------------------------------------------*/
int CUSPARSE_CREATE(ptr_t *handle){
    cusparseHandle_t thandle = 0; 
    int error= (int)cusparseCreate(&thandle);
    *handle  = (ptr_t)thandle;
    return error;
}
int CUSPARSE_DESTROY(ptr_t *handle){
    return (int)cusparseDestroy((cusparseHandle_t)(*handle));
}
int CUSPARSE_GET_VERSION(ptr_t *handle, int *version){
    return (int)cusparseGetVersion((cusparseHandle_t)(*handle), version);
}
int CUSPARSE_SET_KERNEL_STREAM(ptr_t * handle, ptr_t *streamId){
    return (int)cusparseSetKernelStream((cusparseHandle_t)(*handle), (cudaStream_t)(*streamId)); 
}

int CUSPARSE_CREATE_MAT_DESCR(ptr_t *descrA){
    cusparseMatDescr_t tdescrA = 0;
    int error= (int)cusparseCreateMatDescr(&tdescrA);
    *descrA  = (ptr_t)tdescrA;
    return error;
}
int CUSPARSE_DESTROY_MAT_DESCR(ptr_t *descrA){
    return (int)cusparseDestroyMatDescr((cusparseMatDescr_t)(*descrA));
}

int CUSPARSE_SET_MAT_TYPE(ptr_t *descrA, int *type){
    return (int)cusparseSetMatType((cusparseMatDescr_t)(*descrA), (cusparseMatrixType_t)(*type));
}
int CUSPARSE_GET_MAT_TYPE(const ptr_t *descrA){
    return (int)cusparseGetMatType((const cusparseMatDescr_t)(*descrA));
}
int CUSPARSE_SET_MAT_FILL_MODE(ptr_t *descrA, int *fillMode){
    return (int)cusparseSetMatFillMode((cusparseMatDescr_t)(*descrA), (cusparseFillMode_t)(*fillMode));
}
int CUSPARSE_GET_MAT_FILL_MODE(const ptr_t *descrA){
    return (int)cusparseGetMatFillMode((const cusparseMatDescr_t)(*descrA));
}
int CUSPARSE_SET_MAT_DIAG_TYPE(ptr_t *descrA, int *diagType){
    return (int)cusparseSetMatDiagType((cusparseMatDescr_t)(*descrA), (cusparseDiagType_t)(*diagType));
}
int CUSPARSE_GET_MAT_DIAG_TYPE(const ptr_t *descrA){
    return (int)cusparseGetMatDiagType((const cusparseMatDescr_t)(*descrA));
}
int CUSPARSE_SET_MAT_INDEX_BASE(ptr_t *descrA, int *base){
    return (int)cusparseSetMatIndexBase((cusparseMatDescr_t)(*descrA), (cusparseIndexBase_t)(*base));
}
int CUSPARSE_GET_MAT_INDEX_BASE(const ptr_t *descrA){
    return (int)cusparseGetMatIndexBase((const cusparseMatDescr_t)(*descrA));
}

int CUSPARSE_CREATE_SOLVE_ANALYSIS_INFO(ptr_t *info){
    cusparseSolveAnalysisInfo_t tinfo = 0; 
    int error = (int)cusparseCreateSolveAnalysisInfo(&tinfo);
    *info  = (ptr_t)tinfo;
    return error;
}

int CUSPARSE_DESTROY_SOLVE_ANALYSIS_INFO(ptr_t *info){
    return (int)cusparseDestroySolveAnalysisInfo((cusparseSolveAnalysisInfo_t)(*info));
}

/*---------------------------------------------------------------------------*/
/*------------------------- SPARSE LEVEL 1 ----------------------------------*/
/*---------------------------------------------------------------------------*/
int CUSPARSE_SAXPYI(ptr_t *handle, 
                    int *nnz, 
                    float *alpha, 
                    const ptr_t *xVal, 
                    const ptr_t *xInd, 
                    ptr_t *y, 
                    int *idxBase){
    return (int)cusparseSaxpyi((cusparseHandle_t)(*handle),
                               *nnz,
                               *alpha,
                               (const float *)(*xVal),
                               (const int *)(*xInd),
                               (float *)(*y), 
                               (cusparseIndexBase_t)(*idxBase));
}
    
int CUSPARSE_DAXPYI(ptr_t *handle, 
                    int *nnz, 
                    double *alpha, 
                    const ptr_t *xVal, 
                    const ptr_t *xInd, 
                    ptr_t *y, 
                    int *idxBase){
    return (int)cusparseDaxpyi((cusparseHandle_t)(*handle),
                               *nnz,
                               *alpha,
                               (const double *)(*xVal),
                               (const int *)(*xInd),
                               (double *)(*y), 
                               (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_CAXPYI(ptr_t *handle, 
                    int *nnz, 
                    cuComplex *alpha, 
                    const ptr_t *xVal, 
                    const ptr_t *xInd, 
                    ptr_t *y, 
                    int *idxBase){
    return (int)cusparseCaxpyi((cusparseHandle_t)(*handle),
                               *nnz,
                               *alpha,
                               (const cuComplex *)(*xVal),
                               (const int *)(*xInd),
                               (cuComplex *)(*y), 
                               (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_ZAXPYI(ptr_t *handle, 
                    int *nnz, 
                    cuDoubleComplex *alpha, 
                    const ptr_t *xVal, 
                    const ptr_t *xInd, 
                    ptr_t *y, 
                    int *idxBase){
    return (int)cusparseZaxpyi((cusparseHandle_t)(*handle),
                               *nnz,
                               *alpha,
                               (const cuDoubleComplex *)(*xVal),
                               (const int *)(*xInd),
                               (cuDoubleComplex *)(*y), 
                               (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_SDOTI(ptr_t *handle,
                    int   *nnz,
                    const ptr_t *xVal, 
                    const ptr_t *xInd, 
                    const ptr_t *y,
                    float *resultHostPtr, 
                    int *idxBase){
    return (int)cusparseSdoti((cusparseHandle_t)(*handle),  
                              *nnz, 
                              (const float *)(*xVal), 
                              (const int *)(*xInd), 
                              (const float *)(*y),
                              resultHostPtr, 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_DDOTI(ptr_t *handle,
                   int   *nnz,
                   const ptr_t *xVal, 
                   const ptr_t *xInd, 
                   const ptr_t *y,
                   double *resultHostPtr, 
                   int *idxBase){
    return (int)cusparseDdoti((cusparseHandle_t)(*handle),  
                              *nnz, 
                              (const double *)(*xVal), 
                              (const int *)(*xInd), 
                              (const double *)(*y),
                              resultHostPtr, 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_CDOTI(ptr_t *handle,
                   int   *nnz,
                   const ptr_t *xVal, 
                   const ptr_t *xInd, 
                   const ptr_t *y,
                   cuComplex *resultHostPtr, 
                   int *idxBase){
    return (int)cusparseCdoti((cusparseHandle_t)(*handle),  
                              *nnz, 
                              (const cuComplex *)(*xVal), 
                              (const int *)(*xInd), 
                              (const cuComplex *)(*y),
                              resultHostPtr, 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_ZDOTI(ptr_t *handle,
                   int   *nnz,
                   const ptr_t *xVal, 
                   const ptr_t *xInd, 
                   const ptr_t *y,
                   cuDoubleComplex *resultHostPtr, 
                   int *idxBase){
    return (int)cusparseZdoti((cusparseHandle_t)(*handle),  
                              *nnz, 
                              (const cuDoubleComplex *)(*xVal), 
                              (const int *)(*xInd), 
                              (const cuDoubleComplex *)(*y),
                              resultHostPtr, 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_CDOTCI(ptr_t *handle,
                    int   *nnz,
                    const ptr_t *xVal, 
                    const ptr_t *xInd, 
                    const ptr_t *y,
                    cuComplex *resultHostPtr, 
                    int *idxBase){
    return (int)cusparseCdotci((cusparseHandle_t)(*handle),
                               *nnz, 
                               (const cuComplex *)(*xVal), 
                               (const int *)(*xInd), 
                               (const cuComplex *)(*y), 
                               resultHostPtr, 
                               (cusparseIndexBase_t)*(idxBase));
}

int CUSPARSE_ZDOTCI(ptr_t *handle,
                    int   *nnz,
                    const ptr_t *xVal, 
                    const ptr_t *xInd, 
                    const ptr_t *y,
                    cuDoubleComplex *resultHostPtr, 
                    int *idxBase){
    return (int)cusparseZdotci((cusparseHandle_t)(*handle),
                               *nnz, 
                               (const cuDoubleComplex *)(*xVal), 
                               (const int *)(*xInd), 
                               (const cuDoubleComplex *)(*y), 
                               resultHostPtr, 
                               (cusparseIndexBase_t)*(idxBase));    
}

int CUSPARSE_SGTHR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *y, 
                   ptr_t *xVal, 
                   const ptr_t *xInd, 
                   int *idxBase){
    return (int)cusparseSgthr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const float *)(*y), 
                              (float *)(*xVal), 
                              (const int *)(*xInd), 
                              (cusparseIndexBase_t)(*idxBase));
}
    
int CUSPARSE_DGTHR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *y, 
                   ptr_t *xVal, 
                   const ptr_t *xInd, 
                   int *idxBase){
    return (int)cusparseDgthr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const double *)(*y), 
                              (double *)(*xVal), 
                              (const int *)(*xInd), 
                              (cusparseIndexBase_t)(*idxBase)); 
}

int CUSPARSE_CGTHR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *y, 
                   ptr_t *xVal, 
                   const ptr_t *xInd, 
                   int *idxBase){
    return (int)cusparseCgthr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const cuComplex *)(*y), 
                              (cuComplex *)(*xVal), 
                              (const int *)(*xInd), 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_ZGTHR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *y, 
                   ptr_t *xVal, 
                   const ptr_t *xInd, 
                   int *idxBase){
    return (int)cusparseZgthr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const cuDoubleComplex *)(*y), 
                              (cuDoubleComplex *)(*xVal), 
                              (const int *)(*xInd), 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_SGTHRZ(ptr_t *handle, 
                    int *nnz, 
                    ptr_t *y, 
                    ptr_t *xVal, 
                    const ptr_t *xInd, 
                    int *idxBase){
    return (int)cusparseSgthrz((cusparseHandle_t)(*handle), 
                               *nnz, 
                               (float *)(*y), 
                               (float *)(*xVal), 
                               (const int *)(*xInd), 
                               (cusparseIndexBase_t)(*idxBase));
}
    
int CUSPARSE_DGTHRZ(ptr_t *handle, 
                    int *nnz, 
                    ptr_t *y, 
                    ptr_t *xVal, 
                    const ptr_t *xInd, 
                    int *idxBase){
    return (int)cusparseDgthrz((cusparseHandle_t)(*handle), 
                               *nnz, 
                               (double *)(*y), 
                               (double *)(*xVal), 
                               (const int *)(*xInd), 
                               (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_CGTHRZ(ptr_t *handle, 
                    int *nnz, 
                    ptr_t *y, 
                    ptr_t *xVal, 
                    const ptr_t *xInd, 
                    int *idxBase){
    return (int)cusparseCgthrz((cusparseHandle_t)(*handle), 
                               *nnz, 
                               (cuComplex *)(*y), 
                               (cuComplex *)(*xVal), 
                               (const int *)(*xInd), 
                               (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_ZGTHRZ(ptr_t *handle, 
                    int *nnz, 
                    ptr_t *y, 
                    ptr_t *xVal, 
                    const ptr_t *xInd, 
                    int *idxBase){
    return (int)cusparseZgthrz((cusparseHandle_t)(*handle), 
                               *nnz, 
                               (cuDoubleComplex *)(*y), 
                               (cuDoubleComplex *)(*xVal), 
                               (const int *)(*xInd), 
                               (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_SSCTR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *xVal, 
                   const ptr_t *xInd, 
                   ptr_t *y, 
                   int *idxBase){
    return (int)cusparseSsctr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const float *)(*xVal), 
                              (const int *)(*xInd), 
                              (float *)(*y), 
                              (cusparseIndexBase_t)(*idxBase));
}
    
int CUSPARSE_DSCTR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *xVal, 
                   const ptr_t *xInd, 
                   ptr_t *y, 
                   int *idxBase){
    return (int)cusparseDsctr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const double *)(*xVal), 
                              (const int *)(*xInd), 
                              (double *)(*y), 
                              (cusparseIndexBase_t)(*idxBase));    
}

int CUSPARSE_CSCTR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *xVal, 
                   const ptr_t *xInd, 
                   ptr_t *y, 
                   int *idxBase){
    return (int)cusparseCsctr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const cuComplex *)(*xVal), 
                              (const int *)(*xInd), 
                              (cuComplex *)(*y), 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_ZSCTR(ptr_t *handle, 
                   int *nnz, 
                   const ptr_t *xVal, 
                   const ptr_t *xInd, 
                   ptr_t *y, 
                   int *idxBase){
    return (int)cusparseZsctr((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (const cuDoubleComplex *)(*xVal), 
                              (const int *)(*xInd), 
                              (cuDoubleComplex *)(*y), 
                              (cusparseIndexBase_t)(*idxBase));
}


int CUSPARSE_SROTI(ptr_t *handle, 
                   int *nnz, 
                   ptr_t *xVal, 
                   const ptr_t *xInd, 
                   ptr_t *y, 
                   float *c, 
                   float *s, 
                   int *idxBase){
    return (int)cusparseSroti((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (float *)(*xVal), 
                              (const int *)(*xInd), 
                              (float *)(*y), 
                              *c, 
                              *s, 
                              (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_DROTI(ptr_t *handle, 
                   int *nnz, 
                   ptr_t *xVal, 
                   const ptr_t *xInd, 
                   ptr_t *y, 
                   double *c, 
                   double *s, 
                   int *idxBase){
    return (int)cusparseDroti((cusparseHandle_t)(*handle), 
                              *nnz, 
                              (double *)(*xVal), 
                              (const int *)(*xInd), 
                              (double *)(*y), 
                              *c, 
                              *s, 
                              (cusparseIndexBase_t)(*idxBase));
}

/*---------------------------------------------------------------------------*/
/*------------------------- SPARSE LEVEL 2 ----------------------------------*/
/*---------------------------------------------------------------------------*/
int CUSPARSE_SCSRMV(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    float *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *x, 
                    float *beta, 
                    ptr_t *y){
    return (int)cusparseScsrmv((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const float *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const float *)(*x), 
                               *beta, 
                               (float *)(*y));
}
    
int CUSPARSE_DCSRMV(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    double *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *x, 
                    double *beta, 
                    ptr_t *y){
    return (int)cusparseDcsrmv((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const double *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const double *)(*x), 
                               *beta, 
                               (double *)(*y));
}

int CUSPARSE_CCSRMV(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    cuComplex *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *x, 
                    cuComplex *beta, 
                    ptr_t *y){
    return (int)cusparseCcsrmv((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const cuComplex *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const cuComplex *)(*x), 
                               *beta, 
                               (cuComplex *)(*y));
}

int CUSPARSE_ZCSRMV(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    cuDoubleComplex *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *x, 
                    cuDoubleComplex *beta, 
                    ptr_t *y){
    return (int)cusparseZcsrmv((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const cuDoubleComplex *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const cuDoubleComplex *)(*x), 
                               *beta, 
                               (cuDoubleComplex *)(*y));
}


int CUSPARSE_SCSRSV_ANALYSIS(ptr_t *handle, 
                             int *transA, 
                             int *m, 
                             const ptr_t *descrA, 
                             const ptr_t *csrValA, 
                             const ptr_t *csrRowPtrA, 
                             const ptr_t *csrColIndA, 
                             ptr_t *info){
    return (int)cusparseScsrsv_analysis((cusparseHandle_t)(*handle), 
                                        (cusparseOperation_t)(*transA), 
                                        *m, 
                                        (const cusparseMatDescr_t)(*descrA), 
                                        (const float *)(*csrValA), 
                                        (const int *)(*csrRowPtrA), 
                                        (const int *)(*csrColIndA), 
                                        (cusparseSolveAnalysisInfo_t)(*info));    
}

int CUSPARSE_DCSRSV_ANALYSIS(ptr_t *handle, 
                             int *transA, 
                             int *m, 
                             const ptr_t *descrA, 
                             const ptr_t *csrValA, 
                             const ptr_t *csrRowPtrA, 
                             const ptr_t *csrColIndA, 
                             ptr_t *info){
    return (int)cusparseDcsrsv_analysis((cusparseHandle_t)(*handle), 
                                        (cusparseOperation_t)(*transA), 
                                        *m, 
                                        (const cusparseMatDescr_t)(*descrA), 
                                        (const double *)(*csrValA), 
                                        (const int *)(*csrRowPtrA), 
                                        (const int *)(*csrColIndA), 
                                        (cusparseSolveAnalysisInfo_t)(*info));    
}

int CUSPARSE_CCSRSV_ANALYSIS(ptr_t *handle, 
                             int *transA, 
                             int *m, 
                             const ptr_t *descrA, 
                             const ptr_t *csrValA, 
                             const ptr_t *csrRowPtrA, 
                             const ptr_t *csrColIndA, 
                             ptr_t *info){
    return (int)cusparseCcsrsv_analysis((cusparseHandle_t)(*handle), 
                                        (cusparseOperation_t)(*transA), 
                                        *m, 
                                        (const cusparseMatDescr_t)(*descrA), 
                                        (const cuComplex *)(*csrValA), 
                                        (const int *)(*csrRowPtrA), 
                                        (const int *)(*csrColIndA), 
                                        (cusparseSolveAnalysisInfo_t)(*info));    
}

int CUSPARSE_ZCSRSV_ANALYSIS(ptr_t *handle, 
                             int *transA, 
                             int *m, 
                             const ptr_t *descrA, 
                             const ptr_t *csrValA, 
                             const ptr_t *csrRowPtrA, 
                             const ptr_t *csrColIndA, 
                             ptr_t *info){
    return (int)cusparseZcsrsv_analysis((cusparseHandle_t)(*handle), 
                                        (cusparseOperation_t)(*transA), 
                                        *m, 
                                        (const cusparseMatDescr_t)(*descrA), 
                                        (const cuDoubleComplex *)(*csrValA), 
                                        (const int *)(*csrRowPtrA), 
                                        (const int *)(*csrColIndA), 
                                        (cusparseSolveAnalysisInfo_t)(*info));    
} 

int CUSPARSE_SCSRSV_SOLVE(ptr_t *handle, 
                          int *transA, 
                          int *m, 
                          float *alpha, 
                          const ptr_t *descrA, 
                          const ptr_t *csrValA, 
                          const ptr_t *csrRowPtrA, 
                          const ptr_t *csrColIndA, 
                          ptr_t *info, 
                          const ptr_t *x, 
                          ptr_t *y){
    return (int)cusparseScsrsv_solve((cusparseHandle_t)(*handle), 
                                     (cusparseOperation_t)(*transA), 
                                     *m, 
                                     *alpha, 
                                     (const cusparseMatDescr_t)(*descrA), 
                                     (const float *)(*csrValA), 
                                     (const int *)(*csrRowPtrA), 
                                     (const int *)(*csrColIndA), 
                                     (cusparseSolveAnalysisInfo_t)(*info), 
                                     (const float *)(*x), 
                                     (float *)(*y));    
}

int CUSPARSE_DCSRSV_SOLVE(ptr_t *handle, 
                          int *transA, 
                          int *m, 
                          double *alpha, 
                          const ptr_t *descrA, 
                          const ptr_t *csrValA, 
                          const ptr_t *csrRowPtrA, 
                          const ptr_t *csrColIndA, 
                          ptr_t *info, 
                          const ptr_t *x, 
                          ptr_t *y){
    return (int)cusparseDcsrsv_solve((cusparseHandle_t)(*handle), 
                                     (cusparseOperation_t)(*transA), 
                                     *m, 
                                     *alpha, 
                                     (const cusparseMatDescr_t)(*descrA), 
                                     (const double *)(*csrValA), 
                                     (const int *)(*csrRowPtrA), 
                                     (const int *)(*csrColIndA), 
                                     (cusparseSolveAnalysisInfo_t)(*info), 
                                     (const double *)(*x), 
                                     (double *)(*y));    
}

int CUSPARSE_CCSRSV_SOLVE(ptr_t *handle, 
                          int *transA, 
                          int *m, 
                          cuComplex *alpha, 
                          const ptr_t *descrA, 
                          const ptr_t *csrValA, 
                          const ptr_t *csrRowPtrA, 
                          const ptr_t *csrColIndA, 
                          ptr_t *info, 
                          const ptr_t *x, 
                          ptr_t *y){
    return (int)cusparseCcsrsv_solve((cusparseHandle_t)(*handle), 
                                     (cusparseOperation_t)(*transA), 
                                     *m, 
                                     *alpha, 
                                     (const cusparseMatDescr_t)(*descrA), 
                                     (const cuComplex *)(*csrValA), 
                                     (const int *)(*csrRowPtrA), 
                                     (const int *)(*csrColIndA), 
                                     (cusparseSolveAnalysisInfo_t)(*info), 
                                     (const cuComplex *)(*x), 
                                     (cuComplex *)(*y));  
}

int CUSPARSE_ZCSRSV_SOLVE(ptr_t *handle, 
                          int *transA, 
                          int *m, 
                          cuDoubleComplex *alpha, 
                          const ptr_t *descrA, 
                          const ptr_t *csrValA, 
                          const ptr_t *csrRowPtrA, 
                          const ptr_t *csrColIndA, 
                          ptr_t *info, 
                          const ptr_t *x, 
                          ptr_t *y){
    return (int)cusparseZcsrsv_solve((cusparseHandle_t)(*handle), 
                                     (cusparseOperation_t)(*transA), 
                                     *m, 
                                     *alpha, 
                                     (const cusparseMatDescr_t)(*descrA), 
                                     (const cuDoubleComplex *)(*csrValA), 
                                     (const int *)(*csrRowPtrA), 
                                     (const int *)(*csrColIndA), 
                                     (cusparseSolveAnalysisInfo_t)(*info), 
                                     (const cuDoubleComplex *)(*x), 
                                     (cuDoubleComplex *)(*y));    
}

/*---------------------------------------------------------------------------*/
/*------------------------- SPARSE LEVEL 3 ----------------------------------*/
/*---------------------------------------------------------------------------*/
int CUSPARSE_SCSRMM(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    int *k,  
                    float *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *B, 
                    int *ldb, 
                    float *beta, 
                    ptr_t *C, 
                    int *ldc){
    return (int)cusparseScsrmm((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *k,  
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const float *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const float *)(*B), 
                               *ldb, 
                               *beta, 
                               (float *)(*C), 
                               *ldc);
}
                     
int CUSPARSE_DCSRMM(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    int *k,  
                    double *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *B, 
                    int *ldb, 
                    double *beta, 
                    ptr_t *C, 
                    int *ldc){
    return (int)cusparseDcsrmm((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *k,  
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const double *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const double *)(*B), 
                               *ldb, 
                               *beta, 
                               (double *)(*C), 
                               *ldc);
}

int CUSPARSE_CCSRMM(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    int *k,  
                    cuComplex *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *B, 
                    int *ldb, 
                    cuComplex *beta, 
                    ptr_t *C, 
                    int *ldc){
    return (int)cusparseCcsrmm((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *k,  
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const cuComplex *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const cuComplex *)(*B), 
                               *ldb, 
                               *beta, 
                               (cuComplex *)(*C), 
                               *ldc);
}

int CUSPARSE_ZCSRMM(ptr_t *handle,
                    int *transA, 
                    int *m, 
                    int *n, 
                    int *k,  
                    cuDoubleComplex *alpha,
                    const ptr_t *descrA, 
                    const ptr_t *csrValA, 
                    const ptr_t *csrRowPtrA, 
                    const ptr_t *csrColIndA, 
                    const ptr_t *B, 
                    int *ldb, 
                    cuDoubleComplex *beta, 
                    ptr_t *C, 
                    int *ldc){
    return (int)cusparseZcsrmm((cusparseHandle_t)(*handle),
                               (cusparseOperation_t)(*transA), 
                               *m, 
                               *n, 
                               *k,  
                               *alpha,
                               (const cusparseMatDescr_t)(*descrA), 
                               (const cuDoubleComplex *)(*csrValA), 
                               (const int *)(*csrRowPtrA), 
                               (const int *)(*csrColIndA), 
                               (const cuDoubleComplex *)(*B), 
                               *ldb, 
                               *beta, 
                               (cuDoubleComplex *)(*C), 
                               *ldc);
}

/*---------------------------------------------------------------------------*/
/*------------------------- CONVERSIONS -------------------------------------*/
/*---------------------------------------------------------------------------*/
int CUSPARSE_SNNZ(ptr_t *handle, 
                  int *dirA, 
                  int *m, 
                  int *n, 
                  const ptr_t *descrA,
                  const ptr_t *A, 
                  int *lda, 
                  ptr_t *nnzPerRowCol, 
                  int *nnzHostPtr){
    return (int)cusparseSnnz((cusparseHandle_t)(*handle), 
                             (cusparseDirection_t)(*dirA), 
                             *m, 
                             *n, 
                             (const cusparseMatDescr_t)(*descrA),
                             (const float *)(*A), 
                             *lda, 
                             (int *)(*nnzPerRowCol), 
                             nnzHostPtr);
}

int CUSPARSE_DNNZ(ptr_t *handle, 
                  int *dirA, 
                  int *m, 
                  int *n, 
                  const ptr_t *descrA,
                  const ptr_t *A, 
                  int *lda, 
                  ptr_t *nnzPerRowCol, 
                  int *nnzHostPtr){
    return (int)cusparseDnnz((cusparseHandle_t)(*handle), 
                             (cusparseDirection_t)(*dirA), 
                             *m, 
                             *n, 
                             (const cusparseMatDescr_t)(*descrA),
                             (const double *)(*A), 
                             *lda, 
                             (int *)(*nnzPerRowCol), 
                             nnzHostPtr);
}

int CUSPARSE_CNNZ(ptr_t *handle, 
                  int *dirA, 
                  int *m, 
                  int *n, 
                  const ptr_t *descrA,
                  const ptr_t *A, 
                  int *lda, 
                  ptr_t *nnzPerRowCol, 
                  int *nnzHostPtr){
    return (int)cusparseCnnz((cusparseHandle_t)(*handle), 
                             (cusparseDirection_t)(*dirA), 
                             *m, 
                             *n, 
                             (const cusparseMatDescr_t)(*descrA),
                             (const cuComplex *)(*A), 
                             *lda, 
                             (int *)(*nnzPerRowCol), 
                             nnzHostPtr);
}

int CUSPARSE_ZNNZ(ptr_t *handle, 
                  int *dirA, 
                  int *m, 
                  int *n, 
                  const ptr_t *descrA,
                  const ptr_t *A, 
                  int *lda, 
                  ptr_t *nnzPerRowCol, 
                  int *nnzHostPtr){
    return (int)cusparseZnnz((cusparseHandle_t)(*handle), 
                             (cusparseDirection_t)(*dirA), 
                             *m, 
                             *n, 
                             (const cusparseMatDescr_t)(*descrA),
                             (const cuDoubleComplex *)(*A), 
                             *lda, 
                             (int *)(*nnzPerRowCol), 
                             nnzHostPtr);
}

int CUSPARSE_SDENSE2CSR(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerRow,                                                 
                        ptr_t *csrValA, 
                        ptr_t *csrRowPtrA, 
                        ptr_t *csrColIndA){
    return (int)cusparseSdense2csr((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const float *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerRow),                                                 
                                   (float *)(*csrValA), 
                                   (int *)(*csrRowPtrA), 
                                   (int *)(*csrColIndA));
}
 
int CUSPARSE_DDENSE2CSR(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerRow,                                                 
                        ptr_t *csrValA, 
                        ptr_t *csrRowPtrA, 
                        ptr_t *csrColIndA){
    return (int)cusparseDdense2csr((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const double *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerRow),                                                 
                                   (double *)(*csrValA), 
                                   (int *)(*csrRowPtrA), 
                                   (int *)(*csrColIndA));
}

int CUSPARSE_CDENSE2CSR(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerRow,                                                 
                        ptr_t *csrValA, 
                        ptr_t *csrRowPtrA, 
                        ptr_t *csrColIndA){
    return (int)cusparseCdense2csr((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const cuComplex *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerRow),                                                 
                                   (cuComplex *)(*csrValA), 
                                   (int *)(*csrRowPtrA), 
                                   (int *)(*csrColIndA));
}

int CUSPARSE_ZDENSE2CSR(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerRow,                                                 
                        ptr_t *csrValA, 
                        ptr_t *csrRowPtrA, 
                        ptr_t *csrColIndA){
    return (int)cusparseZdense2csr((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const cuDoubleComplex *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerRow),                                                 
                                   (cuDoubleComplex *)(*csrValA), 
                                   (int *)(*csrRowPtrA), 
                                   (int *)(*csrColIndA));
}


int CUSPARSE_SCSR2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *csrValA, 
                        const ptr_t *csrRowPtrA, 
                        const ptr_t *csrColIndA,
                        ptr_t *A, 
                        int *lda){
    return (int)cusparseScsr2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const float *)(*csrValA), 
                                   (const int *)(*csrRowPtrA), 
                                   (const int *)(*csrColIndA),
                                   (float *)(*A), 
                                   *lda);
}

int CUSPARSE_DCSR2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *csrValA, 
                        const ptr_t *csrRowPtrA, 
                        const ptr_t *csrColIndA,
                        ptr_t *A, 
                        int *lda){
    return (int)cusparseDcsr2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const double *)(*csrValA), 
                                   (const int *)(*csrRowPtrA), 
                                   (const int *)(*csrColIndA),
                                   (double *)(*A), 
                                   *lda);
}

int CUSPARSE_CCSR2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *csrValA, 
                        const ptr_t *csrRowPtrA, 
                        const ptr_t *csrColIndA,
                        ptr_t *A, 
                        int *lda){
    return (int)cusparseCcsr2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const cuComplex *)(*csrValA), 
                                   (const int *)(*csrRowPtrA), 
                                   (const int *)(*csrColIndA),
                                   (cuComplex *)(*A), 
                                   *lda);
}

int CUSPARSE_ZCSR2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *csrValA, 
                        const ptr_t *csrRowPtrA, 
                        const ptr_t *csrColIndA,
                        ptr_t *A, 
                        int *lda){
    return (int)cusparseZcsr2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const cuDoubleComplex *)(*csrValA), 
                                   (const int *)(*csrRowPtrA), 
                                   (const int *)(*csrColIndA),
                                   (cuDoubleComplex *)(*A), 
                                   *lda);
}

int CUSPARSE_SDENSE2CSC(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerCol,                                                 
                        ptr_t *cscValA, 
                        ptr_t *cscRowIndA, 
                        ptr_t *cscColPtrA){
    return (int)cusparseSdense2csc((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const float *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerCol),                                                 
                                   (float *)(*cscValA), 
                                   (int *)(*cscRowIndA), 
                                   (int *)(*cscColPtrA));
}
 
int CUSPARSE_DDENSE2CSC(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerCol,                                                 
                        ptr_t *cscValA, 
                        ptr_t *cscRowIndA, 
                        ptr_t *cscColPtrA){
    return (int)cusparseDdense2csc((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const double *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerCol),                                                 
                                   (double *)(*cscValA), 
                                   (int *)(*cscRowIndA), 
                                   (int *)(*cscColPtrA));
}

int CUSPARSE_CDENSE2CSC(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerCol,                                                 
                        ptr_t *cscValA, 
                        ptr_t *cscRowIndA, 
                        ptr_t *cscColPtrA){
    return (int)cusparseCdense2csc((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const cuComplex *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerCol),                                                 
                                   (cuComplex *)(*cscValA), 
                                   (int *)(*cscRowIndA), 
                                   (int *)(*cscColPtrA));
}

int CUSPARSE_ZDENSE2CSC(ptr_t *handle,
                        int *m, 
                        int *n,  
                        const ptr_t *descrA,                            
                        const ptr_t *A, 
                        int *lda,
                        const ptr_t *nnzPerCol,                                                 
                        ptr_t *cscValA, 
                        ptr_t *cscRowIndA, 
                        ptr_t *cscColPtrA){
    return (int)cusparseZdense2csc((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n,  
                                   (const cusparseMatDescr_t)(*descrA),                            
                                   (const cuDoubleComplex *)(*A), 
                                   *lda,
                                   (const int *)(*nnzPerCol),                                                 
                                   (cuDoubleComplex *)(*cscValA), 
                                   (int *)(*cscRowIndA), 
                                   (int *)(*cscColPtrA));
}

int CUSPARSE_SCSC2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *cscValA, 
                        const ptr_t *cscRowIndA, 
                        const ptr_t *cscColPtrA,
                        ptr_t *A, 
                        int *lda){
    return (int)cusparseScsc2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const float *)(*cscValA), 
                                   (const int *)(*cscRowIndA), 
                                   (const int *)(*cscColPtrA),
                                   (float *)(*A), 
                                   *lda);
}
    
int CUSPARSE_DCSC2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *cscValA, 
                        const ptr_t *cscRowIndA, 
                        const ptr_t *cscColPtrA,
                        ptr_t *A, 
                        int *lda){
       return (int)cusparseDcsc2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const double *)(*cscValA), 
                                   (const int *)(*cscRowIndA), 
                                   (const int *)(*cscColPtrA),
                                   (double *)(*A), 
                                   *lda);
}

int CUSPARSE_CCSC2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *cscValA, 
                        const ptr_t *cscRowIndA, 
                        const ptr_t *cscColPtrA,
                        ptr_t *A, 
                        int *lda){
    return (int)cusparseCcsc2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const cuComplex *)(*cscValA), 
                                   (const int *)(*cscRowIndA), 
                                   (const int *)(*cscColPtrA),
                                   (cuComplex *)(*A), 
                                   *lda);
}

int CUSPARSE_ZCSC2DENSE(ptr_t *handle,
                        int *m, 
                        int *n, 
                        const ptr_t *descrA,  
                        const ptr_t *cscValA, 
                        const ptr_t *cscRowIndA, 
                        const ptr_t *cscColPtrA,
                        ptr_t *A, 
                        int *lda){
    return (int)cusparseZcsc2dense((cusparseHandle_t)(*handle),
                                   *m, 
                                   *n, 
                                   (const cusparseMatDescr_t)(*descrA),  
                                   (const cuDoubleComplex *)(*cscValA), 
                                   (const int *)(*cscRowIndA), 
                                   (const int *)(*cscColPtrA),
                                   (cuDoubleComplex *)(*A), 
                                   *lda);
}

int CUSPARSE_XCOO2CSR(ptr_t *handle,
                      const ptr_t *cooRowInd, 
                      int *nnz, 
                      int *m, 
                      ptr_t *csrRowPtr, 
                      int *idxBase){
    return (int)cusparseXcoo2csr((cusparseHandle_t)(*handle),
                                 (const int *)(*cooRowInd), 
                                 *nnz, 
                                 *m, 
                                 (int *)(*csrRowPtr), 
                                 (cusparseIndexBase_t)(*idxBase));
}
    
int CUSPARSE_XCSR2COO(ptr_t *handle,
                      const ptr_t *csrRowPtr, 
                      int *nnz, 
                      int *m, 
                      ptr_t *cooRowInd, 
                      int *idxBase){
    return (int)cusparseXcsr2coo((cusparseHandle_t)(*handle),
                                 (const int *)(*csrRowPtr), 
                                 *nnz, 
                                 *m, 
                                 (int *)(*cooRowInd), 
                                 (cusparseIndexBase_t)(*idxBase));
}     

int CUSPARSE_SCSR2CSC(ptr_t *handle,
                      int *m, 
                      int *n, 
                      const ptr_t *csrVal, 
                      const ptr_t *csrRowPtr, 
                      const ptr_t *csrColInd, 
                      ptr_t *cscVal, 
                      ptr_t *cscRowInd, 
                      ptr_t *cscColPtr, 
                      int *copyValues, 
                      int *idxBase){
    return (int)cusparseScsr2csc((cusparseHandle_t)(*handle),
                                 *m, 
                                 *n, 
                                 (const float *)(*csrVal), 
                                 (const int *)(*csrRowPtr), 
                                 (const int *)(*csrColInd), 
                                 (float *)(*cscVal), 
                                 (int *)(*cscRowInd), 
                                 (int *)(*cscColPtr), 
                                 *copyValues, 
                                 (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_DCSR2CSC(ptr_t *handle,
                      int *m, 
                      int *n, 
                      const ptr_t *csrVal, 
                      const ptr_t *csrRowPtr, 
                      const ptr_t *csrColInd, 
                      ptr_t *cscVal, 
                      ptr_t *cscRowInd, 
                      ptr_t *cscColPtr, 
                      int *copyValues, 
                      int *idxBase){
    return (int)cusparseDcsr2csc((cusparseHandle_t)(*handle),
                                 *m, 
                                 *n, 
                                 (const double *)(*csrVal), 
                                 (const int *)(*csrRowPtr), 
                                 (const int *)(*csrColInd), 
                                 (double *)(*cscVal), 
                                 (int *)(*cscRowInd), 
                                 (int *)(*cscColPtr), 
                                 *copyValues, 
                                 (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_CCSR2CSC(ptr_t *handle,
                      int *m, 
                      int *n, 
                      const ptr_t *csrVal, 
                      const ptr_t *csrRowPtr, 
                      const ptr_t *csrColInd, 
                      ptr_t *cscVal, 
                      ptr_t *cscRowInd, 
                      ptr_t *cscColPtr, 
                      int *copyValues, 
                      int *idxBase){
    return (int)cusparseCcsr2csc((cusparseHandle_t)(*handle),
                                 *m, 
                                 *n, 
                                 (const cuComplex *)(*csrVal), 
                                 (const int *)(*csrRowPtr), 
                                 (const int *)(*csrColInd), 
                                 (cuComplex *)(*cscVal), 
                                 (int *)(*cscRowInd), 
                                 (int *)(*cscColPtr), 
                                 *copyValues, 
                                 (cusparseIndexBase_t)(*idxBase));
}

int CUSPARSE_ZCSR2CSC(ptr_t *handle,
                      int *m, 
                      int *n, 
                      const ptr_t *csrVal, 
                      const ptr_t *csrRowPtr, 
                      const ptr_t *csrColInd, 
                      ptr_t *cscVal, 
                      ptr_t *cscRowInd, 
                      ptr_t *cscColPtr, 
                      int *copyValues, 
                      int *idxBase){
    return (int)cusparseZcsr2csc((cusparseHandle_t)(*handle),
                                 *m, 
                                 *n, 
                                 (const cuDoubleComplex *)(*csrVal), 
                                 (const int *)(*csrRowPtr), 
                                 (const int *)(*csrColInd), 
                                 (cuDoubleComplex *)(*cscVal), 
                                 (int *)(*cscRowInd), 
                                 (int *)(*cscColPtr), 
                                 *copyValues, 
                                 (cusparseIndexBase_t)(*idxBase));
}

                                                     


