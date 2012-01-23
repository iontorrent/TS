//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

//      Complex (complex precision) Lapack routines

#ifndef _LAPACKC_H_
#define _LAPACKC_H_


#ifndef _ARCH_H_
#include "arch.h"
#endif

#include "lacomplex.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

// *************************** Utility Routines **********************

    float F77NAME(slamch)(char *t);

    doublecomplex F77NAME(zlamch)(char *t);


  //void F77NAME(zswap)(integer *n, doublecomplex *x, integer *incx, doublecomplex *y, integer *incy);

    void F77NAME(zgesv)(integer *n, integer *k, doublecomplex *A, integer *lda, integer *ipiv,
            doublecomplex *X, integer *ldx, integer *info);

    void F77NAME(zposv)(char *uplo, integer *m, integer *k , doublecomplex *A, integer *lda,
        doublecomplex *X, integer *ldx, integer *info);

    void F77NAME(zgels)(char *trans, integer *m, integer *n, integer *nrhs, doublecomplex *A,
        integer *lda, doublecomplex *B, integer *ldb, doublecomplex *work, integer *lwork, integer *info);

    void F77NAME(ztimmg)(integer *iflag, integer *m, integer *n, doublecomplex *A, integer *lda,
                integer *kl, integer *ku);

    void F77NAME(zlaswp)(integer *n, doublecomplex *A, integer *lda, integer *k1, integer *k2,
                integer *ipiv, integer *incx);

    doublecomplex F77NAME(zopla)(char *subname, integer *m, integer *n, integer *kl, integer *ku,
            integer *nb);

// ******************* LU Factorization Routines **********************

    void F77NAME(zgetrf)(integer *m, integer *n, doublecomplex *A, integer *lda, integer *ipiv,
                integer *info);

    void F77NAME(zgetri)(integer *n, doublecomplex *A, integer *lda, integer *ipiv, 
                doublecomplex *work, integer *lwork, integer *info);

    void F77NAME(zgetf2)(integer *m, integer *n, doublecomplex *A, integer *lda, integer *ipiv,
                integer *info);

    void F77NAME(zgbtrf)(integer *m, integer *n, integer *KL, integer *KU, doublecomplex *BM,
                integer *LDBM, integer *ipiv, integer *info);

    void F77NAME(zgttrf)(integer *N, doublecomplex *DL, doublecomplex *D, doublecomplex *DU,
                doublecomplex *DU2, integer *ipiv, integer *info);

    void F77NAME(zpotrf)(char *UPLO, integer *N, doublecomplex *SM, integer *LDSM,
                integer *info);

    void F77NAME(zsytrf)(char *UPLO, integer *N, doublecomplex *SM, integer *LDSM,
                integer *ipiv, doublecomplex *WORK, integer *LWORK, integer *info);

    void F77NAME(zpbtrf)(char *UPLO, integer *N, integer *KD, doublecomplex *SBM,
                integer *LDSM, integer *info);

    void F77NAME(zpttrf)(integer *N, doublecomplex *D, doublecomplex *E, integer *info);

// ********************* LU Solve Routines ***************************

    void F77NAME(zgetrs)(char *trans, integer *N, integer *nrhs, doublecomplex *A, integer *lda, 
            integer * ipiv, doublecomplex *b, integer *ldb, integer *info);

    void F77NAME(zgbtrs)(char *trans, integer *N, integer *kl, integer *ku, integer *nrhs,
            doublecomplex *AB, integer *ldab, integer *ipiv, doublecomplex *b, integer *ldb, integer *info);

    void F77NAME(zsytrs)(char *uplo, integer *N, integer *nrhs, doublecomplex *A, integer *lda, 
            integer *ipiv, doublecomplex *b, integer *ldb, integer *info);

    void F77NAME(zgttrs)(char *trans, integer *N, integer *nrhs, doublecomplex *DL, 
                doublecomplex *D, doublecomplex *DU, doublecomplex *DU2, integer *ipiv, doublecomplex *b, 
                integer *ldb, integer *info);

    void F77NAME(zpotrs)(char *UPLO, integer *N, integer *nrhs, doublecomplex *A, integer *LDA,
                doublecomplex *b, integer *ldb, integer *info);

    void F77NAME(zpttrs)(integer *N, integer *nrhs, doublecomplex *D, doublecomplex *E, 
                doublecomplex *b, integer *ldb, integer *info);

    void F77NAME(zpbtrs)(char *UPLO, integer *N, integer *KD, integer *nrhs, doublecomplex *AB,
                integer *LDAB, doublecomplex *b, integer *ldb, integer *info);

  // ******************** QR factorizations

  void F77NAME(zgeqrf)(integer *m, integer *n, doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *work, integer *lwork, integer *info);
  void F77NAME(zungqr)(integer *m, integer *n, integer *k, doublecomplex *a, integer *lda, const doublecomplex *tau, doublecomplex *work, integer *lwork, integer *info);
  void F77NAME(zunmqr)(char *side, char *trans, integer *m, integer *n, integer *k, const doublecomplex *a, integer *lda, const doublecomplex *tau, doublecomplex *c, integer *ldc, doublecomplex *work, integer *lwork, integer *info);

  void F77NAME(dgeqrf)(integer *m, integer *n, double *a, integer *lda, double *tau, double *work, integer *lwork, integer *info); 
  void F77NAME(dorgqr)(integer *m, integer *n, integer *k, double *a, integer *lda, const double *tau, double *work, integer *lwork, integer *info); 
  void F77NAME(dormqr)(char *side, char *trans, integer *m, integer *n, integer *k, const double *a, integer *lda, const double *tau, double *c, integer *ldc, double *work, integer *lwork, integer *info); 

// ********************* Eigenvalue/Singular Value Decomposition Drivers

  void F77NAME(zgeev)(char *jobvl, char *jobvr, integer *n, doublecomplex  *a,
		      integer *lda, doublecomplex *w, 
		      doublecomplex *vl, integer *ldvl, 
		      doublecomplex *vr, integer *ldvr, 
		      doublecomplex *work, integer *lwork, double *rwork,
		      integer *info);

  void F77NAME(zheevd)(char *jobz, char *uplo, integer *n, doublecomplex *a, integer *lda, double *w, doublecomplex *work, integer *lwork, double *rwork, integer *lrwork, integer *info);
  void F77NAME(zheevr)(char *jobz, char *range, char *uplo, integer *n, doublecomplex *a, integer *lda, double *vl, double *vu, integer *il, integer *iu, double *abstol, integer *m, double *w, doublecomplex *z, integer *ldz, integer *isuppz, integer *info);

  void F77NAME(zgesvd)(char *jobu, char *jobvt, integer *m, integer *n, doublecomplex *a, integer *lda, double *sing, doublecomplex *u, integer *ldu, doublecomplex *vt, integer *ldvt, doublecomplex *work, integer *lwork, double *rwork, integer *info);
  void F77NAME(zgesdd)(char *jobz, integer *m, integer *n, doublecomplex *a, integer *lda, double *s, doublecomplex *u, integer *ldu, doublecomplex *vt, integer *ldvt, doublecomplex *work, integer *lwork, double *rwork, integer *iwork, integer *info);


// *******************************

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif 
