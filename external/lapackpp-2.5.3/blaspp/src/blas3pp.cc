
#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "lafnames.h"
#include LA_EXCEPTION_H
#include LA_VECTOR_DOUBLE_H
#include LA_SYMM_MAT_DOUBLE_H
#include LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H
#include LA_UPPER_TRIANG_MAT_DOUBLE_H
#include LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H
#include LA_LOWER_TRIANG_MAT_DOUBLE_H
#include LA_SPD_MAT_DOUBLE_H
#include LA_SYMM_BAND_MAT_DOUBLE_H
#include LA_TRIDIAG_MAT_DOUBLE_H
#include LA_BAND_MAT_DOUBLE_H
#include LA_SYMM_TRIDIAG_MAT_DOUBLE_H

#ifdef LA_COMPLEX_SUPPORT
#include "f2c.h"
#include "lapackc.h"
#include LA_VECTOR_COMPLEX_H
#include LA_GEN_MAT_COMPLEX_H
#endif

#include "blas3pp.h"
#include "blas3.h"
#include "blas1.h"
#include "blas1pp.h" // for Blas_Norm2
#include "laindex.h"


void Blas_Mat_Mat_Mult(const LaGenMatDouble &A, 
		       const LaGenMatDouble &B, LaGenMatDouble &C, 
		       bool transpose_A, bool transpose_B,
		       double alpha, double beta)
{
   char transa = transpose_A ? 'T' : 'N';
   char transb = transpose_B ? 'T' : 'N';
   // m, n, k have to be determined according to op(A), op(B)!
   integer m = transpose_A ? A.size(1) : A.size(0);
   integer k = transpose_A ? A.size(0) : A.size(1);
   integer n = transpose_B ? B.size(0) : B.size(1);
   integer lda = A.gdim(0), ldb = B.gdim(0), ldc = C.gdim(0);
   if (alpha != 0.0)
   {
      assert(m == C.size(0));
      assert(n == C.size(1));
      assert(k == (transpose_B ? B.size(1) : B.size(0)) );
   }

   F77NAME(dgemm)(&transa, &transb, &m, &n, &k,
		  &alpha, &A(0,0), &lda, &B(0,0), &ldb,
		  &beta, &C(0,0), &ldc);
}

void Blas_Mat_Mat_Mult(const LaGenMatDouble &A, 
            const LaGenMatDouble &B, LaGenMatDouble &C, 
            double alpha , double beta )
{
   Blas_Mat_Mat_Mult(A, B, C, false, false, alpha, beta);
}

void Blas_Mat_Trans_Mat_Mult(const LaGenMatDouble &A, 
            const LaGenMatDouble &B, LaGenMatDouble &C, 
            double alpha, double beta )
{
   Blas_Mat_Mat_Mult(A, B, C, true, false, alpha, beta);
}

void Blas_Mat_Mat_Trans_Mult(const LaGenMatDouble &A, 
            const LaGenMatDouble &B, LaGenMatDouble &C, 
            double alpha,  double beta )
{
   Blas_Mat_Mat_Mult(A, B, C, false, true, alpha, beta);
}



double my_Dot_Prod(const LaGenMatDouble &dx, const LaGenMatDouble &dy)
{
  assert(dx.size(0)*dx.size(1)==dy.size(0)*dy.size(1));
  integer n = dx.size(0)*dx.size(1);
  integer incx = dx.inc(0)*dx.inc(1);
  integer incy = dy.inc(0)*dy.inc(1);

  return F77NAME(ddot)(&n, &dx(0,0), &incx, &dy(0,0), &incy);
}

void Blas_Mat_Mat_Mult(const LaGenMatDouble &A, 
		       const LaGenMatDouble &B, LaVectorDouble &C)
{
  // calculate only the diagonal of A times B
  int msize = std::min(A.size(0), B.size(1));
  assert(A.size(1) == B.size(0));
  assert(C.size() >= msize);
  for (int i=0; i < msize; i++)
    C(i) = my_Dot_Prod( A.row(i), B.col(i) );
}

void Blas_Mat_Trans_Mat_Mult(const LaGenMatDouble &A, 
			     const LaGenMatDouble &B, LaVectorDouble &C)
{
  // calculate only the diagonal of A times B
  int msize = std::min(A.size(0), B.size(1));
  assert(A.size(0) == B.size(0));
  assert(C.size() >= msize);
  for (int i=0; i < msize; i++)
    C(i) = my_Dot_Prod( A.col(i), B.col(i) );
}

void Blas_Mat_Mat_Trans_Mult(const LaGenMatDouble &A, 
			     const LaGenMatDouble &B, LaVectorDouble &C)
{
  // calculate only the diagonal of A times B
  int msize = std::min(A.size(0), B.size(1));
  assert(A.size(1) == B.size(1));
  assert(C.size() >= msize);
  for (int i=0; i < msize; i++)
    C(i) = my_Dot_Prod( A.row(i), B.row(i) );
}






#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_

void Blas_Mat_Mat_Solve(LaUnitLowerTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha)
{
        char side = 'L', uplo = 'L', transa = 'N', diag = 'U';
        integer m = B.size(0), n = B.size(1), 
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha, 
                &A(0,0), &lda, &B(0,0), &ldb);
}

#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
void Blas_Mat_Mat_Mult(LaUnitUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'U', transa = 'N', diag = 'U';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}

void Blas_Mat_Mat_Solve(LaUnitUpperTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'U', transa = 'N', diag = 'U';
        integer m = B.size(0), n = B.size(1), 
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha, 
                &A(0,0), &lda, &B(0,0), &ldb);
}

#endif

#ifdef _LA_LOWER_TRIANG_MAT_DOUBLE_H_
void Blas_Mat_Mat_Mult(LaLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'L', transa = 'N', diag = 'N';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}

void Blas_Mat_Mat_Solve(LaLowerTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'L', transa = 'N', diag = 'N';
        integer m = B.size(0), n = B.size(1), 
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha, 
                &A(0,0), &lda, &B(0,0), &ldb);
}
#endif


#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H_
void Blas_Mat_Mat_Mult(LaUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'U', transa = 'N', diag = 'N';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}

void Blas_Mat_Mat_Solve(LaUpperTriangMatDouble &A, 
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'U', transa = 'N', diag = 'N';
        integer m = B.size(0), n = B.size(1), 
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha, 
                &A(0,0), &lda, &B(0,0), &ldb);
}
#endif


#ifdef _LA_UNIT_LOWER_TRIANG_MAT_DOUBLE_H_
void Blas_Mat_Trans_Mat_Solve(LaUnitLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'L', transa = 'T', diag = 'U';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}
#endif

#ifdef _LA_UNIT_UPPER_TRIANG_MAT_DOUBLE_H_
void Blas_Mat_Trans_Mat_Solve(LaUnitUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'U', transa = 'T', diag = 'U';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}
#endif

#ifdef LA_LOWER_TRIANG_MAT_DOUBLE_H
void Blas_Mat_Mat_Mult(LaUnitLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'L', transa = 'N', diag = 'U';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrmm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}

void Blas_Mat_Trans_Mat_Solve(LaLowerTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'L', transa = 'T', diag = 'N';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}
#endif


#ifdef _LA_UPPER_TRIANG_MAT_DOUBLE_H 
void Blas_Mat_Trans_Mat_Solve(LaUpperTriangMatDouble &A,
            LaGenMatDouble &B, double alpha )
{
        char side = 'L', uplo = 'U', transa = 'T', diag = 'N';
        integer m = B.size(0), n = B.size(1),
                lda = A.gdim(0), ldb = B.gdim(0);

  F77NAME(dtrsm)(&side, &uplo, &transa, &diag, &m, &n, &alpha,
                &A(0,0), &lda, &B(0,0), &ldb);
}

#endif

#ifdef _LA_SYMM_MAT_DOUBLE_H_
void Blas_Mat_Mat_Mult(LaSymmMatDouble &A, LaGenMatDouble &B, 
		       LaGenMatDouble &C, 
		       double alpha , double beta,
		       bool b_left_side )
{
  char side = b_left_side ? 'L' : 'R';
  
  if (side=='L')
  {
    assert ( B.size(1)==C.size(0) && A.size(0)==B.size(0) && A.size(0)==C.size(1) );
  }
  else
  {
    assert ( B.size(0)==C.size(1) && A.size(0)==B.size(1) && A.size(0)==C.size(0) );
  }
  
  char uplo = 'L';
  integer m = C.size(0), n = C.size(1), lda = A.gdim(0),
          ldb = B.gdim(0), ldc = C.gdim(0);

  F77NAME(dsymm)(&side, &uplo, &m, &n, &alpha, &A(0,0), &lda, 
                &B(0,0), &ldb, &beta, &C(0,0), &ldc);
}

void Blas_R1_Update(LaSymmMatDouble &C, LaGenMatDouble &A,
		    double alpha , double beta, bool right_transposed )
{
  char trans = right_transposed ? 'N' : 'T';

  char uplo = 'L';
  integer n = C.size(0), k,
          lda = A.gdim(0), ldc = C.gdim(0);

  if (trans=='N')
  {
    k = A.size(1);
    assert ( A.size(0)==n );
  }
  else
  {
    k = A.size(0);
    assert ( A.size(1)==n );
  }

  F77NAME(dsyrk)(&uplo, &trans, &n, &k, &alpha, &A(0,0), &lda,
                &beta, &C(0,0), &ldc);
}

void Blas_R2_Update(LaSymmMatDouble &C, LaGenMatDouble &A,
            LaGenMatDouble &B, double alpha , double beta, bool right_transposed )
{
  char trans = right_transposed ? 'N' : 'T';
  char uplo = 'L';
  integer n = C.size(0), k, lda = A.gdim(0),
          ldb = B.gdim(0), ldc = C.gdim(0);

  if (trans=='N')
  {
    k = A.size(1);
    assert( A.size(0)==n && B.size(0)==n && B.size(1)==k);
  }
  else
  {
    k = A.size(0);
    assert( A.size(1)==n && B.size(1)==n && B.size(0)==k);
  }

  F77NAME(dsyr2k)(&uplo, &trans, &n, &k, &alpha, &A(0,0), &lda,
                &B(0,0), &ldb, &beta, &C(0,0), &ldc);
}

#endif


#ifdef LA_COMPLEX_SUPPORT
void Blas_Mat_Mat_Mult(const LaGenMatComplex &A, 
		       const LaGenMatComplex &B, LaGenMatComplex &C, 
		       bool hermit_A, bool hermit_B, 
		       LaComplex _alpha, LaComplex _beta)
{
   char transa = hermit_A ? 'C' : 'N';
   char transb = hermit_B ? 'C' : 'N';
   // m, n, k have to be determined according to op(A), op(B)!
   integer m = hermit_A ? A.size(1) : A.size(0);
   integer k = hermit_A ? A.size(0) : A.size(1);
   integer n = hermit_B ? B.size(0) : B.size(1);
   integer lda = A.gdim(0), ldb = B.gdim(0), ldc = C.gdim(0);
   doublecomplex alpha(_alpha);
   doublecomplex beta(_beta);

   // Check for correct matrix sizes
   if (alpha.r != 0.0 || alpha.i != 0.0)
   {
      assert(m == C.size(0));
      assert(n == C.size(1));
      assert(k == (hermit_B ? B.size(1) : B.size(0)) );
   }

   F77NAME(zgemm)(&transa, &transb, &m, &n, &k,
		  &alpha, &A(0,0), &lda, &B(0,0), &ldb,
		  &beta, &C(0,0), &ldc);
}

void Blas_Mat_Mat_Mult(const LaGenMatComplex &A, 
		       const LaGenMatComplex &B, LaGenMatComplex &C, 
		       LaComplex _alpha, LaComplex _beta)
{
   Blas_Mat_Mat_Mult(A, B, C, false, false, _alpha, _beta);
}

void Blas_Mat_Trans_Mat_Mult(const LaGenMatComplex &A, 
			     const LaGenMatComplex &B, LaGenMatComplex &C, 
			     LaComplex _alpha, LaComplex _beta)
{
   Blas_Mat_Mat_Mult(A, B, C, true, false, _alpha, _beta);
}

void Blas_Mat_Mat_Trans_Mult(const LaGenMatComplex &A, 
			     const LaGenMatComplex &B, LaGenMatComplex &C, 
			     LaComplex _alpha, LaComplex _beta)
{
   Blas_Mat_Mat_Mult(A, B, C, false, true, _alpha, _beta);
}
#endif // LA_COMPLEX_SUPPORT

// ////////////////////////////////////////////////////////////
// Scaling 

template<class matT, class vecT>
void mat_scale(matT& A, vecT& tmpvec,
	       typename matT::value_type scalar)
{
   int M=A.size(1);
   // max column-sum
   if (M==1)
   {
      // only one column
      tmpvec.ref(A);
      Blas_Scale(scalar, tmpvec);
   }
   else
   {
      for (int k=0; k<M; ++k)
      {
	 tmpvec.ref( A.col(k) );
	 Blas_Scale(scalar, tmpvec);
      }
   }
}

void Blas_Scale(double da, LaGenMatDouble &A)
{
   LaVectorDouble T;
   mat_scale(A, T, da);
}
// void Blas_Scale(float da, LaGenMatFloat &A)
// {
//    LaVectorFloat T;
//    mat_scale(A, T, da);
// }
// void Blas_Scale(int da, LaGenMatInt &A)
// {
//    LaVectorInt T;
//    mat_scale(A, T, da);
// }
// void Blas_Scale(long int da, LaGenMatLongInt &A)
// {
//    LaVectorLongInt T;
//    mat_scale(A, T, da);
// }

#ifdef LA_COMPLEX_SUPPORT
void Blas_Scale(COMPLEX s, LaGenMatComplex &A)
{
   LaVectorComplex T;
   mat_scale(A, T, s);
}
#endif

// ////////////////////////////////////////////////////////////
// Matrix norms

template<class matT, class vecT>
double max_col_sum(const matT& A, vecT& tmpvec)
{
   int M=A.size(1);
   // max column-sum
   if (M==1)
   {
      // only one column
      tmpvec.ref(A);
      return Blas_Norm1( tmpvec );
   }
   else
   {
      LaVectorDouble R(M);
      for (int k=0; k<M; ++k)
      {
	 tmpvec.ref( A( LaIndex(), LaIndex(k) ) );
	 R(k) = Blas_Norm1( tmpvec );
      }
      return Blas_Norm_Inf(R);
   }
}

double Blas_Norm1(const LaGenMatDouble &A)
{
   LaVectorDouble T;
   return max_col_sum(A, T);
}

// ////////////////////////////////////////
// max row norm

template<class matT, class vecT>
double max_row_sum(const matT& A, vecT& tmpvec)
{
   int M=A.size(0);
   // max row-sum
   if (M==1)
   {
      // only one row
      tmpvec.ref(A);
      return Blas_Norm1( tmpvec );
   }
   else
   {
      LaVectorDouble R(M);
      for (int k=0; k<M; ++k)
      {
	 tmpvec.ref( A( LaIndex(k), LaIndex() ) );
	 R(k) = Blas_Norm1( tmpvec );
      }
      return Blas_Norm_Inf(R);
   }
}

double Blas_Norm_Inf(const LaGenMatDouble &A)
{
   LaVectorDouble T;
   return max_row_sum(A, T);
}

// ////////////////////////////////////////
// max frobenius norm

template<class matT, class vecT>
double max_fro_sum(const matT& A, vecT& tmpvec)
{
   int M=A.size(1);
   // calculate this column-wise
   if (M==1)
   {
      // only one column
      tmpvec.ref(A);
      return Blas_Norm2( tmpvec );
   }
   else
   {
      LaVectorDouble R(M);
      for (int k=0; k<M; ++k)
      {
	 tmpvec.ref( A( LaIndex(), LaIndex(k) ) );
	 R(k) = Blas_Norm2( tmpvec );
      }
      return Blas_Norm2(R);
   }
}

double Blas_NormF(const LaGenMatDouble &A)
{
   LaVectorDouble T;
   return max_fro_sum(A, T);
}

/** DEPRECATED, use Blas_Norm_Inf instead. */
double Norm_Inf(const LaGenMatDouble &A) 
{ return Blas_Norm_Inf(A); }


#ifdef LA_COMPLEX_SUPPORT

double Blas_Norm1(const LaGenMatComplex &A)
{
   LaVectorComplex T;
   return max_col_sum(A, T);
}
double Blas_Norm_Inf(const LaGenMatComplex &A)
{
   LaVectorComplex T;
   return max_row_sum(A, T);
}
double Blas_NormF(const LaGenMatComplex &A)
{
   LaVectorComplex T;
   return max_fro_sum(A, T);
}
/** DEPRECATED, use Blas_Norm_Inf instead. */
double Norm_Inf(const LaGenMatComplex &A) 
{ return Blas_Norm_Inf(A); }

#endif // LA_COMPLEX_SUPPORT


// ////////////////////////////////////////////////////////////

#ifdef __GNUC__
#  define LA_STD_ABS std::abs
#else
#  define LA_STD_ABS fabs
#endif

double Norm_Inf(const LaBandMatDouble &A)
{
    // integer kl = A.subdiags(), ku = A.superdiags(); 
    integer N=A.size(1);
    integer M=N;

    // slow version

    LaVectorDouble R(M);
    integer i;
    integer j;
    for (i=0; i<M; i++)
    {
        R(i) = 0.0;
        for (j=0; j<N; j++)
            R(i) += 
		LA_STD_ABS (A(i,j));
    }

    double max = R(0);

    // report back largest row sum
    for (i=1; i<M; i++)
        if (R(i) > max) max=R(i);

    return max;
}


double Norm_Inf(const LaSymmMatDouble &S)
{
    integer N = S.size(0); // square matrix

    // slow version

    LaVectorDouble R(N);
    integer i; 
    integer j;      

    for (i=0; i<N; i++)
    {
        R(i) = 0.0;
        for (j=0; j<N; j++)
            R(i) += 
		LA_STD_ABS (S(i,j));
    }
     
    double max = R(0);

    // report back largest row sum
    for (i=1; i<N; i++)
        if (R(i) > max) max=R(i);

    return max;
}


double Norm_Inf(const LaSpdMatDouble &S)
{
    integer N = S.size(0); //SPD matrices are square

    // slow version

    LaVectorDouble R(N);
    integer i; 
    integer j;      

    for (i=0; i<N; i++)
    {
        R(i) = 0.0;
        for (j=0; j<N; j++)
            R(i) += 
		LA_STD_ABS (S(i,j));
    }
     
    double max = R(0);

    // report back largest row sum
    for (i=1; i<N; i++)
        if (R(i) > max) max=R(i);

    return max;
}


double Norm_Inf(const LaSymmTridiagMatDouble &S)
{
    integer N = S.size();   // S is square
    LaVectorDouble R(N);

    R(0) = LA_STD_ABS(S(0,0)) + LA_STD_ABS(S(0,1));

    for (integer i=1; i<N-1; i++)
    {
        R(i) = LA_STD_ABS(S(i,i-1)) + LA_STD_ABS(S(i,i)) + LA_STD_ABS(S(i,i+1));
    }

    R(N-1) = LA_STD_ABS(S(N-1,N-2)) + LA_STD_ABS(S(N-1,N-1));

    return Norm_Inf(R);
}


double Norm_Inf(const LaTridiagMatDouble &T)
{
    integer N = T.size();   // T is square
    LaVectorDouble R(N);

    R(0) = LA_STD_ABS(T(0,0)) + LA_STD_ABS(T(0,1));

    for (int i=1; i<N-1; i++)
    {
        R(i) = LA_STD_ABS(T(i,i-1)) + LA_STD_ABS(T(i,i)) + LA_STD_ABS(T(i,i+1));
    }

    R(N-1) = LA_STD_ABS(T(N-1,N-2)) + LA_STD_ABS(T(N-1,N-1));

    return Norm_Inf(R);
}

// #undef LA_STD_ABS
