//
//              LAPACK++ 1.1 Linear Algebra Package 1.1
//               University of Tennessee, Knoxvilee, TN.
//            Oak Ridge National Laboratory, Oak Ridge, TN.
//        Authors: J. J. Dongarra, E. Greaser, R. Pozo, D. Walker
//                 (C) 1992-1996 All Rights Reserved
//
//                             NOTICE
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose and without fee is hereby granted
// provided that the above copyright notice appear in all copies and
// that both the copyright notice and this permission notice appear in
// supporting documentation.
//
// Neither the Institutions (University of Tennessee, and Oak Ridge National
// Laboratory) nor the Authors make any representations about the suitability 
// of this software for any purpose.  This software is provided ``as is'' 
// without express or implied warranty.
//
// LAPACK++ was funded in part by the U.S. Department of Energy, the
// National Science Foundation and the State of Tennessee.

#if 1 
#include "lapackpp.h"
#endif

#if 0
#include "lafnames.h"       /* macros for LAPACK++ filenames */
#include LA_GEN_MAT_DOUBLE_H
#include LA_VECTOR_DOUBLE_H 
#include "blaspp.h"
#include LA_SOLVE_DOUBLE_H
#include LA_GENERATE_MAT_DOUBLE_H
#include LA_EXCEPTION_H
#include LA_UTIL_H
#include "lasvd.h"
#endif

double residual(const LaGenMatDouble &A, const LaVectorDouble &x, 
    const LaVectorDouble& b)
{
    int M = A.size(0);
    int N = A.size(1);


    std::cout << "\tNorm_Inf(A*x-b)" << Norm_Inf(A*x-b) << std::endl;
    std::cout << "\tNorm_Inf(A) " << Norm_Inf(A) << std::endl;
    std::cout << "\tNorm_Inf(x) " << Norm_Inf(x) << std::endl;
    std::cout << "\tMacheps :" << Mach_eps_double() << std::endl;

    if (M>N)
    {
        LaVectorDouble Axb = A*x-b;
        LaVectorDouble R(N);

        Blas_Mat_Trans_Vec_Mult(A, Axb, R);
        return Norm_Inf(R) / 
            (Norm_Inf(A)* Norm_Inf(x) * N * Mach_eps_double());

    }
    else
    {
        return Norm_Inf(A*x-b ) /
                ( Norm_Inf(A)* Norm_Inf(x) * N * Mach_eps_double());
    }
}


bool testQRsolve(int M, int N)
{
#ifndef HPPA
   const char fname[] = "TestGenLinearSolve(LaGenMat, x, b) ";
#else
   char *fname = NULL;
#endif
   bool error = false;

   double aa[] = { 1, 2, 3, 4, 5, 6 };
   double bb[] = { 7, 8, 9 };

   {
      LaGenMatDouble A2tmp(aa, 3, 2, false), A2(A2tmp);
      LaGenMatDouble B2(bb, 3, 1, true);
      LaGenMatDouble X2(2, 1);

      std::cout << fname << ": LaQRLinearSolve: Matrix A=" << A2
		<< "  Right hand side B=" << B2;
      LaQRLinearSolveIP(A2, X2, B2);
      std::cout << "  Solution X=" << X2;
      //std::cout << "Residual " << residual(A2tmp, X2, B2) << std::endl;
      double cc2[] = { -1, 2 };
      double diff = Norm_Inf(X2 - LaGenMatDouble(cc2, 2, 1));
      std::cout << "Diff to known solution: " << diff << std::endl
		<< std::endl;
      if (diff > 1e-10)
	 error = true;
   }

   {
      LaGenMatDouble A1(aa, 2, 3, true);
      LaGenMatDouble B1(bb, 2, 1, true);
      LaGenMatDouble X1(3, 1);

      std::cout << fname << ": LaQRLinearSolve: Matrix A=" << A1
		<< "  Right hand side B=" << B1;
      LaQRLinearSolveIP(A1, X1, B1);
      std::cout << "  Solution X=" << X1;
      double cc1[] = { -3.0556, 0.1111, 3.2778 };
      double diff = Norm_Inf(X1 - LaGenMatDouble(cc1, 3, 1));
      std::cout << "Diff to known solution: " << diff << std::endl
		<< std::endl;
      if (diff > 1e-4)
	 error = true;
   }

   {
      LaGenMatDouble A3(aa, 2, 2, true);
      LaGenMatDouble B3(bb, 2, 1, true);
      LaGenMatDouble X3(2, 1);

      std::cout << fname << ": LaQRLinearSolve: Matrix A=" << A3
		<< "  Right hand side B=" << B3;
      LaQRLinearSolveIP(A3, X3, B3);
      std::cout << "  Solution X=" << X3;
      double cc3[] = { -6, 6.5 };
      double diff = Norm_Inf(X3 - LaGenMatDouble(cc3, 2, 1));
      std::cout << "Diff to known solution: " << diff << std::endl
		<< std::endl;
      if (diff > 1e-10)
	 error = true;
   }

   return error;
}


int TestGenLinearSolve(int M,int N)
{
    LaGenMatDouble A(M,N);
    LaVectorDouble x(N), b(M);
    bool error = false;

#ifndef HPPA
    const char fname[] = "TestGenLinearSolve(LaGenMat, x, b) ";
#else
    char *fname = NULL;
#endif

    //char e = 'e';
    double norm;
    double res;

#ifdef __x86_64
    la::rand(A); // LaGenerateMatDouble doesn't work on amd64
#else
    LaGenerateMatDouble(A);
#endif
    std::cout << "Generated matrix A=" << std::endl
	      << A << std::endl;

    // save a snapshot of what A looked like before the solution
    LaGenMatDouble old_A = A;



    b = 1.1;

    std::cerr << fname << ": testing LaLinearSolve(Gen,...) M= "<< M
        << " N = " << N << std::endl;

    LaLinearSolve(A, x, b);

    if ( (norm = Norm_Inf( old_A - A)) >  0.0)  // this is an exact test, not
                                         // necessary to worry about
                                         // round-off issues.  We
                                         // are testing to see A was
                                         // overwritten.
    {
        std::cerr << fname << ": overwrote 1st arg.\n";
        std::cerr << "       error norm: " << norm << std::endl;
        error = true; // exit(1);
    }

    res = residual(A,x,b);
    if (res > 1)
    {
        std::cerr << fname << "resdiual " << res << " is to too high.\n";
        error = true; // exit(1);
    }
    else
	std::cerr << fname << ": LaLinearSolve() success.\n\n";


    // now try the in-place solver


    std::cerr << fname << ": testing LaLinearSolveIP(Gen,...) \n";
    LaLinearSolveIP(A, x, b);


    res = residual(old_A, x, b);

    if (res > 1)
    {
        std::cerr << fname << "resdiual " << res << " is to too high.\n";
        error = true; // exit(1);
    }
    else
	std::cerr << fname << ": LaLinearSolveIP() success.\n\n";

    std::cout << fname << ": Matrix A=" << A
	      << std::endl;
    LaVectorDouble S(std::min(M,N));
    LaGenMatDouble U(M,M), VT(N,N);
//     S = 0;
//     U = 0;
//     VT = 0;
    LaSVD_IP(A, S, U, VT);

    std::cout << fname << ": Matrix A=" << A
	      << "  Singular values sigma = " << S
	      << "  Left S.vect. U = " << U
	      << "  Right S.vect. VT = " << VT
	      << std::endl;

    error = error || testQRsolve(M, N);

    if (error)
        exit(1);
    return 0;
}

int main(int argc, char **argv)
{

    std::cout.precision(4);
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
    LaException::enablePrint(true);

    if (argc < 2)
    {
        std::cerr << "Usage " << argv[0] << " M [ N ] " << std::endl;
        exit(1);
    }
    int M = atoi(argv[1]);
    int N;
    if (argc < 3)
	N = M;
    else 
	N = atoi(argv[2]);

    std::cout << "Testing " << M << " x " << N << " system." << std::endl;

    TestGenLinearSolve(M,N);

    return 0;
}

