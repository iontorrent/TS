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


#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

// This file is no longer building the DLL
#undef BUILDING_LAPACK_DLL

#include <iostream>
#include "lafnames.h"
#include LA_VECTOR_DOUBLE_H
#include LA_VECTOR_COMPLEX_H

#include "blaspp.h"

int main(int argc, char *argv[])
{
    bool output = true;

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " M N \n";
        exit(1);
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    if (argc > 3)
	if (std::string(argv[3])=="q")
	    output = false;


//  test Blas_Sum()

    LaVectorDouble Sum(M);
    Sum = N;
    double ans = Blas_Norm1(Sum);
    if (output) {
    fprintf(stdout,"\nBlas_Sum() A:%dx1, value:%d\n",M,N);
    std::cout << "output:\n" << ans << std::endl;
    }

//  test Blas_Add_Mult()

    LaVectorDouble X(M);
    LaVectorDouble Y(M);
    X = N;
    Y = N;
    double scalar = M;
    Blas_Add_Mult(Y, scalar,X);
    if (output) {
    fprintf(stdout,"\nBlas_Add_Mult() alpha:%d, X:%dx1, Y:%dx1\n",\
                    M,N,N); 
    std::cout << "output:\n" << Y << std::endl;
    }

//  test Blas_Copy()

    X = M*N;
    Blas_Copy(Y, X);
    if (output) {
    std::cout <<"\nBlas_Copy(LaVectorDouble):\n" << "X:\n" << X << "\nY:\n" << Y << std::endl; 
    }

//  test Blas_Dot_Prod()

    X = M;
    Y = N;
    double cans = Blas_Dot_Prod(X,Y);
    if (output) {
    fprintf(stdout,"\nBlas_Dot_Prod() X = %d, Y = %d\n",M,N); 
    fprintf(stdout,"  X is %dx1, Y is %dx1\n",M,M); 
    std::cout << "\nAns:\n" << cans << std::endl;
    }
//  test Blas_Norm2()

    ans = Blas_Norm2(X);
    if (output) {
    fprintf(stdout,"\nBlas_Norm2() X = %d\n",M); 
    fprintf(stdout,"  X is %dx1\n",M); 
    std::cout << "\nAns:\n" << ans << std::endl;
    }

// see note in blas1++.cc
//#if 0
//  test Blas_Scale()

    double scale = 5.0;
    X = 1.1;
    if (output) {
    fprintf(stdout,"\nBlas_Scale() scale = 5.0, X = 1.1\n");
    }
    Blas_Scale(scale,X);
    if (output) {
    std::cout <<"X:\n"<< X << std::endl;
    }
// #endif

//  test Blas_Swap()

    LaVectorDouble A(5);
    LaVectorDouble B(5);
    A = 1.1;
    B = 2.0;
    if (output) {
    fprintf(stdout,"\nBlas_Swap() A = 1.1, B = 2.0\n");
    }
    Blas_Swap(A,B);
    if (output) {
    std::cout <<"A:\n"<< A << "\nB:\n" << B << std::endl;
    }


//  test Blas_Index_Max()


    int index;
    X = 8.0;
    X (M/2) = 64.0;
    if (output) {
    fprintf(stdout,"\nBlas_Index_Max() X = 8.0\n");
    }
    index = Blas_Index_Max(X);
    if (output) {
    std::cout <<"index:\n"<< index << std::endl;
    }

    // ////////////////////////////////////////////////////////////

    // and now the same for complex

    // ////////////////////////////////////////////////////////////
    {
	
//  test Blas_Sum()

    LaVectorComplex Sum(M);
    Sum = N;
    LaComplex ans = Blas_Norm1(Sum);
    if (output) {
    fprintf(stdout,"\nBlas_Sum() A:%dx1, value:%d\n",M,N);
    std::cout << "output:\n" << ans << std::endl;
    }

//  test Blas_Add_Mult()

    LaVectorComplex X(M);
    LaVectorComplex Y(M);
    X = N;
    Y = N;
    LaComplex scalar = M;
    Blas_Add_Mult(Y, scalar,X);
    if (output) {
    fprintf(stdout,"\nBlas_Add_Mult() alpha:%d, X:%dx1, Y:%dx1\n",\
                    M,N,N); 
    std::cout << "output:\n" << Y << std::endl;
    }

    Blas_Mult(Y, scalar, X);
    Blas_Norm1(Sum);
    Blas_Norm2(Sum);

//  test Blas_Copy()

    X = M*N;
    Blas_Copy(Y, X);
    if (output) {
    std::cout <<"\nBlas_Copy(LaVectorComplex):\n" << "X:\n" << X << "\nY:\n" << Y << std::endl; 
    }

//  test Blas_Dot_Prod()

    X = M;
    Y = N;
    LaComplex cans = Blas_H_Dot_Prod(X,Y);
    if (output) {
    fprintf(stdout,"\nBlas_H_Dot_Prod() X = %d, Y = %d\n",M,N); 
    fprintf(stdout,"  X is %dx1, Y is %dx1\n",M,M); 
    std::cout << "\nAns:\n" << cans << std::endl;
    }

    cans = Blas_U_Dot_Prod(X,Y);

//  test Blas_Norm2()

    ans = Blas_Norm2(X);
    if (output) {
    fprintf(stdout,"\nBlas_Norm2() X = %d\n",M); 
    fprintf(stdout,"  X is %dx1\n",M); 
    std::cout << "\nAns:\n" << ans << std::endl;
    }

// see note in blas1++.cc
//#if 0
//  test Blas_Scale()

    LaComplex scale = 5.0;
    X = 1.1;
    if (output) {
    fprintf(stdout,"\nBlas_Scale() scale = 5.0, X = 1.1\n");
    }
    Blas_Scale(scale,X);
    if (output) {
    std::cout <<"X:\n"<< X << std::endl;
    }
// #endif

//  test Blas_Swap()

    LaVectorComplex A(5);
    LaVectorComplex B(5);
    A = 1.1;
    B = 2.0;
    if (output) {
    fprintf(stdout,"\nBlas_Swap() A = 1.1, B = 2.0\n");
    }
    Blas_Swap(A,B);
    if (output) {
    std::cout <<"A:\n"<< A << "\nB:\n" << B << std::endl;
    }


//  test Blas_Index_Max()


    int index;
    X = LaComplex(8.0);
    X (M/2) = LaComplex(64.0);
    if (output) {
    fprintf(stdout,"\nBlas_Index_Max() X = 8.0\n");
    }
    index = Blas_Index_Max(X);
    if (output) {
    std::cout <<"index:\n"<< index << std::endl;
    }
    }
    
    return 0;
}
