#define LA_BOUNDS_CHECK

#include <stdlib.h>
#include "lafnames.h"
#include LA_GEN_MAT_COMPLEX_H
#include LA_GEN_QR_FACT_COMPLEX_H
#include LA_UTIL_H
#include LA_GENERATE_MAT_DOUBLE_H
#include LA_EXCEPTION_H

bool output = true;

void test(int M, int N)
{
    LaGenMatComplex A(M,N);
    //LaVectorComplex x(N), b(M);
    //bool error = false;

#ifndef HPPA
    const char fname[] = "TestGenLinearSolve(LaGenMat, x, b) ";
#else
    char *fname = NULL;
#endif

    //double norm;
    //double res;

    LaRandUniform(A, -1, 1);

    // save a snapshot of what A looked like before the solution
    LaGenMatComplex old_A = A;

    if (output)
	std::cout << fname << ": testing LaGenQRFactComplex M= "<< M
	      << " N = " << N << " for A = " << A << std::endl;

    LaGenQRFactComplex F(A);

    if (output)
	std::cout << fname << ": resulting A = " << A << std::endl;
    
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
	N = ( atoi(argv[2]) > 0 ? atoi(argv[2]) : M );

    if (argc > 2)
	if (std::string(argv[2])=="q" || 
	    (argc > 3 && std::string(argv[3])=="q") )
	    output = false;

    if (output)
	std::cout << "Testing " << M << " x " << N << " system." << std::endl;

    test(M,N);

    return 0;
}


