// Testing functions for the BLAS sub-library

#include <iostream>
//#define LA_COMPLEX_SUPPORT
#include <lavd.h>
#include <lavc.h>
#include <blas1pp.h>
//#include <lapackpp/blas3pp.h>
using namespace std;

// #ifndef __FILE__
// # define __FILE__ "unknown"
// #endif
// #ifndef __LINE__
// # define __LINE__ "unknown"
// #endif

#define check_assert(expr)  if (!(expr)) { std::cout << __FILE__ << ": Failed check in line " << __LINE__ << std::endl; error = true; }

int main()
{
    bool error = false;

    {
	// Checking Blas_H_Dot_Prod
	LaVectorComplex a(3),b(3);
	a=LaComplex(1,0);
	b=LaComplex(1,0);
	check_assert(Blas_H_Dot_Prod(a,b) == LaComplex(3.0));
    }

    {
	// Checking LaVectorDouble::inject
	LaVectorDouble m,n;
	m.resize(3,1);
	m=2.0;
	n.resize(2,1);
	n=1.0;
	m(LaIndex(0,1)).inject(n);
	check_assert(m(0) == 1.0);
	check_assert(m(1) == 1.0);
	check_assert(m(2) == 2.0);
    }

    {
      LaVectorComplex a(1),b(2);
      a(0) = LaComplex(2,2);
      if (0) std::cout << "Blas_Norm1(a)=" << Blas_Norm1(a) << "\n"
		<< "Blas_Norm2(a)=" << Blas_Norm2(a) << "\n"
		<< "Blas_Norm_Inf(a)=" << Blas_Norm_Inf(a) << "\n"
		<< std::endl;
      check_assert(Blas_Norm1(a) == hypot(2,2));
    }

    return error ? 1 : 0;
} 
