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



#include <stdlib.h>
#include "lafnames.h"
#include LA_VECTOR_DOUBLE_H
#include LA_EXCEPTION_H
#include <string>

#include <stdlib.h>

bool output = true;
bool all_ok = true;

//#ifdef __GNUC__
#define ASSERT(expr) if (!(expr)) \
{ std::cout << "FAILED assertion in " << __FILE__ << ":" << __LINE__ \
            << ": " #expr << std::endl; all_ok = false; }
            

int main(int argc, char *argv[])
{
    int N;

    LaException::enablePrint(true);

    if (argc < 2)
    {
        std::cout << "Usage " << argv[0] << " N " << std::endl;
	assert(argc >= 2);
        exit(1);
    }
    if (argc > 2)
      if (std::string(argv[2])=="q")
	output = false;

    N = atoi(argv[1]);
    if (N < 2) {
        std::cout << "N must be >= 2" << std::endl;
	assert(N >= 2);
	exit(1);
    }

    double v[100];
    int i;
    for (i=0; i<100; i++)
        v[i] = i;

    // Test constructors
    //
    LaVectorDouble A;
    if (output) {
    std::cout << std::endl << "null constructor " << std::endl;
    std::cout << "A(): " << A.info() << std::endl;
    }
    ASSERT(A.size() == 0);
    ASSERT(A.ref_count() == 1);

    LaVectorDouble C(N);
    if (output) {
    std::cout << std::endl << "(int) constructor " << std::endl;
    std::cout << "C(N) : " << C.info() << std::endl;
    }
    ASSERT(C.size() == N);
    ASSERT(A.ref_count() == 1);

    LaVectorDouble B(1,N);
    if (output) {
    std::cout << std::endl ;
    std::cout << "(int, int) constructor " << std::endl;
    std::cout << "B(N) : " << B.info() << std::endl;
    }
    ASSERT(B.size() == N);

    LaVectorDouble F(v, 10);
    if (output) {
    std::cout << std::endl ;
    std::cout << "(double*, int) constructor " << std::endl;
    std::cout << "F(v,10): " << F.info() << std::endl;
    }
    ASSERT(F.size() == 10);
    ASSERT(F(1) == 1);
    ASSERT(F(2) == 2);

    C=5.5;
    if (output) {
    std::cout << std::endl;
    std::cout << "test operator=(double) " << std::endl;
    std::cout << "C = 5.5: " << C.info() << std::endl;
    std::cout << C << std::endl;
    }
    ASSERT(C(0) == 5.5);
    ASSERT(C(1) == 5.5);

    LaIndex I(1,N-1);
    C(I)=7.7;
    C(I);
    if (output) {
    std::cout << std::endl;
    std::cout << "test C(const LaIndex&) constructor" << std::endl;
    std::cout << "C(I)=7.7: " << C(I).info() << std::endl;
    std::cout << "C(I)== " << C(I) << std::endl;
    std::cout << "C== " << C << std::endl;

    std::cout << std::endl;
    std::cout << "test start(),inc(),end() " << std::endl;
    std::cout << "C.start(): " << C.start() << std::endl;
    std::cout << "C.inc(): " << C.inc() << std::endl;
    std::cout << "C.end(): " << C.end() << std::endl;
    }
    ASSERT(C(0) == 5.5); // unchanged from above assignment
    ASSERT(C(1) == 7.7);
    ASSERT(C.start() == 0);
    ASSERT(C.inc() == 1);
    ASSERT(C.end() == N-1);

    A.ref(C);
    if (output) {
    std::cout <<std::endl;
    std::cout << "test ref(const LaGenMatDouble &)" << std::endl;
    std::cout << "A.ref(C): "<< A.info() << std::endl;
    std::cout << A << std::endl;
    }
    ASSERT(A.size() == N);
    ASSERT(A(0) == 5.5);
    ASSERT(A(1) == 7.7);
    ASSERT(A.addr() == C.addr());
    ASSERT(A.ref_count() == 2);
    ASSERT(C.ref_count() == 2);

    LaVectorDouble AA(N);
    AA = 1.1;
    A.inject(AA);
    if (output) {
    std::cout <<std::endl;
    std::cout << "AA = 1.1\n";
    std::cout << "test inject(const LaGenMatDouble &)" << std::endl;
    std::cout << "A.inject(AA): "<< A.info() << std::endl;
    std::cout << A << std::endl;
    }
    ASSERT(A.size() == N);
    ASSERT(A(0) == 1.1);
    ASSERT(A(1) == 1.1);
    ASSERT(C(0) == 1.1);
    ASSERT(C(1) == 1.1);
    ASSERT(A.addr() == C.addr());
    ASSERT(A.ref_count() == 2);
    ASSERT(C.ref_count() == 2);

    AA = 2.2;
    A.copy(AA); // the reference to C will be detached here
    if (output) {
    std::cout <<std::endl;
    std::cout << "test copy(const LaGenMatDouble &)" << std::endl;
    std::cout << "A.copy(C): "<< A.info() << std::endl;
    std::cout << "       C : "<< C.info() << std::endl;
    std::cout << A << std::endl;
    }
    ASSERT(A(0) == 2.2);
    ASSERT(A(1) == 2.2);
    ASSERT(C(0) == 1.1);
    ASSERT(C(1) == 1.1);
    ASSERT(A.addr() != C.addr());
    ASSERT(A.ref_count() == 1);
    ASSERT(C.ref_count() == 1);

    LaVectorDouble D(C);        // D is also N,1
    if (output) {
    std::cout << std::endl << "test X(const &X) constructor " << std::endl;
    std::cout << "D(C) :" << D.info() << std::endl;
    }
    ASSERT(D.size() == N);
    ASSERT(D(0) == C(0));
    ASSERT(D(1) == C(1));
    ASSERT(D.addr() != C.addr());
    ASSERT(C.ref_count() == 1);
    ASSERT(D.ref_count() == 1);

    // Test the submatrixview
    LaVectorDouble G = C(LaIndex(1, N-1));
    ASSERT(G.is_submatrixview() == true);
    ASSERT(G.size() == N-1);
    ASSERT(G(0) == C(1));

    // Inject a vector in another vector
    LaVectorDouble E(N+1);
    C = 3.3;
    E = 47.11;
    // test inject()
    E(LaIndex(1, N)).inject(C);
    ASSERT(E(0) == 47.11);
    ASSERT(E(1) == 3.3);

    // Test the semantics of shallow_assign which forces a
    // copy-by-reference i.e. a ref() instead of deep-copy
    // i.e. copy().
    LaVectorDouble CC = C.shallow_assign();
    ASSERT(CC(0) == C(0));
    ASSERT(CC.addr() == C.addr());

    // the shallow_assign flag is not propagated into the new object
    LaVectorDouble DD = CC;
    ASSERT(DD(0) == C(0));
    ASSERT(CC.addr() == C.addr());
    ASSERT(DD.addr() != C.addr());

    // end of tests

    if (output)
       std::cout << "Test results: "
		 << (all_ok ? "All ok." : "At least one failed.")
		 << std::endl;

    return all_ok ? 0 : 1;
}
