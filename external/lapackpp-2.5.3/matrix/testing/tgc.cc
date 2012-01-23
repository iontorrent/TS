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


//#include "lapackpp.h"

#define LA_BOUNDS_CHECK

#include <stdlib.h>
#include "lafnames.h"
#include LA_GEN_MAT_COMPLEX_H
#include LA_EXCEPTION_H
#include "blas3pp.h"

            
int main(int argc, char *argv[])
{
   int M, N;
   int i,j;
   bool output = true;

   LaException::enablePrint(true);

   if (argc <3)
   {
      std::cout << "Usage " << argv[0] << " M N " << std::endl;
      exit(1);
   }
   if (argc > 3)
      if (std::string(argv[3])=="q")
	 output = false;

   M = atoi(argv[1]);
   N = atoi(argv[2]);
    
   COMPLEX v[100]; // = {4.0};
   for (int k=0;k<100;k++) v[k] = LaComplex(4.0, 1.1);
    
   // Test constructors
   //

   LaGenMatComplex A;
   if (output) {
      std::cout << std::endl << "null consturctor " << std::endl;
      std::cout << "A(): " << A.info() << std::endl;
    
      std::cout << std::endl << "(int, int) constructor " << std::endl;
   }
   LaGenMatComplex C(M,N);
   if (output) {
      std::cout << "C(M,N): " << C.info() << std::endl;

      // C.debug(1);
      std::cout << std::endl << "X(const &X) constructor " << std::endl;
   }
   LaGenMatComplex D(C);        // D is also N,N
   if (output) {
      std::cout << "D(C) :" << D.info() << std::endl;


      std::cout << std::endl; 
   }
   LaGenMatComplex K, O(100,100);
   LaGenMatComplex L(O(LaIndex(2,4),LaIndex(2,8)));
   if (output) 
      std::cout << "L(O(LaIndex(2,4),LaIndex(2,8)))\n";
   L = 0.0;
   if (output) 
      std::cout << "L: " << L.info() << std::endl;

   if (output) 
      std::cout << std::endl <<"K.copy(L) " << std::endl;
   K.copy(L);
   if (output) 
      std::cout << "K: " << K.info() << std::endl;
   if (output) 
      std::cout << "K:\n" << K << std::endl;

    
   LaIndex I(2,M-1), J(1,N-1);       // evens, odd
   if (output) 
      std::cout << std::endl << "create indices  I=" << I << ", J=" << J << std::endl;

   LaGenMatComplex E(C(I, J));
   if (output) 
      std::cout << std::endl << "X(const &X) constructor with submatrices " << std::endl;
   if (output) 
      std::cout << "E(C(I,J)): " << E.info() << std::endl;


   for (j=0;j<N; j++)
      for (i=0; i<M; i++)
	 C(i,j) = LaComplex(i + j/100.0);

   if (output) {
      std::cout << std::endl;   
      std::cout << "test operator(int, int)"  << std::endl;
      std::cout << "Initalize C(i,j) = i + j/100.0 " << std::endl;
      std::cout << "C: " << C.info() << std::endl;
      std::cout <<  C << std::endl;

      std::cout << std::endl;
      std::cout <<  "operator(LaIndex, LaIndex)" << std::endl;  
      std::cout << "C(I,J) " << C(I,J).info() << std::endl;
      std::cout <<  C(I,J) << std::endl;

      std::cout << std::endl;
      std::cout << "test missing indices (default to whole row or column" << std::endl;
      std::cout << "C(LaIndex(),J) " << C(LaIndex(),J).info() << std::endl;
      std::cout << C(LaIndex(),J) << std::endl;
      std::cout << std::endl;
      std::cout << "C(I,LaIndex()) " << C(I,LaIndex()).info() << std::endl;
      std::cout << C(I,LaIndex()) << std::endl;
   }

   if (output) 
      std::cout << std::endl;
   LaGenMatComplex F;
   if (output) 
      std::cout << "F.ref(C(I,J))\n";
   F.ref(C(I,J));
   if (output) 
      std::cout << F.info() << std::endl;
   F = 4.44;
   if (output) {
      std::cout <<"F:\n" << std::endl;
      std::cout << F << std::endl;
   }

   E.inject(F); // changed due to changed operator= semantics
   if (output) {
      std::cout << std::endl;
      std::cout << "operator=() " << std::endl;
      std::cout << "E = F : " << E.info() << std::endl;
   }

   D = C;
   if (output) {
      std::cout << std::endl;
      std::cout << "operator=(const Matrix&) "<< std::endl;
      std::cout << "D = C : " << D.info() << std::endl;
      std::cout << D << std::endl;


      std::cout << std::endl;
      std::cout << "test automatic destructuion of temporaries: " << std::endl;
   }
   LaGenMatComplex B;
   for (int c=0; c<10; c++)
   {
      B.ref(C(I,J));
      if (output) 
	 std::cout << "B.ref(C(I,J)): " << B.info() << std::endl;
   }

   C.ref(C);
   if (output) {
      std::cout << std::endl;
      std::cout <<"test C.ref(C) case works correctly." << std::endl;
				    std::cout << "C.ref(C) " << C.info() << std::endl;
   }

   Blas_NormF(D);

   return 0;
}
