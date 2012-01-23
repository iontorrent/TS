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

#include <stdlib.h>         /* for atoi() and exit() */

#include "lafnames.h"
#include LA_EXCEPTION_H
#include LA_GEN_MAT_DOUBLE_H
#include "laexcp.h"
#include LA_TEMPLATES_H
#include <string>

bool output = true;

int test_scale(int N)
{
   LaGenMatDouble bla1, bla2;
   la::ones(bla1, N);
   la::ones(bla2, N);
   bla1 *= 2.0;
   bla2.add(1.0);
   if (bla1.equal_to(bla2))
     if (output)
      std::cout << "la::equal is true (correct)" << std::endl;
   else {
     if (output)
      std::cout << "la::equal is false (oops, wrong)" << std::endl;
      return 1; 
   }
   return 0;
}

int test_templates()
{
  LaGenMatDouble bla, eye;
  la::ones(bla, 4, 5);
  la::zeros(bla, 3, 3);
  la::eye(eye, 3, 3);
  LaGenMatDouble foo(la::ones<LaGenMatDouble>(1, 3));
  la::from_diag(bla, foo);
  if (la::equal(eye, bla))
    if (output)
    std::cout << "la::equal is true (correct)" << std::endl;
  else {
    if (output)
    std::cout << "la::equal is false (oops, wrong)" << std::endl;
    return 1; 
  }
  LaGenMatDouble eye2(la::from_diag(la::ones<LaGenMatDouble>(1, 4)));
  if (la::equal(eye2, la::eye<LaGenMatDouble>(4, 4)))
    if (output)
    std::cout << "la::equal is true (correct)" << std::endl;
  else {
    if (output)
    std::cout << "la::equal is false (oops, wrong)" << std::endl;
    return 1;
  }
  la::trace(bla);
  la::rand(bla);
  bla = la::linspace<LaGenMatDouble>(1, 49, 10);
  LaGenMatDouble blub = la::repmat(bla, 2, 4);
  if (output)
  std::cout << "repmat: " << blub << "end of blub." << std::endl;

  if (output)
  std::cout << "Row index 1 of blub: " << blub.row(1) << std::endl;
  blub.row(1) = 2.0;
  if (output)
  std::cout << "Row 1 assigned to 2; blub is: " << blub << std::endl;
  if (blub(1,1) != 2.0)
      return 1;

  return 0;
}

int test_index()
{
  LaGenMatDouble A(10,10), B, C;
  LaIndex II(1,9,2);
  LaIndex JJ(1,1,1);

  // Note: The elements of A are not yet initialized; now
  // initialize them to zero
  A = 0.0;
  B.ref(A(II,II));   // B references the following elements of A:
                     //    (1, 1), (1, 3), (1, 5), (1, 7), (1, 9)
                     //    (3, 1), (3, 3), (3, 5), (3, 7), (3, 9)
                     //    (5, 1), (5, 3), (5, 5), (5, 7), (5, 9)
                     //    (7, 1), (7, 3), (7, 5), (7, 7), (7, 9)
                     //    (9, 1), (9, 3), (9, 5), (9, 7), (9, 9)
  B(2,3) = 3.1;      // Set A(5,7) = 3.1
  if (A(5, 7) != 3.1)
     return 1;

  C.ref(B(LaIndex(2,2,4), JJ));
                     // C references the following elements of B:
                     //    (2, 1)
                     // C references the following elements of A:
                     //    (5, 3)

  C = 1.1;           // Set B(2,1) = A(5,3) = 1.1
  if (A(5, 3) !=  1.1)
     return 1;
  if (B(2, 1) !=  1.1)
     return 1;

  if (output) {
  std::cout << "Test the index operator; A: the following matrix should be almost zeros" << std::endl;
  std::cout << A << std::endl;
  std::cout << "Test the index operator; B: This should be a 5x5 matrix" << std::endl;
  std::cout << B << std::endl;
  std::cout << "Test the index operator; C: This should return '1.1'" << std::endl;
  std::cout << C;
  C.Info(std::cout) << std::endl;
  }

  return 0;
}

int test_inject() 
{
   LaGenMatDouble A(3,3), B(3,3), C(2,2);

   A = 0;
   // element assignments to submatrix views will work:
   A(LaIndex(0,2),LaIndex(0,2))=1;
   B(LaIndex(0,2),LaIndex(0,2))=2;
   // note: parts of B are still uninitialized
   C(LaIndex(0,1),LaIndex(0,1))=3;

   A(LaIndex(0,1),LaIndex(0,1)).inject(C);
   // B(LaIndex(0,1),LaIndex(0,1))=C;
   // ^^ will break due to copy(), but inject() would work.

   // use a submatrix view to modify a column of A
   LaGenMatDouble D(A(LaIndex(), LaIndex(1)));
   D(0,0) = 4;
   D(1,0) = 5;
   D(2,0) = 6;

   // and check whether the resulting matrix is correct
   double expected_res[] = { 3, 4, 1,
			     3, 5, 1,
			     1, 6, 1};
   LaGenMatDouble expected(expected_res, 3, 3, true); // row-ordered

   if (output)
   std::cout<< " Matrix A: " << A <<std::endl;
   //std::cout<<B<<std::endl;

   if (!expected.equal_to(A))
      return 1;

   return 0;
}

int main(int argc, char *argv[])
{
    int M, N;
    int i,j;

    LaException::enablePrint(true);

    if (argc <3)
    {
        std::cout << "Usage " << argv[0] << " M N " << std::endl;
	assert(argc >= 3);
        exit(1);
    }
    if (argc > 3)
      if (std::string(argv[3])=="q")
	output = false;

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    
    double v[100]; // = {4.0};
    for (int k=0;k<100;k++) v[k] = 4.0;
    
    // Test constructors
    //

    LaGenMatDouble A;
    if (output)
    std::cout << std::endl << "null consturctor " << std::endl;
    if (output)
    std::cout << "A(): " << A.info() << std::endl;

    if (output)
    std::cout << std::endl << "(int, int) constructor " << std::endl;
    LaGenMatDouble C(M,N);
    if (output)
    std::cout << "C(M,N): " << C.info() << std::endl;

    // C.debug(1);
    if (output)
    std::cout << std::endl << "X(const &X) constructor " << std::endl;
    LaGenMatDouble D(C);        // D is also N,N
    if (output)
    std::cout << "D(C) :" << D.info() << std::endl;


    if (output)
    std::cout << std::endl; 
    LaGenMatDouble K, O(100,100);
    LaGenMatDouble L(O(LaIndex(2,4),LaIndex(2,8)));
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

    
    LaIndex I(std::min(2,M-1),M-1), J(1,N-1);       // evens, odd
    if (output)
    std::cout << std::endl << "create indices  I=" << I << ", J=" << J << std::endl;

    LaGenMatDouble E(C(I, J));
    if (output)
    std::cout << std::endl << "X(const &X) constructor with submatrices " << std::endl;
    if (output)
    std::cout << "E(C(I,J)): " << E.info() << std::endl;


    for (j=0;j<N; j++)
        for (i=0; i<M; i++)
            C(i,j) = i + j/100.0;

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

    std::cout << std::endl;
    }
    LaGenMatDouble F;
    if (output)
    std::cout << "F.ref(C(I,J))\n";
    F.ref(C(I,J));
    if (output)
    std::cout << F.info() << std::endl;
    F = 4.44;
    if (output)
    std::cout <<"F:\n" << std::endl;
    if (output)
    std::cout << F << std::endl;

    E.inject(F); // changed due to changed operator= semantics
    if (output) {
    std::cout << std::endl;
    std::cout << "operator=() " << std::endl;
    std::cout << "E = F : " << E.info() << C << std::endl;
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
    LaGenMatDouble B;
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

    return test_index() +
	test_inject() +
	test_templates() +
	test_scale(N);
}
