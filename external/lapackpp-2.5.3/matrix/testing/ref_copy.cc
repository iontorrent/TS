// Testing functions for matrix assignments

#include "gmd.h"
#include <iostream>
using namespace std;

// #ifndef __FILE__
// # define __FILE__ "unknown"
// #endif
// #ifndef __LINE__
// # define __LINE__ "unknown"
// #endif

#define check_assert(expr)  if (!(expr)) { std::cout << __FILE__ << ": Failed check in line " << __LINE__ << std::endl; error = true; }

int main(int argc, char* argv[]){
  bool error = false;

  // Just an example matrix
  LaGenMatDouble A(1,3);
  A(0,0)=0; A(0,1)=1; A(0,2)=2;
  A.debug(1);

  LaGenMatDouble B = A(LaIndex(),LaIndex(1,2)); // huh? Is a reference to A
  cout << "^^^^^^^ here was the error" << endl;

  LaGenMatDouble D(A(LaIndex(),LaIndex(0,1))); // copy ctor? also a reference. sigh.
  cout << "^^^^^^^ and another error" << endl;

  //  A.debug(0);
  LaGenMatDouble C;
  C = A(LaIndex(),LaIndex(1,2)); // correctly creates a copy of A

  cout << "B.info= " << B.info() << endl;
  cout << "D.info= " << B.info() << endl;
  cout << "C.info= " << C.info() << endl;

  cout << "A is" << endl << A << "B is" << endl << B
       << "C is" << endl << C << "D is" << endl << D;

  LaGenMatDouble B2 = A(LaIndex(),LaIndex(1,2)).copy(); // huh? Is a reference to A
  LaGenMatDouble D2(A(LaIndex(),LaIndex(0,1)).copy()); // copy ctor? also a reference. sigh.

  B(0,0) = 10;
  D(0,0) = 20;
  cout << "A is" << endl << A << "B is" << endl << B
       << "C is" << endl << C << "D is" << endl << D;

  // Unfortunately these are known to fail:
  check_assert(A(0,1) == 1);
  check_assert(A(0,0) == 0);
  if (error) {
    cout << "Ignoring known errors." << endl;
    error = false;
  }

  // These, on the other hand, work correctly:
  check_assert(C(0,0) == 1);

  check_assert(B2(0,0) == 1);
  check_assert(D2(0,0) == 0);

  return error ? 1 : 0;
}

