//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.


#ifndef _LA_BAND_MAT_DOUBLE_H_
#define _LA_BAND_MAT_DOUBLE_H_

#include "arch.h"
#ifndef _LA_GEN_MAT_DOUBLE_H_
#include LA_GEN_MAT_DOUBLE_H
#endif


#define BOUNDS_CHK
//#define SPARSE_CHK
#ifdef LA_NO_BOUNDS_CHECK
#undef BOUNDS_CHK
#endif
#ifdef LA_NO_SPARSE_CHECK
#undef SPARSE_CHK
#endif

class DLLIMPORT LaBandMatDouble
{
  LaGenMatDouble data_;  // internal storage.

  int N_;       // N_ is (NxN)
  int kl_;      // kl_ = # subdiags
  int ku_;      // ku_ = # superdiags
  static double outofbounds_; // value returned if index is out of range.
  static int debug_;         // print debug info.
  static int *info_;         // print matrix info only, not values
                             //   originally 0, set to 1, and then
                             //   reset to 0 after use.


public:

  // constructors

  LaBandMatDouble();
  LaBandMatDouble(int,int,int);
  LaBandMatDouble(const LaBandMatDouble &);

  // destructor

  ~LaBandMatDouble();

  // operators

  LaBandMatDouble& operator=(double);
  LaBandMatDouble& operator=(const LaBandMatDouble&);
  inline double& operator()(int,int);
  inline const double& operator()(int,int) const;
  friend std::ostream& operator<<(std::ostream &, const LaBandMatDouble &);


  // member functions

  inline int size(int) const;           // submatrix size
  inline int inc(int d) const;          // explicit increment
  inline int gdim(int d) const;         // global dimensions

  inline LaBandMatDouble& ref(LaBandMatDouble &);
  LaBandMatDouble copy(const LaBandMatDouble &);
  inline double* addr() const {        // return address of matrix.
        return data_.addr();}
  inline int ref_count() const {        // return ref_count of matrix.
        return data_.ref_count();}
  inline LaIndex index(int d) const {     // return indices of matrix.
        return data_.index(d);}
  inline int superdiags() {     // return # of superdiags of matrix.
        return (ku_);}
  inline int superdiags() const { // return # of superdiags of const matrix.
        return (ku_);}
  inline int subdiags() {     // return # of subdiags of matrix.
        return (kl_);}
  inline int subdiags() const {  // return # of subdiags of const matrix.
        return (kl_);}
  inline int shallow() const {      // return shallow flag.
        return data_.shallow();}
  inline int debug() const {    // return debug flag.
        return debug_;}
  inline int debug(int d) { // set debug flag for lagenmat.
        return debug_ = d;}

  LaBandMatDouble& resize(const LaBandMatDouble&);

  inline const LaBandMatDouble& info() const {
        int *t = info_;
        *t = 1;
        return *this;};

  inline LaBandMatDouble print_data() const 
    { std::cout << data_; return *this;}

};


  
  // member functions and operators

inline LaBandMatDouble& LaBandMatDouble::ref(LaBandMatDouble &ob)
{

  data_.ref(ob.data_);
  N_ = ob.N_;
  kl_ = ob.kl_;
  ku_ = ob.ku_;

  return *this;
}

inline int LaBandMatDouble::size(int d) const
{
   return(data_.size(d));
}

inline int LaBandMatDouble::inc(int d) const
{
   return(data_.inc(d));
}

inline int LaBandMatDouble::gdim(int d) const
{
   return(data_.gdim(d));
}

inline double& LaBandMatDouble::operator()(int i, int j)
{
#ifdef LA_BOUNDS_CHECK
   assert(i >= 0);
   assert(i < N_);
   assert(j >= 0);
   assert(j < N_);
#endif

   if (i<0)
   {
      if (-i<=kl_)
	 return data_(kl_+i-j,j);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }

   if (i>=j)
   {
      if (i-j<=kl_)
	 return data_(kl_+ku_+i-j,j);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }

   else //  (j>i)
   {
      if (j-i<=ku_)
	 return data_(kl_+ku_+i-j,j); // kl_ is factorization storage here.
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }
}


inline const double& LaBandMatDouble::operator()(int i, int j) const
{
#ifdef LA_BOUNDS_CHECK
   assert(i >= 0);
   assert(i < N_);
   assert(j >= 0);
   assert(j < N_);
#endif

   if (i<0)
   {
      if (-i<=kl_)
	 return data_(kl_+i-j,j);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }

   if (i>=j)
   {
      if (i-j<=kl_)
	 return data_(kl_+ku_+i-j,j);
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }

   else // (j>i)
   {
      if (j-i<=ku_)
	 return data_(kl_+ku_+i-j,j); // kl_ is factorization storage here.
      else
      {
#ifdef LA_BOUNDS_CHECK
	 assert(0);
#else
	 return outofbounds_;
#endif
      }
   }
}

#endif // _LA_BAND_MAT_DOUBLE_H_
