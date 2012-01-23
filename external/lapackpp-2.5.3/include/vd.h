//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.
//
//      Lapack++ "Shared" Vector Double Class
//
//      A lightweight vector class with minimal overhead.
//
//      shallow assignment
//      unit stride
//      inlined access A(i)
//      optional (compile-time) array bounds checking through 
//              VECTOR_DOUBLE_BOUNDS_CHECK
//      A(i) is the same as A[i]
//      auto conversion to double*
//      a null vector has size of 0, but has the ref_count structure
//              has been initalized
//

#ifndef _VECTOR_DOUBLE_H_
#define _VECTOR_DOUBLE_H_    

#include <iostream>       // for formatted printing of matrices

#ifndef __ASSERT_H
#include <cassert>     // cheap "error" protection used in checking
#endif                  // checking array bounds.

#include "arch.h"

typedef struct vrefDouble {
      typedef double value_type;
      int        sz;                                        
      value_type*    data;                                       
      int        ref_count;
      int        vref_ref_count;
      vrefDouble(value_type *_data, int _sz)
	 : sz(_sz)
	 , data(_data)
	 , ref_count(2)
	 , vref_ref_count(1)
      {};
      vrefDouble(int _sz)
	 : sz(_sz)
	 , data(new value_type[sz])
	 , ref_count(1)
	 , vref_ref_count(1)
      {};
} vrefDouble;
                        

/** Lightwight vector class.
 *
 * One-dimensional storage class with minimal overhead. It is one step
 * above a C array, only in that it utilizes share-semantics (similar
 * to C++ string classes) for optimizing memory usage. This vector is
 * not intended for mathematical denotations, but rather used as
 * building block for other LAPACK++ matrix classes. */
class DLLIMPORT VectorDouble
{
   public:
      /// The type of the values in this vector
      typedef double value_type;
   private:
      /// The type of the internal management structure
      typedef vrefDouble vref_type;
      vref_type *p;
      value_type *data;            // performance hack, avoid COMPLEX
      // indirection to data.

      /** Dereferences the vref management structure,
       * deleting it if that was the last reference. */
      inline void unref_vref();
      /** Make this class a reference to the given vref
       * management structure. Make sure to call unref_vref()
       * beforehand, if suitable. */
      inline void ref_vref(vref_type* other);

    public:                                                            
                                                                       
        /*::::::::::::::::::::::::::*/                                 
        /* Constructors/Destructors */                                 
        /*::::::::::::::::::::::::::*/                                 
                                                                       
    //inline VectorDouble();     // this should behave as VectorDouble(0)
    VectorDouble(unsigned);                             
    VectorDouble(unsigned, double);   // can't be inlined because of 'for'
                                       // statement.
    VectorDouble(double*, unsigned);
    VectorDouble(double*, unsigned, unsigned, bool);
    VectorDouble(const VectorDouble&); 
    ~VectorDouble() ;                              
                                                                       
        /*::::::::::::::::::::::::::::::::*/                           
        /*  Indices and access operations */                           
        /*::::::::::::::::::::::::::::::::*/                           
                                                                       
    inline double&      operator[](int); 
    inline const double&      operator[](int) const;  // read only
    inline double&      operator()(int); 
    inline const double&      operator()(int) const; // read only
    inline              operator    double*(); 
    inline int          size() const;
    inline int          null() const;
           int          resize(unsigned d);
    inline int          ref_count() const;  // return the number of ref counts
    inline double*      addr() const;
                                                                       
        /*::::::::::::::*/                                             
        /*  Assignment  */                                             
        /*::::::::::::::*/                                             
                                                                       
    inline  VectorDouble& operator=(const VectorDouble&);
            VectorDouble& operator=(double);
    inline  VectorDouble& ref(const VectorDouble &);
            VectorDouble& inject(const VectorDouble&);
            VectorDouble& copy(const VectorDouble&);

    /* I/O */                                                      
    friend std::ostream&   operator<<(std::ostream&, const VectorDouble&);       

};                                                                     


    // operators and member functions

inline int VectorDouble::null() const
{
    return (size() == 0) ;
}

inline int VectorDouble::size() const
{
    return   p-> sz;
}


inline int VectorDouble::ref_count() const
{
    return p->ref_count;
}

inline double* VectorDouble::addr() const
{
    return data;
}

inline VectorDouble::operator double*() 
{
    return data;
}


inline double& VectorDouble::operator()(int i)
{
#ifdef VECTOR_DOUBLE_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif 
    return data[i];
}

inline const double& VectorDouble::operator()(int i) const
{
#ifdef VECTOR_DOUBLE_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif
    return data[i];
}

//  [] *always* performs bounds-check 
//  *CHANGE*  [] is the same as ()
inline double& VectorDouble::operator[](int i)
{
#ifdef VECTOR_DOUBLE_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

inline const double& VectorDouble::operator[](int i) const
{
#ifdef VECTOR_DOUBLE_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

inline void VectorDouble::ref_vref(vref_type* other)
{
   p = other;
   data = p->data;
   p->ref_count++;
   p->vref_ref_count++;
}

inline void VectorDouble::unref_vref()
{
   if (--(p->ref_count) == 0)              // perform garbage col.
   {
      delete [] p->data;
      delete p;
   } 
   else
   {
      // Check whether the internal management structure needs to
      // be deleted
      if (--(p->vref_ref_count) == 0)
	 delete p;
   }
}

inline VectorDouble& VectorDouble::ref(const VectorDouble& m)
{
   if (&m != this)
   {
      unref_vref();
      ref_vref(m.p);
   }
   return *this;
}

inline VectorDouble& VectorDouble::operator=(const VectorDouble& m)
{

    return  ref(m);
}




#endif 
// _VECTOR_DOUBLE_H_

