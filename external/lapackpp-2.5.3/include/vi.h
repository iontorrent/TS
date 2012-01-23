//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.
//
//      Lapack++ "Shared" Vector Int Class
//
//      A lightweight vector class with minimal overhead.
//
//      shallow assignment
//      unit stride
//      inlined access A(i)
//      optional (compile-time) array bounds checking through 
//              VECTOR_INT_BOUNDS_CHECK
//      A(i) is the same as A[i]
//      auto conversion to int*
//      a null vector has size of 0, but has the ref_count structure
//              has been initalized
//

#ifndef _VECTOR_INT_H_
#define _VECTOR_INT_H_    

#include <iostream>       // for formatted printing of matrices

#ifndef __ASSERT_H
#include <cassert>     // cheap "error" protection used in checking
#endif                  // checking array bounds.

#include "arch.h"

typedef struct vrefInt {
      typedef int value_type;
      int        sz;                                        
      value_type*    data;                                       
      int        ref_count;
      int        vref_ref_count;
      vrefInt(value_type *_data, int _sz)
	 : sz(_sz)
	 , data(_data)
	 , ref_count(2)
	 , vref_ref_count(1)
      {};
      vrefInt(int _sz)
	 : sz(_sz)
	 , data(new value_type[sz])
	 , ref_count(1)
	 , vref_ref_count(1)
      {};
} vrefInt;
                        


class DLLIMPORT VectorInt
{
   public:
      /// The type of the values in this vector
      typedef int value_type;
   private:
      /// The type of the internal management structure
      typedef vrefInt vref_type;
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
                                                                       
    //inline VectorInt();     // this should behave as VectorInt(0)
    VectorInt(unsigned size);                             
    VectorInt(unsigned size, int scalar);   // can't be inlined because of 'for'
                                       // statement.
    VectorInt(int*, unsigned size);
    VectorInt(int*, unsigned, unsigned, bool);
    VectorInt(const VectorInt&); 
    ~VectorInt() ;                              
                                                                       
        /*::::::::::::::::::::::::::::::::*/                           
        /*  Indices and access operations */                           
        /*::::::::::::::::::::::::::::::::*/                           
                                                                       
    inline int&     operator[](int); 
    inline const int&     operator[](int) const; // read only
    inline int&     operator()(int); 
    inline const int&     operator()(int) const; // read only
    inline              operator    int*(); 
    inline int          size() const;
    inline int          null() const;
           int          resize(unsigned d);
    inline int          ref_count() const;  // return the number of ref counts
    inline int*     addr() const;
                                                                       
        /*::::::::::::::*/                                             
        /*  Assignment  */                                             
        /*::::::::::::::*/                                             
                                                                       
    inline  VectorInt& operator=(const VectorInt&);
            VectorInt& operator=(int);
    inline  VectorInt& ref(const VectorInt &);
            VectorInt& inject(const VectorInt&);
            VectorInt& copy(const VectorInt&);

    /* I/O */                                                      
    friend std::ostream&   operator<<(std::ostream&, const VectorInt&);       

};                                                                     


    // operators and member functions

inline int VectorInt::null()    const
{
    return (size() == 0) ;
}

inline int VectorInt::size() const
{
    return   p-> sz;
}


inline int VectorInt::ref_count() const
{
    return p->ref_count;
}

inline int* VectorInt::addr() const
{
    return data;
}

inline VectorInt::operator int*() 
{
    return data;
}


inline int& VectorInt::operator()(int i)
{
#ifdef VECTOR_INT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif 
    return data[i];
}

inline const int& VectorInt::operator()(int i) const
{
#ifdef VECTOR_INT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif
    return data[i];
}

//  *CHANGE*  [] is the same as ()
inline int& VectorInt::operator[](int i)
{
#ifdef VECTOR_INT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

//  *CHANGE*  [] is the same as ()
inline const int& VectorInt::operator[](int i) const
{
#ifdef VECTOR_INT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

inline void VectorInt::ref_vref(vref_type* other)
{
   p = other;
   data = p->data;
   p->ref_count++;
   p->vref_ref_count++;
}

inline void VectorInt::unref_vref()
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

inline VectorInt& VectorInt::ref(const VectorInt& m)
{
   if (&m != this)
   {
      unref_vref();
      ref_vref(m.p);
   }
   return *this;
}

inline VectorInt& VectorInt::operator=(const VectorInt& m)
{

    return  ref(m);
}




#endif 
// _VECTOR_INT_H_

