//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.
//
//      Lapack++ "Shared" Vector Complex Class
//
//      A lightweight vector class with minimal overhead.
//
//      shallow assignment
//      unit stride
//      inlined access A(i)
//      optional (compile-time) array bounds checking through 
//              VECTOR_COMPLEX_BOUNDS_CHECK
//      A(i) is the same as A[i]
//      auto conversion to complex*
//      a null vector has size of 0, but has the ref_count structure
//              has been initalized
//

#ifndef _VECTOR_COMPLEX_H_
#define _VECTOR_COMPLEX_H_    

#include <iostream>       // for formatted printing of matrices
#include "arch.h"
#include "lacomplex.h"

#ifndef __ASSERT_H
#include <cassert>     // cheap "error" protection used in checking
#endif                  // checking array bounds.

#ifndef LA_COMPLEX_SUPPORT
/* An application must define LA_COMPLEX_SUPPORT if it wants to use
 * complex numbers here. */
# error "The macro LA_COMPLEX_SUPPORT needs to be defined if you want to use complex-valued matrices."
#endif

typedef struct vrefComplex {
      typedef COMPLEX value_type;
      int        sz;                                        
      value_type * data;                                       
      int        ref_count;
      int        vref_ref_count;
      vrefComplex(value_type *_data, int _sz)
	 : sz(_sz)
	 , data(_data)
	 , ref_count(2)
	 , vref_ref_count(1)
      {};
      vrefComplex(int _sz)
	 : sz(_sz)
	 , data(new value_type[sz])
	 , ref_count(1)
	 , vref_ref_count(1)
      {};
} vrefComplex;
                        


class DLLIMPORT VectorComplex
{
   public:
      /// The type of the values in this vector
      typedef COMPLEX value_type;
   private:
      /// The type of the internal management structure
      typedef vrefComplex vref_type;
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
                                                                       
    //inline VectorComplex();     // this should behave as VectorComplex(0)
    VectorComplex(unsigned);                             
    VectorComplex(unsigned, COMPLEX);   // can't be inlined because of 'for'
                                       // statement.
    VectorComplex(COMPLEX*, unsigned);
    VectorComplex(COMPLEX*, unsigned, unsigned, bool);
    VectorComplex(const VectorComplex&); 
    ~VectorComplex() ;                              
                                                                       
        /*::::::::::::::::::::::::::::::::*/                           
        /*  Indices and access operations */                           
        /*::::::::::::::::::::::::::::::::*/                           
                                                                       
    inline COMPLEX&     operator[](int); 
    inline const COMPLEX&     operator[](int) const;  // read only
    inline COMPLEX&     operator()(int); 
    inline const COMPLEX&     operator()(int) const; // read only
    inline              operator    COMPLEX*(); 
    inline int          size() const;
    inline int          null() const;
           int          resize(unsigned d);
    inline int          ref_count() const;  // return the number of ref counts
    inline COMPLEX*     addr() const;
                                                                       
        /*::::::::::::::*/                                             
        /*  Assignment  */                                             
        /*::::::::::::::*/                                             
                                                                       
    inline  VectorComplex& operator=(const VectorComplex&);
            VectorComplex& operator=(COMPLEX);
    inline  VectorComplex& ref(const VectorComplex &);
            VectorComplex& inject(const VectorComplex&);
            VectorComplex& copy(const VectorComplex&);

    /* I/O */                                                      
    friend std::ostream&   operator<<(std::ostream&, const VectorComplex&);       

};                                                                     


    // operators and member functions

inline int VectorComplex::null()    const
{
    return (size() == 0) ;
}

inline int VectorComplex::size() const
{
    return   p-> sz;
}


inline int VectorComplex::ref_count() const
{
    return p->ref_count;
}

inline COMPLEX* VectorComplex::addr() const
{
    return data;
}

inline VectorComplex::operator COMPLEX*() 
{
    return data;
}


inline COMPLEX& VectorComplex::operator()(int i)
{
#ifdef VECTOR_COMPLEX_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif 
    return data[i];
}

inline const COMPLEX& VectorComplex::operator()(int i) const
{
#ifdef VECTOR_COMPLEX_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif
    return data[i];
}

//  [] *always* performs bounds-check 
//  *CHANGE*  [] is the same as ()
inline COMPLEX& VectorComplex::operator[](int i)
{
#ifdef VECTOR_COMPLEX_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

//  [] *always* performs bounds-check 
//  *CHANGE*  [] is the same as ()
inline const COMPLEX& VectorComplex::operator[](int i) const
{
#ifdef VECTOR_COMPLEX_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

inline void VectorComplex::ref_vref(vref_type* other)
{
   p = other;
   data = p->data;
   p->ref_count++;
   p->vref_ref_count++;
}

inline void VectorComplex::unref_vref()
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

inline VectorComplex& VectorComplex::ref(const VectorComplex& m)
{
   if (&m != this)
   {
      unref_vref();
      ref_vref(m.p);
   }
   return *this;
}

inline VectorComplex& VectorComplex::operator=(const VectorComplex& m)
{

    return  ref(m);
}


#ifndef LA_COMPLEX_SUPPORT
// Repeat this warning again
# error "The macro LA_COMPLEX_SUPPORT needs to be defined if you want to use complex-valued matrices."
#endif


#endif 
// _VECTOR_COMPLEX_H_

