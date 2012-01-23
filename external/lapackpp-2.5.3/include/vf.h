//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.
//
//      Lapack++ "Shared" Vector Float Class
//
//      A lightweight vector class with minimal overhead.
//
//      shallow assignment
//      unit stride
//      inlined access A(i)
//      optional (compile-time) array bounds checking through 
//              VECTOR_FLOAT_BOUNDS_CHECK
//      A(i) is the same as A[i]
//      auto conversion to float*
//      a null vector has size of 0, but has the ref_count structure
//              has been initalized
//

#ifndef _VECTOR_FLOAT_H_
#define _VECTOR_FLOAT_H_    

#include <iostream>       // for formatted printing of matrices

#ifndef __ASSERT_H
#include <cassert>     // cheap "error" protection used in checking
#endif                  // checking array bounds.

#include "arch.h"

typedef struct vrefFloat {
      typedef float value_type;
      int        sz;                                        
      value_type*    data;                                       
      int        ref_count;
      int        vref_ref_count;
      vrefFloat(value_type *_data, int _sz)
	 : sz(_sz)
	 , data(_data)
	 , ref_count(2)
	 , vref_ref_count(1)
      {};
      vrefFloat(int _sz)
	 : sz(_sz)
	 , data(new value_type[sz])
	 , ref_count(1)
	 , vref_ref_count(1)
      {};
} vrefFloat;
                        


class DLLIMPORT VectorFloat
{
   public:
      /// The type of the values in this vector
      typedef float value_type;
   private:
      /// The type of the internal management structure
      typedef vrefFloat vref_type;
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
                                                                       
    //inline VectorFloat();     // this should behave as VectorFloat(0)
    VectorFloat(unsigned);                             
    VectorFloat(unsigned, float);   // can't be inlined because of 'for'
                                       // statement.
    VectorFloat(float*, unsigned);
    VectorFloat(float*, unsigned, unsigned, bool);
    VectorFloat(const VectorFloat&); 
    ~VectorFloat() ;                              
                                                                       
        /*::::::::::::::::::::::::::::::::*/                           
        /*  Indices and access operations */                           
        /*::::::::::::::::::::::::::::::::*/                           
                                                                       
    inline float&       operator[](int); 
    inline const float&       operator[](int) const;  //read only
    inline float&       operator()(int); 
    inline const float&       operator()(int) const; // read only
    inline              operator    float*(); 
    inline int          size() const;
    inline int          null() const;
           int          resize(unsigned d);
    inline int          ref_count() const;  // return the number of ref counts
    inline float*       addr() const;
                                                                       
        /*::::::::::::::*/                                             
        /*  Assignment  */                                             
        /*::::::::::::::*/                                             
                                                                       
    inline  VectorFloat& operator=(const VectorFloat&);
            VectorFloat& operator=(float);
    inline  VectorFloat& ref(const VectorFloat &);
            VectorFloat& inject(const VectorFloat&);
            VectorFloat& copy(const VectorFloat&);

    /* I/O */                                                      
    friend std::ostream&   operator<<(std::ostream&, const VectorFloat&);       

};                                                                     


    // operators and member functions

inline int VectorFloat::null()  const
{
    return (size() == 0) ;
}

inline int VectorFloat::size() const
{
    return   p-> sz;
}


inline int VectorFloat::ref_count() const
{
    return p->ref_count;
}

inline float* VectorFloat::addr() const
{
    return data;
}

inline VectorFloat::operator float*() 
{
    return data;
}


inline float& VectorFloat::operator()(int i)
{
#ifdef VECTOR_FLOAT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif 
    return data[i];
}

inline const float& VectorFloat::operator()(int i) const
{
#ifdef VECTOR_FLOAT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif
    return data[i];
}

//  [] *always* performs bounds-check 
//  *CHANGE*  [] is the same as ()
inline float& VectorFloat::operator[](int i)
{
#ifdef VECTOR_FLOAT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

//  [] *always* performs bounds-check 
//  *CHANGE*  [] is the same as ()
inline const float& VectorFloat::operator[](int i) const
{
#ifdef VECTOR_FLOAT_BOUNDS_CHECK
    assert(0<=i && i<size());
#endif  
    return data[i];
}

inline void VectorFloat::ref_vref(vref_type* other)
{
   p = other;
   data = p->data;
   p->ref_count++;
   p->vref_ref_count++;
}

inline void VectorFloat::unref_vref()
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

inline VectorFloat& VectorFloat::ref(const VectorFloat& m)
{
   if (&m != this)
   {
      unref_vref();
      ref_vref(m.p);
   }
   return *this;
}

inline VectorFloat& VectorFloat::operator=(const VectorFloat& m)
{

    return  ref(m);
}




#endif 
// _VECTOR_FLOAT_H_

