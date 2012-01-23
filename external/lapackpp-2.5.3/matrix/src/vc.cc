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


#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "lafnames.h"
#include LA_EXCEPTION_H
#include VECTOR_COMPLEX_H
#include "vtmpl.h"

VectorComplex::VectorComplex(unsigned n)
   : p(new vref_type(n))
   , data(p->data)
{                                                                      
    if (!data) throw LaException("VectorComplex(unsigned)", "out of memory");
}                                                                      


VectorComplex::VectorComplex(const VectorComplex& m)
{
   // shallow assignment semantics
   ref_vref(m.p);
}

VectorComplex::VectorComplex(value_type *d, unsigned n)
   : p(new vref_type(d, n))
   , data(p->data)
{
   if (!d) throw LaException("VectorComplex(unsigned)", "data is NULL");
}                                                                      

VectorComplex::VectorComplex(value_type *d, unsigned m, unsigned n, 
			   bool row_ordering)
   : p(row_ordering ? new vref_type(m*n) : new vref_type(d, m*n))
   , data(p->data)
{                                                                      
   if (!d)
      throw LaException("VectorComplex", "data is NULL");

   if(!row_ordering)
   {
      // nothing else to do
   }
   else // row ordering
   {
      if (!data)
	 throw LaException("VectorComplex", "out of memory");
      for(unsigned i=0; i < m*n; i++)
      {
	 data[m*(i%n)+(i/n)]=d[i];  // reorder the data to column-major
      }
   }
}                                                                      
                                                                       
VectorComplex::VectorComplex(unsigned n, value_type scalar)
   : p(new vref_type(n))
   , data(p->data)
{
   if (!data)
      throw LaException("VectorComplex(int,double)", "out of memory");
   vtmpl::assign(*this, scalar);
}

VectorComplex::~VectorComplex()
{
   unref_vref();
}

int VectorComplex::resize(unsigned d)
{
   return vtmpl::resize(*this, d);
}

VectorComplex& VectorComplex::inject(const VectorComplex& m)
{
   if (m.size() != size())
      throw LaException("VectorComplex::inject(VectorComplex)", "vector sizes do not match");

   return vtmpl::inject(*this, m);
}

VectorComplex& VectorComplex::copy(const VectorComplex &m)
{
   return vtmpl::copy(*this, m);
}

std::ostream& operator<<(std::ostream& s, const VectorComplex& m)
{
   return vtmpl::print(s, m);
}

VectorComplex& VectorComplex::operator=(value_type scalar)
{
   // Very simple version:
   //for (int i=0; i<size(); i++)
   // data[i] = scalar;
   //return *this;

   // Less simple version:
   //return vtmpl::assign(*this, scalar);

   // Heavily optimized version.

   // Cache the complex value
   double s_re = scalar.r;
   double s_im = scalar.i;
   value_type * iter = data;
   value_type * end;
   const int _blocksize = 8;

   // This idea and algorithm is borrowed from dscal.f
   int m = size() % _blocksize;
   if (m != 0) 
   {
      end = data + m;
      for ( ; iter != end; ++iter)
      {
	 iter->r = s_re;
	 iter->i = s_im;
      }
      if (size() < _blocksize)
	 return *this;
   }

   end = data + size();
   for ( ; iter != end; iter += _blocksize)
   {
      iter->r = 
	 iter[1].r = 
	 iter[2].r = 
	 iter[3].r = 
	 iter[4].r = 
	 iter[5].r = 
	 iter[6].r = 
	 iter[7].r = 
	 s_re;
      iter->i =
	 iter[1].i =
	 iter[2].i =
	 iter[3].i =
	 iter[4].i =
	 iter[5].i =
	 iter[6].i =
	 iter[7].i =
	 s_im;
   }

   return *this;
}
