// -*-C++-*- 

// Copyright (C) 2005
// Christian Stimming <stimming@tuhh.de>

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2, or (at
// your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.

#ifndef VTMPL_H
#define VTMPL_H

#include <cstring> // for memcpy()
#include "lafnames.h"
#include LA_EXCEPTION_H  // for assert()

/** This file and this namespace includes the template functions
 * that are common to all simple vector classes.
 *
 * This way we do not start to switch from normal classes to
 * template classes to the outside, but in the inside
 * implementation all classes already use the identical function
 * code. Of course this has the advantage that one bugfix will be
 * available in all classes at once.*/
namespace vtmpl {

/** Resize to a \e new vector of size n. The element values of the
 * new vector are \e uninitialized, even if resizing to a smaller
 * vector. */
template<class V> inline int
resize(V& vec, int new_size)
{
   assert(new_size >= 0);

   // this actually frees memory first, then resizes it.  it reduces
   // internal fragmentation of memory pool, and the resizing of
   // matrices > 1/2 available memory.

   vec.ref(V(0));   // possibly free up destination
   if (new_size > 0)
      vec.ref(V(new_size));
   return new_size;
}

/** Copy elements of s into the memory space referenced by the
 * left-hand side, without first releasing it. The effect is
 * that if other vectors share memory with left-hand side,
 * they too will be affected. Note that the size of s must be
 * the same as that of the left-hand side vector. 
 *
 * @note If you rather wanted to create a new copy of \c s,
 * you should use \c copy() instead. */
template<class V> inline V&
inject(V& dest, const V& src)
{
   assert(src.size() == dest.size());
   typedef typename V::value_type value_type;

   const value_type *srcptr = src.addr();
   value_type *destptr = dest.addr();
      
   int N = dest.size();
   //for (int i=0; i<N; i++)
   // destptr[i] = srcptr[i];
   memcpy(destptr, srcptr, N*sizeof(value_type));

   return dest;
}

/** Release left-hand side (reclaiming memory space if
 * possible) and copy elements of elements of \c s. Unline \c
 * inject(), it does not require conformity, and previous
 * references of left-hand side are unaffected. */
template<class V> inline V&
copy(V& dest, const V& src)
{
   dest.resize(src.size());

   return inject(dest, src);
}

/** Prints this vector to the ostream. */
template<class V> inline std::ostream&
print(std::ostream& s, const V& m)
{
   int n = m.size();
   for (int i=0; i<n; i++)
      s << m(i) << "  ";
   s << std::endl;
   return s;
}

/** Set elements of left-hand size to the scalar value s. No
 * new vector is created, so that if there are other vectors
 * that reference this memory space, they will also be
 * affected. */
template<class V> inline V&
assign(V& vec, typename V::value_type scalar)
{
   typename V::value_type *iter = vec.addr();
   //for (int i=0; i<vec.size(); i++)
   //      iter[i] = scalar;

   // Writing it in the following way basically cuts the number of
   // memory read/writes in half.
   typename V::value_type * end;

   // This algorithm is borrowed from dscal.f
   int m = vec.size() % 5;
   if (m != 0) 
   {
      end = vec.addr() + m;
      for ( ; iter != end; ++iter)
	 *iter = scalar;

      if (vec.size() < 5)
	 return vec;
   }
   end = vec.addr() + vec.size();
   for ( ; iter != end; iter+=5)
   {
      *iter = scalar;
      iter[1] = scalar;
      iter[2] = scalar;
      iter[3] = scalar;
      iter[4] = scalar;
   }

   return vec;
}

} // namespace

#endif // VTMPL_H
