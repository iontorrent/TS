// -*-C++-*- 

// Copyright (C) 2004 
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

//      LAPACK++ (V. 1.1)
//      (C) 1992-1996 All Rights Reserved.

/** @file
 * @brief Column vector
 */

#ifndef _LA_COL_VECTOR_DOUBLE_H_
#define _LA_COL_VECTOR_DOUBLE_H_

#include "lafnames.h"

#ifndef _LA_GEN_MAT_DOUBLE_H_
#include LA_GEN_MAT_DOUBLE_H
#endif


/** @brief Column vector
 *
 * A column vector is simply an nx1 matrix, only that it can be
 * constructed and accessed by a single dimension. (FIXME: It is
 * unclear whether this class is actually well supported inside
 * Lapack++; probably you should only use @ref LaVectorDouble
 * instead.)
 */
class LaColVectorDouble: public LaGenMatDouble
{
public:

    inline LaColVectorDouble();
    inline LaColVectorDouble(int);
    inline LaColVectorDouble(double*, int);
    inline LaColVectorDouble(const LaGenMatDouble&);

    inline int size() const;
    inline int inc() const;
    inline LaIndex index() const;
    inline int start() const;
    inline int end() const;


    inline double& operator()(int i);
    inline const double& operator()(int i) const ;
    inline LaColVectorDouble operator()(const LaIndex&);

    inline LaColVectorDouble& operator=(const LaGenMatDouble &A);
    inline LaColVectorDouble& operator=(double d);
    
};


inline LaColVectorDouble::LaColVectorDouble() : LaGenMatDouble(0,1) {}
inline LaColVectorDouble::LaColVectorDouble(int i) : LaGenMatDouble(i,1) {}


inline LaColVectorDouble::LaColVectorDouble(double *d, int m) : 
    LaGenMatDouble(d,m,1) {}


inline LaColVectorDouble::LaColVectorDouble(const LaGenMatDouble& G) : 
        LaGenMatDouble(G)
{
        assert(G.size(1)==1);

}
        
//note that vectors can be either stored columnwise, or row-wise

// this will handle the 0x0 case as well.

inline int LaColVectorDouble::size() const 
{ return LaGenMatDouble::size(0)*LaGenMatDouble::size(1); }

inline double& LaColVectorDouble::operator()(int i)
{ 
    return LaGenMatDouble::operator()(i,0);
}

inline const double& LaColVectorDouble::operator()(int i) const
{ 
    return LaGenMatDouble::operator()(i,0);
}

inline LaColVectorDouble LaColVectorDouble::operator()(const LaIndex& I)
{ 
    return LaGenMatDouble::operator()(I,LaIndex(0,0)); 
}

inline LaColVectorDouble& LaColVectorDouble::operator=(const LaGenMatDouble &A)
{
    LaGenMatDouble::copy(A);
    return *this;
}

inline LaColVectorDouble& LaColVectorDouble::operator=(double d)
{
    LaGenMatDouble::operator=(d);
    return *this;
}

#if 0
inline LaColVectorDouble& LaColVectorDouble::copy(const LaGenMatDouble &A)
{
    assert(A.size(1) == 1);   //make sure A is a column vector.
    LaGenMatDouble::copy(A);
    return *this;
}


inline LaColVectorDouble& LaColVectorDouble::ref(const LaGenMatDouble &A)
{
    assert(A.size(0) == 1 || A.size(1) == 1);
    LaGenMatDouble::ref(A);
    return *this;
}

inline LaColVectorDouble& LaColVectorDouble::inject(const LaGenMatDouble &A)
{
    assert(A.size(0) == 1 || A.size(1) == 1);
    LaGenMatDouble::inject(A);
    return *this;
}
#endif
   

inline int LaColVectorDouble::inc() const
{
    return LaGenMatDouble::inc(0);
}

inline LaIndex LaColVectorDouble::index() const
{
    return LaGenMatDouble::index(0);
}

inline int LaColVectorDouble::start() const
{
    return LaGenMatDouble::start(0);
}

inline int LaColVectorDouble::end() const
{
    return LaGenMatDouble::end(0);
}

#endif 
    // _LA_COL_VECTOR_DOUBLE_H_
