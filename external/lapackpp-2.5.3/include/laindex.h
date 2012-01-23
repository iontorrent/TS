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
 * @brief Matrix index class LaIndex
 */

#ifndef _LA_INDEX_H_
#define _LA_INDEX_H_

#include <iostream>
#include "arch.h"

/** @brief Matrix index class.
 *
 * This index class is used to adress submatrices of other matrices,
 * without actually copying any matrix elements in memory.
 *
 * Note that we name it "LaIndex" to avoid confusion with the
 * "index()" string function in C, or other generic Index()
 * functions.
 */
class DLLIMPORT LaIndex
{
   private:
      friend class LaGenMatDouble;
      friend class LaGenMatFloat;
      friend class LaGenMatInt;
      friend class LaGenMatLongInt;
      friend class LaGenMatComplex;
      int start_;
      int inc_;
      int end_;
   public:
      /** Construct a null index. Start, Increment, and End are
       * zero. null() will return true. 
       *
       * When used for addressing in one of the matrix classes, a
       * null index like this is equivalent to adressing "all
       * elements". To get the first row of a matrix \c A, you
       * could therefore write <tt>A( LaIndex(0), LaIndex()
       * )</tt>. This corresponds to the MATLAB notation of a
       * single <tt>:</tt> (colon).
       */
      inline LaIndex()
	 : start_(0)
	 , inc_(0)
	 , end_(0)
      {}
      
      /** Construct an index that points only to one element, specified
       * by the given argument.
       * 
       * Start and End is the given argument; Increment is unity but has
       * no effect. */
      inline LaIndex(int start) 
	 : start_(start)
	 , inc_(1)
	 , end_(start)
      {}

      /** Construct an index that points to a range of elements,
       * starting from the first argument and ending at the second
       * argument. The element at the starting index and the
       * element at the ending index are both included. This
       * corresponds to the MATLAB notation <tt>start:end</tt>.
       *
       * \a start and \a end are given by the two
       * arguments. Increment is unity. \a end has to be greater
       * or equal to \a start .*/
      LaIndex(int start, int end);

      /** Construct an index that points to a range of elements
       * with a given increment. The element at the starting index
       * is included; the element at the ending index is included
       * if its difference to the starting index is an integer
       * multiple of the increment.  This corresponds to the
       * MATLAB notation <tt>start:increment:end</tt>, but note
       * the different order of arguments in this constructor.
       *
       * The \a increment has to be nonzero. The \a end has to be
       * greater or equal to \a start if \a increment is
       * positive. If \a increment is negative, then \a end has to
       * be lesser or equal to \a start . Please watch out with
       * negative increments: The respective matrix functions may
       * or may not be correctly implemented for negative
       * increments. If in doubt, use only positive increments.
       *
       * Start, End, and Increment are given by the three
       * arguments. (We apologize for the confusing order of the
       * arguments.) */
      LaIndex(int start, int end, int increment);

      /** Copy constructor */
      inline LaIndex(const LaIndex &s)
	 : start_(s.start_)
	 , inc_(s.inc_)
	 , end_(s.end_)
      {}

   protected:
      // must have multply defined start(), inc() and end() member
      // functions for both const and non-const objects because
      // compiler complains in LaVector*.h, for example, about
      // assignment to const member.  (LaVector*.h line 112, 113,
      // 114) FIXME: Probably we should return by-value, not
      // by-reference!
      /** Returns the start index by-reference. That is, the index
       * of the first indexed element.
       *
       * DEPRECATED. Use the set() methods instead. */
      inline int& start() { return start_;}
      /** Returns the increment by-reference
       *
       * DEPRECATED. Use the set() methods instead.  */
      inline int& inc() { return inc_;}
      /** Returns the end index by-reference. That is, the index
       * of the last indexed element. (Not to be confused with the
       * index of the one-after-the-last element!)
       *
       * DEPRECATED. Use the set() methods instead.  */
      inline int& end() { return end_;}
   public:

      /** Returns the start index. That is, the index of the first
       * indexed element. */
      inline int start() const { return start_;}

      /** Returns the increment. It is guaranteed to be nonzero. */
      inline int inc() const { return inc_;}

      /** Returns the end index. That is, the index of the last
       * indexed element.
       *
       * (Not to be confused with the index of the
       * one-after-the-last element!) To be more precise, the
       * element at this ending index is being used if and only if
       * its difference to the starting index is an integer
       * multiple of the increment. */
      inline int end() const { return end_;}

      /** Returns the number of elements that are indexed by this
       * object. */
      inline int length() const { return ((end()-start())/inc() + 1);}

      /** Returns true if this is a null index which cannot be
       * used for indexing. */
      inline bool null() const { return (start() == 0 && 
					 inc() == 0 && end() == 0);}

   protected:
      /** DEPRECATED. Changes this index so that it points to a
       * range of elements, starting from the first argument and
       * ending at the second argument.
       *
       * Start and End are given by the two arguments. Increment
       * will be set to unity. 
       *
       * Deprecated. Use the set() method instead. Defining such
       * an operator() which is an assignment instead of a value
       * retrieval makes the indexing operations quite hard to
       * read.  Instead, use the set() method.
       */
      inline LaIndex& operator()(int start, int end){
	 start_=start; inc_=1; end_=end; return *this;}
   public:

      /** Changes this index so that it points to a range of
       * elements, starting from the first argument and ending at
       * the second argument. The element at the starting index
       * and the element at the ending index are both
       * included. This corresponds to the MATLAB notation
       * <tt>start:end</tt>.
       *
       * Start and End are given by the two arguments. Increment
       * will be set to unity. */
      inline LaIndex& set(int start, int end) {
	 start_=start; inc_=1; end_=end; return *this;}

      /** Changes this index so that it points to a range of
       * elements with a given increment. The element at the
       * starting index is included; the element at the ending
       * index is included if its difference to the starting index
       * is an integer multiple of the increment. The increment
       * argument has to be nonzero. This corresponds to the
       * MATLAB notation <tt>start:increment:end</tt>, but note
       * the different order of arguments in this constructor.
       *
       * The \a increment has to be nonzero. The \a end has to be
       * greater or equal to \a start if \a increment is
       * positive. If \a increment is negative, then \a end has to
       * be lesser or equal to \a start . Please watch out with
       * negative increments: The respective matrix functions may
       * or may not be correctly implemented for negative
       * increments. If in doubt, use only positive increments.
       *
       * Start, End, and Increment are given by the three
       * arguments. (We apologize for the confusing order of the
       * arguments, but this follows the three-valued
       * constructor.) */
      LaIndex& set(int start, int end, int increment);

      /** Shifts this index by the given argument. 
       *
       * The given argument is added to the Start and End
       * indices. Increment is unchanged.  */
      inline LaIndex& operator+=(int i) {
	 start_+=i; end_+=i; return *this;}

      /** Returns a new index, shifted by the given argument.
       *
       * The given argument is added to the Start and End
       * indices. Increment is unchanged. A new object is returned. */
      inline LaIndex operator+(int i)
      {
	  LaIndex r(*this);
	  r.start_ += i;
	  r.end_   += i;
	  return r;
      }

      /** Copy assignment */
      inline LaIndex& operator=(const LaIndex& i) {
	 start_=i.start_; inc_=i.inc_; end_=i.end_; 
	 return *this;}

      /** Equality predicate */
      inline bool operator==(const LaIndex& s) {
	 return start_ == s.start_ && inc_ == s.inc_ && end_ == s.end_;
      }
      /** Inequality predicate. (New in lapackpp-2.4.5) */
      inline bool operator!=(const LaIndex& s) {
	 return !operator==(s);
      }
};

inline std::ostream& operator<<(std::ostream& s, const LaIndex& i)
{
    s << "(" << i.start() << ":" << i.inc() << ":" << i.end() << ")";

    return s;
}

#endif  
//  LA_INDEX_H_

