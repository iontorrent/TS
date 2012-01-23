// Copyright (C) 2010 NICTA (www.nicta.com.au)
// Copyright (C) 2010 Conrad Sanderson
// 
// This file is part of the Armadillo C++ library.
// It is provided without any warranty of fitness
// for any purpose. You can redistribute this file
// and/or modify it under the terms of the GNU
// Lesser General Public License (LGPL) as published
// by the Free Software Foundation, either version 3
// of the License or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)



//! \addtogroup span
//! @{



enum span_helper
  {
  span_whole_vector
  };



class span
  {
  public:
  
  static const span_helper all = span_whole_vector;
  
  inline
  span(const u32 in_a, const u32 in_b)
    : a(in_a)
    , b(in_b)
    {
    }
  
  const u32 a;
  const u32 b;
  };



//! @}
