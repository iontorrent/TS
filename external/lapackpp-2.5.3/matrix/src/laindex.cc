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

#include "laindex.h"
#include "laexcp.h"

LaIndex::LaIndex(int start, int end)
   : start_(start)
     , inc_(1)
     , end_(end)
{
   if (!(start <= end))
      throw LaException("LaIndex(int,int)", "assertion (start <= end) failed");
}

LaIndex::LaIndex(int start, int end, int increment)
   : start_(start)
   , inc_(increment)
   , end_(end)
{ 
   if (!(increment != 0))
      throw LaException("LaIndex(int,int,int)", "assertion (increment != 0) failed");
   if (increment > 0)
   {
      if (!(start <= end))
	 throw LaException("LaIndex(int,int,int)", "assertion (start <= end) failed");
   }
   else
   {
      if (!(start >= end))
	 throw LaException("LaIndex(int,int,int)", "assertion (start >= end) failed");
   }
}


LaIndex& LaIndex::set(int start, int end, int increment) 
{
   if (!(increment != 0))
      throw LaException("LaIndex::set(int,int,int)", "assertion (increment != 0) failed");
   if (increment > 0)
   {
      if (!(start <= end))
	 throw LaException("LaIndex::set(int,int,int)", "assertion (start <= end) failed");
   }
   else
   {
      if (!(start >= end))
	 throw LaException("LaIndex::set(int,int,int)", "assertion (start >= end) failed");
   }

   start_=start;
   inc_=increment;
   end_=end; 

   return *this;
}
