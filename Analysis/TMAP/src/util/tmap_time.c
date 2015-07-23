/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/* The MIT License

   Copyright (c) 2008 Genome Research Ltd (GRL).

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include "tmap_time.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

static double ONE_MILLIONTH = 1e-6;

double 
tmap_time_cputime()
{
  struct rusage r;
  getrusage(RUSAGE_SELF, &r);
  return r.ru_utime.tv_sec + r.ru_stime.tv_sec + ONE_MILLIONTH * (r.ru_utime.tv_usec + r.ru_stime.tv_usec);
}

double 
tmap_time_realtime()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  if (gettimeofday(&tp, NULL) != 0)
  {
    printf ("gettimeofday failed, errno is %d : %s\n", errno, strerror (errno));
    exit (1);
  }
  //printf ("\ntime of day is %ld sec : %ld usec\n", tp.tv_sec, tp.tv_usec);
  return tp.tv_sec + ONE_MILLIONTH*tp.tv_usec;
}
