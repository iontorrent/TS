/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * DCT0Finder.cpp
 *
 *  Created on: Nov 26, 2012
 *      Author: ionadmin
 */
#include "stdio.h"
#include "unistd.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "DCT0Finder.h"

static int GetSlopeAndOffset(double *trace, uint32_t startIdx, uint32_t endIdx, double *slope, double *offset, FILE *logfp);
double FindT0Specific(double *trace, uint32_t trace_length, uint32_t T0Initial,FILE *log_fp);

uint32_t DCT0Finder(double *trace, uint32_t trace_length, FILE *logfp)
{
	uint32_t lidx2=0;
	double traceDv[trace_length];
	int32_t rc = -1;
	int32_t T0Initial=-1;
	double T0Specific=0;

	memset(traceDv,0,sizeof(traceDv));
	// turn the trace into a derivative
	for(lidx2=trace_length-1;lidx2>0;lidx2--)
	{
		traceDv[lidx2] = trace[lidx2] - trace[lidx2-1];
	}
	traceDv[0] = 0;
	if(logfp != NULL)
	{
		fprintf(logfp,"  derivative trace:\n\n");
		for(uint32_t i=0;i<100 && i < trace_length;i+=10)
		{
			fprintf(logfp," (%d)  ",i);
			for(int j=0;j<10 && ((i+j) < trace_length);j++)
				fprintf(logfp," %.0lf",traceDv[i+j]);
			fprintf(logfp,"\n");
		}
		fprintf(logfp,"\n\n");
	}

	// now, find t0Estimate
	for(lidx2=1;lidx2<(trace_length-4);lidx2++)
	{
		if ((fabs(traceDv[lidx2]) > 10) &&
			(fabs(traceDv[lidx2+1]) > 10) &&
			(fabs(traceDv[lidx2+2]) > 10))
		{
			// this is the spot..
			T0Initial = lidx2;
			break;
		}
	}

	if(logfp != NULL)
		fprintf(logfp,"\n  Initial T0 guess=%d\n",T0Initial);

	T0Specific = FindT0Specific(trace,trace_length,T0Initial,logfp);

	if((T0Initial > 4) && (T0Initial < (int32_t)(trace_length-4)) &&
	   (T0Specific < (T0Initial+12)) && (T0Specific > (T0Initial-12)))
	{
		// we found a point pretty close to the original one...
		rc = T0Specific;
		if(logfp != NULL)
			fprintf(logfp,"using new T0Specific=%.2lf T0Guess=%df\n",T0Specific,T0Initial);
	}
	else
	{
		if(logfp != NULL)
			fprintf(logfp,"Rejecting the new T0Specific=%.2lf\n",T0Specific);
	}

//	if(rc > 8)
//		rc -= 8; // make sure we start a good bit before t0

	return rc;
}

double FindT0Specific(double *trace, uint32_t trace_length, uint32_t T0InitialGuess, FILE *logfp)
{
	double T0Max = 0;
	double T0Specific=0;
	double slope1=0,offset1=0;
	double slope2=0,offset2=0;

	uint32_t rhswidth,xguess;

	for(rhswidth=06;rhswidth<12;rhswidth++)
	{
		for(xguess=T0InitialGuess-4;xguess<T0InitialGuess+4 && ((xguess+rhswidth)<trace_length);xguess++)
		{
			GetSlopeAndOffset(trace,0,xguess,&slope1,&offset1,logfp);
			GetSlopeAndOffset(trace,xguess+1,xguess+rhswidth,&slope2,&offset2,logfp);

//			if(logfp != NULL)
//			{
//				fprintf(logfp,"  slope1=%.2lf offset1=%.2lf\n",slope1,offset1);
//				fprintf(logfp,"  slope2=%.2lf offset2=%.2lf\n",slope2,offset2);
//			}
			if(slope1 != 0 && offset1 != 0 &&
			   slope2 != 0 && offset2 != 0 &&
			   slope1 != slope2)
			{
				T0Specific = -(offset1-offset2)/(slope1-slope2);
				if(logfp != NULL)
					fprintf(logfp,"  Try x=%d rhsw=%d T0=%.2lf\n",xguess,rhswidth,T0Specific);

				if(T0Specific > T0Max)
					T0Max = T0Specific;
			}

		}
	}
	return T0Max;
}

int GetSlopeAndOffset(double *trace, uint32_t startIdx, uint32_t endIdx, double *slope, double *offset, FILE *logfp)
{
	double sumTime=0;
	double sumTimesq=0;
	double sumTimexVal=0;
	double sumVal=0;
	double num1,num2,den;
	uint32_t idx;

	*slope=0;
	*offset=0;

	for(idx=startIdx;idx<endIdx;idx++)
	{
		sumTime += (double)idx;
		sumTimesq += (double)(idx*idx);
		sumVal += trace[idx];
		sumTimexVal += ((double)idx)*trace[idx];
	}

//	if(logfp != NULL)
//	{
//		fprintf(logfp,"    sumTime=%.2lf sumTimesq=%.2lf sumVal=%.2lf sumTimexVal=%.2lf\n",sumTime,sumTimesq,sumVal,sumTimexVal);
//	}
	num1 = ((sumTimesq*sumVal) - (sumTime*sumTimexVal));
	num2 = (((endIdx-startIdx)*sumTimexVal) - (sumTime*sumVal));
	den =  (((endIdx-startIdx)*sumTimesq) - (sumTime*sumTime));
	if(den != 0)
	{
		*offset = num1/den;
		*slope = num2/den;
	}
	return 0;
}


