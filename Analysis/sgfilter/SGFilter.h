/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SGFILTER_H
#define SGFILTER_H

#pragma once

// #include "Utility.h"
#include "LeastSquaresFit.h"
#include "RingBuffer.h"

#include "dbgmem.h"

class SGFilter {
	private:
		int nSpread;                   // number of points to use on each side of filtered pt
		int nCoeffs;                   // number of polynomial coefficients
		int nPoints;                   // total number of points in the filter
		TNT::Array2D<float> sgCoeffs;  // matrix of filter coefficients

		void CreateFilterMatrix(int spread,int coeffs) {
			if (spread < 1)
			{
				// spread must be at least 1
				spread = 1;
			}

			if (coeffs < 1)
			{
				// number of coefficients must be at least 1
				coeffs = 1;
			}
    
			nSpread = spread;
			nCoeffs = coeffs;
			nPoints = spread * 2 + 1;

			if (nPoints <= nCoeffs)
			{
				// spread is too small for number of coefficients
				nPoints = nCoeffs | 0x1;
				nSpread = (nPoints - 1) >> 1;
			}

			// matrix of X values raised to 0th, 1st, 2nd, etc.,.. orders
			TNT::Array2D<float> AMatrix(nPoints, nCoeffs);
    
			// generate the X^N values
			for (int x = -nSpread, row = 0; x <= nSpread; x++, row++)
			{
				float fx = (float)x;

				for (int col = 0; col < nCoeffs; col++)
				{
					AMatrix[row][col] = pow(fx,(float)col);
				}
			}

			// the algorithm to compute the SG coefficients is exactly the same
			// as that required to solve a polynomial least-squares fit
			sgCoeffs = LeastSquaresFit::PartialSolve(AMatrix);

			// sgCoeffs now contains a 2D matrix.  Each row of the matrix is the filter coefficients
			// needed to compute the filtered representation of a set of data points and a particular
			// derivative order.
			// i.e., row 0 is the coefficients required to compute the filtered output without taking any derivative
			//       row 1 computes the first derivative of the input data
			//       row 2 the second order...and so on....
		}


	public:
		SGFilter(): nSpread(0), nCoeffs(0), nPoints(0) {};

		SGFilter(int spread,int coeffs)
		{
			SetFilterParameters(spread,coeffs);
		}

		// sets the filter's spread and number of coefficients, recomputes the filter matrix
		void SetFilterParameters(int spread,int coeffs)
		{
			CreateFilterMatrix(spread,coeffs);		// compute new one
		}

		/// <summary>
		/// Filters an array of floats.  If order is non-zero, the returned data is some derivative of
		/// the input data.  (i.e., order = 1 generates the first derivative, 2 the second, etc.,...)
		/// The array is extended on each end by spread points, and the extended points are just replications
		/// of the input data at the start and end of the array.
		///
		/// TODO: *Create version that takes a pointer to points seperated by an arbitrary index (a data-point stride)
		///       useful for processing data directly from a RawImage.
		///		  *Also can create a version that can process data in-place.  Would require pulling nPoints of data into
		///       a temporary ring-buffer and then rolling each new point through the buffer.
		/// </summary>
		/// <param name="data">array of floating-point numbers to be filtered</param>
		/// <param name="length">number of points at (data) to filter</param>
		/// <param name="order">which derivative to take of the input data (0 is no derivative)</param>
		/// <returns>The array of filtered data</returns>
		float *FilterData(float *data,int length, int order)
		{
			float *ret = (float *)malloc(sizeof(float) * length);

			FilterDataImpl(data,ret,length,order);

			return ret;
		}

#ifdef NOTUSED
		//TODO: 
		void FilterDataImpl(RawData_t *data, int length, int order) {
			RingBuffer<RawData_t>myBuf;
			int spread = 2*nSpread;

			if (order >= sgCoeffs.dim1())
			{
				// throw new Exception("Can't generate a derivative of order " + order.ToString() + " with " + nCoeffs + " coefficients");
				order = sgCoeffs.dim1() - 1;
			}
	
			int npts = length;
			int pt;
			for (pt = 0; pt < npts; pt++)
			{
				float val = 0;
				for (int coeff = 0, dndx = pt - nSpread; coeff < sgCoeffs.dim2(); coeff++, dndx++)
				{
					if (dndx < 0)
						val += data[0] * sgCoeffs[order][coeff];
					else if (dndx >= npts)
						val += data[npts - 1] * sgCoeffs[order][coeff];
					else
						val += data[dndx] * sgCoeffs[order][coeff];
				}
				if (!myBuf.insert((RawData_t)floor(val))) {std::cerr << "false\n";}
				if (myBuf.remaining() < RingBuffer<RawData_t>::ARRAY_SIZE - spread) {
					data[pt - spread] = myBuf.remove();
				}
				//ret[pt] = val;
			}
			pt -= spread;
			while (!myBuf.isEmpty()) {
				data[pt] = myBuf.remove();
				pt++;
			}
		}
#endif /* NOTUSED */
		/// Same as the other FilterData, except it takes an output array to put the result into
		float *FilterDataImpl(float *data,float *dest,int length, int order)
		{
			if (order >= sgCoeffs.dim1())
			{
				// throw new Exception("Can't generate a derivative of order " + order.ToString() + " with " + nCoeffs + " coefficients");
				order = sgCoeffs.dim1() - 1;
			}

			int npts = length;
			float *ret = dest;

			for (int pt = 0; pt < npts; pt++)
			{
				float val = 0;
				for (int coeff = 0, dndx = pt - nSpread; coeff < sgCoeffs.dim2(); coeff++, dndx++)
				{
					if (dndx < 0)
						val += data[0] * sgCoeffs[order][coeff];
					else if (dndx >= npts)
						val += data[npts - 1] * sgCoeffs[order][coeff];
					else
						val += data[dndx] * sgCoeffs[order][coeff];
				}

				ret[pt] = val;
			}

			return ret;
		}

		/// Same as the other FilterData, except it takes an output array to put the result into
		double *FilterDataImpl(double *data,double *dest,int length, int order)
		{
			if (order >= sgCoeffs.dim1())
			{
				// throw new Exception("Can't generate a derivative of order " + order.ToString() + " with " + nCoeffs + " coefficients");
				order = sgCoeffs.dim1() - 1;
			}

			int npts = length;
			double *ret = dest;

			for (int pt = 0; pt < npts; pt++)
			{
				double val = 0;
				for (int coeff = 0, dndx = pt - nSpread; coeff < sgCoeffs.dim2(); coeff++, dndx++)
				{
					if (dndx < 0)
						val += data[0] * sgCoeffs[order][coeff];
					else if (dndx >= npts)
						val += data[npts - 1] * sgCoeffs[order][coeff];
					else
						val += data[dndx] * sgCoeffs[order][coeff];
				}

				ret[pt] = val;
			}

			return ret;
		}
		bool TEST_valid() {
			return (bool)(static_cast<bool>(nSpread) | true);
		}
		~SGFilter(void)
		{
		}
};
#endif // SGFILTER_H
