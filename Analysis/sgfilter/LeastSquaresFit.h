/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LEASTSQUARESFIT_H
#define LEASTSQUARESFIT_H

#pragma once

#include "tnt.h"

class LeastSquaresFit
{
public:

	/// <summary>
    /// Solves the matrix equation A x B = C, where A and C are knowns and B is unknown.
    /// </summary>
    /// <param name="Amat">A must either be a square matrix, or have more rows than columns</param>
    /// <param name="Cmat">C must be a single colum matrix with as many rows as A</param>
    /// <returns>The GeneralMatrix B is returned.  B will have as many rows as A had columns,
    /// and will have only a single column</returns>
	static TNT::Array1D<float> Solve(TNT::Array2D<float> &Amat,TNT::Array1D<float> &Cmat)
    {
		if (Amat.dim2() > Amat.dim1())
			return TNT::Array1D<float>();

		if (Amat.dim1() != Cmat.dim1())
			return TNT::Array1D<float>();

		TNT::Array2D<float> ATrans = TNT::transpose(Amat);
		TNT::Array2D<float> ATA = TNT::matmult(ATrans,Amat);
		TNT::Array2D<float> ATAInv = TNT::invert(ATA);
		TNT::Array2D<float> ATAInvTimesATrans = TNT::matmult(ATAInv,ATrans);
		TNT::Array1D<float> ret = TNT::matmult(ATAInvTimesATrans,Cmat);

		return ret;
    }

	/// <summary>
	/// Prepares the (A^TA)^-1A^T matrix used in the least squares solver.
	/// can be used in cases where most of the preparation can be re-used over again to save
	/// processing time
    /// </summary>
    /// <param name="Amat">A must either be a square matrix, or have more rows than columns</param>
	/// <returns>(A^T*A)^(-1) * A^T</returns>
	static TNT::Array2D<float> PartialSolve(TNT::Array2D<float> &Amat)
    {
		if (Amat.dim2() > Amat.dim1())
			return TNT::Array2D<float>();

		TNT::Array2D<float> ATrans = TNT::transpose(Amat);
		TNT::Array2D<float> ATA = TNT::matmult(ATrans,Amat);
		TNT::Array2D<float> ATAInv = TNT::invert(ATA);
		TNT::Array2D<float> ATAInvTimesATrans = TNT::matmult(ATAInv,ATrans);

		return ATAInvTimesATrans;
    }
};

#endif //LEASTSQUARESFIT_H
