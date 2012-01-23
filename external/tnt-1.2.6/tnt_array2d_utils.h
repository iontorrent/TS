/*
*
* Template Numerical Toolkit (TNT)
*
* Mathematical and Computational Sciences Division
* National Institute of Technology,
* Gaithersburg, MD USA
*
*
* This software was developed at the National Institute of Standards and
* Technology (NIST) by employees of the Federal Government in the course
* of their official duties. Pursuant to title 17 Section 105 of the
* United States Code, this software is not subject to copyright protection
* and is in the public domain. NIST assumes no responsibility whatsoever for
* its use by other parties, and makes no guarantees, expressed or implied,
* about its quality, reliability, or any other characteristic.
*
*/


#ifndef TNT_ARRAY2D_UTILS_H
#define TNT_ARRAY2D_UTILS_H

#include <cstdlib>
#include <cassert>

namespace TNT
{


template <class T>
std::ostream& operator<<(std::ostream &s, const Array2D<T> &A)
{
    int M=A.dim1();
    int N=A.dim2();

    s << M << " " << N << "\n";

    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            s << A[i][j] << " ";
        }
        s << "\n";
    }


    return s;
}

template <class T>
std::istream& operator>>(std::istream &s, Array2D<T> &A)
{

    int M, N;

    s >> M >> N;

	Array2D<T> B(M,N);

    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
        {
            s >>  B[i][j];
        }

	A = B;
    return s;
}


template <class T>
Array2D<T> operator+(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] + B[i][j];
		}
		return C;
	}
}

template <class T>
Array2D<T> operator-(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] - B[i][j];
		}
		return C;
	}
}


template <class T>
Array2D<T> operator*(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] * B[i][j];
		}
		return C;
	}
}




template <class T>
Array2D<T> operator/(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] / B[i][j];
		}
		return C;
	}
}





template <class T>
Array2D<T>&  operator+=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] += B[i][j];
		}
	}
	return A;
}



template <class T>
Array2D<T>&  operator-=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] -= B[i][j];
		}
	}
	return A;
}



template <class T>
Array2D<T>&  operator*=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] *= B[i][j];
		}
	}
	return A;
}





template <class T>
Array2D<T>&  operator/=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] /= B[i][j];
		}
	}
	return A;
}

/**
    Matrix Multiply:  compute C = A*B, where C[i][j]
    is the dot-product of row i of A and column j of B.


    @param A an (m x n) array
    @param B an (n x k) array
    @return the (m x k) array A*B, or a null array (0x0)
        if the matrices are non-conformant (i.e. the number
        of columns of A are different than the number of rows of B.)


*/
template <class T>
Array2D<T> matmult(const Array2D<T> &A, const Array2D<T> &B)
{
    if (A.dim2() != B.dim1())
        return Array2D<T>();

    int M = A.dim1();
    int N = A.dim2();
    int K = B.dim2();

    Array2D<T> C(M,K);

    for (int i=0; i<M; i++)
        for (int j=0; j<K; j++)
        {
            T sum = 0;

            for (int k=0; k<N; k++)
                sum += A[i][k] * B [k][j];

            C[i][j] = sum;
        }

    return C;

}

template <class T>
Array1D<T> matmult(const Array2D<T> &A, const Array1D<T> &B)
{
    if (A.dim2() != B.dim1())
        return Array1D<T>();

    int M = A.dim1();
    int N = A.dim2();

    Array1D<T> C(M);

    for (int i=0; i<M; i++)
	{
        T sum = 0;
        for (int j=0; j<N; j++)
        {
            sum += A[i][j] * B [j];
        }

        C[i] = sum;
	}

    return C;
}

template <class T>
Array2D<T> transpose(const Array2D<T> &A)
{
    int M = A.dim1();
    int N = A.dim2();

    Array2D<T> C(N,M);

    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
        {
            C[j][i] = A[i][j];
        }

    return C;
}

/**
	multiples the source row by the scalar mult and adds it to the dest row
**/
template <class T>
void rowadd(Array2D<T> &A,int source,int dest,T mult)
{
    int N = A.dim2();

	for (int j=0; j<N; j++)
    {
        A[dest][j] += mult * A[source][j];
    }
}

/**
	multiplies the specified row by a scalar
**/
template <class T>
void rowmult(Array2D<T> &A,int row,T mult)
{
    int N = A.dim2();

	for (int j=0; j<N; j++)
    {
        A[row][j] = mult * A[row][j];
    }
}


/**
	performs a basic matrix inversion...must be a square matrix
**/
template <class T>
Array2D<T> invert(const Array2D<T> &A)
{
    int M = A.dim1();
    int N = A.dim2();

	Array2D<T> ACopy = A.copy();
    Array2D<T> C(M,N);

    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
        {
			if (j == i)
				C[i][j] = (T)1.0;
			else
	            C[i][j] = (T)0.0;
        }

    for (int col=0; col<N; col++)
	{
		// make sure A[col][col] is non-zero
		if (ACopy[col][col] == (T)0.0)
		{
			// find a non-zero element in the same column
			T max = abs(ACopy[0][col]);
			int mrow = 0;
			for (int row=1;row < M; row++)
			{
				if (abs(ACopy[row][col]) > max)
				{
					max = abs(ACopy[row][col]);
					mrow = row;
				}
			}

			// hopefully we found something
			if (max != (T)0.0)
			{
				rowadd(ACopy,mrow,col,(T)(1.0/(double)max));
				rowadd(C,mrow,col,(T)(1.0/(double)max));
			}
		}

		// if this is still non-zero, then the whole column is zero..skip it
		if (ACopy[col][col] == (T)0.0)
			continue;

		// normalize this row
		if (ACopy[col][col] != (T)1.0)
		{
			T mult = (T)(1.0/(double)(ACopy[col][col]));
			rowmult(ACopy,col,mult);
			rowmult(C,col,mult);
		}

		// clear out all other rows at this column position
		for (int row=0;row < M; row++)
		{
			if (row == col)
				continue;

			T mult = -ACopy[row][col];
			rowadd(ACopy,col,row,mult);
			rowadd(C,col,row,mult);
		}
	}

    return C;
}

} // namespace TNT

#endif
