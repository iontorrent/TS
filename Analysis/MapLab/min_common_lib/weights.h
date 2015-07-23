/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __weights_h__
#define __weights_h__

#include "rerror.h"
#include "resource.h"
#include "ncbi_weight_matrix.h"
#include <algorithm>
#include <string.h>
#include <float.h>
#include <limits.h>
#define DESCR_SZ 512

template <class T, int dim> class WEIGHTS
{

public:

    T       mx [dim][dim];
    float   gip;
    float   gep;
    int     minmx;
    int     maxmx;
    int     offs;
    int     dimv;
    char    descr[DESCR_SZ];
    bool    ok;

            WEIGHTS ();
            WEIGHTS (int mat, int mis, int gip, int gep); // create diagonal matrix            
            WEIGHTS (int lev, float gip, float gep, bool byte_range = false); // create diagonal matrix
            WEIGHTS (const char* name, const char* fname, float gip, float gep); // load from file
            WEIGHTS (const char* name, const double* weights, float gip, float gep); // to allow load from database
            WEIGHTS (const char* name, const int* weights, float gip, float gep); // to allow load from database

            ~WEIGHTS (){}

    char     get_char (int x, int y);
    void     make_unsigned ();
    void     set_diag (int mat, int mis, int gip, int gep); // set to diagonal matrix
};

// create unitary matrix and set default gip/gap values
template <class T, int dim>
WEIGHTS<T, dim>::WEIGHTS ()
{
    maxmx = 1;
    minmx = -1;
    offs = 0;
    dimv = dim;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            if (i == j)
                mx[i][j] = maxmx;
            else
                mx[i][j] = minmx;

    gip = 5.;
    gep = 1.;
    ok = true;
    sprintf (descr, "default diagonal matrix, maxmx = %d, minmx = %d; gip_adj = %g, gep_adj = %g", maxmx, minmx, gip, gep);
}

template <class T, int dim>
WEIGHTS<T, dim>::WEIGHTS (int mat, int mis, int gip, int gep)
{
    maxmx = std::max (mis, mat);
    minmx = std::min (mis, mat);
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            if (i == j)
                mx[i][j] = mat;
            else
                mx[i][j] = mis;

    WEIGHTS<T, dim>::gip = gip;
    WEIGHTS<T, dim>::gep = gep;
    ok = true;
    sprintf (descr, "default diagonal matrix, maxmx = %d, minmx = %d; gip_adj = %g, gep_adj = %g", maxmx, minmx, gip, gep);
}

template <class T, int dim>
void WEIGHTS<T, dim>::set_diag (int mat, int mis, int gip, int gep) // reset to diagonal matrix
{
    maxmx = std::max (mis, mat);
    minmx = std::min (mis, mat);
    offs = 0;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            if (i == j)
                mx[i][j] = mat;
            else
                mx[i][j] = mis;

    WEIGHTS<T, dim>::gip = gip;
    WEIGHTS<T, dim>::gep = gep;
    ok = true;
    sprintf (descr, "diagonal matrix, maxmx = %d, minmx = %d; gip_adj = %d, gep_adj = %d", maxmx, minmx, gip, gep);
}


/*
create diagonal matrix and set gip/gap values
lev - similarity level (0-100%)
gip - gap init penalty in mismatches
gep - gap extension penalty in mismatches
NOTE: 2*gip + N*gep should ALWAYS be > - N * (worst mismatch penalty) to avoid stairstep alignment
*/
template <class T, int dim>
WEIGHTS<T, dim>::WEIGHTS (int lev, float gip, float gep, bool byte_range)
{
    if (byte_range)
    {
        // adjust the lev to lowest possible integer ratio in the range 0..10
        int best_nominator = 1;
        int best_denominator = 1;
        float best_distance = FLT_MAX;

        for (int nominator = 1; nominator < 10; nominator ++)
            for (int denominator = 1; denominator < 10; denominator ++)
            {
                float distance = float (nominator) / denominator - float (lev) / 100;
                if (distance < 0) distance = -distance;
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_denominator = denominator;
                    best_nominator = nominator;
                }
                else if (distance == best_distance && nominator < best_nominator)
                {
                    best_denominator = denominator;
                    best_nominator = nominator;
                }
            }
        maxmx = best_denominator - best_nominator;
        minmx = -best_nominator;

    }
    else
    {
        maxmx = 100 - lev;
        minmx = -lev;
    }
    offs = 0;
    dimv = dim;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            if (i == j)
                mx[i][j] = (T) maxmx;
            else
                mx[i][j] = (T) minmx;

    WEIGHTS<T, dim>::gip = gip * -minmx;
    WEIGHTS<T, dim>::gep = gep * -minmx;
    sprintf (descr, "custom diagonal matrix, maxmx = %d, minmx = %d; gip_adj = %g, gep_adj = %g", maxmx, minmx, WEIGHTS<T, dim>::gip, WEIGHTS<T, dim>::gep);
    //clog << descr << endl;
    ok = true;
}

/*
load matrix from file (fname) and set gip/gap values
gip - gap init penalty in mismatches
gep - gap extension penalty in mismatches
NOTE: 2*gip + N*gep should ALWAYS be > - N * (worst mismatch penalty) to avoid stairstep alignment
*/
template <class T, int dim>
WEIGHTS<T, dim>::WEIGHTS (const char* name, const char* fname, float gip, float gep)
{
    int ii = 0;
    int jj = 0;

    MemWrapper <char> alpha_buf (dim);
    MemWrapper <T> value_buf (dim*dim);
    int read_dim = readNcbiMatrix (fname, dim, alpha_buf, value_buf);
    if (read_dim != dim) ers << "Alphabet is too small in " << fname << ThrowEx(BadMatrixFormat);

    maxmx = INT_MIN;
    minmx = INT_MAX;
    for (ii = 0; ii < dim; ii++)
    {
        for (jj = 0; jj < dim; jj++)
        {
            T tmp = value_buf [ii * dim + jj];
            mx [ii][jj] = tmp;
            maxmx = std::max (maxmx, (int) tmp);
            minmx = std::min (minmx, (int) tmp);
            if (ii > jj && mx[ii][jj] != mx[jj][ii])
                ers << "Symmetry violation in " << fname << " (pos " << ii << "," << jj << ")" << ThrowEx (BadMatrixFormat);
        }
    }

    this->gip = (gip * -minmx);
    this->gep = (gep * -minmx);
    sprintf (descr, "%s matrix, maxmx = %d, minmx = %d; gip_adj = %g, gep_adj = %g", name, maxmx, minmx, WEIGHTS<T, dim>::gip, WEIGHTS<T, dim>::gep);
    ok = true;
}

/*
load matrix from array of doubles (to allow interfacing with database) and set gip/gap values
gip - gap init penalty in mismatches
gep - gap extension penalty in mismatches
NOTE: 2*gip + N*gep should ALWAYS be > - N * (worst mismatch penalty) to avoid stairstep alignment
*/
template <class T, int dim>
WEIGHTS<T, dim>::WEIGHTS (const char* name, const double* weights, float gip, float gep) // to allow load from database
{
    int i, j;
    maxmx = INT_MIN;
    minmx = INT_MAX;
    offs = 0;
    dimv = dim;

    int cur_no = 0;
    for (i = 0; i < dim; i ++)
    {
        for (j = 0; j < dim; j++)
            mx[i][j] = (T) weights [cur_no++];
    }

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
        {
            maxmx = std::max (maxmx, (int) mx[i][j]);
            minmx = std::min (minmx, (int) mx[i][j]);
            if (mx[i][j] != mx[j][i]) ers << "Symmetry violation" << ThrowEx (BadMatrixFormat);
        }

    WEIGHTS<T, dim>::gip = (gip * -minmx);
    WEIGHTS<T, dim>::gep = (gep * -minmx);
    sprintf (descr, "%s matrix, maxmx = %d, minmx = %d; gip_adj = %g, gep_adj = %g", name, maxmx, minmx, WEIGHTS<T, dim>::gip, WEIGHTS<T, dim>::gep);
    ok = true;
}

template <class T, int dim>
WEIGHTS<T, dim>::WEIGHTS (const char* name, const int* weights, float gip, float gep) // to allow load from database
{
    int i, j;
    maxmx = INT_MIN;
    minmx = INT_MAX;
    offs = 0;
    dimv = dim;

    int cur_no = 0;
    for (i = 0; i < dim; i ++)
    {
        for (j = 0; j < dim; j++)
            mx[i][j] = (T) weights [cur_no++];
    }

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
        {
            maxmx = std::max (maxmx, (int) mx[i][j]);
            minmx = std::min (minmx, (int) mx[i][j]);
            if (mx[i][j] != mx[j][i]) ers << "Symmetry violation" << ThrowEx (BadMatrixFormat);
        }

    WEIGHTS<T, dim>::gip = (gip * -minmx);
    WEIGHTS<T, dim>::gep = (gep * -minmx);
    sprintf (descr, "%s matrix, maxmx = %d, minmx = %d; gip_adj = %g, gep_adj = %g", name, maxmx, minmx, WEIGHTS<T, dim>::gip, WEIGHTS<T, dim>::gep);
    ok = true;
}

//returns symbolic character corresponding to match level
template <class T, int dim>
char WEIGHTS<T, dim>::get_char (int x, int y)
{
    char c = ' ';

    if (x == y)
        c = '*';
    else if ((int)mx[x][y] > maxmx / 2)
        c = '|';
    else if ((int)mx[x][y] > maxmx / 4)
        c = ':';
    else if ((int)mx[x][y] > 0)
        c = '.';

    return c;
}

template <class T, int dim>
void WEIGHTS<T, dim>::make_unsigned ()
{
    if (minmx >= 0)
        return;

    offs = -minmx;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            mx[i][j] += (T)offs;
}

typedef WEIGHTS<int, 24> WMatrix;

#endif
