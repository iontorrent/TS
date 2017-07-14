/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cinttypes>

#include "ByteSwapUtils.h"
#include "LSRowImageProcessor.h"

#define N_CORRECTION_GROUPS 8
#define SKIP_ROWS_AT_END    10

#define PINNED_HIGH_LIMIT   16373
#define PINNED_LOW_LIMIT    10

#define LSROWIMAGE_MAGIC_VALUE    0xFF115E3A

using namespace arma;
// generate an electrical cross-talk correction from the lsrow image
// if the file pointed to by lsimg_path does not exist, the method returns NULL
// If lsimg_path does exist, then a correction is generated and a pointer to
// a ChannelXTCorrection object with all the relevant information is returned
ChannelXTCorrection *LSRowImageProcessor::GenerateCorrection(const char *lsimg_path)
{
    uint32_t magic_value;
    uint32_t file_version;
    uint32_t rows = 0;
    uint32_t cols = 0;
    int nread;

    // the lsrowimage file contains:
    // uint32 magic value (0xFF115E3A)
    // uint32 file version number (0+)
    // uint32 rows
    // uint32 columns
    // uint16[rows*columns] high-speed and low-speed reference data, in row major order
    //  ..with the first row in the image being a high-speed collected row
    //    and the next row in the image being the low-speed reference for the previous high-speed row data

    FILE *lsrowfile;

    lsrowfile = fopen(lsimg_path,"rb");

    // if we have trouble opening the file, just return NULL
    if(lsrowfile == NULL)
        return NULL;

    nread = fread(&magic_value,sizeof(int32_t),1,lsrowfile);
    if(nread != 1)
    {
        printf("Ivalid lsrowfile detected\n");
        fclose(lsrowfile);
        return NULL;
    }
    magic_value = BYTE_SWAP_4(magic_value);

    if(magic_value != LSROWIMAGE_MAGIC_VALUE)
    {
        printf("Ivalid lsrowfile detected\n");
        fclose(lsrowfile);
        return NULL;
    }

    nread = fread(&file_version,sizeof(int32_t),1,lsrowfile);
    if(nread != 1)
    {
        printf("Ivalid lsrowfile detected\n");
        fclose(lsrowfile);
        return NULL;
    }
    file_version = BYTE_SWAP_4(file_version);

    if(file_version != 0)
    {
        printf("Unsupported lsrowimage file version\n");
        fclose(lsrowfile);
        return NULL;
    }

    nread = fread(&rows,sizeof(int32_t),1,lsrowfile);
    if(nread != 1)
    {
        printf("Ivalid lsrowfile detected\n");
        fclose(lsrowfile);
        return NULL;
    }
    rows = BYTE_SWAP_4(rows);

    nread = fread(&cols,sizeof(int32_t),1,lsrowfile);
    if(nread != 1)
    {
        printf("Ivalid lsrowfile detected\n");
        fclose(lsrowfile);
        return NULL;
    }
    cols = BYTE_SWAP_4(cols);

    int tot_pts = rows*cols;

    printf("reading lsrowfile with %d rows and %d columns\n",rows,cols);
    uint16_t *img = new uint16_t[tot_pts];

    nread = fread(img,sizeof(uint16_t),tot_pts,lsrowfile);
    if(nread != tot_pts)
    {
        printf("Ivalid lsrowfile detected\n");
        delete [] img;
        fclose(lsrowfile);
        return NULL;
    }

    // byte swap the image
    for(int i=0;i < tot_pts;i++)
        img[i] = BYTE_SWAP_2(img[i]);

    // this version of the code generates a correction for columns-modulo eight. (it generates a correction
    // for columns that belong to one of eight different groups, where membership is determined from 
    // (column % 8))
    ChannelXTCorrection *xtptr = new ChannelXTCorrection();
    float *pvects = xtptr->AllocateVectorStorage(N_CORRECTION_GROUPS,nLen);
    float **vect_ptrs = xtptr->AllocateVectorPointerStorage(N_CORRECTION_GROUPS);
    xtptr->SetVectorIndicies(indicies,nLen);

    bool correction_valid = true;

    for(int i=0;i < N_CORRECTION_GROUPS;i++)
    {
        vect_ptrs[i] = pvects+nLen*i;
        if (GenerateGroupCorrection(i,vect_ptrs[i],rows,cols,img) == false)
        {
            correction_valid = false;
            break;
        }
    }

    delete [] img;
    fclose(lsrowfile);

    if(!correction_valid)
    {
        printf("Unable to compute valid correction\n");
        delete xtptr;
        return NULL;
    }

    return xtptr;
}

// generates a set of correction coefficients for a group of columns
// this basically solves the matrix equation Ax=B for x, where A are the measured high-speed 
// pixel values and B are the low-speed measured pixels values.  The special image collected on the PGM
// contains pairs of high-speed and low-speed rows for the same pixels that can be used to populate
// these matricies
// 
// As written, A is a nLen column matrix, with many rows in it (one for each example pixel we extract from the data)
// and B is an nLen row vector, again with one row per example pixel we extract from the data.
// When the equation is solved, both sides are multiplied by A-tranpose, which turns the left side into a square
// nLen by nLen matrix, and the rhs into an nLen row vector.  In order to facilitate ease of adding example pixels to
// the equations sequentially, A and B are not computed, and instead lhs = (A-transpose x A) and rhs = (A-transpose x B)
// are computed directly from the low-level data, and the solution is determined from lhs x = rhs
bool LSRowImageProcessor::GenerateGroupCorrection(int group_num,float *vect_output,int rows,int cols,uint16_t *img)
{
    double lhs[nLen*nLen];
    double rhs[nLen];

    memset(lhs,0,sizeof(lhs));
    memset(rhs,0,sizeof(rhs));

    // the image contains some invalid rows at the end that should be skipped
    int row_limit = rows - SKIP_ROWS_AT_END;

    int mcnt = 0;

    for(int column = group_num;column < cols;column += N_CORRECTION_GROUPS)
    {
        // make sure we have enough space on the left and right-hand sides to use this particular column
        if (((column + indicies[0]) < 0) || ((column + indicies[nLen-1]) >= cols))
            continue;

        // every other row is the start of a pair of rows, one high-speed and one reference
        for(int row = 0;row < row_limit;row += 2)
        {
            double amat[nLen];
            bool skip = false;

            uint16_t *hsrow = &img[row*cols];
            uint16_t *lsrow = &img[(row+1)*cols];

            // get the data points that are to be added to the matrix, making sure to filter out entries
            // that might reference pinned values
            for(int i=0;i < nLen;i++)
            {
                uint16_t temp = hsrow[indicies[i]+column];

                if((temp < PINNED_LOW_LIMIT) || (temp > PINNED_HIGH_LIMIT))
                {
                    skip = true;
                    break;
                }
                amat[i] = (double)temp;

                temp = lsrow[indicies[i]+column];
                if((temp < PINNED_LOW_LIMIT) || (temp > PINNED_HIGH_LIMIT))
                {
                    skip = true;
                    break;
                }
            }

            // if everything checks out, add this data into the matrix equation
            if(!skip)
            {
                mcnt++;    
                AccumulateMatrixData(lhs,rhs,amat,(double)(lsrow[column]));
            }
        }
    }


    Mat<double> lhs_matrix(nLen,nLen);
    Col<double> rhs_vector(nLen);
    Col<double> coeffs(nLen);
    bool result_ok = true;

    for(int col=0;col < nLen;col++)
        for(int row=0;row <= col;row++)
        {
            lhs_matrix(row,col) = lhs[row*nLen+col];
            lhs_matrix(col,row) = lhs[row*nLen+col];
        }

    for(int row=0;row < nLen;row++)
        rhs_vector(row) = rhs[row];

    try {
      //LaSpdMatFactorize(lhs_matrix,lhs_matrix_fact);
      //LaLinearSolve(lhs_matrix_fact,coeffs,rhs_vector);
      coeffs = solve(lhs_matrix,rhs_vector);
    }
    catch (std::runtime_error& le) {
        result_ok = false;
        coeffs.zeros(nLen);
    }

    // make sure derived coefficients are valid
    for(int row=0;row < nLen;row++)
        if(std::isnan(coeffs(row)))
        {
            result_ok = false;
            break;
        }
        else
            vect_output[row] = coeffs(row);

    printf("group %d correction coefficients: ",group_num);
    for(int row=0;row < nLen;row++)
        printf("%11.8lf ",coeffs(row));

    printf("\n");

    return(result_ok);
}

// Adds one example pixel's data into the lhs and rhs matrix and vector
void LSRowImageProcessor::AccumulateMatrixData(double *lhs,double *rhs,double *amat,double bval)
{
    for(int col=0;col < nLen;col++)
        for(int row=0;row <= col;row++)
            lhs[row*nLen+col] += amat[row]*amat[col];

    for(int row=0;row < nLen;row++)
        rhs[row] += amat[row]*bval;
}

