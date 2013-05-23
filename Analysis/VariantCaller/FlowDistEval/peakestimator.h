/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     peakestimator.h
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#ifndef PEAKESTIMATOR_H
#define PEAKESTIMATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include "sys/types.h"
#include "sys/stat.h"
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <levmar.h>

#define LIMIT 6
#define MAXSIGDEV  1200

double deter(double a[][LIMIT],int forder);
int chckdgnl(double array[][LIMIT],int forder);
double expGauss (double x, double b1, double c1);
double expDoubleGauss(double x, double b1, double b2, double c1, double c2);
void doubleGaussianfunc(double *p, double *x, int m, int n, void *data);
void jacDoubleGaussFunc(double *p, double *jac, int m, int n, void *data);
double hessDoubleGaussFunc(double *p, int m, int n);

void gaussianfunc(double *p, double *x, int m, int n, void *data);
void jacgaussfunc(double *p, double *jac, int m, int n, void *data) ;
void get_freq( int array[], int length, int first_pos, double ** freq) ;
void runLMS ( int* _x, int n, int* _params, float * ret_value, int start, int end, int refPolymer, int varPolymer, int DEBUG);
double* getLogLikelihood(int* flowSignals, int n, double mean, int numComponents);
double calculateLogLikelihood(double* means, double* variances, double* mixingProbs, int* flowSignals, int n, int numComponents, double** weights);
void runEM(int* flowSignals, double* means, double* variances, double* mixingProbs, double** weights, int n, int numComponents);
void  runUpdateStep(int* flowSignals, double* means, double* variances, double* mixingProbs, double** weights, int n, int numComponents);
void updateVariances(double* means, double* variances, double** weights, int* flowSignals, int n, int numComponents);
void updateMeans(double* means, double** weights, int* flowSignals, int n, int numComponents);
void updateMixingProbs(double* mixingProbs, double** weights, int n, int numComponents);
int runEstimateStep(int* flowSignals, double* means, double* variances, double* mixingProbs, double** weights, int n, int numComponents);
void initPriors(double* means, double* variances, double* mixingProbs, int numComponents, double mean);
double calculateNormalPDF(double mean, double variance, double x);
#endif // PEAKESTIMATOR_H
