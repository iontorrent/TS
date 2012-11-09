/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/**
 * secantFind.h
 * Exported interface: secantFind
 * Root-finding algorithm based on the secant method.
 * 
 * @author Magnus Jedvert
 * @version 1.1 april 2012
*/
#pragma once
#ifndef SECANTFIND_H
#define SECANTFIND_H

#include <cmath>
#include <algorithm>
#include <functional>
using namespace std;

// the sign-function, couldn't find it in std
template <typename T> 
static int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// makes a delta that is not too small, to avoid problem with dividing with zero
static double validDelta(const double delta, const double minVal = 0.01) {
    return abs(delta) > minVal ? delta: sgn(delta) * minVal;
}

/**
 * secantFind - Returns x such that fun(x) = aim. a is a starting value, and fA should be fA = fun(a)
 * b is the next value, these are to get the secant method started. n is the number of iterations used
 */
template<typename F>
double secantFind(const F &fun, int n, double aim, double a, double fA, double b) {
    double fB;

    for (int i = 0; i < n; ++i) {
        const double delta = validDelta(b - a);
        b = a + delta;    

        fB = fun(b);
        const double dErr = (fB - fA) / delta;
        fA = fB;         

        a = b;
        b = max(2.0, a + (aim - fB) / dErr); 
    }

    return b;
};
#endif // SECANTFIND_H
