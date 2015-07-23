/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved. */

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifndef __GAP_COST_TEMPL_H__
#define __GAP_COST_TEMPL_H__

#include <algorithm>

namespace genstr
{

// default gap cost parameters
#define DEF_GIP 8
#define DEF_GEP 1
#define DEF_GAPMAX 20
#define DEF_GIP1 8
#define DEF_GAPMAX1 20
#define DEF_GIP3 3
#define DEF_GAPMAX3 15


// basic virtual gap cost evaluator
template <class ValueType> class GapCost
{
public:
    ValueType operator () (unsigned gapSize)
    {
        return calc (gapSize);
    }
    virtual ValueType calc (unsigned gapSize) = 0;
};

// gap costs for which cost(N) can be calculated from cost(N-1)
template <class ValueType> class DeterministicGapCost : public GapCost<ValueType>
{
    virtual ValueType next (ValueType prevCost = (ValueType) 0) = 0;
};

template <class ValueType> class ConvexGapCost : public GapCost<ValueType>
{
};

// normal Smith-Waterman (and Needleman-Wunch) affine gap cost
template <class ValueType> class AffineGapCost : public ConvexGapCost<ValueType>
{
    ValueType gip_;
    ValueType gep_;
public:
    AffineGapCost (ValueType gip = (ValueType) DEF_GIP, ValueType gep = (ValueType) DEF_GEP)
    {
        configure (gip, gep);
    }
    ValueType gip () const { return gip_; }
    ValueType gep () const { return gep_; }
    void configure (ValueType gip, ValueType gep)
    {
        gip_ = gip, gep_ = gep;
    }
    ValueType calc (unsigned gapSize)
    {
        if (!gapSize) return 0;
        else return gip_ + gep_ * gapSize;
    }
    ValueType next (ValueType prev = (ValueType) 0)
    {
        if (prev == 0) return gip_ + gep_;
        else return prev + gep_;
    }
};

template <class ValueType>
inline double calc_b (ValueType gip, ValueType gmaxp)
{
    return (double (gmaxp) - double (gip)) / double (gip);
}
template <class ValueType>
inline double calc_a (ValueType gmaxp, double b)
{
    return (-b * double (gmaxp));
}

template <class ValueType> class BoundAffineGapCost : public AffineGapCost<ValueType>
// cost evaluator that is linear up to some length, then constant
{
    ValueType gmaxp_;
public:
    BoundAffineGapCost (ValueType gip = (ValueType) DEF_GIP, ValueType gep = (ValueType) DEF_GEP, ValueType gmaxp = (ValueType) DEF_GAPMAX)
    :
    AffineGapCost <ValueType> (gip, gep),
    gmaxp_ (gmaxp)
    {
    }
    ValueType gmaxp () const { return gmaxp_; }
    void configure (ValueType gip, ValueType gep, ValueType gmaxp)
    {
        AffineGapCost<ValueType>::configure (gip, gep);
        gmaxp_ = gmaxp;
    }
    ValueType calc (ValueType gapSize)
    {
        ValueType cost = AffineGapCost<ValueType>::calc (gapSize);
        if (cost > gmaxp_) cost = gmaxp_;
        return cost;
    }
    ValueType next (ValueType prev = (ValueType) 0)
    {
        if (prev == 0) return std::min (gmaxp_, this->gip_ + this->gep_);
        else return std::min (gmaxp_, prev + this->gep_);
    }
};

// asymptotic cost evaluator that gives less increment of penalty for longer gaps
template <class ValueType> class AsymptoticGapCost : public ConvexGapCost<ValueType>
{
    // asymptotic equation is:
    // COST = a / (GAPSIZE + b) + gmaxp
    // where
    //     b = (gmaxp - gip) / gip
    //     a = - b * gmaxp

    ValueType gip_;
    ValueType gmaxp_;
    double a_;
    double b_;
public:
    AsymptoticGapCost (ValueType gip = (ValueType ) DEF_GIP, ValueType gmaxp = (ValueType) DEF_GAPMAX)
    {
        configure (gip, gmaxp);
    }
    ValueType gip () const { return gip_; }
    ValueType gmaxp () const { return gmaxp_; }
    void configure (ValueType gip, ValueType gmaxp)
    {
        gip_ = gip;
        gmaxp_ = gmaxp;
        b_ = calc_b<ValueType> (gip, gmaxp);
        a_ = calc_a <ValueType> (gmaxp, b_);
    }
    ValueType calc (unsigned gapSize)
    {
        if (!gapSize) return (ValueType) 0;
        else return (ValueType) (a_ / (gapSize + b_) + gmaxp_);
    }
    ValueType next (ValueType prev = (ValueType) 0)
    {
        if (prev == 0) return gip_;
        else
        {
            ValueType l = gmaxp_ - prev;
            return prev + l * (1 - a_ / (a_ + l));
        }
    }
};

template <class ValueType> class AffineTripletGapCost : public GapCost <ValueType>
{
    // Superposition of two asymptotic gap costs (different cost for third positions)
    ValueType gip1_;
    ValueType gip3_;
    ValueType gep1_;
    ValueType gep3_;
public:
    AffineTripletGapCost (ValueType gip1 = (ValueType) DEF_GIP1, ValueType gmaxp1 = (ValueType) DEF_GAPMAX1, ValueType gip3 = (ValueType) DEF_GIP3, ValueType gmaxp3 = (ValueType) DEF_GAPMAX3)
    {
        configure (gip1, gmaxp1, gip3, gmaxp3);
    }
    ValueType gip1 () const { return gip1_; }
    ValueType gep1() const { return gep1_; }
    ValueType gip3 () const { return gip3_; }
    ValueType gep3 () const { return gep3_; }
    void configure (ValueType gip1, ValueType gep1, ValueType gip3, ValueType gep3)
    {
        gip1_ = gip1;
        gep1_ = gep1;
        gip3_ = gip3;
        gep3_ = gep3;
    }
    ValueType calc (unsigned gapSize)
    {
        unsigned rem = gapSize %  3;
        unsigned triplets = gapSize / 3;
        return (triplets ? gip3_ + triplets * gep3_ : 0) + (rem ? gip1_ + rem * gep1_ : 0);
    }
};

template <class ValueType> class AsymptoticTripletGapCost : public GapCost <ValueType>
{
    // Superposition of two asymptotic gap costs (different cost for third positions)
    ValueType gip1_;
    ValueType gip3_;
    ValueType gmaxp1_;
    ValueType gmaxp3_;
    double a1_;
    double a3_;
    double b1_;
    double b3_;
public:
    AsymptoticTripletGapCost (ValueType gip1 = (ValueType) DEF_GIP1, ValueType gmaxp1 = (ValueType) DEF_GAPMAX1, ValueType gip3 = (ValueType) DEF_GIP3, ValueType gmaxp3 = (ValueType) DEF_GAPMAX3)
    {
        configure (gip1, gmaxp1, gip3, gmaxp3);
    }
    ValueType gip1 () const { return gip1_; }
    ValueType gmaxp1() const { return gmaxp1_; }
    ValueType gip3 () const { return gip3_; }
    ValueType gmaxp3 () const {    return gmaxp3_; }
    void configure (ValueType gip1, ValueType gmaxp1, ValueType gip3, ValueType gmaxp3)
    {
        gip1_ = gip1;
        gmaxp1_ = gmaxp1;
        gip3_ = gip3;
        gmaxp3_ = gmaxp3;
        b1_ = calc_b<ValueType> (gip1, gmaxp1_);
        a1_ = calc_a <ValueType> (gmaxp1_, b1_);
        b3_ = calc_b<ValueType> (gip3, gmaxp3_);
        a3_ = calc_a <ValueType> (gmaxp3_, b3_);
    }
    ValueType calc (unsigned gapSize)
    {
        if (!gapSize) return (ValueType) 0;
        if (gapSize % 3 == 0) return (ValueType) (a3_ / (gapSize + b3_) + gmaxp3_);
        else return (ValueType) (a1_ / (gapSize + b1_) + gmaxp1_);
    }
    // that would be nice to introduce 'phase' concept, where finite number of phases (and states associated with them)
    // will be kept for the position. This will make possible non-tracing (and quadratic in time!) calculation for phased gap costs
};

} // namespace genstr

#endif // __GAP_COST_TEMPL_H__
