/*
 *  Created on: 8-25-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 87314 $
 *  Last changed by:  $Author: manninjm $
 *  Last change date: $Date: 2011-04-21 10:21:51 -0700 (Thu, 21 Apr 2011) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef FILTER_HPP_
#define FILTER_HPP_

#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include <samita/common/types.hpp>

/*!
    lifetechnologies namespace
*/
namespace lifetechnologies {

typedef std::unary_function<const Align&, bool> SamitaAlignFilter;

/*
 Filter predicate class that represents two other predicate classes
*/
template <typename First, typename Second>
class FilterPair : SamitaAlignFilter
{
public:
    FilterPair(const First& f, const Second& s): m_first(f), m_second(s) {}

    bool operator () (Align const &a)
    {
        return (m_first(a) && m_second(a));
    }
private:
    First m_first;
    Second m_second;
};

/*
 Filter predicate class that represents three other predicate classes
*/
template <typename First, typename Second, typename Third>
class FilterTriple : SamitaAlignFilter
{
public:
    FilterTriple(const First& f, const Second& s, const Third& t): m_first(f), m_second(s), m_third(t) {}

    bool operator () (Align const &a)
    {
        return (m_first(a) && m_second(a) && m_third(a));
    }
private:
    First m_first;
    Second m_second;
    Third m_third;
};

/*
 Filter predicate class that represents an arbitrary number of other predicate classes
*/
class FilterChain : SamitaAlignFilter
{
private:
    typedef boost::function<bool(Align const &a)> FilterFunctor;
public:
    FilterChain() {}

    FilterChain(const FilterFunctor& f1)
    {
        filters.push_back( f1 );
    }

    void add(const FilterFunctor& f1)
    {
        filters.push_back(f1);
    }
    bool operator()(Align const &a) const
    {
        BOOST_FOREACH(FilterFunctor f, filters)
        {
            if (!f(a))
                return false;
        }
        return true;
    }
private:
    std::vector<FilterFunctor> filters;
};

/*
 Filter predicate class that passes through alignment records with the specified flags
*/
class RequiredFlagFilter : SamitaAlignFilter
{
    public:
        RequiredFlagFilter(int flag=0) : m_flag(flag) {}
        bool operator() (Align const &a) const
        {
            return ((a.getFlag() & m_flag) == m_flag);
        }
        void setFlag(int flag) {m_flag = flag;}
    private:
        int m_flag;
};

/*
 Filter predicate class that filters out alignment records with the specified flags
*/
class FlagFilter : SamitaAlignFilter
{
    public:
        FlagFilter(int flag=0) : m_flag(flag) {}
        bool operator() (Align const &a) const
        {
            return (!(a.getFlag() & m_flag));
        }
        void setFlag(int flag) {m_flag = flag;}
    private:
        int m_flag;
};

/*
 Filter predicate class that filters out alignment records below the specified mapping quality
*/
class MapQualFilter : SamitaAlignFilter
{
    public:
        MapQualFilter(int qual=0) : m_mapq(qual) {}
        bool operator() (Align const &a) const
        {
            const int32_t mq = a.getQual(); // Only look at MAPQ
            return (mq >= m_mapq);
        }
        void setMapQual(int qual) {m_mapq = qual;}
    private:
        int m_mapq;
};

/*
 Filter predicate class that captures many common flag and mapping quality parameters
*/
class StandardFilter : SamitaAlignFilter
{
    public:
        StandardFilter(bool mapped = true, bool primary = true, bool duplicates = false, bool proper = false, bool mates = false, int mapq = 0)
        {
            int filteringFlags = 0;
            int requiredFlags = 0;

            // set up required flags
            if (proper)
                requiredFlags |= BAM_FPROPER_PAIR;
            m_requiredFlagFilter.setFlag(requiredFlags);

            // set up filtering flags
            if (mapped)
                filteringFlags |= BAM_FUNMAP;
            if (primary)
                filteringFlags |= BAM_FSECONDARY;
            if (duplicates)
                filteringFlags |= BAM_FDUP;
            if (mates)
                filteringFlags |= BAM_FMUNMAP;
            m_filteringFlagsFilter.setFlag(filteringFlags);

            m_mapqFilter.setMapQual(mapq);
        }

        /*
        Specifies the flag bits that, if set, cause an Alignment to be filtered
        */
        void setFilteringFlags(int flags) {m_filteringFlagsFilter.setFlag(flags);}

        /*
        Specifies the flag bits that must be set for an Alignment to pass through filter
        */
        void setRequiredFlags(int flags) {m_requiredFlagFilter.setFlag(flags);}

        void setMapQual(int mapq) {m_mapqFilter.setMapQual(mapq);}

        bool operator() (Align const &a) const
        {
            return (m_requiredFlagFilter(a) && m_filteringFlagsFilter(a) && m_mapqFilter(a));
        }
    private:
        MapQualFilter m_mapqFilter;
        FlagFilter m_filteringFlagsFilter;
        RequiredFlagFilter m_requiredFlagFilter;
};


} //namespace lifetechnologies

#endif //FILTER_HPP_
