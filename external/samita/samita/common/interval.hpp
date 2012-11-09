/*
 *  Created on: 04-16-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 47998 $
 *  Last changed by:  $Author: moultoka $
 *  Last change date: $Date: 2010-07-12 11:44:57 -0400 (Mon, 12 Jul 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef INTERVAL_HPP_
#define INTERVAL_HPP_

#include <cstdlib>
#include <cassert>
#include <string>
#include <boost/lexical_cast.hpp>
#include <samita/exception/exception.hpp>
#include <lifetech/string/util.hpp>

namespace lifetechnologies
{

class Interval
{
public:
    Interval() : m_sequence(), m_start(-1), m_end(-1) {}
    Interval(std::string const& sequence, int32_t const start, int32_t const end) :
        m_sequence(sequence), m_start(start), m_end(end) {}

    void setSequence(std::string const& seq)
    {
        m_sequence = seq;
    }
    std::string const& getSequence() const
    {
        return m_sequence;
    }

    void setInterval(int32_t const start, int32_t const end)
    {
        setStart(start);
        setEnd(end);
    }
    void setStart(int32_t const start)
    {
        m_start = start;
    }
    int32_t getStart() const
    {
        return m_start;
    }

    void setEnd(int32_t const end)
    {
        m_end = end;
    }
    int32_t getEnd() const
    {
        return m_end;
    }

    size_t getLength() const
    {
        if(m_end <= 0) return 0;
        assert(m_end > m_start);
        return (m_end - m_start + 1);
    }
    /*!
     Test if a sequence interval abuts another sequence interval
     \return true if abutting; otherwise false
     */
    bool abuts(Interval const& other) const
    {
        if (m_sequence == other.getSequence())
        {
            int32_t start2 = other.getStart();
            int32_t end2 = other.getEnd();
            return ((start2  == (m_end+1)) || ((end2+1) == m_start));
        }
        return false;
    }

    /*!
     Expand interval by amount in both directions
     */
    void expand(const int amount)
    {
        m_start = std::max(m_start-amount, 0);
        m_end += amount;
    }

    /*!
     Test if an interval intersects another interval
     \return true if intersecting; otherwise false
     */
    bool intersects(Interval const& other, int const allowable_error=0) const
    {
        if (m_sequence == other.getSequence())
        {
            int32_t start2 = std::max(other.getStart() - allowable_error, 0);
            int32_t end2 = other.getEnd() + allowable_error;
            return ((start2 >= m_start && start2 <= m_end) || (end2 >= m_start && end2 <= m_end)
                    || ((m_start >= start2) && (m_end <= end2)));
        }
        return false;
    }

    /*!
     Add two intersecting intervals
     \return A new interval representing the union of both input intervals.
     \exception invalid_argument
     */
    Interval& operator+=(Interval const& other)
    {
        if (intersects(other) || abuts(other))
        {
            m_start = std::min(m_start, other.getStart());
            m_end = std::max(m_end, other.getEnd());
            return *this;
        }
        std::stringstream sstrm;
        sstrm << "interval " << *this << " does not intersect " << other;
        throw std::invalid_argument(sstrm.str());
    }

    /*!
     Add two intersecting intervals
     \return A new interval representing the union of both input intervals.
     \exception invalid_argument
     */
    const Interval operator+(Interval const& other) const
    {
        Interval intersection = *this;
        intersection += other;
        return intersection;
    }

    /*!
     Less than operator
     \return True if this interval is to the left of the other interval.
     */
    bool operator<(Interval const& other) const
    {
        return (m_start < other.getStart());
    }

    /*!
     Intersect two intersecting intervals
     \return A new interval representing the intersection of both input intervals.
     \exception invalid_argument
     */
    Interval intersect(Interval const& other)
    {
        if (intersects(other))
        {
            return Interval(m_sequence,
                    std::max(m_start, other.getStart()),
                    std::min(m_end, other.getEnd()));
        }
        std::stringstream sstrm;
        sstrm << "interval " << *this << " does not intersect " << other;
        throw std::invalid_argument(sstrm.str());
    }
    std::string toString() const
    {
        std::stringstream sstrm;
        sstrm << m_sequence;
        if (m_start > 0)
        {
            sstrm << ":" << m_start;
            if (m_end > 0)
                sstrm << "-" << m_end;
        }
        return sstrm.str();
    }
protected:
    friend std::ostream &operator<<(std::ostream &out, Interval const& interval);

    //members
    std::string m_sequence;
    int32_t m_start;
    int32_t m_end;
};

/*!
 Class that represents a simple interval on a sequence. Coordinates are 1-based and closed ended.
 */
class SequenceInterval : public Interval
{
public:
    SequenceInterval() :
        Interval()
    {
    }
    SequenceInterval(std::string const& sequence, int32_t const start, int32_t const end) :
        Interval(sequence, start, end)
    {
    }

    SequenceInterval(char const* region)
    {
        SequenceInterval::parse(region, m_sequence, m_start, m_end);
    }

    /*!
     Static method to parse a sequence interval string in the form "chr1:1-1000"
      \exception boost::bad_lexical_cast
     */
    static void parse(char const* region, std::string& sequence, int& beg, int& end)
    {
        std::string cleanedRegion = string_util::remove_char(region, ',');
        //const int32_t MAX_END = 1<<29;
        std::vector<std::string> tokens;
        string_util::tokenize(cleanedRegion, ":", tokens);
        size_t nTokens = tokens.size();
        if (nTokens > 0)
        {
            sequence = tokens[0];
            if (nTokens == 2)
            {
                // caller specified a range like chr3:10-1000
                std::vector<std::string> range;
                string_util::tokenize(tokens[1], "-", range);
                size_t nRange = range.size();
                if (nRange > 0)
                    beg = boost::lexical_cast<int>(range[0]);
                if (nRange > 1)
                    end = boost::lexical_cast<int>(range[1]);
                else
                    //end = MAX_END; // rest of chromosome
                    end = -1;
            }
            else
            {
                // If no beg and end specified, just chr, assume entire chr
                //beg = 0; end = MAX_END;
                beg=-1; end=-1;
            }
        } else {
            // throw exception?!
            sequence = "UNKNOWN"; beg=-1; end=-1;
        }
        // FIXME - must be consistent with bam_parse_region
    }

    void setName(std::string const& name)
    {
        m_name = name;
    }
    std::string const& getName() const
    {
        return m_name;
    }

    bool operator==(SequenceInterval const& other) const
    {
        return ((m_sequence == other.getSequence()) && (m_start == other.getStart()) && (m_end == other.getEnd()));
    }

    /*!
     Split a sequence interval based on the grainsize
     \return A new sequence interval representing right half of the input interval.  The input interval is updated
     to represent the new left half. Unspecified if !isDivisible().
     When bisected with grainsize of 1, the interval "chr1:10-1000" becomes "chr1:10-550" and the new interval is "chr1:551-1000"
     \note
     When bisected with a grainsize of 3, the interval "chr1:10-1000" becomes "chr1:10-549" and the new interval is "chr1:550-1000"
     */
    SequenceInterval split(size_t grainsize=16384)
    {
        SequenceInterval right;
        right.setSequence(m_sequence);

        // split the interval in two at the user specified boundary
        size_t midpoint =  m_start + (size_t)(getLength() / 2);
        midpoint -= (midpoint % grainsize);
        // set coordinates for new interval
        right.setStart(midpoint + 1);
        right.setEnd(m_end);
        // set new end coordinate for this this interval
        setEnd(midpoint);
        return right;
    }
private:
    std::string m_name;
};

/*!
 typedef for an array of sequence intervals
 */
typedef std::vector<SequenceInterval> SequenceIntervalArray;

/*!
 << operator for outputting an Interval to an output stream
 */
inline std::ostream &operator<<(std::ostream &out, Interval const& interval)
{
    out << interval.toString();
    return out;
}

} // namespace lifetechnologies

#endif //INTERVAL_HPP_
