/*
 *  Created on: 05-05-2010
 *      Author: Keith Moulton
 *
 *  Latest revision:  $Revision: 47952 $
 *  Last changed by:  $Author: moultoka $
 *  Last change date: $Date: 2010-07-08 11:05:34 -0400 (Thu, 08 Jul 2010) $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef FEATURE_HPP_
#define FEATURE_HPP_

#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <samita/common/types.hpp>
#include <samita/common/interval.hpp>

namespace lifetechnologies
{


/*!
 Typedef representing a set of feature attribute values
 */
typedef std::set<std::string> FeatureAttributeValues;

/*!
 Typedef representing a container of <key, set<value> > feature attributes.
 The key is a string and the value is a set of strings.
 */
typedef std::map<std::string, FeatureAttributeValues > FeatureAttributeMap;

/*!
 Class representing a feature
 */
class Feature : public Interval
{
public:
    Feature() :
        Interval(), m_score(-1), m_depth(1), m_frame("."), m_strand(".")
    {
    }
    Feature(std::string const& name, std::string const& src, std::string const& type, int32_t const start, int32_t const end, double const score = 0.0,
            std::string const& frm = ".", std::string const& strand = ".") :
                Interval(name, start, end), m_source(src), m_type(type), m_score(score),
                m_depth(), m_frame(frm), m_strand(strand), m_attributes()
    {
    }

    void setSource(std::string const& source)
    {
        m_source = source;
    }
    std::string const& getSource() const
    {
        return m_source;
    }

    void setAttribute(std::string const& att)
    {
    	m_types.insert(att);
    }

    std::string const& getAttribute() const
    {
    	return m_type;
    }

    void addTypes(std::set<std::string> const& types)
    {
    	m_types.insert(types.begin(), types.end());
    }

    std::set<std::string> const& getTypes() const
    {
    	return m_types;
    }

    void setType(std::string const& type)
    {
        m_type = type;
    }
    std::string const& getType() const
    {
        return m_type;
    }

    void setScore(double const score)
    {
        m_score = score;
    }
    double getScore() const
    {
        return m_score;
    }

    void setDepth(unsigned int const depth)
    {
        m_depth = depth;
    }
    unsigned int getDepth() const
    {
        return m_depth;
    }

    void setStrand(std::string const& strand)
    {
        m_strand = strand;
    }
    void setStrand(Strand const strand)
    {
        if (strand == lifetechnologies::FORWARD)
            m_strand = "+";
        else if (strand ==  lifetechnologies::REVERSE)
            m_strand = "-";
        else
            m_strand = ".";
    }
    std::string getStrand() const
    {
        return m_strand;
    }

    void setFrame(std::string const &frame)
    {
        m_frame = frame;
    }
    std::string getFrame() const
    {
        return m_frame;
    }

    void setAttribute(std::string const& name, std::string const& value)
    {
        m_attributes[name].insert(value);
    }
    void setAttribute(std::string const& name, char const* value)
    {
        m_attributes[name].insert(value);
    }
    FeatureAttributeValues getAttribute(std::string const& name) const
    {
        FeatureAttributeMap::const_iterator iter = m_attributes.find(name);
        if (iter != m_attributes.end())
        {
            return iter->second;
        }
        return FeatureAttributeValues();
    }
    FeatureAttributeMap const& getAttributes() const
    {
        return m_attributes;
    }

    /*!
     Add two intersecting features
     \return A feature representing the union of both input features.
     \exception invalid_argument
     */
    Feature& operator+=(Feature const& other)
    {
        Interval::operator+=(other);
        mergeAttributes(other);
        return *this;
    }

    /*!
     Add two intersecting features
     \return A new feature representing the union of both input features.
     \exception invalid_argument
     */
    const Feature operator+(Feature const& other) const
    {
        Feature intersection = *this;
        intersection += other;
        return intersection;
    }

    /*!
     Intersect two intersecting features
     \return A new feature representing the intersection of both input features.
     \exception invalid_argument
     */
    Feature intersect(Feature const& other) const
    {
        Feature intersection = *this;
        intersection.Interval::intersect(other);
        intersection.mergeAttributes(other);
        return intersection;
    }

    int32_t getSize() const
    {
    	if(isEmpty())
    		return 0;

    	return m_end - m_start + 1;
    }

    bool isEmpty() const
    {
    	return !(m_start && m_end);
    }

    Feature operator & (Feature const& other) const
    {
  		Feature feature;
  		feature.setSequence(m_sequence);

  		if(intersects(other))
    	{
    		feature.setInterval(std::max(m_start, other.getStart()), std::min(m_end, other.getEnd()));

    		feature.m_types = m_types;
    		feature.m_types.insert(other.m_types.begin(), other.m_types.end());
    		// interval.statistics.m_coverage += other.statistics.m_coverage;
    	}
  		else
  		{
  			feature.setInterval(0, 0);
  		}

    	return feature;
    }

    struct Contains
    {
      bool operator() (Feature const& left, Feature const& right) const
      {
        return (left & right) == left;
      }
    };

    bool operator==(Interval const& other) const
    {
      return ((m_sequence == other.getSequence()) && (m_start == other.getStart()) && (m_end == other.getEnd()));
    }



    bool operator!=(Interval const& other) const
    {
    	return !(*this == other);
    }

protected:
    std::string m_source;
    std::string m_type;
    std::set<std::string> m_types;
    double m_score;
    unsigned int m_depth;
    std::string m_frame;
    std::string m_strand;
    FeatureAttributeMap m_attributes;
private:
    void mergeAttributes(Feature const& other)
    {
        FeatureAttributeMap const& attrs = other.getAttributes();
        FeatureAttributeMap::const_iterator attrs_iter = attrs.begin();
        FeatureAttributeMap::const_iterator attrs_end = attrs.end();
        while (attrs_iter != attrs_end)
        {
            std::string const& name = attrs_iter->first;
            FeatureAttributeValues const& values = attrs_iter->second;
            m_attributes[name].insert(values.begin(), values.end());
            ++attrs_iter;
        }
    }
};

typedef std::vector<Feature> FeatureArray;
typedef std::multiset<Feature> FeatureSet;

} //namespace lifetechnologies

#endif //FEATURE_HPP_
