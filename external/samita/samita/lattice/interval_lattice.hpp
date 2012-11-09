/*        File: lattice.hpp
 *  Created on: Jul 27, 2010
 *      Author: Caleb J Kennedy (caleb.kennedy@lifetech.com)
 * 
 *  Latest revision:  $Revision: $
 *  Last changed by:  $Author: kennedcj $
 *  Last change date: $Date: $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef __INTERVAL_LATTICE_HPP__
#define __INTERVAL_LATTICE_HPP__ 1

// #include <samita/common/interval.hpp>
#include <samita/common/feature.hpp>
#include <samita/graph/graph.hpp>

namespace lifetechnologies {

struct Contains {
	bool operator()(Feature const& left, Feature const& right)
	{
		return (left & right) == left;
	}
};

class FeatureLattice : public Graph<Feature, int, Feature::Contains>
{
	public:
		enum AddType
		{
			LT_UNADD,
			LT_ERROR,
			LT_GENERATOR,
			LT_CANDIDATE,
			LT_NEW
		};

	  FeatureLattice(Feature const& bottom) : m_type(LT_UNADD)
	  {
	  	m_bottom = new Node<Feature>(bottom);
	    m_top = m_bottom;
	  }

	  FeatureLattice(std::string const& block="", int32_t max=INT_MAX) : m_type(LT_UNADD)
	  {
	  	assert(max >= 0);

	  	Feature feature;
	  	feature.setSequence(block);
	  	feature.setInterval(1, max);

	  	m_bottom = new Node<Feature>(feature);

	  	// std::cout << "m_bottom = " << m_bottom->label().toString() << std::endl;
	  	m_top = m_bottom;
	  }

	  template<class Pruner>
	  iterator<Pruner> getBottom(bool traverse=true, Pruner const& prune=Pruner())
	  {
	  	return begin(traverse, prune);
	  }

	  template<class Pruner>
	  iterator<Pruner> getTop(bool traverse=true, Pruner const& prune=Pruner())
	  {
	  	return iterator<Pruner>(prune, traverse ? ++m_color : m_color, m_top);
	  }

	  template<class Pruner>
	  iterator<Pruner> insert(Feature const& interval, int32_t coverage=0, Pruner const& prune=Pruner())
	  {
	  	m_type = LT_UNADD;
	  	iterator<Pruner> newFeature = iterator<Pruner>(prune, m_color, m_add(interval, m_bottom, prune));

	  	switch(m_type)
	  	{
	  	  case LT_UNADD:
	  	  {

	  		  break;
	  	  }
	  	  case LT_ERROR:
	  	  {
	  	  	// throw exception
	  	  	break;
	  	  }
	  	  case LT_GENERATOR:
	  	  {
	  	  	// add coverage
	  	  	break;
	  	  }
	  	  case LT_CANDIDATE:
	  	  {
	  	  	iterator<Pruner> tmp = newFeature;

	  	  	newFeature = insert((*newFeature).label() + interval, 0, prune);

	  	  	Node<Feature>::iterator edge = (*tmp).begin(), last = (*tmp).end();
	  	  	while(edge != last)
	  	  	{
	  	  		(*newFeature).orphan(*tmp);

	  	  		if(&(*edge).target() != &(*newFeature))
	  	  		{
	  	  			bool found = false;

	  	  			Node<Feature>::iterator child = (*newFeature).begin(), final = (*newFeature).end();
	  	  			while(child != final)
	  	  			{
	  	  				if(&(*edge).target() == &(*child).target())
	  	  				{
	  	  					found = true;
	  	  					break;
	  	  				}

	  	  				++child;
	  	  			}

	  	  			if(!found)
	  	  			{
	  	  				(*newFeature).adopt((*edge).target());
	  	  			}
	  	  		}

	  	  		(*tmp).orphan(edge);
	  	  		++edge;
	  	  	}

	  	  	//delete &(*tmp);

	  	    break;
	  	  }
	  	  case LT_NEW:
	  	  {
	  	  	// return newFeature
	  	  	break;
	  	  }
	  	  default:
	  	  {
	  	  	std::cerr << "unknown m_type" << std::endl;
	  	  }
	  	}

	  	if(coverage)
	  	{
	  		// TODO: Add coverage
	  	}
	  	std::cout << "number of intervals = " << g_size << std::endl;
	    return newFeature;
	  }

	  template<class Pruner>
	  iterator<Pruner> find(Feature const& interval, Pruner const& prune=Pruner())
	  {
	  	// std::cout << "--- FIND ---" << std::endl;
	  	Node<Feature> * generator = m_bottom;

	  	bool max = true;

	  	int halt = 0;
  	  while(max /*&& halt < 10*/) {
  	  	max = false;
  	  	typename Node<Feature>::iterator edge = generator->begin(), last = generator->end();
  	  	while(edge != last) {

  	  		// std::cout << "IF(" << interval.toString() << " <= " << (*edge).target().label().toString() << ") && (" << (*edge).target().label().toString() << " <= " << generator->label() << ") ";
  	  		if(compare(interval, (*edge).target().label()) && compare((*edge).target().label(), generator->label())) {
  	  			// std::cout << "TRUE" << std::endl;
  	  			generator = &(*edge).target();
  	  			max = true;
  	  			break;
  	  		} else {
  	  			// std::cout << "FALSE" << std::endl;
  	  		}
  	  		++edge;
  	  	}
  	  	++halt;
  	  }

  	  // std::cout << "--- FIND ---" << std::endl;

  	  return iterator<Pruner>(prune, m_color, generator);
	  }



	private:
	  AddType m_type;
	  Node<Feature> * m_top;

	  uint32_t m_nmeets;

	  template<class Pruner>
	  Node<Feature> * m_add(Feature const& interval, Node<Feature> * generator, Pruner const& prune)
	  {
  	  bool max = true;

	  	// std::cout << "ADD INTERVAL " << interval.toString() << " number of generator edges = " << generator->degree() << std::endl;

	  	int halt = 0;
  	  while(max /*&& halt < 10*/) {
  	  	max = false;
  	  	typename Node<Feature>::iterator edge = generator->begin(), last = generator->end();
  	  	while(edge != last) {

  	  		// std::cout << "IF(" << interval.toString() << " <= " << (*edge).target().label().toString() << ") && (" << (*edge).target().label().toString() << " <= " << generator->label() << ") ";
  	  		if(compare(interval, (*edge).target().label()) && compare((*edge).target().label(), generator->label())) {
  	  			// std::cout << "TRUE" << std::endl;
  	  			generator = &(*edge).target();
  	  			max = true;
  	  			break;
  	  		} else {
  	  			// std::cout << "FALSE" << std::endl;
  	  		}
  	  		++edge;
  	  	}
  	  	++halt;
  	  }

  	  if(generator->label() == interval) {
  	  	// std::cout << "RETURN(" << generator->label().toString() << ")" << std::endl;
  	  	m_type = LT_GENERATOR;
  	  	return generator;
  	  }

	  	std::vector< Node<Feature>* > parents;
	  	typename Node<Feature>::iterator edge = generator->begin(), last = generator->end();
	  	while(edge != last) {
	  		if(compare(generator->label(), (*edge).target().label())) {
	  			++edge;
	  			continue;
	  		}

	  		Node<Feature> * candidate = &(*edge).target();


	  		//std::set<Attribute> d = attributes(iterator<Pruner>(prune, g_color, candidate));

	  		// std::cout << "IF(!" << candidate->label().toString() << " <= " << interval.toString() << " && !" << interval.toString() << " <= " << candidate->label().toString() << ") ";

	  		if(!compare(candidate->label(), interval) && !compare(interval, candidate->label())) {
	  		//if(candidate->label().intersects(interval)) {
	  		  // std::cout << "TRUE" << std::endl;

	  			if(candidate->label().intersects(interval) && candidate->label().getTypes() == interval.getTypes())
	  			{
	  				m_type = LT_CANDIDATE;
	  				return candidate;
	  			}

	  			candidate = m_add(candidate->label() & interval, candidate, prune);

	  		} else {
	  			// std::cout << "FALSE" << std::endl;
	  		}

	  		bool add = true;
	  		unsigned i = 0;
	  		unsigned numberOfParents = parents.size();
	  		while(!parents.empty()) {
	  			if(i >= numberOfParents) break;
	  			if(compare(candidate->label(), parents.back()->label())) {
	  				add = false;
	  				break;
	  			} else if(compare(parents.back()->label(), candidate->label())) {
	  				parents.erase(parents.end() - 1);
	  			}
	  			++i;
	  		}

	  		if(add)
	  			parents.push_back(candidate);

	  		++edge;
	  	}


	  	//std::set<std::string> types = interval.getTypes();
	  	//types.insert((*generator).label().getTypes().begin(), (*generator).label().getTypes().end());

	  	Node<Feature> * newFeature = new Node<Feature>(Feature(interval));

	  	if(newFeature)
	  	{
	  		m_type = LT_NEW;
	  	}
	  	else
	  	{
	  		m_type = LT_ERROR;
	  	}

	  	m_top = compare(newFeature->label(), m_top->label()) ? newFeature : m_top;
	  	++g_order;

	  	//std::set<Attribute> c = attributes(*newConcept);

	  	// std::cout << "NEWINTERVAL = " << newFeature->label().toString() << std::endl;

  		//std::set<Attribute> g = attributes(iterator<Pruner>(prune, g_color, generator));

	  	//std::set<std::string> atts;
	  	typename std::vector< Node<Feature>* >::iterator parent = parents.begin(), p_last = parents.end();
	  	while(parent != p_last) {
	  		(*parent)->orphan(*generator);
	  		generator->orphan(**parent);
	  		//std::set<Attribute> p = attributes(iterator<Pruner>(prune, g_color, *parent));

	  		// std::cout << (*parent)->label().toString() << "->ORPHAN(" << generator->label().toString() << ")" << std::endl;
	  		//// std::cout << generator->label().toString() << "->ORPHAN(" << (*parent)->label().toString() << ")" << std::endl;



	  		(*parent)->adopt(*newFeature);
	  		newFeature->adopt(**parent);

	  		const_cast<Feature&>((*parent)->label()).addTypes((*newFeature).label().getTypes());

	  		// std::cout << (*parent)->label().toString() << "->ADOPT(" << newFeature->label().toString() << ")" << std::endl;


	  		//// std::cout << newFeature->label().toString() << "->ADOPT(" << (*parent)->label().toString() << ")" << std::endl;

	  		//// std::cout << p << "->ADOPT(" << c << ")" << std::endl;
	  		//relation::Graph<Concept, Set, Subset>::insert(iterator<Pruner>(prune, g_color, *parent), generator->label());
	  		// atts.insert((*parent)->label().getTypes().begin(), (*parent)->label().getTypes().end());
	  		++parent;
	  	}

	  	//const_cast<Feature&>(newFeature->label()).addTypes(atts);
	  	newFeature->adopt(*generator);
	  	generator->adopt(*newFeature);
	  	//// std::cout << c << "->ADOPT(" << g << ")" << std::endl;
	  	const_cast<Feature&>((*newFeature).label()).addTypes((*generator).label().getTypes());
	  	// std::cout << newFeature->label().toString() << "->ADOPT(" << generator->label().toString() << ")" << std::endl;

	  	++g_size;
	  	//relation::Graph<Concept, Set, Subset>::insert(iterator<Pruner>(prune, g_color, newConcept), generator->label());

	  	return newFeature;
	 }
};

}

#endif // __INTERVAL_LATTICE_HPP__
