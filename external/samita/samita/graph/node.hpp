/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * node.hpp
 *
 *  Created on: Mar 15, 2010
 *      Author: kennedcj
 */

#ifndef NODE_HPP_
#define NODE_HPP_

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/assume_abstract.hpp>

#include <cassert>
#include <iostream>

/*!
 * @mainpage
 * @par C++ container class for labeled nodes and weighted edges.
 */

namespace lifetechnologies
{

	/*
	 * For color indexing.
	 */

  enum Color {BLACK};

  /*
   * Define node 'Label' and edge 'Weight' classes for the container.
   */

  template<class Label, class Weight=int>

  /*
   * @abstract Nodes contain labels as well as a double linked list of weighted edges. Each edge contains a pointer to
   * its adjacent node as well as pointers to the next and previous edges. The iterator class provides standardized methods
   * for incremental access to adjacent nodes (edges).
   *
   * For most useful purposes nodes and edges are collected into graphs (see graph.hpp).
   */

  class Node {
    public:
  		Node(void) : data(Label()), n_color(BLACK) {
  			n_ideg = n_odeg = 0;
  			first = last = NULL;
  		}

  		/*
  		 * @abstract Constructor to generate a labeled node with no edges.
  		 */

  	  Node(Label const& label) : data(label), n_color(BLACK) {
  	  	n_ideg = n_odeg = 0;
  	  	first = last = NULL;
  	  }

  	  /*!
  	   * @brief Brief description of node edges.
  	   *
  	   *
  	   */

      class Edge {
  	    public:
      		Edge(void) : data(Weight()), node(NULL) {
      			prev = next = NULL;
      		}

      		/*!
      		 * @param weight the edge's weight.
      		 */

  	  	  Edge(Weight const& weight) : data(weight), node(NULL) {
  	  	  	prev = next = NULL;
  	  	  }

  	  	  /*
  	  	   * @abstract Overloaded ostream insertion operator.
  	  	   *
  	  	   * @param out stream
  	  	   * @param edge adjacent node data
  	  	   * @return updated stream
  	  	   */



  		    Node& target(void) const {
  		    	assert(node);
  		    	return *node;
  		    }

  		    /*
  		     * @return edge weight.
  		     */

  		    Weight const& weight(void) const {return data;};

  	    //private:
  		    Weight data;
  	      Node *node;
  	      Edge *prev, *next;

  	      friend class Node;
  	      template<class L, class W, class C> friend class Graph;

  	      friend class boost::serialization::access;

  	     template<class Archive>
  	     void serialize(Archive &archive, const unsigned version) {
  	    	 std::cout << "SERIALIZE EDGE ... ";
  	       archive & data;
  	       archive & node;

  	       archive.register_type(static_cast<Node*>(NULL));
  	       archive.register_type(static_cast<Edge*>(NULL));

  	       archive & prev;
  	       archive & next;
  	       std::cout << "DONE" << std::endl;
  	     }
      };

      /*!
       * @par
       * The iterator class encapsulates pointer manipulation and provides methods for incrementally accessing individual edges and their adjacent nodes. Iterators represent the 'client' interface to node objects.
       *
       * @code
       * ...
       * @endcode
       */

  	  class iterator : public std::iterator<std::input_iterator_tag, Edge> {
  	    public:

  	  	  /*!
  	  	   * @param e edge pointer.
  	  	   * @param r true for reverse iterators, which provide access to nodes in the opposite direction (back-to-front).
  	  	   */

  	  	  iterator(Edge* e=NULL, bool r=false) : edge(e), reverse(r) {};

  	  	  /*!
  	  	   * @return reference to next iterator (default) or previous iterator (if reverse).
  	  	   */

  	  	  iterator& operator++(void) {
  	  		  if(reverse)
  	  			  edge = edge->prev;
  	  		  else
  	  	      edge = edge->next;

  	  		  return *this;
  	  	  }

  	  	  /*
  	  	   * @abstract Relational operator to test for non-equivalence.
  	  	   *
  	  	   * @param right iterator to compare
  	  	   * @return true if *this != right
  	  	   */

  	  	  bool operator!=(iterator const& right) {
  	  	    return edge != right.edge;
  	  	  }

  	  	  /*
  	  	   * @abstract Dereference operator.
  	  	   *
  	  	   * @return edge object
  	  	   */

  	  	  Edge const& operator*(void) {
  	  	  	assert(edge);
  	  	    return *edge;
  	  	  }

  	    private:
  	  	  Edge* edge;
  	  	  bool reverse;

  	  };

  	  /*!
  	   * @return an iterator to the first edge.
  	   */

  	  iterator begin(void) const {
  	  	return iterator(first);
  	  }

  	  /*!
  	   * @return a reverse iterator to the last edge.
  	   */

  	  iterator rbegin(void) const {
  	  	return iterator(last, true);
  	  }

  	  /*!
  	   * @return an iterator one past the last edge.
  	   */

  	  iterator end(void) const {
  	  	return iterator();
  	  }

  	  /*!
  	   * @return a reverse iterator one prior to the first edge.
  	   */

  	  iterator rend(void) const {
  	  	return end();
  	  }

  	  /*!
  	   * @param c color assignment for the node.
  	   */

  	  void color(int c) {n_color = c;}

  	  /*!
  	   * @return the node's color.
  	   */

  	  int color(void) const {return n_color;}

  	  /*!
  	   * @return the node's label.
  	   */

  	  Label const& label(void) const {/*std::cout << "DATA = " << data << std::endl;*/ return data;}

  	  /*!
  	   * @par Node degree
  	   * The number of edges (alternatively adjacent nodes). There are two different types: In-edges are those that point in from an adjacent node \e to this node, while out-edges are those that point out toward an adjacent node \e from this node.
  	   *
  	   * @param out the number of out-edges if true (default), else in-edges.
  	   * @return the number of adjacent nodes (edges).
  	   */

  	  int degree(bool out=true) const {return out ? n_odeg : n_ideg;}

  	  /*!
  	   * @return true if node has no edges.
  	   */

  	  bool empty(void) const {return !degree();}

  	  /*!
  	   * @par Adopt a node
  	   * Node adoption splices a new node into the current list of edges or generates a singleton edge if the node is empty. Runs in constant time.
  	   *
  	   * @param node target.
  	   * @param weight of new edge.
  	   */

      void adopt(Node& node, Weight const& weight=Weight()) {
      	if(first == NULL) {
        	last = new Edge(weight);
          first = last;
        } else {
          last->next = new Edge(weight);
          last->next->prev = last;
          last = last->next;
        }

        last->node = &node;

        // std::cout << "adopted node = " << last->node->label() << std::endl;

        last->next = NULL;

        n_odeg++;
        node.n_ideg++;
      }

      /*
       * @abstract Overloaded method to remove an iterator (edge) but not its adjacent node!
       *
       * @param edge iterator to be orphaned
       * @return incremented iterator
       */

      iterator orphan(iterator& edge) {

      	/*
      	 * @abstract Spice out edge from front or increment (if first).
      	 */

      	if(edge != begin())
      	  (*edge).prev->next = (*edge).next;
      	else
      	  first = first->next;

      	/*
      	 * @abstract Splice out edge from behind or decrement (if last).
      	 */

      	if((*edge).next != NULL)
      	  (*edge).next->prev = (*edge).prev;
		    else
		      last = last->prev;

      	/*
      	 * @abstract Decrement out- and in-degrees
      	 */

      	n_odeg--;
      	(*edge).node->n_ideg--;

      	/*
      	 * @abstract Increment past the doomed edge and delete it. Return a valid iterator to the next edge.
      	 */

      	iterator tmp(edge);
      	++tmp;
      	delete &*edge;
      	return tmp;
      }

      bool adjacent(Node const& node) const {
      	iterator edge = begin();
      	while(edge != end()) {
      		if((*edge).node == &node)
      			return true;
      		++edge;
      	}
      	return false;
      }

      /*
       * @abstract Overloaded method to find an iterator to the adjacent node and remove it from the list of edges.
       *
       * @param node to be removed
       * @return incremented iterator if target is found, else end of the edge list
       */

      iterator orphan(Node const& node) {
      	iterator edge = begin();
        while(edge != end()) {
          if((*edge).node == &node) {
            return orphan(edge);
          }
          ++edge;
        }
      	return end();
      }

      /*
       * @abstract Overloaded ostream insertion operator.
       *
       * @param out stream
       * @param node data to insert
       * @return updated stream
       */

      /*
	    std::ostream& operator<<(std::ostream& out) {
	      out << data;
	      return out;
	    }
	    */

      void clear()
      {
      	iterator edge = begin();
      	while(edge != end()) {
      		orphan(edge);
      	  ++edge;
      	}
      }

  	  ~Node(void) {

  	  }

    protected:
  	  // Node(void) {};

      Label data;
    	int n_color, n_ideg, n_odeg;
    	Edge *first, *last;

    	friend class boost::serialization::access;

    	template<class Archive>
    	void serialize(Archive &archive, const unsigned verstion) {
    		std::cout << "SERIALIZE NODE ... ";
    		archive & data;
    		archive & n_color;
    		archive & n_ideg;
    		archive & n_odeg;
    		archive & first;
    		archive & last;
    		std::cout << "DONE" << std::endl;
    	}

  };


  template<class Label, class Weight>
  inline std::ostream& operator<<(std::ostream& out, Node<Label, Weight> const& node) {
  	return out << node.label();
  }

}

#endif /* NODE_HPP_ */
