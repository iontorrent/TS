/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * graph.hpp
 *
 *  Created on: Feb 28, 2010
 *      Author: kennedcj
 */

#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include "prune.hpp"
#include <stack>

namespace relation {

	/*
	 * Define 'Label' and 'Weight' classes for each node in the graph and the 'Compare' class for inequality relations between nodes.
	 */

  template< class Label, class Weight=int, class Compare=std::less<Label> >

  /*
   * @abstract Graphs are collections of labeled nodes connected by weighted edges. Nodes may inserted into graphs arbitrarily (defined here)
   * or according to some order (or partial order) defined by methods within 'Compare'. The iterator class provides standardized methods for
   * incremental access to nodes within the graph. By default graphs are unweighted (integer types with default values zero).
   */

  class Graph {
    public:

  		/*
  		 * @abstract Specific kinds of graphs.
  		 *
  		 * @enum SIMPLE Restrict self-incident edges (loops) and multi-edges connecting two nodes with more than one edge.
  		 * @enum DIRECTED Each node is confined to a strict parent-child relationship. No child node can point to its parent directly, although indirect association is possible if the graph is cyclic.
  		 * @enum TRAVERSE Primes the iterator for traversal (changes the graphs color).
  		 */
  	  enum Kind {SIMPLE=1, DIRECTED=1, TRAVERSE=1};

  	  /*
  	   * @abstract Constructor to generate a simple undirected empty graph.
  	   */

	    Graph(bool s=SIMPLE, bool d=!DIRECTED) : simple(s), directed(d), g_color(BLACK), g_size(0), g_order(0), g_root(NULL) {};

      /*
       * @abstract The graph class can be considered a container of nodes connected to each other by weighted edges. The iterator
       * class defined here encapsulates pointer manipulation and provides methods for incrementally accessing individual nodes and
       * their associated edges (adjacent nodes). Iterators represent the 'client' interface to graph objects. Each iterator is
       * populated with a 'Pruner' object, a functor that determines which nodes to ignore (prune) and may be used to process individual
       * nodes during iteration. Non-visited (uncolored) nodes are stored on a stack and iteration proceeds as a depth-first-search over
       * the (unpruned) nodes in the graph.
       *
       * @example
       * @code
       * ...
       * @endcode
       * @endexample
       */

	    template< class Pruner=None<Label, Weight> >
  	  class iterator : public std::iterator< std::input_iterator_tag, Node<Label, Weight> > {
  	    public:
  	  	  iterator(void) : i_color(BLACK), prune(Pruner()) {
  	  	  	colored.push(NULL);
  	  	  }

  	  	  /*
  	  	   * @abstract Constructor to generate a new graph iterator from a node pointer.
  	  	   *
  	  	   * @param p pruner object
  	  	   * @param c color of the iterator
  	  	   * @param n node pointer
  	  	   */

  	  	  iterator(const Pruner& p, unsigned c, Node<Label, Weight>* n=NULL) : i_color(c), prune(p) {
  	  		  //std::cout << "Pruner kind = " << prune.getKind() << std::endl;
	    		  //if(n != NULL) std::cout << "iterator(" << c << ", " << *n << ")" << std::endl;
	    		  // colored.push(NULL);
	    		  times = 0;
	    		  colored.push(n);
	    	  }

  	  	  /*
  	  	   * @return a stack of iterators representing the range of traversed nodes from first to the last (exclusive)
  	  	   */

  	  	  std::stack<iterator> path(void) const {return i_path;};

  	  	  int color(void) const {return i_color;}

  	  	  Pruner const& pruner(void) const {return prune;}

  	  	  /*
  	  	   * @abstract Increment (prefix).
  	  	   *
  	  	   * @return reference to next iterator on the stack.
  	  	   */

	    	  iterator& operator++(void) {
	    		  ++times;
	    		  std::cout << "times = " << times << std::endl;
	    		  Node<Label, Weight>* node = colored.top();
	    		  // std::cout << "node->color(" << color << ")" << std::endl;
	    		  node->color(i_color);

  	    		// std::cout << "prune(" << *colored.top() << ", " << color << ")" << std::endl;
	      		std::cout << "if(" << prune.getKind() << "operator()(" << node->label() << "))" << std::endl;
	      	  if(prune(*node))
	      	  	return *this;

	    	    i_path.push(iterator(Pruner(), i_color, node));
	    		  colored.pop();

	    		  /*
	    		  if(node->empty()) {

	    			  colored.pop();
	    		  }
	    		  */
	    		  // else {
	    		    typename Node<Label, Weight>::iterator edge = node->begin(), last = node->end();
	    		    while(edge != last) {
	    		  	  std::cout << "if(" << (*edge).node->color() << " != " << i_color << " && ";
	    		  	  //if(prune == NULL)
	    		  		//std::cout << "NULL";
	    		  	//else
	    		  		//std::cout << "!prune(" << (*edge).target().label() << "))" << std::endl;

	    		    	if((*edge).node->color() != i_color && !prune(edge)) {
	    		    		std::cout << (*edge).node->label() << "->color(" << i_color << ")" << std::endl;

	    		  	  	(*edge).node->color(i_color);

	    		  	  	std::cout << "colored.push(" << (*edge).node->label() << ")" << std::endl;

	    			      colored.push((*edge).node);
	    			    }
	    			    ++edge;
	    		    }
	    		  // }

	    		    // if(colored.size() == 10) exit(1);


	    		  std::stack< Node<Label, Weight>* > remains = colored;
	    		  std::cout << "{";
	    		  while(!remains.empty()) {
	    		  	//std::cout << remains.top()->label() << " ";
	    			  remains.pop();
	    		  }

	    		  std::cout << "}" << std::endl;

	    		  if(colored.empty()) {
	    			  // std::cout << "return iterator()" << std::endl;

	    	  	  colored.push(NULL);
	    		  }

	    	    return *this;
	    	  }

  	  	  /*
  	  	   * @abstract Relational operator tests for non-equivalence between two nodes.
  	  	   *
  	  	   * @param right iterator to compare
  	  	   * @return true if *this != right
  	  	   */

	    	  bool operator!=(iterator const& right) const {
	    	  	// if(times == 10) exit(1);

	    		  // std::cout << "Graph::operator!=(";

	    		// std::cout << (colored.empty() ? "" : "!") << "colored.empty(),";




	    		// std::cout << ",";



	    		// std::cout << ")" << std::endl;

	    		  if(colored.top() == NULL)
	    			  std::cout << "NULL";
	    		  else
	    			  std::cout << *colored.top();

	    		  std::cout << " != ";

	    		  if(right.colored.top() == NULL)
	    			  std::cout << "NULL";
	    		  else
	    			  //std::cout << right.colored.top()->label();

	    		  std::cout << std::endl;

	    		  return colored.top() != right.colored.top();
	    	  }

  	  	  /*
  	  	   * @abstract Dereference operator.
  	  	   *
  	  	   * @return the node at the top of the stack
  	  	   */

  	    	Node<Label, Weight>& operator*(void) const {
	      		return *colored.top();
	      	}

	      	// Pruner const& pruner(void) const {return prune;}


	      	unsigned times;

  	    private:
	      	friend class Graph;

	      	int i_color;
		      std::stack< Node<Label, Weight>* > colored;
		      std::stack<iterator> i_path;
		      Pruner prune;
  	  };

  	  /* @abstract Standard access iterators.
  	   *
  	   * @param traverse change graph color to prime iterator for traversal if true
  	   * @param prune pruner (prune nothing by default)
  	   * @return an iterator to the first node in the graph
  	   */

  	  template<class Pruner>
  	  iterator<Pruner> begin(bool traverse=!TRAVERSE, const Pruner& prune=Pruner()) {
  	  	std::cout << "begin(" << traverse << ", " << prune.getKind() << ")" << std::endl;
  	  	return iterator<Pruner>(prune, traverse ? ++g_color : g_color, g_root);
  	  }

  	  /*
  	   * @return the last node in the graph (NULL)
  	   */

  	  template<class Pruner>
  	  iterator<Pruner> end(const Pruner& prune=Pruner()) {
  	  	return iterator<Pruner>(prune, g_color);
  	  }

  	  /*
  	   * @return the number of nodes
  	   */

  	  int order(void) const {return g_order;};

  	  /*
  	   * @return the number of edges
  	   */

  	  int size(void) const {return g_size;};

  	  /*
  	   * @return true if graph is empty (zero nodes)
  	   */

  	  bool empty(void) const {return !order();};

  	  /*
  	   * @abstract Overloaded method to connect two nodes (represented by iterators) with a new weighted edge.
  	   *
  	   * @param parent iterator
  	   * @param child iterator
  	   * @param weight of edge between parent and child
  	   * @return nothing
  	   */

  	  template<class Pruner>
	    void insert(iterator<Pruner> parent, iterator<Pruner> child, Weight const& weight=Weight()) {
	    	(*parent).adopt(*child, weight);

	    	/*
	    	 * @abstract Reciprocate adjacency for directed graphs and increment number of edges.
	    	 */

	    	if(!directed)
	    		(*child).adopt(*parent, weight);

	    	++g_size;
	    }

  	  /*!
  	   * @abstract Overloaded method to create a new labeled node and connected it to the graph at a position specified by parent.
  	   *
  	   * @param parent inserts new node here
  	   * @param label for the new node
  	   * @param prune pruner passed to returned iterator (default does no pruning)
  	   * @param weight of edge between parent and new node
  	   * @return iterator to new inserted node
  	   *
  	   * @example
  	   * \code
  	   *
  	   * #include <string>
  	   *
  	   *
  	   *
  	   * typedef Graph<std::string> GRAPH;
			 * typedef None<std::string, int> ALL;
			 *
			 *
  	   *
  	   * \endcode
  	   * @endexample
  	   */

  	  template<class Pruner>
	    iterator<Pruner> insert(iterator<Pruner> parent, Label const& label, Pruner const& prune=Pruner(), Weight const& weight=Weight()) {
	    	iterator<Pruner> child(prune, g_color, new Node<Label, Weight>(label));

	    	if(parent != end<Pruner>())
	    		insert(parent, child, weight);
	    	else
	    		if(!empty())
	    			insert(iterator<Pruner>(prune, g_color, g_root), child, weight);
	    			// g_root->adopt(*child, weight);
	    		else
	    			g_root = &*child;

	    	++g_order;
	    	return child;


	    	//// std::cout << "child = " << *child << std::endl;
	    	//if(parent != end()) {
	    		//std::cerr << "parent = ";
	    		//std::cerr << *parent << ").adopt(" << *child << ", " << weight << ")" << std::endl;

		    	//insert(parent, child, weight);
		    	//return child;
	    	//}

    		//g_root = &*child;
    		//// // std::cout << "g_root = " << *g_root << std::endl;
    		//return begin();
	    }

  	  template<class Pruner>
  	  std::stack< iterator<Pruner> > traverse(iterator<Pruner> first, iterator<Pruner> last) {
  	  	std::cout << "first color = " << first.i_color << " last.color = " << last.i_color << std::endl;

  	  	iterator<Pruner> start(first.prune, ++g_color, &*first);
  	  	while(start != last) ++start;
  	  	return start.path();
  	  }

  	  template<class Pruner>
  	  iterator<Pruner> find(const Label& label, const Pruner &prune=Pruner()) {

  	  	iterator<> found = begin< None<Label, Weight> >(TRAVERSE), last = end< None<Label, Weight> >();
  	  	while(found != last) {
  	  		if((*found).label() == label) {
  	  			std::cout << "FOUND " << label << std::endl;
  	  			break;
  	  		}
  	  		++found;
  	  	}

  	  	return iterator<Pruner>(prune, g_color, &*found);
  	  }

  	  template<class Pruner>
  	  void erase(iterator<Pruner> first, iterator<Pruner> last) {
  	  	std::stack< Node<Label, Weight>* > doomed;
  	  	iterator<Pruner> start(first.prune, ++g_color, &*first);

  	  	while(start != last) {
  	    	typename Node<Label, Weight>::iterator edge = (*start).begin(), last = (*start).end();
  	    	doomed.push(&*start);
  	    	//std::cout << "doomed = " << *start << std::endl;
  	    	++start;

  	    	while(edge != last) {
  	    		//std::cout << doomed.top()->label() << "->orphan(" << (*edge).target().label() << ")" << std::endl;
  	    		doomed.top()->orphan(edge);

  	    		if(!directed)
  	    			(*edge).node->orphan(*doomed.top());

  	    		--g_size;
  	    		++edge;
  	    	}
  	    	//Node<Label, Weight> *doomed = &*first;
  	    	//erase(first);
  	    	//++first;
  	    	//std::cout << "delete = " << *doomed << std::endl;
  	    	//delete doomed;

  	    	--g_order;
  	    	std::cout << "size = " << g_size << " order = " << g_order << std::endl;
  	    }

  	    while(!doomed.empty()) {
  	      delete doomed.top();
  	      doomed.pop();
  	    }
  	  }

  	  void erase(Label const& label) {
  	  	// TODO
  	  }

  	  /*
  	   * @abstract Graph destructor. Remove edges and delete nodes.
  	   */

	    virtual ~Graph(void) {
	    	erase(begin< None<Label, Weight> >(), end< None<Label, Weight> >());
  	  }

    protected:
	    bool simple, directed;
	    int g_color, g_size, g_order;
	    Node<Label, Weight> *g_root;
	    Compare compare;

    	friend class boost::serialization::access;

    	template<class Archive>
    	void serialize(Archive &archive, const unsigned verstion) {
    		archive & simple;
    		archive & g_color;
    		archive & g_size;
    		archive & g_order;
    		archive & g_root;
    		archive & compare;
    	}

  };
}

#endif /* GRAPH_HPP_ */
