/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * test.cpp
 *
 *  Created on: Mar 22, 2010
 *      Author: kennedcj
 */

#include "graph.hpp"
#include <string>

using namespace relation;

typedef Graph<std::string> GRAPH;
typedef None<std::string, int> ALL;
typedef Print<std::string, int> PRINT;
typedef std::stack< GRAPH::iterator<PRINT> > PATH;

int main(int argc, char *argv[]) {

	Node<std::string, int> root("root"),
												 node1("A"),
												 node2("B"),
												 node3("C");

	root.adopt(node1, 0);
	root.adopt(node2, 0);
	root.adopt(node3, 0);

	Node<std::string, int>::iterator neighbor = root.begin();
	while(neighbor != root.end()) {
		std::cout << "neighbor = " << (*neighbor).target().label() << " ";
		++neighbor;
	}
	std::cout << std::endl;

	std::cout << "root.degree() = " << root.degree() << std::endl;

	/*
	 * Orphan edges back-to-front.
	 */

	neighbor = root.rbegin();
	while(!root.empty()) {
		std::cout << "root.orphan(" << (*neighbor).target().label() << ")" << std::endl;
		root.orphan(neighbor);
		++neighbor;
	}

  std::cout << "root.degree() = " << root.degree() << std::endl;

  root.adopt(node1, 0);
  root.adopt(node2, 0);
  root.adopt(node3, 0);

  neighbor = root.begin();
	while(neighbor != root.end()) {
		std::cout << "neighbor = " << (*neighbor).target().label() << " weight = " << (*neighbor).weight() << " ";
		++neighbor;
	}
	std::cout << std::endl;

	std::cout << "root.degree() = " << root.degree() << std::endl;

	/*
	 * Orphan edges front-to-back.
	 */

	neighbor = root.begin();
	while(!root.empty()) {
		std::cout << "root.orphan(" << (*neighbor).target().label() << ")" << std::endl;
		root.orphan(neighbor);
		++neighbor;
	}

  std::cout << "root.degree() = " << root.degree() << std::endl;

  GRAPH graph(GRAPH::SIMPLE, GRAPH::DIRECTED);
  GRAPH::iterator<> A = graph.insert(graph.begin<ALL>(), "A");
  GRAPH::iterator<> B = graph.insert(A, "B");
  GRAPH::iterator<> C = graph.insert(A, "C");
  GRAPH::iterator<> D = graph.insert(A, "D");
  GRAPH::iterator<> E = graph.insert(B, "E"); graph.insert(C, E);
  GRAPH::iterator<> F = graph.insert(B, "F"); graph.insert(D, F);
  GRAPH::iterator<> G = graph.insert(C, "G"); graph.insert(D, G);
  GRAPH::iterator<> H = graph.insert(E, "H"); graph.insert(F, H); graph.insert(G, H);

  // Print<std::string, int> print;

  PRINT print(PRINT::DOT, "test_graph.dot");

  std::cout << "---TRAVERSE!!!---" << std::endl;

  PATH path = graph.traverse(graph.find("A", print), graph.end(print));

  std::cout << "---TRAVERSE---" << std::endl;

  std::cout << "number of nodes = " << graph.order() << " number of edges = " << graph.size() << std::endl;

  std::cout << "PATH = {";
  while(!path.empty()) {
  	std::cout << (*path.top()).label();
  	path.pop();
  }
  std::cout << "}" << std::endl;

  std::cout << "DESTRUCTING" << std::endl;

  return 0;
}
