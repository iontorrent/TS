/*        File: GraphTest.hpp
 *  			Date: May 6, 2010
 *      Author: Caleb J Kennedy (caleb.kennedy@lifetech.com)
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef GRAPH_TEST_HPP_
#define GRAPH_TEST_HPP_ 1

#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>
#include "graph.hpp"
#include "prune.hpp"
#include <string>
#include <map>

typedef lifetechnologies::Graph<std::string> STRING_GRAPH;
typedef lifetechnologies::Print<std::string> PRINT_STRING;
typedef STRING_GRAPH::iterator<PRINT_STRING> STRING_GRAPH_ITERATOR;
typedef std::stack<STRING_GRAPH_ITERATOR> PATH;

typedef lifetechnologies::Graph<int> INT_GRAPH;
typedef lifetechnologies::None<int> ALL_INT;
typedef INT_GRAPH::iterator<ALL_INT> INT_GRAPH_ITERATOR;

class GraphTest : public CppUnit::TestFixture
{
  public:

    static CppUnit::Test *suite()
    {
      CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("GraphTest");

      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("graphInsertTest", &GraphTest::graphInsertTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("diGraphInsertTest", &GraphTest::diGraphInsertTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("completeGraphTest", &GraphTest::completeGraphTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("findTest", &GraphTest::findTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("traversalTest", &GraphTest::traversalTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("iteratorTest", &GraphTest::iteratorTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("printTest", &GraphTest::printTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<GraphTest>("printPrunerTest", &GraphTest::printPrunerTest));

      return suiteOfTests;
    }

    void setUp()
    {
    	//m_strPrint = new PRINT_STRING(PRINT_STRING::DOT, "wonderland.dot");
    	/*
    	 * Make a small but non-trivial graph: a Boolean lattice.
    	 */

    	m_strGraph = new STRING_GRAPH;

    	/*
    	 * The logical statements are all possible answers to the question: 'Who stole the tarts made by the Queen of Hearts?' (See Knuth KH.
    	 * Intelligent machines in the 21st century: foundations of inference and inquiry. (2003) Phil Trans Roy Soc Lond A; 361(1813):2859-73.)
    	 */

    	std::string alice               = "\"Alice stole the tarts!\"";
    	std::string knave               = "\"The Knave of Hearts stole the tarts!\"";
    	std::string noOne               = "\"No one stole the tarts!\"";
    	std::string aliceOrKnave        = "\"Alice or the Knave stole the tarts!\"";
    	std::string aliceOrNoOne        = "\"Alice or no one stole the tarts!\"";
    	std::string knaveOrNoOne        = "\"The Knave or no one stole the tarts!\"";
    	std::string aliceOrKnaveOrNoOne = "\"Alice or the Knave or no one stole the tarts!\"";

    	/*
    	 * Insert the basic assertions (atoms) ...
    	 */

			m_graphItrs["\" \""] = m_strGraph->insert(m_strGraph->begin<PRINT_STRING>(), "\" \"");
    	m_graphItrs[alice]  = m_strGraph->insert(m_graphItrs.find("\" \"")->second, alice);
    	m_graphItrs[knave]  = m_strGraph->insert(m_graphItrs.find("\" \"")->second, knave);
    	m_graphItrs[noOne]  = m_strGraph->insert(m_graphItrs.find("\" \"")->second, noOne);

    	/*
    	 * ... and their logical disjunctions.
    	 */

    	m_graphItrs[aliceOrKnave] = m_strGraph->insert(m_graphItrs.find(alice)->second, aliceOrKnave);
    	  m_strGraph->insert(m_graphItrs.find(aliceOrKnave)->second, m_graphItrs.find(knave)->second);

    	m_graphItrs[aliceOrNoOne] = m_strGraph->insert(m_graphItrs.find(alice)->second, aliceOrNoOne);
    	  m_strGraph->insert(m_graphItrs.find(aliceOrNoOne)->second, m_graphItrs.find(noOne)->second);

    	m_graphItrs[knaveOrNoOne] = m_strGraph->insert(m_graphItrs.find(knave)->second, knaveOrNoOne);
    	  m_strGraph->insert(m_graphItrs.find(knaveOrNoOne)->second, m_graphItrs.find(noOne)->second);

    	m_graphItrs[aliceOrKnaveOrNoOne] = m_strGraph->insert(m_graphItrs.find(aliceOrKnave)->second, aliceOrKnaveOrNoOne);
    	  m_strGraph->insert(m_graphItrs.find(aliceOrKnaveOrNoOne)->second, m_graphItrs.find(aliceOrNoOne)->second);
    	  m_strGraph->insert(m_graphItrs.find(aliceOrKnaveOrNoOne)->second, m_graphItrs.find(knaveOrNoOne)->second);
    }

    void tearDown()
    {
    	delete m_strGraph;
    }

    void graphInsertTest()
    {
    	STRING_GRAPH graph;
    	STRING_GRAPH_ITERATOR graphItr = graph.insert(graph.begin<PRINT_STRING>(), "NodeA");

    	CPPUNIT_ASSERT(graph.size() == 0);
    	CPPUNIT_ASSERT(graph.order() == 1);
    	CPPUNIT_ASSERT((*graphItr).label() == "NodeA");

    	graphItr = graph.insert(graphItr, "NodeB");

    	CPPUNIT_ASSERT(graph.size() == 1);
    	CPPUNIT_ASSERT(graph.order() == 2);
    	CPPUNIT_ASSERT((*graphItr).label() == "NodeB");

    	CPPUNIT_ASSERT((*graph.begin<PRINT_STRING>()).adjacent(*graphItr) && (*graphItr).adjacent(*graph.begin<PRINT_STRING>()));
    }

    void diGraphInsertTest()
    {
    	STRING_GRAPH graph(STRING_GRAPH::SIMPLE, STRING_GRAPH::DIRECTED);
    	STRING_GRAPH_ITERATOR graphItr = graph.insert(graph.begin<PRINT_STRING>(), "NodeA");

    	CPPUNIT_ASSERT(graph.size() == 0);
    	CPPUNIT_ASSERT(graph.order() == 1);
    	CPPUNIT_ASSERT((*graphItr).label() == "NodeA");

    	graphItr = graph.insert(graphItr, "NodeB");

    	CPPUNIT_ASSERT(graph.size() == 1);
    	CPPUNIT_ASSERT(graph.order() == 2);
    	CPPUNIT_ASSERT((*graphItr).label() == "NodeB");

    	CPPUNIT_ASSERT((*graph.begin<PRINT_STRING>()).adjacent(*graphItr) && !(*graphItr).adjacent(*graph.begin<PRINT_STRING>()));
    }

    void completeGraphTest()
    {
    	/*
    	 * Node ids.
    	 */

    	int nodes[5] = {0, 1, 2, 3, 4};

    	/*
    	 * Make a simple, undirected graph with five nodes. Store graph iterators (Nodes) in an array.
    	 */

    	INT_GRAPH graph;
    	INT_GRAPH_ITERATOR* graphItrs = new INT_GRAPH_ITERATOR[5];

    	/*
    	 * Label each Node with its id and insert it into the graph. The first Node (id = 0) becomes root. The rest are adjacent to root.
    	 */

    	for(unsigned i = 0; i < 5; i++)
    		graphItrs[i] = graph.insert(graph.begin<ALL_INT>(), nodes[i]);

    	CPPUNIT_ASSERT(graph.order() == 5);

    	/*
    	 * Connect each Node to every other Node (ie make a complete graph with five nodes: the 'star').
    	 */

    	for(unsigned i = 1; i < 5; i++)
    		for(unsigned j = 4; j > i; j--)
    			graph.insert(graphItrs[i], graphItrs[j]);

    	CPPUNIT_ASSERT(graph.size() == 10);

    	/*
    	 * Test the graph for proper Node degrees and adjacency.
    	 */

    	for(unsigned i = 0; i < 5; i++) {
    		INT_GRAPH_ITERATOR graphItr = graphItrs[i];

    		CPPUNIT_ASSERT((*graphItr).degree() == 4);

    		lifetechnologies::Node<int>::iterator edge = (*graphItr).begin();
    		while(edge != (*graphItr).end()) {
    			if((*graphItr).label() != nodes[i])
    				CPPUNIT_ASSERT((*edge).target().label() == nodes[i]);

    			++edge;
    		}
    	}
    }

    void findTest()
    {
    	STRING_GRAPH_ITERATOR graphItr1 = m_strGraph->find<PRINT_STRING>("\"Alice stole the tarts!\"");
    	STRING_GRAPH_ITERATOR graphItr2 = m_strGraph->find<PRINT_STRING>("\"Alice or the Knave stole the tarts!\"");
    	STRING_GRAPH_ITERATOR graphItr3 = m_strGraph->find<PRINT_STRING>("\"The Queen of Hearts stole the tarts!\"");
    	STRING_GRAPH_ITERATOR graphEnd  = m_strGraph->end<PRINT_STRING>();

			CPPUNIT_ASSERT(graphItr1 != graphItr2);
			CPPUNIT_ASSERT(graphItr1 != graphEnd && graphItr2 != graphEnd);                    // Catch possible seg faults.
    	CPPUNIT_ASSERT((*graphItr1).label() == "\"Alice stole the tarts!\"");              // Found.
    	CPPUNIT_ASSERT((*graphItr2).label() == "\"Alice or the Knave stole the tarts!\""); // Found.
    	CPPUNIT_ASSERT(!(graphItr3 != graphEnd));                                          // Not found.
    }

    void traversalTest()
    {
    	PATH path = m_strGraph->traverse(m_strGraph->begin<PRINT_STRING>(), m_strGraph->end<PRINT_STRING>());

    	CPPUNIT_ASSERT(path.size() == 8);

    	while(!path.empty()) {
    		CPPUNIT_ASSERT(m_graphItrs.find((*path.top()).label()) != m_graphItrs.end());
    		path.pop();
    	}
    }

    void iteratorTest()
    {
    	unsigned nNodes = 0;
    	STRING_GRAPH_ITERATOR graphItr1 = m_strGraph->begin<PRINT_STRING>(STRING_GRAPH::TRAVERSE), last = m_strGraph->end<PRINT_STRING>();

    	while(graphItr1 != last) {
    		CPPUNIT_ASSERT(m_graphItrs.find((*graphItr1).label()) != m_graphItrs.end());
    		++graphItr1;
    		++nNodes;
    	}

    	CPPUNIT_ASSERT(nNodes == 8);
    }

    void printTest()
    {
    	PRINT_STRING print(PRINT_STRING::DOT, "wonderland.dot");

    	PATH path = m_strGraph->traverse(m_strGraph->begin(true, print), m_strGraph->end(print));
    }

    void printPrunerTest()
    {

    }

  private:
    PRINT_STRING m_strPrint;
    STRING_GRAPH *m_strGraph;
    std::map<std::string, STRING_GRAPH_ITERATOR> m_graphItrs;
};

#endif
