/*        File: NodeTest.hpp
 *  Created on: May 4, 2010
 *      Author: Caleb J Kennedy (caleb.kennedy@lifetech.com)
 * 
 *  Latest revision:  $Revision: $
 *  Last changed by:  $Author: kennedcj $
 *  Last change date: $Date: $
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef NODE_TEST_H_
#define NODE_TEST_H_ 1

#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>
#include "node.hpp"
#include <string>

class NodeTest : public CppUnit::TestFixture
{
  public:

    static CppUnit::Test *suite()
    {
      CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("NodeTest");

      suiteOfTests->addTest(new CppUnit::TestCaller<NodeTest>("edgeTest", &NodeTest::edgeTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<NodeTest>("configTest", &NodeTest::configTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<NodeTest>("colorTest", &NodeTest::colorTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<NodeTest>("adjacentTest", &NodeTest::adjacentTest));
      suiteOfTests->addTest(new CppUnit::TestCaller<NodeTest>("iteratorTest", &NodeTest::iteratorTest));

      return suiteOfTests;
    }

    void setUp()
    {
    	m_strData.push_back("NodeA");
    	m_strData.push_back("NodeB");
    	m_strData.push_back("NodeC");

    	root = new lifetechnologies::Node<std::string>("Root");

    	nodes = new lifetechnologies::Node<std::string>*[m_strData.size()];
      for(unsigned i = 0; i < m_strData.size(); i++)
      	nodes[i] = new lifetechnologies::Node<std::string>(m_strData[i]);
    }

    void tearDown()
    {
    	for(unsigned i = 0; i < m_strData.size(); i++)
    		delete nodes[i];

    	delete nodes;
    	delete root;
    }

    void baseline()
    {
    	CPPUNIT_ASSERT(root->degree() == 0);      // Out-degrees.
    	CPPUNIT_ASSERT(root->degree(false) == 0); // In-degrees.
    	CPPUNIT_ASSERT(root->empty());
    	CPPUNIT_ASSERT(!root->color());						// Black.

    	for(unsigned i = 0; i < m_strData.size(); i++) {
    	  CPPUNIT_ASSERT(nodes[i]->degree() == 0);
    	  CPPUNIT_ASSERT(nodes[i]->degree(false) == 0);
    	  CPPUNIT_ASSERT(nodes[i]->empty());
    	  CPPUNIT_ASSERT(!nodes[i]->color());
    	}
    }

    void edgeTest()
    {
      baseline();

      /*
       * Root adopts each node. In-degree for each adopted node is 1.
       */

      for(unsigned i = 0; i < m_strData.size(); i++) {
        root->adopt(*nodes[i]);
        CPPUNIT_ASSERT(nodes[i]->degree(false));
      }

      CPPUNIT_ASSERT(root->degree() == m_strData.size());

      /*
       * Root orphans each node.
       */

      for(unsigned i = 0; i < m_strData.size(); i++)
      	root->orphan(*nodes[i]);

      CPPUNIT_ASSERT(root->degree() == 0);
      CPPUNIT_ASSERT(root->empty());

      /*
       * Retest.
       */

      for(unsigned i = 0; i < m_strData.size(); i++) {
        root->adopt(*nodes[i]);
        CPPUNIT_ASSERT(nodes[i]->degree(false));
      }

      CPPUNIT_ASSERT(root->degree() == m_strData.size());

      for(unsigned i = 0; i < m_strData.size(); i++)
      	root->orphan(*nodes[i]);

      baseline();
    }

    void configTest()
    {
    	baseline();

    	/*
    	 * Make a small graph.
    	 */

    	root->adopt(*nodes[0]);
    	root->adopt(*nodes[1]);
    	nodes[0]->adopt(*nodes[2]);
    	nodes[1]->adopt(*nodes[2]);

    	CPPUNIT_ASSERT(root->degree() == 2);
    	CPPUNIT_ASSERT(nodes[0]->degree());
    	CPPUNIT_ASSERT(nodes[1]->degree());
    	CPPUNIT_ASSERT(nodes[2]->degree() == 0);
    	CPPUNIT_ASSERT(nodes[2]->degree(false) == 2);

    	for(unsigned i = 0; i < m_strData.size(); i++)
    		nodes[i]->clear();

    	root->clear();

    	baseline();
    }

    void colorTest()
    {
    	baseline();

    	root->color(256);
    	CPPUNIT_ASSERT(root->color() == 256);

    	root->color(0);

    	baseline();
    }

    void adjacentTest()
    {
    	baseline();

    	root->adopt(*nodes[0]);
    	nodes[0]->adopt(*nodes[1]);
    	CPPUNIT_ASSERT(root->adjacent(*nodes[0]) && !(nodes[0]->adjacent(*root)));

    	nodes[0]->clear();
    	root->clear();

    	baseline();
    }

    void iteratorTest()
    {
    	baseline();

    	unsigned i = 0;
    	for(; i < m_strData.size(); i++)
        root->adopt(*nodes[i], 1);

    	/*
    	 * Forward iteration.
    	 */

    	i = 0;
			lifetechnologies::Node<std::string>::iterator rootItr = root->begin();

    	while(rootItr != root->end()) {
    		CPPUNIT_ASSERT((*rootItr).target().label() == m_strData[i++]);
    		CPPUNIT_ASSERT((*rootItr).weight());
    		++rootItr;
    	}

    	CPPUNIT_ASSERT(i == m_strData.size());

    	/*
    	 * Reverse iteration.
    	 */

    	i = m_strData.size();
    	rootItr = root->rbegin();

    	while(rootItr != root->rend()) {
    		CPPUNIT_ASSERT((*rootItr).target().label() == m_strData[i-- - 1]);
    		CPPUNIT_ASSERT((*rootItr).weight());
    		++rootItr;
      }

    	CPPUNIT_ASSERT(!i);

    	/*
    	 * Remove internal node.
    	 */

    	root->orphan(*nodes[1]);

    	i = 0;
    	rootItr = root->begin();

    	while(rootItr != root->end()) {
    		++i;
    		++rootItr;
    	}

    	CPPUNIT_ASSERT(i == root->degree());
    	CPPUNIT_ASSERT(i == m_strData.size() - 1);
    	CPPUNIT_ASSERT((*root->begin()).target().label() == m_strData.front());
    	CPPUNIT_ASSERT((*root->rbegin()).target().label() == m_strData.back());

    	/*
    	 * Clear root by iterator-orphaning.
    	 */

    	rootItr = root->begin();

    	while(!root->empty()) {
    		root->orphan(rootItr);
    		++rootItr;
    	}

    	baseline();
    }

  private:
    lifetechnologies::Node<std::string> *root, **nodes;
    std::vector<std::string> m_strData;
};

#endif // NODE_TEST_H_
