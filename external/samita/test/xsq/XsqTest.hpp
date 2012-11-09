/*
 *  Created on:
 *      Author: Matthew Muller
 *
 *  Copyright 2010 Life Technologies. All rights reserved.
 *  Use is subject to license terms.
 */

#ifndef XSQ_TEST_HPP_
#define XSQ_TEST_HPP_

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>
#include <log4cxx/logger.h>
#include <boost/filesystem.hpp>
#include <tbb/tbb.h>
#include <tbb/task.h>
#include <tbb/partitioner.h>
#include <new>
#include <tbb/tbb_exception.h>
#include "samita/xsq/xsq_io.hpp"
#include "hdf5_hl.h"

using namespace std;
using namespace lifetechnologies;
using namespace tbb;

XsqMultiWriter *xxx;

static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.xsq");
const size_t max_size_t = std::numeric_limits<std::size_t>::max();

struct MyPanel : public PanelI {
	string filename;
	size_t pcIndex;
	uint32_t panelNum;

	MyPanel(string const& _filename, size_t const& _pcIndex, uint32_t const& _panelNum ) :
			filename(_filename), pcIndex(_pcIndex), panelNum(_panelNum) {}
	size_t getPanelNumber() const {return panelNum; }
	size_t size() const { return 1000; }
	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const { return 75; }
	std::string getPanelContainerName() const { return "pc"; }
	size_t getPanelContainerIndex() const { return pcIndex; }
	std::string getFilename() const { return filename; }
	const bool isReadTypeDataVirtual(XsqReadType const& readType) const { return false; }
};

template <class Value>
class IncrementingFunctor {
	size_t* m_count;
public :
	IncrementingFunctor(size_t &count) : m_count(&count) {}
	void operator()(Value const& value) {
		++(*m_count);
	}
};

inline std::ostream &operator<<(std::ostream &stream, QualityValueArray const& arr) {
	for (QualityValueArray::const_iterator it = arr.begin(); it != arr.end(); ++it)
		stream << char(*it + 33);
	return stream;
}

class XsqTest: public CppUnit::TestFixture
{
public:
    static CppUnit::Test *suite()
    {
    	CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("Test");
//Unit Tests These tests need to run on any system.  No dependencies on external files!!!!
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("scribble", &XsqTest::scribble));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("colorEncodingTest", &XsqTest::colorEncodingTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("inspectXsqTest", &XsqTest::inspectXsqContent));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("partialIterationTest", &XsqTest::partialIterationTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("splittingTest", &XsqTest::splittingTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("partitionTest", &XsqTest::partitionTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("xsqWriterTest", &XsqTest::xsqWriterTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("xsqMultiWriter", &XsqTest::xsqMultiWriter));
      	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("setCallsTest", &XsqTest::setCallsTest));
      	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("editInPlaceTest", &XsqTest::editInPlaceTest));

//System dependent tests
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("perfVsCsfastaTest", &XsqTest::perfVsCsfastaTest));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("perfXsqOnlyTest", &XsqTest::perfXsqOnlyTest));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("perfXsqIterationTest", &XsqTest::perfXsqIterationTest));

//Time consuming tests
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("writingInParallel", &XsqTest::writingInParallel));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("chengyongsFileTest", &XsqTest::chengyongsFileTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("sowmisFileTest", &XsqTest::sowmisFileTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bigXsqWriterTest", &XsqTest::bigXsqWriterTest));

//Debugging tests
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("zhengsBug", &XsqTest::zhengsBug));
	    suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug20100117", &XsqTest::bug20100117));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug20110126", &XsqTest::bug20110126));

//Demos
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("printCallQV", &XsqTest::printCallQV));

        return suiteOfTests;
    }
    void setUp()
    {
       //LOG4CXX_INFO(g_log, "setUp");
    }

    void tearDown()
    {
       //LOG4CXX_INFO(g_log, "tearDown");
    }

    string toBinary(uint8_t const& num) {
    	stringstream stream;
    	for (int i=7; i>=0; --i)
    		stream << ((num >> i) & 1);
    	return stream.str();
    }

    void printCallQV() {
    	for (uint8_t callqv = 0;;) {
    		 //uint8_t call = (callqv & 0xc0) >> 6; //0810
    		 //uint8_t qv = callqv & 0x3f; //0810
    	     //uint8_t call = callqv >> 6; // 0810simplified
    		 uint8_t call = callqv & 0x03; //BCL convention
    		 uint8_t qv = callqv >> 2; //BCL convention

    		 printf("%3d\t%s\t%1d\t%2d\n", callqv, toBinary(callqv).c_str(), (short)call, (short)qv);
    		if (callqv++ == 255) break;
    	}
    }

    void scribble() {

    }

    void panelRangeSpecifierTest() {
    	try {

			map<XsqReadType, size_t> readLengths;
			readLengths[XSQ_READ_TYPE_F3] = 75;
			readLengths[XSQ_READ_TYPE_F5] = 35;

			URL url0("file");
			CPPUNIT_ASSERT_EQUAL(string("file"), url0.getPath());
			URL url1("file?foo=bar");
			CPPUNIT_ASSERT_EQUAL(string("file"), url1.getPath());
			CPPUNIT_ASSERT_EQUAL(string("foo=bar"), url1.getQuery());
			CPPUNIT_ASSERT_EQUAL(string("bar"), url1.getParameter("foo"));

			PanelRangeSpecifier psr0("file1.xsq");

			PanelRangeSpecifier psr1("file2.xsq?tag=F3");
			CPPUNIT_ASSERT_EQUAL(string("file2.xsq"), psr1.getPath());
			CPPUNIT_ASSERT_EQUAL(XSQ_READ_TYPE_F3, psr1.getReadTypes()[0]);

			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(0), psr1.getPanelStart().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(0), psr1.getPanelStart().panelIndex);
			CPPUNIT_ASSERT_EQUAL(max_size_t, psr1.getPanelEnd().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(max_size_t, psr1.getPanelEnd().panelIndex);

			PanelRangeSpecifier psr2("file2.xsq?tag=F3&start=3.50");
			CPPUNIT_ASSERT_EQUAL(string("file2.xsq"), psr2.getPath());
			CPPUNIT_ASSERT_EQUAL(XSQ_READ_TYPE_F3, psr2.getReadTypes()[0]);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(3), psr2.getPanelStart().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(50), psr2.getPanelStart().panelIndex);
			CPPUNIT_ASSERT_EQUAL(max_size_t, psr2.getPanelEnd().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(max_size_t, psr2.getPanelEnd().panelIndex);

			PanelRangeSpecifier psr3("file2.xsq?tag=R3&start=3.50&end=10.11");
			CPPUNIT_ASSERT_EQUAL(string("file2.xsq"), psr3.getPath());
			CPPUNIT_ASSERT_EQUAL(XSQ_READ_TYPE_R3, psr3.getReadTypes()[0]);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(3), psr3.getPanelStart().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(50), psr3.getPanelStart().panelIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(10), psr3.getPanelEnd().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(11), psr3.getPanelEnd().panelIndex);
			CPPUNIT_ASSERT(!psr3(MyPanel("/file1.xsq", 3, 50)));

			PanelRangeSpecifier psr4("file:///file2.xsq?tag=R3&start=3.50&end=10.11");
			CPPUNIT_ASSERT_EQUAL(string("/file2.xsq"), psr4.getPath());
			CPPUNIT_ASSERT_EQUAL(XSQ_READ_TYPE_R3, psr4.getReadTypes()[0]);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(3), psr4.getPanelStart().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(50), psr4.getPanelStart().panelIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(10), psr4.getPanelEnd().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(11), psr4.getPanelEnd().panelIndex);
			CPPUNIT_ASSERT(!psr4(MyPanel("/file1.xsq", 3, 50)));
			CPPUNIT_ASSERT(!psr4(MyPanel("/file2.xsq", 2, 50)));
			CPPUNIT_ASSERT(!psr4(MyPanel("/file2.xsq", 3, 49)));
			CPPUNIT_ASSERT( psr4(MyPanel("/file2.xsq", 3, 50)));
			CPPUNIT_ASSERT( psr4(MyPanel("/file2.xsq", 10, 10)));
			CPPUNIT_ASSERT( psr4(MyPanel("/file2.xsq", 10, 11)));
			CPPUNIT_ASSERT(!psr4(MyPanel("/file2.xsq", 10, 12)));
			CPPUNIT_ASSERT(!psr4(MyPanel("/file2.xsq", 11, 10)));

			PanelRangeSpecifier psr5("file2.xsq?tag=F3&start=2&end=2");
			CPPUNIT_ASSERT_EQUAL(string("file2.xsq"), psr5.getPath());
			CPPUNIT_ASSERT_EQUAL(XSQ_READ_TYPE_F3, psr5.getReadTypes()[0]);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), psr5.getPanelStart().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(0), psr5.getPanelStart().panelIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), psr5.getPanelEnd().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(max_size_t, psr5.getPanelEnd().panelIndex);
			CPPUNIT_ASSERT(!psr5(MyPanel("file1.xsq", 1, 1)));
			CPPUNIT_ASSERT(!psr5(MyPanel("file2.xsq", 1, 1)));
			CPPUNIT_ASSERT(!psr5(MyPanel("file2.xsq", 1, max_size_t)));
			CPPUNIT_ASSERT( psr5(MyPanel("file2.xsq", 2, 0)));
			CPPUNIT_ASSERT( psr5(MyPanel("file2.xsq", 2, max_size_t)));
			CPPUNIT_ASSERT(!psr5(MyPanel("file2.xsq", 3, 0)));

			PanelRangeSpecifier psr6("file2.xsq?tag=F3&start=2&end=4");
			CPPUNIT_ASSERT_EQUAL(string("file2.xsq"), psr6.getPath());
			CPPUNIT_ASSERT_EQUAL(XSQ_READ_TYPE_F3, psr6.getReadTypes()[0]);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), psr6.getPanelStart().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(0), psr6.getPanelStart().panelIndex);
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(4), psr6.getPanelEnd().libraryIndex);
			CPPUNIT_ASSERT_EQUAL(max_size_t, psr6.getPanelEnd().panelIndex);
			CPPUNIT_ASSERT(!psr6(MyPanel("file1.xsq", 1, 1)));
			CPPUNIT_ASSERT(!psr6(MyPanel("file2.xsq", 1, 1)));
			CPPUNIT_ASSERT(!psr6(MyPanel("file2.xsq", 1, max_size_t)));
			CPPUNIT_ASSERT( psr6(MyPanel("file2.xsq", 2, 0)));
			CPPUNIT_ASSERT( psr6(MyPanel("file2.xsq", 2, max_size_t)));
			CPPUNIT_ASSERT( psr6(MyPanel("file2.xsq", 3, 0)));
			CPPUNIT_ASSERT( psr6(MyPanel("file2.xsq", 3, 20)));
			CPPUNIT_ASSERT( psr6(MyPanel("file2.xsq", 3, max_size_t)));
			CPPUNIT_ASSERT( psr6(MyPanel("file2.xsq", 4, 0)));
			CPPUNIT_ASSERT( psr6(MyPanel("file2.xsq", 4, max_size_t)));
			CPPUNIT_ASSERT(!psr6(MyPanel("file2.xsq", 5, 0)));
			CPPUNIT_ASSERT(!psr6(MyPanel("file2.xsq", max_size_t, 0)));

    	} catch (string s) {
    		cerr << s << endl;
    		throw s;
    	}
    }

    void colorEncodingTest() {
    	ColorEncoding encoding = SOLID_ENCODING;
    	CPPUNIT_ASSERT(-1 == encoding.getOffset());
    	CPPUNIT_ASSERT("11" == encoding.getProbeset());
    	CPPUNIT_ASSERT(1 == encoding.getStride());
    }

    void inspectXsqContent()
    {
    	try {
    		XsqReader xsq;
			bool exception_caught = false;
			try {

				/* Save old error handler */
				herr_t (*old_func)(void*);
				void *old_client_data;
				H5Eget_auto(&old_func, &old_client_data);

				/* Turn off error handling */
				H5Eset_auto(NULL, NULL);

				xsq.open("a/file/that/shouldn't/exist", 0);

				/* Restore previous error handler */
				H5Eset_auto(old_func, old_client_data);
			} catch (file_format_exception _e) {
				exception_caught = true;
			}
			CPPUNIT_ASSERT(exception_caught);
			CPPUNIT_ASSERT(xsq.open("examples/data/xsq/example.barcode.c11.xsq", 0));
			CPPUNIT_ASSERT(xsq.open("examples/data/xsq/example.nobarcode.c11.xsq", 1));
			cerr << xsq.size() << endl;
			CPPUNIT_ASSERT(xsq.size() == 372);

			size_t count = 0;
			for (XsqReader::panel_iterator it = xsq.panels_begin(); it != xsq.panels_end(); ++it) {
				cerr << "Panel{ File=" << it->getFilename() << ",PCNAME=" << it->getPanelContainerName() << ",Panel=" << it->getPanelNumber() << ",size=" << it->size() << " }" << endl;
				size_t subCount = 0;
				for (Panel::panel_fragments_const_iterator fragment = it->begin(); fragment != it->end(); ++fragment) {
					//cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
					//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
					//cerr << ">" << fragment->getName(XSQ_READ_TYPE_R3) << endl;
					//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_R3) <<
					//		fragment->getColors(XSQ_READ_TYPE_R3, SOLID_ENCODING) << endl;
					++subCount;
				}
				count += subCount;
				CPPUNIT_ASSERT(subCount == 31);
				it->release();
			}
			CPPUNIT_ASSERT(count == 372);

			count = 0;
			cerr << "Testing XsqReader::fragement_const_iterator" << endl;
	    	for (XsqReader::fragment_const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
//				cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
//				cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
//				cerr << ">" << fragment->getName(XSQ_READ_TYPE_R3) << endl;
//				cerr << fragment->getPrimerBases(XSQ_READ_TYPE_R3) <<
//						fragment->getColors(XSQ_READ_TYPE_R3, SOLID_ENCODING) << endl;
				++count;
	    	}
	    	CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(372), count);

	    	count = 0;
	    	IncrementingFunctor<Fragment> counter(count);
			for_each(xsq.begin(),xsq.end(),IncrementingFunctor<Fragment>(count));
			CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(372), count);

			cerr << "finished inspectXsqContent()" << endl;
    	} catch (string s) {
    		cerr << "Exception: " << s << endl;
    		throw s;
    	}
    }

    void partialIterationTest() {
    	try {
    		XsqReader xsq;
    		xsq.open("examples/data/xsq/example.barcode.bs,c11,c1303.xsq?start=1.1&end=1.1", 0);
    		CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(31), xsq.size());
    		size_t count = 0;
    		for (XsqReader::fragment_const_iterator frag = xsq.begin(); frag != xsq.end(); ++frag) {
    			++count;
//    			cerr << ">" << frag->getName(XSQ_READ_TYPE_F3) << endl;
//    			cerr << frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
//    			cerr << frag->getColors(XSQ_READ_TYPE_F3, _5500_PLUS4_ENCODING) << endl;
//    			cerr << frag->getBases(XSQ_READ_TYPE_F3) << endl;
    		}
    		CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(31), count);
    	} catch (string s) {
    		cerr << "Exception: " << s << endl;
    		throw s;
    	}
    }

	struct partition {
		partition() : begin(0), end(0), size(0) {}
		void clear() {
			begin = 0;
			end = 0;
			size = 0;
		}
		size_t begin;
		size_t end;  //One past the end.
		size_t size;
	};

    void splittingDemonstration() {
		const size_t NUM_PANELS = 1000;
		uint32_t panels[NUM_PANELS];
		uint32_t sum = 0;
		srand ( time(NULL) );
		for (size_t i=0; i<NUM_PANELS; ++i)
			sum += panels[i] = 300000 + (200000 - rand() % 400000);

		const size_t NUM_PARTITIONS = 8;

		boost::shared_array<partition> partitions(new partition[NUM_PARTITIONS]);

		size_t partition_size = sum/NUM_PARTITIONS;
		partitions[0].begin = 0;
		for (size_t i=0, j=0; i<NUM_PANELS; ++i) { //i panel, j partition
			partitions[j].size += panels[i];
			partitions[j].end = i+1;
			if (partitions[j].size > partition_size && j+1 < NUM_PARTITIONS) {
				partitions[++j].begin = i+1;
				partitions[j].end = i+2;
			}
		}

		cout << "panels: [";
		for (size_t i=0; i<NUM_PANELS; ++i) {
			cout << panels[i] << ", ";
		}
		cout << "]" << endl;

		cout << "sum=" << sum << endl;
		cout << "partition_size=" << partition_size << endl;

		for (size_t i=0; i<NUM_PARTITIONS; ++i) {
			cout << i << " " << partitions[i].begin << " " << partitions[i].end << " " << partitions[i].size << " [";
			size_t sum = 0;
			for (size_t j=partitions[i].begin; j<partitions[i].end; ++j) {
				cout << panels[j] << ", ";
				sum += panels[j];
			}
			cout << "]" << endl;
			CPPUNIT_ASSERT_EQUAL(sum, partitions[i].size);
		}
    }

    void splittingTest() {
    	try {
    		XsqReader xsq;
    		xsq.open("examples/data/xsq/example.barcode.bs,c11,c1303.xsq", 0);
    		vector<string> URLs = xsq.getURLs();
    		vector<XsqReader> readers = xsq.divideEvenly(8,31);
    		cerr << "(";
    		for (vector<string>::const_iterator p_url = URLs.begin(); p_url < URLs.end(); ++p_url)
    		    cerr << *p_url << ",";
    		cerr << ")";
    		cerr << " partititioned to (";
    		for (vector<XsqReader>::iterator it = readers.begin(); it != readers.end(); ++it) {
    			URLs = it->getURLs();
    			cerr << "(";
    			for (vector<string>::const_iterator p_url = URLs.begin(); p_url < URLs.end(); ++p_url)
    			    cerr << *p_url << ",";
    			cerr << "),";
    		}
    		cerr << ")" << endl;
    		size_t numReads = 0;
    		for (vector<XsqReader>::iterator it = readers.begin(); it != readers.end(); ++it) {
    			URLs = it->getURLs();
    			for (vector<string>::const_iterator p_url = URLs.begin(); p_url < URLs.end(); ++p_url)
    				cerr << "now iterating over " << *p_url << endl;
    			for (XsqReader::fragment_const_iterator frag = it->begin(); frag != it->end(); ++frag) {
    				//cerr << frag->getName(XSQ_READ_TYPE_F3) << endl;
    				++numReads;
    			}
    			it->close();
    		}
    		CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(279), numReads);
    	} catch (string s) {
    		cerr << "Exception: " << s << endl;
    		throw s;
    	}
    }

    void chengyongsFileTest() {
		XsqReader xsq;
		xsq.open("/local/test/reads/xsq/DefaultLibrary_F3.xsq",0);
		cerr << "Num reads:"<< xsq.size() << endl;
		size_t count = 0;
		time_t start = time(NULL);
		for (XsqReader::fragment_const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
			//fragment->getName(XSQ_READ_TYPE_F3);
			//fragment->getPrimerBases(XSQ_READ_TYPE_F3);
			//fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING);
			//cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
			//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
			++count;
			if (count % 100000 == 0 && time(NULL) != start)
				cerr << "Read " << count << " fragments, " << count / (time(NULL) - start) << " fragments per second." << endl;
		}
		CPPUNIT_ASSERT_EQUAL(xsq.size(), count);
		xsq.close();
		cerr << count << endl;

//			count = 0;
//			xsq = XsqReader();
//			xsq.open("non_versioned/cyang_examples_20101122/Indexing_F3.xsq",0);
//			for (XsqReader::fragment_const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
//				cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
//				cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
//				++count;
//			}
//			xsq.close();
    }

    //Demonstrates the partitioning algorithm
    void partitionTest() {
    	cerr << "partitionTest()" << endl;
    	const size_t NUM_PANELS = 1000;
    	size_t num_partitions = 8;
		srand ( time(NULL) );
		std::vector<uint32_t> panels;
		for (size_t i=0; i<NUM_PANELS; ++i)
			panels.push_back(300000 + (200000 - rand() % 400000));
		std::cerr << "panels: [";
		for (size_t i=0; i<NUM_PANELS; ++i) {
			std::cerr << panels[i] << ", ";
		}
		std::cerr << "]" << std::endl;
		uint32_t sum = total(panels.begin(), panels.end());
		std::cerr << 0 << "-" << NUM_PANELS-1 << ",size=" << sum << std::endl << std::endl;
		vector<size_t> partitionStarts = lifetechnologies::partition(panels, num_partitions, sum/NUM_PANELS);
		vector<double> sizes;
		for (vector<size_t>::const_iterator it=partitionStarts.begin(); it != partitionStarts.end(); ++it) {
			vector<uint32_t>::const_iterator first = panels.begin() + *it;
			vector<uint32_t>::const_iterator last = it + 1 < partitionStarts.end() ? panels.begin() + *(it+1) : panels.end();
			uint32_t size = total(first, last);
			sizes.push_back(size);
			cerr << *it << "-" << (it + 1 < partitionStarts.end() ? *(it+1) : NUM_PANELS - 1) << ", size=" << total(first, last) << endl;
		}
		sort(sizes.begin(), sizes.end());
		const double maxDiff = abs((double)sizes.front() - (double)sizes.back()) / sizes.front();
		cerr << "maximum difference: " << maxDiff << endl;
		CPPUNIT_ASSERT(maxDiff < 0.05);
    }

    void writeSomething() {
    	//    		const unsigned char cqv = 110;
    	//    		const uint8_t qv = cqv >> 2;
    	//    		const char call = cqv & 0x03;
    	//    		const unsigned char ncqv = qv << 2 | call;
    	//    		cerr << cqv << " " << (size_t)qv << " " << (size_t)call << " " << ncqv << endl;
    	// exit(1);
    	unsigned char* data = new unsigned char[75 * 30];
    	unsigned char** arr = new unsigned char*[30];
    	arr[0] = data;
    	for (size_t i=0; i<30; i++)
    		arr[i] = arr[0] + 75 * i;

		for (size_t i=0; i<30; ++i)
			for (size_t j=0; j<75; ++j)
				arr[i][j] = 103;

		for (size_t i=0; i<30; ++i) {
			cerr << i << ": ";
			for (size_t j=0; j<75; ++j) {
				cerr << (uint16_t)arr[i][j] << " ";
			}
			cerr << endl;
		}
		boost::filesystem::remove("foo.hdf");
		hid_t fileHid = H5Fcreate("foo.hdf", H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
		hsize_t* dims = new hsize_t[2];
		dims[0] = 30;
		dims[1] = 75;
		hid_t dataspaceHid = H5Screate_simple(2, dims, NULL);
		hid_t datasetHid = H5Dcreate(fileHid, "CallQV", H5T_STD_U8LE_g, dataspaceHid, H5P_DEFAULT );
		H5Dwrite(datasetHid, H5T_STD_U8LE_g, dataspaceHid, dataspaceHid, H5P_DEFAULT, data);
		H5Fclose(fileHid);
    }

    void readWriteReadTest(string const& src, string const& dest) {
		XsqReader xsqReader;
		xsqReader.open(src,0);
		boost::filesystem::path outfile(dest);
		if (!boost::filesystem::exists(outfile.parent_path())) boost::filesystem::create_directories(outfile.parent_path());
		if (boost::filesystem::exists(outfile)) boost::filesystem::remove(outfile);
		size_t count1 = 0;
		time_t start = time(NULL);
		XsqWriter writer(outfile.file_string());
		for (XsqReader::fragment_const_iterator fragment = xsqReader.begin(); fragment != xsqReader.end(); ++fragment) {
			if (++count1 % 1000000 == 0) cout << "Copied " << count1 << " fragments, " << count1 / (time(NULL) - start + 1) << " fragments per second." << endl;
			writer << *fragment;
		}
		writer.close();

		xsqReader.close();
		xsqReader = XsqReader();
		xsqReader.open(outfile.file_string(), 0);
		start = time(NULL);
		size_t count2 = 0;
		for (XsqReader::fragment_const_iterator fragment = xsqReader.begin(); fragment != xsqReader.end(); ++fragment) {
			if (++count2 % 1000000 == 0) cout << "Read " << count2 << " fragments, " << count2 / (time(NULL) - start + 1) << " fragments per second." << endl;
			fragment->getName(XSQ_READ_TYPE_F3);
			fragment->getPrimerBases(XSQ_READ_TYPE_F3);
			fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING);
		}
		CPPUNIT_ASSERT_EQUAL(count1, count2);
		xsqReader.close();
    }
    

    void xsqWriterTest() {
    	readWriteReadTest("examples/data/xsq/example.barcode.c11.xsq", "non_versioned/xsqWriterTest/0.xsq");
    }

    void bigXsqWriterTest() {
    	readWriteReadTest("/local/test/reads/xsq/DefaultLibrary_F3.xsq", "/local/test/reads/xsq/0.xsq");
    }

    void sowmisFileTest() {
    	try {
    		XsqReader reader;
    		reader.open("/local/mullermw/data/xsq/DH10b_1_2plus4_6P5_base.h5?start=1.487&end=1.487", 0);
    		reader.begin();
    		uint8_t* colors = new uint8_t[50];
    		uint8_t* qvs = new uint8_t[50];
    		for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
    			cerr << fragment->getName(XSQ_READ_TYPE_F3) << endl;
    			//cerr << fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
    			fragment->getCalls(colors, qvs, XSQ_READ_TYPE_F3, SOLID_ENCODING);
    			for (size_t i=0; i<50; ++i)
    				cerr << static_cast< unsigned int >( colors[i] );
    			cerr << endl;
    			for (size_t i=0; i<50; ++i)
    				cerr << static_cast< unsigned int >( qvs[i] ) << " ";
    			cerr << endl;
    			//cerr << fragment->getColors(XSQ_READ_TYPE_F3, _5500_PLUS4_ENCODING) << endl;
    			//cerr << fragment->getBases(XSQ_READ_TYPE_F3) << endl;
    		}
    		cerr << "done" << endl;
    		//readWriteReadTest("/local/mullermw/data/xsq/HumanChr21HetChr22Hom_SmallIndel_F3F5.h5", "non_versioned/sowmisFileTest/0.xsq");
    	} catch (XsqException e) {
    		cerr << e.what() << endl;
    	}
    }

    void perfVsCsfastaTest() {
    	boost::filesystem::path outfile("non_versioned/perfVsCsfastaTest/DefaultLibrary_F3.csfasta");
    	if (!boost::filesystem::exists(outfile.parent_path())) boost::filesystem::create_directories(outfile.parent_path());
    	//boost::filesystem::remove(outfile);
    	XsqReader reader;
    	string url = "/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.1&end=1.20";
    	if (!boost::filesystem::exists(outfile)) {
			reader.open(url, 0);
			ofstream out(outfile.file_string().c_str());
			cerr << "Preparing csfasta file." << endl;
			for (XsqReader::fragment_const_iterator frag=reader.begin(); frag != reader.end(); ++frag) {
				out << ">" << frag->getName(XSQ_READ_TYPE_F3) << endl;
				out << frag->getPrimerBases(XSQ_READ_TYPE_F3) << frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
			}
			reader.close();
    	}

    	ifstream in(outfile.file_string().c_str());
    	size_t count = 0;
    	size_t numChars;
    	time_t start = time(NULL);


	while(in.good()) {
        	string name;
        	string seq;
    		getline(in, name);
    		getline(in, seq);
    		numChars += name.size();
    		numChars += seq.size();
    		time_t elapsed = time(NULL) - start;
    		if (++count % 100000 == 0 && elapsed > 0) cerr << "Read " << count << " records, " << (count / elapsed) << " fragments per second." << endl;
    	}
    	cerr << "CSFASTA: Read " << count << " reads in " << time(NULL) - start << " seconds." << endl;
    	in.close();

    	reader = XsqReader();
    	reader.open(url, 0);
    	count = 0;
    	start = time(NULL);
    	for (XsqReader::fragment_const_iterator frag=reader.begin(); frag != reader.end(); ++frag) {
	  string name = frag->getName(XSQ_READ_TYPE_F3);
	  string seq = frag->getPrimerBases(XSQ_READ_TYPE_F3) + frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING);
	  numChars += name.size();
	  numChars += seq.size();
	  time_t elapsed = time(NULL) - start;
    		if (++count % 100000 == 0 && elapsed > 0) cerr << "Read " << count << " records, " << (count / elapsed) << " fragments per second." << endl;
    	}
    	cerr << "XSQ: Read " << count << " reads in " << time(NULL) - start << " seconds." << endl;
    }


    void perfXsqOnlyTest() {
    	boost::filesystem::path outfile("non_versioned/perfVsCsfastaTest/DefaultLibrary_F3.csfasta");
    	if (!boost::filesystem::exists(outfile.parent_path())) boost::filesystem::create_directories(outfile.parent_path());
    	//boost::filesystem::remove(outfile);
    	XsqReader reader;
    	string url = "/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.1&end=1.20";
    	if (!boost::filesystem::exists(outfile)) {
			reader.open(url, 0);
			ofstream out(outfile.file_string().c_str());
			cerr << "Preparing csfasta file." << endl;
			for (XsqReader::fragment_const_iterator frag=reader.begin(); frag != reader.end(); ++frag) {
				out << ">" << frag->getName(XSQ_READ_TYPE_F3) << endl;
				out << frag->getPrimerBases(XSQ_READ_TYPE_F3) << frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
			}
			reader.close();
    	}

    	ifstream in(outfile.file_string().c_str());
    	size_t count = 0;
    	size_t numChars;
    	time_t start = time(NULL);

    	reader = XsqReader();
    	reader.open(url, 0);
    	count = 0;
    	start = time(NULL);
    	for (XsqReader::fragment_const_iterator frag=reader.begin(); frag != reader.end(); ++frag) {
	  string name = frag->getName(XSQ_READ_TYPE_F3);
	  string seq = frag->getPrimerBases(XSQ_READ_TYPE_F3) + frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING);
	  numChars += name.size();
	  numChars += seq.size();
	  time_t elapsed = time(NULL) - start;
    		if (++count % 100000 == 0 && elapsed > 0) cerr << "Read " << count << " records, " << (count / elapsed) << " fragments per second." << endl;
    	}
    	cerr << "XSQ: Read " << count << " reads in " << time(NULL) - start << " seconds." << endl;
    }

    void perfXsqIterationTest() {
    	boost::filesystem::path outfile("non_versioned/perfVsCsfastaTest/DefaultLibrary_F3.csfasta");
    	if (!boost::filesystem::exists(outfile.parent_path())) boost::filesystem::create_directories(outfile.parent_path());
    	//boost::filesystem::remove(outfile);
    	XsqReader reader;
    	string url = "/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.1&end=1.20";
    	if (!boost::filesystem::exists(outfile)) {
			reader.open(url, 0);
			ofstream out(outfile.file_string().c_str());
			cerr << "Preparing csfasta file." << endl;
			for (XsqReader::fragment_const_iterator frag=reader.begin(); frag != reader.end(); ++frag) {
				out << ">" << frag->getName(XSQ_READ_TYPE_F3) << endl;
				out << frag->getPrimerBases(XSQ_READ_TYPE_F3) << frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
			}
			reader.close();
    	}

    	ifstream in(outfile.file_string().c_str());
    	size_t count = 0;
    	time_t start = time(NULL);

    	reader = XsqReader();
    	reader.open(url, 0);
    	count = 0;
    	start = time(NULL);
    	for (XsqReader::fragment_const_iterator frag=reader.begin(); frag != reader.end(); ++frag) {
	  time_t elapsed = time(NULL) - start;
    		if (++count % 100000 == 0 && elapsed > 0) cerr << "Read " << count << " records, " << (count / elapsed) << " fragments per second." << endl;
    	}
    	cerr << "XSQ: Read " << count << " reads in " << time(NULL) - start << " seconds." << endl;
    }

    void writingInParallel() {

    	class Task : public tbb::task {
    		string url;
    		XsqWriter* writer;
    	public:
    		Task(XsqWriter* writer) : url(""), writer(writer) {}
    		Task(string const& url, XsqWriter* writer) : url(url), writer(writer) {}

    		tbb::task* execute() {
    			if (url.empty()) {
    				Task& a = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.1&end=1.2", writer);
    				Task& b = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.3&end=1.4", writer);
    				Task& c = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.5&end=1.6", writer);
    				Task& d = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.7&end=1.8", writer);
    				Task& e = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.9&end=1.10", writer);
    				Task& f = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.11&end=1.12", writer);
    				Task& g = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.13&end=1.14", writer);
    				Task& h = *new ( allocate_child() ) Task("/local/test/reads/xsq/DefaultLibrary_F3.xsq?start=1.15&end=1.16", writer);
    				set_ref_count(9);
    				spawn(b);
    				spawn(c);
    				spawn(d);
    				spawn(e);
    				spawn(f);
    				spawn(g);
    				spawn(h);
    				spawn_and_wait_for_all(a);
    			} else {
					XsqReader reader;
					reader.open(url, 0);
					for (XsqReader::fragment_const_iterator it = reader.begin(); it != reader.end(); ++it)
						*writer << (*it);
					cerr<< "done " << url << endl;
    			}
    			return NULL;
    		}
    	};
    	boost::filesystem::path outfile("non_versioned/writingInParallel/1.xsq");
    	if (!boost::filesystem::exists(outfile.parent_path())) boost::filesystem::create_directories(outfile.parent_path());
    	if (boost::filesystem::exists(outfile)) boost::filesystem::remove(outfile);
    	XsqWriter writer(outfile.file_string());
    	task::spawn_root_and_wait(*new(task::allocate_root()) Task(&writer));
    	writer.close();
    }

    void xsqMultiWriter() {
    	boost::filesystem::path outdir("non_versioned/XsqMultiWriter");
    	if (boost::filesystem::exists(outdir)) boost::filesystem::remove_all(outdir);
    	boost::filesystem::create_directories(outdir);
    	XsqReader reader;
    	reader.open("examples/data/xsq/example.barcode.c11.xsq", 0);
		reader.open("examples/data/xsq/example.nobarcode.c11.xsq", 1);
		XsqMultiWriter writer(outdir.string());
		for (XsqReader::fragment_const_iterator it = reader.begin(); it != reader.end(); ++it)
			writer << (*it);
		reader.close();
		writer.close();
		int numFiles = 0;
		for (boost::filesystem::directory_iterator it(outdir); it != boost::filesystem::directory_iterator(); ++it)
			++numFiles;
		CPPUNIT_ASSERT_EQUAL(2, numFiles);
		CPPUNIT_ASSERT(boost::filesystem::exists(outdir / "example.barcode.c11.xsq"));
		CPPUNIT_ASSERT(boost::filesystem::exists(outdir / "example.nobarcode.c11.xsq"));
    }

    /**
     * Copy an XSQ file, changing the 5th solid color of every read to 0 and verify.
     */
    void setCallsTest() {
    	boost::filesystem::path outFile("non_versioned/setCallsTest/out.xsq");
    	boost::filesystem::create_directories(outFile.parent_path());
    	if (boost::filesystem::exists(outFile)) boost::filesystem::remove(outFile);
    	XsqReader reader;
    	reader.open("examples/data/xsq/example.nobarcode.bs,c11,c1303.xsq", 0);
    	XsqWriter writer(outFile.string());
    	uint8_t colors[75];
    	uint8_t qvs[75];
    	for (XsqReader::fragment_const_iterator frag = reader.begin(); frag != reader.end(); ++frag) {
    		frag->getCalls(colors, qvs, XSQ_READ_TYPE_F3, SOLID_ENCODING);
    		colors[5] = 0;
    		qvs[6] = 0;
    		qvs[7] = 1;
    		qvs[8] = 2;
    		qvs[9] = 63;
    		frag->setCalls(colors, qvs, XSQ_READ_TYPE_F3, SOLID_ENCODING);
    		writer << (*frag);
    	}
    	writer.close();
    	reader.close();
    	reader = XsqReader();
    	reader.open(outFile.string(), 0);
    	for (XsqReader::fragment_const_iterator frag = reader.begin(); frag != reader.end(); ++frag) {
    		//frag->getCalls(colors, qvs, XSQ_READ_TYPE_F3, SOLID_ENCODING);
    		//cerr << frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
    		//cerr << frag->getQualityValues(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
    		//cerr << frag->getBases(XSQ_READ_TYPE_F3) << endl;
    		CPPUNIT_ASSERT(colors[5] == 0 || colors[5] == 5);
    	}
    	reader.close();
    }

    /**
     * Change the calls in an XSQ file.
     */
    void editInPlaceTest() {
    	boost::filesystem::path outFile("non_versioned/editInPlaceTest/example.nobarcode.bs,c11,c1303.xsq");
    	boost::filesystem::create_directories(outFile.parent_path());
    	boost::filesystem::remove(outFile);
    	boost::filesystem::copy_file("examples/data/xsq/example.nobarcode.bs,c11,c1303.xsq", outFile);
    	XsqReader reader(false);
    	reader.open(outFile.string(), 0);
    	uint8_t colors[75];
    	uint8_t qvs[75];
    	for (XsqReader::fragment_const_iterator frag = reader.begin(); frag != reader.end(); ++frag) {
    		frag->getCalls(colors, qvs, XSQ_READ_TYPE_F3, SOLID_ENCODING);
    		//cerr << frag->getName() << endl;
    		colors[5] = 0;
    		qvs[6] = 0;
    		qvs[7] = 1;
    		qvs[8] = 2;
    		qvs[9] = 63;
    		frag->setCalls(colors, qvs, XSQ_READ_TYPE_F3, SOLID_ENCODING);
    	}
    	reader.close();
    	reader = XsqReader();
    	reader.open(outFile.string(), 0);
    	for (XsqReader::fragment_const_iterator frag = reader.begin(); frag != reader.end(); ++frag) {
    		frag->getCalls(colors, qvs, XSQ_READ_TYPE_F3, SOLID_ENCODING);
    		//cerr << frag->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
    		CPPUNIT_ASSERT(colors[5] == 0 || colors[5] == 5);
    	}
    	reader.close();
    }

    void filteringTest() {
    	XsqReader reader;
    	reader.open("examples/data/xsq/example.nobarcode.bs,c11,c1303.xsq", 0);
    	reader.setSkipFilteredFragments(false);
    	size_t count = 0;
    	string fragmentName;
    	bool isFiltered;
    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
    		count++;
    		fragmentName = fragment->getName();
    		isFiltered = fragment->isFiltered();
    		CPPUNIT_ASSERT(isFiltered == (fragmentName == "0_1_1_1_218"));
    	}
    	CPPUNIT_ASSERT(count == 93);
    	count = 0;
    	reader.setSkipFilteredFragments(true);
    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment)
    		count++;
    	CPPUNIT_ASSERT(count == 92);
    }

    void readFilteringTest() {
    	XsqReader reader;
    	reader.open("examples/data/xsq/example.nobarcode.bs,c11,c1303.xsq", 0);
    	string fragmentName;
    	bool isF3Filtered;
    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
    		fragmentName = fragment->getName();
    		isF3Filtered = fragment->isReadFiltered(XSQ_READ_TYPE_F3);
    		CPPUNIT_ASSERT(isF3Filtered == (fragmentName == "0_1_1_1_218" || fragmentName == "0_1_1_1_224"));
    	}
    }

    static string toString(uint8_t* const start, uint8_t* const end) {
    	stringstream stream;
    	for (uint8_t* i = start; i< end; ++i)
    		stream << (uint16_t)*i;
    	return stream.str();
    }

    void readTrimTest() {
    	XsqReader reader;
    	reader.open("examples/data/xsq/example.nobarcode.bs,c11,c1303.xsq", 0);
    	string fragmentName;
    	size_t trimStartLength, trimEndLength;
    	const XsqReadType readType = XSQ_READ_TYPE_F3;
    	uint8_t* uint8_75 = new uint8_t[75];
    	uint8_t* p_colors;
    	uint8_t* p_qv;
    	size_t readLength;
    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
    		fragmentName = fragment->getName();
    		trimStartLength = fragment->getTrim(readType, true);
    		trimEndLength = fragment->getTrim(readType, false);
    		if (fragmentName == "0_1_1_1_218") {
    			string bases = fragment->getBases(readType);
    			string colors = fragment->getColors(readType, BASE_ENCODING);
    			char* arr = fragment->getBaseRead(readType);
    			fragment->getCalls(uint8_75, NULL, readType, BASE_ENCODING);
    			string getCallsP = toString(uint8_75, uint8_75+75);
    			fragment->getCalls(&(p_colors), &(p_qv), readLength, readType, BASE_ENCODING);
    			string getCallsPP((const char*)p_colors);
    			CPPUNIT_ASSERT(0 == trimStartLength);
    			CPPUNIT_ASSERT(75 == trimEndLength);
    			CPPUNIT_ASSERT(bases == string(75, 'N'));
    			CPPUNIT_ASSERT(colors == string(75, '.'));
    			CPPUNIT_ASSERT(bases == string(arr));
    			CPPUNIT_ASSERT(getCallsP == string(75, '5'));
    			CPPUNIT_ASSERT(getCallsPP == string(75, 'N'));
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << bases << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << colors << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << arr << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << getCallsP << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << p_colors << endl;
    		} else if (fragmentName == "0_1_1_1_224") {
    			string bases = fragment->getBases(readType);
    			string colors = fragment->getColors(readType, BASE_ENCODING);
    			char* arr = fragment->getBaseRead(readType);
    			fragment->getCalls(uint8_75, NULL, readType, BASE_ENCODING);
    			string getCallsP = toString(uint8_75, uint8_75+75);
    			fragment->getCalls(&(p_colors), &(p_qv), readLength, readType, BASE_ENCODING);
    			string getCallsPP((const char*)p_colors);
    			CPPUNIT_ASSERT(0 == trimStartLength);
    			CPPUNIT_ASSERT(75 == trimEndLength);
    			CPPUNIT_ASSERT(bases == string(75, 'N'));
    			CPPUNIT_ASSERT(colors == string(75, '.'));
    			CPPUNIT_ASSERT(bases == string(arr));
    			CPPUNIT_ASSERT(getCallsP == string(75, '5'));
    			CPPUNIT_ASSERT(bases == getCallsPP);
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << fragment->getBases(readType) << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << colors << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << arr << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << getCallsP << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << p_colors << endl;
    		} else if (fragmentName == "0_1_1_1_225") {
    			string bases = fragment->getBases(readType);
    			string colors = fragment->getColors(readType, BASE_ENCODING);
    			char* arr = fragment->getBaseRead(readType);
    			fragment->getCalls(uint8_75, NULL, readType, BASE_ENCODING);
    			string getCallsP = toString(uint8_75, uint8_75+75);
    			fragment->getCalls(&(p_colors), &(p_qv), readLength, readType, BASE_ENCODING);
    			string getCallsPP((const char*)p_colors);
    			CPPUNIT_ASSERT(19 == trimStartLength);
    			CPPUNIT_ASSERT(0 == trimEndLength);
    			CPPUNIT_ASSERT(bases.find(string(19, 'N')) == 0);
    			CPPUNIT_ASSERT(colors.find(string(19, '.')) == 0);
    			CPPUNIT_ASSERT(bases == string(arr));
    			CPPUNIT_ASSERT(getCallsP.find(string(19, '5')) == 0);
    			CPPUNIT_ASSERT(bases == getCallsPP);
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << fragment->getBases(readType) << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << colors << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << arr << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << getCallsP << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << p_colors << endl;
    		} else if (fragmentName == "0_1_1_1_228") {
    			string bases = fragment->getBases(readType);
    			string colors = fragment->getColors(readType, BASE_ENCODING);
    			char* arr = fragment->getBaseRead(readType);
    			fragment->getCalls(uint8_75, NULL, readType, BASE_ENCODING);
    			string getCallsP = toString(uint8_75, uint8_75+75);
    			fragment->getCalls(&(p_colors), &(p_qv), readLength, readType, BASE_ENCODING);
    			string getCallsPP((const char*)p_colors);
    			CPPUNIT_ASSERT(0 == trimStartLength);
    			CPPUNIT_ASSERT(21 == trimEndLength);
    			CPPUNIT_ASSERT(bases.rfind(string(21, 'N')) == 75 - 21);
    			CPPUNIT_ASSERT(colors.rfind(string(21, '.')) == 75 - 21);
    			CPPUNIT_ASSERT(bases == string(arr));
    			CPPUNIT_ASSERT(getCallsP.rfind(string(21, '5')) == 75 - 21);
    			CPPUNIT_ASSERT(bases == getCallsPP);
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << fragment->getBases(readType) << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << colors << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << arr << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << getCallsP << endl;
    			//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << " " << p_colors << endl;
    		} else {
    			CPPUNIT_ASSERT(0 == trimStartLength);
    			CPPUNIT_ASSERT(0 == trimEndLength);
    		}
    		//cerr << fragmentName << " " << trimStartLength << " " << trimEndLength << endl;
    	}
    }

    void missingTagTest() {
    	//create a copy of an xsq file that is missing one of the tags directories.
    	const boost::filesystem::path originalXsq("examples/data/xsq/example.nobarcode.c11.xsq");
    	const boost::filesystem::path dir("non_versioned/missingTagTest/");
    	boost::filesystem::remove_all(dir);
    	boost::filesystem::create_directories(dir);
    	const boost::filesystem::path xsq = dir / originalXsq.filename();
    	boost::filesystem::copy_file(originalXsq, xsq);
    	const hid_t hFile = H5Fopen(xsq.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    	if (hFile < 0) throw "error opening file";
    	const hid_t hGroup = H5Gopen(hFile, "/DefaultLibrary/0001");
    	if (hGroup < 0) throw "error opening group";
    	if (H5Ldelete(hGroup, "R3", H5P_DEFAULT) < 0) throw "error deleting group";
    	if (H5Fclose(hFile) < 0) throw "error closing file";

    	XsqReader reader(false);
    	reader.setMaskFilteredAndTrimmedBases(true);
    	reader.open(xsq.string(), 1);
    	size_t count = 0;
    	uint8_t colors[100];
    	uint8_t qvs[100];
    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
    		++count;
    		cerr << fragment->getName() << " " << (fragment->isReadFiltered(XSQ_READ_TYPE_R3) ? "FILTERED" : "") << endl;
    		fragment->getCalls(colors, qvs, XSQ_READ_TYPE_R3, SOLID_ENCODING);
    		colors[0] = 0;
    		qvs[0] = 63;
    		fragment->setCalls(colors, qvs, XSQ_READ_TYPE_R3, SOLID_ENCODING);
    		cerr << fragment->getColors(XSQ_READ_TYPE_R3, SOLID_ENCODING) << endl;
    	}
    	cerr << "Processed " << count << " fragments." << endl;
    	reader.close();

    	XsqReader reader2;
    	reader2.setMaskFilteredAndTrimmedBases(true);
    	reader2.open(xsq.string(), 1);
    	boost::filesystem::path writeXsq = dir / ("rewrite."+originalXsq.filename());
    	XsqWriter writer(writeXsq.string());
    	count = 0;
    	for (XsqReader::fragment_const_iterator fragment = reader2.begin(); fragment != reader2.end(); ++fragment) {
			++count;
			const bool readIsFiltered = fragment->isReadFiltered(XSQ_READ_TYPE_R3);
			cerr << fragment->getName() << " " << (readIsFiltered ? "FILTERED" : "") << endl;
			const string colors = fragment->getColors(XSQ_READ_TYPE_R3, SOLID_ENCODING);
			CPPUNIT_ASSERT(readIsFiltered || colors[0] == '0');
			cerr << colors << endl;
			writer << *fragment;
		}
		cerr << "Processed " << count << " fragments." << endl;
		reader2.close();
		writer.close();
    	boost::filesystem::remove_all(dir);
    }

    void zhengsBug1() {
//    	XsqReader xsq;
//    	xsq.open("examples/data/xsq/example.nobarcode.c11.xsq", 0);
//    	typedef XsqReader::fragment_const_iterator Iterator;
//    	Iterator frag;
//    	//for (frag = xsq.begin(); frag != xsq.end(); frag++) {
//    	//	cerr << frag->getName(XSQ_READ_TYPE_F3) << endl;              //-----------error occurs
//    	//}
//    	size_t numproc = 1;
//    	vector<XsqReader> readers = xsq.divideEvenly(numproc, 1000000);
//    	XsqReader& reader = readers[0];
//    	for (frag = reader.begin(); frag != reader.end(); frag++) {
//    		cerr << frag->getName(XSQ_READ_TYPE_F3) << endl;              //-----------error occurs
//    	}
//		xsqreader **slist = new xsqreader*[numproc];
//		int i;
//		if (numproc > readers.size()) numproc = readers.size();
//		for (i = 0; i < numproc; i++) {
//			xsqreader_real *xs  = new xsqreader_real();
//			XsqReader *r = new XsqReader();
//			//*r = readers[i];
//			*r = xsq;
//			xs->init(r, rt1, rt2);
//			slist[i] = (xsqreader *)xs;
//		}
//		fprintf(stderr, "%d\n", i);
//		return slist;
        XsqReader xsq;

        xsq.open("examples/data/xsq/example.barcode.bs,c11,c1303.xsq", 1);

        boost::filesystem::path outdir("Nout");
        if (boost::filesystem::exists(outdir)) boost::filesystem::remove_all(outdir);
        boost::filesystem::create_directories(outdir);
        XsqMultiWriter w("Nout");

        vector<XsqReader> readers = xsq.divideEvenly(3, 10);
        fprintf(stderr, "%d\n", static_cast<int>(readers.size()));
        cerr << readers[0].getURLs()[0] << endl;
        cerr << "---" << endl;
        cerr << readers[1].getURLs()[0] << endl;
        typedef XsqReader::fragment_const_iterator Iterator;
        Iterator f1=readers[0].begin(), f2=readers[1].begin();
        do {
            if (f1 != readers[0].end()) {
            	cerr << "f1 " << f1->getName(XSQ_READ_TYPE_F3) << endl;
                w << (*f1);
                f1++;
            } else break;
            if (f2 != readers[1].end()) {
            	cerr << "f2 " << f2->getName(XSQ_READ_TYPE_F3) << endl;
                w << (*f2);
                f2++;
            } else break;
        } while (1);
        w.close();
        xsq.close();
    }

//    XsqMultiWriter w("Nout");
//
//    static void *process_run1(void *t)
//    {
//       XsqReader *xsq = (XsqReader *) t;
//       for (Iterator f1=xsq->begin(); f1 != xsq->end(); f1++) {
//            w <<  (*f1);
//       }
//    }
//    xsqreader **xsqwrapper(int &numproc, char *rsfile, int mode, char ** &fn, int &nf, char *s)
//    {
//        XsqReader xsq;
//
//        xsq.open("xsq/example.barcode.bs,c11,c1303.xsq", 1);
//
//        vector<XsqReader> readers = xsq.divideEvenly(3, 10);
//        fprintf(stderr, "%d\n", readers.size());
//        pthread_t thread_id[2];
//        pthread_create( &thread_id[0], NULL, &process_run1, (void *)(&readers[0]));
//        pthread_create( &thread_id[1], NULL, &process_run1, (void *)(&readers[1]));
//
//            pthread_join( thread_id[0], NULL);
//            pthread_join( thread_id[1], NULL);
//
//
//        w.close();
//        xsq.close();
//    }

    void zhengsBug2() {
    	for (uint8_t value = 0; value < 255; ++value) {
    		uint8_t color = value & 0x03;
    		uint8_t qv = value >> 2;
    		uint8_t newValue = qv << 2 | color;
    		cerr << static_cast<int>(value) << " " << static_cast<int>(color) << " " << static_cast<int>(qv) << " " << static_cast<int>(newValue) << endl;
    	}
//    	XsqReader reader;
//    	reader.open("examples/data/xsq/example.barcode.bs,c11,c1303.xsq", 1);
//    	cerr << reader.size() << endl;
//    	vector<XsqReader> readers = reader.divideEvenly(4, 10);
//    	cerr << readers.size() << endl;
    }

    void zhengsBug3() {
    	XsqReader reader;
    	reader.open("/local/mullermw/Desktop/HumanChr21HetChr22Hom_SmallIndel_F3F5.h5", 0);
    	size_t count = 0;
    	for (XsqReader::panel_const_iterator panel = reader.panels_begin(); panel != reader.panels_end(); ++panel) {
    		count++;
    		cerr << panel->size() << ", ";
    	}
    	vector<XsqReader> readers = reader.divideEvenly(8, 0);
    	cerr << count << "  " << readers.size() << endl;
    }

    void zhengsBug() {
//    	uint32_t sizes[] = {79202,79202,79202,79202,79202,79202,79202,79202,41201,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,39601,25087};
//    	vector<uint32_t> vec;
//    	for (int i=0; i<114; ++i)
//    		vec.push_back(sizes[i]);
//    	vector<size_t> partitionStarts = lifetechnologies::partition(vec, 8, 0);
//    	cerr << partitionStarts.size() << " partitions"<< endl;
//    	for (vector<size_t>::const_iterator it = partitionStarts.begin(); it != partitionStarts.end(); ++it) {
//    		cerr << *it << endl;
//    	}
//    	for (size_t i=0; i<256; ++i) {
//    		cerr << i << " " << ( i >> 2 < 63 ? boost::lexical_cast<char>( i & 0x03 ) : '.' ) << " " << ( i >> 2) << endl;
//    	}
//    	char c = '1';
//    	size_t s = boost::lexical_cast<size_t>(c);
//    	cerr << s << endl;
    	cerr << char(33) << endl;
    }

    void bug20100117() {
		XsqReader xsq;
		xsq.open("/local/mullermw/data/xsq/DefaultLibrary_F3.xsq?start=1.1&end=1.10",1);
		cerr << xsq.size() << endl;
		vector<XsqReader> readers = xsq.divideEvenly(8, 0);
		XsqReader xsq0 = readers[0];
		cerr << xsq.size() << endl;
		size_t count = 0;
		time_t start = time(NULL);
		for (size_t i=0; i<readers.size(); ++i) {
			XsqReader& reader = readers[i];
			cerr << "Reader# " << i << " " << reader.size() << " " << reader.numPanels() << endl;
			for (XsqReader::panel_const_iterator panel = reader.panels_begin(); panel != reader.panels_end(); ++panel)
				cerr << "  Panel: " << panel->getPanelNumber() << " " << panel->size() << endl;
			for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
				fragment->getName(XSQ_READ_TYPE_F3);
				fragment->getPrimerBases(XSQ_READ_TYPE_F3);
				fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING);
				//cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
				//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
				++count;
				if (count % 100000 == 0 && time(NULL) != start)
					cerr << "Read " << count << " fragments, " << count / (time(NULL) - start) << " fragments per second." << endl;
			}
		}
		CPPUNIT_ASSERT_EQUAL(xsq.size(), count);
		xsq.close();
		cerr << count << endl;
    }

    void bug20110126() {
    	XsqReader reader;
    	reader.open("/local/mullermw/data/DH10b_1_2plus4_6P5_base.h5?start=1&end=1",1);
    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
			cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, _5500_PLUS4_ENCODING) << endl;
    	}
    }

    static void *pppp(void *p)
    {
    	typedef XsqReader::fragment_const_iterator Iterator;
        XsqReader *xsq = (XsqReader *) p;
        Iterator f;
        for (f = xsq->begin(); f != xsq->end(); f++) {
            *xxx << (*f);
        }
        return p;
    }
    void bug20110201() {
        //test
    	boost::filesystem::remove_all("/local/mullermw/data/zhangh/out");
    	boost::filesystem::create_directory("/local/mullermw/data/zhangh/out");
        XsqReader xsq;
        xsq.open("/local/mullermw/data/zhangh/huref_simulation_sRNA_35Frag.xsq", 1);
        xsq.open("/local/mullermw/data/zhangh/huref_simulation_sRNA_35Frag_dup.xsq", 2);
        vector<XsqReader> readers = xsq.divideEvenly(2, 10);

        XsqMultiWriter w("/local/mullermw/data/zhangh/out");
        xxx = &w;
        pthread_t thread_id[2];
        pthread_create(&thread_id[0], NULL, &pppp, (void *) (&(readers[0])));
        pthread_create(&thread_id[1], NULL, &pppp, (void *) (&(readers[1])));
        pthread_join(thread_id[0], NULL);
        pthread_join(thread_id[1], NULL);
    }

    template <class T> bool isBlank(T c_str) {
    	if (c_str == NULL) return true;
    	for (T ptr = c_str; *ptr != 0; ++ptr)
    		if (!isspace(*ptr)) return false;
    	return true;
    }

    char* trim(char* str) {
    	char *end;

    	// Trim leading space
    	while(isspace(*str)) str++;

    	if(*str == 0)  // All spaces?
    		return str;

    	// Trim trailing space
    	end = str + strlen(str) - 1;
    	while(end > str && isspace(*end)) end--;

    	// Write new null terminator
    	*(end+1) = 0;

    	return str;

    }

    void isBlankTest() {
    	CPPUNIT_ASSERT_EQUAL(true, isBlank("      "));
    	CPPUNIT_ASSERT_EQUAL(true, isBlank(""));
    	CPPUNIT_ASSERT_EQUAL(true, isBlank("\n"));
    	CPPUNIT_ASSERT_EQUAL(true, isBlank("\t"));
    	CPPUNIT_ASSERT_EQUAL(true, isBlank("\r\n"));
    	CPPUNIT_ASSERT_EQUAL(false, isBlank("foo   "));
    	CPPUNIT_ASSERT_EQUAL(false, isBlank("   foo"));
    	CPPUNIT_ASSERT_EQUAL(false, isBlank("  foo "));
    	char arr[] = {' ', ' ', 'f', 'o', 'o', ' ', ' ', 0};
    	char expected[] = {'f', 'o', 'o'};
    	CPPUNIT_ASSERT_EQUAL(true, strcmp(expected, trim(arr)) != 0);
    }

    void bug20110207() {

    	PanelRangeSpecifier specifier1("/local/mullermw/data/xsq/Indexing3panels.xsq?start=2.1&end=2.3");
    	PanelRangeSpecifier specifier2("/local/mullermw/data/xsq/Indexing3panels.xsq?start=1.1&end=1.3");
    	cerr << (specifier2 == specifier1) << endl;

    	XsqReader reader;
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=1&end=1", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=2&end=2", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=3&end=3", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=4&end=4", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=5&end=5", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=6&end=6", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=7&end=7", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=8&end=8", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=9&end=9", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=10&end=10", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=11&end=11", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=12&end=12", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=13&end=13", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=14&end=14", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=15&end=15", 1);
    	reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=16&end=16", 1);
    	int count1 = 0;
    	map<size_t, size_t> counts;
    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			//cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
			//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
			++count1;
		}
    	int count2 = 0;
    	vector<XsqReader> readers = reader.divideEvenly(8, 10);
    	cerr << readers.size() << endl;
    	for (vector<XsqReader>::iterator it = readers.begin(); it != readers.end(); ++it) {
    		vector<string> urls = it->getURLs();
    		for (vector<string>::const_iterator jt = urls.begin(); jt != urls.end(); ++jt) {
    			cerr << *jt << endl;
    		}
    		cerr << "----" << endl;
    		for (XsqReader::fragment_const_iterator fragment = it->begin(); fragment != it->end(); ++fragment) {
    			//cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
    			//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, SOLID_ENCODING) << endl;
    			++count2;
    			++counts[fragment->getPanelContainerNumber()];
    		}
    	}
    	cerr << count1 << "  " << count2 << endl;
    	CPPUNIT_ASSERT(count1 == count2);
    	//for (map<size_t, size_t>::const_iterator it = counts.begin(); it != counts.end(); ++it) {
    	//	cerr << it->first << " " << it->second << endl;
    	//}
    }

    void bug20110216() {
    	XsqReader reader(false);
    	reader.open("/local/mullermw/data/xsq/DefaultLibrary_PE.xsq?start=1.561&end=1.561", 1);

    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			//cerr << ">" << fragment->getName(XSQ_READ_TYPE_F5) << endl;
			//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F5) <<  fragment->getColors(XSQ_READ_TYPE_F5, SOLID_ENCODING) << endl;
    		vector<XsqReadType> readTypes = fragment->getReadTypes();
    		unsigned char* calls = new unsigned char[1000];
    		unsigned char* qvs = new unsigned char[1000];
   			fragment->getCalls(calls, qvs, XSQ_READ_TYPE_F5, SOLID_ENCODING);
   			fragment->setCalls(calls, qvs, XSQ_READ_TYPE_F5, SOLID_ENCODING);
		}
    }

    void zhengsPartitionTest() {
    	XsqReader reader;
    	reader.open("/local/mullermw/data/xsq/solid0054_20110102_PE_LFD_RD_SetA_1_HuRef100_F3.full.xsq", 0);
    	const vector<XsqReader> readers = reader.divideEvenly(48, 10);
    	size_t counter = 0;
    	for (vector<XsqReader>::const_iterator it=readers.begin(); it != readers.end(); ++it) {
    		const vector<string> urls = it->getURLs();
    		for (vector<string>::const_iterator jt = urls.begin(); jt != urls.end(); ++jt) {
    			cerr << ++counter << " " << *jt << endl;
    		}
    	}
    }

    void regressPartitioningError20110221() {
    	XsqReader reader;
    	reader.open("/local/mullermw/data/xsq/huref_simulation_50X50LMP_30X.h5?start=1.1&end=1.100", 0);
    	const vector<XsqReader> readers = reader.divideEvenly(48, 10);
    }

	void bug_20020302_panel12_problem() {
//    	XsqReader reader;
//    	reader.open("/local/mullermw/data/xsq/DH10b_1_2plus4_6P5_base.dimas.h5?start=0&end=1", 0);
//    	for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
//			//cerr << ">" << fragment->getName(XSQ_READ_TYPE_F3) << endl;
//			//cerr << fragment->getPrimerBases(XSQ_READ_TYPE_F3) <<  fragment->getColors(XSQ_READ_TYPE_F3, _5500_PLUS4_ENCODING) << endl;
//		}
		XsqReader reader;
		reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=1.1&end=1.1", 0);
		reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=13.3&end=16.1", 0);
		reader.open("/local/mullermw/data/xsq/Indexing3panels.xsq?start=11.1&end=11.2", 0);
		vector<XsqReader> readers = reader.divideEvenly(8,10);
		cerr << readers.size() << endl;
		for (vector<XsqReader>::iterator subreader = readers.begin(); subreader != readers.end(); ++subreader) {
			for (XsqReader::fragment_const_iterator fragment = subreader->begin(); fragment != subreader->end(); ++fragment) {
				cerr << fragment->getName()  << endl;
			}
		}
	}

	void bug_1489_maxXY_problem() {
		XsqReader reader;
		reader.open("examples/data/xsq/example.nobarcode.c11.xsq", 1);
		for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			cerr << fragment->getName() << endl;
		}
	}

	void bug_1478_longReads_problem() {
		XsqReader reader;
		reader.open("/local/mullermw/data/250x250MP.xsq", 1);
		for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			cerr << fragment->getName() << endl;
			char* read = fragment->getColorRead(XSQ_READ_TYPE_F3);
			cerr << read << endl;
		}
	}

	void first_5500_bc_file() {
		XsqReader reader;
		reader.open("/local/mullermw/data/xsq/Inst_2011_03_09_1_01.xsq", 1);
		for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			cerr << fragment->getName() << endl;
			char* read = fragment->getColorRead(XSQ_READ_TYPE_F3);
			cerr << read << endl;
		}
	}

	void refcorProblem() {
		XsqReader reader;
		reader.open("/local/mullermw/tshooting/issue01564/K_20110207_PE_BC_MAGNUM1_WT_FC1_F3.xsq?start=1.88&end=1.88", 1);
		for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			cerr << fragment->getName() << endl;
			char* read = fragment->getBaseRead(XSQ_READ_TYPE_F3);
			cerr << fragment->getBases(XSQ_READ_TYPE_F3) << endl;
			cerr << read << endl;
		}
	}

	void issue02307() {
		XsqReader reader;
		reader.open("/local/mullermw/data/xsq/K_201001207_PE_BC_MAGNUM1_SR_FC2_289_BC_F3only.Base.sowmi.xsq?start=3.647&end=6.610", 1);
		for (XsqReader::fragment_const_iterator fragment = reader.begin(); fragment != reader.end(); ++fragment) {
			cerr << fragment->getName() << endl;
			cerr << fragment->getBases(XSQ_READ_TYPE_F3) << endl;
		}
	}

//    XsqReader *test_read[8];
//    void *ppppp(void *targ)
//    {
//        long long i = (long long) targ;
//        typedef XsqReader::fragment_const_iterator Iterator;
//        Iterator f;
//        char line[1000];
//        sprintf(line, "%lld.file",  i);
//        FILE *fp = fopen(line, "w");
//        for (f = test_read[i]->begin(); f != test_read[i]->end(); f++) {
//            fprintf(fp, "%s\n", f->getNameChar());
//        }
//       fclose(fp);
//    }
//    void bug20110208() {
//        XsqReader xsq;
//        xsq.open("/local/mullermw/data/xsq/Indexing3panels.xsq", 1);
//        vector<XsqReader> readers = xsq.divideEvenly(8, 10);
//        int np = readers.size();
//        int i;
//        pthread_t thread_id[np];
//        for (i = 0; i < np;  i++) {
//            test_read[i] = &(readers[i]);
//            pthread_create( &thread_id[i], NULL, &ppppp, (void *) i);
//        }
//
//        for (i = 0; i < np;  i++) {
//            pthread_join(thread_id[i], NULL);
//        }
//
//    }

};



class ChenyTest: public CppUnit::TestFixture {
public :
	static CppUnit::Test *suite() {
		CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("YongzhisTests");
		suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("scribble", &XsqTest::scribble));
    	//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("perfXsqOnlyTest", &XsqTest::perfXsqOnlyTest));
		/*  Yongzhi, register your tests here */
		return suiteOfTests;
	}
};


class DimaTests: public CppUnit::TestFixture {
public :
	static CppUnit::Test *suite() {
		CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("DimaTests");
		suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bigXsqWriterTest", &XsqTest::bigXsqWriterTest));
		return suiteOfTests;
	}
};

class MullermwTest: public CppUnit::TestFixture {
public :
	static CppUnit::Test *suite() {
		CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("MullermwTests");
		//Time consuming tests
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("writingInParallel", &XsqTest::writingInParallel));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("chengyongsFileTest", &XsqTest::chengyongsFileTest));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("sowmisFileTest", &XsqTest::sowmisFileTest));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bigXsqWriterTest", &XsqTest::bigXsqWriterTest));

		//Debugging tests
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("zhengsBug", &XsqTest::zhengsBug));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug20100117", &XsqTest::bug20100117));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug20110126", &XsqTest::bug20110126));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug20110201", &XsqTest::bug20110201));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("isBlankTest", &XsqTest::isBlankTest));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug20110207", &XsqTest::bug20110207));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("zhengsPartitionTest", &XsqTest::zhengsPartitionTest));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug20110216", &XsqTest::bug20110216));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("filteringTest", &XsqTest::filteringTest));
    	//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("readFilteringTest", &XsqTest::readTrimTest));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("regressPartitioningError20110221", &XsqTest::regressPartitioningError20110221));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug_20020302_panel12_problem", &XsqTest::bug_20020302_panel12_problem));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug_1489_maxXY_problem", &XsqTest::bug_1489_maxXY_problem));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("bug_1478_longReads_problem", &XsqTest::bug_1478_longReads_problem));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("first_5500_bc_file", &XsqTest::first_5500_bc_file));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("refcorProblem", &XsqTest::refcorProblem));
		suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("missingTagTest", &XsqTest::missingTagTest));
		//suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("issue02307", &XsqTest::issue02307));
		return suiteOfTests;
	}
};

class UnitTests: public CppUnit::TestFixture {
public :
	static CppUnit::Test *suite() {
		CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("UnitTests");
  	    suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("scribble", &XsqTest::scribble));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("colorEncodingTest", &XsqTest::colorEncodingTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("inspectXsqTest", &XsqTest::inspectXsqContent));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("partialIterationTest", &XsqTest::partialIterationTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("partitionTest", &XsqTest::partitionTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("splittingTest", &XsqTest::splittingTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("xsqWriterTest", &XsqTest::xsqWriterTest));
        suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("xsqMultiWriter", &XsqTest::xsqMultiWriter));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("setCallsTest", &XsqTest::setCallsTest));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("editInPlaceTest", &XsqTest::editInPlaceTest));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("filteringTest", &XsqTest::filteringTest));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("readFilteringTest", &XsqTest::readFilteringTest));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("readTrimTest", &XsqTest::readTrimTest));
    	suiteOfTests->addTest(new CppUnit::TestCaller<XsqTest>("missingTagTest", &XsqTest::missingTagTest));
    	return suiteOfTests;
	}
};


#endif //TEST_HPP_
