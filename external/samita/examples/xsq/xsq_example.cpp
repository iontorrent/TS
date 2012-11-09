/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 *
 *  Created on: Sep 21, 2010
 *      Author: mullermw
 */

#include <samita/xsq/xsq_io.hpp>

using namespace lifetechnologies;
using namespace std;

/*
 * Iterate over all F3 reads in a file.
 */
int example1() {
	LtsReader lts;
	lts.open("file0.h5:/F3");
	for (LtsReader::fragment_const_iterator read = lts.begin(); read != lts.end(); ++read) {
		string name = read->getName();
		string seq = read->getSeqStr();
	}
	return 0;
}

/*
 * Iterate over all F3 reads in a three Lts files.
 */
int example2() {
	LtsReader lts;
	lts.open("file0.h5:/F3");
	lts.open("file1.h5:/F3");
	lts.open("file2.h5:/F3");
	for (LtsReader::fragment_const_iterator read = lts.begin(); read != lts.end(); ++read) {
		string name = read->getName();
		string seq = read->getSeqStr();
	}
	return 0;
}

/*
 * Iterate over all reads from Barcode_1 in 2 lts files.
 */
int example3() {
	LtsReader lts;
	lts.open("minimal_lts_barcode_0.h5:/F3/Barcode_1");
	lts.open("minimal_lts_barcode_1.h5:/F3/Barcode_1");
	for (LtsReader::fragment_const_iterator read = lts.begin(); read != lts.end(); ++read) {
		string name = read->getName();
		string seq = read->getSeqStr();
	}
	return 0;
}

/*
 * Iterate over 2 Barcodes in 1 lts file.
 */
int example4() {
	LtsReader lts;
	lts.open("minimal_lts_barcode_0.h5:/F3/Barcode_1");
	lts.open("minimal_lts_barcode_0.h5:/F3/Barcode_2");
	for (LtsReader::fragment_const_iterator read = lts.begin(); read != lts.end(); ++read) {
		string name = read->getName();
		string seq = read->getSeqStr();
	}
	return 0;
}

/*
 * Iterate over panels 1-100 from Barcode_1 in 1 lts file.
 */
int example5() {
	LtsReader lts;
	lts.open("minimal_lts_barcode_0.h5:/F3/Barcode_1/1-100");
	for (LtsReader::fragment_const_iterator read = lts.begin(); read != lts.end(); ++read) {
		string name = read->getName();
		string seq = read->getSeqStr();
	}
	return 0;
}

class ExampleFilter {
	public:
	ExampleFilter() {}
	bool operator()(FragmentImpl const &a) const {
		return a.getBeadFilterFlag() == 0 && a.getReadFilterFlag() == 0;
	}
};

int example6() {

	ExampleFilter filter;

	LtsReader lts;
	lts.open("minimal_lts_barcode_0.h5");
	LtsReader::filter_iterator<ExampleFilter> iter(filter, lts.begin(), lts.end());
	LtsReader::filter_iterator<ExampleFilter> end(filter, lts.end(), lts.end());

	for (LtsReader::filter_iterator<ExampleFilter> read = iter; read != end; ++read) {
		//do something with read.
	}
	return 0;
}

int main(int argc, char **argv) {

}
