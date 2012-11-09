/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * exercise_lts_api.cpp
 *
 *  Created on: Sep 7, 2010
 *      Author: mullermw
 */
#include "lts/lts_reader.hpp"

using namespace ltstools;
using namespace std;

int exercise_lts_api(int argc, char **argv) {

	map<string, string> barcode2lib;

	// Define the lts content.
	LtsApi api;
	api = api.addLtsFile("file1", barcode2lib);
	api = api.addLtsFile("file2", barcode2lib);
	api = api.addLtsFile("file3", barcode2lib);
	Lts lts = api.newLts();

	// Examine high level lts data
	const vector<Lane> lanes = lts.listLanes();
	for (vector<Lane>::fragment_const_iterator i = lanes.begin(); i != lanes.end(); ++i) {
		Lane lane = *i;
		LtsMetadata metadata = lane.getMetadata();
		metadata.getApplicationType();
		metadata.getFileCreationTime();
		metadata.getFileVersion();
		metadata.getHDF5Version();
		metadata.getInstrumentSerialNumber();
		metadata.getInstrumentType();
		metadata.getLaneNumber();
		metadata.getLibraryType();
		metadata.getRunName();
		metadata.getSampleName();
		metadata.getSoftware();
	}
	const vector<Library> libraries = lts.listLibraries();
	for (vector<Library>::fragment_const_iterator i = libraries.begin(); i != libraries.end(); ++i) {
		Library library = *i;
		library.getName();
	}
	const vector<Panel> panels = lts.listPanels();
	for (vector<Panel>::fragment_const_iterator i = panels.begin(); i != panels.end(); ++i) {
		Panel panel = *i;
	}

	//Look at the panels, decide how to slice.
	const vector<Panel> allPanels = lts.listPanels();
	const unsigned int chunksize = allPanels.size() / 8;

	//Declare the iteration range and iterate over reads.
	LaneLibPanelRangePredicate pred("MYLANE123", "MYLIB123", 1, chunksize);
	Reads reads = lts.getReads(pred, F3);
	for (Reads::fragment_const_iterator i = reads.begin(); i != reads.end(); ++i) {
		const FragmentImpl read = *i;
		read.getBeadId();
		read.getSequence();
		read.getQualities();
		//do something with read.
	}

	return 1;
}
