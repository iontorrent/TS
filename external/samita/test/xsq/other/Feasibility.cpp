/*
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * Main.cpp
 *
 *  Created on: Sep 2, 2010
 *      Author: mullermw
 */

/*
 * Main.cpp
 *
 *  Created on: Aug 10, 2010
 *      Author: mullermw
 */
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <stdlib.h>
#include <hdf5.h>
#include <boost/timer.hpp>


using namespace std;

#define FALSE   0
#define PANEL_NUMBER        1
#define PRIMER_BASE 'T'
#define CALL_AND_QV_DATASET "Panels/0001/F3/CallAndQV"
#define XY_DATASET          "Panels/0001/Beads/xyLocations"
#define READS_QUAL "reads.qual"
#define READS_CSFASTA "reads.csfasta"

//Bypasses the separate_call_and_qv functionality. Enabling this will
//result in bad call and qv values.
#define SKIP_SEPRARATE_CALL_AND_QV false


typedef int (*handle_read_fp)(const char* panel, unsigned short int* xy, unsigned int length, const char* calls, unsigned int* qvs);

herr_t print_h_object_info(hid_t loc_id, const char *name, void *opdata);
herr_t read_panels(hid_t loc_id, const char *name, void *opdata);
int separate_call_and_qv(const unsigned char& callAndQV, unsigned int& call, unsigned int& qv);

int to_csfasta(const char *filename);
int process_csfasta_file(const char *filename, handle_read_fp);
int speed_test(const char *filename);
int print_all_call_and_qv();
int iterate_over_panels(const char *filename);

ofstream* csfasta_out = NULL;
ofstream* quals_out = NULL;

int main(int argc, char **argv) {

	string command(argv[1]);

	if (command == "speed_test") return speed_test(argv[2]);

	if (command == "dump_call_and_qv") return print_all_call_and_qv();

	if (command == "iterate_over_panels") return iterate_over_panels(argv[2]);

	if (command == "to_csfasta") return to_csfasta(argv[2]);

	if (command == "read_csfasta") return process_csfasta_file(argv[2], NULL);

	cerr << "Unrecognized command: " + command << endl;

	return -1;
}

herr_t print_h_object_info(hid_t loc_id, const char *name, void *opdata) {
    H5G_stat_t statbuf;

    /*
     * Get type of the object and display its name and type.
     * The name of the object is passed to this function by
     * the Library. Some magic :-)
     */
    H5Gget_objinfo(loc_id, name, FALSE, &statbuf);
    switch (statbuf.type) {
    case H5G_GROUP:
         printf(" Object with name %s is a group \n", name);
         break;
    case H5G_DATASET:
         printf(" Object with name %s is a dataset \n", name);
         break;
    case H5G_TYPE:
         printf(" Object with name %s is a named datatype \n", name);
         break;
    default:
         printf(" Unable to identify an object ");
    }
    return 0;
}

herr_t read_panels(hid_t loc_id, const char *name, void *opdata) {
    H5G_stat_t statbuf;

    H5Gget_objinfo(loc_id, name, FALSE, &statbuf);
    switch (statbuf.type) {
    case H5G_GROUP:
         printf(" Object with name %s is a group \n", name);
         break;
    default:
    	printf(" Unexpected Object: %s \n", name);
    	return 0;
    }
    clog << "about to open" << endl;
    hid_t dset = H5Dopen (loc_id, CALL_AND_QV_DATASET);
    clog << "opening" << endl;
    hid_t dcpl = H5Dget_create_plist (dset);
    H5D_layout_t layout = H5Pget_layout (dcpl);
    printf ("\nStorage layout for %s is: ", CALL_AND_QV_DATASET);
    switch (layout) {
        case H5D_COMPACT:
            printf ("H5D_COMPACT\n");
            break;
        case H5D_CONTIGUOUS:
            printf ("H5D_CONTIGUOUS\n");
            break;
        case H5D_CHUNKED:
            printf ("H5D_CHUNKED\n");
            break;
        default:
        	break;
    }

    H5Dclose(dset);
    return 0;
}

/**
 *  callAndQV  call is in the 2 most significant bits:  11000000
 *             qv   is in the 6 least significant bits: 00111111
 */
int separate_call_and_qv(const unsigned char& callAndQV, unsigned int& call, unsigned int& qv) {
#if SKIP_SEPRARATE_CALL_AND_QV
	call = 0;
	qv = 0;
	return 1;
#else
	call =  callAndQV >> 6;
	qv   =  callAndQV &0x3f;
	return 1;
#endif
}

void print_dataset_info(hid_t & dset)
{
    hid_t dcpl = H5Dget_create_plist(dset);
    H5D_layout_t layout = H5Pget_layout(dcpl);
    cerr << "\nStorage layout for " << CALL_AND_QV_DATASET << " is:";
    switch (layout){
        case H5D_COMPACT:
            cerr << "H5D_COMPACT";
            break;
        case H5D_CONTIGUOUS:
            cerr << "H5D_CONTIGUOUS";
            break;
        case H5D_CHUNKED:
            cerr << "H5D_CHUNKED";
        default:
            break;
    }
    cerr << endl;

	hsize_t dims[2] = {0, 0};
	hid_t space_id = H5Dget_space(dset);
	H5Sget_simple_extent_dims(space_id, dims, NULL);
	cerr << "Dimensions: [" << dims[0] << "x" << dims[1] << "]" << endl;
}

int log_message(const char* msg) {
	clog << clock() << " " << msg <<endl;
	return 0;
}

int log_message(string msg) {
	return log_message(msg.c_str());
}

vector<string> get_panel_names(hid_t file_id) {
	hid_t panelGroup = H5Gopen(file_id, "/DefaultLibrary");
	if (panelGroup < 0) throw domain_error("Can't find group: /DefaultLibrary");
 	hsize_t num_obj;
	H5Gget_num_objs(panelGroup, &num_obj);
	vector<string> panelNames;
	unsigned int buffer_size = 10;
	char* buffer = new char[buffer_size];
	for (unsigned int i=0; i!=num_obj; i++) {
		H5Gget_objname_by_idx(panelGroup, i, buffer, buffer_size);
		string panelName(buffer);
		panelNames.push_back(panelName);
	}
	delete buffer;
	H5Gclose(panelGroup);
	return panelNames;
}

int process_lts_file(const char *filename, handle_read_fp handle_read) {

	herr_t status;
	hsize_t dims[2] = {0, 0};
	hid_t file_id           = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) throw new domain_error("unable to open file");
	log_message("Iterating over all panels");
	clock_t start_time = clock();
	unsigned int num_beads_processed = 0;
	vector<string> panelNames = get_panel_names(file_id);
	for (vector<string>::const_iterator iter = panelNames.begin();
		 iter != panelNames.end(); iter++) {

		stringstream callAndQvPath, xyPath;
		callAndQvPath << "/DefaultLibrary/" << *iter <<  "/F3/ColorCallQV";
		xyPath << "/DefaultLibrary/" << *iter << "/Fragments/yxLocation";
	       
		hid_t dset_callAndQv    = H5Dopen(file_id, callAndQvPath.str().c_str());
		hid_t dset_xy           = H5Dopen(file_id, xyPath.str().c_str());
		hid_t space_id          = H5Dget_space(dset_callAndQv);
		status = H5Sget_simple_extent_dims(space_id, dims, NULL);

		/*
		 * Allocate array of pointers to rows.
		 */
		unsigned      char** callAndQvData;
		              char** callData;
		unsigned       int**  qvData;
		unsigned short int**  xyData;

		callAndQvData = new unsigned      char*[dims[0]];
		callData      = new               char*[dims[0]];
		qvData        = new unsigned       int*[dims[0]];
		xyData        = new unsigned short int*[dims[0]];

		/*
		 * Allocate space for integer data.
		 */
		callAndQvData[0] = new unsigned       char[dims[0] * dims[1]];
		callData[0]      = new                char[dims[0] * dims[1]];
		qvData[0]        = new unsigned        int[dims[0] * dims[1]];
		xyData[0]        = new unsigned short  int[dims[0] * 2];

		/*
		 * Set the rest of the pointers to rows to the correct addresses.
		 */
		for (unsigned int i=1; i<dims[0]; i++) {
			callAndQvData[i] = callAndQvData[0] + i * dims[1];
			callData[i]      = callData     [0] + i * dims[1];
			qvData[i]        = qvData       [0] + i * dims[1];
			xyData[i]        = xyData       [0] + i * 2;
		}

		/*
		 * Read the data.
		 */
		status = H5Dread (dset_callAndQv, H5T_NATIVE_UCHAR_g, H5S_ALL, H5S_ALL, H5P_DEFAULT,
					callAndQvData[0]);
		H5Dclose(dset_callAndQv);

		status = H5Dread (dset_xy,        H5T_NATIVE_INT16_g, H5S_ALL, H5S_ALL, H5P_DEFAULT,
					xyData[0]);
		H5Dclose(dset_xy);

		//Separate call and qv, handle, read_info..
		for (unsigned int i=0; i!=dims[0]; i++) {
			for (unsigned int j=0; j!=dims[1]; j++) {
				unsigned int call, qv;
				separate_call_and_qv(callAndQvData[i][j], call, qv);
				callData[i][j] = call;
				qvData  [i][j] = qv;
			}
			if (handle_read != NULL) handle_read((*iter).c_str(), xyData[i], dims[1], callData[i], qvData[i]);
		}

		delete callAndQvData[0];
		delete callData[0];
		delete qvData[0];
		delete xyData[0];
		delete callAndQvData;
		delete callData;
		delete qvData;
		delete xyData;
		clock_t elapsed_time = clock() - start_time;
		float elapsed_time_sec = (float)elapsed_time/CLOCKS_PER_SEC;
		num_beads_processed += dims[0];
		float beads_per_sec = num_beads_processed / elapsed_time_sec;
		char buffer[100];
		sprintf(buffer, "finished panel:%s %d beads, %0.2f s, %0.0f beads/s.", (*iter).c_str(), num_beads_processed, elapsed_time_sec, beads_per_sec);
		log_message(buffer);
	}
	log_message("Finished Iteration");
	return 1;
}

int csfasta_writer(const char* panel, unsigned short int* xy, unsigned int length, const char* calls, unsigned int* qvs) {
	stringstream desc_line;
	desc_line << ">" << atoi(panel) << "_" << xy[0] << "_" << xy[1] << "_" << "F3";
	*csfasta_out << desc_line.str().c_str() << endl;
	*quals_out << desc_line.str().c_str() << endl;
	*csfasta_out << PRIMER_BASE;
	for (unsigned int i=0; i<length; i++) {
		*csfasta_out << (int)calls[i];
		*quals_out << qvs[i];
		if (i+1<length) *quals_out << ' ';
	}
	*csfasta_out << endl;
	*quals_out << endl;
	return 1;
}


/* The subcommands */

/**
 * Reads each panel into memory and discards it.
 */
int speed_test(const char *filename) {

	process_lts_file(filename, NULL);
	log_message("Finished speed_test.");
	return 0;
}

/**
 * Prints the reads out to csfasta format.
 */
int to_csfasta(const char *filename) {
	csfasta_out = new ofstream(READS_CSFASTA);
	quals_out = new ofstream(READS_QUAL);
	process_lts_file(filename, csfasta_writer);
	delete csfasta_out;
	delete quals_out;
	log_message("Finished to_csfasta");
	return 0;
}

/**
 * Reads in a csfasta file, parses it to
 */
int process_csfasta_file(const char *filename, handle_read_fp handle_read) {
	ifstream csfasta_in(filename);
	string line0, seq;
	int lastPanelNum = -1;
	int panelNum;
	int num_beads_processed = 0;
	clock_t start_time = clock();
	char buffer100[100];
	while (csfasta_in >> line0) {
		stringstream stream(line0);
		string panel,x,y;
		unsigned short xy[2];
		getline(stream, panel, '>');
		getline(stream, panel, '_');
		getline(stream, x, '_');
		getline(stream, y, '_');
		xy[0] = atoi(x.c_str());
		xy[1] = atoi(y.c_str());
		csfasta_in >> seq;
		if (handle_read != NULL) handle_read(panel.c_str(), xy, seq.size() - 1, seq.substr(1).c_str(), NULL);
		panelNum = atoi(panel.c_str());
		num_beads_processed++;
		if (panelNum != lastPanelNum) {
			clock_t elapsed_time = clock() - start_time;
			float elapsed_time_sec = elapsed_time / CLOCKS_PER_SEC;
			float beads_per_sec = num_beads_processed / elapsed_time_sec;
			sprintf(buffer100, "finished panel:%s %d beads, %0.2f s, %0.0f beads/s.", panel.c_str(), num_beads_processed, elapsed_time_sec, beads_per_sec);
			log_message(buffer100);
			lastPanelNum = panelNum;
		}
	}
	csfasta_in.close();
	return 0;
}


/**
 * Prints information about the panels.
 */
int iterate_over_panels(const char *filename) {
	hid_t       file_id;   /* file identifier */
	file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	H5Giterate(file_id, "/Panels", NULL, print_h_object_info, NULL);
	return 0;
}

/**
 * Prints all possible callAndQV values, their associated chars and separated call and qv values.
 */
int print_all_call_and_qv() {
	for (unsigned int i = 0; i<256; i++) {
		unsigned char callAndQv = i;
		unsigned int call,qv;
		separate_call_and_qv(callAndQv, call, qv);
		cout << i << " " << callAndQv << " " << call << " " << qv << endl;
	}
	clog << "finished successfully";
	return 1;
}
