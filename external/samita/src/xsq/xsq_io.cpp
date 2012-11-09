/* -*- Mode: C++; indent-tabs-mode: 't; c-basic-offset: 8; tab-width: 8-*-   vi:set noexpandtab ts=8 sw=8:
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * xsq_io.cpp
 *
 *  Created on: Sep 27, 2010
 *      Author: mullermw
 */
#include "samita/xsq/xsq_io.hpp"
#include "hdf5_hl.h"
#include <boost/thread/mutex.hpp>
#include <cassert>
#include <algorithm>

//#include <log4cxx/logger.h>

namespace lifetechnologies
{

using namespace boost;
using namespace std;

static boost::mutex io_mutex;
//static log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("lifetechnologies.samita.xsq");

const string _HDATASET_BASECALLQV = "BaseCallQV";
const string _HDATASET_COLORCALLQV = "ColorCallQV";
const string _HDATASET_NAME_YX = "yxLocation";
const string _HDATASET_NAME_FILTERING = "Filtering";
const string _HDATASET_NAME_TRIM_START_LENGTH = "TrimStartLength";
const string _HDATASET_NAME_TRIM_END_LENGTH = "TrimEndLength";
const string _HGROUP_NAME_FRAGMENTS = "Fragments";
const string _HGROUP_NAME_RUN_METADATA = "RunMetadata";
const string _HGROUP_NAME_TAG_DETAILS = "TagDetails";
const string _HGROUP_NAME_INDEXING = "Indexing";
const string _HGROUP_NAME_UNCLASSIFIED = "Unclassified";
const string _HGROUP_NAME_DEFAULT_LIBRARY = "DefaultLibrary";
const string _HGROUP_NAME_COLOR_ENCODING = "DatasetColorEncoding";
const string _HATTRIBUTE_NAME_TAG_SEQUENCE = "TagSequence";
const string _HATTRIBUTE_NAME_NUM_BASE_CALLS = "NumBaseCalls";
const string _HATTRIBUTE_NAME_FRAGMENT_COUNT = "FragmentCount";
const string _HATTRIBUTE_NAME_IS_BASE_PRESENT = "IsBasePresent";
const string _HATTRIBUTE_NAME_IS_COLOR_PRESENT = "IsColorPresent";
const string _HATTRIBUTE_LIBRARY_NAME = "LibraryName";
const string _HATTRIBUTE_UUID = "UUID";
const string _HATTRIBUTE_INDEX_NAME = "IndexName";
const string _HATTRIBUTE_NAME_ID = "IndexID";
const size_t _HGROUP_DEFAULT_SIZE_HINT = 50;

const size_t NPOS = std::numeric_limits<size_t>::max();
const static	PanelFragmentsIterator<Fragment const> end_of_panel;
const static	FragmentIterator<Fragment const> end_of_fragment;
 
//Supporting Utilities

//Represents a node in an HDF file.
class HDFNode {
	bool rooted;
	vector<string> elements;
public :
	HDFNode(string path) : rooted(path[0] == '/') { string_util::tokenize(path, "/", elements); }
	HDFNode(HDFNode parent, string name) {
		copy(parent.elements.begin(), parent.elements.end(), back_inserter(this->elements));
		string_util::tokenize(name, "/", elements);
	}
	size_t size() const { return elements.size(); }
	size_t depth() const { return elements.size(); }
	HDFNode first() const {
		return HDFNode(elements.begin(), elements.begin()+1, rooted);
	}

	HDFNode ancestor(size_t const& num) const {
		return HDFNode(elements.begin(), elements.end() - num, rooted );
	}
	HDFNode parent() const {
		return ancestor(1);
	}
	HDFNode relativeTo( size_t const& num) const {
		return HDFNode(elements.begin()+num, elements.end(), false);
	}
	string name() const { return *(elements.end() - 1); }
	string str() const {
		stringstream buf;
		if (rooted) buf << "/";
		for (vector<string>::const_iterator it = elements.begin(); it != elements.end(); ++it)
			buf << *it << (it + 1 != elements.end() ? "/" : "");
		return buf.str();
	}
private :
	HDFNode(vector<string>::const_iterator begin, vector<string>::const_iterator end, bool const& rooted) : rooted(rooted), elements(begin, end) {}
};

//Closes the enclosed hid_t in the destructor.
class HIDCloser {
	const hid_t m_obj;
public :
	HIDCloser(hid_t const& _obj) : m_obj(_obj) {}
	~HIDCloser() {
		if (m_obj < 0) return;
		H5I_type_t type = H5Iget_type(m_obj);
		switch (type) {
		case H5I_FILE :
			H5Fclose(m_obj);
			break;
		case H5I_DATATYPE :
			H5Tclose(m_obj);
			break;
		case H5I_GROUP :
			H5Gclose(m_obj);
			break;
		case H5I_DATASPACE :
			H5Sclose(m_obj);
			break;
		case H5I_DATASET :
			H5Dclose(m_obj);
			break;
		case H5I_ATTR :
			H5Aclose(m_obj);
			break;
		default:
			cerr << "HIDCloser can't close objects of type " << type << endl;
		}
	}
};

hid_t createHdfStringType(size_t const& size) {
	hid_t hid = H5Tcopy(H5T_C_S1_g);
	if (hid < 0) throw XsqException("Error in H5Tcopy().");
	if (0 > H5Tset_size(hid, size)) throw XsqException("Error in H5Tset_size().");
	return hid;
}

const hid_t _HDATASPACE_SCALAR = H5Screate(H5S_SCALAR);
HIDCloser closer0(_HDATASPACE_SCALAR);
const hid_t _HDATATYPE_STRING = createHdfStringType(255);
HIDCloser closer1(_HDATATYPE_STRING);

//Support query on the HDF5 Table ColorEncoding.
struct ColorEncodingTableEntry {
	char  datasetName[255];
	int32_t offset;
	char  encoding[255];
	uint32_t stride;
	uint32_t numColorCalls;
};
#define NFIELDS_COLOR_ENCODING_TABLE 5
size_t COLOR_ENCODING_TABLE_DST_OFFSETS[NFIELDS_COLOR_ENCODING_TABLE] = {
							     HOFFSET( ColorEncodingTableEntry, datasetName),
							     HOFFSET( ColorEncodingTableEntry, offset),
							     HOFFSET( ColorEncodingTableEntry, encoding),
							     HOFFSET( ColorEncodingTableEntry, stride),
							     HOFFSET( ColorEncodingTableEntry, numColorCalls) };
size_t COLOR_ENCODING_TABLE_DST_SIZES[NFIELDS_COLOR_ENCODING_TABLE] = {
								sizeof( ColorEncodingTableEntry().datasetName ),
								sizeof( ColorEncodingTableEntry().offset ),
								sizeof( ColorEncodingTableEntry().encoding ),
								sizeof( ColorEncodingTableEntry().stride ),
								sizeof( ColorEncodingTableEntry().numColorCalls ) };
hid_t COLOR_ENCODING_TABLE_FIELD_TYPES[NFIELDS_COLOR_ENCODING_TABLE] = {
	_HDATATYPE_STRING,
	H5T_STD_I32LE_g,
	_HDATATYPE_STRING,
	H5T_STD_U32LE_g,
	H5T_STD_U32LE_g
};

const char* COLOR_ENCODING_TABLE_FIELD_NAMES[NFIELDS_COLOR_ENCODING_TABLE] = {
	"DataSetName", 	"Offset", "Encoding", "Stride", "NumColorCalls"
};

//Get the name of an HDF5 object as a string.
string getName(hid_t loc_id) {
	size_t nameSize = 0;
	do {
		nameSize += 100;
		char* name;
		shared_array<char> p(name = new char[nameSize]); //ensures delete[]
		H5Iget_name(loc_id, name, nameSize);
		if (strlen(name) != nameSize-1) return string(name);
	} while (nameSize < 1000000);
	throw XsqException("Can't fetch names with size > 1000000");
}

hid_t openHdfFile(string const& filename, bool const& readOnly) {
	hid_t hid = H5Fopen(filename.c_str(), readOnly ? H5F_ACC_RDONLY : H5F_ACC_RDWR, H5P_DEFAULT);
	if (hid < 0) throw file_format_exception(filename, "xsq");
	return hid;
}

hid_t openHdfGroup(hid_t const& loc_id, string const& name) {
	hid_t hid = H5Gopen(loc_id, name.c_str());
	if (hid < 0) {
		stringstream msg;
		msg << "Error opening Hdf Group: " << name;
		throw XsqException(msg.str());
	}
	return hid;
}

//opens a child of loc_id and returns it's hid_t
hid_t openAnyChildGroup(hid_t const& loc_id) {
	hsize_t num_objs;
	H5Gget_num_objs(loc_id, &num_objs);
	H5O_info_t info;
	string loc_name = getName(loc_id);
	hsize_t i=0;
	for (;i!=num_objs; ++i) {
		if (H5Oget_info_by_idx(loc_id, loc_name.c_str(), H5_INDEX_NAME, H5_ITER_NATIVE, i, &info, H5P_DEFAULT ) < 0) {
			stringstream msg;
			msg << "Error getting child information in '" << loc_name;
			throw XsqException(msg.str());
		}
		if (info.type == H5O_TYPE_GROUP) break;
	}
	if (i == num_objs) {
		stringstream msg;
		msg << "Group has no child groups: " << loc_name;
		throw XsqException(msg.str());
	}
	size_t childNameSize = 0;
	do {
		childNameSize+=100;
		char* childName;
		shared_array<char> p(childName = new char[childNameSize]); //ensures delete[]
		H5Lget_name_by_idx(loc_id, loc_name.c_str(), H5_INDEX_NAME, H5_ITER_NATIVE, i, childName, childNameSize, H5P_DEFAULT);
		if (strlen(childName) != childNameSize-1) return H5Gopen(loc_id, childName);
	} while (childNameSize < 1000000);
	throw XsqException("Can't open pathsnames with size > 1000000");
}

hid_t openHdfDataset(hid_t const& loc_id, string const& name) {
	hid_t hid = H5Dopen(loc_id, name.c_str());
	if (hid < 0 ) {
		stringstream msg;
		msg << "error opening HDF Dataset " << name;
		throw XsqException(msg.str());
	}
	return hid;
}

void getHdfTableInfo(hid_t loc_id, string name, hsize_t* nfields, hsize_t* nrecords) {
	if (0 > H5TBget_table_info(loc_id, name.c_str(), nfields, nrecords)) {
		stringstream msg;
		msg << "Error calling H5TBget_table_info().";
		throw XsqException(msg.str());
	}
}

//gets a string typed attribute of an HDF5 object as a string.
string getStringAttribute(hid_t const& loc_id, string const& attr_name, string const& _default) {
	if (!H5Aexists(loc_id, attr_name.c_str())) return _default;
	hid_t attr_id = H5Aopen(loc_id, attr_name.c_str(), H5P_DEFAULT);
	if (attr_id < 0) return _default;
	HIDCloser closer0(attr_id);
	hid_t type_id = H5Aget_type(attr_id);
	if (type_id < 0) return _default;
	HIDCloser closer1(type_id);
	if (H5Tget_class(type_id) != H5T_STRING) {
		stringstream msg;
		msg << "Attribute is not a string: " << attr_name;
		throw XsqException(msg.str());
	}
	hsize_t attr_size = H5Aget_storage_size(attr_id);
	char* buf = new char[attr_size];
	shared_array<char> sa(buf); //Ensure delete[]
	if (H5Aread(attr_id, type_id, buf) < 0) {
		stringstream msg;
		msg << "Error reading attribute: " << attr_name;
		throw XsqException(msg.str());
	}
	return string_util::trim(string(buf));
}

//gets an integer typed attribute of an HDF5 object as a uint32_t
uint32_t getUInt32Attribute(hid_t const& loc_id, string const& attr_name) {
	if (!H5Aexists(loc_id, attr_name.c_str())) {
		stringstream msg;
		msg << "The attribute of " << getName(loc_id) << " named '" << attr_name << "' does not exist." << endl;
		throw XsqException(msg.str());
	}
	hid_t attr_id = H5Aopen(loc_id, attr_name.c_str(), H5P_DEFAULT);
	if (attr_id < 0) {
		stringstream msg;
		msg << "There was a problem getting the attribute of " << getName(loc_id) << " named '" << attr_name << "'." << endl;
		throw XsqException(msg.str());
	}
	HIDCloser closer0(attr_id);
	hid_t type_id = H5Aget_type(attr_id);
	if (type_id < 0) {
		stringstream msg;
		msg << "There was a problem getting the type for the attribute of " << getName(loc_id) << " named '" << attr_name << "'." << endl;
		throw XsqException(msg.str());
	}
	HIDCloser closer1(type_id);
	if (H5Tget_class(type_id) != H5T_INTEGER) {
		stringstream msg;
		msg << "The attribute of " << getName(loc_id) << " named '" << attr_name << "' is not an integer." << endl;
		throw XsqException(msg.str());
	}
	uint32_t value;
	if (H5Aread(attr_id, type_id, &value) < 0) {
		stringstream msg;
		msg << "Error reading attribute: " << attr_name;
		throw XsqException(msg.str());
	}
	return value;
}

//gets an integer typed attribute of an HDF5 object as a uint32_t
uint32_t getUInt8Attribute(hid_t const& loc_id, string const& attr_name) {
	if (!H5Aexists(loc_id, attr_name.c_str())) {
		stringstream msg;
		msg << "The attribute of " << getName(loc_id) << " named '" << attr_name << "' does not exist." << endl;
		throw XsqException(msg.str());
	}
	hid_t attr_id = H5Aopen(loc_id, attr_name.c_str(), H5P_DEFAULT);
	if (attr_id < 0) {
		stringstream msg;
		msg << "There was a problem getting the attribute of " << getName(loc_id) << " named '" << attr_name << "'." << endl;
		throw XsqException(msg.str());
	}
	HIDCloser closer0(attr_id);
	hid_t type_id = H5Aget_type(attr_id);
	if (type_id < 0) {
		stringstream msg;
		msg << "There was a problem getting the type for the attribute of " << getName(loc_id) << " named '" << attr_name << "'." << endl;
		throw XsqException(msg.str());
	}
	//HIDCloser closer1(type_id);
	if (H5Tget_class(type_id) != H5T_INTEGER) {
		stringstream msg;
		msg << "The attribute of " << getName(loc_id) << " named '" << attr_name << "' is not an integer." << endl;
		throw XsqException(msg.str());
	}
	uint8_t value;
	if (H5Aread(attr_id, type_id, &value) < 0) {
		stringstream msg;
		msg << "Error reading attribute: " << attr_name;
		throw XsqException(msg.str());
	}
	return value;
}


void getHdfGroupInfo(hid_t group_id, H5G_info_t *group_info) {
	if (H5Gget_info(group_id, group_info) < 0) {
		stringstream msg;
		msg << "Error Hdf Group info.";
		throw XsqException(msg.str());
	}
}

hid_t createHdfGroup(hid_t loc_id, string name) {
	hid_t hid = H5Gcreate(loc_id, name.c_str(), _HGROUP_DEFAULT_SIZE_HINT);
	if (hid < 0) {
		stringstream msg;
		msg << "Error creating group: '" << name << "'." << endl;
		throw XsqException(msg.str());
	}
	return hid;
}

hid_t createHdfDataspace(int rank, const hsize_t* dims) {
	hid_t hid = H5Screate_simple(rank, dims, NULL);
	if (hid < 0) {
		stringstream msg;
		msg << "Error creating dataspace.";
		throw XsqException(msg.str());
	}
	return hid;
}

hid_t createHdfDataset(hid_t const& loc_id, string const& name, hid_t const& type_id, hid_t const& space_id) {
	hid_t hid = H5Dcreate(loc_id, name.c_str(), type_id, space_id, H5P_DEFAULT );
	if (hid < 0) {
		stringstream msg;
		msg << "Error creating dataset: '" << name << "'";
		throw XsqException(msg.str());
	}
	return hid;
}

void writeHdfDataset(hid_t const& dataset_id, hid_t const& mem_type_id, hid_t const& space_id, const void* buf) {
	if (H5Dwrite(dataset_id, mem_type_id, space_id, space_id, H5P_DEFAULT, buf) < 0) {
		stringstream msg;
		msg << "Error writing to dataset.";
		throw XsqException(msg.str());
	}
}

void overwriteHdfDataset(hid_t const& loc_id, string const& path, const void* buf) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering overwriteHdfDataset(): " << path << endl;
# endif
	const hid_t dataset_id = openHdfDataset(loc_id, path);
	const HIDCloser closer0(dataset_id);
	hid_t mem_type_id = H5Dget_type(dataset_id);
	HIDCloser closer1(mem_type_id);
	hid_t space_id = H5Dget_space(dataset_id);
	HIDCloser closer2(space_id);
	writeHdfDataset(dataset_id, mem_type_id, space_id, buf);
}

void readHdfTable(hid_t const& loc_id, string const& name, size_t dst_size, const size_t *dst_offset, const size_t *dst_sizes, void *dst_buf) {
	if (0 > H5TBread_table(loc_id, name.c_str(), dst_size, dst_offset, dst_sizes, dst_buf )) {
		stringstream msg;
		msg << "Error calling H5TBread_table().";
		throw XsqException(msg.str());
	}
}

void readHdfDataset(hid_t loc_id, hid_t mem_type_id, void *buf) {
	if (H5Dread(loc_id, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0) {
		stringstream msg;
		msg << "Error reading Hdf Dataset.";
		throw XsqException(msg.str());
	}
}

void visitHdf(hid_t const& object_id, H5O_iterate_t op, void *op_data ) {
	if (0 > H5Ovisit(object_id, H5_INDEX_NAME, H5_ITER_INC, op, op_data)) {
		stringstream msg;
		msg << "a problem occured while visiting hdf.";
		throw XsqException(msg.str());
	}
}

void setHdfIntAttribute(hid_t const& loc_id, string const& name, uint32_t const& value) {
	hid_t hid = H5Acreate(loc_id, name.c_str(), H5T_STD_U32LE_g, _HDATASPACE_SCALAR, H5P_DEFAULT);
	if (hid < 0) {
		stringstream msg;
		msg << "Error in H5Acreate(): " << name << "=" << value;
		throw XsqException(msg.str());
	}
	HIDCloser closer1(hid);
	if (0 > H5Awrite(hid, H5T_STD_U32LE_g, &value)) {
		stringstream msg;
		msg << "Error in H5Awrite(): " << name << "=" << value;
		throw XsqException(msg.str());
	}
}

// NB: Take values by value, not const ref. Must be resized to exactly 255 chars
void setHdfStringAttribute(hid_t const& loc_id, string const& name, string const& value) {
	hid_t hid = H5Acreate(loc_id, name.c_str(), _HDATATYPE_STRING, _HDATASPACE_SCALAR, H5P_DEFAULT);
	if (hid < 0) {
		stringstream msg;
		msg << "Error in H5Acreate(): " << name << "=" << value;
		throw XsqException(msg.str());
	}
	HIDCloser closer0(hid);
	// HDATATYPE_STRING is 255 chars. Values must be 255 chars (underlying buffer)
	char buf[255] = {'\0'};
	value.copy(buf, std::min(value.size(), static_cast<size_t>(254)));
	//buf[254] = '\0'; // Ensure null terminated - null preserved by 254 above
	if (0 > H5Awrite(hid, _HDATATYPE_STRING, buf)) {
		stringstream msg;
		msg << "Error in H5Awrite(): " << name << "=" << value;
		throw XsqException(msg.str());
	}
}

void createHdfTable(string const& name, hid_t const& loc_id, const char** field_names, size_t const& nfields,
		            size_t const& nrecords, hsize_t const& type_size, const size_t* field_offsets,
		            const hid_t* field_types, void* data) {
	H5TBmake_table(name.c_str(), loc_id, name.c_str(), nfields, nrecords, type_size, field_names, field_offsets, field_types, 10, NULL, 0, data);
}

bool checkHdfLinkExists(hid_t const& loc_id, string const& path) {
	const bool val = H5Lexists(loc_id, path.c_str(), H5P_DEFAULT);
	if (val < 0) {
		stringstream msg;
		msg << "Error in H5Lexists(" << loc_id << ",\"" << path << "\")";
		throw XsqException(msg.str());
	}
	return val;
}
//End Utilities

//XSQ API Class implementations
bool PanelRangeSpecifier::accept(PanelI const& panel) const {
	if (panel.getFilename() != this->getPath()) return false;
	PanelPosition position(panel.getPanelContainerIndex(), panel.getPanelNumber());
	return !(this->panelStart > position || this->panelEnd < position);
}

map<size_t, vector<PanelRangeSpecifier> > convert( vector<Panel> const& panels) {
	map<size_t, vector<PanelRangeSpecifier> > index;
	if (panels.empty()) return index;
	Panel firstPanel = panels.front();
	Panel lastPanel = firstPanel;
	for (vector<Panel>::const_iterator it=panels.begin()+1; it != panels.end(); ++it) {
		if (it->getFileIndex() != lastPanel.getFileIndex() ||
			it->getPanelContainerIndex() != lastPanel.getPanelContainerIndex() ||
			!it->isPrecededBy(lastPanel) ) {
			index[lastPanel.getFileIndex()].push_back(
					PanelRangeSpecifier(
							firstPanel.getFilename(),
							PanelRangeSpecifier::PanelPosition(firstPanel.getPanelContainerIndex(), firstPanel.getPanelNumber()),
							PanelRangeSpecifier::PanelPosition(lastPanel.getPanelContainerIndex(), lastPanel.getPanelNumber())
					)
			);
			firstPanel = *it;
		}
		lastPanel = *it;
	}
	index[lastPanel.getFileIndex()].push_back(
			PanelRangeSpecifier(
				firstPanel.getFilename(),
				PanelRangeSpecifier::PanelPosition(firstPanel.getPanelContainerIndex(), firstPanel.getPanelNumber()),
				PanelRangeSpecifier::PanelPosition(lastPanel.getPanelContainerIndex(), lastPanel.getPanelNumber())
			)
	);
# ifdef XSQ_READER_TRACE_ON
	Panel front = panels.front();
	Panel back = panels.back();
	cerr << front.getFileIndex() << "." << front.getPanelContainerIndex() << "." << front.getPanelNumber() << "..." << back.getFileIndex() << "." << back.getPanelContainerIndex() << "." <<  back.getPanelNumber() << " converted to " << endl;
	for (map<size_t, vector<PanelRangeSpecifier> >::const_iterator it = index.begin(); it != index.end(); ++it) {
		cerr << it->first << "= [" << endl;
		for (vector<PanelRangeSpecifier>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
			cerr << it2->str() << endl;
		}
		cerr << "]" << endl;
	}
#endif
	return index;
}

//Find all the panels and register them.
herr_t laneInfoOpenH5OvisitCallback(  hid_t o_id, const char *name_, const H5O_info_t *object_info, void *op_data)  {
# ifdef XSQ_READER_TRACE_ON
	cerr << "laneInfoOpenH5OvisitCallback(" << o_id << "," << name_ << ", xxx,xxx,xxx)" << endl;
# endif
	const string name(name_);
	const HDFNode hpath(string("/").append(name_).c_str());
	shared_ptr<LaneImpl> lane = *((shared_ptr<LaneImpl>*)op_data);

	//Ignore everything in the unclassified, indexing dirs.
	if (hpath.first().name()       == _HGROUP_NAME_UNCLASSIFIED ||
		hpath.first().name()       == _HGROUP_NAME_INDEXING ) return 0;

	//Register Tag Details
	if (object_info->type == H5O_TYPE_GROUP &&
		hpath.depth() == 3 &&
		hpath.ancestor(2).name() == _HGROUP_NAME_RUN_METADATA &&
		hpath.parent().name() == _HGROUP_NAME_TAG_DETAILS) {
		hid_t group_id = openHdfGroup(o_id, name_);
		HIDCloser closer0(group_id);

		XsqReadType readType = toReadType(hpath.name());
		lane->tagSequences[readType] = getStringAttribute(group_id, _HATTRIBUTE_NAME_TAG_SEQUENCE, "");
		lane->readTypeGroupNames[readType] = hpath.name();

		//Register BaseCalls if present.
		if (!H5Aexists(group_id, _HATTRIBUTE_NAME_IS_BASE_PRESENT.c_str()) || getUInt8Attribute(group_id, _HATTRIBUTE_NAME_IS_BASE_PRESENT)) {
			EncodingInfo encodingInfo;
			encodingInfo.hDatasetName = _HDATASET_BASECALLQV;
			encodingInfo.readType = readType;
			encodingInfo.encoding = BASE_ENCODING;
			encodingInfo.numCalls = getUInt32Attribute(group_id, _HATTRIBUTE_NAME_NUM_BASE_CALLS);
			lane->encodingInfo[encodingInfo.readType][encodingInfo.encoding] = encodingInfo;
		}

//		//Register the ColorCalls if present.
//		//TODO uncomment condition.  Remains commented until the IS_COLOR_PRESENT attribute is added to the spec, assume all XSQ files have colors
//		if (H5Aexists(group_id, _HATTRIBUTE_NAME_IS_COLOR_PRESENT.c_str()) && getIntAttribute(group_id, _HATTRIBUTE_NAME_IS_COLOR_PRESENT)) {
//			EncodingInfo encodingInfo;
//			encodingInfo.hDatasetName = _HDATASET_COLORCALLQV;
//			encodingInfo.readType = readType;
//			encodingInfo.encoding = SOLID_ENCODING;
//			encodingInfo.numCalls = getIntAttribute(group_id, _HATTRIBUTE_NAME_NUM_COLOR_CALLS);
//			lane->encodingInfo[encodingInfo.readType][encodingInfo.encoding] = encodingInfo;
//		}

		return 0;
	}

	//Register the ColorEncodings
	if (object_info->type == H5O_TYPE_DATASET &&
		hpath.depth() == 4 &&
		hpath.ancestor(3).name() == _HGROUP_NAME_RUN_METADATA &&
		hpath.ancestor(2).name() == _HGROUP_NAME_TAG_DETAILS &&
		hpath.name() == _HGROUP_NAME_COLOR_ENCODING ) {
		hsize_t nfields;
		hsize_t nrecords;
		getHdfTableInfo(o_id, name, &nfields, &nrecords);

		shared_array<ColorEncodingTableEntry> result(new ColorEncodingTableEntry[nrecords]);
		readHdfTable(o_id, name_, sizeof(ColorEncodingTableEntry), COLOR_ENCODING_TABLE_DST_OFFSETS, COLOR_ENCODING_TABLE_DST_SIZES, result.get() );
		XsqReadType readType = toReadType(hpath.parent().name());
		for (ColorEncodingTableEntry* ce = result.get(); ce < result.get()+nrecords; ++ce) {
			EncodingInfo encodingInfo;
			encodingInfo.hDatasetName = ce->datasetName;
			encodingInfo.readType = readType;
			encodingInfo.encoding = ColorEncoding(ce->encoding, ce->offset, ce->stride);
			encodingInfo.numCalls = ce->numColorCalls;
			lane->encodingInfo[readType][encodingInfo.encoding] = encodingInfo;
		}
	}

	//Register a Panel
	if (object_info->type          == H5O_TYPE_GROUP &&
		hpath.depth()              == 2 &&
		strspn(hpath.name().c_str(), "0123456789") == hpath.name().size() ) {
			shared_ptr<PanelContainerImpl> panelContainer = lane->resolvePanelContainer(hpath.parent().str());
			panelContainer->resolvePanel(name);
			return 0;
	}
	return 0;
}

void LaneImpl::open(weak_ptr<LaneImpl> wp_self) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering LaneImpl::open())" << endl;
# endif
	this->wp_self = wp_self;
	if (hdfHandle >= 0 ) return; //Already open.
	//TODO throw file_not_found_exception, file_not_readable_exception
	hdfHandle = openHdfFile(filename, this->m_readOnly);
	visitHdf(hdfHandle, &laneInfoOpenH5OvisitCallback, &wp_self);
# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting LaneImpl::open())" << endl;
# endif
}

shared_ptr<PanelContainerImpl> LaneImpl::resolvePanelContainer(string const& hdf5path) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering LaneImpl::resolvePanelContainer())" << endl;
# endif
	string panelContainerName = hdf5path.substr(hdf5path.rfind('/')+1);
	if (panelContainerIndex.find(panelContainerName) == panelContainerIndex.end())
		return createPanelContainer(hdf5path);
	return panelContainers[panelContainerIndex[panelContainerName]];
# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting LaneImpl::resolvePanelContainer())" << endl;
# endif
}

shared_ptr<PanelContainerImpl> LaneImpl::createPanelContainer(string const& hdf5path) {
	string panelContainerName = HDFNode(hdf5path).name();
	//TODO read all these variables out of the file metadata.
	hid_t pcGroupHid_t = openHdfGroup(hdfHandle, hdf5path);
	HIDCloser closer0(pcGroupHid_t); //Ensure close

	H5G_info_t info;
	getHdfGroupInfo(pcGroupHid_t, &info);

	if (info.nlinks < 1) {
		stringstream msg;
		msg << "Empty Panel Container: '" << hdf5path << "' in file: " << filename;
		throw XsqException(msg.str());
	}
	hid_t aPanelGroup = openAnyChildGroup(pcGroupHid_t);
	HIDCloser closer1(aPanelGroup);
	shared_ptr<PanelContainerImpl> sp_pcImpl(
		new PanelContainerImpl(
			wp_self.lock(),
			hdf5path,
			panelContainerName == _HGROUP_NAME_DEFAULT_LIBRARY ? 1 : getUInt32Attribute(pcGroupHid_t, _HATTRIBUTE_NAME_ID)
		)
	);

	sp_pcImpl->wp_self = sp_pcImpl;
	panelContainers[sp_pcImpl->index] = sp_pcImpl;
	panelContainerIndex[panelContainerName] = sp_pcImpl->index;
	return sp_pcImpl;
}

size_t LaneImpl::size() const {
	size_t size = 0;
	for (map<size_t, shared_ptr<PanelContainerImpl> >::const_iterator it = panelContainers.begin(); it != panelContainers.end(); ++it)
		size += it->second->size();
	return size;
}

string PanelContainerImpl::getName() const {
	return HDFNode(this->hdf5Path).name();
}

size_t PanelContainerImpl::size() const {
	size_t size = 0;
	for (map<size_t, shared_ptr<PanelImpl> >::const_iterator it = panels.begin(); it != panels.end(); ++it)
		size += it->second->size();
	return size;
}

shared_ptr<PanelImpl> PanelContainerImpl::resolvePanel(string const& hdfPath) {
	const size_t panelNumber = atoi(HDFNode(hdfPath).name().c_str());
	if (this->panels.find(panelNumber) == this->panels.end())
		return createPanel(hdfPath);
	return panels[panelNumber];
}

shared_ptr<PanelImpl> PanelContainerImpl::createPanel(string const& hdfPath) {
	const shared_ptr<PanelImpl> sp_Panel(new PanelImpl(wp_self.lock(), hdfPath));
	sp_Panel->wp_self = sp_Panel;
	this->panels.insert(make_pair(sp_Panel->getPanelNumber(), sp_Panel));
	return sp_Panel;
}

PanelImpl::PanelImpl(shared_ptr<PanelContainerImpl> const& _wp_panelContainer, string const& hdf5path)
	: wp_panelContainer(_wp_panelContainer),
	  hdf5Path(hdf5path),
	  panelNumber(atoi(HDFNode(hdf5path).name().c_str())),//Take the panel group name as the panel number
	  dataChanged(false),
	  filteringLoaded(false)
{
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering PanelImpl::PanelImpl(xxx," << hdf5path << ")" << endl;
# endif
	hid_t panelGroup_id = openHdfGroup(wp_panelContainer.lock()->getHDFFileHandle(), this->hdf5Path);
	HIDCloser closer0(panelGroup_id);

	this->numFragments = getUInt32Attribute(panelGroup_id, _HATTRIBUTE_NAME_FRAGMENT_COUNT);
# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting PanelImpl::PanelImpl()" << endl;
# endif
}

void PanelImpl::loadCallAndQV(XsqReadType const& readType, ColorEncoding const& encoding) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering ReadDatasetImpl::loadCallAndQV(): " << this->wp_panelContainer.lock()->getHDFFileHandle() << " " << this->hdf5Path.c_str() << "\n";
# endif

	//assertColorsAvailable(readType, encoding);
	const string readTypeGroupName(wp_panelContainer.lock()->wp_lane.lock()->getReadTypeGroupName(readType));
	const hid_t fileHid = this->wp_panelContainer.lock()->getHDFFileHandle();
	const hid_t thisPanelHid = openHdfGroup(fileHid, this->hdf5Path.c_str());
	const HIDCloser closer0(thisPanelHid);
	const bool tagExists = checkHdfLinkExists(thisPanelHid, readTypeGroupName);
	const size_t readLength = wp_panelContainer.lock()->getNumCalls(readType, encoding);
	/*
	 * Allocate array of pointers to rows.
	 */
	const shared_array<unsigned char*> callAndQvArr(new unsigned char*[this->numFragments]);
	/*
	 * Allocate space for integer data.
	 */
	const shared_array<unsigned char> callAndQvData(new unsigned char[this->numFragments * readLength]);
	callAndQvArr[0] = callAndQvData.get();

	/*
	 * Set the rest of the pointers to rows to the correct addresses.
	 */
	for (unsigned int i=1; i<this->numFragments; i++)
		callAndQvArr[i] = callAndQvArr[0] + i * readLength;

	if (tagExists) {
		stringstream path;
		path << this->hdf5Path << "/" << readTypeGroupName << "/" << wp_panelContainer.lock()->getDatasetName(readType, encoding);
		hid_t h5dCallAndQv = openHdfDataset(fileHid, path.str().c_str());
		HIDCloser closer1(h5dCallAndQv);
		readHdfDataset(h5dCallAndQv, H5T_NATIVE_UCHAR_g, callAndQvData.get());
	} else {
		memset(callAndQvData.get(), 255, this->numFragments * readLength);
	}
	CallAndQVInfo value;
	value.arr = callAndQvArr;
	value.data = callAndQvData;
	this->callAndQVs[readType][encoding] = value;
	# ifdef XSQ_READER_TRACE_ON
		cerr << "exiting ReadDatasetImpl::loadCallAndQV()" << endl;
	# endif
}

void PanelImpl::loadYX() {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering ReadDatasetImpl::loadXY() " << endl;
# endif
	/*
	 * Allocate space for data.
	 */

	uint16_t* yxData = (uint16_t*)malloc( (1 + this->numFragments) * 2 * sizeof(uint16_t));
	if(yxData == NULL)
	{
		stringstream msg;
		msg << "loadXY::yxData: Unable to allocate " << this->numFragments * 2 * sizeof(uint16_t) << " bytes. Out of Memory" << endl;
		throw XsqException(msg.str());
	}

	shared_array<uint16_t> sp_yxData(yxData);

	/*
	 * Make the HDF API calls.
	 */
	string path = HDFNode(HDFNode(this->hdf5Path), string(_HGROUP_NAME_FRAGMENTS).append("/").append(_HDATASET_NAME_YX)).str();
	hid_t h5dYX = openHdfDataset(this->wp_panelContainer.lock()->getHDFFileHandle(), path);
	HIDCloser closer0(h5dYX);

	readHdfDataset(h5dYX, H5T_NATIVE_UINT16_g, yxData); // FIXME - purify notes reading 536 bytes more than sizeof yxData

	this->yxData = sp_yxData;

# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting ReadDatasetImpl::loadXY()" << endl;
# endif
}

bool PanelImpl::loadFiltering() {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering ReadDatasetImpl::loadFiltering() " << endl;
# endif
	/*
	 * Make the HDF API calls.
	 */
	stringstream path;
	path << this->hdf5Path << "/" << _HGROUP_NAME_FRAGMENTS << "/" << _HDATASET_NAME_FILTERING;
	const hid_t& fileHid = this->wp_panelContainer.lock()->getHDFFileHandle();
	const bool hasFiltering = checkHdfLinkExists(fileHid, path.str());
	if (hasFiltering) {
		const hid_t hid_t = openHdfDataset(this->wp_panelContainer.lock()->getHDFFileHandle(), path.str());
		HIDCloser closer0(hid_t);
		uint8_t* data = new uint8_t[this->numFragments];
		readHdfDataset(hid_t, H5T_NATIVE_UCHAR_g, data);
		this->filtering.reset(data);
	}
	filteringLoaded = true;
# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting ReadDatasetImpl::loadFiltering() " << endl;
# endif
	return hasFiltering;
}

bool PanelImpl::loadFiltering(XsqReadType const& readType) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering ReadDatasetImpl::loadFiltering(" << readType << ") " << endl;
# endif
	const hid_t& fileHid = this->wp_panelContainer.lock()->getHDFFileHandle();
	const hid_t thisPanelHGroup = openHdfGroup(fileHid, hdf5Path);
	const HIDCloser closer0(thisPanelHGroup);
	const string& readTypeGroupName = this->wp_panelContainer.lock()->wp_lane.lock()->getReadTypeGroupName(readType);
	const bool tagExists = checkHdfLinkExists(thisPanelHGroup, readTypeGroupName);
	FilterTrimInfo& filterTrimInfo = this->filterTrimMap[readType];
	const string pathToFilteringDataset = readTypeGroupName + "/" + _HDATASET_NAME_FILTERING;
	const bool filteringDatasetExists = tagExists ? checkHdfLinkExists(thisPanelHGroup, pathToFilteringDataset) : 0;
	const bool readsAreFiltered = filteringDatasetExists || !tagExists;
	if (readsAreFiltered) {
		uint8_t* data = new uint8_t[this->numFragments];
		if (filteringDatasetExists) {
			const hid_t hid_t = openHdfDataset(thisPanelHGroup, pathToFilteringDataset);
			HIDCloser closer1(hid_t);
			readHdfDataset(hid_t, H5T_NATIVE_UCHAR_g, data);
		} else //!tagExists: all reads are filtered
			memset(data, 1, this->numFragments);
		filterTrimInfo.filtering.reset(data);
	}
	filterTrimInfo.filteringLoaded = true;

# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting ReadDatasetImpl::loadFiltering(" << readType << ") " << endl;
# endif
	return readsAreFiltered;
}

bool PanelImpl::loadTrimLength(XsqReadType const& readType, bool const& start) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering ReadDatasetImpl::loadTrimLength(" << readType << ") " << endl;
# endif
	const hid_t& fileHid = this->wp_panelContainer.lock()->getHDFFileHandle();
	const hid_t thisPanelHGroup = openHdfGroup(fileHid, hdf5Path);
	const HIDCloser closer0(thisPanelHGroup);
	const string& readTypeGroupName = this->wp_panelContainer.lock()->wp_lane.lock()->getReadTypeGroupName(readType);
	const bool tagExists = checkHdfLinkExists(thisPanelHGroup, readTypeGroupName);
	FilterTrimInfo& filterTrimInfo = this->filterTrimMap[readType];
	const string pathToTrimDataset = readTypeGroupName + "/" + (start ? _HDATASET_NAME_TRIM_START_LENGTH : _HDATASET_NAME_TRIM_END_LENGTH);
	const bool trimDatasetExists = tagExists ? checkHdfLinkExists(thisPanelHGroup, pathToTrimDataset) : 0;
	if (trimDatasetExists) {
		const hid_t hid_t = openHdfDataset(thisPanelHGroup, pathToTrimDataset);
		HIDCloser closer0(hid_t);
		uint16_t* data = new uint16_t[this->numFragments];
		readHdfDataset(hid_t, H5T_NATIVE_UINT16_g, data);
		(start ? filterTrimInfo.trimStartLength : filterTrimInfo.trimEndLength).reset(data);
	}
	(start ? filterTrimInfo.startTrimLoaded : filterTrimInfo.endTrimLoaded) = true ;
# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting ReadDatasetImpl::loadTrimLength(" << readType << ") " << endl;
# endif
	return trimDatasetExists;
}

unsigned char* PanelImpl::getCallAndQV(XsqReadType const& readType, ColorEncoding const& encoding, size_t const& rowNum ) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering PanelImpl::getCallAndQV(" << readType << ", XXXX," << rowNum << ")" << endl;
# endif
	//assertColorsAvailable(readType, encoding);
	if (callAndQVs.find(readType) == callAndQVs.end() ||
	    callAndQVs[readType].find(encoding) == callAndQVs[readType].end()) loadCallAndQV(readType, encoding);
# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting PanelImpl::getCallAndQV()" << endl;
# endif
	return callAndQVs.find(readType)->second.find(encoding)->second.arr[rowNum];
}

uint16_t* PanelImpl::getYX(size_t const& rowNum) {
# ifdef XSQ_READER_TRACE_ON
	cerr << "entering PanelImpl::getXY(" << rowNum << ")" << endl;
# endif
	if (yxData == NULL) loadYX();
# ifdef XSQ_READER_TRACE_ON
	cerr << "exiting PanelImpl::getXY()" << endl;
# endif
	return &(yxData[rowNum*2]);
}

bool PanelImpl::isFiltered(size_t const& rowNum) {
    if (!filteringLoaded) loadFiltering();
	if (filtering == NULL) return false;
	return filtering[rowNum] != 0;
}

bool PanelImpl::isReadFiltered(XsqReadType const& readType, size_t const& rowNum) {
	if (isFiltered(rowNum)) return true;
	const FilterTrimInfo& filterTrimInfo = filterTrimMap[readType];
	if (!filterTrimInfo.filteringLoaded) loadFiltering(readType);
	if (filterTrimInfo.filtering == NULL) return false;
	return filterTrimInfo.filtering[rowNum] != 0;
}

size_t PanelImpl::getTrim(XsqReadType const& readType, bool const& start, size_t const& rowNum) {
	if (isReadFiltered(readType, rowNum) ) return start ? 0 : wp_panelContainer.lock()->getNumCalls(readType, BASE_ENCODING);
	const FilterTrimInfo& filterTrimInfo = filterTrimMap[readType];
	const bool trimLoaded = start ? filterTrimInfo.startTrimLoaded : filterTrimInfo.endTrimLoaded;
	if (!trimLoaded) loadTrimLength(readType, start);
	const shared_array<uint16_t>& sp = start ? filterTrimInfo.trimStartLength : filterTrimInfo.trimEndLength;
	if (sp == NULL) return 0;
	return sp[rowNum];
}

void PanelImpl::writeData() {
	const vector<XsqReadType>& readTypes = this->getLane().getReadTypes();
	for (vector<XsqReadType>::const_iterator outer = readTypes.begin(); outer != readTypes.end(); ++outer) {
		const XsqReadType& readType = *outer;
		const vector<EncodingInfo>& encodingInfoList = this->getLane().getEncodingInfo(readType);
		for (vector<EncodingInfo>::const_iterator encodingInfo = encodingInfoList.begin(); encodingInfo != encodingInfoList.end(); ++encodingInfo) {
			const ColorEncoding& encoding = encodingInfo->encoding;
			if ( this->callAndQVs.find(readType) == this->callAndQVs.end() ||
				 this->callAndQVs[readType].find(encoding) == callAndQVs[readType].end()) continue;
			const CallAndQVInfo& callAndQVInfo = callAndQVs[readType][encoding];
			const string pathToReadType = this->getHDF5Path() + "/" +
					this->wp_panelContainer.lock()->wp_lane.lock()->getReadTypeGroupName(readType);
			const string pathToDataset = pathToReadType + "/" + encodingInfo->hDatasetName;
			if (checkHdfLinkExists(this->getLane().getHDFHandle(), pathToReadType) &&
				checkHdfLinkExists(this->getLane().getHDFHandle(), pathToDataset))
				overwriteHdfDataset(this->getLane().getHDFHandle(), pathToDataset, callAndQVInfo.data.get());
		}
	}
}

const bool PanelImpl::isReadTypeDataVirtual(XsqReadType const& readType) const {
	const boost::shared_ptr<LaneImpl> lane = wp_panelContainer.lock()->wp_lane.lock();
	return !checkHdfLinkExists(lane->getHDFHandle(), (this->getHDF5Path()+"/"+lane->getReadTypeGroupName(readType)));
}

map<size_t, PanelContainer> Lane::getPanelContainers() const {
	map<size_t, PanelContainer> panelContainers;
	for (map<size_t, shared_ptr<PanelContainerImpl> >::const_iterator it = this->mp_impl->panelContainers.begin(); it != this->mp_impl->panelContainers.end(); ++it)
		panelContainers.insert(pair<size_t, PanelContainer>(it->first, it->second));
	return panelContainers;
}

vector<Panel> PanelContainer::getPanels() const {
	vector<Panel> readDatasets;
	for(map<size_t, shared_ptr<PanelImpl> >::const_iterator it = this->mp_impl->panels_begin(); it != this->mp_impl->panels_end(); ++it)
		readDatasets.push_back(Panel(it->second));
	return readDatasets;
}

Panel::panel_fragments_const_iterator Panel::begin(bool const& maskFilteredAndTrimmedBases) const { return panel_fragments_const_iterator(*this, maskFilteredAndTrimmedBases); }

Panel::panel_fragments_const_iterator Panel::begin() const { return begin(false); }

Panel::panel_fragments_const_iterator Panel::end() const { return end_of_panel;}

  XsqReader::XsqReader(vector<Panel> const& panels, bool const& readOnly, bool const& skipFilteredFragments, bool const& maskFilteredAndTrimmedBases) : m_state(INIT), m_readOnly(readOnly), m_skipFilteredFragments(skipFilteredFragments), m_maskFilteredAndTrimmedBases(maskFilteredAndTrimmedBases) {
	const map<size_t, vector<PanelRangeSpecifier> > index = convert(panels);
	for (map<size_t, vector<PanelRangeSpecifier> >::const_iterator i=index.begin(); i != index.end(); ++i) {
		for (vector<PanelRangeSpecifier>::const_iterator j=i->second.begin(); j != i->second.end(); ++j) {
			this->open(*j, i->first);
		}
	}
}

void XsqReader::close() {
	m_state = CLOSED;
	for (vector<Panel>::iterator it=m_Panels.begin(); it != m_Panels.end(); ++it)
		it->release();
}

bool XsqReader::open(PanelRangeSpecifier const& specifier, size_t const& filenumber) {
	assert(m_state == INIT);
	if (find(m_PanelRangeSpecifiers.begin(), m_PanelRangeSpecifiers.end(), specifier) != m_PanelRangeSpecifiers.end()) return false;
	const string filename = specifier.getPath();
	map<size_t, Lane>::const_iterator it = m_lanes.find(filenumber);
	if (it == m_lanes.end()) {
		m_lanes[filenumber] = Lane(filename, filenumber, this->m_readOnly);
	} else if ( it->second.getFilename() != filename) {
		stringstream msg;
		msg << filenumber << " " << "is already associated with " << it->second.getFilename();
		throw XsqException(msg.str());
	}
	m_Panels_correct = false;
	m_PanelRangeSpecifiers.push_back(specifier);
	return true;
}

size_t XsqReader::size() {
	assert(m_state != CLOSED);
	this->updatePanels();
	size_t count = 0;
	for (vector<Panel>::const_iterator it=m_Panels.begin(); it != m_Panels.end(); ++it)
		count += it->size();
	return count;
}

//returns the sum of elements in arr from begin to end_incl (inclusive)
uint32_t total(vector<uint32_t>::const_iterator const first, vector<uint32_t>::const_iterator const& last) {
	assert(first <= last);
	uint32_t sum = 0;
	for (vector<uint32_t>::const_iterator it = first; it != last; ++it)
		sum += *it;
	return sum;
}

//returns the start indices of a partitioning of source into numPartition sets. numPartitions will be reduced to keep subsets larger than minimum_size.
vector<size_t> partition(vector<uint32_t> const& source, size_t numPartitions, uint32_t const& minimum_size) {
//		cerr << source.size() << " " << numPartitions << " " << minimum_size << endl;
//		cerr << "[";
//		for (vector<uint32_t>::const_iterator it = source.begin(); it != source.end(); ++it)
//			cerr << *it << (it + 1 == source.end() ? "" : ",");
//		cerr << "]" << endl;
    	const size_t NUM_ELEMENTS = source.size();
    	const uint32_t sum = total(source.begin(), source.end());
    	uint32_t expected_size = sum/numPartitions;
    	vector<size_t> partition_list;
    	while (numPartitions > 1 && expected_size < minimum_size)
    		expected_size = sum/--numPartitions;
    	if (numPartitions < 2) {
    		partition_list.push_back(0);
    		return partition_list;
    	}
    	//Use Dynamic Programming to calculate an optimal partition_size
    	size_t* A = new size_t[NUM_ELEMENTS];
    	size_t* B = new size_t[NUM_ELEMENTS];

    	A[0] = 0;B[0]=0;
    	for (size_t i=1; i<NUM_ELEMENTS; ++i) {
    		A[i] = NPOS;
    		B[i] = 0;
    		for (size_t j=0; j<i; ++j) {
    			uint32_t weight = total(source.begin()+j+1, source.begin()+i+1);
    			uint32_t diff = weight > expected_size ? weight - expected_size : expected_size - weight;
    			if (A[j] + diff < A[i])
    			{
    				A[i] = A[j] + diff;
    				B[i] = j;
    			}

    		}
    	}


    	for (size_t j=NUM_ELEMENTS-1; j>0;)
    	{
    		partition_list.push_back(B[j]);
    		j=B[j];
    	}
    	reverse(partition_list.begin(), partition_list.end());
    	delete[] A;
    	delete[] B;
//    	cerr << partition_list.size() << endl;
//    	cerr << "[";
//    	for (vector<size_t>::const_iterator it = partition_list.begin(); it != partition_list.end(); ++it)
//    		cerr << *it << (it + 1 == partition_list.end() ? "" : ",");
//    	cerr << "]" << endl;
    	return partition_list;
    }
    
/*!
 * \param source vector of element sizes, the first entry of this vector must be zero and does not represent and element.
 * \param numPartitions number of partitions to produce
 * \param minimum_size minimum size of a partition
 * \return the start indices of a partitioning of source into numPartition sets. numPartitions will be reduced to keep subsets larger than minimum_size.
 */
vector<size_t> partition2(vector<uint32_t> const& source, size_t numPartitions, uint32_t const& minimum_size) {
	//cerr << source << endl;
	const size_t NUM_ELEMENTS = source.size();
	const uint32_t sum = total(source.begin(), source.end());
	uint32_t expected_size = sum/numPartitions;
	vector<size_t> partition_list;
	while (numPartitions > 1 && expected_size < minimum_size)
		expected_size = sum/--numPartitions;
	if (numPartitions < 2) {
		partition_list.push_back(0);
		return partition_list;
	}
	//Loop, reducing expected_size, until we get numPartitions.
	size_t loopsRemaining = 2; //Limit the number attempts to partition
	while (partition_list.size() < numPartitions && loopsRemaining-- > 0) {
		cerr << expected_size << endl;
		partition_list.clear();
		//Use Dynamic Programming to calculate an optimal partition_size
		size_t* A = new size_t[NUM_ELEMENTS];
		size_t* B = new size_t[NUM_ELEMENTS];

		A[0] = 0;B[0]=0;
		for (size_t i=1; i<NUM_ELEMENTS; ++i) {
			A[i] = NPOS;
			B[i] = 0;
			for (size_t j=0; j<i; ++j) {
				uint32_t weight = total(source.begin()+j+1, source.begin()+i);
				uint32_t diff = weight > expected_size ? weight - expected_size : expected_size - weight;
				if (A[j] + diff < A[i])
				{
					A[i] = A[j] + diff;
					B[i] = j;
				}

			}
		}
		delete[] A;

		for (size_t j=NUM_ELEMENTS-1; j>0;)
		{
			partition_list.push_back(B[j]);
			j=B[j];
		}
		delete[] B;
		//Adjust expected_size for next loop.
		expected_size -= (uint32_t)(expected_size * ( ( (float)numPartitions - partition_list.size() ) / numPartitions ));
		reverse(partition_list.begin(), partition_list.end());
		//cerr << partition_list.size() <<  " " << partition_list << endl;
		//TODO guarantee that partition_list isn't bigger than numPartitions.
	}

	//If there are still too few partitions
	//split the largest non-singleton partition until partition_list.size() == numPartitions
	while (partition_list.size() < numPartitions) {
		//find the largest non-singleton partition with size > minimum_size
		size_t indexOfLargest = partition_list.size();
		size_t sizeOfLargest = 0;
		for (size_t i=0; i<partition_list.size(); ++i) {
			const vector<uint32_t>::const_iterator begin = source.begin() + partition_list[i] + 1; //Add 1 for source's meaningless leading zero element.
			const vector<uint32_t>::const_iterator end = i == partition_list.size() - 1 ? source.end() : source.begin() + partition_list[i+1] + 1;
			const uint32_t size = total(begin, end);
			if (size > sizeOfLargest && // is it largest found so far?
				size/2 >= minimum_size &&  //is it sufficiently large
				end - begin > 1) //is it a non-singleton
			{
				indexOfLargest = i;
				sizeOfLargest = size;
			}
		}
		if (sizeOfLargest == 0) break; //No candidate for split found.
		const size_t indexStartOfLargest = partition_list[indexOfLargest];
		const size_t indexOnePastEndOfLargest = indexOfLargest == partition_list.size() - 1 ? NUM_ELEMENTS : partition_list[indexOfLargest + 1];
		const size_t middle = (indexStartOfLargest + indexOnePastEndOfLargest) / 2;
		const vector<size_t>::iterator insertPosition = partition_list.begin() + indexOfLargest + 1;
		partition_list.insert(insertPosition, middle);
	}
	//cerr << partition_list.size() <<  " " << partition_list << endl;
	return partition_list;
}

vector<XsqReader> XsqReader::divideEvenly(size_t const& numReaders_, uint32_t const& minNumFragmentsPerReader) {
	//assert(m_state == INIT);
	assert(numReaders_ > 0);
	size_t numReaders = numReaders_;
	while( this->size() / numReaders < minNumFragmentsPerReader && numReaders > 1 ) numReaders--;

	if (numReaders < 2) {
		vector<XsqReader> vec;
		vec.push_back(*this);
		return vec;
	}

	vector<uint32_t> sizes;
	sizes.push_back(0);
	for (vector<Panel>::const_iterator it=m_Panels.begin(); it != m_Panels.end(); ++it)
		sizes.push_back(it->size());
	vector<size_t> partitionStarts = partition(sizes, numReaders, minNumFragmentsPerReader);

	//Handle case where partition returns too many partitions.
	while (partitionStarts.size() > numReaders) partitionStarts.pop_back();

	vector<vector<Panel> > partitions;
	for (size_t i=0; i<partitionStarts.size(); ++i) {
		vector<Panel> partition;
		copy(m_Panels.begin() + partitionStarts[i], i+1 < partitionStarts.size() ? m_Panels.begin() + partitionStarts[i+1] : m_Panels.end(), back_inserter(partition));
		partitions.push_back(partition);
	}

	vector<XsqReader> readers;
	for (vector<vector<Panel> >::const_iterator partition=partitions.begin(); partition != partitions.end(); ++partition)
	  readers.push_back(XsqReader(*partition, this->m_readOnly, this->m_skipFilteredFragments, this->m_maskFilteredAndTrimmedBases));

	// Diagnostics - output resulting split
	/*if(g_log->isDebugEnabled()) {
	  LOG4CXX_INFO(g_log, "divideEvenly() returning " << readers.size() << " partitions.");
	  for (size_t i=0; i<readers.size(); ++i) {
		LOG4CXX_INFO(g_log, "Reader #" << i);
		vector<string> urls = readers[i].getURLs();
		for (vector<string>::const_iterator url = urls.begin(); url != urls.end(); ++url)
			LOG4CXX_INFO(g_log, "  " << *url);
	  }
	}*/

	return readers;
}

vector<string> XsqReader::getURLs() const {
	assert(m_state != CLOSED);
	vector<string> vec;
	for (vector<PanelRangeSpecifier>::const_iterator it = this->m_PanelRangeSpecifiers.begin(); it != this->m_PanelRangeSpecifiers.end(); ++it)
		vec.push_back(it->str());
	return vec;
}

XsqReader::fragment_const_iterator XsqReader::begin() {
	assert(m_state != CLOSED);
	m_state = ITERATING;
	if (!m_Panels_correct) this->updatePanels();
	if (this->m_Panels.empty()) return end();
	return fragment_const_iterator(&this->m_Panels, m_skipFilteredFragments, m_maskFilteredAndTrimmedBases);
}

XsqReader::fragment_const_iterator XsqReader::end() {
	assert(m_state != CLOSED);
	if (!m_Panels_correct) this->updatePanels();
	return end_of_fragment;
}

XsqReader::panel_iterator XsqReader::panels_begin() {
	assert(m_state != CLOSED);
	m_state = ITERATING;
	if (!m_Panels_correct) this->updatePanels();
	if (this->m_Panels.empty()) return panels_end();
	return m_Panels.begin();
}

XsqReader::panel_iterator XsqReader::panels_end() {
	assert(m_state != CLOSED);
	if (!m_Panels_correct) updatePanels();
	return m_Panels.end();
}

void XsqReader::updatePanels() {
	assert(m_state != CLOSED);
	vector<Panel> tmp_readDatasets;
	for (map<size_t, Lane>::const_iterator lanePair=m_lanes.begin(); lanePair != m_lanes.end(); ++lanePair) {
		map<size_t, PanelContainer> panelContainers = lanePair->second.getPanelContainers();
		for (map<size_t, PanelContainer>::const_iterator pcPair=panelContainers.begin(); pcPair != panelContainers.end(); ++pcPair) {
			vector<Panel> panels = pcPair->second.getPanels();
			for (vector<Panel>::const_iterator panel=panels.begin(); panel != panels.end(); ++panel) {
				for (vector<PanelRangeSpecifier>::const_iterator specifier=m_PanelRangeSpecifiers.begin(); specifier != m_PanelRangeSpecifiers.end(); ++ specifier) {
					if (!specifier->accept(*panel)) continue;
					tmp_readDatasets.push_back(*panel);
					break;
				}
			}
		}
	}
	m_Panels = tmp_readDatasets;
	m_Panels_correct = true;
}

void XsqWriter::PanelBuffer::reset(size_t const& size, int barcode, int panel) {
	 this->bufferSize = size;
	 this->actualSize = 0;
	 this->barcode = barcode;
	 this->panel = panel;
	 this->callAndQVData.clear();
	 this->isReadTypeVirtual.clear();

	 uint16_t* yxData = (uint16_t*)malloc(bufferSize * 2 * sizeof(uint16_t));
	 if(yxData == NULL)
	 {
		stringstream msg;
		msg << "init::yxData: Unable to allocate " << bufferSize * 2 * sizeof(uint16_t) << " bytes. Out of Memory" << endl;
		throw XsqException(msg.str());
	 }

	 this->yxData = boost::shared_array<uint16_t>(yxData);
}

void XsqWriter::PanelBuffer::prepareForReads(XsqReadType const& readType, ColorEncoding const& colorEncoding, size_t const& readLength) {
	unsigned char* data = new unsigned char[this->bufferSize * readLength];
	unsigned char** arr = new unsigned char*[this->bufferSize];

	arr[0] = data;
	 for (size_t i=1; i<bufferSize; ++i)
		 arr[i] = arr[0] + i * readLength;
	this->callAndQVData[readType][colorEncoding].readLength = readLength;
	this->callAndQVData[readType][colorEncoding].data = shared_array<unsigned char>(data);
	this->callAndQVData[readType][colorEncoding].arr = shared_array<unsigned char*>(arr);
}

XsqWriter::PanelBuffer & XsqWriter::getThreadPanelBuffer() {
	boost::mutex::scoped_lock lock(io_mutex);
	return this->panelBuffers[pthread_self()];
}

XsqWriter& XsqWriter::operator<<(FragmentI const& fragment) {
	if (this->fileNumber == NPOS ) this->init(fragment.getLane());
	assert(this->fileNumber == fragment.getFileNumber());
	PanelBuffer & buffer = this->getThreadPanelBuffer();
	if (buffer.barcode == -1 || buffer.panel != fragment.getPanelNumber() || static_cast<unsigned int>(buffer.barcode) != fragment.getPanelContainerNumber()) {
		int barcode = fragment.getPanelContainerNumber();
		int panel = fragment.getPanelNumber();
		if ( buffer.barcode != -1) this->write(buffer); 
		if (this->panelContainerGroups.find(barcode) == this->panelContainerGroups.end()) {
			boost::mutex::scoped_lock lock(io_mutex);
			// Try again, now that we've obtained the lock
			if (this->panelContainerGroups.find(barcode) == this->panelContainerGroups.end()) {
				hid_t pcGroupHid = createHdfGroup(this->hdfHandle, fragment.getPanelContainerName());
				setHdfStringAttribute(pcGroupHid, _HATTRIBUTE_LIBRARY_NAME, "UNKNOWN");
				setHdfStringAttribute(pcGroupHid, _HATTRIBUTE_UUID, "UNKNOWN");
				setHdfStringAttribute(pcGroupHid, _HATTRIBUTE_INDEX_NAME, "UNKNOWN");
				setHdfIntAttribute(pcGroupHid, _HATTRIBUTE_NAME_ID, barcode);
				this->panelContainerGroups[barcode] = pcGroupHid;
			} // else created before we obtained the lock
		}
		buffer.reset(fragment.getPanel().size(), barcode, panel); 
		for (map<XsqReadType, map<ColorEncoding, EncodingInfo> >::const_iterator pair = this->encodings.begin(); pair != encodings.end(); ++pair) {
			const XsqReadType& readType = pair->first;
			const map<ColorEncoding, EncodingInfo>& infos = pair->second;
			for (map<ColorEncoding, EncodingInfo>::const_iterator info = infos.begin(); info != infos.end(); ++info)
				buffer.prepareForReads(readType, info->first, fragment.getNumCalls(readType,info->first)); // check this function
		}
	}
	
	buffer.yxData[2*buffer.actualSize] = fragment.getY();
	buffer.yxData[2*buffer.actualSize+1] = fragment.getX(); 

	for (map<XsqReadType, map<ColorEncoding, EncodingInfo> >::const_iterator pair = this->encodings.begin(); pair != encodings.end(); ++pair) {
		const XsqReadType& readType = pair->first;
		if (buffer.isReadTypeVirtual.find(readType) == buffer.isReadTypeVirtual.end())
			buffer.isReadTypeVirtual[readType] = fragment.getPanel().isReadTypeDataVirtual(readType);
		const map<ColorEncoding, EncodingInfo>& encodings = pair->second;
		for (map<ColorEncoding, EncodingInfo>::const_iterator info = encodings.begin(); info != encodings.end(); ++info) {
			const ColorEncoding& encoding = info->first;
			unsigned char* callQvs = fragment.getCallQVs(readType, encoding);
			memcpy(buffer.callAndQVData[readType][encoding].arr[buffer.actualSize],callQvs,buffer.callAndQVData[readType][encoding].readLength);
		}
	}
	++buffer.actualSize;
	return *this;
}

void XsqWriter::init(LaneI& lane) {
	boost::mutex::scoped_lock lock(io_mutex);
	if (this->fileNumber != NPOS) return;
	hid_t runMetatdataGroupHid = createHdfGroup(this->hdfHandle, _HGROUP_NAME_RUN_METADATA);
	HIDCloser closer0(runMetatdataGroupHid);

	hid_t tagDetailsGroupHid = createHdfGroup(runMetatdataGroupHid, _HGROUP_NAME_TAG_DETAILS);
	HIDCloser closer1(tagDetailsGroupHid);

	for (vector<XsqReadType>::const_iterator readType = lane.getReadTypes().begin(); readType != lane.getReadTypes().end(); ++readType) {
		hid_t tagGroupHid = createHdfGroup(tagDetailsGroupHid, to_string(*readType));
		HIDCloser closer2(tagGroupHid);
		bool isBasePresent = lane.isColorsAvailable(*readType, BASE_ENCODING);
		if (isBasePresent) {
			uint32_t numCalls = isBasePresent ? lane.getNumCalls(*readType, BASE_ENCODING) : 0;
			setHdfIntAttribute(tagGroupHid, _HATTRIBUTE_NAME_NUM_BASE_CALLS, numCalls);
		}
		setHdfStringAttribute(tagGroupHid, _HATTRIBUTE_NAME_TAG_SEQUENCE, lane.getTagSequence(*readType));
		setHdfIntAttribute(tagGroupHid, _HATTRIBUTE_NAME_IS_BASE_PRESENT, isBasePresent);

		const vector<EncodingInfo>& encodingInfos = lane.getEncodingInfo(*readType);
		shared_array<ColorEncodingTableEntry> data(new ColorEncodingTableEntry[encodingInfos.size()]);
		for (size_t i=0; i<encodingInfos.size(); ++i) {
			const ColorEncoding& encoding = encodingInfos[i].encoding;
			const EncodingInfo& info = encodingInfos[i];
			this->encodings[*readType][encoding] = info;
			strcpy(data[i].datasetName, encodingInfos[i].hDatasetName.c_str());
			strcpy(data[i].encoding, encodingInfos[i].encoding.getProbeset().c_str());
			data[i].numColorCalls = encodingInfos[i].numCalls;
			data[i].offset = encodingInfos[i].encoding.getOffset();
			data[i].stride = encodingInfos[i].encoding.getStride();
		}
		createHdfTable(_HGROUP_NAME_COLOR_ENCODING, tagGroupHid, COLOR_ENCODING_TABLE_FIELD_NAMES, NFIELDS_COLOR_ENCODING_TABLE, encodingInfos.size(), sizeof(ColorEncodingTableEntry), COLOR_ENCODING_TABLE_DST_OFFSETS, COLOR_ENCODING_TABLE_FIELD_TYPES, data.get());
	}
	//copy metadata

	this->fileNumber = lane.getFileNumber(); // Set Last, as this is used as a semaphore in other functions
}

void XsqWriter::write(PanelBuffer const& buffer) {
	boost::format panelGroupNameFormat("%+04d");
	panelGroupNameFormat % buffer.panel;
	const std::string panelGroupName = panelGroupNameFormat.str();
	//LOG4CXX_INFO(g_log, "Thread [" << pthread_self() << "]: creating group " << buffer.barcode << " / " << panelGroupNameFormat.str());
	hid_t panelGroupHid = createHdfGroup(this->panelContainerGroups[buffer.barcode], panelGroupName.c_str());
	HIDCloser closer0(panelGroupHid);
	setHdfIntAttribute(panelGroupHid, _HATTRIBUTE_NAME_FRAGMENT_COUNT, buffer.actualSize);

	hid_t fragmentsGroupHid = createHdfGroup(panelGroupHid, "Fragments");
	HIDCloser closer1(fragmentsGroupHid);

	shared_array<hsize_t> dims(new hsize_t[2]);
	dims[0] = buffer.actualSize;
	dims[1] = 2;
	hid_t dataspaceHid = createHdfDataspace(2, dims.get());
	HIDCloser closer2(dataspaceHid);

	hid_t datasetHid = createHdfDataset(fragmentsGroupHid, _HDATASET_NAME_YX, H5T_STD_U16LE_g, dataspaceHid);
	HIDCloser closer3(datasetHid);

	writeHdfDataset(datasetHid, H5T_STD_U16LE_g, dataspaceHid, buffer.yxData.get());

	dims[1] = 1;
	dataspaceHid = createHdfDataspace(2, dims.get());
	HIDCloser closer4(dataspaceHid);

	datasetHid = createHdfDataset(fragmentsGroupHid, _HDATASET_NAME_FILTERING, H5T_STD_U8LE_g, dataspaceHid);
	HIDCloser closer5(datasetHid);

	typedef std::map<ColorEncoding, CallAndQvData > innerMap_t;
	typedef std::map<XsqReadType, innerMap_t > outerMap_t;
	for (outerMap_t::const_iterator outer = buffer.callAndQVData.begin();
		 outer != buffer.callAndQVData.end(); ++outer) {
		const XsqReadType& readType = outer->first;
		const map<XsqReadType, bool>::const_iterator it = buffer.isReadTypeVirtual.find(readType);
		if (it != buffer.isReadTypeVirtual.end() && it->second == true)
			continue;
		const hid_t tagGroupHid = createHdfGroup(panelGroupHid, to_string(readType).c_str());
		HIDCloser closer6(tagGroupHid);

		for(innerMap_t::const_iterator inner = outer->second.begin();
			inner != outer->second.end(); ++inner) {
			const ColorEncoding& encoding = inner->first;
			dims[1] = inner->second.readLength;
			dataspaceHid = createHdfDataspace(2, dims.get());
			HIDCloser closer7(dataspaceHid);
//			if (this->encodings.find(readType) == this->encodings.end())
//				throw "read type not found";
//			if (this->encodings[readType].find(encoding) == encodings[readType].end())
//				throw "encoding not found";
			const string& hDatasetName = this->encodings[readType][encoding].hDatasetName;
//			cerr << "'" << hDatasetName << "' " << encodings[readType][encoding].encoding << endl ;
			datasetHid = createHdfDataset(
					tagGroupHid,
					hDatasetName,
					H5T_STD_U8LE_g,
					dataspaceHid
			);
			HIDCloser closer8(datasetHid);

			writeHdfDataset(datasetHid, H5T_STD_U8LE_g, dataspaceHid, inner->second.data.get());
		}
	}
	//TODO write filtering data.
}

XsqWriter& XsqMultiWriter::createXsqFile(size_t const& fileNumber, string const& fileName) {
	boost::mutex::scoped_lock lock(io_mutex); //synchronize file creation.
	if (writers.find(fileNumber) == writers.end()) {
		boost::filesystem::path path = m_outputDirectory / fileName;
		writers.insert(std::make_pair(fileNumber, XsqWriter(path.string())));
	}
	return writers.find(fileNumber)->second;
}

vector<int8_t> compare(vector<size_t>::const_iterator const& first_begin,  vector<size_t>::const_iterator const& first_end,
		                vector<size_t>::const_iterator const& second_begin, vector<size_t>::const_iterator const& second_end) {
	vector<int8_t> vec;
	int8_t curr = 0;
	for (vector<size_t>::const_iterator first=first_begin, second=second_begin;
		 first != first_end && second != second_end;
		 ++first, ++second)
		if (curr == 0 && *first != *second) curr = *first < *second ? -1 : 1;
		vec.push_back(curr);
	return vec;
}

//End XSQ API Class implementations

} // end namespace lifetechnologies

