/* -*- Mode: C++; indent-tabs-mode: 't; c-basic-offset: 8; tab-width: 8  -*-   vi:set noexpandtab ts=8 sw=8:
 * Copyright (C) 2010 Life Technologies Corporation. All rights reserved.
 */

/*
 * xsq_io.hpp
 *
 *  Created on: Sep 21, 2010
 *      Author: mullermw
 */

#ifndef XSQ_READER_HPP_
#define XSQ_READER_HPP_

//#define XSQ_READER_TRACE_ON

#include <boost/iterator/filter_iterator.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <hdf5.h>
#include "types.hpp"

// Malloc
#include <cstdlib>

/*!
	lifetechnologies namespace
*/
namespace lifetechnologies {

extern const size_t NPOS;

/* Forward declarations */
class LaneImpl;
class PanelContainerImpl;
class PanelImpl;
class FragmentImpl;
class Lane;
class PanelContainer;
class Panel;
class Fragment;
template <class Value> class PanelFragmentsIterator;
template <class Value> class FragmentIterator;

herr_t laneInfoOpenH5OvisitCallback(  hid_t o_id, const char *name_, const H5O_info_t *object_info, void *op_data);
uint32_t total(std::vector<uint32_t>::const_iterator const first, std::vector<uint32_t>::const_iterator const& last);
std::vector<size_t> partition(std::vector<uint32_t> const& source, size_t numPartitions, uint32_t const& minimum_size);
std::vector<size_t> partition2(std::vector<uint32_t> const& source, size_t numPartitions, uint32_t const& minimum_size);
/* End Forward Declarations */

struct EncodingInfo {
	std::string hDatasetName;
	XsqReadType readType;
	ColorEncoding encoding;
	size_t numCalls;
};

//Interfaces

//Handle Types
/*!
 * Info about a Lane of data.
 */
 
class LaneI {
public :
	virtual ~LaneI() {;}
	virtual std::string getFilename() const = 0;
	virtual size_t getFileNumber() const = 0;
	virtual std::string getInstrumentSerialNumber() const = 0;
	virtual Instrument  getInstrumentType() const = 0;
	virtual std::string getHDF5Version() const = 0;
	virtual std::string getFileVersion() const = 0;
	virtual std::string getSoftware() const = 0;
	virtual std::string getRunName() const = 0;
	virtual std::string getSampleName() const = 0;
	virtual uint8_t    getLaneNumber() const = 0;
	virtual LibraryType getLibraryType() const = 0; /*!< Frag, Mate-pair etc. */
	virtual Application getApplicationType() const = 0;
	virtual time_t getFileCreationTime() const = 0;
	virtual hid_t getHDFHandle() const = 0;
	//virtual size_t getNumPanelContainers() const = 0;
	//virtual size_t getNumPanels() const = 0;
	virtual size_t size() const = 0;
	virtual const std::vector<XsqReadType>& getReadTypes() = 0;
	virtual bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual const std::vector<ColorEncoding>& getColorEncodings(XsqReadType const& readType) = 0;
	virtual const std::string& getTagSequence(XsqReadType const& readType)  const = 0;
	virtual size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual const std::vector<EncodingInfo>& getEncodingInfo(XsqReadType const& readType) = 0;
	//TODO virtual std::map<size_t, PanelContainer> getPanelContainers() const = 0;
	//TODO virtual std::string getSeqStr(XsqReadType, ColorEncoding) const = 0;
};

/*!
 * A PanelContainerInfo (aka Read Set) is a set of reads that share the same Lane (Xsq file), Barcode Assignment
 * and XsqReadType(F3,R3 etc.)  This is a different set of reads than those contained in the
 * Barcode Group in the XSQ file...those contain reads from multiple library types.
 *
 * Note, it is not possible to navigate from ReadSetInfo to individual reads.
 */
class PanelContainerI {
public:
	virtual ~PanelContainerI() {;}
	virtual LibraryType getLibraryType() const = 0; /*!< Type of the associated library. */
	virtual std::string getHDF5Path() const = 0; /*!< The Path to the Group containing ReadSet */
	virtual std::string getName() const = 0; /*!< returns name of associated barcode if hasBarcode() == true, otherwise returns empty string. */
	virtual std::string getPrimerBases(XsqReadType const& type) const = 0; /*!< return the primer base adjacent to the insert, in the 5-3' direction. */
	virtual std::string getFilename() const = 0;/*!< The Name of the file containing this readset */
	virtual size_t size() const = 0;
	//virtual std::vector<PanelI> getPanels() const = 0;
};

/*!
 * Represents an HDF Panel Group,, containg Fragments.
 */
class PanelI {
public :
	virtual ~PanelI() {;}
	virtual size_t getPanelNumber() const = 0;
	virtual size_t size() const = 0;
	virtual size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual std::string getPanelContainerName() const = 0;
	virtual size_t getPanelContainerIndex() const = 0;
	virtual std::string getFilename() const = 0;
	virtual const bool isReadTypeDataVirtual(XsqReadType const& readType) const = 0;
};

/*!
 * Represents a single read from a Life Technologies sequencing instrument.
 * A read consists of 1 or more sequences of color calls, each associated with an encoding.
 * When reads have been decoded to bases, the color calls may be discarded.  Attempts to
 * access missing color sequences will throw an exception so
 * To be safe, use
 * isBaseCalled() and isColorCallAvailable(n) before calling getBases() and getColors(n)
 * respectively.
 */
 
static const char * const bspace = "ACGTN"; 
 
class FragmentI {
public :
	virtual ~FragmentI() {;}
	/*!
	 * The common name of the read in traditional csfasta form:
	 *
	 * PANEL_X_Y_[FR][35]
	 *
	 * ex. 1972_6_4_F3  = Panel 1972, X=6, Y=4
	 *
	 */
	virtual std::string getName(XsqReadType const& readType) const = 0;
	virtual std::string getName() const = 0;
	virtual const char * getNameChar() const = 0;
	/*!
	 *  get the X coordinate of this read.
	 */
	virtual size_t getRowNum() const = 0;
	virtual size_t getX()  const = 0;
	/*!
	 * get the Y coordinate of this read.
	 */
	virtual size_t getY()  const = 0;

	virtual bool isBasesAvailable(XsqReadType const& readType) const = 0;
	/*!
	 * return the base calls if isBaseCalled() == true
	 * otherwise throw an exception.
	 */
	virtual std::string getBases(XsqReadType const& readType) const = 0;

	virtual QualityValueArray getBaseQVs(XsqReadType const& readType) const=0;

	virtual bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	/*!
	 * return the color calls for the encoding if isColorCallAvialable(encodingNumber) == true
	 * otherwise throw an exception.
	 */
	virtual std::string getColors(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual char * getColorRead(XsqReadType const& readType) const = 0;
	virtual char * getColorReadNoPrimer(XsqReadType const& readType) const = 0;

	/*!
	 * return the bases quality values if isBaseCalled() == true
	 * otherwise throw an exception.
	 */
	virtual QualityValueArray getQualityValues(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;

	virtual unsigned char* getCallQVs(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual void getCalls(uint8_t** arr, uint8_t** qvs, size_t &num_calls, XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual void getCalls(uint8_t * colors, uint8_t * qvs, XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual size_t getPanelNumber() const = 0; /*!< convenience method */
	virtual LibraryType getLibraryType() const = 0;
	virtual std::string getFilename() const = 0;
	virtual size_t getFileNumber() const = 0;
	virtual std::string getPanelContainerName() const = 0;
	virtual size_t getPanelContainerNumber() const = 0;
	virtual std::string getPrimerBases(XsqReadType const& readType) const = 0;
	virtual size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const = 0;
	virtual PanelContainerI& getPanelContainer() const = 0;
	virtual PanelI& getPanel() const = 0;
	virtual LaneI& getLane() const = 0;
	virtual const std::vector<XsqReadType>& getReadTypes() const = 0;
	//virtual const vector<ColorEncoding>& getEncodings(XsqReadType const& readType) const = 0;


};

//Impl types
class LaneImpl : public LaneI {
	friend herr_t lifetechnologies::laneInfoOpenH5OvisitCallback(hid_t, const char*, const H5O_info_t*, void*);
	friend class Lane;
	std::map<size_t, boost::shared_ptr<PanelContainerImpl> > panelContainers;
	boost::weak_ptr<LaneImpl> wp_self;
	std::map<std::string, size_t> panelContainerIndex;
	std::string filename;
	size_t fileIndex;
	std::string instrumentSerialNumber;
	Instrument  instrumentType;
	std::string hdf5Version;
	std::string fileVersion;
	std::string software;
	std::string runName;
	std::string sampleName;
	uint8_t     laneNumber;
	LibraryType libraryType;
	Application application;
	time_t      fileCreationTime;
	hid_t       hdfHandle;
	//TODO organize these into more consise data structure.
	std::vector<XsqReadType> readTypes;
	std::map<XsqReadType, std::string > tagSequences;
	std::map<XsqReadType, std::string > readTypeGroupNames;
	std::map<XsqReadType, std::map<ColorEncoding, EncodingInfo> > encodingInfo;
	std::map<XsqReadType, std::vector<ColorEncoding> > colorEncodings;
	std::map<XsqReadType, std::vector<EncodingInfo> > encodingInfoVec;
	const bool m_readOnly;
	size_t maxNumCalls;

public :
	LaneImpl(std::string const& _filename, size_t const& _fileIndex, bool const& _readOnly) : filename(_filename), fileIndex(_fileIndex), hdfHandle(-1), m_readOnly(_readOnly), maxNumCalls(0) {}
	virtual ~LaneImpl() {
# ifdef XSQ_READER_TRACE_ON
		std::cerr << "~LaneImpl()" << std::endl;
# endif
	}
	std::string getFilename() const { return this->filename; }
	size_t getFileNumber() const { return this->fileIndex; }
	std::string getInstrumentSerialNumber() const { return this->instrumentSerialNumber; }
	Instrument  getInstrumentType() const { return this->instrumentType; }
	std::string getHDF5Version() const { return this->hdf5Version; }
	std::string getFileVersion() const { return this->fileVersion; }
	std::string getSoftware() const { return this->software; }
	std::string getRunName() const { return this->runName; }
	std::string getSampleName() const { return this->sampleName; }
	uint8_t    getLaneNumber() const { return this->laneNumber; }
	LibraryType getLibraryType() const { return this->libraryType; }
	Application getApplicationType() const { return this->application; }
	time_t getFileCreationTime() const { return this->fileCreationTime; }
	hid_t getHDFHandle() const { return this->hdfHandle; }
	size_t size() const; //Return the number of reads contained.
	const bool& isReadOnly() { return this->m_readOnly; }
	bool hasReadType(XsqReadType const& readType) const {
		return tagSequences.find(readType) != tagSequences.end();
	}
	void assertHasReadType(XsqReadType const& readType) const {
		if (!hasReadType(readType)) {
			std::stringstream msg;
			msg << this->getFilename() << " does not have Read Type: " << to_string(readType);
			throw XsqException(msg.str());
		};
	}
	bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const {
		assertHasReadType(readType);
		std::map<ColorEncoding, EncodingInfo> m = this->encodingInfo.find(readType)->second;
		return m.find(encoding) != m.end();
		return false;
	}

	void assertColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const {
		if (!isColorsAvailable(readType, encoding)) {
			std::stringstream msg;
			if (encoding == BASE_ENCODING)
				msg << "Base Encoding";
			else
				msg << "Color Encoding " << encoding;
			msg << " does not exist for tag " << readType << " in " << this->filename << ".  Available encodings are: ";
			for (std::map<XsqReadType, std::map<ColorEncoding, EncodingInfo> >::const_iterator it=this->encodingInfo.begin(); it != encodingInfo.end(); ++it)
				for (std::map<ColorEncoding, EncodingInfo>::const_iterator jt=it->second.begin(); jt != it->second.end(); ++jt)
					msg << "{" << it->first << "," << jt->first << "},";
			msg << ".";
			throw XsqException(msg.str());
		}
	}

	const std::string& getTagSequence(XsqReadType const& readType)  const {
		assertHasReadType(readType);
		return this->tagSequences.find(readType)->second;
	}

	const std::string& getReadTypeGroupName(XsqReadType const& readType) const {
		assertHasReadType(readType);
		return this->readTypeGroupNames.find(readType)->second;
	}

	const std::vector<ColorEncoding>& getColorEncodings(XsqReadType const& readType) {
		assertHasReadType(readType);
		if (this->colorEncodings.find(readType) == this->colorEncodings.end()) {
			std::vector<ColorEncoding> vec;
			for (std::map<ColorEncoding, EncodingInfo>::const_iterator it = this->encodingInfo[readType].begin(); it != this->encodingInfo[readType].end(); ++it)
				vec.push_back(it->first);
			this->colorEncodings[readType] = vec;
		}
		return this->colorEncodings[readType];
	}

	const std::vector<EncodingInfo>& getEncodingInfo(XsqReadType const& readType) {
		assertHasReadType(readType);
		if (this->encodingInfoVec.find(readType) == this->encodingInfoVec.end()) {
			std::vector<EncodingInfo> vec;
			for (std::map<ColorEncoding, EncodingInfo>::const_iterator it = this->encodingInfo[readType].begin(); it != this->encodingInfo[readType].end(); ++it)
				vec.push_back(it->second);
			this->encodingInfoVec[readType] = vec;
		}
		return this->encodingInfoVec[readType];
	}

	std::string getDatasetName(XsqReadType const& readType, ColorEncoding const& encoding) const {
		assertColorsAvailable(readType, encoding);
		return this->encodingInfo.find(readType)->second.find(encoding)->second.hDatasetName;
	}

	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const {
		return this->encodingInfo.find(readType)->second.find(encoding)->second.numCalls;
	}

	size_t getMaxNumCalls() {
		if (maxNumCalls != 0) return maxNumCalls;
		size_t temp = 0;
		const std::vector<XsqReadType> readTypes = this->getReadTypes();
		for (std::vector<XsqReadType>::const_iterator readType=readTypes.begin(); readType != readTypes.end(); ++readType) {
			const std::vector<ColorEncoding> colorEncodings = this->getColorEncodings(*readType);
			for (std::vector<ColorEncoding>::const_iterator colorEncoding=colorEncodings.begin(); colorEncoding != colorEncodings.end(); ++colorEncoding)
				temp = std::max(temp, this->getNumCalls(*readType, *colorEncoding));
		}
		return ( maxNumCalls = temp );
	}

	const std::vector<XsqReadType>& getReadTypes() {
		if (this->readTypes.size() != this->encodingInfo.size()) {
			this->readTypes.clear();
			for (std::map<XsqReadType, std::map<ColorEncoding, EncodingInfo> >::const_iterator it = this->encodingInfo.begin();
				 it != this->encodingInfo.end();
				 ++it) {
				this->readTypes.push_back(it->first);
			}
		}
		return this->readTypes;
	}

private : /* Functions */
	void open(boost::weak_ptr<LaneImpl> wp_self);
	boost::shared_ptr<PanelContainerImpl> resolvePanelContainer(std::string const& hdfpath);
	boost::shared_ptr<PanelContainerImpl> createPanelContainer(std::string const& hdfpath);
};

class PanelContainerImpl : public PanelContainerI {
	friend herr_t lifetechnologies::laneInfoOpenH5OvisitCallback(hid_t, const char*, const H5O_info_t*, void*);
	friend class LaneImpl;
	friend class PanelImpl;
	const boost::weak_ptr<LaneImpl>	 wp_lane;
	boost::weak_ptr<PanelContainerImpl> wp_self;
	std::map<size_t, boost::shared_ptr<PanelImpl> > panels;
	std::string                    hdf5Path;
	size_t                         index;
	size_t                         numFragments;

public:
	PanelContainerImpl(boost::shared_ptr<LaneImpl> const& _lane,
					   std::string const& _hdf5path, size_t const& _index) :
			   wp_lane(_lane), hdf5Path(_hdf5path), index(_index) {}
	virtual ~PanelContainerImpl() {
# ifdef XSQ_READER_TRACE_ON
		std::cerr << "~PanelContainerImpl()" << std::endl;
# endif
	}
	std::string getName() const;
	size_t getIndex() const { return index; }
	std::string getHDF5Path() const { return this->hdf5Path; }

	LibraryType getLibraryType() const { return this->wp_lane.lock()->getLibraryType(); }
	std::string getFilename() const { return this->wp_lane.lock()->getFilename(); }
	size_t getFileIndex() const { return this->wp_lane.lock()->getFileNumber(); }
	hid_t getHDFFileHandle() const { return this->wp_lane.lock()->getHDFHandle(); }
	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const { return this->wp_lane.lock()->getNumCalls(readType, encoding); }
	std::string getPrimerBases(XsqReadType const& readType) const { return this->wp_lane.lock()->getTagSequence(readType); }
	std::vector<ColorEncoding> getColorEncodings(XsqReadType const& readType) const { return this->wp_lane.lock()->getColorEncodings(readType); }
	bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) { return this->wp_lane.lock()->isColorsAvailable(readType, encoding); }
	void assertColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const { return this->wp_lane.lock()->assertColorsAvailable(readType, encoding); }
	std::string getDatasetName(XsqReadType const& readType, ColorEncoding const& encoding) const { return this->wp_lane.lock()->getDatasetName(readType, encoding); }
	size_t size() const; //Return the number of reads contained.
	size_t getMaxNumCalls() const { return this->wp_lane.lock()->getMaxNumCalls(); }

	LaneI& getLane() const { return *wp_lane.lock(); }
	const std::vector<XsqReadType>& getReadTypes() const { return wp_lane.lock()->getReadTypes(); }

	//Returns the number of increments or decrements needed to move from panels[p1] to panels[p2]
	int32_t panelDifference(size_t const p1, size_t const p2) const {
		std::map<size_t, boost::shared_ptr<PanelImpl> >::const_iterator iof_p1 = panels.find(p1);
		std::map<size_t, boost::shared_ptr<PanelImpl> >::const_iterator iof_p2 = panels.find(p2);
		if (iof_p1 == panels.end() || iof_p2 == panels.end()) return NPOS;
		return p1 < p2 ? distance(iof_p1, iof_p2) : -distance(iof_p2, iof_p1);
	}

	typedef std::map<size_t, boost::shared_ptr<PanelImpl> >::const_iterator panel_const_iterator;
	panel_const_iterator panels_begin() const { return panels.begin(); }
	panel_const_iterator panels_end() const { return panels.end(); }

private:
	boost::shared_ptr<PanelImpl> resolvePanel(std::string const& hdfPath);
	boost::shared_ptr<PanelImpl> createPanel(std::string const& hdfPath);
};

class PanelImpl : public PanelI {
	friend class PanelContainerImpl;
	friend class Panel;
	const boost::weak_ptr<PanelContainerImpl> wp_panelContainer;
	boost::weak_ptr<PanelImpl> wp_self;
	std::string       hdf5Path;
	size_t            panelNumber;
	size_t            numFragments;
	bool dataChanged;
	bool filteringLoaded;

	struct CallAndQVInfo {
		boost::shared_array<unsigned char*> arr;
		boost::shared_array<unsigned char> data;
	};
	typedef std::map<XsqReadType, std::map<ColorEncoding, CallAndQVInfo> > CallQvMap;
	CallQvMap callAndQVs;

	struct FilterTrimInfo {
		bool filteringLoaded;
		bool startTrimLoaded;
		bool endTrimLoaded;
		boost::shared_array<uint8_t> filtering;
		boost::shared_array<uint16_t> trimStartLength;
		boost::shared_array<uint16_t> trimEndLength;
		FilterTrimInfo() : filteringLoaded(false), startTrimLoaded(false), endTrimLoaded(false) {}
	};
	std::map<XsqReadType, FilterTrimInfo> filterTrimMap;

	boost::shared_array<uint16_t*> yxArr;
	boost::shared_array<uint16_t> yxData;
	boost::shared_array<uint8_t> filtering;

public :
	PanelImpl(boost::shared_ptr<PanelContainerImpl> const& backref, std::string const& _hdf5Path);
	virtual ~PanelImpl() {
# ifdef XSQ_READER_TRACE_ON
		std::cerr << "~PanelImpl()" << std::endl;
# endif
	}
	size_t getPanelNumber()    const { return this->panelNumber; }
	bool isPrecededBy(PanelImpl const& other) const { return this->wp_panelContainer.lock()->panelDifference(this->panelNumber, other.panelNumber) == -1; }
	size_t size()                const { return numFragments; }
	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const {
		return wp_panelContainer.lock()->getNumCalls(readType, encoding);
	}
	std::string getFilename()    const { return this->wp_panelContainer.lock()->getFilename(); }
	size_t getFileIndex() const { return this->wp_panelContainer.lock()->getFileIndex(); }
	LibraryType getLibraryType() const { return this->wp_panelContainer.lock()->getLibraryType(); }
	std::string getPanelContainerName()   const { return this->wp_panelContainer.lock()->getName(); }
	size_t getPanelContainerIndex() const { return this->wp_panelContainer.lock()->getIndex();}
	std::string getPrimerBases(XsqReadType const& readType) const { return this->wp_panelContainer.lock()->getPrimerBases(readType); }
	std::vector<ColorEncoding> getColorEncodings(XsqReadType const& readType) const {	return this->wp_panelContainer.lock()->getColorEncodings(readType); }
	bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const {
		return this->wp_panelContainer.lock()->isColorsAvailable(readType, encoding);
	}
	void assertColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const {
		this->wp_panelContainer.lock()->assertColorsAvailable(readType, encoding);
	}
	unsigned char* getCallAndQV(XsqReadType const& readType, ColorEncoding const& encoding, size_t const& rowNum);
	uint16_t* getYX(size_t const& rowNum);
	size_t getY(size_t const& rowNum) { return this->getYX(rowNum)[0]; }
	size_t getX(size_t const& rowNum) { return this->getYX(rowNum)[1]; }
	bool isFiltered(size_t const& rowNum);
	bool isReadFiltered(XsqReadType const& readType, size_t const& rowNum);
	size_t getTrim(XsqReadType const& readType, bool const& start, size_t const& rowNum);
	LaneI& getLane() const { return this->wp_panelContainer.lock()->getLane(); }
	PanelContainerI& getPanelContainer() const { return *this->wp_panelContainer.lock(); }
	const std::vector<XsqReadType>& getReadTypes() const { return this->wp_panelContainer.lock()->getReadTypes(); }
	const std::string& getHDF5Path() const { return hdf5Path; }
	void setDataChanged() {
		this->dataChanged = true;
	}
	size_t getMaxNumCalls() const { return wp_panelContainer.lock()->getMaxNumCalls(); }
	const bool isReadTypeDataVirtual(XsqReadType const& readType) const;
	
private :
	void loadCallAndQV(XsqReadType const& readType, ColorEncoding const& colorEncoding);
	void loadYX();
	bool loadFiltering();
	bool loadFiltering(XsqReadType const& readType);
	bool loadTrimLength(XsqReadType const& readType, bool const& start);
	void release() {
		if (!this->wp_panelContainer.lock()->wp_lane.lock()->isReadOnly())
			this->writeData();
		callAndQVs.clear();
		filterTrimMap.clear();
		yxArr = boost::shared_array<uint16_t*>(NULL);
		yxData = boost::shared_array<uint16_t>(NULL);
		filtering = boost::shared_array<uint8_t>(NULL);
		filteringLoaded = false;
	}
	void writeData();
};

class FragmentImpl : public FragmentI {
	friend class Fragment;
	boost::weak_ptr<PanelImpl> wp_Panel;
	size_t rowNum;
	const bool m_maskFilteredAndTrimmedBases;
	unsigned char * calls_buffer;
	unsigned char * qvs_buffer;
	char * tag_name_buffer;

public :
	FragmentImpl(boost::shared_ptr<PanelImpl> parent, size_t const& rowNum, bool const& maskFilteredAndTrimmedBases, size_t const& maxNumCalls) : wp_Panel(parent), rowNum(rowNum), m_maskFilteredAndTrimmedBases(maskFilteredAndTrimmedBases) {init(maxNumCalls);	}
	virtual ~FragmentImpl(){dist();}
	bool operator==(FragmentImpl const& other) const {
		return this->wp_Panel.lock() == other.wp_Panel.lock() && this->rowNum == other.rowNum;
	}

	LaneI& getLane() const { return this->wp_Panel.lock()->getLane(); }
	PanelContainerI& getPanelContainer() const { return this->wp_Panel.lock()->getPanelContainer(); }
	PanelI& getPanel() const { return *this->wp_Panel.lock(); }

	const std::vector<XsqReadType>& getReadTypes() const { return this->wp_Panel.lock()->getReadTypes(); }

	size_t getRowNum() const { return rowNum; }
	size_t getFileIndex() const { return this->wp_Panel.lock()->getFileIndex(); }
	size_t getPanelContainerNumber() const { return this->wp_Panel.lock()->getPanelContainerIndex(); }
	size_t getPanelNumber() const {	return this->wp_Panel.lock()->getPanelNumber(); }
	size_t getX() const { return this->wp_Panel.lock()->getX(this->rowNum); }
	size_t getY() const { return this->wp_Panel.lock()->getY(this->rowNum); }
	bool isFiltered() const { return this->wp_Panel.lock()->isFiltered(this->rowNum); }
	bool isReadFiltered(XsqReadType const& readType ) { return this->wp_Panel.lock()->isReadFiltered(readType, this->rowNum); }
	size_t getTrim(XsqReadType const& readType, bool const& start) const { return this->wp_Panel.lock()->getTrim(readType, start, this->rowNum); }
	/*!
	 * The common name of the read in traditional csfasta form:
	 *
	 * PANEL_X_Y_[FR][35]
	 *
	 * ex. 1972_6_4_F3  = Panel 1972, X=6, Y=4
	 *
	 */
	std::string getName(XsqReadType const& readType) const {
		if (wp_Panel.lock() == NULL) return "NULL";
		std::stringstream s;
		s << this->getFileIndex();
		s << '_';
		s << this->getPanelContainerNumber();
		s << '_';
		s << this->getPanelNumber();
		s << '_';
		s << this->getY();
		s << '_';
		s << this->getX();
		s << '_';
		s << readType;
		return s.str();
	}

	std::string getName() const {
		if (wp_Panel.lock() == NULL) return "NULL";
		std::stringstream s;
		s << this->getFileIndex();
		s << '_';
		s << this->getPanelContainerNumber();
		s << '_';
		s << this->getPanelNumber();
		s << '_';
		s << this->getY();
		s << '_';
		s << this->getX();
		return s.str();
	}
	
	const char * getNameChar() const {
		if (wp_Panel.lock() == NULL) return "NULL";
		sprintf(tag_name_buffer,"%u_%u_%u_%u_%u", (unsigned int)this->getFileIndex(), (unsigned int)this->getPanelContainerNumber(), (unsigned int)this->getPanelNumber(), (unsigned int)this->getY(), (unsigned int)this->getX());
		return tag_name_buffer;
	}

	LibraryType getLibraryType()   const { return this->wp_Panel.lock()->getLibraryType(); }
	std::string getFilename()      const { return this->wp_Panel.lock()->getFilename(); }
	size_t getFileNumber() const { return this->wp_Panel.lock()->getFileIndex(); }
	std::string getPanelContainerName()     const { return this->wp_Panel.lock()->getPanelContainerName(); }
	std::string getPrimerBases(XsqReadType const& readType)   const { return this->wp_Panel.lock()->getPrimerBases(readType); }
	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const {
		return this->wp_Panel.lock()->getNumCalls(readType, encoding);
	}

	bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const {
		return wp_Panel.lock()->isColorsAvailable(readType, encoding);
	}
	bool isBasesAvailable(XsqReadType const& readType) const { return this->isColorsAvailable(readType, BASE_ENCODING); }
	void assertCallAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const {
		wp_Panel.lock()->assertColorsAvailable(readType, encoding);
	}


	//Call and QV getters


	unsigned char* getCallQVs(XsqReadType const& readType, ColorEncoding const& encoding) const {
		return wp_Panel.lock()->getCallAndQV(readType, encoding, this->rowNum);
	}

 // This code returns calls as char (not int '1'!=1) and qv's as ints
 // memory managed inside
 //-------------------------------------------------------------------------------------
	void getCalls(uint8_t** colors, uint8_t** qvs, size_t &num_calls, XsqReadType const& readType, ColorEncoding const& encoding) const {
		num_calls = this->getNumCalls(readType, encoding);
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, encoding, this->rowNum);
		uint8_t qual; size_t i=0; 
		if(encoding==SOLID_ENCODING) {
			calls_buffer[0]=(this->wp_Panel.lock()->getPrimerBases(readType))[0];
			for (; i<num_calls; ++i) {
				qual = ( callAndQv[i] >> 2 );
				//special case: set colors[i]=5 and qvs[i] 0 when qual > 62
				if (calls_buffer != NULL ) calls_buffer[i+1] = qual > 62 ? 0x2E : ((callAndQv[i] & 0x03) | 0x30);
				if (qvs_buffer != NULL) qvs_buffer[i] = qual > 62 ? 0 : qual;
			}
		} else {
			const bool applyMask = encoding == BASE_ENCODING && m_maskFilteredAndTrimmedBases;
			const size_t firstNonTrimmedPosition = applyMask ? getTrim(readType, true) : 0;
			const size_t onePastLastNonTrimmedPosition = num_calls - (applyMask ? getTrim(readType, false) : 0 );
			bool isMasked;
			for (; i<num_calls; ++i) {
				qual = ( callAndQv[i] >> 2 );
				isMasked = ( qual > 62 ) || ( i < firstNonTrimmedPosition ) || ( i >= onePastLastNonTrimmedPosition );
				//special case: set colors[i]=5 and qvs[i] 0 when qual > 62
				if (calls_buffer != NULL ) calls_buffer[i] = isMasked ? (encoding==BASE_ENCODING?0x4E:0x2E) : ( encoding==BASE_ENCODING ? bspace[(callAndQv[i] & 0x03)] : ((callAndQv[i] & 0x03) | 0x30));
				if (qvs_buffer != NULL) qvs_buffer[i] = isMasked ? 0 : qual;
			}
		}
		*colors = (uint8_t *) calls_buffer;
		*qvs = (uint8_t *) qvs_buffer;
	}
	
	// This code returns calls as integers (not ASCII chars 1!='1')
 // memory managed outside
 //-------------------------------------------------------------------------------------
	void getCalls(uint8_t* colors, uint8_t* qvs, XsqReadType const& readType, ColorEncoding const& encoding) const {
		const size_t &num_calls = this->getNumCalls(readType, encoding);
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, encoding, this->rowNum);
		const bool applyMask = encoding == BASE_ENCODING && m_maskFilteredAndTrimmedBases;
		const size_t firstNonTrimmedPosition = applyMask ? getTrim(readType, true) : 0;
		const size_t onePastLastNonTrimmedPosition = num_calls - (applyMask ? getTrim(readType, false) : 0 );
		bool isMasked;
		uint8_t qual;
		for (size_t i=0; i<num_calls; ++i) {
			qual = ( callAndQv[i] >> 2 );
			isMasked = ( qual > 62 ) || ( i < firstNonTrimmedPosition ) || ( i >= onePastLastNonTrimmedPosition );
			//special case: set colors[i]=5 and qvs[i] 0 when qual > 62
			if (colors != NULL ) colors[i] = isMasked ? 5 : callAndQv[i] & 0x03;
			if (qvs != NULL) qvs[i] = isMasked ? 0 : qual;
		}
		return;
	}
		

	void setCalls (uint8_t* colors, uint8_t* qvs, XsqReadType const& readType, ColorEncoding const& encoding) const {
		assert(colors != NULL);
		assert(qvs != NULL);
		const size_t& num_calls = this->getNumCalls(readType, encoding);
		PanelImpl& panel = *wp_Panel.lock();
		panel.setDataChanged();
		unsigned char* callAndQv = panel.getCallAndQV(readType, encoding, this->rowNum);
		for (size_t i=0; i<num_calls; ++i)
			//special case: use call=0 Qv=63 when color[i] > 3 or qv[i] < 2 .  set Qv=62 when qvs[i] > 62.
			callAndQv[i]   = ( colors[i] > 3 || qvs[i] < 2 ? 63 : qvs[i] > 62 ? 62 : qvs[i] ) << 2 | (colors[i] > 3 ? 0 : colors[i]);
	}

	// Note this code returns a pointer to SOLiD 2BE Sequence that includes primer base
	//-----------------------------------------------------------------------------------
	char * getColorRead(XsqReadType const& readType) const {
		const size_t num_calls = this->getNumCalls(readType, SOLID_ENCODING);
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, SOLID_ENCODING, this->rowNum);
		calls_buffer[0]=(this->wp_Panel.lock()->getPrimerBases(readType))[0];
		size_t i=0;
		//0xFC == 63, missing quality value token.
		//0x2E == 'N'
		//| 0x30  converts [0,1,2,3....] to ['1','2','3'...]
		for ( ; i<num_calls; ++i) calls_buffer[i+1] = ( callAndQv[i] & 0xFC ) == 0xFC ? 0x2E : (callAndQv[i] & 0x03) | 0x30 ;
		calls_buffer[i+1] = '\0';
		return (char *)calls_buffer;
	}

	// Note this code returns a pointer to SOLiD 2BE Sequence in int no primer base
	//-----------------------------------------------------------------------------------
 	char * getColorReadNoPrimer(XsqReadType const& readType) const {
		const size_t num_calls = this->getNumCalls(readType, SOLID_ENCODING);
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, SOLID_ENCODING, this->rowNum);
		size_t i=0;
		for ( ; i<num_calls; ++i) calls_buffer[i] = ( callAndQv[i] & 0xFC ) == 0xFC ? 5 : (callAndQv[i] & 0x03);
		return (char *)calls_buffer;
	}

	// returns a pointer to Base Space sequence.
	char * getBaseRead(XsqReadType const& readType) const {
		const size_t num_calls = this->getNumCalls(readType, BASE_ENCODING);
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, BASE_ENCODING, this->rowNum);
		const size_t firstNonTrimmedPosition = m_maskFilteredAndTrimmedBases ? getTrim(readType, true) : 0;
		const size_t onePastLastNonTrimmedPosition = num_calls - (m_maskFilteredAndTrimmedBases ? getTrim(readType, false) : 0 );
		size_t i=0;
		bool isMasked;
		for (; i<num_calls; ++i) {
			isMasked = (( callAndQv[i] & 0xFC ) == 0xFC) || ( i < firstNonTrimmedPosition ) || ( i >= onePastLastNonTrimmedPosition );
			calls_buffer[i] = isMasked ? 'N' : NUCLEOTIDE_CODE[(callAndQv[i] & 0x03)];
		}
        calls_buffer[i] = '\0';
        return (char *)calls_buffer;
	}

	/*!
	 * return the color calls for the encoding if isColorCallAvialable(encodingNumber) == true
	 * otherwise throw an exception.
	 *
	 * INEFFICIENT!!!
	 */
	std::string getColors(XsqReadType const& readType, ColorEncoding const& encoding) const {
# ifdef XSQ_READER_TRACE_ON
		std::cerr << "entering FragmentImpl::getColors(), this->rowNum=" << this->rowNum << std::endl;
# endif
		//assertCallAvailable(readType, encoding);
		std::stringstream colors;
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, encoding, this->rowNum);
		const size_t num_calls = this->getNumCalls(readType, encoding);
		const bool applyMask = encoding == BASE_ENCODING && m_maskFilteredAndTrimmedBases;
		const size_t firstNonTrimmedPosition = applyMask ? getTrim(readType, true) : 0;
		const size_t onePastLastNonTrimmedPosition = num_calls - (applyMask ? getTrim(readType, false) : 0 );
		bool isMasked;
		for (size_t i=0; i<num_calls; ++i) {
			isMasked = ( callAndQv[i] >> 2 > 62 ) || ( i < firstNonTrimmedPosition ) || ( i >= onePastLastNonTrimmedPosition );
			colors << ( isMasked ? '.' : boost::lexical_cast<char>( callAndQv[i] & 0x03 ) );
		}
		return colors.str();
	}

	/*!
	 *
	 * INEFFICIENT!!!
	 */
	QualityValueArray getQualityValues(XsqReadType const& readType, ColorEncoding const& encoding) const {
# ifdef XSQ_READER_TRACE_ON
		std::cerr << "entering FragmentImpl::getQualityValues(), this->rowNum=" << this->rowNum << std::endl;
# endif
		//assertCallAvailable(readType, encoding);
		QualityValueArray arr;
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, encoding, this->rowNum);
		const size_t &num_calls = this->getNumCalls(readType, encoding);
		uint8_t qual;
		for (size_t i=0; i<num_calls; ++i) {
			qual = callAndQv[i] >> 2;
			arr.push_back( qual > 62 ? 0 : qual);
		}
		return arr;
	}

	/*!
	 * return the base calls if available
	 * otherwise throw an exception.
	 *
	 * INEFFICIENT!!!
	 */
	std::string getBases(XsqReadType const& readType) const {
		std::stringstream bases;
		unsigned char* callAndQv = wp_Panel.lock()->getCallAndQV(readType, BASE_ENCODING, this->rowNum);
		const size_t &num_calls = this->getNumCalls(readType, BASE_ENCODING);
		const size_t firstNonTrimmedPosition = m_maskFilteredAndTrimmedBases ? getTrim(readType, true) : 0;
		const size_t onePastLastNonTrimmedPosition = num_calls - (m_maskFilteredAndTrimmedBases ? getTrim(readType, false) : 0 );
		bool isMasked;
		for (size_t i=0; i<num_calls; ++i) {
			isMasked = ( callAndQv[i] >> 2 > 62 ) || ( i < firstNonTrimmedPosition ) || ( i >= onePastLastNonTrimmedPosition );
			bases << (isMasked ? 'N' : boost::lexical_cast<char>( NUCLEOTIDE_CODE[callAndQv[i] & 0x03] ) );
		}
		return bases.str();
	}

	/*!
	 * return the base qvs if available
	 * otherwise throw an exception.
	 *
	 * INEFFICIENT!!!
	 */
	QualityValueArray getBaseQVs(XsqReadType const& readType) const {
		return getQualityValues(readType, BASE_ENCODING);
	}

private :
	FragmentImpl* incrementRowNum() {
		if (++rowNum >= this->wp_Panel.lock()->size()) rowNum = NPOS;
		return this;
	}

	void init(size_t const& maxNumCalls){
		calls_buffer = (unsigned char *) malloc(maxNumCalls+6);//plus 1 for line term, plus 5 for primer bases.		
		if(calls_buffer == NULL)
		{
			std::stringstream msg;
			msg << "calls_buffer: Unable to allocate " << maxNumCalls+6 << " bytes. Out of Memory" << std::endl;
			throw XsqException(msg.str());
		}
		qvs_buffer =  (unsigned char *) malloc(maxNumCalls+1);
		if(qvs_buffer == NULL)
		{
			std::stringstream msg;
			msg << "qvs_buffer: Unable to allocate " << maxNumCalls+1 << " bytes. Out of Memory" << std::endl;
			throw XsqException(msg.str());
		}
		tag_name_buffer =  (char *) malloc(1000);
		if(tag_name_buffer == NULL)
		{
			std::stringstream msg;
			msg << "tag_name_buffer: Unable to allocate " << 1000 << " bytes. Out of Memory" << std::endl;
			throw XsqException(msg.str());
		}
	}

	void dist(){
		if(calls_buffer!=NULL) {free(calls_buffer);calls_buffer=NULL;}
		if(qvs_buffer!=NULL) {free(qvs_buffer);qvs_buffer=NULL;}
		if(tag_name_buffer!=NULL) {free(tag_name_buffer);tag_name_buffer=NULL;}
	}

};

class Lane : LaneI {
	boost::shared_ptr<LaneImpl> mp_impl;
public :
	Lane() : mp_impl() {}
	Lane(std::string const& filename, size_t const& fileIndex, bool const& readOnly) : mp_impl(new LaneImpl(filename, fileIndex, readOnly)) {
		mp_impl->open(mp_impl);
	}
	Lane(std::string const& filename, size_t const& fileIndex) : mp_impl(new LaneImpl(filename, fileIndex, false)) {
		mp_impl->open(mp_impl);
	}
	virtual ~Lane() {;}
	std::string getFilename()               const { return mp_impl->getFilename(); }
	size_t getFileNumber() const { return mp_impl->getFileNumber(); }
	std::string getInstrumentSerialNumber() const { return mp_impl->getInstrumentSerialNumber(); }
	Instrument  getInstrumentType()         const { return mp_impl->getInstrumentType(); }
	std::string getHDF5Version()            const { return mp_impl->getHDF5Version(); }
	std::string getFileVersion()            const { return mp_impl->getFileVersion(); }
	std::string getSoftware()               const { return mp_impl->getSoftware(); }
	std::string getRunName()                const { return mp_impl->getRunName(); }
	std::string getSampleName()             const { return mp_impl->getSampleName(); }
	uint8_t    getLaneNumber()              const { return mp_impl->getLaneNumber(); }
	LibraryType getLibraryType()            const { return mp_impl->getLibraryType(); } /*!< Frag, Mate-pair etc. */
	Application getApplicationType()        const { return mp_impl->getApplicationType(); }
	time_t getFileCreationTime()            const { return mp_impl->getFileCreationTime(); }
	hid_t getHDFHandle()                    const { return mp_impl->getHDFHandle(); }
	size_t size()                           const { return mp_impl->size(); }
	const std::vector<XsqReadType>& getReadTypes() {return mp_impl->getReadTypes(); }
	std::map<size_t, PanelContainer> getPanelContainers() const;
	const std::vector<ColorEncoding>& getColorEncodings(XsqReadType const& readType) { return mp_impl->getColorEncodings(readType); }
	const std::string& getTagSequence(XsqReadType const& readType)  const { return mp_impl->getTagSequence(readType); }
	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getNumCalls(readType, encoding); }
	bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->isColorsAvailable(readType, encoding); }
	const std::vector<EncodingInfo>& getEncodingInfo(XsqReadType const& readType) { return mp_impl->getEncodingInfo(readType); }
private :
	Lane(boost::weak_ptr<LaneImpl> const& impl) : mp_impl(impl) {}
};

class PanelContainer : public PanelContainerI {
	boost::shared_ptr<PanelContainerImpl> mp_impl;
public:
	PanelContainer(boost::shared_ptr<PanelContainerImpl> const& impl) : mp_impl(impl) {}
	virtual ~PanelContainer() {;}
	LibraryType getLibraryType() const { return mp_impl->getLibraryType(); }
	std::string getHDF5Path()    const { return mp_impl->getHDF5Path(); }
	std::string getName()   const { return mp_impl->getName(); }
	std::string getPrimerBases(XsqReadType const& type) const { return mp_impl->getPrimerBases(type); }
	std::string getFilename()    const { return mp_impl->getFilename(); }
	size_t size() const {return mp_impl->size(); }
	std::vector<Panel> getPanels() const;
};

class Panel : public PanelI {
	friend class Fragment;
	friend class PanelFragmentsIterator<const Fragment>;
	boost::shared_ptr<PanelImpl> mp_impl;
public :
	Panel() : mp_impl() {}
	Panel(PanelContainer &panelContainerInfo, std::string const& hpath);
	Panel(boost::shared_ptr<PanelImpl> const& impl) : mp_impl(impl) {}
	virtual ~Panel() {;}

	bool operator==(Panel const& other) const { return mp_impl == other.mp_impl; }
	bool operator!=(Panel const& other) const { return mp_impl != other.mp_impl; }

	size_t getPanelNumber() const { return mp_impl->getPanelNumber(); }
	bool isPrecededBy(Panel const& other) const { return mp_impl->isPrecededBy(*other.mp_impl); }
	size_t size() const { return mp_impl->size(); }
	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getNumCalls(readType, encoding); }
	std::string getPanelContainerName() const { return mp_impl->getPanelContainerName(); }
	size_t getPanelContainerIndex() const { return mp_impl->getPanelContainerIndex(); }
	std::string getFilename() const { return mp_impl->getFilename(); }
	size_t getFileIndex() const { return mp_impl->getFileIndex(); }
	size_t getMaxNumCalls() const { return mp_impl->getMaxNumCalls(); }

	typedef PanelFragmentsIterator<Fragment const> panel_fragments_const_iterator;
	panel_fragments_const_iterator begin(bool const& maskFilteredAndTrimmedBases) const;
	panel_fragments_const_iterator begin() const;
	panel_fragments_const_iterator end() const;
	void release() { if (mp_impl != NULL) mp_impl->release() ; } //Frees up cached data.
	const bool isReadTypeDataVirtual(XsqReadType const& readType) const { return mp_impl->isReadTypeDataVirtual(readType); }
};

class Fragment : public FragmentI {
	friend class Panel;
	friend class PanelFragmentsIterator<Fragment const>;
	boost::shared_ptr<FragmentImpl> mp_impl;
public :
	Fragment(Panel parent, size_t const& rowNum, bool const& maskFilteredAndTrimmedBases, size_t const& maxNumCalls) : mp_impl(new FragmentImpl(parent.mp_impl, rowNum, maskFilteredAndTrimmedBases, maxNumCalls)) {}
	virtual ~Fragment() {;}
	bool operator==(Fragment const& other) const { return *this->mp_impl == *other.mp_impl; }

	size_t getRowNum() const { return mp_impl->getRowNum(); }
	std::string getName(XsqReadType const& readType) const {
# ifdef XSQ_READER_TRACE_ON
	std::cerr << "entering Read::getName() " << std::endl;
# endif
		return mp_impl->getName(readType);
	}
	std::string getName() const { return mp_impl->getName(); }
	const char * getNameChar() const { return mp_impl->getNameChar(); }
	size_t getX() const { return mp_impl->getX(); }
	size_t getY() const { return mp_impl->getY(); }
	unsigned char* getCallQVs(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getCallQVs(readType, encoding); }
	void getCalls(uint8_t** colors, uint8_t** qvs, size_t &num_calls, XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getCalls(colors, qvs, num_calls, readType, encoding); }
 void getCalls(uint8_t* colors, uint8_t* qvs, XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getCalls(colors, qvs, readType, encoding); }	
	void setCalls(uint8_t* colors, uint8_t* qvs, XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->setCalls(colors, qvs, readType, encoding); }
	std::string getBases(XsqReadType const& readType) const { return mp_impl->getBases(readType); }
	QualityValueArray getBaseQVs(XsqReadType const& readType) const { return mp_impl->getBaseQVs(readType); }
	std::string getColors(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getColors(readType, encoding); }
	char * getBaseRead(XsqReadType const& readType) const { return mp_impl->getBaseRead(readType); }
	char * getColorRead(XsqReadType const& readType) const { return mp_impl->getColorRead(readType); }
	char * getColorReadNoPrimer(XsqReadType const& readType) const { return mp_impl->getColorReadNoPrimer(readType); }
	QualityValueArray getQualityValues(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getQualityValues(readType, encoding); }
	bool isFiltered() const { return mp_impl->isFiltered(); }
	bool isReadFiltered(XsqReadType const& readType) const { return mp_impl->isReadFiltered(readType); }
	size_t getTrim(XsqReadType const& readType, bool const& start ) const { return mp_impl->getTrim(readType, start); }
	size_t getPanelNumber()       const { return mp_impl->getPanelNumber(); }
	LibraryType getLibraryType()   const { return mp_impl->getLibraryType(); }
	std::string getFilename()      const { return mp_impl->getFilename(); }
	size_t getFileNumber() const { return mp_impl->getFileNumber(); }
	std::string getPanelContainerName()     const { return mp_impl->getPanelContainerName(); }
	size_t getPanelContainerNumber() const { return mp_impl->getPanelContainerNumber(); }
	std::string getPrimerBases(XsqReadType const& readType)   const { return mp_impl->getPrimerBases(readType); }
	size_t getNumCalls(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->getNumCalls(readType, encoding); }
	bool isColorsAvailable(XsqReadType const& readType, ColorEncoding const& encoding) const { return mp_impl->isColorsAvailable(readType, encoding); }
	bool isBasesAvailable(XsqReadType const& readType) const { return this->isColorsAvailable(readType, BASE_ENCODING); }
	LaneI& getLane() const { return mp_impl->getLane(); }
	PanelContainerI& getPanelContainer() const { return mp_impl->getPanelContainer(); }
	PanelI& getPanel() const { return mp_impl->getPanel(); }
	const std::vector<XsqReadType>& getReadTypes() const { return mp_impl->getReadTypes(); }

private :
 
	size_t incrementRowNum() {
		mp_impl->incrementRowNum();
		return this->getRowNum();
	}
};

//Read iteration
template <class Value>
class PanelFragmentsIterator
	: public boost::iterator_facade<PanelFragmentsIterator<Value>, Value, boost::forward_traversal_tag>
{

	Panel m_panel;
	Fragment m_Fragment;
	/// \cond DEV
	friend class boost::iterator_core_access;

   struct enabler {};
   /// \endcond

public:
   	PanelFragmentsIterator(): m_panel(), m_Fragment(m_panel, NPOS, false, 0) {}

	explicit PanelFragmentsIterator(Panel panel, bool const& maskFilteredAndTrimmedBases) : m_panel(panel), m_Fragment(m_panel, 0, maskFilteredAndTrimmedBases, m_panel.getMaxNumCalls()) {}

	template <class OtherValue>
	PanelFragmentsIterator(PanelFragmentsIterator<OtherValue> const& other,
			            typename boost::enable_if<boost::is_convertible<OtherValue*,Value*>, enabler>::type = enabler());


 private:
	template <class OtherValue>
	bool equal(PanelFragmentsIterator<OtherValue> const& other) const {
		if (this->m_Fragment.getRowNum() == NPOS && other.m_Fragment.getRowNum() == NPOS) return true;
		return this->m_panel == other.m_panel && this->m_Fragment == other.m_Fragment;
	}

	void increment() {
# ifdef XSQ_READER_TRACE_ON
		std::cerr << "entering increment()" << std::endl;
# endif
		m_Fragment.incrementRowNum();
# ifdef XSQ_READER_TRACE_ON
		std::cerr << "exiting increment()" << std::endl;
# endif
	}

	Value& dereference() const { return m_Fragment;	}
};


//Fragment iteration
template <class Value>
class FragmentIterator
	: public boost::iterator_facade<FragmentIterator<Value>, Value, boost::forward_traversal_tag>
{

	std::vector<Panel>* m_panels;
	std::vector<Panel>::iterator m_panelIterator;
	Panel::panel_fragments_const_iterator m_FragmentsIterator;
	bool m_skipFilteredFragments;
	bool m_maskFilteredAndTrimmedBases;
	/// \cond DEV
	friend class boost::iterator_core_access;
	template <class> friend class FragmentIterator;

    struct enabler {};
    /// \endcond

public:
	FragmentIterator() : m_panels(NULL), m_skipFilteredFragments(false), m_maskFilteredAndTrimmedBases(false) {}

	explicit FragmentIterator(std::vector<Panel>* panels, bool const& skipFilteredReads, bool const& maskFilteredAndTrimmedBases) :
			m_panels(panels),
			m_panelIterator(m_panels->begin()),
			m_FragmentsIterator(Panel::panel_fragments_const_iterator(*m_panelIterator, maskFilteredAndTrimmedBases)),
			m_skipFilteredFragments(skipFilteredReads),
			m_maskFilteredAndTrimmedBases(maskFilteredAndTrimmedBases)
	{ }

	template <class OtherValue>
	FragmentIterator(FragmentIterator<OtherValue> const& other, typename boost::enable_if<boost::is_convertible<OtherValue*,Value*>, enabler>::type = enabler());

 private:
	template <class OtherValue>
	bool equal(FragmentIterator<OtherValue> const& other) const {
		return (this->m_panels == NULL && other.m_panels == NULL) ||
			   (this->m_panelIterator == other.m_panelIterator && this->m_FragmentsIterator == other.m_FragmentsIterator);
	}

	void increment() {
		do {
			if (++this->m_FragmentsIterator == this->m_panelIterator->end()) {
				this->m_panelIterator->release();
				if (++this->m_panelIterator == this->m_panels->end() ) {
					m_panels = NULL;
					m_panelIterator = std::vector<Panel>::iterator();
					m_FragmentsIterator = Panel::panel_fragments_const_iterator();
				} else {
					this->m_FragmentsIterator = Panel::panel_fragments_const_iterator(*this->m_panelIterator, m_maskFilteredAndTrimmedBases);
				}
			}
		} while (m_panels != NULL && m_skipFilteredFragments && m_FragmentsIterator->isFiltered());
	}

	Value& dereference() const {
		return *this->m_FragmentsIterator;
	}

};

/*!
 * Defines a range of reads an Lts File.
 */
class PanelRangeSpecifier : public URL {
public :
	struct PanelPosition {
		size_t libraryIndex;
		size_t panelIndex;
		PanelPosition(std::string const& s, size_t const& defaultLibraryIndex, size_t const& defaultPanelIndex) : libraryIndex(defaultLibraryIndex), panelIndex(defaultPanelIndex) {
			if (s.empty()) return;
			try {
				std::vector<std::string> vec;
				boost::algorithm::split_regex(vec, s, boost::regex("\\."));
				this->libraryIndex = boost::lexical_cast<size_t>(vec[0]);
				this->panelIndex = vec.size() > 1 ? boost::lexical_cast<size_t>(vec[1]) : defaultPanelIndex;
			} catch (std::exception e) {
				throw parse_exception("'" + s + "' cannot be parsed as a PanelRangeSpecifier.");
			}
		}
		PanelPosition(size_t const& libraryIndex, size_t const& panelIndex) :libraryIndex(libraryIndex), panelIndex(panelIndex) {}
		bool operator==(PanelPosition const& rhs) const {
			return this->libraryIndex == rhs.libraryIndex && this->panelIndex == rhs.panelIndex;
		}
		bool operator<(PanelPosition const& rhs) const {
			if (this->libraryIndex != rhs.libraryIndex) return this->libraryIndex < rhs.libraryIndex;
			return this->panelIndex < rhs.panelIndex;
		}
		bool operator>(PanelPosition const& rhs) const {
			if (this->libraryIndex != rhs.libraryIndex) return this->libraryIndex > rhs.libraryIndex;
			return this->panelIndex > rhs.panelIndex;
		}
		bool operator<=(PanelPosition const& rhs) const {
			if (*this == rhs) return true;
			return *this < rhs;
		}
		bool operator>=(PanelPosition const& rhs) const {
			if (*this == rhs) return true;
			return *this > rhs;
		}
	};
private :
	PanelPosition panelStart;
	PanelPosition panelEnd;
	std::vector<XsqReadType> readTypes;
	static std::string toUrl(std::string const& filename_, PanelPosition const& panelStart_, PanelPosition const& panelEnd_) {
		std::stringstream s;
		s << filename_ << "?start=" << panelStart_.libraryIndex << "." << panelStart_.panelIndex << "&end=" << panelEnd_.libraryIndex << "." << panelEnd_.panelIndex;
		return s.str();
	}
public :
	PanelRangeSpecifier(std::string const& string) :
		URL(string),
		panelStart(this->getParameter("start"),0,0),
		panelEnd(this->getParameter("end"), NPOS, NPOS),
		readTypes(toReadTypes(this->getParameterValues("tag")))
	{}
	PanelRangeSpecifier(std::string const& filename_, PanelPosition const& panelStart_, PanelPosition const& panelEnd_ ) :
		URL(toUrl(filename_, panelStart_, panelEnd_)),
		panelStart(panelStart_),
		panelEnd(panelEnd_)
	{}
	PanelRangeSpecifier(PanelRangeSpecifier const& rhs) :
		URL(rhs), panelStart(rhs.panelStart), panelEnd(rhs.panelEnd), readTypes(rhs.readTypes)
	{}
	bool accept(PanelI const& panel) const;
	bool operator()(PanelI const& panel) const {
		return this->accept(panel);
	}
	PanelPosition getPanelStart() const { return this->panelStart; }
	PanelPosition getPanelEnd() const { return this->panelEnd; }
	std::vector<XsqReadType> const& getReadTypes() const { return this->readTypes; }
	bool specifiesPanel(Panel panel);
	std::string str() const {
		return toUrl(this->getPath(), this->getPanelStart(), this->getPanelEnd() );
	}

};

//std::ostream& operator<<(std::ostream& lhs, PanelRangeSpecifier const& rhs);

/*!
 * Class used to open and iterate over reads in XSQ files.
 *
 * Using XSQ content requires the following steps:
 *
 * -# Creating the XsqReader Object.
 * -# Adding source Data to the XsqReader using open() with Panel Range Specifiers.
 *    open() may be called multiple times.
 * -# Creating iterators on the reads.
 *
 * The
 *
 * XSQ is a format for storing NGS reads. XSQ content is organized heirarchically as Files, PanelContainers, Panels and
 * Fragments.  One accesses XSQ content through iterators over Fragments.  Fragments are always accessed according
 * to their natural ordering which is by FILE NUMBER , PanelContaner NUMBER, Panel NUMBER, Y COORDINATE, X COORDINATE.
 *
 * PanelContainerG: The content of the /<LibraryName>_<Index> or /DefaultLibrary groups in an XSQ file.
 *
 * PanelG : The content of the <ImageUnitID> tag in an XSQ file.
 *
 * PanelG#: Panel number of the PanelG.  PanelGs may be uniquely identified within a single XSQ file
 *          with a 2 numbers separated by a dot:  The first number specifies the PanelContainerG, the
 *          second specifies the PanelG#:  ex. 1.1 specifies PanelG 1 in PanelContainerG 1.
 *
 * TagG : The F3,R3,R5 groups containing reads from a single tag.
 *
 * MaskString:  A string that specifies masked positions in a read.
 * 				The string follows a CIGAR like syntax with M(mask) and I(include) operators.
 * 				The mask status of the last specified position extends to all unspecified positions.
 * 				The empty string "" is interpretted as equals "1I";
 *              Example1:  ""         : All positions are included.
 *              		   "50I"      : All positions are included.
 *                         "10I1M39I" : Position 11 is masked, all others included.
 *
 * Panel Range Specifier(PRS) is a URL that specifies a 'contiguous' range of reads to iterate over.
 *
 * In this context, the reads in an XSQ File are naturally sorted by: PanelContainerG#, PanelG#, Y, X.
 * A PRS is a URL that specifies a contiguous sublist of reads sorted this way.
 *
 * 		file://<Path>[?<query>]
 *
 * The file:// prefix is optional.  Paths that do not start with / are interpreted as relative file paths.
 * Paths that start with / are interpreted as absolute file paths.
 *
 * A URL with no query specifies all the reads in the file.  The content can be constrained
 * to a single contiguous range by using these query keys:
 *
 *   Key     |  Meaning                                       | Multiple Key Interpretation
 * ------------------------------------------------------------------------------------
 *   start   | PanelG Start, syntax=<PanelGID>*                | not allowed.
 *   end     | PanelG End, syntax= same as start.             | not allowed.
 *
 * A PanelGId is a dot separated tuple of 2 numbers:  The first specifies the PanelContainer number, the second specifies
 * the PanelG number.  Ex. 1.1 = PanelG 1 in PanelContainerG 1.
 *
 * Examples:
 *
 * file:///path/to/file1.xsq   All reads in /path/to/file1.xsq
 * file1.xsq                   All reads in file1.xsq (relative path)
 * file1.xsq?start=2.1         All reads in file1.xsq between Panel 2.1 and the end of the file.
 * file1.xsq?start=2.1&end=3   All reads in file1.xsq between Panel 2.1 and the end of Panel Container 3.
 * file1.xsq?start=2.1&end=3.100  All reads in file1.xsq between Panel 2.1 and Panel 3.100.
 *
 *  Individual reads can be filtered or masked using the following:
 *  ***THESE ARE OTHER KEYS THAT MAY OR MAY NOT BE USED, THEY CURRENTLY DO NOTHING***
 *
 *   Key     |  Meaning                                       | Multiple Key Interpretation
 * ------------------------------------------------------------------------------------
 *   tag     | F3, R3, F5 tags                                | logical OR
 *   mask    | syntax= <BarcodeG#>.<MaskString>               | only one per barcode allowed.
 *   bfilter | bead level filter syntax= single digit         | not allowed.
 *   rfilter | read level filter syntax= single digit         | not allowed.
 *
 * A bead is filtered if bfilter & read.filter > 0, likewise for rfilter.
 *
 * Every XSQ file in the XSQReader is associated with a unique file number that governs
 * the natural ordering of the files.  The natural order of reads when multiple files are
 * used is File, PanelContainer, Panel, Y, X.  This file number is chosen when calling
 * XsqReader.open().  Conflicting file number assignments will throw an exception.
 *
 *
 * Examples of usage:
 *
 * Iterate over all F3 reads in a single Xsq file.
 *
 * \code
 * XsqReader xsq;
 * xsq.open("file0.xsq", 0);
 * for (XsqReader::const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
 * 		fragment->getName(XSQ_READ_TYPE_F3);
 * 	    fragment->getBases();
 * }
 * \endcode
 * ----------
 *
 * Iterate over all F3 reads in a three Xsq files.
 * \code
 * XsqReader xsq;
 * xsq.open("file0.xsq",0);
 * xsq.open("file1.xsq",1);
 * xsq.open("file2.xsq",2);
 * for (XsqReader::const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
 * 		fragment->getName(XSQ_READ_TYPE_F3)
 * 		fragment->getBases();
 * }
 * \endcode
 * ----------
 *
 * Iterate over all reads from PanelContainer 1 in 2 xsq files.
 * \code
 * XsqReader xsq;
 * xsq.open("file0.xsq?start=1&end=1",0);
 * xsq.open("file1.xsq?start=1&end=1",1);
 * for (XsqReader::const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
 * 		fragment->getName(XSQ_READ_TYPE_F3)
 * 		fragment->getBases();
 * }
 * \endcode
 * ----------
 * Iterate over 2 non contiguous PanelContainers in 1 xsq file.
 * \code
 * XsqReader xsq;
 * xsq.open("file0.xsq?start=1&end=1");
 * xsq.open("file0.xsq?start=3&end=3");
 * for (XsqReader::const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
 * 		fragment->getName(XSQ_READ_TYPE_F3)
 * 		fragment->getBases();
 * }
 * \endcode
 * ----------
 *
 * Iterate over panels 1-100 from PanelContainer 1 in 1 xsq file.
 * \code
 * XsqReader xsq;
 * xsq.open("file0.xsq?start=1.1&end=1.100");
 * for (XsqReader::const_iterator fragment = xsq.begin(); fragment != xsq.end(); ++fragment) {
 * 		fragment->getName(XSQ_READ_TYPE_F3)
 * 		fragment->getBases();
 * }
 * \endcode
 * ------------
 *
 * * This example demonstrates creating an XsqReader and then subdividing it into
 * several smaller XsqReaders. This could be used to divide work among several
 * threads.
 *
 * \code
 * XsqReader xsq;
 * xsq.open("minimal_xsq_barcode_0.h5");
 * vector<XsqReader> readers = xsq.divideEvenly(8,1000000);
 * \endcode
 *
 * ***NONE OF THE EXAMPLES BELOW ARE IMPLEMENTED***
 *
 *This example demonstrates creating an XsqReader and iterating over reads
 *having neither read or bead level filtering flags set.
 *
 *  \code
 *	class ExampleFilter
 *	{
 *		public:
 *			ExampleFilter() {}
 *			bool operator() (Read const &a) const
 *			{
 *				return a.getBeadFilterFlag() == 0 && a.getReadFilterFlag() == 0
 *			}
 *	};
 *
 *	ExampleFilter filter;
 *
 *	XsqReader xsq;
 *	xsq.open("minimal_xsq_barcode_0.h5");
 *	XsqReader::filter_iterator<ExampleFilter> iter(filter, xsq.begin(), xsq.end());
 *	XsqReader::filter_iterator<ExampleFilter> end(filter, xsq.end(), xsq.end());
 *
 *	for (XsqReader::filter_iterator<ExampleFilter> read = iter; read != end; ++read) {
 *		//do something with read.
 *	}
 *	return 0;
 * \endcode
 *
 * --------------
 *
 *
 *
 */

class XsqReader
{
	enum State {
		INIT,       //Setting up.  No Iterators created.
		ITERATING,  //Iterators have been created.
		CLOSED     //Closed.  No more iterators.
	};

	std::map<size_t, Lane> m_lanes; //Key = filenumber
	std::vector<PanelRangeSpecifier> m_PanelRangeSpecifiers;
	std::vector<Panel> m_Panels;
	bool m_Panels_correct;
	State m_state;
	int m_errorState;
	bool m_readOnly;
	bool m_skipFilteredFragments;
	bool m_maskFilteredAndTrimmedBases;

	static const bool DEFAULT_READ_ONLY = true;
	static const bool DEFAULT_SKIP_FILTERED_READS = true;
	static const bool DEFAULT_MASK_FILTERED_AND_TRIMMED_BASES = true;

	public:

		XsqReader() :
			m_state(INIT),
			m_errorState(0),
			m_readOnly(DEFAULT_READ_ONLY),
			m_skipFilteredFragments(DEFAULT_SKIP_FILTERED_READS),
			m_maskFilteredAndTrimmedBases(DEFAULT_MASK_FILTERED_AND_TRIMMED_BASES)
		{}

		XsqReader(bool const& readOnly) :
			m_state(INIT),
			m_errorState(0),
			m_readOnly(readOnly),
			m_skipFilteredFragments(DEFAULT_SKIP_FILTERED_READS),
			m_maskFilteredAndTrimmedBases(DEFAULT_MASK_FILTERED_AND_TRIMMED_BASES)
		{}
private:
        XsqReader(std::vector<Panel> const& panels, bool const& readOnly,
        		bool const& skipFilteredFragments, bool const& maskFilteredAndTrimmedBases);

public:

		~XsqReader() { close(); }

		/*!
		  Close all open XSQ files.
		*/
		void close();

		/*!
		 * Open the specified range of reads and add it to the set.
		 * An exception will be thrown if any iterators have
		 * been created.
		 */
		bool open(PanelRangeSpecifier const& specifier, size_t const& filenumber);

		/*!
		 * same as open(PanelRangeSpecifier(specifier));
		 */
		bool open(std::string const& specifier, size_t const& filenumber) {
			PanelRangeSpecifier s(specifier);
			return open(s, filenumber);
		}

		void setSkipFilteredFragments(bool const& skipFilteredFragments) {
			m_skipFilteredFragments = skipFilteredFragments;
		}

		void setMaskFilteredAndTrimmedBases(bool const& maskFilteredAndTrimmedBases) {
			m_maskFilteredAndTrimmedBases = maskFilteredAndTrimmedBases;
		}

		/*!
		 * \return the number of fragments specified by this XsqReader.
		 */
		size_t size();

		size_t numPanels() {
			if (!m_Panels_correct) updatePanels();
			return this->m_Panels.size();
		}

		/*!
		 * Create new XsqReaders, each containing evenly divided subsets of this Reader.
		 * if this reader contains fewer than minNumFragmentsPerReader fragments, it will return a vector
		 * containing this reader.
		 */
		std::vector<XsqReader> divideEvenly(size_t const& numReaders, uint32_t const& minNumFragmentsPerReader);

		/*!
		 * \return 2 new XsqReaders of approximately same size.
		 * \throw exception if Xsq reader contains fewer than 2 fragments.
		 */
		std::vector<XsqReader> bisect()
		{
			if (size() < 2) throw XsqException("too few reads"); //TODO better exception.
			return divideEvenly(2, 0);
		}

		std::vector<std::string> getURLs() const;

		typedef FragmentIterator<Fragment const> fragment_const_iterator;

		template<class Predicate>
		struct filter_iterator : public boost::filter_iterator<Predicate, XsqReader::fragment_const_iterator >
		{
			filter_iterator(Predicate const& predicate, XsqReader::fragment_const_iterator const& begin, XsqReader::fragment_const_iterator const& end) :
				boost::filter_iterator<Predicate,  XsqReader::fragment_const_iterator >(predicate, begin, end)
			{}
		}; // struct filter_iterator

		/*!
		  Begin iterator
		  \return iterator to the first alignment record. Note:  To avoid the overhead of repeated creation of Fragment
		  Objects, Dereferencing the iterator returned by begin() will always return the same Fragment Object. An explicit
		  copy-by-value is be required to examine multiple Fragment objects.
		*/
		fragment_const_iterator begin();

		/*!
		  End iterator
		  \return iterator to the end of the alignment records
		*/
		fragment_const_iterator end();

		//End Fragment Iteration

		//Dataset Iteration
		typedef std::vector<Panel>::iterator panel_iterator;
		typedef std::vector<Panel>::const_iterator panel_const_iterator;
		panel_iterator panels_begin();
		panel_iterator panels_end();


		//panel_const_iterator datasets_begin();
		//panel_const_iterator datasets_end();

   private:
		void updatePanels();

}; //class XsqReader



/*!
  * Class for writing to an Xsq file.
  */
class XsqWriter {

	 boost::filesystem::path path;
	 hid_t hdfHandle;
	 std::map<size_t, hid_t> panelContainerGroups;  //A vector implementation may be better.
	 size_t fileNumber;
	 std::map<XsqReadType, std::map<ColorEncoding, EncodingInfo> > encodings;
	 struct CallAndQvData {
		 size_t readLength;
		 boost::shared_array<unsigned char*> arr;
		 boost::shared_array<unsigned char> data;
	 };
	 
	 
	 struct PanelBuffer {
		 PanelBuffer() : bufferSize(NPOS), actualSize(NPOS), barcode(-1), panel(0), callAndQVData(), isReadTypeVirtual(), yxArr(), yxData() {;}
		 void reset(size_t const& size, int barcode, int panel);
		 void prepareForReads(XsqReadType const& readType, ColorEncoding const& encoding, size_t const& readLength);
		 size_t bufferSize;
		 size_t actualSize;
		 int barcode;
		 unsigned int panel;
		 std::map<XsqReadType, std::map<ColorEncoding, CallAndQvData> > callAndQVData;
		 std::map<XsqReadType, bool> isReadTypeVirtual;
		 boost::shared_array<uint16_t*> yxArr;
		 boost::shared_array<uint16_t> yxData;
	 };
	
	 // Was boost::thread::id
	 typedef std::map<pthread_t, PanelBuffer> PanelBufferMap;
 	 PanelBufferMap panelBuffers;

public :
	/*!
	 * Construct a new XsqWriter.
	 */
	XsqWriter(std::string const& filepath) : path(filepath), hdfHandle(-1), panelContainerGroups(), fileNumber(NPOS), encodings(), panelBuffers()
	{
		hdfHandle = H5Fcreate(this->path.string().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
		if (hdfHandle < 0) {
			std::stringstream msg;
			msg << "error opening " << filepath << " for Xsq writing.";
			throw XsqException(msg.str());
		}
	}

	/*!
	 * Write a read to the XsqWriter.
	 * \param fragment the read to be written.
	 * \return this XsqWriter
	 * \throw exception if read is not associated an unregistered LaneInfo.
	 */
	XsqWriter& operator<<(FragmentI const& fragment);

	void close() {
		for (PanelBufferMap::const_iterator it = panelBuffers.begin();
			 it != panelBuffers.end(); ++it) {
			write(it->second);
		}
		panelBuffers.clear();
		if (H5Fclose(this->hdfHandle) < 0) {
			std::stringstream msg;
			msg << "Error closing hdfHandle for " << path;
			throw new XsqException(msg.str());
		}
	}

private :
	void init(LaneI& lane);
	void write(PanelBuffer const& buffer);
	PanelBuffer & getThreadPanelBuffer();
};

class XsqMultiWriter
{
	boost::filesystem::path m_outputDirectory;
	std::map<size_t, XsqWriter> writers;
	struct closer { void operator() (std::pair<size_t, XsqWriter> p) {
			p.second.close();
	}};
public :
	XsqMultiWriter(std::string const& directoryPath) : m_outputDirectory(directoryPath), writers() {}
 	XsqMultiWriter& operator<<(FragmentI const& fragment) {
 		size_t fileNumber = fragment.getFileNumber();
 		std::map<size_t, XsqWriter>::iterator it = writers.find(fileNumber);
 		XsqWriter& writer = (it == writers.end()) ? createXsqFile(fileNumber, boost::filesystem::path(fragment.getFilename()).filename()) : it->second;
 		writer << fragment;
 		return *this;
 	}
 	void close() { std::for_each(writers.begin(), writers.end(), closer()); }

private :
 	XsqWriter& createXsqFile(size_t const& fileNumber, std::string const& fileName);
}; // class XsqMultiWriter

} //namespace lifetechnologies

#endif //XSQ_READER_HPP_

