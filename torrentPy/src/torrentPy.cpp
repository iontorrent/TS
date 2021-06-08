// Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
#include <Python.h>

#include "torrentPy.h"

#include "numpy/noprefix.h"
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>
#include <ctime>
#include <functional>
#include <set>
#include <libgen.h>
#include "api/BamReader.h"
#include "file-io/ion_util.h"

#include "SystemMagicDefines.h"
#include "Mask.h"
#include "Image.h"
#include "RawWells.h"
#include "DiffEqModel.h"
#include "RegionParams.h"
#include "DNTPRiseModel.h"
#include "ReservoirSample.h"
#include "Calibration/FlowAlignment.h"
#include "VariantCaller/Bookkeeping/MiscUtil.h"
#include "BaseCaller/DPTreephaser.h"
#include "BaseCaller/BaseCallerUtils.h"
#include "CorrNoiseCorrector.h"
#include "PairPixelXtalkCorrector.h"
#include "IonErr.h"
#include "Calibration/LinearCalibrationModel.h"


using namespace boost::python;
using namespace std;

template<typename T> handle<> toNumpy( const T& d ){

    NPY_TYPES ArrayType = NPY_OBJECT;
    if( boost::is_same<typename T::value_type, int>::value )
        ArrayType = NPY_INT;
    if( boost::is_same<typename T::value_type, unsigned int>::value )
        ArrayType = NPY_UINT;
    else if ( boost::is_same<typename T::value_type, float>::value )
        ArrayType =  NPY_FLOAT;
    else if ( boost::is_same<typename T::value_type, char>::value )
        ArrayType =  NPY_CHAR;
    else if ( boost::is_same<typename T::value_type, double>::value )
        ArrayType =  NPY_DOUBLE;
    else if ( boost::is_same<typename T::value_type, short>::value )
        ArrayType =  NPY_SHORT;
    else if ( boost::is_same<typename T::value_type, unsigned short>::value )
        ArrayType =  NPY_USHORT;
    else if ( boost::is_same<typename T::value_type, long>::value )
        ArrayType =  NPY_LONG;
    else if ( boost::is_same<typename T::value_type, unsigned long>::value )
        ArrayType =  NPY_ULONG;

    npy_intp dims[]={(npy_intp)d.size()};
    handle<> array( PyArray_SimpleNew(1,dims, ArrayType ) );

    int i = 0;
    for(typename T::const_iterator it = d.begin(); it != d.end(); ++it ){
        typename T::value_type* data = (typename T::value_type*)PyArray_GETPTR1(array.get(),i++);
        *data = *it;
    }
    return array;
}

template<typename T_SRC, typename T_OUT> void convert_numpy(vector<T_OUT>& outputArr, PyObject* dataH)
{
    int len = outputArr.size();
    for(int i=0; i<len; ++i){
        npy_intp idx=static_cast<npy_intp>(i);
        T_OUT val = static_cast<T_OUT>(boost::python::extract<T_SRC>(PyArray_GETITEM(dataH,PyArray_GETPTR1(dataH, idx))));
        outputArr[i]=val;
    }
}

template<typename T> vector<T> fromNumpy( const boost::python::object& data ){
    PyObject* dataH = data.ptr();
    int len = PyArray_Size( dataH );
    int type = PyArray_TYPE(dataH);

    if( PyArray_NDIM(dataH) != 1 ){
        throw std::runtime_error("Only 1d arrays are supported");
    }

    vector<T> outputArr(len);

    if( type==NPY_DOUBLE )
            convert_numpy<npy_double,T>(outputArr,dataH);
    else if( type == NPY_FLOAT)
        convert_numpy<npy_float,T>(outputArr,dataH);
    else if( type==NPY_INT )
        convert_numpy<npy_int,T>(outputArr,dataH);
    else if( type == NPY_UINT)
        convert_numpy<npy_uint,T>(outputArr,dataH);
    else if( type == NPY_SHORT)
        convert_numpy<npy_short,T>(outputArr,dataH);
    else if( type == NPY_CHAR)
        convert_numpy<npy_char,T>(outputArr,dataH);
    else if( type == NPY_LONG)
        convert_numpy<npy_long,T>(outputArr,dataH);
    else if( type == NPY_ULONG)
        convert_numpy<npy_ulong,T>(outputArr,dataH);

    return outputArr;
}

// I need the constructor for object assignment. Otherwise, two objects share the same dpTreephaser that could be deleted if the destructor of one object is called.
TreePhaser::TreePhaser(const TreePhaser& copy_me){
    *this = copy_me;
    dpTreephaser = new DPTreephaser(*(copy_me.dpTreephaser));
}

TreePhaser::TreePhaser(const string &_flowOrder)
{
    flowOrder = ion::FlowOrder(_flowOrder, _flowOrder.length());
    dpTreephaser = new DPTreephaser(flowOrder);
}

void TreePhaser::setCalibFromTxtFile(const string &model_file, int threshold)
{
    calibModel.InitializeModelFromTxtFile(model_file, threshold);
}

void TreePhaser::setCalibFromBamFile(const string &bam_file){
	bam_header_recalibration.clear();
	block_hash.clear();
	BamTools::BamReader bamReader;
    if(!bamReader.Open(bam_file))
        throw std::runtime_error( std::string("Can't open bam file: ")+bam_file );
    BamTools::SamHeader samHeader = bamReader.GetHeader();
    bamReader.Close();

    if (not samHeader.HasComments()){
    	cerr << "Failed to set Calibration from the BAM file: No @CO tag found from the BAM header." << endl;
    	return;
    }

    // Get eligible run_id for the flowOrder being initialized
    set<string> eligible_run_id;
    for (BamTools::SamReadGroupConstIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr) {
        if (itr->ID.empty()){
            cerr << "TVC ERROR: BAM file has a read group without ID." << endl;
            exit(EXIT_FAILURE);
        }
        // We need a flow order to do variant calling so throw an error if there is none.
        if (not itr->HasFlowOrder()) {
            cerr << "TVC ERROR: read group " << itr->ID << " does not have a flow order." << endl;
        exit(EXIT_FAILURE);
        }
        // I only accept the RUNID whose flow order is the same as the flow_order being initialized to TreePhaser.
        if (itr->FlowOrder == flowOrder.full_nucs()){
        	string my_run_id = itr->ID.substr(0, itr->ID.find("."));
        	eligible_run_id.insert(my_run_id);
        }
    }

    unsigned int num_parsing_errors = 0;
    bool is_live = false;
    // Read comment lines from Sam header
    for (unsigned int i_co=0; i_co<samHeader.Comments.size(); i_co++) {
      // There might be all sorts of comments in the file
      // therefore must find the unlikely magic code in the line before trying to parse
      string magic_code = "6d5b9d29ede5f176a4711d415d769108"; // md5hash "This uniquely identifies json comments for recalibration."

      if (samHeader.Comments[i_co].find(magic_code) == std::string::npos) {
        //cout << endl << "No magic code found in comment line "<< i_co <<endl;
        //cout << samHeader.Comments.at(i_co) << endl;
        continue;
      }

      // Parse recalibration Json object
      Json::Value recal_params(Json::objectValue);
      Json::Reader recal_reader;
      if (not recal_reader.parse(samHeader.Comments[i_co], recal_params)) {
        cerr << "Failed to parse recalibration comment line " << recal_reader.getFormattedErrorMessages() << endl;
        num_parsing_errors++;
        continue;
      }

      string my_block_key = recal_params["MasterKey"].asString();

      // Assumes that the MasterKey is written in the format <run_id>.block_X<x_offset>_Y<y_offset>
      int end_runid = my_block_key.find(".");
      int x_loc     = my_block_key.find("block_X")+7;
      int y_loc     = my_block_key.find("_Y");

      // glorified assembly language
      string runid = my_block_key.substr(0,end_runid);
      int x_coord = atoi(my_block_key.substr(x_loc,y_loc-x_loc).c_str());
      int y_coord = atoi(my_block_key.substr(y_loc+2, my_block_key.size()-y_loc+2).c_str());

      // Skip the ineligible RUNID
      if (eligible_run_id.find(runid) == eligible_run_id.end()){
    	  cout << "The flow order used in the run "<< runid << " does not match the flow order being initialized to the TreePhaser object."<< endl;
    	  continue;
      }

      //recalModel.InitializeFromJSON(recal_params, my_block_key, false, max_flows_by_run_id.at(runid));
      // void RecalibrationModel::InitializeFromJSON(Json::Value &recal_params, string &my_block_key, bool spam_enabled, int over_flow_protect) {
      // The calibration comment line contains  info about the hp threshold used during base calling, so set to zero here
      // XXX FIXME: The number of flows in the TVC group can be larger than the one specified in the calibration block.
      LinearCalibrationModel tmp_recalModel;
      tmp_recalModel.InitializeModelFromJson(recal_params, flowOrder.num_flows());
      bam_header_recalibration.insert(pair<string,LinearCalibrationModel>(my_block_key, tmp_recalModel));
      block_hash.insert(pair<string, pair<int,int > >(runid,pair<int,int>(x_coord,y_coord)));
      is_live = true;
    }

    // Verbose output
    if (is_live){
        cout << "Recalibration was detected from comment lines in bam file:" << endl;
        cout << bam_header_recalibration.size() << " unique blocks of recalibration info detected." << endl;
    }else{
    	cout << "No valid Recalibration was detected in the bam file."<< endl;
    }
    if (num_parsing_errors > 0) {
      cout << "Failed to parse " << num_parsing_errors << " recalibration comment lines." << endl;
    }
}

void TreePhaser::setCalibFromJson(const string &json_model, int threshold)
{
    Json::Value json(Json::objectValue);
    Json::Reader recal_reader;
    if (not recal_reader.parse(json_model, json)) {
      throw std::runtime_error("Failed to parse recalibration file");
    }
    calibModel.InitializeModelFromJson(json, threshold);
}

void TreePhaser::setCAFIEParams(double cf, double ie, double dr)
{
    dpTreephaser->SetModelParameters( cf, ie, dr );
}

void TreePhaser::setStateProgression(bool diagonalStates)
{
    dpTreephaser->SetStateProgression( diagonalStates );
}

bool ApplyCalibration(DPTreephaser *dpTreephaser, const LinearCalibrationModel& calibModel, int calib_x, int calib_y){
	bool calibration_set = false;
	dpTreephaser->DisableRecalibration();
	if( calibModel.is_enabled()){
	    const vector<vector<vector<float> > > * aPtr = calibModel.getAs(calib_x, calib_y);
	    const vector<vector<vector<float> > > * bPtr = calibModel.getBs(calib_x, calib_y);
	    if (aPtr == 0 or bPtr == 0) {
	        std::cerr<< "Error finding recalibration model for x: " << calib_x << " y: " << calib_y << std::endl;
	    }
	    else{
	    	dpTreephaser->SetAsBs(aPtr, bPtr);
	    	calibration_set = true;
	    }
	}
	return calibration_set;
}

bool TreePhaser::applyCalibForXY(int calib_x, int calib_y){
	return ApplyCalibration(dpTreephaser, calibModel, calib_x, calib_y);
}

void TreePhaser::disableCalibration(){
	dpTreephaser->DisableRecalibration();
}

bool TreePhaser::applyCalibForQueryName(const string &qname){
	bool calibration_set = false;
	int well_x = 0, well_y = 0;
	string runid = qname.substr(0, qname.find(":"));
	std::pair <std::multimap<string,pair<int,int> >:: const_iterator, std::multimap<string,pair<int,int> >:: const_iterator> blocks;

	// Disable the previous Calibration
	disableCalibration();

	if (bam_header_recalibration.empty()){
		cerr << "No valid calibration was initialized from a BAM file. Please run this->setCalibFromBamFile(bam_path) first."<<endl;
		return calibration_set;
	}

	// Do the steps as in InitialzeBaseCallers in InputStructures.cpp
	ion_readname_to_xy(qname.c_str(), &well_x, &well_y);
	blocks = block_hash.equal_range(runid);
	int tx = 0, ty = 0;
	for (std::multimap<string,pair<int,int> >:: const_iterator it = blocks.first; it!=blocks.second; ++it) {
		int ax = it->second.first;
		int ay = it->second.second;
		if ((ax<=well_x) && (ay<=well_y)) {
			// potential block including this point because it is less than the point coordinates
			// take the coordinates largest & closest
			if (ax >tx)
				tx = ax;
			if (ay>ty)
				ty = ay;
		}
	}
	string found_key = runid + ".block_X" + to_string(tx) + "_Y" + to_string(ty);
	MultiAB multi_ab;
	// found_key in map to get iterator
	map<string, LinearCalibrationModel>::const_iterator my_calibModel;
	my_calibModel = bam_header_recalibration.find(found_key);
	if (my_calibModel!=bam_header_recalibration.end()){
		my_calibModel->second.getAB(multi_ab, well_x, well_y);
		if (multi_ab.Valid()){
			dpTreephaser->SetAsBs(multi_ab.aPtr, multi_ab.bPtr);
			calibration_set = true;
		}else{
			cerr << "Unable to find the calibration data for the X-Y of the read "<< qname;
		}
	}else{
		cerr << "Unable to find the calibration data for the read "<< qname <<". Please check the BAM file header.";
	}

	return calibration_set;
}

object queryStates( DPTreephaser* dpTreephaser, const LinearCalibrationModel& calibModel, const string& sequence, int maxFlows, int calib_x, int calib_y, bool getStates )
{
    BasecallerRead read;
    read.sequence = std::vector<char>(sequence.begin(), sequence.end());
    if (min(calib_x, calib_y) >= 0){
        ApplyCalibration(dpTreephaser, calibModel, calib_x, calib_y);
    }
    if( getStates ){
        vector< vector<float> > queryStates;
        vector< int > hpLength;
        dpTreephaser->QueryAllStates(read, queryStates, hpLength, maxFlows);
        vector<float> predictions = read.prediction;

        //packing vector of vectors into a list of numpy
        boost::python::list queryStates_list;
        for(auto &row : queryStates ){
            queryStates_list.append(toNumpy(row));
        }

        boost::python::dict ret;
        ret["predictions"]=toNumpy(predictions);
        ret["states"]=queryStates_list;
        ret["hp_len"]=toNumpy(hpLength);
        return object(ret);
    }
    dpTreephaser->Simulate(read, maxFlows);
    vector<float> predictions = read.prediction;
    dict ret;
    ret["predictions"]=toNumpy(predictions);
    return object(ret);
}

object TreePhaser::queryAllStates( const string& sequence, int maxFlows, int calib_x, int calib_y )
{
    return queryStates(dpTreephaser, calibModel, sequence, maxFlows, calib_x, calib_y, true);
}


object TreePhaser::Simulate( const string& sequence, int maxFlows=-1)
{
    return queryStates(dpTreephaser, calibModel, sequence, maxFlows, -1, -1, false);
}

void makeOutput(const BasecallerRead& read, boost::python::dict& output)
{
    output["seq"]=toNumpy(read.sequence);
    output["predicted"]=toNumpy(read.prediction);
    output["norm_additive"]=toNumpy(read.additive_correction);
    output["norm_multipl"]=toNumpy(read.multiplicative_correction);
    std::vector<float> res;
    res.resize(read.prediction.size());
    //res=read.normalized_measurements-read.prediction
    std::transform(read.normalized_measurements.begin(), read.normalized_measurements.end(), read.prediction.begin(), res.begin(), std::minus<float>() );
    output["residual"]=toNumpy(res);
}

boost::python::dict TreePhaser::treephaserSolve(boost::python::object _signal, boost::python::object _keyVec)
{
    (void)_keyVec;

    std::vector<float> signal = fromNumpy<float>(_signal);
    BasecallerRead read;
    int nFlow = signal.size();

    if(flowOrder.num_flows() < nFlow)
        throw std::runtime_error("Flow cycle is shorter than number of flows to solve");

    read.SetData(signal, nFlow);
    dpTreephaser->Solve(read, nFlow);

    boost::python::dict output;
    makeOutput(read,output);

    return output;
}

boost::python::dict TreePhaser::treephaserSWANSolve(boost::python::object _signal, boost::python::object _keyVec)
{
    std::vector<float> signal = fromNumpy<float>(_signal);
    std::vector<int> keyVec = fromNumpy<int>(_keyVec);
    BasecallerRead read;
    int nFlow = signal.size();
    int nKeyFlow = keyVec.size();

    if(flowOrder.num_flows() < nFlow)
        throw std::runtime_error("Flow cycle is shorter than number of flows to solve");

    read.SetDataAndKeyNormalize(signal.data(), nFlow, keyVec.data(), nKeyFlow-1);
    dpTreephaser->NormalizeAndSolve_SWnorm(read, nFlow);

    boost::python::dict output;
    makeOutput(read,output);

    return output;
}


boost::python::dict TreePhaser::treephaserDPSolve(boost::python::object _signal, boost::python::object _keyVec)
{
    std::vector<float> signal = fromNumpy<float>(_signal);
    std::vector<int> keyVec = fromNumpy<int>(_keyVec);
    BasecallerRead read;
    int nFlow = signal.size();
    int nKeyFlow = keyVec.size();

    if(flowOrder.num_flows() < nFlow)
        throw std::runtime_error("Flow cycle is shorter than number of flows to solve");

    read.SetDataAndKeyNormalize(signal.data(), nFlow, keyVec.data(), nKeyFlow-1);
    dpTreephaser->NormalizeAndSolve_GainNorm(read, nFlow);

    boost::python::dict output;
    makeOutput(read,output);

    return output;
}

boost::python::dict TreePhaser::treephaserAdaptiveSolve(boost::python::object _signal, boost::python::object _keyVec)
{
    std::vector<float> signal = fromNumpy<float>(_signal);
    std::vector<int> keyVec = fromNumpy<int>(_keyVec);
    BasecallerRead read;
    int nFlow = signal.size();
    int nKeyFlow = keyVec.size();

    if(flowOrder.num_flows() < nFlow)
        throw std::runtime_error("Flow cycle is shorter than number of flows to solve");

    read.SetDataAndKeyNormalize(signal.data(), nFlow, keyVec.data(), nKeyFlow-1);
    dpTreephaser->NormalizeAndSolve_Adaptive(read, nFlow);

    boost::python::dict output;
    makeOutput(read,output);

    return output;
}

void TreePhaser::treephaserSolveMulti(const string &solverName, boost::python::list readList, boost::python::object _keyVec)
{
    std::function<boost::python::dict(TreePhaser*, boost::python::object,boost::python::object)> solver;

    if(solverName=="treephaser")
        solver = &TreePhaser::treephaserSolve;
    else if( solverName=="treephaserSWAN" )
        solver = &TreePhaser::treephaserSWANSolve;
    else if( solverName=="treephaserDP")
        solver = &TreePhaser::treephaserDPSolve;
    else if( solverName=="treephaserAdaptive" )
        solver = &TreePhaser::treephaserAdaptiveSolve;
    else
        throw std::runtime_error("Solver name can only be: treephaser, treephaserSWAN, treephaserDP or treephaserAdaptive");

    boost::python::stl_input_iterator<boost::python::dict> begin(readList), end;
    for(auto read=begin; read!=end; ++read){
        int row = boost::python::extract<float>((*read)["row"]);
        int col = boost::python::extract<float>((*read)["col"]);
        std::vector<float> phase=fromNumpy<float>((*read)["phase"]);

        dpTreephaser->SetModelParameters(phase[0],phase[1],phase[2]);

        if( calibModel.is_enabled() ){
            const vector<vector<vector<float> > > * aPtr = 0;
            const vector<vector<vector<float> > > * bPtr = 0;
            aPtr = calibModel.getAs(col, row);
            bPtr = calibModel.getBs(col, row);
            dpTreephaser->SetAsBs(aPtr, bPtr);
        }
        else{
            dpTreephaser->SetAsBs(NULL, NULL);
        }
        boost::python::dict output = solver(this,(*read)["meas"],_keyVec);
        read->update(output);
    }
}

boost::python::object seqToFlows( const std::string& sequence, const std::string& flowOrder ){
    npy_intp dims[]={static_cast<long int>(flowOrder.size())};
    handle<> array( PyArray_SimpleNew(1,dims,NPY_INT) );

    std::string::const_iterator base_ptr = sequence.begin();
    for (int flow = 0; flow < (int)flowOrder.size(); ++flow) {
      int* seq = static_cast<int*>(PyArray_GETPTR1(array.get(),flow));
      *seq=0;
      while ( (*base_ptr == flowOrder[flow]) && (base_ptr != sequence.end())) {
        base_ptr++;
        (*seq)++;
      }
      //printf("Flow:%d base_ptr: %c seq: %d",flow,*base_ptr,*seq);
    }
    return object(array);
}

bool getNextAlignment(BamTools::BamAlignment &alignment, BamTools::BamReader &bamReader, const std::map<std::string, int> &groupID, std::vector< BamTools::BamAlignment > &alignmentSample, std::map<std::string, int> &wellIndex, unsigned int nSample, int minRow, int maxRow, int minCol, int maxCol) {
    if(nSample > 0) {
        // We are randomly sampling, so next read should come from the sample that was already taken from the bam file
        if(alignmentSample.size() > 0) {
            alignment = alignmentSample.back();
            alignmentSample.pop_back();
            alignment.BuildCharData();
            return(true);
        } else {
            return(false);
        }
    } else {
        // No random sampling, so we're either returning everything or we're looking for specific read names
        bool storeRead = false;
        while(bamReader.GetNextAlignment(alignment)) {
            if(groupID.size() > 0) {
                std::string thisReadGroupID = "";
                if( !alignment.GetTag("RG", thisReadGroupID) || (groupID.find(thisReadGroupID)==groupID.end()) )
                    continue;
            }

            int thisCol=0,thisRow=0;
            if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &thisRow, &thisCol))
                std::cerr << "Error parsing read name: " << alignment.Name << "\n";

            if( !(thisRow>=minRow && thisRow<maxRow && thisCol>=minCol && thisCol<maxCol) )
                continue;

            if(wellIndex.size() > 0) {
                // We are filtering by position, so check if we should skip or keep the read
                std::stringstream wellIdStream;
                wellIdStream << thisCol << ":" << thisRow;
                std::map<std::string, int>::iterator wellIndexIter;
                wellIndexIter = wellIndex.find(wellIdStream.str());
                if(wellIndexIter != wellIndex.end()) {
                    // If the read ID matches we should keep, unless its a duplicate
                    if(wellIndexIter->second >= 0) {
                        storeRead=true;
                        wellIndexIter->second=-1;
                    } else {
                        std::cerr << "WARNING: found extra instance of readID " << wellIdStream.str() << ", keeping only first\n";
                        continue;
                    }
                } else {
                    // read ID is not one we should keep
                    continue;
                }
            }
            else
                storeRead = true;

            if(storeRead)
                break;
        }
        return(storeRead);
    }
}

bool getTagParanoid(BamTools::BamAlignment &alignment, const std::string &tag, int64_t &value) {
    char tagType = ' ';
    if(alignment.GetTagType(tag, tagType)) {
        switch(tagType) {
        case BamTools::Constants::BAM_TAG_TYPE_INT8: {
            int8_t value_int8 = 0;
            alignment.GetTag(tag, value_int8);
            value = value_int8;
        } break;
        case BamTools::Constants::BAM_TAG_TYPE_UINT8: {
            uint8_t value_uint8 = 0;
            alignment.GetTag(tag, value_uint8);
            value = value_uint8;
        } break;
        case BamTools::Constants::BAM_TAG_TYPE_INT16: {
            int16_t value_int16 = 0;
            alignment.GetTag(tag, value_int16);
            value = value_int16;
        } break;
        case BamTools::Constants::BAM_TAG_TYPE_UINT16: {
            uint16_t value_uint16 = 0;
            alignment.GetTag(tag, value_uint16);
            value = value_uint16;
        } break;
        case BamTools::Constants::BAM_TAG_TYPE_INT32: {
            int32_t value_int32 = 0;
            alignment.GetTag(tag, value_int32);
            value = value_int32;
        } break;
        case BamTools::Constants::BAM_TAG_TYPE_UINT32: {
            uint32_t value_uint32 = 0;
            alignment.GetTag(tag, value_uint32);
            value = value_uint32;
        } break;
        default: {
            alignment.GetTag(tag, value);
        } break;
        }
        return(true);
    } else {
        return(false);
    }
}

void getQuickStats(BamTools::BamReader& bamReader, std::map< std::string, int > &keyLen, unsigned int &nFlowFZ, unsigned int &nFlowZM) {

    if(!bamReader.IsOpen())
        throw std::runtime_error("Bam file not yet open");

    BamTools::SamHeader samHeader = bamReader.GetHeader();
    for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr ) {
        if(itr->HasID())
            keyLen[itr->ID] = itr->HasKeySequence() ? itr->KeySequence.length() : 0;
        if(itr->HasFlowOrder())
            nFlowZM = std::max(nFlowZM,(unsigned int) itr->FlowOrder.length());
    }
    BamTools::BamAlignment alignment;
    std::vector<uint16_t> flowIntFZ;
    while(bamReader.GetNextAlignment(alignment)) {
        if(alignment.GetTag("FZ", flowIntFZ))
            nFlowFZ = flowIntFZ.size();
        break;
    }
    bamReader.Rewind();
    //    if(nFlowFZ==0)
    //        std::cout << "NOTE: bam file has no flow signals in FZ tag: " + bamFile + "\n";
    //    if(nFlowZM==0)
    //        std::cout << "NOTE: bam file has no flow signals in ZM tag: " + bamFile + "\n";
}


//Ported from BamUtils

//this could probably be faster -- maybe with an std::transform
void reverse_comp(std::string& c_dna) {
    for (unsigned int i = 0; i<c_dna.length(); i++) {
        switch (c_dna[i]) {
        case 'A':
            c_dna[i] = 'T';
            break;
        case 'T':
            c_dna[i] = 'A';
            break;
        case 'C':
            c_dna[i] = 'G';
            break;
        case 'G':
            c_dna[i] = 'C';
            break;
        case '-':
            c_dna[i] = '-';
            break;

        default:
            break;
        }
    }
    std::reverse(c_dna.begin(), c_dna.end());

}

void dna( string& qDNA, const vector<BamTools::CigarOp>& cig, const string& md, string& tDNA) {

    int position = 0;
    string seq;
    string::const_iterator qDNA_itr = qDNA.begin();

    for (vector<BamTools::CigarOp>::const_iterator i = cig.begin(); i != cig.end(); ++i) {
        if ( i->Type == 'M') {
            unsigned int count = 0;
            while (qDNA_itr != qDNA.end()) {

                if (count >= i->Length) {
                    break;
                } else {
                    seq += *qDNA_itr;
                    ++qDNA_itr;
                    ++count;
                }
            }
        } else if ((i->Type == 'I') || (i->Type == 'S')) {
            unsigned int count = 0;
            while (qDNA_itr != qDNA.end()) {
                if (count >= i->Length) {
                    break;
                }
                ++qDNA_itr;
                ++count;
            }
            //bool is_error = false;

            //            if (i->Type == 'S') {
            //                soft_clipped_bases += i->Length;
            //                //is_error = true;
            //            }
        }
        position++;
    }

    tDNA.reserve(seq.length());
    int start = 0;
    string::const_iterator md_itr = md.begin();
    std::string num;
    int md_len = 0;
    char cur;

    while (md_itr != md.end()) {

        cur = *md_itr;

        if (std::isdigit(cur)) {
            num+=cur;
            //md_itr.next();
        }
        else {
            if (num.length() > 0) {
                md_len = strtol(num.c_str(),NULL, 10);
                num.clear();

                tDNA += seq.substr(start, md_len);
                start += md_len;
            }
        }

        if (cur == '^') {
            //get nuc
            ++md_itr;
            char nuc = *md_itr;
            while (std::isalpha(nuc)) {
                tDNA += nuc;
                ++md_itr;
                nuc = *md_itr;
            }
            num += nuc; //it's a number now will
            //lose this value if i don't do it here
            //cur = nuc;

        } else if (std::isalpha(cur)) {
            tDNA += cur;
            start++;

        }
        ++md_itr;
    }

    //clean up residual num if there is any
    if (num.length() > 0) {
        md_len = strtol(num.c_str(),NULL, 10);
        num.clear();
        tDNA += seq.substr(start, md_len);
        start += md_len;
    }
}


void padded_alignment(const vector<BamTools::CigarOp>& cig, string& qDNA, string& tDNA,  string& pad_query, string& pad_target, string& pad_match, bool isReversed) {

    int sdna_pos = 0;
    unsigned int tdna_pos = 0;
    pad_target.reserve(tDNA.length());
    pad_query.reserve(tDNA.length());
    pad_match.reserve(tDNA.length());
    string::iterator tdna_itr = tDNA.begin();
    unsigned int tot = 0;
    //find out if the first cigar op could be soft clipped or not
    bool is_three_prime_soft_clipped;


    for (vector<BamTools::CigarOp>::const_iterator i = cig.begin(); i!=cig.end(); ++i) {
        //i.op();		i.len();
        if (isReversed) {
            if (tot > ( cig.size() - 3) ){
                if (i->Type == 'S')
                    is_three_prime_soft_clipped = true;
                else
                    is_three_prime_soft_clipped = false;

            }
        } else {
            if (tot < 2) {
                if (i->Type == 'S')
                    is_three_prime_soft_clipped = true;
                else
                    is_three_prime_soft_clipped = false;

            }
        }

        if (i->Type == 'I' ) {
            pad_target.append(i->Length, '-');

            unsigned int count = 0;

            tdna_itr = qDNA.begin();
            advance(tdna_itr, sdna_pos);

            while (tdna_itr != tDNA.end() ) {
                if (count >= i->Length) {
                    break;
                } else {
                    pad_query += *tdna_itr;
                    ++tdna_itr;
                    //++tdna_pos;
                    ++sdna_pos;
                    ++count;
                }
            }
            pad_match.append(i->Length, '+');
        }
        else if(i->Type == 'D' || i->Type == 'N') {
            pad_target.append( tDNA.substr(tdna_pos, i->Length));
            sdna_pos += i->Length;
            tdna_pos += i->Length;
            pad_query.append(i->Length, '-');
            pad_match.append(i->Length, '-');
        }
        else if(i->Type == 'P') {
            pad_target.append(i->Length, '*');
            pad_query.append(i->Length, '*');
            pad_match.append(i->Length, ' ');
        } else if (i->Type == 'S') {

            //            if (!truncate_soft_clipped) {

            //                    pad_source.append(i->Length, '-');
            //                    pad_match.append(i->Length, '+');
            //                    pad_target.append(i->Length, '+');

            //            }
            //            int count = 0;
            //            while (tdna_itr != tDNA.end()) {
            //                if (count >= i->Length) {
            //                    break;
            //                }
            //                ++tdna_pos;
            //                ++tdna_itr;
            //                ++count;
            //            }
        }

        else if (i->Type == 'H') {
            //nothing for clipped bases
        }else {
            std::string ps, pt, pm;
            ps.reserve(i->Length);
            pm.reserve(i->Length);

            ps = qDNA.substr(sdna_pos,i->Length); //tdna is really qdna

            tdna_itr = tDNA.begin();
            advance(tdna_itr, tdna_pos);

            unsigned int count = 0;

            while (tdna_itr != tDNA.end()) {
                if (count < i->Length) {
                    pt += *tdna_itr;
                } else {
                    break;
                }

                ++tdna_itr;
                ++count;

            }
            for (unsigned int z = 0; z < ps.length(); z++) {
                if (ps[z] == pt[z]) {
                    pad_match += '|';
                } else {
                    pad_match += ' ';
                }
            }//end for loop
            pad_target += pt;
            pad_query += ps;

            sdna_pos += i->Length;
            tdna_pos += i->Length;
            if( tdna_pos >= tDNA.size() )
                break;
        }
        tot++;
    }
    /*
    std::cerr << "pad_source: " << pad_source << std::endl;
    std::cerr << "pad_target: " << pad_target << std::endl;
    std::cerr << "pad_match : " << pad_match << std::endl;
    */
}

std::vector<int> score_alignments(string& pad_source, string& pad_target, string& pad_match ){

    int n_qlen = 0;
    int t_len = 0;
    int t_diff = 0;
    int match_base = 0;
    int num_slop = 0;

    int consecutive_error = 0;

    //using namespace std;
    for (int i = 0; (unsigned int)i < pad_source.length(); i++) {
        //std::cerr << " i: " << i << " n_qlen: " << n_qlen << " t_len: " << t_len << " t_diff: " << t_diff << std::endl;
        if (pad_source[i] != '-') {
            t_len = t_len + 1;
        }

        if (pad_match[i] != '|') {
            t_diff = t_diff + 1;

            if (i > 0 && pad_match[i-1] != '|' && ( ( pad_target[i] == pad_target[i - 1] ) || pad_match[i] == '-' ) ) {
                consecutive_error = consecutive_error + 1;
            } else {
                consecutive_error = 1;
            }
        } else {
            consecutive_error = 0;
            match_base = match_base + 1;
        }
        if (pad_target[i] != '-') {
            n_qlen = n_qlen + 1;
        }
    }


    //get qual vals from  bam_record
    std::vector<double> Q;

    //setting acceptable error rates for each q score, defaults are
    //7,10,17,20,47
    //phred_val == 7
    Q.push_back(0.2);
    //phred_val == 10
    Q.push_back(0.1);
    //phred_val == 17
    Q.push_back(0.02);
    //phred_val == 20
    Q.push_back(0.01);
    //phred_val == 47
    Q.push_back(0.00002);

    std::vector<int> q_len_vec(Q.size(), 0);

    int prev_t_diff = 0;
    int prev_loc_len = 0;
    int i = pad_source.length() - 1;

    for (std::vector<std::string>::size_type k =0; k < Q.size(); k++) {
        int loc_len = n_qlen;
        int loc_err = t_diff;
        if (k > 0) {
            loc_len = prev_loc_len;
            loc_err = prev_t_diff;
        }

        while ((loc_len > 0) && (static_cast<int>(i) >= num_slop) && i > 0) {

            if (q_len_vec[k] == 0 && (((loc_err / static_cast<double>(loc_len))) <= Q[k]) /*&& (equivalent_length(loc_len) != 0)*/) {

                q_len_vec[k] = loc_len;

                prev_t_diff = loc_err;
                prev_loc_len = loc_len;
                break;
            }
            if (pad_match[i] != '|') {
                loc_err--;
            }
            if (pad_target[i] != '-') {

                loc_len--;
            }
            i--;
        }
    }
    return q_len_vec;
}


void PyRawDat::loadImg( const std::string& fname ){
    if(fname!=curr_fname){
        curr_fname = fname;

        img.SetImgLoadImmediate(true);
        if( !img.LoadRaw( fname.c_str() ) )
            throw std::runtime_error("Can't open dat file.");

        if( normalize )
            img.SetMeanOfFramesToZero(norm_start, norm_end);
    }
}

void PyRawDat::SetMeanOfFramesToZero( int norm_start /*=1*/, int norm_end /*=3*/ )
{
    img.SetMeanOfFramesToZero(norm_start, norm_end);
}

void PyRawDat::ApplyXTChannelCorrection( void )
{
    const char *rawDir = dirname ( ( char * ) curr_fname.c_str() );
    ImageTransformer::CalibrateChannelXTCorrection ( rawDir,"lsrowimage.dat" );
    ImageTransformer::XTChannelCorrect ( (RawImage *)img.GetImage(),get_current_dir_name() );
}

void PyRawDat::ApplyRowNoiseCorrection( int thumbnail /*=1*/ )
{
    CorrNoiseCorrector rnc;
    rnc.CorrectCorrNoise((RawImage *)img.GetImage(),1,thumbnail, true);
}

void PyRawDat::ApplyPairPixelXtalkCorrection( float pair_xtalk_fraction )
{
    PairPixelXtalkCorrector xtalkCorrector;
    xtalkCorrector.Correct((RawImage *)img.GetImage(), pair_xtalk_fraction);
}

boost::python::object PyRawDat::LoadSlice(int row, int col, int h, int w ){

    h = std::min( h, img.GetRows()-row );
    w = std::min( w, img.GetCols()-col );
    int frames = (uncompress)? img.GetUnCompFrames() : img.GetFrames();

    npy_intp dims[]={h, w, frames};
    handle<> array( PyArray_SimpleNew(3,dims,NPY_FLOAT) );

    float* d = new float[frames];

    for( int r = row; r < row+h; ++r ){
        for( int c = col; c < col+w; ++c ){
            if( uncompress )   img.GetUncompressedTrace(d, frames, c, r );

            for( int f = 0; f < frames; ++f ){
                float* data = (float*)PyArray_GETPTR3(array.get(),r-row,c-col,f);
                *data = (uncompress)? d[f] : img.At(r,c,f);
            }
        }
    }
    delete[] d;
    return object(array);
}

boost::python::object PyRawDat::LoadWells( const boost::python::tuple& pos ){

    int length = boost::python::len( pos );
    int frames = (uncompress)? img.GetUnCompFrames() : img.GetFrames();

    npy_intp dims[]={length, frames};
    handle<> array( PyArray_SimpleNew(2,dims,NPY_FLOAT) );
    float* d = new float[frames];

    for(int i = 0; i < length; ++i){
        boost::python::tuple t = boost::python::extract<boost::python::tuple>(pos[i]);
        int col = boost::python::extract<int>(t[0]);
        int row = boost::python::extract<int>(t[1]);
        int frames = (uncompress)? img.GetUnCompFrames() : img.GetFrames();
        if( uncompress )   img.GetUncompressedTrace(d, frames, col, row );

        for( int f = 0; f < frames; ++f ){
            float* data = (float*)PyArray_GETPTR2(array.get(), i, f);
            *data = (uncompress)? d[f] : img.At(row,col,f);
        }
    }
    delete[] d;
    return object(array);
}

boost::python::object PyWells::LoadWells(int row , int col, int h, int w){
    IonErr::SetThrowStatus(true);
    RawWells wells("",fname.c_str());
    std::vector<int32_t> col_list;
    std::vector<int32_t> row_list;

    for( int r = row; r < row+h; ++r ){
        for( int c = col; c < col+w; ++c ){
            col_list.push_back(c);
            row_list.push_back(r);
        }
    }

    wells.OpenForRead();

    h = std::min( h, (int)wells.NumRows()-row );
    w = std::min( w, (int)wells.NumCols()-col );
    int flows = wells.NumFlows();
    npy_intp dims[]={h, w, flows};
    handle<> array( PyArray_SimpleNew(3,dims,NPY_FLOAT) );

    for( int r = row; r < row+h; ++r ){
        for( int c = col; c < col+w; ++c ){
            for( int f = 0; f < flows; ++f ){
                float* data = (float*)PyArray_GETPTR3(array.get(),r-row,c-col,f);
                *data = wells.At(r,c,f);
            }
        }
    }

    wells.Close();
    return object(array);
}

boost::python::object PyWells::LoadWellsFlow(int row , int col, int h, int w, int flow){

    RawWells wells("",fname.c_str());
//    std::vector<int32_t> col_list;
//    std::vector<int32_t> row_list;

//    for( int r = row; r < row+h; ++r ){
//        for( int c = col; c < col+w; ++c ){
//            col_list.push_back(c);
//            row_list.push_back(r);
//        }
//    }

    wells.OpenForRead();

    h = std::min( h, (int)wells.NumRows()-row );
    w = std::min( w, (int)wells.NumCols()-col );
    int flows = wells.NumFlows();
    if( flow > flows )
        throw std::runtime_error("flow value too large");

    npy_intp dims[]={h, w};
    handle<> array( PyArray_SimpleNew(2,dims,NPY_FLOAT) );

    for( int r = row; r < row+h; ++r ){
        for( int c = col; c < col+w; ++c ){
            float* data = (float*)PyArray_GETPTR2(array.get(),r-row,c-col);
            *data = wells.At(r,c,flow);
        }
    }

    wells.Close();
    return object(array);
}


boost::python::object LoadBfMaskByType( std::string fname, MaskType maskType ){
    uint32_t nRow, nCol;

    FILE *fp = fopen( fname.c_str(), "rb");
    if( !fp )
        throw std::runtime_error("Can't open file!");

    if ((fread (&nRow, sizeof(uint32_t), 1, fp )) != 1)
        throw std::runtime_error("Can't read nRow!");
    if ((fread (&nCol, sizeof(uint32_t), 1, fp )) != 1)
        throw std::runtime_error( "Can't read nCol!");

    //int64_t dataSkipBytes = sizeof(uint16_t);
    //int64_t offset;

    npy_intp dims[]={nRow, nCol};
    handle<> array( PyArray_SimpleNew(2,dims,NPY_BOOL) );

    uint16_t *mask = new uint16_t[nRow*nCol];
    if( fread(mask, sizeof(uint16_t), nRow*nCol, fp) != nRow*nCol ){
        delete[] mask;
        fclose(fp);
        throw std::runtime_error("Can't read mask from file!");
    }
    for( size_t r=0; r<nRow; ++r ){
        for( size_t c = 0; c < nCol; ++c ){
            bool* data = (bool*)PyArray_GETPTR2(array.get(),r,c);
            *data = (*(mask+r*nCol+c) & maskType)>0;
        }
    }
    delete[] mask;
    fclose(fp);
    return object(array);
}

template<typename T> void appendData( boost::python::dict& data, const std::string& key, const T& value ){
    data[key] = value;
}



void PyBam::Open( std::string _fname ){
    fname = _fname;
    numRecords = 0;

    if(!bamReader.Open(fname))
        throw std::runtime_error( std::string("Can't open bam file: ")+fname );

    //open/create index by default
    if( !bamReader.LocateIndex()){
        bamReader.CreateIndex();
        bamReader.LocateIndex();
    }

    boost::python::dict brec;
    brec = ReadBamHeader();
    appendRecord(brec);

    BamTools::RefVector refVector = bamReader.GetReferenceData();
    for(BamTools::RefVector::iterator rv_it = refVector.begin(); rv_it != refVector.end(); ++rv_it){
        refNames.append(rv_it->RefName);
    }
    //GetNumRecords();
}

void PyBam::SetSampleSize( int nSample ){
    sample_size = nSample;
    ReservoirSample< BamTools::BamAlignment > sample;

    if( !bamReader.IsOpen() )
        throw std::runtime_error("BAM file is not open");

    if(nSample > 0) {
        int randomSeed = clock();
        sample.Init(nSample,randomSeed); //TODO: replace the non-random seed
    }

    BamTools::BamAlignment alignment;
    bamReader.Rewind();
    while(bamReader.GetNextAlignmentCore(alignment)) {
//        if(haveWantedGroups) {
//            std::string thisReadGroupID = "";
//            alignment.BuildCharData();
//            alignment.GetTag("RG", thisReadGroupID);
//            if( !alignment.GetTag("RG", thisReadGroupID) || (my_cache.groupID.find(thisReadGroupID)==my_cache.groupID.end()) )
//                continue;
//        }
//        nRead++;
        if(nSample > 0){
            int thisCol = 0;
            int thisRow = 0;

            if( maxRow!=MAX_INT || maxCol!=MAX_INT ){
                //Unfortunately we need read name if we are to restrict by position on the chip. Slow, but inevitable.
                alignment.BuildCharData();
                if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &thisRow, &thisCol)){
                    printf("Error parsing read name: %s\n", alignment.Name.c_str());
                    continue;
                }

                if( thisRow>=minRow && thisRow<maxRow && thisCol>=minCol && thisCol<maxCol )
                    sample.Add(alignment);
            }
            else{
                sample.Add(alignment);
            }
        }
    }
    bamReader.Rewind();
    if(nSample > 0){
        sample.Finished();
        alignmentSample = sample.GetData();
        //fill in the character data for the sampled reads. Much faster than getting alignement data for all reads.
        for(std::vector< BamTools::BamAlignment >::iterator it = alignmentSample.begin(); it!=alignmentSample.end(); ++it){
            (*it).BuildCharData();
        }
    }
}

int PyBam::GetNumRecords(void){
    if( numRecords>0 )
        return numRecords;

    if( !bamReader.IsOpen() )
        throw std::runtime_error("BAM file is not open");

    BamTools::BamAlignment alignment;
    for( ; bamReader.GetNextAlignmentCore( alignment ); numRecords++ );
    bamReader.Rewind();

    return numRecords;
}

boost::python::dict PyBam::ReadBamHeader( void ){
    if( !bamReader.IsOpen() )
        throw std::runtime_error("BAM file is not open");

    header.clear();
    BamTools::SamHeader samHeader = bamReader.GetHeader();
    for (BamTools::SamReadGroupIterator itr = samHeader.ReadGroups.Begin(); itr != samHeader.ReadGroups.End(); ++itr ) {
        appendToHeader("ID",itr->HasID(),itr->ID);
        appendToHeader("FlowOrder",itr->HasFlowOrder(),itr->FlowOrder);
        appendToHeader("KeySequence",itr->HasKeySequence(),itr->KeySequence);
        appendToHeader("Description",itr->HasDescription(),itr->Description);
        appendToHeader("Library",itr->HasLibrary(),itr->Library);
        appendToHeader("PlatformUnit",itr->HasPlatformUnit(),itr->PlatformUnit);
        appendToHeader("PredictedInsertSize",itr->HasPredictedInsertSize(),itr->PredictedInsertSize);
        appendToHeader("ProductionDate",itr->HasProductionDate(),itr->ProductionDate);
        appendToHeader("Program",itr->HasProgram(),itr->Program);
        appendToHeader("Sample",itr->HasSample(),itr->Sample);
        appendToHeader("SequencingCenter",itr->HasSequencingCenter(),itr->SequencingCenter);
        appendToHeader("SequencingTechnology",itr->HasSequencingTechnology(),itr->SequencingTechnology);
        flow_order_by_read_group[itr->ID] = itr->FlowOrder;
        key_seq_by_read_group[itr->ID] = itr->KeySequence;
        string runid = itr->ID.substr(0, itr->ID.find("."));
        pair< map<string, TreePhaser>::iterator, bool> insert_new_tp;
        insert_new_tp = treephaser_by_runid.insert(pair<string, TreePhaser>{runid, TreePhaser(itr->FlowOrder)});
        if (insert_new_tp.second){
        	insert_new_tp.first->second.setCalibFromBamFile(fname);
        }
    }
    return header;
}


void PyBam::appendToHeader(const string &key, bool has_key, string value){
    if( ! header.has_key(key) )
        header[key] = boost::python::list();

    boost::python::list l = extract<boost::python::list> (header[key]);
    if( has_key )
        l.append(value);
    else
        l.append("");
}

void PyBam::appendRecord(const dict &rec){
    boost::python::list iterkeys = (boost::python::list) rec.iterkeys();
    for( int i = 0; i < boost::python::len(iterkeys); ++i ){
        std::string key = boost::python::extract<std::string>(iterkeys[i]);

        if( ! data.has_key(key) )
            data[key] = boost::python::list();

        boost::python::list l = extract<boost::python::list> (data[key]);
        l.append(rec[key]);
    }
}

bool PyBam::getNextRead( boost::python::dict& brec ){

    BamTools::BamAlignment alignment;
    bool haveMappingData;
    std::map<std::string, int> groupID;
    std::map<std::string, int> wellIndex;

    if( !bamReader.IsOpen() )
        throw std::runtime_error("BAM file is not open");

    if( !getNextAlignment(alignment,bamReader,groupID,alignmentSample,wellIndex,sample_size, minRow, maxRow, minCol, maxCol ) )
        return false;


    int thisCol = 0;
    int thisRow = 0;
    if(1 != ion_readname_to_rowcol(alignment.Name.c_str(), &thisRow, &thisCol))
        throw std::runtime_error("Error parsing read name");

    std::string readGroup = "";
    alignment.GetTag("RG", readGroup);

    appendData(brec, "keySeq", key_seq_by_read_group[readGroup] );

    // Store values that will be returned
    appendData( brec, "id", alignment.Name);
    appendData( brec, "readGroup", readGroup);
    appendData( brec, "col", thisCol);
    appendData( brec, "row", thisRow);

//    std::map<std::string, int>::iterator keyLenIter;
//    keyLenIter = keyLen.find(readGroup);
//    appendData( brec, "AdapterLeft", (keyLenIter != keyLen.end()) ? keyLenIter->second : 0);

//    int64_t clipAdapterRight = 0;
//    getTagParanoid(alignment,"ZA",clipAdapterRight);
//    appendData( brec, "AdapterRight", clipAdapterRight);

    int64_t adapterOverlap = 0;
    getTagParanoid(alignment,"ZB",adapterOverlap);
    appendData( brec, "adapterOverlap", adapterOverlap);

    int64_t flowClipLeft = 0;
    getTagParanoid(alignment,"ZF",flowClipLeft);
    appendData( brec, "flowClipLeft", flowClipLeft);

    int64_t flowClipRight = 0;
    getTagParanoid(alignment,"ZG",flowClipRight);
    appendData( brec, "flowClipRight", flowClipRight);

    std::vector<uint16_t> flowInt;

    if(alignment.GetTag("FZ", flowInt)){
        appendData( brec, "flow",toNumpy(flowInt));
    }

    std::vector<int32_t> zcTag;

    if(alignment.GetTag("ZC", zcTag)){
        appendData( brec, "adapterTag",toNumpy(zcTag));
    }

    std::vector<int16_t> flowMeasured; // round(256*val), signed
    if(alignment.GetTag("ZM", flowMeasured)){
        appendData( brec, "meas",toNumpy(flowMeasured));
    }

    // experimental tag for Project Razor: "phase" values
    std::vector<float> flowPhase;
    if(alignment.GetTag("ZP", flowPhase)){
        appendData( brec, "phase",toNumpy(flowPhase));
    }

    std::vector<int16_t>   additiveCrrection;
    if(alignment.GetTag("Ya", additiveCrrection)){
        appendData( brec, "AdditiveCorrection",toNumpy(additiveCrrection));
    }

    std::vector<int16_t>   multiplicativeCorrection;
    if(alignment.GetTag("Yb", multiplicativeCorrection)){
        appendData( brec, "MultiplicativeCorrection",toNumpy(multiplicativeCorrection));
    }

    std::vector<int16_t>   rawMeasurements;
    if(alignment.GetTag("Yw", rawMeasurements)){
        appendData( brec, "RawMeasurements",toNumpy(rawMeasurements));
    }

    std::vector<int16_t>   calibratedMeasurements;
    if(alignment.GetTag("Yx", calibratedMeasurements)){
        appendData( brec, "calibratedMeasurements",toNumpy(calibratedMeasurements));
    }

    // limit scope of loop as we have too many "loop" variables named i running around
    unsigned int nBases = alignment.QueryBases.length();
    appendData( brec, "fullLength", nBases);
    appendData( brec, "base", alignment.QueryBases);

    npy_intp dims[]={nBases};
    handle<> array( PyArray_SimpleNew(1,dims,NPY_SHORT) );

    for(unsigned int i=0; i<nBases; i++) {
        short* data = (short*)PyArray_GETPTR1(array.get(),i);
        *data = ((short) alignment.Qualities[i]) - 33;
        //TODO - fill in proper flowindex info
        //out_flowIndex(nReadsFromBam,i)  = 0;
    }
    appendData( brec, "qual",array);

    if(alignment.IsMapped()) {
        haveMappingData=true;
        appendData( brec, "aligned_flag", alignment.AlignmentFlag);
        appendData( brec, "aligned_base", alignment.AlignedBases);
        appendData( brec, "aligned_refid", alignment.RefID);
        appendData( brec, "aligned_pos", alignment.Position);
        appendData( brec, "aligned_mapq", alignment.MapQuality);
        appendData( brec, "aligned_bin", alignment.Bin);


        string md;
        string tseq_bases;
        string qseq_bases;
        string pretty_tseq;
        string pretty_qseq;
        string pretty_aln;
        unsigned int left_sc, right_sc;

        alignment.GetTag("MD", md);

        RetrieveBaseAlignment(alignment.QueryBases, alignment.CigarData, md,
                              tseq_bases, qseq_bases, pretty_tseq, pretty_qseq, pretty_aln, left_sc, right_sc);

        //    printf("Base Q prior to reverse: %s\n", qseq_bases.c_str());
        if (alignment.IsReverseStrand()) {
            RevComplementInPlace(qseq_bases);
            RevComplementInPlace(tseq_bases);
            RevComplementInPlace(pretty_qseq);
            RevComplementInPlace(pretty_aln);
            RevComplementInPlace(pretty_tseq);
        }

        std::vector<int> qlen = score_alignments(pretty_qseq, pretty_tseq, pretty_aln );

        vector<int>     qseq;         // The query or read sequence in the alignment, including gaps.
        vector<int>     tseq;         // The target or reference sequence in the alignment, including gaps.
        vector<int>     aln_flow_idx; // The aligned flow index including gaps as -1

        vector<char>    aln;          // The alignment string
        vector<char>    flowOrder;    // The flow order for the alignment, including deleted reference bases.

        const string&     main_flow_order = flow_order_by_read_group[readGroup];

        //vector<uint16_t>  fz_tag;
        //fz_tag.resize(main_flow_order.size());

        if( flowAlign ){
            std::string tseq_bases_plus_key = key_seq_by_read_group[readGroup] + tseq_bases;
            std::string qseq_bases_plus_key = key_seq_by_read_group[readGroup] + qseq_bases;

            PerformFlowAlignment(tseq_bases_plus_key, qseq_bases_plus_key, main_flow_order, 0,
                                 flowOrder, qseq, tseq, aln_flow_idx, aln, /*debug*/ false);

            appendData( brec, "qseq", toNumpy(qseq) );
            appendData( brec, "tseq", toNumpy(tseq) );
            appendData( brec, "aln_flow_idx", toNumpy(aln_flow_idx) );
        }

        appendData( brec, "tDNA", pretty_tseq);
        appendData( brec, "qDNA", pretty_qseq);
        appendData( brec, "match", pretty_aln);

        appendData( brec, "qseq_bases", qseq_bases);
        appendData( brec, "tseq_bases", tseq_bases);

        appendData( brec, "q7Len", qlen[0]);
        appendData( brec, "q10Len", qlen[1]);
        appendData( brec, "q17Len", qlen[2]);
        appendData( brec, "q20Len", qlen[3]);
        appendData( brec, "q47Len", qlen[4]);

        std::ostringstream oss,oss1;
        std::copy(aln.begin(), aln.end(), std::ostream_iterator<char>(oss));
        appendData( brec, "aln", oss.str() );
        std::copy(flowOrder.begin(), flowOrder.end(), std::ostream_iterator<char>(oss1));
        appendData( brec, "flowOrder", oss1.str() );
    }
    return true;
}

boost::python::dict PyBam::next( void ){
    boost::python::dict brec;
    bool r = getNextRead( brec );
    if( r == false ){
        PyErr_SetString(PyExc_StopIteration, "No more data.");
        boost::python::throw_error_already_set();
    }
    return brec;
}

boost::python::dict PyBam::ReadBam( void ){
    boost::python::dict brec;

    while( getNextRead( brec ) ) {
        appendRecord(brec);
    }
    bamReader.Rewind();
    return data;
}

void PyBam::SimulateCafie( boost::python::dict& read )
{
    object temp = read["phase"];
    float* phaseParams =  static_cast<float*>(PyArray_DATA(temp.ptr()));

    object measuredIntensity = read["meas"];
    unsigned int nFlow = PyArray_SIZE( measuredIntensity.ptr() );

    std::string groupId = boost::python::extract<std::string>(read["readGroup"]);
    std::string query_name = boost::python::extract<std::string>(read["id"]);
    std::string runid = query_name.substr(0, query_name.find(":"));
    //printf("cf: %g, meas: %d, groupId: %s,  nFlow: %d, flowOrder: %s \n", phaseParams[0], measuredIntensity[0], groupId.c_str(), nFlow, flow_order.c_str());

    map<string, TreePhaser>::iterator find_dpTreephaser = treephaser_by_runid.find(runid);
    if (find_dpTreephaser == treephaser_by_runid.end()){
    	cerr << "No RUNID was found in the BAM header for the read "<< query_name<<endl;
    	return;
    }
    if (suppress_recalibration){
    	find_dpTreephaser->second.disableCalibration();
    }else{
    	find_dpTreephaser->second.applyCalibForQueryName(query_name);
    }
    find_dpTreephaser->second.setCAFIEParams(phaseParams[0], phaseParams[1], phaseParams[2]);

    // Iterate over all reads
    BasecallerRead tseq_read, qseq_read;
    std::string sequence = key_seq_by_read_group[groupId] + static_cast<std::string>( boost::python::extract<std::string>(read["tseq_bases"]) );
    std::copy( sequence.begin(), sequence.end(), std::back_inserter(tseq_read.sequence) );

    sequence = key_seq_by_read_group[groupId] + static_cast<std::string>( boost::python::extract<std::string>(read["qseq_bases"]) );
    std::copy( sequence.begin(), sequence.end(), std::back_inserter(qseq_read.sequence) );
    find_dpTreephaser->second.Simulate_BasecallerRead( tseq_read, nFlow );
    find_dpTreephaser->second.Simulate_BasecallerRead( qseq_read, nFlow );

    npy_intp dims[]={(npy_intp)tseq_read.prediction.size()};
    handle<> tseq_read_out( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* tseq_read_out_data = (float*)PyArray_DATA( tseq_read_out.get() );
    std::copy(tseq_read.prediction.begin(), tseq_read.prediction.end(), tseq_read_out_data);

    dims[0]=qseq_read.prediction.size();
    handle<> qseq_read_out( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* qseq_read_out_data = (float*)PyArray_DATA( qseq_read_out.get() );
    std::copy(qseq_read.prediction.begin(), qseq_read.prediction.end(), qseq_read_out_data);

    read["tseq_read"] = tseq_read_out;
    read["qseq_read"] = qseq_read_out;
}

void PyBam::PhaseCorrect( boost::python::dict& read, object keyFlow )
{
    object temp = read["phase"];
    float* phaseParams =  static_cast<float*>(PyArray_DATA(temp.ptr()));

    object measuredIntensity = read["meas"];
    unsigned int nFlow = PyArray_SIZE( measuredIntensity.ptr() );

    std::string groupId = boost::python::extract<std::string>(read["readGroup"]);

    std::string query_name = boost::python::extract<std::string>(read["id"]);
    std::string runid = query_name.substr(0, query_name.find(":"));

    map<string, TreePhaser>::iterator find_dpTreephaser = treephaser_by_runid.find(runid);
    if (find_dpTreephaser == treephaser_by_runid.end()){
    	cerr << "No RUNID was found in the BAM header for the read "<< query_name<<endl;
    	return;
    }
    if (suppress_recalibration){
    	find_dpTreephaser->second.disableCalibration();
    }else{
    	find_dpTreephaser->second.applyCalibForQueryName(query_name);
    }
    find_dpTreephaser->second.setCAFIEParams(phaseParams[0], phaseParams[1], phaseParams[2]);
    int nKeyFlow = PyArray_SIZE(keyFlow.ptr());
    vector <int> keyVec(nKeyFlow);
    for(int iFlow=0; iFlow < nKeyFlow; iFlow++)
      keyVec[iFlow] = *(int*)(PyArray_GETPTR1(keyFlow.ptr(), iFlow));

    // Iterate over all reads
    BasecallerRead basecaller_read;
    vector <float> sigVec(nFlow);
    for(unsigned int iFlow=0; iFlow < nFlow; iFlow++)
        sigVec[iFlow] = *(int16_t*)PyArray_GETPTR1( measuredIntensity.ptr(), iFlow );

    basecaller_read.SetDataAndKeyNormalize(&(sigVec[0]), (int)nFlow, &(keyVec[0]), nKeyFlow-1);
    find_dpTreephaser->second.NormalizeAndSolve_SWnorm_BasecallerRead(basecaller_read, nFlow);

    npy_intp dims[]={nFlow};
    handle<> predicted_out( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* predicted_out_data = (float*)PyArray_DATA( predicted_out.get() );
    handle<> residual_out( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* residual_out_data = (float*)PyArray_DATA( residual_out.get() );

    for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
      predicted_out_data[iFlow] = basecaller_read.prediction[iFlow];
      residual_out_data[iFlow]  = basecaller_read.normalized_measurements[iFlow] - basecaller_read.prediction[iFlow];
    }

    read["predicted"] = predicted_out;
    read["residual"] = residual_out;
}

PyBam::PyBam(string _fname) : numRecords(0), sample_size(0), flowAlign(1)
{
    SetChipRegion(0,MAX_INT,0,MAX_INT);
    Open(_fname);
}



float PyBkgModel::AdjustEmptyToBeadRatio(float etbR, float NucModifyRatio, float RatioDrift, int flow, bool fitTauE)
{
    if (!fitTauE)
      // if_use_obsolete_etbR_equation==true, therefore Copy(=1.0) and phi(=0.6) are not used
      return( xAdjustEmptyToBeadRatioForFlow(etbR, 0., 1.0, 0.6, NucModifyRatio, RatioDrift, flow, true) );
    return( xAdjustEmptyToBeadRatioForFlowWithAdjR(etbR,NucModifyRatio,RatioDrift,flow) );
}

float PyBkgModel::TauBFromLinearModel( float etbR, float tauR_m, float tauR_o, float minTauB, float maxTauB ){
    return xComputeTauBfromEmptyUsingRegionLinearModel(tauR_m, tauR_o, etbR, minTauB, maxTauB);
}

boost::python::object PyBkgModel::IntegrateRedFromObservedTotalTracePy ( object purple_obs, object blue_hydrogen,  object deltaFrame, float tauB, float etbR)
{
    PyObject* purpleH = purple_obs.ptr();
    PyObject* blueH = blue_hydrogen.ptr();
    PyObject* deltaFrameV = deltaFrame.ptr();

    int szPurpleH = PyArray_Size( purpleH );
    int szBlueH = PyArray_Size( blueH );
    int szDeltaFrame = PyArray_Size( deltaFrameV );

    //printf("szPurpleH:%d, szBlueH:%d, szDeltaFrame:%d\n", szPurpleH, szBlueH, szDeltaFrame);

    if( szPurpleH != szBlueH || szPurpleH != szDeltaFrame || szPurpleH == 0 )
        throw std::runtime_error("Incompatible input array sizes");

    int len = szPurpleH;

    //printf("Type: purpleH:%d, blueH:%d, deltaFrameV:%d\n", PyArray_TYPE(purpleH), PyArray_TYPE(blueH), PyArray_TYPE(deltaFrameV));

    if( PyArray_TYPE(purpleH)!=NPY_FLOAT ||
            PyArray_TYPE(blueH)!=NPY_FLOAT ||
            PyArray_TYPE(deltaFrameV)!=NPY_FLOAT )
        throw std::runtime_error("Expect floating point data");

    float* purpleH_data = (float*)PyArray_DATA( purpleH );
    float* blueH_data = (float*)PyArray_DATA( blueH );
    float* deltaFrame_data = (float*)PyArray_DATA( deltaFrameV );

    npy_intp dims[]={szDeltaFrame};
    handle<> out_array( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* out_data = (float*)PyArray_DATA( out_array.get() );
    MathModel::IntegrateRedFromObservedTotalTrace ( out_data, purpleH_data, blueH_data, len, deltaFrame_data, tauB, etbR);

    return object(out_array);
}

boost::python::object PyBkgModel::BlueTracePy ( object blue_hydrogen,  object deltaFrame, float tauB, float etbR)
{
    PyObject* blueH = blue_hydrogen.ptr();
    PyObject* deltaFrameV = deltaFrame.ptr();

    int szBlueH = PyArray_Size( blueH );
    int szDeltaFrame = PyArray_Size( deltaFrameV );

    int len = szBlueH;

    if( PyArray_TYPE(blueH)!=NPY_FLOAT ||
            PyArray_TYPE(deltaFrameV)!=NPY_FLOAT )
        throw std::runtime_error("Expect floating point data");

    float* blueH_data = (float*)PyArray_DATA( blueH );
    float* deltaFrame_data = (float*)PyArray_DATA( deltaFrameV );

    npy_intp dims[]={szDeltaFrame};
    handle<> out_array( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* out_data = (float*)PyArray_DATA( out_array.get() );
    MathModel::BlueSolveBackgroundTrace ( out_data, blueH_data, len, deltaFrame_data, tauB, etbR);
    return object(out_array);
}

boost::python::object PyBkgModel::PurpleTracePy ( object blue_hydrogen,  object red_hydrogen, object deltaFrame, float tauB, float etbR)
{
    PyObject* blueH = blue_hydrogen.ptr();
    PyObject* redH = red_hydrogen.ptr();
    PyObject* deltaFrameV = deltaFrame.ptr();

    int szBlueH = PyArray_Size( blueH );
    int szRedH = PyArray_Size( redH );
    int szDeltaFrame = PyArray_Size( deltaFrameV );

    int len = szBlueH;
    if( szBlueH != szRedH || szBlueH == 0 )
        throw std::runtime_error("Incompatible input array sizes");

    if( PyArray_TYPE(blueH)!=NPY_FLOAT || PyArray_TYPE(redH)!=NPY_FLOAT ||
            PyArray_TYPE(deltaFrameV)!=NPY_FLOAT )
        throw std::runtime_error("Expect floating point data");

    float* blueH_data = (float*)PyArray_DATA( blueH );
    float* redH_data = (float*)PyArray_DATA( redH );
    float* deltaFrame_data = (float*)PyArray_DATA( deltaFrameV );

    npy_intp dims[]={szDeltaFrame};
    handle<> out_array( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* out_data = (float*)PyArray_DATA( out_array.get() );
    MathModel::PurpleSolveTotalTrace ( out_data, blueH_data, redH_data, len, deltaFrame_data, tauB, etbR);
    return object(out_array);
}


boost::python::object PyBkgModel::CalculateNucRisePy ( object timeFrame, int sub_steps, float C, float t_mid_nuc, float sigma, float nuc_span)
{
    PyObject* timeFrameH = timeFrame.ptr();
    int len = PyArray_Size( timeFrameH );

    float* timeFrame_data = (float*) PyArray_DATA( timeFrameH );

    npy_intp dims[]={len};
    handle<> out_array( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* out_data = (float*)PyArray_DATA( out_array.get() );

    MathModel::SigmaRiseFunction(out_data, len, timeFrame_data, sub_steps, C, t_mid_nuc, sigma, nuc_span, true );
    return object(out_array);
}


boost::python::object PyBkgModel::GenerateRedHydrogensFromNucRisePy ( object nucRise, object deltaFrame, int sub_steps, int my_start_index, float C, float Amplitude,
                                                                      float Copies, float krate, float kmax, float diffusion, int hydrogenModelType )
{
    PyObject* nucRiseH = nucRise.ptr();
    PyObject* deltaFrameH = deltaFrame.ptr();
    int szNucRise = PyArray_Size( nucRiseH );
    int len = PyArray_Size( deltaFrameH );

    if( PyArray_TYPE(nucRiseH)!=NPY_FLOAT ||
            PyArray_TYPE(deltaFrameH)!=NPY_FLOAT )
        throw std::runtime_error("Expect floating point data");

    float* deltaFrame_data = (float*) PyArray_DATA( deltaFrameH );
    float* nucRise_data = (float*) PyArray_DATA( nucRiseH );

    if( szNucRise != len || len == 0 )
        throw std::runtime_error("Incompatible input array sizes");

    if( krate==0 )
        throw std::runtime_error("krate is null");

    npy_intp dims[]={len};
    handle<> out_array( PyArray_SimpleNew(1,dims,NPY_FLOAT) );
    float* out_data = (float*)PyArray_DATA( out_array.get() );

    float molecules_to_micromolar_conv = 0.000062;
    PoissonCDFApproxMemo my_math;
    my_math.Allocate(MAX_HPXLEN+1,512,0.05);
    my_math.GenerateValues();

    MathModel::ComputeCumulativeIncorporationHydrogens(out_data, len, deltaFrame_data, nucRise_data, sub_steps,my_start_index,
                                            C, Amplitude, Copies, krate, kmax, diffusion, molecules_to_micromolar_conv, &my_math, hydrogenModelType);
    return object(out_array);
}

boost::python::object lightFlowAlignment(
    // Inputs:
    const string&             t_char,
    const string&             q_char,
    const string&             main_flow_order,
    bool match_nonzero, // enforce phase identity
    float strange_gap  // close gaps containing too many errors
    )
{
    vector<char>             flowOrder;
    vector<int>              qseq;
    vector<int>              tseq;
    vector<int>              aln_flow_index;
    vector<char>             aln;
    string                   synthetic_reference;

    boost::python::dict ret;

    LightFlowAlignment(t_char, q_char, main_flow_order, match_nonzero, strange_gap,
                       flowOrder, qseq, tseq, aln_flow_index, aln, synthetic_reference);

    ret["flowOrder"] = toNumpy(flowOrder);
    ret["qseq"] = toNumpy(qseq);
    ret["tseq"] = toNumpy(tseq);
    ret["align_index"] = toNumpy(aln_flow_index);
    ret["aln"] = toNumpy(aln);
    ret["synthetic_reference"] = boost::python::str(synthetic_reference.c_str());
    return(ret);
}


BOOST_PYTHON_MODULE(torrentPyLib)
{
    boost::python::docstring_options local_docstring_options(true, true, false);

    import_array();

    class_<PyBkgModel>("BkgModel")
            .def_readwrite("C", &PyBkgModel::C,"Nuc Concentration (default value: 50)")
            .def_readwrite("sens", &PyBkgModel::sens,"Sensitivity")
            .def("AdjustEmptyToBeadRatio", &PyBkgModel::AdjustEmptyToBeadRatio,"AdjustEmptyToBeadRatio( etbR, NucModifyRatio, RatioDrift, flow, should_fitTauE)")
            .def("TauBFromLinearModel", &PyBkgModel::TauBFromLinearModel,"TauBFromLinearModel( etbR, tauR_m, tauR_o, minTauB, maxTauB )")
            .def("IntegrateRedFromObservedTotalTrace", &PyBkgModel::IntegrateRedFromObservedTotalTracePy,"IntegrateRedFromObservedTotalTrace( purple_trace, blue_hydrogen,  deltaFrame, tauB, etbR)")
            .def("BlueTrace", &PyBkgModel::BlueTracePy,"BlueTrace( blue_hydrogen, deltaFrame, tauB, etbR)")
            .def("PurpleTrace", &PyBkgModel::PurpleTracePy,"PurpleTrace( blue_hydrogen, red_hydrogen, deltaFrame, tauB, etbR)")
            .def("CalculateNucRise", &PyBkgModel::CalculateNucRisePy,"CalculateNucRise( timeFrame, sub_steps, C, t_mid_nuc, sigma, nuc_span)")
            .def("GenerateRedHydrogensFromNucRise", &PyBkgModel::GenerateRedHydrogensFromNucRisePy,"GenerateRedHydrogensFromNucRise( nucRise, deltaFrame, sub_steps, my_start_index, C, Amplitude, Copies, krate, kmax, diffusion, hydrogenModelType )")
            ;


    class_<PyRawDat>("RawDatReader",boost::python::init<std::string,bool>("Reader for Ion dat files",(boost::python::arg("fname"),boost::python::arg("normalize")=true)))
            .def("LoadSlice", &PyRawDat::LoadSlice,"LoadSlice(start_row, start_col, height, width )")
            .def("LoadWells", &PyRawDat::LoadWells, "LoadWells(tuple((pos_col,pos_row))")
            .def("ApplyRowNoiseCorrection", &PyRawDat::ApplyRowNoiseCorrection,"ApplyRowNoiseCorrection( thumbnail )")
            .def("ApplyXTChannelCorrection", &PyRawDat::ApplyXTChannelCorrection,"ApplyXTChannelCorrection()")
            .def("Normalize", &PyRawDat::SetMeanOfFramesToZero,"SetMeanOfFramesToZero(norm_start, norm_end)")
            .def("ApplyPairPixelXtalkCorrection", &PyRawDat::ApplyPairPixelXtalkCorrection, "ApplyPairPixelXtalkCorrection( amount_pair_xtalk_fraction )")
            .def_readwrite("normalize", &PyRawDat::normalize, "bool - should normalize")
            .def_readwrite("xtcorrect", &PyRawDat::xtcorrect, "bool - should apply crosstalk correction")
            .def_readwrite("norm_start", &PyRawDat::norm_start, "first frame to use for normalization")
            .def_readwrite("norm_end", &PyRawDat::norm_end, "last frame to use for normalization")
            .def_readwrite("uncompress", &PyRawDat::uncompress, "should uncompress trace data")
            ;

    class_<PyWells>("WellsReader",boost::python::init<std::string>("Reader for Ion wells files",(boost::python::arg("fname"))))
            .def("LoadWells",&PyWells::LoadWells,"LoadWells( start_row, start_col, height, width )")
            .def("LoadWellsFlow",&PyWells::LoadWellsFlow,"LoadWellsFlow( start_row, start_col, height, width, flow )")
            ;

    class_<PyBam>("BamReader",boost::python::init<std::string>("Reader for bam files",(boost::python::arg("fname"))))
            .def("ReadBamHeader",&PyBam::ReadBamHeader,"ReadBamHeader() - return bam file header")
            .def("ReadBam",&PyBam::ReadBam,"ReadBam() - returns all reads in a single dictionary. May be very slow for large bam files.")
            .def("GetNumRecords", &PyBam::GetNumRecords,"GetNumRecords() - returns the number of reads in the bam file. This is a slow operation.")
            .def("Rewind", &PyBam::Rewind,"Rewind() - return read position to the beginning of the file.")
            .def("Close", &PyBam::Close,"Close() - close the BAM file that was opened.")
            .def("SetSampleSize", &PyBam::SetSampleSize, "SetSampleSize( nSamples ) - set read sample size to nSamples.")
            .def("SetDNARegion", &PyBam::SetDNARegion, "SetDNARegion( leftRefId, leftPosition, rightRefId, rightPosition ) - set the range of DNA coordinates for bam file iterator.")
            .def("SetChipRegion", &PyBam::SetChipRegion, "SetChipRegion( minRow, maxRow, minCol, maxCol ) - restrict bam file iterator to a region of the chip.")
            .def("Jump", &PyBam::Jump, "Jump( refId, position ) - move bam file iterator to this position in the bam file.")
            .def("SimulateCafie", &PyBam::SimulateCafie, "SimulateCafie( dict_read ) - simulate phasing effects.")
            .def("PhaseCorrect", &PyBam::PhaseCorrect, "PhaseCorrect( dict_read, keyFlow ) - apply phase correction to the read.")
            .def("SuppressRecalibration", &PyBam::SuppressRecalibration, "Jump( flag ) - Suppress recalibration in SimulateCafie and PhaseCorrect.")
            .def_readonly("data",&PyBam::data)
            .def_readonly("header",&PyBam::header, "Bam header.")
            .def_readonly("sample_size",&PyBam::sample_size, "Current sample size.")
            .def_readonly("reference_names",&PyBam::refNames, "List of alignement references.")
            .def_readwrite("flowAlign",&PyBam::flowAlign, "Should flow space alignment be done for the reads.")
            .def("__iter__",&PyBam::__iter__,return_internal_reference<>())
            .def("__enter__",&PyBam::__enter__,return_internal_reference<>())
            .def("__exit__",&PyBam::__exit__, "Exit")
            .def("next",&PyBam::next,"Returns bam file iterator.")
            ;


    enum_<MaskType>("BfMask")
            .value("MaskNone",MaskNone)
            .value("MaskEmpty",MaskEmpty)
            .value("MaskBead",MaskBead)
            .value("MaskLive",MaskLive)
            .value("MaskDud ",MaskDud )
            .value("MaskReference",MaskReference)
            .value("MaskTF",MaskTF)
            .value("MaskLib",MaskLib)
            .value("MaskPinned",MaskPinned)
            .value("MaskIgnore",MaskIgnore)
            .value("MaskWashout",MaskWashout)
            .value("MaskExclude",MaskExclude)
            .value("MaskKeypass",MaskKeypass)
            .value("MaskFilteredBadKey",MaskFilteredBadKey)
            .value("MaskFilteredShort",MaskFilteredShort)
            .value("MaskFilteredBadPPF",MaskFilteredBadPPF)
            .value("MaskFilteredBadResidual",MaskFilteredBadResidual)
            .value("MaskAll",MaskAll)
            ;

    def("LoadBfMaskByType", LoadBfMaskByType, "LoadBfMaskByType( file_name, BfMask_Type ) - load beadfind mask from a file. Set values to 1 where mask value matches specified type.");
    def("seqToFlow", seqToFlows,"seqToFlows( sequence, flowOrder ) - map base sequence to flow space.");
    def("LightFlowAlignment", lightFlowAlignment, "Flow alignment assuming the base alignment can mostly be relied upon.");

    class_<TreePhaser>("TreePhaser",boost::python::init<std::string>("Interface to DPTreephaser",(boost::python::arg("flowOrder"))))
            .def("setCalibFromTxtFile",&TreePhaser::setCalibFromTxtFile,"setCalibFromTxtFile( model_file, threshold) Set calibration model from file named model_file, with homopolymer limit threshold.")
            .def("setCalibFromJson",&TreePhaser::setCalibFromJson,"Set calibration model from json")
            .def("setCalibFromBamFile",&TreePhaser::setCalibFromBamFile,"Set calibration model from BAM file")
            .def("applyCalibForQueryName",&TreePhaser::applyCalibForQueryName,"Apply calibration model for the BAM read")
            .def("applyCalibForXY",&TreePhaser::applyCalibForXY,"Apply calibration model for the X-Y coordination")
            .def("disableCalibration",&TreePhaser::disableCalibration,"Disable the calibration model that was previously applied")
            .def("setCAFIEParams",&TreePhaser::setCAFIEParams,"Set CAFIE parameters")
            .def("queryAllStates",&TreePhaser::queryAllStates,"Query all states")
            .def("Simulate",&TreePhaser::Simulate,"Simulate")
            .def("setStateProgression",&TreePhaser::setStateProgression,"Set State Progression Model to Diagonal")
            .def("treephaserSolve",&TreePhaser::treephaserSolve,"Unnormalized solver")
            .def("treephaserSWANSolve",&TreePhaser::treephaserSWANSolve,"treephaserSWAN solver")
            .def("treephaserDPSolve",&TreePhaser::treephaserDPSolve,"DP treephaser solver")
            .def("treephaserAdaptiveSolve",&TreePhaser::treephaserAdaptiveSolve,"Adaptive treephaser solver")
            .def("treephaserSolveMulti",&TreePhaser::treephaserSolveMulti,"Basecall a list of reads")
            ;
}

