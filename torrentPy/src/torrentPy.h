// Copyright (C) 2016 Thermo Fisher Scientific. All Rights Reserved.
#include <Python.h>
#include "Image.h"
#include "BaseCaller/DPTreephaser.h"
#include "Calibration/LinearCalibrationModel.h"
#include <boost/python.hpp>
#include <limits>
#include "BaseCallerUtils.h"

class TreePhaser{
public:
    TreePhaser(const std::string& sequence);
    TreePhaser(const TreePhaser& copy_me);
    ~TreePhaser(){ delete dpTreephaser; }
    void setCalibFromTxtFile( const std::string& model_file, int threshold );
    void setCalibFromBamFile( const std::string& bam_file );
    void setCalibFromJson(const std::string& json_model, int threshold );
    void setCAFIEParams(double cf, double ie, double dr);
    void setStateProgression(bool diagonalStates);
    void setSolverName( const std::string& _solverName ){ solverName = _solverName; }
    bool applyCalibForQueryName(const string& qname);
    bool applyCalibForXY(int calib_x, int calib_y);
    void disableCalibration();
    void Simulate_BasecallerRead(BasecallerRead& basecaller_read, int max_flows, bool state_inphase=false) {dpTreephaser->Simulate(basecaller_read, max_flows, state_inphase);};
    void NormalizeAndSolve_SWnorm_BasecallerRead(BasecallerRead& basecaller_read, int max_flows)  {dpTreephaser->NormalizeAndSolve_SWnorm(basecaller_read, max_flows);};
    boost::python::object queryAllStates(const string &sequence, int maxFlows, int calib_x=0, int calib_y=0);
    boost::python::object Simulate(const string &sequence, int maxFlows);
    boost::python::dict treephaserSolve(boost::python::object _signal, boost::python::object _keyVec);
    boost::python::dict treephaserAdaptiveSolve(boost::python::object _signal, boost::python::object _keyVec);
    boost::python::dict treephaserDPSolve(boost::python::object _signal, boost::python::object _keyVec);
    boost::python::dict treephaserSWANSolve(boost::python::object _signal, boost::python::object _keyVec);
    void treephaserSolveMulti( const string &solverName, boost::python::list readList, boost::python::object _keyVec);

private:
    DPTreephaser *dpTreephaser;
    LinearCalibrationModel calibModel; // For Calibration from TXT or JSON
    std::string solverName;
    ion::FlowOrder flowOrder;
    map<string, LinearCalibrationModel> bam_header_recalibration; // For Calibration stored in BAM header
    multimap<string,pair<int,int> > block_hash;  // For bam_header_recalibration, from run id, find appropriate block coordinates available
};

class PyRawDat{
    Image img;
    std::string fname;
    std::string curr_fname;

    void loadImg( const std::string& fname );

public:
    bool normalize;
    bool xtcorrect;
    bool uncompress;
    int norm_start, norm_end;

    PyRawDat( std::string _fname, bool _normalize ) : curr_fname(""), normalize(_normalize), xtcorrect(false), uncompress(true), norm_start(1), norm_end(3) { fname = _fname; loadImg(fname); }
    ~PyRawDat(){
        img.Close();
    }

    boost::python::object LoadSlice(int row , int col, int h, int w);
    boost::python::object LoadWells(const boost::python::tuple& pos);
    void ApplyRowNoiseCorrection( int thumbnail = 1 );
    void ApplyXTChannelCorrection(void);
    void SetMeanOfFramesToZero(int norm_start=1, int norm_end=3);
    void ApplyPairPixelXtalkCorrection( float pair_xtalk_fraction );
};

class PyWells{
    std::string fname;

public:
    PyWells( std::string _fname ): fname(_fname){}
    boost::python::object LoadWells(int rows, int cols , int h, int w);
    boost::python::object LoadWellsFlow(int rows, int cols , int h, int w, int flow);
};

class PyBam{
private:
    std::string fname;
    BamTools::BamReader bamReader;
    int numRecords;
    std::map< std::string, int > keyLen;
    std::vector< BamTools::BamAlignment > alignmentSample;
    map<string,string> flow_order_by_read_group;
    map<string,string> key_seq_by_read_group;
    map<string, TreePhaser> treephaser_by_runid;
    int minRow, maxRow, minCol, maxCol;
    bool suppress_recalibration = false;

    void appendToHeader( const std::string& key, bool has_key, std::string value );
    void appendRecord( const boost::python::dict &rec );
    bool getNextRead( boost::python::dict& );

public:
    boost::python::dict data;
    boost::python::dict header;
    boost::python::list refNames;
    unsigned int sample_size;
    int flowAlign;

    void Open( std::string _fname );
    void Close(){ bamReader.Close(); }
    int GetNumRecords( void );

    boost::python::dict ReadBamHeader(void);
    boost::python::dict ReadBam(void);
    void Rewind( void ) { bamReader.Rewind(); }
    void SuppressRecalibration(bool flag) { suppress_recalibration = flag; }
    void SetSampleSize( int nSample );

    bool SetDNARegion( int leftRefId, int leftPosition, int rightRefId, int rightPosition ){ return bamReader.SetRegion( leftRefId, leftPosition, rightRefId, rightPosition ); }
    void SetChipRegion( int _minRow, int _maxRow, int _minCol, int _maxCol ){ minRow=_minRow; maxRow=_maxRow; minCol=_minCol; maxCol=_maxCol;}
    bool Jump( int refId, int position ){ return bamReader.Jump( refId, position ); }
    void SimulateCafie( boost::python::dict& read );
    void PhaseCorrect(boost::python::dict& read , boost::python::object keyFlow );

    ~PyBam(){ Close(); }
    PyBam(std::string _fname);

    PyBam& __iter__( void ){ return *this; }
    PyBam& __enter__( void ){ return *this; }
    bool __exit__(const boost::python::object& type, const boost::python::object& msg, const boost::python::object& traceback){Close(); return false;}
    boost::python::dict next( void );

};

class PyBkgModel{
public:
    float C;
    float sens;
    PyBkgModel() : C(50.),sens(1.256 * 2./100000.){}
    float AdjustEmptyToBeadRatio(float etbR, float NucModifyRatio, float RatioDrift, int flow, bool fitTauE);
    float TauBFromLinearModel(float etbR, float tauR_m, float tauR_o , float minTauB, float maxTauB);
    boost::python::object IntegrateRedFromObservedTotalTracePy (boost::python::object purple_obs, boost::python::object blue_hydrogen, boost::python::object deltaFrame, float tauB, float etbR);
    boost::python::object BlueTracePy ( boost::python::object blue_hydrogen,  boost::python::object deltaFrame, float tauB, float etbR);
    boost::python::object PurpleTracePy ( boost::python::object blue_hydrogen,  boost::python::object red_hydrogen, boost::python::object deltaFrame, float tauB, float etbR);
    boost::python::object CalculateNucRisePy (boost::python::object timeFrame, int sub_steps, float C, float t_mid_nuc, float sigma, float nuc_span);
    boost::python::object GenerateRedHydrogensFromNucRisePy (boost::python::object nucRise, boost::python::object deltaFrame, int sub_steps, int my_start_index, float C, float Amplitude,
                                                              float Copies, float krate, float kmax, float diffusion , int hydrogenModelType);
};

boost::python::object lightFlowAlignment(
    // Inputs:
    const string&             t_char,
    const string&             q_char,
    const string&             main_flow_order,
    bool match_nonzero, // enforce phase identity
    float strange_gap  // close gaps containing too many errors
    );
