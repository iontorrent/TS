/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include "Image.h"
#include "Mask.h"
#include "GridMesh.h"
#include "T0Calc.h"
#include "T0Model.h"
#include "TraceChunk.h"
#include "Utils.h"
#include "HandleExpLog.h"
#include "SynchDat.h"
#include "SynchDatSerialize.h"
#include "OptArgs.h"
#include "VencoLossless.h"
#include "SvdDatCompress.h"
//#include "SvdDatCompressPlus.h"
#include "DeltaComp.h"
#include "DeltaCompFst.h"
#include "DeltaCompFstSmX.h"

//#include "SvdDatCompressPlusPlus.h"
//#include "CelebrandilLossy.h"
#include "SampleQuantiles.h"
#include "SampleStats.h"
#include "PJobQueue.h"
#include "H5File.h"
#include "SynchDatConvert.h"
using namespace std;

void usage() {
  cout << "TraceDriver - Program for converting images (dat files) to synchronized dats (sdats)" << endl;
  cout << "for trying in silico the synchronized dat process." << endl;
  cout << "" << endl;
  cout << "usage:" << endl;
  cout << "   TraceDriver --source-dir /path/to/dats --output-dir /output/for/sdat/files" << endl;
  cout << "" << endl;
  cout << "Options:" << endl;
  cout << "   --source-dir   where to read dat files from [required]" << endl;
  cout << "   --output-dir   where to write the dat files to [required]" << endl;
  cout << "   --debug-files  output some debug files [false]" << endl;
  cout << "   --compression  type of compression to use 'none','delta','svd','svd+','svd++', [delta]" << endl;
  cout << "   --num-flows    only convert first N flows [all]" << endl;
  cout << "   --err-con    constant to control compression error" << endl;
  cout << "   -h,--help      this message." << endl;
  exit(1);
}



void GenerateDataChunks(TraceConfig &config, T0Calc &t0, const struct RawImage *raw, 
                        int rowStep, int colStep, GridMesh<SigmaEst> &sigmaTMid,
                        GridMesh<TraceChunk> &mTraceChunks) {
  mTraceChunks.Init(raw->rows, raw->cols, rowStep, colStep);
  int rowStart,rowEnd,colStart, colEnd;
  for (size_t bIx = 0; bIx < t0.GetNumRegions(); bIx++) {
    SigmaEst &est = sigmaTMid.GetItem(bIx);
    t0.GetRegionCoords(bIx, rowStart, rowEnd, colStart, colEnd);
    TraceChunk &chunk = mTraceChunks.GetItem(bIx);
    float t0Time = t0.GetT0(bIx);
    chunk.mSigma = est.mSigma;
    chunk.mTMidNuc = est.mTMidNuc;
    t0Time -= config.time_start_slop;
    chunk.mStartDetailedTime = config.start_detailed_time;
    chunk.mStopDetailedTime = config.stop_detailed_time;
    chunk.mLeftAvg = config.left_avg;
    chunk.mOrigFrames = raw->frames;
    chunk.mT0 = t0Time/raw->baseFrameRate;
    // Old style using t0 instead of tmid nuc
    //    chunk.mTime.SetUpTime(raw->uncompFrames, chunk.mT0, chunk.mStartDetailedTime, chunk.mStopDetailedTime, chunk.mLeftAvg);
    chunk.mTime.SetUpTime(raw->uncompFrames, chunk.mTMidNuc, chunk.mStartDetailedTime, chunk.mStopDetailedTime, chunk.mLeftAvg);
    chunk.mBaseFrameRate = raw->baseFrameRate;
    chunk.mTimePoints.resize(chunk.mTime.mTimePoints.size());
    copy(chunk.mTime.mTimePoints.begin(), chunk.mTime.mTimePoints.end(), chunk.mTimePoints.begin());
    chunk.mTime.SetupConvertVfcTimeSegments(raw->frames, raw->timestamps, raw->baseFrameRate, raw->rows * raw->cols);
    chunk.SetChipInfo(raw->rows, raw->cols, raw->frames);
    chunk.SetDimensions(rowStart, rowEnd-rowStart, colStart, colEnd-colStart, 0, chunk.mTime.npts());
    chunk.ZeroData();
    chunk.mTime.ConvertVfcSegmentsOpt(rowStart, rowEnd, colStart, colEnd, 
                                      raw->rows, raw->cols, raw->frames,
                                      raw->image, &chunk.mData[0]);
  }
}

// void EstimateSigma(T0Calc &t0, SigmaTMidNucEstimation &sigmaEst, GridMesh<SigmaEst> &sigmaTMid) {
//   t0.CalculateSlopePostT0(2);
//   for (size_t i = 0; i < t0.GetNumRegions(); i++) {
//     float slopeEst = t0.GetSlope(i);
//     float t0Est = t0.GetT0(i);
//     SigmaEst &est = sigmaTMid.GetItem(i);
//     est.mSigma = 0;
//     est.mTMidNuc = 0;
//     if (t0Est > 0 && slopeEst > 0) {
//       est.mT0 = t0Est;
//       est.mRate = slopeEst;
//       sigmaEst.Predict(t0Est, slopeEst, est.mSigma, est.mTMidNuc);
//     }
//     else {
//       int rowStart,rowEnd,colStart, colEnd;
//       t0.GetRegionCoords(i, rowStart, rowEnd, colStart, colEnd);
//       cout << "Couldn't estimate sigma for region: " << rowStart << "," << colStart << endl;
//     }
//   }
// }

void ShiftWell(size_t wellIdx, int frames, float shift, size_t frameStep, short *scratch, short *image) {

  int maxPt = frames -1;
  for (int i = 0; i < maxPt; i++) {
    float spt = (float) i+shift;
    int left = (int) spt;
    int right = left+1;
    float frac = (float) right - spt;
    
    if (left < 0) left = 0;
    if (right < 0) right = 0;
    if (left > maxPt) left = maxPt;
    if (right > maxPt) right = maxPt;
    scratch[i] = (short) (image[left*frameStep + wellIdx]*frac+image[right*frameStep + wellIdx]* (1.0f-frac) + .5f);
  }
  /* copy back. */
  for (int i = 0; i < maxPt; i++) {
    image[i*frameStep + wellIdx] = scratch[i];
  }
}

void ShiftTraces(T0Calc &t0, vector<float> &wellT0, int frames, float framerate, int *timestamps, short *image) {
  int rowStart,rowEnd,colStart, colEnd;
  size_t numRow = t0.GetNumRow();
  size_t numCol = t0.GetNumCol();
  size_t numWells = numRow * numCol;
  framerate = framerate/ 1000.0f;
  short scratch[frames];
  for (size_t bIx = 0; bIx < t0.GetNumRegions(); bIx++) {
    t0.GetRegionCoords(bIx, rowStart, rowEnd, colStart, colEnd);
    float t0Region = t0.GetT0(bIx);
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++)  {
        size_t well = row * numCol + col;
        float shift = (t0Region - wellT0[well]) * framerate;
        ShiftWell(well, frames, shift, numWells, scratch, image);
      }
    }
  }
}

void DcOffsetWell(size_t wellIdx, size_t frameStep, float t0, int *timestamps, int frames, short *image) {
  float offset = 0.0f;
  size_t count = 0;
  int fIx = 0;
  t0 = max(t0,0.0f);
  // Calculate the dc offset or just use first frame if no other guess
  while(fIx < frames && t0 > timestamps[fIx]) {
    offset += image[fIx * frameStep + wellIdx];
    count++;
    fIx++;
  }
  if (count > 0) {
    offset /= count;
  }
  else {
    offset = image[wellIdx]; // 0th frame for this well
  }
  // Do subtraction
  for (fIx = 0; fIx < frames; fIx++) {
    image[fIx * frameStep + wellIdx] = round(image[fIx * frameStep + wellIdx] - offset);
  }
}

void DcOffSet(T0Calc &t0, vector<float> &wellT0, int frames, int *timestamps, short *image, float slop) {
  int rowStart,rowEnd,colStart, colEnd;
  size_t numRow = t0.GetNumRow();
  size_t numCol = t0.GetNumCol();
  size_t numWells = numRow * numCol;
  for (size_t bIx = 0; bIx < t0.GetNumRegions(); bIx++) {
    t0.GetRegionCoords(bIx, rowStart, rowEnd, colStart, colEnd);
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++)  {
        size_t well = row * numCol + col;
        float t0 = wellT0[well];
        DcOffsetWell(well, numWells, t0-slop, timestamps, frames, image);
      }
    }
  }
}

void OutputSigmaTmidEstimates(GridMesh<SigmaEst> &sigmaTMid, const char *fileName) {
  ofstream out(fileName);
  int rowStart,rowEnd,colStart, colEnd;      
  out << "rowStar" << "\t" << "rowEnd" << "\t" << "colStart" << "\t" << "colEnd" << "\t" << "sigma.t0.est" << "\t" << "tmidnuc.t0.est" << "\t" << "t0.est" << "\t" << "rate" <<endl;
  for (size_t bIx = 0; bIx < sigmaTMid.GetNumBin(); bIx++) {            
    sigmaTMid.GetBinCoords(bIx, rowStart, rowEnd, colStart, colEnd);
    SigmaEst &est = sigmaTMid.GetItem(bIx);
    out << rowStart << "\t" << rowEnd << "\t" << colStart << "\t" << colEnd << "\t" << est.mSigma << "\t" << est.mTMidNuc << "\t" << est.mT0 << "\t" << est.mRate << endl;
  }
  out.close();
}

void OutputTraceChunks(GridMesh<TraceChunk> &traceChunks, const char *fileName) {
  //ofstream out("flow_0_data_chunks.txt");
  ofstream out(fileName);

  int rowStart,rowEnd,colStart, colEnd;      
  //      for (size_t bIx = 0; bIx < traceChunks.GetNumBin(); bIx++) {
  for (size_t bIx = 0; bIx < 1; bIx++) {
    traceChunks.GetBinCoords(bIx, rowStart, rowEnd, colStart, colEnd);
    TraceChunk &tc = traceChunks.GetItem(bIx);
    float sum = 0;
    for (int i = 0; i < tc.mTime.npts(); i++) {
      sum += tc.mTime.deltaFrameSeconds[i];
    }
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        out << row << "\t" << col;
        for (size_t frame = 0; frame < tc.mDepth; frame++) {
          out << "\t" << tc.At(row, col, frame);
        }
        out << endl;
      }
    }
  }
  out.close();
}

class CreateSDat : public PJob {
public:
  CreateSDat() {
    config = NULL;
    wellT0 = NULL;
    bfT0 = NULL;
    sigmaTMid = NULL;
  }
  void Init(TraceConfig *_config, const std::string &_input, const std::string &_output, vector<float> *_wellT0, T0Calc *_bfT0, GridMesh<SigmaEst> *_sigmaTMid) {
    config = _config;
    input = _input;
    output = _output;
    wellT0 = _wellT0;
    bfT0 = _bfT0;
    sigmaTMid = _sigmaTMid;

    if (config->compressionType == "svd") {
      SvdDatCompress *dc = new SvdDatCompress(config->precision, config->numEvec);
      serializer.SetCompressor(dc);
      cout << "Doing lossy svd compression. (" << serializer.GetCompressionType() << ")" << endl;
    }
    // else if (config->compressionType == "svd+") {
    //   SvdDatCompressPlus *dc = new SvdDatCompressPlus();
    //   serializer.SetCompressor(dc);
    //   cout << "Doing lossy svd compression. (" << serializer.GetCompressionType() << ")" << endl;
    // }
    // else if (config->compressionType == "svd++") {
    //   SvdDatCompressPlusPlus *dc = new SvdDatCompressPlusPlus();
    //   if (config->errCon >0 )
    //     dc->SetErrCon(config->errCon);
    //   if (config->rankGood > 0 )
    //     dc->SetRankGood(config->rankGood);
    //   if (config->pivot > 0)
    //     dc->SetPivot(config->pivot);
    //   serializer.SetCompressor(dc);
    //   cout << "Doing both lossy svd and delta compression . (" << serializer.GetCompressionType() << ")" << endl;
    // }
    else if (config->compressionType == "delta") {
      VencoLossless *venco = new VencoLossless();
      serializer.SetCompressor(venco);
      cout << "Doing lossless delta compression. (" << serializer.GetCompressionType() << ")" << endl;
    }
    else if (config->compressionType == "delta-plain") {
      DeltaComp *delta = new DeltaComp();
      serializer.SetCompressor(delta);
      cout << "Doing lossless delta plain compression. (" << serializer.GetCompressionType() << ")" << endl;
    }
    else if (config->compressionType == "delta-plain-fst") {
      DeltaCompFst *delta = new DeltaCompFst();
      serializer.SetCompressor(delta);
      cout << "Doing lossless delta plain fast compression. (" << serializer.GetCompressionType() << ")" << endl;
    }
    else if (config->compressionType == "delta-plain-fst-smx") {
      DeltaCompFstSmX *delta = new DeltaCompFstSmX();
      serializer.SetCompressor(delta);
      cout << "Doing lossless delta plain fast compression. (" << serializer.GetCompressionType() << ")" << endl;
    }
    else if (config->compressionType == "none") {
      TraceCompressor *vanilla = new TraceNoCompress();
      serializer.SetCompressor(vanilla);
      cout << "Doing no compression. (" << serializer.GetCompressionType() << ")" << endl;
    }
    else {
      ION_ABORT("Don't recognize compression type: " + config->compressionType);
    }
  }

  void Run() {
    Image datImg;
    datImg.LoadRaw(input.c_str());
    const RawImage *datRaw = datImg.GetImage(); 
    //    ShiftTraces(*bfT0, *wellT0, datRaw->frames, datRaw->baseFrameRate, datRaw->timestamps, datRaw->image);
    SynchDat sdat; //GridMesh<TraceChunk> traceChunks;
    GenerateDataChunks(*config, *bfT0, datRaw, config->row_step, config->col_step, *sigmaTMid, sdat.mChunks);
    serializer.Write(output.c_str(), sdat);

    if (config->doDebug) {
      char buffer[2048];
      string tmp = input.substr(input.size()-8,8);
      snprintf(buffer, sizeof(buffer), "comIn_%s", tmp.c_str());
      OutputTraceChunks(sdat.mChunks,buffer);

      TraceChunkSerializer readSer;
      //      GridMesh<TraceChunk> traceIn;
      SynchDat sdatIn;
      readSer.Read(output.c_str(),sdatIn);
      snprintf(buffer, sizeof(buffer), "decomOut_%s", tmp.c_str());
      OutputTraceChunks(sdatIn.mChunks,buffer);
    }
    datImg.Close();
  }

  TraceConfig *config;
  std::string input;
  std::string output;
  vector<float> *wellT0;
  T0Calc *bfT0;
  GridMesh<SigmaEst> *sigmaTMid;
  TraceChunkSerializer serializer;
};

int main(int argc, const char *argv[]) {
  OptArgs opts;  
  TraceConfig config;
  string inputDir;
  string outputDir;
  bool help;
  opts.ParseCmdLine(argc, argv);
  opts.GetOption(inputDir, "", '-', "source-dir");
  opts.GetOption(outputDir, "", '-', "output-dir");
  opts.GetOption(config.precision, "5", '-', "precision");
  opts.GetOption(config.numEvec, "7", '-', "num-evec");
  opts.GetOption(config.doDebug, "false", '-', "debug-files");
  opts.GetOption(config.compressionType, "delta", '-', "compression");
  opts.GetOption(config.numFlows, "-1", '-', "num-flows");
  opts.GetOption(config.numCores, "4", '-', "num-cores");
  opts.GetOption(config.errCon,"0",'-',"err-con");
  opts.GetOption(config.rankGood,"0",'-',"rank-good");
  opts.GetOption(config.pivot,"0",'-',"pivot");
  opts.GetOption(help, "false", 'h', "help");
  opts.GetOption(config.use_hard_est, "false",'-', "use-hard-est");
  opts.GetOption(config.t0_hard, "0", '-', "t0-hard");
  opts.GetOption(config.tmid_hard, "0", '-', "tmid-hard");
  opts.GetOption(config.sigma_hard, "0", '-', "sigma-hard");
  opts.GetOption(config.row_step, "100", '-', "row-step");
  opts.GetOption(config.col_step, "100", '-', "col-step");
  opts.GetOption(config.bg_param, "", '-', "bg-param");
  opts.GetOption(config.grind_acq_0, "0", '-', "grind-acq0");
  if(help || inputDir.empty() || outputDir.empty()) {
    usage();
  }
  char *explog_path = NULL;
  explog_path = MakeExpLogPathFromDatDir(inputDir.c_str());
  int numFlows = config.numFlows;
  if (numFlows < 0) { 
    numFlows = GetTotalFlows(explog_path); 
  }

  // Check and setup our compression type
  TraceChunkSerializer serializer;
  serializer.SetRecklessAbandon(true);
  if (config.compressionType == "svd") {
    SvdDatCompress *dc = new SvdDatCompress(config.precision, config.numEvec);
    serializer.SetCompressor(dc);
    cout << "Doing lossy svd compression. (" << serializer.GetCompressionType() << ")" << endl;
  }
  // else if (config.compressionType == "svd+") {
  //   SvdDatCompressPlus *dc = new SvdDatCompressPlus();
  //   serializer.SetCompressor(dc);
  //   cout << "Doing lossy svd compression. (" << serializer.GetCompressionType() << ")" << endl;
  // }
  // else if (config.compressionType == "svd++") {
  //   SvdDatCompressPlusPlus *dc = new SvdDatCompressPlusPlus();
  //   if (config.errCon >0 )
  //     dc->SetErrCon(config.errCon);
  //   if (config.rankGood > 0 )
  //     dc->SetRankGood(config.rankGood);
  //   if (config.pivot > 0)
  //     dc->SetPivot(config.pivot);
  //   serializer.SetCompressor(dc);
  //   cout << "Doing lossy svd compression for good traces and delta for bad ones. (" << serializer.GetCompressionType() << ")" << endl;
  // }
  else if (config.compressionType == "delta") {
    VencoLossless *venco = new VencoLossless();
    serializer.SetCompressor(venco);
    cout << "Doing lossless delta compression. (" << serializer.GetCompressionType() << ")" << endl;
  }
  else if (config.compressionType == "delta-plain") {
    DeltaComp *delta = new DeltaComp();
    serializer.SetCompressor(delta);
    cout << "Doing lossless delta plain compression. (" << serializer.GetCompressionType() << ")" << endl;
  }
  else if (config.compressionType == "delta-plain-fst") {
    DeltaCompFst *delta = new DeltaCompFst();
    serializer.SetCompressor(delta);
    cout << "Doing lossless delta plain fast compression. (" << serializer.GetCompressionType() << ")" << endl;
  }
  else if (config.compressionType == "delta-plain-fst-smx") {
    DeltaCompFstSmX *delta = new DeltaCompFstSmX();
    serializer.SetCompressor(delta);
    cout << "Doing lossless delta plain fast compression. (" << serializer.GetCompressionType() << ")" << endl;
  }
  else if (config.compressionType == "none") {
    TraceCompressor *vanilla = new TraceNoCompress();
    serializer.SetCompressor(vanilla);
    cout << "Doing no compression. (" << serializer.GetCompressionType() << ")" << endl;
  }
  else {
    ION_ABORT("Don't recognize compression type: " + config.compressionType);
  }

  const char *id = GetChipId(explog_path);
  if (explog_path) free (explog_path);
  ChipIdDecoder::SetGlobalChipId(id);
  ImageTransformer::CalibrateChannelXTCorrection(inputDir.c_str(), "lsrowimage.dat");

  Image bfImg1;
  string bfFile = inputDir + "/beadfind_pre_0003.dat";
  bfImg1.LoadRaw(bfFile.c_str());
  const RawImage *bf1raw = bfImg1.GetImage(); 
  Mask mask(bf1raw->cols, bf1raw->rows);
  ImageTransformer::XTChannelCorrect(bfImg1.raw,bfImg1.results_folder);

  bfImg1.FilterForPinned (&mask, MaskEmpty, false);

  Image bfImg2;
  string bfFile2 = inputDir + "/beadfind_pre_0001.dat";
  bfImg2.LoadRaw(bfFile2.c_str());
  ImageTransformer::XTChannelCorrect(bfImg2.raw,bfImg1.results_folder);

  bfImg2.FilterForPinned (&mask, MaskEmpty, false);
  const RawImage *bf2raw = bfImg2.GetImage(); 


  GridMesh<T0Prior> t0Prior;
  T0Calc bfT0;
  /* Calc t0 and get prior. */
  cout << "Doing beadfind t0" << endl;
  GenerateBfT0Prior(config, bf1raw->image, bf1raw->baseFrameRate, bf1raw->rows, bf1raw->cols,
                    bf1raw->frames, bf1raw->timestamps,
                    config.row_step, config.col_step, &mask, bfT0, t0Prior);

  GridMesh<T0Prior> t0Prior2;
  T0Calc bfT02;
  GenerateBfT0Prior(config, bf2raw->image, bf2raw->baseFrameRate, bf2raw->rows, bf2raw->cols,
                    bf2raw->frames, bf2raw->timestamps,
                    config.row_step, config.col_step, &mask, bfT02, t0Prior2);

  SigmaTMidNucEstimation sigmaEst;
  sigmaEst.Init(config.rate_sigma_intercept, config.rate_sigma_slope, 
                config.t0_tmid_intercept, config.t0_tmid_slope, bf1raw->baseFrameRate);
  GridMesh<SigmaEst> sigmaTMid;
  bfImg1.Close();
  bfImg2.Close();

  // Calculate individual well t0 by looking at neighboring regions
  vector<float> wellT0;
  bfT0.CalcIndividualT0(wellT0, 0);
  vector<float> wellT02;
  bfT02.CalcIndividualT0(wellT02, 0);
  for (size_t i =0; i< wellT0.size();i++) {
    if (wellT0[i] > 0 && wellT02[i] > 0) {
      wellT0[i] = (wellT0[i] + wellT02[i])/2.0f;
    }
    else {
      wellT0[i] = max(wellT0[i], wellT02[i]);
    }
  }

  // Average the region level t0, should we do this first and then just do sinle well level?
  for (size_t bIx = 0; bIx < bfT0.GetNumRegions(); bIx++) {
    double t1 = bfT0.GetT0(bIx);
    double t2 = bfT02.GetT0(bIx);
    if (t1 > 0 && t2 > 0) {
      t1 = (t1 + t2)/2.0;
    }
    else {
      t1 = max(t1,t2);
    }
    bfT0.SetT0(bIx, t1);
  }

  // Single thread first dat
  for (size_t datIx = 0; datIx < 1; ++datIx) {
    cout << "Doing: " << datIx << endl;
    char buffer[2048];
    snprintf(buffer, sizeof(buffer), "%s/acq_%.4d.dat", inputDir.c_str(), (int) datIx);
    string datFile = buffer;
    /* Use prior to calculate t0 and slope. */
    Image datImg;
    T0Calc t0;
    datImg.LoadRaw(datFile.c_str());
    //    ImageTransformer::XTChannelCorrect(datImg.raw,datImg.results_folder);
    const RawImage *datRaw = datImg.GetImage(); 

    /* Estimate sigma and t_mid_nuc */
    if (datIx == 0) {
      cout << "Doing acquisition t0" << endl;
      GenerateAcqT0Prior(config, datRaw->image, datRaw->baseFrameRate, datRaw->rows, datRaw->cols,
                         datRaw->frames, datRaw->timestamps,
                         config.row_step, config.col_step, &mask, t0, t0Prior);
      
      ClockTimer timer;
      cout << "Estimating sigma." << endl;
      sigmaTMid.Init(datRaw->rows, datRaw->cols, config.row_step, config.col_step);
      for (size_t bIx = 0; bIx < t0.GetNumRegions(); bIx++) {
        t0.SetT0(bIx, bfT0.GetT0(bIx));
      }
      EstimateSigmaValue(t0, sigmaEst, sigmaTMid);
      timer.PrintMilliSeconds(cout,"Sigma Est took:");
      string sigmaFile = outputDir + "/sigma_tmid_est.txt";
      OutputSigmaTmidEstimates(sigmaTMid, sigmaFile.c_str());
    }

    /* For each region do shifting */
    ClockTimer timer;
    cout << "Shifting traces" << endl;
    timer.StartTimer();
    //    ShiftTraces(bfT0, wellT0, datRaw->frames, datRaw->baseFrameRate, datRaw->timestamps, datRaw->image);
    timer.PrintMilliSeconds(cout,"Shift took:");
    if (!config.bg_param.empty()) {
      Mat<int> rowsCols;
      Mat<float> tmidSigma;
      Mat<float> fitTmidSigma;
      string path = config.bg_param + ":/region/region_offset_RowCol";
      if (!H5File::ReadMatrix(path, rowsCols)) {
        ION_ABORT("Couldn't read file: " + path);
      }
      path = config.bg_param + ":/region/region_param/flow_0020";
      if (!H5File::ReadMatrix(path, fitTmidSigma)) {
        ION_ABORT("Couldn't read file: " + path);
      }
      for (size_t i = 0; i < rowsCols.n_rows; i++) {
        int row = rowsCols.at(i,0);
        int col = rowsCols.at(i,1);
        SigmaEst &est = sigmaTMid.GetItemByRowCol(row, col);
        float tmid_est =  fitTmidSigma.at(i,61);
        float sigma_est = fitTmidSigma.at(i,66);
        est.mTMidNuc = tmid_est;
        est.mSigma = sigma_est;
      }
      string fitSigmaFile = outputDir + "/bg_fit_sigma_tmid_est.txt";
      OutputSigmaTmidEstimates(sigmaTMid, fitSigmaFile.c_str());

      path = config.bg_param + ":/region/region_init_param";
      if (!H5File::ReadMatrix(path, tmidSigma)) {
        ION_ABORT("Couldn't read file: " + path);
      }
      for (size_t i = 0; i < rowsCols.n_rows; i++) {
        int row = rowsCols.at(i,0);
        int col = rowsCols.at(i,1);
        SigmaEst &est = sigmaTMid.GetItemByRowCol(row, col);
        float tmid_est =  tmidSigma.at(i,0);
        float sigma_est = tmidSigma.at(i,1);
        est.mTMidNuc = tmid_est;
        est.mSigma = sigma_est;
      }
      string sigmaFile = outputDir + "/supplied_sigma_tmid_est.txt";
      OutputSigmaTmidEstimates(sigmaTMid, sigmaFile.c_str());
    }
    else if (config.use_hard_est) {
      for (size_t i = 0; i < bfT0.GetNumRegions(); i++) {
        bfT0.SetT0(i,config.t0_hard * datRaw->baseFrameRate + config.time_start_slop);
      }
      for (size_t i = 0; i < sigmaTMid.GetNumBin(); i++) {
        SigmaEst &est = sigmaTMid.GetItem(i);
        est.mTMidNuc = config.tmid_hard;
        est.mSigma = config.sigma_hard;
        est.mT0 = config.t0_hard;
      }
    }
    /* Use t0 and sigma to get the time compression bkgModel wants. */
    cout << "Generating chunks" << endl;
    //    GridMesh<TraceChunk> traceChunks;
    SynchDat sdat;
    if (datIx == 0  && config.grind_acq_0 > 0) {
      int nTimes = config.grind_acq_0;
      timer.StartTimer();
      for (int i = 0; i <nTimes; i++) {
        //GridMesh<TraceChunk> traceChunken;
        SynchDat sdatIn;
        GenerateDataChunks(config, bfT0, datRaw, config.row_step, config.col_step, sigmaTMid, sdatIn.mChunks);
        snprintf(buffer, sizeof(buffer), "%s/acq_%.4d.sdat", outputDir.c_str(), (int)datIx);
        serializer.Write(buffer, sdatIn);
      }
      size_t usec = timer.GetMicroSec();
      cout << "Took: " << usec / 1.0e6 << " seconds, " << usec / (nTimes * 1.0f) << " usec per load." << endl;
      timer.PrintMilliSeconds(cout,"Chunks took:");
      exit(0);
    }
    else {
      timer.StartTimer();
        if (datIx == 0 && config.doDebug) {
          OutputTraceChunks(sdat.mChunks,"flow_0_data_chunks.txt");
        }
      GenerateDataChunks(config, bfT0, datRaw, config.row_step, config.col_step, sigmaTMid, sdat.mChunks);
      timer.PrintMilliSeconds(cout,"Chunks took:");
    }
    datImg.Close();    

    /* Serialize onto disk. */
    snprintf(buffer, sizeof(buffer), "%s/acq_%.4d.sdat", outputDir.c_str(), (int)datIx);
    serializer.Write(buffer, sdat);
    /* Read back in first flow for checking */
    if (datIx == 0) {
      TraceChunkSerializer readSerializer;
      readSerializer.SetRecklessAbandon(true);
      //      GridMesh<TraceChunk> traceChunksIn;  
      SynchDat sdatIn;
      readSerializer.Read(buffer, sdatIn);
      if (datIx == 0 && config.doDebug) {
        OutputTraceChunks(sdatIn.mChunks, "flow_0_data_chunks_read.txt");
      }
      SampleQuantiles<float> s(50000);
      SampleQuantiles<float> s2(50000);
      SampleQuantiles<float> sAbs(50000);
      SampleStats<double> ss;
      int diffCount = 0;
      for (size_t bIx = 0; bIx < sdatIn.mChunks.mBins.size(); bIx++) {
        if (sdatIn.mChunks.mBins[bIx].mT0 != sdat.mChunks.mBins[bIx].mT0) {
          cout << "Got: " << sdatIn.mChunks.mBins[bIx].mT0 << " vs: " << sdat.mChunks.mBins[bIx].mT0 << endl;
          exit(1);
        }
        for (size_t i = 0; i < sdatIn.mChunks.mBins[bIx].mData.size(); i++) {
          double diff = (double)sdatIn.mChunks.mBins[bIx].mData[i] - (double)sdat.mChunks.mBins[bIx].mData[i];
          if (!std::isfinite(diff)) {
            cout << "NaNs!!" << endl;
          }
          if (diffCount < 10 && fabs(diff) > .00001) { // != 0) {
            diffCount++;
            cout << "Bin: " << bIx << " well: " << i << " diff is: " << diff << endl;
          }
          s.AddValue(diff);
          sAbs.AddValue(fabs(diff));
          ss.AddValue(sqrt(diff * diff));
          s2.AddValue(sqrt(diff * diff));
        }
      }
      cout << "Median rms: " << s2.GetMedian()  << " Avg: " << ss.GetMean() << " diff: " << s.GetMedian() << endl;
      cout << "Abs(diff) Quantiles:" << endl;
      for (size_t i = 0; i <= 100; i+=10) {
        cout << i << "\t" << sAbs.GetQuantile(i/100.0) << endl;
      }
    }      
  }
  // do the next N flows multithreaded
  if (numFlows > 1) {
    PJobQueue jQueue (config.numCores, numFlows-1);  
    vector<CreateSDat> jobs(numFlows -1);
    // for (int i = 0; i < 4; i++) {
    //   char buffer[2048];
    //   snprintf(buffer, sizeof(buffer), "%s/beadfind_pre_%.4d.dat", inputDir.c_str(), (int) i);
    //   string input = buffer;
    //   snprintf(buffer, sizeof(buffer), "%s/beadfind_pre_%.4d.sdat", outputDir.c_str(), (int)i);
    //   string output = buffer;
    //   jobs[i].Init(&config, input, output, &wellT0, &bfT0, &sigmaTMid);
    //   jQueue.AddJob(jobs[i]);
    // }

    // jQueue.WaitUntilDone();
    for (int i = 1; i < numFlows; i++) {
      char buffer[2048];
      snprintf(buffer, sizeof(buffer), "%s/acq_%.4d.dat", inputDir.c_str(), (int) i);
      string input = buffer;
      snprintf(buffer, sizeof(buffer), "%s/acq_%.4d.sdat", outputDir.c_str(), (int)i);
      string output = buffer;
      jobs[i-1].Init(&config, input, output, &wellT0, &bfT0, &sigmaTMid);
      jQueue.AddJob(jobs[i-1]);
    }
    jQueue.WaitUntilDone();
  }
  /* Serialize into backbround models */
  cout << "Done." << endl;
}
