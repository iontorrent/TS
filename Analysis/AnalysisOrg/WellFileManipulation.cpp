/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "WellFileManipulation.h"
#include "Mask.h"
#include "IonErr.h"
#include "ImageSpecClass.h"

using namespace std;

void SetWellsToLiveBeadsOnly(RawWells &rawWells, Mask *maskPtr)
{
  // Get subset of wells we want to track, live only...
  vector<int> subset;
  size_t maskWells = maskPtr->H() * maskPtr->W();
  subset.reserve(maskWells);
  for (size_t i = 0; i < maskWells; i++) {
    if (maskPtr->Match(i, MaskLive)) {
      subset.push_back(i);
    }
  }
  rawWells.SetSubsetToWrite(subset);
}



void GetMetaDataForWells(char *explog_path, RawWells &rawWells, const char *chipType)
{
  const char * paramsToKeep[] = {"Project","Sample","Start Time","Experiment Name","User Name","Serial Number","Oversample","Frame Time", "Num Frames", "Cycles", "Flows", "LibraryKeySequence", "ChipTemperature", "PGMTemperature", "PGMPressure","W2pH","W1pH","Cal Chip High/Low/InRange"};
  char* paramVal = NULL;
  for (size_t pIx = 0; pIx < sizeof(paramsToKeep)/sizeof(char *); pIx++)
  {
    if ((paramVal = GetExpLogParameter(explog_path, paramsToKeep[pIx])) != NULL)
    {
      string value = paramVal;
      size_t pos = value.find_last_not_of("\n\r \t");
      if (pos != string::npos)
      {
        value = value.substr(0,pos+1);
      }
      rawWells.SetValue(paramsToKeep[pIx], value);
      free (paramVal);
    }

  }
  rawWells.SetValue("ChipType", chipType);
}



void CreateWellsFileForWriting (RawWells &rawWells, Mask *maskPtr,
                                CommandLineOpts &inception_state,
                                int numFlows,
                                int numRows, int numCols,
                                const char *chipType)
{
  // set up wells data structure
  MemUsage ("BeforeWells");
  //rawWells.SetFlowChunkSize(flowChunk);
  rawWells.SetCompression (inception_state.bkg_control.signal_chunks.wellsCompression);
  rawWells.SetRows (numRows);
  rawWells.SetCols (numCols);
  rawWells.SetFlows (numFlows);
  rawWells.SetFlowOrder (inception_state.flow_context.flowOrder); // 6th duplicated code
  SetWellsToLiveBeadsOnly (rawWells,maskPtr);
  // any model outputs a wells file of this nature
  GetMetaDataForWells ((char*)(inception_state.sys_context.explog_path.c_str()),rawWells,chipType);

  rawWells.OpenForWrite();
  rawWells.WriteRanks(); // dummy, written for completeness
  rawWells.WriteInfo();  // metadata written, do not need to rewrite
  rawWells.Close(); // just create in this routine
  MemUsage ("AfterWells");
}




/////////////////////////////////////////////////
//

WriteFlowDataClass::WriteFlowDataClass(unsigned int saveQueueSize, CommandLineOpts &inception_state, ImageSpecClass &my_image_spec, const RawWells & rawWells)
:queueSize(0),packQueuePtr(NULL),writeQueuePtr(NULL)
{
  queueSize = saveQueueSize;

  if(queueSize > 0){
    size_t flowDepth = inception_state.bkg_control.signal_chunks.save_wells_flow;
    size_t spaceSize = my_image_spec.rows * my_image_spec.cols;

    packQueuePtr = new SemQueue();
    writeQueuePtr = new SemQueue();
    assert(packQueuePtr != NULL && writeQueuePtr != NULL);
    packQueuePtr->init(queueSize);
    writeQueuePtr->init(queueSize);
    stepSize = rawWells.GetStepSize();
    unsigned int bufferSize = stepSize * stepSize * flowDepth;
    for(unsigned int item = 0; item < queueSize; ++item) {
      ChunkFlowData* chunkData = new ChunkFlowData(spaceSize, flowDepth, bufferSize);
      packQueuePtr->enQueue(chunkData);
    }

    filePath = rawWells.GetHdf5FilePath();
    numCols = my_image_spec.cols;
    saveAsUShort = inception_state.sys_context.well_convert;

  }
}

WriteFlowDataClass::~WriteFlowDataClass()
{
  if(packQueuePtr != NULL){
    packQueuePtr->clear();
    delete packQueuePtr;
  }
  if(writeQueuePtr != NULL){
      writeQueuePtr->clear();
      delete writeQueuePtr;
  }
}

void WriteFlowDataClass::InternalThreadFunction()
{

  fprintf ( stdout, "SaveWells: saving thread starts\n");

  H5E_auto2_t old_func;
  void *old_client_data;
  /* Turn off error printing as we're not sure this is actually an hdf5 file. */
  H5Eget_auto2 ( H5E_DEFAULT, &old_func, &old_client_data );
  H5Eset_auto2 ( H5E_DEFAULT, NULL, NULL );
  hid_t hFile = H5Fopen ( filePath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
  if ( hFile < 0 ){
    fprintf ( stdout, "SaveWells: ERROR - failed to open wells file %s\n", filePath.c_str());
    return;
  }

  RWH5DataSet wells;
  wells.mName = "wells";
  wells.mDataset = H5Dopen2 ( hFile, wells.mName.c_str(), H5P_DEFAULT );
  if ( wells.mDataset < 0 ) {
    fprintf ( stdout, "SaveWells: ERROR - failed to open wells dataset from file %s\n", filePath.c_str());
    return ;
  }
  wells.mDatatype  = H5Dget_type ( wells.mDataset );  /* datatype handle */
  wells.mDataspace = H5Dget_space ( wells.mDataset ); /* dataspace handle */
  H5Eset_auto2 ( H5E_DEFAULT, old_func, old_client_data );

  wells.mSaveAsUShort = saveAsUShort;

  RawWellsWriter writer;

  bool quit = false;
  while(!quit) {
    ChunkFlowData* chunkData = (writeQueuePtr)->deQueue();
    if(chunkData == NULL) {
      continue;
    }

    quit = chunkData->lastFlow;

    uint32_t currentRowStart = chunkData->wellChunk.rowStart,
        currentRowEnd = chunkData->wellChunk.rowStart + min ( chunkData->wellChunk.rowHeight, stepSize );
    uint32_t currentColStart = chunkData->wellChunk.colStart,
        currentColEnd = chunkData->wellChunk.colStart + min ( chunkData->wellChunk.colWidth, stepSize );

    for ( currentRowStart = 0, currentRowEnd = stepSize;
        currentRowStart < chunkData->wellChunk.rowStart + chunkData->wellChunk.rowHeight;
        currentRowStart = currentRowEnd, currentRowEnd += stepSize ) {
      currentRowEnd = min ( ( uint32_t ) ( chunkData->wellChunk.rowStart + chunkData->wellChunk.rowHeight ), currentRowEnd );
      for ( currentColStart = 0, currentColEnd = stepSize;
          currentColStart < chunkData->wellChunk.colStart + chunkData->wellChunk.colWidth;
          currentColStart = currentColEnd, currentColEnd += stepSize ) {
        currentColEnd = min ( ( uint32_t ) ( chunkData->wellChunk.colStart + chunkData->wellChunk.colWidth ), currentColEnd );

        chunkData->clearBuffer();
        chunkData->bufferChunk.rowStart = currentRowStart;
        chunkData->bufferChunk.rowHeight = currentRowEnd - currentRowStart;
        chunkData->bufferChunk.colStart = currentColStart;
        chunkData->bufferChunk.colWidth = currentColEnd - currentColStart;
        chunkData->bufferChunk.flowStart = chunkData->wellChunk.flowStart;
        chunkData->bufferChunk.flowDepth = chunkData->wellChunk.flowDepth;

        int idxCount = 0;
        for ( size_t row = currentRowStart; row < currentRowEnd; row++ ) {
          for ( size_t col = currentColStart; col < currentColEnd; col++ ) {
            int idx = row * numCols + col;
            for ( size_t fIx = chunkData->wellChunk.flowStart; fIx < chunkData->wellChunk.flowStart + chunkData->wellChunk.flowDepth; fIx++ ) {
              uint64_t ii = idxCount * chunkData->wellChunk.flowDepth + fIx - chunkData->wellChunk.flowStart;
              uint64_t nn = ( uint64_t ) chunkData->indexes[idx] * chunkData->wellChunk.flowDepth + fIx - chunkData->wellChunk.flowStart;
              chunkData->dsBuffer[ii] = chunkData->flowData[nn];
            }
            idxCount++;
          }
        }
        if(writer.WriteWellsData(wells, chunkData->bufferChunk, chunkData->dsBuffer) < 0 ) {
          ION_ABORT ( "ERROR - Unsuccessful write to HDF5 file: " +
              ToStr ( chunkData->bufferChunk.rowStart ) + "," + ToStr ( chunkData->bufferChunk.colStart ) + "," +
              ToStr ( chunkData->bufferChunk.rowHeight ) + "," + ToStr ( chunkData->bufferChunk.colWidth ) + " x " +
              ToStr ( chunkData->bufferChunk.flowStart ) + "," + ToStr ( chunkData->bufferChunk.flowDepth ));
        }
      }
    }
    (packQueuePtr)->enQueue(chunkData);
  }

  wells.Close();
  if ( hFile != RWH5DataSet::EMPTY ) {
    H5Fclose ( hFile );
    hFile = RWH5DataSet::EMPTY;
  }

  fprintf ( stdout, "SaveWells: saving thread exits\n");

}


bool WriteFlowDataClass::start(){

  if(packQueuePtr == NULL || writeQueuePtr == NULL || queueSize == 0) return false;
  return StartInternalThread();

}

void WriteFlowDataClass::join(){
  JoinInternalThread();
}







