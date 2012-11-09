/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BARCODE_H
#define BARCODE_H

/*
 * Valid keywords in barcodeList.txt file
 */
#define BC_FILE_FILE_ID   "file_id"
#define BC_FILE_BARCODE   "barcode"
#define BC_FILE_SCORE_MODE  "score_mode"
#define BC_FILE_SCORE_CUTOFF "score_cutoff"

/*
 * Valid values for barcodeEntry.type
 */
#define BC_TYPE_NONE 0

#include <iostream>
#include <sstream>
#include <fstream>

#include "Mask.h"
#include "file-io/ion_util.h"
/*
 * Set of variables describing a bar code match to a read
 */
typedef struct
{
  char *id_str;   // this sequence's unique identification
  char *matching_code; //bar code that matches
  unsigned int bc_left;
  unsigned int bc_right;
  unsigned int length;
  int errors;

} bcmatch;

struct BarcodeEntry
{
  int barcodeIndex;
  char *barcode; // barcode base sequence
  char *id_str; // barcode unique ID
  int barcodeLen; // length of the barcode base sequence
  int *flowSpaceVec; // flow-space vector representation for the barcode
  int numFlows; // number of flows for the flow-space representation, includes 5' adapter
  unsigned int type; // representation of which match algorithm to use
  char *fiveprimeadapter; //internal fiveprimeadapter sequence
  int adapterLen; // length of the adapter string
  int startBase; // start & end base we use for scoring
  int endBase; // start and end arer inclusive, so 4,4 means its only one base we care about
  int startFlow; // calculated from the start base & end base, used for scoring/matching
  int endFlow;
  double residual; // cumulative residual errors from all sequences matching this barcode
  int resCount; // count of how many sequences matched this barcode
  double *flowRes; // cumulative per-flow residual
};

#define MAX_FLOW_VEC_SIZE 1000
#define BASES(val) ((int)((val)/100.0+0.5))

class barcode
{
public:
  barcode ();
  ~barcode ();
  void BCclose();
  bool ReadBarCodeFile ( char *filename );
  bcmatch *exactMatch ( char *bases );
  // flowSpaceTrim - finds the closest barcode in flowspace to the sequence passed in, and then trims to exactly the expected flows so it can be error tolerant in base space
  // minScore can be set to alter the default behavior of the matching
  bcmatch *flowSpaceTrim ( unsigned short *flowVals, unsigned short numFlowVals, const char *readName = NULL );
  bool bcmatch_destroy ( bcmatch *bcm );
  int GetNumBarcodes() {return num_barcodes;}
  char *GetBarcode ( int index ) {if ( barcodes ) return barcodes[index].barcode; else return NULL;}
  char *GetBarcodeId ( int index ) {if ( barcodes ) return barcodes[index].id_str; else return NULL;}
  void SetKey ( char *key, int keyLen );
  void SetFlowOrder ( char *flowOrder );
  void SetRTDebug ( bool _flag ) {rtbug = _flag;}
  void SetupScoreHistStrm ( const std::string &fn, int printMode=0 );
  void SetupBarcodeMask ( int _w, int _h );
  void SetupBarcodeMask ( const char *bfmask_filename );
  void SetMask ( const char *readname, int barcodeEntryIndex );
  void SetMatchMask ( const char *readname, int errors );
  void OutputRawMask ( const char *bcmask_filename );
  void OutputRawMask2 ( const char *bcmask_filename );
  void DumpResiduals();
  int GetScoreMode() {return bc_score_mode;}
  void SetScoreMode ( int mode ) {bc_score_mode = mode;}
  double GetScoreCutoff() {return bc_score_cutoff;}
  void SetScoreCutoff ( double val ) {bc_score_cutoff = val;}
  static void OutputTextMask ( const char *bcmask_filename, const char *bcmask_text_out );
protected:

  void CalculateFlowVector ( BarcodeEntry *barcodeEntry );

private:

  // List of strings which are bar codes
  BarcodeEntry *barcodes;
  bool delete_barcodes();
  bool init_barcodes();
  bool add_barcode ( int bcIndex, char *id_str, char *barcode, char *fiveprimeadapter, int type );
  int num_barcodes;
  char *key;
  int keyLen;
  char *bc_file_id;
  int bc_score_mode;
  double bc_score_cutoff;
  char *readline ( FILE *fd );
  char *flowOrder;
  int nFlows; // number of flows in flowOrder
  int flowVecList[MAX_FLOW_VEC_SIZE]; // work buffer
  bool rtbug;  // debug print flag
  std::ofstream scoreHistStrm;
  int scoreHistPrintMode;  //0 - off, 1 - just best bc, 2 - all bc
  Mask *barcodeMask;
  Mask *barcodeMatchMask;
};

#endif // BARCODE_H
