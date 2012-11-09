/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include "Utils.h"
#include <assert.h>

#include "BarCode.h"


bool hackDebug = false;


/*
 * Plain constructor
 */
barcode::barcode ()
{
  barcodes = NULL;
  num_barcodes = 0;
  bc_file_id = NULL;

  keyLen = 0;
  flowOrder = strdup ( "TACG" );
  nFlows = strlen ( flowOrder );
  key = NULL;
  SetKey ( "TCAG", 4 );
  rtbug = false;
  scoreHistPrintMode = 0;

  barcodeMask = NULL;
  bc_score_mode = 0; // percent match
  bc_score_cutoff = 0.9; // match 90% and above
}

/*
 * Destructor
 */
barcode::~barcode ()
{

  delete_barcodes();
  free ( flowOrder );

  if ( key )
    free ( key );
  if ( barcodeMask!=NULL )
    delete barcodeMask;
}


void barcode::SetKey ( char *_key, int len )
{
  if ( key )
    free ( key );

  keyLen = len;
  key = ( char * ) malloc ( len+1 );
  memcpy ( key, _key, len );
  key[len] = 0; // key passed in comes from sff file so we can't assume its NULL-terminated
}

/*
 * Initialize an array of strings to store bar codes
 */
bool barcode::init_barcodes()
{
  delete_barcodes();

  return false;
}

/*
 * Add a barcode to the internal list
 */
bool barcode::add_barcode ( int bcIndex, char *id_str, char *sequence, char *fiveprimeadapter, int type )
{
  int barcodeLen = strlen ( sequence );
  barcodes = ( BarcodeEntry * ) realloc ( barcodes, ( ( num_barcodes+1 ) *sizeof ( BarcodeEntry ) ) );
  barcodes[num_barcodes].barcodeIndex = bcIndex;
  barcodes[num_barcodes].id_str = strdup ( id_str );
  barcodes[num_barcodes].barcode = strdup ( sequence );
  barcodes[num_barcodes].barcodeLen = barcodeLen;
  barcodes[num_barcodes].flowSpaceVec = NULL; // CalculateFlowVector will try and free previously allocated vector so need to set to NULL here to make sure we don't pass in un-initialized mem
  barcodes[num_barcodes].numFlows = 0;
  barcodes[num_barcodes].type = type;
  if ( fiveprimeadapter )
  {
    barcodes[num_barcodes].fiveprimeadapter = strdup ( fiveprimeadapter );
    barcodes[num_barcodes].adapterLen = strlen ( fiveprimeadapter );
  }
  else
  {
    barcodes[num_barcodes].fiveprimeadapter = NULL;
    barcodes[num_barcodes].adapterLen = 0;
  }
  barcodes[num_barcodes].startBase = keyLen;
  barcodes[num_barcodes].endBase = keyLen + barcodes[num_barcodes].barcodeLen - 1;
  barcodes[num_barcodes].flowRes = NULL; // it gets set in CalculateFlowVector
  CalculateFlowVector ( &barcodes[num_barcodes] );
  barcodes[num_barcodes].residual = 0.0;
  barcodes[num_barcodes].resCount = 0;
  num_barcodes += 1;
  if ( rtbug )
  {
    fprintf ( stderr, "%d Added new barcode: '%s' id_str: '%s' vec: ", num_barcodes, barcodes[num_barcodes-1].barcode, barcodes[num_barcodes-1].id_str );
    int flow;
    for ( flow=0;flow<barcodes[num_barcodes-1].numFlows;flow++ )
    {
      fprintf ( stderr, "%d", barcodes[num_barcodes-1].flowSpaceVec[flow] );
      if ( flow == ( barcodes[num_barcodes-1].startFlow-1 ) )
        fprintf ( stderr, " " );
      if ( flow == ( barcodes[num_barcodes-1].endFlow ) )
        fprintf ( stderr, " " );
    }
    fprintf ( stderr, "\n" );
  }
  return false;
}

/*
 * Free memory associated with a bar code list
 */
bool barcode::delete_barcodes()
{
  if ( barcodes )
  {
    for ( int i = 0; i < num_barcodes; i++ )
    {
      free ( barcodes[i].barcode );
      free ( barcodes[i].flowSpaceVec );
      if ( barcodes[i].fiveprimeadapter ) free ( barcodes[i].fiveprimeadapter );
      if ( barcodes[i].flowRes )
        free ( barcodes[i].flowRes );
    }
    free ( barcodes );
  }
  barcodes = NULL;

  num_barcodes = 0;

  if ( bc_file_id ) free ( bc_file_id );
  bc_file_id = NULL;

  return false;
}

void barcode::SetFlowOrder ( char *_flowOrder )
{
  // release memory from prior flow order
  if ( flowOrder )
    free ( flowOrder );

  if ( _flowOrder != NULL )
    flowOrder = strdup ( _flowOrder );
  else
    flowOrder = strdup ( "TACG" );
  nFlows = strlen ( flowOrder );

  // and we need to update the barcode list to reflect the new flow order
  if ( barcodes )
  {
    for ( int i = 0; i < num_barcodes; i++ )
    {
      CalculateFlowVector ( &barcodes[i] );
    }
  }
}

void barcode::SetupScoreHistStrm ( const std::string &fn, int printMode )
{
  if ( fn!="" )
  {
    scoreHistPrintMode = printMode;
    // std::cout << "Print Mode: " << scoreHistPrintMode << ". Score hist. fn: " << fn << "\n";
    scoreHistStrm.exceptions ( std::ofstream::failbit | std::ofstream::badbit );
    try
    {
      scoreHistStrm.open ( fn.c_str() );
    }
    catch ( std::ofstream::failure e )
    {
      std::cerr << "Error trying to open " << fn << "\n";
    }
    scoreHistStrm << "read name\tbc_id\ttotal flow errors\ttotal flows\tscore\n";
  }
  else
  {
    scoreHistPrintMode = 0;  //No file to write to.
  }
}

void barcode::SetupBarcodeMask ( int _w, int _h )
{
  barcodeMask = new Mask ( _w, _h );
  //Reserve highest barcode number (in hex: 0xffff) as unset
  //Will set these later, for particular barcode ids.
  //0 will be set for unclassified barcodes
  barcodeMask->Init ( _w,_h,MaskAll );
  //Same way set up error mask
  barcodeMatchMask = new Mask ( _w,_h );
  barcodeMatchMask->Init ( _w,_h,MaskAll );
}

// Can get coordinates by reading in bead finding mask file (bfmask.bin)
void barcode::SetupBarcodeMask ( const char *bfmask_filename )
{
  std::cerr << "Reading in bead find mask file '" << bfmask_filename << "'. . . ";
  Mask bf ( bfmask_filename,false );
  SetupBarcodeMask ( bf.W(),bf.H() );
  std::cerr << "Completed.\n";
}

void barcode::SetMask ( const char *readname, int barcodeEntryIndex )
{
  int barcodeId;
  assert ( barcodeEntryIndex<num_barcodes );
  if ( barcodeEntryIndex==-1 ) //Note, -1 is ok, reserved for unclassified
  {
    barcodeId=0;
  }
  else   //Get read barcodeId from the index of *barcodes array.
  {
    barcodeId = barcodes[barcodeEntryIndex].barcodeIndex;
  }
  if ( barcodeMask != NULL )
  {
    assert ( barcodeId>=0 && barcodeId < 0xffff ); //check bcID is within masking limits
    int32_t x,y;
    ion_readname_to_xy ( readname, &x, &y );
    barcodeMask->SetBarcodeId ( x,y, ( uint16_t ) barcodeId );
    if ( rtbug ) fprintf ( stderr, "MaskSet:\t%s\t%d\t%d\t%d\n", readname, x, y, barcodeId );
  }
}

void barcode::SetMatchMask ( const char *readname, int errors )
{
  if ( barcodeMatchMask != NULL )
  {
    assert ( errors>=0 && errors < 0xffff ); //check within masking limits
    int32_t x,y;
    ion_readname_to_xy ( readname, &x, &y );
    barcodeMatchMask->SetBarcodeId ( x,y, ( uint16_t ) errors ); //reuses SetBarcodeId function to set the # of mismatches
    if ( rtbug ) fprintf ( stderr, "MatchMaskSet:\t%s\t%d\t%d\t%d\n", readname, x, y, errors );
  }
}

void barcode::OutputRawMask2 ( const char *bcmask_filename )
{
  int rv=barcodeMatchMask->WriteRaw ( bcmask_filename );
  if ( rv != 0 )
  {
    std::cerr << "ERROR: Failed to write barcode mismatch mask. IO Error." << std::endl;
  }
  assert ( rv==0 );
}

void barcode::OutputRawMask ( const char *bcmask_filename )
{
  int rv=barcodeMask->WriteRaw ( bcmask_filename );
  if ( rv != 0 )
  {
    std::cerr << "ERROR: Failed to write barcode ID mask. IO Error." << std::endl;
  }
  assert ( rv==0 );
  //rv=barcodeMask->Export("test",MaskAll);
  //assert(rv==0);

  if ( rtbug )
  {
    std::string newFile ( bcmask_filename );
    newFile += ".debug";
    OutputTextMask ( bcmask_filename,newFile.c_str() );
  }
}

void barcode::OutputTextMask ( const char *bcmask_filename, const char *bcmask_text_out )
{

  std::cerr << "Loading binary mask file '" << bcmask_filename << "'\n";
  Mask recheck ( bcmask_filename ); //check if it read in
  std::ofstream outStrm;
  std::cerr << "Writing text mask file to '" << bcmask_text_out << "'\n";
  outStrm.open ( bcmask_text_out );
  bool isGood = outStrm.good();
  assert ( isGood );
  if ( !isGood )
  {
    std::cerr << "Failure to write file.  Exiting.\n";
    exit ( 1 );
  }
  int h = recheck.H(); // max. y value
  int w = recheck.W(); // max. x value

  //fprintf (fOut, "%d %d\n", w, h);
  outStrm << "#Barcode locations, first row is flowcell's width and height.\n";
  outStrm << "#row\tcol\tvalue\n";
  outStrm << w << "\t" << h << "\t-1\n";
  for ( int row=0; row<h; row++ )  // y
  {
    for ( int col=0; col<w; col++ ) // x
    {
      uint16_t barcodeId = recheck.GetBarcodeId ( col,row );
      if ( barcodeId!=0xffff )   //Don't bother to output empties
      {
        outStrm << row << "\t" << col << "\t" << barcodeId << "\n";
        isGood = isGood & outStrm.good();
      }
    }
  }
  if ( !isGood )
  {
    std::cerr << "Failure to write file.  Exiting.\n";
    exit ( 1 );
  }
  else
  {
    std::cerr << "Completed successfully.\n";
  }
}


void barcode::CalculateFlowVector ( BarcodeEntry *bce )
{

  if ( bce->flowSpaceVec )
    free ( bce->flowSpaceVec );

  memset ( flowVecList, 0, sizeof ( int ) * MAX_FLOW_VEC_SIZE );

  int flow = 0;
  int curBase = 0;

  // build a string from the key + barcode + 5'-adapter - we will want the full flow-space vector from this sequence
  int numBases = keyLen + bce->barcodeLen + bce->adapterLen;
  char *ptr = ( char * ) malloc ( numBases+1 );
  strcpy ( ptr, key );
  strcat ( ptr, bce->barcode );
  if ( bce->adapterLen > 0 )
    strcat ( ptr, bce->fiveprimeadapter );

  bce->startFlow = -1;
  bce->endFlow = -1;

  int baseOffset = 0;
  // MGD - uncommenting the two lines below causes the endFlow to point to the second-to-last base where the last base would match the first insert adapter base - more of a debug thing as we normally want to ensure we match all bases in the barcode
  // if (bce->fiveprimeadapter[0] == bce->barcode[bce->barcodeLen-1])
  // baseOffset = 1;

  while ( curBase < numBases && flow < MAX_FLOW_VEC_SIZE )
  {
    while ( ( ptr[curBase] == flowOrder[flow%nFlows] ) && ( curBase < numBases ) )
    {
      flowVecList[flow]++;
      curBase++;
    }
    // grab the next flow after we sequence through the key, this will be the first flow we will want to count towards barcode matching/scoring, even if its a 0-mer flow
    if ( curBase >= keyLen && bce->startFlow == -1 )
      bce->startFlow = flow+1;
    // grab the last positive incorporating flow for the barcode, any 0-mer flows after this and before the insert or adapter would not be counted in the barcode matching/scoring
    if ( curBase >= ( keyLen+bce->barcodeLen-baseOffset ) && bce->endFlow == -1 )
    {
      bce->endFlow = flow;
    }
    flow++;
  }
  if ( bce->endFlow == -1 )
    bce->endFlow = flow - 1;

  assert ( flow < MAX_FLOW_VEC_SIZE );

  bce->flowSpaceVec = ( int * ) malloc ( sizeof ( int ) * flow );
  memcpy ( bce->flowSpaceVec, flowVecList, sizeof ( int ) * flow );
  bce->numFlows = flow;

  if ( bce->flowRes )
    free ( bce->flowRes );
  bce->flowRes = ( double * ) malloc ( sizeof ( double ) * bce->numFlows );
  memset ( bce->flowRes, 0, sizeof ( double ) * bce->numFlows );

  free ( ptr );
}

/*
 * destructor for the bcmatch struct
 */
bool barcode::bcmatch_destroy ( bcmatch *bcm )
{
  if ( bcm )
  {
    if ( bcm->matching_code )
    {
      free ( bcm->matching_code );
    }
    if ( bcm->id_str )
      free ( bcm->id_str );
  }
  free ( bcm );

  return false;
}

void barcode::BCclose()
{

  delete_barcodes();

}

/*
 * Reads a file containing bar codes
 * Format:
 *
 * file_id identifier_(not_used_for_anything)
 * [score_mode mode]
 *  [score_cutoff value]
 * barcode index,id_str,sequence,fiveprimeadapter,annotation,type(string),length,floworder
 * barcode 0,id_str,TTAACCGG,CTGCTGTACGGCCAAGGCGT,test fragment,text,8,TACG
 * ...
 */
bool barcode::ReadBarCodeFile ( char *filename )
{
  init_barcodes();

  // Open file
  FILE *fd = NULL;
  if ( NULL == ( fd = fopen ( filename, "rb" ) ) )
  {
    fprintf ( stderr, "%s: %s\n", filename, strerror ( errno ) );
    return true;
  }

  // default token indexes to V0 file formats, these will adjust as necessary depending on discovered keywords
  // MGD - in the future, we will parse up a json file and not need to worry about these annoying version changes
  int token_idString = 0;
  int token_barcodeSequence = 1;
  int token_fimeprimeAdapter = 2;

  //Read in barcodes
  char line[1024];
  char *key = NULL;
  char *arg = NULL;
  while ( fgets ( line, sizeof ( line ), fd ) )
  {
    //tokenize at first space
    static const char whitespace[] = " \t\r\n";
    key = strtok ( line, whitespace );
    arg = strtok ( NULL, whitespace );
    if ( rtbug ) fprintf ( stderr, "key = '%s', arg = '%s'\n", key, arg );
    //match token
    if ( strcmp ( key, BC_FILE_FILE_ID ) == 0 )
    {
      bc_file_id = strdup ( arg );
    }
    else if ( strcmp ( key, BC_FILE_SCORE_MODE ) == 0 )
    {
      bc_score_mode = 0;
      if ( arg )
      {
        int ret = sscanf ( arg, "%d", &bc_score_mode );
        if ( ret != 1 )
          bc_score_mode = 0;
      }
      // hack - looks like a V1 file
      token_idString = 1;
      token_barcodeSequence = 2;
      token_fimeprimeAdapter = 3;
    }
    else if ( strcmp ( key, BC_FILE_SCORE_CUTOFF ) == 0 )
    {
      bc_score_cutoff = 0;
      if ( arg )
      {
        int ret = sscanf ( arg, "%lf", &bc_score_cutoff );
        if ( ret != 1 )
          bc_score_cutoff = 0;
      }
      // hack - looks like a V1 file
      token_idString = 1;
      token_barcodeSequence = 2;
      token_fimeprimeAdapter = 3;
    }
    else if ( strcmp ( key, BC_FILE_BARCODE ) == 0 )
    {
      //tokenize arg by comma
      char *ptr = arg; // ptr points to our current token
      int tokenCount = 0;
      char *token[20]; // list of tokens (will actually all just point to locations in the line)
      while ( ptr )
      {
        token[tokenCount] = ptr; // point to the start of the token
        tokenCount++;
        // find the next delimeter
        ptr = strchr ( ptr, ',' );
        // if its not NULL (end of string), make it NULL and advance one char (to start of next token)
        if ( ptr != NULL )
        {
          *ptr = 0;
          ptr++;
        }
      }

      // tokens are:
      //   0 - index
      //   1 - ID string [only with V1 and newer formats]
      //   2 - barcode sequence
      //   3 - fiveprimeadapter
      //   4 - annotation
      //   5 - type
      //   6 - length
      //   7 - floworder

      ToUpper ( token[token_barcodeSequence] );
      ToUpper ( token[token_fimeprimeAdapter] );

      // input validation
      char c = token[token_fimeprimeAdapter][0];
      if ( ( c != 'A' ) && ( c != 'C' ) && ( c != 'G' ) && ( c != 'T' ) )
        token[token_fimeprimeAdapter][0] = 0; // not valid input, so just clear it to NULL

      //index, and make sure it isn't 0 and/or invalid.
      int bcIndex = atoi ( token[0] );
      if ( bcIndex<=0 )
      {
        fprintf ( stderr, "Error, invalid index, %s\n", token[0] );
        assert ( false );
      }
      add_barcode ( bcIndex, token[token_idString], token[token_barcodeSequence], token[token_fimeprimeAdapter], BC_TYPE_NONE );
    }
    else if ( strcmp ( key, BC_FILE_SCORE_MODE ) == 0 )
    {
      //TODO:handle this value
    }
    else if ( strcmp ( key, BC_FILE_SCORE_CUTOFF ) == 0 )
    {
      //TODO:handle this value
    }
    else
    {
      fprintf ( stderr, "Unknown entry: %s\n", key );
    }
  }

  fclose ( fd );
  return false;
}

/*
 * The method to search a given read for a matching bar code
 * Uses strncmp function to generate a match
 * N.B. Returns zero-based offsets into the read.  SFF is 1 based offset.
 */
bcmatch* barcode::exactMatch ( char *bases )
{
  if ( barcodes == NULL )
  {
    fprintf ( stderr, "Error: no barcodes defined.\n" );
    return NULL;
  }

  //DEBUG:
  if ( rtbug )
  {
    unsigned int lmax = 50;
    unsigned int limit = ( strlen ( bases ) < lmax ? strlen ( bases ) :lmax );
    unsigned int i = 0;
    for ( i = 0; i < limit; i++ )
      fprintf ( stdout, "%c", bases[i] );
    if ( i >= lmax ) fprintf ( stdout, "..." );
    fprintf ( stdout, "\n" );
  }

  for ( int i = 0; i < num_barcodes; i++ )
  {
    int len_barcode = strlen ( barcodes[i].barcode );
    int comp = strncmp ( barcodes[i].barcode,bases+keyLen,len_barcode );
    //if (rtbug) fprintf (stderr, "looking to match '%s' in '%s'\n", barcodes[i].barcode, bases+keyLen);
    //fprintf (stderr, "comp val is %d\n", comp);
    if ( comp == 0 )
    {
      bcmatch *bcm = ( bcmatch * ) malloc ( sizeof ( bcmatch ) );
      bcm->matching_code = strdup ( barcodes[i].barcode );
      bcm->bc_left = keyLen;
      bcm->bc_right = keyLen + barcodes[i].barcodeLen;
      bcm->length = barcodes[i].barcodeLen;
      if ( rtbug ) fprintf ( stdout, "%*s\n", bcm->bc_right,barcodes[i].barcode );
      if ( rtbug ) fprintf ( stdout, "%d %d %d\n", bcm->bc_left,bcm->bc_right,bcm->length );
      return bcm;
    }
  }
  if ( rtbug ) fprintf ( stdout, "No match\n" );
  return NULL;
}

/*
 * flowSpaceTrim - finds the closest barcode in flowspace to the sequence passed in,
 * and then trims to exactly the expected flows so it can be error tolerant in base space
 */
bcmatch *barcode::flowSpaceTrim ( unsigned short *flowVals, unsigned short numFlowVals, const char* readName )
{
  if ( barcodes == NULL )
  {
    fprintf ( stderr, "Error: no barcodes defined.\n" );
    return NULL;
  }

  int minErrors = ( int ) bc_score_cutoff; // allows match with this many errors minimum when in bc_score_mode 1

  double bestScore = bc_score_cutoff;

  int totalFlows;

  // find best barcode match
  BarcodeEntry *bce;
  int bestBarcodeIndex = -1;

  std::string scoreHistLineBest;


  if ( hackDebug || rtbug )
  {
    printf ( "Inv: " );
    for ( int i=0;i<numFlowVals;i++ ) printf ( "%d", BASES ( flowVals[i] ) );
    printf ( "\n" );
  }
  if ( rtbug )
    scoreHistPrintMode = 1;

  double old_weightedscore=0.0;
  double old_weightedErr = 0;

  // barcode loop
  for ( int i = 0; i < num_barcodes; i++ )
  {

    bce = &barcodes[i];

    if ( hackDebug )
    {
      printf ( "Trying barcode %s\n", bce->barcode );
    }

    // calculate a score for the barcode comparison
    double totalErrors = 0;
    double weightedErr = 0;
    int flow;

    // lets not try and look at more flows than we were provided!
    int endFlow = bce->endFlow;
    if ( endFlow >= numFlowVals )
      endFlow = numFlowVals-1;

    totalFlows = endFlow - bce->startFlow + 1;

    for ( flow=bce->startFlow;flow<=endFlow;flow++ )
    {

      double delta = bce->flowSpaceVec[flow] - BASES ( flowVals[flow] );
      // AS: for weighted scoring may want to give more weight to accurate 1-mers
      double delta_res = 2*fabs ( flowVals[flow]/100.0 - BASES ( flowVals[flow] ) ); //used for weighted scoring

      if ( delta < 0 ) delta = -delta;
      totalErrors += delta;
      weightedErr += delta* ( 1-delta_res );
    }

    double score = 0.0;
    double weightedscore = 0.0;
    if ( totalFlows > 0 )
    {
      score = 1.0 - ( double ) totalErrors/ ( double ) totalFlows;
      weightedscore = 1.0 - ( double ) weightedErr/ ( double ) totalFlows;
    }

    std::string scoreHistLine = "";
    if ( scoreHistPrintMode )
    {
      std::ostringstream temp;
      if ( readName!=NULL ) temp << readName;
      temp << "\t" << bce->barcodeIndex << "\t" << totalErrors << "\t" << totalFlows << "\t" << score << "\n";
      scoreHistLine = temp.str();
      if ( scoreHistPrintMode>=2 )
      {
        scoreHistStrm << scoreHistLine;
      }
    }

    // see if this score is the best (best starts at minimum so might never hit)
    if ( bc_score_mode == 1 || bc_score_mode == 2 ) // looks at flow-space absolute error counts, not ratios
    {
      if ( totalErrors <= minErrors )
      {
        minErrors = totalErrors;
        bestBarcodeIndex = i;
        old_weightedErr = weightedErr;
        if ( scoreHistPrintMode==1 )
          scoreHistLineBest = scoreHistLine;
      }
      // use weighted error to resolve conflicts
      else if ( ( bc_score_mode == 2 ) && ( totalErrors == minErrors ) && ( bestBarcodeIndex > -1 ) && ( weightedErr < old_weightedErr ) )
      {
        bestBarcodeIndex = i;
        old_weightedErr = weightedErr;
      }
    }
    else   //default score mode
    {
      if ( score > bestScore )
      {
        bestScore = score;
        bestBarcodeIndex = i;
        old_weightedscore = weightedscore;
        if ( scoreHistPrintMode==1 )
          scoreHistLineBest = scoreHistLine;
      }
      // use weighted score to resolve conflicts
      else if ( ( fabs ( bestScore - score ) < 0.000001 ) && ( bestBarcodeIndex > -1 ) && ( weightedscore < old_weightedscore ) )
        bestBarcodeIndex = i;
    }

    if ( hackDebug )
    {
      printf ( "errs: %.2lf  denom: %d  Score: %.4lf\nVec: ", totalErrors, totalFlows, score );
      for ( int i=0;i<=bce->endFlow;i++ )
        printf ( "%d", bce->flowSpaceVec[i] );
      printf ( "\n" );
    }

  }
  // end barcode loop

  // generate the barcode match struct and return to user
  // MGD note - we might just want to have the user pass in a pre-allocated struct or something so we don't thrash mem so bad
  if ( bestBarcodeIndex > -1 )
  {
    bce = &barcodes[bestBarcodeIndex];

    if ( hackDebug )
    {
      printf ( "Found match: %s\n", bce->barcode );
    }
    if ( rtbug )
    {
      std::cout << "i, errs, flows, score: " << scoreHistLineBest;
    }

    // since the match done in flowspace allows for errors, we need to see where in the input read we really want to clip in terms of bases
    // count the number of bases called based on the input flowVals but using the barcode's expected number of flows
    int bases = 0;
    for ( int flow=0;flow<bce->numFlows-1;flow++ )
      bases += BASES ( flowVals[flow] );
    // special-case for the last flow since bases may match library insert as well
    // additional is how many we found minus how many we expected for the barcode end (or 5'-adpater end)
    if ( BASES ( flowVals[bce->numFlows-1] ) > 0 )
    { // if we called at least one base on the last flow, need to make sure we clip correctly
      int additional = BASES ( flowVals[bce->numFlows-1] ) - bce->flowSpaceVec[bce->numFlows-1];
      bases += bce->flowSpaceVec[bce->numFlows-1];
      if ( additional < 0 )
      {
        bases += additional; // remove bases that spilled over into real read
        if ( hackDebug || rtbug )
        {
          printf ( "Subtracted %d bases from clip\n", -additional );
        }
      }
    }

    if ( scoreHistPrintMode==1 ) scoreHistStrm << scoreHistLineBest;

    // keep track of the average residual error in the fit for each barcode type

    int endFlow = bce->endFlow;
    if ( endFlow >= numFlowVals )
      endFlow = numFlowVals-1;

    // filter first - don't count 0 flows
    /*  double residual1 = 0.0;
      int res1count = 0;
      for(int flow=bce->startFlow;flow<=endFlow;flow++) {
       double delta = flowVals[flow]/100.0 - BASES(flowVals[flow]);
       double dsq = delta*delta;
       if (bce->flowSpaceVec[flow] > 0) {
        residual1 += dsq;
        res1count++;
        }
      }
      residual1 = sqrt(residual1/res1count); */
    // if (residual1 > 0.05) {
    // return NULL;
    // }

    // keep track of data for what passes filters
    double residual = 0.0;
    for ( int flow=bce->startFlow;flow<=endFlow;flow++ )
    {
      double delta = flowVals[flow]/100.0 - BASES ( flowVals[flow] );
      double dsq = delta*delta;
      residual += dsq;
      bce->flowRes[flow] += sqrt ( dsq );
    }
    residual = sqrt ( residual/ ( endFlow-bce->startFlow+1 ) );

    bce->residual += residual;
    bce->resCount++;

    //count number of errors in matching barcode
    int errors = 0;
    for ( int flow=bce->startFlow;flow<=endFlow;flow++ )
    {
      double delta = bce->flowSpaceVec[flow] - BASES ( flowVals[flow] );
      if ( delta < 0 ) delta = -delta;
      errors += delta;
    }

    // return the matching barcode to the user
    bcmatch *bcm = ( bcmatch * ) malloc ( sizeof ( bcmatch ) );
    bcm->matching_code = strdup ( bce->barcode ); // MGD - not crazy about allocating mem that caller is responsible for freeing
    bcm->id_str = strdup ( bce->id_str );
    bcm->bc_left = keyLen;
    bcm->bc_right = bases;
    bcm->length = bcm->bc_right - bcm->bc_left + 1; // MGD - no idea what this should be set to, and its not used by the caller in barcodeSplit.cpp
    bcm->errors=errors;
    bcm->id_str = strdup ( bce->id_str );
    return bcm;
  }

  return NULL;
}

void barcode::DumpResiduals()
{
  for ( int i = 0; i < num_barcodes; i++ )
  {
    // printf("%s %d %d\n", barcodes[i].barcode, i+1, barcodes[i].resCount);
    printf ( "Barcode %d %s count: %d res: %.5lf\n", i+1, barcodes[i].barcode, barcodes[i].resCount, barcodes[i].residual/barcodes[i].resCount );
    int base = 0;
    for ( int flow=barcodes[i].startFlow;flow<=barcodes[i].endFlow;flow++ )
    {
      char b = '*';
      if ( barcodes[i].flowSpaceVec[flow] > 0 )
      {
        b = barcodes[i].barcode[base];
        base++;
        while ( b == barcodes[i].barcode[base] )
          base++;
      }
      printf ( "%c: %.4lf ", b, barcodes[i].flowRes[flow]/barcodes[i].resCount );
    }
    printf ( "\n" );
  }
}

char *barcode::readline ( FILE *fd )
{
  char *line = ( char * ) malloc ( 1024 );
  if ( NULL == ( fgets ( line, 1023, fd ) ) )
  {
    free ( line );
    return NULL;
  }
  return line;
}
