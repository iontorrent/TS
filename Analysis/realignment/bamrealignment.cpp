/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#include <list>
#include <time.h>
#include "OptArgs.h"
#include "Realigner.h"


using namespace std;
using namespace BamTools;


int PrintHelp()
{
  printf ("local realignment of reads in a mapped BAM file.\n");
  printf ("Usage: bamrealignment [options]\n");
  printf ("\n");
  printf ("Required Arguments:\n");
  printf ("  -i,--input                  FILE       mapped input BAM file\n");
  printf ("  -o,--output                 FILE       locally realigned mapped BAM file\n");
  printf ("Arguments with default values:\n");
  printf ("  -f,--format    (def. 1)    [0-1]       output format: 0 - compressed BAM, 1 - uncompressed BAM\n");
  printf ("  -t,--threads   (def. 8)     INT        number of threads used by bam writer to do compression.\n");
  printf ("  -s,--scores           INT,INT,INT,INT  scores for match, mismatch, gap open, gap extend\n");
  printf ("                 (def. 4,-6,-5,-2)\n");
  printf ("  -c,--clipping  (def. 2)    [0-4]       sets read clipping\n");
  printf ("                               0         no clipping at all\n");
  printf ("                               1         semi-global: read can start and end anywhere in ref.\n");
  printf ("                               2         semi-global + soft clip bead end of read\n");
  printf ("                               3         semi-global + soft clip key end of read\n");
  printf ("                               4         semi-global + soft clip both ends\n");
  printf ("  -a,--anchors   (def. true)  BOOL       reduce matching anchors at the ends to `bandwidth` bases\n");
  printf ("  -b,--bandwidth (def. 10)    INT        diagonal bandwidth for tubed alignment\n");
  printf ("  -v,--verbose   (def. false) BOOL       print alignment information for each read\n");
  printf ("-------------------------------------------\n");

  return 1;
}


int main (int argc, const char *argv[])
{
  printf ("------------- bamrealignment --------------\n");

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  vector<int> score_vals(4);

  string input_bam  = opts.GetFirstString  ('i', "input", "");
  string output_bam = opts.GetFirstString  ('o', "output", "");
  opts.GetOption(score_vals, "4,-6,-5,-2", 's', "scores");
  int    clipping   = opts.GetFirstInt     ('c', "clipping", 2);
  bool   anchors    = opts.GetFirstBoolean ('a', "anchors", true);
  int    bandwidth  = opts.GetFirstInt     ('b', "bandwidth", 10);
  bool   verbose    = opts.GetFirstBoolean ('v', "verbose", false);
  bool   debug      = opts.GetFirstBoolean ('d', "debug", false);
  int    format     = opts.GetFirstInt     ('f', "format", 1);
  int  num_threads  = opts.GetFirstInt     ('t', "threads", 8);


  if (input_bam.empty() or output_bam.empty())
    return PrintHelp();

  opts.CheckNoLeftovers();


  BamReader reader;
  if (!reader.Open(input_bam)) {
    fprintf(stderr, "bamrealignment: Failed to open input file %s\n", input_bam.c_str());
    return 1;
  }

  SamHeader header = reader.GetHeader();
  RefVector refs   = reader.GetReferenceData();

  BamWriter writer;
  writer.SetNumThreads(num_threads);
  if (format == 1)
    writer.SetCompressionMode(BamWriter::Uncompressed);
  else
    writer.SetCompressionMode(BamWriter::Compressed);

  if (!writer.Open(output_bam, header, refs)) {
    fprintf(stderr, "bamrealignment: Failed to open output file %s\n", output_bam.c_str());
    return 1;
  }


  // The meat starts here ------------------------------------

  if (verbose)
    cout << "Verbose option is activated, each alignment will print to screen." << endl
         << "  After a read hit RETURN to continue to the next one," << endl
         << "  or press q RETURN to quit the program," << endl
         << "  or press s Return to silence verbose," << endl
         << "  or press c RETURN to continue printing without further prompt." << endl << endl;

  unsigned int readcounter = 0;
  unsigned int mapped_readcounter = 0;
  unsigned int realigned_readcounter = 0;
  unsigned int pos_update_readcounter = 0;
  unsigned int start_position_shift;
  int new_position;

  string  md_tag, new_md_tag, input = "x";
  vector<CigarOp>    new_cigar_data;
  vector<MDelement>  new_md_data;
  bool position_shift = false;
  time_t start_time = time(NULL);

  Realigner aligner;
  aligner.verbose_ = verbose;
  aligner.debug_   = debug;
  if (!aligner.SetScores(score_vals))
    cout << "bamrealignment: Four scores need to be provided: match, mismatch, gap open, gap extend score!" << endl;

  aligner.SetAlignmentBandwidth(bandwidth);

  BamAlignment alignment;
  while(reader.GetNextAlignment(alignment)){
    readcounter ++;
    
    /*if(debug) {
      cout << alignment.Name << endl;
      if (alignment.Name.compare("8HDVJ:00239:00175") == 0)
	aligner.verbose_ = true;
    }*/
    
    if ( (readcounter % 100000) == 0 )
       cout << "Processed " << readcounter << " reads. Elapsed time: " << (time(NULL) - start_time) << endl;

    if (alignment.IsMapped()) {

      mapped_readcounter++;
      aligner.SetClipping(clipping);
      aligner.SetStrand(!alignment.IsReverseStrand());
      if (aligner.verbose_) {
    	cout << endl;
        if (alignment.IsReverseStrand())
          cout << "The read is from the reverse strand." << endl;
        else
          cout << "The read is from the forward strand." << endl;
      }

      if (!alignment.GetTag("MD", md_tag)) {
    	if (aligner.verbose_)
          cout << "Warning: Skipping read " << alignment.Name << ". It is mapped but missing MD tag." << endl;
      } else if (aligner.CreateRefFromQueryBases(alignment.QueryBases, alignment.CigarData, md_tag, anchors)) {

        if (!aligner.computeSWalignment(new_cigar_data, new_md_data, start_position_shift)) {
          if (aligner.verbose_)
            cout << "Error in the alignment! Not updating read information." << endl;
          writer.SaveAlignment(alignment);  // Write alignment unchanged
          continue;
        }
        

        if (!aligner.addClippedBasesToTags(new_cigar_data, new_md_data, alignment.QueryBases.size())) {
          if (aligner.verbose_)
            cout << "Error when adding clipped anchors back to tags! Not updating read information." << endl;
          writer.SaveAlignment(alignment);  // Write alignment unchanged
          continue;
        }
        new_md_tag = aligner.GetMDstring(new_md_data);
        realigned_readcounter++;

        // adjust start position of read
        if (!aligner.LeftAnchorClipped() and start_position_shift != 0) {
          new_position = aligner.updateReadPosition(alignment.CigarData, (int)start_position_shift, alignment.Position);
          if (new_position != alignment.Position) {
            pos_update_readcounter++;
            position_shift = true;
            alignment.Position = new_position;
          }
        }


        if (aligner.verbose_){
          cout << alignment.Name << endl;
          cout << "------------------------------------------" << endl;
          // Wait for input to continue or quit program
          if (input.size() == 0)
            input = 'x';
          else if (input[0] != 'c' and input[0] != 'C')
            getline(cin, input);
          if (input.size()>0){
            if (input[0] == 'q' or input[0] == 'Q')
              return 1;
            else if (input[0] == 's' or input[0] == 'S')
              aligner.verbose_ = false;
          }
        }

        // Finally update alignment information
        alignment.CigarData = new_cigar_data;
        alignment.EditTag("MD", "Z" , new_md_tag);

      } // end of CreateRef else if
      else if (aligner.verbose_) {
        cout << alignment.Name << endl;
        cout << "------------------------------------------" << endl;
        // Wait for input to continue or quit program
        if (input.size() == 0)
          input = 'x';
        else if (input[0] != 'c' and input[0] != 'C')
          getline(cin, input);
        if (input.size()>0){
          if (input[0] == 'q' or input[0] == 'Q')
            return 1;
          else if (input[0] == 's' or input[0] == 'S')
            aligner.verbose_ = false;
        }
      }
    } // end of if isMapped

    writer.SaveAlignment(alignment);

  } // end while loop over reads

  if (aligner.invalid_cigar_in_input)
    cerr << "WARNING bamrealignment: There were invalid cigar string / md tag pairs in the input bam file." << endl;

  // ----------------------------------------------------------------
  // program end -- output summary information
  cout << "File " << input_bam << " contained " << readcounter << " reads; "
       << mapped_readcounter << " of which where mapped." << endl
       << "Realigned " << realigned_readcounter << " reads and " << pos_update_readcounter << " changed their start position."
       << endl << "Processing time: " << (time(NULL)-start_time) << " seconds." << endl;
    cout << "INFO: The output BAM file may be unsorted." << endl;
  cout << "------------------------------------------" << endl;
  return 0;
}

//




