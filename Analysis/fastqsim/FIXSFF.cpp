/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
// basic sff reader

#include <stdio.h>
#include <inttypes.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../file-io/ion_util.h"

#include "../ByteSwapUtils.h"

#define MAGIC ('.' | ('s' << 8) | ('f' << 16) | ('f' << 24))

struct CommonHeader {
    uint32_t	magic_number;
    char		version[4];
    uint64_t	index_offset;
    uint32_t	index_length;
    uint32_t	number_of_reads;
    uint16_t	header_length;
    uint16_t	key_length;
    uint16_t	number_of_flows_per_read;
    uint8_t		flowgram_format_code;
};

char		*flow_chars;
char		*key_sequence;

struct ReadHeader {
    uint16_t	read_header_length;
    uint16_t	name_length;
    uint32_t	number_of_bases;
    uint16_t	clip_qual_left;
    uint16_t	clip_qual_right;
    uint16_t	clip_adapter_left;
    uint16_t	clip_adapter_right;
};
char		name[256];
uint16_t	*flowgram_values; // [NUMBER_OF_FLOWS_PER_READ];
uint8_t		*flow_index_per_base; // * number_of_bases;
char		*bases; // * number_of_bases;
uint8_t		*quality_scores; // * number_of_bases;

char QualToFastQ(int q)
{
  // from http://maq.sourceforge.net/fastq.shtml
  char c = (q <= 93 ? q : 93) + 33;
  return c;
}

int FastQToQual(char c)
{
  return c - 33;
}

static char *hackkey = "TCAG";
static int hackkeylen = 4;

int main(int argc, char *argv[])
{
  char	*fastqFileName = NULL;
  char	*sffFileName = NULL;
  bool	forceClip = false;
  bool	keyPass = false;
  bool	allReads = false;
  int	minReadLen = 8; // min read length after key-pass that we will write out to fastq file
  int	readCol = -1;
  int	readRow = -1;
  bool	findRead = false;
  int row, col;

  // process command-line args
  int argcc = 1;
  while (argcc < argc) {
      if (argv[argcc][0] == '-') {
          switch (argv[argcc][1]) {
            case 'a': // output all reads
              allReads = true;
              break;

            case 'R': // report read at row & column
              argcc++;	
              readRow = atoi(argv[argcc]);
              break;

            case 'C': // report read at row & column
              argcc++;	
              readCol = atoi(argv[argcc]);
              break;

            case 'q': // convert to fastq
              argcc++;
              fastqFileName = argv[argcc];
              break;

            case 'c': // force qual clip left to 5
              forceClip = true;
              break;

            case 'k': // force keypass
              keyPass = true;
              argcc++;
              hackkey = argv[argcc];
              hackkeylen = strlen(hackkey);
              break;

            case 'l': // set min readlength for fastq file output filter
              argcc++;
              minReadLen = atoi(argv[argcc]);
              break;

            default:
              //sffFileName = argv[argcc];
              break;
          }
      }
      else {
          sffFileName = argv[argcc];
      }
      argcc++;
  }

  if (!sffFileName) {
      printf("Usage: SFFRead [args] sffFile.sff\n");
      exit(0);
  }

  if (readCol > -1 && readRow > -1) {
      findRead = true;
      allReads = true;// makes it search all reads
  }

  FILE *fp;
  fp = fopen(sffFileName, "r+");
  if (fp) {
      if (!findRead && !fastqFileName)
        printf("Reading file: %s\n", sffFileName);
      CommonHeader h;

      // Fix the flow_format_code problem: make sure it is set to 1
      fpos_t p, start;

      fgetpos (fp, &p);
      fgetpos (fp, &start);
      start = p;
      int elements_read = fread(&h, 31, 1, fp);
      assert(elements_read == 1);
      h.flowgram_format_code = 1;
      fsetpos (fp, &p);
      fwrite (&h, 31, 1, fp);
      fsetpos (fp, &p);

      elements_read = fread(&h, 31, 1, fp);
      assert(elements_read == 1);
      ByteSwap8(h.index_offset);
      ByteSwap4(h.index_length);
      ByteSwap4(h.number_of_reads);
      ByteSwap2(h.header_length);
      ByteSwap2(h.key_length);
      ByteSwap2(h.number_of_flows_per_read);
      if (!findRead && !fastqFileName) {
          printf("Magic:	%u	%s\n", h.magic_number, (h.magic_number == MAGIC ? "Yes" : "No"));
          printf("Version: %d%d%d%d\n", h.version[0], h.version[1], h.version[2], h.version[3]);
          printf("Index offset: %lu  length: %u\n", h.index_offset, h.index_length);
          printf("Number of reads: %u\n", h.number_of_reads);
          printf("Header length: %hu\n", h.header_length);
          printf("Key length: %u\n", h.key_length);
          printf("Flows per read: %hu\n", h.number_of_flows_per_read);
          printf("Flowgram format: %hhu\n", h.flowgram_format_code);
      }

      flow_chars = (char *)malloc(h.number_of_flows_per_read);
      key_sequence = (char *)malloc(h.key_length);
      elements_read = fread(flow_chars, h.number_of_flows_per_read, 1, fp);
      assert(elements_read == 1);
      elements_read = fread(key_sequence, h.key_length, 1, fp);
      assert(elements_read == 1);
      int i;
      if (!findRead && !fastqFileName) {
          printf("Key sequence: ");
          for(i=0;i<h.key_length;i++)
            printf("%c", key_sequence[i]);

          printf("\nFlow chars:\n");
          for(i=0;i<h.number_of_flows_per_read;i++)
            printf("%c", flow_chars[i]);
          printf("\n");
      }
      int padBytes = (8-((31 + h.number_of_flows_per_read + h.key_length) & 7));
      char padData[8];
      //		fprintf (stdout, "Pad Bytes = %d\n", padBytes);
      elements_read = fread(padData, padBytes, 1, fp);
      assert(elements_read == 1);

      fgetpos(fp, &p);
      //		fprintf (stdout, "We are at %ld\n", (p.__pos - start.__pos));

      // -- read the reads
      int numReads = h.number_of_reads;

      // pre-allocate space so we be fast
      flowgram_values = (uint16_t *)malloc(sizeof(uint16_t) * h.number_of_flows_per_read);
      int maxBases = h.number_of_flows_per_read * 10; // problems if ever a 10-mer hits every flow!
      flow_index_per_base = (uint8_t *)malloc(sizeof(uint8_t) * maxBases);
      bases = (char *)malloc(maxBases);
      quality_scores = (uint8_t *)malloc(sizeof(uint8_t) * maxBases);

      for(i=0;i<numReads;i++) {
          ReadHeader r;

#define FIXIT
#ifdef FIXIT
          fpos_t pos;
          // Get position ready-to-read header
          fgetpos (fp, &pos);
          // Read header
          elements_read = fread(&r, 16,  1, fp);
          assert(elements_read == 1);
          // byte swap
          ByteSwap2(r.read_header_length);
          ByteSwap2(r.name_length);
          //			fprintf (stdout, "Old read header length = %d\n", r.read_header_length);
          // Fix the read_header_length
          // read_header_length is "16 + name_length" rounded up to nearest divisible by 8
          r.read_header_length = 16 + r.name_length;
          r.read_header_length += (8 - (r.read_header_length & 0x7)) & 0x7;
          //			fprintf (stdout, "New read header length = %d\n", r.read_header_length);
          // Byte swap
          ByteSwap2(r.read_header_length);
          ByteSwap2(r.name_length);
          // Rewind file pointer
          fsetpos (fp, &pos);
          // Write header out again
          fwrite (&r, 16, 1, fp);
          // Rewind again
          fsetpos (fp, &pos);
#endif

          // Read it in and continue
          fpos_t readStart;
          fgetpos(fp, &readStart);
          elements_read = fread(&r, 16,  1, fp);
          assert(elements_read == 1);

          //ByteSwap2(r.read_header_length);
          //ByteSwap2(r.name_length);
          ByteSwap2(r.clip_qual_left);
          ByteSwap2(r.clip_qual_right);
          ByteSwap2(r.clip_adapter_left);
          ByteSwap2(r.clip_adapter_right);

          // Fix clipping values
          r.clip_qual_left = 5;
          r.clip_adapter_left = 0;
          r.clip_qual_right = 0;
          r.clip_adapter_right = 0;
          ByteSwap2(r.clip_qual_left);
          ByteSwap2(r.clip_qual_right);
          ByteSwap2(r.clip_adapter_left);
          ByteSwap2(r.clip_adapter_right);

          // Rewind file pointer to beginning of read header
          fsetpos (fp, &pos);
          // Write read header
          fwrite (&r, 16, 1, fp);
          // Rewind file pointer to beginning of read header
          fsetpos (fp, &pos);
          // Read corrected header
          elements_read = fread(&r, 16,  1, fp);
          assert(elements_read == 1);

          ByteSwap2(r.read_header_length);
          ByteSwap4(r.number_of_bases);
          ByteSwap2(r.name_length);
          ByteSwap2(r.clip_qual_left);
          ByteSwap2(r.clip_qual_right);
          ByteSwap2(r.clip_adapter_left);
          ByteSwap2(r.clip_adapter_right);

          //printf("Read header length: %d\n", r.read_header_length);
          //printf("Read name length: %d\n", r.name_length);

          /*
             flow_index_per_base = (uint8_t *)malloc(sizeof(uint8_t) * r.number_of_bases);
             bases = (char *)malloc(r.number_of_bases);
             quality_scores = (uint8_t *)malloc(sizeof(uint8_t) * r.number_of_bases);
             */

          if (r.name_length > 0) {
              elements_read = fread(name, r.name_length, 1, fp);
              assert(elements_read == 1);
              name[r.name_length] = 0; // so we can easily print it
          }
          if(1 != ion_readname_to_rowcol(name, &row, &col)) {
              fprintf (stderr, "Error parsing read name: '%s'\n", name);
              continue;
          }

          int readPadLength = ((8 - ((16 + r.name_length) & 7)))%8;
          elements_read = fread(padData, readPadLength, 1, fp);
          assert(elements_read == 1);
          /*
             printf("Read: %s (r%d|c%d) has %d bases\n",
             (r.name_length > 0 ? name : "NONAME"),
             row, col,
             r.number_of_bases);
             printf("Clip left: %d qual: %d right: %d qual: %d\n",
             r.clip_adapter_left, r.clip_qual_left,
             r.clip_adapter_right, r.clip_qual_right);
             printf("Flowgram values:\n");
             */
          elements_read = fread(flowgram_values, h.number_of_flows_per_read, sizeof(uint16_t), fp);
          assert(elements_read == sizeof(uint16_t));
          elements_read = fread(flow_index_per_base, r.number_of_bases, sizeof(uint8_t), fp);
          assert(elements_read == sizeof(uint8_t));
          elements_read = fread(bases, r.number_of_bases, 1, fp);
          assert(elements_read == 1);
          elements_read = fread(quality_scores, r.number_of_bases, sizeof(uint8_t), fp);

          int bytesRead = h.number_of_flows_per_read * sizeof(uint16_t) + 3 * r.number_of_bases;
          readPadLength = (8 - (bytesRead & 7))%8;
          elements_read = fread(padData, readPadLength, 1, fp);
          assert(elements_read == 1);
          fpos_t readEnd;
          fgetpos(fp, &readEnd);
          //			fprintf (stdout, "At end of read. Size: %ld\n", readEnd.__pos-readStart.__pos);
          //if ((readEnd.__pos-readStart.__pos) != r.read_header_length) {
          //	fprintf (stdout, "mismatch in read_header_length\n");
          //	exit (1);
          //}

          // parse the name to get the row & col, if matched, print out read
          if(1 != ion_readname_to_rowcol(name, &row, &col)) {
              fprintf (stderr, "Error parsing read name: '%s'\n", name);
              continue;
          }

          if (row == readRow && col == readCol) {
              //printf("Ionogram: ");
              int i;
              for(i=0;i<h.number_of_flows_per_read;i++) {
                  printf("%.2lf ", (double)(ByteSwap2(flowgram_values[i]))/100.0);
              }
              printf("\n");
          }
          /*
             int f;
             for(f=0;f<h.number_of_flows_per_read;f++)
             printf("%d ", ByteSwap2(flowgram_values[f]));
             printf("\nFlow index per base:\n");
             unsigned int b;
             for(b=0;b<r.number_of_bases;b++)
             printf("%d ", flow_index_per_base[b]);
             printf("\nBases called:\n");
             for(b=0;b<r.number_of_bases;b++)
             printf("%c", bases[b]);
             printf("\nQuality scores:\n");
             for(b=0;b<r.number_of_bases;b++)
             printf("%d ", quality_scores[b]);
             printf("\nDone with this read\n\n");
             */

          /*
             if (name) free(name);
             free(flowgram_values);
             free(flow_index_per_base);
             free(bases);
             free(quality_scores);
             */
      }
      free(flowgram_values);
      free(flow_index_per_base);
      free(bases);
      free(quality_scores);


      fclose(fp);
  }

  return 0;
}
