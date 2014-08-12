/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <numeric>
#include "OptArgs.h"
#include "H5Replay.h"

#define MAXTHREADS 16
#define SIGNAL_IN "wells"
#define SIGNAL_OUT "wells"
#define H5FILE_IN "./1.wells"
#define H5FILE_OUT "./extract.h5"

#define At(x) at((x))

using namespace std;

void usage() {
  cout << "ExtractWells - Program for extracting given wells values in default dataset wells" << endl;
  cout << "Outputs dataset /wells to the output h5file";
  cout << "" << endl;
  cout << "Usage:" << endl;
  cout << "   ExtractWells [--source wells] [--input h5file] -" << endl;
  cout << "   ExtractWells [--source wells] [--input h5file] --positions posfile.txt" << endl;
  cout << "Options:" << endl;
  cout << "   -i,--input         h5file to read from (default " << H5FILE_IN << ")" << endl;
  cout << "   -s,--source         dataset to read from (default " << SIGNAL_IN << ")" << endl;
  cout << "   -o,--output    h5file to write to (default " << H5FILE_OUT << ")" << endl;
  cout << "   -d,--destination    h5file to write to (default " << SIGNAL_OUT << ")" << endl;
  cout << "   -f,--flowlimit      number of flows to use (default = number flows in input h5file)" << endl;
  cout << "   -p,--positions      file of tab-delimited 0-based integer row,col positions, 1 per line" << endl;
  cout << " if -p is omitted use stdin stream of tab-delimited 0-based integer row,col positions, 1 per line" << endl;
  cout << "   -h,--help      this message." << endl;
  exit(1);
}

struct thread_args {
  hsize_t row;
  hsize_t col;
  vector<hsize_t> *chunks;
  vector<hsize_t> *chunks_out;
  vector<hsize_t> *dims;
  string *h5file_in;
  string *source;
  string *h5file_out;
  string *destination;
  hsize_t offset_out;
  hsize_t count_out;
  vector<size_t> *input_positions;
  int thread_id;
  unsigned int flowlimit;
};

size_t RowColToIndex(size_t row, size_t col, size_t height, size_t width){
  if (row >= height) {
    fprintf(stderr, "0-indexed row=%lu must be < height=%lu\n", row, height);
    fprintf(stderr, "0-indexed col=%lu must be < width=%lu\n", col, width);
    exit(1);
  }
  if (col >= width) {
    fprintf(stderr, "0-indexed row=%lu must be < height=%lu\n", row, height);
    fprintf(stderr, "0-indexed col=%lu must be < width=%lu\n", col, width);
    exit(1);
  }
  return ( row*width + col );
}

size_t RowColToIndex1(size_t row, size_t col, size_t height, size_t width){
  if (row >= height) {
    fprintf(stderr, "in thread 0-indexed row=%lu must be < height=%lu\n", row, height);
    exit(1);
  }
  if (col >= width) {
    fprintf(stderr, "in thread 0-indexed col=%lu must be < width=%lu\n", col, width);
    exit(1);
  }
  return ( row*width + col );
}

void DumpStartingStateOfExtractWells (int argc, const char *argv[])
{
  char myHostName[128] = { 0 };
  gethostname (myHostName, 128);
  fprintf (stdout, "\n");
  fprintf (stdout, "Hostname = %s\n", myHostName);
  time_t rawtime;
  time(&rawtime);
  fprintf (stdout, "Start Time = %s", ctime (&rawtime));
  fprintf (stdout, "Version = %s-%s (%s) (%s)\n",
           IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
           IonVersion::GetSvnRev().c_str(), IonVersion::GetBuildNum().c_str());
  fprintf (stdout, "Command line = ");
  for (int i = 0; i < argc; i++)
    fprintf (stdout, "%s ", argv[i]);
  fprintf (stdout, "\n");
  fflush (NULL);
}

vector<unsigned int> thread_flags;

void *do_subset(void *arguments);

int main(int argc, const char *argv[]) {
  OptArgs opts;
  string position_file;
  string h5file_in;
  string source;
  string h5file_out;
  string destination;
  string positions_file;
  bool help;
  string flowlimit_arg;
  unsigned int flowlimit;
  vector<string>otherArgs;

  DumpStartingStateOfExtractWells (argc,argv);

  opts.ParseCmdLine(argc, argv);
  opts.GetOption(h5file_in, "", 'i', "input");
  opts.GetOption(source, "", 's', "source");
  opts.GetOption(h5file_out, "", 'o', "output");
  opts.GetOption(destination, "", 'd', "destination");
  opts.GetOption(flowlimit_arg, "", 'f', "flowlimit");
  opts.GetOption(positions_file, "", 'p', "positions");
  opts.GetOption(help, "false", 'h', "help");
  opts.GetLeftoverArguments(otherArgs);

  // input data processing
  string line;
  vector<size_t> row_val;
  vector<size_t> col_val;
  ifstream filestream;
  if ( ! positions_file.empty() )
    filestream.open(&positions_file.At(0));
  istream &input = ( filestream.is_open() ) ? filestream : cin;


		      
  while ( getline(input, line) )
  {
    int num = -1;
    vector<size_t> ints;
    istringstream ss(line);
    while ( ss >> num && ints.size() < 2 ) {
      if (num < 0) {
	fprintf(stderr, "Found negative integer %d\n", num);
	exit(-1);
      }
      else
	ints.push_back((size_t)num);
    }
    if (ints.size() != 2) {
      fprintf(stderr, "Found %d integers in %s, expected 2\n", (int)ints.size(), &line[0]);
      continue;
    }
    row_val.push_back(ints.at(0));
    col_val.push_back(ints.at(1));
  }
  if (row_val.size() == 0 ) {
      fprintf(stdout, "No positions to extract, check input\n");
      exit(0);
  }    
  vector<size_t>input_positions(row_val.size(), 0);

  int numCPU = (int)sysconf( _SC_NPROCESSORS_ONLN );
  int numThreads = MAXTHREADS < numCPU ? MAXTHREADS : numCPU;
  fprintf(stdout, "Using %d threads of %d cores\n", numThreads, numCPU);

  if (source.empty())
    source = source + SIGNAL_IN;
  H5ReplayReader reader = H5ReplayReader(h5file_in, &source[0]);
  if ( h5file_out.empty() )
    h5file_out = h5file_out + H5FILE_OUT;
  if ( destination.empty() )
    destination = destination + SIGNAL_OUT;

  reader.Open();
  int rank = reader.GetRank();
  vector<hsize_t>dims(rank);
  vector<hsize_t>chunks(rank);
  reader.GetDims(dims);
  reader.GetChunkSize(chunks);
  reader.Close();

  // convert input row, col positions to indices
  for (hsize_t i=0; i<input_positions.size(); i++)
    input_positions.At(i) = RowColToIndex(row_val.At(i), col_val.At(i), dims.At(0), dims.At(1));
  sort(input_positions.begin(), input_positions.end());

  fprintf(stdout, "Opened for read %s:%s with rank %d, row x col x flow dims=[ ", &h5file_in[0], &source[0], rank);
  for (int i=0; i<rank; i++)
    fprintf(stdout, "%d ", (int)dims.At(i));
  fprintf(stdout, "], chunksize=[ ");
  for (int i=0; i<rank; i++)
    fprintf(stdout, "%d ", (int)chunks.At(i));
  fprintf(stdout, "]\n");

  
  H5ReplayRecorder recorder = H5ReplayRecorder(h5file_out, &destination[0],reader.GetType(),2);
  recorder.CreateFile();


  {
    vector<hsize_t> dims_pos(1, input_positions.size());
    string pos_name = "position";
    H5ReplayRecorder recorder_pos = H5ReplayRecorder(h5file_out, &pos_name[0],H5T_NATIVE_ULONG,1);
    recorder_pos.CreateDataset(dims_pos);
  }

  {
    string chip_dims = "chip_dims";
    H5ReplayRecorder recorder_chip_dims = H5ReplayRecorder(h5file_out, &chip_dims[0],H5T_NATIVE_ULLONG,1);
    vector<hsize_t> offset_dims(1,0);
    vector<hsize_t> count_dims(1,3);
    recorder_chip_dims.CreateDataset(count_dims);
    recorder_chip_dims.Write(offset_dims, count_dims, offset_dims, count_dims, &dims[0]);
  }
  if (flowlimit_arg.empty())
    flowlimit = dims.At(2);
  else
    flowlimit = atoi(flowlimit_arg.c_str());

  flowlimit = (flowlimit < dims.At(2)) ? flowlimit : dims.At(2);
  fprintf(stdout, "Using %u flows\n", flowlimit);

  // chunks no bigger than 100000
  vector<hsize_t>chunks_out(2);
  chunks_out.At(0) = (input_positions.size() < 10000) ? input_positions.size() : 100000;
  chunks_out.At(1) = chunks.At(2);

  recorder.CreateDataset(chunks_out);
  vector<hsize_t> extension(2);
  extension.At(0) = input_positions.size();
  extension.At(1) = dims.At(2);
  recorder.ExtendDataSet(extension); // extend if necessary

  fprintf(stdout, "Opening for write %s:%s with rank %d, position x flow chunks=[ ", &h5file_out[0], &destination[0], (int)chunks_out.size());
  for (int i=0; i<(int)chunks_out.size(); i++)
    fprintf(stdout, "%d ", (int)chunks_out.At(i));
  fprintf(stdout, "]\n");

  int max_threads_ever = (dims.At(0)/chunks.At(0) +1)*(dims.At(1)/chunks.At(1) +1);
  thread_flags.resize (max_threads_ever, 0);
  // fprintf(stdout, "max_threads_ever = %d\n", max_threads_ever);
  unsigned int thread_id = 0;
  vector<thread_args> my_args( max_threads_ever );

  size_t runningCount = 0;

  // layout is rows x cols x flows
  for (size_t row=0; row<dims.At(0); ) {
    for (size_t col=0; col<dims.At(1); ) {

      size_t ix = 0;
      hsize_t offset_out = 0;
      hsize_t count_out = 0;

      vector<size_t> limit(2);
      limit.At(0) = ( row+chunks.At(0) < dims.At(0) ) ? row+chunks.At(0) : dims.At(0);
      limit.At(1) = ( col+chunks.At(1) < dims.At(1) ) ? col+chunks.At(1) : dims.At(1);
      // fprintf(stdout, "Block row=%lu, col=%lu, count=[%lu %lu]\n", row, col, limit.At(0), limit.At(1));
      // bool first_time=true;
      for (size_t rr=row; rr<limit.At(0) && ix < input_positions.size(); rr++) {
	for (size_t cc=col; cc<limit.At(1) && ix < input_positions.size(); cc++) {
	  size_t pos = input_positions.At(ix);
	  size_t chp_indx = RowColToIndex(rr,cc, dims.At(0), dims.At(1));
	  // if (first_time)
	  //   fprintf(stdout, "Entering loop with pos=%lu, ix=%lu, chp_indx=%lu\n", pos, ix, chp_indx);
	  // first_time = false;

	  if ( chp_indx < pos)
	    continue;

	  while ( chp_indx > pos){
	    // fprintf(stdout, "chp_indx=%lu > pos=%lu, incrementing ix=%lu\n", chp_indx, pos, ix);
	    ix++;
	    if (ix == input_positions.size()){
	      break;
	    }
	    pos = input_positions.At(ix);
	    // first_time = true;
	  }

	  if( chp_indx == pos){
	    if ( count_out == 0)
	      offset_out = runningCount;
	    count_out++;
	    runningCount++;
	    // fprintf(stdout, "found: rr=%d, cc=%d, pos=%d, index=%d, ix=%lu, runningCount=%lu\n", (int)rr, (int)cc, (int)pos, (int)chp_indx, ix, runningCount);
	    ix++;
	    continue;
	  }
	  
	}
      }

      assert (ix <= input_positions.size() );
      assert (runningCount <= input_positions.size() );
      
      if (count_out > 0) {
	pthread_t thread;
	int thread_status = 0;

	assert( thread_id < thread_flags.size() );
	my_args.at(thread_id).row = row;
	my_args.at(thread_id).col = col;
	my_args.at(thread_id).chunks = &chunks;
	my_args.at(thread_id).chunks_out = &chunks_out;
	my_args.at(thread_id).dims = &dims;
	my_args.at(thread_id).h5file_in = &h5file_in;
	my_args.at(thread_id).source = &source;
	my_args.at(thread_id).h5file_out = &h5file_out;
	my_args.at(thread_id).destination = &destination;
	my_args.at(thread_id).offset_out = offset_out;
	my_args.at(thread_id).count_out = count_out;
	my_args.at(thread_id).input_positions = &input_positions;
	my_args.at(thread_id).thread_id = thread_id;
	my_args.at(thread_id).flowlimit = flowlimit;

	// fprintf(stdout, "creating thread %d from row=%d (max %d), column=%d (max %d), offset_out=%llu, count_out=%llu\n", thread_id, (int)row, (int)dims.At(0), (int)col, (int)dims.At(1), offset_out, count_out);
	while (accumulate(thread_flags.begin(), thread_flags.end(), 0) > numThreads) {
	  // only have to be approximate, don't worry about races
	  fprintf(stdout, "Sleeping before creating thread %d from row=%d (max %d), column=%d (max %d), offset_out=%llu, count_out=%llu ...\n", thread_id, (int)row, (int)dims.At(0), (int)col, (int)dims.At(1), offset_out, count_out);
	  sleep(1);
	}
	thread_flags.At(thread_id) = 1;
	thread_status = pthread_create(&thread, NULL, do_subset, (void *)&my_args[thread_id]);
	// do_subset((void *)&my_args[thread_id]);
	assert (thread_status >= 0);
	thread_id++;
      }
      col += chunks.At(1);
      //fflush(stdout);
    }
    row += chunks.At(0);
  }
  while (accumulate(thread_flags.begin(), thread_flags.end(), 0) > 0) {
    // wait for the threads to finish
    // fprintf(stdout, "Waiting ...\n");
    sleep(1);
  }

  assert (runningCount == input_positions.size() );
  cout << "Done." << endl;
  pthread_exit(NULL);
}

void *do_subset(void* data)
{
  pthread_detach( pthread_self() );

  struct thread_args *args = (struct thread_args *)data;
  hsize_t row = args->row;
  hsize_t col = args->col;
  vector<hsize_t>chunks( * (args->chunks));
  vector<hsize_t>dims(* (args->dims));
  string h5file_in = string(* (args->h5file_in));
  string source = string( * (args->source));
  string h5file_out = string(* (args->h5file_out));
  string destination = string(* (args->destination));
  hsize_t offset_out_0 = args->offset_out;
  hsize_t count_out_0 = args->count_out;
  vector<size_t>input_positions(* (args->input_positions));
  unsigned int thread_id = args->thread_id;
  unsigned int flowlimit = args->flowlimit;

  H5ReplayReader reader = H5ReplayReader(h5file_in, &source[0]);
  H5ReplayRecorder recorder = H5ReplayRecorder(h5file_out, &destination[0]);

  // read a chunk in
  vector<hsize_t> offset(3);
  offset.At(0) = row;
  offset.At(1) = col;
  offset.At(2) = 0;
      
  vector<hsize_t> count(3);
  count.At(0) = ( row+chunks.At(0) < dims.At(0) ) ? chunks.At(0) : dims.At(0)-row;
  count.At(1) = ( col+chunks.At(1) < dims.At(1) ) ? chunks.At(1) : dims.At(1)-col;
  count.At(2) = dims.At(2);

  vector<float> signal(count.At(0) * count.At(1) * count.At(2));
  vector<float> subset(count_out_0*count.At(2));

  vector<hsize_t> offset_in(1);
  offset_in.At(0) = 0;

  vector<hsize_t> count_in(1, signal.size());
      
  // fprintf(stdout, "thread_id %d reading %d from row=%d (max %d), %d from column=%d (max %d)\n", thread_id, (int)count.At(0), (int)row, (int)dims.At(0), (int)count.At(1), (int)col, (int)dims.At(1));
  reader.Read(offset, count, offset_in, count_in, &signal[0]);

  // now subset well at a time
  size_t ix = 0;
  size_t incr = 0;
  vector<size_t> subset_input_positions(count_out_0);
  for (size_t rr=row; rr<(row+count.At(0)) && ix < input_positions.size(); rr++) {
    for (size_t cc=col; cc<(col+count.At(1)) && ix < input_positions.size(); cc++) {
      size_t pos = input_positions.At(ix);
      size_t chp_indx = RowColToIndex1(rr,cc, dims.At(0), dims.At(1));
      if ( chp_indx < pos)
	continue;

      while ( chp_indx > pos){
	ix++;
	if (ix == input_positions.size())
	  break;
	pos = input_positions.At(ix);
	
      }

      if( chp_indx == pos) {
	size_t array_indx = RowColToIndex1(rr-row,cc-col,count.At(0),count.At(1))*dims.At(2);
	// fprintf(stdout, "thread_id=%d: Position match %lu at chip rr=%lu, rr-row=%d, cc=%lu, cc-col=%d, (array_indx=%lu)/(dims[2]=%llu)=%llu\n", thread_id, pos, rr, (int)(rr-row), cc, (int)(cc-col), array_indx, dims.At(2), array_indx/dims.At(2));
	for(size_t flow=0; flow<flowlimit; flow++)
	  subset.At(incr*dims.At(2) + flow) = signal.At( array_indx + flow);
	assert (incr < count_out_0 );
	subset_input_positions.At(incr) = input_positions.At(ix);
	incr++;
	ix++;
      }
      assert (ix <= input_positions.size() );
    }
  }

  assert (incr == count_out_0 );

  // write the chunk back out
  vector<hsize_t> count_out(2);
  count_out.At(0) = count_out_0;
  count_out.At(1) = dims.At(2);

  vector<hsize_t> offset_out(2);
  offset_out.At(0) = offset_out_0;
  offset_out.At(1) = 0;

  vector<hsize_t> offset_in_subset(1,0);
  vector<hsize_t> count_in_subset(1,subset.size());
  recorder.Write(offset_out, count_out, offset_in_subset, count_in_subset, &subset[0]);
  
  {
    string position = "position";
    H5ReplayRecorder recorder_pos = H5ReplayRecorder(h5file_out, &position[0]);

    vector<hsize_t> offset_pos(1, offset_out.At(0));
    vector<hsize_t> count_pos(1, count_out.At(0));
    recorder_pos.Write(offset_pos, count_pos, offset_in, count_pos, &subset_input_positions[0]);
  }
  
  thread_flags.at(thread_id) = 0;
  int ret = 0;
  pthread_exit(&ret);

  return(NULL);
}  
