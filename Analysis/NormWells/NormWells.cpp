/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <numeric>
#include "OptArgs.h"
#include "H5Replay.h"
#include "NoKeyCall.h"

#define MAXTHREADS 16
#define SIGNAL_IN "/wells"
#define SIGNAL_OUT "/bead/normsignal"

using namespace std;

void usage() {
  cout << "NormWells - Program for normalizing well values in default dataset wells" << endl;
  cout << "Adds output dataset bead/normsignal to the input h5file";
  cout << "" << endl;
  cout << "Usage:" << endl;
  cout << "   NormWells [--source wells] [--destination bead/normsignal] h5file" << endl;
  cout << "Options:" << endl;
  cout << "   -s,--source         dataset to read from (default " << SIGNAL_IN << ")" << endl;
  cout << "   -d,--destination    dataset to write to (default " << SIGNAL_OUT <<"), but can be same as source" << endl;
  cout << "   -f,--flowlimit      number of flows to use (default = number flows in input h5file)" << endl;
  cout << "   -h,--help      this message." << endl;
  exit(1);
}

struct compute_norm_args {
  hsize_t row;
  hsize_t col;
  vector<hsize_t> *chunks;
  vector<hsize_t> *dims;
  string *h5file;
  string *source;
  string *destination;
  int thread_id;
  unsigned int flowlimit;
};

void DumpStartingStateOfNormWells (int argc, const char *argv[])
{
  char myHostName[128] = { 0 };
  gethostname (myHostName, 128);
  fprintf (stdout, "\n");
  fprintf (stdout, "Hostname = %s\n", myHostName);
  time_t rawtime;
  time(&rawtime);
  fprintf (stdout, "Start Time = %s", ctime (&rawtime));
  fprintf (stdout, "Version = %s.%s (%s) (%s)\n",
           IonVersion::GetVersion().c_str(), IonVersion::GetRelease().c_str(),
           IonVersion::GetGitHash().c_str(), IonVersion::GetBuildNum().c_str());
  fprintf (stdout, "Command line = ");
  for (int i = 0; i < argc; i++)
    fprintf (stdout, "%s ", argv[i]);
  fprintf (stdout, "\n");
  fflush (NULL);
}

vector<unsigned int> thread_flags;

void *compute_norm(void *arguments);

int main(int argc, const char *argv[]) {
  OptArgs opts;  
  string h5file;
  string source;
  string destination;
  vector<string> infiles;
  bool help;
  string flowlimit_arg;
  unsigned int flowlimit;

  DumpStartingStateOfNormWells (argc,argv);

  opts.ParseCmdLine(argc, argv);
  opts.GetOption(h5file, "", '-', "h5file");
  opts.GetOption(source, "", 's', "source");
  opts.GetOption(destination, "", 'd', "destination");
  opts.GetOption(flowlimit_arg, "", 'f', "flowlimit");
  opts.GetOption(help, "false", 'h', "help");
  opts.GetLeftoverArguments(infiles);
  if(help || infiles.empty() || (infiles.size() > 1) ) {
    usage();
  }
  h5file = infiles.front();

  int numCPU = (int)sysconf( _SC_NPROCESSORS_ONLN );
  int numThreads = MAXTHREADS < numCPU ? MAXTHREADS : numCPU;
  fprintf(stdout, "Using %d threads of %d cores\n", numThreads, numCPU);

  if (source.empty())
    source = source + SIGNAL_IN;
  H5ReplayReader reader = H5ReplayReader(h5file, &source[0]);
  if ( destination.empty() )
    destination = destination + SIGNAL_OUT;

  H5ReplayRecorder recorder = (source.compare(destination)==0)
    ? H5ReplayRecorder(h5file, &destination[0])
    : H5ReplayRecorder(h5file, &destination[0],reader.GetType(),reader.GetRank());

  reader.Open();
  int rank = reader.GetRank();
  vector<hsize_t>dims(rank,0);
  vector<hsize_t>chunks(rank,0);
  reader.GetDims(dims);
  reader.GetChunkSize(chunks);
  reader.Close();

  fprintf(stdout, "Opening for read %s:%s with rank %d, row x col x flow dims=[ ", &h5file[0], &source[0], rank);
  for (int i=0; i<rank; i++)
    fprintf(stdout, "%d ", (int)dims[i]);
  fprintf(stdout, "], chunksize=[ ");
  for (int i=0; i<rank; i++)
    fprintf(stdout, "%d ", (int)chunks[i]);
  fprintf(stdout, "]\n");

  if (flowlimit_arg.empty())
    flowlimit = dims[2];
  else
    flowlimit = atoi(flowlimit_arg.c_str());

  flowlimit = (flowlimit < dims[2]) ? flowlimit : dims[2];
  fprintf(stdout, "Using %u flows\n", flowlimit);

  // hard code region size to be at least 100x100
  chunks[0] = (chunks[0] < 100) ? 100 : chunks[0];
  chunks[1] = (chunks[1] < 100) ? 100 : chunks[1];

  recorder.CreateDataset(chunks);
  
  int max_threads_ever = (dims[0]/chunks[0] +1)*(dims[1]/chunks[1] +1);
  thread_flags.resize (max_threads_ever, 0);
  // fprintf(stdout, "max_threads_ever = %d\n", max_threads_ever);
  unsigned int thread_id = 0;
  vector<compute_norm_args> my_args( max_threads_ever );
  
  // layout is rows x cols x flows
  for (hsize_t row=0; row<dims[0]; ) {
    for (hsize_t col=0; col<dims[1]; ) {
      pthread_t thread;
      int thread_status;

      assert( thread_id < thread_flags.size() );
      my_args.at(thread_id).row = row;
      my_args.at(thread_id).col = col;
      my_args.at(thread_id).chunks = &chunks;
      my_args.at(thread_id).dims = &dims;
      my_args.at(thread_id).h5file = &h5file;
      my_args.at(thread_id).source = &source;
      my_args.at(thread_id).destination = &destination;
      my_args.at(thread_id).thread_id = thread_id;
      my_args.at(thread_id).flowlimit = flowlimit;

      fprintf(stdout, "creating thread %d from row=%d (max %d), column=%d (max %d)\n", thread_id, (int)row, (int)dims[0], (int)col, (int)dims[1]);
      while (accumulate(thread_flags.begin(), thread_flags.end(), 0) > numThreads) {
	// only have to be approximate, don't worry about races
	// fprintf(stdout, "Sleeping ...\n");
	sleep(1);
      }
      thread_flags[thread_id] = 1;
      thread_status = pthread_create(&thread, NULL, compute_norm, (void *)&my_args[thread_id]);
      // compute_norm((void *)&my_args[thread_id]);
      assert (thread_status >= 0);
      thread_id++;

      col += chunks[1];
      //fflush(stdout);
    }
    row += chunks[0];
  }
  while (accumulate(thread_flags.begin(), thread_flags.end(), 0) > 0) {
    // wait for the threads to finish
    // fprintf(stdout, "Waiting ...\n");
    sleep(1);
  }

  cout << "Done." << endl;
  pthread_exit(NULL);
}

void *compute_norm(void* data)
{
  // fprintf(stdout, "Thread count %d\n", thread_count);

  pthread_detach( pthread_self() );

  struct compute_norm_args *args = (struct compute_norm_args *)data;
  hsize_t row = args->row;
  hsize_t col = args->col;
  vector<hsize_t>chunks( * (args->chunks));
  vector<hsize_t>dims(* (args->dims));
  string h5file = string(* (args->h5file));
  string source = string( * (args->source));
  string destination = string(* (args->destination));
  unsigned int thread_id = args->thread_id;
  unsigned int flowlimit = args->flowlimit;

  H5ReplayReader reader = H5ReplayReader(h5file, &source[0]);
  H5ReplayRecorder recorder = H5ReplayRecorder(h5file, &destination[0]);
  
  // read a chunk in
  vector<hsize_t> offset(3);
  offset[0] = row;
  offset[1] = col;
  offset[2] = 0;
      
  vector<hsize_t> count(3);
  count[0] = ( row+chunks[0] < dims[0] ) ? chunks[0] : dims[0]-row;
  count[1] = ( col+chunks[1] < dims[1] ) ? chunks[1] : dims[1]-col;
  count[2] = dims[2];

  vector<float> signal(count[0] * count[1] * count[2], 0);
  vector<float> normed_signal(signal.size(), 0);

  vector<hsize_t> offset_in(1);
  offset_in[0] = 0;

  vector<hsize_t> count_in(1);
  count_in[0] = signal.size();
      
  // fprintf(stdout, "thread_id %d reading %d from row=%d (max %d), %d from column=%d (max %d)\n", thread_id, (int)count[0], (int)row, (int)dims[0], (int)count[1], (int)col, (int)dims[1]);
  reader.Read(offset, count, offset_in, count_in, &signal[0]);

  // now normalize a well at a time
  size_t ix = 0;
  for (size_t rr=0; rr<count[0]; rr++) {
    for (size_t cc=0; cc<count[1]; cc++) {
      size_t wellindex = ix*count[2];
      ix++;
      vector<double> peaks;
      assert (flowlimit <= dims[2] );
      vector<float>dat(signal.begin()+wellindex, signal.begin()+wellindex+flowlimit);
      vector<float>normalized(dat.size(), 0);
      NoKeyCall nkc;
      nkc.SetBeadLocation(cc+col,rr+row);
      nkc.GetPeaks(dat, peaks);
      float failure_to_normalize_value = -1.0f;
      nkc.NormalizeBeadSignal(dat, peaks, normalized, failure_to_normalize_value);
      // vector<float>normalized(dat.size(), 1);
      
      // stick it back in signal
      double v = 0;
      for (size_t fnum = 0; fnum < count[2]; fnum++){
	if (fnum < flowlimit) {
	  *(normed_signal.begin()+wellindex+fnum) = normalized[fnum];
	  v += fabs(normalized[fnum]);
	}
	else {
	  *(normed_signal.begin()+wellindex+fnum) = 0;
	}
      }
      assert(v>0);
    }
  }

  // write the chunk back out
  vector<hsize_t> extension(3);
  extension[0] = row+count[0];
  extension[1] = col+count[1];
  extension[2] = count[2];
  
  recorder.ExtendDataSet(extension); // extend if necessary
  recorder.Write(offset, count, offset_in, count_in, &normed_signal[0]);
  
  thread_flags.at(thread_id) = 0;
  int ret = 0;
  pthread_exit(&ret);

  return(NULL);
}  
