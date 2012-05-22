/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "DualGaussMixModel.h"
#include "Separator.h"
#include "KMlocal.h"
#include "Utils.h"
#include "LinuxCompat.h"
#include "dbgmem.h"

#define NUM_SAMPLE_REGIONS 3

// k-means cluster default params
KMterm term (100, 0, 0, 0, // run for 100 stages
             0.10, // min consec RDL
             0.10, // min accum RDL
             3, // max run stages
             0.50, // init. prob. of acceptance
             10, // temp. run length
             0.95); // temp. reduction factor

// Random number generator function for STL to ensure rand() is called
// thus honoring any srand() calls
ptrdiff_t MyStlRandom (ptrdiff_t i) { return rand() % i; }

Separator::Separator (bool _bfTest)
{
  bfTest = _bfTest;
  work = NULL;
  bead = NULL;
  start = 0;
  end = 0;
  w = 0;
  h = 0;
  numKeyFlows = 0;
  minPeakThreshold = 20;
  strncpy (experimentDir, "./", 3);
  avgTFSig = 0.0;
  avgLibSig = 0.0;
  numGroups = 1;
  groups = NULL;
  regenerateGroups = false;
  num_regions = 0;
  region_list = NULL; // not owned by Separator...do not free
  avgRegionKeySignal = NULL;
  timeStart = 687; //25
  timeEnd = 1429; //36
  rCnt = 0;
  flowOrder = NULL;
  numFlowsPerCycle = 0;
  emptyFraction = 0.10;
  beadFindIx = 0;
  pthread_mutex_init (&lock, NULL);
}

Separator::~Separator()
{
  if (work)
    delete[] work;

  if (avgRegionKeySignal)
    delete [] avgRegionKeySignal;
  region_list = NULL;

  if (groups)
    delete[] groups;
  CleanupBeads();

  if (flowOrder)
    free (flowOrder);
}

void Separator::CleanupBeads()
{
  if (bead)
  {
    free (bead);

    bead = NULL;
  }
}

void Separator::SetSize (int _w, int _h, int _numKeyFlows, int _numGroups)
{
  w = _w;
  h = _h;
  numKeyFlows = _numKeyFlows;
  rCnt = 0; //Static region counter variable.

  if (work)
    delete[] work;
  work = new double[w * h * numKeyFlows];
  memset (work, 0, sizeof (double) * w * h * numKeyFlows);

  frameStride = (h * w);

  CleanupBeads();

  bead = (int16_t *) malloc (sizeof (int16_t) * numKeyFlows * w * h * (end
                             - start));
  memset (bead, 0, sizeof (int16_t) * numKeyFlows * w * h * (end - start));
  beadIdx1_Len = w * h * (end - start);
  beadIdx2_Len = (end - start);

  if (_numGroups > 0)
  {
    numGroups = _numGroups;
    if (groups)
      delete[] groups;
    groups = NULL;
    groups = new unsigned char[w * h];
    memset (groups, 0, sizeof (unsigned char) * w * h);
    regenerateGroups = true;
  } // else we leave things as is

  chipSampleRegions.clear();
  chipSampleIdx.clear();
  beadFindIx = 0;

  //debug
  //fprintf (stdout, "Separator:bead allocating %lu\n",w*h*(end-start)*numKeyFlows*sizeof(float));
}

void Separator::SetFlowOrder (char *_flowOrder)
{
  if (flowOrder)
    free (flowOrder);
  flowOrder = strdup (_flowOrder);
  numFlowsPerCycle = strlen (flowOrder);
}

void Separator::SetDir (char *_experimentDir)
{
  strncpy (experimentDir, _experimentDir, 256);
}

// Compare function for qsort for Descending order
static int doubleCompare (const void *v1, const void *v2)
{
  double val1 = * (double *) v1;
  double val2 = * (double *) v2;

  if (val1 < val2)
    return 1;
  else if (val2 < val1)
    return -1;
  else
    return 0;
}

double Separator::GetDistOnDiag (int index, int nCol, double theta)
{
  int row = index / nCol;
  int col = index % nCol;
  /*
   Calcuate the distance on the y = row/col * x diagonal that this point
   is closest to. We know the angle (alpha) of this point and we also know
   the angle of the diagonal (theta).
   cos(abs(theta - alpha)) * hypotenuse = projection onto diagonal
   */
  double hyp = sqrt (row * row + col * col);
  double alpha = 1.570796;
  if (col > 0)
  {
    alpha = atan ( ( (double) row) / col);
  }
  double adj = cos (abs (theta - alpha)) * hyp;
  return adj;
}

int Separator::GetRegionBin (int index, int nCol, double theta, const std::vector<double> &breaks)
{
  double dist = GetDistOnDiag (index, nCol, theta);
  for (size_t i = 0; i < breaks.size(); i++)
  {
    if (dist <= breaks[i])
    {
      return i;
    }
  }
  if (dist <= breaks.back() + 2.0)
  {
    return breaks.size() - 1;
  }
  fprintf (stderr, "Couldn't find right bin for dist: %f\n", dist);
  exit (1);
}

void Separator::GetChipSample (Image *image, Mask *mask, double fraction)
{
  const RawImage *raw = image->GetImage();
  const double *results = image->GetResults();
  int totalWells = raw->rows * raw->cols;

  // Calculate distances from origin
  double length = (sqrt (raw->rows * raw->rows + raw->cols * raw->cols))
                  / (NUM_SAMPLE_REGIONS);
  for (int i = 0; i < (NUM_SAMPLE_REGIONS); i++)
  {
    chipSampleRegions.push_back ( (i + 1) * length);
  }
  double theta = atan ( ( (double) raw->rows) / raw->cols);
  // Setup our samples and reserve some memory to avoid too many reallocations
  int stepSize = raw->rows * raw->cols;
  if (fraction > 0)
  {
    stepSize = std::min ( (int) (1 / fraction), stepSize);
  }
  chipSampleIdx.resize (NUM_SAMPLE_REGIONS);
  for (int i = 0; i < NUM_SAMPLE_REGIONS; i++)
  {
    chipSampleIdx[i].reserve ( (int) fraction * totalWells
                               / (NUM_SAMPLE_REGIONS + 1));
  }

  // Put each sample in the appropriate region
  int nanCount = 0;
  for (int i = 0; i < totalWells; i += stepSize)
  {
    if ( (*mask) [i] & (MaskPinned | MaskIgnore | MaskExclude))
    {
      continue;
    }
    if (!std::isnan (results[i]))
    {
      int bin = GetRegionBin (i, raw->cols, theta, chipSampleRegions);
      chipSampleIdx[bin].push_back (i);
    }
    else
    {
      nanCount++;
    }
  }
  srand (42); // Make the "random" sampling reproducible
  // Randomize as we want to grab some random points of these
  for (size_t i = 0; i < chipSampleIdx.size(); i++)
  {
    std::random_shuffle (chipSampleIdx[i].begin(), chipSampleIdx[i].end(), MyStlRandom);
  }
}

bool Separator::isInRegion (int index, int nCol, Region *region)
{
  int col = index % nCol;
  int row = index / nCol;
  bool inCol = (col >= region->col && col < (region->col + region->w));
  bool inRow = (row >= region->row && row < (region->row + region->h));
  return inCol && inRow;
}

void Separator::FindCoords (double dist, double theta, int &row, int &col)
{
  row = (int) dist * sin (theta);
  col = (int) dist * cos (theta);
}

void Separator::AddGlobalSample (int nRow, int nCol, int size, Region *region,
                                 const std::vector<double> &breaks, double theta,
                                 std::vector<double> &sampleData, const double *results)
{
  std::vector<double> centerDist;
  double sum = 0;
  double slush = 100; // acts as a prior to spread out the distance based weight a bit more
  // Each region will add a number of samples proportional to inverse distance squared
  // So closest region will contribute most and furthest region the least
  int rRow = region->row + region->h / 2;
  int rCol = region->col + region->w / 2;
  for (size_t i = 0; i < breaks.size(); i++)
  {
    int row1 = 0, col1 = 0;
    double previous = 0;
    if (i > 0)
    {
      FindCoords (breaks[i - 1], theta, row1, col1);
      previous = breaks[i - 1];
    }
    int row2 = 0, col2 = 0;
    FindCoords (breaks[i], theta, row2, col2);

    //    double middle = GetDistOnDiag(row * nCol + col, nCol, theta);
    double dist1 = sqrt ( (row1 - rRow) * (row1 - rRow) + (col1 - rCol)
                          * (col1 - rCol));
    double dist2 = sqrt ( (row2 - rRow) * (row2 - rRow) + (col2 - rCol)
                          * (col2 - rCol));
    double dist = ( (dist1 + dist2) / 2.0) + slush;
    centerDist.push_back (dist);
    sum += (1 / (dist));
  }

  int sSize = std::max ( (percentForSample * nRow * nCol) - size, 0.0);
  for (size_t i = 0; i < chipSampleIdx.size(); i++)
  {
    size_t nPts = sSize * (1 / (centerDist[i])) / sum;
    int dupes = 0;
    for (size_t pIx = 0; pIx < chipSampleIdx[i].size() && pIx < nPts; pIx++)
    {
      int index = chipSampleIdx[i][pIx];
      if (!isInRegion (index, nCol, region))
      {
        sampleData.push_back (results[index]);
      }
      else
      {
        dupes++;
      }
    }
  }
}

void Separator::FindBeads (Image *image, Region *region, Mask *mask, char *prepend)
{
  // promote neighbor subtracted wells as 'potential' beads, and count them

  beadFindIx++;
  int k = 0;
  int wellCount = 0;
  int x, y;
  const RawImage *raw = image->GetImage();
  const double *results = image->GetResults();
  bool printDebug = false;
  char fileName[512] = {'\0'};

  pthread_mutex_lock (&lock);

  if (chipSampleIdx.empty() && bfTest)
  {
    snprintf (fileName, 512, "%s/%s", experimentDir, "stats-samples.txt");
    if (printDebug)
      statsOut.open (fileName);
    percentForSample = 0.2;
    if (percentForSample > 0)
    {
      GetChipSample (image, mask, percentForSample * NUM_SAMPLE_REGIONS);
    }
  }

  snprintf (fileName, 512, "%s/%s%s", experimentDir, prepend, "beadfind.txt");
  FILE *fp = NULL;
  fopen_s (&fp, fileName, "ab");
  rCnt++; //increment static region counter
  fprintf (fp, "Region %04d\n", rCnt);

  // get min/max for beadfind metric
  double bfmin = 9999999.0, bfmax = -9999999.0;
  double bfval;
  for (y = region->row; y < (region->row + region->h); y++)
  {
    for (x = region->col; x < (region->col + region->w); x++)
    {
      if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                          | MaskExclude))
      {
        continue;
      }
      else
      {
        wellCount++;
        bfval = results[x + y * raw->cols];
        if (bfval < bfmin)
          bfmin = bfval;
        if (bfval > bfmax)
          bfmax = bfval;
      }
    }
  }
  fprintf (fp, "Wells not pinned or ill-behaved: %d\n", wellCount);

  // Open Histogram Plot Data File
  FILE *hfp = NULL;
  snprintf (fileName, 512, "%s/%s%s", experimentDir, prepend,
            "beadfindData.txt");
  fopen_s (&hfp, fileName, "ab");

  // Histogram plotting data, region initialization
  fprintf (hfp, "%d = ", rCnt);

  // bail here if we don't have enough wells to evaluate
  if (wellCount < 2)
  {
    fprintf (fp,
             "\tSeparator::FindBeads - not enough valid wells to separate.\n");
    if (fp)
      fclose (fp);
    if (hfp)
    {
      // print empty data so a plot can be generated
      // Histogram Plot Metrics
      fprintf (hfp, "%0.2lf ", 0.0);
      fprintf (hfp, "\n");
      fclose (hfp);
    }
    pthread_mutex_unlock (&lock);
    return;
  }

  int stages = 100;
  term.setAbsMaxTotStage (stages);
  int numMetrics = 1;
  int pt = 0;
  std::vector<double> sampleData;
  double theta = atan ( ( (double) raw->rows) / raw->cols);
  if (bfTest)
  {
    AddGlobalSample (raw->rows, raw->cols, wellCount, region,
                     chipSampleRegions, theta, sampleData, results);
    if (printDebug)
      statsOut.flush();
  }
  std::vector<float> dataCluster (wellCount + sampleData.size());
  for (y = region->row; y < (region->row + region->h); y++)
  {
    for (x = region->col; x < (region->col + region->w); x++)
    {
      if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                          | MaskExclude))
      {
        continue;
      }
      else
      {
        dataCluster[pt++] = results[x + y * raw->cols];
        // Histogram Plot Metrics
        char stats[256];
        int index = y * raw->cols + x;
        double sDist = GetDistOnDiag (index, raw->cols, theta);
        snprintf (stats, sizeof (stats),
                  "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\n", beadFindIx,
                  region->row + region->h / 2, region->col + region->w
                  / 2, -1, index, x, y, sDist, results[index]);
        if (printDebug)
        {
          statsOut << stats;
          statsOut.flush();
        }
        fprintf (hfp, "%0.2lf ", results[x + y * raw->cols]);
      }
    }
  }

  // Close Histogram Plot Data file
  fprintf (hfp, "\n");
  fclose (hfp);

  pthread_mutex_unlock (&lock);

  for (unsigned int i = 0; i < sampleData.size(); i++)
  {
    dataCluster[pt++] = sampleData[i];
  }

  int beadCount = 0;
  if (bfTest)
  {
    DualGaussMixModel dgm (100000);
    MixModel model = dgm.FitDualGaussMixModel (&dataCluster[0],
                     dataCluster.size());
    pthread_mutex_lock (&lock);
    std::cout << "Region: " << beadFindIx << " (" << region->col << ","
              << region->row << ")\tmu1: " << model.mu1 << " var1: "
              << model.var1 << "\tmu2: " << model.mu2 << " var2: "
              << model.var2 << " mix: " << model.mix << std::endl;
    SepModel sepModel;
    sepModel.model = model;
    sepModel.row = region->row;
    sepModel.col = region->col;
    mModels.push_back (sepModel);
    pthread_mutex_unlock (&lock);
    std::vector<int> cluster (wellCount, 0);
    dgm.AssignToCluster (&cluster[0], model, &dataCluster[0],
                         cluster.size(), .35);
    pt = 0;
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore | MaskExclude))
        {
          continue;
        }
        else
        {
          if (cluster[pt] == 2)   // so counts both singles and doubles here
          {
            (*mask) [x + y * raw->cols] = MaskBead;
            beadCount++;
          }
          pt++;
        }
      }
    }

    //--- mark top 10% of empty wells as Ignored - copies R&D pipeline.
    //--- ensures that marginal wells which might be beads do not get used in neighbor subtract.
    // get list of empty wells
    // create an array guaranteed to hold all empty wells.
    double *emptyWells = new double[pt];
    pt = 0;
    int emptyCount = 0;
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                            | MaskExclude))
        {
          continue;
        }
        if (cluster[pt] < 2)
        {
          // This well is empty
          emptyWells[emptyCount] = results[x + y * raw->cols];
          emptyCount++;
        }
        pt++;
      }
    }

    // sort highest to lowest (descending sort)
    qsort (emptyWells, emptyCount, sizeof (double), doubleCompare);
    //int cutoffIndex = emptyCount / 10; //top 10%
    int cutoffIndex = (int) (emptyCount * emptyFraction);
    double cutoffVal = emptyWells[cutoffIndex];
    delete[] emptyWells;

    // mark top 10% of empties with MaskIgnore
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                            | MaskBead | MaskExclude))
        {
          continue;
        }
        if (results[x + y * raw->cols] > cutoffVal)
        {
          //(*mask)[x+y*raw->cols] |= MaskIgnore;
          (*mask) [x + y * raw->cols] = MaskBead;
        }
      }
    }

  }
  else
  {

    KMdata dataPts (numMetrics, wellCount + sampleData.size());
    pt = 0;
    for (unsigned int i = 0; i < dataCluster.size(); i++)
    {
      dataPts[pt++][0] = dataCluster[i];
    }

    dataPts.buildKcTree();
    int numClusters = 2;
    KMfilterCenters ctrs (numClusters, dataPts); // allocate centers
    KMlocalLloyds kmLloyds (ctrs, term);
    ctrs = kmLloyds.execute();

    // and see what cluster each belongs to
    KMctrIdxArray closeCtr = new KMctrIdx[dataPts.getNPts() ];
    double* sqDist = new double[dataPts.getNPts() ];
    ctrs.getAssignments (closeCtr, sqDist);

    // need to first determine what clusters have what populations
    // approach will be to take the first 100 items per cluster (or all if less than 100)
    // and look at avg peak for all, then max will be doubles, middle will be singles, and min will be empties

    int counts[numClusters];
    double vals[numClusters];
    memset (counts, 0, sizeof (counts));
    memset (vals, 0, sizeof (vals));
    pt = 0;
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                            | MaskExclude))
        {
          continue;
        }
        else
        {
          int group = closeCtr[pt];
          if (counts[group] < 100)
          {
            counts[group]++;
            vals[group] += results[x + y * raw->cols];
          }
          pt++;
        }
      }
    }

    int sortedOrder[numClusters];
    for (k = 0; k < numClusters; k++)
    {
      sortedOrder[k] = k; // initialize the list to default order
      vals[k] /= counts[k]; // normalize the metric vals in each group
      fprintf (fp, "Group %d metric avg is %.2lf\n", k, vals[k]);
    }

    // check/update sortlist
    int checks;
    for (checks = 0; checks < numClusters - 1; checks++)   // worst case bubble sort is to do n-1 passes
    {
      for (k = 0; k < numClusters - 1; k++)
      {
        if (vals[sortedOrder[k]] > vals[sortedOrder[k + 1]])   // need to swap
        {
          int temp = sortedOrder[k];
          sortedOrder[k] = sortedOrder[k + 1];
          sortedOrder[k + 1] = temp;
        }
      }
    }

    // mark our beads
    pt = 0;

    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                            | MaskExclude))
        {
          continue;
        }
        else
        {
          if (sortedOrder[closeCtr[pt]] > 0)   // so counts both singles and doubles here
          {
            (*mask) [x + y * raw->cols] = MaskBead;
            beadCount++;
          }
          pt++;
        }
      }
    }

    //--- mark top 10% of empty wells as Ignored - copies R&D pipeline.
    //--- ensures that marginal wells which might be beads do not get used in neighbor subtract.
    // get list of empty wells
    // create an array guaranteed to hold all empty wells.
    double *emptyWells = new double[pt];
    pt = 0;
    int emptyCount = 0;
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                            | MaskExclude))
        {
          continue;
        }
        if (sortedOrder[closeCtr[pt]] == 0)
        {
          // This well is empty
          emptyWells[emptyCount] = results[x + y * raw->cols];
          emptyCount++;
        }
        pt++;
      }
    }

    // sort highest to lowest (descending sort)
    qsort (emptyWells, emptyCount, sizeof (double), doubleCompare);
    //int cutoffIndex = emptyCount / 10; //top 10%
    int cutoffIndex = (int) (emptyCount * emptyFraction);
    double cutoffVal = emptyWells[cutoffIndex];
    delete[] emptyWells;

    // mark top 10% of empties with MaskIgnore
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        if ( (*mask) [x + y * raw->cols] & (MaskPinned | MaskIgnore
                                            | MaskBead | MaskExclude))
        {
          continue;
        }
        if (results[x + y * raw->cols] > cutoffVal)
        {
          //(*mask)[x+y*raw->cols] |= MaskIgnore;
          (*mask) [x + y * raw->cols] = MaskBead;
        }
      }
    }
    delete[] closeCtr;
    delete[] sqDist;
  }
  //  Mark region to be ignored if it lacks sufficient number of beads
  //  Minimum threshold is .5% of wells
  if (beadCount < (region->w * region->h) / 500)
  {
    fprintf (fp, "Beadcount %d < %d\n", beadCount, (region->w * region->h)
             / 500);
    fprintf (fp, "Mark this region Ignored: r%03d c%03d\n", region->row,
             region->col);
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        (*mask) [x + y * raw->cols] |= MaskIgnore;
      }
    }
  }
  else
  {
    fprintf (fp, "Found %d beads\n", beadCount);
  }

  // cleanups

  if (fp)
    fclose (fp);

  // if we have multiple groups, then we can cluster/separate beads in a region into 2 or more groups
  // here, we are just using the beadfind metric (in results) to do this, and we separate by thresholds on the metric
  // this should be very similar to a size selection approach
  if (regenerateGroups && numGroups > 1)
  {
    double bfMetric[w * h];
    int count = 0;
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        // Commented out - redundant, no? bpp 2010-05-18
        //if ((*mask)[x+y*raw->cols] & (MaskPinned | MaskIgnore | MaskExclude)) {
        // continue;
        //}
        if ( (*mask) [x + y * raw->cols] & MaskBead)
        {
          bfMetric[count] = results[x + y * raw->cols];
          count++;
        }
      }
    }

    // sort lowest to highest
    qsort (bfMetric, count, sizeof (double), doubleCompare);

    // assign each to a group
    int cutoff = count / numGroups;
    int whichGroup = 0;
    count = 0;
    for (y = region->row; y < (region->row + region->h); y++)
    {
      for (x = region->col; x < (region->col + region->w); x++)
      {
        // Commented out - redundant, no? bpp 2010-05-18
        //if ((*mask)[x+y*raw->cols] & (MaskPinned | MaskIgnore | MaskExclude)) {
        // continue;
        //}
        if ( (*mask) [x + y * raw->cols] & MaskBead)
        {
          groups[x + y * raw->cols] = whichGroup;
          count++;
          if ( (count % cutoff == 0) && (whichGroup < (numGroups - 1)))
            whichGroup++;
        }
      }
    }
    regenerateGroups = false;
  }
}

void Separator::CalcSignal (Image *image, Region *region, Mask *mask,
                            MaskType these, int flow, SeparatorSignalType signalType)
{
  //Unused parameter generates compiler warning, so...
  if (signalType) {};

  //Uncomment next line to generate debug file
  //NOTE:  Not thread safe!!
  //#define DEBUG

  FILE *fp = NULL;
#ifdef DEBUG
  char fileName[512] = {0};
  snprintf (fileName, 512, "%s/%s", experimentDir, "beadTrace.txt");
  fopen_s (&fp, fileName, "ab");
#endif
  if (fp) fprintf (fp, "# Flow = %d\n", flow);

  const RawImage *raw = image->GetImage();
  uint64_t x, y, frame;
  uint64_t frameStart = image->GetFrame (timeStart);
  uint64_t frameEnd = image->GetFrame (timeEnd);
  //fprintf (stdout, "Separator: CalcSignal %d %d (expect 25 36)\n", frameStart, frameEnd);
  for (y = region->row; y < (uint64_t) (region->row + region->h); y++)
  {
    for (x = region->col; x < (uint64_t) (region->col + region->w); x++)
    {
      if ( (*mask) [x + y * raw->cols] & these)
      {
        for (frame = frameStart; frame < frameEnd; frame++)
        {
          work[x + y * raw->cols + flow * raw->cols * raw->rows]
          += raw->image[frame * raw->frameStride + x + y * raw->cols]; // todo: optimize work index
        }
        if (fp) fprintf (fp, "%04d %04d ", (int) y, (int) x);
        for (frame = start; frame < (uint64_t) end; frame++)
        {
          //   record raw traces for live bead categorization
          uint64_t idx = flow * beadIdx1_Len + (x + y * raw->cols)
                         * beadIdx2_Len + frame - start;
          bead[idx] = raw->image[frame * raw->frameStride + x + y * raw->cols];
          if (fp) fprintf (fp, "%d ", bead[idx]);
        }
        if (fp) fprintf (fp, "\n");
      }
    }
  }
  if (fp) fprintf (fp, "\n");
  if (fp) fclose (fp);
  fp = NULL;
}

void Separator::Categorize (SequenceItem *seqList, int numSeqListItems,
                            Mask *mask)
{
  // For each nuke flow, for each well, entire trace.
  // For a given well, store the trace for each nuke flow (7).
  // Frames = 183 (one trace) * nuke flows (7) * number of beads
  // Keep an index of which flows are 1mers and 0mers.
  // Calculate the average for each 1mer frame per flow
  // Calculate the average for each 0mer frame per flow.
  // Subtract the omer from 1mer for each frame

  // categorize is responsible for two jobs:
  // 1. separating beads based on a list of keys and mask types, and will further mark ambiguous beads
  // 2. rank each population according to some metric (RQS, signal intensity, etc), then mark as dud/live/ambiguous

  // separator keys are 3 bases each, and each will have 3 1-mers and 3 0-mers in the data
  // note on above comment - should be generic to handle any keys now

  int x, y;
  int i;
  int flow;

  double mult[numSeqListItems][numKeyFlows];
  for (i = 0; i < numSeqListItems; i++)
  {
    int onemerCount = 0;
    int zeromerCount = 0;
    for (flow = 0; flow < numKeyFlows; flow++)
    {
      // count 1-mers
      if (seqList[i].Ionogram[flow] == 1)
        onemerCount++;
      // count 0-mers
      if (seqList[i].Ionogram[flow] == 0)
        zeromerCount++;
    }
    // calc ionogram multipliers such that the avg 1-mer minus 0-mer trace can be rapidly constructed
    for (flow = 0; flow < numKeyFlows; flow++)
    {
      if (seqList[i].Ionogram[flow] == 1)
        mult[i][flow] = 1.0 / onemerCount;
      else if (seqList[i].Ionogram[flow] == 0)
        mult[i][flow] = -1.0 / zeromerCount;
    }
  }

  //debug
  for (i = 0; i < numSeqListItems; i++)
  {
    fprintf (stdout, "%s\n", seqList[i].seq);
    int flow;
    for (flow = 0; flow < seqList[i].numKeyFlows; flow++)
    {
      fprintf (stdout, "%d ", seqList[i].Ionogram[flow]);
    }
    fprintf (stdout, "\n");
  }

  //debug - will write every bead's averaged 1-mer minus averaged 0-mer trace.
  FILE *fp = NULL;
  char fileName[512] = { 0 };

//Uncomment next line to generate debug files
//#define DEBUG
#ifdef DEBUG
  snprintf (fileName, 512, "%s/%s", experimentDir, "keypassTrace.txt");
  fopen_s (&fp, fileName, "wb");
#endif
  if (fp) fprintf (fp, "# KeyId row col frame_value (avg(1mer) - avg(0mer))\n");

  // calculate average key signal per region, if we have been configured with a set of regions to compute over
  int *region_map = NULL;
  int *region_cnt = NULL;

  // make reverse lookup from x-y coordinate to region number
  // and allocate space for per-region key signal averages
  if (num_regions > 0)
  {
    region_map = new int[w * h];
    region_cnt = new int[num_regions];

    avgRegionKeySignal = new float[ (end - start) * num_regions];
    memset (avgRegionKeySignal, 0,
            sizeof (float[ (end - start) * num_regions]));
    memset (region_cnt, 0, sizeof (int[num_regions]));

    for (int rnum = 0; rnum < num_regions; rnum++)
    {
      Region *region = &region_list[rnum];

      for (y = region->row; y < (region->row + region->h); y++)
      {
        for (x = region->col; x < (region->col + region->w); x++)
        {
          region_map[x + y * w] = rnum;
        }
      }
    }
  }

  float *beadFinal = new float[end - start];

  // Calculate the average keyflow metric for every well
  // Take the average of the traces of the three 1-mers and four 0-mers and subtract the 0-mer from the 1-mer.
  // For each well, find the maximum trace signal, if it is greater than minPeakThreshold, it is (a)Live!
  for (int seq = 0; seq < numSeqListItems; seq++)
  {
    //  For each key type, check all beaded wells
    for (y = 0; y < h; y++)
    {
      for (int x = 0; x < w; x++)
      {
        if ( (*mask) [x + y * w] & MaskBead)
        {

          int rnum = -1;
          if (num_regions > 0)
            rnum = region_map[x + y * w];

          if (fp) fprintf (fp, "%01d %04d %04d ", seq, y, x);

          bool dumpFlag = false;
          //if ( (y==293 && x==347)) dumpFlag = true;
          //if ( (y==301 && x==311)) dumpFlag = true;
          //if ( (y==547 && x==698)) dumpFlag = true;
          //if (dumpFlag) fprintf (stdout, "%d %d\n", y, x);

          for (int t = 0; t < (end - start); t++)
          {
            beadFinal[t] = 0.0;
            for (flow = 0; flow < numKeyFlows; flow++)
            {
              uint64_t idx = flow * beadIdx1_Len + (x + y * w) * beadIdx2_Len + t;
              beadFinal[t] += bead[idx] * mult[seq][flow];

              if (dumpFlag) fprintf (stdout, "%d ", bead[idx]);

            }
            if (dumpFlag) fprintf (stdout, "\n");

            // we only care about the difference between 1mer and 0mer average
            /*
             beadFinal[t] = (bead[onemers[seq][0]][x+y*w][t] +
             bead[onemers[seq][1]][x+y*w][t] +
             bead[onemers[seq][2]][x+y*w][t])/3.0 -
             (bead[zeromers[seq][0]][x+y*w][t] +
             bead[zeromers[seq][1]][x+y*w][t] +
             bead[zeromers[seq][2]][x+y*w][t] +
             bead[zeromers[seq][3]][x+y*w][t])/4.0;
             */

            if (fp) fprintf (fp, "%0.2f ", beadFinal[t]);

          }

          if (fp) fprintf (fp, "\n");


          //Mark any bead as live if it surpassed the threshold.  Once marked Live, it should not get marked Dud
          //TODO: optimise the loop by continuing as soon as minPeakThreshold is reached.
          //Also, if a bead is alive it is also keypassing, so assign it to a key group
          float minS = 0.0, maxS = 0.0;
          for (int t = 0; t < end - start; t++)
          {
            if (t == 0 || beadFinal[t] > maxS)
            {
              maxS = beadFinal[t];
            }
            if (t == 0 || beadFinal[t] < minS)
            {
              minS = beadFinal[t];
            }
          }
          if (maxS > minPeakThreshold)
          {
            (*mask) [x + y * w] |= MaskLive;//Set Live
            (*mask) [x + y * w] &= (~MaskDud);//unset dud (in case it was previously marked Dud)
            //Assign to key flow
            (*mask) [x + y * w] |= seqList[seq].type;

            // if this bead is 'Live' ...add it's key signal into the per-region key signal average
            if (rnum != -1)
            {
              region_cnt[rnum]++;
              for (int t = 0; t < end - start; t++)
                avgRegionKeySignal[ (end - start) * rnum + t]
                += beadFinal[t];
            }

          }
          else
          {
            //Only mark as Dud if its not been marked as Live
            if (! ( (*mask) [x + y * w] & MaskLive))
            {
              (*mask) [x + y * w] |= MaskDud;
            }
          }

          //Test for ambiguous keypass.  If bead is marked both TF and Lib, then unmark and set MaskAmbiguous
          if ( (*mask) [x + y * w] & MaskTF && (*mask) [x + y * w]
               & MaskLib)
          {
            //Unset MaskLib
            (*mask) [x + y * w] &= (~MaskLib);
            //Unset MaskTF
            (*mask) [x + y * w] &= (~MaskTF);
            //Unset MaskLive
            (*mask) [x + y * w] &= (~MaskLive);
            //Unset MaskDud
            (*mask) [x + y * w] &= (~MaskDud);
            //Set MaskAmbiguous
            (*mask) [x + y * w] |= MaskAmbiguous;
          }
        }
      }
    }
  }

  if (fp) fclose (fp);
  fp = NULL;
#undef DEBUG

#define DEBUG
  // normalize per-region key signal averages
  if (num_regions > 0)
  {
    // debug dump of per-region key signal
#ifdef DEBUG
    memset (fileName, 0, 512);
    snprintf (fileName, 512, "%s/%s", experimentDir,
              "per_region_key_signals.txt");
    fopen_s (&fp, fileName, "wb");
#endif

    for (int rnum = 0; rnum < num_regions; rnum++)
    {
      for (int t = 0; t < end - start; t++)
      {
        if (region_cnt > 0)
          avgRegionKeySignal[ (end - start) * rnum + t]
          /= region_cnt[rnum];

        if (fp) fprintf (fp, "%f ", avgRegionKeySignal[ (end - start) * rnum + t]);

      }
      if (fp) fprintf (fp, "\n");
    }

    if (fp) fclose (fp);
  }
#undef DEBUG

  if (region_map != NULL)
    delete[] region_map;
  if (region_cnt != NULL)
    delete[] region_cnt;
  delete[] beadFinal;

#ifdef DEBUG
  memset (fileName, 0, 512);
  snprintf (fileName, 512, "%s/%s", experimentDir, "rqsList.txt");
  fopen_s (&fp, fileName, "wb");
#endif

  // Rank beads
  // Sum signal from all the wells, per keypass flow
  double signalSum[numKeyFlows];
  double rMean[4]; // Mean signal per nuc
  double **avg = new double *[numKeyFlows];

  for (i = 0; i < numKeyFlows; i++)
  {
    avg[i] = new double[end - start];
    memset (avg[i], 0, sizeof (double) * (end - start));
  }

  for (i = 0; i < numSeqListItems; i++)
  {
    memset (signalSum, 0, sizeof (double) * numKeyFlows);
    int wellCount = 0;

    for (y = 0; y < h; y++)
    {
      for (x = 0; x < w; x++)
      {
        if ( (*mask) [x + y * w] & seqList[i].type)
        {
          if ( (*mask) [x + y * w] & MaskLive)
          {
            for (int flow = 0; flow < numKeyFlows; flow++)
            {
              signalSum[flow] += work[x + y * w + flow * w * h];

              // Calculate average trace from all Live beads per flow
              for (int frame = 0; frame < (end - start); frame++)
              {
                uint64_t idx = flow * beadIdx1_Len
                               + (x + y * w) * beadIdx2_Len + frame;
                avg[flow][frame] += bead[idx];
              }
            }
            wellCount++;
          }
        }
      }
    }

    // Write out average trace per flow
    FILE *afp = NULL;
    char fileName[512] = { 0 };
    snprintf (fileName, 512, "%s/avgNukeTrace_%s.txt", experimentDir,
              seqList[i].seq);
    fopen_s (&afp, fileName, "wb");

    for (int flow = 0; flow < numKeyFlows; flow++)
    {
      fprintf (afp, "%d ", flow);
      for (int frame = 0; frame < (end - start); frame++)
      {
        avg[flow][frame] /= wellCount;
        fprintf (afp, "%0.1f ", avg[flow][frame]);
      }
      fprintf (afp, "\n");
    }

    double max[4] = { 0 };
    double tmpMax = 0;

    char PGMNucs[4] = { 'T', 'A', 'C', 'G' };
    int nuc;
    double avgSig = 0.0;
    int avgCount = 0;
    for (nuc = 0; nuc < 4; nuc++)
    {
      // fprintf (afp, "For nuc %d, 0-mer is %d and 1-mer is %d\n", nuc, seqList[i].zeromers[nuc], seqList[i].onemers[nuc]);
      if (seqList[i].zeromers[nuc] > -1 && seqList[i].onemers[nuc] > -1
          && seqList[i].zeromers[nuc] < numKeyFlows
          && seqList[i].onemers[nuc] < numKeyFlows)
      {
        // fprintf (afp, "%c ", flowOrder[nuc]);
        fprintf (afp, "%c ", PGMNucs[nuc]);
        for (int frame = 0; frame < (end - start); frame++)
        {
          // MGD note - need to figure out how to avg 0-mers & 1-mers past 7 flows to support long keys
          // or just leave as is and not report certain flows if they occure in 3rd cycle or beyond
          double avg1mer = avg[seqList[i].onemers[nuc]][frame];
          double avg0mer = avg[seqList[i].zeromers[nuc]][frame];
          fprintf (afp, "%0.1f ", avg1mer - avg0mer);

          tmpMax = avg1mer - avg0mer;
          if (frame == 0 || tmpMax > max[nuc])
            max[nuc] = tmpMax;
        }
        fprintf (afp, "\n");
        avgSig += max[nuc];
        avgCount++;
      }
    }
    avgSig /= avgCount;

    fclose (afp);

    if (seqList[i].type == MaskTF)
      avgTFSig = avgSig;
    else
      avgLibSig = avgSig;

    //// DEBUG print out
    //fprintf (stdout, "Live Beads = %d\n", wellCount);
    //fprintf (stdout, "Average signal per flow key = %s:\n", seqList[i].seq);
    //for (int flow=0;flow<numKeyFlows;flow++) {
    //    signalSum[flow] /= wellCount;
    //    fprintf (stdout, "flow[%d] = %12.3lf\n", flow, signalSum[flow]);
    //}


    for (int nuc = 0; nuc < 4; nuc++)
    {
      if (seqList[i].zeromers[nuc] > -1 && seqList[i].onemers[nuc] > -1
          && seqList[i].zeromers[nuc] < numKeyFlows
          && seqList[i].onemers[nuc] < numKeyFlows)
        rMean[nuc] = signalSum[seqList[i].onemers[nuc]]
                     - signalSum[seqList[i].zeromers[nuc]];
      else
        rMean[nuc] = 0.0;
    }

    double rqs;

    for (y = 0; y < h; y++)
    {
      for (x = 0; x < w; x++)
      {
        if ( (*mask) [x + y * w] & seqList[i].type)
        {
          if ( (*mask) [x + y * w] & MaskLive)
          {
            rqs = GetRQS (&work[x + y * w], seqList[i].zeromers,
                          seqList[i].onemers, rMean);

#ifdef DEBUG
            fprintf (fp,"%01d %04d %04d %0.3lf ",i,y,x,rqs);
            for (int flow=0;flow<numKeyFlows;flow++)
            {
              fprintf (fp, "%0.2lf ", work[ (flow*h*w) + (x+y*w) ]);
            }
            fprintf (fp, "\n");
#endif

            work[x + y * w] = rqs;
          }
        }
      }
    }
  }

  for (i = 0; i < numKeyFlows; i++)
    delete[] avg[i];
  delete[] avg;

  CleanupBeads();

#ifdef DEBUG
  fclose (fp);
#endif

  return;
}

//  rMean is a vector containing the average signal in all bead wells for the first three flows (T,A,C)
//  The mean was calculated from the 1mer - 0mer for each nuke.
//  The score we calculate here is the variance of each well
double Separator::GetRQS (double *signal, int *zeromers, int *onemers,
                          double *rMean)
{
  double sig;
  int nuc;
  double rqs = 0.0;
  int rqsCount = 0;

  for (nuc = 0; nuc < 4; nuc++)
  {
    if (zeromers[nuc] > -1 && onemers[nuc] > -1 && zeromers[nuc]
        < numKeyFlows && onemers[nuc] < numKeyFlows)
    {
      sig
      = (signal[onemers[nuc] * w * h] - signal[zeromers[nuc] * w
          * h]) - rMean[nuc];
      sig *= sig;
      rqs += sig;
      rqsCount++;
    }
  }
  if (rqsCount > 0)
    rqs /= rqsCount;

  return rqs;
}

//  This calculation preserves sign, which is what we need to determine which key it is.
double Separator::GetCat (double *signal, int *zeromers, int *onemers)
{
  int nuc;
  double mean = 0.0;
  int meanCount = 0;
  for (nuc = 0; nuc < 4; nuc++)
  {
    if (zeromers[nuc] > -1 && onemers[nuc] > -1 && zeromers[nuc]
        < numKeyFlows && onemers[nuc] < numKeyFlows)
    {
      mean += signal[onemers[nuc] * w * h]
              - signal[zeromers[nuc] * w * h];
      meanCount++;
    }
  }
  mean /= meanCount;

  //double rqs = 0.0;
  //double s;
  //for(i=0;i<3;i++) {
  // s = mean - sig[i];
  // s = s * s;
  // rqs += s;
  //}
  //rqs /= 3.0;
  return mean;
}
