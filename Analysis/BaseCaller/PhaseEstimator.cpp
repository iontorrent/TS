/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     PhaseEstimator.cpp
//! @ingroup  BaseCaller
//! @brief    PhaseEstimator. Estimator of phasing parameters across chip

#include <algorithm>
#include <vector>
#include <string>
#include <cassert>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "PhaseEstimator.h"
#include "RawWells.h"
#include "Mask.h"
#include "IonErr.h"
#include "DPTreephaser.h"
#include "BaseCallerUtils.h"


#include <iostream>
//#include <fstream>


void PhaseEstimator::PrintHelp()
{
  printf ("Phasing estimation options:\n");
  printf ("     --phasing-estimator       STRING   phasing estimation algorithm [spatial-refiner-2]\n");
  printf ("     --libcf-ie-dr             cf,ie,dr don't estimate phasing and use specified values [not using]\n");
  printf ("     --phasing-residual-filter FLOAT    maximum sum-of-squares residual to keep a read in phasing estimation [1.0]\n");
  printf ("     --max-phasing-levels      INT      max number of rounds of phasing in SpatialRefiner [%d]\n",max_phasing_levels_default_);
  printf ("\n");
}


class CompareDensity {
public:
  CompareDensity(const vector<unsigned int>& density) : density_(density) {}
  bool operator() (int i, int j) { return density_[i] > density_[j]; }
private:
  const vector<unsigned int> &density_;
};


PhaseEstimator::PhaseEstimator()
{
  chip_size_x_ = 0;
  chip_size_y_ = 0;
  region_size_x_ = 0;
  region_size_y_ = 0;
  num_regions_x_ = 0;
  num_regions_y_ = 0;
  num_regions_ = 0;
  wells_ = NULL;
  mask_ = NULL;
  jobs_in_progress_ = 0;
  result_regions_x_ = 13;
  result_regions_y_ = 12;
  residual_threshold_ = 1.0;

  train_subset_count_ = 1;
  train_subset_ = 0;

  average_cf_ = 0;
  average_ie_ = 0;
  average_dr_ = 0;

  use_pid_norm_ = false;

  windowSize_ = DPTreephaser::kWindowSizeDefault_;
  max_phasing_levels_ = max_phasing_levels_default_;
}

void PhaseEstimator::InitializeFromOptArgs(OptArgs& opts)
{
  phasing_estimator_      = opts.GetFirstString ('-', "phasing-estimator", "spatial-refiner-2");
  string arg_cf_ie_dr     = opts.GetFirstString ('-', "libcf-ie-dr", "");
  residual_threshold_     = opts.GetFirstDouble ('-', "phasing-residual-filter", 1.0);
  max_phasing_levels_     = opts.GetFirstInt    ('-', "max-phasing-levels", max_phasing_levels_default_);
  use_pid_norm_           = opts.GetFirstString ('-', "keynormalizer", "keynorm-old") == "keynorm-new";
  windowSize_             = opts.GetFirstInt    ('-', "window-size", DPTreephaser::kWindowSizeDefault_);

  if (!arg_cf_ie_dr.empty()) {
    phasing_estimator_ = "override";
    result_regions_x_ = 1;
    result_regions_y_ = 1;
    result_cf_.assign(1, 0.0);
    result_ie_.assign(1, 0.0);
    result_dr_.assign(1, 0.0);
    if (3 != sscanf (arg_cf_ie_dr.c_str(), "%f,%f,%f", &result_cf_[0], &result_ie_[0], &result_dr_[0])) {
      fprintf (stderr, "Option Error: libcf-ie-dr %s\n", arg_cf_ie_dr.c_str());
      exit (EXIT_FAILURE);
    }
    return; // --libcf-ie-dr overrides other phasing-related options
  }
}

void PhaseEstimator::DoPhaseEstimation(RawWells *wells, Mask *mask, const ion::FlowOrder& flow_order, const vector<KeySequence>& keys,
    int region_size_x, int region_size_y,  bool use_single_core)
{
  flow_order_.SetFlowOrder(flow_order.str(), min(flow_order.num_flows(), 120));
  keys_ = keys;
  chip_size_x_ = mask->W();
  chip_size_y_ = mask->H();
  region_size_x_ = region_size_x;
  region_size_y_ = region_size_y;


  printf("Phase estimation mode = %s\n", phasing_estimator_.c_str());

  if (phasing_estimator_ == "override") {
    // Nothing to do!

  } else if (phasing_estimator_ == "spatial-refiner") {

    int num_workers = max(numCores(), 2);
    if (use_single_core)
      num_workers = 1;

    wells->Close();
    wells->OpenForIncrementalRead();
    SpatialRefiner(wells, mask, num_workers);


  } else if (phasing_estimator_ == "spatial-refiner-2") {

    int num_workers = max(numCores(), 2);
    if (use_single_core)
      num_workers = 1;

    wells->Close();
    wells->OpenForIncrementalRead();

    train_subset_count_ = 2;
    train_subset_cf_.resize(train_subset_count_);
    train_subset_ie_.resize(train_subset_count_);
    train_subset_dr_.resize(train_subset_count_);
    train_subset_regions_x_.resize(train_subset_count_);
    train_subset_regions_y_.resize(train_subset_count_);


    for (train_subset_ = 0; train_subset_ < train_subset_count_; ++train_subset_) {
      SpatialRefiner(wells, mask, num_workers);
      train_subset_cf_[train_subset_] = result_cf_;
      train_subset_ie_[train_subset_] = result_ie_;
      train_subset_dr_[train_subset_] = result_dr_;
      train_subset_regions_x_[train_subset_] = result_regions_x_;
      train_subset_regions_y_[train_subset_] = result_regions_y_;
    }

  } else
    ION_ABORT("Requested phase estimator is not recognized");

  // Compute mean cf, ie, dr

  average_cf_ = 0;
  average_ie_ = 0;
  average_dr_ = 0;
  int count = 0;

  for (int r = 0; r < result_regions_x_*result_regions_y_; r++) {
    if (result_cf_[r] || result_ie_[r] || result_dr_[r]) {
      average_cf_ += result_cf_[r];
      average_ie_ += result_ie_[r];
      average_dr_ += result_dr_[r];
      count++;
    }
  }
  if (count > 0) {
    average_cf_ /= count;
    average_ie_ /= count;
    average_dr_ /= count;
  }
}


void PhaseEstimator::ExportResultsToJson(Json::Value &json)
{
  // Save phase estimates to BaseCaller.json

  for (int r = 0; r < result_regions_x_*result_regions_y_; r++) {
    json["CFbyRegion"][r] = result_cf_[r];
    json["IEbyRegion"][r] = result_ie_[r];
    json["DRbyRegion"][r] = result_dr_[r];
  }
  json["RegionRows"] = result_regions_y_;
  json["RegionCols"] = result_regions_x_;

  json["CF"] = average_cf_;
  json["IE"] = average_ie_;
  json["DR"] = average_dr_;
}




float PhaseEstimator::GetWellCF(int x, int y) const
{
  if (train_subset_count_ == 1) {
    int result_size_y = ceil(chip_size_y_ / (double) result_regions_y_);
    int result_size_x = ceil(chip_size_x_ / (double) result_regions_x_);
    int region = (y / result_size_y) + (x / result_size_x) * result_regions_y_;
    return result_cf_[region];

  } else {
    int my_subset = 1-get_subset(x,y);
    int result_size_y = ceil(chip_size_y_ / (double) train_subset_regions_y_[my_subset]);
    int result_size_x = ceil(chip_size_x_ / (double) train_subset_regions_x_[my_subset]);
    int region = (y / result_size_y) + (x / result_size_x) * train_subset_regions_y_[my_subset];
    return train_subset_cf_[my_subset][region];
  }
}

float PhaseEstimator::GetWellIE(int x, int y) const
{
  if (train_subset_count_ == 1) {
    int result_size_y = ceil(chip_size_y_ / (double) result_regions_y_);
    int result_size_x = ceil(chip_size_x_ / (double) result_regions_x_);
    int region = (y / result_size_y) + (x / result_size_x) * result_regions_y_;
    return result_ie_[region];

  } else {
    int my_subset = 1-get_subset(x,y);
    int result_size_y = ceil(chip_size_y_ / (double) train_subset_regions_y_[my_subset]);
    int result_size_x = ceil(chip_size_x_ / (double) train_subset_regions_x_[my_subset]);
    int region = (y / result_size_y) + (x / result_size_x) * train_subset_regions_y_[my_subset];
    return train_subset_ie_[my_subset][region];
  }
}

float PhaseEstimator::GetWellDR(int x, int y) const
{
  if (train_subset_count_ == 1) {
    int result_size_y = ceil(chip_size_y_ / (double) result_regions_y_);
    int result_size_x = ceil(chip_size_x_ / (double) result_regions_x_);
    int region = (y / result_size_y) + (x / result_size_x) * result_regions_y_;
    return result_dr_[region];

  } else {
    int my_subset = 1-get_subset(x,y);
    int result_size_y = ceil(chip_size_y_ / (double) train_subset_regions_y_[my_subset]);
    int result_size_x = ceil(chip_size_x_ / (double) train_subset_regions_x_[my_subset]);
    int region = (y / result_size_y) + (x / result_size_x) * train_subset_regions_y_[my_subset];
    return train_subset_dr_[my_subset][region];
  }
}

void PhaseEstimator::SpatialRefiner(RawWells *wells, Mask *mask, int num_workers)
{
  printf("PhaseEstimator::analyze start\n");

  num_regions_x_ = (chip_size_x_+region_size_x_-1) / region_size_x_;
  num_regions_y_ = (chip_size_y_+region_size_y_-1) / region_size_y_;
  num_regions_ = num_regions_x_ * num_regions_y_;

  int num_levels = 1;
  if (num_regions_x_ >= 2 and num_regions_y_ >= 2 and max_phasing_levels_ >= 2)
    num_levels = 2;
  if (num_regions_x_ >= 4 and num_regions_y_ >= 4 and max_phasing_levels_ >= 3)
    num_levels = 3;
  if (num_regions_x_ >= 8 and num_regions_y_ >= 8 and max_phasing_levels_ >= 4)
    num_levels = 4;

  printf("Using numEstimatorFlows %d, chip is %d x %d, region is %d x %d, numRegions is %d x %d, numLevels %d\n",
      flow_order_.num_flows(), chip_size_x_, chip_size_y_, region_size_x_, region_size_y_, num_regions_x_, num_regions_y_, num_levels);

  // Step 1. Use mask to build region density map

  region_num_reads_.assign(num_regions_, 0);
  for (int x = 0; x < chip_size_x_; x++)
    for (int y = 0; y < chip_size_y_; y++)
      if (mask->Match(x, y, (MaskType)(MaskTF|MaskLib)) and
         !mask->Match(x, y, MaskFilteredBadResidual) and
         !mask->Match(x, y, MaskFilteredBadPPF) and
         !mask->Match(x, y, MaskFilteredBadKey))
        region_num_reads_[(x/region_size_x_) + (y/region_size_y_)*num_regions_x_]++;

  // Step 2. Build the tree of estimation subblocks.

  int max_subblocks = 2*4*4*4;
  vector<Subblock> subblocks;
  subblocks.reserve(max_subblocks);
  subblocks.push_back(Subblock());
  subblocks.back().cf = 0.0;
  subblocks.back().ie = 0.0;
  subblocks.back().dr = 0.0;
  subblocks.back().begin_x = 0;
  subblocks.back().end_x = num_regions_x_;
  subblocks.back().begin_y = 0;
  subblocks.back().end_y = num_regions_y_;
  subblocks.back().level = 1;
  subblocks.back().pos_x = 0;
  subblocks.back().pos_y = 0;
  subblocks.back().superblock = NULL;

  for (unsigned int idx = 0; idx < subblocks.size(); idx++) {
    Subblock &s = subblocks[idx];
    if (s.level == num_levels) {
      s.subblocks[0] = NULL;
      s.subblocks[1] = NULL;
      s.subblocks[2] = NULL;
      s.subblocks[3] = NULL;
      continue;
    }

    int cut_x = (s.begin_x + s.end_x) / 2;
    int cut_y = (s.begin_y + s.end_y) / 2;

    for (int i = 0; i < 4; i++) {
      subblocks.push_back(s);
      subblocks.back().cf = -1.0;
      subblocks.back().ie = -1.0;
      subblocks.back().dr = -1.0;
      subblocks.back().level++;
      subblocks.back().superblock = &s;
      s.subblocks[i] = &subblocks.back();
    }

    s.subblocks[0]->end_x = cut_x;
    s.subblocks[0]->end_y = cut_y;
    s.subblocks[0]->pos_x = (s.pos_x << 1);
    s.subblocks[0]->pos_y = (s.pos_y << 1);
    s.subblocks[1]->begin_x = cut_x;
    s.subblocks[1]->end_y = cut_y;
    s.subblocks[1]->pos_x = (s.pos_x << 1) + 1;
    s.subblocks[1]->pos_y = (s.pos_y << 1);
    s.subblocks[2]->end_x = cut_x;
    s.subblocks[2]->begin_y = cut_y;
    s.subblocks[2]->pos_x = (s.pos_x << 1);
    s.subblocks[2]->pos_y = (s.pos_y << 1) + 1;
    s.subblocks[3]->begin_x = cut_x;
    s.subblocks[3]->begin_y = cut_y;
    s.subblocks[3]->pos_x = (s.pos_x << 1) + 1;
    s.subblocks[3]->pos_y = (s.pos_y << 1) + 1;
  }

  // Step 3. Populate region search order in lowermost subblocks


  for (unsigned int idx = 0; idx < subblocks.size(); idx++) {
    Subblock &s = subblocks[idx];
    if (s.level != num_levels)
      continue;

    s.sorted_regions.reserve((s.end_x - s.begin_x) * (s.end_y - s.begin_y));
    for (int region_x = s.begin_x; region_x < s.end_x; region_x++)
      for (int region_y = s.begin_y; region_y < s.end_y; region_y++)
        s.sorted_regions.push_back(region_x + region_y*num_regions_x_);

    sort(s.sorted_regions.begin(), s.sorted_regions.end(), CompareDensity(region_num_reads_));
  }

  // Step 4. Populate region search order in remaining subblocks

  for (int level = num_levels-1; level >= 1; --level) {
    for (unsigned int idx = 0; idx < subblocks.size(); idx++) {
      Subblock &s = subblocks[idx];
      if (s.level != level)
        continue;

      assert(s.subblocks[0] != NULL);
      assert(s.subblocks[1] != NULL);
      assert(s.subblocks[2] != NULL);
      assert(s.subblocks[3] != NULL);
      unsigned int sum_regions = s.subblocks[0]->sorted_regions.size()
          + s.subblocks[1]->sorted_regions.size()
          + s.subblocks[2]->sorted_regions.size()
          + s.subblocks[3]->sorted_regions.size();
      s.sorted_regions.reserve(sum_regions);
      vector<int>::iterator V0 = s.subblocks[0]->sorted_regions.begin();
      vector<int>::iterator V1 = s.subblocks[1]->sorted_regions.begin();
      vector<int>::iterator V2 = s.subblocks[2]->sorted_regions.begin();
      vector<int>::iterator V3 = s.subblocks[3]->sorted_regions.begin();
      while (s.sorted_regions.size() < sum_regions) {
        if (V0 != s.subblocks[0]->sorted_regions.end())
          s.sorted_regions.push_back(*V0++);
        if (V2 != s.subblocks[2]->sorted_regions.end())
          s.sorted_regions.push_back(*V2++);
        if (V1 != s.subblocks[1]->sorted_regions.end())
          s.sorted_regions.push_back(*V1++);
        if (V3 != s.subblocks[3]->sorted_regions.end())
          s.sorted_regions.push_back(*V3++);
      }
    }
  }


  // Step 5. Show time. Spawn multiple worker threads to do phasing estimation

  region_reads_.clear();
  region_reads_.resize(num_regions_);
  action_map_.assign(num_regions_,0);
  subblock_map_.assign(num_regions_,' ');

  pthread_mutex_init(&region_loader_mutex_, NULL);
  pthread_mutex_init(&job_queue_mutex_, NULL);
  pthread_cond_init(&job_queue_cond_, NULL);

  wells_ = wells;
  mask_ = mask;

  job_queue_.push_back(&subblocks[0]);
  jobs_in_progress_ = 0;

  pthread_t worker_id[num_workers];

  for (int worker = 0; worker < num_workers; worker++)
    if (pthread_create(&worker_id[worker], NULL, EstimatorWorkerWrapper, this))
      ION_ABORT("*Error* - problem starting thread");

  for (int worker = 0; worker < num_workers; worker++)
    pthread_join(worker_id[worker], NULL);

  pthread_cond_destroy(&job_queue_cond_);
  pthread_mutex_destroy(&job_queue_mutex_);
  pthread_mutex_destroy(&region_loader_mutex_);



  // Print a silly action map
  //! @todo Get rid of action map once confidence in spatial refiner performance is high

  for (int region_y = 0; region_y < num_regions_y_; region_y++) {
    for (int region_x = 0; region_x < num_regions_x_; region_x++) {
      int region = region_x + region_y * num_regions_x_;
      if (action_map_[region] == 0)
        printf(" ");
      else
        printf("%d",action_map_[region]);
      printf("%c", subblock_map_[region]);
    }
    printf("\n");
  }

  // Crunching complete. Retrieve phasing estimates

  result_regions_x_ = 1 << (num_levels-1);
  result_regions_y_ = 1 << (num_levels-1);
  result_cf_.assign(result_regions_x_*result_regions_y_,0.0);
  result_ie_.assign(result_regions_x_*result_regions_y_,0.0);
  result_dr_.assign(result_regions_x_*result_regions_y_,0.0);

  for (unsigned int idx = 0; idx < subblocks.size(); idx++) {
    Subblock *current = &subblocks[idx];
    if (current->level != num_levels)
      continue;
    while (current) {
      if (current->cf >= 0) {
        result_cf_[subblocks[idx].pos_y + result_regions_y_ * subblocks[idx].pos_x] = current->cf;
        result_ie_[subblocks[idx].pos_y + result_regions_y_ * subblocks[idx].pos_x] = current->ie;
        result_dr_[subblocks[idx].pos_y + result_regions_y_ * subblocks[idx].pos_x] = current->dr;
        break;
      }
      current = current->superblock;
    }
  }

  printf("PhaseEstimator::analyze end\n");
}






size_t PhaseEstimator::LoadRegion(int region)
{
  if (region_num_reads_[region] == 0) // Nothing to load ?
    return 0;
  if (region_reads_[region].size() > 0) // Region already loaded?
    return 0;

  ClockTimer timer;
  timer.StartTimer();

  region_reads_[region].reserve(region_num_reads_[region]);

  int region_x = region % num_regions_x_;
  int region_y = region / num_regions_x_;

  int begin_x = region_x * region_size_x_;
  int begin_y = region_y * region_size_y_;
  int end_x = min(begin_x + region_size_x_, chip_size_x_);
  int end_y = min(begin_y + region_size_y_, chip_size_y_);

  // Mutex needed for wells access, but not needed for region_reads access
  pthread_mutex_lock(&region_loader_mutex_);

  wells_->SetChunk(begin_y, end_y-begin_y, begin_x, end_x-begin_x, 0, flow_order_.num_flows());
  wells_->ReadWells();

  vector<float> well_buffer(flow_order_.num_flows());

  for (int y = begin_y; y < end_y; y++) {
    for (int x = begin_x; x < end_x; x++) {

      if (get_subset(x,y) != train_subset_)
        continue;

      if (!mask_->Match(x, y, MaskLive))
        continue;
      if (!mask_->Match(x, y, MaskBead))
        continue;

      // A little help from friends in BkgModel
      if (mask_->Match(x, y, MaskFilteredBadResidual))
        continue;
      if (mask_->Match(x, y, MaskFilteredBadPPF))
        continue;
      if (mask_->Match(x, y, MaskFilteredBadKey))
        continue;

      int cls = 0;
      if (!mask_->Match(x, y, MaskLib)) {  // Not a library bead?
        cls = 1;
        if (!mask_->Match(x, y, MaskTF))   // Not a tf bead?
          continue;
      }

      for (int flow = 0; flow < flow_order_.num_flows(); ++flow)
        well_buffer[flow] = wells_->At(y,x,flow);

      // Sanity check. If there are NaNs in this read, print warning
      vector<int> nanflow;
      for (int flow = 0; flow < flow_order_.num_flows(); ++flow) {
        if (!isnan(well_buffer[flow]))
          continue;
        well_buffer[flow] = 0;
        nanflow.push_back(flow);
      }
      if(nanflow.size() > 0) {
        fprintf(stderr, "ERROR: BaseCaller read NaNs from wells file, x=%d y=%d flow=%d", x, y, nanflow[0]);
        for(unsigned int flow=1; flow < nanflow.size(); flow++) {
          fprintf(stderr, ",%d", nanflow[flow]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
      }

      region_reads_[region].push_back(BasecallerRead());
      if (use_pid_norm_) {
          region_reads_[region].back().SetDataAndKeyNormalizeNew(&well_buffer[0],
              flow_order_.num_flows(), keys_[cls].flows(), keys_[cls].flows_length()-1, false /*true*/);
      } else {
          region_reads_[region].back().SetDataAndKeyNormalize(&well_buffer[0],
              flow_order_.num_flows(), keys_[cls].flows(), keys_[cls].flows_length()-1);
      }

      bool keypass = true;
      for (int flow = 0; flow < (keys_[cls].flows_length() - 1); flow++) {
        if ((int) (region_reads_[region].back().raw_measurements[flow] + 0.5) != keys_[cls][flow])
          keypass = false;
        if (isnan(region_reads_[region].back().raw_measurements[flow]))
          keypass = false;
      }

      if (!keypass) {
        region_reads_[region].pop_back();
        continue;
      }
    }
  }

  pthread_mutex_unlock(&region_loader_mutex_);

  region_num_reads_[region] = region_reads_[region].size();

  return timer.GetMicroSec();
}



void *PhaseEstimator::EstimatorWorkerWrapper(void *arg)
{
  static_cast<PhaseEstimator*>(arg)->EstimatorWorker();
  return NULL;
}


void PhaseEstimator::EstimatorWorker()
{

  DPTreephaser treephaser(flow_order_, windowSize_);
  vector<BasecallerRead *>  useful_reads;
  useful_reads.reserve(10000);

  while (true) {

    pthread_mutex_lock(&job_queue_mutex_);
    while (job_queue_.empty()) {
      if (jobs_in_progress_ == 0) {
        pthread_mutex_unlock(&job_queue_mutex_);
        return;
      }
      // No jobs available now, but more may come, so stick around
      pthread_cond_wait(&job_queue_cond_, &job_queue_mutex_);
    }
    Subblock &s = *job_queue_.front();
    job_queue_.pop_front();
    jobs_in_progress_++;
    pthread_mutex_unlock(&job_queue_mutex_);


    // Processing

    int numGlobalIterations = 1;  // 3 iterations at top level, 1 at all other levels
    if (s.level == 1)
      numGlobalIterations = 3;

    for (int iGlobalIteration = 0; iGlobalIteration < numGlobalIterations; iGlobalIteration++) {

      ClockTimer timer;
      timer.StartTimer();
      size_t iotimer = 0;

      treephaser.SetModelParameters(s.cf, s.ie, s.dr);
      useful_reads.clear();

      for (vector<int>::iterator region = s.sorted_regions.begin(); region != s.sorted_regions.end(); ++region) {


        iotimer += LoadRegion(*region);
        // Ensure region loaded.
        // Grab reads, filter
        // Enough reads? Stop.

        if (action_map_[*region] == 0 and region_num_reads_[*region])
          action_map_[*region] = s.level;

        // Filter. Reads that survive filtering are stored in useful_reads
        //! \todo: Rethink filtering. Maybe a rule that adjusts the threshold to keep at least 20% of candidate reads.

        for (vector<BasecallerRead>::iterator R = region_reads_[*region].begin(); R != region_reads_[*region].end(); ++R) {

          for (int flow = 0; flow < flow_order_.num_flows(); flow++)
            R->normalized_measurements[flow] = R->raw_measurements[flow];

          treephaser.Solve    (*R, min(100, flow_order_.num_flows()));
          use_pid_norm_ ? (void)treephaser.PIDNormalize(*R, 8, 40) : (void)treephaser.Normalize(*R, 11, 80);
          treephaser.Solve    (*R, min(120, flow_order_.num_flows()));
          use_pid_norm_ ? (void)treephaser.PIDNormalize(*R, 8, 80) : (void)treephaser.Normalize(*R, 11, 100);
          treephaser.Solve    (*R, min(120, flow_order_.num_flows()));

          float metric = 0;
          for (int flow = 20; flow < 100 and flow < flow_order_.num_flows(); ++flow) {
            if (R->normalized_measurements[flow] > 1.2)
              continue;
            float delta = R->normalized_measurements[flow] - R->prediction[flow];
            if (!isnan(delta))
              metric += delta * delta;
            else
              metric += 1e10;
          }

          if (metric > residual_threshold_) {
            //printf("\nRejecting metric=%1.5f solution=%s", metric, R->sequence.c_str());
            continue;
          }
          useful_reads.push_back(&(*R));
        }

        if (useful_reads.size() >= 5000)
          break;
      }

      if (s.level > 1 and useful_reads.size() < 1000) // Not enough reads to even try
        break;

      // Do estimation with reads collected, update estimates
      float parameters[3];
      parameters[0] = s.cf;
      parameters[1] = s.ie;
      parameters[2] = s.dr;
      NelderMeadOptimization(useful_reads, treephaser, parameters, use_pid_norm_);
      s.cf = parameters[0];
      s.ie = parameters[1];
      s.dr = parameters[2];

      printf("Completed (%d,%d,%d) :(%2d-%2d)x(%2d-%2d), total time %5.2lf sec, i/o time %5.2lf sec, %d reads, CF=%1.2f%% IE=%1.2f%% DR=%1.2f%%\n",
          s.level, s.pos_x, s.pos_y, s.begin_x, s.end_x, s.begin_y, s.end_y,
          (double)timer.GetMicroSec()/1000000.0, (double)iotimer/1000000.0, (int)useful_reads.size(),
          100.0*s.cf, 100.0*s.ie, 100.0*s.dr);
    }

    if (useful_reads.size() >= 1000 or s.level == 1) {

      for (int region_x = s.begin_x; region_x <= s.end_x and region_x < num_regions_x_; region_x++) {
        for (int region_y = s.begin_y; region_y <= s.end_y and region_y < num_regions_y_; region_y++) {
          int region = region_x + region_y * num_regions_x_;
          if     (region_x == s.begin_x and region_y == s.begin_y)
            subblock_map_[region] = '+';
          else if(region_x == s.begin_x and region_y == s.end_y)
            subblock_map_[region] = '+';
          else if(region_x == s.end_x and region_y == s.begin_y)
            subblock_map_[region] = '+';
          else if(region_x == s.end_x and region_y == s.end_y)
            subblock_map_[region] = '+';
          else if (region_x == s.begin_x)
            subblock_map_[region] = '|';
          else if (region_x == s.end_x)
            subblock_map_[region] = '|';
          else if (region_y == s.begin_y)
            subblock_map_[region] = '-';
          else if (region_y == s.end_y)
            subblock_map_[region] = '-';
        }
      }
    }


    if (s.subblocks[0] == NULL or useful_reads.size() < 4000) {
      // Do not subdivide this block
      for (vector<int>::iterator region = s.sorted_regions.begin(); region != s.sorted_regions.end(); region++)
        region_reads_[*region].clear();

      pthread_mutex_lock(&job_queue_mutex_);
      jobs_in_progress_--;
      if (jobs_in_progress_ == 0)  // No more work, let everyone know
        pthread_cond_broadcast(&job_queue_cond_);
      pthread_mutex_unlock(&job_queue_mutex_);

    } else {
      // Subdivide. Spawn new jobs:
      pthread_mutex_lock(&job_queue_mutex_);
      jobs_in_progress_--;
      for (int subjob = 0; subjob < 4; subjob++) {
        s.subblocks[subjob]->cf = s.cf;
        s.subblocks[subjob]->ie = s.ie;
        s.subblocks[subjob]->dr = s.dr;
        job_queue_.push_back(s.subblocks[subjob]);
      }
      pthread_cond_broadcast(&job_queue_cond_);  // More work, let everyone know
      pthread_mutex_unlock(&job_queue_mutex_);
    }
  }
}




float PhaseEstimator::EvaluateParameters(vector<BasecallerRead *>& useful_reads, DPTreephaser& treephaser, const float *parameters, const bool usePIDNorm)
{
  float try_cf = parameters[0];
  float try_ie = parameters[1];
  float try_dr = parameters[2];
  if (try_cf < 0 or try_ie < 0 or try_dr < 0 or try_cf > 0.04 or try_ie > 0.04 or try_dr > 0.01)
    return 1e10;

  treephaser.SetModelParameters(try_cf, try_ie, try_dr);

  float metric = 0;
  for (vector<BasecallerRead *>::iterator read = useful_reads.begin(); read != useful_reads.end(); ++read) {

    treephaser.Simulate(**read, 120);
    float normalizer = (usePIDNorm ? treephaser.PIDNormalize(**read, 8, 100) : treephaser.Normalize(**read, 20, 100));

    for (unsigned int flow = 20; flow < 100 and flow < (*read)->raw_measurements.size(); flow++) {
      if ((*read)->raw_measurements[flow] > 1.2)
        continue;
      float delta = (*read)->raw_measurements[flow] - (*read)->prediction[flow] * normalizer;
      metric += delta * delta;
    }
  }

  return isnan(metric) ? 1e10 : metric;
}





#define kReflectionAlpha    1.0
#define kExpansionGamma     2.0
#define kContractionRho     0.5
#define kReductionSigma     0.5
#define kNumParameters      3
#define kMaxEvaluations     50

void PhaseEstimator::NelderMeadOptimization (vector<BasecallerRead *>& useful_reads, DPTreephaser& treephaser, float *parameters, const bool usePIDNorm)
{

  int num_evaluations = 0;

  //
  // Step 1. Pick initial vertices, evaluate the function at vertices, and sort the vertices
  //

  float   vertex[kNumParameters+1][kNumParameters];
  float   value[kNumParameters+1];
  int     order[kNumParameters+1];

  for (int iVertex = 0; iVertex <= kNumParameters; iVertex++) {

    for (int iParam = 0; iParam < kNumParameters; iParam++)
      vertex[iVertex][iParam] = parameters[iParam];

    switch (iVertex) {
      case 0:                 // First vertex just matches the provided starting values
        break;
      case 1:                 // Second vertex has higher CF
        vertex[iVertex][0] += 0.004;
        break;
      case 2:                 // Third vertex has higher IE
        vertex[iVertex][1] += 0.004;
        break;
      case 3:                 // Fourth vertex has higher DR
        vertex[iVertex][2] += 0.001;
        break;
      default:                // Default for future parameters
        vertex[iVertex][iVertex-1] *= 1.5;
        break;
    }

    value[iVertex] = EvaluateParameters(useful_reads, treephaser, vertex[iVertex], usePIDNorm);
    num_evaluations++;

    order[iVertex] = iVertex;

    for (int xVertex = iVertex; xVertex > 0; xVertex--) {
      if (value[order[xVertex]] < value[order[xVertex-1]]) {
        int x = order[xVertex];
        order[xVertex] = order[xVertex-1];
        order[xVertex-1] = x;
      }
    }
  }

  // Main optimization loop

  while (num_evaluations < kMaxEvaluations) {

    //
    // Step 2. Attempt reflection (and possibly expansion)
    //

    float center[kNumParameters];
    float reflection[kNumParameters];

    int worst = order[kNumParameters];
    int secondWorst = order[kNumParameters-1];
    int best = order[0];

    for (int iParam = 0; iParam < kNumParameters; iParam++) {
      center[iParam] = 0;
      for (int iVertex = 0; iVertex <= kNumParameters; iVertex++)
        if (iVertex != worst)
          center[iParam] += vertex[iVertex][iParam];
      center[iParam] /= kNumParameters ;
      reflection[iParam] = center[iParam] + kReflectionAlpha * (center[iParam] - vertex[worst][iParam]);
    }

    float reflectionValue = EvaluateParameters(useful_reads, treephaser, reflection, usePIDNorm);
    num_evaluations++;

    if (reflectionValue < value[best]) {    // Consider expansion:

      float expansion[kNumParameters];
      for (int iParam = 0; iParam < kNumParameters; iParam++)
        expansion[iParam] = center[iParam] + kExpansionGamma * (center[iParam] - vertex[worst][iParam]);
      float expansionValue = EvaluateParameters(useful_reads, treephaser, expansion, usePIDNorm);
      num_evaluations++;

      if (expansionValue < reflectionValue) {   // Expansion indeed better than reflection
        for (int iParam = 0; iParam < kNumParameters; iParam++)
          reflection[iParam] = expansion[iParam];
        reflectionValue = expansionValue;
      }
    }

    if (reflectionValue < value[secondWorst]) { // Either reflection or expansion was successful

      for (int iParam = 0; iParam < kNumParameters; iParam++)
        vertex[worst][iParam] = reflection[iParam];
      value[worst] = reflectionValue;

      for (int xVertex = kNumParameters; xVertex > 0; xVertex--) {
        if (value[order[xVertex]] < value[order[xVertex-1]]) {
          int x = order[xVertex];
          order[xVertex] = order[xVertex-1];
          order[xVertex-1] = x;
        }
      }
      continue;
    }


    //
    // Step 3. Attempt contraction (reflection was unsuccessful)
    //

    float contraction[kNumParameters];
    for (int iParam = 0; iParam < kNumParameters; iParam++)
      contraction[iParam] = vertex[worst][iParam] + kContractionRho * (center[iParam] - vertex[worst][iParam]);
    float contractionValue = EvaluateParameters(useful_reads, treephaser, contraction, usePIDNorm);
    num_evaluations++;

    if (contractionValue < value[worst]) {  // Contraction was successful

      for (int iParam = 0; iParam < kNumParameters; iParam++)
        vertex[worst][iParam] = contraction[iParam];
      value[worst] = contractionValue;

      for (int xVertex = kNumParameters; xVertex > 0; xVertex--) {
        if (value[order[xVertex]] < value[order[xVertex-1]]) {
          int x = order[xVertex];
          order[xVertex] = order[xVertex-1];
          order[xVertex-1] = x;
        }
      }
      continue;
    }


    //
    // Step 4. Perform reduction (contraction was unsuccessful)
    //

    for (int iVertex = 1; iVertex <= kNumParameters; iVertex++) {

      for (int iParam = 0; iParam < kNumParameters; iParam++)
        vertex[order[iVertex]][iParam] = vertex[best][iParam] + kReductionSigma * (vertex[order[iVertex]][iParam] - vertex[best][iParam]);

      value[order[iVertex]] = EvaluateParameters(useful_reads, treephaser, vertex[order[iVertex]], usePIDNorm);
      num_evaluations++;

      for (int xVertex = iVertex; xVertex > 0; xVertex--) {
        if (value[order[xVertex]] < value[order[xVertex-1]]) {
          int x = order[xVertex];
          order[xVertex] = order[xVertex-1];
          order[xVertex-1] = x;
        }
      }
    }
  }

  for (int iParam = 0; iParam < kNumParameters; iParam++)
    parameters[iParam] = vertex[order[0]][iParam];
}


void PhaseEstimator::ExportTrainSubsetToJson(Json::Value &json)
{
    if (phasing_estimator_ == "spatial-refiner-2") {
        json["TrainSubsetCount"] = train_subset_count_;

        for(int i = 0; i < train_subset_count_; ++i)
        {
            json["RegionRows"][i] = train_subset_regions_y_[i];
            json["RegionCols"][i] = train_subset_regions_x_[i];

            for (int r = 0; r < result_regions_x_*result_regions_y_; r++)
            {
                json["CFbyRegion"][i][r] = train_subset_cf_[i][r];
                json["IEbyRegion"][i][r] = train_subset_ie_[i][r];
                json["DRbyRegion"][i][r] = train_subset_dr_[i][r];
            }
        }
    }
}

bool PhaseEstimator::LoadPhaseEstimationTrainSubset(const string& phase_file_name, Mask *mask,
                                                    int region_size_x, int region_size_y)
{
    Json::Value json;
    chip_size_x_ = mask->W();
    chip_size_y_ = mask->H();
    region_size_x_ = region_size_x;
    region_size_y_ = region_size_y;
    num_regions_x_ = (chip_size_x_+region_size_x_-1) / region_size_x_;
    num_regions_y_ = (chip_size_y_+region_size_y_-1) / region_size_y_;
    num_regions_ = num_regions_x_ * num_regions_y_;

	ifstream ifs(phase_file_name.c_str());
    if(ifs)
    {
		ifs >> json;	
		ifs.close();

		train_subset_count_ = 0;
		train_subset_count_ = json["TrainSubset"]["TrainSubsetCount"].asInt();
		if(train_subset_count_ < 1)
		{
			cout << "PhaseEstimator ERROR: train_subset_count_ in file " << phase_file_name << " is less than 1." << endl;
			return false;
		}

		train_subset_cf_.resize(train_subset_count_);
		train_subset_ie_.resize(train_subset_count_);
		train_subset_dr_.resize(train_subset_count_);
		train_subset_regions_y_.resize(train_subset_count_);
		train_subset_regions_x_.resize(train_subset_count_);

		for(int i = 0; i < train_subset_count_; ++i)
		{
			train_subset_regions_y_[i] = json["TrainSubset"]["RegionRows"][i].asInt();
			train_subset_regions_x_[i] = json["TrainSubset"]["RegionCols"][i].asInt();
			int n = train_subset_regions_y_[i] * train_subset_regions_x_[i];

			vector<float> cf;
			vector<float> ie;
			vector<float> dr;

			for(int j = 0; j < n; ++j)
			{
				cf.push_back(json["TrainSubset"]["CFbyRegion"][i][j].asFloat());
				ie.push_back(json["TrainSubset"]["IEbyRegion"][i][j].asFloat());
				dr.push_back(json["TrainSubset"]["DRbyRegion"][i][j].asFloat());
			}

			train_subset_cf_[i] = cf;
			train_subset_ie_[i] = ie;
			train_subset_dr_[i] = dr;
		}

		result_regions_y_ = train_subset_regions_y_[train_subset_count_ - 1];
		result_regions_x_ = train_subset_regions_x_[train_subset_count_ - 1];

		result_cf_ = train_subset_cf_[train_subset_count_ - 1];
		result_ie_ = train_subset_ie_[train_subset_count_ - 1];
		result_dr_ = train_subset_dr_[train_subset_count_ - 1];
        average_cf_ = 0;
        average_ie_ = 0;
        average_dr_ = 0;
        int count = 0;

        for (int r = 0; r < result_regions_x_*result_regions_y_; r++) {
          if (result_cf_[r] || result_ie_[r] || result_dr_[r]) {
            average_cf_ += result_cf_[r];
            average_ie_ += result_ie_[r];
            average_dr_ += result_dr_[r];
            count++;
          }
        }
        if (count > 0) {
          average_cf_ /= count;
          average_ie_ /= count;
          average_dr_ /= count;
        }
		
		return true;
	}
	else
	{
		return false;
	}
}
