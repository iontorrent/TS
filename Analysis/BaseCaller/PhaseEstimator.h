/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     PhaseEstimator.h
//! @ingroup  BaseCaller
//! @brief    PhaseEstimator. Estimator of phasing parameters across chip

#ifndef PHASEESTIMATOR_H
#define PHASEESTIMATOR_H

#include <unistd.h>
#include <math.h>
#include <vector>
#include <string>
#include <deque>
#include <cassert>

#include "json/json.h"
#include "RawWells.h"
#include "Mask.h"
#include "DPTreephaser.h"
#include "OptArgs.h"
#include "BaseCallerUtils.h"

using namespace std;

//! @brief    Estimator of phasing parameters across chip.
//!           Implements spatial-refiner, the default estimation algorithm.
//!           Also acts as a wrapper/dispatcher for other estimation methods.
//! @ingroup  BaseCaller

class PhaseEstimator {
public:
  //! Constructor.
  PhaseEstimator();

  static void PrintHelp();

  //! @brief  Initialize the object.
  //! @param  opts                Command line options
  void InitializeFromOptArgs(OptArgs& opts);

  //! @brief  Perform phasing estimation using appropriate algorithm.
  //! @param  wells               Wells reader object
  //! @param  mask                Mask object
  //! @param  flow_order          Flow order object, also stores number of flows
  //! @param  keys                Key sequences in use
  //! @param  region_size_x       Width of hdf5 chunk
  //! @param  region_size_y       Height of hdf5 chunk
  //! @param  use_single_core     Do not use multithreading?
//  void DoPhaseEstimation(RawWells *wells, Mask *mask, int num_flows, int region_size_x, int region_size_y,
//      const string& flow_order, const vector<KeySequence>& keys, bool use_single_core);
  void DoPhaseEstimation(RawWells *wells, Mask *mask, const ion::FlowOrder& flow_order, const vector<KeySequence>& keys,
      int region_size_x, int region_size_y, bool use_single_core);

  //! @brief    Save phasing estimates and configuration to json.
  //! @param    json                Json value object, to be populated by filtering statistics
  void ExportResultsToJson(Json::Value &json);

  //! @brief    Retrieve carry forward estimate for a specified well
  //! @param    x                 X coordinate
  //! @param    y                 Y coordinate
  //! @return   Carry forward estimate
  float GetWellCF(int x, int y) const;

  //! @brief    Retrieve incomplete extension estimate for a specified well
  //! @param    x                 X coordinate
  //! @param    y                 Y coordinate
  //! @return   Incomplete extension estimate
  float GetWellIE(int x, int y) const;

  //! @brief    Retrieve droop estimate for a specified well
  //! @param    x                 X coordinate
  //! @param    y                 Y coordinate
  //! @return   Droop estimate
  float GetWellDR(int x, int y) const;


protected:

  //! @brief    Run spatial-refiner, the nelder-mead based estimator with progressive chip partitioning
  //! @param    wells               Wells reader object
  //! @param    mask                Mask object
  //! @param    num_workers         Number of worker threads to spawn
  void SpatialRefiner(RawWells *wells, Mask *mask, int num_workers);

  //! @brief    Pthread wrapper for calling EstimatorWorker
  //! @param    arg                 Pointer to PhaseEstimator
  static void *EstimatorWorkerWrapper(void *arg);

  //! @brief    Phasing estimation worker thread.
  //! @details  Upon fetching the next available Subblock from the job queue,
  //!           performs solving and Nelder-Mead-based phasing estimation
  //!           and possibly adds the further-partitioned subblocks to the queue.
  void EstimatorWorker();

  //! @brief    Makes sure the reads in specified region are loaded into memory
  //! @param    region              Requested region index
  //! @return   Time in microseconds to complete the I/O
  size_t LoadRegion(int region);

  //! @brief    Execute Nelder-Mead optimization for CF,IE,DR giving best data fit
  //! @param    useful_reads        Solved reads to be used for fitting.
  //! @param    treephaser          Phasing solver/simulator
  //! @param    parameters          Initial and final CF,IE,DR estimates
  static void NelderMeadOptimization (vector<BasecallerRead *>& useful_reads, DPTreephaser& treephaser, float *parameters);

  //! @brief    Evaluates the data fit for a specific set of CF,IE,DR candidate values
  //! @param    useful_reads        Solved reads to be used for fitting.
  //! @param    treephaser          Phasing solver/simulator
  //! @param    parameters          Candidate CF,IE,DR values
  //! @return   Squared error between measurements and predicted fit. The lower the better the fit.
  static float EvaluateParameters(vector<BasecallerRead *>& useful_reads, DPTreephaser& treephaser, const float *parameters);



  //! @brief  Sub-block of the chip for which phasing can be estimated. Also a node in chip partition tree.
  struct Subblock {
    // Position
    int                 begin_x;                  //!< Horizontal start (inclusive) on the chunk grid
    int                 end_x;                    //!< Horizontal end (noninclusive) on the chunk grid
    int                 begin_y;                  //!< Vertical start (inclusive) on the chunk grid
    int                 end_y;                    //!< Vertical end (noninclusive) on the chunk grid
    int                 level;                    //!< Partition level. 1=whole chip, 2=quarter chip, 3=1/16th chip,...
    int                 pos_x;                    //!< Horizontal position relative to other subblocks at the same level
    int                 pos_y;                    //!< Vertical position relative to other subblcoks at the same level
    // Payload
    float               cf;                       //!< Carry forward estimate for this subblock
    float               ie;                       //!< Incomplete extension estimate
    float               dr;                       //!< Droop estimate
    vector<int>         sorted_regions;           //!< Chunks comprising this subblock, in the order of fetching preference
    // Partition tree
    Subblock*           subblocks[4];             //!< Subblocks resulting from partitioning this block
    Subblock*           superblock;               //!< Parent block
  };

  // Estimator-independent settings and results placeholders
  string                phasing_estimator_;       //!< Name of the phasing estimation method to be used
  int                   chip_size_x_;             //!< Chip size in wells along dimension X
  int                   chip_size_y_;             //!< Chip size in wells along dimension Y
  int                   result_regions_x_;        //!< Number of regions in X dimension for which estimates are provided
  int                   result_regions_y_;        //!< Number of regions in Y dimension for which estimates are provided
  vector<float>         result_cf_;               //!< Carry forward estimates for all regions
  vector<float>         result_ie_;               //!< Incomplete extension estimates for all regions
  vector<float>         result_dr_;               //!< Droop estimates for all regions
  float                 residual_threshold_;      //!< Maximum sum-of-squares residual to keep a read in phasing estimation

  // Data needed by SpatialRefiner worker threads
  ion::FlowOrder        flow_order_;              //!< Flow order object, also stores number of flows used for phasing estimation
  vector<KeySequence>   keys_;                    //!< Key sequences, 0 = library, 1 = TFs.
  RawWells              *wells_;                  //!< Wells file reader
  Mask                  *mask_;                   //!< Beadfind and filtering outcomes for wells
  int                   region_size_x_;           //!< Wells hdf5 dataset chunk width
  int                   region_size_y_;           //!< Wells hdf5 dataset chunk height
  int                   num_regions_x_;           //!< Number of chunks per chip width
  int                   num_regions_y_;           //!< Number of chunks per chip height
  int                   num_regions_;             //!< Total number of chunks on the chip
  vector<unsigned int>  region_num_reads_;        //!< Number of usable reads for each chunk
  vector<vector<BasecallerRead> > region_reads_;  //!< Storage for loaded reads for each chunk
  pthread_mutex_t       region_loader_mutex_;     //!< Wells reading mutex
  pthread_mutex_t       job_queue_mutex_;         //!< Job queue access mutex
  pthread_cond_t        job_queue_cond_;          //!< Signaling variable to wake up workers
  deque<Subblock*>      job_queue_;               //!< Queue of subblocks awaiting estimation
  int                   jobs_in_progress_;        //!< Number of ongoing estimation jobs
  vector<int>           action_map_;              //!< Ascii-art: Which chunks were loaded at which stage
  vector<char>          subblock_map_;            //!< Ascii-art: Region grid

  // Data needed for spatial refiner with estimation sub-setting
  vector<vector<float> > train_subset_cf_;
  vector<vector<float> > train_subset_ie_;
  vector<vector<float> > train_subset_dr_;
  vector<int>           train_subset_regions_x_;
  vector<int>           train_subset_regions_y_;
  int                   train_subset_count_;
  int                   train_subset_;

  int                   get_subset(int x, int y) const { return (x+y) % train_subset_count_; }

};




#endif // PHASEESTIMATOR_H
