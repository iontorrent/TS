/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DUALGAUSSMIXMODEL_H
#define DUALGAUSSMIXMODEL_H

#include <vector>
#include <stdint.h>
/** Utility class just to hold stats for two clusters. */
class MixModel {
 public:
  MixModel() {
    mix = 0;
    mu1 = 0;
    var1 = 0;
    var1sq2p = 0;
    mu2 = 0;
    var2 = 0;
    var2sq2p = 0;
    count = 0;
    threshold = 0;
    thresholdSet = false;
    refMean = 0.0;
  }

  double mix;
  double mu1;
  double var1;
  double var1sq2p;
  double mu2;
  double var2;
  double var2sq2p;
  double threshold;
  bool thresholdSet;
  double refMean;
  int count;
};

/**
 * Fit a basic two cluster one dimensional gaussian mixture model as 
 * described by "The Elements of Statistical Learning" Hastie, Tibshirani
 * and Friedman section 8.5.1 (pp236-239)
 */
class DualGaussMixModel {

public:
	static const double GPI =  3.1415926535897932384626433832795;

  /** Constructor - How many points to use (sample) from total */
  DualGaussMixModel(int n=-1);

  static void SetThreshold(MixModel &m);

  void SetTrim(double trim) { mTrim = trim; }

  /** Calculate the best 2 center mixture model. */
  MixModel FitDualGaussMixModel(const float *data, int length);

  /** Calculate the best 2 center mixture model. */
  MixModel FitDualGaussMixModel(const float *data, const int8_t *assignments, int length);

  /** Predict cluster for a particular data point. */
  static int PredictCluster(MixModel &m, float data, double threshold, double &ownership);

  /** Assign the points to clusters based on models. */
  void AssignToCluster(int *cluster, MixModel &model, const float *data, 
		       int length, double threshold);

  static bool CalculateResponsibility(const MixModel &m, float data, float &ownership);  
  /** Guassian density function. */
  static double DNorm(double x, double mu, double var, double varsq2p);

  /** Update the probability that a point comes from each cluster. */
  void UpdateResponsibility(std::vector<float> &ownership,
			    const std::vector<float> &data, 
			    const MixModel &model);

  bool UpdateResponsibility(std::vector<float> &ownership,
			    const std::vector<std::pair<float,int8_t> > &data, 
			    const MixModel &model);

  /** Calculate the weighted mean for each distribution based on
   *   probability a point comes from that distribution. */
  static double CalcWeightedMean(const std::vector<float> &weights,
				 const std::vector<float> &values,
				 bool inverse = false);

  static double CalcWeightedMean(const std::vector<float> &weights,
				 const std::vector<std::pair<float,int8_t> > &values,
				 bool inverse);

  /** Calculate the weighted variance for each distribution based on
   *   probability a point comes from that distribution. */
  static double CalcWeightedVar(double mu,
				const std::vector<float> &weights,
				const std::vector<float> &values,
				bool inverse = false);

  static double CalcWeightedVar(double mu,
				const std::vector<float> &weights,
				const std::vector<std::pair<float, int8_t> > &values,
				bool inverse);
  
  /** Calcuate the best mean and variance for distribution based on cluster
   * membership. */
  void UpdateModel(MixModel &update, 
		   const std::vector<float> &data,
		   const std::vector<float> &cluster);
  
  void UpdateModel(MixModel &update, 
		   const std::vector<std::pair<float,int8_t> > &data,
		   const std::vector<float> &cluster);
		   
   /** Iterate the model until converged or until max iterations have been reached. */
  void ConvergeModel(const std::vector<float> &data, MixModel &model);
		   
  void ConvergeModel(const std::vector<std::pair<float,int8_t> > &data, MixModel &model);

  /** Pick two random points 25th and 75th percentile to start model off */
  void InitializeModel(const std::vector<float> &data, MixModel &model);

  void InitializeModel(const std::vector<std::pair<float, int8_t> > &data, MixModel &model);

  /** Trim off outlier data. */
  void TrimData(std::vector<float> &data, double trim);

  void TrimData(std::vector<std::pair<float,int8_t> > &data, double trim);

private:
  int mMaxPoints;         ///< How many data points to use max (sample down to this)
  double mTrim;           ///< Fraction to trim (each end trims mTrim/2
  int mMaxIter;           ///< Maximum number of iterations to perform 
  double mConvergeDelta;  ///< Converged with mean and variance is less than this

};

#endif // DUALGAUSSMIXMODEL_H
