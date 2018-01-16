/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     NNModel.h
//! @ingroup  BaseCaller
//! @brief    NNModel. load network model from file

#ifndef NNMODEL_H
#define NNMODEL_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace std;

namespace NN{
	class NNModel;
	class Layer;
	class DenseLayer;
	class DataChunk;
}

class NN::NNModel{
public:
	friend class PerBaseQual;
	NNModel(const string &fname);
	~NNModel();
	vector<float>  CalculateOutput(NN::DataChunk* pred);
	int GetInputLength(){
		return (int)layerDimIn_[0];
	};

private:
	void LoadWeights(const string &fname);
	int mLayers;   // Number of Layers in the network
	unsigned int* layerTypes_;
	unsigned int* layerNames_;
	unsigned int* layerDimIn_;
	unsigned int* layerDimOut_;
	std::vector<Layer *> m_Layers;   // Vector of Layers
};

class NN::Layer{
public:
	Layer() {}
	virtual ~Layer() {}
	virtual void LoadWeights(const string &fname) = 0;
	virtual void SetName(int name) = 0;
	virtual void SetDim(const vector<int> dims) = 0;
	virtual NN::DataChunk* GetOutput(NN::DataChunk* pred) = 0;

	//virtual unsigned int GetRows() const = 0;
	//virtual unsigned int GetCols() const = 0;
	//virtual unsigned int GetOuputs() const = 0;
	int mName;
};

class NN::DenseLayer : public Layer{
public:

	void LoadWeights(const string &fname);
	void SetName(int name);
	void SetDim(const vector<int> dims);
	NN::DataChunk* GetOutput(NN::DataChunk* pred);
	std::vector<std::vector<float> > mWeights;
	std::vector<float> mBias;

	//virtual unsigned int GetRows() const { return 1; }
	//virtual unsigned int GetCols() const { return dim1; }
	//virtual unsigned int GetOuputs() const { return dim2; }

	// input and output dimensions of the layer
	int mDim1;
	int mDim2;
	int mName;
};

class NN::DataChunk{  // only one dimensional data for now
public:
	DataChunk(void){}
	DataChunk(size_t size) : mData(size){}
	DataChunk(size_t size, float init) : mData(size, init){}
	std::vector<float> mData;
	~DataChunk() {}
	size_t GetDataDim(void) const { return 1; }
	void SetData(vector<float> const & d) { mData = d; };
	vector<float> & GetSetData(){ return mData; }
	vector<float> const & GetData() const { return mData; }
	void PrintValues() {
	    cout << "DataChunk values:" << endl;
	    for(size_t i = 0; i < mData.size(); ++i)
	    	cout << mData[i] << " ";
	    cout << endl;
	  }
};

#endif // NNMODEL_H
