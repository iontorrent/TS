#include "NNModel.h"
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>
#include "IonErr.h"
#include <hdf5.h>


#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


NN::NNModel::NNModel(const string &fname){
	layerTypes_ = NULL;
	layerNames_ = NULL;
	layerDimIn_ = NULL;
	layerDimOut_ = NULL;
	LoadWeights(fname);
}

NN::NNModel::~NNModel(){
	for(unsigned int i = 0; i < m_Layers.size(); ++i) {
		delete m_Layers[i];
	}
	delete [] layerTypes_;
	delete [] layerNames_;
	delete [] layerDimIn_;
	delete [] layerDimOut_;
}

vector<float> NN::NNModel::CalculateOutput(NN::DataChunk* pred){
	//cout << endl << "Calculating output" << endl;
	NN::DataChunk *input = pred;
	NN::DataChunk *output = NULL;
	for(int i = 0; i < (int)m_Layers.size(); ++i){
		//cout << "layer" <<i <<endl;
		output = m_Layers[i]->GetOutput(input);
		if(input != pred)
			delete input;
		input = NULL;
		input = output;
	}
	// last layer
	vector<float> flat_out = output->GetData();
	// Default Activation - softmax
	cout << flat_out[0] <<endl;
	float sum = 0.0;
	for(unsigned int j = 0; j < flat_out.size(); j++) {
		if (flat_out[j] < 10)
			flat_out[j] = exp(flat_out[j]);
		else
			flat_out[j] = exp(10);
		sum += flat_out[j];
	}
	for(unsigned int j = 0; j < flat_out.size(); ++j) {
		flat_out[j] /= sum;
	}
	delete output;
	return flat_out;
}

void NN::NNModel::LoadWeights(const string &fname) {
	Layer *l = NULL;

	if(H5Fis_hdf5(fname.c_str()) > 0)
	{
		cout << endl << "NNModel::Init... load model from " << fname.c_str() << endl;
		hid_t root = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);  // open file
		if(root < 0)
		{
			ION_ABORT("ERROR: cannot open HDF5 file " + fname);
		}
		hid_t NNarch = H5Gopen(root, "/arch", H5P_DEFAULT);   //open group
		if (NNarch < 0)
		{
			H5Fclose(root);
			ION_ABORT("ERROR: fail to open HDF5 group arch");
		}
		hid_t dstypes = H5Dopen(NNarch, "types", H5P_DEFAULT);  // open layer types
		if (dstypes < 0)
		{
			H5Gclose(NNarch);
			H5Fclose(root);
			ION_ABORT("ERROR: fail to open HDF5 dataset types");
		}
		hsize_t dSize = H5Dget_storage_size(dstypes);
		dSize /= sizeof(H5T_NATIVE_UINT);
		layerTypes_ = new unsigned int[dSize];
		herr_t ret = H5Dread(dstypes, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, layerTypes_);
		if (ret < 0)
		{
			delete [] layerTypes_;
			layerTypes_ = 0;
			ION_ABORT("ERROR: fail to read HDF5 dataset types");
		}
		H5Dclose(dstypes);
		hid_t dsnames = H5Dopen(NNarch, "names", H5P_DEFAULT);  // open layer names
		if (dsnames < 0)
		{
			H5Gclose(NNarch);
			H5Fclose(root);
			ION_ABORT("ERROR: fail to open HDF5 dataset names");
		}
		layerNames_ = new unsigned int[dSize];
		ret = H5Dread(dsnames, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, layerNames_);
		if (ret < 0)
		{
			delete [] layerNames_;
			layerNames_ = 0;
			ION_ABORT("ERROR: fail to read HDF5 dataset names");
		}
		H5Dclose(dsnames);
		hid_t dsdimin = H5Dopen(NNarch, "dimin", H5P_DEFAULT);  // open layer input dimension
		if (dsdimin < 0)
		{
			H5Gclose(NNarch);
			H5Fclose(root);
			ION_ABORT("ERROR: fail to open HDF5 dataset dimin");
		}
		layerDimIn_ = new unsigned int[dSize];
		ret = H5Dread(dsdimin, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, layerDimIn_);
		if (ret < 0)
		{
			delete [] layerDimIn_;
			layerDimIn_ = 0;
			ION_ABORT("ERROR: fail to read HDF5 dataset dimin");
		}
		H5Dclose(dsdimin);
		hid_t dsdimout = H5Dopen(NNarch, "dimout", H5P_DEFAULT);  // open layer input dimension
		if (dsdimout < 0)
		{
			H5Gclose(NNarch);
			H5Fclose(root);
			ION_ABORT("ERROR: fail to open HDF5 dataset dimout");
		}
		layerDimOut_ = new unsigned int[dSize];
		ret = H5Dread(dsdimout, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, layerDimOut_);
		if (ret < 0)
		{
			delete [] layerDimOut_;
			layerDimOut_ = 0;
			ION_ABORT("ERROR: fail to read HDF5 dataset dimout");
		}
		H5Dclose(dsdimout);
		H5Gclose(NNarch);
		// load weight
		hid_t NNweight = H5Gopen(root, "/weight", H5P_DEFAULT);   //open group weight
		if (NNweight < 0)
		{
			H5Fclose(root);
			ION_ABORT("ERROR: fail to open HDF5 group weight");
		}
		H5Gclose(NNweight);
		H5Fclose(root);

		for(unsigned int i = 0; i < dSize; i++){
			cout<< "Reading layer: "<< i<< endl;
			if((int)layerTypes_[i] == 1){  // "Dense layer"
				vector<int> dims(2); // support only two dimensions for now
				dims.at(0) = layerDimIn_[i];
				dims.at(1) = layerDimOut_[i];
				l = new DenseLayer();
				l->SetName((int)layerNames_[i]);
				l->SetDim(dims);
				l->LoadWeights(fname);
			}else if((int)layerTypes_[i] == 2) { // "Dropout layer"
				continue;
			}
			m_Layers.push_back(l);
		}
	}else{
		ION_ABORT("ERROR: The file is not an HDF5 file " + fname);
	}
}


void NN::DenseLayer::LoadWeights(const string &fname){
	hid_t root = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);  // open file
	hid_t NNweight = H5Gopen(root, "/weight", H5P_DEFAULT);   //open group weight
	if (NNweight < 0)
	{
		H5Fclose(root);
		ION_ABORT("ERROR: fail to open HDF5 group weight");
	}
	char buf[100];
	sprintf(buf, "weight%d", mName);
	hid_t dsweight = H5Dopen(NNweight, buf, H5P_DEFAULT);  // open weight dataset
	if (dsweight < 0)
	{
		H5Gclose(NNweight);
		H5Fclose(root);
		ION_ABORT("ERROR: fail to open HDF5 dataset weight");
	}
	hid_t filespace = H5Dget_space(dsweight);
	int dimension = H5Sget_simple_extent_ndims(filespace);
	hsize_t dim[dimension];
	int status = H5Sget_simple_extent_dims ( filespace, dim, NULL );
	if ( status <0 )
	{
		H5Sclose ( filespace );
	    ION_ABORT ( "Internal Error in H5Sget_simple_extent_dims - Read Weight" );
	}
	hsize_t size = H5Dget_storage_size(dsweight);
	size /= sizeof(float);
	cout << "Layer weight: "<<(int)dim[0]<< " * " << (int)dim[1] <<endl;
	float* weight = new float[size];
	herr_t ret = H5Dread(dsweight, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, weight);
	H5Dclose(dsweight);
	if(ret < 0)
	{
		H5Gclose(NNweight);
		H5Fclose(root);
		ION_ABORT("ERROR: fail to read HDF5 attribute weight");
	}
	for(unsigned int i = 0; i < dim[0]; i++){
		vector<float> v(size/dim[0]);
		copy(weight + i * size/dim[0], weight + (i + 1) * size/dim[0], v.begin());
		mWeights.push_back(v);
	}
	delete [] weight;
 	// Load bias
	char buf1[100];
	sprintf(buf1, "bias%d", mName);
	hid_t dsbias = H5Dopen(NNweight, buf1, H5P_DEFAULT);  // open weight dataset
	if (dsbias < 0)
	{
		H5Gclose(NNweight);
		H5Fclose(root);
		ION_ABORT("ERROR: fail to open HDF5 dataset bias");
	}
	filespace = H5Dget_space(dsbias);
	dimension = H5Sget_simple_extent_ndims(filespace);
	size = H5Dget_storage_size(dsbias);
	size /= sizeof(float);
	float* bias = new float[size];
	ret = H5Dread(dsbias, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, bias);
	H5Dclose(dsbias);
	mBias.resize(dim[1]);
	copy(bias , bias + dim[1], mBias.begin());
	delete [] bias;
}

NN::DataChunk* NN::DenseLayer::GetOutput(NN::DataChunk* pred){
	size_t size = mWeights[0].size();
	size_t size8 = size >> 3;
	NN::DataChunk *output = new NN::DataChunk(size, 0);
	float * x = pred->GetSetData().data();
	float * y = output->GetSetData().data();
	for(unsigned int i = 0; i < mWeights.size(); ++i){
		const float * w = mWeights[i].data();
		size_t k = 0;
		for(unsigned int j = 0; j < size8; ++j){
			y[k] += w[k] * x[i];
			y[k + 1] += w[k + 1] * x[i];
			y[k + 2] += w[k + 2] * x[i];
			y[k + 3] += w[k + 3] * x[i];
			y[k + 4] += w[k + 4] * x[i];
			y[k + 5] += w[k + 5] * x[i];
			y[k + 6] += w[k + 6] * x[i];
			y[k + 7] += w[k + 7] * x[i];
			k += 8;
		}
		while (k < size) { y[k] += w[k] * x[i]; ++k; }  // leftovers
	}

	for (unsigned int i = 0; i < size; ++i) {
	    y[i] += mBias[i];
	}
	return output;
}

void NN::DenseLayer::SetDim(const vector<int> dims){
	mDim1 = dims.at(0);
	mDim2 = dims.at(1);
}

void NN::DenseLayer::SetName(int layerName){
	mName = layerName;
}


