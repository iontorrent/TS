/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "CaptureImageState.h"
#include "IonH5File.h"
#include "IonH5Arma.h"
#include "ImageTransformer.h"
#include "Vecs.h"
#include <malloc.h>

CaptureImageState::CaptureImageState(std::string _h5File)
{
  h5file = _h5File;
}

// clean up old h5 file
void CaptureImageState::CleanUpOldFile()
{
  if (isFile (h5file.c_str()))
    remove(h5file.c_str());  
}

// capture gain correction
void CaptureImageState::WriteImageGainCorrection(int rows, int cols)
{
  for (int row = 0;row < rows;row++)
  {
    for (int col = 0;col < cols;col++)
    {
      // no nans allowed!
      assert ( !isnan(ImageTransformer::gain_correction[row*cols + col]) );
    }
  }

  std::string h5_name = h5file + ":/imgGain/gain_corr";
  assert(ImageTransformer::gain_correction != NULL);
  std::vector<float> gain(ImageTransformer::gain_correction, ImageTransformer::gain_correction + rows*cols);    
  printf("[CaptureImageState] Writing Image gain correction to %s\n",h5file.c_str());
  H5File::WriteVector (h5_name, gain, false);    
}
void CaptureImageState::LoadImageGainCorrection(int rows, int cols)
{
  std::string h5_name = h5file + ":/imgGain/gain_corr";
  std::vector<float> gain;  
  printf("[CaptureImageState] Loading Image gain correction from %s\n",h5file.c_str());   
  H5File::ReadVector (h5_name, gain);  
  
  if (ImageTransformer::gain_correction == NULL) {
    ImageTransformer::gain_correction = (float *)memalign(VEC8F_SIZE_B,sizeof(float)*rows*cols);
  }
  std::fill(ImageTransformer::gain_correction, ImageTransformer::gain_correction + rows * cols, 0);
  std::copy(gain.begin(), gain.end(), ImageTransformer::gain_correction);
  for (int row = 0;row < rows;row++)
  {
    for (int col = 0;col < cols;col++)
    {
      // deal with legacy nan in h5 data generated in
      // ImageTransformer::GainCalculationFromBeadfind
      if ( isnan(ImageTransformer::gain_correction[row*cols + col]) )
	ImageTransformer::gain_correction[row*cols + col] = 1.0f;
    }
  }

}


// capture cross-talk correction vectors
void CaptureImageState::WriteXTCorrection()
{  
  std::string h5_name = h5file + ":/XTalk_corr/";
  
  if ( ImageTransformer::custom_correction_data != NULL ){  
    ChannelXTCorrectionDescriptor xt_vectors = ImageTransformer::custom_correction_data->GetCorrectionDescriptor();
    if ( xt_vectors.xt_vector_ptrs != NULL ){
      printf("[CaptureImageState] Writing electrical XeTalk correction to %s\n",h5file.c_str());
      
      float **vects = xt_vectors.xt_vector_ptrs;
      int nvects = xt_vectors.num_vectors;
      int *col_offset = xt_vectors.vector_indicies;  
      int vector_len = xt_vectors.vector_len;
      
      //vector number and length
      int num_and_len[] = {nvects, vector_len};
      std::vector<int> H5num_and_len(num_and_len, num_and_len+2);  
      H5File::WriteVector (h5_name+"num_and_length", H5num_and_len);  
      
      //column offset
      std::vector<int> H5col_offset(col_offset, col_offset+nvects-1);
      H5File::WriteVector (h5_name+"vector_indicies", H5col_offset);
      
      //vectors
      arma::Mat<float> H5vectors;
      H5vectors.set_size (nvects, vector_len);
      
      for ( int vndx=0; vndx < nvects; vndx++ ) {
        for ( int vn=0; vn < vector_len; vn++ )  
          H5vectors.at (vndx, vn) = vects[vndx][vn];
      }
      H5Arma::WriteMatrix (h5_name+"xt_vectors", H5vectors, false);
      
    } 
  }
}

void CaptureImageState::LoadXTCorrection()
{  
  std::string h5_name = h5file + ":/XTalk_corr/";
  
  if (H5File::DatasetExist<std::string> (h5_name+"xt_vectors")){
    printf("[CaptureImageState] Loading electrical XeTalk correction from %s\n",h5file.c_str());
    
    //vector number and length
    std::vector<int> H5num_and_len;
    H5File::ReadVector (h5_name+"num_and_length", H5num_and_len);
    int nvects = H5num_and_len[0];
    int vector_len = H5num_and_len[1];
    
    //column offset
    std::vector<int> H5col_offset;
    H5File::ReadVector (h5_name+"vector_indicies",H5col_offset);
    
    //vectors
    arma::Mat<float> H5vectors;
    H5Arma::ReadMatrix (h5_name+"xt_vectors", H5vectors);
    
    //Create image transformer objects  
    ChannelXTCorrection *xtptr = new ChannelXTCorrection();
    float *pvects = xtptr->AllocateVectorStorage(nvects,vector_len);
    float **vect_ptrs = xtptr->AllocateVectorPointerStorage(nvects);
    xtptr->SetVectorIndicies(&H5col_offset[0],vector_len);
   
    for ( int vndx=0; vndx < nvects; vndx++ ) {
      vect_ptrs[vndx] = pvects+vector_len*vndx;
      for ( int vn=0; vn < vector_len; vn++ )  
        vect_ptrs[vndx][vn] = H5vectors.at (vndx, vn);
    }
    
    ImageTransformer::custom_correction_data = xtptr;
    ImageTransformer::selected_chip_xt_vectors = ImageTransformer::custom_correction_data->GetCorrectionDescriptor();
  }
  else{
    printf("[CaptureImageState] No electrical XeTalk correction found\n");
  }
}


// also keep timestamps from ImageSpecClass
void CaptureImageState::WriteImageSpec(ImageSpecClass &my_image_spec, int frames)
{
  std::string h5_name = h5file + ":/imgSpec/";
  std::vector<int> tstamp(my_image_spec.timestamps,  my_image_spec.timestamps + frames);    
  printf("[CaptureImageState] Writing Image Spec to %s\n",h5file.c_str());
  H5File::WriteVector (h5_name + "timestamps", tstamp, false);
  // save number of frames
  std::vector<int> H5frames(1,frames);
  H5File::WriteVector (h5_name + "frames", H5frames, false);
}
void CaptureImageState::LoadImageSpec(ImageSpecClass &my_image_spec)
{
  std::string h5_name = h5file + ":/imgSpec/";
  std::vector<int> frames;  
  std::vector<int> tstamp;  
  printf("[CaptureImageState] Loading Image Spec from %s\n",h5file.c_str());   
  H5File::ReadVector (h5_name + "timestamps", tstamp);  
  H5File::ReadVector (h5_name + "frames", frames);  
  
  if (my_image_spec.timestamps == NULL)
    my_image_spec.timestamps = new int[frames[0]];
  std::copy(tstamp.begin(), tstamp.end(), my_image_spec.timestamps);
}


