/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef VECTORIZATION_H 
#define VECTORIZATION_H 

#define VEC_INC 4
typedef float v4sf __attribute__ ((vector_size (sizeof(float)*4)));
typedef int v4si __attribute__ ((vector_size (sizeof(int)*4)));
typedef float v8sf __attribute__ ((vector_size (sizeof(float)*8)));
typedef int v8si __attribute__ ((vector_size (sizeof(int)*8)));
 
#define LOAD_8FLOATS_FLOWS(dstV, src, idx1, idx2, end) \
            ((float *)&dstV)[0] = src[idx1+0][idx2]; \
            if ((idx1+8) < end) { \
              ((float *)&dstV)[1] = src[idx1+1][idx2]; \
              ((float *)&dstV)[2] = src[idx1+2][idx2]; \
              ((float *)&dstV)[3] = src[idx1+3][idx2]; \
              ((float *)&dstV)[4] = src[idx1+4][idx2]; \
              ((float *)&dstV)[5] = src[idx1+5][idx2]; \
              ((float *)&dstV)[6] = src[idx1+6][idx2]; \
              ((float *)&dstV)[7] = src[idx1+7][idx2]; \
            } \
            else {\
              if ((idx1+1) < end)  \
                ((float *)&dstV)[1] = src[idx1+1][idx2]; \
	      if ((idx1+2) < end)  \
		((float *)&dstV)[2] = src[idx1+2][idx2]; \
	      if ((idx1+3) < end)  \
		((float *)&dstV)[3] = src[idx1+3][idx2]; \
	      if ((idx1+4) < end)  \
		((float *)&dstV)[4] = src[idx1+4][idx2]; \
	      if ((idx1+5) < end)  \
		((float *)&dstV)[5] = src[idx1+5][idx2]; \
	      if ((idx1+6) < end)  \
		((float *)&dstV)[6] = src[idx1+6][idx2]; \
	      if ((idx1+7) < end)  \
		((float *)&dstV)[7] = src[idx1+7][idx2]; \
	    }
 
#define UNLOAD_8FLOATS_FLOWS(dst, srcV, idx1, idx2, end) \
            dst[idx1+0][idx2] = ((float *)&srcV)[0]; \
            if ((idx1+8) < end) { \
              dst[idx1+1][idx2] = ((float *)&srcV)[1]; \
	      dst[idx1+2][idx2] = ((float *)&srcV)[2]; \
	      dst[idx1+3][idx2] = ((float *)&srcV)[3]; \
	      dst[idx1+4][idx2] = ((float *)&srcV)[4]; \
	      dst[idx1+5][idx2] = ((float *)&srcV)[5]; \
	      dst[idx1+6][idx2] = ((float *)&srcV)[6]; \
	      dst[idx1+7][idx2] = ((float *)&srcV)[7]; \
	    } \
            else {\
              if ((idx1+1) < end)  \
                dst[idx1+1][idx2] = ((float *)&srcV)[1]; \
              if ((idx1+2) < end)  \
                dst[idx1+2][idx2] = ((float *)&srcV)[2]; \
              if ((idx1+3) < end)  \
                dst[idx1+3][idx2] = ((float *)&srcV)[3]; \
              if ((idx1+4) < end)  \
		dst[idx1+4][idx2] = ((float *)&srcV)[4]; \
	      if ((idx1+5) < end)  \
		dst[idx1+5][idx2] = ((float *)&srcV)[5]; \
	      if ((idx1+6) < end)  \
		dst[idx1+6][idx2] = ((float *)&srcV)[6]; \
	      if ((idx1+7) < end)  \
		dst[idx1+7][idx2] = ((float *)&srcV)[7]; \
	    }

#define LOAD_8FLOATS_FRAMES(dstV, src, idx, end) \
            ((float *)&dstV)[0] = src[idx]; \
            if ((idx+8) < end) { \
              ((float *)&dstV)[1] = src[idx+1]; \
              ((float *)&dstV)[2] = src[idx+2]; \
              ((float *)&dstV)[3] = src[idx+3]; \
              ((float *)&dstV)[4] = src[idx+4]; \
              ((float *)&dstV)[5] = src[idx+5]; \
              ((float *)&dstV)[6] = src[idx+6]; \
              ((float *)&dstV)[7] = src[idx+7]; \
            } \
            else {\
              if ((idx+1) < end)  \
                ((float *)&dstV)[1] = src[idx+1]; \
	      if ((idx+2) < end)  \
		((float *)&dstV)[2] = src[idx+2]; \
	      if ((idx+3) < end)  \
		((float *)&dstV)[3] = src[idx+3]; \
	      if ((idx+4) < end)  \
		((float *)&dstV)[4] = src[idx+4]; \
	      if ((idx+5) < end)  \
		((float *)&dstV)[5] = src[idx+5]; \
	      if ((idx+6) < end)  \
		((float *)&dstV)[6] = src[idx+6]; \
	      if ((idx+7) < end)  \
		((float *)&dstV)[7] = src[idx+7]; \
	    }
 
#define UNLOAD_8FLOATS_FRAMES(dst, srcV, idx, end) \
            dst[idx+0] = ((float *)&srcV)[0]; \
            if ((idx+8) < end) { \
              dst[idx+1] = ((float *)&srcV)[1]; \
	      dst[idx+2] = ((float *)&srcV)[2]; \
	      dst[idx+3] = ((float *)&srcV)[3]; \
	      dst[idx+4] = ((float *)&srcV)[4]; \
	      dst[idx+5] = ((float *)&srcV)[5]; \
	      dst[idx+6] = ((float *)&srcV)[6]; \
	      dst[idx+7] = ((float *)&srcV)[7]; \
	    } \
            else {\
              if ((idx+1) < end)  \
                dst[idx+1] = ((float *)&srcV)[1]; \
              if ((idx+2) < end)  \
                dst[idx+2] = ((float *)&srcV)[2]; \
              if ((idx+3) < end)  \
                dst[idx+3] = ((float *)&srcV)[3]; \
              if ((idx+4) < end)  \
		dst[idx+4] = ((float *)&srcV)[4]; \
	      if ((idx+5) < end)  \
		dst[idx+5] = ((float *)&srcV)[5]; \
	      if ((idx+6) < end)  \
		dst[idx+6] = ((float *)&srcV)[6]; \
	      if ((idx+7) < end)  \
		dst[idx+7] = ((float *)&srcV)[7]; \
	    }

#define LOAD_4FLOATS_FRAMES(dstV, src, idx, end) \
            ((float *)&dstV)[0] = src[idx]; \
            if ((idx+4) < end) { \
              ((float *)&dstV)[1] = src[idx+1]; \
              ((float *)&dstV)[2] = src[idx+2]; \
              ((float *)&dstV)[3] = src[idx+3]; \
            } \
            else {\
              if ((idx+1) < end)  \
                ((float *)&dstV)[1] = src[idx+1]; \
	      if ((idx+2) < end)  \
		((float *)&dstV)[2] = src[idx+2]; \
	      if ((idx+3) < end)  \
		((float *)&dstV)[3] = src[idx+3]; \
	    }
 
#define UNLOAD_4FLOATS_FRAMES(dst, srcV, idx, end) \
            dst[idx+0] = ((float *)&srcV)[0]; \
            if ((idx+4) < end) { \
              dst[idx+1] = ((float *)&srcV)[1]; \
	      dst[idx+2] = ((float *)&srcV)[2]; \
	      dst[idx+3] = ((float *)&srcV)[3]; \
	    } \
            else {\
              if ((idx+1) < end)  \
                dst[idx+1] = ((float *)&srcV)[1]; \
              if ((idx+2) < end)  \
                dst[idx+2] = ((float *)&srcV)[2]; \
              if ((idx+3) < end)  \
                dst[idx+3] = ((float *)&srcV)[3]; \
	    }



#define LOAD_4FLOATS_FLOWS(dstV, src, idx1, idx2, end) \
            ((float *)&dstV)[0] = src[idx1+0][idx2]; \
            if ((idx1+4) < end) { \
              ((float *)&dstV)[1] = src[idx1+1][idx2]; \
              ((float *)&dstV)[2] = src[idx1+2][idx2]; \
              ((float *)&dstV)[3] = src[idx1+3][idx2]; \
            } \
            else {\
              if ((idx1+1) < end)  \
                ((float *)&dstV)[1] = src[idx1+1][idx2]; \
	      if ((idx1+2) < end)  \
		((float *)&dstV)[2] = src[idx1+2][idx2]; \
	      if ((idx1+3) < end)  \
		((float *)&dstV)[3] = src[idx1+3][idx2]; \
	    }
 
#define UNLOAD_4FLOATS_FLOWS(dst, srcV, idx1, idx2, end) \
            dst[idx1+0][idx2] = ((float *)&srcV)[0]; \
            if ((idx1+4) < end) { \
              dst[idx1+1][idx2] = ((float *)&srcV)[1]; \
	      dst[idx1+2][idx2] = ((float *)&srcV)[2]; \
	      dst[idx1+3][idx2] = ((float *)&srcV)[3]; \
	    } \
            else {\
              if ((idx1+1) < end)  \
                dst[idx1+1][idx2] = ((float *)&srcV)[1]; \
              if ((idx1+2) < end)  \
                dst[idx1+2][idx2] = ((float *)&srcV)[2]; \
              if ((idx1+3) < end)  \
                dst[idx1+3][idx2] = ((float *)&srcV)[3]; \
	    }


#define LOAD_8INTS_FLOWS(dstV, src, idx1, idx2, end) \
	((int *)&dstV)[0] = src[idx1+0][idx2]; \
            if ((idx1+8) < end) { \
              ((int *)&dstV)[1] = src[idx1+1][idx2]; \
              ((int *)&dstV)[2] = src[idx1+2][idx2]; \
              ((int *)&dstV)[3] = src[idx1+3][idx2]; \
              ((int *)&dstV)[4] = src[idx1+4][idx2]; \
              ((int *)&dstV)[5] = src[idx1+5][idx2]; \
              ((int *)&dstV)[6] = src[idx1+6][idx2]; \
              ((int *)&dstV)[7] = src[idx1+7][idx2]; \
            } \
            else {\
                    if ((idx1+1) < end)  \
                      ((int *)&dstV)[1] = src[idx1+1][idx2]; \
                    if ((idx1+2) < end)  \
                      ((int *)&dstV)[2] = src[idx1+2][idx2]; \
                    if ((idx1+3) < end)  \
                      ((int *)&dstV)[3] = src[idx1+3][idx2]; \
                    if ((idx1+4) < end)  \
                          ((int *)&dstV)[4] = src[idx1+4][idx2]; \
                        if ((idx1+5) < end)  \
                          ((int *)&dstV)[5] = src[idx1+5][idx2]; \
                        if ((idx1+6) < end)  \
                          ((int *)&dstV)[6] = src[idx1+6][idx2]; \
                        if ((idx1+7) < end)  \
                          ((int *)&dstV)[7] = src[idx1+7][idx2]; \
            }
 
#define UNLOAD_8INTS_FLOWS(dst, srcV, idx1, idx2, end) \
                dst[idx1+0][idx2] = ((int *)&srcV)[0]; \
            if ((idx1+8) < end) { \
                        dst[idx1+1][idx2] = ((int *)&srcV)[1]; \
                        dst[idx1+2][idx2] = ((int *)&srcV)[2]; \
                        dst[idx1+3][idx2] = ((int *)&srcV)[3]; \
                        dst[idx1+4][idx2] = ((int *)&srcV)[4]; \
                        dst[idx1+5][idx2] = ((int *)&srcV)[5]; \
                        dst[idx1+6][idx2] = ((int *)&srcV)[6]; \
                        dst[idx1+7][idx2] = ((int *)&srcV)[7]; \
            } \
            else {\
                    if ((idx1+1) < end)  \
                          dst[idx1+1][idx2] = ((int *)&srcV)[1]; \
                    if ((idx1+2) < end)  \
                          dst[idx1+2][idx2] = ((int *)&srcV)[2]; \
                    if ((idx1+3) < end)  \
                          dst[idx1+3][idx2] = ((int *)&srcV)[3]; \
                    if ((idx1+4) < end)  \
                          dst[idx1+4][idx2] = ((int *)&srcV)[4]; \
                        if ((idx1+5) < end)  \
                          dst[idx1+5][idx2] = ((int *)&srcV)[5]; \
                        if ((idx1+6) < end)  \
                          dst[idx1+6][idx2] = ((int *)&srcV)[6]; \
                        if ((idx1+7) < end)  \
                          dst[idx1+7][idx2] = ((int *)&srcV)[7]; \
            }


/* Vectorized Routine declarations */
 
void PurpleSolveTotalTrace_Vec(int numfb, float **vb_out, float **blue_hydrogen, 
    float **red_hydrogen, int len, float *deltaFrame, float *tauB, float *etbR, 
    float gain);

void RedSolveHydrogenFlowInWell_Vec(int numfb, float **vb_out, float **red_hydrogen, 
    int len, float *deltaFrame, float tauB);

void BlueSolveBackgroundTrace_Vec(int numfb, float **vb_out, float **blue_hydrogen, int len, 
    float *deltaFrame, float *tauB, float *etbR);

void MultiplyVectorByScalar_Vec(float *my_vec, float my_scalar, int len);

void Dfderr_Step_Vec(int numfb, float** dst, float** et, float** em, int len);
void Dfdgain_Step_Vec(int numfb, float** dst, float** src, float** em, int len, float gain);


#endif // VECTORIZATION_H
