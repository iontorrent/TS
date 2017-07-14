/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * Vecs.h
 *
 *  Created on: Aug 14, 2013
 *      Author: mbeauchemin
 */

#ifndef VECS_H
#define VECS_H
#ifdef __AVX__
#include <xmmintrin.h>
#include "immintrin.h"
#endif

#define VEC2_SIZE     2
#define VEC4_SIZE     4
#define VEC8_SIZE     8
#define VEC16_SIZE    16
#define VEC32_SIZE    32
#define VEC64_SIZE    64

#define VEC4F_SIZE_B  (VEC4_SIZE*4)
#define VEC4I_SIZE_B  (VEC4_SIZE*4)
#define VEC4S_SIZE_B  (VEC4_SIZE*2)
#define VEC4DI_SIZE_B  (VEC4_SIZE*8)
#define VEC2DI_SIZE_B  (VEC2_SIZE*8)

#define VEC8F_SIZE_B  (VEC8_SIZE*4)
#define VEC8I_SIZE_B  (VEC8_SIZE*4)
#define VEC8S_SIZE_B  (VEC8_SIZE*2)
#define VEC16S_SIZE_B (VEC16_SIZE*2)
#define VEC16F_SIZE_B  (VEC16_SIZE*4)
#define VEC32S_SIZE_B (VEC32_SIZE*2)
#define VEC32F_SIZE_B (VEC32_SIZE*4)
#define VEC64S_SIZE_B (VEC64_SIZE*2)
#define VEC64F_SIZE_B (VEC64_SIZE*4)

#define LD_VEC4F(x) (v4f)  {x,x,x,x}
#define LD_VEC4I(x) (v4i)  {x,x,x,x}
#define LD_VEC4S(x) (v4s)  {x,x,x,x}

#define LD_VEC8F(x) (v8f)  {x,x,x,x,x,x,x,x}
#define LD_VEC8I(x) (v8i)  {x,x,x,x,x,x,x,x}
#define LD_VEC8S(x) (v8s) {x,x,x,x,x,x,x,x}
#define LD_VEC8SU(x) (v8su) {x,x,x,x,x,x,x,x}


#ifdef WIN32
//typedef struct{
//	float A[CNC_VEC_SIZE];
//}CNCVecf_u;
#else
typedef float     v4f  __attribute__ ((vector_size (VEC4F_SIZE_B)));
typedef int       v4i  __attribute__ ((vector_size (VEC4I_SIZE_B)));
typedef short int v4s  __attribute__ ((vector_size (VEC4S_SIZE_B)));
typedef long long int v4di __attribute__ ((vector_size (VEC4DI_SIZE_B)));
typedef long long int v2di __attribute__ ((vector_size (VEC2DI_SIZE_B)));
typedef float     v8f  __attribute__ ((vector_size (VEC8F_SIZE_B)));
typedef float     v16f  __attribute__ ((vector_size (VEC16F_SIZE_B)));
typedef int       v8i  __attribute__ ((vector_size (VEC8I_SIZE_B)));
typedef short int v8s  __attribute__ ((vector_size (VEC8S_SIZE_B)));
typedef short unsigned int v8su  __attribute__ ((vector_size (VEC8S_SIZE_B)));
typedef short int v16s __attribute__ ((vector_size (VEC16S_SIZE_B)));
typedef short int v32s __attribute__ ((vector_size (VEC32S_SIZE_B)));
typedef float     v32f  __attribute__ ((vector_size (VEC32F_SIZE_B)));
typedef short int v64s __attribute__ ((vector_size (VEC64S_SIZE_B)));
typedef float     v64f  __attribute__ ((vector_size (VEC64F_SIZE_B)));

typedef union{
	v4f V;
	float A[VEC4_SIZE];
}v4f_u;
typedef union{
	v4i V;
	int A[VEC4_SIZE];
}v4i_u;

typedef union{
	v4s V;
	short int A[VEC4_SIZE];
}v4s_u;

typedef union{
	v8i V;
	int A[VEC8_SIZE];
}v8i_u;
typedef union{
	v8s V;
	v8s V8[1];
	short int A[VEC8_SIZE];
}v8s_u;
typedef union{
	v8f V;
	v8f V8[1];
	float A[VEC8_SIZE];
}v8f_u;
typedef union{
	v8su V;
	short unsigned int A[VEC8_SIZE];
}v8su_u;

typedef union{
	v16s V;
	v8s V8[2];
	short int A[VEC16_SIZE];
}v16s_u;
typedef union{
	v16f V;
	v8f  V8[2];
	float A[VEC16_SIZE];
}v16f_u;

typedef union{
	v32s V;
	v8s V8[4];
	short int A[VEC32_SIZE];
}v32s_u;
typedef union{
	v32f V;
	v8f  V8[4];
	float A[VEC32_SIZE];
}v32f_u;

typedef union{
	v64s V;
	v8s V8[8];
	short int A[VEC64_SIZE];
}v64s_u;
typedef union{
	v64f V;
	v8f  V8[8];
	float A[VEC64_SIZE];
}v64f_u;


#if 0 //def __AVX__
// Clang: __builtin_ia32_pshufd has been removed (Jun 2009)  (Use _mm_shuffle_epi32 directly)
// http://llvm.org/viewvc/llvm-project?view=revision&revision=72995
// http://llvm.org/viewvc/llvm-project?view=revision&revision=72996
#define LD8_VS_VFI(src_ptr,output_var) {\
					/* convert the 2 4(16-bit ints) to 2 4(32-bit ints) samples*/ \
					register v4i srcData=*((v4i*)src_ptr); \
					register v4i LtmpL = __builtin_ia32_pmovsxwd128 ((v8s)srcData); \
					register v8s srcData2 = (v8s)__builtin_ia32_pshufd(srcData,0x4e); /* swap the 64-bit words */ \
					register v4i LtmpH = __builtin_ia32_pmovsxwd128 (srcData2); \
					/* convert the 2 4(32-bit ints) to 8(32-bit ints) */ \
					register v8i LvalV=LD_VEC8I(0); \
					LvalV = __builtin_ia32_vinsertf128_si256(LvalV,LtmpL,0); \
					LvalV = __builtin_ia32_vinsertf128_si256(LvalV,LtmpH,1); \
					/* then, convert the 8(32-bit ints) to 8(32-bit floats) */ \
					(output_var) = __builtin_ia32_cvtdq2ps256(LvalV); \
					}

#else
#define LD8_VS_VFI(src_ptr,output_var) {\
					  for(int k_idx=0;k_idx<VEC8_SIZE;k_idx++) \
					  { \
						  (output_var)[k_idx] = src_ptr[k_idx]; \
					  } \
				    }
#endif
#define LD_VEC8S_CVT_VEC8F(src_ptr,output_var) LD8_VS_VFI(src_ptr,output_var.V)

#define LD_VEC16S_CVT_VEC16F(src_ptr,output_var) {\
		  for(int k_idx=0;k_idx<VEC16_SIZE;k_idx++) \
		  { \
			  (output_var).A[k_idx] = src_ptr[k_idx]; \
		  } \
	    }

#ifdef __AVX__
#define LD_VEC4S_CVT_VEC4F(src_ptr,output_var) {\
					/* convert the 2 4(16-bit ints) to 2 4(32-bit ints) samples*/ \
					v8s Lsr0 = *((v8s*)src_ptr); \
					v4i LtmpH = __builtin_ia32_pmovsxwd128 (Lsr0); \
					/* then, convert the 4(32-bit ints) to 4(32-bit floats) */ \
					(output_var).V = __builtin_ia32_cvtdq2ps(LtmpH); \
					}
#else
#define LD_VEC4S_CVT_VEC4F(src_ptr,output_var) {\
					  for(int k_idx=0;k_idx<VEC4_SIZE;k_idx++) \
					  { \
						  (output_var).A[k_idx] = src_ptr[k_idx]; \
					  } \
				    }
#endif

#ifdef __AVX__
#define CVT8_VF_VSI(output_var, input_var) {\
	/* convert the 8(32-bit floats) to 8(32-bit ints) */ \
    v8i LvalV = __builtin_ia32_cvtps2dq256(input_var);  /* rounded */ \
    /*v8i LvalV = __builtin_ia32_cvttps2dq256(input_var.V);*/  /* truncated */ \
	/* convert the 8(32-bit ints) to 2 4(32-bit ints) */ \
	v4i LtmpL = __builtin_ia32_vextractf128_si256(LvalV,0); \
	v4i LtmpH = __builtin_ia32_vextractf128_si256(LvalV,1); \
	/*convert the 2 4(32-bit ints) to 8(16-bit ints) and store back to memory*/ \
	(output_var) = __builtin_ia32_packssdw128(LtmpL,LtmpH); \
	}
#else
#define CVT8_VF_VSI(output_var, input_var) {\
	  for(int k_idx=0;k_idx<VEC8_SIZE;k_idx++) \
	  { \
	    (output_var)[k_idx] = input_var[k_idx]; \
	  } \
	}
#endif
#define CVT_VEC8F_VEC8S(output_var, input_var) CVT8_VF_VSI(output_var.V,input_var.V)

#ifdef __AVX__
#define CVT_VEC4F_VEC4S(output_var, input_var) {\
	/* convert the 4(32-bit floats) to 4(32-bit ints) */ \
    v4i LvalV = __builtin_ia32_cvtps2dq(input_var.V);  /* rounded */ \
    /*v8i LvalV = __builtin_ia32_cvttps2dq256(input_var.V);*/  /* truncated */ \
	/* convert the 8(32-bit ints) to 2 4(32-bit ints) */ \
	/*v2i LtmpL = __builtin_ia32_vextractf128_si256(LvalV,0);*/ \
	/*v2i LtmpH = __builtin_ia32_vextractf128_si256(LvalV,1);*/ \
	/*convert the 2 4(32-bit ints) to 8(16-bit ints) and store back to memory*/\
	 v8s tmpVal = __builtin_ia32_packssdw128(LvalV,LvalV); \
	 (output_var).V = (v4s)__builtin_ia32_vec_ext_v2di ((v2di)tmpVal, 0);\
	}
#else
#define CVT_VEC4F_VEC4S(output_var, input_var) {\
	  for(int k_idx=0;k_idx<VEC4_SIZE;k_idx++) \
	  { \
	    (output_var).A[k_idx] = input_var.A[k_idx]; \
	  } \
	}
#endif

#define LD_VS_VF(input_var,output_var) {\
		for(uint tmpi=0;tmpi<(sizeof(output_var)/sizeof(output_var.V8[0]));tmpi++) \
			LD8_VS_VFI((&(input_var)[8*tmpi]), output_var.V8[tmpi]) }


#define CVT_VF_VS(output_var, input_var) {\
		for(uint tmpi=0;tmpi<(sizeof(input_var)/sizeof(input_var.V8[0]));tmpi++) \
			CVT8_VF_VSI(output_var.V8[tmpi],input_var.V8[tmpi]) }



#endif // WIN32


#endif // VECS_H
