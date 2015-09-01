/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
 *
 * GpuPipelineDefines.h
 *
 *  Created on: Jul 10, 2014
 *      Author: jakob
 */

#ifndef GPUPIPELINEDEFINES_H_
#define GPUPIPELINEDEFINES_H_

//#define GPU_NUM_FLOW_BUF 20

//LDG LOADER
#if __CUDA_ARCH__ >= 350
#define LDG_ACCESS(buf, idx) \
    (__ldg((buf) + (idx)))
#else
#define LDG_ACCESS(buf, idx) \
    ((buf)[(idx)])
#endif



#define NUM_SAMPLES_RF 200


//performs rezeroing in Generate Bead traces on uncompressed traces instead of on the compressed ones
#define UNCOMPRESSED_REZERO 0


#define PROJECTION_ONLY 0

//if any of the following is set we perform a second per flow rezeroing on the empty traces in singelflowfit
#define EMPTY_TRACES_REZERO_SHARED_UNCOMPRESSED_INPUT 0   //to use this STORE_EMPTY_UNCOMPRESSED has to be set in GenerateBeadtrace kernel
#define EMPTY_TRACES_REZERO_SHARED_COMPRESSED_INPUT   0   //buggy...

//toggles per flow second rezeroing for the bead traces in single fit flow kernel
#define FG_TRACES_REZERO 1

//if set empty trace average is stored uncompressed in Generate Bead Traces
#define STORE_EMPTY_UNCOMPRESSED EMPTY_TRACES_REZERO_SHARED_UNCOMPRESSED_INPUT

#define EMPTY_IN_SHARED  ( EMPTY_TRACES_REZERO_SHARED_UNCOMPRESSED_INPUT || EMPTY_TRACES_REZERO_SHARED_COMPRESSED_INPUT )

#endif /* GPUPIPELINEDEFINES_H_ */
