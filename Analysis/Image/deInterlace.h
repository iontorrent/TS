/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef DEINTERLACE_H
#define DEINTERLACE_H

#undef UNICODE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <memory.h>

#ifndef WIN32
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>		// for sysconf ()
#define FILE_HANDLE  int
#else
#include <windows.h>
#define FILE_HANDLE HANDLE
#endif

#include <limits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "datahdr.h"
#include "ByteSwapUtils.h"


#define FRAME_SCALE 1000.0

#ifndef WIN32
int deInterlaceData(
#else
extern "C" int __declspec(dllexport) deInterlaceData(
#endif
        char *fname, short *_out, int *_timestamps, int start_frame, int end_frame,
		int mincols, int minrows, int maxcols, int maxrows, int ignoreErrors);

#ifndef WIN32
int deInterlaceHdr(
#else
extern "C" int __declspec(dllexport) deInterlaceHdr(
#endif
        char *fname, int *_rows, int *_cols, int *_frames, int *_unCompFrames);

#endif // DEINTERLACE_H
