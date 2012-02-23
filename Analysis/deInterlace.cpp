/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * deInterlace.c
 *
 *  Created on: Apr 1, 2010
 *      Author: Mark Beauchemin
 */
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
#define MY_FILE_HANDLE  int
#else
#include <windows.h>
#define MY_FILE_HANDLE HANDLE
#endif

#include <limits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "datahdr.h"
#include "ByteSwapUtils.h"

#define KEY_0     0x44
#define KEY_8_1   0x99
#define KEY_16_1  0xBB
#define KEY_SVC_1 0xCC

#define KEY_8     0x9944
#define KEY_16    0xBB44
#define KEY_SVC   0xCC44

#define PLACEKEY  0xdeadbeef
#define IGNORE_CKSM_TYPE_ALL    0x01
#define IGNORE_CKSM_TYPE_1FRAME 0x02
#define IGNORE_ALWAYS_RETURN    0x04
//#define DEBUG

typedef struct{
	char *CurrentAllocPtr;
	int CurrentAllocLen;
	int PageSize;
	int fileLen;
	MY_FILE_HANDLE hFile;
#ifdef WIN32
	MY_FILE_HANDLE mFile;
#endif
}DeCompFile;

DeCompFile *convert_to_fh(MY_FILE_HANDLE fd)
{
	DeCompFile *dc = (DeCompFile *)malloc(sizeof(*dc));

	if(dc == NULL)
		return NULL;

	memset(dc,0,sizeof(*dc));
	dc->PageSize = 4096;
	dc->hFile = fd;

#ifdef WIN32
	int dwFileSize;
	SYSTEM_INFO SysInfo; // system information; used to get granularity

	// Get the system allocation granularity.
	GetSystemInfo(&SysInfo);
	dc->PageSize = SysInfo.dwAllocationGranularity;
	dc->fileLen = dwFileSize = GetFileSize(dc->hFile, NULL);
	// Create a file mapping object for the file
	// Note that it is a good idea to ensure the file size is not zero
	dc->mFile = CreateFileMapping(dc->hFile, // current file handle
			NULL, // default security
			PAGE_READONLY, // read/write permission
			0, // size of mapping object, high
			dwFileSize, // size of mapping object, low
			NULL); // name of mapping object
#else
	struct stat statbuf;
	fstat(fd,&statbuf);
	dc->fileLen = statbuf.st_size;
#endif

	return dc;
}

DeCompFile *OpenFile(char *fname)
{
	MY_FILE_HANDLE rc;
	DeCompFile *dc = NULL;

#ifndef WIN32
	rc = open (fname, O_RDONLY);
	if (rc<0)
	{
		printf ("Failed Trying to open %s \n", fname);
	}
#else

	rc = CreateFile((const char *) fname, GENERIC_READ, 0, NULL,
			OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (rc == INVALID_HANDLE_VALUE)
	{
		printf("failed to open file %s\n", fname);
	}
#endif
	else
	{
		dc = convert_to_fh(rc);
	}

	return dc;
}

void CloseFile(DeCompFile *dc)
{
#ifndef WIN32
	close(dc->hFile);
#else
	CloseHandle(dc->mFile);
	CloseHandle(dc->hFile);
#endif
	free(dc);
}

MY_FILE_HANDLE get_fd(DeCompFile *dc)
{
	MY_FILE_HANDLE rc = dc->hFile;
#ifdef WIN32
	CloseHandle(dc->mFile); // long story

	dc->mFile = 0;
#endif
	free(dc);
	return (rc);
}


char *GetFileData(DeCompFile *dc, int offset, int len, unsigned int &cksm)
{
	unsigned char *cksmPtr = NULL;
	int i;

	if((offset + len) > dc->fileLen)
		return NULL;

	dc->CurrentAllocLen = len + (offset % dc->PageSize);
#ifndef WIN32
	dc->CurrentAllocPtr = (char *)mmap(0,dc->CurrentAllocLen,PROT_READ,MAP_PRIVATE,dc->hFile,(offset - (offset % 4096)));
#else
	dc->CurrentAllocPtr = (char *) MapViewOfFile(dc->mFile, FILE_MAP_READ, 0, (offset
			- (offset % dc->PageSize)), dc->CurrentAllocLen);
#endif
	if(dc->CurrentAllocPtr == NULL)
		return NULL;
	cksmPtr = (unsigned char *)(dc->CurrentAllocPtr + (offset % dc->PageSize));
	for(i=0;i<len;i++)
		cksm += *cksmPtr++;

	return dc->CurrentAllocPtr + (offset % dc->PageSize);
}

void FreeFileData(DeCompFile *dc)
{
#ifndef WIN32
	munmap(dc->CurrentAllocPtr,dc->CurrentAllocLen);
#else
	UnmapViewOfFile(dc->CurrentAllocPtr);
#endif
	dc->CurrentAllocPtr = NULL;
	dc->CurrentAllocLen = 0;
}
// inputs:
//        fd:  input file descriptor
//        out:  array of unsigned short pixel values (three-dimensional   frames:rows:cols
//        frameStride:  rows*cols
//        start_frame:  first frame to decode into out
//        end frame:    last frame to decode into out
//        timestamp:    place for timestamps if required.  NULL otherwise
int LoadCompressedImage(DeCompFile *fd, short *out, int rows, int cols, int totalFrames,
		int start_frame, int end_frame, int *timestamps, int mincols,
		int minrows, int maxcols, int maxrows, int ignoreErrors)

{
	int frameStride = rows * cols;
	short *imagePtr = (short *) out;
	char *CompPtr;
	unsigned char *cksmPtr;
	short *PrevPtr;
	int frame, x, y, len;
	unsigned short val;
	unsigned int state = 0; // first entry better be a state change
	unsigned int total = 0;
	int total_offset = 0;
	int total_errcnt=0;
	unsigned int Transitions = 0;
	unsigned short adder16;
//	unsigned short prevValue = 0;
	unsigned int cksum=0;
	unsigned int tmpcksum=0;
	//	unsigned short skip=0;
	//	unsigned int sentinel,ReTransitions,retotal;
	//	unsigned int recLen;
//	DeCompFile *fd = convert_to_fh(in_fd);

	struct _expmt_hdr_cmp_frame frameHdr;
	unsigned int offset = 0;
	unsigned short *WholeFrameOrig = NULL, *WholeFrame = NULL;
	unsigned short *unInterlacedData = NULL;
#ifdef DEBUG
	unsigned short *UnCompressPtr;
	unInterlacedData = (unsigned short *)malloc(2*frameStride);
#endif

	WholeFrameOrig = WholeFrame = (unsigned short *) malloc(2 * frameStride);

	offset = 0;
	len = 2 * frameStride + 8 + sizeof(struct _file_hdr)
			+ sizeof(struct _expmt_hdr_v3);
	// mmap the beginning of the file
	CompPtr = GetFileData(fd, offset, len, cksum);
	if(CompPtr == NULL) {
        free(unInterlacedData);
		if(ignoreErrors&IGNORE_ALWAYS_RETURN)
			return 0;
		else
			exit(-1);
    }

	CompPtr += sizeof(struct _file_hdr) + sizeof(struct _expmt_hdr_v3);
	// Get TimeStamp
	if (timestamps)
	{
		memcpy(&timestamps[0], CompPtr, 4);
		ByteSwap4(timestamps[0]);
	}
	CompPtr += 4;

	CompPtr += 4; // skip the compressed flag as we know the first frame is not..	

	PrevPtr = (short *) WholeFrameOrig; // save for next frame

	// read in the first frame directly
	for (y = 0; y < rows; y++)
	{
		for (x = 0; x < cols; x++)
		{
			val = *(unsigned short *) CompPtr;
			if (x >= mincols && x < maxcols && y >= minrows && y < maxrows)
			{
				*imagePtr++ = (short) (BYTE_SWAP_2(val) & 0x3fff);
			}
			*WholeFrame = (short) (BYTE_SWAP_2(val) & 0x3fff);
			total += *WholeFrame++;

			CompPtr += 2;
		}
	}

	FreeFileData(fd);
	offset += len;
	len = frameStride * 2 + sizeof(struct _expmt_hdr_cmp_frame); // add in extra stuff in front of frames
	for (frame = 1; frame <= end_frame; frame++)
	{
		if (start_frame >= frame)
			imagePtr = (short *) out;

		WholeFrame = WholeFrameOrig;
		PrevPtr = (short *) WholeFrame;

#ifdef DEBUG
		len = 2*frameStride + sizeof(struct _expmt_hdr_cmp_frame);
		CompPtr = GetFileData(fd,offset,len,cksum);
		if(CompPtr == NULL)
		{
			printf("corrupt file\n");
			exit(2);
		}


		frameHdr.timestamp = *(unsigned int *)CompPtr;
		CompPtr += 4;

		UnCompressPtr = (unsigned short *)CompPtr;
		for(y=0;y<frameStride;y++)
		unInterlacedData[y] = BYTE_SWAP_2(UnCompressPtr[y]);

		CompPtr += 2*frameStride;
		memcpy(&frameHdr.len,CompPtr,sizeof(frameHdr)-4);

#else
		len = sizeof(struct _expmt_hdr_cmp_frame);

		CompPtr = GetFileData(fd, offset, len,cksum);
		if(CompPtr == NULL)
		{
			printf("corrupt file! Failed to get file data\n");
			if(ignoreErrors&IGNORE_ALWAYS_RETURN)
				return 0;
			else
				exit(-1);
		}


		memcpy(&frameHdr, CompPtr, len);
#endif
		ByteSwap4(frameHdr.Compressed);
		ByteSwap4(frameHdr.Transitions);
		ByteSwap4(frameHdr.len);
		ByteSwap4(frameHdr.sentinel);
		ByteSwap4(frameHdr.timestamp);
		ByteSwap4(frameHdr.total);

		// Get TimeStamp
		if (timestamps && (frame >= start_frame) && (frame <= end_frame))
			timestamps[frame - start_frame] = frameHdr.timestamp;

		if(!frameHdr.Compressed)
		{
			// subtract off the un-used bytes from the checksum
			for(y=8;y<len;y++)
				cksum -= (unsigned char)CompPtr[y];
			FreeFileData(fd);
			offset += 8; // special because we didn't use the whole header


			len = rows*cols*2;
			CompPtr = GetFileData(fd, offset, len,cksum);
			if(CompPtr == NULL)
			{
				printf("corrupt file\n");
				if(ignoreErrors&IGNORE_ALWAYS_RETURN)
					return 0;
				else
					exit(-1);
			}

			// read in the first frame directly
			for (y = 0; y < rows; y++)
			{
				for (x = 0; x < cols; x++)
				{
					val = *(unsigned short *) CompPtr;
					if (x >= mincols && x < maxcols && y >= minrows && y < maxrows)
					{
						*imagePtr++ = (short) (BYTE_SWAP_2(val) & 0x3fff);
					}
					*WholeFrame = (short) (BYTE_SWAP_2(val) & 0x3fff);
					total += *WholeFrame++;

					CompPtr += 2;
				}
			}
			FreeFileData(fd);
			offset += len;
			continue;  // done with this frame
		}
		else
		{
			if (frameHdr.sentinel != PLACEKEY)
			{
				printf("corrupt file!  Bad Sentinel\n");
				if(!(ignoreErrors&IGNORE_CKSM_TYPE_ALL))
				{
					if(!(ignoreErrors&IGNORE_ALWAYS_RETURN))
						exit(-1);
				}
			}

			FreeFileData(fd);
			offset += len;
		}
		len = frameHdr.len - sizeof(frameHdr) + 8;
		CompPtr = GetFileData(fd, offset, len,cksum);
		if(CompPtr == NULL)
		{
			printf("Failed to get file data\n");
			if(ignoreErrors&IGNORE_ALWAYS_RETURN)
				return 0;
			else
				exit(-1);
		}


		state = 0; // first entry better be a state change
		total = total_offset; //init to offset, used to correct count during specific bit errors
		Transitions = 0;

		for (y = 0; y < rows; y++)
		{
			for (x = 0; x < cols; x++)
			{
				// uncompress one sample
				//				if((unsigned char)CompPtr[0] == KEY_0 &&
				//						 (unsigned char)CompPtr[1] == KEY_SVC_1)
				//				{
				//					// this is a state change
				//					CompPtr +=2;
				//					// retrieve the number of elements to skip...
				//					skip = (unsigned char)CompPtr[1] | ((unsigned char)CompPtr[0] << 8);
				//					CompPtr += 2;
				//					for(i=0;i<skip;i++)
				//					{
				//						PUT_VALUE(prevValue);
				//						if(++x >=cols)
				//						{
				//							x=0;
				//							y++;
				//						}
				//					}
				//					Transitions++;
				//				}
				if ((unsigned char) CompPtr[0] == KEY_0
						&& ((unsigned char) CompPtr[1] == KEY_8_1
								|| (unsigned char) CompPtr[1] == KEY_16_1))
				{
					if ((unsigned char) CompPtr[1] == KEY_8_1)
						state = 8;
					else
						state = 16;
					CompPtr += 2;
					Transitions++;
				}
				switch (state)
				{
				case 8:
					val = *PrevPtr++ + *CompPtr++;
//					prevValue = val;
					if (x >= mincols && x < maxcols && y >= minrows && y
							< maxrows)
					{
						*imagePtr++ = val;
					}
					*WholeFrame++ = val;
					break;
				case 16:
					adder16 = *(unsigned char *) (CompPtr + 1)
							| ((*(unsigned char *) CompPtr) << 8);
					val = *PrevPtr++ + *((short *) &adder16);
//					prevValue = val;
					if (x >= mincols && x < maxcols && y >= minrows && y
							< maxrows)
					{
						*imagePtr++ = val;
					}
					*WholeFrame++ = val;

					CompPtr += 2;
					break;
				default:
					{
						printf("corrupt file\n");
						if(!(ignoreErrors&IGNORE_CKSM_TYPE_ALL))
						{
							if(ignoreErrors&IGNORE_ALWAYS_RETURN)
								return 0;
							else
								exit(-1);
						}
					}
					break;
				}
				if (unInterlacedData && *(WholeFrame - 1) != unInterlacedData[y
						* cols + x])
					printf("doesn't match %x %x\n", *(WholeFrame - 1),
							unInterlacedData[y * cols + x]);
				total += *(WholeFrame - 1);
			}
		}

		if (Transitions != frameHdr.Transitions)
		{
			printf("transitions don't match!!\n");
			printf("corrupt file\n");
			if(!(ignoreErrors&IGNORE_CKSM_TYPE_ALL))
			{
				if(ignoreErrors&IGNORE_ALWAYS_RETURN)
					return 0;
				else
					exit(-1);
			}
		}
		if (total != frameHdr.total)
		{
			total_errcnt++;
			// If IGNORE_CKSM_TYPE_1FRAME is set then allow only 1 frame to fail
			if((ignoreErrors&IGNORE_CKSM_TYPE_1FRAME) && (total_errcnt < 2)){
				total_offset += frameHdr.total-total;
				cksum += frameHdr.total-total;
				printf("totals failed in frame %d, ignore for now. Err'd Frames:%d  diff:+/-0x%x\n",frame,total_errcnt,abs(total_offset));
			} else {
				printf("totals don't match!! Err'd frames:%d\n",total_errcnt);
				printf("corrupt file\n");
				if(!(ignoreErrors&IGNORE_CKSM_TYPE_ALL))
				{
					if(ignoreErrors&IGNORE_ALWAYS_RETURN)
						return 0;
					else
						exit(-1);
				}
			}
		}
		
		FreeFileData(fd);
		offset += len;
	}

	len = 4;
	cksmPtr = (unsigned char *)GetFileData(fd, offset, len,tmpcksum);
	if ((end_frame >= (totalFrames-1)) && cksmPtr)
	{
		// there is a checksum?
		tmpcksum = cksmPtr[3];
		tmpcksum |= cksmPtr[2] << 8;
		tmpcksum |= cksmPtr[1] << 16;
		tmpcksum |= cksmPtr[0] << 24;
		if(tmpcksum != cksum)
		{
			printf("checksums don't match %x %x %x-%x-%x-%x\n",cksum,tmpcksum,cksmPtr[0],cksmPtr[1],cksmPtr[2],cksmPtr[3]);
			printf("corrupt file\n");
			if(!(ignoreErrors&IGNORE_CKSM_TYPE_ALL))
			{
				if(ignoreErrors&IGNORE_ALWAYS_RETURN)
					return 0;
				else
					exit(-1);
			}
		}
	}
	FreeFileData(fd);

#ifdef DEBUG
	free(unInterlacedData);
#endif
	// try to get a checksum at the end of the file...
	// if it's there, use it to validate the file.
	// oterwise, ignore it...

	if (WholeFrameOrig)
		free(WholeFrameOrig);

	CloseFile(fd);

	return 1;
}

#undef KEY_0
#define KEY_0     0x7f
#define KEY_16_1  0xBB

#ifndef WIN32
#define __debugbreak()
#endif

void InterpolateFramesBeforeT0(int* regionalT0, short *out, int rows, int cols, int start_frame, int end_frame, 
    int mincols, int minrows, int maxcols, int maxrows, int x_region_size, int y_region_size, int* timestamps) {

    int num_regions_x = cols/x_region_size; 
    int num_regions_y = rows/y_region_size;
    if(cols%x_region_size)
        num_regions_x++;
    if(rows%y_region_size)
	num_regions_y++;

    short* firstFrame = out;
    short* imageFramePtr = NULL;
    int y_reg, x_reg, nelems_x, nelems_y, x, y, realx, realy;
    int regionNum, t0, midFrameNum;
    short* leftFramePtr = NULL, *rightFramePtr = NULL, *imagePtr = NULL, *lastFrame = NULL, *startFrame = NULL; 
    short* midFrame = NULL;
    short* t0PlusTwoFrame = NULL, *t0PlusTwoFramePtr = NULL, *midFramePtr = NULL;
    unsigned int startX = 0, endX = 0;
    for (int frame=1; frame<=end_frame; ++frame) {
        if (frame >= start_frame && frame <= end_frame)
	    imageFramePtr = out + ((frame-start_frame)*(maxcols-mincols)*(maxrows-minrows));
	else
	    imageFramePtr = NULL;

        for(y_reg = minrows/y_region_size;y_reg < num_regions_y;y_reg++)
	{
	    for(x_reg = mincols/x_region_size;x_reg < num_regions_x;x_reg++)
	    {
                regionNum = y_reg*num_regions_x+x_reg;
                t0 = regionalT0[regionNum];   
                if (t0 < 0)
                    continue;
                if (t0 >= frame) {
                    midFrameNum = t0/2;
                    if (frame < midFrameNum) {
                        startFrame = firstFrame;
                        lastFrame = out + ((t0-start_frame)*(maxcols-mincols)*(maxrows-minrows));
                        startX = timestamps[0];
                        endX = timestamps[midFrameNum];
                    }
                    else if (frame == midFrameNum) {
		        midFrame = out + ((t0-start_frame)*(maxcols-mincols)*(maxrows-minrows));
                    }
                    else {
			lastFrame = out + ((t0 + 1 -start_frame)*(maxcols-mincols)*(maxrows-minrows));
			t0PlusTwoFrame = out + ((t0 + 2 -start_frame)*(maxcols-mincols)*(maxrows-minrows));
                        startFrame = out + ((t0-start_frame)*(maxcols-mincols)*(maxrows-minrows)); 
                        startX = timestamps[midFrameNum];
                        endX = timestamps[t0 + 1];
                    }
		    nelems_x = x_region_size;
		    nelems_y = y_region_size;

		    if (((x_reg+1)*x_region_size) > cols)
			    nelems_x = cols - x_reg*x_region_size;
		    if (((y_reg+1)*y_region_size) > rows)
			    nelems_y = rows - y_reg*y_region_size;

		    realy=y_reg*y_region_size;
		    for(y = 0;y<(int)nelems_y;y++,realy++)
		    {
		        realx=x_reg*x_region_size;
			imagePtr = (imageFramePtr + ((realy - minrows)*(maxcols-mincols) + (realx - mincols)));
			leftFramePtr = (startFrame + ((realy - minrows)*(maxcols-mincols) + (realx - mincols)));
			rightFramePtr = (lastFrame + ((realy - minrows)*(maxcols-mincols) + (realx - mincols)));
                        if (frame == midFrameNum) {
                            midFramePtr = (midFrame + ((realy - minrows)*(maxcols-mincols) + (realx - mincols)));
                        }
                        else if (frame > midFrameNum) {
                            t0PlusTwoFramePtr = (t0PlusTwoFrame + ((realy - minrows)*(maxcols-mincols) + (realx - mincols)));
                        }

			for(x=0;x<(int)nelems_x;)
			{
			    // interpolate
                            if (frame < midFrameNum) {
			        if(imageFramePtr) {
                                    if(realx >= mincols && realx < maxcols && realy >= minrows && realy < maxrows) 
                                        *imagePtr = (short)(((float)(*rightFramePtr - *leftFramePtr) / ((float)endX - (float)startX))*((float)timestamps[frame] - (float)startX)) + *leftFramePtr;
                                    imagePtr++;
                                }
                                rightFramePtr++;
                             }
                             else if (frame == midFrameNum) {
				if(imageFramePtr) {
                                    if(realx >= mincols && realx < maxcols && realy >= minrows && realy < maxrows)
                                        *imagePtr = *midFramePtr;
                                    imagePtr++;
                                }
                                midFramePtr++;
                             }
                             else {
                                float avg = (*rightFramePtr + *t0PlusTwoFramePtr) / 2;
				if(imageFramePtr) {
                                    if(realx >= mincols && realx < maxcols && realy >= minrows && realy < maxrows) 
                                        *imagePtr = (short)(((avg - (float)*leftFramePtr) / ((float)endX - (float)startX))*((float)timestamps[frame] - (float)startX)) + *leftFramePtr;                                   
                                    imagePtr++;
                                }
                                rightFramePtr++;
                                t0PlusTwoFramePtr++;
                             }
                             leftFramePtr++;
                             x++;
                             realx++;
			}
		    }
		}
            }
        }
    }
}



// inputs:
//        fd:  input file descriptor
//        out:  array of unsigned short pixel values (three-dimensional   frames:rows:cols
//        frameStride:  rows*cols
//        start_frame:  first frame to decode into out
//        end frame:    last frame to decode into out
//        timestamp:    place for timestamps if required.  NULL otherwise
int LoadCompressedRegionImage(DeCompFile *fd, short *out, int rows, int cols, int totalFrames,
		int start_frame, int end_frame, int *timestamps, int mincols,
		int minrows, int maxcols, int maxrows,
		int x_region_size, int y_region_size, unsigned int offset, bool ignoreErrors)

{
	int frameStride = rows * cols;
	short *imagePtr = (short *) out;
	unsigned char *CompPtr,*StartCompPtr;
	unsigned char *cksmPtr;
	int frame, x, y, len;
	unsigned short val;
	short Val[8]={0};
	unsigned int state = 0/*,LastState=0*/; // first entry better be a state change
	unsigned int total = 0;
	unsigned int Transitions = 0;
	unsigned int cksum=0;
	unsigned int tmpcksum=0;

	struct _expmt_hdr_cmp_frame frameHdr;
	unsigned short *WholeFrameOrig = NULL, *LocalWholeFrameOriginal=NULL,*WholeFrame = NULL,*PrevWholeFrame=NULL,*PrevWholeFrameOriginal=NULL;
	unsigned short *unInterlacedData = NULL;
	short *imageFramePtr = NULL;
#ifdef DEBUG
	unsigned short *UnCompressPtr;
	unInterlacedData = (unsigned short *)malloc(2*frameStride);
#endif

	WholeFrameOrig = LocalWholeFrameOriginal = WholeFrame = (unsigned short *) malloc(2 * frameStride);

	uint32_t y_reg,x_reg,i;
	uint32_t nelems_x,nelems_y;
	int realx,realy;
	int WholeImage=0;
	uint32_t roff=0;
//	unsigned char rgroupCksum,groupCksum;

	uint32_t num_regions_x = cols/x_region_size;
	uint32_t num_regions_y = rows/y_region_size;
	if(cols%x_region_size)
		num_regions_x++;
	if(rows%y_region_size)
		num_regions_y++;

	uint32_t *reg_offsets = (uint32_t *)malloc(num_regions_x*num_regions_y*4);
	CompPtr = (unsigned char *)GetFileData(fd, 0, offset,cksum);
	FreeFileData(fd); // for cksum

        // regional t0
        int numRegions = num_regions_x*num_regions_y;
        int regionalT0[numRegions];
        for (int reg=0; reg < numRegions; ++reg)
            regionalT0[reg] = -1;

	if ((start_frame == 0) && (mincols == 0) && (minrows == 0) &&
		(maxcols == cols) && (maxrows == rows))
		WholeImage=1;

	for (frame = 0; frame <= end_frame; frame++)
	{
		if (frame >= start_frame && frame <= end_frame)
			imageFramePtr = out + ((frame-start_frame)*(maxcols-mincols)*(maxrows-minrows));
		else
			imageFramePtr = NULL;

		PrevWholeFrameOriginal = LocalWholeFrameOriginal;
		if(WholeImage && imageFramePtr)
		{ // optimization... in the case of getting the whole frame, just write into one location....
			WholeFrame = (short unsigned int *)imageFramePtr;
			imageFramePtr = NULL;
		}
		else
		{
			WholeFrame = WholeFrameOrig;
		}
		LocalWholeFrameOriginal = WholeFrame;

		len = 8;
		CompPtr = (unsigned char *)GetFileData(fd, offset, len,cksum);
		if(CompPtr == NULL)
		{
			__debugbreak();
			printf("corrupt file! Failed to get file data\n");
			if(ignoreErrors&IGNORE_ALWAYS_RETURN)
				return 0;
			else
				exit(-1);
		}

		frameHdr.timestamp  = *(unsigned int *)CompPtr;
		CompPtr += 4;
		frameHdr.Compressed = *(unsigned int *)CompPtr;
		CompPtr += 4;
		ByteSwap4(frameHdr.Compressed);
		ByteSwap4(frameHdr.timestamp);
		FreeFileData(fd);
		offset += len;

//		printf("hdr: offset=%d Comp=%d\n",offset,frameHdr.Compressed);

		// Get TimeStamp
		if (timestamps && (frame >= start_frame) && (frame <= end_frame))
			timestamps[frame - start_frame] = frameHdr.timestamp;

		if(!frameHdr.Compressed)
		{
			len = rows*cols*2;
			CompPtr = (unsigned char *)GetFileData(fd, offset, len,cksum);
			if(CompPtr == NULL)
			{
				__debugbreak();
				printf("corrupt file\n");
				if(ignoreErrors&IGNORE_ALWAYS_RETURN)
					return 0;
				else
					exit(-1);
			}

			// read in the first frame directly
			for (y = 0; y < rows; y++)
			{
				for (x = 0; x < cols; x++)
				{
					val = *(unsigned short *) CompPtr;
					if (imageFramePtr && x >= mincols && x < maxcols && y >= minrows && y < maxrows)
					{
						*imageFramePtr++ = (short) (BYTE_SWAP_2(val) & 0x3fff);
					}
					*WholeFrame = (short) (BYTE_SWAP_2(val) & 0x3fff);
					total += *WholeFrame++;

					CompPtr += 2;
				}
			}

			FreeFileData(fd);
			offset += len;
			continue;  // done with this frame
		}
		else
		{
#ifdef DEBUG

			len = 2*frameStride;
			CompPtr = (unsigned char *)GetFileData(fd,offset,len,cksum);
			if(CompPtr == NULL)
			{
				printf("corrupt file\n");
				exit(2);
			}

			UnCompressPtr = (unsigned short *)CompPtr;
			for(y=0;y<frameStride;y++)
			unInterlacedData[y] = /*BYTE_SWAP_2(*/UnCompressPtr[y]/*)*/;

			FreeFileData(fd);
			offset += len;

#endif


			len = sizeof(struct _expmt_hdr_cmp_frame)-8;

			CompPtr = (unsigned char *)GetFileData(fd, offset, len,cksum);
			if(CompPtr == NULL)
			{
				__debugbreak();
				printf("corrupt file! Failed to get file data\n");
				if(ignoreErrors&IGNORE_ALWAYS_RETURN)
					return 0;
				else
					exit(-1);
			}


			memcpy(&frameHdr.len, CompPtr, len);

			ByteSwap4(frameHdr.Transitions);
			ByteSwap4(frameHdr.len);
			ByteSwap4(frameHdr.sentinel);
			ByteSwap4(frameHdr.total);

			if (frameHdr.sentinel != PLACEKEY)
			{
				printf("corrupt file!  No Sentinel\n");
				__debugbreak();
				if(!ignoreErrors)
					exit(2);
			}

			FreeFileData(fd);
			offset += len;
		}
		len = frameHdr.len - sizeof(frameHdr) + 8;
		if(minrows==0 && mincols==0 && maxrows==rows && maxcols==cols)
		{
			StartCompPtr = CompPtr = (unsigned char *)GetFileData(fd, offset, len,cksum);
	//		printf("frame=%d len=%d\n",frame,len);
			if(CompPtr == NULL)
			{
				__debugbreak();
				printf("Failed to get file data\n");
				if(ignoreErrors&IGNORE_ALWAYS_RETURN)
					return 0;
				else
					exit(-1);
			}

			memcpy(&reg_offsets[0],CompPtr,num_regions_x*num_regions_y*4);
		}
		else
		{
			StartCompPtr = NULL;
			CompPtr = (unsigned char *)GetFileData(fd,offset,num_regions_x*num_regions_y*4,cksum);
			if(CompPtr == NULL)
			{
				__debugbreak();
				printf("Failed to get file data\n");
				if(ignoreErrors&IGNORE_ALWAYS_RETURN)
					return 0;
				else
					exit(-1);
			}
			memcpy(&reg_offsets[0],CompPtr,num_regions_x*num_regions_y*4);
			FreeFileData(fd);
		}
		for(i=0;i<num_regions_x*num_regions_y;i++)
			reg_offsets[i] = BYTE_SWAP_4(reg_offsets[i]);
		// the offsets are now ready
//		CompPtr += num_regions_x*num_regions_y*4; // go past the direct indexes

		total = 0;
		Transitions = 0;
		
		for(y_reg = minrows/y_region_size;y_reg < num_regions_y;y_reg++)
		{
			for(x_reg = mincols/x_region_size;x_reg < num_regions_x;x_reg++)
			{
				roff = reg_offsets[y_reg*num_regions_x+x_reg];
                if (roff != 0xFFFFFFFF) 
				{
                    //printf("Region inside window: %d ",regionNum);
			           if (regionalT0[y_reg*num_regions_x+x_reg] == -1) {
                                       if (frame == 1) {
				           regionalT0[y_reg*num_regions_x+x_reg] = -2;
                                       }
                                       else 
                                           regionalT0[y_reg*num_regions_x+x_reg] = frame;
                                    }


					if(StartCompPtr)
					{
						CompPtr = StartCompPtr + reg_offsets[y_reg*num_regions_x+x_reg] - sizeof(frameHdr) + 8;
					}
					else
					{
						int loffset = reg_offsets[y_reg*num_regions_x+x_reg] - sizeof(frameHdr) + 8;
						int tlen = 3*x_region_size*y_region_size;
						if(tlen > ((int)frameHdr.len - loffset - 16))
							tlen = frameHdr.len-loffset-16;
						CompPtr = (unsigned char *)GetFileData(fd,offset+loffset,tlen,cksum);

						if(CompPtr == NULL)
						{
							__debugbreak();
							printf("Failed to get file data\n");
							if(ignoreErrors&IGNORE_ALWAYS_RETURN)
								return 0;
							else
								exit(-1);
						}

					}
				}
				state = 0; // first entry better be a state change
				nelems_x = x_region_size;
				nelems_y = y_region_size;

				if (((x_reg+1)*x_region_size) > (uint32_t)cols)
					nelems_x = cols - x_reg*x_region_size;
				if (((y_reg+1)*y_region_size) > (uint32_t)rows)
					nelems_y = rows - y_reg*y_region_size;

				realy=y_reg*y_region_size;

				for(y = 0;y<(int)nelems_y;y++,realy++)
				{
//					ptr = (int16_t *)(frame_data + ((y+(y_reg*y_region_size))*w + (x_reg*x_region_size)));
					// calculate imagePtr
					realx=x_reg*x_region_size;
					if(imageFramePtr)
						imagePtr = (imageFramePtr + ((realy - minrows)*(maxcols-mincols) + (realx - mincols)));
					else
						imagePtr = NULL;
					// calculate WholeFrame
					WholeFrame = (LocalWholeFrameOriginal + (realy*cols + realx));
					PrevWholeFrame = (PrevWholeFrameOriginal + (realy*cols + realx));

					for(x=0;x<(int)nelems_x;)
					{
                        if (roff == 0xFFFFFFFF) 
						{
							if(imagePtr)
							{
								if(realx >= mincols && realx < maxcols && realy >= minrows && realy < maxrows)
									*imagePtr = *WholeFrame;
								 imagePtr++;
							}
                            *WholeFrame++ = *PrevWholeFrame++;
                            realx++;
                            x++;
                            continue;
                        }
						if ((unsigned char) CompPtr[0] == KEY_0)
						{
							if ((unsigned char) CompPtr[1] == KEY_16_1)
								state = 16;
							else
								state = CompPtr[1] & 0xf;
							CompPtr += 2;
							Transitions++;
						}

						switch (state)
						{
						case 3:
							// get 8 values
							Val[0] = (CompPtr[0] >> 5) & 0x7;
							Val[1] = (CompPtr[0] >> 2) & 0x7;
							Val[2] = ((CompPtr[0] << 1) & 0x6) | ((CompPtr[1] >> 7) & 1);
							Val[3] = ((CompPtr[1] >> 4) & 0x7);
							Val[4] = ((CompPtr[1] >> 1) & 0x7);
							Val[5] = ((CompPtr[1] << 2) & 0x4) | ((CompPtr[2] >> 6) & 3);
							Val[6] = ((CompPtr[2] >> 3) & 0x7);
							Val[7] = ((CompPtr[2] ) & 0x7);
							CompPtr += 3;
							break;

						case 4:
							Val[0] = (CompPtr[0] >> 4) & 0xf;
							Val[1] = (CompPtr[0]) & 0xf;
							Val[2] = (CompPtr[1] >> 4) & 0xf;
							Val[3] = (CompPtr[1]) & 0xf;
							Val[4] = (CompPtr[2] >> 4) & 0xf;
							Val[5] = (CompPtr[2]) & 0xf;
							Val[6] = (CompPtr[3] >> 4) & 0xf;
							Val[7] = (CompPtr[3]) & 0xf;
							CompPtr += 4;
							break;

						case 5:
							Val[0] = (CompPtr[0] >> 3) & 0x1f;
							Val[1] = ((CompPtr[0] << 2) & 0x1c) | ((CompPtr[1] >> 6) & 0x3);
							Val[2] = (CompPtr[1] >> 1) & 0x1f;
							Val[3] = ((CompPtr[1] << 4) & 0x10) | ((CompPtr[2] >> 4) & 0xf);
							Val[4] = ((CompPtr[2] << 1) & 0x1e) | ((CompPtr[3] >> 7) & 0x1);
							Val[5] = (CompPtr[3] >> 2) & 0x1f;
							Val[6] = ((CompPtr[3] << 3) & 0x18) | ((CompPtr[4] >> 5) & 0x7);
							Val[7] = (CompPtr[4]) & 0x1f;
							CompPtr += 5;
							break;

						case 6:
							Val[0] = (CompPtr[0] >> 2) & 0x3f;
							Val[1] = ((CompPtr[0] << 4) & 0x30) | ((CompPtr[1] >> 4) & 0xf);
							Val[2] = ((CompPtr[1] << 2) & 0x3c) | ((CompPtr[2] >> 6) & 0x3);
							Val[3] = (CompPtr[2] & 0x3f);
							Val[4] = (CompPtr[3] >> 2) & 0x3f;
							Val[5] = ((CompPtr[3] << 4) & 0x30) | ((CompPtr[4] >> 4) & 0xf);
							Val[6] = ((CompPtr[4] << 2) & 0x3c) | ((CompPtr[5] >> 6) & 0x3);
							Val[7] = (CompPtr[5] & 0x3f);
							CompPtr += 6;
							break;


						case 7:
							Val[0] = (CompPtr[0] >> 1) & 0x7f;
							Val[1] = ((CompPtr[0] << 6) & 0x40) | ((CompPtr[1] >> 2) & 0x3f);
							Val[2] = ((CompPtr[1] << 5) & 0x60) | ((CompPtr[2] >> 3) & 0x1f);
							Val[3] = ((CompPtr[2] << 4) & 0x70) | ((CompPtr[3] >> 4) & 0x0f);
							Val[4] = ((CompPtr[3] << 3) & 0x78) | ((CompPtr[4] >> 5) & 0x07);
							Val[5] = ((CompPtr[4] << 2) & 0x7c) | ((CompPtr[5] >> 6) & 0x3);
							Val[6] = ((CompPtr[5] << 1) & 0x7e) | ((CompPtr[6] >> 7) & 0x1);
							Val[7] = (CompPtr[6] & 0x7f);
							CompPtr += 7;
							break;

						case 8:
							Val[0] = CompPtr[0];
							Val[1] = CompPtr[1];
							Val[2] = CompPtr[2];
							Val[3] = CompPtr[3];
							Val[4] = CompPtr[4];
							Val[5] = CompPtr[5];
							Val[6] = CompPtr[6];
							Val[7] = CompPtr[7];
							CompPtr += 8;
							break;

						case 16:
							Val[0] = (CompPtr[0] << 8) | CompPtr[1];
							Val[1] = (CompPtr[2] << 8) | CompPtr[3];
							Val[2] = (CompPtr[4] << 8) | CompPtr[5];
							Val[3] = (CompPtr[6] << 8) | CompPtr[7];
							Val[4] = (CompPtr[8] << 8) | CompPtr[9];
							Val[5] = (CompPtr[10] << 8) | CompPtr[11];
							Val[6] = (CompPtr[12] << 8) | CompPtr[13];
							Val[7] = (CompPtr[14] << 8) | CompPtr[15];
							CompPtr += 16;
							break;

						default:
							{
								printf("corrupt file\n");
								__debugbreak();
								if(!ignoreErrors)
									exit(2);
							}
							break;

						}

//						groupCksum = *CompPtr++;

						if(state != 16)
						{
							for(i=0;i<8;i++)
								Val[i] -= 1 << (state-1);
						}

						for(i=0;i<8;i++)
						{
							Val[i] += PrevWholeFrame[i];
						}
						PrevWholeFrame += 8;

						for(i=0;i<8;i++)
						{
//							rgroupCksum += (unsigned char)Val[i];
							if (imageFramePtr)
							{
								if(realx >= mincols && realx < maxcols && realy >= minrows && realy	< maxrows)
									*imagePtr = Val[i];
								imagePtr++;
							}
							total += Val[i];
							if (unInterlacedData && Val[i] != unInterlacedData[y
									* cols + x])
								printf("doesn't match %x %x\n", Val[i],
										unInterlacedData[y * cols + x]);

							*WholeFrame++ = Val[i];
							x++;
							realx++;
						}
//						if(rgroupCksum != groupCksum)
//						{
//							printf("Here's a problem %x %x!!\n",groupCksum,rgroupCksum);
//							exit(-1);
//						}
					}
				}
				if(StartCompPtr == NULL)
				{
					FreeFileData(fd);
				}
			}
		}
		if(mincols==0 && maxcols == cols && minrows==0 && maxrows==rows)
		{
			if (Transitions != frameHdr.Transitions)
			{
				printf("transitions don't match %x %x!!\n",Transitions,frameHdr.Transitions);
				printf("corrupt file\n");
				__debugbreak();
				if(!ignoreErrors)
					exit(2);
			}
			if (total != frameHdr.total)
			{
				printf("totals don't match!! %x %x %d\n",total,frameHdr.total,offset + len);
				printf("corrupt file\n");
				__debugbreak();
				if(!ignoreErrors)
					exit(2);
			}
		}
		FreeFileData(fd);
		offset += len;
	}

        InterpolateFramesBeforeT0(regionalT0, out, rows, cols, start_frame, end_frame, 
            mincols, minrows, maxcols, maxrows, x_region_size, y_region_size, timestamps);

	if(mincols==0 && maxcols == cols && minrows==0 && maxrows==rows)
	{
		len = 4;
		cksmPtr = (unsigned char *)GetFileData(fd, offset, len,tmpcksum);
		if ((end_frame >= (totalFrames-1)) && cksmPtr)
		{
			// there is a checksum?
			tmpcksum = cksmPtr[3];
			tmpcksum |= cksmPtr[2] << 8;
			tmpcksum |= cksmPtr[1] << 16;
			tmpcksum |= cksmPtr[0] << 24;
			if(tmpcksum != cksum)
			{
				printf("checksums don't match %x %x %x-%x-%x-%x\n",cksum,tmpcksum,cksmPtr[0],cksmPtr[1],cksmPtr[2],cksmPtr[3]);
				printf("corrupt file\n");
				if(!ignoreErrors)
					exit(2);
			}
		}
		FreeFileData(fd);
	}
	free(reg_offsets);

#ifdef DEBUG
	free(unInterlacedData);
#endif
	// try to get a checksum at the end of the file...
	// if it's there, use it to validate the file.
	// oterwise, ignore it...

	if (WholeFrameOrig)
		free(WholeFrameOrig);

	CloseFile(fd);

	return 1;
}


static int chan_interlace[] =
{ 1, 0, 3, 2, 5, 4, 7, 6 };
static int GetDeInterlaceInfo(int interlaceType, int rows, int cols, int x, int y)
{
	int rc=0;

	switch (interlaceType)
	{
		case 0:
		rc = y * cols + x + 1 - (x % 2) * 2;
		break;
		case 1:
		if (y >= (rows / 2))
			rc = 2 + (1 - (x % 2)) + (x / 2) * 4 + (y - rows / 2) * cols * 2;
		else
			rc = 1 - (x % 2) + (x / 2) * 4 + y * cols * 2;
		break;
		case 2:
		if (y >= (rows / 2))
			rc = 4 + ((x & 3) ^ 1) + (x / 4) * 8 + (y - rows / 2) * (cols) * 2;
		else
			rc = ((x & 3) ^ 1) + (x / 4) * 8 + (rows / 2 - y - 1) * (cols) * 2;
		break;
		case 3:
		if (y >= rows / 2)
			rc = chan_interlace[x & 3] + (x & ~0x3) + (y - rows / 2) * cols * 2;
		else
			rc = chan_interlace[x & 3] + (x & ~0x3) + (rows / 2 - y) * cols * 2
					- cols;
		break;
	}
	return rc;
}


int LoadUnCompressedImage(DeCompFile *fd, short *out, int offset,
		int interlaceType, int rows, int cols, int start_frame, int end_frame,
		int *timestamps, int mincols, int minrows, int maxcols, int maxrows)
{
	unsigned short *CompPtr;
	unsigned short *rawFrame;
	unsigned short *tmpBufPtr;
	unsigned short val, *imagePtr = (unsigned short *) out;
	int frame;
	int frameStride = rows * cols;
	int x, y, j;
	int len;
//	DeCompFile *fd = convert_to_fh(in_fd);
	unsigned int cksum=0;
//	int ArrayOffset;

	if(mincols == 0 && minrows == 0 && maxcols == cols && maxrows == rows)
		tmpBufPtr = NULL; // full frame case..
	else
		tmpBufPtr = (unsigned short *)malloc(rows*cols*2);

	offset += start_frame * 2 * rows * cols + 4*start_frame; // point to the right start frame
	for (frame = start_frame; frame <= end_frame; frame++)
	{
		// map the whole frame
		len = 2 * frameStride + 4;
		CompPtr = (unsigned short *)GetFileData(fd, offset, len,cksum);

		if(CompPtr == NULL)
			return 0;

		if (timestamps)
		{
			timestamps[frame - start_frame]
					= BYTE_SWAP_4(*(unsigned int *)CompPtr);
		}
		rawFrame = CompPtr + 2;

		if(tmpBufPtr)
			imagePtr = tmpBufPtr;
		else
			imagePtr = (unsigned short *)out + ((frame-start_frame)*cols*rows);
		// do old stuff here

		j = 0;
		if (interlaceType == 0)
		{
			for (y = 0; y < rows; y++)
			{
				for (x = 0; x < cols; x++)
				{
					// channels 1 & 2 are swapped
					val = rawFrame[y*cols + x + 1 - (x%2)*2];
					val = BYTE_SWAP_2(val);
					*imagePtr++ = (short) (val & 0x3fff);
				}
			}
		}
		else if (interlaceType == 1)
		{
			for (y = 0; y < rows / 2; y++)
			{
				for (x = 0; x < cols; x += 2)
				{
					 // start at the first half of our image, second channel
					j = y * cols + x + 1;
					val = *rawFrame++;
					imagePtr[j] = (short) (BYTE_SWAP_2(val)	& 0x3fff);
					j--;
					val = *rawFrame++;
					imagePtr[j] = (short) (BYTE_SWAP_2(val) & 0x3fff);

					// jump down to the second half of our image, second channel
					j = (y + rows / 2) * cols + x + 1;
					val = *rawFrame++;
					imagePtr[j] = (short) (BYTE_SWAP_2(val)	& 0x3fff);
					j--;
					val = *rawFrame++;
					imagePtr[j] = (short) (BYTE_SWAP_2(val)	& 0x3fff);
				}
			}
		}
		else if (interlaceType == 2)
		{
			for (y = 0; y < rows / 2; y++)
			{
				for (x = 0; x < cols; x += 4)
				{
					// bottom 1/2 is inverted vertically.  Four samples in a row are from the bottom
					j = (rows / 2 - y - 1) * cols + x; // start at the first half of our image
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);

					// top 1/2 is not inverted vertically.  Four samples in a row are from the top
					j = (y + rows / 2) * cols + x; // jump down to the second half of our image
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);
					val = *rawFrame++;
					imagePtr[j++] = (short) (BYTE_SWAP_2(val) & 0x3fff);
				}
			}
		}
		else if (interlaceType == 3)
		{
			for (y = 0; y < rows; y++)
			{
				for (x = 0; x < cols; x++)
				{
					val = rawFrame[GetDeInterlaceInfo(interlaceType,rows,cols,x,y)];
					val = BYTE_SWAP_2(val);
					*imagePtr++ = (short) (val & 0x3fff);
				}
			}
		}

		if(tmpBufPtr)
		{
			imagePtr = (unsigned short *)out + ((frame-start_frame)*(maxcols-mincols)*(maxrows-minrows));

			// copy it to the local region buffer
			for (y = minrows; y < maxrows; y++)
			{
				for (x = mincols; x < maxcols; x++)
				{
					*imagePtr++ = tmpBufPtr[y*cols+x];
				}
			}
		}

		FreeFileData(fd);
		offset += len;
	}

	if(tmpBufPtr)
		free(tmpBufPtr);

	CloseFile(fd);

	return 1;
}


int LoadImage(char *fname, short *out, int offset, int interlaceType,
		int rows, int cols, int totalFrames, int start_frame, int end_frame, 
		int *timestamps, int mincols, int minrows, int maxcols, int maxrows,
		int x_region_size, int y_region_size, int ignoreErrors)
{
	int rc;

	DeCompFile *fd = OpenFile(fname);
	if(fd == NULL)
	{
		printf("failed to open %s\n",fname);
		if(ignoreErrors&IGNORE_ALWAYS_RETURN)
			return 0;
		else
			exit(-1);
	}

	if (maxcols == 0)
		maxcols = cols;
	if (maxrows == 0)
		maxrows = rows;

	if (interlaceType == 4)
	{
		rc = LoadCompressedImage(fd, out, rows, cols, totalFrames, start_frame, end_frame,
				timestamps, mincols, minrows, maxcols, maxrows,ignoreErrors);
	}
	else if ((interlaceType == 5) || (interlaceType == 6))
	{
		rc = LoadCompressedRegionImage(fd, out, rows, cols, totalFrames, start_frame, end_frame,
				timestamps, mincols, minrows, maxcols, maxrows,
				x_region_size,y_region_size,offset,ignoreErrors);
	}
	else
	{
		rc = LoadUnCompressedImage(fd, out, offset, interlaceType, rows, cols,
				start_frame, end_frame, timestamps, mincols, minrows, maxcols,
				maxrows);
	}
	return rc;
}
#if 0
static int getVFCFrames(int *timestamps, int frames)
{
	int rc = 0;
	int i=0;
	int baseTime;
	int prevTime;
	int deltaTime;
	int safecnt;

	// first frame is always the base unit of time
	baseTime = timestamps[0];
	if(baseTime == 0)
	{
		// buggy old code put the start time of the frames
		i++;
		baseTime = timestamps[i];
		rc++;
	}
	prevTime=0;
	for(;i<frames;i++)
	{
//		printf("found timestamp %d) %d",i,timestamps[i]);
		deltaTime = timestamps[i] - prevTime;
		prevTime = timestamps[i];
		safecnt=0;
		while(deltaTime > 0 && safecnt++ < 100)
		{
			deltaTime -= baseTime;
			rc++; // another frame
		}
	}

	return rc;
}
#endif

#ifndef WIN32
int deInterlace_c(
#else
extern "C" int __declspec(dllexport) deInterlace_c(
#endif
        char *fname, short **_out, int **_timestamps,
		int *_rows, int *_cols, int *_frames, int *_uncompFrames,
		int start_frame, int end_frame,
		int mincols, int minrows, int maxcols, int maxrows, int ignoreErrors)
{

	DeCompFile *fd;
	struct _file_hdr hdr;
	unsigned short frames, interlaceType, uncompFrames=0;
	int rows, cols;
	int x_region_size=0,y_region_size=0;
	int offset = 0;
	unsigned int cksum=0;
	int *timestamps=NULL;

	fd = OpenFile(fname);

	char *file_memory = GetFileData(fd, 0, fd->PageSize,cksum);

	if(file_memory == NULL)
	{
		printf("failed to get file memory\n");
		return 0;
	}

	//	char *startPtr = file_memory;

	memcpy(&hdr, file_memory, sizeof(hdr));
	file_memory += sizeof(hdr);

	ByteSwap4(hdr.signature);
	ByteSwap4(hdr.struct_version);
	ByteSwap4(hdr.header_size);
	ByteSwap4(hdr.data_size);

	switch (hdr.struct_version)
	{
	case 3:
	{
		struct _expmt_hdr_v3 expHdr;
		memcpy(&expHdr, file_memory, sizeof(expHdr));
		//BYTE_SWAP_4(expHdr.first_frame_time);
		rows = BYTE_SWAP_2(expHdr.rows);
		cols = BYTE_SWAP_2(expHdr.cols);
		frames = BYTE_SWAP_2(expHdr.frames_in_file);
		uncompFrames = BYTE_SWAP_2(expHdr.uncomp_frames_in_file);
//		channels = BYTE_SWAP_2(expHdr.channels);
		interlaceType = BYTE_SWAP_2(expHdr.interlaceType);
		//unsigned int sample_rate = BYTE_SWAP_4(expHdr.sample_rate);
		//unsigned short frame_interval = BYTE_SWAP_2(expHdr.frame_interval);
		// int a = 1;

		offset = sizeof(expHdr) + sizeof(hdr);
	}

		break;

	case 4:
	{
		struct _expmt_hdr_v4 expHdr;
		memcpy(&expHdr, file_memory, sizeof(expHdr));
		//BYTE_SWAP_4(expHdr.first_frame_time);
		rows = BYTE_SWAP_2(expHdr.rows);
		cols = BYTE_SWAP_2(expHdr.cols);
		x_region_size = BYTE_SWAP_2(expHdr.x_region_size);
		y_region_size = BYTE_SWAP_2(expHdr.y_region_size);
		frames = BYTE_SWAP_2(expHdr.frames_in_file);
		uncompFrames = BYTE_SWAP_2(expHdr.uncomp_frames_in_file);
//		channels = BYTE_SWAP_2(expHdr.channels);
		interlaceType = BYTE_SWAP_2(expHdr.interlaceType);
		//unsigned int sample_rate = BYTE_SWAP_4(expHdr.sample_rate);
		//unsigned short frame_interval = BYTE_SWAP_2(expHdr.frame_interval);
		// int a = 1;

		offset = sizeof(expHdr) + sizeof(hdr);
		//printf("offset=%d %ld %ld\n",offset,sizeof(expHdr) , sizeof(hdr));
	}

		break;

	default:
		FreeFileData(fd);
		CloseFile(fd);
		return 0;
		break;
	}

	FreeFileData(fd);
	CloseFile(fd);

	if(_rows)
		*_rows = rows;
	if(_cols)
		*_cols = cols;
	if(_frames)
		*_frames = frames;
	if(_uncompFrames)
	{
		if(uncompFrames && (uncompFrames > frames) && (uncompFrames < (4*frames)))
			*_uncompFrames = uncompFrames;
		else
			*_uncompFrames = frames;
	}

	if(maxcols == 0)
		maxcols = cols;
	if(maxrows == 0)
		maxrows = rows;

#define CHECK_INPUT(a,b,size,type) \
	if(a && (*a == NULL)) \
		*a = (type *)malloc(size); \
	if (a && *a) \
		b = *a; \
	else \
		b = (type *)malloc(size); \
	if (!b) \
		fprintf(stderr,"Failed to allocate memory for %d bytes in call to deInterlace(), segfault likely...\n",(unsigned int)size);

	if (start_frame >= frames)
		start_frame = frames - 1;
	if ((end_frame >= frames) || (end_frame == 0))
		end_frame = frames-1;

	if (_out)
	{
		short *out;
		size_t nBytes = (maxrows-minrows)*(maxcols-mincols);
		nBytes *= 2*(end_frame-start_frame+1);

		CHECK_INPUT(_out,out,nBytes,short);
		CHECK_INPUT(_timestamps,timestamps,sizeof(int)*frames,int);

		LoadImage(fname,out,offset,(int)interlaceType,rows,cols,frames,
			start_frame,end_frame,timestamps,mincols,minrows,maxcols,maxrows,x_region_size,y_region_size,ignoreErrors);
	}


	return 1;
}

#ifndef WIN32
int deInterlaceData(
#else
extern "C" int __declspec(dllexport) deInterlaceData(
#endif
        char *fname, short *_out, int *_timestamps, int start_frame, int end_frame,
		int mincols, int minrows, int maxcols, int maxrows, int ignoreErrors)
{
#if 0
	char fname2[256];

	sprintf(fname2,"%s_params",fname);
	FILE *fp = fopen(fname2,"w");
	if(fp)
	{
		fprintf(fp,"deInterlaceData called with\n");
		fprintf(fp,"  fname=%s\n",fname);
		fprintf(fp,"  _out=%p %p\n",_out,(_out?*_out:NULL));
		fprintf(fp,"  start_frame %d\n",start_frame);
		fprintf(fp,"  end_frame %d\n",end_frame);
		fprintf(fp,"  mincols %d\n",mincols);
		fprintf(fp,"  minrows %d\n",minrows);
		fprintf(fp,"  maxcols %d\n",maxcols);
		fprintf(fp,"  maxrows %d\n",maxrows);
		fclose(fp);
	}
#endif
	return deInterlace_c(fname,&_out,&_timestamps,
						  NULL,NULL,NULL,NULL,start_frame,end_frame,
						  mincols,minrows,maxcols,maxrows,ignoreErrors);
}



#ifndef WIN32
int deInterlaceHdr(
#else
extern "C" int __declspec(dllexport) deInterlaceHdr(
#endif
        char *fname, int *_rows, int *_cols, int *_frames, int *_unCompFrames)
{
#if 0
	char fname2[256];

	sprintf(fname2,"%s_params",fname);
	FILE *fp = fopen(fname2,"w");
	if(fp)
	{
		fprintf(fp,"deInterlaceHdr called with\n");
		fprintf(fp,"  fname=%s\n",fname);
		fprintf(fp,"  rows %d\n",*_rows);
		fprintf(fp,"  cols %d\n",*_cols);
		fprintf(fp,"  frames %d\n",*_frames);
		fprintf(fp,"  uncompFrames %d\n",*_unCompFrames);
		fclose(fp);
	}
#endif
	return deInterlace_c(fname,NULL,NULL,_rows,_cols,_frames,_unCompFrames,0,0,0,0,0,0,false);
}



#ifdef WIN32
int main(int argc, char *argv[])
{
	int start_frame = 0;
	int end_frame = 0;
	short *output=NULL;
	int *timestamps=NULL;
	int minx, miny, maxx, maxy;
	char *fname = argv[1];
	int frameStride;
	int rows,cols;
//	DWORD bytesWritten;
//	char fnameOut[100];
//	int i;
//	char *ptr;
	int frames=0, uncompFrames=0;

	sscanf(argv[2], "%d", &start_frame);
	sscanf(argv[3], "%d", &end_frame);
	sscanf(argv[4], "%d", &minx);
	sscanf(argv[5], "%d", &miny);
	sscanf(argv[6], "%d", &maxx);
	sscanf(argv[7], "%d", &maxy);

	frameStride = (maxx - minx) * (maxy - miny);

	if(frameStride)
	{
		output = (short *) malloc(2 * frameStride * (end_frame-start_frame+1));
		memset(output, 0,2 * frameStride * (end_frame-start_frame+1));
	}
//	for(i=0;i<20;i++)
	{
		deInterlace_c(fname, &output, &timestamps,  &rows, &cols,
			&frames, &uncompFrames, start_frame, end_frame, 
			minx, miny, maxx, maxy, 0);
//		printf("%d) loaded %p\n",i,ptr);
	}

#if 0
	for (i = start_frame; i < end_frame; i++)
	{
		sprintf(fnameOut, "%s_%d", fname, i);
		MY_FILE_HANDLE hDbgFile = CreateFile((const char *) fnameOut,
				GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
				NULL);

		WriteFile(hDbgFile, output + frameStride * i, frameStride * 2,
				&bytesWritten, NULL);
		CloseHandle(hDbgFile);
		printf("%x %x %x %x\n", output[i * 4], output[i * 4 + 1], output[i * 4
				+ 2], output[i * 4 + 3]);
	}
#endif
	return 0;
}
#endif


