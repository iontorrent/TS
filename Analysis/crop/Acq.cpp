/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include "crop/Acq.h"
#include "ByteSwapUtils.h"
#include "AdvCompr.h"
#include "datahdr.h"
#include "Utils.h"
#include "DCT0Finder.h"
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;

struct HDR {
	_file_hdr	fileHdr;
	_expmt_hdr_v3	expHdr;
};

#define KEY_0     0x7F
#define KEY_16_1  0xBB

#define PLACEKEY  0xdeadbeef
#define ACQENDTIME 5.0
#define REGACQENDTIME 5.0

int Acq::counter = 0;

Acq::Acq()
{
	counter = 0;
	data = NULL;
	timestamps = NULL;
	w = 0;
	h = 0;
	numFrames = 0;
	frameStride = 0;
	pinnedLow = 0;
	pinnedHigh = 0;
	image=NULL;
	x_region_size = 64;
	y_region_size = 64;
        num_regions_x = 0;
        num_regions_x = 0;
}

Acq::~Acq()
{
	/* This is the raw->image data created and free'd by the Image loader,
	   when used by Crop.cpp.  Don't know if anything else uses this.
	if (data)
		free(data);
	*/
	if (timestamps)
		free (timestamps);
}

void Acq::SetSize(int _w, int _h, int _numFrames, int _numUncompFrames)
{
	w = _w;
	h = _h;
	numFrames = _numFrames;
	uncompFrames = _numUncompFrames;
	frameStride = w*h;

	data = (unsigned short *)malloc(sizeof(unsigned short) * w*h*numFrames);
	timestamps = (int *)malloc(sizeof(int) * numFrames);
	Clear();
}

void Acq::SetData(Image *_image)
{
	const RawImage *raw=NULL;;

	image = _image;

	if(image)
	{
		raw = image->GetImage();
		w = raw->cols;
		h = raw->rows;
		numFrames = raw->frames;
		uncompFrames = raw->uncompFrames;
		data = (unsigned short *)raw->image;
	}
	else
	{
		w = h = numFrames = uncompFrames = 0;
		data = NULL;
	}

	frameStride = w*h;
	if (timestamps)
	{
		free(timestamps);
		timestamps=NULL;
	}

	if(image)
	{
		timestamps = (int *)malloc(sizeof(int) * numFrames);

		for (int i = 0; i < numFrames; i++) {
			timestamps[i] = ((int *)raw->timestamps)[i];
		}
	}
}

void Acq::SetWellTrace(double *trace, int x, int y)
{
	int i;
	int offset = x+w*y;
	unsigned short traceVal;
	for(i=0;i<numFrames;i++) {
		if (trace[i] < 0.0) {
			traceVal = 0;
			pinnedLow++;
		} else if (trace[i] > 0x3fff) {
			traceVal = 0x3fff;
			pinnedHigh++;
		} else {
			traceVal = (unsigned short)(trace[i]+0.5);
		}
		data[offset] = traceVal;
		offset += frameStride;
	}
}

void Acq::Clear()
{
	memset(data, 0, sizeof(unsigned short) * w*h*numFrames);
	pinnedHigh = 0;
	pinnedLow = 0;
}

void Acq::Write()
{
	char acqName[MAX_PATH_LENGTH];
	sprintf(acqName, "acq_%04d.dat", counter);
	counter++;
	Write(acqName, 0, 0, w, h);
}

bool Acq::Write(const char *acqName, int ox, int oy, int ow, int oh)
{
	// open up the acq file
	FILE *fp;
	fp = fopen(acqName, "wb");
	if (!fp) {
		printf("Warning!  Could not open file: %s for writing?\n", acqName);
		return false;
	}

	HDR	hdr;

	if(ow == 0)
	{
		ox = 0;
		ow = w;
	}
	if(oh == 0)
	{
		oy = 0;
		oh = h;
	}
	// set up the file header
	hdr.fileHdr.signature = 0xdeadbeef;
	hdr.fileHdr.struct_version = 0x3;
	hdr.fileHdr.header_size = sizeof(_expmt_hdr_v3);
	unsigned long totalSize = numFrames * (ow*oh*2 + sizeof(uint32_t)); // data + timestamp
	// hdr.fileHdr.data_size = numFrames*w*h*2; // old bad format
	hdr.fileHdr.data_size = totalSize; // new good format

	ByteSwap4(hdr.fileHdr.signature);
	ByteSwap4(hdr.fileHdr.struct_version);
	ByteSwap4(hdr.fileHdr.header_size);
	ByteSwap4(hdr.fileHdr.data_size);

	// setup the data header
	memset(&hdr.expHdr, 0, sizeof(hdr.expHdr));
	hdr.expHdr.first_frame_time = 0;
	hdr.expHdr.rows = oh;
	hdr.expHdr.cols = ow;
	hdr.expHdr.frames_in_file = numFrames;
	hdr.expHdr.uncomp_frames_in_file = uncompFrames;
	hdr.expHdr.interlaceType = 0;
	// hdr.expHdr.sample_rate = 
	// hdr.expHdr.full_scale_voltage[0] = 
	// hdr.expHdr.full_scale_voltage[1] = 
	// hdr.expHdr.full_scale_voltage[2] = 
	// hdr.expHdr.full_scale_voltage[3] = 
	hdr.expHdr.channel_offset[0] =
	hdr.expHdr.channel_offset[1] =
	hdr.expHdr.channel_offset[2] =
	hdr.expHdr.channel_offset[3] = 
	hdr.expHdr.electrode = 
	hdr.expHdr.frame_interval = 0; 

	ByteSwap4(hdr.expHdr.first_frame_time);
	ByteSwap2(hdr.expHdr.rows);
	ByteSwap2(hdr.expHdr.cols);
	ByteSwap2(hdr.expHdr.frames_in_file);
	ByteSwap2(hdr.expHdr.uncomp_frames_in_file);
	ByteSwap2(hdr.expHdr.channels);
	ByteSwap2(hdr.expHdr.interlaceType);

	// write file & data headers
	fwrite(&hdr, sizeof(HDR), 1, fp);

	// write each frame block (timestamp & frame data)
	int frame;
	int offset = ox+oy*w;
	int ix, iy;
	unsigned short *ptr;
	unsigned short val[4];
	for(frame=0;frame<numFrames;frame++) {
		uint32_t timestampOut = BYTE_SWAP_4(timestamps[frame]);
		fwrite(&timestampOut, sizeof(timestampOut), 1, fp);
		for(iy=0;iy<oh;iy++)
		{
			ptr = &data[offset + iy*w];
			for(ix=0;ix<ow;)
			{
				if ((ix%4) == 0)
				{
					val[0] = BYTE_SWAP_2(ptr[1]);
					val[1] = BYTE_SWAP_2(ptr[0]);
					val[2] = BYTE_SWAP_2(ptr[3]);
					val[3] = BYTE_SWAP_2(ptr[2]);
				}

				fwrite(&val[ix%4], sizeof(uint16_t), 1, fp);
				ix++;
				ptr++;
			}
		}
		offset += frameStride;
	}

	fclose(fp);
	return true;
}

uint16_t Acq::get_val(uint32_t x, uint32_t y, uint32_t rframe, uint32_t numFrames)
{
//	return image->GetInterpolatedValueAvg(rframe,x,y,numFrames);
#if 1
	uint64_t accum = 0;
	uint32_t i;
//	static uint32_t cntr=0;

	for(i=0;i<numFrames;i++)
	{
		accum += image->GetInterpolatedValue(rframe+i,x,y);
	}
	accum /= numFrames;

//	if(cntr++ < 20)
//		printf("ptr=%x rc=%x",*ptr)
	return accum;
#endif
}

uint8_t *bwrt(uint8_t *src, uint32_t len, uint8_t *dst)
{
	for(uint32_t i=0;i<len;i++)
	{
		*dst++ = *src++;
	}
	return dst;
}

bool Acq::WriteVFC(const char *acqName, int ox, int oy, int ow, int oh, bool verbose)
{
    // open up the acq file
	const RawImage *raw = image->GetImage();
	FILE *fp;
	fp = fopen(acqName, "wb");
	if (!fp) {
		printf("Warning!  Could not open file: %s for writing?\n", acqName);
		return false;
	}


    _file_hdr	fileHdr;
	_expmt_hdr_v4	expHdr;

//	uint32_t vfr_array[100];
	uint32_t vfr_array_cnt = 0;
	uint32_t vfr_total_cnt=0;
//	uint32_t k;
	uint32_t offset=0;
//	uint8_t *buffer = (uint8_t *)malloc(4*w*h);
//	uint8_t *bptr;

#if 0
#define ADD_FRAME(cnt,frms) \
{ \
	for(k=0;k<cnt;k++) \
	{ \
		if ((vfr_total_cnt + frms) <= (uint32_t)uncompFrames) \
		{ \
			vfr_array[vfr_array_cnt++] = frms; \
			vfr_total_cnt += frms; \
		} \
	} \
}

	ADD_FRAME(1  ,1);   // first frame is always at the base acquisition time
	ADD_FRAME(1  ,8);   // this accounts for the first second T=1
	ADD_FRAME(1  ,4);   // this accounts for the first second T=1
	ADD_FRAME(52 ,1);   // the next 3 seconds  T=5
	ADD_FRAME(12 ,4);   // the next 3 seconds T=8
	ADD_FRAME(8  ,8);   // the next 4 seconds T=8
#endif

	unsigned int sample_rate = image->GetImage()->timestamps[0];

	vfr_array_cnt = raw->frames;
	vfr_total_cnt = raw->uncompFrames;

	memset(&fileHdr,0,sizeof(fileHdr));
	memset(&expHdr,0,sizeof(expHdr));


	// set up the file header
    fileHdr.signature = 0xdeadbeef;
	fileHdr.struct_version = 0x4;
	fileHdr.header_size = sizeof(_expmt_hdr_v3);
	unsigned long totalSize = vfr_array_cnt * (ow*oh*2 + sizeof(uint32_t)); // data + timestamp
	// hdr.fileHdr.data_size = numFrames*w*h*2; // old bad format
	fileHdr.data_size = totalSize; // new good format

	ByteSwap4(fileHdr.signature);
	ByteSwap4(fileHdr.struct_version);
	ByteSwap4(fileHdr.header_size);
	ByteSwap4(fileHdr.data_size);

	// setup the data header
    expHdr.first_frame_time = 0;
	expHdr.rows = oh;
	expHdr.cols = ow;
	expHdr.frames_in_file = vfr_array_cnt;
	expHdr.uncomp_frames_in_file = vfr_total_cnt;
	expHdr.interlaceType = 5;
	expHdr.x_region_size = x_region_size;
	expHdr.y_region_size = y_region_size;
	expHdr.sample_rate = sample_rate;
//	expHdr.channel_offset[0] = raw->

	ByteSwap4(expHdr.first_frame_time);
	ByteSwap2(expHdr.rows);
	ByteSwap2(expHdr.cols);
	ByteSwap2(expHdr.frames_in_file);
	ByteSwap2(expHdr.uncomp_frames_in_file);
	ByteSwap2(expHdr.interlaceType);
	ByteSwap2(expHdr.x_region_size);
	ByteSwap2(expHdr.y_region_size);
	ByteSwap4(expHdr.sample_rate);

	// write file & data headers
	fwrite(&fileHdr, sizeof(fileHdr), 1, fp);
	offset += sizeof(fileHdr);
	fwrite(&expHdr, sizeof(expHdr), 1, fp);
	offset += sizeof(expHdr);

    if (verbose)
        printf("offset=%d %zu %zu\n",offset,sizeof(expHdr),sizeof(fileHdr));
	// write each frame block (timestamp & frame data)
	uint32_t frame;
//	int offset;
	int ix, iy;
//	uint16_t *ptr;
//	unsigned short val;
	unsigned int rframe=0;
	unsigned int frameCnt=0;
	int16_t *frame_data,*sptr;

	frame_data = (int16_t *)malloc(2*ow*oh);
	int16_t *prev_data = NULL;
	int16_t *results_data = (int16_t *)malloc(2*ow*oh);


	for(frame=0;frame<vfr_array_cnt;frame++) {
//		offset = ox+oy*w + rframe*frameStride;
//		bptr = buffer;
//		frameCnt = vfr_array[frame];
//		if((rframe+frameCnt) > (unsigned int)uncompFrames)
//		{
//			rframe = numFrames-1;
//			frameCnt=1;
//		}
//		printf("frame %d:  offset=%d\n",frame,offset);
		uint32_t timestampOut = BYTE_SWAP_4(raw->timestamps[frame]); // write out the last time stamp...
		fwrite(&timestampOut, sizeof(timestampOut), 1, fp);
//		bptr = bwrt(&timestampOut, sizeof(timestampOut),bptr);
		offset += sizeof(timestampOut);

//		printf("ts=%d ",sample_rate*(rframe+frameCnt+1));
		int16_t *ptr = frame_data;
		uint16_t tmp[4];
		uint64_t results_len=0;
		uint32_t comp;

		results_len=0;
		comp=0;


		// save the data into frame_data
		for(iy=0;iy<oh;iy++)
		{
			sptr = &raw->image[frame*raw->frameStride+(iy+oy)*raw->cols+ox];
			for(ix=0;ix<ow;ix++)
			{
				*ptr++ = *sptr++ & 0x3fff;
//				image->GetInterpolatedValueAvg4(ptr,rframe,ox+ix,oy+iy,frameCnt);
//				*ptr++ = get_val(ox,iy+oy,rframe,frameCnt);
			}
		}

		if(prev_data)
		{
			if(PrevFrameSubtract(ow,oh,frame_data,prev_data,results_data,&results_len,&comp) == 0)
			{
//				printf("pfc worked %ld!!\n",results_len);
				comp = htonl(comp);
			}
			else
			{
//				printf("pfc didn't work frame=%d %x\n",frame,comp);
			}
		}

		fwrite(&comp, sizeof(comp), 1, fp);
		offset += sizeof(comp);


		if(!comp)
		{
			ptr = frame_data;
			for(iy=0;iy<oh;iy++)
			{
				for(ix=0;ix<ow;ix++)
				{

					tmp[0] = BYTE_SWAP_2(*ptr);
					fwrite(&tmp[0],2,1,fp);
					ptr++;
				}
			}
			//printf("frame %d: %d bytes\n",frame,oh*ow*2);
			offset += oh*ow*2;
		}
		else
		{
			// write out the compressed data
			fwrite(results_data,1,results_len,fp);
			//printf("frame %d: %ld bytes\n",frame,results_len);

			offset += results_len;
		}


		if(prev_data)
		{
			int16_t *tmp_data = frame_data;
			frame_data = prev_data;
			prev_data = tmp_data;
		}
		else
		{
			prev_data = frame_data;
			frame_data = (int16_t *)malloc(2*ow*oh);
		}

        if (verbose)
        {
		if(comp)
			printf(".");
		else
			printf("-");
		fflush(stdout);
        }

//					printf("val=%x %x %x %x   ptr=%x %x %x %x\n",val[0],val[1],val[2],val[3],ptr[0],ptr[1],ptr[2],ptr[3]);

		rframe += frameCnt;
	}

    if (verbose)
        printf("\n  Size=%d\n",offset);

	free(frame_data);
	free(results_data);
	if(prev_data)
		free(prev_data);
	fclose(fp);
	return true;
}
bool Acq::WriteThumbnailVFC(const char *acqName, int cropx, int cropy, int kernx, int kerny, int region_len_x, int region_len_y, int marginx, int marginy,int ow, int oh, bool verbose)
{
    // open up the acq file
    const RawImage *raw = image->GetImage();
    FILE *fp;
    fp = fopen(acqName, "wb");
    if (!fp) {
        printf("Warning!  Could not open file: %s for writing?\n", acqName);
        return false;
    }

    _file_hdr	fileHdr;
    _expmt_hdr_v4	expHdr;

//	uint32_t vfr_array[100];
    uint32_t vfr_array_cnt = 0;
    uint32_t vfr_total_cnt=0;
//	uint32_t k;
    uint32_t offset=0;
//	uint8_t *buffer = (uint8_t *)malloc(4*w*h);
//	uint8_t *bptr;

#if 0
#define ADD_FRAME(cnt,frms) \
{ \
    for(k=0;k<cnt;k++) \
    { \
        if ((vfr_total_cnt + frms) <= (uint32_t)uncompFrames) \
        { \
            vfr_array[vfr_array_cnt++] = frms; \
            vfr_total_cnt += frms; \
        } \
    } \
}

    ADD_FRAME(1  ,1);   // first frame is always at the base acquisition time
    ADD_FRAME(1  ,8);   // this accounts for the first second T=1
    ADD_FRAME(1  ,4);   // this accounts for the first second T=1
    ADD_FRAME(52 ,1);   // the next 3 seconds  T=5
    ADD_FRAME(12 ,4);   // the next 3 seconds T=8
    ADD_FRAME(8  ,8);   // the next 4 seconds T=8
#endif

    unsigned int sample_rate = image->GetImage()->timestamps[0];

    vfr_array_cnt = raw->frames;
    vfr_total_cnt = raw->uncompFrames;

    memset(&fileHdr,0,sizeof(fileHdr));
    memset(&expHdr,0,sizeof(expHdr));


    // set up the file header
    fileHdr.signature = 0xdeadbeef;
    fileHdr.struct_version = 0x4;
    fileHdr.header_size = sizeof(_expmt_hdr_v3);
    unsigned long totalSize = vfr_array_cnt * (ow*oh*2 + sizeof(uint32_t)); // data + timestamp
    // hdr.fileHdr.data_size = numFrames*w*h*2; // old bad format
    fileHdr.data_size = totalSize; // new good format

    ByteSwap4(fileHdr.signature);
    ByteSwap4(fileHdr.struct_version);
    ByteSwap4(fileHdr.header_size);
    ByteSwap4(fileHdr.data_size);

    // setup the data header
    expHdr.first_frame_time = 0;
    expHdr.rows = oh;
    expHdr.cols = ow;
    expHdr.frames_in_file = vfr_array_cnt;
    expHdr.uncomp_frames_in_file = vfr_total_cnt;
    expHdr.interlaceType = 5;
    expHdr.x_region_size = x_region_size;
    expHdr.y_region_size = y_region_size;
    expHdr.sample_rate = sample_rate;
//	expHdr.channel_offset[0] = raw->

    ByteSwap4(expHdr.first_frame_time);
    ByteSwap2(expHdr.rows);
    ByteSwap2(expHdr.cols);
    ByteSwap2(expHdr.frames_in_file);
    ByteSwap2(expHdr.uncomp_frames_in_file);
    ByteSwap2(expHdr.interlaceType);
    ByteSwap2(expHdr.x_region_size);
    ByteSwap2(expHdr.y_region_size);
    ByteSwap4(expHdr.sample_rate);

    // write file & data headers
    fwrite(&fileHdr, sizeof(fileHdr), 1, fp);
    offset += sizeof(fileHdr);
    fwrite(&expHdr, sizeof(expHdr), 1, fp);
    offset += sizeof(expHdr);

    if (verbose)
        printf("offset=%d %zu %zu\n",offset,sizeof(expHdr),sizeof(fileHdr));
    // write each frame block (timestamp & frame data)
    uint32_t frame;
//	int offset;
    int ix, iy;
//	uint16_t *ptr;
//	unsigned short val;
    unsigned int rframe=0;
    unsigned int frameCnt=0;
    int16_t *frame_data,*sptr;

    frame_data = (int16_t *)malloc(2*ow*oh);
    int16_t *prev_data = NULL;
    int16_t *results_data = (int16_t *)malloc(2*ow*oh);

    for(frame=0;frame<vfr_array_cnt;frame++) {
//		offset = ox+oy*w + rframe*frameStride;
//		bptr = buffer;
//		frameCnt = vfr_array[frame];
//		if((rframe+frameCnt) > (unsigned int)uncompFrames)
//		{
//			rframe = numFrames-1;
//			frameCnt=1;
//		}
//		printf("frame %d:  offset=%d\n",frame,offset);
        uint32_t timestampOut = BYTE_SWAP_4(raw->timestamps[frame]); // write out the last time stamp...
        fwrite(&timestampOut, sizeof(timestampOut), 1, fp);
//		bptr = bwrt(&timestampOut, sizeof(timestampOut),bptr);
        offset += sizeof(timestampOut);
        if (verbose)
        {
            printf("\nframe: %d\t timestamp offset: %d %zu",frame,offset,sizeof(timestampOut));
        }

//		printf("ts=%d ",sample_rate*(rframe+frameCnt+1));
        int16_t *ptr = frame_data;
        uint16_t tmp[4];
        uint64_t results_len=0;
        uint32_t comp;

        results_len=0;
        comp=0;


        // save the data for entire thumbnail into frame_data - combining all regions
        for (int y=0;y<cropy;y++)
        {
            int region_origin_y =  y * region_len_y + marginy;
            for (iy=0; iy<kerny;iy++)
            {
                for (int x=0;x<cropx;x++)
                {
                    int region_origin_x= x * region_len_x + marginx;
                    sptr = &raw->image[frame*raw->frameStride+(iy+region_origin_y)*raw->cols+region_origin_x];
                    for(ix=0;ix<kernx;ix++)
                    {
                        *ptr++ = *sptr++ & 0x3fff;
                    }
                }
            }
        }


        if(prev_data) //previous timeframe
        {
            if(PrevFrameSubtract(ow,oh,frame_data,prev_data,results_data,&results_len,&comp) == 0)
            {
                if (verbose)
                    printf("\npfc worked %" PRIu64 "!!",results_len);
                comp = htonl(comp);
            }
            else
            {
                if (verbose)
                    printf("\npfc didn't work frame=%d %x",frame,comp);
            }
        }

        fwrite(&comp, sizeof(comp), 1, fp);
        offset += sizeof(comp);
        if (verbose)
        {
            printf("\nframe: %d\t comp offset: %d %zu",frame,offset,sizeof(comp));
        }

        if(!comp)
        {
            ptr = frame_data;
            for (int y=0;y<cropy;y++)
            {
                for (int x=0;x<cropx;x++)
                {
                    for(iy=0;iy<kerny;iy++)
                    {
                        for(ix=0;ix<kernx;ix++)
                        {
                            tmp[0] = BYTE_SWAP_2(*ptr);
                            fwrite(&tmp[0],2,1,fp);
                            offset+=2;
                            ptr++;
                        }
                    }
                }
            }
//			offset += oh*ow*2;
            if (verbose)
            {
                printf("\nframe: %d\t !comp tmp offset: %d %d",frame,offset,oh*ow*2);
            }
        }
        else
        {
            // write out the compressed data
            fwrite(results_data,1,results_len,fp);
            offset += results_len;
            if (verbose)
            {
                printf("\nframe: %d\t results compressed data offset: %d ",frame,offset);
            }
        }


        if(prev_data)
        {
            int16_t *tmp_data = frame_data;
            frame_data = prev_data;
            prev_data = tmp_data;
        }
        else
        {
            prev_data = frame_data;
            frame_data = (int16_t *)malloc(2*ow*oh);
        }

        if (verbose)
        {
        if(comp)
            printf(".");
        else
            printf("-");
        fflush(stdout);
        }

//					printf("\nval=%x %x %x %x   ptr=%x %x %x %x\n",val[0],val[1],val[2],val[3],ptr[0],ptr[1],ptr[2],ptr[3]);

        rframe += frameCnt;
    }

    if (verbose)
        printf("\n  Size=%d\n",offset);

    free(frame_data);
    free(results_data);
    if(prev_data)
        free(prev_data);
    fclose(fp);
    return true;
}


#if 0
int Acq::PrevFrameSubtract(int elems, int16_t *framePtr, int16_t *prevFramePtr, int16_t *results, uint64_t *out_len, uint32_t ow, uint32_t oh)
{
	uint32_t x,y;
	uint16_t *src1=(uint16_t *)framePtr;
	uint16_t *src2=(uint16_t *)prevFramePtr;
	int state = 0;
	int newState = 0;
	uint8_t *resp = (uint8_t *)((uint32_t *)results + 3);
	uint32_t total=0;
	uint8_t *limit = (uint8_t *)(((int8_t *)results) + elems*2 - 1000);
	uint32_t *lenp = (uint32_t *)results;
	uint32_t len,Transitions=0;
	int failed=0;
	int16_t Val0,Val1,Val2,Val3,Val4,Val5,Val6,Val7,src2val;
	uint8_t Tval;
	uint32_t y_reg,x_reg;
	uint32_t nelems_y,nelems_x;
	uint32_t *indexPtr;
	uint32_t num_regions_x = ow/x_region_size;
	uint32_t num_regions_y = oh/y_region_size;

	if(ow%x_region_size)
		num_regions_x++;
	if(oh%y_region_size)
		num_regions_y++;


	*(uint32_t *)resp = htonl(PLACEKEY);
	resp +=4;

	indexPtr = (uint32_t *)resp;
	resp += 4*num_regions_x*num_regions_y; // leave room for indexes


	for(y_reg = 0;y_reg < num_regions_y;y_reg++)
	{
		for(x_reg = 0;x_reg < num_regions_x;x_reg++)
		{
			state = 0;
			nelems_x = x_region_size;
			nelems_y = y_region_size;

			if (((x_reg+1)*x_region_size) > (uint32_t)ow)
				nelems_x = ow - x_reg*x_region_size;
			if (((y_reg+1)*y_region_size) > (uint32_t)oh)
				nelems_y = oh - y_reg*y_region_size;

			indexPtr[y_reg*num_regions_x+x_reg] = htonl((uint32_t)(resp - (uint8_t *)results));

			for(y = 0;y<nelems_y;y++)
			{
				src1 = (uint16_t *)(framePtr     + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
				src2 = (uint16_t *)(prevFramePtr + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
				for(x=0;x<nelems_x;x+=8)
				{
					if (__builtin_expect((resp > limit),0))
					{
						failed = 1;
						break;
					}


					__builtin_prefetch(src1+0x10);
					__builtin_prefetch(src1+0x18);
					__builtin_prefetch(src2+0x10);
					__builtin_prefetch(src2+0x18);


#define GET_VAL(a) \
					a = (*src1++ & 0x3fff); \
					src2val = *src2++ & 0x3fff; \
					total += (uint16_t)a; \
					a -= src2val;

					GET_VAL(Val0);
					GET_VAL(Val1);
					GET_VAL(Val2);
					GET_VAL(Val3);
					GET_VAL(Val4);
					GET_VAL(Val5);
					GET_VAL(Val6);
					GET_VAL(Val7);

#define CHK(a) \
		    ((Val0 < (1<<(a-1))) && (Val0 > -(1<<(a-1))) && \
			 (Val1 < (1<<(a-1))) && (Val1 > -(1<<(a-1))) && \
			 (Val2 < (1<<(a-1))) && (Val2 > -(1<<(a-1))) && \
			 (Val3 < (1<<(a-1))) && (Val3 > -(1<<(a-1))) && \
			 (Val4 < (1<<(a-1))) && (Val4 > -(1<<(a-1))) && \
			 (Val5 < (1<<(a-1))) && (Val5 > -(1<<(a-1))) && \
			 (Val6 < (1<<(a-1))) && (Val6 > -(1<<(a-1))) && \
			 (Val7 < (1<<(a-1))) && (Val7 > -(1<<(a-1))))

#define WRITE_TRANSITION(a) \
			if((state != newState) || (Tval == KEY_0)) \
			{ \
				state = newState; \
				*resp++ = KEY_0; \
				*resp++ = a; \
				Transitions++; \
			}


#define ADD_OFFSET(a) \
		Val0 += 1 << (a - 1); \
		Val1 += 1 << (a - 1); \
		Val2 += 1 << (a - 1); \
		Val3 += 1 << (a - 1); \
		Val4 += 1 << (a - 1); \
		Val5 += 1 << (a - 1); \
		Val6 += 1 << (a - 1); \
		Val7 += 1 << (a - 1);

#if 0
					if (state != 4 && CHK(3))
					{
						newState = 3;
						ADD_OFFSET(3);
						Tval = (Val0 << 5) | (Val1 << 2) | (Val2 >> 1);
						WRITE_TRANSITION(0x33);
						*resp++ = Tval;
						*resp++ = (Val2 << 7) | (Val3 << 4) | (Val4 << 1) | (Val5 >> 2);
						*resp++ = (Val5 << 6) | (Val6 << 3) | Val7;
					}else
#endif
					if (state != 5 && CHK(4))
					{
						newState = 4;
						ADD_OFFSET(4);
						Tval = (Val0 << 4) | Val1;
						WRITE_TRANSITION(0x44);
						*resp++ = Tval;
						*resp++ = (Val2 << 4) | Val3;
						*resp++ = (Val4 << 4) | Val5;
						*resp++ = (Val6 << 4) | Val7;
					}
					else if (state != 6 && CHK(5))
					{
						newState = 5;
						ADD_OFFSET(5);
						Tval = (Val0 << 3) | (Val1 >> 2);
						WRITE_TRANSITION(0x55);
						*resp++ = Tval;
						*resp++ = (Val1 << 6) | (Val2 << 1) | (Val3 >> 4);
						*resp++ = (Val3 << 4) | (Val4 >> 1);
						*resp++ = (Val4 << 7) | (Val5 << 2) | (Val6 >> 3);
						*resp++ = (Val6 << 5) | (Val7);
					}
					else if (state != 7 && CHK(6))
					{
						newState = 6;
						ADD_OFFSET(6);
						Tval = ((Val0) << 2) | ((Val1) >> 4);
						WRITE_TRANSITION(0x66);
						*resp++ = Tval;
						*resp++ = (Val1 << 4) | ((Val2) >> 2);
						*resp++ = (Val2 << 6) | (Val3);
						*resp++ = (Val4 << 2) | ((Val5) >> 4);
						*resp++ = (Val5 << 4) | ((Val6) >> 2);
						*resp++ = (Val6 << 6) | (Val7);
					}
					else if (state != 8 && CHK(7))
					{
						newState = 7;
						ADD_OFFSET(7);
						Tval = (Val0 << 1) | (Val1 >> 6);
						WRITE_TRANSITION(0x77);
						*resp++ = Tval;
						*resp++ = (Val1 << 2) | (Val2 >> 5);
						*resp++ = (Val2 << 3) | (Val3 >> 4);
						*resp++ = (Val3 << 4) | (Val4 >> 3);
						*resp++ = (Val4 << 5) | (Val5 >> 2);
						*resp++ = (Val5 << 6) | (Val6 >> 1);
						*resp++ = (Val6 << 7) | (Val7);
					}
					else if (CHK(8))
					{
						newState = 8;
						ADD_OFFSET(8);
						Tval = Val0;
						WRITE_TRANSITION(0x88);
						*resp++ = Tval;
						*resp++ = Val1;
						*resp++ = Val2;
						*resp++ = Val3;
						*resp++ = Val4;
						*resp++ = Val5;
						*resp++ = Val6;
						*resp++ = Val7;
					}
					else
					{
						newState = 16;
						// were here because we need the whole 16 bits....
						Tval = ((Val0 & 0xff00) >> 8);
						WRITE_TRANSITION(0xBB);
						*resp++ = Tval;
						*resp++ = (Val0 & 0xff);
						*resp++ = ((Val1 & 0xff00) >> 8);
						*resp++ = (Val1 & 0xff);
						*resp++ = ((Val2 & 0xff00) >> 8);
						*resp++ = (Val2 & 0xff);
						*resp++ = ((Val3 & 0xff00) >> 8);
						*resp++ = (Val3 & 0xff);
						*resp++ = ((Val4 & 0xff00) >> 8);
						*resp++ = (Val4 & 0xff);
						*resp++ = ((Val5 & 0xff00) >> 8);
						*resp++ = (Val5 & 0xff);
						*resp++ = ((Val6 & 0xff00) >> 8);
						*resp++ = (Val6 & 0xff);
						*resp++ = ((Val7 & 0xff00) >> 8);
						*resp++ = (Val7 & 0xff);
					}
//					valBins[state]++;
				}
			}
		}
	}
	len = (uint32_t)(resp - (uint8_t *)results);
	len+= 3;
	len &= ~0x3; // quad-word allign it
	*lenp++ = htonl(len);
	*lenp++ = htonl(Transitions);
	*lenp++ = htonl(total);
//	DTRACE("bins: 2(%x) 3(%x) 4(%x) 5(%x) 6(%x) 7(%x) 8(%x) 16(%x)  T(%d/%d)\n",valBins[2],valBins[3],valBins[4],valBins[5],valBins[6],valBins[7],valBins[8],valBins[16],Transitions,NTransitions);

	*out_len = len; // return the number of bytes to the caller.
#undef GET_VAL

	return failed;
}

#endif

bool Acq::WriteAscii(const char *acqName, int ox, int oy, int ow, int oh)
{	//Unused parameter generates compiler warning, so...
	if (ox) {};
	if (oy) {};
	
	// open up the acq file
	FILE *fp;
	fp = fopen(acqName, "wb");
	if (!fp) {
		printf("Warning!  Could not open file: %s for writing?\n", acqName);
		return false;
	}
	
	// write each well trace.
	int frame;
	int ix, iy;

	for(iy=0;iy<oh;iy++) {
		for(ix=0;ix<ow;ix++) {
			for(frame=0;frame<numFrames;frame++) {
				fprintf (fp, "%u ", data[(frame*frameStride) + (ix+iy*w)]);
			}
			fprintf (fp, "\n");
		}
	}

	fclose(fp);
	return true;
}

void Acq::CalculateNumOfRegions(uint32_t ow, uint32_t oh) {
    num_regions_x = ow/x_region_size;
    num_regions_y = oh/y_region_size;
    
    if(ow%x_region_size)
        num_regions_x++;
    if(oh%y_region_size)
        num_regions_y++;    
}

bool Acq::CheckForRegionAcqTimeWindow(uint32_t baseInterval, uint32_t timeStamp, int regionNum, int* framesToAvg) {
    if (region_acq_start[regionNum] > timeStamp) {
        if (framesToAvg)
            framesToAvg[regionNum] += 1;
        return false;
    }
    else {
        if (framesToAvg && framesToAvg[regionNum] != -1) {
            framesToAvg[regionNum] += 1;  
        }  
        return true;
    }
}

int Acq::DeltaCompressionOnRegionalAcquisitionWindow(
    int elems, 
    uint32_t baseFrameRate,
    uint32_t timeStamp,
    unsigned int frameNum,
    int16_t* firstFramePtr,
    int16_t *framePtr, 
    int16_t* prevFramePtr, 
    int16_t *results, 
    uint64_t *out_len, 
    uint32_t ow, 
    uint32_t oh) 
{

    uint32_t x,y;
    uint16_t *src1=(uint16_t *)framePtr;
    uint16_t *src2=(uint16_t *)prevFramePtr;
    uint16_t *src3=(uint16_t *)firstFramePtr;
    int state = 0;
    int newState = 0;
    uint8_t *resp = (uint8_t *)((uint32_t *)results + 3);
    uint32_t total=0;
    uint8_t *limit = (uint8_t *)(((int8_t *)results) + elems*2 - 1000);
    uint32_t *lenp = (uint32_t *)results;
    uint32_t len,Transitions=0;
    int failed=0;
    int16_t Val0,Val1,Val2,Val3,Val4,Val5,Val6,Val7,src2val;
    uint8_t Tval;
    uint32_t y_reg,x_reg;
    uint32_t nelems_y,nelems_x;
    uint32_t *indexPtr;
    uint32_t regionNum = 0;
    bool regAcq;

    *(uint32_t *)resp = htonl(PLACEKEY);
    resp +=4;

    indexPtr = (uint32_t *)resp;
    resp += 4*num_regions_x*num_regions_y; // leave room for indexes

    for(y_reg = 0;y_reg < num_regions_y;y_reg++)
    {
        for(x_reg = 0;x_reg < num_regions_x;x_reg++)
        {
            state = 0;
            regAcq = false;
            nelems_x = x_region_size;
            nelems_y = y_region_size;
            regionNum = y_reg*num_regions_x+x_reg;

            if (((x_reg+1)*x_region_size) > (uint32_t)ow)
                nelems_x = ow - x_reg*x_region_size;
            if (((y_reg+1)*y_region_size) > (uint32_t)oh)
                nelems_y = oh - y_reg*y_region_size;

            //printf("Region: %d\n", regionNum);
            if (!CheckForRegionAcqTimeWindow(baseFrameRate, timeStamp, regionNum)) {
                indexPtr[regionNum] = htonl(0xFFFFFFFF);
                regAcq = true;   
            }
            else {
                indexPtr[regionNum] = htonl((uint32_t)(resp - (uint8_t *)results));
            }

            for(y = 0;y<nelems_y;y++)
            {
                src1 = (uint16_t *)(framePtr     + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
                src2 = (uint16_t *)(prevFramePtr + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
                src3 = (uint16_t *)(firstFramePtr + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
                for(x=0;x<nelems_x;x+=8)
                {
                    // copy first frame data till we start saving frames for this region
                    if (regAcq) {
                        //printf("Region: %d, offset: %d\n", regionNum, 0xFFFFFFFF);
                        for (int i=0; i<8; ++i) {
                            *src1 = (*src3 & 0x3fff);
                            //if (frameNum == 1 && *src2 != *src3)
                            //    printf("region: %d, y: %d, x: %d\n", regionNum, y, x);
                            //printf("%d ", *src3);
                            src1++;
                            src2++;
                            src3++;
                        }
                        //printf("\n");
                        continue; 
                    }

                    if (__builtin_expect((resp > limit),0))
                    {
                        failed = 1;
                        break;
                    }


                    __builtin_prefetch(src1+0x10);
                    __builtin_prefetch(src1+0x18);
                    __builtin_prefetch(src2+0x10);
                    __builtin_prefetch(src2+0x18);


#define GET_VAL(a) \
                    a = (*src1++ & 0x3fff); \
                    src2val = *src2++ & 0x3fff; \
                    total += (uint16_t)a; \
                    a -= src2val;

                    GET_VAL(Val0);
                    GET_VAL(Val1);
                    GET_VAL(Val2);
                    GET_VAL(Val3);
                    GET_VAL(Val4);
                    GET_VAL(Val5);
                    GET_VAL(Val6);
                    GET_VAL(Val7);
                        
                    //printf("%d %d %d %d %d %d %d %d\n", Val0, Val1, Val2, Val3, Val4, Val5, Val6, Val7);

#define CHK(a) \
            ((Val0 < (1<<(a-1))) && (Val0 > -(1<<(a-1))) && \
             (Val1 < (1<<(a-1))) && (Val1 > -(1<<(a-1))) && \
             (Val2 < (1<<(a-1))) && (Val2 > -(1<<(a-1))) && \
             (Val3 < (1<<(a-1))) && (Val3 > -(1<<(a-1))) && \
             (Val4 < (1<<(a-1))) && (Val4 > -(1<<(a-1))) && \
             (Val5 < (1<<(a-1))) && (Val5 > -(1<<(a-1))) && \
             (Val6 < (1<<(a-1))) && (Val6 > -(1<<(a-1))) && \
             (Val7 < (1<<(a-1))) && (Val7 > -(1<<(a-1))))

#define WRITE_TRANSITION(a) \
            if((state != newState) || (Tval == KEY_0)) \
            { \
                state = newState; \
                *resp++ = KEY_0; \
                *resp++ = a; \
                Transitions++; \
            }


#define ADD_OFFSET(a) \
        Val0 += 1 << (a - 1); \
        Val1 += 1 << (a - 1); \
        Val2 += 1 << (a - 1); \
        Val3 += 1 << (a - 1); \
        Val4 += 1 << (a - 1); \
        Val5 += 1 << (a - 1); \
        Val6 += 1 << (a - 1); \
        Val7 += 1 << (a - 1);

                    if (state != 5 && CHK(4))
                    {
                        newState = 4;
                        ADD_OFFSET(4);
                        Tval = (Val0 << 4) | Val1;
                        WRITE_TRANSITION(0x44);
                        *resp++ = Tval;
                        *resp++ = (Val2 << 4) | Val3;
                        *resp++ = (Val4 << 4) | Val5;
                        *resp++ = (Val6 << 4) | Val7;
                    }
                    else if (state != 6 && CHK(5))
                    {
                        newState = 5;
                        ADD_OFFSET(5);
                        Tval = (Val0 << 3) | (Val1 >> 2);
                        WRITE_TRANSITION(0x55);
                        *resp++ = Tval;
                        *resp++ = (Val1 << 6) | (Val2 << 1) | (Val3 >> 4);
                        *resp++ = (Val3 << 4) | (Val4 >> 1);
                        *resp++ = (Val4 << 7) | (Val5 << 2) | (Val6 >> 3);
                        *resp++ = (Val6 << 5) | (Val7);
                    }
                    else if (state != 7 && CHK(6))
                    {
                        newState = 6;
                        ADD_OFFSET(6);
                        Tval = ((Val0) << 2) | ((Val1) >> 4);
                        WRITE_TRANSITION(0x66);
                        *resp++ = Tval;
                        *resp++ = (Val1 << 4) | ((Val2) >> 2);
                        *resp++ = (Val2 << 6) | (Val3);
                        *resp++ = (Val4 << 2) | ((Val5) >> 4);
                        *resp++ = (Val5 << 4) | ((Val6) >> 2);
                        *resp++ = (Val6 << 6) | (Val7);
                    }
                    else if (state != 8 && CHK(7))
                    {
                        newState = 7;
                        ADD_OFFSET(7);
                        Tval = (Val0 << 1) | (Val1 >> 6);
                        WRITE_TRANSITION(0x77);
                        *resp++ = Tval;
                        *resp++ = (Val1 << 2) | (Val2 >> 5);
                        *resp++ = (Val2 << 3) | (Val3 >> 4);
                        *resp++ = (Val3 << 4) | (Val4 >> 3);
                        *resp++ = (Val4 << 5) | (Val5 >> 2);
                        *resp++ = (Val5 << 6) | (Val6 >> 1);
                        *resp++ = (Val6 << 7) | (Val7);
                    }
                    else if (CHK(8))
                    {
                        newState = 8;
                        ADD_OFFSET(8);
                        Tval = Val0;
                        WRITE_TRANSITION(0x88);
                        *resp++ = Tval;
                        *resp++ = Val1;
                        *resp++ = Val2;
                        *resp++ = Val3;
                        *resp++ = Val4;
                        *resp++ = Val5;
                        *resp++ = Val6;
                        *resp++ = Val7;
                    }
                    else
                    {
                        newState = 16;
                        // were here because we need the whole 16 bits....
                        Tval = ((Val0 & 0xff00) >> 8);
                        WRITE_TRANSITION(KEY_16_1);
                        *resp++ = Tval;
                        *resp++ = (Val0 & 0xff);
                        *resp++ = ((Val1 & 0xff00) >> 8);
                        *resp++ = (Val1 & 0xff);
                        *resp++ = ((Val2 & 0xff00) >> 8);
                        *resp++ = (Val2 & 0xff);
                        *resp++ = ((Val3 & 0xff00) >> 8);
                        *resp++ = (Val3 & 0xff);
                        *resp++ = ((Val4 & 0xff00) >> 8);
                        *resp++ = (Val4 & 0xff);
                        *resp++ = ((Val5 & 0xff00) >> 8);
                        *resp++ = (Val5 & 0xff);
                        *resp++ = ((Val6 & 0xff00) >> 8);
                        *resp++ = (Val6 & 0xff);
                        *resp++ = ((Val7 & 0xff00) >> 8);
                        *resp++ = (Val7 & 0xff);
                    }
                }
            }
        }
    }
    len = (uint32_t)(resp - (uint8_t *)results);
    len+= 3;
    len &= ~0x3; // quad-word allign it
    *lenp++ = htonl(len);
    *lenp++ = htonl(Transitions);
    *lenp++ = htonl(total);
    *out_len = len; // return the number of bytes to the caller.
#undef GET_VAL

    return failed;

}

int Acq::FramesBeforeT0Averaged(
    int elems, 
    uint32_t baseFrameRate,
    uint32_t timeStamp,
    unsigned int frameNum,
    int* framesToAvg,
    int32_t* avgFramePtr,
    int16_t* firstFramePtr,
    int16_t *framePtr, 
    int16_t* prevFramePtr, 
    int16_t *results, 
    uint64_t *out_len, 
    uint32_t ow, 
    uint32_t oh) 
{

    uint32_t x,y;
    uint16_t *src1=(uint16_t *)framePtr;
    uint16_t *src2=(uint16_t *)prevFramePtr;
    uint16_t *src3=(uint16_t *)firstFramePtr;
    uint32_t *src4=(uint32_t *)avgFramePtr;
    int state = 0;
    int newState = 0;
    uint8_t *resp = (uint8_t *)((uint32_t *)results + 3);
    uint32_t total=0;
    uint8_t *limit = (uint8_t *)(((int8_t *)results) + elems*2 - 1000);
    uint32_t *lenp = (uint32_t *)results;
    uint32_t len,Transitions=0;
    int failed=0;
    int16_t Val0,Val1,Val2,Val3,Val4,Val5,Val6,Val7,src2val;
    uint8_t Tval;
    uint32_t y_reg,x_reg;
    uint32_t nelems_y,nelems_x;
    uint32_t *indexPtr;
    uint32_t regionNum = 0;
    bool regAcq, avgFrameHit;

    *(uint32_t *)resp = htonl(PLACEKEY);
    resp +=4;

    indexPtr = (uint32_t *)resp;
    resp += 4*num_regions_x*num_regions_y; // leave room for indexes

    //bool printVal = true;
    for(y_reg = 0;y_reg < num_regions_y;y_reg++)
    {
        for(x_reg = 0;x_reg < num_regions_x;x_reg++)
        {
            state = 0;
            regAcq = false;
            avgFrameHit = false;
            nelems_x = x_region_size;
            nelems_y = y_region_size;
            regionNum = y_reg*num_regions_x+x_reg;

            if (((x_reg+1)*x_region_size) > (uint32_t)ow)
                nelems_x = ow - x_reg*x_region_size;
            if (((y_reg+1)*y_region_size) > (uint32_t)oh)
                nelems_y = oh - y_reg*y_region_size;

            if (!CheckForRegionAcqTimeWindow(baseFrameRate, timeStamp, regionNum, framesToAvg)) {
                indexPtr[regionNum] = htonl(0xFFFFFFFF);
                regAcq = true;   
            }
            else {
                avgFrameHit = true;
                indexPtr[regionNum] = htonl((uint32_t)(resp - (uint8_t *)results));
            }

            
            for(y = 0;y<nelems_y;y++)
            {
                src1 = (uint16_t *)(framePtr     + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
                src2 = (uint16_t *)(prevFramePtr + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
                src3 = (uint16_t *)(firstFramePtr + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
                src4 = (uint32_t *)(avgFramePtr + ((y+(y_reg*y_region_size))*ow + (x_reg*x_region_size)));
                for(x=0;x<nelems_x;x+=8)
                {
                    // copy first frame data till we start saving frames for this region
                    if (regAcq) {
                        for (int i=0; i<8; ++i) {
                            *src4 += *src1;
                            /*if (regionNum == 0 && printVal) {
                                printf("frameNum: %d\n", frameNum);
                                printf("%d %d %d %d %d %d %d %d\n", src1[0], src1[1], src1[2], src1[3], src1[4], src1[5], src1[6], src1[7]);
                                printVal = false;
                            }*/

                            *src1 = (*src3 & 0x3fff);
                            src1++;
                            src3++;
                            src4++;
                        }
                        continue; 
                    }
                    else if (framesToAvg[regionNum] != -1) {
                        for (int i=0; i<8; ++i) {
                            *src4 += src1[i];
                            *src4 /= framesToAvg[regionNum];
                            src1[i] = *src4++;
                        }
                    }

                    if (__builtin_expect((resp > limit),0))
                    {
                        failed = 1;
                        break;
                    }


                    __builtin_prefetch(src1+0x10);
                    __builtin_prefetch(src1+0x18);
                    __builtin_prefetch(src2+0x10);
                    __builtin_prefetch(src2+0x18);


#define GET_VAL(a) \
                    a = (*src1++ & 0x3fff); \
                    src2val = *src2++ & 0x3fff; \
                    total += (uint16_t)a; \
                    a -= src2val;

                    GET_VAL(Val0);
                    GET_VAL(Val1);
                    GET_VAL(Val2);
                    GET_VAL(Val3);
                    GET_VAL(Val4);
                    GET_VAL(Val5);
                    GET_VAL(Val6);
                    GET_VAL(Val7);
                        
                    //printf("%d %d %d %d %d %d %d %d\n", Val0, Val1, Val2, Val3, Val4, Val5, Val6, Val7);

#define CHK(a) \
            ((Val0 < (1<<(a-1))) && (Val0 > -(1<<(a-1))) && \
             (Val1 < (1<<(a-1))) && (Val1 > -(1<<(a-1))) && \
             (Val2 < (1<<(a-1))) && (Val2 > -(1<<(a-1))) && \
             (Val3 < (1<<(a-1))) && (Val3 > -(1<<(a-1))) && \
             (Val4 < (1<<(a-1))) && (Val4 > -(1<<(a-1))) && \
             (Val5 < (1<<(a-1))) && (Val5 > -(1<<(a-1))) && \
             (Val6 < (1<<(a-1))) && (Val6 > -(1<<(a-1))) && \
             (Val7 < (1<<(a-1))) && (Val7 > -(1<<(a-1))))

#define WRITE_TRANSITION(a) \
            if((state != newState) || (Tval == KEY_0)) \
            { \
                state = newState; \
                *resp++ = KEY_0; \
                *resp++ = a; \
                Transitions++; \
            }


#define ADD_OFFSET(a) \
        Val0 += 1 << (a - 1); \
        Val1 += 1 << (a - 1); \
        Val2 += 1 << (a - 1); \
        Val3 += 1 << (a - 1); \
        Val4 += 1 << (a - 1); \
        Val5 += 1 << (a - 1); \
        Val6 += 1 << (a - 1); \
        Val7 += 1 << (a - 1);

                    if (state != 5 && CHK(4))
                    {
                        newState = 4;
                        ADD_OFFSET(4);
                        Tval = (Val0 << 4) | Val1;
                        WRITE_TRANSITION(0x44);
                        *resp++ = Tval;
                        *resp++ = (Val2 << 4) | Val3;
                        *resp++ = (Val4 << 4) | Val5;
                        *resp++ = (Val6 << 4) | Val7;
                    }
                    else if (state != 6 && CHK(5))
                    {
                        newState = 5;
                        ADD_OFFSET(5);
                        Tval = (Val0 << 3) | (Val1 >> 2);
                        WRITE_TRANSITION(0x55);
                        *resp++ = Tval;
                        *resp++ = (Val1 << 6) | (Val2 << 1) | (Val3 >> 4);
                        *resp++ = (Val3 << 4) | (Val4 >> 1);
                        *resp++ = (Val4 << 7) | (Val5 << 2) | (Val6 >> 3);
                        *resp++ = (Val6 << 5) | (Val7);
                    }
                    else if (state != 7 && CHK(6))
                    {
                        newState = 6;
                        ADD_OFFSET(6);
                        Tval = ((Val0) << 2) | ((Val1) >> 4);
                        WRITE_TRANSITION(0x66);
                        *resp++ = Tval;
                        *resp++ = (Val1 << 4) | ((Val2) >> 2);
                        *resp++ = (Val2 << 6) | (Val3);
                        *resp++ = (Val4 << 2) | ((Val5) >> 4);
                        *resp++ = (Val5 << 4) | ((Val6) >> 2);
                        *resp++ = (Val6 << 6) | (Val7);
                    }
                    else if (state != 8 && CHK(7))
                    {
                        newState = 7;
                        ADD_OFFSET(7);
                        Tval = (Val0 << 1) | (Val1 >> 6);
                        WRITE_TRANSITION(0x77);
                        *resp++ = Tval;
                        *resp++ = (Val1 << 2) | (Val2 >> 5);
                        *resp++ = (Val2 << 3) | (Val3 >> 4);
                        *resp++ = (Val3 << 4) | (Val4 >> 3);
                        *resp++ = (Val4 << 5) | (Val5 >> 2);
                        *resp++ = (Val5 << 6) | (Val6 >> 1);
                        *resp++ = (Val6 << 7) | (Val7);
                    }
                    else if (CHK(8))
                    {
                        newState = 8;
                        ADD_OFFSET(8);
                        Tval = Val0;
                        WRITE_TRANSITION(0x88);
                        *resp++ = Tval;
                        *resp++ = Val1;
                        *resp++ = Val2;
                        *resp++ = Val3;
                        *resp++ = Val4;
                        *resp++ = Val5;
                        *resp++ = Val6;
                        *resp++ = Val7;
                    }
                    else
                    {
                        newState = 16;
                        // were here because we need the whole 16 bits....
                        Tval = ((Val0 & 0xff00) >> 8);
                        WRITE_TRANSITION(KEY_16_1);
                        *resp++ = Tval;
                        *resp++ = (Val0 & 0xff);
                        *resp++ = ((Val1 & 0xff00) >> 8);
                        *resp++ = (Val1 & 0xff);
                        *resp++ = ((Val2 & 0xff00) >> 8);
                        *resp++ = (Val2 & 0xff);
                        *resp++ = ((Val3 & 0xff00) >> 8);
                        *resp++ = (Val3 & 0xff);
                        *resp++ = ((Val4 & 0xff00) >> 8);
                        *resp++ = (Val4 & 0xff);
                        *resp++ = ((Val5 & 0xff00) >> 8);
                        *resp++ = (Val5 & 0xff);
                        *resp++ = ((Val6 & 0xff00) >> 8);
                        *resp++ = (Val6 & 0xff);
                        *resp++ = ((Val7 & 0xff00) >> 8);
                        *resp++ = (Val7 & 0xff);
                    }
                }
            }
            if(avgFrameHit)
                framesToAvg[regionNum] = -1;
        }
    }
    len = (uint32_t)(resp - (uint8_t *)results);
    len+= 3;
    len &= ~0x3; // quad-word allign it
    *lenp++ = htonl(len);
    *lenp++ = htonl(Transitions);
    *lenp++ = htonl(total);
    *out_len = len; // return the number of bytes to the caller.
#undef GET_VAL

    return failed;

}


bool Acq::WriteRegionBasedAcq(char *acqName, int ox, int oy, int ow, int oh)
{
    // open up the acq file
    const RawImage *raw = image->GetImage();
    const float acqTime = ACQENDTIME;

    FILE *fp;
    fp = fopen(acqName, "wb");
    if (!fp) {
        printf("Warning!  Could not open file: %s for writing?\n", acqName);
        return false;
    }

    _file_hdr    fileHdr;
    _expmt_hdr_v4    expHdr;


    uint32_t vfr_array_cnt = 0;
    uint32_t vfr_total_cnt=0;
    uint32_t offset=0;

    unsigned int sample_rate = raw->timestamps[0];

    float cutOffTime = 0.0;
    for (int i=0; i<raw->frames; ++i) {
        cutOffTime = (float)raw->timestamps[i]/(float)1000.0;
        if (cutOffTime > acqTime) {
            vfr_array_cnt = i; 
            break;
        }           
    }
    vfr_total_cnt = GetUncompressedFrames(vfr_array_cnt);
    //vfr_total_cnt = ceil(acqTime / (sample_rate/(float)1000.0));
    //vfr_array_cnt = GetCompressedFrameNum(vfr_total_cnt);
    printf("VFC frames: %d Total Frames: %d sample rate %d\n", vfr_array_cnt, vfr_total_cnt, sample_rate);

    memset(&fileHdr,0,sizeof(fileHdr));
    memset(&expHdr,0,sizeof(expHdr));

    // set up the file header
    fileHdr.signature = 0xdeadbeef;
    fileHdr.struct_version = 0x4;
    fileHdr.header_size = sizeof(_expmt_hdr_v4);
    unsigned long totalSize = vfr_array_cnt * (ow*oh*2 + sizeof(uint32_t)); // data + timestamp
    // hdr.fileHdr.data_size = numFrames*w*h*2; // old bad format
    fileHdr.data_size = totalSize; // new good format

    ByteSwap4(fileHdr.signature);
    ByteSwap4(fileHdr.struct_version);
    ByteSwap4(fileHdr.header_size);
    ByteSwap4(fileHdr.data_size);

    printf("rows: %d, cols: %d\n", oh, ow);

    // setup the data header
    expHdr.first_frame_time = 0;
    expHdr.rows = oh;
    expHdr.cols = ow;
    expHdr.frames_in_file = vfr_array_cnt;
    expHdr.uncomp_frames_in_file = vfr_total_cnt;
    expHdr.interlaceType = 6;
    expHdr.x_region_size = x_region_size;
    expHdr.y_region_size = y_region_size;
    expHdr.sample_rate = sample_rate;
//    expHdr.channel_offset[0] = raw->

    ByteSwap4(expHdr.first_frame_time);
    ByteSwap2(expHdr.rows);
    ByteSwap2(expHdr.cols);
    ByteSwap2(expHdr.frames_in_file);
    ByteSwap2(expHdr.uncomp_frames_in_file);
    ByteSwap2(expHdr.interlaceType);
    ByteSwap2(expHdr.x_region_size);
    ByteSwap2(expHdr.y_region_size);
    ByteSwap4(expHdr.sample_rate);

    // write file & data headers
    fwrite(&fileHdr, sizeof(fileHdr), 1, fp);
    offset += sizeof(fileHdr);
    fwrite(&expHdr, sizeof(expHdr), 1, fp);
    offset += sizeof(expHdr);

    // write each frame block (timestamp & frame data)
    uint32_t frame;
    int ix, iy;
    int16_t *frame_data,*sptr, *first_frame;

    frame_data = (int16_t *)malloc(2*ow*oh);
    first_frame = (int16_t *)malloc(2*ow*oh);
    memset(first_frame, 0, 2*ow*oh);
    int16_t *prev_data = NULL;
    int16_t *results_data = (int16_t *)malloc(2*ow*oh);

    for(frame=0;frame<vfr_array_cnt;frame++) {
        uint32_t timestampOut = BYTE_SWAP_4(raw->timestamps[frame]); // write out the last time stamp...
        //printf("Frame %d t=%.1f\n", frame, raw->timestamps[frame]/(float)1000);
        fwrite(&timestampOut, sizeof(timestampOut), 1, fp);
        offset += sizeof(timestampOut);

        int16_t *ptr = frame_data;
        uint16_t tmp[4];
        uint64_t results_len=0;
        uint32_t comp;

        results_len=0;
        comp=0;



        // save the data into frame_data
        for(iy=0;iy<oh;iy++)
        {
            sptr = &raw->image[frame*raw->frameStride+(iy+oy)*raw->cols+ox];
            for(ix=0;ix<ow;ix++)
            {
                *ptr++ = *sptr++;
            }
        }

        if (frame == 0) {
            memcpy(first_frame, frame_data, 2*ow*oh);
        }

        if(prev_data)
        {
            if(DeltaCompressionOnRegionalAcquisitionWindow(
                    ow*oh,
                    raw->timestamps[0],
                    raw->timestamps[frame],
                    frame,
                    first_frame,
                    frame_data,
                    prev_data,
                    results_data,
                    &results_len,
                    ow,
                    oh) == 0)
            {
                comp = htonl(1);
            }
            else
            {
                  printf("Delta compression failed\n");
            }
        }

        fwrite(&comp, sizeof(comp), 1, fp);

        offset += sizeof(comp);

        if(!comp)
        {
            ptr = frame_data;
            for(iy=0;iy<oh;iy++)
            {
                for(ix=0;ix<ow;ix++)
                {

                    tmp[0] = BYTE_SWAP_2(*ptr);
                    fwrite(&tmp[0],2,1,fp);
                    ptr++;
                }
            }
        }
        else
        {
            // write out the compressed data
            fwrite(results_data,1,results_len,fp);
            offset += results_len;
        }


        if(prev_data)
        {
            int16_t *tmp_data = frame_data;
            frame_data = prev_data;
            prev_data = tmp_data;
        }
        else
        {
            prev_data = frame_data;
            frame_data = (int16_t *)malloc(2*ow*oh);
        }

        if(comp)
            printf(".");
        else
            printf("-");
        fflush(stdout);

    }

    printf("\n");
    printf("\n");

    free(first_frame);
    free(frame_data);
    free(results_data);
    if(prev_data)
        free(prev_data);
    fclose(fp);
    return true;
}

bool Acq::WriteFrameAveragedRegionBasedAcq(char *acqName, int ox, int oy, int ow, int oh)
{
    // open up the acq file
    const RawImage *raw = image->GetImage();
    const float acqTime = ACQENDTIME;

    FILE *fp;
    fp = fopen(acqName, "wb");
    if (!fp) {
        printf("Warning!  Could not open file: %s for writing?\n", acqName);
        return false;
    }

    _file_hdr    fileHdr;
    _expmt_hdr_v4    expHdr;


    uint32_t vfr_array_cnt = 0;
    uint32_t vfr_total_cnt=0;
    uint32_t offset=0;

    unsigned int sample_rate = raw->timestamps[0];

    float cutOffTime = 0.0;
    for (int i=0; i<raw->frames; ++i) {
        cutOffTime = (float)raw->timestamps[i]/(float)1000.0;
        if (cutOffTime > acqTime) {
            vfr_array_cnt = i; 
            break;
        }           
    }
    vfr_total_cnt = GetUncompressedFrames(vfr_array_cnt);
    //vfr_total_cnt = ceil(acqTime / (sample_rate/(float)1000.0));
    //vfr_array_cnt = GetCompressedFrameNum(vfr_total_cnt);
    printf("VFC frames: %d Total Frames: %d sample rate %d\n", vfr_array_cnt, vfr_total_cnt, sample_rate);

    memset(&fileHdr,0,sizeof(fileHdr));
    memset(&expHdr,0,sizeof(expHdr));

    // set up the file header
    fileHdr.signature = 0xdeadbeef;
    fileHdr.struct_version = 0x4;
    fileHdr.header_size = sizeof(_expmt_hdr_v4);
    unsigned long totalSize = vfr_array_cnt * (ow*oh*2 + sizeof(uint32_t)); // data + timestamp
    // hdr.fileHdr.data_size = numFrames*w*h*2; // old bad format
    fileHdr.data_size = totalSize; // new good format

    ByteSwap4(fileHdr.signature);
    ByteSwap4(fileHdr.struct_version);
    ByteSwap4(fileHdr.header_size);
    ByteSwap4(fileHdr.data_size);

    printf("rows: %d, cols: %d\n", oh, ow);

    // setup the data header
    expHdr.first_frame_time = 0;
    expHdr.rows = oh;
    expHdr.cols = ow;
    expHdr.frames_in_file = vfr_array_cnt;
    expHdr.uncomp_frames_in_file = vfr_total_cnt;
    expHdr.interlaceType = 6;
    expHdr.x_region_size = x_region_size;
    expHdr.y_region_size = y_region_size;
    expHdr.sample_rate = sample_rate;
//    expHdr.channel_offset[0] = raw->

    ByteSwap4(expHdr.first_frame_time);
    ByteSwap2(expHdr.rows);
    ByteSwap2(expHdr.cols);
    ByteSwap2(expHdr.frames_in_file);
    ByteSwap2(expHdr.uncomp_frames_in_file);
    ByteSwap2(expHdr.interlaceType);
    ByteSwap2(expHdr.x_region_size);
    ByteSwap2(expHdr.y_region_size);
    ByteSwap4(expHdr.sample_rate);

    // write file & data headers
    fwrite(&fileHdr, sizeof(fileHdr), 1, fp);
    offset += sizeof(fileHdr);
    fwrite(&expHdr, sizeof(expHdr), 1, fp);
    offset += sizeof(expHdr);

    // write each frame block (timestamp & frame data)
    uint32_t frame;
    int ix, iy;
    int16_t *frame_data,*sptr, *first_frame;

    frame_data = (int16_t *)malloc(2*ow*oh);
    first_frame = (int16_t *)malloc(4*ow*oh);
    int32_t* avgFrame = (int32_t *)malloc(4*ow*oh);
    memset(first_frame, 0, 2*ow*oh);
    memset(avgFrame, 0, 4*ow*oh);
    int16_t *prev_data = NULL;
    int16_t *results_data = (int16_t *)malloc(2*ow*oh);

    int numRegions = (int)(num_regions_x*num_regions_y);
    int* framesToAvg = (int*)malloc(sizeof(int)*numRegions);
    memset(framesToAvg, 0, sizeof(int)*numRegions);

    for(frame=0;frame<vfr_array_cnt;frame++) {
        uint32_t timestampOut = BYTE_SWAP_4(raw->timestamps[frame]); // write out the last time stamp...
        //printf("Frame %d t=%.1f\n", frame, raw->timestamps[frame]/(float)1000);
        fwrite(&timestampOut, sizeof(timestampOut), 1, fp);
        offset += sizeof(timestampOut);

        int16_t *ptr = frame_data;
        uint16_t tmp[4];
        uint64_t results_len=0;
        uint32_t comp;
        //int32_t* avgtmp;

        results_len=0;
        comp=0;



        // save the data into frame_data
        for(iy=0;iy<oh;iy++)
        {
            sptr = &raw->image[frame*raw->frameStride+(iy+oy)*raw->cols+ox];
            for(ix=0;ix<ow;ix++)
            {
                *ptr++ = *sptr++;
            }
        }

        if (frame == 0) {
            /*avgtmp = avgFrame;
            for(iy=0;iy<oh;iy++)
            {
                sptr = &raw->image[frame*raw->frameStride+(iy+oy)*raw->cols+ox];
                for(ix=0;ix<ow;ix++)
                {
                    *avgtmp++ = *sptr++;
                }
            }
            for(iy=0; iy<numRegions; ++iy)
                framesToAvg[iy] = 1; */
            memcpy(first_frame, frame_data, 2*ow*oh);
        }

        if(prev_data)
        {
            if(FramesBeforeT0Averaged(
                    ow*oh,
                    raw->timestamps[0],
                    raw->timestamps[frame],
                    frame,
                    framesToAvg,
                    avgFrame,
                    first_frame,
                    frame_data,
                    prev_data,
                    results_data,
                    &results_len,
                    ow,
                    oh) == 0)
            {
                comp = htonl(1);
            }
            else
            {
                  printf("Delta compression failed\n");
            }
        }

        fwrite(&comp, sizeof(comp), 1, fp);

        offset += sizeof(comp);

        if(!comp)
        {
            ptr = frame_data;
            for(iy=0;iy<oh;iy++)
            {
                for(ix=0;ix<ow;ix++)
                {

                    tmp[0] = BYTE_SWAP_2(*ptr);
                    fwrite(&tmp[0],2,1,fp);
                    ptr++;
                }
            }
        }
        else
        {
            // write out the compressed data
            fwrite(results_data,1,results_len,fp);
            offset += results_len;
        }


        if(prev_data)
        {
            int16_t *tmp_data = frame_data;
            frame_data = prev_data;
            prev_data = tmp_data;
        }
        else
        {
            prev_data = frame_data;
            frame_data = (int16_t *)malloc(2*ow*oh);
        }

        if(comp)
            printf(".");
        else
            printf("-");
        fflush(stdout);

    }

    printf("\n");
    printf("\n");

    free(avgFrame);
    free(framesToAvg);
    free(first_frame);
    free(frame_data);
    free(results_data);
    if(prev_data)
        free(prev_data);
    fclose(fp);
    return true;
}


unsigned int Acq::GetCompressedFrameNum(unsigned int unCompressedFrameNum) {
 
  /* 
    ADD_FRAME(1  ,1);   // first frame is always at the base acquisition time
    ADD_FRAME(1  ,8);   // this accounts for the first second T=1
    ADD_FRAME(1  ,4);   // this accounts for the first second T=1
    ADD_FRAME(52 ,1);   // the next 3 seconds  T=5
    ADD_FRAME(12 ,4);   // the next 3 seconds T=8
    ADD_FRAME(8  ,8);   // the next 4 seconds T=8   
   */

   // assuming above VFC compression scheme
    if (unCompressedFrameNum == 0)
        return 0;
    else if (unCompressedFrameNum >= 1 && unCompressedFrameNum <= 8) 
        return 1;
    else if (unCompressedFrameNum >= 9 && unCompressedFrameNum <= 12)
        return 2;
    else if (unCompressedFrameNum >= 13 && unCompressedFrameNum <= 64)
        return (unCompressedFrameNum - 10);
    else if (unCompressedFrameNum >= 65 && unCompressedFrameNum <= 112)
        return (54 + ((unCompressedFrameNum - 65)/4 + 1));
    else if (unCompressedFrameNum >= 113 && unCompressedFrameNum <= 176)
        return (66 + ((unCompressedFrameNum - 113)/8 + 1)); 
    else 
        return 0;

}

unsigned int Acq::GetUncompressedFrames(unsigned int compressedFrames) {
 
  /* 
    ADD_FRAME(1  ,1);   // first frame is always at the base acquisition time
    ADD_FRAME(1  ,8);   // this accounts for the first second T=1
    ADD_FRAME(1  ,4);   // this accounts for the first second T=1
    ADD_FRAME(52 ,1);   // the next 3 seconds  T=5
    ADD_FRAME(12 ,4);   // the next 3 seconds T=8
    ADD_FRAME(8  ,8);   // the next 4 seconds T=8   
   */

   // assuming above VFC compression scheme
    if (compressedFrames == 1)
        return 1;
    else if (compressedFrames == 2) 
        return 9;
    else if (compressedFrames == 3)
        return 13;
    else if (compressedFrames > 3 && compressedFrames <= 55)
        return 13 + (compressedFrames - 3);
    else if (compressedFrames > 55 && compressedFrames <= 67)
        return (65 + (compressedFrames - 55)*4);
    else if (compressedFrames > 67 && compressedFrames <= 75)
        return (113 + (compressedFrames - 67)*8); 
    else 
        return 0;

}

bool Acq::WriteTimeBasedAcq(char *acqName, int ox, int oy, int ow, int oh)
{
    // open up the acq file
    const RawImage *raw = image->GetImage();
    const float acqTime = ACQENDTIME; // acquisition time window

    FILE *fp;
    fp = fopen(acqName, "wb");
    if (!fp) {
        printf("Warning!  Could not open file: %s for writing?\n", acqName);
        return false;
    }

    _file_hdr    fileHdr;
    _expmt_hdr_v4    expHdr;


    uint32_t vfr_array_cnt = 0;
    uint32_t vfr_total_cnt=0;
    uint32_t offset=0;

    unsigned int sample_rate = image->GetImage()->timestamps[0];

    float cutOffTime = 0.0;
    for (int i=0; i<raw->frames; ++i) {
        cutOffTime = (float)raw->timestamps[i]/(float)1000.0;
        if (cutOffTime > acqTime) {
            vfr_array_cnt = i;
            break;
        }
    }
    vfr_total_cnt = GetUncompressedFrames(vfr_array_cnt);
    //vfr_total_cnt = ceil(acqTime / (sample_rate/(float)1000.0));
    //vfr_array_cnt = GetCompressedFrameNum(vfr_total_cnt);
    printf("VFC frames: %d Total Frames: %d sample rate %d\n", vfr_array_cnt, vfr_total_cnt, sample_rate);

    //vfr_array_cnt = raw->frames;
    //vfr_total_cnt = raw->uncompFrames;

    memset(&fileHdr,0,sizeof(fileHdr));
    memset(&expHdr,0,sizeof(expHdr));

    // set up the file header
    fileHdr.signature = 0xdeadbeef;
    fileHdr.struct_version = 0x4;
    fileHdr.header_size = sizeof(_expmt_hdr_v4);
    unsigned long totalSize = vfr_array_cnt * (ow*oh*2 + sizeof(uint32_t)); // data + timestamp
    // hdr.fileHdr.data_size = numFrames*w*h*2; // old bad format
    fileHdr.data_size = totalSize; // new good format

    ByteSwap4(fileHdr.signature);
    ByteSwap4(fileHdr.struct_version);
    ByteSwap4(fileHdr.header_size);
    ByteSwap4(fileHdr.data_size);



    // setup the data header
    expHdr.first_frame_time = 0;
    expHdr.rows = oh;
    expHdr.cols = ow;
    expHdr.frames_in_file = vfr_array_cnt;
    expHdr.uncomp_frames_in_file = vfr_total_cnt;
    expHdr.interlaceType = 6;
    expHdr.x_region_size = x_region_size;
    expHdr.y_region_size = y_region_size;
    expHdr.sample_rate = sample_rate;
//    expHdr.channel_offset[0] = raw->

    ByteSwap4(expHdr.first_frame_time);
    ByteSwap2(expHdr.rows);
    ByteSwap2(expHdr.cols);
    ByteSwap2(expHdr.frames_in_file);
    ByteSwap2(expHdr.uncomp_frames_in_file);
    ByteSwap2(expHdr.interlaceType);
    ByteSwap2(expHdr.x_region_size);
    ByteSwap2(expHdr.y_region_size);
    ByteSwap4(expHdr.sample_rate);

    // write file & data headers
    fwrite(&fileHdr, sizeof(fileHdr), 1, fp);
    offset += sizeof(fileHdr);
    fwrite(&expHdr, sizeof(expHdr), 1, fp);
    offset += sizeof(expHdr);

    // write each frame block (timestamp & frame data)
    uint32_t frame;
    int ix, iy;
    int16_t *frame_data,*sptr;

    frame_data = (int16_t *)malloc(2*ow*oh);
    int16_t *prev_data = NULL;
    int16_t *results_data = (int16_t *)malloc(2*ow*oh);

    for(frame=0;frame<vfr_array_cnt;frame++) {
        uint32_t timestampOut = BYTE_SWAP_4(raw->timestamps[frame]); // write out the last time stamp...
        fwrite(&timestampOut, sizeof(timestampOut), 1, fp);
        offset += sizeof(timestampOut);

        int16_t *ptr = frame_data;
        uint16_t tmp[4];
        uint64_t results_len=0;
        uint32_t comp;
        uint32_t comprType=0;

        results_len=0;
        comp=0;



        // save the data into frame_data
        for(iy=0;iy<oh;iy++)
        {
            sptr = &raw->image[frame*raw->frameStride+(iy+oy)*raw->cols+ox];
            for(ix=0;ix<ow;ix++)
            {
                *ptr++ = *sptr++;
            }
        }

        if(prev_data)
        {
            if(PrevFrameSubtract(
                    ow,oh,
                    frame_data,
                    prev_data,
                    results_data,
                    &results_len,
                    &comprType) == 0)
            {
                comp = htonl(comprType);
            }
            else
            {
                  printf("Delta compression failed\n");
            }
        }

        fwrite(&comp, sizeof(comp), 1, fp);

        offset += sizeof(comp);

        if(!comp)
        {
            ptr = frame_data;
            for(iy=0;iy<oh;iy++)
            {
                for(ix=0;ix<ow;ix++)
                {

                    tmp[0] = BYTE_SWAP_2(*ptr);
                    fwrite(&tmp[0],2,1,fp);
                    ptr++;
                }
            }
        }
        else
        {
            // write out the compressed data
            fwrite(results_data,1,results_len,fp);
            offset += results_len;
        }


        if(prev_data)
        {
            int16_t *tmp_data = frame_data;
            frame_data = prev_data;
            prev_data = tmp_data;
        }
        else
        {
            prev_data = frame_data;
            frame_data = (int16_t *)malloc(2*ow*oh);
        }

        if(comp)
            printf(".");
        else
            printf("-");
        fflush(stdout);

    }

    printf("\n");
    printf("\n");

    free(frame_data);
    free(results_data);
    if(prev_data)
        free(prev_data);
    fclose(fp);
    return true;
}

bool Acq::WritePFV(char *acqName, int ox, int oy, int ow, int oh, char *options)
{
	// open up the acq file
    const RawImage *raw = image->GetImage();

    int fd;
    fd = open(acqName, O_CREAT | O_WRONLY,0644);
    if (fd < 0) {
        printf("Warning!  Could not open file: %s for writing?\n", acqName);
        return false;
    }
    int dbgType = 0;
    if(options && options[0] >= '1' && options[0] <= '9')
      dbgType = (int)options[0]-(int)'0';
	int framerate=1000/raw->timestamps[0];
	printf("input_framerate=%d\n",framerate);
    if(ox == 0 && oy == 0 && ow == raw->cols && oh == raw->rows)
    {
    	// full image
    	AdvCompr advc(fd, raw->image, raw->cols, raw->rows, raw->frames,
    			raw->uncompFrames, raw->timestamps,NULL,dbgType,acqName,options,framerate);
    	advc.Compress(-1,0);
    }
    else
    {
    	// create a cropped dataset, then save it.
    	short int *image=(short int *)malloc(sizeof(short int)*ow*oh*raw->frames);
    	int x,y;
        for (int f = 0; f < raw->frames; f++) {
          short *crop = image + f * ow * oh;
          short *orig = raw->image + f * raw->rows * raw->cols;
          for(y=oy;y<(oh+oy);y++)
            {
              for(x=ox;x<(ow+ox);x++)
    		{
                  crop[(y-oy)*ow+(x-ox)] = orig[y*raw->cols+x];
    		}
            }
        }

    	AdvCompr advc(fd, image, ow, oh, raw->frames,
    			raw->uncompFrames, raw->timestamps,NULL,dbgType,acqName,options,framerate);
    	advc.Compress(-1,0);

    	free(image);
    }

	close(fd);
    return true;
}

void Acq::PopulateCroppedRegionalAcquisitionWindow(
    const char* t0InfoFile, 
    const char* regionAcqT0File,
    int ox,
    int oy,
    int ow, 
    int oh,
    unsigned int baseframerate) {

    CalculateNumOfRegions(ow, oh);
    //printf("Num of Y regions : %d\n", num_regions_y);
    //printf("Num of X regions : %d\n", num_regions_x);

    region_acq_start.resize(num_regions_y*num_regions_x);
    memset(&region_acq_start[0], 0, sizeof(float)*num_regions_y*num_regions_x);

    region_acq_end.resize(num_regions_y*num_regions_x);
    memset(&region_acq_end[0], 0, sizeof(float)*num_regions_y*num_regions_x);

    vector<int> region_totals(num_regions_y*num_regions_x);
    memset(&region_totals[0], 0, sizeof(int)*num_regions_y*num_regions_x);

    ifstream t0In(t0InfoFile);
    string line;

    int xmin = ox;
    int ymin = oy;
    int xmax = ox + ow;
    int ymax = oy + oh;
    if(t0In.is_open()) {
        getline(t0In, line); // skipping header line

        int wellIdx;
        int key;
        float t0;
        int wellX, wellY, regionNum, yreg_num, xreg_num;
        while(t0In.good()) {
            getline(t0In, line);
            stringstream(line) >> wellIdx >> key >> t0;

            wellY = wellIdx / w;
            wellX = wellIdx - (wellY * w);
            //printf("WellIdx: %d X:%d Y: %d\n", wellIdx, wellX, wellY);
            if (wellX >= xmin && wellY >= ymin && wellX < xmax && wellY < ymax) {
                //printf("Found a well\n");
                wellX -= xmin;
                wellY -= ymin;
            }
            else {
                continue;
            }
            yreg_num = wellY / y_region_size;
            xreg_num = wellX / x_region_size;
            regionNum = yreg_num * num_regions_x + xreg_num;
            //printf("RegionNum: %d\n", regionNum);
           
            region_acq_start[regionNum] += t0;
            region_totals[regionNum] += 1;                   
        }

        ofstream t0Out(regionAcqT0File);
        if (t0Out.is_open()) {
            t0Out << "Region\tt0(in ms)\ttend(in ms)" << endl;
            int vec_len = region_acq_start.size();
            int acqendtime = ACQENDTIME * 1000000;
            for (int i=0; i<vec_len; ++i) {
                if (region_totals[i] != 0) {
	            region_acq_start[i] = floor(region_acq_start[i] / (float)region_totals[i]);
                    region_acq_start[i] *= baseframerate;
                    region_acq_start[i] = (region_acq_start[i] == 0) ? ACQENDTIME*1000 : region_acq_start[i];
                    int acqstarttime = region_acq_start[i]*1000;
		    t0Out << i << "\t" << acqstarttime << "\t" << acqendtime << endl;
		}
		else {
                    region_acq_start[i] = (region_acq_start[i] == 0) ? ACQENDTIME*1000 : region_acq_start[i];
                    // excluded region (since t0 can't be 0)
	            t0Out << i << "\t7000000" << "\t7000000" << endl;
		}
	    }
         }
         else {
            cout << "Unable to open t0Map file to write t0 values\n";
         }
         t0Out.close();
         t0In.close();
    }
    else {
        cout << "Couldn't open" << t0InfoFile << endl;
    }
}

void Acq::ParseT0File(
    const char* t0InfoFile, 
    const char* regionAcqT0File,
    int ox,
    int oy,
    int ow, 
    int oh,
    unsigned int baseframerate) {

    CalculateNumOfRegions(ow, oh);

    region_acq_start.resize(num_regions_y*num_regions_x);
    memset(&region_acq_start[0], 0, sizeof(float)*num_regions_y*num_regions_x);

    region_acq_end.resize(num_regions_y*num_regions_x);
    memset(&region_acq_end[0], 0, sizeof(float)*num_regions_y*num_regions_x);

    ifstream t0In(t0InfoFile);
    string line;

    if(t0In.is_open()) {
        getline(t0In, line); // skipping header line

        int regId;
        int yreg_num, xreg_num;
        float medt0, mad;
        while(t0In.good()) {
            getline(t0In, line);
            if (!line.empty()) {
                stringstream(line) >> regId >> yreg_num >> xreg_num >> medt0 >> mad;

                yreg_num = yreg_num / y_region_size;
                xreg_num = xreg_num / x_region_size;
                regId = yreg_num * num_regions_x + xreg_num;
           
                if (!isfinite(mad))
                    mad = 0;
                
                region_acq_start[regId] = medt0 - 2*mad;
                //region_acq_start[regId] = medt0 - mad; // try with one times the mad
                //region_acq_end[regId] = medt0*baseframerate + REGACQENDTIME*1000;
                //printf("x_reg: %d yreg: %d regId: %d Median: %.2f mad: %.2f t0: %.2f\n",
                //    xreg_num, yreg_num, regId, medt0, mad, region_acq_start[regId]);
            }
        }

        ofstream t0Out(regionAcqT0File);
        if (t0Out.is_open()) {
            t0Out << "Region\tt0(in ms)\ttend(in ms)" << endl;
            int vec_len = region_acq_start.size();
            for (int i=0; i<vec_len; ++i) {
                region_acq_start[i] *= baseframerate;
                if (!excludedRegions.empty() && excludedRegions[i] != 0) {
                    region_acq_start[i] = ACQENDTIME * 1000;           
                    //region_acq_end[i] = ACQENDTIME * 1000;           
                } 
                int acqstarttime = region_acq_start[i]*1000;
                //int acqendtime = region_acq_end[i]*1000;
                int acqendtime = ACQENDTIME*1000000;
		t0Out << i << "\t" << acqstarttime << "\t" << acqendtime << endl;
	    }
         }
         else {
            cout << "Unable to open t0Map file to write t0 values\n";
         }
         t0Out.close();
         t0In.close();
    }
    else {
        cout << "Couldn't open" << t0InfoFile << endl;
    }
}

void Acq::GenerateExcludeMaskRegions(const char* excludeMaskFile) {
    Mask excludeMask(excludeMaskFile);

    CalculateNumOfRegions(w, h);

    excludedRegions.resize(num_regions_x * num_regions_y);
    memset(&excludedRegions[0], 0, sizeof(int)*num_regions_y*num_regions_x);

      for (int row=0; row<h; ++row) {
          for (int col=0; col<w; ++col) {
              int wellIdx = row*w + col;
              if (excludeMask.Match(wellIdx, MaskExclude)) {
                  int yreg_num = row / y_region_size;
		  int xreg_num = col / x_region_size;
		  int regionNum = yreg_num * num_regions_x + xreg_num;
		  excludedRegions[regionNum] += 1;
	      }
           }
      }
        
      int regionSize = y_region_size * x_region_size;
      int moduloRegSizeinX = regionSize;
      int moduloRegSizeinY = regionSize;
      if (w % x_region_size) {
          moduloRegSizeinX = (w % x_region_size) * y_region_size;
      }
      if (h % y_region_size) {
	  moduloRegSizeinY = (h % y_region_size) * x_region_size;
      }

      ofstream fOut("excludedRegions.txt");
      if (fOut.is_open()) {
          for (unsigned i=0; i<excludedRegions.size(); ++i) {
              if (excludedRegions[i] == regionSize ||
                  excludedRegions[i] == moduloRegSizeinX ||
                  excludedRegions[i] == moduloRegSizeinY) {
                  fOut << i << "\n";
              }
              else {
                  excludedRegions[i] = 0;
              }
          }
      }
      else {
          cout << "Couldn't open file to write excluded regions" << endl;
      }
      fOut.close();

}

int Acq::t0est_start=-1;
int Acq::t0est_end=0;
int Acq::slower_frames=0;

static void ADD_FRAME(FILE *fp, uint32_t num, uint32_t nframes, uint32_t *fnums, uint32_t *uncompFrms, uint32_t  *numEntries, uint32_t *maxFrames)
{
	if(num == 0 || nframes == 0)
		return;

	if(maxFrames)
	{
		if(*maxFrames < (num*nframes))
			num = *maxFrames/nframes;


		*maxFrames -= num*nframes;
	}

	if(num == 0 || nframes == 0)
		return;

	for(uint32_t i=0;i<num;i++)
	{
		(fnums)[(*numEntries)++] = nframes;
		(*uncompFrms) += nframes;
		if(fp)
			fprintf(fp,"%d,",nframes);
	}
//	DTRACE("ADD_FRAME(%d): %d(%d) %d\n",regionNum,nframes,num,*uncompFrms);
}


void Acq::getAverageTrace(double *avg_trace, FILE *fp)
{
	uint64_t avg;
	int last_timestamp=0;
	RawImage *raw = image->raw;
	double avg_trace_vfc[image->raw->frames];
	int timestamp_fnums[image->raw->frames];
	double total=0;
	memset(avg_trace_vfc,0,sizeof(avg_trace_vfc));
	// TODO:  don't average pinned pixels

	for(int frame=0;frame<raw->frames;frame++){
		avg=0;
		for(int row=0;row<raw->rows;row++){
			for(int col=0;col<raw->cols;col++){
				avg += raw->image[frame*raw->rows*raw->cols + row*raw->cols + col];
			}
		}
		avg_trace_vfc[frame] = (double)avg;
		avg_trace_vfc[frame] /= (double)(raw->rows*raw->cols);
	}
	double beginning = ((avg_trace_vfc[0] + avg_trace_vfc[1])/2);
	for(int frame=0;frame<raw->frames;frame++){
		avg_trace_vfc[frame] -= beginning;
	}

	// now, turn this trace into an un-vfc'd trace
	// turn timestamps into frame numbers
	for(int i=0;i<raw->frames;i++){
		timestamp_fnums[i] = (raw->timestamps[i]-last_timestamp + 1) / raw->timestamps[0];
		last_timestamp = raw->timestamps[i];
	}
	if(fp){
		fprintf(fp,  "Timestamps: ");
		int last_ts=0;
		for(int i=0;i<raw->frames;i++){
			fprintf(fp," %5d",((raw->timestamps[i]+10)/raw->timestamps[0]) - last_ts);
			last_ts = ((raw->timestamps[i]+10)/raw->timestamps[0]);
		}
		fprintf(fp,"\nTimeFrames: ");
		for(int i=0;i<raw->frames;i++)
			fprintf(fp," %5d",timestamp_fnums[i]);
		fprintf(fp,"\n\n");
		fprintf(fp,"  Avg_trace_vfc = ");
		for(int i=0;i<raw->frames;i++)
			fprintf(fp," %.0lf",avg_trace_vfc[i]);
		fprintf(fp,"\n");
		fprintf(fp,"  Avg_trace = ");
	}

	// now, transform avg_trace_vfc into avg_trace
	for(int frame=0;frame<raw->uncompFrames;frame++){
	      int interf=raw->interpolatedFrames[frame];
	      float mult = raw->interpolatedMult[frame];

	      float prev=0.0f;
	      float next=0.0f;

	      next = avg_trace_vfc[interf];
	      if ( interf )
	        prev = avg_trace_vfc[interf-1];

	      // interpolate
	      avg_trace[frame] = ( prev-next ) *mult + next;
	      if(fp)
	    	  fprintf(fp," %.0lf",avg_trace[frame]);
	}
	if(fp)
		fprintf(fp,"\n\n");
}
// t0 compress raw (already loaded)
void Acq::doT0Compression()
{
	RawImage *raw = image->raw;
	int doLogging=0;

	if(t0est_start == -1)
	{
		doLogging=1;
		const int inlet_side_t0est_start=10;
		const int outlet_side_t0est_start=22;

		const int inlet_side_t0est_transit_frames=24;  // 1.1 seconds nuc time + 8 frames margin
		const int outlet_side_t0est_transit_frames=30; // an extra 6 frames for the outlet

		printf("doing T0 estimation\n");

		// use acq_0000.dat for the timing of t0est_start and t0est_end
		FILE *fp=NULL;
		double   avg_trace[image->raw->uncompFrames];
		memset(avg_trace,0,sizeof(avg_trace));

		fp = NULL;
		fp = fopen("t0est_start_debug.txt","w");

		getAverageTrace(avg_trace,fp);// first, get the average signal

		// find t0est_start and t0est_end using the average trace
		t0est_start = DCT0Finder(avg_trace,raw->uncompFrames,fp);
		if(fp)
			fclose(fp);
		if(t0est_start > 4)
			t0est_start -= 4; // we want to make sure and capture t0

		t0est_start &= ~(1); // round back to the nearest even number of frames

		if(t0est_start >= 12 ){
			double proportion = t0est_start - inlet_side_t0est_start;
			proportion /= (double)(outlet_side_t0est_start-inlet_side_t0est_start);
			proportion *= (double)(outlet_side_t0est_transit_frames-inlet_side_t0est_transit_frames);
			t0est_end = inlet_side_t0est_transit_frames + proportion + t0est_start;
			slower_frames = (int)proportion;
		}
		else
			t0est_start=-1;

		printf("Completed t0 Analysis:  start=%d end=%d\n",t0est_start,t0est_end);
	}

	if(t0est_start > 0)
	{
		// t0 compress this image
		unsigned int fnums[raw->uncompFrames];
		unsigned int uncompFrms=0;
		unsigned int numEntries=0;
		FILE *fp=NULL;
		ADD_FRAME(fp,1, 				  1,fnums,&uncompFrms,&numEntries,NULL);   //first frame is always at the base acquisition time
		ADD_FRAME(fp,t0est_start/8,       8,fnums,&uncompFrms,&numEntries,NULL);   //skip the front of the signal up to t0
		ADD_FRAME(fp,(t0est_start%8)/4,	  4,fnums,&uncompFrms,&numEntries,NULL);   //skip the front of the signal up to t0
		ADD_FRAME(fp,(t0est_start%4)/2,	  2,fnums,&uncompFrms,&numEntries,NULL);   //skip the front of the signal up to t0

		ADD_FRAME(fp,20+(slower_frames/2),1,fnums,&uncompFrms,&numEntries,NULL);   //capture the signal

		ADD_FRAME(fp,2, 				  2,fnums,&uncompFrms,&numEntries,NULL);   // capture the tail
		ADD_FRAME(fp,2,                   4,fnums,&uncompFrms,&numEntries,NULL);   // capture the tail
		ADD_FRAME(fp,2,                   8,fnums,&uncompFrms,&numEntries,NULL);   // capture the tail

		//now, we have the transform..  time to do the work!

		uint16_t *newImage = (uint16_t *)malloc(numEntries*raw->rows*raw->cols*sizeof(uint16_t));
		int *newTimestamps = (int *)malloc(numEntries*sizeof(int));
		// we now have the new image..
		unsigned int oldframe=0;
		for(unsigned int frame=0;frame<numEntries;frame++){
			// if this is a frame that can be directly copied from one to the other, do that.  otherwise, interpolate
			int tmpFrame=0;
			unsigned int tmpOldFrame=0,last_ts=0,tmp_fnum;
			for(tmpFrame=0;tmpFrame<raw->frames;tmpFrame++){
				tmp_fnum = ((raw->timestamps[tmpFrame]+10)/raw->timestamps[0]) - last_ts;
				last_ts = ((raw->timestamps[tmpFrame]+10)/raw->timestamps[0]);
				if(tmpOldFrame >= oldframe){
					if(tmpOldFrame != oldframe || tmp_fnum != fnums[frame])
						tmpFrame=raw->frames;
					break;
				}
			}
			if(tmpFrame < raw->frames){
				// we should copy this frame directly
				memcpy(&newImage[frame*raw->rows*raw->cols],
						&raw->image[tmpFrame*raw->rows*raw->cols],
						raw->rows*raw->cols*sizeof(newImage[0]));
			}
			else
			{
				for(int row=0;row<raw->rows;row++){
					for(int col=0;col<raw->cols;col++){
						// fill in this pixel
						uint32_t sum=0;
						for(unsigned int ig=0;ig<fnums[frame];ig++){
							uint32_t tmp = (uint32_t)image->GetInterpolatedValue(oldframe+ig,col,row);
	//						if(row==0 && col==0 && doLogging){
	//							fp = fopen("t0est_start_debug.txt","a");
	//							if(fp){
	//								fprintf(fp," %5d",tmp);
	//								fclose(fp);
	//							}
	//						}
							sum += tmp;
						}
						newImage[frame*raw->rows*raw->cols + row*raw->cols + col] = sum/fnums[frame];
					}
				}
			}
			newTimestamps[frame] = ((double)oldframe)*(1.0/15.0)*1000.0; // may need some work!
			oldframe += fnums[frame];
		}

		if(doLogging){
			fp = fopen("t0est_start_debug.txt","a");
			if(fp)
			{
				fprintf(fp,  "\n\nOrig Frames: ");
				int last_ts=0;
				for(int i=0;i<raw->frames;i++){
					fprintf(fp," %d",((raw->timestamps[i]+10)/raw->timestamps[0]) - last_ts);
					last_ts = ((raw->timestamps[i]+10)/raw->timestamps[0]);
				}
				fprintf(fp,"\nNew Frames:  ");
				for(unsigned int i=0;i<numEntries;i++)
					fprintf(fp," %d",fnums[i]);
				fprintf(fp,"\n\n");

				fprintf(fp,"Orig trace:   ");
				for(int frame=0;frame<raw->frames;frame++)
					fprintf(fp," %5d",raw->image[frame*raw->rows*raw->cols]);
				fprintf(fp,"\n");

				fprintf(fp,"New  trace:   ");
				for(unsigned int frame=0;frame<numEntries;frame++)
					fprintf(fp," %5d",newImage[frame*raw->rows*raw->cols]);
				fprintf(fp,"\n");

				fclose(fp);
			}
		}

		free(raw->image);
		raw->image=(short *)newImage;
		free(raw->timestamps);
		raw->timestamps=newTimestamps;
		raw->uncompFrames=uncompFrms;
		raw->frames=numEntries;
		// transform is done!!!
	}
}

