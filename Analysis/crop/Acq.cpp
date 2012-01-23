/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include "crop/Acq.h"
#include "ByteSwapUtils.h"
#include "datahdr.h"
#include "Utils.h"
struct HDR {
	_file_hdr	fileHdr;
	_expmt_hdr_v3	expHdr;
};

#define KEY_0     0x7F
#define KEY_16_1  0xBB

#define PLACEKEY  0xdeadbeef

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
	x_region_size = 128;
	y_region_size = 128;
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

bool Acq::WriteVFC(const char *acqName, int ox, int oy, int ow, int oh)
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
#if 1
	if(ow & 7)
	{
		ow += 7;
		ow  &= ~7;
	}
	if(oh & 7)
	{
		oh += 7;
		oh &= ~7;
	}
#endif

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

//	printf("offset=%d %ld %ld\n",offset,sizeof(expHdr),sizeof(fileHdr));
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
				*ptr++ = *sptr++;
//				image->GetInterpolatedValueAvg4(ptr,rframe,ox+ix,oy+iy,frameCnt);
//				*ptr++ = get_val(ox,iy+oy,rframe,frameCnt);
			}
		}

		if(prev_data)
		{
			if(PrevFrameSubtract(ow*oh,frame_data,prev_data,results_data,&results_len,ow,oh) == 0)
			{
//				printf("pfc worked %ld!!\n",results_len);
				comp = htonl(1);
			}
			else
			{
//				printf("pfc didn't work\n");
			}
		}

		fwrite(&comp, sizeof(comp), 1, fp);
//		bptr = bwrt(&comp, sizeof(comp),bptr);

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

//					printf("val=%x %x %x %x   ptr=%x %x %x %x\n",val[0],val[1],val[2],val[3],ptr[0],ptr[1],ptr[2],ptr[3]);



		rframe += frameCnt;
	}

	printf("\n");

	free(frame_data);
	free(results_data);
	if(prev_data)
		free(prev_data);
	fclose(fp);
	return true;
}


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
//	uint32_t valBins[17] = {0};
//	uint32_t ow = eg.devcols;
//	uint32_t oh = eg.devrows;
	uint32_t num_regions_x = ow/x_region_size;
	uint32_t num_regions_y = oh/y_region_size;
//	uint8_t groupCksum;

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
	return failed;
}



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
