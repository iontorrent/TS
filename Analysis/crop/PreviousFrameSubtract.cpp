/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * PreviousFrameSubtract.cpp
 *
 *  Created on: May 14, 2013
 *      Author: mbeauchemin
 */
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/time.h>
#include <sys/shm.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <unistd.h>
#include "datahdr.h"
#include <stdint.h>
#include <string.h>

//#define USE_GENTLE_SMOOTHING 1
//#define USE_REG_AVG 1
//#define USE_12_BIT_DATA

#define KEY_0     0x7F

#define PLACEKEY  0xdeadbeef


#define CHK(a,v) \
		((v.val[0] < (1<<(a-1))) && (v.val[0] > -(1<<(a-1))) && \
		 (v.val[1] < (1<<(a-1))) && (v.val[1] > -(1<<(a-1))) && \
		 (v.val[2] < (1<<(a-1))) && (v.val[2] > -(1<<(a-1))) && \
		 (v.val[3] < (1<<(a-1))) && (v.val[3] > -(1<<(a-1))) && \
		 (v.val[4] < (1<<(a-1))) && (v.val[4] > -(1<<(a-1))) && \
		 (v.val[5] < (1<<(a-1))) && (v.val[5] > -(1<<(a-1))) && \
		 (v.val[6] < (1<<(a-1))) && (v.val[6] > -(1<<(a-1))) && \
		 (v.val[7] < (1<<(a-1))) && (v.val[7] > -(1<<(a-1))))

#define ADD_OFFSET(a,v) \
		v.val[0] += 1 << (a - 1); \
		v.val[1] += 1 << (a - 1); \
		v.val[2] += 1 << (a - 1); \
		v.val[3] += 1 << (a - 1); \
		v.val[4] += 1 << (a - 1); \
		v.val[5] += 1 << (a - 1); \
		v.val[6] += 1 << (a - 1); \
		v.val[7] += 1 << (a - 1);

#define WRITE_TRANSITION(ns) \
		if((state != ns) || (Tval == KEY_0)) \
		{ \
			state = ns; \
			*resp++ = KEY_0; \
			*resp++ = state; \
			Transitions++; \
		}

#define GETVAL(a,tot,s1,s2,msk)  { \
		{tot += *s1 & msk; a.val[0] = (s2?((*s1++ & msk) - (*s2++ & msk)):(*s1++ & msk);/*if(a.val[0] >= 0)msk |= a.val[0];else msk |= -a.val[0];*/} \
		{tot += *s1 & msk; a.val[1] = ((*s1++ & msk) - (*s2++ & msk));/*if(a.val[1] >= 0)msk |= a.val[1];else msk |= -a.val[1];*/} \
		{tot += *s1 & msk; a.val[2] = ((*s1++ & msk) - (*s2++ & msk));/*if(a.val[2] >= 0)msk |= a.val[2];else msk |= -a.val[2];*/} \
		{tot += *s1 & msk; a.val[3] = ((*s1++ & msk) - (*s2++ & msk));/*if(a.val[3] >= 0)msk |= a.val[3];else msk |= -a.val[3];*/} \
		{tot += *s1 & msk; a.val[4] = ((*s1++ & msk) - (*s2++ & msk));/*if(a.val[4] >= 0)msk |= a.val[4];else msk |= -a.val[4];*/} \
		{tot += *s1 & msk; a.val[5] = ((*s1++ & msk) - (*s2++ & msk));/*if(a.val[5] >= 0)msk |= a.val[5];else msk |= -a.val[5];*/} \
		{tot += *s1 & msk; a.val[6] = ((*s1++ & msk) - (*s2++ & msk));/*if(a.val[6] >= 0)msk |= a.val[6];else msk |= -a.val[6];*/} \
		{tot += *s1 & msk; a.val[7] = ((*s1++ & msk) - (*s2++ & msk));/*if(a.val[7] >= 0)msk |= a.val[7];else msk |= -a.val[7];*/}}




#define CHK_LSB(v) \
		(((v.val[0] % 4) | (v.val[1] % 4) | (v.val[2] % 4) | (v.val[3] % 4) | \
		 (v.val[4] % 4) | (v.val[5] % 4) | (v.val[6] % 4) | (v.val[7] % 4)) == 0)

#define SHIFTRIGHT(v) \
       {v.val[0] /= 4; \
		v.val[1] /= 4; \
		v.val[2] /= 4; \
		v.val[3] /= 4; \
		v.val[4] /= 4; \
		v.val[5] /= 4; \
		v.val[6] /= 4; \
		v.val[7] /= 4;}

#define WRITE_OUT_2BITVALS(v,ns) \
		Tval = ((v.val[0] << 6) & 0xc0) | ((v.val[1] << 4) & 0x30) | ((v.val[2] << 2) & 0xC) | ((v.val[3]) & 0x3);	 \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = ((v.val[4] << 6) & 0xc0) | ((v.val[5] << 4) & 0x30) | ((v.val[6] << 2) & 0xC) | ((v.val[7]) & 0x3);	 \
		resp += 2;

#define WRITE_OUT_3BITVALS(v,ns) \
		Tval = ((v.val[0] << 5) & 0xe0) | ((v.val[1] << 2) & 0x1C) | ((v.val[2] >> 1) & 0x3);	 \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = ((v.val[2] << 7) & 0x80) | ((v.val[3] << 4) & 0x70) | ((v.val[4] << 1) & 0x0E) | ((v.val[5] >> 2) & 0x1) ;	\
		resp[2] = ((v.val[5] << 6) & 0xC0) | ((v.val[6] << 3) & 0x38) | (v.val[7] & 0x7);	\
		resp += 3;

#define WRITE_OUT_4BITVALS(v,ns) \
		Tval = ((v.val[0] << 4) & 0xf0) | (v.val[1] & 0xf);	 \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = ((v.val[2] << 4) & 0xf0) | (v.val[3] & 0xf);	\
		resp[2] = ((v.val[4] << 4) & 0xf0) | (v.val[5] & 0xf);	\
		resp[3] = ((v.val[6] << 4) & 0xf0) | (v.val[7] & 0xf);	\
		resp += 4;

#define WRITE_OUT_5BITVALS(v,ns) \
		Tval = ((v.val[0] << 3) & 0xf8) | ((v.val[1] >> 2) & 7); \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = ((v.val[1] << 6) & 0xC0) | ((v.val[2] << 1) & 0x3E) | ((v.val[3] >> 4) & 0x1); \
		resp[2] = ((v.val[3] << 4) & 0xF0) | ((v.val[4] >> 1) & 0x0F); \
		resp[3] = ((v.val[4] << 7) & 0x80) | ((v.val[5] << 2) & 0x7C) | ((v.val[6] >> 3) & 0x3); \
		resp[4] = ((v.val[6] << 5) & 0xE0) | (v.val[7] & 0x1F); \
		resp += 5;

#define WRITE_OUT_6BITVALS(v,ns) \
		Tval = (((v.val[0]) << 2) & 0xFC) | (((v.val[1]) >> 4) & 0x3); \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = ((v.val[1] << 4) & 0xF0) | (((v.val[2]) >> 2) & 0x0F); \
		resp[2] = ((v.val[2] << 6) & 0xC0) |   (v.val[3] & 0x3F); \
		resp[3] = ((v.val[4] << 2) & 0xFC) | (((v.val[5]) >> 4) & 0x3); \
		resp[4] = ((v.val[5] << 4) & 0xF0) | (((v.val[6]) >> 2) & 0x0F); \
		resp[5] = ((v.val[6] << 6) & 0xC0) |   (v.val[7] & 0x3F); \
		resp += 6;

#define WRITE_OUT_7BITVALS(v,ns) \
		Tval = ((v.val[0] << 1) & 0xFE) | ((v.val[1] >> 6) & 0x1); \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = ((v.val[1] << 2) & 0xFC) | ((v.val[2] >> 5) & 0x03); \
		resp[2] = ((v.val[2] << 3) & 0xF8) | ((v.val[3] >> 4) & 0x07); \
		resp[3] = ((v.val[3] << 4) & 0xF0) | ((v.val[4] >> 3) & 0x0F); \
		resp[4] = ((v.val[4] << 5) & 0xE0) | ((v.val[5] >> 2) & 0x1F); \
		resp[5] = ((v.val[5] << 6) & 0xC0) | ((v.val[6] >> 1) & 0x3F); \
		resp[6] = ((v.val[6] << 7) & 0x80) |  (v.val[7] & 0x7F); \
		resp += 7;

#define WRITE_OUT_8BITVALS(v,ns) \
		Tval = v.val[0]; \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = v.val[1]; \
		resp[2] = v.val[2]; \
		resp[3] = v.val[3]; \
		resp[4] = v.val[4]; \
		resp[5] = v.val[5]; \
		resp[6] = v.val[6]; \
		resp[7] = v.val[7]; \
		resp += 8;

#define WRITE_OUT_16BITVALS(v,ns) \
		Tval = ((v.val[0] & 0xff00) >> 8); \
		WRITE_TRANSITION(ns); \
		resp[0] = Tval; \
		resp[1] = (v.val[0] & 0xff); \
		resp[2] = ((v.val[1] & 0xff00) >> 8); \
		resp[3] = (v.val[1] & 0xff); \
		resp[4] = ((v.val[2] & 0xff00) >> 8); \
		resp[5] = (v.val[2] & 0xff); \
		resp[6] = ((v.val[3] & 0xff00) >> 8); \
		resp[7] = (v.val[3] & 0xff); \
		resp[8] = ((v.val[4] & 0xff00) >> 8); \
		resp[9] = (v.val[4] & 0xff); \
		resp[10] = ((v.val[5] & 0xff00) >> 8); \
		resp[11] = (v.val[5] & 0xff); \
		resp[12] = ((v.val[6] & 0xff00) >> 8); \
		resp[13] = (v.val[6] & 0xff); \
		resp[14] = ((v.val[7] & 0xff00) >> 8); \
		resp[15] = (v.val[7] & 0xff); \
		resp += 16;

#define DETERMINE_COMPR(v,rs,ns) \
	if (CHK(6,v)) \
	{	\
		if (CHK(5,v)) \
		{ \
			if (CHK(4,v)) \
			{ \
				if(CHK(3,v)) \
				   	ns = 3 | rs; \
				else \
    			  ns = 4 | rs; \
			} \
			else \
				ns = 5 | rs; \
		} \
		else \
			ns = 6 | rs; \
	} \
	else \
	{ \
		if (CHK(8,v)) \
		{ \
			if (CHK(7,v)) \
				ns = 7 | rs; \
			else \
				ns = 8 | rs; \
		} \
		else \
			ns = 0xB | rs; \
	}

#define WRITE_OUT_VALS(v,ns) \
		switch (ns & 0xf) \
		{ \
		case 2: \
			WRITE_OUT_2BITVALS(v,ns); \
			break; \
		case 3: \
			WRITE_OUT_3BITVALS(v,ns); \
			break; \
		case 4: \
			WRITE_OUT_4BITVALS(v,ns); \
			break; \
		case 5: \
			WRITE_OUT_5BITVALS(v,ns); \
			break; \
		case 6: \
			WRITE_OUT_6BITVALS(v,ns); \
			break; \
		case 7: \
			WRITE_OUT_7BITVALS(v,ns); \
			break; \
		case 8: \
			WRITE_OUT_8BITVALS(v,ns); \
			break; \
		case 0xB: \
			WRITE_OUT_16BITVALS(v,ns); \
			break; \
		}



typedef struct {
//	uint16_t uvals[8];
	int16_t  val[8];
}s16vals_t;

int PrevFrameSubtract(uint32_t w, uint32_t h, int16_t *framePtr, int16_t *prevFramePtr,
		int16_t *results, uint64_t *out_len, uint32_t *comprType)
{
	uint32_t elems=w*h;
	uint32_t x, y;
//	uint16_t *nextFramePtr = (uint16_t *) framePtr + w*h;
	uint16_t *src_cur = (uint16_t *) framePtr;
	uint16_t *src_prev = (uint16_t *) prevFramePtr;
//	uint16_t *src_next = (uint16_t *) framePtr + w*h;
	int16_t state = 0;
	int16_t newState0 = 0;
//	int16_t newState1 = 0;
	uint32_t RightShift0=0;
//	uint32_t RightShift1=0;
	uint8_t *resp = (uint8_t *) ((uint32_t *) results + 3);
	uint32_t total = 0;
	uint8_t *limit = (uint8_t *) (((int8_t *) results) + elems * 2 - 1000);
	uint32_t *lenp = (uint32_t *) results;
	uint32_t len, Transitions = 0,allignLen;
	uint32_t offset;
	int failed = 0;
	s16vals_t Val0={{0,0,0,0,0,0,0,0}};
//	int16_t Val0Msk=0;
//	s16vals_t Val1={{0,0,0,0,0,0,0,0}};
	uint8_t Tval;
	uint32_t y_reg, x_reg;
	uint32_t nelems_y, nelems_x;
	uint32_t *indexPtr;
	uint32_t valBins[17] =	{ 0 };
	uint32_t localValBins[17] = {0};
	uint32_t localValDbgBins[17][8] = {{0}};
//	uint32_t w = eg.cols;
//	uint32_t h = eg.rows;
	uint32_t x_region_size = 64;
	uint32_t y_region_size = 64;
	uint32_t num_regions_x = w / x_region_size;
	uint32_t num_regions_y = h / y_region_size;
	int16_t limitVal;
#ifndef USE_GENTLE_SMOOTHING
	uint32_t skipped=0;
#endif
	uint32_t i;
	uint16_t cv;
#ifdef USE_12_BIT_DATA
	const uint16_t mask=0x3ffc;
#else
	const uint16_t mask=0x3fff;
#endif
#ifdef USE_REG_AVG
	int64_t RegAvg64;
	int16_t RegAvg=0;
#endif
	//	uint32_t dtotal=0; // dummy total
//	int16_t *ssrc1,*ssrc2;
	//	uint8_t groupCksum;
	//uint32_t total2 = 0;
	//uint32_t total3 = 0;
//	uint32_t src1Val,src2Val;

	if (w % x_region_size)
		num_regions_x++;
	if (h % y_region_size)
		num_regions_y++;

	*(uint32_t *) resp = htonl(PLACEKEY);
	resp += 4;

	indexPtr = (uint32_t *) resp;
	resp += 4 * num_regions_x * num_regions_y; // leave room for indexes

#if 0
	{
		src1 = (uint16_t *) framePtr;
		for(y=0;y<eg.devrows;y++)
		{
			for(x=0;x<eg.devcols;x++)
			{
				total2 += (uint16_t)(*src1++ & 0x3fff);
			}
		}
	}
#endif
	for (y_reg = 0; y_reg < num_regions_y; y_reg++)
	{
		for (x_reg = 0; x_reg < num_regions_x; x_reg++)
		{
			state = 0;
			nelems_x = x_region_size;
			nelems_y = y_region_size;

			if (((x_reg + 1) * x_region_size) > (uint32_t) w)
				nelems_x = w - x_reg * x_region_size;
			if (((y_reg + 1) * y_region_size) > (uint32_t) h)
				nelems_y = h - y_reg * y_region_size;

			indexPtr[y_reg * num_regions_x + x_reg] = htonl(
					(uint32_t) ((uint64_t) resp - (uint64_t) results));

			if (__builtin_expect((resp > limit), 0))
			{
				failed = 1;
				break;
			}
#ifdef USE_REG_AVG
			RegAvg64=0;
			for (y = 0; y < nelems_y; y++)
			{
				offset = (y + (y_reg * y_region_size))*w+(x_reg* x_region_size);

				int16_t *src_curi  = (int16_t *) (framePtr + offset);
				int16_t *src_previ = (int16_t *) (prevFramePtr);
				if(src_previ)
					src_previ += offset;
//				src_next = (uint16_t *) (nextFramePtr + offset);

//				if(x_reg == (num_regions_x-1))
//				{
//					printf("y_reg=%d  (%x/%x %x/%x)\n",y_reg,src_curi[0],src_previ[0],src_curi[1],src_previ[1]);
//				}
				for (x = 0; x < nelems_x; x ++)
				{
//					*src_cur = (2*(*src_cur & 0x3fff) + (*src_prev & 0x3fff) + (*src_next++ & 0x3fff))/4;
					RegAvg64 += ((*src_curi++ & 0x3fff) - (src_previ?(*src_previ++ & 0x3fff):0));

//					for(uint32_t i=0;i<8;i++)
//					{
//						Val0.val[i] = ((*src_cur++ & 0x3fff) - (*src_prev++ & 0x3fff));
//						RegAvg64 += Val0.val[i];
//					}
				}
			}
			RegAvg = (RegAvg64/(nelems_y*nelems_x));
			*resp++ = (RegAvg >> 8) & 0xff; // write out the region average
			*resp++ = (RegAvg)      & 0xff; // write out the region average
//  		printf("%d/%d) (%d/%d) %d\n",x_reg,y_reg,nelems_x,nelems_y,RegAvg);
#endif
			memset(&localValBins,0,sizeof(localValBins));
			for (y = 0; y < nelems_y; y++)
			{
				offset = (y + (y_reg * y_region_size))*w+(x_reg* x_region_size);

				src_cur = (uint16_t *) (framePtr + offset);
				src_prev = (uint16_t *) (prevFramePtr);
				if(src_prev)
					src_prev += offset;

				for (x = 0; x < nelems_x; x += 8)
				{
					if (__builtin_expect((resp > limit), 0))
					{
						failed = 1;
						break;
					}

					for(i=0;i<8;i++)
					{
						cv = *src_cur++ & mask;
//						if ((cv >= 16380) || (cv < 384))
//							cv = 0;// check for pinned
						total += cv;
						Val0.val[i] = cv;
						if(src_prev)
							Val0.val[i] -= (*src_prev++ & mask)
#ifdef USE_REG_AVG
						+ RegAvg;
#endif
						;
#ifdef USE_12_BIT_DATA
						// round the value to the nearest 12-bit value
						if ((Val0.val[i] & 0x3) > 0x1)
						{
							Val0.val[i] = (Val0.val[i] + 0x4) & ~0x3;
							total += 0x4 - (Val0.val[i] & 0x3);
						}
						else
						{
							Val0.val[i] &= ~0x3;
							total -= Val0.val[i] & 0x3;
						}
#endif
					}


					RightShift0 = 0;
					if(CHK_LSB(Val0))
					{
						RightShift0=0x20;
						SHIFTRIGHT(Val0);
					}


					DETERMINE_COMPR(Val0,RightShift0,newState0);

#ifdef USE_GENTLE_SMOOTHING
					uint16_t goodCnt=0;
					int32_t i;
					if(newState0 != (state + 1))
					{
						limitVal = (1<<(state-1));

						// try and smooth it back to the previous bin if possible
						for(i=0;i<8;i++)
						{
							if ((Val0.val[i] >= limitVal))
							{
							}
							else if((Val0.val[i] <= -limitVal))
							{
							}
							else
							{
								goodCnt++;
							}
						}
					}

					if(goodCnt > 4 || newState0 == (state + 1))
					{
						limitVal = (1<<(state-1));
//						int16_t maxLimit = limitVal + (((1<<(state)) - limitVal)/2);

						// try and smooth it back to the previous bin if possible
						for(i=0;i<8;i++)
						{
							if ((Val0.val[i] >= limitVal)/* && (Val0.val[i] < maxLimit)*/)
							{
								// this one is high
								int16_t local_diff =  Val0.val[i] - (limitVal-1);
								Val0.val[i] -= local_diff;
								if(RightShift0==0x20)
									local_diff *=4;
								(src_cur-8)[i] -= local_diff;
								total -= local_diff;
							}
							else if((Val0.val[i] <= -limitVal)/* && (Val0.val[i] > -maxLimit)*/)
							{
								// this one is low
								int16_t local_diff = (1-limitVal) - Val0.val[i];
								Val0.val[i] += local_diff;
								if(RightShift0==0x20)
									local_diff *=4;
								(src_cur-8)[i] += local_diff;
								total += local_diff;
							}
						}
						DETERMINE_COMPR(Val0,RightShift0,newState0);
					}
#else
					if(newState0 == (state - 1) && !skipped)
					{
						skipped=1;
						newState0 = state;
					}
					else
						skipped=0;
#endif
					if((newState0 & 0xf) != 0xB)
					{
						ADD_OFFSET((newState0 & 0xf),Val0);
					}

					WRITE_OUT_VALS(Val0,newState0);


					valBins[state&0xf]++;
					localValBins[state&0xf]++;

					uint32_t missed=0;
					limitVal = ((1<<((newState0&0xf)-1))-1);

					for(i=0;i<8;i++)
					{
						if ((Val0.val[i] < limitVal) && (Val0.val[i] > -limitVal))
							missed++;
					}
					localValDbgBins[state&0xf][missed]++;


				}
			}
//			printf("%d/%d (%d) bins: %04x %04x %04x %04x %04x %04x %04x %04x  T(%d)\n",y_reg,x_reg,RegAvg,localValBins[2],localValBins[3],localValBins[4],localValBins[5],localValBins[6],localValBins[7],localValBins[8],localValBins[0xb],Transitions);
		}
	}
	len = (uint32_t) ((uint64_t) resp - (uint64_t) results);
	allignLen = (len+3) & ~0x3;

	for(;len<allignLen;len++)
		*resp++ = 0; // initialize the whole buffer...

	*lenp++ = htonl(len);
	*lenp++ = htonl(Transitions);
	*lenp++ = htonl(total);
#if 0
	{

		src1 = (uint16_t *) framePtr;
		for(y=0;y<eg.devrows;y++)
		{
			for(x=0;x<eg.devcols;x++)
			{
				total3 += (uint16_t)(*src1++ & 0x3fff);
			}
		}
		if(total != total2)
		{
			DTRACE("Totals don't match.. uhoh  %x %x %x\n",total,total2,total3);
		}
	}
#endif
//	printf("%x bins: 2(%x) 3(%x) 4(%x) 5(%x) 6(%x) 7(%x) 8(%x) 16(%x)  T(%d)\n",(uint32_t)(resp - (uint8_t *)results),valBins[2],valBins[3],valBins[4],valBins[5],valBins[6],valBins[7],valBins[8],valBins[0xb],Transitions);

//	for(i=0;i<8;i++)
//		printf("%x bins: 2(%x) 3(%x) 4(%x) 5(%x) 6(%x) 7(%x) 8(%x) 16(%x)  T(%d)\n",(uint32_t)(resp - (uint8_t *)results),localValDbgBins[2][i],localValDbgBins[3][i],localValDbgBins[4][i],localValDbgBins[5][i],localValDbgBins[6][i],localValDbgBins[7][i],localValDbgBins[8][i],localValDbgBins[0xb][i],Transitions);

	*out_len = len; // return the number of bytes to the caller.
	if(failed)
		*comprType = 0;
	else
#ifdef USE_REG_AVG
	    *comprType = 3;
#else
	    *comprType = 2;
#endif

	return failed;
}




