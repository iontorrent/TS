/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * RowSumCorrector.cpp
 *
 *  Created on: Aug 13, 2015
 *      Author: ionadmin
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "RowSumCorrector.h"


void RemoveDCOffset(short int *avgs, int rows, int frames)
{
	for(int row=0;row<rows;row++){
		float avg=0;
		for(int frame=0;frame<frames;frame++){
			avg += avgs[frame*rows+row];
		}
		avg /= frames;
		for(int frame=0;frame<frames;frame++){
			avgs[frame*rows+row] -= avg;
		}
	}
}

void Smooth(short int *avgs, int rows, int frames, int span)
{
	int bufLen=sizeof(short int)*rows*frames;
	short int *tmp = (short int *)malloc(bufLen);
	for(int frame=0;frame<frames;frame++){
		for(int row=0;row<rows;row++){
			float avg=0;
			int start_row = row-span;
			int end_row = row+span;
			if(start_row<0)
				start_row=0;
			if(end_row>=rows)
				end_row=rows-1;
			for(int trow=start_row;trow<end_row;trow++){
				avg += avgs[frame*rows+row];
			}
			avg /= end_row-start_row;
			tmp[frame*rows+row]=avg;
		}
	}
	memcpy(avgs,tmp,bufLen);
	free(tmp);
}

void Smooth_simple(short int *avgs, int elems, int span)
{
	short int *tmp = (short int *)malloc(sizeof(short int)*elems);
	for (int i=0;i<elems;i++){
		if(span > i)
			span = i;
		if((i+span) > elems)
			span = elems-i;
		int start_elem = i-span;
		int end_elem = i+span+1;
		if(start_elem<0)
			start_elem=0;
		if(end_elem>elems)
			end_elem=elems;
		float avg = 0;
		for(int trow=start_elem;trow<end_elem;trow++){
			avg += avgs[trow];
		}
		avg /= end_elem-start_elem;
		tmp[i]=avg;
	}
	memcpy(avgs,tmp,elems*sizeof(short int));
	free(tmp);
}

short int *GetFineGrained(short int *avgs, int rows, int frames)
{
	// average bottom and top together as they are sampled at the same time
	// the output should be 1/2 the number of rows
	int out_rows = rows/2;
	short int *rc = (short int *)malloc(sizeof(short int)*out_rows*frames);
	for(int frame=0;frame<frames;frame++){
		for(int row=0;row<out_rows;row++){
			rc[frame*out_rows+row] = (avgs[frame*rows+row] + avgs[frame*rows+(rows-1-row)])/2; // average together top and bottom
		}
	}
	return rc;
}

short int *Get_Mean_trace_nodcoffset(short int *avgs, int rows, int frames)
{
	short int *rc = (short int *)malloc(sizeof(short int)*frames*rows/2);
	for(int frame=0;frame<frames;frame++){
		// get this frame's average
		float avg = 0;
		for(int row=0;row<rows;row++){
			avg += avgs[frame*rows+row];
		}
		avg /= (float)rows;
		rc[frame*rows/2] = avg;
	}

	// now, remove the dc offset

	float avg=0;
	for(int frame=0;frame<3;frame++){
		avg += rc[frame*rows/2];
	}
	avg /= 3;

	// finally, fill in the rest of the array

	for(int frame=0;frame<frames;frame++){
		short int tavg = rc[frame*rows/2] - avg;
		for(int row=0;row<rows/2;row++){
			rc[frame*rows/2 + row] = tavg;
		}
	}

	return rc;
}

short int *CreateRowSumCorrection(short int *image, int rows, int cols, int frames)
{
	// from Donohues matlab code...
	int frameStride=rows*cols;

	int avgsLen=sizeof(short int)*rows*frames;
	short int *avgs = (short int *)malloc(avgsLen);
	short int *corr = (short int *)malloc(avgsLen);
	memset(corr,0xff,avgsLen);
	for(int frame=0;frame<frames;frame++){
		for(int row=0;row<rows;row++){
			float avg = 0;
			for(int col=0;col<cols;col++){
				avg += image[frame*frameStride+row*cols+col];
			}
			avg /= (float)cols;
			avgs[frame*rows+row] = (short int)avg;
		}
	}

//	// the first four and last four rows are reference pixels
//	for(int row=0;row<4;row++){
//		for(int frame=0;frame<frames;frame++){
//			avgs[frame*rows+row] = avgs[frame*rows+4];
//		}
//	}
//
//	for(int row=rows-5;row<rows;row++){
//		for(int frame=0;frame<frames;frame++){
//			avgs[frame*rows+row] = avgs[frame*rows+rows-6];
//		}
//	}

	// get fine-grained time history
	short int *fine = GetFineGrained(avgs,rows,frames);  // should be 400xframes

	Smooth_simple(fine,frames*rows/2,2);

	short int *mean_trc = Get_Mean_trace_nodcoffset(avgs,rows,frames); // replicate out to 400xframes
	Smooth_simple(mean_trc,frames*rows/2,3*rows/2);

	for(int i=0;i<frames*rows/2;i++)
		fine[i] -= mean_trc[i];

	float dt = 1.0f/(15*rows*2);

	//% values are good for P2 and Proton, need to figure out values
	//% for other options
	float Ce,Cfb,Ca,Rref;
#ifndef BB_DC
	Ce = 10e-6; Cfb = 0.17e-6; Ca = 0.50e-6; Rref = 75e3;
#else
	if(eg.ChipInfo.ChipMajorRev < 0x20){
		if(RAPTOR_SYSTEM()){
			Ce = 15e-6; Cfb = 0.0625e-6; Ca = 0.44e-6; Rref = 50e3;
		}
		else{
			Ce = 10e-6; Cfb = 0.0625e-6; Ca = 0.44e-6; Rref = 75e3;
		}
	}
	else{
		if(RAPTOR_SYSTEM()){
			Ce = 15e-6; Cfb = 0.17e-6; Ca = 0.50e-6; Rref = 50e3;
		}
		else{
			Ce = 10e-6; Cfb = 0.17e-6; Ca = 0.50e-6; Rref = 75e3;
		}
	}
#endif
	//% coefficients for va component
	float coefa1 = (Ce*Cfb*Rref)/(dt*Cfb + dt*Ce + Ce*Cfb*Rref);
	float coefa2 = -(Ce*Ca*Rref + dt*Ca)/(dt*Cfb + dt*Ce + Ce*Cfb*Rref);
	float coefa3 = Ce*Ca*Rref/(dt*Cfb + dt*Ce + Ce*Cfb*Rref);

	//% coefficients fo vrow component
	float coefr1 = Rref*(Ca+Cfb)/(dt*(Ca+Cfb+Ce)+Rref*(Ca+Cfb));
	float coefr2 = (dt+Rref)*Cfb/(dt*(Ca+Cfb+Ce)+Rref*(Ca+Cfb));
	float coefr3 = -(Rref*Cfb)/(dt*(Ca+Cfb+Ce)+Rref*(Ca+Cfb));

	float *vfrows = (float *)malloc(sizeof(float)*(rows/2)*frames);
	memset(vfrows,0,sizeof(float)*(rows/2)*frames);
	float *vfavgs = (float *)malloc(sizeof(float)*(rows/2)*frames);
	memset(vfavgs,0,sizeof(float)*(rows/2)*frames);

	vfrows[0] = fine[0];
	vfavgs[0] = mean_trc[0];

	for(int n=1;n<(rows/2)*frames;n++){
		vfrows[n] = coefr1 * vfrows[n-1] +
				coefr2 * fine[n] +
				coefr3 * fine[n-1];

		vfavgs[n] = coefa1 * vfavgs[n-1] +
				coefa2 * mean_trc[n] +
				coefa3 * mean_trc[n-1];
	}

	// now, pack vfrows back into avgs
	for(int frame=0;frame<frames;frame++){
		for(int row=0;row<rows/2;row++){
			corr[frame*rows+row] = corr[frame*rows+rows-1-row] = vfrows[frame*rows/2 + row] + vfavgs[frame*rows/2 + row];
		}
	}
	// remove the dc offset from the correction
	for(int row=0;row<rows;row++){
		float avg=0;
		for(int frame=0;frame<4;frame++){
			avg += corr[frame*rows+row];
		}
		avg /=4;
		for(int frame=0;frame<frames;frame++){
			corr[frame*rows+row] -= avg;
		}
	}

#if 0
	//debug
	for(int crow=0;crow<2;crow++){
		int row=0;
		if(crow==0)
			row=399;
		if(crow==1)
			row=400;
		printf("%d) corr: ",row);
		for(int frame=0;frame<frames;frame++){
			printf(" %d",corr[frame*rows+row]);
		}
		printf("\n");
		printf("   vfrows: ");
		for(int frame=0;frame<frames;frame++){
			printf(" %.0f",vfrows[frame*rows/2+row]);
		}
		printf("\n");
		printf("   vfavgs: ");
		for(int frame=0;frame<frames;frame++){
			printf(" %.0f",vfavgs[frame*rows/2+row]);
		}
		printf("\n");
		printf("   fine: ");
		for(int frame=0;frame<frames;frame++){
			printf(" %d",fine[frame*rows/2+row]);
		}
		printf("\n");

		printf("\n");

	}
#endif
	free(vfrows);
	free(vfavgs);
	free(mean_trc);
	free(fine);
	free(avgs);

	return corr;
}



int WriteRowSumCorrection(char *fname, short int *corr, int rows, int frames)
{
	// save the correction
	int rc=0;
	int corrLen=rows*frames*sizeof(short int);
	int fd = open(fname,O_WRONLY | O_CREAT,0644);
	if(fd >= 0)
	{
		struct rowcorr_header hdr;
		hdr.magic = ROWCORR_MAGIC_VALUE;
		hdr.version = 1;
		hdr.rows = rows;
		hdr.frames = frames;
		hdr.framerate = 15;
		hdr.pixsInSums = 1;
		hdr.sumsPerRow = 1;

		if(write(fd,&hdr,sizeof(hdr)) < (int)sizeof(hdr)){
			printf("%s: Failed to write out the header\n",__FUNCTION__);
		}

		if(write(fd,corr,corrLen) < (int)corrLen){
			printf("%s: Failed to write out the data\n",__FUNCTION__);
		}
		else{
//					printf("data=");
//					for(int i=0;i<raw->frames;i++){
//						printf("%d ",avgs[i*raw->rows]);
//					}
//					printf("\n");
		}


		close(fd);
		rc = 1;
	}

	return rc;
}

#ifndef BB_DC
int GenerateRowCorrFile(char *srcdir, char *name)
{
	Image loader;
	char fname[2048];
	int rc=0;

	// open the spatial thumbnail.
	// it is interpolated to 15fps.
	sprintf(fname,"%s/%s_spa",srcdir,name);
    if ( loader.LoadRaw ( fname, 0, 1, false, false ) ) {
    	const RawImage *raw = loader.GetImage();
    	int frameStride=raw->rows*raw->cols;
    	// create the row averages

    	short int *corr = CreateRowSumCorrection(raw->image,raw->rows,raw->cols,raw->frames); // turn it into a correction

    	if(corr){
    		char fnBuf[2048];
    		sprintf(fnBuf,"%s_rowcorr",fname);
        	rc = WriteRowSumCorrection(fnBuf,corr,raw->rows,raw->frames);
    		free(corr);
    	}
    }
    return rc;
}

// reads the rowcorr file into memory.
//  picks out the rows that are applicable to the image being corrected
//  finally, time-interpolates into the same vfc array.
short int *ReadRowCorrection(char *srcdir, char *name, Image *img)
{
	short int *rc = NULL;
	const RawImage *raw = img->GetImage();
	char fnBuf[2048];
	sprintf(fnBuf,"%s/../thumbnail/%s_spa_rowcorr",srcdir,name);

	int fd = open(fnBuf,O_RDONLY, 0644);

	if(fd >= 0)
	{

		struct rowcorr_header hdr;
		if(read(fd,&hdr,sizeof(hdr)) != (int)sizeof(hdr)){
			printf("%s: Failed to read the header\n",__FUNCTION__);
		}
		if(hdr.magic == ROWCORR_MAGIC_VALUE && hdr.version == 1){
			printf("sucesfully opened %s\n",fnBuf);
			int rows = hdr.rows;
			int frames = hdr.frames;
			int avgsLen = sizeof(short int)*rows*frames;
			int nframes[200];
			int nts=0;
			short int *avgs = (short int *)malloc(avgsLen);

			if(read(fd,avgs,avgsLen) < (int)avgsLen){
				printf("%s: Failed to read the data\n",__FUNCTION__);
			}

			// figure out the time compression of this block
			{
				int prevTs=0;
				for(int pt=0;pt<raw->frames;pt++){
					nframes[pt] = (raw->timestamps[pt]-prevTs + 2)/raw->timestamps[0];
					prevTs = raw->timestamps[pt];
				}
				nts = raw->frames;
			}

			// turn the averages into the right time compression and number of rows
			{
				int avgsoLen = sizeof(unsigned short int)*raw->rows*nts;
				int whole_chip_size=raw->rows;//*8; // 8 vertical blocks in full chip
				int y_position = raw->chip_offset_y;
				int start_row = y_position*hdr.rows/whole_chip_size;
				int end_row = (y_position+raw->rows)*hdr.rows/whole_chip_size;
				int applicable_rows=end_row-start_row;


				short int *avgso = (short int *)malloc(avgsoLen);
				memset(avgso,0,avgsoLen);
				int start_frame=0;
				for(int pt=0;pt<nts;pt++){
					for(int row=0;row<raw->rows;row++){
						for(int nf=0;nf<nframes[pt];nf++){
							avgso[pt*raw->rows+row] += avgs[(start_frame+nf)*hdr.rows/*+start_row*/+(row/**applicable_rows/raw->rows*/)];
						}
						avgso[pt*raw->rows+row] /= nframes[pt];
					}
					start_frame += nframes[pt];
					rc = (short int *)avgso;
				}
			}
			free(avgs);
		}
		else{
			printf("bad magic key %s \n",fnBuf);
		}
		close(fd);
	}
	else{
		printf("Failed to opened %s\n",fnBuf);
	}
	return rc;
}

int CorrectRowAverages(char *srcdir, char *name, Image *img)
{
	int rc = 0;
	const RawImage *raw = img->GetImage();
	int frameStride = raw->rows*raw->cols;
//	// generate the row averages file
//	rc = GenerateRowAverages(srcdir,name);

	// open the row averages file
	img->SetOffsetFromChipOrigin(srcdir);

	short int *cor = ReadRowCorrection(srcdir,name,img);
	if(cor){

		// correct the acq array
		for(int frame=0;frame<raw->frames;frame++){
			for(int row=0;row<raw->rows;row++){
				for(int col=0;col<raw->cols;col++){
					raw->image[frame*frameStride + row*raw->cols + col] -= cor[frame*raw->rows+row];
				}
			}
		}
		free(cor);
	}
	return rc;
}
#endif

