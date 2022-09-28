/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */
#include "RawSpatial.h"
#include "deInterlace.h"
#include "CorrNoiseCorrector.h"
#include "ComparatorNoiseCorrector.h"
#include "ChipIdDecoder.h"
#include "AdvCompr.h"
#include "Mask.h"


RawSpatial::RawSpatial(QWidget *parent): SpatialPlot(parent)
{
}

void RawSpatial::SetOption(QString txt, int state)
{
    qDebug() << __PRETTY_FUNCTION__ << ": " << txt << state;

    if(txt == "Zero traces")
        zeroState = state;
    else if(txt == "Rmv Row Noise")
        RowNoiseState = state;
    else if(txt == "Rmv Col Noise")
        ColNoiseState = state;
    else if(txt == "Gain Correct")
        gainCorrState = state;
    else if(txt == "NeighborSubtract")
        neighborState = state;
    else if(txt == "RefSubtract")
        RefState = state;
    else if(txt == "EmptySubtract")
        EmptySubState = state;
    else if(txt == "Column Flicker")
        ColFlState = state;
    else if(txt == "Adv Compression")
        AdvcState = state;
    else if(txt == "Show Masked")
        MaskReads = state;
    else if(txt == "StdDev")
        stdState = state;
    else if(txt == "noPCA")
        noPCAState = state;
    else if(txt == "T0Correct")
        t0CorrectState = state;
    else if(txt == "Navigate")
    	display_blocks = state;
    else if(txt == "Histogram")
    	display_histogram = state;
    else if(txt == "AverageSub")
    	AverageSubState = state;
    else if(txt == "BitsNeeded")
    	display_bitsNeeded = state;

    render();
}

void RawSpatial::getBlockSize(QString dir, int &blockRows, int &blockCols)
{
	dir += "/explog_final.txt";
	QByteArray qb = dir.toLatin1();

	char buf[1024];
	int len=0;
	FILE *fp = fopen(qb.data(),"r");
	if(fp){
		// the file exists
		while ((len = fread(buf,1,sizeof(buf)-1,fp)) > 0){
			buf[len]=0; // null terminate the string
			char *ptr = strstr(buf,"Rows:");
			if(ptr){
				ptr += 5;
				sscanf(ptr,"%d",&blockRows);
				blockRows /= 8;
			}
			ptr = strstr(buf,"Columns:");
			if(ptr){
				ptr += 8;
				sscanf(ptr,"%d",&blockCols);
				blockCols /= 12;
			}
		}
		fclose(fp);
	}

    qDebug() << __PRETTY_FUNCTION__ << ": blockRows=" << blockRows << " blockCols=" << blockCols;
}

void RawSpatial::DoubleClick(int x, int y)
{
	qDebug() << __PRETTY_FUNCTION__ << ": " << x << " - " << y;

	if (display_blocks && cols > 0 && rows > 0) {
		// is thumbnail in the directory name right now?
		int blockRows = 0, blockCols = 0;

		// open this block.
		int xIdx = x / (cols / 12);
		int yIdx = y / (rows / 8);

		QString newFname = fname;
		// replace the directory name with the new directory..
		int pos = 0;
		int nextLastPos = 0;
		int lastPos = 0;
		qDebug() << __PRETTY_FUNCTION__ << ": newFname=" << newFname;
		while ((pos = newFname.indexOf(QString('/'), pos + 1)) > 0) {
			nextLastPos = lastPos;
			lastPos = pos;
		}
		if (nextLastPos > 0 && lastPos > 0) {
			QString fN = newFname.right(newFname.size() - lastPos - 1);
			QString dirN = newFname.left(nextLastPos);
			qDebug() << __PRETTY_FUNCTION__ << ": lastPos=" << lastPos
					<< "nextLastPos=" << nextLastPos << " dirN=" << dirN;
			if ((fname.indexOf("/thumbnail/", 0)) > 0) {
				// switch to the clicked on block data

				fN.replace("_spa", "");
				qDebug() << __PRETTY_FUNCTION__ << ": fN=" << fN;

				getBlockSize(dirN, blockRows, blockCols);

				if (blockRows > 0 && blockCols > 0) {
					QString newFn = dirN + "/X"
							+ QString::number(xIdx * blockCols) + "_Y"
							+ QString::number(yIdx * blockRows) + "/" + fN;

					qDebug() << __PRETTY_FUNCTION__ << ": old=" << fname
							<< " new=" << newFn;

					emit fileNameChanged(newFn);
					//fname = newFn; // load this file..
					//render();

					// change the check box name to thumb
				}
			} else {
				// switch to the thumbnail data
				fN.push_back("_spa");
				QString newFn = dirN + "/thumbnail/" + fN;
				qDebug() << __PRETTY_FUNCTION__ << ": old=" << fname
						<< " new=" << newFn;

				emit fileNameChanged(newFn);
				//fname = newFn; // load this file..
				//render();
			}
		}
	}
	else {
		SpatialPlot::DoubleClick(x,y);
	}
}

//void RawSpatial::fileNameChanged(QString fname)
//{
//}

void RawSpatial::doConvert(int &loading)
{

    if((rowNoiseRemoved && !RowNoiseState) ||
            (colNoiseRemoved && !ColNoiseState) ||
            (gainCorrApplied && !gainCorrState) ||
            (neighborSubApplied && !neighborState) ||
            (RefSubApplied && !RefState) ||
            (ColFlApplied && !ColFlState) ||
            (EmptySubApplied && !EmptySubState) ||
            (AdvcApplied && !AdvcState) ||
            (stdApplied && !stdState) ||
            (noPCAApplied != noPCAState) ||
            (lastcurDisplayFlow != curDisplayFlow) ||
			(AverageSubApplied != AverageSubState)){
        // re-load the image.0...+1
        last_fname.clear();
    }

    if(fname == ""){
        free(out);
        out=NULL;
        last_fname = _fname;
        traces_len=0;
    }
    if (last_fname != _fname){
        // load a new dat file
        char fnameBuf[4096];
        loading=1;
        lastcurDisplayFlow=curDisplayFlow;
        printf("triggered %s %s\n",last_fname.toLatin1().data(),_fname.toLatin1().data());
        last_fname = _fname;
        if(out)
            free(out);
        out=NULL;

        if(timestamps)
            free(timestamps);
        timestamps=NULL;

        if(gainImage)
            free(gainImage);
        gainImage=NULL;


        QByteArray ba = fname.toLatin1();
        strcpy(fnameBuf,ba.data());
        char * fn = fnameBuf;
        char *sptr=fn;

        if(noPCAState && !strstr(fn,"_noPCA")){
                char tmpName[4096];
                struct stat statBuf;
                sprintf(tmpName,"%s_noPCA",fn);
                if(stat(tmpName,&statBuf) == 0){
                    strcpy(fn,tmpName); // only copy if the file exists
                }
        }
        noPCAApplied=noPCAState;

        if(curDisplayFlow >= 0){
            // change the raw dat name to this flow
            char *ptr = sptr;
            char *lptr=NULL;
            while((ptr = strstr(ptr,"acq_"))){
                lptr = ptr;
                ptr++;
            }
            if(lptr){
                lptr += strlen("acq_");
                char saveChar=lptr[4];
                sprintf(lptr,"%04d",lastcurDisplayFlow);
                lptr[4]=saveChar;
            }

        }
        printf("about to call deInterlace with %s\n",fn);
        deInterlace_c (fn, &out, &timestamps,
                       &rows, &cols, &frames, &uncompFrames,
                       0, 0,
                       0, 0, 0, 0, 0, &imageState );
        printf("\ndone\n\n");
        rowNoiseRemoved=0;
        colNoiseRemoved=0;
        gainCorrApplied=0;
        neighborSubApplied=0;
        RefSubApplied=0;
        EmptySubApplied=0;
        ColFlApplied=0;
        AdvcApplied=0;
        stdApplied=0;
        AverageSubApplied=0;
        if(endX==0){
            startX=0;
            startY=0;
            endX=cols;
            endY=rows;
        }
        for(uint i=0;i<(uint)frames && i < (sizeof(traces_ts)/sizeof(traces_ts[0]));i++)
            traces_ts[i]=((double)timestamps[i])/1000.0f;
        traces_len=frames;
        loading=0;

        if(out){
            if(mMask){
                free(mMask);
                mMask=NULL;
            }
            mMask = (char *)malloc(rows*cols);
            memset(mMask,0,rows*cols);
        }

    }

    if(out){

        // determine t0's for each 100x100 block
        FindT0s();

        //we have loaded a image into memory..
        if(!rowNoiseRemoved && RowNoiseState){
            ChipIdDecoder::SetGlobalChipId("p2.2.2");
            CorrNoiseCorrector rnc;
            /*double RowNoise = */rnc.CorrectCorrNoise(out, rows, cols, frames, 1, 0, 0,0,-1,0);
            rowNoiseRemoved=1;
        }
        if(!colNoiseRemoved && ColNoiseState){
            ChipIdDecoder::SetGlobalChipId("p2.2.2");
            CorrNoiseCorrector rnc;
            /*double ColNoise = */rnc.CorrectCorrNoise(out, rows, cols, frames, 2, 0, 0,0,-1,0);
            colNoiseRemoved=1;
        }
        if(!gainCorrApplied && gainCorrState && gainVals){
            GainCorrect(out,rows,cols,frames);
            gainCorrApplied=1;
        }
        if(!neighborSubApplied && neighborState){
            // do neighbor subtraction
            NeighborSubtract(out, rows, cols, frames,NULL,0);
            neighborSubApplied=1;
        }
        if(!RefSubApplied && RefState && refMask){
            // do neighbor subtraction
            NeighborSubtract(out, rows, cols, frames, refMask,1);
            RefSubApplied=1;
        }
        if(!EmptySubApplied && EmptySubState && refMask){
            // do neighbor subtraction
            NeighborSubtract(out, rows, cols, frames, refMask,0);
            EmptySubApplied=1;
        }
        if(!ColFlApplied && ColFlState){
            // do column flicker correction
            ComparatorNoiseCorrector cnc;
            cnc.CorrectComparatorNoise(out,rows,cols,frames,NULL,0,1,0);
            ColFlApplied=1;
        }
        if(!AdvcApplied && AdvcState){
            ChipIdDecoder::SetGlobalChipId("p2.2.2");
            AdvCompr advc(-1/*ExtFd*/, out, cols, rows, frames, uncompFrames,
                    timestamps, timestamps, 1, (char *)"",(char *)"",15,1);
            advc.Compress(-1,0,0,0);
            AdvcApplied=1;
        }
        if(!stdApplied && stdState){
            // take spatial standard deviation of the pixel offsets
            TakeStdDev();
            stdApplied=1;
        }
        if(!AverageSubApplied && AverageSubState){
        	// subtract the average signa
        	SubAverage();
        	AverageSubApplied=1;
        }
        CalculateBitsNeededData();

        fflush(stdout);
        UpdateTraceData();
    }
}

void RawSpatial::CalculateBitsNeededData()
{
	int frameStride=rows*cols;

	memset(bitsNeededData,0,sizeof(bitsNeededData));

	for(int frm=1;frm<frames;frm++){
		short *imgPtr=&out[frm*frameStride];
		short *pimgPtr=&out[(frm-1)*frameStride];
		for(int idx=0;idx<frameStride;idx++){
			short val = imgPtr[idx]-pimgPtr[idx];
			if(val <0)
				val=-val;
			for(int bit=15;bit >=0;bit--){
				if((1<<bit) & val){
					bitsNeededData[bit]++;
					break;
				}
			}
		}
	}

}


extern int32_t DCT0Finder(float *trace, uint32_t trace_length, FILE *logfp);

void RawSpatial::FindT0s()
{
    // make a map of the t0 values for each 100x100 region
    t0Height=(rows+99)/100; // round up
    t0Width =(cols+99)/100; // round up

    if(t0Map)
        free(t0Map);
    t0Map = (int32_t *)malloc(t0Height*t0Width*sizeof(t0Map[0]));

    for(int th=0;th<t0Height;th++){
        for(int tw=0;tw<t0Width;tw++){

            float avgTrace[frames];
            for(int frame=0;frame<frames;frame++){
                // get an average for this block..
                uint64_t sum=0;
                uint64_t sumCnt=0;
                for(int y=th*100;y<((th+1)*100) && y < rows;y++){
                    for(int x=tw*100;x<((tw+1)*100) && x < cols;x++){
                        sum += out[frame*cols*rows + y*cols + x];
                        sumCnt++;
                    }
                }
                avgTrace[frame] = (float)sum/(float)sumCnt;
            }

            // now, find t0 for this average trace..
            t0Map[th*t0Width + tw] = DCT0Finder(avgTrace,frames,NULL);
        }
    }
}


void RawSpatial::TakeStdDev()
{
    int spanW=2;
    int spanH=2;

    // make a mask of the first frame..
    for(int r=0;r<rows;r++){
        int starty=r-spanH;
        int endy=r+spanH;

        if(starty<=0)
            starty=1;
        if(endy >= rows)
            endy=rows;
        for(int c=0;c<cols;c++){
            int startx=c-spanW;
            int endx=c+spanW;

            if(startx <= 0)
                startx=1;
            if(endx >= cols)
                endx=cols;

            float localSum=0;
            int localSumCnt=0;
            for(int ry=starty;ry<endy;ry++){
                for(int rx=startx;rx<endx;rx++){
                    float ls = (out[ry*cols+rx-1] - out[ry*cols+rx]);
                    localSum += ls*ls;
                    localSumCnt++;
                }
            }
            localSum /= (float)localSumCnt;
            out[rows*cols + r*cols+c]  = sqrt(localSum);
        }
    }
}

float RawSpatial::Get_Data(int frame, int y, int x)
{
    float rc = 0;
    //float t0=0;
    if(out){
        if(t0CorrectState){
            int th = y/100;
            int tw = x/100;
            int t0 = t0Map[th*t0Width + tw];
            frame += t0;
            if(frame >= frames)
                frame = frames-1;
        }
        rc = out[frame*rows*cols + y*cols + x];
        if(zeroState)
            rc -= out[0*rows*cols + y*cols + x];
        if(MaskReads && mMask && mMask[y*cols+x])
            rc = 0; // only display non-masked items
        if(display_blocks){
        	int ym = y%(rows/8);
        	int xm = x%(cols/12);
        	if(ym < 3 || xm < 3)
        		rc = 0;
        }
    }
    return rc;
}

//uint16_t *RawSpatial::ReadLSRowImage(char *fname, int rows, int cols, short **outPtr)
//{

//    return 0;
//}

#define NOT_PINNED(a) 1
inline bool ReferenceWell ( uint16_t maskVal )
{
  // is this well a valid reference coming out of beadfind?
  bool isReference = ((maskVal & MaskReference) == MaskReference);
  bool isIgnoreOrAmbig = ((maskVal & MaskIgnore) == MaskIgnore);
  bool isPinned = ((maskVal & MaskPinned) == MaskPinned);

  return ( isReference && !isPinned && !isIgnoreOrAmbig );
}
inline bool EmptyWell ( uint16_t maskVal )
{
  // is this well a valid reference coming out of beadfind?
  bool isEmpty = ((maskVal & MaskEmpty) == MaskEmpty);
  bool isReference = ((maskVal & MaskReference) == MaskReference);
  //bool isIgnoreOrAmbig = ((maskVal & MaskIgnore) == MaskIgnore);
  //bool isPinned = ((maskVal & MaskPinned) == MaskPinned);

  return ( isEmpty && isReference); //!isPinned && !isIgnoreOrAmbig );
}

void RawSpatial::SubAverage()
{
	int frameStride=rows*cols;
	for(int frm=0;frm<frames;frm++){
		// get average value for this frame
		short *imptr=&out[frm*frameStride];
		uint64_t avg=0;
		for(int idx=0;idx<frameStride;idx++){
			avg += imptr[idx];
	    }
		avg /= (uint64_t)frameStride;
		short avgs=(short)avg;

		// subtract the average value
		for(int idx=0;idx<frameStride;idx++){
			imptr[idx] -= avgs;
	    }
	}
}
void RawSpatial::GainCorrect(short int *raw, int h, int w, int npts)
{
    for(int frame=0;frame<npts;frame++){
        for(int y=0;y<h;y++){
            int idx = frame*h*w + y*w + 0;
            for(int x=0;x<w;x++,idx++){
                float tmp = raw[idx];
                tmp *= gainVals[y*w+x];
                raw[idx] = tmp;
            }
        }
    }
}


void RawSpatial::NeighborSubtract(short int *raw, int h, int w, int npts, uint16_t *mask, int use_ref )
{
    int frameStride=w*h;
    int span=40;
    int64_t *lsum = (int64_t *)malloc(sizeof(int64_t)*h*w);
    int *lsumNum = (int *)malloc(sizeof(int)*w*h);
    int thumbnail=(w==1200 && h==800);

    memset(lsumNum,0,sizeof(int)*w*h);
    // build the matrix
    for(int frame=0;frame<npts;frame++){
        short int *rptr = &raw[frame*frameStride];
        for(int y=0;y<h;y++){
            int x=0;
            if((mask==NULL) || (use_ref && ReferenceWell(mask[y*w+x])) ||
                    (!use_ref && EmptyWell(mask[y*w+x]))){
                if(y>0){
                    lsum[y*w+x]   =lsum[(y-1)*w+x] + rptr[y*w+x];
                    lsumNum[y*w+x]=lsumNum[(y-1)*w+x] + 1;
                }
                else{
                    lsum[y*w+x]=rptr[x];
                    lsumNum[y*w+x]=1;
                }
            }
            else{
                if(y>0){
                    lsum[y*w+x]=lsum[(y-1)*w+x];
                    lsumNum[y*w+x]=lsumNum[(y-1)*w+x];
                }
                else
                    lsum[y*w+x]=0;
            }
            for(x=1;x<w;x++){
                if((mask==NULL) || (use_ref && ReferenceWell(mask[y*w+x])) ||
                        (!use_ref && EmptyWell(mask[y*w+x]))){
                    if(y>0){
                        lsum[y*w+x]=lsum[(y-1)*w+x] + lsum[y*w+x-1] - lsum[(y-1)*w+x-1] + rptr[y*w+x];
                        lsumNum[y*w+x]=lsumNum[(y-1)*w+x] + lsumNum[y*w+x-1] - lsumNum[(y-1)*w+x-1] + 1;
                    }
                    else{
                        lsum[y*w+x]= lsum[y*w+x-1] + rptr[y*w+x];
                        lsumNum[y*w+x]= lsumNum[y*w+x-1] + 1;
                    }				}
                else{
                    if(y>0){
                        lsum[y*w+x]=lsum[(y-1)*w+x] + lsum[y*w+x-1] - lsum[(y-1)*w+x-1];
                        lsumNum[y*w+x]=lsumNum[(y-1)*w+x] + lsumNum[y*w+x-1] - lsumNum[(y-1)*w+x-1];
                    }
                    else{
                        lsum[y*w+x]= lsum[y*w+x-1];
                        lsumNum[y*w+x]= lsumNum[y*w+x-1];
                    }
                }
            }
        }
        // save the mean for this frame
//        if(saved_mean)
//            saved_mean[frame]=(float)lsum[(h-1)*w+(w-1)]/(float)(w*h);
        // now, neighbor subtract each well
        for(int y=0;y<h;y++){
            for(int x=0;x<w;x++){
                if((mask != NULL) && ((mask[y*w+x] & MaskPinned) == MaskPinned)){
//					rptr[y*w+x] = 8192;
//					printf("pinned %d/%d\n",y,x);
                    rptr[y*w+x] = 0;
                    continue;
                }
                int start_x=x-span;
                int end_x  =x+span;
                int start_y=y-span;
                int end_y  =y+span;

                if(thumbnail){
                    // were limited to this 100x100 block
                    start_x=x-(x%100);
                    end_x=start_x + 100;
                    start_y=y-(y%100);
                    end_y=start_y+100;
                }

                if(start_x < 0)
                    start_x=0;
                if(end_x >= w)
                    end_x = w-1;
                if(start_y < 0)
                    start_y=0;
                if(end_y >= h)
                    end_y=h-1;
                int end=end_y*w+end_x;
                int start=start_y*w+start_x;
                int diag1=end_y*w+start_x;
                int diag2=start_y*w+end_x;

//				if(y < 12 && x < 12)
//					printf(" (%d/%d = %d)",(int)(lsum[end_y][end_x]-lsum[start_y][start_x]),((end_y-start_y+1)*(end_x-start_x+1)),
//							(int)((lsum[end_y][end_x]-lsum[start_y][start_x])/((end_y-start_y+1)*(end_x-start_x+1)) + 8192));
                int64_t avg = lsum[end]+lsum[start]-lsum[diag1]-lsum[diag2];
                int64_t num=lsumNum[end]+lsumNum[start]-lsumNum[diag1]-lsumNum[diag2];
                if(num){
                    avg /= num;
                    rptr[y*w+x] = rptr[y*w+x] - (short int)avg/* + 8192*/;
                }
                else{
//                    printf("Problems here %d/%d\n",y,x);
                }
            }
        }
    }
    free(lsum);
    free(lsumNum);
}




