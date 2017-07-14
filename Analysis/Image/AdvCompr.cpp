/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include <malloc.h>
#include <string.h>
#include "AdvCompr.h"
#ifndef WIN32
#include "PCACompression.h"
#ifdef PCASPLINE
#include "PcaSpline.h"
#endif
#include "Utils.h"
#endif

//#define SMOOTH_BEGINNING_OF_TRACE 1
//#define POST_CLEAR_FRONT_PORCH 1
//#define PCA_CONVERGE_BASIS_VECTORS 1

//#define SCALE_TRACES 1
//#define CLEAR_BEGINNING_OF_TRACE 1
//#define SHORTEN_P2_FRONT_PORCH 1

//#define OUTPUT_BASIS_VECTORS 1
//#define OUTPUT_VECTOR_COEFFS 1
//#define PCA_UNCOMP_MEAN_SMOOTHING 1
//#define USE_LESS_BASIS_VECTS 3
//#define OUTPUT_VECTOR_COEFFS 1
//#define CLEAR_MEAN 1

#ifndef RDR_MAX_REGIONS
#define RDR_MAX_REGIONS 96
#endif


#define XTALK_FRAC (0.30f)

#define XTALK_FRAC_EARLY 1
#define GAIN_CORR_IN_ADVTEST 1

#define NOT_PINNED(idx) (mMask == NULL || (mMask[(idx)]))

#define V8_SUM(a) (a.A[0]+a.A[1]+a.A[2]+a.A[3]+a.A[4]+a.A[5]+a.A[6]+a.A[7])
#define SIGS_ACC(idx,frame) (mCorr_sigs[((idx)*(npts) + (frame))])
#define NOISE_ACC(idx,frame) (mCorr_noise[((idx)*(npts) + (frame))])


// limits on the gain term we allow
#define ADV_COMPR_MIN_GAIN 0.65f
#define ADV_COMPR_MAX_GAIN 1.35f
#define ADV_COMPR_DEFAULT_GAIN 1.0f

// pinning defaults - do these need to change for different chips?
#define ADV_COMPR_PIN_HIGH 16380.0f
#define ADV_COMPR_PIN_LOW 5.0f

// high and low initial values 
#define ADV_COMPR_MAX_VAL 100000.0f
#define ADV_COMPR_MIN_VAL -100000.0f
float *AdvCompr::gainCorrection[ADVC_MAX_REGIONS] = {NULL};
uint32_t AdvCompr::gainCorrectionSize[ADVC_MAX_REGIONS] = {0};
char *AdvCompr::threadMem[ADVC_MAX_REGIONS] = {NULL};
uint32_t AdvCompr::threadMemLen[ADVC_MAX_REGIONS] = {0};

// constructor
AdvCompr::AdvCompr(FILE_HANDLE _fd, short int *_raw, int _w, int _h,
		int _nImgPts, int _nUncompImgPts, int *_timestamps_compr,
		int *_timestamps_uncompr, int _testUncompr, char *_fname,
		char *_options, int _frameRate, int RemoveRowNoise)
{
	ThreadNum=-1;
	GblMemPtr=NULL;
	GblMemLen=0;
    do_cnc = false;//true;
//    do_rnc = true;
	w = _w;
	h = _h;
	npts = _nImgPts;
	fd = _fd;
	raw = _raw;
	nUncompImgPts = _nUncompImgPts;
	timestamps_compr = _timestamps_compr;
	timestamps_uncompr = _timestamps_uncompr;
	testType = _testUncompr;
	nPcaBasisVectors=0;
	nPcaFakeBasisVectors=0;
	nSplineBasisVectors=0;
	nPcaSplineBasisVectors=0;
	nDfcCoeff=0;
	frameRate=_frameRate;
	CorrectRowNoise = RemoveRowNoise;

	nBasisVectors=0;

	SplineStrategy[0]=0;
	inp_options=_options;

	if (_fname)
		strncpy(fname, _fname, sizeof(fname));
	else
		fname[0] = 0;

	FREE_STRUCTURES(0);

	memset(&hdr, 0, sizeof(hdr));
	ntrcs = 0;
	t0est = 0;
	pinnedTrcs = 0;
	memset(&timing, 0, sizeof(timing));

	doGain = 0;

	cblocks = w / BLK_SZ_X;
	rblocks = h / BLK_SZ_Y;
	if (w % BLK_SZ_X)
		cblocks++;
	if (h % BLK_SZ_Y)
		rblocks++;
}

// deconstructor
AdvCompr::~AdvCompr()
{
}

#ifndef WIN32


void AdvCompr::NeighborSubtract(short int *raw, int h, int w, int npts, float *saved_mean)
{
	int frameStride=w*h;
	int span=7;
	int64_t *lsum = (int64_t *)malloc(sizeof(int64_t)*h*w);
	int *lsumNum = (int *)malloc(sizeof(int)*w*h);

	memset(lsumNum,0,sizeof(int)*w*h);
	// build the matrix
	for(int frame=0;frame<npts;frame++){
		short int *rptr = &raw[frame*frameStride];
		for(int y=0;y<h;y++){
			int x=0;
			if(NOT_PINNED(y*w+h)){
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
				if(NOT_PINNED(y*w+h)){
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
		if(saved_mean)
			saved_mean[frame]=(float)lsum[(h-1)*w+(w-1)]/(float)(w*h);
		// now, neighbor subtract each well
		for(int y=0;y<h;y++){
			for(int x=0;x<w;x++){
				if(NOT_PINNED(y*w+x)){
//					rptr[y*w+x] = 8192;
//					printf("pinned %d/%d\n",y,x);
					rptr[y*w+x] = 0;
					continue;
				}
				int start_x=x-span;
				int end_x  =x+span;
				int start_y=y-span;
				int end_y  =y+span;
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
					rptr[y*w+x] = rptr[y*w+x] - (short int)avg + 8192;
				}
				else{
					printf("Problems here %d/%d\n",y,x);
				}
			}
		}
	}
	free(lsum);
	free(lsumNum);
}

float *AdvCompr::AllocateGainCorrection(uint32_t region, uint32_t len)
{
	float *rc = NULL;

	if(region < ADVC_MAX_REGIONS){
		if(gainCorrection[region] == NULL || gainCorrectionSize[region] != len)
		{
//			AdvComprPrintf("%s: Allocating gainCorrection[%d] (%p)\n",__FUNCTION__,region,gainCorrection[region]);
			if(gainCorrection[region])
				free(gainCorrection[region]);
			gainCorrection[region] = (float *) memalign(VEC8F_SIZE_B,len);
			gainCorrectionSize[region] = len;
			for (unsigned int i = 0; i < (len/sizeof(float)); i++)
				gainCorrection[region][i] = ADV_COMPR_DEFAULT_GAIN;
		}
		rc = gainCorrection[region];
	}
	return rc;
}

// compress a memory image into a file
//   threadNum is used to keep static memory and avoid malloc calls for each invocation
int AdvCompr::Compress(int _threadNum, uint32_t region, int targetAvg, int _timeTransform)
{
//	int num_pinned;
	double start = AdvCTimer();
//	double start1, start2;
	uint32_t len = w * h * sizeof(float);

	timeTransform = _timeTransform;
	ThreadNum = _threadNum;
	ntrcs = ntrcsL = BLK_SZ;
	if (ntrcsL % (VEC8_SIZE * 4))
		ntrcsL = (ntrcs + (VEC8_SIZE * 4)) & (~((VEC8_SIZE * 4) - 1));

	if(region  < ADVC_MAX_REGIONS)
	{
		gainCorr = AllocateGainCorrection(region,len);
	}
	else
	{
		gainCorr = (float *) memalign(VEC8F_SIZE_B,
				w * h * sizeof(float));
	}

#if 0
	#ifndef BB_DC
	if (fname && strstr(fname, "beadfind_pre_0003")){
//		doGain = 1;
		ComputeGain_FullChip();
		// compute gain separately from compression code for now

	}
	else
    #endif
#endif
		doGain = 0;

	ParseOptions(inp_options);

	nBasisVectors=nPcaBasisVectors+nPcaFakeBasisVectors+
			nPcaSplineBasisVectors+nSplineBasisVectors+nDfcCoeff;

	ALLOC_STRUCTURES(nBasisVectors);

	// write out header
	memset(&FileHdrM, 0, sizeof(FileHdrM));
	FileHdrM.signature = BYTE_SWAP_4(0xdeadbeef);
	FileHdrM.struct_version = BYTE_SWAP_4(4);
	FileHdrM.header_size = BYTE_SWAP_4(sizeof(_expmt_hdr_v4));
	FileHdrM.data_size = 0;
	if (fd >= 0)
		Write(fd, &FileHdrM, sizeof(FileHdrM));

	memset(&FileHdrV4, 0, sizeof(FileHdrV4));
	FileHdrV4.rows = BYTE_SWAP_2(h);
	FileHdrV4.cols = BYTE_SWAP_2(w);
	FileHdrV4.x_region_size = BYTE_SWAP_2(BLK_SZ_X); // not really used..
	FileHdrV4.y_region_size = BYTE_SWAP_2(BLK_SZ_Y); // not really used..

	npts_newfr = npts;
	nUncompImgPts_newfr = nUncompImgPts;
	TimeTransform(timestamps_compr,timestamps_newfr,frameRate,npts_newfr,nUncompImgPts_newfr);
	FileHdrV4.frames_in_file = BYTE_SWAP_2(npts_newfr);
	FileHdrV4.uncomp_frames_in_file = BYTE_SWAP_2(nUncompImgPts_newfr);

	FileHdrV4.interlaceType = BYTE_SWAP_2(7); // PCA
	if(timeTransform)
		FileHdrV4.sample_rate = BYTE_SWAP_4(15);
	else
		FileHdrV4.sample_rate = BYTE_SWAP_4(frameRate);

	if (fd >= 0)
		Write(fd, &FileHdrV4, sizeof(FileHdrV4));

	// write out timestamps
	if (fd >= 0){
		Write(fd, timestamps_newfr, sizeof(timestamps_newfr[0]) * npts_newfr);
	}

	double start_xtc=AdvCTimer();
	int total_iter = 0;

#ifdef XTALK_FRAC_EARLY
	findPinned();
	if(CorrectRowNoise)
		ApplyGain_FullChip_xtalkcorr_sumRows(XTALK_FRAC, w, h, npts, (short unsigned int *)raw);
	else
		ApplyGain_FullChip_xtalkcorr(XTALK_FRAC, w, h, npts, (short unsigned int *)raw);

#else
	ApplyGain_FullChip(0);
#endif
	timing.xtalk += AdvCTimer() - start_xtc;

	if(CorrectRowNoise){
        if(region >= RDR_MAX_REGIONS)
			NNSubtractComparatorSigs_tn(100);
		else
			NNSubtractComparatorSigs(500);
	}

#if 0
#ifndef PYEXT
//    if(do_rnc)
    {
		double start2 = AdvCTimer();
        CorrNoiseCorrector rnc;
		rnc.CorrectCorrNoise(raw, h, w, npts, 1, (region >= 96), 0,0,ThreadNum,targetAvg);
		timing.rcn += AdvCTimer() - start2;
    }
#endif
#endif



	for (int rblk = 0; rblk < rblocks; rblk++)
	{
		int rstart = rblk * BLK_SZ_Y;
		int rend = rstart + BLK_SZ_Y;

		if (rend > h)
			rend = h;

#ifndef BB_DC
		AdvComprPrintf(".");
#endif
		fflush(stdout);

		for (int cblk = 0; cblk < cblocks; cblk++)
		{
			int cstart = cblk * BLK_SZ_X;
			int cend = cstart + BLK_SZ_X;

			if (cend > w)
				cend = w;

			memset(&hdr, 0, sizeof(hdr));
			hdr.ver = 0;
			hdr.key = ADVCOMPR_KEY;
			hdr.nBasisVec = 0;
			hdr.npts = npts;

			hdr.rstart = rstart;
			hdr.rend = rend;
			hdr.cstart = cstart;
			hdr.cend = cend;

			ntrcs = (hdr.rend - hdr.rstart) * (hdr.cend - hdr.cstart);

//			if(rblk==0)
//				printf("%s: (%d-%d)(%d-%d)\n",__FUNCTION__,hdr.rstart,hdr.rend,hdr.cstart,hdr.cend);
			double startClr = AdvCTimer();
			CLEAR_STRUCTURES();
			timing.clear += AdvCTimer() - startClr;


			if (doGain)
			{
				ComputeGain();
				CLEAR_STRUCTURES();
			}
			PopulateTraceBlock_ComputeMean();
			t0est = findt0(&t0estIdx);
//			printf("t0est=%d t0estIdx=%d\n",t0est,t0estIdx);

			SetMeanOfFramesToZero_SubMean(npts); // subtracts mean as well

			AddDcOffsetToMeanTrace();

			if (testType)
				doTestComprPre();

			hdr.nBasisVec = nBasisVectors;

		    CreateSampleMatrix();

			total_iter += CompressBlock();

			if (fd >= 0)
				WriteTraceBlock();

			if (testType)
				doTestComprPost();

		}
	}


	FREE_STRUCTURES(1);

	timing.overall = AdvCTimer() - start;
//	DumpTiming("Compress");

	return total_iter;
}

void AdvCompr::ApplyGain_FullChip(float targetAvg)
{
	int frameStride=w*h;

	//int itrc = 0;
	v8f  *lgcV = (v8f *) &gainCorr[0];
	v8f_u tmpV;
	v8s_u tmpVS;
	v8f   sumV=LD_VEC8F(0);

	for (int itrc = 0; itrc < frameStride; itrc+=8,lgcV++)
	{
		{ // load the first frame as our subtract point
			short int *rtrc = &raw[0 * frameStride + itrc];
			LD_VEC8S_CVT_VEC8F(rtrc, tmpV);
			sumV = tmpV.V;
		}

		for (int pt = 0; pt < npts; pt++)
		{
			short int *rtrc = &raw[pt * frameStride + itrc];

			LD_VEC8S_CVT_VEC8F(rtrc, tmpV);
			tmpV.V = sumV + ((tmpV.V - sumV) * *lgcV);
			//  put it back
			CVT_VEC8F_VEC8S(tmpVS,tmpV);
			*(v8s *)rtrc = tmpVS.V;
		}
	}
}

void AdvCompr::ComputeGain_FullChip()
{
	int frameStride=w*h;

	//int itrc = 0;
	v8f_u tmpV;
	v8f_u tmpV1;
	v8f_u tmpV2;
	//v8s_u tmpVS;
	float mean_trc[npts];

	printf("in %s\n",__FUNCTION__);

//	printf("%s: firtst_trc = ",__FUNCTION__);
//	for(int pt=0;pt<npts;pt++){
//		printf(" %d",raw[pt*frameStride]);
//	}

	for (int pt = 0; pt < npts; pt++)
	{
		short int *rtrc = &raw[pt * frameStride];
		v8f_u sum = {LD_VEC8F(0)};
		for (int itrc = 0; itrc < frameStride; itrc+=8,rtrc+=8)
		{
			LD_VEC8S_CVT_VEC8F(rtrc, tmpV);
			sum.V += tmpV.V; // for avg trace
		}
		mean_trc[pt]=0;
		for (int l = 0; l < 8; l++)
			mean_trc[pt] += sum.A[l];
//		printf("mean_trc[%d] = %f / %d\n",pt,mean_trc[pt],frameStride);
		mean_trc[pt]/=(float)frameStride;

	}

//	printf("%s: mean_trc = ",__FUNCTION__);
//	for(int pt=0;pt<npts;pt++){
//		printf(" %.0f",mean_trc[pt]);
//	}

	// we now have mean_trc
	// get mean_trc height
	float mean_lowest_point = 16384.0f;
	float mean_highest_point = 0.0f;
	int lowest_pt=0;
	int highest_pt=0;
	// use the last point as the height
	highest_pt=npts-1;
	lowest_pt=1;
	mean_highest_point=mean_trc[highest_pt];
	mean_lowest_point = mean_trc[lowest_pt];

	float mean_height=mean_highest_point - mean_lowest_point;

//	if(highest_pt < lowest_pt){
//		//reverse the points to make a positive gain.
//		int tmp=highest_pt;
//		highest_pt=lowest_pt;
//		lowest_pt=tmp;
//	}


	v8f mean_heightV=LD_VEC8F(mean_height);
	v8f *lgcV = (v8f *) &gainCorr[0];
	for (int itrc = 0; itrc < frameStride; itrc+=8,lgcV++)
	{
		short int *rtrc1 = &raw[highest_pt*frameStride + itrc];
		short int *rtrc2 = &raw[lowest_pt*frameStride + itrc];

		LD_VEC8S_CVT_VEC8F(rtrc1, tmpV1);
		LD_VEC8S_CVT_VEC8F(rtrc2, tmpV2);
		*lgcV = mean_heightV/(tmpV1.V-tmpV2.V);
	}
//	printf("%s: first row= ",__FUNCTION__);
//	for(int pt=0;pt<10;pt++){
//		printf(" %.02f",gainCorr[pt]);
//	}
}

void AdvCompr::AddDcOffsetToMeanTrace()
{
	//TODO: figure out if the average needs to be offset
	for (int i = 0; i < npts; i++)
		mean_trc[i] += 8192; // add back some dc offset
}

void AdvCompr::ParseOptions(char *options)
{
	// pcaSpline must be last option for now
//	const char *dflt_options="pcaReal=5,pcaFake=4,pcaSpline=0,SplineSt=no-knots";
	const char *pcaRealOpt="pcaReal=";     // real pca basis vectors
	const char *pcaFakeOpt="pcaFake=";     // made up pca basis vectors
#ifdef PCASPLINE
	const char *pcaSplineOpt="pcaSpline="; // real pca basis vectors from spline code
	const char *SplineStOpt="SplineSt=";   // spline knot basis vector strategy
	const char *SplineTikOpt="SplineTik="; // spline knot basis vector strategy
	const char *SplineOrdOpt="SplineOrd=";
#endif
	const char *CorrectRowNoiseOpt="CorrectRowNoise";
//	const char *DfcCoeffOpt="DfcCoeff=";  // number of dfc coefficients

	// set up the defaults
	strcpy(SplineStrategy,"no-knots");
	nPcaBasisVectors=6;
	nPcaFakeBasisVectors=0;
#ifdef PCASPLINE
	nPcaSplineBasisVectors=0;
	nSplineBasisVectors=0;
	spline_order = -1;
	tikhonov_k = -1;
#endif
	//	CorrectRowNoise = 1;
	if(options == NULL)
		return;

	char *ptr = strstr(options,pcaRealOpt);
	if(ptr)
	{
		ptr += strlen(pcaRealOpt);
//		printf("%s: pcaRealOpt=%s\n",__FUNCTION__,ptr);
		sscanf(ptr,"%d,",&nPcaBasisVectors);
	}
	ptr = strstr(options,pcaFakeOpt);
	if(ptr)
	{
		ptr += strlen(pcaFakeOpt);
		sscanf(ptr,"%d,",&nPcaFakeBasisVectors);
	}
#ifdef PCASPLINE
	ptr = strstr(options,pcaSplineOpt);
	if(ptr)
	{
		ptr += strlen(pcaSplineOpt);
		sscanf(ptr,"%d,",&nPcaSplineBasisVectors);
	}
	ptr = strstr(options, SplineTikOpt);
	if (ptr) { 
	  ptr += strlen(SplineTikOpt);
	  sscanf(ptr,"%f,",&tikhonov_k);
	}
	ptr = strstr(options, SplineOrdOpt);
	if (ptr) { 
	  ptr += strlen(SplineOrdOpt);
	  sscanf(ptr,"%d,",&spline_order);
	}
	ptr = strstr(options,SplineStOpt);
	if(ptr)
	{
		ptr += strlen(SplineStOpt);
		strncpy(SplineStrategy,ptr,sizeof(SplineStrategy)-1);
		SplineStrategy[sizeof(SplineStrategy)-1]=0;
		PcaSpline compressor(nPcaSplineBasisVectors, SplineStrategy);
		if (spline_order < 0) {
		  spline_order = compressor.GetOrder(); // use class default if unset.
		}
		if (tikhonov_k < 0) {
		  tikhonov_k = compressor.GetTikhonov();
		}
		compressor.SetTikhonov(tikhonov_k);
		compressor.SetOrder(spline_order);
		
		PcaSplineRegion compressed;
		compressed.n_wells = ntrcs;
		compressed.n_frames = npts;
		nSplineBasisVectors = compressor.NumBasisVectors(nPcaSplineBasisVectors, compressed.n_frames,
							  compressor.GetOrder(), SplineStrategy);
                nSplineBasisVectors = nSplineBasisVectors - nPcaSplineBasisVectors;
        AdvComprPrintf("Using PcaSpline: %d basis vec %.2f tikhonov for %d pca, %s knots order %d\n",
			       nSplineBasisVectors + nPcaSplineBasisVectors, tikhonov_k, nPcaSplineBasisVectors, SplineStrategy, spline_order);
			       
	}
#endif
	if(strstr(options,CorrectRowNoiseOpt))
	{
		CorrectRowNoise = 0;
	}


#ifndef BB_DC
	AdvComprPrintf("%s: Options - %s\n",__FUNCTION__,options);
	AdvComprPrintf("%s: nPcaBasisVectors-%d nPcaFakeBasisVectors-%d nPcaSplineBasisVectors-%d nSplineBasisVectors-%d nDfcCoeff-%d\n",
			__FUNCTION__,nPcaBasisVectors,nPcaFakeBasisVectors,nPcaSplineBasisVectors,nSplineBasisVectors,nDfcCoeff);
#endif
}

#ifndef BB_DC
int AdvComprTest(const char *_fname, Image *image, char *options, bool do_cnc)
{
	char tstName[1024], tstName2[1024], *lptr, *last_lptr = NULL, *fnlptr;
	RawImage *raw = image->raw;
	int dbgType=1;
	static int doneOnce=0;
	strcpy(tstName2, _fname);
	lptr = (char *) tstName2;

	if(options && options[0] >= '0' && options[0] <= '9')
	  dbgType = (int)options[0]-(int)'0';
        if (dbgType == 0) {
          return 0;
        }
	AdvComprPrintf("%s: dbgType=%d options=%s %s\n", __FUNCTION__, dbgType, options, _fname);

	while ((lptr = strstr(lptr, "/")))
	{
		if (lptr)
			lptr++;
		last_lptr = lptr;
	}
	if (last_lptr)
		fnlptr = last_lptr;
	else
		fnlptr = (char *) tstName2;

	lptr = strstr(fnlptr, ".");
	if (lptr)
		*lptr = 0;

	sprintf(tstName, "%s_testPCA.dat", tstName2);

#ifdef GAIN_CORR_IN_ADVTEST
	if(!doneOnce/* && (strstr(_fname,"beadfind_pre_0003") == NULL)*/)
	{
		doneOnce=1;
//		static const char *gainName="beadfind_pre_0003.dat";
		static const char *gainName="Gain.lsr";
		// compute the gain image first...
		char tstName3[1024];
		last_lptr=NULL;
		strcpy(tstName3,_fname);
		lptr = (char *)tstName3;
		while((lptr = strstr(lptr,"/")))
		{
			if(lptr)
			lptr++;
			last_lptr = lptr;
		}
		if(last_lptr)
		strcpy(last_lptr,gainName);
		else
		strcpy(tstName3, gainName);
		printf("Loading gain mask from %s\n",tstName3);
		if(AdvCompr::ReadGain(0, raw->cols, raw->rows, tstName3) == 0){
			// load from beadfind
		}

//		Image img;
//
//		img.LoadRaw(tstName3); // this will call AdvCompress for gain calculation
//		AdvCompr advc(-1, img.raw->image, img.raw->cols, img.raw->rows,
//				img.raw->frames, img.raw->uncompFrames, img.raw->timestamps, img.raw->timestamps, dbgType, tstName3,options);
//		advc.Compress(0,0,1024);
//		img.Cleanup();
	}
#endif
	//dbgType 1  - replace raw with pca/unpca'd data
	//dbgType 2  - save residuals
	//dbgType 3  - save smoothed residuals
	//dbgType 4  - timeshift
	//dbgType 5  - just do gain correct and cnc.  then write the data back to raw
	//dbgType 6  - save corrected and zero'd traces
        //dbgType 7  - write out file and read back in again

//	int framerate=1000/image->raw->timestamps[0];
//	printf("input_framerate=%d\n",framerate);

	int fd = -1;
	if (/*dbgType != 1 && */(dbgType > 6))
		fd = open(tstName, O_WRONLY | O_CREAT | O_TRUNC, 0666);

//	if(fd >= 0 || dbgType == 1 || (dbgType >= 4 && dbgType <= 5))
	{
		if (fd >= 0)
			AdvComprPrintf("opened %s for writing fd=%d\n", tstName, fd);
		{
			int frameRate=1000/raw->timestamps[0];
			AdvCompr advc(fd, raw->image, raw->cols, raw->rows,
					raw->frames, raw->uncompFrames, raw->timestamps, raw->timestamps, dbgType, tstName,options,frameRate);
                        advc.SetDoCNC(do_cnc);
			advc.Compress(-1,0);
		}
		if (fd >= 0) {
                  close(fd);
                  fd = -1;
                }
		if (dbgType > 7) {
                  // double check that we cleared everything out
                  memset(raw->image, 0, sizeof(short) * raw->rows * raw->cols);
		  fd = open(tstName, O_RDONLY);
		  assert(fd >= 0);
		}
		if (fd >= 0)
		{
			AdvComprPrintf("Debug Uncompressing %s\n", tstName);
			{
				AdvCompr advc(fd, raw->image, raw->cols, raw->rows,
						raw->frames, raw->uncompFrames, raw->timestamps, raw->timestamps, dbgType, tstName,options);
				advc.UnCompress();
			}
		}
		if ((dbgType != 2) && (dbgType != 3))
			((RawImage *) raw)->imageState = IMAGESTATE_GainCorrected
					| IMAGESTATE_ComparatorCorrected; // don't do it again
	}
	return 0;
}
#endif

int AdvCompr::PopulateTraceBlock_ComputeMean()
{
	int itrc = 0;
	v8f_u tmpV;
	int frameStride = w * h;
	short int *rtrc;
	int r, c, pt;
	double start = AdvCTimer();
	float meanPt;
	v8f_u meanV;
//	v8f dymmy = {};

	ntrcs = ntrcsL = (hdr.rend - hdr.rstart) * (hdr.cend - hdr.cstart);
	if (ntrcs % (VEC8_SIZE * 4))
		ntrcsL = (ntrcs + (VEC8_SIZE * 4)) & (~((VEC8_SIZE * 4) - 1));

	// copy trcs_state from mMask
	{
		v8i *pinnedV = (v8i *) &trcs_state[0];
		for (r = hdr.rstart; r < hdr.rend; r++)	{
			c = hdr.cstart;
			v8i *maskV = (v8i *) &mMask[r*w+c];
			for (; c < hdr.cend;c += VEC8_SIZE){
				*pinnedV++ = *maskV++;
			}
		}
	}

	for (pt = 1; pt < npts; pt++)
	{
		meanV.V=LD_VEC8F(0);
		itrc = 0;
		for (r = hdr.rstart; r < hdr.rend; r++)
		{
			c = hdr.cstart;
			rtrc = &raw[pt * frameStride + r * w + c];
			v8f rowNoiseV;
			if(CorrectRowNoise)
				rowNoiseV = LD_VEC8F(NOISE_ACC(r,pt));// row noise for this point..
			else
				rowNoiseV = LD_VEC8F(0); // no row noise

			v8f *tPtrV = (v8f *) &TRCS_ACCESS(itrc,pt);
			for (; c < hdr.cend; c += VEC8_SIZE, rtrc += VEC8_SIZE, tPtrV++)
			{
				LD_VEC8S_CVT_VEC8F(rtrc, tmpV);

				tmpV.V -= rowNoiseV;

//				tmpV.V = LD_VEC8F(mCorr_sigs[r*npts+pt]); //DEBUG

				meanV.V += tmpV.V; //sum all wells for mean trace
				*tPtrV = tmpV.V;
				itrc += VEC8_SIZE;
			}
		}
		meanPt=0;
		for(int k=0;k<VEC8_SIZE;k++)
			meanPt += meanV.A[k];
		mean_trc[pt] = meanPt/(float)ntrcs;
	}

	// copy the second point to the first
	mean_trc[0] = mean_trc[1];
	v8f *rtrcV0 = (v8f *) &TRCS_ACCESS(0,0);
	v8f *rtrcV1 = (v8f *) &TRCS_ACCESS(0,1);

	for(int idx=0;idx<ntrcs/8;idx++){
		*rtrcV0++ = *rtrcV1++;
	}

	// now, remove the dc component from the mean trace
	float meanAvg=0;
	for (pt = 0; pt < npts; pt++)
		meanAvg += mean_trc[pt];
	meanAvg /= (float)npts;
	for (pt = 0; pt < npts; pt++)
		mean_trc[pt] -= meanAvg;

	timing.populate += AdvCTimer() - start;
	return 1;
}

void AdvCompr::ComputeMeanTrace()
{
	int j, k, pt;
	int cnt = ntrcsL / VEC8_SIZE;
	v8f *ltrc;
	v8f_u sum;

	double start = AdvCTimer();
	memset(mean_trc, 0, sizeof(float[npts]));

	for (pt = 0; pt < npts; pt++)
	{
		sum.V = LD_VEC8F(0);
		ltrc = (v8f *) (&TRCS_ACCESS(0,pt));
		for (j = 0; j < cnt; j++)
			sum.V += *ltrc++;

		for (k = 0; k < VEC8_SIZE; k++)
			mean_trc[pt] += sum.A[k];
	}

	cnt = ntrcs/*-pinnedTrcs*/;
	if (cnt > 0)
	{
//		   AdvComprPrintf("MeanTrc = ");
		for (pt = 0; pt < npts; pt++)
		{
			mean_trc[pt] /= (float) cnt;
//			      AdvComprPrintf("%.02f ",mean_trc[pt]);
		}
//		   AdvComprPrintf("\n");
	}
	timing.mean += AdvCTimer() - start;
}

void AdvCompr::SubtractMeanFromTraces()
{
	int j, pt, itrc;
	int cnt = ntrcsL / VEC8_SIZE;
	v8f *ltrc;
	v8f_u mean;

	double start = AdvCTimer();

	// subtract the mean from every trace
	for (pt = 0; pt < npts; pt++)
	{
		mean.V = LD_VEC8F(mean_trc[pt]);
		ltrc = (v8f *) (&TRCS_ACCESS(0,pt));
		for (j = 0; j < cnt; j++)
		{
			*ltrc -= mean.V;
			ltrc++;
		}
	}

	// now re-zero all pinned traces
	for (itrc = 0; itrc < ntrcs; itrc++)
	{
		if (trcs_state[itrc]==0)
		{
			for (pt = 0; pt < npts; pt++)
				TRCS_ACCESS(itrc,pt) = 0;
		}
	}

	for (itrc = ntrcs; itrc < ntrcsL; itrc++)
	{
		for (pt = 0; pt < npts; pt++)
			TRCS_ACCESS(itrc,pt) = 0;
	}
//   AdvComprPrintf("%s: pinned %d of %d\n",__FUNCTION__,pinned,ntrcs);
	timing.SubtractMean += AdvCTimer() - start;
}

void AdvCompr::SetMeanOfFramesToZero_SubMean(int earlyFrames)
{
	int itrc, pt;
	int lw=ntrcsL/VEC8_SIZE;
	double start = AdvCTimer();

	if (earlyFrames == 0)
		earlyFrames = npts;

	for (itrc = 0; itrc < ntrcsL; itrc+=4*VEC8_SIZE)
	{
		v8f avgV0 = LD_VEC8F(0.0f);
		v8f avgV1 = LD_VEC8F(0.0f);
		v8f avgV2 = LD_VEC8F(0.0f);
		v8f avgV3 = LD_VEC8F(0.0f);
		v8f *trcsV = (v8f *)&TRCS_ACCESS(itrc,0);
		for (pt = 0; pt < earlyFrames; pt++,trcsV+=lw)
		{
			avgV0 += trcsV[0];
			avgV1 += trcsV[1];
			avgV2 += trcsV[2];
			avgV3 += trcsV[3];
		}
		v8f efV=LD_VEC8F((float)earlyFrames);
		avgV0 /= efV;
		avgV1 /= efV;
		avgV2 /= efV;
		avgV3 /= efV;
		trcsV = (v8f *)&TRCS_ACCESS(itrc,0);
		for (pt = 0; pt < npts; pt++,trcsV+=lw)
		{
			v8f meanV = LD_VEC8F(mean_trc[pt]);
			trcsV[0] -= avgV0 + meanV;
			trcsV[1] -= avgV1 + meanV;
			trcsV[2] -= avgV2 + meanV;
			trcsV[3] -= avgV3 + meanV;
		}
	}

	timing.SetMeanToZero += AdvCTimer() - start;
}

// each iteration computes vectors and returns them in coeff
// the first iteration starts with the random guesses
// but each subsequent iteration starts with the results of the
// previous regions' result

int AdvCompr::CompressBlock()
{
	int total_iter = 0;
	double start = AdvCTimer();

	// call specific compress
	if (nPcaBasisVectors || nPcaFakeBasisVectors)
	{
	  PCACompr pca(nPcaBasisVectors, nPcaFakeBasisVectors,
		       npts, nSamples, nSamples, t0estIdx,
		       sampleMatrix, trcs_coeffs, basis_vectors);
	  pca.Compress();

  	  timing.CompressBlock += AdvCTimer() - start;

	  ExtractVectors();
	}
#ifdef PCASPLINE
	else if (nSplineBasisVectors+nPcaSplineBasisVectors)
	{
	  PcaSpline compressor(nPcaSplineBasisVectors, SplineStrategy);
	  compressor.SetTikhonov(tikhonov_k);
	  compressor.SetOrder(spline_order);
	  PcaSplineRegion compressed;
	  compressed.n_wells = ntrcs;
	  compressed.n_frames = npts;
	  compressed.n_basis = compressor.NumBasisVectors(nPcaSplineBasisVectors, compressed.n_frames,
							  compressor.GetOrder(), SplineStrategy);
	  compressed.basis_vectors = basis_vectors; // needs to be aligned
	  compressed.coefficients = trcs_coeffs;
	  compressor.LossyCompress(ntrcs, npts, trcs, nSamples, sampleMatrix, compressed);

  	  timing.CompressBlock += AdvCTimer() - start;
	}
#endif

	return (total_iter);
}

void AdvCompr::CreateSampleMatrix()
{
	int good_wells = 0;
	int * __restrict state_begin = trcs_state;
	int * __restrict state_end = trcs_state + ntrcs;
	double start = AdvCTimer();
	while (state_begin != state_end)
	{
          // count good wells and fill them in with number of basis vectors
          if (*state_begin){
            good_wells++;
          }
          state_begin++;
	}
	int sample_rate=ntrcs/nSamples;
	for (int i = 0; i < npts; i++)
	{
		float *__restrict col_start = &TRCS_ACCESS(0,i);
		float *__restrict col_end = col_start + ntrcs;
		float *__restrict sample_col_begin = sampleMatrix + nSamples * i;
		uint32_t check_count = 0;
		for(int nt=0;nt<ntrcs;nt+=sample_rate)
		{
			{
				*sample_col_begin++ = *col_start;
				check_count++;
				if(check_count == nSamples)
					break;
			}
			col_start+=sample_rate;
		}
		if(check_count != nSamples)
		{
			AdvComprPrintf("check_count(%d) != n_sample_rows(%d) good_wells(%d) sample_rate(%d) ntrcs(%d) col_start(%p) col_end(%p) i(%d)\n",
					check_count,nSamples,good_wells,sample_rate,ntrcs,col_start,col_end,i);
			assert(check_count == nSamples);
		}
	}
	timing.createSample += AdvCTimer() - start;
}

float AdvCompr::ExtractVectors()

{
	double start = AdvCTimer();
	uint32_t ntrcsLV = ntrcsL / VEC8_SIZE;
	float ptgv[npts * hdr.nBasisVec];
	float gv[npts];
	int pt;

	for (pt = 0; pt < npts; pt++)
		gv[pt] = 1.0f;

	for (int nvect = 0; nvect < hdr.nBasisVec; nvect++)
	{
		float ssum = 0;
		for (pt = 0; pt < npts; pt++)
		{
			float ptgvv = COEFF_ACCESS(pt,nvect)*gv[pt];
			ssum+=ptgvv*ptgvv;
		}
		ssum = sqrt(ssum);
		if(ssum == 0)
			ssum = 1.0f; // dont divide by 0
		for (pt = 0; pt < npts; pt++)
			ptgv[nvect*npts+pt] = ((COEFF_ACCESS(pt,nvect)*gv[pt])/ssum);
	}



   for (int itrc=0;itrc < ntrcsL;itrc+=4*VEC8_SIZE)
   {
	   for(int nvect=0;nvect<hdr.nBasisVec;nvect++)
	   {
		  v8f *sumPtr=(v8f *)&TRC_COEFF_ACC(nvect,itrc);
		  v8f *trcsUp = (v8f *)&TRCS_ACCESS(itrc,0);
		  float *ptgvp = &ptgv[nvect*npts];
		  v8f sumU0=LD_VEC8F(0);
		  v8f sumU1=LD_VEC8F(0);
		  v8f sumU2=LD_VEC8F(0);
		  v8f sumU3=LD_VEC8F(0);
		  for (int pt=0;pt < npts;pt++)
		  {
			  v8f ptgvV = LD_VEC8F(ptgvp[pt]);
			  sumU0 += trcsUp[0] * ptgvV;
			  sumU1 += trcsUp[1] * ptgvV;
			  sumU2 += trcsUp[2] * ptgvV;
			  sumU3 += trcsUp[3] * ptgvV;
			  trcsUp += ntrcsLV;
		  }

		  sumPtr[0] = sumU0;
		  sumPtr[1] = sumU1;
		  sumPtr[2] = sumU2;
		  sumPtr[3] = sumU3;
	   }
   }
   timing.extractVect += AdvCTimer()-start;
   return 0;
}


int AdvCompr::GetBitsNeeded()
{
	int npts = hdr.npts;
	int ntrcs = (hdr.rend - hdr.rstart) * (hdr.cend - hdr.cstart);
	int nvec = hdr.nBasisVec;

	double minChange[nvec];
	float numCountsNeeded[nvec];
	double tmpD;
	int nv, i, bit, rc = 0;
	double start = AdvCTimer();

	for (nv = 0; nv < nvec; nv++)
	{
		maxVals[nv] = ADV_COMPR_MIN_VAL;
		minVals[nv] = ADV_COMPR_MAX_VAL;
		minChange[nv] = ADV_COMPR_MAX_VAL;
	}
	for (nv = 0; nv < nvec; nv++)
	{
		for (i = 0; i < ntrcs; i++)
		{
			if (trcs_state[i]==0)
				continue;
			if (TRC_COEFF_ACC(nv,i) > maxVals[nv])
				maxVals[nv] = TRC_COEFF_ACC(nv,i);
			if (TRC_COEFF_ACC(nv,i) < minVals[nv])
				minVals[nv] = TRC_COEFF_ACC(nv,i);
		}
	}

	for (nv = 0; nv < nvec; nv++)
	{
		for (i = 0; i < npts; i++)
		{
			tmpD = basis_vectors[nv * nvec + i];
			if (tmpD < 0)
				tmpD = -tmpD;
			tmpD = 1 / tmpD;
			if (tmpD < minChange[nv])
				minChange[nv] = tmpD;
		}
	}
	for (nv = 0; nv < nvec; nv++)
	{
		numCountsNeeded[nv] = ((maxVals[nv] - minVals[nv])
				/ (float) minChange[nv]);
		for (bit = 1; bit < 24; bit++)
		{
			if ((1 << bit) > numCountsNeeded[nv])
				break;
		}

		bitsNeeded[nv] = bit;

		if (bitsNeeded[nv] > rc)
			rc = bitsNeeded[nv];

		if (bitsNeeded[nv] > 24)
		{
	#ifndef BB_DC
			AdvComprPrintf("got a weird numBitsNeeded value %d on vector %d min %f max %f\n",
					bitsNeeded[nv],nv,minVals[nv],maxVals[nv]);
	#endif
			bitsNeeded[nv] = 20; //cap at a reasonable max
		}
		if (bitsNeeded[nv] < 3)
		{
	#ifndef BB_DC
			AdvComprPrintf("got a weird numBitsNeeded value %d on vector %d min %f max %f\n",
					bitsNeeded[nv],nv,minVals[nv],maxVals[nv]);
	#endif
			bitsNeeded[nv] = 3; //cap at a reasonable max
		}
	}
#if 0
	AdvComprPrintf("\ntrcs_coeffs Max: ");
	for(int i=0;i<nvec;i++)
	{
		AdvComprPrintf("%.03f ",max[i]);
	}
	AdvComprPrintf("\ntrcs_coeffs Min: ");
	for(int i=0;i<nvec;i++)
	{
		AdvComprPrintf("%.03f ",min[i]);
	}
	AdvComprPrintf("\ntrcs_coeffs minChange: ");
	for(int i=0;i<nvec;i++)
	{
		AdvComprPrintf("%.03f ",minChange[i]);
	}
	AdvComprPrintf("\ntrcs_coeffs numCountsNeeded: ");
	for(int i=0;i<nvec;i++)
	{
		AdvComprPrintf("%d ",(int)numCountsNeeded[i]);
	}
	AdvComprPrintf("\ntrcs_coeffs numBitsNeeded: ");
	for(int i=0;i<nvec;i++)
	{
		AdvComprPrintf("%d ",numBitsNeeded[i]);
	}
	AdvComprPrintf("\nMaxBitsNeeded: %d\n\n",rc);
#endif

	timing.getBits += AdvCTimer() - start;

	return rc;
}

#define PACKBITS(v,cbv,bn,op,pb) \
	{ \
		cbv += (bn); \
		if (cbv < pb) \
		{ \
			*op |= (v << (pb - cbv)); \
		} \
		else if (cbv == pb) \
		{ \
			cbv = 0; \
			*op++ |= v; \
		} \
		else \
		{ \
			*op++ |= v >> (cbv - pb) & ((1 << ((bn)-(cbv - pb)))-1); \
			cbv -= pb; \
			*op |= (v << (pb - cbv)); \
		} \
	}

uint64_t AdvCompr::PackBits(uint64_t *buffer, uint64_t *bufferEnd)
{
	// pack the trcs_coeffs into buffer using hdr.numBits bits per value
	uint32_t curBitValue = 0;
	uint32_t ntrcs = ((hdr.rend - hdr.rstart) * (hdr.cend - hdr.cstart));
	uint32_t nvec = hdr.nBasisVec;
	uint32_t nlvec;
	uint32_t itrc, nv;
	uint64_t *output = buffer;
	uint64_t rc;
	float units[nvec];
	float max[nvec];
	float valf;
	uint32_t ptrBits = sizeof(*buffer) * 8; // in bits
	uint64_t val;
	double start = AdvCTimer();

	for (nv = 0; nv < nvec; nv++)
	{
		max[nv] = (1 << bitsNeeded[nv]) - 1;
		units[nv] = ((maxVals[nv] - minVals[nv])
				/ (float) ((1 << bitsNeeded[nv]) - 1));
	}

	for (itrc = 0; itrc < ntrcs; itrc++)
	{
		nlvec = nvec;

		if (trcs_state[itrc])
		{
			for (nv = 0; nv < nlvec; nv++)
			{
				valf = ((TRC_COEFF_ACC(nv,itrc) - minVals[nv]) / units[nv])
					+ 1.0f;
				if (valf <= 1)
					valf = 1;
				if (valf > max[nv])
					valf = max[nv];
				val = valf;
				val &= (1 << bitsNeeded[nv]) - 1;

				PACKBITS(val, curBitValue, bitsNeeded[nv], output, ptrBits);

				if (output > bufferEnd)
				{
					AdvComprPrintf("PACK: we went too far!(%d/%d/%d) %p %p %p\n",
							itrc, nv, ntrcs, output, buffer, bufferEnd);
					exit(-1);
				}
			}
		}
		else
		{
//			printf("pinned %d/%d\n",itrc/(hdr.cend - hdr.cstart)+hdr.rstart,itrc%(hdr.cend - hdr.cstart)+hdr.cstart);
			for (nv = 0; nv < nlvec; nv++)
			{
				val=0;
				PACKBITS(val, curBitValue, bitsNeeded[nv], output, ptrBits);

				if (output > bufferEnd)
				{
					AdvComprPrintf("PACK: we went too far!(%d/%d/%d) %p %p %p\n",
							itrc, nv, ntrcs, output, buffer, bufferEnd);
					exit(-1);
				}

			}
		}
	}
	timing.PackBits += AdvCTimer() - start;
	rc = (uint64_t) output - (uint64_t) buffer;
//	AdvComprPrintf("%s: output=%p buffer=%p rc=%lx\n",__FUNCTION__,output,buffer,rc);
	return rc;
}

int AdvCompr::SaveResidual(int add)
{
	int r, c, pt, itrc = 0;
	short int *rowPtr;


	AdvComprPrintf("replacing values with residuals(%d) (%d/%d) (%d/%d)\n",
			add,hdr.rstart, hdr.rend, hdr.cstart, hdr.cend);
//	AdvComprPrintf("  1) %.02f %.01f %.01f\n", TRCS_ACCESS(0,0),
//			TRCS_ACCESS(0,1), TRCS_ACCESS(0,2));
	for (r = hdr.rstart; r < hdr.rend; r++)
	{
		rowPtr = &raw[w * r];
		for (c = hdr.cstart; c < hdr.cend; c++)
		{
			for (pt = 0; pt < npts; pt++)
			{
//				if (add == 2)
//				{
//					rowPtr[c + pt * w * h] = (short int)((float)(TRCS_ACCESS(itrc,pt)) + (float)mean_trc[pt]/*/Ggv[pt]*/+ 0.5f);
//				}
//				else
				if (add == 1)
				{
					rowPtr[c + pt * w * h] += TRCS_ACCESS(itrc,pt)/*/Ggv[pt]*/
							+ 0.5;
				}
				else
				{
					rowPtr[c + pt * w * h] = (TRCS_ACCESS(itrc,pt)/*/Ggv[pt]*/
							+ 0.5);
					if (add == 2)
						rowPtr[c + pt * w * h] += mean_trc[pt];
					else
						rowPtr[c + pt * w * h] += 8192;
				}
//				if(rowPtr[c+pt*w*h] > 16383)
//					rowPtr[c+pt*w*h] = 16383;
//				if(rowPtr[c+pt*w*h] < 0)
//					rowPtr[c+pt*w*h]=0;

			}
			itrc++;
		}
	}

	for (int itrc = 0; itrc < hdr.nBasisVec; itrc++)
	{
		// save the coefficient traces as the first pixels
		AdvComprPrintf("%d) ", itrc);
		for (pt = 0; pt < npts; pt++)
		{
			AdvComprPrintf("%.02f ", 10.0f * basis_vectors[pt + itrc * npts]);
			raw[itrc + pt * w * h] = (short int) (8192.0f
					+ (100.0f * basis_vectors[pt + itrc * npts]));
		}
		AdvComprPrintf("\n");
	}
	return 0;
}

void AdvCompr::SaveRawTrace()
{
	int r, c, pt, lpt, itrc = 0;
	int frameStride = w * h;

//	AdvComprPrintf("trc3= ");
//    for(int pt=0;pt<npts;pt++)
//    	AdvComprPrintf("%.02f ",mean_trc[pt]+TRCS_ACCESS(1,pt));
//    AdvComprPrintf("\n");

	for (r = hdr.rstart; r < hdr.rend; r++)
	{
		for (c = hdr.cstart; c < hdr.cend; c++)
		{
			for (pt = 0; pt < npts; pt++)
			{
				lpt = pt;
				raw[(lpt * frameStride) + (r * w) + c] = mean_trc[pt]
						+ TRCS_ACCESS(itrc,pt);
			}
			itrc++;
		}
	}
}

int AdvCompr::WriteTraceBlock()
{
	// write the info for this block out to file
	int nvect = hdr.nBasisVec;
	double start = AdvCTimer();
	uint64_t dl = 0;

	GetBitsNeeded();

	for (int i = 0; i < nvect; i++)
		dl += bitsNeeded[i] * ntrcs;

	dl += 6 * ntrcs; // for number of traces
	dl /= 8 * 8; // its in bits.. convert to 64-bit words
	dl++;

	uint64_t buffer[dl];
	uint64_t *bufferEnd = buffer + (dl);
	memset(buffer, 0, dl * sizeof(uint64_t));

	dl = PackBits(buffer, bufferEnd);

	dl /= 8; // its in bytes... convert to 64-bit words
	dl++;

//	AdvComprPrintf("%s: len=%lx dl=%lx\n",__FUNCTION__,(uint64_t)bufferEnd-(uint64_t)buffer,dl*8);
	//	if(dl%(8*8))// round up the number of bytes to the nearest 64 bits

	hdr.datalength = dl;

	//write out the hdr
	int old_npts = hdr.npts;
	int new_npts = npts_newfr;
	hdr.npts = new_npts;

	Write(fd, &hdr, sizeof(hdr));

	hdr.npts = old_npts; // restore the old value

	// write out the bits needed per vec
	Write(fd, bitsNeeded, sizeof(bitsNeeded[0]) * nvect);

	if(frameRate != 15 && timeTransform){
		float nMeanTrc[npts_newfr];
		TimeTransform_trace(npts,npts_newfr,timestamps_compr,timestamps_newfr,mean_trc,nMeanTrc,1);

		// write out the mean trace
		Write(fd, nMeanTrc, sizeof(nMeanTrc));

		float nBasisVects[nvect*npts_newfr];
		TimeTransform_trace(npts,npts_newfr,timestamps_compr,timestamps_newfr,basis_vectors,nBasisVects,nvect);

		// write out the basis vectors
		Write(fd, nBasisVects, sizeof(nBasisVects));
	}
	else{
		// write out the mean trace
		Write(fd, mean_trc, sizeof(float) * npts);

		// write out the pca vectors
		Write(fd, basis_vectors, sizeof(float) * nvect * npts);
	}
	// write out the min vectors
	Write(fd, minVals, sizeof(float) * nvect);

	// write out the max vectors
	Write(fd, maxVals, sizeof(float) * nvect);

	// write out the raw trace coefficients
	Write(fd, buffer, hdr.datalength * sizeof(buffer[0]));

	timing.write += AdvCTimer() - start;
	return 0;
}

void AdvCompr::ComputeGain()
{
	float lowest_point;
	float highest_point;
	float mean_lowest_point;
	float mean_highest_point;
	float mean_height;
	int itrc = 0;
	int pt, r, c;
	int tooHigh = 0;
	int tooLow = 0;
	int justRight = 0;
	int pinned = 0;
	int Nan = 0;
//	float saved_mean_trc[npts];

	PopulateTraceBlock_ComputeMean();

	mean_lowest_point = 0.0f;
	mean_highest_point = 0.0f;
	for (pt = 0; pt < npts; pt++)
	{
		if (mean_trc[pt] < mean_lowest_point)
			mean_lowest_point = mean_trc[pt];
		if (mean_trc[pt] > mean_highest_point)
			mean_highest_point = mean_trc[pt];
	}
	mean_height=mean_highest_point - mean_lowest_point;
	t0est = findt0(&t0estIdx);
	printf("%s: t0=%d t0idx=%d\n",__FUNCTION__,t0est,t0estIdx);
	for (pt = 0; pt < npts; pt++)
	{
		mean_trc[pt]=0;
	}
	// so the next func doesnt' subtract the mean.
	SetMeanOfFramesToZero_SubMean(npts);

	itrc = 0;
	// gain is ratio of mean to local trace
	for (r = hdr.rstart; r < hdr.rend; r++)
	{
		c = hdr.cstart;
		float *gcP = &gainCorr[r * w + c];
		for (; c < hdr.cend; c++,gcP++)
		{
			if (trcs_state[itrc]) // don't do pinned pixels
			{
				lowest_point = 0.0f;
				highest_point = 0.0f;
				for (pt = 0; pt < npts; pt++)
				{
					if (TRCS_ACCESS(itrc,pt) < lowest_point)
						lowest_point = TRCS_ACCESS(itrc,pt);
					if (TRCS_ACCESS(itrc,pt) > highest_point)
						highest_point = TRCS_ACCESS(itrc,pt);
				}
				*gcP = mean_height/(highest_point - lowest_point);
				if(*gcP != *gcP)
				{
					Nan++;
					*gcP=1.0;
				}
				if (*gcP > ADV_COMPR_MAX_GAIN)
				{
					tooHigh++;
					*gcP = ADV_COMPR_MAX_GAIN;
				}
				else if (*gcP < ADV_COMPR_MIN_GAIN)
				{
					tooLow++;
					*gcP = ADV_COMPR_MIN_GAIN;
				}
				else
				{
					justRight++;
				}
			}
			else
			{
				pinned++;
				*gcP = 1.0;
			}
			itrc++;
		}
	}
	AdvComprPrintf("%s: tooHigh=%d tooLow=%d justRight=%d pinned=%d nan=%d\n",__FUNCTION__,tooHigh,tooLow,justRight,pinned,Nan);
}

void AdvCompr::doTestComprPre()
{
	if (testType == 2)
	{
		SaveResidual(3);
	}
	else if (testType == 3)
	{
		ComputeMeanTrace();
		t0est = findt0(&t0estIdx);

		SubtractMeanFromTraces();
		for (int i = 0; i < npts; i++)
			mean_trc[i] += 8192; // add back some dc offset

		SaveResidual(2);
	}
}

void AdvCompr::doTestComprPost()
{

	if (testType == 1)
	{
		ExtractTraceBlock(); // extract into image
	}
	else if (testType == 4)
	{
		short int *oldRaw=raw;
		raw = (short int *)malloc(2*w*h*npts);
		ExtractTraceBlock(); // extract into image
		for(int frame=0;frame<npts;frame++){
			short int *ptr1=&raw[frame*w*h];
			short int *ptr2=&oldRaw[frame*w*h];
			for(int idx=0;idx<w*h;idx++){
				*ptr2 = *ptr2 - *ptr1 + 8192;
				ptr2++;
				ptr1++;
			}
		}
		free(raw);
		raw=oldRaw;
	}
	else if (testType == 5)
	{
		SetMeanOfFramesToZero_SubMean(npts);
		ComputeMeanTrace();
		SubtractMeanFromTraces();
		t0est = findt0(&t0estIdx);

		for (int i = 0; i < npts; i++)
			mean_trc[i] += 8192; // add back some dc offset
		SaveRawTrace();
	}
	else if (testType == 6)
	{
		// add a dc offset to mean trc
		for (int pt = 0; pt < npts; pt++)
			mean_trc[pt] = 0;
		SaveRawTrace();
	}
}

static int GetSlopeAndOffset(float *trace, uint32_t startIdx, uint32_t endIdx,
		float *slope, float *offset, FILE *logfp);
float FindT0Specific(float *trace, uint32_t trace_length, uint32_t T0Initial,
		FILE *log_fp);

int32_t DCT0Finder(float *trace, uint32_t trace_length, FILE *logfp)
{
	uint32_t lidx2 = 0;
	float traceDv[trace_length];
	int32_t rc = -1;
	int32_t T0Initial = -1;
	float T0Specific = 0;

	// turn the trace into a derivative
	for (lidx2 = trace_length - 1; lidx2 > 0; lidx2--)
	{
		traceDv[lidx2] = fabs(trace[lidx2] - trace[lidx2 - 1]);
	}
	traceDv[0] = 0;
	if (logfp != NULL)
	{
		fprintf(logfp, "  trace:\n\n");
		for (uint32_t i = 0; i < 100 && i < trace_length; i += 10)
		{
			fprintf(logfp," (%d)  ",i);
			for(int j=0;j<10 && (i+j) < trace_length;j++)
				fprintf(logfp,"%.0lf ",trace[i + j]);
			fprintf(logfp,"\n");
		}
		fprintf(logfp, "\n\n");
		fprintf(logfp, "  derivative trace:\n\n");
		for (uint32_t i = 0; i < 100 && i < trace_length; i += 10)
		{
			fprintf(logfp," (%d)  ",i);
			for(int j=0;j<10 && (i+j) < trace_length;j++)
				fprintf(logfp,"%.0lf ",traceDv[i + j]);
			fprintf(logfp,"\n");
		}
		fprintf(logfp, "\n\n");
	}

	// now, find t0Estimate
	float maxDv = 0;
	for (lidx2 = 1; lidx2 < (trace_length - 4); lidx2++)
	{
		if (maxDv < traceDv[lidx2])
		{
			maxDv = traceDv[lidx2];
		}
	}
	maxDv /= 2.0f;
	if (logfp != NULL)
		fprintf(logfp, "maxDv=%.2f tl=%d\n", maxDv, trace_length);

	for (lidx2 = 1; lidx2 < (trace_length - 4); lidx2++)
	{
		if ((traceDv[lidx2] > maxDv) && (traceDv[lidx2 + 1] > maxDv)
				&& (traceDv[lidx2 + 2] > maxDv))
		{
			// this is the spot..
			T0Initial = lidx2;
			break;
		}
	}

	if (logfp != NULL)
		fprintf(logfp, "\n  Initial T0 guess=%d\n", T0Initial);

	T0Specific = FindT0Specific(trace, trace_length, T0Initial, logfp);

	if ((T0Initial > 2) && (T0Initial < (int32_t) (trace_length - 4)))
	{
		if((T0Specific < (T0Initial + 12)) && (T0Specific > (T0Initial - 12)))
		{
			// we found a point pretty close to the original one...
			rc = T0Specific;
			if (logfp != NULL)
				fprintf(logfp, "using new T0Specific=%.2lf T0Guess=%df\n",
						T0Specific, T0Initial);
		}
		else
		{
			if (logfp != NULL)
				fprintf(logfp, "Rejecting the new T0Specific=%.2lf\n", T0Specific);
			rc = T0Initial; // use the initial guess
		}
	}
	else
		rc = 8; // only use the first couple of points as "before t0"

	return rc;
}

float FindT0Specific(float *trace, uint32_t trace_length,
		uint32_t T0InitialGuess, FILE *logfp)
{
	float T0Max = 0;
	float T0Specific = 0;
	float slope1 = 0, offset1 = 0;
	float slope2 = 0, offset2 = 0;

	uint32_t rhswidth, xguess;

	for (rhswidth = 06; rhswidth < 12; rhswidth++)
	{
		for (xguess = T0InitialGuess - 4;
				xguess < T0InitialGuess + 4
						&& ((xguess + rhswidth) < trace_length); xguess++)
		{
			GetSlopeAndOffset(trace, 0, xguess, &slope1, &offset1, logfp);
			GetSlopeAndOffset(trace, xguess + 1, xguess + rhswidth, &slope2,
					&offset2, logfp);

//			if(logfp != NULL)
//			{
//				fprintf(logfp,"  slope1=%.2lf offset1=%.2lf\n",slope1,offset1);
//				fprintf(logfp,"  slope2=%.2lf offset2=%.2lf\n",slope2,offset2);
//			}
			if (slope1 != 0 && offset1 != 0 && slope2 != 0 && offset2 != 0
					&& slope1 != slope2)
			{
				T0Specific = -(offset1 - offset2) / (slope1 - slope2);
				if (logfp != NULL)
					fprintf(logfp, "  Try x=%d rhsw=%d T0=%.2lf\n", xguess,
							rhswidth, T0Specific);

				if (T0Specific > T0Max)
					T0Max = T0Specific;
			}

		}
	}
	return T0Max;
}

int GetSlopeAndOffset(float *trace, uint32_t startIdx, uint32_t endIdx,
		float *slope, float *offset, FILE *logfp)
{
	float sumTime = 0;
	float sumTimesq = 0;
	float sumTimexVal = 0;
	float sumVal = 0;
	float num1, num2, den;
	uint32_t idx;

	*slope = 0;
	*offset = 0;

	for (idx = startIdx; idx < endIdx; idx++)
	{
		sumTime += (double) idx;
		sumTimesq += (double) (idx * idx);
		sumVal += trace[idx];
		sumTimexVal += ((double) idx) * trace[idx];
	}

//	if(logfp != NULL)
//	{
//		fprintf(logfp,"    sumTime=%.2lf sumTimesq=%.2lf sumVal=%.2lf sumTimexVal=%.2lf\n",sumTime,sumTimesq,sumVal,sumTimexVal);
//	}
	num1 = ((sumTimesq * sumVal) - (sumTime * sumTimexVal));
	num2 = (((endIdx - startIdx) * sumTimexVal) - (sumTime * sumVal));
	den = (((endIdx - startIdx) * sumTimesq) - (sumTime * sumTime));
	if (den != 0)
	{
		*offset = num1 / den;
		*slope = num2 / den;
	}
	return 0;
}

float AdvCompr::findt0(int *idx)
{
	FILE *fp=NULL;
	float rc=-1;

//	// UN_COMMENT for T0 debug
//	{
//		char name[256];
//		sprintf(name,"dbgT0_%d_%d.txt",hdr.rstart,hdr.cstart);
//		fp = fopen(name,"w+");
//	}

	rc = DCT0Finder(mean_trc, npts, fp);
	if (fp)
		fclose(fp);

	if(rc > 0){

		// find the index associated with this timestamp...
	if(idx)
		*idx=rc;
//		int ts=0;
//
//		for(ts=0;ts<npts;ts++){
//			if(timestamps_compr[ts] >= rc){
//				if(idx)
//					*idx=ts;
//				break;
//			}
//		}
	}

	return rc;
}

#ifndef BB_DC
struct lsrow_header {
    uint32_t magic;
    uint32_t version;
    uint32_t rows;
    uint32_t cols;
};
#define LSROWIMAGE_MAGIC_VALUE    0xFF115E3A
#endif

void AdvCompr::WriteGain(int region, int w, int h, char *destPath)
{
	char fnBuf[512];
	float *LCorr = AllocateGainCorrection(region,w*h*sizeof(float));

	if(LCorr == NULL){
		AdvComprPrintf("%s: not writing region %d (%p)\n",__FUNCTION__,region,gainCorrection[region]);
		return;
	}

	if(region < 97)
		sprintf(fnBuf,"%s/Gain.lsr",destPath);
	else
		sprintf(fnBuf,"%s/Gain_spatial.lsr",destPath);
	FILE *fp = fopen(fnBuf,"wb");
	if(fp)
	{
		struct lsrow_header hdr;

		hdr.magic = LSROWIMAGE_MAGIC_VALUE;
		hdr.version = 0;
		hdr.rows = h;
		hdr.cols = w;

		if (fp)
		{
			fwrite(&hdr,sizeof(hdr),1,fp);
			fwrite(LCorr,h * w,sizeof(float),fp);
			fclose(fp);
		}
		int nonone=0;
		for(int idx=0;idx<w*h;idx++){
			if(LCorr[idx] != 1.0)
				nonone++;
		}

		AdvComprPrintf("%s: wrote %s (%p) valid=%d\n",__FUNCTION__,fnBuf,gainCorrection[region],nonone);
	}

}

float *AdvCompr::ReadGain(int region, uint32_t cols, uint32_t rows, char *srcPath)
{
	float *rc = NULL;

	struct lsrow_header hdr;
	FILE *ofile = fopen(srcPath,"r");

	if (ofile)
	{
		float *data = (float *)malloc(rows*cols*sizeof(float));
		int lrc = fread(&hdr,sizeof(hdr),1,ofile);
		if(lrc > 0 &&
		   hdr.magic == LSROWIMAGE_MAGIC_VALUE &&
		   hdr.rows == rows &&
		   hdr.cols == cols)
		{
			lrc = fread(data,sizeof(float),hdr.rows * hdr.cols,ofile);
			printf("read %d elements from file (%d/%d) opbuf(%d/%d) %s [%f %f %f %f]\n",lrc,
					hdr.rows,hdr.cols,rows,cols,srcPath,data[0],data[1],data[2],data[3]);

			if(lrc > 0){
				rc = ReSetGain(region,cols,rows,data);
			}
		}
		else
		{
			printf("%s: %d %x/%x %d/%d %d/%d",__FUNCTION__,lrc,hdr.magic,LSROWIMAGE_MAGIC_VALUE,
					hdr.rows,rows, hdr.cols, cols);
		}
		fclose(ofile);
		free(data);
	}
	else{
		printf("failed to open %s\n",srcPath);
	}

	return rc;
}

float *AdvCompr::ReSetGain(int region, int w, int h, float *gainPtr)
{
//	uint32_t len=w*h*sizeof(float);
	int high=0;
	int low=0;
	int justRight=0;
	int nanCnt=0;

	float *LCorr = AllocateGainCorrection(region,w*h*sizeof(float));

	if(LCorr == NULL)
		return NULL;

	for (int i=0;i<(w*h);i++)
	{
		LCorr[i] = gainPtr[i];
		if(LCorr[i] != LCorr[i] || LCorr[i]==0)
		{
			LCorr[i] = 1.0f;
			nanCnt++;
		}
		else if(LCorr[i] > ADV_COMPR_MAX_GAIN)
		{
			LCorr[i] = ADV_COMPR_MAX_GAIN;
			high++;
		}
		else if(LCorr[i] < ADV_COMPR_MIN_GAIN)
		{
			LCorr[i] = ADV_COMPR_MIN_GAIN;
			low++;
		}
		else
			justRight++;
	}
	AdvComprPrintf("  ReSetGain %d) nan:%d %d/%d/%d [%f %f %f %f]  [%f %f %f %f]\n",
			region,nanCnt,high,justRight,low,gainPtr[0],gainPtr[1],gainPtr[2],gainPtr[3],LCorr[0],LCorr[1],LCorr[2],LCorr[3]);
	return LCorr;
}

void AdvCompr::SetGain(int region, int w, int h, float conv, uint16_t *gainPtr)
{
//	uint32_t len=w*h*sizeof(float);
	int high=0;
	int low=0;
	int justRight=0;
	int nanCnt=0;
	int zero=0;

	float *LCorr = AllocateGainCorrection(region,w*h*sizeof(float));

	if(LCorr == NULL)
		return;

	for (int i=0;i<(w*h);i++)
	{
		LCorr[i] = 1.0f/(((float)gainPtr[i]) * conv);
		if(LCorr[i] != LCorr[i])
		{
			LCorr[i] = 1.0f;
			nanCnt++;
		}
		else if(gainPtr[i] == 0){
			LCorr[i] = 1.0f;
			zero++;
		}
		else if(LCorr[i] > ADV_COMPR_MAX_GAIN)
		{
			LCorr[i] = ADV_COMPR_MAX_GAIN;
			high++;
		}
		else if(LCorr[i] < ADV_COMPR_MIN_GAIN)
		{
			LCorr[i] = ADV_COMPR_MIN_GAIN;
			low++;
		}
		else{
			justRight++;
		}
	}

	// now, replace all of the

	AdvComprPrintf("  SetGain %d) %p nan:%d %d/%d/%d/%d [%d %d %d %d] conv=%f  [%f %f %f %f]\n",
			region,gainPtr,nanCnt,high,justRight,low,zero, gainPtr[0],gainPtr[1],gainPtr[2],gainPtr[3],conv,LCorr[0],LCorr[1],LCorr[2],LCorr[3]);

}

void AdvCompr::ClearGain(int region)
{
	AdvComprPrintf("%s: clearing region %d(%p)\n",__FUNCTION__,region,gainCorrection[region]);
	if(gainCorrection[region] != NULL)
	{
		free(gainCorrection[region]);
		gainCorrection[region] = NULL;
		gainCorrectionSize[region] = 0;
	}
}

#endif


int AdvCompr::UnCompress(int _threadNum)
{
	double start = AdvCTimer();
	uint64_t *fb=NULL;
	uint32_t fbLen=0;
	ThreadNum=_threadNum;
#ifdef WIN32
	wchar_t wfname[1024];
	FILE_HANDLE fd;
	size_t mbrc;

	mbstowcs_s(&mbrc,wfname,fname,strlen(fname));

	fd = CreateFile ( (LPCSTR)fname, GENERIC_READ, 0, NULL,
			OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL );
	if ( fd == INVALID_HANDLE_VALUE )
	{
		printf ( "failed to open file \n" );
		exit(-1);
	}
#else
	int fd = open(fname, O_RDONLY);
	if (fd < 0)
	{
		AdvComprPrintf("failed to open file %s\n", fname);
		exit(-1);
	}
#endif

	// read in file header
	Read(fd, &FileHdrM, sizeof(FileHdrM));
	ByteSwap4(FileHdrM.struct_version);
	ByteSwap4(FileHdrM.signature);

	Read(fd, &FileHdrV4, sizeof(FileHdrV4));
	ByteSwap4(FileHdrV4.first_frame_time);
	ByteSwap2(FileHdrV4.rows);
	ByteSwap2(FileHdrV4.cols);
	ByteSwap2(FileHdrV4.x_region_size);
	ByteSwap2(FileHdrV4.y_region_size);
	ByteSwap2(FileHdrV4.frames_in_file);
	ByteSwap2(FileHdrV4.uncomp_frames_in_file);
	ByteSwap4(FileHdrV4.sample_rate);
	ByteSwap2(FileHdrV4.hw_interlace_type);
	ByteSwap2(FileHdrV4.interlaceType);

	if (FileHdrV4.frames_in_file != npts || w != FileHdrV4.cols
			|| h != FileHdrV4.rows)
	{
		AdvComprPrintf("PCA hdr issues:  nImgPts(%d) != nImgPts(%d)\n",
				FileHdrV4.frames_in_file, npts);
		exit(-1);
	}

	cblocks = w / FileHdrV4.x_region_size;
	rblocks = h / FileHdrV4.y_region_size;
	if (w % FileHdrV4.x_region_size)
		cblocks++;
	if (h % FileHdrV4.y_region_size)
		rblocks++;

	ntrcs = ntrcsL = FileHdrV4.x_region_size * FileHdrV4.y_region_size;

	if (ntrcsL % (VEC8_SIZE * 4))
		ntrcsL = (ntrcs + (VEC8_SIZE * 4)) & (~((VEC8_SIZE * 4) - 1));

	// read in timestamps
	Read(fd, &timestamps_uncompr[0],
			sizeof(timestamps_uncompr[0]) * FileHdrV4.frames_in_file);

	for (int rblk = 0; rblk < rblocks; rblk++)
	{

		AdvComprPrintf(".");
		fflush(stdout);

		for (int cblk = 0; cblk < cblocks; cblk++)
		{
			Read(fd, &hdr, sizeof(hdr));
			// make sure hdr contains what we expect

			if ((hdr.key != ADVCOMPR_KEY) || (hdr.rend > h) || (hdr.cend > w)
					|| (hdr.npts != npts) || (hdr.ver != 0))
			{
				AdvComprPrintf(
						"hdr doesn't look correct %x rend(%d) cend(%d) npts (%d != %d) ver(%d)\n",
						hdr.key, hdr.rend, hdr.cend, hdr.npts, npts, hdr.ver);
			}
			else
			{
				ntrcs = (hdr.cend - hdr.cstart) * (hdr.rend - hdr.rstart);


				if(fb == NULL || fbLen < sizeof(uint64_t) * hdr.datalength)
				{
					if(fb )
						free(fb );

					fbLen=sizeof(uint64_t) * hdr.datalength;
					fb = (uint64_t *) malloc(fbLen);
				}

				ALLOC_STRUCTURES(hdr.nBasisVec);

//				CLEAR_STRUCTURES();

				// read in the bits needed per vec
				Read(fd, bitsNeeded, sizeof(int) * (hdr.nBasisVec));

				// read in mean_trc
				Read(fd, &mean_trc[0], sizeof(float) * hdr.npts);

				// read in basis_vectors
				Read(fd, &basis_vectors[0],
						sizeof(float) * hdr.npts * (hdr.nBasisVec));

				// read in the min vectors
				Read(fd, &minVals[0],
						sizeof(float) * (hdr.nBasisVec));

				// read in the max vectors
				Read(fd, &maxVals[0],
						sizeof(float) * (hdr.nBasisVec));

				// read in trcs_coeffs
				Read(fd, &fb[0],
						hdr.datalength * sizeof(uint64_t));

				UnPackBits(fb,
						&fb[hdr.datalength]);

				ExtractTraceBlock();


			}
		}
	}

	if(fb)
		free(fb);

	FREE_STRUCTURES(1);

#ifdef WIN32
	CloseHandle(fd);
#else
	close(fd);
#endif
	timing.overall = AdvCTimer() - start;
	return 0;
}

double AdvCompr::AdvCTimer()
{
#ifndef WIN32
	struct timeval tv;
	gettimeofday(&tv, NULL);
	double curT = (double) tv.tv_sec + ((double) tv.tv_usec / 1000000);
	return (curT);
#else
	return 0;
#endif
}



int AdvCompr::ExtractTraceBlock()
{
	int itrc = 0;
	int frameStride = w * h;
	short int *rtrc;
	int r, c, nv, pt;
	double start = AdvCTimer();
	int nbasisV=hdr.nBasisVec;

#ifdef PCA_UNCOMP_MEAN_SMOOTHING
		int t0 = DCT0Finder(mean_trc, npts, NULL);

		float nmean_trc[hdr.npts];
		for (pt = 0; pt < hdr.npts; pt++){
			int span=3;
			int start_pt=pt-span;
			int end_pt=pt+span;
			if(start_pt<0)
				start_pt=0;
			if(end_pt>hdr.npts)
				end_pt=hdr.npts;

			float avg=0;
			for(int npt=start_pt;npt<end_pt;npt++){
				avg += mean_trc[npt];
			}
			avg /= (float)(end_pt-start_pt);
			nmean_trc[pt] = avg;
		}
		if(t0 > 0 && t0 < (npts-1)){
			// we found a valid t0 estimate.
			// make the transition sharper.
			// shouldn't matter as the neighbor subtracted data is what matters.
			for (pt = 0; pt < t0; pt++)
				nmean_trc[pt]=nmean_trc[0]; // copy it back
		}
		for (pt = 0; pt < hdr.npts; pt++)
			mean_trc[pt]=nmean_trc[pt]; // copy it back
#endif

#ifdef USE_LESS_BASIS_VECTS
		nbasisV=USE_LESS_BASIS_VECTS;
#endif

#ifdef WIN32
	for (r = hdr.rstart; r < hdr.rend; r++)
	{
		for (c = hdr.cstart; c < hdr.cend; c++,itrc++)
		{
			rtrc = (short int *) &raw[r * w + c];

			if (trcs_state[itrc] < 0)
			{
				for (pt = 0; pt < hdr.npts; pt++)
					rtrc[pt * frameStride] = 0;
			}
			else
			{
				for (pt = 0; pt < hdr.npts; pt++)
				{
					float tmp = mean_trc[pt];

					for (nv = 0; nv < hdr.nBasisVec; nv++)
					{
						tmp += basis_vectors[pt + nv * npts]
								* TRC_COEFF_ACC(nv,itrc);
					}

					rtrc[pt * frameStride] = (short) (tmp + 0.5f);
				}
			}
		}
	}

#else

	if (((hdr.cend-hdr.cstart) % (4*VEC8_SIZE)) == 0)
	{
		for (r = hdr.rstart; r < hdr.rend; r++)
		{
			for (c = hdr.cstart; c < hdr.cend; c+=4*VEC8_SIZE,itrc+=4*VEC8_SIZE)
			{
				rtrc = (short int *) &raw[r * w + c];
				for (pt = 0; pt < hdr.npts; pt++)
				{
					v8f_u tmpV1,tmpV2,tmpV3,tmpV4;
					tmpV1.V = tmpV2.V = tmpV3.V = tmpV4.V = LD_VEC8F(mean_trc[pt]);
					v8s_u *ov=(v8s_u*)&rtrc[pt * frameStride];

					for (nv = 0; nv < nbasisV; nv++)
					{
						v8f *coeffV=(v8f *)&TRC_COEFF_ACC(nv,itrc);
						v8f bV = LD_VEC8F(basis_vectors[pt + nv * npts]);
						tmpV1.V += bV * coeffV[0];
						tmpV2.V += bV * coeffV[1];
						tmpV3.V += bV * coeffV[2];
						tmpV4.V += bV * coeffV[3];
					}

					CVT_VEC8F_VEC8S((ov[0]),tmpV1);
					CVT_VEC8F_VEC8S((ov[1]),tmpV2);
					CVT_VEC8F_VEC8S((ov[2]),tmpV3);
					CVT_VEC8F_VEC8S((ov[3]),tmpV4);
				}

				for(int k=0;k<(4*VEC8_SIZE);k++)
				{
					if(trcs_state[itrc+k] < 0) // if any are pinned
					{
//						printf("%s:%d Pinned %d/%d\n",__FUNCTION__,__LINE__,(itrc+k)/w,(itrc+k)%w);
						// check each one.. this should be the exception
						for (pt = 0; pt < hdr.npts; pt++)
							rtrc[pt * frameStride + k] = 0;
					}
				}
			}
		}
	}
	else
		{
		for (r = hdr.rstart; r < hdr.rend; r++)
		{
			for (c = hdr.cstart; c < hdr.cend; c+=VEC8_SIZE,itrc+=VEC8_SIZE)
			{
				rtrc = (short int *) &raw[r * w + c];
				for (pt = 0; pt < hdr.npts; pt++)
				{
					v8f_u tmpV1;
					tmpV1.V = LD_VEC8F(mean_trc[pt]);
					v8s_u *ov=(v8s_u*)&rtrc[pt * frameStride];

					for (nv = 0; nv < hdr.nBasisVec; nv++)
					{
						v8f *coeffV=(v8f *)&TRC_COEFF_ACC(nv,itrc);
						v8f bV = LD_VEC8F(basis_vectors[pt + nv * npts]);
						tmpV1.V += bV * coeffV[0];
					}

					CVT_VEC8F_VEC8S((ov[0]),tmpV1);
				}

				for(int k=0;k<(VEC8_SIZE);k++)
				{
					if(trcs_state[itrc+k] < 0) // if any are pinned
					{
//						printf("%s:%d Pinned \n",__FUNCTION__,__LINE__);
						// check each one.. this should be the exception
						for (pt = 0; pt < hdr.npts; pt++)
							rtrc[pt * frameStride + k] = 0;
					}
				}
			}
		}
	}
#endif

#ifdef OUTPUT_BASIS_VECTORS
	// put the mean frame in the first trace
	for (pt = 0; pt < hdr.npts; pt++){
		float bV = mean_trc[pt];
		raw[0 + pt*frameStride] = bV;
	}

	// put the basis vectors in the first X pixels
	for (nv = 0; nv < hdr.nBasisVec; nv++){
		for (pt = 0; pt < hdr.npts; pt++){
			float bV = 200.0f *basis_vectors[pt + nv * npts];
			raw[nv+1 + pt*frameStride] = 8192+bV;
		}
	}
#endif

#ifdef OUTPUT_VECTOR_COEFFS
	// put the coefficients in the last X frames
	itrc=0;
	for (r = hdr.rstart; r < hdr.rend; r++)
	{
		for (c = hdr.cstart; c < hdr.cend; c++,itrc++)
		{
			rtrc = (short int *) &raw[r * w + c];
			for (pt = hdr.npts-hdr.nBasisVec-1,nv=0; pt < hdr.npts; pt++,nv++){
				rtrc[pt*frameStride] = 8192+TRC_COEFF_ACC(nv,itrc);
			}
		}
	}

#endif
	timing.Extract += AdvCTimer() - start;
	return 0;
}

void AdvCompr::Write(FILE_HANDLE fd, void *buf, int len)
{
	int llen;
	int orig_len = len;
	char *dptr = (char *) buf;
	int FailCnt = 0;
	int cntr = 0;
//	static pthread_mutex_t sem = PTHREAD_MUTEX_INITIALIZER;
//	AdvComprPrintf("Write %d\n",len);

//    pthread_mutex_lock(&sem);

	while (len > 0 && (FailCnt < 10))
	{
#ifdef WIN32
		llen = WriteFile(fd,dptr,len,NULL,NULL);
#else
		llen = write(fd, dptr, len);
#endif
		if (llen > 0)
		{
			FailCnt = 0;
			len -= llen;
			dptr += llen;
			if (len)
			{
				AdvComprPrintf(
						"%d(%d): Retrying write with %d(%d) bytes left orig=%d fd=%d\n",
						cntr++, FailCnt, len, llen, orig_len, fd);
			}
		}
		else
		{
			FailCnt++;
		}
	}
//    pthread_mutex_unlock(&sem);
}

int AdvCompr::Read(FILE_HANDLE fd, void *buf, int len)
{
//	AdvComprPrintf("Read %d\n",len);
#ifdef WIN32
	DWORD rc=0;
	ReadFile(fd,buf,len,&rc,NULL);
	return (int)rc;
#else
	return read(fd, buf, len);
#endif
}

#define UNPACKBITS(v,cbv,bn,pb,ip) \
	{ \
		cbv += bn; \
		if (cbv < pb) \
		{ \
			v = (*ip >>  (pb - cbv)) & ((1 << bn)-1); \
		} \
		else if (cbv == pb) \
		{ \
			v = (*ip++ >>  (pb - cbv)) & ((1 << bn)-1); \
			cbv = 0; \
		} \
		else \
		{ \
			cbv -= pb; \
			v = (*ip++ <<  cbv) & ~((1<<cbv)-1); \
			v = (v | (*ip >> (pb - cbv))) & ((1 << bn)-1); \
		} \
	}

// upack the trcs_coeffs from buffer using hdr.numBits bits per value
void AdvCompr::UnPackBits(uint64_t *trcs_coeffs_buffer, uint64_t *bufferEnd)
{
	uint32_t curBitValue = 0;
	uint32_t nvec = hdr.nBasisVec;
	uint32_t nlvec;
	uint32_t nv;
	int itrc;
	uint64_t *input = trcs_coeffs_buffer;
#ifdef WIN32
	float *units=(float *)malloc(sizeof(float)*nvec);
#else
	float units[nvec];
#endif
	float valf;
	uint32_t ptrBits = sizeof(*trcs_coeffs_buffer) * 8; // in bits
	uint64_t val;
	double start = AdvCTimer();

	for (nv = 0; nv < nvec; nv++)
	{
		units[nv] =
				((maxVals[nv] - minVals[nv]) / (float) (1 << bitsNeeded[nv]));
	}

	nlvec = hdr.nBasisVec;
	for (itrc = 0; itrc < ntrcs; itrc++)
	{
		int pinned=0;
		for (nv = 0; nv < nlvec; nv++)
		{
			UNPACKBITS(val, curBitValue, bitsNeeded[nv], ptrBits, input);


			pinned |= (val <=0); // pinned pixel
			val--; // to account for +1 in packBits (0 is reserved for pinned)

			valf = (float) val;
			valf *= units[nv];
			valf += minVals[nv];
			TRC_COEFF_ACC(nv,itrc) = valf;

			if (input > bufferEnd)
			{
				AdvComprPrintf("UnPack: we went too far!(%d/%d/%d) %p %p %p\n",
						itrc, nv, ntrcs, input, trcs_coeffs_buffer, bufferEnd);
				exit(-1);
			}
		}
		if (pinned)
		{
			trcs_state[itrc] = -1;
//			printf("pinned pixel %d\n",itrc);
			for (nv = 0; nv < nvec; nv++)
				TRC_COEFF_ACC(nv,itrc) = 0.0f; // zero the trace
		}
		else
		{
			trcs_state[itrc] = nlvec;
		}
	}
#ifdef WIN32
	free(units);
//	free(mask);
#endif

	timing.UnPackBits += AdvCTimer() - start;

}

void AdvCompr::DumpDetailedTiming(const char *label)
{
	AdvComprPrintf("\n%sTiming: %.02lf\n", label, timing.overall);
	AdvComprPrintf("  xtalk         %.02f\n", timing.xtalk);
	AdvComprPrintf("  ccn           %.02f\n", timing.ccn);
	AdvComprPrintf("  rcn           %.02f\n", timing.rcn);
	AdvComprPrintf("  clear         %.02f\n", timing.clear);
	AdvComprPrintf("  populate      %.02lf\n", timing.populate);
	AdvComprPrintf("  SetMeanToZero %.02lf\n", timing.SetMeanToZero);
	AdvComprPrintf("  SubtractMean  %.02lf\n", timing.SubtractMean);
	AdvComprPrintf("  createSample  %.02lf\n", timing.createSample);
	AdvComprPrintf("  ComprBlock    %.02lf\n", timing.CompressBlock);
//	AdvComprPrintf("    computemean  %.02lf\n", timing.mean);
	AdvComprPrintf("  extractVect  %.02lf\n", timing.extractVect);
	AdvComprPrintf("  write      %.02lf\n", timing.write);
	AdvComprPrintf("  extractBlk %.02lf\n", timing.Extract);
	AdvComprPrintf("  getBits    %.02lf\n", timing.getBits);
	AdvComprPrintf("  packBits   %.02lf\n", timing.PackBits);
	AdvComprPrintf("  UnPackBits %.02lf\n", timing.UnPackBits);
}

void AdvCompr::DumpTiming(const char *label)
{
#ifndef BB_DC
	DumpDetailedTiming(label);
#else
	DTRACE("%s: ov(%.02lf) xtalk(%.02lf) cnc(%.02lf) rnc(%.02lf) pop(%.02lf) zero(%.02lf) comp(%.02lf) incomp(%.02lf) extractVect(%.02lf) wr(%.02lf)\n",
			__FUNCTION__,timing.overall,timing.xtalk,timing.ccn,timing.rcn,timing.populate,timing.SetMeanToZero+timing.SubtractMean,timing.CompressBlock,timing.inCompress,timing.extractVect,timing.write);
#endif

}

int AdvComprUnCompress(char *fname, short int *raw, int w, int h, int nImgPts,
		int *_timestamps, int startFrame, int endFrame, int mincols,
		int minrows, int maxcols, int maxrows, int ignoreErrors)
{
	int rc = 0;
#ifdef WIN32
	FILE_HANDLE fh;
#else
	FILE_HANDLE fh=-1;
#endif


	if (startFrame != 0 || endFrame != (nImgPts - 1) || mincols != 0
			|| maxcols != w || minrows != 0 || maxrows != h)
	{
		AdvComprPrintf("Allocating temporary storage\n");

		// we are getting a window....
		// allocate the full memory temporarily
		int frm, r, c;
		short int *rawP, *NewRawP;
		short int *NewRaw = (short int *) malloc(
				sizeof(short int) * w * h * nImgPts);
		int *timestamps = (int *) malloc(sizeof(int) * nImgPts);
		AdvCompr advc(fh, NewRaw, w, h, nImgPts, 0,NULL,timestamps,0,fname,NULL);
		rc = advc.UnCompress();
		if (rc == 0)
		{
			// copy into output
			for (frm = startFrame; frm <= endFrame; frm++)
			{
				for (r = minrows; r < maxrows; r++)
				{
					rawP = raw
							+ (frm - startFrame) * (maxcols - mincols)
									* (maxrows - minrows)
							+ (r - minrows) * (maxcols - mincols);
					NewRawP = &NewRaw[frm * w * h + r * w + mincols];
					for (c = mincols; c < maxcols; c++)
					{
						*rawP++ = *NewRawP++;
					}
				}
			}
			memcpy(_timestamps, timestamps + startFrame,
					sizeof(int) * (endFrame - startFrame + 1));
		}
		free(timestamps);
		free(NewRaw);
	}
	else
	{
		AdvCompr advc(fh, raw, w, h, nImgPts, 0,NULL,_timestamps,0,fname,NULL);
		rc = advc.UnCompress();
	}

//	PCADumpTiming("Uncompress",&timing);

	return rc;
}

typedef struct {
	int timestamp;
	int numFrames;
}transitions_t;

void AdvCompr::TimeTransform(int *timestamps, int *newtimestamps,
		int frameRate, int &tmp_npts, int &tmp_nUncompImgPts)
{
	if(frameRate==15 || !timeTransform){
		// just copy timestamps to newtimestamps
		memcpy(newtimestamps,timestamps,tmp_npts*sizeof(timestamps[0]));
	}
	else{
		// first, find transition points.
		int nframes[tmp_npts];
		int prevTs=0;
		int prevNframes=0;
		int tr=0;
		transitions_t transitions[10];

#ifdef SHORTEN_P2_FRONT_PORCH
		for(int pt=0;pt<tmp_npts;pt++){
			nframes[pt] = (timestamps[pt]-prevTs + 2)/timestamps[0];
			prevTs = timestamps[pt];
			prevNframes=nframes[pt];
		}
		if(nframes[1] == 8 && nframes[2] == 8 && nframes[3] == 8 && nframes[4] == 8){
			int oldts = timestamps[4];
			for(int j=1;j<=4;j++){
				timestamps[j] = timestamps[0]*(1+4*j);
			}
			for(int j=5;j<tmp_npts;j++){//now make all the other timestamps line up
				timestamps[j] -= oldts - timestamps[4];
			}
		}
#endif
		prevTs=0;
		prevNframes=0;
		for(int pt=0;pt<tmp_npts;pt++){
			nframes[pt] = (timestamps[pt]-prevTs + 2)/timestamps[0];
			if(nframes[pt] != prevNframes && prevNframes){
				transitions[tr].timestamp = prevTs;
				transitions[tr++].numFrames = prevNframes;
			}
			prevTs = timestamps[pt];
			prevNframes=nframes[pt];
		}
		transitions[tr].timestamp = prevTs;
		transitions[tr++].numFrames = prevNframes;

		// now, build a new array of timestamps based on the transition points.
		int cur_ts=0;
		int new_ts=0;
		int new_fr = 1000000/15; // in usecs
		int nts=0;
		int nuncomp=0;
		for(int ts_tr=0;ts_tr<tr;ts_tr++){
			do{
				new_ts = cur_ts + new_fr*transitions[ts_tr].numFrames;
				cur_ts = new_ts;
				newtimestamps[nts++] = cur_ts/1000; // back to msecs
				nuncomp += transitions[ts_tr].numFrames;
			}while(new_ts < (1000*transitions[ts_tr].timestamp));
		}
		tmp_nUncompImgPts=nuncomp;
		tmp_npts=nts;
	}
}

void AdvCompr::TimeTransform_trace(int npts_oldfr, int npts_newfr, int *timestamps_oldfr,
		int *timestamps_newfr, float *vectors, float * nvectors, int nvect)
{
	// fill in nvectors with the data from vectors.
	// interpolate the values
	for(int vect=0;vect<nvect;vect++){
		for(int pt=0;pt<npts_newfr;pt++){
			// first, find which two points this new value is between
			int prevTs=0;
			int curTs=0;
			int oldpt=0;
			for(;oldpt<npts_oldfr;oldpt++){
				if(timestamps_oldfr[oldpt] >= timestamps_newfr[pt]){
					curTs = timestamps_oldfr[oldpt];
					break;
				}
				prevTs=timestamps_oldfr[oldpt];
			}
			if(oldpt == npts_oldfr){ // hit the end
				nvectors[npts_newfr*vect+pt] = vectors[vect*npts_oldfr + npts_oldfr-1];
			}
			else{
				// we should now have the two points to interpolate..
				float weight = (float)((curTs-prevTs) - (timestamps_newfr[pt]-prevTs))/(float)(curTs-prevTs);
				float prev=0;
				if(oldpt)
					prev = vectors[vect*npts_oldfr + oldpt-1];
				else
					prev = vectors[vect*npts_oldfr + oldpt];
				float next=vectors[vect*npts_oldfr + oldpt];
				nvectors[npts_newfr*vect+pt] = ( prev-next ) * weight + next;
			}
		}
	}
}
void AdvCompr::xtalkCorrect(float xtalk_fraction)
{
    float denominator = 1.0f/(1.0f-(2.f*xtalk_fraction));
    v8f xtfV=LD_VEC8F(xtalk_fraction);
    v8f xtfVcomp=LD_VEC8F(1-xtalk_fraction);
    v8f denominatorV=LD_VEC8F(denominator);
    /*-----------------------------------------------------------------------------------------------------------*/
    // doublet xtalk correction - electrical xtalk between two neighboring pixels in the same column is xtalk_fraction
    //
    // Model is:
    // p1 = (1-xtalk_fraction)*c1 + xtalk_fraction * c2
    // p2 = (1-xtalk_fraction)*c2 + xtalk_fraction * c1
    // where p1,p2 - observed values, and c1,c2 - actual values. We solve the system for c1,c2.
    //
    // c1 = ((1-xtalk_fraction)*p1 - xtalk_fraction * p2)/(1 - 2*xtalk_fraction)
    // c2 = ((1-xtalk_fraction)*p2 - xtalk_fraction * p1)/(1 - 2*xtalk_fraction)
    /*-----------------------------------------------------------------------------------------------------------*/
	for(int vect=0;vect<hdr.nBasisVec;vect++){
	    int idx=0;
		for(int r=hdr.rstart;r<hdr.rend;r+=2,idx+=2*(hdr.cend-hdr.cstart)){
    		v8f *coefV1P=(v8f *)&TRC_COEFF_ACC(vect,idx);
    		v8f *coefV2P=(v8f *)&TRC_COEFF_ACC(vect,idx+(hdr.cend-hdr.cstart));
			for( int c=hdr.cstart; c<hdr.cend; c+=8,coefV1P++,coefV2P++ ){
        		v8f coefV1=*coefV1P;
        		v8f coefV2=*coefV2P;

                *coefV1P = (xtfVcomp*coefV1 - xtfV*coefV2) * denominatorV;
                *coefV2P = (xtfVcomp*coefV2 - xtfV*coefV1) * denominatorV;
    		}
    	}
    }
}

//Caution - this code is awaiting final P2 chip. It should be tested when valid data
//is available.
void AdvCompr::xtalkCorrect_raw(float xtalk_fraction, int w, int h, int npts, short int *raw)
{
    int phase = 0;//(raw->chip_offset_y)%2;
//    float denominator = (1-2*xtalk_fraction);
    int frameStride=w*h;
    /*-----------------------------------------------------------------------------------------------------------*/
    // doublet xtalk correction - electrical xtalk between two neighboring pixels in the same column is xtalk_fraction
    //
    // Model is:
    // p1 = (1-xtalk_fraction)*c1 + xtalk_fraction * c2
    // p2 = (1-xtalk_fraction)*c2 + xtalk_fraction * c1
    // where p1,p2 - observed values, and c1,c2 - actual values. We solve the system for c1,c2.
    /*-----------------------------------------------------------------------------------------------------------*/
    for( int f=0; f<npts; ++f ){
        for( int c=0; c<w; ++c ){
            for(int r=phase; r<h-1; r+=2 ){
                float p1_0 = raw[r*w+c];
                float p1 = raw[f*frameStride+r*w+c]-p1_0;
                float p2_0 = raw[(r+1)*w+c];
                float p2 = raw[f*frameStride+(r+1)*w+c]-p2_0;
                raw[f*frameStride+(r)*w+c]   = p1_0 + ((p1 - xtalk_fraction*p2)/(1.0f-xtalk_fraction));//((1.0f-xtalk_fraction)*p1-xtalk_fraction*p2)/denominator;
                raw[f*frameStride+(r+1)*w+c] = p2_0 + ((p2 - xtalk_fraction*p1)/(1.0f-xtalk_fraction));//((1.0f-xtalk_fraction)*p2-xtalk_fraction*p1)/denominator;
            }
        }
    }
}


void AdvCompr::ApplyGain_FullChip_xtalkcorr_sumRows(float xtalk_fraction, int w, int h, int npts, short unsigned int *raw)
{
	int frameStride=w*h;
	v8f_u tmpV1,tmpV2;
	v8s_u tmpVS1,tmpVS2;
	v8f_u tmpO1,tmpO2;
	v8f   sumV1,sumV2;
	short unsigned int *rtrc1,*rtrc2;
	v8f   xtf=LD_VEC8F(xtalk_fraction);
	v8f   oomxtf=LD_VEC8F(1.0f/(1.0f-xtalk_fraction));
	v8f_u rowSumCntrs1[npts];
	v8f_u rowSumCntrs2[npts];
	v8f_u maskV1,maskV2;

	for (int y=0;y<h;y+=2){//int itrc = 0; itrc < frameStride; itrc+=8,lgcV++)
		// zero the row sum counters for these two rows.
		memset(rowSumCntrs1,0,sizeof(rowSumCntrs1));
		memset(rowSumCntrs2,0,sizeof(rowSumCntrs2));

		for(int x=0;x<w;x+=8){
			 // load the first frame as our subtract point
			int itrc1=y*w+x;
			int itrc2=itrc1+w;
			rtrc1 = &raw[itrc1];
			rtrc2 = &raw[itrc2];
			LD_VEC8S_CVT_VEC8F(rtrc1, tmpV1);
			LD_VEC8S_CVT_VEC8F(rtrc2, tmpV2);
			sumV1 = tmpV1.V;
			sumV2 = tmpV2.V;

			v8f lgcV1 = *((v8f *) &gainCorr[itrc1]);
			v8f lgcV2 = *((v8f *) &gainCorr[itrc2]);
			maskV1 = *((v8f_u *) &mMask[itrc1]);
			maskV2 = *((v8f_u *) &mMask[itrc2]);

			for (int pt = 0; pt < npts; pt++,rtrc1+=frameStride,rtrc2+=frameStride)
			{

				LD_VEC8S_CVT_VEC8F(rtrc1, tmpV1);
				LD_VEC8S_CVT_VEC8F(rtrc2, tmpV2);

				// subtract off the static offset
				tmpV1.V -= sumV1;
				tmpV2.V -= sumV2;

				// do gain correction
				tmpV1.V *= lgcV1;
				tmpV2.V *= lgcV2;

				// now do xtalk correction
                tmpO1.V = ((tmpV1.V - xtf * tmpV2.V ) * oomxtf);//((1.0f-xtalk_fraction)*p1-xtalk_fraction*p2)/denominator;
                tmpO2.V = ((tmpV2.V - xtf * tmpV1.V ) * oomxtf);//((1.0f-xtalk_fraction)*p2-xtalk_fraction*p1)/denominator;


				// add back the offset
                tmpO1.V += sumV1;
                tmpO2.V += sumV2;

				// sum up the row averages..
                rowSumCntrs1[pt].V += (tmpO1.V * maskV1.V);
				rowSumCntrs2[pt].V += (tmpO2.V * maskV2.V);

				//  put it back
				CVT_VEC8F_VEC8S(tmpVS1,tmpO1);
				CVT_VEC8F_VEC8S(tmpVS2,tmpO2);
				*(v8s *)rtrc1 = tmpVS1.V;
				*(v8s *)rtrc2 = tmpVS2.V;
			}
		}

		v8f_u num_non_pinnedV1, num_non_pinnedV2;
		v8f *maskV1=(v8f *)&mMask[y*w];
		v8f *maskV2=(v8f *)&mMask[(y+1)*w];

		num_non_pinnedV1.V = LD_VEC8F(0);
		num_non_pinnedV2.V = LD_VEC8F(0);

		for(int x=0;x<(w/8);x++){
			num_non_pinnedV1.V += *maskV1++;
			num_non_pinnedV2.V += *maskV2++;
		}

		float num_non_pinned1 = V8_SUM(num_non_pinnedV1);
		float num_non_pinned2 = V8_SUM(num_non_pinnedV2);

		// populate mCorr_sigs
		// need number of non-pinned signals to divide by...
		for(int pt=0;pt<npts;pt++){
			mCorr_sigs[y*npts+pt]     = V8_SUM(rowSumCntrs1[pt])/num_non_pinned1;
			mCorr_sigs[(y+1)*npts+pt] = V8_SUM(rowSumCntrs2[pt])/num_non_pinned2;
		}
	}
}

void AdvCompr::ApplyGain_FullChip_xtalkcorr(float xtalk_fraction, int w, int h, int npts, short unsigned int *raw)
{
	int frameStride=w*h;
	v8f_u tmpV1,tmpV2;
	v8s_u tmpVS1,tmpVS2;
	v8f_u tmpO1,tmpO2;
	v8f   sumV1,sumV2;
	short unsigned int *rtrc1,*rtrc2;
	v8f   xtf=LD_VEC8F(xtalk_fraction);
	v8f   oomxtf=LD_VEC8F(1.0f/(1.0f-xtalk_fraction));
	v8f_u maskV1,maskV2;


	for (int y=0;y<h;y+=2){//int itrc = 0; itrc < frameStride; itrc+=8,lgcV++)

		for(int x=0;x<w;x+=8){
			 // load the first frame as our subtract point
			int itrc1=y*w+x;
			int itrc2=itrc1+w;
			rtrc1 = &raw[itrc1];
			rtrc2 = &raw[itrc2];
			LD_VEC8S_CVT_VEC8F(rtrc1, tmpV1);
			LD_VEC8S_CVT_VEC8F(rtrc2, tmpV2);
			sumV1 = tmpV1.V;
			sumV2 = tmpV2.V;

			v8f lgcV1 = *((v8f *) &gainCorr[itrc1]);
			v8f lgcV2 = *((v8f *) &gainCorr[itrc2]);
			maskV1 = *((v8f_u *) &mMask[itrc1]);
			maskV2 = *((v8f_u *) &mMask[itrc2]);

			for (int pt = 0; pt < npts; pt++,rtrc1+=frameStride,rtrc2+=frameStride)
			{

				LD_VEC8S_CVT_VEC8F(rtrc1, tmpV1);
				LD_VEC8S_CVT_VEC8F(rtrc2, tmpV2);

				// subtract off the static offset
				tmpV1.V -= sumV1;
				tmpV2.V -= sumV2;

				// do gain correction
				tmpV1.V *= lgcV1;
				tmpV2.V *= lgcV2;

				// now do xtalk correction
                tmpO1.V = ((tmpV1.V - xtf * tmpV2.V ) * oomxtf);//((1.0f-xtalk_fraction)*p1-xtalk_fraction*p2)/denominator;
                tmpO2.V = ((tmpV2.V - xtf * tmpV1.V ) * oomxtf);//((1.0f-xtalk_fraction)*p2-xtalk_fraction*p1)/denominator;


				// add back the offset
                tmpO1.V += sumV1;
                tmpO2.V += sumV2;

				//  put it back
				CVT_VEC8F_VEC8S(tmpVS1,tmpO1);
				CVT_VEC8F_VEC8S(tmpVS2,tmpO2);
				*(v8s *)rtrc1 = tmpVS1.V;
				*(v8s *)rtrc2 = tmpVS2.V;
			}
		}
	}
}


void AdvCompr::findPinned()
{

	uint16_t *fptr;
	uint16_t high = 16380;
	uint16_t low = 5;
	int pinned=0;
//	int pre_pinned=0;
//	int post_pinned=0;

//	for(int y=0;y<h;y++){
//		for(int x=0;x<w;x++){
//			if(mMask[y*w+x])
//				pre_pinned++;
//		}
//	}

	for(int idx=0;idx<w*h;idx++){
		mMask[idx]=1.0f;
	}

	for(int frame = 0;frame<npts;frame+=(npts-1)){ // first and last frame
		fptr = (uint16_t *)&raw[frame*h*w];
		for(int idx=0;idx<w*h;idx++){
				if(fptr[idx] > high || fptr[idx] < low){
					if(mMask[idx]){
						pinned++;
						mMask[idx] = 0;
					}
				}
		}
	}
//	printf("adv: pinned: %d\n",pinned);

//	for(int y=0;y<h;y++){
//		for(int x=0;x<w;x++){
//			if(mMask[y*w+x])
//				post_pinned++;
//		}
//	}
//	printf("%s: pre=%d post=%d\n",__FUNCTION__,pre_pinned,post_pinned);

}


// neighbor-subtract the comparator signals to get a noise correction
void AdvCompr::NNSubtractComparatorSigs(int span) {

    // Create an initial sum from 0 to span for each frame
    // Store the initial sums in an array
    int start_ry 	= std::max(0-span,0);
    int end_ry   	= std::min(0+span,h);
	float nn_sum[npts];
	float nn_avg 	= 0;
	for (int frame = 0; frame < npts; frame++) {
		nn_sum[frame] = 0;
		for (int ry=start_ry; ry<=end_ry; ry++) {
			nn_sum[frame] += SIGS_ACC(ry, frame);
		}
	}

	// For each y, shift the initial sum as needed
	// to avoid recalculating the whole thing
	for (int y = 0; y < h; y++) {
		int start_y = std::max(y-span,0);
		int end_y   = std::min(y+span,h);

		for (int frame = 0; frame < npts; frame++) {
			// We already calculated the sum when y == 0
			// as the initial values in nn_sum
			if (y>0) {
				if (y-span > 0) {
					nn_sum[frame] -= SIGS_ACC(start_y-1, frame);
				}
				if (y+span <= h) {
					nn_sum[frame] += SIGS_ACC(end_y, frame);
				}
			}
			nn_avg = nn_sum[frame];
			nn_avg /= (float)(end_y-start_y);
			float tmp = SIGS_ACC(y,frame) - nn_avg;
			if(tmp != tmp)
				tmp=0;
			NOISE_ACC(y,frame) = tmp;

		}
	}

#if 0
	// apply the correction right here..
	v8s *srcPtr = (v8s *)&raw[0];
	int frameStrideV=(h*w/8);

	for (int y = 0; y < h; y++) {
		for(int x=0;x<w;x+=8,srcPtr++){
			for (int frame = 1; frame < npts; frame++)
			{
				v8s corr=LD_VEC8S((short int)(NOISE_ACC(y,frame)));
				srcPtr[frame*frameStrideV] -= corr;
			}
		}
	}
#endif

}

// neighbor-subtract the comparator signals to get a noise correction
void AdvCompr::NNSubtractComparatorSigs_tn(int span)
{
	for (int y = 0; y < h; y++) {
		int start_y = std::max(y-span,0);
		int end_y   = std::min(y+span,h);
		// keep the ns within the 100x100 block

		start_y = y - y%100; // the beginning of the 100x100 block
		end_y=start_y+100;

		for (int frame = 0; frame < npts; frame++){
			float nn_avg=0;
			for(int ry=start_y;ry<end_y;ry++){
				nn_avg += SIGS_ACC(ry,frame);
			}
			nn_avg /= (float)(end_y-start_y);
			float tmp = SIGS_ACC(y,frame) - nn_avg;
			if(tmp != tmp)
				tmp=0;
			NOISE_ACC(y,frame) = tmp;
		}
	}
}


