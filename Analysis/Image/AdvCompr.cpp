/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include <malloc.h>
#include <string.h>
#include "AdvCompr.h"
#ifndef WIN32
#include "PCACompression.h"
#include "PcaSpline.h"
//#include "DfcCompr.h"
#include "Utils.h"
#endif

// limits on the gain term we allow
#define ADV_COMPR_MIN_GAIN 0.85f
#define ADV_COMPR_MAX_GAIN 1.15f
#define ADV_COMPR_DEFAULT_GAIN 1.0f

// pinning defaults - do these need to change for different chips?
#define ADV_COMPR_PIN_HIGH 16380.0f
#define ADV_COMPR_PIN_LOW 5.0f

// high and low initial values 
#define ADV_COMPR_MAX_VAL 100000.0f
#define ADV_COMPR_MIN_VAL -100000.0f
float *AdvCompr::gainCorrection[ADVC_MAX_REGIONS] = {NULL};
char *AdvCompr::threadMem[ADVC_MAX_REGIONS] = {NULL};
uint32_t AdvCompr::threadMemLen[ADVC_MAX_REGIONS] = {0};

// constructor
AdvCompr::AdvCompr(FILE_HANDLE _fd, short int *_raw, int _w, int _h,
		int _nImgPts, int _nUncompImgPts, int *_timestamps_compr,
		int *_timestamps_uncompr, int _testUncompr, char *_fname, char *_options)
{
	ThreadNum=-1;
	GblMemPtr=NULL;
	GblMemLen=0;
    do_cnc = true;
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

	nBasisVectors=0;

	SplineStrategy[0]=0;
	options=_options;

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

// compress a memory image into a file
//   threadNum is used to keep static memory and avoid malloc calls for each invocation
int AdvCompr::Compress(int _threadNum, uint32_t region)
{
//	int num_pinned;
	double start = AdvCTimer();
//	double start1, start2;

	ThreadNum = _threadNum;
	ntrcs = ntrcsL = BLK_SZ;
	if (ntrcsL % (VEC8_SIZE * 4))
		ntrcsL = (ntrcs + (VEC8_SIZE * 4)) & (~((VEC8_SIZE * 4) - 1));

	if(region  < ADVC_MAX_REGIONS)
	{
		if (gainCorrection[region] == NULL)
		{
			gainCorrection[region] = (float *) memalign(VEC8F_SIZE_B,
					w * h * sizeof(float));
			for (int i = 0; i < w * h; i++)
				gainCorrection[region][i] = ADV_COMPR_DEFAULT_GAIN;
		}
		gainCorr = gainCorrection[region];
	}
	else
	{
		gainCorr = (float *) memalign(VEC8F_SIZE_B,
				w * h * sizeof(float));
	}

	if (fname && strstr(fname, "beadfind_pre_0003"))
		doGain = 1;
	else
		doGain = 0;

	ParseOptions(options);

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
	FileHdrV4.frames_in_file = BYTE_SWAP_2(npts);
	FileHdrV4.uncomp_frames_in_file = BYTE_SWAP_2(nUncompImgPts);
	FileHdrV4.interlaceType = BYTE_SWAP_2(7); // PCA
	FileHdrV4.sample_rate = BYTE_SWAP_4(15);
	if (fd >= 0)
		Write(fd, &FileHdrV4, sizeof(FileHdrV4));

	// write out timestamps
	if (fd >= 0)
		Write(fd, timestamps_compr, sizeof(timestamps_compr[0]) * npts);

	int total_iter = 0;

#ifndef PYEXT
        if (do_cnc) {
		double start2 = AdvCTimer();
		ComparatorNoiseCorrector cnc;
		cnc.CorrectComparatorNoise(raw, h, w, npts, NULL, 0, 1,
				false, ThreadNum);
		timing.ccn += AdvCTimer() - start2;
	}
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
			PopulateTraceBlock();

			SetMeanOfFramesToZero(npts);

			if (testType)
				doTestComprPre();

//				ComputeMeanTrace(); now done by setMeanOfFramesToZero
			t0est = findt0();


//			SubtractMeanFromTraces();

			for (int i = 0; i < npts; i++)
				mean_trc[i] += 8192; // add back some dc offset

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
	DumpTiming("Compress");

	return total_iter;
}


void AdvCompr::ParseOptions(char *options)
{
	// pcaSpline must be last option for now
//	const char *dflt_options="pcaReal=5,pcaFake=4,pcaSpline=0,SplineSt=no-knots";
	const char *pcaRealOpt="pcaReal=";     // real pca basis vectors
	const char *pcaFakeOpt="pcaFake=";     // made up pca basis vectors
	const char *pcaSplineOpt="pcaSpline="; // real pca basis vectors from spline code
	const char *SplineStOpt="SplineSt=";   // spline knot basis vector strategy
	const char *SplineTikOpt="SplineTik="; // spline knot basis vector strategy
	const char *SplineOrdOpt="SplineOrd=";
//	const char *DfcCoeffOpt="DfcCoeff=";  // number of dfc coefficients

	// set up the defaults
	strcpy(SplineStrategy,"no-knots");
	nPcaBasisVectors=5;
	nPcaFakeBasisVectors=4;
	nPcaSplineBasisVectors=0;
	nSplineBasisVectors=0;
	spline_order = -1;
	tikhonov_k = -1;
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

	sprintf(tstName, "%s_testPCA.dat", fnlptr);

#if 0
	if(AdvCompr::gainCorrection[region] == NULL && (strstr(_fname,"beadfind_pre_0003") == NULL))
	{
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
		strcpy(last_lptr,"beadfind_pre_0003.dat");
		else
		strcpy(tstName3, "beadfind_pre_0003.dat");
		printf("Loading Beadfind mask from %s\n",tstName3);
		Image img;

		img.LoadRaw(tstName3); // this will call AdvCompress for gain calculation
		img.Cleanup();
	}
#endif
	//dbgType 1  - replace raw with pca/unpca'd data
	//dbgType 2  - save residuals
	//dbgType 3  - save smoothed residuals
	//dbgType 4  - timeshift
	//dbgType 5  - just do gain correct and cnc.  then write the data back to raw
	//dbgType 6  - save corrected and zero'd traces
        //dbgType 7  - write out file and read back in again
	int fd = -1;
	if (/*dbgType != 1 && */(dbgType > 6))
		fd = open(tstName, O_WRONLY | O_CREAT | O_TRUNC, 0666);

//	if(fd >= 0 || dbgType == 1 || (dbgType >= 4 && dbgType <= 5))
	{
		if (fd >= 0)
			AdvComprPrintf("opened %s for writing fd=%d\n", tstName, fd);
		{
			AdvCompr advc(fd, raw->image, raw->cols, raw->rows,
					raw->frames, raw->uncompFrames, raw->timestamps, raw->timestamps, dbgType, tstName,options);
                        advc.SetDoCNC(do_cnc);
			advc.Compress(-1);
		}
		if (fd >= 0) {
                  close(fd);
                  fd = -1;
                }
		if (dbgType > 6) {
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

int AdvCompr::PopulateTraceBlock()
{
	int itrc = 0;
	v8f_u tmpV;
	int frameStride = w * h;
	short int *rtrc;
	int r, c, pt;
	double start = AdvCTimer();
	float meanPt;
	v8f_u lowV, highV,meanV;
	highV.V = LD_VEC8F(ADV_COMPR_PIN_HIGH);
	lowV.V = LD_VEC8F(ADV_COMPR_PIN_LOW);

	ntrcs = ntrcsL = (hdr.rend - hdr.rstart) * (hdr.cend - hdr.cstart);
	if (ntrcs % (VEC8_SIZE * 4))
		ntrcsL = (ntrcs + (VEC8_SIZE * 4)) & (~((VEC8_SIZE * 4) - 1));

	memset(trcs_state,0,trcs_state_len);
//	pinnedTrcs=0;
	for (pt = 0; pt < npts; pt++)
	{
		meanV.V=LD_VEC8F(0);
		itrc = 0;
		for (r = hdr.rstart; r < hdr.rend; r++)
		{
			c = hdr.cstart;
			v8f *lgcV = (v8f *) &gainCorr[r * w + c];
			rtrc = &raw[pt * frameStride + r * w + c];
			v8f *tPtrV = (v8f *) &TRCS_ACCESS(itrc,pt);
			v8f *pinnedV = (v8f *) &trcs_state[itrc];
			for (; c < hdr.cend;
					c += VEC8_SIZE,rtrc += VEC8_SIZE,pinnedV++, lgcV++, tPtrV++)
			{
				LD_VEC8S_CVT_VEC8F(rtrc, tmpV);

				// now, do the pinned pixel comparisons
#ifdef __AVX__
				*pinnedV += __builtin_ia32_cmpps256(tmpV.V , lowV.V, _CMP_LT_OS);
				*pinnedV += __builtin_ia32_cmpps256(highV.V, tmpV.V, _CMP_LT_OS);
#else
				for(int k=0;k<VEC8_SIZE;k++)
					trcs_state[itrc+k] += (tmpV.A[k] < lowV.A[k]) | (tmpV.A[k] > highV.A[k]);
#endif

				// gain correct
				tmpV.V *= *lgcV;
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
		//   AdvComprPrintf("MeanTrc = ");
		for (pt = 0; pt < npts; pt++)
		{
			mean_trc[pt] /= (float) cnt;
			//      AdvComprPrintf("%.02f ",mean_trc[i]);
		}
		//   AdvComprPrintf("\n");
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
		if (trcs_state[itrc] < 0)
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

void AdvCompr::SetMeanOfFramesToZero(int earlyFrames)
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
//	  PCACompr pca(nPcaBasisVectors, nPcaFakeBasisVectors,
//		       npts, ntrcs, ntrcsL, t0est,
//		       trcs, trcs_coeffs, trcs_state, basis_vectors);
	  PCACompr pca(nPcaBasisVectors, nPcaFakeBasisVectors,
		       npts, nSamples, nSamples, t0est,
		       sampleMatrix, trcs_coeffs, basis_vectors);
	  pca.Compress();
  	  timing.CompressBlock += AdvCTimer() - start;

	  ExtractVectors();
	}
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
//	  CreateSampleMatrix(hdr.nBasisVec);
//	  compressor.SampleImageMatrix(ntrcs, npts, trcs, trcs_state, sample_target_size, n_sample, &sample_mem);
	  compressor.LossyCompress(ntrcs, npts, trcs, nSamples, sampleMatrix, compressed);

  	  timing.CompressBlock += AdvCTimer() - start;
//	  NormalizeVectors();
//	  ExtractVectors();

	}


	return (total_iter);
}

void AdvCompr::NormalizeVectors()
{
	for(int nvect=0;nvect<hdr.nBasisVec;nvect++)
	{
		float ssq=0.0f;
		for (int i=0;i < npts;i++)
		   ssq += COEFF_ACCESS(i,nvect)*COEFF_ACCESS(i,nvect);

		if (ssq == 0.0f)
		{
		   // woah..it's not at all orthogonal to the vectors we already have...
		   // bail out....
			AdvComprPrintf("************* %s: vec %d has no magnitude *************\n",__FUNCTION__,nvect);
		}
		else
		{
			float tmag = sqrt(ssq);

			for (int i=0;i < npts;i++)
			   COEFF_ACCESS(i,nvect) /= tmag;
		}
	}
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
          if (*state_begin >= 0){
            good_wells++;
            *state_begin = hdr.nBasisVec; // update state to indicate the number of basis vectors now
          }
          state_begin++;
	}
//	int sample_rate = 16;//std::max(1, good_wells / (int)TARGET_SAMPLES)+1;
//	int n_sample_rows = (int) (ceil((float) good_wells / nSamples));
	int sample_rate=/*good_wells*/ntrcs/nSamples;
	for (int i = 0; i < npts; i++)
	{
		float *__restrict col_start = &TRCS_ACCESS(0,i);
		float *__restrict col_end = col_start + ntrcs;
//		state_begin = trcs_state;
		float *__restrict sample_col_begin = sampleMatrix + nSamples * i;
		uint32_t check_count = 0;
//		int good_count = sample_rate - 1;
		for(int nt=0;nt<ntrcs;nt+=sample_rate)
		{
//			if (/*state_begin[nt] >= 0 &&*/ ++good_count == sample_rate)
			{
				*sample_col_begin++ = *col_start;
//				good_count = 0;
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

void AdvCompr::ComputeEmphasisVector(float *gv, float mult, float adder, float width)
{
//	   float dt;
//	   const float width=10.0f;
	   float gvssq = 0.0f;
	   int t0estLocal=t0est+7;// needs to point to the middle of the incorporation

	   for (int i=0;i < npts;i++)
	   {
	      float dt=i-t0estLocal;
	      gv[i]=mult*exp(-dt*dt/width)+adder;
//	      ptgv[i]=COEFF_ACCESS(i,nvect)*gv[i];
	      gvssq += gv[i]*gv[i];
	   }
	   gvssq = sqrt(gvssq);

	  for (int i=0;i < npts;i++)
		 gv[i]/=gvssq;
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

//	ComputeEmphasisVector(gv, 2.0f, 1.4f, 10.0f);

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
			if (trcs_state[i] < 0)
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

		if (trcs_state[itrc] >= 0)
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


	AdvComprPrintf("replacing values with residuals (%d/%d) (%d/%d)\n",
			hdr.rstart, hdr.rend, hdr.cstart, hdr.cend);
	AdvComprPrintf("  1) %.02f %.01f %.01f\n", TRCS_ACCESS(0,0),
			TRCS_ACCESS(0,1), TRCS_ACCESS(0,2));
	for (r = hdr.rstart; r < hdr.rend; r++)
	{
		rowPtr = &raw[w * r];
		for (c = hdr.cstart; c < hdr.cend; c++)
		{
			for (pt = 0; pt < npts; pt++)
			{
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
	Write(fd, &hdr, sizeof(hdr));

	// write out the bits needed per vec
	Write(fd, bitsNeeded, sizeof(bitsNeeded[0]) * nvect);

	// write out the mean trace
	Write(fd, mean_trc, sizeof(float) * npts);

	// write out the pca vectors
	Write(fd, basis_vectors, sizeof(float) * nvect * npts);

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
	float mean_lowest_point;
	int itrc = 0;
	int pt, r, c;
	int tooHigh = 0;
	int tooLow = 0;
	int justRight = 0;
	int pinned = 0;
	int Nan = 0;
//	float saved_mean_trc[npts];

	PopulateTraceBlock();

	SetMeanOfFramesToZero(npts);

	t0est = findt0();

	// first, set mean of frames to zero
	for(pt=0;pt<npts;pt++)
	{
//		saved_mean_trc[pt] = mean_trc[pt];
		mean_trc[pt]=0;
	}

	SetMeanOfFramesToZero(t0est);

	ComputeMeanTrace();
//	// now, put the mean trace back
//	for(pt=0;pt<npts;pt++)
//		 mean_trc[pt] = saved_mean_trc[pt];

	mean_lowest_point = 0.0f;
	for (pt = 0; pt < npts; pt++)
	{
		if (mean_trc[pt] < mean_lowest_point)
			mean_lowest_point = mean_trc[pt];
	}

	if (mean_lowest_point == 0.0f)
	{
		AdvComprPrintf("mean_lowest_point= 0.0; t0est=%d ",t0est);
		for (pt = 0; pt < npts; pt++)
			AdvComprPrintf("%.0lf ",mean_trc[pt]);
		AdvComprPrintf("\n");
	}

	itrc = 0;
	// gain is ratio of mean to local trace
	for (r = hdr.rstart; r < hdr.rend; r++)
	{
		c = hdr.cstart;
		float *gcP = &gainCorr[r * w + c];
		for (; c < hdr.cend; c++,gcP++)
		{
			if (trcs_state[itrc] >= 0) // don't do pinned pixels
			{
				lowest_point = 1.0f;
				for (pt = 0; pt < npts; pt++)
				{
					if (TRCS_ACCESS(itrc,pt) < lowest_point)
						lowest_point = TRCS_ACCESS(itrc,pt);
				}
				*gcP = mean_lowest_point / lowest_point;
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
//	AdvComprPrintf("%s: tooHigh=%d tooLow=%d justRight=%d pinned=%d nan=%d\n",__FUNCTION__,tooHigh,tooLow,justRight,pinned,Nan);
}

void AdvCompr::doTestComprPre()
{
	if (testType == 2)
		SaveResidual(2);
	else if (testType == 3)
	{
		ComputeMeanTrace();
		t0est = findt0();

		//  	ClearFrontPorch();
		//    ComputeMeanTrace();

		SubtractMeanFromTraces();
//	    CheckTraceBlock();
		for (int i = 0; i < npts; i++)
			mean_trc[i] += 8192; // add back some dc offset

		SaveResidual(2);
	}
}

void AdvCompr::doTestComprPost()
{

	if (testType == 1)
	{
//			for (int i = 0; i < npts; i++)
//				mean_trc[i] += 8192; // add back some dc offset
//		for(int itrc=0;itrc<ntrcs;itrc++)
//		{
//			if(trcs_state[itrc]==0)
//				trcs_state[itrc]=(hdr.nPcaVec+hdr.nSpline);
//		}
		ExtractTraceBlock(); // extract into image
	}
	if (testType == 4)
	{
//		SetMeanOfFramesToZero(npts);
//		ComputeMeanTrace();
//		SubtractMeanFromTraces();
//		t0est = findt0();

		for (int i = 0; i < npts; i++)
			mean_trc[i] += 8192; // add back some dc offset

//		// Apply emphasis vector to residuals
//		ApplyEmphasisToTrcs();
		SaveResidual(0);
	}
	else if (testType == 5)
	{
		SetMeanOfFramesToZero(npts);
		ComputeMeanTrace();
		SubtractMeanFromTraces();
		t0est = findt0();

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
			if (slope1 != 0 && offset1 != 0 && slope2 != 0 && slope2 != 0
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

float AdvCompr::findt0()
{
	FILE *fp=NULL;
	float rc=-1;

	// UN_COMMENT for T0 debug
//	{
//		char name[256];
//		sprintf(name,"dbgT0_%d_%d.txt",hdr.rstart,hdr.cstart);
//		fp = fopen(name,"w+");
//	}

	rc = DCT0Finder(mean_trc, npts, fp);
	if (fp)
		fclose(fp);

	return rc;
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

					for (nv = 0; nv < hdr.nBasisVec; nv++)
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
						// check each one.. this should be the exception
						for (pt = 0; pt < hdr.npts; pt++)
							rtrc[pt * frameStride + k] = 0;
					}
				}
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
	AdvComprPrintf("  ccn           %.02f\n", timing.ccn);
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
	DTRACE("%s: ov(%.02lf) cnc(%.02lf) pop(%.02lf) zero(%.02lf) comp(%.02lf) incomp(%.02lf) extractVect(%.02lf) wr(%.02lf)\n",
			__FUNCTION__,timing.overall,timing.ccn,timing.populate,timing.SetMeanToZero+timing.SubtractMean,timing.CompressBlock,timing.inCompress,timing.extractVect,timing.write);
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



