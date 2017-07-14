/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PCACompression.h"


// each iteration computes vectors and returns them in coeff
// the first iteration starts with the random guesses
// but each subsequent iteration starts with the results of the
// previous regions' result

int PCACompr::Compress()
{
	int total_iter = 0;
	int nvect;
	int order = 3;
//   double start=PCATimer();

//	parent->hdr.nPcaVec = 0;

	for (int i = 0; i < npts; i++)
		for (int nvect = 0; nvect < (nRvect+nFvect); nvect++)
			COEFF_ACCESS(i,nvect) = /*((float) rand() / RAND_MAX) - 0.5f*/0.1
					* i - 10.0f;

	// at each step, trcs and coeff are modified during this process
	// after the final iteration, trcs will contain the residual error w.r.t. the PCA vectors
	// and coeff will contain the vectors
	for (nvect = 0; nvect < (nRvect+nFvect); nvect++)
	{
		if (nvect < nRvect)
			total_iter += ComputeNextVector(nvect,1);
		else
			ComputeOrderVect(nvect, order++);

		smoothNextVector(nvect,1);

		EnsureOrthogonal(nvect, (nvect < nRvect));

		SubtractVector(nvect, 1);

//		parent->hdr.nPcaVec = nvect + 1;
	}

//	for(int itrc=0;itrc<ntrcs;itrc++)
//	{
//		if(trcs_state[itrc]==0)
//			trcs_state[itrc]=nvect;
//	}
//   timing.CompressBlock += PCATimer()-start;
//	ExtractVectors((nRvect+nFvect),1);

   return(total_iter);
}

void PCACompr::smoothNextVector(int nvect, int blur)
{
	float trc[npts];

	for (int j=0;j < npts;j++)
	   trc[j] = COEFF_ACCESS(j,nvect);

	for (int j=0;j < npts;j++){
		float sum=0.0f;
		int start_frame = std::max(0,j-blur);
		int end_frame = std::min(npts,j+blur+1);
		for(int fr=start_frame;fr<end_frame;fr++){
			sum += trc[fr];
		}

		COEFF_ACCESS(j,nvect) = sum/(float)(end_frame-start_frame);

	}


}


int PCACompr::ComputeNextVector(int nvect, int skip)
{
   float tmag = 0.0f;
   float last_tmag = 0.0f;
   int iter=0;
   int retry=0;
   bool failed = true;
   float p[npts] __attribute__ ((aligned (VEC8F_SIZE_B)));
   float t[npts] __attribute__ ((aligned (VEC8F_SIZE_B)));
//   double start = PCATimer();
   float gv[npts];
   float ptgv[npts];

#if 1
   ComputeEmphasisVector(gv,1.0f,1.0f,10.0f);
#else
   for (int j=0;j < npts;j++)
	   gv[j]=1.0f;
#endif

   for (int j=0;j < npts;j++)
      p[j] = COEFF_ACCESS(j,nvect);

   while ((retry < 4) && failed)
   {
      failed = false;
      iter=0;
//      float end_threshold = 0.005f;
      float end_threshold = 0.00001f;

      do
      {
         last_tmag = tmag;

         for (int j=0;j < npts;j++)
            ptgv[j] = p[j]*gv[j];
         tmag = 0.0f;
         for (int j=0;j < npts;j++)
            tmag += ptgv[j]*ptgv[j];
         tmag = sqrt( tmag );
         for (int j=0;j < npts;j++)
            ptgv[j] = ptgv[j]/tmag;

         AccumulateDotProducts(ptgv,t,skip);

         tmag = 0.0f;
         for (int j=0;j < npts;j++)
            tmag += t[j]*t[j];
         tmag = sqrt( tmag );
         for (int j=0;j < npts;j++)
            p[j] = t[j]/tmag;

//         PCAPRINTF("%s: nv %d iter=%d tmag=%.04f %.04f\n",__FUNCTION__,nvect,iter,fabs(tmag - last_tmag)/tmag, tmag);
         iter++;
      }
      while (((fabs(tmag - last_tmag)/tmag > end_threshold) && (iter < 20)) && (!failed));

      if (failed)
      {
         for (int j=0;j < npts;j++)
         {
            p[j] = 0.1*j-10.0f;
            AdvComprPrintf("%f%c",p[j],(j==(npts-1))?'\n':',');
         }
      }

      retry++;
   }

   // zero the first several points of the vector
   // and make sure to re-normalize it
   int zpt=0; //5
   for (int j=0;j < zpt;j++)
	   COEFF_ACCESS(j,nvect) = p[zpt];
   for (int j=zpt;j < npts;j++)
	   COEFF_ACCESS(j,nvect) = p[j];

   tmag = 0.0f;
   for (int j=0;j < npts;j++)
      tmag += COEFF_ACCESS(j,nvect)*COEFF_ACCESS(j,nvect);
   tmag = sqrt( tmag );
   for (int j=0;j < npts;j++)
	   COEFF_ACCESS(j,nvect) = COEFF_ACCESS(j,nvect)/tmag;

//   timing.computeNext += PCATimer()-start;

//   PCAPRINTF("%s: nv %d iter=%d dc=%d   ",__FUNCTION__,nvect,iter,dc);
//   for (int j=0;j < npts;j++)
//	   PCAPRINTF("%.04f ",COEFF_ACCESS(j,nvect));
//   PCAPRINTF("\n");

   return(iter);
}

void PCACompr::ComputeEmphasisVector(float *gv, float mult, float adder, float width)
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
//		  gv[i]=1.0f;
}
float PCACompr::SubtractVector(int nvect, int skip)

{
   uint32_t ntrcsLV=ntrcsL/VEC8_SIZE;
	float ptgv[npts];
	float gv[npts];
	int pt;

	for (pt = 0; pt < npts; pt++)
		gv[pt] = 1.0f;

//	ComputeEmphasisVector(gv, 2.0f, 1.4f, 10.0f);
//		ComputeEmphasisVector(gv, 0.0f, 1.0f, 10.0f);

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
			ptgv[pt] = ((COEFF_ACCESS(pt,nvect)*gv[pt])/ssum);
	}


   for (int itrc=0;itrc < ntrcsL;itrc+=4*VEC8_SIZE*skip)
   {
	   {
		  v8f *sumPtr=(v8f *)&TRC_COEFF_ACC(nvect,itrc);
		  v8f *trcsUp = (v8f *)&TRCS_ACCESS(itrc,0);
		  v8f sumU0=LD_VEC8F(0);
		  v8f sumU1=LD_VEC8F(0);
		  v8f sumU2=LD_VEC8F(0);
		  v8f sumU3=LD_VEC8F(0);
		  for (int pt=0;pt < npts;pt++)
		  {
			  v8f ptgv_tmp = LD_VEC8F(ptgv[pt]/*COEFF_ACCESS(pt,nvect)*/);
			  sumU0 += trcsUp[0] * ptgv_tmp;
			  sumU1 += trcsUp[1] * ptgv_tmp;
			  sumU2 += trcsUp[2] * ptgv_tmp;
			  sumU3 += trcsUp[3] * ptgv_tmp;
			  trcsUp += ntrcsLV;
		  }
		  trcsUp = (v8f *)&TRCS_ACCESS(itrc,0);
		  for (int pt=0;pt < npts;pt++)
		  {
			  v8f ptgv_tmp = LD_VEC8F(ptgv[pt]/*COEFF_ACCESS(pt,nvect)*/);

			  trcsUp[0] -= sumU0 * ptgv_tmp;
			  trcsUp[1] -= sumU1 * ptgv_tmp;
			  trcsUp[2] -= sumU2 * ptgv_tmp;
			  trcsUp[3] -= sumU3 * ptgv_tmp;
			  trcsUp += ntrcsLV;
		  }
		  sumPtr[0] = sumU0;
		  sumPtr[1] = sumU1;
		  sumPtr[2] = sumU2;
		  sumPtr[3] = sumU3;
	   }
   }
   return 0;
}

// accumulate the sum of all traces
int PCACompr::AccumulateDotProducts(float *p, float *t, int skip)
{
   v8f sumU0,sumU1,sumU2,sumU3;
   v8f_u pU,pU0,pU1,pU2,pU3;
   v8f *trcsV;
   int pt;
   int k;
   int lw=ntrcsL/VEC8_SIZE;
//   double start = PCATimer();

  for (pt=0;pt < npts;pt++)
	  t[pt]=0.0f;

   for (int itrc=0;itrc < ntrcsL;itrc+=VEC8_SIZE*4*skip)
   {
	  sumU0=LD_VEC8F(0);
	  sumU1=LD_VEC8F(0);
	  sumU2=LD_VEC8F(0);
	  sumU3=LD_VEC8F(0);
	  trcsV=(v8f *)&TRCS_ACCESS(itrc,0);

	  for (pt=0;pt < npts;pt++,trcsV+=lw)
	  {
		  pU.V=LD_VEC8F(p[pt]);
		  sumU0 += (trcsV[0])*pU.V;
		  sumU1 += (trcsV[1])*pU.V;
		  sumU2 += (trcsV[2])*pU.V;
		  sumU3 += (trcsV[3])*pU.V;
	  }

	  trcsV=(v8f *)&TRCS_ACCESS(itrc,0);
      for (pt=0;pt < npts;pt++,trcsV+=lw)
      {
		  pU0.V=sumU0*(trcsV[0]);
		  pU1.V=sumU1*(trcsV[1]);
		  pU2.V=sumU2*(trcsV[2]);
		  pU3.V=sumU3*(trcsV[3]);

		  pU0.V+=pU1.V + pU2.V + pU3.V;
    	  for(k=0;k<VEC8_SIZE;k++)
    		  t[pt] += pU0.A[k];
      }
   }

//   parent->timing.Accumulate += PCATimer()-start;
   return 0;
}


void PCACompr::ComputeOrderVect(int nvect, int order)
{
//	int pvn = nvect-(NVECTS-NPVECTS);
    int porder = order+1;
//    float tmp[npts];

    // generate a polynomial vector of order 'porder'
    int zpt=8;  //8
    for (int i=0;i < zpt;i++)
    	COEFF_ACCESS(i,nvect) = 0.0f;
    for (int i=zpt;i < npts;i++)
    	COEFF_ACCESS(i,nvect) = pow((float)(i-npts/2),porder);
    EnsureOrthogonal(nvect,0);
}

void PCACompr::EnsureOrthogonal(int nvect, int check)
{
    // remove any part of it that is not orthogonal to the vectors we already have
    for (int ov=0;ov < nvect;ov++)
    {

       float sum=0.0f;
       for (int i=0;i < npts;i++)
          sum += COEFF_ACCESS(i,nvect)*COEFF_ACCESS(i,ov);

       for (int i=0;i < npts;i++)
    	   COEFF_ACCESS(i,nvect) -= COEFF_ACCESS(i,ov)*sum;
    }

    // normalize what's left
    float ssq=0.0f;
    for (int i=0;i < npts;i++)
       ssq += COEFF_ACCESS(i,nvect)*COEFF_ACCESS(i,nvect);

    if (ssq == 0.0f)
    {
       // woah..it's not at all orthogonal to the vectors we already have...
       // bail out....
    	AdvComprPrintf("************* polynomial vector not orthogonal to PCA vectors *************\n");
    }

    float tmag = sqrt(ssq);

    for (int i=0;i < npts;i++)
       COEFF_ACCESS(i,nvect) /= tmag;

}







