/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "peakestimator.h"



double deter(double a[][LIMIT],int forder);
int chckdgnl(double array[][LIMIT],int forder);
double expGauss (double x, double b1, double c1);
double expDoubleGauss(double x, double b1, double b2, double c1, double c2);
void doubleGaussianfunc(double *p, double *x, int m, int n, void *data);
void jacDoubleGaussFunc(double *p, double *jac, int m, int n, void *data);
double hessDoubleGaussFunc(double *p, int m, int n);

void gaussianfunc(double *p, double *x, int m, int n, void *data);
void jacgaussfunc(double *p, double *jac, int m, int n, void *data) ;
void get_freq( int array[], int length, int first_pos, double ** freq) ;


using namespace std;



/* structure for passing user-supplied data to the objective function and its Jacobian */
struct xtradata{  int start; };

double deter(double a[][LIMIT],int forder)
{
  int i,j,k;
  double mult;
  double deter=1;
  for(i=0;i<forder;i++)
  {
	for(j=0;j<forder;j++)
	{
	  mult=a[j][i]/a[i][i];
	  for(k=0;k<forder;k++)
	  {
		if(i==j) break;
		a[j][k]=a[j][k]-a[i][k]*mult;
	  }
	}
  }
  for(i=0;i<forder;i++)
  {
	deter=deter*a[i][i];
  }
  return(deter);
}


int chckdgnl(double array[][LIMIT],int forder)
{
  int i,j,k = 0;
  //double nonzero;
  for(i=0;i<forder;i++)
  {
	 if(array[i][i]==0)
	 {
		for(j=0;j<forder;j++)
		{
		  if(array[i][j]!=0)
		  {
			 k=j;
			 break;
		  }
		  if(j==(forder)) //forder-1
			 return(0);
		}
		for(j=0;j<forder;j++)
		{
		  array[j][i]=array[j][i]-array[j][k];
		}
	 }
  }
  return(1);
}




double expGauss (double x, double b1, double c1) {
//	double retVal = 0;
	return exp(-pow((x-b1),2)/(2*c1*c1));
}

double expDoubleGauss(double x, double b1, double b2, double c1, double c2) {
	return exp(-pow((x -(b1+b2)),2)/(2*(pow(c1,2) + pow(c2,2))));
	
}

void doubleGaussianfunc(double *p, double *x, int m, int n, void *data) {
struct xtradata *dat;
dat = (struct xtradata *)data;
int startPos = dat->start;
register int i;
for (i = 0; i < n; ++i) {
	//x[i] = (p[0]*p[1])/(sqrt(pow(p[0],2) + pow(p[1],2))) * expDoubleGauss(i, p[2], p[3], p[4], p[5]);
	x[i] = p[0] * expGauss(i+startPos, p[2], p[4]) + p[1] * expGauss(i+startPos, p[3], p[5]);

}

}

void jacDoubleGaussFunc(double *p, double *jac, int m, int n, void *data) {
struct xtradata *dat;
dat = (struct xtradata *)data;
int startPos = dat->start;

register int i, j;
for (i=j=0; i < n; ++i) {


	jac[j++] = expGauss(i+startPos, p[2], p[4]);
	jac[j++] = expGauss(i+startPos, p[3], p[5]);
	jac[j++] = p[0] * expGauss(i+startPos, p[2], p[4]) * ( (i+startPos - p[2])/( pow(p[4],2))) ;
	jac[j++] = p[1] * expGauss(i+startPos, p[3], p[5]) * ( (i+startPos - p[3])/( pow(p[5],2))) ;
	jac[j++] = p[0] * expGauss(i+startPos, p[2], p[4]) * ( pow((i+startPos-p[2]),2)/(pow(p[4],3)));
	jac[j++] = p[1] * expGauss(i+startPos, p[3], p[5]) * ( pow((i+startPos-p[3]),2)/(pow(p[5],3)));
	
}

}

double hessDoubleGaussFunc(double *p, int m, int n) {
	double retValue = 0;
	double hessm[m][m];
	register int i;
	for (i = 0; i < n; ++i) {
		hessm[0][0] += 	0;
		hessm[0][1] +=  0;
		hessm[0][2] += (i-p[2])/pow(p[4],2) * expGauss(i, p[2], p[4]);
		hessm[0][3] += 0;
		hessm[0][4] += pow((i-p[2]),2)/pow(p[4],3) * expGauss(i, p[2], p[4]);
		hessm[0][5] += 0;
		
		hessm[1][0] += 0;
		hessm[1][1] += 0;
		hessm[1][2] += 0;
		hessm[1][3] += (i-p[3])/pow(p[5],2) * expGauss(i, p[3], p[5]);
		hessm[1][4] += 0;
		hessm[1][5] += pow((i-p[3]),2)/pow(p[5],3) * expGauss(i, p[3], p[5]);
		
		hessm[2][0] += (i - p[2])/pow(p[4],2) * expGauss(i, p[2], p[4]);
		hessm[2][1] += 0;
		hessm[2][2] += -1 * p[0] * (pow(p[4],2) - pow((p[2] - i),2)) / pow(p[4],4) * expGauss(i, p[2], p[4]);
		hessm[2][3] += 0;
		hessm[2][4] += -1 * p[0] * (p[2] - i) * (pow((p[2]-i),2) - 2 * pow(p[4],2)) / pow(p[4],5) * expGauss(i, p[2], p[4]);
		hessm[2][5] += 0;
		
		hessm[3][0] += 0;
		hessm[3][1] += (i - p[3])/pow(p[5],2) * expGauss(i, p[3], p[5]);
		hessm[3][2] += 0;
		hessm[3][3] += -1 * p[1] * (pow(p[5],2) - pow((p[3] - i),2)) / pow(p[5],4) * expGauss(i, p[3], p[5]);
		hessm[3][4] += 0;
		hessm[3][5] += -1 * p[1] * (p[3] - i) * (pow((p[3]-i),2) - 2 * pow(p[5],2)) / pow(p[5],5) * expGauss(i, p[3], p[5]);
		
		hessm[4][0] += pow((p[2]-i),2) / pow(p[4],3) * expGauss(i, p[2], p[4]);
		hessm[4][1] += 0;
		hessm[4][2] += -1 * p[0] * (p[2] - i) * (pow((p[2]-i),2) - 2 * pow(p[4],2)) / pow(p[4],5) * expGauss(i, p[2], p[4]);
		hessm[4][3] += 0;
		hessm[4][4] += p[0] * pow((p[2]-i),2) * (pow((p[2]-i),2) - 3 * pow(p[4],2)) / pow(p[4],6) * expGauss(i, p[2], p[4]);
		hessm[4][5] += 0;
		
		hessm[5][0] += 0;
		hessm[5][1] += pow((p[3]-i),2) / pow(p[5],3) * expGauss(i, p[3], p[5]);
		hessm[5][2] += 0;
		hessm[5][3] += -1 * p[1] * (p[3] - i) * (pow((p[3]-i),2) - 2 * pow(p[5],2)) / pow(p[5],5) * expGauss(i, p[3], p[5]);
		hessm[5][4] += 0;
		hessm[5][5] += p[1] * pow((p[3]-i),2) * (pow((p[3]-i),2) - 3 * pow(p[5],2)) / pow(p[5],6) * expGauss(i, p[3], p[5]);
				
	}
	/*
	if(chckdgnl(hessm,m)==0)
	 retValue=0;
	else
	 retValue=deter(hessm,m);
	*/
	retValue = 0;
	return retValue;

}

void gaussianfunc(double *p, double *x, int m, int n, void *data) {
struct xtradata *dat;
dat = (struct xtradata *)data;
int startPos = dat->start;
register int i;
for (i = 0; i < n; i++) {
	x[i] = p[0] * expGauss(i+startPos, p[1], p[2]);
	
}
}

void jacgaussfunc(double *p, double *jac, int m, int n, void *data) {
register int i, j;
struct xtradata *dat;
dat = (struct xtradata *)data;
int startPos = dat->start;

for (i=j=0; i < n; ++i) {
	jac[j++] = expGauss(+startPos, p[1], p[2]);
	jac[j++] = p[0] * expGauss(i+startPos, p[1], p[2]) * ( (i+startPos - p[1])/( pow(p[2],2))) ;
	jac[j++] = p[0] * expGauss(i+startPos, p[1], p[2]) * ( pow((i+startPos -p[1]),2)/(pow(p[2],3)));
}

}



// Input:
//
// _x - array of size n distribution of flow-intensities
//  n - array size
// _params - array with input parameters
// [0] = hp_min_allele_frequency x 100  (def 15)
// [1] = hp_min_dist_peaks_short        (def 85)
// [2] = hp_min_dist_peaks_long         (def 85)
// [3] = hp_long_size                   (def  8)
// [4] = hp_peak_max_deviation          (def 35)
// [5] = hp_theta_lms                   (def 15, where 1E-15)
// [6] = hp_call_em                     (1)
// [7] = hp_theta_em                    (def 5, 1E-5)
//
// start - start index in the _x array 
// end - last index in the _x array
// refPolymer - ref candidate size
// varPolymer - var candidate size
// *************************************************************
// Output:
//
// returnValue[0] = probSinglePeak;float
// returnValue[1] = uniModalMean;  in [0,1199]
// returnValue[2] = probTwoPeaks;  float
// returnValue[3] = firstPeakMean; in [0,1199]
// returnValue[4] = secondPeakMean;in [0,1199] 
// returnValue[5] = firstPeakFreq; in [0,1]
// returnValue[6] = secondPeakFreq;in [0,1]
// returnValue[7] = singlePeakSTD;
// returnValue[8] = firstPeakSTD;
// returnValue[9] = secondPeakSTD;
// returnValue[10] = LMS (0) or EM (1)
  
  //std::cout << "inAPI call: " << returnValue[0] << " " << returnValue[1] << " " << returnValue[2] << " " << returnValue[3] << " " << returnValue[4] << " " << returnValue[5]<< " " << returnValue[6] << endl;
 

void runLMS ( int* _x, int n, int* inparams, float* returnValue, int start, int end, int refPolymer, int varPolymer, int DEBUG)
{

 //float * returnValue = new float[13];
 for (int counter = 0; counter < 13; counter++)
 	 returnValue[counter] = 0;
 
 int *x = _x;
 int *params = inparams;

 
 
int m = 3;
double p[m], pv[m], opts[LM_OPTS_SZ], info[LM_INFO_SZ], infov[LM_INFO_SZ];
double lb[m], ub[m];
double q[6];
double lbdb[6], ubdb[6];

double varThreshold[12] = {18.0, 18.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 63.0}; //{14.0, 18.0, 26.0, 34.0, 42.0, 52.0, 78.0, 90.0, 100.0, 100.0, 100.0, 100.0};
struct xtradata data; 
register int i, j;
int numElements = end - start;
int totCoverage = 0;
int totNonZeroElements = 0;
double minIntensityForSmallPeak = params[0] / 100.0; //minimum intensity for small peak (minimum allele frequency)
int minFlowPeakDist = params[2]; // minflow peak distance for long hp
if (refPolymer < params[3]) //if hp length is smaller than hp_long_size 
	minFlowPeakDist = params[1]; // minflow peak distance for short hp
int maxPeakDeviation = params[4]; // maximum deviation for peak from 100 (default 35)
bool callEM = false;
bool lowStringency = false;
if(params[6] == 1) {
	callEM = true;
	lowStringency = true;
}
int calledByLMS = false;
int calledByEM = false;
double strandBias = params[8]/100.0;

//cout << "Finished assigning params " << strandBias << " " << refPolymer << " " << start << " " << end  << " n = " <<n << endl;
//cout << "x[0]  = " << x[0] << " " << x[200] << endl;
//cout << "Calling get_freq with " << endl;
double* yval = new double[1200];

get_freq(_x, n, 0, &yval); 

//cout << "finished getting freq " << endl;
//cout << "MinIntensity for small peak = " << minIntensityForSmallPeak << endl;
double probRefPeak = 0;
double probVarPeak = 0; 
double prob1 = 0;
double prob2 = 0;
double uniModalMean = 0;
double uniModalStd = 0;
double uniModalPeakDistance = 0;
bool isUniModalReference = true;

double biModalRefIntensity = 0.0;
double biModalVarIntensity = 0.0;
double biModalRefMean = 0;
double biModalVarMean = 0;
double biModalRefStd = 0;
double biModalVarStd = 0;
double biModalPeakSep = 0.0;

int peakOneHomLen = 0;
int peakTwoHomLen = 0;
bool isBothPeaksRef = false;
bool isBothPeaksVar = false;
bool uniModalConverged = false;
bool biModalConverged = false;
bool isFinalCallHomReference = false;
bool isFinalCallHomVariant = false;
bool isFinalCallHetVariant = false;

double rawProb1 = 0;
double rawProb2 = 0;
double residual = 0;
  
// optimization control parameters; passing to levmar NULL instead of opts reverts to defaults 
  opts[0]=LM_INIT_MU; opts[1]=1E-15; opts[2]=1E-10; opts[3]=1E-10;
  opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used 

   
 /* initial parameters estimate:  */
  if (refPolymer == 0)
	refPolymer = 1;
	
  p[0]=0.01; p[1]=refPolymer*100; p[2]=5.0;
   pv[0]=0.01; pv[1]=varPolymer*100; pv[2]=5.0;
  lb[0] = 0.0001; lb[1] = 10; lb[2] = 1.0; 
  ub[0] = 1.0; ub[1] = MAXSIGDEV; ub[2] =  max(100, refPolymer*6); 
  
  data.start = start;
  //assign a new vector with trimmed values
  double z[numElements];
  if (DEBUG)
  	  cout << "x[i] = " << start << endl;
  for (i = 0; i < numElements; i++)  {
	z[i] = yval[i+start];
	if (DEBUG)
		cout << x[i+start] << "," ;
	if (x[i+start] != 0) {
		totCoverage += x[i+start];
		totNonZeroElements += x[i+start];
	}
	
  }
  if (DEBUG)
  	  cout << endl;

 // cout << "finished assigning z values " << endl;
 // cout << "Total NonZero Elements = " << totNonZeroElements << endl;
  int *zLow = new int[totNonZeroElements];
  int counter = 0;
  for (i = 0; i < numElements; i++) {
	for (j = 0; j < x[i+start]; j++)
		zLow[counter++] = i+start;
		//cout << i << "," << zLow[counter-1] << endl;
  }
  
   //cout << "finished assigning zlow values " << endl;
  if(totCoverage < 50) {
  	callEM = true;
  }

  if(callEM==true) {
  	double* result_1 = getLogLikelihood(zLow, totNonZeroElements, refPolymer*100, 1);
  	double* result_2 = getLogLikelihood(zLow, totNonZeroElements, refPolymer*100, 2);
  	prob1 = 0;
  	prob2 = 0;
  	if((result_2[6] < (result_1[6])) && (result_2[7] > 2) && (fabs(result_2[0] - result_2[3]) >=  minFlowPeakDist) && (fabs(result_2[2]-result_2[5])<0.90)) //lower BIC of 2-component model suggests 2 peaks
  	{
  		prob2 = 0.6;
  		prob1 = 0.4;
  		biModalConverged = true;
  		q[0] = result_2[2]; //amplitude or mixing probability of peak1
  		q[1] = result_2[5]; //amplitude or m  p of peak2
  		q[2] = result_2[0]; // mean of peak1
  		q[3] = result_2[3]; //mean of peak2
  		q[4] = sqrt(result_2[1]); // std dev of peak1
  		q[5] = sqrt(result_2[4]); // std dev of peak2
  		uniModalConverged = true;
  		uniModalMean = result_1[0]; //
  		uniModalStd = sqrt(result_1[1]);
  	}
  	else if(result_1[6] < result_2[6]){
  		uniModalConverged = true;
  		uniModalMean = result_1[0]; //
  		uniModalStd = sqrt(result_1[1]);
  		prob1 = 0.8;
  		prob2 = 0.2;
  	}
  	else {
  		uniModalConverged = true;
  		uniModalMean = result_1[0];
  		uniModalStd = sqrt(result_1[1]);
  		prob1 = 0.6;
  		prob2 = 0.4;
  	}
  	calledByEM = true;
  	free(result_1);
  	free(result_2);
   }
  else {
	calledByLMS = true;

  dlevmar_bc_dif(gaussianfunc, p, z, m, numElements, lb, ub, NULL, 1000, opts, info, NULL, NULL, (void *) &data);
  dlevmar_bc_dif(gaussianfunc, pv, z, m, numElements, lb, ub, NULL, 1000, opts, infov, NULL, NULL, (void *) &data);
  if (DEBUG)
  	  cout << "finished unimodal fit : info[6] = " << info[6] << " infov[6] = " << infov[6]  << " " << pv[1] << endl;

  if (info[6] != 4 && info[6] != 7) 
	probRefPeak = 1/info[1];
  if (infov[6] != 4 && infov[6] != 7)
	probVarPeak = 1/infov[1];

  if (probRefPeak >= probVarPeak) {
	prob1 = probRefPeak;
	uniModalMean = p[1];
	uniModalStd = p[2];
  } 
  else {
	prob1 = probVarPeak;
	uniModalMean = pv[1];
	uniModalStd = pv[2];
  }
  if(prob1 > 0 && (info[6] != 3 || infov[6] != 3))
	uniModalConverged = true; 
  if (DEBUG)
  	  cout << "UniModelConverged = " << uniModalConverged << endl;
  
  m = 6;
  

  q[0] = 0.01; q[1] = 0.01;  q[4] = 10.0; q[5] = 10.0;
  q[2] = refPolymer * 100.0;
  q[3] = varPolymer * 100.0;
  
  lbdb[0] = 0.0009; lbdb[1] = 0.0009; lbdb[2] = 50; lbdb[3] = 50;  lbdb[4] = 1.0; lbdb[5] = 1.0;
  ubdb[0] = 1.0; ubdb[1] = 1.0; ubdb[2] = MAXSIGDEV;  ubdb[3] = MAXSIGDEV; ubdb[4] = max(80, refPolymer*6); ubdb[5] = max(100, refPolymer*6);
  if (refPolymer > 6) {
	opts[2]=1E-12; opts[3]=1E-12;
  }
  else {
	opts[2]=1E-15; opts[3]=1E-15;
  }
  
  dlevmar_bc_dif(doubleGaussianfunc, q, z, m, numElements, lbdb, ubdb, NULL, 5000, opts, info, NULL, NULL, (void *) &data);

  int numTries = 0;
  	 while (info[6] == 3 && numTries <= 3) {
  	 	 opts[2] *= 100; //lower the convergence threshold and try again
  	 	 opts[3] *= 100;
         dlevmar_bc_dif(doubleGaussianfunc, q, z, m, numElements, lbdb, ubdb, NULL, 1000, opts, info, NULL, NULL, (void *) &data);
  	 	 numTries++;
  }

 if(info[6] != 4 && info[6] != 7 && info[6] != 3)
	biModalConverged = true;
  
 if (biModalConverged) { 
	prob2 = 1/info[1];
 }
 if (DEBUG)
 	 cout << "biModalConverged = " << biModalConverged << endl;
 
}
 if(uniModalConverged || (prob1>0 && !calledByEM)) { //even if uniModal didnt converge we can get a chi-sq from the last iteration and use that to scale the prob
	if (fabs(uniModalMean - refPolymer*100) >= fabs(uniModalMean - varPolymer*100))
		isUniModalReference = false;
	
	uniModalPeakDistance = abs((int)uniModalMean - (int)floor(uniModalMean/100 + 0.5)*100);
 }

 if(biModalConverged) {
	peakOneHomLen = (int)floor(q[2]/100 + 0.5);
	peakTwoHomLen = (int)floor(q[3]/100 + 0.5);
	
	if (peakOneHomLen == peakTwoHomLen && peakOneHomLen == refPolymer) {
		isBothPeaksRef = true;
		biModalRefMean = q[2]; //it doesn't which one we pick for reference as both peak point to reference allele
		biModalRefStd = q[4];
		biModalRefIntensity = calledByEM?q[0]:(q[0]*q[4])/(q[0]*q[4] + q[1]*q[5]); 
		biModalVarMean = q[3];
		biModalVarStd = q[5];
		biModalVarIntensity = calledByEM?q[1]:(q[1]*q[5])/(q[0]*q[4] + q[1]*q[5]);
	}
	else if (peakOneHomLen == refPolymer ) { //check if either of the peaks match ref allele
		biModalRefMean = q[2];
		biModalRefStd = q[4];
		biModalRefIntensity = calledByEM?q[0]:(q[0]*q[4])/(q[0]*q[4] + q[1]*q[5]);
		biModalVarMean = q[3];
		biModalVarStd = q[5];
		biModalVarIntensity = calledByEM?q[1]:(q[1]*q[5])/(q[0]*q[4] + q[1]*q[5]);
	}
	else if (peakTwoHomLen == refPolymer) {
		biModalRefMean = q[3];
		biModalRefStd = q[5];
		biModalRefIntensity = calledByEM?q[1]:(q[1]*q[5])/(q[0]*q[4] + q[1]*q[5]);
		biModalVarMean = q[2];
		biModalVarStd = q[4];
		biModalVarIntensity = calledByEM?q[0]:(q[0]*q[4])/(q[0]*q[4] + q[1]*q[5]);
	}
	else if ( (peakOneHomLen == varPolymer) && (peakTwoHomLen == varPolymer) ) {
		isBothPeaksVar = true;
		if (abs(peakOneHomLen-(varPolymer)) <= abs(peakTwoHomLen-(varPolymer))) {
			biModalRefMean = q[3]; //just set one to ref does not matter which
			biModalVarMean = q[2];
			biModalRefStd = q[5];
			biModalVarStd = q[4];
			biModalRefIntensity = calledByEM?q[1]:(q[1]*q[5])/(q[0]*q[4] + q[1]*q[5]);
			biModalVarIntensity = calledByEM?q[0]:(q[0]*q[4])/(q[0]*q[4] + q[1]*q[5]);
		}
		else {
			biModalRefMean = q[2];
			biModalVarMean = q[3];
			biModalRefStd = q[4];
			biModalVarStd = q[5];
			biModalRefIntensity = calledByEM?q[0]:(q[0]*q[4])/(q[0]*q[4] + q[1]*q[5]);
			biModalVarIntensity = calledByEM?q[1]:(q[1]*q[5])/(q[0]*q[4] + q[1]*q[5]);
		}
	}
	else { //not sure what could be happening converged on a peak that we are not expecting
		if (abs(peakOneHomLen-varPolymer) <= abs(peakTwoHomLen-varPolymer) ) {
			biModalRefMean = q[3]; //just set one to ref
			biModalVarMean = q[2];
			biModalRefStd = q[5];
			biModalVarStd = q[4];
			biModalRefIntensity = calledByEM?q[1]:(q[1]*q[5])/(q[0]*q[4] + q[1]*q[5]);
			biModalVarIntensity = calledByEM?q[0]:(q[0]*q[4])/(q[0]*q[4] + q[1]*q[5]);
		}
		else {
			biModalRefMean = q[2];
			biModalVarMean = q[3];
			biModalRefStd = q[4];
			biModalVarStd = q[5];
			biModalRefIntensity = calledByEM?q[0]:(q[0]*q[4])/(q[0]*q[4] + q[1]*q[5]);
                        biModalVarIntensity = calledByEM?q[1]:(q[1]*q[5])/(q[0]*q[4] + q[1]*q[5]);

		}
	}
	
	//calculate the seperation between two peaks
	biModalPeakSep = fabs(q[2]-q[3]);
	if (biModalRefStd == ubdb[5] || biModalRefStd == lbdb[5] || biModalVarStd == ubdb[5] || biModalVarStd == lbdb[5]) {
		prob2 = 0; //if the peak converged on either  upper or lower bounds of variance allowed, NEED TO CLOSELY WATCH THIS NOT TO CAUSE FN
		biModalConverged = false;
	}
	
  }
  
  
 //calcalate the residual i.e observations that are outside the fit. If the residual is greater than a given threshold we should reject the fit.
 if (uniModalConverged && !biModalConverged) {
 	int min2sigma = (int) floor(uniModalMean - 2*uniModalStd);
 	int max2sigma = (int) floor(uniModalMean + 2*uniModalStd);
 	
 	if (min2sigma < start)
 		min2sigma = start;
 	if (max2sigma > end)
 		max2sigma = end;
 	
 	int totalInFit = 0;
 	for (int counter = start; counter < end; counter++)
 		totalInFit += x[counter];
 	
 	if (totalInFit >= totCoverage)
 		residual = 0;
 	else
 		residual = (float)totalInFit/totCoverage;
 	 
 }
 else if (biModalConverged) {
 	 bool isMinRef = false;
 	 if (biModalRefMean < biModalVarMean)
 	 	 isMinRef = true;
 	 int min2sigma = 0;
 	 int max2sigma = 0;
 	 if (isMinRef) {
 	 	 min2sigma = (int) floor(biModalRefMean - 2*biModalRefStd);
 	 	 max2sigma = (int) floor(biModalVarMean + 2*biModalVarStd);
 	 }
 	 else {
 	 	max2sigma =  (int) floor(biModalRefMean + 2*biModalRefStd);
 	 	 min2sigma = (int) floor(biModalVarMean - 2*biModalVarStd);
 	 }
 	 
 	 if (min2sigma < start)
 		min2sigma = start;
 	 if (max2sigma > end)
 		max2sigma = end;
 	
 	 int totalInFit = 0;
 	 for (int counter = start; counter < end; counter++)
 		totalInFit += x[counter];
 	
 	 if (totalInFit >= totCoverage)
 		residual = 0;
 	 else
 		residual = (float)totalInFit/totCoverage;
 	 
 }
 
  //TO DO if both prob1 and prob2 == 0 , may we can try fitting one peak solution again with smaller TAU
  
  //Normalize the two probabilities before scaling them , atleast one hypothesis has converged
  if (uniModalConverged || biModalConverged) {
	if(calledByEM == false) {
		calledByLMS = true;
	} 
	rawProb1 = prob1/(prob1+prob2); //normalized prob1
	rawProb2 = prob2/(prob1+prob2); //normalized prob2
	//cout << "RawProb1 = " << rawProb1 << " RawProb2 = " << rawProb2 << endl;
	
	//if (!uniModalConverged) rawProb1 = 0; //set the probability of UniModal to zero after normalization
	
	if (uniModalConverged && !biModalConverged) // uniModal only biModal did not converge
	{
		if (!isUniModalReference) { //if unimodal is signalling a reference allele then there is no need to downscale the probabilities, just call a reference
			if (uniModalPeakDistance > maxPeakDeviation) { // if distance from 100 is greater than 35 ie. peak mean is >235 < 250 for a 2 mer variant, then it is most likely to be FP so just call reference
				rawProb1 = 0.5;//5050 probability  
				isFinalCallHomReference = true;
			}				
			else {
				rawProb1 = rawProb1 * ((50-uniModalPeakDistance)/50); //probability is scaled down by distance from 50.
				//apply variance penalty
				if (varPolymer < 11 && uniModalStd > varThreshold[varPolymer]) {
					double varDist = fabs(uniModalStd - varThreshold[varPolymer]);
					if (varDist >= 10)
						rawProb1 = 0.01;
					else
						rawProb1 = rawProb1 * (10 - varDist)/10;
				
				}
				isFinalCallHomVariant = true;
			}
			
		}
		rawProb2 = 0;
	
	}
	else if (!uniModalConverged  && biModalConverged) //biModal only uniModal did not converge
	{
		//uniModalProb can be nonzero even if it didnt converge so scale the prob if its non-zero
		if (rawProb1 > 0) {
			if (uniModalPeakDistance > maxPeakDeviation) { 
				rawProb1 = 0.01;  
			}			
			else {
				rawProb1 = rawProb1 * ((50-uniModalPeakDistance)/50); //probability is scaled down by distance from 50.
			}
		
		}
		
		if (isBothPeaksRef) {
			//nothing to do for now, not need to scale the prob
			isFinalCallHomReference = true;
		}
		else if (isBothPeaksVar) {
			//again no scaling required for allele freq or peak distance
			isFinalCallHomVariant = true;
		}
		else { //now scale
			if (biModalPeakSep < minFlowPeakDist) {
				rawProb2 = 0.01; //set to very small value such that they dont pass the score filter
				isFinalCallHomReference = true; //highly likely to be FP call so just go with Reference call
			}
			else {
				if(biModalPeakSep < 100)
					rawProb2 *= biModalPeakSep/100; //scale the prob by distance between two peaks
				isFinalCallHetVariant = true;
			}
			if (biModalRefIntensity < minIntensityForSmallPeak) {
				//isFinalCallHomVariant = true;
				rawProb1 = 0.9; //we think this is most likely a homozygous (unimodal) variant but check the distance from 50 and scale accordingly
				uniModalPeakDistance =  abs((int)biModalVarMean - (int)floor(biModalVarMean/100 + 0.5)*100);
				if (uniModalPeakDistance > maxPeakDeviation) {
				   rawProb1 = 0.01;
				   isFinalCallHomReference = true;
				   isFinalCallHomVariant = false;
				}
				else {
					 rawProb1 = rawProb1 * ((50-uniModalPeakDistance)/50); //probability is scaled down by distance from 50.
					 isFinalCallHomVariant = true;

				}
				rawProb2 = 0.01;
				uniModalStd = min(uniModalStd,biModalVarStd); // setting the std dev to biModal var std as uniModal peak can have higher std dev because of the presence of small ref peak.
			}
			if (biModalVarIntensity < minIntensityForSmallPeak) { //thereshold for variant is set to 15%, doesnt quite mean allele freq is 15%, it is just an estimate
				rawProb2 = 0.01;
				rawProb1 = 0.9;
				isFinalCallHomReference = true;
			}
			else if (biModalVarIntensity < 0.5) {
				if (lowStringency) {
						if (biModalVarIntensity < 0.25)
								rawProb2 = rawProb2 *(biModalVarIntensity/0.25);
				}
				else
						rawProb2 = rawProb2 *(biModalVarIntensity/0.50);

				isFinalCallHetVariant = true;
			}

			if (!isFinalCallHomReference && varPolymer < 11 && biModalVarStd > varThreshold[varPolymer] ) {
			 	double varDist = fabs(biModalVarStd - varThreshold[varPolymer]);
                        	if (varDist >= 10 ) {
                                	rawProb2 = 0.01;
                                	isFinalCallHomReference = true;
                        	}
                        	else  {
                                	rawProb2 = rawProb2 * (10-varDist)/10;
                        	}
			}
			
			//scale down for Strand Bias
			if (!isFinalCallHomReference) {
				if (isFinalCallHomVariant) {
					rawProb1 = rawProb1 * (1.0-strandBias); // if strandBias = 1 then probability of variant is 0.
				}
				else 
					rawProb2 = rawProb2 * (1.0-strandBias);
					
			}

				
		}
		
		//if uniModal points to variant and bimodal is het variant then just add the two probabilities as both cases point to variant
		if (!isUniModalReference && isFinalCallHetVariant)
			rawProb2 += rawProb1;
		if (!isFinalCallHomVariant && !isFinalCallHomReference) // check
			rawProb1 = 0.01;
		//finally adjust scale for Bayesian score
	
	}
	else if (uniModalConverged && biModalConverged) { //both converged
		//scale the biModal probability first
		if (isBothPeaksRef) {
			//leave the scaling as is but just to hom reference
			rawProb2 = 0;
			rawProb1 = 1;
			isFinalCallHomReference = true;
		}
		else if (isBothPeaksVar && !isUniModalReference) {
			rawProb2 = 0;
			rawProb1 = 1;
			isFinalCallHomVariant = true;
		}
		else  {
			if (biModalPeakSep < minFlowPeakDist) {
				rawProb2 = 0.01; //set to very small value such that they dont pass the score filter
				if (isUniModalReference) {
					isFinalCallHomReference = true;
					rawProb1 = 0.9;
				}
				else 
					isFinalCallHomVariant = true;
			}
			else {
				//cout << "Scaling by peak dist rawProb2 = " << rawProb2 << endl;
				rawProb2 *= biModalPeakSep/100; //scale the prob by distance between two peaks
				if (!isUniModalReference) {
					isFinalCallHetVariant = true;
				}
				//cout << "after rawProb2 = " << rawProb2 << endl;
			}
			
			if (biModalRefIntensity < minIntensityForSmallPeak) {
				isFinalCallHomVariant = true;
				//rawProb1 = 0.9; //CHANGE: Dont scale the unimodal probability
				rawProb2 = 0.01;
				uniModalStd = min(uniModalStd,biModalVarStd); // setting the std dev to biModal var std as uniModal peak can have higher std dev because of the presence of small ref peak.
			}
			if (biModalVarIntensity < minIntensityForSmallPeak) { //thereshold for variant is set to 15%, doesnt quite mean allele freq is 15%, it is just an estimate
				rawProb2 = 0.01;
				//rawProb1 = 0.9;  //CHANGE: Dont scale the unimodal probability
				isFinalCallHomReference = true;
			}
			else if (biModalVarIntensity < 0.5) {
				//cout << "Scaling by intensity rawProb2 = " << rawProb2 << endl;
				if (lowStringency) {
					if (biModalVarIntensity < 0.25)
						rawProb2 = rawProb2 *(biModalVarIntensity/0.25);
				}
				else {
						rawProb2 = rawProb2 * (biModalVarIntensity/0.50);
				}
				//isFinalCallHetVariant = true;
				//cout << " after rawProb2 = " << rawProb2 << endl;
			}
			
			
			if (!isFinalCallHomReference && varPolymer < 11 && biModalVarStd > varThreshold[varPolymer] ) {
						double varDist = fabs(biModalVarStd - varThreshold[varPolymer]);
						if (varDist >= 20 ) {
								rawProb2 = 0.01;
								//isFinalCallHomReference = true;
						}
						else  {
								rawProb2 = rawProb2 * (20-varDist)/20;
						}
						//cout << "After Scaling by std dev rawProb2 = " << rawProb2 << endl;
			}
			
			
		
		}
		
		//scale the uniModal Prob
		if (/*!isUniModalReference &&*/ !isFinalCallHomReference) {
			//cout << "Scaling unimodal by peak dist rawProb1 = " << rawProb1 << endl;
			if (uniModalPeakDistance > maxPeakDeviation) 
				rawProb1 = 0.01;//something very small so that it does not pass thru filters
			else
				rawProb1 = rawProb1 * ((50-uniModalPeakDistance)/50);
				
			//cout << "after peak dist rawProb1 = " << rawProb1 << endl;
			
			 if (varPolymer < 11 && uniModalStd > varThreshold[varPolymer]) {
                                double varDist = fabs(uniModalStd - varThreshold[varPolymer]);
                                if (varDist >= 20)
                                        rawProb1 = 0.01;
                                else
                                        rawProb1 = rawProb1 * (20 - varDist)/20;
                                
                               // cout << "Scaling by std dev rawProb1 = " << rawProb1 << endl;
                                
              }

			//if the both uniModal and biModal still point to varaint increase probability of var call
			if (rawProb1 > 0.1 && rawProb2 > 0.1 /*&& biModalVarIntensity > 0.5 */ ) {
					if (rawProb1 >= rawProb2)
						rawProb1 += rawProb2;
					else
						rawProb2 += rawProb1;
			}

			//cout << "Final rawProb = " << rawProb1 << "  " << rawProb2 << endl;

		}
		//cout << "Final rawProbs = " << rawProb1 << "  " << rawProb2 << endl;
		//finally if call still points to bimodal apply strand bias penalty
		if (rawProb2 > rawProb1) 
			rawProb2 = rawProb2 * (1.0-strandBias);
		//cout << "after stdbias rawProbs = " << rawProb1 << "  " << rawProb2 << endl;
	}
	
	//if residual is greater than a threshold set the probability of variant to low value; this is mainly used to identify noisy non-guassian flow signals
	if (residual > 0.20) {
		rawProb1 = 0.09;
		rawProb2 = 0.09;
		
	}
	
	//adjust the probabilities based on the variant call made
	if (isFinalCallHomReference) {
		rawProb1 = 1.0;
		returnValue[1] = refPolymer*100; 
	}
	else if (isFinalCallHomVariant) 
	{
		rawProb1 += 0.4;
		if (rawProb1 > 1.0) rawProb1 = 1.0;
		returnValue[1] = uniModalMean;
	}
	else {
		if (rawProb1 >= rawProb2) {
			rawProb1 += 0.4;
			if (rawProb1 > 1.0) rawProb1 = 1.0;
		}
		else {
			rawProb2 += 0.4;
			if (rawProb2 > 1.0) rawProb2 = 1.0;
		}
		returnValue[1] = uniModalMean;
	}
	
	returnValue[0] = rawProb1;
	returnValue[2] = rawProb2;
	returnValue[3] = q[2]; 
	returnValue[4] = q[3];
	returnValue[5] = (q[0]*q[4])/(q[0]*q[4]+q[1]*q[5]);
	returnValue[6] = (q[1]*q[5])/(q[0]*q[4]+q[1]*q[5]);
	
  }
  else {
	returnValue[0] = 0;
	returnValue[1] = 0;
	returnValue[2] = 0;
	returnValue[3] = 0;
	returnValue[4] = 0;
	returnValue[5] = 0;
	returnValue[6] = 0;
	
  }
  returnValue[7] = uniModalStd;
  returnValue[8] = biModalRefStd;
  returnValue[9] = biModalVarStd;
  returnValue[10] = -1;
  if(calledByLMS == true)
	returnValue[10] = 0;
  else if(calledByEM == true)
	returnValue[10] = 1;

  returnValue[11] = prob1;
  returnValue[12] = prob2;
  
  delete[] zLow;
  delete[] yval;
  
  //returnValue[1] = uniModalMean;  returnValue[3] = q[2]; returnValue[4] = q[3]; returnValue[5] = q[0]/(q[0]+q[1]); returnValue[6] = q[1]/(q[0]+q[1]);
  
  //std::cout << "inAPI call: " << returnValue[0] << " " << returnValue[1] << " " << returnValue[2] << " " << returnValue[3] << " " << returnValue[4] << " " << returnValue[5]<< " " << returnValue[6] << endl;

  //env->ReleaseFloatArrayElements(_returnValue, returnValue,0);
  
  
}

void get_freq( int array[], int length, int first_pos, double ** freq) 
{
	int sum = 0;
	register int j;
	//cout << "length= " << length << "firstpos =  " << first_pos << endl;
	for (j=first_pos; j < length; j++) {
		//cout << "j = " << j << " " << array[j] << endl;
		sum += array[j];
	}
		
	for (j=0; j < length; j++) {
		if (sum != 0)
			(*freq)[j] = (double)array[j]/sum;
		else
			(*freq)[j] = 0;
	}
		
	
}

double calculateNormalPDF(double mean, double variance, double x) {
	double dev = (x-mean);
	if(dev == 0.0)
		dev =0.0001;
	double exponentialTerm = pow(2.718, -0.5 * ( (x-mean) * (x-mean) / (variance) ) );

	if(exponentialTerm == 0)
		exponentialTerm = 0.0001;
	return ((1 / sqrt(2 * 3.1416 * variance)) * exponentialTerm);
}


void initPriors(double* means, double* variances, double* mixingProbs, int numComponents, double mean) {
	int m = numComponents / 2;
	means[m] = mean;
	int i=0;
	for(i=m-1;i>=0;i--) {
		means[i] = means[i+1] - 100;
	}
	for(i=m+1;i<numComponents;i++) {
		means[i] = means[i-1] + 100;
	}
	for(i=0; i< numComponents ; i++) {
		variances[i] = 20;
		mixingProbs[i] = 1.0 / numComponents;
	}
}

int runEstimateStep(int* flowSignals, double* means, double* variances, double* mixingProbs, double** weights, int n, int numComponents) {
	int i=0;
	int j=0;
	double threshold = 0.00001;
	int converged = 1; //true

	for(i=0;i<n; i++) {
		double denominator = 0;
		double* d = (double*) malloc(numComponents * sizeof(double));
		for(j=0;j<numComponents;j++) {
			d[j] = calculateNormalPDF(means[j], variances[j], flowSignals[i]) * mixingProbs[j];
			denominator = denominator + d[j];
		}
		//printf("denominator is %f \n", denominator);
		for(j=0; j<numComponents; j++) {
			double temp = d[j] / denominator;
			if( (fabs((weights[i][j] - temp))) > threshold) {
				converged = 0; //false
				//printf("weight is %d %d %f \n", i, j, temp);
			}
			weights[i][j] = temp;
		}
		free(d);

	}
	return converged;
}

void updateMixingProbs(double* mixingProbs, double** weights, int n, int numComponents) {
	int i=0;
	int j=0;
	double sum = 0;
	for(j=0;j<numComponents;j++) {
		sum = 0;
		for(i=0;i<n;i++) {
			//printf("%d %d %f \n", i, j, weights[i][j]);
			sum = sum + weights[i][j];
		}
		mixingProbs[j] = sum / n;
	}
}

void updateMeans(double* means, double** weights, int* flowSignals, int n, int numComponents) {
	int i=0;
	int j=0;
	double sum = 0;
	double mean = 0;
	for(j=0;j<numComponents;j++) {
		sum =0;
		mean = 0;
		for(i=0;i<n;i++) {
			sum = sum + weights[i][j];
		}
		if(sum == 0) sum = 0.0001;
		for(i=0;i<n;i++) {
			mean = mean + (weights[i][j] * flowSignals[i] / sum);
			//printf("at %d flow is %d  mean is %f and sum is %f \n", i, flowSignals[i], mean, sum);
		}

		means[j] = mean;
	}
}

void updateVariances(double* means, double* variances, double** weights, int* flowSignals, int n, int numComponents) {
	int i=0;
	int j=0;
	double variance = 0;
	double sum = 0;
	for(j =0;j<numComponents; j++) {
		sum = 0;
		variance = 0;
		for(i = 0; i<n; i++) {
			sum = sum + weights[i][j];
			variance = variance + (weights[i][j] * (flowSignals[i] - means[j]) * (flowSignals[i] - means[j]) );
		}
		//printf(" sum is %f \n", sum);
		if(sum == 0) sum = 0.0001;
		variances[j] = variance / (sum);
		//printf("variance of %d is %f \n", j, variances[j]);
		if(variances[j] == 0.0)
			variances[j] = 0.0001;
	}
}

void  runUpdateStep(int* flowSignals, double* means, double* variances, double* mixingProbs, double** weights, int n, int numComponents) {

	updateMixingProbs(mixingProbs, weights, n, numComponents);
	updateMeans(means, weights, flowSignals, n, numComponents);
	updateVariances(means, variances, weights, flowSignals, n, numComponents);
}

void runEM(int* flowSignals, double* means, double* variances, double* mixingProbs, double** weights, int n, int numComponents) {
	int converged = 0;
	int numIterations = 0;
	while(converged == 0 && numIterations <= 5000) { // runs until it converges
		converged = runEstimateStep(flowSignals, means, variances, mixingProbs, weights, n, numComponents);
		int j=0;
		for(j=0; j<numComponents; j++) {
			//printf("Component - %d \n", j);
			//	printf("Mean is %f , Variance is %f, mixing probability is %f \n", means[j], variances[j], mixingProbs[j]);
		}
		runUpdateStep(flowSignals, means, variances, mixingProbs, weights, n, numComponents);
		numIterations++;
	}

}

double calculateLogLikelihood(double* means, double* variances, double* mixingProbs, int* flowSignals, int n, int numComponents, double** weights) {

	int i=0;
	int j=0;
	double sumOfWeights = 0;
	double logEvidence = 0;
	for(i =0 ;i<n; i++) {
		sumOfWeights = 0;
		for(j=0;j<numComponents;j++) {
			sumOfWeights += calculateNormalPDF(means[j], variances[j], flowSignals[i]) * mixingProbs[j];
		}
		logEvidence += log(sumOfWeights);
	}
	return logEvidence;


}

/*
	output - double[]
	output[0] - Mean1
	output[1] - variance1
	output[2] - mixingProb1
	output[3] - Mean2
	output[4] - variance2
	output[5] - mixingProb2
	output[6] - BIC
	output[7] - Mean seperation metric for 2 components
*/
double* getLogLikelihood(int* flowSignals, int n, double mean, int numComponents) {

	double* result = (double*)malloc(8*sizeof(double));
	int s =0;
	for(s=0;s<8;s++)
		result[s] = -999;
	if(numComponents == 0)
		return result;
	mean = ceil( (mean / 100.0)) * 100;
	double* means = (double*) malloc(numComponents * sizeof(double));
	double* variances = (double*)malloc(numComponents * sizeof(double));
	double* mixingProbs = (double*)malloc(numComponents * sizeof(double));

	initPriors(means, variances, mixingProbs, numComponents, mean);
	double** weights = (double**)malloc(n * sizeof(double*));
	int i=0;
	int j=0;
	for(i =0;i<n; i++) {
		weights[i] = (double*)malloc(numComponents * sizeof(double));
		for(j=0;j<numComponents; j++)
			weights[i][j] = 0;
	}
	runEM(flowSignals, means, variances, mixingProbs, weights, n, numComponents);


	double d = calculateLogLikelihood(means, variances, mixingProbs, flowSignals, n, numComponents, weights);
	//printf("loglikelihood with this model is %f \n", d);
	int numParameters = 3*numComponents - 1;
	double bic = (-2 * d) + (2 * numParameters * log(n));

	result[0] = means[0];
	result[1] = variances[0];
	result[2] = mixingProbs[0];
	if(numComponents == 2) {
		result[3] = means[1];
		result[4] = variances[1];
		result[5] = mixingProbs[1];
		double separationTestMetric = fabs(means[0]-means[1]) / sqrt(0.5*(variances[0] + variances[1]));

		result[7] = separationTestMetric;
	}
	result[6] = bic;
	free(means);
	free(variances);
	free(mixingProbs);
	for(i=0;i<n;i++) {
		free(weights[i]);
	}
	return result;
}




