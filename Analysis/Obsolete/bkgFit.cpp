/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Ion Torrent Systems, Inc.
// Command-line interface to background model
// (c) 2009
// $Rev: 6112 $
//	$Date: 2010-08-06 13:27:58 -0700 (Fri, 06 Aug 2010) $
//


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "Utils.h"
#include "BkgModel.h"

#define MAX_LINE_LEN 1048576

void processArgs(int argc, char *argv[], char **inFileName, char **outBaseName, char **flowOrder, int *nFrame);
int  GetNumCols(char *filename);
void readData(char *fileName, int nLines, int nWell, double **dat, double **bkg);
void writeMatrix(char *outFileName, float *data, int nRow, int nCol);

int main(int argc, char *argv[]) {
  char *inFileName = NULL;
  char *outBaseName = NULL;
  char *flowOrder = NULL;
  int nFrame = 0;
	
  // Parse command line arguments
  processArgs(argc,argv,&inFileName,&outBaseName,&flowOrder,&nFrame);

  // Determine number of flows, make sure it is a multiple of nFrame
  int nLines = GetNumLines(inFileName);
  if(0 != (nLines % nFrame)) {
    fprintf(stderr,"Number of lines in input file %s is %d, should be a multiple of nFrames %d\n",inFileName,nLines,nFrame);
    exit(EXIT_FAILURE);
  }
  int nFlow = nLines/nFrame;

  // Determine number of wells, make sure we have at least one to process
  int nCols;
  nCols = GetNumCols(inFileName);
  if(nCols < 2) {
    fprintf(stderr,"Number of columns in input file %s is %d, should be at least 2 (1 background + multiple wells)\n",inFileName,nCols);
    exit(EXIT_FAILURE);
  }
  int nWell = nCols-1;

  // Read the input data
  double *dat = NULL;
  double *bkg = NULL;
  readData(inFileName,nLines,nWell,&dat,&bkg);


  //
  // Fit background model
  //


    int my_nuc_block[NUMFB];
  // TODO: Objects should be isolated!!!!
  GlobalDefaultsForBkgModel::SetFlowOrder(flowOrder);
  GlobalDefaultsForBkgModel::GetFlowOrderBlock(my_nuc_block,0,NUMFB);
  InitializeLevMarSparseMatrices(my_nuc_block);

  float sigma_guess=1.2;
  float t0_guess=23;
  float dntp_uM=50.0;
  BkgModel *bkgmodel = new BkgModel(nWell,nFrame,sigma_guess,t0_guess,dntp_uM);

  // Iterate over flows
  struct bead_params p;
  struct reg_params rp;
  short *imgBuffer  = new short[nWell * nFrame];
  short *bkgBuffer  = new short[nFrame];
  float *out_sig    = new float[nWell*nFlow];
  float *out_sim_fg = new float[nWell*nFlow*nFrame];
  float *out_sim_bg = new float[nFlow*nFrame];
  memset(out_sim_fg,0,nWell*nFlow*nFrame);
  memset(out_sim_bg,0,nFlow*nFrame);
  int fitFramesPerFlow = 0;
  for(int iFlow=0, iFlowBatch=0; iFlow < nFlow; iFlow++) {
    //copy well data into imgBuffer and bkg data into bkgBufer
    int flowOffset = iFlow * nFrame;
    for(int iWell=0, iImg=0; iWell<nWell; iWell++)
      for(int iFrame=0; iFrame<nFrame; iFrame++, iImg++)
        imgBuffer[iImg] = (short) dat[iWell*nLines+flowOffset+iFrame];
    for(int iFrame=0; iFrame<nFrame; iFrame++)
      bkgBuffer[iFrame] = (short) bkg[iFlow*nFrame+iFrame];

    // Pass data to background model
    bool last_flow = (iFlow == (nFlow-1));
    bkgmodel->ProcessImage(imgBuffer,bkgBuffer,iFlow,last_flow,false);

    // If the model has been fit, store results
    if ((((1+iFlow) % NUMFB) == 0) || last_flow) {
      bkgmodel->GetRegParams(&rp);
      for (int iWell=0; iWell < nWell; iWell++) {
        bkgmodel->GetParams(iWell,&p);
        // Store estimated signal per well
        for (int iFlowParam=0, iFlowOut=iFlow-NUMFB+1; iFlowParam < NUMFB; iFlowParam++, iFlowOut++)
          out_sig[iWell*nFlow+iFlowOut]    = p.Ampl[iFlowParam]*p.Copies;
        // Simulate data from the fitted model
        float *fg,*bg,*feval,*isig,*pf;
        int tot_frms = bkgmodel->GetModelEvaluation(iWell,&p,&rp,&fg,&bg,&feval,&isig,&pf);
        fitFramesPerFlow = floor(tot_frms/(double)NUMFB);
        for (int iFlowParam=0, iFlowOut=iFlow-NUMFB+1; iFlowParam < NUMFB; iFlowParam++, iFlowOut++) {
          for (int iFrame=0; iFrame < fitFramesPerFlow; iFrame++) {
            out_sim_fg[iWell*nLines+iFlowOut*nFrame+iFrame] = fg[iFlowParam*fitFramesPerFlow+iFrame];
            if(iWell==0) 
              out_sim_bg[iFlowOut*nFrame+iFrame] = bg[iFlowParam*fitFramesPerFlow+iFrame];
          }
        }
      }
      iFlowBatch++;
    }
  }

  // Cleanup
  delete [] imgBuffer;
  delete [] bkgBuffer;
  delete bkgmodel;
   CleanupLevMarSparseMatrices();
  GlobalDefaultsForBkgModel::StaticCleanup(); 
  free(dat);
  free(bkg);

  // Write out results
  char *outFileSig = new char[strlen(outBaseName)+50];
  sprintf(outFileSig,"%s.intensityEstimate.txt",outBaseName);
  writeMatrix(outFileSig,out_sig,nFlow,nWell);
  delete [] out_sig;
  char *outFileSimFg = new char[strlen(outBaseName)+50];
  sprintf(outFileSimFg,"%s.fittedSignal.txt",outBaseName);
  writeMatrix(outFileSimFg,out_sim_fg,nFlow*nFrame,nWell);
  delete [] out_sim_fg;
  char *outFileSimBg = new char[strlen(outBaseName)+50];
  sprintf(outFileSimBg,"%s.fittedBkg.txt",outBaseName);
  writeMatrix(outFileSimBg,out_sim_bg,nFlow*nFrame,1);
  delete [] out_sim_bg;

  printf("\n");
  printf("Simulated data has %d frames per flow\n",fitFramesPerFlow);

  exit(EXIT_SUCCESS);
}

void writeMatrix(char *outFileName, float *data, int nRow, int nCol) {
  FILE *fp = fopen(outFileName, "wb");
  if (!fp) {
    perror (outFileName);
    exit (1);
  }
  for(int iRow=0; iRow<nRow; iRow++) {
    fprintf(fp,"%1.5f",data[iRow]);
    for(int iCol=1; iCol<nCol; iCol++)
      fprintf(fp,"\t%1.5f",data[iCol*nRow+iRow]);
    fprintf(fp,"\n");
  }
  fclose(fp);
}

void readData(char *fileName, int nLines, int nWell, double **dat, double **bkg) {
  *dat = (double *) malloc (sizeof(double) * nWell*nLines);
  *bkg = (double *) malloc (sizeof(double) * nLines);

  //Open the input file
  FILE *fp = fopen(fileName, "rb");
  if(!fp) {
    perror(fileName);
    exit(EXIT_FAILURE);
  }
  char lineBuf[MAX_LINE_LEN];
  char *line = lineBuf;
  int nChar = MAX_LINE_LEN;
  for(int iLine=0; iLine<nLines; iLine++) {
    int bytes_read = getline(&line,(size_t *)&nChar,fp);
    if(bytes_read < 0) {
      fprintf(stderr,"Problem reading line %d of %s\n",iLine+1,fileName);
      exit(EXIT_FAILURE);
    }
    int iCol=0;
    char *pch = strtok(line," \t");
    while (pch != NULL) {
      if(iCol>nWell) {
        fprintf(stderr,"problem parsing line %d of %s - found more than the expected %d entries.\n",iLine+1,fileName,nWell+1);
        exit(EXIT_FAILURE);
      } else if(iCol==0) {
        if(1 != sscanf(pch,"%lf",(*bkg)+iLine)) {
          fprintf(stderr,"problem parsing first field in line %d of %s\n",iLine+1,fileName);
          exit(EXIT_FAILURE);
        }
      } else {
        if(1 != sscanf(pch,"%lf",(*dat)+(iCol-1)*nLines+iLine)) {
          fprintf(stderr,"problem parsing field %d in line %d of %s\n",iCol+1,iLine+1,fileName);
          exit(EXIT_FAILURE);
        }
      }
      iCol++;
      pch = strtok(NULL, " \t");
    }
  }
  fclose(fp);

}

int GetNumCols (char *filename) {

  FILE *fp = fopen(filename,"rb");
  if(!fp) {
    perror(filename);
    return(-1);
  }
  char lineBuf[MAX_LINE_LEN];
  char *line = lineBuf;
  int nChar = MAX_LINE_LEN;
  int bytes_read = getline(&line,(size_t *)&nChar,fp);
  if(bytes_read < 0) {
    fprintf(stderr,"Error trying to read first line from %s\n",filename);
    exit(EXIT_FAILURE);
  }
  fclose(fp);
  
  int nCol=0;
  char *pch = strtok(line," \t");
  while (pch != NULL) {
    nCol++;
    pch = strtok(NULL, " \t");
  }

  return(nCol);
}

void processArgs (int argc, char *argv[], char **inFileName, char **outBaseName, char **flowOrder, int *nFrame) {
  int argcc = 1;
  while (argcc < argc) {
    if(argv[argcc][0] == '-') {
      switch (argv[argcc][1]) {
        case 'f':
          argcc++;
          if(1 != sscanf(argv[argcc],"%ul",nFrame)) {
            fprintf(stderr,"-f option should specify a positive integer\n");
            exit(EXIT_FAILURE);
          }
          break;

        case 'o':
          argcc++;
          *outBaseName = strdup(argv[argcc]);
          break;

        case 'r':
          argcc++;
          *flowOrder = strdup(argv[argcc]);
          break;
				
        default:
          fprintf (stderr, "Unknown option %s\n", argv[argcc]);
          exit(EXIT_FAILURE);
          break;
      }
    } else {
      *inFileName = strdup(argv[argcc]);
    }
    argcc++;
  }
	
  if (!*inFileName) {
    fprintf (stderr, "No input data file specified\n");
    exit(EXIT_FAILURE);
  } else if(*nFrame == 0) {
    fprintf (stderr, "Must specify a positive number of frames\n");
    exit(EXIT_FAILURE);
  }

  if(*flowOrder == NULL) {
    *flowOrder = strdup("TACG");
  } else {
    if(4 != strlen(*flowOrder)) {
      fprintf (stderr, "flowOrder must be of length 4\n");
      exit(EXIT_FAILURE);
    }
  }
  if(*outBaseName == NULL)
    *outBaseName = strdup("bkg");
}

