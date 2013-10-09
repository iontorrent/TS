

#include "xtalk_sim.h"
#include "DiffEqModel.h"
#include "utils.h"

FILE* CreateImageFile(const char *fname,int rows,int cols)
{
	FILE *fptr;
	
	fptr = fopen(fname,"wb");
	fwrite(&rows,1,4,fptr);
	fwrite(&cols,1,4,fptr);
	
	return(fptr);
}

FILE *CreateWellLogFile(const char *fname,DiffEqModel *pModel)
{
	FILE *fptr;
	SignalAverager *ha = pModel->head_avg;
	int num_wells = 0;
	
	fptr = fopen(fname,"wb");
	
	while(ha != NULL)
	{
		num_wells++;
		ha = ha->GetNext();
	}

	fwrite(&num_wells,1,4,fptr);
	
	ha = pModel->head_avg;
	
	while(ha != NULL)
	{
		int row,col;
		row = ha->GetRow();
		col = ha->GetCol();
		
		fwrite(&row,1,4,fptr);
		fwrite(&col,1,4,fptr);
		
		ha = ha->GetNext();
	}
	
	return(fptr);
}

void CaptureBottom(FILE *fptr,DiffEqModel *pModel,int k)
{
	for (int j=0;j < pModel->nj;j++)
	{
		for (int i=0;i < pModel->ni;i++)
		{
			float val = (float)(pModel->cmatrix[i+j*pModel->ni+k*pModel->ni*pModel->nj]);
			fwrite(&val,1,sizeof(float),fptr);
		}
	}
}

void CaptureBottom(const char *fname,DiffEqModel *pModel,int k)
{
	FILE *bfile = CreateImageFile(fname,pModel->nj,pModel->ni);
	
	CaptureBottom(bfile,pModel,k);
	
	fclose(bfile);
}

void CaptureSide(FILE *fptr,DiffEqModel *pModel,int j)
{
	for (int k=0;k < pModel->nk;k++)
	{
		for (int i=0;i < pModel->ni;i++)
		{
			float val = (float)(pModel->cmatrix[i+j*pModel->ni+k*pModel->ni*pModel->nj]);
			fwrite(&val,1,sizeof(float),fptr);
		}
	}
}

void CaptureSide(const char *fname,DiffEqModel *pModel,int j)
{
	FILE *sfile = CreateImageFile(fname,pModel->nk,pModel->ni);
	
	CaptureSide(sfile,pModel,j);
	
	fclose(sfile);
}

void CaptureWellData(FILE *well_log,DiffEqModel *pModel,int k)
{
	SignalAverager *ha = pModel->head_avg;
	float single_simTime = (float)(pModel->simTime);
	
	fwrite(&single_simTime,1,sizeof(float),well_log);
	while(ha != NULL)
	{
		float well_avg = (float)(ha->GetAverage(&pModel->cmatrix[k*pModel->ni*pModel->nj]));
		fwrite(&well_avg,1,sizeof(float),well_log);
		ha = ha->GetNext();
	}
}

void CaptureWellData(FILE *well_log,DiffEqModel *pModel,int k,BoundaryClass class_mask)
{
	SignalAverager *ha = pModel->head_avg;
	float single_simTime = (float)(pModel->simTime);
	
	fwrite(&single_simTime,1,sizeof(float),well_log);
	while(ha != NULL)
	{
		float well_avg = (float)(ha->GetAverage(&pModel->cmatrix[k*pModel->ni*pModel->nj],&pModel->boundary_class[k*pModel->ni*pModel->nj],class_mask));
		fwrite(&well_avg,1,sizeof(float),well_log);
		ha = ha->GetNext();
	}
}

int main(int argc,char *argv[])
{
	DiffEqModel *pModel;
	int ni,nj,nk;
	int wellz;
	DATA_TYPE hexpitch,well_fraction;
	DATA_TYPE wash_buffer_capacity;
	Timer prof_timer;
	FILE *side_log;
	FILE *bot_log;
	FILE *well_log[13];
	FILE *well_top_log;
	int side_img_slice_location;
	int bottom_img_slice_location;
		
	ni = 320;	// this needs to be divisible by 16 for vector math to work!
	nj = 192;
	nk = 174;
//	ni = 80;	// this needs to be divisible by 16 for vector math to work!
//	nj = 80;
//	nk = 80;
//	ni = 80;	// this needs to be divisible by 16 for vector math to work!
//	nj = 80;
//	nk = 63;
	side_img_slice_location = 97;
//	side_img_slice_location = 36;
	bottom_img_slice_location = 0;
	wellz = 13;
	hexpitch = 1.68f;
	well_fraction = 0.82f;

	wash_buffer_capacity = 0.004f;
//	wash_buffer_capacity = FLT_MAX/2.0f;
	
	pModel = new DiffEqModel(ni,nj,nk);

	// warning...the hex well pattern looks like crap unless these are chosen
	// carefully in relation to the value of hexpitch
	pModel->dx = 0.2078f;
	pModel->dy = 0.21f;
	pModel->dz = 0.1f;
	pModel->dcoeff = (5.28E-5)*((1E+6)/100)*((1E+6)/100);	// OH-
//	pModel->dcoeff = (5.28E-6)*((1E+6)/100)*((1E+6)/100);	// dNTP (guesstimate)

	DATA_TYPE min_sz = (pModel->dx < pModel->dy) ? pModel->dx : pModel->dy;
	min_sz = (min_sz < pModel->dz) ? min_sz : pModel->dz;
	
	// dt is the time step we will take
	// this should be chosen such that dt*dcoeff/min([dx dy dz])^2 is less than
	// 1...probably an order of magnitude less....
	// dt is in seconds
	pModel->dt = 1.0f/(10.0f*pModel->dcoeff/(min_sz*min_sz));

	// initialize all the special stuff...boundary conditions, convection, buffering
	// initial conditions
	pModel->BuildHexagonalWellPattern(wellz,hexpitch,well_fraction);
	pModel->SetBufferEffect(wash_buffer_capacity);
	pModel->SetConvection(wellz);

//	for (int row=0;row < 24;row++)
//		pModel->SetSignalInWell(wellz,row,10,hexpitch,well_fraction,100.0f);

//	for (int i=70;i<75;i++)
//		for (int j=95;j<99;j++)
//			for (int k=21;k<25;k++)
//				pModel->cmatrix[i+j*ni+k*ni*nj]=100.0f;


	pModel->SetEdgeBoundaryClass();
	pModel->SetupIncorpSignalInjection(wellz,12,10,hexpitch,well_fraction);
//	pModel->SetupIncorpSignalInjection(wellz,4,4,hexpitch,well_fraction);
	
	// populate all wells with buffering like we get when beads are loaded into wells
	// to keep things simple, the buffering is added uniformly to the entire volume of each well
	DATA_TYPE avg_well_buffering = pModel->AddUpWellBuffering(wellz,12,10,hexpitch,well_fraction,wash_buffer_capacity);
//	DATA_TYPE avg_well_buffering = pModel->AddUpWellBuffering(wellz,4,4,hexpitch,well_fraction,wash_buffer_capacity);
	pModel->AddBeadBufferingToAllWells(wellz,avg_well_buffering*0.5f,wash_buffer_capacity);
	pModel->ConfigureBeadSize(1.25f/2.0f);
	pModel->SetupIncorporationSignalInjectionOnGPU();


//	for (int row=0;row < 24;row++)
//		for (int col=0;col < 50;col++)
//		{
//			if ((rand() % 100) < 50)
//				pModel->AddBeadBufferingToSingleWell(wellz,row,col,hexpitch,well_fraction,avg_well_buffering*0.5f,wash_buffer_capacity);
//		}


//	pModel->AddBeadBufferingToSingleWell(wellz,12,12,hexpitch,well_fraction,avg_well_buffering*0.3f,wash_buffer_capacity);
//	DATA_TYPE avg_well_buffering = pModel->AddUpWellBuffering(wellz,4,4,hexpitch,well_fraction,wash_buffer_capacity);
//	pModel->AddBeadBufferingToSingleWell(wellz,4,4,hexpitch,well_fraction,avg_well_buffering*0.5f,wash_buffer_capacity);
//	pModel->AddBeadBufferingToSingleWell(wellz,5,6,hexpitch,well_fraction,avg_well_buffering*0.75f,wash_buffer_capacity);
	
	// all the boundary conditions should be set-up before this call
	pModel->initializeModel();
	
	side_log = CreateImageFile("side_log.dat",pModel->nk,pModel->ni);
	bot_log = CreateImageFile("bot_log.dat",pModel->nj,pModel->ni);
	for (int i=0;i < 13;i++)
	{
		char wfname[512];
		sprintf(wfname,"well_log_%d.dat",i);
		well_log[i] = CreateWellLogFile(wfname,pModel);
	}
	well_top_log = CreateWellLogFile("well_top_log.dat",pModel);

	// puts the row/col of each well in the log file so we can understand the spatial arrangement of the traces
	
	prof_timer.restart();
	while (pModel->simTime < 4.0f)
	{
		DATA_TYPE endTime = (DATA_TYPE)((int)((pModel->simTime+0.0001)*10000+0.5))/10000;
	
		int ncalls=pModel->runModel(endTime);
		
		printf("sim time: %f, dt: %f, iter: %d, elapsed real time: %f, throughput %f\n",pModel->simTime,pModel->dt*1000000.0f,ncalls,prof_timer.elapsed(),
				(prof_timer.elapsed()/pModel->simTime)/3600.0f);
		CaptureBottom("bot_img.dat",pModel,bottom_img_slice_location);
		CaptureSide("side_img.dat",pModel,side_img_slice_location);

		CaptureBottom(bot_log,pModel,bottom_img_slice_location);
		CaptureSide(side_log,pModel,side_img_slice_location);
		
		CaptureWellData(well_log[0],pModel,0);
		fflush(well_log[0]);
		for (int i=1;i < 13;i++)
		{
			CaptureWellData(well_log[i],pModel,i,(BoundaryClass)(PX_BOUND | MX_BOUND |	PY_BOUND | MY_BOUND));
			fflush(well_log[i]);
		}
		
		CaptureWellData(well_top_log,pModel,wellz);
		fflush(bot_log);
		fflush(side_log);
		fflush(well_top_log);
	}
	
	pModel->shutdownModel();
	
	delete pModel;
	
	fclose(side_log);
	fclose(bot_log);
	for (int i=0;i < 13;i++)
		fclose(well_log[i]);
	fclose(well_top_log);
	
	return(1);
}

