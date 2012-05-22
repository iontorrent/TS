/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include <sstream>
#include "BkgModelHdf5.h"

BkgParamH5::BkgParamH5() {
  amplDS = NULL;
  beadDCDS = NULL;
  beadDC_bgDS = NULL;
  kRateDS = NULL;
  resErrorDS = NULL;
  beadInitParamDS = NULL;
  darkOnceParamDS = NULL;
  darknessParamDS = NULL;
  emptyOnceParamDS = NULL;
  regionInitParamDS = NULL;
  regionOffsetDS = NULL;

  regionParamDS.clear();
  regionParamDSExtra.clear();
  emphasisParamDS.clear();
  beadFblk_avgErrDS.clear();
  beadFblk_clonalDS.clear();
  beadFblk_corruptDS.clear();
}


std::string BkgParamH5::Init(SystemContext &sys_context, SpatialContext &loc_context, int numFlows, ImageSpecClass &my_image_spec) {
    //wells_output_directory is a better choice than basecaller_output_directory
    //hgBgDbgFile = ToStr(sys_context.dat_source_directory) + "/bg_param.h5";
    //cout << "dat_source_directory:" << sys_context.dat_source_directory << endl;
    //cout << "basecaller_output_directory:" << sys_context.basecaller_output_directory << endl;
    //cout << "wells_output_directory:" << sys_context.wells_output_directory << endl;
    if (ToStr(sys_context.wells_output_directory)=="")
         hgBgDbgFile = sys_context.experimentName;
    else
        hgBgDbgFile = ToStr(sys_context.wells_output_directory);
    if (hgBgDbgFile[hgBgDbgFile.length()-1] != '/')
        hgBgDbgFile += '/';
    hgBgDbgFile += "bg_param.h5";
    cout << "hgBgDbgFile:" << hgBgDbgFile << endl;

    h5BgDbg.Init();
    h5BgDbg.SetFile(hgBgDbgFile);
    h5BgDbg.Open(true);

    int blocksOfFlow = NUMFB;
    int nFlowBlks = ceil(float(numFlows)/NUMFB);
    //int numParam = sizeof(reg_params)/sizeof(float); // +1 not necessary
    int numParam = sizeof(reg_params_H5)/sizeof(float); // +1 not necessary
    hsize_t rp_dims[2], rp_chunks[2];
    char s[128];
    string str;

    ///------------------------------------------------------------------------------------------------------------
    /// bead parameters
    ///------------------------------------------------------------------------------------------------------------
    try {
    beadDC.Init(loc_context.cols, loc_context.rows, numFlows);
    beadDC.SetRange(0,loc_context.cols, 0, loc_context.rows, 0, blocksOfFlow);
    beadDC.AllocateBuffer();
    beadDCDS = h5BgDbg.CreateDataSet("/bead/fg_bead_DC", beadDC, 3);
    h5BgDbg.CreateAttribute(beadDCDS->getDataSetId(),"description","bead_DC per flow");
    h5BgDbg.makeParamNames("fg_beadDC_",numFlows,str);
    h5BgDbg.CreateAttribute(beadDCDS->getDataSetId(),"paramNames",str.c_str());

    amplMultiplier.Init(loc_context.cols, loc_context.rows, numFlows);
    amplMultiplier.SetRange(0,loc_context.cols, 0, loc_context.rows, 0, blocksOfFlow);
    amplMultiplier.AllocateBuffer();
    amplDS = h5BgDbg.CreateDataSet("/bead/ampl_multiplier", amplMultiplier, 3);
    h5BgDbg.CreateAttribute(amplDS->getDataSetId(),"description","Amplitude-multiplier per flow");
    h5BgDbg.makeParamNames("ampl_",numFlows,str);
    h5BgDbg.CreateAttribute(amplDS->getDataSetId(),"paramNames",str.c_str());

    kRateMultiplier.Init(loc_context.cols, loc_context.rows, numFlows);
    kRateMultiplier.SetRange(0,loc_context.cols, 0, loc_context.rows, 0, blocksOfFlow);
    kRateMultiplier.AllocateBuffer();
    kRateDS = h5BgDbg.CreateDataSet("/bead/k_rate_multiplier", kRateMultiplier, 3);
    h5BgDbg.CreateAttribute(kRateDS->getDataSetId(),"description","K-rate-multiplier per flow");
    h5BgDbg.makeParamNames("kRate_",numFlows,str);
    h5BgDbg.CreateAttribute(kRateDS->getDataSetId(),"paramNames",str.c_str());

    bgResidualError.Init(loc_context.cols, loc_context.rows, numFlows);
    bgResidualError.SetRange(0,loc_context.cols, 0, loc_context.rows, 0, blocksOfFlow);
    bgResidualError.AllocateBuffer();
    resErrorDS = h5BgDbg.CreateDataSet("/bead/res_error", bgResidualError, 3);
    h5BgDbg.CreateAttribute(resErrorDS->getDataSetId(),"description","Residual-error per flow");
    h5BgDbg.makeParamNames("resErr_",numFlows,str);
    h5BgDbg.CreateAttribute(resErrorDS->getDataSetId(),"paramNames",str.c_str());

    beadInitParam.Init(loc_context.cols, loc_context.rows, 4);
    beadInitParam.SetRange(0,loc_context.cols, 0, loc_context.rows, 0, beadInitParam.GetNumZ());
    beadInitParam.AllocateBuffer();
    beadInitParamDS = h5BgDbg.CreateDataSet("/bead/bead_init_param", beadInitParam, 3);
    h5BgDbg.CreateAttribute(beadInitParamDS->getDataSetId(),"description","Bead-init-param, same for all flows, 4 data points: Copies, R, dmult, gain");
    h5BgDbg.CreateAttribute(beadInitParamDS->getDataSetId(),"paramNames","Copies, R, dmult, gain");

    beadFblk_avgErr.set_size(loc_context.cols, loc_context.rows);
    beadFblk_clonal.set_size(loc_context.cols, loc_context.rows);
    beadFblk_corrupt.set_size(loc_context.cols, loc_context.rows);

    rp_dims[0] = rp_chunks[0] = loc_context.cols;
    rp_dims[1] = rp_chunks[1] = loc_context.rows;
    for (int i=0; i<nFlowBlks; i++) {
      int f = (i+1)*NUMFB;
      if (i==nFlowBlks-1)
          f = numFlows;
      sprintf(s,"/bead/flowBlkParams/avgErr/flow_%04d",f);
      H5DataSet *ptr = h5BgDbg.CreateDataSet(s, 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(beadFblk_avgErr.at(0,0)));
      h5BgDbg.CreateAttribute(ptr->getDataSetId(),"description","avg-err per flow-block");
      h5BgDbg.CreateAttribute(ptr->getDataSetId(),"paramNames","avgErr");
      beadFblk_avgErrDS.push_back(ptr);

      sprintf(s,"/bead/flowBlkParams/clonal/flow_%04d",f);
      ptr = h5BgDbg.CreateDataSet(s, 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(beadFblk_clonal.at(0,0)));
      h5BgDbg.CreateAttribute(ptr->getDataSetId(),"description","clonal flag per flow-block");
      h5BgDbg.CreateAttribute(ptr->getDataSetId(),"paramNames","clonal");
      beadFblk_clonalDS.push_back(ptr);

      sprintf(s,"/bead/flowBlkParams/corrupt/flow_%04d",f);
      ptr = h5BgDbg.CreateDataSet(s, 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(beadFblk_corrupt.at(0,0)));
      h5BgDbg.CreateAttribute(ptr->getDataSetId(),"description","corrupt flag per flow-block");
      h5BgDbg.CreateAttribute(ptr->getDataSetId(),"paramNames","corrupt");
      beadFblk_corruptDS.push_back(ptr);
    }

    }
    catch( char * str ) {
       cout << "Exception raised while creating bead datasets in BkgParamH5::Init(): " << str << '\n';
    }
    ///------------------------------------------------------------------------------------------------------------
    /// region parameters
    ///------------------------------------------------------------------------------------------------------------
    try {
    //cout << "Init...regionalParams.n_rows,n_cols=" << regionalParams.n_rows << "," << regionalParams.n_cols << endl;
    //rp_dims[0] = loc_context.numRegions * nFlowBlks;
    char s[128];
    rp_dims[0] = rp_chunks[0] = loc_context.numRegions;
    rp_dims[1] = rp_chunks[1] = numFlows;
    beadDC_bg.set_size(rp_dims[0], rp_dims[1]);
    beadDC_bgDS = h5BgDbg.CreateDataSet("/region/bg_bead_DC", 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(beadDC_bg.at(0,0)));
    h5BgDbg.CreateAttribute(beadDC_bgDS->getDataSetId(),"description","bg_bead_DC");
    h5BgDbg.makeParamNames("bg_beadDC_",numFlows,str);
    h5BgDbg.CreateAttribute(beadDC_bgDS->getDataSetId(),"paramNames",str.c_str());

    rp_dims[0] = rp_chunks[0] = loc_context.numRegions;
    rp_dims[1] = rp_chunks[1] = 2;
    regionOffset.set_size(rp_dims[0], rp_dims[1]);
    regionOffsetDS = h5BgDbg.CreateDataSet("/region/region_offset_RowCol", 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(regionOffset.at(0,0)));
    h5BgDbg.CreateAttribute(regionOffsetDS->getDataSetId(),"description","region-offset-xy");
    h5BgDbg.CreateAttribute(regionOffsetDS->getDataSetId(),"paramNames","row,col");

    regionalParams.set_size(loc_context.numRegions, numParam);
    regionalParamsExtra.set_size(loc_context.numRegions, 9);

    emptyOnceParam.Init(loc_context.numRegions, my_image_spec.uncompFrames, numFlows);
    emptyOnceParam.SetRange(0,loc_context.numRegions, 0, my_image_spec.uncompFrames, 0, blocksOfFlow);
    emptyOnceParam.AllocateBuffer();
    emptyOnceParamDS = h5BgDbg.CreateDataSet("/region/emptyTrace", emptyOnceParam, 3);
    h5BgDbg.CreateAttribute(emptyOnceParamDS->getDataSetId(),"description","Empty-trace per region, per imgFrame , per flow");
    h5BgDbg.makeParamNames("empty_",numFlows,str);
    h5BgDbg.CreateAttribute(emptyOnceParamDS->getDataSetId(),"paramNames",str.c_str());

    darkOnceParam.Init(loc_context.numRegions, NUMNUC, MAX_COMPRESSED_FRAMES);
    darkOnceParam.SetRange(0,loc_context.numRegions, 0, NUMNUC, 0, darkOnceParam.GetNumZ());
    darkOnceParam.AllocateBuffer();
    darkOnceParamDS = h5BgDbg.CreateDataSet("/region/darkMatter/missingMass", darkOnceParam, 3);
    sprintf(s,"Missing-mass per region, per nucleotide, %d data points",MAX_COMPRESSED_FRAMES);
    h5BgDbg.CreateAttribute(darkOnceParamDS->getDataSetId(),"description",s);
    h5BgDbg.makeParamNames("missingMass_",MAX_COMPRESSED_FRAMES,str);
    h5BgDbg.CreateAttribute(darkOnceParamDS->getDataSetId(),"paramNames",str.c_str());

    rp_dims[0] = rp_chunks[0] = loc_context.numRegions;
    rp_dims[1] = rp_chunks[1] = NUMFB;
    darknessParam.set_size(loc_context.numRegions, NUMFB);
    darknessParamDS = h5BgDbg.CreateDataSet("/region/darkMatter/darkness", 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(darknessParam.at(0,0)));
    h5BgDbg.CreateAttribute(darknessParamDS->getDataSetId(),"description","Darkness per region, per flowBuffer");
    h5BgDbg.makeParamNames("darkness_",NUMFB,str);
    h5BgDbg.CreateAttribute(darknessParamDS->getDataSetId(),"paramNames",str.c_str());

    rp_dims[1] = rp_chunks[1] = 2;
    regionInitParam.set_size(loc_context.numRegions, 2);
    regionInitParamDS = h5BgDbg.CreateDataSet("/region/region_init_param", 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(regionInitParam.at(0,0)));
    h5BgDbg.CreateAttribute(regionInitParamDS->getDataSetId(),"description","Region-init-param per region, 2 data points: t_mid_nuc_start & sigma_start");
    h5BgDbg.CreateAttribute(regionInitParamDS->getDataSetId(),"paramNames","t_mid_nuc_start,sigma_start");

    emphasisParam.Init(loc_context.numRegions, MAX_HPLEN+1, MAX_COMPRESSED_FRAMES);
    emphasisParam.SetRange(0,loc_context.numRegions, 0, MAX_HPLEN+1, 0, emphasisParam.GetNumZ());
    emphasisParam.AllocateBuffer();

    for (int i=0; i<nFlowBlks; i++) {
        int f = (i+1)*NUMFB;
        if (i==nFlowBlks-1)
            f = numFlows;
        sprintf(s,"/region/emphasis/flow_%04d",f);
        H5DataSet *ptr = h5BgDbg.CreateDataSet(s, emphasisParam, 3);
        sprintf(s,"Emphasis param per region, per HP length, %d data points per flow-block",MAX_COMPRESSED_FRAMES);
        h5BgDbg.CreateAttribute(ptr->getDataSetId(),"description",s);
        h5BgDbg.makeParamNames("emphasis_",NUMFB,str);
        h5BgDbg.CreateAttribute(ptr->getDataSetId(),"paramNames",str.c_str());
        emphasisParamDS.push_back(ptr);


        rp_dims[0] = rp_chunks[0] = loc_context.numRegions;
        rp_dims[1] = rp_chunks[1] = numParam;
        sprintf(s,"/region/region_param/flow_%04d",f);
        ptr = h5BgDbg.CreateDataSet(s, 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(regionalParams.at(0,0)));
        sprintf(s,"Region_param per region, %d data points per flow-block",numParam);
        h5BgDbg.CreateAttribute(ptr->getDataSetId(),"description",s);
        h5BgDbg.makeParamNames("regionParam_",numParam,str);
        h5BgDbg.CreateAttribute(ptr->getDataSetId(),"paramNames",str.c_str());
        regionParamDS.push_back(ptr);
        rp_dims[1] = rp_chunks[1] = regionalParamsExtra.n_cols;
        sprintf(s,"/region/region_param_extra/flow_%04d",f);
        ptr = h5BgDbg.CreateDataSet(s, 2, rp_dims, rp_chunks, 3, h5BgDbg.GetH5Type(regionalParamsExtra.at(0,0)));
        sprintf(s,"Region_param_extra per region, %d data points per flow-block",regionalParamsExtra.n_cols);
        h5BgDbg.CreateAttribute(ptr->getDataSetId(),"description",s);
        h5BgDbg.CreateAttribute(ptr->getDataSetId(),"paramNames","midNucTime_0,midNucTime_1,midNucTime_2,midNucTime_3,sigma_0,sigma_1,sigma_2,sigma_3,sigma");
        regionParamDSExtra.push_back(ptr);
    }
    }
    catch( char * str ) {
       cout << "Exception raised while creating region datasets in BkgParamH5::Init(): " << str << '\n';
    }
    ///------------------------------------------------------------------------------------------------------------
    /// savePointers to be passed to BkgModel::WriteBeadParameterstoDataCubes
    ///------------------------------------------------------------------------------------------------------------
    savePointers();
    return hgBgDbgFile;
}


void BkgParamH5::DumpBkgModelRegionInfoH5(BkgFitterTracker &GlobalFitter, int flow, bool last_flow)
// DumpBkgModelRegionInfoH5() called by IncrementalWrite() for every flow in ProcessImageToWell.cpp
{
  int numRegions = GlobalFitter.numFitters;

  for (int r = 0; r<numRegions; r++) {

      ///------------------------------------------------------------------------------------------------------------
      /// emptyTrace
      // use copyCube_element to copy DataCube element to mEmptyOnceParam in BkgParamH5::DumpBkgModelRegionInfoH5
      ///------------------------------------------------------------------------------------------------------------
      int imgFrames = GlobalFitter.BkgModelFitters[r]->get_emptytrace_imgFrames();
      float *buff = GlobalFitter.BkgModelFitters[r]->get_emptytrace_bg_buffers();
      float *bg_bead_dc = GlobalFitter.BkgModelFitters[r]->get_emptytrace_bg_dc_offset();

      for (int j=0; j<imgFrames; j++) {
          ptrs.copyCube_element(ptrs.mEmptyOnceParam,r,j,flow,buff[j]);
      }

      ///------------------------------------------------------------------------------------------------------------
      /// bg_bead_DC
      ///------------------------------------------------------------------------------------------------------------
      ptrs.copyMatrix_element(ptrs.mBeadDC_bg,r,flow,bg_bead_dc[flow]);
  }


  if (CheckFlowForWrite(flow, last_flow) ) {
    struct reg_params rp;
    struct reg_params_H5 rp5;
    //int numParam = sizeof(reg_params)/sizeof(float); // +1 naot necessary
    int numParam = sizeof(reg_params_H5)/sizeof(float); // +1 not necessary
  //cout << "DumpBkgModelRegionInfoH5... numParam=" << numParam << ", sizeof(reg_params) " << sizeof(reg_params) << " / size(float)" << sizeof(float) << endl;

    for (int r = 0; r<numRegions; r++) {
        GlobalFitter.BkgModelFitters[r]->GetRegParams(rp);
        int iFlow1 = flow%NUMFB + 1;

        ///------------------------------------------------------------------------------------------------------------
        /// darkMatter
        // use copyCube_element to copy DataCube element to mDarkOnceParam in BkgParamH5::DumpBkgModelRegionInfoH5
        ///------------------------------------------------------------------------------------------------------------
        int npts = GlobalFitter.BkgModelFitters[r]->get_tim_c_npts();
        for (int i=0; i<NUMNUC; i++) {
            float * missingMass = GlobalFitter.BkgModelFitters[r]->get_region_darkMatter(i);
            for (int j=0; j<npts; j++)
                ptrs.copyCube_element(ptrs.mDarkOnceParam,r,i,j,missingMass[j]);
            for (int j=npts; j<MAX_COMPRESSED_FRAMES; j++)
                ptrs.copyCube_element(ptrs.mDarkOnceParam,r,i,j,0); // pad 0's
        }

        ///------------------------------------------------------------------------------------------------------------
        /// darkness
        // use copyMatrix_element to copy Matrix element to mDarknessParam in BkgParamH5::DumpBkgModelRegionInfoH5
        ///------------------------------------------------------------------------------------------------------------
        for (int i=0; i<iFlow1; i++)
            ptrs.copyMatrix_element(ptrs.mDarknessParam,r,i,rp.darkness[i]);
        for (int i=iFlow1; i<NUMFB; i++)
            ptrs.copyMatrix_element(ptrs.mDarknessParam,r,i,0); //pad 0's


        ///------------------------------------------------------------------------------------------------------------
        /// region_size, only once at flow+1=NUMFB
        ///------------------------------------------------------------------------------------------------------------
        regionOffset.at(r,0) = GlobalFitter.BkgModelFitters[r]->get_region_row();
        regionOffset.at(r,1) = GlobalFitter.BkgModelFitters[r]->get_region_col();

        ///------------------------------------------------------------------------------------------------------------
        /// regionInitParam, only once at flow+1=NUMFB
        ///------------------------------------------------------------------------------------------------------------
        regionInitParam.at(r,0) = GlobalFitter.BkgModelFitters[r]->get_t_mid_nuc_start();
        regionInitParam.at(r,1) = GlobalFitter.BkgModelFitters[r]->get_sigma_start();

        ///------------------------------------------------------------------------------------------------------------
        /// regionalParams, many times at each (flow+1)%NUMFB==0
        ///------------------------------------------------------------------------------------------------------------
        //float *rpp = (float *) &rp;
        float *rpp5 = (float *) &rp5;
        reg_params_copyTo_reg_params_H5(rp,rp5);
        for (int i = 0; i < numParam; i++)
            regionalParams.at(r, i) = rpp5[i];

        ///------------------------------------------------------------------------------------------------------------
        /// regionalParamsExtra, many times at each (flow+1)%NUMFB==0
        ///------------------------------------------------------------------------------------------------------------
        regionalParamsExtra.at(r, 0) = GetModifiedMidNucTime(&rp.nuc_shape,TNUCINDEX,0);
        regionalParamsExtra.at(r, 1) = GetModifiedMidNucTime(&rp.nuc_shape,ANUCINDEX,0);
        regionalParamsExtra.at(r, 2) = GetModifiedMidNucTime(&rp.nuc_shape,CNUCINDEX,0);
        regionalParamsExtra.at(r, 3) = GetModifiedMidNucTime(&rp.nuc_shape,GNUCINDEX,0);
        regionalParamsExtra.at(r, 4) = GetModifiedSigma(&rp.nuc_shape,TNUCINDEX);
        regionalParamsExtra.at(r, 5) = GetModifiedSigma(&rp.nuc_shape,ANUCINDEX);
        regionalParamsExtra.at(r, 6) = GetModifiedSigma(&rp.nuc_shape,CNUCINDEX);
        regionalParamsExtra.at(r, 7) = GetModifiedSigma(&rp.nuc_shape,GNUCINDEX);
        regionalParamsExtra.at(r, 8) = rp.nuc_shape.sigma;

    }


  ///------------------------------------------------------------------------------------------------------------
  /// write to dataset (DS) in the .h5 output file
  ///------------------------------------------------------------------------------------------------------------
  //note: regionParam only has to allocate enough regionalParam.n_rows per flow-block
  // only the DS has to have the right # of rows, as in the CreateDataSet()
  //cout << "DumpBkgModelRegionInfoH5... flow= " << flow << endl;
  size_t starts[2];
  size_t ends[2];
  //starts[0] = flow/NUMFB * numRegions;
  starts[0] = 0;
  ends[0] = starts[0] + numRegions;
  if (last_flow && (flow+1) % NUMFB != 0) {
      ends[0] = starts[0] + (flow+1) % NUMFB;
      //starts[0] = ceil((float) flow  / NUMFB);
      cout << "DumpBkgModelRegionInfoH5... lastflow= " << flow << endl;
      cout << "changing starts[0] from " << (flow/NUMFB)* numRegions << " to " << starts[0] << " at flow " << flow << endl;
    }
  starts[1] = 0;
  ends[1] = regionalParams.n_cols; // numParam, not numParam+1
  ION_ASSERT(ends[0] <= regionalParams.n_rows, "ends[0] > regionalParam.n_rows");
  ION_ASSERT(ends[1] <= regionalParams.n_cols, "ends[1] > regionalParam.n_cols");

  // write matrix to the DS w/ the right start/offset (or the right DS as in the later versions)
  int iBlk = ceil(float(flow+1)/NUMFB) - 1;
  arma::Mat<float> m = arma::trans(regionalParams);
  regionParamDS[iBlk]->WriteRangeData(starts, ends, m.memptr());

  ends[1] = regionalParamsExtra.n_cols;
  arma::Mat<float> m1 = arma::trans(regionalParamsExtra);
  regionParamDSExtra[iBlk]->WriteRangeData(starts, ends, m1.memptr());
  }
}

void BkgParamH5::IncrementalWriteParam(DataCube<float> &cube, H5DataSet *set, int flow, int saveWellsFrequency,int numFlows)
{
  int testWellFrequency = saveWellsFrequency*NUMFB; // block size
  if (((flow+1) % (saveWellsFrequency*NUMFB) == 0 && (flow != 0))  || (flow+1) >= numFlows) {
    fprintf(stdout, "Writing incremental wells at flow: %d\n", flow);
    MemUsage("BeforeWrite");
    size_t starts[3];
    size_t ends[3];
    cube.SetStartsEnds(starts, ends);
    set->WriteRangeData(starts, ends, cube.GetMemPtr());
    cube.SetRange(0, cube.GetNumX(), 0, cube.GetNumY(), flow+1, flow + 1 + min(testWellFrequency,numFlows-(flow+1)));
    MemUsage("AfterWrite");
  }
}


void BkgParamH5::IncrementalWrite(BkgFitterTracker &GlobalFitter, int flow, int numFlows, bool last_flow) {
  if (ptrs.mAmpl != NULL) {
    DumpBkgModelRegionInfoH5(GlobalFitter, flow, last_flow);

    IncrementalWriteParam(beadDC,beadDCDS,flow,1,numFlows);
    IncrementalWriteParam(amplMultiplier,amplDS,flow,1,numFlows);
    IncrementalWriteParam(kRateMultiplier,kRateDS,flow,1,numFlows);
    IncrementalWriteParam(bgResidualError,resErrorDS,flow,1,numFlows);
    IncrementalWriteParam(emptyOnceParam,emptyOnceParamDS,flow,1,numFlows);
    beadDC_bgDS->WriteMatrix(beadDC_bg,flow);

    if (CheckFlowForWrite(flow,false)) {
      beadInitParamDS->WriteDataCube(beadInitParam);
      darkOnceParamDS->WriteDataCube(darkOnceParam);
      //emptyOnceParamDS->WriteDataCube(emptyOnceParam);
      regionInitParamDS->WriteMatrix(regionInitParam);
      darknessParamDS->WriteMatrix(darknessParam);
      regionOffsetDS->WriteMatrix(regionOffset);
    }
    if (CheckFlowForWrite(flow,last_flow)) {
        int iBlk = ceil(float(flow+1)/NUMFB) - 1;
        emphasisParamDS[iBlk]->WriteDataCube(emphasisParam);
        beadFblk_avgErrDS[iBlk]->WriteMatrix(beadFblk_avgErr);
    }
  }
}


void BkgParamH5::Close() {
  if (amplDS != NULL) {
    amplDS->Close();            amplDS = NULL;
    beadDCDS->Close();          beadDCDS = NULL;
    beadDC_bgDS->Close();       beadDC_bgDS = NULL;
    kRateDS->Close();           kRateDS = NULL;
    resErrorDS->Close();        resErrorDS = NULL;
    beadInitParamDS->Close();   beadInitParamDS = NULL;
    darkOnceParamDS->Close();   darkOnceParamDS = NULL;
    darknessParamDS->Close();   darknessParamDS = NULL;
    emptyOnceParamDS->Close();  emptyOnceParamDS = NULL;
    regionInitParamDS->Close(); regionInitParamDS = NULL;
    regionOffsetDS->Close(); regionOffsetDS = NULL;

    close_Vector(regionParamDS);
    close_Vector(regionParamDSExtra);
    close_Vector(emphasisParamDS);

    close_Vector(beadFblk_avgErrDS);
    close_Vector(beadFblk_clonalDS);
    close_Vector(beadFblk_corruptDS);
  }

}



void BkgParamH5::savePointers()
{
    //regionalParams & regionalParamsExtra don't need to be sotred, because their values are already in the BkgModel
    // or could be calculated directly

    ptrs.mAmpl = &amplMultiplier;
    ptrs.mBeadDC = &beadDC;
    ptrs.mBeadDC_bg = &beadDC_bg;
    ptrs.mKMult = &kRateMultiplier;
    ptrs.mResError = &bgResidualError;
    ptrs.mBeadInitParam = &beadInitParam;
    ptrs.mDarkOnceParam = &darkOnceParam;
    ptrs.mDarknessParam= &darknessParam;

    ptrs.mEmptyOnceParam = &emptyOnceParam;
    ptrs.mRegionInitParam = &regionInitParam;
    ptrs.mEmphasisParam= &emphasisParam;

    ptrs.mBeadFblk_avgErr= &beadFblk_avgErr;
    ptrs.mBeadFblk_clonal= &beadFblk_clonal;
    ptrs.mBeadFblk_corrupt= &beadFblk_corrupt;

    ptrs.mRegionOffset= &regionOffset;

}


void BkgParamH5::close_Vector(vector<H5DataSet * > &vec)
{
    int sz = vec.size();
    if (sz > 0) {
        for (int i=0; i<sz; i++)
            if (vec[i]!=NULL)
                vec[i]->Close();
        vec.clear();
    }
}


