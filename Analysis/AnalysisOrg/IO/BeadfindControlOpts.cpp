/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BeadfindControlOpts.h"
#include "Utils.h"
#include "string.h"

void BeadfindControlOpts::DefaultBeadfindControl()
{
  //maxNumKeyFlows = 0;
  //minNumKeyFlows = 99;
  bfMinLiveRatio = .0001;
  bfMinLiveLibSnr = 3;
  bfMinLiveTfSnr = -1;
  bfTfFilterQuantile = 1;
  bfLibFilterQuantile = 1;
  skipBeadfindSdRecover = 1;
  beadfindThumbnail = 0;
  beadfindSmoothTrace = false;
  filterNoisyCols = "none";
  beadMaskFile = NULL;
  maskFileCategorized = false;
  sprintf (bfFileBase, "beadfind_post_0003.dat");
  sprintf (preRunbfFileBase, "beadfind_pre_0003.dat");
  BF_ADVANCED = true;
  SINGLEBF = true;
  noduds = false;
  beadfindUseSepRef = false;
  numThreads = -1;
  minTfPeakMax = 40.0f;
  minLibPeakMax = 15.0f;
  bfOutputDebug = 1;
  bfMult = 1.0;
  sdAsBf = true;
  useBeadfindGainCorrection = true;
  useDatacollectGainCorrectionFile = false;
  useSignalReference = 4;
  useSignalReferenceSet = false;
  blobFilter = false;
  bfOutputDebug = 2;
  beadfindType = "differential";
  predictFlowStart = -1;
  predictFlowEnd = -1;
  meshStepX = 100;
  meshStepY = 100;
}

BeadfindControlOpts::~BeadfindControlOpts()
{
  if (beadMaskFile)
    free (beadMaskFile);
}

void BeadfindControlOpts::SetThumbnail(bool tn)
{
    beadfindThumbnail = tn ? 1 : 0;
}

void BeadfindControlOpts::PrintHelp()
{
	printf ("     BeadfindControlOpts\n");
    printf ("  -b,--beadfindfile          FILE              prerun bf file base []\n");
    printf ("  -b,--beadfindFile          FILE              same as --beadfindfile []\n");
    printf ("     --beadfind-type         STRING            beadfind type [naoh,positive]\n");
	printf ("     --use-beadmask          FILE              bead mask file name []\n");
	printf ("     --beadmask-categorized  BOOL              beadmask categorized [false]\n");
	printf ("     --beadfind-basis        STRING            beadfind basis []\n");
	printf ("     --beadfind-dat          STRING            beadfind dat []\n");
	printf ("     --beadfind-bgdat        STRING            beadfind bgdat []\n");
	printf ("     --beadfind-sdasbf       BOOL              beadfind sdAsBf [true]\n");
	printf ("     --beadfind-bfmult       FLOAT             beadfind mult [1.0]\n");
	printf ("     --beadfind-minlive      DOUBLE            beadfind minlive ratio [0.0001]\n");
	printf ("     --beadfind-minlivesnr   DOUBLE            beadfind min live lib SNR [4.0]\n");
	printf ("     --beadfind-min-tf-snr   DOUBLE            beadfind min live tf SNR [-1]\n");
	printf ("     --beadfind-tf-min-peak  FLOAT             beadfind min tf peakmMax [40.0]\n");
	printf ("     --beadfind-lib-min-peak FLOAT             beadfind min lib peakmMax [10.0]\n");
	printf ("     --beadfind-lib-filt     DOUBLE            beadfind lib filter quantile [1.0]\n");
	printf ("     --beadfind-tf-filt      DOUBLE            beadfind tf filter quantile [1.0]\n");
	printf ("     --beadfind-skip-sd-recover          INT   beadfind skip beadfind sd recover [1]\n");
	printf ("     --beadfind-sep-ref      BOOL              beadfind use seperated ref [false]\n");
	printf ("     --beadfind-smooth-trace BOOL              beadfind lagone filt [0]\n");
	printf ("     --beadfind-diagnostics  INT               beadfind output debug [0]\n");
	printf ("     --beadfind-washout      BOOL              beadfind washout [false]\n");
	printf ("     --beadfind-gain-correction          BOOL  beadfind gain correction [true]\n");
	printf ("     --datacollect-gain-correction       BOOL  datacollect gain correction from Gain.lsr file[false]\n");
	printf ("     --beadfind-blob-filter  BOOL              beadfind blob filter [false]\n");
	printf ("     --beadfind-predict-start            INT   beadfind predict flow start [-1]\n");
	printf ("     --beadfind-predict-end  INT               beadfind predict flow end [-1]\n");
	printf ("     --beadfind-sig-ref-type INT               beadfind use signal reference []\n");
	printf ("     --beadfind-zero-flows   STRING            beadfind double tap flows []\n");
	printf ("     --beadfind-num-threads  INT               beadfind number of threads [-1]\n");
	printf ("     --beadfind-mesh-step    VECTOR_INT        beadfind mesh steps for differntial separator [100,100]\n");
    printf ("     --beadfind-acq-threshold    VECTOR_INT    beadfind threshold for hinge model for acquisition flow [-10,12,5,500]\n");
    printf ("     --beadfind-bf-threshold    VECTOR_INT     beadfind threshold for hinge model for beadfind flow [-5,300,-20000,-10]\n");
    printf ("     --exclusion-mask          FILE            exclusion mask file name []\n");
	printf ("     --bfold                 BOOL              BF_ADVANCED [true]\n");
	printf ("     --noduds                BOOL              noduds [false]\n");
    printf ("\n");
}

void BeadfindControlOpts::SetOpts(OptArgs &opts, Json::Value& json_params)
{
    beadfindType = RetrieveParameterString(opts, json_params, '-', "beadfind-type", "differential");
    if ( beadfindType != "differential" )
    {
        fprintf ( stderr, "*Error* - Illegal option to --beadfind-type: %s, valid options are 'differential'\n",
                  beadfindType.c_str() );
        exit ( EXIT_FAILURE );
    }
    string s1 = RetrieveParameterString(opts, json_params, '-', "use-beadmask", "");
    if(s1.length() > 0)
    {
        sprintf(beadMaskFile, "%s", s1.c_str());
    }
    maskFileCategorized = RetrieveParameterBool(opts, json_params, '-', "beadmask-categorized", false);

    exclusionMaskFile = RetrieveParameterString(opts, json_params, '-', "exclusion-mask", "");

    string s2 = RetrieveParameterString(opts, json_params, '-', "beadfind-basis", "naoh");
    if(s2.length() > 0)
    {
        bfType = s2;
        if ( bfType != "naoh" && bfType != "positive"  && bfType != "nobuffer")
        {
            fprintf ( stderr, "*Error* - Illegal option to --beadfind-basis: %s, valid options are 'naoh','nobuffer' or 'positive'\n",	bfType.c_str() );
            exit ( EXIT_FAILURE );
        }
    }
    bfDat = RetrieveParameterString(opts, json_params, '-', "beadfind-dat", "beadfind_pre_0003.dat");
    bfBgDat = RetrieveParameterString(opts, json_params, '-', "beadfind-bgdat", "");
    sdAsBf = RetrieveParameterBool(opts, json_params, '-', "beadfind-sdasbf", true);
    bfMult = RetrieveParameterFloat(opts, json_params, '-', "beadfind-bfmult", 1.0);
    bfMinLiveRatio = RetrieveParameterDouble(opts, json_params, '-', "beadfind-minlive", 0.0001);
    bfMinLiveLibSnr = RetrieveParameterDouble(opts, json_params, '-', "beadfind-minlivesnr", 4.0);
    bfMinLiveTfSnr = RetrieveParameterDouble(opts, json_params, '-', "beadfind-min-tf-snr", 7);
    minTfPeakMax = RetrieveParameterFloat(opts, json_params, '-', "beadfind-tf-min-peak", 40.0);
    minLibPeakMax = RetrieveParameterFloat(opts, json_params, '-', "beadfind-lib-min-peak", 10.0);
    bfLibFilterQuantile = RetrieveParameterDouble(opts, json_params, '-', "beadfind-lib-filt", 1.0);
    bfTfFilterQuantile = RetrieveParameterDouble(opts, json_params, '-', "beadfind-tf-filt", 1.0);
    skipBeadfindSdRecover = RetrieveParameterInt(opts, json_params, '-', "beadfind-skip-sd-recover", 1);
    filterNoisyCols = RetrieveParameterString(opts, json_params, '-', "beadfind-filt-noisy-col", "none");
    beadfindUseSepRef = RetrieveParameterBool(opts, json_params, '-', "beadfind-sep-ref", false);
    beadfindSmoothTrace = RetrieveParameterBool(opts, json_params, '-', "beadfind-smooth-trace", false);
    bfOutputDebug = RetrieveParameterInt(opts, json_params, '-', "beadfind-diagnostics", 2);
    bool b1 = RetrieveParameterBool(opts, json_params, '-', "bead-washout", false);
    SINGLEBF = !b1;
    useBeadfindGainCorrection = RetrieveParameterBool(opts, json_params, '-', "beadfind-gain-correction", true);
    useDatacollectGainCorrectionFile = RetrieveParameterBool(opts, json_params, '-', "datacollect-gain-correction", false);
    blobFilter = RetrieveParameterBool(opts, json_params, '-', "beadfind-blob-filter", false);
    predictFlowStart = RetrieveParameterInt(opts, json_params, '-', "beadfind-predict-start", -1);
    predictFlowEnd = RetrieveParameterInt(opts, json_params, '-', "beadfind-predict-end", -1);
    useSignalReference = RetrieveParameterInt(opts, json_params, '-', "beadfind-sig-ref-type", -1);
    if(useSignalReference == -1)
    {
        useSignalReference = 4;
    }
    else
    {
        useSignalReferenceSet = true;
    }
    doubleTapFlows = RetrieveParameterString(opts, json_params, '-', "beadfind-zero-flows", "");
    numThreads = RetrieveParameterInt(opts, json_params, '-', "beadfind-num-threads", -1);

    //jz the following comes from CommandLineOpts::GetOpts
    BF_ADVANCED = RetrieveParameterBool(opts, json_params, '-', "bfold", true);
    noduds = RetrieveParameterBool(opts, json_params, '-', "noduds", false);
    string sb = RetrieveParameterString(opts, json_params, 'b', "beadfindfile", "");
    if(sb.length() > 0)
    {
        snprintf ( preRunbfFileBase, 256, "%s", sb.c_str() );
        bfFileBase[0] = '\0';
        SINGLEBF = true;
    }
    vector<int> vec;
    RetrieveParameterVectorInt(opts, json_params, '-', "beadfind-mesh-step", "", vec);
    if(vec.size() == 2)
    {
        meshStepX = vec[0];
        meshStepY = vec[1];
    }

    RetrieveParameterVectorFloat(opts, json_params, '-', "beadfind-acq-threshold","-10,3,5,500", beadfindAcqThreshold);
    if( (beadfindAcqThreshold.size() != 4) || (beadfindAcqThreshold[0] > beadfindAcqThreshold[1]) || (beadfindAcqThreshold[2] > beadfindAcqThreshold[3]) ){
        fprintf ( stderr, "*Error* - Illegal option value --beadfind-acq-threshold. Provide 4 values minFirstSlope,maxFirstSlope,minSecondSlope,maxSecondSlope. \n");
        exit ( EXIT_FAILURE );
    }
    RetrieveParameterVectorFloat(opts, json_params, '-', "beadfind-bf-threshold","-5,300,-2000,-10", beadfindBfThreshold);
    if( (beadfindBfThreshold.size() != 4) || (beadfindBfThreshold[0] > beadfindBfThreshold[1]) || (beadfindBfThreshold[2] > beadfindBfThreshold[3]) ){
        fprintf ( stderr, "*Error* - Illegal option value --beadfind-bf-threshold. Provide 4 values minFirstSlope,maxFirstSlope,minSecondSlope,maxSecondSlope. \n");
        exit ( EXIT_FAILURE );
    }
}
