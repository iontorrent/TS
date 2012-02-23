/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "GlobalDefaultsForBkgModel.h"
#include "Utils.h"
#include "IonErr.h"

float GlobalDefaultsForBkgModel::kmax_default[NUMNUC]  = { 18.0,   20.0,   17.0,   18.0 };
float GlobalDefaultsForBkgModel::krate_default[NUMNUC] = { 18.78,   20.032,   25.04,   31.3 };
float GlobalDefaultsForBkgModel::d_default[NUMNUC]     = {159.923,189.618,227.021,188.48};
 float GlobalDefaultsForBkgModel::krate_adj_limit[NUMNUC] = {2.0,2.0,2.0,2.0};
 float GlobalDefaultsForBkgModel::sigma_mult_default[NUMNUC] = {1.162,1.124,1.0,0.8533};
 float GlobalDefaultsForBkgModel::t_mid_nuc_delay_default[NUMNUC] = {0.69,1.78,0.0,0.17}; 
float GlobalDefaultsForBkgModel::nuc_flow_frame_width = 22.5;  // 1.5 seconds * 15 frames/second
int GlobalDefaultsForBkgModel::time_left_avg = 5;
int GlobalDefaultsForBkgModel::time_start_detail = -5;
int GlobalDefaultsForBkgModel::time_stop_detail = 16;

float GlobalDefaultsForBkgModel::dampen_kmult = 0;
bool GlobalDefaultsForBkgModel::var_kmult_only = false;
bool GlobalDefaultsForBkgModel::generic_test_flag = false;
 
void GlobalDefaultsForBkgModel::SetFlowOrder(char *_flowOrder)
{
  if (flowOrder)
    free(flowOrder);
  if (glob_flow_ndx_map)
    free(glob_flow_ndx_map);

  flowOrder = strdup(_flowOrder);
  flow_order_len = strlen(flowOrder);
  glob_flow_ndx_map = new int[flow_order_len];

  for (int i=0;i < flow_order_len;i++)
  {
    switch (toupper(flowOrder[i]))
    {
      case 'T':
        glob_flow_ndx_map[i]=TNUCINDEX;
        break;
      case 'A':
        glob_flow_ndx_map[i]=ANUCINDEX;
        break;
      case 'C':
        glob_flow_ndx_map[i]=CNUCINDEX;
        break;
      case 'G':
        glob_flow_ndx_map[i]=GNUCINDEX;
        break;
      default:
        glob_flow_ndx_map[i]=DEFAULTNUCINDEX;
        break;
    }
  }
}

void GlobalDefaultsForBkgModel::GetFlowOrderBlock(int *my_flow, int i_start, int i_stop)
{
  for (int i=i_start; i<i_stop; i++)
      my_flow[i-i_start] = GetNucNdx(i);
}


float GlobalDefaultsForBkgModel::clonal_call_scale[] = {0.902,0.356,0.078,0.172,0.436,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
float GlobalDefaultsForBkgModel::clonal_call_penalty = 1600.0;

float GlobalDefaultsForBkgModel::sens_default = 1.256;
float GlobalDefaultsForBkgModel::tau_R_m_default = -24.36;
float GlobalDefaultsForBkgModel::tau_R_o_default = 25.16;

float GlobalDefaultsForBkgModel::emp[]  = {6.86, 1.1575, 2.081, 1.230, 7.2625, 1.91, 0.0425, 19.995};

int  *GlobalDefaultsForBkgModel::glob_flow_ndx_map = NULL;
int   GlobalDefaultsForBkgModel::flow_order_len = 4;
char *GlobalDefaultsForBkgModel::flowOrder = NULL;
bool  GlobalDefaultsForBkgModel::no_RatioDrift_fit_first_20_flows = false;
float GlobalDefaultsForBkgModel::emphasis_ampl_default = 7.25;
float GlobalDefaultsForBkgModel::emphasis_width_default = 2.89;


#define MAX_LINE_LEN    2048
#define MAX_DATA_PTS    80
// Load optimized defaults from GeneticOptimizer runs
void GlobalDefaultsForBkgModel::SetGoptDefaults(char *fname)
{
  struct stat fstatus;
  int         status;
  FILE *param_file;
  char *line;
  int nChar = MAX_LINE_LEN;
  float d[10];
  
  int num = 0;

  line = new char[MAX_LINE_LEN];

  status = stat(fname,&fstatus);
  
  if (status == 0)
  {
    // file exists
    printf("GOPT: loading parameters from %s\n",fname);
    
    param_file=fopen(fname,"rt");
    
    bool done = false;
    
    while (!done)
    {
        int bytes_read = getline(&line,(size_t *)&nChar,param_file);

        if (bytes_read > 0)
        {
          if (bytes_read >= MAX_LINE_LEN || bytes_read < 0) {
            ION_ABORT("Read: " + ToStr(bytes_read) + " into a buffer only: " + 
                      ToStr(MAX_LINE_LEN) + " long for line: '" + ToStr(line) + "'");
          }
          line[bytes_read]='\0';
            if (strncmp("km_const",line,8) == 0)            
            {
                num = sscanf(line,"km_const: %f %f %f %f",&d[0],&d[1],&d[2],&d[3]);
                if (num > 0)                    
                    for (int i=0;i<NUMNUC;i++) kmax_default[i] = d[i];
            }
            if (strncmp("krate",line,5) == 0)            
            {
                num = sscanf(line,"krate: %f %f %f %f",&d[0],&d[1],&d[2],&d[3]);
                if (num > 0)            
                    for (int i=0;i<NUMNUC;i++) krate_default[i] = d[i];
            }
            if (strncmp("d_coeff",line,7) == 0)            
            {
                num = sscanf(line,"d_coeff: %f %f %f %f",&d[0],&d[1],&d[2],&d[3]);
                if (num > 0)             
                    for (int i=0;i<NUMNUC;i++) d_default[i] = d[i];
            }
            if (strncmp("sens",line,4) == 0)
                num = sscanf(line,"sens: %f",&sens_default);
            if (strncmp("tau_R_m",line,7) == 0)
                num = sscanf(line,"tau_R_m: %f",&tau_R_m_default);
            if (strncmp("tau_R_o",line,7) == 0)
                num = sscanf(line,"tau_R_o: %f",&tau_R_o_default);
                              
            if (strncmp("emphasis",line,8) == 0)
            {            
               num = sscanf(line,"emphasis: %f %f %f %f %f %f %f %f", &d[0],&d[1],&d[2],&d[3],&d[4],&d[5],&d[6],&d[7]);
                for (int i=0;i<num;i++) emp[i]=d[i];
            }
            if (strncmp("emp_amp",line,7) == 0)
                num = sscanf(line,"emp_amplitude: %f",&emphasis_ampl_default);
            if (strncmp("emp_width",line,7) == 0)
                num = sscanf(line,"emp_width: %f",&emphasis_width_default);
                
            if (strncmp("clonal_call_scale",line,17) == 0)
            {            
               num = sscanf(line,"clonal_call_scale: %f %f %f %f %f", &d[0],&d[1],&d[2],&d[3],&d[4]);
                for (int i=0;i<num;i++) clonal_call_scale[i]=d[i];
            }        
            if (strncmp("sigma_mult", line, 10) == 0)
            {
               num = sscanf(line,"sigma_mult: %f %f %f %f", &d[0],&d[1],&d[2],&d[3]);
                for (int i=0;i<num;i++) sigma_mult_default[i]=d[i];
            }     
            if (strncmp("t_mid_nuc_delay", line, 15) == 0)
            {
               num = sscanf(line,"t_mid_nuc_delay: %f %f %f %f", &d[0],&d[1],&d[2],&d[3]);
                for (int i=0;i<num;i++) t_mid_nuc_delay_default[i]=d[i];
            }    
            if (strncmp("nuc_flow_timing", line, 15) == 0)
            {
               num = sscanf(line,"nuc_flow_frame_width: %f", &d[0]);
                nuc_flow_frame_width = d[0];
            }                 
            if (strncmp("time_compression", line, 16) == 0)
            {
               num = sscanf(line,"time_compression: %f %f %f", &d[0], &d[1],&d[2]);
                time_left_avg = d[0];
                time_start_detail = d[1];
                time_stop_detail = d[2];
            }      
        }
        else
            done = true;
    }

    fclose(param_file);
    
    //output defaults used
    printf("default parameters used: \n");
    printf("kmax: %f\t%f\t%f\t%f\n",kmax_default[0],kmax_default[1],kmax_default[2],kmax_default[3]);
    printf("krate: %f\t%f\t%f\t%f\n",krate_default[0],krate_default[1],krate_default[2],krate_default[3]);
    printf("d: %f\t%f\t%f\t%f\n",d_default[0],d_default[1],d_default[2],d_default[3]);
    printf("sigma_mult: %f\t%f\t%f\t%f\n",sigma_mult_default[0],sigma_mult_default[1],sigma_mult_default[2],sigma_mult_default[3]);
    printf("t_mid_nuc_delay: %f\t%f\t%f\t%f\n",t_mid_nuc_delay_default[0],t_mid_nuc_delay_default[1],t_mid_nuc_delay_default[2],t_mid_nuc_delay_default[3]);
    printf("sens: %f\n",sens_default);
    printf("tau_R_m: %f\n",tau_R_m_default);
    printf("tau_R_o: %f\n",tau_R_o_default);
    printf("emp: %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",emp[0],emp[1],emp[2],emp[3],emp[4],emp[5],emp[6],emp[7]);
    printf("emp_amplitude: \t%f\n",emphasis_ampl_default);
    printf("emp_width: \t%f\n",emphasis_width_default);
    printf("clonal_call_scale: %f\t%f\t%f\t%f\t%f\n",clonal_call_scale[0], clonal_call_scale[1], clonal_call_scale[2], clonal_call_scale[3], clonal_call_scale[4]);
    printf("\n");
       
  }
  else
    printf("GOPT: parameter file %s does not exist\n",fname);
  
  delete [] line;
}

// This function is used during GeneticOptimizer runs in which case the above SetGoptDefaults is disabled
void GlobalDefaultsForBkgModel::ReadEmphasisVectorFromFile(char *experimentName)
{
  struct stat fstatus;
  int         status;
  char fname[512];
  FILE *evect_file;
  char *line;
  int nChar = MAX_LINE_LEN;
  float read_data[MAX_DATA_PTS];
  int pset=0;
  
  sprintf(fname,"%s/emphasis_vector.txt", experimentName);
  status = stat(fname,&fstatus);
  
  line = new char[MAX_LINE_LEN];
    
  if (status == 0)
  {
    // file exists
    printf("loading emphasis vector parameters from %s\n",fname);
    
    evect_file=fopen(fname,"rt");
  
    // first line contains the number of points
    int bytes_read = getline(&line, (size_t *) &nChar,evect_file);

    if (bytes_read > 0)
    {
      int evect_size;

      sscanf(line,"%d",&evect_size);
      int i=0;
      for (; (i < evect_size) && (i < MAX_DATA_PTS);i++)
      {
        bytes_read = getline(&line, (size_t *) &nChar,evect_file);
        sscanf(line,"%f",&read_data[i]);
      }
      // pick 3 gopt parameter sets:
      // 0 = all params, 1 - no nuc-dep factors, 2 - no nuc-dep factors plus no emphasis      
      if (evect_size == 13) pset = 2;
        else if (evect_size == 23) pset = 1;
          else if (evect_size == 43) pset = 0;
            else {
              fprintf (stderr, "Unrecognized number of points (%d) in %s\n", evect_size, fname);
              exit (1);
              }
    }

    fclose(evect_file);

    int dv = 0;

    // copy the configuration values into the right places

    // first value scales add km terms
    kmax_default[TNUCINDEX] *= read_data[dv];
    kmax_default[ANUCINDEX] *= read_data[dv];
    kmax_default[CNUCINDEX] *= read_data[dv];
    kmax_default[GNUCINDEX] *= read_data[dv++];

    // 2-5 values scale individual terms
    if (pset == 0){
      kmax_default[TNUCINDEX] *= read_data[dv++];
      kmax_default[ANUCINDEX] *= read_data[dv++];
      kmax_default[CNUCINDEX] *= read_data[dv++];
      kmax_default[GNUCINDEX] *= read_data[dv++];
    }
    
    krate_default[TNUCINDEX] *= read_data[dv];
    krate_default[ANUCINDEX] *= read_data[dv];
    krate_default[CNUCINDEX] *= read_data[dv];
    krate_default[GNUCINDEX] *= read_data[dv++];
    
    if (pset == 0){
      krate_default[TNUCINDEX] *= read_data[dv++];
      krate_default[ANUCINDEX] *= read_data[dv++];
      krate_default[CNUCINDEX] *= read_data[dv++];
      krate_default[GNUCINDEX] *= read_data[dv++];
    }

    d_default[TNUCINDEX] *= read_data[dv];
    d_default[ANUCINDEX] *= read_data[dv];
    d_default[CNUCINDEX] *= read_data[dv];
    d_default[GNUCINDEX] *= read_data[dv++];
    
    if (pset == 0){
      d_default[TNUCINDEX] *= read_data[dv++];
      d_default[ANUCINDEX] *= read_data[dv++];
      d_default[CNUCINDEX] *= read_data[dv++];
      d_default[GNUCINDEX] *= read_data[dv++];
    }
    
    sigma_mult_default[TNUCINDEX] *= read_data[dv];
    sigma_mult_default[ANUCINDEX] *= read_data[dv];
    sigma_mult_default[CNUCINDEX] *= read_data[dv];
    sigma_mult_default[GNUCINDEX] *= read_data[dv++];

    if (pset == 0){
      sigma_mult_default[TNUCINDEX] *= read_data[dv++];
      sigma_mult_default[ANUCINDEX] *= read_data[dv++];
      sigma_mult_default[CNUCINDEX] *= read_data[dv++];
      sigma_mult_default[GNUCINDEX] *= read_data[dv++];
    }
    
    t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv];
    t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv];
    t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv];
    t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];

    if (pset == 0){
      t_mid_nuc_delay_default[TNUCINDEX] *= read_data[dv++];
      t_mid_nuc_delay_default[ANUCINDEX] *= read_data[dv++];
      t_mid_nuc_delay_default[CNUCINDEX] *= read_data[dv++];
      t_mid_nuc_delay_default[GNUCINDEX] *= read_data[dv++];
    }
    
    sens_default *= read_data[dv++];

    tau_R_m_default = read_data[dv++];
    tau_R_o_default = read_data[dv++];

    if (pset == 0 || pset==1){
      for (int vn=0;vn < 8;vn++)
        emp[vn] = read_data[dv++];

      emphasis_ampl_default = read_data[dv++];
      emphasis_width_default = read_data[dv++]; 
    }

    clonal_call_scale[0] = read_data[dv++];
    clonal_call_scale[1] = read_data[dv++];
    clonal_call_scale[2] = read_data[dv++];
    clonal_call_scale[3] = read_data[dv++];
    clonal_call_scale[4] = read_data[dv++];

  }
  else{
    fprintf(stderr, "emphasis file: %s \tstatus: %d\n",fname,status);
    exit (1);
    }

  delete [] line;  
  
  //output defaults used
    printf("GOPT parameters used: \n");
    printf("km_const: %f\t%f\t%f\t%f\n",kmax_default[TNUCINDEX],kmax_default[ANUCINDEX],kmax_default[CNUCINDEX],kmax_default[GNUCINDEX]);
    printf("krate: %f\t%f\t%f\t%f\n",krate_default[TNUCINDEX],krate_default[ANUCINDEX],krate_default[CNUCINDEX],krate_default[GNUCINDEX]);
    printf("d_coeff: %f\t%f\t%f\t%f\n",d_default[TNUCINDEX],d_default[ANUCINDEX],d_default[CNUCINDEX],d_default[GNUCINDEX]);
    printf("sigma_mult: %f\t%f\t%f\t%f\n",sigma_mult_default[0],sigma_mult_default[1],sigma_mult_default[2],sigma_mult_default[3]);
    printf("t_mid_nuc_delay: %f\t%f\t%f\t%f\n",t_mid_nuc_delay_default[0],t_mid_nuc_delay_default[1],t_mid_nuc_delay_default[2],t_mid_nuc_delay_default[3]);
    printf("sens: %f\n",sens_default);
    printf("tau_R_m: %f\n",tau_R_m_default);
    printf("tau_R_o: %f\n",tau_R_o_default);
    printf("emphasis: %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",emp[0],emp[1],emp[2],emp[3],emp[4],emp[5],emp[6],emp[7]);
    printf("emp_amplitude: \t%f\n",emphasis_ampl_default);
    printf("emp_width: \t%f\n",emphasis_width_default);
    printf("clonal_call_scale: %f\t%f\t%f\t%f\t%f\n",clonal_call_scale[0], clonal_call_scale[1], clonal_call_scale[2], clonal_call_scale[3], clonal_call_scale[4]);
    printf("\n");
}

char *GlobalDefaultsForBkgModel::xtalk_name = NULL;

void GlobalDefaultsForBkgModel::ReadXtalk(char *name)
{
      xtalk_name=strdup(name);
}

char *GlobalDefaultsForBkgModel::chipType = NULL;

void GlobalDefaultsForBkgModel::SetChipType(char *name)
{
      chipType=strdup(name);
}


