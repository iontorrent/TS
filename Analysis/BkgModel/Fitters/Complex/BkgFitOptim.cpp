/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgFitOptim.h"
#include "BkgFitStructures.h"

using namespace std;


#define DVAL(X) {X,#X}

std::map<PartialDerivComponent, std::string> PartialDerivNames = {
             DVAL(TBL_END),DVAL(TBL_END),DVAL(DFDP),DVAL(DFDR),
             DVAL(DFDPDM),DVAL(DFDGAIN),DVAL(DFDA),DVAL(DFDDKR),
             DVAL(DFDSIGMA),DVAL(DFDTSH),DVAL(DFDERR),DVAL(DFDT0),
             DVAL(DFDTAUMR),DVAL(DFDTAUOR),DVAL(DFDRDR),DVAL(DFDKRATE),
             DVAL(DFDSENS),DVAL(DFDD),DVAL(DFDPDR),DVAL(DFDMR),DVAL(DFDKMAX),
             DVAL(DFDT0DLY),DVAL(DFDSMULT),DVAL(DFDTAUE),DVAL(YERR),DVAL(FVAL)
            };

std::string ComponentName( PartialDerivComponent comp )
{
    std::string readable_name;
    try {
        readable_name = PartialDerivNames[comp];
        if(readable_name.substr(0,3)=="DFD")
            readable_name=readable_name.substr(3); //cleanup the readable name by removing redundant part
    } catch (const std::out_of_range&) {
        readable_name = "UNKNOWN";
    }
    return readable_name;
}


void InitializeLevMarFitter(const mat_table_build_instr *btbl,fit_instructions *instr, int flow_block_size)
{
  int np;

  // first things first...coun't the number of independent parameters in the table
  for (np=0;btbl[np].comp != TBL_END;np++) ;

  // now figure out how many entries there will be in the 'input lines' array
  // this is a permutation of every independent parameter w/ every other indepdent
  // parameter where both affect at least some of the same data points
  int input_cnt = 0;
  for (int i=0;i<np;i++)
  {
    // we start at 'i' because we always count the case each parameter w/ itself...these
    // form the diagonal terms of the JTJ matrix
    for (int j=i;j<np;j++)
    {
      int sum = 0;
      for (int f=0;f < flow_block_size;f++)
        sum += btbl[i].GetAffectedFlow(f) * btbl[j].GetAffectedFlow(f);

      // if sum isn't 0, then these two parameters affect some of the same flows
      // so make an entry in the input table for the dot product of these two
      if (sum != 0)
        input_cnt++;
    }
  }

  // ok...now we know how many entries are both tables...go ahead and allocate them
  // input instructions include one row for every permuted pair, plus one row for each
  // independent parameter for construction of the RHS matrix
  mat_assy_input_line *mls = new mat_assy_input_line[input_cnt+np];
  for(int i = 0 ; i < input_cnt+np ; ++i )
  {
    mls[i].realloc( flow_block_size );
  }

  // the output matrix gets one row for each indepdendent parameter
  struct delta_mat_output_line *ols = new struct delta_mat_output_line[np];

  // now build the input and output lines
  input_cnt = 0;
  for (int i=0;i<np;i++)
  {
    // we start at 'i' because we always count the case each parameter w/ itself...these
    // form the diagonal terms of the JTJ matrix
    for (int j=i;j<np;j++)
    {
      int sum = 0;
      for (int f=0;f < flow_block_size;f++)
        sum += btbl[i].GetAffectedFlow(f) * btbl[j].GetAffectedFlow(f);

      if (sum != 0)
      {
        // create the input line for this parameter pair
        mls[input_cnt].comp1 = btbl[i].comp;
        mls[input_cnt].comp2 = btbl[j].comp;
        mls[input_cnt].matId = JTJ_MAT;
        mls[input_cnt].mat_row = j;
        mls[input_cnt].mat_col = i;

        // AND the two affected_flow arrays together to get the flows affected by both of
        // these parameters
        for (int f=0;f < flow_block_size;f++)
          mls[input_cnt].SetAffectedFlow(f, btbl[i].GetAffectedFlow(f) * btbl[j].GetAffectedFlow(f) );

        input_cnt++;
      }
    }

    // now add the input line for the rhs for parameter 'i'
    mls[input_cnt].comp1 = btbl[i].comp;
    mls[input_cnt].comp2 = YERR;
    mls[input_cnt].matId = RHS_MAT;
    mls[input_cnt].mat_row = i;
    mls[input_cnt].mat_col = 0;
    for (int f=0;f < flow_block_size;f++)
      mls[input_cnt].SetAffectedFlow(f, btbl[i].GetAffectedFlow(f) );
    input_cnt++;

    // create an output line for parameter 'i'
    ols[i].delta_ndx = i;
    ols[i].bead_params_func = btbl[i].bead_params_func;
    ols[i].reg_params_func  = btbl[i].reg_params_func;
    ols[i].array_index      = btbl[i].array_index;
    ols[i].name             = btbl[i].name;
  }

  // fill in the top-level structure
  instr->input = mls;
  instr->output = ols;
  instr->input_len = input_cnt;
  instr->output_len = np;
}




void DumpBuildInstructionTable(mat_table_build_instr *tbl, int flow_block_size)
{
  for (int i=0;true;i++)
  {
    std::string pcomp;

    pcomp = ComponentName( tbl[i].comp );

    printf("%s % 4d [",pcomp.c_str(), tbl[i].array_index );
    for (int j=0;j < flow_block_size-1;j++)
    {
      printf("%d,",tbl[i].GetAffectedFlow(j));
    }
    printf("%d]\n",tbl[i].GetAffectedFlow(flow_block_size-1));

    if (tbl[i].comp == TBL_END)
      break;
  }
}

// creates a set of build instructions from one entry in the master fit table
void master_fit_type_entry::CreateBuildInstructions(const int *my_nuc, int flow_key, int flow_block_size)
{
  // if there is a high-level fit descriptor, create a set of build instructions
  // from the high-level descriptor
  if (fds.size()>0)
  {
    int row_cnt = 0;

    // first figure out how many entries the build instruction table will need
    for (int i=0;fds[i].comp != TBL_END;i++)
    {
      switch (fds[i].ptype)
      {
        case ParamTypeAFlows:
        case ParamTypeCFlows:
        case ParamTypeGFlows:
        case ParamTypeAllFlow:
          row_cnt++;
          break;
        case ParamTypeNotKey:
          row_cnt += std::max(0, flow_block_size-flow_key);
          break;
        case ParamTypePerFlow:
          row_cnt += flow_block_size;
          break;
        case ParamTypePerNuc:
          row_cnt += NUMNUC;
          break;
        case ParamTypeAllButFlow0:
          row_cnt += flow_block_size-1;
          break;
        default:
          break;
      }
    }

    // add one for the list-end indication
    row_cnt++;

    // allocate the build instruction table
    mb = new mat_table_build_instr[row_cnt];

    // zero it out
    for(int i = 0 ; i < row_cnt ; ++i )
    {
      mb[i].realloc( flow_block_size );
      for(int j = 0 ; j < flow_block_size ; ++j )
      {
        mb[i].SetAffectedFlow(j,0);
      }
    }

    // start at the beginning
    row_cnt = 0;

    // now create the rows
    for (int i=0;fds[i].comp != TBL_END;i++)
    {
      switch (fds[i].ptype)
      {
        case ParamTypeAllFlow:
          mb[row_cnt].comp = fds[i].comp;
          mb[row_cnt].bead_params_func = fds[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
          for (int j=0;j < flow_block_size;j++) mb[row_cnt].SetAffectedFlow(j, 1);
          mb[row_cnt].name = ComponentName(fds[i].comp);
          row_cnt++;
          break;
        case ParamTypeNotKey:
          // create an independent paramter per flow except for key flows
          for (int row=flow_key;row < flow_block_size;row++)
          {
            mb[row_cnt].comp = fds[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // BeadParams structure
            mb[row_cnt].bead_params_func = fds[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
            mb[row_cnt].array_index = row - flow_key;
            mb[row_cnt].name = ComponentName(fds[i].comp)+"_f"+std::to_string(row);

            // indicate which flow this specific parameter affects
            mb[row_cnt].SetAffectedFlow(row, 1);
            row_cnt++;
          }
          break;
        case ParamTypeAllButFlow0:
          // create an independent paramter per flow except for the first flow
          for (int row=1;row < flow_block_size;row++)
          {
            mb[row_cnt].comp = fds[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // BeadParams structure
            mb[row_cnt].bead_params_func = fds[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
            mb[row_cnt].array_index = row - 1;
            mb[row_cnt].name = ComponentName(fds[i].comp)+"_f"+std::to_string(row);

            // indicate which flow this specific parameter affects
            mb[row_cnt].SetAffectedFlow(row, 1);
            row_cnt++;
          }
          break;
        case ParamTypePerFlow:
          // create an independent paramter per flow
          for (int row=0;row < flow_block_size;row++)
          {
            mb[row_cnt].comp = fds[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // bead_params structure
            mb[row_cnt].bead_params_func = fds[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
            mb[row_cnt].array_index = row;
            mb[row_cnt].name = ComponentName(fds[i].comp)+"_f"+std::to_string(row);

            // indicate which flow this specific parameter affects
            mb[row_cnt].SetAffectedFlow(row, 1);
            row_cnt++;
          }
          break;
        case ParamTypePerNuc:
          // create an independent parameter per nucleotide
          for (int nuc=0;nuc < NUMNUC;nuc++)
          {
            mb[row_cnt].comp = fds[i].comp;

            // individual parameters for each nucleotide are assumed to be consecutive in the
            // BeadParams structure
            mb[row_cnt].bead_params_func = fds[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
            mb[row_cnt].array_index = nuc;
            mb[row_cnt].name = ComponentName(fds[i].comp)+"_n"+std::to_string(nuc);

            // indicate which flows this specific parameter affects
            for (int j=0;j < flow_block_size;j++)
            {
              // TODO: very bad code here - isolate objects
              //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == nuc)
              if (my_nuc[j]==nuc)
                mb[row_cnt].SetAffectedFlow(j, 1);
            }
            row_cnt++;
          }

          break;
        case ParamTypeAFlows:
          // create one parameter for all flows of a single nucleotide (A)
          mb[row_cnt].comp = fds[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mb[row_cnt].bead_params_func = fds[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
          mb[row_cnt].name = ComponentName(fds[i].comp);

          // indicate which flows this specific parameter affects
          for (int j=0;j < flow_block_size;j++)
          {
            // TODO: very bad to have object here
            //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == 1)
            if (my_nuc[j] == 1)
              mb[row_cnt].SetAffectedFlow(j, 1);
          }
          row_cnt++;

          break;
        case ParamTypeCFlows:
          // create one parameter for all flows of a single nucleotide (C)
          mb[row_cnt].comp = fds[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mb[row_cnt].bead_params_func = fds[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
          mb[row_cnt].name = ComponentName(fds[i].comp);

          // indicate which flows this specific parameter affects
          for (int j=0;j < flow_block_size;j++)
          {
            // TODO: very bad to have object here
            //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == 2)
            if (my_nuc[j]==2)
              mb[row_cnt].SetAffectedFlow(j, 1);
          }
          row_cnt++;

          break;
        case ParamTypeGFlows:
          // create one parameter for all flows of a single nucleotide (G)
          mb[row_cnt].comp = fds[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mb[row_cnt].bead_params_func = fds[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fds[i].reg_params_func;
          mb[row_cnt].name = ComponentName(fds[i].comp)+"_G";

          // indicate which flows this specific parameter affects
          for (int j=0;j < flow_block_size;j++)
          {
            // TODO: Very bad to have object here
            //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == 3)
            if (my_nuc[j]==3)
              mb[row_cnt].SetAffectedFlow(j, 1);
          }
          row_cnt++;

          break;
        default:
          break;
      }
    }

    mb[row_cnt].comp = TBL_END;
  }

  // now create the input and output matrix maniuplation tables from the build
  // instructions
  if (mb != NULL)
    InitializeLevMarFitter(mb,& (fi), flow_block_size);
}

void master_fit_type_table::set_base_bkg_model_fit_type()
{
    // set fit_type_hash_table
    fit_type_hash_table = { // nested list-initialization
       {"FitWellAmpl", {"wellAmpl","TableEnd"}},
       {"FitWellAmplBuffering", {"wellR","wellAmpl","TableEnd"}},
       {"FitWellPostKey", {"wellR","wellCopies","wellDmult","wellAmplPostKey","TableEnd"}},
       {"FitWellAll", {"wellR","wellCopies","wellDmult","wellAmpl","wellKmult","TableEnd"}},
       {"FitWellPostKeyNoDmult", {"wellR","wellCopies","wellAmplPostKey","TableEnd"}},
       // region-wide fits
       {"FitRegionTmidnucPlus", {"R","Copies","Ampl","TMidNuc","TableEnd"}},
       {"FitRegionInit2",      {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","D","NucModifyRatio","TauRM","TauRO","RatioDrift","CopyDrift","TableEnd"}},
       {"FitRegionInit2TauE",  {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","D","NucModifyRatio","TauE","RatioDrift","CopyDrift","TableEnd"}},
       {"FitRegionInit2TauENoRDR",{"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","D","NucModifyRatio","TauE","CopyDrift","TableEnd"}},
       {"FitRegionFull",       {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","D","NucModifyRatio","TauRM","TauRO","RatioDrift","CopyDrift","TableEnd"}},
       {"FitRegionFullTauE",   {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","D","NucModifyRatio","TauE","RatioDrift","CopyDrift","TableEnd"}},
       {"FitRegionFullTauENoRDR", {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","D","NucModifyRatio","TauE","CopyDrift","TableEnd"}},
       {"FitRegionInit2NoRDR", {"R","Copies","Ampl","TShift","TMidNuc","Sigma","TMidNucDelay","SigmaMult","Krate","D","NucModifyRatio","TauRM","TauRO","CopyDrift","TableEnd"}},
       {"FitRegionFullNoRDR",  {"R","Copies","Ampl","TShift","TMidNuc","Sigma","TMidNucDelay","SigmaMult","Krate","D","NucModifyRatio","TauRM","TauRO","CopyDrift","TableEnd"}},
       {"FitRegionTimeVarying",{"TMidNuc","RatioDrift","CopyDrift","TableEnd"}},
       {"FitRegionDarkness",   {"Darkness","Ampl","TableEnd"}},
       //region-wide fits without diffusion
       {"FitRegionInit2TauENoD",  {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","NucModifyRatio","TauE","RatioDrift","CopyDrift","TableEnd"}},
       {"FitRegionInit2TauENoRDRNoD", {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","NucModifyRatio","TauE","CopyDrift","TableEnd"}},
       {"FitRegionFullTauENoD",   {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","NucModifyRatio","TauE","RatioDrift","CopyDrift","TableEnd"}},
       {"FitRegionFullTauENoRDRNoD",  {"R","Copies","Ampl","TShift","TMidNuc","Sigma","Krate","NucModifyRatio","TauE","CopyDrift","TableEnd"}},
       };

    // set each entry for base_bkg_model_fit_type
    for (auto p: fit_type_hash_table)
    {
      CreateBkgModelFitType(p.first, p.second);
      base_bkg_model_fit_type[p.first].CreateBuildInstructions(_nucIdentity, max(_flowKey - _flowStart, 0 ), _flowBlockSize);
    }
}


fit_descriptor master_fit_type_table::make_fit_param_entry(std::string param)
{
  fit_descriptor fd;
  if (param == "wellAmpl")            fd = fit_descriptor({DFDA,      & BeadParams::AccessAmpl,   0,      ParamTypePerFlow });
  else if (param == "wellR")          fd = fit_descriptor({DFDR,      & BeadParams::AccessR,      0,      ParamTypeAllFlow });
  else if (param == "wellCopies")     fd = fit_descriptor({DFDP,      & BeadParams::AccessCopies, 0,      ParamTypeAllFlow });
  else if (param == "wellDmult")      fd = fit_descriptor({DFDPDM,    & BeadParams::AccessDmult,  0,      ParamTypeAllFlow });
  else if (param == "wellKmult")      fd = fit_descriptor({DFDDKR,    & BeadParams::AccessKmult,  0,      ParamTypePerFlow });
  else if (param == "wellAmplPostKey")fd = fit_descriptor({DFDA,      & BeadParams::AccessAmplPostKey, 0, ParamTypeNotKey  });

  else if (param == "Ampl")           fd = fit_descriptor({DFDA,      0, & reg_params::AccessAmpl,        ParamTypePerFlow });
  else if (param == "R")              fd = fit_descriptor({DFDR,      0, & reg_params::AccessR,           ParamTypeAllFlow });
  else if (param == "Copies")         fd = fit_descriptor({DFDP,      0, & reg_params::AccessCopies,      ParamTypeAllFlow });
  else if (param == "D")              fd = fit_descriptor({DFDD,      0, & reg_params::AccessD,           ParamTypePerNuc });
  else if (param == "Darkness")       fd = fit_descriptor({DFDERR,    0, & reg_params::AccessDarkness,    ParamTypeAllFlow });
  else if (param == "TShift")         fd = fit_descriptor({DFDTSH,    0, & reg_params::AccessTShift,      ParamTypeAllFlow });
  else if (param == "TMidNuc")        fd = fit_descriptor({DFDT0,     0, & reg_params::AccessTMidNuc,     ParamTypeAllFlow });
  else if (param == "TMidNucDelay")   fd = fit_descriptor({DFDT0DLY,  0, & reg_params::AccessTMidNucDelay,ParamTypePerNuc });
  else if (param == "Sigma")          fd = fit_descriptor({DFDSIGMA,  0, & reg_params::AccessSigma,       ParamTypeAllFlow });
  else if (param == "SigmaMult")      fd = fit_descriptor({DFDSMULT,  0, & reg_params::AccessSigmaMult,   ParamTypePerNuc });
  else if (param == "Krate")          fd = fit_descriptor({DFDKRATE,  0, & reg_params::AccessKrate,       ParamTypePerNuc });
  else if (param == "NucModifyRatio") fd = fit_descriptor({DFDMR,     0, & reg_params::AccessNucModifyRatio, ParamTypePerNuc });
  else if (param == "TauRM")          fd = fit_descriptor({DFDTAUMR,  0, & reg_params::AccessTauRM,       ParamTypeAllFlow });
  else if (param == "TauRO")          fd = fit_descriptor({DFDTAUOR,  0, & reg_params::AccessTauRO,       ParamTypeAllFlow });
  else if (param == "TauE")           fd = fit_descriptor({DFDTAUE,   0, & reg_params::AccessTauE,        ParamTypeAllFlow });
  else if (param == "RatioDrift")     fd = fit_descriptor({DFDRDR,    0, & reg_params::AccessRatioDrift,  ParamTypeAllFlow });
  else if (param == "CopyDrift")      fd = fit_descriptor({DFDPDR,    0, & reg_params::AccessCopyDrift,   ParamTypeAllFlow });
  else if (param == "TableEnd")      fd = fit_descriptor({TBL_END,    0, 0,   ParamTableEnd });
  else { 
    throw FitTypeException(param + ": Not a valid bead/reg param \n");
  }

  return fd;
}

void master_fit_type_table::CreateBkgModelFitType(const string &fitName, const std::vector<std::string> &fit_params) {
  master_fit_type_entry mfte;
  mfte.name = fitName.c_str();
  mfte.fds.resize(fit_params.size());
  for (size_t i=0; i<fit_params.size(); i++) {
        mfte.fds[i] = make_fit_param_entry(fit_params[i]);
  }
  base_bkg_model_fit_type[fitName] = mfte;
}

fit_instructions *master_fit_type_table::GetFitInstructionsByName(const string& name)
{
  pthread_mutex_lock(&addFit);
  auto it = base_bkg_model_fit_type.find(name);
  if (it != base_bkg_model_fit_type.end()) {
    pthread_mutex_unlock(&addFit);
    return &(it->second.fi);
  }
  else {
    pthread_mutex_unlock(&addFit);
    throw FitTypeException(name + " :Not a valid fit type name\n");
  }
}

const vector<fit_descriptor>& master_fit_type_table::GetFitDescriptorByName(const string& name)
{
  pthread_mutex_lock(&addFit);
  auto it = base_bkg_model_fit_type.find(name);
  if (it != base_bkg_model_fit_type.end()) {
    pthread_mutex_unlock(&addFit);
    return it->second.fds;
  }
  else {
    pthread_mutex_unlock(&addFit);
    throw FitTypeException(name + " :Not a valid fit type name\n");
  }
}


// @TODO:  Potential bug here due to historical optimization
// does not update for further blocks of flows
// Needs to update as blocks of flows arrive
// fit instructions are optimized for first block of flows only
// can be in error for later blocks of flows.
master_fit_type_table::master_fit_type_table( 
    const FlowMyTears & tears, 
    int flow_start, 
    int flow_key, 
    int flow_block_size
  ) :_flowStart(flow_start), _flowKey(flow_key), _flowBlockSize(flow_block_size)
{
  _nucIdentity = new int[_flowBlockSize];
  tears.GetFlowOrderBlock(_nucIdentity, _flowStart, _flowStart + _flowBlockSize );

  // for thread safe adding of dynamic fit types during  signal processing

  pthread_mutex_init(&addFit, NULL);

  set_base_bkg_model_fit_type();

  // go through the master table of fit types and generate all the build
  // instructions for each type of fitting we are going to do
 
  //for (auto it=base_bkg_model_fit_type.begin(); it != base_bkg_model_fit_type.end(); ++it) {
  //  it->second.CreateBuildInstructions(_nucIdentity, max(_flowKey - _flowStart, 0 ), _flowBlockSize);
  //}
}

master_fit_type_table::~master_fit_type_table()
{
  for (auto it=base_bkg_model_fit_type.begin(); it != base_bkg_model_fit_type.end(); ++it) {
    master_fit_type_entry *ft = &(it->second);

    // make sure there is a high-level descriptor for this row
    // if there wasn't one, then the row might contain a hard link to
    // a statically allocated matrix build instruction which we don't
    // want to free
    if (ft->fds.size() > 0)
    {
      if (ft->mb != NULL)
      {
        delete [](ft->mb);
        ft->mb = NULL;
      }
    }

    if (ft->fi.input != NULL)
      delete [] ft->fi.input;

    if (ft->fi.output != NULL)
      delete [] ft->fi.output;

    ft->fi.input  = NULL;
    ft->fi.output = NULL;
  }

  delete [] _nucIdentity;
  pthread_mutex_destroy(&addFit);
}

void master_fit_type_table::addBkgModelFitType(
  const string& fitName,
  const vector<string>& paramNames)
{
  pthread_mutex_lock(&addFit);

  try {
    auto it = base_bkg_model_fit_type.find(fitName);
    if (it == base_bkg_model_fit_type.end()) {
      CreateBkgModelFitType(fitName, paramNames);
      base_bkg_model_fit_type[fitName].CreateBuildInstructions(_nucIdentity, max(_flowKey - _flowStart, 0 ), _flowBlockSize);
    }
  }
  catch(exception &ft) {
    pthread_mutex_unlock(&addFit);
    throw FitTypeException(ft.what());
  }
  pthread_mutex_unlock(&addFit);
}
