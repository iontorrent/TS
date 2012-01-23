/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgFitOptim.h"





//@TODO BAD CODE STYLE: function in header
void InitializeLevMarFitter(struct mat_table_build_instr *btbl,fit_instructions *instr)
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
      for (int f=0;f < NUMFB;f++)
        sum += btbl[i].affected_flows[f] * btbl[j].affected_flows[f];

      // if sum isn't 0, then these two parameters affect some of the same flows
      // so make an entry in the input table for the dot product of these two
      if (sum != 0)
        input_cnt++;
    }
  }

  // ok...now we know how many entries are both tables...go ahead and allocate them
  // input instructions include one row for every permuted pair, plus one row for each
  // independent parameter for construction of the RHS matrix
  struct mat_assy_input_line *mls = new struct mat_assy_input_line[input_cnt+np];

  // the output matrix gets one row for each indepdendent parameter
  struct delta_mat_output_line *ols = new struct delta_mat_output_line[np];

  // it's a good idea to clear these at the start
  memset(mls,0,sizeof(struct mat_assy_input_line[input_cnt+np]));
  memset(ols,0,sizeof(struct delta_mat_output_line[np]));

  // now build the input and output lines
  input_cnt = 0;
  for (int i=0;i<np;i++)
  {
    // we start at 'i' because we always count the case each parameter w/ itself...these
    // form the diagonal terms of the JTJ matrix
    for (int j=i;j<np;j++)
    {
      int sum = 0;
      for (int f=0;f < NUMFB;f++)
        sum += btbl[i].affected_flows[f] * btbl[j].affected_flows[f];

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
        for (int f=0;f < NUMFB;f++)
          mls[input_cnt].affected_flows[f] = btbl[i].affected_flows[f] * btbl[j].affected_flows[f];

        input_cnt++;
      }
    }

    // now add the input line for the rhs for parameter 'i'
    mls[input_cnt].comp1 = btbl[i].comp;
    mls[input_cnt].comp2 = YERR;
    mls[input_cnt].matId = RHS_MAT;
    mls[input_cnt].mat_row = i;
    mls[input_cnt].mat_col = 0;
    for (int f=0;f < NUMFB;f++)
      mls[input_cnt].affected_flows[f] = btbl[i].affected_flows[f];
    input_cnt++;

    // create an output line for parameter 'i'
    ols[i].delta_ndx = i;
    ols[i].param_ndx = btbl[i].param_ndx;
  }

  // fill in the top-level structure
  instr->input = mls;
  instr->output = ols;
  instr->input_len = input_cnt;
  instr->output_len = np;
}

void DumpBuildInstructionTable(struct mat_table_build_instr *tbl)
{
  for (int i=0;true;i++)
  {
    char *pcomp;

    switch (tbl[i].comp)
    {
      case TBL_END:
        pcomp="TBL_END    ";
        break;
      case DFDR:
        pcomp="DFDR       ";
        break;
      case DFDA:
        pcomp="DFDA       ";
        break;
      case DFDT0:
        pcomp="DFDT0      ";
        break;
      case DFDP:
        pcomp="DFDP       ";
        break;
      case DFDTSH:
        pcomp="DFDTSH     ";
        break;
      case DFDSIGMA:
        pcomp="DFDSIGMA   ";
        break;
      case DFDKRATE:
        pcomp="DFDKRATE   ";
        break;
      case DFDD:
        pcomp="DFDD       ";
        break;
      case DFDRDR:
        pcomp="DFDRDR     ";
        break;
      case DFDGAIN:
        pcomp="DFDGAIN    ";
        break;
      case YERR:
        pcomp="YERR       ";
        break;
      default:
        pcomp = "UNKNOWN";
    }

    printf("%s % 4d [",pcomp,tbl[i].param_ndx);
    for (int j=0;j < NUMFB-1;j++)
    {
      printf("%d,",tbl[i].affected_flows[j]);
    }
    printf("%d]\n",tbl[i].affected_flows[NUMFB-1]);

    if (tbl[i].comp == TBL_END)
      break;
  }
}

// creates a set of build instructions from one entry in the master fit table
void CreateBuildInstructions(struct master_fit_type_table *mfte, int *my_nuc)
{
  // if there is a high-level fit descriptor, create a set of build instructions
  // from the high-level descriptor
  if (mfte->fd != NULL)
  {
    struct fit_descriptor *fd = mfte->fd;
    int row_cnt = 0;

    // first figure out how many entries the build instruction table will need
    for (int i=0;fd[i].comp != TBL_END;i++)
    {
      switch (fd[i].ptype)
      {
        case ParamTypeAFlows:
        case ParamTypeCFlows:
        case ParamTypeGFlows:
        case ParamTypeAllFlow:
          row_cnt++;
          break;
        case ParamTypeNotKey:
          row_cnt += (NUMFB-KEY_LEN);
          break;
        case ParamTypePerFlow:
          row_cnt += NUMFB;
          break;
        case ParamTypePerNuc:
          row_cnt += NUMNUC;
          break;
        case ParamTypeAllButFlow0:
          row_cnt += NUMFB-1;
          break;
        default:
          break;
      }
    }

    // add one for the list-end indication
    row_cnt++;

    // allocate the build instruction table
    mfte->mb = new struct mat_table_build_instr[row_cnt];

    // zero it out
    memset(mfte->mb,0,sizeof(struct mat_table_build_instr[row_cnt]));

    // start at the beginning
    row_cnt = 0;

    // now create the rows
    for (int i=0;fd[i].comp != TBL_END;i++)
    {
      switch (fd[i].ptype)
      {
        case ParamTypeAllFlow:
          mfte->mb[row_cnt].comp = fd[i].comp;
          mfte->mb[row_cnt].param_ndx = fd[i].param_ndx;
          for (int j=0;j < NUMFB;j++) mfte->mb[row_cnt].affected_flows[j] = 1;
          row_cnt++;
          break;
        case ParamTypeNotKey:
          // create an independent paramter per flow except for key flows
          for (int row=KEY_LEN;row < NUMFB;row++)
          {
            mfte->mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // bead_params structure
            mfte->mb[row_cnt].param_ndx = fd[i].param_ndx+ (row-KEY_LEN);

            // indicate which flow this specific parameter affects
            mfte->mb[row_cnt].affected_flows[row] = 1;
            row_cnt++;
          }
          break;
        case ParamTypeAllButFlow0:
          // create an independent paramter per flow except for the first flow
          for (int row=1;row < NUMFB;row++)
          {
            mfte->mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // bead_params structure
            mfte->mb[row_cnt].param_ndx = fd[i].param_ndx+ (row-1);

            // indicate which flow this specific parameter affects
            mfte->mb[row_cnt].affected_flows[row] = 1;
            row_cnt++;
          }
          break;
        case ParamTypePerFlow:
          // create an independent paramter per flow
          for (int row=0;row < NUMFB;row++)
          {
            mfte->mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // bead_params structure
            mfte->mb[row_cnt].param_ndx = fd[i].param_ndx+row;

            // indicate which flow this specific parameter affects
            mfte->mb[row_cnt].affected_flows[row] = 1;
            row_cnt++;
          }
          break;
        case ParamTypePerNuc:
          // create an independent parameter per nucleotide
          for (int nuc=0;nuc < NUMNUC;nuc++)
          {
            mfte->mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // bead_params structure
            mfte->mb[row_cnt].param_ndx = fd[i].param_ndx+nuc;

            // indicate which flows this specific parameter affects
            for (int j=0;j < NUMFB;j++)
            {
              // TODO: very bad code here - isolate objects
              //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == nuc)
              if (my_nuc[j]==nuc)
                mfte->mb[row_cnt].affected_flows[j] = 1;
            }
            row_cnt++;
          }

          break;
        case ParamTypeAFlows:
          // create one parameter for all flows of a single nucleotide (A)
          mfte->mb[row_cnt].comp = fd[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mfte->mb[row_cnt].param_ndx = fd[i].param_ndx;

          // indicate which flows this specific parameter affects
          for (int j=0;j < NUMFB;j++)
          {
            // TODO: very bad to have object here
            //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == 1)
            if (my_nuc[j] == 1)
              mfte->mb[row_cnt].affected_flows[j] = 1;
          }
          row_cnt++;

          break;
        case ParamTypeCFlows:
          // create one parameter for all flows of a single nucleotide (C)
          mfte->mb[row_cnt].comp = fd[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mfte->mb[row_cnt].param_ndx = fd[i].param_ndx;

          // indicate which flows this specific parameter affects
          for (int j=0;j < NUMFB;j++)
          {
            // TODO: very bad to have object here
            //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == 2)
            if (my_nuc[j]==2)
              mfte->mb[row_cnt].affected_flows[j] = 1;
          }
          row_cnt++;

          break;
        case ParamTypeGFlows:
          // create one parameter for all flows of a single nucleotide (G)
          mfte->mb[row_cnt].comp = fd[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mfte->mb[row_cnt].param_ndx = fd[i].param_ndx;

          // indicate which flows this specific parameter affects
          for (int j=0;j < NUMFB;j++)
          {
            // TODO: Very bad to have object here
            //if (GlobalDefaultsForBkgModel::GetNucNdx(j) == 3)
            if (my_nuc[j]==3)
              mfte->mb[row_cnt].affected_flows[j] = 1;
          }
          row_cnt++;

          break;
        default:
          break;
      }
    }

    mfte->mb[row_cnt].comp = TBL_END;
  }

  // now create the input and output matrix maniuplation tables from the build
  // instructions
  if (mfte->mb != NULL)
    InitializeLevMarFitter(mfte->mb,& (mfte->fi));
}



