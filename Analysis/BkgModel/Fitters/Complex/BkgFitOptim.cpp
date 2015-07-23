/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgFitOptim.h"





//@TODO BAD CODE STYLE: function in header
void InitializeLevMarFitter(mat_table_build_instr *btbl,fit_instructions *instr, int flow_block_size)
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

  // it's a good idea to clear these at the start
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
  if (fd != NULL)
  {
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
    for (int i=0;fd[i].comp != TBL_END;i++)
    {
      switch (fd[i].ptype)
      {
        case ParamTypeAllFlow:
          mb[row_cnt].comp = fd[i].comp;
          mb[row_cnt].bead_params_func = fd[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fd[i].reg_params_func;
          for (int j=0;j < flow_block_size;j++) mb[row_cnt].SetAffectedFlow(j, 1);
          row_cnt++;
          break;
        case ParamTypeNotKey:
          // create an independent paramter per flow except for key flows
          for (int row=flow_key;row < flow_block_size;row++)
          {
            mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // BeadParams structure
            mb[row_cnt].bead_params_func = fd[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fd[i].reg_params_func;
            mb[row_cnt].array_index = row - flow_key;

            // indicate which flow this specific parameter affects
            mb[row_cnt].SetAffectedFlow(row, 1);
            row_cnt++;
          }
          break;
        case ParamTypeAllButFlow0:
          // create an independent paramter per flow except for the first flow
          for (int row=1;row < flow_block_size;row++)
          {
            mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // BeadParams structure
            mb[row_cnt].bead_params_func = fd[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fd[i].reg_params_func;
            mb[row_cnt].array_index = row - 1;

            // indicate which flow this specific parameter affects
            mb[row_cnt].SetAffectedFlow(row, 1);
            row_cnt++;
          }
          break;
        case ParamTypePerFlow:
          // create an independent paramter per flow
          for (int row=0;row < flow_block_size;row++)
          {
            mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each flow are assumed to be consecutive in the
            // bead_params structure
            mb[row_cnt].bead_params_func = fd[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fd[i].reg_params_func;
            mb[row_cnt].array_index = row;

            // indicate which flow this specific parameter affects
            mb[row_cnt].SetAffectedFlow(row, 1);
            row_cnt++;
          }
          break;
        case ParamTypePerNuc:
          // create an independent parameter per nucleotide
          for (int nuc=0;nuc < NUMNUC;nuc++)
          {
            mb[row_cnt].comp = fd[i].comp;

            // individual parameters for each nucleotide are assumed to be consecutive in the
            // BeadParams structure
            mb[row_cnt].bead_params_func = fd[i].bead_params_func;
            mb[row_cnt].reg_params_func  = fd[i].reg_params_func;
            mb[row_cnt].array_index = nuc;

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
          mb[row_cnt].comp = fd[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mb[row_cnt].bead_params_func = fd[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fd[i].reg_params_func;

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
          mb[row_cnt].comp = fd[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mb[row_cnt].bead_params_func = fd[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fd[i].reg_params_func;

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
          mb[row_cnt].comp = fd[i].comp;

          // the table entry should point to the specific parameter for this nucleotide
          mb[row_cnt].bead_params_func = fd[i].bead_params_func;
          mb[row_cnt].reg_params_func  = fd[i].reg_params_func;

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



