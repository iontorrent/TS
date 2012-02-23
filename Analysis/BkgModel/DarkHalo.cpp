/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "DarkHalo.h"


Halo::Halo()
{
  dark_matter_compensator  = NULL;
  for (int i=0; i<4; i++)
    dark_nuc_comp[i] = NULL;
  nuc_flow_t = 0;
}

void Halo::Alloc (int _npts)
{
  // set up buffer here as time_c tells me how many points I have
  npts = _npts;
  nuc_flow_t = NUMNUC * npts; // some objects repeat per nucleotide
  dark_matter_compensator  = new float[nuc_flow_t];
  for (int i=0; i<NUMNUC; i++)
    dark_nuc_comp[i] = &dark_matter_compensator[i*npts];
  ResetDarkMatter();
}

void Halo::ResetDarkMatter()
{
  memset (dark_matter_compensator,0,sizeof (float[nuc_flow_t]));
  memset (weight,0,sizeof(float[NUMNUC]));
}

void Halo::Delete()
{
  for (int i=0; i<NUMNUC; i++)
    dark_nuc_comp[i] = NULL;
  if (dark_matter_compensator != NULL) delete [] dark_matter_compensator;

}

void Halo::AccumulateDarkMatter(float *residual, int inuc)
{
        for (int i=0;i<npts;i++)
        {
          dark_nuc_comp[inuc][i] += residual[i];
        }
        weight[inuc]++;
}

void Halo::NormalizeDarkMatter ()
{
  // now normalize all the averages.  If any don't contain enough to make a good average
  // replace the error term with all ones
  for (int nnuc=0;nnuc < NUMNUC;nnuc++)
  {
    float *et = dark_nuc_comp[nnuc];

    if (weight[nnuc] > 1.0)
    {
      for (int i=0;i<npts;i++)
        et[i] = et[i]/weight[nnuc];
    }
    else
    {
      for (int i=0;i<npts;i++)
        et[i] = 0.0;  // no data, no offset
    }
    weight[nnuc]=1.0; //
  }

}

void Halo::DumpDarkMatterTitle (FILE *my_fp)
{
  // ragged columns because of variable time compression
  fprintf (my_fp, "col\trow\tNucID\t");
  for (int j=0; j<40; j++)
    fprintf (my_fp,"V%d\t",j);
  fprintf (my_fp,"Neat");
  fprintf (my_fp,"\n");
}

void Halo::DumpDarkMatter (FILE *my_fp, int x, int y, float darkness)
{
  // this is a little tricky across regions, as time compression is somewhat variable
  // 4 lines, one per nuc_flow
  for (int NucID=0; NucID<NUMNUC; NucID++)
  {
    fprintf (my_fp, "%d\t%d\t%d\t", x,y, NucID);
    int npts = nuc_flow_t/4;
    int j=0;
    for (; j<npts; j++)
      fprintf (my_fp,"%0.3f\t", dark_matter_compensator[NucID*npts+j]);
    int max_npts = 40;  // always at least this much time compression?
    for (;j<max_npts; j++)
      fprintf (my_fp,"%0.3f\t", 0.0);
    fprintf (my_fp,"%f",darkness);
    fprintf (my_fp, "\n");
  }
}
