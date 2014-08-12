/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <algorithm>
#include <cstdio>
#include "DarkHalo.h"

using namespace std;

Halo::Halo()
{
  for (int i=0; i<NUMDM; i++)
    dark_nuc_comp[i] = NULL;
  nuc_flow_t = 0;
  mytype = Unspecified;
  training_only=true;
}

void Halo::Alloc (int _npts)
{
  // set up buffer here as time_c tells me how many points I have
  npts = _npts;
  nuc_flow_t = NUMDM * npts; // some objects repeat per nucleotide
  dark_matter_compensator.resize(nuc_flow_t);
  SetupDarkNumComp();
  ResetDarkMatter();
}

void Halo::SetupDarkNumComp()
{
  for (int i=0; i<NUMDM; i++)
    dark_nuc_comp[i] = &dark_matter_compensator[i*npts];
}

void Halo::ResetDarkMatter()
{
  fill(dark_matter_compensator.begin(), dark_matter_compensator.end(), 0.0f);
  fill(weight, weight+NUMDM, 0.0f);
}

void Halo::Delete()
{
  dark_matter_compensator.resize(0);
  for (int i=0; i<NUMDM; i++)
    dark_nuc_comp[i] = NULL;
}

void Halo::AccumulateDarkMatter (float *residual, int inuc)
{
  if (not dark_matter_compensator.empty())
  {
    for (int i=0;i<npts;i++)
    {
      dark_nuc_comp[inuc][i] += residual[i];
    }
    weight[inuc]++;
  }
  else
    printf ("dark matter not allocated, illegal access\n");
}

void Halo::NormalizeDarkMatter ()
{
  // now normalize all the averages.  If any don't contain enough to make a good average
  // replace the error term with all ones
  for (int nnuc=0;nnuc < NUMDM;nnuc++)
  {
    float *et = dark_nuc_comp[nnuc];

    if (weight[nnuc] >= 1.0f)
    {
      for (int i=0;i<npts;i++)
        et[i] = et[i]/weight[nnuc];
    }
    else
    {
      for (int i=0;i<npts;i++)
        et[i] = 0.0f;  // no data, no offset
    }
    weight[nnuc]=1.0f; //
  }

}

void Halo::DumpDarkMatterTitle (FILE *my_fp)
{
  if (not dark_matter_compensator.empty())
  {
    // ragged columns because of variable time compression
    fprintf (my_fp, "col\trow\tNucID\t");
    for (int j=0; j<MAX_COMPRESSED_FRAMES; j++)
      fprintf (my_fp,"V%d\t",j);
    fprintf (my_fp,"Neat");
    fprintf (my_fp,"\n");
  }
}

void Halo::DumpDarkMatter (FILE *my_fp, int x, int y, float darkness)
{
  if (not dark_matter_compensator.empty())
  {
    // this is a little tricky across regions, as time compression is somewhat variable
    // 4 lines, one per nuc_flow
    for (int NucID=0; NucID<NUMDM; NucID++)
    {
      fprintf (my_fp, "%d\t%d\t%d\t", x,y, NucID);
      int npts = nuc_flow_t/NUMDM;
      int j=0;
      for (; j<npts; j++)
        fprintf (my_fp,"%0.3f\t", dark_matter_compensator[NucID*npts+j]);
      int max_npts = MAX_COMPRESSED_FRAMES;  // always at least this much time compression?
      for (;j<max_npts; j++)
        fprintf (my_fp,"%0.3f\t", 0.0);
      fprintf (my_fp,"%f",darkness);
      fprintf (my_fp, "\n");
    }
  }
}
