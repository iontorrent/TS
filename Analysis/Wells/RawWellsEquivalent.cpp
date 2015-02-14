/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <assert.h>
#include <iostream>
#include <stdlib.h>

#include "Utils.h"
#include "NumericalComparison.h"
#include "RawWells.h"
#include "OptArgs.h"
#include "IonVersion.h"

using namespace std;

NumericalComparison<double> CompareWells(const string &queryFile, const string &goldFile, 
					 float epsilon, double maxAbsVal) {
  
  NumericalComparison<double> compare(epsilon);
  string queryDir, queryWells, goldDir, goldWells;
  FillInDirName(queryFile, queryDir, queryWells);
  FillInDirName(goldFile, goldDir, goldWells);

  RawWells queryW(queryDir.c_str(), queryWells.c_str());
  RawWells goldW(goldDir.c_str(), goldWells.c_str());
  
  struct WellData goldData;
  goldData.flowValues = NULL;
  struct WellData queryData;
  queryData.flowValues = NULL;
  cout << "Opening query." << endl;
  queryW.OpenForRead();
  cout << "Opening gold." << endl;
  goldW.OpenForRead();

  // check if any 1.wells is saved as unsigned short
  bool ushortg = goldW.GetSaveAsUShort();
  bool ushortq = queryW.GetSaveAsUShort();
  bool difType = false;
  float* copies = NULL;
  string fileName;
  unsigned int numCols = 0;
  unsigned int dsSize0 = 0;

  if(ushortg && (!ushortq))
  {
    difType = true;
    cout << "RawWellsEquivalent WARNING: " << goldFile << " is saved as unsigned short and \n" << queryFile << " is saved as float. \nYou may want to re-run it with a bigger epsilon." << endl;
    fileName = goldFile;
    numCols = goldW.NumCols();
    dsSize0 = numCols * goldW.NumRows();
  }

  if(ushortq && (!ushortg))
  {
    difType = true;
    cout << "RawWellsEquivalent WARNING: " << queryFile << " is saved as unsigned short and \n" << goldFile << " is saved as float. \nYou may want to re-run it with a bigger epsilon." << endl;
    fileName = queryFile;
    numCols = queryW.NumCols();
    dsSize0 = numCols * queryW.NumRows();
  }

  if(difType)
  {
    hid_t root = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if(root < 0)				
    {				
      cerr << "RawWellsEquivalent ERROR: Fail to open " << fileName << endl;
      cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because number of copies is not available from " << fileName << endl;
    }	
	else
	{
      
      H5G_info_t group_info;
      group_info.nlinks = 0;
      if(H5Gget_info(root, &group_info) < 0)
	  {
        H5Fclose(root);
        cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because wells_copies is not available from " << fileName << endl;
	  }
	  else
	  {
        char name[10];
        string sName;
        for(unsigned int i = 0; i < group_info.nlinks; ++i)
        {
          int size = H5Gget_objname_by_idx(root, i, NULL, 0);
          if(H5Gget_objname_by_idx(root, i, name, size + 1) < 0)
		  {
            H5Fclose(root);
            cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because wells_copies is not available from " << fileName << endl;
		  }
		  else
		  {
            sName = name;
            if(sName == "wells_copies")
			{
              break;
			}
		  }
		}
        if(sName == "wells_copies")
		{
          hid_t ds = H5Dopen2(root, "wells_copies", H5P_DEFAULT);
          if(ds < 0)
		  {
            H5Fclose(root);
            cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because wells_copies cannot be opened from " << fileName << endl;
		  }
		  else
		  {
            hid_t dataSpace = H5Dget_space(ds);
            if(dataSpace < 0)
			{
              H5Dclose(ds);
              H5Fclose(root);
              cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because dataSpace is not able to open in wells_copies from " << fileName << endl;          
			}
			else
			{
              hssize_t dsSize = H5Sget_simple_extent_npoints(dataSpace);
              H5Sclose(dataSpace);
              if(dsSize != dsSize0)
			  {
                H5Dclose(ds);
                H5Fclose(root);
                cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because wells_copies data set size is wrong in " << fileName << endl;          
			  }
			  else
			  {
                copies = new float[dsSize0];
                if(copies == NULL)
				{
                  H5Dclose(ds);
                  H5Fclose(root);
                  cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because allocating copies fails." << endl;          
			    }
				else
				{
                  herr_t ret = H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, copies);
                  H5Dclose(ds);
                  H5Fclose(root);
                  if(ret < 0)
				  {
				    delete [] copies;
				    copies = NULL;
                    cerr << "RawWellsEquivalent ERROR: wells files may not be comparable because loading wells_copies fails in " << fileName << endl;          
				  }
			    }
			  }
			}
		  }
		}
	  }
	}
  }

  unsigned int numFlows = goldW.NumFlows();
  while( !queryW.ReadNextRegionData(&queryData) ) {
    assert(!goldW.ReadNextRegionData(&goldData));
    for (unsigned int i = 0; i < numFlows; i++) {
      if(difType && copies)
	  {
		float flowValues2 = -1.0;
		int index = -1;
		if(ushortg)
		{
		  flowValues2 = goldData.flowValues[i];
		  index = goldData.y * numCols + goldData.x;
		}
		else if(ushortq)
		{
		  flowValues2 = queryData.flowValues[i];
		  index = queryData.y * numCols + queryData.x;
		}

		if(isfinite(flowValues2) && copies[index] > 0)
		{
		  flowValues2 *= copies[index];
		}

        if(isfinite(flowValues2) && fabs(flowValues2) < maxAbsVal)
		{
		  if(ushortg && isfinite(queryData.flowValues[i]) && fabs(queryData.flowValues[i]) < maxAbsVal)
		  {
		    compare.AddPair(queryData.flowValues[i], flowValues2);
		  }
		  else if(ushortq && isfinite(goldData.flowValues[i]) && fabs(goldData.flowValues[i]) < maxAbsVal)
		  {
		    compare.AddPair(flowValues2, goldData.flowValues[i]);
		  }
        }
	  }
	  else
	  {
        if (isfinite(queryData.flowValues[i]) && isfinite(goldData.flowValues[i]) && 
	    (fabs(queryData.flowValues[i]) < maxAbsVal && fabs(goldData.flowValues[i]) < maxAbsVal)) {
	      compare.AddPair(queryData.flowValues[i], goldData.flowValues[i]);
		}
	  }
    }
  }

  if(copies)
  {
    delete [] copies;
    copies = NULL;
  }

  const SampleStats<double> ssX = compare.GetXStats();
  const SampleStats<double> ssY = compare.GetYStats();
  cout << "query values: "  << ssX.GetMean() << " +/- "  << ssX.GetSD() << endl;
  cout << "gold values: "  << ssY.GetMean() << " +/- "  << ssY.GetSD() << endl;
  return compare;
}

int main(int argc, const char *argv[]) {

  OptArgs opts;
  opts.ParseCmdLine(argc, argv);
  string queryFile, goldFile;
  double epsilon;
  bool help = false;
  bool version = false;
  int allowedWrong = 0;
  double maxAbsVal = 0;
  double minCorrelation = 1;
  opts.GetOption(queryFile, "", 'q', "query-wells");
  opts.GetOption(goldFile, "", 'g', "gold-wells");
  opts.GetOption(epsilon, "0.0", 'e', "epsilon");
  opts.GetOption(allowedWrong, "0", 'm', "max-mismatch");
  opts.GetOption(minCorrelation, "1", 'c', "min-cor");
  opts.GetOption(maxAbsVal, "1e3", '-', "max-val");
  opts.GetOption(help, "false", 'h', "help");
  opts.GetOption(version, "false", 'v', "version");
  opts.CheckNoLeftovers();
  
  if (version) {
  	fprintf (stdout, "%s", IonVersion::GetFullVersion("RawWellsEquivalent").c_str());
  	exit(0);
  }
  
  if (queryFile.empty() || goldFile.empty() || help) {
    cout << "RawWellsEquivalent - Check to see how similar two wells files are to each other" << endl 
	 << "options: " << endl
	 << "   -g,--gold-wells    trusted wells to compare against." << endl
	 << "   -q,--query-wells   new wells to check." << endl
	 << "   -e,--epsilon       maximum allowed difference to be considered equivalent." << endl 
	 << "   -m,--max-mixmatch  maximum number of non-equivalent entries to allow." << endl
	 << "   -c,--min-cor       minimum correlation allowed to be considered equivalent." << endl
	 << "      --max-val       maximum absolute value considered (avoid extreme values)." << endl
	 << "   -h,--help          this message." << endl
	 << "" << endl 
         << "usage: " << endl
	 << "   RawWellsEquivalent -e 10 --query-wells query.wells --gold-wells gold.wells " << endl;
    exit(1);
  }

  NumericalComparison<double> compare = CompareWells(queryFile, goldFile, epsilon, maxAbsVal);
  cout << compare.GetCount() << " total values. " << endl
       << compare.GetNumSame() << " (" << (100.0 * compare.GetNumSame())/compare.GetCount() <<  "%) are equivalent. " << endl
       << compare.GetNumDiff() << " (" << (100.0 * compare.GetNumDiff())/compare.GetCount() <<  "%) are not equivalent. " << endl 
       << "Correlation of: " << compare.GetCorrelation() << endl;

  if((compare.GetCount() - allowedWrong) > compare.GetNumSame() || 
     (compare.GetCorrelation() < minCorrelation && compare.GetCount() != compare.GetNumSame())) {
     cout << "Wells files not equivalent for allowed mismatch: " << allowedWrong 
     << " minimum correlation: " << minCorrelation << endl;
     return 1;
  }
  cout << "Wells files equivalent for allowed mismatch: " << allowedWrong 
       << " minimum correlation: " << minCorrelation << endl;
  return 0;
}
