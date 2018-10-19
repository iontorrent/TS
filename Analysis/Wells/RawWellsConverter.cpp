/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <string>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>

#include "Utils.h"
#include "RawWells.h"
#include "OptArgs.h"
#include "IonVersion.h"

using namespace std;

int main(int argc, const char *argv[]) 
{
	OptArgs opts;
	opts.ParseCmdLine(argc, argv);
	string inFile, outFile;
	bool help = false;
	bool version = false;
	double lower = -5.0;
	double upper = 28.0;
	opts.GetOption(inFile, "", 'i', "input-file");
	opts.GetOption(outFile, "", 'o', "output-file");
	opts.GetOption(lower, "-5.0", '-', "wells-convert-low");
	opts.GetOption(upper, "28.0", '-', "wells-convert-high");
	opts.GetOption(help, "false", 'h', "help");
	opts.GetOption(version, "false", 'v', "version");
	opts.CheckNoLeftovers();
  
	if (version) 
	{
		fprintf (stdout, "%s", IonVersion::GetFullVersion("RawWellsConverter").c_str());
		exit(0);
	}

	if (inFile.empty() || help)
	{
		cout << "RawWellsConverter - Convert unsigned short type wells file to float type wells file, or vice versa." << endl 
			 << "options: " << endl
			 << "   -i,--input-file    input wells file." << endl
			 << "   -o,--output-file   output wells file." << endl
			 << "     ,--wells-convert-low   lower bound for converting to unsigned short." << endl
			 << "     ,--wells-convert-high  upper bound for converting to unsigned short." << endl
			 << "   -h,--help          this message." << endl
			 << "" << endl 
			 << "usage: " << endl
			 << "   RawWellsConverter -i input_path/1.wells -o output_path/1.wells " << endl;
		exit(1);
	}

	struct stat sb;
	if(stat(inFile.c_str(), &sb) != 0)
	{
		cerr << "RawWellsConverter ERROR: " << inFile << " does not exist." << endl; 
		exit (1);
	}

	if (outFile.empty())
	{
		outFile = inFile;
		outFile += ".converted";
	}

	string cmd("cp ");
	cmd += inFile;
	cmd += " ";
	cmd += outFile;
	int ret0 = system(cmd.c_str());

	hid_t root = H5Fopen(outFile.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	if(root < 0)				
	{				
		cerr << "RawWellsConverter ERROR: Fail to open " << outFile << endl;
		exit(1);
	}	

	H5G_info_t group_info;
	group_info.nlinks = 0;
	if(H5Gget_info(root, &group_info) < 0)
	{
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: Fail H5Gget_info." << endl;
		exit(1);
	}

	char name[10];
	string sName;
	bool bWells = false;
	bool bCopies = false;
	for(unsigned int i = 0; i < group_info.nlinks; ++i)
	{
		int size = H5Gget_objname_by_idx(root, i, NULL, 0);
		if(H5Gget_objname_by_idx(root, i, name, size + 1) < 0)
		{
			H5Fclose(root);
			cerr << "RawWellsConverter ERROR: Fail H5Gget_objname_by_idx." << endl;
			exit(1);
		}
		else
		{
			sName = name;
			if(sName == "wells")
			{
				bWells = true;
			}
			if(sName == "wells_copies")
			{
				bCopies = true;
			}
		}
	}

	if(!bWells)
	{
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: There is no dataset wells." << endl;
		exit(1);
	}

	hid_t dsWells = H5Dopen2(root, "wells", H5P_DEFAULT);
	if(dsWells < 0)
	{
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: Fail H5Dopen2 wells." << endl;
		exit(1);
	}
	  
	bool saveAsUShort = false;
	if(H5Aexists(dsWells, "convert_low") > 0)
	{
        hid_t attrLower = H5Aopen(dsWells, "convert_low", H5P_DEFAULT );
		H5Aread(attrLower, H5T_NATIVE_FLOAT, &lower); 
		saveAsUShort = true;
		H5Aclose(attrLower);
	}
	if(H5Aexists(dsWells, "convert_high") > 0)
	{
        hid_t attrUpper = H5Aopen(dsWells, "convert_high", H5P_DEFAULT);
		H5Aread(attrUpper, H5T_NATIVE_FLOAT, &upper); 
		saveAsUShort = true;
		H5Aclose(attrUpper);
	}

	hid_t dataSpace = H5Dget_space(dsWells);
	if(dataSpace < 0)
	{
		H5Dclose(dsWells);
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: Fail H5Dget_space wells." << endl;
		exit(1);
	}

	hssize_t dsSize = H5Sget_simple_extent_npoints(dataSpace);		
	if(dsSize < 1)
	{
		H5Sclose(dataSpace);
		H5Dclose(dsWells);
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: Wrong size of dataset wells - " << dsSize << endl;
		exit(1);
	}

	int nRows = 0;
	int nCols = 0;
	int nFlows = 0;

	int rank = H5Sget_simple_extent_ndims(dataSpace);
    if(rank != 3)
	{
		bCopies = false;
	}
	else
	{
		hsize_t dims_out[3];
		int status_n = H5Sget_simple_extent_dims(dataSpace, dims_out, NULL);
		if(status_n < 0)
		{
			bCopies = false;
		}
		else
		{
			nRows = dims_out[0];
			nCols = dims_out[1];
			nFlows = dims_out[2];
		}
	}

	float* fPtr = new float[dsSize];
	unsigned short* usPtr = new unsigned short[dsSize];
	if(fPtr == NULL || usPtr == NULL)
	{
		H5Sclose(dataSpace);
		H5Dclose(dsWells);
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: Fail to allocate fPtr or usPtr." << endl;
		exit(1);
	}

	hid_t dcpl = H5Dget_create_plist(dsWells);
	if(dcpl < 0)
	{
		H5Sclose(dataSpace);
		H5Dclose(dsWells);
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: Fail H5Dget_create_plist." << endl;
		exit(1);
	}
	hid_t dapl = H5Dget_access_plist(dsWells);
	if(dapl < 0)
	{
		H5Pclose(dcpl);
		H5Sclose(dataSpace);
		H5Dclose(dsWells);
		H5Fclose(root);
		cerr << "RawWellsConverter ERROR: Fail H5Dget_access_plist." << endl;
		exit(1);
	}

	if(saveAsUShort)
	{
		cout << "RawWellsConverter: converting unsigned short wells file - " << inFile << " to float wells file - " << outFile << " with boundary (" << lower << "," << upper << ")" << endl;
	
		herr_t ret = H5Dread(dsWells, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, usPtr);
		H5Dclose(dsWells);
		if(ret < 0)
		{
			delete [] fPtr;
			delete [] usPtr;
			H5Sclose(dataSpace);
			H5Pclose(dcpl);
			H5Pclose(dapl);
			H5Fclose(root);
			cerr << "RawWellsConverter ERROR: Fail to read dataset wells." << endl;
			exit(1);
		}

		float factor = 65535.0 / (upper - lower);
		float* fPtr2 = fPtr;
		unsigned short* usPtr2 = usPtr;

		for(unsigned int i = 0; i < dsSize; ++i, ++fPtr2, ++usPtr2)
		{
			(*fPtr2) = (float)(*usPtr2) / factor + lower;
		}

		delete [] usPtr;

		if(bCopies)
		{
			vector<float> copies(nRows * nCols, 1.0);
			hid_t dsCopies = H5Dopen2(root, "wells_copies", H5P_DEFAULT);
			if(dsCopies < 0)
			{
				cerr << "RawWellsConverter WARNING: 1.wells files does not have wells_copies." << endl;
			}
			else
			{
				hid_t dataSpace2 = H5Dget_space(dsCopies);
				if(dataSpace2 < 0)
				{
					H5Dclose(dsCopies);
					cerr << "RawWellsConverter WARNING: fail to H5Dget_space for dataset wells_copies." << endl;          
				}
				else
				{
					hssize_t dsSize2 = H5Sget_simple_extent_npoints(dataSpace2);
					H5Sclose(dataSpace2);
					if(dsSize2 != (hssize_t)(nRows * nCols))
					{
						H5Dclose(dsCopies);
						cerr << "RawWellsConverter WARNING: dataset wells_copies size is " << dsSize2 << ", it is different from nRows * nCols = " << nRows * nCols << endl;          
					}
					else
					{
						herr_t ret = H5Dread(dsCopies, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &copies[0]);
						H5Dclose(dsCopies);
						if(ret < 0)
						{
							copies.resize(nRows * nCols, 1.0);
							cerr << "RawWellsConverter WARNING: failto load dataset wells_copies." << endl;          
						}
					}
				}
			}
			
			uint64_t fptrCount = 0;
			uint64_t copyCount = 0;
			for(int row = 0; row < nRows; ++row)
			{
				for(int col = 0; col < nCols; ++col)
				{
					for(int flow = 0; flow < nFlows; ++flow)
					{
						if(copies[copyCount] > 0)
						{
							fPtr[fptrCount] *= copies[copyCount];
						}
						else
						{
							fPtr[fptrCount] = -1.0;
						}
						
						++fptrCount;
					}

					++copyCount;
				}
			}
		}

	    H5Ldelete(root, "wells", H5P_DEFAULT);

		hid_t dsWells2 = H5Dcreate2 (root, "wells", H5T_NATIVE_FLOAT, dataSpace, H5P_DEFAULT, dcpl, dapl);
		if(dsWells2 < 0)
		{
			delete [] fPtr;
			H5Sclose(dataSpace);
			H5Pclose(dcpl);
			H5Pclose(dapl);
			H5Fclose(root);
			cerr << "RawWellsConverter ERROR: Fail to create dataset wells." << endl;
			exit(1);
		}

		ret = H5Dwrite(dsWells2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fPtr);
		delete [] fPtr;
		H5Dclose(dsWells2);
		H5Sclose(dataSpace);
		H5Pclose(dcpl);
		H5Pclose(dapl);
		H5Fclose(root);
		if(ret < 0)
		{
			cerr << "RawWellsConverter ERROR: Fail to write dataset wells." << endl;
			exit(1);
		}
	}
	else
	{
		cout << "RawWellsConverter: converting float wells file - " << inFile << " to unsigned short wells file - " << outFile << " with boundary (" << lower << "," << upper << ")" << endl;
	
		herr_t ret = H5Dread(dsWells, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fPtr);
		H5Dclose(dsWells);
		if(ret < 0)
		{
			delete [] fPtr;
			delete [] usPtr;
			H5Sclose(dataSpace);
			H5Pclose(dcpl);
			H5Pclose(dapl);
			H5Fclose(root);
			cerr << "RawWellsConverter ERROR: Fail to read dataset wells." << endl;
			exit(1);
		}

		float factor = 65535.0 / (upper - lower);
		float* fPtr2 = fPtr;
		unsigned short* usPtr2 = usPtr;

		for(unsigned int i = 0; i < dsSize; ++i, ++fPtr2, ++usPtr2)
		{
			if(*fPtr2 < lower)
			{
				(*usPtr2) = 0;
			}
			else if(*fPtr2 > upper)
			{
				(*usPtr2) = 65535;
			}
			else
			{
				(*usPtr2) = (unsigned short)((*fPtr2 - lower) * factor);
			}
		}

		delete [] fPtr;

	    H5Ldelete(root, "wells", H5P_DEFAULT);

		hid_t dsWells2 = H5Dcreate2 (root, "wells", H5T_NATIVE_USHORT, dataSpace, H5P_DEFAULT, dcpl, dapl);
		if(dsWells2 < 0)
		{
			delete [] usPtr;
			H5Sclose(dataSpace);
			H5Pclose(dcpl);
			H5Pclose(dapl);
			H5Fclose(root);
			cerr << "RawWellsConverter ERROR: Fail to create dataset wells." << endl;
			exit(1);
		}

		ret = H5Dwrite(dsWells2, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, usPtr);
		delete [] usPtr;
		if(dsWells2 < 0)
		{
			H5Dclose(dsWells2);
			H5Sclose(dataSpace);
			H5Pclose(dcpl);
			H5Pclose(dapl);
			H5Fclose(root);
			cerr << "RawWellsConverter ERROR: Fail to write dataset wells." << endl;
			exit(1);
		}

		float lower2 = (float)lower;
		float upper2 = (float)upper;
		hsize_t dimsa[1];
		dimsa[0] = 1;
		hid_t dataspacea = H5Screate_simple(1, dimsa, NULL);
		hid_t attrLower = H5Acreate(dsWells2, "convert_low", H5T_NATIVE_FLOAT, dataspacea, H5P_DEFAULT, H5P_DEFAULT );
		H5Awrite(attrLower, H5T_NATIVE_FLOAT, &lower2);
		H5Aclose(attrLower);
		hid_t attrUpper = H5Acreate(dsWells2, "convert_high", H5T_NATIVE_FLOAT, dataspacea, H5P_DEFAULT, H5P_DEFAULT );
		H5Awrite(attrUpper, H5T_NATIVE_FLOAT, &upper2);
		H5Aclose(attrUpper);
		H5Sclose(dataspacea);

		H5Dclose(dsWells2);
		H5Sclose(dataSpace);
		H5Pclose(dcpl);
		H5Pclose(dapl);
		H5Fclose(root);
	}

	return 0;
}
