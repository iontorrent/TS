/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <iostream>
#include <sstream>

#ifdef __linux__
#include <sys/vfs.h>
#endif
#ifdef __APPLE__
#include <sys/uio.h>
#include <sys/mount.h>
#endif
#include <errno.h>
#include <assert.h>
#include "ByteSwapUtils.h"
#include "datahdr.h"
#include "LinuxCompat.h"
// #include "Raw2Wells.h"
#include "Image.h"
#include "crop/Acq.h"
#include "IonVersion.h"
#include "Utils.h"
#include "crop/MergeAcq.h"


using namespace std;

string joinPath(const char *expPath, const char *destName)
{
    string fname = destName;
    string inFile = expPath;
    inFile += "/" + fname;
    return inFile;
}


double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double t_sec = (double) tv.tv_sec + ((double) tv.tv_usec/1000000);
  return t_sec;
}


int copy_file(const char *src, const char *dst, const char *filename, bool verbose=false)
{
    string path1 = src;
    string path2 = dst;
    string fname = filename;
    path1 += "/" + fname;
    path2 += "/" + fname;
    char cmd[1024];
    if (verbose)
        sprintf (cmd, "cp -fv %s %s", path1.c_str(), path2.c_str());
    else
        sprintf (cmd, "cp -f %s %s", path1.c_str(), path2.c_str());
    int rc = system (cmd);
    return rc;
}


int copy_explog(const char *expPath, const char *destPath)
{
    char cmd[1024];
    sprintf (cmd, "cp -fv %s/explog*.txt %s/", expPath, destPath);
    int rc = system (cmd);
    return rc;
}


int copy_all_txt(const char *expPath, const char *destPath, bool verbose=false)
{
    char cmd[1024];
    if (verbose)
        sprintf (cmd, "cp -fv %s/*.txt %s/", expPath, destPath);
    else
        sprintf (cmd, "cp -f %s/*.txt %s/", expPath, destPath);
    int rc = system (cmd);
    return rc;
}


void copy_misc_files(const char *expPath, const char *destPath, bool verbose=false)
{
    const char* args[] = {"lsrowimage.dat","gainimage.dat","reimage.dat","rmsimage.dat"};
    vector<string> filesToMove(args, args + sizeof(args)/sizeof(args[0])); //init vector
    for (size_t n=0; n<filesToMove.size(); n++)
        copy_file(expPath,destPath,filesToMove[n].c_str());

    copy_all_txt(expPath,destPath,verbose);   // Copy all .txt files
}


static int do_mkdir(const char *path, mode_t mode)
{
    struct stat st;
    int status = 0;

    if (stat(path, &st) != 0)
    {
        /* Directory does not exist. EEXIST for race condition */
        if (mkdir(path, mode) != 0 && errno != EEXIST)
            status = -1;
    }
    else if (!S_ISDIR(st.st_mode))
    {
        errno = ENOTDIR;
        status = -1;
    }

    return(status);
}


int make_dir(const char *path, bool verbose=false, mode_t mode=0777)
{
    if (verbose)
        cout << "make_dir... " << path << endl << flush;
    char *copypath = strdup(path);
    char *pp = copypath;
    char *sp;
    int status = 0;
    // ensure all directories in path exist
    while (status == 0 && (sp = strchr(pp, '/')) != 0)
    {
        if (sp != pp)
        {
            /* Neither root nor double slash in path */
            *sp = '\0';
            status = do_mkdir(copypath, mode);
            *sp = '/';
        }
        pp = sp + 1;
    }
    if (status == 0)
        status = do_mkdir(path, mode);
    free(copypath);
    if (verbose)
        cout << "make_dir... " << path << " return status=" << status << endl << flush;
    return (status);
    /*
    // only 1 level, need to call this recursively from the top down
    umask (0);  // make permissive permissions so its easy to delete.
    if (mkdir (destPath, 0777)) {
        if (errno == EEXIST) {
            //already exists? well okay...
        }
        else {
            perror (destPath);
            exit (1);
        }
    }
    */
}


string make_subdir_name(int kernx, int kerny, int cropx, int cropy)
{
	stringstream ss;
	ss << kernx << "x" << cropx;
	if (kernx!=kerny || cropx!=cropy) 
		ss << "_" << kerny << "x" << cropy;
	return (ss.str());
}


string make_image_filename(const char *destPath, int k, int file_type)
{
    char destName[MAX_PATH_LENGTH];
    if (file_type==0)
        sprintf(destName, "%s/beadfind_pre_%04d.dat", destPath, k);
    else if (file_type==1)
        sprintf(destName, "%s/prerun_%04d.dat", destPath, k);
    else if (file_type==2)
        sprintf(destName, "%s/acq_%04d.dat", destPath, k);
    else {
        cerr << "file_type error: " << file_type<< endl<< flush;
        exit(1);
    }
    string filename = destName;
    return filename;
}

int is_multipleOf(int n, int base)
{
    return ((n%base)==0 ? true:false);
}

struct crop_region {
  int region_origin_x;
  int region_origin_y;
  int region_len_x;
  int region_len_y;
  char region_name[24];
};


struct foobar {
  struct crop_region *CropRegion;
  const RawImage *raw;
  char *destName;
  char *destPath;
  Acq *saver;
  Image *loader;
  int doAscii;
  int vfc;
  int i;
};


void create_thumbnail(int cropx, int cropy, int kernx, int kerny, int region_len_x, int region_len_y, int marginx, int marginy, int thumbnail_len_x, int thumbnail_len_y, int file_type, char *expPath, char *destPath, int flowstart, int flowlimit, int ignoreChecksumErrors, int dont_retry)
{
    if (flowstart>0 && file_type<2) // do nothing for prerun files if flowstart>0
        return;
    string name, destName;
    bool allocate = true; // this doesn't matter, not used in the functions
    int fileNum = 0;
    struct stat buffer;

    //initialize image object to load full .dat files
    Acq acq;
    Image *origImage = new Image;
    origImage->SetImgLoadImmediate(false);
    origImage->SetIgnoreChecksumErrors(ignoreChecksumErrors);
    if (dont_retry)
        origImage->SetTimeout(1,1); // if requested...do not bother waiting for the files to show up
    else
        origImage->SetTimeout(5,300); // wait 300 sec at 5 sec intervals


    while ( true )
    {
        if (flowstart>fileNum)
            continue;
        if (flowlimit>0 && fileNum>=flowlimit)
            break;
        destName = make_image_filename(expPath,fileNum,file_type);
        name = destName.c_str();
        if (stat (name.c_str(), &buffer)==0)
        {
            string finalDestName = make_image_filename(destPath,fileNum,file_type);

            //load full image
            origImage->LoadRaw_noWait_noSem(name.c_str(), 0, allocate, false);
            cout << "File load time: " << origImage->FileLoadTime << endl;

            //set data
            acq.SetData(origImage);
//            std::cout << "Set data" << endl;

            //write file
            acq.WriteThumbnailVFC(finalDestName.c_str(), cropx, cropy, kernx, kerny, region_len_x, region_len_y, marginx, marginy, thumbnail_len_x, thumbnail_len_y,false);
            cout << "Successfully created final thumbnail: " << finalDestName << endl;

            fileNum++;
        }
        else
            break;
    }
}

void usage ( int cropx, int cropy, int kernx, int kerny )
{
  fprintf ( stdout, "Thumbnail - Utility to chunk a raw data set into x*y cropped regions, then merge (x*y) cropped regions into a bigger one\n" );
  fprintf ( stdout, "options:\n" );
  fprintf ( stdout, "   -a\tOutput flat files; ascii text\n" );
  fprintf ( stdout, "   -b\tUse alternate sampling rate\n" );
  fprintf ( stdout, "   -x\tNumber of blocks along x axis. Default is %d\n",cropx );
  fprintf ( stdout, "   -y\tNumber of blocks along y axis. Default is %d\n",cropy );
  fprintf ( stdout, "   -X\tkernel X size (multiple of 8). Default is %d\n",kernx );
  fprintf ( stdout, "   -Y\tkernel Y size (multiple of 8). Default is %d\n",kerny );
  fprintf ( stdout, "   -t\tchip type 314, 316, 318, or 900 (Proton thumbnail). Default is 900.\n" );
  fprintf ( stdout, "   -i\tInput directory containing raw data\n" );
  fprintf ( stdout, "   -o\tOutput directory.\n" );
  fprintf ( stdout, "   -f\tConverts only the one file named as an argument\n" );
  fprintf ( stdout, "   -F\tFlowLimit\n" );
  fprintf ( stdout, "   -h\tPrints this message and exits.\n" );
  fprintf ( stdout, "   -v\tPrints version information and exits.\n" );
  fprintf ( stdout, "   -c\tOutput a variable rate frame compressed data set.  Default to whole chip\n" );
  fprintf ( stdout, "   -n\tOutput a non-variable rate frame compressed data set.\n" );
  fprintf (stdout, "   -z\tTells the image loader not to wait for a non-existent file\n");
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "usage:\n" );
  fprintf ( stdout, "   Thumbnail -i /results/analysis/PGM/testRun1 -t [314|316|318] \n" );
  fprintf ( stdout, "\n" );
  exit ( 1 );
}


int main ( int argc, char *argv[] )
{
  int cropx = 9, cropy = 9; //number of regions - x direction and y direction
  int kernx = 64, kerny = 64; //size of regions - x size and y size
  int region_len_x = 0;
  int region_len_y = 0;
  char *expPath  = const_cast<char*> ( "." );
  char *destPath = const_cast<char*> ( "." );
  int chipType = 0;
  int flowstart = 0;
  int flowlimit = 0;
  //int alternate_sampling=0;
  int doAscii = 0;
  int vfc = 1;
  int dont_retry = 1;
  int ignoreChecksumErrors = 1;
  if ( argc<=2 ) {
    usage ( cropx, cropy, kernx, kerny );
  }
  int argcc = 1;
  while ( argcc < argc ) {
    switch ( argv[argcc][1] ) {
    case 'a':
      doAscii = 1;
      break;

    case 'x':
      argcc++;
      cropx = atoi ( argv[argcc] );
      break;

    case 'y':
      argcc++;
      cropy = atoi ( argv[argcc] );
      break;

    case 'X':
      argcc++;
      kernx = atoi ( argv[argcc] );
      break;

    case 'Y':
      argcc++;
      kerny = atoi ( argv[argcc] );
      break;

    case 'F':
      argcc++;
      flowlimit = atoi ( argv[argcc] );
      break;

    case 'i':
      argcc++;
      expPath = argv[argcc];
      break;

    case 'o':
      argcc++;
      destPath = argv[argcc];
      break;

    case 't':
      argcc++;
      chipType = atoi ( argv[argcc] );
      break;

    case 'z':
      dont_retry = 1;
      break;

    case 'c':
      vfc=1;
      cropx=0;
      cropy=0;
      break;

    case 'n':
      vfc=0;
      break;

    case 'b':
      //alternate_sampling=1;
      break;

    case 'v':
      fprintf ( stdout, "%s", IonVersion::GetFullVersion ( "Thumbnail" ).c_str() );
      exit ( 0 );
      break;

    case 'h':
      usage ( cropx, cropy, kernx, kerny );
      break;

    default:
      argcc++;
      fprintf ( stdout, "\n" );

    }
    argcc++;
  }

  if (chipType==0)
  {
      cout << "Unknown chipType, setting to default 900 (thumbnail)" << endl << flush;
      chipType = 900;
  }

// Create results folder
  string dst = destPath;
  dst += "/" + make_subdir_name(kernx,kerny,cropx,cropy);
  destPath = const_cast<char*> (dst.c_str());
  make_dir(destPath);

  // Initialize array of crop regions
//  int numRegions = cropx * cropy;
//  pthread_t threads[numRegions];

  // Calculate regions based on chip type and number of blocks requested per axis
  // cropx is number of regions to carve along x axis
  // cropy is number of regions to carve along the y axis
  if ( chipType == 314 ) {
    //[1280,1152]
    // x axis length is 1280 pixels
    // y axis length is 1152 pixels
    region_len_x = 1280 / cropx;
    region_len_y = 1152 / cropy;
  } else if ( chipType == 316 ) {
    //[2736,2640]
    region_len_x = 2736 / cropx;
    region_len_y = 2640 / cropy;
  } else if ( chipType == 318 ) {
    //[3392,3792]
    region_len_x = 3392 / cropx;
    region_len_y = 3792 / cropy;
  } else if ( chipType == 900 ) {
      //[3392,3792], [1200,800] for thumbnail
      region_len_x = 1200 / cropx;
      region_len_y = 800 / cropy;
  } else {
    fprintf ( stderr, "Unknown chip: %d\n", chipType );
    exit ( 1 );
  }

  if (!is_multipleOf(kernx,8))
  {
      int kernx_new = (kernx/8+1)*8;
      printf ( "kernx=%d is not a multiple of 8, changing to %d ", kernx, kernx_new);
      fflush ( stdout );
      kernx = kernx_new;
  }

  if (!is_multipleOf(kerny,8))
  {
      int kerny_new = (kerny/8+1)*8;
      printf ( "kerny=%d is not a multiple of 8, changing to %d ", kerny, kerny_new);
      fflush ( stdout );
      kerny = kerny_new;
  }

  assert(region_len_x >= kernx);
  assert(region_len_y >= kerny);
  int marginx = (region_len_x - kernx) / 2;
  int marginy = (region_len_y - kerny) / 2;

  int thumbnail_len_x = kernx*cropx;
  int thumbnail_len_y = kerny*cropy;

  if (flowstart==0)
  {
      copy_misc_files(expPath,destPath);   // Copy explog.txt file: all .txt files
      cout << "\n\n\n----------------------Copied all miscellaneous files.----------------------\n" << endl;
  }

  cout << "\n\n ----------------------Beadfind files:----------------------\n\n" << endl;
  create_thumbnail(cropx, cropy, kernx, kerny, region_len_x, region_len_y, marginx, marginy, thumbnail_len_x, thumbnail_len_y, 0, expPath, destPath, flowstart, flowlimit, ignoreChecksumErrors, dont_retry); //beadfind files
  cout << "\n\n ----------------------Prerun files:----------------------\n\n" << endl;
  create_thumbnail(cropx, cropy, kernx, kerny, region_len_x, region_len_y, marginx, marginy, thumbnail_len_x, thumbnail_len_y, 1, expPath, destPath, flowstart, flowlimit, ignoreChecksumErrors, dont_retry); //prerun files
  cout << "\n\n ----------------------Acq files:----------------------\n\n" << endl;
  create_thumbnail(cropx, cropy, kernx, kerny, region_len_x, region_len_y, marginx, marginy, thumbnail_len_x, thumbnail_len_y, 2, expPath, destPath, flowstart, flowlimit, ignoreChecksumErrors, dont_retry); //acq files

  return EXIT_SUCCESS;
}
